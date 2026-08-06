import json
import logging
import os
import re
import sys
import time
from base64 import b64decode
from datetime import timedelta

from django.conf import settings
from django.core.files.base import ContentFile
from django.utils.translation import gettext as _
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from google import genai
from google.genai.types import Content, Part
from huey import crontab
from huey.contrib.djhuey import db_periodic_task, db_task
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from openai import OpenAI

from agent.models import GenVideo, TipkoRequest, VideoSegment
from agent.task_utils.tipko_source import Api
from agent.task_utils.video_rendering import FFmpegTimeoutError, FinalVideoRenderer
from agent.utils import ensure_google_api_key, get_temporary_file_path

# Configure logging for Huey tasks
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

tipko_api = Api(
    endpoint=settings.TIPKO_API_ENDPOINT,
    username=settings.TIPKO_API_USERNAME,
    password=settings.TIPKO_API_PASSWORD,
)


def _format_srt_timestamp(seconds: float) -> str:
    total_milliseconds = round(seconds * 1000)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def generate_srt_from_elevenlabs_alignment(
    alignment, max_words_per_screen: int | None = None
) -> str:
    """Create SRT subtitle content from ElevenLabs character-level timings."""
    if isinstance(alignment, dict):
        characters = alignment.get("characters")
        start_times = alignment.get("character_start_times_seconds")
        end_times = alignment.get("character_end_times_seconds")
    else:
        characters = alignment.characters
        start_times = alignment.character_start_times_seconds
        end_times = alignment.character_end_times_seconds
    if not (characters and start_times and end_times):
        raise ValueError("ElevenLabs response did not contain alignment data")
    if not (len(characters) == len(start_times) == len(end_times)):
        raise ValueError("ElevenLabs alignment data has inconsistent lengths")

    max_words = max_words_per_screen or 8
    words = []
    word_characters = []
    word_start = None
    word_end = None

    def append_word() -> None:
        nonlocal word_characters, word_start, word_end
        if word_characters and word_start is not None and word_end is not None:
            words.append(("".join(word_characters), word_start, word_end))
        word_characters = []
        word_start = None
        word_end = None

    for character, start_time, end_time in zip(characters, start_times, end_times):
        if character.isspace():
            append_word()
            continue
        if start_time is None or end_time is None:
            continue
        if word_start is None:
            word_start = start_time
        word_characters.append(character)
        word_end = end_time
    append_word()

    if not words:
        raise ValueError("ElevenLabs alignment did not contain timed words")

    subtitles = []
    subtitle_words = []
    subtitle_start = None
    subtitle_end = None

    def append_subtitle() -> None:
        nonlocal subtitle_words, subtitle_start, subtitle_end
        if subtitle_words and subtitle_start is not None and subtitle_end is not None:
            subtitles.append(
                (
                    subtitle_start,
                    subtitle_end,
                    " ".join(subtitle_words),
                )
            )
        subtitle_words = []
        subtitle_start = None
        subtitle_end = None

    for word, start_time, end_time in words:
        if subtitle_start is None:
            subtitle_start = start_time
        subtitle_words.append(word)
        subtitle_end = end_time
        if len(subtitle_words) >= max_words or word.endswith((".", "!", "?")):
            append_subtitle()
    append_subtitle()

    return "\n\n".join(
        f"{index}\n{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}\n{text}"
        for index, (start, end, text) in enumerate(subtitles, start=1)
    )


def save_elevenlabs_srt_file(video: GenVideo, alignment) -> None:
    """Save SRT subtitles derived from the alignment returned by ElevenLabs TTS."""
    video.status = GenVideo.Statuses.GENERATING_SUBTITLES
    video.save(update_fields=["status", "updated_at"])

    srt_content = generate_srt_from_elevenlabs_alignment(
        alignment,
        video.subtitle_max_words_per_screen,
    )
    is_valid, validation_message = validate_srt_content(srt_content, video)
    if not is_valid:
        raise ValueError(f"{_('Neveljavna SRT vsebina')}: {validation_message}")

    filename = f"srt_{video.id}.srt"
    video.srt_content = srt_content
    video.srt_file.save(filename, ContentFile(srt_content), save=False)
    video.status = GenVideo.Statuses.SUBTITLES_READY
    video.save()
    logger.info(
        "SRT subtitles generated from ElevenLabs alignment for video %s: %s",
        video.id,
        validation_message,
    )


@db_task()
def regenerate_elevenlabs_srt_file(video: GenVideo) -> None:
    """Regenerate only SRT subtitles from a saved ElevenLabs alignment."""
    try:
        if not video.elevenlabs_alignment:
            raise ValueError("Video has no saved ElevenLabs alignment data")
        save_elevenlabs_srt_file(video, video.elevenlabs_alignment)
        render_final_video(video)
    except Exception as e:
        logger.error(
            "Error regenerating ElevenLabs SRT file for video %s: %s",
            video.id,
            e,
        )
        video.status = GenVideo.Statuses.FAILED
        video.error_type = GenVideo.ErrorTypes.SRT_GENERATION
        video.error_details = _(
            "Napaka pri ustvarjanju podnapisov (ElevenLabs): %(error)s"
        ) % {"error": str(e)}
        video.save()
        raise


def validate_srt_content(srt_content: str, video: GenVideo) -> tuple[bool, str]:
    """
    Validate SRT subtitle content before saving.

    Checks:
    - Content is not empty
    - Contains valid subtitle blocks (number, timecode, text)
    - Timecodes are in correct format
    - No missing or empty subtitle entries
    - Last subtitle end is near the end of the video duration

    Args:
        srt_content: The SRT file content string

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not srt_content or not srt_content.strip():
        return False, "SRT content is empty"

    def _parse_srt_timestamp(value: str) -> timedelta:
        """Parse SRT timestamp (HH:MM:SS,mmm or HH:MM:SS.mmm) into timedelta."""
        normalized = value.replace(",", ".")
        hours, minutes, seconds = normalized.split(":")
        return timedelta(
            hours=int(hours),
            minutes=int(minutes),
            seconds=float(seconds),
        )

    lines = srt_content.strip().split("\n")

    # Basic structure check - should have at least 3 lines (number, timecode, text)
    if len(lines) < 3:
        return False, "SRT content too short - missing subtitle blocks"

    # Track subtitle blocks
    subtitle_count = 0
    last_subtitle_end = None
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Check if line is a subtitle number
        if not line.isdigit():
            return False, f"Expected subtitle number at line {i+1}, got: {line}"

        subtitle_num = int(line)
        if subtitle_num != subtitle_count + 1:
            return (
                False,
                f"Subtitle numbering error: expected {subtitle_count + 1}, got {subtitle_num}",
            )

        i += 1
        if i >= len(lines):
            return False, f"Subtitle {subtitle_num}: Missing timecode"

        # Check timecode format (00:00:00,000 --> 00:00:00,000)
        timecode_line = lines[i].strip()
        if "-->" not in timecode_line:
            return (
                False,
                f"Subtitle {subtitle_num}: Invalid timecode format - missing '-->' separator",
            )

        # Validate timecode structure
        parts = timecode_line.split("-->")
        if len(parts) != 2:
            return False, f"Subtitle {subtitle_num}: Invalid timecode format"

        start_time = parts[0].strip()
        end_time = parts[1].strip()

        # Check timecode pattern (HH:MM:SS,mmm)
        timecode_pattern = r"^\d{2}:\d{2}:\d{2}[,\.]\d{1,3}$"
        if not re.match(timecode_pattern, start_time):
            return (
                False,
                f"Subtitle {subtitle_num}: Invalid start time format: {start_time}",
            )
        if not re.match(timecode_pattern, end_time):
            return (
                False,
                f"Subtitle {subtitle_num}: Invalid end time format: {end_time}",
            )

        try:
            start_td = _parse_srt_timestamp(start_time)
            end_td = _parse_srt_timestamp(end_time)
        except ValueError:
            return False, f"Subtitle {subtitle_num}: Could not parse timecode"

        if end_td <= start_td:
            return (
                False,
                f"Subtitle {subtitle_num}: End time must be greater than start time",
            )

        last_subtitle_end = end_td

        i += 1
        if i >= len(lines):
            return False, f"Subtitle {subtitle_num}: Missing text content"

        # Check for subtitle text (at least one non-empty line)
        has_text = False
        while i < len(lines) and lines[i].strip():
            if lines[i].strip():
                has_text = True
            i += 1

        if not has_text:
            return False, f"Subtitle {subtitle_num}: Empty text content"

        subtitle_count += 1

    # Final check - should have at least one subtitle
    if subtitle_count == 0:
        return False, "No valid subtitles found in SRT content"

    video_duration_seconds = getattr(video, "duration", None) or getattr(
        video, "voice_duration", None
    )
    if video_duration_seconds:
        allowed_early_end_seconds = float(
            getattr(settings, "SRT_LAST_SUBTITLE_ALLOWED_EARLY_END_SECONDS", 2)
        )
        min_last_end = timedelta(
            seconds=max(float(video_duration_seconds) - allowed_early_end_seconds, 0)
        )
        if last_subtitle_end and last_subtitle_end <= min_last_end:
            return (
                False,
                (
                    "Last subtitle ends too early: "
                    f"{last_subtitle_end.total_seconds():.3f}s <= "
                    f"{min_last_end.total_seconds():.3f}s "
                    "(video_duration - allowed_early_end_seconds)"
                ),
            )
        elif last_subtitle_end:
            if last_subtitle_end > timedelta(seconds=video_duration_seconds):
                return (
                    False,
                    (
                        "Last subtitle ends after video duration: "
                        f"{last_subtitle_end.total_seconds():.3f}s > "
                        f"{video_duration_seconds:.3f}s"
                    ),
                )

    logger.info(f"✓ SRT validation passed: {subtitle_count} subtitles validated")
    return True, f"Valid SRT with {subtitle_count} subtitles"


@db_task()
def generate_voice_file_eleven_labs(video: int) -> None:
    """
    Generate voice file from scenario using ElevenLabs SDK.

    Args:
        video.id: ID of the GenVideo instance
    """
    try:
        logger.info(f"ELEVENLABS TTS")
        video.status = GenVideo.Statuses.GENERATING_VOICE
        video.save()

        video.segments.all().delete()  # Clear existing segments if any

        if not video.scenario:
            raise ValueError(f"Video {video.id} has no scenario to convert to speech")

        if not settings.ELEVENLABS_API_KEY:
            raise ValueError("ELEVENLABS_API_KEY is not configured in settings")

        # Check text length (ElevenLabs charges per character)
        text_length = len(video.scenario)
        logger.info(f"Content script length: {text_length} characters")

        # Warning for very long texts (>5000 chars may be expensive)
        if text_length > 5000:
            logger.warning(f"Long text ({text_length} chars) may consume many credits")

        # Backward compatible handling:
        # - old records stored ElevenLabs voice IDs in video.voice_model
        # - new records store ElevenLabs model IDs (e.g. eleven_v3)
        configured_default_voice = getattr(
            settings,
            "ELEVENLABS_DEFAULT_VOICE_ID",
            "21m00Tcm4TlvDq8ikWAM",
        )
        selected_value = (video.voice_model or "").strip()
        if selected_value.startswith("eleven_"):
            voice_id = configured_default_voice
            model_id = selected_value
        else:
            voice_id = selected_value or configured_default_voice
            model_id = "eleven_v3"

        logger.info(
            "Generating voice for video %s with voice %s and model %s",
            video.id,
            voice_id,
            model_id,
        )

        # Initialize ElevenLabs client
        client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)

        # Generate speech and character-level timing from the same TTS request.
        response = client.text_to_speech.convert_with_timestamps(
            voice_id=voice_id,
            text=video.scenario,
            model_id=model_id,
            language_code=(video.language or "sl").split("-")[0],
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
                speed=1.2,
            ),
        )

        audio_bytes = b64decode(response.audio_base_64)

        # Save the audio file to the model
        filename = f"voice_{video.id}.mp3"
        video.voice_file.save(filename, ContentFile(audio_bytes), save=False)

        try:
            with get_temporary_file_path(video.voice_file) as temp_audio_path:
                duration = get_audio_duration(temp_audio_path)
                if duration > settings.MAX_VOICE_DURATION_SECONDS:
                    raise ValueError(
                        f"Voice file duration {duration:.2f}s exceeds maximum allowed {settings.MAX_VOICE_DURATION_SECONDS}s"
                    )
                video.voice_duration = duration
                logger.info(f"Voice file duration: {duration:.2f} seconds")
        except Exception as e:
            logger.warning(f"Could not get audio duration: {e}")

        logger.info(
            f"✓ Voice file generated successfully for video {video.id} [ElevenLabs]"
        )

        alignment = response.normalized_alignment or response.alignment
        video.elevenlabs_alignment = {
            "characters": alignment.characters,
            "character_start_times_seconds": alignment.character_start_times_seconds,
            "character_end_times_seconds": alignment.character_end_times_seconds,
        }
        save_elevenlabs_srt_file(video, alignment)
        get_video_segments(video)

    except GenVideo.DoesNotExist:
        logger.error(f"Video with id {video.id} does not exist")
        raise
    except Exception as e:
        logger.error(f"✗ Error generating voice file for video {video.id}: {str(e)}")
        video = GenVideo.objects.get(id=video.id)
        video.status = GenVideo.Statuses.FAILED
        if video.voice_file:
            video.error_type = GenVideo.ErrorTypes.SRT_GENERATION
            video.error_details = _(
                "Napaka pri ustvarjanju podnapisov (ElevenLabs): %(error)s"
            ) % {"error": str(e)}
        else:
            video.error_type = GenVideo.ErrorTypes.VOICE_GENERATION
            video.error_details = _(
                "Napaka pri ustvarjanju zvočne datoteke (ElevenLabs): %(error)s"
            ) % {"error": str(e)}
        video.save()
        raise


@db_task()
def generate_voice_file_openai(video: int) -> None:
    """
    Generate voice file from scenario using OpenAI TTS API.
    Much cheaper than ElevenLabs - ~$0.015 per 1000 characters.

    Args:
        video.id: ID of the GenVideo instance
    """
    try:
        logger.info(f"OPENAI TTS")
        video.status = GenVideo.Statuses.GENERATING_VOICE
        video.save()

        if not video.scenario:
            raise ValueError(f"Video {video.id} has no scenario to convert to speech")

        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not configured in settings")

        # Check text length
        text_length = len(video.scenario)
        logger.info(f"Content script length: {text_length} characters")
        logger.info(f"Estimated cost: ${(text_length / 1000) * 0.015:.4f}")

        # Use voice_model directly (should be OpenAI voice ID)
        # OpenAI voices: alloy, echo, fable, onyx, nova, shimmer
        voice = video.voice_model or "alloy"

        # Validate voice is a valid OpenAI voice
        valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        if voice not in valid_voices:
            logger.warning(f"Invalid voice '{voice}', using default 'alloy'")
            voice = "alloy"

        logger.info(f"Generating voice for video {video.id} with OpenAI voice: {voice}")

        # Initialize OpenAI client
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        # Generate speech from scenario
        response = client.audio.speech.create(
            model="tts-1",  # or "tts-1-hd" for higher quality
            voice=voice,
            input=video.scenario,
        )

        # Save the audio file to the model
        filename = f"voice_{video.id}.mp3"
        video.voice_file.save(filename, ContentFile(response.content), save=False)

        # Update video status
        video.status = GenVideo.Statuses.VOICE_READY
        video.save()

        logger.info(
            f"✓ Voice file generated successfully for video {video.id} [OpenAI]"
        )

    except GenVideo.DoesNotExist:
        logger.error(f"Video with id {video.id} does not exist")
        raise
    except Exception as e:
        logger.error(f"✗ Error generating voice file for video {video.id}: {str(e)}")
        video = GenVideo.objects.get(id=video.id)
        video.status = GenVideo.Statuses.FAILED
        video.error_type = GenVideo.ErrorTypes.VOICE_GENERATION
        video.error_details = _(
            "Napaka pri ustvarjanju zvočne datoteke (OpenAI): %(error)s"
        ) % {"error": str(e)}
        video.save()
        raise


def get_audio_duration(audio_file_path):
    """
    Get duration of audio file in seconds using ffprobe.

    Args:
        audio_file_path: Path to audio file

    Returns:
        float: Duration in seconds
    """
    import json
    import subprocess

    cmd = [
        "ffprobe",
        "-v",
        "quiet",  # Suppress ffprobe output
        "-print_format",
        "json",  # Output in JSON format
        "-show_format",  # Show format information (includes duration)
        str(audio_file_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    duration = float(data["format"]["duration"])
    logger.info(f"Audio duration: {duration:.2f} seconds")
    return duration


@db_task()
def generate_voice_file_gemini(video: int) -> None:
    """
    Generate voice file from scenario using Google Gemini Audio Generation via LangChain.
    Part of Google AI services - uses same API key as Gemini.
    Free tier available with Gemini API!

    Args:
        video.id: ID of the GenVideo instance
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        logger.info(f"GEMINI TTS")

        video.status = GenVideo.Statuses.GENERATING_VOICE
        video.save()

        if not video.scenario:
            raise ValueError(f"Video {video.id} has no scenario to convert to speech")

        ensure_google_api_key()

        # Check text length
        text_length = len(video.scenario)
        logger.info(f"Content script length: {text_length} characters")

        # Map voice models to Gemini voice names
        gemini_voices = {
            "alloy": "Puck",  # Neutral
            "echo": "Charon",  # Male, authoritative
            "fable": "Kore",  # Warm, storytelling
            "onyx": "Fenrir",  # Deep male
            "nova": "Aoede",  # Female, energetic
            "shimmer": "Puck",  # Balanced
        }

        voice_name = gemini_voices.get(video.voice_model, "Puck")

        logger.info(
            f"Generating voice for video {video.id} with Gemini voice: {voice_name}"
        )

        # Initialize Gemini model with audio generation
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-tts",
            google_api_key=settings.GOOGLE_API_KEY,
        )

        response = model.invoke(
            f"Pripravi mi zvokovni posnetek za naslednji text, spodaj imaš še scenarij, ki ga upoštevaj:\n ------ \n {video.scenario}\n ------ \n {video.scenario}",
            generation_config={
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {"voice_name": voice_name}
                    }
                },
            },
        )

        logger.info(f"Response type: {type(response)}")
        logger.info(f"Response attributes: {dir(response)}")
        if hasattr(response, "additional_kwargs"):
            logger.info(f"Additional kwargs keys: {response.additional_kwargs.keys()}")
        if hasattr(response, "response_metadata"):
            logger.info(f"Response metadata keys: {response.response_metadata.keys()}")

        # Extract audio data from response
        audio_data = None
        if (
            hasattr(response, "additional_kwargs")
            and "audio" in response.additional_kwargs
        ):
            audio_data = response.additional_kwargs["audio"]
            logger.info(f"Found audio in additional_kwargs, type: {type(audio_data)}")
        elif (
            hasattr(response, "response_metadata")
            and "audio" in response.response_metadata
        ):
            audio_data = response.response_metadata["audio"]
            logger.info(f"Found audio in response_metadata, type: {type(audio_data)}")

        if audio_data:
            # Check if it's already bytes or needs base64 decoding
            import base64

            if isinstance(audio_data, dict) and "data" in audio_data:
                # Gemini returns {data: base64_string}
                audio_content = base64.b64decode(audio_data["data"])
            elif isinstance(audio_data, str):
                # If it's a string, try base64 decode
                audio_content = base64.b64decode(audio_data)
            elif isinstance(audio_data, bytes):
                # Already bytes, use directly
                audio_content = audio_data
            else:
                raise ValueError(f"Unexpected audio data type: {type(audio_data)}")

            # Save the audio file to the model
            filename = f"voice_{video.id}.wav"
            video.voice_file.save(filename, ContentFile(audio_content), save=False)
            video.save()

            # Get audio duration using ffprobe
            try:
                with get_temporary_file_path(video.voice_file) as temp_audio_path:
                    duration = get_audio_duration(temp_audio_path)
                    video.voice_duration = duration
                    logger.info(f"Voice file duration: {duration:.2f} seconds")
            except Exception as e:
                logger.warning(f"Could not get audio duration: {e}")

            # Update video status
            video.status = GenVideo.Statuses.VOICE_READY
            video.save()

            logger.info(
                f"✓ Voice file generated successfully for video {video.id} [Google/Gemini] - {len(audio_content)} bytes, {video.voice_duration:.2f}s"
            )

            # Automatically generate SRT file
            generate_srt_file(video)
        else:
            logger.error(f"Response content: {response}")
            raise ValueError(_("V odgovoru ni zvočnih podatkov"))

    except GenVideo.DoesNotExist:
        logger.error(f"Video with id {video.id} does not exist")
        raise
    except Exception as e:
        logger.error(f"✗ Error generating voice file for video {video.id}: {str(e)}")
        video = GenVideo.objects.get(id=video.id)
        video.status = GenVideo.Statuses.FAILED
        video.save()
        raise


@db_task()
def get_video_segments(video_instance: GenVideo) -> None:
    try:
        ensure_google_api_key()

        video_instance.status = GenVideo.Statuses.GENERATING_SEGMENTS
        video_instance.save()

        video_instance.segments.all().delete()  # Clear existing segments if any

        # prompt the model for the minutes
        model = init_chat_model("gemini-3-flash-preview", model_provider="google_genai")
        logger.info(
            f"Video segments prompt for video {video_instance.id}: {video_instance.video_segments_keywords_prompt}"
        )
        model_response = model.invoke(video_instance.video_segments_keywords_prompt)

        # Check for blocked content
        if hasattr(model_response, "response_metadata"):
            block_reason = model_response.response_metadata.get("block_reason")
            if block_reason:
                error_msg = f"Gemini blocked content: {block_reason}"
                logger.error(error_msg)
                video_instance.status = GenVideo.Statuses.FAILED
                video_instance.error_type = GenVideo.ErrorTypes.SEGMENTS_GENERATION
                video_instance.error_details = _(
                    "Gemini je blokiral vsebino. Vsebina morda krši Gemini politiko. Poskusite preformulirati prompt ali scenarij."
                )
                video_instance.save()
                raise ValueError(error_msg)

        data = model_response.content

        # Check for empty response
        if not data or (isinstance(data, str) and not data.strip()):
            error_msg = "Gemini returned empty response"
            logger.error(error_msg)
            video_instance.status = GenVideo.Statuses.FAILED
            video_instance.error_type = GenVideo.ErrorTypes.SEGMENTS_GENERATION
            video_instance.error_details = _(
                "Gemini je vrnil prazen odgovor. Poskusite preformulirati prompt."
            )
            video_instance.save()
            raise ValueError(error_msg)

        logger.info(data)
        if isinstance(data, str):
            data = data.strip().strip("`").strip().strip("json").strip("python")
            data = json.loads(data)
        elif isinstance(data, dict) and "text" in data:
            data = json.loads(data["text"])
        elif isinstance(data, list):
            logger.info(len(data))
            data = json.loads(data[0]["text"])
        else:
            raise ValueError(_("Nepričakovan format odgovora modela za video segmente"))

        for i, segment_data in enumerate(data):
            start = float(segment_data["start"].strip())
            end = float(segment_data["end"].strip())
            logger.info(segment_data["start"])
            logger.info(segment_data["end"])

            if start >= end:
                video_instance.status = GenVideo.Statuses.FAILED
                video_instance.error_type = GenVideo.ErrorTypes.SEGMENTS_GENERATION
                video_instance.error_details = _(
                    "Neveljavni časi segmenta %(segment)s: začetek %(start)s >= konec %(end)s"
                ) % {"segment": i + 1, "start": start, "end": end}
                video_instance.save()

            # If this is the last segment and we have voice_duration, use it
            if i == len(data) - 1 and video_instance.voice_duration:
                logger.info(
                    f"Adjusting last segment end_time from {end} to {video_instance.voice_duration}"
                )
                end = video_instance.voice_duration
            elif len(data) > i + 1:
                next_start = float(data[i + 1]["start"].strip())
                if next_start > end + 0.01:
                    end = next_start - 0.01

            VideoSegment.objects.create(
                video=video_instance,
                text=segment_data["text"],
                order=i + 1,
                query=", ".join(segment_data["keywords"]),
                start_time=start,
                end_time=end,
            )
            logger.info(
                f"✓ Created segment {i+1} for video {video_instance.id}: {segment_data['text'][:50]}..."
            )

        video_instance.status = GenVideo.Statuses.SEGMENTS_READY
        video_instance.save()

    except Exception as e:
        logger.error(
            f"✗ Error generating segments for video {video_instance.id}: {str(e)}"
        )
        video_instance.status = GenVideo.Statuses.FAILED
        video_instance.error_type = GenVideo.ErrorTypes.SEGMENTS_GENERATION
        video_instance.error_details = _(
            "Napaka pri generiranju segmentov: %(error)s"
        ) % {"error": str(e)}
        video_instance.save()
        raise


@db_task()
def generate_srt_file(video: GenVideo) -> None:
    """
    Generate SRT subtitle file from audio file with gemini,

    Args:
        video: GenVideo instance
    """
    try:
        video.status = GenVideo.Statuses.GENERATING_SUBTITLES
        video.save()

        if not video.voice_file:
            raise ValueError(f"Video {video.id} has no voice_file to generate SRT from")

        if video.language == "sl":
            request_transcription(video)

        else:
            client = genai.Client()
            with get_temporary_file_path(video.voice_file) as voice_file:
                logger.info(f"Uploading voice file for video {video.id}: {voice_file}")
                gemini_file = client.files.upload(file=voice_file)
                while gemini_file.state.name == "PROCESSING":
                    time.sleep(2)
                    gemini_file = client.files.get(name=gemini_file.name)

            if video.subtitle_max_words_per_screen:
                logger.info(
                    f"Using max_words_per_screen={video.subtitle_max_words_per_screen} for video {video.id}"
                )
                subtitle_limit_prompt = f"""
    Omeji število besed na zaslonu na največ {video.subtitle_max_words_per_screen}.
    Če je v enem segmentu več besed, jih razdeli v več delov, da bo na zaslonu hkrati največ {video.subtitle_max_words_per_screen} besed.
    Razdeli smiselno, po stavkih ali pomišljajih, ne pa naključno v sredini stavka.
                """
            else:
                subtitle_limit_prompt = ""

            contents = [
                Content(
                    role="user",
                    parts=[
                        Part.from_uri(
                            file_uri=gemini_file.uri, mime_type=gemini_file.mime_type
                        ),
                        Part.from_text(text=f"""
    Vrni mi samo vsebino SRT datoteke za podnapise iz priloženega zvočnega posnetka, brez dodatnih pojasnil ali besedila.
    Zgeneriraj podnapise in mi vrni vsebino za SRT datoteko. Bodi pozoren, da bo vsebina SRT datoteke pravilno formatirana in bo vsebovala TOČNE časovne kode za vsak podnapis.
    {subtitle_limit_prompt}
    Vsebuje naj tudi časovne kode, ki naj bodo vse v formatu HH:MM:SS,mmm --> HH:MM:SS,mmm.
    Spodaj je primer za enkratno referenco:
    1
    00:02:16,612 --> 00:02:19,376
    Senator, we're making
    our final approach into Coruscant.
    """),
                    ],
                )
            ]
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=contents,
            )
            filename = f"srt_{video.id}.srt"
            srt_content = response.text.strip("`").strip("srt")
            logger.info(f"SRT content generated for video {video.id}:\n{srt_content}")
            # Validate SRT content before saving
            is_valid, validation_message = validate_srt_content(srt_content, video)
            if not is_valid:
                raise ValueError(f"{_('Neveljavna SRT vsebina')}: {validation_message}")

            logger.info(f"SRT validation result: {validation_message}")

            video.srt_content = srt_content
            video.srt_file.save(filename, ContentFile(srt_content), save=False)
            video.status = GenVideo.Statuses.SUBTITLES_READY
            video.save()

            logger.info(f"✓ SRT file generated for video {video.id}: {filename}")

            # Automatically generate video segments if video has none
            get_video_segments(video)

    except Exception as e:
        logger.error(f"✗ Error generating SRT file for video {video.id}: {str(e)}")
        video.status = GenVideo.Statuses.FAILED
        video.error_type = GenVideo.ErrorTypes.SRT_GENERATION
        video.error_details = _("Napaka pri generiranju SRT datoteke: %(error)s") % {
            "error": str(e)
        }
        video.save()
        # raise


@db_task()
def render_final_video(video: GenVideo) -> None:
    """
    Combine all VideoSegment clips with ffmpeg, add audio and subtitles.

    Process:
    1. Download each video from URL stored in video_proposals
    2. Cut each video according to segment start_time/end_time
    3. Concatenate all clips in order
    4. Add voice_file as audio track
    5. Burn-in subtitles from srt_file
    6. Save to GenVideo.final_file

    Args:
        video: GenVideo instance
    """
    try:
        FinalVideoRenderer(video).render()
    except FFmpegTimeoutError as e:
        logger.error(f"Error rendering video {video}: {str(e)}")
        video.status = GenVideo.Statuses.FAILED
        video.error_type = GenVideo.ErrorTypes.TIMEOUT
        video.progress = ""
        video.save()
        raise
    except Exception as e:
        logger.error(f"Error rendering video {video}: {str(e)}")
        video.status = GenVideo.Statuses.FAILED
        video.error_type = GenVideo.ErrorTypes.RENDERING
        video.error_details = _("Napaka pri renderiranju končnega videa: %(error)s") % {
            "error": str(e)
        }
        video.save()
        raise


@db_task()
def request_transcription(video: GenVideo) -> None:
    # if there's no sound file, eject
    if video.voice_file is None:
        raise ValueError("Can't transcribe without a sound file.")

    # Uporabi context manager za S3 kompatibilnost
    with get_temporary_file_path(video.voice_file) as temp_path:
        task_id = tipko_api.upload(temp_path)
        TipkoRequest.objects.create(video=video, tipko_task_id=task_id)
        logger.info("File successfully uploaded.")


# @db_periodic_task(crontab(minute="*/1"))
# def check_status_and_download_transcription() -> None:
#     waiting_tipkos = TipkoRequest.objects.filter(
#         tipko_task_id__isnull=False,
#         status="PENDING",
#     )

#     for tipko_instance in waiting_tipkos:
#         status_response = tipko_api.get_status(tipko_instance.tipko_task_id)
#         logger.info(
#             f"Checking status for {tipko_instance.id}: {status_response['status']}"
#         )
#         if status_response["status"] == "done":
#             logger.info(f"checking transcription for {tipko_instance.id}")
#             srt_response = tipko_api.get_transcription_file(
#                 tipko_instance.tipko_task_id
#             )
#             if srt_response.status_code != 200:
#                 logger.error(
#                     f"Failed to download transcription for {tipko_instance.id}: {srt_response.status_code}"
#                 )
#                 continue
#             srt_content = srt_response.content.decode("utf-8")
#             video = tipko_instance.video
#             video.srt_content = srt_content
#             video.save()
#             filename = f"transcript_{tipko_instance.tipko_task_id}.srt"
#             video.srt_file.save(
#                 filename,
#                 ContentFile(srt_content),
#                 save=False,
#             )
#             tipko_instance.status = "DONE"
#             tipko_instance.save()
#             logger.info("Transcription downloaded and saved.")
#             # Automatically generate video segments if video has none
#             get_video_segments(video)
