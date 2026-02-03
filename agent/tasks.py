import json
import logging
import os
import sys
import time

from django.conf import settings
from django.core.files.base import ContentFile
from elevenlabs.client import ElevenLabs
from google import genai
from google.genai.types import Content, Part
from huey.contrib.djhuey import db_task
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from openai import OpenAI

from agent.models import GenVideo, VideoSegment
from agent.utils import ensure_google_api_key, get_temporary_file_path

# Configure logging for Huey tasks
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


def validate_srt_content(srt_content: str) -> tuple[bool, str]:
    """
    Validate SRT subtitle content before saving.

    Checks:
    - Content is not empty
    - Contains valid subtitle blocks (number, timecode, text)
    - Timecodes are in correct format
    - No missing or empty subtitle entries

    Args:
        srt_content: The SRT file content string

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not srt_content or not srt_content.strip():
        return False, "SRT content is empty"

    lines = srt_content.strip().split("\n")

    # Basic structure check - should have at least 3 lines (number, timecode, text)
    if len(lines) < 3:
        return False, "SRT content too short - missing subtitle blocks"

    # Track subtitle blocks
    subtitle_count = 0
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
        import re

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
        video.status = GenVideo.Statuses.GENERATING_VOICE
        video.save()

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

        # Use voice_model from video or default (Rachel voice)
        voice_id = video.voice_model or "21m00Tcm4TlvDq8ikWAM"

        logger.info(f"Generating voice for video {video.id} with voice {voice_id}")

        # Initialize ElevenLabs client
        client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)

        # Generate speech from scenario
        audio_generator = client.text_to_speech.convert(
            voice_id=voice_id,
            text=video.scenario,
            model_id="eleven_multilingual_v2",
        )

        # Collect audio chunks into bytes
        audio_bytes = b"".join(audio_generator)

        # Save the audio file to the model
        filename = f"voice_{video.id}.mp3"
        video.voice_file.save(filename, ContentFile(audio_bytes), save=False)

        # Update video status
        video.status = GenVideo.Statuses.VOICE_READY
        video.save()

        logger.info(
            f"✓ Voice file generated successfully for video {video.id} [ElevenLabs]"
        )

    except GenVideo.DoesNotExist:
        logger.error(f"Video with id {video.id} does not exist")
        raise
    except Exception as e:
        logger.error(f"✗ Error generating voice file for video {video.id}: {str(e)}")
        video = GenVideo.objects.get(id=video.id)
        video.status = GenVideo.Statuses.FAILED
        video.error_type = GenVideo.ErrorTypes.VOICE_GENERATION
        video.error_details = f"Error generating voice file (ElevenLabs): {str(e)}"
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
        video.error_details = f"Error generating voice file (Gemini): {str(e)}"
        video.error_details = f"Error generating voice file (OpenAI): {str(e)}"
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
            raise ValueError("No audio data in response")

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
                video_instance.error_details = f"{error_msg}. Vsebina morda krši Gemini politiko. Poskusite preformulirati prompt ali scenarij."
                video_instance.save()
                raise ValueError(error_msg)

        data = model_response.content

        # Check for empty response
        if not data or (isinstance(data, str) and not data.strip()):
            error_msg = "Gemini returned empty response"
            logger.error(error_msg)
            video_instance.status = GenVideo.Statuses.FAILED
            video_instance.error_type = GenVideo.ErrorTypes.SEGMENTS_GENERATION
            video_instance.error_details = (
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
            raise ValueError("Unexpected model response format for video segments")

        for i, segment_data in enumerate(data):
            start = float(segment_data["start"].strip())
            end = float(segment_data["end"].strip())
            logger.info(segment_data["start"])
            logger.info(segment_data["end"])

            if start >= end:
                video_instance.status = GenVideo.Statuses.FAILED
                video_instance.error_type = GenVideo.ErrorTypes.SEGMENTS_GENERATION
                video_instance.error_details = f"Invalid segment times for segment {i+1}: start {start} >= end {end}"
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
        video_instance.error_details = f"Error generating segments: {str(e)}"
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

        client = genai.Client()
        with get_temporary_file_path(video.voice_file) as voice_file:
            logger.info(f"Uploading voice file for video {video.id}: {voice_file}")
            gemini_file = client.files.upload(file=voice_file)
            while gemini_file.state.name == "PROCESSING":
                time.sleep(2)
                gemini_file = client.files.get(name=gemini_file.name)

        contents = [
            Content(
                role="user",
                parts=[
                    Part.from_uri(
                        file_uri=gemini_file.uri, mime_type=gemini_file.mime_type
                    ),
                    Part.from_text(
                        text="""
Vrni miiii samo vsebino SRT datoteke za podnapise iz priloženega zvočnega posnetka, brez dodatnih pojasnil ali besedila.
Zgeneriraj podnapise in mi vrni vsebino za SRT datoteko.
Vsebuje naj tudi časovne kode, ki naj bodo vse v formatu HH:MM:SS,mmm --> HH:MM:SS,mmm.
Spodaj je primer za enkratno referenco:
1
00:02:16,612 --> 00:02:19,376
Senator, we're making
our final approach into Coruscant.
"""
                    ),
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
        is_valid, validation_message = validate_srt_content(srt_content)
        if not is_valid:
            raise ValueError(f"Invalid SRT content: {validation_message}")

        logger.info(f"SRT validation result: {validation_message}")

        video.srt_content = srt_content
        video.srt_file.save(filename, ContentFile(srt_content), save=False)
        video.status = GenVideo.Statuses.SUBTITLES_READY
        video.save()

        logger.info(f"✓ SRT file generated for video {video.id}: {filename}")

        # Automatically generate video segments if video has none
        if not video.segments.exists():
            get_video_segments(video)

    except Exception as e:
        logger.error(f"✗ Error generating SRT file for video {video.id}: {str(e)}")
        video.status = GenVideo.Statuses.FAILED
        video.error_type = GenVideo.ErrorTypes.SRT_GENERATION
        video.error_details = f"Error generating SRT file: {str(e)}"
        video.save()
        raise


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
    import subprocess
    import tempfile
    from pathlib import Path

    from agent.utils import (
        get_temporary_file,
        get_temporary_file_from_url,
        get_temporary_file_path,
    )

    try:
        video.status = GenVideo.Statuses.RENDERING
        video.save()

        # Get all segments with video URLs
        segments = video.segments.filter(video_proposals__0__selected=True).order_by(
            "order"
        )

        if not segments.exists():
            raise ValueError(f"Video {video} has no segments with selected videos")

        if not video.voice_file:
            raise ValueError(f"Video {video} has no voice file")

        logger.info(f"Starting render for video {video} with {segments.count()} clips")

        # Create temp directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Step 1: Cut and prepare each clip
            clip_files = []
            for i, segment in enumerate(segments):
                video_url = segment.video_proposals[0].get("video_url")
                if not video_url:
                    raise ValueError(f"Segment {segment.id} has no video URL")

                # Check if this is an image instead of video
                is_image = segment.video_proposals[0].get("is_image", False)

                # Get video dimensions from proposals
                width = segment.video_proposals[0].get("width")
                height = segment.video_proposals[0].get("height")

                # Validate dimensions
                if width is None or height is None:
                    logger.warning(
                        f"Segment {segment.id} missing dimensions (width={width}, height={height}), assuming vertical video"
                    )
                    width = 1080
                    height = 1920

                horizontal_mode = segment.video_proposals[0].get(
                    "horizontal_mode", "crop"
                )

                output_file = temp_path / f"clip_{i:03d}.mp4"
                duration = segment.duration()

                logger.info(
                    f"Processing clip {i+1}/{segments.count()}: {duration:.2f}s from URL (dimensions: {width}x{height}, is_image: {is_image}, mode: {horizontal_mode})"
                )

                if is_image:
                    # Handle image: download and convert to video
                    with get_temporary_file(video_url) as input_file:
                        # Determine if image is horizontal and needs special processing
                        is_horizontal = width > height

                        logger.info(f"Converting image to video: {width}x{height}")

                        if is_horizontal:
                            target_aspect = 9 / 16  # width/height for vertical video

                            if horizontal_mode == "crop":
                                # Mode 1: Pure crop - extract center portion
                                crop_width = int(height * target_aspect)
                                crop_height = height
                                crop_x = int((width - crop_width) / 2)

                                logger.info(
                                    f"Horizontal image - CROP mode: {width}x{height} -> {crop_width}x{crop_height}"
                                )

                                # Create video from image with crop
                                cmd = [
                                    "ffmpeg",
                                    "-loop",
                                    "1",
                                    "-i",
                                    input_file,
                                    "-t",
                                    str(duration),
                                    "-vf",
                                    f"crop={crop_width}:{crop_height}:{crop_x}:0,scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1",
                                    "-c:v",
                                    "libx264",
                                    "-preset",
                                    "medium",
                                    "-crf",
                                    "23",
                                    "-r",
                                    "30",
                                    "-g",
                                    "30",
                                    "-pix_fmt",
                                    "yuv420p",
                                    "-an",
                                    "-y",
                                    str(output_file),
                                ]

                            elif horizontal_mode == "blur":
                                # Mode 2: Blur background - scale image to fit width, blur and extend to fill height
                                logger.info(
                                    f"Horizontal image - BLUR mode: {width}x{height}"
                                )

                                # Complex filter for blur
                                video_filter = (
                                    "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:5[blurred];"
                                    "[0:v]scale=1080:-1:force_original_aspect_ratio=decrease[main];"
                                    "[blurred][main]overlay=(W-w)/2:(H-h)/2,setsar=1"
                                )

                                cmd = [
                                    "ffmpeg",
                                    "-loop",
                                    "1",
                                    "-i",
                                    input_file,
                                    "-t",
                                    str(duration),
                                    "-filter_complex",
                                    video_filter,
                                    "-c:v",
                                    "libx264",
                                    "-preset",
                                    "medium",
                                    "-crf",
                                    "23",
                                    "-r",
                                    "30",
                                    "-g",
                                    "30",
                                    "-pix_fmt",
                                    "yuv420p",
                                    "-an",
                                    "-y",
                                    str(output_file),
                                ]

                            elif horizontal_mode == "blur_crop":
                                # Mode 3: Blur & Crop - crop to square (width = height), then blur for top/bottom
                                crop_width = height  # Make it square!
                                crop_height = height
                                crop_x = int((width - crop_width) / 2)

                                logger.info(
                                    f"Horizontal image - BLUR&CROP mode: {width}x{height} -> crop {crop_width}x{crop_height} (square) + blur"
                                )

                                # Complex filter - use original image for blur background
                                video_filter = (
                                    "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:5[blurred];"
                                    f"[0:v]crop={crop_width}:{crop_height}:{crop_x}:0,scale=1080:-1:force_original_aspect_ratio=decrease[main];"
                                    "[blurred][main]overlay=(W-w)/2:(H-h)/2,setsar=1"
                                )

                                cmd = [
                                    "ffmpeg",
                                    "-loop",
                                    "1",
                                    "-i",
                                    input_file,
                                    "-t",
                                    str(duration),
                                    "-filter_complex",
                                    video_filter,
                                    "-c:v",
                                    "libx264",
                                    "-preset",
                                    "medium",
                                    "-crf",
                                    "23",
                                    "-r",
                                    "30",
                                    "-g",
                                    "30",
                                    "-pix_fmt",
                                    "yuv420p",
                                    "-an",
                                    "-y",
                                    str(output_file),
                                ]
                            else:
                                # Default to crop if unknown mode
                                crop_width = int(height * target_aspect)
                                crop_height = height
                                crop_x = int((width - crop_width) / 2)

                                cmd = [
                                    "ffmpeg",
                                    "-loop",
                                    "1",
                                    "-i",
                                    input_file,
                                    "-t",
                                    str(duration),
                                    "-vf",
                                    f"crop={crop_width}:{crop_height}:{crop_x}:0,scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1",
                                    "-c:v",
                                    "libx264",
                                    "-preset",
                                    "medium",
                                    "-crf",
                                    "23",
                                    "-r",
                                    "30",
                                    "-g",
                                    "30",
                                    "-pix_fmt",
                                    "yuv420p",
                                    "-an",
                                    "-y",
                                    str(output_file),
                                ]
                        else:
                            # For vertical image, use existing approach (scale and pad)
                            logger.info(
                                f"Vertical image - scaling with padding if needed"
                            )

                            cmd = [
                                "ffmpeg",
                                "-loop",
                                "1",
                                "-i",
                                input_file,
                                "-t",
                                str(duration),
                                "-vf",
                                "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1",
                                "-c:v",
                                "libx264",
                                "-preset",
                                "medium",
                                "-crf",
                                "23",
                                "-r",
                                "30",
                                "-g",
                                "30",
                                "-pix_fmt",
                                "yuv420p",
                                "-an",
                                "-y",
                                str(output_file),
                            ]

                        logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        if result.returncode != 0:
                            logger.error(f"FFmpeg stderr: {result.stderr}")
                            raise RuntimeError(
                                f"FFmpeg image to video conversion failed for clip {i}: {result.stderr}"
                            )

                        logger.info(
                            f"Successfully created video from image {i}: {output_file}"
                        )
                        clip_files.append(output_file)
                else:
                    # Handle video (existing code)
                    with get_temporary_file(video_url) as input_file:
                        # Determine if video is horizontal and needs special processing
                        is_horizontal = width > height

                        if is_horizontal:
                            target_aspect = 9 / 16  # width/height for vertical video

                            if horizontal_mode == "crop":
                                # Mode 1: Pure crop - extract center portion
                                crop_width = int(height * target_aspect)
                                crop_height = height
                                crop_x = int((width - crop_width) / 2)

                                logger.info(
                                    f"Horizontal video - CROP mode: {width}x{height} -> {crop_width}x{crop_height}"
                                )

                                # Simple filter, can use -vf
                                cmd = [
                                    "ffmpeg",
                                    "-i",
                                    input_file,
                                    "-t",
                                    str(duration),
                                    "-vf",
                                    f"crop={crop_width}:{crop_height}:{crop_x}:0,scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1",
                                    "-c:v",
                                    "libx264",
                                    "-preset",
                                    "medium",
                                    "-crf",
                                    "23",
                                    "-r",
                                    "30",
                                    "-g",
                                    "30",
                                    "-pix_fmt",
                                    "yuv420p",
                                    "-an",
                                    "-y",
                                    str(output_file),
                                ]

                            elif horizontal_mode == "blur":
                                # Mode 2: Blur background - scale video to fit width, blur and extend to fill height
                                logger.info(
                                    f"Horizontal video - BLUR mode: {width}x{height}"
                                )

                                # Complex filter, must use -filter_complex
                                video_filter = (
                                    "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:5[blurred];"
                                    "[0:v]scale=1080:-1:force_original_aspect_ratio=decrease[main];"
                                    "[blurred][main]overlay=(W-w)/2:(H-h)/2,setsar=1"
                                )

                                cmd = [
                                    "ffmpeg",
                                    "-i",
                                    input_file,
                                    "-t",
                                    str(duration),
                                    "-filter_complex",
                                    video_filter,
                                    "-c:v",
                                    "libx264",
                                    "-preset",
                                    "medium",
                                    "-crf",
                                    "23",
                                    "-r",
                                    "30",
                                    "-g",
                                    "30",
                                    "-pix_fmt",
                                    "yuv420p",
                                    "-an",
                                    "-y",
                                    str(output_file),
                                ]

                            elif horizontal_mode == "blur_crop":
                                # Mode 3: Blur & Crop - crop to square (width = height), then blur for top/bottom
                                crop_width = height  # Make it square!
                                crop_height = height
                                crop_x = int((width - crop_width) / 2)

                                logger.info(
                                    f"Horizontal video - BLUR&CROP mode: {width}x{height} -> crop {crop_width}x{crop_height} (square) + blur"
                                )

                                # Complex filter, must use -filter_complex
                                # Use ORIGINAL video [0:v] for blur background, cropped square for main
                                video_filter = (
                                    "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:5[blurred];"
                                    f"[0:v]crop={crop_width}:{crop_height}:{crop_x}:0,scale=1080:-1:force_original_aspect_ratio=decrease[main];"
                                    "[blurred][main]overlay=(W-w)/2:(H-h)/2,setsar=1"
                                )

                                cmd = [
                                    "ffmpeg",
                                    "-i",
                                    input_file,
                                    "-t",
                                    str(duration),
                                    "-filter_complex",
                                    video_filter,
                                    "-c:v",
                                    "libx264",
                                    "-preset",
                                    "medium",
                                    "-crf",
                                    "23",
                                    "-r",
                                    "30",
                                    "-g",
                                    "30",
                                    "-pix_fmt",
                                    "yuv420p",
                                    "-an",
                                    "-y",
                                    str(output_file),
                                ]
                            else:
                                # Default to crop if unknown mode
                                crop_width = int(height * target_aspect)
                                crop_height = height
                                crop_x = int((width - crop_width) / 2)

                                cmd = [
                                    "ffmpeg",
                                    "-i",
                                    input_file,
                                    "-t",
                                    str(duration),
                                    "-vf",
                                    f"crop={crop_width}:{crop_height}:{crop_x}:0,scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1",
                                    "-c:v",
                                    "libx264",
                                    "-preset",
                                    "medium",
                                    "-crf",
                                    "23",
                                    "-r",
                                    "30",
                                    "-g",
                                    "30",
                                    "-pix_fmt",
                                    "yuv420p",
                                    "-an",
                                    "-y",
                                    str(output_file),
                                ]
                        else:
                            # For vertical video, use existing approach (scale and pad)
                            logger.info(
                                f"Vertical video - scaling with padding if needed"
                            )

                            cmd = [
                                "ffmpeg",
                                "-i",
                                input_file,
                                "-t",
                                str(duration),
                                "-vf",
                                "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1",
                                "-c:v",
                                "libx264",
                                "-preset",
                                "medium",
                                "-crf",
                                "23",
                                "-r",
                                "30",
                                "-g",
                                "30",
                                "-pix_fmt",
                                "yuv420p",
                                "-an",
                                "-y",
                                str(output_file),
                            ]

                        logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        if result.returncode != 0:
                            logger.error(f"FFmpeg stderr: {result.stderr}")
                            raise RuntimeError(
                                f"FFmpeg clip processing failed for clip {i}: {result.stderr}"
                            )

                        logger.info(f"Successfully created clip {i}: {output_file}")
                        clip_files.append(output_file)

            # Step 2: Create concat file
            concat_file = temp_path / "concat.txt"
            with open(concat_file, "w") as f:
                for clip in clip_files:
                    f.write(f"file '{clip}'\n")

            # Step 3: Concatenate all clips
            concatenated_file = temp_path / "concatenated.mp4"
            print("Concatenating clips...")

            cmd = [
                "ffmpeg",
                "-f",
                "concat",  # Use concat demuxer (reads from concat.txt file)
                "-safe",
                "0",  # Allow absolute file paths in concat.txt
                "-i",
                str(concat_file),  # Input: concat.txt with list of video files
                "-c",
                "copy",  # Copy video/audio streams without re-encoding (fast!)
                "-y",  # Overwrite output file without asking
                str(concatenated_file),  # Output: single concatenated video
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg concatenation failed: {result.stderr}")

            # Step 4: Add audio and subtitles (if available)
            final_output = temp_path / "final.mp4"
            print("Adding audio and subtitles...")
            with get_temporary_file_path(video.voice_file) as voice_file:
                cmd = [
                    "ffmpeg",
                    "-i",
                    str(concatenated_file),  # Input 1: Concatenated video (no audio)
                    "-i",
                    voice_file,  # Input 2: Voice audio file (narration)
                ]

                # Add subtitles if available
                if video.srt_file:
                    with get_temporary_file_path(video.srt_file) as srt_file:
                        # Get subtitle parameters from video model
                        font_size = video.subtitle_font_size or 12
                        font_family = video.subtitle_font_family or "Montserrat"
                        font_weight = video.subtitle_font_weight or "900"
                        stroke_weight = video.subtitle_stroke_weight or 3
                        shadow = video.subtitle_shadow or 1
                        vertical_position = video.subtitle_vertical_position or 10

                        # Determine if font should be bold based on weight
                        bold = 1 if int(font_weight) >= 700 else 0

                        # Calculate MarginV (distance from bottom in pixels)
                        # In ASS format, MarginV is the margin from the bottom edge
                        # For 1080p video, convert percentage to pixels from bottom
                        # Higher percentage = higher position = larger margin from bottom
                        max_margin_v = 300
                        margin_v = int((vertical_position / 100) * max_margin_v)
                        print("Margin:", margin_v)
                        # margin_v = 200

                        # Build style string dynamically (ASS subtitle format)
                        # FontName: Font family to use
                        # FontSize: Size in pixels
                        # Bold: 1=bold, 0=normal
                        # PrimaryColour: Text color in &HAABBGGRR format (white = &H00FFFFFF)
                        # OutlineColour: Stroke/outline color (black = &H00000000)
                        # Outline: Stroke width in pixels
                        # Shadow: Shadow offset/intensity
                        # MarginV: Vertical margin from bottom edge in pixels
                        style = f"FontName={font_family},FontSize={font_size},Bold={bold},PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline={stroke_weight},Shadow={shadow},MarginV={margin_v}"

                        # Burn-in subtitles with custom style
                        # subtitles filter: Renders SRT file onto video permanently
                        # force_style: Overrides all subtitle styling with our custom style
                        subtitle_filter = f"subtitles={srt_file}:force_style='{style}'"
                        cmd.extend(
                            [
                                "-vf",
                                subtitle_filter,  # Apply subtitle video filter
                                "-c:v",
                                "libx264",  # Video codec: H.264 (must re-encode to burn-in subtitles)
                                "-preset",
                                "medium",  # Encoding speed preset
                                "-crf",
                                "23",  # Quality level (constant rate factor)
                            ]
                        )
                        # Add audio settings
                        cmd.extend(
                            [
                                "-c:a",
                                "aac",  # Audio codec: AAC
                                "-b:a",
                                "192k",  # Audio bitrate: 192 kbps (good quality)
                                "-shortest",  # End output when shortest input stream ends (video or audio)
                                "-y",  # Overwrite output file without asking
                                str(final_output),  # Output file path
                            ]
                        )
                        result = subprocess.run(cmd, capture_output=True, text=True)
                else:
                    # No subtitles - just copy video stream (fast, no re-encoding)
                    cmd.extend(["-c:v", "copy"])  # Copy video codec as-is

                    # Add audio settings
                    cmd.extend(
                        [
                            "-c:a",
                            "aac",  # Audio codec: AAC
                            "-b:a",
                            "192k",  # Audio bitrate: 192 kbps
                            "-shortest",  # End when shortest input stream ends
                            "-y",  # Overwrite output file without asking
                            str(final_output),  # Output file path
                        ]
                    )

                    result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg final render failed: {result.stderr}")

                # Step 5: Save to model
                print("Saving final video to database...")
                with open(final_output, "rb") as f:
                    filename = f"final_video_{video.id}.mp4"
                    video.final_file.save(filename, ContentFile(f.read()), save=False)

                video.status = GenVideo.Statuses.COMPLETED
                video.save()

                print(f"Video {video} rendered successfully!")

    except Exception as e:
        print(f"Error rendering video {video}: {str(e)}")
        video.status = GenVideo.Statuses.FAILED
        video.error_type = GenVideo.ErrorTypes.RENDERING
        video.error_details = f"Error rendering final video: {str(e)}"
        video.save()
        raise
