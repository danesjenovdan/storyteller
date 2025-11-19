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


@db_task()
def generate_scenario_from_prompt(video_instance: GenVideo) -> None:
    """
    Generate scenario from start_prompt using Gemini.
    This runs first and creates the scenario text.
    """
    try:
        ensure_google_api_key()

        if not video_instance.start_prompt:
            logger.warning(
                f"Video {video_instance.id} has no start_prompt, skipping scenario generation"
            )
            return

        video_instance.status = GenVideo.Statuses.GENERATING_SCENARIO
        video_instance.save()

        logger.info(
            f"Generating scenario for video {video_instance.id} from start_prompt"
        )

        model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
        model_response = model.invoke(video_instance.start_prompt)
        video_instance.scenario = model_response.content
        video_instance.status = GenVideo.Statuses.SCENARIO_READY
        video_instance.save()

        logger.info(f"✓ Scenario generated for video {video_instance.id}")

    except Exception as e:
        logger.error(
            f"✗ Error generating scenario for video {video_instance.id}: {str(e)}"
        )
        video_instance.status = GenVideo.Statuses.FAILED
        video_instance.save()
        raise


@db_task()
def simplify_scenario(video_instance: GenVideo) -> None:
    """
    Generate content_script from scenario using Gemini.
    This runs second and creates the simplified script.
    """
    try:
        ensure_google_api_key()

        video_instance.status = GenVideo.Statuses.GENERATING_SCRIPT
        video_instance.save()

        logger.info(f"Simplifying scenario for video {video_instance.id}")

        # prompt the model for the minutes
        model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
        model_response = model.invoke(video_instance.simplify_prompt)
        video_instance.content_script = model_response.content
        video_instance.status = GenVideo.Statuses.SCRIPT_READY
        video_instance.save()

        logger.info(f"✓ Content script generated for video {video_instance.id}")

    except Exception as e:
        logger.error(
            f"✗ Error generating content script for video {video_instance.id}: {str(e)}"
        )
        video_instance.status = GenVideo.Statuses.FAILED
        video_instance.save()
        raise


@db_task()
def generate_voice_file_eleven_labs(video: int) -> None:
    """
    Generate voice file from content_script using ElevenLabs SDK.

    Args:
        video.id: ID of the GenVideo instance
    """
    try:
        video.status = GenVideo.Statuses.GENERATING_VOICE
        video.save()

        if not video.content_script:
            raise ValueError(
                f"Video {video.id} has no content_script to convert to speech"
            )

        if not settings.ELEVENLABS_API_KEY:
            raise ValueError("ELEVENLABS_API_KEY is not configured in settings")

        # Check text length (ElevenLabs charges per character)
        text_length = len(video.content_script)
        logger.info(f"Content script length: {text_length} characters")

        # Warning for very long texts (>5000 chars may be expensive)
        if text_length > 5000:
            logger.warning(f"Long text ({text_length} chars) may consume many credits")

        # Use voice_model from video or default (Rachel voice)
        voice_id = video.voice_model or "21m00Tcm4TlvDq8ikWAM"

        logger.info(f"Generating voice for video {video.id} with voice {voice_id}")

        # Initialize ElevenLabs client
        client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)

        # Generate speech from content_script
        audio_generator = client.text_to_speech.convert(
            voice_id=voice_id,
            text=video.content_script,
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
        video.save()
        raise


@db_task()
def generate_voice_file_openai(video: int) -> None:
    """
    Generate voice file from content_script using OpenAI TTS API.
    Much cheaper than ElevenLabs - ~$0.015 per 1000 characters.

    Args:
        video.id: ID of the GenVideo instance
    """
    try:
        video.status = GenVideo.Statuses.GENERATING_VOICE
        video.save()

        if not video.content_script:
            raise ValueError(
                f"Video {video.id} has no content_script to convert to speech"
            )

        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not configured in settings")

        # Check text length
        text_length = len(video.content_script)
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

        # Generate speech from content_script
        response = client.audio.speech.create(
            model="tts-1",  # or "tts-1-hd" for higher quality
            voice=voice,
            input=video.content_script,
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
        video.save()
        raise


@db_task()
def generate_voice_file_gemini(video: int) -> None:
    """
    Generate voice file from content_script using Google Gemini Audio Generation via LangChain.
    Part of Google AI services - uses same API key as Gemini.
    Free tier available with Gemini API!

    Args:
        video.id: ID of the GenVideo instance
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        video.status = GenVideo.Statuses.GENERATING_VOICE
        video.save()

        if not video.content_script:
            raise ValueError(
                f"Video {video.id} has no content_script to convert to speech"
            )

        ensure_google_api_key()

        # Check text length
        text_length = len(video.content_script)
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
            f"Pripravi mi zvokovni posnetek za naslednji text, spodaj imaš še scenarij, ki ga upoštevaj:\n ------ \n {video.content_script}\n ------ \n {video.scenario}",
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

            # Update video status
            video.status = GenVideo.Statuses.VOICE_READY
            video.save()

            logger.info(
                f"✓ Voice file generated successfully for video {video.id} [Google/Gemini] - {len(audio_content)} bytes"
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
        model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
        logger.info(
            f"Video segments prompt for video {video_instance.id}: {video_instance.video_segments_keywords_prompt}"
        )
        model_response = model.invoke(video_instance.video_segments_keywords_prompt)
        data = model_response.content
        data = json.loads(data.strip("`").strip("python"))
        for i, segment_data in enumerate(data):
            logger.info(segment_data["start"])
            logger.info(segment_data["end"])
            VideoSegment.objects.create(
                video=video_instance,
                text=segment_data["text"],
                order=i + 1,
                query=", ".join(segment_data["keywords"]),
                start_time=float(segment_data["start"].strip()),
                end_time=float(segment_data["end"].strip()),
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
Zgeneriraj podnapise in mi vrni vsebino za SRT datoteko. Vsebuje naj tudi časovne kode. Spodaj je primer za enkratno referenco:
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
            model="gemini-2.5-flash",
            contents=contents,
        )
        filename = f"srt_{video.id}.srt"
        srt_content = response.text.strip("`").strip("srt")
        video.srt_content = srt_content
        video.srt_file.save(filename, ContentFile(srt_content), save=False)
        video.status = GenVideo.Statuses.SUBTITLES_READY
        video.save()

        logger.info(f"✓ SRT file generated for video {video.id}: {filename}")

        # Automatically generate video segments
        get_video_segments(video)

    except Exception as e:
        logger.error(f"✗ Error generating SRT file for video {video.id}: {str(e)}")
        video.status = GenVideo.Statuses.FAILED
        video.save()
        raise


@db_task()
def render_final_video(video: GenVideo) -> None:
    """
    Combine all VideoSegment clips with ffmpeg, add audio and subtitles.

    Process:
    1. Cut each VideoSegment.video_file according to start_time/end_time
    2. Concatenate all clips in order
    3. Add voice_file as audio track
    4. Burn-in subtitles from srt_file
    5. Save to GenVideo.final_file

    Args:
        video_id: ID of the GenVideo instance
    """
    import subprocess
    import tempfile
    from pathlib import Path

    try:
        video.status = GenVideo.Statuses.RENDERING
        video.save()

        # Get all segments with video files
        segments = (
            video.segments.filter(video_file__isnull=False)
            .exclude(video_file="")
            .order_by("order")
        )

        if not segments.exists():
            raise ValueError(f"Video {video} has no segment video files")

        if not video.voice_file:
            raise ValueError(f"Video {video} has no voice file")

        logger.info(f"Starting render for video {video} with {segments.count()} clips")

        # Create temp directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Step 1: Cut and prepare each clip
            clip_files = []
            for i, segment in enumerate(segments):
                with get_temporary_file_path(segment.video_file) as input_file:
                    duration = segment.end_time - segment.start_time
                    output_file = temp_path / f"clip_{i:03d}.mp4"

                    logger.info(
                        f"Processing clip {i+1}/{segments.count()}: {duration:.2f}s"
                    )

                    # Cut video to exact duration (no audio, we'll add it later)
                    # Scale to consistent resolution (1080x1920 for portrait)
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
                        "-an",  # Remove audio
                        "-y",
                        str(output_file),
                    ]

                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        raise RuntimeError(
                            f"FFmpeg clip processing failed: {result.stderr}"
                        )

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
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                "-y",
                str(concatenated_file),
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
                    str(concatenated_file),
                    "-i",
                    voice_file,
                ]

                # Add subtitles if available
                if video.srt_file:
                    with get_temporary_file_path(video.srt_file) as srt_file:
                        # Burn-in subtitles
                        subtitle_filter = f"subtitles={srt_file}"
                        cmd.extend(
                            [
                                "-vf",
                                subtitle_filter,
                                "-c:v",
                                "libx264",
                                "-preset",
                                "medium",
                                "-crf",
                                "23",
                            ]
                        )
                else:
                    # Just copy video
                    cmd.extend(["-c:v", "copy"])

                # Add audio settings
                cmd.extend(
                    [
                        "-c:a",
                        "aac",
                        "-b:a",
                        "192k",
                        "-shortest",  # End when shortest stream ends
                        "-y",
                        str(final_output),
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
        video.save()
        raise
