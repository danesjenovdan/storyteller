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
    import subprocess
    import json

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
        model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
        logger.info(
            f"Video segments prompt for video {video_instance.id}: {video_instance.video_segments_keywords_prompt}"
        )
        model_response = model.invoke(video_instance.video_segments_keywords_prompt)
        data = model_response.content
        logger.info(data)
        data = json.loads(data.strip("`").strip("python"))
        for i, segment_data in enumerate(data):
            start = float(segment_data["start"].strip())
            end = float(segment_data["end"].strip())
            logger.info(segment_data["start"])
            logger.info(segment_data["end"])

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
        video.error_type = GenVideo.ErrorTypes.SRT_GENERATION
        video.error_details = f"Error generating SRT file: {str(e)}"
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
                    duration = segment.duration()
                    output_file = temp_path / f"clip_{i:03d}.mp4"

                    logger.info(
                        f"Processing clip {i+1}/{segments.count()}: {duration:.2f}s from {segment.video_file.name}"
                    )

                    # Cut video to exact duration (no audio, we'll add it later)
                    # Scale to consistent resolution (1080x1920 for portrait)
                    cmd = [
                        "ffmpeg",
                        "-i",
                        input_file,  # Input video file (segment video)
                        "-t",
                        str(duration),  # Set output duration (cut to segment length)
                        "-vf",  # Video filter
                        # scale: Resize video to 1080x1920, maintain aspect ratio (decrease if needed)
                        # pad: Add black bars if video is smaller than 1080x1920
                        # (ow-iw)/2: Center horizontally, (oh-ih)/2: Center vertically
                        # setsar=1: Set Sample Aspect Ratio to 1:1 (square pixels)
                        "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1",
                        "-c:v",
                        "libx264",  # Video codec: H.264
                        "-preset",
                        "medium",  # Encoding speed preset (balance speed/quality)
                        "-crf",
                        "23",  # Constant Rate Factor: quality level (lower=better, 18-28 range)
                        "-r",
                        "30",  # Force constant frame rate: 30 fps (consistent across all clips)
                        "-g",
                        "30",  # GOP size (keyframe interval): 1 keyframe per second at 30fps
                        "-pix_fmt",
                        "yuv420p",  # Pixel format: ensures compatibility
                        "-an",  # Remove audio (we'll add voice later)
                        "-y",  # Overwrite output file without asking
                        str(output_file),  # Output file path
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
