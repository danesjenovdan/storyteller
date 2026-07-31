"""
Final video rendering logic for GenVideo instances.

Encapsulates all ffmpeg command construction and execution needed to combine
VideoSegment clips into a single final video with audio, subtitles and an
optional logo overlay.
"""

import json
import logging
import subprocess
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Optional

from django.conf import settings
from django.core.files.base import ContentFile

from agent.models import GenVideo
from agent.utils import get_temporary_file, get_temporary_file_path

logger = logging.getLogger(__name__)


class FFmpegTimeoutError(RuntimeError):
    """Raised when an ffmpeg invocation exceeds its allotted timeout."""


class FinalVideoRenderer:
    """
    Renders the final video for a single GenVideo instance.

    Usage:
        FinalVideoRenderer(video).render()

    Does not catch or translate exceptions into GenVideo.status/error_type -
    that is the responsibility of the caller (see agent.tasks.render_final_video),
    so this class can be reused/tested independently of that error-reporting
    policy.
    """

    ENCODING_ARGS = [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-r",
        "60",
        "-g",
        "60",
        "-pix_fmt",
        "yuv420p",
    ]

    # Maximum allowed drift (seconds) between the concatenated video's actual
    # duration and voice_duration before we bother rescaling the video
    # timeline to match.
    DURATION_SYNC_THRESHOLD_SECONDS = 0.05

    def __init__(self, video: GenVideo):
        self.video = video

    def render(self) -> None:
        """
        Combine all VideoSegment clips with ffmpeg, add audio and subtitles.

        Process:
        1. Download each video/image from URL stored in video_proposals
        2. Cut/convert each clip according to segment start_time/end_time
        3. Concatenate all clips in order
        4. Add voice_file as audio track
        5. Burn-in subtitles from srt_file and optionally overlay a logo
        6. Save to GenVideo.final_file
        """
        self.video.status = GenVideo.Statuses.RENDERING
        self.video.progress = "Initializing rendering process"
        self.video.save()

        segments = self._get_selected_segments()

        logger.info(
            f"Starting render for video {self.video} with {segments.count()} clips"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            clip_files = self._prepare_clips(segments, temp_path)
            concatenated_file = self._concatenate_clips(clip_files, temp_path)
            concatenated_file = self._sync_video_duration(concatenated_file, temp_path)
            final_output = self._compose_final_output(concatenated_file, temp_path)
            self._save_final_file(final_output)

        logger.info(f"Video {self.video} rendered successfully!")

    # ------------------------------------------------------------------
    # Segment validation & clip preparation
    # ------------------------------------------------------------------

    def _get_selected_segments(self):
        segments = self.video.segments.filter(
            video_proposals__0__selected=True
        ).order_by("order")

        if not segments.exists():
            raise ValueError(f"Video {self.video} has no segments with selected videos")

        if not self.video.voice_file:
            raise ValueError(f"Video {self.video} has no voice file")

        return segments

    def _prepare_clips(self, segments, temp_path: Path) -> list[Path]:
        total = segments.count()
        return [
            self._prepare_clip(i, total, segment, temp_path)
            for i, segment in enumerate(segments)
        ]

    def _prepare_clip(self, index: int, total: int, segment, temp_path: Path) -> Path:
        proposal = segment.video_proposals[0]
        video_url = proposal.get("video_url")
        if not video_url:
            raise ValueError(f"Segment {segment.id} has no video URL")

        is_image = proposal.get("is_image", False)
        width = proposal.get("width")
        height = proposal.get("height")

        if width is None or height is None:
            logger.warning(
                f"Segment {segment.id} missing dimensions (width={width}, height={height}), "
                "assuming vertical video"
            )
            width, height = 1080, 1920

        horizontal_mode = proposal.get("horizontal_mode", "crop")
        animation_mode = {
            "in": proposal.get("animation_in", "none"),
            "mid": proposal.get("animation_mid", proposal.get("animation", "none")),
            "out": proposal.get("animation_out", "none"),
        }

        output_file = temp_path / f"clip_{index:03d}.mp4"
        duration = segment.duration()

        logger.info(
            f"Processing clip {index+1}/{total}: {duration:.2f}s from URL "
            f"(dimensions: {width}x{height}, is_image: {is_image}, mode: {horizontal_mode}, "
            f"animation_in: {animation_mode['in']}, animation_mid: {animation_mode['mid']}, "
            f"animation_out: {animation_mode['out']})"
        )
        self.video.progress = f"Processing clip {index+1}/{total}: {duration:.2f}s"
        self.video.save()

        with get_temporary_file(video_url) as input_file:
            cmd = self._build_clip_command(
                input_file,
                output_file,
                width,
                height,
                duration,
                is_image,
                horizontal_mode,
                animation_mode,
            )
            self._run_ffmpeg(
                cmd, error_context=f"clip {index}" + (" (image)" if is_image else "")
            )

        logger.info(f"Successfully created clip {index}: {output_file}")
        return output_file

    # ------------------------------------------------------------------
    # Per-clip ffmpeg command construction
    # ------------------------------------------------------------------

    def _build_clip_command(
        self,
        input_file: str,
        output_file: Path,
        width: int,
        height: int,
        duration: float,
        is_image: bool,
        horizontal_mode: str,
        animation_mode: dict,
    ) -> list[str]:
        is_horizontal = width > height

        if is_horizontal:
            vf_filter, filter_complex = self._horizontal_filter(horizontal_mode)
        else:
            vf_filter, filter_complex = self._vertical_filter(), None

        cmd = ["ffmpeg"]
        if is_image:
            cmd.extend(["-loop", "1"])
        cmd.extend(["-i", input_file, "-t", str(duration)])

        if filter_complex:
            cmd.extend(
                [
                    "-filter_complex",
                    self._with_animation(filter_complex, animation_mode, duration),
                ]
            )
        else:
            cmd.extend(
                ["-vf", self._with_animation(vf_filter, animation_mode, duration)]
            )

        cmd.extend(self.ENCODING_ARGS)
        cmd.extend(["-an", "-y", str(output_file)])
        return cmd

    def _horizontal_filter(
        self, horizontal_mode: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Returns (vf_filter, filter_complex) for a horizontal source; only one is set."""
        if horizontal_mode == "blur":
            return None, (
                "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:5[blurred];"
                "[0:v]scale=1080:-1:force_original_aspect_ratio=decrease[main];"
                "[blurred][main]overlay=(W-w)/2:(H-h)/2,setsar=1"
            )
        if horizontal_mode == "blur_crop":
            return None, (
                "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:5[blurred];"
                "[0:v]crop=ih:ih:(iw-ih)/2:0,scale=1080:-1:force_original_aspect_ratio=decrease[main];"
                "[blurred][main]overlay=(W-w)/2:(H-h)/2,setsar=1"
            )
        # "crop" and any unrecognized mode default to a centered 9:16 crop.
        return (
            "crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale=1080:1920:force_original_aspect_ratio=decrease,"
            "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1"
        ), None

    def _vertical_filter(self) -> str:
        return (
            "scale=1080:1920:force_original_aspect_ratio=decrease,"
            "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1"
        )

    # ------------------------------------------------------------------
    # Animation filters
    # ------------------------------------------------------------------

    @staticmethod
    def _animation_entry_filter(animation: str, clip_duration: float) -> Optional[str]:
        safe_duration = max(float(clip_duration or 0.0), 0.1)
        animation = (animation or "none").strip().lower()
        entry_duration = min(0.6, safe_duration / 2)

        if animation == "fade":
            return f"fade=t=in:st=0:d={entry_duration:.3f}"
        return None

    @staticmethod
    def _animation_mid_filter(animation: str, clip_duration: float) -> Optional[str]:
        safe_duration = max(float(clip_duration or 0.0), 0.1)
        animation = (animation or "none").strip().lower()

        if animation == "zoom_in":
            return (
                f"scale=iw*(1+0.10*t/{safe_duration:.3f}):"
                f"ih*(1+0.10*t/{safe_duration:.3f}):eval=frame,"
                "crop=1080:1920:(iw-1080)/2:(ih-1920)/2"
            )
        if animation == "zoom_out":
            return (
                f"scale=iw*(1.20-0.10*t/{safe_duration:.3f}):"
                f"ih*(1.20-0.10*t/{safe_duration:.3f}):eval=frame,"
                "crop=1080:1920:(iw-1080)/2:(ih-1920)/2"
            )
        if animation == "subtle_pan_lr":
            return (
                # Uniform overscan gives enough travel distance to avoid visible step movement.
                "scale=iw*1.30:ih*1.30:eval=frame,"
                f"crop=1080:1920:'(iw-1080)*min(t/{safe_duration:.3f}\\,1)':(ih-1920)/2"
            )
        if animation == "subtle_pan_ud":
            return (
                # Uniform overscan gives enough travel distance to avoid visible step movement.
                "scale=iw*1.30:ih*1.30:eval=frame,"
                f"crop=1080:1920:(iw-1080)/2:'(ih-1920)*min(t/{safe_duration:.3f}\\,1)'"
            )
        return None

    @staticmethod
    def _animation_exit_filter(animation: str, clip_duration: float) -> Optional[str]:
        safe_duration = max(float(clip_duration or 0.0), 0.1)
        animation = (animation or "none").strip().lower()
        exit_duration = min(0.6, safe_duration / 2)
        fade_out_start = max(0.0, safe_duration - exit_duration)

        if animation == "fade":
            return f"fade=t=out:st={fade_out_start:.3f}:d={exit_duration:.3f}"
        return None

    @classmethod
    def _segment_animation_filter(cls, animation, clip_duration: float) -> Optional[str]:
        if isinstance(animation, dict):
            animation_in = animation.get("in", "none")
            animation_mid = animation.get("mid", "none")
            animation_out = animation.get("out", "none")
        else:
            animation_in = "none"
            animation_mid = (animation or "none").strip().lower()
            animation_out = "none"

        filters = [
            filter_value
            for filter_value in (
                cls._animation_entry_filter(animation_in, clip_duration),
                cls._animation_mid_filter(animation_mid, clip_duration),
                cls._animation_exit_filter(animation_out, clip_duration),
            )
            if filter_value
        ]

        return ",".join(filters) if filters else None

    @classmethod
    def _with_animation(cls, base_filter: str, animation, clip_duration: float) -> str:
        animation_filter = cls._segment_animation_filter(animation, clip_duration)
        if not animation_filter:
            return base_filter
        return f"{base_filter},{animation_filter}"

    # ------------------------------------------------------------------
    # Concatenation
    # ------------------------------------------------------------------

    def _concatenate_clips(self, clip_files: list[Path], temp_path: Path) -> Path:
        concat_file = temp_path / "concat.txt"
        with open(concat_file, "w") as f:
            for clip in clip_files:
                f.write(f"file '{clip}'\n")

        concatenated_file = temp_path / "concatenated.mp4"
        logger.info("Concatenating clips...")
        self.video.progress = "Concatenating clips."
        self.video.save()

        cmd = [
            "ffmpeg",
            "-f",
            "concat",  # Use concat demuxer (reads from concat.txt file)
            "-safe",
            "0",  # Allow absolute file paths in concat.txt
            "-i",
            str(concat_file),  # Input: concat.txt with list of video files
            "-fflags",
            "+genpts",  # Regenerate continuous timestamps across clip boundaries
            "-vsync",
            "cfr",  # Normalize to constant frame rate to avoid 1-frame drops
            "-r",
            "60",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-y",  # Overwrite output file without asking
            str(concatenated_file),  # Output: single concatenated video
        ]
        self._run_ffmpeg(cmd, error_context="concatenation")
        return concatenated_file

    """
    Tole probi tud izklopit
    """
    def _sync_video_duration(self, concatenated_file: Path, temp_path: Path) -> Path:
        """
        Rescale the concatenated (silent) video so its duration exactly matches
        voice_duration.

        Each clip is cut independently with "-t duration" (see
        _build_clip_command); ffmpeg can't output a partial frame, so this
        tends to yield a slightly *shorter* clip than requested. These small,
        one-directional rounding errors accumulate across segments, so on
        longer videos (more segments) the concatenated video timeline can end
        up noticeably shorter than the voice track. Since subtitles are
        burned onto this video's own timeline while their timestamps were
        computed against the voice track, that drift shows up as subtitles
        progressively lagging behind the narration - worse the longer the
        video is.

        We fix this by measuring the actual concatenated duration and
        rescaling it (via setpts) to exactly match voice_duration before
        subtitles are burned in.
        """
        target_duration = self.video.voice_duration
        if not target_duration:
            return concatenated_file

        actual_duration = self._probe_duration(concatenated_file)
        if actual_duration <= 0:
            return concatenated_file

        drift = target_duration - actual_duration
        if abs(drift) < self.DURATION_SYNC_THRESHOLD_SECONDS:
            return concatenated_file

        logger.info(
            f"Concatenated video duration ({actual_duration:.3f}s) drifted from "
            f"voice duration ({target_duration:.3f}s) by {drift:+.3f}s for video "
            f"{self.video.id}; rescaling video timeline to match."
        )
        self.video.progress = "Synchronizing video and audio timing."
        self.video.save()

        speed_factor = target_duration / actual_duration
        synced_file = temp_path / "concatenated_synced.mp4"
        cmd = [
            "ffmpeg",
            "-i",
            str(concatenated_file),
            "-vf",
            f"setpts={speed_factor:.6f}*PTS",
            "-r",
            "60",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-y",
            str(synced_file),
        ]
        self._run_ffmpeg(cmd, error_context="duration sync")
        return synced_file

    def _probe_duration(self, file_path: Path) -> float:
        """Return the media duration (seconds) of a local file via ffprobe."""
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            str(file_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])

    # ------------------------------------------------------------------
    # Final composition (audio + subtitles + logo)
    # ------------------------------------------------------------------

    def _compose_final_output(self, concatenated_file: Path, temp_path: Path) -> Path:
        final_output = temp_path / "final.mp4"
        logger.info(f"Adding audio and subtitles for video {self.video.id}...")

        with ExitStack() as stack:
            logger.info(
                f"Preparing temporary media files for final render (video {self.video.id})"
            )
            voice_file = self._prepare_voice_file(stack)
            srt_file = self._prepare_subtitle_file(stack)
            logo_file = self._prepare_logo_file(stack)

            cmd = [
                "ffmpeg",
                "-i",
                str(concatenated_file),  # Input 1: Concatenated video (no audio)
                "-i",
                voice_file,  # Input 2: Voice audio file (narration)
            ]

            self.video.progress = "Preparing final video with audio and subtitles."
            self.video.save()

            use_logo_overlay = bool(logo_file)
            if use_logo_overlay:
                # Loop logo image so overlay is available for entire output timeline.
                cmd.extend(["-stream_loop", "-1", "-i", logo_file])  # Input 3

            subtitle_filter = (
                self._build_subtitle_filter(srt_file) if srt_file else None
            )
            cmd.extend(self._build_output_args(subtitle_filter, use_logo_overlay))
            cmd.extend(
                ["-c:a", "aac", "-b:a", "192k", "-shortest", "-y", str(final_output)]
            )

            ffmpeg_timeout = int(
                getattr(settings, "FFMPEG_FINAL_RENDER_TIMEOUT_SECONDS", 300)
            )
            logger.info(
                f"Running final FFmpeg command for video {self.video.id} (timeout={ffmpeg_timeout}s)"
            )
            logger.debug(f"Final FFmpeg command: {' '.join(cmd)}")

            self._run_ffmpeg(cmd, timeout=ffmpeg_timeout, error_context="final render")

        return final_output

    def _prepare_voice_file(self, stack: ExitStack) -> str:
        logger.info(f"Preparing voice file: {self.video.voice_file.name}")
        voice_file = stack.enter_context(get_temporary_file_path(self.video.voice_file))
        logger.info(f"Voice file ready: {voice_file}")
        return voice_file

    def _prepare_subtitle_file(self, stack: ExitStack) -> Optional[str]:
        if not self.video.srt_file:
            return None
        logger.info(f"Preparing subtitle file: {self.video.srt_file.name}")
        srt_file = stack.enter_context(get_temporary_file_path(self.video.srt_file))
        logger.info(f"Subtitle file ready: {srt_file}")
        return srt_file

    def _prepare_logo_file(self, stack: ExitStack) -> Optional[str]:
        if not (self.video.logo and self.video.logo.logo_file):
            return None
        try:
            logger.info(f"Preparing logo file: {self.video.logo.logo_file.name}")
            logo_file = stack.enter_context(
                get_temporary_file_path(self.video.logo.logo_file)
            )
            logger.info(f"Logo file ready: {logo_file}")
            return logo_file
        except Exception as logo_error:
            logger.warning(
                f"Could not load logo for video {self.video.id}, rendering without logo: {logo_error}"
            )
            return None

    def _build_subtitle_filter(self, srt_file: str) -> str:
        video = self.video
        font_size = video.subtitle_font_size or 12
        font_family = video.subtitle_font_family or "Montserrat"
        font_weight = video.subtitle_font_weight or "900"
        stroke_weight = video.subtitle_stroke_weight or 3
        shadow = video.subtitle_shadow or 1
        vertical_position = video.subtitle_vertical_position or 10

        bold = 1 if int(font_weight) >= 700 else 0
        max_margin_v = 300
        margin_v = int((vertical_position / 100) * max_margin_v)
        style = (
            f"FontName={font_family},FontSize={font_size},Bold={bold},"
            "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            f"Outline={stroke_weight},Shadow={shadow},MarginV={margin_v}"
        )
        escaped_srt = self._escape_ffmpeg_filter_path(srt_file)
        return f"subtitles='{escaped_srt}':force_style='{style}'"

    def _build_logo_overlay_filter(self, video_label: str) -> str:
        video = self.video
        logo_position = getattr(video, "logo_position", "top_right")
        logo_size_percent = max(
            5, min(40, int(getattr(video, "logo_size_percent", 15) or 15))
        )
        logo_width = int(1080 * (logo_size_percent / 100.0))
        logo_x = "24" if logo_position == "top_left" else "main_w-overlay_w-24"
        logo_y = "24"

        return (
            f"[2:v]scale={logo_width}:-1[logo];"
            f"[{video_label}][logo]overlay={logo_x}:{logo_y}:format=auto:eof_action=repeat:shortest=0[vout]"
        )

    def _build_output_args(
        self, subtitle_filter: Optional[str], use_logo_overlay: bool
    ) -> list[str]:
        video_encode_args = ["-c:v", "libx264", "-preset", "medium", "-crf", "23"]

        if subtitle_filter and use_logo_overlay:
            filter_complex = f"[0:v]{subtitle_filter}[vsub];" + (
                self._build_logo_overlay_filter("vsub")
            )
            return [
                "-filter_complex",
                filter_complex,
                "-map",
                "[vout]",
                "-map",
                "1:a:0",
                *video_encode_args,
            ]

        if subtitle_filter:
            return [
                "-vf",
                subtitle_filter,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                *video_encode_args,
            ]

        if use_logo_overlay:
            filter_complex = self._build_logo_overlay_filter("0:v")
            return [
                "-filter_complex",
                filter_complex,
                "-map",
                "[vout]",
                "-map",
                "1:a:0",
                *video_encode_args,
            ]

        # No subtitles/logo - keep original fast path.
        return ["-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0"]

    @staticmethod
    def _escape_ffmpeg_filter_path(path_value: str) -> str:
        return (
            path_value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        )

    # ------------------------------------------------------------------
    # Shared ffmpeg execution
    # ------------------------------------------------------------------

    def _run_ffmpeg(
        self,
        cmd: list[str],
        *,
        timeout: Optional[int] = None,
        error_context: str = "ffmpeg",
    ) -> subprocess.CompletedProcess:
        """
        Run an ffmpeg command, logging its invocation/duration and translating
        failures into exceptions with the relevant stderr context attached.
        """
        logger.info(f"Running FFmpeg command ({error_context}): {' '.join(cmd)}")
        start_time = time.monotonic()
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired as timeout_error:
            elapsed = time.monotonic() - start_time
            stderr_preview = (timeout_error.stderr or "")[-2000:]
            logger.error(
                f"FFmpeg command timed out after {elapsed:.1f}s ({error_context}) "
                f"for video {self.video.id}"
            )
            if stderr_preview:
                logger.error(f"FFmpeg stderr tail before timeout:\n{stderr_preview}")
            raise FFmpegTimeoutError(
                f"FFmpeg {error_context} timed out after {elapsed:.1f}s"
            ) from timeout_error

        elapsed = time.monotonic() - start_time
        logger.info(f"FFmpeg command finished in {elapsed:.1f}s ({error_context})")
        if result.returncode != 0:
            stderr_tail = (result.stderr or "")[-4000:]
            logger.error(
                f"FFmpeg {error_context} failed for video {self.video.id}. "
                f"stderr tail:\n{stderr_tail}"
            )
            raise RuntimeError(f"FFmpeg {error_context} failed: {result.stderr}")
        return result

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _save_final_file(self, final_output: Path) -> None:
        logger.info("Saving final video to database...")
        with open(final_output, "rb") as f:
            filename = f"final_video_{self.video.id}.mp4"
            self.video.final_file.save(filename, ContentFile(f.read()), save=False)

        self.video.status = GenVideo.Statuses.COMPLETED
        self.video.progress = ""
        self.video.save()
