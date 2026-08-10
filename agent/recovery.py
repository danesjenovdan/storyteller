import logging
from datetime import timedelta

from django.conf import settings
from django.db.models import F, Q
from django.utils import timezone

from agent.models import GenVideo

logger = logging.getLogger(__name__)

IN_PROGRESS_STATUSES = (
    GenVideo.Statuses.GENERATING_VOICE,
    GenVideo.Statuses.GENERATING_SUBTITLES,
    GenVideo.Statuses.GENERATING_SEGMENTS,
    GenVideo.Statuses.RENDERING,
)


def reset_recovery_state(video_id: int) -> None:
    GenVideo.objects.filter(pk=video_id).update(
        recovery_attempts=0,
        last_recovery_at=None,
        recovery_claimed_at=None,
    )


def enqueue_next_step(video: GenVideo) -> None:
    """Enqueue the next idempotent pipeline step from persisted video state."""
    from agent.tasks import (
        continue_elevenlabs_pipeline,
        generate_srt_file,
        generate_voice_file_eleven_labs,
        generate_voice_file_gemini,
        generate_voice_file_openai,
        get_video_segments,
        regenerate_elevenlabs_srt_file,
        render_final_video,
    )

    if video.status == GenVideo.Statuses.GENERATING_VOICE:
        if video.voice_file:
            if video.elevenlabs_alignment:
                continue_elevenlabs_pipeline(video.id)
            else:
                GenVideo.objects.filter(pk=video.pk).update(
                    status=GenVideo.Statuses.VOICE_READY,
                    recovery_claimed_at=None,
                )
            return

        provider = getattr(settings, "TTS_PROVIDER", "elevenlabs")
        if provider == "elevenlabs":
            generate_voice_file_eleven_labs(video.id)
        elif provider == "gemini":
            generate_voice_file_gemini(video.id)
        else:
            generate_voice_file_openai(video.id)
        return

    if video.status == GenVideo.Statuses.GENERATING_SUBTITLES:
        if video.srt_file:
            get_video_segments(video.id)
        elif video.elevenlabs_alignment:
            continue_elevenlabs_pipeline(video.id)
        else:
            generate_srt_file(video.id)
        return

    if video.status == GenVideo.Statuses.GENERATING_SEGMENTS:
        get_video_segments(video.id)
        return

    if video.status == GenVideo.Statuses.RENDERING:
        render_final_video(video.id)


def recover_stuck_videos(
    stale_after: timedelta | None = None,
    max_attempts: int | None = None,
) -> int:
    """Claim and requeue stale in-progress videos exactly once per recovery lease."""
    if stale_after is None:
        stale_after = timedelta(
            minutes=getattr(settings, "HUEY_RECOVERY_STALE_MINUTES", 10)
        )
    if max_attempts is None:
        max_attempts = getattr(settings, "HUEY_RECOVERY_MAX_ATTEMPTS", 3)
    now = timezone.now()
    stale_before = now - stale_after
    claim_expires_before = now - stale_after
    stale_videos = GenVideo.objects.filter(
        status__in=IN_PROGRESS_STATUSES,
        updated_at__lt=stale_before,
    ).filter(
        Q(recovery_claimed_at__isnull=True)
        | Q(recovery_claimed_at__lt=claim_expires_before)
    )

    recovered = 0
    for video in stale_videos.iterator():
        if video.recovery_attempts >= max_attempts:
            GenVideo.objects.filter(
                pk=video.pk,
                status=video.status,
                recovery_attempts__gte=max_attempts,
            ).filter(
                Q(recovery_claimed_at__isnull=True)
                | Q(recovery_claimed_at__lt=claim_expires_before)
            ).update(
                status=GenVideo.Statuses.FAILED,
                error_type=GenVideo.ErrorTypes.RECOVERY,
                error_details="Automatic recovery limit reached after interrupted processing.",
                recovery_claimed_at=None,
            )
            continue

        claimed = GenVideo.objects.filter(
            pk=video.pk,
            status=video.status,
            recovery_attempts__lt=max_attempts,
        ).filter(
            Q(recovery_claimed_at__isnull=True)
            | Q(recovery_claimed_at__lt=claim_expires_before)
        ).update(
            recovery_attempts=F("recovery_attempts") + 1,
            last_recovery_at=now,
            recovery_claimed_at=now,
        )
        if not claimed:
            continue

        video.refresh_from_db()
        try:
            enqueue_next_step(video)
        except Exception:
            GenVideo.objects.filter(pk=video.pk).update(recovery_claimed_at=None)
            logger.exception("Could not requeue interrupted video %s", video.id)
            continue

        recovered += 1
        logger.warning(
            "Requeued interrupted video %s from %s (attempt %s/%s)",
            video.id,
            video.status,
            video.recovery_attempts,
            max_attempts,
        )

    return recovered