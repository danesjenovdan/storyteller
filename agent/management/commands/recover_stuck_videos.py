from datetime import timedelta

from django.core.management.base import BaseCommand
from django.conf import settings

from agent.recovery import recover_stuck_videos


class Command(BaseCommand):
    help = "Requeue stale, interrupted video processing tasks."

    def add_arguments(self, parser):
        parser.add_argument("--stale-minutes", type=int)
        parser.add_argument("--max-attempts", type=int)

    def handle(self, *args, **options):
        if not settings.HUEY_RECOVERY_ENABLED:
            self.stdout.write("Huey recovery is disabled.")
            return
        stale_after = (
            timedelta(minutes=options["stale_minutes"])
            if options["stale_minutes"] is not None
            else None
        )
        recovered = recover_stuck_videos(
            stale_after=stale_after,
            max_attempts=options["max_attempts"],
        )
        self.stdout.write(self.style.SUCCESS(f"Requeued {recovered} interrupted video(s)."))