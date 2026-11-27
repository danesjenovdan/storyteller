import uuid as uuid_lib

from behaviors.behaviors import Authored, Timestamped
from django.conf import settings
from django.db import models
from django.urls import reverse


class RegistrationLink(Timestamped, Authored):
    uuid = models.UUIDField(unique=True, default=uuid_lib.uuid4, editable=False)
    used_at = models.DateTimeField(null=True, blank=True)
    user = models.ForeignKey(
        "auth.User",
        on_delete=models.SET_NULL,
        related_name="registration_links",
        null=True,
    )

    def __str__(self):
        return str(self.uuid)

    def get_absolute_url(self):
        return f"{reverse('registration')}?uuid={self.uuid}"


class UserPaidApiUsage(Timestamped):
    user = models.ForeignKey(
        "auth.User",
        on_delete=models.CASCADE,
        related_name="api_usages",
    )
    video = models.ForeignKey(
        "agent.GenVideo",
        on_delete=models.CASCADE,
        related_name="api_usages",
    )
    api = models.CharField(max_length=255)
    usage_type = models.CharField(max_length=50)
    usage_amount = models.PositiveIntegerField(default=0)

    success = models.BooleanField(null=True, blank=True)

    def __str__(self):
        return f"{self.user} - {self.api}"


class Settings(Timestamped):
    user = models.OneToOneField(
        "auth.User",
        on_delete=models.CASCADE,
    )
    paid_api_limit = models.PositiveIntegerField(
        default=settings.DEFAULT_PAID_API_LIMIT
    )

    def __str__(self):
        return f"Settings for {self.user}"
