import uuid as uuid_lib

from behaviors.behaviors import Authored, Timestamped
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
