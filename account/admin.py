from django.contrib import admin
from django.utils.html import format_html
from django.utils.translation import gettext_lazy as _

from .models import RegistrationLink, Settings, UserPaidApiUsage


@admin.register(RegistrationLink)
class RegistrationLinkAdmin(admin.ModelAdmin):
    list_display = (
        "uuid",
        "author",
        "user",
        "used_at",
        "created",
        "registration_link",
    )
    readonly_fields = (
        "uuid",
        "author",
        "user",
        "used_at",
        "created",
        "modified",
        "registration_link",
    )
    fields = (
        "registration_link",
        "uuid",
        "author",
        "user",
        "used_at",
        "created",
        "modified",
    )

    def save_model(self, request, obj, form, change):
        if not change:
            obj.author = request.user
        super().save_model(request, obj, form, change)

    def registration_link(self, obj):
        if not obj.pk:
            return "-"
        url = obj.get_absolute_url()
        return format_html('<a href="{0}" target="_blank">{0}</a>', url)

    registration_link.short_description = _("Registracijska povezava")


@admin.register(Settings)
class SettingsAdmin(admin.ModelAdmin):
    list_display = (
        "user",
        "paid_api_limit",
        "created",
        "modified",
    )
    readonly_fields = (
        "created",
        "modified",
    )


@admin.register(UserPaidApiUsage)
class UserPaidApiUsageAdmin(admin.ModelAdmin):
    list_display = (
        "user",
        "video",
        "api",
        "usage_type",
        "usage_amount",
        "created",
    )
    readonly_fields = (
        "user",
        "video",
        "api",
        "usage_type",
        "usage_amount",
        "created",
    )
