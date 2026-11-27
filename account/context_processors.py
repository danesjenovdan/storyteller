from datetime import datetime

from account.models import Settings, UserPaidApiUsage


def users_quota(request):
    print("Request user:", request.user)
    if request.user.is_authenticated:
        settings = getattr(request.user, "settings", None)
        if not settings:
            settings = Settings.objects.create(user=request.user)

        used = UserPaidApiUsage.objects.filter(
            user=request.user,
            created__date=datetime.now().date(),
        ).count()
        return {
            "users_quota": settings.paid_api_limit,
            "users_quota_used": used,
        }
    return {
        "users_quota": 3,
        "users_quota_used": 7,
    }
