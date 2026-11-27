from django.contrib import messages
from django.contrib.auth.views import LoginView, LogoutView
from django.db import transaction
from django.shortcuts import redirect, render
from django.urls import reverse_lazy
from django.utils import timezone
from django.utils.translation import gettext as _

from .forms import LoginForm, SignUpForm
from .models import RegistrationLink


def registration(request):
    uuid_value = request.GET.get("uuid") or request.POST.get("uuid")
    registration_link = None
    if uuid_value:
        registration_link = RegistrationLink.objects.filter(
            uuid=uuid_value, used_at__isnull=True
        ).first()

    if not registration_link:
        messages.error(
            request,
            _("Registracijska povezava ni veljavna ali je bila že uporabljena."),
        )
        return render(request, "account/signup.html", {"form": None})

    form = None
    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            with transaction.atomic():
                user = form.save()
                registration_link.user = user
                registration_link.used_at = timezone.now()
                registration_link.save(update_fields=["user", "used_at", "modified"])
            messages.success(
                request,
                _("Račun je bil uspešno ustvarjen. Prijavite se s svojimi podatki."),
            )
            return redirect("login")
    else:
        form = SignUpForm()

    return render(
        request,
        "account/signup.html",
        {"form": form, "uuid": uuid_value},
    )


class AccountLoginView(LoginView):
    template_name = "account/login.html"
    form_class = LoginForm
    redirect_authenticated_user = True


class AccountLogoutView(LogoutView):
    next_page = reverse_lazy("login")
