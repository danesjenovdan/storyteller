from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.utils.translation import gettext_lazy as _

User = get_user_model()


class SignUpForm(UserCreationForm):
    email = forms.EmailField(
        required=True,
        label=_("E-pošta"),
        widget=forms.EmailInput(
            attrs={"class": "form-control", "placeholder": _("E-pošta")}
        ),
    )

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ("username", "email")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["username"].widget.attrs.update(
            {"class": "form-control", "placeholder": _("Uporabniško ime")}
        )
        self.fields["password1"].widget.attrs.update(
            {"class": "form-control", "placeholder": _("Geslo")}
        )
        self.fields["password2"].widget.attrs.update(
            {"class": "form-control", "placeholder": _("Ponovite geslo")}
        )

    def clean_email(self):
        email = self.cleaned_data["email"]
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError(
                _("Uporabnik s tem e-poštnim naslovom že obstaja.")
            )
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data["email"]
        if commit:
            user.save()
        return user


class LoginForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["username"].widget.attrs.update(
            {
                "class": "form-control",
                "placeholder": _("Uporabniško ime"),
                "autofocus": True,
            }
        )
        self.fields["password"].widget.attrs.update(
            {"class": "form-control", "placeholder": _("Geslo")}
        )
