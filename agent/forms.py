from django import forms
from django.utils.translation import gettext_lazy as _

from .models import GenVideo


class VideoCreateForm(forms.ModelForm):
    """Form for creating a new video with title, scenario, and prompt."""

    voice_model = forms.ChoiceField(
        choices=[],
        required=False,
        widget=forms.Select(
            attrs={
                "class": "form-select",
            }
        ),
        label=_("Glasovni model"),
    )

    class Meta:
        model = GenVideo
        fields = [
            "title",
            "scenario",
            "modify_prompt",
            "voice_model",
            "language",
            "voice_file",
        ]
        widgets = {
            "title": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": _("Vnesite naslov videa"),
                }
            ),
            "scenario": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "rows": 5,
                    "placeholder": _("Opišite scenarij videa"),
                    "required": False,
                }
            ),
            "modify_prompt": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "rows": 8,
                    "placeholder": _("Vnesite prompt za LLM"),
                }
            ),
            "language": forms.Select(
                attrs={
                    "class": "form-select",
                }
            ),
        }
        labels = {
            "title": _("Naslov"),
            "scenario": _("Scenarij"),
            "modify_prompt": _("Navodila za LLM"),
            "voice_model": _("Glasovni model"),
            "language": _("Jezik"),
            "voice_file": _("Naložite glasovno datoteko"),
        }

    def __init__(self, *args, voice_models=None, **kwargs):
        super().__init__(*args, **kwargs)
        if voice_models:
            self.fields["voice_model"].choices = voice_models
