from django import forms

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
        label="Glasovni model",
    )

    class Meta:
        model = GenVideo
        fields = ["title", "scenario", "modify_prompt", "voice_model"]
        widgets = {
            "title": forms.TextInput(
                attrs={"class": "form-control", "placeholder": "Vnesite naslov videa"}
            ),
            "scenario": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "rows": 5,
                    "placeholder": "Opišite scenarij videa",
                }
            ),
            "modify_prompt": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "rows": 8,
                    "placeholder": "Vnesite prompt za LLM",
                }
            ),
        }
        labels = {
            "title": "Naslov",
            "scenario": "Scenarij",
            "modify_prompt": "Navodila za LLM",
            "voice_model": "Glasovni model",
        }

    def __init__(self, *args, voice_models=None, **kwargs):
        super().__init__(*args, **kwargs)
        if voice_models:
            self.fields["voice_model"].choices = voice_models
