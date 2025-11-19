from django import forms

from .models import GenVideo


class VideoCreateForm(forms.ModelForm):
    """Form for creating a new video with title, scenario, and prompt."""

    class Meta:
        model = GenVideo
        fields = ["title", "start_prompt", "scenario", "prompt"]
        widgets = {
            "title": forms.TextInput(
                attrs={"class": "form-control", "placeholder": "Vnesite naslov videa"}
            ),
            "start_prompt": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "rows": 5,
                    "placeholder": "Vpiši začetni ukaz LLM-ju za da ti generira scenarij videa",
                }
            ),
            "scenario": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "rows": 5,
                    "placeholder": "Opišite scenarij videa",
                }
            ),
            "prompt": forms.Textarea(
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
            "prompt": "Prompt",
        }


class ScenarioEditForm(forms.ModelForm):
    """Form for editing the scenario after it's generated from start_prompt."""

    class Meta:
        model = GenVideo
        fields = ["scenario", "prompt"]
        widgets = {
            "scenario": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "rows": 10,
                    "placeholder": "Uredite scenarij videa",
                }
            ),
            "prompt": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "rows": 6,
                    "placeholder": "Prompt za simplifikacijo scenarija",
                }
            ),
        }
        labels = {
            "scenario": "Scenarij",
            "prompt": "Prompt za simplifikacijo",
        }


class ContentScriptEditForm(forms.ModelForm):
    """Form for editing the content_script after LLM generation."""

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
        fields = ["content_script", "voice_model"]
        widgets = {
            "content_script": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "rows": 15,
                    "placeholder": "Uredite vsebinski skript",
                }
            ),
        }
        labels = {
            "content_script": "Vsebinski skript",
        }

    def __init__(self, *args, voice_models=None, **kwargs):
        super().__init__(*args, **kwargs)
        if voice_models:
            self.fields["voice_model"].choices = voice_models
