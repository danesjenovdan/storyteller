import json
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from .content_apis.pexels import _pick_preferred_video_file
from .content_apis.pixabay import _pick_video_variant
from .models import GenVideo, VideoSegment
from .tasks import generate_srt_from_elevenlabs_alignment


class ElevenLabsTimestampResponse:
    audio_base_64 = "dGVzdC1hdWRpbw=="


class ElevenLabsTimestampResponseTests(SimpleTestCase):
    def test_uses_sdk_audio_base_64_field(self) -> None:
        from base64 import b64decode

        response = ElevenLabsTimestampResponse()

        self.assertEqual(b64decode(response.audio_base_64), b"test-audio")


class ElevenLabsAlignment:
    def __init__(self, characters, starts, ends) -> None:
        self.characters = characters
        self.character_start_times_seconds = starts
        self.character_end_times_seconds = ends


class ElevenLabsSubtitleGenerationTests(SimpleTestCase):
    def test_generates_srt_from_saved_alignment_dictionary(self) -> None:
        alignment = {
            "characters": list("One two"),
            "character_start_times_seconds": [index / 10 for index in range(7)],
            "character_end_times_seconds": [
                (index + 1) / 10 for index in range(7)
            ],
        }

        srt_content = generate_srt_from_elevenlabs_alignment(alignment)

        self.assertEqual(srt_content, "1\n00:00:00,000 --> 00:00:00,700\nOne two")

    def test_generates_srt_from_character_alignment(self) -> None:
        alignment = ElevenLabsAlignment(
            characters=list("Hello world. Again!"),
            starts=[index / 10 for index in range(19)],
            ends=[(index + 1) / 10 for index in range(19)],
        )

        srt_content = generate_srt_from_elevenlabs_alignment(alignment)

        self.assertEqual(
            srt_content,
            "1\n00:00:00,000 --> 00:00:01,200\nHello world.\n\n"
            "2\n00:00:01,300 --> 00:00:01,900\nAgain!",
        )

    def test_respects_maximum_words_per_screen(self) -> None:
        alignment = ElevenLabsAlignment(
            characters=list("One two three four"),
            starts=[index / 10 for index in range(18)],
            ends=[(index + 1) / 10 for index in range(18)],
        )

        srt_content = generate_srt_from_elevenlabs_alignment(
            alignment,
            max_words_per_screen=2,
        )

        self.assertEqual(
            srt_content,
            "1\n00:00:00,000 --> 00:00:00,700\nOne two\n\n"
            "2\n00:00:00,800 --> 00:00:01,800\nthree four",
        )


class UpdateVideoScenarioTests(TestCase):
    def setUp(self) -> None:
        self.user = get_user_model().objects.create_user(
            username="scenario-owner",
            password="test-password",
        )
        self.video = GenVideo.objects.create(
            user=self.user,
            title="Scenario update",
            scenario="Old scenario",
            voice_model="alloy",
            voice_duration=12.5,
            srt_content="Old subtitles",
            elevenlabs_alignment={"characters": ["O"]},
            status=GenVideo.Statuses.COMPLETED,
            error_type=GenVideo.ErrorTypes.RENDERING,
            error_details="Old error",
            progress="Old progress",
        )
        self.video.voice_file.name = "voice_files/old-voice.mp3"
        self.video.srt_file.name = "srt_files/old.srt"
        self.video.final_file.name = "final_videos/old-final.mp4"
        self.video.save()
        VideoSegment.objects.create(
            video=self.video,
            text="Old segment",
            order=1,
            query="old",
            start_time=0,
            end_time=1,
        )
        self.url = reverse("update_video_scenario", args=[self.video.id])
        self.client.force_login(self.user)

    def post(self, payload):
        return self.client.post(
            self.url,
            data=json.dumps(payload),
            content_type="application/json",
        )

    def test_rejects_empty_scenario_without_changing_video(self) -> None:
        response = self.post({"scenario": "  "})

        self.assertEqual(response.status_code, 400)
        self.video.refresh_from_db()
        self.assertEqual(self.video.scenario, "Old scenario")
        self.assertTrue(self.video.voice_file)
        self.assertEqual(self.video.segments.count(), 1)

    def test_rejects_scenario_update_for_another_users_video(self) -> None:
        other_user = get_user_model().objects.create_user(
            username="other-user",
            password="test-password",
        )
        self.client.force_login(other_user)

        response = self.post({"scenario": "New scenario"})

        self.assertEqual(response.status_code, 404)
        self.video.refresh_from_db()
        self.assertEqual(self.video.scenario, "Old scenario")
        self.assertEqual(self.video.segments.count(), 1)

    @override_settings(TTS_PROVIDER="openai")
    @patch("agent.views.generate_voice_file_openai")
    def test_updates_scenario_clears_dependencies_and_starts_voice_generation(
        self, mock_generate_voice
    ) -> None:
        with self.captureOnCommitCallbacks(execute=True):
            response = self.post({"scenario": "New scenario"})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])
        self.video.refresh_from_db()
        self.assertEqual(self.video.scenario, "New scenario")
        self.assertFalse(self.video.voice_file)
        self.assertIsNone(self.video.voice_duration)
        self.assertFalse(self.video.srt_file)
        self.assertEqual(self.video.srt_content, "")
        self.assertIsNone(self.video.elevenlabs_alignment)
        self.assertEqual(self.video.status, GenVideo.Statuses.GENERATING_VOICE)
        self.assertIsNone(self.video.error_type)
        self.assertEqual(self.video.error_details, "")
        self.assertEqual(self.video.progress, "")
        self.assertEqual(self.video.segments.count(), 0)
        self.assertTrue(self.video.final_file)
        mock_generate_voice.assert_called_once_with(self.video)


class PexelsVideoFilePickerTests(SimpleTestCase):
    def test_prefers_full_hd_even_when_larger_option_exists(self) -> None:
        video_files = [
            {"link": "https://cdn.test/video-4k.mp4", "width": 3840, "height": 2160},
            {"link": "https://cdn.test/video-fhd.mp4", "width": 1920, "height": 1080},
            {"link": "https://cdn.test/video-hd.mp4", "width": 1280, "height": 720},
        ]

        picked = _pick_preferred_video_file(video_files)

        self.assertIsNotNone(picked)
        self.assertEqual(picked.get("link"), "https://cdn.test/video-fhd.mp4")

    def test_returns_largest_available_when_full_hd_missing(self) -> None:
        video_files = [
            {"link": "https://cdn.test/video-hd.mp4", "width": 1280, "height": 720},
            {"link": "https://cdn.test/video-qhd.mp4", "width": 2560, "height": 1440},
            {"link": "https://cdn.test/video-sd.mp4", "width": 854, "height": 480},
        ]

        picked = _pick_preferred_video_file(video_files)

        self.assertIsNotNone(picked)
        self.assertEqual(picked.get("link"), "https://cdn.test/video-qhd.mp4")

    def test_falls_back_to_first_url_when_dimensions_invalid(self) -> None:
        video_files = [
            {
                "link": "https://cdn.test/video-stream.m3u8",
                "width": None,
                "height": None,
            },
            {
                "link": "https://cdn.test/video-no-height.mp4",
                "width": 1920,
                "height": None,
            },
        ]

        picked = _pick_preferred_video_file(video_files)

        self.assertIsNotNone(picked)
        self.assertEqual(picked.get("link"), "https://cdn.test/video-stream.m3u8")


class PixabayVideoVariantPickerTests(SimpleTestCase):
    def test_prefers_full_hd_even_when_larger_variant_exists(self) -> None:
        videos = {
            "large": {
                "url": "https://cdn.test/video-4k.mp4",
                "width": 3840,
                "height": 2160,
            },
            "medium": {
                "url": "https://cdn.test/video-fhd-portrait.mp4",
                "width": 1080,
                "height": 1920,
            },
            "small": {
                "url": "https://cdn.test/video-hd.mp4",
                "width": 1280,
                "height": 720,
            },
        }

        picked = _pick_video_variant(videos)

        self.assertIsNotNone(picked)
        self.assertEqual(picked.get("url"), "https://cdn.test/video-fhd-portrait.mp4")

    def test_returns_largest_available_when_full_hd_missing(self) -> None:
        videos = {
            "large": {
                "url": "https://cdn.test/video-hd.mp4",
                "width": 1280,
                "height": 720,
            },
            "medium": {
                "url": "https://cdn.test/video-qhd.mp4",
                "width": 2560,
                "height": 1440,
            },
            "small": {
                "url": "https://cdn.test/video-sd.mp4",
                "width": 640,
                "height": 360,
            },
        }

        picked = _pick_video_variant(videos)

        self.assertIsNotNone(picked)
        self.assertEqual(picked.get("url"), "https://cdn.test/video-qhd.mp4")

    def test_falls_back_to_first_url_when_dimensions_invalid(self) -> None:
        videos = {
            "large": {
                "url": "https://cdn.test/video-stream.m3u8",
                "width": None,
                "height": None,
            },
            "medium": {
                "url": "https://cdn.test/video-no-height.mp4",
                "width": 1920,
                "height": None,
            },
            "small": {"url": None, "width": 1280, "height": 720},
        }

        picked = _pick_video_variant(videos)

        self.assertIsNotNone(picked)
        self.assertEqual(picked.get("url"), "https://cdn.test/video-stream.m3u8")
