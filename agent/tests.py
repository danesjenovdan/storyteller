from django.test import SimpleTestCase

from .content_apis.pexels import _pick_preferred_video_file
from .content_apis.pixabay import _pick_video_variant
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
