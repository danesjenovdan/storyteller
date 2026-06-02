from django.urls import path

from . import views

urlpatterns = [
    path("", views.video_list, name="video_list"),
    path("videos/", views.video_list, name="video_list"),
    path("videos/create/", views.video_create, name="video_create"),
    path(
        "videos/modify-scenario/",
        views.modify_scenario_with_gemini,
        name="modify_scenario_with_gemini",
    ),
    path("videos/<int:video_id>/", views.video_detail, name="video_detail"),
    path(
        "videos/<int:video_id>/generate-voice/",
        views.generate_voice,
        name="generate_voice",
    ),
    path("videos/<int:video_id>/render/", views.render_video, name="render_video"),
    path(
        "videos/<int:video_id>/regenerate-segments/",
        views.regenerate_segments,
        name="regenerate_segments",
    ),
    path(
        "videos/<int:video_id>/regenerate-srt/",
        views.regenerate_srt,
        name="regenerate_srt",
    ),
    path(
        "videos/<int:video_id>/set-subtitle-style/",
        views.set_subtitle_style,
        name="set_subtitle_style",
    ),
    path(
        "videos/<int:video_id>/upload-logo/",
        views.upload_logo,
        name="upload_logo",
    ),
    path(
        "videos/<int:video_id>/set-logo/",
        views.set_video_logo,
        name="set_video_logo",
    ),
    path(
        "videos/<int:video_id>/set-logo-settings/",
        views.set_logo_settings,
        name="set_logo_settings",
    ),
    path(
        "video-segments/<int:video_segment_id>/search-videos/",
        views.search_videos,
        name="search_videos",
    ),
    path(
        "video-segments/<int:video_segment_id>/search-images/",
        views.search_images,
        name="search_images",
    ),
    path(
        "video-segments/<int:video_segment_id>/save-video/",
        views.save_selected_video,
        name="save_selected_video",
    ),
    path(
        "video-segments/<int:video_segment_id>/upload-image/",
        views.upload_segment_image,
        name="upload_segment_image",
    ),
]
