from django.urls import path

from . import views

urlpatterns = [
    path("", views.video_list, name="video_list"),
    path("videos/", views.video_list, name="video_list"),
    path("videos/create/", views.video_create, name="video_create"),
    path("videos/<int:video_id>/", views.video_detail, name="video_detail"),
    path(
        "videos/<int:video_id>/edit-scenario/",
        views.video_edit_scenario,
        name="video_edit_scenario",
    ),
    path(
        "videos/<int:video_id>/edit-script/",
        views.video_edit_script,
        name="video_edit_script",
    ),
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
        "video-segments/<int:video_segment_id>/videos-selector/",
        views.video_segment_videos_selector,
        name="video_segment_videos_selector",
    ),
    path(
        "video-segments/<int:video_segment_id>/search-videos/",
        views.search_pexels_videos,
        name="search_pexels_videos",
    ),
    path(
        "video-segments/<int:video_segment_id>/save-video/",
        views.save_selected_video,
        name="save_selected_video",
    ),
]
