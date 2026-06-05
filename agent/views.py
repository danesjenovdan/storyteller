import json
import os
import subprocess
import tempfile
from functools import wraps

from django.conf import settings as django_settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.translation import gettext as _
from django.utils.translation import ngettext
from PIL import Image

from agent.content_apis import (
    resolve_sources,
    search_images_by_sources,
    search_videos_by_sources,
)
from agent.tasks import (
    generate_srt_file,
    generate_voice_file_eleven_labs,
    generate_voice_file_gemini,
    generate_voice_file_openai,
    get_audio_duration,
    render_final_video,
)
from agent.utils import get_temporary_file_path

from .forms import VideoCreateForm
from .models import GenVideo, UsersLogo, VideoSegment

# Create your views here.


def ajax_login_required(view_func):
    """
    Decorator for AJAX views that returns JSON error instead of redirect when not authenticated.
    """

    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({"error": _("Authentication required")}, status=401)
        return view_func(request, *args, **kwargs)

    return wrapper


@login_required(login_url="/admin/login/")
def video_list(request):
    user = request.user
    videos = user.videos.all().order_by("-created_at")
    return render(request, "agent/videos.html", {"videos": videos})


@login_required(login_url="/admin/login/")
def modify_scenario_with_gemini(request):
    """AJAX endpoint to modify scenario using Gemini API."""
    import json

    from django.http import JsonResponse
    from langchain.chat_models import init_chat_model

    from agent.utils import ensure_google_api_key

    if request.method != "POST":
        return JsonResponse({"error": _("Method not allowed")}, status=405)

    try:
        data = json.loads(request.body)
        scenario = data.get("scenario", "")
        modify_prompt = data.get("modify_prompt", "")

        if not modify_prompt:
            return JsonResponse({"error": _("Modify prompt is required")}, status=400)

        # Ensure Google API key is set
        ensure_google_api_key()

        # Build the prompt - if scenario is empty, just use the modify_prompt
        if scenario:
            full_prompt = f"Ukaz:\n{modify_prompt}\n\nScenarij:\n{scenario}"
        else:
            full_prompt = modify_prompt

        # Call Gemini model
        model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
        model_response = model.invoke(full_prompt)

        return JsonResponse(
            {"success": True, "modified_scenario": model_response.content}
        )

    except Exception as e:
        return JsonResponse(
            {"error": _("Error calling Gemini: %(error)s") % {"error": str(e)}},
            status=500,
        )


@login_required(login_url="/admin/login/")
def video_create(request):
    """View for creating a new video with title, scenario, and prompt."""
    # Get TTS provider from settings
    tts_provider = django_settings.TTS_PROVIDER

    # Voice models based on TTS provider
    if tts_provider == "elevenlabs":
        VOICE_MODELS = [
            ("", _("--- Izberite glasovni model ---")),
            ("21m00Tcm4TlvDq8ikWAM", "Rachel - Calm and composed"),
            ("AZnzlk1XvdvUeBnXmlld", "Domi - Strong and authoritative"),
            ("EXAVITQu4vr4xnSDxMaL", "Bella - Soft and warm"),
            ("ErXwobaYiN019PkySvjV", "Antoni - Well-rounded and versatile"),
            ("MF3mGyEYCl7XYWbV9V6O", "Elli - Emotional and expressive"),
            ("TxGEqnHWrfWFTfGW9XjX", "Josh - Deep and authoritative"),
            ("VR6AewLTigWG4xSOukaG", "Arnold - Crisp and clear"),
            ("pNInz6obpgDQGcFmaJgB", "Adam - Deep and resonant"),
            ("yoZ06aMxZJJ28mfd3POQ", "Sam - Dynamic and raspy"),
        ]
    elif tts_provider == "gemini":
        VOICE_MODELS = [
            ("", _("--- Izberite glasovni model ---")),
            ("Puck", "Puck - Neutral and balanced"),
            ("Charon", "Charon - Male, authoritative"),
            ("Kore", "Kore - Warm, storytelling"),
            ("Fenrir", "Fenrir - Deep male"),
            ("Aoede", "Aoede - Female, energetic"),
        ]
    else:  # openai
        VOICE_MODELS = [
            ("", _("--- Izberite glasovni model ---")),
            ("alloy", "Alloy - Neutral and balanced"),
            ("echo", "Echo - Male, clear and expressive"),
            ("fable", "Fable - British accent, warm"),
            ("onyx", "Onyx - Deep and authoritative"),
            ("nova", "Nova - Female, energetic"),
            ("shimmer", "Shimmer - Female, soft and warm"),
        ]
    if request.method == "POST":
        form = VideoCreateForm(request.POST, request.FILES, voice_models=VOICE_MODELS)
        if form.is_valid():
            video = form.save(commit=False)
            video.user = request.user
            video.save()
            if video.scenario:
                # If only scenario is provided, directly simplify to scenario
                messages.success(
                    request, _("Video ustvarjen! Generiranje zvočne datoteke...")
                )
                video.status = GenVideo.Statuses.GENERATING_VOICE
                video.save()
                if tts_provider == "elevenlabs":
                    generate_voice_file_eleven_labs(video)
                elif tts_provider == "openai":
                    generate_voice_file_openai(video)
                elif tts_provider == "gemini":
                    generate_voice_file_gemini(video)
                return redirect("video_detail", video_id=video.id)
            elif video.voice_file:
                with get_temporary_file_path(video.voice_file) as temp_audio_path:
                    duration = get_audio_duration(temp_audio_path)
                    video.voice_duration = duration
                    video.save()
                generate_srt_file(video)
                return redirect("video_detail", video_id=video.id)
            else:
                messages.success(request, _("Video mora vsebovati scenario!"))
                return render(request, "agent/video_create.html", {"form": form})
        else:
            messages.error(request, _("Napaka pri ustvarjanju videa."))
    else:
        form = VideoCreateForm(voice_models=VOICE_MODELS)

    return render(request, "agent/video_create.html", {"form": form})


@ajax_login_required
def search_videos(request, video_segment_id):
    """
    AJAX endpoint to search videos from selected content sources.
    """

    video_segment = get_object_or_404(
        VideoSegment, id=video_segment_id, video__user=request.user
    )

    query = request.GET.get("query", video_segment.query)
    page = int(request.GET.get("page", 1))
    sources = resolve_sources(request.GET.getlist("sources"))

    try:
        duration = video_segment.end_time - video_segment.start_time

        videos, warnings = search_videos_by_sources(
            sources=sources,
            query=query,
            page=page,
            min_duration=duration,
        )

        deduped_videos = []
        seen_urls = set()
        for video in videos:
            video_url = video.get("video_url")
            if not video_url or video_url in seen_urls:
                continue
            seen_urls.add(video_url)
            deduped_videos.append(video)

        if not deduped_videos and warnings:
            return JsonResponse({"error": "; ".join(warnings)}, status=500)

        response = {
            "videos": deduped_videos,
            "query": query,
            "total": len(deduped_videos),
        }
        if warnings:
            response["warnings"] = warnings

        return JsonResponse(response)

    except Exception as e:
        return JsonResponse(
            {"error": _("Unexpected error: %(error)s") % {"error": str(e)}},
            status=500,
        )


@ajax_login_required
def search_images(request, video_segment_id):
    """
    Search for images across selected content sources.
    Returns list of images with metadata.
    """
    video_segment = get_object_or_404(
        VideoSegment, id=video_segment_id, video__user=request.user
    )

    query = request.GET.get("query", video_segment.query)
    page = int(request.GET.get("page", 1))
    sources = resolve_sources(request.GET.getlist("sources"))

    if not query:
        return JsonResponse({"error": _("Query parameter is required")}, status=400)

    try:
        images, warnings = search_images_by_sources(
            sources=sources,
            query=query,
            page=page,
            duration=video_segment.duration(),
        )

        deduped_images = []
        seen_urls = set()
        for image in images:
            image_url = image.get("video_url")
            if not image_url or image_url in seen_urls:
                continue
            seen_urls.add(image_url)
            deduped_images.append(image)

        if not deduped_images and warnings:
            return JsonResponse({"error": "; ".join(warnings)}, status=500)

        response = {
            "images": deduped_images,
            "query": query,
            "total": len(deduped_images),
        }
        if warnings:
            response["warnings"] = warnings

        return JsonResponse(response)

    except Exception as e:
        return JsonResponse(
            {"error": _("Unexpected error: %(error)s") % {"error": str(e)}},
            status=500,
        )


@ajax_login_required
def upload_segment_image(request, video_segment_id):
    """
    Upload a video or image for a video segment.
    Media will be used during rendering.
    """

    if request.method != "POST":
        return JsonResponse({"error": _("Method not allowed")}, status=405)

    video_segment = get_object_or_404(
        VideoSegment, id=video_segment_id, video__user=request.user
    )

    try:
        if "image" not in request.FILES:
            return JsonResponse({"error": _("No file provided")}, status=400)

        uploaded_file = request.FILES["image"]

        # Check if it's video or image
        is_video = uploaded_file.content_type.startswith("video/")
        is_image = uploaded_file.content_type.startswith("image/")

        if not is_video and not is_image:
            return JsonResponse(
                {"error": _("File must be a video or image")}, status=400
            )

        # Create a unique filename
        ext = os.path.splitext(uploaded_file.name)[1]
        if is_video:
            filename = f"segment_videos/segment_{video_segment_id}_{uploaded_file.name}"
        else:
            filename = f"segment_images/segment_{video_segment_id}_{uploaded_file.name}"

        # Save the file
        file_path = default_storage.save(filename, uploaded_file)
        file_url = default_storage.url(file_path)

        if is_image:
            # Get image dimensions
            uploaded_file.seek(0)  # Reset file pointer
            img = Image.open(uploaded_file)
            width, height = img.size

            return JsonResponse(
                {
                    "success": True,
                    "image_url": file_url,
                    "width": width,
                    "height": height,
                    "is_image": True,
                    "message": _("Image uploaded successfully"),
                }
            )
        else:
            # Get video dimensions and duration using ffprobe
            # For S3 storage, we need to download the file temporarily
            temp_file_created = False

            # Check if we're using local storage or S3
            if django_settings.ENABLE_S3:
                # S3 storage - use uploaded file directly, save to temp
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                    uploaded_file.seek(0)  # Reset file pointer
                    tmp_file.write(uploaded_file.read())
                    full_path = tmp_file.name
                    temp_file_created = True
            else:
                # Local storage
                full_path = os.path.join(django_settings.MEDIA_ROOT, file_path)

            try:
                cmd = [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_format",
                    "-show_streams",
                    str(full_path),
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    return JsonResponse(
                        {
                            "error": _("Could not analyze video: %(error)s")
                            % {"error": result.stderr}
                        },
                        status=500,
                    )

                data = json.loads(result.stdout)

                # Get video stream info
                video_stream = next(
                    (s for s in data.get("streams", []) if s["codec_type"] == "video"),
                    None,
                )
                if not video_stream:
                    return JsonResponse(
                        {"error": _("No video stream found")}, status=400
                    )

                width = int(video_stream.get("width", 0))
                height = int(video_stream.get("height", 0))
                duration = float(data.get("format", {}).get("duration", 0))

                return JsonResponse(
                    {
                        "success": True,
                        "video_url": file_url,
                        "width": width,
                        "height": height,
                        "duration": duration,
                        "is_image": False,
                        "message": _("Video uploaded successfully"),
                    }
                )
            finally:
                # Clean up temp file if we created one
                if temp_file_created and os.path.exists(full_path):
                    os.unlink(full_path)

    except Exception as e:
        return JsonResponse(
            {"error": _("Error uploading file: %(error)s") % {"error": str(e)}},
            status=500,
        )


@ajax_login_required
def save_selected_video(request, video_segment_id):
    """
    Save selected video URL to VideoSegment.
    Video will be downloaded later during rendering.
    """
    import json

    from django.http import JsonResponse

    if request.method != "POST":
        return JsonResponse({"error": _("Method not allowed")}, status=405)

    video_segment = get_object_or_404(
        VideoSegment, id=video_segment_id, video__user=request.user
    )

    try:
        data = json.loads(request.body)
        video_url = data.get("video_url")
        video_metadata = data.get("metadata", {})

        allowed_in_animations = {"none", "fade"}
        allowed_mid_animations = {
            "none",
            "zoom_in",
            "zoom_out",
            "subtle_pan_lr",
            "subtle_pan_ud",
        }
        allowed_out_animations = {"none", "fade"}

        requested_in = (video_metadata.get("animation_in") or "").strip()
        requested_mid = (video_metadata.get("animation_mid") or "").strip()
        requested_out = (video_metadata.get("animation_out") or "").strip()

        normalized_in = (
            requested_in if requested_in in allowed_in_animations else "none"
        )
        normalized_mid = (
            requested_mid if requested_mid in allowed_mid_animations else "none"
        )
        normalized_out = (
            requested_out if requested_out in allowed_out_animations else "none"
        )

        if not video_url:
            return JsonResponse({"error": _("video_url is required")}, status=400)

        # Check if we're updating existing video or adding new one
        existing_proposal = None
        if video_segment.video_proposals:
            for proposal in video_segment.video_proposals:
                if proposal.get("video_url") == video_url or proposal.get("selected"):
                    existing_proposal = proposal
                    break

        if existing_proposal:
            final_animation_in = (
                normalized_in
                if requested_in
                else existing_proposal.get("animation_in", "none")
            )
            final_animation_mid = (
                normalized_mid
                if requested_mid
                else existing_proposal.get("animation_mid", "none")
            )
            final_animation_out = (
                normalized_out
                if requested_out
                else existing_proposal.get("animation_out", "none")
            )

            # Update existing proposal - preserve all existing data and update with new metadata
            existing_proposal.update(
                {
                    "source": video_metadata.get(
                        "source", existing_proposal.get("source")
                    ),
                    "source_id": video_metadata.get(
                        "id", existing_proposal.get("source_id")
                    ),
                    "pexels_id": video_metadata.get(
                        "id", existing_proposal.get("pexels_id")
                    ),
                    "pexels_url": video_metadata.get(
                        "url", existing_proposal.get("pexels_url")
                    ),
                    "video_url": video_url,
                    "user": video_metadata.get("user", existing_proposal.get("user")),
                    "duration": video_metadata.get(
                        "duration", existing_proposal.get("duration")
                    ),
                    "width": video_metadata.get(
                        "width", existing_proposal.get("width")
                    ),
                    "height": video_metadata.get(
                        "height", existing_proposal.get("height")
                    ),
                    "horizontal_mode": video_metadata.get(
                        "horizontal_mode",
                        existing_proposal.get("horizontal_mode", "crop"),
                    ),
                    "animation_in": final_animation_in,
                    "animation_mid": final_animation_mid,
                    "animation_out": final_animation_out,
                    "animation": final_animation_mid,
                    "is_image": video_metadata.get(
                        "is_image", existing_proposal.get("is_image", False)
                    ),
                    "selected": True,
                }
            )
            video_segment.video_proposals = [existing_proposal]
        else:
            # Save new video proposal
            video_segment.video_proposals = [
                {
                    "source": video_metadata.get("source"),
                    "source_id": video_metadata.get("id"),
                    "pexels_id": video_metadata.get("id"),
                    "pexels_url": video_metadata.get("url"),
                    "video_url": video_url,
                    "user": video_metadata.get("user"),
                    "duration": video_metadata.get("duration"),
                    "width": video_metadata.get("width"),
                    "height": video_metadata.get("height"),
                    "horizontal_mode": video_metadata.get("horizontal_mode", "crop"),
                    "animation_in": normalized_in,
                    "animation_mid": normalized_mid,
                    "animation_out": normalized_out,
                    "animation": normalized_mid,
                    "is_image": video_metadata.get("is_image", False),
                    "selected": True,
                }
            ]

        video_segment.save()

        # Update video status
        gen_video = video_segment.video
        total_segments = gen_video.segments.count()
        completed_segments = gen_video.segments.filter(
            video_proposals__0__selected=True
        ).count()

        if completed_segments == 1:
            # First video selected, update status
            gen_video.status = GenVideo.Statuses.SELECTING_VIDEOS
            gen_video.save()
        elif completed_segments == total_segments:
            # All videos selected
            gen_video.status = GenVideo.Statuses.VIDEOS_SELECTED
            gen_video.save()

        return JsonResponse(
            {
                "success": True,
                "message": _("Video URL uspešno shranjen."),
                "video_url": video_url,
            }
        )

    except Exception as e:
        return JsonResponse(
            {"error": _("Error saving video: %(error)s") % {"error": str(e)}},
            status=500,
        )


@login_required(login_url="/admin/login/")
def video_detail(request, video_id):
    """
    Display video details, status, and all associated files.
    Shows:
    - Video metadata and status
    - Content script
    - Voice file (audio)
    - SRT subtitles
    - All VideoSentences with their video clips
    - Final rendered video (if available)
    """
    video = get_object_or_404(GenVideo, id=video_id, user=request.user)

    # Get all video segments ordered by order
    segments = video.segments.all().order_by("order")

    # Calculate progress - count segments with selected videos (either downloaded or URL saved)
    total_segments = segments.count()
    completed_segments = sum(
        1
        for seg in segments
        if seg.video_proposals and seg.video_proposals[0].get("selected")
    )
    progress_percentage = (
        (completed_segments / total_segments * 100) if total_segments > 0 else 0
    )

    context = {
        "video": video,
        "segments": segments,
        "user_logos": request.user.logos.all().order_by("-created_at"),
        "total_segments": total_segments,
        "completed_segments": completed_segments,
        "progress_percentage": progress_percentage,
    }

    return render(request, "agent/video_detail.html", context)


@login_required(login_url="/admin/login/")
def render_video(request, video_id):
    """
    Trigger rendering of final video from all VideoSegment clips.
    Videos will be downloaded during rendering from stored URLs.
    """
    if request.method != "POST":
        return redirect("video_detail", video_id=video_id)

    video = get_object_or_404(GenVideo, id=video_id, user=request.user)

    # Validate that all segments have video URLs selected
    segments = video.segments.all()
    total_segments = segments.count()

    segments_with_urls = [
        seg
        for seg in segments
        if seg.video_proposals and seg.video_proposals[0].get("selected")
    ]

    if len(segments_with_urls) != total_segments:
        missing_count = total_segments - len(segments_with_urls)
        error_message = ngettext(
            "Ne moreš renderirati videa - manjka izbrani video klip (%(selected)s/%(total)s)",
            "Ne moreš renderirati videa - manjkajo izbrani video klipi (%(selected)s/%(total)s)",
            missing_count,
        ) % {"selected": len(segments_with_urls), "total": total_segments}
        messages.error(
            request,
            error_message,
        )
        return redirect("video_detail", video_id=video_id)

    if not video.voice_file:
        messages.error(
            request, _("Ne moreš renderirati videa - manjka zvočna datoteka")
        )
        return redirect("video_detail", video_id=video_id)

    # Trigger rendering task
    render_final_video(video)

    messages.success(
        request, _("Renderiranje videa se je začelo! To lahko traja nekaj minut.")
    )

    return redirect("video_detail", video_id=video_id)


@login_required(login_url="/admin/login/")
def generate_voice(request, video_id):
    """
    Generate voice file from scenario based on TTS provider.
    """
    from django.conf import settings as django_settings

    if request.method != "POST":
        return redirect("video_detail", video_id=video_id)

    video = get_object_or_404(GenVideo, id=video_id, user=request.user)

    if not video.scenario:
        messages.error(
            request, _("Ne moreš generirati zvoka - manjka vsebinski skript")
        )
        return redirect("video_detail", video_id=video_id)

    if not video.voice_model:
        messages.error(
            request,
            _(
                "Ne moreš generirati zvoka - manjka glasovni model. Uredi skript in izberi glas."
            ),
        )
        return redirect("video_edit_script", video_id=video_id)

    # Get TTS provider from settings
    tts_provider = django_settings.TTS_PROVIDER

    # Trigger voice generation based on provider
    if tts_provider == "elevenlabs":
        generate_voice_file_eleven_labs(video)
    elif tts_provider == "gemini":
        generate_voice_file_gemini(video)
    else:  # openai
        generate_voice_file_openai(video)

    messages.success(
        request,
        _(
            "Generiranje zvočnega posnetka se je začelo (%(provider)s)! Posnetek bo kmalu na voljo."
        )
        % {"provider": tts_provider.upper()},
    )

    return redirect("video_detail", video_id=video_id)


@login_required(login_url="/admin/login/")
def regenerate_segments(request, video_id):
    """
    Regenerate video segments from content script.
    Deletes existing segments and creates new ones.
    """
    from django.contrib import messages
    from django.shortcuts import redirect

    from agent.tasks import get_video_segments

    if request.method != "POST":
        return redirect("video_detail", video_id=video_id)

    video = get_object_or_404(GenVideo, id=video_id, user=request.user)

    if not video.scenario:
        messages.error(
            request, _("Ne moreš generirati segmentov - manjka vsebinski skript")
        )
        return redirect("video_detail", video_id=video_id)

    # Delete existing segments
    video.segments.all().delete()

    # Trigger segment generation task
    get_video_segments(video)

    messages.success(request, _("Segmenti se ponovno generirajo..."))
    return redirect("video_detail", video_id=video_id)


def regenerate_srt(request, video_id):
    """
    Regenerate SRT subtitle file from voice file.
    Deletes existing SRT and creates new one.
    """
    from django.contrib import messages
    from django.shortcuts import redirect

    from agent.tasks import generate_srt_file

    if request.method != "POST":
        return redirect("video_detail", video_id=video_id)

    video = get_object_or_404(GenVideo, id=video_id, user=request.user)

    if not video.voice_file:
        messages.error(
            request, _("Ne moreš generirati podnapisov - manjka zvočna datoteka")
        )
        return redirect("video_detail", video_id=video_id)

    # Delete existing SRT file
    if video.srt_file:
        video.srt_file.delete()
    video.srt_content = ""
    video.save()

    # Trigger SRT generation task
    generate_srt_file(video)

    messages.success(
        request,
        _("Generiranje podnapisov se je začelo! Podnapisi bodo kmalu na voljo."),
    )

    return redirect("video_detail", video_id=video_id)


@login_required(login_url="/admin/login/")
def set_subtitle_style(request, video_id):
    """
    Set subtitle style for video rendering.
    """
    import json

    from django.http import JsonResponse

    if request.method != "POST":
        return JsonResponse({"error": _("Method not allowed")}, status=405)

    video = get_object_or_404(GenVideo, id=video_id, user=request.user)

    try:
        data = json.loads(request.body)
        font_size = data.get("font_size", 12)
        font_family = data.get("font_family", "Montserrat")
        font_weight = data.get("font_weight", "900")
        stroke_weight = data.get("stroke_weight", 3)
        shadow = data.get("shadow", 1)
        vertical_position = data.get("vertical_position", 10)

        video.subtitle_font_size = int(font_size)
        video.subtitle_font_family = font_family
        video.subtitle_font_weight = str(font_weight)
        video.subtitle_stroke_weight = int(stroke_weight)
        video.subtitle_shadow = int(shadow)
        video.subtitle_vertical_position = int(vertical_position)
        video.save()

        return JsonResponse(
            {
                "success": True,
                "font_size": font_size,
                "font_family": font_family,
                "font_weight": font_weight,
                "stroke_weight": stroke_weight,
                "shadow": shadow,
                "vertical_position": vertical_position,
            }
        )

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@login_required(login_url="/admin/login/")
def upload_logo(request, video_id):
    """
    Upload a new logo for the current user and optionally select it for this video.
    """
    video = get_object_or_404(GenVideo, id=video_id, user=request.user)

    if request.method != "POST":
        return JsonResponse({"error": _("Method not allowed")}, status=405)

    try:
        if "logo" not in request.FILES:
            return JsonResponse({"error": _("No logo file provided")}, status=400)

        uploaded_file = request.FILES["logo"]
        if not uploaded_file.content_type.startswith("image/"):
            return JsonResponse({"error": _("Logo must be an image")}, status=400)

        user_logo = UsersLogo.objects.create(user=request.user, logo_file=uploaded_file)

        # Newly uploaded logo becomes selected for convenience.
        video.logo = user_logo
        video.save(update_fields=["logo", "updated_at"])

        return JsonResponse(
            {
                "success": True,
                "logo": {
                    "id": user_logo.id,
                    "url": user_logo.logo_file.url,
                    "name": os.path.basename(user_logo.logo_file.name),
                },
            }
        )
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@login_required(login_url="/admin/login/")
def set_video_logo(request, video_id):
    """
    Select which user logo should be used for a specific video.
    """
    import json

    video = get_object_or_404(GenVideo, id=video_id, user=request.user)

    if request.method != "POST":
        return JsonResponse({"error": _("Method not allowed")}, status=405)

    try:
        data = json.loads(request.body)
        logo_id = data.get("logo_id")

        if logo_id in (None, "", "null"):
            video.logo = None
            video.save(update_fields=["logo", "updated_at"])
            return JsonResponse({"success": True, "logo_id": None})

        selected_logo = get_object_or_404(UsersLogo, id=logo_id, user=request.user)
        video.logo = selected_logo
        video.save(update_fields=["logo", "updated_at"])

        return JsonResponse(
            {
                "success": True,
                "logo_id": selected_logo.id,
                "logo_url": selected_logo.logo_file.url,
            }
        )
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@login_required(login_url="/admin/login/")
def set_logo_settings(request, video_id):
    """
    Save logo placement settings (corner and size) for a video.
    """
    import json

    video = get_object_or_404(GenVideo, id=video_id, user=request.user)

    if request.method != "POST":
        return JsonResponse({"error": _("Method not allowed")}, status=405)

    try:
        data = json.loads(request.body)

        logo_position = data.get("logo_position", GenVideo.LogoPositions.TOP_RIGHT)
        logo_size_percent = int(data.get("logo_size_percent", 15))

        if logo_position not in {
            GenVideo.LogoPositions.TOP_LEFT,
            GenVideo.LogoPositions.TOP_RIGHT,
        }:
            return JsonResponse({"error": _("Invalid logo position")}, status=400)

        logo_size_percent = max(5, min(40, logo_size_percent))

        video.logo_position = logo_position
        video.logo_size_percent = logo_size_percent
        video.save(update_fields=["logo_position", "logo_size_percent", "updated_at"])

        return JsonResponse(
            {
                "success": True,
                "logo_position": logo_position,
                "logo_size_percent": logo_size_percent,
            }
        )
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
