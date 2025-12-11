import requests
from django.conf import settings as django_settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render

from agent.tasks import (
    generate_voice_file_eleven_labs,
    generate_voice_file_gemini,
    generate_voice_file_openai,
    render_final_video,
)

from .forms import VideoCreateForm
from .models import GenVideo, VideoSegment

# Create your views here.


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
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
        scenario = data.get("scenario", "")
        modify_prompt = data.get("modify_prompt", "")

        if not modify_prompt:
            return JsonResponse({"error": "Modify prompt is required"}, status=400)

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
        return JsonResponse({"error": f"Error calling Gemini: {str(e)}"}, status=500)


@login_required(login_url="/admin/login/")
def video_create(request):
    """View for creating a new video with title, scenario, and prompt."""
    # Get TTS provider from settings
    tts_provider = django_settings.TTS_PROVIDER

    # Voice models based on TTS provider
    if tts_provider == "elevenlabs":
        VOICE_MODELS = [
            ("", "--- Izberite glasovni model ---"),
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
            ("", "--- Izberite glasovni model ---"),
            ("Puck", "Puck - Neutral and balanced"),
            ("Charon", "Charon - Male, authoritative"),
            ("Kore", "Kore - Warm, storytelling"),
            ("Fenrir", "Fenrir - Deep male"),
            ("Aoede", "Aoede - Female, energetic"),
        ]
    else:  # openai
        VOICE_MODELS = [
            ("", "--- Izberite glasovni model ---"),
            ("alloy", "Alloy - Neutral and balanced"),
            ("echo", "Echo - Male, clear and expressive"),
            ("fable", "Fable - British accent, warm"),
            ("onyx", "Onyx - Deep and authoritative"),
            ("nova", "Nova - Female, energetic"),
            ("shimmer", "Shimmer - Female, soft and warm"),
        ]
    if request.method == "POST":
        form = VideoCreateForm(request.POST, voice_models=VOICE_MODELS)
        if form.is_valid():
            video = form.save(commit=False)
            video.user = request.user
            video.save()

            if video.scenario:
                # If only scenario is provided, directly simplify to scenario
                messages.success(
                    request, "Video ustvarjen! Generiranje zvočne datoteke..."
                )
                video.status = GenVideo.Statuses.GENERATING_VOICE
                video.save()
                generate_voice_file_gemini(video)
                return redirect("video_detail", video_id=video.id)
            else:
                messages.success(request, "Video mora vsebovati scenario!")
                return render(request, "agent/video_create.html", {"form": form})
        else:
            messages.error(request, "Napaka pri ustvarjanju videa.")
    else:
        form = VideoCreateForm(voice_models=VOICE_MODELS)

    return render(request, "agent/video_create.html", {"form": form})


@login_required(login_url="/admin/login/")
def video_segment_videos_selector(request, video_segment_id):
    """
    Query Pexels API for partial video selection.
    """
    video_segment = get_object_or_404(
        VideoSegment, id=video_segment_id, video__user=request.user
    )

    # Get query from GET parameter or use default
    query = request.GET.get("query", video_segment.query)

    return render(
        request,
        "agent/partial_videos_selector.html",
        {
            "video_segment": video_segment,
            "query": query,
            "duration": video_segment.end_time - video_segment.start_time,
        },
    )


@login_required(login_url="/admin/login/")
def search_pexels_videos(request, video_segment_id):
    """
    AJAX endpoint to search Pexels videos with custom query.
    """

    video_segment = get_object_or_404(
        VideoSegment, id=video_segment_id, video__user=request.user
    )

    query = request.GET.get("query", video_segment.query)
    page = request.GET.get("page", 1)

    if not django_settings.PEXELS_API_KEY:
        return JsonResponse({"error": "PEXELS_API_KEY is not configured"}, status=500)

    try:
        # Calculate segment duration
        duration = video_segment.end_time - video_segment.start_time

        # Pexels API endpoint for video search
        url = "https://api.pexels.com/videos/search"
        headers = {"Authorization": django_settings.PEXELS_API_KEY}

        keywords = query.split(",")
        keywords.append(query)
        videos = []
        ids = []
        for keyword in keywords:
            keyword = keyword.strip()

            params = {
                "query": keyword,
                "orientation": "portrait",
                "per_page": 20,
                "page": page,
                "size": "medium",
            }

            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()

            data = response.json()

            # Filter videos by duration
            min_duration = duration
            max_duration = duration + 15

            print(f"DEBUG: Total videos from Pexels: {len(data.get('videos', []))}")

            for video_item in data.get("videos", []):
                video_duration = video_item.get("duration", 0)
                print(
                    f"DEBUG: Checking video {video_item.get('id')}: duration={video_duration}"
                )

                # if min_duration <= video_duration <= max_duration:
                if min_duration <= video_duration:
                    # Get portrait video file
                    video_file = None
                    for file in video_item.get("video_files", []):
                        width = file.get("width", 0)
                        height = file.get("height", 0)
                        if width < height:
                            video_file = file
                            print(f"DEBUG: Found portrait file: {width}x{height}")
                            break

                    if video_file:
                        video_id = video_item.get("id")
                        if video_id in ids:
                            print(f"DEBUG: Skipping duplicate video {video_id}")
                            continue
                        ids.append(video_item.get("id"))
                        videos.append(
                            {
                                "id": video_item.get("id"),
                                "image": video_item.get("image"),
                                "duration": video_duration,
                                "video_url": video_file.get("link"),
                                "width": video_file.get("width"),
                                "height": video_file.get("height"),
                                "user": video_item.get("user", {}).get(
                                    "name", "Unknown"
                                ),
                                "url": video_item.get("url"),
                            }
                        )
                        print(f"DEBUG: Added video {video_item.get('id')}")

        print(f"DEBUG: Returning {len(videos)} filtered videos")

        return JsonResponse({"videos": videos, "query": query, "total": len(videos)})

    except requests.exceptions.RequestException as e:
        return JsonResponse({"error": f"Error fetching videos: {str(e)}"}, status=500)
    except Exception as e:
        return JsonResponse({"error": f"Unexpected error: {str(e)}"}, status=500)


@login_required(login_url="/admin/login/")
def save_selected_video(request, video_segment_id):
    """
    Save selected video to VideoSegment.
    Downloads video from Pexels and saves to video_file field.
    """
    import json

    import requests
    from django.core.files.base import ContentFile
    from django.http import JsonResponse

    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    video_segment = get_object_or_404(
        VideoSegment, id=video_segment_id, video__user=request.user
    )

    try:
        data = json.loads(request.body)
        video_url = data.get("video_url")
        video_metadata = data.get("metadata", {})

        if not video_url:
            return JsonResponse({"error": "video_url is required"}, status=400)

        # Download video from Pexels
        print(f"Downloading video from: {video_url}")
        response = requests.get(video_url, stream=True)
        response.raise_for_status()

        # Save video to model
        filename = (
            f"segment_{video_segment.id}_pexels_{video_metadata.get('id', 'video')}.mp4"
        )
        video_segment.video_file.save(
            filename, ContentFile(response.content), save=False
        )

        # Save metadata to video_proposals
        video_segment.video_proposals = [
            {
                "pexels_id": video_metadata.get("id"),
                "pexels_url": video_metadata.get("url"),
                "user": video_metadata.get("user"),
                "duration": video_metadata.get("duration"),
                "width": video_metadata.get("width"),
                "height": video_metadata.get("height"),
                "selected": True,
            }
        ]

        video_segment.save()

        # Update video status
        gen_video = video_segment.video
        total_segments = gen_video.segments.count()
        completed_segments = (
            gen_video.segments.filter(video_file__isnull=False)
            .exclude(video_file="")
            .count()
        )

        if completed_segments == 1:
            # First video selected, update status
            gen_video.status = GenVideo.Statuses.SELECTING_VIDEOS
            gen_video.save()
        elif completed_segments == total_segments:
            # All videos selected
            gen_video.status = GenVideo.Statuses.VIDEOS_SELECTED
            gen_video.save()

        # Find next VideoSegment for the same video
        next_segment = (
            VideoSegment.objects.filter(
                video=video_segment.video, order__gt=video_segment.order
            )
            .order_by("order")
            .first()
        )

        # Determine redirect URL
        if next_segment:
            from django.urls import reverse

            redirect_url = reverse(
                "video_segment_videos_selector",
                kwargs={"video_segment_id": next_segment.id},
            )
            message = f"Video uspešno shranjen. Preusmerjam na naslednji segment ({next_segment.order}/{video_segment.video.segments.count()})..."
        else:
            from django.urls import reverse

            redirect_url = reverse(
                "video_detail", kwargs={"video_id": video_segment.video.id}
            )
            message = "Video uspešno shranjen. Vsi segmenti so obdelani!"

        return JsonResponse(
            {
                "success": True,
                "message": message,
                "video_file_url": (
                    video_segment.video_file.url if video_segment.video_file else None
                ),
                "redirect_url": redirect_url,
                "has_next": next_segment is not None,
            }
        )

    except requests.exceptions.RequestException as e:
        return JsonResponse({"error": f"Error downloading video: {str(e)}"}, status=500)
    except Exception as e:
        return JsonResponse({"error": f"Error saving video: {str(e)}"}, status=500)


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

    # Calculate progress
    total_segments = segments.count()
    completed_segments = (
        segments.filter(video_file__isnull=False).exclude(video_file="").count()
    )
    progress_percentage = (
        (completed_segments / total_segments * 100) if total_segments > 0 else 0
    )

    context = {
        "video": video,
        "segments": segments,
        "total_segments": total_segments,
        "completed_segments": completed_segments,
        "progress_percentage": progress_percentage,
    }

    return render(request, "agent/video_detail.html", context)


@login_required(login_url="/admin/login/")
def render_video(request, video_id):
    """
    Trigger rendering of final video from all VideoSentence clips.
    """

    if request.method != "POST":
        return redirect("video_detail", video_id=video_id)

    video = get_object_or_404(GenVideo, id=video_id, user=request.user)

    # Validate that all segments have video files
    segments = video.segments.filter(video_file__isnull=False).exclude(video_file="")

    total_segments = video.segments.count()

    if segments.count() != total_segments:
        messages.error(
            request,
            f"Ne moreš renderirati videa - manjkajo video klipi ({segments.count()}/{total_segments})",
        )
        return redirect("video_detail", video_id=video_id)

    if not video.voice_file:
        messages.error(request, "Ne moreš renderirati videa - manjka zvočna datoteka")
        return redirect("video_detail", video_id=video_id)

    # Trigger rendering task
    render_final_video(video)

    messages.success(
        request, "Renderiranje videa se je začelo! To lahko traja nekaj minut."
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
        messages.error(request, "Ne moreš generirati zvoka - manjka vsebinski skript")
        return redirect("video_detail", video_id=video_id)

    if not video.voice_model:
        messages.error(
            request,
            "Ne moreš generirati zvoka - manjka glasovni model. Uredi skript in izberi glas.",
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
        f"Generiranje zvočnega posnetka se je začelo ({tts_provider.upper()})! Posnetek bo kmalu na voljo.",
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
            request, "Ne moreš generirati segmentov - manjka vsebinski skript"
        )
        return redirect("video_detail", video_id=video_id)

    # Delete existing segments
    video.segments.all().delete()

    # Trigger segment generation task
    get_video_segments(video)

    messages.success(
        request, "Generiranje segmentov se je začelo! Segmenti bodo kmalu na voljo."
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
        return JsonResponse({"error": "Method not allowed"}, status=405)

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
