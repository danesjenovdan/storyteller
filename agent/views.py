import requests
from django.conf import settings as django_settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from functools import wraps

from agent.tasks import (
    generate_voice_file_eleven_labs,
    generate_voice_file_gemini,
    generate_voice_file_openai,
    render_final_video,
)

from .forms import VideoCreateForm
from .models import GenVideo, VideoSegment

# Create your views here.


def ajax_login_required(view_func):
    """
    Decorator for AJAX views that returns JSON error instead of redirect when not authenticated.
    """
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({"error": "Authentication required"}, status=401)
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


@ajax_login_required
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

        keywords = [
            *query.split(","),  # split query by commas
            query,  # also add full query
        ]
        videos = []
        ids = []
        for keyword in keywords:
            keyword = keyword.strip()

            params = {
                "query": keyword,
                #"orientation": "portrait",
                "per_page": 10,
                "page": page,
                "size": "medium",
            }

            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()

            data = response.json()

            # Filter videos by duration
            min_duration = duration
            max_duration = duration + 15

            # print(f"DEBUG: Total videos from Pexels: {len(data.get('videos', []))}")

            for video_item in data.get("videos", []):
                video_duration = video_item.get("duration", 0)
                # print(
                #     f"DEBUG: Checking video {video_item.get('id')}: duration={video_duration}"
                # )

                # if min_duration <= video_duration <= max_duration:
                if min_duration <= video_duration:
                    # Get portrait video file
                    video_file = None
                    for file in video_item.get("video_files", []):
                        width = file.get("width", 0)
                        height = file.get("height", 0)
                        video_file = file
                        if width < height:
                            break

                    if video_file:
                        video_id = video_item.get("id")
                        if video_id in ids:
                            # print(f"DEBUG: Skipping duplicate video {video_id}")
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
                        # print(f"DEBUG: Added video {video_item.get('id')}")

        # print(f"DEBUG: Returning {len(videos)} filtered videos")

        return JsonResponse({"videos": videos, "query": query, "total": len(videos)})

    except requests.exceptions.RequestException as e:
        return JsonResponse({"error": f"Error fetching videos: {str(e)}"}, status=500)
    except Exception as e:
        return JsonResponse({"error": f"Unexpected error: {str(e)}"}, status=500)


@ajax_login_required
def search_pexels_images(request, video_segment_id):
    """
    Search for images on Pexels based on query string.
    Returns list of images with metadata.
    """
    video_segment = get_object_or_404(
        VideoSegment, id=video_segment_id, video__user=request.user
    )

    query = request.GET.get("query", video_segment.query)
    page = int(request.GET.get("page", 1))
    per_page = 20

    if not query:
        return JsonResponse({"error": "Query parameter is required"}, status=400)

    if not django_settings.PEXELS_API_KEY:
        return JsonResponse({"error": "Pexels API key not configured"}, status=500)

    try:
        # Search Pexels for images
        headers = {"Authorization": django_settings.PEXELS_API_KEY}
        response = requests.get(
            "https://api.pexels.com/v1/search",
            headers=headers,
            params={
                "query": query,
                "per_page": per_page,
                "page": page,
                "orientation": "portrait",  # Prefer vertical images
            },
            timeout=10,
        )

        if response.status_code != 200:
            return JsonResponse(
                {"error": f"Pexels API error: {response.status_code}"}, status=500
            )

        data = response.json()
        images = []

        for photo in data.get("photos", []):
            # Get the large portrait image
            src = photo.get("src", {})
            image_url = src.get("large") or src.get("large2x") or src.get("original")
            
            images.append({
                "id": photo.get("id"),
                "image": src.get("medium"),  # Preview image
                "video_url": image_url,  # Full resolution for rendering
                "width": photo.get("width"),
                "height": photo.get("height"),
                "user": photo.get("photographer", "Unknown"),
                "url": photo.get("url"),
                "is_image": True,
                "duration": video_segment.duration(),  # Use segment duration for images
            })

        return JsonResponse({"images": images, "query": query, "total": len(images)})

    except requests.exceptions.RequestException as e:
        return JsonResponse({"error": f"Error fetching images: {str(e)}"}, status=500)
    except Exception as e:
        return JsonResponse({"error": f"Unexpected error: {str(e)}"}, status=500)


@ajax_login_required
def upload_segment_image(request, video_segment_id):
    """
    Upload a video or image for a video segment.
    Media will be used during rendering.
    """
    from PIL import Image
    from django.core.files.storage import default_storage
    import os
    import subprocess
    import json
    import tempfile

    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    video_segment = get_object_or_404(
        VideoSegment, id=video_segment_id, video__user=request.user
    )

    try:
        if 'image' not in request.FILES:
            return JsonResponse({"error": "No file provided"}, status=400)

        uploaded_file = request.FILES['image']
        
        # Check if it's video or image
        is_video = uploaded_file.content_type.startswith('video/')
        is_image = uploaded_file.content_type.startswith('image/')
        
        if not is_video and not is_image:
            return JsonResponse({"error": "File must be a video or image"}, status=400)

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
            
            return JsonResponse({
                "success": True,
                "image_url": file_url,
                "width": width,
                "height": height,
                "is_image": True,
                "message": "Image uploaded successfully",
            })
        else:
            # Get video dimensions and duration using ffprobe
            # For S3 storage, we need to download the file temporarily
            import tempfile
            
            if hasattr(default_storage, 'path'):
                # Local storage
                full_path = default_storage.path(file_path)
            else:
                # S3 or other remote storage - download to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                    uploaded_file.seek(0)
                    tmp_file.write(uploaded_file.read())
                    full_path = tmp_file.name
            
            try:
                cmd = [
                    "ffprobe",
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    "-show_streams",
                    str(full_path),
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    return JsonResponse({"error": f"Could not analyze video: {result.stderr}"}, status=500)
                
                data = json.loads(result.stdout)
                
                # Get video stream info
                video_stream = next((s for s in data.get('streams', []) if s['codec_type'] == 'video'), None)
                if not video_stream:
                    return JsonResponse({"error": "No video stream found"}, status=400)
                
                width = int(video_stream.get('width', 0))
                height = int(video_stream.get('height', 0))
                duration = float(data.get('format', {}).get('duration', 0))
                
                return JsonResponse({
                    "success": True,
                    "video_url": file_url,
                    "width": width,
                    "height": height,
                    "duration": duration,
                    "is_image": False,
                    "message": "Video uploaded successfully",
                })
            finally:
                # Clean up temp file if we created one
                if not hasattr(default_storage, 'path') and os.path.exists(full_path):
                    os.unlink(full_path)

    except Exception as e:
        return JsonResponse({"error": f"Error uploading file: {str(e)}"}, status=500)


@ajax_login_required
def save_selected_video(request, video_segment_id):
    """
    Save selected video URL to VideoSegment.
    Video will be downloaded later during rendering.
    """
    import json

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

        # Check if we're updating existing video or adding new one
        existing_proposal = None
        if video_segment.video_proposals:
            for proposal in video_segment.video_proposals:
                if proposal.get("video_url") == video_url or proposal.get("selected"):
                    existing_proposal = proposal
                    break
        
        if existing_proposal:
            # Update existing proposal - preserve all existing data and update with new metadata
            existing_proposal.update({
                "pexels_id": video_metadata.get("id", existing_proposal.get("pexels_id")),
                "pexels_url": video_metadata.get("url", existing_proposal.get("pexels_url")),
                "video_url": video_url,
                "user": video_metadata.get("user", existing_proposal.get("user")),
                "duration": video_metadata.get("duration", existing_proposal.get("duration")),
                "width": video_metadata.get("width", existing_proposal.get("width")),
                "height": video_metadata.get("height", existing_proposal.get("height")),
                "horizontal_mode": video_metadata.get("horizontal_mode", existing_proposal.get("horizontal_mode", "crop")),
                "is_image": video_metadata.get("is_image", existing_proposal.get("is_image", False)),
                "selected": True,
            })
            video_segment.video_proposals = [existing_proposal]
        else:
            # Save new video proposal
            video_segment.video_proposals = [
                {
                    "pexels_id": video_metadata.get("id"),
                    "pexels_url": video_metadata.get("url"),
                    "video_url": video_url,
                    "user": video_metadata.get("user"),
                    "duration": video_metadata.get("duration"),
                    "width": video_metadata.get("width"),
                    "height": video_metadata.get("height"),
                    "horizontal_mode": video_metadata.get("horizontal_mode", "crop"),
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
                "message": "Video URL uspešno shranjen.",
                "video_url": video_url,
            }
        )

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
        messages.error(
            request,
            f"Ne moreš renderirati videa - manjkajo izbrani video klipi ({len(segments_with_urls)}/{total_segments})",
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

    messages.success(request, "Segmenti se ponovno generirajo...")
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
            request, "Ne moreš generirati podnapisov - manjka zvočna datoteka"
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
