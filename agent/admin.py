import json
from django.contrib import admin
from django.utils.html import format_html

from agent.models import GenVideo, VideoSegment


class VideoSegmentsInline(admin.TabularInline):
    model = VideoSegment
    extra = 0
    fields = ("order", "text", "query", "start_time", "end_time", "video_preview")
    readonly_fields = ("video_preview",)

    def video_preview(self, obj):
        if obj.video_proposals and len(obj.video_proposals) > 0:
            video_url = obj.video_proposals[0].get("video_url")
            if video_url:
                return format_html(
                    '<video controls style="max-width: 200px; max-height: 300px;">'
                    '<source src="{}" type="video/mp4">'
                    'Your browser does not support video.'
                    '</video>',
                    video_url
                )
            return "No URL"
        else:
            return "No video selected"
    
    video_preview.short_description = "Video"


@admin.register(GenVideo)
class GenVideoAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "user", "status", "created_at")
    search_fields = ("title", "user__username")
    list_filter = ("status", "created_at")
    inlines = [VideoSegmentsInline]


@admin.register(VideoSegment)
class VideoSegmentAdmin(admin.ModelAdmin):
    list_display = ("id", "video__title", "order", "text")
    search_fields = ("video__title", "text")
    list_filter = ("video",)
