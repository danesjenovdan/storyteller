from django.contrib import admin

from agent.models import GenVideo, VideoSegment


class VideoSegmentsInline(admin.TabularInline):
    model = VideoSegment
    extra = 0
    fields = ("order", "text", "query", "start_time", "end_time", "video_file")
    readonly_fields = ("video_file",)


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
