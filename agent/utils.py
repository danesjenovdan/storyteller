import os
import tempfile
from contextlib import contextmanager
from typing import Generator

import requests
from django.conf import settings
from django.core.files.base import File


@contextmanager
def get_temporary_file_path(file_field: File) -> Generator[str, None, None]:
    # If s3 is not enabled then return the direct path
    if not settings.ENABLE_S3:
        yield file_field.path
        return

    temp_fd = None
    temp_path = None

    try:
        # Create a temporary file
        file_extension = os.path.splitext(file_field.name)[1]
        temp_fd, temp_path = tempfile.mkstemp(suffix=file_extension)

        # Transfer content to temporary file
        with os.fdopen(temp_fd, "wb") as temp_file:
            temp_fd = None  # Prevent closing twice

            # Read from storage and write to temporary file
            file_field.seek(0)  # Go to the beginning of the file
            for chunk in file_field.chunks():
                temp_file.write(chunk)

        yield temp_path

    finally:
        # Cleanup
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except OSError:
                pass

        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


@contextmanager
def get_temporary_file_from_url(file_url: str) -> Generator[str, None, None]:
    """
    Download file from URL and create a temporary file.
    
    Args:
        file_url: URL of the file to download
        
    Yields:
        Path to temporary file
    """
    temp_path = None

    try:
        # Create a temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")

        # Download file from URL
        response = requests.get(file_url, stream=True)
        response.raise_for_status()

        # Write content to temporary file
        with os.fdopen(temp_fd, "wb") as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)

        yield temp_path

    finally:
        # Cleanup
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def get_file_size(file_field: File) -> int:
    """Get file size regardless of storage backend."""
    try:
        return file_field.size
    except (AttributeError, NotImplementedError):
        # Fallback for storage backends that don't support size
        file_field.seek(0, 2)  # Seek to end
        size = file_field.tell()
        file_field.seek(0)  # Reset to beginning
        return size


def file_exists(file_field: File) -> bool:
    """Check if file exists regardless of storage backend."""
    try:
        return file_field.storage.exists(file_field.name)
    except (AttributeError, NotImplementedError):
        # Fallback
        try:
            file_field.size
            return True
        except (ValueError, OSError):
            return False


def ensure_google_api_key():
    """
    Ensure GOOGLE_API_KEY is set in environment variables.

    Raises:
        ValueError: If GOOGLE_API_KEY is not configured in settings
    """
    if not os.environ.get("GOOGLE_API_KEY", None):
        os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY

    if not os.environ.get("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY is not configured in settings")
