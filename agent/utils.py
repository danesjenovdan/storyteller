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


@contextmanager
def get_temporary_file(file_source: str) -> Generator[str, None, None]:
    """
    Get temporary file from either local path or URL.
    Automatically detects if the source is a local file path or remote URL.

    Args:
        file_source: Either a local file path (e.g. /media/...) or remote URL (http://...)

    Yields:
        Path to temporary file
    """
    from django.core.files.storage import default_storage
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Check if it's a URL (starts with http:// or https://)
    if file_source.startswith(('http://', 'https://')):
        # Remote file - download it
        logger.info(f"Downloading remote file: {file_source}")
        with get_temporary_file_from_url(file_source) as temp_path:
            yield temp_path
        return
    
    # Local file path
    logger.info(f"Processing local file: {file_source}, S3 enabled: {settings.ENABLE_S3}")
    
    # Check if using local storage vs S3
    if not settings.ENABLE_S3:
        # Local storage - file is already on disk
        # Convert /media/... to absolute path
        if file_source.startswith('/media/'):
            absolute_path = os.path.join(settings.MEDIA_ROOT, file_source[7:])  # Remove /media/
            logger.info(f"Local file resolved to: {absolute_path}")
            yield absolute_path
        elif file_source.startswith('media/'):
            absolute_path = os.path.join(settings.MEDIA_ROOT, file_source[6:])  # Remove media/
            logger.info(f"Local file resolved to: {absolute_path}")
            yield absolute_path
        else:
            # Already absolute path
            logger.info(f"Using absolute path: {file_source}")
            yield file_source
    else:
        # S3 storage - need to download to temp file
        logger.info(f"Downloading from S3: {file_source}")
        temp_path = None
        try:
            # Get file from storage
            file_name = file_source.lstrip('/')
            if file_name.startswith('media/'):
                file_name = file_name[6:]  # Remove media/ prefix
            
            file = default_storage.open(file_name, 'rb')
            
            # Create temporary file
            file_extension = os.path.splitext(file_name)[1]
            temp_fd, temp_path = tempfile.mkstemp(suffix=file_extension)
            
            # Write content to temporary file
            with os.fdopen(temp_fd, 'wb') as temp_file:
                for chunk in file.chunks():
                    temp_file.write(chunk)
            
            file.close()
            logger.info(f"S3 file downloaded to: {temp_path}")
            yield temp_path
        finally:
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
