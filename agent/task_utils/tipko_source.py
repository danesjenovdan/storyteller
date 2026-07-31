import base64
import os
from datetime import datetime, timedelta

import requests
from tusclient.client import TusClient
from tusclient.fingerprint import interface
from tusclient.uploader import Uploader


class TokenManager:
    def __init__(self, endpoint: str, username: str, password: str):
        self.endpoint = endpoint
        self.username = username
        self.password = password
        self.access_token = None
        self.refresh_token = None
        self.token_expiry = None

    def get_new_tokens(self):
        response = requests.post(
            f"{self.endpoint}/api/token/",
            json={"username": self.username, "password": self.password},
        )
        if response.status_code in {200, 201}:
            data = response.json()
            self.access_token = data["access"]
            self.refresh_token = data["refresh"]
            self.token_expiry = datetime.now() + timedelta(
                minutes=5
            )  # Assume 5-minute expiry
        else:
            raise Exception("Failed to obtain new tokens")

    def refresh_access_token(self):
        response = requests.post(
            f"{self.endpoint}/api/token/refresh/", json={"refresh": self.refresh_token}
        )
        if response.status_code in {200, 201}:
            data = response.json()
            self.access_token = data["access"]
            self.refresh_token = data["refresh"]
            self.token_expiry = datetime.now() + timedelta(
                minutes=5
            )  # Assume 5-minute expiry
        else:
            self.get_new_tokens()

    def get_auth_header(self):
        if not self.access_token or datetime.now() >= self.token_expiry:
            if self.refresh_token:
                self.refresh_access_token()
            else:
                self.get_new_tokens()
        return {"Authorization": f"Bearer {self.access_token}"}


class Tus:
    route_result_upload = "/api/inference/upload/"

    class Fingerprinter(interface.Fingerprint):
        def __init__(self, fp):
            self.fp = fp

        def get_fingerprint(self, fs):
            return self.fp

    def __init__(self, token_manager: TokenManager, endpoint: str, timeout: int = 10):
        self.timeout = timeout
        self.token_manager = token_manager
        self.endpoint = endpoint.strip("/")
        self.tus_client = TusClient(endpoint + "/upload/")

    def upload(self, path: str):
        if not os.path.isfile(path):
            return None
        fingerprint = str(hash(path))
        response = requests.post(
            self.endpoint + self.route_result_upload,
            json={
                "filename": os.path.basename(path),
                "fingerprint": fingerprint,
            },
            headers=self.token_manager.get_auth_header(),
            timeout=self.timeout,
        ).json()
        remote_file = response.pop("fileId")
        metadata = response.pop("tusMetadata")

        uploader = Uploader(
            file_path=path,
            client=self.tus_client,
            metadata=metadata,
            fingerprinter=self.Fingerprinter(fingerprint),
            chunk_size=100_000_000,
        )
        uploader.upload()
        return remote_file


class Api:
    def __init__(self, endpoint: str, username: str, password: str):
        self.username = username
        self.password = password
        self.endpoint = endpoint
        self.token_manager = TokenManager(endpoint, username, password)
        self.tus = Tus(self.token_manager, endpoint)

    def upload(self, audio: str):
        data = {
            "label": os.path.basename(audio),
            "languageModel": "general",
            "denoise": "off",
            "punctuation": "dictation-and-auto",
            "numbers": {
                "cardinal": {
                    "min": 10,
                },
                "ordinal": {
                    "min": 10,
                },
                "minNumbersForConcat": 5,
            },
            "diarization": "off",
            "dictionaries": ["medical", "legal", "metric"],
            "statusCallbackUrl": None,
            "callbackUrl": None,
            "fileId": self.tus.upload(audio),
        }

        # Make the request
        response = requests.post(
            self.endpoint.strip("/") + "/api/inference/predict/",
            json=data,
            headers=self.token_manager.get_auth_header(),
        )

        # Handle the response
        if response.status_code == requests.codes.ok:
            data = response.json()
            task_id = data["taskId"]
            return task_id
        else:
            raise Exception(f"Request failed with status code {response.status_code}")

    def get_status(self, task_id: str):
        response = requests.get(
            f"{self.endpoint.strip('/')}/api/inference/task/{task_id}/",
            headers=self.token_manager.get_auth_header(),
        )
        if response.status_code == requests.codes.ok:
            return response.json()
        else:
            raise Exception(f"Request failed with status code {response.status_code}")

    def get_transcription_file(self, task_id: str):
        response = requests.get(
            f"{self.endpoint.strip('/')}/api/inference/task/{task_id}/vtt",
            headers=self.token_manager.get_auth_header(),
        )
        if response.status_code == requests.codes.ok:
            return response
        else:
            raise Exception(f"Request failed with status code {response.status_code}")


    def get_srt_file(self, task_id: str):
            response = requests.get(
                f"{self.endpoint.strip('/')}/api/inference/task/{task_id}/srt",
                headers=self.token_manager.get_auth_header(),
            )
            if response.status_code == requests.codes.ok:
                return response
            else:
                raise Exception(f"Request failed with status code {response.status_code}")
