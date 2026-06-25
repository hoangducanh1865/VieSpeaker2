"""Small Python 3.9-compatible client for pyannoteAI speaker diarization."""

import hashlib
import time
from pathlib import Path

import requests


class PyannoteAIClient:
    API_URL = "https://api.pyannote.ai/v1"

    def __init__(self, api_key, poll_interval=5, timeout=1800, session=None):
        if not api_key:
            raise ValueError("PYANNOTEAI_API_KEY is required")
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.session = session or requests.Session()
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._get("/test")

    def _get(self, route):
        response = self.session.get(
            f"{self.API_URL}{route}", headers=self.headers, timeout=60
        )
        response.raise_for_status()
        return response

    def _post(self, route, payload):
        response = self.session.post(
            f"{self.API_URL}{route}", json=payload, headers=self.headers, timeout=60
        )
        response.raise_for_status()
        return response

    @staticmethod
    def _media_url(audio_path):
        digest = hashlib.md5()  # nosec B324 - content identifier, not cryptography
        with open(audio_path, "rb") as audio:
            for chunk in iter(lambda: audio.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"media://{digest.hexdigest()}"

    def upload(self, audio_path):
        audio_path = str(Path(audio_path))
        media_url = self._media_url(audio_path)
        presigned_url = self._post("/media/input", {"url": media_url}).json()["url"]
        with open(audio_path, "rb") as audio:
            response = self.session.put(presigned_url, data=audio, timeout=300)
        response.raise_for_status()
        return media_url

    def diarize(self, audio_path, model="precision-2"):
        media_url = self.upload(audio_path)
        response = self._post(
            "/diarize",
            {
                "url": media_url,
                "model": model,
                "confidence": False,
                "exclusive": True,
            },
        )
        job_id = response.json()["jobId"]
        return self.retrieve(job_id)

    def retrieve(self, job_id):
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            job = self._get(f"/jobs/{job_id}").json()
            status = job["status"]
            if status == "succeeded":
                return job
            if status in {"failed", "canceled"}:
                detail = job.get("output", {}).get("error", "no error detail")
                raise RuntimeError(f"pyannoteAI job {status}: {detail} (job {job_id})")
            time.sleep(self.poll_interval)
        raise TimeoutError(f"pyannoteAI job did not finish within {self.timeout}s: {job_id}")
