from viespeaker.pyannoteai_client import PyannoteAIClient


class Response:
    def __init__(self, payload=None):
        self.payload = payload or {}

    def json(self):
        return self.payload

    def raise_for_status(self):
        return None


class Session:
    def __init__(self):
        self.calls = []
        self.jobs = iter(
            [
                {"status": "running"},
                {
                    "status": "succeeded",
                    "output": {
                        "diarization": [
                            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}
                        ]
                    },
                },
            ]
        )

    def get(self, url, **kwargs):
        self.calls.append(("get", url, kwargs))
        if url.endswith("/test"):
            return Response({"status": "OK"})
        return Response(next(self.jobs))

    def post(self, url, **kwargs):
        self.calls.append(("post", url, kwargs))
        if url.endswith("/media/input"):
            return Response({"url": "https://upload.example.test"})
        return Response({"jobId": "job-123"})

    def put(self, url, **kwargs):
        self.calls.append(("put", url, kwargs))
        return Response()


def test_cloud_diarization_flow(tmp_path, monkeypatch):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    session = Session()
    monkeypatch.setattr("viespeaker.pyannoteai_client.time.sleep", lambda _: None)

    client = PyannoteAIClient(
        "secret", poll_interval=0, timeout=10, session=session
    )
    result = client.diarize(audio)

    assert result["status"] == "succeeded"
    diarize_call = next(
        call for call in session.calls if call[0] == "post" and call[1].endswith("/diarize")
    )
    assert diarize_call[2]["json"]["model"] == "precision-2"
    assert diarize_call[2]["json"]["exclusive"] is True
