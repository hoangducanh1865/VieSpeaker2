import json

def test_loconet_scores_map_tracks_to_clusters(tmp_path):
    from viespeaker.loconet_adapter import scores_to_segments

    events = json.loads(json.dumps([
        {"frame_idx": 0, "track_id": 10, "score": 0.9},
        {"frame_idx": 5, "track_id": 10, "score": 0.8},
        {"frame_idx": 10, "track_id": 20, "score": 0.2},
    ]))
    segments = scores_to_segments(
        events,
        {3: [10], 4: [20]},
        inference_stride=5,
        min_segment_sec=0.2,
    )

    assert segments == [
        {"start": 0.0, "end": 0.4, "speaker_id": "SPEAKER_03"}
    ]
