"""Pure data conversion helpers for the vendored LoCoNet adapter."""


def scores_to_segments(events, clustered_identities, inference_stride=5,
                       min_segment_sec=0.2, speaker_threshold=0.5, fps=25.0):
    track_to_cluster = {}
    for cluster_id, track_ids in clustered_identities.items():
        for track_id in track_ids:
            track_to_cluster[int(track_id)] = int(cluster_id)

    active = []
    for event in events:
        if float(event.get("score", 0.0)) < speaker_threshold:
            continue
        cluster_id = track_to_cluster.get(int(event["track_id"]))
        if cluster_id is None:
            continue
        start = int(event["frame_idx"]) / fps
        end = (int(event["frame_idx"]) + inference_stride) / fps
        active.append({
            "start": start,
            "end": end,
            "speaker_id": f"SPEAKER_{cluster_id:02d}",
        })

    active.sort(key=lambda item: (item["speaker_id"], item["start"]))
    merged = []
    for item in active:
        if (
            merged
            and merged[-1]["speaker_id"] == item["speaker_id"]
            and item["start"] <= merged[-1]["end"] + 1e-6
        ):
            merged[-1]["end"] = max(merged[-1]["end"], item["end"])
        else:
            merged.append(dict(item))
    return [
        item for item in sorted(merged, key=lambda value: value["start"])
        if item["end"] - item["start"] >= min_segment_sec
    ]
