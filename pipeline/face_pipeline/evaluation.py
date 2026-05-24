import os
import argparse
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

def load_annotation_from_file(file_path, uri="default_audio_id"):
    """
    Reads a space-separated diarization file and returns a pyannote Annotation.
    Expected file format per line: start_time end_time speaker_label
    """
    annotation = Annotation(uri=uri)
    
    # Check if file exists to prevent errors
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find the file: {file_path}")
        
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue 
            
            # Unpack the line (assuming: start end speaker)
            parts = line.split()
            if len(parts) >= 3:
                start = float(parts[0])
                end = float(parts[1])
                speaker = parts[2]
                
                # Add segment to the annotation
                annotation[Segment(start, end)] = speaker
                
    return annotation

def measure_dataset_pollution(reference: Annotation, hypothesis: Annotation):
    """
    Measures how much overlapping/mixed speech will end up in your dataset 
    if you cut audio based on the hypothesis segments.
    """
    total_pollution_time = 0.0
    severe_pollution_time = 0.0
    problematic_segments = []

    # Iterate through every segment predicted by your model
    for hyp_segment, _, hyp_speaker in hypothesis.itertracks(yield_label=True):
        
        # Crop the ground truth to look ONLY at the timeframe of this prediction
        ref_in_this_segment = reference.crop(hyp_segment)
        
        # Get the unique ground-truth speakers that actually spoke during this time
        actual_speakers = ref_in_this_segment.labels()
        
        # If there is more than 1 actual speaker in this single predicted segment,
        # your cut audio will contain multiple voices.
        if len(actual_speakers) > 1:
            
            speaker_durations = {}
            for spk in actual_speakers:
                # Calculate how long each actual speaker spoke during this segment
                spk_timeline = ref_in_this_segment.label_timeline(spk)
                speaker_durations[spk] = spk_timeline.duration()
            
            # Find the dominant speaker (the one who spoke the longest in this clip)
            sorted_durations = sorted(speaker_durations.values(), reverse=True)
            dominant_duration = sorted_durations[0]
            total_speech_duration = sum(sorted_durations)
            
            # The "pollution" is the duration of all non-dominant speakers combined
            impurity_duration = total_speech_duration - dominant_duration
            total_pollution_time += impurity_duration
            if impurity_duration > 0.5:
                severe_pollution_time += impurity_duration
            
            # Save this data so you know which audio cuts to throw away
            problematic_segments.append({
                "time": f"[{hyp_segment.start:.3f} --> {hyp_segment.end:.3f}]",
                "predicted_as": hyp_speaker,
                "actual_speakers": actual_speakers,
                "pollution_duration": impurity_duration
            })

    return total_pollution_time, severe_pollution_time, problematic_segments

def measure_coverage(reference: Annotation, hypothesis: Annotation):
    """
    Measures coverage as hypothesis speech duration divided by the total video duration.
    """
    hypothesis_speech_duration = hypothesis.get_timeline().duration()
    video_duration = reference.get_timeline().extent().duration
    if video_duration == 0:
        return 0.0

    return hypothesis_speech_duration / video_duration

def evaluate_diarization_files(ref_file_path, hyp_file_path):
    """
    Loads reference and hypothesis files and computes the Diarization Error Rate.
    """
    # 1. Define a common URI so pyannote knows these belong to the same audio
    common_uri = "evaluation_session_01"
    
    # 2. Load the text files into Annotation objects
    reference = load_annotation_from_file(ref_file_path, uri=common_uri)
    hypothesis = load_annotation_from_file(hyp_file_path, uri=common_uri)
    hypothesis_speech_duration = hypothesis.get_timeline().duration()
    
    # 3. Initialize the standard Diarization Error Rate metric
    # (Imported directly from pyannote.metrics)
    der_metric = DiarizationErrorRate()
    
    # 4. Compute the final DER score
    der_score = der_metric(reference, hypothesis)
    
    # 5. Extract detailed components (False Alarms, Missed Detections, Confusion)
    components = der_metric.compute_components(reference, hypothesis)
    
    # 6. Print the results clearly
    print(f"--- Evaluation Results ---")
    print(f"Reference File:  {ref_file_path}")
    print(f"Hypothesis File: {hyp_file_path}")
    print(f"Overall DER:     {der_score * 100:.2f}%\n")

    coverage = measure_coverage(reference, hypothesis)
    print(f"Coverage:        {coverage * 100:.2f}%")
    
    print("Detailed Components (in seconds):")
    total = components['total']
    false_alarm_pct = (components['false alarm'] * 100 / total) if total > 0 else 0.0
    missed_detection_pct = (components['missed detection'] * 100 / total) if total > 0 else 0.0
    confusion_pct = (components['confusion'] * 100 / total) if total > 0 else 0.0
    print(f"- Total Speech:      {total:.2f}s (100.00%)")
    print(f"- False Alarms:      {components['false alarm']:.2f}s ({false_alarm_pct:.2f}%)")
    print(f"- Missed Detection:  {components['missed detection']:.2f}s ({missed_detection_pct:.2f}%)")
    print(f"- Speaker Confusion: {components['confusion']:.2f}s ({confusion_pct:.2f}%)")

    pollution_time, severe_pollution_time, bad_segments = measure_dataset_pollution(reference, hypothesis)

    print("=== Dataset Pollution Report ===")
    print(f"Total polluted audio duration: {pollution_time:.2f} seconds\n Total severe polluted audio duration: {severe_pollution_time:.2f} seconds\n")
    print(f"Impurity (threshold=0.0s): {pollution_time * 100 / hypothesis_speech_duration:.2f}%")
    print(f"Impurity (threshold=0.5s): {severe_pollution_time * 100 / hypothesis_speech_duration:.2f}%")

    # if bad_segments:
    #     print("Warning: The following segments contain multiple voices and should be discarded:")
    #     for bad_seg in bad_segments:
    #         print(f" - Segment {bad_seg['time']} predicted as {bad_seg['predicted_as']}")
    #         print(f"   Actual speakers inside: {bad_seg['actual_speakers']}")
    #         print(f"   Amount of overlapping/wrong voice: {bad_seg['pollution_duration']:.2f}s\n")
    # else:
    #     print("Great news! No multi-speaker pollution detected in your segments.")

# ==========================================
# How to use it:
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate diarization results using DER metric and pollution analysis"
    )
    parser.add_argument(
        "--reference_file",
        type=str,
        nargs="?",
        default="data/diarization_test_set/label/interview_noise.txt",
        help="Path to the reference (ground truth) diarization file"
    )
    parser.add_argument(
        "--hypothesis_file",
        type=str,
        nargs="?",
        default="pipeline/face_pipeline/result/result_cleansing/diarization_cleansed.txt",
        help="Path to the hypothesis (predicted) diarization file"
    )
    
    args = parser.parse_args()
    
    evaluate_diarization_files(args.reference_file, args.hypothesis_file)