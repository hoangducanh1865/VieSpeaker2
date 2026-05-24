import argparse
import cv2
import os

from face.face_detection_model.SCRFD.nets.nn import FaceDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', default='data/diarization_test_set/video/interview_noise.mp4', help='path to your input video file')
    parser.add_argument('--model', default='weights/model_1.onnx', help='model file path')
    # Added argument for the output directory
    parser.add_argument('--out_dir', default='results', help='folder to save face snapshots')
    args = parser.parse_args()

    detector = FaceDetector(onnx_file=args.model)
    stream = cv2.VideoCapture(args.video_path)

    if not stream.isOpened():
        print("Error opening video stream or file")
        return

    # Create the output directory if it doesn't already exist
    os.makedirs(args.out_dir, exist_ok=True)

    frame_count = 0
    saved_count = 0  # Keep track of how many snapshots we actually save

    while True:
        success, frame = stream.read()

        if success:
            boxes, _ = detector.detect(frame, input_size=(640, 640))
            
            # Check if any faces were detected before drawing and saving
            if boxes is not None and len(boxes) > 0:
                boxes = boxes.astype('int32')
                for box in boxes:
                    x_min, y_min, x_max, y_max, _ = box
                    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 255), 1)

                    cv2.line(frame, (int(x_min), int(y_min)), (int(x_min + 15), int(y_min)), (255, 0, 255), 3)
                    cv2.line(frame, (int(x_min), int(y_min)), (int(x_min), int(y_min + 15)), (255, 0, 255), 3)

                    cv2.line(frame, (int(x_max), int(y_max)), (int(x_max - 15), int(y_max)), (255, 0, 255), 3)
                    cv2.line(frame, (int(x_max), int(y_max)), (int(x_max), int(y_max - 15)), (255, 0, 255), 3)

                    cv2.line(frame, (int(x_max - 15), int(y_min)), (int(x_max), int(y_min)), (255, 0, 255), 3)
                    cv2.line(frame, (int(x_max), int(y_min)), (int(x_max), int(y_min + 15)), (255, 0, 255), 3)

                    cv2.line(frame, (int(x_min), int(y_max - 15)), (int(x_min), int(y_max)), (255, 0, 255), 3)
                    cv2.line(frame, (int(x_min), int(y_max)), (int(x_min + 15), int(y_max)), (255, 0, 255), 3)


            # Keep track of progress in the terminal
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames... Saved {saved_count} snapshots.")

        else:
            break
            
    stream.release()
    print(f"Video processing complete. Saved {saved_count} snapshots to '{args.out_dir}'.")


if __name__ == '__main__':
    main()