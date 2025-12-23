import sys
import cv2
import numpy as np


def load_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        sys.exit(1)

    return cap


def get_details(capture):
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: width={width}, height={height}, fps={fps}, frames={frame_count}")
    return width, height, fps


def create_video_writer(output_path, width, height, fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    if not out.isOpened():
        print(f"Erreur: Impossible de créer le fichier de sortie '{output_path}'")
        sys.exit(1)

    print(f"Output file created: {output_path}")
    return out


def detect_motion(capture, out, threshold = 25):
    ref, previous_frame = capture.read()
    if not ref:
        print("Error reading the first frame")
        return 0
    prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    print("detection begins")
    print(f"threshold limit", {threshold})
    print("Press q to stop")

    frame_count = 0
    motion_detected_count = 0

    while True:
        ref, current_frame = capture.read()
        if not ref:
            print("Error reading the first frame")
            break
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(prev_gray, current_gray)

        # We will apply threshold to isolate motion zones
        # Pixels with difference > threshold become 255 (white), others become 0 (black)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)

        # Then we apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the current frame to draw on
        motion_frame = current_frame.copy()

        # Draw rectangles around detected motion
        motion_detected = False
        for contour in contours:
            # Filter small contours (noise)
            if cv2.contourArea(contour) > 500:  # Minimum area threshold
                motion_detected = True
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                # Draw rectangle
                cv2.rectangle(motion_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if motion_detected:
            motion_detected_count += 1
            # We will add text to indicate motion detected
            cv2.putText(motion_frame, "DETECTED MOUVEMENT", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Then we add frame counter
        cv2.putText(motion_frame, f"Frame: {frame_count}", (10, motion_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Save and display processed frame
        out.write(motion_frame)

        # Display the motion detection result
        cv2.imshow('Mouvement detection', motion_frame)

        cv2.imshow('Threshold', thresh)

        # Update previous frame for next iteration
        prev_gray = current_gray

        frame_count += 1

        # Display progress every 30 frames
        if frame_count % 30 == 0:
            print(f"Processed frames: {frame_count} | detected mouvement: {motion_detected_count}")

        # Check for 'q' key to quit
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print(f"\n User requested to quit after {frame_count} frames")
            break


    print(f" Percentage: {(motion_detected_count / frame_count * 100):.1f}%")

    return frame_count


def main(input_video_path, output_path="output.mp4", threshold=25):
    print("Beginning video processing...")

    cap = load_video(input_video_path)
    width, height, fps = get_details(cap)
    out = create_video_writer(output_path, width, height, fps)

    detect_motion(cap, out, threshold)

    cap.release()
    out.release()
    print("Done.")


if __name__ == "__main__":
    # Configuration
    input_video = "/Users/Apple/PycharmProjects/TestOpenCV/video0.mp4"
    output_video = "motion_detected_output.mp4"
    detection_threshold = 25

    main(input_video, output_video, detection_threshold)
