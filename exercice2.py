import sys
import numpy as np
import cv2


def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        sys.exit(1)
    else:
        print("Video is opened")
        return cap


def read_video_properties(capture):
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video properties: width={width}, height={height}, fps={fps}, frames={frame_count}")
    return width, height, fps, frame_count


def apply_voronoi_to_frame(frame, max_points=200):
    h, w = frame.shape[:2]

    # Convert frame to grayscale for feature detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect feature points (corners) to use as Voronoi seeds
    points = cv2.goodFeaturesToTrack(gray, max_points, 0.01, 10)

    if points is None:
        return frame

    # Convert points to integer tuples
    points = [(int(p[0][0]), int(p[0][1])) for p in points]

    # Create Voronoi structure
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in points:
        # Ensure points are within bounds to avoid C++ errors
        if 0 <= p[0] < w and 0 <= p[1] < h:
            subdiv.insert(p)

    # Get Voronoi cells
    facets, centers = subdiv.getVoronoiFacetList([])
    voronoi_frame = np.zeros_like(frame)

    # Draw colored cells
    for i in range(len(facets)):
        facet = facets[i]
        center = centers[i]

        pts = np.array(facet, np.int32)

        # Instead of masking (slow), sample the color at the center of the cell
        cx, cy = int(center[0]), int(center[1])

        # Clamp coordinates to frame dimensions
        cx = min(max(cx, 0), w - 1)
        cy = min(max(cy, 0), h - 1)

        # Get color from original frame at the center point
        b, g, r = frame[cy, cx]
        color = (int(b), int(g), int(r))

        # Fill the cell
        cv2.fillPoly(voronoi_frame, [pts], color)

        # Draw the point (seed) on top
        cv2.circle(voronoi_frame, (cx, cy), 2, (0, 0, 0), -1)

    return voronoi_frame


def apply_voronoi_to_video(input_path, output_path, points=200):
    cap = load_video(input_path)

    # Unpack tuple instead of accessing attributes like .height
    w, h, fps, frame_count = read_video_properties(cap)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print("Processing video... Press Ctrl+C to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        voronoi_frame = apply_voronoi_to_frame(frame, points)
        out.write(voronoi_frame)

    cap.release()
    out.release()
    print("Processing complete.")


def main(video_path):
    cap = load_video(video_path)

    # Unpack tuple properties
    w, h, fps, frame_count = read_video_properties(cap)
    output = cv2.VideoWriter('voronoi_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        voronoi_frame = apply_voronoi_to_frame(frame)

        # Write to file
        output.write(voronoi_frame)

        # Show on screen
        cv2.imshow('Voronoi Effect', voronoi_frame)

        # Added waitKey to update window and handle exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    output.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    v_p = "/Users/Apple/PycharmProjects/TestOpenCV/motion_detected_output.mp4"
    main(v_p)