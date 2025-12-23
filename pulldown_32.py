import cv2

def open_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    return cap

def create_writer(output_path, frame_width, frame_height, fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    return out

def apply_pulldown32(cap, writer):
    pulldown_pattern = [3, 2]
    pulldown_index = 0
    input_frames = 0
    output_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        repeat_count = pulldown_pattern[pulldown_index]
        for _ in range(repeat_count):
            writer.write(frame)
            output_frames += 1

        pulldown_index = (pulldown_index + 1) % len(pulldown_pattern)
        input_frames += 1

    return input_frames, output_frames

def convert_24fps_to_30fps(input_video, output_video):
    cap = open_video(input_video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = create_writer(
        output_video,
        width,
        height,
        fps=30
    )

    in_frames, out_frames = apply_pulldown32(cap, writer)
    cap.release()
    writer.release()

    print(f"Input frames  : {in_frames}")
    print(f"Output frames : {out_frames}")

if __name__ == "__main__":
    convert_24fps_to_30fps(
        input_video="/Users/Apple/PycharmProjects/TestOpenCV/video0.mp4",
        output_video="/Users/Apple/PycharmProjects/TestOpenCV/video0_30fps_pulldown32.mp4"
    )