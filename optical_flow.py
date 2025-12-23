import cv2
import numpy as np


def interpolate_optical_flow(frame1, frame2, alpha=0.5):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                        0.5,
                                        3,
                                        15,
                                        3,
                                        5,
                                        1.2,
                                        0)
    h, w = gray1.shape
    flow = flow * alpha

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)

    interpolated = cv2.remap(
        frame1,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR
    )
    return interpolated

def convert_24_to_30_optical_flow(input_video, output_video):
    cap = cv2.VideoCapture(input_video)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (width, height)
    )

    ret, prev = cap.read()
    if not ret:
        raise RuntimeError("Cannot read video")

    writer.write(prev)

    while True:
        ret, curr = cap.read()
        if not ret:
            break

        mid = interpolate_optical_flow(prev, curr, alpha=0.5)

        writer.write(mid)
        writer.write(curr)

        prev = curr

    cap.release()
    writer.release()


if __name__ == "__main__":
    convert_24_to_30_optical_flow(
        "/Users/Apple/PycharmProjects/TestOpenCV/video0.mp4",
        "/Users/Apple/PycharmProjects/TestOpenCV/output_30fps_opticalflow.mp4"
    )