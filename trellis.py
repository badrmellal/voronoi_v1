import sys

import cv2
import numpy as np

def load_video(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file: " + input_path)
    return cap

# here we will generate multiple candidate states for one frame

def generate_candidate_states(frame):
    dark = cv2.convertScaleAbs(frame, alpha=0.7, beta=0)
    normal = frame.copy()
    bright = cv2.convertScaleAbs(frame, alpha=1.3, beta=0)
    return [dark, normal, bright]

# cost of a single state (mean brightness)
def state_cost(state):
    gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

# penalize sudden changes between frames
def transition_cost(prev_cost, curr_cost):
    return abs(curr_cost - prev_cost)

def trellis_optimization(video_input_path, video_output_path):
    cap = load_video(video_input_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    n_frames = len(frames)
    n_states = 3  # dark, normal, bright that we set before

    # trellis structure
    dp = np.full((n_frames, n_states), np.inf)
    backtrack = np.zeros((n_frames, n_states), dtype=int)

    states = generate_candidate_states(frames[0])
    costs = [state_cost(s) for s in states]

    for s in range(n_states):
        dp[0, s] = costs[s]

    for t in range(1, n_frames):
        curr_states = generate_candidate_states(frames[t])
        curr_costs = [state_cost(s) for s in curr_states]

        prev_states = generate_candidate_states(frames[t - 1])
        prev_costs = [state_cost(s) for s in prev_states]

        for s in range(n_states):
            for ps in range(n_states):
                cost = (
                        dp[t - 1, ps]
                        + transition_cost(prev_costs[ps], curr_costs[s])
                )

                if cost < dp[t, s]:
                    dp[t, s] = cost
                    backtrack[t, s] = ps

    # Backtracking to find the best path
    best_state = np.argmin(dp[-1])
    best_path = [best_state]

    for t in range(n_frames - 1, 0, -1):
        best_state = backtrack[t, best_state]
        best_path.append(best_state)

    best_path.reverse()

    # Optimized video generation
    for t in range(n_frames):
        states = generate_candidate_states(frames[t])
        out.write(states[best_path[t]])

    out.release()
    print("Trellis optimization completed")

if __name__ == "__main__":
    input_path = "/Users/Apple/PycharmProjects/TestOpenCV/video0.mp4"
    output_path = "/Users/Apple/PycharmProjects/TestOpenCV/trellis_output.mp4"

    trellis_optimization(input_path, output_path)