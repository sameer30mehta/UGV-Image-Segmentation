"""
TerrainAI Backend — Velocity Estimator
Estimates UGV velocity from consecutive video frames using sparse optical flow
(Lucas-Kanade). The estimated velocity drives adaptive model selection:
  - Low velocity  (< 15 px/s)  → mit_b3  (accurate model)
  - Mid velocity  (< 40 px/s)  → mit_b1  (balanced model)
  - High velocity (>= 40 px/s) → mit_b0  (fast / real-time model)
"""
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Optical-flow velocity estimation
# ---------------------------------------------------------------------------

def estimate_velocity(prev_frame_rgb, curr_frame_rgb, dt=0.2):
    """
    Estimate apparent motion (pixel velocity) between two consecutive RGB frames
    using Lucas-Kanade sparse optical flow on the lower 35 % of the image
    (road / ground region most relevant for UGV navigation).

    Parameters
    ----------
    prev_frame_rgb : np.ndarray  (H, W, 3) uint8, RGB
        Previous frame.
    curr_frame_rgb : np.ndarray  (H, W, 3) uint8, RGB
        Current frame.
    dt : float
        Time delta between the two frames in seconds (e.g. 0.2 for 200 ms).

    Returns
    -------
    velocity : float
        Estimated velocity in *pixels per second*.
    avg_displacement : float
        Average displacement in pixels between the two frames.
    num_good_points : int
        Number of successfully tracked feature points.
    """
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame_rgb, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_frame_rgb, cv2.COLOR_RGB2GRAY)

    h, w = prev_gray.shape

    # Region of interest: bottom 35 %, central 60 % — captures road / ground
    y_start, y_end = int(0.65 * h), h
    x_start, x_end = int(0.20 * w), int(0.80 * w)

    prev_roi = prev_gray[y_start:y_end, x_start:x_end]
    curr_roi = curr_gray[y_start:y_end, x_start:x_end]

    # Shi-Tomasi corner detection parameters
    feature_params = dict(
        maxCorners=300,
        qualityLevel=0.2,
        minDistance=5,
        blockSize=7,
    )

    prev_pts = cv2.goodFeaturesToTrack(prev_roi, mask=None, **feature_params)
    if prev_pts is None or len(prev_pts) == 0:
        return 0.0, 0.0, 0

    # Lucas-Kanade optical flow parameters
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    curr_pts, status, _error = cv2.calcOpticalFlowPyrLK(
        prev_roi, curr_roi, prev_pts, None, **lk_params
    )

    good_old = prev_pts[status.ravel() == 1]
    good_new = curr_pts[status.ravel() == 1]

    if len(good_old) == 0:
        return 0.0, 0.0, 0

    displacements = np.sqrt(
        (good_new[:, 0] - good_old[:, 0]) ** 2
        + (good_new[:, 1] - good_old[:, 1]) ** 2
    )

    avg_displacement = float(np.mean(displacements))
    velocity = avg_displacement / dt  # pixels per second

    return velocity, avg_displacement, int(len(good_old))


# ---------------------------------------------------------------------------
# Adaptive model selection based on velocity
# ---------------------------------------------------------------------------

# Thresholds (pixels / second) — tweak as needed for your camera / resolution
VELOCITY_LOW = 15.0   # Below this → scene is nearly static → use best model
VELOCITY_HIGH = 40.0  # Above this → fast motion → use lightest model


def select_model_by_velocity(velocity):
    """
    Choose the best model key based on estimated velocity.

    Returns
    -------
    model_key : str
        One of 'mit_b3', 'mit_b1', 'mit_b0'.
    tier : str
        Human-readable label: 'accurate', 'balanced', or 'fast'.
    """
    if velocity < VELOCITY_LOW:
        return "mit_b3", "accurate"
    elif velocity < VELOCITY_HIGH:
        return "mit_b1", "balanced"
    else:
        return "mit_b0", "fast"


def estimate_velocities_for_frames(frames_rgb, dt=0.2):
    """
    Compute per-frame velocity for a list of RGB frames sampled at interval *dt*.

    The first frame has no predecessor so its velocity is set to 0.

    Returns
    -------
    velocities : list[dict]
        One entry per frame with keys:
          velocity, displacement, tracked_points, model_key, model_tier
    """
    n = len(frames_rgb)
    results = []

    for i in range(n):
        if i == 0:
            vel, disp, pts = 0.0, 0.0, 0
        else:
            vel, disp, pts = estimate_velocity(frames_rgb[i - 1], frames_rgb[i], dt=dt)

        model_key, tier = select_model_by_velocity(vel)

        results.append({
            "frame_index": i,
            "velocity": round(vel, 2),
            "displacement_px": round(disp, 2),
            "tracked_points": pts,
            "model_key": model_key,
            "model_tier": tier,
        })

    return results
