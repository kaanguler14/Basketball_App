# utils_fixed.py
import math
import numpy as np
import torch

def get_device():
    """Automatically select devices"""
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


def score(ball_pos, hoop_pos):
    """
    Return True if the ball's predicted path (using two recent points) crosses the rim area.
    More robust than original: checks indexes correctly, avoids division-by-zero and falls back
    to direct-in-box check if fit fails.
    """
    if len(hoop_pos) < 1 or len(ball_pos) < 2:
        return False

    # hoop center and size
    hoop_cx, hoop_cy = hoop_pos[-1][0]
    hoop_w = hoop_pos[-1][2]
    hoop_h = hoop_pos[-1][3]

    rim_height = hoop_cy - 0.5 * hoop_h  # y coordinate (image coords: larger y -> lower)

    # Find a pair of consecutive points where first is above rim and next is below/around rim
    x_pts = []
    y_pts = []
    idx_found = None
    # iterate from newest-1 down to 0 (so we get the most recent crossing)
    for i in range(len(ball_pos) - 2, -1, -1):
        y_above = ball_pos[i][0][1]
        y_next = ball_pos[i + 1][0][1]
        if y_above < rim_height and y_next >= rim_height:
            x_pts = [ball_pos[i][0][0], ball_pos[i + 1][0][0]]
            y_pts = [y_above, y_next]
            idx_found = i
            break

    # If we didn't find consecutive above->below sample last two points (fallback)
    if len(x_pts) < 2:
        # try last two points if they are distinct
        try:
            x_pts = [ball_pos[-2][0][0], ball_pos[-1][0][0]]
            y_pts = [ball_pos[-2][0][1], ball_pos[-1][0][1]]
        except Exception:
            return False

    # If still not enough, bail out
    if len(x_pts) < 2:
        return False

    # Try linear fit (x as independent variable for y = m*x + b)
    try:
        # fit y = m*x + b -> np.polyfit(x,y,1) returns [m,b] if we pass x,y
        m, b = np.polyfit(x_pts, y_pts, 1)
    except Exception:
        return False

    # avoid division by zero (nearly vertical/horizontal)
    if abs(m) < 1e-6:
        # fallback: check if any recent ball center entered rim box
        rim_x1 = hoop_cx - 0.5 * hoop_w
        rim_x2 = hoop_cx + 0.5 * hoop_w
        rim_y1 = hoop_cy - 0.5 * hoop_h
        rim_y2 = hoop_cy + 0.5 * hoop_h
        for center, *_ in ball_pos[-15:]:
            if rim_x1 <= center[0] <= rim_x2 and rim_y1 <= center[1] <= rim_y2:
                return True
        return False

    # predicted x where y == rim_height -> rim_height = m*x + b -> x = (rim_height - b) / m
    predicted_x = (rim_height - b) / m

    rim_x1 = hoop_cx - 0.5 * hoop_w
    rim_x2 = hoop_cx + 0.5 * hoop_w

    # rebound / buffer zone
    hoop_rebound_zone = max(10, int(0.15 * hoop_w))

    if rim_x1 - hoop_rebound_zone <= predicted_x <= rim_x2 + hoop_rebound_zone:
        return True

    # final fallback: check any recent ball centers inside rim bounding box
    for center, *_ in ball_pos[-15:]:
        if rim_x1 <= center[0] <= rim_x2 and (center[1] >= rim_height - hoop_rebound_zone and center[1] <= rim_height + hoop_rebound_zone):
            return True

    return False


def detect_down(ball_pos, hoop_pos):
    """Return True if the latest ball y is sufficiently below the hoop (down phase)."""
    if len(hoop_pos) < 1 or len(ball_pos) < 1:
        return False
    hoop_cy = hoop_pos[-1][0][1]
    hoop_h = hoop_pos[-1][3]
    threshold = hoop_cy + 0.6 * hoop_h
    return ball_pos[-1][0][1] > threshold


def detect_up(ball_pos, hoop_pos):
    """
    Detect if ball is in the 'up' region (near backboard / release area).
    Made more tolerant (wider x-range and y-range).
    """
    if len(hoop_pos) < 1 or len(ball_pos) < 1:
        return False

    hoop_cx, hoop_cy = hoop_pos[-1][0]
    hoop_w, hoop_h = hoop_pos[-1][2], hoop_pos[-1][3]

    x1 = hoop_cx - 6.0 * hoop_w
    x2 = hoop_cx + 6.0 * hoop_w
    y1 = hoop_cy - 3.0 * hoop_h
    y2 = hoop_cy - 0.2 * hoop_h  # not too close to rim center

    bx, by = ball_pos[-1][0]
    return (x1 < bx < x2) and (y1 < by < y2)


def in_hoop_region(center, hoop_pos):
    """Is point near the hoop? (tolerant region)"""
    if len(hoop_pos) < 1:
        return False
    x, y = center
    hoop_cx, hoop_cy = hoop_pos[-1][0]
    hoop_w, hoop_h = hoop_pos[-1][2], hoop_pos[-1][3]

    x1 = hoop_cx - 1.5 * hoop_w
    x2 = hoop_cx + 1.5 * hoop_w
    y1 = hoop_cy - 1.5 * hoop_h
    y2 = hoop_cy + 0.75 * hoop_h

    return (x1 < x < x2) and (y1 < y < y2)


def clean_ball_pos(ball_pos, frame_count):
    """
    Remove noisy/incorrect points and old points.
    - Pop last if it jumps too far from previous
    - Remove non-square detections
    - Remove old points from the front (pop(0))
    """
    if len(ball_pos) > 1:
        w1, h1 = ball_pos[-2][2], ball_pos[-2][3]
        w2, h2 = ball_pos[-1][2], ball_pos[-1][3]
        x1, y1 = ball_pos[-2][0][0], ball_pos[-2][0][1]
        x2, y2 = ball_pos[-1][0][0], ball_pos[-1][0][1]
        f1, f2 = ball_pos[-2][1], ball_pos[-1][1]
        f_dif = f2 - f1

        dist = math.hypot(x2 - x1, y2 - y1)
        max_dist = 4 * math.hypot(w1, h1)

        # Ball should not move > max_dist within few frames
        if dist > max_dist and f_dif < 5:
            ball_pos.pop()  # remove last noisy
        # Ball shape should be roughly square-ish
        elif (w2 * 1.4 < h2) or (h2 * 1.4 < w2):
            ball_pos.pop()

    # remove points older than threshold from front
    while len(ball_pos) > 0 and (frame_count - ball_pos[0][1] > 30):
        ball_pos.pop(0)

    return ball_pos


def clean_hoop_pos(hoop_pos):
    """
    Remove hoop jumps, keep hoop history bounded.
    """
    if len(hoop_pos) > 1:
        x1, y1 = hoop_pos[-2][0][0], hoop_pos[-2][0][1]
        x2, y2 = hoop_pos[-1][0][0], hoop_pos[-1][0][1]
        w1, h1 = hoop_pos[-2][2], hoop_pos[-2][3]
        w2, h2 = hoop_pos[-1][2], hoop_pos[-1][3]
        f1, f2 = hoop_pos[-2][1], hoop_pos[-1][1]
        f_dif = f2 - f1
        dist = math.hypot(x2 - x1, y2 - y1)
        max_dist = 0.5 * math.hypot(w1, h1)

        if dist > max_dist and f_dif < 5:
            hoop_pos.pop()
        if (w2 * 1.3 < h2) or (h2 * 1.3 < w2):
            hoop_pos.pop()

    # cap history length
    while len(hoop_pos) > 25:
        hoop_pos.pop(0)

    return hoop_pos
