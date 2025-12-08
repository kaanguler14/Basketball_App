# utils_fixed.py
# OPTIMIZED VERSION: Print statements removed, performance improved

import math
import numpy as np
import torch
import cv2
import logging

# Logging konfigürasyonu
logger = logging.getLogger(__name__)


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

    rim_height = hoop_cy - 0.5 * hoop_h

    # Find a pair of consecutive points where first is above rim and next is below/around rim
    x_pts = []
    y_pts = []
    
    for i in range(len(ball_pos) - 2, -1, -1):
        y_above = ball_pos[i][0][1]
        y_next = ball_pos[i + 1][0][1]
        if y_above < rim_height and y_next >= rim_height:
            x_pts = [ball_pos[i][0][0], ball_pos[i + 1][0][0]]
            y_pts = [y_above, y_next]
            break

    # Fallback: try last two points
    if len(x_pts) < 2:
        try:
            x_pts = [ball_pos[-2][0][0], ball_pos[-1][0][0]]
            y_pts = [ball_pos[-2][0][1], ball_pos[-1][0][1]]
        except Exception:
            return False

    if len(x_pts) < 2:
        return False

    # Try linear fit
    try:
        m, b = np.polyfit(x_pts, y_pts, 1)
    except Exception:
        return False

    # avoid division by zero
    if abs(m) < 1e-6:
        rim_x1 = hoop_cx - 0.5 * hoop_w
        rim_x2 = hoop_cx + 0.5 * hoop_w
        rim_y1 = hoop_cy - 0.5 * hoop_h
        rim_y2 = hoop_cy + 0.5 * hoop_h
        for center, *_ in ball_pos[-15:]:
            if rim_x1 <= center[0] <= rim_x2 and rim_y1 <= center[1] <= rim_y2:
                return True
        return False

    predicted_x = (rim_height - b) / m

    rim_x1 = hoop_cx - 0.5 * hoop_w
    rim_x2 = hoop_cx + 0.5 * hoop_w
    hoop_rebound_zone = max(10, int(0.15 * hoop_w))

    if rim_x1 - hoop_rebound_zone <= predicted_x <= rim_x2 + hoop_rebound_zone:
        return True

    # final fallback
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

    x1 = hoop_cx - 4.0 * hoop_w
    x2 = hoop_cx + 4.0 * hoop_w
    y1 = hoop_cy - 3.0 * hoop_h
    y2 = hoop_cy - 1.10 * hoop_h

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
    - Remove old points from the front
    
    NOTE: pop(0) is O(n). For better performance, consider using 
    collections.deque with maxlen in the caller.
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

        if dist > max_dist and f_dif < 5:
            ball_pos.pop()
        elif (w2 * 1.4 < h2) or (h2 * 1.4 < w2):
            ball_pos.pop()

    # Remove old points - limit iterations to avoid performance issues
    removed = 0
    max_remove = 10  # Prevent excessive removals in one call
    while len(ball_pos) > 0 and (frame_count - ball_pos[0][1] > 30) and removed < max_remove:
        ball_pos.pop(0)
        removed += 1

    return ball_pos


def clean_hoop_pos(hoop_pos):
    """
    Remove hoop jumps, keep hoop history bounded.
    
    NOTE: pop(0) is O(n). For better performance, consider using 
    collections.deque with maxlen in the caller.
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
        if len(hoop_pos) > 1 and ((w2 * 1.3 < h2) or (h2 * 1.3 < w2)):
            hoop_pos.pop()

    # Cap history length - limit iterations
    removed = 0
    max_remove = 5
    while len(hoop_pos) > 25 and removed < max_remove:
        hoop_pos.pop(0)
        removed += 1

    return hoop_pos
