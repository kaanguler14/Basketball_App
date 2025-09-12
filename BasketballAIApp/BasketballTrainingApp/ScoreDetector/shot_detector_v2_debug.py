# File: shot_detector_v2_debug.py

from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
# utils içindeki fonksiyonları kullanmaya devam ediyoruz ama fallback mantık da var
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos, get_device

class ShotDetector:
    def __init__(self, video_path=None, model_path=None, debug=False):
        self.debug = debug

        # paths (kendi ortamına göre değiştirebilirsin)
        self.model_path = model_path or "D://repos//Basketball_App//BasketballAIApp//Trainings//kagglebest.pt"
        self.video_path = video_path or "D://repos//Basketball_App//BasketballAIApp//clips//training7.mp4"

        # load model
        self.model = YOLO(self.model_path)
        self.device = get_device()

        # Try to read real class names from the model (safe)
        try:
            names = self.model.names
            if isinstance(names, dict):
                max_idx = max(names.keys())
                self.class_names = [names.get(i, "").lower() for i in range(max_idx + 1)]
            else:
                self.class_names = [n.lower() for n in names]
        except Exception:
            self.class_names = ["basketball", "rim"]

        # video capture (or webcam)
        # self.cap = cv2.VideoCapture(0)
        self.cap = cv2.VideoCapture(self.video_path)

        # data arrays
        self.ball_pos = []   # [( (x,y), frame_idx, w, h, conf ), ...]
        self.hoop_pos = []   # [( (x,y), frame_idx, w, h, conf ), ...]

        self.frame_count = 0
        self.frame = None

        self.makes = 0
        self.attempts = 0

        # state for fallback shot detection
        self.prev_ball_y = None
        self.shot_in_progress = False  # to avoid double counting
        self.last_attempt_frame = -999

        # overlay
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)
        self.overlay_text = "waiting.."

        # start
        self.run()

    def run(self):
        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                if self.debug:
                    print("Video ended or cannot read frame.")
                break

            # predict for this frame (predict is simpler and reliable for single frames)
            try:
                results = self.model.predict(self.frame, conf=0.30, device=self.device)
            except Exception as e:
                print("Model predict error:", e)
                break

            # parse detections
            for r in results:
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes:
                    # get bbox
                    try:
                        x1, y1, x2, y2 = box.xyxy[0]
                    except Exception:
                        continue
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    # conf robust
                    try:
                        conf = float(box.conf[0])
                    except Exception:
                        try:
                            conf = float(box.conf)
                        except Exception:
                            conf = 0.0

                    # class id robust
                    try:
                        cls = int(box.cls[0])
                    except Exception:
                        try:
                            cls = int(box.cls)
                        except Exception:
                            cls = -1

                    current_class = self.class_names[cls] if (0 <= cls < len(self.class_names)) else f"class_{cls}"
                    center = (int(x1 + w / 2), int(y1 + h / 2))

                    # draw rects for visualization (use cvzone if available)
                    try:
                        if "basket" in current_class and (conf > 0.15):
                            cvzone.cornerRect(self.frame, (x1, y1, w, h))
                        elif "rim" in current_class and (conf > 0.15):
                            cvzone.cornerRect(self.frame, (x1, y1, w, h))
                    except Exception:
                        # fallback draw
                        if "basket" in current_class:
                            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        elif "rim" in current_class:
                            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # append ball pos (use more permissive conf threshold near hoop)
                    if ("basket" in current_class) and (conf > 0.25 or (in_hoop_region(center, self.hoop_pos) and conf > 0.12)):
                        self.ball_pos.append((center, self.frame_count, w, h, conf))
                        if self.debug:
                            print(f"[F{self.frame_count}] Ball detected {center} conf={conf:.2f}")

                    # append hoop pos
                    if ("rim" in current_class or "hoop" in current_class) and conf > 0.35:
                        self.hoop_pos.append((center, self.frame_count, w, h, conf))
                        if self.debug:
                            print(f"[F{self.frame_count}] Rim detected {center} conf={conf:.2f}")

            # clean and draw motions
            self.clean_motion()

            # shot detection (try util functions first; if they don't detect, fallback)
            detected_via_utils = self.try_utils_detection()
            if not detected_via_utils:
                self.fallback_detection()

            # display overlay and counters
            self.display_score()

            self.frame_count += 1

            cv2.imshow("Frame", self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def clean_motion(self):
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        for i in range(len(self.ball_pos)):
            cv2.circle(self.frame, self.ball_pos[i][0], 2, (0, 0, 255), 2)

        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)

    def try_utils_detection(self):
        """
        Call your utils-based detection (detect_up / detect_down / score).
        Return True if utils detected and incremented counts (so fallback not needed).
        """
        used = False
        try:
            # old logic: detect up/down + count every 10 frames
            if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
                up = detect_up(self.ball_pos, self.hoop_pos)
                down = detect_down(self.ball_pos, self.hoop_pos)
                if self.debug:
                    print(f"[UTILS] detect_up={up}, detect_down={down}")
                # If utils finds a valid up->down in recent frames, rely on it.
                if up and down:
                    # check ordering via frames
                    up_frame = None
                    down_frame = None
                    # find frames for last up/down
                    try:
                        # find last ball frame when detect_up was true: approximate with last ball frame
                        up_frame = self.ball_pos[-1][1]
                        down_frame = self.ball_pos[-1][1]
                    except Exception:
                        up_frame = self.frame_count
                        down_frame = self.frame_count
                    # only count if up_frame < down_frame (safe-guard)
                    if up_frame <= down_frame and (self.frame_count - self.last_attempt_frame) > 5:
                        self.attempts += 1
                        self.last_attempt_frame = self.frame_count
                        used = True
                        if score(self.ball_pos, self.hoop_pos):
                            self.makes += 1
                            self.overlay_color = (0, 255, 0)
                            self.overlay_text = "Score"
                            self.fade_counter = self.fade_frames
                            if self.debug:
                                print(f"[UTILS] SCORE counted at frame {self.frame_count}")
                        else:
                            self.overlay_color = (0, 0, 255)
                            self.overlay_text = "Miss"
                            self.fade_counter = self.fade_frames
                            if self.debug:
                                print(f"[UTILS] MISS counted at frame {self.frame_count}")
        except Exception as e:
            if self.debug:
                print("Utils detection error:", e)
            used = False

        return used

    def fallback_detection(self):
        """
        Simple heuristic:
         - When ball goes from above hoop area to below hoop area (downward), count an attempt.
         - If ball passes through rim bbox during that passage, count as make.
        """
        if len(self.hoop_pos) == 0 or len(self.ball_pos) < 2:
            return

        # latest hoop
        hoop_cx, hoop_cy = self.hoop_pos[-1][0]
        hoop_w = max(20, int(self.hoop_pos[-1][2]))
        hoop_h = max(6, int(self.hoop_pos[-1][3]))

        rim_x1 = hoop_cx - hoop_w // 2
        rim_x2 = hoop_cx + hoop_w // 2
        rim_y1 = hoop_cy - hoop_h // 2
        rim_y2 = hoop_cy + hoop_h // 2

        # thresholds relative to hoop vertical position
        upper_thresh = hoop_cy - max(20, hoop_h)
        lower_thresh = hoop_cy + max(20, hoop_h)

        # latest two ball points
        prev_center, prev_frame, _, _, _ = self.ball_pos[-2]
        curr_center, curr_frame, _, _, _ = self.ball_pos[-1]
        prev_y = prev_center[1]
        curr_y = curr_center[1]

        # debug
        if self.debug:
            print(f"[FALLBACK] prev_y={prev_y}, curr_y={curr_y}, hoop_y={hoop_cy}, last_attempt_frame={self.last_attempt_frame}")

        # detect downward crossing (from above upper_thresh to below lower_thresh)
        if (prev_y < upper_thresh) and (curr_y > lower_thresh) and (curr_y > prev_y) and (curr_frame - self.last_attempt_frame) > 5:
            # New attempt
            self.attempts += 1
            self.last_attempt_frame = curr_frame
            self.shot_in_progress = True
            if self.debug:
                print(f"[FALLBACK] Attempt counted at frame {curr_frame}. prev_y={prev_y} curr_y={curr_y}")

            # Check if any recent ball positions during this passage went inside rim box -> make
            made = False
            # examine recent ball_pos entries from prev_frame..curr_frame (use last 15 entries)
            for center, fnum, *_ in self.ball_pos[-30:]:
                cx, cy = center
                if rim_x1 <= cx <= rim_x2 and rim_y1 <= cy <= rim_y2:
                    made = True
                    break

            if made:
                self.makes += 1
                self.overlay_color = (0, 255, 0)
                self.overlay_text = "Score"
                if self.debug:
                    print(f"[FALLBACK] MAKE detected at frame {curr_frame}")
            else:
                self.overlay_color = (0, 0, 255)
                self.overlay_text = "Miss"
                if self.debug:
                    print(f"[FALLBACK] MISS detected at frame {curr_frame}")

            self.fade_counter = self.fade_frames

        # reset shot_in_progress when ball is far away (so future shots can be counted)
        if self.shot_in_progress:
            # if ball is well below hoop or well above hoop, consider finished
            if curr_y > (lower_thresh + 40) or curr_y < (upper_thresh - 40):
                self.shot_in_progress = False
                if self.debug:
                    print("[FALLBACK] shot_in_progress reset")

    def display_score(self):
        # counter overlay
        text = f"{self.makes}/{self.attempts}"
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        # overlay text for shot result
        if hasattr(self, "overlay_text") and self.fade_counter > 0:
            (text_width, text_height), _ = cv2.getTextSize(self.overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 6)
            text_x = max(self.frame.shape[1] - text_width - 40, 10)
            text_y = 100
            cv2.putText(self.frame, self.overlay_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3, self.overlay_color, 6)

        # fade overlay
        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            overlay = np.full(self.frame.shape, self.overlay_color, dtype=np.uint8)
            try:
                self.frame = cv2.addWeighted(self.frame, 1 - alpha, overlay, alpha, 0)
            except Exception:
                pass
            self.fade_counter -= 1

if __name__ == "__main__":
    # debug=True ile çalıştır, console'a kısa loglar yazsın
    sd = ShotDetector(debug=True)
