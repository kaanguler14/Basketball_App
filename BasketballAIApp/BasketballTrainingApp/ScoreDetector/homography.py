import numpy as np
import pointSelection as ps
import cv2

def compute_homography(video_points_dict,minimap_points_dict):
    common_labels = [l for l in ps.court_labels if l in video_points_dict and l in minimap_points_dict]
    if len(common_labels) < 4:
        raise ValueError("Homography için en az 4 ortak nokta gerekli!")
    video_pts = np.array([video_points_dict[l] for l in common_labels], dtype=np.float32)
    minimap_pts = np.array([minimap_points_dict[l] for l in common_labels], dtype=np.float32)

    H1, err1, proj1, errs1 = compute_h_and_error(video_pts, minimap_pts)
    h_img = ps.minimap_img.shape[0]
    minimap_pts_flipped = np.array([[x, h_img - y] for (x, y) in minimap_pts], dtype=np.float32)
    H2, err2, proj2, errs2 = compute_h_and_error(video_pts, minimap_pts_flipped)

    use_flip = False
    H, proj, errs, err = H1, proj1, errs1, err1
    if H1 is None and H2 is None:
        raise ValueError("Homography hesaplanamadı")
    elif H1 is None or (err2 + 1.0 < err1):
        use_flip = True
        H, proj, errs, err = H2, proj2, errs2, err2
    print(f"Seçilen dönüşüm: flip_y={use_flip}, ortalama reproj hatası={err:.2f}")
    return H, use_flip, h_img


def compute_h_and_error(video_pts, minimap_pts):
    H, mask = cv2.findHomography(video_pts, minimap_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None, float('inf'), None, None
    proj = cv2.perspectiveTransform(video_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    errs = np.linalg.norm(proj - minimap_pts, axis=1)
    return H, float(np.mean(errs)), proj, errs