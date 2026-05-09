import os
import cv2
import json
import numpy as np
import pandas as pd
import mediapipe as mp
from dataclasses import dataclass, asdict

# =========================================================
# MediaPipe Tasks imports
# =========================================================
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================================================
# Config
# =========================================================

VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv"]

# Landmark indices in MediaPipe hand:
# wrist=0
# thumb: 1,2,3,4
# index: 5,6,7,8
# middle: 9,10,11,12

# =========================================================
# Data classes
# =========================================================

@dataclass
class FrameFeatures:
    frame_id: int
    f_thumb: float
    f_index: float
    f_middle: float
    d_ti: float
    d_tm: float
    d_im: float
    area_tri: float
    F: float
    D: float

@dataclass
class Keyframes:
    shape_start: int
    preshape: int
    enclosure: int
    final: int

# =========================================================
# Utility
# =========================================================

def norm(v):
    return np.linalg.norm(v)

def angle_between(v1, v2, eps=1e-8):
    v1n = norm(v1) + eps
    v2n = norm(v2) + eps
    c = np.dot(v1, v2) / (v1n * v2n)
    c = np.clip(c, -1.0, 1.0)
    return np.arccos(c)

def triangle_area(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    return 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])

def smooth_array(x, win=7):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < win or win < 3:
        return x.copy()
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(x, kernel, mode='same')

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# =========================================================
# Finger feature extraction
# =========================================================

def finger_flexion(p0, p1, p2, p3, base=None, thumb=False):
    v1 = p1 - p0
    v2 = p2 - p1
    v3 = p3 - p2

    a1 = angle_between(v1, v2)
    a2 = angle_between(v2, v3)

    if base is not None and not thumb:
        vb = p0 - base
        a0 = angle_between(vb, v1)
        return 0.3 * a0 + 0.5 * a1 + 0.2 * a2
    else:
        return 0.5 * a1 + 0.5 * a2

def landmarks_to_hand_dict(hand_landmarks, img_w, img_h):
    pts = []
    for lm in hand_landmarks:
        x = lm.x * img_w
        y = lm.y * img_h
        pts.append(np.array([x, y], dtype=np.float32))

    return {
        "W":  pts[0],
        "T0": pts[1],
        "T1": pts[2],
        "T2": pts[3],
        "T3": pts[4],
        "I0": pts[5],
        "I1": pts[6],
        "I2": pts[7],
        "I3": pts[8],
        "M0": pts[9],
        "M1": pts[10],
        "M2": pts[11],
        "M3": pts[12],
    }

def extract_three_finger_features(hand_kpts, frame_id):
    W = hand_kpts["W"]

    T0, T1, T2, T3 = hand_kpts["T0"], hand_kpts["T1"], hand_kpts["T2"], hand_kpts["T3"]
    I0, I1, I2, I3 = hand_kpts["I0"], hand_kpts["I1"], hand_kpts["I2"], hand_kpts["I3"]
    M0, M1, M2, M3 = hand_kpts["M0"], hand_kpts["M1"], hand_kpts["M2"], hand_kpts["M3"]

    palm_scale = norm(M0 - I0) + 1e-6

    f_thumb = finger_flexion(T0, T1, T2, T3, thumb=True)
    f_index = finger_flexion(I0, I1, I2, I3, base=W, thumb=False)
    f_middle = finger_flexion(M0, M1, M2, M3, base=W, thumb=False)

    d_ti = norm(T3 - I3) / palm_scale
    d_tm = norm(T3 - M3) / palm_scale
    d_im = norm(I3 - M3) / palm_scale

    area_tri = triangle_area(T3, I3, M3) / (palm_scale ** 2)

    F = 0.4 * f_thumb + 0.3 * f_index + 0.3 * f_middle
    D = (d_ti + d_tm + d_im) / 3.0

    return FrameFeatures(
        frame_id=frame_id,
        f_thumb=float(f_thumb),
        f_index=float(f_index),
        f_middle=float(f_middle),
        d_ti=float(d_ti),
        d_tm=float(d_tm),
        d_im=float(d_im),
        area_tri=float(area_tri),
        F=float(F),
        D=float(D),
    )

# =========================================================
# Keyframe extraction (no contact)
# =========================================================

def extract_keyframes_no_contact(feature_list):
    A = np.array([f.area_tri for f in feature_list], dtype=np.float32)
    D = np.array([f.D for f in feature_list], dtype=np.float32)
    F = np.array([f.F for f in feature_list], dtype=np.float32)

    A_s = smooth_array(A, win=7)
    D_s = smooth_array(D, win=7)
    F_s = smooth_array(F, win=7)

    dA = np.diff(A_s, prepend=A_s[0])
    dD = np.diff(D_s, prepend=D_s[0])
    dF = np.diff(F_s, prepend=F_s[0])

    drive = 0.4 * (-dA) + 0.3 * (-dD) + 0.3 * (dF)
    tau = float(np.mean(drive) + 0.5 * np.std(drive))

    k = 3
    kf1 = 0
    for t in range(len(drive) - k):
        if np.all(drive[t:t+k] > tau):
            kf1 = t
            break

    tail_start = int(0.8 * len(A_s))
    stability = np.abs(dA) + np.abs(dD) + np.abs(dF)
    if tail_start >= len(stability):
        tail_start = max(0, len(stability) - 1)
    kf4 = tail_start + int(np.argmin(stability[tail_start:]))

    if kf4 <= kf1 + 2:
        kf4 = len(A_s) - 1

    eps = 1e-6
    pA = (A_s[kf1] - A_s) / (A_s[kf1] - A_s[kf4] + eps)
    pD = (D_s[kf1] - D_s) / (D_s[kf1] - D_s[kf4] + eps)
    pF = (F_s - F_s[kf1]) / (F_s[kf4] - F_s[kf1] + eps)

    progress = 0.4 * pA + 0.3 * pD + 0.3 * pF
    progress = np.clip(progress, 0.0, 1.2)

    valid = np.arange(kf1, kf4 + 1)
    kf2 = valid[np.argmin(np.abs(progress[valid] - 0.4))]
    kf3 = valid[np.argmin(np.abs(progress[valid] - 0.75))]

    kf2 = max(int(kf2), kf1 + 1)
    kf3 = max(int(kf3), kf2 + 1)
    kf4 = max(int(kf4), kf3 + 1)

    return Keyframes(
        shape_start=int(kf1),
        preshape=int(kf2),
        enclosure=int(kf3),
        final=int(kf4),
    ), {
        "A_s": A_s.tolist(),
        "D_s": D_s.tolist(),
        "F_s": F_s.tolist(),
        "progress": progress.tolist(),
        "drive": drive.tolist(),
    }

# =========================================================
# Visualization
# =========================================================

def draw_selected_points(frame, hand_kpts, color=(0, 255, 0)):
    names = ["W", "T0", "T1", "T2", "T3", "I0", "I1", "I2", "I3", "M0", "M1", "M2", "M3"]
    for n in names:
        p = hand_kpts[n].astype(int)
        cv2.circle(frame, tuple(p), 3, color, -1)

    chains = [
        ["T0", "T1", "T2", "T3"],
        ["I0", "I1", "I2", "I3"],
        ["M0", "M1", "M2", "M3"],
    ]
    for chain in chains:
        for i in range(len(chain) - 1):
            p1 = hand_kpts[chain[i]].astype(int)
            p2 = hand_kpts[chain[i+1]].astype(int)
            cv2.line(frame, tuple(p1), tuple(p2), color, 2)

    T3 = hand_kpts["T3"].astype(int)
    I3 = hand_kpts["I3"].astype(int)
    M3 = hand_kpts["M3"].astype(int)
    cv2.line(frame, tuple(T3), tuple(I3), (255, 0, 0), 2)
    cv2.line(frame, tuple(T3), tuple(M3), (255, 0, 0), 2)
    cv2.line(frame, tuple(I3), tuple(M3), (255, 0, 0), 2)

# =========================================================
# Tasks hand detector
# =========================================================

def create_hand_landmarker(model_path, num_hands=1):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)
    return detector

# =========================================================
# Video processing
# =========================================================

def process_video(
    video_path,
    output_dir,
    model_path,
    grasp_mode="three_finger_grasp",
    object_size=None,
    save_debug_video=False,
    num_hands=1
):
    if object_size is None:
        object_size = {"size_scalar": None}

    ensure_dir(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return None

    basename = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(output_dir, basename + "_keyframes")
    ensure_dir(frames_dir)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    debug_writer = None
    if save_debug_video:
        debug_path = os.path.join(output_dir, basename + "_debug.mp4")
        debug_writer = cv2.VideoWriter(
            debug_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )

    detector = create_hand_landmarker(model_path, num_hands=num_hands)

    feature_list = []
    raw_frames = []
    hand_kpts_list = []

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        raw_frames.append(frame.copy())
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_result = detector.detect(mp_image)

        if detection_result.hand_landmarks and len(detection_result.hand_landmarks) > 0:
            hand_landmarks = detection_result.hand_landmarks[0]
            hand_kpts = landmarks_to_hand_dict(hand_landmarks, w, h)
            hand_kpts_list.append(hand_kpts)

            feat = extract_three_finger_features(hand_kpts, frame_id)
            feature_list.append(feat)

            vis = frame.copy()
            draw_selected_points(vis, hand_kpts, color=(0, 255, 0))
            if debug_writer is not None:
                debug_writer.write(vis)
        else:
            hand_kpts_list.append(None)
            feature_list.append(None)
            if debug_writer is not None:
                debug_writer.write(frame)

        frame_id += 1

    cap.release()
    if debug_writer is not None:
        debug_writer.release()
    detector.close()

    valid_indices = [i for i, x in enumerate(feature_list) if x is not None]
    if len(valid_indices) < 10:
        print(f"[WARN] Too few valid hand frames in {video_path}")
        return None

    valid_features = [feature_list[i] for i in valid_indices]
    valid_hand_kpts = [hand_kpts_list[i] for i in valid_indices]
    valid_raw_frames = [raw_frames[i] for i in valid_indices]

    for i, feat in enumerate(valid_features):
        feat.frame_id = i

    keyframes, aux = extract_keyframes_no_contact(valid_features)

    kf_map = {
        "shape_start": keyframes.shape_start,
        "preshape": keyframes.preshape,
        "enclosure": keyframes.enclosure,
        "final": keyframes.final,
    }

    for name, idx in kf_map.items():
        frame = valid_raw_frames[idx].copy()
        hand_kpts = valid_hand_kpts[idx]
        draw_selected_points(frame, hand_kpts, color=(0, 255, 0))
        cv2.putText(
            frame,
            f"{name}: {idx}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )
        cv2.imwrite(os.path.join(frames_dir, f"{name}_{idx:04d}.jpg"), frame)

    df = pd.DataFrame([asdict(f) for f in valid_features])
    df["A_s"] = aux["A_s"]
    df["D_s"] = aux["D_s"]
    df["F_s"] = aux["F_s"]
    df["progress"] = aux["progress"]
    df["drive"] = aux["drive"]

    csv_path = os.path.join(output_dir, basename + "_features.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    fs = valid_features[keyframes.shape_start]
    fp = valid_features[keyframes.preshape]
    fe = valid_features[keyframes.enclosure]
    ff = valid_features[keyframes.final]

    summary = {
        "video_path": video_path,
        "grasp_mode": grasp_mode,
        "object_size": object_size,
        "num_valid_frames": len(valid_features),
        "keyframes": asdict(keyframes),
        "shape_start_params": {
            "A": fs.area_tri,
            "D": fs.D,
            "F": fs.F,
            "f_thumb": fs.f_thumb,
            "f_index": fs.f_index,
            "f_middle": fs.f_middle,
        },
        "preshape_params": {
            "A": fp.area_tri,
            "D": fp.D,
            "F": fp.F,
            "f_thumb": fp.f_thumb,
            "f_index": fp.f_index,
            "f_middle": fp.f_middle,
        },
        "enclosure_params": {
            "A": fe.area_tri,
            "D": fe.D,
            "F": fe.F,
            "f_thumb": fe.f_thumb,
            "f_index": fe.f_index,
            "f_middle": fe.f_middle,
        },
        "final_params": {
            "A": ff.area_tri,
            "D": ff.D,
            "F": ff.F,
            "f_thumb": ff.f_thumb,
            "f_index": ff.f_index,
            "f_middle": ff.f_middle,
        },
        "timing_params": {
            "T_total": int(keyframes.final - keyframes.shape_start),
            "T_start_pre": int(keyframes.preshape - keyframes.shape_start),
            "T_pre_enc": int(keyframes.enclosure - keyframes.preshape),
            "T_enc_final": int(keyframes.final - keyframes.enclosure),
        }
    }

    json_path = os.path.join(output_dir, basename + "_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] Processed: {video_path}")
    print(f"     CSV : {csv_path}")
    print(f"     JSON: {json_path}")
    return summary

# =========================================================
# Batch
# =========================================================

def parse_object_size_from_filename(filename):
    """
    例如:
    tripod_size_30.mp4
    tripod_size_45.mp4
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split("_")
    size_scalar = None
    for i, p in enumerate(parts):
        if p.lower() == "size" and i + 1 < len(parts):
            try:
                size_scalar = float(parts[i + 1])
            except:
                pass
    return {"size_scalar": size_scalar}

def process_path(input_path, output_dir, model_path, grasp_mode="three_finger_grasp", save_debug_video=False):
    ensure_dir(output_dir)

    if os.path.isfile(input_path):
        object_size = parse_object_size_from_filename(input_path)
        process_video(
            input_path,
            output_dir,
            model_path=model_path,
            grasp_mode=grasp_mode,
            object_size=object_size,
            save_debug_video=save_debug_video
        )
    else:
        files = sorted(os.listdir(input_path))
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in VIDEO_EXTS:
                vp = os.path.join(input_path, fn)
                object_size = parse_object_size_from_filename(vp)
                process_video(
                    vp,
                    output_dir,
                    model_path=model_path,
                    grasp_mode=grasp_mode,
                    object_size=object_size,
                    save_debug_video=save_debug_video
                )

# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="video file or directory")
    parser.add_argument("--output", type=str, required=True, help="output directory")
    parser.add_argument("--model", type=str, required=True, help="path to hand_landmarker.task")
    parser.add_argument("--grasp_mode", type=str, default="three_finger_grasp")
    parser.add_argument("--debug_video", action="store_true")
    args = parser.parse_args()

    process_path(
        input_path=args.input,
        output_dir=args.output,
        model_path=args.model,
        grasp_mode=args.grasp_mode,
        save_debug_video=args.debug_video
    )
