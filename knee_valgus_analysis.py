
import os
import glob
import cv2
import numpy as np
import pandas as pd
from types import SimpleNamespace
from natsort import natsorted

import mediapipe as mp
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ----------------------------- Config ---------------------------------
ANNOTATE_EVERY = 30       # save annotated frame every N frames (set to 1 to save all)
RESIZE_MAX_WIDTH = None   # e.g., 960; set None to keep original resolution
# ----------------------------------------------------------------------

# MediaPipe landmark name mappings
pose_landmark_names = {
    0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer", 7: "left_ear",
    8: "right_ear", 9: "mouth_left", 10: "mouth_right", 11: "left_shoulder",
    12: "right_shoulder", 13: "left_elbow", 14: "right_elbow", 15: "left_wrist",
    16: "right_wrist", 17: "left_pinky", 18: "right_pinky", 19: "left_index",
    20: "right_index", 21: "left_thumb", 22: "right_thumb", 23: "left_hip",
    24: "right_hip", 25: "left_knee", 26: "right_knee", 27: "left_ankle",
    28: "right_ankle", 29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index"
}

hand_landmark_names = {
    0: "wrist", 1: "thumb_cmc", 2: "thumb_mcp", 3: "thumb_ip", 4: "thumb_tip",
    5: "index_mcp", 6: "index_pip", 7: "index_dip", 8: "index_tip",
    9: "middle_mcp", 10: "middle_pip", 11: "middle_dip", 12: "middle_tip",
    13: "ring_mcp", 14: "ring_pip", 15: "ring_dip", 16: "ring_tip",
    17: "pinky_mcp", 18: "pinky_pip", 19: "pinky_dip", 20: "pinky_tip"
}

# ----------------------------- IO helpers -----------------------------

def create_folders():
    for folder in ['input_frames', 'input_videos', 'output_frames', 'output_data']:
        os.makedirs(folder, exist_ok=True)

def save_csv_files(face_data, body_data, hand_data, clip_name):
    video_data_folder = f"output_data/{clip_name}_data"
    os.makedirs(video_data_folder, exist_ok=True)

    face_df = pd.DataFrame(face_data)
    body_df = pd.DataFrame(body_data)
    hand_df = pd.DataFrame(hand_data)

    face_df.to_csv(f"{video_data_folder}/{clip_name}_face.csv", index=False, float_format='%.10f')
    body_df.to_csv(f"{video_data_folder}/{clip_name}_body.csv", index=False, float_format='%.10f')
    hand_df.to_csv(f"{video_data_folder}/{clip_name}_hand.csv", index=False, float_format='%.10f')

    print(f"Saved CSV files in {video_data_folder}/")
    return face_df, body_df, hand_df

# ----------------------------- Extractors -----------------------------

def extract_face_data(results, frame_num):
    """
    Face mesh has 468 landmarks. No 'visibility' attribute on face landmarks.
    """
    face = {'frame': frame_num, 'face_detection_confidence': 1.0 if results and results.face_landmarks else 0.0}
    if results and results.face_landmarks:
        for i, lm in enumerate(results.face_landmarks.landmark):
            face[f'face_{i}_x'] = lm.x
            face[f'face_{i}_y'] = lm.y
            face[f'face_{i}_z'] = lm.z
    else:
        for i in range(468):
            face[f'face_{i}_x'] = np.nan
            face[f'face_{i}_y'] = np.nan
            face[f'face_{i}_z'] = np.nan
    return face

def extract_body_data(results, frame_num):
    """
    Pose has 33 landmarks with 'visibility'.
    """
    body = {'frame': frame_num, 'pose_detection_confidence': 1.0 if results and results.pose_landmarks else 0.0}
    if results and results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            jn = pose_landmark_names[i]
            body[f'{jn}_x'] = lm.x
            body[f'{jn}_y'] = lm.y
            body[f'{jn}_z'] = lm.z
            body[f'{jn}_visibility'] = lm.visibility
    else:
        for i in range(33):
            jn = pose_landmark_names[i]
            body[f'{jn}_x'] = np.nan
            body[f'{jn}_y'] = np.nan
            body[f'{jn}_z'] = np.nan
            body[f'{jn}_visibility'] = np.nan
    return body

def extract_hand_data(results, frame_num):
    """
    Hands each have 21 landmarks. No 'visibility' attribute on hands.
    """
    hand = {
        'frame': frame_num,
        'left_hand_detection_confidence': 1.0 if results and results.left_hand_landmarks else 0.0,
        'right_hand_detection_confidence': 1.0 if results and results.right_hand_landmarks else 0.0
    }

    # Left
    if results and results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            jn = hand_landmark_names[i]
            hand[f'left_{jn}_x'] = lm.x
            hand[f'left_{jn}_y'] = lm.y
            hand[f'left_{jn}_z'] = lm.z
    else:
        for i in range(21):
            jn = hand_landmark_names[i]
            hand[f'left_{jn}_x'] = np.nan
            hand[f'left_{jn}_y'] = np.nan
            hand[f'left_{jn}_z'] = np.nan

    # Right
    if results and results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            jn = hand_landmark_names[i]
            hand[f'right_{jn}_x'] = lm.x
            hand[f'right_{jn}_y'] = lm.y
            hand[f'right_{jn}_z'] = lm.z
    else:
        for i in range(21):
            jn = hand_landmark_names[i]
            hand[f'right_{jn}_x'] = np.nan
            hand[f'right_{jn}_y'] = np.nan
            hand[f'right_{jn}_z'] = np.nan

    return hand

# ----------------------------- Processing: FRAMES -----------------------------

def _maybe_resize(img_bgr):
    if RESIZE_MAX_WIDTH is None:
        return img_bgr
    h, w = img_bgr.shape[:2]
    if w <= RESIZE_MAX_WIDTH:
        return img_bgr
    scale = RESIZE_MAX_WIDTH / float(w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

def process_frames_folder(frames_dir):
    """
    Process a directory of images as a clip. Uses static_image_mode=True.
    """
    print(f"Processing frames in: {frames_dir}")
    exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.webp")
    frame_paths = []
    for e in exts:
        frame_paths.extend(glob.glob(os.path.join(frames_dir, e)))
    if not frame_paths:
        print("No images found."); return

    frame_paths = natsorted(frame_paths)
    total_frames = len(frame_paths)
    clip_name = os.path.basename(os.path.normpath(frames_dir))

    out_annot = f"output_frames/{clip_name}_frames"
    os.makedirs(out_annot, exist_ok=True)

    holistic = mp_holistic.Holistic(
        static_image_mode=True,     # IMPORTANT for per-image processing
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    face_data, body_data, hand_data = [], [], []

    for frame_num, fpath in enumerate(frame_paths):
        img_bgr = cv2.imread(fpath)
        if img_bgr is None:
            # unreadable; push NaN rows to keep alignment
            empty = SimpleNamespace(face_landmarks=None, pose_landmarks=None,
                                    left_hand_landmarks=None, right_hand_landmarks=None)
            face_data.append(extract_face_data(empty, frame_num))
            body_data.append(extract_body_data(empty, frame_num))
            hand_data.append(extract_hand_data(empty, frame_num))
            continue

        img_bgr = _maybe_resize(img_bgr)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        face_data.append(extract_face_data(results, frame_num))
        body_data.append(extract_body_data(results, frame_num))
        hand_data.append(extract_hand_data(results, frame_num))

        if ANNOTATE_EVERY and (frame_num % ANNOTATE_EVERY == 0):
            annotated = img_bgr.copy()
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    annotated, results.face_landmarks,
                    mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated, results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(annotated, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(annotated, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            cv2.imwrite(os.path.join(out_annot, f"frame_{frame_num:06d}.png"), annotated)

        if (frame_num + 1) % 100 == 0:
            print(f"Progress: {frame_num + 1}/{total_frames}")

    holistic.close()
    save_csv_files(face_data, body_data, hand_data, clip_name)
    print(f"✓ Completed folder: {clip_name}")

def process_all_frame_folders(root="input_frames"):
    """
    input_frames/
      clipA/  (images)
      clipB/
      ...
    """
    if not os.path.isdir(root):
        print(f"Missing folder: {root}")
        return
    subdirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not subdirs:
        print("No frame folders found.")
        return
    print(f"Found {len(subdirs)} frame folders")
    for d in natsorted(subdirs):
        try:
            process_frames_folder(d)
        except Exception as e:
            print(f"✗ Error processing {d}: {e}")
    print("All frame folders processed!")

# ----------------------------- Processing: VIDEOS -----------------------------

def process_video(video_path):
    print(f"Processing video: {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video."); return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1

    holistic = mp_holistic.Holistic(
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    face_data, body_data, hand_data = [], [], []
    frame_num = 0

    clip_name = os.path.splitext(os.path.basename(video_path))[0]
    out_annot = f"output_frames/{clip_name}_frames"
    os.makedirs(out_annot, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = _maybe_resize(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        face_data.append(extract_face_data(results, frame_num))
        body_data.append(extract_body_data(results, frame_num))
        hand_data.append(extract_hand_data(results, frame_num))

        if ANNOTATE_EVERY and (frame_num % ANNOTATE_EVERY == 0):
            annotated = frame.copy()
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    annotated, results.face_landmarks,
                    mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated, results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(annotated, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(annotated, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            cv2.imwrite(os.path.join(out_annot, f"frame_{frame_num:06d}.png"), annotated)

        frame_num += 1
        if total_frames > 0 and frame_num % 100 == 0:
            print(f"Progress: {frame_num}/{total_frames} frames")
    cap.release()
    holistic.close()

    save_csv_files(face_data, body_data, hand_data, clip_name)
    print(f"✓ Completed video: {clip_name}")
    return True

def process_all_videos():
    video_files = glob.glob("input_videos/*.mp4")
    if not video_files:
        print("No MP4 files found in input_videos/")
        return
    print(f"Found {len(video_files)} videos to process")
    for vp in natsorted(video_files):
        try:
            process_video(vp)
        except Exception as e:
            print(f"✗ Error processing {vp}: {e}")
    print("All videos processed!")

# ----------------------------- Main -----------------------------------

def main():
    print("MediaPipe Holistic Extraction")
    print("=============================")
    create_folders()

    # Prefer frame folders when present; also process videos if present.
    has_frame_dirs = any(os.path.isdir(os.path.join("input_frames", d)) for d in os.listdir("input_frames") or [])
    has_videos = bool(glob.glob("input_videos/*.mp4"))

    if not has_frame_dirs and not has_videos:
        print("Place per-frame images under input_frames/<clip_name>/ or MP4 files in input_videos/")
        return

    if has_frame_dirs:
        process_all_frame_folders("input_frames")
    if has_videos:
        process_all_videos()

if __name__ == "__main__":
    main()