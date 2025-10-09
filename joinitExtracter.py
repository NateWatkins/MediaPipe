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

# ============================ User-tunable globals ============================
ANNOTATE_EVERY = 1    
RESIZE_MAX_WIDTH = None   # e.g., 960; set None to keep original resolution

OUTPUT_ROOT = "output_subjects"
# ---- Paths ----


IMAGE_EXTS = (".jpg", ".JPG", ".png", ".PNG")
VIDEO_EXTS = (".mov", ".MOV",".mp4",".m4v")

#Frame Folder Controls 
FRAME_LIMIT = 270      
FRAME_STRIDE = 1        
### This is the video controls
FRAME_LIMIT_VIDEO = 270  
FRAME_STRIDE_VIDEO = 1    


detectionConfidence = .5
trackingConfidence = .5




def get_frame_paths(frames_dir):
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(frames_dir, f"*{ext}")))
    return natsorted(paths)

def validate_inputs():
    video_ok  = bool(VIDEO_PATH and os.path.isfile(VIDEO_PATH) and VIDEO_PATH.endswith(VIDEO_EXTS))
    frames_ok = bool(FRAMES_DIR and os.path.isdir(FRAMES_DIR))
    if not video_ok:
        print(f"⚠ VIDEO_PATH not usable (skipping): {VIDEO_PATH}")
    if not frames_ok:
        print(f"⚠ FRAMES_DIR not usable (skipping): {FRAMES_DIR}")
    if video_ok or frames_ok:
        print("✓ At least one valid input found.")
    else:
        print("✗ No valid inputs found.")
    return video_ok, frames_ok

# ============================ Constants ============================
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

def create_folders():
    for folder in ['output_frames', 'output_data']:
        os.makedirs(folder, exist_ok=True)
from scipy.signal import butter, filtfilt

def _butterworth_lowpass(signal, cutoff_hz=4, fs=30.0, order=4):
    nyquist = 0.5 * fs
    wn = cutoff_hz / nyquist
    wn = min(max(wn, 1e-6), 0.999999)  # clamp to (0,1)
    b, a = butter(order, wn, btype="low", analog=False)
    return filtfilt(b, a, signal)


def _interp_short_nans(series, limit=5):
    # interpolate short NaN runs; leave long gaps as NaN
    return series

def _build_allowlist_cols(df, bases):
    # expand joint base names -> existing x/y/z columns in df
    cols = []
    for base in bases:
        for axis in ("x","y","z"):
            c = f"{base}_{axis}"
            if c in df.columns:
                cols.append(c)
    return cols

def save_csv_files(face_data, body_data, hand_data, clip_name, fps=30.0, cutoff_hz=4.0, order=2, folder_type=None):

    subject_id = os.path.basename(os.path.dirname(FRAMES_DIR))
    base_folder = os.path.join(OUTPUT_ROOT, subject_id, (folder_type or "Frames"), clip_name)
    os.makedirs(base_folder, exist_ok=True)
    frames_folder = video_folder = base_folder
    # ---- to DataFrames ----
    face_df = pd.DataFrame(face_data)
    body_df = pd.DataFrame(body_data)
    hand_df = pd.DataFrame(hand_data)


    # ----  Frames ----
    face_df.to_csv(os.path.join(base_folder, f"{clip_name}_face.csv"), index=False, float_format="%.10f")
    body_df.to_csv(os.path.join(base_folder, f"{clip_name}_body.csv"), index=False, float_format="%.10f")
    hand_df.to_csv(os.path.join(base_folder, f"{clip_name}_hand.csv"), index=False, float_format="%.10f")

   
        

    # # ---- make FILTERED copies ----
    # face_f = face_df.copy()
    # body_f = body_df.copy()
    # hand_f = hand_df.copy()

    # === What we filter (keep this list small & meaningful) ===
    # HANDS (both sides, ASL-relevant)
    hand_keys = [
        # per your df: left_<joint>_* and right_<joint>_*
        "left_wrist", "right_wrist",
        "left_thumb_ip", "right_thumb_ip",
        "left_thumb_tip", "right_thumb_tip",
        "left_index_mcp", "right_index_mcp",
        "left_index_tip", "right_index_tip",
        "left_middle_tip", "right_middle_tip",
        "left_ring_tip", "right_ring_tip",
        "left_pinky_tip", "right_pinky_tip",
    ]

    # BODY (minimal arm chain anchors)
    body_keys = [
        "left_shoulder", "right_shoulder",
        "left_elbow",    "right_elbow",
        "left_wrist",    "right_wrist",
        "nose", "left_ear", "right_ear",
        "left_eye", "right_eye",
        "mouth_left", "mouth_right",
    ]

    # FACE: we are NOT filtering the 468-pt mesh; we rely on the pose anchors above.

    # ---- build column allowlists (x/y/z for each joint base) ----
    # body_allow = _build_allowlist_cols(body_f, body_keys)
    # hand_allow = _build_allowlist_cols(hand_f, hand_keys)

    # detection/confidence fields to skip
    skip_cols = {
        "frame",
        "face_detection_confidence",
        "pose_detection_confidence",
        "left_hand_detection_confidence",
        "right_hand_detection_confidence",
    }

    # # ---- apply Butterworth ONLY to allowlisted columns ----
    # for col in body_allow:
    #     if col in skip_cols: 
    #         continue
    #     s = _interp_short_nans(body_f[col].astype(float), limit=5)
    #     #body_f[col] = _butterworth_lowpass(s.values, cutoff_hz, fs=fps, order=order)

 
    # for col in hand_allow:
    #     if col in skip_cols:
    #         continue

    #     s = _interp_short_nans(hand_f[col].astype(float), limit=5)
    #     #hand_f[col] = _butterworth_lowpass(s.values, cutoff_hz, fs=fps, order=order)


    # # ---- write FILTERED ----
    # face_f.to_csv(os.path.join(fil_folder, f"{clip_name}_face.csv"), index=False, float_format="%.10f")
    # body_f.to_csv(os.path.join(fil_folder, f"{clip_name}_body.csv"), index=False, float_format="%.10f")
    # hand_f.to_csv(os.path.join(fil_folder, f"{clip_name}_hand.csv"), index=False, float_format="%.10f")

    # # tiny metadata file for reproducibility
    # meta = {
    #     "fps": fps, "cutoff_hz": cutoff_hz, "order": order,
    #     "filtered_body_joints": body_keys,
    #     "filtered_hand_joints": hand_keys,
    # }

    # print(f"✓ Saved filtered   in {fil_folder}/")

    if( folder_type == "Frames"):
        print(f"✓ Saved raw data  in {base_folder}/")
    elif( folder_type == "Video"):
        print(f"✓ Saved raw data  in {base_folder}/")

    return face_df, body_df, hand_df      #, face_f, body_f, hand_f

def extract_face_data(results, frame_num):
    face = {'frame': frame_num}
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
    body = {'frame': frame_num}
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
            #body[f'{jn}_pose_detection_confidence'] = 0.0
    return body

def extract_hand_data(results, frame_num):
    hand = {
        'frame': frame_num,
    }
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

def _maybe_resize(img_bgr):
    if RESIZE_MAX_WIDTH is None:
        return img_bgr
    h, w = img_bgr.shape[:2]
    if w <= RESIZE_MAX_WIDTH:
        return img_bgr
    scale = RESIZE_MAX_WIDTH / float(w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

def process_frames_folder(frames_dir=None):
    if not frames_dir or not os.path.isdir(frames_dir):
        print(f"Invalid frames_dir: {frames_dir}")
        return
    print(f"Processing frames in: {frames_dir}")
    frame_paths = get_frame_paths(frames_dir)
    if not frame_paths:
        print("No images found."); return
    total_frames = len(frame_paths)
    print(f"Using {total_frames} frames ")
    clip_name = os.path.basename(os.path.normpath(frames_dir))
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=detectionConfidence,
        min_tracking_confidence=trackingConfidence,
        refine_face_landmarks=True,
        enable_segmentation=True,  ##Get the segmentation mask of the person vs background pickle file. docs
        smooth_segmentation=True,
    )

    face_data, body_data, hand_data = [], [], []
    for frame_num, fpath in enumerate(frame_paths):
        img_bgr = cv2.imread(fpath)
        if img_bgr is None:
            empty = SimpleNamespace(face_landmarks=None, pose_landmarks=None,
                                    left_hand_landmarks=None, right_hand_landmarks=None)
            face_data.append(extract_face_data(empty, frame_num))
            body_data.append(extract_body_data(empty, frame_num))
            hand_data.append(extract_hand_data(empty, frame_num))
            continue
        img_bgr_resized = _maybe_resize(img_bgr)
        rgb = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        face_data.append(extract_face_data(results, frame_num))
        body_data.append(extract_body_data(results, frame_num))
        hand_data.append(extract_hand_data(results, frame_num))
        # if ANNOTATE_EVERY and (frame_num % ANNOTATE_EVERY == 0):
        #     annotated = img_bgr_resized.copy()
        #     if results.face_landmarks:
        #         mp_drawing.draw_landmarks(
        #             annotated, results.face_landmarks,
        #             mp_holistic.FACEMESH_TESSELATION,
        #             landmark_drawing_spec=None,
        #             connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        #         )
        #     if results.pose_landmarks:
        #         mp_drawing.draw_landmarks(
        #             annotated, results.pose_landmarks,
        #             mp_holistic.POSE_CONNECTIONS,
        #             landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        #         )
        #     if results.left_hand_landmarks:
        #         mp_drawing.draw_landmarks(annotated, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        #     if results.right_hand_landmarks:
        #         mp_drawing.draw_landmarks(annotated, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        #     #cv2.imwrite(os.path.join(out_annot, f"frame_{frame_num:06d}.png"), annotated)
        if (frame_num + 1) % 100 == 0:
            print(f"Progress: {frame_num + 1}/{total_frames}")
        
    holistic.close()
    save_csv_files(face_data, body_data, hand_data, clip_name, folder_type="Frames")
    print(f"✓ Completed folder: {clip_name}")

def process_video(video_path=None):
    print(f"Processing video: {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video."); return None
    total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=detectionConfidence,
        min_tracking_confidence=trackingConfidence,
        refine_face_landmarks=True,
        enable_segmentation=True,
        smooth_segmentation=True,
    )
    face_data, body_data, hand_data = [], [], []
    frame_num = 0
    processed = 0
    clip_name = os.path.basename(os.path.normpath(video_path))
    out_annot = f"output_frames/{clip_name}_frames"
    #os.makedirs(out_annot, exist_ok=True)
    print(f"Video testing controls → limit={FRAME_LIMIT_VIDEO}, stride={FRAME_STRIDE_VIDEO}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if isinstance(FRAME_STRIDE_VIDEO, int) and FRAME_STRIDE_VIDEO > 1:
            if (frame_num % FRAME_STRIDE_VIDEO) != 0:
                frame_num += 1
                continue
        frame_resized = _maybe_resize(frame)
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        # if results.segmentation_mask is not None: #If the Segmentation mask is available
        #     mask_bin = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
        #     mask_dir = os.path.join("output_frames", f"{clip_name}_masks")
        #     os.makedirs(mask_dir, exist_ok=True)
        #     mask_path = os.path.join(mask_dir, f"frame_{frame_num:06d}_mask.png")
            #cv2.imwrite(mask_path, mask_bin)
        face_data.append(extract_face_data(results, frame_num))
        body_data.append(extract_body_data(results, frame_num))
        hand_data.append(extract_hand_data(results, frame_num))
        # if ANNOTATE_EVERY and ((processed) % ANNOTATE_EVERY == 0):
        #     annotated = frame_resized.copy()
        #     if results.face_landmarks:
        #         mp_drawing.draw_landmarks(
        #             annotated, results.face_landmarks,
        #             mp_holistic.FACEMESH_TESSELATION,
        #             landmark_drawing_spec=None,
        #             connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        #         )
        #     if results.pose_landmarks:
        #         mp_drawing.draw_landmarks(
        #             annotated, results.pose_landmarks,
        #             mp_holistic.POSE_CONNECTIONS,
        #             landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        #         )
        #     if results.left_hand_landmarks:
        #         mp_drawing.draw_landmarks(annotated, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        #     if results.right_hand_landmarks:
        #         mp_drawing.draw_landmarks(annotated, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            # cv2.imwrite(os.path.join(out_annot, f"frame_{frame_num:06d}.png"), annotated)
        processed += 1
        frame_num += 1
        
        if processed % 100 == 0:
            if total_frames_raw > 0:
                print(f"Processed {processed} frames (raw index ~{frame_num}/{total_frames_raw})")
            else:
                print(f"Processed {processed} frames")
    cap.release()
    holistic.close()
    save_csv_files(face_data, body_data, hand_data, clip_name, folder_type="Video")
    print(f"✓ Completed video: {clip_name} ")
    return True



def main():
    print("MediaPipe Holistic Extraction (Batch)")
    print("====================================")

    done_file = "done.txt"
    done = set(open(done_file).read().split()) if os.path.exists(done_file) else set()
    global OUTPUT_ROOT
    BASE_OUT = OUTPUT_ROOT  # remember original root


    if not os.path.exists("participants.txt"):
        print("No participants.txt file found.")
        return

    with open("participants.txt") as f:
        lines = [l.strip() for l in f if l.strip()]

    

    for line in lines:
        video_path, frames_dir = line.split("|", 1)
        tag_frames = f"{line}#frames"
        tag_video  = f"{line}#video"

        clip_name = os.path.splitext(os.path.basename(video_path))[0]
        subject_id = os.path.normpath(frames_dir).split(os.sep)[-3]
        line_root = os.path.join(BASE_OUT, subject_id)
        os.makedirs(line_root, exist_ok=True)
        OUTPUT_ROOT = line_root  

        os.makedirs(os.path.join(OUTPUT_ROOT, subject_id), exist_ok=True)

        global VIDEO_PATH, FRAMES_DIR
        VIDEO_PATH, FRAMES_DIR = video_path, frames_dir

        print(f"\n=== Processing {clip_name} ===")
        video_ok, frames_ok = validate_inputs()

        if frames_ok and tag_frames not in done:
            process_frames_folder(frames_dir)
            with open(done_file, "a") as df: df.write(tag_frames + "\n")

        if video_ok and tag_video not in done:
            process_video(video_path)
            with open(done_file, "a") as df: df.write(tag_video + "\n")
        OUTPUT_ROOT = BASE_OUT
    print("\nAll participants processed ✓")


if __name__ == "__main__":
    main()
