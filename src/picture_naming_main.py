import os
import glob
import cv2
import numpy as np
import pandas as pd
from types import SimpleNamespace
from natsort import natsorted
import mediapipe as mp
from contextlib import contextmanager

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ============================ User-tunable globals ============================
ANNOTATE_EVERY = 1    
RESIZE_MAX_WIDTH = None   # e.g., 960; set None to keep original resolution

OUTPUT_ROOT = "../output_subjects"
# ---- Paths ----

TESTING_MODE = False #True = FrameLimit ON 



CENTER_AND_SCALE = True
NORMALIZE_TO_NOSE = True
LOWPASS_FILTER = False
MINMAX_SCALE = True


if CENTER_AND_SCALE: NORMALIZE_TO_NOSE = False



FILTER_SWITCH = NORMALIZE_TO_NOSE or LOWPASS_FILTER or MINMAX_SCALE



IMAGE_EXTS = (".jpg", ".JPG", ".png", ".PNG")
VIDEO_EXTS = (".mov", ".MOV",".mp4",".m4v")

#Frame Folder Controls 
FRAME_LIMIT = 50      
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
    os.makedirs("output_data", exist_ok=True)
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
        for axis in ("x","y"):
            c = f"{base}_{axis}"
            if c in df.columns:
                cols.append(c)
    return cols

def _apply_lowpass(df, cols, cutoff_hz=4.0, fs=30.0, order=4):
    for col in cols:
        if col in df.columns:
            signal = df[col].values
            if np.isnan(signal).all():
                continue  # skip all-NaN columns
            filtered = _butterworth_lowpass(signal, cutoff_hz=cutoff_hz, fs=fs, order=order)
            df[col] = filtered
    return df


def minmax_scale(dframe, x=None, y=None):


    if isinstance(dframe, pd.DataFrame):
        df = dframe.copy()

        xcols = [c for c in df.columns if c.endswith('_x')]
        ycols = [c for c in df.columns if c.endswith('_y')]

        # Compute bounds if not provided
        if x is None:
            xmin = np.nanmin(df[xcols].to_numpy()) if xcols else 0.0
            xmax = np.nanmax(df[xcols].to_numpy()) if xcols else 1.0
        else:
            xmin, xmax = x
        if y is None:
            ymin = np.nanmin(df[ycols].to_numpy()) if ycols else 0.0
            ymax = np.nanmax(df[ycols].to_numpy()) if ycols else 1.0
        else:
            ymin, ymax = y

        # Guard against zero range
        xden = (xmax - xmin) if (xmax - xmin) not in (0, np.nan) else 1.0
        yden = (ymax - ymin) if (ymax - ymin) not in (0, np.nan) else 1.0

        if xcols:
            df[xcols] = (df[xcols] - xmin) / xden
        if ycols:
            df[ycols] = (df[ycols] - ymin) / yden

        return df

    # Case 2: ndarray
    arr = np.asarray(dframe)
    if arr.ndim == 3 and arr.shape[2] >= 2:
        # x
        if x is None:
            min_x = np.nanmin(arr[:, :, 0])
            max_x = np.nanmax(arr[:, :, 0])
        else:
            min_x, max_x = x
        xden = (max_x - min_x) if (max_x - min_x) not in (0, np.nan) else 1.0
        arr[:, :, 0] = (arr[:, :, 0] - min_x) / xden

        # y
        if y is None:
            min_y = np.nanmin(arr[:, :, 1])
            max_y = np.nanmax(arr[:, :, 1])
        else:
            min_y, max_y = y
        yden = (max_y - min_y) if (max_y - min_y) not in (0, np.nan) else 1.0
        arr[:, :, 1] = (arr[:, :, 1] - min_y) / yden
        return arr

    # Fallback: nothing to do
    return dframe



def save_csv_files(face_data, body_data, hand_data, fps=30.0, cutoff_hz=4.0, order=2, folder_type=None):
    base_folder = os.path.join(OUTPUT_ROOT, clip_name)   # <- OUTPUT_ROOT already scoped by use_output_root
    os.makedirs(base_folder, exist_ok=True)

    # ---- to DataFrames ----
    face_df = pd.DataFrame(face_data)
    body_df = pd.DataFrame(body_data)
    hand_df = pd.DataFrame(hand_data)

    # If we’re in a “filtered” run, normalize in-place before saving
    if NORMALIZE_TO_NOSE:
        face_df, body_df, hand_df = _normalize_to_nose(face_df, body_df, hand_df)
        print("✓ Normalized face, body, and hand data to nose position (nose at origin).")

    if MINMAX_SCALE:
        face_df = minmax_scale(face_df)
        body_df = minmax_scale(body_df)
        hand_df = minmax_scale(hand_df)
        print("✓ Applied min-max scaling to face, body, and hand data.")
        
    if LOWPASS_FILTER:
        face_df = _apply_lowpass(face_df, face_cols, cutoff_hz=cutoff_hz, fs=fps, order=order)
        body_df = _apply_lowpass(body_df, body_cols, cutoff_hz=cutoff_hz, fs=fps, order=order)
        hand_df = _apply_lowpass(hand_df, hand_cols, cutoff_hz=cutoff_hz, fs=fps, order=order)
        print(f"✓ Applied lowpass Butterworth filter with cutoff {cutoff_hz} Hz and order {order} to face, body, and hand data.")
    if CENTER_AND_SCALE:
        order = [pose_landmark_names[i] for i in range(33)]
        body_arr = np.stack(
            [np.stack([body_df[f"{name}_x"].to_numpy(),
                    body_df[f"{name}_y"].to_numpy()], axis=-1)
            for name in order],
            axis=1,
        )
        body_arr_t, T, S = center_and_scale_pose_data(body_arr)
        tx, ty = T[:,0], T[:,1]
        s = np.where(np.isfinite(S), S, 1.0)

        def apply_T_S(df):
            xcols = [c for c in df.columns if c.endswith("_x")]
            ycols = [c for c in df.columns if c.endswith("_y")]
            df[xcols] = df[xcols].sub(tx, axis=0).mul(s, axis=0)
            df[ycols] = df[ycols].sub(ty, axis=0).mul(s, axis=0)

        apply_T_S(face_df)
        apply_T_S(hand_df)
        for j, name in enumerate(order):
            body_df[f"{name}_x"] = body_arr_t[:, j, 0]
            body_df[f"{name}_y"] = body_arr_t[:, j, 1]
        print("✓ Centered and scaled pose data (nose at origin, shoulders at unit distance).")




    # ---- write once to the scoped output dir ----
    face_df.to_csv(os.path.join(base_folder, f"{subject_id}_{folder_type}_face.csv"), index=False, float_format="%.10f")
    body_df.to_csv(os.path.join(base_folder, f"{subject_id}_{folder_type}_body.csv"), index=False, float_format="%.10f")
    hand_df.to_csv(os.path.join(base_folder, f"{subject_id}_{folder_type}_hand.csv"), index=False, float_format="%.10f")

    if folder_type == "Frames":
        print(f"✓ Saved data in {base_folder}/")
    elif folder_type == "Video":
        print(f"✓ Saved data in {base_folder}/")

    
    return face_df, body_df, hand_df, None, None, None


def _normalize_to_nose(face_df, body_df, hand_df):
    # Only if nose exists
    if 'nose_x' not in body_df.columns or 'nose_y' not in body_df.columns:
        return face_df, body_df, hand_df

    ax = body_df['nose_x']  # anchor series per row
    ay = body_df['nose_y']

    def _shift_df(df):
        xcols = [c for c in df.columns if c.endswith('_x')]
        ycols = [c for c in df.columns if c.endswith('_y')]
        # Vectorized row-wise subtraction; NaNs in ax/ay will propagate (acceptable)
        df[xcols] = df[xcols].sub(ax, axis=0)
        df[ycols] = df[ycols].sub(ay, axis=0)
        return df

    face_df = _shift_df(face_df)
    body_df = _shift_df(body_df)
    hand_df = _shift_df(hand_df)
    return face_df, body_df, hand_df

def extract_face_data(results, frame_num):
    face = {'frame': frame_num}
    if results and results.face_landmarks:
        for i, lm in enumerate(results.face_landmarks.landmark):
            face[f'face_{i}_x'] = lm.x
            face[f'face_{i}_y'] = lm.y
            #face[f'face_{i}_z'] = lm.z
    else:
        for i in range(468):
            face[f'face_{i}_x'] = np.nan
            face[f'face_{i}_y'] = np.nan
            #face[f'face_{i}_z'] = np.nan
    return face

def extract_body_data(results, frame_num):
    body = {'frame': frame_num}
    if results and results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            jn = pose_landmark_names[i]
            body[f'{jn}_x'] = lm.x
            body[f'{jn}_y'] = lm.y
            #body[f'{jn}_z'] = lm.z
            body[f'{jn}_visibility'] = lm.visibility
    else:
        for i in range(33):
            jn = pose_landmark_names[i]
            body[f'{jn}_x'] = np.nan
            body[f'{jn}_y'] = np.nan
            #body[f'{jn}_z'] = np.nan
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
            #hand[f'left_{jn}_z'] = lm.z
    else:
        for i in range(21):
            jn = hand_landmark_names[i]
            hand[f'left_{jn}_x'] = np.nan
            hand[f'left_{jn}_y'] = np.nan
            #hand[f'left_{jn}_z'] = np.nan
    if results and results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            jn = hand_landmark_names[i]
            hand[f'right_{jn}_x'] = lm.x
            hand[f'right_{jn}_y'] = lm.y
            #hand[f'right_{jn}_z'] = lm.z
    else:
        for i in range(21):
            jn = hand_landmark_names[i]
            hand[f'right_{jn}_x'] = np.nan
            hand[f'right_{jn}_y'] = np.nan
            #hand[f'right_{jn}_z'] = np.nan
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

        if (frame_num + 1) % 100 == 0:
            print(f"Progress: {frame_num + 1}/{total_frames}")
        if frame_num >= FRAME_LIMIT and TESTING_MODE:
            print("Frame limit reached.")
            break
        
    holistic.close()
    save_csv_files(face_data, body_data, hand_data, folder_type="Frames")

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

        face_data.append(extract_face_data(results, frame_num))
        body_data.append(extract_body_data(results, frame_num))
        hand_data.append(extract_hand_data(results, frame_num))

        processed += 1
        frame_num += 1
        
        if processed % 100 == 0:
            if total_frames_raw > 0:
                print(f"Processed {processed} frames (raw index ~{frame_num}/{total_frames_raw})")
            else:
                print(f"Processed {processed} frames")
        if processed >= FRAME_LIMIT and TESTING_MODE:
            print("Frame limit reached.")
            break
        
    cap.release()
    holistic.close()
    save_csv_files(face_data, body_data, hand_data, folder_type="Video")
    print(f"✓ Completed video: {clip_name} ")
    return True

def center_and_scale_pose_data(pose_data_2d):


    ### Part 1: Setup the variables we'll need

    # Get the number of frames
    num_frames = pose_data_2d.shape[0]

    # Set up the return variables by preallocating numpy arrays of the appropriate dimensions
    # np.full_like() and np.full() create arrays filled with whatever the second parameter is, in this case np.nan
    # np.full() takes a tuple of dimensions for the first parameter, np.full_like() gleans the dimensions from an array you pass
    transformed_pose_data = np.full_like(pose_data_2d, np.nan, dtype=float)
    translation_vectors = np.full((num_frames, 2), np.nan, dtype=float)
    scaling_factors = np.full(num_frames, np.nan, dtype=float)

    # Create variables for pose indices (assuming 0=nose, 11=left shoulder, 12=right shoulder)
    nose_idx = 0
    ls_idx = 11
    rs_idx = 12

    for frame_idx in range(num_frames):
        frame_data = pose_data_2d[frame_idx].copy()

        nose = frame_data[nose_idx]
        left_shoulder = frame_data[ls_idx] # We only need this to check for NaN
        right_shoulder = frame_data[rs_idx] # We only need this to check for NaN

        nose_valid  = check_landmark_validity(nose.reshape(1, 2))[0]
        ls_valid    = check_landmark_validity(left_shoulder.reshape(1, 2))[0]
        rs_valid    = check_landmark_validity(right_shoulder.reshape(1, 2))[0]

        if not (nose_valid and ls_valid and rs_valid):
            continue

        # Shift the pose coordinates to center the nose at origin (0, 0)
        translation_vector = -nose
        frame_data += translation_vector # Apply translation

        left_shoulder_t = frame_data[ls_idx]
        right_shoulder_t = frame_data[rs_idx]

        # Calculate the current distance between the shoulders
        shoulder_dist = np.linalg.norm(right_shoulder_t - left_shoulder_t)

       
        if shoulder_dist == 0:

            transformed_pose_data[frame_idx] = frame_data
            translation_vectors[frame_idx] = translation_vector
            continue # Skip to next frame

        # Calculate the number you need to multiply the shoulder coordinates by to give them a distance of 1
        scaling_factor = 1.0 / shoulder_dist

       
        frame_data *= scaling_factor

        # Store results for this frame
        transformed_pose_data[frame_idx] = frame_data
        translation_vectors[frame_idx] = translation_vector
        scaling_factors[frame_idx] = scaling_factor

    return transformed_pose_data, translation_vectors, scaling_factors

def check_landmark_validity(coords):

    # Make sure the input array has the appropriate shape
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("Input coords must have shape (N, 2)")
    
    # Make an array size N with 1/True where an x or y is NaN, 0/False otherwise
    has_nan = np.isnan(coords).any(axis=1)
    
    out_of_bounds = ~has_nan & ((coords < 0) | (coords > 1)).any(axis=1)

    
    is_valid = ~(has_nan | out_of_bounds)
    
    if not is_valid:
        print("Landmark is valid: ", coords)


    return is_valid

def ensure_mp_dirs(root, sub_id):
    unf = os.path.join(root, sub_id, "Unfiltered_MP_Data")
    fil = os.path.join(root, sub_id, "Filtered_MP_Data")
    os.makedirs(unf, exist_ok=True)
    os.makedirs(fil, exist_ok=True)
    return unf, fil

@contextmanager
def use_output_root(path):
    global OUTPUT_ROOT
    _old = OUTPUT_ROOT
    OUTPUT_ROOT = path
    try:
        yield
    finally:
        OUTPUT_ROOT = _old



def main():
    print("MediaPipe Holistic Extraction (Batch)")
    print("====================================")
    global BASE_OUT
    done_file = "../config/done.csv"
    done = set(open(done_file).read().split()) if os.path.exists(done_file) else set()
    global OUTPUT_ROOT
    BASE_OUT = OUTPUT_ROOT  # remember original root
    

    #Big if/for loop here for filtered and non filtered.


    if not os.path.exists("../config/participants.csv"):
        print("No participants.csv file found.")
        return

    with open("../config/participants.csv") as f:
        lines = [l.strip() for l in f if l.strip()]

 

    for line in lines:
        sub_id, video_path, frames_dir = line.split(",")
        tag_frames = f"{line}#frames"
        tag_video  = f"{line}#video"
        global clip_name
        global subject_id
        subject_id = sub_id
        UNFILT_DIR, FILT_DIR = ensure_mp_dirs(BASE_OUT, sub_id)
        target_dir = FILT_DIR if FILTER_SWITCH else UNFILT_DIR
        





        global VIDEO_PATH, FRAMES_DIR
        VIDEO_PATH, FRAMES_DIR = video_path, frames_dir

        print(f"\n=== Processing {sub_id} ===")
        video_ok, frames_ok = validate_inputs()

        if frames_ok and tag_frames not in done:
            with use_output_root(target_dir):
                clip_name = "Frames"
                process_frames_folder(frames_dir)
            with open(done_file, "a") as df: df.write(tag_frames + "\n")

        if video_ok and tag_video not in done:
            with use_output_root(target_dir):
                clip_name = "RGB"
                process_video(video_path)
            with open(done_file, "a") as df: df.write(tag_video + "\n")
        OUTPUT_ROOT = BASE_OUT

    print("\nAll participants processed ✓")



if __name__ == "__main__":
    main()


