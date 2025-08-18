import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import glob

# MediaPipe setup with heaviest model for maximum accuracy
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# MediaPipe landmark name mappings
pose_landmark_names = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index"
}

hand_landmark_names = {
    0: "wrist",
    1: "thumb_cmc",
    2: "thumb_mcp",
    3: "thumb_ip",
    4: "thumb_tip",
    5: "index_mcp",
    6: "index_pip",
    7: "index_dip",
    8: "index_tip",
    9: "middle_mcp",
    10: "middle_pip",
    11: "middle_dip",
    12: "middle_tip",
    13: "ring_mcp",
    14: "ring_pip",
    15: "ring_dip",
    16: "ring_tip",
    17: "pinky_mcp",
    18: "pinky_pip",
    19: "pinky_dip",
    20: "pinky_tip"
}


#Main Function
def process_video(video_path):
    print(f"Processing: {os.path.basename(video_path)}")
    
    capture = cv2.VideoCapture(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    
    # Declaring Model
    holistic = mp_holistic.Holistic(
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    face_data, body_data, hand_data = [], [], []
    frame_num = 0
    
    # Create frame output folder per video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_folder = f"output_frames/{video_name}_frames"
    os.makedirs(frame_folder, exist_ok=True)
    
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)
        
        # Extract data for all three categories
        face_data.append(extract_face_data(results, frame_num))
        body_data.append(extract_body_data(results, frame_num))
        hand_data.append(extract_hand_data(results, frame_num))
        
        # Save annotated frame every 30 frames
        if frame_num % 30 == 0:
            annotated_frame = frame.copy()
            
            # Draw all landmarks
            if results.face_landmarks:
                mp_drawing.draw_landmarks(annotated_frame, results.face_landmarks)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(annotated_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(annotated_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(annotated_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            cv2.imwrite(f"{frame_folder}/frame_{frame_num:06d}.png", annotated_frame)
        
        frame_num += 1
        if frame_num % 100 == 0:
            print(f"Progress: {frame_num}/{total_frames} frames")
    
    capture.release() #Need this line to clear memory
    holistic.close()
    
    return face_data, body_data, hand_data, video_name






#Folder Structure
def create_folders():
    folders = ['input_videos', 'output_frames', 'output_data']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)





def extract_face_data(results, frame_num):
    # Extract 468 face landmarks and confidence scores
    face_data = {'frame': frame_num}
    
    # Add overall detection confidence
    face_data['face_detection_confidence'] = 1.0 if results.face_landmarks else 0.0
    
    if results.face_landmarks:
        for i, landmark in enumerate(results.face_landmarks.landmark):
            face_data[f'face_{i}_x'] = landmark.x
            face_data[f'face_{i}_y'] = landmark.y
            face_data[f'face_{i}_z'] = landmark.z
            face_data[f'face_{i}_visibility'] = landmark.visibility
    else:
        # Fill NaN 
        for i in range(468):
            face_data[f'face_{i}_x'] = np.nan
            face_data[f'face_{i}_y'] = np.nan
            face_data[f'face_{i}_z'] = np.nan
            face_data[f'face_{i}_visibility'] = np.nan
    
    return face_data

def extract_body_data(results, frame_num):
    # Extract 33 pose landmarks and confidence scores
    body_data = {'frame': frame_num}
    
    # Add overall detection confidence
    body_data['pose_detection_confidence'] = 1.0 if results.pose_landmarks else 0.0
    
    if results.pose_landmarks:
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            joint_name = pose_landmark_names[i]
            body_data[f'{joint_name}_x'] = landmark.x
            body_data[f'{joint_name}_y'] = landmark.y
            body_data[f'{joint_name}_z'] = landmark.z
            body_data[f'{joint_name}_visibility'] = landmark.visibility
    else:
        # Fill NaN 
        for i in range(33):
            joint_name = pose_landmark_names[i]
            body_data[f'{joint_name}_x'] = np.nan
            body_data[f'{joint_name}_y'] = np.nan
            body_data[f'{joint_name}_z'] = np.nan
            body_data[f'{joint_name}_visibility'] = np.nan
    
    return body_data

def extract_hand_data(results, frame_num):
    # Extract left and right hand landmarks 21 points and confidence scores
    hand_data = {'frame': frame_num}
    
    # Add overall detection confidence for both hands
    hand_data['left_hand_detection_confidence'] = 1.0 if results.left_hand_landmarks else 0.0
    hand_data['right_hand_detection_confidence'] = 1.0 if results.right_hand_landmarks else 0.0
    
    # Left hand
    if results.left_hand_landmarks:
        for i, landmark in enumerate(results.left_hand_landmarks.landmark):
            joint_name = hand_landmark_names[i]
            hand_data[f'left_{joint_name}_x'] = landmark.x
            hand_data[f'left_{joint_name}_y'] = landmark.y
            hand_data[f'left_{joint_name}_z'] = landmark.z
            # Hand landmarks don't have visibility scores in MediaPipe
    else:
        for i in range(21):
            joint_name = hand_landmark_names[i]
            hand_data[f'left_{joint_name}_x'] = np.nan
            hand_data[f'left_{joint_name}_y'] = np.nan
            hand_data[f'left_{joint_name}_z'] = np.nan
    
    # Right hand
    if results.right_hand_landmarks:
        for i, landmark in enumerate(results.right_hand_landmarks.landmark):
            joint_name = hand_landmark_names[i]
            hand_data[f'right_{joint_name}_x'] = landmark.x
            hand_data[f'right_{joint_name}_y'] = landmark.y
            hand_data[f'right_{joint_name}_z'] = landmark.z
            # Hand landmarks don't have visibility scores in MediaPipe
    else:
        for i in range(21):
            joint_name = hand_landmark_names[i]
            hand_data[f'right_{joint_name}_x'] = np.nan
            hand_data[f'right_{joint_name}_y'] = np.nan
            hand_data[f'right_{joint_name}_z'] = np.nan
    
    return hand_data

def save_csv_files(face_data, body_data, hand_data, video_name):
    # Create separate folder for this video's data
    video_data_folder = f"output_data/{video_name}_data"
    os.makedirs(video_data_folder, exist_ok=True)
    
    # Save separate CSV files 
    face_df = pd.DataFrame(face_data)
    body_df = pd.DataFrame(body_data)
    hand_df = pd.DataFrame(hand_data)
    
    # Save with maximum float precision
    face_df.to_csv(f"{video_data_folder}/{video_name}_face.csv", index=False, float_format='%.10f')
    body_df.to_csv(f"{video_data_folder}/{video_name}_body.csv", index=False, float_format='%.10f')#10 digits
    hand_df.to_csv(f"{video_data_folder}/{video_name}_hand.csv", index=False, float_format='%.10f')
    
    print(f"Saved CSV files in {video_data_folder}/")
    return face_df, body_df, hand_df

def process_all_videos():
    # Process all MP4 files in input_videos folder
    video_files = glob.glob("input_videos/*.mp4")
    
    if not video_files:
        print("No MP4 files found in input_videos folder")
        print("Please place your videos in the input_videos folder")
        return
    
    print(f"Found {len(video_files)} videos to process")
    
    for video_path in video_files:
        try:
            face_data, body_data, hand_data, video_name = process_video(video_path)
            save_csv_files(face_data, body_data, hand_data, video_name)
            print(f"✓ Completed: {video_name}")
        except Exception as e:
            print(f"✗ Error processing {video_path}: {e}")
    
    print("All videos processed!")

def main():
    print("MediaPipe Sign Language Analysis System")
    print("=====================================")
    
    # Create folder structure
    create_folders()
    
    # Process all videos
    process_all_videos()

if __name__ == "__main__":
    main()