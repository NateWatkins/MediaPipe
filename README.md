# MediaPipe Sign Language Analysis

This is my project for analyzing sign language videos using Google's MediaPipe.

## What does this do?

It tracks:
- Body pose (33 landmarks) -
- Hand landmarks (21 per hand) - 
- Face landmarks (468 points) - 


## How to use it

1. Put your videos in the `input_videos/` folder 
2. Run `python3 knee_valgus_analysis.py`
3. `output_data/` folder for your CSV files


Each video gets its own folder with three files:
- `videoname_body.csv` - all the body joint positions
- `videoname_hand.csv` - left and right hand landmarks  
- `videoname_face.csv` - face landmarks 



**Body landmarks:**
- `nose_x`, `nose_y`, `nose_z`
- `left_shoulder_x`, `right_shoulder_y` 
- `left_knee_x`, `right_ankle_z`
...


**Hand landmarks:**
- `left_wrist_x`, `right_thumb_tip_y`
- `left_index_mcp_z` (mcp = metacarpophalangeal joint)
- `right_pinky_pip_x` (pip = proximal interphalangeal joint)
...



## Requirements

You'll need:
```
opencv-python
mediapipe  
pandas
numpy
```

Just do `pip install opencv-python mediapipe pandas numpy` and you should be good.

## Some notes from my testing

- The script uses MediaPipe's heaviest model (model_complexity=2) for better accuracy
- It saves every 30th frame as a PNG with all the landmarks drawn on it 
- If it can't detect hands/pose in a frame, it fills with NaN values instead of skipping
- The z-coordinate is depth relative to the wrist for hands, nose for body/face


## File structure it creates

```
├── input_videos/          # Put your MP4s here
├── output_data/           # CSV files go here
│   └── videoname_data/
│       ├── videoname_body.csv
│       ├── videoname_hand.csv
│       └── videoname_face.csv
└── output_frames/         # Sample frames with landmarks
    └── videoname_frames/
        ├── frame_000000.png
        ├── frame_000030.png
        └── ...
```

## Issues I ran into

- MediaPipe can be picky about video formats. MP4 usually works fine
- The face detection is pretty robust but hand detection struggles in low light
- Processing takes forever on longer videos (like 5+ minutes for a 30 second clip)
- Make sure your videos aren't too dark or the detection gets wonky


## Project Notes – Timeline of Changes

Set model to complexity 2 for better nose tracking (all videos will be normalized to nose).
Confirmed the body is detected well; happy with confidence values.
Made sure the video is processed as one continuous stream, not as separate images.
Tested on different sign language videos with different lighting.
Input: now pulls all videos from a folder automatically.
Output: creates separate CSV files for each video.
Deciding between one big data frame or three (face, body, hands).
Kept all float values — no rounding or cutting off.
Saved all frames, even if some keypoints are missing (use NaN instead of skipping).
Saved confidence values for every point in every frame.
Final CSVs have the same number of rows as video frames.
Noticed lighting affects confidence — front lighting is best. Shadows and dim light cause problems.
Tracking keypoints: nose, head, wrist.



Goals:
30 frames per second find the frame rate of the kinect Camera
Data from kinect is just frames not a video# MediaPipe
