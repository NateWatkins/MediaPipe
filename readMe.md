# Motion Analysis with MediaPipe

## Overview
This project is focused on analyzing human motion using MediaPipe.  
The goal is to take raw RGB frames or video, extract body/hand/face landmarks, smooth them with filtering, and then generate useful outputs like CSV files, annotated frames, segmentation masks, and position/velocity plots.

Essentially this program goes from:
video -> landmarks -> filtered data -> visualizations





## Quick Start Guide
First go to jointExtracter.py and testVideo.py to change the input and output paths to your MediaPipeFolder and Data

Navigate to your MediaPipe directory and Run:
1. **python3 jointExtracter.py**
2. **python3 testVideo.py**


## Main Functions

### save_csv_files()
Handles saving both raw (unfiltered) and filtered data.  
- Writes face, body, and hand landmarks to CSVs.  
- Applies a Butterworth low-pass filter to smooth noisy joints.  



### _butterworth_lowpass()
Applies a Butterworth filter to a signal.  
- cutoff_hz: sets which frequencies are smoothed out.  
- fs: sampling rate (frames per second).  
- order: filter steepness.  
Used by save_csv_files to create the filtered dataset.



### process_frames_folder()
Processes a directory of RGB frames.  
- Runs MediaPipe Holistic on each frame.  
- Extracts landmarks for body, hands, and face.  
- Saves annotated frames and calls save_csv_files.  
This is useful for testing a short sequence of frames (like the first 200).


### process_video()
Processes an entire video file.  
- Extracts landmarks across all frames.  
- Outputs segmentation masks, annotated frames, and CSVs.  
This is the “full pipeline” version of the frame-based workflow.



