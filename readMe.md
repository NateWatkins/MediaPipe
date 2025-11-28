# MediaPipe + ELAN/ASL-LEX Processing Pipeline

## Overview
This project processes signing videos to support linguistic and basic kinematic analysis. It has two main steps:

1. Extract pose, hand, and face landmarks from videos or frames using MediaPipe Holistic.
2. Merge ELAN annotation data with ASL-LEX so each signed token has lexical properties (lexical class, frequency, iconicity, etc.) for statistical analysis.

The extracted MediaPipe data is intended for downstream statistical tests on movement patterns, durations, and positional behavior across signs and participants.



## 1. MediaPipe Extraction

### Input Format
The extraction script reads participants from:

    config/participants.csv

Each row:

    subject_id,video_path,frames_folder

The script processes whichever inputs exist (video, frames, or both).

### What It Produces
For every clip, three CSV files:

    *_face.csv
    *_body.csv
    *_hand.csv

These contain 2D landmark coordinates for all frames.

### Output Location
Results are stored per subject:

    output_subjects/<subject_id>/
        Unfiltered_MP_Data/
        Filtered_MP_Data/

### Transform Options
The extraction pipeline includes optional transforms:

- Normalize to Nose — moves nose to (0,0)
- Min–Max Scaling — scales coordinates to [0,1]
- Low-Pass Filtering — smooths trajectories with a Butterworth filter
- Center and Scale to Shoulders — translates and scales pose so shoulder distance = 1

These are toggled at the top of `main.py`.

### Run Extraction

    python main.py

---

## 2. ELAN + ASL-LEX Merge

### Why Merge With ASL-LEX
ELAN provides which sign was produced and its timing.
ASL-LEX provides lexical properties of that sign.
Merging them allows analyses such as:

- duration differences across lexical classes
- patterns relating movement to phonological or lexical features
- linking kinematic behavior to known linguistic properties

### What the Script Does
The merge script:

1. Finds an ELAN `.txt` file (location may vary).
2. Loads ASL-LEX.
3. Cleans gloss names.
4. Removes rejected trials.
5. Joins ELAN trials with ASL-LEX.
6. Runs a Mann–Whitney U test comparing Noun vs Verb durations.f
7. Saves:

       Elan_ASLLEX_Joined.csv
       Elan_ASLLEX_Joined_Results.csv

### Run Merge

    python3 merge_elan_asllex.py

---

## 3. Project Structure

    project/
      main.py
      merge_elan_asllex.py
      config/
        participants.csv
        done.csv
      output_subjects/
      MediaPipe10.20.25Backup/

---

## 4. Installation

    pip3 install mediapipe opencv-python pandas scipy numpy natsort

---

## 5. Summary

- Add participants to `config/participants.csv`.
- Run `main.py` to generate MediaPipe landmark CSVs.
- Run `merge_elan_asllex.py` to attach ASL-LEX properties.
- Use the CSVs for linguistic and kinematic statistical analysis.
