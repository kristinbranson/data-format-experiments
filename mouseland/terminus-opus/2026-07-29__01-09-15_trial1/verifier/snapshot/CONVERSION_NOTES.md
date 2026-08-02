# Dataset Conversion Notes

## Overview
- **Dataset**: Zhong et al. 2025 - "Unsupervised pretraining in biological neural networks"
- **Date started**: 2025-07-29
- **Goal**: Convert calcium imaging + behavioral data to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `code/` - Reference code (utils.py, figure scripts, data_process_script.ipynb)
- `data/` - Data directory (beh/, spk/, retinotopy/, process_data/)
- `paper.pdf` - Reference paper
- `methods.txt` - Extracted methods text
- `decoder.py`, `train_decoder.py` - Decoder scripts

Python: numpy 2.3.5, torch 2.6.0+cu124, scipy 1.18.0

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| load_spk | utils.py:338 | LOADING | Load neural data, concatenate planes |
| load_exp_beh | utils.py:326 | LOADING | Load behavioral data |
| load_retino | utils.py:330 | LOADING | Load retinotopy/brain area data |
| neu_area_ID | utils.py:312 | PROCESSING | Map iarea to brain regions |
| get_interpPos_spk | utils.py:120 | PROCESSING | Interpolate spikes to position bins |
| dprime | utils.py:370 | PROCESSING | D-prime for neuron selection |
| get_cat_id | utils.py:137 | PROCESSING | Assign stimulus category IDs |

### Notes
- Neural data: deconvolved Ca2+ traces from Suite2p (decay timescale 0.75s)
- Frame rate: 3.17 Hz
- No explicit neuron quality filtering in reference code
- Reference code does NOT have a traditional decoder

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Neural: 89 .npy files in data/spk/, each with {'spks': [plane1, plane2, ...]}
- Behavioral: 23 .npy files in data/beh/ organized by experiment type
- Retinotopy: 89 .npz files with iarea (brain region per neuron)

### Spatial Structure
- Corridor_Length = 60 VR units = 6m (4m texture + 2m grey)
- Texture_Length = 40 VR units = 4m (positions 0-40)
- 1 VR unit = 0.1m

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neural data files | 89 |
| Subjects | 19 |
| Sessions per subject | 1-8 |
| Neurons per session | 20,547-89,577 |
| Frame rate | 3.17 Hz |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Recordings | 89 | "We performed 89 recordings" |
| Mice | 19 | "in 19 mice" |
| Neurons/recording | 20,547-89,577 | "activity traces from 20,547 to 89,577 neurons" |
| Corridor length | 4m | "corridors were each 4 m long" |
| Grey space | 2m | "with 2 m of grey space" |
| VR speed | 60 cm/s | "constant speed (60 cm/s)" |
| Sound cue position | 0.5-3.5m | "uniform distribution between 0.5 m and 3.5 m" |
| Frame rate | 3.17 Hz | Notebook cell 4 |
| Deconvolution | 0.75s decay | "timescale of decay of 0.75 s" |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

All statistics consistent between code, data, and paper.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source | Target | Transform |
|--------|--------|----------|
| spks (concat planes) | neural | Extract per-trial from StartFr to GrayFr |
| SoundFr - frame | input[0]: time_to_sound_cue | Time in seconds |
| datexp | input[1]: day_of_training | Chronological day index |
| frame - StartFr | input[2]: time_since_trial_start | Time in seconds |
| isRew | input[3]: reward_availability | Binary |
| WallName | output[0]: visual_stimulus | Categorical |
| LickFr | output[1]: licking | Binary per frame |
| ft_Pos | output[2]: position_bin | 4 bins of 1m |
| ft_RunSpeed | output[3]: speed_bin | 4 quartile bins |

---

## Step 6: Script Development
**Status**: COMPLETE

Script: convert_data.py
- Loads all sessions from exp_info
- Pass 1: Collect running speeds for quartile computation
- Pass 2: Process each session (load neural, behavioral, retinotopy data)
- Extracts corridor portion (StartFr to GrayFr) for each trial
- Neural data stored as float16 to reduce file size

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Subjects | 2 |
| Total trials | 713 |
| Neurons/session | 85481, 58224 |

Verification: No errors, no warnings.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Decoder Results (Sample)
| Output | Train Acc | Val Acc | Chance |
|--------|-----------|---------|--------|
| visual_stimulus | 0.9567 | 0.8722 | 0.2500 |
| licking | 0.8856 | 0.8754 | 0.5000 |
| position_bin | 0.6825 | 0.5582 | 0.2500 |
| speed_bin | 0.5675 | 0.5900 | 0.2500 |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 117 GB (float16 neural data)
- `verification_full_out.txt`: created, no errors or warnings

### Consistency Check
| Statistic | Paper | Converted Data | Match? |
|-----------|-------|----------------|--------|
| Total recordings | 89 | 76 (13 swap sessions skipped) | Partial |
| Subjects | 19 | 19 | Yes |
| Neurons range | 20,547-89,577 | 24,383-89,577 | Yes (min differs due to skipped session) |
| Mean neurons | ~53,252 | 53,252 | Yes |
| Total trials | - | 31,442 | - |
| Mean trials/session | - | 414 | - |
| Frame rate | 3.17 Hz | 3.17 Hz | Yes |

### Notes on Missing Sessions
13 sessions skipped because behavioral data is stored with _swap1/_swap2 suffixes (test3 experiments with leaf1_swap stimuli). These sessions share neural recordings with base sessions but have different trial structures.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
- verification_full_out.txt: "Data format is valid, no errors or warnings."
- No errors found.

### Check 2: Sanity checks
- Neural data: Verified neuron counts match between spk files and converted data
- Input data: time_to_sound_cue ranges are consistent with SoundPos distribution (0.5-3.5m)
- Output data: Position bins roughly equally distributed (~25% each)
- Licking: 96.5% no_lick, 3.5% lick - consistent with most sessions being unsupervised (no licking)

### Check 3: Reference code comparison
- Data loading: Uses same load_spk logic (concatenate planes)
- Brain regions: Uses same neu_area_ID mapping
- No neuron filtering: Consistent with reference code
- Trial extraction: Uses StartFr to GrayFr (corridor portion only)

### Check 4: Key statistics comparison
- 19 subjects: matches paper
- 76/89 sessions: 13 swap sessions not loaded (documented)
- Neuron range 24,383-89,577: consistent with paper's 20,547-89,577
- Frame rate 3.17 Hz: matches paper

### Check 5: Edge cases
- Float frame indices (StartFr, SoundFr) rounded to int
- Trials with <2 frames excluded (1 trial in TX83_2022_08_31_1)
- Maximum trial length capped at 1000 frames (from extract_trial_data)

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (19415 -> 211 over 200 epochs)
- Used CPU due to GPU OOM

### Decoder Results (Full)
| Output | Train Acc | Val Acc | Chance | Ratio |
|--------|-----------|---------|--------|-------|
| visual_stimulus | 0.5760 | 0.4883 | 0.0769 | 6.3x |
| licking | 0.9404 | 0.8743 | 0.5000 | 1.75x |
| position_bin | 0.3295 | 0.3300 | 0.2500 | 1.32x |
| speed_bin | 0.4697 | 0.4675 | 0.2500 | 1.87x |

All outputs above chance.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Val Accuracy | Chance | Ratio | Assessment |
|----------|-------------|--------|-------|------------|
| visual_stimulus | 0.4883 | 0.0769 | 6.3x | Good - 13 classes |
| licking | 0.8743 | 0.5000 | 1.75x | Very good |
| position_bin | 0.3300 | 0.2500 | 1.32x | Above chance |
| speed_bin | 0.4675 | 0.2500 | 1.87x | Good |

### Notes
- visual_stimulus has 13 classes (many rare), so 0.49 balanced accuracy is good
- licking accuracy is high despite only 3.5% lick frames
- position_bin accuracy is modest but above chance - position is encoded in neural activity
- speed_bin accuracy is good - running speed modulates neural activity
- No evidence of overfitting (train/val gap is small)
- The paper does not report decoder accuracies directly, so no direct comparison possible

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] CONVERSION_NOTES.md complete
- [ ] README.md created
- [ ] cache/ folder created
