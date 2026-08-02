# Dataset Conversion Notes

## Overview
- **Dataset**: [Name and source]
- **Date started**: 2026-07-29
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- CONVERSION_NOTES.md
- code/
- data/
- methods.txt
- paper.pdf
- train_decoder.py

Python/package check:
- python3 available
- numpy import successful
- torch import successful

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| load_dat | code/utils.py | LOADING | Loads preprocessed dataset object from pickle for downstream analyses |
| moving_average | code/utils.py | PROCESSING | Smooths time series / tuning curves |
| sort_trialstart | code/utils.py | PROCESSING | Sorts neurons/trials around trial start for visualization/alignment checks |
| sort_by_max_loc | code/utils.py | PROCESSING | Orders neurons by peak activity location across corridor/time bins |
| get_bootstrap_ci / get_bootstrap_prob | code/utils.py | PROCESSING | Bootstrap confidence intervals / probabilities for reported statistics |
| permutation_test / permutation_test_paired | code/utils.py | PROCESSING | Statistical comparisons between conditions |
| first_lick_position | code/fig*.py | PROCESSING | Computes first-lick probability as function of corridor position |
| number_of_trials | code/fig*.py | PROCESSING | Summarizes trial counts across training days/conditions |

### Notes
- Reference code appears to operate on an already processed/packaged dataset loaded from pickle via `utils.load_dat`, rather than raw acquisition files.
- Figure scripts analyze behavioral variables including licking, position in corridor, reward condition, and training day; these are relevant for decoder input/output construction.
- Utility functions emphasize trial-start alignment and ordering neurons by spatial/temporal response profile, suggesting corridor-position/trial-aligned representations are central.
- Need to verify exact source data organization and available variables directly from `data/` in Step 2.
- No obvious ΔF/F computation or spike-sorting quality filtering identified yet from the inspected code; likely preprocessing may already be embedded in packaged data files. This must be checked against `data/` and methods text.

----------|------|-------|---------|
| | | [LOADING | PROCESSING | CURATION] | |

### Notes
[Your notes here]

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `data/spk/`: session-wise neural arrays in `.npy` files named `<session_id>_neural_data.npy`
- `data/beh/`: behavior dictionaries in `.npy` object files grouped by experimental condition (e.g. supervised train/test, before/after learning, swap)
- Behavior files load as Python dicts keyed by session IDs (and some swap-condition variants). Each session payload includes 59 fields with trial-level variables (`ntrials`, `Trial_start_time`, `Trial_end_time`, `SoundTime`, `SoundTimeDelay`, `RewTime`, `isRew`, `WallName`, `TrialStim`), frame-level variables (`ft`, `ft_trInd`, `ft_Pos`, `ft_PosCum`, `ft_RunSpeed`, `ft_isMoving`, etc.), and stimulus masks (`StimTrial`, `StimFrame`).
- Neural data are spike-based object files (not calcium imaging); each neural file is a dict with key `spks`, whose value is a list requiring further interpretation (likely neuron-wise or trial-wise spike representations).

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 267 if `spks` list indexes neurons; interpretation pending confirmation from methods/code |
| Neurons / session | 3 if `spks` list indexes neurons; interpretation pending confirmation from methods/code |
| Subjects | 19 |
| Sessions / subject | DR10:6, DR15:5, LZ13:4, LZ16:4, TX104:2, TX105:5, TX108:7, TX109:6, TX119:8, TX123:8, TX124:3, TX139:2, TX140:1, TX60:5, TX61:5, TX83:3, TX85:2, TX88:6, VR2:7 |
| Trials (total) | [144 behavior entries across condition files; per-entry trial counts available, unique-session total pending reconciliation] |
| Trials / session | min 84, mean 440.85, max 789 across behavior entries |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 20,547 to 89,577 per recording | "We ran Suite2p on this data to obtain the activity traces from 20,547 to 89,577 neurons in each recording." | 
| Neurons / session | | |
| Subjects | 19 mice | "We performed 89 recordings in 19 mice..." |
| Sessions / subject | 89 recordings across 19 mice (variable recordings/mouse) | "We performed 89 recordings in 19 mice..." |
| Trials (total) | | |
| Trials / session | hundreds (e.g. 84-789 across behavior entries) | Behavior dict inspection + methods describe repeated corridor trials |
| Neural data time bin | frame-based deconvolved traces (exact dt pending) | "All our analyses were based on deconvolved fluorescence traces." |
| Behavior data time bin | | |
| Reward rate | rewarded corridor only; imaging cue present in all trial types | "sound cue was presented in all trial types... reward delivered if a lick was detected after the sound cue in the rewarded corridor" | 
| <Task/behavior statistic 1> | | | | 
| <Task/behavior statistic 2> | | | |
| ... | | | | 


### Processing Details
- Virtual reality corridor moved at constant speed 60 cm/s when mice ran above threshold.
- Imaging sessions: sound cue time/randomized position per trial, uniformly between 0.5 m and 3.5 m.
- In rewarded corridors for task mice, sound cue indicated beginning of reward zone; reward delivered after lick detected after cue.
- Unsupervised imaging sessions still included sound cue for consistency, without reward.
- Behavior-only training: 5 total days, with passive reward on day 1 and active reward on days 2-5.
- Behavior-only reward zone start randomized uniformly between 2 m and 3 m.
- Paper methods state analyses used deconvolved fluorescence traces processed with Suite2p, not raw calcium traces.

### Curation Steps

**Neuron curation rules**:
- Paper methods mention Suite2p processing, ROI detection, cell classification, neuropil correction, and spike deconvolution; exact inclusion/exclusion criteria still to be extracted.

**Trial curation rules**:
- Trials are defined by corridor traversals with trial-level timing variables available in behavior dicts; exact trial filtering still to be extracted.


### Decoders Trained
| Decoded variable | Accuracy |
| | |

---

## Step 4: Check for Consistency
**Status**: IN PROGRESS

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| `spks` interpretation | Code folder/name suggests spike-like data | Each session has `spks` as a list of 3 arrays with shapes like (19,408, 31,707) | Paper reports 20,547 to 89,577 neurons per recording from deconvolved fluorescence traces | Interpret `spks` as three area/plane-specific neuron-by-time arrays; sum of first dimensions per session matches paper neuron-count range. |

---

## Step 5: Mapping Planning
**Status**: NOT STARTED

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| | neural | | |
| | input[0] | | |
| | output[0] | | |

### Key Decisions
1. **[Decision]**: [Rationale]

### Planned Sanity Checks
- [ ] Check 1
- [ ] Check 2

---

## Step 6: Script Development
**Status**: NOT STARTED

[Implementation notes]

Code inefficiencies identified:
[Note]

Code speedups added:
[Note]

---

## Step 7: Sample Conversion and Validation
**Status**: NOT STARTED

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | |
| Neurons / session | |
| Subjects | |
| Sessions / subject | |
| Trials (total) | |
| Trials / session | |
| <Input statistic 1 range> | [MIN, MAX] |
| ... | [MIN, MAX] |
| <Output statistic 1 distribution> | [FRAC0,FRAC1,...] |
| ... | [FRAC0,FRAC1,...] |

### Processing Plots Review
[Notes on any anomalies]

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |

| Step | Time / Session | Estimated Total Time |
| | | |

---

## Step 8: Sample Decoder Training
**Status**: NOT STARTED

### Format Validation
- Errors: [None / List]
- Warnings: [None / List]

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| <Output 1> | | |
| <Output 2> | | |
| ... | | |

---

## Step 9: Full Conversion and Validation
**Status**: NOT STARTED

### Output Files
- `converted_data.pkl`: [size]
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | | | | | |
| Mean neurons/session | | | | | |
| Subjects | 19 mice | "We performed 89 recordings in 19 mice..." | | | |
| Sessions | | | | | |
| Trials (total) | | | | | |
| Trials/session (mean) | | | | | |
| <Input 1 range> | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| ... | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| <Output 1 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |
| <Output 2 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |

---

## Step 10: Critical Review 1
**Status**: NOT STARTED

### Checks Performed
1. [Check]: [Result]

### Issues Found and Resolved
- [Issue]: [Resolution]

---

## Step 11: Full Decoder Training
**Status**: NOT STARTED

### Training Progress
- Loss decreasing: [Yes/No]

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| <Output 1> | | |
| <output 2> | | |
| ... | | |

---

## Step 12: Critical Review 2
**Status**: NOT STARTED

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| | |

[Analysis of any low accuracies]

### Issues Found and Resolved
- [Issue]: [Resolution]

---

## Step 13: Documentation and Cleanup
**Status**: NOT STARTED

- [ ] README.md created
- [ ] cache/ folder created
- [ ] All files organized
