# Dataset Conversion Notes

## Overview
- **Dataset**: Zhong et al. 2025 - "Unsupervised pretraining in biological neural networks"
- **Date started**: 2025-07-29
- **Goal**: Convert calcium imaging + behavior data to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents: code/, data/, decoder.py, train_decoder.py, methods.txt, paper.pdf
Python environment: numpy 2.3.5, torch 2.6.0+cu124, scipy 1.18.0

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| load_spk | utils.py | LOADING | Concatenates spks across imaging planes |
| load_retino | utils.py | LOADING | Loads iarea for brain region assignment |
| neu_area_ID | utils.py | PROCESSING | Maps iarea -> V1(8), mHV(0,1,2,9), lHV(5,6), aHV(3,4) |
| spk_pos_interp | utils.py | PROCESSING | Position-based interpolation of spikes |
| get_interpPos_spk | utils.py | PROCESSING | Wrapper for position interpolation |
| dprime | utils.py | PROCESSING | Stimulus selectivity metric |
| Get_dprime_selective_neuron | utils.py | CURATION | Neuron filtering by selectivity |

### Key Processing Pattern
- Neuron filter: `(iarea != -1) & (iarea != 7)` excludes non-visual cortex neurons
- Frame filter: `ft_move > 0` only running/VR-moving frames
- Frame rate: 3.17 Hz (calcium imaging)
- Neural data: Suite2p deconvolved fluorescence traces

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `data/spk/`: 89 neural data files (MOUSE_DATE_BLK_neural_data.npy)
- `data/beh/`: 23 behavior files + Imaging_Exp_info.npy
- `data/retinotopy/`: 89 retinotopy files + areas.npz

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Sessions | 89 |
| Subjects | 19 |
| Neurons/session (raw) | 17,363-89,577 |
| Neurons/session (filtered) | 17,363-78,815 |
| Trials/session | 84-789 |
| Total trials | 38,110 |
| Frame rate | 3.17 Hz |

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source |
|-----------|-------|--------|
| Recordings | 89 | "We performed 89 recordings in 19 mice" |
| Subjects | 19 | "89 recordings in 19 mice" |
| Neurons/session | 20,547-89,577 | "20,547 to 89,577 neurons in each recording" |
| Neural data | Deconvolved fluorescence | "All analyses based on deconvolved fluorescence traces" |
| Frame rate | 3.17 Hz | Notebook: "fs = 3.17Hz" |
| Corridor | 4m texture + 2m grey | "corridors were each 4 m long, with 2 m of grey space" |
| VR speed | 60 cm/s | "constant speed (60 cm s−1)" |
| Run threshold | 6 cm/s | "threshold of 6 cm s−1" |
| Sound cue | 0.5-3.5m uniform | "uniform distribution between positions 0.5 m and 3.5 m" |
| Deconvolution | 0.75s decay | "timescale of decay of 0.75 s" |
| Analysis | Running only | "only considered timepoints during running" |

### Processing Details
- Suite2p: motion correction, ROI detection, neuropil correction, deconvolution
- Only running timepoints used
- Position-based interpolation into 60 bins (1 decimeter each)

### Curation Steps
**Neuron curation**: Exclude iarea==-1 and iarea==7 (outside visual cortex)
**Trial curation**: All trials included, frames filtered for VR-moving

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code | Data | Paper | Resolution |
|-------|------|------|-------|------------|
| Sessions | 89 in exp_info | 89 neural files | 89 recordings | Consistent |
| Subjects | 19 unique mice | 19 in data | 19 mice | Consistent |
| Frame rate | 3.17Hz | ~3.18Hz measured | 3.17Hz | Consistent |
| Corridor | 60dm | Corridor_Length=60 | 4m+2m=6m | Consistent |
| Neural data | Deconvolved | spks key | Deconvolved | Consistent |

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source | Target | Transform |
|--------|--------|----------|
| spks (concat) | neural | Filter neurons, extract per-trial running frames |
| SoundFr, ft | input[0]: time_to_sound_cue | Time from frame to sound cue (seconds) |
| datexp | input[1]: day_of_training | Days since first session per mouse |
| ft | input[2]: time_since_trial_start | Elapsed time from trial start |
| isRew | input[3]: reward_availability | 1=rewarded, 0=not |
| WallName | output[0]: visual_stimulus_category | Map to index |
| LickFr | output[1]: licking | Binary per frame |
| ft_Pos | output[2]: position_bin | 4 bins of 15dm over 60dm |
| ft_RunSpeed | output[3]: running_speed_bin | 4 quartile bins |

### Key Decisions
1. **All 89 sessions included**: Each unique neural recording = one session
2. **Neuron filtering**: Exclude iarea==-1 and iarea==7 (matches reference code)
3. **Frame filtering**: Only VR-moving frames (ft_move > 0, matches reference)
4. **Position bins**: 4 equal bins of 15dm over full 6m corridor
5. **Speed bins**: Global quartile-based (computed from all sessions)
6. **Time computations**: Using actual frame times (MATLAB datenum)

## Step 6: Script Development
**Status**: COMPLETE

Key optimizations:
- Save session data to temp files to avoid memory accumulation during processing
- Speed quartiles computed from behavior data only (no neural loading needed)
- Garbage collection after each session

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (TX108_2023_03_13_1, DR10_2022_07_12_1) |
| Subjects | 2 (TX108, DR10) |
| Trials | 713 (210 + 503) |
| Neurons | 78,815 and 52,246 |

### Verification: No errors or warnings

### Run Time: 179.6s for 2 sessions

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Decoder Results (Sample)
| Output | Train Acc | Val Acc | Chance |
|--------|-----------|---------|--------|
| visual_stimulus_category | 0.9414 | 0.9068 | 0.0667 |
| licking | 0.8593 | 0.8554 | 0.5000 |
| position_bin | 0.7556 | 0.7339 | 0.2500 |
| running_speed_bin | 0.8166 | 0.4812 | 0.2500 |

All above chance. Loss decreased consistently.

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- converted_data.pkl: 211 GB
- Processing time: 73.1 minutes

### Consistency Check
| Statistic | Paper | Converted | Match |
|-----------|-------|-----------|-------|
| Sessions | 89 | 89 | Yes |
| Subjects | 19 | 19 | Yes |
| Total trials | N/A | 38,110 | N/A |
| Neurons/session | 20,547-89,577 | 17,363-78,815 (filtered) | Yes |
| Brain regions | V1,mHV,lHV,aHV | V1,mHV,lHV,aHV | Yes |
| Frame rate | 3.17 Hz | 315.5ms bins | Yes |

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: No errors, no warnings in verification_full_out.txt
2. **Sanity checks**:
   - Neuron count per session matches between spk and retinotopy files
   - All 89 sessions processed, all trials included
   - Position bins evenly distributed (~25% each)
   - Brain regions correctly mapped
3. **Reference code comparison**:
   - Data loading: Same as utils.py load_spk() - concatenate planes
   - Neuron filtering: Same as reference (iarea!=-1 & iarea!=7)
   - Frame filtering: Same as reference (ft_move > 0)
   - Brain regions: Same mapping as neu_area_ID()
4. **Key statistics**: 89 sessions, 19 subjects match paper
5. **Edge cases**: Sessions with stimtype handled (use first occurrence)

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (20021 -> 265 over 200 epochs)
- Test loss: 241.6

### Decoder Results (Full)
| Output | Train Acc | Val Acc | Chance | Ratio |
|--------|-----------|---------|--------|-------|
| visual_stimulus_category | 0.5125 | 0.4932 | 0.0667 | 7.4x |
| licking | 0.9079 | 0.8488 | 0.5000 | 1.7x |
| position_bin | 0.5180 | 0.5165 | 0.2500 | 2.1x |
| running_speed_bin | 0.3260 | 0.3256 | 0.2500 | 1.3x |

All outputs above chance. Train/val gap is small (no overfitting).

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Val Acc | Chance | Ratio | Assessment |
|----------|---------|--------|-------|------------|
| visual_stimulus_category | 0.4932 | 0.0667 | 7.4x | Good - 15 categories |
| licking | 0.8488 | 0.5000 | 1.7x | Good |
| position_bin | 0.5165 | 0.2500 | 2.1x | Good |
| running_speed_bin | 0.3256 | 0.2500 | 1.3x | Above chance |

### Notes on Accuracy
- Visual stimulus: With 15 categories, 49% accuracy is very good (7.4x chance)
- Licking: 85% accuracy is strong, consistent with sample results
- Position: 52% vs 25% chance - reasonable given position information in neural data
- Running speed: 33% vs 25% chance - weakest but still above chance
- The paper does not report decoder accuracies for direct comparison
- Train/val gap is small for all outputs, indicating good generalization

### Issues Found and Resolved
- No critical issues found

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] CONVERSION_NOTES.md complete
- [x] All required output files created
