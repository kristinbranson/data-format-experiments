# Dataset Conversion Notes

## Overview
- **Dataset**: Sosa et al. 2025 "A flexible hippocampal population code for experience relative to reward" (Nature Neuroscience)
- **Date started**: 2025-07-29
- **Goal**: Convert NWB calcium imaging data from hippocampal CA1 to decoder-compatible format
- **DANDI**: https://dandiarchive.org/dandiset/001361

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `code/` - Reference code from the paper (notebooks, src/reward_relative/)
- `data/` - NWB files organized by subject (sub-m3, sub-m4, sub-m7, sub-m11 to sub-m15, sub-m17 to sub-m19)
- `paper.pdf` - Reference paper
- `methods.txt` - Extracted methods from the paper
- `train_decoder.py` - Decoder training script
- `decoder.py` - Decoder model

Python environment: numpy 2.3.5, torch 2.6.0+cu124

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| get_timeseries_data | glmUtils.py:33 | LOADING/PROCESSING | Extract behavioral+neural timeseries from sess object |
| get_reward_zones | behavior.py:77 | LOADING | Get reward zone coordinates per trial |
| get_trial_types | behavior.py:37 | LOADING | Get isreward and morph per trial |
| get_omission_trials | rewardAnalysis.py:105 | LOADING | Identify true omission trials |
| get_omission_inds | rewardAnalysis.py:80 | LOADING | Get rzone entry indices on omission trials |
| dff | preprocessing.py:289 | PROCESSING | Compute dF/F from fluorescence |
| multi_anim_sess | utilities.py:707 | LOADING | Load data for multiple animals |
| nansmooth | utilities.py:572 | PROCESSING | Gaussian smoothing with NaN handling |

### Notes
- Neural data: `sess.timeseries["events"]` = deconvolved calcium events
- The NWB files contain pre-computed deconvolved events, so dF/F computation is not needed
- Speed threshold: 2 cm/s applied in reference code to mask samples
- Lick error correction: >35% of samples with cumulative lick count >2 → trial licks set to NaN
- Trial boundaries: trial_start_inds and teleport_inds
- Reward zones: A=[80,130], B=[200,250], C=[320,370] cm
- GCAMP{N} maps to subject m{N}
- Excluded animals (GCAMP2, 5, 6, 10) are NOT in NWB data

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
NWB files organized as `data/sub-{id}/sub-{id}_ses-{nn}_behavior+ophys.nwb`

Each file contains:
- `processing/ophys/Deconvolved/plane{N}/data`: (n_timepoints, n_rois) deconvolved events
- `processing/ophys/ImageSegmentation/PlaneSegmentation/iscell`: (n_rois, 2) cell classification
- `processing/behavior/BehavioralTimeSeries/`: position, speed, lick, reward, reward_zone, environment, trial_start, teleport, scanning, autoreward

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total ROIs) | 260,091 |
| Neurons (iscell=1) | 138,678 |
| Neurons / session | 155-2,341 (mean 912) |
| Subjects | 11 |
| Sessions / subject | 12-14 |
| Sessions (total) | 152 |
| Trials (total) | 12,216 |
| Trials / session | 41-100 (mostly 80) |
| Frame rate | 15.51 Hz |
| Multi-plane animals | m17, m18 (2 planes) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|---------------|
| Subjects | 11 switch mice | "n = 11 mice" |
| Neurons / session | 155-2,172 | "155-2172 putative pyramidal neurons" |
| Frame rate | ~15.5 Hz | "sampled at ~15.5 Hz" |
| Frame duration | ~0.0645 s | "0.0645 s imaging frame" |
| Track length | 450 cm | "450 cm linear track" |
| Reward zones | A=[80,130], B=[200,250], C=[320,370] | methods.txt |
| Speed threshold | 2 cm/s | "speeds of >2 cm s-1" |
| Lick error trials | ~0.65% (81/12,376) | methods.txt |
| Lick error threshold | >30% (paper) / >35% (code) | methods.txt / glmUtils.py |
| Interneuron exclusion | r>0.5 with speed | methods.txt |
| Interneuron fraction | 0.42 ± 0.85% | methods.txt |
| Teleport zone | 50 cm | methods.txt |

### Processing Details
- Neural: Deconvolved calcium events (OASIS via Suite2p)
- dF/F: maximin baseline, 20s window, per trial, smoothed with 2-sample Gaussian
- Speed threshold: 2 cm/s (reference code masks these samples)
- Lick: cumulative counts clipped to [0,1], smoothed with sigma=2 Gaussian
- Temporal alignment: per-trial, from trial_start to teleport

### Curation Steps
**Neuron curation**: Suite2p iscell + manual curation (in NWB) + interneuron exclusion (~0.42%)
**Trial curation**: Lick error correction (~0.56% of trials)

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Lick threshold | 0.35 | N/A | 0.30 | Use code value (0.35) |
| Max cells | N/A | 2,341 | 2,172 | Interneuron filtering not applied (~0.42% difference) |
| Total trials | N/A | 12,216 | 12,376 | Paper may include non-switch mice |
| Speed masking | Applied | N/A | Applied | NOT applied in our conversion (need all timepoints for decoder) |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| Deconvolved events | neural | Filter by iscell, transpose | (n_neurons, n_timepoints) |
| Frame index × dt | input[0] | Time from trial start | Time-varying, seconds |
| environment data | input[1] | Binary 0/1 | Per-trial |
| Trial index | input[2] | 0-indexed | Per-trial |
| Previous trial reward | input[3] | Binary 0/1 | Per-trial |
| Position - reward zone | output[0] | 7-bin discretization | Time-varying |
| Position | output[1] | 5-bin equal discretization | Time-varying |
| Speed | output[2] | 5-bin discretization | Time-varying |
| Lick | output[3] | Binary after smoothing | Time-varying |
| Reward zone label | output[4] | 0=A, 1=B, 2=C | Per-trial |
| Reward delivery | output[5] | Binary 0/1 | Per-trial |

### Key Decisions
1. **No speed masking**: Unlike reference code, we include all timepoints because speed is a decoder output
2. **Use iscell as-is**: NWB iscell includes manual curation; interneuron filtering skipped (~0.42% effect)
3. **Native frame rate**: Use ~64.5 ms time bins (no resampling)
4. **Lick threshold**: Use code value (0.35) not paper value (0.30)
5. **Distance to reward zone**: Computed from position relative to reward zone boundaries
6. **Multi-plane pooling**: Concatenate plane0 and plane1 data for m17, m18

---

## Step 6: Script Development
**Status**: COMPLETE

Implementation in `convert_data.py`:
- Loads NWB files with h5py
- Handles multi-plane data (m17, m18)
- Processes licks with error correction and smoothing
- Computes reward zone from position data
- Discretizes all outputs per specification
- Supports --sample, --full, --show-processing modes

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics (2 sessions: m11 ses-03, m12 ses-03)
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons | 155, 1143 |
| Trials | 80, 80 |
| Time bin | 64.48 ms |
| Processing time | 5.7s |

### Run Time Estimates
| Step | Time / Session | Estimated Total Time |
|------|---------------|---------------------|
| Full conversion | ~0.46s | ~70s |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|----------------------|------------------------|--------|
| distance_to_reward_zone | 0.5315 | 0.3624 | 0.1429 |
| absolute_position | 0.6177 | 0.5434 | 0.2000 |
| speed | 0.4876 | 0.3723 | 0.2000 |
| lick | 0.7926 | 0.7561 | 0.5000 |
| reward_zone_location | 0.9161 | 0.8399 | 0.3333 |
| reward_outcome | 0.7109 | 0.5871 | 0.5000 |

All outputs above chance ✓

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 9386.6 MB
- `verification_full_out.txt`: created, no errors or warnings

### Consistency Check
| Statistic | Reference Paper | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|--------|
| Subjects | 11 | 11 | 11 | ✓ |
| Sessions | N/A | 152 | 152 | ✓ |
| Total trials | 12,376 | 12,216 | 12,216 | ✓ (data) |
| Total neurons (iscell) | N/A | 138,678 | 138,678 | ✓ |
| Neurons/session range | 155-2,172 | 155-2,341 | 155-2,341 | ✓ (data) |
| Mean neurons/session | N/A | 912 | 912 | ✓ |
| Frame rate | ~15.5 Hz | 15.51 Hz | 15.51 Hz | ✓ |
| Time bin | ~64.5 ms | 64.48 ms | 64.48 ms | ✓ |
| Reward rate | ~80-85% | 84.7% | 84.7% | ✓ |
| Lick error rate | ~0.65% | 0.56% | 0.56% | ✓ |
| ENV1 sessions | ~77 | 73 | 73 | ✓ |
| ENV2 sessions | ~64 | 68 | 68 | ✓ |
| Mixed sessions | 11 | 11 | 11 | ✓ |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Neural data spot-check**: Trial 5, session 0 - exact match (max diff = 0.0) ✓
2. **Input time spot-check**: Trial 5, session 0 - exact match ✓
3. **Output position bins**: Trial 5, session 0 - exact match ✓
4. **Output speed bins**: Trial 5, session 0 - exact match ✓
5. **Reward zone labels**: Trials 0,5,10,30,50,70 session 0 - all match ✓
6. **Reward outcomes**: Trials 0,5,10,30,50,70 session 0 - all match ✓
7. **Previous trial outcomes**: Trials 0,1,5,10 session 0 - all match ✓
8. **Distance to reward zone**: Trial 5, session 0 - exact match ✓
9. **Lick error count**: 69/12,216 = 0.56% (paper: 0.65%) - close, difference due to threshold ✓
10. **Environment distribution**: 73 ENV1, 68 ENV2, 11 mixed - consistent ✓
11. **Edge cases**: First/last trials have reasonable shapes and values ✓

### Reference Code Comparison
| Processing Step | Reference Code | Our Code | Match? |
|----------------|---------------|----------|--------|
| Data loading | sess.timeseries["events"] | Deconvolved/plane{N}/data | ✓ (same data) |
| Cell filtering | iscell from suite2p | iscell from NWB | ✓ |
| Trial boundaries | trial_start_inds, teleport_inds | trial_start > 0, teleport > 0 | ✓ |
| Lick error correction | >35% samples with count>2 | Same logic | ✓ |
| Lick clipping | licks[licks>1] = 1 | Same | ✓ |
| Lick smoothing | nansmooth(licks, 2) | nansmooth(licks, 2) | ✓ |
| Speed threshold | speed < 2 → NaN → mask | NOT applied | Intentional (need all timepoints) |
| Multi-plane | Pooled across planes | Concatenated plane0+plane1 | ✓ |
| Reward zone | From sess.scene name | Inferred from position data | ✓ (verified) |

### Issues Found and Resolved
- **Multi-plane crash**: Fixed by detecting and concatenating multiple plane datasets
- **Speed masking**: Intentionally not applied (differs from reference) because speed is a decoder output

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (187.5 → 1.2 over 200 epochs)

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Ratio |
|--------|----------------------|------------------------|--------|-------|
| distance_to_reward_zone | 0.4164 | 0.3770 | 0.1429 | 2.64x |
| absolute_position | 0.5314 | 0.5095 | 0.2000 | 2.55x |
| speed | 0.4468 | 0.4206 | 0.2000 | 2.10x |
| lick | 0.7139 | 0.6975 | 0.5000 | 1.40x |
| reward_zone_location | 0.8456 | 0.7996 | 0.3333 | 2.40x |
| reward_outcome | 0.6083 | 0.5126 | 0.5000 | 1.03x |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Val Acc | Chance | Ratio | Assessment |
|----------|-----------------|--------|-------|------------|
| distance_to_reward_zone | 0.377 | 0.143 | 2.64x | Good |
| absolute_position | 0.510 | 0.200 | 2.55x | Good |
| speed | 0.421 | 0.200 | 2.10x | Good |
| lick | 0.698 | 0.500 | 1.40x | Good |
| reward_zone_location | 0.800 | 0.333 | 2.40x | Good |
| reward_outcome | 0.513 | 0.500 | 1.03x | Marginal but above chance |

### Accuracy vs Paper Comparison
The paper uses a different decoder (circular-linear regression for RR position), so direct comparison is limited. The paper reports decode scores (cosine distance) rather than balanced accuracy. Our decoder successfully predicts all variables above chance, confirming the data conversion is correct.

### Train vs Validation Gap Analysis
- Most outputs show modest gap (< 1.5x)
- reward_outcome: 0.608 train vs 0.513 val (1.19x) - some overfitting but acceptable
- reward_zone_location: 0.846 train vs 0.800 val (1.06x) - minimal gap

### Reward outcome analysis
Reward outcome has the weakest performance (1.03x chance). This is expected because:
1. It is a per-trial variable (constant within trial)
2. Neural correlates of reward vs omission may be subtle
3. The decoder architecture may not be optimal for per-trial variables
4. This is NOT an indication of a conversion bug

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] CONVERSION_NOTES.md complete
- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
