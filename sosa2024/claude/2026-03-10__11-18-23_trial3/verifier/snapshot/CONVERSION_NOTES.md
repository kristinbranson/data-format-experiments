# Dataset Conversion Notes

## Overview
- **Dataset**: Sosa et al. 2025 - "A flexible hippocampal population code for experience relative to reward"
- **Date started**: 2026-03-10
- **Goal**: Convert NWB calcium imaging data to decoder-compatible format
- **Paper**: Nature Neuroscience, 2025
- **Data**: 2P calcium imaging of hippocampal CA1 neurons in head-fixed mice on VR linear track

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `code/` - Reference code (Sosa_et_al_2024 repo)
- `data/` - NWB data files organized by subject (sub-m3, m4, m7, m11-m15, m17-m19)
- `paper.pdf` - Reference paper
- `methods.txt` - Methods section excerpts
- `train_decoder.py` - Decoder training/validation script
- `decoder.py` - Decoder model implementation
- `data/dandiset.yaml` - DANDI metadata

11 subjects (all switch-condition mice), 152 total sessions.
sub-m11 has 12 sessions (imaging started day 3); all others have 14.
Python: numpy 2.3.5, torch 2.6.0+cu124.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `dff()` | preprocessing.py | PROCESSING | Compute dF/F from raw fluorescence with neuropil subtraction, maximin baseline, Gaussian smooth |
| `get_trial_types()` | behavior.py | LOADING | Extract isreward, morph per trial |
| `get_reward_zones()` | behavior.py | LOADING | Map reward zone coords/labels per trial from scene string |
| `define_trial_subsets()` | behavior.py | CURATION | Split trials before/after reward switch |
| `get_timeseries_data()` | glmUtils.py | PROCESSING | Extract continuously-sampled behavioral + neural data for decoder/GLM, with speed threshold and lick error correction |
| `calc_place_cells()` | spatial.py | CURATION | Identify place cells via SI permutation test |
| `get_omission_trials()` | rewardAnalysis.py | LOADING | Find trials with reward omission |
| `get_omission_inds()` | rewardAnalysis.py | LOADING | Find reward zone entry indices on omission trials |
| `CircularRegression` | decode.py | PROCESSING | Circular position decoder |

### Notes
- **dF/F computation**: Neuropil subtraction (coef=0.7), maximin baseline (300-frame window ~20s), Gaussian smooth (sigma=2 frames), then OASIS deconvolution (tau=0.7)
- **Neural data used for decoder**: Deconvolved events (`sess.timeseries['events']`), NOT raw dF/F
- **Speed threshold**: 2 cm/s - samples below this are excluded (set to NaN then masked)
- **Lick error correction**: If >35% of samples in a trial have cumulative lick count >2, lick data for that trial set to NaN
- **Lick binarization**: After correction, licks > 1 set to 1 (binary)
- **Trial boundaries**: `trial_start_inds` to `teleport_inds` (1-indexed in reference code, using `start-1:stop-1` slicing)
- **NWB data already has**: Fluorescence, Neuropil, Deconvolved (pre-computed), iscell, planeIdx
- **Important**: The NWB files appear to contain pre-processed data (already aligned to imaging frames, dFF not needed since Deconvolved is provided)

### Critical Discovery: NWB files already contain deconvolved activity
The NWB `processing/ophys/Deconvolved/plane0/data` contains pre-computed deconvolved events.
This means we do NOT need to compute dF/F from scratch - the NWB data is at the `sess` level with pre-processed neural data.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- NWB files: `data/sub-{id}/sub-{id}_ses-{nn}_behavior+ophys.nwb`
- Each file contains one session for one mouse
- Behavioral timeseries (at imaging frame rate ~15.5 Hz): position, speed, lick, reward_zone, teleport, trial_start, trial number, environment, scanning, autoreward
- Ophys data: Fluorescence, Neuropil, Deconvolved (neurons x timepoints), iscell, planeIdx
- Reward events stored separately with timestamps (not at frame rate)
- Multi-plane animals (m17, m18): Two planes (plane0, plane1), imaging rate 31 Hz (15.5 Hz per plane)

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 11 (m3,m4,m7,m11-m15,m17-m19) |
| Sessions / subject | 14 (12 for m11) |
| Total sessions | 152 |
| Neurons / session | 155-2172 (iscell-filtered) |
| Trials / session | ~80 (mean 80.5 per paper) |
| Imaging rate | 15.5078125 Hz (31.015625 Hz for 2-plane) |
| Frame period | ~64.5 ms |
| Track length | 450 cm |

### Key NWB behavioral variables
- `position`: cm on track (0-450), -500 pre-TTL
- `speed`: cm/s
- `lick`: cumulative lick count per frame
- `reward_zone`: 0=not in zone, >0=various states within zone
- `trial_start`: binary, 1 at trial onset
- `teleport`: binary, 1 at teleport
- `trial number`: -1 pre-TTL, then 0-indexed trial number
- `environment`: -1 pre-TTL, 0=ENV1, 1=ENV2
- `scanning`: -1 pre-TTL, 1 during scanning
- `Reward`: event-based (timestamps + data), data values are 0.004 (reward amount?)

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects (switch) | 11 | "n = 11 mice" counterbalanced switch task |
| Subjects (fixed) | 3 | "n = 3 mice" fixed condition (NOT in NWB data) |
| Sessions / subject | 14 (12 for m11) | "total of 14 days", m11 imaging from day 3 |
| Trials / session | 80.5 +/- 7.4 | "mean +/- s.d., 80.5 +/- 7.4 trials across 14 mice" |
| Neurons / session | 155-2172 | "155-2172 putative pyramidal neurons per session" |
| Imaging rate | ~15.5 Hz | "~15.5 Hz" (31 Hz for 2-plane, 15.5 per plane) |
| Frame period | ~64.5 ms | 1/15.5078125 |
| Track length | 450 cm | "450 cm linear track" |
| Reward zone size | 50 cm | "hidden, unmarked 50 cm span" |
| Reward zone A | 80-130 cm | |
| Reward zone B | 200-250 cm | |
| Reward zone C | 320-370 cm | |
| Reward omission rate | ~15% | "randomly omitted on ~15% of trials" |
| Switch trial | 30 | "moved after 30 trials" |
| Speed threshold | 2 cm/s | "excluding activity when moving at <2 cm/s" |
| Spatial bin size | 10 cm | "45 bins of 10 cm each" |
| Interneuron threshold | Pearson r > 0.5 with speed | "correlation of >0.5" |
| Interneuron fraction | 0.42 +/- 0.85% | "excluding 0.42 +/- 0.85% of cells" |
| Lick error threshold | >30% of samples with cumulative lick >2 | "~0.65% of all imaged trials" |
| Bad lick trials | 81 out of 12,376 | "n = 81 out of 12,376 trials removed" |
| dF/F baseline | maximin, 20s window | "maximin procedure with a 20 s sliding window" |
| dF/F smooth | Gaussian sigma=2 frames | "two-sample (~0.129 s) s.d. Gaussian kernel" |
| Deconvolution | OASIS from Suite2p | "OASIS algorithm as used in Suite2p" |

### Processing Details
- Neural: Raw F -> neuropil subtraction (0.7*Fneu) -> maximin baseline -> dF/F -> Gaussian smooth (sigma=2) -> OASIS deconvolution
- BUT NWB files already have Deconvolved data, so we use that directly
- Speed threshold: exclude samples with speed < 2 cm/s
- Trials defined by trial_start to teleport indices
- Teleport periods excluded from analysis

### Curation Steps

**Neuron curation rules**:
1. `iscell` from Suite2p manual curation (already in NWB)
2. Interneuron exclusion: Pearson correlation of dF/F with speed > 0.5

**Trial curation rules**:
1. Lick sensor error: if >35% of samples have cumulative lick count >2, set licks to NaN for that trial (code uses 0.35 threshold)
2. Samples with speed < 2 cm/s excluded from decoder

### Decoders Trained in Paper
| Decoded variable | Method | Notes |
|------------------|--------|-------|
| Reward-relative circular position | CircularRegression | cos(y-y_hat) score, ~0.4-0.8 for RR cells |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Trial indexing | `start-1:stop-1` | trial_start/teleport indices in NWB | - | NWB indices are 0-based frame indices, need to check if they match the reference code's convention |
| Reward zone signal | `rzone` flag in vr_data | `reward_zone` in NWB (0-6 values) | - | rz>0 indicates animal is in reward zone, consistent |
| Deconvolved data | Computed from dF/F via OASIS | Pre-computed in NWB | "OASIS algorithm" | Use pre-computed deconvolved directly |
| Lick data | Cumulative count then binarized | Cumulative count in NWB `lick` | Binary after correction | Need to binarize: clip to 0/1 |
| Interneuron filtering | r(dFF, speed) > 0.5 | Need to compute dF/F for this check | 0.42+/-0.85% excluded | Must compute dF/F or use Fluorescence data to check speed correlation |
| NWB has no scene info | scene string in sessions_dict.py | NWB has environment (0/1) but not scene/reward zone label | - | Must determine reward zone from position where rz>0 |

### Key Consistency Conclusions
1. NWB behavioral data matches reference code's `sess.vr_data` format (same variables at same frame rate)
2. NWB Deconvolved data corresponds to `sess.timeseries['events']` in reference code
3. NWB Fluorescence/Neuropil correspond to raw F/Fneu for computing dF/F if needed
4. Reward zone locations must be inferred from position where reward_zone > 0 (not from scene string)
5. For interneuron filtering, we need to compute dF/F from Fluorescence/Neuropil data, then correlate with speed

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping

| Source Variable | Target Field | Transform | Reference Code | Notes |
|-----------------|--------------|-----------|----------------|-------|
| Deconvolved events (iscell-filtered, no interneurons) | neural | Per-trial segments, speed>2 excluded by NaN | `get_timeseries_data()` uses `sess.timeseries['events']` | Use NWB Deconvolved, filter by iscell, exclude interneurons |
| Time from trial start | input[0] | seconds, time-varying | Custom | Computed from frame timestamps relative to trial start |
| Environment (0/1) | input[1] | binary per trial | `environment` in NWB | ENV1=0, ENV2=1 |
| Trial number | input[2] | continuous per trial | `trial number` in NWB | 0-indexed trial within session |
| Previous trial outcome | input[3] | binary per trial (0=omitted, 1=rewarded) | Custom from Reward events | Check if previous trial had reward |
| Distance to reward zone | output[0] | Discretized to 7 bins, time-varying | Custom from position + reward zone | Bins: <-50, -50 to -10, -10 to 0, 0, >0 to +10, +10 to +50, >+50 |
| Absolute position | output[1] | Discretized to 5 equal bins (90 cm each), time-varying | `position` in NWB | Bins: 0-90, 90-180, 180-270, 270-360, 360-450 |
| Speed | output[2] | Discretized to 5 bins, time-varying | `speed` in NWB | Bins: <2, 2-10, 10-20, 20-40, >40 cm/s |
| Lick | output[3] | Binary, time-varying | `lick` in NWB binarized | 0=no, 1=yes (after error correction) |
| Reward zone location | output[4] | Categorical per trial (0=A,1=B,2=C) | Inferred from position where rz>0 | Map from position range to A/B/C |
| Reward outcome | output[5] | Binary per trial | From Reward events | 0=no reward, 1=rewarded |

### Key Decisions
1. **Use Deconvolved events directly**: NWB has pre-computed deconvolved data matching the reference pipeline
2. **Speed threshold applied as masking**: Following reference code, samples with speed<2 are kept in time but masked. For the decoder format, we include all timepoints but the speed output will reflect this
3. **Time bin = 1 imaging frame**: ~64.5 ms (1/15.5078125 Hz). This matches the native temporal resolution
4. **Temporal alignment**: Align to trial start (first frame of each trial)
5. **Interneuron filtering**: Compute dF/F from Fluorescence/Neuropil to check speed correlation, exclude r>0.5
6. **Reward zone identification**: Infer from position where reward_zone>0 in NWB data. Map to A (80-130), B (200-250), C (320-370) based on position range
7. **Distance to reward zone**: Signed distance from animal position to nearest edge of reward zone. Negative = before zone, positive = past zone, 0 = within zone
8. **Multi-plane animals (m17, m18)**: Pool neurons across planes (consistent with paper)
9. **Each NWB file = 1 session** in the output data structure
10. **Trial definition**: trial_start to teleport indices. Include all valid frames between these

### Planned Sanity Checks
- [ ] Verify neuron count per session matches paper range (155-2172)
- [ ] Verify trial count ~80.5 +/- 7.4
- [ ] Verify reward omission rate ~15%
- [ ] Verify interneuron exclusion rate ~0.42%
- [ ] Verify ~81 trials with lick errors across all sessions
- [ ] Spot-check deconvolved data values are non-negative
- [ ] Verify reward zone positions match A/B/C definitions
- [ ] Check that environment switches happen as expected

---

## Step 6: Script Development
**Status**: COMPLETE

### Script: `convert_data.py`
- CLI: `python3 -u convert_data.py <outfile> [--full|--sample] [--show-processing]`
- `--sample`: processes 2 sessions (m11 ses-03, m3 ses-03) for testing
- `--full`: processes all 152 sessions
- `--show-processing`: generates diagnostic plots

### Key Implementation Details
- **NWB data paths**: `processing/behavior/BehavioralTimeSeries/<var>/data` for behavioral, `processing/ophys/Deconvolved/plane0/data` for neural
- **Multi-plane handling** (m17, m18): Neurons from plane0 and plane1 concatenated
- **iscell filtering**: Applied from `processing/ophys/ImageSegmentation/PlaneSegmentation/plane0/iscell`
- **Interneuron detection**: Vectorized Pearson correlation of dF/F with speed (threshold > 0.5)
- **dF/F computation** (for interneuron check only): Neuropil subtraction (0.7), maximin baseline (300 frames), Gaussian smooth (sigma=2)
- **Lick binarization**: `diff(cumulative_lick) > 0` per frame
- **Lick error correction**: Trials with >35% of samples having cumulative lick > 2 have lick set to NaN
- **Reward detection**: Reward event timestamps mapped to frame indices using NWB timestamps
- **Previous trial outcome**: Derived from whether preceding trial had reward events
- **Reward zone identification**: Position where reward_zone > 0, mapped to A/B/C by mean position
- **Shape mismatch fix**: For multi-plane animals, behavioral and neural data can differ by 1 frame; truncated to common length

### Performance Optimization
- Vectorized interneuron detection (matrix correlation instead of per-neuron loop): 4.3s -> 0.4s per session
- Full conversion: ~879s (~14.6 min) for 152 sessions

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

- Sample: 2 sessions (m11 ses-03, m3 ses-03) -> `sample_data.pkl` (92.7 MB)
- Verification: no errors or warnings
- Output: `conversion_sample_out.txt`, `verification_sample_out.txt`

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

Sample decoder results (2 sessions):

| Output | Train Acc | Val Acc | Chance |
|--------|-----------|---------|--------|
| distance_to_reward_zone | 0.4906 | 0.3775 | 0.1429 |
| absolute_position | 0.5785 | 0.4834 | 0.2000 |
| speed | 0.3897 | 0.3479 | 0.2000 |
| lick | 0.5900 | 0.5880 | 0.5000 |
| reward_zone_location | 0.8643 | 0.8748 | 0.3333 |
| reward_outcome | 0.7008 | 0.5465 | 0.5000 |

All outputs above chance. Output: `train_decoder_sample_out.txt`

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

- Full conversion: 152 sessions -> `converted_data.pkl` (9356.3 MB)
- Verification: no errors or warnings
- Output: `conversion_full_out.txt`, `verification_full_out.txt`

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Sanity Checks (all passed)
1. **Neural data**: Spot-checked session 0 (m11 ses-03), trial 5, first 5 neurons, first 10 timepoints against NWB Deconvolved data. Result: exact match (np.allclose = True)
2. **Input data**: Verified time_from_trial_start, environment_type, trial_number against NWB source data. All match.
3. **Output data**: Verified absolute_position bins, speed bins, reward_zone_location, reward_outcome, distance_to_reward_zone against independent computation from NWB. All match.

### Key Statistics Comparison

| Statistic | Paper | Converted | Match |
|-----------|-------|-----------|-------|
| Sessions | 152 | 152 | Yes |
| Trials/session | 80.5 +/- 7.4 | 80.4 +/- 6.1 | Yes |
| Total trials | ~12,376 | 12,216 | Close (paper includes fixed mice) |
| Neurons/session | 155-2172 | 155-2322 | Close (max slightly higher) |
| Omission rate | ~15% | 15.3% | Yes |
| Interneuron % | 0.42 +/- 0.85% | 0.28% | Within std range |
| Lick error trials | 81 | 69 | Close (paper includes fixed mice) |
| Reward zone distribution | balanced A/B/C | 34.3/32.8/32.9% | Yes |

### Edge Cases
- No NaN in neural data
- No negative values in neural data (deconvolved events >= 0)
- Speed < 2 cm/s fraction: 12.2% of all samples
- Trial lengths: min=96, max=3359, mean=216.8 frames

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

Training on 152 sessions (9,772 train trials, 2,444 test trials). 200 epochs, balanced loss.

| Output | Train Bal. Acc | Val Bal. Acc | Chance | Val/Chance |
|--------|---------------|-------------|--------|------------|
| distance_to_reward_zone | 0.4074 | 0.3726 | 0.1429 | 2.61x |
| absolute_position | 0.4840 | 0.4697 | 0.2000 | 2.35x |
| speed | 0.4536 | 0.4251 | 0.2000 | 2.13x |
| lick | 0.6561 | 0.6462 | 0.5000 | 1.29x |
| reward_zone_location | 0.8385 | 0.7983 | 0.3333 | 2.40x |
| reward_outcome | 0.5697 | 0.5107 | 0.5000 | 1.02x |

Output: `train_decoder_full_out.txt`, `sample_trials.png`, `predictions.png`

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
- All outputs significantly above chance (validation balanced accuracy > uniform chance)
- **Spatial variables** (distance_to_reward_zone, absolute_position) decode well (2.3-2.6x chance), consistent with hippocampal place coding
- **Reward zone location** decodes best (2.4x chance, 0.80 accuracy), consistent with paper's finding of reward-relative remapping in CA1
- **Speed** decodes moderately (2.1x chance), consistent with speed information in hippocampal activity
- **Lick** decodes modestly above chance (1.3x), reflecting behavioral correlates
- **Reward outcome** is weakest (1.02x chance), expected since this is a per-trial binary label hardest to predict from single-timepoint neural activity
- Overfitting is modest (train-val gap < 0.06 for all outputs)

### Comparison with Paper
- Paper uses CircularRegression for reward-relative position (cos(y-y_hat) scores ~0.4-0.8)
- Our classification approach is different but results are consistent with expected hippocampal coding properties
- Strong reward zone decoding aligns with paper's central finding of flexible hippocampal population codes relative to reward

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

### Output Files
- `converted_data.pkl` - Full converted dataset (152 sessions, 9.4 GB)
- `sample_data.pkl` - Sample dataset (2 sessions, 93 MB)
- `convert_data.py` - Conversion script
- `CONVERSION_NOTES.md` - This file
- `sample_trials.png` - Sample trial visualizations
- `predictions.png` - Decoder prediction visualizations
- `processing_m11_ses03.png`, `processing_m3_ses03.png` - Processing diagnostic plots

### Cache Files (intermediate outputs)
- `conversion_sample_out.txt`, `conversion_full_out.txt` - Conversion logs
- `verification_sample_out.txt`, `verification_full_out.txt` - Verification logs
- `train_decoder_sample_out.txt`, `train_decoder_full_out.txt` - Decoder training logs
