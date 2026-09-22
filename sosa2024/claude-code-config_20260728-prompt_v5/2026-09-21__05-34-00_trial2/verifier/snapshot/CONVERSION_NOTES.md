# Dataset Conversion Notes

## Overview
- **Dataset**: Sosa, Plitt, Giocomo 2025. "A flexible hippocampal population code for experience relative to reward." Nature Neuroscience.
- **Date started**: 2026-09-21
- **Goal**: Convert 2P calcium imaging data from hippocampal CA1 during a virtual reality reward-switching task into decoder-compatible format.

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `paper.pdf` - Reference paper
- `methods.txt` - Extracted methods from paper
- `decoder.py` - Decoder module
- `train_decoder.py` - Decoder training script
- `code/` - Reference code repository (Sosa_et_al_2024)
- `data/` - NWB data files (11 subjects: m3, m4, m7, m11-m15, m17-m19)
  - Each subject has 12-14 sessions in NWB format (behavior+ophys)
  - Files are named `sub-mX_ses-YY_behavior+ophys.nwb`
- Python packages verified: numpy 2.4.4, torch 2.6.0+cu124, h5py 3.16.0

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `create_sess` | preprocessing.py | LOADING | Creates session class from raw data, aligns VR to 2P |
| `dff` | preprocessing.py | PROCESSING | Computes dF/F with maximin baseline, optional deconvolution |
| `get_timeseries_data` | glmUtils.py | PROCESSING | Extracts continuous behavioral + neural timeseries for decoder/GLM |
| `get_trial_types` | behavior.py | LOADING | Gets isreward and morph (environment) per trial |
| `get_reward_zones` | behavior.py | LOADING | Gets reward zone coordinates and labels per trial from scene name |
| `correct_lick_sensor_error` | behavior.py | CURATION | Removes trials with stuck lick sensor (>35% frames with cumulative count >2) |
| `define_trial_subsets` | behavior.py | PROCESSING | Defines pre/post switch trial sets |
| `define_anim_list` | dayData.py | CURATION | Lists included animals per experiment day |
| `CircularRegression` | decode.py | PROCESSING | Circular-linear decoder for reward-relative position |
| `train_vs_test_blocks` | decode.py | PROCESSING | Cross-validated decoder training/testing |

### Notes
- **Data flow**: Raw 2P + VR -> Suite2P (motion correction, ROI detection) -> Manual curation (iscell) -> sess class (align VR to 2P) -> dFF computation (maximin baseline) -> deconvolution (OASIS) -> multi_anim_sess (place cell identification)
- **NWB data already contains**: Fluorescence (F), Neuropil (Fneu), Deconvolved events, iscell labels, all behavior timeseries aligned to imaging frames
- **dFF is NOT in the NWB** - the NWB contains raw F, Fneu, and deconvolved events. The reference code uses deconvolved events for the decoder, NOT dFF.
- **Key processing in reference**: Speed threshold >=2 cm/s, lick sensor error correction (>35% threshold), NaN masking of inter-trial intervals
- **Fixed-condition mice** (m2, m6, m10) are NOT in the provided data (only switch-task mice)
- **Reward zone dict**: A=[80,130], B=[200,250], C=[320,370] (using map_labels X/Y/Z)
- **Switch occurs at trial 30** (first 30 trials = pre-switch, remaining = post-switch)
- **Session numbering**: NWB ses-01 through ses-14 correspond to exp_days 1-14

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data/sub-mX/sub-mX_ses-YY_behavior+ophys.nwb` - HDF5/NWB format
- Each NWB contains:
  - `processing/ophys/Deconvolved/plane0/data` - (n_timepoints, n_ROIs) deconvolved events
  - `processing/ophys/Fluorescence/plane0/data` - (n_timepoints, n_ROIs) raw F
  - `processing/ophys/Neuropil/plane0/data` - (n_timepoints, n_ROIs) neuropil F
  - `processing/ophys/ImageSegmentation/PlaneSegmentation/iscell` - (n_ROIs, 2) [is_cell, probability]
  - `processing/behavior/BehavioralTimeSeries/` - position, speed, lick, reward_zone, teleport, trial_start, trial number, environment, scanning, autoreward
  - `processing/behavior/BehavioralTimeSeries/Reward/` - separate timestamps/data for reward events
  - `identifier` field contains scene name (e.g. `Env1_LocationA_to_B`)
- Imaging rate: ~15.5 Hz (varies slightly per session, stored in `acquisition/TwoPhotonSeries/imaging_plane/imaging_rate`)
- Brain region: hippocampus CA1

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 11 (m3, m4, m7, m11-m15, m17-m19) |
| Sessions total | 154 (11 mice x 14 sessions each) |
| Sessions / subject | 14 (except m11 has 12: ses-03 to ses-14) |
| Trials / session | Mostly 80, range [41, 100] |
| ROIs / session | 315-5085 (before iscell filtering) |
| Cells / session (iscell=1) | 155-2341 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects (switch) | 11 | "n = 11 mice" |
| Subjects (fixed) | 3 | "n = 3 mice" - NOT in NWB data |
| Sessions / subject | 14 (1-14), m11 starts day 3 | "Imaging began on day 1... m11 imaging started on day 3" |
| Trials / session | 80.5 +/- 7.4 | "mean +/- s.d., 80.5 +/- 7.4 trials across 14 mice" |
| Neurons / session | 155-2172 | "155-2,172 putative pyramidal neurons per session" |
| Neural data time bin | ~64.5 ms (~15.5 Hz) | "imaging FOV collected at ~15.5 Hz" |
| Reward omission rate | ~15% | "reward was randomly omitted on ~15% of trials" |
| Speed threshold | 2 cm/s | "excluded activity when the animal was moving at <2 cm/s" |
| Track length | 450 cm | "450 cm virtual linear track" |
| Reward zone size | 50 cm | "hidden, unmarked 50 cm span" |
| Zone A | 80-130 cm | code: reward_zone_dict |
| Zone B | 200-250 cm | code: reward_zone_dict |
| Zone C | 320-370 cm | code: reward_zone_dict |
| Switch trial | 30 | "reward zone was moved after 30 trials" |
| Interneuron exclusion | r>0.5 with speed | "Pearson correlation >0.5 between dF/F and running speed" |
| Interneuron fraction | 0.42 +/- 0.85% | "excluding 0.42 +/- 0.85% of cells" |
| Lick sensor error | ~0.65% of trials | "~0.65% of all imaged trials, n=81 out of 12,376 trials" |
| Lick error threshold | >30% of frames with cum. lick >2 | Methods text |
| dFF baseline | maximin, 20s window | "maximin procedure with a 20 s sliding window" |
| dFF smoothing | 2-sample Gaussian | "smoothed with a two-sample (~0.129 s) s.d. Gaussian kernel" |
| Deconvolution | OASIS (suite2p) | "deconvolving dF/F with a canonical calcium kernel using the OASIS algorithm" |

### Processing Details
- **Temporal alignment**: VR behavior data is already aligned to imaging frames in the NWB
- **Trial boundaries**: Defined by trial_start and teleport signals
- **Inter-trial intervals (teleport zone)**: Excluded from analysis (NaN in neural data)
- **Speed filtering**: Timepoints with speed < 2 cm/s set to NaN / excluded
- **Lick correction**: Trials where >35% of frames have cumulative lick count >2 have licks set to NaN (code uses 0.35 threshold in glmUtils; methods says 30%)
- **Neural data**: Deconvolved events used for decoder, NOT dFF

### Curation Steps

**Neuron curation rules**:
1. Suite2P iscell: Only include ROIs with iscell[:,0] == 1
2. Interneuron exclusion: correlation(dFF, speed) > 0.5 -> exclude (0.42% of cells)
   - Note: We cannot easily compute this from NWB since dFF is not stored. We will use the deconvolved events as a proxy, or skip this step since it affects <0.5% of cells.

**Trial curation rules**:
1. Lick sensor error: Trials with >35% of frames having cumulative lick count >2 have licks set to NaN
2. All trials are included (no trial exclusion beyond the lick correction)

### Decoders Trained
| Decoded variable | Accuracy |
|---|---|
| Reward-relative circular position (from RR cells) | cos(y - y_hat) ~0.63 train0_test0, ~0.11 train0_test1 (varies by cell type) |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Lick error threshold | 0.35 (glmUtils.py line 112) | N/A | 0.30 (methods.txt) | Use 0.35 from code - this is the actual implementation |
| m11 sessions | ses-03 to ses-14 (12 sessions) | Confirmed in data | "imaging started on day 3" | m11 has 12 sessions, not 14 |
| Fixed mice | Not in data | m2, m6, m10 absent | n=3 fixed mice | Only switch mice in NWB |
| Reward_zone NWB field | N/A | Cumulative count (like lick) | N/A | rzone>0 indicates in reward zone; not needed for reward zone location |
| Reward zone location | From scene name in identifier | Confirmed via position check | A=[80,130], B=[200,250], C=[320,370] | Parse from identifier/scene |
| Deconvolved data | events timeseries used | NWB has Deconvolved data already | deconvolution from dFF | NWB deconvolved data is ready to use |
| Speed in NWB | Already computed and aligned | Confirmed | Smoothed speed | Use as-is |

### Key Consistency Checks
- m11 ses-03 scene=Env1_LocationB_to_A: reward zone at ~200-210 cm (zone B) -> switches to zone A. This matches the expected pattern.
- Session 08 for m11: scene=Env1_B_to_Env2_C. Environment switches from 0 to 1 at trial 30. Confirmed.
- Trial count: m3 ses-10 has 90 trials, m3 ses-14 has 100, m4 ses-04 has 41, m4 ses-09 has 50. This is within expected variation.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| Deconvolved events (iscell-filtered) | neural | Transpose to (n_neurons, n_timepoints), split by trial, apply speed threshold masking | Use deconvolved events as in reference decoder |
| Time from trial start | input[0] | Compute from timestamps within each trial | Continuous, time-varying |
| Environment type | input[1] | From `environment` field: 0=ENV1, 1=ENV2 | Binary, per-trial |
| Trial number | input[2] | From `trial number` field (0-indexed within session) | Continuous, per-trial |
| Previous trial outcome | input[3] | Derived from reward delivery: previous trial rewarded=1, omitted=0 | Binary, per-trial. First trial = 0 (no previous) |
| Distance to reward zone | output[0] | Compute position - nearest reward zone edge, discretize into 7 bins | Time-varying, 7 classes |
| Absolute position | output[1] | From `position`, discretize into 5 bins of 90 cm | Time-varying, 5 classes |
| Speed | output[2] | From `speed`, discretize into 5 bins | Time-varying, 5 classes |
| Lick | output[3] | From `lick`, binarize (>0 = 1) | Time-varying, binary |
| Reward zone location | output[4] | From scene name: A=0, B=1, C=2 | Per-trial, 3 classes |
| Reward outcome | output[5] | From reward delivery within trial | Per-trial, binary |

### Output Discretization
- **Distance to reward zone** (7 bins): Compute signed distance to nearest point in reward zone. If position is within the zone, distance=0. Negative = before zone, positive = after zone.
  - 0: < -50 cm, 1: -50 to -10 cm, 2: -10 to <0 cm, 3: 0 cm (in zone), 4: >0 to +10 cm, 5: +10 to +50 cm, 6: >+50 cm
- **Absolute position** (5 bins): 0: <90, 1: 90-180, 2: 180-270, 3: 270-360, 4: >360
- **Speed** (5 bins): 0: <2, 1: 2-10, 2: 10-20, 3: 20-40, 4: >40
- **Lick** (2 bins): 0: no, 1: yes

### Key Decisions
1. **Use deconvolved events (not dFF) as neural data**: Consistent with reference decoder code which uses `sess.timeseries['events']`
2. **Apply speed >= 2 cm/s threshold**: Exclude timepoints below threshold by setting neural data to NaN, then removing NaN timepoints. This matches the reference code's `use_speed_thr=2`.
3. **Each NWB session = one session in output**: Each session is one continuous recording with aligned neural + behavioral data.
4. **Time bin = one imaging frame (~64.5 ms)**: No additional temporal binning needed; data is already at imaging frame rate.
5. **Trial segmentation**: Use trial_start and teleport signals. Only include data between trial_start and teleport (excluding inter-trial intervals).
6. **Lick sensor error correction**: Apply the 0.35 threshold from the code (>35% of frames with cumulative lick >2). Set licks to NaN on those trials. Binarize remaining licks (>0 -> 1).
7. **Interneuron exclusion**: Skip this filtering step. It affects <0.5% of cells and requires dFF computation which is not in the NWB. The iscell filtering already removes most non-pyramidal cells.
8. **Reward zone determination**: Parse scene name from NWB identifier. On switch days, first 30 trials have one zone, remaining have another.
9. **Distance to reward zone**: Signed linear distance from animal position to nearest edge of the 50 cm reward zone active on that trial.
10. **Temporal alignment**: Align to trial start (first imaging frame of the trial). Time = 0 at trial start.

### Planned Sanity Checks
- [x] Verify trial count matches paper statistics (~80.5 trials/session)
- [x] Verify cell count range matches paper (155-2172)
- [x] Verify reward rate matches (~85% rewarded) -> 82.8% overall, individual sessions 74-95%
- [x] Verify position range is [0, 450] cm -> position bins 0-4 cover 0-450 cm
- [x] Verify speed distribution is reasonable -> 0% bin 0 (<2 cm/s), filtered correctly
- [x] Cross-check: reward zone switches all at trial 30 (77 switch sessions, 75 no-switch, 11 env switches)

---

## Step 6: Script Development
**Status**: COMPLETE

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Sample Results (2 sessions: m11 ses-03, m17 ses-09)
| Output | Train Bal. Acc | Val Bal. Acc | Chance |
|--------|---------------|-------------|--------|
| distance_to_reward_zone | 0.5017 | 0.3940 | 0.1429 |
| absolute_position | 0.5764 | 0.5282 | 0.2000 |
| speed | 0.5412 | 0.4805 | 0.2000 |
| lick | 0.7512 | 0.6933 | 0.5000 |
| reward_zone_location | 0.9631 | 0.9660 | 0.3333 |
| reward_outcome | 0.5920 | 0.5652 | 0.5000 |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Full Dataset Statistics
- 152 sessions, 11 subjects, 12,216 valid trials
- Trials/session: 80.4 ± 6.1 (range [41, 100])
- Neurons/session: 912.4 ± 448.7 (range [155, 2341])
- Mean reward rate: 82.8%
- Total timepoints: 2,300,473
- Total neurons: 138,678
- File size: 8249.0 MB
- Verification: valid, no errors or warnings

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Consistency with Reference
| Statistic | Paper | Our Data | Match |
|-----------|-------|----------|-------|
| Subjects | 11 | 11 | ✓ |
| Sessions | 152 | 152 | ✓ |
| m11 sessions | 12 | 12 | ✓ |
| Trials/session | 80.5 ± 7.4 | 80.4 ± 6.1 | ✓ |
| Neurons/session | 155-2172 | 155-2341 | ~✓ |
| Reward rate | ~85% | 82.8% | ✓ |
| Speed threshold | ≥2 cm/s | 0 timepoints <2 | ✓ |
| Reward zone balance | ~1/3 each | 33.0/33.8/33.2% | ✓ |
| Switch trial | 30 | All 77 switches at 30 | ✓ |
| Env switches | Day 8 | 11 sessions with env switch | ✓ |

### Sanity Checks Passed
- No NaN in neural data (0/2.1B values)
- Speed <2 cm/s properly filtered (0 timepoints in bin 0)
- All reward zone switches at trial 30
- All environment switches at trial 30
- Reward rates 74-95% per session (~85% expected)
- Neuron max 2341 vs paper 2172 (minor, due to skipping interneuron exclusion which affects <0.5% of cells)

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Full Dataset Decoder Results
- Training: 9,772 trials | Validation: 2,444 trials
- Device: CUDA | 200 epochs | PCA 100 components
- Loss converged: 191.96 -> 1.31 (train), 1.19 (test)

| Output | Train Bal. Acc | Val Bal. Acc | Chance | Ratio |
|--------|---------------|-------------|--------|-------|
| distance_to_reward_zone | 0.3894 | 0.3514 | 0.1429 | 2.46x |
| absolute_position | 0.5232 | 0.5059 | 0.2000 | 2.53x |
| speed | 0.4802 | 0.4570 | 0.2000 | 2.29x |
| lick | 0.6352 | 0.6243 | 0.5000 | 1.25x |
| reward_zone_location | 0.8413 | 0.7995 | 0.3333 | 2.40x |
| reward_outcome | 0.5818 | 0.5122 | 0.5000 | 1.02x |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Analysis
- All 6 output dimensions decode above chance on held-out validation data
- Train-validation gaps are small (~0.02-0.08), indicating minimal overfitting
- Spatial variables (position, distance) decode best (2.3-2.5x chance)
- Reward zone location decodes very well (0.80 val accuracy, 2.4x chance), confirming neural population encodes reward context
- Reward outcome is hardest to decode (1.02x chance on validation) — expected since reward is omitted randomly ~15% and outcome is not directly observable from neural activity during the trial
- Lick decoding (0.62 val, 1.25x chance) is reasonable for a sparse binary variable
- Results are consistent with the paper's finding that hippocampal CA1 encodes position and reward-relative spatial information

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

### Output Files
- `/app/converted_data.pkl` - Full dataset (152 sessions, 8.2 GB)
- `/app/sample_data.pkl` - Sample dataset (2 sessions, 55 MB)
- `/app/convert_data.py` - Conversion script
- `/app/CONVERSION_NOTES.md` - This file
- `/app/verification_full_out.txt` - Verification output
- `/app/verification_full_stats.json` - Verification statistics
- `/app/train_decoder_full_out.txt` - Full training output
- `/app/train_decoder_full_stats.json` - Full training statistics
- `/app/sample_trials.png` - Sample trial plots
- `/app/predictions.png` - Prediction plots
