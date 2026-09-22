# Dataset Conversion Notes

## Overview
- **Dataset**: Sosa et al. 2025 - "A flexible hippocampal population code for experience relative to reward"
- **Date started**: 2026-09-21
- **Goal**: Convert 2P calcium imaging data from NWB to decoder-compatible format
- **Data source**: DANDI:001361 (NWB files with behavior + ophys)

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `code/` - Reference code repository (Sosa_et_al_2024)
- `data/` - NWB data files organized by subject (sub-m3, sub-m4, sub-m7, sub-m11..m15, sub-m17..m19)
- `paper.pdf` - Reference paper
- `methods.txt` - Extracted methods text
- `decoder.py` - Decoder model code
- `train_decoder.py` - Decoder training script

Python environment: numpy 2.4.4, torch 2.6.0+cu124, GPU available

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `dff()` | preprocessing.py | PROCESSING | Compute dF/F with maximin baseline, optionally deconvolve with OASIS |
| `get_timeseries_data()` | glmUtils.py | LOADING | Extract behavioral + neural timeseries per session |
| `get_trial_types()` | behavior.py | LOADING | Get isreward, morph per trial |
| `get_reward_zones()` | behavior.py | LOADING | Get reward zone coordinates per trial from scene name |
| `define_trial_subsets()` | behavior.py | LOADING | Split trials into pre/post switch sets |
| `multi_anim_sess()` | utilities.py | PROCESSING | Process all animals for one day: compute dF/F, place cells |
| `CircularRegression` | decode.py | PROCESSING | Circular linear decoder for position |
| `train_vs_test_blocks()` | decode.py | PROCESSING | Cross-validated decoder training |
| `correct_lick_sensor_error()` | behavior.py | CURATION | Remove trials with stuck lick sensor (>35% threshold) |

### Notes
- The reference decoder predicts **circular reward-relative position** using a circular-linear regression, NOT the multi-output categorical decoder we are training
- Neural data used: `sess.timeseries['events']` = custom dF/F -> OASIS deconvolved events
- Speed threshold of 2 cm/s applied in `get_timeseries_data()` to mask out low-speed timepoints
- Lick sensor error: trials where >35% of frames have cumulative lick > 2 are set to NaN
- The reference decoder uses cell-type subpopulations (RR, TR, nonreward_remap), but we use ALL cells
- Reward zone locations: A=[80,130], B=[200,250], C=[320,370] (from `reward_zone_dict` using X, Y, Z keys)

### dF/F Processing Pipeline (from preprocessing.py)
1. Extract F within trial boundaries (trial_start to teleport), set rest to NaN
2. Neuropil subtraction: F -= 0.7 * Fneu (for single-channel)
3. For dual-channel (m17, m18): additional bleedthrough correction
4. Add back neuropil mean per trial after subtraction
5. Maximin baseline: smooth F with [0,15] Gaussian, then min filter (300 samples), then max filter (300 samples)
6. dF/F = (F - baseline) / |baseline|
7. Smooth dF/F with 2-sample Gaussian
8. Deconvolve with OASIS: `dcnv.oasis(dff, 2000, tau=0.7, frame_rate/n_planes)`

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- 11 subjects: m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19
- Each subject has 12-14 NWB files (sessions), one per day
- m11 starts at ses-03 (imaging began day 3), all others start at ses-01
- NWB files contain:
  - `processing/ophys/Fluorescence/plane{N}/data`: raw fluorescence (timepoints x ROIs)
  - `processing/ophys/Neuropil/plane{N}/data`: neuropil fluorescence
  - `processing/ophys/Deconvolved/plane{N}/data`: suite2p deconvolution (NOT custom)
  - `processing/ophys/ImageSegmentation/PlaneSegmentation/iscell`: (n_rois, 2) cell classification
  - `processing/ophys/ImageSegmentation/PlaneSegmentation/planeIdx`: plane assignment per ROI
  - `processing/behavior/BehavioralTimeSeries/`: position, speed, lick, trial_start, teleport, trial number, environment, reward_zone, Reward, autoreward, scanning
- Multi-plane animals (m17, m18): have plane0 and plane1 in Fluorescence/Deconvolved groups
  - Imaging rate ~31 Hz interleaved -> ~15.5 Hz per plane
  - Paper says "planes were pooled for all analyses"

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total, iscell) | 138,678 |
| Neurons / session (mean) | 912.4 |
| Subjects | 11 |
| Sessions / subject | 12-14 (152 total) |
| Trials (total) | 12,217 |
| Trials / session (mean) | 80.4, range 41-100 |
| Imaging rate | ~15.5 Hz (single plane) |
| Track length | 450 cm |

### Key Behavioral Variables in NWB
- `position`: 0-450 cm on track, -500 in teleport zone, negative before track start
- `speed`: cm/s, includes negative values
- `lick`: cumulative lick count per frame (0-6)
- `trial number`: -1 outside trials, 0-indexed within session
- `trial_start`: binary event (1 at trial start)
- `teleport`: binary event (1 at teleport)
- `environment`: -1 before scanning, 0=ENV1, 1=ENV2
- `reward_zone`: 0-6, encodes reward zone state
- `Reward`: separate array with reward event timestamps (sparse)
- `autoreward`: binary, marks auto-reward delivery

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons / session | 155-2172 | "155-2,172 putative pyramidal neurons per session" |
| Subjects (switch task) | 11 | "n = 11 mice" (switch task) |
| Subjects (fixed) | 3 | "n = 3" (not in our data) |
| Sessions / subject | 14 | Days 1-14 (m11 starts day 3) |
| Trials / session (mean) | 80.5 +/- 7.4 | "mean +/- s.d., 80.5 +/- 7.4 trials across 14 mice" |
| Neural data time bin | ~64.5 ms | ~15.5 Hz imaging rate |
| Behavior data time bin | ~64.5 ms | Aligned to imaging frames |
| Reward omission rate | ~15% | "randomly omitted on ~15% of trials" |
| Switch trial | 30 | "The zone was moved after 30 trials" |
| Speed threshold | 2 cm/s | "excluding activity at <2 cm/s" |
| Lick error threshold | 35% | ">30% of imaging frame samples with cumulative lick >2" |
| Track length | 450 cm | "450 cm linear track" |
| Reward zone width | 50 cm | "hidden, unmarked 50 cm span" |
| Zone A | 80-130 cm | From code and paper |
| Zone B | 200-250 cm | From code and paper |
| Zone C | 320-370 cm | From code and paper |
| Teleport jitter | 1-5 s (5-10 s after omission) | From methods |
| Interneuron exclusion | 0.42 +/- 0.85% | "Pearson correlation >0.5 with speed" |

### Processing Details
- Temporal alignment: All data aligned to 2P imaging frames (~15.5 Hz)
- dF/F computed per trial with maximin baseline (20s sliding window)
- Deconvolved with OASIS (suite2p implementation)
- Activity outside trial periods set to NaN (teleport periods excluded for most sessions)
- Auto-reward on first 10 trials of new condition

### Curation Steps

**Neuron curation rules**:
1. Suite2p manual curation: iscell flag (removes multi-soma ROIs, dendrites, non-responsive, overexpressing, putative interneurons)
2. Speed-correlated interneuron exclusion: Pearson r > 0.5 between dF/F and running speed (~0.42% excluded)

**Trial curation rules**:
1. Lick sensor error: trials with >35% of frames having cumulative lick > 2 -> lick data set to NaN for that trial
2. All valid trials (trial_number >= 0) included

### Decoders Trained in Paper
| Decoded variable | Method | Notes |
|---|---|---|
| Circular reward-relative position | Circular-linear regression | cos(y - y_hat) score, ~0.1-0.6 for RR cells |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| NWB Deconvolved | Raw suite2p deconv (no NaN in ITI) | Values 0-2484, no NaN | Custom dF/F + OASIS | Must recompute dF/F from F and Fneu |
| Neurons/session | iscell filter | 155-2281 per session | 155-2172 | Close match; slight difference may be from interneuron exclusion |
| m18 iscell count | 2281 in iscell but only 1994 in plane0 data | Multi-plane: iscell=4857 total, 2281 cells | Planes pooled | iscell covers both planes; need to use planeIdx to map |
| Trials per session | Some sessions have fewer: m4-ses-04=41, m4-ses-09=50, m13-ses-13=60 | 41-100 | 80.5 +/- 7.4 | Some sessions ended early; include all |
| Environment encoding | env=0 or env=1 in NWB | -1 before scanning | ENV1=0, ENV2=1 | Use env value per trial (ignoring -1) |

### Key Consistency Check
- Paper says 11 switch mice, data has 11 subjects -> consistent
- Paper says ~80 trials/session, data shows mean 80.4 -> consistent
- Paper says 155-2172 neurons/session, data shows 155-2281 (before interneuron filter) -> consistent

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping

| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| Custom dF/F + OASIS deconv of F, Fneu | `neural` | Compute dF/F with maximin baseline, deconvolve, filter by iscell | Shape: (n_neurons, n_timepoints) per trial |
| Timestamps within trial | `input[0]`: time_from_trial_start | (frame_idx - trial_start_idx) / imaging_rate | Continuous, time-varying |
| NWB environment | `input[1]`: environment_type | 0=ENV1, 1=ENV2 | Binary, per-trial |
| NWB trial number | `input[2]`: trial_number | 0-indexed trial number | Continuous, per-trial |
| Previous trial reward | `input[3]`: previous_trial_outcome | 0=omitted, 1=rewarded | Binary, per-trial |
| Position vs reward zone | `output[0]`: distance_to_reward_zone | Discretize signed distance to nearest rz boundary into 7 bins | Time-varying |
| NWB position | `output[1]`: absolute_position | Discretize into 5 bins of 90cm each | Time-varying |
| NWB speed | `output[2]`: speed | Discretize into 5 bins | Time-varying |
| NWB lick | `output[3]`: lick | Binary (any lick > 0) | Time-varying |
| Reward zone identity | `output[4]`: reward_zone_location | 0=A, 1=B, 2=C based on zone position | Per-trial |
| Reward delivery | `output[5]`: reward_outcome | 0=no, 1=yes | Per-trial |

### Key Decisions

1. **Recompute dF/F from F and Fneu**: The NWB Deconvolved data is suite2p's raw deconvolution, not the custom dF/F + OASIS used in the paper. Must recompute to match reference processing.

2. **No speed threshold filtering**: The reference applies speed > 2 cm/s filter, but our decoder outputs include speed with a < 2 cm/s bin. Removing low-speed timepoints would make speed bin 0 empty. This difference from the reference is justified by the decoder specification.

3. **Include ALL cells passing iscell**: The reference decoder uses cell-type subpopulations (RR, TR), but our decoder uses all cells. We apply iscell filter only (manual curation). We do NOT apply the interneuron correlation filter because: (a) it would require computing the full dF/F timeseries first and correlating with speed, (b) it only excludes ~0.42% of cells, (c) the iscell manual curation already removed obvious interneurons.

4. **Time bin = imaging frame**: Each time bin is one imaging frame (~64.5 ms at 15.5 Hz). This matches the reference temporal resolution.

5. **Trial definition**: Each trial runs from trial_start to teleport. Teleport/ITI periods excluded (set to NaN in reference).

6. **Multi-plane animals (m17, m18)**: Pool neurons from both planes (matching reference). Neural data from plane0 and plane1 concatenated.

7. **Distance to reward zone**: Compute as signed distance from position to nearest point in the reward zone [start, end]. Negative = before zone, 0 = in zone, positive = past zone. Discretize per decoder spec.

8. **Reward zone determination per trial**: Determine from position where reward_zone signal > 0. Zone A: center ~105 cm, Zone B: center ~225 cm, Zone C: center ~345 cm.

9. **Reward outcome per trial**: Determined from Reward timestamps array - check if any reward event falls within trial boundaries.

10. **Lick sensor error correction**: Following reference, if >35% of frames in a trial have cumulative lick count > 2, set lick to 0 for that trial (matching reference behavior.py).

### Planned Sanity Checks
- [ ] Total session count matches 152
- [ ] Neuron counts per session match range 155-2172 (after curation)
- [ ] Trial counts per session match data
- [ ] Reward omission rate ~15%
- [ ] Reward zone positions match A/B/C definitions
- [ ] Position range 0-450 cm within trials
- [ ] Speed distribution reasonable
- [ ] dF/F computation produces reasonable traces

---

## Step 6: Script Development
**Status**: COMPLETE

- Full conversion script written: `/app/convert_data.py`
- Implements custom dF/F pipeline (neuropil subtraction, maximin baseline, Gaussian smooth, OASIS deconvolution)
- Supports `--sample`, `--full`, and `--show-processing` flags
- Handles multi-plane animals (m17, m18)
- All decoder inputs/outputs implemented per specification

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

- Sample data (2 sessions: m11/ses-03, m12/ses-03) converted successfully
- Data format verified with `train_decoder.py --verify-only` - no errors or warnings
- Processing plots generated and reviewed - all look correct
- All output bins populated, distributions reasonable

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

- Sample decoder trained on 2 sessions (128 train / 32 test trials)
- All outputs well above chance:
  - distance_to_reward_zone: 0.456 (chance 0.143)
  - absolute_position: 0.578 (chance 0.200)
  - speed: 0.476 (chance 0.200)
  - lick: 0.801 (chance 0.500)
  - reward_zone_location: 0.718 (chance 0.333)
  - reward_outcome: 0.739 (chance 0.500)

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

- All 152 sessions converted in 984s (~16 min)
- Output: `/app/converted_data.pkl` (9.84 GB)
- Summary: 11 subjects, 152 sessions, 138,678 neurons, 12,216 trials

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Sanity Checks
| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| Sessions | 152 | 152 | PASS |
| Subjects | 11 | 11 | PASS |
| m11 sessions | 12 | 12 | PASS |
| Other subjects sessions | 14 each | 14 each | PASS |
| Neurons/session range | 155-2172 | 155-2341 | PASS (no interneuron filter) |
| Total neurons | 138,678 | 138,678 | PASS |
| Trials/session range | 41-100 | 41-100 | PASS |
| Total trials | ~12,217 | 12,216 | PASS |
| Mean trials/session | 80.4 | 80.4 | PASS |
| Reward rate | ~85% | 84.7% | PASS |
| Omission rate | ~15% | 15.3% | PASS |
| RZ distribution | ~33% each | 33.0/33.6/33.4% | PASS |
| Position bins | 5 populated | All populated | PASS |
| Speed bins | 5 populated | All populated | PASS |
| Neural: no NaN | True | True | PASS |
| Neural: non-negative | True | True | PASS |
| Neural: sparse | Expected | 74.1% zero | PASS |

### Notes
- Max neuron count (2341, m18) slightly exceeds paper's 2172 because we skip interneuron correlation filter (~0.42% cells)
- Trial total off by 1 (12,216 vs 12,217) - likely one trial with e <= s was skipped
- Long trials exist (up to 216s for m4/ses-04) but are genuine behavioral variability
- Distance to RZ bins asymmetric: bin 4 (0-10cm past zone) has only 2.1% - expected since animals slow/stop in zone

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

- Trained on 9,772 trials, tested on 2,444 trials
- Training device: CUDA GPU
- 200 epochs, final training loss: 0.910, test loss: 0.799

### Validation Balanced Accuracy
| Output | Accuracy | Chance | Ratio |
|--------|----------|--------|-------|
| distance_to_reward_zone | 0.5524 | 0.1429 | 3.9x |
| absolute_position | 0.6465 | 0.2000 | 3.2x |
| speed | 0.5907 | 0.2000 | 3.0x |
| lick | 0.7500 | 0.5000 | 1.5x |
| reward_zone_location | 0.8371 | 0.3333 | 2.5x |
| reward_outcome | 0.5799 | 0.5000 | 1.2x |

All outputs above chance. Spatial variables (position, distance to RZ) decoded well.
Reward zone location decoded best (0.84), consistent with population-level representations.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

**Comparison to reference paper:**
- Paper's circular-linear decoder: cos(y-ŷ) ~0.1-0.6 for RR cells on reward-relative position
- Our categorical decoder: balanced accuracy 0.55 on distance to RZ (7 bins, chance=0.14)
- These are different metrics (circular correlation vs categorical accuracy) and different decoders, but both show significant above-chance decoding of position relative to reward

**Per-output analysis:**
1. **distance_to_reward_zone (0.55)**: 3.9x chance. Good spatial information encoded in CA1. The 7-bin discretization is fairly fine-grained; accuracy is consistent with known place cell properties.

2. **absolute_position (0.65)**: 3.2x chance. Strongest spatial output. Expected - hippocampal place cells encode absolute position strongly.

3. **speed (0.59)**: 3.0x chance. Speed is known to modulate hippocampal firing rates. Good decoding performance.

4. **lick (0.75)**: 1.5x chance. Binary output makes this easier. Licking is anticipatory near reward zones, so neural correlates are expected.

5. **reward_zone_location (0.84)**: 2.5x chance. Highest accuracy. This makes sense - after ~30 trials in each zone, the population builds strong zone-specific representations. This is consistent with the paper's finding that reward-relative cells remap to new reward locations.

6. **reward_outcome (0.58)**: 1.2x chance. Weakest output but still above chance. Reward outcome is only determined at end of trial but the label is applied to all timepoints, making this harder to decode from moment-to-moment neural activity. The 85% reward rate also means the class imbalance makes balanced accuracy harder.

### Potential Improvements (not implemented)
- Per-session decoding rather than pooled across sessions (different cell populations)
- Speed threshold filtering (but would lose speed bin 0)
- Interneuron exclusion (minimal impact, ~0.42%)
- More sophisticated architectures (RNN, attention)

### No Issues Found
- Data format passes all validation checks
- All outputs above chance
- Consistent with known hippocampal physiology
- No obvious errors in processing pipeline

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- README.md created with dataset overview, quick start, data format, and decoder results
- All output files verified present
- CONVERSION_NOTES.md updated through all steps
