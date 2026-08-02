# Dataset Conversion Notes

## Overview
- **Dataset**: International Brain Laboratory (IBL) Brain-Wide Map (BWM) - Neuropixels recordings
- **Date started**: 2025-07-29
- **Goal**: Convert IBL BWM data to decoder-compatible format for predicting choice, prior, wheel speed, and whisker motion energy from neural activity

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `code/` - Reference code (code_zhang2025/, ibllib/)
- `data/` - IBL ONE cache with neural and behavioral data
- `datapaper.pdf` - "A brain-wide map of neural activity during complex behaviour"
- `methodpaper.pdf` - "Exploiting correlations across trials and behavioral sessions"
- `dataarchitecture.pdf` - White paper about data architecture
- `methods.txt` - Key excerpts from papers
- `decoder.py` - Decoder implementation
- `train_decoder.py` - Decoder training script
- `Dockerfile`, `docker-compose.yaml` - Container setup

Python environment: numpy 2.3.5, torch 2.6.0+cu124, CUDA available

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| prepare_data | ibl_data_utils.py | LOADING | Loads spikes (all clusters, no QC filter), trials, behaviors for a session |
| merge_probes | ibl_data_utils.py | LOADING | Merges spikes from multiple probes in same session |
| load_trials_and_mask | ibl_data_utils.py | CURATION | Loads trials, creates mask (min_rt=0.08, max_rt=2.0, max_trial_len=10.0, exclude_nochoice=True) |
| list_brain_regions | ibl_data_utils.py | PROCESSING | Gets Beryl-mapped brain regions |
| select_brain_regions | ibl_data_utils.py | PROCESSING | Selects clusters by region |
| bin_spiking_data | ibl_data_utils.py | PROCESSING | Bins spikes into time bins per trial |
| bin_behaviors | ibl_data_utils.py | PROCESSING | Extracts choice/block/reward from trials_df, bins continuous behaviors |
| load_target_behavior | ibl_data_utils.py | LOADING | Loads wheel speed (abs velocity), whisker motion energy |
| align_spike_behavior | ibl_data_utils.py | CURATION | Removes trials with missing behavior data, applies trials_mask |
| get_behavior_per_interval | ibl_data_utils.py | PROCESSING | Interpolates behavior to uniform time bins per trial |
| interpolate_position | wheel.py | PROCESSING | Interpolates wheel position to 1000 Hz |
| velocity_filtered | wheel.py | PROCESSING | Butterworth lowpass filter (order=8, corner=20Hz), then diff*fs |

### Key Parameters (from 0_data_caching.py)
- interval_len: 2 seconds
- binsize: 0.02 (20 ms)
- align_time: 'stimOn_times'
- time_window: (-0.5, 1.5)
- Behaviors: choice, reward, block, wheel-speed, whisker-motion-energy
- No QC filtering on clusters (qc=None in load_spiking_data)
- Beryl mapping for brain regions

### Notes
- The code loads ALL clusters (not just good ones) - qc=None
- Wheel speed = abs(velocity) where velocity is computed from Butterworth-filtered position
- Whisker motion energy loaded from ROIMotionEnergy files
- Trial mask excludes: RT < 0.08s or > 2s, NaN in key fields, no-choice trials (choice==0)
- Multiple probes merged per session
- Behaviors interpolated to 20ms bins within trial windows

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
IBL ONE cache organized as: `data/one_cache/{lab}/Subjects/{subject}/{date}/001/alf/`
- Trials: `_ibl_trials.table.pqt` (columns: goCue_times, response_times, choice, stimOn_times, contrastLeft, contrastRight, probabilityLeft, feedback_times, feedbackType, rewardVolume, firstMovement_times, intervals_0, intervals_1)
- Spikes: `probe{XX}/pykilosort/*/spikes.times.npy`, `spikes.clusters.npy`
- Clusters: `clusters.channels.npy`, `clusters.depths.npy`, `clusters.metrics.pqt`
- Channels: `channels.brainLocationIds_ccf_2017.npy`
- Wheel: `_ibl_wheel.position.npy`, `_ibl_wheel.timestamps.npy`
- Motion energy: `leftCamera.ROIMotionEnergy.npy`, `rightCamera.ROIMotionEnergy.npy`
- Camera times: `_ibl_leftCamera.times.npy`, `_ibl_rightCamera.times.npy`

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Total probes in bwm_release | 699 |
| Sessions in bwm_release | 459 |
| Subjects | 139 |
| Sessions with spike data | 409 |
| Sessions with trials | 409 |
| Sessions with wheel | 352 |
| Sessions with motion energy | ~387 |
| Complete sessions (all data) | 341 |
| Probes per session | 1 (219) or 2 (240) |
| Example: neurons per probe | ~898 (all), ~76 good (label>=1) |
| Example: trials per session | ~565 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|---------------|
| Sessions used | 433 | "We apply our models to 433 IBL sessions" (Zhang2025) |
| Brain regions | 270 | "covering 270 brain regions" (Zhang2025) |
| Behavioral variables | 4 | "choice, prior, wheel speed, and whisker motion energy" (Zhang2025) |
| Time bins | 20ms | "divided into 20-ms bins, producing T = 100 time steps" (Zhang2025) |
| Trial length | 2s | "split into 2-s trials" (Zhang2025) |
| Time steps per trial | 100 | "T = 100 time steps" (Zhang2025) |
| Choice alignment | stimOn, -0.5 to 1.5s | "align trials to the stimulus onset, -0.5s before to 1.5s post-onset" |
| Prior alignment | stimOn, -0.6 to -0.1s | "align trials to stimulus onset, -0.6s to -0.1s pre-onset" |
| Wheel/whisker alignment | firstMovement, 0 to 1s | "first movement onset, 0 to 1s after" |
| Mice trained | 139 | "We trained 139 mice (94 male and 45 female)" (BWM) |
| Total units | 621,733 | "produced 621,733 units" (BWM) |
| Well-isolated neurons | 75,708 | "75,708 well-isolated neurons" (BWM) |
| Neurons per probe (avg) | 108 good | "averaging 108 per probe" (BWM) |
| Min trials per session | 400 | "Only sessions with at least 400 trials were retained" (BWM) |
| Unbiased block | First 90 trials | "initial 90 unbiased trials" |
| Block length | 20-100 trials | "truncated to lie between 20 and 100 trials" |
| Reward rate | ~80% | From task description |

### Processing Details
- Neural: Align to stimOn_times, window (-0.5, 1.5)s, bin at 20ms -> 100 time steps
- Choice: Binary, per-trial, from trials_df
- Prior: probabilityLeft mapped to categories (0.2->0, 0.5->1, 0.8->2)
- Wheel speed: Continuous, time-varying, discretized into 3 bins
- Whisker motion energy: Continuous, time-varying, discretized into 3 bins
- Trial filtering: min_rt=0.08, max_rt=2.0, exclude no-choice, exclude NaN trials

### Curation Steps

**Neuron curation rules**:
- Reference code uses ALL clusters (no QC filtering, qc=None)
- Brain regions mapped via Beryl mapping
- 'root' and 'void' regions excluded in some analyses but not in data caching

**Trial curation rules**:
- Reaction time between 0.08s and 2.0s
- No NaN in: stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType
- Exclude no-choice trials (choice == 0)
- Max trial length 10.0s (feedback_times - goCue_times)

### Decoders Trained
| Decoded variable | Type | Notes |
|-----------------|------|-------|
| Choice | Binary classification | left/right |
| Prior | 3-class classification | 0.2, 0.5, 0.8 |
| Wheel speed | 3-bin discretized, time-varying | |
| Whisker motion energy | 3-bin discretized, time-varying | |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Sessions | Process from bwm_release (459) | 341 complete, 409 with spikes | 433 used | Some sessions may work with allow_nans=True; will process all sessions with spikes+trials and handle missing behaviors |
| QC filtering | qc=None (all clusters) | 898 total vs 76 good per probe | 75,708 well-isolated (108/probe) | Code uses ALL clusters, not just good ones. Paper stats about well-isolated are for different analysis |
| Alignment | stimOn_times for all in code | - | Choice/prior: stimOn; wheel/whisker: firstMovement | Task spec says align to stimOn for all - will use stimOn |
| Prior bins | 50ms in paper | 20ms in code params | 50ms for prior in paper | Task spec says 20ms bins for all - will use 20ms |

### Key Decision: Alignment
The task description says "Temporally align based on stimulus onset" for all variables. The reference code (0_data_caching.py) also uses stimOn_times with window (-0.5, 1.5). I will follow this approach.

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

---

## Step 7: Sample Conversion and Validation
**Status**: NOT STARTED

---

## Step 8: Sample Decoder Training
**Status**: NOT STARTED

---

## Step 9: Full Conversion and Validation
**Status**: NOT STARTED

---

## Step 10: Critical Review 1
**Status**: NOT STARTED

---

## Step 11: Full Decoder Training
**Status**: NOT STARTED

---

## Step 12: Critical Review 2
**Status**: NOT STARTED

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
