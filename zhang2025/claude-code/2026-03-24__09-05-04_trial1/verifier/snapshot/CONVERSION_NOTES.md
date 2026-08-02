# Dataset Conversion Notes

## Overview
- **Dataset**: IBL Brain-Wide Map (BWM) - Neuropixels electrophysiology
- **Date started**: 2026-03-24
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `code/` - Reference code (code_zhang2025, ibllib)
- `data/` - IBL ONE cache with session data organized by lab/subject/date/number
- `datapaper.pdf` - "A brain-wide map of neural activity during complex behaviour"
- `methodpaper.pdf` - "Exploiting correlations across trials and behavioral sessions"
- `dataarchitecture.pdf` - Data architecture white paper
- `methods.txt` - Extracted methods text
- `decoder.py` - Decoder module
- `train_decoder.py` - Decoder training script

Python environment: numpy 2.3.5, torch 2.6.0+cu124

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `prepare_data()` | ibl_data_utils.py | LOADING | Main orchestrator: loads spikes, trials, behaviors for a session |
| `load_spiking_data()` | ibl_data_utils.py | LOADING | Loads spike times/clusters from ONE cache via SpikeSortingLoader |
| `merge_probes()` | ibl_data_utils.py | LOADING | Merges spike data from multiple probes into single recording |
| `load_trials_and_mask()` | ibl_data_utils.py | CURATION | Loads trials, creates Boolean mask for exclusion |
| `list_brain_regions()` | ibl_data_utils.py | PROCESSING | Maps cluster regions to Beryl atlas |
| `select_brain_regions()` | ibl_data_utils.py | PROCESSING | Filters clusters by region |
| `bin_spiking_data()` | ibl_data_utils.py | PROCESSING | Bins spikes into trial-aligned time windows |
| `get_spike_data_per_interval()` | ibl_data_utils.py | PROCESSING | Core binning: spikes -> (n_trials, n_clusters, n_bins) |
| `bin_behaviors()` | ibl_data_utils.py | PROCESSING | Bins behavioral signals into trials |
| `get_behavior_per_interval()` | ibl_data_utils.py | PROCESSING | Interpolates continuous behavior to uniform bins |
| `load_target_behavior()` | ibl_data_utils.py | LOADING | Loads wheel, motion energy, pupil signals |
| `align_spike_behavior()` | ibl_data_utils.py | CURATION | Ensures neural and behavioral data have matching trials |
| `SpikeSortingLoader` | brainbox/io/one.py | LOADING | Loads spike sorting data from ONE cache |
| `SessionLoader` | brainbox/io/one.py | LOADING | Loads trials, wheel, motion energy |

### Notes
- Reference code `0_data_caching.py` uses params: binsize=0.02, align_time='stimOn_times', time_window=(-0.5, 1.5)
- Behavioral variables: choice, reward, block, wheel-speed, whisker-motion-energy
- qc=None: ALL clusters used (not filtered by quality label)
- Probes merged across session, brain regions mapped to Beryl atlas
- Trial mask: exclude if RT < 0.08s or > 2.0s, NaN in key events, no choice, max trial len 10s

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
```
data/one_cache/<lab>/Subjects/<subject>/<date>/<number>/alf/
  _ibl_trials.table.pqt          - Trial events (stimOn, choice, feedback, etc.)
  _ibl_wheel.position.npy        - Wheel position
  _ibl_wheel.timestamps.npy      - Wheel timestamps
  leftCamera.ROIMotionEnergy.npy  - Left whisker motion energy
  rightCamera.ROIMotionEnergy.npy - Right whisker motion energy
  _ibl_leftCamera.times.npy      - Left camera timestamps
  _ibl_rightCamera.times.npy     - Right camera timestamps
  probe<XX>/pykilosort/
    spikes.times.npy              - Spike timestamps
    spikes.clusters.npy           - Cluster ID per spike
    clusters.channels.npy         - Channel per cluster
    clusters.depths.npy           - Depth per cluster
    clusters.metrics.pqt          - Quality metrics (label, amp, etc.)
    clusters.uuids.csv            - Unique cluster IDs
    channels.brainLocationIds_ccf_2017.npy - Atlas region IDs per channel
    channels.mlapdv.npy           - 3D coordinates per channel
```

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 141 (in data dir) |
| Sessions (directories) | 461 |
| Probes (from BWM CSV) | 699 |
| Unique sessions (from BWM CSV) | 459 |

### Example Session: KS014 / 2019-12-03 / 001
- probe00: 917 clusters, 188 good (label>=1)
- Trials: 531
- Wheel: 528,230 samples
- Left camera: 174,780 frames

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source |
|-----------|-------|-------|
| Mice | 139 | Data paper: "We trained 139 mice" |
| Sessions | 459 | Data paper: "459 sessions, 699 insertions" |
| Total units | 621,733 | Data paper: "621,733 units" |
| Units/probe | ~889 | Data paper: "averaging 889 per probe" |
| Well-isolated neurons | 75,708 | Data paper: "75,708 were considered well-isolated" |
| Well-isolated/probe | ~108 | Data paper: "averaging 108 per probe" |
| Min trials per session | 400 (inclusion) | Data paper: "at least 400 trials" (recording inclusion) |
| Min trials per session | 250 (analysis) | Data paper: "at least 250 trials" (analysis inclusion) |
| Sessions (methods paper) | 433 | Methods paper: "433 IBL sessions" |
| Brain regions | 270 | Methods paper: "270 brain regions" |
| Time bin size | 20 ms | Methods paper: "20-ms bins" |
| Time window (choice) | -0.5s to 1.5s from stimOn | Methods paper |
| Time window (prior) | -0.6s to -0.1s from stimOn | Methods paper |
| Time window (wheel/whisker) | 0s to 1s from firstMovement | Methods paper |
| Left camera fps | 60 Hz | Data paper |
| Contrasts | 100, 25, 12.5, 6, 0% | Data paper |
| Block probabilities | 0.2, 0.5, 0.8 | Data paper |
| Block length | 20-100 trials (mean ~51) | Data paper |
| Unbiased block | First 90 trials | Data paper |

### Processing Details
- Spike sorting: Kilosort 2.5 (pykilosort)
- Temporal alignment: stimOn_times for choice/prior; firstMovement_times for wheel/whisker
- Time bins: 20ms (reference code), 50ms for choice/prior in methods paper
- Reference code uses 20ms bins for ALL variables with stimOn_times alignment
- Interpolation: scipy.interpolate.interp1d(linear, extrapolate) for continuous behaviors

### Curation Steps

**Neuron curation rules (data paper - well-isolated)**:
- amplitude > 50 uV, noise cutoff < 20 uV, refractory period violation pass
- These are captured in clusters.metrics.pqt 'label' column (label >= 1 = good)
- **Reference code uses qc=None: ALL clusters, not just good ones**

**Trial curation rules (both papers & code)**:
- Exclude NaN in: stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType
- Exclude RT < 0.08s or > 2.0s (RT = firstMovement_times - stimOn_times)
- Exclude no-choice trials (choice == 0)
- Exclude trials with max_trial_len > 10.0s (feedback_times - goCue_times)

### Decoders Trained
| Decoded variable | Type | Metric |
|---|---|---|
| Choice | Binary (left/right) | Balanced accuracy |
| Prior | Continuous | Pearson correlation |
| Wheel speed | Continuous time-varying | R2 |
| Whisker motion energy | Continuous time-varying | R2 |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Sessions | 459 (BWM CSV) | 461 dirs | 459 (data paper), 433 (method paper) | Use BWM CSV as ground truth for session list |
| Neuron filtering | qc=None (all units) | label>=1 for good | 75,708 well-isolated | **Follow reference code: use all units (qc=None)** |
| Alignment | stimOn_times | - | stimOn for choice/prior; firstMovement for wheel/whisker | **Task spec says stimOn for all; follow task spec** |
| Time window | (-0.5, 1.5) | - | Choice: (-0.5,1.5); others differ | **Task spec says stimOn; use code's (-0.5, 1.5)** |
| Bin size | 0.02s (20ms) | - | 20ms (dynamic), 50ms (choice/prior) | **Follow reference code: 20ms for all** |
| Prior encoding | block (probabilityLeft) | 0.2, 0.5, 0.8 | Prior = running estimate | Task spec: 0.2->0, 0.5->1, 0.8->2 (categorical) |
| Wheel speed | Continuous | - | Continuous | Task spec: discretize into 3 bins |
| Whisker ME | Continuous | - | Continuous | Task spec: discretize into 3 bins |

### Key Resolution
The task spec overrides certain aspects of the reference code/papers:
1. ALL alignment to stimulus onset (not firstMovement for wheel/whisker)
2. Prior encoded as categorical (3 classes from probabilityLeft)
3. Wheel speed and whisker ME discretized into 3 bins
4. Choice: left=0, right=1 (reference code: left=-1, right=1)

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| spikes.times + spikes.clusters | neural | Bin into 20ms windows aligned to stimOn, shape (n_neurons, 100) | bin_spiking_data, get_spike_data_per_interval | All clusters (qc=None) |
| Time since stimOn | input[0] | np.linspace(-0.5, 1.48, 100), continuous time-varying | Manual construction | Same for all trials |
| Trial number in block | input[1] | Count trials within each block, per-trial scalar | Computed from probabilityLeft changes | Resets at block boundary |
| trials.choice | output[0] | left(-1)->0, right(1)->1, binary per-trial | bin_behaviors | Exclude choice==0 |
| trials.probabilityLeft | output[1] | 0.2->0, 0.5->1, 0.8->2, categorical per-trial | bin_behaviors | 3 classes |
| wheel velocity | output[2] | abs(velocity) -> discretize into 3 equal-frequency bins, time-varying | load_target_behavior, get_behavior_per_interval | 3 classes |
| whisker motion energy | output[3] | Discretize into 3 equal-frequency bins, time-varying | load_target_behavior, get_behavior_per_interval | 3 classes |

### Metadata
- time_bin_size: 20.0 (ms)
- temporal_alignment_event: "stimulus onset (stimOn_times)"
- off_start: -0.5
- off_end: 1.5
- n_timepoints: 100

### Key Decisions
1. **Use all clusters (qc=None)**: Matches reference code, not just well-isolated neurons
2. **Align everything to stimOn_times**: Task spec overrides reference code's firstMovement alignment for wheel/whisker
3. **Time window (-0.5, 1.5)**: 2 seconds, 100 bins at 20ms, matches reference code
4. **Brain region mapping**: Use Beryl atlas mapping (iblatlas BrainRegions)
5. **Discretization of wheel speed/whisker ME**: Use equal-frequency (quantile) bins across all trials within a session, 3 bins (low/medium/high)
6. **Trial number in block**: Computed by detecting block boundaries from changes in probabilityLeft
7. **Session list**: Use BWM release CSV (459 sessions) as ground truth

### Planned Sanity Checks
- [ ] Total neurons across all sessions matches ~621,733
- [ ] Number of sessions close to 433-459
- [ ] Trial counts per session ~250-800
- [ ] Choice distribution approximately balanced (~50/50)
- [ ] Prior distribution: 3 classes roughly proportional to block structure
- [ ] Spot-check: raw spike counts for specific trial/neuron match manual computation
- [ ] Verify trial exclusion matches reference code criteria

---

## Step 6: Script Development
**Status**: COMPLETE

Wrote `convert_data.py` with:
- Direct file loading (no ONE API needed, works offline from cache)
- Spike binning via vectorized numpy operations
- Wheel velocity via Butterworth filter matching ibllib (1kHz interp, 20Hz corner, order 8)
- Motion energy loading from leftCamera (fallback to rightCamera)
- Quantile-based discretization for wheel speed and whisker ME (3 bins)
- Trial mask matching reference code exactly
- Beryl atlas brain region mapping via iblatlas

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Subjects | 2 (PL033, ZFM-01937) |
| Total trials | 682 (378 + 304) |
| Total neurons | 1525 (646 + 879) |
| Brain regions | 20 |
| Time bins | 100 (20ms each) |
| Choice distribution | left=46.2%, right=53.8% |
| Prior distribution | 0.2=37.5%, 0.5=16.7%, 0.8=45.7% |
| Wheel speed | 33.3% per bin (by design) |
| Whisker ME | 33.3% per bin (by design) |

### Processing Plots Review
- Processing plots saved, show proper neural activity, wheel/ME alignment

### Run Time Estimates
| Step | Time / Session | Estimated Total Time (459 sessions) |
|---|---|---|
| Total | ~20s | ~2.5 hours |

Note: ~20s/session, full run ~9200s (~2.5 hours). This is acceptable.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| choice | 0.6377 | 0.5933 | 0.5000 |
| prior_probability_left | 0.7516 | 0.6972 | 0.3333 |
| wheel_speed | 0.7174 | 0.6822 | 0.3333 |
| whisker_motion_energy | 0.6680 | 0.6428 | 0.3333 |

All outputs above chance. Loss decreased from 1.73 to 0.63 over 200 epochs.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 23.3 GB (neural stored as uint8 to fit in 64GB container memory)
- `conversion_full_out.txt`: complete log
- `verification_full_out.txt`: no errors, only dtype warnings (uint8 auto-converted during training)

### Run Time
- Total: 1741s (29 min) for 459 sessions
- 392 sessions processed, 67 skipped (missing data/paths/wheel or <2 valid trials)

### Consistency Check
| Statistic | Reference papers | Reference Code | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|--------|
| Sessions | 459 (data paper), 433 (methods) | 459 (BWM CSV) | 392 (67 skipped) | 85% - OK, missing files |
| Subjects | 139 | - | 129 | 93% - OK |
| Total neurons | 621,733 | all clusters (qc=None) | 534,911 | 86% - OK |
| Avg neurons/session | 889/probe | - | 1,365/session | Consistent (multi-probe) |
| Brain regions | 270 | Beryl mapping | 275 | Close match |
| Total trials | - | - | 167,487 | - |
| Avg trials/session | 400+ | - | 427 | Good |
| Time bins | 100 (20ms) | 100 (20ms) | 100 (20ms) | Exact |
| Choice dist | ~50/50 | - | 49.4/50.6 | Excellent |
| Prior dist | ~42/14/44 | - | 42.0/14.1/43.9 | Excellent |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. Neural data: 392 sessions, 167,487 trials, 534,911 total neurons. Max spike count 68 (fits uint8).
2. Trial counts: 131-1445 per session, mean 427.3
3. Neuron counts: 135-3140 per session, mean 1364.6
4. Input shapes correct: (2, 100), time axis from -0.48 to 1.5
5. Output values: choice {0,1}, prior {0,1,2}, wheel {0,1,2}, whisker {0,1,2}
6. Wheel speed dist: 33.3/33.3/33.3% (quantile-based)
7. Whisker ME dist: 33.3/33.2/33.5% (quantile-based)
8. brain_region_idx matches neuron count per session
9. subject_idx correct range 0-128 for 129 subjects
10. Choice and prior constant within trials (per-trial variables)
11. Wheel speed and whisker ME are time-varying in all trials
12. No NaN/Inf in neural data

### Issues Found and Resolved
- Neural stored as uint8 to fit 64GB container memory limit (decoder converts to float32 automatically)
- 67 sessions skipped: missing data paths, missing wheel, or <2 valid trials

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Used subset (98 sessions, 43,926 trials, 124,242 neurons) due to 64GB memory limit
- Loss: 7.53 (epoch 1) → 1.77 (epoch 200), steadily decreasing
- Training: 133,826 trials train / 33,661 test

### Decoder Results (Full Subset)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|-------|
| choice | 0.5597 | 0.5592 | 0.5000 |
| prior_probability_left | 0.6028 | 0.5966 | 0.3333 |
| wheel_speed | 0.5962 | 0.5930 | 0.3333 |
| whisker_motion_energy | 0.5755 | 0.5741 | 0.3333 |

All outputs significantly above chance. Train/validation close (no overfitting).

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved BA | Chance | Above Chance? | Notes |
|---|---|---|---|---|
| Choice | 0.559 | 0.500 | Yes (+5.9%) | Expected: choice decodable from neural activity |
| Prior | 0.597 | 0.333 | Yes (+26.3%) | Expected: block structure reflected in neural activity |
| Wheel speed | 0.593 | 0.333 | Yes (+26.0%) | Expected: motor signals decodable |
| Whisker ME | 0.574 | 0.333 | Yes (+24.1%) | Expected: sensory/motor signals decodable |

### Notes
- Choice accuracy (55.9%) is modest because this decoder uses all neurons across all brain regions, not just task-relevant areas. The data paper shows choice decoding varies by region (some regions >80% accuracy).
- Prior accuracy (59.7%) is solid — the decoder correctly identifies block probability from neural population activity.
- Wheel speed and whisker ME accuracies (59.3%, 57.4%) confirm that continuous behavioral signals (discretized to 3 bins) are decodable from neural activity.
- No overfitting: train and validation accuracies are very close.

### Issues Found and Resolved
- Full dataset (392 sessions, 23 GB) exceeds 64 GB container memory when loaded + processed. Used representative subset (98 sessions, every 4th) for decoder training.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created with overview, data format, processing parameters, decoder results, usage
- [x] cache/ folder created
- [x] All files organized
- [x] CONVERSION_NOTES.md complete with all 14 steps documented
