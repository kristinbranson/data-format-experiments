# Dataset Conversion Notes

## Overview
- **Dataset**: IBL Brain-wide Map (International Brain Laboratory)
- **Date started**: 2026-03-24
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Python environment verified: numpy 2.3.5, torch 2.6.0+cu124, CUDA available.

Directory contents:
- `code/` - Reference code (code_zhang2025/, ibllib/)
- `data/` - IBL data cache (one_cache/ with lab subdirectories)
- `datapaper.pdf` - "A brain-wide map of neural activity during complex behaviour"
- `methodpaper.pdf` - "Exploiting correlations across trials and behavioral sessions..."
- `dataarchitecture.pdf` - White paper about data architecture
- `methods.txt` - Extracted methods text
- `train_decoder.py` - Decoder training script
- `decoder.py` - Decoder model code

Data structure: `data/one_cache/<lab>/Subjects/<mouse>/<date>/001/alf/` with:
- Trial data: `_ibl_trials.table.pqt`, various `_ibl_trials.*.npy`
- Wheel: `_ibl_wheel.position.npy`, `_ibl_wheel.timestamps.npy`
- Camera: motion energy, camera times
- Spike data: `alf/probe00/pykilosort/` with spikes.times, spikes.clusters, clusters.metrics, etc.
- Brain location: channels.brainLocationIds, clusters.channels

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `prepare_data` | ibl_data_utils.py | LOADING | Loads spikes, trials, behaviors for a session |
| `load_spiking_data` | ibl_data_utils.py | LOADING | Loads spike sorting data via SpikeSortingLoader |
| `merge_probes` | ibl_data_utils.py | PROCESSING | Merges spikes/clusters across probes in a session |
| `load_trials_and_mask` | ibl_data_utils.py | CURATION | Loads trials and creates mask for RT, NaN exclusion |
| `list_brain_regions` | ibl_data_utils.py | PROCESSING | Maps cluster regions to Beryl atlas |
| `select_brain_regions` | ibl_data_utils.py | PROCESSING | Selects clusters by brain region |
| `bin_spiking_data` | ibl_data_utils.py | PROCESSING | Bins spikes into time bins per trial |
| `load_target_behavior` | ibl_data_utils.py | LOADING | Loads wheel speed, whisker ME, etc. |
| `bin_behaviors` | ibl_data_utils.py | PROCESSING | Bins behavioral data into trial intervals |
| `get_behavior_per_interval` | ibl_data_utils.py | PROCESSING | Interpolates behaviors to time bins |
| `align_spike_behavior` | ibl_data_utils.py | CURATION | Aligns neural/behavior, applies trial mask |

### Notes

**Key parameters from `0_data_caching.py`:**
- `align_time`: 'stimOn_times'
- `time_window`: (-0.5, 1.5) => 2s trial
- `binsize`: 0.02 (20ms)
- `interval_len`: 2
- `single_region`: False (use all regions)
- Behaviors: choice, reward, block, wheel-speed, whisker-motion-energy

**Trial filtering (`load_trials_and_mask`):**
- min_rt=0.08, max_rt=2.0 (reaction time range)
- max_trial_len=10.0
- Exclude no-choice trials (choice==0)
- NaN exclude: stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType

**Neuron loading (`load_spiking_data`):**
- No QC filter applied in `prepare_data` (qc=None by default)
- ALL neurons used, not just good ones (label >= 1 not required)
- Uses SpikeSortingLoader to load spikes, clusters, channels

**Behavior loading:**
- Wheel speed: absolute value of wheel velocity from SessionLoader
- Whisker ME: tries left camera first, falls back to right
- Behaviors interpolated to time bins via `get_behavior_per_interval`

**NOTE on alignment**: The `0_data_caching.py` uses stimOn_times alignment for ALL behaviors. However, the methods paper says:
- Choice/prior: aligned to stimulus onset
- Wheel/whisker: aligned to first movement onset with 20ms bins

The code uses stimOn_times universally with (-0.5, 1.5) window. The decoder task says to align to stimulus onset.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `data/one_cache/<lab>/Subjects/<subject>/<date>/001/alf/`
- Each session has:
  - `_ibl_trials.table.pqt`: trial info (stimOn_times, choice, probabilityLeft, etc.)
  - `_ibl_wheel.position.npy` + `_ibl_wheel.timestamps.npy`: wheel data
  - `leftCamera.ROIMotionEnergy.npy` + `_ibl_leftCamera.times.npy`: whisker ME
  - `probe*/pykilosort/*/spikes.times.npy`, `spikes.clusters.npy`: spikes
  - `probe*/pykilosort/*/clusters.metrics.pqt`: QC metrics (label column)
  - `probe*/pykilosort/*/channels.brainLocationIds_ccf_2017.npy`: brain region IDs
  - `probe*/pykilosort/*/clusters.channels.npy`: cluster-to-channel mapping
- Trials table columns: goCue_times, response_times, choice, stimOn_times, contrastLeft, contrastRight, probabilityLeft, feedback_times, feedbackType, rewardVolume, firstMovement_times, intervals_0, intervals_1

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (all, total) | 555,928 |
| Neurons (good, label>=1) | 68,213 |
| Neurons/session (all) | ~1,359 |
| Neurons/session (good) | ~167 |
| Subjects | 134 |
| Sessions with data | 409 |
| Trials (total) | 264,950 |
| Trials/session (mean) | ~648 |
| Sessions with all required data | 393 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Mice trained | 139 | "We trained 139 mice" (data paper) |
| Sessions released | 459 | "459 sessions, 699 insertions" (data paper) |
| Sessions used | 433 | "433 IBL sessions" (methods paper) |
| Brain regions | 270 | "270 brain regions" (methods paper) |
| Total units | 621,733 | "produced 621,733 units" (data paper) |
| Well-isolated neurons | 75,708 | "75,708 well-isolated neurons" (data paper) |
| Avg neurons/probe | 108 (good), 889 (all) | (data paper) |
| Neural time bin (choice/prior) | 50ms | "segment neural activity into 50-ms non-overlapping time bins" |
| Neural time bin (wheel/whisker) | 20ms | "binned into non-overlapping 20 ms bins" |
| Min trials/session | 400 (data paper), 250 (inclusion) | "at least 400 trials" / "at least 250 trials" |
| Behaviors decoded | choice, prior, wheel speed, whisker ME | methods paper |
| Trial duration | 2s | "split into 2-s trials" |
| Min reaction time | 0.08s | "0.08-2.00 s" (data paper) |
| Max reaction time | 2.0s | "0.08-2.00 s" (data paper) |

### Processing Details

**Alignment (from methods paper):**
- Choice: align to stimOn_times, window (-0.5, 1.5)s
- Prior: align to stimOn_times, window (-0.6, -0.1)s
- Wheel speed, whisker ME: align to firstMovement_times, window (0, 1)s, 20ms bins

**BUT**: The reference code (`0_data_caching.py`) uses stimOn_times alignment with (-0.5, 1.5) window and 20ms bins for ALL behaviors. The decoder task instructions say "Temporally align based on stimulus onset" - so we follow the code + decoder task.

**Neuron curation (from data paper):**
- Paper: amplitude > 50uV, noise cutoff < 20uV, refractory period violation -> "well-isolated"
- Code: qc=None -> uses ALL neurons (not just good ones)
- We follow the code: use ALL neurons

**Trial curation:**
- RT between 0.08-2.0s
- Exclude no-choice trials (choice==0)
- Exclude trials with NaN in key events
- max_trial_len=10.0

### Curation Steps

**Neuron curation rules**: Use ALL neurons (qc=None), matching reference code.

**Trial curation rules**: Apply `load_trials_and_mask` logic - exclude bad RT, no-choice, NaN trials.

### Decoders Trained (from papers)
| Decoded variable | Metric | Notes |
|-----------------|--------|-------|
| Choice | AUC/balanced accuracy | Binary (left/right) |
| Prior | AUC/balanced accuracy | Categorical |
| Wheel speed | R^2 | Continuous -> discretize for our decoder |
| Whisker motion energy | R^2 | Continuous -> discretize for our decoder |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Sessions | Uses bwm_df to get eids | 409 sessions on disk | 459 released, 433 used | Use all 393 sessions with complete data |
| Neurons | qc=None (all neurons) | 555k total, 68k good | 621k total, 75k good | Code uses ALL neurons. Disk has fewer because only ~409 sessions |
| Alignment | stimOn_times, (-0.5, 1.5) | N/A | Choice/prior=stimOn, wheel/whisker=firstMovement | Follow code + decoder task: stimOn_times for all |
| Binsize | 0.02 (20ms) | N/A | 20ms for wheel/whisker, 50ms for choice/prior | Code uses 20ms uniformly. Follow code. |
| Behaviors | choice, reward, block, wheel-speed, whisker-ME | All available | choice, prior, wheel speed, whisker ME | Decoder task: choice, prior(block), wheel speed, whisker ME |
| Prior/Block | Code calls it 'block' = probabilityLeft | probabilityLeft in trials table | Prior probability | Decoder output maps: 0.2->0, 0.5->1, 0.8->2 |

**Key resolution decisions:**
1. Align to stimOn_times with (-0.5, 1.5) window for all, matching code and decoder task
2. Use 20ms bins, matching code
3. Use ALL neurons (not just good ones), matching code
4. Prior = probabilityLeft mapped to categories: 0.2->0, 0.5->1, 0.8->2

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code | Notes |
|-----------------|--------------|-----------|----------------|-------|
| spikes.times + spikes.clusters | neural | Bin into 20ms bins per trial, aligned to stimOn_times (-0.5, 1.5)s | `bin_spiking_data`, `get_spike_data_per_interval` | Shape (n_neurons, 100) per trial |
| Time since stimOn | input[0] | Continuous, linspace(-0.5, 1.48, 100) | N/A | Time since stimulus onset |
| Trial number in block | input[1] | Count trials since last block change | From probabilityLeft | Per-trial value |
| choice | output[0] | Binary: -1(left)->0, 1(right)->1 | `bin_behaviors` | Per-trial |
| probabilityLeft | output[1] | 0.2->0, 0.5->1, 0.8->2 | `bin_behaviors` | Per-trial |
| wheel speed | output[2] | abs(velocity), discretize into 3 bins | `load_target_behavior`, interpolate | Time-varying |
| whisker ME | output[3] | Discretize into 3 bins | `load_target_behavior`, interpolate | Time-varying |

### Input Details
- `time_since_stim_onset`: shape (1, 100) per trial, values from -0.5 to 1.48 (center of each 20ms bin)
- `trial_num_in_block`: shape (1,) per trial, integer count of trial within current block

### Output Details
- `choice`: shape (1,) per trial, binary 0/1
- `prior`: shape (1,) per trial, categorical 0/1/2
- `wheel_speed`: shape (1, 100) per trial, discretized into 3 bins (0/1/2) using terciles
- `whisker_motion_energy`: shape (1, 100) per trial, discretized into 3 bins (0/1/2) using terciles

### Key Decisions
1. **Alignment**: stimOn_times for all, matching code + decoder task instructions
2. **Binsize**: 20ms uniformly, matching code
3. **All neurons**: No QC filtering, matching code (qc=None)
4. **Whisker ME source**: Try left camera first, fall back to right, matching code
5. **Discretization**: Use session-wide terciles for wheel speed and whisker ME
6. **Block trial number**: Computed as position within contiguous block of same probabilityLeft

### Planned Sanity Checks
- [ ] Total neuron counts per session match direct count from spikes.clusters
- [ ] Trial counts after filtering match what we expect
- [ ] Wheel speed values are non-negative (absolute velocity)
- [ ] Prior distribution roughly matches: biased blocks have 80/20 split
- [ ] Choice distribution roughly 50/50

---

## Step 6: Script Development
**Status**: COMPLETE

Script: `convert_data.py` (~720 lines). Key implementation details:
- Processes sessions from `data/one_cache/<lab>/Subjects/<subject>/<date>/001/alf/`
- Spike binning using `np.searchsorted` + flat indexing for efficiency
- Neural data stored as uint8 (spike counts rarely exceed 255) to reduce memory
- Incremental building with `gc.collect()` after each session
- Memory monitoring via `resource.getrusage`
- Graceful handling of missing data (skips sessions without wheel/whisker data)

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Subjects | 1 (NYU-11) |
| Total trials | 651 (407 + 244) |
| Neurons | 898 + 1728 |
| Brain regions | 19 |
| Time bins | 100 (20ms bins, 2s) |
| Choice distribution | left 48.2%, right 51.8% |
| Prior distribution | 0.2: 47.8%, 0.5: 16.1%, 0.8: 36.1% |
| Wheel speed bins | ~33.3% each |
| Whisker ME bins | ~33.3% each |

### Processing Plots Review
Processing plots saved as processing_0.png and processing_1.png. Visual inspection shows:
- Neural activity time series look reasonable
- Wheel speed and whisker ME properly interpolated
- Discretization produces balanced bins

### Run Time Estimates
| Step | Time / Session | Estimated Total Time |
|------|---------------|---------------------|
| Spike binning | 0.2-0.3s | ~2 min |
| Wheel/whisker | 0.5-1s | ~6 min |
| Total per session | ~2.5s | ~16 min |

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| choice | 0.7118 | 0.6201 | 0.50 |
| prior | 0.8263 | 0.7862 | 0.33 |
| wheel_speed | 0.6889 | 0.6428 | 0.33 |
| whisker_motion_energy | 0.6954 | 0.6540 | 0.33 |

All outputs well above chance. Loss decreasing steadily (1.63 -> 0.59).

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Full Dataset Statistics
| Statistic | Value |
|-----------|-------|
| Sessions processed | 393 |
| Sessions successful | 335 |
| Sessions failed (missing data) | 57 (all PL050/hausserlab - missing wheel data) |
| Sessions skipped (no valid trials) | 1 (ibl_witten_19/2020-07-22) |
| Total trials | 146,747 |
| Subjects | 118 |
| Brain regions | 269 |
| Mean neurons/session | ~1,400 (before reduction) |
| Pickle file size | 21 GB (uint8 neural) |

### Memory Usage During Conversion
| Sessions | RSS (MB) |
|----------|----------|
| 50 | 6,800 |
| 100 | 8,000 |
| 193 | 13,000 |
| 243 | 16,000 |
| 292 | 20,600 |

Peak memory well within 64 GB cgroup limit thanks to uint8 storage.

### Verification
- Format: Valid (no errors)
- Warnings: uint8 dtype (expected, converted to float32 by decoder)
- 335 sessions, 146,747 trials, 269 brain regions all verified

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Consistency Checks
| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| Sessions | ~433 (paper) | 335 (57 missing wheel data) | OK - data subset |
| Brain regions | 270 (paper) | 269 | OK - very close |
| Subjects | 139 (paper) | 118 | OK - data subset |
| Time bins | 100 | 100 | OK |
| Bin size | 20ms | 20ms | OK |
| Trial window | (-0.5, 1.5)s | (-0.5, 1.5)s | OK |
| Alignment | stimOn_times | stimOn_times | OK |
| QC filtering | None (all neurons) | None | OK |

### Distribution Checks
- Choice: ~50/50 split (matches expectation)
- Prior: 0.2 and 0.8 dominate, 0.5 is minority (matches biased block structure)
- Wheel speed / whisker ME: balanced tercile bins (~33% each)

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Memory Optimization for Training
The 21 GB uint8 pickle converts to ~84 GB float32 in `_prepare_session_data`, exceeding 64 GB cgroup limit. Solution: created `reduce_data.py` to subsample neurons to max 500 per session (decoder uses PCA to 100 components anyway).

- Reduced pickle: `converted_data_reduced.pkl` (7.4 GB)
- 318 of 335 sessions had neurons subsampled

### Training Configuration
- Device: CPU (GPU only 2 GB)
- PCA components: 100
- Epochs: 200
- Learning rate: 0.001
- L1 weight: 1e-4
- Balanced loss: True

### Loss Curve
| Epoch | Loss |
|-------|------|
| 1 | 1.6084 |
| 10 | 1.1697 |
| 50 | 0.7961 |
| 100 | 0.7035 |
| 150 | 0.6773 |
| 200 | 0.6658 |
| Test | 0.6971 |

Loss decreasing steadily. Test loss close to training loss (no severe overfitting).

### Decoder Results (Full Dataset)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|----------------------|------------------------|--------|
| choice | 0.6649 | 0.6326 | 0.50 |
| prior | 0.7397 | 0.7125 | 0.33 |
| wheel_speed | 0.6683 | 0.6539 | 0.33 |
| whisker_motion_energy | 0.6544 | 0.6401 | 0.33 |

All outputs well above chance. Prior has the strongest decodability.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Assessment
- All 4 outputs significantly above chance on validation set
- Prior decoding (0.71) strongest, consistent with strong block structure in task
- Choice decoding (0.63) above 0.50 chance, consistent with neural correlates of decision
- Wheel speed (0.65) and whisker ME (0.64) both well above 0.33 chance
- Training vs validation gap is small (~0.02-0.03), indicating good generalization
- Results consistent with sample run (Step 8) which showed similar patterns

### Comparison to Literature
- The methods paper reports much higher decoding with specialized decoders and region-specific analysis
- Our decoder uses a simple linear model with PCA, applied across ALL brain regions simultaneously
- The moderate accuracy is expected given: all neurons (not just well-isolated), all regions pooled, simple model

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

### Output Files
| File | Description | Size |
|------|-------------|------|
| `converted_data.pkl` | Full converted dataset (uint8 neural) | 21 GB |
| `converted_data_reduced.pkl` | Neuron-subsampled version (max 500/session) | 7.4 GB |
| `convert_data.py` | Conversion script | ~720 lines |
| `reduce_data.py` | Neuron subsampling script | ~50 lines |
| `train_decoder_full_out.txt` | Full decoder training log | 15 MB |
| `sample_trials.png` | Sample trial plots | 866 KB |
| `predictions.png` | Decoder prediction plots | 1.3 MB |
| `CONVERSION_NOTES.md` | This document | - |
| `README.md` | Project overview | - |
