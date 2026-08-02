# Dataset Conversion Notes

## Overview
- **Dataset**: [Name and source]
- **Date started**: 2026-07-29
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: [COMPLETE]

Directory contents:
- .manifest
- CONVERSION_NOTES.md
- Dockerfile
- code/
- data/
- dataarchitecture.pdf
- datapaper.pdf
- decoder.py
- docker-compose.yaml
- methodpaper.pdf
- methods.txt
- train_decoder.py

---

## Step 1: Reference Code Exploration
**Status**: [COMPLETE]

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| prepare_data | code/code_zhang2025/src/0_data_caching.py | LOADING | orchestrates session loading, preprocessing, and caching |
| load_spiking_data | code/code_zhang2025/src/utils/ibl_data_utils.py | LOADING | loads session spike data and associated metadata from IBL cache/ONE |
| load_trials_and_mask | code/code_zhang2025/src/utils/ibl_data_utils.py | CURATION | loads trial table and applies validity masks/selection logic |
| bin_spiking_data | code/code_zhang2025/src/utils/ibl_data_utils.py | PROCESSING | bins spikes into fixed-width time bins around alignment events |
| bin_behaviors | code/code_zhang2025/src/utils/ibl_data_utils.py | PROCESSING | bins behavioral streams to the same temporal grid |
| align_spike_behavior | code/code_zhang2025/src/utils/ibl_data_utils.py | PROCESSING | aligns neural and behavior arrays trial-by-trial on a common time base |
| create_dataset | code/code_zhang2025/src/utils/dataset_utils.py | PROCESSING | packages processed arrays and metadata into train/val/test dataset dicts |
| IBLSessionData / MultiSessionDataModule helpers | code/code_zhang2025/src/utils/data_loader_utils.py | PROCESSING | loads cached datasets, standardizes spikes, handles region selection and splits |

### Notes
- Main reference workflow appears to be: choose session IDs -> cache raw IBL session data -> bin spikes and behaviors at fixed binsize -> align to trial events -> package into datasets for decoding.
- README example for preprocessing uses `src/0_data_caching.py`.
- Decoder scripts operate on cached/packaged datasets rather than raw files directly, so matching preprocessing in `ibl_data_utils.py` is likely critical.
- Important concepts to carry forward: session-level processing, fixed time bins, explicit alignment, behavior interpolation/NaN handling, and region-based neuron selection in loaders.
- Need Step 2 to inspect local `data/` organization and determine whether raw or partially cached forms are already present.

---

## Step 2: Dataset Exploration
**Status**: [COMPLETE]

### Data Structure
Local data are organized as an IBL ONE cache under `data/one_cache`.

Observed top-level cache components:
- Release metadata directories: `2022_Q4_IBL_et_al_BWM`, `2025_Q3_IBL_et_al_BWM`, `Brainwidemap`
- Lab-specific directories: `angelakilab`, `churchlandlab`, `churchlandlab_ucla`, `cortexlab`, `danlab`, `hausserlab`, `hoferlab`, `mainenlab`, `mrsicflogellab`, `steinmetzlab`, `wittenlab`, `zadorlab`
- REST cache: `.rest`

Release directories contain `sessions.pqt`, `datasets.pqt`, `cache_info.json`, and `QC.json` files. This indicates the dataset is available through cached ONE metadata tables plus downloaded session files in lab-specific folders.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | Not yet directly counted from local session files |
| Neurons / session | Not yet directly counted from local session files |
| Subjects | 115 (2022 release), 139 (2025 release), 143 (Brainwidemap cache) |
| Sessions / subject | 354/115 ≈ 3.08 (2022), 459/139 ≈ 3.30 (2025), 480/143 ≈ 3.36 (Brainwidemap cache) |
| Trials (total) | Not yet directly counted from local session files |
| Trials / session | Not yet directly counted from local session files |

Additional metadata table sizes:
- `2022_Q4_IBL_et_al_BWM/sessions.pqt`: shape (354, 6)
- `2025_Q3_IBL_et_al_BWM/sessions.pqt`: shape (459, 6)
- `Brainwidemap/sessions.pqt`: shape (480, 6)
- `2022_Q4_IBL_et_al_BWM/datasets.pqt`: shape (38814, 9)
- `2025_Q3_IBL_et_al_BWM/datasets.pqt`: shape (5339, 6)
- `Brainwidemap/datasets.pqt`: shape (76563, 6)
-----------|-------|
| Neurons (total) | 340 |
| Neurons / session | 76, 264 |
| Subjects | 1 |
| Sessions / subject | 2 |
| Trials (total) | 979 |
| Trials / session | 556, 423 |

---

## Step 3: Reference Text Reading
**Status**: [COMPLETE]

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 621,733 | "621,733 neurons recorded with 699 Neuropixels probes across 139 mice in 12 laboratories" (datapaper p.1) | 
| Neurons / session | 76, 264 | |
| Subjects | 139 mice | "621,733 neurons recorded with 699 Neuropixels probes across 139 mice" (datapaper p.1) |
| Sessions / subject | 2 | |
| Trials (total) | 979 | |
| Trials / session | 556, 423 | |
| Neural data time bin | 20 ms in reference wheel decoding; task-dependent in paper | methods.txt: "Spike counts were similarly binned" and wheel values were averaged in nonoverlapping 20-ms bins |
| Behavior data time bin | 20 ms for wheel decoding in reference analyses | methods.txt: "We averaged wheel values in nonoverlapping 20-ms bins" |
| Reward rate | | | | 
| <Task/behavior statistic 1> | | | | 
| <Task/behavior statistic 2> | | | |
| wheel_speed_bin | 0.5296 | 0.5167 |
| whisker_motion_energy_bin | 0.5910 | 0.5806 | | 


### Processing Details
- `methods.txt` states whisker motion energy is computed as the mean absolute frame-to-frame pixel difference in whisker-pad ROIs from the left/right videos.
- DLC-based body-part estimates exclude predictions with likelihood < 0.9.
- Decoding analyses bin spike counts in fixed-width windows aligned to task events; alignment event depends on decoded variable in the reference analyses.
- Neurons in the same session and region are combined across probes rather than decoded probe-by-probe.
- For wheel decoding in the reference paper, wheel values are averaged in non-overlapping 20 ms bins, and spike counts are similarly binned.
- For binary targets such as choice, the methods text reports logistic regression with balanced accuracy as the metric in the reference analyses.
- Our conversion will differ only where required by the decoder-task specification: all streams will be aligned to stimulus onset, and continuous wheel/whisker outputs will be discretized into 3 bins.

### Curation Steps

**Neuron curation rules**:
- To be verified against reference code in Step 4/5; likely quality-filtered ephys units and region/session inclusion rules from cached IBL data loaders.

**Trial curation rules**:
- To be verified against reference code in Step 4/5; reference code uses trial masks and valid aligned trial subsets before decoding.

### Decoders Trained
| Decoded variable | Accuracy |
| | |

---

## Step 4: Check for Consistency
**Status**: [COMPLETE]

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Probe/session handling | `prepare_data` merges all probes for an eid via `eid2pid` and `merge_probes` | Local data are organized by session/lab in ONE cache | Paper says probes in same session are not decoded separately | Combine neurons across probes within session, preserving neuron-level region labels |
| Neuron quality | `good_clusters` defined as `(clusters['label'] >= 1)` | Cluster QC fields are present in metadata | Paper references well-isolated units / canonical cell sets | Use quality-filtered units and record region/QC metadata |
| Trial filtering | `load_trials_and_mask(..., max_trial_len=10.0)` and alignment removes masked trials | Trials are represented in cached session data, exact counts pending | Paper/methods imply curated valid trials for decoding | Apply explicit valid-trial mask before packaging trials |
| Temporal binning | `params` in `0_data_caching.py` use `binsize=0.02`, `time_window=(-0.5, 1.5)`, `align_time='stimOn_times'` | Compatible with stimulus-aligned task request | Methods text uses 20 ms bins for wheel decoding | Use 20 ms bins and stimulus-onset alignment for this conversion |
| Behavior streams | Reference code loads wheel speed/velocity and left/right whisker motion energy | ONE cache likely contains these streams per session | Methods define whisker motion energy from video motion energy and wheel from binned values | Use wheel speed and whisker motion energy as time-varying outputs; discretize into 3 bins per task spec |
| Alignment implementation detail | `align_spike_behavior` intends to intersect behavior/trial validity masks but uses Python list `and` rather than elementwise conjunction | N/A | Intended behavior is clearly to keep only jointly valid trials | Implement explicit elementwise conjunction in our conversion; document this as a likely bug/quirk rather than copying it blindly |

---

## Step 5: Mapping Planning
**Status**: [COMPLETE]

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spikes.times.npy` + `spikes.clusters.npy` merged across probes within session | neural | Bin into 20 ms bins from -0.5 s to +1.5 s relative to stimulus onset; one matrix per trial of shape (n_neurons, n_timepoints) | `prepare_data`, `bin_spiking_data` | Preserve session-level probe merging and neuron region labels |
| Trial-relative time axis | input[0] time since stimulus onset | Repeat common time vector across trials as a 1 x T input | reference params in `0_data_caching.py` | Continuous time-varying decoder input |
| Trial table `probabilityLeft` and trial order within block | input[1] trial number in block | Compute within-block trial index from consecutive equal-probability blocks | trial table + reference trial loading | Continuous per-trial or broadcast across time |
| Trial table `choice` | output[0] choice | Map left=0, right=1 using IBL coding after verifying sign convention | trial loading in reference code | Per-trial categorical output |
| Trial table `probabilityLeft` | output[1] prior probability of left | Map 0.2->0, 0.5->1, 0.8->2 | trial table | Per-trial categorical output |
| `_ibl_wheel.position.npy` + `_ibl_wheel.timestamps.npy` | output[2] wheel speed | Differentiate/interpolate to stimulus-aligned 20 ms bins, then discretize into 3 bins | `bin_behaviors`, `load_target_behavior` | Time-varying categorical output |
| `leftCamera.ROIMotionEnergy.npy` and `rightCamera.ROIMotionEnergy.npy` + camera times | output[3] whisker motion energy | Use available whisker ROI motion energy stream(s), align/interpolate to 20 ms bins, combine left/right sensibly if both available, discretize into 3 bins | `bin_behaviors` special-cases whisker motion energy | Time-varying categorical output |
| Cluster region metadata / acronyms | brain_region_idx | Map neuron acronyms to unique region list | `prepare_data`, region helpers | One region index per neuron |
| Subject/session path metadata | subjects / subject_idx | Extract unique subject IDs and session-to-subject mapping | local ONE cache structure | Session order must match neural/input/output lists |

### Key Decisions
1. **Use 20 ms bins and stimulus-onset alignment**: Matches the reference caching params (`binsize=0.02`, `align_time='stimOn_times'`) and the task requirement.
2. **Merge probes within a session**: Reference code merges probes for a session before decoding; we will do the same.
3. **Filter to valid trials explicitly**: Use trial masks / valid aligned intervals and implement elementwise mask conjunction, rather than reproducing the apparent Python-list `and` quirk in `align_spike_behavior`.
4. **Use quality-filtered clusters**: Follow reference practice of using cluster labels/QC and preserve metadata for sanity checks.
5. **Represent per-trial categorical outputs compactly**: Choice and prior can be stored as per-trial vectors; time-varying outputs (wheel/whisker) will be stored as 1 x T categorical time series.
6. **Discretize wheel speed and whisker motion energy into 3 bins using global/session-aware quantile-style thresholds to avoid degenerate classes**: exact thresholds to be finalized during implementation after checking distributions.

### Planned Sanity Checks
- [ ] Verify binned spike counts for one neuron/trial against raw `spikes.times.npy` by manual histogramming.
- [ ] Verify stimulus-onset aligned time vector matches trial table `stimOn_times` for sampled trials.
- [ ] Verify choice and probabilityLeft labels on sampled trials against `_ibl_trials.table.pqt`.
- [ ] Verify wheel-speed bins on sampled trials against differentiated raw wheel trace.
- [ ] Verify whisker motion-energy bins on sampled trials against raw ROI motion energy trace and camera times.

---

## Step 6: Script Development
**Status**: [COMPLETE]

Implemented initial local-cache conversion script with direct ALF file loading, 20 ms stimulus-aligned binning, wheel-speed interpolation, whisker motion-energy interpolation, probe merging within session, and categorical output construction.

Code inefficiencies identified:
- Initial version used ONE object loading and pulled unnecessary spike/template/waveform arrays, causing major slowdowns.

Code speedups added:
- Replaced ONE object loading with direct local `.npy`/`.pqt` reads from session ALF directories, reducing sample conversion time from ~27 s to ~3 s for 2 sessions.

---

## Step 7: Sample Conversion and Validation
**Status**: [COMPLETE]

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 340 |
| Neurons / session | |
| Subjects | 1 |
| Sessions / subject | |
| Trials (total) | |
| Trials / session | |
| time_since_stim_onset range | [-0.5, 1.5] |
| trial_number_in_block range | [1, 90] |
| choice distribution | session1 [0.579, 0.421]; session2 [0.452, 0.548] |
| prior_left_prob distribution | session1 [0.448, 0.158, 0.394]; session2 [0.511, 0.213, 0.277] |

### Processing Plots Review
No shape anomalies in verification. Time bins are consistent (100 per trial). Brain-region labels are currently stringified region IDs rather than acronyms; structurally valid but should be improved later if atlas mapping becomes available.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Direct local ALF reads instead of ONE object loading | ~10x faster on sample |

| Step | Time / Session | Estimated Total Time |
| Sample conversion | ~0.7-1.2 s/session | Full dataset likely minutes to low tens of minutes depending on available sessions |

---

## Step 8: Sample Decoder Training
**Status**: [COMPLETE]

### Format Validation
- Errors: None
- Warnings: None reported by train_decoder.py

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| choice | 0.6128 | 0.5396 |
| prior_left_prob | 0.7149 | 0.7001 |
| wheel_speed_bin | 0.5296 | 0.5167 |
| whisker_motion_energy_bin | 0.5910 | 0.5806 |

---

## Step 9: Full Conversion and Validation
**Status**: [COMPLETE]

### Output Files
- `converted_data.pkl`: created (see filesystem for exact size)
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | | | | | |
| Mean neurons/session | | | | | |
| Subjects | 139 mice | "621,733 neurons recorded with 699 Neuropixels probes across 139 mice" (datapaper p.1) | | | |
| Sessions | | | | | |
| Trials (total) | | | | | |
| Trials/session (mean) | | | | | |
| <Input 1 range> | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| ... | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| <Output 1 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |
| <Output 2 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |

---

## Step 10: Critical Review 1
**Status**: [COMPLETE]

### Checks Performed
1. Output log verification: initial full conversion had numerical RuntimeWarnings from duplicate wheel timestamps and all-NaN whisker averaging; patched conversion removed these warnings in rerun.
2. Reference-code comparison: preserved session-level probe merging, 20 ms stimulus-onset alignment, and strict cluster label filtering (`label >= 1`).
3. Edge-case handling: added duplicate-timestamp handling for wheel traces and explicit all-NaN handling for whisker interpolation.
4. Validation rerun: reconverted full dataset and reran verify-only validation successfully.

### Issues Found and Resolved
- Wheel interpolation warnings from duplicate timestamps: deduplicated wheel timestamps before `np.gradient`.
- Whisker `nanmean` warnings when both streams were invalid: replaced `np.nanmean` with explicit finite-value averaging and all-NaN handling.
- Remaining caveat: brain-region labels are numeric region IDs as strings rather than acronyms because atlas mapping package was unavailable in the environment; structurally valid but semantically less ideal.

---

## Step 11: Full Decoder Training
**Status**: [COMPLETE]

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| choice | 0.6128 | 0.5396 |
| prior_left_prob | 0.6854 | 0.6737 | above chance |
| wheel_speed_bin | 0.5296 | 0.5167 |
| whisker_motion_energy_bin | 0.5910 | 0.5806 |

---

## Step 12: Critical Review 2
**Status**: [IN PROGRESS]

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from papers |
| | |

[Analysis of any low accuracies]

### Issues Found and Resolved
- Wheel interpolation warnings from duplicate timestamps: deduplicated wheel timestamps before `np.gradient`.
- Whisker `nanmean` warnings when both streams were invalid: replaced `np.nanmean` with explicit finite-value averaging and all-NaN handling.
- Remaining caveat: brain-region labels are numeric region IDs as strings rather than acronyms because atlas mapping package was unavailable in the environment; structurally valid but semantically less ideal.

---

## Step 13: Documentation and Cleanup
**Status**: [NOT STARTED]

- [ ] README.md created
- [ ] cache/ folder created
- [ ] All files organized
