# Dataset Conversion Notes

## Overview
- **Dataset**: MAP (Mesoscale Activity Project) - Brain-wide neural activity underlying memory-guided movement
- **Date started**: 2024
- **Goal**: Convert NWB electrophysiology data to decoder-compatible format
- **Papers**: Li et al. (data paper), Wang et al. (method paper)
- **Data source**: DANDI archive 000363

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `/app/data/` - 28 subject directories with 174 NWB files
- `/app/code/` - MapVideoAnalysis repository code
- `/app/datapaper.pdf` - Data paper
- `/app/methodpaper.pdf` - Method paper  
- `/app/ChenLiuEtAl2023_SpikeSortingQC.pdf` - Spike sorting QC white paper
- `/app/methods.txt` - Extracted methods text
- `/app/train_decoder.py` - Decoder training script
- `/app/decoder.py` - Decoder model

Python environment verified: numpy 2.4.4, torch 2.6.0+cu124

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| sliding_histogram | preprocessing_DJ_2022Aug.py | PROCESSING | Convert spike times to firing rates with sliding bins |
| process_one_sess | preprocessing_DJ_2022Aug.py | LOADING/PROCESSING | Full session processing pipeline |
| process_one_area | preprocessing_DJ_2022Aug.py | PROCESSING | Process neurons for one brain area |
| get_regular_trial_mask | functions_for_r2.py | CURATION | Filter trials: no early lick, no auto/free water, no ignore, no photostim |
| create_4fold_trial_type_mask | functions_for_r2.py | PROCESSING | Stratify trials by type (hit/miss x left/right) |
| temporal_alignment_embed_and_ephys | functions_for_r2.py | PROCESSING | Align embedding vectors with ephys data |
| align_markers_between_lims | Sherlock/align_markers.py | PROCESSING | Align marker tracking data to go cue |
| loadmat | preprocessing_utils.py | LOADING | Load .mat files with proper dict recovery |
| check_fr | preprocessing_utils.py | CURATION | Remove zero-variance neurons |
| get_period | preprocessing_utils.py | PROCESSING | Get time windows: all(-3,3.5), sample(-1.9,-1.2), delay(-1.2,0), post_go(0,1) |

### Notes
- Original code processes .mat files exported from DataJoint; our data is in NWB format
- Default parameters: stride=0.05s, bin_width=0.1s, begin_time=-3.0, end_time=3.5 (relative to go cue)
- QC mode: "classifier" - uses region-specific logistic regression classifiers
- In NWB files, the `classification` column in units table has "good" or "unlabelled" values
- Spike times in NWB are absolute (session time), need to subtract go cue time per trial
- The code filters for "regular trials": no early lick, no auto water, no free water, correctness != -1, no stimulation
- However, for our decoder task, we want to KEEP all trials (including photostim, early lick, ignore) since these are decoder outputs/inputs
- We should still filter auto_water and free_water trials as they are not genuine behavioral trials

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- NWB files with behavior+ecephys+ogen (optogenetics)
- 6 sessions lack "+ogen" in filename (no photostim in those sessions)
- Each NWB file contains:
  - **Units table**: spike_times, classification (good/unlabelled), anno_name (brain region), quality metrics, is_good_trials, obs_intervals, waveform_mean
  - **Trials table**: start_time, stop_time, trial_instruction (left/right), early_lick, outcome (hit/miss/ignore), auto_water, free_water, photostim_onset/power/duration, task, task_protocol
  - **BehavioralEvents**: go_start_times, sample_start_times, delay_start_times, presample_start_times, left_lick_times, right_lick_times, photostim_start/stop_times, trialend_start/stop_times
  - **BehavioralTimeSeries**: Camera0_side_TongueTracking (x,y,likelihood), JawTracking, NoseTracking

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total, all) | 272,227 |
| Neurons (total, good) | 69,453 |
| Neurons / session (good, mean) | 399.2 |
| Neurons / session (good, median) | 390 |
| Neurons / session (good, range) | 0-923 |
| Subjects | 28 |
| Sessions (total NWB files) | 174 |
| Sessions (with good units) | 173 |
| Sessions / subject | 3-10 |
| Trials (total) | 94,990 |
| Trials / session (mean) | 545.9 |
| Trials / session (range) | 264-800 |

Note: sub-440958_ses-20190216T162508 has 0 good units.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total, good) | 69,943 | "69,943 good units recorded across 173 behavioral sessions" |
| Neurons / session | ~404 | 69943/173 |
| Subjects | 28 | From data |
| Sessions | 173 | "173 behavioral sessions" |
| Probe insertions | 655 | "655 probe insertions" |
| Good unit fraction | 25.9% | "25.9% of clusters reported by Kilosort2" |
| Neural data time bin (stride) | 50ms | Default stride=0.05 in code |
| Neural data bin width | 100ms | Default bw=0.1 in code |
| Time window | -3.0 to 3.5s | Default in code |
| Photostim fraction | ~25% | "~25% randomly interleaved trials" |
| Photostim mice | 17 | "N = 17 VGAT-ChR2-EYFP mice" |
| Behavior performance | 83.2% | "reduced behavior performance from 83.2% to 71.7%" |
| ALM good units | 8,717 | From methods |
| Striatum good units | 7,664 | From methods |
| Thalamus good units | 12,808 | From methods |
| Midbrain good units | 7,495 | From methods |
| Medulla good units | 2,928 | From methods |
| QC false alarm rate | 4.3-7.8% | Region-specific |

### Processing Details
- Temporal alignment: All times relative to Go cue onset
- Trial structure: presample -> sample (tone, -1.85s before go) -> delay (-1.2s before go) -> go cue (0) -> response
- Photostimulation: During last 0.5s of delay (onset at -1.2s relative to go), 100ms ramp-down
- Spike sorting: Kilosort2 with region-specific QC classifiers
- 5 QC classifiers: cortex, striatum, thalamus, midbrain, medulla

### Curation Steps

**Neuron curation rules**:
- Use `classification == "good"` from NWB units table (QC classifier output)
- This is equivalent to the region-specific logistic regression classifiers described in the paper

**Trial curation rules (for R2 analysis in paper)**:
- No early lick (early_lick_trials == 0)
- No auto water (auto_water_trials == 0)
- No free water (free_water_trials == 0)
- No ignore/no-response (correctness != -1)
- No photostimulation (stimulation[:,0] == 0)

**Trial curation for our decoder (different from paper)**:
- Filter: auto_water == 1 and free_water == 1 (not genuine behavioral trials)
- KEEP: early lick trials (early_lick is a decoder output)
- KEEP: ignore/no-response trials (outcome is a decoder output)  
- KEEP: photostimulation trials (photostim is a decoder input)

### Decoders Trained (in papers)
| Decoded variable | Method | Metric |
|---|---|---|
| Choice (left/right) | Single-neuron logistic regression | AUC ~0.65 threshold for significance |
| Motor type | Single-neuron AUC | AUC |
| Neural activity from video | Ridge regression | R² |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| N good units | N/A | 69,453 | 69,943 | ~0.7% difference - likely minor version differences in QC classifier. The NWB files have pre-computed classification. Acceptable. |
| N sessions | N/A | 174 NWB files | 173 behavioral sessions | 1 session (sub-440958_ses-20190216T162508) has 0 good units, so 173 usable sessions. Match! |
| Good unit % | N/A | 69453/272227 = 25.5% | 25.9% | Close match |
| Bin width vs stride | bw=0.1, stride=0.05 | N/A | Task says 50ms bins | Task specifies 50ms bins. Use stride=0.05, bw=0.05 (NOT 0.1 as in paper). See Step 5. |
| Trial filtering | get_regular_trial_mask excludes early lick, ignore, photostim | N/A | Paper analyzes regular trials only | For decoder, we keep all trials except auto/free water. Different from paper but required by decoder task. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| units.spike_times (good only) | neural | Bin into 50ms firing rates, align to go cue, extract [-2.5, 1.5]s | 80 time bins |
| tone onset relative to each timepoint | input[0] | Time from tone onset (continuous, s) | Compute per-timepoint |
| photostim on/off | input[1] | Binary: 1 if photostim active at timepoint, 0 otherwise | Time-varying |
| First lick direction after go cue | output[0] | Categorical: 0=left, 1=right, 2=no_lick | Per-trial |
| trials.outcome | output[1] | Categorical: 0=hit, 1=miss, 2=ignore | Per-trial |
| trials.early_lick | output[2] | Categorical: 0=no_early, 1=early | Per-trial |
| tongue y-position | output[3] | Discretized: 0=<p40, 1=p40-p60, 2=>p60, 3=not_visible | Time-varying, per-session percentiles |
| units.anno_name | brain_region_idx | Map to brain_regions list | Per-neuron |

### Key Decisions
1. **Bin width = 50ms**: Task specifies 50ms bins. The reference code uses 100ms bin width with 50ms stride, but the task says "50-ms-width bins". Use 50ms width with 50ms stride (non-overlapping).
2. **Trial filtering**: Keep all trials except auto_water=1 and free_water=1. This differs from the paper\'s regular trial mask but is required by the decoder task.
3. **Lick choice**: Determine from first lick after go cue (left_lick_times vs right_lick_times). No lick = ignore outcome.
4. **Tongue y-position**: Use likelihood > 0.9 threshold for visibility. Discretize visible positions using session-wide 40th/60th percentiles. Bin tongue data into 50ms bins aligned to go cue.
5. **Tone onset**: The sample/tone onset is at a fixed time relative to go cue (typically -1.85s). Represent as continuous time from tone onset.
6. **Photostim input**: Binary time series, 1 during photostim period (onset to onset+duration).
7. **Brain regions**: Use anno_name from units table, group into major regions.

### Planned Sanity Checks
- [ ] Total good neurons ~69,453 across all sessions
- [ ] 173 sessions with good units
- [ ] 28 subjects
- [ ] Outcome distribution matches expectations (~83% hit rate on non-stim trials)
- [ ] Photostim on ~25% of trials in VGAT mice
- [ ] Go cue at time 0 in aligned data
- [ ] Spike rates are reasonable (0-100 Hz typical)
- [ ] Tongue y-position discretization produces expected distribution

---

## Step 6: Script Development
**Status**: COMPLETE

Script `/app/convert_data.py` written with:
- NWB file loading with pynwb
- Unit filtering by classification == 'good'
- Trial filtering: remove auto_water and free_water trials
- Recording coverage filtering: only include trials where go_cue window falls within recording
- Vectorized firing rate computation using np.searchsorted and np.histogram
- Vectorized tongue y-position computation
- Tone onset detection from sample_start_times events
- Photostimulation input from trial metadata
- Processing plots for --show-processing mode

Code inefficiencies identified:
- Initial tongue computation was O(n_trials * n_bins * n_frames) = very slow
- Initial firing rate used mask operations instead of searchsorted

Code speedups added:
- Used np.searchsorted for tongue frame lookup (52-117s -> 0.2-0.3s per session)
- Used np.searchsorted for spike time windowing
- Total speedup: 182s -> 8.2s for 2 sessions (22x)

---

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons (total) | 985 |
| Neurons / session | 459, 526 |
| Subjects | 2 |
| Trials (total) | 874 |
| Trials / session | 354, 520 |
| time_from_tone_onset range | [-0.6, 5.7] |
| photostim_on range | [0.0, 1.0] |
| lick_choice distribution | left 47.1%, right 44.9%, no_lick 8.0% |
| outcome distribution | hit 66.6%, miss 25.4%, ignore 8.0% |
| early_lick distribution | no_early 97.3%, early 2.7% |
| tongue_y distribution | not_visible 75.3%, below_p40 10.9%, p40-p60 8.1%, above_p60 5.7% |

### Processing Plots Review
- Firing rates show expected patterns around go cue
- Tone onset input correctly shows time from tone
- Photostim input correctly marks stimulation period
- Tongue y-position shows expected pattern (mostly not visible, visible around response)

### Run Time Estimates
| Step | Time / Session | Estimated Total Time |
|------|----------------|---------------------|
| Load data | 0.3-0.7s | ~80s |
| Firing rates | 1.8-3.1s | ~400s |
| Tongue y | 0.2-0.3s | ~40s |
| Total | 4.1s avg | ~12 min |

---

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|----------------------|------------------------|--------|
| lick_choice | 0.778 | 0.696 | 0.333 |
| outcome | 0.770 | 0.692 | 0.333 |
| early_lick | 0.905 | 0.787 | 0.500 |
| tongue_y_position | 0.731 | 0.673 | 0.250 |

All outputs well above chance. Loss decreased from 8.19 to 0.52.

---

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 11,337 MB
- `verification_full_out.txt`: created

### Key Statistics
- 173 sessions (1 skipped: 0 good units)
- 28 subjects
- 69,453 total good neurons
- 89,532 total trials
- 293 brain regions
- Processing time: 589.8s (3.4s/session average)

### Consistency Check
| Statistic | Reference Papers | Converted Data | Match? |
|-----------|------------------|----------------|--------|
| Total good neurons | 69,943 | 69,453 | ~99.3% (minor QC version diff) |
| Sessions | 173 | 173 | Yes |
| Subjects | 28 | 28 | Yes |
| Mean neurons/session | ~404 | 401.5 | Yes |
| Good unit fraction | 25.9% | 25.5% | Close |
| Hit rate | ~83% | 68.5% | Expected lower (includes stim/early lick trials) |

Note: Hit rate is lower than paper's 83.2% because we include photostim trials (which reduce performance) and early lick trials. The paper's 83.2% is for regular non-stim trials only.

### Warnings
- 2 trials with all-zero neural data (edge cases at recording boundaries)

---

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed

**Check 1: Output log verification**
- No errors in verification output
- 2 warnings about zero neural data (edge cases at recording boundaries, acceptable)

**Check 2: Sanity checks**
- Neural: Computed firing rate for session 0, trial 10, neuron 5, bin 50 (go cue) = 40.0 Hz. Matches direct NWB computation. np.allclose = True.
- Input: Time from tone onset for session 0, trial 0, bin 0 = -0.625. Matches direct computation. np.allclose = True.
- Output: Session 0, trial 0: choice=2 (no_lick), outcome=2 (ignore), early_lick=0 (no early). All match NWB data.

**Check 3: Reference code comparison**
- Data loading: We load NWB files directly; reference code loads .mat files. Same underlying data.
- Neuron filtering: We use classification=="good" from NWB; reference uses QC classifier output. Same result.
- Trial filtering: We filter auto_water and free_water (matching reference). We additionally filter trials outside recording window.
- We do NOT filter early_lick, ignore, or photostim trials (differs from reference get_regular_trial_mask), because these are decoder outputs/inputs.
- Temporal alignment: We align to go cue onset, same as reference.
- Binning: We use 50ms non-overlapping bins (task spec). Reference uses 100ms width, 50ms stride. Difference justified by task specification.
- Input construction: time_from_tone_onset and photostim_on. These are decoder-specific.
- Output construction: lick_choice, outcome, early_lick, tongue_y_position. These follow task specification.

**Check 4: Key statistics comparison**
| Statistic | Reference | Converted | Match |
|-----------|-----------|-----------|-------|
| Sessions | 173 | 173 | Yes |
| Subjects | 28 | 28 | Yes |
| Total good neurons | 69,943 | 69,453 | ~99.3% |
| Mean neurons/session | ~404 | 401.5 | Yes |
| Hit rate (regular trials) | 83.2% | 81.6% | Close (minor filtering diff) |

**Check 5: Edge cases**
- Recording coverage: Fixed issue where trials outside recording window had zero neural data
- Tone onset: Fallback to go_time - 1.85 if no sample event found
- Photostim timing: Correctly computed from trial_start + photostim_onset string
- Sessions with 0 good units: Correctly skipped (1 session)

### Issues Found and Resolved
- Zero neural data in 1061 trials: Fixed by filtering trials outside recording window
- After fix: Only 2 edge-case trials with zero data remain

---

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (19.74 -> 0.65)
- Test loss: 0.657

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Ratio |
|--------|----------------------|------------------------|--------|-------|
| lick_choice | 0.711 | 0.681 | 0.333 | 2.04x |
| outcome | 0.702 | 0.661 | 0.333 | 1.98x |
| early_lick | 0.788 | 0.751 | 0.500 | 1.50x |
| tongue_y_position | 0.702 | 0.669 | 0.250 | 2.68x |

All outputs well above chance. Train/val gap is small (no overfitting).

---

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

**Check 1: Accuracy vs chance**
| Variable | Val Accuracy | Chance | Ratio | Status |
|----------|-------------|--------|-------|--------|
| lick_choice | 0.681 | 0.333 | 2.04x | Good |
| outcome | 0.661 | 0.333 | 1.98x | Good |
| early_lick | 0.751 | 0.500 | 1.50x | Good |
| tongue_y_position | 0.669 | 0.250 | 2.68x | Good |

All outputs > 1.5x chance.

**Check 2: Accuracy comparison to papers**
- The reference papers use single-neuron logistic regression with AUC metric for choice decoding
- AUC > 0.65 is considered significant for individual neurons
- Our decoder uses population-level neural activity, so higher accuracy is expected
- Choice decoding at 68.1% balanced accuracy is reasonable given the full brain-wide population

**Check 3: Train vs validation gap**
| Variable | Train Acc | Val Acc | Gap |
|----------|-----------|---------|-----|
| lick_choice | 0.711 | 0.681 | 0.030 |
| outcome | 0.702 | 0.661 | 0.041 |
| early_lick | 0.788 | 0.751 | 0.037 |
| tongue_y_position | 0.702 | 0.669 | 0.033 |

No gap exceeds 1.5x ratio. No overfitting concerns.

### Issues Found and Resolved
- No issues found. All accuracies are reasonable and consistent.

---

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created with processing plots
- [x] All files organized
- [x] CONVERSION_NOTES.md complete
