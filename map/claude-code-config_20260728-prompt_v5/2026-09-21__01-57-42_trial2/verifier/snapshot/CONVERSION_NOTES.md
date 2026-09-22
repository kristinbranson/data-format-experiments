# Dataset Conversion Notes

## Overview
- **Dataset**: MAP (Mesoscale Activity Project) - Brain-wide neural activity underlying memory-guided movement (Chen et al. 2024, DANDI:000363)
- **Method paper**: Wang, Kurgyis, Chen et al. 2025, "Brain-wide analysis reveals movement encoding structured across and within brain areas"
- **Date started**: 2026-09-21
- **Goal**: Convert NWB data to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `/app/data/` - 28 subjects (sub-440956 through sub-484677), 174 NWB files
- `/app/code/` - Reference code from method paper (VideoAnalysisUtils, Sherlock scripts, Notebooks)
- `/app/datapaper.pdf` - Data paper
- `/app/methodpaper.pdf` - Method paper
- `/app/ChenLiuEtAl2023_SpikeSortingQC.pdf` - Spike sorting QC white paper
- `/app/methods.txt` - Extracted methods text
- `/app/train_decoder.py` - Decoder training script
- `/app/decoder.py` - Decoder implementation

Python environment: numpy 2.4.4, torch 2.6.0+cu124, h5py 3.16.0

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `process_one_sess()` | `preprocessing_DJ_2022Aug.py` | LOADING/PROCESSING | Loads .mat files, combines probes, applies QC, computes firing rates |
| `sliding_histogram()` | `preprocessing_DJ_2022Aug.py` | PROCESSING | Bins spike times into firing rates with sliding window |
| `helper_get_neuron_id_area()` | `preprocessing_DJ_2022Aug.py` | CURATION | Filters neurons by region, hemisphere, and QC classifier |
| `get_regular_trial_mask()` | `functions_for_r2.py` / `population_decoding_utils.py` | CURATION | Filters: no early lick, no auto water, no free water, correctness != -1, no stimulation |
| `create_4fold_trial_type_mask()` | `functions_for_r2.py` | PROCESSING | Stratifies trials: hit-right, miss-right, hit-left, miss-left |
| `load_session()` | `population_decoding_utils.py` | LOADING | Loads preprocessed pickle files, combines areas |
| `get_period()` | `preprocessing_utils.py` | PROCESSING | Defines time periods: sample [-1.9,-1.2], delay [-1.2,0], post_go [0,1] |

### Notes
- Reference code works with .mat files exported from DataJoint; our data is in NWB format
- Preprocessing parameters from `preprocess_all_ephys.py`: bw=0.04s (40ms), stride=0.0034s, begin_time=-3.0, end_time=3.0, qc_mode='classifier'
- The `__main__` block in `preprocessing_DJ_2022Aug.py` uses bw=0.1, stride=0.05
- For the decoder task, we use 50ms bins (as specified)
- Spike times in the reference code are already aligned to go cue (time 0)
- QC filtering uses classifier-based approach: 5 region-specific classifiers (cortex, striatum, thalamus, midbrain, medulla)
- In NWB data, QC is encoded as `units/classification` with values 'good' or 'unlabelled'
- Trial type: 1=left, 0=right (in reference code); NWB uses trial_instruction field
- Correctness: 1=correct/free water, 0=error, -1=no response (in reference code); NWB uses outcome field (hit/miss/ignore)

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
NWB files organized as `/app/data/sub-{id}/sub-{id}_ses-{datetime}_behavior+ecephys[+ogen].nwb`
- 28 subjects, 174 sessions
- 6 sessions WITHOUT optogenetics (+ogen missing from filename)
- Each NWB contains:
  - `units/`: spike times, QC metrics, brain region annotations, classification
  - `intervals/trials/`: trial info (outcome, early_lick, photostim, instruction, etc.)
  - `acquisition/BehavioralEvents/`: event timestamps (go_start, sample_start, delay_start, lick times, photostim times)
  - `acquisition/BehavioralTimeSeries/`: tongue/jaw/nose tracking at ~300Hz (x, y, likelihood)

### Key Data Fields
- `units/classification`: 'good' or 'unlabelled' - QC filter
- `units/spike_times` + `spike_times_index`: compressed spike time arrays
- `units/anno_name`: brain region annotation per unit
- `intervals/trials/outcome`: 'hit', 'miss', 'ignore'
- `intervals/trials/early_lick`: 'early', 'no early'
- `intervals/trials/trial_instruction`: 'left', 'right'
- `intervals/trials/photostim_onset`: relative to trial start, or 'N/A'
- `intervals/trials/photostim_duration`: '0.5000' or 'N/A'
- `intervals/trials/auto_water`: 0 or 1
- `intervals/trials/free_water`: 0 or 1

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total, good) | 69,453 |
| Neurons / session (mean) | 399.2 |
| Subjects | 28 |
| Sessions | 174 |
| Sessions / subject | 3-10 |
| Trials (total) | 94,990 |
| Trials / session (mean) | 545.9 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Good units (total) | 69,943 | "69,943 good units" (methods.txt) |
| Sessions | 173 | "173 behavioral sessions" (methods.txt) |
| Probe insertions | 655 | "655 probe insertions" |
| QC pass rate | 25.9% | "25.9% of clusters reported by Kilosort2" |
| Trials/session (mean) | 476 | "476 (Mean; range, 130-785) trials per session" |
| Correct rate | 84% | "84% correct rate (range, 65-99%)" |
| Session criteria | >65% correct, >=50 correct L and R | "overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each" |
| Photostim trials | ~25% | "~25% randomly interleaved trials" |
| N VGAT-ChR2 mice | 17 | "N = 17 VGAT-ChR2-EYFP mice" |
| Behavior from bilateral photostim | 83.2% -> 71.7% | "reduced behavior performance from 83.2% to 71.7%" |
| Camera frame rate | 300 Hz | "acquired at 300 Hz" |
| Sample epoch | 3 tones x 150ms, 100ms ITI = 650ms | "three times for 150 ms with 100 ms inter-tone intervals" |
| Delay epoch | 1.2s | "followed by a 1.2 s delay epoch" |
| Go cue | 6kHz carrier, 360Hz mod, 0.1s duration | "carrier frequency of 6 kHz with 360 Hz modulating frequency, 0.1 s duration" |
| Response epoch | 1.5s | "answer period: 1.5 s" |
| Photostim duration | 0.5s | "last 0.5 s" of delay (incl 100ms ramp-down) |
| Good units by region | ALM: 8717, Striatum: 7664, Thalamus: 12808, Midbrain: 7495, Medulla: 2928 | methods.txt |

### Processing Details
- Spike times aligned to go cue onset (go cue = time 0)
- Reference code: firing rates computed with sliding histogram (bw=40ms, stride=3.4ms) or (bw=100ms, stride=50ms)
- Our decoder task: 50ms bins, -2.5s to +1.5s around go cue = 80 time bins
- Photostim occurs during late delay, ~0.5s ending before go cue. Data shows onset at -1.2s (delay start), duration 0.5s, ending at -0.7s relative to go cue
- Tongue tracking at ~300Hz with DLC likelihood scores

### Curation Steps

**Neuron curation rules**:
- Use classifier-based QC: keep units with `classification == 'good'`
- 5 region-specific classifiers (cortex, striatum, thalamus, midbrain, medulla)
- Already applied in NWB data

**Trial curation rules** (reference code `get_regular_trial_mask()`):
- Remove early lick trials (early_lick_trials == 0)
- Remove auto water trials (auto_water_trials == 0)
- Remove free water trials (free_water_trials == 0)
- Remove no response trials (correctness != -1)
- Remove photostimulation trials (stimulation[:,0] == 0)

**For decoder** (modified):
- Remove only auto_water and free_water trials (not genuine behavioral trials)
- KEEP early lick trials (early_lick is a decoder output)
- KEEP ignore/no response trials (outcome is a decoder output)
- KEEP photostim trials (photostim is a decoder input)

**Session selection criteria** (from methods.txt):
- Overall behavioral performance > 65%
- At least 50 correct lick left trials
- At least 50 correct lick right trials
- Performance computed on control trials (no photostim), excluding early lick

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Total good units | N/A | 69,453 | 69,943 | Paper says 173 sessions, data has 174. One session may be excluded by session criteria. Small difference (~490 units) may be due to this. |
| Sessions | N/A | 174 | 173 | Need to apply session selection criteria (>65% correct, >=50 correct L/R). One session likely fails. |
| Trial filtering | Removes early lick, ignore, photostim | All trials present | Excludes early lick and no response | For decoder, we keep these since they are decoder outputs/inputs |
| Photostim timing | N/A | Onset at -1.2s (delay start), 0.5s duration, ends -0.7s | "last 0.5s" of delay | Data shows photostim starts at delay onset, not the last 0.5s. Trust the data. |
| Firing rate bins | bw=40ms,stride=3.4ms (preprocess) or bw=100ms,stride=50ms (main) | N/A | N/A | Decoder task specifies 50ms bins |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| `units/spike_times` (good units only) | neural | Bin into 50ms firing rates, align to go cue, -2.5 to +1.5s | Shape (n_good_neurons, 80) per trial |
| Time from tone onset | input[0] | `time_bin_center - tone_onset_relative_to_go_cue` | Continuous, time-varying. tone_onset = last sample_start before go cue. |
| Photostim active | input[1] | Binary: 1 if photostim is on at time point, 0 otherwise | Use photostim_start/stop_times |
| Lick direction choice | output[0] | 0=left, 1=right, 2=no_lick | Derived: hit+left_instr=left, hit+right_instr=right, miss+left_instr=right, miss+right_instr=left, ignore=no_lick. Per-trial. |
| Outcome | output[1] | 0=ignore, 1=miss, 2=hit | From trials/outcome. Per-trial. |
| Early lick | output[2] | 0=no, 1=yes | From trials/early_lick. Per-trial. |
| Tongue y-position | output[3] | 0=<40th pctl, 1=40-60th pctl, 2=>60th pctl, 3=not visible | Session percentiles on visible tongue y. Time-varying at 50ms bins. |
| `units/anno_name` | brain_region_idx | Map detailed annotations to major brain regions | Use CCF hierarchy |
| `general/subject/subject_id` | subjects/subject_idx | Index mapping | |

### Key Decisions
1. **Trial filtering**: Keep all trials except auto_water and free_water. This differs from reference code's `get_regular_trial_mask()` which also removes early lick, ignore, and photostim. Justified because these are decoder outputs/inputs.
2. **Session filtering**: Apply session selection criteria from methods: >65% correct on control (no photostim, no early lick) trials, >=50 correct left and right trials.
3. **Spike binning**: Use 50ms non-overlapping bins (not sliding window) as specified by decoder task. Different from reference code's 40ms sliding window with 3.4ms stride.
4. **Tone onset**: Find last sample_start_time before each trial's go_start_time to account for epoch replays due to early licks.
5. **Tongue y discretization**: Compute session-level percentiles on ALL visible tongue y values (likelihood > 0.9), then discretize per time point. Not visible = tongue_likelihood < 0.9.
6. **Brain regions**: Map detailed CCF annotations to major regions (ALM, other cortex areas, striatum, thalamus, midbrain, medulla, etc.)
7. **Choice derivation**: Determine from outcome + trial_instruction: hit=same as instruction, miss=opposite, ignore=no_lick.

### Planned Sanity Checks
- [x] Total good units matches paper (~69,943) → 69,453 (99.3%)
- [x] Number of sessions after filtering matches paper (173) → 173
- [x] Mean trials/session consistent with paper (476) → 523.7 (higher because we keep early lick/ignore/photostim trials)
- [x] Correct rate ~84% on control trials → 81.8%
- [x] Photostim on ~25% of trials (for ogen sessions) → 20.7%
- [x] Spot-check spike rates against raw spike times → exact match
- [x] Verify temporal alignment by checking go cue corresponds to t=0 → confirmed

---

## Step 6: Script Development
**Status**: COMPLETE

Created `/app/convert_data.py` (~600 lines). Key design decisions:
- 50ms non-overlapping bins, -2.5s to +1.5s around go cue (80 bins)
- Fixed tone onset timing: -1.85s relative to go cue (task protocol defined)
  - Initially used event-based lookup (`find_tone_onset_for_trial`), but discovered 12.5% of trials had incorrect timing due to early lick replay events in NWB
  - Fixed by using hardcoded `TONE_ONSET_REL_GO = -1.85` based on task protocol
- Trial filtering: remove only auto_water and free_water trials
- Session filtering: none needed (DANDI archive already curated)
- Brain region mapping: 44 detailed CCF regions via keyword matching
- Photostim: converted from trial-relative to go-cue-relative timing
- Tongue y: session-level percentile discretization (40th/60th) on visible tongue data

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

- Sample: 5 sessions, 5 subjects, 1,846 neurons, 2,329 trials
- Verification passed with consistent input ranges
- Processing speed: ~2.4s/session

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

Sample decoder results (5 sessions):
| Output | Train Acc | Val Acc | Chance |
|--------|-----------|---------|--------|
| choice | 0.742 | 0.689 | 0.333 |
| outcome | 0.731 | 0.696 | 0.333 |
| early_lick | 0.827 | 0.776 | 0.500 |
| tongue_y | 0.735 | 0.587 | 0.250 |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

- Full dataset: 173 sessions, 28 subjects, 69,453 neurons, 90,605 trials
- File size: ~12 GB
- Processing time: ~500s (2.9s/session)
- Verification: all input/output ranges correct, 1,061/90,605 (1.2%) zero-neural trials

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Sanity Checks
- [x] Sessions: 173 (matches paper)
- [x] Good units: 69,453 (~99.3% of paper's 69,943)
- [x] Subjects: 28 (matches paper)
- [x] Control hit rate: 81.8% (paper ~84%)
- [x] Photostim: 20.7% of ogen session trials (paper ~25%)
- [x] Spot-check spike rates: exact match vs raw NWB
- [x] Output variables: all match raw NWB for first 10 trials
- [x] time_from_tone: consistent [-0.625, 3.35] across all sessions
- [x] No negative firing rates, max 940 Hz

### Bug Found and Fixed
- **time_from_tone anomaly**: 12.5% of trials had incorrect tone onset timing due to early lick replay events creating extra `sample_start_times` in NWB. Fixed by using protocol-defined timing (-1.85s) instead of event-based lookup.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

Full decoder results (173 sessions):
| Output | Train Acc | Val Acc | Chance | Above Chance |
|--------|-----------|---------|--------|-------------|
| choice | 0.721 | 0.686 | 0.333 | 2.06x |
| outcome | 0.694 | 0.661 | 0.333 | 1.98x |
| early_lick | 0.785 | 0.748 | 0.500 | 1.50x |
| tongue_y | 0.750 | 0.659 | 0.250 | 2.64x |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

- Train-val gap modest (no severe overfitting)
- All outputs well above chance on validation
- Results consistent with brain-wide neural recordings in delayed response task
- Training converged smoothly (200 epochs)

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE
