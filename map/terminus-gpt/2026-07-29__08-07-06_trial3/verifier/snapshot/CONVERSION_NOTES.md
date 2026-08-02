# Dataset Conversion Notes

## Overview
- **Dataset**: [Name and source]
- **Date started**: 2026-07-29
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- .manifest
- CONVERSION_NOTES.md
- ChenLiuEtAl2023_SpikeSortingQC.pdf
- Dockerfile
- code
- data
- datapaper.pdf
- decoder.py
- docker-compose.yaml
- methodpaper.pdf
- methods.txt
- train_decoder.py
---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| load_data | code/VideoAnalysisUtils/population_decoding_utils.py | LOADING | Loads a pickled ephys session dict and returns a reduced dict with neural activity, trial variables, bin centers, regions/coordinates, and stimulation fields. |
| get_regular_trial_mask | code/VideoAnalysisUtils/population_decoding_utils.py | CURATION | Defines regular trials by excluding early lick, auto water, free water, no-response, and stimulation trials. |
| get_ventral_medial_mask | code/VideoAnalysisUtils/population_decoding_utils.py | PROCESSING | Splits medulla neurons by CCF coordinate into ventral-medial vs dorsolateral groups. |
| nested_cross_validation | code/VideoAnalysisUtils/population_decoding_utils.py | PROCESSING | Runs decoding analyses on trial-by-time neural features. |
| preprocess_all_ephys (script) | code/Sherlock/preprocess_all_ephys.py | PROCESSING | Batch preprocessing script for all ephys sessions; likely creates the pickled session files consumed downstream. |
| medulla_population_decoding (script) | code/Sherlock/medulla_population_decoding.py | PROCESSING | Example downstream analysis using session_dict['X'], trial labels, bin_centers, and region masks. |

### Notes
- Reference analyses operate primarily on preprocessed per-session pickle files rather than directly on NWB in downstream scripts.
- Session dict fields observed in reference code include at least: `X`, `bin_centers`, `early_lick_trials`, `auto_water_trials`, `free_water_trials`, `lick_directions`, `stimulation`, and likely anatomical metadata such as `ccf_coordinate`.
- `get_regular_trial_mask` excludes: early-lick trials, auto-water trials, free-water trials, no-response trials, and photostimulation trials (`stimulation[:,0] == 0` required).
- Downstream decoding code indexes `session_dict['bin_centers']` as the common time axis and uses trial-aligned neural arrays over time.
- The codebase suggests spike data were already binned into firing rates before downstream analyses; this will be important when matching our 50 ms binning/alignment pipeline.
- Need to match session/trial variable semantics from these preprocessed dicts when exploring raw data in Step 2.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Data are organized under `data/` by subject folders named `sub-<subject_id>`.
- Each session is a single NWB file named like `sub-<subject_id>_ses-<timestamp>_behavior+ecephys+ogen.nwb`.
- NWB top-level groups observed: `acquisition`, `analysis`, `general`, `intervals`, `processing`, `stimulus`, `units`.
- Trial metadata are stored under `intervals/trials`.
- Behavioral streams are stored under `acquisition/BehavioralEvents` and `acquisition/BehavioralTimeSeries`.
- Spike-sorted unit metadata are stored in the `units` table, which includes quality-control metrics such as `amplitude_cutoff`, `isi_violation`, `presence_ratio`, `isolation_distance`, `l_ratio`, `d_prime`, `nn_hit_rate`, `silhouette_score`, and `classification`.
- Subject metadata are under `general/subject` and include `subject_id`, `description`, `sex`, `species`, and `date_of_birth`.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 272227 |
| Neurons / session | min 493, mean 1564.52, max 3191 |
| Subjects | 28 |
| Sessions / subject | mean 6.21 (174 total sessions / 28 subjects) |
| Trials (total) | 94990 |
| Trials / session | min 264, mean 545.92, max 800 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 69943 good units | "Overall, the dataset consisted of 69,943 good units recorded across 173 behavioral sessions" |
| Subjects | 28 mice total dataset; 17 VGAT-ChR2-EYFP mice for photoinhibition subset | "Photoinhibition ... on a subset of ~25% randomly interleaved trials ... (N = 17 VGAT-ChR2-EYFP mice)" |
| Sessions | 173 behavioral sessions | "69,943 good units recorded across 173 behavioral sessions" |
| Video sampling rate | 300 Hz | "High-speed videos ... were acquired at 300 Hz" |
| Photostimulation trial fraction | ~25% of trials | "deployed on a subset of ~25% randomly interleaved trials" |
| Photostim timing vs go cue | Ends before Go cue | "photoinhibition always ended before the 'Go' cue" |
| Good-unit curation | Region-specific QC classifiers on 15 metrics | "we used 15 cluster quality metrics to train classifiers ... Applying the trained classifiers ... provided lists of units that were labeled as 'good'" |
| Region good-unit examples | ALM 8717; striatum 7664; thalamus 12808; midbrain 7495; medulla 2928 | region counts quoted in methods |

### Processing Details
- Behavioral task includes presample, sample, delay, go, and trial end epochs, with event times stored in NWB behavioral events.
- Photoinhibition occurs during the late delay epoch and ends before the Go cue, so go-cue alignment should preserve post-photostim behavior without overlap at the cue.
- Tongue, jaw, and nose were tracked from high-speed video at 300 Hz using DeepLabCut.
- Reference downstream code uses pre-binned trial-aligned neural data with a common `bin_centers` time axis.

### Curation Steps

**Neuron curation rules**:
- Use only units labeled as good by the region-specific QC classifier / classification output, matching the papers' curated dataset rather than all NWB units.
- QC is based on 15 spike-sorting metrics and region-specific logistic-regression classifiers described in the white paper/methods.

**Trial curation rules**:
- Reference decoding code defines regular trials by excluding early lick, auto water, free water, no-response trials, and stimulation trials.
- For this decoder task we must still include photostimulation as an input variable, so we may need a broader trial set than the paper's regular-trial analyses depending on downstream validation behavior.

### Decoders Trained
| Decoded variable | Accuracy |
| choice / movement-related variables | reported in method paper; exact values to extract later if needed |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Session count | Downstream code uses curated preprocessed session pickles | 174 NWB session files | 173 behavioral sessions | Treat raw NWB as source universe; expect one session to be excluded by curation or validity checks in the paper pipeline. We will identify/exclude any session failing required alignment/behavior criteria during conversion if needed. |
| Unit count | Reference analyses use curated session dicts | 272,227 total NWB units | 69,943 good units | Use `units/classification == 'good'` to match paper curation rather than all units. |
| Trial filtering | `get_regular_trial_mask` excludes early lick, auto/free water, no-response, stimulation | Raw NWB contains all trial types and explicit trial annotations | Papers/decoding analyses often focus on regular trials | Keep raw trial annotations available; decide in mapping step whether to include all trials or curated subsets depending on decoder task requirements. |
| Photostimulation representation | Reference code stores `stimulation` in session dict | Raw NWB stores `photostim_onset` / `photostim_duration` per trial and photostim event times | Methods: ~25% trials, late delay, ends before Go cue | Build time-varying photostim input from raw trial/event fields aligned to go cue. |
| Choice/outcome labels | Reference code uses `lick_directions` and no-response filtering | Raw NWB uses `trial_instruction`, lick event times, and `outcome` values `hit/ignore/miss` | Methods describe memory-guided lick task | Use raw NWB trial annotations and event times to reconstruct decoder outputs consistent with reference semantics. |

### Final Understanding
- Raw NWB files contain the full uncurated unit set plus trial/event/video metadata.
- Reference analyses operate on curated per-session pickles derived from these NWB files.
- To match the papers/code, neuron inclusion should be restricted to units with `classification == 'good'`.
- Trial/event alignment should use behavioral event timestamps (not trial start/stop alone), with Go cue as the alignment event for this task.
- Photostimulation should be represented from raw timing fields as a time-varying decoder input; note that stimulation ends before the Go cue in this paradigm.
- The 174-vs-173 session discrepancy will be checked during conversion; likely one session is excluded in the published curated dataset.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `units/spike_times` + `units/spike_times_index` for units with `classification == 'good'` | neural | Bin spike times into 50 ms bins from -2.5 s to +1.5 s relative to each trial's `go_start_times`; convert to firing rates or keep counts consistently across all sessions | Reference session dict uses trial-aligned neural array `X` with `bin_centers` | Must match go-cue alignment required by task; use only good units to match paper curation |
| `acquisition/BehavioralEvents/go_start_times` | metadata alignment / trial anchor | Use per-trial go cue onset as t=0 | `bin_centers` in reference session dict | All 174 sessions have `go_start_times` |
| Time relative to go cue and tone onset derived from event times | input[0] (`time_from_tone_onset`) | Build time-varying continuous vector per trial: bin center time minus tone/sample onset time, optionally masked/defined over whole window | event-time handling in reference pipeline | Need exact tone event mapping from task structure; likely `sample_start_times` corresponds to auditory cue epoch in this task |
| `photostim_onset` / `photostim_duration` or `photostim_start_times` / `photostim_stop_times` | input[1] (`photostimulation_on`) | Build binary time-varying vector over bins for whether photostimulation is on | reference session dict field `stimulation` | Methods say photostim ends before Go cue; binary vector should therefore mostly occupy pre-go bins on stim trials |
| `trial_instruction` and/or lick event times | output choice | Map left=0, right=1 per trial | reference `lick_directions` | Need to determine whether choice should reflect instructed side or actual lick direction; prefer actual choice semantics from reference code if available |
| `outcome` | output outcome | Map ignore=0, miss=1, hit=2 per trial | regular-trial code excludes no-response trials using outcome-like semantics | Direct categorical mapping from NWB strings |
| `early_lick` | output early lick | Map `no early`=0, `early`=1 per trial | reference `early_lick_trials` | Direct categorical mapping from NWB strings |
| `Camera0_side_TongueTracking/data` + timestamps | output tongue y-position | Interpolate/assign tongue y to decoder bins, then discretize per session using 40th/60th percentiles into 3 categories | video processing used in papers; our discretization is task-specific | Need to identify which column is y-position; likely second column if data are [x, y, likelihood] |
| subject folder / `general/subject/subject_id` | subjects / subject_idx | Unique subject list and per-session index | N/A | Use subject IDs from NWB |
| unit anatomical metadata such as `anno_name` or `electrode_group` | brain_regions / brain_region_idx | Map each good unit to a brain-region string and integer index | reference analyses use region masks and medulla subregions | Need to confirm best raw field for region naming |

### Key Decisions
1. **Neuron curation**: Include only units with `classification == 'good'` to match the curated dataset described in the papers.
2. **Temporal alignment**: Align all trials to `go_start_times` and extract [-2.5, +1.5] s windows with 50 ms bins.
3. **Photostim input**: Represent photostimulation as a binary time-varying input built from raw timing fields, not by excluding stim trials.
4. **Outcome mapping**: Use direct NWB categorical values `ignore`, `miss`, `hit` mapped to 0/1/2.
5. **Early lick mapping**: Use direct NWB categorical values `no early` and `early` mapped to 0/1.
6. **Session discrepancy handling**: Start from all 174 NWB sessions, then exclude only sessions that fail required data integrity checks (e.g. missing variables or too few valid trials).
7. **Tongue discretization**: Compute session-wide percentiles on valid tongue y-values, then discretize each binned time point into 3 categories.

### Planned Sanity Checks
- [ ] Recompute binned spikes for one unit/trial directly from raw `spike_times` and verify equality with saved converted neural matrix using `np.allclose()`.
- [ ] Recompute photostim binary vector for one trial from raw onset/duration and verify equality with converted input using `np.allclose()`.
- [ ] Recompute one trial's outcome/early-lick labels from `intervals/trials` and verify equality with converted outputs using `np.allclose()`.
- [ ] Recompute binned tongue y category for one trial from raw tongue timestamps/data and verify equality with converted output using `np.allclose()`.

---

## Step 6: Script Development
**Status**: COMPLETE

- Implemented `convert_data.py` with CLI `python -u convert_data.py <outpicklefile>` and `--full`, `--sample`, `--show-processing` options.
- Implemented NWB loading, good-unit filtering, go-cue alignment, 50 ms spike binning, photostim input construction, categorical outputs, subject/session metadata, and processing plots.
- Script compiles successfully with `python3 -m py_compile convert_data.py`.

Code inefficiencies identified:
- Current implementation loops over good units within each trial and may be slow for full conversion.

Code speedups added:
- Used direct NumPy histogram binning and avoided repeated file opens; further optimization may be needed after sample timing is measured.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 834 |
| Neurons / session | 459, 375 |
| Subjects | 1 |
| Sessions / subject | 2 |
| Trials (total) | 848 |
| Trials / session | 368, 480 |
| time_from_tone_onset range | [-0.6, 5.7] |
| photostimulation_on range | [0, 0] for sampled sessions |
| choice distribution | approx [0.448, 0.552] and [0.504, 0.496] |
| outcome distribution | approx [ignore,miss,hit]=[0.196,0.367,0.438] and [0.008,0.073,0.919] |

### Processing Plots Review
- Verification passed with no reported errors or warnings after rerun.
- All sessions have 80 time bins as expected.
- Sampled sessions had photostimulation input all zeros, likely because these particular sessions/trials lacked stimulation; this should be checked on broader data.
- Full-conversion runtime likely too slow with current nested unit-by-trial histogram loops.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Optimized implementation | ~1.5 s/session on sample |

| Step | Time / Session | Estimated Total Time |
| Conversion | ~1.5 s/session | ~4.5 min for 174 sessions |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None after rerun with corrected sample-time handling and guarded trial-validity logic

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| choice | 0.6582 | 0.6153 |
| outcome | 0.7162 | 0.6137 |
| early_lick | 0.8109 | 0.6833 |
| tongue_y_position | 0.4814 | 0.4531 |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: [size]
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Total neurons | | | | | |
| Mean neurons/session | | | | | |
| Subjects | | | | | |
| Sessions | | | | | |
| Trials (total) | | | | | |
| Trials/session (mean) | | | | | |
| <Input 1 range> | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| ... | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| <Output 1 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |
| <Output 2 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |

---

## Step 10: Critical Review 1
**Status**: IN PROGRESS

### Checks Performed
1. [Check]: [Result]

### Issues Found and Resolved
- [Issue]: [Resolution]

---

## Step 11: Full Decoder Training
**Status**: IN PROGRESS

### Training Progress
- Loss decreasing: [Yes/No]

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| <Output 1> | | |
| <output 2> | | |
| ... | | |

---

## Step 12: Critical Review 2
**Status**: NOT STARTED

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Papers |
| | |

[Analysis of any low accuracies]

### Issues Found and Resolved
- [Issue]: [Resolution]

---

## Step 13: Documentation and Cleanup
**Status**: NOT STARTED

- [ ] README.md created
- [ ] cache/ folder created
- [ ] All files organized
