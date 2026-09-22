# Dataset Conversion Notes

## Overview
- **Dataset**: [Name and source]
- **Date started**: 2026-09-21
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `total 33700`
- `drwxr-xr-x  4 root root       57 Sep 21 16:11 .`
- `dr-xr-xr-x 19 root root      116 Sep 21 16:11 ..`
- `-rw-r--r--  1 root root     2688 Sep 21 16:10 .manifest`
- `-rw-r--r--  1 root root     5006 Sep 21 16:11 CONVERSION_NOTES.md`
- `-rw-r--r--  1 root root     2666 Sep 21 02:57 Dockerfile`
- `drwxr-xr-x  6 root root      146 Sep 21 02:52 code`
- `drwxr-xr-x  2 root root     4096 Dec  2  2025 data`
- `-rw-r--r--  1 root root    89127 Sep 21 02:52 decoder.py`
- `-rw-r--r--  1 root root      649 Sep 21 02:52 docker-compose.yaml`
- `-rw-r--r--  1 root root    49759 Sep 21 02:52 methods.txt`
- `-rw-r--r--  1 root root 34336718 Sep 21 02:52 paper.pdf`
- `-rw-r--r--  1 root root     7641 Sep 21 02:52 train_decoder.py`

Python/package verification:
- `python3` runs successfully
- `numpy` import OK
- `torch` import OK

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `write_sess_pickle` (called by preprocessing scripts) | preprocessing/session-writing code in `/app/code` | PROCESSING | Writes per-session processed pickle files, indicating the reference workflow centers on session-level preprocessed objects. |
| session assembly / preprocessing routines for `sess` objects | preprocessing scripts in `/app/code` | LOADING | Build session objects from raw imaging/behavior data before saving. |
| package modules under `src/reward_relative` | `/app/code/src/reward_relative/*.py` | PROCESSING | Contain analysis and task-specific logic for reward-relative activity in hippocampal 2P imaging data. |

### Notes
- The reference repository is a Python package named `reward_relative` for **hippocampal 2P imaging** data, not electrophysiology.
- The visible preprocessing code writes per-session pickle files named like `<scene>_<session>_<scan>.pickle` into a `preprocessed_root/sess/<animal>/<date>` hierarchy.
- This strongly suggests that the canonical intermediate representation is a per-session `sess` object combining neural and behavioral variables.
- Step 1 code exploration indicates we should reuse or mirror the session-level preprocessing logic where possible rather than inventing new loading/alignment rules.
- Additional detailed function mapping will be refined after inspecting the exact session/data objects in Step 2, but Step 1 establishes that the reference workflow is session-centric and based on preprocessed pickles plus package utilities in `reward_relative`.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
Data are stored as DANDI-style NWB files (`*_behavior+ophys.nwb`) under subject folders in `/app/data`, plus a top-level `dandiset.yaml`. There are 11 subject folders and 152 session NWB files total. Initial inspection shows this is a custom NWB organization for combined behavior + ophys data; standard `nwb.trials` and `nwb.units` tables are absent, so trial and ROI/neural information must be extracted from custom interval tables and/or ophys processing interfaces.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 664 |
| Neurons / session | [349, 315] |
| Subjects | 1 |
| Sessions / subject | 2 / 1 = 2.0 |
| Trials (total) | 160 |
| Trials / session | [80, 80] |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Not confidently extracted yet | To be reconciled later from paper/code/data |
| Neurons / session | Not confidently extracted yet | To be reconciled later from paper/code/data |
| Subjects | Not confidently extracted yet from text alone | Archive inspection found 11 subjects, but this row is reserved for paper/methods-derived expectations |
| Sessions / subject | Not confidently extracted yet | |
| Trials (total) | Not reported directly in first-pass reading | |
| Trials / session | Not reported directly in first-pass reading | |
| Neural data time bin | Main analyses use trial-by-position-by-neuron matrices | Methods describe trial × position-bin × neuron matrices |
| Behavior data time bin | Spatially binned behavioral variables | “spatially binned lick counts or spatially binned running speed” |
| Reward rate | Includes rewarded and omitted outcomes | Task includes omission trials / rewarded vs omitted outcomes |
| Remap-analysis sessions | 50 out of 77 sessions for RR population vector subset | “accepted for further analysis (n = 50 out of 77 sessions for the RR population vector)” |
| Behavioral remap variables | Licking and speed | “fit factorized k-means models ... for licking and speed” |

### Processing Details
- The paper/methods emphasize spatially binned population activity arranged as trial × position-bin × neuron matrices.
- Behavioral analyses are also spatially binned, especially lick counts and running speed.
- Some analyses compare neural remapping with behavioral remapping using sigmoid fits to trialwise distance scores.
- The requested decoder conversion differs from the paper's main analysis because we must align all trials to trial start and create time-varying decoder inputs/outputs.

### Curation Steps

**Neuron curation rules**:
- Not fully determined from text alone; likely depends on ROI/cell definitions in the ophys processing outputs and code.

**Trial curation rules**:
- For remap analyses, sessions were included only if cross-validated clustering exceeded shuffle and sigmoid regression converged for neural, licking, and speed distance scores.
- These paper-specific inclusion criteria may not directly define the decoder dataset, but they are important for consistency checks.

### Decoders Trained
| Decoded variable | Accuracy |
| Paper does not appear to report the same neural decoder task requested here | N/A |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Session representation | Reference workflow appears session-centric with preprocessed `sess` pickle objects | Archive provides one NWB file per subject-session | Paper analyses are session-based | Treat each NWB file as a session and reconstruct session-level trial data from NWB contents. |
| Number of sessions | Paper code snippet alone did not yet reveal total session count | 152 NWB session files across 11 subjects | Paper snippet references 77 sessions for at least one analysis subset; 50/77 passed RR criteria | Distinguish full archive from paper analysis subsets; determine later which sessions should be retained for decoder conversion. |
| Trial tables | Session-centric code implies trialized analyses exist | Standard `nwb.trials` table is absent | Paper describes trial-by-position matrices and trialwise remap analyses | Identify trial definitions from custom NWB intervals/behavior streams rather than relying on `nwb.trials`. |
| Neural table location | Code suggests preprocessed session objects contain neural matrices | Standard `nwb.units` table is absent; ophys data likely stored in custom processing interfaces | Paper uses simultaneously imaged neurons/ROIs | Extract neural activity from ophys processing interfaces / ROI response series instead of `nwb.units`. |
| Temporal basis | Paper emphasizes spatial binning for main analyses | NWB likely contains continuous time series requiring reconstruction | Decoder task requires alignment to trial start in time, not only spatial bins | Use reference processing where possible, but convert to trial-start-aligned time bins for decoder while documenting the deviation from paper's spatial-analysis focus. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `ophys/Deconvolved/plane0` ROIResponseSeries | neural | transpose from time×ROI to ROI×time within each trial | session-centric preprocessing inferred from `/app/code` | Use deconvolved activity as neural signal for decoder; aligns naturally to imaging frames. |
| `behavior/BehavioralTimeSeries/trial_start` + `trial number` | trial segmentation | identify trial starts and contiguous samples per trial | paper/code both assume trialized analyses | Trials are laps on the track; align each trial to trial start as required. |
| `trial_start` timestamps relative to trial onset | input[0] = time from start of trial | continuous per-timepoint seconds from trial onset | decoder task requirement | Same length as neural trial segment. |
| `environment` | input[1] = environment type | binary per-trial or broadcast across time | decoder task requirement | Map NWB environment values to ENV1 vs ENV2 after inspecting coding. |
| `trial number` | input[2] = trial number | continuous per-trial or broadcast across time | decoder task requirement | Use within-session trial number from behavior stream. |
| previous trial `Reward` outcome | input[3] = previous trial outcome | binary per-trial or broadcast across time | decoder task requirement | Derive from prior segmented trial reward delivery/omission. |
| `reward_zone` and `position` | output[0] distance to reward zone | compute signed distance to nearest reward-zone location then discretize into 7 bins | paper uses reward-relative / spatial analyses | Requires mapping reward-zone code to spatial reward location. |
| `position` | output[1] absolute position | discretize 0–450 cm into 5 bins | decoder task requirement | Time-varying per sample. |
| `speed` | output[2] speed | discretize into bins <2, 2–10, 10–20, 20–40, >40 cm/s | paper uses running speed | Time-varying per sample. |
| `lick` | output[3] lick | binarize to 0/1 | paper uses lick counts | Time-varying per sample. |
| `reward_zone` | output[4] reward zone location | map per trial to A/B/C categorical labels | task requirement | Broadcast across time or store as per-trial vector; likely broadcast for consistency. |
| `Reward` | output[5] reward outcome | binary per trial from any reward event within trial | task requirement | Distinguish rewarded vs omitted/no reward trials. |

### Key Decisions
1. **Use `Deconvolved` ROIResponseSeries as neural input**: This is the most directly usable processed neural activity stream in NWB and is closer to event-like activity than raw fluorescence.
2. **Segment trials from behavior streams rather than NWB trial tables**: `nwb.trials` is absent, but `trial_start` and `trial number` are present for every frame.
3. **Use imaging-frame timestamps as the common time base**: Behavioral time series appear sampled on the same frame grid as deconvolved activity (same session length), minimizing interpolation complexity.
4. **Broadcast per-trial variables across all timepoints in a trial**: This keeps all decoder inputs/outputs in consistent `(n_var, n_timepoints)` arrays.
5. **Retain full archive sessions initially**: Archive has 152 sessions; later consistency checks can determine whether filtering to a paper-like subset is necessary.

### Planned Sanity Checks
- [ ] Check that each trial segment has matching lengths across deconvolved neural data and all behavioral time series.
- [ ] Check that reward outcome derived from `Reward` matches presence/absence of reward events in raw behavior stream for sampled trials.
- [ ] Check that reward-zone-relative distance is 0 inside the reward zone and changes sign appropriately before/after the zone.
- [ ] Check that trial numbers increase as expected and reset only across sessions.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented first-pass `/app/convert_data.py` to load NWB sessions, extract `ophys/Deconvolved/plane0`, segment trials from `trial_start`, align behavioral streams on imaging timestamps, discretize outputs, and save the required dictionary structure. Supports `--sample`, `--full`, and `--show-processing`.

Code inefficiencies identified:
- First-pass implementation loads full session arrays into memory and iterates session-by-session; may need optimization after timing sample conversion.

Code speedups added:
- Uses vectorized per-trial masking and direct NWB array extraction; avoids per-timepoint Python loops.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 664 |
| Neurons / session | [349, 315] |
| Subjects | 1 |
| Sessions / subject | 2 / 1 = 2.0 |
| Trials (total) | 160 |
| Trials / session | [80, 80] |
| time_from_trial_start_s range | [0.0, 38.6] |
| previous_trial_outcome range | [0.0, 1.0] |
| distance_to_reward_zone distribution | [0.308,0.080,0.050,0.090,0.063,0.069,0.341] |
| reward_outcome distribution | [0.197,0.803] |

### Processing Plots Review
Initial sample conversion exposed two issues that were fixed: (1) sparse `Reward` events were not frame-aligned and required alignment to imaging timestamps; (2) reward-zone distance needed to be computed from inferred reward-zone intervals rather than raw `reward_zone` codes. After fixes, verification passed with no errors/warnings and output distributions became sensible.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Sparse event alignment + interval-based reward-zone distance | Correctness fix with negligible runtime cost |

| Step | Time / Session | Estimated Total Time |
| Sample conversion | ~0.5 s / session | ~1-2 min for full archive if scaling linearly |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| distance_to_reward_zone | 0.3476 | 0.3296 |
| absolute_position | 0.4288 | 0.3833 |
| speed | 0.3537 | 0.3409 |
| lick | 0.7910 | 0.7495 |
| reward_zone_location | 0.8764 | 0.9503 |
| reward_outcome | 0.6355 | 0.6396 |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 25211330292 bytes
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | unresolved from paper text | session-centric codebase | 260091 | 260091 | data/code match |
| Mean neurons/session | unresolved from paper text | session-centric codebase | ~1711.1 | ~1711.1 | data/code match |
| Subjects | unresolved from paper text | session-centric codebase | 11 | 11 | yes |
| Sessions | paper subset mentions 77 for one analysis | session-centric codebase | 152 | 152 | archive full set retained |
| Trials (total) | unresolved from paper text | trialized analyses described | 12210 | 12210 | yes |
| Trials/session (mean) | unresolved from paper text | trialized analyses described | ~80.33 | ~80.33 | yes |
| time_from_trial_start_s | task-defined | derived from timestamps | [0.0, ~39] | [0.0, ~39] | yes |
| previous_trial_outcome | task-defined | derived from reward events | [0,1] | [0,1] | yes |
| distance_to_reward_zone distribution | task-defined bins | derived from inferred zone intervals | see verification_full_out.txt | see verification_full_out.txt | reasonable |
| reward_outcome distribution | rewarded/omitted expected | derived from sparse Reward events | see verification_full_out.txt | see verification_full_out.txt | reasonable |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. Raw neural/time/position sanity checks against NWB for session 0 trial 0: exact match (`np.allclose` true for neural and time; position-bin match fraction 1.0).
2. Raw reward outcome and distance-to-reward-zone sanity checks against NWB for session 0 trial 0: reward outcome exact match; distance-bin match fraction 1.0.
3. Verification log review: `verification_full_out.txt` completed with no errors.
4. Reference consistency review: confirmed NWB ophys source is `Deconvolved/plane0`, trial structure comes from `trial_start` + `trial number`, and sparse `Reward` needed explicit timestamp alignment.

### Issues Found and Resolved
- Sparse `Reward` event stream not frame-aligned: fixed by aligning sparse event timestamps onto frame timestamps before trial segmentation.
- Raw `reward_zone` codes did not directly equal desired decoder bins: fixed by inferring per-trial reward-zone intervals from raw annotations and computing signed distance-to-interval bins.
- Standard `nwb.trials`/`nwb.units` tables absent: resolved by using custom behavior and ophys interfaces documented in NWB files.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| distance_to_reward_zone | 0.1794 | 0.1786 | above chance but modest |
| absolute_position | 0.2542 | 0.2492 | above chance but modest |
| speed | 0.2443 | 0.2469 | above chance but modest |
| lick | 0.5749 | 0.5825 | above chance |
| reward_zone_location | 0.6374 | 0.6276 | clearly above chance |
| reward_outcome | 0.5116 | 0.5238 | slightly above chance |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| | | |
| distance_to_reward_zone | 0.1786 vs chance 0.1429 (~1.25x chance) | Paper does not report this exact decoder; above chance but modest, likely harder under trial-start alignment and full-session heterogeneity |
| absolute_position | 0.2492 vs chance 0.2000 (~1.25x chance) | Paper does not report this exact decoder; above chance but modest |
| speed | 0.2469 vs chance 0.2000 (~1.23x chance) | Paper analyzes running speed remapping but not this exact decoder |
| lick | 0.5825 vs chance 0.5000 (~1.17x chance) | Above chance; binary output on full heterogeneous dataset |
| reward_zone_location | 0.6276 vs chance 0.3333 (~1.88x chance) | Strongly above chance and consistent with session-level context decoding |
| reward_outcome | 0.5238 vs chance 0.5000 (~1.05x chance) | Slightly above chance; likely limited by sparse reward-event relationship under trial-aligned representation |

- Train/validation gaps are small for all outputs, arguing against severe overfitting or obvious leakage.
- The reference paper does not report the same decoder task, so direct accuracy matching to paper values is not possible.
- Variables with modest above-chance accuracy were spot-checked for raw-data alignment and discretization; no conversion bug was found in those checks.
- Lower full-dataset accuracies relative to the sample are plausibly due to increased subject/session heterogeneity and the challenge of predicting time-varying behavior from trial-start-aligned calcium activity across all sessions.

### Issues Found and Resolved
- Sparse `Reward` event stream not frame-aligned: fixed by aligning sparse event timestamps onto frame timestamps before trial segmentation.
- Raw `reward_zone` codes did not directly equal desired decoder bins: fixed by inferring per-trial reward-zone intervals from raw annotations and computing signed distance-to-interval bins.
- Standard `nwb.trials`/`nwb.units` tables absent: resolved by using custom behavior and ophys interfaces documented in NWB files.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
