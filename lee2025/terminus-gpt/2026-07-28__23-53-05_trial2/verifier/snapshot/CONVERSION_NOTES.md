# Dataset Conversion Notes

## Overview
- **Dataset**: [Name and source]
- **Date started**: 2026-07-29
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- CONVERSION_NOTES.md
- code/
- data/
- methods.txt
- paper.pdf
- train_decoder.py

Python/package verification:
- python3: working
- numpy: import successful
- torch: import successful

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| load_dat | code/georepca1/src/utils.py | LOADING | Load per-animal data dictionary from joblib or MATLAB-derived files; converts selected MATLAB object fields to Python-friendly format. |
| mat2joblib | code/georepca1/src/utils.py | LOADING | Convert original MATLAB animal files into joblib format for downstream analyses. |
| generate_behav_dict | code/georepca1/src/utils.py | LOADING | Build behavior dictionary across animals for later analyses/decoding. |
| get_rate_maps | code/georepca1/src/utils.py | PROCESSING | Compute occupancy-normalized spatial activity/rate maps from position and calcium trace data with smoothing. |
| get_split_half | code/georepca1/src/utils.py | CURATION | Compute split-half reliability per cell from first vs second half session rate maps. |
| get_shuffle_split_half | code/georepca1/src/utils.py | CURATION | Generate shuffle null distribution for split-half reliability significance testing. |
| get_place_cells | code/georepca1/src/utils.py | CURATION | Mark place cells using shuffle-based split-half reliability p-values. |
| get_shr_within | code/georepca1/src/utils.py | CURATION | Apply place-cell reliability analysis across all days/sessions for one animal. |
| fit_decoder | code/georepca1/src/utils.py | PROCESSING | Train within-session decoder from traces to binned behavioral features. |
| test_decoder | code/georepca1/src/utils.py | PROCESSING | Evaluate decoder on held-out data with temporal binning/downsampling. |
| decode_position_within | code/georepca1/src/utils.py | PROCESSING | End-to-end within-session position decoding using traces, maps, and behavior. |

### Notes
- Reference code is a CA1 calcium imaging analysis repository (`georepca1`).
- Neural data are calcium traces (`trace`), not spikes/ephys, so no spike binning or dF/F recomputation appears to be the primary loading step in the reference utilities.
- Core spatial processing uses position plus trace to build 15x15 rate maps (`get_rate_maps`, default `fps=30`, Gaussian smoothing parameter `filter_size=1.5`).
- Cell curation is based on place-cell / split-half reliability significance using shuffles (`get_place_cells`, `get_shr_within`), suggesting quality/functional filtering is important in the reference analyses.
- The repository includes explicit within-session position decoding functions (`fit_decoder`, `test_decoder`, `decode_position_within`), which are directly relevant to the requested decoder task.
- `main.py` orchestrates figure analyses by loading animal data, behavior dictionaries, RSMs, and decoding outputs from utility functions in `src/utils.py`.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `data/` contains 7 animals with both original MATLAB v7.3 `.mat` files and extensionless compressed/joblib-converted files used by the reference code.
- Additional files: `behav_dict` (joblib behavior summary across animals) and `precomputed_results/`.
- Per-animal joblib file structure: top-level dict keyed by animal ID.
- Per-animal nested keys observed: `SFPs`, `blocked`, `centroids`, `envs`, `maps`, `position`, `trace`.
- `blocked` is a Python list (not a numpy array).
- `envs`: numpy array of session environment labels, shape `(n_sessions, 1)`.
- `position`: numpy array of x/y trajectories, shape `(n_sessions, 2, n_frames_max)`.
- `trace`: numpy array of calcium traces, shape `(n_sessions, n_neurons, n_frames_max)`.
- `SFPs`: spatial footprints, example shape for QLAK-CA1-08 `(35, 35, 515, 31)`.
- `centroids`: example shape for QLAK-CA1-08 `(515, 2, 31)`.
- `maps`: dict with keys `sampling`, `smoothed`, `unsmoothed`; example shapes for QLAK-CA1-08: `sampling` `(15, 15, 31)`, `smoothed` `(15, 15, 515, 31)`, `unsmoothed` `(15, 15, 515, 31)`.
- `behav_dict` top-level keys are animal IDs; nested keys include `position`, `envs`, `maps_shape`.
- Array sizes vary by animal. Observed session counts / neuron counts / frames per animal:
  - QLAK-CA1-08: 31 sessions, 515 neurons, 71866 frames
  - QLAK-CA1-30: 31 sessions, 875 neurons, 71866 frames
  - QLAK-CA1-50: 31 sessions, 942 neurons, 71866 frames
  - QLAK-CA1-51: 21 sessions, 554 neurons, 72219 frames
  - QLAK-CA1-56: 31 sessions, 862 neurons, 72091 frames
  - QLAK-CA1-74: 31 sessions, 713 neurons, 72060 frames
  - QLAK-CA1-75: 31 sessions, 952 neurons, 72071 frames

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 5413 |
| Neurons / session | not yet derived at session level; animal-level recorded cells vary 515-952 |
| Subjects | 7 |
| Sessions / subject | 31, 31, 31, 21, 31, 31, 31 |
| Trials (total) | not native; decoder trials will be derived by splitting long sessions into 1-minute trials |
| Trials / session | not native; to be constructed |

-----------|-------|
| Neurons (total) | 5413 |
| Neurons / session | not yet derived at session level; animal-level recorded cells: 515, 875, 942, 554, 862, 713, 952 |
| Subjects | 7 |
| Sessions / subject | 31, 31, 31, 21, 31, 31, 31 |
| Trials (total) | not native; decoder trials will be derived by splitting long sessions into 1-minute trials |
| Trials / session | not native; to be constructed |

-----------|-------|
| Neurons (total) | 5413 |
| Neurons / session | not yet derived at session level; animal-level recorded cells: 515, 875, 942, 554, 862, 713, 952 |
| Subjects | 7 |
| Sessions / subject | 31, 31, 31, 21, 31, 31, 31 |
| Trials (total) | not native; decoder trials will be derived by splitting long sessions into 1-minute trials |
| Trials / session | not native; to be constructed |

-----------|-------|
| Neurons (total) | 5413 |
| Neurons / session | not yet derived at session level; animal-level recorded cells: 515, 875, 942, 554, 862, 713, 952 |
| Subjects | 7 |
| Sessions / subject | 31, 31, 31, 21, 31, 31, 31 |
| Trials (total) | not native; decoder trials will be derived by splitting long sessions into 1-minute trials |
| Trials / session | not native; to be constructed |

-----------|-------|
| Neurons (total) | 5413 |
| Neurons / session | not yet derived at session level; animal-level recorded cells: 515, 875, 942, 554, 862, 713, 952 |
| Subjects | 7 |
| Sessions / subject | 31, 31, 31, 21, 31, 31, 31 |
| Trials (total) | not native; decoder trials will be derived by splitting long sessions into 1-minute trials |
| Trials / session | not native; to be constructed |

-----------|-------|
| Neurons (total) | 5413 |
| Neurons / session | not yet derived at session level; animal-level recorded cells: 515, 875, 942, 554, 862, 713, 952 |
| Subjects | 7 |
| Sessions / subject | 31, 31, 31, 21, 31, 31, 31 |
| Trials (total) | not native; decoder trials will be derived by splitting long sessions into 1-minute trials |
| Trials / session | not native; to be constructed |

-----------|-------|
| Neurons (total) | 5413 |
| Neurons / session | animal-level recorded cells: 515, 875, 942, 554, 862, 713, 952 |
| Subjects | 7 |
| Sessions / subject | 31, 31, 31, 21, 31, 31, 31 |
| Trials (total) | not native; decoder trials will be derived by splitting long sessions into 1-minute trials |
| Trials / session | not native; to be constructed |

-----------|-------|
| Neurons (total) | 5413 |
| Neurons / session | varies by animal; examples: 515, 875, 942, 554, 862, 713, 952 recorded cells per animal-level array |
| Subjects | 7 |
| Sessions / subject | 31, 31, 31, 21, 31, 31, 31 |
| Trials (total) | not native; decoder trials will need to be derived by splitting long sessions into 1-minute trials |
| Trials / session | not native; to be constructed |

-----------|-------|
| Neurons (total) | 18443 across converted sample sessions |
| Neurons / session | |
| Subjects | 2 |
| Sessions / subject | 31, 31 |
| Trials (total) | 2418 |
| Trials / session | 39 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 5413 | "5,413 unique neurons across 207 sessions in 10 geometries" |
| Subjects | 7 | inferred from data + methods context; methods describe mice across randomized sequences |
| Sessions (total) | 207 | "5,413 unique neurons across 207 sessions" |
| Geometries | 10 | "sequence of 10 geometrically distinct environments" |
| Rate maps (total) | 69744 | "forming 69,744 rate maps" |
| Arena size | 75 x 75 cm | "open square (75 × 75 cm)" |
| Spatial partition for task | 3 x 3 grid | "partitioned an open square (75 × 75 cm) into a 3 × 3 grid space" |
| Neural data time bin / sampling | 30 Hz acquisition | "DAQ simultaneously acquired behavioral and cellular imaging streams at 30 Hz" |
| Behavior data time bin / sampling | 30 Hz acquisition | same quote |
| Session duration | 40 min | "All sessions were 40 min" |
| Place-cell split halves | first and second 20 min | "each session half (first and second 20 min)" |
| Shuffle count for place cells | 1000 | "1000 circular shuffles" |
| Rising-phase threshold | z > 2.5 | "set to 1 whenever this z-scored vector exceeded 2.5" |

### Processing Details
- Calcium preprocessing in the paper: compute derivative of each filtered calcium trace, smooth with Gaussian kernel (SD = 5 frames), estimate noise from negative half of derivative distribution, z-score by this noise estimate, then binarize rising phases with threshold z > 2.5.
- The resulting binary rising-phase vector is treated as the firing rate in subsequent analyses.
- Behavioral and cellular imaging streams were timestamped and acquired simultaneously at 30 Hz for post-hoc alignment.
- Position was tracked with DeepLabCut.
- Sessions were 40 min, one per day.
- Animals experienced randomized sequences of 10 geometries that started and ended with square; same sequence could repeat up to three times within animal.
- Open square was partitioned conceptually into a 3x3 grid, directly relevant to the requested decoder output format.
- Within-session position decoding used 5-fold splits of spatially binned position and trace data, one-hot position labels, and Euclidean decoding error on held-out data.

### Curation Steps

**Neuron curation rules**:
- Manual inspection of motion-corrected calcium imaging data.
- Manual verification of spatial footprints to remove lens artifacts.
- Place-cell criterion for some analyses: split-half reliability above 99th percentile of 1000 shuffles.

**Trial curation rules**:
- No native trial structure; analyses use full 40-min sessions and session halves (20 min / 20 min) for place-cell reliability.

### Decoders Trained
| Decoded variable | Accuracy |
| Position within session | Not reported here; method reports Euclidean decoding error rather than accuracy |

-----------|-------|--------------|
| Neurons (total) | 18443 across converted sample sessions | | 
| Neurons / session | | |
| Subjects | 2 | |
| Sessions / subject | 31, 31 | |
| Trials (total) | 2418 | |
| Trials / session | 39 | |
| Neural data time bin | | |
| Behavior data time bin | | |
| Reward rate | | | | 
| <Task/behavior statistic 1> | | | | 
| <Task/behavior statistic 2> | | | |
| ... | | | | 


### Processing Details
- Calcium preprocessing in the paper: compute the derivative of each filtered calcium trace, smooth with a Gaussian kernel (SD = 5 frames), estimate noise from the negative half of the derivative distribution, z-score by this noise estimate, then binarize rising phases with threshold z > 2.5.
- The resulting binary rising-phase vector is treated as the firing rate in subsequent analyses.
- Motion-corrected calcium data were manually inspected; spatial footprints were manually verified to remove lens artifacts.
- Position was tracked with DeepLabCut.
- Cells were tracked across sessions using landmarks, spatial footprints, and/or centroids.
- Place cells were identified from split-half rate map correlations (first vs second 20 min) exceeding the 99th percentile of 1000 circular-shuffle correlations.
- Within-session position decoding used 5-fold splits of spatially binned position and trace data, one-hot position labels, and Euclidean decoding error on held-out data.

### Curation Steps

**Neuron curation rules**:
- Remove lens artifacts by manual verification of spatial footprints.
- Place cell criterion for some analyses: split-half reliability above 99th percentile of 1000 shuffles.

**Trial curation rules**:
- Session-halves are first and second 20 min for place-cell reliability analysis.

### Decoders Trained
| Decoded variable | Accuracy |
| | |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Total neurons / sessions | Reference utilities operate on per-animal arrays and maps | Data-derived totals are 5413 neurons across 207 sessions | Methods report 5413 neurons across 207 sessions | Match confirmed |
| Spatial maps | `get_rate_maps` and stored maps use 15x15 spatial bins | Stored `maps['sampling']` shapes are `(15,15,n_sessions)` | Paper reports 75x75 cm arena and 3x3 conceptual partition; code uses 15x15 rate maps for analyses | No discrepancy; decoder output will later coarsen position to 3x3 bins as required by task |
| Place-cell curation | Code uses split-half reliability and 1000 shuffles (`get_place_cells`, `get_shr_within`) | Data contain maps/position/trace needed for this computation | Methods describe first/second 20 min split-half and 1000 circular shuffles, 99th percentile threshold | Match confirmed |
| Position decoding | Code contains within-session decoding functions (`fit_decoder`, `test_decoder`, `decode_position_within`) | Data contain aligned position and trace arrays per session | Methods describe 5-fold decoding of spatially binned position from traces | Match confirmed at conceptual level |
| Neural trace representation | Code names variable `trace` and uses it directly for maps/decoding | Stored trace array dtype is float64; sampled values are exactly binary {0,1} with sparse events (~1.5% ones in sample) | Methods state analyses use binarized rising-phase vectors treated as firing rates | Resolved: stored traces are already binarized/preprocessed rising-phase vectors, consistent with methods |

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `rec['trace'][session]` | neural | Use directly as binary rising-phase activity; transpose to `(n_neurons, n_timepoints)` per trial after splitting session into 1-minute chunks | `get_rate_maps`, `decode_position_within` use `trace` directly | Methods confirm stored traces are already binarized rising-phase vectors |
| `rec['blocked'][session]` and/or `rec['envs'][session]` | input[0:9] | Convert session geometry name to 3x3 open/blocked indicator vector using reference `get_env_mat`, with `blocked` used as a cross-check | `get_environment_label`, `get_env_mat` | Decoder input should represent which arena partitions are blocked; static per trial |
| `rec['position'][session]` | output[time] | Discretize x/y position into 3x3 spatial bins over time | Paper conceptual 3x3 partition; decoding functions spatially bin position | Output should be time-varying categorical position |
| animal ID | subjects / subject_idx | Map unique animal names to subject indices | `load_dat` | 7 mice |
| CA1 recording site | brain_regions / brain_region_idx | Single region CA1 for all neurons | paper + dataset context | brain_region_idx all zeros |

### Key Decisions
1. **Trialization**: Split each native 40-minute session into 40 consecutive 1-minute trials at 30 Hz to satisfy decoder requirement of at least two trials per session.
2. **Neural representation**: Use stored binary `trace` arrays directly rather than recomputing calcium preprocessing, because methods + sampled values confirm they are already binarized rising-phase vectors.
3. **Decoder input**: Represent environment geometry as a 9-dimensional binary mask over the conceptual 3x3 partition, derived primarily from `envs` via the reference `get_env_mat` mapping and cross-checked against `blocked`.
4. **Decoder output**: Use time-varying 3x3 discretized position labels from x/y coordinates, matching the task requirement and the paper's conceptual partition.
5. **Temporal alignment**: Preserve native 30 Hz alignment between trace and position within each session when splitting into 1-minute trials.
6. **Neuron curation**: Initially include all stored neurons because artifact removal/preprocessing appears already reflected in the released data; place-cell filtering is analysis-specific and may not be appropriate for a general decoder unless required by reference decoding code.

### Planned Sanity Checks
- [ ] Verify `trace` and `position` have identical frame counts within every session.
- [ ] Verify 40-minute sessions at 30 Hz yield about 72,000 frames and therefore 40 one-minute trials of about 1800 frames each.
- [ ] Verify geometry vector from `envs` matches reference `get_env_mat` and is consistent with `blocked` for several named environments.
- [ ] Verify discretized 3x3 position labels match raw x/y coordinates on spot-checked frames.
- [ ] Verify total subjects/sessions match 7 / 207 after conversion.

-----------------|--------------|-----------|----------------------------|-------|
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
**Status**: COMPLETE

- Implemented `convert_data.py` with sample/full modes, 30 Hz alignment, 1-minute trialization, 3x3 geometry input from reference `get_env_mat`, 3x3 position output discretization, and per-session removal of non-finite neurons.

Code inefficiencies identified:
[Note]

Code speedups added:
[Note]

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 18443 across converted sample sessions |
| Neurons / session | |
| Subjects | 2 |
| Sessions / subject | 31, 31 |
| Trials (total) | 2418 |
| Trials / session | 39 |
| <Input statistic 1 range> | [MIN, MAX] |
| ... | [MIN, MAX] |
| <Output statistic 1 distribution> | [FRAC0,FRAC1,...] |
| ... | [FRAC0,FRAC1,...] |

### Processing Plots Review
- Verification passed with no errors or warnings.
- All converted trials have exactly 1800 timepoints.
- Current valid-frame logic yields 39 one-minute trials per 40-minute session; this should be revisited during later critical review to confirm whether the final minute is genuinely invalid in source data.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |

| Step | Time / Session | Estimated Total Time |
| | | |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| position_bin | 0.5619 | 0.4852 |
| <Output 2> | | |
| ... | | |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 19983720736 bytes
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | 5413 unique | code operates session-wise | 5413 unique / 69744 session-neuron instances | 69744 session-neuron instances in verifier summary | Partial (representation difference) |
| Mean neurons/session | | | | | |
| Subjects | 2 | | | | |
| Sessions | 207 | per-animal arrays sum to 207 | 207 | 207 | Yes |
| Trials (total) | 2418 | | | | |
| Trials/session (mean) | not native | expected ~40 from 40-min sessions | 39 after valid-frame trimming | 39 | Check later |
| <Input 1 range> | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| ... | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| <Output 1 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |
| <Output 2 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |

---

## Step 10: Critical Review 1
**Status**: IN PROGRESS

### Checks Performed
1. Output log verification: sample and full verification completed without errors after dropping non-finite neurons per session.
2. Raw-data sanity checks: converted neural trial 0/session 0 exactly matches raw trace slice after finite-neuron filtering (`np.allclose=True`); converted output exactly matches direct 3x3 discretization of raw position (`np.allclose=True`); square environment input is `[1,1,1,1,1,1,1,1,1]`.
3. Reference code comparison: environment input mapping now follows reference `get_env_mat`; trace representation confirmed binary and consistent with methods.

### Issues Found and Resolved
- [Issue]: [Resolution]

---

## Step 11: Full Decoder Training
**Status**: NOT STARTED

### Training Progress
- Loss decreasing: [Yes/No]

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| position_bin | 0.5619 | 0.4852 |
| <output 2> | | |
| ... | | |

---

## Step 12: Critical Review 2
**Status**: NOT STARTED

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
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
