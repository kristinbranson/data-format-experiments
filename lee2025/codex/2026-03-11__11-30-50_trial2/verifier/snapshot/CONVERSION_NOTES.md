# Dataset Conversion Notes

## Overview
- **Dataset**: Data from "Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping"
- **Date started**: 2026-03-11
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `.manifest`
- `CONVERSION_NOTES.md`
- `Dockerfile`
- `__pycache__/`
- `code/`
- `data/`
- `decoder.py`
- `docker-compose.yaml`
- `methods.txt`
- `paper.pdf`
- `train_decoder.py`

Environment checks:
- `python3` available: `3.13.12`
- `numpy` import works: `2.3.5`
- `torch` import works: `2.6.0+cu124`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_dat` | `code/georepca1/src/utils.py` | LOADING | Loads one animal dataset from `data/` as MATLAB or joblib; joblib is the default path used by the authors. |
| `generate_behav_dict` | `code/georepca1/src/utils.py` | LOADING | Builds a lightweight per-animal behavior dictionary with `position`, `envs`, and map shape metadata. |
| `get_env_mat` | `code/georepca1/src/utils.py` | PROCESSING | Converts environment identity strings to binary 3x3 occupancy masks for the geometric layouts. |
| `get_rate_maps` | `code/georepca1/src/utils.py` | PROCESSING | Bins x-y position into a `15 x 15` grid, accumulates event traces, smooths with Gaussian filter, divides by occupancy, and returns rate maps plus occupancy. |
| `clean_rate_maps` | `code/georepca1/src/utils.py` | PROCESSING | Masks out pixels outside the legal geometry for each environment so invalid spatial bins become `NaN`. |
| `get_split_half` | `code/georepca1/src/utils.py` | PROCESSING | Computes split-half map correlation within a session from first vs second half rate maps. |
| `get_shuffle_split_half` | `code/georepca1/src/utils.py` | PROCESSING | Circularly shifts behavior relative to traces to build a shuffle null for split-half reliability. |
| `get_place_cells` | `code/georepca1/src/utils.py` | CURATION | Identifies place cells from split-half reliability significance relative to shuffled data. |
| `get_shr_within` | `code/georepca1/src/utils.py` | CURATION | Applies place-cell reliability computation across all days for one animal. |
| `fit_decoder` | `code/georepca1/src/utils.py` | PROCESSING | Temporally smooths traces, averages in non-overlapping 3-frame bins, one-hot encodes 2D position, and fits Gaussian Naive Bayes with flat priors. |
| `test_decoder` | `code/georepca1/src/utils.py` | PROCESSING | Applies the trained Naive Bayes model to temporally binned traces and returns decoded positions and squared spatial error. |
| `decode_position_within` | `code/georepca1/src/utils.py` | PROCESSING | Reference within-session decoder: filters to moving periods, filters low-activity cells, uses 5-fold CV, and reports position decoding error. |

### Notes
- Codebase structure is small. Main analysis entrypoint is `code/georepca1/main.py`; most substantive logic is in `code/georepca1/src/utils.py`.
- `code/README.md` documents the dataset fields expected in each animal file: `SFPs`, `blocked`, `centroids`, `envs`, `maps`, `position`, and `trace`.
- Neural data are not raw fluorescence and no delta-F/F computation appears in the reference code. The README states `trace` is already "rise-extracted calcium traces" where `1` marks a significant event.
- Core positional representation in the reference code uses x-y behavior and calcium-event traces with frame-aligned time series.
- Spatial maps are generated on a `15 x 15` grid via `get_rate_maps`; smoothing is Gaussian with sigma `1.5` bins.
- Environment geometry is represented as 3x3 layouts via `get_env_mat`, consistent with the target decoder input requirement.
- Invalid bins outside each geometry are explicitly masked out by `clean_rate_maps`.
- Curation in the reference code is light at load time:
  - Unregistered cells are represented by `NaN` entries for a given day/session.
  - Place-cell significance is computed for paper analyses, but the within-session decoder itself does not restrict to place cells.
  - The decoder excludes time points with low running speed and cells with too few events during those valid time points.
- Reference decoder details likely relevant for conversion:
  - Behavioral samples are filtered by velocity after Gaussian smoothing of speed (`v_filt_size=5`, threshold `5` in position units/sec after bin scaling).
  - Cells are retained per day if event count during valid movement samples exceeds `5`.
  - Temporal binning is by average pooling over `3` frames in both traces and position.
  - Cross-validation is `5`-fold within day/session.
- `blocked` is documented in the README but does not appear to be used directly in the analysis code; the code more commonly uses environment labels and `get_env_mat` to encode geometry.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `data/` contains one dataset per animal in two parallel formats:
  - joblib files named by animal ID (for example `data/QLAK-CA1-51`)
  - original MATLAB v7.3 files (for example `data/QLAK-CA1-51.mat`)
- Additional files:
  - `data/behav_dict`: lightweight joblib containing per-animal `position`, `envs`, and map-shape metadata
  - `data/precomputed_results/`: large directory with cached analysis outputs from the reference code, including per-animal split-half reliability (`*_shr`) and partitioned RSM files
- Raw per-animal fields confirmed from joblib and `.mat` inspection:
  - `SFPs`: spatial footprints, joblib shape `(35, 35, n_cells, n_sessions)`
  - `blocked`: Python list of length `n_sessions`; each element is a one-item list containing either `-1` or an array of blocked partition IDs in the 3x3 layout
  - `centroids`: cell centroids, shape `(n_cells, 2, n_sessions)`
  - `envs`: environment labels, shape `(n_sessions, 1)`, strings such as `square`, `o`, `l`, `u`, `bit donut`, `rectangle`, `+`, `glenn`, `i`, `t`
  - `maps['sampling']`: occupancy/sampling maps, shape `(15, 15, n_sessions)`
  - `maps['smoothed']`: smoothed event-rate maps, shape `(15, 15, n_cells, n_sessions)`
  - `maps['unsmoothed']`: unsmoothed event-rate maps, shape `(15, 15, n_cells, n_sessions)`
  - `position`: frame-aligned x-y position, shape `(n_sessions, 2, n_frames)`
  - `trace`: frame-aligned rise-event calcium traces, shape `(n_sessions, n_cells, n_frames)`
- MATLAB v7.3 file organization is consistent with the joblib content, though dimension order differs in the HDF5 container before conversion.
- Session lengths are nearly fixed at about 40 minutes:
  - frame counts per session are one of `71866`, `72060`, `72071`, `72091`, or `72219`
  - at the reference-code frame rate of `30 Hz`, this corresponds to about `39.93` to `40.12` minutes per session
- There are no native trial objects in the raw dataset. For the requested decoder format, one-minute trials will need to be derived by splitting each session time series into consecutive 1800-frame chunks.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 5413 tracked cells across all animals (sum of registered-cell catalogs per animal from `*_shr` shapes) |
| Neurons / session | Mean 336.93 registered cells with non-NaN data per session; range 113-564 |
| Subjects | 7 mice |
| Sessions / subject | 31 for 6 mice; 21 for `QLAK-CA1-51` |
| Trials (total) | 8187 derived one-minute trials using floor(`n_frames / 1800`) per session |
| Trials / session | 39 or 40 derived one-minute trials per session (mean 39.55) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 5,413 unique neurons | "5,413 unique neurons across 207 sessions in 10 geometries" |
| Neurons / session | Not reported directly in paper; implied by 5,413 neurons / 207 sessions and raw data must be checked separately | "5,413 unique neurons across 207 sessions" |
| Subjects | 7 mice inferred from dataset/code; paper discusses animals but does not state the count in the extracted text | Not explicitly stated in extracted text |
| Sessions / subject | Up to 31 daily sessions (three repetitions of an 11-session sequence with overlap at square) | "up to three total repetitions (31 days)" and "Sequence 1: Session 1-11; Sequence 2: Session 11-21; Sequence 3: Session 21-31" |
| Trials (total) | No native trials reported; sessions are continuous 40 min recordings | "All sessions were 40 min" |
| Trials / session | No native trials reported; conversion will derive one-minute trials | "All sessions were 40 min" |
| Neural data time bin | 33.33 ms native frame interval (30 Hz) | "behavioral and cellular imaging streams at 30 Hz" |
| Behavior data time bin | 33.33 ms native frame interval (30 Hz) | "behavioral and cellular imaging streams at 30 Hz" |
| Reward rate | N/A; free exploration task with no reward variable described | Not reported |
| Environment count | 10 geometries | "sequence of 10 geometrically distinct environments" |
| Environment size | 75 x 75 cm square reference arena | "open square (75 x 75 cm)" |
| Rate maps used in paper | 69,744 | "forming 69,744 rate maps" |
| Place-cell criterion | Split-half map correlation > 99th percentile of 1000 shuffled position controls | "1000 circular shuffles" and "exceeded the 99th percentile" |
| Decoder benchmark | No numeric accuracy/error value reported in text; paper reports decoding error decreased across recordings and matched recent work | "significant decrease in position decoding error across recordings, achieving the maximum decoding accuracy reported in recent work" |


### Processing Details
- Neural and behavioral streams are acquired simultaneously at `30 Hz` with recorded timestamps for post-hoc alignment.
- Each session is a continuous `40 min` free-exploration recording, one session per day.
- The task is a repeated geometric-deformation sequence:
  - animals start and end each sequence in the square environment
  - sequence length is effectively 11 sessions because the square is repeated at sequence boundaries
  - up to 3 repetitions produce 31 total days in the full animals
- Calcium preprocessing from the paper:
  - motion correction, cell segmentation, and transient extraction performed in MATLAB before the released dataset
  - derivative of filtered calcium trace smoothed with Gaussian kernel, standard deviation `5` frames
  - noise estimated from negative derivative values via half-normal assumption
  - z-scored derivative thresholded at `2.5` to create the final binary rising-phase vector
- The binary rising-phase vector is the neural signal used in all subsequent analyses; this is the signal that should be treated as neural activity for conversion.
- Position was derived from DeepLabCut head tracking.
- Bayesian position decoder in the paper:
  - within-session
  - `5`-fold split
  - spatially binned position converted to one-hot representation
  - Gaussian Naive Bayes with flat prior and default `var_smoothing` (`1e-9`)
  - error measured as Euclidean distance between predicted and actual spatial bin

### Curation Steps

**Neuron curation rules**:
- Motion-corrected imaging was manually inspected for artifacts.
- Spatial footprints were manually verified to remove lens artifacts.
- Cells were tracked across sessions using brain-surface landmarks, spatial footprints, and/or centroids.
- Place-cell significance was assessed for some analyses, but the paper states that the high reliability of the recordings "motivated the inclusion of all cells in subsequent analyses."

**Trial curation rules**:
- No native trial structure is described in the paper; sessions are continuous 40 min recordings.
- Place-cell analysis uses first vs second 20 min session halves.
- Position decoding is within-session on spatially binned, frame-aligned data; the paper text does not describe additional trial rejection rules.

### Decoders Trained
| Decoded variable | Accuracy |
| Position (Bayesian within-session) | No explicit numeric value reported in extracted paper text; paper reports decreasing error across recordings and Figure 1F provides the trend |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Total unique neurons | Reference analyses operate on all registered cells; `main.py` lists 7 animals | Sum of per-animal cell catalogs is 5,413 | "5,413 unique neurons" | Fully consistent. Use 5,413 as the total tracked-cell count benchmark. |
| Total sessions | Code loops over 7 animals and all days in each animal file | 207 sessions total from `behav_dict` / `*_shr` shapes | "207 sessions" | Fully consistent. |
| Total rate maps | Not stated directly in code, but one rate map exists for each valid cell-session combination | Sum of non-NaN entries in `*_shr` across sessions is 69,744 | "forming 69,744 rate maps" | Fully consistent. This is a strong sanity check for registration / missing-cell handling. |
| Cells per animal | Code defines 7 animals and uses all registered cells | Per-animal totals: `515, 875, 942, 554, 862, 713, 952` | "mean number of cells per animal = 773 ± 68 SE, minimum = 515, maximum = 952" | Fully consistent. Raw data reproduce the mean/SE/min/max reported in the paper. |
| Session duration | Decoder code assumes `fps=30`; no explicit exact frame count requirement | Raw sessions have 71,866 to 72,219 frames, i.e. about 39.93 to 40.12 min | "All sessions were 40 min" | Consistent up to minor frame-count variation. Use raw frame counts directly and derive one-minute trials by floor division into 1800-frame chunks. |
| Native vs derived trials | Reference code analyzes continuous sessions/days, not trials | No native trials in raw files | Paper describes continuous 40 min sessions | Not a contradiction. Conversion will derive 1-minute trials solely to satisfy the target decoder format. |
| Neural signal definition | Reference code uses `trace` directly as event-like firing-rate proxy; no dF/F computation | `trace` arrays are binary in sampled checks (`0/1`) | Paper says binary rising-phase vector thresholded at `2.5` is used as "firing rate" | Fully consistent. Do not recompute dF/F or deconvolution. |
| Place-cell filtering | `get_place_cells` exists, but within-session decoder uses all cells and only filters by movement / activity | Raw files include all registered cells per animal, with missing sessions encoded as `NaN` | Paper says high reliability "motivated the inclusion of all cells in subsequent analyses" | Use all registered cells, then apply per-session valid-cell handling via non-NaN masking; do not restrict conversion to place cells. |
| Raw dimension order | MATLAB HDF5 layout stores session-first arrays for several fields | Joblib files expose analysis-ready shapes such as `trace (sessions, cells, frames)` and `maps (15,15,cells,sessions)` | Paper does not discuss storage order | Use joblib as the primary source to match reference code behavior and avoid axis-order mistakes. |
| Geometry encoding | Reference code mostly uses `envs` labels plus canonical `get_env_mat(env)` | Raw files also contain explicit `blocked` partition lists in 3x3 indexing | Paper emphasizes a 3x3 partitioned square and 10 geometries | Use `blocked` for decoder input construction because it directly represents blocked partitions in the raw dataset. Keep `envs` as session metadata / cross-check against code conventions. |
| Number of sessions per animal | `main.py` implies up to three sequences | Six animals have 31 sessions; one animal has 21 sessions | Paper says sequences repeated "up to three total repetitions (31 days)" | Consistent because the paper says "up to" three repetitions. The 21-session animal reflects fewer completed repeats, not a mismatch. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `trace[session, valid_cells, frame]` | `neural[session][trial]` | Session-level movement filter + low-activity cell filter; within each one-minute chunk keep only movement-valid frames, Gaussian smooth traces, and average-pool every 3 frames; store as `(n_neurons, n_timepoints)` | `decode_position_within`, `fit_decoder`, `test_decoder` | This mirrors the reference position-decoder preprocessing more closely than storing raw 30 Hz binary events. |
| `blocked[session]` | `input[session][trial]` | Convert blocked partition list into a length-9 binary geometry vector that is constant within a trial | README field description; cross-check with `get_env_mat` / `envs` | Use the raw `blocked` field because it directly encodes which 3x3 partitions are occluded. Same vector repeated conceptually for every time point, but stored as static per-trial input. |
| `position[session, :, frame]` | `output[session][trial]` | Compute movement-valid frames, discretize position to aligned 3 x 3 bins, snap blocked bins to nearest open bin, average-pool coordinates in 3-frame groups, and collapse to a 9-class categorical index with shape `(1, n_timepoints)` | `decode_position_within`, `fit_decoder`, `test_decoder` | Output is time-varying and uses the same temporally pooled samples as the neural data. |
| animal ID | `subjects`; `subject_idx` | Unique list of mouse IDs; session-level index into list | `main.py` animal list | Session order in converted data will follow animal-file order, then session/day order within animal. |
| Constant `CA1` | `brain_regions`; `brain_region_idx` | Single region for all neurons | Paper and prompt context | All neurons are from hippocampal CA1. |
| `envs[session]` | `metadata['session_info']` or auxiliary metadata | Preserve for interpretability and cross-checking | README and code | Not a decoder input by itself; use as metadata and geometry sanity check. |

### Key Decisions
1. **Use joblib files as the primary raw source**: They expose the same field names as the reference code and already match the analysis-ready axis ordering, reducing the risk of silent transpose errors relative to the MATLAB HDF5 layout.
2. **Apply the reference decoder’s session-level preprocessing before trial export**: The paper/code decode position after filtering to movement-valid samples, dropping very-low-activity cells, smoothing traces, and average-pooling every 3 frames. Exporting this processed representation is both closer to the reference decoder and substantially more memory-efficient.
3. **Split sessions into full one-minute chunks first, but within each chunk keep only movement-valid frames and pool in groups of 3**: This satisfies the user’s trial-format requirement while remaining consistent with the reference decoder’s sample-selection logic.
4. **Retain all session-valid registered cells initially, then apply the reference decoder’s low-activity cell filter per session**: This matches the code path in `decode_position_within` more closely than exporting every registered cell regardless of activity.
5. **Drop cells with any NaNs within a session before applying the activity filter**: Raw files track cells across days, but the target format is session-centric and downstream decoder code should not receive NaNs for nonexistent cells.
6. **Use raw `blocked` partitions to build decoder inputs**: This satisfies the requested input specification directly and avoids ambiguities from canonical geometry labels and plotting/masking orientation conventions.
7. **Encode geometry as 9 binary partition features**: One feature per 3x3 partition provides the decoder with the full static environmental context. The planned feature semantics are `1 = open`, `0 = blocked`, consistent with the reference code’s `get_env_mat` convention after constructing the vector from `blocked`.
8. **Encode mouse location as one categorical output over time with 9 possible values**: This matches the requested decoder task and keeps the output compact and explicitly categorical.
9. **Derive 3x3 position bins from the same arena scale used by the reference decoder logic**: Planned implementation is to compute a per-animal spatial scale from the maximum tracked coordinate across that animal’s sessions, then bin positions into 3 equal partitions without re-centering. This mirrors the reference code’s use of a global max over sessions for position binning.
10. **Snap blocked-bin position samples to the nearest open partition before export**: The reference decoder cleans positions/predictions to the nearest valid bin when evaluating decoding. Performing the analogous cleanup at the coarse 3x3 level should reduce label noise from tracking jitter in physically inaccessible regions.
11. **Use row-major partition labels for outputs and document them explicitly**: `output_values[0]` will name the 9 bins in the same 3x3 order used for blocked partitions, so geometry input and position output share a common indexing scheme.

### Planned Sanity Checks
- [x] Check that total converted sessions, cells, and derived trials match counts expected from raw data (`207` sessions; `8187` full one-minute trials; session cell counts equal non-NaN registered cells).
- [x] Check that converted neural slices exactly match raw `trace` values for selected `(session, trial, cell, frame)` coordinates.
- [x] Check that converted geometry inputs reconstruct the raw `blocked` lists exactly for selected sessions.
- [x] Check that converted position-bin outputs match direct discretization from raw `position` samples for selected `(session, trial, frame)` coordinates.
- [x] Check that blocked spatial partitions have near-zero occupancy in the converted output for each session, allowing only rare boundary/noise exceptions.
- [x] Check that per-animal/session statistics remain consistent with the paper and precomputed reference outputs (5,413 total cells; 69,744 valid cell-session rate maps; 773 ± 68 SE cells/animal).

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `convert_data.py` with the required CLI:
- `python -u convert_data.py <outpicklefile>`
- `--full` to process all sessions
- `--sample` to process only 2 sessions
- `--show-processing` to save up to 2 session-level processing figures

Implementation highlights:
- Uses joblib animal files from `data/` to match the reference code storage path.
- Infers a 3x3 coordinate transform per animal by minimizing occupancy in raw blocked partitions, then uses that transform to align position bins with the raw `blocked` geometry definition.
- Applies preprocessing modeled on the reference within-session decoder:
  - movement filtering from smoothed speed (`5 cm/s` threshold)
  - session-level low-activity cell filtering (`>5` events during movement-valid frames)
  - Gaussian smoothing of traces
  - non-overlapping 3-frame average pooling
- Splits sessions into one-minute chunks, keeping chunks as trials and exporting time-varying pooled position labels.
- Snaps position samples that land in blocked bins to the nearest open bin to reduce label noise from tracking jitter.
- Stores decoder input as a static 9-feature geometry vector (`1=open`, `0=blocked`) and output as a single 9-class categorical position stream.

Code inefficiencies identified:
- Animal joblib loads are relatively slow (roughly tens of seconds per animal), so the script processes animals sequentially and frees memory between animals.

Code speedups added:
- Moved heavy preprocessing to vectorized NumPy/Scipy operations.
- Performs one transform inference per animal instead of per session.
- Uses session-level filtering before trial splitting to reduce exported data size.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 335 active CA1 neurons across the 2 sample sessions |
| Neurons / session | 182, 153 |
| Subjects | 1 represented in sample (`QLAK-CA1-08`) |
| Sessions / subject | 2 |
| Trials (total) | 77 |
| Trials / session | 38, 39 |
| `partition_4_open` range | [0.0, 1.0] |
| Other geometry inputs range | [1.0, 1.0] in this 2-session sample |
| `position_bin` distribution | [0.1025, 0.0830, 0.1183, 0.0839, 0.0536, 0.1342, 0.1005, 0.1203, 0.2037] |

### Processing Plots Review
- Reviewed `processing_QLAK-CA1-08_s00.png` and `processing_QLAK-CA1-08_s01.png`.
- Geometry input matches session environment structure (`square` all open; `o` has blocked center partition).
- Raw and aligned/snapped occupancy plots are consistent with the geometry input.
- Pooled position output varies across bins and avoids obvious occupancy of blocked space.
- Neural preprocessing plots show sparse raw events becoming smooth pooled features, as intended.
- Initial sample run produced one all-zero neural trial warning and a minimum pooled length of 4 samples. Fixed by excluding trials with fewer than 10 pooled samples or no processed neural activity, then re-ran Step 7.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |
| Session-level movement filtering + 3-frame pooling | Reduces exported trial length from 1800 raw frames to ~289 pooled valid samples on average in sample |
| Session-level active-cell filtering | Reduces exported neurons to 153-182 active cells in sample sessions |
| One transform inference per animal | Avoids repeated geometry-alignment computation per session |

| Step | Time / Session | Estimated Total Time |
| | | |
| Sample conversion (measured) | ~1.67 s/session after first animal load; first animal load ~62.46 s | ~13.0 min for all 207 sessions using `7 * 62.46 s + 207 * 1.67 s` |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| `position_bin` | 0.5131 | 0.4152 |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 3.2G
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | 5413 tracked cells | 5413 tracked cells used across analyses | 5413 tracked cells | 68862 exported session-neurons after activity filter | By design different representation |
| Mean neurons/session | Not explicitly reported | ~337 registered/session implied from code+data | 336.93 non-NaN registered/session | 332.67 active/session | Yes, close after decoder-specific activity filter |
| Subjects | 7 inferred from full dataset/code | 7 animals in `main.py` | 7 | 7 | Yes |
| Sessions | 207 | 207 | 207 | 207 | Yes |
| Trials (total) | N/A (continuous sessions) | N/A (continuous sessions) | 8187 full one-minute chunks before quality curation | 8109 exported trials | Yes, after documented trial curation |
| Trials/session (mean) | N/A | N/A | 39.55 full chunks/session | 39.17 exported/session | Yes, after documented trial curation |
| Geometry input range | N/A | Canonical 0/1 geometry masks | Raw blocked/open partitions imply binary 0/1 features | All 9 inputs in [0, 1] | Yes |
| `position_bin` distribution | Not numerically reported | N/A | Depends on pooled occupancy | [0.0977, 0.1104, 0.1130, 0.0893, 0.0707, 0.0864, 0.1093, 0.1567, 0.1664] | Plausible and consistent with free exploration |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `verification_full_out.txt` reports "Data format is valid, no errors or warnings." No fixes required at this step.
2. **Independent raw-data sanity checks with `np.allclose()`**: Reconstructed converted trials directly from original joblib animal files without calling `convert_data.py`, and compared against `converted_data.pkl`.
3. **Reference code comparison**: Compared conversion logic to `load_dat`, `decode_position_within`, `fit_decoder`, and `test_decoder` in the reference code.
4. **Key statistics comparison**: Checked subjects, sessions, cells/animal, rate-map count benchmark, mean active cells/session, trial counts, and output distribution against paper/raw/reference outputs.
5. **Edge-case review**: Investigated sessions with unusually low exported trial counts and confirmed they coincide with unusually low movement-valid fractions rather than indexing bugs.

### Issues Found and Resolved
- **All-zero sample trial warning**: Found during Step 7 verification. Resolved by excluding trials with fewer than 10 pooled samples or no processed neural signal, then re-running sample/full conversion and validation.
- **Neural sanity check**: For `(session 0, trial 0)`, `(session 31, trial 5)`, and `(session 176, trial 3)`, independently reconstructed processed neural matrices from raw `trace` + movement mask + active-cell filter + smoothing + pooling matched converted arrays exactly with `np.allclose()`.
- **Input sanity check**: For the same three spot-checks, independently reconstructed 9-element geometry vectors from raw `blocked` matched converted trial inputs exactly with `np.allclose()`.
- **Output sanity check**: For the same three spot-checks, independently reconstructed pooled/snapped 9-class position outputs from raw `position` matched converted arrays exactly with `np.allclose()`.
- **Reference-code alignment**:
  - Data loading: matches `load_dat` by using the joblib animal files.
  - Neuron filtering: stricter than the paper’s general analyses but intentionally matches the reference position decoder by using session-level movement-valid activity thresholding.
  - Temporal alignment: preserves the simultaneous neural/behavior streams and applies filtering on the same frame axis.
  - Binning: uses the same idea as the reference decoder, namely position binning from a per-animal global spatial scale and 3-frame average pooling.
  - Input construction: adapted from raw `blocked` rather than `envs` strings because the decoder task explicitly requires blocked-partition geometry.
  - Output construction: adapted from the reference x-y positional binning to a single 9-class categorical label required by the task.
- **Key-statistics comparison**:
  - Paper/raw/reference agree on `5413` tracked cells, `207` sessions, and `69744` valid cell-session rate maps.
  - Converted data keep all `207` sessions and `7` subjects.
  - Converted mean active cells/session is `332.67`, close to the raw mean registered cells/session (`336.93`), consistent with the reference decoder’s additional activity filter.
  - Converted total trials are `8109` rather than the raw floor-split count `8187` because four low-movement sessions lose chunks during trial-quality curation.
- **Edge cases**:
  - Only four sessions exported fewer than 35 trials:
    - `QLAK-CA1-74_s17` (`34`)
    - `QLAK-CA1-74_s26` (`31`)
    - `QLAK-CA1-74_s27` (`31`)
    - `QLAK-CA1-75_s01` (`24`)
  - These sessions have the lowest movement-valid fractions (`0.217` to `0.351`), which explains the reduced post-filter trial counts.
  - No converted output samples occupy blocked geometry bins.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| `position_bin` | 0.7767 | 0.6856 | Chance = 0.1111; test loss = 0.925141 |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| `position_bin` | Validation balanced accuracy `0.6856` (chance `0.1111`) | Paper reports a significant decrease in decoding error across recordings and states the dataset achieves the maximum decoding accuracy reported in recent work, but the extracted text does not provide a numeric accuracy/error value |

[Analysis of any low accuracies]
- Accuracy is far above chance (`0.6856 / 0.1111 = 6.17x chance`), so there is no indication of a temporal-alignment or label-construction bug from the final decoder result.
- Training vs validation gap is modest (`0.7767 / 0.6856 = 1.13x`), well below the `1.5x` threshold called out for overfitting concern.
- Because the paper’s extracted text gives only qualitative decoder benchmarking rather than a numeric position-decoding score, a strict numeric paper-to-converted comparison is not possible from the available manuscript text. The converted dataset is nevertheless consistent with the qualitative paper expectation of strong position decodability.

### Issues Found and Resolved
- **Accuracy vs chance**: No issue. Validation balanced accuracy for `position_bin` is strongly above chance.
- **Accuracy comparison to paper**: No numeric paper value available in extracted text; documented this limitation explicitly rather than inventing a comparison target.
- **Train/validation gap**: No issue. Gap does not suggest serious leakage or overfitting.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
