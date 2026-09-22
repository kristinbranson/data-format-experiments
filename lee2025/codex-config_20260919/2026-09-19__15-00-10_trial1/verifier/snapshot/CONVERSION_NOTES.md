# Dataset Conversion Notes

## Overview
- **Dataset**: Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping (provided paper/code/data)
- **Date started**: 2026-09-19
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `.manifest`
- `CONVERSION_NOTES.md`
- `Dockerfile`
- `code/`
- `data/`
- `decoder.py`
- `docker-compose.yaml`
- `methods.txt`
- `paper.pdf`
- `train_decoder.py`

Environment check: Python 3.13.15, NumPy 2.4.4, and PyTorch 2.6.0+cu124 import successfully. Checkpoint confirmed with `ls -la /app/CONVERSION_NOTES.md`.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_dat` | `code/georepca1/src/utils.py:61` | LOADING | Loads one animal's MATLAB or joblib dictionary; converts `envs`, `position`, and `trace` to NumPy for MATLAB input. |
| `generate_behav_dict` | `code/georepca1/src/utils.py:130` | LOADING | Extracts position, environment labels, and map shapes into a lightweight all-animal dictionary. |
| `get_env_mat` | `code/georepca1/src/utils.py:215` | PROCESSING | Maps each named arena geometry to a 3x3 occupied-partition matrix (1 accessible, 0 blocked). |
| `get_rate_maps` | `code/georepca1/src/utils.py:313` | PROCESSING | Spatially bins position into 15x15, sums events, Gaussian-smooths numerator and occupancy (sigma 1.5 bins), divides and multiplies by 30 fps. |
| `get_split_half` / `get_place_cells` / `get_shr_within` | `code/georepca1/src/utils.py:356,416,442` | CURATION | Quantifies split-half spatial reliability against circularly shifted position; p<0.05 marks place cells for analyses, but does not alter stored traces. |
| `clean_rate_maps` | `code/georepca1/src/utils.py:463` | PROCESSING | Masks 15x15 rate maps using a 5x upsampled/transposed/flipped 3x3 arena geometry. |
| `fit_decoder` / `test_decoder` | `code/georepca1/src/utils.py:1776,1804` | PROCESSING | Gaussian-smooths traces over 3 frames, averages non-overlapping 3-frame bins (10 Hz), averages/integers position, and uses flat-prior Gaussian Naive Bayes. |
| `decode_position_within` | `code/georepca1/src/utils.py:1845` | CURATION | Bins position to 15x15; retains movement >5 cm/s and cells with >5 active-frame events; performs contiguous 5-fold within-session decoding. |

### Notes
- The seven animals are explicitly listed in `main.py`: QLAK-CA1-08, -30, -50, -51, -56, -74, and -75.
- Native neural data are already processed **rise-extracted calcium event traces**, with 1 denoting a significant event and NaN denoting a cell not registered that day. Delta-F/F must not be recomputed; the raw fluorescence needed for that operation is not the reference decoder input.
- Position and trace samples are frame-aligned at 30 Hz. Reference spatial decoding smooths traces (Gaussian sigma 3 frames) and pools by 3 frames, yielding 100 ms samples, but the requested one-minute trials motivate preserving full 30 Hz event timing unless decoder memory/runtime requires reference-equivalent temporal pooling (resolved in Step 5).
- `blocked` is described in the README as indices in row-major 3x3 geometry, with -1 for none. `get_env_mat` is the authoritative named-geometry representation and returns occupied (not blocked) cells.
- Reference place-cell reliability is an analysis label, not a global cell filter. The reference within-session decoder instead applies session-specific activity (>5 events while moving) and velocity (>5 cm/s) filters. How these apply to the requested categorical decoder is deferred to cross-source consistency and mapping steps.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Seven paired source files are present as MATLAB `.mat` and compressed extensionless joblib files, named by animal. Joblib is the reference code's default and preserves a top-level `{animal_id: dataset}` dictionary.
- Per animal fields: `SFPs` `(35,35,n_global_cells,n_days)` float64 footprints; `blocked` length-`n_days` nested lists of blocked 3x3 indices; `centroids` `(n_global_cells,2,n_days)`; `envs` `(n_days,1)` strings; `maps` containing 15x15 `sampling`, `smoothed`, and `unsmoothed` maps; `position` `(n_days,2,n_frames)` float64; and `trace` `(n_days,n_global_cells,n_frames)` float64 binary calcium events plus NaNs for unregistered cells.
- Within each animal, all sessions have the same recorded length and fully finite position. Registered cell traces are finite binary {0,1}; unregistered cell/session pairs are entirely NaN. Neural and position frames correspond one-to-one.
- Position coordinates span approximately 0 to 75 cm on both axes. Each arena name repeats in three 10-geometry sequences (two for animal 51); square appears once more as the final sequence endpoint.
- Other files are derived/precomputed analyses (`*_shr`, `*_rsm_partitioned`, `within_decoding`, map correlations, pixelwise correlations, `behav_dict`) and are not primary conversion sources.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 5,413 longitudinal cell identities across subjects; 69,744 registered session-neuron instances |
| Neurons / session | 113-564, mean 336.93 registered cells |
| Subjects | 7 |
| Sessions / subject | 31 for six subjects; 21 for QLAK-CA1-51 (207 total) |
| Trials (total) | No native trials; 8,187 complete non-overlapping 60 s trials at 30 Hz after dropping final partial minutes |
| Trials / session | 39 for animals 08/30/50 (71,866 frames); 40 for animals 51/56/74/75 (72,219/72,091/72,060/72,071 frames) |

Per-animal cell identities / registered session-neuron instances: 08 515/6,631; 30 875/11,812; 50 942/12,427; 51 554/4,831; 56 862/11,807; 74 713/9,680; 75 952/12,556. Only 112 of 69,744 registered instances have <=5 events over the full session, confirming that the reference decoder's activity filter is mild before its separate running filter.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 5,413 unique; 69,744 rate maps/session-neuron instances | Paper p.309 / methods.txt: “5,413 unique neurons across 207 sessions…forming 69,744 rate maps” |
| Neurons / session | 69,744 / 207 = 336.93 mean | Derived exactly from reported totals |
| Subjects | 7 (4 male, 3 female) | STAR Methods p.e1 |
| Sessions / subject | Up to 31 days; 207 total | Figure 1D legend and paper p.309 |
| Trials (total) | No native trials | Sessions were continuous free exploration; one-minute trials are required downstream |
| Trials / session | One 40-minute recording/day | STAR Methods p.e2: “All sessions were 40 min” |
| Neural data time bin | 1/30 s = 33.333 ms | STAR Methods p.e2: cellular imaging at 30 Hz |
| Behavior data time bin | 1/30 s = 33.333 ms | STAR Methods p.e2: simultaneous behavioral/cellular streams at 30 Hz, timestamp-aligned |
| Reward rate | N/A | Free exploration; no reward outcomes described |
| Arena size | 75 x 75 cm, conceptual 3x3 grid | Main text pp.308-309 |
| Arena conditions | 10 geometries, repeated up to 3 sequences | Main text p.309 / Figure 1D |
| Session duration | 40 min | STAR Methods p.e2 |


### Processing Details
- Motion correction and cell segmentation/transient extraction were completed upstream. Each median-subtracted calcium trace was differentiated, Gaussian-smoothed with sigma 5 frames, noise-scaled using the negative derivative half-normal distribution, thresholded at z>2.5, and binarized. The paper treats this rising-phase vector as firing rate in all analyses.
- Neural and head position streams were acquired simultaneously at 30 Hz and timestamped for post-hoc alignment; the distributed arrays are already equal length and aligned.
- Position was obtained using DeepLabCut head tracking. Longitudinal cells were registered using brain landmarks, footprints, and/or centroids.
- Reference rate maps use 5x5 cm spatial pixels (15x15 over 75 cm) and a 5 cm Gaussian smoothing standard deviation. Reference code implements 1.5 spatial-bin smoothing, nominally 7.5 cm; this discrepancy is addressed in Step 4 and does not affect the requested raw time-series representation.
- Paper position decoding uses 5-fold withheld data, one-hot spatial position, Gaussian Naive Bayes, flat priors, and Euclidean position error. Figure 1F reports error rather than classification accuracy: the group curve is approximately 20-22 cm initially and approximately 11-13 cm late in recording (read from graph; no exact numerical table in text), with significant decline across days (ANOVA p<0.0001, F=7.9845).

### Curation Steps

**Neuron curation rules**:
Spatial footprints were manually verified to remove lens artifacts before release. Place cells were separately identified when split-half map correlation exceeded the 99th percentile of 1,000 circular position shuffles (p<0.01 in paper); this is an analysis selection, not preprocessing of the released trace population.

**Trial curation rules**:
No trial rejection is described. Each session is a continuous 40-minute free-exploration recording. Sessions occur once per day to avoid photobleaching.

### Decoders Trained
| Decoded variable | Accuracy |
| 15x15 spatial position | Mean Euclidean error approximately 20-22 cm early and 11-13 cm late; paper Figure 1F |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Population totals | Main code lists 7 animals and loads all registered session cells | Arrays contain 5,413 global identities, 207 days, 69,744 finite session-cell pairs | 5,413 / 207 / 69,744 | Exact agreement. |
| Session duration | Decoder operates on all available frames | 71,866-72,219 frames = 39.93-40.12 min at 30 Hz | 40 min | Small acquisition/timestamp variation is expected; preserve complete 60 s chunks and omit <60 s tail. |
| Neural representation | Uses distributed binary `trace` directly | Registered traces contain only 0/1 | Rise-extracted, z>2.5 binarized events treated as firing rate | Exact agreement; do not compute dF/F or re-extract events. |
| Stream alignment | Passes same time indices from `position` and `trace`; both 30 Hz | Identical session frame counts with no missing position samples | Simultaneous 30 Hz acquisition and timestamp alignment | Exact agreement. |
| Blocked-grid orientation | `get_env_mat` gives display-oriented 3x3 occupancy; map cleaning transposes/flips for plotted rate-map axes | `blocked` row-major indices correspond to `(y,x)`; position arrays are `(x,y)` | Conceptual row-major 3x3 layout | Transpose source blocked matrices to `(x_bin,y_bin)` before flattening, so input dimension `x*3+y` matches output class. Zero-occupancy bins empirically match this transform for all 10 geometries. |
| Place-cell threshold | `get_place_cells` defaults to alpha=0.05; main script stores continuous p-values and plots p<.05/.01/.001 | Precomputed p-values exist; raw traces retain all cells | Formal definition is >99th shuffle percentile (p<.01) | Not a preprocessing discrepancy. No place-cell-only filtering for conversion; paper decoding explicitly uses all registered cells. |
| Decoder cell/movement selection | `decode_position_within`: speed >5 cm/s and >5 events during retained frames | Nearly all cells have >5 events over whole sessions (69,632/69,744) | Paper decoder text omits these implementation details but says all registered cells | Reference implementation is authoritative for reproducing its Euclidean error. Requested time-varying categorical output should retain all frames and registered cells so trials represent full location occupancy; otherwise outputs would cease to be contiguous one-minute trials. |
| Spatial smoothing | `get_rate_maps` default sigma=1.5 of 5 cm bins (7.5 cm) | Distributed maps reflect existing processing; time series are unsmoothed | Methods says sigma 5 cm | Irrelevant to exported raw aligned event vectors. Do not substitute rate maps for neural time series. |
| Position benchmark | Precomputed code output: 5-fold error overall 13.49 cm (range 5.76-32.46 across fold-days) | `within_decoding` has shape `(days,5)` per animal | Figure 1F shows ~20-22 cm early to ~11-13 cm late | Quantitatively consistent with plotted group trend. The requested 9-class task differs from the paper's 225-bin Euclidean-error decoder, so only qualitative comparison is valid. |

Final understanding: every recording day becomes one target session, registered/finite CA1 event traces become neural rows, aligned head position becomes a 3x3 categorical output at every retained frame, and static blocked geometry becomes nine decoder-input dimensions. Continuous sessions are cut from frame zero into exact 1,800-frame trials; only the final incomplete segment is excluded.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `trace[day, registered_cells, :]` | `neural[session][trial]` | Select cells finite on that day; Gaussian smooth each binary event vector along time with sigma=3 source frames; average non-overlapping 3-frame bins; split into 600-bin (60 s) float32 trials | `fit_decoder`, `test_decoder` | Shape `(n_registered_cells,600)`; 100 ms bins exactly reproduce reference decoder temporal preprocessing. |
| `blocked[day]` | `input[session][trial][0:9]` | Build row-major 3x3 blocked mask, transpose from source `(y,x)` to target `(x,y)`, flatten as `x*3+y`; 1=blocked, 0=accessible; repeat same 1D float32 vector for each trial | `get_env_mat`, `clean_rate_maps` | Static per trial as required. Square source sentinel -1 maps to all zeros. |
| `position[day,0:2,:]` | `output[session][trial][0,:]` | Average x/y over the same non-overlapping 3 source frames, floor each coordinate into 3 equal bins over 0-75 cm (clip to 0..2), then class=`x_bin*3+y_bin` | Behavioral pooling in `fit_decoder`/`test_decoder`; spatial binning in `decode_position_within` | Categorical int8, shape `(1,600)`, classes 0..8. |
| animal ID | `subjects`, `subject_idx` | Seven canonical IDs; per-day session points to subject | `main.py` animal list | Session order: listed animal order, then chronological day. |
| CA1 recording | `brain_regions`, `brain_region_idx` | `brain_regions=['CA1']`; zero vector per session | Paper experiment description | One label per retained registered neuron. |
| `SFPs`, `centroids` | metadata only / omitted from decoder arrays | Used upstream for manual verification and registration; source global cell indices retained in each `session_info` record | Paper preprocessing | Not decoder task variables and too large to duplicate. |
| `envs` | `metadata.session_info[*].environment` | Preserve exact environment string | Reference loader | `blocked` is the actual decoder input; environment name is descriptive metadata. |
| `maps` (`sampling`, `smoothed`, `unsmoothed`) | metadata provenance only | Do not export; derived from the same trace/position and would leak target spatial information if used as inputs | `get_rate_maps`, `clean_rate_maps` | Not an independent experimental variable. |

### Key Decisions
1. **Temporal bins are 100 ms**: The reference position decoder Gaussian-smooths trace events with sigma equal to 3 frames and applies non-overlapping 3-frame average pooling to both neural and position streams. Applying this once over each full session before trial slicing exactly matches that processing, avoids artificial trial-edge smoothing discontinuities, and keeps the full dataset tractable (~6.7 GB neural float32 rather than ~20 GB at 30 Hz).
2. **Exact one-minute trials**: Start at session frame zero, use consecutive 1,800-frame (600 pooled-bin) chunks, and drop only the terminal 2.0-55.5 s partial segment. Fixed-length trials satisfy the requirement and preserve 8,187/8,187 possible complete minutes.
3. **All registered cells, no place-cell or running filter**: A registered cell is finite at the first frame (and verified finite throughout). This gives the paper's exact 69,744 rate-map count. Place-cell labels are downstream analyses. Running-only filtering would destroy contiguous one-minute timing and remove stationary position labels; the paper describes Figure 1F as using all registered cells.
4. **Position class order**: Class `3*x_bin+y_bin`, where each bin is [0,25), [25,50), or [50,75] cm. Output labels are `x0_y0` ... `x2_y2`. The source blocked mask is transposed so its flattened dimensions use the same class coordinate system; raw occupancy confirmed blocked/accessible correspondence.
5. **Geometry input is blockedness, not named condition**: Nine float32 binary dimensions retain exact geometry and generalize across names. No environment identity one-hot is added because the explicit requested input is geometry.
6. **No added speed or position features**: Speed is derivable behavior but not requested as decoder input; position is solely the categorical output. Adding position-derived inputs would create leakage even though the provided decoder currently learns only from neural activity.
7. **Numeric types**: Neural and input are float32 as expected by validator/trainer; output is int8 categorical. Pickle protocol 4/5 preserves arrays without precision relevant to the source binary events.
8. **Session metadata**: Preserve subject, zero-based source day, environment, source/retained/dropped frames, registered global cell indices, source and target sampling rates, and transforms for auditability.

### Planned Sanity Checks
- [ ] Direct raw-neural spot checks with `np.allclose(converted, gaussian_filter1d(raw)[...,reshape(...,3)].mean(-1))` for multiple sessions/trials/cells/time bins.
- [ ] Direct raw-input checks with `np.allclose(converted_input, raw blocked mask reshaped then transposed/flattened)` including square sentinel and multi-block geometries.
- [ ] Direct raw-output checks with `np.allclose(converted_output, 3*x_bin+y_bin)` from independently pooled raw position for multiple trial edges.
- [ ] Verify exactly 7 subjects, 207 sessions, 69,744 session-neuron instances, 8,187 trials, 600 time bins/trial, and trial counts 39 or 40 as derived independently.
- [ ] Verify every output is 0..8, every input is binary, every neural value finite/nonnegative, and each geometry's blocked output classes have zero or negligible occupancy.
- [ ] Compare source joblib and original MATLAB arrays on sampled values to ensure default loading does not alter content.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with the required CLI (`OUTPUT`, `--full` default, `--sample`, `--show-processing`). It loads animals sequentially, selects registered cells per day, processes full sessions before slicing, validates every array, records source-cell/session provenance, writes via a temporary file and atomic rename, and prints load/session/pickle timings. `--show-processing` produces a six-stage audit figure for up to two sessions: raw events, smoothed/pooled neural data, aligned raw/pooled position, occupancy plus blocked mask, categorical output, and trial-boundary/check summary. Python compilation, CLI help, blocked-orientation, position-pooling, and synthetic neural-pooling tests passed.

Code inefficiencies identified:
Naively preserving 30 Hz float32 would require about 20 GB and would diverge from the reference decoder's pooling. Loading every animal simultaneously would additionally materialize tens of GB of compressed joblib arrays. Per-trial filtering would repeat convolution and introduce edge effects. A float64 filter output would double processed neural memory.

Code speedups added:
Animals are loaded one at a time; Gaussian filtering writes directly to a preallocated float32 array; filtering/pooling is vectorized over cells and time; the 3-frame reshape/mean replaces Python time loops; full-session processing is done once; only terminal incomplete data are omitted; garbage collection occurs between animals; and static 1D geometry avoids needless time replication. Expected output is ~6.7 GB plus pickle overhead rather than ~20 GB.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 338 session-neuron instances |
| Neurons / session | 185, 153 (mean 169) |
| Subjects | 7 names retained; sample contains sessions from subject index 0 |
| Sessions / subject | 2 sample sessions for QLAK-CA1-08 |
| Trials (total) | 78 |
| Trials / session | 39, 39 |
| Timepoints / trial | exactly 600 |
| Blocked geometry input range | [0, 1] (square all 0; O arena center blocked) |
| Position class range | [0, 8] |
| Position class distribution | [0.103526, 0.102308, 0.103697, 0.061603, 0.043761, 0.117222, 0.134893, 0.103782, 0.229209] |

### Processing Plots Review
Both audit plots were visually inspected. Raw events are sparse binary transients; smoothed/pooled events retain their timing without shifts. Raw and pooled x/y trajectories overlap. Output class transitions coincide with crossings of 25/50 cm boundaries and remain aligned across the 60 s trial boundary. The square has all-open input; the O geometry has its center marked blocked with exactly 0.000 occupancy. No artifact or misalignment was observed.

`train_decoder.py --verify-only` reports: valid format, no errors, no warnings; expected 2 sessions/78 trials; all T=600; input dimension 9; output dimension 1; all nine output classes present; and CA1 neuron total 338. Manual file/shape/dtype assertions in the converter also passed. Required sample, logs, and plots exist.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Full-session vectorized float32 filtering/pooling | Two sessions processed in 1.67 s after source load; avoids per-trial convolution |
| Sequential animal loading | Bounds source memory and avoids loading six unnecessary animals in sample mode |
| 100 ms reference pooling | Approximately 3x lower output I/O/training volume than 30 Hz |

| Step | Time / Session | Estimated Total Time |
| Source loading | 9.77 s for first animal (~0.32 s amortized/session) | ~1-2 min across seven animals |
| Processing + plots | 0.84 s/session at 169 mean cells; plots add overhead | ~5.5 min scaled linearly to 337 mean cells across 207 sessions without full-mode plots |
| Pickle write | 0.03 s for 31 MiB | ~6-15 s for estimated ~6.3 GiB |
| Total conversion | 11.48 s sample | ~7-8 min, below 15-minute optimization threshold |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| position_3x3_bin | 0.4409 | 0.3702 |

Training completed on CUDA. Mean loss decreased monotonically from 2.271358 at epoch 1 to 1.716947 at epoch 200; test loss was 1.801236. Validation balanced accuracy is 3.33x uniform chance (0.1111), and all completion criteria pass. The 0.071 absolute train-validation gap (ratio 1.19) is modest.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 6.173 GiB (shown as 6.2G by `ls`)
- `verification_full_out.txt`: created

Full conversion took 194.13 s before serialization plus 5.70 s to write (3.33 minutes total), substantially below the conservative estimate and optimization threshold. Internal validation passed before writing. The external verifier reports a valid format with no errors or warnings and completed successfully.

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | 5,413 identities / 69,744 rate maps | Code processes registered finite cell/session pairs | 5,413 / 69,744 | 69,744 session-neuron rows, with source IDs retained | Yes |
| Mean neurons/session | 336.93 from totals | all registered session cells | 336.93, range 113-564 | 336.93, range 113-564 | Yes |
| Subjects | 7 | 7 hard-coded IDs | 7 | 7 | Yes |
| Sessions | 207 | all animal days | 207 | 207 | Yes |
| Trials (total) | N/A (continuous sessions) | N/A | 8,187 complete minutes | 8,187 | Yes |
| Trials/session (mean) | 40-min sessions | complete acquired frames | 39 for 93 sessions, 40 for 114 sessions; mean 39.55 | same | Yes |
| Timepoints/trial | N/A | 3-frame temporal pooling | 600 expected | exactly 600 throughout | Yes |
| Geometry inputs | 10 geometries | binary 3x3 matrices | binary blocked values | dimensions range 0..1 (one dimension is always accessible across this geometry set) | Yes |
| Position output range | 3x3 task requirement | reference uses binned x/y | 0..8 | 0..8, every class present | Yes |
| Position output distribution | Not reported | Not reported | independently derived below | [0.099663, 0.075412, 0.115696, 0.098495, 0.057266, 0.141213, 0.135121, 0.076848, 0.200287] | N/A; internally exact |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Read `/app/verification_full_out.txt` in full. It begins “Data format is valid, no errors or warnings,” reports all expected counts/ranges, and ends “Data verification complete.” There are no warnings requiring justification.
2. **Independent raw-neural check**: `/app/sanity_checks.py` loads original joblibs directly (never through conversion functions), independently selects finite source cells, casts the row to float32, calls `scipy.ndimage.gaussian_filter1d(sigma=3)`, reshapes into groups of three, and compares selected early/middle/late cells and pooled times with `np.allclose(rtol=1e-6, atol=1e-7)`. Tested day 0 and final day for every animal, including trial boundary bins 599/600. PASS.
3. **Independent raw-input check**: Builds each blocked mask directly from raw nested indices, transposes `(y,x)` to `(x,y)`, and compares first and last trials with `np.allclose`. Tested square and every non-square geometry across selected days. PASS. Checked blocked classes had exactly 0.00000000 occupancy.
4. **Independent raw-output check**: Averages raw position in independent 3-frame blocks, floors/clips each axis, constructs `3*x+y`, concatenates all converted trials, and compares whole selected sessions with `np.allclose`. PASS. Every adjacent last-bin/first-bin pair at all one-minute boundaries was separately asserted against raw indices, ruling out skipped/duplicated bins.
5. **Independent totals check**: Counts sessions, finite source cells per day, and floor(raw_frames/1800) directly from all seven joblibs, then uses `np.allclose` against converted totals. Raw and converted both give 207, 69,744, and 8,187. PASS.
6. **MATLAB/joblib loading check**: Used `h5py` to dereference sampled original `.mat` trace and position arrays and compared them to joblib values with `np.allclose` (including NaN handling). PASS, confirming reference default joblib loading preserves source values.
7. **Reference processing comparison**:

| Stage | Reference | Conversion | Result / rationale |
|-------|-----------|------------|--------------------|
| (a) Loading | `joblib.load(data/animal)[animal]` in `load_dat` | Same, sequential by animal | Exact match |
| (b) Neuron/trial filtering | Registered cells are finite; paper decoding says all registered; code additionally applies movement/activity only inside its 225-bin benchmark | All finite registered cells; complete one-minute chunks | Exact paper population count; movement selection intentionally not used because it would break contiguous trial/output timing |
| (c) Temporal alignment | Position and trace share frame indices at 30 Hz | Same raw indices, identically grouped by 3 before slicing | Exact; raw allclose and boundary tests pass |
| (d) Binning | `fit_decoder` Gaussian sigma=3 frames and non-overlap 3-frame average pooling; behavior pooled identically | Same | Exact logic; 100 ms bins |
| (e) Input construction | Named geometry represented by 3x3 masks via `get_env_mat`; `blocked` records blocked partitions | Direct blocked mask, transposed to position axis order | Same geometry with decoder-task-required blockedness semantics; zero blocked occupancy validates orientation |
| (f) Output construction | Paper/code discretize 75 cm x/y into spatial bins and one-hot/flatten positions | Required 3x3 version, class `3*x+y` | Difference mandated by task; bin edges at 0/25/50/75 cm |

8. **Key statistics**: Checked all numerical statistics in paper/methods/code/data: seven animals (4 male/3 female in paper), 5,413 longitudinal identities, 207 sessions, 69,744 registered rate maps, 10 geometries, up to 31 days, 75 cm arena, 30 Hz raw sampling, nominal 40 min recordings, 100 ms reference decoder bins, neuron range 113-564, and exact subject session counts. All applicable converted statistics match. Paper does not report 3x3 class fractions or derived one-minute count, so those were independently recomputed from raw arrays.
9. **Edge cases**: Verified source lengths 71,866/72,219/72,091/72,060/72,071; exact complete-minute counts 39/40; terminal partial frames are the only excluded samples and are recorded per session. Position coordinate 75 cm is clipped safely into bin 2. Square sentinel -1 produces no blocked inputs. Unregistered all-NaN rows are excluded, while low-activity registered cells remain. Every session has >=39 trials. No missing/nonfinite converted value exists.

### Issues Found and Resolved
- **Blocked array axes were ambiguous during exploration**: Empirical raw occupancy showed source row-major blocked indices are `(y,x)` while position/rate-map indexing is `(x,y)`. Resolution: transpose before flattening. Independent checks across the geometry set show zero occupancy in blocked output classes.
- **Nominal vs acquired duration**: Paper says 40 min, while three subjects have 39 min 55.5 s and others slightly exceed 40 min. Resolution: preserve every exact 60 s segment, drop only incomplete tails, and record dropped frames; this yields fixed trial lengths without padding or fabricating data.
- **No issues were revealed by the completed Step 10 test suite**, so no conversion/revalidation iteration was necessary. `/app/sanity_checks_out.txt` records the passing audit.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes. CUDA training completed all 200 epochs; loss 2.315921 (epoch 1), 2.238750 (10), 1.803717 (50), 1.401178 (100), 1.239697 (150), 1.158603 (200). Test loss: 1.139808.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| position_3x3_bin | 0.6965 | 0.6088 | Uniform chance 0.1111; validation is 5.48x chance |

`train_decoder.py` finished successfully without GPU fallback. The run trained on 6,531 trials and tested on 1,656. `/app/sample_trials.png` and `/app/predictions.png` were generated and visually reviewed; neural activity is sparse/smoothed as expected, geometry is static, target trajectories are piecewise spatial states, and predictions reproduce long states plus many transitions.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| position_3x3_bin | Validation balanced accuracy 0.6088; training 0.6965; uniform chance 0.1111 | Paper decodes 15x15 position and reports Euclidean error rather than categorical accuracy: overall precomputed mean 13.49 cm, Figure 1F approximately 20-22 cm early to 11-13 cm late |

**Accuracy vs chance**: 0.6088 is 5.48x chance and far exceeds the 1.5x review threshold (0.1667). The achieved classification is also substantially stronger than the two-session sample (0.3702), as expected from broader shared-decoder training and generally larger later populations.

**Accuracy comparison to paper**: The paper contains one neural position-decoding result (Figure 1F and associated text), expressed as Euclidean centimeters for 225 spatial bins, not accuracy. No paper categorical/balanced accuracy is available. The metrics therefore cannot be equated numerically. They agree qualitatively: both show strong above-chance spatial information in CA1. Our 25 cm-wide 3x3 classes and 60.88% balanced accuracy are compatible with the reference's 13.49 cm average fine-grid error, but this is an inference, not an exact conversion.

**Train-validation gap**: Absolute gap is 0.0877 and ratio is 1.144, below the specified >1.5x concern threshold. Test loss (1.1398) is slightly lower than final training loss (1.1586), providing no overfitting warning. Trials are split within each session; static geometry repeats as intended, but neural outputs remain evaluated on held-out one-minute segments.

**Leakage/alignment review**: Position is present only under `output`, never `input`; geometry inputs are static blocked masks. Independent raw tests verify exact neural/output alignment. Predictions visibly track but do not perfectly copy outputs, and the validation score is meaningfully below training, arguing against leakage.

### Issues Found and Resolved
- No output has low or below-threshold accuracy, so the conditional low-accuracy debugging protocol was not triggered.
- The paper/task metric mismatch was explicitly resolved by reporting both without inventing a paper accuracy or transforming one metric into the other.
- No Step 12 issue required conversion changes or reruns.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

`README.md` documents the dataset, loading, representation, preprocessing, key statistics, accuracy, and reproduction commands. Investigation-only `sanity_checks.py`, its output, the temporary paper rendering, and Python bytecode were moved to `cache/`; `README_CACHE.md` describes each. Required pickles, logs, processing plots, and decoder plots remain at project root. A final placeholder/status scan found no unfinished section; all Steps 0-13 are COMPLETE.
