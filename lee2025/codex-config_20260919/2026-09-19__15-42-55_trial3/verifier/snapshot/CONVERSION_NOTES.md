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
| `load_dat` | `code/georepca1/src/utils.py:61` | LOADING | Loads one animal's joblib or MATLAB dictionary; MATLAB fields `envs`, `position`, and `trace` are converted to NumPy arrays. |
| `get_env_mat` | `code/georepca1/src/utils.py:215` | PROCESSING | Maps each named arena geometry to a binary 3x3 accessible/blocked matrix. |
| `get_rate_maps` | `code/georepca1/src/utils.py:313` | PROCESSING | Spatially bins synchronized position and event traces into 15x15 event-rate maps at 30 Hz and optionally smooths by 1.5 bins. |
| `get_split_half` / `get_place_cells` / `get_shr_within` | `code/georepca1/src/utils.py:356,415,442` | CURATION | Measures within-session spatial reliability and identifies place cells against circular-shift shuffles; used for paper analyses, not as the default population filter in the paper's decoder. |
| `clean_rate_maps` | `code/georepca1/src/utils.py:463` | PROCESSING | Applies environment-specific 15x15 masks derived from the 3x3 geometry. |
| `fit_decoder` / `test_decoder` | `code/georepca1/src/utils.py:1776,1806` | PROCESSING | Gaussian naive Bayes position decoding after 3-frame pooling; Gaussian-smooths neural traces with sigma 3 frames and uses flat class priors. |
| `decode_position_within` | `code/georepca1/src/utils.py:1845` | CURATION | Within-day 5-fold decoding; retains frames above 5 cm/s (velocity smoothed with sigma 5 frames), retains cells with >5 events on retained frames, and uses 15x15 positions. |
| `generate_behav_dict` | `code/georepca1/src/utils.py:130` | LOADING | Builds a lighter dictionary of position, environment names, and map shapes. |

### Notes
- The repository README identifies seven animals: `QLAK-CA1-08`, `-30`, `-50`, `-51`, `-56`, `-74`, and `-75`; `main.py` analyzes all seven.
- Native `trace` is already a rise-extracted calcium-event series (1 = significant event); delta-F/F must **not** be recomputed. Missing/unregistered cells are represented by NaNs for a day.
- `position` and `trace` are synchronized at 30 Hz in the code and are consistently passed transposed as time x 2 and time x cells.
- `maps` are derivative analysis products. For the requested time-varying decoder dataset, the source synchronized `trace` and `position` streams should be used rather than reverse-engineering maps.
- The paper decoder's 15x15 position target differs from the requested 3x3 target; its temporal smoothing/pooling is decoder-specific and should not be baked into the converted neural values unless required after validation.
- CellReg supplies cross-day registration, but each recording day is naturally one target session; only cells valid on that day can be included because the target format does not support NaN neuron channels.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Seven extensionless compressed joblib files and matching MATLAB `.mat` files, one per animal. Conversion will use joblib, as does the reference code, because it contains the same already-converted arrays and loads substantially faster.
- Each joblib file is `{animal_id: fields}`. Fields are: `SFPs` (35x35 spatial footprints x CellReg cell x day), `centroids` (cell x 2 x day), `blocked` (blocked 3x3 partition indices per day; `-1` means none), `envs` (arena names), `maps` (`sampling`, `smoothed`, `unsmoothed` 15x15 products), `position` (day x 2 x frame), and `trace` (day x CellReg cell x frame).
- `trace` finite values are exactly `{0,1}`. A cell absent from a day is NaN for every frame; this was verified across all files. `position` is finite throughout, ranges from 0 to 75 cm, and shares exactly the same day/frame axes as `trace`.
- Frame counts are constant across days within animal: 71,866 (`08`,`30`,`50`), 72,219 (`51`), 72,091 (`56`), 72,060 (`74`), and 72,071 (`75`). At 30 Hz these are 39.93--40.12 min recordings.
- Six animals have 31 recording days in three 10-geometry sequences plus a final square; `QLAK-CA1-51` has 21 days in two sequences plus a final square. Total recording days/sessions = 207.
- No native trial hierarchy exists. Under the requested non-overlapping 1-minute segmentation (1,800 frames), the data yield 8,187 complete trials: 39/day for animals `08`,`30`,`50`, and 40/day for the others. Only the incomplete tail is excluded.
- `/app/data/.fetch_complete` is the supplied Zenodo download manifest (record 14867736). There is no separate data README. `behav_dict` and `precomputed_results/` are derived/reference products, not alternate raw streams.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 5,413 unique within-animal CellReg identities; 69,744 recorded session-neuron instances |
| Neurons / session | mean 336.93; range 113--564 |
| Subjects | 7 |
| Sessions / subject | 31 for 6 subjects; 21 for `QLAK-CA1-51` (207 total) |
| Trials (total) | Native: none (continuous sessions); requested 1-min segmentation: 8,187 complete trials |
| Trials / session | 39 for 93 sessions; 40 for 114 sessions |

Per-animal CellReg identities: `08` 515, `30` 875, `50` 942, `51` 554, `56` 862, `74` 713, `75` 952. Per-animal summed recorded session-neuron instances: 6,631; 11,812; 12,427; 4,831; 11,807; 9,680; 12,556, respectively. Of 69,744 instances, 69,632 have >5 events across the whole day (the reference decoder applies its >5 criterion after running-frame selection, which is examined later).

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 5,413 unique; 69,744 session rate maps | methods.txt: “5,413 unique neurons across 207 sessions ... forming 69,744 rate maps” |
| Neurons / session | 69,744 / 207 = 336.93 mean session-neuron instances | Derived directly from reported totals |
| Subjects | 7 (4 male, 3 female) | STAR Methods: “Naive male (4) and female (3) mice” |
| Sessions / subject | Up to 31; 207 total | Paper: sequences repeated “up to three times”; methods: one session/day |
| Trials (total) | N/A natively; continuous 40-min sessions | methods.txt: “All sessions were 40 min” |
| Trials / session | N/A natively; requested conversion creates 1-min trials | Decoder-task requirement |
| Neural data time bin | 1/30 s = 33.333 ms | methods.txt: imaging acquired at 30 Hz |
| Behavior data time bin | 1/30 s = 33.333 ms, simultaneously acquired and timestamp-aligned | methods.txt: behavioral and cellular streams simultaneously acquired at 30 Hz |
| Reward rate | N/A | Mice freely explored; no reward outcome/task is described |
| Arena size | 75 x 75 cm, imagined 3x3 grid | methods.txt |
| Conditions | 10 geometries; each partition 25 x 25 cm | paper Figures 1 and partition analysis |
| Session duration | 40 min, one/day | methods.txt |


### Processing Details
- Behavioral video and miniscope streams were acquired simultaneously at 30 Hz and every frame timestamped for post-hoc alignment. The released `position` and `trace` arrays are already aligned on an identical frame axis.
- Imaging preprocessing corrected motion, segmented cells/extracted calcium traces, took the derivative of the median-subtracted trace, Gaussian-smoothed it with sigma 5 frames, noise-z-scored it using a half-normal estimate, and binarized values above 2.5. The resulting rising-phase vector was treated as firing rate in every analysis.
- Head position was obtained with DeepLabCut. Cross-session tracking used brain-surface landmarks, spatial footprints, and/or centroids.
- Reference rate maps used 5x5 cm spatial pixels (15x15 over the 75-cm arena), a 5-cm Gaussian smoothing kernel, and 30-Hz event counts divided by occupancy.
- Reference Bayesian position decoding used a within-session 5-fold split, one-hot spatial position, Gaussian Naive Bayes with flat priors/default `var_smoothing=1e-9`, and Euclidean spatial error. Reference code additionally selects speed >5 cm/s (velocity sigma 5 frames), cells with >5 retained-frame events, smooths neural traces sigma 3 frames, and pools 3 frames.
- The paper reports a significant decline in mean position-decoding error across days (ANOVA p<0.0001, F=7.9845). Figure 1F is an error curve, not categorical accuracy: visually it falls from roughly 22 cm to roughly 11 cm. Thus it is not numerically comparable to the requested 9-class balanced accuracy, but it supplies a qualitative above-chance expectation.

### Curation Steps

**Neuron curation rules**:
Motion-corrected videos were manually inspected; spatial footprints were manually verified to remove lens artifacts. The paper states that high reliability motivated inclusion of all cells in subsequent analyses (mean unique cells/animal 773 ± 68 SE, range 515--952). Place-cell status is an analysis label, not a general exclusion rule. Missing/unregistered day-cell combinations are NaN and cannot be neuron channels for that day.

**Trial curation rules**:
No trial curation was reported because sessions were continuous free exploration. For the reference decoder only, frames were restricted by the code to smoothed movement speed >5 cm/s. The downstream task explicitly requires complete fixed one-minute trials, so whether frame deletion is compatible is resolved in Step 5.

### Decoders Trained
| Decoded variable | Accuracy |
| 15x15 within-session position | Paper reports Euclidean error, approximately 22 cm early to 11 cm late in Figure 1F; no categorical accuracy reported |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Dataset size | Seven named animals in `main.py` | 7 animals, 207 days, 5,413 CellReg IDs, 69,744 valid day-cell instances | Exactly 5,413 unique neurons, 207 sessions, 69,744 rate maps | Exact agreement. |
| Acquisition/alignment | Position and trace are transposed together and decoded at `fps=30` | Identical day/frame dimensions, finite position, 0/1-or-NaN traces | Simultaneous, timestamped 30-Hz streams | Exact agreement; no resampling or inferred offset is needed. |
| Session length | Assumes continuous arrays | 71,866--72,219 frames (39.93--40.12 min) | 40 min | Small acquisition-length variation is normal. Form complete 1,800-frame trials from frame zero and exclude only the final incomplete tail. |
| Neural representation | `trace` is used directly | Finite trace values exactly 0 or 1 | Binarized rising phase (>2.5 noise-z), treated as firing rate | Exact agreement; no delta-F/F or deconvolution. |
| Cell inclusion | Reference position decoder dynamically retains valid cells with >5 events during speed-selected frames; other analyses include registered cells | NaN denotes an unregistered day-cell; 69,632/69,744 valid instances exceed 5 events over the full session | Manual quality curation already done; high reliability motivated inclusion of all cells | No contradiction: >5 is decoder feature selection, not upstream imaging QC. For conversion, preserve all valid manually curated cells; fixed trials must not have a neuron axis that changes by trial. |
| Frame inclusion | Position decoder selects smoothed speed >5 cm/s | Full aligned streams supplied | Free exploration; no general frame exclusion described | Speed filtering is specific to the paper's decoder. Deleting frames would destroy contiguous one-minute trials, so retain all frames for the requested format. |
| Spatial output | Reference decoder uses 15x15 bins and 3-frame pooled positions | Position is 0--75 cm; empirical blocked-bin consistency shows native coordinates need transpose into canonical geometry order | Rate-map pixels are 5x5 cm; conceptual arena is 3x3 partitions ordered by rows | Requested task overrides to 3x3. Use 25-cm bins and flatten as `y_bin*3+x_bin` so output classes share the source `blocked` partition order. |
| Geometry | `get_env_mat` gives accessible-bin matrices and applies plot/map orientation transforms elsewhere | `blocked` gives a consistent canonical blocked-index set per named geometry for all animals | Partitions ordered West-to-East, North-to-South | Use `blocked` directly to create 9 binary “is blocked” inputs; avoids plotting-orientation transforms. |
| Rate-map smoothing | `get_rate_maps` default sigma 1.5 bins (7.5 cm) | Supplied unsmoothed maps exactly reproduce trace/position binning when the single coordinate at 75 cm is excluded; stored maps are events/frame | STAR Methods says 5-cm sigma; repository README says 2.5-cm kernel | The sources disagree about a derived product. This conversion does not use rate maps, so the discrepancy cannot affect neural/time alignment or 3x3 labels. |
| Position edge at 75 cm | Buffer intends to keep maxima in bounds, but floating behavior is fragile | A rare exact `(75,75)` sample is omitted from stored 15x15 maps | Arena includes the 75-cm boundary | For categorical output, clip `floor(position/25)` into 0--2. The physical boundary belongs to the outer bin; retaining it preserves aligned frames. |
| Published decoding | Code returns per-day 5-fold Euclidean errors | Precomputed result: grand mean 13.49 cm; nominal day mean declines from 22.77 cm (day 1) to 11.50 cm (day 31) | Figure 1F visually shows ~22 to ~11 cm and significant decline | Exact agreement between figure and precomputed output. Requested validation reports balanced class accuracy, so only qualitative comparison is possible. |

Final understanding: each recording day is one decoder session with a single geometry and a valid-cell population; synchronized binary calcium events and x-y positions are retained at 30 Hz, split into contiguous complete one-minute trials, geometry is static per trial, and position is mapped to one 9-class time-varying output. All seven animals and all 207 days are included.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `trace[day, valid_cell, frame]` | `neural[session][trial]` | Select cells finite on that day; per 1-min trial Gaussian-smooth at sigma 3 source frames and non-overlapping mean-pool 3 frames; transpose/retain as float32 `(neurons,600)` | `fit_decoder` | Matches the reference decoder's 3-frame smoothing/pooling while preventing smoothing leakage across held-out trial boundaries. No dF/F recomputation. |
| `blocked[day]` | `input[session][trial][0:9]` | Nine float32 binary flags in canonical source partition order; 1=blocked, 0=accessible; repeated as a static `(9,)` value for each trial | README `blocked`; `get_env_mat` cross-check | Meets explicit static geometry-input task. Names are `blocked_partition_0` ... `_8`. |
| `position[day,:,frame]` | `output[session][trial][0,:]` | Mean-pool x and y over the same 3-frame windows; `clip(floor(coord/25),0,2)`; flatten in canonical geometry order as `y_bin*3+x_bin`; uint8 `(1,600)` | `fit_decoder`, requested 3x3 override | Class names `y0_x0` ... `y2_x2`; output is time-varying, categorical, and aligned to `blocked` indices. |
| Animal ID | `subjects`, `subject_idx` | Seven fixed IDs; one subject index for each day-session | `main.py` animals | Session ordering is animal-list order, then zero-based day. |
| Recording region | `brain_regions`, `brain_region_idx` | `['CA1']`; zeros for every retained neuron | Paper/methods | All recordings target dorsal hippocampal CA1. |
| `envs[day]`, day/frame details, valid source-cell indices | `metadata['session_info']` | One dict/session with animal, source day, arena name, blocked indices, source/used/discarded frames, trials, and source neuron indices | `load_dat` / native fields | Retains provenance needed for independent spot checks. |
| `SFPs`, `centroids` | Not copied | Imaging morphology/registration provenance only | `trace_sfps` | Not a decoder input/output and would greatly duplicate data; registration result is represented by valid source indices. |
| `maps` | Not copied | Derived spatial summaries | `get_rate_maps` | Avoid circularity: decode framewise position from synchronized trace, not a trace-derived rate map. |
| `envs` string | Metadata only | Preserve exact arena name | — | Geometry itself is represented numerically by `blocked`. |

### Key Decisions
1. **Session unit**: Each source animal-day is one target session. Neuron identity is stable across all trials within that day, while NaN registration varies across days.
2. **Trial boundaries**: Starting at source frame 0, take consecutive 1,800-frame (60-s) blocks. Exclude only a final incomplete block; never pad or overlap. All resulting trials contain 600 bins of 100 ms.
3. **Temporal processing**: Match the paper decoder's `temporal_bin_size=3` and neural Gaussian sigma 3 frames. Process trials independently so random train/validation trial splitting cannot leak neural samples across a boundary. Mean position coordinates before spatial discretization, keeping neural and output windows identical.
4. **Cell curation**: Retain every cell finite for the entire source day. These cells already passed motion/manual footprint QC. Do not place-cell-filter or activity-filter the stored dataset: the paper motivates all-cell inclusion, and the code's >5-event selection is an internal decoder feature-selection step tied to its deletion of low-speed frames.
5. **No speed-frame deletion**: Complete one-minute trials and fixed time bins are explicit downstream requirements. Removing scattered low-speed frames would turn elapsed time into irregular samples and violate the trial definition.
6. **Geometry encoding**: Use nine blocked/not-blocked flags directly from `blocked`, rather than a 10-condition one-hot, because the requested input is compositional arena geometry and the source explicitly defines index order.
7. **Spatial encoding**: Use one categorical output with nine values, not two three-class outputs, because the task says 3x3 = 9 spatial bins. Native position is `(x,y)`, while blocked partitions are canonical matrix rows/columns; exhaustive testing of all eight axis/flip transforms across 207 sessions identifies `(row,column)=(y,x)` (only 0.0031% pooled samples in blocked bins, versus 13.6--21.7% for alternatives). Therefore flatten as `y*3+x`.
8. **Boundary handling**: Clip exact 75-cm coordinates to bin 2. They are valid physical edge samples; dropping them would desynchronize streams.
9. **Data types**: float32 neural and input arrays avoid validator warnings and preserve the reference decoder's continuous smoothed activity; uint8 output is exact and compact.
10. **Sample selection**: `--sample` converts the first two day-sessions (`QLAK-CA1-08`, days 0 and 1), which provides square and center-blocked geometries with the same subject but independently curated neuron populations.
11. **Metadata offsets**: Trials align to their own segment start; `off_start=0.0`, `off_end=60.0`, `time_bin_size=100.0` ms, and per-trial start times are recorded in `session_info` by the deterministic trial index.

All available experimental variables were assessed: neural events, x-y position, arena name, blocked partitions, footprints, centroids, occupancy/event maps, animal, day, and CA1 region. Only synchronized events, geometry, position, and provenance are relevant to this decoder target.

### Planned Sanity Checks
- [x] Native count check: assert 7 animals, 207 sessions, 5,413 CellReg IDs, 69,744 valid day-cell instances, and 8,187 complete source-minute trials.
- [x] Neural check: independently load raw joblib data and use SciPy smoothing plus reshape/mean on at least three named session/trial/cell slices; require `np.allclose` to converted neural values.
- [x] Input check: independently reconstruct each 9-vector from raw `blocked` for at least three sessions; require `np.allclose` to converted trial input.
- [x] Output check: independently pool raw position and calculate 3x3 labels for at least three trials including first/last boundaries; require `np.allclose` to converted output.
- [x] Alignment check: plot raw/pool coordinates, spatial class, and neural activity with source and pooled time axes for two sessions; verify exact 3:1 correspondence and no shifts.
- [x] Geometry check: ensure every session's position classes have zero/negligible occupancy in blocked partitions after accounting for coordinate/display orientation; investigate exceptions rather than assuming orientation.
- [x] Shape/value check: every trial is neural `(N,600)`, input `(9,)`, output `(1,600)`; neural/input finite, inputs binary, outputs integer 0--8, and every session has 39 or 40 trials.
- [x] Provenance check: verify session ordering, `subject_idx`, region-vector length, and source neuron indices against raw arrays.
- [x] Published-result check: distinguish requested balanced 9-class accuracy from the paper's 15x15 Euclidean error; require above 1/9 balanced chance and investigate below 1/6 (1.5x chance). Final full validation is 0.6078, or 5.47x chance.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with mutually exclusive `--full` / `--sample` modes (full is default) and `--show-processing`. The script loads reference joblib data, converts animal-days in deterministic order, asserts synchronized dimensions and native full counts, constructs all target fields, stores JSON-compatible provenance, reports per-session/load/write timing, and writes with the highest pickle protocol. Syntax compilation and CLI help completed without errors.

Code inefficiencies identified:
Naive per-neuron/per-frame Python loops and per-trial SciPy calls would add substantial overhead; holding derivative `SFPs` and `maps` while processing would waste many GiB; converting binary arrays to float64 would double output/working memory.

Code speedups added:
One vectorized SciPy call filters `(neuron, trial, frame)` batches per session while preserving independent trial boundaries; reshape/mean performs pooling; arrays are float32; large unused native fields are released immediately after extracting trace/position/geometry; each animal is loaded only once; output labels and geometry are compact; garbage collection occurs between animals.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 338 session-neuron instances |
| Neurons / session | 185, 153 (mean 169) |
| Subjects | 7 declared; 1 represented in sample |
| Sessions / subject | 2 for `QLAK-CA1-08` |
| Trials (total) | 78 |
| Trials / session | 39, 39 |
| Geometry input ranges | partition 4 `[0,1]`; all others `[0,0]` in this two-geometry sample |
| Neural range / means | `[0,1]`; means 0.005413 and 0.006126 |
| Position class distribution | `[0.104,0.062,0.135,0.102,0.044,0.104,0.104,0.117,0.229]` |
| Output range | `[0,8]` |

### Processing Plots Review
`processing_QLAK-CA1-08_day00.png` and `processing_QLAK-CA1-08_day01.png` show raw trajectories with 25-cm boundaries, binary events, sigma-3 smoothing, aligned three-frame pooled traces/positions, categorical labels, exact trial coverage, discarded tail, and geometry. The square input is all zero; the `o` input has only partition 4 blocked, and class 4 has exactly zero occupancy in that session. Raw and pooled positions overlay without a shift. No anomaly was found.

Manual pickle inspection confirmed required keys, float32 `(N,600)` neural arrays, float32 static `(9,)` inputs, uint8 `(1,600)` outputs, 1,666 source tail frames excluded per sample session, finite values, stable neurons within session, and complete metadata. `/app/train_decoder.py --verify-only` reported: “Data format is valid, no errors or warnings.”

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Batched `(neuron,trial,frame)` Gaussian filter and reshape pooling | Avoids 78 per-trial SciPy calls in sample and 8,187 in full conversion |
| Release unused maps/SFPs after joblib load | Reduces resident memory before processing |
| float32 processing/output | Halves neural storage and filter traffic versus float64 |

| Step | Time / Session | Estimated Total Time |
| Native joblib load (sample animal) | 9.02 s / animal (amortized 0.29 s/session) | Prior all-animal exploration loads totaled ~137 s |
| Conversion plus processing plot | 0.96 s, 0.84 s (plots dominate) | No plots in full run; conservative <90 s conversion work |
| Sample write | 0.03 s for 0.030 GiB | Full expected ~6.2 GiB; estimated <60 s |
| Total | Sample 10.89 s | Conservative full estimate ~4.8 min, well below 15 min |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| `position_3x3` | 0.4489 | 0.3914 |

Training completed on CUDA. Loss decreased monotonically from 2.280069 (epoch 1) to 1.718018 (epoch 200); test loss was 1.802889. Validation balanced accuracy is 3.52x uniform chance (0.1111), so the sample comfortably meets the above-chance criterion.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 6,627,692,973 bytes (6.173 GiB)
- `verification_full_out.txt`: created; valid with no errors or warnings

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | 5,413 identities / 69,744 rate maps | All registered cells by day | 5,413 / 69,744 | 69,744 session-neuron instances with source indices | Yes |
| Mean neurons/session | 69,744/207 = 336.93 | Valid registered cells | 336.93 (range 113--564) | 336.93 (range 113--564) | Yes |
| Subjects | 7 | 7 named animals | 7 files | 7 | Yes |
| Sessions | 207 | All animal-days | 207 | 207 | Yes |
| Trials (total) | N/A (continuous 40-min sessions) | N/A | 8,187 complete requested minutes | 8,187 | Yes (task-derived) |
| Trials/session (mean) | N/A | N/A | 39 for 93 sessions; 40 for 114 | identical | Yes |
| Geometry inputs | 10 3x3 geometries | binary `get_env_mat` / source `blocked` | each source blocked index represented; index 7 never blocked | per-partition ranges `[0,1]`, except index 7 `[0,0]` | Yes |
| Position range | 75x75 cm / 9 partitions | binned position | 0--75 cm | classes 0--8 | Yes |
| Position distribution | not numerically reported | not reported for 3x3 task | independently derived from source | `[0.100,0.098,0.135,0.075,0.057,0.077,0.116,0.141,0.200]` | Yes |

The corrected full conversion took 167.71 s (load, processing, and 5.87-s write). Spot checks at sessions 0, 30, 31, 92, 93, 113, and 206 confirmed animal/day ordering, geometry, neuron counts, shapes, and source provenance. All 600-bin trials are finite.

Iteration 1 found and fixed: an explicit geometry consistency test found 17.14% of initially encoded output samples apparently occupying blocked source indices. Testing all eight transpose/flip mappings showed the issue was an axis-order mismatch: native `(x,y)` must map to canonical blocked matrix `(row,column)=(y,x)`. Alternatives left 13.55--21.70% blocked occupancy; transpose leaves 152/4,912,200 (0.00309%) samples, confined to four sessions and attributable to isolated pose/pooling boundary samples. `convert_data.py`, sample/full pickles, all validation logs, plots, and sample training were regenerated. Corrected sample validation balanced accuracy is 0.3914 (3.52x chance).

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output-log verification**: Read `verification_full_out.txt` and searched the full/sample conversion, verification, and sample-training logs for errors, warnings, invalid values, failures, or tracebacks. Both verifier runs say “Data format is valid, no errors or warnings”; conversion and training finished successfully. There are no validator warnings to waive.
2. **Independent raw-data `np.allclose` checks**: Loaded the original joblib files directly, without calling `convert_data.py`. Independently reconstructed neural smoothing/pooling, geometry, position classes, and retained-cell provenance for `(session, day, trial, converted-cell row)` `(0,0,5,3)`, `(102,9,20,7)`, and `(206,30,39,11)`. Every neural, input, output, and source-index comparison passed `np.allclose`; neural maximum absolute difference was exactly 0 and all three output mismatch counts were 0. These cover three animals, an intermediate session, and the last trial of the last session.
3. **Reference-code comparison**:
   - **Loading**: Reference `load_dat` and this script both use the animal joblib dictionary's `trace`, `position`, `blocked`, and `envs`; no derivative maps are treated as raw data.
   - **Neuron/trial filtering**: Both recognize NaN day-cell entries as absent. The conversion retains every finite, manually curated day-cell. It deliberately does not apply `decode_position_within`'s speed >5 cm/s and >5 retained-event feature selection because those are model-specific, would delete noncontiguous timepoints, and conflict with complete elapsed one-minute trials. No native trials exist; only the incomplete session tail is excluded.
   - **Temporal alignment**: Reference functions index position and traces with the same frame mask. Conversion uses their identical 30-Hz frame axes and the same three-frame windows, with no inferred offset or resampling.
   - **Binning**: Reference `fit_decoder`/`test_decoder` Gaussian-smooth traces with sigma 3 frames and apply non-overlapping `AvgPool1d(3)`. Conversion matches this numerically using SciPy plus reshape/mean. It applies smoothing independently inside trials to avoid train/validation leakage. Position is likewise three-frame mean-pooled.
   - **Input construction**: The reference position decoder has no geometry covariate. The explicit downstream task adds it; the conversion maps raw `blocked` indices directly to nine static binary flags and cross-checks them against `get_env_mat`.
   - **Output construction**: Reference code mean-pools continuous position then encodes 15x15 bins. The requested task overrides only spatial resolution to 3x3. Conversion pools first, clips the valid 75-cm edge, and encodes `y*3+x`, which is the orientation empirically consistent with raw blocked indices.
4. **Key-statistics comparison**: Raw, paper, code, and converted counts agree at 7 animals, 207 sessions, 5,413 cross-day CellReg identities, 69,744 day-cell instances, and mean 336.93 neurons/session (113--564). The task-derived dataset has exactly 8,187 trials: 93 sessions with 39 and 114 with 40. Validator ranges are binary inputs, output 0--8, and 600 bins/trial. Global output fractions are `[0.100,0.098,0.135,0.075,0.057,0.077,0.116,0.141,0.200]`; all classes have substantial support.
5. **Geometry/orientation audit**: Exhaustively compared all eight coordinate transpose/flip transforms. `y*3+x` uniquely reduces blocked occupancy to 152/4,912,200 pooled samples (0.003094%), versus 13.55--21.70% for alternatives. A second direct raw audit found 445/14,736,600 source frames (0.003020%) in blocked bins. Of the 152 pooled exceptions, 150 have blocked raw-frame support and two arise when averaging three boundary positions; they are source pose/boundary observations, not a conversion shift. Editing or deleting them would modify synchronized behavior without a supplied validity flag, so they are retained.
6. **Edge cases/off-by-one audit**: Trial zero starts at source frame 0 and each trial covers exactly `[1800*i,1800*(i+1))`; the independent final check covers session 206 trial 39 through frame 71,999. Used frames are 70,200 for 39-trial sessions and 72,000 for 40-trial sessions. Expected tails are 1,666 frames for animals `08`,`30`,`50`, 219 for `51`, 91 for `56`, 60 for `74`, and 71 for `75`. Exact coordinate 75 is clipped into outer bin 2 rather than lost. Session order, subject indices, region-vector lengths, trial shapes, finiteness, and stable within-session neuron axes all pass script assertions and the validator.

### Issues Found and Resolved
- **Initial x/y flattening mismatch**: The first conversion used the wrong axis order, producing 17.14% apparent blocked occupancy. Changed output construction to `y*3+x`, regenerated sample/full datasets and every affected validation/training log, and reran all Step 10 checks. Corrected occupancy is 0.003094% and independent raw comparisons pass exactly.
- **Residual blocked observations**: Investigated every raw and pooled exception. They are present in the supplied pose stream or result from valid three-frame boundary averaging, not lost alignment; no transformation, curation, or imputation is warranted.
- **Review conclusion**: No unresolved errors or warnings remain, and no additional conversion change is indicated by the repeated checks.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes; 2.318670 at epoch 1 to 1.156360 at epoch 200. Test loss was 1.139768.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| `position_3x3` | 0.6982 | 0.6078 | Validation is 5.47x uniform chance (0.1111); complete CUDA run on 6,531 train and 1,656 validation trials. |

The required command completed with exit code 0 and wrote `train_decoder_full_out.txt`. `--plot-samples` also generated the sample/prediction diagnostics. No GPU-memory fallback or runtime warning occurred.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
|---|---:|---|
| `position_3x3` | Validation balanced accuracy 0.6078 (training 0.6982) | The paper reports a different target/metric: 15x15 within-session Euclidean error, grand mean 13.49 cm in the supplied result, declining from 22.77 cm on nominal day 1 to 11.50 cm on day 31. It reports no 3x3 categorical accuracy. Both results demonstrate strong position information. |

**Accuracy versus chance**: Uniform nine-class chance is 0.1111 and the requested 1.5x diagnostic threshold is 0.1667. Validation accuracy 0.6078 is 5.47x chance and 3.65x the diagnostic threshold. Every global class has 5.7--20.0% support, and balanced accuracy prevents the largest class from dominating the score.

**Paper comparison**: The paper's only decoding of the same variable is Bayesian position decoding in Figure 1F and STAR Methods. It reports Euclidean distance for 15x15 bins, not categorical accuracy; supplied `precomputed_results/within_decoding` exactly supports a 13.4906-cm grand mean, with the nominal-day means above. The paper also analyzes RSM prediction and unsuccessful animal-identity decoding (animal-identity shuffle chi-square p=1.000), but those use population similarity matrices and are not neural-to-position decoders, so they are not comparison targets for this output. There is no evidence that the converted decoder underperforms the applicable published result.

**Train-validation gap**: The ratio is 0.6982/0.6078 = 1.149, below the 1.5 overfitting trigger; the absolute gap is 0.0904. Test loss (1.139768) is slightly lower than the final training objective (1.156360). The trial-level random split and trial-independent smoothing prevent boundary leakage.

**Required low-accuracy diagnostics**: Although accuracy is not low, all prescribed checks were completed. Three independently selected raw trials—including the final trial—have exact output equality and exact neural alignment; processing plots show synchronized three-frame windows; class proportions are non-degenerate; native binary-event/manual-QC processing was preserved; and each major conversion step was compared with the reference code in Step 10. Visual review of `sample_trials.png` and `predictions.png` shows static geometry, plausible sparse smoothed events, time-varying labels, and substantial prediction/target agreement without a systematic time shift.

### Issues Found and Resolved
- No new conversion issue was found. The earlier axis-order issue remained fixed under full training and all repeated checks.
- The metric mismatch with the paper cannot be removed without violating the explicitly requested 3x3 categorical task; it is documented rather than misrepresented as a direct accuracy comparison.
- Review conclusion: accuracy, loss, class support, generalization gap, raw alignment, and reference consistency all pass; no rerun is required.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created with `README_CACHE.md`
- [x] All files organized

`README.md` documents the dataset, shapes, class encoding, loading example, processing, reproduction commands, statistics, and final accuracy. The temporary paper-page raster and Python bytecode were moved under `cache/`; required datasets, scripts, logs, processing plots, and decoder plots remain in the project root. A final nonempty-file check found every required deliverable, and a status/placeholder search found no incomplete workflow section.
