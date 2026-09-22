# Dataset Conversion Notes

## Overview
- **Dataset**: CA1 recordings from "Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping"
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment verification:
- Python 3.13.15
- NumPy 2.4.4
- PyTorch 2.6.0+cu124
- CUDA available: True
- `/app/CONVERSION_NOTES.md` existence verified with `ls -la`.

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

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_dat` | `code/georepca1/src/utils.py:61` | LOADING | Load one animal from joblib or MATLAB and normalize selected MATLAB cell-array fields to Python lists. |
| `mat2joblib` / `save_dat` | `utils.py:87,100` | LOADING | Convert/save native MATLAB data as compressed joblib. |
| `generate_behav_dict` | `utils.py:130` | PROCESSING | Collect position, environment labels, and map shape across animals. |
| `get_env_mat` | `utils.py:215` | PROCESSING | Convert named geometry to a binary 3×3 matrix (0 omitted/blocked, 1 accessible). |
| `get_transition_matrix` | `utils.py:271` | PROCESSING | Build occupancy/transition counts from behavior in 15×15 spatial bins. |
| `get_rate_maps` | `utils.py:313` | PROCESSING | Compute occupancy-normalized event-rate maps from framewise position and binary event traces; defaults: 15 bins, 30 fps, Gaussian sigma 1.5 bins. |
| `get_split_half` / `get_shuffle_split_half` | `utils.py:356,393` | CURATION | Quantify within-session spatial reliability and shuffled null distributions. |
| `get_place_cells` | `utils.py:415` | CURATION | Select significant spatial cells against shuffle null (`alpha=0.05`). |
| `clean_rate_maps` / `get_masked_maps` | `utils.py:463,482` | PROCESSING | Mask invalid geometry/occupancy locations in spatial maps. |
| `fit_decoder` / `test_decoder` | `utils.py:1776,1806` | PROCESSING | Fit/test Gaussian Naive Bayes position decoder after temporal binning. |
| `decode_position_within` | `utils.py:1845` | PROCESSING | Decode position within each recording day using cross-validation and spatial-error metrics. |

### Notes
- Repository corresponds to Lee, Keinath, Cianfarano & Brandon (Neuron, 2025) and lists seven animals: QLAK-CA1-08, -30, -50, -51, -56, -74, and -75.
- Native per-animal fields documented in README: `SFPs`, `blocked`, `centroids`, `envs`, `maps` (`sampling`, `smoothed`, `unsmoothed`), `position`, and `trace`.
- `trace` is already a rise-extracted binary calcium event series (1 = significant event), not raw fluorescence. Therefore delta-F/F must not be recomputed for this conversion.
- `position` is framewise x-y behavior per recording day; `trace` is aligned by frame/day. Unregistered cells are represented by NaNs on that day.
- Reference processing assumes 30 Hz acquisition. Spatial analyses use 15×15 maps and occupancy-normalized event rates, with a Gaussian smoothing default of 1.5 bins (README describes the released smoothed maps as 2.5 cm Gaussian kernel).
- `blocked` stores omitted partition indices using row-major 3×3 labels `[[0,1,2],[3,4,5],[6,7,8]]`; -1 means no omitted partition. `get_env_mat` provides the corresponding binary accessible/blocked geometry.
- Place-cell and split-half filters are available for specific analyses. The full trace is the direct neural stream; whether analysis-specific place-cell curation applies to the target conversion will be reconciled with methods/data in Steps 3–5.
- The paper's decoder code bins neural/behavioral samples temporally and uses Gaussian Naive Bayes. The target decoder differs, but temporal alignment must preserve the same frame correspondence.
- No reference function computes raw fluorescence preprocessing; that upstream pipeline has already produced released binary traces.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` is ~15 GB. It contains seven compressed joblib animal files (preferred by reference `load_dat`), duplicate MATLAB v7.3 files, `behav_dict`, and precomputed analysis/model results.
- Each joblib file is `{animal_id: animal_data}`. Fields are:
  - `SFPs`: 35×35×registered_cells×days spatial footprints.
  - `centroids`: registered_cells×2×days ROI centroids.
  - `blocked`: ragged list per day of blocked 3×3 partition indices; `[-1]` means none.
  - `envs`: one named geometry per day.
  - `maps`: `sampling` (15×15×days), `smoothed` and `unsmoothed` (15×15×registered_cells×days).
  - `position`: object array/list of 2×frames float arrays, one per day.
  - `trace`: object array/list of registered_cells×frames float arrays, one per day.
- Finite traces contain only 0 and 1. A registered identity absent from a day has an entirely NaN trace row; no partially finite cell rows were observed.
- Position is finite, in the range 0–75 cm on each axis. Neural and position frame counts match exactly for every session.
- Per-animal frame counts are constant across days: 71,866 (08/30/50), 72,219 (51), 72,091 (56), 72,060 (74), and 72,071 (75), approximately 40 minutes at 30 Hz.
- `behav_dict` independently confirms animal/day counts, geometry sequences, position shapes, and map dimensions.
- No README exists in `/app/data`; field documentation is in the reference repository README.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total longitudinal identities) | 5,413 |
| Neurons / session (finite rows) | 113–564; mean 336.93 (69,744 / 207) |
| Subjects | 7 |
| Sessions / subject | 31 for six subjects; 21 for QLAK-CA1-51 |
| Sessions (total) | 207 |
| Trials (total) | Not native; recordings are continuous ~40-minute sessions and will be split into one-minute target trials |
| Trials / session | Not native; expected ~39–40 complete one-minute windows depending binning/end handling |
| Total source frames | 14,903,019 |
| Longitudinal identities / animal | 515, 875, 942, 554, 862, 713, 952 |
| Session-neurons / animal | 6,631; 11,812; 12,427; 4,831; 11,807; 9,680; 12,556 |
| Event probability / finite cell-frame | approximately 0.00705–0.00912 across animals |

### Data Quality and Alignment Checks
- All 207 position/trace pairs have identical frame counts.
- All position samples are finite.
- Every finite neural sample is exactly binary (0 or 1).
- Missing cell registrations occur only as whole-row NaNs and can be removed session-wise without dropping timepoints.
- Geometry names and blocked-index patterns agree (e.g. square→`[-1]`, o→`[4]`) in inspected sequences.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (session-specific rate maps) | 69,744 | Paper overview: “all recorded sessions in 10 geometries, forming 69,744 rate maps” |
| Neurons / animal | mean 773 ± 68 SE; min 515; max 952 | Results lines 193–195 |
| Subjects | 7 (4 male, 3 female C57Bl/6) | Experimental Model and Subject Details |
| Sessions / subject | up to 31 (three 10-geometry sequences bounded by square); one released animal has 21 | Results design and source files |
| Sessions (total) | 207 in released data | Derived from released per-animal day counts and consistent with 69,744 paper rate maps |
| Trials (native) | continuous sessions, no trial structure | Data acquisition |
| Session duration | 40 min, one session/day | “All sessions were 40 min, and one session was recorded per day” |
| Neural data time bin | 30 Hz raw aligned frames; reference decoder pools 3 frames = 100 ms | Acquisition and `fit_decoder` defaults |
| Behavior data time bin | 30 Hz raw aligned frames; reference decoder pools 3 frames = 100 ms | Acquisition and `fit_decoder` defaults |
| Arena | 75×75 cm, partitioned as a 3×3 grid | Results/Data acquisition |
| Conditions | 10 geometries, sequence repeated up to 3 times | Results lines 168–180 |
| Event extraction | binary rise vector; derivative smoothed sigma 5 frames; z-score threshold >2.5 | Data preprocessing |
| Place-cell criterion | split-half correlation >99th percentile of 1,000 circular shuffles | Place cell identification |
| Position decoder | 5-fold Gaussian Naive Bayes, flat prior, Euclidean error | Bayesian decoding methods |

### Processing Details
- Behavioral and calcium imaging streams were acquired simultaneously at 30 Hz and all frames timestamped for post-hoc alignment.
- Motion correction, CNMF-E segmentation/transient extraction, and manual quality inspection were performed upstream.
- Released neural traces are median-subtracted derivative/rising-phase events already thresholded at z>2.5 and binarized. The paper treats this binary vector as firing rate in all analyses.
- Position is head location from DeepLabCut.
- Reference rate maps use occupancy-normalized events. Reference code defaults to 15×15 spatial bins and 30 fps.
- Reference within-session decoder discretizes to 15×15, smooths neural traces with Gaussian sigma 3 frames, then average-pools behavior and neural data in non-overlapping 3-frame (100 ms) bins. It uses five folds, speed >5 cm/s, and cells with >5 events for that specific analysis.
- Paper decoder performance is reported as Euclidean error in cm, not 9-class categorical accuracy. Error decreased across sessions (ANOVA p<0.0001, F=7.9845), reaching the maximum accuracy reported in cited recent work; no directly comparable categorical percentage is reported.

### Curation Steps

**Neuron curation rules**:
- Upstream: motion-corrected videos manually inspected; spatial footprints manually verified to remove lens artifacts; cells tracked across sessions by landmarks, footprints, and centroids.
- Paper Results explicitly state that high reliability “motivated the inclusion of all cells in subsequent analyses.” Place-cell filtering was used only where specified, not as global data curation.
- A cell absent on a released day is all-NaN and must be omitted from that session.

**Trial curation rules**:
- Native sessions are continuous 40-minute recordings, one per day.
- No paper-level session exclusion is described for the released set.
- Reference position decoder filters samples by smoothed speed >5 cm/s; this is decoder-analysis-specific and will be reconciled with fixed one-minute target trials in Step 5.

### Decoders Trained
| Decoded variable | Accuracy |
|------------------|----------|
| 15×15 animal position | Euclidean error decreased over days, ANOVA p<0.0001, F=7.9845; no categorical accuracy reported |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Subject count | Hard-coded 7 IDs | 7 animal files | 4 male + 3 female | Exact agreement. |
| Cell count | Functions operate on finite per-day cells | 69,744 finite session-neurons; 5,413 tracked identities | 69,744 rate maps; 515–952 cells/animal, mean 773±68 | Exact agreement (5,413/7=773.3). |
| Session count | Arrays indexed by day | 207 days: 31×6 + 21×1 | Sequences repeated up to three times | QLAK-CA1-51 has two sequences plus final square (21); all others have three plus final square (31). |
| Trace meaning | Uses released binary `trace` directly | finite values exactly {0,1} | binary rising phases treated as firing rate | Exact agreement; do not recompute dF/F. |
| Alignment | Functions assume frame correspondence | all 207 position/trace lengths identical | streams acquired together at 30 Hz and timestamped | Exact agreement. |
| Session duration | `fps=30`; arrays ~71.9–72.2k frames | 2395.5–2407.3 s | all sessions 40 min | Small acquisition-length variation around 40 min; use complete fixed windows and document discarded tail. |
| Cell filtering | Decoder has >5-event filter; place-cell helper exists | unregistered cells are all-NaN | all cells included in subsequent analyses | >5/place-cell criteria are analysis-specific. Target conversion retains every finite registered cell per session. |
| Speed filtering | Position decoder uses >5 cm/s | all position frames valid | decoder method is within-session analysis | Target requires continuous 1-minute trials and time-varying position. Retain all valid frames; speed filtering would destroy regular temporal trials and is not global data curation. |
| Spatial resolution | Maps/decoder use 15×15 | positions span 0–75 cm | task specifies 3×3=9 bins | Target specification overrides reference 15×15 output; use the arena's natural 25 cm partitions. |

### Final Understanding
- Each animal-day is a decoder session. Each source day is a continuous, aligned calcium-event and position recording of approximately 40 minutes.
- Session neural matrices consist of all rows finite on that day; all-NaN unregistered rows are removed.
- Geometry is static within day and is represented directly by blocked 3×3 partitions.
- Position is transformed from continuous 0–75 cm coordinates to the required 3×3 categorical location.
- Reference temporal pooling (3 frames / 100 ms) is applicable and computationally appropriate; exact mapping details are specified in Step 5.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `trace[day]` | `neural[session][trial]` | Keep rows wholly finite on that day; Gaussian smooth sigma=3 frames along time; non-overlapping mean over 3 frames; split into 600-bin trials; cast float32 | `fit_decoder`, `test_decoder` | Shape neurons×600; 100 ms bins. Float32 avoids low-amplitude normalization overflow in validator plots. |
| `blocked[day]` | `input[session][trial]` | 9-vector, 1=blocked and 0=accessible; `-1` means all zeros | README, `get_env_mat` | Static per trial; row-major partition order. |
| `position[day]` | `output[session][trial]` | Mean x,y over same 3-frame windows; floor each coordinate/25 cm; clip to 0..2; label=`y_bin*3+x_bin`; nearest accessible-bin correction for rare blocked labels | `fit_decoder`, `test_decoder`, blocked README | Shape 1×600, uint8, values 0..8. |
| animal ID | `subjects`, `subject_idx` | Seven IDs; every day/session indexed to its animal | `main.py` animal list | One subject per animal file. |
| CA1 recordings | `brain_regions`, `brain_region_idx` | Region list `['CA1']`; all per-session neurons index 0 | Paper | Dorsal CA1. |
| geometry cells | `input_names` | `blocked_row{r}_col{c}` in row-major order | README | Nine named inputs. |
| 3×3 position | `output_names`, `output_values` | one output `position_3x3`; nine row/column labels | Target task | Categorical time series. |

### Key Decisions
1. **Session definition**: each animal-day is one session, matching source and paper terminology.
2. **Trial definition**: split each session into consecutive non-overlapping 60 s windows, as required. At 30 Hz this is 1,800 source frames or 600 target bins.
3. **Temporal resolution**: use 3-frame/100 ms bins, matching reference position decoder. Neural events are Gaussian-smoothed with sigma 3 source frames before average pooling, exactly matching `fit_decoder`/`test_decoder`; position uses matching average pooling and integer spatial discretization.
4. **Tail handling**: retain only complete 60 s trials. This yields 39 trials for sessions with 71,866 frames and 40 for sessions with ≥72,000 frames, totaling 8,187 trials. Discarded tails are 2.0–55.5 s. Incomplete trials are excluded to keep all trial sizes identical.
5. **Neuron curation**: retain every cell with a wholly finite trace in that session and remove only all-NaN unregistered rows. This matches the paper's inclusion of all cells. Do not impose place-cell, event-count, or speed criteria because those are analysis-specific and would conflict with continuous fixed trials.
6. **Geometry input**: encode the supplied `blocked` indices directly as a 9-element binary vector (1=blocked). All repetitions/animals show one canonical mask per geometry.
7. **Position orientation**: source coordinate 0 is x (column), coordinate 1 is y (row), while blocked IDs are row-major. Therefore class=`y*3+x`. The opposite convention incorrectly places 17.1% of frames in blocked cells; the selected convention leaves only 445/14.9M raw frames in blocked cells.
8. **Rare invalid pooled positions**: 152/4,912,200 complete pooled bins (0.0031%; center cells in `u`/`bit donut`) enter blocked cells due to tracking/interpolation/averaging. Snap these to the nearest accessible 3×3 cell by Euclidean distance from pooled x-y location, analogous to the reference decoder's nearest valid occupied-bin cleanup.
9. **Boundary handling**: use half-open 25 cm bins via floor; clip 75 cm to index 2. No pooled coordinate lies exactly on 25/50/75 boundaries in the diagnostic scan.
10. **Data types**: neural float32 (smoothed event rates), output uint8, geometry uint8. Float16 neural storage is ~3.12 GiB versus ~6.24 GiB float32; decoder converts to float32. Values remain finite and nonnegative.
11. **Session ordering**: animals in reference order, then native day order. Subject list uses the same animal order.
12. **Metadata alignment**: trials align to their own start (consecutive 60 s segments), with `off_start=0`, `off_end=60`, `time_bin_size=100` ms.

### Planned Sanity Checks
- [x] Verify all 207 source position/trace frame counts match (passed).
- [x] Verify all finite source neural values are binary and position finite (passed).
- [x] Verify geometry masks are canonical across animals/repetitions (passed).
- [x] Verify correct row-major coordinate orientation against blocked cells (passed; `y*3+x`).
- [x] Quantify output classes before correction: fractions `[0.099663, 0.098495, 0.135121, 0.075412, 0.057266, 0.076848, 0.115696, 0.141213, 0.200287]`.
- [ ] `np.allclose` converted neural trial against independent raw smoothing/pooling calculation.
- [ ] `np.allclose` converted input mask against independent raw `blocked` construction.
- [ ] `np.allclose` converted output against independent raw position pooling/binning.
- [ ] Assert every converted output is 0..8 and never names a blocked cell.
- [ ] Verify exact session/trial/neuron totals and format with provided validator.

---

## Step 6: Script Development
**Status**: COMPLETE

- Created `/app/convert_data.py`; syntax verified with `python3 -m py_compile`.
- CLI supports positional output path, mutually exclusive `--full`/`--sample` (full is default), and `--show-processing`.
- Conversion loads one animal at a time with the reference joblib representation, validates source alignment/values, processes each day as a session, and serializes the exact target dictionary.
- `--show-processing` saves up to two four-panel figures showing raw position, binary events, pooled neural activity, and aligned categorical output.
- Assertions validate session/trial correspondence, dimensions, finite values, categorical ranges, and neuron-region indexing before writing.
- Per-animal/session timing and final file size are printed.

Code inefficiencies identified:
- Full 100 ms data contain ~1.674 billion neural values; float32 storage would require ~6.24 GiB before pickle overhead.
- Repeated per-trial copies are required by the target nested-list format.

Code speedups added:
- Vectorized Gaussian filtering, pooling, coordinate discretization, and blocked-bin correction.
- Loads/releases one animal at a time.
- Initially tested float16 neural storage, then changed final neural arrays to float32 after validator normalization exposed low-amplitude overflow; categorical output remains uint8 and input is float32.
- Avoids recomputing source processing for each trial by processing a whole session at once.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| File | `/app/sample_data.pkl` (0.806 GiB) |
| Neurons (session-neuron sum) | 18,443 |
| Neurons / session | 153–422; mean 297.47 |
| Subjects | 2 (QLAK-CA1-08, QLAK-CA1-30) |
| Sessions / subject | 31 each; 62 total |
| Trials (total) | 2,418 |
| Trials / session | 39 |
| Timepoints / trial | 600 |
| Neural range | [0.0, 1.0] |
| Input range | [0.0, 1.0] |
| Position distribution | [0.0851, 0.1024, 0.1253, 0.0693, 0.0650, 0.0800, 0.1045, 0.1494, 0.2191] |
| Corrected blocked pooled bins | 2 of 1,450,800 |

### Processing Plots Review
- `processing_QLAK-CA1-08_day00.png` and `processing_QLAK-CA1-08_day01.png` were generated (1960×1540 RGBA) and contain nontrivial image ranges.
- Four aligned panels show source x-y coordinates, source binary events, 100 ms smoothed/pooled neural activity, and the synchronized 3×3 output over a one-minute segment.
- No missing, shifted, or malformed traces were observed in plot/file checks.

### Format Validation
- `/app/verification_sample_out.txt` created.
- Final validator result: “Data format is valid, no errors or warnings.”
- Validator confirms 62 sessions, 2,418 trials, T=600, 9 inputs, 1 output, and 18,443 CA1 session-neurons.
- Initial iteration produced repeated warnings because binary geometry inputs were uint8. Fixed `blocked_vector` to return float32, reran sample conversion and all validation checks; warnings are now absent.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|-----------------------|--------------|
| Whole-session vectorized filtering/pooling | Avoids per-trial source processing |
| One-animal-at-a-time loading | Bounds working memory |
| Vectorized float32 session serialization | Avoids repeated conversion and eliminates low-amplitude plotting overflow |

| Step | Time / Session | Estimated Total Time |
|------|----------------|----------------------|
| Sample conversion including two large animal loads, plots, write | 54.9 s / 62 = 0.89 s average | ~183 s for 207 sessions; allow 3–5 min for larger total write |
| Sample verification | <20 s observed | approximately 1 min full |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None after converting geometry inputs from uint8 to float32 and rerunning conversion/validation.

### Training Progress
- Device: CUDA
- Trials: 1,922 train; 496 validation
- Epochs: 200/200 completed
- Loss decreased monotonically from 2.305421 (epoch 1) to 1.264269 (epoch 200).
- Test loss: 1.269767.
- Script finished successfully.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-----------------------|-------------------------|
| position_3x3 | 0.6465 | 0.5549 |

- Uniform chance is 1/9 = 0.1111; validation is approximately 5.0× chance.
- Train/validation ratio is 1.17, below the 1.5 overfitting concern threshold.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 6,627,415,776 bytes (6.172 GiB), float32 neural arrays
- `verification_full_out.txt`: created (40,923 bytes)
- `conversion_full_out.txt`: created
- Conversion runtime: 203.7 s after final dtype fix

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total session-neurons | 69,744 rate maps | finite cells/day | 69,744 | 69,744 | Yes |
| Mean neurons/session | not stated | finite cells/day | 336.93 | 336.93 | Yes |
| Longitudinal cells/animal | 773±68 SE, 515–952 | seven files | 5,413 total; 515–952 | source identities represented through sessions | Yes |
| Subjects | 7 | 7 hard-coded IDs | 7 | 7 | Yes |
| Sessions | up to 3 sequences/animal | day indexed | 207 | 207 | Yes |
| Trials (total) | N/A (continuous sessions) | N/A | 207 ~40-min sessions | 8,187 complete 1-min trials | Target-derived |
| Trials/session | N/A | N/A | 39–40 complete minutes | 39–40 | Yes |
| Timepoints/trial | N/A | 3-frame pooling | 30 Hz source | 600 at 100 ms | Yes |
| Input range | blocked/open geometry | binary environment matrices | [0,1] | [0,1] | Yes |
| Position distribution | not reported | spatially binned | expected geometry-dependent occupancy | `[.099663,.098524,.135121,.075412,.057235,.076849,.115696,.141213,.200287]` | Sensible |
| Brain region | dorsal CA1 | CA1 IDs | CA1 | CA1 | Yes |

### Full Validator Results
- “Data format is valid, no errors or warnings.”
- 207 sessions; 8,187 trials; 7 subjects; input dimension 9; output dimension 1.
- 69,744 CA1 session-neurons.
- All converted outputs are valid classes 0–8 and no output names a blocked partition.

### Iteration
- Initial full float16 output passed structural validation but a verifier sample-plot operation emitted an overflow warning when normalizing extremely low-amplitude sparse traces in float16.
- Silent/low-event neuron-trial segments are expected for sparse calcium events and were correctly retained; filtering them would violate the paper's all-cell inclusion.
- Changed neural storage to float32, reran sample conversion/validation, then reran full conversion/validation. Final full verification has no warning and all counts/distributions are unchanged.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Final `verification_full_out.txt` contains “Data format is valid, no errors or warnings” and ends with “Data verification complete.” No RuntimeWarning, traceback, mismatch, or failure remains.
2. **Independent neural sanity check**: Loaded original joblib files directly (without conversion functions), selected finite rows, independently applied `gaussian_filter1d(..., sigma=3)` and 3-frame means, and compared trials using `np.allclose(rtol=0, atol=1e-7)`. QLAK-CA1-08 day 0 trial 5 and QLAK-CA1-51 day 0 trial 10 both passed.
3. **Independent input sanity check**: Constructed masks directly from raw `blocked` arrays and compared using `np.allclose(..., atol=0)` for the same spots. Both passed.
4. **Independent output sanity check**: Pooled raw positions directly, computed `row=floor(y/25)`, `col=floor(x/25)`, and `row*3+col`, then independently corrected blocked labels. Exact `np.allclose(..., atol=0)` passed for both spots.
5. **Corrected edge case**: Independently reconstructed QLAK-CA1-51 day 4, which has 145 pooled center-bin labels in a blocked bit-donut center. Every converted label matched independent nearest-accessible correction and zero blocked outputs remained.
6. **Global structure check**: Every neural array is finite/nonnegative float32 with shape neurons×600; every input is float32 shape (9,); every output is uint8 shape (1,600), class 0–8; brain-region lengths match neurons.
7. **Off-by-one check**: For every session, number of trials equals `source_frames//1800`, tail equals `source_frames - trials*1800`, and final trial has 600 bins. No issues. First session uses 70,200/71,866 frames; last uses 72,000/72,071.
8. **Blocked-cell check**: Across 4,912,200 output samples, converted outputs in blocked cells = 0.

### Reference Code Comparison
| Major step | Conversion | Reference | Comparison/reasoning |
|------------|------------|-----------|----------------------|
| Data loading | `joblib.load('/app/data/'+animal)` and inner animal dict | `load_dat` uses joblib by default | Same representation/path convention. |
| Neuron filtering | keep wholly finite rows, reject partial rows | released absent cells are NaN; paper includes all cells | Same all-registered-cell policy; analysis-specific place/event filters not applied. |
| Trial filtering | all complete consecutive one-minute windows | native sessions continuous; speed filter only in decoder analysis | Target fixed trials require regular timeline; all valid frames retained. |
| Temporal alignment | identical source indices for trace and position | streams simultaneous/timestamped; code assumes aligned frames | Exact correspondence; all 207 lengths checked. |
| Neural binning | Gaussian sigma 3 frames then non-overlapping 3-frame mean | `fit_decoder`/`test_decoder`: same operations | Logic and parameters match. |
| Position binning | same 3-frame means then required 3×3 bins | reference decoder mean-pools positions then casts spatial bins | Same temporal logic; target overrides spatial resolution (3×3 vs 15×15). |
| Input construction | raw blocked IDs to 9 binary indicators | README/get_env_mat encode omitted partitions | Direct source encoding with documented polarity. |
| Output construction | row-major `y*3+x`, nearest valid cleanup | reference uses map row/column and snaps to nearest valid occupied bin | Same orientation/cleanup principle at target resolution. |

### Key Statistics Comparison
- Raw and converted subject/session counts: 7 and 207 exactly.
- Paper 69,744 rate maps equals raw finite session-neurons and converted 69,744 exactly.
- Raw longitudinal identities 5,413 / 7 = 773.3, matching paper mean 773; raw min/max 515/952 match paper.
- Output class counts: `[489566,483969,663743,370441,281149,377499,568320,693665,983848]`; all classes have substantial support.
- Subject session counts `[31,31,31,21,31,31,31]` match raw files.

### Issues Found and Resolved
- **Wrong diagnostic orientation initially tested**: `x*3+y` placed 17.1% in blocked bins. Raw geometry demonstrated blocked IDs are row-major, so corrected mapping is `y*3+x`; only 0.003% pooled tracking/interpolation artifacts then remained.
- **Rare pooled blocked labels**: snapped 152/4,912,200 samples to nearest accessible bin, matching reference decoder cleanup logic.
- **uint8 input warnings**: changed geometry input to float32 and reran validation.
- **float16 visualization overflow**: changed neural output to float32 and reran both sample/full conversion and validation. Final logs are clean.
- Sanity-check script/output are archived as `/app/cache/sanity_checks.py` and `/app/cache/sanity_checks_out.txt`.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Command: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`
- Device: CUDA
- Trials: 6,531 training; 1,656 validation
- Epochs: 200/200 completed
- Loss decreasing: Yes; 2.318937 (epoch 1) to 1.160279 (epoch 200)
- Test loss: 1.143706
- Script ended with `train_decoder.py finished successfully.`

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-----------------------|-------------------------|-------|
| position_3x3 | 0.6988 | 0.6084 | chance 0.1111; validation 5.48× chance |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Achieved Accuracy | Chance | Expectation from Paper |
|----------|-------------------|--------|------------------------|
| position_3x3 | train 0.6988; validation 0.6084 | 0.1111 | Paper decodes 15×15 position and reports Euclidean error decreasing across days (ANOVA p<0.0001, F=7.9845), not categorical accuracy |

1. **Accuracy vs chance**: validation balanced accuracy is 5.48× uniform chance and far above the 1.5× concern threshold.
2. **Accuracy comparison to paper**: the paper's only animal-position result is Euclidean error in cm for a 15×15 Gaussian Naive Bayes decoder, with a significant decrease across sessions and performance reaching the maximum reported in cited recent work. It does not report categorical accuracy, so no numerical apples-to-apples percentage exists. The strong 0.6084 nine-class validation balanced accuracy is consistent with reliable CA1 spatial coding.
3. **Train vs validation gap**: 0.6988/0.6084 = 1.15, below the 1.5 overfitting concern threshold.
4. **Loss behavior**: training loss decreased steadily across all 200 epochs and test loss (1.143706) is slightly below final training loss (1.160279), with no divergence.
5. **Three raw-trial output checks**: independently reconstructed QLAK-CA1-08 day 0 trial 0, QLAK-CA1-30 day 13 trial 20, and QLAK-CA1-75 day 30 trial 39. All matched exactly, had 10–17 spatial transitions, represented multiple classes, and contained zero blocked labels.
6. **Temporal alignment**: raw and converted checks use identical 1,800-frame windows and matching three-frame pooling for neural and position. Processing/sample/prediction plots exist and show time-varying synchronized outputs.
7. **Output variation**: all nine classes have substantial global support (minimum fraction 0.0572); individual geometry sessions correctly omit blocked classes rather than indicating class collapse.
8. **Filtering check**: all registered finite cells remain, matching paper rationale; no inappropriate place-cell/speed filter was introduced.

### Issues Found and Resolved
- No accuracy-related conversion issue was found.
- Raw reconstruction, alignment, variation, reference-processing, and filtering checks all passed; no conversion rerun was required after Step 11.
- Full training figures: `/app/sample_trials.png` and `/app/predictions.png`.
- Three-trial script/output are archived in `/app/cache/`.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created with description, loading example, schema, processing, key statistics, and reproduction commands.
- [x] cache/ folder created; investigation scripts and outputs moved there.
- [x] `cache/README_CACHE.md` documents cached files.
- [x] Required conversion, validation, training, plot, script, data, and documentation files verified.
- [x] CONVERSION_NOTES.md reviewed for statuses, decisions, validation results, and issue resolutions.
