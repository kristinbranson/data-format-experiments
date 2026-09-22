# Dataset Conversion Notes

## Overview
- **Dataset**: Longitudinal Track2p mouse barrel-cortex imaging dataset
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
- `docker-compose.yaml~`
- `methods.txt`
- `paper.pdf`
- `train_decoder.py`

Environment check: Python 3.13.15, NumPy 2.4.4, and PyTorch 2.6.0+cu124 import successfully. Checkpoint confirmed `/app/CONVERSION_NOTES.md` exists.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_stat_ds_plane` | `code/track2p/io/loaders.py` | LOADING/CURATION | Loads Suite2p `stat.npy` and `iscell.npy`; retains `iscell[:,0]==1` when threshold is `None`, otherwise probability `iscell[:,1] > iscell_thr`. |
| `load_all_imgs` | `code/track2p/io/s2p_loaders.py` | LOADING | Loads Suite2p `ops.npy` mean images/channel metadata for every day and plane. |
| `get_all_ds_assign` | `code/track2p/match/loop.py` | PROCESSING/CURATION | Hungarian assignment of registered ROIs followed by an automatically selected Otsu/minimum IOU threshold. |
| `get_all_pl_match_mat` | `code/track2p/match/loop.py` | PROCESSING | Propagates pairwise matches through consecutive recordings; tracking stops at the first missing match. |
| `generate_suite2p_indices` | `code/track2p/t2p.py` | PROCESSING | Converts indices in the iscell-filtered arrays back to original Suite2p ROI indices. |
| `DataManagement.import_files` | `code/track2p/gui/data_management.py` | LOADING/CURATION | Restricts match matrix to rows present on every day, applies the exact tracking iscell threshold, and indexes F/spks/Fneu using matched-cell indices. |
| `DataManagement.F_processing` | `code/track2p/gui/data_management.py` | PROCESSING | Optional fluorescence baseline subtraction: `Fc=F-neucoeff*Fneu` (default neucoeff 0), Gaussian then 60-s min/max baseline, returns `Fc-Flow` (despite GUI label dF/F0, it does not divide by baseline). |
| `save_in_s2p_format` | `code/track2p/t2p.py` | PROCESSING | Saves tracked-all-days subsets of F, Fneu, spks, stat and iscell in Suite2p format. |

### Notes
- Track2p operates on Suite2p outputs (`F.npy`, `Fneu.npy`, `spks.npy`, `stat.npy`, `iscell.npy`, `ops.npy`) and does not contain behavioral synchronization logic; native supplied data must therefore determine neural/motion alignment.
- Default and example runs use `iscell_thr=0.5` on Suite2p's probability column. Correct downstream indexing requires using the same threshold used to create the Track2p match matrix.
- For longitudinal visualization/export, only rows with a valid match on every day are retained. The decoder task, however, is session based; whether supplied data are already curated/tracked must be resolved from the data and paper before imposing another filter.
- Available neural traces are raw fluorescence, Suite2p deconvolved `spks`, and the GUI's baseline-subtracted fluorescence. No true delta-F-over-F computation exists in the reference repository. We should not invent dF/F unless the provided dataset/reference methods explicitly require it.
- The NPY utility assumes 30 Hz only for generic converted inputs; actual Suite2p `ops['fs']` is authoritative where present.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` is 16 GB and contains six subject folders: `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, and `jm046`. Sessions are dated subfolders in chronological order; each has one `suite2p/plane0` and one `move_deve` directory.
- `suite2p/plane0`: `F.npy`, `Fneu.npy`, and `spks.npy` are matched-neuron x imaging-frame float32 arrays. `stat.npy` contains ROI geometry dictionaries, `iscell.npy` is `(n_neurons,2)` float64, and `ops.npy` contains acquisition/processing parameters and mean/registration images. The distributed files are already Track2p-curated to cells present across every supplied day for that subject, with row identity matched across days.
- `move_deve`: `motion_energy_glob.npy` is one uint64 whole-body motion-energy value per retained video frame; `tstamps.npy` contains relative camera timestamps; `interframe_int.npy=np.diff(tstamps)`. All are finite. README states missing camera frames cause motion arrays shorter than two-photon traces and may be represented as missing or interpolated.
- `README.md` and `load_data.ipynb` document loading. The notebook warns raw F requires paper-style dF/F processing and offers `spks.npy` as an alternative. Optional manual tracking ground-truth CSVs exist for jm038, jm039, and jm046; they are validation annotations, not decoder variables.
- All sessions have one 512x512 plane, nominal `ops['fs']=30 Hz`, two imaging channels, `tau=0.3`, Suite2p `neucoeff=0.7`, and 36,000 or 54,000 frames. Exactly one Suite2p `badframes` entry is true (jm032/2023-10-24); reference handling must be checked before excluding a timepoint.
- The supplied `iscell` arrays are already matched/filtered: first column is one for every row, and all probabilities exceed 0.5 (range 0.50025-0.99841). Reapplying `>0.5` changes nothing.
- Motion length deficits occur in 9 sessions: 2, 3, 116, 2, 2, 148, 1, 1, and 1 frames (276 total). In every deficient session, rounded timestamp gaps infer exactly the same count and positions. Three jm046 sessions have timestamp pauses but no length deficit, so length—not timestamp jitter alone—is the reliable missing-frame criterion described by the data README.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 2,998 unique longitudinally tracked cells; 20,445 session-neuron instances |
| Neurons / session | 221 (jm031), 370 (jm032), 685 (jm038), 746 (jm039), 541 (jm040), 435 (jm046); mean 498.66 across sessions |
| Subjects | 6 |
| Sessions / subject | 7 each except jm040 with 6; 41 total |
| Trials (total) | Native recordings are continuous (no native trials); 1,090 complete 60-s decoder trials are implied at 30 Hz |
| Trials / session | 20 for 36,000-frame sessions (jm031/jm032); 30 for 54,000-frame sessions (others) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 3,156 implied by 526 mean x 6 (reported mean is rounded); distributed data contain 2,998 | "On average 526 (±190 std) neurons per mouse were successfully tracked across all days" |
| Neurons / session | 526 ± 190 tracked cells per mouse | Paper Results/method excerpt |
| Subjects | 6 | "A total of 6 mice were used in the study" |
| Sessions / subject | At least 6 consecutive daily sessions; full dataset P7-P14 | Paper Results and chronic imaging Methods |
| Trials (total) | No native trials; continuous spontaneous recording | Experiment had no trial events |
| Trials / session | N/A natively; requested conversion uses 60-s blocks | Decoder task overrides paper's 2-min CV blocks |
| Neural data time bin | 30-Hz raw; 10-frame mean (333.33 ms) for decoding | "Imaging rate was 30 Hz" and "averaging in bins of 10 consecutive timestamps" |
| Behavior data time bin | 30-Hz raw, microscope-triggered; 10-frame mean (333.33 ms) for decoding | Videography and Decoding Methods |
| Reward rate | N/A | Spontaneous behavior, no reward/task |
| Session duration | 20 min reported | "each session lasted 20 minutes" |
| Tracked fraction | 33% ± 11% of first-day detected cells | Paper Results |
| Motion definition | Sum of squared pixelwise difference between consecutive video frames | Preprocessing videography Methods |
| Example same-day continuous decoding R² | 0.26 early (P8); 0.69 late (P14) | Figure 7B |
| Example cross-day continuous decoding R² | 0.15 early→early; 0.11 early→late; 0.61 late→late | Figure 7F |


### Processing Details
- Two-photon imaging and behavior video were acquired at 30 Hz. The microscope triggered each camera frame, providing direct framewise synchronization.
- Suite2p separately performed motion correction, ROI detection, signal extraction, and spike deconvolution for each recording. Paper analyses used baseline-corrected fluorescence, called dF/F, with default Suite2p parameters.
- For all paper decoding, both fluorescence and motion-energy traces were averaged over 10 consecutive timestamps (effective bin 1/3 s). The paper used continuous motion and ridge regression; 5x5 nested cross-validation split recordings into consecutive 2-minute blocks. The present task instead requires 60-s trials and five within-session percentile categories.
- Motion energy is a whole-frame scalar: square consecutive-frame pixel differences and sum across pixels. Supplied motion energy is already processed, so this operation must not be repeated.
- Same-day continuous decoding improved with age and motion coupling emerged after P11. Exact categorical accuracy is not reported, so Figure 7 R² values are qualitative rather than directly comparable to the requested balanced classification accuracy.

### Curation Steps

**Neuron curation rules**:
Suite2p ROIs with classifier probability strictly greater than 0.5 were considered cells. Track2p then affine-registered sessions using the anatomical tdTomato channel, used Hungarian assignment to maximize ROI IOU, Otsu-thresholded candidate matches, and retained cells successfully tracked across all included days. The distributed arrays already contain that curated population.

**Trial curation rules**:
No behavioral trials or rejected-trial rule is described. Recordings are continuous spontaneous behavior under sensory-minimized conditions. Paper cross-validation used consecutive 2-minute blocks but did not discard behavioral blocks.

### Decoders Trained
| Decoded variable | Accuracy |
| Continuous motion energy from neural population (ridge regression) | Figure examples: same-day R² 0.26 early and 0.69 late; cross-day R² 0.11-0.61 |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Session duration | 20 min at 30 Hz | 14 arrays have 36,000 frames; 27 have 54,000 frames and camera duration ~1,814.5 s | "each session lasted 20 minutes" | Preserve every supplied complete minute: the 54,000-frame files unambiguously contain ~30 min. Treat paper statement as a summary/protocol discrepancy, not a reason to truncate valid data. |
| Tracked-cell count | `>0.5` iscell, then complete all-day tracks | 2,998 cells (499.7 ±197.7 sample SD/mouse) | 526 ±190/mouse; 33 ±11% of day-1 cells | Supplied, publication-linked Track2p exports are authoritative for conversion. Difference is modest and likely reflects dataset/export revision or rounded manuscript analysis; do not add/remove already curated cells. |
| Neural signal | GUI offers F, spks, or baseline-subtracted F; its `F_processing` implements Suite2p maximin subtraction but has a GUI default `neucoeff=0.0` | Raw F/Fneu and deconvolved spks are supplied; ops specify Suite2p defaults `neucoeff=0.7`, maximin, 60 s, sigma 10, percentile 8 | Analyses used baseline-corrected fluorescence (called dF/F) with default Suite2p parameters | Compute `Fc=F-0.7*Fneu`, then reference maximin baseline and `Fc-Flow`; use ops values. This matches paper and Suite2p settings more closely than the GUI helper's accidental zero-neuropil default. Do not divide by Flow because neither reference implementation nor Suite2p preprocessing does so. |
| Temporal denoising | No behavioral analysis in Track2p core; GUI displays frame data | Imaging and motion are framewise aligned at 30 Hz | Decoder averages both neural and behavior in 10-frame bins | Mean every 10 consecutive frames before task-specific discretization, yielding 333.333-ms bins and exactly 180 bins per 60-s trial. |
| Missing video frames | Not handled in Track2p core | 276 deficits (0.0141%); timestamp gaps exactly identify missing indices | Camera was imaging-triggered; data README permits missing values or interpolation | Insert missing samples at timestamp-identified indices and linearly interpolate only those points. This maintains frame alignment, fixed trial lengths, and all neural data; record counts and test reconstructed non-missing values exactly. |
| Timestamp units | Not used | Camera timestamp median increment is ~3.359e-5 stored units; direct index lengths correspond to imaging frames | Both streams acquired/triggered at 30 Hz | Use imaging `ops['fs']=30` for elapsed seconds. Use behavior timestamps only to locate missing frame indices; avoids uncertain stored-unit scaling and minor camera-clock jitter. |
| Decoder target | N/A | Continuous motion energy | Continuous motion ridge regression after averaging | Required target overrides reference: compute within-session 20/40/60/80% thresholds after 10-frame averaging and encode classes 0-4. |
| Trial/block definition | N/A | Continuous recordings | Paper CV used consecutive 2-min blocks | Required 60-s trials override paper CV blocks. All recordings divide evenly into complete 60-s trials. |
| Bad imaging frames | Suite2p ops exposes `badframes` | One true entry in 2,214,000 frames | No downstream exclusion described | Retain it: F traces exist, paper does not prescribe deletion, and dropping one frame would break synchronized complete-minute blocks. Its 10-frame average limits influence. |

Final understanding: each dated recording is one decoder session; provided neural rows are already `iscell>0.5`, Track2p-matched complete longitudinal tracks in layer 2/3 barrel cortex. Neural and global motion streams are frame-synchronous at nominal 30 Hz. Reconstruct the very sparse missing behavioral samples, apply the paper's Suite2p baseline correction and 10-frame temporal averaging, discretize the averaged motion into session-specific quintiles, and partition without overlap into complete 60-s trials.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `F.npy`, `Fneu.npy`, Suite2p ops | `neural` | Per session compute `Fc=F-neucoeff*Fneu`; apply maximin baseline (`gaussian_filter`, 60-s minimum then maximum filters) and subtract baseline; mean non-overlapping groups of 10 frames; split columns into 60-s trials, float32 `(neurons,180)` | `DataManagement.F_processing`; paper Calcium processing/Decoding | Use already tracked rows; no additional cell filtering. |
| Imaging frame index and `ops['fs']` | `input[0]` | Absolute elapsed time at the center of each 10-frame bin: mean of frame times, then same 60-s split; float32 `(1,180)` | Paper acquisition/Decoding | Does not reset at trial boundaries because requested input is from session start. |
| `motion_energy_glob.npy`, `tstamps.npy` | `output[0]` | Reconstruct missing frames from timestamp gap multiples; linearly interpolate only missing positions; mean 10-frame groups; calculate session-wide 20/40/60/80 percentiles; `searchsorted(..., side='right')` gives int64 labels 0-4; split `(1,180)` | Paper Videography/Decoding plus data README | All sessions have four distinct thresholds and exactly 20% of bins in each class. |
| Subject directory name | `subjects`, `subject_idx` | Sorted IDs; session index points to owning subject | Data README | `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`. |
| Paper recording site | `brain_regions`, `brain_region_idx` | One region, index zero for each retained neuron | Chronic imaging Methods | `barrel cortex (S1), layer 2/3`. |
| Session directory/date, ops, processing diagnostics | `metadata['session_info']` | Per-session dictionaries with ID, subject/date, source/raw sizes, duration/trial count, neuron count, missing-frame count, quintile edges | Data hierarchy | Allows audit without embedding unused large arrays. |
| ROI `stat`, `iscell`, Suite2p images/offsets, FOV | metadata summary only | Not decoder variables; retain provenance/curation descriptions, do not copy large imaging objects | Track2p code/data README | Spatial anatomy is not requested as decoder input/output. |
| `spks.npy` | not mapped | Available deconvolved activity but not selected | Data notebook offers as alternative | Paper decoder explicitly used baseline-corrected fluorescence, so spks is not mixed with it. |
| `interframe_int.npy` | validation only | Confirm it equals `diff(tstamps)` and localize missing frames | Data README | Redundant with timestamps. |
| Ground-truth CSV | not mapped | Tracking benchmark only | Paper Tracking evaluation | Not an experimental/behavioral variable for this decoder. |

### Key Decisions
1. **Session/trial definition**: One dated recording is one session. Partition the full stream into consecutive, nonoverlapping 60-s blocks. At 30 Hz and 10-frame averaging this is 180 timepoints/trial; all source recordings divide evenly, yielding 1,090 trials with no truncation.
2. **Reference neural signal**: Use paper-style baseline-corrected fluorescence rather than raw F or spks. Copy the reference maximin logic, but pass each session's Suite2p ops parameters (`neucoeff=0.7`, `win_baseline=60 s`, `sig_baseline=10`) to satisfy the paper's "default Suite2p parameters." No division by baseline because the reference implementation returns `Fc-Flow`.
3. **Temporal binning order**: Repair sparse behavior gaps at the 30-Hz frame level, baseline-process neural data at full rate, then average both streams over identical groups of 10 frames. Only after averaging is motion discretized. This exactly preserves the paper's decoder denoising and avoids averaging arbitrary category numbers.
4. **Missing motion**: When behavior is shorter than neural data, round each camera interval relative to its median to map observed values to imaging-frame indices, require the inferred missing count to equal the length deficit, then linearly interpolate. For equal-length streams, retain direct frame pairing even if timestamps jitter; this follows README's explicit length criterion.
5. **Percentile classes**: Select thresholds independently for each full session, as required. Use NumPy linear quantiles and right-sided boundary assignment. Tests show distinct thresholds and exactly `[0.2,0.2,0.2,0.2,0.2]` class fractions in every session.
6. **Validity/curation**: Keep every provided neuron because exports are already `iscell>0.5` and complete Track2p tracks. Keep all trials; there are no trial rejection rules, all arrays are finite, missing behavior is recoverable, and the lone Suite2p bad frame is not prescribed for downstream deletion.
7. **Names/values**: `input_names=['session_elapsed_time_s']`; `output_names=['motion_energy_quintile']`; output values describe quintiles from lowest through highest motion.
8. **Metadata alignment**: `time_bin_size=1000*10/30=333.333... ms`, alignment event is each 60-s block start, `off_start=0.0`, `off_end=60.0`. Time input remains absolute from the recording start.
9. **Sample choice**: `--sample` uses jm031/2023-10-22_a and jm039/2024-05-04_a. Both contain confirmed missing behavior, cover 20- and 30-min lengths and different neuron counts, and are later developmental sessions where the paper predicts stronger decodability.
10. **Types/storage**: Store neural/input as float32 and categorical output/indices as int64. Process neural data in neuron chunks to bound memory, then pickle the required nested lists.

### Planned Sanity Checks
- [ ] Structure: session counts align across neural/input/output/subject/region arrays; each session has >=2 equal-length trials; all trial shapes are `(n_neurons,180)`, `(1,180)`, `(1,180)`.
- [ ] Source coverage: 6 subjects, 41 sessions, 2,998 unique tracked neurons, 20,445 session-neuron instances, and 1,090 trials; no source session or complete minute omitted.
- [ ] Missing frames: for each deficient session, inferred gap count equals source length deficit; reconstructed values at every observed index are `np.allclose` to raw motion.
- [ ] Neural spot check independent of conversion functions: load original F/Fneu, recompute full-rate baseline and 10-frame mean for selected neurons/bins, and require `np.allclose` to converted trial values.
- [ ] Input spot check: independently construct 10-frame-center elapsed times and require `np.allclose`; verify first center 0.15 s and last center duration-0.18333 s.
- [ ] Output spot check: independently reconstruct/average raw motion, calculate session quantiles/classes, and require `np.allclose`/exact equality for at least three trial locations.
- [ ] Distribution: every session output class fractions must be exactly 0.2 each; global binned-motion range expected from exploration is 531,661.5 to 39,962,858.7.
- [ ] Processing plots: for both sample sessions show raw/neuropil-corrected/baseline-corrected/binned neural traces, raw/interpolated/binned motion with gap markers, quintile thresholds/classes, and absolute elapsed-time input.
- [ ] Finite/range checks: no NaN/Inf after processing, class labels exactly 0-4, input monotonic, and neural variability nonzero.

---

## Step 6: Script Development
**Status**: COMPLETE

[Implementation notes]
- Created `/app/convert_data.py` with the required CLI, mutually exclusive `--full`/`--sample` modes (full is default), and `--show-processing` plots for at most two sessions.
- Implemented ordered discovery, strict source-shape/parameter validation, reference maximin baseline correction, 10-frame averaging, timestamp-gap motion repair, within-session quintiles, 60-s splitting, target dictionary creation, and pre-save validation.
- `--show-processing` covers raw F/Fneu, neuropil correction, baseline estimate/subtraction, temporal neural averaging, missing-motion interpolation, motion averaging/thresholds, and final aligned input/output.
- Syntax compilation and CLI help completed without errors.

Code inefficiencies identified:
Loading/processing a full 746 x 54,000 fluorescence array with several intermediate copies could require >600 MB per session. Trial views could also retain an entire session base array and be non-contiguous.

Code speedups added:
Neurons are processed in vectorized 64-neuron chunks; SciPy's O(n) 1-D filters operate across each chunk; F/Fneu use memory mapping; only the 10x smaller temporally averaged result is retained; trial slices are copied into compact contiguous arrays. Source files are read once per session and timing is printed separately for processing and saving.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 967 session-neuron instances |
| Neurons / session | 221, 746 (mean 483.5) |
| Subjects | 2 (jm031, jm039) |
| Sessions / subject | 1, 1 |
| Trials (total) | 50 |
| Trials / session | 20, 30 |
| session_elapsed_time_s range | [0.15, 1799.8167] s overall; [0.15,1199.8167] and [0.15,1799.8167] |
| Neural range | [-101.4336, 2319.8188] baseline-corrected fluorescence |
| motion_energy_quintile distribution | [0.2,0.2,0.2,0.2,0.2] in each session and overall |
| Missing motion frames repaired | 116, 1 |
| Trial shapes | neural `(221,180)` / `(746,180)`; input/output `(1,180)` |

### Processing Plots Review
Reviewed `processing_jm031_2023-10-22_a.png` and `processing_jm039_2024-05-04_a.png`. Raw F and Fneu are aligned; the neuropil correction and slowly varying maximin baseline are plausible; 10-frame means track rather than shift transients. Motion interpolation markers occur only at source gaps and connect neighboring values. Binned motion retains raw dynamics, class transitions follow the displayed thresholds, and elapsed time is monotonic. The later jm039 source has repeated sharp motion peaks after ~1250 s, but these are visible in the original observed signal rather than introduced by conversion. No temporal shifts or conversion artifacts were found.

Manual pickle inspection confirmed required keys, contiguous float32 neural/input arrays, int64 labels, finite values, full class support, correct metadata, and exact source durations. `/app/verification_sample_out.txt` reports: "Data format is valid, no errors or warnings."

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Chunked neuron processing and 10x temporal reduction before storage | Peak memory bounded; sample pickle only 18.5 MiB |
| Memory mapping source F/Fneu | Avoids full duplicate raw arrays |

| Step | Time / Session | Estimated Total Time |
| Sample conversion including two large plots and save | 1.91 s/session average | 78 s by 41-session count |
| Neural-element-scaled estimate (full/sample ratio 21.34x) | 3.81 s / 48.24M elements | ~81 s conservative total |

The estimate accounts for both session lengths and neuron counts; it is far below 15 minutes, so no further optimization is required before full conversion.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| motion_energy_quintile | 0.7399 | 0.3254 |

Training ran on CUDA for 200 epochs. Loss decreased from 77.7930 at epoch 1 to 0.8597 at epoch 200 (minor expected stochastic fluctuation around epoch 160), and test loss was 4.6505. Validation balanced accuracy is above uniform chance 0.20 and above the 1.5x-chance review threshold (0.30). `/app/train_decoder_sample_out.txt` contains the complete run.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 395.3 MiB (filesystem display 396M)
- `verification_full_out.txt`: created
- `conversion_full_out.txt`: created; conversion completed in 43.99 s

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | ~3,156 implied by rounded mean | Complete longitudinal matches only | 2,998 unique; 20,445 session-instances | 2,998 unique; 20,445 session-instances | Data exact; manuscript modest discrepancy documented |
| Mean neurons/session | 526 ±190 per mouse | Same matched rows each subject/day | 498.66 session-weighted; 499.67 ±197.68 per mouse | 498.66 session-weighted | Yes, source exact |
| Subjects | 6 | Per-subject longitudinal runs | 6 | 6 | Yes |
| Sessions | Minimum 6/mouse | All dated inputs | 41 (7,7,7,7,6,7) | 41 (7,7,7,7,6,7) | Yes |
| Trials (total) | N/A continuous | N/A | 1,090 complete minutes implied | 1,090 | Yes |
| Trials/session (mean) | 20-min protocol stated | N/A | 20 for 14 sessions; 30 for 27 | same; mean 26.59 | Yes, no truncation |
| session_elapsed_time_s range | [0,duration] conceptually | N/A | Bin centers expected [0.15,1199.8167/1799.8167] | [0.15,1799.8167] overall | Yes |
| Binned neural range | Not reported | Baseline-corrected F | N/A | [-294.9912,2808.0327], mean 19.8780, SD 51.5313 | Plausible/finite |
| Binned motion source range | Not reported | N/A | [531,661.5,39,962,858.7] | Used for labels; thresholds stored/session | Yes |
| motion quintile distribution | Continuous target in paper | N/A | Required equal percentiles | [0.2,0.2,0.2,0.2,0.2] every session | Yes |

`verification_full_out.txt` says the format is valid with no errors or warnings. Spot checks covered first/last sessions and every subject boundary; all have correct neuron counts, 180 timepoints/trial, expected absolute-time endpoints, and per-class counts of 720 (20-min) or 1,080 (30-min). Metadata totals reproduce 276 missing behavior frames and one Suite2p bad frame. No source session, tracked neuron, or complete minute was lost.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Read `/app/verification_full_out.txt` and `/app/conversion_full_out.txt`; format is valid and there are no errors or warnings. Searching for error/warning/NaN/Inf/mismatch/invalid/failed returned only the affirmative phrase "no errors or warnings." No warning required waiver.
2. **Independent raw neural check**: `/app/cache/sanity_checks.py` does not import conversion code. It directly loads raw F/Fneu and ops, independently applies neuropil/maximin/10-frame processing, and requires `np.allclose` for neurons `[0,3,last]` across all bins in jm031 first day, jm031 missing-frame day, jm039 missing-frame day, and jm046 last day. It also tests trial 5/time 10/neuron 3. All passed (`rtol=1e-6`, `atol=1e-5`).
3. **Independent raw input check**: Independently generated all bin-center times from raw frame counts and 30 Hz, concatenated converted trials, and required `np.allclose` for all 41 sessions (`rtol=1e-7`, `atol=1e-5`). First/last centers also pass. This checks every trial boundary because concatenation must equal the uninterrupted source-derived series.
4. **Independent raw output check**: Independently loads every raw motion/timestamp/interframe file, verifies `interframe_int=np.diff(tstamps)`, reconstructs observed positions, verifies observed values unchanged with `np.allclose(rtol=0,atol=0)`, calculates 10-frame means/quintiles, and compares every converted label and threshold. All 41 sessions passed; range `[531661.5,39962858.7]` matches exploration.
5. **Reference code comparison—loading**: Converter `discover_sessions`/`load_ops_parameters` loads the same Suite2p arrays and ops organization as `track2p/io/loaders.py` and `gui/data_management.py`. Independent ordering audit confirms all 41 raw dated folders occur exactly once in subject/date order.
6. **Reference code comparison—neuron/trial filtering**: Reference `load_stat_ds_plane` uses iscell probability `>0.5`, and `DataManagement.import_files` retains match rows present all days. Distributed arrays already are that output: all first iscell values are 1 and probabilities >0.5; rows match across a subject's sessions. Converter correctly does not filter again. No native trial filter exists; every complete 60-s block is retained.
7. **Reference code comparison—temporal alignment**: Paper states microscope-triggered 30-Hz camera frames. Direct indices are used when lengths match; only README-documented deficits are restored from timestamp gaps. The independent audit proves all 276 source observations/gaps map correctly and no observed value changes.
8. **Reference code comparison—binning**: Paper decoding averages neural and behavioral data in non-overlapping 10-timestamp bins. Converter does exactly this before 60-s splitting; 30 Hz / 10 gives 333.333 ms and 180 bins/trial. Difference from paper's 2-min blocks is explicitly required by the task's 60-s trials.
9. **Reference code comparison—input construction**: The paper decoder did not use elapsed time as a covariate, but the requested decoder input explicitly requires it. Absolute bin-center time (rather than resetting at each artificial trial) implements "from the beginning of the session" precisely.
10. **Reference code comparison—output construction**: Reference uses already-computed global motion energy averaged by 10 and predicts it continuously. Converter uses the same loading/alignment/averaging, then adds required session-specific quintile discretization. Every session has distinct edges and exact 20% classes.
11. **Key statistics**: Raw and converted agree on 6 mice, 41 sessions (7/7/7/7/6/7), 2,998 unique tracked cells, per-session neuron vectors, 20,445 session-neurons, 1,090 complete minutes, 276 missing behavior frames, one Suite2p bad frame, and all input/output ranges. Manuscript count/duration discrepancies remain source-level differences already resolved in Step 4—not conversion losses.
12. **Edge cases/off-by-one review**: Raw sessions are exactly divisible by 1,800 frames/trial and 10 frames/bin. Frame-bin centers use `(10k+4.5)/30`; first=0.15 s and last=duration−5.5/30 s. Concatenating every converted trial exactly reconstructs independently processed session streams, proving neither duplicated nor dropped boundary bins. Missing-frame positions include intact first/last observed frames. Equal quantile thresholds never occur. The single Suite2p bad frame is finite and safely averaged rather than causing a shifted stream. Sample/full modes both contain >=2 trials/session.
13. **Artifact integrity**: Exact sizes are 414,459,976 bytes (full) and 19,418,723 bytes (sample); SHA-256 values were recorded during review. All arrays are finite and contiguous per trial.

### Issues Found and Resolved
- No conversion issue was found in this review, so no repair/re-run iteration was necessary.
- Source-level manuscript discrepancies (20-minute statement versus valid 30-minute files; reported 526±190 versus distributed 499.7±197.7 cells/mouse) were not "fixed" by deleting or fabricating data; preserving publication-linked source arrays is the justified resolution documented in Step 4.
- Reference terminology calls baseline-subtracted activity dF/F although code does not divide by baseline. Converter follows executable reference logic (`Fc-Flow`) and records the exact signal description rather than silently adding a division.

Independent audit output: `/app/cache/sanity_checks_out.txt` ends with `ALL INDEPENDENT SANITY CHECKS PASSED`.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes. Epoch loss decreased from 60.8860 (epoch 1) to 1.0206 (epoch 200); minor stochastic bumps at epochs 160 and 190 are small relative to the overall decline. Test loss: 3.2588.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| motion_energy_quintile | 0.6498 | 0.3084 | Chance 0.2000; validation is 1.542x chance. |

The complete 41-session run trained on 872 trials and tested on 218 trials using CUDA. It finished successfully and produced `/app/train_decoder_full_out.txt`, `/app/sample_trials.png`, and `/app/predictions.png`.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
|----------|-------------------|------------------------|
| motion_energy_quintile (full) | Training balanced accuracy 0.6498; validation 0.3084; chance 0.2000 | Paper has no categorical accuracy. It predicts continuous motion and reports R², with stronger later-day decoding. |
| motion_energy_quintile (sample later sessions) | Training 0.7399; validation 0.3254 | Qualitatively consistent with stronger movement modulation after P11. |
| Continuous motion, same-day P8 example | N/A (different target/metric) | R²=0.26 (Figure 7B) |
| Continuous motion, same-day P14 example | N/A (different target/metric) | R²=0.69 (Figure 7B) |
| Continuous motion, cross-day early→early | N/A (different target/metric) | R²=0.15 (Figure 7F) |
| Continuous motion, cross-day early→late | N/A (different target/metric) | R²=0.11 (Figure 7F) |
| Continuous motion, cross-day late→late | N/A (different target/metric) | R²=0.61 (Figure 7F) |

- **Accuracy versus chance**: Full validation is 1.542x uniform chance and exceeds the requested 1.5x-chance investigation threshold; sample is 1.627x. The full score includes early sessions where the paper explicitly reports weak motion representation, so it is qualitatively reasonable.
- **Paper comparison**: The paper contains R² for continuous ridge regression, not class/balanced accuracy. Treating R²=0.69 as 69% accuracy would be mathematically invalid. All reported example values are tabulated above; plotted pooled distributions provide no additional exact categorical values. The reference provides a qualitative expectation—developmental improvement—which the later-session sample (0.3254) versus all-age full set (0.3084) does not contradict.
- **Three raw trials**: Independently reconstructed output labels for jm031 first session/trial 0, jm031 heavy-gap session/trial 19, and jm046 last session/trial 29; all exactly equal converted labels. The first trial contains only classes 2-4, which is valid local behavior, while each complete session and each held-out session subset contains all five classes.
- **Output variation**: Every full session is exactly 20% per class. Reproducing the decoder split gives training fractions `[0.2003,0.2000,0.1982,0.1969,0.2046]` and held-out fractions `[0.1988,0.2000,0.2073,0.2123,0.1816]`; balanced accuracy correctly compensates for modest whole-trial sampling imbalance.
- **Temporal synchronization**: Raw-output checks, processing plots, and trial concatenation prove index alignment. As a non-model check, population-mean activity versus motion class is positive in 36/41 sessions (median correlation 0.124, max 0.378), consistent with genuine alignment and the paper's variable age dependence. Full `predictions.png` visually shows predictions tracking portions of held-out class transitions.
- **Train/validation gap**: Ratio is 2.107, triggering investigation. `/app/cache/leakage_control.py` shuffles labels within each session while retaining identical neural/input data and decoder settings. It obtains training accuracy 0.3678 but validation 0.1997 (chance 0.2). The architecture can overfit training noise, explaining much of the gap, but cannot generalize shuffled labels. Original held-out accuracy is therefore not caused by train/test duplication or label leakage.
- **Potential preprocessing leakage**: Baseline filtering and session quintile edges are computed on full continuous sessions before splitting. They are unsupervised/reference preprocessing and the task explicitly requires bins "selected per session"; no model fit, neural-output mapping, trial duplication, or interpolation uses held-out labels. Changing this would violate the specified/reference processing rather than repair leakage.
- **Filtering review**: All neural rows already satisfy reference `iscell>0.5` and complete-track curation. Paper-style neuropil/baseline processing and 10-frame averaging were independently rechecked; no alternative filtering change is justified by accuracy.

### Issues Found and Resolved
- **Overly strict audit tolerance**: Initial audit demanded each held-out class be within ±1% of 20%, ignoring that whole 60-s trials are split. Replaced with a support/sampling criterion; all classes are present and deviations are modest. This was an audit-test correction, not a dataset change.
- **Audit import path**: The leakage-control script initially could not import `/app/decoder.py` from `/app/cache`; explicitly added `/app` to its analysis-only import path and reran successfully.
- **Conversion issues**: None found. Consequently no conversion/retraining iteration was warranted. `/app/cache/accuracy_audit_out.txt` ends `ACCURACY AUDIT PASSED`; `/app/cache/leakage_control_out.txt` ends `LEAKAGE NEGATIVE CONTROL PASSED`.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

Created `/app/README.md` with dataset description, loading example, target structure, processing summary, key statistics, reproducibility commands, and validation/decoder results. Created `/app/cache/README_CACHE.md`; all investigation scripts, their logs, rendered paper pages, and generated bytecode are under `/app/cache`. Required conversion, validation, training, and plot artifacts remain at the project root. Final required-file and nonempty-file checks passed, and all workflow steps are marked COMPLETE.
