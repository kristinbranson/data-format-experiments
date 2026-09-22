# Dataset Conversion Notes

## Overview
- **Dataset**: Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p (provided paper/code/data)
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

Environment check: Python 3.13.15; NumPy 2.4.4; PyTorch 2.6.0+cu124 imported successfully. The required checkpoint `ls -la /app/CONVERSION_NOTES.md` succeeded.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `check_nplanes` | `code/track2p/io/s2p_loaders.py` | LOADING | Counts `suite2p/plane*` folders and requires every longitudinal dataset to have the same plane count. |
| `load_all_imgs` | `code/track2p/io/s2p_loaders.py` | LOADING | Loads Suite2p `ops.npy`, functional/anatomical mean images, channel count, and checks channel consistency. |
| `load_all_ds_stat_iscell` | `code/track2p/io/s2p_loaders.py` | CURATION | Loads `stat.npy` and `iscell.npy`; keeps Suite2p labels when threshold is `None`, otherwise probability strictly greater than `iscell_thr`. |
| `run_t2p` | `code/track2p/t2p.py` | PROCESSING | Orchestrates loading, image registration, ROI transformation, optimal matching, saving, and plots. |
| `get_all_ds_assign` | `code/track2p/match/loop.py` | PROCESSING | Hungarian pair assignment from ROI costs, followed by Otsu/minimum thresholding of intersection-over-union. |
| `get_all_pl_match_mat` | `code/track2p/match/loop.py` | CURATION | Propagates pairwise matches across consecutive recordings and reports neurons tracked across all days. |
| `generate_suite2p_indices` | `code/track2p/t2p.py` | PROCESSING | Maps indices in the filtered ROI arrays back to original Suite2p row indices. |
| `save_in_s2p_format` | `code/track2p/t2p.py` | CURATION | Keeps rows without any missing longitudinal match and subsets `F`, `Fneu`, `spks`, `stat`, and `iscell` consistently. |
| `DataManagement.import_files` | `code/track2p/gui/data_management.py` | LOADING | Loads tracked Suite2p traces; uses only all-day matches, identical `iscell` filtering, and optional manual-curation vector. |
| `DataManagement.F_processing` | `code/track2p/gui/data_management.py` | PROCESSING | Computes `Fc=F-neucoeff*Fneu` (default `neucoeff=0`) and subtracts a Suite2p-style maximin baseline after Gaussian smoothing; despite GUI label, it does not divide by baseline. |

### Notes
- The repository is Track2p cell-tracking software, not the paper's downstream behavior-analysis pipeline. It contains no motion-energy loading or temporal-alignment implementation.
- Native inputs are Suite2p products (`F`, `Fneu`, `spks`, `stat`, `iscell`, `ops`) or a reduced NumPy format (`F`, FOV, ROI masks). The reduced converter assigns 30 Hz only as an explicit assumption.
- Default/reference Track2p cell curation is `iscell[:,1] > 0.5`; examples emphasize that downstream trace indexing must use the same threshold used for matching. If `iscell_thr=None`, the binary Suite2p label is used.
- Reference downstream examples select rows tracked on every recording (`~np.any(match_mat == None, axis=1)`) before extracting traces. Optional manual curation is stored as `vector_curation_plane_*`.
- Track2p saves both raw `F` and Suite2p `spks`; it does not mandate one trace type for downstream activity analyses. The GUI offers raw `F`, `spks`, and a baseline-subtracted fluorescence display.
- No electrophysiology is present. Delta-F/F is not inherently required by the tracking algorithm; whether fluorescence processing is needed must be resolved from the supplied data and paper.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`data/` is 16 GB and contains `README.md`, a loader notebook, and six subject folders (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`). Each subject contains chronologically named daily recording sessions. Each session contains one imaging plane in `suite2p/plane0/` and behavior in `move_deve/`.

- `F.npy`, `Fneu.npy`, `spks.npy`: float32 `(tracked_neurons, imaging_frames)`. Rows are already restricted to cells successfully tracked across every day and are in matched longitudinal order.
- `iscell.npy`: float64 `(tracked_neurons, 2)`. Every binary label is 1 and every probability is strictly above 0.5 (global observed minimum >0.500), confirming that the packaged traces have already undergone Track2p's 0.5 cell filter.
- `stat.npy`: object array of ROI dictionaries, one per tracked neuron.
- `ops.npy`: Suite2p settings and registration outputs. All sessions report one plane, 30 Hz, two channels, `neucoeff=0.7`, maximin baseline parameters (`win_baseline=60 s`, `sig_baseline=10`, percentile 8), and `badframes` aligned to imaging frames.
- `motion_energy_glob.npy`: uint64 global motion energy for each available camera frame.
- `tstamps.npy`: float64 camera timestamps, starting at zero; the stored scale has nominal differences about `3.36e-5` (corresponding to about 33.6 ms when expressed in seconds with the dataset's x1000 scale).
- `interframe_int.npy`: float64 timestamp differences, length one less than the camera series; enlarged gaps identify dropped camera frames.
- `ground_truth.csv` exists for jm038, jm039, and jm046 and contains manually identified cells across days for Track2p evaluation, not neural/behavior samples required for this decoder.
- The native recordings are continuous and contain no trial definitions. Splitting into non-overlapping 60-second trials is therefore a downstream requirement, not native curation.
- Camera-frame deficits relative to neural frames occur in 10 sessions and total 276 missing frames: jm031 days 3/4/5 have 2/3/116; jm032 days 3/4/5 have 2/2/148; jm039 day 5 has 1; jm040 day 4 has 1; jm046 day 7 has 1. Timestamp gap counts agree exactly for these deficits. jm046 days 3/5/6 contain several unusually large timestamp intervals despite full array lengths; this is retained as an edge case for alignment review.
- Suite2p reports only one bad neural frame in the entire dataset (`jm032/2023-10-24_a`); all other `badframes` values are false.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 2,998 unique longitudinal cell tracks; 20,445 session-neuron recordings |
| Neurons / session | jm031 221; jm032 370; jm038 685; jm039 746; jm040 541; jm046 435 (mean 498.66 across sessions) |
| Subjects | 6 |
| Sessions / subject | jm031 7; jm032 7; jm038 7; jm039 7; jm040 6; jm046 7 |
| Trials (total) | None natively (continuous recordings); 1,090 complete non-overlapping 60-s segments implied by the requested split |
| Trials / session | None natively; requested conversion yields 20 for each 36,000-frame session and 30 for each 54,000-frame session |

There are 41 sessions. Every neural stream within a session has identical dimensions. jm031 and jm032 sessions contain 36,000 frames = 1,200 s at 30 Hz; all other sessions contain 54,000 frames = 1,800 s at 30 Hz. No imaging tail is left after exact 60-second splitting.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Figure 5: A 285, B 376, C 799, D 728, E 541, F 411; mean stated as 526 ± 190 SD per mouse | “On average 526 (± 190 std) neurons per mouse were successfully tracked across all days” |
| Neurons / session | Same tracked population on every day within a mouse | “We used the subset of cells successfully tracked across all days for all our future analyses.” | 
| Subjects | 6 | “full dataset of 6 mice” |
| Sessions / subject | At least 6 consecutive daily sessions; Figure 5 shows five mice with 7 days and one with 6 | “imaged daily for a minimum of 6 consecutive days” |
| Trials (total) | N/A in paper; recordings are continuous | Methods describe continuous sessions and CV blocks rather than experimental trials |
| Trials / session | N/A in paper | Decoder CV splits were consecutive 2-minute blocks; requested conversion instead requires 60-s trials |
| Neural data time bin | 33.333 ms native; 333.333 ms for paper decoding | “Imaging rate was 30 Hz”; decoding averaged “10 consecutive timestamps” |
| Behavior data time bin | 33.333 ms native; 333.333 ms for paper decoding | “Videos were recorded at 30 Hz”; decoding averaged “10 consecutive timestamps” |
| Reward rate | N/A | Spontaneous, sensory-minimized recordings; no rewards or task trials |
| Session duration | Paper states 20 min | “each session lasted 20 minutes” |
| Brain region | Layer 2/3 barrel cortex, 100–200 µm depth | Chronic 2-photon methods |
| Tracked fraction | 33% ± 11% SD of first-day detected neurons | Results text |
| Example same-day motion decoding | R² = 0.25, 0.06, 0.10, 0.32, 0.70, 0.70, 0.69 from P8–P14 | Supplementary Figure 7A labels |
| Example cross-day motion decoding | R² examples: early→early 0.15, early→late 0.11, late→late 0.61 | Figure 7F labels |


### Processing Details
- Imaging and infrared videography were both acquired at 30 Hz, with microscope acquisition triggering camera frames for direct synchronization.
- Motion energy is already provided: squared pixel differences between each pair of consecutive video frames, summed over all pixels to a scalar at each time point. It is a continuous proxy for global movement/arousal; long whole-body movements dominate over small twitches.
- Imaging was processed separately per recording with Suite2p: motion correction, ROI detection, trace extraction, and spike deconvolution.
- The paper calls its neural representation “baseline corrected fluorescence traces as our dF/F,” using default Suite2p parameters. Supplied Suite2p settings specify neuropil coefficient 0.7, maximin baseline, 60-s window, Gaussian sigma 10 frames, and percentile 8. Reference GUI code implements the maximin baseline subtraction (`Fc - Flow`) but misleadingly labels this dF/F and defaults its neuropil coefficient to zero. This implementation discrepancy is carried to Step 4.
- For event rates, traces were averaged in 10-frame bins before peak detection (height and prominence ≥1 SD); event rate was peaks/minute.
- For decoding, both fluorescence and behavior were averaged over the same non-overlapping 10 consecutive timestamps (effective 3 Hz). The paper used ridge regression, nested 5-fold inner/outer CV, and consecutive 2-minute splitting units. Cross-day models were fit per day with its selected lambda.
- The requested decoder differs: motion energy must be classified into five per-session equal-percentile bins and sessions split into 60-s trials. The paper reports continuous-regression R² rather than categorical accuracy, so it supplies qualitative and R² benchmarks but no directly comparable five-class accuracy.

### Curation Steps

**Neuron curation rules**:
Suite2p cell probability strictly above 0.5, followed by Track2p across-day matching; subsequent functional analyses include only tracks present on every recording day. Anatomical-channel affine registration was the default tracking condition. Manual curation was available but the paper states it was not strictly necessary under these conditions.

**Trial curation rules**:
No experimental trials were defined. Paper decoder splits were based on consecutive 2-minute blocks. Missing camera frames are acknowledged by the data documentation and may be treated as missing or interpolated; the paper provides no further exclusion rule.

### Decoders Trained
| Decoded variable | Accuracy |
| Continuous global motion energy, same-day ridge regression | R² varies strongly by age; example P8–P14: 0.25, 0.06, 0.10, 0.32, 0.70, 0.70, 0.69 |
| Continuous global motion energy, cross-day ridge regression | Example: early→early 0.15, early→late 0.11, late→late 0.61; late-period representations were stable |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Cell filtering | Track2p uses `iscell[:,1] > iscell_thr` (default 0.5), propagates sequential matches, and packaged Suite2p output keeps only all-day matches | Every packaged `iscell` label is 1 and probability >0.5; rows and counts are constant across days | ROIs >0.5 and only neurons tracked across all days were analyzed | Fully consistent; no additional cell filter is appropriate because the release is already curated. |
| Tracked-cell counts | Output depends on the exact Track2p run/version and IoU threshold | A–F are 221, 370, 685, 746, 541, 435 (mean 499.7; 2,998 unique tracks) | Figure 5 gives 285, 376, 799, 728, 541, 411 (mean 523.3; text rounds to 526 ±190) | The released matrices are a different Track2p export/reprocessing from the manuscript figure. There is no mapping that could recreate absent paper rows, nor a principled basis to discard extra valid rows. Preserve every supplied curated cell and explicitly report the discrepancy. |
| Session duration | Suite2p `ops` gives 30 Hz and `nframes` | jm031/jm032 are 20 min; the other four mice are 30 min | Methods say every session was 20 min; example figures display 20 min | Preserve all complete source frames because the deliverable is the full converted dataset and the extra 10 min have aligned neural/behavior recordings and no invalid-period marker. Cropping would discard 18,000 valid samples from 27 sessions. Record per-session durations in metadata. |
| Fluorescence representation | GUI `F_processing` performs maximin baseline subtraction but defaults `neucoeff=0.0`; its output is subtraction, not literal division by F0 | `ops` consistently specifies Suite2p default `neucoeff=0.7`, maximin baseline, sigma 10 frames, window 60 s; raw `F`, `Fneu`, and deconvolved `spks` are supplied | Paper decoder uses baseline-corrected fluorescence (“dF/F”) with default Suite2p parameters | Use `Fc=F-0.7*Fneu`, Gaussian sigma 10 frames, min then max filters over 1,800 frames, and `Fc-Flow`. This follows paper/default Suite2p settings and the reference baseline algorithm; do not divide by `Flow` because the provided implementation does not. |
| Temporal denoising | No downstream decoder code is present, but GUI keeps native samples | All streams nominally 30 Hz | Decoder averages fluorescence and behavior in bins of 10 timestamps | Apply identical non-overlapping 10-frame means to neural and motion streams, yielding a common 3 Hz (333.333 ms) grid. |
| Camera gaps | Reference repository has no motion loader | 276 fewer camera frames over 10 sessions; enlarged intervals locate exactly the deficit. Three full-length jm046 sessions have timestamp anomalies without missing array elements | Camera is imaging-triggered at 30 Hz; release README says locate missing frames with timestamps and either mark missing or interpolate | For deficient sessions, reconstruct 30-Hz index positions from rounded interval/median ratios and linearly interpolate inserted values. Leave full-length sessions untouched as instructed by the README length criterion; their timestamp anomalies do not identify absent array entries. |
| Neural bad frames | Suite2p provides `ops['badframes']` | One marked frame total; its F/Fneu/spks values are finite and comparable to neighbors | No downstream exclusion described | Retain it; Suite2p has already processed it and 10-frame averaging reduces single-frame influence. Removing it would desynchronize behavior. |
| Motion output type | Paper uses continuous ridge regression | Continuous uint64 motion energy | Continuous motion regression with R² | Required task overrides this: discretize the aligned, 10-frame-averaged motion independently per session into five percentile classes. |

Final understanding: the release consists of already cell-curated, longitudinally matched layer 2/3 barrel-cortex traces plus frame-aligned global motion energy. The only necessary curation is repairing documented missing behavior frames. Paper-style baseline correction and 10-frame paired averaging should precede the task-required 60-s segmentation and five-class discretization.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `suite2p/plane0/F.npy`, `Fneu.npy`, `ops.npy` | `neural` | In neuron chunks: `Fc=F-0.7*Fneu`; Gaussian smoothing sigma 10 frames; 1,800-frame min then max baseline; subtract baseline; average each 10 native frames; split columns into 180-bin (60-s) trials; float32 | `DataManagement.F_processing`; paper Methods “baseline corrected fluorescence” and 10-timestamp averaging | Process the entire continuous session before trial splitting so filtering has no artificial trial-edge discontinuities. |
| Native frame indices and `ops['fs']=30` | `input[0]` | Mean elapsed times of each 10-frame block: `(10*k + 4.5)/30` s, preserved as global session elapsed time, then split with trials; shape `(1,180)`, float32 | Paper states 30 Hz and paired averaging | Input is not reset at each 60-s boundary because the requested variable is time since session beginning. |
| `move_deve/motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy` | `output[0]` | If behavior length is deficient, derive observed integer positions from rounded interframe interval / median interval and linearly interpolate missing values; average each 10 frames; compute session-specific 20/40/60/80% thresholds; `searchsorted(..., side='right')` to classes 0–4; split into `(1,180)` trials, int64 | Paper motion-energy definition and paired 10-timestamp averaging; release README missing-frame guidance | Quantile check across all 41 sessions gives exactly 20% per class; binned values are almost all unique, so ties do not distort bins. |
| Subject folder name | `subjects`, `subject_idx` | Subjects sorted lexically; each chronological session points to its subject index | Release README | `['jm031','jm032','jm038','jm039','jm040','jm046']`. |
| Recording location from Methods | `brain_regions`, `brain_region_idx` | One label; zero for every neuron/session | Paper Methods | Label: `barrel cortex (S1), layer 2/3`. |
| Variable meanings | `input_names`, `output_names`, `output_values` | Descriptive strings | Decoder task | Input `time elapsed from session start (s)`; output `motion energy percentile bin`; class names `0-20% (lowest)` through `80-100% (highest)`. |
| Processing/session facts | `metadata` | Store task, timing, transforms, trial policy, session IDs/counts/durations/gaps/quantile edges, and reference discrepancies | All sources | `time_bin_size=1000/3 ms`, alignment is each 60-s segment start, `off_start=0`, `off_end=60`. |

### Key Decisions
1. **All 41 sessions and full durations**: Preserve every valid supplied sample. This yields 1,090 trials: 14×20 plus 27×30. The paper-duration discrepancy is documented rather than silently cropping valid source data.
2. **Paper-style fluorescence, not raw F or spks**: Use neuropil-corrected maximin baseline-subtracted fluorescence because this is the neural signal named for paper decoding. `spks` remains an available alternative but was not the reported decoder input.
3. **No further neuron filtering**: The release is already restricted to >0.5-probability neurons tracked across all days, and all rows pass this criterion. Applying a second quality rule would diverge from the reference and lose data.
4. **Common 3 Hz time grid**: Ten-frame averaging is the paper's decoder denoising and gives a uniform 333.333-ms bin for every session. Exactly 180 bins form each requested 60-s trial.
5. **Behavior gaps interpolated before averaging**: Linear interpolation is explicitly permitted by the release README and avoids NaNs rejected by the validator. Integer positions derived from timestamp gaps reproduce the exact known deficits.
6. **Per-session quantiles after alignment/averaging**: This applies “selected per session” to the exact continuous values represented at decoder timepoints. Across source data it produces exactly balanced classes.
7. **Global elapsed-time input**: Time values continue across trial boundaries, preserving the requested session context. Each value is the mean acquisition time of the ten samples represented by that bin.
8. **Chronological deterministic ordering**: Subjects lexical, sessions chronological, trials chronological. `--sample` takes the first two sessions in this order for reproducibility.
9. **Dtypes**: float32 neural/input minimize the full pickle (estimated neural payload about 340 MB); output uses int64 categorical labels.

### Planned Sanity Checks
- [ ] Directly reload original F/Fneu for selected neuron/timepoints and use `np.allclose` against independently calculated maximin-corrected, 10-frame-averaged converted values.
- [ ] Directly reconstruct original motion for an intact and a dropped-frame session, average, reapply saved quantile edges, and use `np.allclose` against converted output labels.
- [ ] Use `np.allclose` to compare converted elapsed-time inputs to the independent analytical frame-time means for multiple trials, including the final trial.
- [ ] Assert F/Fneu/spks/stat/iscell row consistency, constant neuron count within subject, and all `iscell` probabilities >0.5.
- [ ] Assert repaired motion length equals neural length, downsampled neural/motion/time lengths match, every trial is `(neurons,180)/(1,180)/(1,180)`, and no remainder is dropped.
- [ ] Assert each session has all five classes and exactly 20% class fraction; overall expected counts are 39,240 time bins per class.
- [ ] Assert expected totals: 6 subjects, 41 sessions, 2,998 unique longitudinal tracks, 20,445 session-neurons, 1,090 trials, and 196,200 decoder timepoints.
- [ ] Review processing plots for baseline behavior, repaired/aligned motion, paired downsampling, quantile thresholds, and 60-s boundaries.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with the required CLI (`--full` default, `--sample`, `--show-processing`). The script discovers subjects/sessions deterministically, validates all native stream/ROI dimensions and curation, repairs documented motion gaps, performs continuous-session fluorescence processing in neuron chunks, applies paired 10-frame means, constructs exact quintile labels, creates 60-s trials, validates known full-data totals, and serializes with the highest pickle protocol. Syntax compilation and CLI help both succeeded.

Processing plots contain eight audit panels: raw F/Fneu; neuropil correction and maximin baseline; baseline subtraction and temporal averaging; processed raster; motion repair; paired neural/motion alignment; quantile histogram; categorical output with trial boundaries.

Code inefficiencies identified:
Loading all F/Fneu into writable arrays simultaneously would require several hundred MB per session in addition to filtering intermediates. Repeating filters separately per trial would be both slow and scientifically incorrect at boundaries. `ops.npy` is large and is loaded only once per session.

Code speedups added:
Memory-mapped source arrays; 64-neuron processing chunks; vectorized SciPy filters across neurons; vectorized reshape/mean downsampling; session processing once before slicing; direct list construction and float32 outputs. No parallel disk reads were added because the 16-GB source is I/O-heavy and concurrent session filtering would multiply memory pressure.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 221 unique longitudinal cells; 442 session-neurons |
| Neurons / session | 221, 221 |
| Subjects | 6 identifiers retained; sample sessions both jm031 |
| Sessions / subject | jm031: 2 in sample |
| Trials (total) | 40 |
| Trials / session | 20, 20 |
| Time elapsed input range | [0.15, 1199.8167] s (validator rounds to [0.2, 1199.8]) |
| Trial time bins | exactly 180 for every stream/trial |
| Motion output distribution | [0.200, 0.200, 0.200, 0.200, 0.200] overall and in each session |
| Output range | [0, 4] |

Manual inspection confirmed all required dictionary keys, `(221,180)/(1,180)/(1,180)` neural/input/output shapes, float32/float32/int64 dtypes, chronological time continuity through the final trial, complete metadata, and exact class counts `[720,720,720,720,720]` per sample session. `/app/sample_data.pkl` is 6.2 MiB. The official verifier reported a valid format with no errors or warnings.

### Processing Plots Review
Both processing plots were visually reviewed. Raw fluorescence and neuropil are finite; the maximin baseline follows slow drift without following transients; baseline subtraction preserves calcium transients; 10-frame averaging visibly denoises them; processed rasters retain structured population events. Raw and aligned motion overlap exactly in these two intact sessions. The paired neural/motion panel shows synchronized peaks, quintile boundaries span the skewed motion distribution, and 60-s cuts occur without resetting the categorical series. No anomalies or temporal offsets were observed.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| 64-neuron vectorized chunks and memory maps | Core fluorescence conversion was 0.37 s/session in both sample sessions. |
| Process full session once, then slice trials | Avoids 20 repeated filtering calls and trial-edge artifacts. |
| Plot only first two selected sessions | Diagnostics added about 1.1–1.2 s/session only in requested plot mode. |

| Step | Time / Session | Estimated Total Time |
| Core neural processing | 0.37 s for 221 neurons; scales approximately with neuron×frame count | about 20–30 s for all 41 sessions |
| Complete sample conversion with plots | 1.5 s/session | plotting excluded from full command; not extrapolated |
| Full conversion + pickle save | N/A | conservatively under 1 minute, far below 15-minute optimization threshold |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| Motion energy percentile bin | 0.5337 | 0.2904 |

Training completed on CUDA. Loss decreased from 53.1707 at epoch 1 to 1.2752 at epoch 200 (test loss 2.3832). Validation balanced accuracy is above uniform chance (0.20) for the only output, satisfying the sample criterion. The train/validation ratio is 1.84, which is not interpreted here because the sample contains only two early sessions and eight held-out trials; full-data behavior is reviewed in Steps 11–12.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 414,459,590 bytes (395.3 MiB)
- `verification_full_out.txt`: created; official validator reports no errors or warnings
- `conversion_full_out.txt`: created; computation 47.69 s, serialization 0.32 s, total 48.01 s

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | Figure 5 sum 3,140 unique tracks (text mean rounds to 526/mouse) | Data-dependent; all-day rows after filtering | 2,998 unique tracks; 20,445 session-neurons | 2,998 unique; 20,445 session-neurons | Yes vs released data; documented paper-export discrepancy |
| Mean neurons/session | Figure values mean 523.3 | Data-dependent | 498.66 | 498.66 | Yes vs released data |
| Subjects | 6 | N/A | 6 | 6 | Yes |
| Sessions | Minimum 6/mouse; Figure 5 totals 41 | Loads all supplied daily paths | 41 (7,7,7,7,6,7) | 41 (7,7,7,7,6,7) | Yes |
| Trials (total) | None; paper says 20-min continuous sessions | None | 1,090 requested 60-s segments from all valid durations | 1,090 | Yes vs released data/task |
| Trials/session (mean) | N/A | N/A | 20 for first 14 sessions; 30 for remaining 27; mean 26.59 | Same | Yes |
| Time elapsed range | 30 Hz; paper examples to 20 min | N/A | Binned centers 0.15–1199.8167 or 1799.8167 s | [0.15, 1799.8167] s overall | Yes |
| Neural range | Not numerically reported | Baseline subtraction can be negative | Finite baseline-corrected fluorescence expected | [-294.991, 2808.033], finite | Plausible |
| Motion class distribution | Continuous motion in paper | N/A | Task-defined session quintiles | [0.200,0.200,0.200,0.200,0.200], counts 39,240 each | Yes |

No samples were lost: 36,000/54,000 native frames become exactly 3,600/5,400 paired bins and 20/30 complete trials. The converted total is 196,200 decoder timepoints. Metadata sums to all 276 interpolated camera frames. Spot checks covered the first session, largest-gap session, sole Suite2p-badframe session, first 30-min session, first jm040 session, and final session; all had correct shapes, global time endpoints, balanced labels, and gap counts.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Searched `verification_full_out.txt` for error/warning/invalid/NaN/Inf. The only match is “valid, no errors or warnings.” All dimensions, ranges, class values, subject indices, and region indices pass the official validator.
2. **Independent neural check (`np.allclose`)**: `/app/sanity_checks.py` directly loaded original F and Fneu (without importing conversion code), independently computed `F-0.7Fneu`, Gaussian/maximin baseline subtraction and 10-frame means, and compared neurons 3 and 17 over all 3,600 bins of session 0. `np.allclose(rtol=1e-6, atol=1e-5)` passed. Explicit trial 5/time 10/neuron 3 value: 6.144436.
3. **Independent input check (`np.allclose`)**: Analytical ten-frame mean times were compared over every bin of the first 20-min and final 30-min sessions. `np.allclose(atol=1e-6)` passed, including all trial transitions.
4. **Independent output check (`np.allclose`)**: Raw motion/timestamps were loaded and alignment, 10-frame averaging, quantiles and labels independently recomputed for an intact session, the 116-missing-frame session, and the final 1-missing-frame session. Full labels and saved quantile edges passed `np.allclose`; explicit session 4/trial 5/time 10 label was 0.
5. **Reference code comparison**:
   - Loading: converter memory-maps `F`/`Fneu` and loads Suite2p metadata as the reference `DataManagement.import_files`/loaders do.
   - Neuron filtering: reference loaders apply `iscell[:,1] > 0.5`, and `save_in_s2p_format` takes only all-day match rows. Converter verifies (rather than repeats) these filters because packaged rows already satisfy them; raw/converted row counts match exactly.
   - Temporal alignment: paper camera triggering plus release timestamp-gap guidance maps to `align_motion`; all 276 length deficits are explained by interval ratios and filled only at those indices.
   - Binning: converter uses non-overlapping ten-sample means for both streams, exactly as Methods specifies, then task-required 60-s cuts. No reference operation is performed separately at trial boundaries.
   - Input construction: elapsed session time is task-specific (not a paper decoder variable); it is calculated at the mean time of represented native frames and remains global across trials.
   - Output construction: paper continuous motion is preserved through alignment/averaging, then the task-required per-session percentile transform is applied. Every session is exactly balanced.
6. **Key statistics**: Independent source inventory passed for 6 subjects, 41 sessions, 20,445 session-neurons, and 2,998 unique tracks. Converted totals passed for 1,090 trials and 196,200 bins. Paper-count and duration differences are fully enumerated in Steps 4 and 9. The paper’s 33±11% tracked fraction cannot be recomputed from this release because untracked first-day ROIs were intentionally omitted.
7. **Edge cases/off-by-one audit**: Every raw frame is consumed exactly once; every trial has 180 bins; first/last elapsed times are 0.15/1199.8167 s or 0.15/1799.8167 s; adjacent trials differ by exactly 1/3 s without overlap or gap; labels are always 0–4. Source/converted dimensions match for every session. The one Suite2p bad frame is finite and intentionally retained. Full-length jm046 timestamp anomalies are intentionally not expanded because no behavior samples are absent and the release documentation defines missing frames via length mismatch.
8. **Class/statistical checks**: Each session has exactly 20% in each output class; full counts are `[39240,39240,39240,39240,39240]`. Neural values are finite with full range `[-294.991, 2808.033]`, consistent with baseline-subtracted (not normalized/divided) fluorescence.

### Issues Found and Resolved
- No new conversion defect was found, so no iteration/re-conversion was required in Step 10.
- Previously identified source/reference discrepancies (cell counts, 20- versus 30-minute durations, misleading “dF/F” nomenclature, and camera deficits) were resolved before implementation and are documented in Step 4. Each implemented decision agrees with the provided data and reference processing or is explicitly required by the decoder task.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes. Training loss fell from 62.4325 (epoch 1) to 1.1098 (epoch 200), with the lowest reported checkpoint 1.1032 at epoch 190; test loss was 2.7439. Execution completed successfully on CUDA.
- Training/validation split: 872/218 complete 60-s trials, stratified by the script's per-session trial split behavior.
- The generated sample/prediction plots were visually reviewed. Neural transients and categorical motion structure are present; predictions track part of the class dynamics without any flat-signal or indexing artifact.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| Motion energy percentile bin | 0.6045 | 0.3069 | Uniform chance 0.2000; validation is 1.5345× chance |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| Five-class per-session motion-energy percentile (required task) | Validation balanced accuracy 0.3069; training 0.6045; chance 0.2000 | Not reported as a categorical task. Achieved validation is 1.5345× chance and exceeds the requested 1.5×-chance investigation threshold. |
| Continuous same-day motion decoding (paper task) | Independent diagnostic on converted jm039 P14 first 20 min: held-out ridge R² 0.7184 (a full-30-min diagnostic gave 0.6253) | Example mouse P8–P14 R²: 0.25, 0.06, 0.10, 0.32, 0.70, 0.70, 0.69. The directly comparable P14 diagnostic is within 0.0284 of the reported 0.69. |
| Continuous cross-day motion decoding (paper task) | Not part of the required dataset validator/trainer, which learns session-specific neural projections and performs within-session trial holdout | Reported examples: early→early 0.15, early→late 0.11, late→late 0.61. No categorical cross-day benchmark exists. |

**Accuracy versus chance**: 0.3069 exceeds both chance (0.20) and 1.5× chance (0.30). The context-only elapsed-time diagnostic reached only 0.2147, showing that the meaningful gain is not explained by the required time input alone.

**Train/validation gap**: The ratio is 0.6045/0.3069 = 1.97, so it was investigated. The validator selects disjoint complete trials within every session; an independent split audit confirmed no shared trial indices. No output information is present in neural or input fields. The fixed decoder learns 41 session-specific 100-dimensional projection matrices (about two million projection weights) and its train/test losses are 1.11/2.74, providing direct evidence of model overfitting. Changing the conversion to normalize, leak labels, or discard weak early sessions could narrow the gap but would violate the reference processing/full-data requirement. The held-out performance and paper-style P14 ridge diagnostic both support the current representation.

**Required low-accuracy debugging checks**:

1. Direct raw-output reconstruction passed for session/trial 0/0, the largest gap session 4/5, and final session 40/29. Their within-trial class counts were respectively `[0,0,66,97,17]`, `[144,9,12,14,1]`, and `[77,44,10,5,44]`.
2. Neural/output synchronization was reviewed in processing and full prediction plots. A PCA audit on late jm039 gave |zero-lag PC1/motion correlation| 0.236 and peak |correlation| 0.259 at a five-bin calcium lag; the strong held-out ridge R² confirms useful alignment. The lag direction is consistent with slow GCaMP response rather than a frame-index error.
3. Output variation is exact at session level (20% each). Across trials, 80.3% contain all five classes, every trial contains at least two, and strong within-trial dominance (maximum 99.4%) is expected because percentiles are selected per session, not per 60-s segment. Balanced loss/accuracy address this.
4. Neural filtering was rechecked: every packaged row has Suite2p label 1 and probability >0.5, and every row is an all-day Track2p output. No low-quality/unmatched cells are reintroduced.
5. Neural and behavior processing were independently reproduced from raw files with `np.allclose`; reference-code and Methods comparisons are detailed in Step 10.

### Issues Found and Resolved
- **Potential time-only shortcut**: ruled out by 0.2147 time-only balanced accuracy versus 0.3069 full decoder.
- **Potential split leakage**: ruled out; training and validation use disjoint whole trials in every session. Full-session baseline/quantile estimation is unsupervised and required by the paper/task, and no labels enter neural processing.
- **Large train/validation ratio**: attributed to the fixed high-capacity decoder after direct leakage checks; it is not a conversion defect. Reference-preserving conversion was retained.
- **Potential weak alignment suggested by a population-mean check**: the paper uses PCA/ridge, not the mean. Correct PCA/ridge diagnostics show the expected late-session representation and P14 R² close to the paper. No data change was warranted.
- No conversion change was made in Step 12, so re-running Steps 9–12 was unnecessary.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created with dataset description, loading example, format, statistics, reproduction commands, decoder result, and source discrepancies
- [x] cache/ folder created with `README_CACHE.md`
- [x] All investigation scripts, outputs, rendered paper pages, and Python bytecode cache organized under `cache/`; required deliverables and validation plots/logs remain at project root

Final required-file audit passed for all eleven requested deliverables. Every workflow step is marked COMPLETE. `pypdf` and `pypdfium2` were installed solely to read and render the supplied local paper after no system PDF text/render utility was found. No external project files were inspected.
