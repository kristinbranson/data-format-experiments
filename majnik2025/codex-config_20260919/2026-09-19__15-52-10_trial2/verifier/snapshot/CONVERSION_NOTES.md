# Dataset Conversion Notes

## Overview
- **Dataset**: Track2p longitudinal mouse barrel-cortex two-photon imaging dataset
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

Environment check: Python 3.13.15, NumPy 2.4.4, and PyTorch 2.6.0+cu124 import successfully. Checkpoint `ls -la /app/CONVERSION_NOTES.md` passed.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `run_t2p` | `code/track2p/t2p.py` | PROCESSING | Orchestrates plane discovery, Suite2p loading, image registration, ROI registration/matching, saving, and plots. |
| `load_all_imgs` / `load_all_ds_stat_iscell` | `code/track2p/io/s2p_loaders.py` | LOADING/CURATION | Loads Suite2p images/ROI metadata and retains `iscell[:,0] == 1`, or probability `iscell[:,1] > iscell_thr`. |
| `get_all_ds_assign` | `code/track2p/match/loop.py` | CURATION | Hungarian assignment of registered ROI pairs followed by an automatically estimated Otsu (default) IoU threshold. |
| `get_all_pl_match_mat` | `code/track2p/match/loop.py` | PROCESSING | Propagates accepted matches sequentially across recordings; unmatched later days remain `None`. |
| `generate_suite2p_indices` | `code/track2p/t2p.py` | PROCESSING | Converts indices in iscell-filtered arrays back to original Suite2p indices. |
| `save_in_s2p_format` | `code/track2p/t2p.py` | PROCESSING | Keeps rows present on every recording and saves matched `F`, `Fneu`, `spks`, ROI, and iscell arrays. |
| `DataManagement.import_files` | `code/track2p/gui/data_management.py` | LOADING/CURATION | Loads matched traces using the exact Track2p iscell threshold and all-day-complete match rows. |
| `DataManagement.F_processing` | `code/track2p/gui/data_management.py` | PROCESSING | Optional fluorescence baseline correction: `Fc=F-neucoeff*Fneu` (default `neucoeff=0`), Gaussian smoothing then 60-s maximin baseline, and returns `Fc-Flow` (despite GUI label dF/F0, no division by F0). |

### Notes
- The repository is primarily a longitudinal ROI registration/tracking package, not the behavioral-analysis pipeline. Its native inputs are Suite2p products (`ops.npy`, `stat.npy`, `iscell.npy`, `F.npy`; optionally `Fneu.npy` and `spks.npy`).
- Default `iscell_thr=0.50`. Track2p indexing is defined after this filter, so downstream use must apply the same threshold before match-matrix indexing.
- Longitudinal matched output retains only match-matrix rows without `None` when exporting matched Suite2p data. Manual GUI curation can store `vector_curation_plane_*`, but it is not applied automatically during file import.
- The demo visualizes raw `F` after per-cell z-scoring; optional GUI "dF/F0" is baseline-subtracted fluorescence rather than a ratio. Therefore delta-F/F is not intrinsically required by Track2p. The choice of the supplied neural stream will be resolved from the provided dataset and paper in Steps 2–5.
- The `npy` compatibility loader assumes already-curated cells, creates all-one `iscell`, and assumes 30 Hz; these are compatibility defaults, not evidence about the supplied experiment.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Six subject directories (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`) contain chronologically named daily session directories. There are 41 sessions total: 7 each except 6 for `jm040`.
- Each session has one Suite2p plane with `F.npy`, `Fneu.npy`, `spks.npy`, `iscell.npy`, `stat.npy`, and `ops.npy`, plus `move_deve/{motion_energy_glob,tstamps,interframe_int}.npy`.
- Per subject, neural rows are Track2p-matched cells present across all supplied days. Row identity is stable across sessions of that subject. All saved `iscell` rows have class 1 and probability >0.5 because curation/matching already occurred before this release.
- `F`, `Fneu`, and `spks` are float32 `(neurons, frames)` arrays. `motion_energy_glob` is uint64 `(video_frames,)`; `tstamps` is float64 and `interframe_int == np.diff(tstamps)`. Suite2p `ops['fs']` is 30 Hz and `ops['nframes']` matches neural columns.
- Sessions for `jm031`/`jm032` contain 36,000 neural frames (1,200 s); all other sessions contain 54,000 (1,800 s). Motion vectors are equal length except nine sessions missing 1–148 video samples. Timestamp gaps locate those missing samples exactly in the sessions where vector length is short. Three `jm046` sessions have 3–10-frame timestamp gaps despite equal vector/neural lengths; timestamps therefore remain the authoritative behavior clock.
- Timestamp units are 1000 s per stored unit: median stored interframe interval is about `3.359e-5`, corresponding to about 0.03359 s (29.77 Hz). Behavior must be aligned to the 30-Hz imaging grid using timestamps, rather than simply truncated or paired by index.
- `ground_truth.csv` exists for `jm038`, `jm039`, and `jm046`; it contains manually annotated cross-day ROI identities in the original, pre-Track2p arrays and is evaluation metadata rather than an additional behavioral variable.
- `/app/data/load_data.ipynb` directly loads the released raw fluorescence `F` and advises either computing dF/F as described in the paper or using `spks.npy` for analysis. It uses row z-scores only for visualization.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 2,998 unique Track2p-matched neurons across subjects; 20,445 neuron-session rows |
| Neurons / session | 221, 370, 685, 746, 541, or 435 depending on subject; mean 498.66 across 41 sessions |
| Subjects | 6 |
| Sessions / subject | 7, 7, 7, 7, 6, 7 (41 total) |
| Trials (total) | Native data are continuous sessions, no trials; planned 60-s splitting yields 1,090 complete trials |
| Trials / session | 20 for 36,000-frame sessions; 30 for 54,000-frame sessions |

Available variables: raw somatic fluorescence (`F`), neuropil fluorescence (`Fneu`), Suite2p deconvolved activity (`spks`), cell/ROI metadata (`iscell`, `stat`), imaging metadata/FOV (`ops`), global video motion energy, camera timestamps, and inter-frame intervals. No stimulus, reward, choice, or other task variables are present because recordings capture spontaneous behavior.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Figure 5B shows 285, 376, 799, 728, 541, 411 tracked cells (3,140 summed mouse-level tracks) | “On average 526 (± 190 std) neurons per mouse were successfully tracked across all days” |
| Neurons / session | Same tracked population each day within a mouse; mean stated as 526 ± 190 across mice | “We used the subset of cells successfully tracked across all days for all our future analyses.” |
| Subjects | 6 | “full dataset of 6 mice” |
| Sessions / subject | Minimum 6 consecutive daily sessions from P7–P14 | “imaged daily for a minimum of 6 consecutive days” |
| Trials (total) | N/A in paper (continuous recordings) | Decoder cross-validation splits used consecutive 2-minute blocks. |
| Trials / session | N/A in paper | Sessions analyzed continuously; target conversion imposes 60-second trials. |
| Neural data time bin | 10 imaging frames = 1/3 s for decoding | “averaging in bins of 10 consecutive timestamps”; imaging rate 30 Hz |
| Behavior data time bin | 10 video samples = approximately 1/3 s for decoding | Same 10-timestamp averaging applied to behavior traces; video 30 Hz |
| Reward rate | N/A | Spontaneous behavior under sensory-minimized conditions, no operant task/reward. |
| Motion metric | Sum of squared pixel differences between consecutive video frames | “computed their pixelwise difference… squared… and summed across pixels” |
| Imaging field / region | 720×720 µm, 512×512 pixels, layer 2/3 barrel cortex, depth 100–200 µm | Chronic two-photon imaging Methods |
| Session duration | 20 minutes | “each session lasted 20 minutes” |


### Processing Details
- Imaging and infrared videography were acquired at 30 Hz; microscope acquisition triggered camera frames for synchronization.
- Calcium data were processed separately per session with Suite2p: motion correction, ROI detection, fluorescence extraction, and spike deconvolution. Subsequent analyses used Suite2p-default baseline-corrected fluorescence, referred to in the paper as dF/F.
- Suite2p-default parameters stored in the release are `neucoeff=0.7`, maximin baseline, `sig_baseline=10` frames, and `win_baseline=60 s`. The corresponding operation is neuropil correction `Fc=F-0.7*Fneu`, Gaussian smoothing for baseline estimation, minimum then maximum filtering over 60 s, and subtraction of that baseline. As in Suite2p/Track2p code, this is baseline-subtracted fluorescence rather than literal division by F0.
- Paper decoding averaged both baseline-corrected fluorescence and motion traces over non-overlapping groups of 10 consecutive timestamps, then used ridge regression. Same-day nested cross-validation used 5 inner and 5 outer folds, with splits on consecutive 2-minute blocks.
- The paper’s decoder predicted continuous motion and reported R², not categorical accuracy. Figure 7 example values include early same-day R²≈0.25–0.26 and late same-day R²≈0.69; supplementary example same-day R² by P8–P14 is 0.25, 0.06, 0.10, 0.32, 0.70, 0.70, 0.69. Performance rises after P11. These values provide qualitative alignment expectations but are not directly comparable to the requested 5-class balanced accuracy.

### Curation Steps

**Neuron curation rules**:
Suite2p cell probability strictly greater than 0.5, followed by Track2p longitudinal matching. Track2p uses affine registration (anatomical tdTomato channel for the baseline analysis), Hungarian assignment maximizing IoU, Otsu filtering of the assigned-pair IoU distribution, and retains only cells tracked across all sessions. The released Suite2p arrays already contain this final matched subset, so re-running cell detection/tracking is neither necessary nor possible from the reduced release.

**Trial curation rules**:
The reference analysis treated recordings as continuous and split decoder cross-validation on consecutive 2-minute blocks. For the requested target, sessions will instead be divided into consecutive complete 60-second trials. The paper states and plots a 20-minute analysis duration; this is important because some released arrays contain 30 minutes and is reconciled in Step 4.

### Decoders Trained
| Decoded variable | Accuracy |
| Continuous global motion energy (same-day ridge regression) | R² varied with age; example P8≈0.25/0.26 and P14≈0.69, with supplementary P8–P14 values 0.25, 0.06, 0.10, 0.32, 0.70, 0.70, 0.69. |
| Continuous global motion energy (cross-day ridge regression) | Example early→early R²=0.15, early→late R²=0.11, late→late R²=0.61; strong late-to-late generalization. |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Tracked-cell counts | Track2p exports all-day-complete rows after `iscell>0.5` | `[221,370,685,746,541,435]`, total 2,998, mean 499.7; every supplied row passes 0.5 on every day | Figure 5B shows `[285,376,799,728,541,411]` (3,140); text says 526±190 | Treat the distributed arrays/README as authoritative for the released version. There is no principled filter that can create absent cells, and removing valid released rows to imitate some paper counts would be unjustified. Retain all 2,998 supplied matched cells and explicitly report the release-version discrepancy. |
| Session duration | No experiment-duration logic in Track2p | 14 sessions have 36,000 frames (20 min); 27 have 54,000 (30 min) at 30 Hz | Methods state every session lasted 20 min; Figure 5 and Supplementary Figure 7 show 0–20 min even for the 54,000-frame example subject | Restrict all sessions to their first 36,000 neural frames / 20 min. This matches the paper’s analysis window and equalizes duration while leaving exactly 20 requested 60-s trials/session. |
| Neural processing | GUI `F_processing` defaults `neucoeff=0.0` and baseline subtraction; raw-data notebook says compute paper dF/F or use spks | Every `ops.npy` consistently stores Suite2p analysis values: `neucoeff=0.7`, `baseline=maximin`, `sig_baseline=10`, `win_baseline=60`, `fs=30` | Analyses use baseline-corrected fluorescence with default Suite2p parameters | Follow the analysis Methods and per-session Suite2p ops, not the optional Track2p GUI visualization default: subtract 0.7×neuropil, estimate maximin baseline, subtract baseline, then 10-frame average. |
| Neural/behavior alignment | Reference package has no behavioral loader | Motion/timestamp vectors may be short; integer multiples in inter-frame intervals identify missing camera frames. All sessions cover the first 36,000 expected trigger positions after reconstruction. | Microscope triggered 30-Hz camera; README says interpolate over missing frames | Convert each inter-frame interval to an expected trigger-step count by rounding relative to its session median, cumulatively reconstruct frame positions, and linearly interpolate motion onto imaging frame indices 0…35,999. This is identity for gap-free sessions and inserts 282 missing values within the retained windows. |
| Denoising/binning | Track2p raster GUI optionally averages frames; no decoding code in supplied repository | Raw streams are nominally 30 Hz | Paper averages neural and behavior in non-overlapping bins of 10 timestamps for decoding | Apply the identical 10-frame mean after alignment and baseline correction, yielding 3 Hz / 333.333 ms time bins. |
| Trial splitting | N/A | Continuous recordings | Paper decoder CV used consecutive 2-min blocks | Decoder Task explicitly requires 60-s trials, so use consecutive 60-s blocks (180 binned time points), an intentional downstream-task override. |
| Output representation | N/A | Motion is continuous uint64 before averaging | Paper predicts continuous motion by ridge regression | Decoder Task explicitly requires five session-specific equal-percentile categories. Compute quintiles from the aligned, 10-frame-averaged 20-min session and encode labels 0–4. |

Final understanding: the release contains final longitudinally curated Suite2p cells and synchronized global motion energy. The conversion should not re-run Track2p. It should reproduce the paper’s Suite2p baseline correction, 10-frame decoder binning, first-20-minute analysis window, and timestamp-aware motion alignment, with only the explicitly requested 60-s trials and categorical quintile target differing from the reference decoder.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `suite2p/plane0/F.npy`, `Fneu.npy`, `ops.npy` | `neural[session][trial]` | Full-recording Suite2p default correction: `Fc=F-neucoeff*Fneu`; Gaussian/min/max baseline; `Fc-Flow`; keep first 36,000 frames; mean each 10 frames; split into 20 matrices `(n_neurons,180)` float32 | `DataManagement.F_processing`; Suite2p parameters in `ops` | Processing the full available trace before cropping preserves baseline context for 30-min source arrays. No per-cell z-score: paper z-scoring is explicitly visualization-only. |
| Imaging frame indices and `ops['fs']` | `input[session][trial][0]` | Mean raw-frame elapsed times within each 10-frame bin: `(bin_start+4.5)/30` seconds from session start; split into `(1,180)` float32 | Paper 10-timestamp averaging | Absolute session time continues across trials, as explicitly requested. |
| `motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy` | `output[session][trial][0]` | Infer integer camera-trigger positions from rounded inter-frame-interval/median ratios; linear interpolation onto neural frames 0…35,999; average every 10 frames; calculate within-session 20/40/60/80th percentiles; `searchsorted(..., side='right')` to labels 0–4; split `(1,180)` int64 | Paper videography metric and 10-timestamp averaging; data README missing-frame guidance | Quintile thresholds are session-specific as requested; labels are approximately 20% each. |
| Subject directory names | `subjects`, `subject_idx` | Sorted unique names; session index points to subject | Data README | `['jm031','jm032','jm038','jm039','jm040','jm046']`. |
| Experiment anatomy | `brain_regions`, `brain_region_idx` | One region label and all-zero per-neuron indices | Paper Methods | Region label: `S1 barrel cortex (L2/3)`. |
| Decoder variable definitions | `input_names`, `output_names`, `output_values` | Descriptive global labels | Decoder Task | Input `time_elapsed_from_session_start_s`; output `motion_energy_quintile`; values lowest through highest quintile. |
| Acquisition/processing facts and per-session thresholds | `metadata` | Store timing, alignment, preprocessing, trial/window details, session IDs, source frame counts, dropped-frame counts, and quintile edges | Paper/Data/Code synthesis | `time_bin_size=1000/3 ms`; event is session start; `off_start=off_end=None` because individual trials are fixed windows rather than event-centered epochs. |

### Key Decisions
1. **Use all released matched cells**: Cell detection and longitudinal curation have already been applied. Re-filtering would discard valid tracks, while absent paper-version cells cannot be reconstructed.
2. **Use baseline-corrected fluorescence rather than `spks` or raw `F`**: This matches the neural stream used for the paper’s decoding. Suite2p `ops` values resolve the GUI/reference-method difference in favor of `neucoeff=0.7`.
3. **Process full trace then retain 20 minutes**: Baseline estimation near 20 minutes should use the full acquired context when it exists; the retained 20-minute window matches the paper and provides equal session lengths.
4. **Align motion by inferred trigger ordinal**: Imaging-triggered camera acquisition makes the integer timestamp-gap ratio the direct indication of skipped camera triggers. Linear interpolation follows the distributed README.
5. **Average 10 frames before discretization**: This preserves the paper decoder’s denoising/time resolution. Percentiles are then defined on exactly the continuous values being classified.
6. **Use bin-center elapsed time**: Each averaged neural/output sample represents the mean of ten frame times, so the corresponding input is the mean time, not the left edge.
7. **Use 20 non-overlapping 60-s trials per session**: This obeys the Decoder Task and ensures all sessions have identical trial/time dimensions with no partial trials.
8. **Sample mode selects late example-mouse sessions**: Use the last two `jm039` sessions (paper example P13/P14) because late recordings have the paper’s strongest motion representation, making sample decoder validation informative.

### Planned Sanity Checks
- [ ] Confirm 6 subjects, 41 sessions, 820 trials, 180 bins/trial, 2,998 unique matched neurons, and per-subject row counts constant across days.
- [ ] Independently load raw `F`, `Fneu`, and `ops`, reproduce baseline correction plus 10-frame mean for selected cells/times, and compare to concatenated converted neural trials with `np.allclose()`.
- [ ] Independently construct mean frame times and compare every converted input value with `np.allclose()`.
- [ ] Independently load motion/timestamps, reconstruct trigger positions, interpolate, average, compute quantiles/labels, and compare converted output with `np.allclose()`.
- [ ] Confirm gap-free motion alignment is exactly identical before binning and all reconstructed sessions cover frame 35,999 without extrapolation.
- [ ] Confirm all arrays are finite, trial modalities have identical time lengths, neural rows match region-index lengths, output labels are integers 0–4, and every session has at least two trials.
- [ ] Confirm each output class is approximately 20% per session and report any deviations due to quantile ties.
- [ ] Compare retained window/session/trial/neuron statistics against paper, code, raw data, and release README; document unavoidable release-version count mismatch.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with the required positional output path, default/explicit `--full`, `--sample`, and `--show-processing`. The script discovers sessions deterministically, reconstructs behavior timing gaps, performs chunked full-trace Suite2p baseline correction, crops/averages/splits data, validates shapes and values, records per-session provenance, and writes the target pickle. `--show-processing` produces a ten-panel plot for each of up to two sessions covering every material loading/alignment/processing/discretization step. Syntax compilation and CLI help completed successfully.

Code inefficiencies identified:
Loading and filtering full 30-minute fluorescence arrays for all neurons at once would create several large temporary arrays per session. Recomputing processing for diagnostics would also duplicate the expensive filters.

Code speedups added:
Memory-mapped source arrays; 64-neuron processing chunks; vectorized 10-frame reshaping/means and trial splitting; diagnostic intermediates retained only from the first chunk and only when plotting; no redundant passes over full neural arrays.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 746 unique matched cells; 1,492 neuron-session rows |
| Neurons / session | 746, 746 |
| Subjects | 1 (`jm039`) |
| Sessions / subject | 2 |
| Trials (total) | 40 |
| Trials / session | 20, 20 |
| Time input range (s) | [0.15, 1199.8167] |
| Neural shape / dtype | `(746,180)` float32 per trial |
| Input shape / dtype | `(1,180)` float32 per trial |
| Output shape / dtype | `(1,180)` int64 per trial |
| Motion-quintile distribution | [0.200, 0.200, 0.200, 0.200, 0.200] in each session |

### Processing Plots Review
Both processing plots were inspected. Raw F and Fneu are well behaved; neuropil subtraction lowers the baseline without removing calcium transients; the maximin baseline follows slow drift; corrected/binned rasters preserve coincident population events. Raw and aligned motion overlap exactly in these gap-free sample sessions, 10-frame averaging preserves movement bouts, percentile thresholds are ordered, all five labels occur, and absolute elapsed time/trial boundaries are monotonic. No temporal shifts, clipping, non-finite values, or discretization anomalies were seen.

Manual pickle inspection confirmed all required keys, equal modality dimensions and trial counts, finite neural/input values, labels 0–4, and correct brain-region index lengths. `/app/verification_sample_out.txt` reports valid format with no errors or warnings.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Chunked vectorized filters and means, memory mapping, no diagnostic recomputation | Sample completed in 4.38 s total (about 2.15 s processing/session) |

| Step | Time / Session | Estimated Total Time |
| Baseline correction + conversion | ~2.15 s for 746-neuron, 54k-frame sample sessions | Conservative 2 minutes for 41 sessions, accounting for varying cells/source lengths plus serialization |

Full conversion estimate is far below 15 minutes; no further optimization is required before Step 9.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| Motion-energy quintile | 0.8719 | 0.3674 |

Training completed on CUDA. Loss decreased from 84.268 at epoch 1 to 0.513 at epoch 200 (test loss 6.928). Validation balanced accuracy is 1.84× the uniform chance level of 0.20, so every requested output is above chance. The train/validation gap will be revisited after the larger, more representative full-dataset run.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 282.7 MiB
- `verification_full_out.txt`: created

Full conversion completed in 33.68 s. Verification reports valid format with no errors or warnings. All 41 source sessions are represented in deterministic subject/date order, each with 20 complete trials and no partial trial; all released neural rows within the paper-matched 20-minute window are retained.

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | 3,140 figure-count sum / text mean 526±190 | Retain all-day complete `iscell>0.5` tracks | 2,998 unique; 20,445 neuron-session rows | 2,998 unique; 20,445 neuron-session rows | Exact release match; paper-version discrepancy documented |
| Mean neurons/session | Mouse-level paper mean 526±190 | Subject-constant all-day tracks | 498.66 across sessions | 498.66 | Yes for release |
| Subjects | 6 | N/A | 6 | 6 | Yes |
| Sessions | ≥6 daily/mouse | N/A | 41 (7,7,7,7,6,7) | 41 (7,7,7,7,6,7) | Yes |
| Trials (total) | Continuous; 20-min analysis | N/A | 41×20 possible requested 60-s windows | 820 | Yes |
| Trials/session (mean) | 20-min recordings | N/A | 20 after matched paper window | 20 | Yes |
| Time input range | 0–20 min | N/A | Mean 10-frame centers imply 0.15–1199.8167 s | 0.15–1199.8167 s | Yes |
| Motion output range | Continuous motion energy | N/A | Requested transform gives labels 0–4 | 0–4 | Yes |
| Motion output distribution | Not categorized | N/A | Equal-percentile target | [0.200,0.200,0.200,0.200,0.200] each session | Yes |
| Neural/behavior bin | 10 frames at 30 Hz | GUI supports frame averaging | 10 frames at 30 Hz | 333.333 ms, 180 samples/trial | Yes |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Read `/app/verification_full_out.txt` completely. Format is valid with no errors or warnings; therefore no warning waiver is needed.
2. **Independent neural raw-file check**: `/app/cache/critical_review_checks.py` directly loads original `F`, `Fneu`, and `ops` without importing conversion code. It independently applies `F-0.7*Fneu`, Gaussian/minimum/maximum baseline operations, crops, and 10-frame means. `np.allclose(rtol=1e-5, atol=1e-4)` passes for selected first/last neurons in sessions 0, 4, 21, and 36, covering native 20-min and 30-min recordings, gap-free behavior, 116 missing behavior frames, and a multi-frame gap despite equal source lengths.
3. **Independent input raw-file check**: Independently constructs all 3,600 bin-center times from raw imaging indices. `np.allclose` passes for every input value in all 41 sessions. Differences are continuously 1/3 s, including trial boundaries; first/last values are 0.15/1199.8167 s.
4. **Independent output raw-file check**: Directly loads original motion, timestamps, and stored intervals for all 41 sessions; confirms stored intervals equal timestamp differences; reconstructs trigger positions, interpolates, averages, computes percentiles, and labels independently. Exact `np.allclose(..., rtol=0, atol=0)` passes for every output value. Metadata thresholds also match. All 282 missing retained-window triggers are recovered.
5. **Data loading comparison**: Reference and conversion both use `np.load` on Suite2p arrays and preserve row order. Conversion memory mapping/chunking is computational only and does not change values.
6. **Neuron/trial filtering comparison**: Reference `save_in_s2p_format` applies `iscell>0.5`, selects all-day-complete Track2p matches, and exports those rows. All distributed rows have class 1/probability >0.5; conversion retains all. Paper has continuous sessions; only explicit requested 60-s splitting is added. Each session has 20 trials (well above minimum two).
7. **Temporal alignment comparison**: Reference text says imaging triggers camera acquisition; README identifies missing camera frames and permits interpolation. Conversion uses timestamp interval multiples as skipped-trigger counts and linear interpolation. Direct checks show coverage through retained frame 35,999 in every session without endpoint extrapolation.
8. **Binning comparison**: Paper averages both fluorescence and behavior in non-overlapping groups of 10 timestamps. Conversion does exactly this after correction/alignment, producing 333.333-ms bins. Paper CV used 2-min blocks; 60-s trials are the explicit Decoder Task override.
9. **Input construction comparison**: Session elapsed time is not a paper decoder covariate; it is explicitly required here. Bin-center time correctly represents the mean timestamp of each 10-frame measurement.
10. **Output construction comparison**: Paper retains continuous motion, whereas the task mandates five equal-percentile classes selected per session. Independently recomputed 20/40/60/80th percentiles and labels match exactly; every class has 720/3,600 values in every session.
11. **Key-statistics comparison**: Verified 6 subjects; 41 sessions in counts 7/7/7/7/6/7; subject neuron counts 221/370/685/746/541/435; 2,998 unique cells; 20,445 neuron-session rows; 820 trials; 180 bins/trial; one input/output/brain region; time range 0.15–1199.8167 s; labels 0–4 at 20% each. These match raw release files and the intended processing. The unreconstructable paper/release cell-count discrepancy remains transparently documented in Steps 4 and 9.
12. **Edge-case audit**: Checked first/last raw frames, first/last bins, all 19 within-session trial boundaries, 20-min exact sources, 30-min sources cropped after full-context baseline processing, isolated gaps, 116/148-gap sessions, equal-length sources with timestamp gaps, and a gap beyond the retained window. No off-by-one error, NaN/Inf, shape mismatch, invalid index, tie-induced class imbalance, or source-order mismatch was found.

### Issues Found and Resolved
- **Paper/release cell-count mismatch**: Cannot be “fixed” without inventing or deleting tracks. Resolved by using every released, already-curated row and documenting both values.
- **Mixed 20/30-min source arrays versus 20-min paper analysis**: Resolved by full-context fluorescence processing followed by a uniform first-36,000-frame window.
- **Sparse dropped camera triggers**: Resolved using timestamp-derived trigger ordinals and linear interpolation; all independent comparisons pass.
- No conversion bug was found in this review, so no re-conversion iteration was required. Full check output is saved at `/app/cache/critical_review_checks_out.txt`.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes. CUDA training completed all 200 epochs; loss fell from 97.5515 at epoch 1 to 1.0182 at epoch 200; held-out loss was 3.9986.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| Motion-energy quintile | 0.6629 | 0.2950 | Uniform chance 0.2000; validation is 1.475× chance. |

`/app/train_decoder_full_out.txt` contains the complete run. Required sample and prediction plots were created and inspected: neural events are finite and temporally structured; output labels span all classes; predictions show above-chance but imperfect tracking. Accuracy is above chance, while the marginally-below-1.5× result and train/validation gap trigger the required Step 12 investigation.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| Motion-energy quintile (required categorical validator) | Train balanced accuracy 0.6629; validation balanced accuracy 0.2950; chance 0.2000 | No categorical accuracy reported. Paper reports continuous same-day R² increasing strongly with development. |
| Continuous motion, example P8 (independent paper-style check) | Nested blocked ridge R²=0.2078 | Supplementary Figure 7 example R²≈0.25 (main example annotation ≈0.26). |
| Continuous motion, example P14 (independent paper-style check) | Nested blocked ridge R²=0.6431 | Supplementary/Main Figure 7 example R²≈0.69. |

**Accuracy versus chance**: Full categorical validation is 1.475× chance—0.005 below the heuristic 1.5× threshold—and above chance by 0.095 absolute / 47.5% relative. The late-session sample reached 0.3674 (1.84× chance). This age dependence is expected: the paper’s example continuous R² is only 0.06–0.32 from P8–P11 and 0.69–0.70 from P12–P14, and reports motion representation emerging after P11. Pooling 41 early and late sessions therefore lowers performance relative to the deliberately late sample.

**Paper comparison**: `/app/cache/paper_ridge_check.py` independently reconstructs continuous motion from raw files and performs nested group ridge regression with ten consecutive 2-min groups, matching the reference design. P8/P14 results (0.2078/0.6431) are close to paper values (≈0.25/0.69). Remaining small differences are consistent with an incompletely specified fold assignment/alpha grid; importantly, both magnitude and developmental increase reproduce. This direct continuous-task check is more comparable to the paper than 5-class balanced accuracy.

**Train/validation gap**: 0.6629/0.2950=2.25, exceeding the review threshold. Audit found no leakage: every trial owns a unique, non-overlapping 1,800-frame range; conversion explicitly copies trial arrays; validator uses disjoint 16/4 trial splits per session; raw-output checks pass; no future output enters neural/input construction. The supplied validator learns 41 separate 100-dimensional neural projections plus a shared classifier from only 16 training trials/session, which provides substantial capacity for session-specific fitting. Reducing components/adding regularization would address this model-level overfitting but is outside conversion scope. Altering reference-matched neural preprocessing merely to optimize this validator would make the conversion less faithful.

**Required low-accuracy debugging**:

1. Raw output was verified for every time point in all sessions (stronger than three requested trials); exact equality passed.
2. Neural, aligned continuous motion, binned motion, and categorical output were plotted together in both processing figures; validator sample/prediction plots were also inspected. Events and labels are synchronized with no shift.
3. Output variation is exactly 20% per class in each session; no 99% class or missing label.
4. Neural filtering follows the supplied already-curated all-day-complete `iscell>0.5` tracks; raw curation assertions pass for every session.
5. Baseline correction, 10-frame processing, 20-min window, and motion alignment were re-compared with code/methods and independently reproduced with `np.allclose`.

Conclusion: all evidence supports correct conversion. The near-threshold categorical score is scientifically consistent with weak early developmental motion encoding and technically consistent with the validator’s high-capacity session projections; no justified conversion change was identified.

### Issues Found and Resolved
- **Validation is marginally below 1.5× chance**: Investigated via exhaustive raw output checks, alignment plots, class variation, reference-processing audit, and comparable paper-style continuous ridge decoding. No conversion defect found; continuous P8/P14 results closely reproduce the paper.
- **Train/validation gap >1.5×**: Confirmed disjoint trials and absence of leakage. Attributed to validator model capacity and only 16 training trials/session, not data construction. No conversion changes made.
- No iteration/re-conversion was warranted because all processing comparisons and direct numerical tests passed.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

`/app/README.md` documents the dataset, loading, format, processing, reproduction commands, core statistics, and decoder results. Investigation scripts, outputs, and extracted reference figures are contained in `/app/cache/` and described by `/app/cache/README_CACHE.md`. Generated Python bytecode was removed; original project files were preserved. A final non-empty-file audit passed for every required deliverable, and both pickle files load successfully (sample: 2 sessions/40 trials; full: 41 sessions/820 trials).
