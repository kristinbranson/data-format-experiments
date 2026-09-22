# Dataset Conversion Notes

## Overview
- **Dataset**: Track2p longitudinal mouse barrel-cortex two-photon imaging dataset (provided paper/code/data)
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents (listed after creating this notes file):
- `.manifest`
- `CONVERSION_NOTES.md`
- `Dockerfile`
- `code`
- `data`
- `decoder.py`
- `docker-compose.yaml`
- `docker-compose.yaml~`
- `methods.txt`
- `paper.pdf`
- `train_decoder.py`

Environment verification:
- Python, NumPy, and PyTorch imported successfully (versions printed in terminal during setup).
- `/app/CONVERSION_NOTES.md` existence explicitly verified with `ls -la`.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `track2p()` | `track2p/t2p.py` | PROCESSING | Main pipeline: validate planes, load images, register sessions, match ROIs, assign cross-day identities, generate Suite2p indices, and save results. |
| `load_all_imgs()` | `track2p/io/s2p_loaders.py` | LOADING | Load per-session/per-plane Suite2p mean images used for registration (functional channel and optional second channel). |
| `load_stat_ds_plane()` | `track2p/io/loaders.py` | LOADING/CURATION | Load `stat.npy` for one dataset/plane and retain ROIs passing the configured `iscell.npy` probability threshold. |
| `load_all_ds_stat_iscell()` | `track2p/io/s2p_loaders.py` | LOADING/CURATION | Load cell-filtered Suite2p ROI statistics for every session and imaging plane. |
| `load_all_ds_ops()` / `load_all_ds_mean_img()` | `track2p/io/s2p_loaders.py` | LOADING | Load `ops.npy`; obtain `meanImg` or `meanImg_chan2` per plane. |
| `load_all_ds_centroids()` | `track2p/io/s2p_loaders.py` | PROCESSING | Read ROI centroid coordinates from each filtered stat entry's `med` field. |
| registration loop | `track2p/register/loop.py` | PROCESSING | Register moving-session images/ROIs to a reference session, using image registration and transformed ROI masks. |
| matching loop/utilities | `track2p/match/loop.py`, `track2p/match/utils.py` | PROCESSING/CURATION | Build pairwise ROI assignments using centroid/intersection logic and an overlap metric, then threshold assignments for consistent identities. |
| `generate_suite2p_indices()` | `track2p/t2p.py` | PROCESSING | Convert Track2p identity assignments back to indices in each session's Suite2p arrays. |
| save functions | `track2p/io/savers.py` | PROCESSING | Save Track2p results and optionally Suite2p-compatible outputs; generated `iscell.npy` marks saved tracked cells as cells. |

### Notes
- The repository is Track2p, a longitudinal cell-tracking package. Inputs are session directories containing Suite2p outputs, organized by `suite2p/planeN/`.
- Default input format is `suite2p`; an alternate NPY interface uses `F.npy`, `fov.npy`, and `rois.npy`. The utility notebook converting Suite2p to NPY loads `F.npy`, `stat.npy`, `iscell.npy`, and `ops.npy`, applies the boolean cell mask to both `stat` and `F`, converts each stat entry's `ypix`/`xpix` to an ROI mask, and saves filtered fluorescence and masks.
- Cell curation in the core package uses the second column of Suite2p `iscell.npy` as a probability and retains ROIs above `track_ops.iscell_thr`; the default threshold is 0.50. This filtering occurs before registration/matching, so original Suite2p indices must be handled carefully when using Track2p identities.
- Registration uses motion-corrected mean field-of-view images (`meanImg`, optionally `meanImg_chan2`) and transforms ROI masks between sessions. Matching uses ROI centroid/intersection candidates and overlap/IOU-style threshold metrics to establish cross-day identities.
- Track2p's concern is spatial ROI identity tracking, not extraction of behavioral motion energy. It does not provide trialization or temporal synchronization with behavior.
- Track2p does not compute delta-F/F in the inspected core pipeline. `F.npy` is raw extracted Suite2p ROI fluorescence in the conversion notebook. Therefore any dF/F or neuropil correction used for this decoder must come from the supplied dataset representation or methods, rather than being inferred from Track2p alone.
- Suite2p arrays such as `Fneu.npy` or `spks.npy` are not central to the inspected Track2p tracking pipeline; the relevant neural representation must be resolved from the actual shared data and paper in later ordered steps.
- Track2p output can map longitudinally tracked identities to per-session Suite2p ROI indices. For this task, each recording remains a decoder session; cross-day tracking should only be imposed if the supplied processed data/reference methods explicitly use it.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Six subject directories: `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, and `jm046`.
- Each subject contains dated session directories. Each session has one imaging plane at `suite2p/plane0/` and behavior at `move_deve/`.
- Per-session Suite2p arrays:
  - `F.npy`: ROI fluorescence, float32, shape neurons x imaging frames.
  - `Fneu.npy`: neuropil fluorescence, float32, same shape as F.
  - `spks.npy`: Suite2p deconvolved activity, float32, same shape as F.
  - `iscell.npy`: float64, neurons x 2; cell decision and probability.
  - `stat.npy`: object array of ROI spatial statistics, one entry per neuron.
  - `ops.npy`: Suite2p processing/acquisition dictionary.
- Per-session behavior arrays:
  - `motion_energy_glob.npy`: global video motion energy, uint64 vector.
  - `tstamps.npy`: float64 timestamp vector, same length as motion energy.
  - `interframe_int.npy`: float64 and exactly `np.diff(tstamps)` in a checked session.
- `ground_truth.csv` occurs for jm038, jm039, and jm046. These are semicolon-delimited 7-column, 64-row manual ROI-correspondence tables used to evaluate longitudinal tracking, not behavioral labels.
- No README/documentation file is present in the data tree.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total unique tracked identities across subjects) | 2,998 |
| Neurons / session | subject-constant: 221 (jm031), 370 (jm032), 685 (jm038), 746 (jm039), 541 (jm040), 435 (jm046); range 221-746 |
| Repeated neuron-session observations | 20,445 |
| Subjects | 6 |
| Sessions / subject | 7 each, except jm040 has 6 |
| Sessions (total) | 41 |
| Native imaging samples / session | 36,000 for jm031/jm032; 54,000 for the other subjects |
| Nominal session duration | 1,200 s or 1,800 s at 30 Hz |
| Nominal 60-second trials (before endpoint handling) | 1,090 total; 20 or 30 per session |

### Variables, Dimensions, and Data Quality
- `ops.npy` reports `fs=30`, one plane, two channels, functional channel 2, calcium decay `tau=0.3`, neuropil coefficient 0.7, and Suite2p spike baseline parameters `baseline=maximin`, `win_baseline=60`, `sig_baseline=10`, `prctile_baseline=8`.
- All session arrays within a subject have an identical neuron count, and every `iscell[:,0]` entry is true. Across all 20,445 repeated entries, the boolean decision agrees exactly with `iscell[:,1] > 0.5`. This indicates the distributed Suite2p directories are already Track2p-curated/reindexed to longitudinally retained cells; applying a further cell filter changes nothing.
- Motion energy and timestamps normally equal neural frame count, but 8/41 behavior streams are shorter by 1-148 samples. The exact sessions and endpoint policy must be handled explicitly during mapping; no extrapolation beyond observed behavior should be used without justification.
- Motion energy is nonnegative uint64, finite in inspected sessions, with substantial session-specific offset/scale. The required equal-percentile discretization per session naturally accommodates this scaling.
- Stored `tstamps` increments are about 3.36e-5 in their native units and are not directly seconds despite 30 Hz imaging. Their units/relationship to video timing must be resolved from the paper/methods before temporal alignment is finalized.
- `spks` is nonnegative deconvolved Suite2p activity. Whether to use `spks`, raw F, neuropil-corrected F, or derived dF/F will be resolved against the methods in Steps 3-5 rather than guessed from filenames.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote / location |
|-----------|-------|-------------------------|
| Neurons (tracked across all days) | 526 ± 190 SD per mouse | Methods excerpt: “On average 526 (± 190 std) neurons per mouse were successfully tracked across all days”. |
| Fraction retained from first day | 33% ± 11% SD | Same methods paragraph. |
| Subjects | 6 mice | “full dataset of 6 mice”. |
| Sessions / subject | minimum 6 consecutive daily sessions within P7-P14 | Methods; one illustrated P8-P14 example has 7 sessions. |
| Trials | None in the original experiment; recordings are continuous | Trialization into 60-s chunks is required only by the decoder task. |
| Session duration | Paper methods state 20 minutes | “each session lasted 20 minutes”; supplied arrays include both 20- and 30-minute sessions, to resolve in Step 4. |
| Neural data time bin | 1/30 s native; 10-frame (1/3 s) averaging for paper decoding | Imaging rate 30 Hz; decoding methods average 10 consecutive timestamps. |
| Behavior data time bin | 1/30 s native; 10-frame averaging for paper decoding | Video recorded at 30 Hz and triggered by microscope; same decoder averaging. |
| Brain region/layer | Mouse barrel cortex, layer 2/3, 100-200 µm depth | Chronic imaging methods. |
| Ground-truth tracking set | 64 cells × 3 mice = 192 | Paper response: 64 ROIs manually tracked through every session in each of 3 mice. |
| Behavior variable | Global movement/motion energy | Sum of squared pixel-wise differences between each pair of consecutive video frames. |
| Reward rate | N/A | Spontaneous behavior under sensory-minimized conditions; no rewarded task. |

### Processing Details
- Recordings used dual-channel GCaMP8m and tdTomato imaging in a 720 × 720 µm field at 512 × 512 pixels, 30 Hz. Mice moved spontaneously on a non-motorized treadmill in darkness/sensory-minimized conditions.
- Calcium data were processed separately per recording with Suite2p: motion correction, ROI detection, signal extraction, and spike deconvolution.
- The paper states: “We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters) for all subsequent analyses.” Thus dF/F-like baseline-corrected fluorescence, not deconvolved spikes, is the reference neural stream for decoding.
- Video was acquired at 30 Hz with microscope acquisition triggering each camera frame, providing direct synchronization between modalities.
- Motion energy was calculated from consecutive frames: pixel-wise frame difference, square each pixel difference, then sum across pixels to one scalar per time point.
- Reference analyses mildly denoised both dF/F and behavior by averaging non-overlapping bins of 10 consecutive timestamps, yielding 3 Hz samples. Calcium event-rate analysis also used 10-frame averaging before peak detection.
- Reference decoding predicted continuous behavior from neural activity using ridge regression. Same-day performance used nested 5-fold cross-validation; splitting was based on consecutive 2-minute recording blocks. Cross-day decoding fit a model on one day (with its selected lambda) and evaluated it on other days.

### Curation Steps

**Neuron curation rules**:
- Suite2p ROIs with cell-classification probability above the default threshold 0.5 were considered cells before Track2p.
- Track2p registered fields/ROIs and retained identities successfully tracked across all recording days for the reported longitudinal population analyses.
- The GUI supports visual inspection/curation, but no additional quantitative fluorescence-quality exclusion criterion is stated for this dataset.

**Trial curation rules**:
- Original recordings are continuous rather than trial based. No event/reward trial exclusion is described.
- For conversion, only complete 60-second intervals with all required synchronized streams should be retained; this downstream rule is necessitated by the requested format and will be finalized in Step 5.

### Decoders Trained
| Decoded variable | Metric / reported result |
|------------------|--------------------------|
| Continuous global motion energy, same day | Ridge-regression R² under nested cross-validation; paper shows day/mouse distributions and traces, but does not provide one categorical accuracy scalar in text. |
| Continuous global motion energy, cross day | R² matrix for all train-day/test-day combinations; performance is developmentally structured and shown graphically. |

### Relevance to Required Decoder
- The requested output differs from the paper by discretizing motion energy into five within-session equal-percentile classes. Chance balanced accuracy is therefore 20%; paper R² values cannot be numerically equated to this accuracy.
- Matching the paper's 10-frame averaging remains applicable and produces a common 333.33 ms bin for neural activity, elapsed-time input, and motion-energy output.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Cell filtering | Track2p retains Suite2p ROIs with `iscell` probability >0.5 before matching | All 20,445 repeated ROI entries are accepted; neuron count is constant across a subject's days | Use probability >0.5 and analyze cells tracked across all days | Data are already Track2p-curated/reindexed. Verify threshold but do not remove any additional cells. |
| Tracked-neuron count | Track2p output indexes identities across sessions | Subject counts 221, 370, 685, 746, 541, 435; mean 499.7, sample SD 197.7 | 526 ± 190 SD neurons/mouse | Close agreement; small difference likely reflects rounded paper values or release/curation version. Preserve all 2,998 supplied identities. |
| Session duration | Code is agnostic | jm031/jm032: 20 min; jm038/jm039/jm040/jm046: 30 min | Methods state each session lasted 20 min | Treat actual complete arrays as authoritative and preserve all valid 20/30-min recordings; do not discard 10 valid minutes from 27 sessions. Document metadata. |
| Neural representation | Track2p NPY interface passes `F.npy`; core tracking does not transform temporal traces | Raw `F`, `Fneu`, and deconvolved `spks` are available; F has raw fluorescence scale | Baseline-corrected fluorescence used as dF/F with default Suite2p parameters | Reproduce Suite2p default preprocessing: neuropil-correct `F - 0.7*Fneu`, then default maximin baseline correction (`win_baseline=60 s`, `sig_baseline=10 s`); use this fluorescence stream, not `spks`. |
| dF/F terminology | Installed Suite2p `dcnv.preprocess` subtracts its estimated baseline and returns baseline-corrected fluorescence (it does not divide by baseline) | Ops stores default baseline parameters | Paper calls this baseline-corrected trace “dF/F” | Match the explicit Suite2p algorithm/ops rather than adding an undocumented division. Neural scale is later standardized by decoder code as needed. |
| Temporal synchronization | Track2p has no behavior timing logic | Neural/motion lengths usually match; 8 behavior streams are short by 1-148 endpoint samples. Raw timestamps increment ~3.36e-5 and scale to seconds by ~992, not a clean unit conversion | Camera is microscope-triggered; both modalities at 30 Hz | Align by synchronized sample ordinal at 30 Hz. Truncate each session to the shorter observed stream before 10-frame averaging; never extrapolate missing endpoint behavior. |
| Temporal smoothing | No temporal processing in Track2p | Native streams are nominally 30 Hz | Paper averages neural and behavior in bins of 10 timestamps for decoding | Apply non-overlapping 10-frame means jointly to neural and motion streams, yielding 3 Hz and 180 bins per 60-s trial. |
| Motion output | N/A | Raw uint64 global motion energy with session-specific scale | Same global squared frame-difference measure is decoded continuously | Average raw motion energy in 10-frame bins, then discretize into five equal-frequency bins separately per session as required by this task. |
| Trial definition | N/A | Continuous sessions | Reference CV uses consecutive 2-minute blocks; experiment has no trials | Required task overrides this: split into consecutive non-overlapping 60-s trials, retaining only complete trials after common-stream truncation. |

### Final Consistent Understanding
- The six distributed subject folders are the longitudinal barrel-cortex dataset described in the paper; their subject-level neuron counts and session counts agree closely with reported summary statistics.
- Spatial/cell curation has already been applied by Track2p, and the retained neuron ordering is constant across sessions of each subject.
- Reference temporal neural processing is Suite2p baseline correction of neuropil-corrected fluorescence followed by 10-frame averaging. This will be implemented from the saved per-session ops values.
- Video and imaging are hardware synchronized at 30 Hz. Sample-index alignment is more reliable than interpreting the supplied timestamp magnitudes, while timestamp/motion lengths identify genuinely unavailable endpoint samples.
- Differences from the paper's decoder (60-s trials and five categorical bins rather than continuous ridge regression) are required by the downstream task, not accidental deviations.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `F.npy`, `Fneu.npy`, session `ops.npy` | `neural[session][trial]` | Select already-curated cells; compute `F - neucoeff*Fneu`; apply Suite2p `dcnv.preprocess` with saved baseline parameters; average non-overlapping groups of 10 frames; split into 60-s blocks; float32 neurons × 180 bins | Suite2p `dcnv.preprocess`; paper calcium-processing and decoding methods | Uses baseline-corrected fluorescence called dF/F by paper, not deconvolved `spks`. |
| Synchronized sample ordinal and `ops['fs']=30` | `input[session][trial][0,:]` | Absolute elapsed session time in seconds at each 10-frame averaged sample: mean native sample index / 30; reshape to 1 × 180 float32 | Paper videography synchronization; task specification | Time does not reset at each trial because task asks elapsed time from beginning of session. First value is 0.15 s, the mean time of native samples 0-9. |
| `motion_energy_glob.npy` | `output[session][trial][0,:]` | Truncate to common observed length; 10-frame mean; compute session-specific 20/40/60/80th percentile edges over retained complete-trial samples; `searchsorted(..., side='right')`; split to 1 × 180 int64 | Paper videography preprocessing and 10-frame decoder averaging | Values 0-4 correspond from lowest to highest motion-energy quintile. Distinct edges in all sessions; tiny count differences from ties are retained rather than arbitrarily separating equal motion values. |
| subject directory name | `subjects`, `subject_idx` | Sorted unique IDs; map each chronological session to subject index | Native hierarchy | Six subjects. |
| All ROIs | `brain_region_idx[session]` | Zeros of length n_neurons | Paper chronic imaging methods | Every recorded neuron is in layer 2/3 mouse barrel cortex. |
| Constant labels | names/value metadata | `input_names=['elapsed_time_from_session_start_s']`; `output_names=['motion_energy_quintile']`; class names lowest through highest | Decoder task | One time-varying input and one categorical time-varying output. |

### Session and Trial Ordering
- Sessions are ordered lexicographically by subject ID and then ISO-format session directory name, which is chronological within subject.
- For each session, common native length is `min(F frames, Fneu frames, motion length, timestamp length)`.
- Retain `floor(common_length / 1800) * 1800` native samples, because 1,800 frames = 60 s at 30 Hz. Endpoint samples outside a complete trial are discarded.
- Apply 10-frame averaging before trial reshape; every trial therefore contains exactly 180 bins at 3 Hz (333.333 ms/bin).
- This yields 1,081 trials: jm031 137, jm032 137, jm038 210, jm039 209, jm040 179, jm046 209. Every session retains at least 19 trials.

### Key Decisions
1. **Neural stream**: Use Suite2p baseline-corrected neuropil-corrected fluorescence because this exactly follows the paper; do not substitute inferred spikes.
2. **No additional neuron filtering**: All distributed ROIs already pass `iscell > 0.5` and are Track2p identities retained across all days.
3. **Temporal binning**: Average 10 consecutive synchronized samples for both neural and behavior, matching reference decoding and reducing 30 Hz to 3 Hz.
4. **Alignment**: Use shared frame ordinal at documented 30 Hz because acquisition was hardware synchronized and raw timestamp units are not reliable seconds. Do not interpolate or extrapolate missing behavior.
5. **Incomplete endpoints**: Drop incomplete final 60-s windows. This excludes nine trials whose behavior lacks 1-148 samples and prevents fabricated labels.
6. **Percentile scope**: Determine quintile edges separately for each session, after 10-frame averaging and complete-trial restriction, exactly as “selected per session” requires. Use the full retained session so class semantics are consistent across that session's trials.
7. **Quantile ties**: Assign equal values to the same class using right-sided thresholding. This preserves categorical meaning; all edges are distinct and observed class imbalance is negligible.
8. **Elapsed time**: Store absolute session time at the mean timestamp of each averaged bin. It intentionally continues across trial boundaries.
9. **Numeric types**: Store neural/input as float32 and output as int64 to limit memory and satisfy PyTorch classification expectations.
10. **Metadata**: `time_bin_size=333.3333333333333` ms; alignment is the start of each consecutive 60-s window in a continuous session; `off_start=0.0`, `off_end=60.0`.

### Planned Sanity Checks
- [ ] Compare converted neural values against an independent direct raw-data calculation with `np.allclose()` for selected neuron/time samples.
- [ ] Compare elapsed-time arrays to independently computed mean native frame indices / 30 with `np.allclose()`; verify consecutive trial boundary spacing is 1/3 s.
- [ ] Compare converted output labels to independently averaged raw motion and independently computed per-session percentile thresholds with `np.allclose()`.
- [ ] Verify every trial is neurons × 180, input/output are 1 × 180, and all streams are finite.
- [ ] Verify all output values are integers in 0-4 and each session's class fractions are approximately 20%.
- [ ] Verify 6 subjects, 41 sessions, 2,998 subject-level tracked neurons, 20,445 repeated session-neuron observations, and 1,081 retained trials.
- [ ] Verify every brain-region vector length equals the corresponding session neuron count and contains only index 0.
- [ ] Visually overlay selected baseline-corrected neural traces and motion energy before/after 10-frame averaging, and show percentile thresholds/class labels.

---

## Step 6: Script Development
**Status**: COMPLETE

- Created `/app/convert_data.py` with required invocation `python -u /app/convert_data.py <outpicklefile>`.
- Implemented mutually exclusive `--full` (default behavior) and `--sample` modes plus `--show-processing` for up to two session plots.
- Implemented source discovery, subject/session ordering, iscell-threshold verification, synchronized endpoint truncation, Suite2p neuropil/baseline correction, 10-frame averaging, 60-s trialization, session percentile labeling, metadata construction, shape/value assertions, timing output, and pickle serialization.
- Diagnostic plots show: raw neuropil-corrected versus baseline-corrected fluorescence, 10-frame neural averages, native versus averaged motion energy, and full-session quintile labels with thresholds/counts.
- Local `suite2p_preprocess` avoids the expensive Suite2p import during conversion while reproducing `suite2p.extraction.dcnv.preprocess`; checked on a real-data slice with `np.allclose(rtol=1e-6, atol=1e-6)`.
- Script compiles, CLI help runs, and core functions pass synthetic shape/finite tests.

Code inefficiencies identified:
- Full raw fluorescence must be materialized once per session for baseline filtering; processing all sessions simultaneously would use excessive memory.
- Importing the full Suite2p package is slow and unnecessary after its small baseline algorithm has been verified.

Code speedups added:
- Memory-map source arrays and process one session at a time.
- Use vectorized neuropil subtraction, SciPy filters, block reshaping/means, percentile thresholding, and trial reshaping.
- Serialize only the final float32 neural/input arrays and int64 labels.
- Plot only the requested first two sessions.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (repeated session count) | 442 |
| Neurons / session | 221, 221 |
| Subjects | 1 (`jm031`) |
| Sessions / subject | 2 |
| Trials (total) | 40 |
| Trials / session | 20, 20 |
| Time bins / trial | 180 |
| Elapsed-time input range | [0.15, 1199.8167] s |
| Neural range | session 1 [-45.11, 824.29], session 2 [-47.94, 1144.54] |
| Motion-quintile distribution | each class exactly 0.200 in each sample session |

### Processing Plots Review
- Created `processing_jm031_2023-10-18_a.png` and `processing_jm031_2023-10-19_a.png` (1960 × 1680).
- Plots include every conversion stage: neuropil-corrected raw fluorescence, Suite2p baseline-corrected fluorescence, 10-frame neural means, native and averaged motion energy on the same time axis, and full-session quintile labels/thresholds/counts.
- No endpoint truncation occurred in either sample session. Neural and behavior traces share the same frame-derived time axis; class-count annotations are balanced and threshold ordering is valid.

### Manual and Format Validation
- Pickle keys exactly match the requested schema.
- Neural arrays are float32 `(221,180)`, inputs float32 `(1,180)`, and outputs int64 `(1,180)`.
- All values are finite; output values are exactly integers 0-4.
- Elapsed time continues across trials: trial 0 ends at 59.8167 s and trial 1 starts at 60.15 s, a one-bin (1/3 s) interval.
- `/app/verification_sample_out.txt` reports: “Data format is valid, no errors or warnings.”

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|-----------------------|--------------|
| Per-session memory mapping/vectorized processing | Avoids repeated I/O and Python per-frame loops |
| Local verified Suite2p preprocessing implementation | Avoids slow package import during conversion |
| One-session-at-a-time processing | Bounded peak memory |

| Step | Time / Session | Estimated Total Time |
|------|----------------|----------------------|
| Sample conversion including two plots | 0.8-1.1 s | 1.95 s observed for 2 sessions |
| Full 41-session conversion | approximately 1 s/session plus serialization | approximately 40-60 s; safely below 15 minutes |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None
- Validator summary: 2 sessions, 40 trials, 180 bins/trial, output range 0-4, exactly 20% per class.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-----------------------|-------------------------|
| motion_energy_quintile | 0.5480 | 0.3135 |

- Chance balanced accuracy is 0.2000; validation performance is 1.57× chance.
- Loss decreased consistently from 40.3448 at epoch 1 to 1.2363 at epoch 200; test loss was 2.3393.
- Training completed normally and `train_decoder.py` reported success.
- The train-validation gap is unsurprising for a deliberately small two-session sample and will be reassessed on the full dataset.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 392.8 MiB (393 MiB filesystem display)
- `conversion_full_out.txt`: created; corrected conversion completed in 37.39 s
- `verification_full_out.txt`: created; no errors or warnings

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total unique tracked neurons | 526 ± 190 SD/mouse | Track identities retained across days | 2,998 (499.7 ± 197.7 sample SD/mouse) | 2,998 | Yes; close to rounded paper summary |
| Mean neurons/session | tracked identities are constant within mouse | Track2p-reindexed arrays | 498.66 | 498.66 | Yes |
| Subjects | 6 | N/A | 6 | 6 | Yes |
| Sessions | minimum 6/mouse | arbitrary session lists | 41 (7,7,7,7,6,7) | 41 | Yes |
| Trials (total) | continuous recordings; N/A | N/A | 1,090 nominal complete windows before behavior endpoint loss | 1,081 complete synchronized windows | Yes; 9 incomplete endpoints excluded |
| Trials/session | N/A | N/A | 20 or 30 nominal | 19-30 retained, mean 26.37 | Yes |
| Input elapsed-time range | 20-min paper sessions; data also contain 30-min sessions | N/A | 0-1,800 s nominal | bin centers 0.15-1,799.8167 s | Yes |
| Output distribution | continuous motion energy | N/A | session-specific quintiles planned | [0.199995, 0.200005, 0.200000, 0.200000, 0.200000] | Yes |
| Repeated neuron-session count | N/A | N/A | 20,445 | 20,445 | Yes |

### Full Validation and Spot Checks
- Validator reports valid format with no errors or warnings, 41 sessions, 1,081 trials, fixed 180 bins/trial, one input, one output, and one brain region.
- Independent iteration over every trial confirmed float32 neural/input, int64 outputs, exact shapes, finite values, and labels 0-4.
- Global neural range is [-294.99, 2808.03]; elapsed-time range is [0.15, 1799.8167] s.
- Spot-checked sessions 0, 13, 20, 25, 31, and 40, spanning every subject and both normal/short endpoint cases. Metadata, trial counts, neuron counts, class counts, and elapsed-time endpoints are internally consistent.
- All nine sessions with incomplete final behavior windows retain 19 or 29 complete trials and report the discarded endpoint frame counts; no source data needed by a retained trial were lost.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Read `/app/verification_full_out.txt` completely. Result: “Data format is valid, no errors or warnings.” All listed session dimensions, ranges, region counts, and distributions are expected.
2. **Independent raw neural check**: `/app/cache/critical_review_checks.py` loads original `F`, `Fneu`, `iscell`, and `ops` files directly, independently performs neuropil correction plus Gaussian/minimum/maximum baseline filtering and 10-frame averaging, and uses `np.allclose()` against converted neural trials. Passed for sessions/trials `(0,5)`, `(25,28)`, and `(40,28)`, including two short-stream endpoint cases.
3. **Independent raw input check**: The same script independently calculates each selected trial's mean native frame indices divided by 30 Hz and compares to converted elapsed time with `np.allclose()`. Passed. It additionally checks the complete elapsed-time vector in every one of 41 sessions.
4. **Independent raw output check**: The same script loads raw motion energy directly, independently averages 10-frame blocks, computes per-session quantiles, applies right-sided class thresholds, and compares labels using exact `np.allclose(rtol=0, atol=0)`. Passed for all selected trials.
5. **Reference code/method comparison**:
   - **Loading**: conversion loads Suite2p `F.npy`, `Fneu.npy`, `iscell.npy`, `ops.npy` and motion arrays; Track2p loaders use the same Suite2p directory/ops/stat/iscell conventions.
   - **Neuron filtering**: conversion applies `iscell[:,1] > 0.5`; `track2p/ops/default.py` sets `iscell_thr=0.50`, and methods explicitly state probability >0.5. All supplied cells pass.
   - **Temporal alignment**: conversion aligns synchronized streams by sample ordinal at 30 Hz and common observed length; methods state microscope-triggered 30-Hz video synchronized to 30-Hz imaging.
   - **Binning**: conversion averages non-overlapping 10-frame blocks for neural and behavior; methods state both traces were averaged in bins of 10 timestamps for decoding.
   - **Input construction**: elapsed time is a downstream requirement absent from Track2p; it is correctly derived from synchronized frame ordinal and 30-Hz acquisition.
   - **Output construction**: raw motion is the reference consecutive-frame squared pixel-difference sum; conversion averages it as in the paper, then adds required within-session quintile discretization.
6. **Key statistics comparison**: Post-fix checks confirm 6 subjects, 41 sessions, 1,081 trials, 2,998 subject-level tracked neurons, 20,445 repeated neuron-session entries, mean 498.66 neurons/session, and output fractions `[0.199995, 0.200005, 0.2, 0.2, 0.2]`. These agree with source data and are close to paper's 526 ± 190 neurons/mouse.
7. **Edge cases/off-by-one checks**: For every session, independently recomputed `floor(min(neural,motion,timestamp length)/1800)` and compared trial counts, retained frames, complete elapsed vectors, final bin centers, and metadata. All 41 pass. Nine incomplete final windows are dropped; no interpolation or extrapolation occurs.

### Issues Found and Resolved
- **Iteration 1 — baseline boundary ordering**: Initial code baseline-corrected only the retained complete-trial prefix. In nine short behavior sessions, this made the filter endpoint differ from reference full-recording Suite2p preprocessing. Fixed by baseline-correcting the complete neural recording first and then slicing the synchronized prefix.
- **Affected reruns**: Re-ran sample conversion/verification (unchanged, no warnings), full conversion, full verification, all raw `np.allclose()` checks, reference comparisons, key-statistic checks, and endpoint checks.
- **Final re-check**: Corrected full conversion completed in 37.39 s; validator again reports no errors/warnings; independent checks pass for neural/input/output and every session endpoint.
- **Unfixable warnings**: None. The validator emitted no warnings.

### Final Assessment
- Loading, filtering, temporal alignment, neural processing, temporal averaging, and motion-energy construction match the reference wherever applicable.
- Only task-mandated departures remain: 60-s trialization, elapsed-session-time input, and five-class within-session motion-energy discretization.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes. Training loss fell from 106.4889 (epoch 1) to approximately 1.18 (epoch 200).
- Test loss: 4.043840.
- Device: CUDA.
- Split: 863 training trials, 218 validation trials.
- Training completed normally; `train_decoder.py finished successfully.`

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-----------------------|-------------------------|-------|
| motion_energy_quintile | 0.6168 | 0.2931 | Chance 0.2000; validation is 1.47× chance |

- The complete required command, including `--plot-samples`, finished and its output is saved in `/app/train_decoder_full_out.txt`.
- The paper reports continuous ridge-regression R² rather than five-class balanced accuracy, so no directly equivalent paper accuracy exists.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Achieved Accuracy | Chance | Ratio to Chance | Expectation from Paper |
|----------|-------------------|--------|-----------------|------------------------|
| motion_energy_quintile (validation) | 0.2931 balanced accuracy | 0.2000 | 1.4655× | Paper predicts continuous motion energy and reports R² matrices/plots, not categorical accuracy; no directly comparable scalar. |
| motion_energy_quintile (training) | 0.6168 balanced accuracy | 0.2000 | 3.084× | Not directly comparable to paper's cross-validated continuous R². |

### Check 1: Accuracy vs Chance
- Validation is above chance by 0.0931 absolute (46.6% relative), demonstrating neural information about motion state.
- It is narrowly below the suggested 1.5× heuristic (1.4655×, a difference of 0.0069×). This triggered all prescribed debugging checks rather than being dismissed.
- Three specific output trials were independently reconstructed from raw motion files and match exactly: jm031/2023-10-18 trial 0, jm038/2023-05-06 trial 15, and jm046/2024-09-09 trial 28.
- Output variation is healthy: session-wide classes are essentially exactly 20% each, and 868/1,081 (80.3%) individual trials contain every class. Trials without all classes reflect sustained behavioral states, not missing labels.

### Check 2: Accuracy Comparison to Paper
| Paper decoder | Paper metric | This decoder | Comparison |
|---------------|--------------|--------------|------------|
| Same-day continuous motion-energy ridge regression | Cross-validated R², shown by day/mouse | Five-class balanced accuracy 0.2931 | Metrics/tasks differ; no valid numeric conversion between R² and accuracy. Both show above-null neural prediction of movement. |
| Cross-day continuous motion-energy ridge regression | Train-day/test-day R² matrix | Random held-out 60-s trial validation | Different split and target; paper reports developmentally varying generalization rather than one categorical score. |

- Every decoding accuracy/value reported in the paper was searched. The relevant numerical measure is R² displayed graphically; the text does not state a five-class accuracy.
- Therefore, inventing a paper accuracy or tuning conversion to match a visually estimated R² would be scientifically invalid.

### Check 3: Train vs Validation Gap
- Training/validation ratio is 2.10 (>1.5), confirming model overfit/generalization difficulty.
- Investigation found no data leakage: all 1,081 trial windows have unique start times within session, consecutive starts differ by exactly 60 s, and windows do not overlap.
- The supplied trainer randomly holds out complete trials while fitting a high-capacity nonlinear neural decoder across heterogeneous mice, ages, sessions, and neuron dimensions. Session-specific quintile thresholds intentionally normalize classes but neural-to-behavior relationships still vary across development, as the paper's cross-day R² matrices also show.
- Training loss continues much lower than test loss, consistent with model overfitting; this is not evidence of raw-to-converted mismatch.

### Prescribed Debugging Results
1. **Raw outputs**: all three specific raw trial checks pass exactly.
2. **Temporal alignment**: acquisition is hardware-synchronized and all conversion uses shared frame ordinal. Mean session population-activity/motion correlation over lags -5..+5 bins peaks with neural calcium activity one 333-ms bin after motion (0.0843 versus zero-lag 0.0699), a plausible calcium-response lag. There is no large multi-bin displacement indicating stream misalignment.
3. **Class variation**: global/session distributions are balanced; 80.3% of trials contain all classes.
4. **Neural filtering**: `iscell > 0.5`, Track2p retention, neuropil subtraction, full-recording Suite2p baseline correction, and 10-frame averaging all match reference methods and passed raw `np.allclose()` tests.
5. **Reference match**: processing and alignment were rechecked in Step 10 and no mismatch remains.

### Issues Found and Resolved
- No new conversion issue was found in the accuracy review.
- The earlier baseline-boundary issue found in Step 10 had already been fixed before full training.
- No change was made solely to increase decoder accuracy: shifting labels to compensate for physiology, using inferred spikes contrary to the paper, leaking session labels, or changing percentile scope would reduce scientific validity.

### Final Conclusion
- The categorical decoder performs robustly above chance, loss decreases, and training completes normally.
- The modest generalization score and gap are consistent with heterogeneous longitudinal recordings plus fixed trainer/model behavior. Extensive raw-value, temporal, variation, filtering, and reference checks provide no evidence of a conversion bug.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

### Final Documentation
- `/app/README.md` describes the dataset, processing, loading, target structure, key statistics, reproduction commands, and decoder result.
- `/app/CONVERSION_NOTES.md` contains the complete ordered workflow, all decisions/rationales, reference comparisons, the fixed issue, raw-value checks, validation, and decoder reviews.
- `/app/cache/README_CACHE.md` documents all investigation scripts and cached reference text.
- Temporary extracted paper text was moved to cache; primary required outputs and diagnostic/training plots remain at top level.

