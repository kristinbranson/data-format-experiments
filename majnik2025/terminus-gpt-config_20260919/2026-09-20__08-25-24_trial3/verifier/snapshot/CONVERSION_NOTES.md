# Dataset Conversion Notes

## Overview
- **Dataset**: Longitudinal Track2p mouse barrel-cortex two-photon imaging dataset
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment verification:
- Python 3.13.15
- NumPy 2.4.4
- PyTorch 2.6.0+cu124

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

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `check_nplanes` | `track2p/io/s2p_loaders.py` | LOADING/CURATION | Counts `plane*` folders and requires the same number of planes in every dataset. |
| `load_all_imgs` | `track2p/io/s2p_loaders.py` | LOADING | Loads `ops.npy`; extracts `meanImg`, optional `meanImg_chan2`, and channel count. |
| `load_all_ds_stat_iscell` | `track2p/io/s2p_loaders.py` | LOADING/CURATION | Loads `stat.npy` and `iscell.npy`; retains `iscell[:,0] == 1` by default or `iscell[:,1] > iscell_thr`. |
| `load_all_ds_ops` / `load_all_ds_mean_img` | `track2p/io/s2p_loaders.py` | LOADING | Loads per-plane Suite2p operations and mean images. |
| `load_all_ds_centroids` | `track2p/io/s2p_loaders.py` | PROCESSING | Extracts each retained ROI centroid from Suite2p stat field `med`. |
| NPY loaders | `track2p/io/loaders.py` | LOADING | Support alternate `data_npy/plane*/{rois,F,fov}.npy` representation. |
| matched Suite2p saver | `track2p/io/savers.py` | CURATION/PROCESSING | Removes match rows containing any `None`, applies matching indices to identically iscell-filtered `stat`, `F`, `Fneu`, `spks`, and optional channel-2 arrays, and writes Suite2p-compatible matched outputs. |
| Track2p orchestration | `track2p/t2p.py` | PROCESSING | Coordinates loading, registration, ROI matching, and saving across longitudinal sessions. |

### Notes
- Repository purpose: longitudinal tracking of calcium-imaging ROIs across Suite2p-processed sessions, not behavioral preprocessing.
- Native Suite2p inputs include `ops.npy`, `stat.npy`, `iscell.npy`, `F.npy`, `Fneu.npy`, and `spks.npy`; optional second-channel arrays are also supported.
- Default ROI curation is Suite2p's accepted-cell flag (`iscell[:,0] == 1`). If configured, strict probability thresholding (`iscell[:,1] > iscell_thr`) is used instead. The same mask is applied to every neuron-indexed array before Track2p indices are applied.
- All-day matched exports keep only match-matrix rows with no missing (`None`) session entry. Thus each exported row identifies the same cell in every included day.
- The output demo loads matched fluorescence matrices (`all_f_t2p`) and plots raw `F` traces. For raster visualization only, it applies `scipy.stats.zscore(..., axis=1)`. This z-score is a plotting operation, not a saved neural-data transform.
- No reference package function computes delta-F-over-F, neuropil correction, behavioral motion energy, or temporal resampling. Therefore whether those are already represented in the supplied data must be determined from the data export and paper.
- The Suite2p-to-NPY utility filters `stat` and `F` by the binary iscell mask, constructs boolean ROI masks from `ypix/xpix`, and saves `rois.npy`, filtered `F.npy`, and `fov.npy`.
- Track2p registration/matching affects cell identity across days but not fluorescence time alignment within sessions.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Hierarchy: `/app/data/<mouse>/<YYYY-MM-DD_a>/`.
- Six mice: `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`.
- Session counts: jm031=7, jm032=7, jm038=7, jm039=7, jm040=6, jm046=7 (41 total).
- Each session has one `suite2p/plane0/` folder containing `F.npy`, `Fneu.npy`, `spks.npy`, `iscell.npy`, `stat.npy`, and `ops.npy`.
- Each session has `move_deve/` containing `motion_energy_glob.npy`, `tstamps.npy`, and `interframe_int.npy`.
- No README, table, Track2p match matrix, or other metadata file is present in the data tree.

### Available Variables
| Variable | Shape / dtype | Meaning and observations |
|----------|---------------|--------------------------|
| `F` | neurons x frames, float32 | Suite2p ROI fluorescence; finite, global range 0 to 4004.24. |
| `Fneu` | neurons x frames, float32 | Suite2p neuropil fluorescence; finite, range 8.97 to 1208.12. |
| `spks` | neurons x frames, float32 | Suite2p deconvolved activity; finite, nonnegative, range 0 to 1489.56. |
| `iscell` | neurons x 2, float64 | Accepted-cell flag and classifier probability. Every supplied row has flag 1; probabilities in inspected session are 0.505–0.990. |
| `stat` | neurons, object | Suite2p ROI geometry/statistics (`med`, `xpix`, `ypix`, morphology, neuropil mask, etc.). |
| `ops` | dict | Suite2p configuration/results. `fs=30`, `nplanes=1`, `nchannels=2`, `neucoeff=0.7`, `tau=0.3`, baseline mode `maximin`. |
| `motion_energy_glob` | frames, uint64 | Nonnegative global framewise motion energy; finite, global range 0 to 65,795,161. |
| `tstamps` | frames, float64 | Movement-frame timestamps starting at zero. Numerical units require multiplication by 1000 to yield seconds: median increments become 0.033579–0.033617 s. |
| `interframe_int` | frames-1, float64 | `diff(tstamps)` in the same scaled units. |

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (unique longitudinal matches, summed once/mouse) | 2,998 |
| Neuron-session observations | 20,445 |
| Neurons / session | constant by mouse: 221, 370, 685, 746, 541, 435 (mean 498.7 session observations) |
| Subjects | 6 |
| Sessions / subject | 6–7 |
| Sessions total | 41 |
| Trials (native) | None; sessions are continuous recordings |
| Session frames | 36,000 (14 sessions) or 54,000 (27 sessions) |
| Total neural frames across sessions | 1,962,000 |
| Recording duration | approximately 20 or 30 minutes |

### Data-quality and alignment observations
- The constant neuron count and ordering within each mouse, together with all `iscell` flags being accepted, indicate the files are already curated longitudinal Track2p matched outputs rather than unfiltered raw Suite2p detections.
- Neural `F`, `Fneu`, and `spks` dimensions agree within every session.
- Motion length equals neural length in 32/41 sessions. Nine sessions have fewer motion samples: deficits 1 frame (3 sessions), 2 (3), 3 (1), 116 (1), and 148 (1). These deficits occur at the session end because each movement series begins at timestamp zero; conversion must use only the shared valid prefix rather than invent behavioral values.
- All scanned neural and motion arrays contain finite values.
- Native data have no trials. The requested 60-second trials must be constructed from each shared valid continuous interval.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects | 6 mice | “A total of 6 mice were used in the study” |
| Sessions / subject | at least 6 consecutive days | “Longitudinal two-photon calcium imaging was performed for each mouse for at least 6 consecutive days” |
| Neurons | hundreds tracked per mouse; example 728/1,988 first-day ROIs | “yielded a total of 728 ROIs that were tracked across all days in this example mouse” |
| Brain region | barrel cortex, layer 2/3 | Imaging at 100–200 µm depth under a barrel-cortex cranial window. |
| Neural data time bin | 30 Hz native; 10-frame means for decoding | “Imaging rate was 30 Hz”; decoding averaged “10 consecutive timestamps.” |
| Behavior data time bin | video 30 Hz, hardware triggered by microscope; 10-frame means for decoding | “Videos were recorded at 30 Hz, with the microscope acquisition acting as a trigger.” |
| Session duration | nominally 20 minutes | “each session lasted 20 minutes” (supplied export also contains 30-minute sessions; addressed in Step 4). |
| Behavioral variable | global motion energy | Sum over pixels of squared differences between consecutive video frames. |
| Animals | heterozygous GAD67-Cre mice | Methods, Animals section. |
| Developmental period | daily recordings during second postnatal week (approximately P8–P14) | Main text and figures. |

### Processing Details
- Calcium imaging and videography were both acquired at 30 Hz; microscope acquisition triggered each camera frame, providing direct framewise synchronization.
- Suite2p separately performed motion correction, ROI detection, signal extraction, and spike deconvolution for each session.
- The authors state: “We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters) for all subsequent analyses.” This requires checking whether the supplied `F.npy` is raw or already transformed.
- Motion energy is the scalar sum across all pixels of squared differences between consecutive video frames and is used as a proxy for global movement/arousal.
- Calcium-event analysis averaged traces in 10-frame bins before peak detection. Decoder analysis likewise averaged both dF/F and behavior in non-overlapping bins of 10 consecutive timestamps (effective approximately 3 Hz).
- Same-day paper decoding used ridge regression with nested 5-fold inner/outer cross-validation. Splits were consecutive 2-minute blocks. Cross-day models were fit on one day and evaluated on all others.

### Curation Steps

**Neuron curation rules**:
- Suite2p ROIs with classifier probability strictly above the default 0.5 threshold were considered cells.
- Track2p matched ROIs across consecutive sessions and propagated matches.
- Only cells successfully tracked across every session for a mouse were used in downstream analyses.

**Trial curation rules**:
- The reference experiment is continuous spontaneous behavior, not trial-based. Paper cross-validation uses consecutive 2-minute blocks; the decoder task instead explicitly requires constructing 60-second trials.

### Decoders Trained
| Decoded variable | Accuracy |
|------------------|----------|
| Continuous global motion energy from population dF/F | Evaluated by R², not categorical accuracy. Same-day performance increased with development; once developed, cross-day decoding was described as accurate and stable. The text provides no single scalar R². |

### Relevance to requested decoder
- The requested output changes continuous motion into five per-session equal-percentile classes, so paper R² cannot be directly compared to five-class balanced accuracy (chance = 20%).
- The required 60-second trials differ from the paper's 2-minute cross-validation blocks and are followed as an explicit downstream specification.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Cell filtering | Track2p defaults to Suite2p binary iscell or configurable strict probability threshold; matched exports remove rows missing any day. | All provided rows have `iscell[:,0]=1`; each mouse has a constant neuron count and ordering across days. | Probability >0.5; only neurons tracked through all sessions used downstream. | Data are already cell-filtered and all-day matched. Retain every supplied row; do not filter a second time. |
| Neural signal | `F.npy` is raw ROI fluorescence; Suite2p preprocessing uses neuropil-corrected signal and maximin baseline subtraction. Track2p GUI has a similarly named display transform. | F has raw fluorescence scale (roughly tens to thousands), not normalized dF/F. `Fneu` and ops parameters are present. | “baseline corrected fluorescence traces as our dF/F (using default Suite2p parameters).” | Reproduce default Suite2p analysis processing: `Fc=F-neucoeff*Fneu`, smooth with `sig_baseline`, rolling minimum then maximum over `win_baseline*fs`, and subtract baseline. Do not divide by baseline. |
| Decoder temporal denoising | Demo plots raw F; paper-specific analysis code is not in package. | Neural and motion streams are native framewise arrays. | Both dF/F and behavior averaged over 10 consecutive timestamps for decoding. | Apply non-overlapping 10-frame means to both streams, yielding 10/30 s = 333.333 ms bins (3 Hz). |
| Temporal alignment | Track2p does not alter within-session time. | Neural and motion arrays start together; motion is shorter by 1–148 terminal frames in 9 sessions. Timestamp numeric increments become ~0.0336 s after x1000. | Camera frames were triggered by microscope acquisition at 30 Hz for simple synchronization. | Align by frame index and use only the shared valid prefix. Never interpolate or fabricate missing terminal behavior. Use nominal 30 Hz for bin duration and elapsed-time construction. |
| Session duration | Not imposed by Track2p. | 14 sessions have 36,000 frames; 27 have 54,000 frames. | Methods state each session lasted 20 minutes. | Preserve all valid supplied data. The 30-minute sessions are internally consistent and likely reflect an expanded/exported cohort; arbitrary truncation would discard valid synchronized data. Document converted trial counts. |
| Trial definition | Reference decoder splits on consecutive 2-minute blocks for cross-validation. | Native recordings are continuous. | Same as code/methods. | Explicit downstream instruction overrides this: create non-overlapping 60-second trials. |
| Decoder target | Paper predicts continuous motion with ridge regression and reports R². | Continuous motion-energy arrays are supplied. | Same-day R² rises with age; mature representation transfers across days. | Explicit downstream instruction overrides output type: discretize 10-frame mean motion into five per-session percentile bins. |

### Final Consistent Understanding
- The source is a preprocessed, longitudinally curated export: six mice, 41 sessions, and 2,998 neurons tracked across every available day within each mouse.
- Neural processing for analysis is Suite2p neuropil correction plus its default maximin baseline subtraction, followed by the paper's non-overlapping 10-frame temporal averaging (333.333 ms bins at 30 Hz).
- Motion energy has already been computed exactly as described in the paper and only needs the same 10-frame averaging.
- Neural and video streams are synchronized one-to-one by acquisition trigger. Shared-prefix trimming handles dropped terminal video frames robustly.
- The task-required 60-second trialization and five-class target are deliberate deviations from the paper's continuous ridge-regression setup.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `F.npy`, `Fneu.npy`, `ops.npy` | `neural` | `Fc=F-neucoeff*Fneu`; Gaussian smoothing, rolling min then rolling max baseline; subtract baseline; non-overlapping mean over 10 frames; split into 60-s trials | Suite2p `dcnv.preprocess`; Track2p `F_processing`; paper Methods | float32, shape `(n_neurons,180)` per trial. |
| Frame index and `ops['fs']` | `input[0]` | Elapsed session time at centers of each 10-frame average: `(10*k+4.5)/fs`; split with neural | Hardware synchronization in Methods | float32 seconds, shape `(1,180)`. Nominal 30 Hz is used because fixed bins are required; timestamps are a camera diagnostic and show only small clock-rate deviation. |
| `motion_energy_glob.npy` | `output[0]` | Shared-prefix alignment; non-overlapping 10-frame mean; compute 20/40/60/80% quantiles per session over retained complete trials; `searchsorted(..., side='right')` to labels 0–4 | Motion preprocessing and 10-frame decoder binning in Methods | int64, time-varying shape `(1,180)`. Ties at boundaries deterministically enter the higher bin; motion is sufficiently continuous that ties should be rare. |
| Mouse folder name | `subjects`, `subject_idx` | Sorted unique mouse IDs and per-session lookup | Data hierarchy | Six subjects. |
| Experiment anatomy | `brain_regions`, `brain_region_idx` | Single region `barrel cortex L2/3`; zeros for all neurons | Paper Methods | One region. |
| Session folder/date and processing values | `metadata['session_info']` | Store subject/session ID, source/used frame counts, dropped terminal frames, neurons, trials, fps, quantile edges | Conversion-specific provenance | Enables auditing all curation. |

### Key Decisions
1. **Use baseline-subtracted, neuropil-corrected fluorescence**: This exactly follows the paper's default-Suite2p baseline-corrected signal. Literal division by the baseline was rejected because the repository and Suite2p both subtract, and division creates extreme artifacts around nonpositive baselines.
2. **Retain every supplied neuron**: The export is already probability-filtered and Track2p-matched across all days; a second filter would incorrectly discard curated cells.
3. **Average aligned blocks of 10 raw frames**: Required to match paper decoding. At 30 Hz this is 333.333 ms and 3 samples/s.
4. **Create non-overlapping 60-second trials**: Each trial is exactly 1,800 raw frames or 180 processed samples. Only complete trials are retained.
5. **Handle missing behavior by shared-prefix trimming**: Motion deficits occur only at the end. No interpolation, extrapolation, or padding is justified. This yields 1,081 trials total.
6. **Discretize per session after temporal averaging and complete-trial selection**: This follows “five equal-percentile bins, selected per session” and ensures thresholds describe exactly the saved output population.
7. **Use session-elapsed bin-center times**: Averaged observations represent the centers of their 10-frame windows. Each trial retains absolute session time rather than resetting to zero, as explicitly requested.
8. **Preserve 30-minute source sessions**: Although the paper states nominal 20-minute recordings, these arrays are internally valid and synchronized. Truncation would lose source data without a curation rule.
9. **No z-scoring**: The reference uses z-score only for plotting, not saved analysis traces.

### Planned Sanity Checks
- [ ] Directly reload raw `F`, `Fneu`, and ops for selected sessions; independently reconstruct a saved neural trial and require `np.allclose`.
- [ ] Directly reload raw motion; independently average and discretize selected trial samples and require `np.allclose` for averages and labels.
- [ ] Independently construct elapsed-time values and require `np.allclose` to saved input.
- [ ] Confirm every trial has neural/input/output time length 180 and every session has at least 19 trials.
- [ ] Confirm all arrays are finite, output labels are in 0–4, and per-session class fractions are near 0.2.
- [ ] Confirm converted subject/session/neuron counts equal source counts and trial count equals 1,081.
- [ ] Confirm 10-frame binning and 60-second boundaries have no off-by-one errors at session starts/ends.
- [ ] Compare baseline-correction implementation line-by-line with reference Track2p/Suite2p code.

---

## Step 6: Script Development
**Status**: COMPLETE

- Created `/app/convert_data.py`; syntax compilation passes.
- CLI supports positional output path, mutually exclusive `--full` (default) / `--sample`, and `--show-processing`.
- Functions separately implement discovery, Suite2p baseline correction, 10-frame means, session conversion, processing plots, and structural validation.
- Baseline estimation uses each complete neural session, matching Suite2p preprocessing; behavior availability affects only the retained shared trial interval.
- Every session prints processing time, neuron/trial counts, and output class counts.

Code inefficiencies identified:
- Suite2p maximin baseline filtering necessarily touches each full fluorescence matrix and is the primary compute/memory cost.
- Storing all final trial arrays is required by the target pickle format.

Code speedups added:
- Source fluorescence arrays are initially memory-mapped.
- Filtering and temporal binning are vectorized over neurons/time.
- Data are cast to float32/int64 once and sliced into contiguous trial arrays.
- Sessions are processed sequentially so temporary full-session matrices are released before the next session.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (unique) | 221 |
| Neuron-session observations | 442 |
| Subjects | 1 (`jm031`) |
| Sessions / subject | 2 |
| Trials (total) | 40 |
| Trials / session | 20, 20 |
| Timepoints / trial | 180 |
| Neural range | session 1: [-45.11, 824.29]; session 2: [-47.94, 1144.54] |
| Elapsed-time range | [0.15, 1199.817] s |
| Motion-quintile distribution | [0.200, 0.200, 0.200, 0.200, 0.200] in each session |

### Format Validation
- `/app/verification_sample_out.txt` reports: “Data format is valid, no errors or warnings.”
- Neural/input/output first-trial shapes are `(221,180)`, `(1,180)`, and `(1,180)`.
- Trial-1 bin centers are 0.15–59.8167 s; trial 2 starts at 60.15 s, confirming no overlap/gap beyond the expected 1/3-s sample spacing.
- Metadata accurately records processing parameters, source/used frame counts, quantile edges, and class counts.

### Processing Plots Review
- Created `processing_jm031_2023-10-18_a.png` and `processing_jm031_2023-10-19_a.png` (553 and 595 KiB).
- Each plot includes raw F, Fneu, neuropil-corrected F, estimated baseline, final 10-frame neural means, raw and averaged motion, quantile thresholds, and class histogram.
- Plotted series share the same session-time axes; class histograms are exactly balanced. No missing/empty plot panels or malformed image files were observed.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|-----------------------|--------------|
| Vectorized filtering/binning; sequential sessions; memory mapping | Sample completed in 2.27 s |

| Step | Time / Session | Estimated Total Time |
|------|----------------|----------------------|
| Conversion processing | 1.13 s | ~47 s for 41 sessions, plus serialization |

Full conversion is safely below the 15-minute optimization threshold.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-----------------------|-------------------------|
| Motion energy quintile | 0.5373 | 0.3073 |

- Chance is 0.2000; validation is 1.54× chance.
- Loss decreased from 57.329 at epoch 1 to 1.275 at epoch 200; test loss was 2.624.
- Training finished successfully.
- The sample training/validation ratio is 1.75, suggesting overfitting with only two same-mouse sessions; this will be reassessed on the complete 41-session dataset in Steps 11–12 rather than altering correct conversion logic based on a deliberately small sample.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 393 MiB
- `conversion_full_out.txt`: created
- `verification_full_out.txt`: created; no errors or warnings

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Unique longitudinal neurons | hundreds/mouse; example 728 | all-day match rows retained | 2,998 across mice | 2,998 | Yes |
| Neuron-session observations | not stated | matched rows/session | 20,445 | 20,445 | Yes |
| Subjects | 6 | N/A | 6 | 6 | Yes |
| Sessions | at least 6/mouse | arbitrary longitudinal list | 41 (6–7/mouse) | 41 | Yes |
| Trials (total) | continuous data | 2-min CV blocks in paper | none natively | 1,081 task-required 60-s trials | Expected task deviation |
| Trials/session | N/A | N/A | 19–30 complete shared minutes | 19–30 | Yes |
| Timepoints/trial | N/A | 10-frame averaging | 1,800 raw frames/trial | 180 averaged bins | Yes |
| Neural sampling | 30 Hz; average 10 | maximin baseline subtraction | 30 Hz | 333.333 ms bins | Yes |
| Input range | N/A | N/A | session durations 20/30 min | 0.15–1799.817 s | Yes |
| Output distribution | continuous motion | N/A | continuous nonnegative | [0.199995, 0.200005, 0.200000, 0.200000, 0.200000] | Yes, requested quintiles |

### Full Statistics and Integrity
- Conversion finished in 36.43 s (0.87 s/session), well within estimate and optimization threshold.
- All trial time lengths are exactly 180.
- Neural range after processing: -294.99 to 2808.03; all values finite.
- Class counts: [38,915, 38,917, 38,916, 38,916, 38,916]. The two-sample imbalance is caused by tied values at a quantile edge in one session and is negligible.
- Spot-checked sessions at indices 0, 2, 14, 25, 34, and 40. Source frame counts, used-frame counts, neuron counts, trial counts, and final elapsed times agree with metadata.
- No data were silently lost: only incomplete terminal 60-second intervals are excluded, as documented per session.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output-log verification**: Read `verification_full_out.txt`; validator reports “Data format is valid, no errors or warnings.” Explicit grep found no error/warning/failure messages.
2. **Independent raw neural check**: `critical_checks.py` directly loads original `F`, `Fneu`, and ops, independently repeats neuropil correction, maximin baseline subtraction, and 10-frame averaging, then uses `np.allclose` against saved neural arrays. Passed for jm031/2023-10-18 trial 5, jm031/2023-10-22 trial 18, and jm039/2024-05-04 trial 28.
3. **Independent raw input check**: Independently computes 10-frame bin-center elapsed times from raw frame indices and 30 Hz; `np.allclose` passes for the same three sessions/trials.
4. **Independent raw output check**: Directly loads raw motion, averages 10-frame blocks, computes retained-session quantiles and labels without importing conversion code; exact `np.allclose` passes for all three selected trials.
5. **Reference processing comparison**:
   - **Loading**: converter loads Suite2p `F`, `Fneu`, `iscell`, `ops` and movement arrays exactly as the reference loaders do.
   - **Neuron filtering**: reference applies Suite2p iscell and Track2p all-day matching; supplied export is already filtered/matched (all flags 1 and fixed per-mouse row count), so converter verifies rather than reapplies filtering.
   - **Temporal alignment**: paper says camera was microscope-triggered; converter aligns frame indices and trims only unmatched terminal frames.
   - **Binning**: paper averages 10 consecutive timestamps for both dF/F and behavior; converter uses aligned non-overlapping blocks of exactly 10.
   - **Input construction**: elapsed session time is task-required and built at averaged-bin centers; no corresponding paper variable exists.
   - **Output construction**: source motion energy is used unchanged apart from reference 10-frame averaging, then task-required per-session quintile discretization.
6. **Key statistics comparison**: Source and converted data agree on six mice, 41 sessions, 2,998 unique tracked neurons, 20,445 neuron-session observations, subject/session mapping, and all retained source neuron counts. The 1,081 converted trials equal the independently computed count of complete 1,800-frame shared intervals.
7. **Edge cases/off-by-one checks**: Every trial has 180 bins; first/last centers are trial start +0.15/+59.8167 s; adjacent bins differ by 1/3 s. Sessions short by 1, 2, 3, 116, or 148 motion frames drop only the incomplete terminal minute. Output labels remain 0–4 and per-session fractions differ from 0.2 by <0.001.
8. **Finite/range checks**: All converted neural/input/output arrays are finite; every session contains at least 19 trials and all five output classes.

### Issues Found and Resolved
- **Metadata alignment wording**: Initial metadata named session start as the alignment event while `off_start=0`, `off_end=60` referred to trial boundaries. Fixed `convert_data.py` to define the event as each 60-second trial start while retaining the explicit session-elapsed input description.
- **Revalidation after fix**: Regenerated sample and full pickles; sample/full validator runs passed without warnings; reran all independent checks and all passed.
- **Quantile ties**: One session has two tied samples crossing an edge, producing class counts [683,685,684,684,684]. This is not a failure: exact equal-size groups are impossible without arbitrarily separating identical motion values. Deterministic `searchsorted(side='right')` preserves equal-value semantics; global imbalance is two of 194,580 samples.
- **Paper duration discrepancy**: Text says nominal 20-minute sessions while 27 supplied sessions contain internally consistent 30-minute neural/video data. Preserving valid source data is preferable to unsupported truncation; all duration details are retained in metadata.

`critical_checks_out.txt` ends with `ALL CRITICAL CHECKS PASSED`.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes, from 84.4156 at epoch 1 to 1.1138 at epoch 200.
- Test loss: 3.0653.
- Training used CUDA, 863 training trials and 218 validation trials.
- `train_decoder.py` finished successfully and generated sample/prediction plots.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-----------------------|-------------------------|-------|
| Motion energy quintile | 0.6118 | 0.3072 | Chance 0.2000; validation is 1.536× chance. |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Achieved Accuracy | Chance / Expectation from Paper |
|----------|-------------------|---------------------------------|
| Motion energy quintile | Train 0.6118; validation 0.3072 balanced accuracy | Chance 0.2000. Validation is 1.536× chance. Paper reports continuous-motion R² qualitatively increasing with development and stable cross-day decoding after emergence; no categorical accuracy is reported. |

### Checks and Findings
1. **Accuracy versus chance**: Validation balanced accuracy is above chance and just above the requested 1.5×-chance investigation threshold. It is consistent with meaningful neural information about movement.
2. **Accuracy versus paper**: Every paper decoding statement was reviewed. The paper predicts continuous motion with ridge regression and evaluates R²; it does not report five-class accuracy. Therefore no numerically valid paper-to-current accuracy conversion exists. Qualitatively, above-chance decoding agrees with the paper's finding that population activity predicts motion, while pooling early developmental sessions reasonably limits overall performance.
3. **Train/validation gap**: Training/validation ratio is 1.99 (>1.5), so overfitting was investigated. The provided decoder randomly partitions whole 60-second trials within every session; all 41 sessions have both train and validation trials (863/218), and no trial occurs in both sets. Thus there is no trial leakage. The gap is plausible for session-specific projections from 221–746 neurons with only 19–30 independent trials/session and 200 training epochs.
4. **Raw outputs checked**: Direct-source reconstruction and exact saved-label checks passed for jm031/2023-10-18 trial 5, jm031/2023-10-22 trial 18, and jm039/2024-05-04 trial 28.
5. **Temporal variation checked**: The three checked trials contain 3–5 classes and 34–98 class transitions. Across every session all five classes occur at approximately 20%; there is no 99%-dominant class.
6. **Temporal alignment checked**: Independent raw reconstruction verifies synchronized 10-frame neural/motion blocks with `np.allclose`; processing plots show shared axes and prediction/sample plots are valid nonempty PNGs.
7. **Filtering/processing checked**: Supplied cells are already Suite2p-filtered and all-day Track2p-matched. Neural baseline correction, temporal averaging, and motion processing match the paper/code as documented in Step 10.
8. **Training behavior**: Loss decreased monotonically overall from 84.4156 to 1.1138. Full validation (0.3072) closely reproduces sample validation (0.3073), arguing against a session-count or formatting instability.

### Issues Found and Resolved
- No conversion defect was found during low-accuracy debugging. Altering labels, alignment, or filtering to optimize the provided decoder would depart from raw data and reference processing.
- The train/validation gap is documented as model overfitting rather than data leakage: trial splits are disjoint, every session is represented on both sides, validation remains >1.5× chance, and all raw/alignment checks pass.
- The paper comparison is necessarily qualitative because R² for a continuous target and balanced accuracy for five percentile classes are different metrics and targets.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created with loading instructions, processing summary, structure, and key statistics
- [x] cache/ folder created
- [x] Investigation script and output moved to cache and documented in `cache/README_CACHE.md`
- [x] Required pickles, logs, plots, conversion script, and notes retained in `/app`
- [x] All files organized
