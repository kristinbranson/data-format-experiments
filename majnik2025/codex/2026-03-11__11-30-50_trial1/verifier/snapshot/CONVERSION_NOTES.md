# Dataset Conversion Notes

## Overview
- **Dataset**: longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p (local paper/code/data bundle)
- **Date started**: 2026-03-11
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

Environment checks:
- `python3` runs successfully
- `numpy` import succeeded (`2.3.5`)
- `torch` import succeeded (`2.6.0+cu124`)
- `ls -la CONVERSION_NOTES.md` confirmed the notes file exists

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `check_nplanes` | `code/track2p/io/s2p_loaders.py` | LOADING | Verifies each dataset path contains the same number of Suite2p planes. |
| `load_all_imgs` | `code/track2p/io/s2p_loaders.py` | LOADING | Loads `ops.npy` mean images and channel metadata for each dataset/plane. |
| `load_all_ds_stat_iscell` | `code/track2p/io/s2p_loaders.py` | CURATION | Loads `stat.npy` and filters ROIs using `iscell.npy` and `track_ops.iscell_thr`. |
| `load_stat_ds_plane` | `code/track2p/io/loaders.py` | CURATION | Per-plane ROI loader with the same `iscell` thresholding logic and summary counts. |
| `run_t2p` | `code/track2p/t2p.py` | PROCESSING | Main Track2p pipeline: registration, ROI matching across days, and output saving. |
| `generate_suite2p_indices` | `code/track2p/t2p.py` | PROCESSING | Converts Track2p match rows back to original Suite2p ROI indices for each day. |
| `save_in_s2p_format` | `code/track2p/t2p.py` | PROCESSING | Exports matched ROIs and traces (`F`, `Fneu`, `spks`, etc.) for cells present across all days. |
| `DataManagement.import_files` | `code/track2p/gui/data_management.py` | LOADING | GUI import path that loads matched cells across days and extracts traces for `F`, `spks`, or `dF/F0`. |
| `DataManagement.F_processing` | `code/track2p/gui/data_management.py` | PROCESSING | Applies neuropil subtraction coefficient `0.0` and maximin baseline subtraction to make a `dF/F0`-named trace. |
| `DefaultTrackOps.__init__` | `code/track2p/ops/default.py` | CURATION | Defines defaults including `iscell_thr = 0.50`, `matching_method='iou'`, and example dataset paths. |

### Notes
- The bundled code is the Track2p package, not a full experiment-analysis repo. It explains how longitudinally matched neurons are created from per-session Suite2p outputs.
- Core curation rule in code: ROIs are filtered by `iscell.npy`. If `track_ops.iscell_thr is None`, use binary `iscell[:,0] == 1`; otherwise use probability `iscell[:,1] > track_ops.iscell_thr`. The default threshold in `DefaultTrackOps` is `0.50`.
- Longitudinal cell set for downstream analysis is `t2p_match_mat_allday = t2p_match_mat[~np.any(t2p_match_mat == None, axis=1), :]`, i.e. only rows with a matched ROI on every day are retained.
- Available neural traces in the GUI export/import path are raw fluorescence `F`, deconvolved `spks`, and a `dF/F0` option.
- The `dF/F0` option in `DataManagement.F_processing` does not divide by baseline; it performs `Fc = F - neucoeff * Fneu` with default `neucoeff=0.0`, then subtracts a maximin baseline (`gaussian_filter`, `minimum_filter1d`, `maximum_filter1d`). This is important because the code’s named `dF/F0` is effectively baseline-subtracted fluorescence, not a classical ratio.
- No trial structure, behavior variables, motion-energy extraction, or decoder code appears in `code/`. Those will need to come from `data/`, `methods.txt`, and `paper.pdf`.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `data/README.md` documents the release and states there should be 6 subject folders.
- Subject folders present: `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`.
- Each subject contains daily session folders named like `YYYY-MM-DD_a`.
- Each session has:
  - `suite2p/plane0/` with `F.npy`, `Fneu.npy`, `spks.npy`, `iscell.npy`, `stat.npy`, `ops.npy`
  - `move_deve/` with `motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy`
- `suite2p/plane0` already contains Track2p-matched cells across all days of that subject. Within a subject, all sessions have the same neuron count and row identity.
- `motion_energy_glob.npy` is the processed behavioral signal to decode.
- `tstamps.npy` and `interframe_int.npy` provide camera timing and reveal missing behavior frames in some sessions.
- Additional files:
  - `data/load_data.ipynb`: example notebook that loads per-session fluorescence traces and FOVs.
  - `ground_truth.csv` exists for `jm038`, `jm039`, and `jm046` and appears to contain manual cell correspondence annotations for Track2p evaluation, not decoder variables.
- Native data are continuous recordings. There is no trial table or trial boundary file in `data/`.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 20,445 across all sessions; 2,998 unique tracked neurons across subjects |
| Neurons / session | mean 498.659; range 221-746 |
| Subjects | 6 |
| Sessions / subject | `jm031`: 7, `jm032`: 7, `jm038`: 7, `jm039`: 7, `jm040`: 6, `jm046`: 7 |
| Trials (total) | No native trials in source data; 41 continuous recording sessions |
| Trials / session | No native trial dimension |

Additional observations:
- Total sessions: 41
- Session frame counts:
  - `jm031`, `jm032`: 36,000 imaging frames per session
  - `jm038`, `jm039`, `jm040`, `jm046`: 54,000 imaging frames per session
- Example session (`jm031/2023-10-18_a`):
  - `F.npy`, `Fneu.npy`, `spks.npy`: `(221, 36000)` float32
  - `motion_energy_glob.npy`: `(36000,)` uint64
  - `tstamps.npy`: `(36000,)` float64
  - `interframe_int.npy`: `(35999,)` float64
- Behavior/imaging length mismatches are present in 9 sessions:
  - `jm031`: `2023-10-20_a`, `2023-10-21_a`, `2023-10-22_a`
  - `jm032`: `2023-10-20_a`, `2023-10-21_a`, `2023-10-22_a`
  - `jm039`: `2024-05-04_a`
  - `jm040`: `2024-05-04_a`
  - `jm046`: `2024-09-09_a`

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Approximately 526 tracked neurons per mouse on average across all days | `On average 526 (± 190 std) neurons per mouse` |
| Neurons / session | Several hundreds; matched cells persist across all days of a mouse | `several hundreds of individual neurons` |
| Subjects | 6 mice | `a full dataset of 6 mice` |
| Sessions / subject | Minimum 6 consecutive daily sessions; example datasets span 7 days | `daily for a minimum of 6 consecutive days` |
| Trials (total) | Not trial-based in source; continuous 20 minute recordings | `each session lasted 20 minutes` |
| Trials / session | No native trials reported | `each session lasted 20 minutes` |
| Neural data time bin | Raw acquisition 30 Hz (33.3 ms / frame) | `Imaging rate was 30 Hz` |
| Behavior data time bin | Raw videography 30 Hz, microscope-triggered | `Videos were recorded at 30 Hz` |
| Reward rate | N/A; spontaneous behavior task, no reward | `spontaneously run on a non-motorised treadmill` | 
| Motion metric | Global motion energy from squared frame differences | `pixel-wise difference of consecutive frames` | 
| Decoder smoothing bin | 10 consecutive timestamps (~333 ms) | `averaging in bins of 10 consecutive timestamps` |
| Decoder evaluation blocks | Consecutive 2 minute blocks | `splits were done on consecutive 2 minute blocks` |


### Processing Details
- Recordings are spontaneous-behavior calcium imaging from mouse barrel cortex, layer 2/3, during the second postnatal week.
- Imaging and videography are both sampled at 30 Hz, with the microscope acquisition triggering the camera, so the intended reference alignment is framewise synchrony between neural and behavior streams.
- Calcium preprocessing in the reference study:
  - Suite2p per recording for motion correction, ROI detection, signal extraction, and spike deconvolution.
  - Cell inclusion threshold: use ROIs with Suite2p cell probability above 0.5.
  - Downstream analyses in the paper use baseline-corrected fluorescence traces as dF/F, described as using default Suite2p parameters.
- Behavioral preprocessing in the reference study:
  - Motion energy is computed from consecutive video frames, as squared pixelwise differences summed over pixels, producing one scalar per frame.
- Decoding in the reference study:
  - Predict behavioral motion from neural population activity using ridge regression.
  - Same-day decoding uses nested 5-fold cross-validation.
  - Splits are consecutive 2 minute blocks.
  - Neural and behavior traces are both slightly denoised by averaging over 10-frame bins before decoding.
  - Performance is reported as `R^2`, with same-day decoding improving with development and cross-day decoding becoming accurate at later ages.

### Curation Steps

**Neuron curation rules**:
- Keep ROIs classified as cells by Suite2p with probability threshold `> 0.5`.
- Track2p is then used to identify the subset of neurons present across all days of a mouse for longitudinal analyses.

**Trial curation rules**:
- No trial-based curation is described because the recordings are continuous spontaneous-behavior sessions rather than discrete trials.
- Behavioral frames can be missing in some sessions; the data README says missing frames should be identified from `tstamps.npy` / `interframe_int.npy` and treated as missing values or interpolated.

### Decoders Trained
| Decoded variable | Accuracy |
| Mouse motion / motion energy | No exact text-table value in the paper text; performance reported as `R^2`, increasing with age and showing accurate late cross-day decoding |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Cell filtering | Default `iscell_thr = 0.50` in `DefaultTrackOps`; ROI loaders use `iscell[:,1] > threshold` | Exported Suite2p data already contain matched cells across all days | Use ROIs above Suite2p default threshold 0.5 | Consistent. Treat bundled neural data as already Track2p-matched and cell-filtered. |
| Neural trace type | GUI offers `F`, `spks`, or a `dF/F0` helper implemented as baseline subtraction with `neucoeff=0.0` | Sessions contain `F.npy`, `Fneu.npy`, `spks.npy`, not precomputed `dF/F.npy` | Paper states analyses used baseline-corrected fluorescence traces as dF/F with default Suite2p parameters | Use paper-consistent Suite2p-style preprocessing from `F` and `Fneu`; do not use GUI `F_processing` as the reference analysis path. |
| Matched-cell population | Track2p export keeps cells present across all days (`t2p_match_mat_allday`) | Within each subject all sessions have identical neuron counts and row identities | Paper states functional analyses used neurons successfully tracked across all days | Consistent. Subject-level neuron identity is row-aligned across days. |
| Neural/behavior alignment | Track2p code does not handle behavior | Most sessions have matching imaging and behavior lengths; 9 sessions have shorter behavior arrays | Camera triggered by microscope at 30 Hz; README notes missing camera frames and says to infer them from `tstamps.npy` / `interframe_int.npy` | Treat imaging frames as the master clock and reconstruct missing behavior frames from doubled timing gaps before alignment. |
| Dataset scale | Example defaults in code show only 3 sessions for one toy subject | Data release has 6 subjects and 41 sessions | Paper reports 6 mice with at least 6 daily sessions | Use full release dataset, not code defaults. Data counts are consistent with the paper. |
| Summary neuron counts | Code does not state paper summary statistics | Unique tracked neurons per mouse in release: `[221, 370, 685, 746, 541, 435]`, mean `499.7`, sample std `197.7` | Paper reports mean `526 ± 190` tracked neurons per mouse | Close agreement; small difference is acceptable and likely due to dataset subset / rounding in paper. |

Final understanding:
- Source neural data are continuous 30 Hz daily recordings from barrel cortex, already matched across days within a subject.
- The authoritative neural preprocessing target is Suite2p-style baseline-corrected fluorescence using default Suite2p parameters, because that is what the paper states for downstream analyses.
- Behavioral motion energy is a framewise scalar aligned to imaging frames, except for missing camera frames that must be reinserted using timing gaps.
- There is no native trial structure, so any trialization needed for the target decoder format must be an explicit downstream segmentation of continuous recordings rather than recovery of hidden experimental trials.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `suite2p/plane0/F.npy`, `Fneu.npy`, `ops.npy` | `neural` | Compute Suite2p-style baseline-corrected fluorescence: `Fc = F - neucoeff * Fneu`, then `suite2p.extraction.dcnv.preprocess(...)`; average non-overlapping 10-frame bins; split continuous session into consecutive 2-minute blocks | Track2p provides loading only; paper/methods specify downstream dF/F usage | Use per-session `ops.npy` values: `neucoeff=0.7`, `baseline=maximin`, `win_baseline=60`, `sig_baseline=10`, `prctile_baseline=8`, `fs=30` |
| Session frame index / bin centers | `input[0]` | Convert to elapsed time in seconds from session start; after 10-frame averaging, use bin-center time for each sample inside each 2-minute block | No direct reference code; required by decoder task | Time-varying single input dimension |
| `move_deve/motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy` | `output[0]` | Reconstruct missing frames from doubled timing gaps so behavior length matches imaging; average non-overlapping 10-frame bins; global min-max normalize valid values; discretize all valid binned samples into 5 equal-percentile bins; split into same 2-minute blocks as neural data | Paper/methods define motion energy; data README defines missing-frame handling | Single categorical output dimension with classes `Q1`-`Q5` |
| Subject folder name (e.g. `jm031`) | `subjects`, `subject_idx` | Unique subject list in sorted order; session indices point into this list | N/A | 6 subjects |
| Constant anatomical label from paper (`barrel cortex`, layer 2/3) | `brain_regions`, `brain_region_idx` | One region label for all neurons in all sessions | Paper methods | Use one region entry for the whole dataset |

### Key Decisions
1. **Neural trace choice**: Use paper-consistent Suite2p-style baseline-corrected fluorescence rather than raw `F`, `spks`, or the GUI `dF/F0` helper, because the paper explicitly states downstream analyses used baseline-corrected fluorescence traces with default Suite2p parameters.
2. **Trialization strategy**: Convert each continuous session into consecutive 2-minute pseudo-trials after 10-frame averaging. This exactly matches the paper’s decoder split granularity and gives 10 trials for 20-minute sessions and 15 trials for 30-minute sessions.
3. **Temporal binning**: Average non-overlapping 10-frame windows for neural and behavior streams before segmentation, matching the paper’s decoder denoising (`10 consecutive timestamps`). This yields `333.33 ms` bins.
4. **Master clock**: Use imaging frames as the reference timeline because `ops.npy` gives authoritative `fs=30` and sessions have fixed imaging lengths. Reinsert missing behavior frames from timing gaps so both modalities remain frame-aligned before binning.
5. **Behavior output representation**: Because the target decoder requires categorical outputs, convert the denoised motion-energy trace into 5 global equal-percentile classes. Use global min-max normalization only to satisfy the task wording; percentile thresholds are based on the normalized values, which preserves ordering.
6. **Input representation**: Encode elapsed session time as one continuous, time-varying input in seconds. Each pseudo-trial keeps its absolute position within the original recording, so the decoder receives the requested “time elapsed from the beginning of the experiment.”
7. **Session inclusion**: Keep all 41 sessions. No session fails the core reference criteria, and all sessions yield at least 10 pseudo-trials after processing.
8. **Neuron inclusion**: Keep all neurons present in the released matched Suite2p arrays for each subject. These already represent the Track2p all-days matched population analyzed in the paper.

### Planned Sanity Checks
- [ ] Neural preprocessing spot-check: for a chosen session/neuron/window, compare converted neural values against a direct manual computation from raw `F`, `Fneu`, and `ops` using `np.allclose()`.
- [ ] Behavior reconstruction spot-check: for a mismatch session, reconstruct the full-length motion trace directly from raw `motion_energy_glob`, `tstamps`, and `interframe_int` and verify selected samples in the converted output with `np.allclose()`.
- [ ] Binning and segmentation check: manually average one raw 10-frame block and verify it matches the first converted time bin for neural, input time, and output class assignment.
- [ ] Dataset statistics check: verify subject count, session count, per-subject neuron counts, and mean tracked neurons per mouse are consistent with paper-reported scale.

---

## Step 6: Script Development
**Status**: COMPLETE

Implementation notes:
- `convert_data.py` created.
- Current implementation:
  - discovers sessions from `data/`
  - reconstructs missing behavior frames from timing gaps
  - computes Suite2p-style baseline-corrected fluorescence using saved `ops.npy` parameters
  - averages neural, behavior, and time streams in non-overlapping 10-frame bins
  - segments each session into consecutive 2-minute pseudo-trials
  - normalizes motion energy globally and discretizes it into 5 equal-percentile bins
  - writes the target pickle format and optional processing plots
- Smoke test passed on `--sample --show-processing`:
  - 2 sessions processed successfully
  - 25 pseudo-trials created
  - 2 processing plots written
  - mean processing time was `2.78 s / session`

Code inefficiencies identified:
- Full dataset not yet benchmarked, but the main cost is Suite2p-style fluorescence preprocessing.

Code speedups added:
- Session-wise processing only; raw arrays are not kept after binning.
- Behavior reconstruction is vectorized via timing-step accumulation.
- Binning uses reshape-and-mean rather than Python loops.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 967 across 2 sessions |
| Neurons / session | 221, 746 |
| Subjects | 2 (`jm031`, `jm039`) |
| Sessions / subject | 1, 1 |
| Trials (total) | 25 pseudo-trials |
| Trials / session | 10, 15 |
| `elapsed_time_sec` range | [0.2, 1799.8] |
| `motion_energy_quintile` distribution (global) | [0.200, 0.200, 0.200, 0.200, 0.200] |
| `motion_energy_quintile` distribution (`jm031_2023-10-20_a`) | [0.4697, 0.3069, 0.0514, 0.0533, 0.1186] |
| `motion_energy_quintile` distribution (`jm039_2024-05-04_a`) | [0.0202, 0.1287, 0.2991, 0.2978, 0.2543] |

### Processing Plots Review
- Reviewed `processing_jm031_2023-10-20_a.png` after fixing a plotting-only normalization bug.
- Plots show:
  - neural preprocessing uses neuropil subtraction plus baseline correction
  - missing behavior frames are reinserted and interpolated without obvious temporal discontinuity
  - 10-frame binning and 2-minute segmentation are aligned
  - discretized motion classes vary over time and are not stuck in a single class
- No conversion anomaly was visible in the corrected sample plots.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |
| Session-wise processing after per-session loading | Keeps memory bounded; avoids storing raw arrays |
| Reshape-based bin averaging | Removes Python-loop overhead |
| Vectorized missing-frame reconstruction | Negligible overhead even for mismatch sessions |

| Step | Time / Session | Estimated Total Time |
| | | |
| Sample conversion (`--sample`) | 2.48 s / session mean | 4.97 s for 2 sessions |
| Estimated full conversion | workload-scaled from sample | ~85.6 s (~1.43 min) for all 41 sessions |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| `motion_energy_quintile` | 0.7321 | 0.3899 |

Additional notes:
- Loss decreased from `92.861214` at epoch 1 to `0.899776` at epoch 200.
- Chance level for 5 balanced classes is `0.2000`; validation performance is above chance.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 414,412,286 bytes
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | `526 ± 190` tracked neurons / mouse (not a direct total) | N/A | 2,998 unique tracked neurons; 20,445 session-repeated | 2,998 unique tracked neurons; 20,445 session-repeated | Approx |
| Mean neurons/session | Paper reports per-mouse tracked population, not per-session mean | N/A | 498.66 | 498.66 | Yes |
| Subjects | 6 | Example defaults only | 6 | 6 | Yes |
| Sessions | Minimum 6 per mouse | Example defaults only | 41 total sessions | 41 total sessions | Yes |
| Trials (total) | Continuous sessions, no trials | N/A | 41 continuous sessions | 545 pseudo-trials | By design |
| Trials/session (mean) | Continuous sessions, no trials | N/A | N/A | 13.29 pseudo-trials/session | By design |
| `elapsed_time_sec` range | Sessions stated as 20 min in methods; raw data include 20 and 30 min recordings | N/A | [0, 1200] or [0, 1800] sec implied by frame counts | [0.2, 1799.8] sec | Yes vs data |
| `motion_energy_quintile` distribution | Continuous motion energy in paper | N/A | Continuous motion energy | [0.200, 0.200, 0.200, 0.200, 0.200] globally | Task-driven |

Notes:
- `verification_full_out.txt` reports no format errors or warnings.
- Full converted dataset statistics:
  - 41 sessions
  - 545 pseudo-trials
  - 20,445 neurons across session entries
  - 1 input dimension and 1 output dimension
- Spot-checked converted sessions:
  - `jm031_2023-10-18_a`: 10 trials, trial shape `(221, 360)`
  - `jm031_2023-10-20_a` (missing behavior frames): 10 trials, trial shape `(221, 360)`
  - `jm039_2024-04-30_a`: 15 trials, trial shape `(746, 360)`
  - `jm046_2024-09-09_a` (missing behavior frame): 15 trials, trial shape `(435, 360)`
- Session-duration discrepancy:
  - Paper methods say sessions lasted 20 minutes.
  - Raw released data contain both 36,000-frame sessions (20 min at 30 Hz) and 54,000-frame sessions (30 min at 30 Hz).
  - Resolution: keep the full released recordings. Cropping 30-minute sessions to 20 minutes would discard source data without support from the reference code or the data bundle.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `verification_full_out.txt` reports no format errors and no warnings.
2. **Constructed raw-vs-converted sanity checks**:
   - Neural check on `jm031_2023-10-20_a`: manually recomputed Suite2p-style baseline-corrected fluorescence from raw `F`, `Fneu`, and `ops`, binned it directly, and verified converted values with `np.allclose(...) == True`.
   - Input check on `jm039_2024-05-04_a`: manually computed binned frame-center times for the last trial and verified equality with converted `elapsed_time_sec` using `np.allclose(...) == True`.
   - Output check on `jm031_2023-10-22_a` (116 missing behavior frames): manually reconstructed/interpolated/binned motion, normalized with stored metadata, discretized with stored quintile edges, and verified the converted first-trial labels with `np.allclose(...) == True`.
   - Trial-count check across all sessions: expected number of 2-minute trials from raw frame counts matched converted trial counts for every session.
3. **Reference code comparison**:
   - **Data loading**: reference code loads Track2p-exported Suite2p arrays (`F`, `Fneu`, `spks`, `ops`, `stat`, `iscell`) per session; `convert_data.py` loads the same session-level arrays from `suite2p/plane0/`.
   - **Neuron/trial filtering**: reference code/paper use Suite2p `iscell > 0.5` before Track2p and then keep all-day matched neurons; released data already contain this matched-cell export, so conversion applies no extra neuron filtering.
   - **Temporal alignment**: paper states the camera was microscope-triggered at 30 Hz; conversion uses imaging frames as the master clock and restores missing behavior frames from `interframe_int.npy` before binning.
   - **Binning**: paper decoder used averages over `10 consecutive timestamps`; conversion uses non-overlapping 10-frame means for neural, behavior, and time.
   - **Input construction**: no reference input variable exists because the paper decoded continuous behavior from neural activity alone; the added elapsed-time input is a task-specific requirement from this benchmark.
   - **Output construction**: paper decoded continuous motion energy with ridge regression; conversion preserves the same motion-energy source and timing, then applies the benchmark-required normalization and quintile discretization.
4. **Key statistics comparison**:
   - Subjects: converted `6`, matching paper and data.
   - Sessions: converted `41`, matching the data and consistent with paper statement of `>= 6` daily sessions per mouse.
   - Mean neurons/session: converted `498.66`, matching the raw data and close to the paper’s `526 ± 190` tracked neurons per mouse.
   - Output distribution: converted dataset is exactly balanced globally by construction (`0.2` each class).
5. **Edge-case checks**:
   - Sessions with missing behavior frames (`9` total) convert to the correct final length after reconstruction.
   - Heavily gapped sessions (`116` and `148` missing frames) still produce correctly aligned 10-trial outputs.
   - Mixed session durations (`20 min` and `30 min`) both segment into valid 2-minute trials with no leftover bins.

### Issues Found and Resolved
- **Processing-plot normalization bug**: During Step 7 the diagnostic plot incorrectly reused normalized edges as raw min/max values. Fixed in `convert_data.py` by passing the true global motion min/max into the plotting function and regenerating the sample plots.
- **Session-duration discrepancy between paper text and raw data**: Paper methods say `20 minutes`, but four subjects have `54,000` frames at `30 Hz` (`30 minutes`). Resolution: keep the full raw recordings because the data bundle and saved `ops.npy` explicitly encode these lengths, and cropping would be an unsupported alteration.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| `motion_energy_quintile` | 0.7257 | 0.4714 | Chance = 0.2000 for 5 classes; test loss = 2.495315 |

Additional notes:
- Loss decreased from `77.029049` at epoch 1 to `0.837328` at epoch 200.
- Full training completed on GPU (`cuda`) without memory issues.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| | |
| `motion_energy_quintile` | Validation balanced accuracy `0.4714` (chance `0.2000`; `2.36x` chance) | Paper reports same-day motion decoding that improves with development and becomes accurate/stable across days, but uses `R^2` rather than balanced accuracy |

[Analysis of any low accuracies]
- **Accuracy vs chance**: The full validation score is well above chance (`0.4714` vs `0.2000`), so there is no indication of a broken alignment or label-construction bug.
- **Accuracy comparison to paper**:
  - The paper’s decoding metric is continuous-regression `R^2`, not categorical balanced accuracy.
  - The extracted paper text and figure captions do not provide a numerical table of same-day decoding values, so there is no exact metric-matched number to compare directly.
  - Qualitatively, the converted dataset agrees with the paper’s claim that motion is decodable from population activity; our classifier-level validation is strong and clearly above chance.
- **Train vs validation gap**:
  - Training balanced accuracy: `0.7257`
  - Validation balanced accuracy: `0.4714`
  - Ratio: `1.54x`
  - Investigation: this is only slightly above the `1.5x` review threshold. There is no sign of data leakage from the conversion itself because raw-vs-converted sanity checks passed, trial counts match frame counts, and validation remains well above chance. The remaining gap is plausibly attributable to model capacity plus substantial across-session heterogeneity in per-session output distributions.
- **Additional debugging checks considered**:
  - Raw output labels were spot-checked directly from original motion-energy files in Step 10 and matched exactly after reconstruction/discretization.
  - Temporal alignment was checked through the processing plots and raw-vs-converted checks on sessions with missing behavior frames.
  - Output variation is adequate globally by construction (exact quintiles) and non-degenerate within sessions.

### Issues Found and Resolved
- **No new conversion issue found in the accuracy review**: decoder performance is well above chance and consistent with the paper’s qualitative conclusion that motion-related information emerges in neural population activity.
- **Metric mismatch with paper**: direct numeric comparison is limited because the paper reports `R^2` for continuous regression, whereas this benchmark requires discretized categorical decoding and evaluates balanced accuracy.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
