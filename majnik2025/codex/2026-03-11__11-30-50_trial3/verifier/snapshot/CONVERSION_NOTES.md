# Dataset Conversion Notes

## Overview
- **Dataset**: [Name and source]
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

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `run_t2p` | `code/track2p/t2p.py` | LOADING | Main Track2p pipeline: checks planes, loads Suite2p mean images, runs registration/matching, saves match matrices. |
| `generate_suite2p_indices` | `code/track2p/t2p.py` | PROCESSING | Converts Track2p match indices back to original Suite2p indices after `iscell` filtering. |
| `save_in_s2p_format` | `code/track2p/t2p.py` | PROCESSING | Exports matched-across-all-days neurons into `matched_suite2p`, including `F.npy`, `Fneu.npy`, `spks.npy`, `iscell.npy`, `stat.npy`, `ops.npy`. |
| `check_nplanes` | `code/track2p/io/s2p_loaders.py` | LOADING | Verifies all datasets have the same number of imaging planes. |
| `load_all_imgs` | `code/track2p/io/s2p_loaders.py` | LOADING | Loads `ops.npy` mean images and channel metadata for every dataset/plane. |
| `load_all_ds_stat_iscell` | `code/track2p/io/s2p_loaders.py` | CURATION | Applies Suite2p `iscell` filtering to ROI stats using `track_ops.iscell_thr`. |
| `load_stat_ds_plane` | `code/track2p/io/loaders.py` | CURATION | Loads `stat.npy` and filters ROIs by `iscell` threshold; reports counts before/after filtering. |
| `DataManagement.import_files` | `code/track2p/gui/data_management.py` | LOADING | Canonical downstream loading path for matched data: load `plane*_match_mat.npy`, remove rows containing `None`, load per-day Suite2p traces, apply `iscell` filter, then index by matched rows. |
| `DataManagement.F_processing` | `code/track2p/gui/data_management.py` | PROCESSING | Computes the package’s “dF/F0” display trace by neuropil subtraction with `neucoeff=0.0` and baseline subtraction (`maximin` by default); notably returns baseline-subtracted fluorescence rather than a ratio. |

### Notes
- The codebase is primarily the Track2p package plus demo notebooks. There is no task decoder pipeline in `code/`; the relevant reusable logic is the loading/curation of matched Suite2p traces.
- Canonical neuron selection appears in both `save_in_s2p_format` and `DataManagement.import_files`:
  - First filter per-session ROIs by Suite2p `iscell`.
  - If `track_ops.iscell_thr is None`, keep `iscell[:, 0] == 1`.
  - Otherwise keep `iscell[:, 1] > track_ops.iscell_thr`.
  - Default threshold in `DefaultTrackOps` is `0.50`.
- Canonical longitudinal curation is to use only neurons tracked across all days:
  - `t2p_match_mat_allday = t2p_match_mat[~np.any(t2p_match_mat == None, axis=1), :]`
  - This discards any tracked cell row containing a missing day.
- Canonical matched trace extraction:
  - Load per-day `F.npy` or `spks.npy` (and optionally `Fneu.npy`).
  - Apply `iscell` filtering first.
  - Then index filtered arrays with `t2p_match_mat_allday[:, i].astype(int)` for day `i`.
- The GUI/demo notebooks explicitly describe this as the intended downstream use case: matched cells present on all days, with consistent indexing across days.
- The code supports three trace types for downstream visualization:
  - `F`
  - `spks`
  - `dF/F0`
- The package’s `dF/F0` option is not a true `ΔF/F`; `F_processing` performs `Fc = F - neucoeff * Fneu` with default `neucoeff=0.0`, estimates a baseline `Flow`, and returns `Fc - Flow`.
- No behavioral or motion-energy loading code is present in `code/`; those mappings will need to come from `data/`, `methods.txt`, and `paper.pdf`, while preserving the matched-cell loading rules above.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `data/README.md` confirms this is the Track2p paper dataset and describes the organization.
- There are 6 subject folders: `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`.
- Each subject contains daily session folders named as dates (e.g. `2023-04-30_a`).
- Each session contains:
  - `suite2p/plane0/F.npy`: matched-cell fluorescence traces, shape `(n_neurons, n_frames)`
  - `suite2p/plane0/Fneu.npy`: neuropil traces
  - `suite2p/plane0/spks.npy`: Suite2p deconvolved spike-like traces
  - `suite2p/plane0/iscell.npy`: 2-column Suite2p cell classification/probability array
  - `suite2p/plane0/stat.npy`: ROI metadata
  - `suite2p/plane0/ops.npy`: Suite2p ops dictionary, including `fs=30`, `nframes`, `nchannels`
  - `move_deve/motion_energy_glob.npy`: processed global motion energy from videography
  - `move_deve/tstamps.npy`: timestamps for available behavior frames
  - `move_deve/interframe_int.npy`: inter-frame intervals for behavior timestamps
- `data/README.md` states the `suite2p` folder already contains only neurons “present across all days”, saved using Track2p “save the outputs in suite2p format”. This means the neural matrices are already longitudinally matched within each subject.
- `data/load_data.ipynb` provides helper functions for loading `F.npy`, `ops['meanImg']`, and ROI coordinates. It explicitly warns that for proper analysis one should compute `dF/F` as described in the paper or use `spks.npy`.
- Additional files:
  - `data/jm038/ground_truth.csv`
  - `data/jm039/ground_truth.csv`
  - `data/jm046/ground_truth.csv`
  These appear to be manual cell-matching ground truth tables for Track2p evaluation, not motion-decoder variables.
- Native data are continuous sessions rather than trial-structured experiments. No trial table or event list is present in `data/`.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 20,445 summed across session matrices |
| Neurons / session | 221 to 746 (subject-constant within each mouse) |
| Subjects | 6 |
| Sessions / subject | `jm031`: 7, `jm032`: 7, `jm038`: 7, `jm039`: 7, `jm040`: 6, `jm046`: 7 |
| Trials (total) | No native trials; data are continuous sessions |
| Trials / session | No native trials; to be constructed during conversion |

- Subject-level neuron counts from matched `F.npy`:
  - `jm031`: 221 neurons across 7 sessions
  - `jm032`: 370 neurons across 7 sessions
  - `jm038`: 685 neurons across 7 sessions
  - `jm039`: 746 neurons across 7 sessions
  - `jm040`: 541 neurons across 6 sessions
  - `jm046`: 435 neurons across 7 sessions
- Session counts:
  - Total sessions: 41
- Frame counts:
  - `jm031`, `jm032`: 36,000 imaging frames per session
  - `jm038`, `jm039`, `jm040`, `jm046`: 54,000 imaging frames per session
  - Imaging sampling rate from `ops.npy`: 30 Hz in all sessions
- Behavior stream details:
  - `motion_energy_glob.npy` is length-matched to imaging in most sessions
  - 9 sessions have fewer behavior samples than imaging frames, consistent with `data/README.md` note about missing video frames
  - Mismatch cases found:
    - `jm031/2023-10-20_a`: missing 2 behavior frames
    - `jm031/2023-10-21_a`: missing 3 behavior frames
    - `jm031/2023-10-22_a`: missing 116 behavior frames
    - `jm032/2023-10-20_a`: missing 2 behavior frames
    - `jm032/2023-10-21_a`: missing 2 behavior frames
    - `jm032/2023-10-22_a`: missing 148 behavior frames
    - `jm039/2024-05-04_a`: missing 1 behavior frame
    - `jm040/2024-05-04_a`: missing 1 behavior frame
    - `jm046/2024-09-09_a`: missing 1 behavior frame
- Example dtypes/ranges from `jm038/2023-04-30_a`:
  - `F.npy`: `float32`, shape `(685, 54000)`
  - `Fneu.npy`: `float32`, shape `(685, 54000)`
  - `spks.npy`: `float32`, shape `(685, 54000)`
  - `motion_energy_glob.npy`: `uint64`, shape `(54000,)`
  - `tstamps.npy`: `float64`, shape `(54000,)`
  - `interframe_int.npy`: `float64`, shape `(53999,)`

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | ~3156 inferred as `526 x 6` mice tracked across all days | “On average 526 (± 190 std) neurons per mouse were successfully tracked across all days” |
| Neurons / session | Not stated directly; same tracked cells reused across days within mouse | “cells present across all days” is the intended Track2p downstream use case |
| Subjects | 6 mice | “we used a full dataset of 6 mice” |
| Sessions / subject | Minimum 6 consecutive days, spanning P7 to P14 | “imaged daily for a minimum of 6 consecutive days within the second postnatal week (P7 to P14)” |
| Trials (total) | No trial structure described; recordings are continuous sessions | Decoding described on “consecutive 2 minute blocks of the recording” |
| Trials / session | Not native; recordings later split into blocks/windows for analysis | “splits were done on consecutive 2 minute blocks of the recording” |
| Neural data time bin | 30 Hz raw imaging frames; 10-frame denoised bins for decoding | “Imaging rate was 30 Hz”; “averaging in bins of 10 consecutive timestamps” |
| Behavior data time bin | 30 Hz camera frames; 10-frame denoised bins for decoding | “Videos were recorded at 30 Hz”; “averaging in bins of 10 consecutive timestamps” |
| Reward rate | N/A | Spontaneous behavior dataset, no rewards described |
| Motion variable | Global motion energy from pixelwise frame differences | “computed their pixelwise difference… squared… summed across pixels” |
| Example mouse tracked cells | 728 tracked neurons in Fig. 5 caption | “Raster plots showing the activity of all 728 tracked neurons” |
| Same-day decoder metric | `R^2` vs mouse motion, increases with development | “same-day decoding performance increased with development” |
| Cross-day decoder metric | `R^2`, stable late-late decoding, weaker early-late | “once developed, this representation was indeed stable” |


### Processing Details
- Experimental context:
  - Longitudinal 2-photon calcium imaging in mouse barrel cortex during the second postnatal week.
  - Mice are freely moving on a non-motorised treadmill under sensory-minimised conditions.
  - Motion energy is used as a proxy for arousal/behavioral state.
- Temporal alignment:
  - Imaging and videography were both recorded at 30 Hz.
  - The microscope acquisition triggered the camera, enabling simple synchronization between neural and behavior streams.
  - Published decoding analyses operate on slightly denoised traces produced by averaging in bins of 10 consecutive timestamps.
  - Published cross-validation splits use consecutive 2-minute blocks.
- Neural processing:
  - Each recording was preprocessed with Suite2p separately.
  - Suite2p stages used: motion correction, ROI detection, signal extraction, spike deconvolution.
  - For paper analyses they used “baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)”.
  - `spks.npy` is also available in the data package, but the paper’s main longitudinal analyses reference the baseline-corrected fluorescence traces.
- Behavioral processing:
  - Motion energy is constructed from consecutive video frame differences.
  - For each pair of consecutive frames, compute pixelwise difference, square values, and sum across pixels to obtain a scalar motion measure per time point.
- Decoder processing in paper:
  - Model family: ridge regression.
  - Same-day decoding uses nested cross-validation.
  - Both inner and outer CV use 5 folds.
  - Fold boundaries are based on consecutive 2-minute recording blocks.
  - Cross-day decoding fits on one day and tests on another.
- Important note for this conversion task:
  - The target decoder here differs from the paper’s regression target by requiring categorical outputs and a specified input/output format.
  - The closest paper-consistent adaptation is therefore to preserve the paper’s time alignment and 10-frame averaging, then discretize motion energy into five equal-percentile bins as instructed.

### Curation Steps

**Neuron curation rules**:
- Use ROIs classified as cells by Suite2p with probability above the default threshold 0.5.
- For Track2p downstream analysis, retain neurons tracked across all days within a subject.
- Paper interpretation suggests longitudinal drop in tracked cells is attributed mainly to detection failure rather than tracking failure.

**Trial curation rules**:
- No native trials; recordings are continuous sessions.
- For decoding, the paper uses contiguous temporal blocks rather than event-aligned trials.
- `data/README.md` states some sessions have missing video frames; these should be handled via missing values or interpolation using `tstamps.npy` / `interframe_int.npy`.

### Decoders Trained
| Decoded variable | Accuracy |
| Mouse motion / behavioral state | Reported as `R^2`; same-day performance increases with age, cross-day decoding becomes stable late in development; exact numeric values not stated in extractable text |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Longitudinal cell set | `t2p_match_mat_allday = t2p_match_mat[~np.any(... == None)]` and downstream analyses use only cells present on all days | `data/README.md` says `suite2p/` already contains only successfully tracked neurons present across all days | Paper analyses use the tracked population across all days | Consistent. Use all neurons in provided `suite2p/plane0` directly; no extra Track2p matching step is needed during conversion. |
| Cell filtering threshold | Track2p loaders use `iscell[:,1] > track_ops.iscell_thr`, default `0.50` | Provided `iscell.npy` values are all above `0.5` in sampled sessions, consistent with pre-filtered matched output | “We considered all ROIs above the default threshold of 0.5 as true cells.” | Consistent. No additional ROI filtering beyond trusting provided matched Suite2p export. |
| Neural trace used for paper analyses | Track2p demo loads `F.npy`, `spks.npy`, or package-specific “dF/F0” display trace; notebook warns to compute dF/F properly or use `spks.npy` | Data package contains `F.npy`, `Fneu.npy`, `spks.npy`, but no precomputed dF/F array | “We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters) for all subsequent analyses.” | Need adaptation. For paper-consistent decoding, reconstruct Suite2p-style baseline-corrected fluorescence from `F` and `Fneu` using default parameters if possible. `spks.npy` remains a fallback but is less faithful to the stated analysis choice. |
| Behavioral stream | No behavioral loader in Track2p package; only neural matching/export | `move_deve/motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy` present for every session | Motion energy computed from squared pixelwise differences of consecutive video frames | Consistent. Build decoder output from `motion_energy_glob.npy`; use timestamps to handle missing camera frames. |
| Temporal synchronization | Paper says microscope triggered camera; code does not handle behavior | Most sessions have equal neural/behavior frame counts; 9 sessions have slight behavior shortfall | “microscope acquisition acting as a trigger… simple synchronisation across the two modalities” | Use imaging frames as the master clock and align behavior by timestamp/interpolation onto imaging frame times where frames are missing. |
| Session duration | Track2p code is agnostic to duration | Sessions are either 36,000 frames (20 min) or 54,000 frames (30 min) at 30 Hz | Methods say “each session lasted 20 minutes” | Trust released data durations. Document that two mice (`jm031`, `jm032`) are 20 min and four mice are 30 min; conversion must preserve actual session lengths. |
| Mean tracked neurons per mouse | Not stated in code | Mean from released data is `499.7`, std `180.5` across 6 mice | Paper reports `526 ± 190` tracked neurons/mouse | Likely release/version discrepancy. Keep all six provided mice and document the mismatch explicitly; do not attempt to force counts to match by ad hoc filtering. |
| Example mouse neuron count | N/A in code | No subject has exactly 728 matched cells (`221, 370, 685, 746, 541, 435`) | Fig. 5 caption says example mouse had `728` tracked neurons | Treat as paper/release mismatch or OCR/version difference. Not actionable for conversion logic. |
| Decoder target/format | Paper decodes continuous motion via ridge regression with `R^2` | Data provide continuous motion energy | Target task here requires categorical outputs and `input = time elapsed` | Preserve paper’s temporal processing and alignment, then discretize motion energy into five equal-percentile bins to satisfy target format. |
| Trial structure | Paper decoding uses consecutive temporal blocks | Data are continuous sessions with no explicit trial table | Same-day decoding uses “consecutive 2 minute blocks”; denoising uses 10-frame bins | Construct pseudo-trials as contiguous time windows derived from the continuous recordings. Prefer a block structure compatible with the paper’s 2-minute segmentation. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `suite2p/plane0/F.npy`, `Fneu.npy`, `ops.npy` | `neural` | Reconstruct Suite2p-style baseline-corrected fluorescence: `Fc = F - neucoeff*Fneu`, then `suite2p.extraction.dcnv.preprocess(...)`; then average over 10-frame bins; then split into consecutive 2-minute blocks; final trial arrays shape `(n_neurons, 360)` | Paper methods on Suite2p baseline correction; data notebook warning; not from Track2p GUI `F_processing` | Uses paper-stated fluorescence representation rather than raw `F.npy` or Track2p GUI display mode. |
| Imaging frame clock derived from `ops['fs']` | `input[0]` | Create time elapsed from session start in seconds for every 10-frame bin; split into 2-minute blocks so each trial gets a `(1, 360)` time series of absolute within-session elapsed time | Paper says recordings at 30 Hz; target task specifies time-elapsed input | Input is absolute elapsed time from session start, not relative block time. |
| `move_deve/motion_energy_glob.npy`, `tstamps.npy` | intermediate behavior | Map behavior samples onto imaging frame indices using normalized timestamps; fill missing frames by linear interpolation on the imaging grid; average over 10-frame bins | `data/README.md` guidance for missing frames; paper says 30 Hz synchronized camera and 10-timestamp averaging | Imaging frames are treated as master clock. |
| Binned/interpolated motion energy | `output[0]` | Normalize globally to `[0,1]` across included sessions after 10-frame averaging; discretize into 5 equal-percentile bins using global quintiles; split into 2-minute blocks so each trial is `(1, 360)` categorical integers `0..4` | Paper provides continuous motion energy; target task requires normalized 5-bin categorical output | Global thresholds preserve across-session comparability. |
| Subject folder name (e.g. `jm038`) | `subjects`, `subject_idx` | Sorted unique subject IDs; each session points to its subject index | Data directory organization | Sessions remain separate; no merging across mice. |
| Constant brain area from paper | `brain_regions`, `brain_region_idx` | Single region label, all neurons assigned region index `0` | Paper experimental description | Use `barrel cortex L2/3`. |
| Session folder name/date | `metadata['session_info']` | Store session IDs, dates, durations, frame counts, missing-frame counts | Data directory organization | Helpful for audit and later checks. |

### Key Decisions
1. **Use baseline-corrected fluorescence as neural data**: The paper explicitly states that subsequent analyses used baseline-corrected fluorescence traces (“our dF/F”) with default Suite2p parameters. The Track2p GUI’s `F_processing` is not faithful because it defaults to `neucoeff=0.0`. I will therefore reconstruct the fluorescence using Suite2p’s own preprocessing logic and the per-session `ops.npy` parameters.
2. **Use 10-frame temporal averaging before segmentation**: This exactly matches the paper’s decoding preprocessing and converts 30 Hz traces into 3 Hz traces while denoising both neural and behavior data.
3. **Construct pseudo-trials as consecutive 2-minute blocks**: The native data have no trials, and the paper’s decoder splits recordings into consecutive 2-minute blocks. After 10-frame averaging, each block contains `120 s * 3 Hz = 360` time bins. This satisfies the target format while staying aligned to the published analysis.
4. **Treat imaging frames as the reference clock**: Because the behavior stream occasionally has missing frames, motion energy will be aligned to the imaging grid using `tstamps.npy`, then interpolated over missing positions, matching the data README guidance.
5. **Use absolute elapsed time from session start as the sole decoder input**: This is mandated by the task and remains paper-consistent because the recordings are continuous spontaneous sessions rather than event-locked trials.
6. **Discretize motion energy only at the final step**: All alignment, interpolation, and 10-frame averaging will operate on continuous motion energy first. Discretization into five equal-percentile bins happens after paper-consistent preprocessing to minimize distortion.
7. **Keep all six mice and all provided sessions**: The release differs somewhat from paper summary counts, but there is no principled basis for dropping mice or sessions. Ad hoc filtering would be less defensible than preserving the released dataset and documenting discrepancies.
8. **Use one brain-region label for all neurons**: All recordings are described as layer 2/3 barrel cortex from a single FOV per mouse.

### Planned Sanity Checks
- [ ] Neural sanity check: for a spot-checked session, verify that the converted binned neural trace for one neuron/block equals the manually reconstructed Suite2p-preprocessed trace averaged in exact 10-frame bins (`np.allclose`).
- [ ] Input sanity check: for a spot-checked trial, verify that `input[0]` equals the expected elapsed-time vector derived from frame indices and `fs=30` after 10-frame binning (`np.allclose`).
- [ ] Output sanity check: for sessions with missing camera frames, verify that interpolated motion values at selected frame indices match values obtained by direct raw-data reconstruction from `motion_energy_glob.npy` and `tstamps.npy` (`np.allclose`).
- [ ] Structural sanity check: each 20-minute session should yield exactly 10 trials and each 30-minute session exactly 15 trials, all with `T=360`.
- [ ] Distribution sanity check: motion bins should be close to 20% each globally after quintile discretization.

---

## Step 6: Script Development
**Status**: COMPLETE

[Implementation notes]
- Created `convert_data.py` with CLI:
  - `python -u convert_data.py <outpicklefile>`
  - `--full`
  - `--sample`
  - `--show-processing`
- Conversion implemented as a two-pass pipeline:
  1. Scan behavior streams for all included sessions, align motion to imaging time, average in 10-frame bins, and compute global motion normalization / quintile thresholds.
  2. Process neural data session-by-session, reconstruct Suite2p-style baseline-corrected fluorescence from `F.npy` and `Fneu.npy`, average in 10-frame bins, split into consecutive 2-minute block trials, then save in target format.
- Added built-in validation via `verify_data_format` before writing the pickle.
- Added per-session timing prints and total runtime print.
- Added `processing_<session_id>.png` generation for up to 2 sessions in `--show-processing` mode.

Code inefficiencies identified:
- Suite2p baseline correction requires loading full `F.npy` and `Fneu.npy` arrays for each session.
- Full conversion may still be moderately expensive because baseline correction is performed per session on CPU.

Code speedups added:
- Two-pass design avoids loading all neural sessions simultaneously.
- Motion quantile computation stores only 10-frame-averaged behavior values, which are small.
- Neural data are processed one session at a time and written to in-memory output lists only after temporal binning and block splitting.
- Used `mmap_mode='r'` when only shape inspection was needed.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 906 summed across 2 sessions |
| Neurons / session | `[221, 685]` |
| Subjects | 2 (`jm031`, `jm038`) |
| Sessions / subject | `jm031`: 1, `jm038`: 1 |
| Trials (total) | 25 |
| Trials / session | `[10, 15]` |
| Elapsed time input range (s) | [0.0, 1799.67] |
| Time bins / trial | 360 |
| Motion-energy-bin distribution | [0.200, 0.200, 0.200, 0.200, 0.200] |

### Processing Plots Review
- Reviewed `processing_jm031_2023-10-18_a.png` and `processing_jm038_2023-04-30_a.png`.
- Neural traces show plausible baseline-corrected activity after Suite2p preprocessing and smoother binned rasters after 10-frame averaging.
- Motion alignment plots show no obvious discontinuities or offset between raw and aligned traces in the sampled sessions.
- Discretized motion bins track low- and high-motion epochs rather than flickering randomly, which argues against a temporal shift bug.
- No anomalies found in the sampled sessions.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |
| Two-pass streaming conversion | Avoids holding the full neural dataset in memory |
| Session-by-session neural preprocessing | Keeps peak memory bounded by one session |
| Behavior-only first pass for quantiles | Makes global binning cheap |

| Step | Time / Session | Estimated Total Time |
| | | |
| 20-minute session conversion | ~1.5 s | ~21 s for 14 sessions |
| 30-minute session conversion | ~3.6 s | ~97 s for 27 sessions |
| Full conversion total | N/A | ~120 s (~2.0 min) plus small I/O overhead |

- Files created:
  - `sample_data.pkl`
  - `conversion_sample_out.txt`
  - `verification_sample_out.txt`
- Verification result:
  - `train_decoder.py --verify-only` reported “Data format is valid, no errors or warnings.”
- No efficiency intervention required before full conversion because projected runtime is far below 15 minutes.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| `motion_energy_bin` | 0.6265 | 0.3552 |

- Chance level for the 5-bin output is `0.2000`.
- Training loss decreased from `40.21` at epoch 1 to `1.03` at epoch 200.
- Validation balanced accuracy is above chance, so the sample conversion is usable for proceeding to the full run.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 396M
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | ~3156 inferred tracked cells across 6 mice | N/A | 20445 summed across 41 session matrices | 20445 summed across 41 session matrices | Paper/release mismatch; converted matches data |
| Mean neurons/session | Not stated | N/A | 498.66 | 498.66 | Yes vs data |
| Subjects | 6 | N/A | 6 | 6 | Yes |
| Sessions | Minimum 6 consecutive days per mouse | N/A | 41 | 41 | Yes vs data / paper description |
| Trials (total) | No native trials; decoding uses 2-minute blocks | N/A | No native trials | 545 pseudo-trials | Intended derived quantity |
| Trials/session (mean) | N/A | N/A | N/A | 13.29 | Intended derived quantity |
| Elapsed time input range (s) | 20 min sessions stated in methods | N/A | Raw sessions span 1200 s or 1800 s | [0.0, 1799.67] | Converted matches release data |
| Motion bin distribution | Continuous motion in paper | N/A | Continuous motion | [0.2, 0.2, 0.2, 0.2, 0.2] | Yes for task-derived discretization |

- Full verification results:
  - `train_decoder.py --verify-only` reported “Data format is valid, no errors or warnings.”
  - Total trials: 545
  - Trials/session: 10 for each 20-minute session, 15 for each 30-minute session
  - All trials have `T=360`
- Spot checks:
  - `jm031_2023-10-18_a`: neural/input/output trial shapes `(221, 360)`, `(1, 360)`, `(1, 360)`
  - `jm038_2023-04-30_a`: `(685, 360)`, `(1, 360)`, `(1, 360)`
  - `jm046_2024-09-09_a`: `(435, 360)`, `(1, 360)`, `(1, 360)`
- Runtime:
  - `conversion_full_out.txt` shows total conversion time `60.97 s`
- Known unresolved paper-release mismatches to carry into Step 10:
  - Paper methods say 20-minute sessions, but the released data contain both 20- and 30-minute sessions.
  - Paper text reports `526 ± 190` tracked neurons per mouse; released data average `499.7 ± 180.5`.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `verification_full_out.txt` reported “Data format is valid, no errors or warnings.” No fixes required.
2. **Neural sanity check from raw data**: Recomputed Suite2p baseline-corrected fluorescence for `jm038/2023-04-30_a`, averaged in 10-frame bins, and compared one neuron’s full first 2-minute block against `converted_data.pkl`. `np.allclose == True`, max absolute difference `3.05e-05`.
3. **Input sanity check from raw timing**: Reconstructed the elapsed-time vector directly from frame indices at 30 Hz for `jm038/2023-04-30_a`, trial 2. Compared with converted `input`. `np.allclose == True`, max absolute difference `0.0`.
4. **Output sanity check from raw behavior with missing frames**: Reconstructed motion alignment for `jm031/2023-10-22_a` directly from `motion_energy_glob.npy` and `tstamps.npy`, applied 10-frame averaging, global normalization, and quintile discretization, then compared a converted trial. `np.allclose == True`.
5. **Reference code comparison**:
   - **(a) Data loading**: Reference Track2p downstream usage loads matched-across-all-days Suite2p exports. The provided release already stores that export under each session’s `suite2p/plane0`. The conversion reads these directly.
   - **(b) Neuron filtering**: Reference code uses `iscell > 0.5` and removes rows with `None` in the Track2p match matrix. The release data already reflect this filtering, so the conversion does not apply a second filtering pass.
   - **(c) Temporal alignment**: Track2p code has no behavior loader; paper/README state the camera is synchronized to imaging and that missing frames should be handled using timestamps. The conversion aligns behavior to imaging using `tstamps.npy` and interpolates only missing camera frames.
   - **(d) Binning**: Paper decoding averages both neural and behavior traces in bins of 10 timestamps and uses 2-minute blocks. Conversion does the same.
   - **(e) Input construction**: Not present in the paper; conversion uses elapsed time from session start because this is required by the task.
   - **(f) Output construction**: Paper decodes continuous motion; conversion preserves the same pre-discretization processing, then bins motion into quintiles because the task requires categorical outputs.
6. **Key statistics comparison**:
   - Subjects: converted data matches release and paper (`6`).
   - Sessions: converted data matches release (`41`) and paper description (minimum 6 days per mouse).
   - Neuron counts/session: converted data matches release exactly.
   - Trial counts: derived quantity; converted data behaves as intended (`10` trials for 20-minute sessions, `15` for 30-minute sessions).
   - Output distribution: exactly balanced globally by construction (`0.2` each bin).
7. **Edge-case check**:
   - Sessions with missing camera frames were converted without NaNs or length mismatches.
   - Both 20-minute and 30-minute sessions convert to exact multiples of 2-minute trials.
   - No off-by-one errors found at session ends; all trials have exactly `T=360`.

### Issues Found and Resolved
- **Paper vs release neuron-count mismatch (`526 ± 190` vs `499.7 ± 180.5`)**: Investigated and left unchanged. The converted data match the released files exactly; forcing agreement would require unjustified filtering.
- **Paper methods say 20-minute sessions, release contains 20- and 30-minute sessions**: Investigated and left unchanged. Conversion preserves actual durations from the release and remains internally consistent.
- **Track2p GUI `F_processing` differs from paper-stated fluorescence preprocessing**: Resolved during script development by switching to Suite2p’s `dcnv.preprocess` with session-specific `ops.npy` parameters instead of reusing the GUI helper.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| `motion_energy_bin` | 0.7172 | 0.4611 | Above 5-way chance (`0.2000`); loss decreased from `91.78` to `0.87` |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| `motion_energy_bin` | Validation balanced accuracy `0.4611` (chance `0.2000`, 2.31x chance) | Paper reports successful same-day and stable late cross-day decoding of continuous mouse motion with `R^2`, but no directly comparable categorical accuracy value is given in extractable text |

[Analysis of any low accuracies]
- Accuracy is not low relative to chance, so no conversion bug is suggested by the decoder result.
- The paper’s decoder metric is continuous-regression `R^2`, whereas this task requires 5-class balanced accuracy after discretization. Because the target and metric differ, only qualitative comparison is valid:
  - Paper expectation: motion should be decodable, especially at later developmental stages.
  - Observed result: full-dataset validation accuracy is well above chance, consistent with the expected presence of motion-related information in the neural activity.
- Train/validation gap:
  - Training balanced accuracy: `0.7172`
  - Validation balanced accuracy: `0.4611`
  - Ratio: `1.56x`
  - Investigation: this is slightly above the requested `1.5x` threshold, but there is no sign of leakage (trials are split into held-out blocks, validation accuracy is far from perfect, and the conversion sanity checks against raw data all passed). Most likely this reflects ordinary model overfitting rather than a conversion bug.

### Issues Found and Resolved
- **Direct numerical comparison to paper accuracy is not possible**: The paper reports `R^2` for continuous ridge regression on motion, while this task requires balanced accuracy for quintile classification. Resolved by documenting the metric mismatch and using qualitative agreement instead of a false numeric comparison.
- **Mild train/validation gap**: Investigated; no evidence of data leakage or temporal misalignment was found, and raw-data sanity checks support the conversion. No conversion change made.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
