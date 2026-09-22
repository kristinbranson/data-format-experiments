# Dataset Conversion Notes

## Overview
- **Dataset**: Track2p longitudinal barrel cortex motion-energy dataset from `/app/data`, based on the paper "longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p"
- **Date started**: 2026-09-21
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

Environment checks:
- `python3` available
- `numpy 2.4.4` imports successfully
- `torch 2.6.0+cu124` imports successfully

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `run_t2p(track_ops)` | `code/track2p/t2p.py` | PROCESSING | Main Track2p pipeline: load suite2p datasets, register days, match ROIs across days, save match matrices and optional suite2p-formatted matched outputs. |
| `generate_suite2p_indices(track_ops)` | `code/track2p/t2p.py` | PROCESSING | Converts match-matrix indices from iscell-filtered coordinates back to original suite2p indices for each day. |
| `save_in_s2p_format(track_ops)` | `code/track2p/t2p.py` | PROCESSING | Exports longitudinally matched cells into `matched_suite2p`, saving filtered/matched `F`, `Fneu`, `spks`, `stat`, `iscell`, and optional channel-2 arrays. |
| `check_nplanes(track_ops)` | `code/track2p/io/s2p_loaders.py` | LOADING | Verifies every dataset has the same number of suite2p planes and records it in `track_ops`. |
| `load_all_imgs(track_ops)` | `code/track2p/io/s2p_loaders.py` | LOADING | Loads `ops.npy` mean images and channel counts from each session/plane. |
| `load_all_ds_stat_iscell(track_ops)` | `code/track2p/io/s2p_loaders.py` | CURATION | Loads `stat.npy` and applies suite2p `iscell` filtering using `track_ops.iscell_thr`. |
| `load_all_ds_ops(track_ops)` | `code/track2p/io/s2p_loaders.py` | LOADING | Loads per-session/plane `ops.npy` dictionaries. |
| `load_all_ds_mean_img(track_ops, ch=1)` | `code/track2p/io/s2p_loaders.py` | LOADING | Extracts `meanImg` or `meanImg_chan2` from `ops.npy` for visualization. |
| `load_all_ds_centroids(all_ds_stat_iscell, track_ops)` | `code/track2p/io/s2p_loaders.py` | PROCESSING | Gets ROI centroids (`stat['med']`) after `iscell` filtering. |
| `load_track_ops(track_ops_path)` | `code/track2p/io/loaders.py` | LOADING | Loads saved Track2p configuration from `track_ops_postreg.npy`-style file. |
| `load_stat_ds_plane(track_ops_path, track_ops, plane_idx=0)` | `code/track2p/io/loaders.py` | CURATION | Loads `stat.npy` for one plane and filters ROIs by `iscell` threshold. |
| `npy_to_s2p(track_ops)` | `code/track2p/io/savers.py` | PROCESSING | Converts simplified NumPy input (`F.npy`, `fov.npy`, `rois.npy`) into suite2p-style folder structure. |
| `DefaultTrackOps.__init__` | `code/track2p/ops/default.py` | CURATION | Defines default tracking parameters; critically sets `iscell_thr = 0.50`. |

### Notes
- The repository does not implement behavioral decoding. It implements longitudinal cell registration/tracking and exports matched neural traces in suite2p-compatible format.
- The key curation rule in the reference code is ROI filtering by suite2p `iscell`. If `track_ops.iscell_thr` is `None`, keep rows with `iscell[:, 0] == 1`; otherwise keep rows with `iscell[:, 1] > iscell_thr`. The default threshold is `0.50`.
- The code preserves suite2p trace arrays directly (`F.npy`, `Fneu.npy`, `spks.npy`) after `iscell` filtering and cell matching. It does not compute `dF/F` anywhere in the repository sections inspected for loading/export, so conversion should not invent a `dF/F` transform unless raw data forces it.
- `save_in_s2p_format()` first removes cells not matched across all days in a match matrix row (`t2p_match_mat_allday = t2p_match_mat[~np.any(t2p_match_mat == None, axis=1), :]`) and then indexes filtered `F/Fneu/spks/stat/iscell` arrays with those matched-cell indices. This is the core longitudinal-neuron selection logic.
- Mean images and ROI shapes are used for registration/visualization only. For downstream decoding, the relevant reference outputs are the matched neural traces and the suite2p metadata (`ops`, `stat`, `iscell`, match matrices, suite2p indices).
- The code assumes imaging data in suite2p organization, supports multiple planes, and can save one matched suite2p folder per session/day under `track2p/matched_suite2p/`.
- No code related to motion-energy extraction, trialization, temporal binning for decoder inputs/outputs, or session splitting into 60 s trials appears in the reference repository. Those parts will need to come from the raw data organization and paper/methods, while respecting the above suite2p/Track2p cell filtering and matching decisions.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data/README.md` describes the release and folder organization.
- There are 6 subject folders: `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`.
- Each subject contains daily session folders named by recording date plus `_a` suffix.
- Each session contains:
  - `suite2p/plane0/`
    - `F.npy`: raw fluorescence traces, shape `(n_neurons, n_frames)`, `float32`
    - `Fneu.npy`: neuropil fluorescence traces
    - `spks.npy`: suite2p deconvolved activity, shape `(n_neurons, n_frames)`, `float32`
    - `iscell.npy`: shape `(n_neurons, 2)`; already matched/exported Track2p cells
    - `ops.npy`: suite2p metadata, includes `fs=30`, `nframes`, `meanImg`
    - `stat.npy`: suite2p ROI metadata per cell
  - `move_deve/`
    - `motion_energy_glob.npy`: processed global motion energy from behavior video, one value per available camera frame
    - `tstamps.npy`: timestamps for available behavior frames
    - `interframe_int.npy`: inter-frame intervals used to identify dropped/missing camera frames
- `load_data.ipynb` in `/app/data` provides example loading. It explicitly says `F.npy` contains raw fluorescence and suggests either computing `dF/F` “the way as described in the paper” or using `spks.npy` directly.
- `ground_truth.csv` exists for `jm038`, `jm039`, and `jm046`. It is a semicolon-separated table of manual cross-day cell correspondences (`D1`-`D7` columns). This looks useful for Track2p evaluation, not for the requested decoder.
- All inspected sessions have exactly one suite2p plane: `plane0`.
- The suite2p folders already contain neurons “present across all days” for that subject, so neuron row identities are aligned across days within each subject per `/app/data/README.md`.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 2,998 unique matched neurons across subjects; 20,445 neuron-session entries summed across all sessions |
| Neurons / session | Subject-specific constant counts: 221 (`jm031`), 370 (`jm032`), 685 (`jm038`), 746 (`jm039`), 541 (`jm040`), 435 (`jm046`) |
| Subjects | 6 |
| Sessions / subject | `jm031`: 7, `jm032`: 7, `jm038`: 7, `jm039`: 7, `jm040`: 6, `jm046`: 7 |
| Trials (total) | No native trials; continuous recordings. If split into 60 s windows at 30 Hz, there are 1,090 exact full windows total |
| Trials / session | No native trials; 20 windows for 36,000-frame sessions and 30 windows for 54,000-frame sessions |

Additional native-data observations:
- 41 sessions total.
- Session frame counts are either 36,000 frames (20 min at 30 Hz) or 54,000 frames (30 min at 30 Hz).
- `ops['nframes']` matches `F.shape[1]` in inspected sessions.
- `motion_energy_glob.npy` and `tstamps.npy` always have the same length.
- In 32/41 sessions, motion length matches imaging frame count exactly.
- In 9/41 sessions, camera-frame samples are missing relative to imaging frames by 1, 2, 3, 116, or 148 frames depending on session.
- Missing behavior frames are visible as approximately doubled intervals in `interframe_int.npy`, consistent with `/app/data/README.md`.
- `tstamps.npy` appears to be in kiloseconds rather than seconds: `(session_duration_seconds) / (tstamps[-1] - tstamps[0])` is consistently about 992, so multiplying timestamps by 1000 should recover near-second units.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Average 526 ± 190 tracked neurons per mouse across all days | “On average 526 (± 190 std) neurons per mouse were successfully tracked across all days using Track2p” |
| Neurons / session | Hundreds of tracked neurons per mouse/day; same tracked population across days within mouse | “yielded hundreds of identified neurons tracked across all days” |
| Subjects | 6 mice | “we used a full dataset of 6 mice imaged daily” |
| Sessions / subject | Minimum 6 consecutive daily sessions; example dataset 7 sessions from one mouse | “for at least 6 consecutive days” and “n=7 imaging sessions from one mouse” |
| Trials (total) | No native trials described; recordings are continuous sessions | Paper describes continuous recordings and 2 min blocks for cross-validation, not trialized data |
| Trials / session | Not applicable natively; session is continuous | “each session lasted 20 minutes” |
| Neural data time bin | Native sampling 30 Hz; decoding analyses average 10 timestamps | “Imaging rate was 30 Hz” and “averaging in bins of 10 consecutive timestamps” |
| Behavior data time bin | Native videography 30 Hz; decoding analyses average 10 timestamps | “Videos were recorded at 30 Hz” and “averaging in bins of 10 consecutive timestamps” |
| Reward rate | Not applicable | No reward-based task; spontaneous behavior dataset |
| Motion variable | Global motion energy from squared pixelwise frame differences | “computed their pixelwise difference… squared all individual pixel-wise values and summed across pixels” |
| Decoder evaluation metric | Continuous regression performance reported as R2, same-day and cross-day | “Graphs indicating R2 values for same-day cross-validated decoding performance” |
| Developmental transition | Behavioral-state representation emerges around P11 | “marking the emergence of a stable behavioral state representation” |


### Processing Details
- Imaging modality: chronic 2-photon calcium imaging in mouse barrel cortex layer 2/3, 512×512 pixels, 720×720 µm field of view.
- Sampling/alignment: imaging at 30 Hz; videography at 30 Hz with microscope acquisition triggering camera frames, described as allowing “simple synchronisation across the two modalities.”
- Neural preprocessing used in paper: Suite2p motion correction, ROI detection, signal extraction, spike deconvolution, then keep ROIs with Suite2p cell-classifier probability above 0.5.
- Neural signal used for downstream analyses in paper: baseline-corrected fluorescence traces treated as dF/F using default Suite2p parameters.
- Behavioral preprocessing: motion energy computed from consecutive video-frame differences, squaring pixel differences and summing over pixels to get one scalar per timepoint.
- Decoder preprocessing in paper: both dF/F and behavior traces are slightly denoised by averaging in bins of 10 consecutive timestamps (at 30 Hz this corresponds to ~333 ms bins).
- Same-day decoder evaluation in paper: nested cross-validation with inner and outer 5-fold loops, splitting recordings into consecutive 2-minute blocks.
- Cross-day decoder evaluation in paper: fit on one day, test on all other days using the tracked matched-neuron population.

### Curation Steps

**Neuron curation rules**:
- Use Suite2p preprocessing separately on each recording.
- Accept ROIs classified as true cells using the default Suite2p threshold of 0.5.
- Track cells across days with Track2p and analyze the neurons successfully tracked across all recorded days for a mouse.

**Trial curation rules**:
- No native trials are described in the paper; recordings are continuous spontaneous-behavior sessions.
- Decoder evaluation blocks in the paper use consecutive 2-minute chunks for cross-validation, but the user task explicitly requires conversion into 60-second trials for downstream decoding.

### Decoders Trained
| Decoded variable | Accuracy |
| Mouse motion / behavioral state from neural population activity | No explicit numeric table in text; figure captions/reporting use R2 and state that same-day decoding increases with development and late cross-day decoding is accurate/stable |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Neural data representation | Track2p code exports matched `F`, `Fneu`, `spks`, `stat`, `iscell` in suite2p format; no `dF/F` computation inside repository | Release contains matched `F.npy`, `Fneu.npy`, `spks.npy`, `ops.npy`, `stat.npy`, `iscell.npy` per session | Downstream analyses used “baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)” | Treat Track2p output as the matched-neuron substrate and reproduce Suite2p-style dF/F preprocessing from the saved suite2p arrays/ops when building decoder inputs. If exact reconstruction proves impossible, document the fallback. |
| Cell filtering | Reference code applies `iscell[:,1] > 0.5` by default | Released matched sessions already satisfy this: minimum observed `iscell[:,1]` across sessions is `0.500251773...` | Paper says “We considered all ROIs above the default threshold of 0.5 as true cells.” | No extra ROI filtering beyond the released matched suite2p files is needed; the data are already filtered consistently with code/paper. |
| Longitudinal matching | Track2p `save_in_s2p_format()` keeps only rows matched across all days for a mouse | Data README states neurons are “present across all days” and row identities match across days within mouse; neuron count is constant within a subject across sessions | Paper reports neurons “tracked across all days” for downstream analyses | Use the provided session files directly as already matched across all recorded days per subject; no need to rerun Track2p. |
| Session duration | Track2p code is agnostic to duration | Sessions are either 36,000 frames (20 min) or 54,000 frames (30 min) at 30 Hz | Methods text says “each session lasted 20 minutes” | Use actual frame counts in the released data as the source of truth and document the mismatch. This likely reflects heterogeneity between cohorts or an imprecise summary in the manuscript. |
| Behavior-neural alignment | Track2p code does not handle behavior | Motion arrays usually match imaging length, but 9 sessions have missing behavior frames detectable in `tstamps.npy`/`interframe_int.npy` | Data README explicitly mentions missing camera frames and says missing-frame indices can be obtained from timestamps / interframe intervals; methods say camera was triggered by microscope at 30 Hz | Reconstruct per-imaging-frame behavior by using timestamps/interframe intervals to place available motion samples on the full imaging frame grid, then interpolate or otherwise fill only the missing camera-frame positions. |
| Decoder protocol | Track2p repository has no decoder | Data are continuous recordings, not trials | Paper decoding uses ridge regression on continuous traces, 10-frame averaging, and 2-minute CV blocks | For the requested deliverable, preserve the paper’s 10-frame denoising/binning but adapt the representation to the user’s 60-second trial format and categorical output requirement. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `suite2p/plane0/F.npy` + `suite2p/plane0/Fneu.npy` + `suite2p/plane0/ops.npy` | `neural` | Compute `dF = F - ops['neucoeff'] * Fneu`, then apply Suite2p `dcnv.preprocess()` with saved default params (`baseline`, `win_baseline`, `sig_baseline`, `fs`, `prctile_baseline`, `batch_size`); average in non-overlapping 10-frame bins; split into 60 s trials; transpose to `(n_neurons, n_timepoints)` per trial | Track2p `save_in_s2p_format()` preserves matched `F/Fneu`; Suite2p `pipeline_s2p.py` deconvolution path shows `dF = F - neucoeff*Fneu` then `preprocess()` | Chosen to match the paper’s downstream analyses, which use Suite2p baseline-corrected fluorescence traces as dF/F. |
| Imaging frame index / `ops['fs']` | `input[0]` | Construct elapsed-time array in seconds on the imaging axis, then average in the same 10-frame bins and split into 60 s trials. Use absolute time from session start, not time-within-trial | Not from Track2p code; required by user decoder spec | One input dimension named `time_from_session_start_sec`. |
| `move_deve/motion_energy_glob.npy` + `tstamps.npy` + `interframe_int.npy` | `output[0]` | Reconstruct full per-imaging-frame motion series by assigning measured motion to frame indices inferred from timestamps and filling missing camera frames by interpolation; average in non-overlapping 10-frame bins; compute session-specific quintile edges from the full binned session; discretize into 5 categories; split into 60 s trials | No corresponding Track2p code; follows data README + paper motion-energy definition | One time-varying categorical output named `motion_energy_bin`; categories are session-specific quintiles. |
| Subject folder name (e.g. `jm031`) | `subjects`, `subject_idx` | Unique sorted subject list; per-session subject index | N/A | Session order will follow subject/date sorting for reproducibility. |
| Paper/methods metadata | `brain_regions`, `brain_region_idx` | Single region list `['barrel cortex L2/3']`; all neurons map to region index 0 | N/A | All recordings are from barrel cortex layer 2/3. |

### Key Decisions
1. **Use matched Suite2p exports directly**: The release already contains Track2p outputs restricted to neurons present across all days within each mouse, which matches the paper’s downstream analysis population and avoids rerunning the tracking algorithm.
2. **Use Suite2p-style baseline-corrected fluorescence rather than raw `F.npy` or `spks.npy` for `neural`**: The paper explicitly states that subsequent analyses, including decoding, used baseline-corrected fluorescence traces as dF/F. Because `F.npy` is raw fluorescence and `Fneu.npy` plus `ops.npy` are available, reproducing the Suite2p preprocessing is better matched to the reference than using `spks.npy`.
3. **Apply the paper’s 10-frame averaging before decoder formatting**: This preserves the denoising/binning used in the reference decoding while reducing dimensionality. At 30 Hz, 10 frames = 1/3 s bins, so 60 s trials contain exactly 180 time bins.
4. **Use imaging frames as the canonical time axis**: Neural traces are complete on this axis; behavior is mapped onto it. This avoids drifting to the shorter camera series in sessions with dropped video frames.
5. **Interpolate only missing camera-frame positions**: The data README explicitly points to timestamps/inter-frame intervals for handling missing camera frames. Filling only inferred missing samples preserves measured motion everywhere else while producing aligned dense behavior for the decoder.
6. **Split sessions into contiguous 60 s windows without overlap**: This matches the user requirement and is exact because all sessions are multiples of 1,800 imaging frames (60 s at 30 Hz).
7. **Use session-specific motion-energy quintiles after denoising**: The user requires five equal-percentile bins “selected per session.” Applying percentiles on the binned behavior series aligns the labels with the actual decoder target time base.
8. **Keep all sessions and all full windows**: Every session yields at least 20 trials, so no session needs exclusion for insufficient trials. No extra neuron/session filtering beyond the provided matched-cell release is justified.
9. **Represent input as absolute session time in seconds**: The user asked specifically for time elapsed from session start. This variable should continue increasing across trials rather than resetting at each 60 s boundary.

### Planned Sanity Checks
- [ ] Neural preprocessing sanity check: for selected neurons/session, verify converted neural bins equal manual computation from raw `F`, `Fneu`, `ops`, Suite2p `preprocess()`, and 10-frame averaging via `np.allclose()`.
- [ ] Behavior alignment sanity check: for a no-drop session, verify reconstructed full motion equals raw `motion_energy_glob.npy` exactly; for a drop session, verify inferred missing-frame count equals `n_imaging_frames - len(motion)` and non-missing positions match exactly.
- [ ] Trialization sanity check: verify each trial is exactly 180 bins and concatenating all trials reconstructs the full-session binned arrays.
- [ ] Output discretization sanity check: verify per-session class fractions are approximately 0.2 each after quintile binning, allowing small deviations from ties.
- [ ] Metadata/count sanity check: verify subject count, session count, neurons/session, and trials/session match the raw data and paper-derived expectations.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with the following structure:
- Session discovery from `/app/data` with deterministic ordering.
- Representative `--sample` mode selecting two sessions that exercise both a dropped-camera-frames edge case and a different session duration.
- Suite2p-style neural preprocessing:
  - load `F.npy`, `Fneu.npy`, `ops.npy`
  - compute neuropil-subtracted traces `F - neucoeff * Fneu`
  - apply `suite2p.extraction.dcnv.preprocess()` using saved per-session Suite2p parameters
- Behavior alignment:
  - infer available behavior-frame positions from `tstamps.npy`
  - reconstruct a dense motion-energy series on the imaging frame grid
  - linearly interpolate only missing camera-frame positions
- Downstream formatting:
  - average neural, time, and motion traces in 10-frame bins
  - discretize binned motion into five session-specific bins
  - split each session into non-overlapping 60 s trials
  - package as the required dictionary and save to pickle
- Processing visualizations saved as `processing_<subject>_<session>.png` when `--show-processing` is used.
- Runtime reporting with per-session timing and ETA.

Code inefficiencies identified:
- Plot generation reloads `ops.npy` once inside the plotting helper; negligible for current scale but avoidable if necessary.
- Conversion is currently single-process by session; likely acceptable for 41 sessions, but full-run timing will determine whether further optimization is required.

Code speedups added:
- Use `mmap_mode='r'` when discovering sessions so cataloging does not load full arrays.
- Process one session at a time to cap peak memory.
- Use vectorized reshape/mean operations for 10-frame binning.
- Avoid temporary concatenations when trializing by slicing already binned arrays.
- First execution test via `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing` completed without errors.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 1,055 neuron-session entries across 2 sessions |
| Neurons / session | 370 (`jm032_2023-10-22_a`), 685 (`jm038_2023-04-30_a`) |
| Subjects | 2 |
| Sessions / subject | 1 each (`jm032`, `jm038`) |
| Trials (total) | 50 |
| Trials / session | 20 and 30 |
| `time_from_session_start_sec` range | `[0.15, 1799.82]` overall; `[0.15, 1199.82]` and `[0.15, 1799.82]` per session |
| `motion_energy_bin` distribution | `[0.2, 0.2, 0.2, 0.2, 0.2]` overall and exactly balanced within each sample session |

### Processing Plots Review
- Generated:
  - `processing_jm032_2023-10-22_a.png`
  - `processing_jm038_2023-04-30_a.png`
- The two plotted sessions intentionally cover:
  - one session with substantial dropped camera frames (`jm032_2023-10-22_a`, 148 missing frames)
  - one session with perfect frame alignment (`jm038_2023-04-30_a`, 0 missing frames)
- Quantitative checks underlying the plots showed no anomalies:
  - reconstructed motion matched raw observed samples exactly at all non-missing timestamps
  - inferred missing-frame counts matched `n_imaging_frames - len(motion_energy_glob)` exactly
  - 10-frame binning produced exactly 180 bins per 60 s trial
  - quintile discretization yielded balanced classes in both plotted sessions

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |
| Per-session processing only | Keeps peak memory bounded while remaining fast enough for the full 41-session dataset |
| Vectorized 10-frame binning | Avoids Python loops over time bins |
| Timestamp-derived frame indexing | Avoids expensive search/matching during behavior alignment |

| Step | Time / Session | Estimated Total Time |
| | | |
| Sample conversion | ~1.20 s mean over 2 representative sessions | ~52.5 s estimated for all 41 sessions |
| Format verification (`train_decoder.py --verify-only`) | ~1 s for sample file | Negligible compared with conversion |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| `motion_energy_bin` | 0.6811 | 0.3295 |

Notes:
- Chance level is 0.2000 (5 classes).
- Training loss decreased monotonically from `70.768858` at epoch 1 to `0.918494` at epoch 200.
- Validation balanced accuracy is above chance, indicating the conversion is coherent enough to support decoder training on the sample.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 396M
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | “On average 526 (±190 std) neurons per mouse” across 6 mice implies ~3156 by rough multiplication, but paper does not give an exact total | Track2p code has no fixed dataset statistic | 2,998 unique matched neurons across subjects; 20,445 neuron-session entries across 41 sessions | 2,998 unique matched neurons represented across 20,445 neuron-session entries | Matches reference data exactly; paper statistic is approximate and somewhat higher but within the reported spread |
| Mean neurons/session | Paper reports per-mouse matched-cell counts, not per-session counts | N/A | 498.66 neuron-session entries/session; 499.67 unique matched neurons/mouse | 498.66 neuron-session entries/session; 499.67 unique matched neurons/mouse | Matches reference data |
| Subjects | 6 mice | N/A | 6 | 6 | Yes |
| Sessions | Minimum 6 consecutive daily sessions per mouse; full dataset of 6 mice | N/A | 41 total sessions (7,7,7,7,6,7) | 41 total sessions (7,7,7,7,6,7) | Yes |
| Trials (total) | No native trials in paper; continuous sessions | N/A | 1,090 exact 60 s windows derivable from frame counts | 1,090 | Yes |
| Trials/session (mean) | No native trial count in paper | N/A | Mean 26.59 (20 for 20 min sessions, 30 for 30 min sessions) | Mean 26.59 (20 or 30) | Yes |
| `time_from_session_start_sec` range | 20 min sessions described in methods; would imply ~`[0.15, 1199.82]` after 10-frame binning | N/A | Data include both 20 min and 30 min sessions, implying overall `[0.15, 1799.82]` | `[0.15, 1799.82]` overall | Matches reference data; paper under-describes longer sessions |
| Neural bin size | 10 timestamps at 30 Hz (~333 ms) | Suite2p preprocessing plus Track2p matching; no decoder code | 10-frame bins feasible from all sessions | 333.33 ms | Yes |
| `motion_energy_bin` distribution | Not categorical in paper; continuous motion trace | N/A | Session-specific quantile discretization target should be approximately balanced by construction | `[0.2, 0.2, 0.2, 0.2, 0.2]` overall and within every session | Yes |

Additional Step 9 checks:
- `train_decoder.py --verify-only` reported: `Data format is valid, no errors or warnings.`
- Spot checks against raw data:
  - `jm031_2023-10-22_a`: raw 36,000 frames -> converted 20 trials, 221 neurons, 116 missing motion frames handled.
  - `jm038_2023-04-30_a`: raw 54,000 frames -> converted 30 trials, 685 neurons, no missing motion frames.
  - `jm046_2024-09-09_a`: raw 54,000 frames -> converted 30 trials, 435 neurons, 1 missing motion frame handled.
- Session-wide trial lengths are uniform: every trial has exactly 180 binned timepoints.
- Full conversion runtime after the timestamp-fix iteration: ~20.6 s total.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `/app/verification_full_out.txt` reports `Data format is valid, no errors or warnings.` No warnings required triage.
2. **Construct raw-data sanity checks with `np.allclose()`**:
   - Neural sanity check:
     - `jm031_2023-10-18_a`, session 0, trial 0, neuron 3, binned time 10: independently recomputed from raw `F`, `Fneu`, `ops`, Suite2p `preprocess()`. `np.allclose=True` (`11.024763107299805` vs `11.024763107299805`).
     - `jm031_2023-10-22_a`, dropped-frame session, session 4, trial 7, neuron 12, binned time 50. `np.allclose=True` (`59.394920349121094` vs `59.394920349121094`).
     - `jm046_2024-09-09_a`, one-missing-frame session, session 40, trial 29, neuron 100, binned time 179. `np.allclose=True` (`15.142362594604492` vs `15.142362594604492`).
   - Input sanity check:
     - Same three spots above compared against independent `np.arange(nframes)/30` time averaging. All `np.allclose=True`.
   - Output sanity check:
     - Same three spots above compared against independently reconstructed motion traces, 10-frame binning, and session-specific quantile discretization. All `np.allclose=True`.
3. **Reference code comparison**:
   - (a) Data loading:
     - Reference: `track2p/t2p.py::save_in_s2p_format()` and `track2p/io/s2p_loaders.py` load suite2p `F/Fneu/spks/stat/iscell/ops`.
     - Mine: `convert_data.py` loads the same matched suite2p files directly from the release.
   - (b) Neuron/trial filtering:
     - Reference: keep ROIs with `iscell[:,1] > 0.5` and only cells matched across all days.
     - Mine: no additional ROI filtering; confirmed all released `iscell` probabilities are already >0.5 and neuron counts are constant within mouse across days, consistent with Track2p matched exports.
   - (c) Temporal alignment:
     - Reference paper/data: camera acquisition triggered by microscope; dropped frames recoverable from timestamps/interframe intervals.
     - Mine: imaging frames are the canonical axis; behavior is mapped onto it by timestamp-derived frame indices and interpolation only over missing camera frames.
   - (d) Binning:
     - Reference paper: average neural and behavior traces in bins of 10 consecutive timestamps.
     - Mine: exact 10-frame non-overlapping averaging before trialization.
   - (e) Input construction:
     - No reference implementation because the user changed decoder inputs.
     - Mine: absolute session time in seconds, binned on the same 10-frame grid.
   - (f) Output construction:
     - Reference paper decodes continuous motion (regression, R2).
     - Mine: same motion-energy signal and alignment, but discretized into five equal-percentile bins per session because the user requires categorical outputs.
4. **Key statistics comparison**:
   - Subject count, session count, neurons/session, and total 60 s trials match the raw data exactly.
   - Full validator confirms uniform trial length (180 bins), single input, single output, and categorical output values 0-4.
   - Output distribution is exactly balanced within each session and overall, as intended by per-session quintile discretization.
   - Paper/data mismatches that remain external to the conversion:
     - Session duration: methods describe 20 min sessions, but the release contains both 20 min and 30 min sessions.
     - Mean tracked neurons/mouse: data release gives 2998/6 = 499.67, whereas the paper text reports 526 ± 190. This is qualitatively consistent with the reported spread but not numerically identical.
5. **Edge-case review**:
   - Large dropped-frame sessions (`jm031_2023-10-22_a`, `jm032_2023-10-22_a`) handled correctly; missing counts 116 and 148 match raw-data differences exactly.
   - Small dropped-frame sessions (1-3 missing frames) handled correctly across all affected recordings.
   - Zero-missing sessions with timestamp jitter in `jm046` exposed a bug in the first full-conversion attempt (duplicate/overshooting timestamp-derived indices despite equal lengths). Fixed by bypassing timestamp inference when behavior length already equals imaging-frame count and using span-based indexing only for true missing-frame sessions.
   - Global sweep after the fix found no mismatches in expected trial counts or missing-frame counts.

### Issues Found and Resolved
- **Issue**: Timestamp-based frame-index inference overshot the imaging grid for `jm046_2024-09-05_a` during the first Step 9 full-conversion run.
  - **Resolution**: Updated `infer_behavior_frame_indices()` to return `np.arange(n_frames)` immediately when `len(tstamps) == n_frames`, and to use a span-based frame step only when there are actual missing camera frames. Re-ran Step 9 full conversion and validation successfully.
- **Issue**: Potential concern that the paper used dF/F whereas the release stores raw `F.npy`.
  - **Resolution**: Confirmed Suite2p package is available and independently reproduced the paper-consistent preprocessing path `dF = F - neucoeff * Fneu` followed by `suite2p.extraction.dcnv.preprocess()`; raw-vs-converted sanity checks passed exactly.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| `motion_energy_bin` | 0.6045 | 0.3027 | Chance = 0.2000; validation is slightly above 1.5x chance (`0.3000`) |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| `motion_energy_bin` | Validation balanced accuracy = `0.3027` (chance = `0.2000`; 1.5x chance = `0.3000`) | Paper reports successful same-day and late cross-day decoding of continuous motion using `R2`, with performance increasing developmentally; no exact balanced-accuracy or categorical-classification values are reported in the text |

Accuracy review:
- **Accuracy vs chance**:
  - Validation balanced accuracy is above chance (`0.3027 > 0.2000`) and slightly above the `1.5x chance` heuristic threshold (`0.3027 > 0.3000`).
  - This is therefore not a below-chance or clearly broken result.
- **Accuracy comparison to paper**:
  - The paper decodes **continuous** motion using ridge regression and reports performance with **R2** matrices/box plots (Fig. 7 / Fig. S7), not categorical accuracy.
  - Because the user-required task changes both the target representation (continuous motion -> 5-class session-wise quintiles) and the decoder family (paper: ridge regression; validation script: classifier trained on the provided format), exact numeric agreement is not available.
  - Qualitatively, the paper says behavioral state can be decoded from neural activity and that performance improves across development. Our converted dataset supports above-chance decoding on the full dataset, which is consistent with that qualitative expectation.
- **Train vs validation gap**:
  - Training balanced accuracy = `0.6045`; validation balanced accuracy = `0.3027`; ratio ≈ `2.00`.
  - Investigated possible leakage/format bugs:
    - Trial-level train/test splits are handled by `train_decoder.py`, not by the conversion script.
    - Outputs are exactly balanced within each session, so the decoder is not exploiting a majority class.
    - Raw-data sanity checks for neural, input, and output values all passed exactly.
    - Missing-frame handling and trial counts were rechecked globally after the timestamp-fix iteration.
  - Conclusion: the gap is consistent with model overfit on a high-dimensional pooled decoding problem, not with an obvious conversion error or data leakage.
- **Low-accuracy debugging checklist**:
  - Raw output values were spot-checked directly against original files on three sessions, including dropped-frame edge cases.
  - Temporal alignment was checked both numerically and through saved processing plots for clean and dropped-frame sessions.
  - Output variation is adequate by construction: each session has exactly balanced 5-way classes.
  - Neural preprocessing follows the paper-consistent Suite2p path rather than an ad hoc transform.

### Issues Found and Resolved
- **Issue**: Train/validation performance gap (`0.6045` vs `0.3027`) exceeds the `1.5x` heuristic.
  - **Resolution**: Investigated class balance, trial splitting, raw-to-converted numerical sanity checks, and missing-frame edge cases. No evidence of leakage or misalignment was found; retained the current conversion because it best matches the reference preprocessing and still gives above-chance validation accuracy.
- **Issue**: Direct numerical comparison to the paper is not possible because the paper reports `R2` for continuous regression, while the required deliverable is 5-class categorical decoding with balanced accuracy.
  - **Resolution**: Documented the metric/task mismatch explicitly and limited the paper comparison to qualitative consistency.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
