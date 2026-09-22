# Dataset Conversion Notes

## Overview
- **Dataset**: [Name and source]
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

Environment verification:
- `python3` available
- `numpy 2.4.4` imports successfully
- `torch 2.6.0+cu124` imports successfully

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `run_t2p(track_ops)` | `code/track2p/t2p.py` | PROCESSING | Main Track2p pipeline: loads suite2p data, registers fields of view, matches ROIs across sessions, saves match matrices and tracked outputs. |
| `generate_suite2p_indices(track_ops)` | `code/track2p/t2p.py` | PROCESSING | Converts Track2p match indices back to original suite2p ROI indices after `iscell` filtering. |
| `save_in_s2p_format(track_ops)` | `code/track2p/t2p.py` | PROCESSING | Builds tracked suite2p-format outputs for cells present across all days; subsets `F`, `Fneu`, `spks`, `stat`, `iscell` using match matrices. |
| `check_nplanes(track_ops)` | `code/track2p/io/s2p_loaders.py` | LOADING | Confirms all datasets have the same number of planes. |
| `load_all_imgs(track_ops)` | `code/track2p/io/s2p_loaders.py` | LOADING | Loads `ops.npy` mean images and determines channel count. |
| `load_all_ds_stat_iscell(track_ops)` | `code/track2p/io/s2p_loaders.py` | CURATION | Loads `stat.npy` and filters ROIs by suite2p `iscell` threshold. |
| `load_stat_ds_plane(track_ops_path, track_ops, plane_idx)` | `code/track2p/io/loaders.py` | CURATION | Loads one plane’s `stat.npy` and applies Track2p-consistent `iscell` filtering. |
| `npy_to_s2p(track_ops)` | `code/track2p/io/savers.py` | LOADING | Converts simple `npy` inputs into minimal suite2p-like outputs (`F`, `ops`, `stat`, `iscell`). |
| `DefaultTrackOps.__init__` | `code/track2p/ops/default.py` | CURATION | Defines default parameters including `iscell_thr = 0.50`, registration channel, and matching method. |

### Notes
- The reference repository is the Track2p tracking package, not a behavioral decoding codebase.
- The repository establishes how longitudinal calcium imaging sessions are curated and indexed:
  - Neural source files are suite2p outputs (`F.npy`, `Fneu.npy`, `spks.npy`, `stat.npy`, `iscell.npy`, `ops.npy`).
  - Cell filtering is based on suite2p `iscell`; default rule is keep ROIs with `iscell[:,1] > 0.5`, unless `iscell_thr is None`, in which case use `iscell[:,0] == 1`.
  - Track2p match matrices are in Track2p index space after `iscell` filtering, so any downstream loading must apply the same filter before indexing.
  - Rows with any `None` in `plane*_match_mat.npy` indicate cells not tracked across all sessions; the demo notebook explicitly keeps only rows with no `None` for all-day matched cells.
- The demo notebook `code/notebooks/demo_t2p_ouputs.ipynb` shows downstream extraction logic:
  - Load `track_ops.npy` and `plane*_match_mat.npy`.
  - Filter each session’s suite2p arrays by the same `iscell` threshold.
  - Subset filtered `F` by rows from `t2p_match_mat_allday = t2p_match_mat[~np.any(t2p_match_mat == None, axis=1), :]`.
- No delta-F/F computation is implemented in the reference code; the examples use raw suite2p `F.npy` traces for tracked cells.
- No additional neuron-quality filtering beyond suite2p `iscell` threshold is present in the reference package.
- `ops.npy` contains imaging metadata including `fs` (sampling frequency) and `nframes`, which will be needed for temporal binning/alignment.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Top-level files in `data/`:
  - `README.md`: dataset description and loading notes.
  - `load_data.ipynb`: example notebook for loading tracked traces and fields of view.
- Subject folders: `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`.
- Each subject contains daily session folders named like `YYYY-MM-DD_a`.
- Each session contains:
  - `suite2p/plane0/`
    - `F.npy`: raw fluorescence traces, shape `(n_tracked_neurons, n_frames)`.
    - `Fneu.npy`: neuropil fluorescence, same shape as `F.npy`.
    - `spks.npy`: suite2p deconvolved activity, same shape as `F.npy`.
    - `iscell.npy`: shape `(n_tracked_neurons, 2)`.
    - `stat.npy`: ROI dictionaries with pixel coordinates and centroids.
    - `ops.npy`: metadata (`fs`, `nframes`, `meanImg`, image size, etc.).
  - `move_deve/`
    - `motion_energy_glob.npy`: processed motion energy from videography.
    - `tstamps.npy`: timestamps for motion samples.
    - `interframe_int.npy`: timestamp differences between consecutive motion samples.
- Additional files:
  - `ground_truth.csv` exists for `jm038`, `jm039`, `jm046`; these appear to be manual cell-tracking ground truth tables, not behavioral labels for decoding.
- Important native property from `data/README.md` and inspection:
  - `suite2p/plane0/*` already contains Track2p outputs saved in suite2p format for cells present across all days of a subject.
  - Therefore neuron count is constant across sessions within a subject, and row `i` refers to the same tracked neuron across that subject’s sessions.
- Native recordings are continuous sessions, not trialized. For the decoder task, 60 s trials will need to be created later from session time.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 20,445 session-neuron entries across all sessions |
| Neurons / session | min 221, mean 498.66, max 746 |
| Subjects | 6 (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`) |
| Sessions / subject | `jm031`: 7, `jm032`: 7, `jm038`: 7, `jm039`: 7, `jm040`: 6, `jm046`: 7 |
| Trials (total) | No native trials; if split into 60 s chunks using 30 Hz frames, 1,090 trials total |
| Trials / session | No native trials; 20 for 36,000-frame sessions and 30 for 54,000-frame sessions |

Additional observations from data inspection:
- Total sessions: 41.
- All inspected sessions have:
  - one imaging plane (`plane0`)
  - `fs = 30 Hz`
  - `meanImg` size `512 x 512`
  - `ops['nframes']` matching `F.shape[1]`
  - `spks.npy` and `Fneu.npy` shapes matching `F.npy`
- Session lengths:
  - Subjects `jm031`, `jm032`: 36,000 imaging frames per session (20 minutes at 30 Hz).
  - Subjects `jm038`, `jm039`, `jm040`, `jm046`: 54,000 imaging frames per session (30 minutes at 30 Hz), except `jm040` has only 6 daily sessions.
- Motion stream lengths usually match imaging frames but some sessions have fewer motion samples due to missing camera frames:
  - missing counts observed: `0, 1, 2, 3, 116, 148`.
  - largest gaps occur in `jm031/2023-10-22_a` and `jm032/2023-10-22_a`.
- `iscell.npy` is already filtered to tracked cells and all observed rows have first column `1`; the second column stores suite2p probabilities slightly above `0.5`.
- `tstamps.npy` / `interframe_int.npy` are not stored in seconds directly; sample interval is approximately `3.36e-05`, which corresponds to about `0.0336 s` if interpreted as kiloseconds. This will need explicit conversion during alignment.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Average 526 tracked neurons per mouse across all days | “On average 526 (± 190 std) neurons per mouse were successfully tracked across all days” |
| Neurons / session | Same tracked cells present across all days for a given mouse | Data README: traces “only includes traces for the cells present across all days” |
| Subjects | 6 mice | “A total of 6 mice were used in the study” |
| Sessions / subject | At least 6 consecutive daily sessions | “for at least 6 consecutive days”; “minimum of 6 consecutive days” |
| Trials (total) | No native trials reported; paper decoding used consecutive 2 min blocks | “splits were done on consecutive 2 minute blocks of the recording” |
| Trials / session | No native trial structure in paper | Not trial-based; continuous recordings |
| Neural data time bin | Native 30 Hz imaging; paper decoder used 10-frame averaging | “Imaging rate was 30 Hz”; “averaging in bins of 10 consecutive timestamps” |
| Behavior data time bin | Native 30 Hz videography; paper decoder used 10-frame averaging | “Videos were recorded at 30 Hz”; “averaging in bins of 10 consecutive timestamps” |
| Reward rate | N/A | No reward task; spontaneous behavior |
| Behavioral variable | Motion energy from global movement video | “assessed behavioural state indirectly... using a ‘motion energy’ metric” |
| Motion energy definition | Sum of squared pixelwise frame differences | “computed their pixelwise difference... squared... and summed across pixels” |
| Decoder metric | R² for ridge-regression prediction of motion | Figure 7 caption: “Graphs indicating R2 values for same-day cross-validated decoding performance” |


### Processing Details
- Imaging:
  - Layer 2/3 of barrel cortex, depth 100-200 µm.
  - FOV 720 x 720 µm, image size 512 x 512 pixels.
  - Native frame rate 30 Hz.
  - Methods text states “each session lasted 20 minutes.”
- Behavior:
  - Infrared videography at 30 Hz.
  - Microscope acquisition triggered camera frames for synchronization.
  - Motion energy is a scalar per frame from consecutive-frame pixel differences.
- Neural analysis:
  - Suite2p preprocessing per recording: motion correction, ROI detection, signal extraction, spike deconvolution.
  - Paper analyses used baseline-corrected fluorescence traces as dF/F, using default Suite2p parameters.
  - Paper also discusses spike deconvolution and event-rate statistics, but dF/F was used for decoding analyses.
- Decoding in the paper:
  - Behavioral variable predicted from neural population activity.
  - Ridge regression with nested 5-fold cross-validation.
  - Splits done on consecutive 2-minute blocks.
  - dF/F and behavior were denoised by averaging 10 consecutive timestamps before decoding.
  - Same-day decoding performance increased with development.
  - Cross-day decoding became stable from later developmental stages.

### Curation Steps

**Neuron curation rules**:
- Keep ROIs classified as cells by Suite2p with probability above 0.5.
- Track2p downstream analyses use cells tracked across all days for a mouse.

**Trial curation rules**:
- No trial structure in the source experiment; recordings are continuous spontaneous-behavior sessions.
- Paper decoding used consecutive 2-minute temporal blocks for cross-validation, not event-defined trials.

### Decoders Trained
| Decoded variable | Accuracy |
| Motion / behavioural state from neural activity | R² metric used in paper; same-day decoding increases with age, late cross-day decoding is stable (Fig. 7). No exact numeric table in extracted text. |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Neural signal used in downstream analysis | Track2p repo/demo notebooks subset suite2p `F.npy` after `iscell` filtering and matching | Released sessions contain `F.npy`, `Fneu.npy`, `spks.npy`; `F.npy` is described in `load_data.ipynb` as “raw fluorescence traces” | “We used baseline corrected fluorescence traces as our dF/F ... for all subsequent analyses.” | Use the released suite2p-format tracked outputs, but compute/use a dF/F-like signal consistent with the paper rather than raw `F.npy` or `spks.npy` for the main decoder. This preserves the paper’s analysis choice while using the available released data. |
| Neuron filtering stage | Reference code applies `iscell > 0.5` before Track2p indexing | Released suite2p outputs already have constant tracked-neuron counts across days; all observed `iscell` first-column values are 1 and second-column values are > 0.5 | Paper says true cells are ROIs above Suite2p default threshold 0.5 | Treat released tracked suite2p outputs as already curated to the paper’s cell criterion; no extra cell filtering beyond validity checks is needed. |
| Tracking across days | Reference notebooks often keep rows with no `None` in `plane*_match_mat.npy` to get all-day cells | Data README states the provided suite2p folders already “only include traces for the cells present across all days” | Functional analyses use neurons tracked across all days | Use the provided session files directly; no need to recompute Track2p matches for this dataset. |
| Session duration | Reference code does not impose duration | Actual sessions are either 36,000 frames (20 min at 30 Hz) or 54,000 frames (30 min at 30 Hz) | Methods text says “each session lasted 20 minutes” | Use actual durations from the released data. Record this as a paper/data discrepancy; data files are authoritative for frame counts. Likely the release combines cohorts or includes longer recordings than the concise methods wording suggests. |
| Temporal structure for decoding | Paper decoding used continuous recordings split into consecutive 2-minute CV blocks after 10-frame averaging | Data are continuous sessions with synchronized imaging and behavior, plus occasional missing behavior frames | Paper does not define event-based trials | For the required benchmark dataset, segment continuous sessions into 60-second trials while preserving continuous-time alignment and using a common time bin size across neural/input/output. |
| Behavior alignment | Reference texts say camera was hardware-triggered by microscope; code repo does not implement behavior loading | Most sessions have equal imaging and behavior lengths, but some sessions have missing behavior samples and explicit timestamps / inter-frame intervals | Data README notes missing camera frames can be treated as missing or interpolated | Use imaging frames as the master timebase and reconstruct behavior on that grid using motion timestamps, with explicit handling of missing camera frames. |

Final understanding after reconciliation:
- Source recordings are continuous spontaneous-behavior calcium imaging sessions from tracked neurons in barrel cortex.
- The provided suite2p files are already Track2p-aligned across days and already restricted to tracked cells present across all days of each mouse.
- The paper’s downstream analyses rely on Suite2p baseline-corrected fluorescence (dF/F), not raw fluorescence or deconvolved spikes.
- Motion energy is the relevant behavioral stream and is synchronized to imaging but may have dropped camera frames that must be repaired/aligned.
- For the requested output format, the closest paper-consistent representation is a time-binned dF/F-like neural signal aligned to time-binned motion energy and split into fixed 60 s chunks.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `suite2p/plane0/F.npy`, `Fneu.npy`, `ops.npy` | `neural` | Compute neuropil-corrected fluorescence `F - 0.7*Fneu`, then Suite2p baseline preprocessing (`baseline='maximin'`, `win_baseline=60`, `sig_baseline=10`, `fs=30`), then average non-overlapping bins of 10 frames, then split into 60 s trials | Paper methods; installed `suite2p.extraction.dcnv.preprocess`; Suite2p defaults discovered locally | Chosen to match the paper’s “baseline corrected fluorescence traces as our dF/F” rather than raw `F.npy` or `spks.npy` |
| Session frame index / imaging clock | `input[0]` | Convert 10-frame bins to elapsed session time in seconds using bin-center times; retain absolute time from session start within each session | Paper native timing (30 Hz); decoder task spec | One time-varying input dimension named `time_from_session_start_sec` |
| `move_deve/motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy` | `output[0]` | Align motion samples to full imaging-frame grid using timestamps, interpolate missing camera frames on imaging timeline, average non-overlapping bins of 10 frames, compute per-session quintile bins, then split into 60 s trials | Paper videography methods; data README on missing frames | One categorical time-varying output dimension named `motion_energy_quintile` |
| Subject folder name (`jm031`, etc.) | `subjects`, `subject_idx` | Unique subject ids; assign per session in sorted session order | Data organization | Session order will be sorted by subject then session date |
| Barrel cortex recording location | `brain_regions`, `brain_region_idx` | Single region for all neurons: `barrel cortex L2/3`; per-session neuron region indices all zeros | Paper methods | All neurons in all sessions share same region label |
| Fixed labels | `output_values` | `['q1_lowest','q2_low','q3_mid','q4_high','q5_highest']` | Decoder task spec | Quintiles computed separately within each session |

### Key Decisions
1. **Neural signal = Suite2p-style baseline-corrected fluorescence, not raw `F.npy` or `spks.npy`**: The paper explicitly states that all downstream analyses, including decoding, used baseline-corrected fluorescence traces as dF/F. The released data do not include a separate dF/F file, but they do include the ingredients (`F`, `Fneu`, `ops`) needed to reconstruct the Suite2p preprocessing choice.
2. **Use 10-frame averaging before trialization**: The paper’s decoding analysis averaged both neural and behavioral traces in bins of 10 consecutive timestamps. Applying the same denoising before building the benchmark dataset is the closest paper-consistent preprocessing.
3. **Use imaging frames as the master clock**: Imaging has fixed frame count and defines the neural samples. Videography is intended to be synchronized but occasionally drops frames. Aligning motion to imaging-frame indices via timestamps avoids shortening sessions or dropping neural data.
4. **Interpolate only missing behavior frames**: The data README explicitly notes missing camera frames and suggests treating them as missing or interpolating over them. Because the decoder requires complete categorical targets at every timepoint, interpolation is the most practical paper-consistent choice, especially since missingness is sparse.
5. **Split into fixed 60-second trials after alignment and denoising**: This is required by the decoder task and preserves equal trial lengths. At 30 Hz with 10-frame averaging, each trial will contain 180 time bins.
6. **Discretize motion per session into quintiles**: The decoder output must be categorical, and the user explicitly requests five equal-percentile bins selected per session. This also normalizes across session-to-session changes in raw motion-energy scale.
7. **Use absolute elapsed session time as decoder input**: The task asks for “time elapsed from the beginning of the session in seconds.” Therefore trial inputs should continue the session clock rather than reset to zero within each trial.
8. **Keep all sessions and all tracked neurons provided in the release**: The release already contains Track2p-tracked neurons present across all days, and all sessions have at least 20 valid 60-second trials, satisfying decoder requirements without extra curation.

### Planned Sanity Checks
- [ ] Neural check: for selected sessions/neurons/timepoints, verify that the saved neural values equal the result of applying Suite2p neuropil subtraction + baseline preprocessing + 10-frame averaging to raw `F/Fneu`.
- [ ] Input check: verify `time_from_session_start_sec` equals the expected bin-center times from the raw 30 Hz imaging timeline.
- [ ] Output check: for selected sessions/timepoints, verify that aligned motion values match the raw `motion_energy_glob.npy` at non-missing timestamps and that interpolated frames only occur where timestamp gaps imply dropped camera frames.
- [ ] Output discretization check: confirm each session’s full binned motion series is partitioned into near-equal 20% class fractions before splitting into trials.
- [ ] Size check: confirm resulting trial counts are exactly 20 or 30 per session and timepoints per trial are exactly 180.

---

## Step 6: Script Development
**Status**: COMPLETE

Implementation notes:
- Wrote `/app/convert_data.py` with CLI:
  - `python -u /app/convert_data.py <outpicklefile>`
  - `--full`
  - `--sample`
  - `--show-processing`
- The script:
  - discovers sessions directly from `/app/data`
  - reconstructs the paper-consistent neural signal using Suite2p neuropil subtraction plus baseline preprocessing
  - aligns motion to the imaging timeline using timestamps
  - averages neural and motion traces in non-overlapping 10-frame bins
  - creates per-session motion-energy quintiles
  - splits sessions into 60-second trials
  - builds the required output dictionary
  - runs `verify_data_format()` before saving
- Smoke test run:
  - `python3 -u /app/convert_data.py /tmp/step6_smoke.pkl --sample`
  - completed successfully
  - processed 2 sessions in ~2.0 s total

Code inefficiencies identified:
- Initial timestamp-to-frame alignment used absolute time scaling and produced false missing-frame counts even in sessions where motion length matched imaging length.

Code speedups added:
- Fixed motion alignment to:
  - bypass timestamp remapping when motion length already equals imaging frame count
  - infer missing-frame indices from relative timestamp steps rather than from absolute timestamp scale
- Processing is session-wise and vectorized:
  - neural preprocessing uses Suite2p on whole session arrays
  - binning uses reshape+mean without Python loops
  - no redundant file I/O across stages

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 1,116 session-neuron entries across 2 sample sessions |
| Neurons / session | 370, 746 |
| Subjects | 2 (`jm032`, `jm039`) |
| Sessions / subject | 1, 1 |
| Trials (total) | 50 |
| Trials / session | 20, 30 |
| `time_from_session_start_sec` range | [0.15, 1799.82] across sample |
| `motion_energy_quintile` distribution | [0.200, 0.200, 0.200, 0.200, 0.200] overall |
| `motion_energy_quintile` distribution per session | `jm032`: [0.2,0.2,0.2,0.2,0.2]; `jm039`: [0.2,0.2,0.2,0.2,0.2] |

### Processing Plots Review
- Generated:
  - `/app/processing_jm032_2023-10-22_a.png`
  - `/app/processing_jm039_2024-04-30_a.png`
- Visual review findings:
  - Neural baseline-corrected traces show plausible calcium-like dynamics without flatlining or clipping.
  - Missing camera frames are explicitly visible and correctly repaired only in the `jm032_2023-10-22_a` session (148 missing frames), matching the raw data discrepancy.
  - Binned motion traces and quintile thresholds appear well aligned to the session timeline.
  - Trial boundaries occur at regular 60 s intervals and neural heatmaps show no discontinuities suggestive of off-by-one splitting errors.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |
| Vectorized reshape+mean binning | Removes Python-loop overhead during temporal averaging |
| Session-wise processing with no redundant loads | Keeps memory bounded and reduces I/O |
| Fixed direct mapping for sessions without dropped camera frames | Avoids unnecessary timestamp reconstruction |

| Step | Time / Session | Estimated Total Time |
| | | |
| 20 min session (no plotting) | ~0.34 s | 14 such sessions => ~4.8 s |
| 30 min session (no plotting) | ~1.02 s | 27 such sessions => ~27.5 s |
| Full conversion total (no plotting) | N/A | ~32.3 s (~0.54 min) |

Files created in Step 7:
- `/app/sample_data.pkl`
- `/app/conversion_sample_out.txt`
- `/app/verification_sample_out.txt`

Format validation result:
- Errors: none
- Warnings: none

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| `motion_energy_quintile` | 0.7096 | 0.2973 |

Additional notes:
- Chance level for `motion_energy_quintile` is `1/5 = 0.2000`.
- Validation accuracy is above chance on the sample dataset.
- Training loss decreased monotonically overall from `81.30` at epoch 1 to `0.87` at epoch 200.
- Test loss at the end of training: `5.9386`.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 414,482,599 bytes
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | ~3156 implied by `6 * 526` tracked neurons across mice (approximate paper summary) | N/A | 2998 unique tracked neurons across subjects | 2998 unique tracked neurons across subjects | Approximate / data exact |
| Mean neurons/session | Paper gives 526 tracked neurons per mouse on average | N/A | 498.66 | 498.66 | Close to paper summary; exact match to data |
| Subjects | 6 | N/A | 6 | 6 | Yes |
| Sessions | “minimum of 6 consecutive days” per mouse | N/A | 41 total sessions | 41 total sessions | Yes |
| Trials (total) | No native trials reported | N/A | 1090 (60 s trialization from raw frames) | 1090 | Yes |
| Trials/session (mean) | No native trials reported | N/A | 26.585 | 26.585 | Yes |
| `time_from_session_start_sec` range | Native sessions reported at 30 Hz; decoding uses time-varying behavior | N/A | [0.15, 1799.82] after 10-frame binning | [0.15, 1799.82] | Yes |
| `motion_energy_quintile` distribution | Paper decodes continuous motion, not categorical bins | N/A | [0.2,0.2,0.2,0.2,0.2] by construction per session | [0.2,0.2,0.2,0.2,0.2] | Yes |

Additional validation notes:
- `/app/verification_full_out.txt` reports:
  - data format valid
  - no errors
  - no warnings
- Converted session-wise neuron counts exactly match the raw data:
  - `221 x 7`, `370 x 7`, `685 x 7`, `746 x 7`, `541 x 6`, `435 x 7`
- Converted trial counts exactly match raw session durations:
  - 14 sessions x 20 trials
  - 27 sessions x 30 trials
- Missing motion-frame counts recovered by alignment exactly match raw length discrepancies:
  - unique values `[0, 1, 2, 3, 116, 148]`
- Spot checks:
  - Early 20 min session: `jm031_2023-10-18_a`
  - Missing-frame session: `jm032_2023-10-22_a`
  - Long 30 min session: `jm039_2024-04-30_a`
  - All showed expected neuron counts, trial counts, and aligned motion traces.

Known paper/data discrepancy retained:
- Methods text states each session lasted 20 minutes.
- Released data contain both 20-minute sessions (`36,000` frames) and 30-minute sessions (`54,000` frames).
- Conversion preserves the released data exactly rather than truncating longer sessions to 20 minutes.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `/app/verification_full_out.txt` contains no errors and no warnings.
2. **Neural sanity checks from raw data**:
   - Recomputed Suite2p-style baseline-corrected fluorescence directly from raw `F.npy` and `Fneu.npy` for:
     - `jm031_2023-10-18_a`, neuron 3, trial 5
     - `jm039_2024-05-04_a`, neuron 10, trial 12
   - Compared directly against `/app/converted_data.pkl` using `np.allclose(..., atol=1e-5)`.
   - Result: both checks passed exactly.
3. **Input sanity check from raw timing definition**:
   - Recomputed bin-center session times for `jm039_2024-04-30_a` directly from the 30 Hz frame grid and 10-frame averaging.
   - Compared the full session input vector to converted data using `np.allclose(..., atol=1e-6)`.
   - Result: passed exactly.
4. **Output sanity checks from raw motion files**:
   - Missing-frame session: `jm032_2023-10-22_a`
     - Rebuilt motion alignment directly from `motion_energy_glob.npy` and `tstamps.npy`
     - Recomputed session quintile bins
     - Compared full converted categorical output using `np.allclose()`
     - Result: passed exactly; recovered missing-frame count = 148, matching raw data and metadata.
   - No-missing session: `jm046_2024-09-06_a`
     - Rebuilt motion bins directly from raw motion file
     - Compared full converted categorical output
     - Result: passed exactly.
5. **Reference code comparison**:
   - **(a) Data loading**:
     - Reference: Track2p loads suite2p `F/Fneu/spks/stat/iscell/ops`.
     - Conversion: loads `F`, `Fneu`, `ops`, and `move_deve/*` from released tracked suite2p sessions.
     - Result: consistent with released data organization and Track2p output format.
   - **(b) Neuron/trial filtering**:
     - Reference: apply `iscell > 0.5`, then all-day Track2p matches.
     - Conversion: no new neuron filtering because the released suite2p folders already contain only tracked all-day cells.
     - Result: consistent with data README and paper.
   - **(c) Temporal alignment**:
     - Reference text: camera triggered by microscope acquisition; data README warns of occasional missing camera frames.
     - Conversion: imaging frames are the master clock; missing camera samples are inserted by timestamp-derived indexing and linearly interpolated.
     - Result: consistent and necessary for complete decoder targets.
   - **(d) Binning**:
     - Reference text: decode after averaging 10 consecutive timestamps.
     - Conversion: uses non-overlapping 10-frame means for both neural and motion streams.
     - Result: matches paper.
   - **(e) Input construction**:
     - Reference paper does not define decoder input as session time.
     - Conversion uses elapsed session time because it is explicitly required by the benchmark task.
     - Result: intentional task-driven difference.
   - **(f) Output construction**:
     - Reference paper decodes continuous motion with ridge regression and evaluates R².
     - Conversion discretizes motion into per-session quintiles because the benchmark requires categorical outputs.
     - Result: intentional task-driven difference.
6. **Key statistics comparison**:
   - Raw vs converted:
     - subjects: 6 vs 6
     - sessions: 41 vs 41
     - total 60 s trials: 1090 vs 1090
     - mean trials/session: 26.585 vs 26.585
     - mean neurons/session: 498.66 vs 498.66
     - missing motion-frame counts: `[0, 1, 2, 3, 116, 148]` recovered exactly
   - Paper vs data:
     - subject count and minimum days/session match
     - tracked-neuron mean is close (`paper ~526`, `released data 499.67`)
     - session duration statement differs (`paper text: 20 min`; released data: 20 and 30 min)
7. **Edge-case checks**:
   - Sessions with 0, 1, 2, 3, 116, and 148 missing camera frames all process successfully.
   - Missing motion frames are internal, not at the first or last imaging frame, so interpolation does not need endpoint extrapolation.
   - Sessions of both lengths (20 min and 30 min) produce exactly 20 or 30 trials, all with 180 time bins.
   - Subject `jm040` has 6 sessions while others have 7; conversion preserves this without assumptions of equal session count.

### Issues Found and Resolved
- **Issue**: Initial timestamp alignment during script development produced false missing-frame counts in sessions whose motion length already matched imaging length.
  - **Resolution**: bypass remapping when lengths already match, and for true missing-frame sessions derive frame indices from the full timestamp span rather than from median local steps.
- **Issue**: Initial sample-mode session selection did not exercise missing-frame behavior sessions.
  - **Resolution**: changed sample-mode selection to include the session with the largest motion-frame mismatch plus one long 30-minute session.
- **Final review result**: no unresolved issues remain after re-checks.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| `motion_energy_quintile` | 0.6137 | 0.3106 | Chance = 0.2000; test loss = 2.7591 |

Additional notes:
- Training used the GPU successfully (`device: cuda` in log).
- Loss decreased from `82.34` at epoch 1 to `1.10` at epoch 200.
- The script completed fully and produced sample plots (`sample_trials.png`, `predictions.png`).

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| `motion_energy_quintile` | Validation balanced accuracy = 0.3106; training balanced accuracy = 0.6137; chance = 0.2000 | Paper reports same-day motion decoding with cross-validated **R²** that increases with development, plus stable late cross-day decoding (Fig. 7). No directly comparable balanced-accuracy value is tabulated in extracted text. |

Analysis:
- **Chance comparison**:
  - Chance level is `0.2`.
  - Achieved validation balanced accuracy is `0.3106`.
  - This is `1.553x` chance, which clears the `1.5x chance` review threshold.
- **Paper comparison**:
  - The paper’s decoder predicts continuous motion using ridge regression and reports **R²**, not categorical balanced accuracy.
  - The current benchmark uses a different target representation:
    - categorical quintiles instead of continuous motion
    - fixed 60-second trials
    - elapsed session time included as an explicit decoder input
  - Therefore only qualitative comparison is defensible.
  - Qualitatively, our result is consistent with the paper’s conclusion that motion becomes decodable from tracked population activity.
- **Train vs validation gap investigation**:
  - Gap ratio: `0.6137 / 0.3106 = 1.98`, which triggered a deeper check.
  - Re-ran `train_validate_decoder()` and examined held-out performance session-by-session:
    - validation balanced accuracy was above chance in **41 / 41 sessions**
    - validation balanced accuracy exceeded `0.3` in **24 / 41 sessions**
    - per-session validation balanced accuracy ranged from `0.2307` to `0.5149`, mean `0.3115`
  - Interpretation:
    - the decoder generalizes above chance for every session
    - there is no evidence of a subset of catastrophically misaligned sessions
    - the train/validation gap is most plausibly ordinary model overfitting on a difficult five-class task, not data leakage or conversion failure
- **Additional low-accuracy debugging checks already completed**:
  - raw-data output verification for missing-frame and no-missing sessions: passed
  - temporal-alignment plots for sample sessions: visually consistent
  - output class balance: exactly 20% per class by session
  - neuron filtering and signal preprocessing: consistent with the paper’s stated Suite2p-based dF/F workflow

### Issues Found and Resolved
- **Issue reviewed**: train/validation gap greater than `1.5x`.
  - **Resolution**: investigated with session-level held-out accuracies and raw-data sanity checks; found no evidence of leakage or conversion bugs, so no code change was warranted.
- **Issue reviewed**: direct paper-vs-benchmark numeric accuracy comparison.
  - **Resolution**: documented that the paper reports regression `R²` for a different target format, so only qualitative comparison is defensible.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

Created during cleanup:
- `/app/README.md`
- `/app/cache/README_CACHE.md`
- `/app/cache/step10_sanity_checks.py`
- `/app/cache/step12_gap_analysis.py`

Directory organization notes:
- Conversion, verification, and training logs remain in `/app/` for direct inspection.
- Reproducibility and investigation helpers are stored in `/app/cache/`.
