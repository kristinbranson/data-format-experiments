# Dataset Conversion Notes

## Overview
- **Dataset**: longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p
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

Setup checks:
- Verified `/app/CONVERSION_NOTES.md` exists with `ls -la /app/CONVERSION_NOTES.md`
- Verified `python3` works and imports `numpy 2.4.4` and `torch 2.6.0+cu124`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `run_t2p(track_ops)` | `code/track2p/t2p.py` | LOADING / PROCESSING | Main Track2p pipeline: checks planes, loads Suite2p images, registers sessions, matches ROIs across sessions, saves outputs. |
| `check_nplanes(track_ops)` | `code/track2p/io/s2p_loaders.py` | LOADING | Verifies all datasets have the same number of planes. |
| `load_all_imgs(track_ops)` | `code/track2p/io/s2p_loaders.py` | LOADING | Loads `ops.npy` mean images and channel metadata for each session and plane. |
| `load_stat_ds_plane(track_ops_path, track_ops, plane_idx=0)` | `code/track2p/io/loaders.py` | CURATION | Loads `stat.npy` and filters ROIs by Suite2p `iscell` using `track_ops.iscell_thr`. |
| `load_all_ds_stat_iscell(track_ops)` | `code/track2p/io/s2p_loaders.py` | CURATION | Applies the same `iscell` filter across all sessions and planes. |
| `generate_suite2p_indices(track_ops)` | `code/track2p/t2p.py` | PROCESSING | Converts Track2p match-matrix indices back to original Suite2p ROI indices after `iscell` filtering. |
| `save_in_s2p_format(track_ops)` | `code/track2p/t2p.py` | PROCESSING | Builds `matched_suite2p` outputs by applying `iscell` filtering, then indexing traces/statistics by all-day tracked-cell rows. |
| `DataManagement.import_files(...)` | `code/track2p/gui/data_management.py` | LOADING / CURATION | GUI downstream loader: loads `plane#_match_mat.npy`, keeps rows with no `None`, loads `F` or `spks`, applies `iscell` filter, then tracked-cell indexing. |
| `F_processing(F, Fneu, fs, ...)` | `code/track2p/gui/data_management.py` | PROCESSING | Optional fluorescence preprocessing for GUI "dF/F0" mode using neuropil subtraction and baseline removal. |
| `npy_to_s2p(track_ops)` | `code/track2p/io/savers.py` | LOADING / PROCESSING | Converts simple `.npy` inputs into a minimal Suite2p-like structure with `F.npy`, `ops.npy`, `stat.npy`, `iscell.npy`. |

### Notes
- `code/README.md` describes Track2p as a longitudinal cell-tracking package operating on datasets that contain Suite2p outputs.
- The core reference code is about registration and matching of ROIs across longitudinal sessions, not about behavioral decoding directly.
- Default curation rule in `code/track2p/ops/default.py` is `iscell_thr = 0.50`; most loaders use:
  - if `iscell_thr is None`: keep `iscell[:, 0] == 1`
  - else: keep `iscell[:, 1] > iscell_thr`
- Downstream analysis pattern is consistent in demo notebook and GUI:
  1. load `plane#_match_mat.npy`
  2. keep only rows without `None` to obtain cells present on all days (`t2p_match_mat_allday`)
  3. load Suite2p arrays (`ops.npy`, `stat.npy`, `F.npy`, optionally `spks.npy`, `Fneu.npy`)
  4. apply the same `iscell` filtering used during tracking
  5. index filtered arrays with `t2p_match_mat_allday[:, i]`
- The repository does not compute a canonical `dF/F` as part of the main tracking pipeline.
- Optional GUI preprocessing labeled `"dF/F0"` uses `F_processing`, which does:
  - neuropil subtraction with `neucoeff=0.0` by default
  - baseline estimation via gaussian smoothing and maximin filtering
  - subtraction of baseline (`F = Fc - Flow`)
  - importantly, it does not divide by baseline, so it is not standard `dF/F`
- The demo notebook `code/notebooks/demo_t2p_ouputs.ipynb` explicitly recommends using the same `iscell` threshold as Track2p and shows loading matched fluorescence traces from `F.npy`.
- Reference code suggests the key longitudinal curation decision is whether to require all-day matched cells (`~np.any(match_mat == None, axis=1)`), which is likely relevant for constructing sessions with comparable neuron identities across days.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data/README.md` explains the data layout and states that the provided `suite2p` folders already contain Track2p outputs saved in Suite2p format, with only cells present across all days for a subject.
- Top level contents:
  - `README.md`
  - `load_data.ipynb`
  - subject folders: `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`
- Subject folders contain daily session folders named like `YYYY-MM-DD_a`.
- Each session contains:
  - `suite2p/plane0/`
    - `F.npy`: fluorescence traces, shape `(n_neurons, n_frames)`, `float32`
    - `Fneu.npy`: neuropil traces, shape `(n_neurons, n_frames)`, `float32`
    - `spks.npy`: Suite2p deconvolved activity, shape `(n_neurons, n_frames)`, `float32`
    - `iscell.npy`: shape `(n_neurons, 2)`; first column is all `1.0` in provided data, second column stores Suite2p cell probability
    - `stat.npy`: object array of ROI dictionaries with keys including `xpix`, `ypix`, `med`
    - `ops.npy`: Suite2p metadata; all sessions have `fs=30`, `meanImg` shape `(512, 512)`, and `nframes` matching the imaging frame count
  - `move_deve/`
    - `motion_energy_glob.npy`: processed global motion energy from video, `uint64`
    - `tstamps.npy`: timestamps for available video frames, `float64`
    - `interframe_int.npy`: timestamp differences, `float64`
- Additional subject-level files:
  - `ground_truth.csv` exists for `jm038`, `jm039`, and `jm046`; these appear to be tracking-evaluation annotations rather than required decoder inputs/outputs.
- Important native-data implication:
  - there is no explicit trial structure in the source data
  - sessions are continuous recordings that will need to be split into 60-second windows for the decoder task
- Important curation implication from the provided files:
  - neuron identities are already matched across all days within a mouse
  - within each subject, every session has the same number of neurons
  - this means the released data are already a curated subset of all detected ROIs

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 2998 unique tracked neurons across subjects; 20445 neuron-session rows across all sessions |
| Neurons / session | subject-specific constant counts: jm031=221, jm032=370, jm038=685, jm039=746, jm040=541, jm046=435 |
| Subjects | 6 |
| Sessions / subject | jm031=7, jm032=7, jm038=7, jm039=7, jm040=6, jm046=7 (41 total sessions) |
| Trials (total) | None natively; source data are continuous sessions. If split into 60-second windows using imaging duration, there are 1090 windows total |
| Trials / session | None natively; 20 windows for 20-minute sessions (36000 frames at 30 Hz) and 30 windows for 30-minute sessions (54000 frames at 30 Hz) |

Additional quantitative observations:
- All imaging sessions are single-plane (`plane0`) and sampled at 30 Hz.
- Session lengths by imaging frames:
  - 36000 frames for `jm031`, `jm032` sessions (20 minutes)
  - 54000 frames for `jm038`, `jm039`, `jm040`, `jm046` sessions (30 minutes)
- Motion-energy length sometimes differs from imaging frame count because of missing video frames.
  - 9/41 sessions have mismatches.
  - mismatch sizes found: 1, 2, 3, 116, 148 frames.
  - affected sessions: `jm031` on 2023-10-20/21/22, `jm032` on 2023-10-20/21/22, `jm039` on 2024-05-04, `jm040` on 2024-05-04, `jm046` on 2024-09-09.
- `tstamps.npy` length always matches `motion_energy_glob.npy`, and `interframe_int.npy` is one sample shorter.
- `interframe_int.npy` median values are around `3.36e-05`; combined with `fs=30`, these timestamps appear to be recorded in kiloseconds rather than seconds.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Not stated as a global total across all sessions; average tracked neurons per mouse is 526 ± 190 SD | `"On average 526 (± 190 std) neurons per mouse"` |
| Neurons / session | Not explicitly stated; all subsequent analyses use cells tracked across all days | `"neurons per mouse were successfully tracked across all days"` |
| Subjects | 6 mice in the full longitudinal dataset | `"we used a full dataset of 6 mice"` |
| Sessions / subject | Minimum 6 consecutive days; example mouse has 7 daily sessions P8-P14 | `"minimum of 6 consecutive days"` / `"n=7 imaging sessions from one mouse"` |
| Trials (total) | No native trial structure described; recordings are continuous sessions | `"each session lasted 20 minutes"` |
| Trials / session | Not applicable in source; decoder-relevant cross-validation used 2-minute blocks | `"splits were done on consecutive 2 minute blocks"` |
| Neural data time bin | 30 Hz native imaging; 10 consecutive timestamps averaged for decoding | `"Imaging rate was 30 Hz"` / `"averaging in bins of 10 consecutive timestamps"` |
| Behavior data time bin | 30 Hz video; 10 consecutive timestamps averaged for decoding | `"Videos were recorded at 30 Hz"` / `"averaging in bins of 10 consecutive timestamps"` |
| Reward rate | N/A for this spontaneous behavior dataset | No reward paradigm described |
| Motion variable | Global motion energy from squared consecutive-frame differences summed over pixels | `"pixelwise difference"` / `"squared"` / `"summed across pixels"` |
| Cell inclusion rule | Keep ROIs above Suite2p default cell threshold 0.5 | `"above the default threshold of 0.5"` |
| Traces used for analyses | Baseline-corrected fluorescence traces treated as dF/F | `"baseline corrected fluorescence traces as our dF/F"` |
| Decoder target in paper | Mouse motion / behavioural state proxy from video-derived motion energy | `"predict a behavioural variable (y, mouse motion)"` |
| Decoder performance expectation | Same-day decoding improves with development; later cross-day decoding is accurate/stable | `"same-day decoding performance increased with development"` / `"allowing for accurate cross-day decoding"` |


### Processing Details
- Experimental context:
  - longitudinal 2-photon imaging in mouse barrel cortex during the second postnatal week
  - recordings are daily within roughly P7-P14
  - imaging is in layer 2/3, single FOV, 512x512 pixels
- Temporal alignment:
  - videography recorded at 30 Hz
  - microscope acquisition triggered camera frames, enabling straightforward synchronization
- Neural preprocessing described in text:
  - Suite2p used separately on each recording for motion correction, ROI detection, signal extraction, and spike deconvolution
  - authors considered ROIs with Suite2p cell probability above 0.5 as true cells
  - for downstream analyses they used baseline-corrected fluorescence traces as dF/F
- Behaviour preprocessing described in text:
  - motion energy is computed from consecutive video frames
  - per timepoint metric is sum of squared pixelwise frame differences
- Decoding preprocessing described in text:
  - neural and behavior traces were slightly denoised by averaging in bins of 10 timestamps
  - same-day decoding used nested cross-validation with 5 outer and 5 inner folds
  - folds were defined on consecutive 2-minute recording blocks
  - cross-day decoding fit on one day and evaluated on all other days

### Curation Steps

**Neuron curation rules**:
- Text explicitly states use of Suite2p cell-probability threshold 0.5.
- Track2p then identifies cells tracked across all days; all subsequent longitudinal analyses use that tracked population.

**Trial curation rules**:
- No native trial structure exists in the source recordings.
- For the paper’s decoding analysis, evaluation blocks are 2-minute consecutive chunks rather than experimenter-defined trials.
- The data README warns that some video frames are missing in some recordings and should be treated as missing values or interpolated using `tstamps.npy` / `interframe_int.npy`.

### Decoders Trained
| Decoded variable | Accuracy |
| Mouse motion / motion energy | Reported qualitatively as increasing across development for same-day decoding, with accurate/stable late cross-day decoding; figure captions report performance using `R2`, but exact numeric values are not available in extracted text |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Tracked-cell indexing | Reference Track2p code expects raw Suite2p outputs plus `plane#_match_mat.npy`; downstream loaders filter by `iscell` then index matched rows | Released `suite2p` folders already have constant neuron counts within each mouse and README says they are saved Track2p outputs with only cells present across all days | Longitudinal analyses use neurons tracked across days | Treat released `suite2p` arrays as already matched/across-day-curated outputs; do not rerun Track2p or apply an additional match matrix |
| `iscell` curation | Reference code uses default `iscell_thr=0.5` and filters ROIs with `iscell[:,1] > 0.5` | Provided `iscell[:,0]` values are all `1.0` and probabilities are already above ~0.50025 | Paper states ROIs above default threshold `0.5` were considered true cells | No extra curation change is needed; the released data already embody the paper’s `iscell` filtering |
| Neural trace type for downstream analysis | Track2p repo mostly demonstrates loading `F.npy` or `spks.npy`; GUI "dF/F0" helper only subtracts baseline and is not canonical dF/F | Released data contain `F.npy`, `Fneu.npy`, and `spks.npy` for matched cells | Paper states analyses used baseline-corrected fluorescence traces as dF/F, with decoding done on denoised dF/F | For decoder conversion, recompute baseline-corrected fluorescence from `F` and `Fneu` using Suite2p-style defaults from `ops.npy`, rather than using raw `F` or the GUI helper |
| Session duration | Code does not constrain session duration | Imaging files show two native durations: 36000 frames (20 min) and 54000 frames (30 min) | Methods text says each session lasted 20 minutes | Use actual session lengths in the released data; document this as a paper-vs-release discrepancy and preserve full available recordings |
| Behaviour alignment | Code repo itself does not include motion-energy alignment logic | `motion_energy_glob.npy` is per video frame; 9 sessions have fewer behavior samples than imaging frames; timestamps and interframe intervals expose dropped camera frames | Paper says video was triggered by the microscope for simple synchronization | Align behavior to imaging frames at 30 Hz and repair missing video frames using timestamp-informed interpolation so output remains frame-aligned with neural data |
| Trial structure | Track2p code uses continuous recordings; paper decoding uses 2-minute CV blocks | Data are continuous sessions with no trial boundaries | Paper decoding uses continuous recordings and blockwise CV, not explicit trials | For this benchmark, impose 60-second non-overlapping trials as required by the user, while preserving within-session temporal order and 30 Hz alignment |
| Dataset scale | Code is agnostic | 6 subjects, 41 sessions, per-subject tracked neurons = [221, 370, 685, 746, 541, 435], mean 499.7, SD 180.5 | 6 mice, mean tracked neurons per mouse 526 ± 190 SD | Treat dataset size as broadly consistent with paper; slight neuron-count difference is acceptable and likely reflects release/version differences rather than a loading bug |

Final consistency understanding:
- The released dataset is not the raw input to Track2p; it is already the post-Track2p matched-cell output saved in Suite2p layout.
- The paper’s functional analyses and decoding rely on baseline-corrected fluorescence-like traces and synchronized motion energy, so conversion should reconstruct that analysis representation rather than rely on raw `F.npy`.
- The key data-cleaning problem unique to the release is missing camera frames in some sessions; this must be handled explicitly during behavioral alignment.
- The user-required 60-second trials are a downstream reformatting step layered on top of the paper’s continuous-session analysis.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `suite2p/plane0/F.npy` + `Fneu.npy` + `ops.npy` | `neural` | Neuropil subtraction with `ops['neucoeff']`, then Suite2p baseline correction using `baseline`, `win_baseline`, `sig_baseline`, `prctile_baseline`, then average in non-overlapping 10-frame bins, then split into 60 s trials | Paper methods for Suite2p dF/F and decoding; `suite2p.extraction.dcnv.preprocess`; Track2p code only provides loading conventions | Chosen to match paper’s decoding traces rather than raw `F.npy` or GUI helper |
| Session frame index / `ops['fs']` | `input[0]` | Construct absolute time-from-session-start in seconds at imaging-frame centers, average in the same 10-frame bins, preserve absolute time across trials | Paper says synchronized 30 Hz imaging/video; decoder task requires time elapsed from session start | Trial 1 starts near 0.15 s when using 10-frame-bin centers; later trials continue from 60 s, 120 s, etc. |
| `move_deve/motion_energy_glob.npy` + `tstamps.npy` + `interframe_int.npy` | `output[0]` | Convert timestamps to seconds, interpolate motion energy onto full imaging frame times when camera drops frames, average in non-overlapping 10-frame bins, discretize within session into 5 equal-percentile bins, then split into 60 s trials | Paper methods for motion metric + synchronization + 10-timestamp decoding denoising | Output is categorical and time-varying as required by the decoder task |
| Subject folder name (e.g. `jm031`) | `subjects`, `subject_idx` | Unique subject lookup table and per-session integer index | Data organization in `data/README.md` | Session order will be sorted by subject, then date |
| Barrel cortex recording site | `brain_regions`, `brain_region_idx` | Single region label for all neurons; region index array of zeros per session | Paper methods | All recordings are from barrel cortex layer 2/3 |
| Session directory name (date) | `metadata['session_info']` | Store per-session identifiers/dates/original sizes | N/A | Useful for spot-checks and reproducibility |

### Key Decisions
1. **Use baseline-corrected fluorescence rather than raw `F` or `spks`**: The paper explicitly states that subsequent analyses, including decoding, used baseline-corrected fluorescence traces as dF/F with Suite2p defaults. The released data include `F`, `Fneu`, and the required Suite2p parameters in `ops.npy`, so this representation can be reconstructed directly.
2. **Use Suite2p-style preprocessing instead of the Track2p GUI helper**: The GUI helper labeled `dF/F0` subtracts a baseline but is not the canonical analysis path in the paper. The paper’s reference processing is closer to `Fc = F - neucoeff * Fneu` followed by Suite2p baseline correction.
3. **Keep the released matched-cell set as-is**: The data are already saved in matched-Suite2p form, so rerunning Track2p matching or filtering to all-day matched rows again would be redundant and risks indexing errors.
4. **Repair missing video frames by interpolation onto imaging frame times**: The README explicitly says missing frames can be interpolated, and the decoder requires a value at every neural time bin. Interpolation preserves all neural samples and maintains synchronization.
5. **Bin neural and behavioral traces in non-overlapping 10-frame windows before trialing**: The paper’s decoding denoises both streams by averaging 10 consecutive timestamps. With 30 Hz acquisition, this gives 3 Hz data and a 333.333 ms time bin that exactly tiles both 20-minute and 30-minute sessions into 60-second trials.
6. **Split sessions into non-overlapping 60-second trials after alignment and binning**: This satisfies the decoder-task requirement while preserving the paper’s continuous-session temporal processing as much as possible.
7. **Use absolute time-from-session-start as the decoder input**: The user specifically requested time elapsed from the beginning of the session. Therefore trial-local time reset is not appropriate.
8. **Discretize motion energy per session into five equal-percentile classes using rank-based equal-frequency assignment**: This guarantees near-balanced class counts even when motion energy has repeated low values or long immobile periods, which is important because the paper decodes a continuous variable but the benchmark requires categorical outputs.
9. **Use the full available session duration from data, even for 30-minute sessions**: This follows the released data rather than truncating to the 20-minute duration stated in methods, avoiding arbitrary data loss.
10. **Keep all 60-second windows that have valid neural data**: Because frame counts are exact multiples of 10 frames and 60 seconds, no partial trailing windows are expected after 10-frame binning.

### Planned Sanity Checks
- [ ] Neural preprocessing check: for one session and one neuron, compare the stored converted values against a direct recomputation from raw `F`, `Fneu`, `ops`, 10-frame averaging, and trial slicing using `np.allclose()`.
- [ ] Input-time check: verify that the first two converted trials contain the expected absolute time centers and that the final value matches the expected session duration after 10-frame binning.
- [ ] Output-alignment check: for a session with missing camera frames, compare converted motion-energy bins against a direct raw-data interpolation plus binning pipeline using `np.allclose()` before discretization.
- [ ] Output-discretization check: verify per-session class counts are close to 20% each and that bin labels are monotonic with the underlying continuous motion values.
- [ ] Dataset-size check: verify 6 subjects, 41 sessions, per-subject constant neuron counts, and expected 20/30 converted trials per session.
- [ ] Reference-statistics check: verify mean tracked neurons per mouse from converted session metadata remains close to the paper’s reported 526 ± 190.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with:
- CLI: `python -u /app/convert_data.py <outpicklefile> [--full|--sample] [--show-processing]`
- Session discovery from `/app/data`
- Paper-style neural preprocessing:
  - load matched-cell `F.npy`, `Fneu.npy`, `ops.npy`
  - neuropil subtraction using `ops['neucoeff']`
  - baseline correction via `suite2p.extraction.dcnv.preprocess` on CPU using session-specific Suite2p parameters
- Behaviour preprocessing:
  - load `motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy`
  - infer timestamp units, convert to seconds
  - interpolate motion energy onto full imaging frame times to repair dropped camera frames
- Temporal processing:
  - non-overlapping averaging in 10-frame bins for neural, motion, and time input
  - non-overlapping 60-second trial splitting after binning
- Output construction:
  - per-session 5-way equal-frequency motion-energy discretization
  - categorical integer output arrays
- Metadata/session summaries:
  - subject/session ids, neuron counts, trial counts, missing-frame counts, session quantiles
- Optional `--show-processing` plots for up to 2 sessions
- Timing printouts and ETA updates during conversion

Sanity checks already embedded in code:
- checks that trial counts match across neural/input/output
- checks that time dimensions are divisible by the chosen bin and trial sizes
- preserves exact session ordering for reproducibility
- stores per-session summaries in metadata for later verification

Code inefficiencies identified:
- Full-session baseline correction is the main expected bottleneck.

Code speedups added:
- Session-wise streaming processing keeps memory bounded.
- Non-overlapping temporal averaging uses reshape/mean vectorization rather than Python loops.
- Motion alignment uses vectorized `np.interp`.
- Plots are restricted to at most 2 sessions.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 442 neuron-session rows across 2 sample sessions |
| Neurons / session | 221, 221 |
| Subjects | 1 (`jm031`) |
| Sessions / subject | 2 |
| Trials (total) | 40 |
| Trials / session | 20, 20 |
| `time_from_session_start_s` range | [0.15, 1199.82] |
| Trial length (time bins) | 180 |
| `motion_energy_quintile` distribution | [0.200, 0.200, 0.200, 0.200, 0.200] |

### Processing Plots Review
- Generated:
  - `/app/processing_jm031_2023-10-18_a.png`
  - `/app/processing_jm031_2023-10-20_a.png`
- The two sample sessions were intentionally chosen to cover:
  - one session with no dropped video frames
  - one session with a 2-frame motion-data mismatch repaired by interpolation
- Automated checks from the converted arrays and validator output show:
  - exact 60-second trial tiling (20 trials/session)
  - exact temporal shape agreement across neural/input/output (`180` binned timepoints per trial)
  - perfectly balanced quintile output distribution as intended
  - no NaNs/Infs and no format warnings
- No anomalies were detected in the sample conversion logs.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |
| Vectorized 10-frame averaging with reshape/mean | Removes Python-loop overhead from neural, input, and output binning |
| Session-wise processing | Keeps memory bounded and avoids large temporary whole-dataset arrays |
| Vectorized `np.interp` for motion alignment | Fast handling of dropped video frames |
| Plot cap of 2 sessions | Prevents diagnostic plotting from dominating runtime |

| Step | Time / Session | Estimated Total Time |
| | | |
| Sample conversion run (`--sample --show-processing`) | ~0.82 s/sample session for two 221-neuron, 20-minute sessions | 1.64 s observed for 2 sessions |
| Full conversion estimate (scaled by neuron-frames) | ~0.103 s per million neuron-frames | ~107 s (~1.8 min) for all 41 sessions |

Files created:
- `/app/sample_data.pkl`
- `/app/conversion_sample_out.txt`
- `/app/verification_sample_out.txt`

Validator inspection summary from `/app/verification_sample_out.txt`:
- No errors
- No warnings
- Input dimension: 1
- Output dimension: 1
- Brain regions: 1
- Per-trial shapes:
  - neural `(221, 180)`
  - input `(1, 180)`
  - output `(1, 180)`

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| `motion_energy_quintile` | 0.5516 | 0.2729 |

Training review:
- Decoder training completed without errors on CUDA.
- Loss decreased monotonically from `39.56` at epoch 1 to `1.22` at epoch 200.
- Validation balanced accuracy exceeded the 5-class chance level of `0.20`.

Files created:
- `/app/train_decoder_sample_out.txt`

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 396M
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | Not explicitly totaled; 526 ± 190 tracked neurons per mouse | N/A | 2998 unique tracked neurons / 20445 neuron-session rows | 2998 unique tracked neurons represented across 20445 neuron-session rows | Yes |
| Mean neurons/session | Not explicitly stated | N/A | 498.66 | 498.66 | Yes |
| Subjects | 6 | N/A | 6 | 6 | Yes |
| Sessions | At least 6 consecutive days per mouse | N/A | 41 | 41 | Yes |
| Trials (total) | Continuous recordings; no native trials | Continuous recordings | Continuous recordings; 1090 derived 60-second windows from actual durations | 1090 | Yes |
| Trials/session (mean) | N/A | N/A | 26.59 derived trials/session | 26.59 | Yes |
| `time_from_session_start_s` range | 20-minute sessions stated in methods, but release contains both 20-minute and 30-minute sessions | N/A | [0.15, 1799.82] after 10-frame binning of actual data | [0.15, 1799.82] | Yes vs release data |
| `motion_energy_quintile` distribution | Paper decodes continuous motion variable (`R2`), not categorical bins | N/A | Continuous motion energy | [0.200, 0.200, 0.200, 0.200, 0.200] by construction per session | Task-defined |

Verification summary from `/app/verification_full_out.txt`:
- No errors
- No warnings
- 41 sessions
- 1090 total trials
- 1 input dimension
- 1 output dimension
- All trials have exactly 180 time bins
- Output classes are exactly balanced at the session level by design

Spot checks:
- Session 0 (`jm031_2023-10-18_a`): 36000 frames -> 20 trials, final input time `1199.8167 s`
- Session 14 (`jm038_2023-04-30_a`): 54000 frames -> 30 trials, final input time `1799.8167 s`
- Session 21 (`jm039_2024-04-30_a`): 54000 frames -> 30 trials
- Session 28 (`jm040_2024-05-01_a`): 54000 frames -> 30 trials
- Session 34 (`jm046_2024-09-03_a`): 54000 frames -> 30 trials

Consistency notes:
- The converted data preserve all 41 released sessions and all matched neurons provided by the release.
- The paper methods text says sessions lasted 20 minutes, but the released data clearly contain both 20-minute and 30-minute sessions; conversion follows the released data rather than truncating recordings.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Reviewed `/app/verification_full_out.txt`. Result: no errors and no warnings. The validator reports 41 sessions, 1090 trials, uniform trial length 180, 1 input dimension, and 1 output dimension.
2. **Raw-data sanity checks using `np.allclose()`**:
   - Neural:
     - Session 0 (`jm031_2023-10-18_a`), session 2 (`jm031_2023-10-20_a`, missing-frame case), and session 14 (`jm038_2023-04-30_a`, 30-minute case)
     - For each, recomputed from raw `F.npy`, `Fneu.npy`, `ops.npy`:
       - neuropil subtraction with `neucoeff`
       - Suite2p baseline correction via `suite2p.extraction.dcnv.preprocess`
       - 10-frame averaging
       - trial slicing
     - Compared one example neuron/trial against `converted_data.pkl`
     - Result: all `np.allclose(..., atol=1e-5)` checks passed exactly; max absolute difference `0.0`
   - Input:
     - For all 41 sessions, reconstructed time-from-session-start directly from raw frame indices and `fs=30`
     - Compared first and last converted trials per session
     - Result: all checks passed
   - Output:
     - For sessions 0, 2, 14, 25, and 40, recomputed from raw `motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy`:
       - timestamp scaling to seconds
       - interpolation to full imaging frames
       - 10-frame averaging
       - equal-frequency 5-bin rank discretization
       - trial slicing
     - Result: all checks passed exactly
3. **Reference code comparison**:
   - Data loading:
     - Reference downstream Track2p logic loads matched Suite2p-format arrays (`F`, `Fneu`, `spks`, `ops`, `stat`, `iscell`)
     - Converted script loads the same file types from the released matched-session folders
   - Neuron filtering:
     - Reference code uses `iscell > 0.5` before tracked indexing
     - Released data already satisfy this and contain only tracked cells
     - Converted script therefore does not apply further neuron dropping
   - Temporal alignment:
     - Paper: video triggered by microscope at 30 Hz
     - Released data: occasional missing video frames exposed by timestamps
     - Converted script aligns motion to imaging frame times and interpolates missing values
   - Binning:
     - Paper decoding averages neural and behavior data in bins of 10 timestamps
     - Converted script does the same before trialing
   - Input construction:
     - Not present in reference because it is benchmark-specific
     - Converted script uses absolute session time as explicitly requested by the benchmark
   - Output construction:
     - Paper decodes continuous motion energy with ridge regression
     - Converted script keeps the same aligned/binned motion signal, then discretizes into 5 within-session classes because the benchmark requires categorical outputs
4. **Key statistics comparison**:
   - Subjects: 6 in paper/release/converted
   - Sessions: release has 41; converted preserves all 41
   - Per-subject tracked neurons from converted data: `[221, 370, 685, 746, 541, 435]`
   - Mean tracked neurons per mouse: `499.67`; sample SD: `197.68`
   - Paper reports `526 ± 190 std`, so converted data are close to the reported scale
   - Total converted trials: `1090`, matching the release-derived session durations exactly
5. **Edge-case checks**:
   - Verified all converted trials have length `180`
   - Verified neural/input/output time dimensions always match
   - Verified `subject_idx` ordering matches sorted subject/session traversal
   - Verified sessions with motion-frame mismatches (9 sessions) still produce exactly aligned outputs after interpolation
   - Verified first input time is `0.15 s` and final time matches session duration minus one 10-frame-bin half-width (`1199.8167 s` for 20-minute sessions, `1799.8167 s` for 30-minute sessions)

### Issues Found and Resolved
- **Documentation issue**: Initial notes incorrectly grouped `jm038` with 20-minute sessions. Raw data review showed all `jm038` sessions are 54000 frames (30 minutes). Updated Step 2, Step 7 estimate context, and Step 9 statistics accordingly.
- **No conversion-code bugs found in Step 10**: Raw-data sanity checks, validator output, and edge-case checks all passed.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| `motion_energy_quintile` | 0.5901 | 0.2569 | Chance = 0.2000; trained on CUDA; 872 train trials / 218 validation trials |

Training details from `/app/train_decoder_full_out.txt`:
- Device used: CUDA
- Loss trajectory:
  - Epoch 1: `80.8151`
  - Epoch 10: `41.3770`
  - Epoch 50: `7.4396`
  - Epoch 100: `1.7656`
  - Epoch 200: `1.1743`
- Test loss: `3.0560`

Files created:
- `/app/train_decoder_full_out.txt`
- `sample_trials.png`
- `predictions.png`

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| | |
| `motion_energy_quintile` | Validation balanced accuracy `0.2569`; chance `0.2000`; ratio to chance `1.2845x` | Paper reports continuous motion decoding using `R2`, with qualitative statements that same-day decoding increases with development and late cross-day decoding is accurate/stable |

Analysis:
- The benchmark output is a 5-class, per-session, equal-frequency categorical variable, whereas the paper decodes a continuous motion signal and reports `R2`. Because both the target representation and metric differ, a direct numeric comparison is not possible.
- The achieved validation accuracy is above chance on both sample (`0.2729`) and full (`0.2569`) datasets.
- The `1.2845x chance` result triggered additional debugging because it is below the heuristic `1.5x` threshold.

Additional Step 12 investigations:
1. **Accuracy vs chance**
   - Chance level for 5 classes: `0.2000`
   - Full validation balanced accuracy: `0.2569`
   - Conclusion: above chance, but modest; investigate thoroughly
2. **Output correctness checks on specific trials**
   - Revalidated outputs directly from raw motion files for sessions 0, 2, 14, 25, and 40
   - Included a session with missing video frames and a 30-minute session
   - Result: reconstructed labels matched converted labels exactly
3. **Temporal alignment checks**
   - Rebuilt motion alignment from raw timestamps and imaging frame clocks
   - Confirmed exact equality of converted outputs after interpolation, 10-frame binning, discretization, and trial slicing
   - Confirmed exact equality of neural preprocessing and input timing on spot-checked sessions
4. **Output variation / class balance**
   - Each session has exactly balanced output fractions `[0.2, 0.2, 0.2, 0.2, 0.2]`
   - Therefore low performance is not due to a collapsed or nearly constant target
5. **Filtering / representation comparison**
   - Tested an alternative full-dataset conversion using `spks.npy` as the neural representation while keeping the same aligned inputs/outputs and decoder settings
   - Result:
     - `spks` training balanced accuracy: `0.5433`
     - `spks` validation balanced accuracy: `0.2526`
   - This is slightly worse than the current Suite2p-style baseline-corrected fluorescence representation (`0.5901` train / `0.2569` validation)
   - Conclusion: the current neural representation is at least as good as the main obvious alternative and remains more faithful to the paper
6. **Train vs validation gap**
   - Current representation: train `0.5901`, validation `0.2569` (ratio `2.30x`)
   - `spks` alternative: train `0.5433`, validation `0.2526` (ratio `2.15x`)
   - Similar gap across representations argues against a conversion-specific label/alignment bug and is more consistent with model overfitting / task difficulty
   - No evidence of leakage was found: labels were reconstructed from raw data exactly, time alignment checks passed, and subject/session indexing is correct

Paper comparison caveat:
- The paper’s figure text does not provide extractable numeric `R2` values from the PDF text; only qualitative conclusions and significance tests are available from the text extraction.
- Because the benchmark requires categorical outputs and uses balanced accuracy, I cannot make a one-to-one numerical comparison to the paper’s reported continuous-motion `R2`.
- The best available comparison is qualitative:
  - paper: motion becomes more decodable later in development and stable across days
  - converted benchmark data: motion is decodable above chance, and accuracy survives stricter categorical reformulation of the task

### Issues Found and Resolved
- **Potential concern investigated**: Validation accuracy was above chance but below the `1.5x chance` heuristic.
  - **Resolution**: Performed raw-label reconstruction checks, temporal-alignment checks, class-balance checks, and alternative-neural-representation benchmarking.
  - **Outcome**: No conversion bug identified; retained current paper-matched preprocessing.
- **No additional code changes were required in Step 12**.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

Created:
- `/app/README.md`
- `/app/cache/README_CACHE.md`
- `/app/cache/review_checks.py`

Moved diagnostic artifacts to `/app/cache/`:
- `processing_jm031_2023-10-18_a.png`
- `processing_jm031_2023-10-20_a.png`
- `sample_trials.png`
- `predictions.png`

Top-level required deliverables present:
- `/app/CONVERSION_NOTES.md`
- `/app/convert_data.py`
- `/app/converted_data.pkl`
- `/app/sample_data.pkl`
- `/app/README.md`
- `/app/train_decoder_full_out.txt`
- `/app/conversion_sample_out.txt`
- `/app/verification_sample_out.txt`
- `/app/train_decoder_sample_out.txt`
- `/app/conversion_full_out.txt`
- `/app/verification_full_out.txt`

Cache verification:
- Ran `/app/cache/review_checks.py`
- Result: raw-data neural/input/output sanity checks all passed
