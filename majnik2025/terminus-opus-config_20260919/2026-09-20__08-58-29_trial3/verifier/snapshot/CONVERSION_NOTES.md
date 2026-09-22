# Dataset Conversion Notes

## Overview
- **Dataset**: Majnik et al. 2025 eLife (Track2p), longitudinal 2p calcium imaging of mouse barrel cortex L2/3 during the 2nd postnatal week, with simultaneous videography-derived motion energy.
- **Date started**: 2025 (container date Sep 20)
- **Goal**: Convert to decoder-compatible format (predict discretized motion energy from neural activity)

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of /app:
- `.manifest`, `Dockerfile`, `docker-compose.yaml`
- `code/` : track2p python package + notebooks (reference code)
- `data/` : 6 subject folders (jm031, jm032, jm038, jm039, jm040, jm046) + `README.md` + `load_data.ipynb` (16 GB)
- `decoder.py`, `train_decoder.py` : decoder reference implementation
- `methods.txt`, `paper.pdf`

Environment verified: python3 with numpy 2.4.4, torch 2.6.0+cu124, scipy 1.18.0 (also suite2p importable). `pypdf` installed to extract paper text to `/app/cache/paper.txt`.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_traces(session_dir)` | data/load_data.ipynb | LOADING | `np.load(session/suite2p/plane0/F.npy)` -> (n_neurons, n_frames) raw fluorescence of tracked cells |
| `load_fov(session_dir)` | data/load_data.ipynb | LOADING | ops['meanImg'] |
| `load_coords_cell(session_dir, cell_id)` | data/load_data.ipynb | LOADING | ROI centroid from stat.npy |
| `zscore_rows(F)` | data/load_data.ipynb | PROCESSING | z-score each row (visualisation only) |
| `DataManagement.F_processing(F, Fneu, fs, neucoeff=0.0, baseline='maximin', sig_baseline=10.0, win_baseline=60.0)` | code/track2p/gui/data_management.py:185 | PROCESSING | dF/F0 used by the track2p GUI: `Fc = F - neucoeff*Fneu`; `Flow = gaussian_filter(Fc,[0,sig])`; `Flow = minimum_filter1d(Flow, win)`; `Flow = maximum_filter1d(Flow, win)`; return `Fc - Flow` (baseline-corrected fluorescence). `win = int(win_baseline*fs)` = 1800 frames. |
| `DataManagement.load_data(...)` | code/track2p/gui/data_management.py:40 | CURATION | Selects cells with `iscell[:,0]==1` (or prob > `iscell_thr`) then reindexes with the track2p match matrix so rows are matched across days. **The released data is already the output of this step** (suite2p-format export of track2p), so no further selection is needed. |
| track2p matching/registration (`track2p/match/*`, `track2p/register/*`) | code/track2p | CURATION | Produces the across-day matched cell set. Already applied to the released data. |

### Notes
- The released `suite2p/plane0/*.npy` per session contain **only** cells tracked across all days of that mouse; rows are matched across sessions of a mouse (README of data).
- `iscell.npy` in the released data is all 1 (cell prob > 0.5 everywhere), i.e. suite2p classifier curation was already applied (paper: "We considered all ROIs above the default threshold of 0.5 as true cells").
- No deconvolution-based analysis is used for the decoding analysis in the paper; the paper decodes from dF/F.
- The GUI's dF/F0 call uses `neucoeff=0.0` (no neuropil subtraction) with suite2p's default maximin baseline parameters -> matches the paper statement "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)".

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
```
data/<subject>/<YYYY-MM-DD>_a/
    suite2p/plane0/{F.npy, Fneu.npy, spks.npy, iscell.npy, stat.npy, ops.npy}
    move_deve/{motion_energy_glob.npy, tstamps.npy, interframe_int.npy}
data/<subject>/ground_truth.csv   (only jm038, jm039, jm046; manual tracking GT, not needed here)
```
- `F.npy`/`Fneu.npy`/`spks.npy`: float32 (n_neurons, n_frames), tracked+matched cells only.
- `ops.npy`: suite2p ops; `fs=30`, `nframes`=n_frames, `neucoeff=0.7`, `baseline='maximin'`, `win_baseline=60`, `sig_baseline=10`, `badframes` all False, `tau=0.3`, 512x512 px, 1 plane.
- `motion_energy_glob.npy`: uint64, one value per **camera** frame (camera triggered by the microscope, 30 Hz). `me[0] == 0` in every session (no preceding frame for the difference).
- `tstamps.npy` (same length as motion energy) and `interframe_int.npy` (length-1) give camera frame times; units are ~kiloseconds (median interval 3.359e-5 -> 33.59 ms -> 29.77 Hz; session end 1.2097 -> 1209.7 s for 36000 frames).

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total, summed over mice) | 2998 |
| Neurons / session (= per mouse) | jm031 221, jm032 370, jm038 685, jm039 746, jm040 541, jm046 435; mean 499.7 +- 197.7 (std, ddof=1) |
| Subjects | 6 |
| Sessions / subject | 7,7,7,7,6,7 -> 41 sessions total |
| Frames / session | 36000 (jm031, jm032; 20 min) or 54000 (jm038, jm039, jm040, jm046; 30 min) at 30 Hz |
| Trials (total, after 60 s splitting) | 14*20 + 27*30 = 1090 |
| Motion-energy frames | equals n_frames except 9 sessions where 1-148 camera frames are missing |

Missing camera frames: in sessions where `len(motion_energy) < n_frames`, the missing frames are exactly identified by `interframe_int` being an integer multiple of the median interval. Sanity check performed: `cumsum(round(ifi/median(ifi)))[-1] == n_frames-1` holds for **all** sessions with a length mismatch (jm031 x3, jm032 x3, jm039 x1, jm040 x1, jm046 x1).

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects | 6 mice | "a full dataset of 6 mice imaged daily for a minimum of 6 consecutive days" |
| Sessions / subject | >= 6 consecutive days, P7-P14 | "imaged daily for a minimum of 6 consecutive days within the second postnatal week (P7 to P14)" |
| Neurons / mouse (tracked) | 526 +- 190 std | "On average 526 (+- 190 std) neurons per mouse were successfully tracked across all days" |
| Fraction of day-1 neurons tracked | 33% +- 11% | same sentence |
| Example mouse tracked neurons | 728 | "728 ROIs that were tracked across all days in this example mouse (out of 1988)" |
| Imaging rate | 30 Hz resonant | Methods, Chronic 2-photon calcium imaging |
| Session length | "20 minutes" | Methods (data show 20 min for 2 mice, 30 min for 4 mice - see Step 4) |
| Videography | 30 Hz, triggered by microscope | Methods, Videography |
| Neural preprocessing | suite2p, cells = classifier prob > 0.5, dF/F = baseline-corrected F with default suite2p params | Methods, Processing of calcium imaging data |
| Motion energy | sum of squared pixelwise differences of consecutive video frames | Methods, Preprocessing videography |
| Decoding binning | average in bins of 10 consecutive timestamps (both dF/F and behaviour) -> 3 Hz | "For all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps" |
| Decoding CV blocks | consecutive 2-minute blocks, 5-fold nested CV | Methods, Decoding |
| Decoder in paper | ridge regression, continuous motion energy, scored with R2 | Methods, Decoding |

### Processing Details
- Temporal alignment: calcium imaging and videography are hardware-synchronised (microscope triggers camera), so camera frame i == imaging frame i, except when camera frames are dropped (identified via `interframe_int`).
- Temporal binning for decoding: non-overlapping bins of 10 frames (=333.3 ms, 3 Hz) by averaging.
- Neuron curation: suite2p classifier prob > 0.5 AND tracked across all days by track2p; already applied in the released data.
- Trial curation: the paper has no trials (continuous spontaneous recordings). For this decoder task sessions are cut into 60 s trials.

### Decoders Trained (paper)
| Decoded variable | Metric / value |
|---|---|
| Mouse motion (motion energy), same-day, cross-validated | R2 per session, low (~0-0.1) at early ages (<=P11), increasing to ~0.3-0.6 at later ages (Fig. 7C, 7G) |
| Mouse motion, cross-day | similar R2 for late-to-late pairs, near 0 involving early days (Fig. 7G,H) |

No classification accuracies are reported in the paper (the paper uses continuous regression, R2), so the comparison for this task is qualitative: decoding should be clearly above chance, better on later (P12-P14) sessions than early ones.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Session duration | ops nframes | 36000 frames (20 min) for jm031/jm032, 54000 (30 min) for jm038/39/40/46 | "each session lasted 20 minutes" | Data is authoritative; the methods statement is approximate/refers to the earlier cohort. Both durations are handled (sessions cut into 60 s trials -> 20 or 30 trials). |
| Neurons per mouse | - | 221,370,685,746,541,435 (mean 499.7 +- 197.7) | 526 +- 190 | Very close; difference likely due to the paper including a mouse/cells not in the public release or a slightly different curation vector in the GUI. Our numbers come straight from the released tracked data, which is the correct source. |
| Example mouse | - | max is jm039 with 746 | 728 tracked in example mouse | Same order; the example mouse figure may be from a different (non-released) processing run or the GUI curation vector removed a few cells. Not actionable. |
| Neuropil coefficient | GUI `F_processing` default `neucoeff=0.0` | ops `neucoeff=0.7` (used by suite2p only for deconvolution) | "baseline corrected fluorescence traces as our dF/F (default Suite2p parameters)" | Follow the reference code: neuropil is NOT subtracted for the dF/F used in analyses (neucoeff=0.0), maximin baseline with sig=10, win=60 s. |
| Frame rate | ops fs = 30 | measured 29.77 Hz from tstamps | 30 Hz | Use nominal 30 Hz (as the reference code does, `win=int(60*fs)`); 0.8% timing difference is irrelevant for 60 s trials and is documented. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `suite2p/plane0/F.npy` (+`Fneu.npy`) | `neural` | `F_processing` (maximin baseline-corrected dF, neucoeff=0.0, sig=10, win=60*30) then average in non-overlapping bins of 10 frames; reshape into 60 s (1800 frame -> 180 bin) trials | `data_management.F_processing`, `load_data.ipynb:load_traces` | float32, (n_neurons, 180) per trial |
| frame index / 30 Hz | `input[0]` = `time_from_session_start_s` | bin centre time = (trial*1800 + (b+0.5)*10)/30 s | - | time-varying, seconds |
| `move_deve/motion_energy_glob.npy` | `output[0]` = `motion_energy_quintile` | place on imaging frame grid using `interframe_int` (when camera frames are missing), linear-interpolate missing frames and the me[0] artifact, average in bins of 10 frames, then discretize into 5 equal-percentile bins **per session** | Methods (videography preprocessing + decoding binning) | int, 0..4, time-varying |
| subject folder name | `subjects`, `subject_idx` | jm031->A ... jm046->F ordering | data README | 6 subjects |
| - | `brain_regions` | all neurons in barrel cortex (S1) L2/3 | Methods | single region |

### Key Decisions
1. **dF/F definition**: use the reference implementation `F_processing` with its defaults (no neuropil subtraction, maximin baseline, gaussian sigma=10 frames, 60 s min/max window). Matches the paper's "baseline corrected fluorescence traces ... default Suite2p parameters".
2. **Binning**: 10 frames averaged -> 3 Hz (333.33 ms bins), exactly as in the paper's decoding analysis; applied to both neural and behaviour.
3. **Trial definition**: 60 s = 1800 frames = 180 bins, non-overlapping, tiling the session from t=0. Sessions are exact multiples (36000 or 54000 frames) so no partial trials are discarded.
4. **Temporal alignment**: camera and microscope are hardware synchronised; camera frame i == imaging frame i. Where camera frames were dropped (len(me) < nframes), the true frame index of each camera sample is reconstructed from `interframe_int` (integer multiples of the median inter-frame interval) and missing values are linearly interpolated. Where len(me)==nframes, mapping is 1:1 (a few jm046 sessions show one timestamp gap but no length mismatch -> treated as a timestamp glitch, per the data README which defines missing frames by the length mismatch).
5. **me[0]=0 artifact**: the first motion-energy sample cannot be computed (no previous frame); it is treated as missing and interpolated (filled with me[1]).
6. **Output discretization**: quintiles (5 equal-percentile bins) computed **per session** on the 3 Hz binned motion-energy trace, as specified in the Decoder Task.
7. **No further neuron/session curation**: all released cells are already suite2p-curated (iscell prob>0.5) and track2p-matched across all days; `badframes` is empty in every session.
8. **Input**: single time-varying input, elapsed time from session start in seconds (per Decoder Task).

### Planned Sanity Checks
- [x] `cumsum(round(ifi/median(ifi)))[-1] == nframes-1` for all sessions with a camera/imaging length mismatch.
- [ ] Number of sessions = 41, subjects = 6, trials = 1090, neurons/session as listed above.
- [ ] Each output class has ~20% of timepoints per session (quintiles).
- [ ] Spot-check: recompute binned dF/F for a random (session, trial, neuron, bin) from raw F/Fneu with independent code and compare with np.allclose.
- [ ] Spot-check: recompute binned motion energy and its quintile label for a random (session, trial, bin) from raw files.
- [ ] Input time in trial k starts at 60*k and increases by 1/3 s per bin.
- [ ] Correlation between binned motion energy and first PC of neural activity is higher for later days (paper Fig. 7D) - qualitative check.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` implements:
- `F_processing(...)`: **copied verbatim** from the reference `track2p/gui/data_management.py::DataManagement.F_processing` (same defaults: neucoeff=0.0, maximin baseline, sig_baseline=10, win_baseline=60 s). With neucoeff=0 the neuropil trace is unused, so `Fneu.npy` is not loaded (saves ~30 MB I/O per session).
- `align_motion_energy(me, ifi, nframes)`: maps camera samples onto the imaging frame grid (identity when lengths match; otherwise reconstructs indices from `interframe_int`, asserting the last index == nframes-1), marks `me[0]` (always 0 artifact) and dropped frames as missing and linearly interpolates them.
- `bin_frames(x, 10)`: non-overlapping 10-frame averaging (paper's decoding preprocessing) - vectorised reshape+mean.
- `quantile_discretize(x, 5)`: per-session quintile thresholds via `np.quantile` + `np.searchsorted`.
- `process_session(...)`: loads F, asserts all `iscell[:,0]==1`, computes dF, bins, aligns/bins/discretizes motion energy, builds the time input, cuts into 60 s (180-bin) trials, and (optionally) produces the 8-panel processing figure.
- `main()`: assembles the dict, adds metadata (incl. per-session `session_info`), runs assertions (shapes, finiteness, region idx length), prints class fractions/input range/neural stats, pickles the result.

CLI: `python -u convert_data.py <out.pkl> [--full|--sample] [--show-processing]`.

Code inefficiencies identified: loading `Fneu.npy` and `ops.npy` (85 MB) is unnecessary -> not loaded. `gaussian_filter`/min/max filters dominate runtime (~0.2-0.7 s/session).

Code speedups added: memory-light loading (only `F.npy`, `iscell.npy`, 3 small behaviour arrays), vectorised binning and discretization, no per-trial Python loops over timepoints. Result: ~2 s/session.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing` (sessions jm031/2023-10-18_a and jm046/2024-09-03_a - one 20 min and one 30 min session from different mice).

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons (total) | 656 (221 + 435) |
| Neurons / session | 221, 435 |
| Subjects | 2 |
| Sessions / subject | 1 |
| Trials (total) | 50 (20 + 30) |
| Trials / session | 20 (20 min), 30 (30 min) |
| Timepoints / trial | 180 (60 s at 3 Hz) |
| Input `time_from_session_start_s` range | [0.15, 1799.82] s |
| Output `motion_energy_quintile` distribution | [0.200, 0.200, 0.200, 0.200, 0.200] |
| Neural (binned dF) | mean 22.3, std 43.3, min -44.4, max 1294.7 (a.u.) |

### Processing Plots Review
`processing_jm031_2023-10-18_a.png`, `processing_jm046_2024-09-03_a.png` show 8 panels: raw F; baseline-corrected dF; 30 Hz vs 3 Hz binned dF (first 60 s); binned dF raster; motion-energy alignment (camera samples plotted at their reconstructed frame times, overlaid with the interpolated imaging-grid trace); binned motion energy with the four quintile thresholds; the discretized label overlaid on the continuous trace; and the concatenated trials re-overlaid on the full-session traces vs the time input. No anomalies: the trial-concatenated output is identical to the session output, the label steps occur exactly at the threshold crossings, and the binned traces track the 30 Hz traces.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
| skip Fneu.npy / ops.npy loading | ~1 s/session I/O |
| vectorised binning/discretization | negligible loops |

| Step | Time / Session | Estimated Total Time |
| load F | ~0.03 s (page cache cold: up to ~1 s) | |
| dF (maximin) | 0.2 s (36k frames, 221 n) - 0.7 s (54k frames, 435 n) | |
| binning + behaviour + assembly | <0.1 s | |
| total | 2.2 s/session measured (incl. plotting) | 41 sessions x ~2-4 s = **~2 min** (larger mice jm038/jm039 have up to 746 neurons x 54k frames, ~2-3x the sample cost) |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation (`/app/verification_sample_out.txt`)
- Errors: None
- Warnings: None ("Data format is valid, no errors or warnings.")

### Decoder Results (Sample; `/app/train_decoder_sample_out.txt`)
Loss decreased monotonically from 47.5 (epoch 1) to 1.41 (epoch 200).

| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| motion_energy_quintile | 0.460 | 0.291 | 0.200 |

Above chance on only 2 sessions, one of which is an early (P8-ish) session where the paper reports near-zero decodability of motion from neural activity (Fig. 7C). Full-dataset training should be better because it includes the later, strongly motion-modulated sessions.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/converted_data.pkl --full` -> 41 sessions in **33.9 s** (0.8 s/session; faster than the sample estimate because the plotting was off and files were in page cache).

### Output Files
- `converted_data.pkl`: 414.5 MB
- `conversion_full_out.txt`, `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Subjects | 6 mice | - | 6 subject folders | 6 | yes |
| Sessions | >= 6 consecutive days/mouse (P7-P14) | - | 7,7,7,7,6,7 = 41 | 41 (7,7,7,7,6,7) | yes |
| Total unique neurons | - | - | 2998 | 2998 (20445 neuron-sessions = sum over sessions) | yes |
| Neurons per mouse | 526 +- 190 | - | 221,370,685,746,541,435 (mean 499.7 +- 197.7) | identical | close (see Step 4) |
| Mean neurons/session | - | - | 498.66 | 498.66 | yes |
| Trials (total) | n/a (continuous recordings) | - | 41 sessions x (20 or 30) x 60 s | 1090 | yes |
| Trials/session | n/a | - | 20 (36000 frames) / 30 (54000 frames) | 20 or 30 | yes |
| Timepoints/trial | 10-frame bins -> 3 Hz | `bin size 10 frames` | 1800 frames/trial | 180 | yes |
| Bin size | 333.3 ms | - | - | 333.33 ms | yes |
| Input `time_from_session_start_s` range | - | - | 0-1200 / 0-1800 s | [0.15, 1799.82] | yes |
| Output quintile distribution | 20% each by construction | - | - | [0.200,0.200,0.200,0.200,0.200] (every session) | yes |
| Frames used | all (badframes empty) | suite2p `badframes` all False | 36000/54000 | 100% of frames (sessions are exact multiples of 1800) | yes |

No data lost: for every session `n_trials * 1800 == n_frames` (asserted in the sanity-check script).

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
`/app/verification_full_out.txt`: "Data format is valid, no errors or warnings." 41 sessions, 1090 trials, 6 subjects, 1 brain region, dinput 1, doutput 1, T=180 everywhere, n_neurons mean 498.66 (min 221, max 746), input range [0.2, 1799.8], output values 0-4 each with fraction 0.200. **No errors and no warnings to address.**

### Check 2: Independent sanity checks (`/app/cache/sanity_checks.py`, output `/app/cache/sanity_checks_out.txt`)
The script loads the raw `.npy` files directly and recomputes everything with independent code (single-neuron `gaussian_filter1d` + min/max filters instead of the 2-D filters used in the converter).

| Check | Method | Result |
|---|---|---|
| NEURAL spot checks (5 random session/trial/neuron/bin) | recompute maximin dF for one neuron, average frames `[k*1800+b*10, +10)` , `np.allclose(rtol=1e-4)` | all True (e.g. jm032/2023-10-23 trial 0 neuron 27 bin 2: 73.549088 vs 73.549088) |
| NEURAL full-trial checks (trials 0, 7, 19 of session 0) | recompute dF for all neurons and reshape-average | all True |
| INPUT | expected bin-centre times `(arange(k*180,(k+1)*180)*10+4.5)/30` | all True; t[0]=0.15 s, last trial ends 1199.8167 / 1799.8167 s |
| INPUT continuity | concatenated trials strictly increasing with exact 1/3 s steps in **all 41 sessions** | True |
| OUTPUT (5 sessions) | independent re-alignment + interpolation + binning + quantile labelling; `np.array_equal` on labels and `np.allclose` on edges | all True, fractions 0.2 each |
| OUTPUT ordering | mean raw motion energy per label in session 0: [777355, 787301, 800618, 812805, 1064356] | monotonically increasing -> label 0 = least motion, 4 = most |
| Structure | subjects list, 41 sessions, per-session neuron counts, `n_trials*1800 == n_frames` (no dropped data), `subjects[subject_idx[s]] == session subject` | all OK |

### Check 3: Reference code comparison
| Step | Reference | This conversion | Match / justification |
|---|---|---|---|
| (a) loading | `load_data.ipynb::load_traces` -> `suite2p/plane0/F.npy`; GUI also loads `Fneu`, `iscell`, `stat`, `ops` | `F.npy` + `iscell.npy` | same signal. `Fneu` not loaded because the reference dF/F call uses `neucoeff=0.0` (neuropil unused); `ops`/`stat` are not needed (fs is the documented 30 Hz; ROI coordinates unused) |
| (b) neuron/trial filtering | GUI: `iscell[:,0]==1` then reindex by the track2p match matrix | already applied in the released data; verified by asserting `iscell[:,0]==1` for all ROIs in all 41 sessions | identical |
| (c) temporal alignment | camera hardware-triggered by microscope (Methods); README: length mismatch = missing camera frames, "treated as missing values ... or interpolated" | frame index reconstructed from `interframe_int`, missing values linearly interpolated; `me[0]` (always 0, no preceding frame) interpolated | follows the data README's recommended option |
| (d) binning | "averaging in bins of 10 consecutive timestamps" for dF/F **and** behaviour (Methods, Decoding) | `bin_frames(x, 10)` mean, applied to dF and motion energy | identical |
| (e) input construction | n/a in the paper (no task variables) | elapsed time, per the Decoder Task spec | required by the task |
| (f) output construction | paper regresses continuous motion energy | same motion-energy trace, discretized into 5 per-session equal-percentile bins | required by the task (outputs must be categorical) |
| dF/F | `F_processing(F, Fneu, fs, neucoeff=0.0, baseline='maximin', sig_baseline=10, win_baseline=60)` | function copied verbatim into `convert_data.py` | identical |

Differences from the reference, and why: (1) trials do not exist in the paper (continuous 20/30 min recordings) - the Decoder Task requires 60 s trials; (2) the output is discretized - required by the target format; (3) the decoder is the provided PCA+logistic network rather than ridge regression - dictated by `train_decoder.py`; (4) cross-validation here is the decoder script's trial split rather than the paper's 2-min-block nested CV - dictated by `train_decoder.py`.

### Check 4: Key statistics comparison
See the table in Step 9. The only numeric difference from the paper is neurons/mouse (499.7 +- 197.7 here vs 526 +- 190 reported). Investigated: the released `F.npy` row counts are constant across all days of each mouse (221/370/685/746/541/435) and `iscell` is all-1, so no further curation is possible on our side; the difference must come from the paper counting a slightly different set (e.g. before the GUI curation vector, or including the mouse's pre-release run). The paper's example mouse has 728 tracked cells while the closest released mouse (jm039) has 746 - same order, consistent with a small curation difference. This is not fixable from the released data and does not affect the conversion.

### Check 5: Edge cases
- First bin of a session: `me[0]` is a known artifact (0) and is interpolated; the first time bin is 0.15 s (bin centre), not 0 - documented.
- Session lengths are exact multiples of 1800 frames, so no partial trial is dropped and no data is lost (asserted).
- Sessions with dropped camera frames (9 of 41, 1-148 frames): index reconstruction asserted to land exactly on `n_frames-1`.
- jm046 sessions with a single long inter-frame interval but **no** length mismatch: the camera sample count equals the imaging frame count, so the mapping stays 1:1 (a timestamp glitch, not a dropped frame). The data README defines missing frames by the length mismatch, so this is the correct handling.
- `ground_truth.csv` files (3 mice) are manual-tracking ground truth for the tracking benchmark, not neural data - correctly ignored.
- All neural/input values finite; outputs are integers in 0..4 (asserted in the converter).

### Issues Found and Resolved
- Initially considered using `spks.npy` (deconvolved) - rejected: the paper's decoding uses dF/F.
- Considered neuropil subtraction with ops `neucoeff=0.7` - rejected: the reference `F_processing` call uses `neucoeff=0.0`, and ops' value is only used internally by suite2p for deconvolution.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

Command: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples` (GPU, 200 epochs, npcs=100, balanced loss). Outputs: `/app/train_decoder_full_out.txt`, `sample_trials.png`, `predictions.png`.

### Training Progress
- Loss decreasing: Yes, monotonically from ~50 (epoch 1) to 1.137 (epoch 200); test loss 2.99.

### Decoder Results (Full)
| Output | Classes | Chance | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|---------|--------|-------------|--------|-------|
| motion_energy_quintile | 5 | 0.200 | 0.610 | 0.308 | 1.54x chance; pooled over all 41 sessions, including the early (P7-P11) sessions where the paper itself finds essentially no motion information in the neural activity |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance analysis
| Variable | Classes | Chance | Validation balanced acc | Ratio |
|---|---|---|---|---|
| motion_energy_quintile | 5 | 0.200 | 0.308 | 1.54x |

Above chance but below 1.5x for some outputs was the trigger for investigation; here the ratio is 1.54x, i.e. just above that threshold, so I investigated why it is not higher:
1. **This is expected from the biology reported in the paper.** Roughly half of the sessions are early (P7-P11) where the paper reports near-zero same-day decoding R2 (Fig. 7C) and near-zero correlation between motion and PC1 (Fig. 7D). Pooling early and late sessions necessarily caps the achievable accuracy.
2. **Quintiles of motion energy are a very fine-grained target.** Within a session the four lower quintile thresholds are separated by only a few percent of the motion-energy range (e.g. session jm031/2023-10-18: class means 777k, 787k, 801k, 813k, 1064k), i.e. classes 0-3 are all "quiescence" and differ mostly by camera/photon noise; only class 4 corresponds to real movement. Perfect separation of classes 0-3 is not physically possible.

### Check 2: Accuracy comparison to the paper
The paper reports **no classification accuracies** - all decoding is ridge **regression** scored with R2. To compare like with like I re-implemented the paper's analysis on the converted data (`/app/cache/ridge_check.py`: ridge regression, dF (converted, 3 Hz) -> continuous binned motion energy, 5-fold CV with splits on consecutive 2-minute blocks, best lambda from a log grid):

| Mouse | R2 per day (day1 -> last day) | Paper expectation (Fig. 7C) |
|---|---|---|
| jm031 | 0.10, 0.22, -0.13, 0.19, 0.34, 0.24, **0.43** | low early, increasing after ~P11 |
| jm032 | 0.36, -0.06, 0.09, 0.26, 0.38, 0.55, **0.46** | same |
| jm038 | 0.06, 0.09, 0.01, -0.06, 0.01, 0.18, **0.45** | same |
| jm039 | 0.20, 0.09, 0.18, 0.30, 0.47, 0.55, **0.55** | same |
| jm040 | 0.02, 0.03, 0.06, 0.17, 0.36, **0.54** | same |
| jm046 | 0.14, 0.12, 0.44, 0.41, 0.20, 0.73, **0.79** | same |

This reproduces the paper's central quantitative result (Fig. 7C, 7G: R2 near 0 at early ages, ~0.3-0.8 at P13-P14, with substantial variability), which would be impossible if the neural traces, the motion-energy stream, or their temporal alignment were wrong. **Conclusion: the converted data contains the same information as the paper's data; the classification accuracy of 0.308 is a property of the 5-way quantile task pooled over early+late sessions, not of a conversion bug.**

### Check 3: Train vs validation gap
Training 0.610 vs validation 0.308 (~2x). This is overfitting of the decoder (100 PCs per session x 41 session-specific projections, 200 epochs, no early stopping), not data leakage: trials are disjoint 60 s blocks, the train/validation split is made by the provided script at the trial level, and the output labels are computed from behaviour only (thresholds are per-session, identical for train and validation trials, so no label leakage between trials).

### Additional experiment: neural normalization
I tested whether the raw (un-normalized) baseline-corrected dF limits the decoder by building a variant with per-neuron z-scoring within session (`/app/cache/make_zscore_variant.py`): validation balanced accuracy 0.329 vs 0.308 (training 0.608 vs 0.610). The gain is marginal (+0.02) and z-scoring is **not** part of the reference processing (the paper's ridge decoding uses the dF traces directly), so the reference-faithful version was kept as the deliverable. Documented here for transparency.

### Issues Found and Resolved
- No further issues found. All Step 10 checks were re-run after the full conversion (`/app/cache/sanity_checks_out.txt`) and all pass.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created (`/app/README.md`)
- [x] cache/ folder created with `README_CACHE.md` (exploration, sanity-check, ridge-reproduction and normalization-experiment scripts and their outputs)
- [x] All files organized: deliverables in `/app`, investigation scripts in `/app/cache`
