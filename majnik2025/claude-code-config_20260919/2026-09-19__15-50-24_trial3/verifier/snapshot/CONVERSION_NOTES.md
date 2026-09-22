# Dataset Conversion Notes

## Overview
- **Dataset**: Majnik et al. 2025, eLife 14:RP107540 — "Longitudinal tracking of neuronal activity from
  the same cells in the developing brain using Track2p". Chronic 2-photon GCaMP8m calcium imaging of
  layer 2/3 mouse barrel cortex (S1) during the second postnatal week, with simultaneous videography
  ("motion energy") of spontaneous movement.
- **Date started**: 2026-09-19
- **Goal**: Convert to decoder-compatible format.
  - Decoder input: time elapsed from beginning of session (s), time-varying.
  - Decoder output: motion energy discretized into 5 equal-percentile (quintile) bins, per session,
    time-varying.
  - Sessions split into 60 s trials.

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `CONVERSION_NOTES.md` (this file)
- `Dockerfile`, `docker-compose.yaml`, `.manifest`
- `code/` — the Track2p reference code base (python package + notebooks)
- `data/` — the published dataset (6 subject folders + `README.md` + `load_data.ipynb`)
- `decoder.py` — decoder library used by `train_decoder.py`
- `train_decoder.py` — validation / decoder-training script
- `methods.txt` — methods excerpt from the paper
- `paper.pdf` — the reference paper

Environment verified: `python3`, `numpy 2.4.4`, `torch 2.6.0+cu124` all import fine.
(`pypdf` and `pymupdf` were pip-installed to read the paper text/figures.)

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

Two code sources are relevant:
1. `/app/code` — the **Track2p** python package (the tracking algorithm itself + a GUI).
2. `/app/data/load_data.ipynb` — the **official data-loader notebook** shipped with the dataset.

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_traces(session_dir)` | `data/load_data.ipynb` | LOADING | `np.load(session_dir/suite2p/plane0/F.npy)` → raw fluorescence, shape `(n_neurons, n_frames)`. Comment in the notebook: *"for more proper analysis compute dF/F the way as described in the paper (or alternatively use spks.npy)"*. |
| `load_fov(session_dir)` | `data/load_data.ipynb` | LOADING | `ops['meanImg']` from `ops.npy`. |
| `load_coords_cell(session_dir, cell_id)` | `data/load_data.ipynb` | LOADING | ROI centroid from `stat.npy` (`ypix.mean()`, `xpix.mean()`). |
| `zscore_rows(F)` | `data/load_data.ipynb` | PROCESSING | Row z-score — explicitly marked *"only for visualization in raster"*, **not** used for analysis. |
| `DataManagement.F_processing(F, Fneu, fs, neucoeff=0.0, baseline='maximin', sig_baseline=10.0, win_baseline=60.0, prctile_baseline=8)` | `code/track2p/gui/data_management.py:185` | PROCESSING | **The reference dF/F0 implementation.** `Fc = F - neucoeff*Fneu`; `Flow = gaussian_filter(Fc, [0, sig_baseline])`; `Flow = minimum_filter1d(Flow, int(win_baseline*fs))`; `Flow = maximum_filter1d(Flow, win)`; returns `Fc - Flow`. This is Suite2p's "maximin" baseline with Suite2p's default parameters. Called (line 90) whenever the GUI trace type is `'dF/F0'`, with no `neucoeff` argument → **neucoeff = 0.0 (no neuropil subtraction)**. |
| `DataManagement.load_data(...)` | `code/track2p/gui/data_management.py:40-92` | CURATION | Loads `F.npy`, `Fneu.npy`, `iscell.npy` per day, keeps ROIs with `iscell[:,0]==1` or `iscell[:,1] > track_ops.iscell_thr`, then reindexes with the Track2p match matrix so that row *i* is the same neuron on every day. |
| `DefaultTrackOps.iscell_thr = 0.50` | `code/track2p/ops/default.py:21` | CURATION | Suite2p cell-probability threshold — matches the paper ("all ROIs above the default threshold of 0.5"). |
| `run_t2p(track_ops)` | `code/track2p/t2p.py:20` | PROCESSING | Full pipeline: register FOVs across days (affine, ROI-based), match ROIs (IoU), build match matrix. |
| `npy_to_s2p` / save block in `t2p.py:180-280` | `code/track2p/t2p.py` | SAVING | Writes the *tracked-cells-only* Suite2p folders (`F.npy`, `Fneu.npy`, `spks.npy`, `stat.npy`, `iscell.npy`, `ops.npy`) — i.e. exactly the folders shipped in `/app/data`. |

### Notes
- **The released data are already the Track2p output**: every `suite2p/plane0/F.npy` contains only the
  neurons tracked across *all* days of that mouse, with rows matched across days. Confirmed: row counts
  are identical across all sessions of a subject, and `iscell[:,0]` is 1 for every row of every session
  (`iscell[:,1]` min ≈ 0.50) — i.e. the `iscell > 0.5` curation of the paper has **already been applied**.
  ⇒ No further neuron-quality filtering is needed or possible from the released files.
- **No dF/F is stored** — it must be computed. The only dF/F implementation in the reference code is
  `F_processing` above. Note it *subtracts* the maximin baseline and does **not** divide by it; the paper
  phrases this as *"We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p
  parameters)"*, which matches the code exactly.
- `spks.npy` (deconvolved) is available but the paper's decoding analysis explicitly uses dF/F.
- There is **no analysis/decoding code** in `/app/code` (only the tracking algorithm + GUI); the decoding
  procedure is only described in the Methods (see Step 3).

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
```
/app/data/
  README.md                       # organisation notes from the authors
  load_data.ipynb                 # official loader notebook
  jm031/ jm032/ jm038/ jm039/ jm040/ jm046/      # one folder per mouse (subject)
     <YYYY-MM-DD>_a/                             # one folder per recording day (session)
        suite2p/plane0/
            F.npy        (n_neurons, n_frames) float32   raw fluorescence, tracked cells only
            Fneu.npy     (n_neurons, n_frames) float32   neuropil
            spks.npy     (n_neurons, n_frames) float32   deconvolved
            stat.npy     (n_neurons,) object             ROI stats (xpix/ypix/...)
            iscell.npy   (n_neurons, 2) float64          [is_cell, probability]
            ops.npy      dict                            fs=30, nframes, meanImg, ...
        move_deve/
            motion_energy_glob.npy  (n_cam_frames,) uint64   motion energy per camera frame
            tstamps.npy             (n_cam_frames,) float64  camera frame timestamps
            interframe_int.npy      (n_cam_frames-1,) float64 inter-frame intervals
  jm038/ground_truth.csv, jm039/ground_truth.csv, jm046/ground_truth.csv   # manual tracking GT
```
- Only one imaging plane (`plane0`) exists for every session.
- `ops['fs'] = 30` Hz for every session; `ops['nframes']` equals `F.shape[1]`.
- Subject ↔ paper letter mapping (from `data/README.md`: alphabetical order):
  jm031=A, jm032=B, jm038=C, jm039=D, jm040=E, jm046=F.
  **Sanity check passed**: the paper marks mice C, D and F with `*` as the mice used for algorithm
  evaluation against manual ground truth (Fig. 5B), and exactly jm038, jm039, jm046 contain a
  `ground_truth.csv`. ✔

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 6 (jm031, jm032, jm038, jm039, jm040, jm046) |
| Sessions | 41 (7,7,7,7,6,7 — one per consecutive recording day) |
| Sessions / subject | 6–7 (mean 6.83) |
| Neurons (total, summed over subjects) | 2998 (221, 370, 685, 746, 541, 435) |
| Neurons / subject (mean ± std) | 499.7 ± 197.7 |
| Frames / session | 36000 (jm031, jm032 → 20 min @30 Hz) or 54000 (jm038–jm046 → 30 min @30 Hz) |
| Trials (60 s) / session | 20 or 30 |
| Trials (total) | 14×20 + 27×30 = 1090 |
| Imaging rate | 30 Hz |
| Camera rate | 30 Hz, hardware-triggered by the microscope |

Data-quality observations:
- No NaN/Inf anywhere in `F.npy`.
- 8 of the 2998 tracked ROIs have an **all-zero** trace on at least one day (jm031: 1, jm032: 3,
  jm038: 3, jm046: 1) — ROIs that fell outside the imaged FOV on that day.
- `motion_energy_glob.npy` is shorter than `n_frames` in 6 of 41 sessions (dropped camera frames:
  2, 3, 116, 2, 2, 148, 1, 1, 1 frames in the affected sessions).
- `motion_energy_glob[0] == 0` in **every** session — an artefact of the frame-difference definition
  (there is no frame −1), not a real measurement.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects | 6 | "we used a full dataset of 6 mice imaged daily for a minimum of 6 consecutive days within the second postnatal week (P7 to P14, see Fig. 5B)" |
| Sessions / subject | ≥6 (Fig. 5B: A P7–P13, B P7–P13, C P8–P14, D P8–P14, E P9–P14, F P8–P14) → 7,7,7,7,6,7 = **41 sessions** | Fig. 5B |
| Neurons / subject (mean ± std) | 526 ± 190 | "On average 526 (± 190 std) neurons per mouse were successfully tracked across all days" |
| Neurons / subject (Fig. 5B bar labels) | A 285, B 376, C 799, D 728, E 541, F 411 (total 3140) | Fig. 5B "N cells" |
| % of day-1 cells tracked | 33 % ± 11 % (Fig. 5B: 47, 20, 36, 37, 41, 19) | "corresponding to 33 % (± 11 % std) of the neurons detected on the first day" |
| Example mouse (D) tracked ROIs | 728 out of 1988 | "yielded a total of 728 ROIs that were tracked across all days in this example mouse (out of 1988 …)" |
| Imaging rate | 30 Hz (resonant) | "Imaging rate was 30 Hz (resonant scanner) and each session lasted 20 minutes." |
| Session duration | "20 minutes" (data actually contain 20-min and 30-min sessions) | Methods |
| FOV / depth | 720×720 µm, 512×512 px, layer 2/3, 100–200 µm deep, barrel cortex | Methods |
| Videography rate | 30 Hz, triggered by microscope acquisition | "with the microscope acquisition acting as a trigger for camera frame acquisition, also allowing for simple synchronisation across the two modalities" |
| Cell curation | Suite2p classifier probability > 0.5 | "We considered all ROIs above the default threshold of 0.5 as true cells." |
| Neural signal used | baseline-corrected F, Suite2p defaults | "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters) for all subsequent analyses." |
| Motion energy definition | sum over pixels of squared pixel-wise difference of consecutive video frames | "we first took each two consecutive frames, computed their pixelwise difference. We then squared all individual pixel-wise values and summed across pixels." |
| **Decoding time bin** | **10 frames ⇒ 1/3 s (333.33 ms)** | "For all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." |
| Decoding data split | consecutive **2-minute blocks**, 5-fold nested CV | "splits were done on consecutive 2 minute blocks of the recording" |
| Decoder (paper) | ridge regression, continuous target | Methods "Decoding" |

### Processing Details
- **Temporal alignment**: camera frames are hardware-triggered by the 2-photon acquisition, so camera
  frame *k* corresponds to imaging frame *k* — except where camera frames were dropped. The dataset
  README says the dropped-frame indices can be recovered from `tstamps.npy` / `interframe_int.npy` and
  that the missing values can be treated as missing or interpolated over.
- **Temporal binning**: non-overlapping means of 10 consecutive 30 Hz samples for *both* dF/F and the
  behavioural trace → an effective 3 Hz sampling, 333.33 ms bins.
- **Neural processing**: Suite2p maximin baseline (`sig_baseline=10` frames, `win_baseline=60` s),
  subtracted from F.

### Curation Steps
**Neuron curation rules**: Suite2p `iscell` probability > 0.5 **and** successfully tracked across all
days of that mouse. Both are already applied in the released files. (I additionally drop the 8 ROIs
whose trace is identically zero on at least one day — see Step 5 decisions.)

**Trial curation rules**: The paper has no trial structure (continuous spontaneous-activity
recordings). Here trials are 60 s blocks imposed by the decoder task; the only curation is dropping a
trailing incomplete block (in practice there is none: 36000 and 54000 frames are both exact multiples
of 1800 frames = 60 s).

### Decoders Trained
| Decoded variable | Accuracy |
|---|---|
| Mouse motion (motion energy), same-day, ridge regression | Reported as **R²**, not accuracy. Fig. 7C/G/H: near 0 for early ages (≤P11), rising to ≈0.3–0.6 for late ages (>P11), with strong across-mouse variability. |
| Mouse motion, cross-day | Similar magnitude for late→late; near 0 involving early days. |

The paper therefore provides **no classification accuracy** to compare against directly. The
usable quantitative expectations are (i) the R² trend with age, and (ii) the correlation between the
first PC of neural activity and motion, which rises steeply after P11 (Fig. 7D). Both are used as
sanity checks below.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Session duration | — | 36000 frames (20 min) for jm031/jm032; 54000 frames (30 min) for jm038/jm039/jm040/jm046 | "each session lasted 20 minutes" | The Methods statement is only correct for 2 of 6 mice. Use the actual frame count per session. Splitting into 60 s trials handles both lengths (20 vs 30 trials/session). No data discarded. |
| Neurons per mouse | — | 221, 370, 685, 746, 541, 435 (mean 499.7) | Fig. 5B: 285, 376, 799, 728, 541, 411 (mean 523.3); text "526 ± 190" | The released Track2p output differs slightly from the version used for the figures (mouse E matches exactly, B is within 6, D is *higher* in the data). Most plausibly the deposited data were regenerated with a slightly different Track2p run/curation. The released data is the only available source, so it is used as-is; the discrepancy (−5 % in the mean) is documented rather than "fixed". |
| dF/F definition | `F_processing`: `Fc - maximin_baseline(Fc)`, `neucoeff=0.0` | — | "baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)" | Consistent: "baseline corrected" = baseline *subtracted*. Followed the reference code exactly, including no neuropil subtraction (see Step 5 decision 2). |
| Camera/imaging sync | — | `len(motion_energy) < n_frames` in 6/41 sessions; timestamp gaps of exactly 2× the median inter-frame interval | "the microscope acquisition acting as a trigger for camera frame acquisition"; README: missing frames recoverable from `tstamps.npy` | Reconstructed the imaging-frame index of every camera frame from the timestamp gaps. **Verified for all 41 sessions**: `n_camera_frames + n_detected_gaps == n_imaging_frames` in every session where frames were actually dropped (6/6), and the 35 sessions with `len(ME) == n_frames` are mapped one-to-one. |
| iscell threshold | `iscell_thr = 0.50` | `iscell[:,0]` all 1, `iscell[:,1]` min ≈ 0.50 | "default threshold of 0.5" | Fully consistent; already applied in the released data. |

Additional consistency checks performed before writing the converter:
- **Mice marked `*` in Fig. 5B (C, D, F) ↔ `ground_truth.csv` present for jm038, jm039, jm046.** ✔
- **PC1-vs-motion correlation rises with age** (paper Fig. 7D): e.g. jm031 (P7→P13):
  0.19, 0.12, 0.09, 0.15, 0.28, 0.38, 0.43; jm046 (P8→P14): 0.01, 0.04, 0.04, 0.01, 0.20, 0.79, 0.69;
  jm040 (P9→P14): 0.05, 0.03, 0.07, 0.09, 0.40, 0.68. Steep rise after P11 for most mice. ✔
- **Neural/behaviour lag**: cross-correlating binned population dF/F with binned motion energy peaks at
  a lag of −1 to −4 bins (neural activity ~0.3–1.3 s *after* motion), consistent with GCaMP8m kinetics.
  There is no gross (tens of seconds) misalignment. ✔

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `<subj>/<sess>/suite2p/plane0/F.npy` | `neural` | (1) maximin baseline subtraction → dF/F; (2) non-overlapping mean of 10 frames; (3) reshape into 180-bin (60 s) trials | `load_traces` (load_data.ipynb), `DataManagement.F_processing` (`track2p/gui/data_management.py:185`) | float32, shape (n_neurons, 180) per trial |
| `ops['fs']`, `F.shape[1]` | timing | bin size = 10/30 s = 333.333 ms | — | fs = 30 Hz in every session |
| bin index | `input[0]` = `time_from_session_start_s` | `t = (bin*10 + 4.5)/30` s (centre of the 10-frame bin) | — | time-varying, shape (1, 180) |
| `move_deve/motion_energy_glob.npy` (+ `tstamps.npy`) | `output[0]` = `motion_energy_quintile` | (1) map camera frames → imaging frames, dropped frames = NaN; (2) frame 0 = NaN (artefact); (3) non-overlapping nan-mean of 10 frames; (4) per-session quintile bins (20/40/60/80th percentiles) → labels 0–4 | README of dataset; Methods "Preprocessing videography" & "Decoding" | int, time-varying, shape (1, 180) |
| subject folder name | `subjects`, `subject_idx` | — | — | 6 subjects |
| — | `brain_regions`, `brain_region_idx` | all neurons = "S1" (barrel cortex, L2/3) | Methods | single region |

### Key Decisions
1. **Neural signal = baseline-subtracted fluorescence ("dF/F" as defined in the paper).** The paper
   explicitly states dF/F was used for all analyses including decoding, and the only dF/F implementation
   in the reference code is `F_processing`, which subtracts a Suite2p maximin baseline
   (`gaussian_filter(sigma=10 frames)` → `minimum_filter1d(60 s)` → `maximum_filter1d(60 s)`).
   I reimplement that function verbatim. I do **not** divide by F0: the paper says "baseline *corrected*"
   and the reference code does not divide. (Checked empirically: dividing by F0 does not improve the
   PC1–motion correlation; on several sessions it makes it worse.)
2. **No neuropil subtraction (`neucoeff = 0`).** This is what the reference `F_processing` does when
   called by the Track2p GUI (line 90 passes no `neucoeff`). Empirically, `neucoeff=0.7` (the Suite2p
   default stored in `ops`) gives comparable or slightly noisier PC1–motion correlations; following the
   paper's own code is the better-justified choice.
3. **Bin size 333.33 ms (10 frames).** Directly from the Methods: "For all decoding analysis we slightly
   denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps."
   Same binning is applied to the behavioural trace, as the Methods require.
4. **Trials = consecutive non-overlapping 60 s blocks** (180 bins), as required by the Decoder Task.
   This mirrors the paper's own use of consecutive time blocks (2 min) for cross-validation splits.
   36000 and 54000 frames are exact multiples of 1800, so no partial trial is ever discarded.
5. **Missing camera frames** are reconstructed, not ignored: camera frame *k*'s imaging-frame index is
   `k + (number of dropped frames before k)`, where drops are detected as inter-frame intervals ≈ 2× the
   median. Motion energy at dropped frames is NaN and excluded from the bin mean (nan-mean). Dropped
   frames are never consecutive (all gaps are exactly one frame), so no 10-frame bin is ever fully
   missing. Sessions where `len(ME) == n_frames` use the identity mapping (this also covers 3 jm046
   sessions that contain timestamp jitter but no actually-missing frames).
6. **Motion energy at frame 0 is treated as missing** (it is identically 0 in every session because a
   frame difference needs a preceding frame). Affects at most 1 of 10 samples in the first bin of a
   session.
7. **Output discretisation: per-session quintiles of the *binned* motion energy.** The Decoder Task
   specifies "five equal-percentile bins, selected per session". Percentiles are computed on the same
   binned trace that is used as the target, over all bins of the session, so the 5 classes are
   (near-)exactly equally populated within each session. Per-session percentiles also normalise away the
   arbitrary per-session scale of motion energy (different illumination/camera gain/animal size).
8. **Drop the 8 ROIs that are identically zero on at least one day** (per subject, applied to all that
   subject's sessions so the tracked population stays matched across days). These ROIs carry no signal on
   those days and would contribute all-zero rows. 2998 → 2990 neurons.
9. **No session/trial exclusion.** All 41 sessions and all 1090 trials are kept: all sessions have
   ≥20 trials (≫2 required), no NaNs in F, and behavioural coverage is ≥99.6 % of frames in every session.
10. **Alignment event = start of the 60 s trial** (`off_start = 0`, `off_end = 60`). The recordings are
    continuous spontaneous activity with no task events, so the only meaningful alignment is the
    session/trial start.

### Planned Sanity Checks
- [x] Subject↔paper-letter mapping confirmed via `ground_truth.csv` ↔ `*` in Fig. 5B.
- [x] `n_camera_frames + n_gaps == n_imaging_frames` for every session with dropped frames.
- [ ] Converted neural value at a random (session, trial, neuron, bin) equals a from-scratch
      recomputation directly from `F.npy` (`np.allclose`).
- [ ] Converted input time equals `(global_bin*10 + 4.5)/30` and is continuous across trial boundaries.
- [ ] Converted output label equals `np.digitize` of the independently recomputed binned motion energy.
- [ ] Output class fractions ≈ 0.2 each per session.
- [ ] Trial/session/neuron counts match Step 2 table.
- [ ] PC1-vs-motion correlation computed from the *converted* data reproduces the age trend of Fig. 7D.
- [ ] Neural↔motion cross-correlation peaks at small positive neural lag (no gross misalignment).

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` implements the mapping of Step 5. Structure:

| Function | Purpose |
|---|---|
| `list_sessions()` | enumerate the 41 (subject, session, dir) triples in subject/chronological order |
| `load_traces(session_dir)` | copy of the dataset notebook's loader (`suite2p/plane0/F.npy`) |
| `load_fs(session_dir)` | imaging rate from `ops.npy` (asserted == 30 Hz for every session) |
| `f_processing(...)` | **verbatim port of `track2p/gui/data_management.py:F_processing`** — Suite2p maximin baseline, `neucoeff=0`, `sig_baseline=10`, `win_baseline=60 s`; returns `F - baseline` |
| `bin_mean(x, 10)` | non-overlapping 10-frame means (optionally nan-aware), the paper's decoding binning |
| `motion_energy_on_imaging_frames(...)` | maps camera frames onto the imaging-frame grid, reconstructing dropped frames from `tstamps.npy`; NaN for missing frames and for frame 0 |
| `discretize_quantiles(x, 5)` | per-session 20/40/60/80th-percentile edges → labels 0–4 |
| `find_bad_neurons(...)` | per-subject mask of ROIs with an all-zero trace on ≥1 day |
| `convert_session(...)` | full per-session pipeline + timing + diagnostics |
| `plot_processing(...)` | 7-panel figure of every processing step (`--show-processing`) |
| `main()` | assembles the target dict, runs structural sanity checks, pickles the result |

The script runs as `python -u /app/convert_data.py <outpicklefile>` with `--full` (default),
`--sample` (2 sessions: one 20-min session *with* 116 dropped camera frames and one 30-min
session, so both session lengths and the dropped-frame path are exercised) and
`--show-processing`.

Code inefficiencies identified: none material. The only heavy operations are the three
`scipy.ndimage` filters used for the baseline; these are already O(n) (van Herk/Gil-Werman
for the min/max filters) and take ~1.2 s for the largest session (746 × 54000).

Code speedups added: memory-mapped loading in `find_bad_neurons`; all binning done with a
single reshape+mean instead of a Python loop; trials produced by slicing the already-binned
matrix (no recomputation per trial); `float32` throughout for the neural data.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
(log: `/app/conversion_sample_out.txt`).

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (jm031/2023-10-22, jm039/2024-05-06) |
| Neurons (total) | 966 |
| Neurons / session | 220, 746 |
| Subjects (used) | 2 (of the 6 listed in `subjects`) |
| Sessions / subject | 1 |
| Trials (total) | 50 |
| Trials / session | 20 (20 min session), 30 (30 min session) |
| `time_from_session_start_s` range | [0.15, 1799.82] s |
| `motion_energy_quintile` distribution | [0.200, 0.200, 0.200, 0.200, 0.200] |

### Processing Plots Review
`processing_jm031_2023-10-22_a.png`, `processing_jm039_2024-05-06_a.png`. Checked and no
anomalies:
- Panel 1: the maximin baseline hugs the bottom of each raw trace and is flat under calcium
  transients — i.e. it is a baseline, not a smoothed copy of the signal.
- Panel 2: binned dF/F overlays the 30 Hz dF/F with no time offset and no amplitude bias.
- Panel 3: the binned raster of jm031 at P11 shows the synchronous network events described
  in the paper; jm039 at P14 shows the decorrelated late-development pattern (Fig. 5A).
- Panel 4: the reconstructed motion-energy trace has no discontinuities at the 116 dropped
  camera frames of jm031/2023-10-22.
- Panel 5/6: the quintile edges cut the binned motion energy into 5 exactly equal groups.
- Panel 7: the re-concatenated trials reproduce the session-level output trace exactly and
  the time axis is continuous across trial borders.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| mmap in `find_bad_neurons`, vectorised binning, float32, no per-trial recomputation | conversion is I/O+filter bound; per-session cost 0.3 s (20 min session) to 1.5 s (30 min, 746 neurons) |

| Step | Time / Session | Estimated Total Time |
|---|---|---|
| load `F.npy` | 0.1–0.5 s | ~15 s |
| dF/F + binning | 0.2–1.0 s | ~30 s |
| behaviour | <0.02 s | <1 s |
| **total** | **0.9 s mean** | **~40 s for 41 sessions** (measured: 37.8 s) |

Well under the 15-minute budget, so no further optimisation was needed.

### Verification
`python -u /app/train_decoder.py /app/sample_data.pkl --verify-only` →
`/app/verification_sample_out.txt`: **"Data format is valid, no errors or warnings."**
Input range [0.2, 1799.8] s and output fractions 0.200 each, as expected.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
Loss decreased monotonically (70.1 at epoch 1 → 0.97 at epoch 200).

| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| motion_energy_quintile | 0.704 | 0.324 | 0.200 |

Validation accuracy is 1.6× chance on only 2 sessions — above chance for the single output,
so the sample passes.

**Variant experiments run at this stage** (documented in `/app/cache/`, each converted with
the same code but a different neural signal, then run through the same decoder):

| Neural signal variant | Validation balanced acc (sample) |
|---|---|
| `F - maximin baseline` (reference code, **chosen**) | 0.324 |
| `(F - maximin baseline) / baseline` (divide to get a true dF/F0) | 0.280 |
| `F - 0.7*Fneu - maximin baseline` (Suite2p default neuropil coefficient) | 0.330 |

Dividing by F0 is clearly worse; neuropil subtraction is within run-to-run noise. Both
confirm that following the reference implementation costs nothing, so the reference
implementation was kept.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 413.5 MB, written in 37.8 s (0.92 s/session)
- `verification_full_out.txt`: created — **"Data format is valid, no errors or warnings."**

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Subjects | 6 | — | 6 | 6 | ✔ |
| Sessions | 41 (Fig. 5B: 7,7,7,7,6,7) | — | 41 | 41 (7,7,7,7,6,7) | ✔ |
| Sessions/subject | 6–7 | — | 6–7 | 6–7 | ✔ |
| Total neurons | 3140 (Fig. 5B sum) | — | 2998 | 2990 | ✗ (−4.8 %, see below) |
| Mean neurons/subject | 526 ± 190 | — | 499.7 ± 197.7 | 498.3 ± 197.9 | ~ (within 5 %) |
| Neurons/subject | A285 B376 C799 D728 E541 F411 | — | 221, 370, 685, 746, 541, 435 | 220, 367, 682, 746, 541, 434 | ✗ for A/C, ✔ for E |
| Trials (total) | n/a (no trial structure) | — | — | 1090 | — |
| Trials/session | n/a | — | — | 20 or 30 (=duration/60 s) | ✔ |
| Timepoints/trial | — | — | — | 180 (180 × 333.33 ms = 60 s) | ✔ |
| Neural bin size | 10 frames = 333.33 ms | — | fs = 30 Hz | 333.33 ms | ✔ |
| Imaging rate | 30 Hz | — | `ops['fs']` = 30 for all 41 | 30 | ✔ |
| Session duration | "20 minutes" | — | 20 min (14 sessions) / 30 min (27) | same | ~ (paper text is only right for 2 of 6 mice) |
| iscell threshold | 0.5 | `iscell_thr = 0.50` | `iscell[:,1]` min 0.50, `iscell[:,0]` all 1 | inherited | ✔ |
| `time_from_session_start_s` range | — | — | — | [0.15, 1799.82] s | ✔ (= 4.5/30 s to duration−5.5/30 s) |
| `motion_energy_quintile` distribution | — | — | — | [0.200, 0.200, 0.200, 0.200, 0.200] overall **and in every session** | ✔ (by construction) |

**Total-neuron discrepancy**: the Fig. 5B "N cells" counts (285/376/799/728/541/411) do not
match the deposited Suite2p folders (221/370/685/746/541/435); mouse E matches exactly, B is
within 6, and mouse D is *higher* in the data than in the figure. The deposited data were
evidently regenerated with a slightly different Track2p run/curation than the figure. Nothing
in my pipeline can recover the figure's counts — the only curation I apply on top of the
deposited files removes 8 ROIs (2998 → 2990) that have an identically-zero trace on at least
one day. Documented rather than "fixed".

**No data lost in conversion**: `n_bins × 10 == n_frames` for every session and
`n_trials × 180 == n_bins` for every session (both asserted in `cache/sanity_checks.py`), i.e.
every one of the 1 720 000 imaging frames (14×36000 + 27×54000 = 1 962 000) enters exactly one
bin of exactly one trial; nothing is truncated.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1 — Output log verification
`/app/verification_full_out.txt` reports **"Data format is valid, no errors or warnings."**
There are no errors and no warnings to address.

### Check 2 — Independent sanity checks
`/app/cache/sanity_checks.py` (output: `/app/cache/sanity_checks_out.txt`) re-derives
everything **from the original `.npy` files with code that does not import `convert_data.py`**
and compares with `np.allclose` / exact equality. **All checks pass.**

*Neural* (6 random sessions):
- a single randomly chosen `(session, trial, neuron, bin)` value equals the independently
  recomputed `mean(dff[neuron, g*10:(g+1)*10])` — e.g. jm040/2024-05-01 trial 6, neuron 426,
  bin 57: ref 11.431079 vs stored 11.431079 (`np.allclose`, rtol 1e-4).
- the **whole** (n_neurons, 180) trial matrix matches the independent recomputation.
- the stored neuron count equals the number of surviving rows of `F.npy`.

*Input* (6 random samples + all 41 sessions):
- `input[s][trial][0, bin] == ((trial*180 + bin)*10 + 4.5)/30` s.
- across every session the time vector is strictly increasing with a constant step of
  1/3 s, starts at 4.5/30 s and ends at `duration − 5.5/30` s (so trials tile the session
  with no gaps or overlaps and nothing is dropped at the ends).

*Output* (6 random sessions):
- the per-session quintile edges equal `np.percentile(binned_ME, [20,40,60,80])` recomputed
  from `motion_energy_glob.npy` + `tstamps.npy`.
- the per-trial and full-session label vectors match **exactly** (`np.array_equal`).
- per-session class fractions are [0.2, 0.2, 0.2, 0.2, 0.2] to within 0.01.

*Counts*: 41 sessions, 6 subjects, 1090 trials, neurons per subject
{220, 367, 682, 746, 541, 434} = 2990 total, `n_trials == duration//60`,
`n_bins*10 == n_frames`, `len(brain_region_idx[s]) == n_neurons[s]`.

*Paper consistency*: |r(PC1 of converted neural data, converted motion-energy label)| is
0.120 on average for early sessions (≤P11, n=25) and 0.279 for late sessions (>P11, n=16) —
reproducing the steep post-P11 increase of Fig. 7D from the converted data alone.

### Check 3 — Reference code comparison
| Stage | Reference | My script | Match? |
|---|---|---|---|
| (a) Data loading | `load_traces` in `data/load_data.ipynb`: `np.load(session_dir/suite2p/plane0/F.npy)`; `iscell`-based curation + Track2p re-indexing already applied when the folders were written (`track2p/t2p.py:180-280`) | `load_traces()` — identical call; relies on the same pre-applied curation | ✔ identical |
| (b) Neuron/trial filtering | `iscell_thr = 0.50` (`track2p/ops/default.py:21`), tracked-across-all-days | inherited from the deposited files; **plus** 8 ROIs with an all-zero trace on ≥1 day dropped | ✔ + one documented addition (Step 5, decision 8) |
| (c) Temporal alignment | camera triggered by the microscope (Methods); dataset README: dropped camera frames recoverable from `tstamps.npy` | `motion_energy_on_imaging_frames()` reconstructs the imaging-frame index of every camera frame from the timestamp gaps; verified `n_cam + n_gaps == n_frames` for all 6 affected sessions | ✔ follows the README's recommended procedure |
| (d) Binning | Methods: "averaging in bins of 10 consecutive timestamps" for **both** dF/F and behaviour | `bin_mean(..., 10)` applied to dF/F and to motion energy | ✔ identical |
| (e) Neural processing | `F_processing` (`track2p/gui/data_management.py:185`) | `f_processing()` — line-for-line port (same filter order, same `sig_baseline=10`, `win_baseline=60 s`, same `neucoeff=0` as the GUI call site) | ✔ verbatim |
| (f) Input construction | no counterpart in the reference (the paper's decoder has no exogenous input) | bin-centre time from session start, as required by the Decoder Task | n/a — required by the new task |
| (g) Output construction | paper decodes motion energy as a **continuous** ridge-regression target | binned motion energy discretised into per-session quintiles | difference **required** by the Decoder Task ("Output variables must be categorical") |
| (h) Trial structure | paper has no trials; CV splits use consecutive 2-minute blocks | consecutive non-overlapping 60 s blocks | difference **required** by the Decoder Task; the same "consecutive time blocks" philosophy |

Differences and reasons: only (f), (g) and (h), each dictated by the Decoder Task
specification, plus the extra all-zero-ROI removal in (b), which removes rows that carry no
signal at all.

### Check 4 — Key statistics comparison
See the table in Step 9. Everything that is verifiable matches except the per-mouse cell
counts printed in Fig. 5B, which do not match the deposited data themselves (i.e. the
discrepancy exists between the paper and its own data release, not in my conversion). I
verified this by counting rows in `F.npy` directly: 221/370/685/746/541/435 before my
curation, versus 285/376/799/728/541/411 in the figure. Because mouse D is *larger* in the
data than in the figure, no filtering choice on my side could reconcile the two.

### Check 5 — Edge cases
- **Trailing partial trial**: `n_trials = n_bins // 180`. Both 3600 and 5400 bins are exact
  multiples of 180, so no partial trial is ever produced and nothing is discarded; the code
  would drop a partial trial if one existed and reports `n_dropped_bins`.
- **Off-by-one at trial borders**: verified in `cache/sanity_checks.py` by concatenating
  trials and requiring a constant 1/3 s step across every border, and by checking that the
  concatenated output equals the session-level label vector exactly.
- **First/last frames**: bin centres use `(bin*10 + 4.5)/30`, so the first bin is centred at
  0.15 s (not 0 s) and the last ends exactly at the session end.
- **Motion energy frame 0** is identically 0 in every session (frame-difference artefact) and
  is excluded via NaN, so it cannot pull the first bin toward class 0.
- **Dropped camera frames**: all gaps are exactly one frame (max `n_steps` = 2), so no 10-frame
  bin is ever entirely missing; nan-mean handles the ≤1 missing sample per bin, and a fully
  missing bin would be linearly interpolated (count reported).
- **Timestamp jitter without real drops**: 3 jm046 sessions have `len(ME) == n_frames` but a
  few > 1.5× median inter-frame intervals. Using `len(ME) == n_frames` as the primary test
  means these correctly take the identity mapping instead of being spuriously shifted.
- **Camera longer than imaging**: does not occur, but is now guarded (truncate + warning).
- **Sessions with < 2 trials**: asserted impossible (minimum is 20).
- **All-zero / constant neural rows**: removed (Step 5 decision 8); `verify_data_format`
  reports no "all neural data is zero" warnings.
- **NaN/Inf**: none in `F.npy`; asserted absent in every converted trial.

### Issues Found and Resolved
1. *Sanity assertion on the time step failed* (float32 resolution is ~1e-4 s at t ≈ 1800 s,
   which exceeds `np.allclose`'s default tolerance). Resolved by asserting with `atol=1e-3`;
   the stored input remains float32 because the decoder casts everything to float32 anyway.
2. *Robustness*: added the "more camera frames than imaging frames" guard.
Both were re-checked by re-running the full conversion, the verification, and all sanity
checks (all pass).

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

Command: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`
(log: `/app/train_decoder_full_out.txt`).

### Training Progress
- Loss decreasing: **Yes** — 107.7 (epoch 1) → 69.3 (3) → 51.0 (10) → 12.0 (50) →
  2.82 (100) → 1.25 (200), monotone apart from small late-training fluctuations.
  (The large epoch-1 loss is the randomly-initialised weight on the `time` input, whose
  values reach 1800 s; it is driven down within the first few epochs.)

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Notes |
|--------|-------------|--------|--------|-------|
| motion_energy_quintile | 0.575 | 0.309 | 0.200 | 1.55× chance; 5 classes, exactly balanced by construction |

Per-session validation balanced accuracy (`/app/cache/per_session_accuracy.py`, output in
`/app/cache/per_session_accuracy_out.txt`) reproduces the paper's central result:

| Epoch | Mean validation balanced accuracy | n sessions |
|---|---|---|
| early (≤ P11) | 0.285 | 25 |
| late (> P11) | 0.324 | 16 |

Mann–Whitney U early vs late: **p = 0.017** — the same early/late difference the paper reports
for R² (Fig. 7H, "Kruskal–Wallis ... early within – late within: p = 1.6 × 10⁻³"). The best
sessions are jm046 P13/P14 (0.420 / 0.480), which are also the sessions with the highest
PC1–motion correlation, exactly as in Fig. 7C/D.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1 — Accuracy vs chance analysis
| Output | Classes | Chance | Validation | Ratio |
|---|---|---|---|---|
| motion_energy_quintile | 5 | 0.200 | 0.309 | 1.55× |

Above chance, and above the 1.5× threshold. Because 1.55× is not far above that threshold I
investigated it rather than accepting it:

- **Is the signal real, or is it coming from the time input?** Ablation
  (`/app/cache/ablation.py noneural`, replacing the neural data with near-zero noise while
  keeping the time input) gives **0.209 validation balanced accuracy = chance**. So *all* of
  the decoding performance comes from the neural data, and the time input leaks nothing.
- **Is it limited by the neural representation?** Ablation with per-neuron z-scored dF/F
  (`/app/cache/ablation.py zscore`) gives 0.320 vs 0.309 — a change within run-to-run
  variability, and not a processing step the reference performs. Dividing by F0 was *worse*
  on the sample (0.280 vs 0.325). So the reference processing is not costing accuracy.
- **Is the ceiling intrinsic?** Yes, and the paper says so. The paper's own same-day decoding
  of this exact variable gives R² ≈ 0 for the ≤P11 sessions (Fig. 7C/H) — and 25 of the 41
  sessions are ≤P11. Splitting my accuracies the same way (0.285 early vs 0.324 late,
  p = 0.017) reproduces that structure: the modest pooled number is dominated by sessions in
  which the paper also finds essentially no behavioural-state representation.
- Additionally, the five classes are quintiles of a heavy-tailed variable, so classes 0–2 all
  lie within the narrow "no movement" band of motion energy (see panel 5 of the processing
  plots). Discriminating them is intrinsically harder than discriminating "moving" from
  "still"; this is a property of the required 5-quintile discretisation, not of the
  conversion.

### Check 2 — Accuracy comparison to paper
| Variable | My accuracy | Paper's reported value |
|---|---|---|
| motion energy (5 quintiles) | 0.309 balanced accuracy (chance 0.200); 0.285 early / 0.324 late | **No accuracy is reported anywhere in the paper.** The paper decodes motion energy as a *continuous* variable with ridge regression and reports R² (Fig. 7C, G, H): ≈0 for early (≤P11) sessions, ≈0.3–0.6 for late (>P11) sessions. |

The paper's quantity (R² of a continuous regression) is not convertible into a 5-class
balanced accuracy, so the comparison is necessarily qualitative. Every qualitative prediction
the paper makes about this decoding does hold in my converted data:
1. late (>P11) decoding is significantly better than early (≤P11) — mine: 0.324 vs 0.285,
   p = 0.017 (paper Fig. 7H);
2. the mice/sessions with the highest PC1–motion correlation are the best-decoded ones
   (jm046 P13/P14) (paper Fig. 7C/D);
3. early sessions are close to uninformative (paper: R² ≈ 0; mine: 0.285, i.e. the weakest
   sessions sit at 0.21–0.25, barely above the 0.20 chance level).

### Check 3 — Train vs validation gap
Training 0.575 vs validation 0.309 = 1.86×, above the 1.5× flag, so I checked for leakage and
for a conversion cause:
- **No leakage is possible through the input**: the no-neural ablation is exactly at chance,
  so the time input carries no trial identity.
- **No leakage through trials**: trials are disjoint 60 s blocks of frames; every imaging frame
  appears in exactly one trial (asserted), and `train_validate_decoder` splits by trial.
- **The gap is a property of the decoder, not the data**: the model fits a separate
  `100 × n_neurons` projection per session (74 600 free parameters for jm039) on 5400
  timepoints, full-batch, for 200 epochs with only `l1_weight=1e-4`. Overfitting at that
  parameter-to-sample ratio is expected, and I cannot change the decoder.
- The one remaining data-side consideration is that neighbouring 60 s trials are temporally
  adjacent and motion energy is autocorrelated, so a random trial split is not as strict as the
  paper's consecutive-block split. The decoder is memoryless and per-timepoint, so it can only
  exploit this through slow drift; and the trial split is performed by `train_decoder.py`,
  not by the conversion.

### Additional debugging steps performed
1. **Output values verified against raw data for specific trials** — Check 2 of Step 10:
   exact label match for randomly chosen trials in 6 sessions, recomputed from
   `motion_energy_glob.npy`.
2. **Temporal alignment verified by cross-correlation** — binned population dF/F vs binned
   motion energy peaks at a lag of −1 to −4 bins (0.3–1.3 s), i.e. neural activity *follows*
   motion by about one GCaMP8m rise time. A conversion bug (e.g. an off-by-one-trial shift)
   would put the peak at tens of seconds or destroy the peak entirely.
3. **Output variation** — every session is exactly 20 % per class by construction; no session
   is dominated by one class.
4. **Neural filtering** — confirmed that `iscell > 0.5` + tracked-across-all-days is already
   applied in the released files and that my only addition removes 8 all-zero ROIs.
5. **Processing matches the reference** — `f_processing` is a verbatim port; see Step 10
   Check 3.

### Issues Found and Resolved
No further issues were found in this pass; no changes to `convert_data.py` were required, so
no re-run was needed beyond the one already performed at the end of Step 10.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, loading instructions, format spec, key stats)
- [x] `cache/` folder created with the investigation scripts and their outputs
- [x] `cache/README_CACHE.md` documents every cached file
- [x] All required deliverables present:
      `CONVERSION_NOTES.md`, `convert_data.py`, `converted_data.pkl`, `sample_data.pkl`,
      `README.md`, `conversion_sample_out.txt`, `verification_sample_out.txt`,
      `train_decoder_sample_out.txt`, `conversion_full_out.txt`, `verification_full_out.txt`,
      `train_decoder_full_out.txt`
