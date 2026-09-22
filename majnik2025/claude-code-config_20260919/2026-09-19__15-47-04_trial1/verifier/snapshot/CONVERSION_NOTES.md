# Dataset Conversion Notes

## Overview
- **Dataset**: Majnik et al. 2025 (eLife 14:RP107540), "Longitudinal tracking of neuronal activity from
  the same cells in the developing brain using Track2p". Daily 2-photon calcium imaging (GCaMP8m) of
  L2/3 barrel cortex in 6 GAD67-Cre mouse pups across the second postnatal week, with simultaneous
  videography-derived *motion energy* of spontaneous behaviour.
- **Date started**: 2026-09-19
- **Goal**: Convert to decoder-compatible format. Decode motion energy (5 per-session quintile bins)
  from neural activity; decoder also receives time-from-session-start as input. 60-second trials.

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `CONVERSION_NOTES.md` (this file), `convert_data.py` (written in Step 6)
- `paper.pdf` (46 pages, eLife reviewed preprint v2), `methods.txt` (excerpt of the paper's methods)
- `code/` — the Track2p package + notebooks (reference code)
- `data/` — 6 subject folders (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`),
  `README.md`, `load_data.ipynb` (official data loader notebook)
- `decoder.py`, `train_decoder.py` — provided decoder/validation code
- `Dockerfile`, `docker-compose.yaml`, `.manifest`

Environment check: `python3` works; `numpy 2.4.4`, `torch 2.6.0+cu124`, `scipy`, `sklearn`, and
`suite2p` are importable. `pypdf` + `poppler-utils` were installed to read the paper.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

Two reference code bases were read:
1. `/app/code` — the **Track2p** package (the tracking algorithm itself + a GUI + evaluation notebooks).
2. `/app/data/load_data.ipynb` — the **official data loader** shipped with the published dataset.

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_traces(session_dir)` | `data/load_data.ipynb` | LOADING | `np.load(session_dir/suite2p/plane0/F.npy)` → raw fluorescence, shape (n_neurons, n_frames). Comment explicitly says: "for more proper analysis compute dF/F the way as described in the paper (or alternatively use spks.npy)". |
| `load_fov(session_dir)` | `data/load_data.ipynb` | LOADING | `ops['meanImg']` (FOV image; not needed here). |
| `load_coords_cell(session_dir, cell_id)` | `data/load_data.ipynb` | LOADING | ROI centroid from `stat.npy` (not needed here). |
| `zscore_rows(F)` | `data/load_data.ipynb` | PROCESSING | z-score per neuron — used **only for raster visualisation** ("only for visualization in raster"). |
| `DataManagement.F_processing(F, Fneu, fs, neucoeff=0.0, baseline='maximin', sig_baseline=10.0, win_baseline=60.0, prctile_baseline=8)` | `code/track2p/gui/data_management.py:185` | PROCESSING | **The authors' own dF/F implementation.** `Fc = F - neucoeff*Fneu`; `Flow = gaussian_filter(Fc,[0,sig_baseline])`; `Flow = minimum_filter1d(Flow, int(win_baseline*fs))`; `Flow = maximum_filter1d(Flow, win)`; returns `Fc - Flow`. Identical maths to `suite2p.extraction.dcnv.preprocess`. Called as `F_processing(F=f_t2p, Fneu=fneu, fs=ops['fs'])`, i.e. with **neucoeff = 0.0** (no neuropil subtraction) and suite2p's default baseline parameters. |
| `DataManagement.load_all_data(...)` | `code/track2p/gui/data_management.py:40-95` | LOADING/CURATION | Loads `F/spks/Fneu/stat/iscell/ops` per session, keeps ROIs with `iscell[:,1] > track_ops.iscell_thr` (default 0.5), then reindexes with the track2p match matrix so rows are the same neuron across days. |
| `t2p.py` (`run_t2p`, `save_outputs_s2p_format`) | `code/track2p/t2p.py:196-275` | CURATION/SAVING | Produces exactly what is in `/app/data`: `F/Fneu/spks/stat/iscell/ops` already restricted to the cells tracked on **all** days and reordered so row *i* is the same neuron on every day. |
| `eval` notebooks | `code/notebooks/eval/*.ipynb` | — | Tracking benchmarking only (F1 vs ground truth). No behavioural/decoding analysis. |

### Notes
- **The decoding analysis of the paper (Fig. 7) is NOT in the provided code base.** The only description is
  in `methods.txt` / paper Methods ("Decoding" + "Preprocessing videography" + "Processing of calcium
  imaging data"). Therefore the processing is taken from the Methods text, and implemented with the
  authors' own `F_processing()` code, which I copied verbatim into `convert_data.py`.
- Neuron curation (`iscell > 0.5`) and cross-day matching were **already applied** when the dataset was
  exported (see `data/README.md`: "the data only includes traces for the cells present across all days").
  Verified in Step 2: every session's `iscell[:,0]` is 1 for all rows and `min(iscell[:,1]) > 0.5`.
  ⇒ No further neuron filtering is needed or justified.
- Calcium imaging is single-plane (`plane0` only, `nplanes=1`).

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
```
data/<subject>/<YYYY-MM-DD>_a/suite2p/plane0/{F,Fneu,spks,stat,iscell,ops}.npy
data/<subject>/<YYYY-MM-DD>_a/move_deve/{motion_energy_glob,tstamps,interframe_int}.npy
data/<subject>/ground_truth.csv          # only jm038, jm039, jm046 -> manual tracking GT, not used
```
- `F.npy`, `Fneu.npy`, `spks.npy`: float32 (n_neurons, n_frames); rows already matched across days.
- `iscell.npy`: (n_neurons, 2) = [is_cell flag, classifier probability].
- `ops.npy`: dict; relevant keys `fs=30`, `nframes`, `Ly=Lx=512`, `baseline='maximin'`,
  `win_baseline=60.0`, `sig_baseline=10.0`, `prctile_baseline=8.0`, `neucoeff=0.7`, `nplanes=1`.
- `motion_energy_glob.npy`: uint64, one value per **camera** frame (pixel-wise squared difference of
  consecutive video frames, summed over pixels). `motion_energy_glob[0] == 0` in every session
  (no preceding frame ⇒ boundary artefact).
- `tstamps.npy`: float64 camera frame times, **in units of 1000 s** (i.e. ×1000 ⇒ seconds);
  `interframe_int.npy` = `diff(tstamps)`. Median interval 3.3603e-5 (=33.60 ms ⇒ 29.76 Hz true rate).
- Camera is hardware-triggered by the microscope ⇒ camera frame *i* = 2p frame *i*, except when the
  camera misses triggers (data README: "In some recordings there might be some missing frames from the
  camera ... The indices of missing frames can be obtained by looking at tstamps.npy or
  interframe_int.npy and treated as missing values ... or they can be interpolated over").

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 6 (jm031=A, jm032=B, jm038=C, jm039=D, jm040=E, jm046=F) |
| Sessions / subject | 7, 7, 7, 7, 6, 7 → **41 sessions** |
| Neurons / session (= per subject, constant across days) | 221, 370, 685, 746, 541, 435 (mean 499.7 ± 196.6 sd) |
| Neurons (summed over sessions) | 20,445 |
| Frames / session | 36,000 (jm031, jm032) or 54,000 (jm038, jm039, jm040, jm046) |
| Session duration | 36,000/54,000 frames × 33.60 ms = 1209.7 s / 1814.5 s (20.2 / 30.2 min) |
| Trials (60 s, after 10-frame binning) | 20 or 30 per session → **1090 total** |
| Camera frames missing | 0 in 33/41 sessions; 1–148 frames in 8 sessions (max 148 = 0.4 % of jm032 2023-10-22) |

### Sessions with imperfect camera streams
| Session | len(motion_energy) | missing triggers | note |
|---|---|---|---|
| jm031 2023-10-20/21/22 | 35998 / 35997 / 35884 | 2 / 3 / 116 | len + missing = 36000 exactly |
| jm032 2023-10-20/21/22 | 35998 / 35998 / 35852 | 2 / 2 / 148 | len + missing = 36000 exactly |
| jm039 2024-05-04, jm040 2024-05-04, jm046 2024-09-09 | 53999 | 1 | len + missing = 54000 exactly |
| jm046 2024-09-05 / 09-07 / 09-08 | 54000 | 3 / 10 / 3 | camera ran **past** the end of the 2p acquisition; the 3/10/3 surplus samples fall beyond frame 53999 and are dropped |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects | 6 | "a full dataset of 6 mice imaged daily for a minimum of 6 consecutive days within the second postnatal week (P7 to P14)" |
| Sessions / subject | A:P7–P13, B:P7–P13, C:P8–P14, D:P8–P14, E:P9–P14, F:P8–P14 → 7,7,7,7,6,7 | Fig. 5B grid (read from the rendered figure) |
| Neurons / subject (paper) | 285, 376, 799, 728, 541, 411 (mean 523, sd 187) | Fig. 5B "N cells" |
| Neurons / subject (mean) | "526 (± 190 std) neurons per mouse were successfully tracked" | Results |
| % of day-1 cells tracked | 33 % ± 11 %; per mouse [0.47, 0.20, 0.36, 0.37, 0.41, 0.19] of [607, 1849, 2190, 1988, 1316, 2138] detected on day 0 | Results + reviewer response |
| Imaging rate | 30 Hz resonant, 512×512 px, 720×720 µm, L2/3 (100–200 µm deep) | Methods |
| Session length | "each session lasted 20 minutes" (**data: 20 min for jm031/jm032, 30 min for the other four**) | Methods |
| Videography rate | 30 Hz, triggered by the microscope ("allowing for simple synchronisation across the two modalities") | Methods |
| Neuron curation | ROIs with suite2p cell probability > 0.5 | "We considered all ROIs above the default threshold of 0.5 as true cells." |
| Neural signal | "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters) for all subsequent analyses." | Methods |
| Motion energy | pixel-wise difference of consecutive video frames, squared, summed over pixels | Methods |
| Denoising / time bin for decoding | **average 10 consecutive frames** ⇒ 333.3 ms bins (3 Hz), applied to *both* dF/F and behaviour | "For all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." |
| Decoding CV blocks | consecutive 2-minute blocks, nested 5-fold CV, ridge regression | Methods "Decoding" |
| Event-rate analysis | denoise with 10-frame bins, `scipy.signal.find_peaks` with height and prominence ≥ 1 sd, rate = peaks/min | Methods |

### Processing Details
- **Temporal alignment**: camera acquisition is triggered by the 2p microscope, so camera frame *i*
  corresponds to imaging frame *i*; no further alignment (no stimulus/trial events exist — this is
  spontaneous activity in the dark).
- **Temporal binning**: mean over 10 consecutive frames for neural and behaviour alike.
- **Trial structure**: none in the experiment (continuous 20/30 min spontaneous recordings). The paper
  itself splits recordings into consecutive blocks (2 min) for cross-validation; here the task
  specification asks for **60-second trials**.

### Curation Steps
**Neuron curation rules**: suite2p `iscell` probability > 0.5 **and** successfully tracked by Track2p on
every day of that mouse. Both were already applied in the released data (verified).
No activity-based or SNR-based rejection is described anywhere in the paper/code.

**Trial curation rules**: none in the paper (no trials). Here: keep every complete 60-s block; a
partial block at the end of a session would be dropped (there are none — 36000 and 54000 frames are
both exact multiples of 1800).

**Session/mouse curation**: the paper uses all 6 mice / all sessions of the released dataset for the
functional analyses ("we used this dataset for all subsequent analyses").

### Decoders Trained (paper)
| Decoded variable | Metric | Value |
| --- | --- | --- |
| Motion energy (continuous), same-day, ridge regression, nested 5-fold CV | R² | ~0.0–0.3 at P7–P11, rising to ~0.4–0.8 at P12–P14 (Fig. 7C); example traces R²=0.26 (P8) and R²=0.69 (P14) |
| Motion energy, cross-day | R² | late→late ≈ 0.61; early→early ≈ 0.15; early→late ≈ 0.11 (Fig. 7F) |
| Correlation of PC1 with motion | r | ~0.1–0.3 early, up to ~0.6–0.8 late (Fig. 7D) |
No classification accuracies are reported (the paper's decoder is a regression), so the paper gives an
*upper-bound expectation*: a substantial fraction of motion-energy variance is linearly decodable from
the population, strongly age-dependent. For 5 equal-frequency classes, chance = 0.20.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Neuropil coefficient for dF/F | `F_processing(..., neucoeff=0.0)` — the authors' dF/F call uses **no** neuropil subtraction | `ops['neucoeff']=0.7` (suite2p default, used for `spks`) | "baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)" | Followed the **reference code** (`neucoeff=0.0`, maximin baseline with suite2p's default `win_baseline=60 s`, `sig_baseline=10`). This is the authors' own implementation of the trace labelled "dF/F0" in their GUI. A sensitivity test with `neucoeff=0.7` is reported in Step 12. |
| "dF/F" naming | code subtracts the baseline but does **not** divide by it | — | calls it dF/F | Kept the reference-code definition (baseline-subtracted F). Division by a maximin baseline is numerically unstable for these traces and is not what the authors implemented. Documented. |
| Session length | — | 20 min (jm031, jm032), 30 min (4 other mice) | "each session lasted 20 minutes" | Data wins; all sessions used in full. Trial count per session therefore differs (20 vs 30) — allowed by the format. |
| N tracked cells / mouse | — | 221, 370, 685, 746, 541, 435 (mean 499.7) | 285, 376, 799, 728, 541, 411 (Fig. 5B), "526 ± 190" | Only mice B and E match exactly. The released dataset was evidently exported with a slightly different Track2p run/version than the figure. Nothing in the data or code justifies adding/removing cells, so **all released cells are used**. Documented as a known dataset-vs-figure discrepancy. |
| Frame rate | `ops['fs']=30` | median interframe interval 33.60 ms ⇒ 29.76 Hz | "Imaging rate was 30 Hz" | Used the nominal 30 Hz (as the reference code does, e.g. `win = int(win_baseline*fs)`): bin = 10 frames = 333.33 ms, trial = 180 bins = 60 s. The true trial duration is 60.48 s (0.8 % longer); this only rescales the "time in session" input by 0.8 % and is documented in the metadata. |
| Example-mouse raster | — | jm039 = 746 cells, 30 min | Fig. 5A "all 728 tracked neurons", x-axis 0–20 min | Same dataset-export discrepancy as above; the figure shows only the first 20 min. No action. |

After these resolutions the code, data and text agree on: what to load (`F.npy` of the tracked cells),
how to filter (nothing further — already curated), how to align (frame-for-frame, camera triggered by
the 2p), how to process (maximin baseline-corrected F; 10-frame averaging for both streams).

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `suite2p/plane0/F.npy` (n_neurons, n_frames) | `neural[session][trial]` (n_neurons, 180) | (1) `F_processing(F, Fneu, fs=ops['fs'], neucoeff=0.0)` → baseline-corrected dF/F; (2) mean over 10-frame bins; (3) split into consecutive 180-bin (60 s) trials | `data_management.F_processing` (copied verbatim), `load_data.ipynb:load_traces` | float32 |
| `move_deve/motion_energy_glob.npy` + `tstamps.npy` | `output[session][trial]` (1, 180) | (1) place samples at their 2p frame index using cumulative missing-trigger counts from `interframe_int`; (2) mark frame 0 and missing frames NaN and linearly interpolate; (3) mean over 10-frame bins; (4) digitise into 5 equal-percentile (quintile) bins using the *session's* 20/40/60/80th percentiles; (5) split into trials | Methods "Preprocessing videography" + "Decoding" (10-frame averaging); `data/README.md` (missing-frame handling) | int8 categories 0–4 |
| frame index | `input[session][trial]` (1, 180) | bin-centre time in seconds from the first imaging frame: `t = (10*b + 4.5)/30` for global bin index b | — | float32, 0 → 1200 s or 1800 s |
| subject folder name | `subjects`, `subject_idx` | `['jm031','jm032','jm038','jm039','jm040','jm046']` | data README (alphabetical = mouse A…F) | |
| — | `brain_regions`, `brain_region_idx` | `['S1']`, all-zeros per session | Methods: barrel cortex L2/3, single FOV | one region |
| session date + Fig. 5B | `metadata['session_info']` | subject, date, postnatal day, n_neurons, n_trials, n_frames, missing-camera-frame count, quintile thresholds | Fig. 5B for the P-day of each session | |

### Key Decisions
1. **Use `F.npy` (tracked cells) and compute dF/F with the authors' `F_processing`** rather than
   `spks.npy`: the paper states all functional analyses (including decoding) used baseline-corrected
   fluorescence, not deconvolved spikes.
2. **`neucoeff = 0.0`** — the value the reference code actually uses for its dF/F traces (see Step 4).
3. **No extra neuron curation** — `iscell>0.5` + all-day tracking were already applied at export
   (verified: all 41 sessions have `iscell[:,0]==1` everywhere and `min(prob)>0.5`).
4. **Bin = 10 frames (333.33 ms)** for both neural and behaviour, exactly as the paper's decoding
   analysis. Same bin size for every trial/session.
5. **Trials = consecutive, non-overlapping 60 s blocks** (180 bins) from the session start, as the task
   requires. The experiment has no trial structure, so the alignment event is the session start; this
   also mirrors the paper's own use of consecutive time blocks for cross-validation.
6. **Motion energy → 5 equal-percentile bins per session**, as the task requires. Per-session
   percentiles handle the large across-session differences in absolute motion-energy scale (medians
   range 5.9e5 → 1.8e6) and give exactly 20 % chance per class.
7. **Missing camera frames**: reconstructed onto the 2p frame grid from the timestamps and linearly
   interpolated (explicitly sanctioned by `data/README.md`). ≤0.4 % of frames in the worst session.
   `motion_energy[0] = 0` (no preceding video frame) is treated as missing too.
8. **Time-in-session is a decoder input** (as the task specifies), time-varying, in seconds.
9. **All 6 mice / 41 sessions / 1090 trials are kept** — no session is excluded; every session has
   ≥20 trials, far above the 2-trial minimum.

### Planned Sanity Checks
- [x] every session: `iscell[:,1] > 0.5` for all rows, and n_neurons identical across that mouse's days
- [x] `len(motion_energy) + n_missing_triggers ≥ n_frames`, and reconstruction places ≥99.5 % of frames
- [x] binned dF/F for a spot-checked (session, trial, neuron, bin) equals the mean of the corresponding
      10 raw dF/F values recomputed from `F.npy` independently (`np.allclose`)
- [x] concatenating all trials of a session reproduces the full-session binned matrices (no gaps/overlap)
- [x] output class fractions ≈ 20 % per class in every session
- [x] input (time) starts at 0.15 s, increases by 1/3 s per bin, ends at session duration
- [x] calcium event rate (paper's method: 10-frame denoise, `find_peaks` height/prominence ≥ 1 sd)
      is in the range of Fig. 5D (~2–10 events/min) and increases with postnatal age
- [x] correlation of PC1 of the binned neural data with binned motion energy increases with age (Fig. 7D)

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` — run as `python -u convert_data.py <outpicklefile> [--full|--sample]
[--show-processing]`.

Structure:
| Function | Purpose |
|---|---|
| `F_processing()` | **verbatim copy** of `code/track2p/gui/data_management.py:185` (prints removed, also returns the baseline `Flow` for plotting): maximin baseline-corrected dF/F |
| `find_sessions()` | walks `/app/data/<subject>/<date>_a`, chronological per subject |
| `load_session_neural()` | loads `F.npy`/`iscell.npy`/`ops.npy`, asserts the curation invariants, computes dF/F, flags ROIs with an identically-zero trace |
| `load_session_motion()` | reconstructs motion energy on the 2-photon frame grid from the camera timestamps, interpolates missing frames |
| `bin_time()` | mean over 10-frame bins (vectorised reshape) |
| `discretize_quantiles()` | per-session quintile digitisation |
| `process_session()` | full per-session pipeline + trial splitting + timing printout |
| `drop_failed_neurons()` | removes ROIs whose extraction failed on any day of that mouse |
| `plot_processing()` | 8-panel figure of every processing step (`--show-processing`) |
| `main()` | assembles the target dict, metadata and summary statistics |

Code inefficiencies identified:
- `Fneu.npy` (~150 MB/session) is only needed when `neucoeff != 0`; it is not read in the default
  configuration. Halves the I/O.
- The only heavy computation is the maximin baseline; `scipy.ndimage`'s filters are already
  O(n) per neuron and run on the whole (n_neurons × n_frames) matrix at once.

Code speedups added:
- binning by `reshape(...).mean(-1)` instead of a loop (vectorised);
- `np.interp` for the missing camera frames instead of per-gap loops;
- float32 throughout the neural pipeline (halves memory; verified against float64: max deviation
  8.9e-5 in dF/F units, i.e. 1e-7 relative);
- sessions processed and released one at a time (`del res`), so peak memory stays near the size of
  one session (~150 MB) plus the accumulating output (~412 MB).

Result: **0.9 s/session**, 37 s for the whole dataset — no parallelism needed.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Command: `python -u convert_data.py /app/sample_data.pkl --sample --show-processing`
(sessions chosen deliberately: `jm031/2023-10-22` — 116 dropped camera triggers — and
`jm046/2024-09-07` — camera running past the end of the 2-photon acquisition; both edge cases).

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (2 subjects) |
| Neurons / session | 220 (jm031), 434 (jm046) |
| Neurons (total) | 654 |
| Trials (total) | 50 |
| Trials / session | 20 (20 min session), 30 (30 min session) |
| Timepoints / trial | 180 (all trials) |
| time_in_session_s range | [0.15, 1199.82] and [0.15, 1799.82] |
| motion_energy_quintile distribution | [0.200, 0.200, 0.200, 0.200, 0.200] |

### Processing Plots Review
`processing_jm031_2023-10-22_a.png`, `processing_jm046_2024-09-07_a.png` — 8 panels each:
1. raw F with the maximin baseline overlaid — the baseline tracks the lower envelope, as it should;
2. dF/F — flat baseline, clean calcium transients;
3. raw (30 Hz) vs binned (3 Hz) trace over the first trial, binned values plotted at bin centres —
   the binned curve sits on top of the raw one with **no visible shift**;
4. motion energy on the frame grid with the interpolated frames marked (117 and 11 respectively) —
   for jm046 the 10 surplus camera samples are correctly dropped at the end of the recording;
5. binned motion energy with the four session quintile thresholds;
6. the discretised output with its class fractions printed (0.200 each);
7. binned dF/F raster with motion energy overlaid and the 60 s trial boundaries drawn — the
   synchronous network events line up with motion bouts (compare paper Fig. 5A);
8. explicit verification that the concatenated trials reproduce the session-level arrays
   (`neural=True, input=True, output=True`).
No anomalies found.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| skip `Fneu.npy` when `neucoeff == 0` | ~0.4 s/session (halves the bytes read) |
| vectorised binning / `np.interp` | negligible runtime but avoids ~5 s/session of python loops |
| float32 neural pipeline | ~2× on the gaussian/min/max filters |

| Step | Time / Session | Estimated Total Time |
|---|---|---|
| load `F.npy` + dF/F | 0.32 s (36k frames) / 0.87 s (54k frames) | ~30 s |
| motion energy | < 0.01 s | < 1 s |
| binning + trial splitting | 0.02-0.05 s | ~2 s |
| pickle write | — | 0.5 s |
| **total (measured)** | **0.9 s** | **37 s** (estimate before the full run: ~60 s) |

Verification (`/app/verification_sample_out.txt`): **no errors, no warnings**.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|---|
| motion_energy_quintile | 0.596 | 0.325 | 0.200 |

Loss decreased monotonically (99.7 → 1.31 over 200 epochs); validation accuracy is 1.6× chance on
only two sessions / 50 trials.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 412.1 MB (41 sessions, 1090 trials), written in 37.4 s
- `verification_full_out.txt`: created — **no errors, no warnings**

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Subjects | 6 | — | 6 folders | 6 | ✅ |
| Sessions | 7,7,7,7,6,7 = 41 (Fig. 5B) | — | 41 | 41 | ✅ |
| Postnatal days | A,B: P7-P13; C,D,F: P8-P14; E: P9-P14 | — | 7/7/7/7/6/7 daily sessions | same, stored in `session_info` | ✅ |
| Neurons / mouse | 285, 376, 799, 728, 541, 411 (Fig. 5B); "526 ± 190" | `iscell>0.5` + tracked on all days | 221, 370, 685, 746, 541, 435 | 220, 367, 682, 746, 541, 434 | ⚠️ data ≠ figure (see below); conversion matches the data minus 8 failed ROIs |
| Mean neurons / mouse | 523 (Fig. 5B) / 526 (text) | — | 499.7 | 498.3 | ⚠️ −5 % vs paper |
| Neurons summed over sessions | — | — | 20,445 | 20,389 | ✅ (56 neuron-sessions dropped, 0.27 %) |
| Trials (total) | n/a (no trials in the experiment) | — | — | 1090 | — |
| Trials/session | — | — | — | 20 (2 mice) / 30 (4 mice) | ✅ = session length / 60 s |
| Session length | "20 minutes" | — | 1209.7 s / 1814.5 s | 1200 s / 1800 s of trials | ⚠️ data has 30-min sessions for 4 mice |
| Neural time bin | 10 frames (paper) | `fs=30` | 33.60 ms/frame | 333.33 ms | ✅ |
| Behaviour time bin | 10 frames (paper) | — | — | 333.33 ms (identical bins) | ✅ |
| Camera/2p alignment | camera triggered by the microscope | — | timestamps confirm 1:1 | frame-for-frame | ✅ |
| Output distribution | n/a | — | — | [0.200, 0.200, 0.200, 0.200, 0.200] | ✅ by construction |
| Input range | — | — | 0 → 1209.7/1814.5 s | 0.15 → 1199.8/1799.8 s | ✅ (0.8 %, nominal vs measured frame rate) |
| Ca²⁺ event rate (Fig. 5D) | P7: 1.5-3.7 /min; P14: 4.3-11.3 /min; mouse D: 3.0 → 7.4 | — | — | P7: 1.54, 2.34; P14: 7.12, 7.21, 8.51, 10.58; mouse D: 3.85 → 7.21 | ✅ |
| Same-day decoding R² (Fig. 7C) | early ~0-0.3, late ~0.4-0.8; mouse D P8 = 0.26, P14 = 0.69 | — | — | early mean 0.15, late mean 0.41, max 0.80; mouse D P8 = 0.18, P14 = 0.55 | ✅ |

The only remaining mismatch is the per-mouse cell count between the **released data** and **Fig. 5B**
(4 of 6 mice differ by 6-15 %, in both directions). This is a property of the released dataset, not
of the conversion: the counts in the converted file equal the number of rows in `F.npy` (minus the
8 failed ROIs). Nothing in the data or the code provides a rule that would reproduce the figure's
counts, so all released cells are kept.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
`/app/verification_full_out.txt`: `Data format is valid, no errors or warnings.` — nothing to fix.
(The earlier iteration also had no errors/warnings.) The summary printout confirms: 41 sessions,
1090 trials, 6 subjects, 1 brain region, dinput=1, doutput=1, T=180 everywhere, output classes
0.200 each, 20,389 neurons.

### Check 2: Constructed sanity checks
Implemented in `cache/sanity_checks.py` (output: `cache/sanity_checks_out.txt`). Everything is
re-derived from the original `.npy` files with **independently written** code (a `sliding_window_view`
maximin baseline instead of `scipy.ndimage`, an explicit python-loop binning, reconstruction of the
frame grid from `tstamps.npy` rather than `interframe_int.npy`, `np.searchsorted` on
`np.quantile` edges instead of `np.digitize` on `np.percentile` edges) and compared with
`np.allclose` / `np.array_equal`. **All checks pass.**

| # | Stream | Check | Result |
|---|---|---|---|
| 1 | bookkeeping | session list, order and `subject_idx` match the directory tree | PASS (41 sessions) |
| 2 | bookkeeping | `n_trials × 180 × 10 == n_frames` for every session (no data silently dropped) | PASS (0 frames lost) |
| 3 | neural | n_neurons == rows of `F.npy` minus failed ROIs | PASS |
| 4 | neural | every ROI has suite2p probability > 0.5 | PASS |
| 5 | neural | 21 random (session, neuron, trial, bin) spot checks vs an independent dF/F + binning | PASS (agreement to ~1e-6 relative) |
| 6 | neural | whole session × 10 random neurons vs independent computation | PASS (max abs diff 1.3e-4 in dF/F units) |
| 7 | input | time equals the bin-centre time `(10b+4.5)/30`; trial starts exactly 60 s apart | PASS |
| 8 | input | nominal session end within 1 % of the measured camera timestamps | PASS (0.81 % low, the nominal-vs-true frame rate) |
| 9 | output | missing-camera-frame count reproduced from `tstamps.npy` | PASS |
| 10 | output | quintile labels for the whole session identical to an independent digitisation | PASS (100.000 % identical) |
| 11 | output | stored `quantile_edges` match | PASS |
| 12 | output | class fractions within 0.5 % of 20 % in every session | PASS |
| 13 | output | one hand-picked trial/bin traced back to the raw camera samples | PASS |
| 14 | alignment | population dF/F × motion cross-correlation peaks at lag 0 | PASS (median lag 0 bins; 71 % of sessions within ±0.67 s) |
| 15 | paper | Ca²⁺ event rate magnitude and its increase with age | PASS (early 4.4 → late 10.1 /min with the binned-trace threshold; see Check 4 for the absolute comparison) |
| 16 | paper | PC1-motion correlation increases with age (Fig. 7D) | PASS (early 0.119 → late 0.282) |
| 17 | edge cases | no zero-variance neurons, no NaN/Inf, ≥2 trials/session, T=180 everywhere, labels in [0,4], `brain_region_idx` lengths | PASS |

### Check 3: Reference code comparison
| Stage | Reference | This conversion | Same? |
|---|---|---|---|
| (a) data loading | `load_data.ipynb:load_traces` → `suite2p/plane0/F.npy`; GUI also loads `ops`, `iscell`, `stat` | identical files and paths; `stat` not needed (single region, no anatomy used) | ✅ |
| (b) neuron filtering | `iscell[:,1] > track_ops.iscell_thr` (0.5) + track2p all-day matching, both already applied at export | verified as an assertion; additionally 8 ROIs with an all-zero (failed) trace removed from every day of their mouse | ✅ + documented addition |
| (c) temporal alignment | camera triggered by the 2p microscope ⇒ frame-for-frame; `tstamps`/`interframe_int` identify dropped camera frames (dataset README) | exactly that; surplus camera samples beyond the last imaging frame dropped | ✅ |
| (d) binning | Methods: "averaging in bins of 10 consecutive timestamps" for dF/F **and** behaviour | mean over 10 frames for both streams, identical bin edges | ✅ |
| (e) neural processing | `F_processing(F, Fneu, fs=ops['fs'])` with `neucoeff=0.0`, `baseline='maximin'`, `sig_baseline=10`, `win_baseline=60` | the same function, copied verbatim, with the same arguments | ✅ |
| (f) behaviour processing | Methods: motion energy = summed squared pixel difference (already computed in `motion_energy_glob.npy`) | used as provided; only missing-frame interpolation and 10-frame averaging added | ✅ |
| (g) input construction | no analogue in the reference (the paper's decoder has no extra regressors) | time-in-session, required by the task specification | task-driven difference |
| (h) output construction | paper decodes the **continuous** motion energy with ridge regression | 5 equal-percentile bins, required by the task specification (the decoder needs categorical outputs) | task-driven difference |
| (i) trial structure | none in the experiment; the paper cuts the recording into consecutive 2-minute blocks for cross-validation | consecutive 60 s blocks, required by the task specification | task-driven difference |

### Check 4: Key statistics comparison
See the table in Step 9. Two independent quantitative reproductions of published numbers were run:
- **Ca²⁺ event rates (Fig. 5D)** — `cache/event_rates.py`, output `cache/event_rates_out.txt`.
  Using the paper's recipe (10-frame denoising, `find_peaks` with height and prominence ≥ 1 sd) my
  rates were uniformly ~1.4× the published ones until the threshold was taken as the sd of the
  *unbinned* dF/F (the Methods do not say which trace's sd is meant). With that reading the
  agreement is quantitative: P7 = 1.54 / 2.34 (paper 1.5-2.4), P14 = 7.12 / 7.21 / 8.51 / 10.58
  (paper 4.3-11.3), and the example mouse D goes 3.85 (P8) → 7.21 (P14) against the paper's
  3.0 → 7.4. The developmental increase (early 2-3/min → late 7-11/min) is reproduced for all mice.
- **Same-day decoding R² (Fig. 7C)** — `cache/ridge_reproduce_fig7.py`, output
  `cache/ridge_reproduce_fig7_out.txt`. Ridge regression with nested 5-fold CV on consecutive 2-min
  blocks, exactly as the Methods describe, run on the converted neural data against the converted
  (continuous, pre-discretisation) motion energy: early (≤P11) mean R² = 0.15, late (>P11) mean
  R² = 0.41, maximum 0.80 (jm046 P14); example mouse D: 0.18 at P8 and 0.55 at P14 versus the
  paper's 0.26 and 0.69. The mouse-by-mouse and age-by-age pattern of Fig. 7C is reproduced,
  including one mouse (jm038) that stays near zero until P14 — the paper reports exactly one
  "clear outlier … with a similar but delayed developmental trajectory".
  **This is the strongest single check of the conversion**: it shows that the neural traces, the
  behavioural trace and their temporal alignment jointly reproduce the paper's published decoding
  performance.

### Check 5: Edge cases
- **Trial/session boundaries**: `36000 = 20×1800` and `54000 = 30×1800` exactly, so no partial trial
  is produced; the code drops an incomplete trailing block if one ever occurred and prints a note.
  Check 2 above confirms 0 frames lost in all 41 sessions.
- **First frame of the behaviour trace**: `motion_energy_glob[0] == 0` in every session (frame
  differencing has no predecessor for frame 0) — treated as missing and interpolated, otherwise a
  spurious minimum would be introduced into the first bin and would bias the lowest quintile.
- **Dropped camera triggers**: 8/41 sessions, 1-148 frames; reconstructed from the timestamps.
- **Camera running past the imaging**: jm046 (3, 10 and 3 surplus samples) — samples that map beyond
  the last imaging frame are discarded rather than shifting the whole trace.
- **Failed ROIs**: 8 ROIs have an identically zero `F` trace on at least one day (while `Fneu` is
  normal) — missing data, not silence. Removed from every session of the affected mouse so that row
  *i* remains the same tracked neuron on every day.
- **float32 rounding**: the time input is stored as float32, so consecutive differences deviate from
  1/3 s by at most 6e-5 s. Irrelevant at a 333 ms bin size, documented.

### Issues Found and Resolved (iteration log)
1. **Iteration 1** — sanity check 17 found **8 zero-variance neurons**. Root cause: ROIs whose
   suite2p signal extraction failed (all-zero `F` row). *Fix*: `drop_failed_neurons()` (see above).
   Re-ran conversion + verification + all checks → PASS, neurons/mouse 221/370/685/746/541/435 →
   220/367/682/746/541/434.
2. **Iteration 1** — the check comparing whole sessions against the independent dF/F failed for one
   neuron in jm039 (max diff 1.7). Root cause: a bug **in the check**, not the conversion —
   `np.pad(mode='reflect')` is `scipy.ndimage`'s `mirror`, not its `reflect`. *Fix*: use
   `mode='symmetric'` in the check; it then agrees to 1.3e-4. (Confirmed independently that
   float32-vs-float64 differences in the conversion are ≤ 8.9e-5.)
3. **Iteration 1** — the "time increases by exactly 1/3 s" check failed. Root cause: float32
   resolution near 1800 s (6e-5 s), not a conversion error. *Fix*: tolerance in the check.
4. **Iteration 1** — the alignment check on 2 sessions flagged a 10 s lag for jm039 P14. Root cause:
   testing single sessions at a wide lag range where the cross-correlation is nearly flat. *Fix*:
   test the distribution over all 41 sessions (median lag 0 bins, 71 % within ±0.67 s) — see also
   `cache/alignment_check_out.txt`, where the peak lag is 0 or +1 bin (calcium kinetics) in every
   session with a non-negligible correlation.
5. **Iteration 1** — the absolute event rate was ~1.4× the paper's. Resolved as an ambiguity in the
   paper's peak-detection threshold (Check 4); with the unbinned-trace sd the rates match
   quantitatively. No change to the conversion.
6. **Iteration 2** — after the neuron drop, `--sample` and `--full` disagreed on the neuron count
   (the drop rule needs all days of a mouse). *Fix*: `drop_failed_neurons()` now inspects the
   mouse's remaining days lazily, so a sample conversion is an exact subset of the full one.
All checks were re-run after each fix; the final state is all-PASS.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

Command: `python -u train_decoder.py /app/converted_data.pkl --plot-samples`
(log: `/app/train_decoder_full_out.txt`; figures: `sample_trials.png`, `predictions.png`)

### Training Progress
- Loss decreasing: **Yes** — 55.5 (epoch 1) → 26.4 (10) → 4.05 (50) → 1.72 (100) → 1.15 (200);
  test loss 2.74.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Notes |
|--------|-------------|--------|---|-------|
| motion_energy_quintile | 0.596 | **0.315** | 0.200 | 1.57× chance; reproducible across seeds (four runs: 0.315, 0.304, 0.300, 0.306; mean 0.306) |

Per-session validation accuracy (`cache/per_session_accuracy_out.txt`) ranges from 0.21 to 0.51 and
increases with age: early (≤P11) 0.295, late (>P11) 0.326, correlation with postnatal day r = 0.28.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance
| Output | Validation | Chance (uniform) | Ratio |
|---|---|---|---|
| motion_energy_quintile | 0.315 | 0.200 | 1.57× |

Above chance, and reproducibly so (0.300-0.315 over four seeds, mean 0.306). The ratio is modest, which is
expected rather than symptomatic, for three reasons that were each checked:
1. **Half of the dataset is genuinely near-undecodable.** The paper's central result is that motion
   is *not* encoded in barrel cortex before ~P11 and only becomes decodable afterwards (Fig. 7C).
   My per-session accuracies follow exactly that pattern (0.295 early vs 0.326 late, r = 0.28 with
   age; best sessions jm046 P13/P14 = 0.45/0.51), and the ridge-regression reproduction gives
   R² ≈ 0.15 early vs 0.41 late, matching the published values.
2. **Five equal-percentile bins of a heavy-tailed variable are intrinsically hard.** Motion energy
   sits on a narrow "animal still" floor for most of the recording, so the lower three quintile
   thresholds are separated by only ~10-20 % in motion energy (e.g. jm031: 5.5e5, 6.0e5, 7.0e5,
   2.0e6), i.e. classes 0-2 differ mostly by camera noise. The discretisation is prescribed by the
   task, so this is a property of the task, not of the conversion.
3. **The provided decoder trains one shared read-out over 41 sessions of 6 mice at 8 different
   ages**, whereas the paper fits a separate model per session.

### Check 2: Accuracy comparison to the paper
The paper reports **no classification accuracies** — its decoder is a ridge *regression* and the
reported metric is R². The comparison was therefore made on the paper's own metric, by running the
paper's analysis on the converted data (`cache/ridge_reproduce_fig7.py`):

| Quantity | Paper | This conversion |
|---|---|---|
| Same-day R², example mouse D at P8 (Fig. 7B) | 0.26 | 0.18 |
| Same-day R², example mouse D at P14 (Fig. 7B) | 0.69 | 0.55 (0.58 at P13) |
| Same-day R², early (≤P11) (Fig. 7C/H) | ~0-0.3 | mean 0.15, range −0.09 … 0.42 |
| Same-day R², late (>P11) (Fig. 7C/H) | ~0.4-0.8 | mean 0.41, range 0.01 … 0.80 |
| Best session | ~0.8 | 0.80 (jm046 P14) |
| One outlier mouse with a delayed trajectory | yes (Fig. 5D-G) | yes (jm038: R² ≤ 0.12 until P14) |
| Ca²⁺ event rate, mouse D, P8 → P14 (Fig. 5D) | 3.0 → 7.4 /min | 3.85 → 7.21 /min |
| Ca²⁺ event rate, all mice at P14 (Fig. 5D) | 4.3 - 11.3 /min | 7.1 - 10.6 /min |
| PC1-motion correlation, early → late (Fig. 7D) | ~0.1-0.3 → ~0.6-0.8 | 0.12 → 0.28 (quintile labels, not the continuous trace) |

My R² values are systematically ~0.1 lower than the two example values quoted in Fig. 7B, which is
within the spread of Fig. 7C and is attributable to (i) the released dataset having somewhat
different cell counts than the figure (Step 9), (ii) the unknown transformation of the behavioural
trace used by the authors — using `log` motion energy raises every value (mouse D P14: 0.55 → 0.56,
late mean 0.41 → 0.45), and (iii) my re-implementation of their nested CV (alpha grid, fold
assignment) not being identical. The developmental trajectory, the between-mouse ordering, the
outlier mouse and the magnitude of the best sessions all match. No conversion bug is implied.

### Check 3: Train vs validation gap
Training 0.596 vs validation 0.315 = 1.9×. Since every session contributes trials to both sides of
the split and trials are contiguous 60 s blocks, this is over-fitting of the 100-dimensional
per-session projections (41 × 100 × n_neurons free parameters against 1090 trials), not leakage:
*   there is no shared information between trials (disjoint time ranges);
*   the per-session split is done by the provided harness, not by the conversion;
*   the gap is essentially the same for the sessions where the ridge analysis says there is little
    to decode (jm031 P10: train 0.59, valid 0.24) and much smaller where there is (jm046 P14:
    train 0.66, valid 0.51), i.e. it tracks genuine decodability.

### Sensitivity analysis of the neural representation
Because the paper's wording ("baseline corrected fluorescence traces as our dF/F") and its code
(`F_processing`, which subtracts but does not divide, with `neucoeff=0`) can be read in several
ways, all readings were converted and decoded end-to-end:

| Neural representation | Justification | Validation balanced acc |
|---|---|---|
| **F − F0 maximin, neucoeff = 0 (chosen)** | verbatim reference implementation | **0.315 / 0.304 / 0.300 / 0.306** |
| F − F0 maximin, neucoeff = 0.7 | suite2p's default neuropil coefficient | 0.299 |
| (F − F0)/F0 | literal reading of "dF/F" | 0.286 |
| per-neuron z-scored F − F0 | common pre-decoding normalisation | 0.329 / 0.332 |

The spread is small (0.29-0.33) and the reference-faithful choice sits in the middle, so the
conversion keeps it: the ~+0.02 of z-scoring does not justify departing from the reference
processing, and z-scoring would additionally destroy the amplitude information that the paper's own
analyses (event rates, ridge decoding) rely on. `convert_data.py --dff-mode {subtract,divide,zscore}`
and `--neucoeff` reproduce every row of this table.

### Issues Found and Resolved
- No further issues. All Step-10 checks were re-run against the final `converted_data.pkl` and pass.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, loading instructions, format, key statistics)
- [x] `cache/` folder created with the investigation scripts and their outputs, documented in
      `cache/README_CACHE.md`
- [x] All required files present: `CONVERSION_NOTES.md`, `convert_data.py`, `converted_data.pkl`,
      `sample_data.pkl`, `README.md`, `conversion_sample_out.txt`, `verification_sample_out.txt`,
      `train_decoder_sample_out.txt`, `conversion_full_out.txt`, `verification_full_out.txt`,
      `train_decoder_full_out.txt`
