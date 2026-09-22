# Dataset Conversion Notes

## Overview
- **Dataset**: Brain-wide neural activity underlying memory-guided movement (Chen et al.) / method paper: Brain-wide analysis reveals movement encoding structured across and within brain areas
- **Date started**: (see file mtime)
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of /app:
- .manifest, Dockerfile, docker-compose.yaml
- ChenLiuEtAl2023_SpikeSortingQC.pdf, datapaper.pdf, methodpaper.pdf, methods.txt
- code/ (MapVideoAnalysis repo: Archive/, Notebooks/, Sherlock/, VideoAnalysisUtils/)
- data/ (DANDI 000363 subset: 29 sub-* folders, 174 NWB files, ~50 GB)
- decoder.py, train_decoder.py

Environment verified: python3 with numpy 2.4.4, torch 2.6.0+cu124, pynwb 4.1.0, h5py.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

Reference code = MapVideoAnalysis (Wang, Kurgyis et al. 2025, Nat Neurosci), analysing the
MAP dataset (Chen et al.) that we have here in NWB form.

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| process_all_sess_parallel(bw, stride, begin_time, end_time, ...) | Sherlock/preprocess_all_ephys.py + VideoAnalysisUtils/preprocessing_DJ_2022Aug.py | LOADING | Top-level entry: bw=0.04 s, stride=0.0034 s, window [-3, 3] s around go cue, qc_mode=classifier |
| process_one_sess | preprocessing_DJ_2022Aug.py | LOADING/CURATION | Concatenates all probes of a session; reads behaviour/task vars; loads the classifier-QC goodunits list; keeps only units with BOTH ephys and histology (CCF) entries |
| helper_get_neuron_id_area | preprocessing_DJ_2022Aug.py | CURATION | Keeps QC-good units, splits by hemisphere (CCF ML midline 5700 um) and region (14 regions: ALM, Medulla, Midbrain, Striatum, Thalamus, Pons, Cerebellum, Hypothalamus, Hippocampus, Orbital, OtherCortex, Olfactory, CorticalSubplate, Pallidum) |
| process_one_area | preprocessing_DJ_2022Aug.py | PROCESSING | Truncates spike times to [begin_time, end_time] RELATIVE TO GO CUE and bins them |
| sliding_histogram(spikeTimes, begin, end, bin_width, stride, rate=True) | preprocessing_DJ_2022Aug.py | PROCESSING | Sliding-window spike counts -> firing RATES (counts / bin_width); bin centres from begin to end spaced by stride; bin = [c-bw/2, c+bw/2) |
| get_regular_trial_mask(ephys_data) | functions_for_r2.py, population_decoding_utils.py | CURATION | early_lick==0 AND auto_water==0 AND free_water==0 AND correctness!=-1 AND stimulation[:,0]==0 |
| align_markers_between_lims(marker_data, go_times, t_min=-3, t_max=1.5) | Sherlock/align_markers.py | PROCESSING | Video markers (nose/tongue/jaw/whisker x,y) from camera_0_side, frame dt = 0.0034 s (300 Hz), aligned to go cue; for each output time point takes the LAST frame falling in [t-dt, t) |
| get_bad_trial_inds | Sherlock/align_markers.py | CURATION | Flags trials where the number of video frames disagrees with the trial duration (frames - end_time/dt outside [0,1]) |
| load_session | population_decoding_utils.py | LOADING | Loads preprocessed per-area pickles, concatenates neurons across areas |
| nested_cross_validation | population_decoding_utils.py | ANALYSIS | Population decoding (logistic regression + PCA16, 5-fold outer / 10-fold inner CV, AUC) |

### Notes
- Neural data: spike times relative to the go cue, binned with a 40 ms sliding window at 3.4 ms stride into firing rates (Hz). The present task requires 50 ms bins over [-2.5, +1.5] s, so I keep the reference convention firing rate = spike count / bin width but use non-overlapping 50 ms bins (80 bins/trial).
- Neuron curation in the reference is the classifier-based QC of Chen/Liu et al. 2023. In the DANDI NWB files this is materialised as units/classification (good vs unlabelled), and units/anno_name holds the CCF annotation (empty for non-QC units). So classification == good reproduces the reference goodunits files.
- The reference additionally requires units to have histology (CCF annotation); in NWB this is a non-empty anno_name.
- Trial curation in the reference excludes early-lick, auto/free water, no-response and photostim trials. The present task explicitly requires early lick, outcome (including ignore = no response) and photostimulation as decoder variables, so those exclusions cannot be applied (see Step 5). Auto-water / free-water trials are not normal task behaviour and are excluded following the reference.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data` is a DANDI-style layout of DANDI:000363 (Mesoscale Activity Map dataset):
`data/sub-<animalid>/sub-<animalid>_ses-<YYYYMMDDThhmmss>_behavior+ecephys[+ogen].nwb`
- 29 sub-* directories, but one is `dandiset.yaml`, so 28 subjects; 174 NWB files (~50 GB).
- NWB 2.6 files (HDF5). Contents per file:
  - `intervals/trials`: start_time, stop_time, trial, trial_uid, task (all `audio delay`), task_protocol (all 1),
    trial_instruction (left/right), outcome (hit/miss/ignore), early_lick (early / no early / early, no lick...),
    auto_water (0/1), free_water (0/1), photostim_onset / photostim_duration / photostim_power (strings, `N/A` when no stim).
  - `acquisition/BehavioralEvents/*`: presample, sample, delay, go, trialend start/stop times, left/right lick times,
    photostim start/stop times. Times are session-clock seconds (`timestamps`), `data` holds the value.
    Exactly one `go_start_times` event per trial in all 174 sessions (verified).
    `sample_start_times` can be more numerous than trials because early licks trigger a replay of the epoch.
  - `acquisition/BehavioralTimeSeries/Camera0_side_{Jaw,Nose,Tongue}Tracking`: (nframes, 3) = (x, y, likelihood),
    300 Hz (dt = 0.0034 s), session-continuous timestamps with gaps between trials (video only recorded during trials).
    All 174 sessions have side-view tongue/jaw/nose. 20 sessions also have whisker, 3 also have Camera3.
  - `units`: 43 columns including spike_times (+index), classification (`good` / `unlabelled`), unit_quality (`good`/`multi`),
    anno_name (CCF structure name; empty for non-good units), the 15 QC metrics (amplitude_cutoff, presence_ratio,
    isi_violation, drift_metric, unit_amp, unit_snr, avg_firing_rate, isolation_distance, d_prime, nn_hit_rate, ...),
    is_good_trials (nunits x ntrials bool), obs_intervals (per-trial observation windows), electrodes (index into electrode table).
  - `general/extracellular_ephys/electrodes`: x (ML), y (DV), z (AP) CCF coordinates in um, `location` = JSON string with
    the targeted coarse region, e.g. `{"brain_regions": "left ALM", ...}`.
- Spike times are in session-clock seconds and exist **only inside trial [start_time, stop_time] intervals**
  (verified: 100% of a unit's spikes lie inside trial intervals; obs_intervals = the trial intervals).

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| NWB files (sessions) | 174 |
| Subjects | 28 |
| Sessions / subject | 3-10 (mean 6.2) |
| Units (all clusters) | 272,227 |
| Units with `classification == good` | 69,453 (25.5% of clusters) |
| Good units also having CCF annotation | 69,453 (100%) |
| Good units / session | mean 399, range 0-923 |
| Sessions with 0 good units | 1 (sub-440958_ses-20190216T162508) |
| Probe insertions (electrode groups) | 659 |
| Trials (total) | 94,990 |
| Trials / session | mean 546, range 264-800 |
| Outcome | hit 65,254 (68.7%), miss 15,641 (16.5%), ignore 14,095 (14.8%) |
| Early lick | early 10,805 (11.4%), no early 84,185 |
| Trial instruction | right 48,913, left 46,077 |
| auto_water = 1 | 1,339 trials |
| free_water = 1 | 2,450 trials |
| Photostim trials (power != N/A) | 18,588 in 168 sessions |
| Time from go cue to trial start | median 3.15 s (96.9% of trials >= 2.5 s) |
| Time from go cue to trial stop | median 1.80 s (84.4% of trials >= 1.5 s) |
| Time from tone (sample) onset to go | 1.85 s median (0.65 s sample + 1.2 s delay) |
| Tongue visible (likelihood > 0.9) | ~10-16% of video frames per session |

### Unique CCF annotations among good units
293 distinct Allen CCF structure names. Grouped with keyword rules into the 14 coarse groups used by the
reference code, the counts closely reproduce the data paper (see Step 3/4).

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Good units (total) | 69,943 | methods.txt: "the dataset consisted of 69,943 good units recorded across 173 behavioral sessions" |
| Fraction of KS2 clusters kept | 25.9% | methods.txt: "This corresponds to 25.9 % of clusters reported by Kilosort2" |
| Sessions | 173 | methods.txt / datapaper: "660 penetrations, 173 behavioral sessions, and 28 mice" |
| Mice | 28 | datapaper Fig 1J legend |
| Penetrations | 660 (655 in methods.txt) | datapaper Fig 1J / methods.txt |
| Neurons / session | median 393 | datapaper: "hundreds of neurons (median = 393)" |
| Units by area | ALM 8,717; striatum 7,664; thalamus 12,808; midbrain 7,495; medulla 2,928 | methods.txt |
| Trials / session | mean 476, range 130-785 | methods.txt: "Mice performed on average 476 (Mean; range, 130-785) trials per session" |
| Correct rate | 84% (range 65-99%) | methods.txt |
| Session inclusion | performance > 65% and >= 50 correct left and >= 50 correct right trials | methods.txt |
| Photostim | ~25% of trials, 17 VGAT-ChR2 mice, last 0.5 s of delay, always ends before go cue | methods.txt |
| Bilateral ALM photostim performance | 83.2% -> 71.7% (93 sessions) | methods.txt |
| Neural bin (method paper) | 40 ms width, 3.4 ms stride | methodpaper: "we binned spikes into firing rates with a bin width of 40 ms and a stride of 3.4 ms" |
| Video | 300 Hz side view, DeepLabCut markers jaw/nose/tongue | methods.txt, methodpaper |
| Tongue occlusion handling | "When the tongue was occluded ... we set the tongue position to its mean value" | methodpaper Methods |
| Choice decoding from video | AUC 0.51 pre-sample, 0.66 sample+delay, 0.99 after go cue (n = 106 sessions) | methodpaper Results |

### Processing Details
- Trials are aligned to the **go cue** (reference preprocessing stores spike times relative to go cue;
  marker alignment script uses `go_times` as t = 0).
- Reference window [-3, +3] s (ephys) and [-3, +1.5] s (markers). The current task specifies [-2.5, +1.5] s.
- Firing rate = spike count / bin width (Hz).
- Task epochs: presample -> sample (tone, 0.65 s: 3 x 150 ms tones with 100 ms gaps) -> delay 1.2 s -> go cue ->
  answer 1.5 s -> consumption 1.5 s.

### Curation Steps
**Neuron curation rules** (reference): keep units labelled `good` by the region-specific logistic-regression
classifiers of the QC white paper (materialised as `units/classification == 'good'` in the NWB files), and require
a CCF annotation (histology). In this dataset all `good` units have an annotation.

**Trial curation rules** (reference, `get_regular_trial_mask`): no early lick, no auto water, no free water,
no no-response (ignore) trials, no photostimulation trials. Sessions: performance > 65%, >= 50 correct left and right trials.

### Decoders Trained
| Decoded variable | Accuracy |
| Choice from behavioural video (AUC) | 0.51 (pre-sample), 0.66 (sample+delay), 0.99 (response) |
| Choice from ALM/medulla population firing rates (AUC, Fig 6/7) | ~0.9+ after go cue |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Good-unit selection | loads classifier `goodunits` .mat lists | `units/classification` in {good, unlabelled}; 69,453 good = 25.5% of 272,227 clusters | 69,943 good units = 25.9% of KS2 clusters | The NWB `classification` column is the published output of the same classifier. Counts agree to 0.7% (the DANDI release differs marginally from the paper snapshot). Use `classification == 'good'`. |
| Number of sessions | n/a | 174 NWB files, one of which (sub-440958_ses-20190216T162508) has 0 good units | 173 behavioral sessions | Dropping the session with no QC-passing units gives exactly 173 sessions. |
| Penetrations | n/a | 659 electrode groups | 655 (methods.txt) / 660 (Fig 1J) | Within rounding of the published numbers; no action. |
| Units per area | 14 coarse groups from CCF label + hemisphere | keyword grouping of 293 CCF names gives Medulla 2,925; Midbrain 7,447; Striatum 7,557; Thalamus ~13,000; cortex ~26,000 | Medulla 2,928; midbrain 7,495; striatum 7,664; thalamus 12,808; ALM 8,717 | Agreement is within 1-2% for medulla/midbrain/striatum, confirming both the good-unit selection and the region grouping. |
| Trials per session | n/a | mean 546, range 264-800 | mean 476, range 130-785 | The NWB trials table contains **all** trials, the paper statistic is computed after the paper's trial curation (early lick / no response / auto water removed) and over their session set. Our mean is expected to be higher. |
| Trial exclusions | excludes early lick, auto/free water, ignore, photostim | all these trials are present in the NWB | "Early lick trials and no response trials were excluded for analysis" | **Cannot** apply for this task: early lick, outcome (which includes `ignore` = no response) are required decoder outputs and photostim is a required decoder input. Only auto-water/free-water trials (water delivered irrespective of behaviour, i.e. not task behaviour) are dropped, following the reference. Documented in Step 5. |
| Alignment | go cue = t 0 | `go_start_times` has exactly 1 event/trial | go cue alignment | Consistent. |
| Tongue tracking | markers from `camera_0_side` at 300 Hz | `Camera0_side_TongueTracking` (x, y, likelihood) at dt = 0.0034 s | same | Consistent. Tongue "not visible" corresponds to low DeepLabCut likelihood (bimodal: ~10-16% of frames > 0.9). |


---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable (NWB) | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------------|--------------|-----------|----------------------------|-------|
| `units/spike_times` (+`spike_times_index`), `classification=='good'` | `neural` | spikes aligned to go cue, counted in 80 non-overlapping 50 ms bins spanning [-2.5, +1.5] s, divided by 0.05 s -> firing rate in Hz | `process_one_area` + `sliding_histogram` (rate = count / bin width) | 50 ms bins required by the task spec instead of the reference 40 ms/3.4 ms sliding window |
| `acquisition/BehavioralEvents/go_start_times` | alignment event | t = 0 | `align_markers_between_lims` (`go_times`), reference spike times are stored relative to go cue | exactly 1 event per trial in all sessions |
| `acquisition/BehavioralEvents/sample_start_times` | `input[0]` = `time_from_tone_onset` | signed seconds from the last sample(tone)-epoch onset preceding the go cue to each bin centre | task/sample epoch in `preprocessing_DJ_2022Aug` (`task_sample_time`) | median tone onset = go - 1.85 s; early-lick replays restart the sample epoch, so the *last* onset before the go cue is the tone the animal actually responded to |
| `acquisition/BehavioralEvents/photostim_start_times` / `photostim_stop_times`, `intervals/trials/photostim_*` | `input[1]` = `photostim_on` | binary per time bin, 1 if the laser was on during the bin | `stimulation` columns in `preprocessing_DJ_2022Aug` | ALM photoinhibition, 0.5 s, always ends by the go cue |
| `intervals/trials/outcome` + `trial_instruction` | `output[0]` = `lick_direction` | ignore -> `no lick`; hit -> instructed side; miss -> opposite side | `behavior_lick_directions` / `trial_type` + `behavior_report` in the reference | validated against the actual first lick after the go cue: 100% / 99.8% agreement in two test sessions |
| `intervals/trials/outcome` | `output[1]` = `outcome` | ignore=0, miss=1, hit=2 | `correctness` (1 correct, 0 error, -1 no response) | same three-way variable |
| `intervals/trials/early_lick` | `output[2]` = `early_lick` | `no early` -> 0, anything containing `early` -> 1 | `behavior_early_report` | |
| `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking` (x, y, likelihood) | `output[3]` = `tongue_y_position` | per 50 ms bin: if any frame in the bin has DeepLabCut likelihood > 0.9 use the **last visible frame's** y, else class 3 (not visible). y is discretised with the session's 40th/60th percentiles of y over all visible frames of the session | `align_markers_between_lims` (marker from `camera_0_side`, last frame inside the bin) | likelihood is strongly bimodal (<0.5% of frames between 0.05 and 0.9), so the visibility threshold is not critical |
| `units/anno_name` (CCF structure) + electrode `x,y,z` | `brain_regions`, `brain_region_idx` | keyword grouping of the 293 CCF structure names into the 14 coarse groups used by the reference (ALM, Orbital, OtherCortex, Striatum, Pallidum, Thalamus, Hypothalamus, Midbrain, Pons, Medulla, Cerebellum, Hippocampus, Olfactory, CorticalSubplate) | `helper_get_neuron_id_area` and the region loop in `process_one_sess` | ALM = motor cortex (MOs/MOp) or frontal pole located anterior to AP +2.0 mm (CCF z < 3400 um), i.e. the ALM craniotomy region. Gives 8,387 units vs 8,717 in the paper (96%). |
| `general/subject/subject_id` | `subjects`, `subject_idx` | 28 mice | | |

### Key Decisions
1. **Neuron curation = `classification == 'good'`**: this column is the published output of the region-specific
   logistic-regression QC classifiers described in methods.txt and the QC white paper, which is exactly what the
   reference pipeline loads from its `goodunits` files. Reproduces 69,453 units (paper: 69,943; 25.5% vs 25.9% of clusters).
   Units are additionally required to have a CCF annotation (histology), as in `process_one_sess`; in practice all
   `good` units have one.
2. **Session curation**: drop sessions with no QC-good units (1 session) or with < 2 usable trials. This yields
   exactly 173 sessions, matching the paper. No further behavioural session filtering is applied because the DANDI
   release is already the curated set used in the paper (the paper's stated >65% criterion is computed on a slightly
   different trial subset, and re-applying it would remove sessions the paper kept).
3. **Trial curation**: exclude `auto_water == 1` and `free_water == 1` trials, following the reference
   `get_regular_trial_mask`; these are trials in which water is delivered irrespective of the animal's choice, so the
   behavioural variables do not reflect a decision. Early-lick trials, no-response (`ignore`) trials and photostim
   trials are **kept**, contrary to the reference, because the decoder specification explicitly requires early lick and
   outcome (which contains `ignore`) as outputs and photostimulation as an input. Trials without a go cue would also be
   dropped (none exist).
4. **Binning**: non-overlapping 50 ms bins (task spec) over [-2.5, +1.5] s = 80 bins; firing rate in Hz = count/0.05,
   as in the reference (`rate=True`).
5. **Truncated trials**: spikes in the NWB exist only inside each trial's [start_time, stop_time]. For ~16% of trials
   the trial ends before go + 1.5 s, and for ~3% it starts after go - 2.5 s, so those bins contain no spikes. This
   identical truncation exists in the reference pipeline (per-trial spike times, window [-3, +3] s); trials are kept.
6. **All outputs are time-varying** (shape (4, 80)); the three per-trial variables are held constant across bins, which
   the decoder spec prefers, and the tongue class varies over time.
7. **Inputs are time-varying** (shape (2, 80)): time from tone onset in seconds and binary photostim.

### Planned Sanity Checks
- [ ] Total good units == 69,453 and 173 sessions / 28 mice after conversion.
- [ ] Per-region unit counts within a few % of the paper (medulla 2,928; midbrain 7,495; striatum 7,664; thalamus 12,808; ALM 8,717).
- [ ] Neural firing rates: session mean in a plausible range (1-10 Hz), no NaN/Inf.
- [ ] Spot-check: recompute the spike count for a given (session, trial, neuron, bin) directly from the NWB file and compare with `np.allclose`.
- [ ] Input 0 at the bin containing the go cue is ~1.85 s; input 0 is monotonically increasing with 0.05 s steps.
- [ ] Input 1 is 1 only before the go cue (photoinhibition ends at the go cue) and only in sessions/trials with photostim power != N/A; total photostim trials ~18.6k before trial curation.
- [ ] Output distributions: hit ~69%, miss ~16%, ignore ~15%; early lick ~11%; choice left/right roughly balanced.
- [ ] Tongue class 3 (not visible) dominates before the go cue and drops after it; classes 0/1/2 split ~40/20/40 among visible bins.
- [ ] Output choice derived from outcome+instruction agrees with the observed first lick direction after the go cue.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` (`python -u convert_data.py <out.pkl> [--full|--sample] [--show-processing] [--nproc N]`).

Structure:
- `classify_region(name, ap)` - maps the 293 CCF structure names onto the 14 coarse groups used by the
  reference pipeline (+ `Unknown`); ALM = motor cortex / frontal pole anterior to AP +2.0 mm.
- `bin_spikes(...)` - vectorised binning: for each unit, one `np.searchsorted` over all trial bin edges at once,
  then `np.diff` per trial. Rates = counts / 0.05 s (reference `sliding_histogram(rate=True)`).
- `tongue_class_per_bin(...)` - per bin uses the last visible video frame inside the bin (reference
  `align_markers_between_lims`); bins with no visible frame get class 3.
- `convert_session(...)` - loads one NWB file, applies neuron/trial curation, builds neural/input/output arrays.
- `plot_processing(...)` - `--show-processing` figures: population PSTH, an example trial raster,
  input 0 / input 1 across trials, all four outputs across trials, plus a per-trial overlay of the raw tongue
  y-trace with the 40/60 percentile thresholds and the resulting discrete class.
- `main()` - multiprocessing pool over sessions (12 workers), assembles the dictionary and pickles it.

Code inefficiencies identified:
- Naive per-trial/per-unit loops over spike times (as in the reference `sliding_histogram`, which is O(units x trials x bins)).
- Reading the full `spike_times` dataset once per session instead of per unit.

Code speedups added:
- Single `searchsorted` per unit over the flattened bin-edge array (vectorised over trials and bins).
- Whole-session reads of `spike_times`, video data and timestamps; per-session multiprocessing.
- float32 firing rates.
Result: ~0.5 s per session single-threaded, so the full dataset converts in ~1-2 minutes.

Issues found and fixed during development:
1. `np.diff` over the flattened edges array mixed adjacent trials -> fixed by reshaping before `np.diff`.
2. **Trials outside the ephys recording**: spikes are exported only for trials during which the probes were
   recording (`units/obs_intervals`). In 8/173 sessions the recording covers a contiguous subset of the trials
   (e.g. trials 125-629 of 630), and the remaining trials would appear as all-zero firing rates. Trials are now
   restricted to those in the observation intervals of every kept unit.
3. A recording stopped mid-trial leaves the last observed trial with no spikes at all; trials with zero spikes
   across all (hundreds of) QC-passing neurons are dropped.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Commands:
```
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing | tee /app/conversion_sample_out.txt
python -u /app/train_decoder.py /app/sample_data.pkl --verify-only | tee /app/verification_sample_out.txt
```

### Sample Statistics (2 sessions of mouse 440956)
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons (total) | 834 (459 + 375) |
| Neurons / session | 417 mean |
| Subjects | 1 |
| Trials (total) | 513 (354 + 159) |
| Trials / session | 354, 159 (the second session's ephys covered only 160 of 480 trials) |
| Timepoints / trial | 80 (50 ms bins, -2.5 to +1.5 s) |
| Mean firing rate | 5.4 / 6.4 Hz |
| time_from_tone_onset range | [-0.6, 5.7] s |
| photostim_on range | [0, 1]; 21.8% / 27.0% of trials stimulated |
| lick_direction | left 0.455, right 0.409, no lick 0.136 |
| outcome | ignore 0.136, miss 0.298, hit 0.566 |
| early_lick | no 0.942, yes 0.058 |
| tongue_y_position | <40th 0.143, 40-60th 0.031, >60th 0.061, not visible 0.765 |

### Processing Plots Review
- Population PSTH is flat around 5 Hz during sample/delay and rises sharply just after the go cue (8-9 Hz at +0.3 s),
  the expected response-epoch activation - confirms the alignment is correct.
- `input 0` is a set of parallel ramps with 0.05 s steps, crossing zero at the tone onset (median -1.85 s
  relative to the go cue).
- `input 1` is on only in the 0.5 s window ending at the go cue (bins -0.9 to -0.5 s in this mouse, whose
  stimulation started at delay onset), never after the go cue - matches "photoinhibition always ended before the Go cue".
- Tongue class 3 (not visible) covers 95-99% of bins before the go cue and drops to 9-27% after it, exactly as
  expected since the tongue is only outside the mouth while licking.
- The tongue overlay figure shows the discrete class following the raw y trace across the 40th/60th percentile lines.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
| vectorised searchsorted binning | ~50x vs per-trial loops |
| whole-session HDF5 reads | ~3x |
| 12-process pool | ~10x |

| Step | Time / Session | Estimated Total Time |
| read spikes | 0.05-0.14 s | ~20 s |
| bin spikes | 0.08-0.22 s | ~40 s |
| tongue discretisation | 0.06-0.08 s | ~15 s |
| whole session | 0.3-0.6 s (larger sessions more) | ~2-4 min with 12 processes (dominated by pickling ~12 GB) |

Estimated output size: 12.2 GB of neural float32 data for the full dataset.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None (the initial "all neural data is zero" warnings were caused by trials outside the ephys
  recording interval and were fixed in Step 6/7)

### Decoder Results (Sample, 2 sessions, 513 trials, 834 neurons)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc |
|--------|--------|-----------------------|-------------------------|
| lick_direction | 0.333 | 0.745 | 0.623 |
| outcome | 0.333 | 0.763 | 0.641 |
| early_lick | 0.500 | 0.827 | 0.734 |
| tongue_y_position | 0.250 | 0.717 | 0.574 |

Training loss fell monotonically from 7.5 to 0.56 over 200 epochs; every output is well above chance.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

```
python -u /app/convert_data.py /app/converted_data.pkl --full --nproc 16 | tee /app/conversion_full_out.txt   # 43 s
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only | tee /app/verification_full_out.txt
```

### Output Files
- `converted_data.pkl`: 11.89 GB (173 sessions, 89,544 trials, 69,453 neurons, 80 bins/trial)
- `verification_full_out.txt`: created, **no errors and no warnings**

### Consistency Check
| Statistic | Reference Papers | Reference Data (NWB) | Converted Data | Match? |
|-----------|------------------|----------------------|----------------|--------|
| Sessions | 173 | 174 files (1 with 0 QC units) | 173 | YES |
| Mice | 28 | 28 | 28 | YES |
| Probe insertions | 655-660 | 659 electrode groups | (659 recorded) | YES |
| Good units (total) | 69,943 | 69,453 (`classification=='good'`) | 69,453 | 0.7% low, = the NWB release |
| Fraction of clusters kept | 25.9% | 25.5% | 25.5% | YES |
| Neurons / session | median 393 | median 390 | median 390, mean 401, range 90-923 | YES |
| Trials / session | mean 476, range 130-785 | mean 546, range 264-800 | mean 518, median 518, range 159-796 | close (see note) |
| Performance (hit/(hit+miss), control trials) | 84% (65-99%) | 81% | 80.4% | close |
| Photostim trials | ~25% | 19.6% | 20.0% | close |
| Early-lick trials | - | 11.4% | 11.6% | YES |
| Outcome (hit/miss/ignore) | - | 0.687/0.165/0.148 | 0.685/0.167/0.148 | YES |
| ALM units | 8,717 | - | 8,387 | 96% |
| Striatum units | 7,664 | - | 7,737 | 101% |
| Thalamus units | 12,808 | - | 12,808 | exact |
| Midbrain units | 7,495 | - | 7,505 | 100% |
| Medulla units | 2,928 | - | 2,925 | 99.9% |
| Input 0 range (time from tone onset) | tone 1.85 s before go | 0.95-1.85 s typical | [-1.5, 11.9] s, median tone-to-go 1.85 s | YES |
| Input 1 range | photostim 0.5 s ending at go cue | same | [0,1], on only within 0.5 s before the go cue | YES |
| lick_direction | - | left/right roughly balanced | left 0.429 / right 0.422 / no lick 0.148 | YES |
| tongue_y_position | tongue visible only while licking | 10-16% of frames visible | not visible 0.755, classes 0/1/2 = 0.144/0.033/0.068 | YES |

Notes:
- Trials/session is higher than the paper's 476 because the paper's number is computed on their analysis subset
  (after removing early-lick and no-response trials) whereas the specification for this conversion requires keeping them.
  The distribution's range (159-796) brackets the paper's (130-785).
- The 173 - and only 173 - sessions come out of the pipeline automatically: the single session without any
  QC-passing unit (sub-440958_ses-20190216T162508) is dropped.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
`/app/verification_full_out.txt` reports `Data format is valid, no errors or warnings.` All earlier warnings
("all neural data is zero") were traced to two real data problems and fixed:
- trials outside the interval in which the probes were recording (`units/obs_intervals`), 8/173 sessions,
- the final trial of a recording that was stopped mid-trial.
No warning remains unresolved.

### Check 2: Sanity checks against the raw NWB files
`/app/cache/sanity_checks.py` reloads the NWB files independently of `convert_data.py` and compares
(`np.allclose`) for 6 sessions spread over the dataset (indices 0, 5, 46, 100, 119, 172):
| Check | What is compared | Result |
|-------|------------------|--------|
| NEURAL | firing rate of a random (trial, neuron, bin) recomputed by counting raw `units/spike_times` in `[go+edge_b, go+edge_b+1)` / 0.05 | 24/24 exact matches (values 0-20 Hz) |
| INPUT 0 | full 80-bin ramp recomputed from `sample_start_times` and `go_start_times` | 12/12 match (atol 1e-4) |
| INPUT 1 | photostim bin mask recomputed from the trials table (`start_time + photostim_onset`, `photostim_duration`) | 12/12 match |
| OUTPUT 0 | decoded lick direction vs the direction of the **first lick after the go cue** taken from `left/right_lick_times` | 247/247 trials agree (100%) |
| OUTPUT 1, 2 | outcome and early-lick codes vs the raw trials table strings | all match |
| OUTPUT 3 | session 40th/60th percentile thresholds recomputed from the raw tongue trace, plus the class of random (trial, bin) pairs recomputed from the raw video | thresholds match to 1e-9, 12/12 classes match |
| TOTAL | | **0 failures** |

### Check 3: Reference code comparison
| Step | Reference code | This conversion | Same? |
|------|----------------|-----------------|-------|
| (a) loading | `preprocessing_DJ_2022Aug.process_one_sess` reads DataJoint `.mat` exports, concatenating all probes of a session | reads the published NWB of the same session; the NWB `units` table already contains all probes | YES (same data, different container) |
| (b) neuron filtering | classifier `goodunits` index lists + require ephys AND histology | `units/classification == 'good'` (the published output of the same classifiers) AND non-empty `anno_name` (histology) | YES; verified by matching per-region counts to the paper |
| (c) temporal alignment | spike times stored relative to the go cue; markers aligned with `go_times` | all streams aligned to `go_start_times` | YES |
| (d) binning | `sliding_histogram`, rate = count / bin width, bins `[c-bw/2, c+bw/2)` | non-overlapping 50 ms bins, rate = count / 0.05 s | Deliberate difference: the task specifies 50 ms bins. Same rate convention. |
| (e) input construction | reference stores `task_sample_time`, `task_cue_time`, `stimulation` (`laser_power`, `laser_on/off` relative to go cue) | input 0 from the sample-epoch onset, input 1 from the photostim on/off times, both expressed relative to the go cue | YES (same variables) |
| (f) output construction | `behavior_report`/`correctness`, `trial_type`, `behavior_early_report`, marker traces from `camera_0_side` using the last frame in each bin | outcome, trial_instruction, early_lick, `Camera0_side_TongueTracking` with the last visible frame in each bin | YES |
| trial curation | `get_regular_trial_mask`: no early lick, no auto/free water, no no-response, no photostim | only auto/free water removed | Deliberate difference, required by the decoder specification (early lick, outcome incl. `ignore` are outputs, photostim is an input). Documented in Step 5. |
| extra curation | reference implicitly only has trials present in the export | trials outside `obs_intervals` and trials with zero spikes are removed | Necessary for the NWB container; equivalent to the reference which only received recorded trials. |

### Check 4: Key statistics comparison
See the table in Step 9: sessions (173 = 173), mice (28 = 28), units (69,453 vs 69,943 = the released
subset), units per region (thalamus exact, midbrain/medulla/striatum within 1%, ALM 96%), median neurons
per session (390 vs 393), trial statistics, behaviour fractions. The only deliberate differences are the
trial counts (we keep early-lick / no-response / photostim trials) and ALM (see below).

*ALM definition*: the papers do not give a numeric ALM boundary. I define ALM as motor cortex
(secondary/primary motor area) and frontal-pole units located more than 2.0 mm anterior to bregma, which is the
ALM craniotomy ("centered on AP 2.5 mm; ML 1.5 mm"). This yields 8,387 units vs 8,717 in the paper (-3.8%).
The residual difference only affects the `brain_region_idx` labels, not the neural data.

### Check 5: Edge cases handled
- Sessions whose ephys covers only part of the behavioural session (8 sessions) - trials restricted to `obs_intervals`.
- Recording stopped mid-trial - trials with zero spikes over all neurons dropped (1 trial in the sample).
- Session with no QC-passing unit - dropped (1 session).
- `anno_name` stored as a float (NaN) instead of a string in some files - `_decode` returns `''` and the unit is dropped.
- Trials where the go cue is missing - dropped (none exist, but the code checks `np.isfinite(go)`).
- Early-lick trials replay the sample/delay epochs, so the tone-to-go interval is not always 1.85 s;
  the **last** sample onset before the go cue is used, and the value is clipped to the trial if the sample
  event is missing (falls back to the nominal 1.85 s).
- Trials whose window extends outside the trial interval (96.9% have >= 2.5 s before, 84.4% >= 1.5 s after the go cue):
  those bins legitimately contain no spikes, exactly as in the reference pipeline, and are kept.
- Sessions with fewer than 2 usable trials would be dropped (none).

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

```
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples | tee /app/train_decoder_full_out.txt
```

### Training Progress
- Loss decreasing: Yes, monotonically from 9.9 (epoch 1) to 0.652 (epoch 200); test loss 0.654.

### Decoder Results (Full: 173 sessions, 89,544 trials, 69,453 neurons)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|--------|-----------------------|-------------------------|-------|
| lick_direction (left/right/no lick) | 0.333 | 0.7119 | 0.6832 | 2.0x chance |
| outcome (ignore/miss/hit) | 0.333 | 0.7018 | 0.6611 | 2.0x chance |
| early_lick (no/yes) | 0.500 | 0.7882 | 0.7511 | 1.5x chance |
| tongue_y_position (3 levels + not visible) | 0.250 | 0.7011 | 0.6506 | 2.6x chance |

Train/validation gaps are small (0.03-0.05), so the model is not overfitting and there is no data leakage.
The decoder is trained on the whole trial at once (all 80 bins, all sessions pooled), which averages over the
long pre-go period during which choice/outcome are only weakly represented; the time-resolved analysis below
shows that the information is there where the papers say it should be.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance analysis
| Output | Validation acc | Chance | Ratio |
|--------|----------------|--------|-------|
| lick_direction | 0.683 | 0.333 | 2.05x |
| outcome | 0.661 | 0.333 | 1.98x |
| early_lick | 0.751 | 0.500 | 1.50x |
| tongue_y_position | 0.651 | 0.250 | 2.60x |
No output is below chance and none is below 1.5x chance. `early_lick` has the lowest ratio, which is expected:
the early lick happens **before** the analysis window in many trials (the window starts 2.5 s before the go cue
while the sample epoch can be replayed several times), and only 11.6% of trials are early-lick trials.

### Check 2: Accuracy comparison to the papers
The papers do not decode these variables with the same architecture, but they do report decoding of **choice**:
| Variable | Paper value | This dataset | Comparison |
|----------|-------------|--------------|------------|
| Choice (AUC) from video, pre-sample | 0.51 +- 0.06 | neural AUC 0.49-0.55 at t = -2.5 to -2.0 s | matches |
| Choice (AUC) from video, sample+delay | 0.66 +- 0.12 | neural AUC 0.54-0.67 | matches |
| Choice (AUC) from video, response epoch | 0.99 +- 0.01 | neural AUC 0.97 / 0.99 at +0.3 s, 0.82-0.95 later | matches |
These numbers come from `/app/cache/timecourse_check.py`, a per-timepoint logistic-regression (PCA 16 + 5-fold CV,
the same procedure as `population_decoding_utils.nested_cross_validation`) on the converted neural data of two
sessions, restricted to the paper's trial subset (lick trials, no early lick, no photostim). The rise of the AUC
exactly at the go cue is a strong check on the temporal alignment.
The balanced accuracy of the provided decoder (0.68 for a 3-way choice over all 80 time bins, including the 2.5 s
before the go cue when choice is barely represented) is consistent with these AUCs.

### Check 3: Train vs validation gap
Maximum ratio train/validation = 0.712/0.683 = 1.04 (lick_direction); all outputs < 1.1. No overfitting.

### Additional debugging checks performed
1. Output values verified against the raw data for individual trials (Step 10 Check 2): choice agrees with the
   first post-go lick on 247/247 spot-checked trials.
2. Temporal alignment verified in three independent ways: the population PSTH jumps at t = 0; the tongue becomes
   visible only after t = 0; the choice AUC jumps from ~0.6 to ~0.98 in the bin after t = 0.
3. Output class balance checked: no output is dominated by a single class beyond what the experiment implies
   (hit 0.685 / miss 0.167 / ignore 0.148; tongue not-visible 0.755 which is inherent - the tongue is only out of
   the mouth while licking).
4. Neuron filtering re-verified against the paper's per-region unit counts (Step 9 table).
5. Processing compared step by step with the reference code (Step 10 Check 3).

### Issues Found and Resolved
- Trials outside the ephys recording window produced all-zero neural data -> restricted trials to `obs_intervals`.
- The last trial of a truncated recording had no spikes -> trials with zero spikes over all neurons are dropped.
- `np.diff` bug across trial boundaries in the first version of the binning code -> fixed and verified by the
  independent spike-count sanity check.
- `anno_name` occasionally stored as a float -> robust decoding helper.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, format specification, loading example, key statistics, decoder results)
- [x] `cache/` folder created with `README_CACHE.md` documenting every exploration and validation script
- [x] All required files present:
  `CONVERSION_NOTES.md`, `convert_data.py`, `converted_data.pkl`, `sample_data.pkl`, `README.md`,
  `conversion_sample_out.txt`, `verification_sample_out.txt`, `train_decoder_sample_out.txt`,
  `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt`,
  plus the `--show-processing` figures `processing_<session_id>.png`.
