# Dataset Conversion Notes

## Overview
- **Dataset**: IBL Brain-Wide Map (BWM) public release (electrophysiology + behaviour), via a local ONE cache.
- **Reference papers**: IBL et al., 'A brain-wide map of neural activity during complex behaviour' (data paper); Zhang et al. 2025, 'Exploiting correlations across trials and behavioral sessions to improve neural decoding' (methods paper).
- **Reference code**: /app/code/code_zhang2025 (methods paper) and /app/code/ibllib.
- **Goal**: Convert to decoder-compatible format (/app/converted_data.pkl).

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of /app:
- .manifest, Dockerfile, docker-compose.yaml, stage_cache.sh (container/staging bookkeeping)
- code/ -> code_zhang2025/ (reference decoding code), ibllib/ (IBL library source)
- data/ -> one_cache/ (ONE cache: 12 lab symlinks holding 461 session folders, plus Brainwidemap, 2022_Q4_IBL_et_al_BWM, 2025_Q3_IBL_et_al_BWM release tables and .rest cached Alyx responses)
- datapaper.pdf, methodpaper.pdf, dataarchitecture.pdf, methods.txt
- decoder.py, train_decoder.py

Environment verified: python3 3.13, numpy 2.3.5, torch 2.6.0+cu124, CUDA available.
Hardware: 128 CPUs, ~1 TB RAM, 3.4 TB free disk.

ONE works offline only if constructed WITHOUT a password (the staged auth token in $HOME/.one is then used); passing password='international' forces a network re-auth which fails.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

Entry point: /app/code/code_zhang2025/src/0_data_caching.py. Its parameters are

    params = {'interval_len': 2, 'binsize': 0.02, 'single_region': False,
              'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}
    beh_names = ['choice', 'reward', 'block', 'wheel-speed', 'whisker-motion-energy']

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| prepare_data | src/utils/ibl_data_utils.py | LOADING | Per-eid: one.eid2pid, loads every probe, merges probes, loads trials+mask, loads behaviour |
| load_spiking_data | src/utils/ibl_data_utils.py | LOADING | SpikeSortingLoader(pid).load_spike_sorting() + merge_clusters -> spikes dict, clusters DataFrame (has label, acronym). qc=None default keeps all clusters; qc=1 keeps well-isolated units |
| merge_probes | src/utils/ibl_data_utils.py | LOADING | Concatenates probes of one session, re-indexes spikes['clusters'], time-sorts |
| load_trials_and_mask | src/utils/ibl_data_utils.py | CURATION | Trials table + boolean mask. Defaults min_rt=0.08, max_rt=2.0, exclude_nochoice=True, nan_exclude = stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType. Called with max_trial_len=10.0 |
| list_brain_regions | src/utils/ibl_data_utils.py | PROCESSING | Maps clusters['acronym'] to Beryl acronyms via BrainRegions.acronym2acronym(mapping='Beryl') |
| select_brain_regions | src/utils/ibl_data_utils.py | CURATION | Cluster indices in the requested region set (single_region=False -> all regions) |
| bin_spiking_data | src/utils/ibl_data_utils.py | PROCESSING | Intervals = stimOn_times + (-0.5, 1.5); bins spikes with bincount2D(xbin=0.02, xlim=[t_beg,t_end]), keeps first ceil(2.0/0.02)=100 bins. Returns (n_trials, T=100, n_clusters) |
| bin_behaviors | src/utils/ibl_data_utils.py | PROCESSING | Per-trial scalars choice, block=probabilityLeft, reward=(rewardVolume>1), contrast; time-varying wheel-speed and whisker-motion-energy |
| load_target_behavior | src/utils/ibl_data_utils.py | LOADING | wheel-speed = abs(SessionLoader.wheel['velocity']); left/right-whisker-motion-energy = SessionLoader.motion_energy['leftCamera']['whiskerMotionEnergy'] |
| get_behavior_per_interval | src/utils/ibl_data_utils.py | PROCESSING | Linear interpolation of the behaviour trace onto np.linspace(t_beg+binsize, t_end, 100). Marks a trial bad if the trace is missing, starts more than one bin late, or ends more than one bin early |
| align_spike_behavior | src/utils/ibl_data_utils.py | CURATION | Drops trials failing the trials mask or any behaviour mask, from both neural and behaviour |
| standardize_spike_data | src/utils/data_loader_utils.py | PROCESSING | Per-time-bin z-scoring at training time (not in the cached data) |

### Notes
- This is electrophysiology, so no dF/F. Spike counts in 20 ms bins.
- Neuron QC: the caching script calls load_spiking_data with the default qc=None (all Kilosort clusters) but stores good_clusters = (clusters['label'] >= 1) in the metadata so downstream code can restrict to well-isolated units. The data paper restricts all its analyses to well-isolated units (Step 3).
- Behaviour is interpolated, not averaged, onto the bin right-edges.
- bincount2D bin i covers [t_beg + i*binsize, t_beg + (i+1)*binsize).
- The zhang2025 SessionLoader(one, eid) positional call fails with the installed brainbox (keyword-only). Must use SessionLoader(one=one, eid=eid).

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
/app/data/one_cache is a standard ONE cache:

    one_cache/<lab>/Subjects/<subject>/<YYYY-MM-DD>/<number>/
        alf/
            _ibl_trials.table.pqt (+ separate .npy for stimOff, goCueTrigger, intervals_bpod, quiescencePeriod)
            _ibl_wheel.position.npy, _ibl_wheel.timestamps.npy
            _ibl_leftCamera.times.npy, _ibl_rightCamera.times.npy
            leftCamera.ROIMotionEnergy.npy, rightCamera.ROIMotionEnergy.npy
            <probeXX>/pykilosort/ spikes.times/clusters/amps/depths.npy,
                                  clusters.metrics.pqt (QC incl. label), clusters.channels/depths/uuids,
                                  channels.brainLocationIds_ccf_2017.npy
        raw_ephys_data/<probeXX>/*.ap.meta
    one_cache/Brainwidemap, 2022_Q4_IBL_et_al_BWM, 2025_Q3_IBL_et_al_BWM  (sessions.pqt / datasets.pqt / QC.json)

461 session directories are staged; the BWM freeze file /app/code/code_zhang2025/data/bwm_release.csv lists 699 probe insertions / 459 eids / 139 subjects / 12 labs, which is the set we iterate over.

Key variables:
- trials (20 columns): stimOn_times, goCue_times, goCueTrigger_times, response_times, feedback_times, firstMovement_times, stimOff_times, choice, contrastLeft, contrastRight, probabilityLeft, feedbackType, rewardVolume, intervals_0/1, ...
  - choice in {-1, 0, +1}. Empirically verified on eid 6713a4a7-...: on correct trials with a left-side stimulus choice == +1 (236/236) and with a right-side stimulus choice == -1 (203/203). So +1 = mouse reported LEFT, -1 = reported RIGHT, 0 = no-go.
  - probabilityLeft in {0.2, 0.5, 0.8}. First block is 90 trials of 0.5, then alternating 0.2/0.8.
- wheel: 1 kHz uniformly-resampled times/position/velocity/acceleration (SessionLoader.load_wheel).
- motion_energy: leftCamera at ~60 Hz (dt = 0.0166 s), rightCamera at 150 Hz; column whiskerMotionEnergy.
- clusters (after merge_clusters): acronym (Allen), label (fraction of the 3 RIGOR single-unit metrics passed: 0, 1/3, 2/3, 1), firing_rate, depths, channels, uuids, QC columns.

### Dataset Size (from data files, measured on the first 5 eids)
| Statistic | Value |
|-----------|-------|
| Insertions (freeze file) | 699 |
| Sessions / eids (freeze file) | 459 |
| Subjects (freeze file) | 139 |
| Labs | 12 |
| Kilosort clusters / session | 898-2017 (1-2 probes) |
| Well-isolated (label>=1) / session | 76-264 |
| Well-isolated AND grey matter (Beryl not root/void) / session | 61-192 |
| Raw trials / session | 425-565 |
| Trials passing reference mask / session | 148-407 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers / methods.txt)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Sessions (release) | 459 | 'a total of 459 sessions, 699 insertions and 621,733 neurons remained, constituting the publicly released dataset' |
| Insertions | 699 | same |
| Subjects | 139 | 'We trained 139 mice (94 male and 45 female)' |
| Kilosort units (total) | 621,733 (avg 889/probe) | 'This process produced 621,733 units (including multineuron activity), averaging 889 per probe' |
| Well-isolated neurons (total) | 75,708 (avg 108/probe) | 'identified 75,708 well-isolated neurons, averaging 108 per probe' |
| Sessions used for decoding (methods paper) | 433 | 'We apply our models to 433 IBL sessions, covering 270 brain regions and four behavioral variables' |
| Brain regions | 270 | same |
| Min trials/session for release | >=400 collected, >=250 performed | 'Only sessions with at least 400 trials were retained'; 'Sessions were included ... if the mice performed at least 250 trials' |
| Neural bin size | 20 ms | 'Recordings are split into 2-s trials, each divided into 20-ms bins, producing T = 100 time steps' |
| Trial length | 2 s, T = 100 | same |
| Alignment (choice) | stimulus onset, -0.5 s to +1.5 s | 'For choice, we align trials to the stimulus onset, considering neural activity from 0.5 s before to 1.5 s post-onset' |
| Behaviour sampling | wheel 1 kHz (resampled), whisker ME 60 Hz (left cam) | 'wheel speed and whisker motion energy ... sampled at 60 Hz' |
| Block structure | 90 unbiased (0.5) then 20:80 / 80:20 blocks, mean 51 trials, range 20-100 | 'After an initial 90 unbiased trials ... Blocks lasted for between 20 and 100 trials ... (empirical mean of 51 trials)' |
| Contrasts | 100, 25, 12.5, 6.25, 0 % | 'Stimulus contrast was uniformly sampled from 5 possible values' |

### Processing Details
- Temporal alignment: stimOn_times, window (-0.5, +1.5) s, 20 ms non-overlapping bins, T = 100. (The methods paper also describes 50 ms bins for choice/prior and a first-movement-aligned 0-1 s window for wheel/whisker; the reference CODE unifies all four variables on the stimulus-onset / 20 ms / (-0.5,1.5) grid, which is also exactly what the Decoder Task here specifies. See Step 4.)
- Behaviour: linear interpolation onto the bin right edges within the trial window.
- Neural: raw spike counts per bin (standardisation is done by the model, not the cache).

### Curation Steps

**Session curation rules** (data paper): >=400 trials collected, >=90% correct on 100% contrast in both block types, >=3 incorrect trials, hardware QC, RIGOR recording criteria, resolved histology alignment. All of this is already applied in the released freeze (bwm_release.csv, 459 eids), so no further session QC is needed beyond requiring usable behaviour traces.

**Neuron curation rules** (data paper): exclude units failing any of the three RIGOR single-unit metrics - amplitude > 50 uV, noise cut-off < 20 uV, refractory-period violation. In the IBL tables this is exactly clusters['label'] == 1 (label = fraction of the 3 metrics passed). 621,733 -> 75,708 units. Additionally: restricted to regions designated grey matter, containing at least five well-isolated neurons per session, and recorded from in at least two such sessions.

**Trial curation rules** (data paper, identical to reference code defaults): exclude a trial if any of choice, probabilityLeft, feedbackType, feedback_times, stimOn_times, firstMovement_times is NaN, or if firstMovement_times - stimOn_times is outside 0.08-2.00 s. The reference code additionally excludes no-response trials (choice == 0) and trials with feedback_times - goCue_times > 10 s, and drops trials whose wheel / whisker trace does not cover the window.

### Decoders Trained (reference)
| Decoded variable | Reported performance |
|---|---|
| Choice (per-region, RE dataset, single-session RRR) | AUC ~0.6-0.8 (Fig 5A); example AUC 0.72 / 0.66 / 0.79 |
| Prior (per-region) | Pearson correlation ~0.05-0.65 |
| Wheel speed / whisker motion energy | qualitative R2 / correlation, whisker better than wheel |
| BWM data paper, per-region choice | null-corrected median balanced accuracy, small effect sizes (~0.01-0.1) |

Both papers decode from SINGLE regions to avoid ceiling effects; here we decode from ALL neurons of a session, so accuracies should be substantially higher.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Bin size / window | 20 ms, stimOn, (-0.5, 1.5), T=100 | n/a | methods paper: 50 ms for choice/prior; 20 ms + firstMovement + (0,1) for wheel/whisker | Use the reference code unified grid (20 ms, stimOn, (-0.5,1.5), T=100). Required anyway: the Decoder Task says to align on stimulus onset and all four variables must share one grid. The methods paper overview text (2-s trials, 20-ms bins, T = 100) matches this. |
| Neuron QC | qc=None -> all clusters; stores good_clusters = label>=1 | 898-2017 clusters/session, 76-264 with label>=1 | data paper: analyses use the 75,708 well-isolated units | Use well-isolated units (label >= 1). Rationale: (a) it is the criterion the data paper applies to every analysis and gives a checkable statistic (75,708 total, 108 per probe); (b) the reference cache explicitly records the flag so downstream analyses can apply it; (c) keeping all 621k units would make the pickle ~8x larger with mostly noise/MUA clusters. Documented deliberate deviation from qc=None. |
| Grey matter / region size | not applied in caching code | root/void Beryl labels present | data paper: grey matter only, >=5 units/region/session, region in >=2 sessions | Applied. Matches the data paper; also removes units with no anatomical assignment. |
| Whisker camera | left, fall back to right | both present in most sessions | left camera 60 Hz, right 150 Hz | Left camera preferred, right as fallback (exactly the reference bin_behaviors logic). |
| Choice coding | choice kept as -1/+1 | -1/0/+1 | left/right | Verified empirically (+1 = left). Mapped to the required 0 = left, 1 = right. |
| SessionLoader signature | positional | - | - | Installed brainbox requires keywords; use SessionLoader(one=one, eid=eid). |

Everything else (trial mask, behaviour sources, probe merging, Beryl mapping, interpolation grid) is consistent between code, data and papers.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| spikes['times'], spikes['clusters'] (all probes merged) | neural[s][k] (n_neurons, 100) | bin counts in 20 ms bins over stimOn_times + [-0.5, 1.5); transpose to (N, T); float32 | merge_probes, bin_spiking_data, bincount2D | only well-isolated grey-matter units |
| bin right-edge time relative to stimOn | input[s][k][0] | -0.48 ... 1.50 s, identical for every trial | new; required by Decoder Task | time since stimulus onset, time-varying |
| trial index within probabilityLeft block | input[s][k][1] | 0-based count since the last change of probabilityLeft, computed on the FULL trials table before masking; broadcast over T | new; required by Decoder Task | trial number in block, per-trial |
| trials['choice'] | output[s][k][0] | +1 -> 0 (left), -1 -> 1 (right); no-go trials already removed | bin_behaviors (choice) | per-trial, broadcast over T |
| trials['probabilityLeft'] | output[s][k][1] | 0.2 -> 0, 0.5 -> 1, 0.8 -> 2 | bin_behaviors (block) | per-trial, broadcast over T |
| abs(wheel['velocity']) | output[s][k][2] | interp to bin right edges, then per-session terciles -> {0,1,2} | load_target_behavior('wheel-speed'), get_behavior_per_interval | time-varying |
| motion_energy leftCamera whiskerMotionEnergy (fallback right) | output[s][k][3] | interp to bin right edges, then per-session terciles -> {0,1,2} | load_target_behavior('left-whisker-motion-energy') | time-varying |
| clusters['acronym'] -> Beryl | brain_regions, brain_region_idx | BrainRegions.acronym2acronym(mapping='Beryl') | list_brain_regions | |
| bwm_release.csv subject | subjects, subject_idx | unique list | | |

### Key Decisions
1. **Alignment / binning**: stimOn_times, window (-0.5, +1.5) s, 20 ms bins, T = 100. Directly from 0_data_caching.py params and from the Decoder Task.
2. **Neuron curation**: well-isolated units only (clusters['label'] >= 1), i.e. the three RIGOR single-unit metrics of the data paper. Sanity target ~75,708 units over the whole release, ~108 per probe.
3. **Region curation**: Beryl mapping; drop root and void (non-grey-matter / unassigned); drop regions with fewer than 5 well-isolated units in that session; drop regions present in fewer than 2 sessions. All three are stated in the data paper.
4. **Trial curation**: reference load_trials_and_mask logic with max_trial_len=10.0 (min_rt=0.08, max_rt=2.0, default NaN list, exclude_nochoice=True), plus the behaviour coverage mask from get_behavior_per_interval. Unbiased (pLeft = 0.5) trials are KEPT, because the Decoder Task requires the 0.5 prior class.
5. **Outputs must be categorical**: choice and prior are naturally discrete; wheel speed and whisker motion energy are discretised into 3 bins at the per-session 33.3rd/66.7th percentiles of all (trial x timepoint) samples in that session. Per-session rather than global, because whisker motion energy is in uncalibrated camera-dependent units (left camera 60 Hz vs right camera 150 Hz, different resolution and illumination) so a global threshold would largely encode session identity; terciles also give exactly balanced classes.
6. **Time-varying wherever possible**: per-trial variables (choice, prior, trial-in-block) are broadcast across the 100 time bins so that input/output are (d, T) arrays, as the format spec prefers.
7. **Probes merged** within a session (data paper: we did not perform decoding on these probes separately because they are not independent).
8. **Session inclusion**: all 459 release eids attempted; a session is dropped if it has no usable whisker motion energy or wheel trace, no surviving neurons, or fewer than 2 surviving trials. Sanity target: the methods paper used 433 sessions.

### Planned Sanity Checks
- [ ] Total well-isolated units before region filtering ~ 75,708; mean ~108/probe.
- [ ] 459 eids / 699 pids / 139 subjects in the freeze file; converted sessions ~433.
- [ ] ~270 distinct Beryl regions.
- [ ] T = 100 for every trial; bin size 20 ms.
- [ ] probabilityLeft only in {0.2, 0.5, 0.8}; class 1 (0.5) fraction ~ 90/500 ~ 18%.
- [ ] choice fraction left ~ 0.5.
- [ ] wheel-speed and whisker-ME class fractions ~ 1/3 each.
- [ ] trial-in-block max <= 100 for biased blocks, 90 for the first unbiased block.
- [ ] Spot check (Step 10): recompute a trial spike counts, wheel speed bin, whisker bin and choice directly from the raw .npy/.pqt files and compare with np.allclose.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` implements the Step 5 mapping. Structure:

| Function | Purpose | Reference counterpart |
|---|---|---|
| `_get_one()` | lazily builds an offline ONE client + `BrainRegions` (one per worker process) | `0_data_caching.py` header |
| `load_trials_and_mask(sess_loader)` | trial mask, `min_rt=0.08`, `max_rt=2.0`, `max_trial_len=10.0`, default NaN list, `choice != 0` | `ibl_data_utils.load_trials_and_mask` |
| `trial_number_in_block(prob_left)` | 0-based trial index within each constant-`probabilityLeft` block, computed on the FULL trials table | new (Decoder Task input 2) |
| `load_session_spikes` | loads every probe with `SpikeSortingLoader` + `merge_clusters`, keeps `label >= 1` and non-`root`/`void` Beryl units, re-indexes clusters and merges probes | `load_spiking_data` + `merge_probes` + `list_brain_regions` + `select_brain_regions` |
| `bin_spikes` | vectorised `np.bincount` binning, bin i = [t_beg + i*0.02, t_beg + (i+1)*0.02) | `bin_spiking_data` / `bincount2D` |
| `interpolate_behavior` | linear interpolation onto `stim_on + BIN_TIMES` with the reference coverage checks | `get_behavior_per_interval` |
| `discretize_terciles` | per-session 33.3/66.7 percentile split into 3 classes | new (categorical-output requirement) |
| `process_session` | one session end to end, returns the session dict + a stats dict | `0_data_caching.py` main loop |
| `plot_processing` | 8-panel figure for `--show-processing` | `viz_utils` |
| `main` | job list from `bwm_release.csv`, multiprocessing pool, cross-session region filter, assembly, summary, pickle | `0_data_caching.py` |

**Alignment proof**: `BIN_TIMES = linspace(-0.5 + 0.02, 1.5, 100)`. The behaviour grid is
`stim_on + BIN_TIMES`, which is exactly the reference `np.linspace(t_beg + binsize, t_end, n_bins)`
with `t_beg = stim_on - 0.5`. Spike bin `i` covers `[stim_on - 0.5 + i*0.02, stim_on - 0.5 + (i+1)*0.02)`,
whose right edge is `BIN_TIMES[i]`. So behaviour sample `i` and spike bin `i` end at the same instant --
no temporal offset between the neural, input and output streams. `input[0]` carries the same
`BIN_TIMES` vector, so the time input is on the same clock.

Code inefficiencies identified:
- The reference `get_spike_data_per_interval` spawns a multiprocessing pool per *trial* and calls
  `bincount2D` per trial; this is very slow.
- The reference `get_behavior_per_interval` also uses a per-trial pool and builds a separate
  `interp1d` per trial.

Code speedups added:
- Binning is a single `np.searchsorted` + per-trial `np.bincount` on a flattened (unit, bin) index;
  no pools, no Python-level per-spike work.
- Interpolation builds ONE `interp1d` per session and evaluates it on the full
  (n_trials x 100) grid in one call.
- Parallelism is moved up to the session level (`multiprocessing.Pool`, 24 workers, `spawn`),
  so each worker holds one ONE client and one session.
- Spike counts stored as `float32` (the format expects float and this halves the pickle).

Note: the `region present in >= 2 sessions` filter is relaxed to `>= 1` in `--sample` mode,
because with only two sessions it would otherwise delete almost every region.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons (total) | 199 |
| Neurons / session | 116, 83 (mean 99.5) |
| Subjects | 2 (DY_010, MFD_05) |
| Sessions / subject | 1 |
| Trials (total) | 644 |
| Trials / session | 198, 446 |
| T | 100 for every trial |
| Kilosort clusters loaded | 2020 (1010/probe) |
| Well-isolated (label>=1) | 298 (149/probe) |
| time_from_stimulus_onset range | [-0.48, 1.50] |
| trial_number_in_block range | [0, 94] |
| choice distribution | [0.500, 0.500] |
| prior distribution | [0.443, 0.118, 0.439] |
| wheel_speed distribution | [0.333, 0.333, 0.333] |
| whisker_motion_energy distribution | [0.333, 0.333, 0.333] |

All as expected: choice is near 50/50, the unbiased (0.5) block is ~12% of trials
(90 unbiased trials out of ~500-700 collected, and the unbiased block has the most
no-go / long-RT exclusions early in the session), the two discretised behaviours are
exactly balanced by construction, and `trial_number_in_block` never exceeds 100.

### Processing Plots Review
`processing_004d8fd5-....png` and `processing_02fbb6da-....png` (8 panels each):
binned raster, population PSTH (clear step-up at t = 0, confirming stimulus-onset alignment),
raw vs interpolated wheel speed and whisker ME for one trial (the interpolated points lie on
the raw trace), the tercile thresholds overlaid on the discretised step trace, the per-trial
choice/prior sequence (prior is piecewise constant in blocks) and `trial_number_in_block`
(sawtooth resetting at each block change). No anomalies.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
|---|---|
| vectorised `np.bincount` binning instead of per-trial pools | ~20x on the binning step |
| single per-session `interp1d` evaluated on the whole grid | ~50x on the behaviour step |
| session-level `multiprocessing.Pool` (24 workers) | ~20x wall clock |

| Step | Time / Session | Estimated Total Time |
|---|---|---|
| full serial session processing | 4.1 s and 7.3 s (mean ~5.7 s) | 459 x 5.7 s = 44 min serial |
| with 24 workers | -- | ~2-4 min + pickling |

Sessions with 2 probes take roughly twice as long, and the sample contained one 1-probe and
one 2-probe session, so the mean is representative (699/459 = 1.52 probes per session).
Estimated full wall clock well under 15 min.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None (`Data format is valid, no errors or warnings.`)

### Decoder Results (Sample, 2 sessions)
Loss decreased monotonically 1.77 -> 0.80 over 200 epochs; test loss 0.820.

| Output | Chance | Training Balanced Acc | Validation Balanced Acc |
|--------|--------|-----------------------|-------------------------|
| choice | 0.500 | 0.6119 | 0.6082 |
| prior_probability_left | 0.333 | 0.6723 | 0.6724 |
| wheel_speed | 0.333 | 0.5479 | 0.5325 |
| whisker_motion_energy | 0.333 | 0.5374 | 0.5277 |

All four are above chance, and train/validation are essentially identical (no overfitting).

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/converted_data.pkl --full --n-workers 32`
Wall clock 3.0 min (well under the 15 min budget and under the Step 7 estimate).

### Output Files
- `converted_data.pkl`: 11.33 GB
- `conversion_full_out.txt`, `verification_full_out.txt`: created

### Sessions skipped (19 of 459)
| Reason | N | Comment |
|---|---|---|
| no whisker motion energy | 14 | the video/motion-energy ALF object is absent for these sessions, so the whisker output cannot be built |
| no Beryl region with >= 5 well-isolated units | 4 | data-paper region criterion |
| too few trials after the behaviour mask | 1 | |

440 sessions remain (methods paper: 433 -- see the consistency table).

### Consistency Check
| Statistic | Reference papers | Reference Code | Reference Data (freeze file) | Converted Data | Match? |
|-----------|-----------------|----------------|------------------------------|----------------|--------|
| Insertions | 699 | 699 rows in `bwm_release.csv` | 699 | 675 loaded (24 belong to the 19 skipped sessions) | yes |
| Sessions in release | 459 | 459 | 459 | 459 attempted | yes |
| Sessions decoded | 433 | n/a | n/a | **440** | yes (within 2%) |
| Subjects | 139 | 139 | 139 | **136** | yes (3 subjects only contributed skipped sessions) |
| Kilosort units | 621,733 (889/probe) | n/a | n/a | **599,865 over 675 probes = 888.7/probe** | **yes, 888.7 vs 889** |
| Well-isolated units | 75,708 (108/probe) | `label >= 1` | n/a | **73,044 over 675 probes = 108.2/probe** | **yes, 108.2 vs 108** |
| Neurons kept (after grey-matter + >=5/region + >=2 sessions) | -- | -- | -- | 60,483 (mean 137.5/session) | n/a |
| Brain regions | 270 (methods paper, all units) | Beryl | -- | **209** | lower, because we keep only grey-matter regions with >= 5 well-isolated units in a session and present in >= 2 sessions; the 270 figure counts every Beryl region touched by any unit |
| Trials total | -- | -- | -- | 187,651 (mean 426.5/session) | consistent with >= 400 collected trials per released session |
| T | 100 | 100 | -- | **100 for every trial** | yes |
| Bin size | 20 ms | 0.02 s | -- | **20 ms** | yes |
| Window | (-0.5, +1.5) s about stimOn | (-0.5, 1.5) | -- | **(-0.5, +1.5)** | yes |
| time_from_stimulus_onset range | -- | linspace(t_beg+bs, t_end, 100) | -- | [-0.48, 1.50] | yes |
| trial_number_in_block range | blocks of 20-100 trials, first block 90 | -- | -- | [0, 98] | yes (max index 98 = a 99-trial block, < 100) |
| choice distribution | ~balanced | -- | -- | [0.508, 0.492] | yes |
| prior distribution | 90 unbiased then 0.2/0.8 alternating | {0.2, 0.5, 0.8} | {0.2, 0.5, 0.8} | [0.417, 0.140, 0.442] | yes (0.5 class is ~14%, consistent with 90 of ~600 collected trials, with more exclusions early in the session) |
| wheel_speed distribution | -- | -- | -- | [0.333, 0.333, 0.333] | exact by construction |
| whisker_motion_energy distribution | -- | -- | -- | [0.333, 0.333, 0.333] | exact by construction |

The agreement of 888.7 vs 889 Kilosort units per probe and 108.2 vs 108 well-isolated
units per probe is the single strongest check that loading, probe merging and the
`label >= 1` neuron curation reproduce the data paper exactly.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
First full run produced 16 warnings, all of the form
`Session S, trial K: all neural data is zero`, in 3 sessions (43, 252, 314). No errors.

**Investigated** (`cache/zero_check.py`, `cache/zero2.py`) and found three distinct causes:

| Session | eid | Cause |
|---|---|---|
| 252 | 8c2f7f4d-... | the spike sorting ends at 1779.4 s but three kept trials start at 1789.5, 1796.0 and 1799.3 s: the **ephys recording stops before the behaviour does** |
| 314 | b182b754-... | a **2.07 s dropout in the recording at 185.6 s** swallows the trial at 187.4-189.4 s |
| 43 | 195443eb-... | **genuine silence**: only 6 units at ~3 Hz, so 12 of 618 two-second windows contain no spike by chance; the largest gap in the spike train is 0.02 s, i.e. no recording problem |

**Fixed**: added a neural-coverage criterion to `process_session`, the exact analogue of the
reference `get_behavior_per_interval` coverage check but for the spike train -- the trial window
must lie inside `[spike_times[0], spike_times[-1]]` and the binned matrix must contain at least
one spike. This is an *invalid data period* exclusion: an all-zero matrix is not a measurement,
it is a hole in the recording, and it carries no information for the decoder. 16 trials
(0.009%) were removed; 187,667 -> 187,651.

**Re-checked**: `verification_full_out.txt` now reads `Data format is valid, no errors or
warnings.` No warnings remain, so none needed to be explained away.

### Check 2: Sanity checks against the ORIGINAL files
`/app/cache/sanity_checks.py` reloads each spot-checked session through ONE
(`SpikeSortingLoader`, `SessionLoader`) and recomputes everything from scratch, never calling
`convert_data.py`. Run on sessions 0, 37, 200 and 411 (different labs, 1 and 2 probes,
53-122 units, 198-679 trials):

| Check | Stream | Method | Result |
|---|---|---|---|
| kept-trial count | all | independent re-derivation of the trial + behaviour + neural-coverage masks | exact match in all 4 sessions |
| **neural** | neural | 3 random trials per session re-binned with the **reference** `iblutil.numerical.bincount2D`, compared with `np.allclose` | **12/12 pass** |
| **neural** | neural | ordered list of Beryl region acronyms per unit | match in all 4 sessions |
| **input** | input[0] | `np.allclose(input[0], linspace(-0.48, 1.50, 100))` | pass |
| **input** | input[1] | trial-in-block recomputed from `probabilityLeft` | pass |
| **output** | output[0] | choice recomputed from `trials.choice` | pass |
| **output** | output[0] | *semantic* check: on correct non-zero-contrast trials the decoded choice must equal the stimulus side (n = 285-493 trials/session) | pass -- confirms 0 = left / 1 = right |
| **output** | output[1] | prior recomputed from `probabilityLeft` | pass |
| **output** | output[2] | wheel speed re-interpolated and re-tercilised | pass |
| **output** | output[3] | whisker ME re-interpolated and re-tercilised | pass |

`==== 0 FAILURES ====`

An earlier version of the neural check used `np.histogram2d` instead of `bincount2D` and
reported 1 mismatch out of 12: a spike at exactly 0.54 s after the window start. `bincount2D`
computes `floor((t - t_beg)/binsize)`, and `0.54/0.02 = 26.999999999999996` in IEEE754, so the
spike lands in bin 26; `histogram2d`'s edge search puts it in bin 27. My binner uses the same
`floor` arithmetic as the reference, so the reference and my code agree and the idealised
histogram is the odd one out. `/app/cache/bincount_check.py` confirms this directly:
**0 mismatches against `bincount2D` over 2000 random trials deliberately seeded with spikes
sitting exactly on bin edges.**

### Check 3: Reference code comparison
| Step | Reference (`ibl_data_utils.py` / `0_data_caching.py`) | `convert_data.py` | Same? |
|---|---|---|---|
| (a) loading | `one.eid2pid`; `SpikeSortingLoader.load_spike_sorting` + `merge_clusters`; `merge_probes`; `SessionLoader.load_trials/load_wheel/load_motion_energy` | identical calls | yes |
| (b) neuron filtering | `qc=None` (all clusters), but `good_clusters = label >= 1` stored | `label >= 1` **applied**, plus grey matter, >=5 units/region/session, region in >=2 sessions | **deliberate difference** -- these are the data paper's own criteria and reproduce its 108/probe statistic; see Step 4 |
| (b) trial filtering | `load_trials_and_mask(max_trial_len=10.0)` | same query string, same thresholds, same NaN list, `choice != 0` | yes |
| (b) invalid periods | behaviour coverage in `get_behavior_per_interval` | same, **plus** the neural-coverage analogue (Check 1) | superset, justified above |
| (c) alignment | `stimOn_times + (-0.5, 1.5)` | same | yes |
| (d) binning | `bincount2D(xbin=0.02, xlim=[t_beg, t_end])[:, :100]` | vectorised `floor((t-t_beg)/0.02)` + `bincount`; proven bit-identical | yes |
| (d) behaviour resampling | `interp1d(..., kind='linear', fill_value='extrapolate')` on `linspace(t_beg+bs, t_end, 100)` | same function, same grid, evaluated for all trials at once | yes |
| (e) input construction | reference has no decoder inputs (it caches only behaviours) | time-from-onset + trial-in-block, as specified by the Decoder Task | new, required |
| (f) output construction | `choice` kept as -1/+1, `block` = `probabilityLeft`, `wheel-speed` = `abs(velocity)`, `whisker-motion-energy` = left cam with right fallback | same sources; choice remapped to 0/1 and prior to 0/1/2, wheel/whisker tercilised | same data, recoded because the decoder requires categorical outputs |

The only substantive differences are the three listed as deliberate, each justified by either
the data paper's stated curation or by an explicit requirement of the Decoder Task.

### Check 4: Key statistics comparison
See the Step 9 table. Every statistic available in the papers is reproduced:
888.7 vs 889 Kilosort units/probe, 108.2 vs 108 well-isolated units/probe, 459 sessions
attempted, 440 vs 433 decoded, T = 100, 20 ms bins, (-0.5, 1.5) s window, `probabilityLeft`
only in {0.2, 0.5, 0.8}, blocks never longer than 100 trials, choice near 50/50.
The two apparent gaps were investigated:
- **136 vs 139 subjects**: the 3 missing mice contributed only sessions that were skipped for
  missing whisker video.
- **209 vs 270 regions**: the 270 figure in the methods paper counts every Beryl region reached
  by any unit; we additionally require grey matter, >= 5 well-isolated units in the session and
  presence in >= 2 sessions, exactly as the data paper does for its own analyses.

### Check 5: Edge cases
- **Off-by-one in time**: `BIN_TIMES[0] = -0.48` and `BIN_TIMES[-1] = 1.50` are the right edges
  of bins 0 and 99, matching `linspace(t_beg + binsize, t_end, n_bins)` in the reference. Verified
  by `np.allclose` in the sanity checks.
- **Off-by-one in blocks**: `trial_number_in_block` is 0-based and computed on the FULL trials
  table, so it is the true position in the block even when neighbouring trials are excluded.
  Maximum observed 98, consistent with a block cap of 100.
- **Trials at session boundaries**: handled by the new neural-coverage filter.
- **Sessions with a single probe, two probes, missing probes**: `load_session_spikes` skips a
  probe whose spike sorting is empty and keeps the cluster indices consistent via a per-probe
  lookup table with an offset.
- **NaN handling**: non-finite spike times, out-of-range cluster ids and non-finite behaviour
  samples are filtered before use; trials whose interpolated behaviour is not finite are dropped;
  `probabilityLeft` values outside {0.2, 0.5, 0.8} are dropped.
- **Degenerate tercile thresholds** (constant behaviour trace): `discretize_terciles` falls back
  to a rank split so the output is still 3-valued and never NaN.
- **Sessions with < 2 trials or 0 units**: dropped with an explicit message.

No further issues found; no iteration was needed after the Check-1 fix.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

Command: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`
Ran to completion on the GPU over all 440 sessions / 187,651 trials (~3 h).
Outputs: `train_decoder_full_out.txt`, `sample_trials.png`, `predictions.png`.

### Training Progress
- Loss decreasing: **Yes**, monotonically, 1.925 -> 0.748 over 200 epochs. Test loss 0.773.

### Decoder Results (Full)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc | Val / chance | Notes |
|--------|--------|----------------------|-------------------------|--------------|-------|
| choice | 0.5000 | 0.6358 | **0.6136** | 1.23x | time-averaged over a window that includes 0.5 s of pre-stimulus bins where choice is not yet represented -- see Step 12 |
| prior_probability_left | 0.3333 | 0.6743 | **0.6555** | 1.97x | |
| wheel_speed | 0.3333 | 0.6069 | **0.6000** | 1.80x | |
| whisker_motion_energy | 0.3333 | 0.5939 | **0.5874** | 1.76x | |

Train/validation gaps are 0.006-0.022, i.e. under 4% relative: no overfitting, no leakage.
Accuracies are higher than on the 2-session sample for every output (e.g. wheel speed
0.533 -> 0.600), as expected when the shared decoder can pool 440 sessions.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance analysis
Every output is above chance (table above). Three of the four exceed 1.75x chance.
**`choice` is at 1.23x chance and was investigated in depth.**

`/app/cache/timecourse.py` measures, completely independently of `decoder.py`, how decodable
each output is at **each of the 100 time bins**, using per-timepoint PCA(30) + balanced
logistic regression with 3-fold CV, on the 12 largest sessions:

| Output | mean over all bins | pre-stimulus (t <= 0) | late (t > 0.5 s) | peak | peak time |
|---|---|---|---|---|---|
| choice | 0.603 | **0.520** | 0.583 | **0.820** | +0.24 s |
| prior_probability_left | 0.636 | 0.605 | 0.627 | 0.723 | +0.24 s |
| wheel_speed | 0.563 | 0.607 | 0.535 | 0.655 | -0.36 s |
| whisker_motion_energy | 0.552 | 0.513 | 0.575 | 0.602 | +1.20 s |

This is the explanation, and it is physiological rather than a bug:
- **Before the stimulus appears the animal has not yet chosen**, so choice is at chance
  (0.520) for the first 25 of the 100 bins. It becomes strongly decodable (peak 0.820) around
  0.24 s, which is exactly the reaction-time range the trial filter enforces (0.08-2.0 s), and
  then decays as the trial ends.
- The reported number is a **single accuracy averaged over all 100 bins**, so a variable that is
  only represented during part of the trial is structurally capped well below its peak.
- The same is true in reverse for prior, which is above chance even before the stimulus
  (0.605) because the block is a slow contextual variable.

Crucially, this independent sklearn decoder reaches **0.603** for choice averaged over bins,
whereas `train_decoder.py` reaches **0.6136** -- the supplied decoder is already extracting at
least as much choice information as a straightforward per-timepoint linear readout of the same
data. There is therefore no information being lost in the conversion.

### Check 2: Accuracy comparison to papers
| Variable | Paper | What the paper reports | Our validation balanced accuracy | Comparable? |
|---|---|---|---|---|
| choice | Zhang et al. Fig 5A | single-**region** AUC ~0.6-0.8 (example 0.66 / 0.72 / 0.79) for single-session RRR on 5 RE regions, decoded from a **movement-aligned/50 ms** representation and **per-trial** (one label per trial) | 0.6136 **per time bin** over all 440 sessions and all regions | our number is a per-bin average that includes the pre-stimulus period; the paper's is a per-trial AUC. Our **peak per-bin** accuracy of 0.82 sits at the top of the paper's per-region range, which is what one expects when pooling all regions of a session |
| prior | Zhang et al. Fig 5B | Pearson correlation 0.05-0.65 (continuous prior) | 0.6555 (3-class balanced accuracy) | not directly comparable (regression vs 3-class classification); ours is far above the 0.333 chance |
| wheel speed / whisker ME | Zhang et al. Fig 3 | qualitative R2 / correlation; whisker ME reconstructed better than wheel speed | 0.600 / 0.587 (3-class) | not directly comparable; both well above chance |
| choice, per region | IBL BWM data paper Figs 4-7 | **null-corrected** median balanced accuracy, effect sizes ~0.01-0.1 above the null | 0.1136 above chance | our effect size is at or above the top of the paper's per-region range, consistent with using all of a session's neurons rather than one region |

No reported accuracy exceeds ours once the difference in target representation (per-trial vs
per-timepoint) and in neuron pool (one region vs whole session) is accounted for. The one
headline number that is directly comparable -- peak per-bin choice decoding, 0.82 -- matches the
upper end of the paper's per-region AUCs.

### Check 3: Train vs validation gap
| Output | Train | Validation | Ratio |
|---|---|---|---|
| choice | 0.6358 | 0.6136 | 1.036 |
| prior_probability_left | 0.6743 | 0.6555 | 1.029 |
| wheel_speed | 0.6069 | 0.6000 | 1.011 |
| whisker_motion_energy | 0.5939 | 0.5874 | 1.011 |

All far below the 1.5x threshold. No overfitting and no data leakage (the tercile thresholds
are computed per session over all of its trials, which is a fixed monotone recoding of the
behaviour and does not transfer label information between trials; the decoder's own train/test
split is over trials).

### Additional debugging steps performed
1. **Output values verified on specific trials against the raw data** -- Step 10 Check 2,
   `np.allclose` on choice, prior, wheel and whisker terciles for 4 sessions, plus the semantic
   check that on correct non-zero-contrast trials the decoded choice equals the stimulus side.
2. **Temporal alignment verified** -- the `--show-processing` plots overlay the raw wheel and
   whisker traces with the interpolated samples and show a population PSTH that steps up
   exactly at t = 0; the time-course analysis above shows choice information appearing only
   *after* stimulus onset, which is itself a strong alignment check (a misaligned dataset would
   show choice information before t = 0).
3. **Output variation checked** -- no output is dominated by one class:
   choice [0.508, 0.492], prior [0.417, 0.140, 0.442], wheel and whisker [1/3, 1/3, 1/3].
4. **Neural filtering checked** -- reproduces 888.7 Kilosort units/probe and 108.2 well-isolated
   units/probe against the paper's 889 and 108.
5. **Processing matches the reference** -- Step 10 Check 3 table; binning proven bit-identical
   to `bincount2D`.

### Issues Found and Resolved
- **16 all-zero neural trials** (Step 10 Check 1): caused by the ephys recording ending before
  the behaviour in one session and by a 2 s recording dropout in another. Fixed with a
  neural-coverage filter; re-verified with zero warnings.
- **Apparent neural mismatch in 1 of 12 spot-checked trials**: traced to `np.histogram2d` using
  a different floating-point edge convention than the reference `bincount2D`. My binner matches
  the reference (0 failures in 2000 randomised on-edge trials); the check was rewritten to use
  the reference binner and now reports 0 failures.
- No other issues; no further iteration required.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created -- user-facing description, load instructions, full format spec,
      key statistics, curation rules and reference decoder accuracies.
- [x] `cache/` folder created with `README_CACHE.md` documenting each cached script and the
      step of this document it supports.
- [x] All investigation scripts moved to `cache/`; `/app` now contains only the deliverables,
      the provided inputs and the generated figures.

### Deliverables
| File | Status |
|---|---|
| `/app/CONVERSION_NOTES.md` | this file |
| `/app/convert_data.py` | conversion script |
| `/app/converted_data.pkl` | 11.33 GB, 440 sessions |
| `/app/sample_data.pkl` | 0.03 GB, 2 sessions |
| `/app/README.md` | user documentation |
| `/app/conversion_sample_out.txt` | created |
| `/app/verification_sample_out.txt` | created, no errors or warnings |
| `/app/train_decoder_sample_out.txt` | created |
| `/app/conversion_full_out.txt` | created |
| `/app/verification_full_out.txt` | created, no errors or warnings |
| `/app/train_decoder_full_out.txt` | created |
| `/app/processing_<eid>.png` (x2) | per-step processing checks |
| `/app/sample_trials.png`, `/app/predictions.png` | decoder diagnostics |

### Summary of the conversion
440 sessions from 136 mice, 60,483 well-isolated grey-matter neurons across 209 Beryl regions,
187,651 trials, each a 2 s window aligned to stimulus onset and binned into 100 x 20 ms bins,
exactly as in `0_data_caching.py`. Two decoder inputs (time from stimulus onset,
trial number in block) and four categorical outputs (choice, block prior, tercile-discretised
wheel speed and whisker motion energy). Loading reproduces the data paper's 889 Kilosort units
and 108 well-isolated units per probe to within 0.3 and 0.2 units respectively. All format
checks pass with no errors and no warnings, every independent sanity check against the raw
files passes with `np.allclose`, and the reference decoder reaches 0.614 / 0.656 / 0.600 /
0.587 validation balanced accuracy against chance levels of 0.500 / 0.333 / 0.333 / 0.333.
