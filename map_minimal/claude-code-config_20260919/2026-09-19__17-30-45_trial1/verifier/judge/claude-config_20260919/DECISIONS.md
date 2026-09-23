# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is the DANDI 000363 release, one NWB (HDF5) file per session under `/app/data/sub-<id>/`. The AI enumerates every session with a single sorted glob over that layout and processes each file exactly once. Files are opened with **`h5py` directly** rather than with `pynwb`, reading the NWB groups by path (`intervals/trials`, `units`, `acquisition/BehavioralEvents`, `acquisition/BehavioralTimeSeries`, `general/subject`, `general/extracellular_ephys`). Sessions are farmed out to a `multiprocessing.Pool` (default 16 workers, run with 24); each worker writes its converted session to a per-session pickle in `/app/work/sessions/`, and the parent then reads those back and concatenates them into the final `/app/converted_data.pkl`. All 174 files are opened; 142 survive curation.

ii.
```python
files = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
...
jobs = [(f, args.cache_dir) for f in files]
with Pool(args.workers) as pool:
    for i, s in enumerate(pool.imap(_worker, jobs)):
```

```python
with h5py.File(path, 'r') as nwb:
    subject = nwb['general/subject/subject_id'][()].decode()
    trials = nwb['intervals/trials']
    trial_start = trials['start_time'][:]
    ...
    go_times = nwb['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
```

```python
for i, s in enumerate(kept):
    with open(s['cache'], 'rb') as fh:
        sess = pickle.load(fh)
    data['neural'].append(sess['neural'])
```

iii. From the final report: "NWB files read directly with h5py." The trajectory shows the AI first dumped the HDF5 tree of a sample file and enumerated every group/attribute before deciding on paths, so it used raw h5py paths rather than `pynwb` because it only needed a handful of arrays per file and wanted the speed (the full conversion runs in ~29 s wall with 24 workers). The per-session cache exists so that workers do not have to ship ~9 GB of arrays back through the `Pool` IPC pipe.

## 1-b. How are the data split into subjects (mice)?

i. Each NWB file names its animal in `general/subject/subject_id` (a numeric string such as `'440956'`, matching the containing `sub-440956/` folder). That value is read per session and carried through assembly, where `subjects` is the sorted set of unique ids over the **kept** sessions and `subject_idx` indexes into it. All 28 animals in the dandiset survive curation.

ii.
```python
subject = nwb['general/subject/subject_id'][()].decode()
```

```python
subjects = sorted({s['subject'] for s in kept})
subject_index = {s: i for i, s in enumerate(subjects)}
...
'subjects': subjects,
'subject_idx': np.array([subject_index[s['subject']] for s in kept], dtype=np.int64),
```

iii. No explicit justification is given beyond it being the canonical animal id in the file; the trajectory shows the AI reading `general/subject` in its first structural scan and confirming that `subject_id` matches the `sub-*` directory name, so no separate grouping step is needed.

## 1-c. How are the data split into sessions?

i. One NWB file is one session, so no splitting is required; the session identifier is the file basename (e.g. `sub-440956_ses-20190209T150135_behavior+ecephys+ogen`), and session order follows the sorted file list. The AI then **rejects sessions on the data paper's behavioural selection criteria**, quoted from `methods.txt`: overall performance > 65% and at least 50 correct lick-left and 50 correct lick-right trials, both measured on control trials (no photostimulation, no early lick). Sessions with no `'good'` units or fewer than 2 kept trials are also dropped. 142 of 174 sessions are retained (vs. 173 for the human reference, which applies no behavioural session filter).

ii.
```python
MIN_PERFORMANCE = 0.65
MIN_CORRECT_PER_DIRECTION = 50
```

```python
# Data paper: overall performance > 65% and >= 50 correct lick-left and
# lick-right trials, measured on control (no photostimulation) trials
# excluding early-lick trials.
control = keep & ~stim_trial & ~early
hit = outcome == 'hit'
miss = outcome == 'miss'
responded = (hit | miss) & control
performance = hit[responded].mean() if responded.any() else 0.0
n_left = int(np.sum(hit & control & (instruction == 'left')))
n_right = int(np.sum(hit & control & (instruction == 'right')))
rejected = (performance <= MIN_PERFORMANCE
            or n_left < MIN_CORRECT_PER_DIRECTION
            or n_right < MIN_CORRECT_PER_DIRECTION)
```

```python
if good.sum() == 0 or len(trial_idx) < 2:
    summary['rejected'] = True
    summary['reject_reason'] = 'no good units or fewer than 2 trials'
```

iii. From the final report: "**Session curation** — the data paper's criterion (`methods.txt`): control-trial performance > 65% and ≥ 50 correct lick-left and lick-right trials. 29 of 174 sessions fail; a 30th has zero good units." The AI grounds this in the instruction that curation must match the reference papers, and `methods.txt` states the criteria verbatim ("We selected experimental sessions for analysis based on following criteria: overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each"). The 2-trial minimum comes from the target-format requirement.

Verified by re-running the criterion: 30 sessions fail the behavioural test, 1 (`sub-440958_ses-20190216T162508`) has no quality-controlled units, and 1 more (`sub-455219_ses-20190807T134913`, 505 valid trials / 463 good units) is lost to an unhandled `IndexError` that the worker's blanket `except` converts into a silent rejection — see 2-c and 9. The AI's own report therefore under-counts its rejections (29 + 1 stated, 32 actual).

## 1-d. How are the data split into trials?

i. Trials come straight from the NWB trials table `intervals/trials`, one row per behavioural trial. The AI asserts that `go_start_times` has exactly one event per trial row and that every go cue falls inside its trial's `[start_time, stop_time]`, so the row-to-event mapping is unambiguous.

ii.
```python
trials = nwb['intervals/trials']
trial_start = trials['start_time'][:]
trial_stop = trials['stop_time'][:]
...
go_times = nwb['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
assert len(go_times) == len(trial_start), 'go cue count != trial count'
assert np.all((go_times >= trial_start) & (go_times <= trial_stop))
```

iii. The trajectory shows the AI checked event-count-vs-trial-count across all 174 files and found `go_start_times` is 1:1 with trials, whereas `sample_start_times` is not (an early lick replays the sample epoch, producing extra onsets) — which is why the go cue is used as the per-trial anchor and the sample onsets need the "last one before the go cue" rule (3-a).

## 1-e. How are trials filtered based on quality controls?

i. Three filters, then a 2-trial minimum:
- **`auto_water` or `free_water` trials are dropped** (1,339 + 2,450 = 3,789 trials across the release), because water was delivered regardless of the animal's choice, so the outcome does not reflect a decision.
- **Trials outside the units' `obs_intervals` are dropped** — in 9 files the ephys covers only part of the behavioural session, and the uncovered trials would otherwise look like a brain-wide silence. Coverage is intersected over *all* good units.
- Sessions with fewer than 2 surviving trials are dropped.

Photostimulation, early-lick and `ignore` trials are deliberately **kept**, departing from the method paper, because they are required decoder inputs/output classes. Result: 74,104 trials over 142 sessions (reference: 90,860 over 173).

ii.
```python
# The method paper excludes "water administration regardless of the
# animals' choice (free water trials)" because the outcome on those
# trials does not reflect the animal's decision; the same holds for
# auto-water trials, so both are dropped.
#
# That paper also excludes photoinhibition, early-lick and ignore
# trials.  Those three cannot be dropped here: photostimulation is a
# required decoder input and early lick / no-lick are required decoder
# output classes, so all three are kept.
good = _str(nwb['units']['classification'][:]) == 'good'
keep = ~(auto_water | free_water)
if good.any():
    keep &= observed_trials(nwb, good, trial_start)
```

```python
def observed_trials(nwb, good, trial_start):
    units = nwb['units']
    obs = units['obs_intervals'][:]
    index = units['obs_intervals_index'][:]
    starts = np.concatenate([[0], index[:-1]])
    covered = np.ones(len(trial_start), dtype=bool)
    for u in np.flatnonzero(good):
        covered &= np.isin(trial_start, obs[starts[u]:index[u], 0])
    return covered
```

iii. From the final report: "**Trial curation** — auto-water/free-water trials dropped (the method paper excludes water given regardless of choice), as are trials outside the units' `obs_intervals` — in 8 files the ephys covers only the first part of the behavioural session, and those trials would otherwise look like a brain-wide silence. Photostim, early-lick and ignore trials are **kept**, deviating from the method paper because the task requires them as an input and as output classes." The trajectory shows the AI scanned all files and established that `obs_intervals` per unit is exactly the trial windows of the observed trials, and that the observed set is identical for every unit within a session (so the intersection over units costs nothing extra).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (a ragged array indexed by `units/spike_times_index`), restricted to units with `units/classification == 'good'` and further restricted by `units/is_good_trials` (see 2-c). The other input is `acquisition/BehavioralEvents/go_start_times/timestamps`, which places the bin edges. Both are on the same session-absolute clock.

ii.
```python
units = nwb['units']
spike_index = units['spike_times_index'][:]
starts = np.concatenate([[0], spike_index[:-1]])
unit_ids = np.flatnonzero(good)
...
spike_times = units['spike_times']
for i, u in enumerate(unit_ids):
    st = spike_times[starts[u]:spike_index[u]]
```

iii. `spike_times` is the only neural representation in the release; the AI's docstring states the binning "mirrors `preprocessing_DJ_2022Aug.process_one_area`", the reference pipeline's go-cue-aligned spike binning.

## 2-b. How is the `neural` data processed?

i. Per-bin firing rate in spikes/s. For each good unit, the absolute bin edges for all trials are built as one flat array, `np.searchsorted` gives the running spike count at every edge, adjacent differences give the per-bin spike count, and the count is divided by the 50 ms bin width. Rates are stored `float32`. No smoothing, normalisation, baseline subtraction, or trial-length adaptation is applied. (I verified numerically against an independent `np.histogram` recomputation: exact match.)

ii.
```python
n_trials = len(go_times)
# (n_trials, N_BINS+1) absolute bin edges, flattened for one searchsorted per unit
edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()

rates = np.empty((len(unit_ids), n_trials, N_BINS), dtype=np.float32)
spike_times = units['spike_times']
for i, u in enumerate(unit_ids):
    st = spike_times[starts[u]:spike_index[u]]
    idx = np.searchsorted(st, edges).reshape(n_trials, N_BINS + 1)
    rates[i] = np.diff(idx, axis=1) / BIN_WIDTH
```

iii. From the docstring: "Spikes are counted in the fixed window around each trial's go cue, exactly as the reference pipeline does (`preprocessing_DJ_2022Aug.process_one_area` bins the go-cue-aligned spike train over a fixed window and divides by the bin width, irrespective of how long the trial itself lasted)." The AI also explicitly documents that on `miss` trials the trial is terminated early by the timeout so the tail bins contain no recorded spikes and are reported as 0 Hz, "which is what the reference pipeline produces as well; no imputation is invented here" (recorded in `metadata['known_limitation']`).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters:
- `units/classification == 'good'` — the verdict of the region-specific spike-sorting QC classifiers of `ChenLiuEtAl2023_SpikeSortingQC.pdf`. This is the same criterion the human reference uses. `units/unit_quality` is not used.
- **Additionally**, `units/is_good_trials` must be `True` for that unit on *every* analysed trial; units failing on any analysed trial are dropped entirely (0.8% of good units across the release; 57,097 → 56,532 in the kept sessions).

A session with zero good units is dropped.

ii.
```python
good = _str(nwb['units']['classification'][:]) == 'good'
...
good[good] = units_good_on_trials(nwb, good, trial_idx)
summary['n_units'] = int(good.sum())
if good.sum() == 0:
    summary['rejected'] = True
    summary['reject_reason'] = 'no unit passes quality control on all trials'
    return summary
```

```python
def units_good_on_trials(nwb, good, trial_idx):
    """Mask over the good units that are quality-flagged on every analysed trial.
    ... A unit has to occupy one row of a fixed
    neuron-by-time matrix on every trial, so units that fail on any analysed
    trial are dropped instead (0.8% of units across the release).
    """
    flags = nwb['units']['is_good_trials'][:][good][:, trial_idx]
    return flags.all(axis=1)
```

iii. From the final report: "**Neuron curation** — `units/classification == 'good'`: the region-specific QC classifiers from the spike-sorting white paper, i.e. the units the data paper analyses. These are exactly the units carrying a CCF annotation. Further restricted to units flagged good on every analysed trial (`is_good_trials`), which costs 0.8% of units but is required for a fixed neuron × trial matrix." The trajectory shows the AI quantified this first (0.81% of good units are not good on all trials; worst session 64%) before adding the filter.

**Defect found:** `is_good_trials` is stored with one **column per observed trial** (its width equals `len(obs_intervals)`, not the number of rows in the trials table), but the code indexes its columns with `trial_idx`, which are indices into the **full** trials table. This coincides only when the observed trials are a leading prefix. In `sub-455219_ses-20190807T134913` the observed trials are rows 125–629 of 630, so `flags[:, trial_idx]` raises `IndexError: index 505 is out of bounds for axis 1 with size 505` (reproduced directly). `_worker` catches it and marks the session rejected, so 505 valid trials and 463 good units are silently discarded.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times, event timestamps and camera timestamps all live on the same session-absolute clock, so no resampling or offset correction is needed. Each trial's absolute bin edges are its go-cue time plus the fixed relative edge grid, and spikes are binned directly against those absolute edges. The alignment event is `go_start_times`, recorded in `metadata['temporal_alignment_event']` as "auditory go cue onset (end of the delay epoch)".

ii.
```python
go_times = nwb['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
...
go = go_times[trial_idx]
...
rates = bin_spikes(nwb, good, go)                 # (units, trials, bins)
```

```python
edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()
```

iii. The AI verified in the trajectory that every go cue lies inside its trial window and that there is exactly one per trial, and asserts both in the code. No further justification is offered because the single global clock makes the alignment a direct lookup.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 of them, spanning −2.5 s to +1.5 s relative to the go cue, identical for every trial and session. The grid is defined once at module level as 81 relative edges and reused everywhere; `metadata['time_bin_size'] = 50.0` (ms), `off_start = -2.5`, `off_end = 1.5`. No rebinning or resampling of an intermediate representation happens — spikes go straight into the final grid.

ii.
```python
# Trial window and binning, fixed by the decoder task.  The reference pipeline
# (MapVideoAnalysis/Sherlock/preprocess_all_ephys.py) uses a -3 .. +3 s window
# around the go cue with 40 ms bins slid by 3.4 ms; here the assignment asks for
# -2.5 .. +1.5 s and 50 ms bins, so bins are laid down back-to-back (stride =
# width) and tile the window exactly.
T_START = -2.5
T_STOP = 1.5
BIN_WIDTH = 0.05
N_BINS = int(round((T_STOP - T_START) / BIN_WIDTH))       # 80
BIN_EDGES = T_START + BIN_WIDTH * np.arange(N_BINS + 1)   # relative to go cue
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2
```

iii. The window and bin width are dictated by the instructions. The AI explicitly documents the deviation from the reference pipeline's −3…+3 s / 40 ms sliding window and why: the assignment fixes the window and width, and back-to-back bins are used so the bins tile the window exactly (the reference pipeline's 3.4 ms stride would produce overlapping bins).

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `acquisition/BehavioralEvents/sample_start_times/timestamps` (the sample-epoch / instruction-tone onsets) together with each trial's go-cue time and `start_time`. The tone taken for a trial is the **last sample onset at or before the go cue and at or after the trial start**.

ii.
```python
def tone_onsets(nwb, trial_start, trial_stop, go_times):
    """Absolute time of the instruction tone that preceded each go cue.

    The instruction is a train of pure tones played over the sample epoch, so the
    sample-epoch onset is the tone onset.  Licking during the sample or delay
    epoch replays the epoch (data paper), which is why there are more
    ``sample_start_times`` than trials; the tone the animal actually answered is
    the last one before the go cue, so that is the one used here.
    """
    sample = nwb['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
    onsets = np.full(len(go_times), np.nan)
    lo = np.searchsorted(sample, trial_start, side='left')
    hi = np.searchsorted(sample, go_times, side='right')
    for i in range(len(go_times)):
        if hi[i] > lo[i]:
            onsets[i] = sample[hi[i] - 1]
    return onsets
```

iii. As quoted in the docstring and the final report: early licks replay the sample epoch, so a trial can carry several sample onsets; the last one before the go cue is the tone the animal actually answered. The AI confirmed in the trajectory that `sample_start_times` outnumbers trials while `go_start_times` does not.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Each bin's value is its absolute centre time minus the trial's tone time, i.e. bin centre (relative to go) plus the go-minus-tone gap. Stored as `float32`, one continuous time-varying row of the `(2, 80)` input array. A fallback substitutes the nominal 0.65 s sample + 1.2 s delay = 1.85 s structure when no sample onset is found inside the trial; across all kept sessions this never fires (`n_trials_missing_tone == 0` for all 142 sessions). The resulting value range, [−1.525, 11.894] s, is identical to the reference's.

ii.
```python
tone = tone_onsets(nwb, trial_start, trial_stop, go_times)[trial_idx]
missing_tone = ~np.isfinite(tone)
if missing_tone.any():
    # No sample-epoch onset recorded between trial start and go cue:
    # fall back to the nominal 0.65 s sample + 1.2 s delay structure.
    tone[missing_tone] = go[missing_tone] - 1.85
summary['n_trials_missing_tone'] = int(missing_tone.sum())
time_from_tone = (go[:, None] + BIN_CENTERS[None, :]
                  - tone[:, None]).astype(np.float32)
```

iii. No processing beyond locating the tone is needed; the value is a plain time difference. The AI kept it continuous rather than converting to a binary onset marker, since the instructions list it as "continuous, time-varying".

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed on the same go-cue-relative grid that defines the neural bins: `BIN_CENTERS` are the centres of the very bins the firing rates are computed in, so input bin *k* and neural bin *k* cover the same interval by construction.

ii.
```python
BIN_EDGES = T_START + BIN_WIDTH * np.arange(N_BINS + 1)   # relative to go cue
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2
```
```python
time_from_tone = (go[:, None] + BIN_CENTERS[None, :] - tone[:, None]).astype(np.float32)
```
```python
inputs.append(np.stack([time_from_tone[j], stim[j]]).astype(np.float32))
```

iii. N/A — a single shared bin grid makes alignment automatic; no interpolation or offset correction.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The trials-table columns `photostim_onset` and `photostim_duration` (both stored as strings, with `'N/A'` on control trials), plus `start_time` (the onset is expressed relative to trial start) and the go-cue time (to re-express it on the bin axis). `photostim_onset != 'N/A'` is also used as the control-trial mask in the session-performance criterion.

ii.
```python
onset = np.array([_parse_float(v) for v in _str(trials['photostim_onset'][:])])
duration = np.array([_parse_float(v) for v in _str(trials['photostim_duration'][:])])
```
```python
def _parse_float(value):
    """Parse the string-valued trial columns; 'N/A' (and anything else that is
    not a number) becomes NaN."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan
```

iii. Docstring: "`photostim_onset` is given relative to the start of the trial and `photostim_duration` in seconds; both are 'N/A' on control trials." The trajectory shows the AI cross-checked `start_time + float(photostim_onset)` against the independent `BehavioralEvents/photostim_start_times` timestamps before settling on the trials-table columns.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary `(n_trials, 80)` `float32` time series. A bin is marked 1 when it **overlaps** the stimulation interval `[onset, onset + duration]` — i.e. `bin_end > on and bin_start < off` — and 0 otherwise. Control trials have NaN bounds and are skipped by the `np.isfinite` mask, so all their bins stay 0. 15,136 of the 74,104 kept trials carry stimulation.

ii.
```python
stim = np.zeros((len(go_times), N_BINS), dtype=np.float32)
on = trial_start + onset - go_times          # relative to the go cue
off = on + duration
valid = np.isfinite(on) & np.isfinite(off)
for i in np.flatnonzero(valid):
    stim[i] = ((BIN_EDGES[1:] > on[i]) & (BIN_EDGES[:-1] < off[i])).astype(np.float32)
```

iii. Docstring: "A bin is marked as stimulated when it overlaps the stimulation interval." The instructions ask for "whether photostimulation is on at every time point", and the AI chose the inclusive (any-overlap) reading so that a bin that is partly illuminated is not reported as laser-off.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The stored onset is trial-relative, so it is converted to the go-cue-relative axis by adding `trial_start` and subtracting the go cue, after which it is compared directly against the same `BIN_EDGES` grid the firing rates use.

ii.
```python
on = trial_start + onset - go_times          # relative to the go cue
off = on + duration
...
stim[i] = ((BIN_EDGES[1:] > on[i]) & (BIN_EDGES[:-1] < off[i])).astype(np.float32)
```

iii. N/A — the conversion is required only because the column is stored relative to trial start rather than to the alignment event.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no lick-direction column, so it is derived from `trial_instruction` (`'left'`/`'right'`, the instructed spout) crossed with `outcome` (`'hit'`/`'miss'`/`'ignore'`): a hit means the animal licked the instructed side, a miss the opposite side, an ignore means no lick.

ii.
```python
# Lick direction: 'hit' means the animal licked the instructed spout,
# 'miss' the opposite one and 'ignore' that it did not lick.  Verified
# against the recorded left/right lick times (perfect agreement).
other = np.where(instruction == 'left', 'right', 'left')
licked = np.where(outcome == 'hit', instruction,
                  np.where(outcome == 'miss', other, 'no lick'))
```

iii. From the final report: "lick direction derived from `trial_instruction` × `outcome`, checked against the recorded lick times (perfect agreement)." The trajectory shows the AI validated the derivation empirically against `BehavioralEvents` lick-time streams rather than assuming the mapping.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Mapped to `0 = left`, `1 = right`, `2 = no lick` via the index into `CHOICE_VALUES`, stored `int8`, and broadcast across all 80 bins so that all four outputs share one `(4, 80)` per-trial array. `output_values[0] = ['left', 'right', 'no lick']`. Resulting distribution: 44.8% left / 44.3% right / 10.9% no lick.

ii.
```python
CHOICE_VALUES = ['left', 'right', 'no lick']
...
choice = np.array([CHOICE_VALUES.index(v) for v in licked], dtype=np.int8)
...
choice = choice[trial_idx]
...
ones = np.ones(N_BINS, dtype=np.int8)
for j in range(len(trial_idx)):
    outputs.append(np.stack([choice[j] * ones, outcome_code[j] * ones,
                             early_code[j] * ones, tongue[j]]).astype(np.int8))
```

iii. Left = 0 / right = 1 follows the instruction's ordering, with a third class for the no-lick case. Choice is one value per trial, so it is repeated across bins to satisfy the format's preference for time-varying `(n_output, n_timepoints)` arrays.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which already stores exactly the three strings `'ignore'`, `'miss'`, `'hit'` the instructions ask for.

ii.
```python
outcome = _str(trials['outcome'][:])
```

iii. No derivation needed — the trajectory's first structural scan printed the unique values of every object-dtype trials column and confirmed `outcome` takes only these three values.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped with a fixed dictionary to `0 = ignore`, `1 = miss`, `2 = hit`, stored `int8`, repeated across all 80 bins. Resulting distribution over kept trials: 10.9% ignore / 15.3% miss / 73.8% hit (the reference's is 14.9 / 16.6 / 68.4 — the AI's higher hit rate follows from dropping the low-performance sessions).

ii.
```python
OUTCOME_CODE = {'ignore': 0, 'miss': 1, 'hit': 2}
OUTCOME_VALUES = ['ignore', 'miss', 'hit']
...
outcome_code = np.array([OUTCOME_CODE[v] for v in outcome], dtype=np.int8)
```

iii. The code assignment follows the instruction's ordering (ignore, miss, hit); as with choice it is one value per trial, repeated across bins.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. The `early_lick` column of the trials table, which holds `'early'` / `'no early'`.

ii.
```python
early = _str(trials['early_lick'][:]) == 'early'
```

iii. The flag is stored explicitly per trial, so no derivation is needed. (The lick that sets it happens during the sample or delay epoch, i.e. inside the −2.5 s part of the window, so the event is in principle decodable from the neural data.)

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The boolean is cast to `int8` (`0 = no`, `1 = yes`) and repeated across all 80 bins. `output_values[2] = ['no', 'yes']`. Distribution: 88.5% no / 11.5% yes (reference: 88.5 / 11.5).

ii.
```python
EARLY_LICK_VALUES = ['no', 'yes']
...
early_code = early.astype(np.int8)
...
early_code = early_code[trial_idx]
```

iii. Follows the instruction's ordering; one value per trial, repeated across bins like the other per-trial outputs.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is `(n_frames, 3)` and whose `timestamps` run at ~294 Hz on the session-absolute clock. Column 1 is `tongue_y` (the value) and column 2 is `tongue_likelihood` (the DeepLabCut confidence that decides visibility). The AI read the series' `description` attribute in the trajectory (`"Time series for TongueTracking position: ('tongue_x', 'tongue_y', 'tongue_likelihood')"`) to establish the column layout rather than assuming it.

ii.
```python
track = nwb['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
ts = track['timestamps'][:]
data = track['data'][:]
...
chunk = data[lo:hi, :]
y = chunk[:, 1]
visible = chunk[:, 2] >= TONGUE_P_CUTOFF
```

iii. This is the only tongue measurement in the release; the AI's session scan confirmed `Camera0_side_TongueTracking` is present in all 174 files.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with DeepLabCut likelihood below 0.5 are treated as "tongue not visible" and excluded. Surviving frames are assigned to the trial's 50 ms bins and averaged within each bin; a bin with no visible frame is left NaN. Mean visible fraction per session is ~26% of bins.

ii.
```python
# DeepLabCut confidence above which the tongue counts as visible.  The tongue
# likelihood in this dataset is essentially binary (median ~5e-5 when the tongue
# is in the mouth, >0.999 when it is out), so any cutoff in (1e-3, 0.99) gives
# the same answer; 0.5 is the DeepLabCut default.
TONGUE_P_CUTOFF = 0.5
```
```python
which = np.searchsorted(BIN_EDGES, ts[lo:hi] - go, side='right') - 1
ok = visible & (which >= 0) & (which < N_BINS)
if not ok.any():
    continue
counts = np.bincount(which[ok], minlength=N_BINS)
sums = np.bincount(which[ok], weights=y[ok], minlength=N_BINS)
seen = counts > 0
y_binned[i, seen] = sums[seen] / counts[seen]
```

iii. From the docstring and final report: the tracker still reports a y-position when the tongue is retracted, so low-likelihood frames must be discarded rather than averaged in; the likelihood distribution is effectively bimodal, so the exact cutoff does not matter and 0.5 (the DeepLabCut default) is used. The AI verified the bimodality in the trajectory before choosing the cutoff.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per session, the 40th and 60th percentiles are taken of the **binned** y-values (i.e. of the same quantity being discretised), pooled over all bins of all analysed trials in which the tongue was visible. Bins are then classed `0` (< p40), `1` (p40 ≤ y ≤ p60), `2` (> p60), and `3` (`'not visible'`) when the bin has no confident detection — including bins with no video frames at all. By construction the three visible classes hold exactly 40/20/40% of the visible bins. Overall: 10.6 / 5.3 / 10.6 / 73.6% (reference: 9.7 / 5.1 / 10.3 / 75.0%).

ii.
```python
TONGUE_PCTILES = (40.0, 60.0)
TONGUE_VALUES = ['y < 40th pctile', '40th-60th pctile', 'y > 60th pctile',
                 'not visible']
```
```python
seen = np.isfinite(y_binned)
classes = np.full((n_trials, N_BINS), 3, dtype=np.int8)
if seen.any():
    p40, p60 = np.percentile(y_binned[seen], TONGUE_PCTILES)
    vals = y_binned[seen]
    classes[seen] = np.where(vals < p40, 0, np.where(vals <= p60, 1, 2)).astype(np.int8)
else:
    p40 = p60 = np.nan
```

iii. From the docstring: "Percentiles are taken over the binned y-values of the whole session (all bins of all analysed trials in which the tongue was visible), i.e. over the same quantity that is being discretised, so that the three visible classes hold 40%/20%/40% of the visible bins as intended." The 40/60 split and the per-session scope are dictated by the instructions; the fourth class is the AI's addition for the (majority) case where the tongue is not out. The per-session `p40`/`p60` are recorded in `metadata['session_info']` for auditing.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps share the session-absolute clock with the spikes and events, so each trial's frame range is located by `np.searchsorted` on the camera timestamps at `go + T_START` and `go + T_STOP`, and each frame is assigned to a bin by `searchsorted` against the same `BIN_EDGES` grid used for the firing rates. No interpolation or offset correction. Frames outside `[0, N_BINS)` are dropped by the explicit range mask. Because the side video is trial-gated, trials whose go cue is less than 2.5 s after trial start have leading bins with no frames, which fall into class 3.

ii.
```python
for i, go in enumerate(go_times):
    lo, hi = np.searchsorted(ts, [go + T_START, go + T_STOP])
    if hi <= lo:
        continue
    ...
    which = np.searchsorted(BIN_EDGES, ts[lo:hi] - go, side='right') - 1
    ok = visible & (which >= 0) & (which < N_BINS)
```

iii. Docstring: "Bins with no video frames at all (the side-view video, like the spike train, only covers the trial itself) are likewise 'not visible'." The AI measured in the trajectory that 3.1% of trials start their window before trial start and 15.6% end after trial stop, and chose to record the consequence in `metadata['known_limitation']` rather than impute.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Five cases, handled explicitly:
- **String `'N/A'` in `photostim_onset` / `photostim_duration`**: `_parse_float` turns any unparseable value into NaN, and `np.isfinite` keeps those trials at zero stimulation.
- **Session never quality-controlled** (`classification` is NaN for all units, `sub-440958_ses-20190216T162508`): `_str` renders it as `'nan'`, no unit matches `'good'`, and the session is rejected.
- **Trials with no spike data**: excluded via `obs_intervals` (and `free_water`/`auto_water`) — see 1-e.
- **Frames with no confident tongue detection, and bins with no video**: assigned the explicit `'not visible'` class rather than imputed.
- **Trial with no sample onset between trial start and go cue**: falls back to the nominal 1.85 s sample+delay structure (this branch never fires in practice; `n_trials_missing_tone == 0` in every kept session).

A good unit without a CCF annotation raises `ValueError` — this is asserted rather than silently defaulted. Separately, `_worker` wraps the whole per-session conversion in a bare `except Exception`, prints the traceback, and returns a rejected stub.

ii.
```python
def _parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan
```
```python
missing_tone = ~np.isfinite(tone)
if missing_tone.any():
    tone[missing_tone] = go[missing_tone] - 1.85
```
```python
group = coarse_region(a, t)
if group is None:
    raise ValueError('good unit without a CCF annotation')
```
```python
def _worker(args):
    path, out_dir = args
    try:
        return process_session(path, out_dir)
    except Exception:
        traceback.print_exc()
        return dict(session=os.path.basename(path), rejected=True,
                    error=traceback.format_exc())
```

iii. The AI's stance, stated across the docstrings, is that data that was never recorded should cause exclusion rather than fabrication ("no imputation is invented here"), while a measurement that legitimately has no value (retracted tongue) gets its own explicit category. The `_worker` try/except is not justified anywhere in the trajectory; it appears to be defensive scaffolding for the parallel run. In the final run the AI piped the output through `tail -8`, so the one traceback it did produce scrolled past unseen, and the loss was never noticed (see 2-c).

## 10-a. What are the most time-consuming steps of the code?

i. The dominant cost is per-session I/O plus the per-unit binary search: pulling each unit's slice of `units/spike_times` out of HDF5 (up to ~11.5 M doubles per session), reading the full tongue-tracking array (~680 k × 3), the per-unit `np.searchsorted` over 81 × n_trials edges, and finally serialising the ~9 GB result. The AI does not profile in code, but it parallelises the whole per-session stage across a `multiprocessing.Pool` (16 by default, run with 24) and the complete conversion took **29 s wall** end-to-end, versus 247 s for the single-process reference. Against that, the per-session cache means the 9 GB payload is written once as 142 small pickles, read back, and written again as one file.

ii.
```python
with Pool(args.workers) as pool:
    for i, s in enumerate(pool.imap(_worker, jobs)):
```
```python
out_path = os.path.join(out_dir, name + '.pkl')
with open(out_path, 'wb') as fh:
    pickle.dump({'neural': neural, 'input': inputs, 'output': outputs, ...}, fh, protocol=4)
```

iii. No explicit justification is given for the parallelisation beyond the 15-minute runtime budget in the instructions; the trajectory shows the AI timed a single session first, extrapolated, and then added the `Pool`. The cache-to-disk design keeps the worker→parent IPC from having to pickle gigabytes through a pipe.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four Python-level loops remain:
- **`bin_spikes`, per unit** — one `searchsorted` per unit over *all* trials at once (the trial dimension is already vectorised by flattening the edge array). This cannot be collapsed further because the spike trains are ragged.
- **`tongue_classes`, per trial** — could be done with one global bin index and a single pair of `bincount` calls over the whole session, as the trial windows are disjoint.
- **`photostim_series`, per stimulated trial** — fully vectorisable in two broadcast comparisons (the human reference does exactly this in one expression); it runs over ~15 k trials.
- **`tone_onsets`, per trial**, and the **final per-trial assembly loop** (`np.stack` per trial) — the former is a pure `np.where` away from being vectorised; the latter is inherent to the required list-of-arrays output format.

ii.
```python
for i, u in enumerate(unit_ids):
    st = spike_times[starts[u]:spike_index[u]]
    idx = np.searchsorted(st, edges).reshape(n_trials, N_BINS + 1)
```
```python
for i in np.flatnonzero(valid):
    stim[i] = ((BIN_EDGES[1:] > on[i]) & (BIN_EDGES[:-1] < off[i])).astype(np.float32)
```
```python
for i in range(len(go_times)):
    if hi[i] > lo[i]:
        onsets[i] = sample[hi[i] - 1]
```

iii. None of these are commented on in the code or the trajectory. In practice the per-session work is cheap enough and parallelised widely enough that none of them is a measurable share of the 29 s runtime, so the AI evidently did not consider them worth optimising.

## 10-c. What processing does the code repeat multiple times?

i. Three kinds of repetition:
- **The 9 GB payload is serialised twice and deserialised once**: each worker pickles its session to `/app/work/sessions/<name>.pkl`, and the parent loads all 142 of them back to `pickle.dump` them into the single output file.
- **Per-trial quantities are computed for the whole trials table and then subset**: `tone_onsets(...)[trial_idx]`, `photostim_series(...)[trial_idx]`, `choice[trial_idx]`, `outcome_code[trial_idx]`, `early_code[trial_idx]` — the rejected trials' values are computed and thrown away. (The neural and tongue paths correctly take the already-filtered `go`.)
- **`photostim_onset` is decoded twice**: once in `process_session` to build `stim_trial`, once again inside `photostim_series`. `obs_intervals` and `is_good_trials` are each read in full (`[:]`) into memory even though only a few rows/columns are needed.

ii.
```python
out_path = os.path.join(out_dir, name + '.pkl')
with open(out_path, 'wb') as fh:
    pickle.dump({'neural': neural, ...}, fh, protocol=4)
...
for i, s in enumerate(kept):
    with open(s['cache'], 'rb') as fh:
        sess = pickle.load(fh)
```
```python
stim_trial = _str(trials['photostim_onset'][:]) != 'N/A'
...
onset = np.array([_parse_float(v) for v in _str(trials['photostim_onset'][:])])
```
```python
tone = tone_onsets(nwb, trial_start, trial_stop, go_times)[trial_idx]
stim = photostim_series(trials, trial_start, go_times)[trial_idx]
```

iii. The cache round-trip is a deliberate design choice (it bounds worker memory, avoids shipping gigabytes through the `Pool` IPC pipe, and makes the run restartable — the AI re-ran the conversion several times during development and `rm -rf work/sessions` appears explicitly when it wanted a clean rebuild). The compute-then-subset pattern is not commented on; the discarded work is small because the filters remove only a few percent of trials.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Modest amounts:
- **Diagnostic statistics** computed per session and never used by the decoder: `mean_rate`, `choice_counts`, `outcome_counts`, `early_counts`, `tongue_counts`, `n_photostim_trials`, `tongue_p40/p60`, `frac_tongue_visible`. These go into `metadata['session_info']` and a separate `conversion_summary.json`.
- **The 142 intermediate session pickles** in `/app/work/sessions/`, ~9 GB of writes that exist only to be read back once.
- **Dead parameters and branches**: `trial_stop` is passed to `tone_onsets` and never used; the missing-tone fallback never fires; the NaN-coordinate hemisphere fallback in `unit_regions` is documented as never occurring in this release.
- **Values computed for rejected trials** then discarded (see 10-c).
- The **hemisphere/probe-target machinery** in `unit_regions` is only needed for the `brain_regions` field, which the decoder itself does not consume — but the target format requires the field, so it is not truly discarded.

ii.
```python
summary.update(
    n_neurons=len(regions),
    n_trials=len(trial_idx),
    regions=regions,
    n_photostim_trials=int(np.sum(stim.any(axis=1))),
    choice_counts=np.bincount(choice, minlength=3).tolist(),
    outcome_counts=np.bincount(outcome_code, minlength=3).tolist(),
    early_counts=np.bincount(early_code, minlength=2).tolist(),
    tongue_counts=np.bincount(tongue.ravel(), minlength=4).tolist(),
    mean_rate=float(rates.mean()),
)
```
```python
def tone_onsets(nwb, trial_start, trial_stop, go_times):   # trial_stop unused
```
```python
side = np.where(np.isnan(ml_unit),
                np.array([t.split(' ')[0] if t else 'left' for t in target]),
                np.where(ml_unit >= ML_MIDLINE, 'left', 'right'))
```

iii. The diagnostics are cheap (`np.bincount` over 80-element rows) and the AI used them throughout the trajectory to sanity-check class balance, per-session tongue percentiles, and the brain-region mapping against the data paper's published unit counts; they are also shipped in `metadata` as an audit trail, which the target format invites ("Add other relevant fields, e.g. `session_info`"). The dead parameters and never-taken branches are described in the code as being there "to keep the code total" — defensive completeness rather than oversight.
