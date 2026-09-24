# Decisions

*Note on provenance: the instructions actually shown to the agent (trajectory step 1) differ slightly from
`/tests/instruction_reference.md`. The agent's copy specified choice as `left = 0, right = 1` (no "no lick"
class) and tongue y-position with only three classes (no "not visible" class). This is noted where it matters.*

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is one NWB file per session under `/app/data/sub-<numeric_id>/`. The AI walks the directory
tree: it lists every `sub-*` directory, then every `*.nwb` file inside it, and calls `process_session()` on
each path. Each file is opened once with `pynwb.NWBHDF5IO(path, 'r')` and closed explicitly with `io.close()`.
Within a file it reads the trials table (`nwb.trials`), the units table (`nwb.units`), the behavioural event
series (`nwb.acquisition['BehavioralEvents']`), the video tracking series
(`nwb.acquisition['BehavioralTimeSeries']`) and the subject metadata (`nwb.subject`). All 174 files are
visited; 105 survive the AI's filters.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
for subj in subjects:
    subj_dir = os.path.join(data_dir, subj)
    files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
    for fname in files:
        fpath = os.path.join(subj_dir, fname)
        result = process_session(fpath, sample_mode=sample_mode)
        if result is not None:
            all_sessions.append(result)
```
```python
io = pynwb.NWBHDF5IO(nwb_path, 'r')
nwb = io.read()
trials = nwb.trials
units = nwb.units
...
be = nwb.acquisition['BehavioralEvents']
bts = nwb.acquisition['BehavioralTimeSeries']
```

iii. The AI first inventoried the dataset (`ls /app/data`, counting files per subject) and confirmed "174
sessions vs 173 in the paper", then used `pynwb` — the standard reader for this published format — with a
plain directory walk, since the dandiset stores exactly one session per file. It explored the NWB internals
interactively (steps 23–38) before committing to the field names it reads.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from `nwb.subject.description`, which holds the mouse name used in the papers
(e.g. `'SC015'`) rather than the numeric DANDI id (`'440956'`) in the folder name. The unique names are
collected in first-appearance order with `OrderedDict.fromkeys`, and `subject_idx` indexes each session into
that list. 25 subjects appear in the final output (the other 3 lost all of their sessions to the session
filter).

ii.
```python
subject_id = nwb.subject.description  # e.g. 'SC015' - mouse name from description
```
```python
unique_subjects = list(OrderedDict.fromkeys(s['subject_id'] for s in all_sessions))
subject_idx = np.array([unique_subjects.index(s['subject_id']) for s in all_sessions], dtype=np.int64)
```

iii. The AI's stated reason (code comment) is that `description` carries the mouse name used in the papers,
which makes the output directly comparable to the figures/tables of the data paper. It did not explicitly
verify that the mapping name↔numeric-id is 1:1 (it is: 28 distinct descriptions for 28 subject ids).

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no grouping or splitting is done. Sessions are identified by `nwb.session_id`
(falling back to the filename). Session order is subject-directory order, then filename order. On top of this
the AI applies a **session-level behavioural selection filter** taken from `methods.txt`: a session is kept
only if performance on control (no-photostim), non-early-lick trials is `> 65%` **and** there are at least 50
correct lick-left and 50 correct lick-right trials. Sessions with zero QC-'good' units are also dropped.
69 of 174 sessions are rejected, leaving 105.

ii.
```python
MIN_PERF = 0.65   # >65% correct
MIN_CORRECT_PER_SIDE = 50  # at least 50 correct per side
...
for i in range(n_trials):
    is_control = trials['photostim_onset'][i] == 'N/A'
    is_no_early = trials['early_lick'][i] == 'no early'
    is_hit = trials['outcome'][i] == 'hit'
    instruction = trials['trial_instruction'][i]
    if is_control and is_no_early:
        control_no_early += 1
        if is_hit:
            correct_control += 1
            if instruction == 'left':
                correct_left += 1
            else:
                correct_right += 1
...
perf = correct_control / control_no_early
if perf <= MIN_PERF or correct_left < MIN_CORRECT_PER_SIDE or correct_right < MIN_CORRECT_PER_SIDE:
    print(f"  Skipping {os.path.basename(nwb_path)}: perf={perf:.1%}, L={correct_left}, R={correct_right}")
    io.close()
    return None
```

iii. From `methods.txt`: "We selected experimental sessions for analysis based on following criteria: overall
behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each." The AI quoted
this and checked borderline sessions (66.1%, 65.6%, 65.8%) deliberately, choosing the strict `>` comparison
(step 55). It noticed that this produced "106 passing sessions out of 174" and separately noticed that the
paper reports 173 sessions, but never reconciled the two numbers.

## 1-d. Are the data correctly split into trials?

i. Trials are rows of `nwb.trials`, with the go cue for trial *i* taken as `go_start_times[i]`. Because in
some files the ephys covers only part of the behavioural session, the AI maps `units/obs_intervals` rows onto
trial rows: identity mapping when the counts are equal, otherwise nearest `start_time` within a 0.5 s
tolerance (unmatched observations get `-1`). Only trials that appear in that mapping are processed.

ii.
```python
def build_obs_to_trial_map(nwb, good_indices):
    obs = units['obs_intervals'][good_indices[0]]
    n_obs = obs.shape[0]
    if n_obs == n_trials:
        return list(range(n_trials))
    trial_starts = np.array([trials['start_time'][i] for i in range(n_trials)])
    obs_to_trial = []
    for i in range(n_obs):
        diffs = np.abs(trial_starts - obs[i, 0])
        min_idx = np.argmin(diffs)
        if diffs[min_idx] < 0.5:  # 0.5s tolerance
            obs_to_trial.append(int(min_idx))
        else:
            obs_to_trial.append(-1)  # no match
    return obs_to_trial
```

iii. The AI hit this while running the first full conversion: session `sub-440956_ses-20190208T133600` has 480
trial rows but only 160 `obs_intervals` (steps 72–79). It verified that all units in a file share the same
`obs_intervals`, that the observed trials are a contiguous leading block, and that the interval starts line up
with `trials.start_time`, then wrote the mapping with a tolerance to be robust.

## 1-e. How are trials filtered based on quality controls?

i. Two filters. (1) The trial must be covered by `obs_intervals`, i.e. have neural data. (2) Following the
data paper, **early lick trials and no-response ('ignore') trials are excluded**. A session with fewer than 2
surviving trials is dropped. No other trial-level curation is applied — in particular `free_water` trials are
kept, and these have no spikes at all, which is the source of the ~1,212 "all neural data is zero" warnings in
the verification output. 48,356 trials survive in 105 sessions.

ii.
```python
# ---- 5. Filter trials ----
# Per methods: "Early lick trials and no response trials were excluded for analysis"
# Also must have neural data (be in obs_intervals)
valid_trial_indices = []
for i in range(n_trials):
    if i not in trials_with_neural:
        continue
    outcome = trials['outcome'][i]
    early = trials['early_lick'][i]
    if early != 'no early':
        continue
    if outcome == 'ignore':
        continue
    valid_trial_indices.append(i)

if len(valid_trial_indices) < 2:
    print(f"  Skipping {os.path.basename(nwb_path)}: <2 valid trials after filtering")
```

iii. The AI wrestled with this explicitly and changed its mind twice (steps 63, 66, 68). It first noted the
conflict: "If we exclude those, early_lick is always 0 and outcome never has 0 … The decoder output explicitly
includes early_lick and outcome with ignore. So I should NOT exclude early lick or ignore trials." It then
edited the code to keep all trials, then reverted, reasoning: "filtering is part of 'curation of data' which
must match the reference … The output encoding stays the same (output_values still lists all possible values),
just some values won't appear in the data." `CONVERSION_NOTES.md` records the consequence: "After filtering,
`early_lick` output is always 0 and `outcome` never has value 0". For the all-zero trials the notes give a
different (and incorrect) explanation: "the obs_intervals for those trials overlap very little with the actual
firing window"; the actual cause is that they are `free_water` trials containing no spikes anywhere.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (session-absolute seconds) for the units whose `units/classification` is `'good'`,
combined with `units/obs_intervals` (used to restrict the spikes considered) and the go-cue times from
`acquisition/BehavioralEvents/go_start_times`.

ii.
```python
def preload_spike_times(nwb, unit_indices):
    return [nwb.units['spike_times'][uid] for uid in unit_indices]
...
all_spike_times = preload_spike_times(nwb, good_indices)
obs_intervals = units['obs_intervals'][good_indices[0]]
go_start_times = be.time_series['go_start_times'].timestamps[:]
```

iii. `spike_times` is the only neural representation in the file. The AI inspected the units table
(steps 23–28) and confirmed `classification`, `obs_intervals` and the ragged `spike_times` layout before
using them.

## 2-b. How is the `neural` data processed?

i. Per trial: spikes of each good unit are first restricted to that trial's `obs_intervals` window, then
histogrammed into 80 bins of 50 ms spanning `[go - 2.5 s, go + 1.5 s)` and divided by the bin width to give
firing rate in Hz (`float32`). No smoothing, normalisation or baseline subtraction. Note that the
`obs_intervals` restriction is an extra step not present in the stated decision: `obs_intervals` ends at the
trial's `stop_time`, which in a minority of retained trials (≈1–15% per session, average loss ≈0.1 s, up to
~0.5 s) falls before `go + 1.5 s`, so the trailing bins of those trials are silently set to 0 Hz.

ii.
```python
def get_spike_times_for_trial_from_cache(all_spike_times, obs_intervals, obs_idx):
    spike_times_list = []
    t_start, t_stop = obs_intervals[obs_idx]
    for st in all_spike_times:
        mask = (st >= t_start) & (st < t_stop)
        spike_times_list.append(st[mask])
    return spike_times_list
```
```python
n_bins = int(round((end_time - begin_time) / bin_width))
bin_edges = np.linspace(begin_time, end_time, n_bins + 1) + go_cue_time
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0 - go_cue_time
fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
for i, st in enumerate(spike_times_by_neuron):
    mask = (st >= bin_edges[0]) & (st < bin_edges[-1])
    st_window = st[mask]
    if len(st_window) > 0:
        counts, _ = np.histogram(st_window, bins=bin_edges)
        fr[i, :] = counts.astype(np.float32) / bin_width
```

iii. `CONVERSION_NOTES.md`: "Spike counts are histogrammed into 50ms bins and divided by bin width (0.05s) to
get firing rates in Hz. This is simpler than the sliding kernel approach in the original preprocessing code
but appropriate for the decoder task." The obs-interval restriction is a side effect of the data structure the
AI adopted in steps 76–90 to solve the "trials without ephys" problem; it is not discussed as a processing
choice anywhere in the notes.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units['classification'] == 'good'` are used; a session with zero such units is skipped
entirely (this drops `sub-440958_ses-20190216T162508`). No individual quality-metric thresholds and no use of
the older `unit_quality` field. 41,197 good units are retained across the 105 kept sessions (median ≈390 per
session).

ii.
```python
good_indices = []
for i in range(n_units):
    if units['classification'][i] == 'good':
        good_indices.append(i)

if len(good_indices) == 0:
    print(f"  Skipping {os.path.basename(nwb_path)}: 0 good units")
    io.close()
    return None
```

iii. Header comment: "QC filtering: use 'good' classification from NWB (matches classifier-based QC in
paper)". This refers to the region-specific logistic-regression classifiers described in `methods.txt` and the
Chen/Liu spike-sorting white paper; the classifier verdict is already stored per unit, so no metric
thresholding is required. The AI noted the 0-good-unit session in step 55 as a special case to skip.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to go-cue onset, taken from `BehavioralEvents/go_start_times.timestamps[trial_idx]`. Spikes
and event timestamps share the session-absolute clock, so alignment is done by adding the go-cue time to the
fixed relative bin edges `[-2.5, 1.5]`; no resampling or interpolation.

ii.
```python
go_cue = go_start_times[trial_idx]
...
bin_edges = np.linspace(begin_time, end_time, n_bins + 1) + go_cue_time
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0 - go_cue_time
```

iii. The AI verified in steps 35–38 that `go_start_times` has one event per trial on the same clock as the
spike times and that the last `sample_start_times` entry sits a constant −1.85 s from the go cue, which
confirmed the clock alignment before it committed to this scheme.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins, 80 bins per trial covering −2.5 s to +1.5 s around the go cue; identical grid for every trial
and session. No rebinning or resampling: spikes are histogrammed directly onto this grid, so there is only one
binning step. `metadata['time_bin_size'] = 50.0` (ms).

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
BEGIN_TIME = -2.5  # relative to go cue
END_TIME = 1.5    # relative to go cue
...
n_bins = int(round((END_TIME - BEGIN_TIME) / BIN_WIDTH))
...
'time_bin_size': BIN_WIDTH * 1000,  # in ms
```

iii. Taken directly from the Decoder Task specification ("2.5 s before to 1.5 s after the go cue",
"50-ms-width bins"). The AI records this as decision 6/7 in its plan (step 42) and the verification output
confirms `T: min 80, max 80` for all sessions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `acquisition/BehavioralEvents/sample_start_times.timestamps` (the tone onsets), `trials.start_time`, and
the trial's go-cue time. The tone used for a trial is the **last** `sample_start_time` that falls inside
`[trial start, go cue)`; if there is none, a fallback of `go_cue - 1.85 s` is used.

ii.
```python
t_start = trials['start_time'][trial_idx]
valid_samples = sample_start_times[
    (sample_start_times >= t_start) & (sample_start_times < go_cue)
]
if len(valid_samples) > 0:
    tone_onset = valid_samples[-1]  # last sample start = actual tone onset
else:
    tone_onset = go_cue - 1.85  # fallback: typical sample-delay duration
```

iii. Step 38: "The last sample start before the go cue is always −1.85 s relative to go cue for the main sample
epoch. Earlier ones are replays due to early licking." This matches `methods.txt` ("Licking early during the
sample/delay epoch triggered a replay of the epoch"), so the last tone before the go cue is the one the animal
actually used.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone onset is expressed relative to the go cue, and each bin's value is its (go-cue-relative) centre
minus that offset — i.e. seconds elapsed since the tone at the centre of each bin. Stored as `float32` in row
0 of the `(2, 80)` input array. It is a continuous, monotonically increasing ramp, not a binary series.

ii.
```python
tone_onset_rel = tone_onset - go_cue  # relative to go cue (should be ~-1.85)
time_from_tone = bin_centers - tone_onset_rel  # time since tone onset at each bin
...
input_trial = np.stack([time_from_tone.astype(np.float32), photostim_on], axis=0)  # (2, n_bins)
```

iii. No extra processing is described beyond locating the tone. The AI checked the resulting ranges in the
verification output and documented them in `CONVERSION_NOTES.md` §7: `[-0.6, 3.3]` for early sessions,
`[-1.5, 3.9]` for sessions 40+ ("a different sample-to-go-cue timing … genuine experimental variation"), plus
one outlier trial reaching 7.4 s that it attributed to "a replay/repeated stimulus" and left in place.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed from `bin_centers`, the very array returned by `compute_firing_rates`, so bin *k* of the
input is by construction the same 50 ms interval as bin *k* of the firing rates. No separate alignment step.

ii.
```python
fr, bin_centers = compute_firing_rates(
    spike_times_trial, go_cue, BEGIN_TIME, END_TIME, BIN_WIDTH
)
...
time_from_tone = bin_centers - tone_onset_rel
```

iii. N/A — alignment is inherited from the shared go-cue-relative grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. `trials.photostim_onset` and `trials.photostim_duration` (strings, with `'N/A'` on non-stimulated trials,
measured from `trials.start_time`), plus `trials.start_time` and the go cue to move them onto the
go-cue-relative axis.

ii.
```python
photostim_onset_trial = trials['photostim_onset'][trial_idx]
if photostim_onset_trial != 'N/A':
    onset_val = float(photostim_onset_trial)
    dur_val = float(trials['photostim_duration'][trial_idx])
    photostim_start_abs = t_start + onset_val
    photostim_stop_abs = photostim_start_abs + dur_val
```

iii. Steps 45–49: the AI inspected the photostim fields, found the onset is stored as a string relative to
trial start, converted one example by hand (onset 1.898 with go cue at trial_start + 3.098 → −1.2 s re go
cue, duration 0.5 s) and cross-checked against the paper's description of ALM photoinhibition during the
delay. It noted the timing did not match "last 0.5 s of delay" exactly and concluded: "Regardless, for the
decoder, I just need to mark which time bins have photostim on."

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1, `float32`) time series over the 80 bins: bins whose centre lies in
`[onset, onset + duration)` (expressed re go cue) are set to 1, all others 0. Non-stimulated trials get an
all-zero row. It is row 1 of the input array.

ii.
```python
photostim_on = np.zeros(len(bin_centers), dtype=np.float32)
...
    ps_start_rel = photostim_start_abs - go_cue
    ps_stop_rel = photostim_stop_abs - go_cue
    photostim_on[(bin_centers >= ps_start_rel) & (bin_centers < ps_stop_rel)] = 1.0
```

iii. The instructions require "whether photostimulation is on at every time point (discrete, time-varying)", so
the AI encodes it per bin rather than as a per-trial flag. (`CONVERSION_NOTES.md` §8 describes this loosely
and inaccurately — "If any photostim occurs within the trial window [-2.5, 1.5], photostimulation is set to
1" — but the code marks only the stimulated bins.)

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Onset and offset are converted from trial-start-relative to go-cue-relative seconds and compared against
`bin_centers`, the same grid used for the firing rates, so no independent alignment is needed.

ii.
```python
ps_start_rel = photostim_start_abs - go_cue
ps_stop_rel = photostim_stop_abs - go_cue
photostim_on[(bin_centers >= ps_start_rel) & (bin_centers < ps_stop_rel)] = 1.0
```

iii. N/A.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no choice column in the NWB file. Choice is derived from `trials.trial_instruction`
(`'left'`/`'right'`) combined with `trials.outcome`: on a hit the animal licked the instructed side, on a miss
it licked the other side.

ii.
```python
instruction = trials['trial_instruction'][trial_idx]
outcome = trials['outcome'][trial_idx]
if outcome == 'hit':
    choice = 0 if instruction == 'left' else 1
elif outcome == 'miss':
    choice = 1 if instruction == 'left' else 0  # miss means licked wrong side
else:
    choice = 0  # shouldn't happen after filtering
```

iii. Step 68: "for 'miss' trials, the mouse licked the wrong side, so choice is opposite of instruction. For
'ignore', there's no lick at all." Having decided to exclude ignore trials, the AI left a defensive `else`
branch defaulting to left: "For ignore trials … I'll assign it to the instruction side … which makes it
somewhat arbitrary but consistent" (in the shipped code the default is a constant 0, and the branch is dead).

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Two categories only — `0 = left`, `1 = right` (`output_values[0] = ['left', 'right']`). The per-trial value
is broadcast across all 80 bins into row 0 of an `(4, 80)` `int64` output array. There is no "no lick"
category, because ignore trials were removed upstream. The verification output shows the realised
distribution as left 0.492 / right 0.508.

ii.
```python
output_trial = np.array([choice, outcome_val, early_lick_val], dtype=np.int64)  # (3,) - tongue added later
...
output_trial = np.zeros((4, n_bins), dtype=np.int64)
output_trial[0, :] = base_output[0]  # choice (constant across time)
```
```python
'output_values': [
    ['left', 'right'],           # choice
    ...
```

iii. The coding follows the AI's copy of the instructions, which listed choice as "left = 0, right = 1" with
no third value. Per-trial values are broadcast over bins so that all four outputs can live in a single
`(n_output, n_timepoints)` array alongside the time-varying tongue output.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which stores `'hit'`, `'miss'` or `'ignore'`.

ii.
```python
outcome = trials['outcome'][trial_idx]
```

iii. The column already holds exactly the three categories the instructions ask for, so no derivation is
needed. The AI checked the outcome vocabulary across subjects (step 50) before relying on it.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A fixed mapping `{'ignore': 0, 'miss': 1, 'hit': 2}` (with `.get(outcome, 0)` as a fallback), broadcast
across the 80 bins into row 1 of the output array. Because ignore trials were filtered out, only values 1 and
2 actually occur (verification output: miss 0.144, hit 0.856).

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome_val = outcome_map.get(outcome, 0)
...
output_trial[1, :] = base_output[1]  # outcome (constant across time)
```
```python
'output_values': [
    ...
    ['ignore', 'miss', 'hit'],   # outcome
```

iii. The code assignment follows the instructions verbatim. `CONVERSION_NOTES.md` §4: "After filtering …
`outcome` never has value 0 (ignore). The output_values definitions still list all possible values for
completeness."

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds `'no early'` / `'early'`.

ii.
```python
early_lick_val = 0 if trials['early_lick'][trial_idx] == 'no early' else 1
```

iii. The flag is stored explicitly per trial, so no derivation from lick times is needed. The AI also verified
(step 67/68) that early-lick trials can still end in a hit or miss because the sample/delay epoch is replayed.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to `0 = no` / `1 = yes` and broadcast across all 80 bins into row 2 of the output array. Since all
early-lick trials were removed by the trial filter, this output is constant 0 for all 48,356 trials in the
shipped dataset (verification output: `early_lick: {no (1.000)}`, decoder balanced accuracy 100%, trivially).

ii.
```python
# 3. Early lick: no=0, yes=1 (should be all 0 after filtering)
early_lick_val = 0 if trials['early_lick'][trial_idx] == 'no early' else 1
...
output_trial[2, :] = base_output[2]  # early_lick (constant across time)
```

iii. The AI knowingly accepted the degeneracy: "Early lick: no=0, yes=1 - all 0 after filtering. This is a
trivially perfect variable" (step 63) and later, after reverting its fix, "the output encoding stays the same
…, just some values won't appear in the data" (step 68). It reported the result in `CONVERSION_NOTES.md` as
"Early lick | 100% (trivial)".

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: `data` is `(n_frames, 3)` =
`(tongue_x, tongue_y, tongue_likelihood)` with matching `timestamps`. Column 1 is the value, column 2 the
DeepLabCut tracking likelihood.

ii.
```python
bts = nwb.acquisition['BehavioralTimeSeries']
tongue_ts = bts.time_series['Camera0_side_TongueTracking']
tongue_data = tongue_ts.data[:]  # (n_frames, 3): x, y, likelihood
tongue_times = tongue_ts.timestamps[:]
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
```

iii. Steps 29–32: the AI enumerated the `BehavioralTimeSeries` members across two subjects, confirmed the
three-column layout, and concluded "The tongue tracking has (x, y, likelihood) columns from the side camera.
The 'tongue y-position' for discretization is from the side view tongue tracking."

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Per trial: frames within `[go - 2.55, go + 1.55)` are selected, frames with `likelihood <= 0.5` are
dropped, the remaining frames are averaged within each 50 ms bin, and bins with no surviving frame are left
as `NaN`. After all trials of a session are processed, the per-bin means from all trials are concatenated,
NaNs removed, and the 40th/60th percentiles computed **per session** from that pooled set.

ii.
```python
lh_mask = tw_lh > 0.5
tw_times_good = tw_times[lh_mask]
tw_y_good = tw_y[lh_mask]

tongue_y_bins = np.full(len(bin_centers), np.nan, dtype=np.float32)
if len(tw_times_good) > 0:
    abs_bin_edges = np.linspace(BEGIN_TIME, END_TIME, len(bin_centers) + 1) + go_cue
    bin_assignments = np.digitize(tw_times_good, abs_bin_edges) - 1
    for b in range(len(bin_centers)):
        b_mask = bin_assignments == b
        if np.any(b_mask):
            tongue_y_bins[b] = np.mean(tw_y_good[b_mask])
```
```python
all_tongue_y = np.concatenate([v for v in s['tongue_y_trial_values']])
valid_tongue = all_tongue_y[~np.isnan(all_tongue_y)]
if len(valid_tongue) > 0:
    p40 = np.percentile(valid_tongue, 40)
    p60 = np.percentile(valid_tongue, 60)
```

iii. `CONVERSION_NOTES.md` §10: "Extracted from DeepLabCut tracking (side camera, Camera0); only positions
with tracking likelihood > 0.5 are used; per-session discretization using 40th and 60th percentiles". The
per-session scope and percentile values come straight from the Decoder Task spec; the likelihood threshold is
the AI's own addition to discard frames where the tongue is not actually visible.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three classes: `0` below the 40th percentile, `1` between the 40th and 60th (inclusive at the upper edge),
`2` above the 60th. **Bins with no visible tongue (NaN) are assigned class 0**, i.e. merged with "low
position", rather than given their own category. The consequence is a heavily skewed output: 83.1% class 0,
5.6% class 1, 11.3% class 2 (of which the no-tongue bins are roughly 72 percentage points of class 0).

ii.
```python
def discretize_tongue_y(tongue_y_values, p40, p60):
    """
    0: < 40th percentile
    1: 40th to 60th percentile
    2: > 60th percentile
    NaN values get class 0 (below 40th percentile - represents no tongue visible)
    """
    result = np.zeros(len(tongue_y_values), dtype=np.int64)
    for i, val in enumerate(tongue_y_values):
        if np.isnan(val):
            result[i] = 0  # no tongue visible = low position
        elif val < p40:
            result[i] = 0
        elif val <= p60:
            result[i] = 1
        else:
            result[i] = 2
    return result
```

iii. The AI's copy of the instructions listed only three tongue classes (no "not visible" class), so it folded
invisible bins into the lowest class on the rationale recorded in the code comment — "no tongue visible = low
position" — and in `CONVERSION_NOTES.md`: "Time bins without valid tracking data default to 0 (low)". The
resulting class imbalance is not discussed.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are on the same session-absolute clock as spikes and events. Frames are selected by an
absolute-time window around the go cue (padded by one bin width on each side) and assigned to bins with
`np.digitize` against `np.linspace(BEGIN_TIME, END_TIME, n_bins+1) + go_cue` — the same edges used for the
firing rates — so bin *k* of the tongue output spans the same interval as bin *k* of the neural data. Frames
falling in the padding region land outside `0..79` and are ignored by the per-bin loop.

ii.
```python
trial_window_start = go_cue + BEGIN_TIME - BIN_WIDTH
trial_window_end = go_cue + END_TIME + BIN_WIDTH
tw_mask = (tongue_times >= trial_window_start) & (tongue_times < trial_window_end)
...
abs_bin_edges = np.linspace(BEGIN_TIME, END_TIME, len(bin_centers) + 1) + go_cue
bin_assignments = np.digitize(tw_times_good, abs_bin_edges) - 1
```

iii. The AI treats all streams as sharing one clock (verified when checking go-cue/sample timing in steps
35–38), so it reuses the same go-cue-relative edge construction rather than interpolating.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled, one is knowingly left in:
- **Sessions never quality-controlled** (no `'good'` units) → session dropped.
- **Missing/NaN brain-area annotation** → relabelled `'unknown'`, which `map_region_to_major` passes through.
- **Fewer `obs_intervals` than trial rows** → obs↔trial mapping with 0.5 s tolerance; unmatched trials dropped.
- **Sessions left with <2 usable trials** → dropped.
- **No tone onset inside the trial** → fallback `go_cue - 1.85 s`.
- **Tongue frames with low likelihood / empty bins** → excluded from the mean, resulting NaN bins coded as class 0.
- **Trials with no spikes at all** (the `free_water` trials) → *kept*, producing ~1,212 all-zero trials that
  the verifier flags; the notes attribute them to a different cause than the real one.

ii.
```python
anno = units['anno_name'][i]
if anno is None or anno == '' or anno == 'nan':
    anno = 'unknown'
```
```python
if diffs[min_idx] < 0.5:  # 0.5s tolerance
    obs_to_trial.append(int(min_idx))
else:
    obs_to_trial.append(-1)  # no match
```
```python
else:
    tone_onset = go_cue - 1.85  # fallback: typical sample-delay duration
```
```python
if np.isnan(val):
    result[i] = 0  # no tongue visible = low position
```

iii. `CONVERSION_NOTES.md` "Known Issues / Warnings": "Some trials in sessions 40-42 (and a few others) have
all-zero neural data across all neurons and time bins. This occurs because the obs_intervals for those trials
overlap very little with the actual firing window… These trials are retained in the data but flagged as
warnings during verification." (Checked against the source: those trials are `free_water == 1` trials whose
`obs_intervals` are of normal length but contain zero spikes for every unit.) For the tongue, the AI's stated
reasoning is that an invisible tongue is equivalent to a retracted, low-position tongue.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant cost is `get_spike_times_for_trial_from_cache`, which re-scans **every** good unit's complete
spike train with a boolean mask **once per trial** — O(n_trials × n_units × n_spikes_per_unit), i.e. on the
order of 10⁹ element comparisons per session (≈400 trials × ≈400 units × ≈10⁴–10⁵ spikes). Secondary costs are
the initial ragged read of `spike_times` unit by unit, the `(n_frames, 3)` tongue array read, the per-neuron
`np.histogram` calls (one per unit per trial), and the element-by-element reads of the trials table in the
session-criteria and trial-filter loops. Pickling the 5.95 GB result is also significant. The first full run
was slow enough that the AI killed it after ~30 minutes of CPU time and partially optimised before rerunning.

ii.
```python
for st in all_spike_times:
    mask = (st >= t_start) & (st < t_stop)
    spike_times_list.append(st[mask])
```
```python
for i, st in enumerate(spike_times_by_neuron):
    mask = (st >= bin_edges[0]) & (st < bin_edges[-1])
    st_window = st[mask]
    if len(st_window) > 0:
        counts, _ = np.histogram(st_window, bins=bin_edges)
```

iii. The AI diagnosed the bottleneck as NWB access only: "The key bottleneck is `get_spike_times_for_trial`
which reads spike_times for each unit individually from the NWB file. I need to batch-read all spike times
upfront" (step 115). It fixed the repeated file access with `preload_spike_times` but kept the per-trial
full-spike-train masking in memory, and did not revisit the cost after the rerun succeeded.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Five:
1. The per-trial × per-unit masking loop in `get_spike_times_for_trial_from_cache` — replaceable by one
   `np.searchsorted` per unit over all trials' bin edges at once (as the reference does), removing the trial
   dimension entirely.
2. The per-unit `np.histogram` loop in `compute_firing_rates` — same fix; `np.diff(np.searchsorted(...))` is
   far cheaper than histogramming.
3. The per-bin loop `for b in range(len(bin_centers))` in the tongue extraction — replaceable by
   `np.bincount` on the bin assignments.
4. The element-by-element trials-table loops (`for i in range(n_trials): trials['outcome'][i] …`) — the whole
   columns could be read once as arrays and filtered with boolean masks.
5. `discretize_tongue_y`'s per-value Python loop — `np.digitize` on the whole array.

ii.
```python
for i in range(n_trials):
    is_control = trials['photostim_onset'][i] == 'N/A'
    is_no_early = trials['early_lick'][i] == 'no early'
    is_hit = trials['outcome'][i] == 'hit'
```
```python
for b in range(len(bin_centers)):
    b_mask = bin_assignments == b
    if np.any(b_mask):
        tongue_y_bins[b] = np.mean(tw_y_good[b_mask])
```
```python
for i, val in enumerate(tongue_y_values):
    if np.isnan(val):
        result[i] = 0
```

iii. The AI did vectorise one of these deliberately: "Let me vectorize the tongue tracking extraction"
(step 123) replaced a per-frame search with a `np.digitize`-plus-per-bin loop, and `preload_spike_times`
removed repeated NWB reads. It did not attempt the others; there is no recorded reasoning for leaving the
per-trial spike masking in place.

## 10-c. What processing does the code repeat multiple times?

i. Repeated work:
- The trials table is walked element-by-element **three times** per session (session-criteria loop, trial
  filter loop, then per-trial processing), re-reading the same columns each time.
- Every unit's full spike train is re-masked for every trial (see 10-a).
- `np.linspace` bin edges and `bin_centers` are rebuilt for every trial in both `compute_firing_rates` and the
  tongue block, although the grid is constant.
- `map_region_to_major` is called twice per neuron — once to build the region set, once to build the index
  array.
- The `sample_start_times` array is re-filtered against the trial window for every trial.
- The whole dataset is walked again in `print_stats` to produce summary statistics.

ii.
```python
all_region_labels = set()
for s in all_sessions:
    for label in s['brain_region_labels']:
        all_region_labels.add(map_region_to_major(label))
...
for s in all_sessions:
    idx = np.array([brain_regions.index(map_region_to_major(label))
                    for label in s['brain_region_labels']], dtype=np.int64)
```
```python
valid_samples = sample_start_times[
    (sample_start_times >= t_start) & (sample_start_times < go_cue)
]
```

iii. Not discussed in the notes or trajectory; these are incidental repetitions rather than deliberate
choices. Only the NWB spike-time reads were identified as repeated work and fixed.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Modest amounts:
- The `mask = (st >= bin_edges[0]) & (st < bin_edges[-1])` pre-filter inside `compute_firing_rates` is
  redundant, since `np.histogram` already ignores out-of-range spikes — and the spikes have already been
  masked once by the obs interval.
- The tongue window is padded by ±1 bin width, and those extra frames are then discarded by the per-bin loop.
- `bin_centers` is returned from `compute_firing_rates` for every trial but only its (identical) values are
  used.
- `perf`, `n_good_units`, `n_valid_trials` are carried in the per-session result dict but never written into
  the output dictionary.
- Outputs are stored as `int64` where `int8` would do (4 values per trial × 80 bins × 48k trials).
- The `early_lick` output channel is computed and stored for every trial and bin although it is constant, and
  the ~1,212 all-zero neural trials contribute nothing to decoding.
- `print_stats` makes a second full pass over the assembled data purely for logging, and a separate
  `sample_data.pkl` is produced by re-running the whole pipeline (this one was required by the AI's own task
  instructions).

ii.
```python
mask = (st >= bin_edges[0]) & (st < bin_edges[-1])
st_window = st[mask]
```
```python
return {
    ...
    'n_good_units': len(good_indices),
    'n_valid_trials': len(valid_trial_indices),
    'perf': perf,
}
```

iii. None of this is discussed; the AI's stated optimisation goal was only to make the full run finish
("The bottleneck is reading large NWB files and processing spike times"), and it stopped optimising once the
background run completed.
