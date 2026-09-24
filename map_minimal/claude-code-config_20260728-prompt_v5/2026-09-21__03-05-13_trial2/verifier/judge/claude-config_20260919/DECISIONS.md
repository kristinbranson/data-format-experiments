# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset (DANDI 000363) is one NWB file per session, laid out as `data/sub-<id>/sub-<id>_ses-<timestamp>_behavior+ecephys[+ogen].nwb`. The AI finds every session with a single sorted glob and opens each file **directly with `h5py`** rather than with `pynwb`, reading the HDF5 groups by path (`intervals/trials`, `units`, `acquisition/BehavioralEvents/...`, `acquisition/BehavioralTimeSeries/...`, `general/extracellular_ephys/electrodes`). All 174 files are found and processed in one pass; per-session results are accumulated into lists and stitched together at the end. String columns are decoded manually from bytes.

ii.
```python
DATA_DIR = '/app/data'

def main():
    nwb_files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', 'sub-*.nwb')))
    print(f"Found {len(nwb_files)} NWB files")
    ...
    for i, fpath in enumerate(nwb_files):
        result = process_session(fpath)
```
```python
def process_session(fpath):
    """Process one NWB session file. Returns session data dict or None."""
    with h5py.File(fpath, 'r') as f:
        trials = f['intervals/trials']
        n_trials = len(trials['id'][:])
        outcomes = np.array([x.decode() if isinstance(x, bytes) else str(x)
                            for x in trials['outcome'][:]])
        ...
        go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
        units = f['units']
        all_spike_times = units['spike_times'][:]
        spike_times_index = units['spike_times_index'][:]
```

iii. The agent first explored one NWB file in detail (via a sub-agent) and enumerated the units table (1,952 units, 39 columns), the trials table, the behavioral event streams and the DLC tracking series before writing any code. It chose raw `h5py` access over `pynwb` for speed — the trajectory shows the first implementation was killed for being too slow, and the agent reworked the loading/binning path explicitly for throughput ("2.5 seconds per session — much better. Full conversion should take about 7-8 minutes"). Because the archive stores exactly one session per file, a glob over `sub-*/` is the complete session list.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the **file name prefix** (`sub-440956`), not from the NWB `general/subject/subject_id` field. Subjects are accumulated in first-encounter order into `all_subjects`, and `subject_idx` is each session's index into that list. This yields 28 subjects, the full set in the dandiset.

ii.
```python
basename = os.path.basename(fpath)
subject_id = basename.split('_')[0]
```
```python
if result['subject_id'] not in all_subjects:
    all_subjects.append(result['subject_id'])
...
'subjects': all_subjects,
'subject_idx': np.array([all_subjects.index(s['subject_id']) for s in all_sessions], dtype=np.int64),
```

iii. The agent noted from its initial exploration that the directory/file naming (`sub-<numeric id>`) is derived from the NWB subject id, so the filename prefix is an equivalent and cheaper identifier (no need to read the `subject` group). No explicit written justification beyond that; the subject count (28) was checked against the directory listing.

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no grouping or splitting is inferred. Sessions are ordered by the sorted file list (chronological within subject because the filename embeds the acquisition timestamp), and the session is identified downstream by its basename (`session_name`).

Importantly, the AI additionally applies a **session-level behavioural curation step** taken verbatim from the data-paper methods: a session is dropped unless (a) performance on control trials — `hits / (hits + misses)` over non-photostim, non-auto-water, non-free-water, non-early-lick trials — exceeds 65%, and (b) it has at least 50 correct lick-left and 50 correct lick-right control trials. This dropped 29 of 174 sessions, leaving **145 sessions**.

ii.
```python
is_control = (photostim_onset_raw == 'N/A') & (auto_water == 0) & (free_water == 0)
is_not_early = (early_licks == 'no early')
control_regular = is_control & is_not_early

control_hits = np.sum(outcomes[control_regular] == 'hit')
control_misses = np.sum(outcomes[control_regular] == 'miss')
control_responded = control_hits + control_misses
if control_responded == 0:
    return None
performance = control_hits / control_responded
if performance < MIN_PERFORMANCE:          # 0.65
    return None

correct_left = np.sum((outcomes == 'hit') & (instructions == 'left') & control_regular)
correct_right = np.sum((outcomes == 'hit') & (instructions == 'right') & control_regular)
if correct_left < MIN_CORRECT_PER_SIDE or correct_right < MIN_CORRECT_PER_SIDE:   # 50
    return None
```

iii. `methods.txt` states: "We selected experimental sessions for analysis based on following criteria: overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each." The agent implemented exactly this and deliberated over the denominator: its first version used hits / all control trials, which gave 0.44 on the first session; it re-read the methods, compared both interpretations across sessions, and settled on `hits/(hits+misses)` because the resulting per-session rates (83–94%) match the paper's reported 84% mean and 65–99% range. It accepted that some released sessions fail the criterion ("That's fine — it's a real session that doesn't meet the criteria").

## 1-d. How are the data split into trials?

i. Trials are the rows of the NWB trials table (`intervals/trials`), one row per behavioural trial, and every per-trial quantity is indexed by that row. Alignment between the trials table and the go-cue event stream is assumed positionally (`go_times[trial_idx]`); the AI verified during exploration that `n_go == n_trials` (368 vs 368) on the session it examined but does not assert it in the conversion code. It also noted that `sample_start_times`/`delay_start_times` contain *more* events than trials (405 and 395 vs 368) because early licks replay the epoch, so those streams cannot be used positionally.

ii.
```python
trials = f['intervals/trials']
n_trials = len(trials['id'][:])
...
go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
...
for t_i, trial_idx in enumerate(trial_indices):
    go_t = go_times[trial_idx]
```

iii. From the trajectory: "n_go is 368 matching trial count, but n_sample is 405 and n_delay is 395" — so the go-cue stream is one-per-trial and can index the trials table directly, while the sample/delay streams need a search (see 3-a).

## 1-e. How are trials filtered based on quality controls?

i. Three filters, all at trial level:
1. **`auto_water == 1` trials are excluded** ("not real behavioral trials"). ~0.9% of trials.
2. **`free_water == 1` trials are excluded** for the same reason. ~3% of trials.
3. **Trials outside the ephys recording** are excluded: the AI takes the min and max spike time over all retained units and keeps a trial only if its full window `[go − 2.5, go + 1.5]` lies inside `[min_spike − 1 s, max_spike + 1 s]`.
A session with fewer than 2 surviving trials is dropped. **Early-lick and no-response (`ignore`) trials are deliberately kept**, against the data paper's analysis convention, because they are required decoder outputs. Photostim trials are kept because photostim is a decoder input. Result: 74,908 trials over 145 sessions.

ii.
```python
# ---- Trial mask: exclude auto_water and free_water ----
trial_mask = (auto_water == 0) & (free_water == 0)
trial_indices = np.where(trial_mask)[0]
if len(trial_indices) < 2:
    return None
```
```python
# ---- Determine neural recording range ----
min_spike_time = np.inf
max_spike_time = 0.0
for uid in good_indices:
    start = spike_times_index[uid - 1] if uid > 0 else 0
    end = spike_times_index[uid]
    if end > start:
        min_spike_time = min(min_spike_time, all_spike_times[start])
        max_spike_time = max(max_spike_time, all_spike_times[end - 1])

valid_trial_mask = np.array([
    (go_times[ti] - TIME_BEFORE >= min_spike_time - 1.0) and
    (go_times[ti] + TIME_AFTER <= max_spike_time + 1.0)
    for ti in trial_indices
])
trial_indices = trial_indices[valid_trial_mask]
```

iii. The recording-range filter was added reactively and is well documented in the trajectory: the format checker reported "all neural data is zero" warnings, the agent traced them to sessions where "spike times end around 1107 seconds, but the go times extend to 3707 seconds", added a max-spike-time bound, re-ran, found one remaining session where "spikes START at ~1679s, not at the beginning", and added the min-spike-time bound as well. After that the verifier reported "Data format is valid, no errors or warnings." The early-lick/ignore retention is justified against the instructions: "since the decoder needs to predict early_lick and outcome (including ignore/miss), I shouldn't exclude those trials even though the reference analysis code does — that's for a different neural encoding analysis, not decoding."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (a single concatenated, per-unit-sorted array) together with `units/spike_times_index` (the ragged offsets), restricted to the units selected by QC (see 2-c), plus `acquisition/BehavioralEvents/go_start_times/timestamps` which supplies the per-trial alignment time. Both are in session-absolute seconds.

ii.
```python
all_spike_times = units['spike_times'][:]
spike_times_index = units['spike_times_index'][:]
...
neural_trials = compute_firing_rates_all_trials(
    all_spike_times, spike_times_index, good_indices, go_times, trial_indices)
```
```python
for uid in unit_indices:
    start = spike_idx[uid - 1] if uid > 0 else 0
    end = spike_idx[uid]
    unit_spikes.append(spike_times[start:end])
```

iii. Spike times are the only neural representation in the file; the agent's exploration summary recorded "spike_times — spike timestamps (with spike_times_index for per-unit access)" and "Spike times are in absolute time (session time)".

## 2-b. How is the `neural` data processed?

i. Spike counts per 50 ms bin, divided by the bin width, i.e. firing rate in Hz. Per unit, the 81 absolute bin edges for a trial are formed by adding the go-cue time to the fixed relative edge vector, `np.searchsorted` gives the running spike count at each edge, and `np.diff` gives the count per bin. Stored as `float32`. No smoothing, no baseline subtraction, no normalisation, no spike-time jitter correction.

ii.
```python
rates_all = np.zeros((n_neurons, n_trials, N_BINS), dtype=np.float32)
for i, st in enumerate(unit_spikes):
    if len(st) == 0:
        continue
    for t_idx in range(n_trials):
        go_t = go_subset[t_idx]
        abs_edges = go_t + BIN_EDGES
        edge_counts = np.searchsorted(st, abs_edges)
        rates_all[i, t_idx, :] = np.diff(edge_counts) / BIN_SIZE
return [rates_all[:, t, :] for t in range(n_trials)]
```

iii. The header states: "Firing rates: spike counts per bin / bin_width (same approach as reference code's sliding_histogram with non-overlapping bins)." The agent had read `sliding_histogram(...)` in `/app/code/VideoAnalysisUtils/preprocessing_utils.py`, which counts spikes in each bin interval and returns `binSpikes / bin_width` when `rate=True`; with stride equal to bin width the sliding windows become the non-overlapping bins used here.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with **`units/unit_quality == 'good'`** are kept; units labelled `'multi'` are dropped. No metric thresholds and no per-trial unit masking (`units/is_good_trials` is not used). A session with no such unit is dropped (none is). This retains **129,690 units over 145 sessions** (≈63% of clusters; e.g. 1,222 of 1,952 in the first session).

The AI's documented intent was the paper's classifier-based QC, but `unit_quality` is not that field: the same units table also carries **`units/classification`** with values `'good'`/`'unlabelled'`, which is the output of the region-specific logistic-regression classifiers described in `methods.txt` and the Chen/Liu white paper (459 of 1,952 in that session, ≈25%). The AI's exploration summary enumerated `unit_quality` ("1,222 good, 730 multi") and never surfaced `classification`, so the choice was made without the two fields ever being compared.

ii.
```python
# ---- Units ----
units = f['units']
unit_quality = np.array([x.decode() if isinstance(x, bytes) else str(x)
                        for x in units['unit_quality'][:]])
good_indices = np.where(unit_quality == 'good')[0]
if len(good_indices) == 0:
    print(f"  Skip: no good units")
    return None
```
Header: `- Neuron filtering: Only "good" units (classifier-based QC per the data paper).`

iii. The stated justification is that `unit_quality == 'good'` "matches the classifier-based QC from the paper" (repeated in the reasoning at step 34 and in the final summary table). No cross-check was made against the paper's published totals (69,943 good units = 25.9% of Kilosort2 clusters across 173 sessions), which are stated in the `methods.txt` the agent read.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. To **go-cue onset**, using `BehavioralEvents/go_start_times/timestamps[trial_idx]`. Spike times and event times share one session-absolute clock, so alignment is done by adding the trial's go-cue time to the fixed relative bin-edge grid and binning the raw spike times against those absolute edges — no resampling, interpolation or per-stream offset.

ii.
```python
go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
...
go_subset = go_times[trial_indices]
...
abs_edges = go_t + BIN_EDGES
edge_counts = np.searchsorted(st, abs_edges)
```

iii. Required by the instructions ("Temporally align based on Go cue onset"). The agent verified the clock assumption during exploration ("Spike times are in absolute time (session time); go cue times give us the alignment event") and cross-checked the derived timing (tone ≈1.85 s before go, photostim at −1.2 to −0.7 s) against `methods.txt`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins, 80 non-overlapping bins spanning −2.5 s to +1.5 s relative to the go cue. The edge/centre grid is computed once at module level and reused for every trial and session, so every trial has exactly 80 timepoints. Spikes are binned directly at 50 ms from raw spike times — there is no intermediate finer binning and hence no rebinning. `metadata['time_bin_size']` is recorded as 50.0 ms.

ii.
```python
TIME_BEFORE = 2.5
TIME_AFTER = 1.5
BIN_SIZE = 0.05
N_BINS = int(round((TIME_BEFORE + TIME_AFTER) / BIN_SIZE))  # 80
BIN_EDGES = np.linspace(-TIME_BEFORE, TIME_AFTER, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```
```python
'time_bin_size': BIN_SIZE * 1000,
'off_start': -TIME_BEFORE,
'off_end': TIME_AFTER,
```

iii. Directly specified by the instructions ("Extract 2.5 s before to 1.5 s after the go cue", "Use 50-ms-width bins"). The verifier confirmed a constant T of 80 for every session.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `acquisition/BehavioralEvents/sample_start_times/timestamps` (the sample-epoch/tone onsets) together with the trial's go-cue time. Because an early lick replays the sample epoch, a trial can have several sample onsets; the AI takes the **last sample start that falls between the previous trial's go cue and this trial's go cue**.

ii.
```python
sample_starts = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]

# For each trial, find the last sample_start before go cue
tone_onsets = np.full(n_trials, np.nan)
for i in range(n_trials):
    go_t = go_times[i]
    lower = go_times[i - 1] if i > 0 else 0.0
    mask_s = (sample_starts > lower) & (sample_starts < go_t)
    candidates = sample_starts[mask_s]
    if len(candidates) > 0:
        tone_onsets[i] = candidates[-1]
```

iii. Header: "Tone onset: last sample_start before each trial's go cue (the successful presentation that led to delay/go, not earlier replays from early licking)." The agent discovered the multiplicity empirically (405 sample events for 368 trials) and reasoned that the replayed epoch means only the final presentation precedes the go cue by the nominal sample+delay interval.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Continuous, time-varying: for each bin, the value is the bin centre (relative to the go cue) minus the tone's time relative to the go cue, i.e. seconds elapsed since the tone at that bin. Typical range is about −0.6 s (start of window, before the tone) to +3.35 s (end of window), consistent with the nominal 0.65 s sample + 1.2 s delay = 1.85 s tone-to-go interval; trials with replays give larger values. If no tone is found for a trial, the AI **falls back to assuming the nominal 1.85 s offset** (in practice this branch is not exercised — every trial in the sessions checked has a sample start between the previous and current go cue).

ii.
```python
tone_t = tone_onsets[trial_idx]
if np.isnan(tone_t):
    time_from_tone = BIN_CENTERS + 1.85
else:
    tone_rel = tone_t - go_t
    time_from_tone = BIN_CENTERS - tone_rel
```

iii. The agent sanity-checked the arithmetic against the task structure: "tone onset is about 1.85 s before the go cue, so at bin_center −2.5 relative to go, time from tone works out to −0.65, and at bin_center 1.5 it's 3.35, which confirms the range checks out." It also investigated sessions whose minimum was −1.5 instead of −0.6 and concluded these reflect real per-trial variation in the tone-to-go interval rather than a lookup error.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is defined *on* the neural bin grid: the same `BIN_CENTERS` vector (go-cue-relative bin centres) used to build the firing-rate bins is shifted by the tone-to-go offset, so element *k* of the input covers exactly the interval of element *k* of the firing-rate matrix. No separate alignment step or interpolation.

ii.
```python
BIN_EDGES = np.linspace(-TIME_BEFORE, TIME_AFTER, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
...
time_from_tone = BIN_CENTERS - tone_rel
inputs = np.stack([time_from_tone.astype(np.float32), photostim_vec], axis=0)
```

iii. N/A — a direct consequence of using one shared grid for all streams.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From the **event streams** `acquisition/BehavioralEvents/photostim_start_times/timestamps` and `photostim_stop_times/timestamps` (absolute session seconds) — not from the trials-table `photostim_onset`/`photostim_duration` columns (which the AI reads only for the control-trial definition in the session filter). If the group is missing or empty, the photostim input is zero everywhere.

ii.
```python
ps_start_abs = np.array([])
ps_stop_abs = np.array([])
if 'photostim_start_times' in f['acquisition/BehavioralEvents']:
    ps_ts = f['acquisition/BehavioralEvents/photostim_start_times/timestamps']
    if ps_ts.shape[0] > 0:
        ps_start_abs = ps_ts[:]
        ps_stop_abs = f['acquisition/BehavioralEvents/photostim_stop_times/timestamps'][:]
```

iii. Reasoning at step 36: "I could also use the photostim_start_times and photostim_stop_times events for finer precision", i.e. the event stream records the measured laser on/off times on the same clock as everything else, whereas the trials-table onset is a string offset from trial start. The agent also checked the six sessions whose filenames lack the `ogen` tag and concluded photostim should simply be zero there.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1) time series over the 80 bins: a bin is 1 if its centre falls in `[stim_start, stim_stop)` for any photostim event overlapping the trial window. Events entirely outside the window are skipped. Non-stimulated trials get an all-zero vector. Stored as `float32` and stacked as input row 1.

ii.
```python
photostim_vec = np.zeros(N_BINS, dtype=np.float32)
if len(ps_start_abs) > 0:
    win_start = go_t + BIN_EDGES[0]
    win_end = go_t + BIN_EDGES[-1]
    for ps_s, ps_e in zip(ps_start_abs, ps_stop_abs):
        if ps_e < win_start or ps_s > win_end:
            continue
        ps_s_rel = ps_s - go_t
        ps_e_rel = ps_e - go_t
        on_mask = (BIN_CENTERS >= ps_s_rel) & (BIN_CENTERS < ps_e_rel)
        photostim_vec[on_mask] = 1.0
```

iii. The instructions require "Whether photostimulation is on at every time point (discrete, time-varying)", so a per-bin binary series rather than a per-trial flag. The agent verified against `methods.txt` that silencing occupies the last 0.5 s of the delay including the 100 ms ramp-down and confirmed empirically that the events sit at −1.2 s to −0.7 s relative to the go cue.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The absolute event times are converted to go-cue-relative times by subtracting the trial's go cue, then compared against the same `BIN_CENTERS` grid used for the firing rates, so bin *k* of the photostim input covers the same interval as bin *k* of the neural matrix. Because photoinhibition always ends before the go cue, the on-bins fall in the delay portion of the window.

ii.
```python
ps_s_rel = ps_s - go_t
ps_e_rel = ps_e - go_t
on_mask = (BIN_CENTERS >= ps_s_rel) & (BIN_CENTERS < ps_e_rel)
```

iii. N/A — same shared grid as the neural data; no interpolation needed since all streams are on one clock.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no choice column, so it is derived from two trials-table columns: `trial_instruction` (`'left'`/`'right'`) and `outcome` (`'hit'`/`'miss'`/`'ignore'`). A hit means the animal licked the instructed side, a miss the opposite side, an ignore means no lick.

ii.
```python
instructions = np.array([x.decode() if isinstance(x, bytes) else str(x)
                        for x in trials['trial_instruction'][:]])
outcomes = np.array([x.decode() if isinstance(x, bytes) else str(x)
                    for x in trials['outcome'][:]])
```

iii. From the reasoning at step 34, the agent works through "the trial's choice (left/right/no-lick)" as a derived quantity from instruction × outcome, having found no direct choice field in the trials table during exploration.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded 0 = left, 1 = right, 2 = no lick, with an explicit third class for `ignore`. `hit` → instructed side; `miss` → opposite side; `ignore` (and any unexpected value) → 2. The per-trial scalar is broadcast across all 80 bins of output row 0, and `output_values[0] = ['left', 'right', 'no_lick']`.

ii.
```python
if outcome == 'ignore':
    choice = 2  # no lick
elif outcome == 'hit':
    choice = 0 if instruction == 'left' else 1
elif outcome == 'miss':
    choice = 1 if instruction == 'left' else 0
else:
    choice = 2

outputs = np.zeros((4, N_BINS), dtype=np.int64)
outputs[0, :] = choice
```

iii. Left = 0 / right = 1 follows the instruction's listing order, with a third class for the no-lick case required by the "left, right, no lick" specification. Broadcasting the per-trial value across bins keeps all four outputs in one `(n_output, n_timepoints)` array, as the target format allows.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the trials-table `outcome` column, which already stores `'hit'`, `'miss'`, `'ignore'`.

ii.
```python
outcomes = np.array([x.decode() if isinstance(x, bytes) else str(x)
                    for x in trials['outcome'][:]])
...
outcome = outcomes[trial_idx]
```

iii. No derivation needed — the exploration established that the trials table carries exactly the three categories the instructions ask for.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to integers with a fixed dictionary, **hit = 0, miss = 1, ignore = 2** (unknown values default to 2), written to output row 1 and repeated across all 80 bins. `output_values[1] = ['hit', 'miss', 'ignore']`, so the codes and names are mutually consistent even though the instruction listed the categories in the opposite order (ignore, miss, hit).

ii.
```python
outcome_val = {'hit': 0, 'miss': 1, 'ignore': 2}.get(outcome, 2)
...
outputs[1, :] = outcome_val
```
```python
'output_values': [
    ['left', 'right', 'no_lick'],
    ['hit', 'miss', 'ignore'],
    ['no', 'yes'],
    ['below_40pct', '40_to_60pct', 'above_60pct', 'not_visible'],
],
```

iii. Not separately justified beyond the format requirement that outputs be categorical with names supplied in `output_values`; the verifier's summary confirmed the label/proportion mapping (hit 0.742, miss 0.152, ignore 0.107) is sensible for this task.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. The trials-table `early_lick` column, whose values are the strings `'no early'` and `'early'` (the flag is set when the mouse licked during the sample or delay epoch, which triggers a replay).

ii.
```python
early_licks = np.array([x.decode() if isinstance(x, bytes) else str(x)
                       for x in trials['early_lick'][:]])
```

iii. The field is explicit in the trials table; the agent's exploration and the methods text both describe early licking as a per-trial condition, and the AI kept these trials specifically because early lick is a decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Binary coding: `'no early'` → 0, anything else → 1, written to output row 2 and repeated across all 80 bins. `output_values[2] = ['no', 'yes']`.

ii.
```python
early_val = 0 if early_licks[trial_idx] == 'no early' else 1
...
outputs[2, :] = early_val
```

iii. Follows the instruction's "Early lick (no, yes, per-trial)" ordering. Per-trial value broadcast across bins like the other categorical outputs; the triggering lick occurs before the go cue so the evidence lies inside the −2.5 s portion of the window.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is `(n_frames, 3)` = (tongue_x, tongue_y, DLC likelihood) sampled at ~294 Hz with matching `timestamps`. Column 1 is the y-position, column 2 the tracking confidence used to decide visibility. If the series is absent the whole output is set to the "not visible" class.

ii.
```python
if 'Camera0_side_TongueTracking' in f['acquisition/BehavioralTimeSeries']:
    tt = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
    tongue_ts = tt['timestamps'][:]
    tdata = tt['data'][:]
    tongue_y = tdata[:, 1]
    tongue_conf = tdata[:, 2]
```

iii. The exploration identified this series as the DLC side-view tongue tracking (x, y, confidence) and measured that the tongue is confidently tracked only ~10.5% of the time: "The tongue is only visible about 10.5% of the time (confidence > 0.9). This makes sense — the tongue is only visible during licking."

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Per session: frames with DLC likelihood > 0.9 are treated as "tongue visible", and the 40th/60th percentiles of the y-values of **those raw visible frames** define the class edges. Per trial and bin, the AI does **not** average the frames inside the bin — it takes the single frame **nearest the bin centre** (accepting it only if it lies within 5 ms, about 1.5 frame periods, of the centre) and classifies that one sample. Bins whose nearest frame is far away (video not running) or not confident are assigned class 3. No smoothing or interpolation. Resulting distribution over all bins: 0.065 / 0.033 / 0.069 / 0.833.

ii.
```python
vis = tongue_conf > DLC_CONFIDENCE_THRESHOLD           # 0.9
if np.any(vis):
    tongue_y_p40 = np.percentile(tongue_y[vis], 40)
    tongue_y_p60 = np.percentile(tongue_y[vis], 60)
```
```python
t_abs_all = go_t + BIN_CENTERS
idx_all = np.clip(np.searchsorted(tongue_ts, t_abs_all), 0, len(tongue_ts) - 1)
idx_prev = np.clip(idx_all - 1, 0, len(tongue_ts) - 1)
dist_cur = np.abs(tongue_ts[idx_all] - t_abs_all)
dist_prev = np.abs(tongue_ts[idx_prev] - t_abs_all)
best_idx = np.where(dist_prev < dist_cur, idx_prev, idx_all)
best_dist = np.minimum(dist_cur, dist_prev)

# Within one frame period (~3.4ms) and confident
close_enough = best_dist < 0.005
confident = tongue_conf[best_idx] > DLC_CONFIDENCE_THRESHOLD
valid = close_enough & confident
```

iii. Header: "Tongue y-position: discretized per session using DLC confidence > 0.9 for visibility, percentiles computed over all visible positions in the session"; the final summary calls 0.9 the "standard DLC confidence threshold". The nearest-frame scheme was adopted when the agent rewrote the first (per-bin `searchsorted`) implementation for speed: "let me also optimize the tongue y-position computation — the current approach is very slow… Let me vectorize it." The 5 ms tolerance is tied to the measured ~3.4 ms frame period so that a bin only counts as unobserved when the camera really has no nearby frame.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four classes, per-session thresholds: 0 = y below the session's 40th percentile, 1 = between the 40th and 60th, 2 = above the 60th, 3 = not visible (low confidence or no nearby frame). Percentiles are computed once per session over the *visible frames only* (not over bins, and not including retracted-tongue frames). Comparisons are `y < p40`, `p40 <= y < p60`, `y >= p60`.

ii.
```python
tongue_y_p40 = np.percentile(tongue_y[vis], 40)
tongue_y_p60 = np.percentile(tongue_y[vis], 60)
...
tongue_y_disc = np.full(N_BINS, 3, dtype=np.int64)
tongue_y_disc[valid & (y_vals < tongue_y_p40)] = 0
tongue_y_disc[valid & (y_vals >= tongue_y_p40) & (y_vals < tongue_y_p60)] = 1
tongue_y_disc[valid & (y_vals >= tongue_y_p60)] = 2
```

iii. The 40/60 split and the per-session scope are given by the instructions ("per-session discretization: <40th percentile … 40th to 60th … >60th … not visible"). Restricting the percentile base to visible frames follows from the agent's finding that the tracker still emits a position when the tongue is retracted, so including those frames would define the percentiles on non-protrusion noise.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps are on the same session-absolute clock as spikes and events, so each bin's absolute centre time (`go_t + BIN_CENTERS`) is looked up in the camera timestamps by `searchsorted`, with both neighbours compared to pick the nearest frame. The output therefore sits on exactly the same go-cue-relative 80-bin grid as the firing rates. When the video is off (e.g. the inter-trial interval, or when the go cue is less than 2.5 s after trial start) no frame is within 5 ms and the bin becomes class 3.

ii.
```python
t_abs_all = go_t + BIN_CENTERS
idx_all = np.searchsorted(tongue_ts, t_abs_all)
...
best_idx = np.where(dist_prev < dist_cur, idx_prev, idx_all)
best_dist = np.minimum(dist_cur, dist_prev)
close_enough = best_dist < 0.005
```

iii. No offset correction is needed because everything shares the NWB global clock; the explicit distance test makes the "no data" case detectable rather than silently snapping to a distant frame.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Five cases:
- **Trials with no spike data** (recording started late or ended early): excluded by the min/max spike-time range test (see 1-e); added only after the format checker flagged all-zero trials.
- **`auto_water` / `free_water` trials**: excluded as non-behavioural.
- **Units without a CCF annotation** (`anno_name` empty or `'nan'`, ≈75% of units): fall back to the probe's target region parsed from the electrode `location` JSON; annotations not present in the lookup table also fall back.
- **Frames with no confidently tracked tongue, or bins with no nearby frame**: assigned the explicit `not_visible` class instead of being imputed; a session with no tongue series at all gets class 3 everywhere.
- **Trials with no locatable tone onset**: silently imputed with the nominal 1.85 s tone-to-go offset (this branch appears not to trigger in practice).
- Sessions with no `'good'` unit or fewer than 2 usable trials return `None` and are dropped; no such drop occurred, including for the one session whose QC classifier labels are entirely missing (it retains 1,201 `unit_quality == 'good'` units and was kept).

ii.
```python
def get_unit_brain_region(anno_name, elec_target):
    if anno_name and anno_name != '' and str(anno_name) != 'nan':
        base = str(anno_name).split(',')[0].strip()
        if base in ANNO_TO_REGION:
            return ANNO_TO_REGION[base]
    return elec_target
```
```python
tongue_y_disc = np.full(N_BINS, 3, dtype=np.int64)
if tongue_ts is not None and tongue_y_p40 is not None:
    ...
```
```python
if np.isnan(tone_t):
    time_from_tone = BIN_CENTERS + 1.85
```

iii. The missing-neural-data handling is the best documented decision in the trajectory (traced from warning → diagnosis → two successive fixes → clean verifier run). The region fallback is justified by the agent's observation that only ~459/1,952 units carry a CCF annotation while every unit has an electrode with a probe-level target: "use anno_name when it's available and fall back to the electrode group's target region only when anno_name is missing". The tongue class 3 exists because the tongue is genuinely absent most of the time, so an explicit category is more faithful than imputation.

## 10-a. What are the most time-consuming steps of the code?

i. Reading each NWB file (the full `spike_times` buffer — up to ~11.5 M doubles — and the ~680 k × 3 tongue array) and the spike-binning loop, which performs one `np.searchsorted` call per (unit × trial): ~1,000 units × ~500 trials ≈ 500 k calls per session. After the optimisation pass the whole conversion ran at ~2.5 s/session (~7–8 minutes for 174 files); the initial version, which called `np.histogram` per unit per trial, was on track for 5–6 hours and was killed. Writing the 21.8 GB pickle is the other large cost.

ii.
```python
    for i, st in enumerate(unit_spikes):
        if len(st) == 0:
            continue
        for t_idx in range(n_trials):
            go_t = go_subset[t_idx]
            abs_edges = go_t + BIN_EDGES
            edge_counts = np.searchsorted(st, abs_edges)
            rates_all[i, t_idx, :] = np.diff(edge_counts) / BIN_SIZE
```

iii. From the trajectory: "The real bottleneck is the nested loop computing histograms per trial per neuron — with 1000+ neurons and 400+ trials that's 400K+ operations per session, and across 174 sessions at 2-3 minutes each this would take 5-6 hours total, which is far too slow." The fix replaced per-trial `np.histogram` with per-trial `np.searchsorted` on the unit's sorted spike train, giving a ~60× speed-up.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several remain:
- **The inner trial loop in `compute_firing_rates_all_trials`.** All 81 × n_trials edges could have been flattened into a single sorted array and passed to one `searchsorted` per unit (edges for successive trials are monotonically increasing), collapsing ~500 k calls per session to ~1 k. This is the largest remaining inefficiency.
- **The per-trial tone-onset loop**, which builds a full boolean mask over all sample-start times for every trial (O(n_trials × n_samples)); one `np.searchsorted(sample_starts, go_times) - 1` would do the whole session at once.
- **The per-trial photostim loop**, which rescans every photostim event in the session for every trial; the events could be matched to trials with one `searchsorted`.
- **The min/max spike-time loop** over all good units, and the Python `for ti in trial_indices` list comprehension that builds `valid_trial_mask`.
- **The per-trial input/output construction loop**, which builds 80-element vectors one trial at a time instead of as `(n_trials, 80)` arrays.

ii.
```python
        for t_idx in range(n_trials):
            go_t = go_subset[t_idx]
            abs_edges = go_t + BIN_EDGES
            edge_counts = np.searchsorted(st, abs_edges)
```
```python
        for i in range(n_trials):
            go_t = go_times[i]
            lower = go_times[i - 1] if i > 0 else 0.0
            mask_s = (sample_starts > lower) & (sample_starts < go_t)
```
```python
                for ps_s, ps_e in zip(ps_start_abs, ps_stop_abs):
                    if ps_e < win_start or ps_s > win_end:
                        continue
```

iii. The agent's stopping criterion was wall-clock adequacy, not full vectorisation: after reaching 2.5 s/session it declared "much better. Full conversion should take about 7-8 minutes" and moved on. The tongue path *was* vectorised across bins for exactly this reason, so the choice of what to optimise was deliberate rather than accidental, but the trial loop inside the binning was left in place.

## 10-c. What processing does the code repeat multiple times?

i. Each NWB file is opened once and most quantities are computed once, but a few things are computed more often than necessary:
- **Spike data is traversed twice**: once to find the min/max spike time per good unit for the trial-range filter, then again to bin. Both walk the same per-unit slices.
- **`tone_onsets` is computed for all `n_trials`**, including trials already excluded by the auto/free-water mask and the ones later dropped by the range filter.
- **The photostim event list is rescanned for every trial**, so the same events are examined n_trials times.
- **String columns are decoded element-by-element in Python** for the whole table (outcome, early_lick, instruction, photostim_onset, anno_name, electrode location), including rows/units that are subsequently dropped.
- The electrode→region lookup *is* memoised (`elec_region_cache`), which avoids repeated JSON parsing.

ii.
```python
        for uid in good_indices:
            start = spike_times_index[uid - 1] if uid > 0 else 0
            end = spike_times_index[uid]
            if end > start:
                min_spike_time = min(min_spike_time, all_spike_times[start])
                max_spike_time = max(max_spike_time, all_spike_times[end - 1])
```
```python
        elec_region_cache = {}
        for uid in good_indices:
            elec_idx = int(electrodes_ref[uid])
            if elec_idx not in elec_region_cache:
                side, region = get_electrode_target_region(elec_locations[elec_idx])
                elec_region_cache[elec_idx] = (side, region)
```

iii. Not discussed in the trajectory; these are consequences of the incremental development (the recording-range scan was bolted on after the binning code already existed, so it became a second pass rather than being folded into the existing one).

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Minor waste only:
- `get_electrode_target_region` parses and returns the **hemisphere** (`side`) of every probe, which is immediately discarded (`_, target_region = ...`); the agent had considered hemisphere-specific region names and then dropped the idea.
- Tone onsets are computed for trials that are later filtered out (see 10-c).
- Outputs are stored as **`int64`** although every value is in 0–3, and `input`/`output` arrays are built per trial rather than sliced from a session array; the neural views are `float32` but the resulting pickle is 21.8 GB.
- `metadata['bin_centers']` duplicates information already implied by `time_bin_size`, `off_start`, `off_end`.
- Local variables `n_neurons`, `n_trials` in `compute_firing_rates_all_trials` and `t_i` in the trial loop are computed but unused.
- Nothing substantive is computed and thrown away: every stream requested by the instructions ends up in the output dictionary, and no extra behavioural streams (licks, jaw/nose tracking, delay/trial-end events) are read.

ii.
```python
def get_electrode_target_region(location_json):
    ...
    side = parts[0]
    region_name = ' '.join(parts[1:])
    major = TARGET_TO_REGION.get(region_name, region_name)
    return side, major
```
```python
            _, target_region = elec_region_cache[elec_idx]
```
```python
outputs = np.zeros((4, N_BINS), dtype=np.int64)
```
```python
'bin_centers': BIN_CENTERS.tolist(),
```

iii. Not discussed in the trajectory; the agent's stated optimisation goals were runtime and correctness of the format check, not output size, and it never revisited the dtype of the output arrays after the format verifier passed.
