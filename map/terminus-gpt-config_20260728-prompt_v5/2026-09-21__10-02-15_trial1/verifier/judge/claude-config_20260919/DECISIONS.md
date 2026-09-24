# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is one NWB file per session under `/app/data/sub-<subject_id>/`. The AI discovers every session with a single recursive glob over that tree and sorts the result, so session order is deterministic. Unlike the reference, the AI does **not** use `pynwb`; it opens each file directly with `h5py` and reads the raw HDF5 groups (`units`, `intervals/trials`, `acquisition/BehavioralEvents`, `acquisition/BehavioralTimeSeries`). Every file is opened exactly once inside a `with` block, and all per-session quantities (units, trials, events, video) are pulled out in that one pass. String columns are byte-decoded through a helper. Sessions are dropped after processing if they yield fewer than 2 trials or zero units.

ii.
```python
files = sorted(DATA_DIR.rglob('*.nwb'))
if args.sample:
    files = files[:2]
...
for i, f in enumerate(files):
    print(f'processing {i+1}/{len(files)}: {f}', flush=True)
    neural, inp, out, sess_regions, subj, info = session_from_file(f, make_plot=show_processing and i < 2)
```

```python
def session_from_file(path, make_plot=False):
    t0 = time.time()
    with h5py.File(path, 'r') as h:
        units = h['units']
        trials = h['intervals']['trials']
        acq = h['acquisition']
        ...
        beh_events = acq['BehavioralEvents']
        beh_ts = acq['BehavioralTimeSeries']
```

```python
def _decode_arr(arr):
    out = []
    for v in arr:
        if isinstance(v, (bytes, np.bytes_)):
            out.append(v.decode())
        else:
            out.append(str(v))
    return np.array(out, dtype=object)
```

iii. From CONVERSION_NOTES Step 2: "`/app/data` contains NWB files organized by subject folders: `sub-<subject_id>/sub-<subject_id>_ses-<timestamp>_behavior+ecephys+ogen.nwb`. Each NWB session file contains an `intervals/trials` table and a `units` table." The AI treated the directory listing as the complete set of sessions and reported 28 subjects / 174 files / 272,227 units / 94,990 trials, which it cross-checked in Step 4 against the papers' 173 sessions and 69,943 good units. The trajectory shows `h5py` was chosen because the AI explored the file layout with `h5py` from the start (Step 2/Step 5 inspection commands) and never introduced `pynwb`; no explicit justification for avoiding `pynwb` is recorded.

## 1-b. How are the data split into subjects?

i. The subject is taken from the containing directory name, `sub-440956` → `'440956'`, rather than from the NWB `subject/subject_id` field. `subjects` is built as a list of unique ids in first-encounter order (i.e. sorted order, because the file list is sorted), and `subject_idx` is the index of each session's subject in that list. This yields the same 28 ids as the reference.

ii.
```python
subject = path.parent.name.replace('sub-', '')
```

```python
if subj not in subjects:
    subjects.append(subj)
subject_idx.append(subjects.index(subj))
...
'subjects': subjects,
'subject_idx': np.asarray(subject_idx, dtype=np.int64),
```

iii. CONVERSION_NOTES Step 5 mapping table: "`subject.subject_id` / folder name → subjects, subject_idx | Collect unique subject IDs and session-to-subject mapping | NWB metadata | One subject per NWB folder." The AI observed that the dataset layout puts exactly one subject per folder, so the folder name is a sufficient and unambiguous animal identifier.

## 1-c. How are the data split into sessions?

i. One NWB file is one session; no grouping or splitting is performed. Session order follows the sorted glob, which (because the filename embeds the acquisition timestamp) is chronological within each subject. Each session's identity is recorded in `metadata['session_info']` as the full file path plus subject, raw trial count, good-unit count and load time. A session is dropped from the output if it produced fewer than 2 usable trials or zero good units; exactly one of the 174 files is dropped this way, leaving 173 sessions.

ii.
```python
files = sorted(DATA_DIR.rglob('*.nwb'))
```

```python
session_info = {
    'file': str(path),
    'subject': subject,
    'n_trials': n_trials,
    'n_good_units': len(good_inds),
    'load_time_sec': time.time() - t0,
}
```

```python
if len(neural) < 2 or neural[0].shape[0] == 0:
    print(f'skipping {f} due to insufficient trials or neurons', flush=True)
    continue
```

iii. CONVERSION_NOTES Step 4 records the discrepancy explicitly: "Local NWB release contains 174 NWB session files ... Papers report 173 behavioral sessions ... Treat local NWB as raw/native release; expect one-session difference or inclusion mismatch and verify downstream filtered set." The 2-trial minimum is the target format's stated requirement ("There needs to be at least two trials within each session").

## 1-d. Are the data correctly split into trials?

i. Trials come from the NWB trials table in `intervals/trials`, one row per behavioural trial. The AI takes the number of trials from the length of `go_start_times` and then slices every trial column to `[:n_trials]`, i.e. it assumes one go cue per trial row and that the two are in the same order. No assertion is made that the two lengths agree (the reference asserts this); a mismatch would be silently truncated rather than raised. In practice `len(go) == len(trials)` in all 174 files, so the mapping is correct.

ii.
```python
go_times = np.asarray(beh_events['go_start_times']['timestamps'][()]).astype(float)
...
n_trials = len(go_times)
trial_instruction = _decode_arr(trials['trial_instruction'][()])[:n_trials]
outcome = _decode_arr(trials['outcome'][()])[:n_trials]
early_lick = _decode_arr(trials['early_lick'][()])[:n_trials]
```

```python
for tr in range(n_trials):
    if not valid_trial_mask[tr]:
        continue
    gt = go_times[tr]
```

iii. CONVERSION_NOTES Step 5 maps "`BehavioralEvents/go_start_times`" as the per-trial alignment event and the trial table as the source of the per-trial labels. The AI noted in Step 2 that the trial table "fields observed include: `start_time`, `stop_time`, `trial`, `photostim_onset`, ... `outcome`, `auto_water`, `free_water`", and that there is no explicit go-cue column, so the event stream must be joined to the table positionally.

## 1-e. How are trials filtered based on quality controls?

i. Two filters, both aimed at absence of spike data rather than at behaviour:

1. **`units/is_good_trials`**: a `(n_units, n_observed_trials)` boolean matrix. The AI restricts it to the good units, takes the per-trial mean, and keeps a trial if *any* good unit flags it good (`frac > 0`). Crucially, `is_good_trials` has *fewer* columns than go cues in 9 of 174 sessions (e.g. 160 columns for 480 trials), and the AI maps those columns onto the *first* `ncol` trials, marking all later trials invalid. I verified against `units/obs_intervals` that the observed trials are indeed the leading ones, so this mapping is correct.
2. **All-zero neural drop**: after building a trial's firing-rate matrix, the trial is discarded if every bin of every neuron is zero. This is what removes the `free_water` trials (which the reference excludes explicitly by column) and any residual unobserved trials.

No behavioural filter is applied — early-lick and `ignore` trials are deliberately retained because they are required decoder outputs. 90,734 of 94,990 trials survive (reference: 90,860).

ii.
```python
valid_trial_mask = np.ones(n_trials, dtype=bool)
if 'is_good_trials' in units:
    igt = np.asarray(units['is_good_trials'][()])
    n_valid_cols = min(igt.shape[1], n_trials)
    trial_good_frac = igt[good_inds, :n_valid_cols].mean(axis=0) if len(good_inds) else np.zeros(n_valid_cols)
    valid_trial_mask[:] = False
    valid_trial_mask[:n_valid_cols] = trial_good_frac > 0
```

```python
for tr in range(n_trials):
    if not valid_trial_mask[tr]:
        continue
    ...
    if np.all(neural == 0):
        continue
    neural_trials.append(neural)
```

iii. CONVERSION_NOTES Step 5 Key Decision 4: "**Trial inclusion**: Retain all trials with valid go cue and sufficient data for the requested outputs, rather than only regular control trials, because the decoder must predict photostimulation and behavioral variables across the dataset." Step 10 records the two debugging iterations that produced the current filter: "Many late trials in some sessions had all-zero neural data: discovered that raw NWB `is_good_trials` covered only a subset of trials (e.g. 160 valid neural trials for a 480-go-cue session). Resolved by filtering to valid trial columns and dropping any residual all-zero neural trials."

## 2-a. What variables in the raw data is the final `neural` data derived from?

i. `units/spike_times` (a flat float64 buffer) together with `units/spike_times_index` (the ragged end-offsets), restricted to units whose `units/classification == 'good'`. The go-cue timestamps (`BehavioralEvents/go_start_times`) supply the per-trial alignment point. A helper handles both the flat-plus-index layout and a hypothetical object-array layout; spike trains for all good units are read once per session and cached in a list.

ii.
```python
def get_unit_spike_times(units_group, unit_index):
    st_ds = units_group['spike_times']
    try:
        x = st_ds[unit_index]
        arr = np.asarray(x, dtype=float)
        if arr.ndim >= 1 and arr.size > 0:
            return arr.ravel()
    except Exception:
        pass
    st_all = st_ds[()]
    if getattr(st_all, 'dtype', None) == object:
        return np.asarray(st_all[unit_index], dtype=float).ravel()
    idx = np.asarray(units_group['spike_times_index'][()])
    start = 0 if unit_index == 0 else idx[unit_index - 1]
    end = idx[unit_index]
    return np.asarray(st_all[start:end], dtype=float).ravel()
```

```python
good_inds = np.flatnonzero(good_mask)
good_spike_times = [get_unit_spike_times(units, int(ui)) for ui in good_inds]
```

iii. CONVERSION_NOTES Step 5 mapping: "`units/spike_times` for units with `classification == 'good'` → neural | Bin spike times into 50 ms spike counts or rates over [-2.5, +1.5] s relative to `BehavioralEvents/go_start_times`". Spike times are the only neural representation in the file. The trajectory (step 779-780) shows the robust helper was added while chasing all-zero trials: "we confirmed that in the problematic session `units['spike_times']` is a flat float64 array with `spike_times_index`, so our original slicing logic should have worked there. However, the newly added helper is more robust and harmless."

## 2-b. How is the `neural` data processed?

i. For each trial and each good unit, the unit's spike times are shifted by that trial's go cue and passed to `np.histogram` against the fixed 81-edge grid. The resulting counts are divided by the 50 ms bin width to give firing rates in Hz, stored as `float32`. No smoothing, normalisation, or baseline subtraction. I independently reproduced the stored rates from the raw HDF5 buffers for a spot-checked (session, trial, neuron) and they agree exactly.

ii.
```python
BIN_SIZE = 0.05
BIN_EDGES = np.linspace(WIN_START, WIN_END, N_BINS + 1)
```

```python
neural = np.zeros((len(good_inds), N_BINS), dtype=np.float32)
for j, st in enumerate(good_spike_times):
    rel = st - gt
    counts, _ = np.histogram(rel, bins=BIN_EDGES)
    neural[j] = counts.astype(np.float32) / BIN_SIZE
```

iii. CONVERSION_NOTES Step 5 Key Decision 3: "**Binning**: Use 50 ms bins from -2.5 s to +1.5 s around go cue, overriding the paper code's finer 3.4 ms stride / 40 ms window because the decoder task specifies this." Step 1 identifies `sliding_histogram` in the reference repo as the "core firing-rate computation" that returns rates rather than counts, and the AI mirrors that by dividing counts by the bin width.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == 'good'` are kept; no individual QC metric (isi_violation, presence_ratio, amplitude_cutoff, …) is thresholded, and `unit_quality` is not used. If the `classification` column were missing entirely the code would fall back to keeping all units, but that path is never taken. This retains 69,453 of 272,227 units (mean 401.5 per session), identical to the reference. The one session whose `classification` is NaN for every unit ends up with zero good units and is dropped by the `neural[0].shape[0] == 0` guard.

ii.
```python
unit_class = _decode_arr(units['classification'][()]) if 'classification' in units else None
good_mask = unit_class == 'good' if unit_class is not None else np.ones(len(units['id']), dtype=bool)
```

```python
if len(neural) < 2 or neural[0].shape[0] == 0:
    print(f'skipping {f} due to insufficient trials or neurons', flush=True)
    continue
```

iii. CONVERSION_NOTES Step 5 Key Decision 1: "**Neuron curation**: Use `classification == 'good'` as the primary neuron filter because this reproduces the paper's reported good-unit count (~69.5k vs 69.9k reported) and matches `qc_mode='classifier'` in reference code." Step 1 records that the reference batch script `Sherlock/preprocess_all_ephys.py` runs with `qc_mode='classifier'`, and Step 3 that the white paper describes "region-specific logistic-regression classifiers trained on 15 QC metrics to label units as good vs unlabeled."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All NWB streams are on one session-absolute clock, so alignment is a subtraction: for each trial, each good unit's spike times are re-expressed relative to that trial's go cue (`rel = st - gt`) before being histogrammed on the fixed relative grid. No resampling, interpolation, or per-stream offset.

ii.
```python
go_times = np.asarray(beh_events['go_start_times']['timestamps'][()]).astype(float)
...
gt = go_times[tr]
abs_edges = gt + BIN_EDGES
abs_centers = gt + BIN_CENTERS
...
rel = st - gt
counts, _ = np.histogram(rel, bins=BIN_EDGES)
```

iii. CONVERSION_NOTES Step 5 Key Decision 2: "**Temporal alignment**: Align all trials to `BehavioralEvents/go_start_times` because the decoder task explicitly requires go-cue alignment." Step 3 also notes the task-relevant reason this matters: "Photoinhibition ended before the Go cue; this is important for go-cue alignment and photostim input construction."

## 2-e. How is the `neural` data temporally binned/resampled?

i. 80 non-overlapping 50 ms bins spanning -2.5 s to +1.5 s relative to the go cue. The 81 edges and 80 centers are built once at module scope from `np.linspace` and reused for every trial and every session, so every trial has exactly 80 timepoints. The verification output confirms `T: mean 80.00, min 80, max 80`. This is a deliberate re-binning relative to the reference pipeline, which used a 40 ms window with a 3.4 ms stride over [-3, +3] s.

ii.
```python
BIN_SIZE = 0.05
WIN_START = -2.5
WIN_END = 1.5
N_BINS = int(round((WIN_END - WIN_START) / BIN_SIZE))
BIN_EDGES = np.linspace(WIN_START, WIN_END, N_BINS + 1)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
```

```python
'time_bin_size': BIN_SIZE * 1000.0,
'off_start': WIN_START,
'off_end': WIN_END,
'n_timepoints': N_BINS,
```

iii. CONVERSION_NOTES Step 3: "For our conversion, decoder alignment must be to Go cue with a [-2.5 s, +1.5 s] window and 50 ms bins, which differs from some reference analysis windows but should preserve source event timing and curation logic where applicable." Step 5 Key Decision 3 repeats that the task specification overrides the reference binning.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `BehavioralEvents/sample_start_times` (the onset of the sample/tone epoch) together with the trial's go cue. For each trial the AI takes the **last** sample onset before the go cue, restricted to a window of 0.05 s to 5.0 s before it. If no sample onset falls in that window, a fixed fallback of `go - 0.6 s` is used.

ii.
```python
sample_times = np.asarray(beh_events['sample_start_times']['timestamps'][()]).astype(float) if 'sample_start_times' in beh_events else None
```

```python
def infer_trial_event_times(event_ts, go_times, default_offset=-0.6, min_pre=0.05, max_pre=5.0):
    out = np.full(go_times.shape, np.nan, dtype=float)
    if event_ts is None or len(event_ts) == 0:
        return go_times + default_offset
    event_ts = np.asarray(event_ts, dtype=float)
    for i, gt in enumerate(go_times):
        cand = event_ts[(event_ts <= gt - min_pre) & (event_ts >= gt - max_pre)]
        if cand.size:
            out[i] = cand[-1]
        else:
            out[i] = gt + default_offset
    return out
```

iii. CONVERSION_NOTES Step 5 mapping: "`BehavioralEvents/sample_start_times` (tone/sample onset) and go cue times → input[0] | Continuous time-from-tone-onset per bin: for each trial/bin, `bin_center_abs_time - sample_onset_time`". The AI identified in Step 5 that `sample_start_times` does not have one entry per trial, so a per-trial nearest-preceding match is needed rather than a positional join.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A single subtraction: the absolute time of each bin center minus the trial's tone onset, producing a continuous, monotonically increasing ramp of 80 values per trial. The per-trial matching itself is a Python loop with boolean masking over all of the session's sample onsets. Two guards were added during Step 10 because the first full run produced NaN/Inf inputs: a `[0.05, 5.0]` s window on which tone counts, and the `go - 0.6 s` fallback plus a second `np.isfinite` check at use site. Empirically the 5 s clamp / fallback changes the answer on ~0.06% of trials (I measured 2 of 3,240 trials over 6 randomly sampled sessions); the reference's unclamped "last tone before go" rule gives gaps up to 5.7 s. The realised range in the full dataset is [-1.9, 6.5] s.

ii.
```python
if sample_times is None or len(sample_times) != n_trials:
    sample_times = infer_trial_event_times(sample_times, go_times)
else:
    sample_times = infer_trial_event_times(sample_times[:], go_times)
```

```python
stime = sample_times[tr] if np.isfinite(sample_times[tr]) else (gt - 0.6)
time_from_tone = abs_centers - stime
```

```python
inp = np.vstack([time_from_tone.astype(np.float32), photostim_on.astype(np.float32)])
```

iii. CONVERSION_NOTES Step 10, Issues Found and Resolved: "Missing tone/sample onset for some trials caused NaN input values: replaced NaN with robust nearest-preceding sample-onset matching and finite fallback (~0.6 s before go cue)." Step 7 comments on the resulting range: "the reported input range for `time_from_tone_onset_sec` is [-0.6, 5.7], indicating that sample onset is often 0.6 s before the go cue and time-from-tone continues increasing through the trial; that's plausible." No justification is given for the specific values 0.05, 5.0 and 0.6.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed on exactly the grid used to bin the spikes. `abs_centers = gt + BIN_CENTERS` is the absolute time of the center of each of the 80 neural bins, so `abs_centers - stime` is by construction the elapsed time from the tone at the center of neural bin *k*. No separate alignment or interpolation step exists.

ii.
```python
gt = go_times[tr]
abs_edges = gt + BIN_EDGES
abs_centers = gt + BIN_CENTERS
...
time_from_tone = abs_centers - stime
```

iii. Implicit in Step 5's mapping ("for each trial/bin, `bin_center_abs_time - sample_onset_time`"): because the go cue defines the bin grid and everything is on the same session clock, sharing `abs_centers` between the neural and input construction guarantees bin-for-bin correspondence.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Primary source: the trials-table columns `photostim_onset` and `photostim_duration` (both stored as strings, with `'N/A'` on unstimulated trials). Secondary/fallback source: the `BehavioralEvents/photostim_start_times` and `photostim_stop_times` event streams, used only for trials whose trial-table entry is `'N/A'`. These are the same variables the reference uses; what the AI does *not* do is bring in `trials/start_time`, which is required because `photostim_onset` is measured from trial start, not from the session clock or the go cue.

ii.
```python
photostim_start = np.asarray(beh_events['photostim_start_times']['timestamps'][()]).astype(float) if 'photostim_start_times' in beh_events else np.array([])
photostim_stop = np.asarray(beh_events['photostim_stop_times']['timestamps'][()]).astype(float) if 'photostim_stop_times' in beh_events else np.array([])
```

```python
if 'photostim_duration' in trials:
    pdur = _decode_arr(trials['photostim_duration'][()])[:n_trials]
    pon = _decode_arr(trials['photostim_onset'][()])[:n_trials] if 'photostim_onset' in trials else np.array(['N/A'] * n_trials, dtype=object)
else:
    pdur = np.array(['N/A'] * n_trials, dtype=object)
    pon = np.array(['N/A'] * n_trials, dtype=object)
```

iii. CONVERSION_NOTES Step 5 mapping: "`BehavioralEvents/photostim_start_times` and `photostim_stop_times` or trial photostim fields → input[1] | Binary time-varying photostim on/off per bin within each aligned trial window | ... Need to handle sessions/trials with `N/A` in trial table by using event timeseries." The trajectory (step 51) records the reason for the dual source: "We also observed a filename without `+ogen` ... which confirms some sessions may lack optogenetics; our script should handle those because it falls back to zeros if photostim events are absent."

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The intent is a binary per-bin indicator: 1 where the bin center falls inside `[onset, onset + duration)`, 0 elsewhere, stored as `float32`. As implemented it does not work. `abs_centers` are **session-absolute** times (`go_time + centers`, i.e. tens to thousands of seconds into the session), while `float(pon[tr])` is an offset of ~2.5 s **from trial start**. The comparison `abs_centers >= pstart` is therefore trivially true and `abs_centers < pstart + pd` trivially false for every trial past the first few seconds of a session, so the channel is identically zero. The full-dataset verification confirms this: `photostimulation_on: [0.0, 0.0]`. By the AI's own Step 4 measurement ~19.6% of trials are stimulated, so roughly a fifth of trials lose their stimulation signal entirely. The `'N/A'` fallback branch is dead code and would raise `TypeError` if reached, since `|=` is applied to a `float32` array with a boolean operand.

ii.
```python
photostim_on = np.zeros(N_BINS, dtype=np.float32)
if pon[tr] != 'N/A' and pdur[tr] != 'N/A':
    try:
        pstart = float(pon[tr])
        pd = float(pdur[tr])
        photostim_on = ((abs_centers >= pstart) & (abs_centers < pstart + pd)).astype(np.float32)
    except Exception:
        pass
else:
    for ps, pe in zip(photostim_start, photostim_stop):
        if pe >= abs_edges[0] and ps <= abs_edges[-1]:
            photostim_on |= ((abs_centers >= ps) & (abs_centers < pe))
    photostim_on = photostim_on.astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 mapping: "Binary time-varying photostim on/off per bin within each aligned trial window", consistent with the target format's rule that time-like inputs be represented as binary series. The AI noticed the all-zero channel in the two-session sample (Step 7: "Sample sessions had no photostimulation trials, so input channel 1 was all zeros", and trajectory step 30: "Photostimulation input is all zeros in these two sample sessions, which is acceptable but means the sample is not representative for that variable") and attributed it to sampling rather than to a bug. It never revisited the channel after the full run, even though the Step 12 review was supposed to check input ranges.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is evaluated on the same 80 bin centers as the neural data, but in the wrong frame of reference: absolute bin-center times are compared against a trial-relative onset. The correct expression is `trial_start + photostim_onset` (absolute) or, equivalently, `trial_start + photostim_onset - go` compared against the relative `BIN_CENTERS`. The `trials/start_time` column is never read anywhere in the script. I confirmed on `sub-480927_ses-20210218T122535` that `photostim_onset` values are ~2.51-2.64 s with `photostim_duration` 0.5 s, and that `start_time + onset - go` lands at -0.50 s — i.e. the light terminates exactly at the go cue, inside the [-2.5, 1.5] s window, so correctly aligned stimulation would be clearly visible in the pre-go bins.

ii.
```python
abs_centers = gt + BIN_CENTERS
...
pstart = float(pon[tr])
pd = float(pdur[tr])
photostim_on = ((abs_centers >= pstart) & (abs_centers < pstart + pd)).astype(np.float32)
```

iii. No justification is recorded. The AI's notes describe the intended alignment ("per bin within each aligned trial window") but the units mismatch is never discussed; Step 3 does record the fact that would have caught it — "Photoinhibition ended before the Go cue; this is important for go-cue alignment and photostim input construction" — and the check was not carried out.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. From the raw lick event streams `BehavioralEvents/left_lick_times` and `right_lick_times`, not from the trials table. For each trial the AI finds the first left lick and the first right lick in the response window `[go, go + 1.5)` and takes whichever comes first as the choice; if there is no lick on either spout the trial is `no lick`. `trial_instruction` is decoded but never used (the reference derives choice from `trial_instruction` × `outcome` instead). I checked the two derivations against each other on a full session and they agree on 100% of trials; globally the AI's `no_lick` fraction (0.150) tracks the `ignore` fraction (0.149).

ii.
```python
left_lick_times = np.asarray(beh_events['left_lick_times']['timestamps'][()]).astype(float) if 'left_lick_times' in beh_events else np.array([])
right_lick_times = np.asarray(beh_events['right_lick_times']['timestamps'][()]).astype(float) if 'right_lick_times' in beh_events else np.array([])
```

```python
# choice from first lick after go cue within response window
lmask = (left_lick_times >= gt) & (left_lick_times < gt + WIN_END)
rmask = (right_lick_times >= gt) & (right_lick_times < gt + WIN_END)
lfirst = left_lick_times[lmask][0] if np.any(lmask) else np.inf
rfirst = right_lick_times[rmask][0] if np.any(rmask) else np.inf
```

iii. CONVERSION_NOTES Step 5 Key Decision 5: "**Choice definition**: Define choice from actual lick behavior when possible (left/right lick times after go cue); assign `no lick` when no lick is observed in the response period or outcome is ignore." The Step 5 mapping row lists "`trial_instruction` + lick event times (`left_lick_times`, `right_lick_times`) + `outcome`" as the sources, with the note "If no post-go lick in window or outcome ignore, assign no lick; otherwise infer first lick direction or instructed/correct side as needed."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded `0 = left`, `1 = right`, `2 = no lick`, named in `output_values[0] = ['left', 'right', 'no_lick']`. Choice is one value per trial, so it is tiled across all 80 bins with `np.full` and written into row 0 of an `(n_output, n_timepoints)` `int64` array, keeping all four outputs in a single per-trial matrix. The response window end (`gt + WIN_END`, 1.5 s) coincides with the end of the extracted trial window.

ii.
```python
if np.isfinite(lfirst) and (lfirst < rfirst):
    choice = 0  # left
elif np.isfinite(rfirst) and (rfirst < lfirst):
    choice = 1  # right
else:
    choice = 2  # no lick
```

```python
out = np.vstack([
    np.full(N_BINS, choice, dtype=np.int64),
    np.full(N_BINS, out_val, dtype=np.int64),
    np.full(N_BINS, early_val, dtype=np.int64),
    tongue_disc.astype(np.int64),
])
```

```python
'output_values': [
    ['left', 'right', 'no_lick'],
    ...
],
```

iii. The left/right/no-lick coding follows the Decoder Task specification directly. The target format says outputs should be time-varying "if at all possible", and the AI satisfies the shape requirement by broadcasting the per-trial value across bins so that all four outputs share one `(4, 80)` array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which already stores the three strings `'hit'`, `'miss'` and `'ignore'` that the Decoder Task asks for. No derivation.

ii.
```python
outcome = _decode_arr(trials['outcome'][()])[:n_trials]
```

iii. CONVERSION_NOTES Step 5 mapping: "`outcome` → output[1] outcome | Map string labels to categorical ignore / miss / hit | NWB trial table | Direct per-trial categorical output." Step 4 confirms the observed values: "`outcome` values `hit`/`miss`/`ignore`."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A fixed dictionary maps the strings to `0 = ignore`, `1 = miss`, `2 = hit`, and the value is tiled across all 80 bins into row 1 of the output array. The lookup uses `.get(..., 0)`, so any unexpected string would be silently coded as `ignore`; in practice only the three known values occur. The dictionary is re-created inside the trial loop. Realised distribution: hit 0.684, miss 0.166, ignore 0.149, consistent with the papers' ~84% correct rate on control trials once photostim and ignore trials are included.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
out_val = outcome_map.get(outcome[tr], 0)
```

```python
'output_values': [
    ...,
    ['ignore', 'miss', 'hit'],
    ...
],
```

iii. The 0/1/2 ordering is taken verbatim from the Decoder Task ("**Outcome** (ignore, miss, hit, per-trial)"). Step 5 Key Decision 6: "**Outcome and early lick**: Use direct categorical trial-table labels without collapsing categories."

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from the `early_lick` column of the trials table, whose values are the strings `'early'` and `'no early'`.

ii.
```python
early_lick = _decode_arr(trials['early_lick'][()])[:n_trials]
```

iii. CONVERSION_NOTES Step 5 mapping: "`early_lick` → output[2] early lick | Map string labels to categorical no / yes | NWB trial table | Direct per-trial categorical output." Step 4 records the observed values: "`early_lick` values `early`/`no early`", and notes that the first attempt at an aggregate summary failed "because `early_lick` is a string field, not integer", which is why the explicit decoder helper exists.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A fixed dictionary maps `'no early' → 0` and `'early' → 1`, tiled across all 80 bins into row 2 of the output array, with `.get(..., 0)` defaulting unknown strings to `no`. Realised distribution: no 0.884, yes 0.116. Note that the lick that sets the flag happens during the sample or delay epoch, so the causal event does fall inside the -2.5 s pre-go window even though the label is per-trial.

ii.
```python
early_map = {'no early': 0, 'early': 1}
early_val = early_map.get(early_lick[tr], 0)
```

```python
'output_values': [
    ...,
    ['no', 'yes'],
    ...
],
```

iii. The 0/1 ordering follows the Decoder Task ("**Early lick** (no, yes, per-trial)"). Step 5 Key Decision 6 covers this together with outcome: use the trial-table label directly, without collapsing categories.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is `(n_frames, 3)` and whose `timestamps` run at ~294 Hz. The AI hard-codes column 1 as y and column 2 as the visibility/likelihood channel, guarded only by a check that there are at least 3 columns and a comment "assume x, y, likelihood"; it does not read the series' `description` attribute, which states the layout explicitly (`('tongue_x', 'tongue_y', 'tongue_likelihood')`). The assumed layout is in fact correct. Column 0 (`tongue_x`) is loaded but never used.

ii.
```python
def choose_tongue_columns(tongue_data):
    if tongue_data.shape[1] < 3:
        raise ValueError('Tongue tracking data expected to have at least 3 columns')
    # assume x, y, likelihood
    return 1, 2
```

```python
tongue = np.asarray(beh_ts['Camera0_side_TongueTracking']['data'][()]).astype(float)
tongue_t = np.asarray(beh_ts['Camera0_side_TongueTracking']['timestamps'][()]).astype(float)
y_col, vis_col = choose_tongue_columns(tongue)
tongue_y = tongue[:, y_col]
tongue_vis = tongue[:, vis_col]
```

iii. CONVERSION_NOTES Step 5 mapping: "`BehavioralTimeSeries/Camera0_side_TongueTracking/data` and timestamps → output[3] tongue y-position | ... | Need to determine coordinate column order; likely x,y,likelihood with low likelihood => not visible." Trajectory step 25: "It also contains tongue tracking as `BehavioralTimeSeries/Camera0_side_TongueTracking/data` with shape (time, 3), likely x/y/likelihood or x/y/confidence, plus timestamps." The column order was assumed from the shape and never confirmed against the description attribute.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. A frame counts as visible if its y and likelihood are both finite and likelihood > 0.5. The 40th and 60th percentiles are computed **once per session over the raw visible frames** (`np.quantile(tongue_y[visible], [0.4, 0.6])`), not over binned means as in the reference. No temporal averaging is performed at all: each output bin takes the value of a single frame (see 8-d). If a session has no visible frame the thresholds degenerate to 0.0/0.0. Because the reference averages within a bin before digitising, and a bin is "visible" if *any* frame in it is, the two approaches give different class balances: the AI reports 83.6% `not_visible` against the reference's ~75%.

ii.
```python
visible = np.isfinite(tongue_y) & np.isfinite(tongue_vis) & (tongue_vis > 0.5)
if np.any(visible):
    q40, q60 = np.quantile(tongue_y[visible], [0.4, 0.6])
else:
    q40, q60 = 0.0, 0.0
```

iii. CONVERSION_NOTES Step 5 Key Decision 7: "**Tongue visibility/discretization**: Use session-wide tongue-y percentiles (40th, 60th) over visible frames only; assign class 3 when tracking confidence indicates not visible." The AI flagged the resulting imbalance in Step 7 as a possible problem but accepted it: "Tongue output was dominated by not-visible frames, which may reflect true tongue absence or a conservative visibility threshold and should be revisited if decoder performance is poor." It was never revisited, because the decoder reached 0.62 balanced accuracy.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four classes, following the Decoder Task exactly: `0` if y is below the session 40th percentile, `1` if between the 40th and 60th percentiles inclusive, `2` if above the 60th percentile, and `3` if the tongue is not visible at that bin. The array is initialised to 3 and only visible bins are overwritten, so `not visible` is the default. Realised distribution: 0.064 / 0.033 / 0.067 / 0.836 — i.e. within the visible bins the split is ~39% / 20% / 41%, as expected for 40/60 percentile edges.

ii.
```python
tongue_disc = np.full(N_BINS, 3, dtype=np.int64)
vis_now = np.isfinite(ty) & np.isfinite(tv) & (tv > 0.5)
tongue_disc[vis_now & (ty < q40)] = 0
tongue_disc[vis_now & (ty >= q40) & (ty <= q60)] = 1
tongue_disc[vis_now & (ty > q60)] = 2
```

```python
'output_values': [
    ...,
    ['lt_40th', '40th_to_60th', 'gt_60th', 'not_visible'],
],
```

iii. Step 5 Key Decision 7 and the Decoder Task's explicit four-way specification ("0: < 40th percentile ... 3: not visible"). Percentiles are per session, as the task requires.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Nearest-following-frame sampling on the same go-cue-relative grid. For each trial the AI computes the absolute times of the 80 bin centers, then `np.searchsorted(tongue_t, abs_centers, side='left')` gives, for each bin center, the index of the first camera frame at or after it; the index is clipped into range and that single frame's y and likelihood are used for the bin. There is no averaging over the frames inside a bin and, critically, no check that the selected frame is actually close to the bin center. The side camera is trial-gated, so during the inter-trial interval there are gaps of up to ~1.2 s (I measured 329 gaps >0.1 s in a 330-trial session); for bins falling in such a gap the code silently adopts the value of the next frame, which may be up to a second away. The reference instead assigns those bins to `not visible`. Clipping at the array ends has the same effect for bins beyond the last frame of the session.

ii.
```python
abs_centers = gt + BIN_CENTERS
...
idx = np.searchsorted(tongue_t, abs_centers, side='left')
idx = np.clip(idx, 0, len(tongue_t) - 1)
ty = tongue_y[idx]
tv = tongue_vis[idx]
```

iii. CONVERSION_NOTES Step 5 mapping: "Align tracking to trial bins; use y coordinate and session-wide 40th/60th percentiles to discretize into 0/1/2; use class 3 when tongue not visible based on tracking confidence/likelihood", citing the reference repo's `temporal_alignment_embed_and_ephys` as the conceptual model. No justification is given for choosing instantaneous sampling over within-bin averaging, and the camera-gap case is not discussed anywhere in the notes or trajectory.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases, handled with a mix of explicit exclusion and silent defaulting:

- **Session never quality-controlled** (`classification` NaN for all units): `_decode_arr` stringifies the NaNs, nothing equals `'good'`, the session yields zero units and is dropped by the `neural[0].shape[0] == 0` guard. One session is dropped this way.
- **Trials with no spike data**: excluded via `is_good_trials` plus the all-zero neural drop (see 1-e).
- **No tone onset in the accepted window**: replaced by a hard-coded `go - 0.6 s`, plus a second `np.isfinite` guard at use site. This *fabricates* a value rather than excluding the trial; 0.6 s is also not the task's nominal tone-to-go interval (sample 0.65 s + delay 1.2 s ≈ 1.85 s). It affects ~0.06% of trials.
- **Missing photostim columns / missing event streams / sessions without optogenetics**: default to `'N/A'` and an all-zero channel.
- **Missing `anno_name` or empty region string**: replaced by `'unknown'`.
- **Frames with no tracked tongue**: `likelihood <= 0.5` or non-finite → class 3 (`not_visible`).
- **Unexpected label strings**: `outcome_map.get(..., 0)` and `early_map.get(..., 0)` silently coerce anything unknown to `ignore` / `no`.
- **Trial/event length mismatch**: trial columns are silently truncated with `[:n_trials]` rather than asserted equal.
- **Unparseable photostim strings**: swallowed by a bare `except Exception: pass`, leaving the channel at zero.

ii.
```python
unit_class = _decode_arr(units['classification'][()]) if 'classification' in units else None
good_mask = unit_class == 'good' if unit_class is not None else np.ones(len(units['id']), dtype=bool)
anno_name = _decode_arr(units['anno_name'][()]) if 'anno_name' in units else np.array(['unknown'] * len(good_mask), dtype=object)
```

```python
stime = sample_times[tr] if np.isfinite(sample_times[tr]) else (gt - 0.6)
```

```python
try:
    pstart = float(pon[tr])
    pd = float(pdur[tr])
    photostim_on = ((abs_centers >= pstart) & (abs_centers < pstart + pd)).astype(np.float32)
except Exception:
    pass
```

```python
r = r if r != '' else 'unknown'
if np.all(neural == 0):
    continue
```

iii. CONVERSION_NOTES Step 10, Issues Found and Resolved, lists the two that were actually driven by observed failures: "Missing tone/sample onset for some trials caused NaN input values: replaced NaN with robust nearest-preceding sample-onset matching and finite fallback (~0.6 s before go cue)" and "Many late trials in some sessions had all-zero neural data ... Resolved by filtering to valid trial columns and dropping any residual all-zero neural trials." The remaining defaults (`.get(..., 0)`, `except: pass`, `[:n_trials]`, all-units fallback) are defensive coding with no documented rationale.

## 10-a. What are the most time-consuming steps of the code?

i. Overwhelmingly the nested trial × unit spike-binning loop. For every trial the code loops over all good units and calls `np.histogram` on that unit's *entire* session-long spike train, so the cost is O(n_trials × n_units × n_spikes_per_unit) — roughly 500 trials × 400 units × tens of thousands of spikes per session. The full conversion took **8,745 s (2 h 26 min)** for 174 sessions, ~50 s/session, against the reference's 247 s total and against the instructions' explicit 15-minute budget. Secondary costs are reading the full `spike_times` buffer and the ~1.3 M × 3 tongue-tracking array per session, and pickling the 12 GB result. The per-trial `np.searchsorted` over camera timestamps and the per-trial lick boolean masks (each scanning all of the session's licks) are minor by comparison.

ii.
```python
for tr in range(n_trials):
    ...
    neural = np.zeros((len(good_inds), N_BINS), dtype=np.float32)
    for j, st in enumerate(good_spike_times):
        rel = st - gt
        counts, _ = np.histogram(rel, bins=BIN_EDGES)
        neural[j] = counts.astype(np.float32) / BIN_SIZE
```

iii. Essentially undocumented. CONVERSION_NOTES Step 6 leaves "Code inefficiencies identified:" and "Code speedups added:" as unfilled template placeholders. Step 7 estimates "~8 s/session, ~23 min for 174 sessions (before optimization)" from a 2-session sample; the actual rate was ~6× that. The only recorded optimisation is in Step 10: "Performance regression from per-trial spike extraction: mitigated by caching spike times per good unit once per session" — which removes the repeated HDF5 reads but leaves the repeated histogramming untouched. Step 9's instruction to kill and re-optimise if the run exceeds 1.5× the estimate was not acted on.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four:

1. **The trial × unit binning loop** — the dominant cost, and fully vectorizable over trials. The reference builds one flat array of all trials' bin edges and issues a single `np.searchsorted` per unit, then differences the running counts: one binary search per (unit, edge) instead of a full array scan per (unit, trial).
2. **`infer_trial_event_times`** — a Python `for` loop over trials that boolean-masks the whole `sample_start_times` array each iteration, O(n_trials × n_events). A single `np.searchsorted(sample, go, 'left') - 1` replaces it.
3. **The per-trial lick search** — `(left_lick_times >= gt) & (left_lick_times < gt + WIN_END)` rescans every lick in the session once per trial; `searchsorted` on the sorted lick arrays would be O(log n).
4. **The region-index build** — `brain_regions_master.index(r)` and `subjects.index(subj)` are O(n) list scans executed once per neuron / per session; with 293 regions and 69,453 neurons a dict lookup is the obvious replacement.

The per-trial output assembly (`np.vstack` of four `np.full` calls, 90,734 times) and the per-trial re-creation of `outcome_map` / `early_map` are also avoidable, though cheap.

ii.
```python
for i, gt in enumerate(go_times):
    cand = event_ts[(event_ts <= gt - min_pre) & (event_ts >= gt - max_pre)]
```

```python
for j, st in enumerate(good_spike_times):
    rel = st - gt
    counts, _ = np.histogram(rel, bins=BIN_EDGES)
```

```python
for r in sess_regions:
    r = r if r != '' else 'unknown'
    if r not in brain_regions_master:
        brain_regions_master.append(r)
    sess_region_idx.append(brain_regions_master.index(r))
```

iii. No justification recorded; the notes identify no vectorizable loops. The instructions required "Write efficient code: Vectorize loops ... Print timing information to find bottlenecks" — the script records `load_time_sec` per session in `session_info` but never prints it, and no per-step timing was ever collected.

## 10-c. What processing does the code repeat multiple times?

i. Three substantive repetitions:

1. **Re-scanning every spike train once per trial.** `np.histogram(st - gt, bins=BIN_EDGES)` touches all of a unit's spikes even though at most 4 s of them can land in the window. With ~500 trials per session, each spike is examined ~500 times, and a full-length temporary `rel = st - gt` is allocated each time. This is the single largest source of wasted work in the script.
2. **Linear list scans for index lookup.** `brain_regions_master.index(r)` is called once per neuron and rescans a list that grows to 293 entries; `subjects.index(subj)` likewise per session. Both duplicate work a dict would do once.
3. **Per-trial reconstruction of constants.** `outcome_map`, `early_map`, `abs_edges` and `abs_centers` are rebuilt inside the trial loop; the lick boolean masks rescan the full session lick arrays per trial.

The AI did eliminate one genuine repetition during debugging — re-reading each unit's spike times from HDF5 on every trial — by hoisting `good_spike_times` out of the loop.

ii.
```python
good_spike_times = [get_unit_spike_times(units, int(ui)) for ui in good_inds]   # hoisted out (good)
...
for tr in range(n_trials):
    ...
    for j, st in enumerate(good_spike_times):
        rel = st - gt                                   # full-length temp, every trial
        counts, _ = np.histogram(rel, bins=BIN_EDGES)   # full scan, every trial
    ...
    outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}    # rebuilt every trial
    early_map = {'no early': 0, 'early': 1}
```

iii. CONVERSION_NOTES Step 10: "Performance regression from per-trial spike extraction: mitigated by caching spike times per good unit once per session." That is the only repetition the AI identified; the repeated histogramming that the caching exposed was not addressed, and Step 6's inefficiency section was left blank.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items, all small relative to 10-a but real:

- **`trial_instruction`** is decoded for every trial and never used, because choice is derived from lick times instead. It is a leftover from the Step 5 plan.
- **`tongue_x` (column 0)** is loaded as part of the full `(n_frames, 3)` array and never read.
- **`photostim_start_times` / `photostim_stop_times`** are read from every session but only feed the `'N/A'` fallback branch, which is unreachable in practice (and would raise `TypeError` if reached).
- **`abs_edges`** is recomputed for every trial but is used only inside that dead fallback branch and in the optional plot.
- **Full per-trial work on trials that are then discarded.** The neural matrix, both inputs and all four outputs are built before `if np.all(neural == 0): continue`, so every dropped trial (~2.6% of trials, and all 320 discarded trials of some sessions) pays the full binning cost for nothing.
- **`session_info['load_time_sec']` and `n_trials`** are stored in the pickle but never reported or used.

ii.
```python
trial_instruction = _decode_arr(trials['trial_instruction'][()])[:n_trials]   # never used again
```

```python
tongue = np.asarray(beh_ts['Camera0_side_TongueTracking']['data'][()]).astype(float)
y_col, vis_col = choose_tongue_columns(tongue)      # column 0 loaded, never read
```

```python
abs_edges = gt + BIN_EDGES        # only used in the dead photostim fallback and the plot
...
else:
    for ps, pe in zip(photostim_start, photostim_stop):
        if pe >= abs_edges[0] and ps <= abs_edges[-1]:
            photostim_on |= ((abs_centers >= ps) & (abs_centers < pe))
```

```python
if np.all(neural == 0):
    continue          # after all per-trial work has already been done
```

iii. Not documented. Step 6's "Code inefficiencies identified" is an empty placeholder, and none of the Step 10 or Step 12 review checks examined dead code or wasted computation.
