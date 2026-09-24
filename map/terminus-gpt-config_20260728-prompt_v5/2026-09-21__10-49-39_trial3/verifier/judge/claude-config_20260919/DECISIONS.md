# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates every NWB file under the hard-coded data root with `Path('/app/data').rglob('*.nwb')`, sorted, and processes one file per session (174 files). Rather than `pynwb`, it opens each file directly with `h5py` and reads the HDF5 groups by path: `intervals/trials` (whole trial table), `units` (whole unit table), `acquisition/BehavioralEvents/*/timestamps` (go cue, sample/tone onsets, left/right licks, photostim start/stop) and `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/{data,timestamps}`. `--sample` truncates the file list to the first 2 files. `load_trial_table` / `load_units_table` slurp *every* dataset in their group into a dict, decoding byte strings via `decode_arr`.

ii.
```python
files = sorted(Path('/app/data').rglob('*.nwb'))
if args.sample:
    files = files[:2]
...
for f in files:
    sn, si, so, subj, regions = process_session(f, show_processing=args.show_processing)
```
```python
def load_trial_table(f):
    g = f['intervals/trials']
    table = {k: decode_arr(g[k][()]) for k in g.keys() if isinstance(g[k], h5py.Dataset)}
    return table

def load_units_table(f):
    g = f['units']
    table = {k: decode_arr(g[k][()]) for k in g.keys() if isinstance(g[k], h5py.Dataset)}
    return table
```
```python
with h5py.File(path, 'r') as f:
    trial_table = load_trial_table(f)
    units = load_units_table(f)
```

iii. From CONVERSION_NOTES Step 2/6: the data are "organized as subject folders under `/app/data`, each containing session-level NWB files", 174 files / 28 subjects / 272,227 units / 94,990 trials were counted, and "Used h5py direct reads instead of full pynwb object loading for conversion" is listed explicitly as a speed-up. The trajectory shows the agent first inspected the NWB tree with h5py (steps 19-20) and located the relevant `BehavioralEvents` / `BehavioralTimeSeries` streams before committing to this loader.

## 1-b. How are the data split into subjects?

i. The subject of a session is taken from the *directory name* containing the NWB file (`path.parent.name`, e.g. `'sub-440956'`), not from `nwb.subject.subject_id`. Subjects are accumulated in first-encounter order into a list, and `subject_idx` is the position of the session's subject in that list (via `list.index`). 28 subjects result, with the same sessions-per-subject counts as the reference.

ii.
```python
subj = path.parent.name
...
if subj not in subjects:
    subjects.append(subj)
subject_idx.append(subjects.index(subj))
```

iii. CONVERSION_NOTES Step 5 maps "Subject folder / NWB subject metadata" → `subjects`, `subject_idx`, "Direct from NWB path/metadata", with the note "Session order must match neural/input/output lists". Step 2 documents the 28 subject folders and their session counts, which the agent used as the ground truth for the split.

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no grouping or splitting is performed. Session order is the sorted file order (which, because filenames embed the acquisition timestamp, is chronological within subject). A session is only emitted if at least 2 trials survive filtering (`if len(sn) < 2: continue`). No per-session identifier is stored; `metadata['session_files']` records the *full* input file list (all 174 paths) rather than only the retained sessions.

ii.
```python
files = sorted(Path('/app/data').rglob('*.nwb'))
for f in files:
    sn, si, so, subj, regions = process_session(f, ...)
    if len(sn) < 2:
        continue
    ...
    neural.append(sn)
```
```python
'metadata': {
    ...
    'session_files': [str(f) for f in files],
}
```

iii. CONVERSION_NOTES Step 2: "Sessions detected: 174 NWB files total", i.e. the file boundary is treated as the session boundary. The 2-trial minimum follows the target-format requirement ("There needs to be at least two trials within each session in order to evaluate the decoder performance"). All 174 sessions were retained in the final run, so the `session_files` list happens to line up with the emitted sessions.

## 1-d. How are the data split into trials?

i. Trials are the rows of the NWB trials table (`intervals/trials`), and the per-trial loop iterates over the go-cue event stream `acquisition/BehavioralEvents/go_start_times/timestamps`, indexing the trial table by the same positional index `i`. So trial *k* of the go-cue stream is assumed to be row *k* of the trials table. There is no assertion that the two lengths agree (the reference asserts this).

ii.
```python
def infer_go_cue_times(table, f=None):
    if f is not None and 'acquisition/BehavioralEvents/go_start_times/timestamps' in f:
        return np.asarray(f['acquisition/BehavioralEvents/go_start_times/timestamps'][:], dtype=float)
    ...
```
```python
for i, go in enumerate(go_times):
    start_idx = int(np.round((go + T_START - session_start) / BIN_SIZE))
    ...
    onset = safe_float(trial_table['photostim_onset'][i])
```

iii. CONVERSION_NOTES Step 10 / trajectory step 34: the agent originally *inferred* the go cue from `start_time`/`stop_time` (`start + min(2.0, (stop-start)/2)`), which produced large blocks of all-zero neural windows. It then discovered the real event stream — "We now found the exact go cue event stream: acquisition/BehavioralEvents/go_start_times/timestamps. Our current script was inferring go cue heuristically from trial start/stop times, which explains the all-zero neural windows" — and switched to it. The heuristic fallback is still in the code but is never reached for this dataset.

## 1-e. How are trials filtered based on quality controls?

i. Two filters, both derived from the neural data rather than from the file's own bookkeeping:
1. **Window out of range**: a trial is skipped if its 80-bin window falls outside the pre-binned session timeline (`start_idx < 0 or end_idx > session_fr.shape[1]`).
2. **All-zero neural**: a trial is skipped if *every* neuron has zero firing in *every* bin (`np.allclose(trial_mats, 0)`) — this check appears twice, identically, back to back.
Sessions with fewer than 2 surviving trials are dropped. No behavioural filter is applied (early-lick and ignore trials are deliberately kept). 3,991 of 94,990 trials are removed (4.2%), leaving 90,999 — close to the reference's 90,860. `obs_intervals` and `free_water`, the file's explicit records of which trials were observed, are never consulted.

ii.
```python
start_idx = int(np.round((go + T_START - session_start) / BIN_SIZE))
end_idx = start_idx + N_BINS
if start_idx < 0 or end_idx > session_fr.shape[1]:
    continue
trial_mats = session_fr[:, start_idx:end_idx].copy()
...
if np.allclose(trial_mats, 0):
    continue
if np.allclose(trial_mats, 0):
    continue
session_neural.append(trial_mats)
```
```python
if len(sn) < 2:
    continue
```

iii. CONVERSION_NOTES Step 10/12 and trajectory steps 61-64: the verifier reported "Session 1 trials 391-417 have all-zero neural data"; the agent found "101 sessions contained all-zero neural trials, sometimes in very large contiguous blocks, indicating many aligned windows are invalid or outside recorded spike support" and concluded "the practical fix is to drop all-zero-neural trials during conversion". Step 4 records the decision to "retain trials needed for outputs (including early lick and no-lick), but document divergence from paper analyses where task requirements demand it", because the paper excludes early-lick/no-response trials but the decoder outputs require them.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (a single ragged buffer) together with `units/spike_times_index` (per-unit end offsets), restricted to the units that pass QC (2-c). The go-cue timestamps define where the windows go. No other neural representation is used.

ii.
```python
def get_spike_times_for_unit(units_group, idx):
    st = units_group['spike_times']
    if 'spike_times_index' in units_group:
        ind = units_group['spike_times_index'][:]
        end = ind[idx]
        start = 0 if idx == 0 else ind[idx - 1]
        return st[start:end]
    return np.asarray(st[idx])
```

iii. CONVERSION_NOTES Step 5 maps "NWB units spike times / curated unit table" → `neural`, "Bin spikes into 50 ms bins from -2.5 s to +1.5 s around go cue; convert to spike counts or firing rates per neuron x time". Spike times are the only neural stream in the file.

## 2-b. How is the `neural` data processed?

i. For speed, each good unit is histogrammed **once over the whole session** on a single fixed 50 ms grid spanning `min(trial start) - 3.0 s` to `max(trial stop) + 2.0 s`, and counts are divided by the bin width to give Hz (`float32`). Per-trial matrices are then slices of that session-wide array. No smoothing, normalisation, or baseline subtraction. Result: `(n_neurons, 80)` in Hz per trial, matching the reference's units and lack of post-processing.

ii.
```python
session_start = float(np.min(start_times) + T_START - 0.5)
session_stop = float(np.max(stop_times) + T_END + 0.5)
session_edges = np.arange(session_start, session_stop + BIN_SIZE, BIN_SIZE)
session_fr = np.zeros((len(good_inds), len(session_edges) - 1), dtype=np.float32)
for j, u in enumerate(good_inds):
    st = get_spike_times_for_unit(f['units'], int(u))
    m = (st >= session_start) & (st < session_stop)
    counts, _ = np.histogram(st[m], bins=session_edges)
    session_fr[j] = counts.astype(np.float32) / BIN_SIZE
```

iii. CONVERSION_NOTES Step 7: "Pre-bin spikes once per session instead of histogramming per trial/unit — Reduced sample session runtime from ~38.7 s to ~1.37 s". The agent's Step 6 note had already identified "Per-trial per-unit spike binning is likely the main bottleneck".

## 2-c. How is the `neural` data filtered based on quality controls?

i. `choose_good_units` walks a priority list of label columns `['classification', 'unit_quality', 'quality', 'label', 'cluster_quality']` and uses the **first column that contains at least one value in `{'good','single','single_unit','single unit'}`**. For 173 of 174 sessions that is `classification == 'good'`, exactly the reference's criterion. For the one session that was never quality-controlled (`sub-440958_ses-20190216T162508`, where `classification` is NaN for all 1,852 units), the test fails and the code silently falls through to the deprecated `unit_quality` column, admitting its 1,201 `'good'`-by-old-label units. Below that there are further fallbacks to boolean flags and to hard-coded metric thresholds (presence_ratio ≥ 0.9, amplitude_cutoff ≤ 0.1, isi_violation ≤ 0.5, nn_hit_rate ≥ 0.9), plus a "keep everything" fallback if the mask is empty; none of these are reached. Totals: 174 sessions, 70,654 units (= the reference's 69,453 + the 1,201 non-QC'd units).

ii.
```python
def choose_good_units(units):
    n = len(units['id'])
    for key in ['classification', 'unit_quality', 'quality', 'label', 'cluster_quality']:
        if key in units:
            vals = units[key]
            sval = np.array([str(...).strip().lower() for v in vals], dtype=object)
            good_words = {'good', 'single', 'single_unit', 'single unit'}
            if np.isin(sval, list(good_words)).any():
                return np.isin(sval, list(good_words))
    ...
    mask = np.ones(n, dtype=bool)
    if 'presence_ratio' in units:
        mask &= np.asarray(units['presence_ratio'], dtype=float) >= 0.9
    ...
    if mask.sum() == 0:
        mask = np.ones(n, dtype=bool)
    return mask
```

iii. CONVERSION_NOTES Step 3: "Five region-specific logistic-regression classifiers labeled clusters as good/unlabeled based on 15 QC metrics and manual curation. Analyses in the papers used units labeled as good." Step 4: "Raw NWB contains 272,227 units total / papers report 69,943 good units → must filter raw units to good/curated units". Step 9 records the residual discrepancy — "Sessions: 174 raw NWB files vs 173 reported; Near-match; one-session discrepancy remains for Step 10 review" and "70,654 summed curated session-neuron counts vs 69,943 — Close" — but the review never closed it.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to go-cue onset (`go_start_times`), but *indirectly*: because the spikes were pre-binned onto one session-wide grid whose origin is unrelated to any go cue, each trial's window is taken as the 80 grid bins starting at the bin index **nearest** to `go + T_START` (`int(np.round(...))`). The grid is therefore snapped to the go cue only to within half a bin: measured over whole sessions the residual offset is uniform in ±25 ms (mean |offset| 12.5 ms; ~80% of trials are off by more than 5 ms). The inputs and the tongue output, in contrast, are placed on an *exactly* go-cue-relative grid, so the neural stream and the other streams are sub-bin inconsistent with one another.

ii.
```python
for i, go in enumerate(go_times):
    start_idx = int(np.round((go + T_START - session_start) / BIN_SIZE))
    end_idx = start_idx + N_BINS
    ...
    trial_mats = session_fr[:, start_idx:end_idx].copy()
```

iii. The pre-binning was introduced purely for speed (Step 7: 38.7 s → 1.37 s per sample session). Neither CONVERSION_NOTES nor the trajectory mentions the rounding, and the Step 10 sanity check that reported `neural_allclose True` happened to land on a trial whose residual was zero, so the jitter was never detected. `metadata['temporal_alignment_event']` is documented as "Go cue onset".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins, 80 bins per trial, spanning -2.5 s to +1.5 s about the go cue — identical to the reference. Spikes are binned once at that resolution (no finer intermediate representation, hence no rebinning). The bin grid is a module-level constant reused for every trial and session, so `T = 80` for all 90,999 trials. `metadata['time_bin_size'] = 50.0` (ms), `off_start = -2.5`, `off_end = 1.5`.

ii.
```python
BIN_SIZE = 0.05
T_START = -2.5
T_END = 1.5
N_BINS = int(round((T_END - T_START) / BIN_SIZE))
BIN_EDGES = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
```
```python
'time_bin_size': 50.0,
'off_start': T_START,
'off_end': T_END,
'n_timepoints': N_BINS,
```

iii. CONVERSION_NOTES Step 5, Key Decisions 1-2: "Use go cue alignment: Required by decoder task and consistent with task structure in methods" and "Use 50 ms bins over [-2.5, +1.5] s: Required by decoder task; should be applied consistently to neural and time-varying input/output streams."

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `acquisition/BehavioralEvents/sample_start_times/timestamps` (the sample-epoch tone onsets) plus the go-cue times. Two branches: if the number of sample events equals the number of trials, they are matched one-to-one by index; otherwise (the usual case — an early lick replays the sample epoch, so there are more tone events than trials) the code takes the **first** tone event falling inside `[trial start_time, trial stop_time]`. If no tone falls in the window, the value silently stays at the trial's `start_time`. The reference instead takes the **last** tone before the go cue; the two differ on roughly 3-7% of trials (the replayed ones).

ii.
```python
sample_starts = np.asarray(trial_table['start_time'], dtype=float)
if 'acquisition/BehavioralEvents/sample_start_times/timestamps' in f:
    sample_event_times = np.asarray(f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:], dtype=float)
    if len(sample_event_times) == len(go_times):
        sample_starts = sample_event_times.copy()
    else:
        for i in range(len(go_times)):
            cand = sample_event_times[(sample_event_times >= start_times[i]) & (sample_event_times <= stop_times[i])]
            if len(cand):
                sample_starts[i] = cand[0]
```

iii. CONVERSION_NOTES Step 5: "BehavioralEvents sample/tone onset times relative to go cue → input[0]: Time-varying continuous input: time from tone onset in seconds at each bin … If multiple sample tones exist, use task-specified tone onset representation relative to aligned bins." Trajectory step 34 lists "use trial-table sample_start_times event stream aligned by trial index" as one of the correctness fixes.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A continuous, time-varying value per bin: the bin centre (which is relative to the go cue) plus the go-cue-to-tone gap, i.e. `centre + (go - tone)`. Stored as `float32` row 0 of the `(2, 80)` input array. Observed range over the full dataset is [-1.5, 11.9] s, dominated by the normal 0.65 s sample + 1.2 s delay structure (tone 1.85 s before the go cue → first bin ≈ -0.625 s), with long tails on replayed-sample trials.

ii.
```python
def build_time_from_tone_vector(sample_start, go_time):
    return (BIN_CENTERS - (sample_start - go_time)).astype(np.float32)[None, :]
```
```python
inp0 = build_time_from_tone_vector(sample_starts[i], go)
...
inp = np.vstack([inp0, inp1]).astype(np.float32)
```

iii. CONVERSION_NOTES Step 7/trajectory step 40: the agent flagged the range as odd ("time_from_tone_onset_sec spans [-0.6, 5.7], which is unusual") and explicitly decided to keep the continuous signed representation rather than a binary onset indicator: "consistent with using a continuous relative-time signal rather than a binary onset indicator", because the Decoder Task specifies "Time from tone onset in seconds (continuous, time-varying)".

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at `BIN_CENTERS`, the same 80 bin centres (-2.475 … +1.475 s re go cue) that nominally define the neural window, so bin *k* of the input is intended to describe the same interval as bin *k* of the firing rates. In practice the neural bins are snapped to the session grid (2-d), so the two streams can be offset from each other by up to 25 ms.

ii.
```python
BIN_EDGES = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
...
return (BIN_CENTERS - (sample_start - go_time)).astype(np.float32)[None, :]
```

iii. Step 5 Key Decision 2: the 50 ms / [-2.5, +1.5] s grid "should be applied consistently to neural and time-varying input/output streams". All NWB streams share one session clock, so no resampling or offset correction is needed.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Primary source: the trials-table columns `photostim_onset` (string, seconds **relative to trial start**, `'N/A'` when the trial was not stimulated) and `photostim_duration` (string seconds). A fallback path uses the `photostim_start_times` / `photostim_stop_times` event streams intersected with the trial window, but it is unreachable for this dataset because both trial columns exist. `start_times` and the go-cue time are used to move the onset onto the go-cue-relative axis. Stimulated trials are ~21-26% per session, consistent with the paper's "~25% randomly interleaved trials".

ii.
```python
if 'photostim_onset' in trial_table and 'photostim_duration' in trial_table:
    onset = safe_float(trial_table['photostim_onset'][i])
    dur = safe_float(trial_table['photostim_duration'][i])
    if np.isfinite(onset) and np.isfinite(dur) and dur > 0:
        abs_on = start_times[i] + onset
        pst = np.array([abs_on], dtype=float)
        pen = np.array([abs_on + dur], dtype=float)
    else:
        pst = np.array([], dtype=float); pen = np.array([], dtype=float)
else:
    pst = ps_starts[(ps_starts >= start_times[i]) & (ps_starts <= stop_times[i] + 0.1)]
    ...
```
```python
def safe_float(x):
    ...
    if s.lower() in {'n/a', 'na', 'none', 'nan', ''}:
        return np.nan
    return float(s)
```

iii. CONVERSION_NOTES Step 10: "Photostim input initially all zeros: resolved by treating trial-table `photostim_onset` as trial-relative and combining with `photostim_duration`." Trajectory step 36 notes the bug ("photostimulation_on is always 0.0") and step 34 the fix ("use trial-table photostim_onset and photostim_duration per trial to build photostim input").

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1) time series: a bin is 1 when its centre lies in `[onset, onset+duration)`, 0 otherwise; non-stimulated trials are all-zero. Stored as row 1 of the `(2, 80)` input array.

**However the implementation is broken by an array-aliasing bug.** `start_times` and `sample_starts` are both `np.asarray(trial_table['start_time'], dtype=float)`, which returns the *same* ndarray object (no copy). The tone-matching loop in 3-a then writes `sample_starts[i] = cand[0]`, which silently overwrites `start_times` with the sample-onset times. By the time the photostim block runs, `start_times[i]` is the tone onset, not the trial start, so `abs_on = start_times[i] + onset` is ~1.25 s too late. Verified on `sub-440956_ses-20190207T120657`: the true stimulus is at -1.2 s re go cue with 0.5 s duration (ending 0.7 s *before* the go cue, matching "photoinhibition during the late delay epoch, ending before the go cue"), whereas the converted input turns on at ~0 s and peaks at +0.25 s *after* the go cue.

ii.
```python
start_times = np.asarray(trial_table['start_time'], dtype=float)
...
sample_starts = np.asarray(trial_table['start_time'], dtype=float)   # same object as start_times
...
        sample_starts[i] = cand[0]        # <-- also mutates start_times
...
abs_on = start_times[i] + onset           # <-- now tone onset + onset, not trial start + onset
```
```python
def build_photostim_vector(starts, stops, go_time):
    x = np.zeros(N_BINS, dtype=np.float32)
    for s, e in zip(starts, stops):
        rs = s - go_time
        re = e - go_time
        on = (BIN_CENTERS >= rs) & (BIN_CENTERS < re)
        x[on] = 1.0
    return x
```

iii. CONVERSION_NOTES Step 3 states the correct expectation — "Photoinhibition occurred during the late delay epoch and ended before the go cue; therefore photostimulation is a pre-go time-varying input in the aligned window" — and Step 5 Key Decision 5, "Construct photostim as a binary time series: Directly matches decoder input specification". The Step 10 sanity check claims `photostim_allclose True`, but it was run on session 0 / trial 0, a trial with no photostimulation, so it could not catch the shift; the documented expectation of a pre-go-cue stimulus was never checked against the converted data.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Intended: convert the absolute onset/offset to go-cue-relative times (`s - go_time`) and compare against the same `BIN_CENTERS` used for the other streams. Actual: because of the 4-b aliasing bug, the go-cue-relative onset is shifted by `+(tone onset - trial start)` ≈ +1.2 to +1.9 s, moving the stimulus from the delay epoch into the response epoch. Averaged over the two sample sessions the converted "photostim on" probability is 0.000 everywhere before the go cue and 0.05-0.26 in the bins from 0 to +0.7 s, the mirror image of the true timing.

ii.
```python
rs = s - go_time
re = e - go_time
on = (BIN_CENTERS >= rs) & (BIN_CENTERS < re)
```

iii. Same as 4-b: the intended alignment is documented in Step 5 ("Build time-varying photostim input aligned to go cue"), and the mis-timing is undocumented and undetected.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is **not** taken from the trials table; it is derived from the raw lick event streams `acquisition/BehavioralEvents/left_lick_times/timestamps` and `right_lick_times/timestamps`, tested against the response window `[go, min(trial stop_time, go + 1.5))`. (The reference instead derives it from `trial_instruction` × `outcome`.) `trial_instruction` is listed among the candidate column names but is never used, because the lick-based branch is unconditional.

ii.
```python
left_licks = f['acquisition/BehavioralEvents/left_lick_times/timestamps'][:] if ... else np.array([])
right_licks = f['acquisition/BehavioralEvents/right_lick_times/timestamps'][:] if ... else np.array([])
choice, outcome, early = infer_choice_outcome_early(trial_table, left_licks, right_licks, go_times)
```
```python
for i in range(n):
    l_post = np.any((left_licks >= go_times[i]) & (left_licks < min(stop[i], go_times[i] + 1.5)))
    r_post = np.any((right_licks >= go_times[i]) & (right_licks < min(stop[i], go_times[i] + 1.5)))
```

iii. CONVERSION_NOTES Step 5 maps "Trial lick direction / no response fields → output[0]: Per-trial categorical choice: left, right, no lick", via "Trial table + behavioral event/trial outcome logic", with the note "Keep no-lick class because decoder task requires it". The agent had observed that the trials table has no explicit choice column (it lists only `early_lick`, `outcome`, `trial_instruction`, `photostim_*`), so it measured the lick directly.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Per trial: left-only licks → 0 (`left`), right-only → 1 (`right`), and **both or neither → 2 (`no lick`)**. The per-trial code is then broadcast across all 80 bins as row 0 of the `(4, 80)` int output array.

The "both → no lick" tie-break is wrong: on those trials the animal *did* lick (they are scored `hit` or `miss`), and the first post-go lick defines the choice. In the full dataset `choice = no lick` occupies 20.7% of trials while `outcome = ignore` is only 14.9%, so ≈5.8% of all trials (≈5,000) carry a `no lick` choice that contradicts their own outcome label. Spot checks on two sessions show 1.1% and 3.0% both-lick trials and 97-99% agreement with the reference's instruction×outcome derivation. A secondary risk is truncating the response window at `stop_time`, which can be as little as 0.21 s after the go cue, although on the sessions checked the no-lick set still matched `outcome == 'ignore'` exactly.

ii.
```python
for i in range(n):
    l_post = ...; r_post = ...
    if l_post and not r_post:
        choice[i] = 0
    elif r_post and not l_post:
        choice[i] = 1
    else:
        choice[i] = 2
```
```python
out = np.vstack([
    np.full(N_BINS, choice[i], dtype=np.int64),
    ...
])
```
```python
'output_values': [
    ['left', 'right', 'no lick'],
    ...
```

iii. CONVERSION_NOTES does not discuss the ambiguous (both-side) case at all; the only recorded rationale is Step 5 ("Keep no-lick class because decoder task requires it") and the Decoder Task's three-class specification. The Step 10 output sanity check covered `early_lick` only, so the choice labels were never spot-checked against the raw trial table.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The trials-table `outcome` column, whose values are the strings `'hit'`, `'miss'`, `'ignore'` — exactly the three categories the Decoder Task asks for. If the column were absent, outcome would be back-derived from choice (`no lick → ignore`, else `hit`), but that branch is never taken.

ii.
```python
if 'outcome' in table:
    ov = np.array([str(...).strip().lower() for v in table['outcome']], dtype=object)
    ...
else:
    outcome[:] = np.where(choice == 2, 0, 2)
```

iii. CONVERSION_NOTES Step 5: "Trial outcome fields → output[1]: Per-trial categorical outcome: ignore, miss, hit", "Map raw trial result codes to requested categories". Trajectory step 34: "use trial-table early_lick and outcome directly".

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are lower-cased and mapped by a substring-tolerant cascade to `0 = ignore`, `1 = miss`, `2 = hit` — the same coding as the reference — then broadcast across all 80 bins as row 1. Full-dataset distribution: ignore 0.149 / miss 0.166 / hit 0.685, i.e. 80.5% correct among trials with a response, consistent with the paper's "84% correct rate (range, 65-99%)" once early-lick and ignore trials are counted in.

ii.
```python
for i, s in enumerate(ov):
    if s == 'hit' or 'correct' in s:
        outcome[i] = 2
    elif s == 'miss' or 'error' in s or 'incorrect' in s:
        outcome[i] = 1
    elif s == 'ignore' or 'no' in s:
        outcome[i] = 0
    else:
        outcome[i] = 0 if choice[i] == 2 else 2
```
```python
'output_values': [..., ['ignore', 'miss', 'hit'], ...]
```

iii. The code order follows the Decoder Task's listed order (`ignore, miss, hit` → 0, 1, 2). The extra substring branches are defensive generalisation for other possible encodings; CONVERSION_NOTES Step 6 describes the script as including "heuristic field detection for go cue, unit QC, and trial labels; these must be validated in subsequent steps".

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. The trials-table `early_lick` column (strings `'no early'` / `'early'`) — the same source as the reference. No derivation from lick timestamps is attempted.

ii.
```python
if 'early_lick' in table:
    ev = np.asarray(table['early_lick'])
    early = np.array([
        1 if str(...).strip().lower() == 'early' else 0
        for v in ev
    ], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 5: "Early lick indicator → output[2]: Per-trial categorical no/yes", "Trial table field or derive from lick timing before go cue", "Keep early-lick trials rather than excluding, due to decoder task". Trajectory step 36 records that the first version produced a constant 0 for early_lick, which was fixed by reading and parsing the actual string encoding.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Exact string match `'early'` → 1, everything else (including `'no early'`) → 0; broadcast across all 80 bins as row 2. Full-dataset distribution: no 0.884 / yes 0.116. This matches the reference's `{'no early': 0, 'early': 1}` mapping, with the difference that the AI's rule is "anything not exactly `'early'` is 0" rather than an explicit two-key dictionary (so an unexpected third label would be silently folded into "no").

ii.
```python
1 if str(v.decode('utf-8') if isinstance(v, (bytes, np.bytes_)) else v).strip().lower() == 'early' else 0
```
```python
np.full(N_BINS, early[i], dtype=np.int64),
```
```python
'output_values': [..., ['no', 'yes'], ...]
```

iii. Coding order follows the Decoder Task ("Early lick (no, yes, per-trial)"). CONVERSION_NOTES Step 10 lists the one output sanity check performed: "compared converted `early_lick` label for session 0 / trial 0 against raw NWB trial-table `early_lick`; converted `0` matched raw `no early`".

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data` and its `timestamps`. Only **column 1** (`tongue_y`) of the `(n_frames, 3)` array is read. Column 2, `tongue_likelihood`, is never read, even though the series' own `description` attribute declares the layout as `('tongue_x', 'tongue_y', 'tongue_likelihood')` and it is the only signal distinguishing a protruded tongue from a retracted one.

ii.
```python
tongue_ts = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:] if ... else np.array([])
tongue_xy = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:] if ... else np.zeros((0,2))
tongue_disc, tongue_thr = discretize_tongue_y(tongue_xy) if len(tongue_ts) else (np.array([], dtype=np.int64), (np.nan, np.nan))
```
```python
def discretize_tongue_y(data_xy):
    y = np.asarray(data_xy[:, 1], dtype=float)
```

iii. CONVERSION_NOTES Step 5 maps "Camera0_side_TongueTracking y coordinate → output[3]" and states the intended rule: "Percentiles computed per session over visible samples only; missing/low-confidence samples -> not visible". The trajectory (step 20) shows the agent found the tracking series but never inspected its channel layout or the likelihood distribution.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Every frame of the session is classified individually: `visible = np.isfinite(y)`, then the 40th/60th percentiles of `y[visible]` split the frames into classes 0/1/2; class 3 is used only where `y` is not finite. Each trial's frames are then assigned to 50 ms bins and each bin takes the **majority vote** of its frames' classes (excluding class 3); a bin with no frames at all becomes class 3.

The visibility test is vacuous: the tracking arrays contain no NaNs (verified — 0 NaNs in all three columns), so `isfinite` is always True and *every* frame is treated as a visible tongue. In reality only ~10.6% of frames have `likelihood ≥ 0.5`; the DeepLabCut tracker emits a plausible y even when the tongue is retracted (likelihood is strongly bimodal: 89% of frames below 0.01, 10.5% at or above 0.99). So the percentile edges are computed on ~89% tracker noise (40th/60th pct = 282.5/296.5 over all frames vs 273.7/288.5 over visible frames), and the required "not visible" class ends up on only 2.0% of bins (bins with no camera frame, i.e. the pre-trial period where video is off) instead of the ~75% the reference obtains.

ii.
```python
def discretize_tongue_y(data_xy):
    y = np.asarray(data_xy[:, 1], dtype=float)
    visible = np.isfinite(y)
    if visible.sum() == 0:
        return np.full(len(y), 3, dtype=np.int64), (np.nan, np.nan)
    q40, q60 = np.nanpercentile(y[visible], [40, 60])
    out = np.full(len(y), 3, dtype=np.int64)
    out[visible & (y < q40)] = 0
    out[visible & (y >= q40) & (y <= q60)] = 1
    out[visible & (y > q60)] = 2
    return out, (q40, q60)
```
```python
def bin_tongue_for_trial(timestamps, labels, go_time):
    out = np.full(N_BINS, 3, dtype=np.int64)
    rel = timestamps - go_time
    for b in range(N_BINS):
        m = (rel >= BIN_EDGES[b]) & (rel < BIN_EDGES[b + 1])
        if np.any(m):
            vals = labels[m]
            vals = vals[vals != 3]
            out[b] = 3 if len(vals) == 0 else np.bincount(vals, minlength=3).argmax()
    return out
```

iii. The intended rule is documented in Step 5 ("over visible samples only; missing/low-confidence samples -> not visible") but no confidence threshold was ever implemented. In trajectory step 36/40 the agent noticed the symptom — "tongue_y_position lacks the 'not_visible' class in the sample, which may be okay for these sessions but should be checked" and then "may simply reflect these sessions" — and dismissed it without checking. No sanity check was run on the tongue output.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per session, as the Decoder Task requires: `< 40th percentile → 0`, `[40th, 60th] → 1`, `> 60th → 2`, not visible → 3. Percentiles are taken over the *raw frames* of the session (the reference takes them over the 50 ms bin means, so that the edges are defined on the same quantity that is discretised), and the class of a bin is the majority frame class rather than the class of the bin mean. The resulting full-dataset distribution is 0.387 / 0.171 / 0.422 / 0.020, which has the roughly 40/20/40 shape the percentile rule implies but is defined on the contaminated sample described in 8-b — the three "visible" classes are in the main tracker-noise distribution of a retracted tongue, not tongue protrusion amplitude.

ii.
```python
q40, q60 = np.nanpercentile(y[visible], [40, 60])
out[visible & (y < q40)] = 0
out[visible & (y >= q40) & (y <= q60)] = 1
out[visible & (y > q60)] = 2
```
```python
'output_values': [..., ['lt_40th', '40th_to_60th', 'gt_60th', 'not_visible']]
```

iii. CONVERSION_NOTES Step 5 Key Decision 6: "Discretize tongue y-position per session: Use session-specific 40th/60th percentiles over visible samples, with a separate not-visible class as required" — i.e. taken directly from the Decoder Task specification.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Exactly on the go-cue-relative grid: frames are selected with `go + T_START <= t < go + T_END`, their times are expressed relative to the go cue, and they are assigned to bins by the shared `BIN_EDGES`. Bin *k* of the tongue output therefore covers the nominal interval of bin *k* of the neural data; the only residual mismatch is the ±25 ms snapping of the neural stream described in 2-d. The camera timestamps share the session clock with the spikes, so no interpolation or offset correction is applied. Bins with no frames (the pre-trial period, since the video is trial-gated) fall into class 3.

ii.
```python
if len(tongue_ts):
    m = (tongue_ts >= go + T_START) & (tongue_ts < go + T_END)
    tongue_trial = bin_tongue_for_trial(tongue_ts[m], tongue_disc[m], go)
else:
    tongue_trial = np.full(N_BINS, 3, dtype=np.int64)
```
```python
rel = timestamps - go_time
for b in range(N_BINS):
    m = (rel >= BIN_EDGES[b]) & (rel < BIN_EDGES[b + 1])
```

iii. Step 5 Key Decision 2 requires the 50 ms / [-2.5, +1.5] s grid to be "applied consistently to neural and time-varying input/output streams", and `--show-processing` plots the discretised tongue trace on the same time axis as the neural raster and the inputs so that misalignment would be visible.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The script is written defensively throughout, with five main mechanisms:
- **`'N/A'`-style strings**: `safe_float` maps `'n/a' | 'na' | 'none' | 'nan' | ''` to NaN, and non-finite photostim onset/duration yields an all-zero photostim vector.
- **Missing columns/streams**: every HDF5 path is guarded by an `in f` test with an empty-array default, and trial columns are looked up through candidate-name lists (`TRIAL_CANDIDATES`) with heuristic fallbacks (e.g. an inferred go cue, `stop = go + 3.0`).
- **Missing QC labels**: the `choose_good_units` cascade falls through to `unit_quality`, then boolean flags, then metric thresholds, then "keep everything" — this is what silently admits the 1,201 non-quality-controlled units of `sub-440958_ses-20190216T162508` (2-c).
- **Missing region metadata**: unparseable electrode-location JSON yields `'unknown'`, and the whole region block falls back to unit columns and then to all-`'unknown'`.
- **Missing spike/camera data**: trials whose window falls outside the binned range or whose neural matrix is all zeros are dropped; sessions left with <2 trials are dropped; bins with no camera frames become the `not_visible` class.

Two of these fallbacks misfire. The `unit_quality` fallback keeps a session the reference (and the papers' 173-session count) excludes. More seriously, the "missing sample event" fallback path is where the 4-b/4-c aliasing bug lives: `sample_starts` and `start_times` are the same ndarray, so patching missing tone times also rewrites the trial start times used for photostimulation.

ii.
```python
def safe_float(x):
    try:
        ...
        if s.lower() in {'n/a', 'na', 'none', 'nan', ''}:
            return np.nan
        return float(s)
    except Exception:
        return np.nan
```
```python
left_licks = f['acquisition/.../left_lick_times/timestamps'][:] if '...' in f else np.array([])
```
```python
if mask.sum() == 0:
    mask = np.ones(n, dtype=bool)
```
```python
sample_starts = np.asarray(trial_table['start_time'], dtype=float)   # aliases start_times
...
        sample_starts[i] = cand[0]
```

iii. CONVERSION_NOTES Step 6 describes the design: "Initial implementation includes heuristic field detection for go cue, unit QC, and trial labels; these must be validated in subsequent steps." Step 10 lists the missing-data issues found and fixed (heuristic go cue, all-zero neural trials, unparsed photostim strings, unknown brain regions). The instructions' Step 10 Check 5 ("There may be minor issues in the data that your code must handle") was answered with these fallbacks rather than with an audit of the data's own validity records (`obs_intervals`, `free_water`, `tongue_likelihood`, NaN `classification`).

## 10-a. What are the most time-consuming steps of the code?

i. The full conversion takes 372.6 s (6.2 min) for 174 sessions — 2.14 s/session on average, 0.98-4.97 s range — plus the time to pickle a 12.1 GB result. Profiling one session (368 trials, 459 good units of 1,952) gives:

| step | time |
|---|---|
| pre-binning all good units over the session grid (`np.histogram` per unit) | 0.58 s (~48%) |
| per-trial tongue binning (`bin_tongue_for_trial`, 80 masked passes per trial) | 0.39 s (~32%) |
| `load_units_table` (reads all 43 unit columns, incl. the 92 MB `spike_times` buffer and the 718k×2 `obs_intervals`) | 0.09 s |
| brain-region JSON parsing (one `json.loads` per unit) | 0.01 s |
| trial table, QC mask, choice loop | <0.02 s |

So runtime is dominated by the two per-unit / per-trial Python loops; file I/O is secondary but includes ~100 MB per session of columns that are never used.

ii.
```python
for j, u in enumerate(good_inds):
    st = get_spike_times_for_unit(f['units'], int(u))
    m = (st >= session_start) & (st < session_stop)
    counts, _ = np.histogram(st[m], bins=session_edges)
    session_fr[j] = counts.astype(np.float32) / BIN_SIZE
```
```python
print(f'processed {path.name} in {time.time()-t0:.2f}s trials={len(session_neural)} good_units={len(good_inds)}')
```

iii. CONVERSION_NOTES Step 6 identified "Per-trial per-unit spike binning is likely the main bottleneck", and Step 7 records the fix and its effect: "Pre-bin spikes once per session instead of histogramming per trial/unit — Reduced sample session runtime from ~38.7 s to ~1.37 s", with an estimate of "~4 min for 174 sessions", against the instructions' 15-minute budget. The tongue loop was never profiled or mentioned.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four:
- **`bin_tongue_for_trial`** — an 80-iteration Python loop that builds a boolean mask over the trial's frames *per bin*, i.e. O(80 × n_frames) per trial. A single `np.floor((t - t0)/BIN)` index plus one `np.bincount` per class would remove it entirely; it is ~32% of the runtime.
- **The per-unit histogram loop** — one `np.histogram` per unit over a ~50,000-bin edge array. Reading the ragged `spike_times` buffer once and using `np.searchsorted` against the flattened per-trial edges (as the reference does) would be both faster and exactly go-cue aligned.
- **The per-trial assembly loop** — inputs, photostim, and outputs are built one trial at a time with `np.full` / `np.vstack`; all of them are pure broadcasts that could be computed for the whole session at once (`CENTERS[None,:] + (go - tone)[:,None]`, etc.).
- **`infer_choice_outcome_early`'s lick loop** — O(n_trials × n_licks) with two full-array comparisons per trial; `np.searchsorted` on the sorted lick streams would make it O(n log n).

Two further micro-inefficiencies: `subjects.index(subj)` and `brain_regions.index(r)` are linear scans inside the per-unit region loop, and `get_spike_times_for_unit` re-reads the whole `spike_times_index` array on every call.

ii.
```python
for b in range(N_BINS):
    m = (rel >= BIN_EDGES[b]) & (rel < BIN_EDGES[b + 1])
    if np.any(m):
        vals = labels[m]
```
```python
for i in range(n):
    l_post = np.any((left_licks >= go_times[i]) & (left_licks < min(stop[i], go_times[i] + 1.5)))
    r_post = np.any((right_licks >= go_times[i]) & (right_licks < min(stop[i], go_times[i] + 1.5)))
```
```python
for r in regions:
    r = str(r)
    if r not in brain_regions:
        brain_regions.append(r)
    reg_idx.append(brain_regions.index(r))
```

iii. The instructions ask for vectorised loops and the notes commit to it ("Write efficient code: Vectorize loops"), but the only vectorisation actually pursued was replacing per-trial spike histogramming with one session-wide histogram per unit. Once the estimate fell under the 15-minute budget (Step 7: "~4 min for 174 sessions") the agent stopped optimising, and Step 7's efficiency review records no further bottleneck analysis.

## 10-c. What processing does the code repeat multiple times?

i. Several things are done twice:
- **The all-zero trial test is literally duplicated** — two identical `if np.allclose(trial_mats, 0): continue` statements in a row; the second can never fire.
- **`spike_times` is read twice**: `load_units_table` pulls the entire ragged buffer (92 MB on the first session) into a dict that is only ever used for label/metric columns, and then `get_spike_times_for_unit` reads the same data again, slice by slice, from the file.
- **`spike_times_index` is re-read from disk on every unit** (`ind = units_group['spike_times_index'][:]` inside the per-unit function), i.e. 459 times per session.
- **String decoding happens twice**: `decode_arr` already normalises the columns, and `choose_good_units` / `get_brain_regions` then re-run `str(v.decode('utf-8') ...)` element-wise over the same columns.
- **`np.asarray(trial_table['start_time'], dtype=float)` is materialised twice** (as `start_times` and `sample_starts`) — and because `np.asarray` does not copy, the two names are the same object, which is the root of the photostim bug in 4-b.
- **Linear `list.index` re-scans** of `subjects` / `brain_regions` for every session and every unit.

ii.
```python
if np.allclose(trial_mats, 0):
    continue
if np.allclose(trial_mats, 0):
    continue
```
```python
def get_spike_times_for_unit(units_group, idx):
    st = units_group['spike_times']
    if 'spike_times_index' in units_group:
        ind = units_group['spike_times_index'][:]
```
```python
start_times = np.asarray(trial_table['start_time'], dtype=float)
...
sample_starts = np.asarray(trial_table['start_time'], dtype=float)
```

iii. Not discussed in CONVERSION_NOTES. The duplication is a residue of the iterative patching visible in the trajectory (steps 61-64), where the all-zero filter was inserted by an in-place `sed`/patch after the conversion had already been written and run; Step 13's cleanup pass did not revisit the script.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Four categories:
- **Whole-unit-table loading**: `load_units_table` reads and decodes all 43 columns — `waveform_mean` (1952×82), `obs_intervals` (718,336×2), `is_good_trials` (1952×368), all 15 QC metrics and the full `spike_times` buffer — of which only `id`, `classification`/`unit_quality`, `electrodes` and (indirectly) the spikes are used. Ironically `obs_intervals`, the one column that would have given the correct trial filter, is loaded and thrown away.
- **Session-wide pre-binning**: every good unit is histogrammed across the *entire* session timeline, including the inter-trial intervals that no trial window ever touches (a 459 × 49,912 = 92 MB array per session), where only 80 bins per trial are needed.
- **Work on trials that are then dropped**: inputs, photostim, tongue binning and outputs are all computed *before* the all-zero test, so ~4,000 discarded trials are fully processed first (the tongue binning in particular is the second-most expensive operation).
- **Dead code paths**: `get_table_dict` is defined and never called; `TRIAL_CANDIDATES` and the go-cue/choice/outcome/photostim heuristic fallbacks are unreachable for this dataset; `discretize_tongue_y` returns thresholds `(q40, q60)` that are never stored or reported; `--show-processing` plots only trial 0 of each session.

ii.
```python
def get_table_dict(group):      # never called
    ...
```
```python
session_fr = np.zeros((len(good_inds), len(session_edges) - 1), dtype=np.float32)   # whole session
```
```python
            tongue_trial = bin_tongue_for_trial(tongue_ts[m], tongue_disc[m], go)
            out = np.vstack([...])
            if np.allclose(trial_mats, 0):
                continue                     # everything above is discarded
```

iii. Not discussed in CONVERSION_NOTES; the notes' only efficiency claims are the h5py-vs-pynwb choice and the session-level pre-binning. The pre-binning trade-off — extra memory and a half-bin alignment error in exchange for speed (2-d) — is nowhere weighed, and the instruction to "avoid unnecessary file I/O" is not addressed for the unit table.
