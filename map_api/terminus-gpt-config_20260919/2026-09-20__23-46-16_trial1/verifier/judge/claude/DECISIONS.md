# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globs all `*.nwb` files under `/app/data` using `Path('/app/data').rglob('*.nwb')`, sorts them, and processes each with `pynwb.NWBHDF5IO`. Each file is one session. Subjects, trials, and units are read from within each NWB file.

ii.
```python
files = sorted(Path('/app/data').rglob('*.nwb'))
...
for path in files:
    r = process_file(path, make_plot=args.show_processing and len(results) < 2)
    if r is not None: results.append(r)
```

```python
with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
    nwb = io.read(); trials = nwb.trials; units = nwb.units
```

iii. The AI uses pynwb as required by the instructions. One NWB file per session is the data organization, so globbing is sufficient.

## 1-b. How are the data split into subjects?

i. Each NWB file records its subject in `nwb.subject.subject_id`. The AI reads this for every session and builds a sorted unique list.

ii.
```python
return dict(..., subject=str(nwb.subject.subject_id), ...)
```

```python
subjects = sorted({r['subject'] for r in results})
subject_map = {x: i for i, x in enumerate(subjects)}
'subject_idx': np.asarray([subject_map[r['subject']] for r in results], dtype=np.int64),
```

iii. The `subject_id` field is the canonical identifier in NWB files.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. Each is identified by `nwb.identifier`. Session order follows sorted file paths.

ii.
```python
files = sorted(Path('/app/data').rglob('*.nwb'))
...
session_id = nwb.identifier
```

iii. The dandiset stores one session per file, so no further splitting is needed.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`nwb.trials`). The number of go cue events is verified to match the trial count.

ii.
```python
trials = nwb.trials; units = nwb.units
...
all_go = np.asarray(ev['go_start_times'].timestamps[:], dtype=np.float64)
if len(all_go) != len(trials):
    raise ValueError(f'{session_id}: go/trial count mismatch')
```

iii. The trials table has one row per trial, verified against go-cue event count.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies three filters: (1) exclude `auto_water == 1` or `free_water == 1` trials, (2) for each classifier-good unit, check `is_good_trials` and drop trials where any retained unit is invalid, using `obs_intervals` to map short validity vectors, and (3) remove trials whose entire neural rate matrix is zero (recording boundary safeguard). A session is dropped if fewer than 2 trials survive.

ii.
```python
auto = as_strings(trials['auto_water']) == '1'
free = as_strings(trials['free_water']) == '1'
trial_keep = ~(auto | free)
for ui in neuron_idx:
    stored = np.asarray(units['is_good_trials'][ui], dtype=bool)
    ...
    trial_keep &= valid
trial_idx = np.flatnonzero(trial_keep)
if len(trial_idx) < 2:
    return None
```

```python
nonzero_neural = np.any(rate_cube != 0, axis=(1, 2))
if not np.all(nonzero_neural):
    trial_idx = trial_idx[nonzero_neural]
```

iii. From CONVERSION_NOTES: "Exclude auto/free-water because they are nonstandard assisted trials and confound ordinary outcome/choice." The per-unit `is_good_trials` intersection ensures no unit is used on a trial where it was invalid. The zero-neural safeguard catches recording boundary artifacts.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `units/spike_times`, the sorted spike times for each unit. Only units with `classification == 'good'` contribute. Go-cue times from `BehavioralEvents/go_start_times` define bin edges.

ii.
```python
classes = as_strings(units['classification'])
neuron_idx = np.flatnonzero(classes == 'good')
spikes = [np.asarray(units['spike_times'][int(i)], dtype=np.float64) for i in neuron_idx]
```

iii. `spike_times` is the only neural representation in the NWB files.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 80 non-overlapping 50ms bins spanning [-2.5, 1.5) s relative to the go cue. For each neuron, the flat trial-edge array is searched with `np.searchsorted`, reshaped, and differenced to get spike counts, then divided by bin width (0.05 s) for Hz. No smoothing or normalization is applied.

ii.
```python
def bin_spikes(spike_times, absolute_edges):
    n_trials, n_edges = absolute_edges.shape
    flat_edges = absolute_edges.ravel()
    rates = np.empty((n_trials, len(spike_times), n_edges - 1), dtype=np.float32)
    for j, spikes in enumerate(spike_times):
        spikes = np.asarray(spikes, dtype=np.float64)
        cumulative = np.searchsorted(spikes, flat_edges, side='left').reshape(n_trials, n_edges)
        rates[:, j, :] = np.diff(cumulative, axis=1) / BIN_S
    return rates
```

iii. Standard spike binning approach consistent with the reference code's `sliding_histogram` but using the decoder-mandated 50ms non-overlapping bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are retained. Sessions with zero such units are dropped. Additionally, per-unit `is_good_trials` validity is intersected across all retained units to exclude trials where any unit is invalid.

ii.
```python
classes = as_strings(units['classification'])
neuron_idx = np.flatnonzero(classes == 'good')
if len(neuron_idx) == 0:
    print(f'SKIP {session_id}: zero classifier-good units', flush=True)
    return None
```

iii. `classification` is the published QC classifier output. The AI also uses `is_good_trials` for trial-level validity, which goes beyond the unit-level QC filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All NWB timestamps share one session-absolute clock. Bin edges are computed as go-cue times plus the relative bin edges. No resampling or offset correction is needed.

ii.
```python
abs_edges = go[:, None] + EDGES_REL[None, :]
```

iii. The go cue is the alignment event specified in the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 80 non-overlapping 50ms bins spanning [-2.5, 1.5) s relative to the go cue. Bin edges are defined via `np.linspace(-2.5, 1.5, 81)`.

ii.
```python
OFF_START, OFF_END, BIN_S = -2.5, 1.5, 0.05
EDGES_REL = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
N_TIME = 80
```

iii. Matches the decoder task specification: 50ms bins, 2.5s before to 1.5s after go cue.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `sample_start_times`, `sample_stop_times`, `delay_start_times`, and `delay_stop_times` from `BehavioralEvents`, plus trial `start_time` and go-cue times.

ii.
```python
sample_starts = np.asarray(ev['sample_start_times'].timestamps[:], dtype=np.float64)
sample_stops = np.asarray(ev['sample_stop_times'].timestamps[:], dtype=np.float64)
delay_starts = np.asarray(ev['delay_start_times'].timestamps[:], dtype=np.float64)
delay_stops = np.asarray(ev['delay_stop_times'].timestamps[:], dtype=np.float64)
all_tone = choose_tone_onsets(starts, all_go, sample_starts, sample_stops,
                              delay_starts, delay_stops)
```

iii. The AI uses all four event streams to trace the sample→delay→go state chain, resolving repeated sample events in early-lick trials.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI implements a state-chain resolution: for each trial, find the delay whose stop is closest to the go cue, then find the sample whose stop is closest to that delay's start. The tone onset is that sample's start time. The time-from-tone for each bin is `absolute_bin_center - tone_onset`.

ii.
```python
def choose_tone_onsets(trial_starts, go_times, sample_starts, sample_stops,
                       delay_starts, delay_stops):
    out = np.empty(len(go_times), dtype=np.float64)
    for i, (start, go) in enumerate(zip(trial_starts, go_times)):
        didx = np.flatnonzero((delay_starts >= start) & (delay_starts <= go))
        sidx = np.flatnonzero((sample_starts >= start) & (sample_starts < go))
        dk = didx[np.argmin(np.abs(delay_stops[didx] - go))]
        sk = sidx[np.argmin(np.abs(sample_stops[sidx] - delay_starts[dk]))]
        out[i] = sample_starts[sk]
    return out
```

```python
tone_time = abs_centers - tone[:, None]
```

iii. From CONVERSION_NOTES: "Select the sample onset whose stop is nearest the final delay start, where the final delay is identified by its stop at go. This resolves repeated early-lick sample states and variable delay durations."

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The time-from-tone is computed at each bin center, which is the same grid used for neural binning. Both use `go + CENTERS_REL`.

ii.
```python
abs_centers = go[:, None] + CENTERS_REL[None, :]
tone_time = abs_centers - tone[:, None]
```

iii. Same bin grid ensures alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from `BehavioralEvents/photostim_start_times` and `photostim_stop_times` absolute timestamps, plus trial `start_time` and `stop_time` to identify which event belongs to each trial.

ii.
```python
ps = np.asarray(ev['photostim_start_times'].timestamps[:], dtype=np.float64)
pe = np.asarray(ev['photostim_stop_times'].timestamps[:], dtype=np.float64)
```

iii. The AI uses BehavioralEvents timestamps rather than the trial table's `photostim_onset`/`photostim_duration` columns.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the AI finds photostim events within the trial's time window. A bin center that falls within [start, stop) is marked 1, else 0. Non-stimulated trials remain all-zero.

ii.
```python
stim = np.zeros((len(trial_idx), N_TIME), dtype=np.float32)
for k, src_i in enumerate(trial_idx):
    a, b = starts[src_i], float(trials['stop_time'][src_i])
    hits = np.flatnonzero((ps >= a) & (ps <= b))
    if len(hits) > 1:
        raise ValueError(f'{session_id} trial {src_i}: multiple photostim intervals')
    if len(hits) == 1:
        h = hits[0]
        stim[k] = ((abs_centers[k] >= ps[h]) & (abs_centers[k] < pe[h])).astype(np.float32)
```

iii. The per-trial loop matches events to trials by time window. The result is a binary time series.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostim is evaluated at the same bin centers (`go + CENTERS_REL`) as the neural data, ensuring alignment.

ii.
```python
stim[k] = ((abs_centers[k] >= ps[h]) & (abs_centers[k] < pe[h])).astype(np.float32)
```

iii. Same bin center grid as neural and tone-time inputs.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `trial_instruction` ('left'/'right') and `outcome` ('hit'/'miss'/'ignore') columns in the trials table.

ii.
```python
instruction = as_strings(trials['trial_instruction'])
outcome = as_strings(trials['outcome'])
...
arr[0] = trial_choice(instruction[src_i], outcome[src_i])
```

```python
def trial_choice(instruction, outcome):
    if outcome == 'ignore': return 2
    if outcome == 'hit': return 0 if instruction == 'left' else 1
    if outcome == 'miss': return 1 if instruction == 'left' else 0
```

iii. Choice is not stored directly; it is fully determined by instruction and outcome.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Hit means the animal licked the instructed side (left=0, right=1). Miss means the opposite side. Ignore (no lick) = 2. The per-trial value is repeated across all 80 time bins.

ii.
```python
def trial_choice(instruction, outcome):
    if outcome == 'ignore': return 2
    if outcome == 'hit': return 0 if instruction == 'left' else 1
    if outcome == 'miss': return 1 if instruction == 'left' else 0
    raise ValueError(f"unknown outcome {outcome!r}")
```

```python
arr[0] = trial_choice(instruction[src_i], outcome[src_i])
```

iii. Standard derivation matching the instruction's left=0, right=1, no lick=2 encoding.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which contains 'ignore', 'miss', 'hit'.

ii.
```python
outcome = as_strings(trials['outcome'])
omap = {'ignore': 0, 'miss': 1, 'hit': 2}
arr[1] = omap[outcome[src_i]]
```

iii. The trials table stores outcome explicitly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three outcome strings are mapped to integers: ignore=0, miss=1, hit=2. The per-trial value is repeated across all 80 bins.

ii.
```python
omap = {'ignore': 0, 'miss': 1, 'hit': 2}
arr[1] = omap[outcome[src_i]]
```

iii. Follows the instructions' ordering.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds 'no early' and 'early'.

ii.
```python
early = as_strings(trials['early_lick'])
arr[2] = 0 if early[src_i] == 'no early' else 1
```

iii. The trials table flags early licking explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0 (no early) or 1 (early). Repeated across all 80 bins.

ii.
```python
arr[2] = 0 if early[src_i] == 'no early' else 1
```

iii. Binary encoding per the instructions.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which has `(n_frames, 3)` data = (tongue_x, tongue_y, tongue_likelihood) with matching timestamps.

ii.
```python
tongue = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
video_t = np.asarray(tongue.timestamps[:], dtype=np.float64)
video_d = np.asarray(tongue.data[:], dtype=np.float64)
```

iii. This is the only tongue tracking stream in the files.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI uses a visibility threshold of 0.9 on `tongue_likelihood` to identify visible frames. The 40th and 60th percentiles are computed over all visible frames in the session (raw frames, not binned). For each trial, the nearest video frame to each bin center is found, and the tongue y-value at that single frame is used for discretization.

ii.
```python
VISIBILITY_THRESHOLD = 0.9
visible_session = video_d[:, 2] >= VISIBILITY_THRESHOLD
q40, q60 = np.percentile(video_d[visible_session, 1], [40, 60])
frame_idx = nearest_indices(video_t, abs_centers.ravel()).reshape(len(trial_idx), N_TIME)
ty = video_d[frame_idx, 1]; tl = video_d[frame_idx, 2]
```

iii. From CONVERSION_NOTES: "Use likelihood >= 0.9 as visible. Likelihood is strongly bimodal and 0.95 produces nearly identical visible fractions; 0.9 avoids treating low-confidence placeholder coordinates as behavior."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Visible frames (likelihood >= 0.9): below 40th percentile = 0, 40th to 60th = 1, above 60th = 2. Non-visible frames = 3.

ii.
```python
tongue_class = np.full((len(trial_idx), N_TIME), 3, dtype=np.int64)
vis = tl >= VISIBILITY_THRESHOLD
tongue_class[vis & (ty < q40)] = 0
tongue_class[vis & (ty >= q40) & (ty <= q60)] = 1
tongue_class[vis & (ty > q60)] = 2
```

iii. Follows the instructions: 0 below 40th, 1 40th-60th, 2 above 60th, 3 not visible.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each bin center (same go-cue-relative grid as neural data), the nearest video frame is found using `nearest_indices` (binary search with tie-breaking). The tongue y-value and likelihood of that single nearest frame are used.

ii.
```python
def nearest_indices(sorted_times, query):
    idx = np.searchsorted(sorted_times, query)
    idx = np.clip(idx, 1, len(sorted_times) - 1)
    left = idx - 1
    use_left = np.abs(query - sorted_times[left]) <= np.abs(sorted_times[idx] - query)
    return np.where(use_left, left, idx)

frame_idx = nearest_indices(video_t, abs_centers.ravel()).reshape(len(trial_idx), N_TIME)
```

iii. The camera timestamps share the session clock, so nearest-frame lookup ensures temporal alignment with neural bins.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases: (1) Sessions with NaN classification (no QC) are dropped (zero good units). (2) Trials with invalid neural recording are excluded via `is_good_trials` intersection and zero-neural safeguard. (3) Tongue frames with low likelihood are treated as "not visible" (class 3).

ii.
```python
classes = as_strings(units['classification'])  # NaN becomes 'nan' string
neuron_idx = np.flatnonzero(classes == 'good')
if len(neuron_idx) == 0:
    return None
```

```python
nonzero_neural = np.any(rate_cube != 0, axis=(1, 2))
if not np.all(nonzero_neural):
    trial_idx = trial_idx[nonzero_neural]
```

```python
tongue_class = np.full((len(trial_idx), N_TIME), 3, dtype=np.int64)  # default not visible
vis = tl >= VISIBILITY_THRESHOLD
```

iii. Missing data is handled by exclusion (sessions/trials with no valid neural data) or explicit categorization (tongue not visible), avoiding fabrication of data.

## 10-a. What are the most time-consuming steps of the code?

i. Reading each NWB file dominates. The full conversion takes ~194s for 174 sessions. Within a session, loading spike_times and video arrays, then the per-neuron searchsorted loop are the main costs. Pickling the ~11 GB result takes ~13.6s.

ii. N/A (timing is from CONVERSION_NOTES)

iii. Work is dominated by I/O and per-neuron binary search, both scaling with data size.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops: (1) per-neuron spike binning loop (one `searchsorted` per neuron, but vectorized across trials), and (2) per-trial photostim event matching loop. The photostim loop could have been vectorized using trial-table columns directly.

ii.
```python
for j, spikes in enumerate(spike_times):
    cumulative = np.searchsorted(spikes, flat_edges, side='left').reshape(n_trials, n_edges)
    rates[:, j, :] = np.diff(cumulative, axis=1) / BIN_S
```

```python
for k, src_i in enumerate(trial_idx):
    a, b = starts[src_i], float(trials['stop_time'][src_i])
    hits = np.flatnonzero((ps >= a) & (ps <= b))
```

iii. The per-neuron loop is inherent (ragged spike arrays). The photostim loop and the tone-onset loop could be vectorized.

## 10-c. What processing does the code repeat multiple times?

i. The AI reads `go_start_times` timestamps twice: once for `all_go_for_validity` during trial filtering, and once for `all_go` used in alignment. These are the same data.

ii.
```python
all_go_for_validity = np.asarray(
    nwb.acquisition['BehavioralEvents'].time_series['go_start_times'].timestamps[:], ...)
...
all_go = np.asarray(ev['go_start_times'].timestamps[:], dtype=np.float64)
```

iii. Minor redundancy; the second read could reuse the first variable.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI stores extra metadata fields (`source_trial_idx`, `q40`, `q60`, `elapsed` per session) that are not used by the decoder but are helpful for debugging. The tone-onset state-chain resolution reads `sample_stop_times`, `delay_start_times`, and `delay_stop_times` which a simpler approach (last sample before go) would not need. Otherwise, all computed fields go into the output.

ii. N/A

iii. The extra metadata is retained for reproducibility; the complex tone computation is the AI's chosen approach.
