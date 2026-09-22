# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is one NWB file per session laid out as `/app/data/sub-<subject_id>/sub-<id>_ses-<timestamp>_behavior+ecephys+ogen.nwb`. The AI discovers every file with a single recursive glob over `/app/data`, sorts the paths for a deterministic session order, and opens each one exactly once with `pynwb.NWBHDF5IO(..., load_namespaces=True)` (no `h5py` anywhere). Inside a session it reads `nwb.trials`, `nwb.units`, `nwb.acquisition['BehavioralEvents'].time_series`, `nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']`, and `nwb.subject`. 174 files are found; 173 are converted (one is dropped, see 2-c), giving 28 subjects, 89,068 trials and 69,453 units.

ii.
```python
files = sorted(Path('/app/data').rglob('*.nwb'))
if not files: raise FileNotFoundError('no NWB files under /app/data')
target = 2 if args.sample else None
results=[]; wall=time.perf_counter()
for path in files:
    r=process_file(path, make_plot=args.show_processing and len(results)<2)
    if r is not None: results.append(r)
    if target is not None and len(results)>=target: break
```

```python
with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
    nwb = io.read(); trials = nwb.trials; units = nwb.units
    session_id = nwb.identifier
    ...
    ev = nwb.acquisition['BehavioralEvents'].time_series
    tongue = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
```

iii. From CONVERSION_NOTES Step 2: "`/app/data` contains 174 NWB files organized as one subject directory per mouse... All files were inspected exclusively with `pynwb.NWBHDF5IO(..., load_namespaces=True)`; `h5py` was not used." The AI ran an explicit all-session census through `pynwb` (174 sessions, 28 subjects, 94,990 trials, 272,227 unfiltered units) before writing the converter, so the glob was verified to be the complete set of sessions. A single pass per file was chosen to avoid redundant I/O ("Avoid unnecessary file I/O").

## 1-b. How are the data split into subjects (mice)?

i. Each file's animal is read from `nwb.subject.subject_id` (a numeric string such as `'440956'`). At assembly the unique ids are sorted into `subjects` and each session gets an index into that list via `subject_idx`. No re-grouping from directory names is done. Result: 28 subjects, 3–10 sessions each.

ii.
```python
return dict(..., subject=str(nwb.subject.subject_id), ...)
```
```python
subjects=sorted({r['subject'] for r in results}); subject_map={x:i for i,x in enumerate(subjects)}
...
'subjects':subjects,
'subject_idx':np.asarray([subject_map[r['subject']] for r in results],dtype=np.int64),
```

iii. CONVERSION_NOTES Step 5 mapping table: "`subject.subject_id` → `subjects`, `subject_idx`; Unique sorted subject IDs and session index lookup". The AI's Step 2 census confirmed "Subjects | 28 | Sessions / subject | 3–10", which it then cross-checked against the paper cohort in Step 9 ("Subjects | 28 | 28 | Yes"). The NWB subject field is treated as the canonical animal identifier; the AI notes the session identifier separately encodes the paper's mouse name (e.g. `SC015_...`).

## 1-c. How are the data split into sessions?

i. One NWB file is one session, so nothing has to be inferred. Each session is labelled by `nwb.identifier` (e.g. `SC015_20190207_120657_s1`), and the sorted file order defines the session order in every list-of-sessions field. Per-session provenance (`session_id`, `source_file`, `source_trial_idx`, `q40`, `q60`, `elapsed`) is written into `metadata['session_info']`.

ii.
```python
files = sorted(Path('/app/data').rglob('*.nwb'))
...
session_id = nwb.identifier
...
'session_info':[{k:r[k] for k in ['session_id','source_file','source_trial_idx','q40','q60','elapsed']} for r in results],
```

iii. Step 2: "one subject directory per mouse (`sub-<id>/sub-<id>_ses-<timestamp>_...nwb`)" and "All 174 sessions share one trial/unit schema." Step 4 resolved the session count against the data paper: 174 NWBs exist but "exactly one session has zero classifier-good units; 173 have usable units", versus the paper's "173 behavioral sessions" — so excluding the unusable file "exactly reconciles usable session count."

## 1-d. How are the data split into trials?

i. Trials are the rows of the NWB trials table. The AI verified (over all 174 sessions) that `BehavioralEvents/go_start_times` has exactly one timestamp per trial row and that each go cue lies inside its trial, so trial row *i* is paired positionally with go event *i*. The code re-asserts this equality per session at runtime. Sample/delay events are explicitly *not* matched positionally because they can repeat within a trial.

ii.
```python
all_go = np.asarray(ev['go_start_times'].timestamps[:], dtype=np.float64)
if len(all_go) != len(trials):
    raise ValueError(f'{session_id}: go/trial count mismatch')
```
```python
go = all_go[trial_idx]; tone = all_tone[trial_idx]
abs_edges = go[:, None] + EDGES_REL[None, :]
```

iii. Step 4: "Go alignment | Processed spike times use go = 0 | `go_start_times` count equals trials in all 174 sessions and every go lies inside its corresponding trial | ... | Pair trial row `i` with go event `i`; this mapping is exact." Step 2 separately flags that "Sample and delay events can have more entries than trials, so they require trial-wise association rather than positional assumptions."

## 1-e. How are trials filtered based on quality controls?

i. Four filters, applied in order:
1. **Assisted-water trials excluded**: `auto_water == 1` or `free_water == 1` (3,766 trials).
2. **Per-unit recording validity**: a trial is kept only if `units/is_good_trials` is true for *every* retained classifier-good unit. For the 3,401 units (8 sessions) whose validity vector is shorter than the trial table, the flags are mapped onto the source trials whose go cue falls inside that unit's `obs_intervals`; trials not covered by any observation interval are invalid. This is what removes leading/trailing trials recorded outside the ephys block (e.g. 159/480 and 206/582 retained in two sessions).
3. **All-zero-neural safeguard**: after binning, any trial whose entire retained-neuron rate matrix is zero is dropped (2 recording-boundary trials).
4. **Minimum session size**: a session with fewer than 2 surviving trials is dropped.

Early-lick, `ignore` (no-response), `miss`, and photostimulation trials are deliberately **kept**, contrary to the reference "regular trial" mask, because they are required decoder inputs/outputs. Net: 89,068 of 94,990 trials retained.

ii.
```python
auto = as_strings(trials['auto_water']) == '1'
free = as_strings(trials['free_water']) == '1'
trial_keep = ~(auto | free)
# A trial is retained only if every retained classifier-good unit is valid.
# Short validity vectors correspond to a contiguous ephys recording block,
# which can start late as well as end early; map them using obs_intervals.
for ui in neuron_idx:
    stored = np.asarray(units['is_good_trials'][ui], dtype=bool)
    if len(stored) == len(trials):
        valid = stored
    else:
        represented = np.zeros(len(trials), dtype=bool)
        obs = np.asarray(units['obs_intervals'][ui], dtype=np.float64).reshape(-1, 2)
        for a, b in obs:
            represented |= (all_go_for_validity >= a) & (all_go_for_validity <= b)
        represented_idx = np.flatnonzero(represented)
        if len(represented_idx) != len(stored):
            raise ValueError(...)
        valid = np.zeros(len(trials), dtype=bool)
        valid[represented_idx] = stored
    trial_keep &= valid
trial_idx = np.flatnonzero(trial_keep)
if len(trial_idx) < 2:
    print(f'SKIP {session_id}: fewer than two valid standard trials', flush=True)
    return None
```
```python
# A whole-session-neuron matrix of zeros is effectively outside usable ephys
# coverage (127 such boundary/misaligned trials were detected in review).
nonzero_neural = np.any(rate_cube != 0, axis=(1, 2))
if not np.all(nonzero_neural):
    trial_idx = trial_idx[nonzero_neural]; ...
```

iii. Step 4/Step 5: "Reference movement/video analyses define 'regular trials' by excluding early lick, auto-water, free-water, no response, and photostimulation. The requested decoder explicitly needs early-lick, no-lick/miss, and photostimulation labels, so these trial classes must be retained. Auto/free-water trials remain candidates for exclusion because their outcomes are not standard instructed behavior." On validity: "For a fixed neuron set per session, retain only classifier-good units valid on every retained trial; do not encode invalid periods as zero firing." The short-vector rule was found and corrected during Step 10 review: the initial right-padding assumption misassigned 505 flags in `SC026_20190807_134913_s20` (recorded block = source trials 125–629); the AI then "Tested all 3,401 short-mask classifier-good units: observation-overlapping go-cue trial counts exactly equal stored validity-vector lengths (zero mapping failures)." The all-zero safeguard was added to clear the last 2 of the validator's 127 all-zero-trial warnings.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (ragged, session-absolute seconds) for the units selected by `units/classification == 'good'`, windowed by `BehavioralEvents/go_start_times`. `units/anno_name` supplies the per-neuron brain-region index and `units/is_good_trials` / `units/obs_intervals` supply validity (1-e).

ii.
```python
classes = as_strings(units['classification'])
neuron_idx = np.flatnonzero(classes == 'good')
...
spikes = [np.asarray(units['spike_times'][int(i)], dtype=np.float64) for i in neuron_idx]
rate_cube = bin_spikes(spikes, abs_edges)
```

iii. Step 5 mapping table: "`units.spike_times` → `neural` | Count spikes in 80 non-overlapping 50-ms bins spanning `[-2.5,+1.5)` relative to go; divide by 0.05 to Hz". Step 1 records that the reference `process_one_sess`/`sliding_histogram` likewise "represents each neuron's trial spike times relative to go cue (go cue = 0), then computes firing rate arrays". Spike times are the only neural representation in the file (the data are ephys, so "delta-F/F is not applicable").

## 2-b. How is the `neural` data processed?

i. Spike counts per 50 ms bin, converted to firing rate in Hz by dividing by the bin width. For each neuron the 81 absolute bin edges of all trials are flattened into one monotonically increasing array, `np.searchsorted` gives the running spike count at each edge, and `np.diff` along the trial axis gives per-bin counts. The result is stored as `float32` and re-shaped to one `(n_neurons, 80)` C-contiguous matrix per trial. No smoothing, baseline subtraction, or normalisation is applied.

ii.
```python
def bin_spikes(spike_times, absolute_edges):
    """Vectorized-across-trials exact histogram for each neuron, returning Hz."""
    n_trials, n_edges = absolute_edges.shape
    flat_edges = absolute_edges.ravel()
    if np.any(np.diff(flat_edges) < 0):
        raise ValueError("trial windows overlap or are not chronological")
    rates = np.empty((n_trials, len(spike_times), n_edges - 1), dtype=np.float32)
    for j, spikes in enumerate(spike_times):
        spikes = np.asarray(spikes, dtype=np.float64)
        cumulative = np.searchsorted(spikes, flat_edges, side='left').reshape(n_trials, n_edges)
        rates[:, j, :] = np.diff(cumulative, axis=1) / BIN_S
    return rates
```
```python
neural = [np.ascontiguousarray(rate_cube[i]) for i in range(len(trial_idx))]
```

iii. Step 5 decision 12: "Flatten all trial edge arrays (monotonic because trial windows do not overlap), use `np.searchsorted` into each neuron's absolute spike train, reshape cumulative indices, and difference along each trial. This exactly matches direct histograms without nested trial loops." Decision 8: "float32 Hz minimizes memory while preserving exact integer-count multiples of 20 Hz." Step 10 check 2 verified the vectorized result against direct `np.histogram` on raw NWB data with `np.allclose` for first/middle/last neurons in 12 trials across 4 sessions, including truncated-recording sessions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == 'good'` — the published region-specific spike-sorting QC classifier — are kept (69,453 of 272,227, 25.5%). The broader legacy `unit_quality == 'good'` field (154,948 units) is explicitly rejected. No individual metric thresholds are applied. A session with zero classifier-good units is skipped entirely (one session, whose `classification` is NaN throughout), and the code raises if any retained unit has a blank/`nan` anatomy label. Unit-level recording validity is handled as a *trial* filter rather than by dropping units (1-e).

ii.
```python
classes = as_strings(units['classification'])
neuron_idx = np.flatnonzero(classes == 'good')
if len(neuron_idx) == 0:
    print(f'SKIP {session_id}: zero classifier-good units', flush=True)
    return None
```
```python
regions = as_strings(units['anno_name'])[neuron_idx].tolist()
if any(x in ('', 'nan') for x in regions):
    raise ValueError(f'{session_id}: classifier-good unit lacks anatomy')
```

iii. Step 4: "Unit QC field | `qc_mode='classifier'` | Both broad `unit_quality` and stricter `classification` exist | Paper says classifier-good units were analyzed | Use `classification == 'good'`; all 69,453 such units have anatomy." Step 3: "Use the published region-specific logistic-regression classifier output... Do not substitute the broader legacy `unit_quality == 'good'`; the classifier labels are explicitly those used for the paper analyses." Step 9 compares 69,453 against the paper's 69,943 and declines to force agreement: "Treat as an archive-version/paper-count difference...; use the labels actually stored in the supplied NWBs. Do not manipulate QC to force a paper total." The AI also decided *not* to apply the method paper's "<10 neurons per area/session" rule because "that rule was for area-level statistical analyses, whereas dropping them would discard valid neural predictors."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to go-cue onset. All NWB streams share one session-absolute clock, so alignment reduces to adding the fixed relative edge grid to each trial's go timestamp; spikes are then binned directly against those absolute edges. No resampling, interpolation, or per-stream offset correction is used. The trial-to-go mapping is positional row *i* ↔ event *i*, asserted at runtime.

ii.
```python
OFF_START, OFF_END, BIN_S = -2.5, 1.5, 0.05
EDGES_REL = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
...
abs_edges = go[:, None] + EDGES_REL[None, :]
abs_centers = go[:, None] + CENTERS_REL[None, :]
```
```python
'temporal_alignment_event':'go cue onset (BehavioralEvents/go_start_times)',
'off_start':OFF_START,'off_end':OFF_END,
```

iii. Step 5: "`BehavioralEvents.go_start_times` → alignment | Trial-row-matched absolute go timestamp | Exactly one per trial and all lie within corresponding trial." Step 10 check 10: "Bin edges are exactly 81 values on `[-2.5,1.5]`, yielding 80 half-open bins and centers `[-2.475,1.475]`. Direct first/last trial and first/middle/last neuron histograms pass."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins, 80 non-overlapping half-open bins spanning `[-2.5, +1.5)` s relative to the go cue, identical for every trial and session. The edge grid is built once at module scope and reused. This deliberately replaces the reference code's 100 ms sliding window with 50 ms stride; only the 50 ms sampling grid is preserved. No further rebinning/downsampling occurs — spikes are histogrammed once directly into the final bins, and the camera stream is sampled once at the same bin centres. `metadata['time_bin_size']` is 50.0 ms.

ii.
```python
OFF_START, OFF_END, BIN_S = -2.5, 1.5, 0.05
EDGES_REL = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
N_TIME = 80
```
```python
'time_bin_size':50.0,
'interval_convention':'[-2.5, 1.5) s; values at 50-ms bin centers',
```

iii. Step 1: "The decoder task mandates 50-ms-width bins and `[-2.5, +1.5] s`; therefore non-overlapping 50-ms bins will supersede the reference 100-ms sliding analysis window while preserving 50-ms temporal sampling." Step 4: "Neural bins | 100-ms window with 50-ms stride | ... | Decoder mandate takes precedence: use non-overlapping 50-ms bins, expressed as Hz, over exactly `[-2.5, +1.5)`."

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `BehavioralEvents/sample_start_times` (the auditory sample/tone onsets), disambiguated using `sample_stop_times`, `delay_start_times`, `delay_stop_times`, `trials.start_time`, and the trial's go cue. Rather than taking a positional or nearest-nominal sample event, the AI walks the task state chain: pick the delay state whose *stop* is nearest the go cue, then pick the sample state whose *stop* is nearest that delay's start, and take that sample's start as the tone onset.

ii.
```python
def choose_tone_onsets(trial_starts, go_times, sample_starts, sample_stops,
                       delay_starts, delay_stops):
    """Choose the auditory sample state that leads into the final pre-go delay.

    Early-lick trials can restart sample states several times. The trial stimulus is
    the final 0.65-s sample state nearest the delay state whose stop is the go cue.
    """
    out = np.empty(len(go_times), dtype=np.float64)
    for i, (start, go) in enumerate(zip(trial_starts, go_times)):
        didx = np.flatnonzero((delay_starts >= start) & (delay_starts <= go))
        sidx = np.flatnonzero((sample_starts >= start) & (sample_starts < go))
        if len(didx) == 0 or len(sidx) == 0:
            raise ValueError(f"trial {i} lacks sample or delay state before go")
        dk = didx[np.argmin(np.abs(delay_stops[didx] - go))]
        sk = sidx[np.argmin(np.abs(sample_stops[sidx] - delay_starts[dk]))]
        out[i] = sample_starts[sk]
    return out
```

iii. Step 4: "Tone onset | Reference epochs assume nominal sample near -1.85 s | Repeated/aborted sample states create extra timestamps in 5,534 trials | ... | Select the 0.65-s sample state whose stop is nearest the final delay start, where the final delay is identified by its stop at go. This resolves repeated early-lick sample states and variable delay durations." The AI first used a nearest-to-`go−1.85 s` rule, found in Step 10 that it "differed from the final sample→delay state in 183 trials (typically valid 0.3-s rather than 1.2-s delays)", and switched to the state-chain rule because it is "semantically correct for 'tone onset'". It also verified "every selected auditory sample is exactly 0.65 s long".

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A continuous, time-varying value: for each trial, the absolute bin centre minus that trial's tone onset, i.e. seconds elapsed since the tone at the centre of each 50 ms bin. Stored as `float32` in row 0 of the `(2, 80)` input array under the name `'time from tone onset (s)'`. Values are negative before the tone. Converted range over the full dataset is `[-1.5, 11.9]` s; the large upper tail comes from trials with repeated/aborted sample states.

ii.
```python
# Inputs: continuous time since tone, binary photostimulation at bin center.
tone_time = abs_centers - tone[:, None]
...
inputs = [np.stack((tone_time[k].astype(np.float32), stim[k])) for k in range(len(trial_idx))]
```
```python
'input_names':['time from tone onset (s)','photostimulation on'],
```

iii. Step 5 mapping: "for each bin center store continuous seconds since tone onset: `absolute_bin_center − tone_onset`". The Decoder Task specifies this input as "continuous, time-varying", so no discretisation is applied. Step 10/12 investigated the extreme values: "Raw event inspection confirms repeated sample states are retries caused by early licks... rare extreme tone-time values indicate malformed or interrupted sequences"; after switching to the state-chain rule these were kept as "genuine selected within-trial events" rather than excluded, since early-lick trials are required outputs.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on exactly the same grid as the firing rates: `abs_centers = go[:, None] + CENTERS_REL[None, :]`, where `CENTERS_REL` are the midpoints of the same 81 edges used to bin spikes. Bin *k* of the input therefore describes the centre of the same interval that bin *k* of the neural matrix counts spikes over. No interpolation or offset correction is needed because tone events and spikes share the session-absolute clock.

ii.
```python
abs_edges = go[:, None] + EDGES_REL[None, :]
abs_centers = go[:, None] + CENTERS_REL[None, :]
...
tone_time = abs_centers - tone[:, None]
```

iii. Step 5 decision 7: "Bin edges are `[-2.5,-2.45,...,+1.5]`; values correspond to centers `[-2.475,...,+1.475]`. Metadata records the half-open interval." Step 10 check 3 independently reconstructed absolute bin centres and the raw sample→delay→go chains from the NWB files and confirmed the stored tone-time arrays with `np.allclose`.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From the event streams `BehavioralEvents/photostim_start_times` and `photostim_stop_times`, attributed to a trial by requiring the stimulation onset to fall within `[trials.start_time, trials.stop_time]`. The trials-table columns `photostim_onset` / `photostim_duration` (strings, `'N/A'` when absent) were examined but not used as the timing source. 18,588 stimulation events exist; each stimulated trial has exactly one, and the code raises if a trial ever contains more than one.

ii.
```python
ps = np.asarray(ev['photostim_start_times'].timestamps[:], dtype=np.float64)
pe = np.asarray(ev['photostim_stop_times'].timestamps[:], dtype=np.float64)
for k, src_i in enumerate(trial_idx):
    a, b = starts[src_i], float(trials['stop_time'][src_i])
    hits = np.flatnonzero((ps >= a) & (ps <= b))
    if len(hits) > 1:
        raise ValueError(f'{session_id} trial {src_i}: multiple photostim intervals')
```

iii. Step 5 mapping: "`photostim_start_times/stop_times` → `input[1]` | Binary at each bin center: 1 iff center is inside the trial's 0.5-s photostimulation interval | 18,588 trials have exactly one interval." The trajectory records the reason for preferring the event streams: "Photostimulation table onsets are session-relative absolute times, so the event timestamps are the safest way to construct the binary time series" (the table field is a string column mixing `'N/A'` with numeric text, which had already broken one of the AI's analysis scripts). Step 4 also notes the AI chose measured event times over protocol assumptions: "NWB event timestamps/power provide the precise time-varying input rather than assuming protocol timing."

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1) time-varying series, not a per-trial flag: a bin is 1 iff its centre falls in `[photostim_start, photostim_stop)`. Non-stimulated trials keep the pre-allocated all-zero row. Stored as `float32` in row 1 of the `(2, 80)` input array. ~20.0 % of retained trials contain at least one stimulated bin; the validator reports the input range as `[0.0, 1.0]`.

ii.
```python
stim = np.zeros((len(trial_idx), N_TIME), dtype=np.float32)
...
    if len(hits) == 1:
        h = hits[0]
        stim[k] = ((abs_centers[k] >= ps[h]) & (abs_centers[k] < pe[h])).astype(np.float32)
```

iii. Step 5 decision 11: "Photostimulation bin state: Evaluate at bin centers, consistent with representing each 50-ms bin by one timepoint." The Decoder Task requires "Whether photostimulation is on at every time point (discrete, time-varying)". Step 10 check 3 verified the binary vectors against raw start/stop intervals with `np.allclose`. Step 3 notes photoinhibition was "typically during the final 0.5 s of delay and ended before go cue", i.e. inside the `[-2.5, 1.5)` window.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Via the same absolute bin centres used for the neural bins (`abs_centers`), compared directly against absolute photostimulation event timestamps. Both are on the session clock, so no shifting or interpolation is required; bin *k* of the photostim row covers the same interval as bin *k* of the rate matrix.

ii.
```python
abs_centers = go[:, None] + CENTERS_REL[None, :]
...
stim[k] = ((abs_centers[k] >= ps[h]) & (abs_centers[k] < pe[h])).astype(np.float32)
```

iii. Step 4/Step 5: all streams "share the same session clock. Go events map one-to-one by trial row; video and spike streams will be indexed by absolute time around each go." No separate alignment logic was considered necessary; the `--show-processing` plots overlay the photostim trace on the same go-relative axis as the population rate to make this visually checkable.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no choice column in the NWB file, so choice is derived from `trials.trial_instruction` (`'left'`/`'right'`) crossed with `trials.outcome` (`'hit'`/`'miss'`/`'ignore'`): a hit means the animal licked the instructed side, a miss means it licked the opposite side, and an ignore means it did not lick. Lick event streams (`left_lick_times`/`right_lick_times`) were considered and rejected as the primary source.

ii.
```python
def trial_choice(instruction, outcome):
    if outcome == 'ignore':
        return 2
    if outcome == 'hit':
        return 0 if instruction == 'left' else 1
    if outcome == 'miss':
        return 1 if instruction == 'left' else 0
    raise ValueError(f"unknown outcome {outcome!r}")
```
```python
instruction = as_strings(trials['trial_instruction'])
outcome = as_strings(trials['outcome'])
```

iii. Trajectory step 31: "Post-go lick events cannot directly define the requested three-way choice because miss trials can contain licks and instruction is not behavioral choice; the NWB outcome plus instructed side implies choice for hit/miss, while ignore means no lick. Specifically, hit maps to instructed direction, miss maps to the opposite direction, and ignore maps to no lick." Step 5 mapping repeats this and Step 10 check 4 verified the derived labels against raw instruction/outcome with `np.allclose`.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded `0 = left`, `1 = right`, `2 = no lick`, written into row 0 of an integer `(4, 80)` per-trial output array and **repeated across all 80 bins**, because the decoder indexes every output dimension at every timepoint and a single array cannot mix scalar and time-varying rows. `output_values[0] = ['left','right','no lick']`. Full-dataset distribution: left 0.429, right 0.422, no lick 0.149.

ii.
```python
outputs = []
for k, src_i in enumerate(trial_idx):
    arr = np.empty((4, N_TIME), dtype=np.int64)
    arr[0] = trial_choice(instruction[src_i], outcome[src_i])
    ...
```
```python
'output_names':['lick direction choice','outcome','early lick','tongue y-position'],
'output_values':[['left','right','no lick'],...],
```

iii. Step 5 decision 6: "Use integer `(4,80)` arrays. Choice/outcome/early-lick are repeated per-trial labels, while tongue category varies in time. This mixed representation is supported by the decoder and keeps all outputs aligned." Trajectory step 34: "The decoder explicitly supports per-trial output vectors `(doutput,)`... However, tongue-y is time-varying while the other three outputs are per-trial, and a single trial output array cannot mix scalar and temporal rows unless all four are represented `(4, 80)`."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `trials.outcome` column, which already contains exactly the three requested categories `'ignore'`, `'miss'`, `'hit'` (raw census: hit 65,254; miss 15,641; ignore 14,095). No derivation from lick or reward streams.

ii.
```python
outcome = as_strings(trials['outcome'])
```

iii. Step 5 mapping: "`trials.outcome` → `output[1]` | ignore=0, miss=1, hit=2; repeat over bins | Required order matches task wording." Step 4 maps this onto the reference code's legacy `correctness` field (`1` correct/free water, `0` error, `-1` no response) and confirms the NWB strings carry the same information in a directly usable form.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A fixed dictionary maps `ignore → 0`, `miss → 1`, `hit → 2` (the order given in the Decoder Task), written into row 1 of the `(4, 80)` array and repeated across all 80 bins. Full-dataset distribution: ignore 0.149, miss 0.166, hit 0.685.

ii.
```python
omap = {'ignore': 0, 'miss': 1, 'hit': 2}
...
    arr[1] = omap[outcome[src_i]]
```

iii. Same as 5-b: codes follow the Decoder Task ordering and the value is per-trial, so it is broadcast across bins so all four outputs share one time-varying array. Step 12 check 6 confirmed the class balance is not degenerate ("no output is 99% one class").

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `trials.early_lick` column, whose values are the strings `'no early'` and `'early'` (raw census: no early 84,185; early 10,805). Nothing is re-derived from lick timestamps.

ii.
```python
early = as_strings(trials['early_lick'])
```

iii. Step 5 mapping: "`trials.early_lick` → `output[2]` | no early=0, early=1; repeat over bins | Retained as requested target rather than filtered out." Step 1 records that the reference `get_regular_trial_mask` *excludes* early-lick trials; the AI keeps them because the Decoder Task requires predicting the flag. The early-lick event itself occurs during sample/delay, i.e. inside the −2.5 s window.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Binary encoding `'no early' → 0`, anything else → 1, written into row 2 of the `(4, 80)` array and repeated across all bins. `output_values[2] = ['no','yes']`. Full-dataset distribution: no 0.884, yes 0.116.

ii.
```python
    arr[2] = 0 if early[src_i] == 'no early' else 1
```

iii. As above; codes follow the Decoder Task ("no, yes"). Step 12 specifically re-audited this output because its decoder accuracy was closest to the 1.5× chance review threshold: "Raw NWB labels were checked with `np.allclose()` on specific trials and exactly match converted repeated labels. Class distribution is 88.35% no / 11.65% yes, providing 10,377 positive trials—not a 99% dominant-class artifact... Changing labels/filtering to inflate accuracy would reduce fidelity to the source data."

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']`, a ~294 Hz stream whose `data` is `(n_frames, 3)` = `(tongue_x, tongue_y, tongue_likelihood)` with explicit `timestamps`. Column 1 provides the y value; column 2 (tracker likelihood) determines visibility. The stream exists in all 174 sessions.

ii.
```python
tongue = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
video_t = np.asarray(tongue.timestamps[:], dtype=np.float64)
video_d = np.asarray(tongue.data[:], dtype=np.float64)
visible_session = video_d[:, 2] >= VISIBILITY_THRESHOLD
```

iii. Step 2: "Data shape is `(video frames, 3)` with columns `(tongue_x, tongue_y, tongue_likelihood)`, explicit timestamps, arbitrary spatial units, and median frame interval about 3.4 ms (~294 Hz). Coordinates are finite even when tracking confidence is tiny; visibility must therefore use the likelihood channel." Step 3 links this to the method paper's DeepLabCut-style marker tracking of "tongue, jaw, and nose".

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Three steps. (1) A per-session visibility mask `likelihood >= 0.9` is computed over the whole session's frames. (2) The 40th and 60th percentiles of `tongue_y` are taken **over the visible frames of that session only** (low-confidence placeholder coordinates are excluded from the threshold estimate). (3) For each 50 ms bin, the single video frame nearest the bin centre is selected (`nearest_indices`, a `searchsorted` + left/right comparison), and its y and likelihood are used to assign a class. No averaging or smoothing within a bin is done — the bin is represented by one frame. The session's `q40`/`q60` are recorded in `metadata['session_info']`. The code raises if a session has no visible frames.

ii.
```python
VISIBILITY_THRESHOLD = 0.9
...
def nearest_indices(sorted_times, query):
    idx = np.searchsorted(sorted_times, query)
    idx = np.clip(idx, 1, len(sorted_times) - 1)
    left = idx - 1
    use_left = np.abs(query - sorted_times[left]) <= np.abs(sorted_times[idx] - query)
    return np.where(use_left, left, idx)
```
```python
visible_session = video_d[:, 2] >= VISIBILITY_THRESHOLD
if not np.any(visible_session):
    raise ValueError(f'{session_id}: no visible tongue frames')
q40, q60 = np.percentile(video_d[visible_session, 1], [40, 60])
frame_idx = nearest_indices(video_t, abs_centers.ravel()).reshape(len(trial_idx), N_TIME)
ty = video_d[frame_idx, 1]; tl = video_d[frame_idx, 2]
```

iii. Step 4: "Tongue visibility | ... no explicit confidence threshold found in supplied code | Tongue stream has `(x,y,likelihood)` and strongly bimodal likelihood | ... | Use likelihood ≥0.9 as visible. Threshold 0.95 gives nearly identical fractions, so 0.9 is conservative without materially changing labels." Trajectory step 29 adds: "0.9 is the standard conservative DLC threshold and is appropriate absent a repository-specific threshold." Step 5 decision: "Percentiles use all visible frames in the session, not low-confidence placeholder coordinates", because (trajectory step 32) "Visible-only tongue percentiles are clearly necessary because low-confidence placeholder coordinates substantially distort thresholds." Session-wide visible fraction is 9.7–16.2 %, which the AI checks is consistent with the tongue only being tracked during protrusion.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four classes, per session: `0` if visible and `y < q40`; `1` if visible and `q40 <= y <= q60`; `2` if visible and `y > q60`; `3` ("not visible") if the selected frame's likelihood is below 0.9. The array is pre-filled with 3 so not-visible is the default. Full-dataset distribution: 0.062 / 0.032 / 0.065 / 0.840, i.e. among visible bins roughly the intended 40/20/40 split.

ii.
```python
tongue_class = np.full((len(trial_idx), N_TIME), 3, dtype=np.int64)
vis = tl >= VISIBILITY_THRESHOLD
tongue_class[vis & (ty < q40)] = 0
tongue_class[vis & (ty >= q40) & (ty <= q60)] = 1
tongue_class[vis & (ty > q60)] = 2
```
```python
'output_values':[...,['below 40th percentile','40th to 60th percentile','above 60th percentile','not visible']],
```

iii. Directly implements the Decoder Task's per-session discretisation spec (0: <40th pct, 1: 40–60th, 2: >60th, 3: not visible). Step 10 warning resolution: "High tongue not-visible frequency is a meaningful category, not missing data: low-likelihood placeholder coordinates are explicitly encoded as class 3." Step 7 review of the `--show-processing` plots reports "raw nearest-frame tongue y, likelihood threshold, percentile thresholds, and final category are shown on a common go-aligned x-axis... No temporal discontinuity or category/threshold inconsistency was observed."

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. This is the only genuinely time-varying output. Camera timestamps are on the same session-absolute clock as spikes and events, so for each of the 80 go-relative bin centres of each trial the nearest camera frame in absolute time is looked up by binary search (vectorised over all trial × bin centres at once). Class *k* of the tongue row therefore describes the centre of the same interval as bin *k* of the neural matrix. There is no interpolation and no maximum-distance guard: if a bin centre falls inside a video gap, the nearest available frame is used regardless of how far away it is.

ii.
```python
frame_idx = nearest_indices(video_t, abs_centers.ravel()).reshape(len(trial_idx), N_TIME)
ty = video_d[frame_idx, 1]; tl = video_d[frame_idx, 2]
```

iii. Step 5 mapping: "Nearest video frame to each bin center". Step 4/Step 9: "Neural spikes, behavioral state events, trial labels, and video coordinates share the same session clock... video and spike streams will be indexed by absolute time around each go." Step 10 check 4 independently recomputed "nearest-frame tongue likelihood/y with raw session percentiles" from the raw NWB files and confirmed the converted classes with `np.allclose`; Step 12 check 5 cites the processing plots as evidence of synchronisation.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several distinct cases, handled by exclusion, explicit mapping, or a dedicated category — never by silent imputation:
- **Session never quality-controlled** (`classification`/`anno_name` are NaN): `as_strings` renders them `'nan'`, no unit matches `'good'`, and the session is skipped with a printed message (1 session).
- **Trials outside the ephys recording block** and units whose `is_good_trials` vector is shorter than the trial table (3,401 units in 8 sessions): flags are mapped onto the trials whose go cue lies inside that unit's `obs_intervals`; uncovered trials are marked invalid. The code raises if the mapping is not exact.
- **Residual all-zero trials** at recording boundaries: removed by an explicit safeguard.
- **Retracted/untracked tongue**: frames below the likelihood threshold are not imputed; the bin is labelled with the explicit `'not visible'` class.
- **Sessions/trials left too small**: a session with <2 usable trials is skipped.
- **Structural anomalies fail loudly** rather than being papered over: go/trial count mismatch, non-monotonic trial edges, >1 photostim interval in a trial, a trial with no sample or delay state before go, a classifier-good unit with no anatomy, and a session with no visible tongue frames all raise `ValueError`. A final `validate()` re-checks every session's shapes, dtypes, finiteness, and categorical ranges before pickling.

ii.
```python
if len(neuron_idx) == 0:
    print(f'SKIP {session_id}: zero classifier-good units', flush=True); return None
...
if len(represented_idx) != len(stored):
    raise ValueError(f'{session_id} unit {ui}: cannot map {len(stored)} validity flags '
                     f'to {len(represented_idx)} observation-covered trials')
...
nonzero_neural = np.any(rate_cube != 0, axis=(1, 2))
...
tongue_class = np.full((len(trial_idx), N_TIME), 3, dtype=np.int64)
```
```python
def validate(data):
    ...
    for n, x, y in zip(data['neural'][s], data['input'][s], data['output'][s]):
        assert n.shape == (nn, N_TIME) and n.dtype == np.float32 and np.isfinite(n).all()
        assert x.shape == (2, N_TIME) and x.dtype == np.float32 and np.isfinite(x).all()
        assert y.shape == (4, N_TIME) and np.issubdtype(y.dtype, np.integer)
        assert set(np.unique(y[0])).issubset({0,1,2})
        ...
```

iii. Step 5 decision 13 and Step 10 "Issues Found and Resolved": "Initial right-padding assumed all recordings started at trial 0. Raw `obs_intervals` proved one session started at source trial 125. Tested every short-mask unit: observation-overlapping go-cue trial count exactly matched stored mask length with zero failures." The guiding principle is stated as "never interpret uncovered trials as recorded zero-spike data" and, for the tongue, "low-likelihood placeholder coordinates are explicitly encoded as class 3." The fail-fast assertions were credited during Step 6: "The script correctly stopped rather than silently misaligning data."

## 10-a. What are the most time-consuming steps of the code?

i. Full conversion of 173 sessions took 198.8 s wall clock (185.6 s compute + 13.2 s pickling an 11.0 GiB file), i.e. ~1.1 s/session, range ~0.5–2.1 s, scaling roughly with unit count. Within a session the dominant costs are NWB I/O and binning: reading each good unit's `spike_times` (one ragged read per unit, up to ~900 units/session), the per-neuron `np.searchsorted` over 81 × n_trials edges, reading the full `(n_frames, 3)` camera array (0.4–1.3 M frames) and its timestamps, and the per-unit `is_good_trials` reads. The AI printed per-session timing to the conversion log and recorded it in `metadata['session_info']['elapsed']`.

ii.
```python
def process_file(path, make_plot=False):
    t0 = time.perf_counter()
    ...
        elapsed = time.perf_counter() - t0
        print(f'{session_id}: {len(trial_idx)}/{len(trials)} trials, {len(neuron_idx)} units, '
              f'visible={visible_session.mean():.3f}, q=({q40:.2f},{q60:.2f}), {elapsed:.2f}s', flush=True)
```
```python
print(f'Wrote {out} ({out.stat().st_size/2**30:.3f} GiB) in {time.perf_counter()-t:.2f}s; total {time.perf_counter()-wall:.2f}s',flush=True)
```

iii. Step 7 estimated "approximately 5–7 minutes, below 15-minute optimization threshold" from the 2-session sample; Step 9 reports the actual "Full conversion runtime: 194.05 s (including 13.57 s serialization), substantially below the 5–7 minute conservative estimate", so no further optimisation was pursued. Step 6 notes the memory/IO shape: "Loading full video arrays and rates creates substantial but manageable per-session memory use; objects are released when each NWB context closes."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The dominant work is already vectorised (spike binning is vectorised across all trials at once via one flattened edge array; nearest-frame video lookup is vectorised across all trial × bin centres at once). Three Python-level loops remain that could be collapsed:
- `choose_tone_onsets` loops over every trial and runs two `np.flatnonzero` scans over the *entire* session's sample/delay arrays each iteration — O(n_trials × n_events). A `np.searchsorted` of the go times into the sorted `sample_start_times` would give the same answer in one vectorised call (verified: the two give identical tone onsets).
- The photostim loop likewise does `np.flatnonzero((ps >= a) & (ps <= b))` over all stimulation events per trial, and re-reads `trials['stop_time'][src_i]` from the HDF5 file once per trial. `searchsorted` on `ps` plus a single bulk read of `stop_time` would vectorise it.
- The output-assembly loop builds one `(4, 80)` array per trial in Python; it could be one `(n_trials, 4, 80)` array with broadcast assignment.
- The `for ui in neuron_idx` validity loop does one ragged HDF5 read per good unit (and, for short masks, an inner loop over `obs_intervals`). The per-neuron loop inside `bin_spikes` is genuinely irreducible because `spike_times` is ragged.

ii.
```python
for i, (start, go) in enumerate(zip(trial_starts, go_times)):
    didx = np.flatnonzero((delay_starts >= start) & (delay_starts <= go))
    sidx = np.flatnonzero((sample_starts >= start) & (sample_starts < go))
```
```python
for k, src_i in enumerate(trial_idx):
    a, b = starts[src_i], float(trials['stop_time'][src_i])
    hits = np.flatnonzero((ps >= a) & (ps <= b))
```
```python
for ui in neuron_idx:
    stored = np.asarray(units['is_good_trials'][ui], dtype=bool)
```

iii. Step 6 "Code speedups added": "Spike counts use `np.searchsorted` over one flattened, monotonic trial-edge array per neuron, vectorizing across trials. Inputs/outputs and nearest video-frame assignment are vectorized over all trial-bin centers." Step 6 "Code inefficiencies identified": "A nested trial × neuron histogram would require billions of Python-level operations." The remaining loops were not revisited because the measured runtime (198.8 s) was already far under the instructions' 15-minute budget, so there was no trigger to optimise further.

## 10-c. What processing does the code repeat multiple times?

i. Little is genuinely recomputed, but there are a few small redundancies:
- `go_start_times` is read twice from the file — once as `all_go_for_validity` for the validity mapping and again as `all_go` a few lines later — and `all_go_for_validity` is loaded even in the majority of sessions where every validity vector is full-length and it is never used.
- `trials['stop_time'][src_i]` is fetched from HDF5 once per trial inside the photostim loop instead of being read once as an array (`start_time` *is* read in bulk).
- `units['is_good_trials'][ui]` is read individually for every classifier-good unit, and `units['obs_intervals'][ui]` again per unit for short-mask sessions.
- Rates are computed for all trials and only afterwards is the all-zero subset dropped, so a small number of discarded trials are binned.
- `as_strings` re-materialises each string column; the bin grid, by contrast, is built once at module scope and reused for every trial and session, and each NWB file is opened exactly once.

ii.
```python
all_go_for_validity = np.asarray(
    nwb.acquisition['BehavioralEvents'].time_series['go_start_times'].timestamps[:],
    dtype=np.float64)
...
all_go = np.asarray(ev['go_start_times'].timestamps[:], dtype=np.float64)
```
```python
EDGES_REL = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
```

iii. Step 6 lists "Avoid unnecessary file I/O" as a goal and the AI's own accounting focuses on the large arrays: "objects are released when each NWB context closes." The duplicated go-cue read is an artefact of the Step 10 iteration that inserted the `obs_intervals`-based validity mapping above the main event-loading block; it was not flagged in the notes. Because the per-session cost is dominated by the spike and video buffers, these repeats are negligible and the AI's overall claim of a single pass per file holds.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Essentially none. Every computed field is written to the output dictionary and every one of the 2 inputs / 4 outputs / neural / subject / region fields is consumed by `train_decoder.py`. The minor exceptions are diagnostic rather than analytic: per-session `q40`, `q60`, `elapsed`, `source_file` and `source_trial_idx` are computed and stored in `metadata['session_info']` purely for provenance/auditing; `ty`/`tl` (the nearest-frame y and likelihood) are materialised in full but are needed for classification and then reused by the plotting routine; the monotonicity check in `bin_spikes` and the whole `validate()` pass are pure verification; and, as noted in 10-c, the handful of trials later removed for being all-zero are binned before they are dropped. Plotting only runs under `--show-processing`.

ii.
```python
'session_info':[{k:r[k] for k in ['session_id','source_file','source_trial_idx','q40','q60','elapsed']} for r in results],
```
```python
validate(data)
print(f'Validated {len(results)} sessions, {sum(map(len,data["neural"]))} trials, '
      f'{sum(len(x) for x in data["brain_region_idx"])} session-neurons',flush=True)
```

iii. Step 6: "Internal assertions validate all session/trial dimensions, finite values, dtypes, and categorical ranges before serialization." The Target Data Format requires `brain_regions`/`brain_region_idx` and metadata even though the decoder does not train on them, and the instructions explicitly ask for sanity checks and timing instrumentation, so the AI treats the provenance fields and validation pass as required deliverables rather than waste.
