# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the AllenSDK project cache. It first tried
`VisualBehaviorOphysProjectCache.from_s3_cache` / `from_local_cache`, but the local
data release in `/app/data` stores the manifest one directory above the cache root, so
the SDK raised `ValueError: max() iterable argument is empty`. The AI therefore switched
to reading the flat metadata CSV `project_metadata/ophys_experiment_table.csv` with
pandas, and loading each experiment directly from its NWB file with
`BehaviorOphysExperiment.from_nwb_path()`. Every experiment listed in the CSV whose NWB
file exists on disk, and whose `session_type` does not contain the string `passive`, is
loaded. Loading happens **twice**: once in `collect_global_info()` (to gather the global
image-name vocabulary and the global running/pupil percentile bin edges) and again in
`process_experiment()` (to build the trials). 202 experiments were retained in the full
run (log: `processed 202/202 ... saved converted_data.pkl with 202 sessions in 2680.74s`).

ii.
```python
DATASET_DIR = Path('data/visual-behavior-ophys-1.1.0')
META_DIR = DATASET_DIR / 'project_metadata'
EXPT_TABLE = META_DIR / 'ophys_experiment_table.csv'
EXPT_DIR = DATASET_DIR / 'behavior_ophys_experiments'

def load_experiment_table():
    return pd.read_csv(EXPT_TABLE)

def get_nwb_path(exp_id: int) -> Path:
    return EXPT_DIR / f'behavior_ophys_experiment_{int(exp_id)}.nwb'

def load_experiment(exp_id: int) -> BehaviorOphysExperiment:
    return BehaviorOphysExperiment.from_nwb_path(str(get_nwb_path(exp_id)))
```
```python
exp_table = load_experiment_table()
exp_table = exp_table[~exp_table['session_type'].astype(str)
                      .str.contains('passive', case=False, na=False)].copy()
chosen_ids, image_names, image_to_idx, run_edges, pupil_edges = collect_global_info(exp_table)
...
for i, exp_id in enumerate(chosen_ids, 1):
    sess = process_experiment(exp_id, row_lookup[exp_id], run_edges, pupil_edges,
                              image_to_idx, blank_idx)
```
```python
def collect_global_info(exp_table):
    for _, row in exp_table.iterrows():
        exp_id = int(row['ophys_experiment_id'])
        nwb_path = get_nwb_path(exp_id)
        if not nwb_path.exists():
            continue
        exp = load_experiment(exp_id)
        ...
```

iii. From the trajectory (steps 17–20): *"AllenSDK's local cloud cache expects a specific
manifest naming/discovery pattern inside the cache directory … We have confirmed the
actual experiment NWB files are directly available … This means we do not need the
AllenSDK project cloud-cache wrapper … `BehaviorOphysExperiment.from_nwb_path` exists,
which lets us bypass the broken project-cache manifest path entirely."* CONVERSION_NOTES
Step 3 records this as *"direct NWB loading via `BehaviorOphysExperiment.from_nwb_path` —
avoided broken local-cache manifest path"*. Passive sessions were excluded because
*"Trial outcome is a required decoder output, so passive sessions are likely
incompatible"* (Step 4/5 notes).

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values of the retained experiments, taken from the
experiment metadata CSV. They are sorted as strings to build the `subjects` list, and
each converted session gets an index into that list. 38 subjects resulted.

ii.
```python
subject = str(meta_row.get('mouse_id', 'unknown'))
...
subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
'subjects': subjects,
'subject_idx': np.asarray([subject_to_idx[s['subject']] for s in sessions], dtype=np.int64),
```

iii. CONVERSION_NOTES Step 5 mapping table: *"`metadata.mouse_id` → subjects /
subject_idx: Unique subject list and per-session index. Session order follows converted
experiment order."* `mouse_id` is the SDK's canonical animal identifier.

## 1-c. How are the data split into sessions?

i. **One converted "session" = one ophys *experiment* (one imaging plane)**, not one
`ophys_session_id`. The AI explicitly rejected grouping planes by session. No filter on
`project_code` is applied, so the 45 `VisualBehaviorMultiscope` experiments present in
`/app/data` are included alongside the 239 `VisualBehavior` (single-plane) experiments.
Because a Multiscope session contains up to 8 simultaneously recorded planes, the same
behavioural session is emitted as up to 8 separate "sessions" carrying identical trials,
running, pupil and outcome data (e.g. `951980471, 951980473, 951980475, 951980479,
951980481, 951980484, 951980486` all report `kept_trials=209`; subject 457841 ends up
with 34 "sessions"). The two project codes also have different imaging rates (measured
here: 32.3 ms/frame single-plane vs 93.2 ms/frame Multiscope), so the delivered dataset
mixes two time-bin sizes while `metadata['time_bin_size']` stores a single median value.
Sessions with fewer than 2 retained trials are dropped.

ii.
```python
def process_experiment(exp_id, meta_row, run_edges, pupil_edges, image_to_idx, blank_idx):
    exp = load_experiment(int(exp_id))
    ...
    region = str(meta_row.get('targeted_structure', 'unknown'))
    subject = str(meta_row.get('mouse_id', 'unknown'))
```
```python
for i, exp_id in enumerate(chosen_ids, 1):
    sess = process_experiment(exp_id, row_lookup[exp_id], ...)
    if sess['n_trials'] >= 2:
        sessions.append(sess)
```
```python
data = {
    'neural': [s['neural'] for s in sessions],   # one entry per *experiment*
    ...
}
```

iii. CONVERSION_NOTES Step 4 discrepancy table: *"Use **ophys experiment** as one
converted session because each experiment has one coherent neuron set and one brain
region/depth assignment; preserve subject mapping across experiments."* Step 5 Key
Decision 1 repeats this. The notes never discuss the consequence for multi-plane
(Multiscope) sessions, and `project_code` is never mentioned or filtered anywhere in the
notes or the code.

## 1-d. How are the data split into trials?

i. Trials come from the SDK `trials` table. A trial is kept if (`go` OR `catch`) AND NOT
`aborted` AND NOT `auto_rewarded`. The trial window is the full behavioural trial,
`start_time` to `stop_time` (half-open), so trials are variable length and include both
the pre-change flashes and the post-change response window. Neural frames are selected by
a boolean mask on the session's `ophys_timestamps`.

ii.
```python
trials = exp.trials.copy()
keep = trials['go'].fillna(False) | trials['catch'].fillna(False)
keep &= ~trials['aborted'].fillna(False)
keep &= ~trials['auto_rewarded'].fillna(False)
trials = trials[keep].copy()
...
for _, tr in trials.iterrows():
    start = float(tr['start_time'])
    stop = float(tr['stop_time'])
    m = (ophys_timestamps >= start) & (ophys_timestamps < stop)
    if m.sum() < 2:
        continue
    sample_times = ophys_timestamps[m]
    trial_neural = neural[:, m].astype(np.float32)
```

iii. From the trajectory (step 6), after reading `trial.py`: *"aborted trials are
identified by 'abort'; if aborted then go/catch/auto_rewarded are false; otherwise catch
comes from `trial_params['catch']`, auto_rewarded from `trial_params['auto_reward']`, and
go is not catch and not auto_rewarded … This is exactly relevant to the decoder
requirement to include Go and Catch while excluding Aborted and Auto-rewarded."*
CONVERSION_NOTES Step 5 Key Decisions 3–4: *"Segment by trial: Trials are behaviorally
defined in SDK/reference methods; time-varying outputs are then assigned within each
trial"* and *"Include only go and catch trials."*

## 1-e. How are trials filtered based on quality controls?

i. Four filters, at three levels:
- **Session level**: any experiment whose `session_type` contains `passive` is dropped
  before loading (78 of 284 available NWB files); experiments whose NWB file is absent are
  skipped; experiments with fewer than 2 kept trials in `collect_global_info` are not
  selected, and sessions with `n_trials < 2` after processing are not appended.
- **Trial level**: aborted / auto-rewarded trials excluded (see 1-d); trials that contain
  fewer than 2 ophys frames are skipped.
- No requirement that `change_time` be non-null, and no other behavioural QC (e.g. no
  engagement / reward-rate filter).

ii.
```python
exp_table = exp_table[~exp_table['session_type'].astype(str)
                      .str.contains('passive', case=False, na=False)].copy()
```
```python
nwb_path = get_nwb_path(exp_id)
if not nwb_path.exists():
    continue
...
if keep.sum() < 2:
    continue
chosen_ids.append(exp_id)
```
```python
if m.sum() < 2:
    continue
...
if sess['n_trials'] >= 2:
    sessions.append(sess)
```

iii. CONVERSION_NOTES Step 3 "Trial curation rules": *"Include GO and CATCH trials.
Exclude aborted trials … and exclude free-reward/auto-rewarded trials."* Step 4: *"Decoder
output requires trial outcome, which is not behaviorally meaningful in passive viewing →
Likely exclude passive sessions."* The ≥2-trial rule follows the target-format statement
*"There needs to be at least two trials within each session in order to evaluate the
decoder performance."* The ≥2-frame rule is not justified in the notes.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `BehaviorOphysExperiment.dff_traces` — the SDK-provided neuropil-corrected ΔF/F traces,
one 1-D array per cell in the `dff` column. `exp.events` is used only as a fallback if
`dff_traces` is empty (never triggered: every line of `conversion_full_out.txt` reports
`signal=dff`). The first implementation used `events`; the AI switched to ΔF/F after the
sample decoder performed poorly.

ii.
```python
def choose_signal(exp):
    dff = exp.dff_traces.copy()
    if isinstance(dff, pd.DataFrame) and len(dff) > 0:
        return 'dff', dff
    events = exp.events.copy()
    return 'events', events

def extract_neural_matrix(signal_df, signal_kind):
    preferred = ['events', 'filtered_events', 'dff']
    col = None
    for c in preferred:
        if c in signal_df.columns:
            col = c
            break
    ...
    arrays = [np.asarray(v, dtype=np.float32) for v in signal_df[col].values]
    n_t = min(len(a) for a in arrays)
    return np.stack([a[:n_t] for a in arrays], axis=0)
```

iii. CONVERSION_NOTES Step 1: *"Neural imaging data are available as SDK-provided
`dff_traces`; this suggests dF/F should not be recomputed from raw fluorescence."*
Step 10 issue log: *"Events-based neural representation produced sparse/all-zero trial
warning and poor sample decoding: resolved by switching to dF/F traces."* Trajectory step
26: *"A likely cause is using sparse events rather than dF/F … dF/F may better carry
stimulus identity in calcium imaging."*

## 2-b. How is the `neural` data processed?

i. Essentially no processing. The per-cell ΔF/F arrays are stacked into an
`(n_neurons, T)` float32 matrix, truncated to the shortest cell trace and to the length of
`ophys_timestamps`, then sliced per trial. No normalisation, smoothing, z-scoring,
baseline subtraction, deconvolution or cross-plane merging is applied (cross-plane merging
is impossible here because each plane is treated as its own session).

ii.
```python
neural = extract_neural_matrix(signal_df, signal_kind)
ophys_timestamps = np.asarray(exp.ophys_timestamps, dtype=float)
n_t = min(neural.shape[1], len(ophys_timestamps))
neural = neural[:, :n_t]
ophys_timestamps = ophys_timestamps[:n_t]
...
trial_neural = neural[:, m].astype(np.float32)
```

iii. CONVERSION_NOTES Step 1: ΔF/F traces are *"already provided rather than computed by
us"*, so the AI treats the SDK output as the finished neural signal. No further
justification is given for omitting normalisation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality control is applied. Every ROI present in `dff_traces` is kept.
The AI noted the whitepaper's ROI classifier/valid-ROI rules but did not implement any
filter, implicitly relying on the released NWB files already containing only valid ROIs.
The only neural-array-level operation that could remove data is the truncation to the
shortest trace / timestamp vector.

ii.
```python
arrays = [np.asarray(v, dtype=np.float32) for v in signal_df[col].values]
n_t = min(len(a) for a in arrays)
return np.stack([a[:n_t] for a in arrays], axis=0)
```
(no ROI filtering code exists)

iii. CONVERSION_NOTES Step 1: *"Cell/ROI curation hooks appear in `cell_specimens.py`
through valid ROI metadata; exact filtering rule still needs confirmation against dataset
contents and methods."* Step 3 records the whitepaper's *"ROIs are labeled with a
multi-label classifier where any ROIs that end up labeled are not considered cell
bodies."* The question is never resolved in later steps — no decision or code follows.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **trial start**: every ophys frame whose timestamp falls in
`[trials.start_time, trials.stop_time)` is taken, in order, as that trial's time axis.
All output streams are then sampled on exactly that same `sample_times` vector, so neural
and outputs share one index by construction. Metadata records
`temporal_alignment_event = 'ophys timestamps within each behaviorally defined trial'`,
`off_start = 0.0`, `off_end = None`.

ii.
```python
m = (ophys_timestamps >= start) & (ophys_timestamps < stop)
if m.sum() < 2:
    continue
sample_times = ophys_timestamps[m]
trial_neural = neural[:, m].astype(np.float32)
trial_output = build_trial_output(tr, sample_times, trial_stim, run_df, eye_df,
                                  run_edges, pupil_edges, image_to_idx, blank_idx)
```
```python
'temporal_alignment_event': 'ophys timestamps within each behaviorally defined trial',
'off_start': 0.0,
'off_end': None,
```

iii. CONVERSION_NOTES Step 4: *"User explicitly requires temporal alignment based on ophys
timestamp → Resample/assign all time-varying streams onto the ophys timestamp grid within
each trial."* Step 5 Key Decision 2: *"Align everything on ophys timestamps."* Step 10
sanity check 2 reports that a raw reconstruction of session 0 / trial 0 ΔF/F matched the
converted trial with `np.allclose`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **No rebinning or resampling of the neural data** — the native ophys frame grid is kept.
The reported bin size is the median over sessions of each session's median inter-frame
interval. Because the AI mixed single-plane (≈32.3 ms) and Multiscope (≈93.2 ms)
experiments, the *actual* bin size is not constant across the delivered sessions even
though a single scalar is stored. Trial lengths in the verification log span 77–389 frames,
consistent with the two different rates.

ii.
```python
'dt_ms': float(np.median(np.diff(ophys_timestamps)) * 1000.0),
...
time_bin_size = float(np.median([s['dt_ms'] for s in sessions])) if sessions else np.nan
...
'time_bin_size': time_bin_size,
```

iii. CONVERSION_NOTES Step 3: *"Neural data time bin: native ophys timestamps."* The AI's
stated rationale is that the task requires ophys-timestamp alignment, so the ophys grid is
the natural bin. The mixture of acquisition rates is never mentioned in the notes or the
trajectory.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. `stimulus_presentations`: the `image_name`, `start_time` and `omitted` columns. A
`stop_time` is synthesised as the **next flash's `start_time`** so each image label also
covers the following grey inter-stimulus interval. Omitted flashes are dropped, so their
intervals fall through to a reserved `blank` class. (The AI did **not** use
`trials.initial_image_name` / `trials.change_image_name`.)

ii.
```python
def ensure_stim_stop_time(stim, use_full_interval=True):
    stim = stim.sort_values('start_time').copy()
    starts = stim['start_time'].to_numpy(dtype=float)
    diffs = np.diff(starts)
    default_interval = float(np.nanmedian(diffs)) if len(diffs) else 0.75
    if use_full_interval:
        next_starts = np.r_[starts[1:], starts[-1] + default_interval]
        stim['stop_time'] = next_starts
        return stim
```
```python
if 'omitted' in stim.columns:
    stim_non_omitted = stim[~stim['omitted'].fillna(False)].copy()
else:
    stim_non_omitted = stim
image_vals = [image_to_idx.get(str(x), blank_idx)
              for x in stim_non_omitted['image_name'].astype(str).values]
```

iii. CONVERSION_NOTES Step 5 mapping: *"`stimulus_presentations.image_name` → output[0]
image identity: For each ophys timepoint within a trial, assign currently displayed
non-grey image identity … Omitted/grey periods need explicit handling; likely separate
`blank/omitted` category."* Step 7: *"Initial implementation produced too many unknown
image_identity bins; fixed by assigning image labels over full image-presentation
intervals rather than only flash durations"* (the unknown fraction dropped from 0.669 to
0.034).

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global vocabulary is built by pooling `image_name` strings over every retained
experiment during the first pass; `'blank'` is prepended so it takes index 0, and the rest
are sorted alphabetically. Each ophys frame is then mapped to the integer code of whatever
image interval contains it, with `blank` as the default for uncovered frames. The
resulting delivered variable has 18 classes: `blank` (3.4 %), 16 real images (`im000` …
`im106`, ~5.8–6.3 % each) and a spurious `'nan'` class (0.000) produced by
`str(np.nan) == 'nan'` for presentations with a missing image name.

ii.
```python
image_names = ['blank'] + sorted(all_images)
image_to_idx = {name: i for i, name in enumerate(image_names)}
```
```python
def interval_assign(sample_times, starts, stops, values, default=-1):
    out = np.full(sample_times.shape, default, dtype=np.int64)
    for s, e, v in zip(np.asarray(starts, float), np.asarray(stops, float),
                       np.asarray(values, np.int64)):
        m = (sample_times >= s) & (sample_times < e)
        out[m] = v
    return out

image_identity = interval_assign(sample_times, stim_non_omitted['start_time'].values,
                                 stim_non_omitted['stop_time'].values, image_vals,
                                 default=blank_idx)
```

iii. Trajectory step 25: the original `default=-1` crashed decoder training with a CUDA
device-side assert on the NLL loss, so *"Patch convert_data.py so `image_identity` always
uses nonnegative class indices by reserving a `blank` class in `output_values[0]` and
using that index as the default."* A global (rather than per-session) vocabulary is used
so codes are consistent across sessions with different image sets (A and B).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is evaluated directly on the trial's `sample_times` (the ophys frame timestamps used
to slice the neural matrix), so it is aligned by construction — one label per neural time
bin, `(5, n_timepoints)` output stacked with the same `n_timepoints` as `neural`. Only
stimulus presentations overlapping the trial window are considered.

ii.
```python
trial_stim = stim[(stim['start_time'] < stop) & (stim['stop_time'] > start)].copy()
trial_output = build_trial_output(tr, sample_times, trial_stim, ...)
...
return np.stack([image_identity, image_change, run_bins, pupil_bins, trial_outcome], axis=0)
```

iii. CONVERSION_NOTES Step 4: *"Resample/assign all time-varying streams onto the ophys
timestamp grid within each trial."* Verification confirms shape consistency
(`Data format is valid, no errors or warnings`).

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `stimulus_presentations.is_change` together with the corresponding `start_time`
(omitted flashes already removed). Because `is_change` is True only for real image
changes, catch trials (sham changes) get an all-zero `image_change` row, matching the
task's semantics without any explicit `go`/`catch` test.

ii.
```python
image_change = np.zeros(sample_times.shape, dtype=np.int64)
if 'is_change' in stim_non_omitted.columns:
    for ct in stim_non_omitted.loc[stim_non_omitted['is_change'].fillna(False),
                                   'start_time'].values:
        idx = np.searchsorted(sample_times, ct, side='left')
        if idx < len(image_change):
            image_change[idx] = 1
```

iii. CONVERSION_NOTES Step 1: *"Stimulus presentations code contains … change markers
(`is_change`)."* Step 5 mapping: *"`stimulus_presentations.is_change` and image identity
transitions → output[1] image change: Binary time series with 1 immediately after an image
identity change, else 0. Must be aligned to ophys timestamps."*

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each change flash in the trial window, `np.searchsorted(..., side='left')` finds the
**first ophys frame at or after the change onset** and that single element is set to 1.
Everything else stays 0. No smoothing, no window, no propagation into the flash or the
following grey period.

ii. See the snippet in 4-a.

iii. The AI's stated reading of the instruction *"Have value of 1 right after a change in
image identity, otherwise 0"* is taken literally as a single-bin impulse at the change
frame. No further justification appears in CONVERSION_NOTES or the trajectory.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Binary, with `output_values = ['no_change', 'change']`. Because only one time bin per
change is labelled 1, the delivered class balance is extremely skewed: the verification log
reports `image_change: {no_change (0.996), change (0.004)}`. The decoder's balanced
accuracy on this variable is 0.6413 (validation).

ii.
```python
'output_values': [
    image_names,
    ['no_change', 'change'],
    ...
]
```

iii. No explicit justification; the AI treated the binary specification as given and did
not revisit the 0.4 % positive rate in the Step 12 accuracy review (which only checked
"above chance").

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same mechanism as image identity: `np.searchsorted` into the trial's `sample_times`
array, which is the ophys frame grid also used to slice `neural`. Alignment error is at
most one ophys frame (≈32 ms single-plane, ≈93 ms Multiscope), always rounding forward.

ii.
```python
idx = np.searchsorted(sample_times, ct, side='left')
if idx < len(image_change):
    image_change[idx] = 1
```

iii. Same as 3-c — everything is computed on the ophys timestamps of the trial.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `BehaviorOphysExperiment.running_speed`, using its `timestamps` and `speed` columns
(with a defensive lookup in case of different capitalisation / column naming).

ii.
```python
def get_running_df(exp):
    run_df = exp.running_speed.copy()
    cols = {c.lower(): c for c in run_df.columns}
    tcol = cols.get('timestamps', 'timestamps')
    scol = cols.get('speed', None)
    if scol is None:
        for c in run_df.columns:
            if 'speed' in c.lower():
                scol = c
                break
    return run_df.rename(columns={tcol: 'timestamps', scol: 'speed'})[['timestamps', 'speed']]
```

iii. CONVERSION_NOTES Step 5 mapping: *"`running_speed` → output[2] running speed."*
Step 3 notes *"Running speed is derived from wheel encoder data with artifact handling and
smoothing/interpolation steps described in the whitepaper"* — i.e. the AI relies on the
SDK's already-processed `running_speed` rather than re-deriving it from the encoder.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Two steps: (1) **nearest-neighbour** resampling of the ~60 Hz speed trace onto the
trial's ophys timestamps (not linear interpolation); (2) digitisation into 5 bins using
globally pre-computed percentile edges. The percentile edges are computed in
`collect_global_info` from the **raw, native-rate, whole-session** speed traces of all
retained experiments concatenated (i.e. including inter-trial intervals and aborted-trial
periods, and counting each Multiscope plane's copy of the same session separately).

ii.
```python
def nearest_assign(sample_times, source_times, source_values):
    idx = np.searchsorted(source_times, sample_times, side='left')
    idx = np.clip(idx, 0, len(source_times) - 1)
    prev_idx = np.clip(idx - 1, 0, len(source_times) - 1)
    choose_prev = np.abs(sample_times - source_times[prev_idx]) < np.abs(sample_times - source_times[idx])
    idx[choose_prev] = prev_idx[choose_prev]
    return source_values[idx]

run_vals = nearest_assign(sample_times, run_df['timestamps'].values, run_df['speed'].values)
run_bins = digitize(run_vals, run_edges)
```
```python
run_vals.append(pd.to_numeric(run_df['speed'], errors='coerce').values)
...
run_edges = make_percentile_bins(np.concatenate(run_vals) if run_vals else np.array([0.0]))
```

iii. CONVERSION_NOTES Step 5 mapping: *"Interpolate/assign onto ophys timestamps, then
discretize globally into 5 equal-percentile bins … Need to decide whether percentile bins
are global across retained dataset or per session; likely global for consistency."*
Key Decision 6: *"Discretize running speed and pupil diameter into 5 percentile bins:
Follow decoder specification; compute cut points on valid retained samples only."*

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four interior percentile cut points at the 20/40/60/80th percentiles of the pooled
distribution, applied with `np.digitize` (right-open), producing labels 0–4 named
`bin_0 … bin_4`. Non-finite values are forced to bin 0. Degenerate (tied) edges are nudged
apart with `np.nextafter`. Because the edges come from the raw native-rate whole-session
distribution but are applied to trial-restricted ophys-rate samples, the delivered bins are
**not** exactly equal-sized: `running_speed_bin: {bin_0 0.170, bin_1 0.172, bin_2 0.193,
bin_3 0.235, bin_4 0.230}`.

ii.
```python
def make_percentile_bins(values, n_bins=5):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    qs = np.linspace(0, 100, n_bins + 1)[1:-1]
    edges = np.percentile(vals, qs)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    return edges

def digitize(values, edges):
    vals = np.asarray(values, dtype=float)
    out = np.digitize(vals, edges, right=False).astype(np.int64)
    out[~np.isfinite(vals)] = 0
    return out
```

iii. Directly follows the Decoder Task instruction *"Running speed, discretized into five
equal percentile bins"*; global edges were chosen *"for consistency"* across sessions
(Step 5 mapping note). The residual bin imbalance is not discussed anywhere.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The speed trace is sampled at exactly the trial's ophys frame timestamps
(`sample_times`), the same vector used to slice the neural matrix, so the two are aligned
row-for-row. Nearest-neighbour matching gives a worst-case offset of half the running
sample interval (~8 ms at 60 Hz).

ii.
```python
run_vals = nearest_assign(sample_times, run_df['timestamps'].values, run_df['speed'].values)
run_bins = digitize(run_vals, run_edges)
...
return np.stack([image_identity, image_change, run_bins, pupil_bins, trial_outcome], axis=0)
```

iii. Same justification as 2-d/3-c: everything is placed on the ophys timestamp grid
within each trial (Step 4 resolution, Step 5 Key Decision 2).

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `BehaviorOphysExperiment.eye_tracking`. The column is chosen by a preference list
`['pupil_diameter', 'pupil_area', 'pupil_width', 'pupil_radius']`; since the SDK table has
no `pupil_diameter`, the selected column is in practice **`pupil_area`**. Rows whose value
is NaN or infinite are dropped before resampling — which implicitly removes blinks, because
the SDK's `filter_on_blinks()` already sets `pupil_area` to NaN wherever `likely_blink` is
True. (The AI never references `likely_blink` explicitly.)

ii.
```python
def pick_pupil_column(eye_tracking):
    candidates = ['pupil_diameter', 'pupil_area', 'pupil_width', 'pupil_radius']
    for c in candidates:
        if c in eye_tracking.columns:
            return c
    for c in eye_tracking.columns:
        if 'pupil' in c.lower() and pd.api.types.is_numeric_dtype(eye_tracking[c]):
            return c
    return None
```
```python
eye_valid = eye_df[['timestamps', pupil_col]].replace([np.inf, -np.inf], np.nan).dropna()
```

iii. CONVERSION_NOTES Step 5 mapping: *"`eye_tracking` pupil diameter field → output[3]
pupil diameter: Interpolate/assign onto ophys timestamps, mask invalid values, discretize
into 5 equal-percentile bins. Must identify exact pupil-diameter column name from
experiment object."* The preference list is the AI's way of resolving that uncertainty at
run time; no note explains why area was accepted as a stand-in for diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Identical pipeline to running speed: drop non-finite rows, nearest-neighbour resample
onto the trial's ophys timestamps, then digitise with globally pre-computed percentile
edges. Global edges are computed from the pooled raw `pupil_area` values of all retained
experiments (note: pooled **before** the NaN drop, since `make_percentile_bins` filters
non-finite values itself). If the experiment has no pupil column or no valid rows at all,
the whole trial is filled with bin 0.

ii.
```python
pupil_col = pick_pupil_column(eye_df)
if pupil_col is None:
    pupil_bins = np.zeros(sample_times.shape, dtype=np.int64)
else:
    eye_valid = eye_df[['timestamps', pupil_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(eye_valid) == 0:
        pupil_bins = np.zeros(sample_times.shape, dtype=np.int64)
    else:
        pupil_vals = nearest_assign(sample_times, eye_valid['timestamps'].values,
                                    eye_valid[pupil_col].values)
        pupil_bins = digitize(pupil_vals, pupil_edges)
```
```python
pupil_vals.append(pd.to_numeric(eye_df[pupil_col], errors='coerce').values)
...
pupil_edges = make_percentile_bins(np.concatenate(pupil_vals) if pupil_vals else np.array([0.0]))
```

iii. Same as 5-b: Step 5 mapping (*"Interpolate/assign onto ophys timestamps, mask invalid
values, discretize into 5 equal-percentile bins"*) and Key Decision 6.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same `make_percentile_bins` / `digitize` pair as running speed: 20/40/60/80th
percentiles of the pooled raw distribution, 5 labels `bin_0 … bin_4`, non-finite → bin 0.
Delivered distribution: `pupil_diameter_bin: {bin_0 0.213, bin_1 0.184, bin_2 0.191,
bin_3 0.192, bin_4 0.219}` — again close to but not exactly uniform.

ii.
```python
pupil_bins = digitize(pupil_vals, pupil_edges)
...
'output_values': [..., [f'bin_{i}' for i in range(5)], ...]
```

iii. Follows the Decoder Task instruction *"Pupil diameter, discretized into five equal
percentile bins"*; global edges chosen for cross-session consistency. Note that percentile
binning is invariant to any monotone transform, so using area rather than diameter has only
a second-order effect on the bin assignment.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Evaluated at the trial's ophys frame timestamps, exactly as for running speed, so it
shares the neural time axis. Eye-tracking is sampled at ~60 Hz, so nearest-neighbour error
is ≲8 ms except across dropped blink stretches, where the nearest surviving sample (which
may be several hundred ms away) is carried in.

ii.
```python
pupil_vals = nearest_assign(sample_times, eye_valid['timestamps'].values,
                            eye_valid[pupil_col].values)
pupil_bins = digitize(pupil_vals, pupil_edges)
```

iii. Same as 5-d — all streams are assigned onto the ophys timestamp grid inside each trial.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the SDK `trials` table: `hit`, `miss`,
`false_alarm`, `correct_reject`, tested in that order on the trial row.

ii.
```python
if bool(trial_row.get('hit', False)):
    outcome = 0
elif bool(trial_row.get('miss', False)):
    outcome = 1
elif bool(trial_row.get('false_alarm', False)):
    outcome = 2
elif bool(trial_row.get('correct_reject', False)):
    outcome = 3
else:
    outcome = 0
```

iii. CONVERSION_NOTES Step 1: *"Trial outcomes available from reference code include `hit`,
`miss`, `false_alarm`, and `correct_reject`; `correct_reject` is catch without false alarm.
Auto-rewarded trials have these outcome labels cleared."* Step 5 mapping: *"`trials`
outcome fields → output[4] trial outcome."*

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The scalar outcome code is broadcast across every time bin of the trial so that the
output block is a uniform `(5, n_timepoints)` array (the variable is conceptually static
per trial but stored time-varying). `output_values[4] = ['hit','miss','false_alarm',
'correct_reject']`. Anything that matches none of the four flags silently becomes code 0
(`hit`). Delivered distribution: hit 0.310, miss 0.565, false_alarm 0.018,
correct_reject 0.107.

ii.
```python
trial_outcome = np.full(sample_times.shape, outcome, dtype=np.int64)
return np.stack([image_identity, image_change, run_bins, pupil_bins, trial_outcome], axis=0)
```
```python
'output_names': ['image_identity', 'image_change', 'running_speed_bin',
                 'pupil_diameter_bin', 'trial_outcome'],
'output_values': [..., ['hit', 'miss', 'false_alarm', 'correct_reject']],
```

iii. CONVERSION_NOTES Step 5 mapping: *"Static per-trial categorical label replicated
across timepoints or stored as per-trial vector."* The target format says *"If at all
possible, make it time-varying"*, and keeping all five rows the same length keeps the
output block rectangular. Trajectory step 29 briefly considered storing it as a per-trial
scalar instead but kept the replicated form.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled:
- **Length mismatches**: cell traces are truncated to the shortest trace, and the neural
  matrix is truncated to `len(ophys_timestamps)`.
- **Missing `stop_time` on stimulus presentations** (this release does not have the
  column): synthesised from the next flash's `start_time`, falling back to a `duration`
  column or a 0.75 s median interval.
- **Missing / NaN behavioural samples**: non-finite running or pupil values → bin 0;
  missing pupil column or an entirely empty eye table → all-zero pupil row.
- **Uncovered stimulus time** (trial start before the first flash, omitted flashes) → the
  reserved `blank` image class, which also removed the `-1` labels that had crashed the
  decoder.
- **Degenerate percentile edges** (heavily tied distributions) → nudged with `np.nextafter`.
- **Degenerate trials/sessions**: trials with <2 frames and sessions with <2 trials dropped;
  missing NWB files skipped.
- **Library warnings**: suppressed with `warnings.filterwarnings` (written as non-raw
  strings, which itself emits two `SyntaxWarning`s visible at the top of
  `conversion_full_out.txt`).

Not handled:
- **No `try/except` around experiment loading** — a single corrupt/unreadable NWB would
  abort the whole run.
- **Unknown trial outcome silently becomes `hit`** rather than being flagged or dropped.
- **`str(np.nan)` → `'nan'`** for presentations with a missing `image_name`, which leaked a
  literal `'nan'` category into `output_values[0]` (fraction 0.000 in the full dataset).

ii.
```python
n_t = min(len(a) for a in arrays)
...
n_t = min(neural.shape[1], len(ophys_timestamps))
```
```python
if 'stop_time' in stim.columns:
    return stim
if 'duration' in stim.columns:
    stim['stop_time'] = stim['start_time'] + pd.to_numeric(stim['duration'], errors='coerce').fillna(0.25)
    return stim
stim['stop_time'] = np.r_[starts[1:], starts[-1] + default_interval]
```
```python
out[~np.isfinite(vals)] = 0
...
image_vals = [image_to_idx.get(str(x), blank_idx) for x in stim_non_omitted['image_name'].astype(str).values]
...
for i in range(1, len(edges)):
    if edges[i] <= edges[i - 1]:
        edges[i] = np.nextafter(edges[i - 1], np.inf)
```

iii. Trajectory step 21: *"the sample conversion now fails because `stimulus_presentations`
in this dataset/SDK object does not have a `stop_time` column. This is a concrete schema
mismatch."* Step 25: the `-1` image labels *"caused a CUDA device-side assert … outputs
must not contain negative labels"*, resolved with the `blank` class. CONVERSION_NOTES
Step 10 lists these as the two issues found and resolved. Mapping NaN behaviour to bin 0 is
not justified in the notes.

## 9-a. What are the most time-consuming steps of the code?

i. NWB loading dominates, and the script loads the whole dataset **twice**. Per-experiment
processing prints ≈4.6–6.1 s (`processed 1/202 exp_id=951980471 … dt=4.81s`), i.e. ≈1050 s
for the 202 processing loads; the total run was 2680.7 s, so the `collect_global_info`
pre-pass accounts for roughly the other ≈1600 s. Within a single experiment, the
`BehaviorOphysExperiment.from_nwb_path` read (which materialises ΔF/F, events, stimulus,
running, eye tracking and trials) is the cost; the numeric work per trial is negligible by
comparison. The full run therefore took ~45 min, well over the instructions' 15-minute
guideline.

ii.
```python
for i, exp_id in enumerate(chosen_ids, 1):
    t1 = time.time()
    sess = process_experiment(exp_id, row_lookup[exp_id], ...)
    ...
    print(f'processed {i}/{len(chosen_ids)} exp_id={exp_id} kept_trials={sess["n_trials"]} '
          f'signal={sess["signal_kind"]} dt={time.time()-t1:.2f}s')
```

iii. CONVERSION_NOTES Step 6: *"Code inefficiencies identified: Repeated experiment loading
during global percentile/image collection may be slow for full conversion."* Trajectory
step 33: *"The main bottleneck is `collect_global_info`, which loads every experiment once
to gather image names and percentile bins and then loads them again for processing."*
Step 434: *"roughly 4.6–4.8 seconds per experiment. With 202 experiments total, the full run
should take on the order of 15–17 minutes"* — an estimate that ignored the pre-pass.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Several remain as Python loops:
- `interval_assign` loops over every stimulus presentation in the trial and builds a full
  boolean mask per flash; a single `np.searchsorted` of `sample_times` into the flash start
  times would do it in one shot.
- The `is_change` loop in `build_trial_output` calls `np.searchsorted` once per change.
- The per-trial loop in `process_experiment` recomputes
  `(ophys_timestamps >= start) & (ophys_timestamps < stop)` over the *entire* session
  timestamp vector for each trial — O(n_trials × T_session) instead of two `searchsorted`
  calls.
- `trials.iterrows()` and `row_lookup = {int(r['ophys_experiment_id']): r for _, r in
  exp_table.iterrows()}` are slow pandas row iteration.
- `ensure_stim_stop_time` performs a `sort_values` + `copy` of the stimulus frame once per
  trial.

ii.
```python
for s, e, v in zip(np.asarray(starts, float), np.asarray(stops, float), np.asarray(values, np.int64)):
    m = (sample_times >= s) & (sample_times < e)
    out[m] = v
```
```python
for _, tr in trials.iterrows():
    start = float(tr['start_time']); stop = float(tr['stop_time'])
    m = (ophys_timestamps >= start) & (ophys_timestamps < stop)
```
```python
row_lookup = {int(r['ophys_experiment_id']): r for _, r in exp_table.iterrows()}
```

iii. CONVERSION_NOTES Step 6: *"Interval assignment currently loops over stimulus
presentations per trial"* is the only one identified; the claimed remedy is vague (*"Uses …
simple vectorized timestamp assignment where possible"*). None of these loops were actually
vectorised, because the optimisation work planned at trajectory step 37 was never applied —
the shell entered a broken heredoc state for several hundred steps and the run that
eventually produced `converted_data.pkl` used the unoptimised script.

## 9-c. What processing does the code repeat multiple times?

i. Substantial repetition:
- **Every NWB file is loaded twice** — once in `collect_global_info` (only to harvest image
  names and running/pupil values) and once in `process_experiment`.
- **Per trial**, `build_trial_output` re-runs `ensure_stim_stop_time` on the trial's
  stimulus slice even though `process_experiment` already computed session-level stop times,
  re-runs `pick_pupil_column`, and re-runs
  `eye_df[['timestamps', col]].replace(...).dropna()` over the **entire session's**
  eye-tracking table (tens of thousands of rows × ~250 trials per session).
- `brain_region_idx` is computed as a zero vector inside `process_experiment` and then
  discarded and recomputed in `main`.

ii.
```python
stim = ensure_stim_stop_time(exp.stimulus_presentations.copy())   # in process_experiment
...
def build_trial_output(trial_row, sample_times, stim_df, run_df, eye_df, ...):
    stim = ensure_stim_stop_time(stim_df)                          # again, per trial
    ...
    pupil_col = pick_pupil_column(eye_df)                          # per trial
    eye_valid = eye_df[['timestamps', pupil_col]].replace([np.inf, -np.inf], np.nan).dropna()  # per trial
```
```python
brain_region_idx = np.zeros((neural.shape[0],), dtype=np.int64)    # in process_experiment
...
'brain_region_idx': [np.full((len(s['brain_region_idx']),), region_to_idx[s['region']],
                             dtype=np.int64) for s in sessions],   # recomputed in main
```

iii. The AI documented only the first item (*"Repeated experiment loading during global
percentile/image collection may be slow"*, Step 6) and planned to fix it
(trajectory step 37: *"patch `convert_data.py` to avoid the costly full
`collect_global_info` experiment pass for `--full`"*), but the patch was never applied.
CONVERSION_NOTES Step 6 nonetheless lists "Code speedups added", and the Step 7 "Run Time
Estimates" table was left empty. The per-trial repetitions are not mentioned anywhere.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Work that is done and then thrown away or never used:
- `collect_global_info` constructs a full `BehaviorOphysExperiment` (ΔF/F, events, cell
  table, stimulus, trials, running, eye) for all 206 experiments although only
  `image_name`, `speed` and the pupil column are consumed; every ΔF/F array read there is
  discarded.
- `choose_signal` does `exp.dff_traces.copy()` (a full copy of every trace) purely to test
  emptiness, and `extract_neural_matrix`'s `preferred = ['events','filtered_events','dff']`
  search is dead code for the dF/F path that was ultimately used.
- Outputs are stored as **`int64`** although every value fits in `int8` — for five rows ×
  ~233 frames × 51 992 trials this is a large avoidable cost, and contributes to the
  8.9 GB `converted_data.pkl`.
- An empty `(0, n_timepoints)` float32 array is allocated for every one of the 51 992
  trials even though `input_names` is empty.
- `brain_region_idx` zeros are computed per session then overwritten (see 9-c).
- `--show-processing` and `--full` are declared in `parse_args()` but **never read** in
  `main()`; no `processing_<session_id>.png` plots are produced at all, despite
  CONVERSION_NOTES containing a "Processing Plots Review" section.

ii.
```python
mode.add_argument('--full', action='store_true', help='Process all sessions')
ap.add_argument('--show-processing', action='store_true', help='Plot visualizations for up to 2 sessions')
```
(neither `args.full` nor `args.show_processing` appears anywhere else in the file)
```python
def choose_signal(exp):
    dff = exp.dff_traces.copy()
    if isinstance(dff, pd.DataFrame) and len(dff) > 0:
        return 'dff', dff
```
```python
trial_input = np.zeros((0, len(sample_times)), dtype=np.float32)
...
sess_output.append(trial_output.astype(np.int64))
```

iii. None of this is documented — CONVERSION_NOTES has no discussion of discarded or
redundant computation beyond the double experiment load, and Step 13 was marked COMPLETE
with the plotting requirement unmet.
