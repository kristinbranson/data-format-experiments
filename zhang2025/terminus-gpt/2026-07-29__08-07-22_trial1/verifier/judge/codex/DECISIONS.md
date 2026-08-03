# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the ONE API. It loads the release session index directly from `data/one_cache/Brainwidemap/sessions.pqt`, iterates through those rows, constructs a filesystem path for each session, and then opens ALF/pykilosort files from disk with `pandas.read_parquet` and `numpy.load`.

ii. 
```python
def session_dir_from_row(row):
    return Path('data') / 'one_cache' / str(row['lab']) / 'Subjects' / str(row['subject']) / str(row['date']) / f"{int(row['number']):03d}"
```

```python
def build_dataset(sample=False):
    sessions = pd.read_parquet('data/one_cache/Brainwidemap/sessions.pqt')
```

```python
def load_trials(session_dir):
    p = find_trial_table(session_dir)
    if p is None:
        return None
    return pd.read_parquet(p)
```

iii. The notes say the data are stored as an IBL ONE cache and that the script should use “real ONE/ALF loading logic,” but the implemented script instead uses direct cache-file access. The trajectory explicitly says the agent replaced a placeholder with “a real first-pass implementation” after inspecting the cache layout.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` column of the session metadata table. As sessions are accepted, the script assigns each new subject the next integer in insertion order and appends that subject index once per kept session.

ii. 
```python
subjects, subject_map, subject_idx = [], {}, []
```

```python
neural, inp, out, subj, region_names, region_idx = loaded
...
if subj not in subject_map:
    subject_map[subj] = len(subjects)
    subjects.append(subj)
subject_idx.append(subject_map[subj])
```

iii. The notes describe the dataset as an IBL cache whose session metadata already exposes subject IDs, so no subject parsing logic beyond using that metadata was justified.

## 1-c. How are the data split into sessions?

i. Each row in `Brainwidemap/sessions.pqt` is treated as one session. The script iterates row-by-row, loads that session, and appends one element to `neural`, `input`, `output`, and `subject_idx` for each successfully loaded session.

ii. 
```python
sessions = pd.read_parquet('data/one_cache/Brainwidemap/sessions.pqt')
...
for i, (_, row) in enumerate(sessions.iterrows()):
    ...
    loaded = load_session(row)
```

iii. The notes identify `sessions.pqt` as the release/session metadata source, so the agent treated that table as the session index.

## 1-d. How are the data split into trials?

i. Trials come from rows of `_ibl_trials.table.pqt`. After an initial mask, the script iterates over `stimOn_times`, creates one neural matrix, one input matrix, and one output matrix per surviving row, and finally removes rows whose mapped choice or prior is invalid.

ii. 
```python
trials = load_trials(session_dir)
...
mask = trial_mask(trials)
trials = trials.loc[mask].reset_index(drop=True)
```

```python
stim_times = trials['stimOn_times'].to_numpy(dtype=np.float64)
for i, st in enumerate(stim_times):
    neural_trials.append(bin_spikes_for_trial(..., st))
    input_trials.append(...)
```

```python
for i in range(len(trials)):
    if not np.isfinite(choice[i]) or not np.isfinite(prior[i]):
        continue
    output_trials.append(...)
    valid_trial_keep.append(i)
```

iii. The notes repeatedly say the conversion should follow the IBL convention where the trial table is the source of per-trial structure, and the implemented code follows that basic split.

## 1-e. How are trials filtered based on quality controls?

i. The implemented filter is minimal. First it keeps only rows with finite `stimOn_times`, `choice`, and `probabilityLeft`. Later it drops trials whose mapped `choice` or mapped `prior` is `NaN`. It does not enforce the reference reaction-time bounds, no-response exclusion before counting trial number in block, or wheel/camera coverage checks.

ii. 
```python
def trial_mask(trials):
    need = ['stimOn_times', 'choice', 'probabilityLeft']
    mask = np.ones(len(trials), dtype=bool)
    for c in need:
        if c in trials.columns:
            mask &= np.isfinite(trials[c].to_numpy())
    return mask
```

```python
for i in range(len(trials)):
    if not np.isfinite(choice[i]) or not np.isfinite(prior[i]):
        continue
```

iii. The notes claim the script should “apply reference trial mask and additionally drop trials missing requested outputs,” but the actual implementation falls back to a much simpler finite-value filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is built from `spikes.times.npy` and `spikes.clusters.npy` in each probe directory. `clusters.metrics.pqt` is consulted only to decide which clusters to keep; it is not used directly to form the neural matrix.

ii. 
```python
st = base / 'spikes.times.npy'
sc = base / 'spikes.clusters.npy'
cm = base / 'clusters.metrics.pqt'
```

```python
spikes_t = np.load(st)
spikes_c = np.load(sc)
```

iii. The notes explicitly planned to derive neural data from `spikes.times`, `spikes.clusters`, and curated cluster metadata, matching the basic ephys-source assumption.

## 2-b. How is the `neural` data processed?

i. For each probe, the script filters spikes to good clusters, renumbers surviving cluster IDs so probes can be merged into one population, and concatenates all probe spike times and cluster IDs. For each trial it bins spikes into 20 ms bins relative to stimulus onset, returning raw spike counts as `float32`. It does not divide by bin width to convert to firing rate, and it does not sort the concatenated multi-probe spike times after merging.

ii. 
```python
uniq = np.array(sorted(np.unique(spikes_c)))
remap = {cid: i + offset for i, cid in enumerate(uniq)}
spikes_c = np.array([remap[c] for c in spikes_c], dtype=np.int64)
offset += len(uniq)
all_times.append(spikes_t)
all_clusters.append(spikes_c)
```

```python
def bin_spikes_for_trial(spike_times, spike_clusters, n_neurons, stim_time):
    edges = stim_time + build_time_edges()
    mat = np.zeros((n_neurons, len(edges) - 1), dtype=np.float32)
    ...
    np.add.at(mat, (sc, tbin), 1)
    return mat
```

iii. The notes say the script should “bin spike counts in 20 ms bins per trial aligned to stimulus onset” and “merge probes within session after unit curation.” The trajectory also says the agent intentionally replaced a placeholder with this “first-pass implementation.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script keeps clusters whose metrics table says `label == 1`, or, if that column is absent, clusters whose `ks2_label` string equals `good`. If neither metric is available, it keeps every observed cluster. The brain-region labels of the kept units are not loaded; every kept unit is labeled `unknown`.

ii. 
```python
if cm.exists():
    m = pd.read_parquet(cm)
    cols = {c.lower(): c for c in m.columns}
    if 'label' in cols:
        good_ids = np.flatnonzero(m[cols['label']].to_numpy() == 1)
    elif 'ks2_label' in cols:
        good_ids = np.flatnonzero(m[cols['ks2_label']].astype(str).str.lower().eq('good').to_numpy())
if good_ids is None:
    good_ids = np.unique(spikes_c)
```

```python
region_names.extend(['unknown'] * len(uniq))
```

iii. The notes justify “good units only” based on the papers and reference code, but the trajectory acknowledges that the implementation is approximate and that “good-unit filtering relies on inferred cluster metric columns.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to stimulus onset by creating bin edges at `stim_time + build_time_edges()`. Spikes are selected inside that absolute window and assigned to bins defined relative to the trial’s own `stimOn_times`.

ii. 
```python
def bin_spikes_for_trial(spike_times, spike_clusters, n_neurons, stim_time):
    edges = stim_time + build_time_edges()
```

```python
lo = np.searchsorted(spike_times, edges[0], side='left')
hi = np.searchsorted(spike_times, edges[-1], side='right')
```

iii. The notes repeatedly state that all trial tensors should be aligned to stimulus onset because that is required by the decoder task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins. No later temporal rebinning is applied. However, the aligned window is only from `-0.2` s to `1.0` s, so each trial has 60 bins rather than the 100 bins used by the reference solution.

ii. 
```python
BIN_SIZE_S = 0.02
T_START = -0.2
T_END = 1.0
```

```python
def build_time_edges():
    return np.arange(T_START, T_END + BIN_SIZE_S, BIN_SIZE_S)
```

iii. The notes justify the 20 ms bin size from the papers, but they do not justify the shortened `-0.2` to `1.0` s window. Later notes explicitly record that the output shapes are `(n_neurons, 60)`, `(2, 60)`, and `(4, 60)`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The input is effectively defined from the chosen trial window and bin size, then used relative to each trial’s `stimOn_times`. The raw trial variable involved is `stimOn_times`, but the values placed into the input tensor are fixed relative-time bin centers.

ii. 
```python
def build_time_centers():
    e = build_time_edges()
    return (e[:-1] + e[1:]) / 2
```

```python
stim_times = trials['stimOn_times'].to_numpy(dtype=np.float64)
...
input_trials.append(np.vstack([centers, np.full_like(centers, trial_in_block[i])]).astype(np.float32))
```

iii. The notes planned a “stimulus-onset-aligned time axis” as the first decoder input and described it as continuous time since stimulus onset replicated across bins.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The script computes this input as the centers of a fixed 20 ms grid from `-0.2` s to `1.0` s and reuses the same vector for every trial in a session. No additional transformation is applied.

ii. 
```python
BIN_SIZE_S = 0.02
T_START = -0.2
T_END = 1.0
```

```python
centers = build_time_centers().astype(np.float32)
...
np.vstack([centers, np.full_like(centers, trial_in_block[i])])
```

iii. The notes say this variable is the “stimulus-onset-aligned time axis” and should be broadcast across time bins, but they do not justify the shorter time window used in code.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is placed on the same bin-center grid used for the neural trial matrices. Neural bins are built from the corresponding edges, and the input tensor stores the matching centers, so the two share the same per-trial time base.

ii. 
```python
edges = stim_time + build_time_edges()
```

```python
centers = build_time_centers().astype(np.float32)
input_trials.append(np.vstack([centers, ...]).astype(np.float32))
```

iii. The notes justify a common 20 ms binning and stimulus-onset alignment across neural, input, and output streams.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the `probabilityLeft` column of the trials table. Consecutive trials with the same `probabilityLeft` value are treated as belonging to one block.

ii. 
```python
prob_left = trials['probabilityLeft'].to_numpy(dtype=np.float32)
trial_in_block = trial_number_in_block(prob_left)
```

```python
def trial_number_in_block(prob_left):
    ...
    if prob_left[i] == prob_left[i - 1]:
        run += 1
    else:
        run = 1
```

iii. The notes explicitly planned to derive trial number in block from the run lengths of `probabilityLeft`, because no native block-counter variable was identified.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The script computes a run length after the initial trial mask: the first trial in a block is labeled `1`, the next `2`, and so on. That per-trial scalar is then broadcast across all time bins for the trial.

ii. 
```python
run = 1
out[0] = 1
for i in range(1, len(prob_left)):
    if prob_left[i] == prob_left[i - 1]:
        run += 1
    else:
        run = 1
    out[i] = run
```

```python
input_trials.append(np.vstack([centers, np.full_like(centers, trial_in_block[i])]).astype(np.float32))
```

iii. The notes justify this as a “custom block-run-length transform,” but they do not mention that the implementation is 1-indexed and is computed after the initial trial filtering.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from the `choice` column in `_ibl_trials.table.pqt`.

ii. 
```python
choice = np.array([map_choice(v) for v in trials['choice'].to_numpy()], dtype=np.float32)
```

iii. The notes mapped `trials.choice` directly to the decoder output, consistent with the task description.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The script maps `choice == 1` to `0` (left), `choice == -1` to `1` (right), and anything else to `NaN`. Trials with `NaN` choice are dropped from the final session lists. The retained choice label is then broadcast across all time bins for the trial.

ii. 
```python
def map_choice(v):
    # IBL usually: 1=CCW(left), -1=CW(right)
    return 0 if v == 1 else 1 if v == -1 else np.nan
```

```python
if not np.isfinite(choice[i]) or not np.isfinite(prior[i]):
    continue
...
np.full(centers.shape, int(choice[i]), dtype=np.int64)
```

iii. The notes say this output should be “Map left=0, right=1,” and later checks in the notes say the first converted trial’s choice matched the raw trial table.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from `probabilityLeft` in the trials table.

ii. 
```python
prob_left = trials['probabilityLeft'].to_numpy(dtype=np.float32)
prior = np.array([map_prior(v) for v in prob_left], dtype=np.float32)
```

iii. The notes explicitly planned to use `trials.probabilityLeft` for this output.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The script maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`. Anything else becomes `NaN` and causes the trial to be dropped. The retained value is broadcast across all time bins for the trial.

ii. 
```python
def map_prior(v):
    if np.isclose(v, 0.2):
        return 0
    if np.isclose(v, 0.5):
        return 1
    if np.isclose(v, 0.8):
        return 2
    return np.nan
```

```python
np.full(centers.shape, int(prior[i]), dtype=np.int64)
```

iii. The notes say this output should map the three block priors to `0/1/2`, matching the task instructions.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii. 
```python
def load_wheel(session_dir):
    alf = session_dir / 'alf'
    pos = alf / '_ibl_wheel.position.npy'
    ts = alf / '_ibl_wheel.timestamps.npy'
```

```python
wheel_ts, wheel_pos = load_wheel(session_dir)
wheel_v_ts, wheel_v = wheel_speed(wheel_ts, wheel_pos)
```

iii. The notes planned to use wheel velocity/speed from the cached wheel dataset and to align it to stimulus onset.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The script computes wheel speed by taking the absolute discrete derivative of position with respect to time, using the midpoint of each raw timestamp interval. It then bins those speeds into the trial’s 20 ms bins by averaging samples that fall in each bin. It does not use `SessionLoader`’s interpolated and filtered wheel velocity.

ii. 
```python
def wheel_speed(ts, pos):
    dt = np.diff(ts)
    dp = np.diff(pos)
    good = dt > 0
    mids = ts[:-1][good] + dt[good] / 2
    vv = np.abs(dp[good] / dt[good]).astype(np.float32)
    return mids, vv
```

```python
ws = bin_signal(wheel_v_ts, wheel_v, st + centers) if wheel_v is not None else np.full(len(centers), np.nan, dtype=np.float32)
```

iii. The notes justify 20 ms wheel binning from the methods paper, but they do not justify replacing the reference wheel-processing pipeline with this hand-written derivative.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The script pools all finite binned wheel-speed values from all kept trial windows in a session, computes the 1/3 and 2/3 quantiles, and uses those as the three-class thresholds. Missing binned values are filled with the lower threshold before digitization.

ii. 
```python
all_wheel_cont.append(ws[np.isfinite(ws)])
...
wq = np.quantile(all_wheel_cont, [1/3, 2/3]) if len(all_wheel_cont) else [0, 0]
```

```python
wb = np.digitize(np.nan_to_num(tmp_wheel[i], nan=wq[0] if np.ndim(wq) else 0.0), wq, right=False).astype(np.int64)
```

iii. The notes explicitly planned three-bin discretization and mention “global quantiles or reference-consistent binning”; later notes say a sample-stage NaN problem was fixed by “filling missing values before discretization.”

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned by evaluating it on the same per-trial bin centers used by the neural data: for trial onset `st`, the target times are `st + centers`.

ii. 
```python
centers = build_time_centers().astype(np.float32)
...
ws = bin_signal(wheel_v_ts, wheel_v, st + centers)
```

iii. The notes justify a common 20 ms stimulus-onset-aligned grid across neural and behavioral outputs.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The script loads `leftCamera.ROIMotionEnergy.npy` and `rightCamera.ROIMotionEnergy.npy` if present, averages them if both exist, and then tries to infer timestamps from a camera features parquet or from the trial timing range. It does not load the released camera times arrays directly.

ii. 
```python
def load_motion_energy(session_dir):
    left_p, right_p = find_motion_energy(session_dir)
    vals = []
    for p in [left_p, right_p]:
        if p is not None and p.exists():
            vals.append(np.load(p).astype(np.float32))
```

```python
me = load_motion_energy(session_dir)
me_ts = guess_motion_timestamps(session_dir, len(me)) if me is not None else None
```

iii. The notes say the agent intended to use video/whisker datasets, but later notes and trajectory explicitly acknowledge that “whisker timestamps are approximated.”

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The script averages left and right camera motion-energy traces when both exist, truncates them to a common length, infers or fabricates timestamps, and then bins the resulting continuous trace into each trial’s 20 ms bins by taking a mean within each bin. No additional filtering or normalization is applied.

ii. 
```python
n = min(len(v) for v in vals)
arr = np.mean(np.vstack([v[:n] for v in vals]), axis=0)
return arr
```

```python
def guess_motion_timestamps(session_dir, n):
    ...
    if feats:
        ...
            if 'times' in c.lower() or 'timestamp' in c.lower():
                ...
                return x[:n]
    ...
    return np.linspace(t0, t1, n, dtype=np.float32)
```

```python
ms = bin_signal(me_ts, me, st + centers) if me is not None else np.full(len(centers), np.nan, dtype=np.float32)
```

iii. The notes justify using cached whisker/video variables, but they also record the known limitation that timestamps are approximated and brain-region metadata remains incomplete.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is thresholded the same way as wheel speed: pool all finite binned values from the session’s kept trial windows, compute the 1/3 and 2/3 quantiles, and digitize each trial’s binned trace after filling missing values with the lower threshold.

ii. 
```python
all_me_cont.append(ms[np.isfinite(ms)])
...
mq = np.quantile(all_me_cont, [1/3, 2/3]) if len(all_me_cont) else [0, 0]
```

```python
mb = np.digitize(np.nan_to_num(tmp_me[i], nan=mq[0] if np.ndim(mq) else 0.0), mq, right=False).astype(np.int64)
```

iii. The notes planned a three-bin discretization for whisker motion energy and later note that NaN handling was changed to force categorical integer bins.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned by evaluating the trace at `st + centers`, the same target times used for wheel speed and the neural bin grid. However, because the timestamps are guessed rather than taken from the released camera time arrays, the alignment depends on an approximation.

ii. 
```python
me_ts = guess_motion_timestamps(session_dir, len(me)) if me is not None else None
...
ms = bin_signal(me_ts, me, st + centers)
```

iii. The notes explicitly acknowledge this limitation: “whisker timestamps are approximated from available camera/session timing because explicit ROI timestamps were not yet loaded.”

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The script mostly skips missing high-level files but imputes missing continuous behavioral traces. It skips sessions with no session directory, no trial table, no `stimOn_times`, fewer than two initially masked trials, or no spikes. It drops trials with missing/invalid `stimOn_times`, `choice`, or `probabilityLeft`, and later drops trials whose mapped `choice` or `prior` is invalid. But if wheel or whisker data are missing within an otherwise valid session, it fills the corresponding trial traces with `NaN`, then converts them into categorical bins using fallback quantiles instead of dropping the trial/session.

ii. 
```python
if not session_dir.exists():
    return None
...
if trials is None or 'stimOn_times' not in trials.columns:
    return None
...
if len(trials) < 2:
    return None
...
if spike_times is None:
    return None
```

```python
ws = ... if wheel_v is not None else np.full(len(centers), np.nan, dtype=np.float32)
ms = ... if me is not None else np.full(len(centers), np.nan, dtype=np.float32)
```

```python
wb = np.digitize(np.nan_to_num(tmp_wheel[i], nan=wq[0] if np.ndim(wq) else 0.0), wq, right=False).astype(np.int64)
mb = np.digitize(np.nan_to_num(tmp_me[i], nan=mq[0] if np.ndim(mq) else 0.0), mq, right=False).astype(np.int64)
```

iii. The notes describe these as “known limitations,” especially approximate whisker timestamps and forced categorical filling for wheel/whisker NaNs, rather than as a deliberate match to the reference.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive work is per-session loading and per-trial processing of spike data: reading `spikes.times.npy` and `spikes.clusters.npy` for every probe, then repeatedly binning spikes trial-by-trial. The trajectory shows the full conversion spent tens of seconds on many sessions, consistent with spike I/O and spike binning dominating runtime.

ii. 
```python
spikes_t = np.load(st)
spikes_c = np.load(sc)
```

```python
for i, st in enumerate(stim_times):
    neural_trials.append(bin_spikes_for_trial(spike_times, spike_clusters, n_neurons, st))
```

iii. The notes estimate “~35-75 s/session” and the trajectory repeatedly reports long waits on individual sessions, so the agent evidently viewed full-session conversion as the runtime bottleneck.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are obvious vectorization candidates: the Python loop that computes `trial_number_in_block`, the per-trial neural binning loop, the per-trial behavior binning loop, the inner per-bin loop inside `bin_signal`, and the Python dictionary/list remapping of cluster IDs. These are all left as explicit Python loops.

ii. 
```python
for i in range(1, len(prob_left)):
    if prob_left[i] == prob_left[i - 1]:
        run += 1
```

```python
for i, st in enumerate(stim_times):
    neural_trials.append(bin_spikes_for_trial(..., st))
    ...
    ws = bin_signal(...)
    ms = bin_signal(...)
```

```python
for i in range(len(centers)):
    m = idx == i
    if np.any(m):
        out[i] = np.nanmean(values[m])
```

iii. The notes do not discuss these loop-level inefficiencies in detail; they only note that full conversion is likely to take hours without further optimization.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several computations. It reloads the trials table inside `guess_motion_timestamps()` even though the session’s trials were already loaded. It computes session-level discretized arrays `wheel_bins_all` and `me_bins_all` and then does not use them, while separately recomputing quantiles `wq` and `mq` for the actual digitization step. It also rebuilds bin edges inside `bin_signal()` for every trial/output stream call.

ii. 
```python
trials = load_trials(session_dir)
...
trials = load_trials(session_dir)
if trials is not None and 'stimOn_times' in trials.columns:
```

```python
wheel_bins_all = discretize_three_bins(all_wheel_cont) if len(all_wheel_cont) else np.array([], dtype=np.float32)
me_bins_all = discretize_three_bins(all_me_cont) if len(all_me_cont) else np.array([], dtype=np.float32)
...
wq = np.quantile(all_wheel_cont, [1/3, 2/3]) if len(all_wheel_cont) else [0, 0]
mq = np.quantile(all_me_cont, [1/3, 2/3]) if len(all_me_cont) else [0, 0]
```

iii. No explicit justification was given for this repetition; the notes instead frame the script as an initial working implementation that still had scientific limitations.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computed values are discarded. `latest_file()` is never used. `wheel_speed()` allocates `v` but never uses it. `load_spikes_and_regions()` reads a `clusters.channels.npy` path into `ch` but never uses it. `wheel_bins_all`, `me_bins_all`, and `n_target` are computed and never used. The code also constructs per-unit `region_names`, but the assembled dataset collapses all regions to `['unknown']` with all-zero `brain_region_idx`.

ii. 
```python
def latest_file(pattern):
    matches = sorted(pattern.parent.glob(pattern.name))
    return matches[-1] if matches else None
```

```python
v = np.zeros_like(pos, dtype=np.float32)
...
ch = base / 'clusters.channels.npy'
```

```python
wheel_bins_all = discretize_three_bins(all_wheel_cont) if len(all_wheel_cont) else np.array([], dtype=np.float32)
me_bins_all = discretize_three_bins(all_me_cont) if len(all_me_cont) else np.array([], dtype=np.float32)
...
n_target = 2 if sample else len(sessions)
```

```python
brain_regions = ['unknown']
...
region_idx = np.zeros(n_neurons, dtype=np.int64)
```

iii. The notes explicitly acknowledge that brain regions are still `unknown`, which makes most of the region-handling work scientifically uninformative in the final dataset.
