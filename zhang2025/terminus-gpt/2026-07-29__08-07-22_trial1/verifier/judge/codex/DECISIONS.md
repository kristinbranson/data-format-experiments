# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the session index by opening `data/one_cache/Brainwidemap/sessions.pqt` directly with pandas, then iterates over rows. For each row it constructs a filesystem path `data/one_cache/<lab>/Subjects/<subject>/<date>/<number>` and directly opens parquet and NumPy files from the ALF tree. It does not use the ONE API, `SessionLoader`, or `SpikeSortingLoader`.

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
return pd.read_parquet(p)
...
return np.load(ts), np.load(pos)
...
spikes_t = np.load(st)
spikes_c = np.load(sc)
```

iii. In `CONVERSION_NOTES.md`, the AI stated it intended to use “IBL/ONE-style” loading and later admitted the script had used a “placeholder session loader” that should be replaced with “exact ONE/ALF loading logic,” but the final implementation still uses direct file-path construction and direct file reads instead of the ONE loaders.

## 1-b. How are the data split into subjects (mice)?

i. Sessions are assigned to subjects using the `subject` field from each row of `sessions.pqt`. A subject list is built incrementally in first-seen order, and `subject_idx` stores the integer index for each kept session.

ii.
```python
subject = str(row['subject'])
```

```python
if subj not in subject_map:
    subject_map[subj] = len(subjects)
    subjects.append(subj)
subject_idx.append(subject_map[subj])
```

iii. The notes say the cache contains session metadata including subject identifiers, so the AI relied on that metadata rather than inferring subjects from filenames.

## 1-c. How are the data split into sessions?

i. The AI treats each row of `data/one_cache/Brainwidemap/sessions.pqt` as one session. It iterates row-by-row and either keeps or skips each session based on whether required files can be loaded.

ii.
```python
sessions = pd.read_parquet('data/one_cache/Brainwidemap/sessions.pqt')
...
for i, (_, row) in enumerate(sessions.iterrows()):
    loaded = load_session(row)
```

iii. The trajectory says the AI concluded the cache has a “concrete filesystem layout” keyed by lab/subject/date/number, and that was enough to drive per-session loading.

## 1-d. How are the data split into trials?

i. The AI loads `_ibl_trials.table.pqt` for a session and treats each row of that table as one trial. After masking, it resets the index and iterates through `stimOn_times`.

ii.
```python
def load_trials(session_dir):
    p = find_trial_table(session_dir)
    if p is None:
        return None
    return pd.read_parquet(p)
```

```python
mask = trial_mask(trials)
trials = trials.loc[mask].reset_index(drop=True)
...
stim_times = trials['stimOn_times'].to_numpy(dtype=np.float64)
for i, st in enumerate(stim_times):
```

iii. The notes identify the trials table as the central source of valid task trials, so the AI followed the one-row-per-trial structure already present in the ALF trials table.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies a minimal mask: it requires finite `stimOn_times`, `choice`, and `probabilityLeft`. Later, it drops trials whose mapped choice or prior is non-finite. It does not apply the reference reaction-time filter, no-response exclusion as part of a reaction-time mask, or wheel/camera coverage checks.

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

iii. The notes say the AI intended to “apply the same trial mask logic as the reference code,” but the implemented mask is much simpler. No explicit justification for omitting reaction-time and coverage filtering was recorded.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural arrays are derived from `spikes.times.npy` and `spikes.clusters.npy` in each probe directory. Cluster metrics are also read to determine which clusters count as “good,” but the neural activity itself is built from spike times and cluster assignments.

ii.
```python
st = base / 'spikes.times.npy'
sc = base / 'spikes.clusters.npy'
cm = base / 'clusters.metrics.pqt'
...
spikes_t = np.load(st)
spikes_c = np.load(sc)
```

iii. The notes explicitly map neural data to `spikes.times`, `spikes.clusters`, and curated cluster metadata, consistent with the code.

## 2-b. How is the `neural` data processed?

i. For each trial, the AI bins spikes into 20 ms bins in a stimulus-aligned window from `-0.2` s to `1.0` s. The output is a neuron-by-time spike-count matrix. Probes are merged by concatenating spikes and remapping cluster IDs with offsets. The code does not convert counts to firing rates and does not sort merged spikes by time after concatenation.

ii.
```python
BIN_SIZE_S = 0.02
T_START = -0.2
T_END = 1.0
```

```python
def bin_spikes_for_trial(spike_times, spike_clusters, n_neurons, stim_time):
    edges = stim_time + build_time_edges()
    mat = np.zeros((n_neurons, len(edges) - 1), dtype=np.float32)
    ...
    np.add.at(mat, (sc, tbin), 1)
    return mat
```

```python
uniq = np.array(sorted(np.unique(spikes_c)))
remap = {cid: i + offset for i, cid in enumerate(uniq)}
spikes_c = np.array([remap[c] for c in spikes_c], dtype=np.int64)
offset += len(uniq)
all_times.append(spikes_t)
all_clusters.append(spikes_c)
```

iii. The notes justify 20 ms bins and stimulus alignment, but they also say the conversion should “match the reference processing.” The final code diverges by using a shorter window and by leaving neural data as counts instead of Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps clusters with `label == 1` when a `label` column exists in `clusters.metrics.pqt`. If not, it falls back to `ks2_label == 'good'`; if neither exists, it keeps all observed clusters. It does not filter out clusters in `void` brain regions, and it does not recover reference atlas labels for units.

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
keep = np.isin(spikes_c, good_ids)
```

iii. The notes say “good units only” and mention QC metadata, but they do not justify the fallbacks or the omission of atlas-based in-brain filtering. Later notes also acknowledge that all final brain regions remain `unknown`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the instructions?

i. Each trial is aligned to `stimOn_times`. The code forms bin edges by adding the fixed time window to each trial’s stimulus onset and bins spikes relative to that event.

ii.
```python
stim_times = trials['stimOn_times'].to_numpy(dtype=np.float64)
for i, st in enumerate(stim_times):
    neural_trials.append(bin_spikes_for_trial(spike_times, spike_clusters, n_neurons, st))
```

```python
edges = stim_time + build_time_edges()
```

iii. The notes repeatedly state that the decoder task should be aligned to stimulus onset, which is what the code implements.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 20 ms bins. No additional temporal rebinning or smoothing is applied after spike counting.

ii.
```python
BIN_SIZE_S = 0.02
```

```python
def build_time_edges():
    return np.arange(T_START, T_END + BIN_SIZE_S, BIN_SIZE_S)
```

iii. The notes say 20 ms was chosen to match the papers and to use a common bin size across streams.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The actual input values are generated from the fixed bin centers of the chosen trial window. They are conceptually relative to `stimOn_times`, because the whole session is aligned to stimulus onset, but the per-trial input tensor directly stores the same `centers` vector for every trial.

ii.
```python
def build_time_centers():
    e = build_time_edges()
    return (e[:-1] + e[1:]) / 2
```

```python
input_trials.append(np.vstack([centers, np.full_like(centers, trial_in_block[i])]).astype(np.float32))
```

iii. The notes say this input should be “stimulus-onset-aligned time axis” and describe it as a derived variable rather than a raw table column.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI computes equally spaced bin centers from `T_START=-0.2`, `T_END=1.0`, and `BIN_SIZE_S=0.02`, then copies that vector into every trial’s input tensor.

ii.
```python
def build_time_edges():
    return np.arange(T_START, T_END + BIN_SIZE_S, BIN_SIZE_S)

def build_time_centers():
    e = build_time_edges()
    return (e[:-1] + e[1:]) / 2
```

iii. The notes justify this as a uniform 20 ms common grid, but do not justify the shorter `[-0.2, 1.0]` window relative to the reference.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The same `centers` vector used in the input tensor is also used to define the spike-binning edges around each stimulus onset. Wheel and whisker outputs are also sampled at `stimulus_onset + centers`, so the input and neural data share a common bin grid.

ii.
```python
centers = build_time_centers().astype(np.float32)
...
input_trials.append(np.vstack([centers, np.full_like(centers, trial_in_block[i])]).astype(np.float32))
```

```python
edges = stim_time + build_time_edges()
```

iii. The notes explicitly aimed for one common 20 ms binning across neural, input, and output streams.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the trial table’s `probabilityLeft` values. A block is inferred whenever `probabilityLeft` changes.

ii.
```python
def trial_number_in_block(prob_left):
    ...
    for i in range(1, len(prob_left)):
        if prob_left[i] == prob_left[i - 1]:
            run += 1
        else:
            run = 1
```

```python
prob_left = trials['probabilityLeft'].to_numpy(dtype=np.float32)
trial_in_block = trial_number_in_block(prob_left)
```

iii. The notes say this input is “derived from block prior run lengths because the requested input is not a native raw variable.”

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI computes a run length within consecutive equal `probabilityLeft` values. The first trial in a block is labeled `1`, the second `2`, and so on. This is computed after the initial trial mask, then broadcast across all time bins for each remaining trial.

ii.
```python
out = np.zeros(len(prob_left), dtype=np.float32)
...
run = 1
out[0] = 1
...
out[i] = run
```

```python
input_trials.append(np.vstack([centers, np.full_like(centers, trial_in_block[i])]).astype(np.float32))
```

iii. The notes justify using block run lengths, but they do not note that the code uses one-based counting and computes the run lengths only after filtering, unlike the reference.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column of the trials table.

ii.
```python
choice = np.array([map_choice(v) for v in trials['choice'].to_numpy()], dtype=np.float32)
```

```python
def map_choice(v):
    return 0 if v == 1 else 1 if v == -1 else np.nan
```

iii. The notes identify `trials.choice` as the raw source and say left/right should be mapped to `0/1`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The code maps `choice == 1` to `0` (“left”), `choice == -1` to `1` (“right”), and any other value to `NaN`. Trials with `NaN` choice are dropped before the final per-trial outputs are assembled. The final choice category is broadcast across all time bins in the trial.

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

iii. The notes and trajectory both describe this as a straightforward recoding required by the decoder task.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from `probabilityLeft` in the trials table.

ii.
```python
prob_left = trials['probabilityLeft'].to_numpy(dtype=np.float32)
prior = np.array([map_prior(v) for v in prob_left], dtype=np.float32)
```

```python
def map_prior(v):
    if np.isclose(v, 0.2):
        return 0
    if np.isclose(v, 0.5):
        return 1
    if np.isclose(v, 0.8):
        return 2
```

iii. The notes say the task explicitly requires the `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2` mapping.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code maps `0.2`, `0.5`, and `0.8` to categorical values `0`, `1`, and `2`. Any other value becomes `NaN`, and such trials are dropped before final output assembly. The resulting label is broadcast across all time bins in the trial.

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

iii. The notes justify the categorical mapping as required by the decoder specification.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

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

iii. The notes map wheel output to the wheel position/timestamp dataset and say it should be aligned and discretized at 20 ms.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI computes an absolute finite-difference speed `abs(diff(position) / diff(time))`, timestamps those speeds at interval midpoints, bins the resulting continuous values into the trial’s 20 ms bins using within-bin averages, and later discretizes them.

ii.
```python
def wheel_speed(ts, pos):
    if ts is None or pos is None or len(ts) < 2:
        return None, None
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

iii. The notes say the AI intended to use “IBL behavior loaders,” but the final code instead implements its own simpler derivative-based speed estimate. No explicit justification for not using `SessionLoader.load_wheel()` was recorded.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI computes session-level tertile thresholds from all finite wheel-speed bins collected across all kept trial windows, then digitizes each trial’s wheel-speed trace into three bins. Missing wheel values are replaced with the lower threshold before digitization.

ii.
```python
all_wheel_cont.append(ws[np.isfinite(ws)])
...
wq = np.quantile(all_wheel_cont, [1/3, 2/3]) if len(all_wheel_cont) else [0, 0]
...
wb = np.digitize(np.nan_to_num(tmp_wheel[i], nan=wq[0] if np.ndim(wq) else 0.0), wq, right=False).astype(np.int64)
```

iii. The notes say wheel and whisker variables would be discretized into three bins and that the exact thresholds would be chosen after inspecting distributions. Later notes say missing values were forced into categorical integer bins “forcing categorical integer bins and filling missing values before discretization.”

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned by evaluating the wheel-speed trace in the same stimulus-centered 20 ms bins used for neural data, i.e. at `stimulus_onset + centers`.

ii.
```python
ws = bin_signal(wheel_v_ts, wheel_v, st + centers) if wheel_v is not None else np.full(len(centers), np.nan, dtype=np.float32)
```

```python
input_trials.append(np.vstack([centers, np.full_like(centers, trial_in_block[i])]).astype(np.float32))
neural_trials.append(bin_spikes_for_trial(spike_times, spike_clusters, n_neurons, st))
```

iii. The notes explicitly aimed to use a uniform 20 ms grid across neural and behavioral streams.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The AI reads `leftCamera.ROIMotionEnergy.npy` and `rightCamera.ROIMotionEnergy.npy` if available, averages them if both exist, and pairs the resulting values with timestamps guessed from a camera features parquet or from a synthetic session-wide linspace if necessary.

ii.
```python
def find_motion_energy(session_dir):
    alf = session_dir / 'alf'
    left = sorted(alf.glob('#*/leftCamera.ROIMotionEnergy.npy'))
    right = sorted(alf.glob('#*/rightCamera.ROIMotionEnergy.npy'))
    return (left[-1] if left else None), (right[-1] if right else None)
```

```python
def load_motion_energy(session_dir):
    left_p, right_p = find_motion_energy(session_dir)
    vals = []
    for p in [left_p, right_p]:
        if p is not None and p.exists():
            vals.append(np.load(p).astype(np.float32))
    ...
    arr = np.mean(np.vstack([v[:n] for v in vals]), axis=0)
    return arr
```

```python
me = load_motion_energy(session_dir)
me_ts = guess_motion_timestamps(session_dir, len(me)) if me is not None else None
```

iii. The notes say the AI intended to use cached whisker/video-derived variables when available. Later notes explicitly acknowledge a limitation: “whisker timestamps are approximated from available camera/session timing because explicit ROI timestamps were not yet loaded.”

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI averages left and right ROI motion-energy traces when both are present, truncating both to a common minimum length. It then bins the resulting trace into the stimulus-centered 20 ms grid using within-bin averages, and later discretizes the binned values into three categories.

ii.
```python
n = min(len(v) for v in vals)
arr = np.mean(np.vstack([v[:n] for v in vals]), axis=0)
return arr
```

```python
ms = bin_signal(me_ts, me, st + centers) if me is not None else np.full(len(centers), np.nan, dtype=np.float32)
```

iii. No explicit justification was given for averaging left and right cameras. The recorded justification focuses instead on using available whisker-related streams and producing categorical bins for the decoder.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. As with wheel speed, the AI computes session-level tertile thresholds from all finite binned whisker-motion values across kept trials, then digitizes each trial trace into bins `0`, `1`, and `2`. Missing values are replaced with the lower threshold before digitization.

ii.
```python
all_me_cont.append(ms[np.isfinite(ms)])
...
mq = np.quantile(all_me_cont, [1/3, 2/3]) if len(all_me_cont) else [0, 0]
...
mb = np.digitize(np.nan_to_num(tmp_me[i], nan=mq[0] if np.ndim(mq) else 0.0), mq, right=False).astype(np.int64)
```

iii. The notes say whisker motion energy would be discretized into three bins, and later say missing values were filled before discretization to ensure categorical outputs.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The AI aligns whisker motion energy by binning it at `stimulus_onset + centers`, using timestamps guessed by `guess_motion_timestamps()`. If camera-feature timestamps are not found, it fabricates timestamps by spreading samples across the session interval or, failing that, by using a uniform 20 ms grid from zero.

ii.
```python
def guess_motion_timestamps(session_dir, n):
    feats = sorted(alf.glob('_ibl_*Camera.features.pqt'))
    if feats:
        ...
        if len(x) >= n:
            return x[:n]
    trials = load_trials(session_dir)
    if trials is not None and 'stimOn_times' in trials.columns:
        t0 = np.nanmin(trials['stimOn_times'].to_numpy()) + T_START
        t1 = np.nanmax(trials['stimOn_times'].to_numpy()) + T_END
        return np.linspace(t0, t1, n, dtype=np.float32)
    return np.arange(n, dtype=np.float32) * BIN_SIZE_S
```

```python
ms = bin_signal(me_ts, me, st + centers) if me is not None else np.full(len(centers), np.nan, dtype=np.float32)
```

iii. The notes explicitly acknowledge this as an approximation rather than exact camera-time alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly handles missing data by skipping sessions missing critical files (`session_dir`, trials table, `stimOn_times`, or spikes) and by skipping sessions with fewer than two masked trials. Missing/invalid `choice` or `prior` causes a trial to be dropped. Missing wheel or whisker data are not grounds for dropping a trial; instead, those traces become `NaN` and are later filled during discretization.

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
if not np.isfinite(choice[i]) or not np.isfinite(prior[i]):
    continue
```

```python
wb = np.digitize(np.nan_to_num(tmp_wheel[i], nan=wq[0] if np.ndim(wq) else 0.0), wq, right=False).astype(np.int64)
mb = np.digitize(np.nan_to_num(tmp_me[i], nan=mq[0] if np.ndim(mq) else 0.0), mq, right=False).astype(np.int64)
```

iii. The notes record that NaN wheel/whisker issues were “fixed by forcing categorical integer bins and filling missing values before discretization,” which explains the implemented imputation behavior.

## 10-a. What are the most time-consuming steps of the code?

i. The code is most likely dominated by repeatedly loading full spike arrays from disk and by the per-trial loops that bin spikes and behavioral traces. The notes also estimate sample conversion at roughly “35-75 s/session depending on spike count,” which is consistent with spike I/O and trial-wise processing being the main costs.

ii.
```python
spikes_t = np.load(st)
spikes_c = np.load(sc)
```

```python
for i, st in enumerate(stim_times):
    neural_trials.append(bin_spikes_for_trial(spike_times, spike_clusters, n_neurons, st))
    ...
    ws = bin_signal(wheel_v_ts, wheel_v, st + centers)
    ms = bin_signal(me_ts, me, st + centers)
```

iii. The only explicit runtime justification in the notes is the spike-count-dependent runtime estimate; there is no deeper profiling discussion.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized: the block run-length loop in `trial_number_in_block`, the per-bin loop in `bin_signal`, the per-trial loop in `load_session`, and the Python-level cluster remapping list comprehension in `load_spikes_and_regions`.

ii.
```python
for i in range(1, len(prob_left)):
    if prob_left[i] == prob_left[i - 1]:
        run += 1
```

```python
for i in range(len(centers)):
    m = idx == i
    if np.any(m):
        out[i] = np.nanmean(values[m])
```

```python
for i, st in enumerate(stim_times):
    neural_trials.append(bin_spikes_for_trial(spike_times, spike_clusters, n_neurons, st))
```

iii. The AI did not document these as optimization targets. The notes only mention rough runtime estimates rather than concrete vectorization choices.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several computations: it rebuilds time edges/centers inside helper calls, rescans trial tables inside `guess_motion_timestamps()` after the trials table was already loaded, and computes discretized full-session arrays `wheel_bins_all` and `me_bins_all` that are not used downstream.

ii.
```python
def build_time_edges():
    return np.arange(T_START, T_END + BIN_SIZE_S, BIN_SIZE_S)

def build_time_centers():
    e = build_time_edges()
    return (e[:-1] + e[1:]) / 2
```

```python
trials = load_trials(session_dir)
...
me_ts = guess_motion_timestamps(session_dir, len(me)) if me is not None else None
```

```python
wheel_bins_all = discretize_three_bins(all_wheel_cont) if len(all_wheel_cont) else np.array([], dtype=np.float32)
me_bins_all = discretize_three_bins(all_me_cont) if len(all_me_cont) else np.array([], dtype=np.float32)
```

iii. No explicit justification was documented for these repeated computations.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest discarded work is the computation of `wheel_bins_all` and `me_bins_all`, which are never used. Other unnecessary work includes allocating an unused `v` array in `wheel_speed()`, reading `clusters.channels.npy` without using it, and creating the unused `n_target` variable.

ii.
```python
wheel_bins_all = discretize_three_bins(all_wheel_cont) if len(all_wheel_cont) else np.array([], dtype=np.float32)
me_bins_all = discretize_three_bins(all_me_cont) if len(all_me_cont) else np.array([], dtype=np.float32)
```

```python
v = np.zeros_like(pos, dtype=np.float32)
```

```python
ch = base / 'clusters.channels.npy'
...
n_target = 2 if sample else len(sessions)
```

iii. The AI did not justify these extra computations. They appear to be leftovers from intermediate development rather than deliberate processing decisions.
