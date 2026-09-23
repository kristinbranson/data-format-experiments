# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI builds a session list from the CSV freeze file `/app/code/code_zhang2025/data/bwm_release.csv`, which lists all 459 sessions and 699 probe insertions in the BWM release. When the datalimit subset file exists, sessions are restricted to those eids. For each session, data is loaded via the ONE API and IBL loaders (`SessionLoader` for trials/wheel/motion energy, `SpikeSortingLoader` for spike sorting per probe).

ii.
```python
BWM_FREEZE_FILE = '/app/code/code_zhang2025/data/bwm_release.csv'

def build_session_list():
    bwm = pd.read_csv(BWM_FREEZE_FILE, index_col=0)
    if os.path.exists(DATALIMIT_FILE):
        sub = pd.read_csv(DATALIMIT_FILE)
        col = 'eid' if 'eid' in sub.columns else sub.columns[0]
        bwm = bwm[bwm.eid.isin(sub[col].astype(str))]
    sessions = []
    for eid, g in bwm.groupby('eid', sort=False):
        g = g.sort_values('probe_name')
        sessions.append({
            'eid': str(eid), 'pids': [str(p) for p in g.pid],
            'probe_names': list(g.probe_name),
            'subject': str(g.subject.iloc[0]),
            ...
        })
    return sessions
```

iii. The AI uses the same freeze file as the Zhang et al. reference code (`bwm_release.csv`) to enumerate sessions, ensuring the session set matches the reference exactly. The ONE API is then used to load each session's data from the local cache.

## 1-b. How are the data split into subjects?

i. Subject names come from the `subject` column in `bwm_release.csv`. At assembly, unique subjects are sorted and `subject_idx` maps each session to its subject index.

ii.
```python
subjects = sorted({r['subject'] for r in results})
subject_index = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array([subject_index[r['subject']] for r in results], dtype=np.int64),
```

iii. The subject identity is already present in the freeze file, so no parsing or derivation is needed.

## 1-c. How are the data split into sessions?

i. Each unique `eid` in `bwm_release.csv` is one session. The groupby on `eid` produces one entry per session with its associated probe insertions.

ii.
```python
for eid, g in bwm.groupby('eid', sort=False):
    sessions.append({'eid': str(eid), 'pids': [str(p) for p in g.pid], ...})
```

iii. Sessions are the natural unit of the release; no splitting is required.

## 1-d. How are the data split into trials?

i. The trials table (`_ibl_trials.table.pqt`) loaded via `SessionLoader` has one row per trial. No further splitting is needed.

ii.
```python
sess_loader = SessionLoader(one=one, eid=eid)
trials_df, trials_mask = load_trials_and_mask(one, eid, sess_loader=sess_loader)
```

iii. The trials table is already one row per trial by design.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies the full set of trial curation criteria from the Zhang et al. reference code's `load_trials_and_mask` with `prepare_data`'s parameters:
- NaN exclusion for `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`
- Reaction time (firstMovement_times - stimOn_times) outside [0.08, 2.0] s
- No-response trials (choice == 0) dropped
- Trial length (feedback_times - goCue_times) > 10 s dropped
- Additionally, trials whose wheel or whisker trace does not cover the full window are dropped (via `bin_behaviour`'s coverage check)

ii.
```python
MIN_RT, MAX_RT, MAX_TRIAL_LEN = 0.08, 2.0, 10.0
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']

def load_trials_and_mask(one, eid, sess_loader=None):
    ...
    query = f'(firstMovement_times - stimOn_times < {MIN_RT})'
    query += f' | (firstMovement_times - stimOn_times > {MAX_RT})'
    query += f' | (feedback_times - goCue_times > {MAX_TRIAL_LEN})'
    for event in NAN_EXCLUDE:
        query += f' | {event}.isnull()'
    query += ' | (choice == 0)'
    mask = ~trials.eval(query)
    return trials, mask.to_numpy()
```

And behaviour coverage:
```python
keep &= beh_good[name]  # trials where behaviour interpolation succeeded
```

iii. The AI explicitly follows the Zhang reference code's `load_trials_and_mask` defaults plus the `max_trial_len=10.0` parameter from `prepare_data`. The behaviour coverage check mirrors `get_behavior_per_interval`'s trial rejection logic.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times` and `spikes.clusters` from each probe's spike sorting, loaded via `SpikeSortingLoader`. The cluster table provides the QC label and anatomical region for filtering.

ii.
```python
loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
spikes, clusters, channels = loader.load_spike_sorting()
clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
```

iii. These are the standard spike sorting outputs from the IBL pipeline.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over the trial window [-0.5, 1.5] s relative to stimulus onset, giving one count per unit per bin. The counts are stored as **spike counts** (float32), NOT divided by the bin width (i.e., not converted to firing rates). When a session has multiple probes, their units are concatenated into one population.

ii.
```python
def bin_spikes(times, clusters, n_clusters, interval_begs, binsize=BINSIZE, n_bins=N_BINS):
    ...
    t_rel = times[flat_idx] - interval_begs[trial_of_spike]
    bin_of_spike = np.floor(t_rel / binsize).astype(np.int64)
    np.clip(bin_of_spike, 0, n_bins - 1, out=bin_of_spike)
    lin = ((trial_of_spike * n_clusters) + clusters[flat_idx]) * n_bins + bin_of_spike
    out += np.bincount(lin, minlength=n_trials * n_clusters * n_bins).reshape(
        n_trials, n_clusters, n_bins).astype(np.float32)
    return out
```

Probes concatenated:
```python
neural = np.concatenate(neural_trials, axis=1)  # (n_trials, n_neurons, T)
```

iii. The AI documents that the binning arithmetic matches `bincount2D` as used by the reference `get_spike_data_per_interval`, and was verified to be bit-identical. The choice to keep spike counts (rather than converting to Hz) follows the Zhang reference code, which also stores counts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three filters applied:
1. Only clusters with `label >= 1` (the "well-isolated" units from the IBL's RIGOR QC metrics)
2. Clusters whose Beryl acronym is `root` or `void` are dropped (grey matter restriction)
3. Clusters that emit no spikes in the session are dropped
Additionally, sessions with fewer than 5 well-isolated grey-matter neurons are dropped entirely.

ii.
```python
QC_LABEL = 1.0
NON_GREY_MATTER = ('root', 'void')
MIN_NEURONS_PER_SESSION = 5

beryl = np.asarray(br.acronym2acronym(clusters_df['acronym'].to_numpy(), mapping='Beryl'), dtype=object)
has_spikes = np.zeros(len(clusters_df), dtype=bool)
if clu.size:
    has_spikes[np.unique(clu)] = True
sel = (~np.isin(beryl, NON_GREY_MATTER)) & has_spikes

if neural.shape[1] < MIN_NEURONS_PER_SESSION:
    raise RuntimeError(f'only {neural.shape[1]} well-isolated grey-matter neurons')
```

iii. The AI justifies `label >= 1` by citing the data paper's explicit neuron inclusion criteria (75,708 well-isolated neurons). The grey-matter restriction (dropping both `root` and `void`) is justified by the data paper's statement that analyses are restricted to grey-matter regions. The `has_spikes` filter matches the reference `bin_spiking_data` which uses `np.unique(regclu)`. The MIN_NEURONS=5 threshold is cited from the data paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to stimulus onset (`stimOn_times`) by computing `interval_begs = stimOn_times + TIME_WINDOW[0]` and then binning spikes relative to this start time. This places time zero at the stimulus onset.

ii.
```python
align_times = trials_df[ALIGN_TIME].to_numpy(dtype=float)
interval_begs = align_times + TIME_WINDOW[0]
...
binned = bin_spikes(times[spike_sel], row_of[clu[spike_sel]],
                    int(sel.sum()), interval_begs[keep_idx])
```

iii. All signals (spikes, wheel, camera) share the same session clock, so alignment is achieved by subtracting the stimulus onset time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, producing T = 100 bins per 2 s trial. No rebinning or smoothing is applied.

ii.
```python
BINSIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
```

iii. The AI cites the reference code's `binsize=0.02` and the method paper's "2-s trials, each divided into 20-ms bins, producing T = 100 time steps".

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Derived from `stimOn_times` in the trials table (the alignment event) combined with the bin grid parameters.

ii.
```python
bin_centres = (TIME_WINDOW[0] + BINSIZE * (np.arange(N_BINS) + 0.5)).astype(np.float32)
inputs[:, 0, :] = bin_centres[None, :]
```

iii. The time values are the bin centres, ranging from -0.49 to +1.49 s. The same values apply to every trial since they are defined relative to stimulus onset.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing of raw data; the variable is constructed from the bin grid definition. Bin centres are computed as `T_START + BINSIZE * (i + 0.5)` for i = 0..99.

ii.
```python
bin_centres = (TIME_WINDOW[0] + BINSIZE * (np.arange(N_BINS) + 0.5)).astype(np.float32)
```

iii. This is a deterministic function of the bin parameters, identical for every trial.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time values ARE the bin centres of the neural binning grid, so they are inherently aligned: bin i of the neural data and bin i of the time input correspond to the same 20 ms window.

ii.
```python
inputs[:, 0, :] = bin_centres[None, :]  # same N_BINS=100 as neural
```

iii. No separate alignment is needed; the input IS the time axis.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from `probabilityLeft` in the trials table. A block is defined as a contiguous run of trials with the same `probabilityLeft` value.

ii.
```python
def trial_number_in_block(probability_left):
    p = np.asarray(probability_left, dtype=float)
    out = np.zeros(len(p), dtype=np.float64)
    counter = 0
    for i in range(1, len(p)):
        same = (p[i] == p[i - 1]) or (np.isnan(p[i]) and np.isnan(p[i - 1]))
        counter = counter + 1 if same else 0
        out[i] = counter
    return out
```

iii. The trials table has no block identifier, so blocks must be recovered from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A 0-based counter that increments within each block and resets to 0 when `probabilityLeft` changes. Computed on the **full** (uncurated) trials table so that the number reflects the animal's true position in the block even when neighbouring trials are later excluded.

ii.
```python
tnb_all = trial_number_in_block(trials_df['probabilityLeft'].to_numpy())
# ... later, after filtering:
inputs[:, 1, :] = tnb_all[keep_idx].astype(np.float32)[:, None]
```

iii. Computing before filtering ensures the block count is the animal's real position. The value is broadcast over all time bins (per-trial constant).

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, where +1 = left choice, -1 = right choice, 0 = no response.

ii.
```python
choice = trials_df['choice'].to_numpy()[keep_idx]
choice_cls = (choice < 0).astype(np.int64)  # left -> 0, right -> 1
```

iii. The IBL convention is +1 for left, -1 for right. No-response trials (choice == 0) are already excluded by the trial mask.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Recoded to binary: left (+1) -> 0, right (-1) -> 1. The value is broadcast over all 100 time bins.

ii.
```python
choice_cls = (choice < 0).astype(np.int64)
outputs[:, 0, :] = choice_cls[:, None]
```

iii. The mapping follows the decoder task specification (left = 0, right = 1).

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, which takes values 0.2, 0.5, or 0.8.

ii.
```python
pleft = trials_df['probabilityLeft'].to_numpy()[keep_idx]
pleft_cls = np.full(len(pleft), -1, dtype=np.int64)
for val, cls in ((0.2, 0), (0.5, 1), (0.8, 2)):
    pleft_cls[np.isclose(pleft, val)] = cls
```

iii. The three values represent the block prior probabilities used in the IBL task.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Mapped to categorical classes: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2, using `np.isclose` for floating-point comparison. The value is broadcast over all time bins. An error is raised if any unexpected value is encountered.

ii.
```python
for val, cls in ((0.2, 0), (0.5, 1), (0.8, 2)):
    pleft_cls[np.isclose(pleft, val)] = cls
if np.any(pleft_cls < 0):
    bad = np.unique(pleft[pleft_cls < 0])
    raise RuntimeError(f'unexpected probabilityLeft values {bad}')
outputs[:, 1, :] = pleft_cls[:, None]
```

iii. The mapping follows the decoder task specification (0.2 -> 0, 0.5 -> 1, 0.8 -> 2).

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The wheel position and timestamps (`_ibl_wheel.position`, `_ibl_wheel.timestamps`) loaded via `SessionLoader.load_wheel()`, which internally interpolates to 1 kHz and computes velocity with a Butterworth low-pass filter.

ii.
```python
sess_loader.load_wheel()
traces['wheel_speed'] = (sess_loader.wheel['times'].to_numpy(),
                         np.abs(sess_loader.wheel['velocity'].to_numpy()))
```

iii. The wheel speed is the absolute value of the velocity provided by `SessionLoader`, matching the reference's `wheel-speed = np.abs(velocity)`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps: (1) `SessionLoader.load_wheel()` interpolates the wheel position to a uniform 1 kHz grid and computes velocity using a Butterworth low-pass filter. (2) The absolute velocity is linearly interpolated onto the **right edges** of the 100 bins of each trial (i.e., at `np.linspace(t_beg + binsize, t_end, n_bins)`). (3) The continuous values are discretized into 3 classes using per-session tercile thresholds.

ii.
```python
def bin_behaviour(target_times, target_vals, interval_begs, binsize=BINSIZE, n_bins=N_BINS):
    ...
    x_interp = np.linspace(interval_begs[k] + binsize, interval_ends[k], n_bins)
    y = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x_interp)
    ...
```

iii. The interpolation to bin right edges matches the Zhang reference code's `get_behavior_per_interval`, which uses the same `np.linspace(t_beg + binsize, t_end, n_bins)` formula. The AI verified this produces identical results.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Per-session terciles: the 33.3% and 66.7% quantiles of all wheel speed values across all retained trials and time bins of the session define two edges. Values <= e1 become class 0, e1 < v <= e2 become class 1, v > e2 become class 2. Degenerate cases (e1 == e2) are handled by splitting the remaining mass.

ii.
```python
def discretize_terciles(values):
    v = np.asarray(values, dtype=np.float64).ravel()
    e1, e2 = np.quantile(v, [1.0 / 3.0, 2.0 / 3.0])
    if e1 == e2:
        above = v[v > e1]
        e2 = np.quantile(above, 0.5) if above.size else e1
        ...
    return float(e1), float(e2)

def apply_terciles(values, e1, e2):
    return ((values > e1).astype(np.int64) + (values > e2).astype(np.int64))
```

iii. Per-session terciles ensure approximately equal class sizes (1/3 each) regardless of session-specific scaling, consistent with the reference code's per-session standardization approach.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is interpolated at the right edge of each neural bin, sharing the same trial alignment (stimulus onset) and window. Bin i of the wheel speed corresponds to the same 20 ms period as bin i of the neural data, though the behavior sample is taken at the bin's right edge rather than its centre.

ii.
```python
x_interp = np.linspace(interval_begs[k] + binsize, interval_ends[k], n_bins)
```

iii. The wheel is on the same session clock as the spikes, so evaluating at the bin edges provides alignment.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The motion energy of the side camera (`leftCamera.ROIMotionEnergy` preferred, `rightCamera.ROIMotionEnergy` as fallback) with its frame times.

ii.
```python
for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sess_loader.load_motion_energy(views=[view])
        me = sess_loader.motion_energy[key]
        traces['whisker_motion_energy'] = (me['times'].to_numpy(),
                                           me['whiskerMotionEnergy'].to_numpy())
        cam_used = view
        break
    except Exception:
        continue
```

iii. Left camera preferred, right as fallback, matching the reference `bin_behaviors` logic.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion energy trace is used as-is (no filtering or normalization). It is linearly interpolated onto the right edges of the 100 bins of each trial, then discretized into 3 per-session tercile classes, identical to the wheel speed processing.

ii.
```python
vals, good = bin_behaviour(tt, tv, interval_begs)
...
wme_edges = discretize_terciles(wme)
wme_cls = apply_terciles(wme, *wme_edges)
```

iii. Same processing pipeline as wheel speed. The AI notes that no additional filtering is needed since the released trace is already processed by the IBL pipeline.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identical to wheel speed: per-session terciles at 33.3%/66.7% quantiles, with degenerate edge handling.

ii. Same `discretize_terciles` and `apply_terciles` functions as wheel speed.

iii. Per-session thresholds are especially important here because the left camera (60 Hz, 1280x1024) and right camera (150 Hz, 640x512) produce different absolute motion energy values.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: interpolated at the right edges of the neural bins, sharing the same trial alignment and window.

ii.
```python
x_interp = np.linspace(interval_begs[k] + binsize, interval_ends[k], n_bins)
```

iii. The camera frame times are on the same session clock as the neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple layers of handling:
- Trials with NaN in any required field are dropped by the trial mask
- Trials where behaviour interpolation fails (missing/insufficient data) are dropped (`good` mask in `bin_behaviour`)
- Trials with NaN in interpolated values are dropped
- Probes with no clusters are skipped
- Sessions with fewer than 2 usable trials raise an error and are skipped
- Sessions with fewer than 5 well-isolated grey-matter neurons are skipped
- Sessions with no whisker motion energy from either camera are skipped
- Degenerate tercile edges (e.g., 78% of whisker ME samples exactly zero) are handled with a fallback split

ii.
```python
# Behaviour coverage check
if np.abs(interval_begs[k] - tt[0]) > binsize: continue
if np.abs(interval_ends[k] - tt[-1]) > binsize: continue
if not np.all(np.isfinite(y)): continue

# Session-level filters
if len(keep_idx) < MIN_TRIALS_PER_SESSION:
    raise RuntimeError(...)
if neural.shape[1] < MIN_NEURONS_PER_SESSION:
    raise RuntimeError(...)
```

iii. Failed sessions and their reasons are recorded in `metadata['failed_sessions']` for transparency.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting from disk dominates (3-7 s per probe), as the spike arrays are hundreds of megabytes. Trial and behaviour loading is ~0.5-0.6 s per session. The full conversion of 459 sessions with 24 workers takes ~4.5 minutes.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()  # the expensive call
```

iii. The cost is mainly file I/O, with session-level parallelism (ProcessPoolExecutor) providing the speedup.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The behaviour interpolation loops over trials (`for k in range(n_trials)` in `bin_behaviour`), doing a separate `interp1d` call for each trial. This could be vectorized by concatenating all trial query points. However, the cost is minimal (~0.05 s/session) since the spike I/O dominates.

ii.
```python
for k in range(n_trials):
    ...
    x_interp = np.linspace(interval_begs[k] + binsize, interval_ends[k], n_bins)
    y = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x_interp)
```

iii. The AI notes this loop is not a bottleneck compared to disk I/O.

## 10-c. What processing does the code repeat multiple times?

i. The AI creates a new ONE client (`get_one()`) in each worker process, which requires reading the cache tables each time. This is necessary for multiprocessing (the ONE client cannot be shared across forks) but is repeated work.

ii.
```python
def convert_session(session_info, ...):
    ...
    one = get_one()
```

iii. This is an inherent requirement of process-based parallelism, not a design inefficiency.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. No significant unnecessary processing was identified. The AI loads only the two required behaviour traces (wheel and whisker), unlike the Zhang reference which loads six. Extra metadata fields (timings, tercile edges, runtime) are stored for debugging but don't represent expensive computation.

ii. N/A

iii. The AI specifically optimized to avoid the reference code's redundant loads.
