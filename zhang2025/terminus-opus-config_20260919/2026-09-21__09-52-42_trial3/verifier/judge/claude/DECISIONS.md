# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses the `bwm_release.csv` freeze file (the same session list used by the Zhang et al. reference code) to enumerate sessions, rather than searching the ONE release index directly. It builds an offline ONE client against a rebuilt local index (`LocalIndex`), because the shipped release tables were stale and did not index revision-folder files. The `DATALIMIT_SUBSET.csv` file is honored if present. For each session, `SpikeSortingLoader` loads spike sorting per probe, `SessionLoader` loads trials, wheel, and motion energy.

ii.
```python
FREEZE_FILE = '/app/code/code_zhang2025/data/bwm_release.csv'
DATALIMIT = '/app/data/DATALIMIT_SUBSET.csv'

def session_list():
    bwm = pd.read_csv(FREEZE_FILE, index_col=0)
    if os.path.exists(DATALIMIT):
        keep = pd.read_csv(DATALIMIT)
        col = 'eid' if 'eid' in keep.columns else keep.columns[0]
        bwm = bwm[bwm.eid.isin(set(keep[col].astype(str)))]
    return bwm

def get_one():
    from one.api import ONE
    ...
    return ONE(base_url='https://openalyx.internationalbrainlab.org', mode='local',
               cache_dir=CACHE_DIR, tables_dir=TABLES_DIR)
```

iii. The AI discovered that the shipped ONE release tables were stale (predating revision folders), causing `SessionLoader.load_trials()` to silently return truncated data. It rebuilt the dataset index from the filesystem and used `bwm_release.csv` as the session list because `one.eid2pid()` requires a remote connection unavailable in local mode.

## 1-b. How are the data split into subjects?

i. Subject names come from the `bwm_release.csv` freeze file (which has a `subject` column for each probe insertion). Subjects are extracted as sorted unique names from the results, and `subject_idx` maps each session to its subject index.

ii.
```python
subjects = sorted({r['subject'] for r in results})
sub_idx = {s: i for i, s in enumerate(subjects)}
# ...
'subject_idx': np.array([sub_idx[r['subject']] for r in results], dtype=int),
```

iii. The freeze file already contains subject information per probe/session.

## 1-c. How are the data split into sessions?

i. Sessions are identified by unique `eid` values from `bwm_release.csv`. Each unique eid is one session. Probes within a session are grouped by eid from the freeze file.

ii.
```python
eids = list(dict.fromkeys(bwm.eid))
# ...
jobs = [(e, bwm[bwm.eid == e], ...) for i, e in enumerate(eids)]
```

iii. Sessions are the natural unit in the BWM release.

## 1-d. How are the data split into trials?

i. The trials table loaded by `SessionLoader.load_trials()` has one row per trial. Trials are indexed by their row position.

ii.
```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
tr = sl.trials
```

iii. The trials table is already one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies the reference `load_trials_and_mask` criteria: NaN exclusion for six columns (`stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`), reaction time bounds (0.08-2.0 s), maximum trial length (`feedback_times - goCue_times > 10 s`), and no-choice exclusion (`choice == 0`). Additionally, trials are dropped if wheel or whisker motion energy doesn't cover the trial window. Trials with invalid `probabilityLeft` values (not in {0.2, 0.5, 0.8}) are also dropped.

ii.
```python
MIN_RT, MAX_RT, MAX_TRIAL_LEN = 0.08, 2.0, 10.0
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']

def trials_and_mask(one, eid):
    ...
    rt = tr['firstMovement_times'] - tr['stimOn_times']
    bad = (rt < MIN_RT) | (rt > MAX_RT)
    bad |= (tr['feedback_times'] - tr['goCue_times']) > MAX_TRIAL_LEN
    bad |= tr['choice'] == 0
    for col in NAN_EXCLUDE:
        bad |= tr[col].isna()
    return tr, (~bad).to_numpy(), sl
```

iii. The AI explicitly followed the reference code's `load_trials_and_mask` function, including the `max_trial_len=10.0` parameter and the full NaN exclusion list.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times` and `spikes.clusters` from each probe, loaded via `SpikeSortingLoader`. The cluster table provides quality labels and anatomical information for filtering.

ii.
```python
ssl = SpikeSortingLoader(pid=r.pid, one=one, eid=eid, pname=r.probe_name)
spikes, clusters, channels = ssl.load_spike_sorting()
# ...
times.append(spikes['times'])
clus.append(spikes['clusters'] + offset)
```

iii. These are the standard spike sorting outputs from the IBL pipeline.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over the trial window (-0.5 to 1.5 s around stimulus onset). Spike counts are stored directly as float32, **not** converted to firing rates. When a session has multiple probes, their units are merged with offset cluster IDs.

ii.
```python
def bin_spikes(spike_times, spike_clusters, unit_idx, align_times):
    ...
    out = np.zeros((len(align_times), n_units, N_BINS), dtype=np.float32)
    for k in range(len(align_times)):
        ...
        tb = ((st[s] - begs[k]) / BINSIZE).astype(np.int64)
        np.clip(tb, 0, N_BINS - 1, out=tb)
        flat = np.bincount(sc[s] * N_BINS + tb, minlength=n_units * N_BINS)
        out[k] = flat.reshape(n_units, N_BINS)
    return out
```

The result is used directly without dividing by BIN:
```python
counts = bin_spikes(spike_times, spike_clusters, unit_idx, align)
# ...
neural=[counts[i] for i in range(n_tr)],
```

iii. The AI's metadata says `'neural_units': 'spike counts per 20 ms bin (float32)'`. The reference code bins spikes the same way but stores as firing rates (counts/BIN).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` are kept (all three RIGOR QC metrics passed). Additionally, units whose Beryl acronym is `root` or `void` are excluded (described as "grey matter" filtering). Sessions with fewer than 5 surviving units are dropped entirely.

ii.
```python
NON_GREY = ('root', 'void')

def good_grey_units(clusters):
    ...
    beryl = np.asarray(BrainRegions().acronym2acronym(
        clusters['acronym'].to_numpy(), mapping='Beryl'), dtype=object)
    keep = (clusters['label'].to_numpy() >= 1) & ~np.isin(beryl, NON_GREY)
    return np.where(keep)[0], beryl

MIN_UNITS_PER_SESSION = 5
# ...
if len(unit_idx) < MIN_UNITS_PER_SESSION:
    raise RuntimeError(...)
```

iii. The AI justified the `label >= 1` filter as matching the paper's 75,708 well-isolated neurons. The `root`/`void` exclusion is described as restricting to grey matter. The `MIN_UNITS_PER_SESSION = 5` threshold was added to address all-zero-trial warnings from sessions with very few neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `stimOn_times` (stimulus onset). The window spans -0.5 to +1.5 s around the onset. Spike times within this window are binned relative to the window start.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
# ...
align = trials[ALIGN_TIME].to_numpy()[keep]
# ...
begs = align_times + TIME_WINDOW[0]
ends = align_times + TIME_WINDOW[1]
# ...
tb = ((st[s] - begs[k]) / BINSIZE).astype(np.int64)
```

iii. All streams share the same session clock, so alignment is simply subtracting the onset time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20 ms, giving 100 bins over the 2 s window. No rebinning or interpolation is applied to the neural data. The time grid for behavior and inputs uses bin end times: `np.linspace(-0.48, 1.5, 100)`.

ii.
```python
BINSIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
# ...
tgrid = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
```

iii. The 20 ms bin size and 100 bins match the reference code parameters and the paper's "T = 100 time steps".

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Derived from the trial time window parameters and bin size, not from any raw data variable. The time grid is constructed as `np.linspace(-0.48, 1.5, 100)`, representing bin end times.

ii.
```python
tgrid = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
inputs[:, 0, :] = tgrid.astype(np.float32)
```

iii. This follows the reference code's `get_behavior_per_interval` convention of using bin end times.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing of raw data. The time grid is a deterministic sequence of bin end times from -0.48 to 1.5 s in 100 steps, broadcast identically to every trial.

ii.
```python
tgrid = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
inputs[:, 0, :] = tgrid.astype(np.float32)
```

iii. The time values are defined by the binning parameters.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time grid uses bin end times (`linspace(-0.48, 1.5, 100)`), while the neural binning uses `floor((t - beg)/binsize)` which assigns bin i to the interval `[off_start + i*binsize, off_start + (i+1)*binsize)`. This means the time input represents the right edge of each neural bin.

ii.
```python
# Neural binning:
tb = ((st[s] - begs[k]) / BINSIZE).astype(np.int64)
# Time input:
tgrid = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
```

iii. The AI explicitly noted in the metadata that the input time reports "bin end times", matching the reference `get_behavior_per_interval` convention.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. Block boundaries are detected where `probabilityLeft` changes value.

ii.
```python
def trial_in_block(prob_left):
    pl = np.asarray(prob_left, dtype=float)
    new = np.ones(len(pl), dtype=bool)
    new[1:] = pl[1:] != pl[:-1]
    idx = np.arange(len(pl))
    return idx - np.maximum.accumulate(np.where(new, idx, 0))
```

iii. The trials table has no explicit block identifier, so blocks are inferred from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The trial-in-block counter is computed over the full trial sequence before masking, so removed trials still advance the count. The count is 0-based within each block. The value is broadcast as a constant across all 100 time bins of a trial.

ii.
```python
tib_all = trial_in_block(trials['probabilityLeft'].to_numpy())
# ...
tib = tib_all[keep].astype(np.float32)
inputs[:, 2, :] = tib[:, None]
```

iii. Computing before masking preserves the animal's true position in the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column in the trials table, which takes values +1 (left), -1 (right), and 0 (no response).

ii.
```python
choice = trials['choice'].to_numpy()[keep]
y_choice = np.where(choice > 0, 0, 1).astype(np.int16)   # +1 left -> 0, -1 right -> 1
```

iii. The choice sign convention was verified empirically by the AI in Step 4.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Recoding: +1 (left) -> 0, -1 (right) -> 1. No-response trials (choice == 0) are already excluded by the trial mask. The per-trial value is broadcast across all 100 time bins.

ii.
```python
y_choice = np.where(choice > 0, 0, 1).astype(np.int16)
outputs[:, 0, :] = y_choice[:, None]
```

iii. Direct mapping as specified by the task instructions.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From the `probabilityLeft` column in the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
pl = trials['probabilityLeft'].to_numpy()[keep]
y_prior = np.select([np.isclose(pl, 0.2), np.isclose(pl, 0.5), np.isclose(pl, 0.8)],
                    [0, 1, 2], default=-1).astype(np.int16)
```

iii. The mapping follows the task specification.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Recoding: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. Uses `np.isclose` for floating point comparison. Trials with unexpected values get -1 and are subsequently filtered out. The per-trial value is broadcast across all 100 time bins.

ii.
```python
y_prior = np.select([np.isclose(pl, 0.2), np.isclose(pl, 0.5), np.isclose(pl, 0.8)],
                    [0, 1, 2], default=-1).astype(np.int16)
if (y_prior < 0).any():
    okp = y_prior >= 0
    keep, align, counts, wheel, whisk = (keep[okp], align[okp], counts[okp],
                                         wheel[okp], whisk[okp])
    y_choice, y_prior = y_choice[okp], y_prior[okp]
outputs[:, 1, :] = y_prior[:, None]
```

iii. Direct mapping as specified. Using `np.isclose` is more robust than exact equality for float comparisons.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.position` and `_ibl_wheel.timestamps`, loaded via `SessionLoader.load_wheel()`. The wheel velocity is computed internally by `SessionLoader`, and speed is the absolute value of velocity.

ii.
```python
sl.load_wheel()
out['wheel_speed'] = (sl.wheel['times'].to_numpy(),
                      np.abs(sl.wheel['velocity'].to_numpy()))
```

iii. Same source and processing as the reference code's `load_target_behavior('wheel-speed')`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. `SessionLoader.load_wheel()` interpolates the raw wheel position onto a 1000 Hz grid and applies a Butterworth low-pass filter to compute velocity. Speed is `abs(velocity)`. The speed trace is linearly interpolated onto the 100-bin grid at bin end times (`linspace(beg + binsize, end, n_bins)`). Trials where the trace doesn't cover the window are marked invalid. Then the trace is discretized into 3 classes at per-session tertiles (33rd and 67th percentiles).

ii.
```python
wt, wv = beh['wheel_speed']
wheel, ok_w = bin_behaviour(wt, wv, align)
# ...
grid = np.linspace(begs[k] + BINSIZE, ends[k], N_BINS)
out[k] = interp1d(t[finite], v[finite], kind='linear',
                  fill_value='extrapolate')(grid)
# ...
y_wheel, thr_w = tertile_bins(wheel)
```

iii. The interpolation matches the reference code's `get_behavior_per_interval`. Tertile discretization is justified as producing balanced classes.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Per-session tertiles: the 33rd and 67th percentiles of all finite wheel speed values across all retained (trial, time-bin) samples in the session. Values are digitized into 3 classes: low (0), medium (1), high (2).

ii.
```python
def tertile_bins(values):
    finite = values[np.isfinite(values)]
    lo, hi = np.percentile(finite, [100 / 3, 200 / 3])
    if not (hi > lo):
        hi = lo + 1e-12
    return (np.digitize(values, [lo, hi]).astype(np.int16), (float(lo), float(hi)))
```

iii. Per-session tertiles give balanced classes and handle session-specific signal scales.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated onto the same time grid used for behavior: `linspace(beg + binsize, end, n_bins)` (bin end times). This differs slightly from the neural binning (which uses bin start boundaries for spike assignment), but each array index i corresponds to the same bin.

ii.
```python
grid = np.linspace(begs[k] + BINSIZE, ends[k], N_BINS)
out[k] = interp1d(t[finite], v[finite], kind='linear',
                  fill_value='extrapolate')(grid)
```

iii. The behavior grid matches the reference code's `get_behavior_per_interval` convention.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `<side>Camera.ROIMotionEnergy` and `_ibl_<side>Camera.times`, loaded via `SessionLoader.load_motion_energy()`. The left camera is preferred; the right is used as fallback.

ii.
```python
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sl.load_motion_energy(views=[view])
        df = sl.motion_energy[cam]
        vals = df['whiskerMotionEnergy'].to_numpy()
        if np.isfinite(vals).any():
            me = (df['times'].to_numpy(), vals, view)
            break
    except Exception:
        continue
```

iii. Follows the reference code's left-then-right camera preference.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy trace is used as-is (no filtering or normalization). It is linearly interpolated onto the bin end times grid, then discretized into 3 classes at per-session tertiles, identical to wheel speed processing.

ii.
```python
mt, mv, me_side = beh['whisker_motion_energy']
whisk, ok_m = bin_behaviour(mt, mv, align)
# ...
y_whisk, thr_m = tertile_bins(whisk)
```

iii. Same interpolation and discretization pipeline as wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Per-session tertiles, identical to wheel speed: 33rd and 67th percentile thresholds, producing 3 balanced classes.

ii.
```python
y_whisk, thr_m = tertile_bins(whisk)
```

iii. Same justification as wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: interpolated at bin end times using `linspace(beg + binsize, end, n_bins)`.

ii.
```python
grid = np.linspace(begs[k] + BINSIZE, ends[k], N_BINS)
out[k] = interp1d(t[finite], v[finite], kind='linear',
                  fill_value='extrapolate')(grid)
```

iii. Consistent with the reference code's behavior resampling convention.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple levels of handling: (1) NaN values in key trial columns cause trial exclusion via the mask. (2) Trials where wheel or whisker ME doesn't cover the window are excluded via the `bin_behaviour` validity check. (3) Sessions with no whisker ME at all are skipped. (4) Sessions with fewer than 5 well-isolated grey-matter units are skipped. (5) Sessions with fewer than 2 valid trials are skipped. (6) Probes with no spike sorting data are skipped. (7) NaN values in behavior traces are handled by interpolating over finite samples only.

ii.
```python
# Behavior coverage check:
if abs(begs[k] - t[0]) > BINSIZE or abs(ends[k] - t[-1]) > BINSIZE:
    continue
finite = np.isfinite(v)
if finite.sum() < 2:
    continue

# Session skip:
if len(unit_idx) < MIN_UNITS_PER_SESSION:
    raise RuntimeError(...)
if beh['whisker_motion_energy'] is None:
    raise RuntimeError('no whisker motion energy (left or right)')
```

iii. The AI documented all skip conditions and verified that each case is handled explicitly rather than silently producing bad data.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting from disk is the dominant cost (~2.4 s per session according to the AI's timing). The AI's code reports per-step timings. Spike binning is the second most expensive step (~0.35 s per session).

ii.
```python
t = time.time()
spike_times, spike_clusters, clusters = load_session_spikes(one, eid, probes)
timings['load_spikes'] = time.time() - t
```

iii. The AI measured and reported timings in CONVERSION_NOTES.md Step 7.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two per-trial loops: the spike binning loop (`for k in range(len(align_times))`) and the behavior interpolation loop (`for k in range(n)`). The spike binning loop does a `bincount` per trial, and the behavior loop does an `interp1d` per trial. Both could potentially be vectorized.

ii.
```python
# Spike binning loop:
for k in range(len(align_times)):
    ...
    flat = np.bincount(sc[s] * N_BINS + tb, minlength=n_units * N_BINS)
    out[k] = flat.reshape(n_units, N_BINS)

# Behavior interpolation loop:
for k in range(n):
    ...
    out[k] = interp1d(t[finite], v[finite], kind='linear',
                      fill_value='extrapolate')(grid)
```

iii. The loops are retained because each trial uses a different slice of the data, making full vectorization complex.

## 10-c. What processing does the code repeat multiple times?

i. The ONE client is created independently in each worker process (`get_one()` is called per session in the worker). The ONE cache index may be rebuilt if missing. The `BrainRegions()` object is instantiated once per call to `good_grey_units`. The behavior loading (`load_behaviour`) calls `SessionLoader.load_wheel()` and `SessionLoader.load_motion_energy()` which may repeat internal processing.

ii.
```python
def convert_session(eid, probes, show=False, outdir='/app'):
    one = get_one()
    # ... each worker creates its own ONE client
```

iii. The per-worker ONE client creation is necessary for process isolation in multiprocessing.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores an extra input variable `stim_onset` (a binary indicator marking the stimulus onset bin) that is not requested by the task instructions. The task only asks for "Time since stimulus onset" and "Trial number in block" as inputs. Additionally, the AI stores neural data as spike counts rather than firing rates, which may require the decoder to learn the bin-width scaling.

ii.
```python
INPUT_NAMES = ['time_from_stim_onset', 'stim_onset', 'trial_in_block']
# ...
onset = np.zeros(N_BINS, dtype=np.float32)
onset_bin = int(round(-TIME_WINDOW[0] / BINSIZE))
onset[onset_bin] = 1.0
inputs[:, 1, :] = onset
```

iii. The AI justified the `stim_onset` indicator by citing the instruction: "If an input is a time such as onset of some stimulus, represent it as a binary time series." However, the decoder task only specifies two inputs.
