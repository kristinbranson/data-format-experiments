# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses the ONE API in local/offline mode against the staged ONE cache at `/app/data/one_cache`. It reads the session list from `bwm_release.csv` (shipped with the zhang2025 reference code), which contains 459 session eids with their probe insertion IDs and probe names. Because the on-disk files use newer revisions than the shipped cache tables (e.g., `#2025-03-03#` for trials), the AI rebuilds the ONE cache tables by walking the `alf/` directory trees on disk, registering the newest revision of each dataset as the default. Sessions are filtered to those available locally. For each session, `SessionLoader` loads trials, wheel, and motion energy, and `SpikeSortingLoader` loads spike sorting per probe.

ii.
```python
FREEZE_FILE = '/app/code/code_zhang2025/data/bwm_release.csv'
bwm_df = pd.read_csv(FREEZE_FILE, index_col=0)
one = get_one()
available = set(one._cache['sessions'].index.astype(str))
eids = [e for e in bwm_df.eid.unique() if e in available]
```

Cache rebuild:
```python
def rebuild_cache_tables():
    # walks alf/ trees, registers newest revision as default
    ...
    parquet.save(root / 'datasets.pqt', ds, meta)
    parquet.save(root / 'sessions.pqt', sessions, meta)
```

iii. The AI discovered that the shipped ONE cache tables did not contain the newer file revisions present on disk (trajectoy steps 30-55). After extensive debugging of ONE's dataset resolution, it built `rebuild_cache_tables()` to generate a merged cache from the files actually staged on disk.

## 1-b. How are the data split into subjects?

i. Subject names come from the `bwm_release.csv` file and from session metadata. During assembly, subjects are collected in encounter order and `subject_idx` maps each session to its position in the subjects list.

ii.
```python
sub = str(meta['subject'])
if sub not in subjects:
    subjects.append(sub)
subject_idx.append(subjects.index(sub))
```

iii. The subject is available directly from the session metadata; no parsing needed.

## 1-c. How are the data split into sessions?

i. Sessions are the unit of the dataset. Each eid from `bwm_release.csv` is one session, processed independently.

ii.
```python
eids = [e for e in bwm_df.eid.unique() if e in available]
jobs = [(e, bwm_df[bwm_df.eid == e]) for e in eids]
```

iii. No splitting needed; the data is already organized by session.

## 1-d. How are the data split into trials?

i. The trials table (`_ibl_trials.table.pqt`) has one row per trial. The AI loads trials via `SessionLoader.load_trials()` and then applies filtering via `load_trials_and_mask`.

ii.
```python
sess_loader = SessionLoader(one=one, eid=eid)
sess_loader.load_trials()
trials_df, trials_mask = load_trials_and_mask(
    one=one, eid=eid, max_trial_len=MAX_TRIAL_LEN, sess_loader=sess_loader)
```

iii. The trials table is already one row per trial; no additional splitting is needed.

## 1-e. How are trials filtered based on quality controls?

i. The AI uses the reference code's `load_trials_and_mask(max_trial_len=10.0)` function, which filters trials based on: no NaN in stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, or feedbackType; reaction time between 0.08 and 2 seconds; goCue-to-feedback duration less than 10 seconds; and a choice was made. Additionally, trials whose wheel speed or whisker motion energy traces do not cover the decoding window are dropped.

ii.
```python
trials_df, trials_mask = load_trials_and_mask(
    one=one, eid=eid, max_trial_len=MAX_TRIAL_LEN, sess_loader=sess_loader)
...
keep = trials_mask & wheel_ok & me_ok & ~np.isnan(align)
```

iii. The AI explicitly chose to use `load_trials_and_mask` from the zhang2025 reference code (trajectory step 65), citing consistency with the reference pipeline. The `max_trial_len=10.0` parameter is taken directly from the reference code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times` (spike timestamps) and `spikes.clusters` (cluster assignments). The cluster table provides anatomical labels but is not used for neural data directly.

ii.
```python
ssl = SpikeSortingLoader(pid=r.pid, one=one, eid=eid, pname=r.probe_name)
sp, cl, ch = ssl.load_spike_sorting()
```

iii. Spike times and cluster IDs are standard for computing binned neural activity.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over the trial window (-0.5 to 1.5 s around stimulus onset), giving spike counts per neuron per bin. The counts are NOT converted to firing rates (Hz); they remain as raw spike counts. When a session has multiple probes, their units are merged using `merge_probes` from the reference code. Clusters that fire at least once are kept; zero-firing clusters are dropped.

ii.
```python
binned = np.zeros((len(kidx), nneurons, NBINS), dtype=np.float32)
for j, k in enumerate(kidx):
    i0 = np.searchsorted(st, beg[k], side='left')
    i1 = np.searchsorted(st, end[k], side='left')
    b = ((st[i0:i1] - beg[k]) / BINSIZE).astype(np.int64)
    np.clip(b, 0, NBINS - 1, out=b)
    np.add.at(binned[j], (remap[sc[i0:i1]], b), 1.0)
```

Merging probes:
```python
spikes, clusters = merge_probes(spikes_list, clusters_list)
```

iii. The AI follows the zhang2025 reference code's binning approach. The AI chose not to convert to firing rate, keeping spike counts (trajectory step 65: metadata says `'neural_units': 'spike counts per 20 ms bin'`).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No quality control filtering is applied to clusters. All Kilosort-sorted clusters are kept, consistent with the zhang2025 reference code where `prepare_data` calls `load_spiking_data` with default `qc=None`. The only filtering is that clusters with zero spikes are implicitly dropped (they don't appear in `np.unique(sc)`).

ii.
```python
# clusters that fire at least once, as in get_spike_data_per_interval
used = np.unique(sc)
remap = np.full(len(clusters), -1, dtype=np.int64)
remap[used] = np.arange(len(used))
nneurons = len(used)
```

iii. The AI's docstring explicitly states: "we bin spike counts using all neurons, sorted by Kilosort 2.5, from each session" and `prepare_data` calls `load_spiking_data` with `qc=None`. This yields ~600k total neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are aligned to stimulus onset (`stimOn_times`). The trial window is defined as `beg = align + TIME_WINDOW[0]` and `end = align + TIME_WINDOW[1]` where `align = trials_df['stimOn_times']`. Spikes within each trial's window are binned relative to the window start.

ii.
```python
align = trials_df[ALIGN_TIME].to_numpy(dtype=float)
beg = align + TIME_WINDOW[0]
end = align + TIME_WINDOW[1]
...
b = ((st[i0:i1] - beg[k]) / BINSIZE).astype(np.int64)
```

iii. All streams share the same session clock, so alignment is a simple subtraction (trajectory step 56).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, producing 100 bins over the 2-second window. No rebinning or interpolation is applied to neural data. The time grid uses bin right edges: `BIN_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS)` = linspace(-0.48, 1.5, 100).

ii.
```python
BINSIZE = 0.02
NBINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
BIN_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS)
```

iii. The AI states this matches the reference code's `params` dict: "split into 2-s trials, each divided into 20-ms bins, producing T = 100 time steps" (trajectory step 65).

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `stimOn_times` in the trials table, which defines the alignment event. The time values are the right edges of the 20 ms bins spanning the decoding window.

ii.
```python
ALIGN_TIME = 'stimOn_times'
BIN_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS)
```

iii. The right-edge convention follows the zhang2025 reference code's `np.linspace(beg + binsize, end, nbins)` grid.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The input is the right edge of each 20 ms bin, computed as `np.linspace(-0.48, 1.5, 100)`. These are fixed values, the same for every trial.

ii.
```python
BIN_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS)
inputs[:, 0, :] = BIN_TIMES[None, :]
```

iii. No processing is needed; the values are defined by the bin grid.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The same bin grid is used for both neural binning and the time input. Spikes are binned into bins whose right edges match `BIN_TIMES`, so bin-for-bin the neural data and the time input describe the same time interval.

ii.
```python
b = ((st[i0:i1] - beg[k]) / BINSIZE).astype(np.int64)  # spike bin assignment
inputs[:, 0, :] = BIN_TIMES[None, :]                     # time input
```

iii. Both use the same 100-bin grid derived from the same window parameters.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. A change in `probabilityLeft` marks the start of a new block, and the trial number within the block is counted from zero.

ii.
```python
def trial_in_block(pleft):
    out = np.zeros(len(pleft), dtype=np.int64)
    count = 0
    for i in range(len(pleft)):
        if i > 0 and pleft[i] != pleft[i - 1]:
            count = 0
        out[i] = count
        count += 1
    return out
```

iii. The trials table has no block identifier, so blocks are inferred from changes in the prior probability.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The trial number within each block is counted from zero, starting over when `probabilityLeft` changes. The count is computed on the full trials table before filtering, so a dropped trial still advances the count (the animal's real position in the block is preserved). The value is broadcast to all time bins of the trial.

ii.
```python
tib = trial_in_block(trials_df['probabilityLeft'].to_numpy())[kidx]
inputs[:, 1, :] = tib[:, None]
```

iii. Computing from the full table before filtering preserves the animal's actual block position.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table. In the IBL convention, +1 is left and -1 is right. The AI recodes to 0 for left and 1 for right.

ii.
```python
choice = trials_df['choice'].to_numpy()[kidx]
choice_out = (choice < 0).astype(np.int64)  # left 0, right 1
```

iii. The AI confirmed the IBL convention by checking against `contrastLeft` on correct trials (trajectory step 60).

## 5-b. What processing is involved in computing `output` *Choice*?

i. Simple recoding: IBL choice +1 (left) becomes 0, and -1 (right) becomes 1. The value is broadcast to all time bins of the trial.

ii.
```python
choice_out = (choice < 0).astype(np.int64)
outputs[:, 0, :] = choice_out[:, None]
```

iii. The instructions specify "left = 0, right = 1".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From the `probabilityLeft` column of the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
pleft = trials_df['probabilityLeft'].to_numpy()[kidx]
prior_out = np.select([np.isclose(pleft, 0.2), np.isclose(pleft, 0.5),
                       np.isclose(pleft, 0.8)], [0, 1, 2], default=-1)
```

iii. The three values and their mapping are given directly in the instructions.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Recoding using `np.select` with `np.isclose` comparisons: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. An error is raised if any unexpected value is found. The value is broadcast to all time bins.

ii.
```python
prior_out = np.select([np.isclose(pleft, 0.2), np.isclose(pleft, 0.5),
                       np.isclose(pleft, 0.8)], [0, 1, 2], default=-1)
if np.any(prior_out < 0):
    raise RuntimeError('unexpected probabilityLeft value')
outputs[:, 1, :] = prior_out[:, None]
```

iii. Using `np.isclose` is slightly more robust against floating-point comparison issues than exact equality.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From the wheel position and timestamps loaded by `SessionLoader.load_wheel()`. The velocity is computed internally by `SessionLoader` (interpolation to 1000 Hz, Butterworth low-pass filter, differentiation). Speed is the absolute value of velocity.

ii.
```python
sess_loader.load_wheel()
wheel_speed, wheel_ok = interp_behavior(
    sess_loader.wheel['times'].to_numpy(),
    np.abs(sess_loader.wheel['velocity'].to_numpy()), beg, end)
```

iii. Same as the reference approach: `np.abs` of the velocity from `SessionLoader`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps: (1) `SessionLoader` internally interpolates wheel position to 1000 Hz and computes velocity via a 20 Hz Butterworth low-pass filter. (2) Speed (absolute velocity) is linearly interpolated onto the bin grid using `interp1d` with extrapolation. (3) The continuous values are discretized into 3 equal-occupancy bins using session-level tertile thresholds.

ii.
```python
# Interpolation onto bin grid
x = interval_begs[k] + BIN_TIMES - TIME_WINDOW[0]
values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
```

```python
# Discretization
def discretize(values, nbins=NBEHBINS):
    edges = np.nanquantile(values, np.arange(1, nbins) / nbins)
    edges = np.unique(edges)
    return np.digitize(values, edges).astype(np.int64)
```

iii. The AI's interpolation uses `scipy.interpolate.interp1d` with extrapolation rather than `np.interp`. The discretization uses `np.nanquantile` with `np.unique` to handle potential duplicate edges.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Equal-occupancy (tertile) thresholds are computed per session using `np.nanquantile` at the 1/3 and 2/3 quantiles. `np.digitize` maps values to bins 0, 1, 2 (low, medium, high).

ii.
```python
edges = np.nanquantile(values, np.arange(1, nbins) / nbins)  # [1/3, 2/3]
edges = np.unique(edges)
return np.digitize(values, edges).astype(np.int64)
```

iii. Session-level tertiles ensure equal class occupancy within each session, avoiding the problem that absolute thresholds would be meaningless across sessions with different wheel speed distributions.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated onto the same time grid as the neural data (right edges of the 20 ms bins relative to stimulus onset), so the two share a time axis bin for bin.

ii.
```python
x = interval_begs[k] + BIN_TIMES - TIME_WINDOW[0]  # = align[k] + BIN_TIMES
values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
```

iii. The wheel timestamps are on the same session clock as the spikes, so interpolating at the bin grid provides alignment.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From the camera ROI motion energy: `leftCamera.ROIMotionEnergy` (preferred) or `rightCamera.ROIMotionEnergy` (fallback), with corresponding frame times.

ii.
```python
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sess_loader.load_motion_energy(views=[view])
        df = sess_loader.motion_energy[cam]
        v, ok = interp_behavior(df['times'].to_numpy(),
                                df['whiskerMotionEnergy'].to_numpy(), beg, end)
    except Exception:
        continue
```

iii. Left camera is preferred, falling back to right, consistent with the reference approach.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion energy trace is used as-is (no filtering or normalization). It is interpolated onto the bin grid using `interp1d` with extrapolation, then discretized into 3 equal-occupancy bins using session-level tertiles, same as the wheel.

ii.
```python
v, ok = interp_behavior(df['times'].to_numpy(),
                        df['whiskerMotionEnergy'].to_numpy(), beg, end)
me_bin = discretize(me_vals[kidx])
```

iii. Same processing pipeline as wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: session-level equal-occupancy tertile thresholds via `np.nanquantile` at 1/3 and 2/3 quantiles.

ii.
```python
me_bin = discretize(me_vals[kidx])
```

iii. Session-level tertiles are necessary because motion energy units depend on camera, illumination, and ROI.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: interpolated onto the same bin grid (right edges) as the neural data.

ii.
```python
x = interval_begs[k] + BIN_TIMES - TIME_WINDOW[0]
values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
```

iii. Camera frame times are on the same session clock as the neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple layers of handling: (1) `load_trials_and_mask` drops trials with NaN in key columns. (2) `interp_behavior` marks trials as bad if the behavioral trace doesn't cover the window or contains NaN. (3) Sessions that raise exceptions or have fewer than 2 usable trials are skipped. (4) Probes with empty spike sorting are skipped. (5) Non-finite spike times/clusters are filtered.

ii.
```python
# Filter non-finite spikes
finite = np.isfinite(st) & np.isfinite(sc)
st, sc = st[finite], sc[finite].astype(np.int64)

# Behavior coverage check
if len(vv) == 0 or np.any(np.isnan(vv)):
    continue
if np.abs(interval_begs[k] - tt[0]) > BINSIZE:
    continue

# Session-level skip
except Exception as e:
    return eid, None, f'{type(e).__name__}: {e}'
```

iii. The AI builds in multiple safety checks, dropping problematic data at each level.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting from disk is the most expensive operation, as the spike arrays for a probe can be hundreds of megabytes. The AI also has a significant ONE cache rebuild step at startup.

ii.
```python
sp, cl, ch = ssl.load_spike_sorting()
```

iii. The cost is dominated by file I/O (trajectory step 67 shows ~124 sessions in 60 seconds).

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial spike binning loop and the per-trial behavior interpolation loop could potentially be vectorized. The spike binning uses `np.add.at` within a trial loop, and the behavior interpolation uses `interp1d` per trial.

ii.
```python
for j, k in enumerate(kidx):
    ...
    np.add.at(binned[j], (remap[sc[i0:i1]], b), 1.0)

for k in range(ntrials):
    ...
    values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
```

iii. Both loops process trials independently and could theoretically be vectorized, but each trial slices a different portion of the session, making the loop straightforward.

## 10-c. What processing does the code repeat multiple times?

i. The ONE client and BrainRegions are recreated for every session in the worker pool (`get_one()` and `BrainRegions()` are called inside `process_session`). The ONE cache rebuild at startup is also redundant if the cache already exists (guarded by a file check).

ii.
```python
def process_session(eid, sub_df):
    one = get_one()
    brainreg = BrainRegions()
```

iii. Creating ONE and BrainRegions per session adds overhead, though it's necessary for process isolation in the multiprocessing pool.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI saves intermediate per-session `.npz` files to `TMP_DIR` which are re-read during assembly. This adds I/O overhead. Also, the brain region mapping to Beryl is computed but the code does not filter out `void` or `root` regions, keeping neurons the atlas places outside the brain.

ii.
```python
f = TMP_DIR / f'{eid}.npz'
np.savez(f, neural=binned, input=inputs, output=outputs, regions=beryl)
```

iii. The intermediate files serve as a checkpoint but add unnecessary disk I/O. The Beryl-mapped regions include `void` entries that represent channels outside the brain.
