# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses the ONE API to load all scientific data, but identifies sessions and probes from the methods-paper freeze CSV (`/app/code/code_zhang2025/data/bwm_release.csv`) rather than dynamically searching the ONE cache. The freeze CSV supplies EID/PID/probe-name tuples that define the cohort. The ONE cache is initialized with `ONE(silent=True)`, the Brainwidemap release tables are loaded, and a metadata fix is applied (canonical trials-table `default_revision` flag set to True in memory). Data arrays are then loaded via `SessionLoader` and `SpikeSortingLoader` keyed by the freeze-file identifiers.

ii.
```python
FREEZE = Path('/app/code/code_zhang2025/data/bwm_release.csv')

def cohort(sample: bool) -> tuple[pd.DataFrame, list[str]]:
    f = pd.read_csv(FREEZE).drop(columns=['Unnamed: 0'], errors='ignore')
    eids = list(dict.fromkeys(f.eid.astype(str)))
    if sample:
        eids = eids[:2]
    return f[f.eid.astype(str).isin(eids)].copy(), eids

def init_one() -> ONE:
    one = ONE(silent=True)
    one.load_cache(tag='Brainwidemap', clobber=True)
    ds = one._cache['datasets']
    mask = ds.rel_path.str.endswith('_ibl_trials.table.pqt')
    ds.loc[mask, 'default_revision'] = True
    return one
```

iii. The AI chose to use the freeze CSV because it provides an exact, reproducible cohort matching the methods paper, with known EID/PID pairs. The ONE cache metadata fix was needed because the Brainwidemap release cache marks the canonical trials table as non-default, causing ONE to return only supplemental columns. The AI documented this in CONVERSION_NOTES Steps 1-2.

## 1-b. How are the data split into subjects?

i. Subject names are extracted from the ONE session cache table or from the freeze CSV metadata. Sessions are grouped by subject for the output dictionary.

ii.
```python
try:
    details = one._cache['sessions'].loc[uuid.UUID(str(eid))]
    subject, lab, date = str(details.subject), str(details.lab), str(details.date)
except (KeyError, ValueError):
    fr = freeze.iloc[0]
    subject, lab, date = str(fr.subject), str(fr.lab), str(fr.date)
```

Assembly:
```python
subjects=sorted({x['subject'] for x in infos})
subject_idx=np.array([subjects.index(x['subject']) for x in infos],dtype=np.int32)
```

iii. Subject identity comes from ONE session metadata, with a fallback to the freeze CSV.

## 1-c. How are the data split into sessions?

i. Each unique EID from the freeze CSV or ONE cache represents one session. Sessions are processed independently.

ii.
```python
freeze, eids = cohort(sample)
# eids is a list of unique session identifiers
```

iii. Sessions are the natural unit of the Brainwidemap release. No splitting is needed.

## 1-d. How are the data split into trials?

i. The trials table loaded via `SessionLoader.load_trials()` has one row per trial.

ii.
```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
tr = sl.trials
```

iii. No splitting needed; the trials table already has one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) all required event columns must be finite, (2) choice must be binary (-1 or 1), (3) probabilityLeft must be in {0.2, 0.5, 0.8}, (4) trial interval must be <= 10 seconds, (5) first movement latency must be between 0.08 and 2.0 seconds, and (6) wheel speed and whisker motion energy must be finite for the full trial window.

ii.
```python
required = ['choice','probabilityLeft','feedbackType','feedback_times','stimOn_times',
            'firstMovement_times','intervals_0','intervals_1']
vals = tr[required].to_numpy(float)
valid = np.all(np.isfinite(vals), axis=1)
valid &= np.isin(choice, [-1, 1]) & np.isin(prior, [0.2, 0.5, 0.8])
valid &= (tr.intervals_1.to_numpy() - tr.intervals_0.to_numpy() <= 10.0)
latency = tr.firstMovement_times.to_numpy() - stim
valid &= (latency >= 0.08) & (latency <= 2.0)
# ...
valid &= np.all(np.isfinite(speed), axis=1) & np.all(np.isfinite(whisk), axis=1)
```

iii. The AI combined trial QC criteria from both the data paper (RT bounds, binary choice) and the reference code's `load_trials_and_mask` (finite events, interval duration <= 10s). Behavior coverage is checked via NaN detection after interpolation. The AI documented this union of criteria in CONVERSION_NOTES Step 4.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times` (spike timestamps) and `spikes.clusters` (cluster assignment per spike), loaded via `SpikeSortingLoader`. Cluster metadata (label, acronym) is used for quality filtering.

ii.
```python
loader = SpikeSortingLoader(pid=str(row.pid), one=one)
spikes, clusters, channels = loader.load_spike_sorting()
clusters = loader.merge_clusters(spikes, clusters, channels)
```

iii. Spike times and cluster assignments are the standard source for binned neural activity.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over the -0.5 to 1.5 s trial window, giving 100 bins per trial. Counts are stored as uint16 integers, NOT divided by the bin width. When a session has multiple probes, units are merged into a single population with continuous indexing.

ii.
```python
def bin_probe(spike_times, spike_clusters, good_ids, stim):
    out = np.zeros((len(stim), len(good_ids), 100), dtype=np.uint16)
    # ...
    for ti, st in enumerate(stim):
        lo, hi = np.searchsorted(spike_times, (st + OFF_START, st + OFF_END))
        # ...
        bi = np.floor((ts[keep] - (st + OFF_START)) / BIN).astype(np.int32)
        flat = ui[keep][valid].astype(np.int64) * 100 + bi[valid]
        cnt = np.bincount(flat, minlength=len(good_ids) * 100).reshape(len(good_ids), 100)
        out[ti] = cnt.astype(np.uint16)
    return out
```

iii. The AI justified storing raw counts (rather than Hz firing rates) by citing the methods paper caching code, which also stores counts: "The reference cache bins counts; no smoothing, z-scoring, or firing-rate conversion is applied before saving." (CONVERSION_NOTES Step 5)

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` are kept. Additionally, clusters whose raw Allen atlas acronym is in the set `{'', 'void', 'root', 'nan', 'none'}` are excluded. No Beryl atlas mapping is applied.

ii.
```python
INVALID_REGIONS = {'', 'void', 'root', 'nan', 'none'}
# ...
ids, labels, acr = cluster_fields(clusters)
region_ok = np.array([x.strip().lower() not in INVALID_REGIONS for x in acr])
keep = (labels >= 1) & region_ok
```

iii. The AI used `label >= 1` matching the reference code's operational quality criterion. For region exclusion, the AI documented: "Use good units with valid non-root atlas acronym. Do not impose analysis-specific >=5/region or >=2-session threshold" (CONVERSION_NOTES Step 4). The AI chose to exclude `root` regions, justifying that they lack a valid anatomical assignment.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are aligned to stimulus onset (`stimOn_times`) by searching for spikes within the window [stimOn - 0.5, stimOn + 1.5] seconds and computing bin indices relative to the window start.

ii.
```python
for ti, st in enumerate(stim):
    lo, hi = np.searchsorted(spike_times, (st + OFF_START, st + OFF_END))
    ts = spike_times[lo:hi]
    bi = np.floor((ts[keep] - (st + OFF_START)) / BIN).astype(np.int32)
```

iii. All streams share the same session clock, so alignment is just relative time computation. This matches the decoder task requirement of stimulus-onset alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The trial window is -0.5 to 1.5 s, divided into 100 bins of 20 ms each. No rebinning or interpolation is applied to the neural data.

ii.
```python
BIN = 0.02
OFF_START, OFF_END = -0.5, 1.5
EDGES_REL = np.arange(OFF_START, OFF_END + BIN / 2, BIN, dtype=np.float64)
CENTERS_REL = ((EDGES_REL[:-1] + EDGES_REL[1:]) / 2).astype(np.float32)
assert EDGES_REL.size == 101 and CENTERS_REL.size == 100
```

iii. The 20 ms bin size and 2 s window match both the methods paper code and the data paper's decoding analysis.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Derived from the fixed bin centers of the trial window, computed from the window parameters (-0.5 to 1.5 s, 20 ms bins). Not from a raw data variable per se.

ii.
```python
CENTERS_REL = ((EDGES_REL[:-1] + EDGES_REL[1:]) / 2).astype(np.float32)
# ...
inp = np.vstack((CENTERS_REL, np.full(100, block_num[raw_i], np.float32)))
```

iii. Time since stimulus onset is defined by the binning grid itself.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. Bin centers are computed as midpoints of the 20 ms bin edges. The same vector is used for all trials.

ii.
```python
EDGES_REL = np.arange(OFF_START, OFF_END + BIN / 2, BIN, dtype=np.float64)
CENTERS_REL = ((EDGES_REL[:-1] + EDGES_REL[1:]) / 2).astype(np.float32)
```

iii. N/A - this is a deterministic grid computation.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The bin centers define both the time input and the neural binning grid. They are the same grid by construction.

ii.
```python
inp = np.vstack((CENTERS_REL, ...))  # same centers used for neural binning
```

iii. By construction, no separate alignment step is needed.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `trials.probabilityLeft` in the trials table. Block boundaries are detected where the prior value changes.

ii.
```python
prior = tr.probabilityLeft.to_numpy(float)
block_num = trial_number_in_block(prior)
```

iii. The trials table has no explicit block identifier, so blocks are inferred from contiguous runs of the same probabilityLeft value.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A loop iterates through all native trials (before filtering). The counter starts at 0 and increments by 1 for each consecutive trial with the same prior. It resets to 0 when the prior changes or is NaN.

ii.
```python
def trial_number_in_block(prior: np.ndarray) -> np.ndarray:
    out = np.zeros(len(prior), dtype=np.float32)
    for i in range(1, len(prior)):
        out[i] = out[i - 1] + 1 if np.isfinite(prior[i]) and prior[i] == prior[i - 1] else 0
    return out
```

iii. Computed before trial filtering so that dropped trials don't alter the block position. The AI noted: "Counter resets whenever probabilityLeft differs from preceding native trial. It is computed before trial filtering so omitted bad/no-go trials do not compress behavioral block position." (CONVERSION_NOTES Step 5)

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table, which takes values +1, -1, or 0 in IBL convention.

ii.
```python
choice = tr.choice.to_numpy(float)
valid &= np.isin(choice, [-1, 1])
```

iii. Choice is a standard trial variable in the IBL task.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps IBL choice -1 to class 0 and IBL choice +1 to class 1. No-go choices (0) are excluded. The AI's notes state "IBL -1 (left) -> 0; +1 (right) -> 1", but this misidentifies the IBL convention: in IBL, +1 = left and -1 = right. So the actual mapping is right -> 0, left -> 1, which is reversed from the instruction requirement of "left = 0, right = 1".

ii.
```python
out[0] = 0 if choice[raw_i] == -1 else 1
```

iii. The AI's CONVERSION_NOTES Step 5 states: "IBL `-1` (left) -> 0; `+1` (right) -> 1; repeat over 100 bins; exclude choice 0/NaN". However, this incorrectly identifies IBL -1 as left. The reference code and IBL documentation establish that +1 = left and -1 = right.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `trials.probabilityLeft`, which takes values 0.2, 0.5, and 0.8.

ii.
```python
prior = tr.probabilityLeft.to_numpy(float)
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
out[1] = prior_map[float(prior[raw_i])]
```

iii. The mapping follows the decoder task specification directly.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. A simple recoding: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. No further processing.

ii.
```python
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
out[1] = prior_map[float(prior[raw_i])]
```

iii. Directly follows the instruction specification.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.position` and `_ibl_wheel.timestamps`, loaded via `SessionLoader.load_wheel()`, which derives velocity internally. The speed is the absolute value of that velocity.

ii.
```python
sl.load_wheel()
w = sl.wheel
speed = interpolate_trials(w['times'].to_numpy(), np.abs(w['velocity'].to_numpy()), stim)
```

iii. The reference code derives wheel speed the same way, as the absolute value of the velocity from SessionLoader.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. SessionLoader interpolates the raw wheel position onto a 1000 Hz grid and applies a 20 Hz Butterworth low-pass filter to derive velocity. The speed (absolute velocity) is then linearly interpolated onto the 100 bin centers of each trial using a vectorized interpolation. Values outside the stream's temporal support become NaN, and trials with any NaN are excluded. Finally, the continuous speed trace is discretized into 3 classes.

ii.
```python
def interpolate_trials(times, values, stim, centers_rel=CENTERS_REL):
    q = np.asarray(stim)[:, None] + centers_rel[None, :]
    flat = np.interp(q.ravel(), times, values, left=np.nan, right=np.nan)
    return flat.reshape(q.shape).astype(np.float32)
```

iii. The interpolation is vectorized across all trials. The use of NaN for out-of-range values replaces the reference's explicit coverage check (`covered()` function).

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Session-level tertiles (1/3 and 2/3 quantiles) are computed from finite values of valid trials. `np.digitize` assigns each value to one of 3 classes (0=low, 1=medium, 2=high). If the two quantile edges are equal (degenerate case), the upper edge is nudged with `np.nextafter`.

ii.
```python
def quantile_classes(x: np.ndarray, valid_trials: np.ndarray):
    vals = x[valid_trials]
    vals = vals[np.isfinite(vals)]
    q = np.quantile(vals, [1/3, 2/3]).astype(float)
    if q[1] <= q[0]:
        q[1] = np.nextafter(q[0], np.inf)
    cls = np.digitize(x, q, right=False).astype(np.uint8)
    return cls, q.tolist()
```

iii. The tertile approach produces approximately equal-sized classes per session, matching the reference's use of percentile-based splitting.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated onto the same bin centers as the neural data, measured relative to the same stimulus onset time.

ii.
```python
q = np.asarray(stim)[:, None] + centers_rel[None, :]  # same centers as neural bins
```

iii. Using the shared time grid ensures alignment by construction.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `<side>Camera.ROIMotionEnergy` and `_ibl_<side>Camera.times`, where left camera is preferred over right.

ii.
```python
for side in ('left', 'right'):
    try:
        sl.load_motion_energy(views=[side])
        key = f'{side}Camera'
        me = sl.motion_energy[key]
        vals = me['whiskerMotionEnergy'].to_numpy()
        whisk = interpolate_trials(me['times'].to_numpy(), vals, stim)
        if np.isfinite(whisk).any():
            return speed, whisk, side, errors
    except Exception as ex:
        errors.append(...)
```

iii. Left camera is preferred, with right as fallback, matching the reference code's approach.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy trace is used as-is (no filtering or normalization). It is linearly interpolated onto the 100 bin centers of each trial, then discretized into 3 classes using session-level tertiles.

ii.
```python
whisk = interpolate_trials(me['times'].to_numpy(), vals, stim)
whisk_cls, whisk_q = quantile_classes(whisk, valid)
```

iii. Same processing pipeline as wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identical approach to wheel speed: session-level 1/3 and 2/3 quantiles computed from finite valid-trial values, applied via `np.digitize`.

ii.
```python
whisk_cls, whisk_q = quantile_classes(whisk, valid)
```

iii. Same tertile-based discretization as wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same approach as wheel: interpolated onto the shared bin center grid relative to stimulus onset.

ii.
```python
whisk = interpolate_trials(me['times'].to_numpy(), vals, stim)
```

iii. Alignment by construction through shared time grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple safeguards: (1) Trials with any NaN in required columns are excluded. (2) Trials where interpolated behavior produces NaN are excluded. (3) Probes that fail to load are logged and skipped; sessions continue with remaining probes. (4) Sessions with fewer than 2 valid trials or no qualified neurons raise a RuntimeError and are excluded. (5) The ONE cache metadata fix handles the trials-table revision anomaly.

ii.
```python
# Failed probe handling
except Exception as ex:
    failures.append({'pid': str(row.pid), ...})
    print(f'  WARNING probe {row.probe_name}/{row.pid} failed: ...')

# Session exclusion
if valid.sum() < 2:
    raise RuntimeError(f'only {valid.sum()} valid trials')
```

iii. The AI documented robust error handling in CONVERSION_NOTES Steps 5-6, including explicit logging of failures and exclusions.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data from disk, which involves reading large spike arrays (tens of millions of spikes per probe).

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
```

iii. File I/O for spike data dominates processing time. The AI noted sample conversion times of 4.5-7.4 seconds per session.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial spike binning loop in `bin_probe` iterates over each trial separately. This could potentially be vectorized by offsetting spike indices across trials. The behavior interpolation is already vectorized in the AI's code.

ii.
```python
for ti, st in enumerate(stim):
    lo, hi = np.searchsorted(spike_times, (st + OFF_START, st + OFF_END))
    # ... per-trial binning
```

iii. The AI vectorized behavior interpolation (unlike the reference which uses a per-trial loop), but kept the spike binning as a per-trial loop for clarity.

## 10-c. What processing does the code repeat multiple times?

i. No significant repeated processing was identified. Each data stream is loaded and processed once per session.

ii. N/A

iii. N/A

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Behavior is loaded and interpolated for ALL native trials (before trial filtering), even though only valid trials are retained. However, this is intentional: behavior validity (finite values) is used as a trial filter, and behavior loading is much cheaper than neural data loading. Neural data is loaded only for valid trials.

ii.
```python
speed, whisk, camera, behavior_errors = load_behavior(one, eid, stim)  # all trials
valid &= np.all(np.isfinite(speed), axis=1)  # filter based on behavior
neural_all, regions, probes, probe_failures = load_neural(one, freeze, stim[idx])  # only valid trials
```

iii. The AI intentionally loads behavior before neural data to avoid wasted spike processing on excluded trials. This is documented as a speedup in CONVERSION_NOTES Step 6.
