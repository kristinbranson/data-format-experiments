# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files from the `/app/data/sub-*/*.nwb` directory using `h5py` (not `pynwb`). It uses `glob.glob` to find all NWB files, then processes each one through `load_session()` which reads behavioral time series, neural fluorescence/neuropil traces, and metadata directly from the HDF5 structure.

ii.
```python
files = sorted(glob.glob('/app/data/sub-*/*.nwb'))
# ...
def load_session(fn):
    """Read everything needed from one NWB file (h5py, no pynwb needed)."""
    with h5py.File(fn, 'r') as f:
        beh = f['processing/behavior/BehavioralTimeSeries']
        g = lambda k: beh[k]['data'][()].astype(np.float64)
        d = dict(
            file=fn,
            subject=f['general/subject/subject_id'][()].decode(),
            session_id=f['general/session_id'][()].decode(),
            # ...
            pos=g('position'), speed=g('speed'), lick=g('lick'),
            # ...
        )
```

iii. The AI chose h5py over pynwb for performance. The CONVERSION_NOTES describe thorough data exploration confirming 152 NWB files across 11 subjects, matching the paper. The data structure was documented in Step 2 of the notes.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `general/subject/subject_id` field within each NWB file. Unique subjects are collected across all sessions.

ii.
```python
subjects = sorted({s['subject'] for s in sessions}, key=lambda x: int(x[1:]))
```

iii. The AI extracts the subject ID directly from the NWB metadata rather than from directory names. The result is the same 11 subjects as found in the sub-* directories.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session ID is extracted from `general/session_id` in the NWB file. All 152 sessions are processed.

ii.
```python
files = sorted(glob.glob('/app/data/sub-*/*.nwb'))
# Each file processed as one session
sess, stats = process_session(fn, show=show, plotdir=plotdir)
```

iii. Documented in CONVERSION_NOTES Step 2: "152 NWB files (87 GB), one file per (subject, session)."

## 1-d. How are the data split into trials?

i. Trials are defined by `trial_start` (frames where the flag is > 0) and `teleport` (frames where the flag is > 0). The trial window used for data extraction is `[start-1, stop-1)`, i.e., one frame before the trial_start flag to one frame before the teleport flag. This window is applied consistently to both neural and behavioral data.

ii.
```python
starts=np.where(g('trial_start') > 0)[0],
stops=np.where(g('teleport') > 0)[0],
# ...
for i, (s, e) in enumerate(zip(starts, stops)):
    sl = slice(s - 1, e - 1)
    # neural and behavior extracted from this slice
```

iii. The AI states this matches the reference code's `preprocessing.dff` window `[start-1, stop-1)`. However, the reference solution applies a +1 offset to the indices before passing them to dff (i.e., `pd.Series(trial_start_idx + 1)`), which makes the effective dff window `[trial_start_idx, trial_end_idx)`. The AI did NOT apply this +1 offset, so its trial window is shifted one frame earlier than the reference solution. The AI is internally consistent (neural and behavior use the same frames), but its first frame is one frame before the trial_start flag.

## 1-e. How are trials filtered based on quality controls?

i. Three filters are applied: (1) the first trial of each session is dropped because previous-trial outcome is undefined (152 trials); (2) trials with lick-sensor errors are dropped (81 trials, matching the paper's count, using the rule: >30% of frames with cumulative lick count > 2); (3) trials with non-finite deconvolved events are dropped.

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    if i == 0:
        continue                      # previous-trial outcome undefined
    if lick_error[i]:
        continue                      # invalid lick data (reference sets to NaN)
    # ...
    if not np.all(np.isfinite(ev)):
        continue
```

Lick error detection:
```python
LICK_ERROR_THRESH = 0.30
L = S['lick'][sl]
lick_error[i] = (np.sum(L > 2) / len(L)) > LICK_ERROR_THRESH
```

iii. The lick error filtering exactly reproduces the paper's 81 removed trials. Dropping the first trial is justified because `previous_trial_outcome` is a required decoder input and is undefined for the first lap. The CONVERSION_NOTES document this extensively in Steps 5 and 10.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw suite2p traces: `Fluorescence` (F) and `Neuropil` (Fneu) from the NWB ophys processing module. The stored `Deconvolved` series is NOT used, as it is suite2p's own deconvolution of raw fluorescence and does not match the paper's processing.

ii.
```python
Fl.append(f['processing/ophys/Fluorescence/plane%d/data' % p][:, keep].T)
Nl.append(f['processing/ophys/Neuropil/plane%d/data' % p][:, keep].T)
d['F'] = np.concatenate(Fl, axis=0).astype(np.float64)
d['Fneu'] = np.concatenate(Nl, axis=0).astype(np.float64)
```

iii. The CONVERSION_NOTES explain: "Neural signal used for decoding in the paper = deconvolved calcium events derived from dF/F (not raw suite2p spks). The NWB `Deconvolved` is suite2p's own deconvolution of raw F; the reference pipeline recomputes dF/F."

## 2-b. How is the `neural` data processed?

i. The AI reimplements the reference `preprocessing.dff` pipeline: (1) restrict F and Fneu to within-trial frames (NaN elsewhere); (2) neuropil subtraction: F - 0.7*Fneu; (3) add back per-trial neuropil mean; (4) per-trial maximin baseline: Gaussian smooth (sigma=15 frames), minimum filter (300 frames), maximum filter (300 frames); (5) dF/F = (F - baseline)/|baseline|; (6) Gaussian smooth dF/F (sigma=2 frames) with NaN-aware smoothing; (7) OASIS deconvolution (tau=0.7, fs=imaging_rate/n_planes, batch=2000). However, `keep_teleports` is always False (the reference conditionally sets it True for certain sessions based on `teleport_sessions` metadata).

ii.
```python
def compute_dff_and_events(F, Fneu, starts, stops, fs):
    """...keep_teleports=False..."""
    f_ = np.full(F.shape, np.nan)
    fneu_ = np.full(F.shape, np.nan)
    for start, stop in zip(starts, stops):
        f_[:, start - 1:stop - 1] = F[:, start - 1:stop - 1]
        fneu_[:, start - 1:stop - 1] = Fneu[:, start - 1:stop - 1]
    # neuropil correction
    f_ -= NEU_COEF * fneu_
    # ... per-trial maximin baseline, dF/F, smoothing, OASIS ...
```

iii. The AI's CONVERSION_NOTES document this as following `preprocessing.dff` exactly. However, the AI always uses `keep_teleports=False`, while the reference solution conditionally handles teleport sessions based on `teleport_metadata.py`. This means that for sessions where the laser was not blanked between trials, the AI's baseline window is restricted to within-trial samples when it should extend into the inter-trial interval.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three filters: (1) suite2p manual curation (`iscell[:,0] == 1`); (2) putative interneurons (Pearson r > 0.5 between dF/F and running speed); (3) cells with numerically unstable baseline (minimum within-trial maximin baseline < 5% of median raw fluorescence, 25 cells total).

ii.
```python
iscell = ps['iscell'][()][:, 0].astype(bool)
# ...
speed_corr = (Xc @ spc) / np.where(denom == 0, np.nan, denom)
is_int = np.nan_to_num(speed_corr, nan=0.0) > INTERNEURON_R_THRESH
keep_cells = (~is_int) & (~bad_baseline)
```

iii. The iscell and interneuron filters match the reference paper. The bad_baseline filter is an addition not in the reference code, justified by the AI as preventing numerically degenerate dF/F values from dominating the decoder's PCA.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data is aligned to trial start by extracting from the `[start-1, stop-1)` window (one frame before the trial_start flag). The time_from_trial_start input is computed relative to the trial_start frame (not the window start), so the first timepoint has a small negative value (~-0.065 s).

ii.
```python
sl = slice(s - 1, e - 1)
ev = events[np.ix_(keep_cells, np.arange(s - 1, e - 1))]
tt = S['t'][sl] - S['t'][s]       # time from trial start (s)
```

iii. The AI's window starts one frame before trial_start. The reference solution uses the window `[trial_start_idx, trial_end_idx)` (starts at the trial_start frame exactly).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data is kept at the native imaging frame rate (~15.5 Hz, 64.48 ms per frame). No rebinning is applied.

ii.
```python
fs = S['imaging_rate'] / S['n_planes']
dt = float(np.median(np.diff(S['t'])))
# ...
time_bin_size=float(np.mean([s['dt'] for s in sessions]) * 1000.0),
```

iii. The paper states "All behavioral and neural time series were sampled at ~15.5 Hz, the imaging frame rate." The AI preserves this native rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the behavioral timestamps array (`beh['position']['timestamps']`).

ii.
```python
t=beh['position']['timestamps'][()].astype(np.float64),
# ...
tt = S['t'][sl] - S['t'][s]       # time from trial start (s)
x[0] = tt
```

iii. All behavioral time series share the same timestamps. The AI chose position timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the timestamp at the trial_start frame index (`S['t'][s]`) is subtracted from the timestamps of each frame in the window. Because the window starts at frame `s-1` (one frame before trial start), the first value is negative (~-0.065 s).

ii.
```python
tt = S['t'][sl] - S['t'][s]       # time from trial start (s)
```

iii. The AI explicitly subtracts the timestamp at the trial_start frame (not at the window start frame), so the time is relative to the trial_start event, with the first sample being one frame before it.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data are extracted from the same frame indices (`sl = slice(s-1, e-1)`), so they share the same time axis. No additional alignment is needed.

ii.
```python
sl = slice(s - 1, e - 1)
ev = events[np.ix_(keep_cells, np.arange(s - 1, e - 1))]
tt = S['t'][sl] - S['t'][s]
```

iii. Both streams use the same frame clock, confirmed in the CONVERSION_NOTES.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
env=g('environment'),
# ...
env_trial[i] = int(np.round(np.median(S['env'][sl])))
```

iii. The environment variable is 0 for ENV1 and 1 for ENV2, matching the paper.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The per-trial environment is computed as the rounded median of the `environment` values within the trial window. Since environment is constant within a trial, this effectively just picks the value.

ii.
```python
env_trial[i] = int(np.round(np.median(S['env'][sl])))
```

iii. Using the median is a robust way to handle any edge effects at trial boundaries. In practice, environment is constant within a trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the loop index `i` in the trial iteration (the within-session lap index from the `starts`/`stops` arrays).

ii.
```python
for i, (s, e) in enumerate(zip(starts, stops)):
    if i == 0:
        continue
    # ...
    x[2] = i                           # lap index within the session
```

iii. The AI preserves the original lap index within the session. Since the first trial (i=0) is always dropped, trial numbers start at 1.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the loop index. The value is constant across all timepoints within a trial. Trial numbers start at 1 (not 0) because the first trial is dropped.

ii.
```python
x[2] = i
```

iii. The AI notes this preserves the original lap index so "the switch at lap 30 stays interpretable."

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` sparse time series timestamps and the `reward_zone` behavior time series. A trial is considered rewarded if a reward was delivered AND the reward_zone flag was active during the trial.

ii.
```python
reward_frames = np.searchsorted(S['t'], S['reward_t'])
# ...
got_reward = np.any((reward_frames >= s - 1) & (reward_frames < e - 1))
isreward[i] = int(got_reward and np.any(rzflag))
```

iii. This matches the reference paper's `behavior.get_trial_types` rule: a trial is rewarded only if both a reward was delivered AND the reward zone was active.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial i (starting from i=1, since i=0 is dropped), the previous trial outcome is `isreward[i-1]`. The first trial of each session is always dropped because the previous outcome is undefined.

ii.
```python
x[3] = isreward[i - 1]             # previous trial outcome
```

iii. Dropping the first trial avoids the ambiguity of what to assign when there is no previous trial. The reference solution instead includes the first trial with previous_reward=0.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the per-trial reward zone coordinates obtained from the VR scene name (extracted from the NWB `identifier` field). The reward zone boundaries are: A=[80,130], B=[200,250], C=[320,370] cm. On switch sessions, the zone changes after 30 trials.

ii.
```python
def scene_reward_zones(scene, n_trials):
    s = scene
    if '_to_' not in s:
        label = s[-1]
        labels = [label] * n_trials
    else:
        pre, post = s.split('_to_')
        first = pre[-1]
        second = post[-1]
        labels = [first] * min(CHANGE_TRIAL, n_trials)
        if n_trials > CHANGE_TRIAL:
            labels += [second] * (n_trials - CHANGE_TRIAL)
    coords = np.array([REWARD_ZONE_DICT[l] for l in labels], dtype=float)
    return coords, np.array(labels)
```

iii. The AI derives reward zones from the scene name, which is the approach used by the reference paper's `behavior.get_reward_zones`. The reference solution instead uses a Viterbi algorithm on observed reward_zone positions. Both approaches produce equivalent zone labels (verified by the AI against data-derived zone entry positions, with median error 0.78 cm).

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the signed distance from the animal's position to the nearest edge of the reward zone is computed. Distance is 0 inside the zone, negative before the zone, and positive after.

ii.
```python
def reward_zone_distance(pos, rz_start, rz_end):
    d = np.zeros_like(pos)
    before = pos < rz_start
    after = pos > rz_end
    d[before] = pos[before] - rz_start
    d[after] = pos[after] - rz_end
    return d
```

iii. This is the distance to the nearest boundary of the zone, with 0 throughout the 50 cm zone (the task asks for distance to "any location in the reward zone").

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit boolean conditions: (0) d < -50, (1) -50 <= d < -10, (2) -10 <= d < 0, (3) d == 0 (in zone), (4) 0 < d <= 10, (5) 10 < d <= 50, (6) d > 50.

ii.
```python
def digitize_rz_distance(d):
    out = np.full(d.shape, 3, dtype=np.int64)   # exactly 0 -> in the zone
    out[d < -50.0] = 0
    out[(d >= -50.0) & (d < -10.0)] = 1
    out[(d >= -10.0) & (d < 0.0)] = 2
    out[(d > 0.0) & (d <= 10.0)] = 4
    out[(d > 10.0) & (d <= 50.0)] = 5
    out[d > 50.0] = 6
    return out
```

iii. This custom function handles the "0 cm" bin (bin 3 = in the zone) explicitly. The reference solution uses `np.digitize` with a `1e-6` epsilon to separate exactly-zero from slightly-positive. The AI's approach uses `<=` on the positive side (e.g., bin 4 is `(0, 10]`) while the reference uses `<` (bin 4 is `[1e-6, 10)`), creating minor differences at exact bin boundaries (e.g., d=10 goes to bin 4 in the AI vs bin 5 in the reference).

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position data and neural data share the same frame indices (`sl = slice(s-1, e-1)`), so distance to reward zone is automatically aligned.

ii.
```python
pos = S['pos'][sl]
d_rz = reward_zone_distance(pos, rz_coords[i, 0], rz_coords[i, 1])
```

iii. Both are extracted from the same frame window.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
pos=g('position'),
# ...
pos = S['pos'][sl]
```

iii. Position directly records the animal's location in the VR corridor in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice and discretizing.

ii.
```python
y[1] = np.digitize(pos, POSITION_EDGES)
```
with `POSITION_EDGES = [90.0, 180.0, 270.0, 360.0]`.

iii. The raw position values are used directly. The 450 cm track is divided into 5 equal bins of 90 cm each.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins using `np.digitize` with edges [90, 180, 270, 360]: (0) < 90 cm, (1) 90-180, (2) 180-270, (3) 270-360, (4) > 360.

ii.
```python
POSITION_EDGES = [90.0, 180.0, 270.0, 360.0]
y[1] = np.digitize(pos, POSITION_EDGES)
```

iii. This is functionally equivalent to the reference's `np.digitize(pos, [-inf,90,180,270,360,inf]) - 1`, producing the same 0-4 bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices as neural data.

ii.
```python
pos = S['pos'][sl]
```

iii. Same frame window ensures alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series (cumulative lick count per frame).

ii.
```python
lick=g('lick'),
# ...
lick = S['lick'][sl]
```

iii. The lick variable records cumulative lick counts per imaging frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: cumulative count >= 1 maps to 1, otherwise 0.

ii.
```python
y[3] = (lick >= 1).astype(np.int64)
```

iii. This is equivalent to the reference's `(licks > 0).astype(int)` for integer-valued cumulative counts. Additionally, trials with lick-sensor errors (>30% of frames with cumulative count > 2) are dropped entirely.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices as neural data.

ii.
```python
lick = S['lick'][sl]
```

iii. Same frame window.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the VR scene name in the NWB `identifier` field, using `scene_reward_zones()` which parses scene names like `Env1_LocationA_to_C` to determine zone labels (A, B, or C) per trial, with a switch after 30 trials.

ii.
```python
scene=f['identifier'][()].decode().split('/')[-1],
# ...
rz_coords, rz_labels = scene_reward_zones(S['scene'], n_trials)
y[4] = RZ_LABELS.index(rz_labels[i])
```

iii. This matches the reference paper's `behavior.get_reward_zones` approach. The reference solution instead uses a Viterbi algorithm on observed reward-zone positions.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene name parsing determines the zone label per trial. For switch sessions (`_to_` in scene name), the first 30 trials get the first zone and remaining trials get the second zone. The label is mapped to 0=A, 1=B, 2=C.

ii.
```python
def scene_reward_zones(scene, n_trials):
    if '_to_' not in s:
        label = s[-1]
        labels = [label] * n_trials
    else:
        pre, post = s.split('_to_')
        first = pre[-1]
        second = post[-1]
        labels = [first] * min(CHANGE_TRIAL, n_trials)
        if n_trials > CHANGE_TRIAL:
            labels += [second] * (n_trials - CHANGE_TRIAL)
```

iii. The zone is verified against data-derived onset positions in the CONVERSION_NOTES (median error 0.78 cm).

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the sparse `Reward` time series timestamps AND the `reward_zone` behavior flag. A trial is rewarded if a reward was delivered within the trial window AND the reward_zone flag was active.

ii.
```python
reward_frames = np.searchsorted(S['t'], S['reward_t'])
# ...
got_reward = np.any((reward_frames >= s - 1) & (reward_frames < e - 1))
isreward[i] = int(got_reward and np.any(rzflag))
```

iii. This matches `behavior.get_trial_types` from the reference code. The reference solution only checks if any reward timestamp falls within the trial (without requiring the reward_zone flag).

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are mapped to frame indices using `searchsorted`. For each trial, the output is 1 if a reward event occurred within the trial window AND the reward zone was active, else 0. The value is constant across all timepoints.

ii.
```python
isreward[i] = int(got_reward and np.any(rzflag))
# ...
y[5] = isreward[i]
```

iii. The AND condition with rzone matches the paper's definition. The overall reward rate is 84.7%, matching the paper.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- **Neural/behavior frame-count mismatch**: In 10 sessions (m17/m18, 2-plane mice), the ophys stream has one frame more than the behavioral stream. Both are truncated to the common length.
- **Numerically unstable dF/F**: 25 cells whose maximin baseline collapses near zero (neuropil larger than ROI trace) are excluded.
- **Lick-sensor errors**: 81 trials with >30% of frames having cumulative lick count > 2 are dropped.
- **First trial**: Dropped per session (previous outcome undefined).
- **Non-finite events**: Trials with NaN/Inf in deconvolved events are dropped.
- **Sessions with < 2 trials**: Would be dropped (none occurred).

ii.
```python
if d['F'].shape[1] != n_beh:
    n = min(d['F'].shape[1], n_beh)
    d['F'] = d['F'][:, :n]
    # ...
# bad_baseline exclusion
bad_baseline = ~(min_base > BASELINE_MIN_FRAC * scale)
keep_cells = (~is_int) & (~bad_baseline)
```

iii. These are documented in CONVERSION_NOTES Steps 9-10. The frame-count mismatch and bad-baseline issues were discovered during the first full conversion run and fixed.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) loading NWB files from disk (I/O bound, ~1-2 s per session); (2) dF/F + OASIS deconvolution (CPU-bound, ~1-5 s per session depending on cell count); (3) the full conversion running 152 sessions took ~199 s wall clock with 12 parallel workers.

ii. N/A

iii. Timing information is documented in CONVERSION_NOTES Step 7.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_session` (iterating over starts/stops to build per-trial arrays) processes trials sequentially. The Viterbi-like sanity check loop (checking zone entry positions per trial) also iterates sequentially. The interneuron detection was already vectorized (matrix-vector product instead of per-cell loop).

ii. N/A

iii. The AI notes that variable trial lengths make vectorization awkward, requiring padding or masking.

## 13-c. What processing does the code repeat multiple times?

i. The code processes each NWB file only once (unlike the reference solution which has separate survey and conversion passes). However, within `process_session`, the trial-level loop computes reward zone coordinates for each trial even though they could be pre-computed for the session.

ii. N/A

iii. The AI used parallel processing (multiprocessing with 8 workers) to compensate for per-session overhead.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes per-session statistics (`stats` dict) that are saved to a CSV file but not used by the decoder. Processing plots (`--show-processing`) generate figures that are not used downstream. The `rz_onset_pos` sanity check data is computed for every trial but only used for a warning message.

ii.
```python
stats = dict(
    file=os.path.basename(fn), subject=S['subject'], session=S['session_id'],
    # ... many fields used only for diagnostics
)
# ...
pd.DataFrame(stats).to_csv(os.path.splitext(args.outfile)[0] + '_session_stats.csv', index=False)
```

iii. These are diagnostic outputs that aid in validation but are not part of the decoder pipeline.
