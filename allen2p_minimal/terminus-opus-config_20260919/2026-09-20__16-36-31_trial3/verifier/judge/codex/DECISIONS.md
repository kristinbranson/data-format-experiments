# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the Allen project cache. It reads the local metadata CSV, keeps only experiment IDs that have a local NWB file, filters those experiments to active familiar sessions, and then loads each experiment with `BehaviorOphysExperiment.from_nwb_path`.

ii.
```python
def get_experiment_table():
    tab = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
    local = [int(os.path.basename(f).split('_')[-1].split('.')[0])
             for f in glob.glob(os.path.join(EXP_DIR, '*.nwb'))]
    tab = tab[tab.ophys_experiment_id.isin(local)]
    tab = tab[(~tab.passive) & (tab.experience_level == 'Familiar')]
    return tab.sort_values(['ophys_session_id', 'ophys_experiment_id'])
...
for _, row in exp_rows.iterrows():
    path = os.path.join(
        EXP_DIR, f'behavior_ophys_experiment_{row.ophys_experiment_id}.nwb')
    exps.append((row, BehaviorOphysExperiment.from_nwb_path(path)))
```

iii. In the trajectory, the agent says it wants to match the Vip-Sst paper by restricting to active familiar-image sessions, and its design notes say the data are loaded from local NWB files with `BehaviorOphysExperiment` rather than through the project cache.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are unique `mouse_id` values from the filtered experiment table. While assembling output, the AI appends a mouse ID to `subjects` the first time it sees it.

ii.
```python
print(f'{len(tab)} experiments, {tab.ophys_session_id.nunique()} sessions, '
      f'{tab.mouse_id.nunique()} mice')
...
if res['mouse'] not in data['subjects']:
    data['subjects'].append(res['mouse'])
data['subject_idx'].append(data['subjects'].index(res['mouse']))
```

iii. The trajectory shows the agent inspected the metadata table and treated `mouse_id` as the animal identifier. There is no separate alternative subject definition discussed.

## 1-c. How are the data split into sessions?

i. Sessions are unique `ophys_session_id` groups. All experiments within one `ophys_session_id` are treated as a single session, and the simultaneously recorded planes are merged.

ii.
```python
return tab.sort_values(['ophys_session_id', 'ophys_experiment_id'])
...
sessions = list(tab.groupby('ophys_session_id'))
...
 - A 'session' is one ophys session (ophys_session_id).  For the multi-plane
   (Mesoscope) sessions the simultaneously recorded imaging planes (experiments)
   are concatenated into a single population, since they are one recording.
```

iii. The trajectory and the script docstring explicitly justify this by saying a mesoscope session is one recording even if it has multiple experiments/planes.

## 1-d. How are the data split into trials?

i. Trials come from `ex0.trials`, but instead of using each trial’s full `start_time` to `stop_time` span, the AI builds fixed-length trial windows from `-2.0` to `+4.0` seconds around each trial’s `change_time` (or sham change time on catch trials).

ii.
```python
OFF_START = -2.0
OFF_END = 4.0
...
trials = ex0.trials
sel = trials[(trials.go | trials.catch) & trials.change_time.notna()]
change_times = sel.change_time.values.astype(float)
...
for k, ct in enumerate(change_times):
    mat[:, k, :] = bin_events_matrix(traces, tss, ct + edges_rel)
```

iii. The trajectory and docstring say the agent aligned trials to the “(sham) change time” and chose a `[-2, +4]` second window because it “fits inside every trial.”

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials marked `go` or `catch` and with non-null `change_time`. Sessions with fewer than two such trials are dropped. After outcome assignment, trials with no recognized outcome label are discarded. If a session lacks usable pupil data, the entire session is skipped.

ii.
```python
sel = trials[(trials.go | trials.catch) & trials.change_time.notna()]
if len(sel) < 2:
    return None
...
if pupil_v is None:
    return None
...
keep = [k for k in range(ntrials) if out_trials[k] >= 0]
if len(keep) < 2:
    return None
```

iii. The trajectory repeatedly cites the instruction to include go and catch trials and exclude aborted and auto-rewarded trials. It also says pupil diameter is a required decoder output, which is why sessions without usable pupil data are dropped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural signal is derived from the AllenSDK `events` table, specifically the `filtered_events` column.

ii.
```python
EVENTS_COL = 'filtered_events'
...
ev = ex.events
traces = np.stack([np.asarray(e, dtype=float) for e in ev[EVENTS_COL].values])
```

iii. The trajectory says the agent switched from raw `events` to `filtered_events` after subset experiments showed better decoder accuracy, and because the SDK/tutorials describe `filtered_events` as the smoothed version of detected events.

## 2-b. How is the `neural` data processed?

i. For each plane, the AI sums `filtered_events` into 100 ms bins in each trial window, concatenates planes across neurons, and then divides each neuron by its standard deviation across all extracted bins in that session.

ii.
```python
BIN_SIZE = 0.1
...
for k, ct in enumerate(change_times):
    mat[:, k, :] = bin_events_matrix(traces, tss, ct + edges_rel)
...
neural_all = np.concatenate(neural_planes, axis=0)
sd = neural_all.reshape(neural_all.shape[0], -1).std(axis=1)
sd[sd == 0] = 1.0
neural_all = (neural_all / sd[:, None, None]).astype(np.float32)
```

iii. The trajectory says this choice was driven by decoder performance: `filtered_events` improved decodability, and per-neuron SD normalization was added because otherwise a few high-variance cells would dominate the decoder’s PCA projection.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit neuron-quality filter. The only neural-specific safeguard is truncating each plane’s event trace and timestamps to a common minimum length if they disagree.

ii.
```python
ts = np.asarray(ex.ophys_timestamps, dtype=float)
ev = ex.events
traces = np.stack([np.asarray(e, dtype=float) for e in ev[EVENTS_COL].values])
n = min(traces.shape[1], len(ts))
traces, tss = traces[:, :n], ts[:n]
```

iii. The trajectory does not mention any additional cell QC. It only adds this length-mismatch guard as a defensive data-integrity step.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to the trial’s `change_time` or sham change time. The neural matrix for a trial spans bins from `-2.0` to `+4.0` seconds relative to that event.

ii.
```python
OFF_START = -2.0
OFF_END = 4.0
...
edges_rel, centers_rel = bin_edges()
...
for k, ct in enumerate(change_times):
    mat[:, k, :] = bin_events_matrix(traces, tss, ct + edges_rel)
```

iii. The trajectory explicitly says the agent aligned to “the (sham) change time” and used a fixed window that fits within every trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 100 ms bins, so the AI rebins the native traces instead of keeping native ophys-frame resolution.

ii.
```python
BIN_SIZE = 0.1
...
'time_bin_size': BIN_SIZE * 1000.0,
```

iii. The trajectory says 100 ms bins were chosen as part of the fixed `[-2, +4]` second change-aligned representation, and the neural design note says events are “summed into 100 ms bins.”

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations`, using `image_name` from non-omitted flashes in the `change_detection` stimulus block. The AI uses a hard-coded global image-name list for familiar image set A.

ii.
```python
image_names = ['im061', 'im062', 'im063', 'im065',
               'im066', 'im069', 'im077', 'im085']
...
sp = ex0.stimulus_presentations
sp = sp[sp.stimulus_block_name.str.contains('change_detection', na=False)]
flash = sp[~sp.omitted.astype(bool)]
flash_start = flash.start_time.values.astype(float)
flash_img = np.array([image_names.index(n) for n in flash.image_name.values])
```

iii. The trajectory says the agent restricted to familiar active sessions so that all locally available sessions share familiar image set A, letting it use one common image label set across sessions.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each 100 ms bin center, the AI finds the most recent non-omitted flash at or before that time and assigns its image code. That image label is held through the gray interval and omissions.

ii.
```python
tc = ct + centers_rel
...
j = np.searchsorted(flash_start, tc, side='right') - 1
j = np.clip(j, 0, len(flash_start) - 1)
img_trials.append(flash_img[j].astype(np.int64))
```

iii. The comment in the code states the intended justification: image identity should reflect “the ongoing 750 ms image presentation interval” and be held through gray screens and omissions. The trajectory ties this to matching the familiar flashed-image task structure.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is sampled at the same per-trial 100 ms bin centers used for the neural data, all relative to each trial’s change time.

ii.
```python
te = ct + edges_rel
tc = ct + centers_rel
...
img_trials.append(flash_img[j].astype(np.int64))
...
outputs.append(np.stack([
    img_trials[k],
    chg_trials[k],
    run_q[n_],
    pup_q[n_],
    np.full(T, out_trials[k], dtype=np.int64),
]).astype(np.int64))
```

iii. The trajectory says the whole representation is built in a common fixed binning scheme around change time, so image identity and neural data share the same time base.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations.is_change` and the corresponding `start_time` values of change flashes, together with each trial’s change-aligned bin centers.

ii.
```python
chg = sp[sp.is_change.astype(bool)]
change_starts = chg.start_time.values.astype(float)
...
i = np.searchsorted(change_starts, tc, side='right') - 1
```

iii. The trajectory says the AI wanted a label for whether “an actual image change just occurred,” so it used actual change events from the stimulus table rather than only trial-table outcome fields.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each bin center, the AI finds the most recent actual image change in the session and marks the bin as `1` if it falls within 750 ms of that change; otherwise `0`.

ii.
```python
FLASH_INTERVAL = 0.75
...
is_chg = np.zeros(T, dtype=np.int64)
valid = i >= 0
is_chg[valid] = (tc[valid] - change_starts[i[valid]] < FLASH_INTERVAL).astype(np.int64)
chg_trials.append(is_chg)
```

iii. The code comment gives the justification: the variable should be `1` “during the 750 ms interval starting at an actual image change,” with catch/sham trials containing no change.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is binary. Bins are assigned `0` for `no_change` and `1` for `change` depending on whether they fall inside the 750 ms post-change interval.

ii.
```python
'output_values': [image_names,
                  ['no_change', 'change'],
                  ...
]
...
is_chg = np.zeros(T, dtype=np.int64)
is_chg[valid] = (tc[valid] - change_starts[i[valid]] < FLASH_INTERVAL).astype(np.int64)
```

iii. The trajectory treats image change as a categorical decoder target and keeps it as the natural binary event label.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image-change labels are computed on the same 100 ms, change-centered trial bins as the neural data.

ii.
```python
te = ct + edges_rel
tc = ct + centers_rel
...
for k, ct in enumerate(change_times):
    mat[:, k, :] = bin_events_matrix(traces, tss, ct + edges_rel)
...
chg_trials.append(is_chg)
```

iii. The trajectory says all outputs are aligned to the fixed `[-2, +4]` second window around each trial’s change or sham-change time.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from the AllenSDK `running_speed` table, specifically its `timestamps` and `speed` columns.

ii.
```python
run = ex0.running_speed
run_t = run.timestamps.values.astype(float)
run_v = run.speed.values.astype(float)
```

iii. The trajectory does not argue for an alternative source; it uses the standard running-speed stream provided by the dataset.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI averages running speed within each 100 ms trial bin, linearly fills missing bin values, and then discretizes each kept session’s trial values into five quantile bins.

ii.
```python
def bin_mean(values, timestamps, t_edges):
    idx = np.searchsorted(timestamps, t_edges)
    ...
    out[good] = sums[good] / counts[good]
    return out
...
r = fill_nans(bin_mean(run_v, run_t, te))
...
run_q = quantize([run_trials[k] for k in keep])
```

iii. The trajectory treats this as part of the fixed-bin decoder representation. It does not give a separate scientific justification beyond using categorical outputs and maximizing decoder performance.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into five equal-quantile bins computed within each session from the kept trials for that session.

ii.
```python
def quantize(values, nq=NQUANTILES):
    v = np.concatenate([np.ravel(a) for a in values])
    edges = np.quantile(v, np.linspace(0, 1, nq + 1)[1:-1])
    return [np.searchsorted(edges, np.ravel(a), side='right').astype(np.int64)
            for a in values]
...
run_q = quantize([run_trials[k] for k in keep])
```

iii. The trajectory consistently describes running speed as “running-speed quintiles,” so the intended justification is to create five equal-percentile categories required by the decoder task.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is reduced into the same 100 ms trial bins used for neural activity, with each bin defined relative to trial change time.

ii.
```python
te = ct + edges_rel
...
r = fill_nans(bin_mean(run_v, run_t, te))
...
outputs.append(np.stack([
    img_trials[k],
    chg_trials[k],
    run_q[n_],
    pup_q[n_],
    np.full(T, out_trials[k], dtype=np.int64),
]).astype(np.int64))
```

iii. The trajectory’s fixed-bin representation makes running speed share the same per-trial temporal grid as the neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `eye_tracking.pupil_area` and `eye_tracking.likely_blink`, not from `pupil_width`.

ii.
```python
eye = ex0.eye_tracking
...
area = eye.pupil_area.values.astype(float)
blink = eye.likely_blink.values.astype(bool)
area = np.where(blink, np.nan, area)
```

iii. The trajectory says pupil diameter is a required output, and the code’s chosen justification is to compute a geometric diameter from measured pupil area after excluding likely blinks.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink-marked samples are set to `NaN`, pupil area is converted to diameter as `2*sqrt(area/pi)`, missing values are linearly filled, values are averaged in each 100 ms trial bin, any remaining binned NaNs are filled again, and the result is quantized into five bins.

ii.
```python
area = np.where(blink, np.nan, area)
diam = 2.0 * np.sqrt(area / np.pi)
diam = fill_nans(diam)
...
p = fill_nans(bin_mean(pupil_v, pupil_t, te))
...
pup_q = quantize([pup_trials[k] for k in keep])
```

iii. The trajectory does not separately defend this pipeline beyond needing a usable categorical pupil-diameter output and skipping sessions where the pupil stream is unusable.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five equal-quantile bins computed within each session from the kept trials of that session.

ii.
```python
NQUANTILES = 5
...
pup_q = quantize([pup_trials[k] for k in keep])
...
'output_values': [image_names,
                  ['no_change', 'change'],
                  ...,
                  ['Q1_0-20%', 'Q2_20-40%', 'Q3_40-60%',
                   'Q4_60-80%', 'Q5_80-100%'],
                  ...]
```

iii. As with running speed, the trajectory frames this as producing the required five percentile-like pupil categories for decoding.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is binned onto the same 100 ms trial grid as the neural data, with bin edges defined relative to each trial’s change time.

ii.
```python
te = ct + edges_rel
...
p = fill_nans(bin_mean(pupil_v, pupil_t, te))
...
outputs.append(np.stack([
    img_trials[k],
    chg_trials[k],
    run_q[n_],
    pup_q[n_],
    np.full(T, out_trials[k], dtype=np.int64),
]).astype(np.int64))
```

iii. The trajectory treats pupil diameter as one more time-varying output on the same fixed change-aligned binning scheme.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
tr = sel.iloc[k]
if tr.hit:
    o = 0
elif tr.miss:
    o = 1
elif tr.false_alarm:
    o = 2
elif tr.correct_reject:
    o = 3
else:
    o = -1
```

iii. The trajectory describes trial outcome as one of the standard Visual Behavior outcomes and keeps it constant within each trial.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps the four outcome booleans to integers `0` to `3` and then repeats the chosen code across every time bin in the trial.

ii.
```python
if tr.hit:
    o = 0
elif tr.miss:
    o = 1
elif tr.false_alarm:
    o = 2
elif tr.correct_reject:
    o = 3
...
np.full(T, out_trials[k], dtype=np.int64)
```

iii. The trajectory says trial outcome is a static per-trial decoder target, so it is broadcast across time bins rather than stored once per trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI uses several defensive fallbacks: it truncates event traces and timestamps to a shared minimum length, linearly fills missing running and pupil values, returns `None` if a signal is completely missing, skips sessions on exceptions, and skips sessions with fewer than two usable trials.

ii.
```python
if not np.any(good):
    return None
...
n = min(traces.shape[1], len(ts))
traces, tss = traces[:, :n], ts[:n]
...
r = fill_nans(bin_mean(run_v, run_t, te))
p = fill_nans(bin_mean(pupil_v, pupil_t, te))
if r is None or p is None:
    return None
...
except Exception as e:
    print(f'session {sid} failed: {e}')
    continue
```

iii. The trajectory explicitly notes one skipped session due to missing eye tracking and says pupil diameter is required. It also frames the mismatch truncation and interpolation as pragmatic guards so one bad session or signal does not crash the whole conversion.

## 9-a. What are the most time-consuming steps of the code?

i. The expensive parts are loading each NWB experiment from disk with `BehaviorOphysExperiment.from_nwb_path` and then binning neural events across all cells and trials for each session.

ii.
```python
for _, row in exp_rows.iterrows():
    path = os.path.join(
        EXP_DIR, f'behavior_ophys_experiment_{row.ophys_experiment_id}.nwb')
    exps.append((row, BehaviorOphysExperiment.from_nwb_path(path)))
...
for k, ct in enumerate(change_times):
    mat[:, k, :] = bin_events_matrix(traces, tss, ct + edges_rel)
```

iii. In the trajectory, the agent says the full conversion takes about 1.2 hours, profiles decoder results on subsets, and optimizes neural binning because the conversion is CPU-heavy.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the inner per-cell event-binning loop into `bin_events_matrix`, but it still loops serially over trials, sessions, experiments, and output construction. The per-trial loops for neural binning and behavioral/output extraction could still be further vectorized.

ii.
```python
for k, ct in enumerate(change_times):
    mat[:, k, :] = bin_events_matrix(traces, tss, ct + edges_rel)
...
for k, ct in enumerate(change_times):
    ...
    img_trials.append(flash_img[j].astype(np.int64))
    ...
    run_trials.append(r)
    pup_trials.append(p)
```

iii. The trajectory explicitly shows the agent replacing an older nested cell-by-trial loop with `bin_events_matrix`, so it was aware of vectorization opportunities and focused on the neural bottleneck first.

## 9-c. What processing does the code repeat multiple times?

i. The code repeatedly computes search-based alignment and binning inside the per-trial loop, and it separately recomputes quantile edges for running and pupil within every session instead of once globally.

ii.
```python
for k, ct in enumerate(change_times):
    te = ct + edges_rel
    tc = ct + centers_rel
    j = np.searchsorted(flash_start, tc, side='right') - 1
    i = np.searchsorted(change_starts, tc, side='right') - 1
    r = fill_nans(bin_mean(run_v, run_t, te))
    p = fill_nans(bin_mean(pupil_v, pupil_t, te))
...
run_q = quantize([run_trials[k] for k in keep])
pup_q = quantize([pup_trials[k] for k in keep])
```

iii. The trajectory does not separately justify this repetition; it mainly focuses on getting a decoder-friendly representation and optimizing the biggest neural-binning hotspot.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code fully computes image, change, running, pupil, neural, and temporary outcome values for every selected trial before filtering out trials whose outcome stays `-1`. It also builds verbose `session_info` metadata that are not used by downstream decoder training.

ii.
```python
out_trials.append(o)
...
keep = [k for k in range(ntrials) if out_trials[k] >= 0]
...
info = {
    'ophys_session_id': int(session_id),
    'ophys_experiment_ids': [int(e) for e in exp_rows.ophys_experiment_id],
    ...
}
session_info.append(res['info'])
```

iii. The trajectory does not call this out explicitly. Its focus is on getting a valid converted dataset and maximizing decoder performance, not on minimizing work that is later discarded.
