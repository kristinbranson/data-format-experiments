# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local experiment metadata CSV, inventories local NWB files, retains only locally present active (`passive == False`) familiar sessions, and loads each retained experiment directly with `BehaviorOphysExperiment.from_nwb_path`. Thus it deliberately uses a paper-motivated subset rather than every released VisualBehavior experiment.

ii.
```python
tab = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
local = [int(os.path.basename(f).split('_')[-1].split('.')[0])
         for f in glob.glob(os.path.join(EXP_DIR, '*.nwb'))]
tab = tab[tab.ophys_experiment_id.isin(local)]
tab = tab[(~tab.passive) & (tab.experience_level == 'Familiar')]
...
BehaviorOphysExperiment.from_nwb_path(path)
```

iii. The trajectory says the agent chose active familiar-image sessions to match the Vip-Sst paper and ensure a shared eight-image vocabulary. It surveyed the local files first and found 110 experiments in 92 sessions.

## 1-b. How are the data split into subjects?

i. Subjects are unique string-valued `mouse_id`s, registered when a successfully processed session is appended; `subject_idx` uses the resulting list position.

ii.
```python
if res['mouse'] not in data['subjects']:
    data['subjects'].append(res['mouse'])
data['subject_idx'].append(data['subjects'].index(res['mouse']))
```

iii. The agent treated the metadata `mouse_id` as the animal identifier. Failed/skipped sessions therefore cannot introduce an otherwise unused subject.

## 1-c. How are the data split into sessions?

i. A session is one `ophys_session_id`. All experiment rows (simultaneously recorded planes) with that ID are grouped and their neurons concatenated.

ii.
```python
sessions = list(tab.groupby('ophys_session_id'))
...
for i, (sid, rows) in enumerate(sessions):
    res = process_session(sid, rows, image_names)
...
neural_all = np.concatenate(neural_planes, axis=0)
```

iii. The trajectory notes that experiments are imaging planes and that multiplane experiments belonging to one behavioral recording should form one population.

## 1-d. How are the data split into trials?

i. The SDK trials table defines trials. Only go and catch rows with a nonmissing (real or sham) `change_time` are selected. Every retained trial is represented by a fixed window from 2 s before through 4 s after that time (60 bins), rather than by its recorded start/stop boundaries.

ii.
```python
sel = trials[(trials.go | trials.catch) & trials.change_time.notna()]
change_times = sel.change_time.values.astype(float)
...
te = ct + edges_rel
```

iii. The agent surveyed trial timing and concluded `[-2, +4]` seconds fits within all selected trials and gives uniform dimensions centered on the instructed change/sham-change event.

## 1-e. How are trials filtered based on quality controls?

i. Selecting `go | catch` excludes aborted and auto-rewarded rows; missing change times are excluded. Trials without one of the four recognized outcomes are subsequently dropped, and a session must retain at least two trials. Sessions lacking usable pupil data are also omitted.

ii.
```python
sel = trials[(trials.go | trials.catch) & trials.change_time.notna()]
...
keep = [k for k in range(ntrials) if out_trials[k] >= 0]
if len(keep) < 2:
    return None
```

iii. This was justified by the explicit go/catch requirement, the minimum-two-trials validator requirement, and pupil diameter being a required output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each experiment's AllenSDK `events.filtered_events` traces and its own `ophys_timestamps`.

ii.
```python
EVENTS_COL = 'filtered_events'
ev = ex.events
traces = np.stack([np.asarray(e, dtype=float)
                   for e in ev[EVENTS_COL].values])
ts = np.asarray(ex.ophys_timestamps, dtype=float)
```

iii. The agent initially tested raw detected events, then selected half-normal-filtered detected events because tutorials supported them and subset decoder tests improved accuracy.

## 2-b. How is the `neural` data processed?

i. Per plane, filtered-event magnitudes are summed into 100 ms bins with cumulative sums. Planes are concatenated, then every neuron is divided by its standard deviation over every extracted trial/bin in that session (zero SD is replaced by one).

ii.
```python
cs = np.concatenate([np.zeros((traces.shape[0], 1)),
                     np.cumsum(traces, axis=1)], axis=1)
return cs[:, idx[1:]] - cs[:, idx[:-1]]
...
sd = neural_all.reshape(neural_all.shape[0], -1).std(axis=1)
sd[sd == 0] = 1.0
neural_all = neural_all / sd[:, None, None]
```

iii. The cumulative-sum implementation replaced a slow per-cell loop. SD scaling was chosen after decoder experiments to prevent high-amplitude cells dominating PCA and improved all-output decoding.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No extra cell/ROI quality criterion is applied. Every row supplied in `ex.events` is retained; only a trace/timestamp length mismatch is truncated to the shorter length.

ii.
```python
n = min(traces.shape[1], len(ts))
traces, tss = traces[:, :n], ts[:n]
```

iii. The agent relied on cells already exposed by the AllenSDK pipeline and added the length guard for malformed stream lengths.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural bins use edges `change_time + [-2.0, ..., 4.0]`; catch `change_time` is the sham-change time. Each plane uses its own synchronized ophys timestamps.

ii.
```python
edges = OFF_START + BIN_SIZE * np.arange(n + 1)
...
mat[:, k, :] = bin_events_matrix(traces, tss, ct + edges_rel)
```

iii. The agent explicitly chose change/sham-change alignment and a common fixed window after checking that it fits all selected trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 100 ms. Native event samples are rebinned by summing all event magnitude within each fixed bin.

ii.
```python
BIN_SIZE = 0.1
...
'time_bin_size': BIN_SIZE * 1000.0
```

iii. The agent used a shared grid to reconcile single-plane (~31 Hz) and multiplane (~10.7 Hz) recordings and make all trials exactly 60 samples.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from non-omitted change-detection `stimulus_presentations`: `start_time` and `image_name`.

ii.
```python
sp = ex0.stimulus_presentations
sp = sp[sp.stimulus_block_name.str.contains('change_detection', na=False)]
flash = sp[~sp.omitted.astype(bool)]
flash_start = flash.start_time.values.astype(float)
flash_img = np.array([image_names.index(n) for n in flash.image_name.values])
```

iii. The agent wanted the actual ongoing flashed-image identity, including presentations around the trial change, rather than only two trial-table labels.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A fixed global list of eight familiar set-A names supplies integer codes. At each bin center the latest non-omitted flash is found, and its code is held until another non-omitted image appears (including through gray screens and omissions).

ii.
```python
image_names = ['im061', 'im062', 'im063', 'im065',
               'im066', 'im069', 'im077', 'im085']
...
j = np.searchsorted(flash_start, tc, side='right') - 1
j = np.clip(j, 0, len(flash_start) - 1)
img_trials.append(flash_img[j].astype(np.int64))
```

iii. The fixed vocabulary was justified because all selected familiar sessions used image set A; holding identity across the 750 ms flash cycle gives a defined label at every neural bin.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Identity is evaluated at the same 100 ms bin centers used by the neural bin edges and has exactly 60 values per trial.

ii.
```python
te = ct + edges_rel
tc = ct + centers_rel
...
outputs.append(np.stack([img_trials[k], ...]))
```

iii. The shared change-relative grid was intended to make the categorical series sample-for-sample aligned to neural activity.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It is derived from `stimulus_presentations.is_change` and the corresponding `start_time` values.

ii.
```python
chg = sp[sp.is_change.astype(bool)]
change_starts = chg.start_time.values.astype(float)
```

iii. Actual `is_change` presentations naturally distinguish go changes from catch/sham changes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Each bin is marked as changed when it lies within the 750 ms interval beginning at the latest real change presentation.

ii.
```python
i = np.searchsorted(change_starts, tc, side='right') - 1
is_chg = np.zeros(T, dtype=np.int64)
valid = i >= 0
is_chg[valid] = (tc[valid] - change_starts[i[valid]] < FLASH_INTERVAL)
```

iii. The 750 ms duration represents the 250 ms image plus 500 ms gray cycle and makes “right after” a transient event rather than the entire post-change trial.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is directly binary: 1 for elapsed time in `[0, 0.75)` s after a true change and 0 otherwise. There is no learned threshold.

ii.
```python
is_chg = np.zeros(T, dtype=np.int64)
is_chg[valid] = (tc[valid] - change_starts[i[valid]] < FLASH_INTERVAL).astype(np.int64)
```

iii. This implements the requested binary change/no-change categories; catch trials remain zero because there is no true change presentation.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change is evaluated at the same change-relative 100 ms centers as image identity and the neural bins.

ii.
```python
tc = ct + centers_rel
...
outputs.append(np.stack([img_trials[k], chg_trials[k], ...]))
```

iii. The common grid ensures one change label per neural time bin.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `running_speed.speed` and `running_speed.timestamps` from the first experiment in a session.

ii.
```python
run = ex0.running_speed
run_t = run.timestamps.values.astype(float)
run_v = run.speed.values.astype(float)
```

iii. These are the AllenSDK's synchronized wheel-derived locomotion measurements; behavior is shared across session planes.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Native samples are averaged within each 100 ms trial bin, empty/missing bins are linearly interpolated (edge values extended), and retained values are quantized into five percentiles separately within each session.

ii.
```python
r = fill_nans(bin_mean(run_v, run_t, te))
...
run_q = quantize([run_trials[k] for k in keep])
```

iii. The agent chose bin means for the shared temporal grid and equal-percentile classes for balanced decoding. The code's placement makes the percentiles session-specific.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four 20/40/60/80% quantile edges are calculated from all retained running bins in the current session; `searchsorted(..., side='right')` yields integer classes 0–4.

ii.
```python
edges = np.quantile(v, np.linspace(0, 1, nq + 1)[1:-1])
np.searchsorted(edges, np.ravel(a), side='right').astype(np.int64)
```

iii. Equal-percentile bins were intended to balance categories, though the trajectory did not explicitly discuss the consequence of recomputing thresholds per session.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running samples whose synchronized timestamps fall within each neural edge interval are averaged, producing one value for each neural bin.

ii.
```python
r = fill_nans(bin_mean(run_v, run_t, ct + edges_rel))
```

iii. The agent relied on the hardware-synchronized clocks and shared absolute bin edges.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses `eye_tracking.pupil_area`, `likely_blink`, and `timestamps`, converting non-blink area to an equivalent circular diameter.

ii.
```python
area = eye.pupil_area.values.astype(float)
blink = eye.likely_blink.values.astype(bool)
area = np.where(blink, np.nan, area)
diam = 2.0 * np.sqrt(area / np.pi)
```

iii. Blink-contaminated frames were treated as missing. The agent regarded diameter derived from area as the requested pupil measure.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames and other NaNs are linearly interpolated across eye frames with endpoint extension. Diameter is averaged into 100 ms bins, any empty bins are interpolated again, and values are divided into session-specific quintiles.

ii.
```python
diam = fill_nans(diam)
...
p = fill_nans(bin_mean(pupil_v, pupil_t, te))
...
pup_q = quantize([pup_trials[k] for k in keep])
```

iii. This was intended to remove blink artifacts, fill minor gaps, and create balanced categorical targets. A session with no usable pupil observations is skipped.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The current session's retained pupil bins define the 20/40/60/80% edges; values become classes 0–4 using right-sided insertion.

ii.
```python
pup_q = quantize([pup_trials[k] for k in keep])
```

iii. The same equal-percentile rationale as running speed was used, with no explicit trajectory justification for session-local rather than dataset-global edges.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Eye observations are averaged using the exact absolute 100 ms edges used for neural activity.

ii.
```python
p = fill_nans(bin_mean(pupil_v, pupil_t, ct + edges_rel))
```

iii. The synchronized timestamp streams and common bin edges provide sample-wise alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome comes from the trials-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
if tr.hit: o = 0
elif tr.miss: o = 1
elif tr.false_alarm: o = 2
elif tr.correct_reject: o = 3
else: o = -1
```

iii. The agent used the SDK's mutually exclusive canonical outcomes for go/catch trials and rejected unrecognized rows.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The four booleans map to codes 0–3 in the documented order, and that static code is repeated across all 60 bins of the trial.

ii.
```python
np.full(T, out_trials[k], dtype=np.int64)
```

iii. Repetition satisfies the downstream time-shaped output representation while preserving a per-trial label.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural trace/timestamp mismatches are truncated. Behavioral NaNs and empty bins are linearly interpolated with endpoint extension. All-missing pupil sessions, sessions with too few valid trials, and sessions raising any processing exception are skipped. Unrecognized outcomes are removed.

ii.
```python
n = min(traces.shape[1], len(ts))
...
if not np.any(good): return None
return np.interp(idx, idx[good], x[good])
...
except Exception as e:
    print(f'session {sid} failed: {e}')
    continue
```

iii. The trajectory found one session with no eye tracking and accepted dropping it because pupil is required. Guards were intended to keep isolated malformed data from aborting the full conversion.

## 9-a. What are the most time-consuming steps of the code?

i. Loading/parsing every NWB experiment and extracting event/behavior tables dominates. Binning all cells and trials is also substantial; the full conversion took roughly 1–1.5 hours in the trajectory.

ii.
```python
exps.append((row, BehaviorOphysExperiment.from_nwb_path(path)))
...
for k, ct in enumerate(change_times):
    mat[:, k, :] = bin_events_matrix(traces, tss, ct + edges_rel)
```

iii. The trajectory timed sessions, observed slow full runs, and optimized event binning after a three-session test.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining per-trial neural loop and output loop could be further batched; the experiment/session loops are naturally I/O-oriented. The agent already vectorized the original per-cell event binning with matrix cumulative sums.

ii.
```python
for k, ct in enumerate(change_times):
    mat[:, k, :] = bin_events_matrix(traces, tss, ct + edges_rel)
...
for k, ct in enumerate(change_times):
    ...
```

iii. The trajectory explicitly identified per-cell/per-trial binning as slow and replaced the cell loop; it retained trial loops for clarity and variable timestamp windows.

## 9-c. What processing does the code repeat multiple times?

i. For each trial, search/summarization operations separately compute image, change, running, and pupil streams. Quantization concatenates session trial arrays after they were already built. Shared behavior tables are read from only the first plane, avoiding redundant behavioral processing, but each plane still loads a full NWB object.

ii.
```python
for k, ct in enumerate(change_times):
    te = ct + edges_rel
    tc = ct + centers_rel
    ...
```

iii. The agent focused optimization on neural binning; it did not claim or implement a second pass over the dataset.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Each NWB experiment object exposes full behavior/stimulus content although only the first plane's behavior is used. Full trial-window continuous running/pupil values exist transiently only to be quantized, and neural SD normalization requires a complete session tensor before trial lists are emitted. Metadata fields are retained for description, but no decoder input variables are produced.

ii.
```python
exps.append((row, BehaviorOphysExperiment.from_nwb_path(path)))
ex0 = exps[0][1]
...
inputs.append(np.zeros((0, T), dtype=np.float32))
```

iii. The trajectory considered the large loads necessary for per-plane neural events and verified that zero-dimensional decoder inputs were supported; it did not identify other discarded work as a major issue.
