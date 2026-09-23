# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the Allen Brain Observatory *Visual Behavior 2P* release (`visual-behavior-ophys` v1.1.0) through the AllenSDK project cache, but using `VisualBehaviorOphysProjectCache.from_local_cache(cache_dir='/app/data')` instead of `from_s3_cache` (no network in the container). The experiment table (`get_ophys_experiment_table()`, one row per imaging plane) is the master listing. Because the local cache holds only 284 of the 1,936 released experiments, the AI intersects the experiment table with the set of NWB files actually present on disk (parsed out of the `behavior_ophys_experiments/` directory listing) *before* loading anything. It then removes passive experiments (`~et.passive`), groups the remaining experiments by `ophys_session_id`, and loads each experiment with `cache.get_behavior_ophys_experiment(id)`. Both project codes present locally (`VisualBehavior`, 239 experiments, single-plane Scientifica ~31 Hz, and `VisualBehaviorMultiscope`, 45 experiments, ~11 Hz per plane) are kept. Sessions are processed in a `multiprocessing.Pool` of 16 workers, one job per `ophys_session_id`. Result: 171 sessions / 38 mice / 29,168 neurons / 43,975 trials.

ii.
```python
def get_cache():
    from allensdk.brain_observatory.behavior.behavior_project_cache import \
        VisualBehaviorOphysProjectCache
    return VisualBehaviorOphysProjectCache.from_local_cache(cache_dir=CACHE_DIR)

cache = get_cache()
exp_table = cache.get_ophys_experiment_table()

# only experiments whose NWB file is present locally
nwb_dir = os.path.join(CACHE_DIR, 'visual-behavior-ophys-1.1.0',
                       'behavior_ophys_experiments')
local_ids = sorted(int(f.split('_')[-1].split('.')[0])
                   for f in os.listdir(nwb_dir) if f.endswith('.nwb'))
et = exp_table.loc[exp_table.index.intersection(local_ids)]

# --- curation: keep only ACTIVE (behaving) sessions
et = et[~et.passive]
...
for sid in session_ids:
    rows = et[et.ophys_session_id == sid].sort_index()
    jobs.append((int(sid), list(rows.index), rows, image_names, ...))
```
```python
# inside process_session
datasets = [cache.get_behavior_ophys_experiment(int(e)) for e in exp_ids]
```

iii. From CONVERSION_NOTES Step 1/2: `from_s3_cache` (used by the tutorials) requires internet and `use_static_cache=True` fails on this directory layout, so `from_local_cache` is the offline equivalent of the reference loading path. The AI documented explicitly that "the full release table has 1936 experiments; only these 284 NWB files are present locally", so the intersection with the on-disk file list is needed to avoid attempting S3 fetches. Filtering to active sessions is justified from the paper ("Imaging was also performed during passive viewing of the same stimulus, which was not analyzed here") and from its own measurement that passive experiments contain 0 licks and 0 rewards. Multiscope sessions are kept because they are the same behavioural task (and are in fact what the Vip-Sst paper analyses); their simultaneously recorded planes share behaviour, stimulus and clock and are therefore merged into one "session".

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values of the retained experiment table, collected after conversion from the per-session results, sorted, and indexed by `subject_idx` (one entry per session). 38 mice.

ii.
```python
'mouse_id': str(meta_rows.iloc[0]['mouse_id']),
...
subjects = sorted({r['mouse_id'] for r in good})
subject_idx = np.array([subjects.index(r['mouse_id']) for r in good], dtype=np.int64)
```

iii. `mouse_id` is the SDK's canonical animal identifier in the experiment/session metadata tables. The AI cross-checked the count (38 mice in the local subset) against `ophys_experiment_table` and noted the published release contains 82 mice, the difference being explained entirely by the local data subset (CONVERSION_NOTES Step 2/Step 9).

## 1-c. How are the data split into sessions?

i. One "session" = one `ophys_session_id`. All imaging planes (experiments) recorded simultaneously in that session are merged: their dF/F matrices are vertically stacked into a single (n_neurons, n_bins) matrix per trial, and each plane contributes its `targeted_structure` to the per-neuron brain-region index. The script asserts the behaviour (`trials` table) is identical across planes of a session. Sessions are emitted in ascending `ophys_session_id` order (not chronological within mouse).

ii.
```python
session_ids = sorted(et.ophys_session_id.unique())
...
rows = et[et.ophys_session_id == sid].sort_index()
jobs.append((int(sid), list(rows.index), rows, ...))
```
```python
# consistency: all planes must share the same behaviour
for d in datasets[1:]:
    t_other = d.trials
    assert len(t_other) == len(trials), 'trial tables differ across planes'
...
for d, eid in zip(datasets, exp_ids):
    ev = np.vstack(d.dff_traces.dff.values).astype(np.float32)
    plane_events.append(ev)
    plane_ts.append(d.ophys_timestamps.astype(np.float64))
    region = meta_rows.loc[eid, 'targeted_structure']
    region_idx_parts.append(np.full(ev.shape[0], region, dtype=object))
...
neural_trials.append(np.vstack(parts).astype(np.float32))   # planes stacked per trial
```

iii. CONVERSION_NOTES Step 2/Step 5: "Multiscope sessions: up to 7 imaging planes per `ophys_session_id`, each stored as a separate NWB 'experiment' but sharing the same behavior/trials and (near-identical) ophys timestamps. Planes of one session should be merged into a single 'session' of neurons." Each plane is binned on **its own** `ophys_timestamps` before stacking, so the small clock offsets between planes are handled rather than ignored.

## 1-d. How are the data split into trials?

i. A trial is one `go` or `catch` row of `BehaviorSession.trials`. Rather than using the trials table's own `start_time`/`stop_time` (variable, 7.3–12.5 s), the AI uses a **fixed window of [-2.0, +4.0] s around `trials.change_time`**, binned at 100 ms → exactly 60 bins for every trial in every session.

ii.
```python
BIN_SIZE = 0.1          # seconds (100 ms)
OFF_START = -2.0
OFF_END = 4.0
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 60
...
trials = d0.trials
keep = (trials.go | trials.catch) & trials.change_time.notna()
tr = trials[keep]
change_times = tr.change_time.values.astype(np.float64)
...
for ct in change_times:
    edges = ct + OFF_START + BIN_SIZE * np.arange(NBINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
```

iii. CONVERSION_NOTES Step 5 decision 4: "Trial window = [-2.0, +4.0] s around the change, i.e. 60 bins. Justification: the observed change time is 2.79–8.31 s after trial start (so -2 s is always inside the trial) and `stop_time - change_time` is 4.20–4.35 s (so +4 s is always inside the trial). The window contains ~2.7 flashes before and 5.3 flashes after the change. Verified: every go/catch window lies inside the ophys, running and eye-tracking streams." The fixed window also satisfies the target-format requirement that time bins be the same size for all trials and sessions and produces equal-length trials.

## 1-e. How are trials filtered based on quality controls?

i. Filters applied, in order:
- **Session level**: only experiments whose NWB file is present locally; only **active** (non-passive) experiments (82 passive experiments removed, 0 licks / 0 rewards); sessions with **no eye-tracking table** dropped (3 sessions: 795625712, 805989030, 832881662) because pupil is a required output; sessions whose pupil trace is entirely NaN dropped; sessions with no neurons dropped; sessions with **fewer than 2** go/catch trials dropped.
- **Trial level**: keep only `go | catch` (which by SDK construction excludes aborted and auto-rewarded trials) and require `change_time` to be non-NaN.
- **Neuron level**: no additional filtering (see 2-c).
The AI verified that no trial window falls outside the ophys/running/eye-tracking time ranges in any of the 202 active experiments, so no trials are lost to truncation.

ii.
```python
et = et[~et.passive]
...
eye = d0.eye_tracking
if eye is None or len(eye) == 0:
    return {'session_id': ophys_session_id, 'skipped': 'no eye tracking'}

trials = d0.trials
keep = (trials.go | trials.catch) & trials.change_time.notna()
tr = trials[keep]
if len(tr) < 2:
    return {'session_id': ophys_session_id, 'skipped': 'fewer than 2 go/catch trials'}
...
pupil_diam = interpolate_nans(pupil_diam_raw)
if pupil_diam is None:
    return {'session_id': ophys_session_id, 'skipped': 'pupil all NaN'}
...
if n_neurons == 0:
    return {'session_id': ophys_session_id, 'skipped': 'no neurons'}
```

iii. CONVERSION_NOTES Step 3/5: the task instruction is explicit ("Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials"), and the SDK notes auto-rewards "bias the animals choice and should not be categorized as hit/miss". Passive sessions were excluded because the mouse is not performing the task, so trial outcome is meaningless there. Sessions without eye tracking were dropped because pupil diameter is a required decoder output; the AI documented that this costs 3/174 sessions, 276/29,444 neurons and 917/44,892 trials, and verified the remaining totals against `behavior_session_table` (0 mismatches over 171 sessions). Sessions with as few as 6 neurons were deliberately **kept** ("the decoder handles small populations, and dropping them would bias the sample").

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `BehaviorOphysExperiment.dff_traces.dff` — the Allen pipeline's dF/F traces — stacked across all planes of a session, with `ophys_timestamps` as the time base. The script also supports `--neural-signal events|filtered_events` (L0 detected calcium events, the paper's choice) but the shipped default, and the one used for `converted_data.pkl`, is `dff`.

ii.
```python
ap.add_argument('--neural-signal', default='dff',
                choices=['events', 'dff', 'filtered_events'], ...)
...
for d, eid in zip(datasets, exp_ids):
    if neural_signal == 'dff':
        ev = np.vstack(d.dff_traces.dff.values).astype(np.float32)
    elif neural_signal == 'filtered_events':
        ev = np.vstack(d.events.filtered_events.values).astype(np.float32)
    else:
        ev = np.vstack(d.events.events.values).astype(np.float32)
    plane_events.append(ev)
    plane_ts.append(d.ophys_timestamps.astype(np.float64))
```

iii. CONVERSION_NOTES Step 8: the AI originally chose `events` to follow the paper ("For all analysis of neural data we used the detected calcium events"), then ran the reference decoder on the *same* two sessions converted three ways and found the event traces are extremely sparse (~0.6–1 % of 100 ms bins non-zero) and decode far worse (image identity 0.236 events vs 0.488 dF/F; dF/F wins 4 of 5 outputs). "Decision: use dF/F as the neural stream ... This is a deliberate, documented deviation from the paper's analysis choice: the paper used events to remove slow calcium decay when *quantifying response magnitudes*, whereas here the goal is maximal decoding performance, and both streams come from the same standard Allen processing pipeline." It also notes dF/F is already computed by the Allen pipeline in the NWB, so nothing needs recomputing.

## 2-b. How is the `neural` data processed?

i. Per plane, the dF/F trace of every cell is bin-averaged into the trial's 100 ms bins using that plane's own `ophys_timestamps`; the per-plane results are then vertically stacked into one (n_neurons, 60) float32 matrix per trial. No normalisation, smoothing, z-scoring, baseline subtraction or detrending is added on top of the Allen pipeline. The bin averaging is vectorised across neurons and bins with `searchsorted` + cumulative sums, and it also returns the sample count per bin so that empty bins can be counted (0 empty neural bins in the whole dataset).

ii.
```python
def bin_average(values, timestamps, edges):
    lo = np.searchsorted(timestamps, edges[0], side='left')
    hi = np.searchsorted(timestamps, edges[-1], side='left')
    sub_ts = timestamps[lo:hi]
    sub = values[:, lo:hi]
    idx = np.searchsorted(sub_ts, edges, side='left')
    counts = np.diff(idx)
    csum = np.concatenate([np.zeros((sub.shape[0], 1), dtype=np.float64),
                           np.cumsum(sub.astype(np.float64), axis=1)], axis=1)
    sums = csum[:, idx[1:]] - csum[:, idx[:-1]]
    out = np.zeros_like(sums)
    nz = counts > 0
    out[:, nz] = sums[:, nz] / counts[nz]
    return out, counts
...
parts = []
for ev, ts in zip(plane_events, plane_ts):
    b, counts = bin_average(ev, ts, edges)
    empty_bins += int((counts == 0).sum())
    parts.append(b)
neural_trials.append(np.vstack(parts).astype(np.float32))
```

iii. CONVERSION_NOTES Step 1/Step 10 (check 3b): "dF/F is already computed and stored in the NWB files; no need to compute dF/F ourselves"; ROI filtering, crosstalk removal, demixing, neuropil subtraction and dF/F are all pipeline steps already applied, so "I apply no extra filtering and use the released traces as-is". Bin averaging (rather than interpolation) was chosen because the bin (100 ms) is ≥ the slowest sampling interval (11 Hz mesoscope, 93 ms), so no data are invented and every bin contains ≥1 sample (verified: 0 empty neural bins). The AI also checked that z-scoring neurons before decoding changed nothing systematically, so no rescaling was added.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality filtering is applied by the conversion script. All cells in the released `dff_traces` are kept (min 6, max 666 neurons/session; 29,168 total). The AI verified there are no NaN/Inf values and no all-zero trials, and that every session's neuron count equals the number of rows for that session's experiments in `ophys_cells_table`.

ii. No filtering code; the relevant verification is:
```python
n_neurons = sum(e.shape[0] for e in plane_events)
if n_neurons == 0:
    return {'session_id': ophys_session_id, 'skipped': 'no neurons'}
```
and the metadata note
```python
'curation': ('... Only valid ROIs are present in the released data (upstream ROI '
             'filtering, crosstalk removal, demixing, neuropil subtraction).'),
```

iii. CONVERSION_NOTES Step 3/Step 5 decision 7: "ROI filtering (union/duplicate/motion-border/dendrite/too small-narrow-dim) is already applied upstream: the released `cell_specimen_table` contains only `valid_roi == True` ROIs." The AI checked this directly in a sample experiment ("`cell_specimen_table` all `valid_roi == True`"), so any additional filter would be redundant. Low-neuron sessions were kept deliberately to avoid biasing the sample.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment event = `trials.change_time` (for catch trials the SDK's **sham** change time). Bin edges are `change_time + [-2.0, -1.9, …, +4.0]` s; each bin's value is the mean of the dF/F samples whose `ophys_timestamps` fall in `[edge_i, edge_{i+1})`. Because each plane is binned against its own timestamps, alignment is exact for every plane of a multiscope session. Metadata records `temporal_alignment_event`, `off_start = -2.0`, `off_end = +4.0`.

ii.
```python
for ct in change_times:
    edges = ct + OFF_START + BIN_SIZE * np.arange(NBINS + 1)
    ...
    for ev, ts in zip(plane_events, plane_ts):
        b, counts = bin_average(ev, ts, edges)
```
```python
'temporal_alignment_event': (
    'image change time (trials.change_time); for catch trials this is the '
    'sham change time, i.e. when the change would have occurred. Neural data '
    'are binned on the ophys timestamps.'),
'off_start': OFF_START,
'off_end': OFF_END,
```

iii. CONVERSION_NOTES Step 5 decision 3: "Alignment event = the image change (`trials.change_time`; for catch trials this is the *sham* change time ... the SDK computes it from the `sham_change` frame). This is the event that defines a trial in this task and is what the paper aligns to." The task instruction "Temporally align based on ophys timestamp" is satisfied by binning each stream onto edges evaluated against `ophys_timestamps`. The AI validated alignment physiologically (Step 12): population dF/F is flat before t=0 and rises sharply in the two bins immediately after t=0 (hit +0.62 z, miss +0.46 z, catch +0.18 z), with a 750 ms flash-locked ripple at exactly the flash times.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **100 ms bins (10 Hz), 60 bins per trial**, identical for every trial and every session. This *is* a rebinning: the native data are 31 Hz (Scientifica single-plane) and 11 Hz (Mesoscope), and both are bin-averaged down onto the common 100 ms grid. `metadata['time_bin_size'] = 100.0` ms.

ii.
```python
BIN_SIZE = 0.1
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 60
edges = ct + OFF_START + BIN_SIZE * np.arange(NBINS + 1)
...
'time_bin_size': BIN_SIZE * 1000.0,
```

iii. CONVERSION_NOTES Step 5 decision 2: "The two rigs sample at 11 Hz (dt = 93 ms) and 31 Hz (dt = 32 ms); a single common bin size is required by the target format. 100 ms is the smallest round bin that is ≥ the slowest sampling interval, so every bin contains at least one ophys sample in every session (verified) and no interpolation/upsampling of the 11 Hz data is needed. It is also matched to the ~200 ms effective resolution of the event traces (whitepaper) and to the paper's 400 ms decoding window (4 bins)." The paper's own choice (interpolating everything onto a common 30 Hz grid) was rejected because it would upsample the mesoscope data ~3×; the deviation is flagged explicitly in the Step 10 reference-code comparison table as "Deviation, justified".

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. `BehaviorSession.stimulus_presentations`, restricted to the `change_detection` stimulus block, using `start_time` and `image_name` (omitted flashes carry `image_name == 'omitted'`). Not the trials table's `initial_image_name`/`change_image_name`.

ii.
```python
sp = d0.stimulus_presentations
sp = sp[sp.stimulus_block_name.str.contains('change_detection')]
sp = sp.sort_values('start_time')
stim_start = sp.start_time.values.astype(np.float64)
stim_img   = sp.image_name.values.astype(str)
stim_change = sp.is_change.values.astype(bool)
img_to_code = {name: i for i, name in enumerate(image_names)}
stim_code = np.array([img_to_code[n] for n in stim_img], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 1/Step 5: the tutorials filter the stimulus table with `stimulus_block_name.str.contains("change_detection")` because AllenSDK ≥ 2.16 adds gray-screen and natural-movie blocks; the AI confirmed 4 blocks in every session. Using the actual flash timeline (rather than the two image names in the trials table) is what lets omitted flashes be labelled correctly and lets the identity be defined per 750 ms image-presentation interval, the paper's own convention.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each bin, the bin **centre** is assigned to the image-presentation interval that contains it: `j = searchsorted(stim_start, center) - 1`. An interval runs from its flash onset to the **onset of the next flash** (capped at 1 s); if a centre falls outside any interval it is labelled `omitted`. Image names are mapped to a **fixed global 17-class list** (the 8 images of set A ∪ the 8 images of set B, sorted, plus `omitted`), so codes are consistent across sessions and image sets.

ii.
```python
stim_end = np.empty_like(stim_start)
stim_end[:-1] = stim_start[1:]
stim_end[-1] = stim_start[-1] + FLASH_INTERVAL
stim_end = np.minimum(stim_end, stim_start + MAX_FLASH_INTERVAL)   # 1.0 s cap
...
j = np.searchsorted(stim_start, centers, side='right') - 1
j = np.clip(j, 0, len(stim_start) - 1)
within = centers < stim_end[j]
img = np.where(within, stim_code[j], img_to_code['omitted'])
img_trials.append(img.astype(np.int64))
```
```python
image_names = sorted({'im000','im031','im035','im045','im054','im073','im075','im106',
                      'im061','im062','im063','im065','im066','im069','im077','im085'})
image_names = image_names + ['omitted']
```

iii. CONVERSION_NOTES Step 5 decision 9: "the value is held for the whole 750 ms image-presentation interval (the paper's convention: 'By image presentation interval we refer to the 750 ms interval beginning with each image presentation'), i.e. the identity of the currently/most recently flashed image, matching the specification 'image identity of the image presented during the non-grey screen'. Omitted flashes get their own class `omitted` (no image was presented)." Step 9 documents a bug found and fixed here: the first version used a *fixed* 750 ms interval, but the measured inter-flash interval is 750.6 ms (range 734–801 ms), so ~1.7 % of bins fell in the sliver between intervals and were mislabelled `omitted` (inflating the omitted class from 3.2 % to 4.9 %) and a change bin was occasionally dropped. Ending each interval at the next flash onset fixed this; the resulting omitted fraction (3.2 %) matches the true omission rate in the change-detection block. A global 17-class list is used "so the shared decoder head is consistent" across image sets A and B.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is computed on exactly the same 100 ms bin grid as the neural data, using the bin centres of the same `edges` array, so output bin *k* corresponds to neural bin *k* by construction.

ii.
```python
edges = ct + OFF_START + BIN_SIZE * np.arange(NBINS + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
...
j = np.searchsorted(stim_start, centers, side='right') - 1
```

iii. CONVERSION_NOTES Step 7/10 (sanity checks 5): for every trial in 10 sessions, "the image label in the bin before t=0 equals `trials.initial_image_name` and in the bin after t=0 equals `trials.change_image_name`" — 0 mismatches. The processing plot (panel 5) overlays the stepped image-identity output on the actual coloured flash bars and shows the step changing exactly at flash onsets with the change flash at t = 0.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `stimulus_presentations.is_change` (change-detection block), combined with the same flash-interval timeline as image identity. `is_change` is True only for genuine changes, so catch (sham) trials get all zeros automatically.

ii.
```python
stim_change = sp.is_change.values.astype(bool)
```

iii. CONVERSION_NOTES Step 5 mapping table: `is_change` is the SDK's per-flash change flag; the AI verified in Step 4 that "`stimulus_presentations.is_change` count exactly equals go + auto_rewarded trials in every experiment", i.e. it marks real changes only, which matches the spec "value of 1 right after a change in image identity".

## 4-b. What processing is involved in computing `output` *Image change*?

i. Every bin whose centre falls inside the image-presentation interval of the change flash gets 1; all other bins get 0. Since an interval is ~750 ms, that is 7–8 consecutive bins starting at t = 0 on go trials, and 0 everywhere on catch trials.

ii.
```python
chg = np.where(within, stim_change[j], False).astype(np.int64)
change_trials.append(chg)
```

iii. CONVERSION_NOTES Step 5 mapping table: "1 for the bins inside the 750 ms interval of the change flash, else 0; catch (sham) trials are all 0". Rationale from Step 3: the paper's unit of behavioural analysis is "the 750 ms interval beginning with each image presentation", so marking the whole change interval (rather than a single bin) is the paper's convention and gives the decoder a label window matched to the calcium response.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary; `output_values[1] = ['no_change', 'change']`. No thresholding of a continuous variable is involved — the categories come directly from the boolean `is_change`.

ii.
```python
'output_values': [
    image_names,
    ['no_change', 'change'],
    ...
]
```

iii. Follows the instruction that image change is a "binary variable". Verified distribution: 0.117 of all bins are 1, which equals 8/60 bins × the 87.5 % go fraction (CONVERSION_NOTES Step 9).

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same bin grid and the same `j`/`within` interval assignment as image identity, hence the same alignment as the neural data. Because the window is change-aligned, the change flag always begins exactly at bin 20 (t = 0).

ii.
```python
j = np.searchsorted(stim_start, centers, side='right') - 1
within = centers < stim_end[j]
chg = np.where(within, stim_change[j], False).astype(np.int64)
```

iii. CONVERSION_NOTES Step 10 sanity check 6: "`image_change` is 1 for exactly the 8 bins of the change flash starting at bin 20 (t = 0) on go trials, and 0 everywhere on catch trials — PASS". Step 12 diagnostics confirm the label is 1 in bins 20–27 for 87 % of trials (= the go fraction).

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `BehaviorSession.running_speed` — the SDK's 60 Hz, 10 Hz low-pass-filtered wheel speed in cm/s — with its own `timestamps`.

ii.
```python
rs = d0.running_speed
run_ts = rs.timestamps.values.astype(np.float64)
run_sp = interpolate_nans(rs.speed.values)
```

iii. CONVERSION_NOTES Step 1/3: `running_speed` is the standard SDK property (whitepaper: 60 Hz encoder, 10 Hz low-pass Butterworth, cm/s); `raw_running_speed` is the unfiltered alternative and was not used. The AI verified the rate (dt = 16.7 ms, 0 NaNs) against the whitepaper.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. (1) Any NaNs in the raw trace are linearly interpolated. (2) The trace is **bin-averaged** into the trial's 100 ms bins. (3) Bins containing no sample (very rare: 8 bins in the whole dataset, 0.0003 %) are filled by linear interpolation of the trace at the bin centre rather than left at 0. (4) The binned values of *all* trials of a session are pooled and discretised into 5 equal-percentile bins.

ii.
```python
rb, rc = bin_average(run_sp[None, :], run_ts, edges)
rb = rb[0]
if (rc == 0).any():
    rb[rc == 0] = np.interp(centers[rc == 0], run_ts, run_sp)
    filled_run += int((rc == 0).sum())
run_trials.append(rb)
...
run_mat = np.vstack(run_trials)
run_lab, run_q = quantile_bins(run_mat.ravel())
run_lab = run_lab.reshape(run_mat.shape)
```

iii. Bin averaging (rather than point sampling) matches the treatment of the neural data and uses all 60 Hz samples. The empty-bin fill was added as "Iteration 2" of Step 10: the AI found that `bin_average` returns 0 for empty bins, which "silently set to 0 and pushed into the lowest quintile" — a genuine bug it detected and fixed itself.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile bins (quintiles) computed **per session** from that session's pooled binned values, via `np.quantile` at 0.2/0.4/0.6/0.8 and `searchsorted(..., side='right')`. The thresholds are stored per session in `metadata['session_info'][i]['run_quantiles']`.

ii.
```python
def quantile_bins(values, nq=NQUANT):
    qs = np.quantile(values, np.arange(1, nq) / nq)
    labels = np.searchsorted(qs, values, side='right')
    return labels.astype(np.int64), qs
```

iii. CONVERSION_NOTES Step 5 decision 8: "the per-session medians span 0–50 cm/s (running) and 63–184 px (pupil) across sessions, i.e. absolute values are not comparable across mice/rigs ... Per-session quintiles give exactly 'five equal percentile bins' within each session and a consistent meaning (relative arousal / relative speed) for the shared decoder head." This was preceded by an explicit measurement across ~30 sessions of cross-session variability (trajectory step 71–72). The verification log confirms exactly 0.200 per class in all 171 sessions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is binned onto the **same** `edges` array (change-time-aligned, 100 ms) as the neural data within the same loop, so the two are index-for-index aligned.

ii.
```python
for ct in change_times:
    edges = ct + OFF_START + BIN_SIZE * np.arange(NBINS + 1)
    ...
    rb, rc = bin_average(run_sp[None, :], run_ts, edges)
```

iii. All streams are already hardware-synced to a common session clock by the Allen sync pipeline (Step 3), so binning each stream against its own timestamps onto shared edges is sufficient. Validated by raw-data spot checks (10 sessions × 10 random bins recomputed from the raw 60 Hz trace — PASS) and by the Step 12 diagnostic that running speed collapses right after the change on hit trials (mean quintile 2.35 → 1.00) but not on misses, the expected licking-related slowdown.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `BehaviorSession.eye_tracking.pupil_area` (NaN where `likely_blink` is True), converted to a diameter as `2·sqrt(area/π)`.

ii.
```python
eye = d0.eye_tracking
eye_ts = eye.timestamps.values.astype(np.float64)
pupil_area = eye.pupil_area.values.astype(np.float64)   # NaN where likely_blink
pupil_diam_raw = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_diam = interpolate_nans(pupil_diam_raw)
```

iii. CONVERSION_NOTES Step 1/5: `eye_tracking` gives ~30 Hz DLC ellipse fits where "`likely_blink` True when fit failed/outlier; pupil_area/width/height are NaN there". The AI used the area-derived equivalent diameter rather than `pupil_width` or `pupil_height` alone; the fraction of blink frames per session is recorded as `pupil_nan_frac` in metadata (mean 3.5 %, max 30 %).

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. (1) Blink NaNs are linearly interpolated within the session (edges held at the nearest valid value). (2) The trace is bin-averaged into the same 100 ms bins. (3) Bins with no eye-camera sample (dropped frames; 3,036 bins = 0.115 % of the dataset, up to 13 % in one session) are filled by linear interpolation at the bin centre instead of being left at 0. (4) Discretised into 5 per-session equal-percentile bins.

ii.
```python
def interpolate_nans(x):
    x = np.asarray(x, dtype=np.float64).copy()
    bad = ~np.isfinite(x)
    if bad.all():
        return None
    if bad.any():
        good = ~bad
        x[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), x[good])
    return x
...
pb, pc = bin_average(pupil_diam[None, :], eye_ts, edges)
pb = pb[0]
if (pc == 0).any():
    pb[pc == 0] = np.interp(centers[pc == 0], eye_ts, pupil_diam)
    filled_pupil += int((pc == 0).sum())
pupil_trials.append(pb)
...
pupil_lab, pupil_q = quantile_bins(pupil_mat.ravel())
```

iii. CONVERSION_NOTES Step 4/10: blink interpolation is "standard practice" and avoids blink artefacts corrupting the trace; the longest blink gap (144 s in one session) is noted and the interpolated fraction is stored per session for transparency. The empty-bin interpolation was the second self-found bug fix (eye camera is only ~30 Hz, so a 100 ms bin can legitimately miss a frame when frames are dropped).

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical to running speed: five equal-percentile bins computed per session from the pooled binned values of that session; thresholds stored in `metadata['session_info'][i]['pupil_quantiles']`.

ii.
```python
pupil_lab, pupil_q = quantile_bins(pupil_mat.ravel())
pupil_lab = pupil_lab.reshape(pupil_mat.shape)
...
'output_values': [..., [f'pupil_quintile_{i+1}' for i in range(NQUANT)], ...]
```

iii. CONVERSION_NOTES Step 5 decision 8: pupil is measured "in camera pixels and depends on rig geometry", with per-session medians spanning 63–184 px, so absolute values are not comparable across sessions; per-session quintiles give a consistent "relative arousal" meaning and satisfy "five equal percentile bins" exactly (verified 0.200 per class in every session).

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same `edges`, same loop, same 100 ms bins as the neural data.

ii.
```python
for ct in change_times:
    edges = ct + OFF_START + BIN_SIZE * np.arange(NBINS + 1)
    ...
    pb, pc = bin_average(pupil_diam[None, :], eye_ts, edges)
```

iii. Eye-tracking timestamps are sync-aligned to the same session clock upstream (Step 3/4), so binning onto the shared edges is sufficient. Verified by raw-data spot checks recomputing the bin mean and quintile label of `2·sqrt(area/π)` after blink interpolation (PASS), and by the processing plot panel showing raw/interpolated/binned/quintile traces together with the session thresholds.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the trials table: `hit`, `miss`, `false_alarm`, `correct_reject`.

ii.
```python
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
outcome = np.full(len(tr), -1, dtype=np.int64)
for k, name in enumerate(OUTCOMES):
    outcome[tr[name].values.astype(bool)] = k
assert (outcome >= 0).all(), 'trial with no outcome label'
```

iii. CONVERSION_NOTES Step 1/4: `Trial._get_trial_data` defines these flags, and the AI verified across all experiments that "hit+miss == go" and "FA+CR == catch" and that go/catch/aborted/auto-rewarded partition the trials table exactly — so on the retained go/catch trials exactly one of the four flags is always set (hence the assert rather than an "other" fallback).

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Each trial's outcome is mapped to an integer 0–3 in the fixed order `[hit, miss, false_alarm, correct_reject]` and **broadcast across all 60 time bins**, so it is stored as a time-varying row that is constant within a trial. `output_values[4] = OUTCOMES`.

ii.
```python
out = np.stack([img_trials[i],
                change_trials[i],
                run_lab[i],
                pupil_lab[i],
                np.full(NBINS, outcome[i], dtype=np.int64)], axis=0)
```

iii. CONVERSION_NOTES Step 5 decision 10: "Outputs all stored time-varying (5 × 60), with the static trial outcome broadcast across time, as instructed ('If at all possible, make it time-varying')." Counts were validated per session against both the trials table and `behavior_session_table` (171/171 sessions, 0 mismatches); the resulting distribution (hit 0.317 / miss 0.558 / FA 0.019 / CR 0.106) matches the metadata tables.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **Blink NaNs in pupil** → linearly interpolated within the session; fraction recorded per session (`pupil_nan_frac`). If the whole trace is NaN the session is skipped.
- **NaNs in running speed** → same `interpolate_nans`.
- **Behavioural bins with no camera/encoder sample** (dropped frames) → filled by linear interpolation of the trace at the bin centre instead of 0 (3,036 pupil bins, 8 running bins in the full dataset).
- **Sessions with no eye-tracking table at all** (3) → skipped, with the reason printed.
- **Sessions with 0 neurons or < 2 go/catch trials** → skipped.
- **Trials with NaN `change_time`** → filtered out (0 occurrences, kept as a guard).
- **Inter-flash interval not exactly 750 ms** (734–801 ms measured) → interval ends at the next flash onset, capped at 1 s, so no unlabelled slivers.
- **Any exception in a session** → caught, the traceback is returned and printed, and the run continues with the remaining sessions.
- **Multi-plane consistency** → asserted that all planes of a session have the same trials table.

ii.
```python
try:
    ...
except Exception as exc:
    import traceback
    return {'session_id': int(ophys_session_id), 'error': f'{exc}',
            'traceback': traceback.format_exc()}
```
```python
if (pc == 0).any():
    pb[pc == 0] = np.interp(centers[pc == 0], eye_ts, pupil_diam)
...
stim_end = np.minimum(stim_end, stim_start + MAX_FLASH_INTERVAL)
...
keep = (trials.go | trials.catch) & trials.change_time.notna()
```
and the per-session counters surfaced in the summary:
```python
print(f'empty neural bins : {sum(r["empty_bins"] for r in good)}')
print(f'interp-filled bins: running {...} pupil {...}')
```

iii. CONVERSION_NOTES Step 10 "Check 5: Edge cases examined" is an explicit table of every edge case, its measured prevalence, and its handling. Two of these were bugs the AI found *itself* during critical review (the fixed-750 ms interval and the zero-filled empty behavioural bins) and fixed, re-running the full conversion and all checks after each fix. Interpolation was preferred over zero-filling because "these were silently set to 0 and pushed into the lowest pupil quintile", which would have systematically corrupted the pupil output in 51 sessions. Skipping (rather than imputing) sessions with no eye tracking at all was chosen because pupil diameter is a required decoder output; the cost was quantified (3 sessions, 276 neurons, 917 trials).

## 9-a. What are the most time-consuming steps of the code?

i. Loading the NWB files via `cache.get_behavior_ophys_experiment()` — ~3 s per imaging plane (5.5 s for a single-plane session, ~21 s for a 7-plane Multiscope session), i.e. essentially all of the ~6.5 s serial cost per session. Binning and output construction take < 0.5 s per session. The AI instrumented the code with per-step timers (`timing['load']`, `timing['events']`, `timing['bin']`) and a live ETA, and parallelised over sessions with a 16-worker pool, bringing the full 171-session run to **69 s wall clock**.

ii.
```python
t0 = time.time()
datasets = [cache.get_behavior_ophys_experiment(int(e)) for e in exp_ids]
timing['load'] = time.time() - t0
...
with Pool(min(args.workers, len(jobs))) as pool:
    for i, r in enumerate(pool.imap_unordered(process_session, jobs)):
        ...
        print(f'  [{done}/{len(jobs)}] ... eta {(len(jobs)-done)*el/done:.0f}s')
```

iii. CONVERSION_NOTES Step 6/7: "Bottleneck is NWB loading (~3 s/experiment). Parallelised over sessions with `multiprocessing.Pool` (16 workers)" — a ~16× speed-up, well under the instructions' 15-minute target. Each NWB is read exactly once per session.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining Python loops are:
- `for ct in change_times:` — the per-trial loop, which calls `bin_average` once per plane per trial and recomputes `searchsorted`/`cumsum` for each trial separately. This could be done once per session by computing all bin edges for all trials at once (a single `searchsorted` over a concatenated edge array and one session-wide cumulative sum), which would remove ~250 × n_planes calls per session.
- The inner `for ev, ts in zip(plane_events, plane_ts)` loop (small: ≤ 7 planes).
- `stim_code = np.array([img_to_code[n] for n in stim_img])` — a per-flash Python dict lookup (~5k elements/session); could use `pd.Series.map` or a factorised lookup.
- The final assembly loop `for i in range(len(tr)): np.stack([...])` — could be one vectorised stack of the five (n_trials, 60) label matrices.
- In `main`, `et[et.ophys_session_id == sid]` inside the session loop rescans the whole table per session (use `groupby` instead).
- In `make_processing_plot`, `for b in range(NBINS)` with a boolean mask over the full trace per bin (O(NBINS × n_samples)) — only runs under `--show-processing`.
The AI itself already vectorised the expensive inner work (bin averaging over neurons and bins via `searchsorted` + `cumsum`) and correctly judged that, with loading dominating by ~13×, the remaining loops are not the bottleneck.

ii. The un-vectorised per-trial loop:
```python
for ct in change_times:
    edges = ct + OFF_START + BIN_SIZE * np.arange(NBINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    parts = []
    for ev, ts in zip(plane_events, plane_ts):
        b, counts = bin_average(ev, ts, edges)
        ...
```
and the already-vectorised core:
```python
idx = np.searchsorted(sub_ts, edges, side='left')
counts = np.diff(idx)
csum = np.concatenate([np.zeros((sub.shape[0], 1)), np.cumsum(sub, axis=1)], axis=1)
sums = csum[:, idx[1:]] - csum[:, idx[:-1]]
```

iii. CONVERSION_NOTES Step 6: "Binning is vectorised over neurons and bins (cumsum + searchsorted) rather than looping" (~20× on the binning step). The AI's stated position is that loading dominates, so further vectorisation of the trial loop would yield negligible wall-clock benefit — consistent with its measurement of < 0.5 s/session for binning vs ~5.5–21 s/session for loading.

## 9-c. What processing does the code repeat multiple times?

i.
- `get_cache()` is called inside **every** `process_session` job, so the project cache and its metadata tables are constructed once per session (171 times) rather than once per worker.
- `bin_average` is re-entered per trial per plane (see 9-b); the `searchsorted` bracketing and cumulative sums are recomputed for each trial even though the underlying traces are fixed for the session.
- `d.events.index.values` is read to collect `cell_specimen_ids` even when the neural stream is `dff`, touching the events table unnecessarily.
- `stim_end` / `stim_code` are computed once per session (good), but `np.searchsorted(stim_start, centers)` is repeated per trial.
- `et[et.ophys_session_id == sid]` scans the experiment table once per session in `main`.
- The three-way neural-stream comparison (events / filtered_events / dF/F) in Step 8 required converting the same sample three times — deliberate, and only on 2 sessions.

ii.
```python
def process_session(args):
    ...
    cache = get_cache()          # rebuilt in every job
```
```python
cell_ids.extend(list(d.events.index.values))   # events touched even in dff mode
```

iii. The AI's own claim (Step 6) is "Each NWB is read exactly once; events/dff are read once per plane", which is true of the expensive NWB I/O. The repeated `get_cache()` is an artefact of the multiprocessing design (the cache object is not picklable across processes) and its cost is metadata-table parsing only, small relative to the ~3 s NWB read.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- **Whole-session loading for a 6 s window per trial**: the SDK materialises the full NWB (licks, rewards, natural-movie stimulus block, cell masks, running acceleration, all eye-tracking columns) although only dF/F, timestamps, trials, the change-detection stimulus block, running speed and pupil area are used.
- **`d.events` accessed in dF/F mode** purely to get `cell_specimen_ids` (which could come from `d.dff_traces.index` or `ophys_cells_table`).
- **`cell_specimen_ids` for all 29,168 neurons**, plus `run_quantiles`, `pupil_quantiles`, `pupil_nan_frac`, `n_go/n_catch/n_hit/...` per session, are written into `metadata['session_info']` and never read by `train_decoder.py` (they are useful provenance, but they are discarded downstream).
- **QC counters** `empty_bins`, `filled_run_bins`, `filled_pupil_bins` and the `timing` dicts are computed per session and only printed.
- **`input` arrays** of shape (0, 60) are allocated for all 43,975 trials although the task has no inputs.
- **`--show-processing` plotting** re-derives the binned running/pupil traces with a slow per-bin loop, entirely for figures.
- The neural data are stored as float32 at full 100 ms resolution for all 60 bins, giving a 2.0 GB pickle; the decoder immediately reduces to 100 PCs per session.

ii.
```python
cell_ids.extend(list(d.events.index.values))
...
'cell_specimen_ids': [int(c) for c in cell_ids],
'empty_bins': int(empty_bins),
'filled_run_bins': int(filled_run),
'timing': timing,
...
inputs = [np.zeros((0, NBINS), dtype=np.float32) for _ in range(len(tr))]
```

iii. Most of these are deliberate: CONVERSION_NOTES Step 9/10 uses the per-session counts, quantile thresholds and `pupil_nan_frac` for the consistency tables and the raw-data spot checks, and the task explicitly asks for `session_info` in metadata and for processing plots. The AI did not flag the `d.events` access in dF/F mode or the (0, 60) input allocation as waste; neither is material (the NWB is already fully parsed by the SDK at load time, and the empty arrays cost nothing).
