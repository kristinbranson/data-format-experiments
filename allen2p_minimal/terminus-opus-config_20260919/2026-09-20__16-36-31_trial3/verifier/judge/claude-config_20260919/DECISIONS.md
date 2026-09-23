# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI works entirely from the local copy of the `visual-behavior-ophys-1.1.0` release rather than the SDK's S3/local cache object. It reads the project metadata CSV `project_metadata/ophys_experiment_table.csv`, intersects it with the `*.nwb` files actually present in `behavior_ophys_experiments/` (284 files), and then applies a scientific filter: **non-passive (active-behavior) sessions with `experience_level == 'Familiar'`**. That leaves 110 experiments / 92 ophys sessions / 38 mice, spanning both the `VisualBehavior` (single-plane) and `VisualBehaviorMultiscope` project codes. Each selected experiment is opened individually with `BehaviorOphysExperiment.from_nwb_path(...)`, which gives the same SDK object interface (`trials`, `stimulus_presentations`, `events`, `running_speed`, `eye_tracking`, `ophys_timestamps`) that the cache API returns. Sessions are then processed one at a time in a loop over `tab.groupby('ophys_session_id')`, and each is discarded from memory before the next is loaded.

ii.
```python
DATA_DIR = '/app/data/visual-behavior-ophys-1.1.0'
EXP_DIR = os.path.join(DATA_DIR, 'behavior_ophys_experiments')
META_DIR = os.path.join(DATA_DIR, 'project_metadata')

def get_experiment_table():
    tab = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
    local = [int(os.path.basename(f).split('_')[-1].split('.')[0])
             for f in glob.glob(os.path.join(EXP_DIR, '*.nwb'))]
    tab = tab[tab.ophys_experiment_id.isin(local)]
    # active behavior (mouse performing the change detection task) with familiar images
    tab = tab[(~tab.passive) & (tab.experience_level == 'Familiar')]
    return tab.sort_values(['ophys_session_id', 'ophys_experiment_id'])
```

```python
    for _, row in exp_rows.iterrows():
        path = os.path.join(
            EXP_DIR, f'behavior_ophys_experiment_{row.ophys_experiment_id}.nwb')
        exps.append((row, BehaviorOphysExperiment.from_nwb_path(path)))
```

```python
    sessions = list(tab.groupby('ophys_session_id'))
    for i, (sid, rows) in enumerate(sessions):
        try:
            res = process_session(sid, rows, image_names)
        except Exception as e:
            print(f'session {sid} failed: {e}')
            continue
```

iii. From the trajectory (steps 14–15): the AI first tried `VisualBehaviorOphysProjectCache.from_local_cache`, found that "Local cache API needs a specific folder structure", and concluded "Simpler to load NWB files directly via `BehaviorOphysExperiment.from_nwb_path`, using the CSV metadata tables for filtering." The scientific restriction is justified in the module docstring as "matching the Vip-Sst paper, which restricted neural analysis to familiar image-set sessions during active behavior" — which matches `methods.txt` ("For neural analysis we used neurons recorded during familiar image set presentations…", "we restricted our analysis to familiar stimuli"). Before committing, the AI ran a full survey of all 110 candidate experiments (step 27) recording cell counts, frame rate, trial counts, trial timing, and eye-tracking availability, and used those statistics to size the trial window and anticipate missing data.

## 1-b. How are the data split into subjects?

i. Subjects are the `mouse_id` field of the experiment table. The AI takes the `mouse_id` of the first experiment row of each session, keeps a list of unique mouse ids in first-encounter order, and stores the index of that mouse in `subject_idx` for the session. 38 mice result.

ii.
```python
    row0 = exp_rows.iloc[0]
    ...
    return dict(..., mouse=str(row0.mouse_id), info=info)
```

```python
        if res['mouse'] not in data['subjects']:
            data['subjects'].append(res['mouse'])
        ...
        data['subject_idx'].append(data['subjects'].index(res['mouse']))
```

iii. Not discussed explicitly beyond the survey step, where the AI counted "38 mice" for the selected experiment set (step 34). `mouse_id` is the SDK's canonical animal identifier and is constant across all experiments of a session, so taking it from the first row is safe.

## 1-c. How are the data split into sessions?

i. A session is one `ophys_session_id`. The experiment table is grouped by `ophys_session_id`, and all imaging planes (experiments) belonging to that id are processed together: their cells are concatenated into a single population for the session. Behavior/stimulus/trial streams are read from the first plane of the session (they are identical across planes of a session). 92 candidate sessions (88 single-plane, 4 Multiscope with 3–7 planes) were found; 91 made it into the output.

ii.
```python
    sessions = list(tab.groupby('ophys_session_id'))
```

```python
def process_session(session_id, exp_rows, image_names):
    ...
    for _, row in exp_rows.iterrows():
        ...
        exps.append((row, BehaviorOphysExperiment.from_nwb_path(path)))
    ex0 = exps[0][1]
    ...
    neural_all = np.concatenate(neural_planes, axis=0)  # (ncells, ntrials, T)
```

iii. Module docstring: "A 'session' is one ophys session (`ophys_session_id`). For the multi-plane (Mesoscope) sessions the simultaneously recorded imaging planes (experiments) are concatenated into a single population, since they are one recording." This matches `methods.txt`: "The data collected in a single continuous recording is defined as a session… For multi-plane imaging experiments, there can be up to 8 imaging planes (8 experiments) per session." In step 35 the AI explicitly checked how planes group into sessions and whether Multiscope planes share behavior data before merging them.

## 1-d. How are the data split into trials?

i. Trials come from the SDK's built-in `trials` table. The AI keeps rows where `go | catch` **and** `change_time` is not NaN — this by construction excludes aborted and auto-rewarded trials (it verified empirically that `go & auto_rewarded == 0` and `catch & auto_rewarded == 0`). Each trial is then represented by a **fixed-length window around the (sham) change time**: `[-2.0, +4.0] s` in 100 ms bins, i.e. exactly 60 bins for every trial in every session. 22,778 trials across 91 sessions (39–396 per session, median 263).

ii.
```python
BIN_SIZE = 0.1          # seconds
OFF_START = -2.0        # seconds relative to change time
OFF_END = 4.0           # seconds relative to change time

def bin_edges():
    n = int(round((OFF_END - OFF_START) / BIN_SIZE))
    edges = OFF_START + BIN_SIZE * np.arange(n + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers
```

```python
    # ---- trials: go and catch only (excludes aborted and auto-rewarded) ----
    trials = ex0.trials
    sel = trials[(trials.go | trials.catch) & trials.change_time.notna()]
    if len(sel) < 2:
        return None
    change_times = sel.change_time.values.astype(float)
```

iii. Module docstring: "Trials: 'go' and 'catch' trials (aborted and auto-rewarded excluded), aligned to the (sham) change time, window [-2.0, +4.0] s, **which fits inside every trial**." The window was chosen empirically: the survey (steps 26/35) measured, over all 110 experiments, a minimum change-to-trial-start of 2.79 s and a minimum stop-time-to-change of ~4.2 s, so `[-2, +4]` never runs outside a trial. Step 24 was devoted to verifying trials-table semantics (`go`/`catch`/`auto_rewarded` overlap, `change_time` presence for catch trials) before writing the selector.

## 1-e. How are trials filtered based on quality controls?

i. Filters applied, in order:
- Session level: only active + Familiar sessions (1-a); a session is skipped entirely if it has fewer than 2 go/catch trials, if it has no eye-tracking rows or all-NaN pupil, if any trial's binned running/pupil cannot be filled, or if any exception is raised while processing it.
- Trial level: `go | catch` with non-null `change_time` (excludes aborted and auto-rewarded); trials whose outcome is none of hit/miss/false_alarm/correct_reject are dropped (`o = -1` → excluded from `keep`); after dropping, at least 2 trials must remain.
- No trial is dropped for neural-data reasons (e.g. the 1,244 / 22,778 trials in which every neuron has zero events in the window are kept).

ii.
```python
    sel = trials[(trials.go | trials.catch) & trials.change_time.notna()]
    if len(sel) < 2:
        return None
```

```python
    if pupil_v is None:
        return None   # pupil diameter is a required decoder output
```

```python
        r = fill_nans(bin_mean(run_v, run_t, te))
        p = fill_nans(bin_mean(pupil_v, pupil_t, te))
        if r is None or p is None:
            return None
```

```python
    keep = [k for k in range(ntrials) if out_trials[k] >= 0]
    if len(keep) < 2:
        return None
```

iii. The instructions say to "Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials", and the AI implements exactly that. The `>= 2` trial requirement is the format requirement ("There needs to be at least two trials within each session in order to evaluate the decoder performance"). Dropping sessions without eye tracking is justified in an inline comment as "pupil diameter is a required decoder output"; the survey had already established this affects only one session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Detected calcium events, not dF/F. Specifically the `filtered_events` column of `BehaviorOphysExperiment.events` (the SDK's detected events convolved with a half-normal filter), together with each plane's own `ophys_timestamps`.

ii.
```python
EVENTS_COL = 'filtered_events'  # AllenSDK detected calcium events, half-Gaussian filtered
...
    for row, ex in exps:
        ts = np.asarray(ex.ophys_timestamps, dtype=float)
        ev = ex.events
        traces = np.stack([np.asarray(e, dtype=float) for e in ev[EVENTS_COL].values])
```

iii. Module docstring: "Neural data: detected calcium events, as used by the paper. We use the AllenSDK 'filtered_events' (the same detected events convolved with a half-normal filter, as recommended in the SDK tutorials)". This follows `methods.txt`: "For all analysis of neural data we used the detected calcium events… We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces, thus removing the slow decay dynamics of GCaMP6f." The AI initially used the raw `events` column, observed that "Raw event magnitudes are extremely sparse and heavy-tailed (max ~1000, 98% zeros), which is poor for a linear PCA decoder" (step 106), grepped the tutorials and SDK for `filtered_events` (step 122) and found the tutorial text recommending it, then compared both on a 12-session subset: image-identity validation accuracy 0.25 (raw events) vs 0.33 (filtered events).

## 2-b. How is the `neural` data processed?

i. Three steps: (1) event magnitudes are **summed into fixed 100 ms bins** on each plane's own ophys timestamps, using a cumulative-sum trick so all cells are binned at once; (2) planes of a Multiscope session are concatenated along the cell axis into one population; (3) **each neuron is divided by its standard deviation** computed over all of that session's extracted trial bins (zero-SD neurons are left unscaled). No mean subtraction, no smoothing, no additional filtering.

ii.
```python
def bin_events_matrix(traces, timestamps, t_edges):
    """Sum of calcium event magnitudes in each bin, for all cells at once."""
    idx = np.clip(np.searchsorted(timestamps, t_edges), 0, traces.shape[1])
    cs = np.concatenate([np.zeros((traces.shape[0], 1)), np.cumsum(traces, axis=1)],
                        axis=1)
    return cs[:, idx[1:]] - cs[:, idx[:-1]]
```

```python
        mat = np.zeros((traces.shape[0], ntrials, T), dtype=np.float32)
        for k, ct in enumerate(change_times):
            mat[:, k, :] = bin_events_matrix(traces, tss, ct + edges_rel)
        neural_planes.append(mat)
        regions += [row.targeted_structure] * traces.shape[0]

    neural_all = np.concatenate(neural_planes, axis=0)  # (ncells, ntrials, T)

    # Normalize each neuron by its standard deviation across all extracted bins of
    # the session. ...
    sd = neural_all.reshape(neural_all.shape[0], -1).std(axis=1)
    sd[sd == 0] = 1.0
    neural_all = (neural_all / sd[:, None, None]).astype(np.float32)
```

iii. Binning is needed to put single-plane (32 ms frames) and Multiscope (93 ms frames) sessions on one common time base — see 2-e. The per-neuron normalization is justified in an inline comment: "Event magnitudes vary by orders of magnitude across cells; without this a handful of high-variance cells would dominate the decoder's PCA projection. The scaling is per neuron and constant in time, so it does not mix information across time or cells." The choice was made empirically (steps 113/121/122): on a 12-session subset, filtered events alone gave image-identity validation accuracy 0.33 and filtered events + per-neuron SD scaling gave 0.356, versus 0.25 for the unnormalized raw events baseline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality control is applied — every cell in the `events` table is kept (the AI inspected `cell_specimen_table.valid_roi` during exploration, found it to be 1.0, and did not use it as a filter). The only neural-data guard is a defensive truncation when the number of event samples and the number of ophys timestamps disagree. Consequences visible in the output: sessions with as few as 7 neurons are kept, and 1,244 of 22,778 trials (5.5%) contain no events at all in the window (the decoder's verifier warns "all neural data is zero" for these).

ii.
```python
        traces = np.stack([np.asarray(e, dtype=float) for e in ev[EVENTS_COL].values])
        # guard against length mismatch
        n = min(traces.shape[1], len(ts))
        traces, tss = traces[:, :n], ts[:n]
```

iii. Never stated explicitly. Implicitly the AI relies on the fact that the Allen pipeline has already performed segmentation/QC on released experiments (its exploratory check of `valid_roi == 1.0` in step 26 supports this), and on its session-level filters. The all-zero-trial warnings were visible in the training log at step 105 and were not acted on.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to the **image change time** (`trials.change_time`), which on catch trials is the sham-change time. The bin edges `change_time + [-2.0 … +4.0]` s are converted to frame indices with `np.searchsorted` **on each plane's own `ophys_timestamps`**, so for Multiscope sessions every plane is resampled onto the same trial-relative grid despite having different frame times. Alignment is therefore exact to within one 100 ms bin for every plane. `metadata['temporal_alignment_event']`, `off_start` and `off_end` record this.

ii.
```python
        ts = np.asarray(ex.ophys_timestamps, dtype=float)
        ...
        for k, ct in enumerate(change_times):
            mat[:, k, :] = bin_events_matrix(traces, tss, ct + edges_rel)
```

```python
    idx = np.clip(np.searchsorted(timestamps, t_edges), 0, traces.shape[1])
```

```python
        'temporal_alignment_event': (
            'image change time on go trials (sham change time on catch trials)'),
        'off_start': OFF_START,
        'off_end': OFF_END,
```

iii. The docstring states trials are "aligned to the (sham) change time, window [-2.0, +4.0] s, which fits inside every trial". The change (or sham change) is the behaviourally and visually defining event of a trial in this task, and aligning to it makes the pre-change baseline, the change, and the response window occupy the same bins in every trial. The AI verified from the survey that this window never crosses a trial boundary.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 100 ms bins, 60 bins per trial, identical for every trial and session; `metadata['time_bin_size'] = 100.0`. Yes — rebinning is applied: native ophys frames are 32 ms for the 87 single-plane sessions and 93 ms for the 4 Multiscope sessions, and event magnitudes are **summed** within each 100 ms bin (running speed and pupil are **averaged** within the same bins).

ii.
```python
BIN_SIZE = 0.1          # seconds
...
        'time_bin_size': BIN_SIZE * 1000.0,
```

```python
def bin_events_matrix(traces, timestamps, t_edges):   # sum within bin (neural)
def bin_mean(values, timestamps, t_edges):            # mean within bin (behaviour)
```

iii. The survey (steps 34–35) established "single-plane dt=32ms (88 sessions) and multiscope dt=93ms (4 sessions)". Since the target format requires "Time bins should be the same size for all trials and sessions", a common grid is required once the two rigs are combined; 100 ms also matches the SDK tutorial's note that detected events have "the resolution of about 200 ms", so 100 ms does not throw away meaningful temporal detail.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From the **stimulus presentations table**, not the trials table: `stimulus_presentations.image_name`, `.start_time`, `.omitted`, restricted to the `change_detection_behavior` stimulus block (this drops the natural-movie and gray-screen blocks).

ii.
```python
    sp = ex0.stimulus_presentations
    sp = sp[sp.stimulus_block_name.str.contains('change_detection', na=False)]
    flash = sp[~sp.omitted.astype(bool)]
    flash_start = flash.start_time.values.astype(float)
    flash_img = np.array([image_names.index(n) for n in flash.image_name.values])
```

iii. The AI inspected `stimulus_presentations` in step 24 (image_name counts, `stimulus_block_name` counts, `omitted`, `is_change`) and found the behaviour block is a separate block from the natural-movie block. Using the presentation table gives the identity of the image actually flashed at every moment of the window, independent of trial bookkeeping, and makes it straightforward to hold identity through omitted flashes.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes 0–7 using a **hard-coded, globally fixed list of the eight image-set-A names** (`im061, im062, im063, im065, im066, im069, im077, im085`), which is also written into `output_values[0]`. For each 100 ms bin the label is the identity of the most recent non-omitted flash, i.e. the image is held through the 500 ms gray inter-stimulus interval and through omitted flashes ("the image presentation interval" of the paper).

ii.
```python
    image_names = ['im061', 'im062', 'im063', 'im065',
                   'im066', 'im069', 'im077', 'im085']
```

```python
        # image identity: identity of the image presented in the ongoing 750 ms
        # image presentation interval, held through the gray screen and omissions
        j = np.searchsorted(flash_start, tc, side='right') - 1
        j = np.clip(j, 0, len(flash_start) - 1)
        img_trials.append(flash_img[j].astype(np.int64))
```

iii. Docstring: "All locally available familiar active sessions use image set A, so image identity labels are shared across sessions." This was verified in step 26 (`act.image_set.value_counts()` → `A 110`). Holding identity through gray/omission follows the paper's "image presentation interval… By image presentation interval we refer to the 750 ms interval beginning with each image presentation. For image omissions we used the 750 ms following the time of the omission."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is evaluated at the **centre of each of the same 60 bins** used for the neural data (`tc = change_time + centers_rel`), so it shares exactly the neural time base by construction.

ii.
```python
    edges_rel, centers_rel = bin_edges()
    ...
    for k, ct in enumerate(change_times):
        te = ct + edges_rel
        tc = ct + centers_rel
        j = np.searchsorted(flash_start, tc, side='right') - 1
```

iii. Implicit: neural, image identity, image change, running and pupil are all computed from the same `edges_rel`/`centers_rel` grid anchored on the same `change_time`, so no cross-stream alignment step is needed. Stimulus, behaviour and ophys timestamps are already on a common hardware-synced clock in the NWB files.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From `stimulus_presentations.is_change` (restricted to the change-detection block): the start times of the flashes flagged as an actual image change.

ii.
```python
    chg = sp[sp.is_change.astype(bool)]
    change_starts = chg.start_time.values.astype(float)
```

iii. The AI examined `is_change` in step 24 alongside `omitted`/`start_time`. Using the stimulus table rather than `trials.change_time` means only *real* changes are marked — catch (sham-change) trials have no `is_change` flash, so they automatically get all-zero labels.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each bin, find the most recent actual change onset; the bin is labelled 1 if its centre is less than `FLASH_INTERVAL = 0.75` s after that onset, otherwise 0. That is: 1 for the 750 ms image-presentation interval that begins at the change (the 250 ms changed image plus the following 500 ms gray), 0 elsewhere. Result: 10.2% of bins are labelled `change`.

ii.
```python
FLASH_INTERVAL = 0.75   # image presentation interval (s), as in the paper
...
        # image change: 1 during the 750 ms interval starting at an actual image
        # change (catch/sham trials contain no change)
        i = np.searchsorted(change_starts, tc, side='right') - 1
        is_chg = np.zeros(T, dtype=np.int64)
        valid = i >= 0
        is_chg[valid] = (tc[valid] - change_starts[i[valid]] < FLASH_INTERVAL).astype(np.int64)
        chg_trials.append(is_chg)
```

iii. The 750 ms interval is taken directly from the paper's definition of the image presentation interval (250 ms image + 500 ms gray), cited in the constant's comment "as in the paper". The instruction asks for a variable that has "value of 1 right after a change in image identity, otherwise 0", which this implements as the one-presentation-interval transient rather than the whole post-change period.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is natively binary, so no thresholding is needed: values are 0/1 with `output_values = ['no_change', 'change']`. The only implicit threshold is the 750 ms duration of the "change" label described in 4-b.

ii.
```python
            'output_values': [image_names,
                              ['no_change', 'change'],
                              ...
```

iii. The instruction specifies a binary variable; no discretization is required.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity: evaluated at the centres of the same 60 bins anchored on the same `change_time`, so it is aligned with the neural data by construction. Because trials are aligned to the change, the `1` block sits at the same bins (bins 20–27) in every go trial.

ii.
```python
        tc = ct + centers_rel
        i = np.searchsorted(change_starts, tc, side='right') - 1
```

iii. Same reasoning as 3-c — a single shared time grid for every stream.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `BehaviorOphysExperiment.running_speed`, using its `speed` (cm/s, the SDK's wrap-corrected, transient-removed, 10 Hz low-pass filtered signal) and `timestamps` columns, taken from the first plane of the session.

ii.
```python
    run = ex0.running_speed
    run_t = run.timestamps.values.astype(float)
    run_v = run.speed.values.astype(float)
```

iii. Not discussed in detail; `running_speed` is the SDK's standard product, and `methods.txt` describes it as the filtered estimate intended for end users. The survey (step 27) confirmed running speed is present with no NaNs in every selected experiment.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. (1) The raw ~60 Hz speed trace is **averaged within each 100 ms bin** of the trial window (`bin_mean`); bins containing no samples become NaN. (2) Remaining NaNs are filled by linear interpolation over the 60-bin vector, with edge extension (`fill_nans`). (3) The binned values are discretized into 5 equal-percentile bins — with the quantile edges computed **within each session**, over all of that session's kept trials, rather than globally across the dataset.

ii.
```python
def bin_mean(values, timestamps, t_edges):
    """Mean of a continuous signal within each bin (NaN if no samples)."""
    idx = np.searchsorted(timestamps, t_edges)
    vals = np.concatenate([values, [0.0]])
    sums = np.add.reduceat(vals, idx[:-1])
    counts = (idx[1:] - idx[:-1]).astype(float)
    out = np.full(len(counts), np.nan)
    good = counts > 0
    out[good] = sums[good] / counts[good]
    return out


def quantize(values, nq=NQUANTILES):
    """Discretize into nq equal-percentile bins (quantile edges from the data)."""
    v = np.concatenate([np.ravel(a) for a in values])
    edges = np.quantile(v, np.linspace(0, 1, nq + 1)[1:-1])
    return [np.searchsorted(edges, np.ravel(a), side='right').astype(np.int64)
            for a in values]
```

```python
        r = fill_nans(bin_mean(run_v, run_t, te))
        ...
        run_trials.append(r)
    ...
    run_q = quantize([run_trials[k] for k in keep])     # inside process_session → per session
```

iii. Averaging (rather than point-sampling/interpolating) is the natural down-sampling operator for a signal sampled faster than the 100 ms bin. The instruction asks for "five equal percentile bins"; the AI's metadata describes them as "each discretized into 5 equal-percentile bins" without stating the scope. The per-session scope is not justified anywhere in the trajectory or the code comments. Its measured effect is that every session contributes exactly 20% of samples to each quintile (verified in the saved data).

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four interior quantile edges (20/40/60/80th percentiles) of the session's own binned running-speed distribution, applied with `np.searchsorted(..., side='right')` to give integer classes 0–4, named `Q1_0-20%` … `Q5_80-100%`. In the saved dataset each class holds exactly 20.0% of bins overall and within every single session; no session has a collapsed (empty) bin.

ii.
```python
NQUANTILES = 5
...
    edges = np.quantile(v, np.linspace(0, 1, nq + 1)[1:-1])
    return [np.searchsorted(edges, np.ravel(a), side='right').astype(np.int64)
            for a in values]
```

```python
            'output_values': [..., ['Q1_0-20%', 'Q2_20-40%', 'Q3_40-60%',
                                    'Q4_60-80%', 'Q5_80-100%'], ...]
```

iii. Directly from the instruction "Running speed, discretized into five equal percentile bins". Using empirical quantiles guarantees balanced classes, which matters for the balanced-accuracy metric the decoder reports.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is binned on exactly the same `change_time + edges_rel` boundaries as the neural data, so bin *i* of the running vector covers the same wall-clock interval as bin *i* of the neural matrix for that trial.

ii.
```python
    for k, ct in enumerate(change_times):
        te = ct + edges_rel
        ...
        r = fill_nans(bin_mean(run_v, run_t, te))
```

iii. Same single-grid argument as 3-c: all streams are binned with the same edges relative to the same event, so no separate alignment step or interpolation onto a foreign timebase is required.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `BehaviorOphysExperiment.eye_tracking`: the `pupil_area` column, the `likely_blink` flag, and `timestamps`. Pupil **diameter** is computed from area as the diameter of the equivalent circle.

ii.
```python
    eye = ex0.eye_tracking
    pupil_t, pupil_v = None, None
    if len(eye) > 0:
        area = eye.pupil_area.values.astype(float)
        blink = eye.likely_blink.values.astype(bool)
        area = np.where(blink, np.nan, area)
        diam = 2.0 * np.sqrt(area / np.pi)   # pupil diameter from area
        diam = fill_nans(diam)
        if diam is not None:
            pupil_t = eye.timestamps.values.astype(float)
            pupil_v = diam
    if pupil_v is None:
        return None   # pupil diameter is a required decoder output
```

iii. Step 24 printed descriptive statistics for `pupil_width`, `pupil_height` and `pupil_area` and the blink fraction (~9%) before the choice was made; the survey then measured eye-tracking availability and NaN fraction for all 110 experiments. Deriving diameter from area (rather than taking `pupil_width` or `pupil_height` alone) combines both fitted ellipse axes and so is less sensitive to axis-specific tracking noise. Blink frames are removed because the fitted ellipse is meaningless during a blink.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. (1) Blink frames → NaN; (2) area → equivalent-circle diameter; (3) all NaNs in the session-long trace linearly interpolated (`fill_nans`), including non-blink tracking dropouts; (4) averaged into the same 100 ms bins as everything else; (5) any bin with no eye-tracking sample filled by a second linear interpolation; (6) discretized into 5 equal-percentile bins with **per-session** quantile edges, exactly as for running speed.

ii.
```python
        diam = 2.0 * np.sqrt(area / np.pi)   # pupil diameter from area
        diam = fill_nans(diam)
```

```python
        p = fill_nans(bin_mean(pupil_v, pupil_t, te))
        if r is None or p is None:
            return None
        pup_trials.append(p)
    ...
    pup_q = quantize([pup_trials[k] for k in keep])
```

iii. Same rationale as running speed. Interpolating across blinks (instead of leaving them missing or mapping them to a fixed class) keeps the discretized signal continuous and avoids creating an artificial "blink" class inside the lowest quintile. Whole sessions without usable eye tracking are dropped instead (one session).

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical machinery to running speed: 20/40/60/80th percentiles of that session's binned pupil-diameter values, giving classes 0–4 labelled `Q1_0-20%` … `Q5_80-100%`. In the output every class holds exactly 20.0% of bins, both overall and per session.

ii.
```python
    pup_q = quantize([pup_trials[k] for k in keep])
```

iii. Directly from the instruction "Pupil diameter, discretized into five equal percentile bins."

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed: averaged over the same `change_time + edges_rel` bin boundaries as the neural data, so bin indices correspond one-to-one.

ii.
```python
        te = ct + edges_rel
        p = fill_nans(bin_mean(pupil_v, pupil_t, te))
```

iii. Same shared-grid argument as 3-c/5-d.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the trials table: `hit`, `miss`, `false_alarm`, `correct_reject`, read from the row of the selected (go/catch) trial.

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

iii. Step 24 checked that these labels partition the go/catch trials as expected (`go & hit`, `go & miss`, `catch & false_alarm`, `catch & correct_reject` counts, and that auto-rewarded trials are neither go nor catch). Hit/miss apply to go trials, false alarm/correct reject to catch trials, which is the standard signal-detection labelling of this task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Mapped to a fixed code 0–3 with `output_values = ['hit','miss','false_alarm','correct_reject']`, then **broadcast as a constant across all 60 bins** of the trial so that it has the same time-varying shape as the other outputs. Trials matching none of the four labels get `-1` and are dropped from the session entirely, so no invalid code reaches the output (the saved data ranges 0–3; distribution: 30.0% hit, 57.4% miss, 2.2% false alarm, 10.4% correct reject).

ii.
```python
        out_trials.append(o)

    keep = [k for k in range(ntrials) if out_trials[k] >= 0]
    if len(keep) < 2:
        return None
```

```python
        outputs.append(np.stack([
            img_trials[k],
            chg_trials[k],
            run_q[n_],
            pup_q[n_],
            np.full(T, out_trials[k], dtype=np.int64),
        ]).astype(np.int64))
```

iii. The format spec says outputs "Can be time-varying or discrete values per trial. If at all possible, make it time-varying", and all five outputs are stacked into one `(5, T)` array, so the static outcome is tiled. Metadata records it as "the trial outcome (hit/miss/false alarm/correct reject, constant within a trial)".

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several guards, all of which either repair the value or drop the unit of data:
- **Ophys trace / timestamp length mismatch**: traces and timestamps are truncated to the shorter length.
- **Blinks and eye-tracker dropouts**: set to NaN and linearly interpolated over the session-long trace.
- **Bins with no behaviour samples**: produced as NaN by `bin_mean`, then linearly interpolated across the 60-bin vector (edge-extended at the ends).
- **Completely missing eye tracking / all-NaN pupil**: the whole session is dropped (1 of 92).
- **Trials with no outcome label**: dropped.
- **Sessions with fewer than 2 usable trials**: dropped.
- **Any other exception**: caught per session in `main`, printed, and the session skipped.
- Not handled: trials in which no neuron fires (5.5% of trials are all-zero neural matrices) are kept; the brain-region list `['VISp','VISl']` and the 8-image name list are hard-coded, so an unexpected structure or image set would raise inside `process_session` and be swallowed by the bare `except`, silently dropping that session.

ii.
```python
        n = min(traces.shape[1], len(ts))
        traces, tss = traces[:, :n], ts[:n]
```

```python
def fill_nans(x):
    """Linear interpolation of NaNs, extended at the edges."""
    x = np.asarray(x, dtype=float)
    good = np.isfinite(x)
    if not np.any(good):
        return None
    if np.all(good):
        return x
    idx = np.arange(len(x))
    return np.interp(idx, idx[good], x[good])
```

```python
        try:
            res = process_session(sid, rows, image_names)
        except Exception as e:
            print(f'session {sid} failed: {e}')
            continue
        if res is None:
            print(f'session {sid} skipped')
            continue
```

iii. The `pupil_v is None → return None` branch is commented "pupil diameter is a required decoder output". The survey step measured the eye-tracking NaN fraction and the number of sessions with zero eye-tracking rows in advance, so the AI knew the cost of this rule (one session). The try/except keeps a single bad session from aborting a multi-hour conversion. In the actual run, exactly one session was reported as "skipped" and none as "failed".

## 9-a. What are the most time-consuming steps of the code?

i. Two steps dominate, both inside `process_session`:
1. **Opening the NWB files and materialising `ex.events`** — for the `filtered_events` column the SDK convolves every cell's full-session event train with a half-normal filter at load time. This is the I/O + compute bottleneck.
2. **Binning the neural data**: `bin_events_matrix` is called once per trial per plane, and each call recomputes `np.cumsum` over the *entire session* trace matrix (`n_cells × ~140,000` frames for single-plane sessions) even though only 60 bins are read out of it. With ~250 trials per session this is ~250 redundant full-session cumsums.

Measured: the final full conversion took ~1 minute per session, ~1.5 h for 92 sessions. Everything after the per-session loop (quantization, assembly, pickling of the 953 MB file) is comparatively cheap.

ii.
```python
        mat = np.zeros((traces.shape[0], ntrials, T), dtype=np.float32)
        for k, ct in enumerate(change_times):
            mat[:, k, :] = bin_events_matrix(traces, tss, ct + edges_rel)
```

```python
def bin_events_matrix(traces, timestamps, t_edges):
    idx = np.clip(np.searchsorted(timestamps, t_edges), 0, traces.shape[1])
    cs = np.concatenate([np.zeros((traces.shape[0], 1)), np.cumsum(traces, axis=1)],
                        axis=1)          # <- recomputed for every trial
    return cs[:, idx[1:]] - cs[:, idx[:-1]]
```

iii. The AI profiled the loading cost up front (step 16: "NWB direct loading works and is fast") and profiled the binning after the first test run (step 39: "Neural binning loop is slow (per-cell per-trial); I'll vectorize with cumsum"), which cut it to ~13 s/session with raw `events`. It did not identify the remaining redundant cumsum, and switching to `filtered_events` then pushed the per-session time back up to ~1 minute.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized across cells (`bin_events` → `bin_events_matrix`). What remains:
- The `for k, ct in enumerate(change_times)` loop over trials in the neural section — the cumulative sum should be hoisted out of the loop and all trials' edges converted to indices in one `np.searchsorted` call, after which the whole `(n_cells, n_trials, T)` matrix is a single fancy-index difference. This is the one loop where vectorization would matter materially.
- The second `for k, ct in enumerate(change_times)` loop that builds the outputs: `np.searchsorted` for image identity and image change could be run once on the flattened `(n_trials × T)` grid of bin centres; `bin_mean` for running and pupil could likewise be applied to all trials' edges at once.
- The per-experiment `np.stack([np.asarray(e) for e in ev[EVENTS_COL].values])` list comprehension, and the `[image_names.index(n) for n in flash.image_name.values]` lookup (a `pd.Categorical`/dict map would be faster).

ii.
```python
        for k, ct in enumerate(change_times):
            mat[:, k, :] = bin_events_matrix(traces, tss, ct + edges_rel)
```

```python
    for k, ct in enumerate(change_times):
        te = ct + edges_rel
        tc = ct + centers_rel
        j = np.searchsorted(flash_start, tc, side='right') - 1
        ...
        r = fill_nans(bin_mean(run_v, run_t, te))
        p = fill_nans(bin_mean(pupil_v, pupil_t, te))
```

iii. The AI addressed only the inner (per-cell) loop, motivated purely by runtime after the first test run. It never revisited the trial-level loops; they are cheap for the behavioural streams but expensive for the neural stream because of the nested cumsum.

## 9-c. What processing does the code repeat multiple times?

i. The clear one: **the full-session `np.cumsum` of the event matrix is recomputed once per trial** (and a fresh `np.concatenate` allocating an `n_cells × (n_frames+1)` array with it), i.e. ~250× per plane instead of once. Hoisting it out of the trial loop would give essentially all of the neural-binning time back.

Smaller repetitions: `bin_edges()` is recomputed for every session; `data['subjects'].index(res['mouse'])` does a linear scan per session; `fill_nans` is applied twice to the pupil signal (once over the session trace, once per trial after binning); the behaviour/stimulus/trial tables are read from `exps[0]` while the other planes of a Multiscope session carry identical copies that are loaded and discarded.

ii.
```python
    cs = np.concatenate([np.zeros((traces.shape[0], 1)), np.cumsum(traces, axis=1)],
                        axis=1)
```
called from
```python
        for k, ct in enumerate(change_times):
            mat[:, k, :] = bin_events_matrix(traces, tss, ct + edges_rel)
```

iii. Not discussed. The AI introduced `bin_events_matrix` specifically as a speed-up over the per-cell version and evidently did not notice that the vectorized helper had moved a session-scale computation inside the trial loop.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Modest amounts:
- **Neural, image, change, running and pupil are computed for every candidate trial, then some trials are dropped** by the `keep` filter (outcome `-1`) — that work is thrown away. Likewise a session can be abandoned (`return None`) after all of its planes' events have been loaded, binned, and normalized.
- **Whole-session products are computed where only ~1/3 of the session is used**: `filtered_events` is materialised (and convolved) for the entire recording, `fill_nans` interpolates the whole session-long pupil trace, and the full `cumsum` covers all frames, but only the 60 bins × ~250 trials inside the `[-2, +4]` s windows are kept.
- `inputs.append(np.zeros((0, T), dtype=np.float32))` allocates an empty array per trial for a decoder that has no inputs; a shared array would do.
- `res['info']` (per-session metadata, including the full experiment-id lists) is assembled for all 91 sessions and stored in `metadata['session_info']`; it is informative but unused by the decoder.
- `neural_all` is built as a dense `(n_cells, n_trials, 60)` array and then re-sliced trial-by-trial with `np.ascontiguousarray`, copying the same data a second time.

ii.
```python
    keep = [k for k in range(ntrials) if out_trials[k] >= 0]
    if len(keep) < 2:
        return None
```

```python
        inputs.append(np.zeros((0, T), dtype=np.float32))
```

```python
    for n_, k in enumerate(keep):
        neural.append(np.ascontiguousarray(neural_all[:, k, :], dtype=np.float32))
```

iii. Not discussed in the trajectory. None of this affects correctness; it is a consequence of computing everything per session first and filtering afterwards, which keeps the code simple.
