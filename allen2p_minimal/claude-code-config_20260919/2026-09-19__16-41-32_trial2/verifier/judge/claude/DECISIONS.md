# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the SDK's S3/project cache. It reads the local release directory
`/app/data/visual-behavior-ophys-1.1.0` directly: the flat metadata table
`project_metadata/ophys_experiment_table.csv` is used to enumerate experiments, and the set of
experiment ids is intersected with the `.nwb` files actually present in
`behavior_ophys_experiments/` (284 files). The table is then filtered to
`project_code == 'VisualBehavior'` (the single-plane 2P rig), `passive == False` (active behavior
sessions), and `experience_level == 'Familiar'`, leaving 88 candidate experiments from 37 mice.
Each selected experiment is loaded from its NWB file with
`BehaviorOphysExperiment.from_nwb_path(...)`, in a 22-process `multiprocessing.Pool`, with an
on-disk per-experiment pickle cache so the slow NWB reads happen once. From each experiment the AI
pulls `ophys_timestamps`, `events` (neural), `running_speed`, `eye_tracking`,
`stimulus_presentations` and `trials`. One session (806456687) is dropped for having no eye
tracking, giving the final 87 sessions / 21,756 trials / 14,437 neurons / 37 mice.

ii.
```python
def select_experiments():
    """Experiment table rows for the sessions we convert (see module docstring)."""
    available = sorted(int(re.search(r'(\d+)\.nwb', f).group(1))
                       for f in os.listdir(NWB_DIR) if f.endswith('.nwb'))
    tbl = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
    sel = tbl[tbl.ophys_experiment_id.isin(available)
              & (tbl.project_code == 'VisualBehavior')
              & (~tbl.passive)
              & (tbl.experience_level == 'Familiar')]
    return sel.sort_values('ophys_experiment_id').reset_index(drop=True)
```
```python
    oeid = int(row['ophys_experiment_id'])
    path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{oeid}.nwb')
    expt = BehaviorOphysExperiment.from_nwb_path(path)
    ts = np.asarray(expt.ophys_timestamps, dtype=float)       # time base for everything
```
```python
    rows = [(i, r) for i, r in expts.iterrows()]
    if workers > 1:
        with Pool(min(workers, len(rows))) as pool:
            results = pool.map(extract_cached, rows, chunksize=1)
```

iii. From the module docstring and the final report: `VisualBehavior` (single plane) was chosen
because "Single-plane sessions contain exactly one imaging plane ("experiment") per session, so a
session has one well defined population of simultaneously recorded neurons, and every session is
sampled at the same 31 Hz ophys frame rate (required because the target format asks for one common
time-bin size across all trials/sessions)". `VisualBehaviorMultiscope` — the rig the reference paper
actually used — was excluded because in this release it is "11 Hz, has up to 8 planes per session,
and ... contains a single mouse" (the AI verified: 34 experiments, 6 sessions, 1 Sst mouse), so
"following it literally here would leave 1 mouse". Passive sessions were excluded because "the lick
spout [is] retracted, so 'trial outcome' (hit/miss/false alarm/correct reject) is undefined".
`Familiar` follows the paper: "we restricted our analysis to familiar stimuli", "For neural analysis
we used neurons recorded during familiar image set presentations"; the AI adds that it also
guarantees every session used the same 8 images of set A so the image-identity category set is
consistent. Intersecting with the on-disk NWB list was done because the metadata table lists more
experiments than the release directory contains.

## 1-b. How are the data split into subjects?

i. Subjects are the `mouse_id` values of the selected experiment-table rows. A subject list is built
in first-appearance order (experiments are sorted by `ophys_experiment_id`), and each session gets
`subject_idx` = index of its mouse in that list. 37 mice result, with 1–7 sessions each.

ii.
```python
        mouse = str(row['mouse_id'])
        if mouse not in subject_list:
            subject_list.append(mouse)
        subject_idx.append(subject_list.index(mouse))
```

iii. Not discussed explicitly beyond the survey work; `mouse_id` is the SDK's canonical animal
identifier in the experiment table, and the AI cross-checked the resulting counts
(37 mice / 88 familiar-active VisualBehavior experiments) against the metadata tables before
writing the converter.

## 1-c. How are the data split into sessions?

i. One session == one `ophys_experiment_id` (one NWB file). No grouping by `ophys_session_id` is
performed, because for the single-plane `VisualBehavior` project each session contains exactly one
imaging plane; the AI verified this from the metadata (168 active VisualBehavior experiments = 168
distinct `ophys_session_id`s). Sessions are ordered by `ophys_experiment_id`. Both
`ophys_experiment_id` and `ophys_session_id` (plus behavior/container ids, cre line, session type,
depth, date) are recorded per session in `metadata['session_info']`.

ii.
```python
    return sel.sort_values('ophys_experiment_id').reset_index(drop=True)
```
```python
        session_info.append({
            'ophys_experiment_id': int(row['ophys_experiment_id']),
            'ophys_session_id': int(row['ophys_session_id']),
            'behavior_session_id': int(row['behavior_session_id']),
            'ophys_container_id': int(row['ophys_container_id']),
            ...
```

iii. Module docstring: "Single-plane sessions contain exactly one imaging plane ("experiment") per
session, so a session has one well defined population of simultaneously recorded neurons". This is
also the definition given in `methods.txt` ("For single-plane imaging experiments, there is only one
imaging plane (referred to as an experiment) per session"), so experiment-level and session-level
splitting coincide here.

## 1-d. How are the data split into trials?

i. Trials come from the SDK's own `trials` table. The kept set is `(go | catch) & ~aborted &
~auto_rewarded`. Each trial spans the ophys frames whose timestamps fall in
`[trials.start_time, trials.stop_time)`, found with `np.searchsorted(..., side='left')` on
`ophys_timestamps`. Trials are therefore variable length (217–389 frames, mean 261 ≈ 8.4 s), because
change time is drawn from a truncated exponential. All streams (neural, running, pupil, image
identity, change) are sliced with the same `[i0:i1)` index range.

ii.
```python
    trials = expt.trials
    keep = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
    trials = trials[keep]

    for tid, tr in trials.iterrows():
        i0 = np.searchsorted(ts, tr['start_time'], side='left')
        i1 = np.searchsorted(ts, tr['stop_time'], side='left')
        if i1 - i0 < 2:
            continue
        ...
        out['neural'].append(np.ascontiguousarray(traces[:, i0:i1]))
        out['running'].append(running[i0:i1])
        out['pupil'].append(pupil[i0:i1])
        out['image_id'].append(image_id[i0:i1])
        out['change'].append(change[i0:i1])
```

iii. Module docstring: "the experiment's own trial definition (SDK `trials` table). 'Go' and 'catch'
trials are kept; 'aborted' (mouse licked before the change) and 'auto-rewarded' (free reward) trials
are excluded, as instructed. Each trial spans [trials.start_time, trials.stop_time)". The AI first
checked on one session that `go`/`catch` and `aborted`/`auto_rewarded` do not overlap ("n go/catch
324 aborted/auto overlap 0"), and in the final report notes the full trial window keeps the
pre-change flashes plus the response window, which is what makes the time-varying outputs
meaningful, and that trials are variable length "by design, since change time is drawn from a
truncated exponential" (`off_start = 0.0`, `off_end = None`).

## 1-e. How are trials filtered based on quality controls?

i. Four filters: (1) trial type — only `go|catch`, never `aborted` or `auto_rewarded`; (2) a trial
must span at least 2 ophys frames (`i1 - i0 >= 2`); (3) exactly one of
`hit/miss/false_alarm/correct_reject` must be True, otherwise the trial is dropped as "not a
scoreable go/catch trial"; (4) session level — a session is dropped if it yields fewer than 2 usable
trials (and also if it has no usable eye tracking). In practice filters (2) and (3) removed nothing:
22,027 go/catch trials existed in the 88 candidate sessions and 21,756 survived, the difference
being exactly the 271 trials of the one session dropped for missing eye tracking. No filtering on
neural activity, running/pupil quality, or behavioral performance is applied.

ii.
```python
        if i1 - i0 < 2:
            continue
        outcome = [tr['hit'], tr['miss'], tr['false_alarm'], tr['correct_reject']]
        if sum(bool(x) for x in outcome) != 1:
            continue                                   # not a scoreable go/catch trial
```
```python
    for res, (_, row) in zip(results, rows):
        if res.get('skip') or len(res.get('neural', [])) < 2:
            skipped.append((int(row['ophys_experiment_id']),
                            res.get('skip', 'fewer than 2 usable trials')))
            continue
        sessions.append(res)
```

iii. Trial-type exclusion is stated to be "as instructed". The ≥2-frame and single-outcome tests are
defensive guards for degenerate rows. The ≥2-trial session rule follows the format requirement
("There needs to be at least two trials within each session in order to evaluate the decoder
performance"). The AI explicitly decided *not* to filter trials on neural activity: "923/21,756
trials (4.2%) have all-zero activity — these come from the low-cell-count Sst/Vip sessions and are
genuine, so I kept them rather than filtering on neural activity."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `filtered_events` column of the SDK's `events` table (detected calcium events convolved with
a half-Gaussian), one row per cell, sampled on `ophys_timestamps`. Cell identity is taken from
`events.index` (cell specimen ids) and stored in `metadata['session_info'][i]['cell_specimen_ids']`.
dF/F is *not* used.

ii.
```python
    # ---- neural: detected calcium events, (n_neurons, n_frames) -------------------
    events = expt.events
    cell_ids = list(events.index)
    traces = np.stack([np.asarray(x, dtype=np.float32)
                       for x in events['filtered_events'].values])
    assert traces.shape == (len(cell_ids), nframes), (traces.shape, nframes)
```

iii. The paper is explicit: "For all analysis of neural data we used the detected calcium events ...
thus removing the slow decay dynamics of the calcium indicator", which the AI quotes in the module
docstring. In the final report it states it measured the alternative: "dF/F decodes better (I
measured ~+0.09 balanced accuracy on a 3-session test) but contradicts the paper's stated
processing, so I kept events."

## 2-b. How is the `neural` data processed?

i. Essentially none beyond selecting the `filtered_events` variant: the per-cell arrays are stacked
into an `(n_neurons, n_frames)` float32 matrix and sliced per trial (`np.ascontiguousarray`). No
z-scoring, smoothing, baseline subtraction, spike deconvolution, or cross-session normalisation is
applied, and no merging across planes is needed (one plane per session). The choice of
`filtered_events` over the raw `events` train is the one processing decision: the AI measured on two
sessions that raw events are nonzero on only ~0.1–0.12% of 31 Hz frames while filtered events are
nonzero on ~1.7%.

ii.
```python
    traces = np.stack([np.asarray(x, dtype=np.float32)
                       for x in events['filtered_events'].values])
```
```python
            x = s['neural'][k]
            T = x.shape[1]
            ...
            n_sess.append(x)
```

iii. Module docstring: "We use the SDK's `filtered_events` (events convolved with a half-Gaussian):
the raw event train is nonzero on only ~0.1% of 31 Hz frames, which carries essentially no
information at single-timepoint resolution, whereas the filtered trace preserves event times and
magnitudes in a form a per-timepoint decoder can use." This is backed by the measurements made in
the exploration steps.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level or session-level neural QC is applied. Every ROI in the released NWB `events`
table is kept (7–666 neurons per session, 14,437 total). Sessions with very few neurons are kept,
and trials whose neural matrix is entirely zero (923 of 21,756, from the sparse Sst/Vip sessions)
are kept.

ii. No filtering code exists; the closest thing is the shape assertion:
```python
    assert traces.shape == (len(cell_ids), nframes), (traces.shape, nframes)
```

iii. Module docstring: "All ROIs released in the NWB files already passed the pipeline's ROI
filtering / QC (valid_roi is True for every cell), so no further neuron curation." The AI verified
this in its survey (`ncells == nvalid` for all 88 experiments). On the all-zero trials it reasoned
they "come from the low-cell-count Sst/Vip sessions and are genuine, so I kept them rather than
filtering on neural activity."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to trial start on the ophys clock. `ophys_timestamps` is the master time base;
`np.searchsorted(ts, trials.start_time, 'left')` and `np.searchsorted(ts, trials.stop_time, 'left')`
give the frame index range, and every stream is sliced with that same range, so column 0 of each
trial is the first ophys frame at or after `start_time`. Metadata records
`temporal_alignment_event = 'trial start (trials.start_time in the AllenSDK trials table); each time
bin is one 2-photon imaging frame (ophys_timestamps), and the stimulus, running-speed and pupil
streams are resampled onto those frame times'`, with `off_start = 0.0` and `off_end = None`.

ii.
```python
    ts = np.asarray(expt.ophys_timestamps, dtype=float)       # time base for everything
...
        i0 = np.searchsorted(ts, tr['start_time'], side='left')
        i1 = np.searchsorted(ts, tr['stop_time'], side='left')
```

iii. Module docstring: "the ophys frame times (`ophys_timestamps`) are the time base: one time bin
per 2-photon frame (~32.3 ms). Stimulus, running and pupil streams are aligned onto those frame
times." This follows the instruction "Temporally align based on ophys timestamp". The AI notes
`off_end` is `None` because the trial length is variable by design.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One bin per two-photon frame; no rebinning, resampling or smoothing of the neural data. The AI's
survey established that all 88 single-plane sessions run at 31 Hz with per-session median frame
interval 32.31–32.33 ms, and `metadata['time_bin_size']` is the mean of the per-session medians,
32.319 ms (`ophys_frame_rate_hz` is stored as its reciprocal; each session's own
`ophys_frame_interval_s` is also kept in `session_info`).

ii.
```python
    out['dt'] = float(np.median(np.diff(ts)))
...
    dt_ms = float(np.mean([s['dt'] for s in sessions]) * 1000.0)
...
            'time_bin_size': dt_ms,
            'ophys_frame_rate_hz': 1000.0 / dt_ms,
```

iii. The frame clock is the acquisition clock, so no rebinning is needed; the restriction to the
single-plane rig was made precisely so that one common bin size is valid — "every session is sampled
at the same 31 Hz ophys frame rate (required because the target format asks for one common time-bin
size across all trials/sessions)".

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From `stimulus_presentations`, restricted to the active change-detection block
(`active == True`) and to actually-shown flashes (`~omitted`), using the columns `image_name`,
`start_time` and `end_time`. The trials table's `initial_image_name` / `change_image_name` are not
used.

ii.
```python
    stim = expt.stimulus_presentations
    stim = stim[stim['active'].astype(bool)]                  # change-detection block only
    shown = stim[~stim['omitted'].astype(bool)]
    image_names = sorted(set(shown['image_name'].dropna()))
```

iii. Using the stimulus table gives the literal frame-by-frame screen content, including the 500 ms
grey inter-stimulus intervals and the 5% omitted flashes; the AI states the task in metadata as
"the identity of the image on the screen (8 natural images of image set A, or gray screen)". The
`active` filter excludes the natural-movie and 5-minute grey-screen blocks the AI found in the
stimulus table ("blocks {'natural_movie_one': 9000, 'change_detection_behavior': 4800, ...}").

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A per-frame integer code is built over the whole session: 0 means "gray screen" (inter-stimulus
interval and omitted flashes) and 1..8 are the images, assigned by writing each flash's code into
the frames spanned by `[start_time, end_time)`. Per-session codes (1..n images present in that
session) are then remapped onto a global alphabetical ordering of images pooled over all sessions so
a code means the same image everywhere; `output_values[0] = ['gray_screen'] + image_names`. All 87
sessions turned out to contain the same 8 images of set A, so the remap is the identity.

ii.
```python
    img_lookup = {name: i + 1 for i, name in enumerate(image_names)}
    image_id = np.zeros(nframes, dtype=np.int8)
    starts = np.searchsorted(ts, shown['start_time'].to_numpy(dtype=float), side='left')
    stops = np.searchsorted(ts, shown['end_time'].to_numpy(dtype=float), side='left')
    codes = shown['image_name'].map(img_lookup).to_numpy(dtype=np.int8)
    for i0, i1, c in zip(starts, stops, codes):
        image_id[i0:i1] = c
```
```python
    for s in sessions:
        remap = np.zeros(len(s['image_names']) + 1, dtype=np.int8)
        for i, name in enumerate(s['image_names']):
            remap[i + 1] = image_names.index(name) + 1
        if not np.array_equal(remap, np.arange(len(remap), dtype=np.int8)):
            s['image_id'] = [remap[a] for a in s['image_id']]
```

iii. Code comments: "0 is reserved for 'gray screen' (inter-stimulus interval and omitted flashes)"
and "per-session image codes are 1..n_images_in_that_session; remap them onto the global ordering
(0 stays 'gray screen') so a category means the same image everywhere". The AI also noted that
restricting to Familiar makes the image set identical across sessions. The resulting distribution
(gray 0.669, each image ≈0.041) was inspected in the verification output, and the per-frame result
was printed for one trial to confirm the alternation image/gray/image.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The image code array is built on the session's ophys frame grid (`np.searchsorted` of flash
start/end times into `ophys_timestamps`) and then sliced with exactly the same `[i0:i1)` trial index
range as the neural matrix, so it is per-frame aligned by construction.

ii.
```python
    starts = np.searchsorted(ts, shown['start_time'].to_numpy(dtype=float), side='left')
    stops = np.searchsorted(ts, shown['end_time'].to_numpy(dtype=float), side='left')
...
        out['neural'].append(np.ascontiguousarray(traces[:, i0:i1]))
        out['image_id'].append(image_id[i0:i1])
```

iii. All streams share the ophys timebase by design ("the stimulus, running-speed and pupil streams
are resampled onto those frame times"); the whitepaper's sync section (all clocks recorded on one
100 kHz IO board) justifies treating the stimulus times as being on the same clock. The AI verified
alignment by printing frame-by-frame image identity around a change for one trial.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From the `is_change` flag of `stimulus_presentations` (active, non-omitted rows), together with
the same flash `start_time` / `end_time` used for image identity. It is *not* derived from
`trials.change_time`.

ii.
```python
    change = np.zeros(nframes, dtype=np.int8)
    is_chg = shown['is_change'].astype(bool).to_numpy()
    for i0, i1 in zip(starts[is_chg], stops[is_chg]):
        change[i0:i1] = 1
```

iii. Code comment: "image changes: 1 while the flash whose identity differs from the preceding one
is on the screen, i.e. over exactly the same frames that carry that image in `image_id`. Sham
changes on catch trials are *not* changes in image identity and stay 0." Using the stimulus table's
`is_change` automatically gives exactly the real identity changes (the AI checked one session:
"is_change rows in active 288 sham 41").

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary per-frame indicator, set to 1 over the ~250 ms (≈8 frames) presentation of the changed
image and 0 everywhere else, including the grey period after it and including the sham change of
catch trials. The final prevalence is 2.6% of timepoints. An earlier version marked the full 750 ms
image-presentation interval (change flash + following grey, 8.0% of timepoints), and this was
changed after an explicit A/B test.

ii.
```python
    change = np.zeros(nframes, dtype=np.int8)
    is_chg = shown['is_change'].astype(bool).to_numpy()
    for i0, i1 in zip(starts[is_chg], stops[is_chg]):
        change[i0:i1] = 1
```
(previous version, replaced:)
```python
    chg_times = shown.loc[shown['is_change'].astype(bool), 'start_time'].to_numpy(dtype=float)
    for c0 in chg_times:
        change[np.searchsorted(ts, c0, side='left'):
               np.searchsorted(ts, c0 + PRESENTATION_INTERVAL, side='left')] = 1
```

iii. The 750 ms version was originally chosen from the paper ("By image presentation interval we
refer to the 750 ms interval beginning with each image presentation"). The AI then built 15-session
datasets with both definitions and trained the reference decoder on each: image-change validation
balanced accuracy was 0.5585 (750 ms) vs 0.6368 (250 ms). Its conclusion: "The 250 ms window
(marking the changed image's own presentation) decodes markedly better and matches 'right after a
change in image identity' more literally."

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is natively binary — no thresholding of a continuous quantity. Values are 0/1 with
`output_values[1] = ['no_change', 'change']`; catch (sham) changes are 0, real changes are 1.

ii.
```python
            out[1] = s['change'][k]
...
        'output_values': [
            ['gray_screen'] + list(image_names),
            ['no_change', 'change'],
```

iii. Metadata: "binary, 1 on the ophys frames during which the changed image is on the screen (the
~250 ms flash whose identity differs from the preceding flash), 0 otherwise; sham changes on catch
trials are 0 because the image identity does not change".

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same mechanism as image identity: built on the ophys frame grid from flash start/end times and
sliced with the trial's `[i0:i1)` frame range, so it is aligned frame-for-frame with the neural
matrix, and it is 1 on exactly the frames on which `image_id` holds the new image.

ii.
```python
        out['change'].append(change[i0:i1])
        out['neural'].append(np.ascontiguousarray(traces[:, i0:i1]))
```

iii. Same justification as 3-c (one shared ophys time base). The AI printed a trial's frame-by-frame
image identity and change flag to confirm the change flag turns on at the frame where the new image
first appears (t = +0.001 s relative to the change).

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `expt.running_speed`, using its `speed` (cm/s, the SDK's wrap-corrected, transient-removed,
10 Hz low-pass filtered signal) and `timestamps` columns.

ii.
```python
    run = expt.running_speed
    running = np.interp(ts, run['timestamps'].to_numpy(dtype=float),
                        run['speed'].to_numpy(dtype=float)).astype(np.float32)
```

iii. Not discussed at length; `running_speed` is the SDK's standard, already-processed locomotion
signal described in `methods.txt` ("Both the unfiltered ... and the filtered running speeds are
available to end users as, respectively, the `running_speed_raw` and `running_speed` attributes").
The AI's survey confirmed that this stream has no NaNs in any of the 88 sessions (`run_nan` = 0).

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Two steps: (1) linear resampling onto the ophys frame times with `np.interp` (which clamps to the
first/last value outside the recorded range, so no NaNs are produced); (2) discretisation into 5
equal-count bins whose edges are the 20/40/60/80th percentiles of the pooled distribution of every
timepoint that ends up in the dataset (edges: 0.05, 4.72, 22.21, 35.65 cm/s). No smoothing,
rectification or per-session normalisation.

ii.
```python
def quantile_bins(values, nbins):
    """Edges cutting `values` into `nbins` equal-count bins (interior edges only)."""
    qs = np.linspace(0, 100, nbins + 1)[1:-1]
    return np.percentile(values, qs)
...
    all_run = np.concatenate([np.concatenate(s['running']) for s in sessions])
    run_edges = quantile_bins(all_run, NQUANTILES)
...
            out[2] = np.digitize(s['running'][k], run_edges)
```

iii. Code comment: "Bins are computed on the pooled distribution over every timepoint that ends up
in the dataset, so each bin holds ~20% of the data and a bin index means the same thing in every
session." This implements the instruction "Running speed, discretized into five equal percentile
bins"; the verification output confirms each bin holds exactly 0.200 of the data.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize` against the four interior percentile edges, giving labels 0–4. Bin names carry the
edges and units, e.g. `running_speed_q1 [-inf,0.05) cm/s`; the edges are also stored in
`metadata['running_speed_bin_edges_cm_per_s']`.

ii.
```python
            out[2] = np.digitize(s['running'][k], run_edges)
...
def bin_names(label, edges, unit):
    lo = np.concatenate([[-np.inf], edges])
    hi = np.concatenate([edges, [np.inf]])
    ...
        names.append(f'{label}_q{i + 1} [{a_s},{b_s}) {unit}')
```

iii. Global (not per-session) quantiles were chosen so that "a bin index means the same thing in
every session"; storing the numeric edges in metadata keeps the mapping interpretable.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The speed trace is interpolated onto `ophys_timestamps` for the whole session before trial
segmentation, then sliced with the same `[i0:i1)` trial frame range as the neural matrix.

ii.
```python
    running = np.interp(ts, run['timestamps'].to_numpy(dtype=float),
                        run['speed'].to_numpy(dtype=float)).astype(np.float32)
...
        out['running'].append(running[i0:i1])
```

iii. Same as 2-d/3-c: everything is put on the ophys frame clock first, which guarantees alignment;
the running encoder is sampled at ~60 Hz (higher than the 31 Hz ophys rate) and all clocks are
hardware-synced, so linear interpolation is appropriate.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `expt.eye_tracking`, columns `pupil_area`, `likely_blink` and `timestamps`. Diameter is computed
from the fitted pupil *area* as the equivalent-circle diameter, `2*sqrt(area/pi)`, in pixels.

ii.
```python
    area = eye_tracking['pupil_area'].to_numpy(dtype=float)
    area[eye_tracking['likely_blink'].to_numpy(dtype=bool)] = np.nan
    good = np.isfinite(area)
    ...
    diam = 2.0 * np.sqrt(area[good] / np.pi)
    return t[good], diam
```

iii. Docstring: "Pupil diameter in pixels, from the SDK's fitted pupil area. `pupil_area` is already
NaN on frames flagged `likely_blink`". The AI surveyed the eye data first and found one session with
no eye tracking at all and a median of ~3% blink/NaN frames elsewhere (max 29.6%).

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames and non-finite samples are dropped, the remaining samples are converted to
equivalent-circle diameter, linearly interpolated onto the ophys frame times (gaps bridged, edges
held constant, so no NaNs remain), then discretised into 5 pooled equal-count bins exactly as
running speed (edges: 75.74, 84.84, 94.69, 110.53 px). A session with fewer than 50% usable eye
frames — or none at all — is dropped entirely rather than imputed.

ii.
```python
    good = np.isfinite(area)
    if good.sum() < 0.5 * len(area):
        # less than half the session is usable pupil data
        return None, None
...
    et, ed = pupil_diameter(eye)
    if et is None:
        return {'oeid': oeid, 'skip': 'no eye tracking'}
    pupil = np.interp(ts, et, ed).astype(np.float32)
...
    all_pupil = np.concatenate([np.concatenate(s['pupil']) for s in sessions])
    pupil_edges = quantile_bins(all_pupil, NQUANTILES)
...
            out[3] = np.digitize(s['pupil'][k], pupil_edges)
```

iii. Blink removal before interpolation avoids propagating blink artefacts; interpolating rather
than leaving NaN is needed because the output must be a valid categorical label at every frame. The
session-level drop is justified in the module docstring: "sessions without eye-tracking data are
dropped, since pupil diameter is a required decoder output."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical scheme to running speed: `np.digitize` against four global interior percentile edges →
labels 0–4, each holding 20% of the data, with human-readable bin names and the edges saved in
`metadata['pupil_diameter_bin_edges_pix']`.

ii.
```python
    pupil_edges = quantile_bins(all_pupil, NQUANTILES)
...
            out[3] = np.digitize(s['pupil'][k], pupil_edges)
...
            bin_names('pupil_diameter', pupil_edges, 'pix'),
```

iii. Same as 5-c: pooling across sessions keeps the meaning of a bin constant, as required by the
instruction "Pupil diameter, discretized into five equal percentile bins".

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated onto `ophys_timestamps` for the whole session before segmentation, then sliced with
the trial's `[i0:i1)` frame range — the same index array as the neural matrix.

ii.
```python
    pupil = np.interp(ts, et, ed).astype(np.float32)
...
        out['pupil'].append(pupil[i0:i1])
```

iii. Same as running speed: the eye camera (30 or 60 Hz, per the AI's survey) is hardware-synced to
the ophys clock, so resampling onto the frame times is valid and guarantees alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four boolean columns of the `trials` table: `hit`, `miss`, `false_alarm`, `correct_reject`.

ii.
```python
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
        outcome = [tr['hit'], tr['miss'], tr['false_alarm'], tr['correct_reject']]
        if sum(bool(x) for x in outcome) != 1:
            continue                                   # not a scoreable go/catch trial
        out['outcome'].append(int(np.argmax(outcome)))
```

iii. These are the SDK's canonical change-detection outcome labels (described in `methods.txt`:
"this trial structure leads to a sampling of 'GO' and 'CATCH' trials, that when combined with mouse
responding, yields 'HIT', 'MISS', 'FALSE ALARM', and 'CORRECT REJECTION' trials"). Requiring exactly
one True guarantees a well-defined single label.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Mapped to an integer 0–3 via `np.argmax` over the four booleans in the fixed order
`['hit','miss','false_alarm','correct_reject']`, then broadcast across all time bins of the trial so
that it is stored as a constant time series (row 4 of the `(5, T)` output matrix). Resulting
distribution: hit 0.299, miss 0.576, false alarm 0.021, correct reject 0.104.

ii.
```python
        out['outcome'].append(int(np.argmax(outcome)))
...
            out[4] = s['outcome'][k]
...
        'output_names': ['image_identity', 'image_change', 'running_speed_bin',
                         'pupil_diameter_bin', 'trial_outcome'],
        'output_values': [..., list(OUTCOMES)],
```

iii. The instruction lists trial outcome as "Static per-trial", but the format spec says "If at all
possible, make it time-varying", and all five outputs must share one `(n_output, T)` matrix, so the
static label is tiled over the trial. The AI describes it in the final report as "trial outcome
(hit/miss/FA/CR) tiled across the trial".

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases: (1) experiment ids listed in the metadata table but absent from the release are
dropped by intersecting with the on-disk NWB files; (2) a session with no eye tracking, or with
<50% usable pupil frames, is skipped entirely and recorded in `metadata['excluded_experiments']`
(one session, 806456687, 271 trials); (3) blink/NaN pupil frames are removed and interpolated over,
with edge values held constant, so no NaN reaches the discretiser; (4) running speed is likewise
interpolated (the AI verified there are no NaNs in the source); (5) omitted flashes (5% of
presentations) are labelled as grey screen rather than dropped; (6) degenerate trials (<2 frames, or
not exactly one outcome flag) are skipped; (7) sessions with <2 usable trials are skipped; (8) a
corrupted per-session cache file is silently re-extracted; (9) an assertion checks that the events
matrix has one column per ophys timestamp. Not handled: there is no try/except around the per-session
extraction itself, so an unreadable NWB file would abort the whole run; and trials whose neural data
is entirely zero are deliberately kept.

ii.
```python
    available = sorted(int(re.search(r'(\d+)\.nwb', f).group(1))
                       for f in os.listdir(NWB_DIR) if f.endswith('.nwb'))
```
```python
    try:
        eye = expt.eye_tracking
    except Exception:
        eye = None
    et, ed = pupil_diameter(eye)
    if et is None:
        return {'oeid': oeid, 'skip': 'no eye tracking'}
```
```python
        if res.get('skip') or len(res.get('neural', [])) < 2:
            skipped.append((int(row['ophys_experiment_id']),
                            res.get('skip', 'fewer than 2 usable trials')))
            continue
    for oeid, why in skipped:
        print(f'  skipping experiment {oeid}: {why}', flush=True)
```
```python
            'excluded_experiments': skipped,
```

iii. The AI's stance is to drop rather than impute anything that would make a required output
undefined ("sessions without eye-tracking data are dropped, since pupil diameter is a required
decoder output"), to interpolate only short within-session gaps (blinks), and to keep a record of
everything excluded in the metadata. It quantified the cost of the one dropped session
(271 of 22,027 trials) before accepting it.

## 9-a. What are the most time-consuming steps of the code?

i. Reading the 88 NWB files through `BehaviorOphysExperiment.from_nwb_path` — each file carries
full-session traces for up to 666 cells × ~140,000 frames plus behaviour tables — dominates; it is
pure I/O + HDF5 decode. Second is serialisation: the per-session cache pickles plus the final 4.11 GB
`converted_data.pkl`. The AI attacked the first cost with a 22-process pool and an on-disk cache, so
the end-to-end run took ~31 s wall (~5.5 min CPU) on a warm cache/page cache; the initial survey pass
over all sessions was the genuinely slow step.

ii.
```python
    if workers > 1:
        with Pool(min(workers, len(rows))) as pool:
            results = pool.map(extract_cached, rows, chunksize=1)
```
```python
def extract_cached(args):
    """extract_session + an on-disk cache, so the (slow) NWB reads happen only once."""
    idx, row = args
    cache = os.path.join(CACHE_DIR, f"{int(row['ophys_experiment_id'])}.pkl")
    if os.path.exists(cache):
        try:
            with open(cache, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
```

iii. The docstring of `extract_cached` states the reasoning explicitly ("so the (slow) NWB reads
happen only once"); the AI checked the machine's core count and free memory before choosing 16–22
workers, and estimated the output size (~3.9 GB) from its survey before running the conversion.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Three Python-level loops remain, none of them on the critical path:
- the per-flash loops that paint `image_id` and `change` (~4,800 flashes per session, two passes);
  these could be done with `np.repeat`/`np.add.reduceat` or an interval-index + `np.searchsorted`
  labelling in one vectorised step;
- the per-trial loop in `extract_session` (`for tid, tr in trials.iterrows()`), including the
  `iterrows()` overhead and the per-trial `np.searchsorted` calls, which could be done as a single
  vectorised `searchsorted` over all start/stop times;
- the assembly loop in `main`, where `np.digitize` is called separately per trial for running and
  pupil (it could be applied once per session), and `image_names.index(name)` is an O(n) lookup
  inside a loop.

ii.
```python
    for i0, i1, c in zip(starts, stops, codes):
        image_id[i0:i1] = c
...
    for i0, i1 in zip(starts[is_chg], stops[is_chg]):
        change[i0:i1] = 1
...
    for tid, tr in trials.iterrows():
        i0 = np.searchsorted(ts, tr['start_time'], side='left')
        i1 = np.searchsorted(ts, tr['stop_time'], side='left')
...
        for k in range(len(s['neural'])):
            ...
            out[2] = np.digitize(s['running'][k], run_edges)
            out[3] = np.digitize(s['pupil'][k], pupil_edges)
```

iii. Not discussed by the AI. Objectively these loops are negligible relative to NWB I/O
(~5,000 iterations per session against ~10^8 floats read), and the AI put its optimisation effort
into parallelism and caching instead, which is where the time actually is.

## 9-c. What processing does the code repeat multiple times?

i. Little algorithmic recomputation, but a lot of data movement is repeated:
- every session's trial arrays are pickled to `/app/cache_sessions/<oeid>.pkl` and then read back
  and pickled again into the final file, so the ~4 GB payload is serialised twice and copied at
  least three times (slice → cache → final dict);
- the pooled running and pupil vectors are materialised by concatenating every trial again
  (`np.concatenate` over all sessions) purely to compute 8 percentile values;
- `np.searchsorted` on the flash start/end times is effectively done twice (once for `image_id`,
  reused for `change` — this one *is* shared) while the per-trial searchsorted is re-done per trial;
- `image_names.index(name)` re-scans the global list for each image in each session.
The repeated work is cheap in CPU terms; the duplicate serialisation is the real cost (it also
doubles peak disk usage, which is why the AI deleted the cache directory at the end).

ii.
```python
    res = extract_session(row)
    tmp = cache + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(res, f, protocol=4)
    os.replace(tmp, cache)
```
```python
    all_run = np.concatenate([np.concatenate(s['running']) for s in sessions])
    all_pupil = np.concatenate([np.concatenate(s['pupil']) for s in sessions])
```

iii. The cache is a deliberate trade — it made the many re-runs during development (750 ms vs 250 ms
change window, dF/F vs events comparisons) nearly free, and the AI removed it (`rm -rf
/app/cache_sessions`) once the final pickle was written.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several things are computed or stored that the decoder never uses:
- `image_id`, `change`, `running` and `pupil` are computed for **all** ~140,000 ophys frames of each
  session, although only the ~55% of frames inside go/catch trial windows are kept (the natural-movie
  block and the two 5-minute grey-screen blocks are covered and then thrown away);
- `n_frames_session` and the full `cell_ids` list are carried through the cache;
- a large `session_info` block is stored per session (cell specimen ids, trial ids, container/behavior
  ids, genotype, equipment, date) — useful provenance, but nothing downstream reads it;
- human-readable `bin_names` strings are generated for the bin labels;
- an all-empty `(0, T)` float32 input array is allocated for each of the 21,756 trials even though
  `input_names` is empty;
- 923 trials with all-zero neural data are retained and stored at full size;
- the per-session cache pickles (~4 GB) are written and then deleted.
- `np.ascontiguousarray` copies each trial's neural slice (necessary for pickling compactly, but a
  copy nonetheless).

ii.
```python
    image_id = np.zeros(nframes, dtype=np.int8)     # whole session, incl. movie/grey blocks
...
            i_sess.append(np.zeros((0, T), dtype=np.float32))   # no decoder inputs
...
            'n_neurons': len(s['cell_ids']),
            'n_trials': len(s['neural']),
            'cell_specimen_ids': [int(c) for c in s['cell_ids']],
            'trial_ids': s['trial_ids'],
            'ophys_frame_interval_s': s['dt'],
```

iii. Not discussed as waste by the AI. Computing the behavioural/stimulus streams session-wide is a
simplifying choice that makes trial extraction a single slice per stream and keeps alignment
trivially correct; the extra metadata is provenance the instructions explicitly invite
(`'metadata': ... Add other relevant fields, e.g. session_info`), and the empty input arrays are
required by the target format ("No inputs for this task").
