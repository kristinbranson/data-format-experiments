# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the AllenSDK object API. It globs the local release tree for
`behavior_ophys_experiment_<id>.nwb` files (284 found), builds an id→path map, and reads the
project metadata CSV `ophys_experiment_table.csv` to get per-experiment metadata (mouse, session,
structure, depth, session type). All neural/behavioral/stimulus arrays are then read directly out
of the NWB HDF5 files with `h5py`, using the exact dataset paths that the SDK's `from_nwb`
readers use (`processing/ophys/event_detection/*`, `intervals/trials`, `processing/running/speed/*`,
`acquisition/EyeTracking/*`, `stimulus/presentation`, `stimulus/templates`).
No `project_code` filter is applied, so both `VisualBehavior` (239 single-plane experiments,
~31 Hz) and `VisualBehaviorMultiscope` (45 experiments from 1 mouse, ~11 Hz) are included.
Experiments are then curated (active only, eye tracking required) before loading.
Final loaded set: 199 experiments, 38 mice, 29,168 neurons, 51,075 trials.

ii.
```python
DATA_ROOT = Path('/app/data')

def find_files():
    files = sorted(glob.glob(str(DATA_ROOT / '**' / 'behavior_ophys_experiment_*.nwb'), recursive=True))
    return {int(re.search(r'(\d+)\.nwb$', f).group(1)): f for f in files}

def metadata_table(name):
    paths = glob.glob(str(DATA_ROOT / '**' / name), recursive=True)
    if len(paths) != 1:
        raise RuntimeError(f'Expected one {name}, found {paths}')
    return pd.read_csv(paths[0])

def select_experiments(files):
    meta = metadata_table('ophys_experiment_table.csv')
    meta = meta[meta.ophys_experiment_id.isin(files)].copy()
    ...
```
and inside `process_experiment`:
```python
with h5py.File(path, 'r') as h:
    ev  = np.asarray(h['processing/ophys/event_detection/data'], dtype=np.float32)
    ots = np.asarray(h['processing/ophys/event_detection/timestamps'], dtype=float)
    tr  = h['intervals/trials']
    run_t = np.asarray(h['processing/running/speed/timestamps'], float)
    eye_t = np.asarray(h['acquisition/EyeTracking/pupil_tracking/timestamps'], float)
```

iii. From CONVERSION_NOTES Step 6: "Full PyNWB/AllenSDK object construction would load unnecessary
images/templates and incur high overhead for 199 large NWBs", so it streams "one NWB at a time with
h5py and read only required datasets", explicitly "reproducing the SDK paths while avoiding costly
full PyNWB object construction" (Step 5, Key Decision 11). Step 10 Check 3 documents a step-by-step
comparison concluding "direct h5py released NWB paths ... SDK `from_nwb` reads the same NWB groups —
Equivalent". Step 2 verified that all 284 local ids appear in `ophys_experiment_table.csv` and that
NWB dF/F cell dimensions sum to 42,147, exactly matching `ophys_cells_table.csv`.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values of the retained experiments, taken from
`ophys_experiment_table.csv`, sorted as strings, and indexed per session. 38 mice result.

ii.
```python
info = {..., 'mouse_id': str(row.mouse_id), ...}
...
subjects = sorted({z['mouse_id'] for z in infos}); smap = {x: i for i, x in enumerate(subjects)}
data = {..., 'subjects': subjects,
        'subject_idx': np.asarray([smap[z['mouse_id']] for z in infos], dtype=np.int64), ...}
```

iii. Step 5 variable mapping: "mouse ID → `subjects`, `subject_idx`; Sorted string IDs and
per-experiment index; NWB `general/subject/subject_id` / metadata CSV; Only retained active
experiments." Step 9 records 38 retained subjects and cross-checks this against an independent raw
scan.

## 1-c. How are the data split into sessions?

i. A "session" in the output is **one ophys experiment = one imaging plane**, not one acquisition
session. For the 239 single-plane `VisualBehavior` experiments this is identical to an acquisition
session. For the 8 multiscope acquisition sessions (1 mouse, up to 7 simultaneously recorded planes
each) the *same* behavioral trials are emitted once per plane, i.e. the identical trial/output
sequence is repeated 3–7 times as separate "sessions" with different neuron sets. This is visible
in the verification log as runs of identical trial counts (`209` ×7, `287` ×7, `309` ×7, `239` ×5,
`196` ×5, `265` ×3) and in that mouse 457841 alone contributes 34 of the 199 sessions.
There is no grouping by `ophys_session_id`; sessions are simply the rows of the curated
metadata table, sorted by `ophys_experiment_id`.

ii.
```python
meta = meta.sort_values('ophys_experiment_id').reset_index(drop=True)
...
for k, row in meta.iterrows():
    eid = int(row.ophys_experiment_id)
    n, x, y, info = process_experiment(row, files[eid], imap, ...)
    neural.append(n); inputs.append(x); outputs.append(y); infos.append(info)
...
info = {'ophys_experiment_id': eid, 'ophys_session_id': int(row.ophys_session_id), ...}
'session_unit': 'ophys experiment / imaging plane'
```

iii. Step 4: "284 experiments map to 247 acquisition sessions; trial tables duplicate across sister
planes ... Treat each experiment as a decoder session because it has a distinct simultaneous neuron
population; report acquisition-level behavior without duplicates." Step 5 Key Decision 1: "each file
has a distinct simultaneous neuron population, matching paper decoding by plane. Multi-plane
behavioral rows are intentionally repeated across distinct neural sessions." Step 10 edge-case
review repeats: "Multi-plane duplicate behavior is intentional because each plane is a separate
simultaneously recorded neuron population."

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. A trial is kept if
`(go | catch) & ~aborted & ~auto_rewarded`. The trial window is the native
`start_time → stop_time` interval, tiled with fixed 100 ms bins starting at `start_time`
(`nbin = floor((stop - start)/0.1)`, bin centers at `start + (j+0.5)*0.1`). Trial lengths are
therefore variable (70–125 bins, mean 84.2) but the bin width is identical everywhere.

ii.
```python
flags = {k: np.asarray(tr[k]).astype(bool) for k in
         ['go','catch','aborted','auto_rewarded','hit','miss','false_alarm','correct_reject']}
keep = (flags['go'] | flags['catch']) & ~flags['aborted'] & ~flags['auto_rewarded']
tids = np.flatnonzero(keep)
starts = np.asarray(tr['start_time'], float); stops = np.asarray(tr['stop_time'], float)
...
for ti in tids:
    nbin = int(np.floor((stops[ti] - starts[ti]) / BIN_S + 1e-9))
    if nbin < 1:
        continue
    centers = starts[ti] + (np.arange(nbin) + 0.5) * BIN_S
```

iii. Step 5 Key Decision 5: "Native variable trial boundaries: retain full SDK/NWB start-to-stop
intervals (7.02–12.61 s), yielding variable T but identical 100 ms bin width. Alignment event is
trial start (`off_start=0`, `off_end=None`)." Step 3/4 confirm the trial flags partition the table
exactly (go+catch+aborted+auto_rewarded = 148,231) and that retained trials are 87.47 % go /
12.53 % catch, matching the whitepaper's designed 87.5 / 12.5 split.

## 1-e. How are trials filtered based on quality controls?

i. Four filters, applied at three levels:
- **Experiment level**: all `*_passive` session types are dropped (82 experiments), and the
  3 remaining active experiments with no `acquisition/EyeTracking/pupil_tracking` group are dropped
  (795953296, 806456687, 833631914). 284 → 199.
- **Trial level**: `(go|catch) & ~aborted & ~auto_rewarded`; trials shorter than one 100 ms bin are
  skipped; trials whose interpolated running **or** pupil series is not everywhere finite are
  skipped; trials that survive but carry none of the four outcome flags raise an error.
- **Session level**: an experiment with fewer than 2 usable trials raises and would abort the run
  (in practice all 199 have ≥ 39 trials).
- **Neuron level**: none (see 2-c).

ii.
```python
meta = meta[~meta.session_type.str.contains('passive', case=False, na=False)].copy()
eye_map = {eid: has_eye(files[eid]) for eid in meta.ophys_experiment_id.astype(int)}
meta = meta[meta.ophys_experiment_id.map(eye_map)].copy()
...
    if nbin < 1:
        continue
    r = interp_valid(centers, run_t, run_x)
    p = interp_valid(centers, eye_t, pupil, pupil_good)
    if not (np.isfinite(r).all() and np.isfinite(p).all()):
        continue
    ...
    else: raise ValueError(f'{eid} trial {ti}: retained trial lacks outcome')
if len(neural) < 2:
    raise ValueError(f'{eid}: fewer than 2 usable trials')
```

iii. Step 4: passive rows "are almost entirely miss/correct-reject because no task responses occur
... Exclude all passive experiments. Their nominal outcomes are not genuine task outcomes and would
create label artifacts." Step 4 on eye tracking: "Mandatory pupil output cannot be fabricated;
exclude experiment/session units without a valid eye stream." Step 5 Key Decision 9: "Drop a trial
if running or pupil has no valid support over its interval; do not create a missing category because
outputs must remain five percentile bins." The ≥2-trial rule follows the format spec ("There needs
to be at least two trials within each session").

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `processing/ophys/event_detection/data` — the **detected calcium event magnitudes**
(time × cell), with `processing/ophys/event_detection/timestamps` as the ophys frame clock.
dF/F (`processing/ophys/dff/traces/data`) is present in every file and was explicitly *not* used.
Note the NWB stores the *unfiltered* event array; the SDK's `filtered_events` (half-Gaussian
smoothed) is computed in `Events.__init__` and is not stored on disk, so the AI's traces are the raw
detected events.

ii.
```python
ev  = np.asarray(h['processing/ophys/event_detection/data'], dtype=np.float32)
ots = np.asarray(h['processing/ophys/event_detection/timestamps'], dtype=float)
if ev.shape[0] != len(ots):
    raise ValueError(f'{eid}: events/timestamps mismatch')
```
metadata: `'neural_signal': 'detected calcium event magnitude, mean in 100 ms bins'`

iii. Step 3: "Reference neural analyses use **detected calcium events**, each with time and
magnitude, rather than recomputing dF/F." This is taken directly from `methods.txt` line 208 ("For
all analysis of neural data we used the detected calcium events as described in Garrett et al.")
and line 179 ("We performed our analyses on discrete calcium events that were regressed from the raw
fluorescence traces, thus removing the slow decay dynamics of the calcium indicator GCaMP6f").
Step 4 records a data-integrity check that event and dF/F arrays match in shape in all 284 files and
that event timestamps equal the dF/F/ophys timestamps.

## 2-b. How is the `neural` data processed?

i. Only re-binning. For each trial the event array is averaged over the native ophys frames that
fall in each half-open 100 ms bin `[edge_j, edge_{j+1})`; bins that contain no native frame (which
happens for the ~10.7 Hz multiscope experiments) are filled by linear interpolation of the two
bracketing frames at the bin centre. The result is transposed to (neurons, time) and stored float32.
No smoothing, no normalisation, no baseline subtraction, no activity threshold.
Because detected events are extremely sparse (≈0.25 % of frame×cell samples are non-zero), 2,502 of
the 51,075 trials (4.9 %) end up with an **entirely zero** neural matrix; the verifier emitted one
warning per such trial.

ii.
```python
def event_bin_means(events, ts, starts, centers):
    """Mean native frame samples in 100-ms bins; interpolate empty bins."""
    nbin = len(centers); nc = events.shape[1]
    out = np.empty((nc, nbin), dtype=np.float32)
    edges = starts + np.arange(nbin + 1) * BIN_S
    lo = np.searchsorted(ts, edges[:-1], side='left')
    hi = np.searchsorted(ts, edges[1:], side='left')
    for j, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            out[:, j] = np.asarray(events[a:b], dtype=np.float32).mean(axis=0)
        else:
            k = np.searchsorted(ts, centers[j]); k = min(max(k, 1), len(ts)-1)
            t0, t1 = ts[k-1], ts[k]
            w = 0.0 if t1 == t0 else (centers[j]-t0)/(t1-t0)
            out[:, j] = (1-w)*events[k-1] + w*events[k]
    return out
```

iii. Step 5 Key Decision 3: "use released detected event magnitudes, not dF/F, matching the paper.
No additional smoothing or activity threshold." Key Decision 4 justifies the 100 ms bin (see 2-e).
Step 10 edge-case review: "Half-open neural bins `[edge_j, edge_{j+1})` avoid double-counting frames.
Empty 11 Hz event bins use local linear interpolation; nonempty bins use means." Note that Step 3
had stated the intent to use "filtered detected events", but Steps 4/5 and the code settle on the
unfiltered stored events, and the zero-activity trials are never mentioned — Step 10 Check 1 claims
the verification log contained no warnings, which is not the case (2,502 warnings).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron/ROI filtering at all: every cell present in the released `event_detection` array is
kept (4–666 per experiment, 29,168 total). No activity, SNR, or event-rate threshold. No trials are
dropped on the basis of neural content (including the 2,502 all-zero trials).

ii. (No filtering code; the only neural-side check is a shape/finiteness assertion.)
```python
if ev.shape[0] != len(ots):
    raise ValueError(f'{eid}: events/timestamps mismatch')
...
assert np.isfinite(n).all() and np.isfinite(o).all()
```

iii. Step 3 curation notes: segmentation/ROI QC ("cells within 3 µm of FOV boundaries", ">70 %
overlap duplicates", "union ROIs", "neuropil subtraction failures ... about 1 % of ROIs") is already
applied in the released files, therefore "Invalid ROIs are not included in final matched cells;
supplied NWB experiment/cell tables represent released post-QC data, so additional ad hoc activity
thresholds would double-filter and are not justified." Step 10 Check 3 records "keep all released
event-array ROIs ... Match; no unjustified double filtering."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **trial start** on the ophys session clock. The 100 ms bin grid is anchored at
`start_time` (bin *j* covers `[start + j·0.1, start + (j+1)·0.1)`, centre `start + (j+0.5)·0.1`),
and every other stream (running, pupil, image identity, change, outcome) is sampled on that same
grid, so all streams share indices by construction. Metadata records
`temporal_alignment_event = 'trial start time on the ophys session clock'`, `off_start = 0.0`,
`off_end = None` (variable-length trials).

ii.
```python
centers = starts[ti] + (np.arange(nbin) + 0.5) * BIN_S
r = interp_valid(centers, run_t, run_x)
p = interp_valid(centers, eye_t, pupil, pupil_good)
n = event_bin_means(ev, ots, starts[ti], centers)
...
'temporal_alignment_event': 'trial start time on the ophys session clock',
'off_start': 0.0, 'off_end': None,
```

iii. Step 1: "Stimulus timing and behavioral timing are in the common session clock. For this task
all streams will ultimately be sampled at `ophys_timestamps`, as explicitly required." Step 5 Key
Decision 5 fixes trial start as the alignment event. Step 10 edge-case review: "Trial bins use
`floor((stop-start)/0.1)` and centers from start+50 ms, guaranteeing every center is before stop."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 100.0 ms for every trial and session (`metadata['time_bin_size'] = 100.0`). Yes — the native data
are rebinned: single-plane experiments are ~30.94 Hz (≈3 frames/bin) and multiscope experiments are
~10.73 Hz (≈1 frame/bin), and both are resampled onto the common 100 ms grid (means for non-empty
bins, linear interpolation for empty ones). Running and pupil (~30 Hz native) are linearly
interpolated to the bin centres.

ii.
```python
BIN_S = 0.100
...
nbin = int(np.floor((stops[ti] - starts[ti]) / BIN_S + 1e-9))
centers = starts[ti] + (np.arange(nbin) + 0.5) * BIN_S
...
'time_bin_size': 100.0,
'native_rate_hz': float(1/np.median(np.diff(ots))),
```

iii. Step 5 Key Decision 4: "Common bin = 100 ms: this is supported by both 31 Hz and 11 Hz
experiments (about 3 and 1 native frames/bin), preserves 250 ms image flashes and 400 ms reference
decoding windows, and keeps variable trial lengths manageable." Step 3: "The target requires one
common bin size; explicit resampling onto a common ophys-time grid will therefore be needed."
Step 10 Check 3 flags this as the one deliberate deviation from the reference: "Reasoned
target-specific difference to obtain common bin size."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The **stimulus presentation** stream, not the trials table: `stimulus/presentation/<set>/timestamps`
(flash onset times) and `.../data` (integer indices), decoded through
`stimulus/templates/<set>/control_description` (the 8 image names per set, e.g. `im065`, `im077`).
A global vocabulary is built by scanning the templates of all retained experiments, giving 17
categories: `gray` (index 0) plus 16 distinct natural images across image sets A and B.

ii.
```python
def presentation_streams(h):
    """Return flash onsets and corresponding image names from NWB TimeSeries."""
    for gname, g in h['stimulus/presentation'].items():
        t = np.asarray(g['timestamps'], float)
        idx = np.asarray(g['data']).astype(int)
        desc = decode_strings(templates[gname]['control_description'][:])
        good = (idx >= 0) & (idx < len(desc)) & np.isfinite(t)
        onsets.extend(t[good]); names.extend(desc[idx[good]])
    order = np.argsort(onsets)
    return np.asarray(onsets)[order], np.asarray(names)[order]

def image_vocabulary(meta, files):
    ...
    return ['gray'] + sorted(n for n in names if n and n.lower() not in {'gray', 'omitted'})
```

iii. Step 5 mapping: "stimulus presentation `image_name`, start/stop → output 0: image identity;
Assign each ophys-aligned bin the presented image category; gray gaps category 0;
`StimulusPresentations.from_nwb`; Eight images pooled by identity labels plus gray; omitted periods
remain gray because no image is shown." Step 3 established the 250 ms image / 500 ms gray /
750 ms-interval cadence from the paper methods.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each 100 ms bin centre the code finds the most recent flash onset at or before the centre and
labels the bin with that image **only if** the centre falls inside the 250 ms visible window
`[onset, onset + 0.25)`; otherwise the bin is `gray` (0). Omitted flashes and inter-flash gray gaps
therefore both map to 0. Labels are integer indices into the global 17-name vocabulary, stored int64.
Resulting occupancy: gray 0.665, images 0.335 — consistent with the 250/750 ms duty cycle.

ii.
```python
image = np.zeros(nbin, dtype=np.int64)
# A natural image is visible for 250 ms after each flash onset.
left  = np.searchsorted(flash_t, centers - 0.250, side='left')
right = np.searchsorted(flash_t, centers, side='right')
for j, (a, b) in enumerate(zip(left, right)):
    if b > a:
        ft = flash_t[b-1]
        if ft <= centers[j] < ft + 0.250:
            image[j] = image_to_idx.get(str(flash_name[b-1]), 0)
```

iii. Step 5 Key Decision 6: "use a global deterministic vocabulary with gray as 0 and the released
natural-image names thereafter. Gray includes inter-flash gaps and omission intervals because
requested identity is the image shown during non-gray screen." Step 10 edge cases: "Stimulus
visibility uses `[onset,onset+0.25)`; inter-flash and omission periods map to gray." Step 9 cites
the 33.47 % image occupancy as a consistency check against the 250/750 ms cadence.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is computed on exactly the same 100 ms `centers` array used to bin the neural data, so the
image row and the neural matrix share indices by construction and are the same length.

ii.
```python
centers = starts[ti] + (np.arange(nbin) + 0.5) * BIN_S
n = event_bin_means(ev, ots, starts[ti], centers)   # neural
image = np.zeros(nbin, dtype=np.int64)              # image identity on the same grid
...
outputs.append(np.vstack([image, change, rl, pl, np.full(len(centers), outcome)]).astype(np.int64))
assert n.ndim == 2 and o.shape == (5, n.shape[1])
```

iii. Step 5/Step 10: all streams are resampled onto one trial-start-anchored grid; the
`--show-processing` plots overlay the image labels on the binned event heat-map on a shared time
axis, and Step 10 Check 2 reports an independent raw-NWB recomputation of the image row matching via
`np.allclose`.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `intervals/trials/change_time` together with the `intervals/trials/go` flag. Catch trials (sham
changes) are deliberately left at 0.

ii.
```python
flags = {k: np.asarray(tr[k]).astype(bool) for k in ['go','catch',...]}
changes = np.asarray(tr['change_time'], float)
...
if flags['go'][ti] and np.isfinite(changes[ti]):
```

iii. Step 5 mapping: "stimulus presentation `is_change` / trial `change_time` → output 1: image
change; Binary impulse in first bin at/after true change presentation onset." Key Decision 7:
"use actual change time/is_change onset and mark exactly one 100 ms bin. Catch trials contain no
identity change and remain zero."

## 4-b. What processing is involved in computing `output` *Image change*?

i. A single-bin impulse. The index of the first bin whose *left edge* is at or after `change_time`
is computed as `ceil((change_time - start_time)/0.1)` and that one bin is set to 1; everything else
is 0. There is no sustained window. Over the whole dataset only 1.03 % of time bins are 1.

ii.
```python
change = np.zeros(nbin, dtype=np.int64)
if flags['go'][ti] and np.isfinite(changes[ti]):
    j = int(np.ceil((changes[ti] - starts[ti]) / BIN_S - 1e-12))
    if 0 <= j < nbin:
        change[j] = 1
```

iii. Step 5 Key Decision 7 reads the instruction "Have value of 1 right after a change in image
identity" literally as a one-bin event rather than a sustained label. Step 12 notes the consequence
and defends it: "Image change is a deliberately sparse one-bin event (1.03 % of time points), yet
reaches 0.635 balanced accuracy, supporting correct temporal alignment."

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary, so the only "threshold" is the temporal one: the boundary is the first bin
edge at/after `change_time` (`ceil`, with a 1e-12 tolerance to avoid a spurious extra bin when the
change time lands exactly on an edge), and the go/catch flag gates whether any bin is marked at all.
`output_values[1] = ['no_change', 'change']`.

ii.
```python
j = int(np.ceil((changes[ti] - starts[ti]) / BIN_S - 1e-12))
if 0 <= j < nbin:
    change[j] = 1
...
'output_values': [vocab, ['no_change','change'], ...]
...
assert set(np.unique(y[1])).issubset({0,1})
```

iii. Step 10 edge-case review: "Change uses first bin at/after onset; direct raw comparisons passed."
Step 7 sanity check: "Exactly one change bin occurs on each sample go trial; catch trials remain
zero."

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same trial-start-anchored 100 ms grid as the neural data — the change index is computed in units
of `BIN_S` from `start_time`, i.e. directly in neural-bin coordinates, so it is the bin immediately
following the one that contains the change.

ii.
```python
j = int(np.ceil((changes[ti] - starts[ti]) / BIN_S - 1e-12))
...
outputs.append(np.vstack([image, change, rl, pl, np.full(len(centers), outcome, dtype=np.int64)]))
```

iii. Step 7 processing-plot review: "Image labels alternate with gray at expected cadence; change
impulses coincide with image transitions." Step 10 Check 2 independently recomputed the one-bin
change events from raw change times and compared with `np.allclose` (passed).

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed/data` and `processing/running/speed/timestamps` — the SDK's
`running_speed` attribute, i.e. the wrap/transient-corrected **and 10 Hz low-pass filtered** linear
speed in cm/s (the unfiltered `running_speed_raw` stream, also present in all 284 files, was not
used).

ii.
```python
run_t = np.asarray(h['processing/running/speed/timestamps'], float)
run_x = np.asarray(h['processing/running/speed/data'], float)
```

iii. Step 5 mapping cites `RunningSpeed.from_nwb` as the reference function. Step 2 verified "Running
speed is timestamped and present in all 284 files (`processing/running/speed/{data,timestamps}`);
unfiltered speed is also present" — i.e. the choice of the filtered stream is deliberate, matching
`methods.txt`, which describes the filtered stream as the `running_speed` attribute.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation of the native ~30 Hz speed onto the 100 ms bin centres (`np.interp`, which
clamps to the nearest valid sample outside the support rather than extrapolating), restricted to
finite samples; then percentile discretisation (see 5-c). No smoothing, no absolute-value or
sign handling — negative (backward) speeds are retained and simply fall in the lowest bins.

ii.
```python
def interp_valid(t_new, t, x, valid=None):
    t = np.asarray(t, float); x = np.asarray(x, float)
    good = np.isfinite(t) & np.isfinite(x)
    if valid is not None:
        good &= np.asarray(valid, bool)
    if good.sum() < 2:
        return np.full(len(t_new), np.nan, dtype=np.float32)
    # np.interp uses nearest valid endpoint outside support, as documented.
    return np.interp(t_new, t[good], x[good]).astype(np.float32)
...
r = interp_valid(centers, run_t, run_x)
```

iii. Step 3: "Running and pupil streams must be interpolated to ophys-aligned bins before percentile
discretization." Step 5 Key Decision 9: "linear interpolate only between valid samples. Edge values
may use nearest valid sample within the stream."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile bins with **per-experiment (per-session) edges**: the 20/40/60/80 %
quantiles are computed over all finite running samples of that experiment's retained trials, and
labels 0–4 are assigned by `searchsorted(..., side='right')`. Edges are stored per session in
`metadata['session_info'][i]['running_quintile_edges']`. Each category holds ~20.0 % of the data both
within a session and globally.

ii.
```python
def quintile_labels(values):
    allv = np.concatenate([v[np.isfinite(v)] for v in values if np.isfinite(v).any()])
    if len(allv) == 0:
        raise ValueError('No finite samples for quintile discretization')
    edges = np.quantile(allv, [0.2, 0.4, 0.6, 0.8])
    labels = [np.searchsorted(edges, v, side='right').astype(np.int64) for v in values]
    return labels, edges
...
run_lab, run_edges = quintile_labels([q[2] for q in prelim])
```

iii. Step 5 mapping: "Linear interpolation to bin centers, then session-specific quintiles 0–4.
Quintile edges fit using all valid retained-session time samples, avoiding trial-specific rank
leakage." Key Decision 8: "compute quintile edges independently for each experiment/session from
valid aligned values; collapse duplicate quantile edges safely if needed ... Store categorical labels
0–4 with output names `0-20%` ... `80-100%`." Step 9 verifies each category is ~0.200 of the data.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is interpolated directly onto the same `centers` array used to bin the neural data, so the
running row is aligned bin-for-bin with the neural matrix (and is guaranteed the same length by the
per-trial assertion).

ii.
```python
centers = starts[ti] + (np.arange(nbin) + 0.5) * BIN_S
r = interp_valid(centers, run_t, run_x)
n = event_bin_means(ev, ots, starts[ti], centers)
...
assert o.shape == (5, n.shape[1])
```

iii. Step 4/Step 1 note that all streams share the hardware-synced session clock ("Temporal
synchronization of all data-streams ... recorded on a single NI PCI-6612 digital IO board"), so
interpolation onto the common grid is valid. Step 7 plot review: "Continuous running/pupil traces and
their quintile steps are temporally coincident with no visible shift." Step 10 Check 2 reproduced the
running interpolation and quintile labels from raw NWB with `np.allclose`.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/data` (an EllipseSeries storing the two pupil ellipse
axes) with `.../timestamps`, plus `acquisition/EyeTracking/likely_blink/data` for blink rejection.
The scalar diameter is the geometric mean of the two axes, `sqrt(width*height)` (an
equivalent-circle diameter), rather than the reference's single `pupil_width` axis.

ii.
```python
eye_t = np.asarray(h['acquisition/EyeTracking/pupil_tracking/timestamps'], float)
axes  = np.asarray(h['acquisition/EyeTracking/pupil_tracking/data'], float)
blink = np.asarray(h['acquisition/EyeTracking/likely_blink/data']).astype(bool)
# EllipseSeries stores width and height (diameters); equivalent circular diameter.
pupil = np.sqrt(np.maximum(axes[:, 0] * axes[:, 1], 0.0))
pupil_good = (~blink) & np.isfinite(pupil) & (pupil > 0)
```

iii. Step 5 mapping: "derive equivalent diameter from ellipse (`sqrt(width*height)` because the
stored values are ellipse diameters) ... Equivalent circular diameter uses both pupil axes and is
robust to ellipse orientation." Step 4: "Eye tracking exists in 281/284 experiments; 3 active
experiments lack it ... Mean 99.1 % of pupil rows finite; mean blink fraction 3.46 % ... Treat
blink/nonfinite samples as missing and temporally interpolate only within valid support; do not label
blinks as a diameter category."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames and non-positive/non-finite samples are removed **before** interpolation; the
remaining valid samples are linearly interpolated onto the 100 ms bin centres with the same
`interp_valid` helper (nearest-value clamping at the edges); then percentile discretisation (6-c).
A trial whose interpolated pupil is not everywhere finite is dropped.

ii.
```python
pupil_good = (~blink) & np.isfinite(pupil) & (pupil > 0)
...
p = interp_valid(centers, eye_t, pupil, pupil_good)
if not (np.isfinite(r).all() and np.isfinite(p).all()):
    continue
```

iii. Step 10 edge cases: "Blinks/nonfinite pupil samples are excluded before interpolation; three
sessions with no eye stream are excluded rather than fabricated." Step 5 Key Decision 9 forbids a
"missing" category "because outputs must remain five percentile bins".

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical machinery to running speed: five equal-percentile bins with **per-experiment** 20/40/60/80 %
quantile edges over that experiment's finite pupil samples, labels 0–4, edges stored in
`session_info[i]['pupil_quintile_edges']`. Each category holds ~20.0 % of the data.

ii.
```python
pup_lab, pup_edges = quintile_labels([q[3] for q in prelim])
...
edges = np.quantile(allv, [0.2, 0.4, 0.6, 0.8])
labels = [np.searchsorted(edges, v, side='right').astype(np.int64) for v in values]
...
'output_values': [..., ['0-20%','20-40%','40-60%','60-80%','80-100%'], ...]
```

iii. Step 5 mapping/Key Decision 8, same rationale as running speed ("session-specific quintiles
0–4"). Implicitly this also normalises away between-session differences in absolute pupil size
(camera geometry), although the notes justify it only in terms of using all retained-session samples
rather than per-trial ranks.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — interpolated directly onto the neural bin centres, so it is aligned
bin-for-bin.

ii.
```python
p = interp_valid(centers, eye_t, pupil, pupil_good)
n = event_bin_means(ev, ots, starts[ti], centers)
outputs.append(np.vstack([image, change, rl, pl, ...]))
```

iii. Same justification as 5-d: all streams live on the hardware-synced session clock and are
resampled to one grid; Step 10 Check 2 reproduced the blink-filtered pupil interpolation and quintile
labels from raw NWB with `np.allclose`.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of `intervals/trials`: `hit`, `miss`, `false_alarm`,
`correct_reject`, checked in that order and mapped to 0–3. A retained trial with none of the four
set raises an error rather than falling back to a sentinel.

ii.
```python
if   flags['hit'][ti]:            outcome = 0
elif flags['miss'][ti]:           outcome = 1
elif flags['false_alarm'][ti]:    outcome = 2
elif flags['correct_reject'][ti]: outcome = 3
else: raise ValueError(f'{eid} trial {ti}: retained trial lacks outcome')
```

iii. Step 3 curation rules: "Use SDK/NWB trial flags ... Outcome mapping: hit/miss for go and false
alarm/correct reject for catch." Step 4 verified the identity exactly: "`hit + miss = go`,
`false_alarm + correct_reject = catch`, and trial classes partition every row" across all 174 active
acquisition sessions.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The integer code is broadcast across all time bins of the trial so that all five outputs form one
`(5, n_timepoints)` matrix; `validate()` asserts that the outcome row is constant within each trial.
`output_values[4] = ['hit','miss','false_alarm','correct_reject']`. Resulting time-weighted
distribution: hit 0.303, miss 0.571, false alarm 0.017, correct reject 0.108 (trial counts
15,682 / 28,990 / 920 / 5,483).

ii.
```python
outputs.append(np.vstack([image, change, rl, pl,
                          np.full(len(centers), outcome, dtype=np.int64)]).astype(np.int64))
...
assert y[4].min() >= 0 and y[4].max() <= 3 and np.unique(y[4]).size == 1
```

iii. Step 5 Key Decision 10: "categories are hit, miss, false alarm, correct reject. Broadcasting a
static outcome over time is semantically equivalent and required by the trainer's homogeneous output
matrix."

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **No eye-tracking group** (3 active experiments): experiment excluded up front by `has_eye()`.
- **Blinks / non-finite / non-positive pupil samples**: masked out before interpolation.
- **Gaps or out-of-range times** in running/pupil: `np.interp` clamps to the nearest valid sample;
  if a stream has < 2 valid samples the interpolation returns all-NaN and the trial is skipped
  (and the experiment then fails the ≥2-trial check).
- **Empty neural bins** (11 Hz multiscope): filled by linear interpolation of the bracketing frames.
- **Events/timestamps length mismatch**: raises.
- **Degenerate trials** (`nbin < 1`), **trials with no outcome flag**, **experiments with < 2 usable
  trials**: skipped or raised.
- **Stimulus indices out of range of the template description**: masked out (`good = (idx >= 0) &
  (idx < len(desc))`).
- Final `validate()` asserts finiteness, shape agreement, categorical ranges, and static outcome for
  every trial.
Not handled: the 2,502 trials (4.9 %) whose neural matrix is entirely zero because no calcium event
was detected in that trial; these are passed through, and Step 10 of CONVERSION_NOTES incorrectly
reports that the verification log contained no warnings.

ii.
```python
def has_eye(path):
    with h5py.File(path, 'r') as h:
        return 'acquisition/EyeTracking/pupil_tracking/data' in h
...
if good.sum() < 2:
    return np.full(len(t_new), np.nan, dtype=np.float32)
...
if not (np.isfinite(r).all() and np.isfinite(p).all()):
    continue
...
if len(neural) < 2:
    raise ValueError(f'{eid}: fewer than 2 usable trials')
```

iii. Step 5 Key Decision 9: "Missing behavior samples: linear interpolate only between valid samples.
Edge values may use nearest valid sample within the stream. Drop a trial if running or pupil has no
valid support over its interval; do not create a missing category because outputs must remain five
percentile bins." Step 4: "Mandatory pupil output cannot be fabricated; exclude experiment/session
units without a valid eye stream."

## 9-a. What are the most time-consuming steps of the code?

i. Total full conversion was 149.3 s for 199 experiments (well under the 15-minute budget). The
dominant costs are I/O: (1) `select_experiments` opens all 284 NWB files once just to test for the
presence of the eye-tracking group; (2) `image_vocabulary` reopens all 199 retained files to read
their stimulus templates; (3) `process_experiment` reads each experiment's **entire** session event
array (48 k–150 k frames × 4–666 cells) plus the full running, eye and stimulus streams into memory.
Per-experiment processing is printed and ranges 0.29–0.94 s (~110 s of the 149 s); the remaining
~40 s is the two extra full passes plus pickling a 2.67 GiB file.

ii.
```python
eye_map = {eid: has_eye(files[eid]) for eid in meta.ophys_experiment_id.astype(int)}   # pass 1
...
def image_vocabulary(meta, files):
    for eid in meta.ophys_experiment_id.astype(int):
        with h5py.File(files[eid], 'r') as h:                                          # pass 2
            ...
...
ev = np.asarray(h['processing/ophys/event_detection/data'], dtype=np.float32)          # pass 3
print(f"{eid}: {len(neural)} trials, {neural[0].shape[0]} neurons, ... {info['seconds']:.2f}s")
```

iii. Step 6: "Full PyNWB/AllenSDK object construction would load unnecessary images/templates and
incur high overhead for 199 large NWBs" → "Stream one NWB at a time with h5py and read only required
datasets"; "Build each experiment once and close its file before proceeding; no redundant full-file
reads during conversion." Step 7 estimated "~2 minutes for 199 sessions plus I/O", which matched the
actual 149 s.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Three Python-level loops remain:
- `event_bin_means`: the per-bin loop over `(lo, hi)` pairs. With ~84 bins × 51,075 trials this is
  ~4.3 M iterations; it could be replaced by `np.add.reduceat` on the frame axis divided by the
  per-bin counts, with the (rare) empty bins patched afterwards.
- the image-identity loop over bins inside `process_experiment`: the whole thing is expressible as
  two `searchsorted` calls plus a vectorised mask `(flash_t[b-1] <= centers) & (centers < flash_t[b-1]+0.25)`.
- the outer per-trial loop over `tids`, and the `quintile_labels` list comprehension over trials
  (could operate on one concatenated array with offsets).

ii.
```python
for j, (a, b) in enumerate(zip(lo, hi)):        # per-bin, per-trial
    if b > a:
        out[:, j] = np.asarray(events[a:b], dtype=np.float32).mean(axis=0)
...
for j, (a, b) in enumerate(zip(left, right)):   # per-bin, per-trial
    if b > a:
        ft = flash_t[b-1]
        if ft <= centers[j] < ft + 0.250:
            image[j] = image_to_idx.get(str(flash_name[b-1]), 0)
```

iii. Step 6 claims the speed-ups that *were* taken: "Use `np.searchsorted` for trial/frame and
stimulus timing lookup" and "Per-bin boolean masks over all native timestamps would scale poorly"
(so searchsorted replaced masks). The remaining per-bin Python loops are not identified as a
residual inefficiency in the notes; Step 7 concluded the run time was already comfortably under the
15-minute threshold, so no further vectorisation was pursued.

## 9-c. What processing does the code repeat multiple times?

i. Each NWB file is opened up to three times — once in `has_eye()` (all 284), once in
`image_vocabulary()` (all 199 retained), once in `process_experiment()`. The eye-tracking presence
test and the template read could both have been folded into the single conversion pass (or the
template names read from one file per image set, since only two image sets exist). Within a trial,
`np.searchsorted` over `flash_t` is called twice (`left`, `right`) and again per bin in the empty-bin
fallback. `quintile_labels` concatenates all trials' values once per output (running, pupil) per
experiment. Nothing substantive is recomputed at the session level.

ii.
```python
eye_map = {eid: has_eye(files[eid]) for eid in meta.ophys_experiment_id.astype(int)}
...
vocab = image_vocabulary(meta, files)
...
n, x, y, info = process_experiment(row, files[eid], imap, ...)
```

iii. Step 6 asserts "no redundant full-file reads during conversion" — true of the conversion loop
itself, but the two preparatory scans are redundant file opens that the notes do not acknowledge.
The cost is small (metadata-only reads) relative to the 149 s total.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- The **entire session** event array (up to 150 k frames × 666 cells) is materialised in RAM even
  though only the retained trial windows (~30 % of the session) are used; an HDF5 slice per trial, or
  one slice spanning min(start)–max(stop), would suffice.
- The neural output is stored as **dense float32** although the detected-event traces are ~99.75 %
  zeros, which is what makes the pickle 2.67 GiB; a sparse or quantised representation would be far
  smaller (the decoder densifies anyway, so this is storage-only waste).
- A per-trial empty `np.empty((0, nbin))` input array is built and pickled for all 51,075 trials —
  required by the format, but it carries no information.
- `flags['catch']` is read and used only inside `keep`, where it is implied by
  `~aborted & ~auto_rewarded` for this table; `info['source_trial_indices']`, `info['seconds']`,
  `info['raw_retained_trials']`, and the per-session quintile edges are pickled into
  `metadata['session_info']` and never read by the decoder.
- The static trial outcome is broadcast across every time bin (required by the trainer's
  `(5, T)` output matrix, but 84× redundant).
- `--show-processing` builds matplotlib figures that are not part of the conversion product.

ii.
```python
ev = np.asarray(h['processing/ophys/event_detection/data'], dtype=np.float32)   # whole session
...
neural.append(n); inputs.append(np.empty((0, nbin), dtype=np.float32))
...
info = {..., 'source_trial_indices': kept_ids, 'seconds': time.time()-t0,
        'running_quintile_edges': run_edges.tolist(), 'pupil_quintile_edges': pup_edges.tolist()}
'session_info': infos,
```

iii. Step 6 lists the memory/IO measures that were taken ("Store neural arrays as float32 and
categorical arrays as int64", "Stream one NWB at a time") and Step 5 Key Decision 11 justifies the
dtype choices. Step 5 Key Decision 10 justifies the outcome broadcast as "required by the trainer's
homogeneous output matrix". The whole-session array read and the dense storage of a near-empty
matrix are not discussed in the notes.
