# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is DANDI:000363 (Mesoscale Activity Map), distributed as one NWB file per
recording session under `/app/data/sub-<subject_id>/`. The AI discovers every session with a
single sorted glob over that layout (174 files) and processes each file exactly once. Rather
than using `pynwb`, it opens each file directly with `h5py` and reads the HDF5 paths it needs:
`identifier`, `general/subject/{subject_id,description}`, the `intervals/trials` columns,
`acquisition/BehavioralEvents/{go,sample}_start_times/timestamps`, the `units/` group
(`spike_times` + `spike_times_index`, `obs_intervals` + index, `classification`, `anno_name`,
`electrodes`), `general/extracellular_ephys/electrodes` (CCF coordinates) and
`acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`. Sessions are farmed out to a
`multiprocessing` pool (default 12 workers, 16 used for the full run), one session per worker,
and the results are re-ordered back into file order before assembly. `--sample` takes the
first 2 files.

ii.
```python
DATA_DIR = '/app/data'
...
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
print(f'found {len(files)} NWB files')
if args.sample:
    files = files[:2]
```
```python
with ctx.Pool(min(args.workers, len(files))) as pool:
    for i, res in enumerate(pool.imap(_worker, list(zip(files, want_raw)))):
        results[i] = res
        _report(i, files, res, t_start)
```
```python
def process_session(fname, want_raw=False):
    t0 = time.time()
    with h5py.File(fname, 'r') as f:
        ident = f['identifier'][()].decode()
        mouse = f['general/subject/description'][()].decode().strip()
        subject_id = f['general/subject/subject_id'][()].decode().strip()

        tr = f['intervals/trials']
        n_trials_table = len(tr['id'])
        start_time = tr['start_time'][:]
        ...
        go = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
        sample_start = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
        assert len(go) == n_trials_table, f'{ident}: go cue count != trial count'
        u = f['units']
```

iii. From CONVERSION_NOTES Step 2: the NWB files are "one file per recording session; all probes
of a session are merged into a single file", so the directory listing is the complete and
unambiguous set of sessions; sorting makes the order deterministic. The counts obtained (174
files, 28 subjects, 272,227 clusters) were checked against `dandiset.yaml` and against the
paper. `h5py` rather than `pynwb` was chosen for speed: Step 6 notes that the whole
`units/spike_times` dataset is read once per session in bulk (≤300 MB) instead of issuing
"~400 small HDF5 reads", and that 16 worker processes give a "~4.5× wall-clock reduction"
(full conversion = 41 s). Step 10 Check 3(a) argues the `h5py`/NWB route is equivalent to the
reference pipeline's per-probe `.mat` loading because "the NWB `units` table is the
concatenation of the same per-probe unit tables".

## 1-b. How are the data split into subjects?

i. Every NWB file stores its animal both as a numeric `subject_id` (e.g. `440956`) and as the
mouse name used in the papers in `general/subject/description` (e.g. `SC015`). The AI reads
both, keeps the numeric id only in `session_info`, and uses the **mouse name** as the public
subject identity: `subjects` is the sorted set of unique mouse names and `subject_idx` indexes
into it per session. This gives 28 subjects with 3–10 sessions each.

ii.
```python
mouse = f['general/subject/description'][()].decode().strip()
subject_id = f['general/subject/subject_id'][()].decode().strip()
```
```python
info = dict(
    session_id=ident, mouse=mouse, subject_id=subject_id, file=os.path.basename(fname), ...)
```
```python
subjects = sorted({r['info']['mouse'] for r in kept})
sub_index = {s: i for i, s in enumerate(subjects)}

data = {
    ...
    'subjects': subjects,
    'subject_idx': np.array([sub_index[r['info']['mouse']] for r in kept], dtype=np.int64),
```

iii. CONVERSION_NOTES Step 2 records the mapping `subject_id` (`440956`) ↔ `description`
(`SC015`) and the mapping table in Step 5 lists `general/subject/description` (`SC015`) →
`subjects`. The mouse name is the identifier used throughout the papers and inside
`nwb.identifier` (`SC015_20190207_120657_s1`), so using it makes the converted data directly
comparable with the papers; the numeric DANDI id is retained in `session_info` so nothing is
lost. The resulting 28 subjects and their session counts were checked against the dandiset.

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no splitting or grouping is needed. Each session is identified
by the NWB `identifier` (`SC015_20190207_120657_s1`, encoding mouse, date, time and session
number). Session order in the output is the sorted file order (which, because the filename
embeds the acquisition timestamp, is chronological within a subject). Per-session bookkeeping
(`session_id`, `mouse`, `subject_id`, file name, unit/trial counts, the 0-based
`trial_index` into the original trials table, mean rate, tongue percentiles, etc.) is stored in
`metadata['session_info']`. 173 of 174 sessions reach the output; one is dropped (see 2-c).

ii.
```python
ident = f['identifier'][()].decode()
```
```python
info = dict(
    session_id=ident, mouse=mouse, subject_id=subject_id, file=os.path.basename(fname),
    n_units_total=len(classification), n_units_good=int((classification == 'good').sum()),
    n_neurons=len(unit_idx), n_trials_table=n_trials_table, n_trials=n_tr,
    n_trials_observed=int(observed.sum()),
    n_excluded_water=int((observed & ((auto_water != 0) | (free_water != 0))).sum()),
    n_excluded_no_spikes=n_zero_trials,
    trial_index=trials.astype(np.int32),   # 0-based index into intervals/trials
    ...)
```
```python
kept = [r for r in results if r is not None]
dropped = [f for f, r in zip(files, results) if r is None]
...
'session_info': [r['info'] for r in kept],
```

iii. Step 2 of CONVERSION_NOTES establishes that the file boundary *is* the session boundary
("one file per recording session; all probes of a session are merged into a single file"), so
nothing has to be inferred. Step 4/Step 9 justify keeping 173 sessions: the paper reports 173
behavioural sessions, and dropping the single file with no quality-controlled units yields
exactly 173. The AI explicitly decided **not** to apply the paper's behavioural session
criterion (performance > 65 %, ≥ 50 correct left and ≥ 50 correct right), because (a) the
released 173 sessions reproduce the published per-area unit counts exactly and so *are* the
paper's sessions, and (b) applying the criterion would bias the required `choice`/`outcome`
output distributions.

## 1-d. How are the data split into trials?

i. Trials are the rows of the NWB trials table `intervals/trials`, one row per behavioural
trial. The AI asserts that the number of go-cue events equals the number of table rows, so the
go cue used for alignment is unambiguous per trial. All per-trial columns (`start_time`,
`stop_time`, `outcome`, `early_lick`, `trial_instruction`, `auto_water`, `free_water`,
`photostim_onset`, `photostim_duration`) are read as whole arrays and then indexed by the
surviving trial indices.

ii.
```python
tr = f['intervals/trials']
n_trials_table = len(tr['id'])
start_time = tr['start_time'][:]
stop_time = tr['stop_time'][:]
outcome = _decode(tr['outcome'][:])
early = _decode(tr['early_lick'][:])
instruction = _decode(tr['trial_instruction'][:])
auto_water = tr['auto_water'][:]
free_water = tr['free_water'][:]
ps_onset = _decode(tr['photostim_onset'][:])
ps_dur = _decode(tr['photostim_duration'][:])

go = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
sample_start = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
assert len(go) == n_trials_table, f'{ident}: go cue count != trial count'
```

iii. CONVERSION_NOTES Step 2 documents that `go_start_times` "has exactly one entry per trial",
whereas `sample_start_times` has *more* entries than trials because an early lick triggers a
replay of the sample epoch — so the go cue is the only event stream that can be mapped 1:1 onto
trials, and the assertion enforces that. The trials table is used directly rather than
re-deriving trial boundaries from event streams.

## 1-e. How are trials filtered based on quality controls?

i. Three filters, all about whether usable spike data exists or whether the trial is a genuine
behavioural report:
 1. **Ephys coverage** — only trials covered by `units/obs_intervals` are kept. If every good
    unit has as many observation intervals as there are table rows, all trials are marked
    observed; otherwise the union of the good units' interval starts is matched back onto
    `trials.start_time` with `searchsorted`. (In 8–9 sessions the ephys covers only the leading
    N trials of the behavioural table.)
 2. **Water-delivery trials** — `auto_water != 0` or `free_water != 0` trials are excluded,
    following the reference `get_regular_trial_mask`.
 3. **Empty windows** — after binning, any trial in which *no* neuron of the ≥90 simultaneously
    recorded neurons fired a single spike in the whole 4 s window is dropped (2 such trials in
    the whole dataset; they arise where `obs_intervals` over-reports coverage by one trial).

A session is dropped if fewer than 2 trials survive (checked both before and after the
empty-window filter). Early-lick, no-response (`ignore`) and photostimulation trials are
deliberately **kept**. Full accounting: 94,370 table trials in the 173 kept sessions → 93,310
observed → −3,764 auto/free water → −2 empty → **89,544** converted trials.

ii.
```python
# ---- which trials did the ephys recording actually cover? ---------------
oii = u['obs_intervals_index'][:]
n_obs = np.diff(np.concatenate([[0], oii]))
n_obs_good = n_obs[unit_idx]
observed = np.zeros(n_trials_table, dtype=bool)
if np.all(n_obs_good == n_trials_table):
    observed[:] = True
else:
    # all units of a session share the same contiguous observed trial set;
    # derive it from the union of their observation intervals to be safe.
    oi = u['obs_intervals']
    for k in unit_idx:
        o = oi[(oii[k] - n_obs[k]):oii[k]]
        idx = np.searchsorted(start_time, o[:, 0] + 1e-9) - 1
        observed[idx[idx >= 0]] = True

# ---- trial curation ------------------------------------------------------
valid = observed & (auto_water == 0) & (free_water == 0)
trials = np.where(valid)[0]
if len(trials) < 2:
    return None
```
```python
# Drop trials in which not one of the (>=90) simultaneously recorded neurons
# fired a single spike in the whole 4 s window. ...
has_data = fr.sum(axis=(0, 2)) > 0
n_zero_trials = int((~has_data).sum())
if n_zero_trials:
    fr = fr[:, has_data, :]
    trials = trials[has_data]
    go_v = go_v[has_data]
    if len(trials) < 2:
        return None
```

iii. From Step 4/Step 5/Step 10: `obs_intervals` is the file's own record of which trials the
ephys recording covered, and Step 2 verified that spikes exist only inside trial intervals.
`auto_water`/`free_water` exclusion is copied straight from the reference
`get_regular_trial_mask` — these are not genuine behavioural reports. The AI explicitly records
that it *cannot* apply the rest of `get_regular_trial_mask` (no early lick ∧ response given ∧
no photostim), because `early_lick`, `outcome == ignore` / `choice == no lick` are **required
decoder outputs** and photostimulation is a **required decoder input**; removing them would
make three of four outputs and one of two inputs constant. This is flagged as the one
deliberate departure from the reference and as an explicitly allowed discrepancy. The empty-
window rule was added after the Step 7 sample run produced the warning "Session 1, trial 159:
all neural data is zero"; the AI traced it to `SC015_20190208_133600`, whose `obs_intervals`
lists 160 trials although the last spike in the file is at t = 1107.44 s while trial 159's
window starts at 1110.14 s. The 2-trial minimum comes from the target-format requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` together with its ragged index `units/spike_times_index`, restricted to
the units that pass QC (`units/classification == 'good'` and a non-empty `units/anno_name`).
The other input is `acquisition/BehavioralEvents/go_start_times/timestamps`, which positions
the bin edges. Unit→region assignment additionally uses `units/electrodes` and
`general/extracellular_ephys/electrodes/{x,z}` (CCF ML/AP), but those affect
`brain_region_idx`, not the rates.

ii.
```python
u = f['units']
classification = _decode(u['classification'][:])
anno = _decode(u['anno_name'][:])
keep = (classification == 'good') & (anno != '')
unit_idx = np.where(keep)[0]
```
```python
spike_index = u['spike_times_index'][:]
spike_times = u['spike_times'][:]
fr = bin_spikes(spike_times, spike_index, unit_idx, go_v)
del spike_times
```

iii. Step 1/Step 5 of CONVERSION_NOTES map `units/spike_times` onto the reference's
`neuron_single_units` field and `go_start_times` onto `task_cue_time[0]`. Spike times are the
only neural representation in the file; this is electrophysiology so no ΔF/F is needed.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin **firing rates in spikes/s**. For each good unit, the
81 bin edges of every trial are built as one flat array of absolute session times;
`np.searchsorted` gives the running spike count at each edge and adjacent differences give the
spike count per bin; the counts are then divided by the 0.05 s bin width. Bins are half-open
`[edge_k, edge_{k+1})`. No smoothing, normalisation, baseline subtraction or z-scoring is
applied. Rates are stored as `float32`, one `(n_neurons, 80)` array per trial.

ii.
```python
def bin_spikes(spike_times, spike_index, unit_idx, go_times):
    """...
    Mirrors `preprocessing_DJ_2022Aug.sliding_histogram(..., rate=True)` with
    stride == bin width: spikes are counted in half-open bins [edge_k, edge_{k+1}).
    """
    n_trials = len(go_times)
    edges = go_times[:, None] + BIN_EDGES[None, :]
    flat_edges = np.ascontiguousarray(edges.ravel())

    fr = np.empty((len(unit_idx), n_trials, N_BINS), dtype=np.float32)
    for i, u in enumerate(unit_idx):
        lo = spike_index[u - 1] if u > 0 else 0
        st = spike_times[lo:spike_index[u]]
        pos = np.searchsorted(st, flat_edges).reshape(n_trials, N_BINS + 1)
        fr[i] = (pos[:, 1:] - pos[:, :-1]).astype(np.float32)
    fr /= BIN_SIZE                      # spike counts -> spikes/s
    return fr
```
```python
neural = [np.ascontiguousarray(fr[:, t, :]) for t in range(n_tr)]
```

iii. Step 1 identified the reference `sliding_histogram(..., rate=True)`, which returns
`binSpikes / bin_width` over half-open bins `[c − bw/2, c + bw/2)`. Step 5 Key Decision 2
states "Neural values are firing rates in spikes/s (`count / 0.05`), as in the reference
(`rate=True`)", and Decision 1 states the half-open convention is chosen to match the
reference. Step 10 Check 2 re-derived the spike counts with an independent boolean-mask
implementation for 24 random (session, trial, neuron, bin) tuples and 6 whole-trial totals and
got exact agreement.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept if `units/classification == 'good'` **and** they have a non-empty CCF
annotation `units/anno_name`. No thresholds on individual QC metrics and no firing-rate
threshold are applied. A session with no such unit is dropped (1 session,
`sub-440958_ses-20190216T162508`, in which `classification`/`anno_name` are unset for all
units). This retains 69,453 of 272,227 clusters (25.5 %), mean 401.5 per session (range
90–923), across 173 sessions.

ii.
```python
classification = _decode(u['classification'][:])
anno = _decode(u['anno_name'][:])
keep = (classification == 'good') & (anno != '')
unit_idx = np.where(keep)[0]
if len(unit_idx) == 0:
    return None
```
```python
'neuron_curation': (
    "units labelled 'good' by the region-specific spike-sorting QC classifiers "
    "(units/classification) and having a CCF histology annotation"),
```

iii. Step 1 identified `helper_get_neuron_id_area` in the reference, which keeps only units in
the classifier QC "good units" list (`qc_mode='classifier'`, the parameter actually used in
`Sherlock/preprocess_all_ephys.py`) *and* requires a non-empty CCF annotation (the reference
intersects ephys unit ids with histology unit ids). `units/classification` is the NWB record of
exactly that classifier, described in `ChenLiuEtAl2023_SpikeSortingQC.pdf`. Step 10 Check 3(b)
notes that in this release all 69,453 `good` units already carry an annotation, so the
histology term removes nothing — it is there for faithfulness to the reference. Validation:
the good fraction of Kilosort2 clusters is 25.5 % vs the paper's 25.9 %, and 11 of the 14
per-area good-unit counts published in the data paper are reproduced **exactly** (the residual
490-unit shortfall is entirely isocortical and attributed to a DANDI release version
difference). Step 5 Decision 3 explicitly rejects an extra firing-rate threshold because the
paper applies a 2 Hz threshold only inside one specific AUC analysis, not to the dataset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **go-cue onset**, taken from
`acquisition/BehavioralEvents/go_start_times/timestamps` (one entry per trial). All NWB times
share one session-absolute clock, so no resampling or offset correction is needed: the fixed
relative edge grid is simply added to each trial's go-cue time to give that trial's absolute
window, and spikes are binned against those absolute edges.

ii.
```python
go = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
...
go_v = go[trials]
...
fr = bin_spikes(spike_times, spike_index, unit_idx, go_v)
```
```python
edges = go_times[:, None] + BIN_EDGES[None, :]
flat_edges = np.ascontiguousarray(edges.ravel())
```
```python
'temporal_alignment_event': 'onset of the auditory go cue (t = 0)',
'off_start': OFF_START,      # -2.5
'off_end': OFF_END,          # +1.5
```

iii. Step 1/Step 3 establish that the reference pipeline aligns everything to the go cue
(`task_cue_time`), and the decoder task mandates the same. Step 10 Check 3(c) records the
comparison: "go cue = 0 …; spikes truncated to `[begin_time, end_time]` around it" vs "spike
times and video timestamps both have the go-cue time subtracted" → identical. Alignment was
verified visually (`--show-processing` panel 0,0 overlays the raw spike raster on the binned
rate and marks the go cue, tone onset, trial start and trial stop) and statistically (the
population PSTH shows a sharp go-cue-locked response and left/right divergence after t = 0;
the time-resolved `choice`/`outcome` decoding accuracies rise sharply at t = 0, which a
multi-bin misalignment would smear).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms, fixed. The window `[−2.5, +1.5)` s relative to the go cue is divided into 80
non-overlapping 50 ms bins. The 81-edge grid is built once at module level and reused for every
trial of every session, so every trial has exactly 80 timepoints. There is no sliding /
overlapping window, no upsampling and no rebinning of an intermediate representation: spikes go
straight from raw times into the 50 ms bins, and the tongue video (≈294 Hz) is averaged
directly into the same 50 ms grid. `metadata['time_bin_size']` is 50.0 ms.

ii.
```python
OFF_START = -2.5          # s, signed time from go cue to start of the extracted window
OFF_END = 1.5             # s, signed time from go cue to end of the extracted window
BIN_SIZE = 0.05           # s, width of the (non-overlapping) spike-count bins
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))     # 80
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(N_BINS + 1)  # (81,) relative to go cue
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])      # (80,)
```
```python
'time_bin_size': BIN_SIZE * 1000.0,      # ms
'n_time_bins': N_BINS,
'bin_centers_s': BIN_CENTERS.tolist(),
```

iii. The window and bin width are mandated by the decoder task. Step 4 and Step 10 Check 3(d)
document the deviation from the reference, which used a 40 ms window with a 3.4 ms sliding
stride: "50 ms / stride 50 ms instead of 40 ms / 3.4 ms — mandated by the decoder-task
specification. Same half-open convention and the same spikes→rate conversion. The data paper
shows results are stable for bin widths 20–320 ms." Non-overlapping bins are also required so
that the neural, input and output streams share one common time axis.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `acquisition/BehavioralEvents/sample_start_times/timestamps` (the sample-epoch / instruction
tone onsets), combined with each trial's go-cue time. Because an early lick during the sample
or delay epoch replays the sample epoch, a trial can carry several tone presentations; the AI
takes the **last** tone onset strictly before the trial's go cue.

ii.
```python
sample_start = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
```
```python
# tone onset: last sample-epoch start before the go cue of that trial
j = np.searchsorted(sample_start, go_v - 1e-9) - 1
assert np.all(j >= 0), f'{ident}: trial without a preceding tone onset'
tone_time = sample_start[j]
```

iii. Step 2 documents that `sample_start_times` "has *more* entries than trials because an early
lick triggers a replay of the sample epoch". Step 5 Decision 7: "Tone onset = the *last*
sample-epoch start before the go cue. … the final one is the instruction the animal must
report, and it is always 1.85 s before the go cue (median). Earlier replays fall outside or at
the edge of the window." Step 5 maps it to the reference's `task_sample_time[0]`. The assertion
guards against a trial with no preceding tone; Step 10 Check 5 records that the tone onset lies
inside the trial for all 94,990 trials.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A continuous, time-varying value: for each trial, each of the 80 bin centres (expressed
relative to the go cue) is shifted by the go-cue-to-tone gap, giving seconds elapsed since the
tone at that bin centre. Negative before the tone. Stored as `float32` in row 0 of the
per-trial `(2, 80)` input array. Observed range over the full dataset: [−1.525, 11.894] s
(`go − tone` is 1.85 s in 87.5 % of trials, 0.95 s / 2.45 s in sessions with 0.3 s / 1.8 s
delays, and up to ~10 s on the 3 % of trials with multiple sample-epoch replays).

ii.
```python
# time from tone onset at every bin centre, (n_trials, N_BINS)
time_from_tone = (BIN_CENTERS[None, :] + (go_v - tone_time)[:, None]).astype(np.float32)
```
```python
inputs = [np.stack([time_from_tone[t], photostim[t]]) for t in range(n_tr)]
```

iii. Step 5 specifies `input[0][t] = t_bin_center − t_tone`. No further processing is required
once the correct tone onset is found. The AI notes in `metadata['input_descriptions']`:
"seconds elapsed since the onset of the instruction tone (sample-epoch start) at each bin
centre; negative before the tone. If an early lick triggered a replay of the sample epoch, the
last tone before the go cue is used." Step 10 Check 2 recomputed the value with an independent
Python `max(x for x in sample_starts if x < go)` implementation for 24 spot checks (max abs
diff 2e-7), and the `--show-processing` plot verifies that the trace crosses zero exactly at
the plotted tone-onset marker.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. By construction: the value is evaluated at the centre of the same 80 go-cue-relative bins
used for the firing rates, so bin *k* of the input covers exactly the same interval as bin *k*
of the neural array. The only per-trial quantity is the scalar `go − tone` offset; the grid
itself is shared.

ii.
```python
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(N_BINS + 1)  # (81,) relative to go cue
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])      # (80,)
```
```python
edges = go_times[:, None] + BIN_EDGES[None, :]        # neural
...
time_from_tone = (BIN_CENTERS[None, :] + (go_v - tone_time)[:, None]).astype(np.float32)  # input
```

iii. All NWB event timestamps are on one session-absolute clock (Step 2), so subtracting the
go-cue time is sufficient and no interpolation or offset correction is needed. The
`--show-processing` figure plots the input against the tone-onset and go-cue markers as an
explicit alignment check (Step 7, point 2: "`time_from_tone_onset` crosses zero exactly at the
plotted tone-onset marker").

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The trials-table columns `intervals/trials/photostim_onset` and
`intervals/trials/photostim_duration` (both stored as strings, `'N/A'` on unstimulated trials,
with the onset measured **relative to the trial start**), together with `trials/start_time` and
the trial's go-cue time to re-express them on the go-cue axis.

ii.
```python
ps_onset = _decode(tr['photostim_onset'][:])
ps_dur = _decode(tr['photostim_duration'][:])
```
```python
has_ps = ps_onset[trials] != 'N/A'
if has_ps.any():
    on = np.array([float(x) for x in ps_onset[trials][has_ps]])
    dur = np.array([float(x) for x in ps_dur[trials][has_ps]])
    ps_t0 = start_time[trials][has_ps] + on - go_v[has_ps]   # rel. to go cue
    ps_t1 = ps_t0 + dur
```

iii. Step 5 maps these onto the reference's `task_stimulation[:, 2:4]`. Step 2/Step 5 Decision 8
record that the values were cross-validated against the independent
`acquisition/BehavioralEvents/photostim_start_times`/`_stop_times` streams and agree to
< 10 ms; the trials-table form was preferred because it is directly indexed by trial and
handles the unstimulated case explicitly with `'N/A'`.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1) time series, `float32`, in row 1 of the input array. The stimulation interval
`[start_time + onset, +duration]` is expressed relative to the go cue, and a bin is set to 1
if it **overlaps** that interval (`edge_k < t1` and `edge_{k+1} > t0`), 0 otherwise. Trials with
`photostim_onset == 'N/A'` keep an all-zero row. Because a 0.5 s stimulus generally straddles
bin boundaries, a stimulated trial typically has 11 bins on (0.55 s).

ii.
```python
photostim = np.zeros((len(trials), N_BINS), dtype=np.float32)
has_ps = ps_onset[trials] != 'N/A'
if has_ps.any():
    ...
    # a bin is "on" if it overlaps the stimulation interval
    ov = (BIN_EDGES[None, :-1] < ps_t1[:, None]) & (BIN_EDGES[None, 1:] > ps_t0[:, None])
    photostim[has_ps] = ov.astype(np.float32)
```
```python
'photostim_on': (
    '1 if the 0.5 s ALM photoinhibition stimulus overlapped the bin, else 0.'),
```

iii. The instructions ask for "whether photostimulation is on at every time point", i.e. a
binary time-varying input rather than a per-trial flag. Step 2 records that the duration is
always 0.5 s and that the onset is −0.5 s relative to the go cue in 122 sessions (late delay)
and −1.2 s in 46 (early delay), always ending before the go cue. Step 7 point 3 documents the
overlap convention and its consequence ("exactly 1 during the plotted stimulation interval
(11 bins = 0.55 s for a 0.5 s stimulus that straddles bin boundaries)"). Step 10 Check 2
recomputed the input with an explicit per-bin overlap loop for 24 spot checks — exact match —
and Check 5 verified that `'N/A'` trials get an all-zero input.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The raw onset is stored relative to trial start, so it is converted to the go-cue axis by
adding `trials.start_time` (making it session-absolute) and subtracting the trial's go-cue
time. It is then compared directly against the same `BIN_EDGES` grid that defines the neural
bins, so bin *k* of the photostim input covers the same interval as bin *k* of the firing
rates.

ii.
```python
ps_t0 = start_time[trials][has_ps] + on - go_v[has_ps]   # rel. to go cue
ps_t1 = ps_t0 + dur
ov = (BIN_EDGES[None, :-1] < ps_t1[:, None]) & (BIN_EDGES[None, 1:] > ps_t0[:, None])
```

iii. Everything is on one session clock, so a single subtraction suffices. The conversion was
independently validated against `BehavioralEvents/photostim_{start,stop}_times` (< 10 ms
agreement, Step 5 Decision 8), and the `--show-processing` all-trial photostim raster shows the
expected `[−1.2, −0.7]` s (and a few sample-epoch) blocks, always ending before the go cue
(Step 7 point 3).

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no lick-direction column in the file, so choice is derived from two trials-table
columns: `trial_instruction` (`'left'` / `'right'`, the side the tone instructed) and `outcome`
(`'hit'` / `'miss'` / `'ignore'`). A hit means the animal licked the instructed side, a miss
means it licked the other side, and an `ignore` means it never licked in the response window.

ii.
```python
outcome = _decode(tr['outcome'][:])
instruction = _decode(tr['trial_instruction'][:])
```
```python
oc = outcome[trials]
ins = instruction[trials]
choice = np.full(len(trials), 2, dtype=np.int64)              # 2 = no lick
licked_left = ((oc == 'hit') & (ins == 'left')) | ((oc == 'miss') & (ins == 'right'))
licked_right = ((oc == 'hit') & (ins == 'right')) | ((oc == 'miss') & (ins == 'left'))
choice[licked_left] = 0
choice[licked_right] = 1
```

iii. Step 5 Decision 9: "Choice derived from `outcome` × `trial_instruction` rather than from
raw lick times: this is the experiment's own definition (`behavior_report` in the reference),
is exact, and does not depend on a lick-detection heuristic." Step 10 Check 2 cross-validated
the derived choice against the **raw lick event streams** (side of the first lick in
`[go, go+1.5]`): 3,669/3,674 lick trials and 447/449 no-lick trials agreed; all five
disagreements were individually inspected and are `ignore` trials containing a single lick
1–47 ms after the go cue (licks already in progress, which the behavioural state machine did
not count as a response), so the trials-table `outcome` was taken as authoritative. Excluding
those boundary artifacts the agreement is 100 % (5,698/5,698 response trials over 12 random
sessions).

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded as `0 = left`, `1 = right`, `2 = no lick`, with `output_values[0] =
['left', 'right', 'no lick']`. It is one value per trial, so it is broadcast across all 80 bins
and written into row 0 of the per-trial `(4, 80)` `int64` output array.

ii.
```python
OUTPUT_NAMES = ['choice', 'outcome', 'early_lick', 'tongue_y_position']
OUTPUT_VALUES = [
    ['left', 'right', 'no lick'],
    ...
]
```
```python
outputs = [np.stack([np.full(N_BINS, choice[t]), np.full(N_BINS, outcome_c[t]),
                     np.full(N_BINS, early_c[t]), tongue_c[t]]).astype(np.int64)
           for t in range(n_tr)]
```

iii. The 3-class coding follows the decoder-output specification (left, right, no lick). Step 5
notes that "the three per-trial variables are broadcast over time, as the spec asks for
time-varying outputs 'if at all possible'", which also keeps all four outputs in a single
`(n_output, n_timepoints)` array with a common time axis. The resulting distribution
(left 0.429 / right 0.422 / no lick 0.148) was checked against the raw trials tables and is
balanced as expected from a ~50/50 left/right instruction schedule.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `intervals/trials/outcome` column, which already contains exactly the
three strings `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
outcome = _decode(tr['outcome'][:])
...
oc = outcome[trials]
```

iii. Step 5 maps `intervals/trials/outcome` to the reference's `behavior_report`. The column
stores exactly the three categories the decoder specification asks for, so no derivation is
needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to `0 = ignore`, `1 = miss`, `2 = hit` with `np.select`, with an
assertion that no unexpected label occurs (the `default=-1` sentinel is checked). The result is
broadcast across all 80 bins into row 1 of the output array. `output_values[1] =
['ignore', 'miss', 'hit']`. The full-dataset distribution is ignore 0.148 / miss 0.167 /
hit 0.685, matching the raw trials tables (0.148 / 0.165 / 0.687).

ii.
```python
outcome_c = np.select([oc == 'ignore', oc == 'miss', oc == 'hit'], [0, 1, 2],
                      default=-1).astype(np.int64)
assert np.all(outcome_c >= 0), f'{ident}: unexpected outcome value'
```
```python
'outcome': 'ignore = no lick in the answer period, miss = incorrect lick, hit = correct lick',
```

iii. The code assignment follows the ordering given in the instructions (ignore, miss, hit).
Broadcast across bins for the same reason as `choice`. Step 10 Check 2 recomputed `outcome`
(along with `early_lick` and `choice`) from the trials table for **every** trial of 8 random
sessions with exact agreement; Step 9 compares the converted distribution against the raw NWB
distribution.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. The `intervals/trials/early_lick` column, which holds the strings `'early'` / `'no early'`.

ii.
```python
early = _decode(tr['early_lick'][:])
```

iii. Step 5 maps it to the reference's `behavior_early_report`. The table flags early licking
explicitly, so no derivation is needed. The flagged lick occurs during the sample or delay
epoch, i.e. inside the −2.5 s window, even though the flag itself is per trial.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A simple boolean comparison mapped to `0 = no`, `1 = yes`, broadcast over all 80 bins into
row 2 of the output array, with `output_values[2] = ['no', 'yes']`. The full-dataset
distribution is no 0.884 / yes 0.116, consistent with the ~11.4 % early rate measured directly
in the raw trials tables.

ii.
```python
early_c = (early[trials] == 'early').astype(np.int64)
```
```python
outputs = [np.stack([..., np.full(N_BINS, early_c[t]), ...]).astype(np.int64)
           for t in range(n_tr)]
```
```python
'early_lick': 'whether the animal licked during the sample or delay epoch',
```

iii. The 0/1 coding follows the instructions ("no, yes"). Broadcast across bins like the other
per-trial outputs. Verified for every trial of 8 random sessions in Step 10 Check 2.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is an
`(n_frames, 3)` DeepLabCut array of `(tongue_x, tongue_y, tongue_likelihood)` with matching
`timestamps` (≈294 Hz, session-absolute seconds). Column 1 (`y`) is the value; column 2
(likelihood) decides whether the tongue is visible in that frame.

ii.
```python
tongue_key = 'acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking'
tdata = f[tongue_key + '/data'][:]
tts = f[tongue_key + '/timestamps'][:]
ty, tvis = tongue_per_bin(tdata, tts, go_v)
```
```python
ycol = track_data[:, 1]
vis_all = track_data[:, 2] > TONGUE_LIKELIHOOD_THRESH
```

iii. Step 1 notes that the reference's marker analysis keeps `tongue_x`, `tongue_y` from
`camera_0_side` — "the same stream that the NWB files expose as
`acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`". Step 5 maps it to
`tracking.camera_0_side.tongue_y` / `..._likelihood`. It is the only tongue measurement in the
file.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Three stages. (1) Frames with likelihood ≤ 0.5 are treated as "tongue not visible" and
excluded — the tracker still emits a position when the tongue is retracted. (2) For each trial,
the frames inside `[go − 2.5, go + 1.5)` are assigned to the 80 bins by their offset from
`go + OFF_START` and the visible ones are averaged per bin with `np.bincount`, producing a
per-bin mean `y` and a per-bin visibility flag; a bin with no visible frame stays NaN /
not-visible. (3) The per-bin means are discretised against the session's 40th and 60th
percentiles (see 8-c). ~25 % of all bins have a visible tongue.

ii.
```python
TONGUE_LIKELIHOOD_THRESH = 0.5   # DLC likelihood is bimodal; any value in (0.01,0.99) works
TONGUE_PCTL = (40.0, 60.0)       # percentiles used to discretise tongue y-position
```
```python
lo = np.searchsorted(track_ts, go_times + OFF_START, side='left')
hi = np.searchsorted(track_ts, go_times + OFF_END, side='left')
for t in range(n_trials):
    a, b = lo[t], hi[t]
    if b <= a:
        continue
    rel = track_ts[a:b] - go_times[t]
    b_idx = np.floor((rel - OFF_START) / BIN_SIZE).astype(np.int64)
    np.clip(b_idx, 0, N_BINS - 1, out=b_idx)
    v = vis_all[a:b]
    if not v.any():
        continue
    cnt = np.bincount(b_idx[v], minlength=N_BINS)
    tot = np.bincount(b_idx[v], weights=ycol[a:b][v], minlength=N_BINS)
    m = cnt > 0
    y[t, m] = tot[m] / cnt[m]
    visible[t, m] = True
```

iii. Step 5 Decision 10: the DeepLabCut likelihood is bimodal (89 % < 0.01, 10.5 % > 0.99) so
"any threshold in (0.01, 0.99) gives the same answer (10.586 % vs 10.517 % of frames for 0.5 vs
0.9)". Step 5 Decision 11: non-visible frames must be excluded because their `y` is meaningless
("ranges −5…364 vs 239…327 when visible), so including it would corrupt the percentiles". Step
2 records an independent visibility sanity check: 99.8 % of recorded lick events fall on frames
with likelihood > 0.9. Averaging within the bin is the natural way to bring a 294 Hz stream onto
the 50 ms neural grid.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Four classes, exactly as specified: `0` if the per-bin mean `y` is below the 40th percentile,
`1` if between the 40th and 60th percentiles (inclusive), `2` if above the 60th percentile, and
`3` if no visible tongue frame fell in the bin. The percentiles are computed **per session**
over the per-bin means of the visible bins only. Sessions with no visible tongue at all get
class 3 everywhere. The realised distribution is 0.1006 / 0.0503 / 0.1006 / 0.7486 — i.e. an
exact 2:1:2 split among the 25 % visible bins, confirming the 40/20/40 split.

ii.
```python
def discretize_tongue(y, visible):
    """Discretise per-bin tongue y-position using per-session percentiles.

    0: y < 40th percentile, 1: 40th-60th percentile, 2: y > 60th percentile,
    3: tongue not visible.  Percentiles are taken over the visible bins of the session
    (y is meaningless where the tongue is not visible).
    """
    cls = np.full(y.shape, 3, dtype=np.int64)
    if visible.any():
        vals = y[visible]
        p40, p60 = np.percentile(vals, TONGUE_PCTL)
        cls[visible & (y < p40)] = 0
        cls[visible & (y >= p40) & (y <= p60)] = 1
        cls[visible & (y > p60)] = 2
    else:
        p40 = p60 = np.nan
    return cls, (p40, p60)
```
```python
OUTPUT_VALUES = [..., ['<40th pct', '40th-60th pct', '>60th pct', 'not visible']]
```

iii. The 40th/60th split, the per-session scope and the "not visible" fourth class are all
specified by the decoder-task instructions. The choice to take percentiles over the *visible
per-bin means* (rather than over raw frames, or over all bins) follows from Decision 11: the
percentiles must be defined on the same quantity that gets discretised, and non-visible `y`
values are meaningless. Step 7 confirms "the tongue classes 0:1:2 are in the expected 2:1:2
ratio (40 %/20 %/40 % of the *visible* bins), confirming the percentile discretisation", and
Step 10 Check 2 recomputed the whole output from the raw DeepLabCut stream with an independent
`np.digitize`-based implementation and independently computed percentiles — 0 mismatching bins
out of 336,000 across 8 sessions. The per-session percentile values are stored in
`session_info['tongue_pctl']`.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps are on the same session-absolute clock as the spikes and events, so
each trial's frame range is found by `np.searchsorted` on the camera timestamps at
`go + OFF_START` and `go + OFF_END`, and frames are assigned to bins by their offset from
`go + OFF_START` — the identical go-cue-relative 50 ms grid used for the firing rates (bin index
clipped to `[0, 79]` to guard against float edge effects). No interpolation or resampling.

ii.
```python
lo = np.searchsorted(track_ts, go_times + OFF_START, side='left')
hi = np.searchsorted(track_ts, go_times + OFF_END, side='left')
for t in range(n_trials):
    a, b = lo[t], hi[t]
    ...
    rel = track_ts[a:b] - go_times[t]
    b_idx = np.floor((rel - OFF_START) / BIN_SIZE).astype(np.int64)
    np.clip(b_idx, 0, N_BINS - 1, out=b_idx)
```

iii. Step 10 Check 3(c) records that the reference likewise aligns markers by subtracting the go
cue time (`times_for_frames − go_times`), so this is identical. The AI documents that the video
is trial-gated (Step 2: "Video is recorded **per trial**: timestamps restart at each trial's
`start_time`, with gaps in between"), so on trials where the go cue falls < 2.5 s after trial
start the leading bins have no frames and fall into class 3; and Step 10 Check 5 lists the
edge cases (1 session with essentially no usable video, 5 sessions where the camera segment
ends near the go cue, 1 session with a DeepLabCut tracking failure). Alignment is checked
visually in the `--show-processing` figure (raw frames, per-bin means, percentile lines and
class labels on one axis) and statistically: "Tongue visibility is ~2 % before the go cue and
jumps to ~80 % just after it" (Step 7).

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Each defect is handled according to what it means:
 - **Session never quality-controlled** (`classification` / `anno_name` unset for every unit):
   `_decode` turns the non-string entries into `''`/`'nan'`, no unit passes the filter, and the
   session is dropped (1 session), leaving exactly 173.
 - **Trials outside the ephys recording**: excluded via `units/obs_intervals` (8–9 sessions).
 - **`obs_intervals` over-reporting coverage by one trial**: detected as "no spike from any
   neuron in the whole 4 s window" and dropped (2 trials dataset-wide).
 - **Water-delivery trials** (`auto_water`, `free_water`): excluded as not being genuine
   behavioural reports.
 - **Trials with no photostimulation** (`'N/A'`): all-zero photostim input rather than NaN.
 - **Frames with no tracked tongue**: excluded from the bin mean; a bin with no visible frame
   becomes the explicit "not visible" class 3; a session with no visible tongue at all gets
   class 3 everywhere and NaN percentiles.
 - **Trial-segmented spike export** (spikes exist only within `[trial start, trial stop]`, so
   ~95 % of `miss` trials have no spikes after ≈ +0.8 s and 3 % of trials none before ≈ −2.2 s):
   those bins are reported as a firing rate of 0 rather than dropping the trials, and the
   consequence is quantified rather than ignored.
 - **Defensive assertions**: one go cue per trial, a tone onset before every go cue, only known
   `outcome` labels; a worker that raises is caught, its traceback printed, and the run exits
   non-zero rather than silently producing a partial dataset.

ii.
```python
def _decode(arr):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])
```
```python
keep = (classification == 'good') & (anno != '')
unit_idx = np.where(keep)[0]
if len(unit_idx) == 0:
    return None
```
```python
has_data = fr.sum(axis=(0, 2)) > 0
n_zero_trials = int((~has_data).sum())
if n_zero_trials:
    fr = fr[:, has_data, :]
    trials = trials[has_data]
    go_v = go_v[has_data]
    if len(trials) < 2:
        return None
```
```python
cls = np.full(y.shape, 3, dtype=np.int64)
if visible.any():
    ...
else:
    p40 = p60 = np.nan
```
```python
'known_limitations': (
    'The NWB export stores spikes only within [trial start, trial stop]. On error '
    '(miss) trials the recorded interval ends ~0.8 s after the go cue, so the last '
    'bins of those trials contain no spikes; 3% of trials likewise lack data before '
    '~-2.2 s. Those bins are reported as a firing rate of 0.'),
```
```python
def _worker(args):
    fname, want_raw = args
    try:
        return process_session(fname, want_raw=want_raw)
    except Exception as exc:        # keep the run alive, report loudly
        import traceback
        traceback.print_exc()
        return {'error': f'{fname}: {exc}'}
```

iii. Step 5 Decision 6 and Step 10 Check 5 give the reasoning: where nothing was recorded the
session or trial is excluded (emitting it would fabricate 4 s of 0 Hz activity); where the
measurement legitimately has no value, as with a retracted tongue, it is represented as an
explicit category rather than imputed. The zero-padding decision is defended quantitatively:
"The alternative — dropping every trial whose window is not fully covered — would delete 95 % of
all `miss` trials and make the `outcome` output nearly degenerate, so it is rejected", and the
AI then *measured* the artifact by retraining the decoder while masking the bins after the end
of the spike record — held-out `outcome` accuracy fell only from 0.659 to 0.652, and the other
three outputs were unchanged. Step 10 Check 5 also documents deliberate non-actions: the
`is_good_trials` flag is not used because spikes are present on the flagged trials with
comparable rates (3.66 vs 3.86 spikes/s), and the one session with a DeepLabCut failure
(`SC027_20190803_150200_s21`, y stuck near 36 px) is left in because only its tongue labels are
affected.

## 10-a. What are the most time-consuming steps of the code?

i. The full conversion takes **41 s** of wall clock for 174 sessions with 16 worker processes;
per-session CPU time is 0.3–2.4 s and scales roughly with unit count. Within a session the cost
is dominated by HDF5 I/O — the bulk read of `units/spike_times` (up to ~11.5 M doubles) and of
the tongue tracking array (~680 k × 3) — followed by the per-unit `np.searchsorted` loop in
`bin_spikes` (one binary search per bin edge per unit, i.e. `n_units × n_trials × 81` searches).
The per-trial Python loop in `tongue_per_bin` is the third cost. Outside the parallel section,
pickling the 11.89 GB result takes 15 s, a third of the total run time. The script prints
per-session and cumulative timing so the bottleneck is visible in the log.

ii.
```python
spike_index = u['spike_times_index'][:]
spike_times = u['spike_times'][:]
fr = bin_spikes(spike_times, spike_index, unit_idx, go_v)
del spike_times
```
```python
for i, u in enumerate(unit_idx):
    lo = spike_index[u - 1] if u > 0 else 0
    st = spike_times[lo:spike_index[u]]
    pos = np.searchsorted(st, flat_edges).reshape(n_trials, N_BINS + 1)
```
```python
print(f'[{i + 1}/{len(files)}] {n["session_id"]}: {n["n_neurons"]} neurons, '
      f'{n["n_trials"]}/{n["n_trials_table"]} trials, '
      f'{n["mean_rate"]:.2f} spikes/s, {n["proc_time"]:.1f}s  [{el:.0f}s]', flush=True)
```
```python
print(f'  wrote {os.path.getsize(args.outfile) / 1e9:.2f} GB in {time.time() - t:.1f} s')
```

iii. Step 6/Step 7 of CONVERSION_NOTES identify these costs and the two optimisations applied:
one `searchsorted` per unit over all trial edges instead of an O(units × trials × spikes) loop,
and one bulk read of `units/spike_times` per session instead of ~400 small HDF5 reads, plus
16 worker processes ("~4.5× wall-clock reduction"). The pre-run estimate was "< 1 minute";
the actual 41 s is far inside the instructions' 15-minute budget, so the AI concluded "no
further optimisation needed".

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four Python loops remain, none of which is on the critical path:
 - `bin_spikes`: the per-unit loop. It is already vectorised over trials (all `n_trials × 81`
   edges are searched in one call), and it cannot be collapsed further because each unit has a
   different number of spikes, so there is no single sorted array to search — this is inherent
   to NWB's ragged storage.
 - `tongue_per_bin`: the per-trial loop. This *could* be vectorised with a single global bin
   index plus one `np.bincount` over all frames of the session (which is how the human reference
   computes its session-wide percentiles), but it runs over trials rather than frames.
 - `assign_brain_regions`: a Python loop over every unit calling `annotation_to_group`, which
   itself scans up to 13 group lists × dozens of substrings per unit. This is ~70 k units ×
   O(300) substring tests dataset-wide and could be replaced by building the map once over the
   293 *distinct* annotation strings and then using a dictionary lookup or `np.unique(...,
   return_inverse=True)`.
 - The `observed` fallback branch loops over every good unit and issues one HDF5 read of
   `obs_intervals` per unit, even though the code's own comment states that "all units of a
   session share the same contiguous observed trial set" — a single unit's intervals would
   suffice (which is what the reference does).
 - Minor: `_decode` is a Python list comprehension over every trial/unit string column, and
   `[float(x) for x in ps_onset[...]]` parses the photostim strings one at a time.

ii.
```python
for i, u in enumerate(unit_idx):            # per-unit, inherently ragged
    ...
for t in range(n_trials):                   # per-trial tongue binning
    ...
def assign_brain_regions(anno, ccf_ap):
    ...
    for i, name in enumerate(anno):         # per-unit substring scan
        grp = annotation_to_group(name)
```
```python
oi = u['obs_intervals']
for k in unit_idx:                          # per-unit HDF5 read, only 1 unit needed
    o = oi[(oii[k] - n_obs[k]):oii[k]]
    idx = np.searchsorted(start_time, o[:, 0] + 1e-9) - 1
    observed[idx[idx >= 0]] = True
```

iii. CONVERSION_NOTES Step 6 documents only the two loops that were deliberately optimised away
(the naive per-trial/per-unit spike loop and the per-unit HDF5 reads) and justifies the
remaining per-unit `searchsorted` loop implicitly by its complexity analysis
(`O(n_units · n_trials · 81 · log n_spikes)`). The notes do not discuss the
`assign_brain_regions` or `observed`-fallback loops; the AI's general justification is that the
41 s total run time is far under budget, so no further vectorisation was pursued.

## 10-c. What processing does the code repeat multiple times?

i. Very little is recomputed. Each NWB file is opened once per conversion run and every quantity
derived from it is computed once; the bin grid (`BIN_EDGES`, `BIN_CENTERS`) and the region
lookup tables are module-level constants reused for every trial and session; the tongue
percentiles are per session and are computed inside the same single pass. The repetitions that
do exist are small:
 - `annotation_to_group` is re-evaluated for every unit even though there are only 293 distinct
   annotation strings in the whole release, so the same substring scan is repeated tens of
   thousands of times.
 - In the `observed` fallback branch the same contiguous observed-trial set is re-derived once
   per good unit.
 - `make_processing_plot` re-opens the NWB file (`h5py.File(raw['fname'])`) to re-read the spike
   times that `process_session` already read, and re-derives the per-frame visibility mask and
   the relative frame times that `tongue_per_bin` already computed — but only in
   `--show-processing` mode, for at most 2 sessions.
 - The per-trial `np.full(N_BINS, choice[t])` / `np.stack` list comprehensions rebuild the same
   broadcast arrays per trial rather than filling one `(n_trials, 4, 80)` array and slicing it.

ii.
```python
for i, name in enumerate(anno):
    grp = annotation_to_group(name)     # same 293 strings re-scanned ~70k times
```
```python
with h5py.File(raw['fname'], 'r') as f:     # plot re-opens the file already read
    st_all = f['units/spike_times']
```
```python
outputs = [np.stack([np.full(N_BINS, choice[t]), np.full(N_BINS, outcome_c[t]),
                     np.full(N_BINS, early_c[t]), tongue_c[t]]).astype(np.int64)
           for t in range(n_tr)]
```

iii. Step 6 explicitly lists "avoid unnecessary file I/O" as a design goal and records the one
bulk read per session as the fix; the file re-open in the plotting path is accepted because it
only runs for ≤2 sessions in a diagnostic mode and keeps the memory-heavy raw spike buffer out
of the returned result. Nothing in the notes addresses the repeated annotation lookups.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Most of what the script computes ends up in the output, but several things are computed and
then never used by the decoder:
 - **Diagnostic statistics in `session_info`**: `mean_rate`, `hemisphere_left_frac`,
   `n_units_total`, `n_units_good`, `n_trials_observed`, `n_excluded_water`,
   `n_excluded_no_spikes`, `trial_index`, `tongue_pctl`, `frac_tongue_visible`, `proc_time`.
   Computing `hemisphere_left_frac` requires reading the electrode `x` (ML) column purely for
   this statistic — the ML coordinate plays no role in the region assignment, which uses only
   AP. `stop_time` is read for every trial and used only in the plotting path.
 - **Redundant filter term**: the `anno != ''` condition, which the AI itself verified removes
   nothing in this release.
 - **`visible` array**: `tongue_per_bin` returns an explicit boolean visibility mask that is
   fully redundant with `~np.isnan(y)`.
 - **Storage inflation**: outputs are cast to `int64` although they take values 0–3 (`int8`
   would be 8× smaller), and `metadata['bin_centers_s']` duplicates information already implied
   by `off_start`/`off_end`/`time_bin_size`.
 - **`--show-processing` mode** computes and retains a large `raw` dict (the whole tracking
   array, spike index, per-trial times) and renders an 8-panel figure; none of this reaches the
   pickle. It is gated to ≤2 sessions.
 - Nothing computed is *thrown away* inside `process_session` itself: every derived array
   (`fr`, `time_from_tone`, `photostim`, `choice`, `outcome_c`, `early_c`, `tongue_c`,
   `region_idx`) goes into the returned result.

ii.
```python
ccf_ml = etab['x'][:][el[unit_idx]]      # only used for hemisphere_left_frac
ccf_ap = etab['z'][:][el[unit_idx]]
```
```python
info = dict(..., mean_rate=float(fr.mean()), tongue_pctl=[float(pctl[0]), float(pctl[1])],
            frac_tongue_visible=float(tvis.mean()),
            hemisphere_left_frac=float(np.mean(ccf_ml >= CCF_MIDLINE_ML)),
            proc_time=time.time() - t0)
```
```python
outputs = [np.stack([...]).astype(np.int64) for t in range(n_tr)]   # int8 would suffice
```
```python
raw = None
if want_raw:
    raw = dict(track_data=tdata, track_ts=tts, go=go_v, tone=tone_time, ...)
```

iii. CONVERSION_NOTES does not frame any of this as waste. The per-session diagnostics are
deliberate: Step 9's trial-accounting table and Step 10's edge-case audit are computed from
exactly these `session_info` fields, and the instructions ask for extra metadata fields such as
`session_info`. The `anno != ''` term is kept for faithfulness to the reference's histology
intersection even though it is a no-op here (Step 10 Check 3b). The hemisphere fraction is
recorded because the reference splits units by hemisphere at CCF ML = 5700 µm, while the
converted `brain_regions` are hemisphere-agnostic to match the paper's published counts
(Step 5 Decision 12). Step 6 states the dtype choices ("Neural data is stored as `float32`;
outputs as `int64`; inputs as `float32`") without justifying `int64` for the outputs.
