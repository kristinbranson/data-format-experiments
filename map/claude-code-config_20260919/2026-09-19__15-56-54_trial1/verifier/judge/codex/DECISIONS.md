# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the dataset by globbing all NWB files under `/app/data/sub-*/*.nwb`, then processing each file as one session. Within `process_session`, it opens the NWB file with `h5py` and reads the needed HDF5 groups directly: subject metadata, the trials table, behavioral event timestamps, tongue-tracking time series, and the units table. It can process sessions in parallel with a worker pool.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
```

```python
with h5py.File(fname, 'r') as f:
    ident = f['identifier'][()].decode()
    mouse = f['general/subject/description'][()].decode().strip()
    subject_id = f['general/subject/subject_id'][()].decode().strip()

    tr = f['intervals/trials']
    ...
    go = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
    sample_start = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
    ...
    u = f['units']
```

iii. In `CONVERSION_NOTES.md` Step 2, the AI states that the NWB release is organized as one file per recording session and that all probes of a session are already merged into that file. In Step 8 it also justifies direct `h5py` access as a speed choice so each large NWB session can be read in one bulk pass.

## 1-b. How are the data split into subjects?

i. The AI splits sessions into subjects using `general/subject/description`, i.e. the mouse name such as `SC015`, not the numeric `subject_id`. At assembly time it takes the sorted unique mouse names and creates `subject_idx` from those names.

ii.
```python
mouse = f['general/subject/description'][()].decode().strip()
subject_id = f['general/subject/subject_id'][()].decode().strip()
```

```python
subjects = sorted({r['info']['mouse'] for r in kept})
sub_index = {s: i for i, s in enumerate(subjects)}
...
'subjects': subjects,
'subject_idx': np.array([sub_index[r['info']['mouse']] for r in kept], dtype=np.int64),
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the mapping plan explicitly says `general/subject/description` (`SC015`) is used for `subjects`, while `subject_id` is preserved only in per-session metadata. The AI appears to prefer the paper-facing mouse names over the numeric NWB IDs.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. Session order is the sorted file order, and each session is identified by the NWB `identifier`.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
```

```python
ident = f['identifier'][()].decode()
```

```python
'session_info': [r['info'] for r in kept],
```

iii. In Step 2 of `CONVERSION_NOTES.md`, the AI documents that `/app/data` has one `behavior+ecephys+ogen.nwb` file per recording session. That file boundary is therefore the session boundary.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table `intervals/trials`. The AI assumes one table row per behavioral trial and verifies that `go_start_times` has exactly one timestamp per trial-table row. After later quality filtering, the surviving trial indices are used to index all trial-wise arrays.

ii.
```python
tr = f['intervals/trials']
n_trials_table = len(tr['id'])
...
go = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
assert len(go) == n_trials_table, f'{ident}: go cue count != trial count'
```

```python
valid = observed & (auto_water == 0) & (free_water == 0)
trials = np.where(valid)[0]
```

iii. In Step 2 of `CONVERSION_NOTES.md`, the AI says `go_start_times` has exactly one entry per trial while other event streams can replay within a trial. That is the stated reason for using the trials table as the authoritative trial split.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials in three stages. First, it uses `units/obs_intervals` to decide whether the ephys recording actually covered a trial. Second, it excludes `auto_water` and `free_water` trials. Third, after spike binning it drops any remaining trial whose entire `[-2.5, 1.5)` window has zero spikes across all kept neurons, treating that as recording-coverage failure. Sessions with fewer than 2 remaining trials are dropped.

ii.
```python
oii = u['obs_intervals_index'][:]
n_obs = np.diff(np.concatenate([[0], oii]))
n_obs_good = n_obs[unit_idx]
observed = np.zeros(n_trials_table, dtype=bool)
if np.all(n_obs_good == n_trials_table):
    observed[:] = True
else:
    oi = u['obs_intervals']
    for k in unit_idx:
        o = oi[(oii[k] - n_obs[k]):oii[k]]
        idx = np.searchsorted(start_time, o[:, 0] + 1e-9) - 1
        observed[idx[idx >= 0]] = True
```

```python
valid = observed & (auto_water == 0) & (free_water == 0)
trials = np.where(valid)[0]
if len(trials) < 2:
    return None
```

```python
has_data = fr.sum(axis=(0, 2)) > 0
...
if len(trials) < 2:
    return None
```

iii. The AI’s justification in `CONVERSION_NOTES.md` Step 5 is that it wants the non-conflicting parts of the reference “regular trial” filter while retaining early-lick, no-response, and photostim trials because those are decoder outputs/inputs. In Step 10 it separately justifies dropping all-zero trials as a known NWB coverage error where `obs_intervals` can over-report by one trial.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` data is derived from `units/spike_times` and `units/spike_times_index`, with `go_start_times` used to place the bin edges. Unit-selection metadata comes from `units/classification` and `units/anno_name`.

ii.
```python
spike_index = u['spike_times_index'][:]
spike_times = u['spike_times'][:]
fr = bin_spikes(spike_times, spike_index, unit_idx, go_v)
```

```python
go = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the variable-mapping table says `units/spike_times` is the NWB equivalent of the reference ephys signal and that the go cue is the shared alignment event.

## 2-b. How is the `neural` data processed?

i. The AI converts spike times into firing rates in spikes/s. It bins each kept unit into 80 non-overlapping 50 ms bins around each trial’s go cue using `np.searchsorted`, computes spike counts by differencing consecutive edge counts, and divides by bin width.

ii.
```python
def bin_spikes(spike_times, spike_index, unit_idx, go_times):
    ...
    edges = go_times[:, None] + BIN_EDGES[None, :]
    flat_edges = np.ascontiguousarray(edges.ravel())

    fr = np.empty((len(unit_idx), n_trials, N_BINS), dtype=np.float32)
    for i, u in enumerate(unit_idx):
        lo = spike_index[u - 1] if u > 0 else 0
        st = spike_times[lo:spike_index[u]]
        pos = np.searchsorted(st, flat_edges).reshape(n_trials, N_BINS + 1)
        fr[i] = (pos[:, 1:] - pos[:, :-1]).astype(np.float32)
    fr /= BIN_SIZE
    return fr
```

iii. `CONVERSION_NOTES.md` Step 1 and Step 5 say the reference code bins spikes with `sliding_histogram(..., rate=True)` and that the decoder task mandates 50 ms bins. The AI therefore keeps the firing-rate representation but swaps in 50 ms non-overlapping bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units with `classification == 'good'` and a non-empty CCF annotation string `anno_name != ''`. If no such units remain, the whole session is dropped.

ii.
```python
classification = _decode(u['classification'][:])
anno = _decode(u['anno_name'][:])
keep = (classification == 'good') & (anno != '')
unit_idx = np.where(keep)[0]
if len(unit_idx) == 0:
    return None
```

iii. In `CONVERSION_NOTES.md` Step 1 and Step 5, the AI says the reference pipeline uses classifier-based “good” units and intersects with histology-annotated units. It treats `anno_name` as the NWB proxy for “has histology”.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to go-cue onset. It takes each retained trial’s absolute go-cue timestamp and adds the shared relative bin-edge grid `[-2.5, 1.5]` to build absolute bin edges for that trial.

ii.
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_SIZE = 0.05
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(N_BINS + 1)
```

```python
go_v = go[trials]
fr = bin_spikes(spike_times, spike_index, unit_idx, go_v)
```

iii. The AI repeatedly notes in `CONVERSION_NOTES.md` that the reference aligns everything to the go cue and that NWB timestamps are on one session-wide clock, so no extra synchronization offset is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 50 ms bins, giving 80 bins over a 4 s window from -2.5 s to +1.5 s relative to the go cue. No smoothing or second-stage temporal rebinning is applied.

ii.
```python
BIN_SIZE = 0.05
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(N_BINS + 1)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
```

iii. In `CONVERSION_NOTES.md` Step 4 and Step 5, the AI says this differs from the paper’s default spike binning only because the task instructions explicitly require 50 ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times` and `go_start_times`. For each trial, the AI chooses the last sample-epoch start before the trial’s go cue and treats that as the instruction-tone onset.

ii.
```python
sample_start = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
...
j = np.searchsorted(sample_start, go_v - 1e-9) - 1
...
tone_time = sample_start[j]
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says early licks can replay the sample epoch, so a trial can contain several sample starts. It therefore uses the final one before the go cue as the tone that actually instructed the animal.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes time from tone onset at each bin center as `bin_center + (go_cue_time - tone_time)`. This yields a continuous per-bin ramp in seconds for every trial.

ii.
```python
time_from_tone = (BIN_CENTERS[None, :] + (go_v - tone_time)[:, None]).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly defines this input as a continuous decoder input rather than a binary event indicator.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on the same 80 go-cue-aligned bin centers that define the neural matrix. The input and neural time axes are therefore identical trial by trial.

ii.
```python
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
...
time_from_tone = (BIN_CENTERS[None, :] + (go_v - tone_time)[:, None]).astype(np.float32)
```

```python
fr = bin_spikes(spike_times, spike_index, unit_idx, go_v)
```

iii. The AI’s notes describe the go cue as the common alignment frame for all streams, with bin centers reused for both neural and decoder inputs.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from `intervals/trials/photostim_onset` and `photostim_duration`, together with `start_time` and the go cue so the stimulation interval can be expressed on the go-cue-relative axis.

ii.
```python
ps_onset = _decode(tr['photostim_onset'][:])
ps_dur = _decode(tr['photostim_duration'][:])
...
has_ps = ps_onset[trials] != 'N/A'
on = np.array([float(x) for x in ps_onset[trials][has_ps]])
dur = np.array([float(x) for x in ps_dur[trials][has_ps]])
ps_t0 = start_time[trials][has_ps] + on - go_v[has_ps]
ps_t1 = ps_t0 + dur
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says the trials-table photostim fields are the NWB equivalent of the reference stimulation timing variables and notes that it cross-checked them against `BehavioralEvents/photostim_*`.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI converts onset and duration strings to floats for trials where stimulation exists, expresses the interval relative to the go cue, and then builds a binary per-bin trace. A bin is marked 1 if the 50 ms bin interval overlaps the stimulation interval at all; otherwise 0. Trials with `photostim_onset == 'N/A'` stay all zero.

ii.
```python
photostim = np.zeros((len(trials), N_BINS), dtype=np.float32)
has_ps = ps_onset[trials] != 'N/A'
if has_ps.any():
    on = np.array([float(x) for x in ps_onset[trials][has_ps]])
    dur = np.array([float(x) for x in ps_dur[trials][has_ps]])
    ps_t0 = start_time[trials][has_ps] + on - go_v[has_ps]
    ps_t1 = ps_t0 + dur
    ov = (BIN_EDGES[None, :-1] < ps_t1[:, None]) & (BIN_EDGES[None, 1:] > ps_t0[:, None])
    photostim[has_ps] = ov.astype(np.float32)
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI explicitly says the photostim input is binary and based on overlap with the stimulation interval, not merely a per-trial flag.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI expresses photostimulation on the same go-cue-relative bin grid used for neural firing rates. It computes stimulation start and stop relative to each trial’s go cue and compares that interval against the shared `BIN_EDGES`.

ii.
```python
ps_t0 = start_time[trials][has_ps] + on - go_v[has_ps]
ps_t1 = ps_t0 + dur
ov = (BIN_EDGES[None, :-1] < ps_t1[:, None]) & (BIN_EDGES[None, 1:] > ps_t0[:, None])
```

iii. The notes state that the go cue is the global alignment anchor, so photostim is moved into that coordinate frame before binning.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is not read from a dedicated column. The AI derives it from the pair of trial-table variables `outcome` and `trial_instruction`.

ii.
```python
oc = outcome[trials]
ins = instruction[trials]
choice = np.full(len(trials), 2, dtype=np.int64)
licked_left = ((oc == 'hit') & (ins == 'left')) | ((oc == 'miss') & (ins == 'right'))
licked_right = ((oc == 'hit') & (ins == 'right')) | ((oc == 'miss') & (ins == 'left'))
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI justifies this as the experiment’s own “outcome × instruction” definition of choice and says it is preferable to a lick-detection heuristic.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI encodes choice as `0=left`, `1=right`, `2=no lick`. It initializes all trials to `2`, then marks left and right trials from the `outcome`/`instruction` logic. The resulting per-trial value is repeated across all 80 bins.

ii.
```python
choice = np.full(len(trials), 2, dtype=np.int64)
licked_left = ((oc == 'hit') & (ins == 'left')) | ((oc == 'miss') & (ins == 'right'))
licked_right = ((oc == 'hit') & (ins == 'right')) | ((oc == 'miss') & (ins == 'left'))
choice[licked_left] = 0
choice[licked_right] = 1
```

```python
outputs = [np.stack([np.full(N_BINS, choice[t]), np.full(N_BINS, outcome_c[t]),
                     np.full(N_BINS, early_c[t]), tongue_c[t]]).astype(np.int64)
           for t in range(n_tr)]
```

iii. The notes say this matches the task’s categorical output definition and keeps output arrays time-shaped by broadcasting trial-level labels across bins.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the NWB trials-table `outcome` column.

ii.
```python
outcome = _decode(tr['outcome'][:])
...
oc = outcome[trials]
```

iii. In the notes, the AI treats this as a direct mapping because the trials table already contains the required categories `ignore`, `miss`, and `hit`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps the string outcomes to integers `0=ignore`, `1=miss`, `2=hit` with `np.select`, checks that no unknown label remains, and repeats the trial-level code across all 80 bins.

ii.
```python
outcome_c = np.select([oc == 'ignore', oc == 'miss', oc == 'hit'], [0, 1, 2],
                      default=-1).astype(np.int64)
assert np.all(outcome_c >= 0), f'{ident}: unexpected outcome value'
```

```python
outputs = [np.stack([np.full(N_BINS, choice[t]), np.full(N_BINS, outcome_c[t]),
                     np.full(N_BINS, early_c[t]), tongue_c[t]]).astype(np.int64)
           for t in range(n_tr)]
```

iii. The AI’s notes describe outcome as a per-trial decoder target that is broadcast over time like the other scalar trial labels.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the `early_lick` trials-table column.

ii.
```python
early = _decode(tr['early_lick'][:])
```

iii. In the notes, the AI describes `early_lick` as a direct behavioral flag already present in the raw data.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI converts `early` to `1` and everything else to `0` by a boolean comparison, then repeats that trial-level value across the 80 bins.

ii.
```python
early_c = (early[trials] == 'early').astype(np.int64)
```

```python
outputs = [np.stack([np.full(N_BINS, choice[t]), np.full(N_BINS, outcome_c[t]),
                     np.full(N_BINS, early_c[t]), tongue_c[t]]).astype(np.int64)
           for t in range(n_tr)]
```

iii. The notes justify keeping early-lick trials because early lick itself is one of the required decoder outputs.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: the `data` array supplies `(x, y, likelihood)` per frame and `timestamps` gives frame times. The AI uses column 1 as `y` and column 2 as a visibility/confidence measure.

ii.
```python
tongue_key = 'acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking'
tdata = f[tongue_key + '/data'][:]
tts = f[tongue_key + '/timestamps'][:]
```

```python
ycol = track_data[:, 1]
vis_all = track_data[:, 2] > TONGUE_LIKELIHOOD_THRESH
```

iii. In Step 2 and Step 5 of `CONVERSION_NOTES.md`, the AI identifies this NWB series as the side-camera DeepLabCut tongue stream used by the reference video pipeline.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI first keeps only frames with `likelihood > 0.5`. Within each retained trial window, it bins visible-frame `y` values into the same 50 ms go-cue-aligned bins and averages visible values within each bin. It then computes session-specific 40th and 60th percentiles over all visible binned `y` values from the extracted windows of that session, and finally discretizes each trial/bin against those percentiles.

ii.
```python
def tongue_per_bin(track_data, track_ts, go_times):
    ...
    ycol = track_data[:, 1]
    vis_all = track_data[:, 2] > TONGUE_LIKELIHOOD_THRESH
    ...
    for t in range(n_trials):
        ...
        cnt = np.bincount(b_idx[v], minlength=N_BINS)
        tot = np.bincount(b_idx[v], weights=ycol[a:b][v], minlength=N_BINS)
        m = cnt > 0
        y[t, m] = tot[m] / cnt[m]
        visible[t, m] = True
```

```python
def discretize_tongue(y, visible):
    cls = np.full(y.shape, 3, dtype=np.int64)
    if visible.any():
        vals = y[visible]
        p40, p60 = np.percentile(vals, TONGUE_PCTL)
        cls[visible & (y < p40)] = 0
        cls[visible & (y >= p40) & (y <= p60)] = 1
        cls[visible & (y > p60)] = 2
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI argues that non-visible tongue `y` values are meaningless and would corrupt the percentiles, so thresholds should be computed only from visible binned values. It also justifies the 0.5 likelihood threshold by saying the distribution is effectively bimodal.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses four categories: `0` for bins below the session 40th percentile, `1` for bins between the 40th and 60th percentiles inclusive, `2` for bins above the 60th percentile, and `3` when the tongue is not visible in that bin.

ii.
```python
TONGUE_PCTL = (40.0, 60.0)
```

```python
cls = np.full(y.shape, 3, dtype=np.int64)
if visible.any():
    vals = y[visible]
    p40, p60 = np.percentile(vals, TONGUE_PCTL)
    cls[visible & (y < p40)] = 0
    cls[visible & (y >= p40) & (y <= p60)] = 1
    cls[visible & (y > p60)] = 2
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly states that class `3` is reserved for non-visible bins and that the percentile thresholds are session-specific.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue tracking to the neural data by selecting camera frames whose absolute timestamps fall inside each trial’s `[go-2.5, go+1.5)` window, then assigning those frames to the same 50 ms go-cue-relative bins as the neural data.

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

iii. The AI’s notes emphasize that NWB timestamps share one session-wide clock, so go-cue-relative binning is sufficient to align video and spikes.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or malformed data conservatively. Sessions with no usable units are dropped. Partial ephys coverage is handled with `obs_intervals`, and trials whose extracted window still contains no spikes at all are dropped as coverage errors. For tongue tracking, low-likelihood frames are treated as missing, bins with no visible frames become class `3`, and if a whole session has no visible tongue then all bins remain class `3`.

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
...
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

iii. In `CONVERSION_NOTES.md`, the AI repeatedly explains that it would rather exclude unrecorded trials/sessions than fabricate zeros, while for tongue visibility it represents “missing because not visible” as an explicit categorical state.

## 10-a. What are the most time-consuming steps of the code?

i. The AI’s own notes identify the dominant costs as bulk NWB reads, especially `units/spike_times` and the tongue-tracking array, plus the per-unit spike-binning loop built around `np.searchsorted`. Across sessions, the total runtime is then managed with multiprocessing.

ii.
```python
spike_index = u['spike_times_index'][:]
spike_times = u['spike_times'][:]
fr = bin_spikes(spike_times, spike_index, unit_idx, go_v)
```

```python
tdata = f[tongue_key + '/data'][:]
tts = f[tongue_key + '/timestamps'][:]
ty, tvis = tongue_per_bin(tdata, tts, go_v)
```

```python
with ctx.Pool(min(args.workers, len(files))) as pool:
    for i, res in enumerate(pool.imap(_worker, list(zip(files, want_raw)))):
        ...
```

iii. In Step 8 and Step 10 of `CONVERSION_NOTES.md`, the AI explicitly calls out one bulk read of `units/spike_times` per session and multiprocessing over sessions as its main runtime strategy.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious remaining loops are the loop over units in `bin_spikes`, the loop over trials in `tongue_per_bin`, the loop over kept units when unioning `obs_intervals`, and the per-trial list comprehensions that assemble final `neural`/`input`/`output` objects.

ii.
```python
for i, u in enumerate(unit_idx):
    lo = spike_index[u - 1] if u > 0 else 0
    st = spike_times[lo:spike_index[u]]
    pos = np.searchsorted(st, flat_edges).reshape(n_trials, N_BINS + 1)
    fr[i] = (pos[:, 1:] - pos[:, :-1]).astype(np.float32)
```

```python
for t in range(n_trials):
    a, b = lo[t], hi[t]
    ...
```

```python
for k in unit_idx:
    o = oi[(oii[k] - n_obs[k]):oii[k]]
    idx = np.searchsorted(start_time, o[:, 0] + 1e-9) - 1
    observed[idx[idx >= 0]] = True
```

iii. The notes mostly justify these as clarity/safety choices rather than trying to fully vectorize ragged spike or frame data.

## 10-c. What processing does the code repeat multiple times?

i. The code does not reopen a session multiple times during conversion, but it still repeats some work inside a session. It loops over all kept units to union `obs_intervals`, even though the notes say sessions generally share one contiguous observed-trial set. It also repeatedly materializes constant 80-bin output rows (`choice`, `outcome`, `early_lick`) for every trial instead of broadcasting once, and `_summary` later concatenates the entire dataset again just to print global summaries.

ii.
```python
for k in unit_idx:
    o = oi[(oii[k] - n_obs[k]):oii[k]]
    idx = np.searchsorted(start_time, o[:, 0] + 1e-9) - 1
    observed[idx[idx >= 0]] = True
```

```python
outputs = [np.stack([np.full(N_BINS, choice[t]), np.full(N_BINS, outcome_c[t]),
                     np.full(N_BINS, early_c[t]), tongue_c[t]]).astype(np.int64)
           for t in range(n_tr)]
```

```python
allin = np.concatenate([np.concatenate([t[:, None, :] for t in s], axis=1)
                        for s in data['input']], axis=1)
allout = np.concatenate([np.concatenate([t[:, None, :] for t in s], axis=1)
                         for s in data['output']], axis=1)
```

iii. There is no strong written justification for these repeats beyond the general theme in `CONVERSION_NOTES.md` Step 8 that the AI optimized the largest I/O and spike-binning costs first and left smaller clarity-oriented repetition alone.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script contains optional and reporting-only processing that is not needed for the downstream decoder dataset itself. If `--show-processing` is enabled it stores raw arrays and makes diagnostic figures; those are for inspection only. Even without plots, `_summary` recomputes dataset-wide concatenations only to print console summaries. The code also computes metadata fields such as `hemisphere_left_frac`, `proc_time`, `tongue_pctl`, and `mean_rate` that are not used by decoder training.

ii.
```python
raw = None
if want_raw:
    raw = dict(track_data=tdata, track_ts=tts, go=go_v, tone=tone_time,
               start_time=start_time[trials], stop_time=stop_time[trials],
               ty=ty, tvis=tvis, pctl=pctl, outcome=oc, instruction=ins,
               spike_index=spike_index, unit_idx=unit_idx, fname=fname)
```

```python
if args.show_processing:
    for r in results:
        if r is not None and r.get('raw') is not None:
            make_processing_plot(r, f'processing_{r["info"]["session_id"]}.png')
```

```python
allin = np.concatenate([np.concatenate([t[:, None, :] for t in s], axis=1)
                        for s in data['input']], axis=1)
allout = np.concatenate([np.concatenate([t[:, None, :] for t in s], axis=1)
                         for s in data['output']], axis=1)
```

iii. The notes frame these diagnostics as sanity checks and documentation aids rather than part of the essential conversion. They are useful for verification, but not required by downstream decoding.
