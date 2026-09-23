# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all session files by globbing `/app/data/sub-*/*.nwb`, then processes each file once in parallel with `multiprocessing.Pool`. Within each file it reads the NWB content directly through `h5py`, pulling the trials table, behavioral event timestamps, unit tables, and tongue-tracking arrays from HDF5 paths.

ii.
```python
DATA_DIR = '/app/data'
...
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
...
with Pool(n_workers) as pool:
    caches = pool.map(_worker, [(p, region_map) for p in files], chunksize=1)
```

```python
with h5py.File(path, 'r') as f:
    tbl = session_trial_table(f)
    ...
    units = f['units']
```

iii. In its final trajectory summary, the agent said the dataset should be read from NWB `units/spike_times`, the trials table, `BehavioralEvents`, and `BehavioralTimeSeries/Camera0_side_TongueTracking`. It justified this as the complete published NWB representation and chose direct HDF5 access plus multiprocessing for speed.

## 1-b. How are the data split into subjects (mice)?

i. The AI treats the `sub-<id>` prefix in each NWB filename as the subject id. It strips `sub-` from the basename and then builds `subjects` and `subject_idx` during assembly.

ii.
```python
subject = os.path.basename(path).split('_')[0].replace('sub-', '')
out = {'file': path, 'session_name': name, 'subject': subject}
```

```python
subjects = OrderedDict()
...
if res['subject'] not in subjects:
    subjects[res['subject']] = len(subjects)
...
data['subjects'] = list(subjects.keys())
data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. The trajectory summary states the dataset contains 28 mice and treats the filename layout as authoritative for subject grouping. The agent did not rely on `nwb.subject.subject_id`; it used the path convention instead.

## 1-c. How are the data split into sessions?

i. The AI uses one NWB file per session. It keeps the sorted file list order and assigns each kept file a `session_name` from the filename without the `.nwb` suffix.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
...
name = os.path.basename(path).replace('.nwb', '')
out = {'file': path, 'session_name': name, 'subject': subject}
```

iii. In the trajectory, the agent consistently described the dataset as “174 NWB sessions” and treated file boundaries as session boundaries. It used deterministic sorting to keep ordering stable.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB trials table at `intervals/trials`. The AI builds a per-session `tbl` dictionary from trial-table columns and pairs it with `go_start_times`, asserting they have the same length.

ii.
```python
def session_trial_table(f):
    trials = f['intervals/trials']
    tbl = {
        'start_time': trials['start_time'][:],
        'stop_time': trials['stop_time'][:],
        'trial_id': trials['trial'][:].astype(np.int64),
        ...
    }
    tbl['go_time'] = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
    assert len(tbl['go_time']) == len(tbl['start_time'])
    return tbl
```

iii. The agent treated the NWB trials table as the canonical trial definition and used `go_start_times` only as a per-trial event stream aligned to those rows.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several filters. It first rejects entire sessions with behavioral performance `<= 0.65` or with fewer than 50 correct left and 50 correct right trials. Within surviving sessions it excludes `free_water` trials, excludes trials with fewer than 20 observed bins inside the trial interval, excludes trials whose neural matrix is all zero after binning, and can later reject whole sessions if tongue video coverage is below 95% or tongue/lick agreement is poor.

ii.
```python
if perf <= MIN_PERFORMANCE or min(n_left, n_right) < MIN_CORRECT_PER_DIRECTION:
    out['rejected'] = 'behavior'
    return out
```

```python
bins = [mask_bins(tbl, t) for t in range(len(tbl['start_time']))]
short = np.array([len(b) < MIN_BINS_PER_TRIAL for b in bins])
keep_trial = (tbl['free_water'] == 0) & ~short
trials = np.flatnonzero(keep_trial)
```

```python
has_spikes = rates.sum(axis=(0, 2)) > 0
trials = trials[has_spikes]
rates = rates[:, has_spikes]
```

```python
if coverage < MIN_VIDEO_COVERAGE:
    out['rejected'] = 'video does not cover the analysis window'
    return out

if precision < MIN_TONGUE_LICK_PRECISION or recall < MIN_TONGUE_LICK_RECALL:
    out['rejected'] = 'tongue tracking failed'
    return out
```

iii. The trajectory summary says the session filter came from the data paper’s behavioral criteria, `free_water` exclusion came from the method paper, and photostim/early-lick/ignore trials were kept because the decoding task needs them as inputs or outputs. It also says the agent added video QC because the tongue output “can’t be built without usable tracking,” and dropped all-zero-spike trials as recording gaps.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `units/spike_times` and `units/spike_times_index`, with go-cue timestamps from `acquisition/BehavioralEvents/go_start_times/timestamps` used to define the bin edges. Only units passing the AI’s unit filter contribute.

ii.
```python
spike_index = units['spike_times_index'][:]
edges = (go[trials][:, None] + BIN_EDGES[None, :]).ravel()
rates = np.zeros((len(good), len(trials), NBINS), dtype=np.float32)
for i, u in enumerate(good):
    lo = 0 if u == 0 else spike_index[u - 1]
    st = units['spike_times'][lo:spike_index[u]]
```

iii. In the trajectory summary, the agent explicitly listed NWB `units/spike_times` plus go-cue times as the neural source data.

## 2-b. How is the `neural` data processed?

i. The AI bins each unit’s spike times into 50 ms bins around each trial’s go cue using `np.searchsorted`, converts counts to firing rates by dividing by `BIN_WIDTH`, and stores per-trial neuron-by-time matrices. After binning it removes all-zero trials and drops units that never fire in any kept analysis bin.

ii.
```python
counts = np.diff(
    np.searchsorted(st, edges).reshape(len(trials), NBINS + 1), axis=1)
rates[i] = counts / BIN_WIDTH
```

```python
has_spikes = rates.sum(axis=(0, 2)) > 0
...
alive = rates.any(axis=(1, 2))
good = good[alive]
rates = rates[alive]
```

iii. The trajectory summary says the agent wanted firing rates in 50 ms bins and considered all-zero four-second trials to be recording gaps rather than real silence. It also described silent units as uninformative and worth removing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept only when `classification == 'good'` and `anno_name != ''`. Sessions with no such units are rejected. After trial filtering, units with no spikes anywhere in the retained analysis windows are removed as “silent.”

ii.
```python
classification = _str_col(units['classification'])
anno = _str_col(units['anno_name'])
good = np.flatnonzero((classification == 'good') & (anno != ''))
if len(good) < MIN_NEURONS_PER_SESSION:
    out['rejected'] = 'no good units'
    return out
```

```python
alive = rates.any(axis=(1, 2))
out['n_units_silent'] = int((~alive).sum())
good = good[alive]
rates = rates[alive]
```

iii. The agent’s final summary says it used the data paper’s QC-classifier label `classification == 'good'` and did not apply the method paper’s 2 Hz firing-rate threshold. It also said CCF annotation was needed for its brain-region grouping and silent units were dropped because they “carry no information.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to go-cue onset. For each kept trial it adds the fixed relative bin grid `BIN_EDGES` to that trial’s absolute `go_time`, bins spikes in those windows, and then keeps only the bins that fall within the trial’s recorded interval.

ii.
```python
T_START = -2.5
T_END = 1.5
BIN_WIDTH = 0.05
...
edges = (go[trials][:, None] + BIN_EDGES[None, :]).ravel()
```

```python
def mask_bins(tbl, trial):
    lo = tbl['start_time'][trial] - tbl['go_time'][trial]
    hi = tbl['stop_time'][trial] - tbl['go_time'][trial]
    keep = np.flatnonzero((BIN_EDGES[:-1] >= lo) & (BIN_EDGES[1:] <= hi))
    return keep
```

iii. The trajectory summary calls go cue onset the alignment event and says the substantive extra choice was to drop unobserved bins rather than zero-fill them when the requested window extended outside the recorded trial interval.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50 ms. The AI defines an 80-bin grid from -2.5 s to +1.5 s around go cue, but then masks each trial down to the subset of bins fully contained within that trial’s start/stop interval, so kept trials can have 51 to 80 time bins.

ii.
```python
BIN_WIDTH = 0.05
BIN_EDGES = np.round(T_START + BIN_WIDTH * np.arange(
    int(round((T_END - T_START) / BIN_WIDTH)) + 1), 10)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0
NBINS = len(BIN_CENTERS)
```

```python
b = bins[t]
neural.append(np.ascontiguousarray(rates[:, k, b]))
```

iii. In the trajectory, the agent emphasized that it kept the specified 50 ms grid and bin width, but intentionally dropped unobserved bins rather than rebinning or zero-filling. It reported that about 82.6% of trials kept all 80 bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times`, `go_time`, and `start_time`. The AI assigns each sample onset to the trial containing it and uses the last sample-epoch onset at or before that trial’s go cue.

ii.
```python
sample = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
go = tbl['go_time']
start = tbl['start_time']
tone = np.full(len(go), np.nan)
trial_of = np.searchsorted(start, sample, side='right') - 1
...
for t, s in zip(trial_of[valid], sample[valid]):
    if s <= go[t]:
        tone[t] = s
rel = tone - go
```

iii. The trajectory summary says the AI used the “last sample-epoch onset before the go cue,” because early licks can replay the sample epoch and create multiple tone onsets per trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI stores tone onset as a per-trial offset relative to go cue (`tone - go`). It then computes time-from-tone for each kept bin as `BIN_CENTERS[b] - tone_rel[t]`, which equals bin-center time relative to go plus the go-to-tone gap. If any trial lacked a tone onset, it would fill that offset with the session median.

ii.
```python
rel = tone - go
if np.any(np.isnan(rel)):
    rel[np.isnan(rel)] = np.nanmedian(rel)
return rel
```

```python
inp = np.empty((2, len(b)), dtype=np.float32)
inp[0] = BIN_CENTERS[b] - tone_rel[t]
```

iii. The trajectory says the input should be “signed seconds from tone onset” and that the last pre-go tone should be used. The median fallback was an extra safeguard; the agent reported no missing tone onsets in practice.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on the same go-cue-aligned bin centers used for the neural data, and then restricted to the same retained per-trial bin subset `b`.

ii.
```python
b = bins[t]
neural.append(np.ascontiguousarray(rates[:, k, b]))
...
inp[0] = BIN_CENTERS[b] - tone_rel[t]
```

iii. The trajectory summary explicitly says this input is stored on the same analysis grid as the neural data, with variable trial length only where bins are dropped as unobserved.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from the trials-table columns `photostim_onset` and `photostim_duration`, together with `start_time` and `go_time` to convert onset times from trial-start coordinates into go-cue-relative coordinates.

ii.
```python
stim = tbl['photostim_onset'] != 'N/A'
...
onset = float(tbl['photostim_onset'][i])
dur = float(tbl['photostim_duration'][i])
on[i] = tbl['start_time'][i] + onset - tbl['go_time'][i]
off[i] = on[i] + dur
```

iii. The trajectory summary says the agent used a binary photostim indicator from the trials table and verified separately that those trial-table values matched the photostim event timestamps.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI converts onset/duration strings to floats, computes a per-trial `[stim_on, stim_off)` interval relative to go cue, and marks each kept bin as `1` when its center lies inside that interval and `0` otherwise. Trials with `photostim_onset == 'N/A'` get all zeros.

ii.
```python
if np.isfinite(stim_on[t]):
    inp[1] = ((BIN_CENTERS[b] >= stim_on[t]) &
              (BIN_CENTERS[b] < stim_off[t])).astype(np.float32)
else:
    inp[1] = 0.0
```

iii. The trajectory summary says the input should be a time-varying photostim state, not just a per-trial flag, because the decoder task asks whether photostimulation is on “at every time point.”

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostim onset and offset are expressed in the same go-cue-relative coordinates as the neural bin centers, and the AI evaluates photostim on exactly the same retained bin indices `b`.

ii.
```python
on[i] = tbl['start_time'][i] + onset - tbl['go_time'][i]
...
inp[1] = ((BIN_CENTERS[b] >= stim_on[t]) &
          (BIN_CENTERS[b] < stim_off[t])).astype(np.float32)
```

iii. The trajectory summary says the photostim stream was put on the same aligned axis as the firing-rate bins by converting trial-start-relative onset times into go-cue-relative times.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from the behavioral event streams `left_lick_times/timestamps` and `right_lick_times/timestamps`, using the first lick after the go cue and before `go + 1.5 s`. It does not derive choice from `trial_instruction` and `outcome`.

ii.
```python
def lick_times(f):
    be = f['acquisition/BehavioralEvents']
    return (np.sort(be['left_lick_times/timestamps'][:]),
            np.sort(be['right_lick_times/timestamps'][:]))
```

```python
choice = np.full(len(go), 2, dtype=np.int8)
for i, g in enumerate(go):
    li = np.searchsorted(left, g)
    ri = np.searchsorted(right, g)
    tl = left[li] if li < len(left) else np.inf
    tr = right[ri] if ri < len(right) else np.inf
    end = g + T_END
    if tl >= end and tr >= end:
        continue
    choice[i] = 0 if tl <= tr else 1
```

iii. In the trajectory summary, the agent called this a “substantive judgment call”: it preferred the direct lick-port readout because “lick direction choice” is a behavioral measurement, and noted that this agrees with the label implied by `outcome × instruction` on more than 99.3% of trials.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI codes choice as `0` for left, `1` for right, and `2` for no lick. It assigns `2` when neither lick port has a lick within the response window, and otherwise uses whichever port is licked first. The resulting per-trial label is then repeated across all retained time bins of that trial.

ii.
```python
choice = np.full(len(go), 2, dtype=np.int8)
...
choice[i] = 0 if tl <= tr else 1
```

```python
outp = np.empty((4, len(b)), dtype=np.int8)
outp[0] = choice[t]
```

iii. The trajectory summary says this was chosen to reflect measured behavior directly, while still preserving the requested `left/right/no lick` categorical coding.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is read directly from the trials-table `outcome` column.

ii.
```python
outcome = np.array([outcome_code[o] for o in tbl['outcome']], dtype=np.int8)
```

iii. The trajectory summary says the NWB trials table already contains the needed `ignore`, `miss`, and `hit` categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps the outcome strings to integer codes `ignore -> 0`, `miss -> 1`, `hit -> 2`, then repeats the per-trial outcome across all retained bins of the trial.

ii.
```python
outcome_code = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = np.array([outcome_code[o] for o in tbl['outcome']], dtype=np.int8)
...
outp[1] = outcome[t]
```

iii. The trajectory summary says outcome was taken directly from the trial table and stored as a time-varying array only so all outputs share the same `(n_output, T)` per-trial shape.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is read directly from the trials-table `early_lick` column.

ii.
```python
early = (tbl['early_lick'] == 'early').astype(np.int8)
```

iii. The trajectory summary says the decoding task requires early-lick trials to be retained and the trial table already flags whether an early lick occurred.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI binarizes the string label to `0` for no early lick and `1` for early lick, then repeats that per-trial label across all retained bins of the trial.

ii.
```python
early = (tbl['early_lick'] == 'early').astype(np.int8)
...
outp[2] = early[t]
```

iii. The trajectory summary says this variable was kept as an output exactly because dropping early-lick trials would collapse that output.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, using the `timestamps`, the y coordinate in column 1 of `data`, and the DeepLabCut likelihood in column 2 to determine visibility.

ii.
```python
key = 'acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking'
ts = f[key + '/timestamps'][:]
data = f[key + '/data'][:]
y = data[:, 1].astype(np.float64)
visible = data[:, 2] > TONGUE_LIKELIHOOD_THRESH
```

iii. The trajectory summary explicitly names the side-view tongue-tracking series as the source of the tongue output.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI first calls frames with likelihood above `0.5` “visible.” It then optionally applies five-sigma velocity-based outlier interpolation to visible y traces, computes mean visible tongue y within each 50 ms bin for each retained trial, rejects sessions with poor video coverage or poor agreement between visible-tongue frames and lick times, and computes session-level percentile thresholds from visible bin means across observed bins.

ii.
```python
visible = data[:, 2] > TONGUE_LIKELIHOOD_THRESH
...
jump = np.abs(vel) > MARKER_OUTLIER_SIGMA * sigma
...
yv = np.interp(np.arange(len(yv)), keep, yv[keep])
```

```python
bin_y = np.full((len(trials), NBINS), np.nan)
...
for k, t in enumerate(trials):
    lo = np.searchsorted(ts, go[t] + BIN_EDGES[:-1])
    hi = np.searchsorted(ts, go[t] + BIN_EDGES[1:])
    ...
    for b in range(NBINS):
        ...
        if vis.any():
            bin_y[k, b] = tongue_y[sl][vis].mean()
```

```python
coverage = float(has_video[observed].mean())
...
if precision < MIN_TONGUE_LICK_PRECISION or recall < MIN_TONGUE_LICK_RECALL:
    out['rejected'] = 'tongue tracking failed'
    return out
```

iii. The trajectory summary says the visibility threshold is acceptable because the likelihoods are strongly bimodal, and says the five-sigma outlier interpolation came from the method paper. It also says extra session-level video QC was added because the tongue output could not be trusted in sessions where the camera ended at go cue or the tracker falsely called the tongue visible almost all the time.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI takes the 40th and 60th percentiles of session-level visible `bin_y` values over observed bins, then assigns per-bin classes `0` for below `p40`, `1` for `[p40, p60]`, `2` for above `p60`, and `3` for bins with no visible tongue frame.

ii.
```python
vis_vals = bin_y[observed & np.isfinite(bin_y)]
if len(vis_vals) >= 10:
    p40, p60 = np.percentile(vis_vals, TONGUE_PCTILES)
else:
    p40 = p60 = np.inf
tongue_class = np.full((len(trials), NBINS), 3, dtype=np.int8)
vis_bin = np.isfinite(bin_y)
tongue_class[vis_bin & (bin_y < p40)] = 0
tongue_class[vis_bin & (bin_y >= p40) & (bin_y <= p60)] = 1
tongue_class[vis_bin & (bin_y > p60)] = 2
```

iii. The trajectory summary says the 40th/60th percentile split came from the task specification and that class `3` was used when the tongue was not visible.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI uses the same go-cue-relative 50 ms bins as the neural data. It locates camera frames falling in each `[go + edge_k, go + edge_{k+1})` bin with `searchsorted`, averages visible y values within those bins, and then keeps only the same trial-specific observed-bin subset used for the neural matrices.

ii.
```python
lo = np.searchsorted(ts, go[t] + BIN_EDGES[:-1])
hi = np.searchsorted(ts, go[t] + BIN_EDGES[1:])
...
outp[3] = tongue_class[k, b]
```

iii. The trajectory summary says all streams share the NWB session clock, so go-cue-relative binning is enough to align the tongue output with the neural data. Its extra masking step mirrors the neural masking.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several missing-data cases explicitly. It stringifies mixed-type HDF5 columns with `_str_col`, fills missing tone offsets with the session median, drops sessions with no good labeled units, drops trials with short observed windows or all-zero neural activity, rejects sessions with insufficient tongue-video coverage or failed tongue/lick agreement, and represents bins without visible tongue as class `3`.

ii.
```python
def _str_col(dataset):
    return np.array([x.decode() if isinstance(x, bytes) else str(x)
                     for x in dataset[:]])
```

```python
if np.any(np.isnan(rel)):
    rel[np.isnan(rel)] = np.nanmedian(rel)
```

```python
if coverage < MIN_VIDEO_COVERAGE:
    out['rejected'] = 'video does not cover the analysis window'
    return out
...
tongue_class = np.full((len(trials), NBINS), 3, dtype=np.int8)
```

iii. The trajectory summary frames these as pragmatic safeguards: missing spikes are treated as recording gaps, missing/invalid tongue tracking causes rejection or “not visible,” and a median fallback exists for tone timing even though the agent reported not needing it.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive work in the AI code is per-session NWB/HDF5 I/O, spike-time binning for every good unit, and the nested trial-by-bin tongue-video aggregation. Pickling large cached session outputs and the final dataset is also substantial.

ii.
```python
with h5py.File(path, 'r') as f:
    ...
    for i, u in enumerate(good):
        ...
        counts = np.diff(
            np.searchsorted(st, edges).reshape(len(trials), NBINS + 1), axis=1)
```

```python
for k, t in enumerate(trials):
    ...
    for b in range(NBINS):
        ...
        if vis.any():
            bin_y[k, b] = tongue_y[sl][vis].mean()
```

iii. The trajectory shows the agent parallelized conversion with 16 to 24 workers, used per-session caches, and reran long conversions several times. That indicates it viewed NWB reading and per-session processing as the runtime bottlenecks.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized further: the loop over stimulated trials in `photostim_intervals`, the event-assignment loop in `tone_onset_times`, the trial loop in `lick_choice`, the per-unit spike binning loop, and especially the nested trial/bin loops used to compute `bin_y`.

ii.
```python
for i in idx:
    onset = float(tbl['photostim_onset'][i])
    dur = float(tbl['photostim_duration'][i])
```

```python
for t, s in zip(trial_of[valid], sample[valid]):
    if s <= go[t]:
        tone[t] = s
```

```python
for i, u in enumerate(good):
    ...
```

```python
for k, t in enumerate(trials):
    ...
    for b in range(NBINS):
        ...
```

iii. The trajectory does not describe further vectorization attempts. The agent instead chose multiprocessing and caching as the main performance strategy.

## 10-c. What processing does the code repeat multiple times?

i. The AI code recomputes trial-bin membership in multiple places. It first builds `bins = [mask_bins(...)]`, later reconstructs an `observed` mask from those same bin lists, and then iterates through the same trial/bin structure again when assembling neural, input, and output arrays. It also repeatedly searches lick and camera timestamps trial by trial.

ii.
```python
bins = [mask_bins(tbl, t) for t in range(len(tbl['start_time']))]
```

```python
observed = np.zeros((len(trials), NBINS), dtype=bool)
for k, t in enumerate(trials):
    observed[k, bins[t]] = True
```

```python
for k, t in enumerate(trials):
    b = bins[t]
    neural.append(np.ascontiguousarray(rates[:, k, b]))
    ...
    outputs.append(outp)
```

iii. The trajectory does not present this repetition as a deliberate optimization choice. It mainly justifies the masking behavior itself, not the repeated reconstruction of masks and per-trial indexing.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI does extra work beyond what the decoder needs. It downloads and walks the Allen ontology to remap regions, computes session-level behavioral/video QC statistics, reads `photostim_power` even though it is unused, caches per-session intermediates, and stores `trial_metadata` and rich metadata that are not used by downstream decoding.

ii.
```python
ONTOLOGY_URL = 'http://api.brain-map.org/api/v2/structure_graph_download/1.json'
...
def load_region_map():
    if not os.path.exists(ONTOLOGY_FILE):
        urllib.request.urlretrieve(ONTOLOGY_URL, ONTOLOGY_FILE)
```

```python
'photostim_power': _str_col(trials['photostim_power']),
```

```python
trial_metadata.append(res['trial_info'])
...
data['trial_metadata'] = trial_metadata
```

iii. The trajectory summary focuses on these steps as part of the agent’s broader curation and documentation goals, not because they were required for decoder training. They are largely extra bookkeeping relative to the reference solution.
