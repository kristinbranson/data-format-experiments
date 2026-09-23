# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads every NWB file under `/app/data/*/*.nwb` with a sorted glob, scans all files once for session-level metadata, then reopens the selected files with `h5py` for conversion. Within each file it reads the trial table from `intervals/trials`, event streams from `acquisition/BehavioralEvents`, units from `units`, and tongue tracking from `acquisition/BehavioralTimeSeries`.

ii.
```python
files = sorted(glob.glob(os.path.join(args.data_dir, '*', '*.nwb')))
...
with Pool(args.nproc) as pool:
    scans = pool.map(scan_session, files)
```

```python
with h5py.File(path, 'r') as f:
    tt = _trial_table(f)
    classification = f['units/classification'][:].astype(str)
    ...
    go_time = tt['go_time'][trials]
    rates = bin_spikes(f, good, go_time)
    tongue_y = tongue_y_per_bin(f, go_time)
```

iii. In its final trajectory summary (step 158), the agent said it chose to read DANDI NWB files directly with `h5py`, and that it spot-checked spike-rate binning and derived choice labels against independent checks.

## 1-b. How are the data split into subjects (mice)?

i. The AI uses the NWB `general/subject/subject_id` field as the mouse identifier for each session, then creates `subjects` as the sorted unique IDs and `subject_idx` as the per-session index into that list.

ii.
```python
subject = str(f['general/subject/subject_id'][()].decode())
```

```python
subjects = sorted({s['subject'] for s in sessions})
subject_idx = np.array([subjects.index(s['subject']) for s in sessions],
                       dtype=np.int64)
```

iii. The trajectory does not give a separate subject-specific justification beyond the general loading summary, but step 158 treats the NWB contents as the canonical source of session metadata.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session, but it does not keep all sessions. It first scans all files, then keeps only sessions with performance `> 0.65`, at least 50 correct left and 50 correct right trials, and at least one `classification == 'good'` unit; after conversion it also drops sessions with fewer than 2 kept trials or no retained neurons.

ii.
```python
files = sorted(glob.glob(os.path.join(args.data_dir, '*', '*.nwb')))
...
with Pool(args.nproc) as pool:
    scans = pool.map(scan_session, files)
```

```python
if s['performance'] <= MIN_PERFORMANCE:
    reasons.append('performance %.3f' % s['performance'])
if min(s['n_correct_left'], s['n_correct_right']) < MIN_CORRECT_PER_DIRECTION:
    reasons.append('correct trials %d/%d'
                   % (s['n_correct_left'], s['n_correct_right']))
if s['n_good_units'] == 0:
    reasons.append('no good units')
...
sessions = [s for s in sessions
            if s['info']['n_neurons'] > 0 and s['info']['n_trials'] >= 2]
```

iii. In step 158 the agent explicitly justified this as applying the data paper’s session-selection criteria, reporting that 143 of 174 sessions were kept because that reproduced the paper’s performance range.

## 1-d. How are the data split into trials?

i. The AI takes trials directly from the NWB trials table. `_trial_table` reads one row per behavioral trial from `intervals/trials`, then adds the aligned `go_time` and the last `sample_start_times` entry before each go cue as `tone_time`.

ii.
```python
tr = f['intervals/trials']
out = {
    'start_time': tr['start_time'][:],
    'stop_time': tr['stop_time'][:],
    'outcome': tr['outcome'][:].astype(str),
    'early_lick': tr['early_lick'][:].astype(str),
    'instruction': tr['trial_instruction'][:].astype(str),
    ...
}
...
out['go_time'] = events['go_start_times']['timestamps'][:]
sample_starts = events['sample_start_times']['timestamps'][:]
idx = np.searchsorted(sample_starts, out['go_time']) - 1
out['tone_time'] = sample_starts[np.clip(idx, 0, len(sample_starts) - 1)]
```

iii. In step 158 the agent justified the tone assignment by saying early licks replay task epochs, so the relevant tone is the last sample onset before the go cue.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials passing `trial_mask(tt)`, which requires `auto_water == 0`, `free_water == 0`, `observed == True`, and `tone_valid == True`. `observed` is computed from `units/obs_intervals`, and `tone_valid` requires that a sample onset exists before the go cue and is not earlier than the trial start. Sessions with fewer than 2 surviving trials are dropped.

ii.
```python
def trial_mask(tt):
    return ((tt['auto_water'] == 0) & (tt['free_water'] == 0)
            & tt['observed'] & tt['tone_valid'])
```

```python
out['observed'] = observed_trials(f, out['start_time'])
...
out['tone_valid'] = (idx >= 0) & (out['tone_time'] >= out['start_time'])
```

iii. In step 158 the agent justified dropping auto-water/free-water trials as non-choice-contingent, keeping photostim/early-lick/ignore trials because they are decoder variables, and dropping trials outside `obs_intervals` because they otherwise create all-zero neural trials. The trajectory does not separately justify the `tone_valid` filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural matrices are derived from `units/spike_times` together with `units/spike_times_index` to unpack each unit’s ragged spike train, and from per-trial `go_time` to define the bin edges.

ii.
```python
units = f['units']
spike_times = units['spike_times'][:]
stop = units['spike_times_index'][:]
start = np.concatenate([[0], stop[:-1]])
unit_ids = np.where(good)[0]
```

```python
edges = go_time[:, None] + BIN_EDGES[None, :]
flat_edges = edges.ravel()
```

iii. Step 158 says the agent read NWB directly, checked rate bins against an independent histogram of raw spike times, and used go-cue-centered alignment.

## 2-b. How is the `neural` data processed?

i. The AI bins each retained unit’s spike times into 80 non-overlapping 50 ms bins around each trial’s go cue, computes spike counts by differencing `searchsorted` positions at adjacent bin edges, and divides by bin width to convert counts to firing rates in Hz. There is no smoothing, normalization, or baseline subtraction.

ii.
```python
rates = np.zeros((n_trials, N_BINS, len(unit_ids)), dtype=np.float32)
for i, u in enumerate(unit_ids):
    st = spike_times[start[u]:stop[u]]
    counts = np.searchsorted(st, flat_edges).reshape(n_trials, N_BINS + 1)
    rates[:, :, i] = np.diff(counts, axis=1) / BIN_WIDTH
```

iii. In step 158 the agent said it was matching the reference’s go-cue-centered 50 ms rate bins and validating them against an independent histogram.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI first keeps units with `classification == 'good'`, then further removes any of those units whose `anno_name` cannot be mapped into one of its Allen-ontology-derived coarse regions. A session is excluded up front if it has no `good` units, and removed after conversion if no neurons remain after region filtering.

ii.
```python
classification = f['units/classification'][:].astype(str)
good = classification == 'good'
region_names, has_region = unit_regions(f, good, region_map)
good_idx = np.where(good)[0][has_region]
good = np.zeros(len(classification), dtype=bool)
good[good_idx] = True
region_names = region_names[has_region]
```

```python
if s['n_good_units'] == 0:
    reasons.append('no good units')
...
sessions = [s for s in sessions
            if s['info']['n_neurons'] > 0 and s['info']['n_trials'] >= 2]
```

iii. In step 158 the agent justified this as matching the white-paper classifier and the reference preprocessing’s requirement that units have both QC and CCF annotation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the instructions?

i. Neural data are aligned to go-cue onset by adding the fixed relative bin edges `[-2.5, 1.5]` s to each trial’s `go_time`, then binning spikes directly against those absolute-time edges.

ii.
```python
T_START = -2.5
T_STOP = 1.5
BIN_WIDTH = 0.05
BIN_EDGES = T_START + BIN_WIDTH * np.arange(N_BINS + 1)
...
edges = go_time[:, None] + BIN_EDGES[None, :]
flat_edges = edges.ravel()
```

iii. In step 158 the agent explicitly said everything was aligned to the go cue because that was the alignment event used in the papers and reference preprocessing.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 50 ms bins over a 4 s window, giving 80 time bins per trial. The AI performs one binning step from spike times to 50 ms firing-rate bins; there is no additional temporal rebinning.

ii.
```python
BIN_WIDTH = 0.05
N_BINS = int(round((T_STOP - T_START) / BIN_WIDTH))
BIN_EDGES = T_START + BIN_WIDTH * np.arange(N_BINS + 1)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2
```

iii. Step 158 says the agent intentionally matched the requested 50 ms bins over `-2.5 s` to `+1.5 s` around go cue.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times` in `acquisition/BehavioralEvents` and the per-trial go cue. For each trial, the AI chooses the last sample onset before that trial’s go cue.

ii.
```python
sample_starts = events['sample_start_times']['timestamps'][:]
idx = np.searchsorted(sample_starts, out['go_time']) - 1
out['tone_time'] = sample_starts[np.clip(idx, 0, len(sample_starts) - 1)]
```

iii. In step 158 the agent justified this by saying early licks replay epochs, so the behaviorally relevant tone is the last sample onset before the go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. After selecting `tone_time` per trial, the AI computes each bin’s value as the bin center on the go-cue-relative grid minus `(tone_time - go_time)`, yielding seconds from tone onset at each of the 80 bins.

ii.
```python
tone_rel = tt['tone_time'][trials] - go_time
time_from_tone = BIN_CENTERS[None, :] - tone_rel[:, None]
```

iii. Step 158 describes this input as “time from tone onset” based on the last sample onset before go cue. The trajectory does not add another formula-level justification beyond that alignment choice.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is expressed on exactly the same 80-bin go-cue-centered time grid used for the neural data. The only difference is that the bin centers are shifted by each trial’s tone-to-go interval.

ii.
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2
...
time_from_tone = BIN_CENTERS[None, :] - tone_rel[:, None]
```

iii. In step 158 the agent said the decoder inputs were constructed on the go-cue-aligned binning used for the rest of the dataset.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from the trial-table columns `photostim_onset`, `photostim_duration`, and `start_time`, together with `go_time` so the stimulation interval can be expressed on the go-cue-relative axis.

ii.
```python
onset = tt['photostim_onset'][trials]
duration = tt['photostim_duration'][trials]
start_time = tt['start_time'][trials]
...
on = start_time[i] + float(onset[i]) - go_time[i]
off = on + float(duration[i])
```

iii. In step 158 the agent said photostim was built from the per-trial `[onset, onset + duration]` window after re-expressing onset relative to the aligned go cue.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI creates a binary 80-bin time series. For each trial with a non-`N/A` onset, it marks a bin as stimulated if the stimulation interval overlaps any part of that 50 ms bin, using bin-edge overlap rather than testing only the bin center.

ii.
```python
photostim = np.zeros((n_trials, N_BINS), dtype=np.float32)
...
photostim[i] = ((BIN_EDGES[:-1] < off) & (BIN_EDGES[1:] > on))
```

iii. In step 158 the agent justified this as making photostim “a binary overlap with the per-trial `[onset, onset+duration]` window.”

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The stimulation onset and offset are converted from trial-start-relative time to go-cue-relative time, and then compared against the same 50 ms bin grid used for neural data.

ii.
```python
on = start_time[i] + float(onset[i]) - go_time[i]
off = on + float(duration[i])
photostim[i] = ((BIN_EDGES[:-1] < off) & (BIN_EDGES[1:] > on))
```

iii. Step 158 explicitly says photostimulation is represented on the same aligned per-trial window as the other streams.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from `outcome` and `trial_instruction`. It does not read an explicit lick-direction label from the file.

ii.
```python
outcome_str = tt['outcome'][trials]
instruction = tt['instruction'][trials]
```

```python
licked_left = (((outcome_str == 'hit') & (instruction == 'left'))
               | ((outcome_str == 'miss') & (instruction == 'right')))
licked_right = (((outcome_str == 'hit') & (instruction == 'right'))
                | ((outcome_str == 'miss') & (instruction == 'left')))
```

iii. In step 158 the agent said it validated the derived choice labels against the first lick after the go cue, and step 145 reports 100% agreement in a spot check.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI maps choice to `0 = left`, `1 = right`, and `2 = no lick`. It initializes every trial to `2`, then sets left/right from the `hit`/`miss` and instructed-side logic, and repeats the per-trial choice value across all 80 bins.

ii.
```python
choice = np.full(n_trials, 2, dtype=np.int64)
...
choice[licked_left] = 0
choice[licked_right] = 1
```

```python
outputs = [np.stack([np.full(N_BINS, choice[i]),
                     np.full(N_BINS, outcome[i]),
                     np.full(N_BINS, early_lick[i]),
                     tongue[i]]).astype(np.int64)
           for i in range(n_trials)]
```

iii. Step 158 says the agent viewed this as the correct derivation because `hit`, `miss`, and `ignore` fully determine whether the animal licked the instructed side, the opposite side, or not at all.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the NWB trial-table `outcome` field.

ii.
```python
outcome_str = tt['outcome'][trials]
```

iii. The trajectory does not add a separate justification beyond using the NWB trial table as the canonical behavioral record.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps strings to integers with `ignore -> 0`, `miss -> 1`, `hit -> 2`, then repeats the per-trial label across all 80 bins.

ii.
```python
outcome = np.full(n_trials, -1, dtype=np.int64)
outcome[outcome_str == 'ignore'] = 0
outcome[outcome_str == 'miss'] = 1
outcome[outcome_str == 'hit'] = 2
```

```python
outputs = [np.stack([np.full(N_BINS, choice[i]),
                     np.full(N_BINS, outcome[i]),
                     np.full(N_BINS, early_lick[i]),
                     tongue[i]]).astype(np.int64)
           for i in range(n_trials)]
```

iii. Step 158 treats outcome as a required decoder target that should be preserved, including `ignore` trials.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from the NWB trial-table `early_lick` field.

ii.
```python
early = tt['early_lick'][trials]
```

iii. In step 158 the agent justified retaining early-lick trials because early lick itself is a required decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI converts the string label to a binary integer with `early == 'early'`, producing `0 = no`, `1 = yes`, then repeats that per-trial label across all 80 bins.

ii.
```python
early_lick = (early == 'early').astype(np.int64)
```

```python
outputs = [np.stack([np.full(N_BINS, choice[i]),
                     np.full(N_BINS, outcome[i]),
                     np.full(N_BINS, early_lick[i]),
                     tongue[i]]).astype(np.int64)
           for i in range(n_trials)]
```

iii. Step 158 says this was a deliberate choice to keep early-lick trials rather than filter them out, because the decoder must predict early lick.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`. The AI uses the series `timestamps`, column 1 of `data` as tongue `y`, and column 2 as the DeepLabCut likelihood used to decide visibility.

ii.
```python
tracking = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
ts = tracking['timestamps'][:]
data = tracking['data'][:]
y = data[:, 1]
visible = data[:, 2] > TONGUE_LIKELIHOOD_THRESHOLD
```

iii. In step 158 the agent described this as side-view DeepLabCut tongue tracking and justified the visibility threshold by saying the likelihood distribution is sharply bimodal.

## 8-b. How is `output` *Tongue y-position* processed?

i. The AI first computes mean tongue `y` per 50 ms bin for each kept trial by averaging only frames whose likelihood exceeds `0.5`. It does this with cumulative sums and `searchsorted`, not by session-wide rebinning. It then pools all finite per-bin values from the kept trials of that session, computes the 40th and 60th percentiles of those values, and discretizes each visible bin with those cutoffs.

ii.
```python
visible = data[:, 2] > TONGUE_LIKELIHOOD_THRESHOLD
...
idx = np.searchsorted(ts, edges)
n_visible = np.diff(csum_n[idx], axis=1)
sum_y = np.diff(csum_y[idx], axis=1)
with np.errstate(invalid='ignore', divide='ignore'):
    mean_y = np.where(n_visible > 0, sum_y / np.maximum(n_visible, 1), np.nan)
```

```python
finite = tongue_y[np.isfinite(tongue_y)]
...
if finite.size >= 10:
    p40, p60 = np.percentile(finite, [40, 60])
    visible = np.isfinite(tongue_y)
    tongue[visible & (tongue_y <= p60)] = 1
    tongue[visible & (tongue_y < p40)] = 0
    tongue[visible & (tongue_y > p60)] = 2
```

iii. In step 158 the agent justified the `0.5` likelihood threshold as effectively inconsequential because of a bimodal confidence distribution, and justified the percentile thresholds as session-specific percentiles over visible bins.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI creates four categories: `0` for visible bins below the 40th percentile, `1` for visible bins from the 40th percentile up to and including the 60th percentile, `2` for visible bins above the 60th percentile, and `3` for bins with no visible tongue sample. If there are fewer than 10 finite values in the session, it leaves all bins as `3`.

ii.
```python
tongue = np.full(tongue_y.shape, 3, dtype=np.int64)
if finite.size >= 10:
    p40, p60 = np.percentile(finite, [40, 60])
    visible = np.isfinite(tongue_y)
    tongue[visible & (tongue_y <= p60)] = 1
    tongue[visible & (tongue_y < p40)] = 0
    tongue[visible & (tongue_y > p60)] = 2
```

iii. Step 158 says the session percentiles are taken over visible bins and that bins without visible tongue are intentionally assigned the explicit “not visible” class. The minimum-10-finite-values fallback is not separately justified in the trajectory.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue tracking to the same go-cue-relative 50 ms bins used for spikes by `searchsorted` on camera timestamps at each trial’s absolute bin edges.

ii.
```python
edges = go_time[:, None] + BIN_EDGES[None, :]
idx = np.searchsorted(ts, edges)
...
mean_y = np.where(n_visible > 0, sum_y / np.maximum(n_visible, 1), np.nan)
```

iii. In step 158 the agent described tongue y as a binwise signal on the same aligned trial window as neural activity.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or problematic data mostly by exclusion. Trials are dropped if they are outside `obs_intervals`, are auto-water/free-water, or fail `tone_valid`. Units are dropped if they are not `classification == 'good'` or lack a usable mapped CCF annotation. Tongue bins with no visible frames become class `3`, and sessions with implausibly high or low visible-tongue fractions are kept but flagged in metadata.

ii.
```python
return ((tt['auto_water'] == 0) & (tt['free_water'] == 0)
        & tt['observed'] & tt['tone_valid'])
```

```python
regions = np.array([region_map.get(a.strip(), None) for a in anno],
                   dtype=object)
keep = np.array([r is not None for r in regions])
```

```python
tongue = np.full(tongue_y.shape, 3, dtype=np.int64)
...
'tongue_tracking_note':
    'the fraction of bins with a visible tongue is 0.24 in the '
    'median session; ... their tongue labels should be treated with '
    'care: %s'
```

iii. Step 158 explicitly justifies dropping trials outside `obs_intervals`, keeping zero-filled spike bins on truncated miss trials so the `miss` class is not removed, and flagging rather than excluding sessions with visibly unreliable tongue tracking.

## 10-a. What are the most time-consuming steps of the code?

i. The expensive parts are two full passes over the NWB files (`scan_session` and `convert_session`), per-unit spike binning with `np.searchsorted`, and reading the full tongue-tracking arrays. The script is parallelized with `multiprocessing.Pool`, and it also optionally writes per-session cache files to disk.

ii.
```python
with Pool(args.nproc) as pool:
    scans = pool.map(scan_session, files)
...
with Pool(args.nproc) as pool:
    for i, result in enumerate(pool.imap(convert_session, jobs)):
        sessions.append(result)
```

```python
for i, u in enumerate(unit_ids):
    st = spike_times[start[u]:stop[u]]
    counts = np.searchsorted(st, flat_edges).reshape(n_trials, N_BINS + 1)
```

iii. The trajectory supports this: step 148 reports the full parallel conversion, and step 158 emphasizes direct NWB reads plus per-unit spike-rate computation and session-wide tongue processing.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining Python loops are over units in `bin_spikes`, over good units in `observed_trials`, over trials for photostim assignment, and over units when mapping regions to names. The tongue computation is already vectorized with cumulative sums and `searchsorted`.

ii.
```python
for u in good_idx:
    interval_starts = obs[start[u]:stop[u], 0]
    ...
    observed &= unit_observed
```

```python
for i, u in enumerate(unit_ids):
    st = spike_times[start[u]:stop[u]]
    counts = np.searchsorted(st, flat_edges).reshape(n_trials, N_BINS + 1)
```

```python
for i in range(n_trials):
    if onset[i] == 'N/A':
        continue
    on = start_time[i] + float(onset[i]) - go_time[i]
    off = on + float(duration[i])
    photostim[i] = ((BIN_EDGES[:-1] < off) & (BIN_EDGES[1:] > on))
```

iii. Step 158 implies the agent intentionally vectorized the expensive parts it cared about, especially tongue processing, and accepted smaller remaining loops.

## 10-c. What processing does the code repeat multiple times?

i. The AI repeats several steps. It opens every file once in `scan_session` and again in `convert_session`. It rebuilds the trial table and recomputes session performance in both passes. It also may perform conversion work twice across runs if the cache is not reused.

ii.
```python
with Pool(args.nproc) as pool:
    scans = pool.map(scan_session, files)
```

```python
with h5py.File(path, 'r') as f:
    tt = _trial_table(f)
    performance, n_left, n_right = session_performance(tt)
```

```python
with h5py.File(path, 'r') as f:
    tt = _trial_table(f)
    ...
    performance, n_left, n_right = session_performance(tt)
```

iii. The trajectory frames the scan pass as necessary for applying session-selection criteria before conversion, but it does not argue that the duplicated file reads or repeated `session_performance` calculation are desirable in themselves.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script does extra work that is not needed for the decoder-format data itself: it downloads Allen ontology metadata if absent, scans and scores all sessions before conversion, computes rejection reasons for sessions that are later discarded, builds hemisphere-qualified coarse region labels instead of using the raw area names, and writes extensive metadata and optional cache files that are not required for downstream decoding.

ii.
```python
if not os.path.exists(STRUCTURE_CSV):
    _download_structures(STRUCTURE_CSV)
region_map = build_region_map()
```

```python
selected, rejected = [], []
for s in scans:
    reasons = []
    ...
    if reasons:
        s['rejected_because'] = '; '.join(reasons)
        rejected.append(s)
```

```python
if cache_file is not None:
    tmp = cache_file + '.tmp%d' % os.getpid()
    with open(tmp, 'wb') as fh:
        pickle.dump(result, fh, protocol=4)
    os.replace(tmp, cache_file)
```

iii. In step 158 the agent explicitly presented the Allen-ontology mapping, session screening, and tongue-quality notes as deliberate design choices, but those are additional processing layers beyond the minimum needed to assemble the requested decoder arrays.
