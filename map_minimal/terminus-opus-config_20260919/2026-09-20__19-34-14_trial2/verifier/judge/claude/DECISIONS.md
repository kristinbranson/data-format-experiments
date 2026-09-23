# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is distributed as one NWB file per session under `data/sub-<subject_id>/`. All session files are found with a glob, and each is opened with `h5py` (not `pynwb`) and processed in parallel using `multiprocessing.Pool`. Subjects, trials, and units are read from HDF5 paths within each file. An Allen ontology JSON file is also loaded for brain region annotation.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, '*', '*.nwb')))
...
with Pool(args.nproc) as pool:
    for res, info in pool.imap(process_session, [(fn, name2anc) for fn in files]):
```

```python
def _process_session(f, fname, name2anc, info):
    tr = f['intervals/trials']
    be = f['acquisition/BehavioralEvents']
    ...
    subject = str(np.asarray(f['general/subject/subject_id'][()]).astype(str))
```

iii. The agent chose h5py over pynwb for speed and used multiprocessing for parallelism. From trajectory step 13: "Let's inspect NWB file structure with h5py (faster than pynwb read for structure exploration)."

## 1-b. How are the data split into subjects?

i. Each NWB file records its animal in `general/subject/subject_id`. That value is read for every session and carried through to assembly, where `subjects` is the sorted set of unique ids and `subject_idx` gives each session's index into that list.

ii.
```python
subject = str(np.asarray(f['general/subject/subject_id'][()]).astype(str))
info['subject'] = subject
```

```python
subjects = sorted({r['subject'] for r in results})
...
'subject_idx': np.array([subjects.index(r['subject']) for r in results]),
```

iii. The agent reads the subject_id from the NWB metadata. The numeric subject IDs are used directly (e.g., '440956' rather than mouse names like 'SC015').

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. However, the AI applies **session-level behavioral performance criteria** from the data paper: overall performance > 65% on control trials (no photostimulation, no early lick) and at least 50 correct lick-left and 50 correct lick-right control trials. Sessions with missing/bad video are also excluded. This reduces from 174 to ~98 sessions.

ii.
```python
PERF_THRESH = 0.65
MIN_CORRECT_PER_DIRECTION = 50
```

```python
control = (photostim_onset == 'N/A') & (early == 'no early')
perf = float((outcome[control] == 'hit').mean()) if control.sum() else 0.0
n_correct_left = int((control & (outcome == 'hit') & (instruction == 'left')).sum())
n_correct_right = int((control & (outcome == 'hit') & (instruction == 'right')).sum())
if not (perf > PERF_THRESH and n_correct_left >= MIN_CORRECT_PER_DIRECTION
        and n_correct_right >= MIN_CORRECT_PER_DIRECTION):
    info['excluded'] = 'behavior'
    return None
```

iii. From trajectory step 60-61: "Session criterion (hit fraction of control non-early trials > 65%, >= 50 correct L and R) gives exactly 106 sessions, matching 'n = 106 sessions' used in the analysis paper." The agent also excludes sessions with non-monotonic video timestamps or insufficient video coverage.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`intervals/trials`), one row per behavioral trial. Go cue events are mapped to trials using `map_events_to_trials`.

ii.
```python
tr = f['intervals/trials']
trial_start = tr['start_time'][:]
trial_stop = tr['stop_time'][:]
...
go_all = be['go_start_times/timestamps'][:]
gidx = map_events_to_trials(go_all, trial_start, trial_stop)
go = np.full(ntrials_all, np.nan)
go[gidx[gidx >= 0]] = go_all[gidx >= 0]
```

iii. The agent maps go cue events into trials using searchsorted on trial start/stop times rather than assuming a 1:1 correspondence.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several trial filters:
1. Free-water and auto-water trials excluded
2. Trials with bad video coverage excluded
3. Trials outside ephys observation intervals (`obs_intervals`) excluded
4. Trials where go cue is missing excluded
5. Trials with no tone onset excluded (post-hoc, after main loop)
6. Trials with zero total spikes across all units excluded (post-hoc)
7. Photostimulation, early-lick and ignore trials are KEPT

ii.
```python
keep = (~auto_water) & (~free_water) & good_video & np.isfinite(go)
...
# obs_intervals filtering
observed = np.ones(ntrials_all, dtype=bool)
for u in unit_idx:
    iv = obs[obs_start[u]:obs_idx[u], 0]
    ti = np.searchsorted(trial_start, iv + 1e-6) - 1
    ...
    observed &= m
keep = keep & observed
```

```python
# Post-hoc: drop trials with NaN inputs or zero spikes
valid = [i for i, x in enumerate(input_trials)
         if np.all(np.isfinite(x)) and neural_trials[i].sum() > 0]
```

iii. From trajectory step 79-84: The agent discovered all-zero neural trials and traced the cause to ephys recording not covering the full behavioral session. It patched obs_intervals filtering to fix this. Auto-water exclusion follows Wang, Kurgyis et al.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times`, the sorted spike times of each unit. Only units with `classification == 'good'` contribute. Go-cue times from `BehavioralEvents/go_start_times` are used to place bin edges.

ii.
```python
sp_index = f['units/spike_times_index'][:]
sp_start = np.concatenate([[0], sp_index[:-1]])
spike_ds = f['units/spike_times']
...
for i, u in enumerate(unit_idx):
    st = spike_ds[sp_start[u]:sp_index[u]]
```

iii. The agent reads spike times from the HDF5 dataset directly using h5py indexing.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to firing rates in Hz. For each unit, bin edges for all trials are computed, flattened, and sorted. `np.searchsorted` gives spike counts at each edge, and differencing gives counts per bin. Counts are divided by bin width (0.05 s) to give Hz. No smoothing or normalization is applied.

ii.
```python
edges = (go[trials][:, None] + OFF_START
         + BIN_SIZE * np.arange(NBINS + 1)[None, :])       # (ntrials, nbins+1)
flat_edges = edges.ravel()
order = np.argsort(flat_edges, kind='stable')
sorted_edges = flat_edges[order]

for i, u in enumerate(unit_idx):
    st = spike_ds[sp_start[u]:sp_index[u]]
    if len(st) == 0:
        continue
    st = np.sort(st)
    counts_sorted = np.searchsorted(st, sorted_edges)
    counts = np.empty_like(counts_sorted)
    counts[order] = counts_sorted
    counts = counts.reshape(len(trials), NBINS + 1)
    rates[i] = np.diff(counts, axis=1).astype(np.float32) / BIN_SIZE
```

iii. The agent pre-sorts the edge array and uses an inverse permutation to un-sort the counts, which is equivalent to the reference approach but with a different optimization strategy.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-level filtering:
1. Only units with `classification == 'good'` are kept (the QC classifier from Chen, Liu et al. 2023).
2. Additionally, units flagged by the per-trial quality flag `units/is_good_trials` on any analyzed trial are dropped, so every neuron is well isolated on every kept trial.

A session with no good units after both filters is dropped.

ii.
```python
classification = _str(f['units/classification'][:])
good_unit = classification == 'good'
unit_idx = np.flatnonzero(good_unit)
...
# Per-trial unit quality flag
ig_all = f['units/is_good_trials'][:]
unit_ok = np.ones(len(unit_idx), dtype=bool)
for i, u in enumerate(unit_idx):
    ...
    unit_ok[i] = bool(flag[keep].all())
unit_idx = unit_idx[unit_ok]
```

iii. From trajectory step 47: The agent investigated `is_good_trials` distribution and decided to use it as an additional quality filter. From the docstring: "Units flagged by the per-trial quality flag `units/is_good_trials` on any analysed trial are dropped, so every neuron is well isolated on every kept trial (affects 4 sessions)."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All NWB times are on the same session-absolute clock. Bin edges are computed as offsets from each trial's go-cue time: `go[trials][:, None] + OFF_START + BIN_SIZE * np.arange(NBINS + 1)`. Spikes are binned against these absolute edges.

ii.
```python
edges = (go[trials][:, None] + OFF_START
         + BIN_SIZE * np.arange(NBINS + 1)[None, :])
```

iii. The agent recognized that all timestamps share a common clock and no additional synchronization is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Spike times are binned into 80 non-overlapping 50 ms bins spanning -2.5 s to +1.5 s relative to the go cue. The bin grid is defined once and reused for every trial and session.

ii.
```python
OFF_START = -2.5           # s, relative to go cue
OFF_END = 1.5              # s, relative to go cue
BIN_SIZE = 0.05            # s (50 ms bins, as requested by the decoder task)
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 80
```

iii. The agent notes in the docstring: "The papers use 40 ms bins with a 3.4 ms stride; the task prescribes 50 ms bins instead, which is the only change to the neural processing."

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` in `BehavioralEvents`, together with the go cue of each trial. The tone taken for a trial is the last one before its go cue. Events are mapped to trials using `map_events_to_trials`.

ii.
```python
sam_all = be['sample_start_times/timestamps'][:]
sidx = map_events_to_trials(sam_all, trial_start, trial_stop)
tone_onset = np.full(ntrials_all, np.nan)
for t, i in zip(sam_all, sidx):
    if i >= 0 and (not np.isfinite(go[i]) or t <= go[i]):
        if not np.isfinite(tone_onset[i]) or t > tone_onset[i]:
            tone_onset[i] = t
```

iii. The agent recognized that early-lick trials replay the sample epoch, so a trial can have multiple tone onsets. From trajectory step 42: "some trials have multiple sample (tone) epochs due to early-lick retriggers."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the time from tone onset is computed as the absolute time of each bin center minus the tone onset time. Bin centers are at `go + OFF_START + BIN_SIZE * (np.arange(NBINS) + 0.5)`.

ii.
```python
bin_centers = OFF_START + BIN_SIZE * (np.arange(NBINS) + 0.5)
...
if np.isfinite(tone_onset[t]):
    time_from_tone = (g + bin_centers) - tone_onset[t]
else:
    time_from_tone = np.full(NBINS, np.nan)
```

iii. Trials with no recorded tone onset get NaN values, and are later dropped in the post-hoc filtering step.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The bin centers are defined relative to the go cue, which is the same alignment event used for the neural data. The formula `(g + bin_centers) - tone_onset[t]` uses the absolute go-cue time plus relative bin centers, ensuring alignment with the neural bins.

ii.
```python
bin_centers = OFF_START + BIN_SIZE * (np.arange(NBINS) + 0.5)
...
time_from_tone = (g + bin_centers) - tone_onset[t]
```

iii. Same go-cue-relative grid used for both neural and input data.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI uses `BehavioralEvents/photostim_start_times` and `photostim_stop_times` event streams (absolute timestamps) rather than the trials table fields. These are mapped to trials via `map_events_to_trials`.

ii.
```python
if 'photostim_start_times' in be:
    pst = be['photostim_start_times/timestamps'][:]
    psp = be['photostim_stop_times/timestamps'][:]
    pidx = map_events_to_trials(pst, trial_start, trial_stop)
    for a, b, i in zip(pst, psp, pidx):
        if i >= 0:
            ps_on[i] = a
            ps_off[i] = b
```

iii. The agent chose the event streams over the trials table columns, which provide absolute timestamps directly. From trajectory step 39: "Photostim: onset relative to trial start, duration 0.5 s, ends at go cue; BehavioralEvents may also have photostim_start_times."

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A bin is 1 if any part of the photostimulation window overlaps with the bin interval (using bin edges, not bin centers). The onset and offset are converted to go-cue-relative times.

ii.
```python
bin_lo = OFF_START + BIN_SIZE * np.arange(NBINS)
bin_hi = bin_lo + BIN_SIZE
...
if np.isfinite(ps_on[t]):
    a, b = ps_on[t] - g, ps_off[t] - g
    photostim = ((bin_hi > a) & (bin_lo < b)).astype(np.float32)
```

iii. The AI uses bin-edge overlap rather than bin-center containment, meaning a bin is marked as stimulated if any part of the bin overlaps with the stimulation period.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The photostim onset and offset are converted to go-cue-relative times by subtracting the go cue time, then compared against the bin edges which are also go-cue-relative.

ii.
```python
a, b = ps_on[t] - g, ps_off[t] - g
photostim = ((bin_hi > a) & (bin_lo < b)).astype(np.float32)
```

iii. Same go-cue-relative coordinate system as the neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `trial_instruction` ('left'/'right') and `outcome` ('hit'/'miss'/'ignore'). A hit means the animal licked the instructed side, a miss means it licked the other side, and ignore means no lick.

ii.
```python
other = {'left': 'right', 'right': 'left'}
choice_map = {'left': 0, 'right': 1, 'no lick': 2}
...
if outcome[t] == 'hit':
    ch = instruction[t]
elif outcome[t] == 'miss':
    ch = other[instruction[t]]
else:
    ch = 'no lick'
```

iii. The agent verified this against recorded lick times: "Cross-checked against the recorded lick times (>99% agreement)."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The derived choice is coded as 0=left, 1=right, 2=no lick and repeated across all 80 bins as a per-trial value.

ii.
```python
out = np.stack([
    np.full(NBINS, choice_map[ch], dtype=np.int64),
    np.full(NBINS, outcome_map[outcome[t]], dtype=np.int64),
    np.full(NBINS, 1 if early[t] == 'early' else 0, dtype=np.int64),
    tclass,
])
```

iii. Choice is a per-trial categorical value repeated across time bins.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the `outcome` column of the trials table, which holds 'ignore', 'miss', and 'hit'.

ii.
```python
outcome = _str(tr['outcome'][:])
```

iii. The trials table stores outcome explicitly with the three categories requested.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to 0=ignore, 1=miss, 2=hit and repeated across all 80 bins.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
...
np.full(NBINS, outcome_map[outcome[t]], dtype=np.int64),
```

iii. Standard categorical encoding, same order as the reference.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds 'early' and 'no early'.

ii.
```python
early = _str(tr['early_lick'][:])
```

iii. The trials table flags early licking explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0=no, 1=yes and repeated across all 80 bins.

ii.
```python
np.full(NBINS, 1 if early[t] == 'early' else 0, dtype=np.int64),
```

iii. Binary encoding of the early lick flag.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose data is (n_frames, 3) = tongue_x, tongue_y, tongue_likelihood, with matching timestamps.

ii.
```python
key = 'acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking'
...
vts = f[key + '/timestamps'][:]
vdata = f[key + '/data'][:]
tongue_y = vdata[:, 1].astype(np.float64)
tongue_vis = vdata[:, 2] > LIKELIHOOD_THRESH
```

iii. Same tongue tracking data source as the reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Multiple steps:
1. Frames with DLC likelihood <= 0.9 are marked as not visible (reference uses 0.5)
2. A 5-sigma velocity outlier rejection is applied to the y-position trace, following Wang, Kurgyis et al., and outliers are imputed from neighboring frames
3. Session-wide 40th and 60th percentiles computed over **raw visible frames** (not bin means)
4. Per trial, frames are binned into 50 ms bins, mean y computed per bin, and digitized into classes 0/1/2. Bins with no visible frames get class 3.

ii.
```python
LIKELIHOOD_THRESH = 0.9
VELOCITY_SIGMA = 5.0
```

```python
tongue_y, tongue_vis = clean_marker(tongue_y, tongue_vis)
...
p40, p60 = np.percentile(tongue_y[tongue_vis], [40, 60])
```

```python
# Per trial binning
b_idx = np.floor((vts[lo:hi] - (g + OFF_START)) / BIN_SIZE).astype(int)
...
ymean = np.where(nvis > 0, ysum / np.maximum(nvis, 1), np.nan)
seen = nvis > 0
tclass[seen] = np.where(ymean[seen] < p40, 0,
                        np.where(ymean[seen] <= p60, 1, 2))
```

iii. From the docstring: "Marker traces are cleaned with the five-sigma velocity outlier criterion of Wang, Kurgyis et al. and outliers are imputed from neighbouring frames." The agent applied additional cleaning not present in the reference solution.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Session-wide 40th/60th percentiles of visible-frame y-positions serve as class boundaries. Categories: 0 (< p40), 1 (p40 to p60 inclusive), 2 (> p60), 3 (not visible). Note the boundary handling uses `<` p40 and `<= p60`, differing from the reference which uses `np.digitize`.

ii.
```python
p40, p60 = np.percentile(tongue_y[tongue_vis], [40, 60])
...
tclass[seen] = np.where(ymean[seen] < p40, 0,
                        np.where(ymean[seen] <= p60, 1, 2))
```

iii. The percentiles follow the instructions (40th/60th, per-session). Percentiles are computed over raw visible frames rather than 50ms bin means.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps share the same session clock as spikes and events. Per trial, frames are selected by searchsorted on the trial window and assigned to bins by their offset from `go + OFF_START`.

ii.
```python
lo = np.searchsorted(vts, g + OFF_START)
hi = np.searchsorted(vts, g + OFF_END)
...
b_idx = np.floor((vts[lo:hi] - (g + OFF_START)) / BIN_SIZE).astype(int)
np.clip(b_idx, 0, NBINS - 1, out=b_idx)
```

iii. Same go-cue-relative bin grid as the neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Sessions with no good units**: dropped (`classification` check)
- **Sessions with bad video**: non-monotonic video timestamps cause session exclusion; low video coverage causes session exclusion
- **Trials outside obs_intervals**: excluded
- **Trials with no tone onset**: get NaN input, then dropped in post-hoc filtering
- **Trials with zero spikes across all units**: dropped in post-hoc filtering
- **Frames with no tracked tongue**: marked not visible, bins with no visible frames get class 3
- **Velocity outliers in tongue tracking**: imputed from neighboring frames

ii.
```python
if np.any(np.diff(vts) <= 0):
    info['excluded'] = 'non-monotonic video timestamps'
    return None
```

```python
valid = [i for i, x in enumerate(input_trials)
         if np.all(np.isfinite(x)) and neural_trials[i].sum() > 0]
```

iii. The agent discovered and handled several edge cases during iterative development (trajectory steps 77-92).

## 10-a. What are the most time-consuming steps of the code?

i. Reading each NWB file and computing firing rates via per-unit searchsorted loops. The AI uses multiprocessing (16 workers) to parallelize session processing. Full conversion takes a few minutes with parallelism.

ii.
```python
with Pool(args.nproc) as pool:
    for res, info in pool.imap(process_session, [(fn, name2anc) for fn in files]):
```

iii. From trajectory step 65: "Test conversion works and is fast (2 s for 6 sessions)." Full conversion produced a 6.5 GB file.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops remain:
1. Per-unit loop for spike binning (one searchsorted per unit)
2. Per-trial loop for tongue y-position processing
3. Per-trial loop for constructing input/output arrays

ii.
```python
for i, u in enumerate(unit_idx):
    st = spike_ds[sp_start[u]:sp_index[u]]
    ...
```

```python
for k, t in enumerate(trials):
    g = go[t]
    ...
```

iii. The per-unit loop cannot be fully vectorized due to ragged spike arrays. The per-trial loop for inputs/outputs could potentially be vectorized (the reference does vectorize most of these operations).

## 10-c. What processing does the code repeat multiple times?

i. The per-trial loop recomputes bin centers/edges on each iteration (though these are constant). The Allen ontology is loaded once and passed to all workers. The main computation is single-pass per session.

ii.
```python
bin_centers = OFF_START + BIN_SIZE * (np.arange(NBINS) + 0.5)
bin_lo = OFF_START + BIN_SIZE * np.arange(NBINS)
bin_hi = bin_lo + BIN_SIZE
```
These are computed once per session call but could be module-level constants.

iii. No major redundant computation; the code is a single pass over the files with parallel processing.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads and processes the Allen ontology JSON to produce detailed hemisphere-labeled brain region names (e.g., "left ALM", "right Thalamus") using a complex hierarchy walk. This is significantly more processing than what the reference does (simple string splitting). The detailed region labels are used in the output but represent more work than necessary.

Additionally, the velocity outlier cleaning of tongue tracking data is extra processing not present in the reference and adds computation without clear benefit for the decoder task.

ii.
```python
def load_ontology():
    with open(ONTOLOGY_FILE) as f:
        root = json.load(f)['msg'][0]
    ...

def region_of(anno_name, name2anc, is_alm_probe):
    anc = name2anc.get(anno_name, [])
    for acr, label in DIVISIONS:
        if acr in anc:
            return label
    ...
```

iii. The agent spent significant effort reproducing the reference code's 14-area grouping (trajectory steps 28, 48-52), but the decoder task doesn't require this specific grouping.
