# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively discovers local session directories under `data/one_cache`, requires an `alf` directory and a trials table, then directly reads Parquet and NumPy files. It does not use the ONE release index or restrict discovery to the Brainwidemap release. Full mode uses every locally discovered candidate; sample mode uses the first two.

ii.
```python
def list_session_dirs(data_root: Path):
    sessions = []
    for p in data_root.glob('*/Subjects/*/*/*'):
        if p.is_dir() and p.name.isdigit() and (p / 'alf').exists():
            sessions.append(p)
    return sorted(sessions)

data_root = Path('data/one_cache')
session_dirs = choose_sessions(list_session_dirs(data_root), mode)
```

iii. The notes say direct ALF reads replaced ONE loading because ONE pulled unnecessary arrays and took about 27 seconds for the sample, whereas direct reads took about 3 seconds. The agent regarded the local tree as an IBL ONE cache and chose speed over release-index-based loading.

## 1-b. How are the data split into subjects?

i. Subject IDs are parsed from the session directory path. A first-seen-order subject list and dictionary are built, and each retained session receives the corresponding `subject_idx`.

ii.
```python
rel = session_dir.relative_to(data_root)
parts = rel.parts
lab, _, subject, date, number = parts[0], parts[1], parts[2], parts[3], parts[4]
if subj not in subject_to_idx:
    subject_to_idx[subj] = len(subjects)
    subjects.append(subj)
out['subject_idx'].append(subject_to_idx[subj])
```

iii. The mapping plan explicitly says to extract subject IDs from local ONE-cache paths and preserve session-to-subject mapping. This avoids a ONE lookup but assumes the directory layout is canonical.

## 1-c. How are the data split into sessions?

i. Each directory matching `<lab>/Subjects/<subject>/<date>/<number>` with an `alf` child is treated as one session. Processing and output appending happen once per such directory.

ii.
```python
for p in data_root.glob('*/Subjects/*/*/*'):
    if p.is_dir() and p.name.isdigit() and (p / 'alf').exists():
        sessions.append(p)
...
for si, session_dir in enumerate(session_dirs):
```

iii. The notes describe “session-level processing” and merging probes within each session, consistent with the cache hierarchy and the reference's session unit.

## 1-d. How are the data split into trials?

i. Rows of `_ibl_trials.table.pqt` define trials. Arrays are indexed by the retained trial-row indices; neural and behavioral windows are separately extracted around each row's `stimOn_times`.

ii.
```python
stim_on = np.asarray(trials_df['stimOn_times'], dtype=float)
...
for i, tr in enumerate(np.where(valid)[0]):
    inp = np.vstack([...])
    out_trial = np.vstack([...])
```

iii. The agent treated the released trials table as already having one row per trial and planned explicit stimulus-aligned trial windows.

## 1-e. How are trials filtered based on quality controls?

i. Trials are retained only if stimulus onset is finite and mapped choice and prior are valid; sessions with fewer than two such trials are skipped. The code does not apply the reference 80 ms–2 s reaction-time mask, does not test first movement, and does not remove trials lacking full wheel/camera coverage. Missing behavioral samples are later converted to class 0.

ii.
```python
valid = np.isfinite(stim_on) & (choice >= 0) & (prior >= 0)
if valid.sum() < 2:
    continue
...
y[~np.isfinite(x)] = 0
```

iii. The notes promised an “explicit valid-trial mask” and jointly valid aligned intervals, but the final mask is narrower than that plan. The rationale was to remove invalid/no-choice trials and ensure decoder-compatible sessions; no justification is given for omitting reaction-time and stream-coverage QC.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural matrices come from `spikes.times.npy` and `spikes.clusters.npy` from every probe's pykilosort revision. `clusters.metrics.pqt` supplies QC labels; cluster channel and channel atlas-ID arrays supply region metadata.

ii.
```python
st_p = src / 'spikes.times.npy'
sc_p = src / 'spikes.clusters.npy'
metrics_p = src / 'clusters.metrics.pqt'
st = np.load(st_p)
sc = np.load(sc_p).astype(int)
metrics = pd.read_parquet(metrics_p)
```

iii. The mapping plan identifies spike times and assignments as the neural source and says probes should be merged within a session, matching the reference workflow.

## 2-b. How is the `neural` data processed?

i. Good clusters from all probes are renumbered into one session population, spikes are time-sorted, and each trial is histogrammed into 100 nonoverlapping 20 ms bins from -0.5 to +1.5 seconds. Values remain raw spike counts (`float32`); unlike the reference, they are not divided by 0.02 to form firing rates.

ii.
```python
bins = np.floor((ts - T_START) / BINSIZE).astype(int)
np.add.at(out, (cl[m], bins[m]), 1)
mats.append(out)
```

iii. The notes state “bin spikes into 20 ms bins” and emphasize matching `bin_spiking_data`, fixed bins, stimulus alignment, and probe merging. They never document a conversion from counts to Hz, and README calls the result spike-count matrices.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters with `label >= 1` are retained. Invalid spike cluster indices are removed. No Beryl mapping is performed and `void`/outside-brain units are not removed; numeric CCF region IDs are stringified for metadata.

ii.
```python
labels = metrics['label'].to_numpy() if 'label' in metrics.columns else np.ones(nclu)
good = labels >= 1
...
acr = np.array([str(x) for x in region_ids], dtype=object)
```

iii. The agent describes this as strict cluster QC matching well-isolated units. It acknowledges that numeric region IDs rather than acronyms are a semantic caveat caused by the atlas package being unavailable, but does not discuss the missing `void` exclusion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. For each valid trial, spike times in `[stimOn-0.5, stimOn+1.5)` are selected, stimulus onset is subtracted, and relative times determine bin indices.

ii.
```python
lo = np.searchsorted(spike_times, edges[0], side='left')
hi = np.searchsorted(spike_times, t0 + T_END, side='left')
ts = spike_times[lo:hi] - t0
bins = np.floor((ts - T_START) / BINSIZE).astype(int)
```

iii. The notes explicitly select `stimOn_times` because both the task and reference cache parameters use stimulus-onset alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The window uses 100 bins of 20 ms. Raw spikes are binned once; there is no later neural resampling or temporal rebinning.

ii.
```python
BINSIZE = 0.02
TIME_BINS = np.arange(T_START, T_END, BINSIZE)
N_BINS = len(TIME_BINS)
```

iii. The agent cites the reference caching parameters and the methods' 20 ms wheel/spike bins.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the configured relative window and bin size, with `stimOn_times` defining time zero. The same relative bin-center vector is used for every trial.

ii.
```python
TIME_BINS = np.arange(T_START, T_END, BINSIZE)
TIME_CENTERS = TIME_BINS + BINSIZE / 2
stim_on = np.asarray(trials_df['stimOn_times'], dtype=float)
```

iii. The notes call it a trial-relative time axis repeated across trials and link the choice to reference parameters.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code adds half a bin to the left-edge grid, producing centers from -0.49 to 1.49 seconds, casts them to `float32`, and copies them into each trial input.

ii.
```python
TIME_CENTERS = TIME_BINS + BINSIZE / 2
inp = np.vstack([TIME_CENTERS.astype(np.float32), ...])
```

iii. Bin centers were chosen to represent each 20 ms neural/behavioral bin on a common continuous time axis.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Neural bin indices use the same `T_START` and `BINSIZE`, while the input gives those bins' centers, so input column `j` corresponds to neural column `j`.

ii.
```python
bins = np.floor((ts - T_START) / BINSIZE).astype(int)
TIME_CENTERS = TIME_BINS + BINSIZE / 2
```

iii. The mapping and sanity-check plan explicitly require the time vector and stimulus-aligned spike bins to match.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived solely from the ordered `probabilityLeft` column. A change in probability starts a new block.

ii.
```python
trial_in_block = trial_number_in_block(trials_df['probabilityLeft'].to_numpy())
```

iii. The notes say the trials table lacks a separate required block variable, so consecutive equal-probability runs define blocks.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The agent loops through all unfiltered trials and assigns a 1-based within-block counter, then broadcasts the retained trial's value over 100 bins. The reference counter is 0-based, so this is offset by one, although computing before filtering correctly preserves the animal's actual block position.

ii.
```python
if i == 0 or p != cur:
    cur = p
    c = 1
else:
    c += 1
out[i] = c
...
np.full(N_BINS, trial_in_block[tr], dtype=np.float32)
```

iii. The notes specify a within-block index and continuous per-trial/broadcast representation but do not justify the 1-based convention.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes directly from the trials table's `choice` column.

ii.
```python
choice = map_choice(trials_df['choice'].to_numpy())
```

iii. The mapping plan says to use the IBL choice code after checking its sign convention.

## 5-b. What processing is involved in computing `output` *Choice*?

i. IBL `+1` is mapped to left/class 0, `-1` to right/class 1, and all other values to -1 so the trial mask drops them. The selected scalar class is repeated at every time bin.

ii.
```python
out[vals == 1] = 0
out[vals == -1] = 1
...
np.full(N_BINS, choice[tr], dtype=np.int64)
```

iii. This follows the requested left=0/right=1 encoding and excludes no-response trials.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes directly from `probabilityLeft` in the trials table.

ii.
```python
prior = map_prior(trials_df['probabilityLeft'].to_numpy())
```

iii. The notes identify this column as the block prior required by the task.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values close to 0.2, 0.5, and 0.8 are mapped to 0, 1, and 2. Anything else is -1 and filtered. The class is broadcast over the trial's 100 bins.

ii.
```python
out[np.isclose(vals, 0.2)] = 0
out[np.isclose(vals, 0.5)] = 1
out[np.isclose(vals, 0.8)] = 2
```

iii. The mapping is explicitly required by the task and documented in the plan.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
tp = session_alf / '_ibl_wheel.timestamps.npy'
pp = session_alf / '_ibl_wheel.position.npy'
return np.load(tp), np.load(pp)
```

iii. The notes identify position and timestamps as the raw sources, consistent with IBL wheel data.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Finite samples are retained, duplicate timestamps are removed, and `np.gradient(position, timestamps)` computes velocity directly on the irregular sampling grid. That signed velocity is linearly interpolated at trial bin centers. The code neither uses absolute value (so it is velocity, not speed) nor the reference SessionLoader's 1 kHz interpolation and 20 Hz low-pass filtering.

ii.
```python
uniq_t, uniq_idx = np.unique(timestamps, return_index=True)
vel = np.gradient(position, timestamps)
y = np.interp(x, timestamps, vel, left=np.nan, right=np.nan)
```

iii. The notes say “differentiate/interpolate” and document deduplication as a fix for gradient warnings. They claim wheel speed, but do not justify preserving the sign or omitting the reference filter.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Finite interpolated values from all retained trials in a session are pooled. Session-level 1/3 and 2/3 quantiles define classes: values at or below q1 are 0, `(q1,q2]` are 1, and above q2 are 2. Nonfinite samples are forced to 0.

ii.
```python
q1, q2 = np.quantile(allv, [1/3, 2/3])
y[x > q1] = 1
y[x > q2] = 2
y[~np.isfinite(x)] = 0
```

iii. The agent chose session-aware quantile thresholds to avoid degenerate classes and create approximately equal class sizes, matching the reference's session-level tertile concept.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Velocity is evaluated at `stimOn_times + TIME_CENTERS`, the same 100 centers represented by the neural bins.

ii.
```python
for t0 in stim_on:
    x = t0 + TIME_CENTERS
    y = np.interp(x, timestamps, vel, left=np.nan, right=np.nan)
```

iii. The notes require all time-varying streams to share the stimulus-aligned 20 ms grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses every available left and right camera `ROIMotionEnergy.npy` stream and the corresponding `_ibl_<side>Camera.times.npy` files.

ii.
```python
for tp, mp in [(left_t, left_me), (right_t, right_me)]:
    if tp is not None and mp is not None and tp.exists() and mp.exists():
        streams.append((np.load(tp), np.load(mp).astype(np.float32)))
```

iii. The notes identify both camera ROI streams and propose combining them “sensibly” if both are available. This differs from the reference's left-preferred, otherwise-right selection.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Each camera trace is linearly interpolated at trial bin centers. Where both views are finite they are averaged; where one is finite that one is used; where neither is finite the value stays NaN until discretization converts it to class 0. No filtering or normalization is applied.

ii.
```python
ys.append(np.interp(x, ts[:len(me)], me, left=np.nan, right=np.nan))
valid = np.isfinite(arr)
summed = np.where(valid, arr, 0.0).sum(axis=0)
y = np.divide(summed, denom, out=np.full(arr.shape[1], np.nan, dtype=np.float32), where=denom > 0)
```

iii. The agent says the released motion-energy stream can be used directly and averaging available views handles missing samples. It documents replacing `nanmean` to suppress all-NaN warnings.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. As for wheel, finite values across all retained trials in a session are split at the 1/3 and 2/3 quantiles. Missing values become low/class 0.

ii.
```python
whisk_bins, whisk_thr = discretize_tertiles(whisk_trials)
```

iii. Session-aware quantiles were selected to balance the three requested categorical classes.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Camera motion energy is interpolated at `stimOn_times + TIME_CENTERS`, giving one value for every neural bin center.

ii.
```python
for t0 in stim_on:
    x = t0 + TIME_CENTERS
    ys.append(np.interp(x, ts[:len(me)], me, left=np.nan, right=np.nan))
```

iii. The common grid was an explicit mapping-plan and sanity-check requirement.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions are skipped for missing trial columns, fewer than two basic-valid trials, missing spikes, wheel, or motion energy, or any caught exception. Missing behavioral samples within otherwise retained trials become class 0. Duplicate wheel timestamps are deduplicated. Missing cluster labels default to all-good; missing region files yield `"void"` strings, while invalid cluster/channel indices are guarded. This is permissive and can silently turn data defects into labels.

ii.
```python
except Exception as e:
    print('skip session due to error', session_dir, repr(e))
...
labels = metrics['label'].to_numpy() if 'label' in metrics.columns else np.ones(nclu)
y[~np.isfinite(x)] = 0
```

iii. The notes emphasize robust full conversion, warning removal, and session skipping. They acknowledge numeric region metadata as a remaining limitation, but do not justify treating missing behavior as genuine low activity or defaulting missing QC labels to good.

## 10-a. What are the most time-consuming steps of the code?

i. Loading large spike arrays and per-trial spike binning are the dominant work; full conversion took about 621 seconds. The agent specifically found ONE-based loading expensive because it loaded extra spike/template/waveform arrays, then sped it up with direct local reads.

ii.
```python
st = np.load(st_p)
sc = np.load(sc_p).astype(int)
...
for t0 in stim_on:
    ...
    np.add.at(out, (cl[m], bins[m]), 1)
```

iii. The notes report a roughly 10x sample speedup (about 27 s to 3 s) from direct ALF reads and estimate roughly 0.7–1.2 seconds per session, although actual session times vary with trial/neuron counts.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial spike-binning loop, wheel interpolation loop, motion-energy trial/view loops, trial-in-block loop, per-trial output assembly, and neuron-region indexing loops are vectorization candidates. The biggest potential gain is assigning spikes to trial/bin coordinates in bulk; output construction and trial counters are simpler mechanical candidates.

ii.
```python
for t0 in stim_on:                       # spike binning
for t0 in stim_on:                       # wheel interpolation
for t0 in stim_on:
    for ts, me in streams:               # motion energy
for i, tr in enumerate(np.where(valid)[0]):
```

iii. The agent's notes identify speed as important but only document replacing ONE loads; they do not discuss these remaining loops or why they were retained.

## 10-c. What processing does the code repeat multiple times?

i. It independently builds the same trial query times for wheel and motion energy, interpolates each stream trial-by-trial, repeatedly fills constant 100-element choice/prior/trial-number arrays, and repeatedly scans revision/file patterns. Probe offsets are also recomputed by summing all preceding cluster lengths.

ii.
```python
x = t0 + TIME_CENTERS
...
np.full(N_BINS, trial_in_block[tr], dtype=np.float32)
np.full(N_BINS, choice[tr], dtype=np.int64)
np.full(N_BINS, prior[tr], dtype=np.int64)
```

iii. The notes do not identify repeated processing. Their optimization effort focused on avoiding unnecessary ONE-loaded arrays rather than consolidating repeated small operations.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `edges = t0 + TIME_BINS` is created only to use `edges[0]`; the rest of that vector is discarded. Parsed `lab`, `date`, `number`, `trial_path`, loop index `si`, and imports `ONE`/`defaultdict` are unused. Optional diagnostic plots are intentionally not part of the pickle. Region loading is used by output metadata and therefore is not discarded.

ii.
```python
edges = t0 + TIME_BINS
lo = np.searchsorted(spike_times, edges[0], side='left')
...
trials_df, trial_path = load_trials_table(session_alf)
```

iii. The agent documents that the earlier ONE approach loaded unnecessary templates/waveforms and removed that overhead. It does not mention the smaller dead computations and variables remaining in the final script.
