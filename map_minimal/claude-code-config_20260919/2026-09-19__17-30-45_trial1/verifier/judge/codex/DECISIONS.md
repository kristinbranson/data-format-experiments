# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers every `sub-*/*.nwb` file, sorts the paths, and processes them in a multiprocessing pool. Each worker opens one NWB session directly with `h5py`, writes a per-session cache pickle, and the parent later combines accepted sessions into the full pickle.

ii.
```python
files = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
jobs = [(f, args.cache_dir) for f in files]
with Pool(args.workers) as pool:
    for i, s in enumerate(pool.imap(_worker, jobs)):
        summaries.append(s)
```
```python
with h5py.File(path, 'r') as nwb:
    subject = nwb['general/subject/subject_id'][()].decode()
    trials = nwb['intervals/trials']
```

iii. The trajectory says NWB is the source format and that direct `h5py` loading was selected for speed. The final response describes the loading as direct NWB access and reports all 174 files considered, with 142 retained after curation.

## 1-b. How are the data split into subjects?

i. The subject ID is read from each NWB file. Accepted sessions' unique IDs are sorted into `subjects`, and each session receives its corresponding integer `subject_idx`.

ii.
```python
subject = nwb['general/subject/subject_id'][()].decode()
subjects = sorted({s['subject'] for s in kept})
subject_index = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array([subject_index[s['subject']] for s in kept], dtype=np.int64),
```

iii. The agent treated the NWB subject field as the canonical identifier. Its final report notes that the retained data contain 28 mice.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session; the filename without `.nwb` is the session name. Sessions are independently processed and cached, then accepted sessions are appended in the sorted-file/`imap` order.

ii.
```python
name = os.path.basename(path).replace('.nwb', '')
with h5py.File(path, 'r') as nwb:
    ...
for i, s in enumerate(kept):
    with open(s['cache'], 'rb') as fh:
        sess = pickle.load(fh)
    data['neural'].append(sess['neural'])
```

iii. The agent inferred the session boundary from the dataset's one-file-per-session organization. It additionally applied behavioral session-selection criteria, yielding 142 retained sessions.

## 1-d. How are the data split into trials?

i. Rows of `intervals/trials` define trials and are paired positionally with `go_start_times`; equality of their counts and containment of every go cue within its trial are asserted. Retained row indices are used consistently across streams.

ii.
```python
trial_start = trials['start_time'][:]
trial_stop = trials['stop_time'][:]
go_times = nwb['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
assert len(go_times) == len(trial_start), 'go cue count != trial count'
assert np.all((go_times >= trial_start) & (go_times <= trial_stop))
trial_idx = np.flatnonzero(keep)
```

iii. The trials table supplies explicit behavioral trial boundaries, while the assertions verify the presumed one-to-one mapping to go cues.

## 1-e. How are trials filtered based on quality controls?

i. The agent drops auto-water and free-water trials and trials not covered by every globally good unit's `obs_intervals`. It keeps photostimulation, early-lick, and ignore trials because they are required inputs/classes. It also rejects sessions failing control-trial performance above 65%, at least 50 correct trials per direction, two retained trials, or usable-unit requirements.

ii.
```python
keep = ~(auto_water | free_water)
if good.any():
    keep &= observed_trials(nwb, good, trial_start)
control = keep & ~stim_trial & ~early
rejected = (performance <= MIN_PERFORMANCE
            or n_left < MIN_CORRECT_PER_DIRECTION
            or n_right < MIN_CORRECT_PER_DIRECTION)
```

iii. The agent cited the data paper's session criteria and the method paper's exclusion of water delivered regardless of choice. It deliberately retained photostim, early-lick, and ignore trials because otherwise required decoder variables/classes would disappear. Observation filtering avoids representing unrecorded ephys as silence.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural arrays come from each unit's ragged `units/spike_times`, indexed by `spike_times_index`, and from `go_start_times` for temporal placement. Unit classification and per-trial flags determine which units contribute.

ii.
```python
spike_index = units['spike_times_index'][:]
spike_times = units['spike_times']
edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()
st = spike_times[starts[u]:spike_index[u]]
```

iii. The agent identified spike timestamps as the released raw neural representation and used the common NWB clock to relate them to go cues.

## 2-b. How is the `neural` data processed?

i. For each retained unit, cumulative insertion indices at all trial-bin edges are found with `searchsorted`; adjacent differences are spike counts and division by 0.05 converts them to firing rates in spikes/s. There is no smoothing, normalization, or baseline subtraction.

ii.
```python
idx = np.searchsorted(st, edges).reshape(n_trials, N_BINS + 1)
rates[i] = np.diff(idx, axis=1) / BIN_WIDTH
```

iii. The agent says this mirrors the reference preprocessing's go-aligned histogram and rate conversion, while replacing its window/bin specification with the assignment's required values.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units must have `classification == 'good'` and must also have `is_good_trials` true on every analyzed trial. A session is rejected if no unit survives.

ii.
```python
good = _str(nwb['units']['classification'][:]) == 'good'
good[good] = units_good_on_trials(nwb, good, trial_idx)
```
```python
flags = nwb['units']['is_good_trials'][:][good][:, trial_idx]
return flags.all(axis=1)
```

iii. The agent tied `classification` to the white paper's region-specific QC classifiers. It reasoned that a fixed neuron-by-time matrix needs the same usable neurons on every trial, so units failing any retained trial are removed; its final response says this removes about 0.8% of units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Relative edges from -2.5 to +1.5 s are added to each trial's absolute go-cue timestamp, and absolute spike times are histogrammed against those edges.

ii.
```python
BIN_EDGES = T_START + BIN_WIDTH * np.arange(N_BINS + 1)
edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()
```

iii. The agent states that spikes and behavioral events share the NWB session clock, so adding go-cue-relative offsets is sufficient and no stream-specific correction is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The result has 80 contiguous, non-overlapping 50-ms bins over [-2.5, +1.5) s. Raw spike timestamps are newly binned at that resolution; no further rebinning or sliding windows are used.

ii.
```python
BIN_WIDTH = 0.05
N_BINS = int(round((T_STOP - T_START) / BIN_WIDTH))
BIN_EDGES = T_START + BIN_WIDTH * np.arange(N_BINS + 1)
```

iii. The 50-ms resolution and window are fixed by the assignment. The agent explicitly chose back-to-back bins rather than the reference analysis's overlapping/sliding 40-ms windows.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `BehavioralEvents/sample_start_times`, trial start and go-cue timestamps, and bin centers. The last sample onset within a trial and no later than its go cue is selected.

ii.
```python
sample = nwb['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
lo = np.searchsorted(sample, trial_start, side='left')
hi = np.searchsorted(sample, go_times, side='right')
onsets[i] = sample[hi[i] - 1]
```

iii. The agent reasoned that early licking can replay the sample epoch, so the final onset before the go cue is the tone the animal ultimately answered.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Absolute tone onset is subtracted from each absolute bin center. If a trial lacks a recorded sample onset, the agent imputes tone onset as 1.85 s before the go cue, based on the nominal 0.65-s sample plus 1.2-s delay.

ii.
```python
tone[missing_tone] = go[missing_tone] - 1.85
time_from_tone = (go[:, None] + BIN_CENTERS[None, :]
                  - tone[:, None]).astype(np.float32)
```

iii. The normal calculation is direct elapsed time. The fallback was justified from the documented task timing to avoid missing decoder input values.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at exactly the same 80 go-cue-relative bin centers as the spike rates.

ii.
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2
time_from_tone = (go[:, None] + BIN_CENTERS[None, :] - tone[:, None])
```

iii. Both arrays derive their time coordinates from the same go cue and fixed bin grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses trial-table `photostim_onset`, `photostim_duration`, and `start_time`, plus each trial's go-cue time.

ii.
```python
onset = np.array([_parse_float(v) for v in _str(trials['photostim_onset'][:])])
duration = np.array([_parse_float(v) for v in _str(trials['photostim_duration'][:])])
on = trial_start + onset - go_times
off = on + duration
```

iii. The onset is stored relative to trial start and therefore must be converted to the go-relative axis; `N/A` denotes control trials.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. String values are parsed to floats, with `N/A` becoming NaN. A bin is one if its interval overlaps the stimulation interval, otherwise zero.

ii.
```python
valid = np.isfinite(on) & np.isfinite(off)
for i in np.flatnonzero(valid):
    stim[i] = ((BIN_EDGES[1:] > on[i]) &
               (BIN_EDGES[:-1] < off[i])).astype(np.float32)
```

iii. The agent chose an overlap rule so the time-varying flag denotes laser exposure during any part of a 50-ms bin.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Stimulation onset/offset are converted to time relative to the go cue and tested against the same relative bin edges used for neural activity.

ii.
```python
on = trial_start + onset - go_times
stim[i] = ((BIN_EDGES[1:] > on[i]) & (BIN_EDGES[:-1] < off[i]))
```

iii. Shared go-cue-relative bins guarantee corresponding neural and photostimulation columns cover the same intervals.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from trial `trial_instruction` and `outcome`: hit means instructed side, miss means the opposite side, and ignore means no lick.

ii.
```python
other = np.where(instruction == 'left', 'right', 'left')
licked = np.where(outcome == 'hit', instruction,
                  np.where(outcome == 'miss', other, 'no lick'))
```

iii. No direct choice column exists. The agent reports verifying this derivation against left/right lick event timestamps with perfect agreement.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Labels are mapped to integer codes 0 left, 1 right, and 2 no lick, then the per-trial code is repeated across all 80 time bins.

ii.
```python
CHOICE_VALUES = ['left', 'right', 'no lick']
choice = np.array([CHOICE_VALUES.index(v) for v in licked], dtype=np.int8)
outputs.append(np.stack([choice[j] * ones, ...]).astype(np.int8))
```

iii. The third class preserves ignore trials required by the task; repetition allows all outputs to share one time-indexed array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trial-table `outcome` field.

ii.
```python
outcome = _str(trials['outcome'][:])
```

iii. The raw categories already correspond exactly to ignore, miss, and hit.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped to integers 0 ignore, 1 miss, and 2 hit and repeated over all bins.

ii.
```python
OUTCOME_CODE = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome_code = np.array([OUTCOME_CODE[v] for v in outcome], dtype=np.int8)
```

iii. This is the categorical encoding required by the decoder format; repetition provides a uniform output shape.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from trial-table `early_lick`, tested for the string `early`.

ii.
```python
early = _str(trials['early_lick'][:]) == 'early'
```

iii. The NWB trial table already records the requested flag.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The Boolean flag is converted to `int8` (0 no, 1 yes) and repeated across all bins.

ii.
```python
early_code = early.astype(np.int8)
outputs.append(np.stack([..., early_code[j] * ones, ...]).astype(np.int8))
```

iii. This supplies the requested binary categorical output while retaining early-lick trials.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses timestamps and columns 1 (y coordinate) and 2 (DeepLabCut likelihood) of `Camera0_side_TongueTracking`.

ii.
```python
track = nwb['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
ts = track['timestamps'][:]
data = track['data'][:]
y = chunk[:, 1]
visible = chunk[:, 2] >= TONGUE_P_CUTOFF
```

iii. The agent identified this as the released side-camera tongue trajectory and used likelihood to distinguish genuine protrusion from unreliable coordinates.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each retained trial, visible frames (likelihood at least 0.5) are assigned to 50-ms go-aligned bins and averaged. Percentiles are then computed over all finite bin means from retained trials in that session. Empty/no-confident-detection bins receive class 3.

ii.
```python
counts = np.bincount(which[ok], minlength=N_BINS)
sums = np.bincount(which[ok], weights=y[ok], minlength=N_BINS)
y_binned[i, seen] = sums[seen] / counts[seen]
p40, p60 = np.percentile(y_binned[seen], TONGUE_PCTILES)
```

iii. The agent says likelihood is essentially binary and 0.5 is the standard cutoff. It computes thresholds on the same binned quantity being classified so visible classes have the intended approximate 40/20/40 proportions.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Values below p40 become 0; values from p40 through p60 become 1; values above p60 become 2; and bins without a visible frame become 3.

ii.
```python
classes = np.full((n_trials, N_BINS), 3, dtype=np.int8)
classes[seen] = np.where(vals < p40, 0,
                         np.where(vals <= p60, 1, 2)).astype(np.int8)
```

iii. This directly implements the agent's reading of the requested boundaries, including a separate not-visible class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera samples are selected from go-2.5 through go+1.5 and assigned by their go-relative timestamp to the same 50-ms edges as neural activity.

ii.
```python
lo, hi = np.searchsorted(ts, [go + T_START, go + T_STOP])
which = np.searchsorted(BIN_EDGES, ts[lo:hi] - go, side='right') - 1
```

iii. The agent relies on shared session-absolute timestamps for camera, spikes, and events, so no interpolation or clock-offset correction is applied.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Non-numeric photostim values become NaN/control; missing tones are imputed from nominal timing; absent video/confident tongue frames become the explicit not-visible class; uncovered trials and unusable units/sessions are removed. Worker exceptions are captured and mark a session rejected rather than terminating the entire conversion.

ii.
```python
except (TypeError, ValueError):
    return np.nan
```
```python
tone[missing_tone] = go[missing_tone] - 1.85
classes = np.full((n_trials, N_BINS), 3, dtype=np.int8)
```
```python
except Exception:
    return dict(session=os.path.basename(path), rejected=True,
                error=traceback.format_exc())
```

iii. The agent distinguishes genuine missing measurements (represented explicitly or conservatively imputed) from missing ephys coverage (excluded to avoid fabricated zero activity). The nominal task structure justifies its tone fallback.

## 10-a. What are the most time-consuming steps of the code?

i. The expensive work is reading large NWB spike/video arrays, per-unit spike binning, tongue binning, and serializing/reloading large per-session and final pickles. The code parallelizes session processing across 16 workers.

ii.
```python
with Pool(args.workers) as pool:
    for i, s in enumerate(pool.imap(_worker, jobs)):
```
```python
for i, u in enumerate(unit_ids):
    st = spike_times[starts[u]:spike_index[u]]
    idx = np.searchsorted(st, edges)
```

iii. The trajectory focused on runtime testing and multiprocessing; the final 9-GB dataset makes NWB I/O, spike searches, and pickle I/O the natural dominant costs.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Trials are already vectorized within each unit's spike search, but loops remain over units, units' observation intervals, trials for tone/tongue work and final assembly, stimulated trials, and region labels. The stimulation loop and some trial assembly could readily be broadcast; ragged spike trains and variable camera slices make complete vectorization less straightforward.

ii.
```python
for i, u in enumerate(unit_ids):
    ...
for i, go in enumerate(go_times):
    ...
for i in np.flatnonzero(valid):
    stim[i] = ...
for j in range(len(trial_idx)):
    neural.append(...)
```

iii. The agent prioritized readable per-session processing and parallelized sessions. It did vectorize all trials' spike edges within each unit and used `bincount` within each tongue trial.

## 10-c. What processing does the code repeat multiple times?

i. Scientific quantities are generally calculated once per session. However, accepted session arrays are serialized to cache files and then deserialized during final assembly, and some source trial arrays are transformed first for all trials and subsequently indexed to retained trials.

ii.
```python
with open(out_path, 'wb') as fh:
    pickle.dump({...}, fh, protocol=4)
...
with open(s['cache'], 'rb') as fh:
    sess = pickle.load(fh)
```

iii. The cache round trip supports multiprocessing and limits parent-process working memory. The agent's rationale was a scalable, self-contained full conversion rather than recomputing derived features.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes extensive diagnostic summaries, writes a separate `conversion_summary.json`, performs region remapping and validation assertions, and creates intermediate session cache files. Most summaries are retained in metadata, but the separate summary file and caches are not used by decoder training; `trial_stop` is otherwise only checked/passed to a helper that does not use it.

ii.
```python
summary.update(... mean_rate=float(rates.mean()))
with open(os.path.join(os.path.dirname(args.out) or '.',
                       'conversion_summary.json'), 'w') as fh:
    json.dump(session_info, fh, indent=1)
```

iii. These extras were used for validation, auditability, and parallel assembly during conversion. They do not change decoder arrays, though the final metadata intentionally preserves most diagnostics.
