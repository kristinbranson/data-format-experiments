# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI glob-discovers every `data_structure_*.mat` in the two ephys folders, pairs it with a same-named motion-energy file when present, opens session files with `h5py`, and catches any session exception in `main`, skipping that session. Although a SciPy loader was appended later, it is defined after the `main()` call and is never invoked. The completed run retained 34 of 47 discovered files and skipped 13.

ii.
```python
for task_dir in EPHYS_DIRS:
    ...
    for data_file in sorted(d.glob('data_structure_*.mat')):
...
try:
    neural_trials, input_trials, output_trials, subject, bri = build_session(sess)
except Exception as e:
    ...
    continue
```

iii. The notes justify restricting loading to neural-recording folders because the decoder requires neural input. They recognized mixed MAT formats and the need for a fallback, but the full-run notes explicitly acknowledge that 13 sessions were skipped and that the result was incomplete.

## 1-b. How are the data split into subjects?

i. The subject is parsed from the filename as the text between `data_structure_` and the next underscore. Subjects are added in first-seen order, and each successfully loaded session gets an index into that list.

ii.
```python
m = re.match(r'data_structure_([^_]+)_(.+)\.mat', data_file.name)
animal, date = m.groups()
...
subject_to_idx[subject] = len(data['subjects'])
data['subject_idx'].append(subject_to_idx[subject])
```

iii. The notes identify filename/session organization and report the resulting 13 subjects; no separate rationale is given for filename parsing.

## 1-c. How are the data split into sessions?

i. Each discovered `data_structure_<animal>_<date>.mat` is treated as one session and appended once to each top-level session list if processing succeeds.

ii.
```python
sessions.append(SessionInfo(animal, date, task_dir, data_file, motion_file))
...
data['neural'].append(neural_trials)
data['input'].append(input_trials)
data['output'].append(output_trials)
```

iii. The notes say the recording-and-video files are session objects and that only ephys sessions are appropriate for a neural decoder.

## 1-d. How are the data split into trials?

i. `bp.Ntrials` defines the trial count. Neural spike fields use MATLAB 1-based trial labels; behavioral arrays are sliced to `n_trials`; trajectory and motion arrays are indexed by trial position. One matrix/input/output is produced for every integer trial from zero through `n_trials-1`.

ii.
```python
n_trials = int(np.array(bp['Ntrials']).squeeze())
mats = [np.zeros((n_units, n_time), dtype=np.float32) for _ in range(n_trials)]
for tr in range(1, n_trials + 1):
    mask = trials == tr
```

iii. The notes describe `obj.trials` as linking behavior, ephys, and video trialwise, but the implementation relies mostly on array position and cluster trial labels.

## 1-e. How are trials filtered based on quality controls?

i. They are not filtered. Early-lick, stimulation, and trials beyond the ephys recording remain, and all `Ntrials` entries are emitted.

ii.
```python
for tr in range(n_trials):
    input_trials.append(...)
    output_trials.append(out)
```

iii. The notes discuss trial inclusion criteria from specialized reference analyses, but no concrete filtering decision was implemented or justified. Verification found one all-zero neural trial.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural matrices come from every probe/unit exposed through `obj.clu`, specifically each unit's `trial`, `trialtm`, and (for filtering) `tm`. Probe location metadata is read but not used to select probes.

ii.
```python
for key in ['tm', 'trial', 'trialtm', 'quality', 'site']:
    ...
    unit[key] = np.array(target[()]).squeeze()
```

iii. The notes correctly concluded that neural trial matrices had to be reconstructed from raw cluster `trial` and `trialtm` fields rather than from an assumed stored `trialdat`.

## 2-b. How is the `neural` data processed?

i. For each retained unit and trial, the code histograms raw `trialtm` values into 25 ms bins over -2.5 to 2.5 seconds and stores spike counts. It applies no go-cue subtraction, conversion to Hz, Gaussian smoothing, normalization, or baseline correction.

ii.
```python
edges = np.concatenate([[time[0] - BIN_SIZE_S / 2], time + BIN_SIZE_S / 2])
...
mats[tr - 1][ui] = np.histogram(trialtm[mask], bins=edges)[0].astype(np.float32)
```

iii. The notes say the intended approach was reference-style go-cue alignment and low-rate filtering. They do not justify retaining unsmoothed counts or omitting alignment.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are retained if `number of tm spikes / max(n_trials seconds, 1)` exceeds 1 Hz. Manual quality labels are loaded but ignored; sessions are not restricted to author-selected probes or required to have at least ten post-filter units.

ii.
```python
session_dur = max(n_trials * 1.0, 1.0)
mean_fr = tm.size / session_dur
if mean_fr > 1.0:
    kept.append(u)
```

iii. The notes cite the paper's >1 Hz rule and discuss a ten-unit session criterion. They acknowledge that the exact threshold implementation still needed reconciliation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It is not aligned to go cue. `trialtm` is histogrammed directly; `bp.ev.goCue` is never passed into `build_neural_trials` or subtracted there.

ii.
```python
def build_neural_trials(units, n_trials):
    ...
    trialtm = np.array(u['trialtm']).astype(float).ravel()
    ... np.histogram(trialtm[mask], bins=edges)
```

iii. The comments and notes claim all streams should be go-cue aligned, but the code does not implement that claim for neural data.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output uses 25 ms bins (200 bins across five seconds). Spikes are rebinned by histogramming; video-derived signals are interpolated to the 25 ms centers. This differs from the reference 5 ms resolution.

ii.
```python
BIN_SIZE_S = 0.025
T_START = -2.5
T_END = 2.5
```

iii. The code calls this an “initial” conservative binning choice. No final reference-based justification is recorded, despite the reference parameter inspection described in the notes.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is a synthetic common grid from the constants `T_START`, `T_END`, and `BIN_SIZE_S`; raw go-cue values are not used to construct the input.

ii.
```python
edges = np.arange(T_START, T_END + BIN_SIZE_S, BIN_SIZE_S)
centers = edges[:-1] + BIN_SIZE_S / 2
```

iii. The notes state that a continuous time-varying input directly matches the requested decoder input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Bin centers from -2.4875 through 2.4875 seconds are cast to `float32`, given a leading singleton feature dimension, and copied for every trial.

ii.
```python
return centers.astype(np.float32)
...
input_trials.append(time[None, :].astype(np.float32))
```

iii. No additional processing rationale is given beyond matching the continuous-input specification.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It has the same number of bins and nominal edges as the neural matrices, but semantic alignment is wrong: the input labels zero as go cue while neural spikes were binned relative to trial start.

ii.
```python
time = common_time_axis()
edges = np.concatenate([[time[0] - BIN_SIZE_S / 2], time + BIN_SIZE_S / 2])
```

iii. The notes assert a shared go-cue-relative axis, but no sanity check caught the missing go-cue subtraction in neural processing.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is derived from `bp.R`, `bp.L`, and the computed outcome/ignore status.

ii.
```python
R = np.ravel(bp['R'])[:n_trials] > 0
L = np.ravel(bp['L'])[:n_trials] > 0
lick[outcomes == 2] = 2
```

iii. The notes planned to use `R`, `L`, and lick-event logic, with none for ignore/no-lick trials.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The code maps instructed left directly to left and instructed right directly to right, then maps ignores to none. It does not reverse the instructed side on incorrect/miss trials, so miss-trial lick direction is wrong. The scalar class is repeated across time.

ii.
```python
lick[L] = 0
lick[R] = 1
lick[outcomes == 2] = 2
```

iii. No justification addresses miss trials. The notes only state the intended left/right/none mapping.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is inferred from `bp.protocol.nums`, if available; the task directory is accepted as an argument but not used in the actual mapping. `bp.autowater`, the direct reference variable, is ignored.

ii.
```python
protocol = bp.get('protocol', {})
nums = protocol.get('nums', None) if isinstance(protocol, dict) else None
```

iii. The notes initially proposed context/protocol/block identity and acknowledged after the full run that the constant context output was incomplete.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. All trials default to class 0. Only when exactly two finite protocol numbers occur does the larger number become class 1. Metadata names class 0 `DR` and class 1 `WC`, the reverse ordering of the requested/reference mapping. In the produced data every bin is class 0/DR.

ii.
```python
ctx = np.zeros(n_trials, dtype=np.int64)
...
if len(uniq) == 2:
    ctx = (vals == uniq.max()).astype(np.int64)
```

iii. The code calls this a “conservative initial mapping”; the notes explicitly report that behavioral context remained constant and required more work.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome uses `bp.hit`, `bp.miss`, and `bp.no`.

ii.
```python
hit = np.ravel(bp['hit'])[:n_trials] > 0
miss = np.ravel(bp['miss'])[:n_trials] > 0
no = np.ravel(bp['no'])[:n_trials] > 0
```

iii. The notes identify these exact fields and cite the reference `getOutcome` function.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Default/`no` is ignore (2), misses are incorrect (0), and hits are correct (1); the per-trial class is repeated at every time bin.

ii.
```python
out = np.full(n_trials, 2, dtype=np.int64)
out[miss] = 0
out[hit] = 1
out[no] = 2
```

iii. The notes justify the direct hit/miss/no categorical mapping required by the task.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses the first camera in `obj.traj`, that camera's `ts` and `frameTimes`, and blindly selects landmark index 0 (`ts[0,...]` for HDF5 or `ts[...,0]` for SciPy). It does not inspect `featNames`, combine the side and bottom tongue features, or apply video/behavior bitcode offset correction. `bp.ev.goCue` is subtracted.

ii.
```python
x = ts[0, 0, :].astype(float)
y = ts[0, 1, :].astype(float)
p = ts[0, 2, :].astype(float)
...
tv = ft[valid] - align_times[tr]
```

iii. The notes say trajectory x/y/confidence makes tongue velocity derivable, but acknowledge that exact feature indices still needed identification.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Frames with confidence over 0.5 and finite values are retained. Euclidean displacement between consecutive retained points is divided by their time difference, assigned to midpoint times, and linearly interpolated onto the common grid. There is no coordinate smoothing, contiguous-run handling, or two-view normalization/averaging.

ii.
```python
valid = ... & (p > 0.5)
speed = np.sqrt(np.diff(xv)**2 + np.diff(yv)**2) / dt
tmid = (tv[:-1] + tv[1:]) / 2
yint = np.interp(time, tmid[finite], speed[finite], left=np.nan, right=np.nan)
```

iii. The notes characterize this as an iterative improvement that yielded visible and not-visible states, but do not justify deviations from the reference likelihood threshold and smoothing.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. One median is calculated over every finite tongue value in a session. Values below it become 0, values at or above it become 1, and NaNs become 2.

ii.
```python
thr = np.nanmedian(arr2d[valid])
out[valid & (arr2d < thr)] = 0
out[valid & (arr2d >= thr)] = 1
```

iii. This is explicitly justified by the prompt's per-session 50th-percentile requirement and missing/not-visible class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Camera frame times have the trial go-cue time subtracted and velocities are interpolated to the 25 ms grid. The required session video-clock offset is not subtracted, and the neural stream itself is not go-cue aligned, so the modalities are not mutually aligned.

ii.
```python
align_times = np.ravel(bp['ev']['goCue']).astype(float)[:n_trials]
tv = ft[valid] - align_times[tr]
yint = np.interp(time, tmid[finite], speed[finite], left=np.nan, right=np.nan)
```

iii. The notes found the reference `findVideoOffset` mechanism but did not implement it here; they nevertheless claimed video alignment was improved.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It is derived from no raw variable. The code never searches `featNames` for `top_paw` and creates an all-missing paw vector for every trial.

ii.
```python
paw = np.full(time.shape, 2, dtype=np.int64)
```

iii. The notes state that inspected feature names appeared to lack paw landmarks and therefore chose to leave paw unavailable, although the human solution locates `top_paw` in the bottom view.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. No velocity is computed; every bin is directly assigned category 2.

ii.
```python
paw = np.full(time.shape, 2, dtype=np.int64)
```

iii. The notes justify this as avoiding fabricated movement when the feature was believed unavailable.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. It is not thresholded. All bins are `not_visible` (2), so the requested session median split never occurs.

ii.
```python
paw = np.full(time.shape, 2, dtype=np.int64)
```

iii. The full-run notes acknowledge the paw output is degenerate and incomplete.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. It is not aligned because no paw samples or times are loaded; only an all-2 array with matching length is emitted.

ii.
```python
np.full(time.shape, 2, dtype=np.int64)
```

iii. There is no alignment justification beyond the claim that paw data were unavailable.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It comes from each session's standalone `motionEnergy_<animal>_<date>.mat`, specifically `me.data`, recursively coerced into one numeric vector per trial when possible.

ii.
```python
m = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
return m['me'].flat[0]
...
data = getattr(me, 'data', None)
```

iii. The notes correctly identify the motion-energy files, their older MATLAB format, and that their data should be interpolated onto a common time axis.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The existing per-frame trace is not spatially reprocessed. Synthetic frame times at 400 Hz are created, shifted by a hard-coded 0.5 seconds and trial go cue, linearly interpolated to 25 ms centers, and edge NaNs are nearest-filled. It does not use actual camera frame times or the bitcode-derived session offset.

ii.
```python
frame_times = np.arange(1, x.size + 1, dtype=float) / 400.0
old_t = frame_times - 0.5 - align_times[tr]
y = np.interp(time, old_t[valid], x[valid], left=np.nan, right=np.nan)
y[:first] = y[first]
y[last+1:] = y[last]
```

iii. The notes cite 400 Hz and reference interpolation, but do not justify the 0.5-second constant or edge filling.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. A session-wide median across all finite aligned values defines categories 0 (`< median`) and 1 (`>= median`); missing values or an unavailable motion file become 2.

ii.
```python
motion_disc = discretize_session_median(motion_aligned, missing_code=2)
```

iii. This follows the prompt's per-session threshold and explicit no-video class; the notes report approximately balanced nonmissing categories.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. It is placed on the same nominal 25 ms array by the synthetic-time interpolation above, but the hard-coded offset is not the reference session offset and neural activity is not go-cue aligned. Thus equal array indices do not establish correct multimodal alignment.

ii.
```python
old_t = frame_times - 0.5 - align_times[tr]
y = np.interp(time, old_t[valid], x[valid], left=np.nan, right=np.nan)
```

iii. The notes intended reference-style alignment and spot checks, but those checks remained unfinished.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing motion files/streams and unavailable tongue or paw samples are encoded as category 2. Interpolation preserves tongue edge NaNs but nearest-fills motion-energy edges. Sessions raising any exception are silently skipped after logging. Some shape/count mismatches are truncated with `min(...)`; no trials are removed for missing ephys.

ii.
```python
for tr in range(min(n_trials, data_arr.size)):
...
except Exception as e:
    print(f'SKIP {sess.data_file.name}: {e!r}')
    continue
```

iii. The notes favor explicit missing categories over imputation, but also acknowledge the skipped sessions. The actual motion-energy edge fill conflicts with the stated no-imputation principle.

## 11-a. What are the most time-consuming steps of the code?

i. The AI does not benchmark individual steps. Likely expensive operations are recursively reading large HDF5 structures, nested unit-by-trial spike histogramming, and per-trial trajectory/motion interpolation; full conversion and decoder training were run separately.

ii.
```python
for ui, u in enumerate(units):
    for tr in range(1, n_trials + 1):
        ... np.histogram(...)
```

iii. CONVERSION_NOTES documents progress but contains no timing profile or explicit answer about bottlenecks.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The nested loop over every unit and every possible trial repeatedly constructs `trials == tr` masks and could be replaced by joint trial/time binning. Output assembly and per-trial video interpolation remain loops because trial streams are ragged, though portions could be batched after binning.

ii.
```python
for ui, u in enumerate(units):
    ...
    for tr in range(1, n_trials + 1):
        mask = trials == tr
```

iii. The AI gives no efficiency justification for these loops.

## 11-c. What processing does the code repeat multiple times?

i. `common_time_axis()` and per-trial constant output arrays are rebuilt per session/trial. HDF5 and SciPy session builders duplicate most assembly logic. Unit code scans all spikes once for the firing-rate estimate and again for histograms, and creates a trial mask for every unit-trial pair.

ii.
```python
time = common_time_axis()
...
np.full((1, time.size), lick[tr], dtype=np.int64)
```

iii. No rationale or explicit repeated-processing analysis is provided.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The recursive `read_bp` loads all behavioral subfields although only a subset is used; unit `quality`, `site`, and probe locations are read but discarded for filtering/region assignment. `L` is loaded although it is redundant with `R` in valid trials. The post-`main()` SciPy fallback code is parsed only after normal import and is unreachable during script execution, so it never contributes to the output.

ii.
```python
def read_bp(h):
    return read_any(h, h['/obj/bp'])
...
probe_locs = read_probe_locations(h)
if not probe_locs:
    probe_locs = ['ALM']
brain_region_idx = np.zeros(len(units), dtype=np.int64)
```

iii. The AI provides no explicit justification; the notes show these fields were explored for potential use, but several never became part of final processing.
