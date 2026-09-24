# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers only `data_structure_*.mat` files in `/app/data/Ephys_Behavior`, sorts them, and loads each session's `obj` with `mat73`. Motion energy is loaded separately from a same-directory `motionEnergy_<subject>_<date>.mat` file with SciPy. Thus “full” means all 25 files in that one folder, not all reference-selected fixed- and randomized-delay sessions.

ii.
```python
def discover_sessions(base=Path('/app/data')):
    files = sorted((base / 'Ephys_Behavior').glob('data_structure_*.mat'))
    return files
...
obj = mat73.loadmat(str(path))['obj']
```

iii. The notes say this folder was chosen because it matches the paper's 25-session DR dataset and supplies the WC/DR variables. They acknowledge that the raw data contain other task-family folders and call this an “initial conversion focused on Ephys_Behavior.”

## 1-b. How are the data split into subjects?

i. Each session's subject comes from `obj.meta.anm`, falling back to the animal token in the filename. Subjects are added in first-session order, and `subject_idx` records the corresponding index for each retained session.

ii.
```python
subject = obj.get('meta', {}).get('anm', re.search(r'data_structure_([^_]+)_', path.name).group(1))
...
if subj not in subject_map:
    subject_map[subj] = len(subjects)
    subjects.append(subj)
subject_idx.append(subject_map[subj])
```

iii. The notes identify subject IDs in both the raw metadata and filenames and report the resulting subject counts, while noting a one-subject discrepancy from the paper.

## 1-c. How are the data split into sessions?

i. Every discovered MATLAB file is treated as one session and processed independently. A session is omitted after processing if it has fewer than two retained trials or fewer than ten parsed units.

ii.
```python
for sf in session_files:
    neural, inputs, outputs, info = process_session(sf, show_processing=args.show_processing)
    if len(neural) < 2:
        continue
    if info['n_units'] < 10:
        continue
```

iii. The ten-unit cutoff is justified from the paper. The notes say all 25 raw Ephys sessions are the intended task subset, although their recorded final counts are internally inconsistent (17 versus 25 retained sessions).

## 1-d. How are the data split into trials?

i. `bp.Ntrials` determines the trial count. Spike `trial` labels are 1-based and populate a separate neuron-by-time matrix for each trial. Behavioral arrays, trajectory entries, and motion-energy entries are indexed by the same zero-based Python trial index.

ii.
```python
n_trials = int(np.asarray(bp['Ntrials']).item())
trial_mats = [np.zeros((n_units, n_bins), dtype=np.float32) for _ in range(n_trials)]
...
trial_mats[tr_ix - 1][ui] = counts.astype(np.float32)
...
for tr in range(n_trials):
```

iii. The mapping plan says raw per-trial arrays and validity fields should be used directly. The code assumes their ordering corresponds across modalities.

## 1-e. How are trials filtered based on quality controls?

i. Trials are retained when `haveEphys` is true and context is known. Context is unknown for photostimulation or early-lick trials, so those are implicitly removed. Ignore trials remain. Video is not required. At the session level, fewer than two retained trials or fewer than ten units causes exclusion.

ii.
```python
dr = (~stim_enable) & (~autowater) & (~early)
wc = (~stim_enable) & autowater & (~early)
...
keep_trials = [i for i in range(n_trials) if have_ephys[i] and context[i] >= 0]
```

iii. The AI cites reference condition logic for excluding stimulation and early trials, uses `haveEphys` for validity, and deliberately keeps ignore trials because ignore is a requested output class. It cites the paper for the ten-unit session threshold.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from `obj.clu`: each parsed unit contributes `trial`, `trialtm`, `tm`, `quality`, and `site`, although only `trial` and `trialtm` are used for binning and `quality` for filtering. No go-cue variable is used in the neural calculation.

ii.
```python
units.append({
    'quality': ...,
    'site': ...,
    'tm': np.asarray(group.get('tm', [])[i]).astype(np.float32),
    'trial': np.asarray(group.get('trial', [])[i]).astype(np.int32),
    'trialtm': np.asarray(group.get('trialtm', [])[i]).astype(np.float32),
})
```

iii. The mapping notes correctly identify `obj['clu']` plus event metadata as the intended source, but state that exact parsing still needed confirmation. The final code never incorporates that event metadata.

## 2-b. How is the `neural` data processed?

i. For each unit and each represented trial, raw `trialtm` values are histogrammed into 10 ms bins from -2.5 to +2.5 seconds. The stored values are spike counts as `float32`; there is no conversion to firing rate, Gaussian smoothing, baseline correction, or normalization.

ii.
```python
for ui, u in enumerate(units):
    ...
    for tr_ix in np.unique(tr):
        x = tt[tr == tr_ix]
        counts, _ = np.histogram(x, bins=t_edges)
        trial_mats[tr_ix - 1][ui] = counts.astype(np.float32)
```

iii. The notes describe this as reproducing or approximating reference preprocessing from raw clusters. They report that raw files lack the precomputed `trialdat`, but give no justification for omitting the reference smoothing and rate conversion.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units with quality labels `garbage` or `noisy` are removed after trimming and lowercasing the string; missing and all other labels are retained. Sessions with fewer than ten such units are removed. There is no >1 Hz mean-rate filter, and labels such as `gabrga`, `real?`, and `poor` are retained.

ii.
```python
def quality_ok(q):
    ...
    return qs not in {'garbage', 'noisy'}
...
units = [u for u in flatten_units(obj.get('clu', [])) if quality_ok(u.get('quality'))]
```

iii. The notes say stripping padded labels fixed an earlier undercount and claim exclusion of only garbage/noisy matches the “reference intent.” They explicitly recognize that 2,367 units exceeds the paper's 1,651 and that paper-side curation remains unmatched, despite separately documenting the paper's >1 Hz rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The code does not align neural activity to the go cue. It bins `trialtm` directly against a nominal -2.5 to +2.5 grid and never reads or subtracts `bp.ev.goCue`.

ii.
```python
tt = u['trialtm']
...
x = tt[tr == tr_ix]
counts, _ = np.histogram(x, bins=t_edges)
```

iii. Comments and documentation assert that this is “fixed binning around go cue using trial-relative spike times” and that go cue is canonical, apparently conflating time relative to trial start with time relative to go cue. No sanity check verifies the subtraction.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output uses 500 non-overlapping 10 ms bins spanning -2.5 to +2.5 seconds. Continuous spike times and video-derived signals are rebinned to this grid; motion energy is also resampled after an artificial time axis is created.

ii.
```python
bin_size = 0.01
t_edges = np.arange(-2.5, 2.5001, bin_size)
...
'time_bin_size': 10.0,
```

iii. The notes claim `params.dt=1/100` and describe the 10 ms bins as matching reference code. The human reference instead finds and uses the authors' 5 ms setting (`1/200`).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not derived from a raw field. It is constructed as the centers of the fixed nominal bin edges and repeated for every trial.

ii.
```python
t_centers = (t_edges[:-1] + t_edges[1:]) / 2
inputs = [t_centers[None, :].astype(np.float32) for _ in keep_trials]
```

iii. The notes identify `obj.bp.ev.goCue` as the required source/alignment event, but the implementation treats the predefined grid itself as time from go cue.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The script averages adjacent edges to make 500 bin centers, from -2.495 to +2.495 seconds, casts them to `float32`, adds a leading input dimension, and duplicates the array per retained trial.

ii.
```python
t_centers = (t_edges[:-1] + t_edges[1:]) / 2
inputs = [t_centers[None, :].astype(np.float32) for _ in keep_trials]
```

iii. Documentation says the fixed range represents time from the go cue; no additional processing rationale is given.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It has the same 500 columns as the neural matrices and uses centers corresponding to their histogram edges. Structurally the arrays align by column, but semantically they do not: neural spikes were never shifted from trial-start time to go-cue time.

ii.
```python
counts, _ = np.histogram(x, bins=t_edges)
...
inputs = [t_centers[None, :].astype(np.float32) for _ in keep_trials]
```

iii. The AI asserts that all streams are go-cue aligned because they use the same grid. That justification overlooks the missing event-time subtraction.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses `bp.L`, `bp.R`, and `bp.no` directly.

ii.
```python
L = np.asarray(bp.get('L')).astype(bool)
R = np.asarray(bp.get('R')).astype(bool)
no = np.asarray(bp.get('no')).astype(bool)
```

iii. The mapping notes describe these as trial condition fields but flag the unresolved question of instructed versus chosen side. The code ultimately assumes they directly encode lick direction.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The code initializes every trial as none (2), assigns left (0) where `L`, right (1) where `R`, then restores none for neither-side or `no` trials. It broadcasts the per-trial code over all time bins. It does not invert the instructed side on miss trials to recover the actual lick.

ii.
```python
out = np.full(L.shape[0], 2, dtype=np.int64)
out[L] = 0
out[R] = 1
out[~(L | R) | no] = 2
```

iii. The notes admit the need for a final chosen-versus-instructed-side decision but do not document one. This implementation effectively selects instructed direction even for incorrect responses.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context uses `bp.stim.enable`, `bp.autowater`, and `bp.early`.

ii.
```python
stim_enable = np.asarray(bp.get('stim', {}).get('enable', np.zeros(ntr))).astype(bool)
autowater = np.asarray(bp.get('autowater', np.zeros(ntr))).astype(bool)
early = np.asarray(bp.get('early', np.zeros(ntr))).astype(bool)
```

iii. The notes initially considered protocol fields, then say sample validation fixed context mapping to the reference condition logic using stimulation, autowater, and early flags.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Nonstimulated, non-early autowater trials are WC (0); nonstimulated, non-early non-autowater trials are DR (1); all others remain -1 and are filtered. The retained per-trial class is broadcast through time.

ii.
```python
dr = (~stim_enable) & (~autowater) & (~early)
wc = (~stim_enable) & autowater & (~early)
out[wc] = 0
out[dr] = 1
```

iii. The code comment and notes explicitly attribute this condition logic to the reference code.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It uses `bp.hit`, `bp.miss`, `bp.no`, and `bp.early`.

ii.
```python
hit = np.asarray(bp.get('hit')).astype(bool)
miss = np.asarray(bp.get('miss')).astype(bool)
no = np.asarray(bp.get('no')).astype(bool)
early = np.asarray(bp.get('early', np.zeros_like(hit))).astype(bool)
```

iii. The notes map these raw behavior flags directly to the requested categorical outcome and discuss whether early trials should be excluded or encoded as ignore; filtering elsewhere excludes them.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The default is ignore (2), hits become correct (1), misses become incorrect (0), and `no` or early trials become ignore. The value is broadcast across time. Early trials do not survive context filtering.

ii.
```python
out = np.full(hit.shape[0], 2, dtype=np.int64)
out[hit] = 1
out[miss] = 0
out[no | early] = 2
```

iii. This follows the requested labels. The notes justify retaining genuine ignore trials because the prompt explicitly requests an ignore output class.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses only `obj.traj[1]`: per-trial `ts`, `frameTimes`, and `featNames` for every feature whose name contains “tongue.” It also uses tracking confidence from the third `ts` coordinate. No go-cue or clock-sync field is used.

ii.
```python
cam_for_kin = obj['traj'][1] if isinstance(obj.get('traj'), list) and len(obj.get('traj')) > 1 else None
...
extract_speed_from_traj_cam(cam_for_kin, tr, ['tongue'])
```

iii. The notes identify `traj` and `findDLCFeatIndex` as the intended sources, but acknowledge that tongue visibility remains almost entirely missing and may reflect incorrect extraction.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Matching feature coordinates are averaged, successive x/y differences are divided by successive frame-time differences, and Euclidean speed is placed on the later frame. Frames need finite coordinates, positive time difference, and any feature confidence above 0.5. Speeds are averaged within each 10 ms bin. There is no reference 5 ms Gaussian smoothing, no contiguous-run handling, no two-camera normalization, and no two-view combination.

ii.
```python
mean_xy = np.nanmean(xy, axis=2)
dxy = np.diff(mean_xy, axis=0)
dt = np.diff(ft)
sp[good] = np.sqrt((dxy[good]**2).sum(axis=1)) / dt[good]
...
out[i] = np.nanmean(values[m])
```

iii. The notes call this a velocity derived from tracking and use missing-data states, but flag the nearly all-not-visible tongue output as unresolved. No rationale is offered for the 0.5 threshold or omitted smoothing/two-view method.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Each trial's already-binned trace is passed independently to `discretize_trace_per_session`; its median over visible bins is used for that trial alone. Visible values below the trial median are 0, values at/above are 1, and missing bins are 2. Despite the function name, no session-wide threshold is computed.

ii.
```python
thr = np.nanmedian(arr[vis])
out[vis] = (arr[vis] >= thr).astype(np.int64)
...
tongue_tv.append(discretize_trace_per_session(...))
```

iii. The AI says the required rule is a per-session 50th percentile, but mistakenly treats each trial call as implementing that rule.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. It is binned directly from raw `frameTimes` using the same nominal edges as neural data. The code neither subtracts the session video-to-behavior clock offset nor the trial's go-cue time. Matching column count therefore does not establish temporal alignment.

ii.
```python
ft = np.asarray(cam['frameTimes'][trial_idx]).astype(np.float32)
...
bin_timeseries_to_edges(tt, tongue_speed, t_edges)
```

iii. The notes say all modalities should align to `bp.ev.goCue` and list spot checks as planned, but the checks remain unchecked and the needed transform was not implemented.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Like tongue velocity, it uses `obj.traj[1]` fields `ts`, `frameTimes`, and `featNames`, selecting all names containing “paw” and reading their confidences.

ii.
```python
pt, paw_speed, paw_vis = extract_speed_from_traj_cam(cam_for_kin, tr, ['paw'])
```

iii. The mapping notes identify `obj.traj` as the intended source. They do not justify averaging all matching paws rather than choosing the reliably tracked `top_paw` used by the human reference.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. All matched paw positions are averaged; finite successive differences are divided by frame intervals to produce Euclidean speed, using confidence >0.5 for visibility, then speeds are averaged into 10 ms bins. There is no Gaussian position smoothing or contiguous-valid-run logic.

ii.
```python
xy = ts[:, :2, :][:, :, feat_idx]
mean_xy = np.nanmean(xy, axis=2)
...
sp[good] = np.sqrt((dxy[good]**2).sum(axis=1)) / dt[good]
```

iii. The notes describe direct velocity derivation and report nondegenerate validation statistics, but do not reconcile this method with the paper's video preprocessing.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Each trial is thresholded independently at its visible-bin median, with 0 below, 1 at/above, and 2 where unavailable. This is not the specified per-session median.

ii.
```python
paw_tv.append(discretize_trace_per_session(
    bin_timeseries_to_edges(pt, paw_speed, t_edges),
    bin_timeseries_to_edges(pt, paw_vis.astype(float), t_edges) > 0))
```

iii. The AI states that thresholds are per-session, but the placement of the call inside the trial loop contradicts that intent.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Raw camera frame times are histogrammed on the nominal neural edges without a video clock correction or go-cue subtraction, so paw velocity is not actually aligned to neural activity.

ii.
```python
ft = np.asarray(cam['frameTimes'][trial_idx]).astype(np.float32)
...
bin_timeseries_to_edges(pt, paw_speed, t_edges)
```

iii. The documentation claims go-cue alignment globally but contains no implemented or validated paw-specific alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from the session's standalone `motionEnergy_<subject>_<date>.mat` variable `me`, then the `.data` attribute is used if present. The in-object `obj.me` copy and actual camera frame times are not used.

ii.
```python
return sio.loadmat(str(mefile), squeeze_me=True, struct_as_record=False).get('me')
...
me_data = getattr(me_struct, 'data', None) if me_struct is not None else None
```

iii. The notes say external files were adopted after sample testing and yielded balanced results. They recognize motion energy as one value per video frame.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. A trial trace is coerced to a 1-D float array. The script invents evenly spaced times across the entire -2.5 to +2.5 second window, averages values into 10 ms bins, and discretizes. Invalid/missing traces become all class 2. It does not use recorded per-frame timing.

ii.
```python
mt = np.linspace(t_edges[0], t_edges[-1], num=md.shape[0], dtype=np.float32)
mb = bin_timeseries_to_edges(mt, md, t_edges)
```

iii. The notes emphasize that external motion energy is nondegenerate, but do not justify stretching every trace uniformly over five seconds.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Each trial is independently split at the median of its finite binned values. Below-median bins are 0, at/above bins are 1, and absent bins/trials are 2. It is not a per-session threshold.

ii.
```python
me_tv.append(discretize_trace_per_session(mb, np.isfinite(mb)))
```

iii. The AI interprets near-50/50 sample distributions as a sanity check, but those distributions are guaranteed within each sufficiently populated trial by the per-trial median and therefore do not validate the required session threshold.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Alignment is artificial: each trace is linearly assigned times from -2.5 to +2.5 seconds regardless of its real side-camera frame times, go cue, or video clock offset. It only shares output length with neural data.

ii.
```python
mt = np.linspace(t_edges[0], t_edges[-1], num=md.shape[0], dtype=np.float32)
```

iii. Notes claim all streams are go-cue aligned but do not discuss this synthetic time assignment or validate it.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing quality labels are accepted. Missing validity arrays receive all-true ephys/all-false video defaults. Missing cameras, malformed features, missing motion files/traces, invalid dimensions, and exceptions during motion conversion yield class-2 output arrays. Invalid spike entries are dropped. Trials without ephys or valid context are dropped. No interpolation is used, but several broad fallbacks silently mask malformed structures.

ii.
```python
have_ephys = np.asarray(...get('haveEphys', np.ones(n_trials))).astype(bool)
...
return ft, np.full(ft.shape, np.nan, dtype=np.float32), np.zeros(ft.shape, dtype=bool)
...
me_tv.append(np.full(len(t_centers), 2, dtype=np.int64))
```

iii. The mapping plan calls for explicit missing-data states, and notes describe crash fixes that convert nonnumeric motion entries to missing. This is reasonable in principle, but the code often cannot distinguish true absence from parsing/alignment failures.

## 11-a. What are the most time-consuming steps of the code?

i. The likely dominant work is loading large MAT files, nested unit/trial spike histogram loops, and especially video binning: for every trial and feature it scans frames separately for each of 500 bins. Optional plotting and pickle serialization add smaller costs. The code records elapsed time per session but contains no profiler.

ii.
```python
for ui, u in enumerate(units):
    for tr_ix in np.unique(tr):
...
for i in range(len(t_edges)-1):
    m = (times >= t_edges[i]) & (times < t_edges[i+1]) ...
```

iii. Notes report roughly 4.5–6.6 seconds per sample session and call the full subset manageable, but leave the “Code inefficiencies” and “Speed-ups” fields blank.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The unit-by-trial spike histograms could be combined with indexed accumulation or 2-D histograms. `bin_timeseries_to_edges` could compute bin indices once and use `bincount`/grouped reduction instead of scanning all frames 500 times. Feature extraction and per-trial discretization are ragged but could still reduce repeated array conversion.

ii.
```python
for tr_ix in np.unique(tr):
    counts, _ = np.histogram(x, bins=t_edges)
...
for i in range(len(t_edges)-1):
    m = (times >= t_edges[i]) & (times < t_edges[i+1])
```

iii. The AI does not explicitly identify vectorization opportunities; its notes leave the relevant efficiency sections empty.

## 11-c. What processing does the code repeat multiple times?

i. Inside every trial it repeatedly unwraps the same session-wide motion-energy structure and converts it to a list. Each tongue/paw trial bins speed and visibility in separate 500-bin scans. The same fixed time input is copied for every trial. Unit spikes are repeatedly masked once per unique trial.

ii.
```python
for tr in range(n_trials):
    me_data = getattr(me_struct, 'data', None) ...
    me_seq = list(me_data)
...
bin_timeseries_to_edges(tt, tongue_speed, t_edges)
bin_timeseries_to_edges(tt, tongue_vis.astype(float), t_edges)
```

iii. The AI provides no explicit justification; its repeated-processing documentation is blank.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `flatten_units` converts and stores each unit's absolute `tm` and `site`, but neither is used. `have_vid` is loaded but not used by the implemented video path. Context labels are built only for metadata samples. The unused `placeholder_timevarying_outputs` function and `defaultdict` import do no runtime work. When processing sessions that are later rejected for fewer than ten units, the script nevertheless computes all trial neural and video outputs first. Optional plots also operate only for inspection.

ii.
```python
'site': group.get('site', ...),
'tm': np.asarray(group.get('tm', [])[i]).astype(np.float32),
...
have_vid = np.asarray(...).astype(bool)
...
if info['n_units'] < 10:
    continue
```

iii. The notes do not document these discarded computations. They focus instead on successful validation and runtime being manageable.
