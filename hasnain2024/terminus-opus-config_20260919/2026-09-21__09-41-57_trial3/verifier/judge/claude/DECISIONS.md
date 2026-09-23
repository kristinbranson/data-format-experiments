# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI parses the reference MATLAB meta scripts (`load<ANM>_ALMVideo.m`) programmatically via `session_meta.py` to get the authoritative session list and probe assignments. Each session's `data_structure_<anm>_<date>.mat` file is loaded by `matio.load_obj()`, which handles both MATLAB v7.3 (HDF5) and v5 formats. Motion energy is loaded separately from `motionEnergy_<anm>_<date>.mat` files. The AI wrote a custom uniform loader (`matio.py`) that normalizes both formats into the same nested Python dict structure. Sessions are processed in parallel using `multiprocessing.Pool`.

ii. Session list construction (from `session_meta.py`):
```python
def get_sessions():
    out = []
    for s in parse_meta_scripts():
        f = find_data_file(s['anm'], s['date'])
        if f is None:
            continue
        s = dict(s)
        s['datafile'] = f
        s['mefile'] = find_me_file(s['anm'], s['date'])
        s['dataset'] = ('randomized_delay' if 'RandomizedDelay' in f else 'fixed_delay')
        s['sessid'] = f"{s['anm']}_{s['date']}"
        out.append(s)
    return out
```

Loading one session (from `matio.py`):
```python
def load_obj(fn, field='obj'):
    if is_v73(fn):
        return load_v73(fn, field)
    return load_v5(fn, field)
```

iii. The AI justified this by noting that the authors' `load<ANM>_ALMVideo.m` files are the definitive record of which sessions entered their analysis, and that globbing data directories would include sessions the authors excluded (3 extra files not in meta). Both MATLAB file formats appear in the data, necessitating both readers.

## 1-b. How are the data split into subjects?

i. The animal ID is extracted from the session ID (the part before the underscore in the `sessid` field, e.g. `JEB13` from `JEB13_2022-09-13`). Subjects are accumulated as sessions are processed, and `subject_idx` maps each session to its subject.

ii. From `convert_data.py`:
```python
anm = sess['anm']
if anm not in data['subjects']:
    data['subjects'].append(anm)
data['subject_idx'].append(data['subjects'].index(anm))
```

Where `sess['anm']` comes from `session_meta.py` parsing `meta.anm` from the MATLAB scripts.

iii. The AI noted that the animal ID is parsed from the reference meta scripts, which define `meta.anm` for each session.

## 1-c. How are the data split into sessions?

i. One session corresponds to one entry in the parsed meta list (one `.mat` file). The AI identifies sessions from both task folders (`Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior`). 44 sessions total (25 fixed-delay, 19 randomized-delay) are found with data present. Sessions with fewer than 10 curated units are skipped (though all 44 pass).

ii. From `convert_data.py`:
```python
sessions = get_sessions()
...
if info['nunits'] < PARAMS['min_units']:
    nskipped.append((sess['sessid'], f"only {info['nunits']} units"))
    continue
```

iii. The AI cited the paper: "Recording sessions were included for analysis only if they had at least 10 units."

## 1-d. How are the data split into trials?

i. Trials are defined by the Bpod trial table (`obj.bp`). The number of trials is `bp.Ntrials`. All per-trial fields are truncated to this length. Trial indices are 1-based (matching MATLAB convention). Spike times carry their trial index in `clu.trial`, so no trial boundary reconstruction is needed.

ii. From `convert_data.py`:
```python
N = int(as_vector(bp['Ntrials'])[0])
hit = as_vector(bp['hit'])[:N].astype(bool)
miss = as_vector(bp['miss'])[:N].astype(bool)
...
trials = np.where(keep)[0] + 1  # 1-based trial numbers
```

iii. The AI noted that trial indices are 1-based as in the MATLAB data structure.

## 1-e. How are trials filtered based on quality controls?

i. Three filters are applied: (1) Early-lick trials (`bp.early`) are excluded. (2) Optogenetic stimulation trials (`bp.stim.enable`) are excluded. (3) Additionally, the AI requires `np.isfinite(gocue) & (hit | miss | no)` and drops trials with zero spikes from all units (recording ended early). The result is 13,762 trials from 14,972 raw.

ii. From `convert_data.py`:
```python
keep = (~early) & (~stim) & np.isfinite(gocue) & (hit | miss | no)
trials = np.where(keep)[0] + 1
...
recorded = spk_per_trial > 0
if n_not_recorded:
    trials = trials[recorded]
```

iii. The AI justified early-lick and stim exclusion by citing the paper: "omitted from all analyses." The `hit | miss | no` filter ensures only valid outcome trials are included. The zero-spike filter handles sessions where ephys recording stopped before the behavioural session ended.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` — the spike-sorted clusters. Each cluster has `trial` (1-based trial index), `trialtm` (spike time relative to trial start), and `quality` (manual curation label). The go cue times `bp.ev.goCue` provide alignment. Only the probe(s) specified in the reference meta scripts are used.

ii. From `convert_data.py`:
```python
clusters = get_clusters(obj, p)
...
tr = np.asarray(c['trial'], dtype=np.int64).ravel()
tm = np.asarray(c['trialtm'], dtype=float).ravel()
...
aligned = tm[sel] - align_times[pos]
```

iii. The AI documented this as matching `alignSpikes.m` + `getSeq.m`.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue (`trialtm - goCue[trial]`), then binned into 10ms bins spanning -2.5 to +2.5 s (500 bins). Counts are divided by dt to get spikes/s, then smoothed with a **causal Gaussian kernel** (gausswin(15) with the first half zeroed, reflect boundary) — a direct port of `mySmooth.m`. This is a key difference from the reference which uses a symmetric Gaussian with 14ms sigma.

ii. From `convert_data.py`:
```python
PARAMS = dict(
    dt=0.01,             # 10 ms bins
    smooth=15,           # causal gaussian window, in bins
    ...
)

def causal_gauss_kernel(N=PARAMS['smooth']):
    k = _gausswin(N)
    k[:N // 2] = 0.0
    return k / k.sum()

def my_smooth(x, N=PARAMS['smooth'], bctype=PARAMS['bctype']):
    ...
    k = causal_gauss_kernel(N)
    ...
    for j in range(xf.shape[1]):
        out[:, j] = np.convolve(xf[:, j], k, mode='same')
```

iii. The AI justified using causal smoothing as "identical to the reference pipeline" citing `mySmooth.m`, and 10ms bins as matching `WorkingWithDataObjs.m` part 2.1.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) Clusters whose quality label (case-insensitive) is in `{garbage, gabrga, noisy, real?, ''}` are dropped. Note that empty string is included, and `poor` is NOT in the drop list. (2) Units with mean firing rate <= 1 Hz are dropped.

ii. From `convert_data.py`:
```python
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?', ''}
...
keep = [i for i, c in enumerate(clusters)
        if str(c.get('quality', '')).strip().lower() not in BAD_QUALITY]
...
meanfr = rates.mean(axis=(0, 1))
keep_u = meanfr > PARAMS['lowFR']
```

iii. The AI cited `findClusters.m` for the quality filter and `removeLowFRClusters.m` / methods text ("firing rates exceeding 1 Hz") for the FR threshold.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to the go cue: spike times within a trial (`clu.trialtm`) minus the go cue time for that trial (`bp.ev.goCue`). The aligned times are then binned.

ii. From `convert_data.py`:
```python
aligned = tm[sel] - align_times[pos]
b = np.floor((aligned - edges[0]) / PARAMS['dt']).astype(np.int64)
```

iii. The AI described this as replicating `alignSpikes.m`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 10 ms bins, 500 bins total spanning -2.5 to +2.5 s from the go cue. No rebinning — spikes are directly binned into this grid. The time axis goes from -2.495 to +2.495 s (bin centres).

ii. From `convert_data.py`:
```python
PARAMS = dict(
    tmin=-2.5,
    tmax=2.5,
    dt=0.01,             # 10 ms bins
    ...
)

def time_axis():
    edges = np.arange(PARAMS['tmin'], PARAMS['tmax'] + 1e-12, PARAMS['dt'])
    tm = edges[:-1] + PARAMS['dt'] / 2
    return edges, tm
```

iii. The AI noted this matches the tutorial parameters in `WorkingWithDataObjs.m` part 2.1 (`params.dt = 1/100`).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Not derived from any raw data variable. It is the bin-centre time axis defined by the binning parameters, representing the time from the go cue onset in seconds.

ii. From `convert_data.py`:
```python
_, tm = time_axis()
tin = taxis.astype(np.float32)[None, :]
...
inp.append(tin.copy())
```

iii. The AI stated this is equivalent to `obj.time` in the reference code.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is computed as bin centres: `edges + dt/2`, where edges go from -2.5 to 2.5 in steps of 0.01. No additional processing.

ii. Same as 3-a.

iii. N/A.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input IS the neural binning grid (bin centres). Spikes are counted into the same bin edges, so the input and neural data share the same time axis by construction.

ii. Same as 3-a — `taxis` is used both for spike binning edges and as the input.

iii. N/A.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial fields: `bp.R` (right instruction), `bp.L` (left instruction), `bp.hit` (correct response), `bp.miss` (incorrect response), and `bp.no` (ignore/no response).

ii. From `convert_data.py`:
```python
tr0 = trials - 1
right = (R[tr0] & hit[tr0]) | (L[tr0] & miss[tr0])
lick = np.where(no[tr0], 2, np.where(right, 1, 0))  # 0 left, 1 right, 2 none
```

iii. The AI verified that lick direction matches the first post-go-cue lick port on 99-100% of trials.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Lick direction is inferred: right = (instructed right AND hit) OR (instructed left AND miss); left = everything else that isn't an ignore; none = ignore trials (`bp.no`). Codes: left=0, right=1, none=2.

ii. Same code as 4-a.

iii. The AI noted that lick direction is not recorded directly but derived from instructed side and outcome.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial field: `bp.autowater`. Autowater trials are WC (water-cued) context; non-autowater are DR (delayed-response) context.

ii. From `convert_data.py`:
```python
aw = as_vector(bp['autowater'])[:N].astype(bool)
...
context = np.where(aw[tr0], 0, 1)  # 0 WC, 1 DR
```

iii. The AI documented this as matching the condition strings in the reference code.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct relabelling: autowater=True -> WC (0), autowater=False -> DR (1).

ii. Same as 5-a.

iii. Straightforward mapping.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Three per-trial fields: `bp.hit`, `bp.miss`, `bp.no`. These are mutually exclusive flags.

ii. From `convert_data.py`:
```python
outcome = np.where(no[tr0], 2, np.where(hit[tr0], 1, 0))  # 0 incorrect, 1 correct, 2 ignore
```

iii. The AI cited `getOutcome.m` from the reference code.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct mapping: miss -> incorrect (0), hit -> correct (1), no -> ignore (2).

ii. Same as 6-a.

iii. The AI noted that ignore trials are kept (unlike the paper's behavioural analyses) because the decoder task requires the ignore category.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DLC tracking in `obj.traj{1}` (side camera), specifically the `tongue` feature. The fields used are `ts` (tracked x, y, likelihood per frame per feature), `frameTimes`, and `featNames`. Also `sglx.bitcode.bitstart` and `bp.ev.bitStart` for the video-ephys clock offset.

ii. From `convert_data.py`:
```python
tongue_ix = side_feats.index('tongue')
Xt, Yt = interp_positions(side, tongue_ix, trials, align_times, taxis, vidshift)
tongue_speed = speed_from_positions(Xt, Yt)
```

iii. The AI cited `findPosition.m` and `findVelocity.m` from the reference code. The tongue feature is tracked on the side camera.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Steps: (1) Interpolate x, y positions from the side camera onto the neural time axis after correcting the video-ephys clock offset. NaN positions (tongue not visible, DLC likelihood low) are propagated. (2) Compute speed as `|gradient(x, y)|` using a NaN-aware central/one-sided difference. (3) Discretize at the per-session 50th percentile of visible values into two classes, with a third class (2) for not-visible timepoints.

ii. From `convert_data.py`:
```python
def interp_positions(traj_trials, featix, trials, align_times, taxis, vidshift):
    ...
    for k, arr in ((0, X), (1, Y)):
        v = ts[:, k, featix]
        arr[i] = np.interp(taxis, tt, v, left=np.nan, right=np.nan)
        nanmask = np.interp(taxis, tt, np.isnan(v).astype(float), left=1.0, right=1.0) > 0
        arr[i][nanmask] = np.nan

def speed_from_positions(X, Y):
    vx = _gradient_nan(X)
    vy = _gradient_nan(Y)
    return np.sqrt(vx ** 2 + vy ** 2)

tongue_cls, tongue_thr = discretize(tongue_speed, tongue_visible)
```

iii. The AI noted this follows `findPosition.m` (interpolation of DLC positions) and `findVelocity.m` (gradient of position). Key difference from reference: no Gaussian smoothing of positions before differentiation, and only the side camera is used (reference uses both side and bottom cameras).

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per-session 50th percentile of visible (non-NaN) tongue speed values. 0 = below median, 1 = at or above median, 2 = not visible.

ii. From `convert_data.py`:
```python
def discretize(values, visible):
    out = np.full(values.shape, 2, dtype=np.int64)
    vis = visible & np.isfinite(values)
    if np.any(vis):
        thr = np.percentile(values[vis], 50)
        out[vis] = (values[vis] >= thr).astype(np.int64)
    return out, thr
```

iii. Matches the decoder task specification.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Camera frame times are corrected by the video-ephys offset (`vidshift = mode(bitstart/fs) - mode(bp.ev.bitStart)`) and aligned to the go cue. Tongue positions are then interpolated directly onto the neural time axis using `np.interp`, so the output shares the same time bins as the neural data.

ii. From `convert_data.py`:
```python
vidshift = get_vidshift(obj)
...
tt = ft - vidshift - align_times[i]
...
arr[i] = np.interp(taxis, tt, v, left=np.nan, right=np.nan)
```

iii. The AI cited `findVideoOffset.m` for the clock correction and `findPosition.m` for the interpolation onto the time axis.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. DLC tracking in `obj.traj{2}` (bottom camera), specifically both `top_paw` and `bottom_paw` features.

ii. From `convert_data.py`:
```python
paw_speeds = []
paw_vis = []
for f in ('top_paw', 'bottom_paw'):
    ix = bot_feats.index(f)
    Xp, Yp = interp_positions(bottom, ix, trials, align_times, taxis, vidshift)
    paw_speeds.append(speed_from_positions(Xp, Yp))
    paw_vis.append(np.isfinite(Xp) & np.isfinite(Yp))
```

iii. The AI uses both tracked paws from the bottom camera.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same as tongue: positions interpolated onto the neural time axis, speed computed as magnitude of gradient, then averaged across the two paw features. Discretized at the per-session median. Not-visible class for timepoints where neither paw is tracked.

ii. From `convert_data.py`:
```python
paw_vis = np.any(np.stack(paw_vis), axis=0)
with np.errstate(invalid='ignore'):
    paw_speed = np.nanmean(np.stack(paw_speeds), axis=0)
paw_speed = np.where(paw_vis & ~np.isfinite(paw_speed), 0.0, paw_speed)
...
paw_cls, paw_thr = discretize(paw_speed, paw_vis)
```

iii. The AI averages both paws' speeds to get a combined paw velocity measure.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: per-session 50th percentile of visible values. 0 = below, 1 = at/above, 2 = not visible.

ii. Same `discretize()` function as for tongue.

iii. Matches decoder task specification.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identical to tongue: video offset correction, go-cue alignment, interpolation onto the neural time axis.

ii. Same `interp_positions` function, using the bottom camera's frame times.

iii. Same as tongue alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Separate `motionEnergy_<anm>_<date>.mat` files, loaded via `matio.load_motion_energy()`. These contain per-trial 400 Hz motion energy traces. Frame times come from the side camera's `frameTimes` in `obj.traj{1}`.

ii. From `convert_data.py`:
```python
me_trials = None
if sess.get('mefile'):
    try:
        me_trials, _thr = load_motion_energy(sess['mefile'])
    except Exception:
        me_trials = None
ME = motion_energy_on_axis(me_trials, side, trials, align_times, taxis, vidshift)
```

iii. The AI handled the three different layouts of motion energy files found in the dataset.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The 400 Hz motion energy trace is interpolated onto the neural time axis using `np.interp` after applying the video-ephys clock correction. Then discretized at the per-session median.

ii. From `convert_data.py`:
```python
def motion_energy_on_axis(me_trials, traj_trials, trials, align_times, taxis, vidshift):
    ...
    ME[i] = np.interp(taxis, tt, d, left=np.nan, right=np.nan)
    ...
me_cls, me_thr = discretize(ME, me_visible)
```

iii. The AI noted this follows `loadMotionEnergy.m`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session 50th percentile of visible (non-NaN) values. 0 = below, 1 = at/above, 2 = no video.

ii. Same `discretize()` function.

iii. Matches decoder task specification.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Frame times from the side camera are corrected by the video-ephys offset and aligned to the go cue, then the motion energy trace is interpolated onto the neural time axis.

ii. From `motion_energy_on_axis`:
```python
ft = np.asarray(t.get('frameTimes', np.nan), dtype=float).ravel()
if ft.size != d.size or not np.any(np.isfinite(ft)):
    ft = np.arange(1, d.size + 1) / VIDEO_FS
    tt = ft - DEFAULT_VIDSHIFT - align_times[i]
else:
    tt = ft - vidshift - align_times[i]
ME[i] = np.interp(taxis, tt, d, left=np.nan, right=np.nan)
```

iii. Same offset/alignment as all other camera streams. Includes a fallback for missing frame times.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases: (1) Trials where the ephys recording stopped before the behavioural session ended (no spikes from any unit) are dropped. (2) Missing `frameTimes` trigger a fallback: `ft = (1:nframes)/400`, shift 0.5 s. (3) NaN positions from DLC (tongue not visible) are propagated through interpolation via a companion NaN-mask interpolation. (4) Isolated visible frames where gradient cannot be computed get speed set to 0 (following `findVelocity.m`). (5) Sessions missing motion energy files get all-NaN ME (labelled "no video").

ii. From `convert_data.py`:
```python
# Recording stopped early
recorded = spk_per_trial > 0
if n_not_recorded:
    trials = trials[recorded]

# Missing frameTimes fallback
if ft.size != ts.shape[0] or not np.any(np.isfinite(ft)):
    ft = (np.arange(1, ts.shape[0] + 1)) / VIDEO_FS
    tt = ft - DEFAULT_VIDSHIFT - align_times[i]

# NaN propagation for interpolation
nanmask = np.interp(taxis, tt, np.isnan(v).astype(float), left=1.0, right=1.0) > 0
arr[i][nanmask] = np.nan
```

iii. The AI documented each edge case in CONVERSION_NOTES.md and verified fixes with sanity checks.

## 11-a. What are the most time-consuming steps of the code?

i. Reading the .mat files dominates runtime. The AI implemented multiprocessing (8 workers by default) to parallelize across sessions. Total conversion runs in ~18 seconds with 16 workers.

ii. From `convert_data.py`:
```python
if nw > 1:
    with Pool(nw) as pool:
        results = pool.map(_worker, jobs)
```

iii. The AI profiled each step and reported timing in CONVERSION_NOTES.md.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-cluster spike binning uses `np.add.at` instead of a per-trial loop, which is a vectorization. However, the per-trial interpolation loops (for video kinematics and motion energy) remain as loops because each trial has a different number of camera frames.

ii. From `convert_data.py`:
```python
np.add.at(counts[:, :, j], (pos[good], b[good]), 1.0)
```

iii. The AI noted these loops cannot be easily vectorized due to variable-length frame counts per trial.

## 11-c. What processing does the code repeat multiple times?

i. The video offset is computed once per session. The time axis is computed once. Loading is done once per session. The smoothing kernel is computed once. No significant repeated processing was identified.

ii. From `convert_data.py`:
```python
vidshift = get_vidshift(obj)  # once per session
```

iii. The AI noted no redundant computations.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads the full `obj` structure (minus `spkWavs` which is explicitly skipped), including fields not used in the conversion (e.g., `channel`, various metadata fields). The `moveThresh` from motion energy files is loaded but not used (median split is used instead). Both `top_paw` and `bottom_paw` are processed and averaged, whereas only one might suffice. The `probe_regions` function reads `obj.ex.probe.loc` to determine brain regions even though this information isn't strictly needed if all units are assumed to be ALM.

ii. From `matio.py`:
```python
SKIP_FIELDS = {'spkWavs', 'spkwavs'}
```

From `convert_data.py`:
```python
me_trials, _thr = load_motion_energy(sess['mefile'])  # _thr is unused
```

iii. The AI optimized loading by skipping spike waveforms but otherwise loads the full object.
