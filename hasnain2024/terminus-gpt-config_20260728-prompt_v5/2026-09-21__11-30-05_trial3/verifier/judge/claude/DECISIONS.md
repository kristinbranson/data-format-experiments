# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by globbing the `Ephys_Behavior` directory for `data_structure_*.mat` files. It only processes sessions from this one directory, missing the `RandomizedDelay_Ephys_Behavior` directory entirely. Files are loaded using `mat73.loadmat()`, which only handles MATLAB v7.3 (HDF5) files. There is no fallback for v5 MATLAB files.

ii. Session discovery:
```python
def discover_sessions(base=Path('/app/data')):
    files = sorted((base / 'Ephys_Behavior').glob('data_structure_*.mat'))
    return files
```

Loading:
```python
obj = mat73.loadmat(str(path))['obj']
```

iii. The AI's CONVERSION_NOTES.md (Step 5) states: "Start from `Ephys_Behavior` because it matches the paper's 25-session DR dataset." The AI acknowledged in Step 9 that it only converts 25 sessions from the Ephys_Behavior subset.

## 1-b. How are the data split into subjects?

i. The subject is extracted from the filename using a regex, or from `obj.meta.anm` if available. The subject name is the part before the first underscore in the filename.

ii.
```python
subject = obj.get('meta', {}).get('anm', re.search(r'data_structure_([^_]+)_', path.name).group(1))
```

iii. The AI uses the filename as a fallback when `meta.anm` is not available. The CONVERSION_NOTES do not specifically justify this approach.

## 1-c. How are the data split into sessions?

i. Each `.mat` file discovered by globbing represents one session. The AI processes all files found in `Ephys_Behavior`. Only 25 sessions are included (the fixed-delay task); the 19 randomized-delay sessions are missing.

ii.
```python
session_files = discover_sessions()
# ...
for sf in session_files:
    neural, inputs, outputs, info = process_session(sf, ...)
```

iii. CONVERSION_NOTES Step 5 states the decision to start from Ephys_Behavior. The AI noted the discrepancy with the paper (25 vs 44 sessions) but did not resolve it.

## 1-d. How are the data split into trials?

i. Trials are identified by `bp.Ntrials`, which gives the total number of trials in each session. Spike times carry trial indices (`u['trial']`), so trial boundaries don't need reconstruction. The per-trial fields (hit, miss, L, R, etc.) each have `Ntrials` entries.

ii.
```python
n_trials = int(np.asarray(bp['Ntrials']).item())
# ...
for tr_ix in np.unique(tr):
    if tr_ix < 1 or tr_ix > n_trials:
        continue
```

iii. The AI correctly uses the trial count from the behavioral data to bound trial indices.

## 1-e. How are trials filtered based on quality controls?

i. Two filters: (1) trials where `haveEphys` is false are dropped, and (2) trials where the context is unknown (neither WC nor DR, which includes early-lick and photostim trials) are dropped. The context assignment function marks early-lick and stim trials as context=-1, and those are filtered out.

ii.
```python
dr = (~stim_enable) & (~autowater) & (~early)
wc = (~stim_enable) & autowater & (~early)
out[wc] = 0
out[dr] = 1
# ...
keep_trials = [i for i in range(n_trials) if have_ephys[i] and context[i] >= 0]
```

iii. The AI's CONVERSION_NOTES Step 3 documents: "early lick and ignore trials omitted from analyses." The filtering of early-lick and stim trials is done indirectly through the context assignment. Additionally, sessions with fewer than 2 valid trials or fewer than 10 units are skipped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `obj.clu`, which contains spike-sorted cluster data. Each cluster has `trial` (trial index), `trialtm` (spike time relative to trial start), and `quality` (curation label). The `go cue` times from `bp.ev.goCue` are NOT used for alignment -- the AI bins spike times directly from `trialtm` without subtracting the go cue time.

ii.
```python
units = [u for u in flatten_units(obj.get('clu', [])) if quality_ok(u.get('quality'))]
# ...
def flatten_units(clu_list):
    # ...
    units.append({
        'quality': group.get('quality', [None] * n)[i],
        'tm': np.asarray(group.get('tm', [])[i]).astype(np.float32),
        'trial': np.asarray(group.get('trial', [])[i]).astype(np.int32),
        'trialtm': np.asarray(group.get('trialtm', [])[i]).astype(np.float32),
    })
```

iii. The AI's CONVERSION_NOTES Step 5 maps `obj['clu']` to the neural target field. However, the code does not subtract go cue times from `trialtm` during binning.

## 2-b. How is the `neural` data processed?

i. Raw spike counts are binned into 10ms time bins from -2.5 to +2.5s (500 bins) using `np.histogram`. The data remains as raw spike counts -- no conversion to firing rates (Hz), no Gaussian smoothing, and no normalization. The bin size is 10ms rather than the reference's 5ms.

ii.
```python
bin_size = 0.01
t_edges = np.arange(-2.5, 2.5001, bin_size)
# ...
counts, _ = np.histogram(x, bins=t_edges)
trial_mats[tr_ix - 1][ui] = counts.astype(np.float32)
```

iii. CONVERSION_NOTES Step 9 states "Time bin: 10 ms, params.dt=1/100." The AI used `params.dt = 1/100` rather than the correct `params.dt = 1/200` from the reference code `getDefaultParams.m`. No smoothing or rate conversion is documented or applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only `garbage` and `noisy` labels are excluded. The reference also excludes `gabrga`, `real?`, and `poor`. There is no minimum firing rate filter (the reference drops units <= 1 Hz mean rate). The AI initially had parsing issues with padded labels but fixed them by stripping whitespace.

ii.
```python
def quality_ok(q):
    if q is None:
        return True
    if isinstance(q, str):
        qs = q.strip().lower()
        return qs not in {'garbage', 'noisy'}
    return True
```

iii. CONVERSION_NOTES Step 10: "excluding only `garbage`/`noisy`, matching reference intent." The AI acknowledged the resulting 2367 neurons exceeds the paper's 1651 units but did not resolve the discrepancy.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does NOT align neural data to the go cue. Spike times (`trialtm`) are binned directly without subtracting `bp.ev.goCue`. The `trialtm` values are relative to trial start, not go cue onset, so the neural data is aligned to trial start rather than go cue.

ii.
```python
def bin_unit_trialtm(units, n_trials, t_edges):
    # ...
    for ui, u in enumerate(units):
        tr = u['trial']
        tt = u['trialtm']
        # Note: no subtraction of go cue time
        for tr_ix in np.unique(tr):
            x = tt[tr == tr_ix]
            counts, _ = np.histogram(x, bins=t_edges)
```

iii. The AI's CONVERSION_NOTES Step 5 mentions "Use `bp.ev.goCue` as the canonical alignment event for all streams" but the code does not implement this subtraction for the neural data.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 10ms bins (500 time points over 5 seconds). The reference uses 5ms bins (1000 time points). No rebinning is applied -- the 10ms bin size is set directly.

ii.
```python
bin_size = 0.01
t_edges = np.arange(-2.5, 2.5001, bin_size)
t_centers = (t_edges[:-1] + t_edges[1:]) / 2
```

iii. CONVERSION_NOTES Step 9 records "Time bin: 10 ms, params.dt=1/100." The reference code uses `params.dt = 1/200 = 5ms`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the time bin centers of the binning grid, spanning -2.5 to +2.5s in 10ms steps.

ii.
```python
t_centers = (t_edges[:-1] + t_edges[1:]) / 2
inputs = [t_centers[None, :].astype(np.float32) for _ in keep_trials]
```

iii. The CONVERSION_NOTES Step 5 maps "Go-cue-relative time vector" to the input field.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time vector is computed from the bin edges. No additional processing is needed since it is a deterministic time axis.

ii.
```python
t_centers = (t_edges[:-1] + t_edges[1:]) / 2
```

iii. N/A -- straightforward computation.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time vector and the neural binning use the same `t_edges`, so they share the same time axis. However, since the neural data is NOT actually aligned to go cue (see 2-d), the input labels the bins as "time from go cue" when the neural data is actually aligned to trial start.

ii.
```python
neural_all = bin_unit_trialtm(units, n_trials, t_edges)
# ...
inputs = [t_centers[None, :].astype(np.float32) for _ in keep_trials]
```

iii. The misalignment between the claimed go-cue alignment (in the time axis labels) and the actual trial-start alignment (in the neural binning) is not documented.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI uses `bp.L`, `bp.R`, and `bp.no` directly. `L` and `R` indicate the *instructed* lick side, not the actual lick direction. The reference derives actual lick direction from the combination of instructed side (R) and outcome (hit/miss).

ii.
```python
def infer_lick_direction(bp):
    L = np.asarray(bp.get('L')).astype(bool)
    R = np.asarray(bp.get('R')).astype(bool)
    no = np.asarray(bp.get('no')).astype(bool)
    out = np.full(L.shape[0], 2, dtype=np.int64)
    out[L] = 0
    out[R] = 1
    out[~(L | R) | no] = 2
    return out
```

iii. The AI's CONVERSION_NOTES Step 5 maps "`obj['bp']['L']`, `obj['bp']['R']`" to the lick direction output. No mention of deriving actual lick direction from hit/miss + instructed side.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Direct mapping: L=True -> left(0), R=True -> right(1), neither or no=True -> none(2). This gives the instructed side, not the actual lick direction. On miss trials, the animal licked the opposite side, so this is incorrect.

ii.
```python
out[L] = 0
out[R] = 1
out[~(L | R) | no] = 2
```

iii. The CONVERSION_NOTES do not discuss the distinction between instructed side and actual lick direction.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Three fields: `bp.stim.enable`, `bp.autowater`, and `bp.early`. WC = autowater & not stim & not early; DR = not autowater & not stim & not early.

ii.
```python
def infer_context_per_trial(bp):
    stim_enable = np.asarray(bp.get('stim', {}).get('enable', np.zeros(ntr))).astype(bool)
    autowater = np.asarray(bp.get('autowater', np.zeros(ntr))).astype(bool)
    early = np.asarray(bp.get('early', np.zeros(ntr))).astype(bool)
    dr = (~stim_enable) & (~autowater) & (~early)
    wc = (~stim_enable) & autowater & (~early)
```

iii. The AI's Step 5 notes: "Reference code: DR = ~stim.enable & ~autowater & ~early; WC = ~stim.enable & autowater & ~early."

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The context is derived from the boolean combination above. WC maps to 0, DR maps to 1. Trials that don't match either condition (early or stim) get -1 and are excluded.

ii.
```python
out[wc] = 0
out[dr] = 1
```

iii. The reference simply uses `autowater` directly: WC if autowater, DR otherwise. The AI's inclusion of early and stim in the context logic is more complex but effectively gives the same result since those trials are filtered out.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Four fields: `bp.hit`, `bp.miss`, `bp.no`, and `bp.early`.

ii.
```python
def infer_outcome(bp):
    hit = np.asarray(bp.get('hit')).astype(bool)
    miss = np.asarray(bp.get('miss')).astype(bool)
    no = np.asarray(bp.get('no')).astype(bool)
    early = np.asarray(bp.get('early', np.zeros_like(hit))).astype(bool)
```

iii. The AI reads the outcome flags directly from the behavioral data.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapping: hit -> correct(1), miss -> incorrect(0), no or early -> ignore(2). The reference uses hit -> correct, miss -> incorrect, and everything else -> ignore (without explicitly reading `bp.no` or `bp.early`, since those trials are already filtered out).

ii.
```python
out[hit] = 1
out[miss] = 0
out[no | early] = 2
```

iii. The explicit mapping of `no` and `early` to ignore is redundant since early trials are already excluded by the context filter, but it's not incorrect.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The AI attempts to extract tongue speed from `obj.traj[1]` (the bottom camera only). It looks for features with "tongue" in the name using keyword matching.

ii.
```python
cam_for_kin = obj['traj'][1] if isinstance(obj.get('traj'), list) and len(obj.get('traj')) > 1 else None
# ...
tt, tongue_speed, tongue_vis = extract_speed_from_traj_cam(cam_for_kin, tr, ['tongue'])
```

iii. The AI only uses one camera (bottom/index 1) for tongue tracking, whereas the reference uses both the side camera ('tongue') and bottom camera ('top_tongue') and combines them.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI extracts x,y coordinates from the tracking data, computes frame-to-frame speed as `sqrt(dx^2 + dy^2) / dt`, and uses a visibility threshold of confidence > 0.5 (vs reference's > 0.9). No Gaussian smoothing is applied to the positions before differentiation. The result is then binned into the time edges and discretized.

ii.
```python
def extract_speed_from_traj_cam(cam, trial_idx, feature_keywords):
    # ...
    visible = np.any(np.isfinite(xy), axis=(1,2)) & np.any(conf > 0.5, axis=1)
    mean_xy = np.nanmean(xy, axis=2)
    dxy = np.diff(mean_xy, axis=0)
    dt = np.diff(ft)
    sp = np.full(dt.shape, np.nan, dtype=np.float32)
    sp[good] = np.sqrt((dxy[good]**2).sum(axis=1)) / dt[good]
```

iii. The result is 99.9% "not visible" across all sessions, indicating the tongue extraction is effectively broken. The AI acknowledged this in CONVERSION_NOTES but did not resolve it.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI calls `discretize_trace_per_session` per-trial rather than per-session, meaning the median threshold is computed per-trial rather than per-session as the instructions require.

ii.
```python
tongue_tv.append(discretize_trace_per_session(
    bin_timeseries_to_edges(tt, tongue_speed, t_edges),
    bin_timeseries_to_edges(tt, tongue_vis.astype(float), t_edges) > 0
))

def discretize_trace_per_session(values, visible_mask):
    # ...
    thr = np.nanmedian(arr[vis])
    out[vis] = (arr[vis] >= thr).astype(np.int64)
```

iii. Despite the function name `discretize_trace_per_session`, it is called once per trial, so the threshold is per-trial not per-session. The instructions specify "per-session threshold."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI uses frame times (`cam['frameTimes']`) directly without correcting for the video-behavior clock offset. There is no call to `findVideoOffset` or equivalent. The binned values use the same `t_edges` as the neural data.

ii.
```python
ft = np.asarray(cam['frameTimes'][trial_idx]).astype(np.float32)
# No offset correction applied
```

iii. The reference computes a video offset using bitcode synchronization (`sglx.bitcode.bitstart`). The AI does not do this.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Same as tongue: extracted from `obj.traj[1]` (bottom camera), looking for features with "paw" in the name.

ii.
```python
pt, paw_speed, paw_vis = extract_speed_from_traj_cam(cam_for_kin, tr, ['paw'])
```

iii. The reference uses only `top_paw` from the bottom camera. The AI's keyword matching for "paw" may match multiple paw features.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same velocity computation as tongue: frame-to-frame speed without smoothing, confidence threshold > 0.5 rather than > 0.9, then binned into time edges.

ii.
```python
# Same extract_speed_from_traj_cam function as tongue
sp[good] = np.sqrt((dxy[good]**2).sum(axis=1)) / dt[good]
```

iii. Results show ~62% "not visible" for paw, compared to ~19% in the reference, suggesting the extraction captures some but not most of the data.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: `discretize_trace_per_session` called per-trial rather than per-session.

ii.
```python
paw_tv.append(discretize_trace_per_session(
    bin_timeseries_to_edges(pt, paw_speed, t_edges),
    bin_timeseries_to_edges(pt, paw_vis.astype(float), t_edges) > 0
))
```

iii. Per-trial threshold instead of per-session threshold.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: frame times used directly without video-behavior clock offset correction.

ii.
```python
ft = np.asarray(cam['frameTimes'][trial_idx]).astype(np.float32)
```

iii. No offset correction applied. See 7-d.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI loads motion energy from external `motionEnergy_<anm>_<date>.mat` files using `scipy.io.loadmat`. It accesses the `me` variable and extracts the `data` field.

ii.
```python
def load_motion_energy_for_session(session_path):
    mefile = session_path.parent / f'motionEnergy_{subj}_{date}.mat'
    return sio.loadmat(str(mefile), squeeze_me=True, struct_as_record=False).get('me')
```

iii. The AI correctly identifies the external motion energy files.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI creates a fake time axis using `np.linspace` rather than using actual camera frame times. The motion energy values are then binned into time edges using this fabricated time axis.

ii.
```python
mt = np.linspace(t_edges[0], t_edges[-1], num=md.shape[0], dtype=np.float32)
mb = bin_timeseries_to_edges(mt, md, t_edges)
me_tv.append(discretize_trace_per_session(mb, np.isfinite(mb)))
```

iii. Using `np.linspace` assumes the motion energy frames are uniformly distributed over the entire time window, which is incorrect. The actual frame times should come from the side camera's `frameTimes`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same per-trial discretization issue: `discretize_trace_per_session` is called per-trial, computing a per-trial median rather than per-session median.

ii.
```python
me_tv.append(discretize_trace_per_session(mb, np.isfinite(mb)))
```

iii. Despite results showing ~50/50 split for motion energy, this is because `nanmedian` on a per-trial basis happens to produce a similar split to per-session median.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is placed on a fake `np.linspace` time axis from -2.5 to +2.5s and binned. No video offset correction is applied. The actual camera frame times are not used.

ii.
```python
mt = np.linspace(t_edges[0], t_edges[-1], num=md.shape[0], dtype=np.float32)
```

iii. This assumes frames are uniformly spaced over the full 5-second window, which is incorrect since camera frames have their own timing.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several missing data cases: (1) sessions without motion energy files get all-"no_video" output; (2) trials without camera tracking data get all-"not_visible" output; (3) trials where `haveEphys` is false are excluded; (4) sessions with fewer than 10 units or fewer than 2 valid trials are skipped. However, the AI does not handle: mismatched frame counts between cameras, NaN frame times, or the v5 vs v7.3 MATLAB file format differences.

ii.
```python
if cam_for_kin is not None and tr < len(cam_for_kin['ts']):
    # process tracking
else:
    tongue_tv.append(np.full(len(t_centers), 2, dtype=np.int64))
    paw_tv.append(np.full(len(t_centers), 2, dtype=np.int64))
```

iii. The AI's CONVERSION_NOTES Step 10 documents fixing quality-string parsing and motion-energy loading issues.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the MATLAB files with `mat73.loadmat` dominates runtime. Per the conversion output, individual sessions take 8-25 seconds each, for a total of approximately 380 seconds (6+ minutes) for all 25 sessions.

ii.
```python
obj = mat73.loadmat(str(path))['obj']
```

iii. The AI's CONVERSION_NOTES Step 7 estimates ~4.5-6.6 seconds per session for sample data.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The `bin_unit_trialtm` function loops over all units and then over all unique trials per unit, calling `np.histogram` once per unit per trial. This could be vectorized with `np.histogram2d` as the reference does. The `bin_timeseries_to_edges` function loops over bins individually rather than using vectorized binning.

ii.
```python
def bin_unit_trialtm(units, n_trials, t_edges):
    for ui, u in enumerate(units):
        for tr_ix in np.unique(tr):
            counts, _ = np.histogram(x, bins=t_edges)

def bin_timeseries_to_edges(times, values, t_edges):
    for i in range(len(t_edges)-1):
        m = (times >= t_edges[i]) & (times < t_edges[i+1])
```

iii. No documentation on efficiency optimization in CONVERSION_NOTES.

## 11-c. What processing does the code repeat multiple times?

i. The `extract_speed_from_traj_cam` function reloads and re-indexes the camera tracking data for each trial and each feature (tongue and paw), rather than caching it. The `feat_names` parsing is repeated per trial.

ii.
```python
for tr in range(n_trials):
    tt, tongue_speed, tongue_vis = extract_speed_from_traj_cam(cam_for_kin, tr, ['tongue'])
    pt, paw_speed, paw_vis = extract_speed_from_traj_cam(cam_for_kin, tr, ['paw'])
```

iii. No documentation on repeated computation.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `flatten_units` function reads `tm` (spike times in absolute time) for every unit, which is never used -- only `trialtm` (trial-relative times) is used for binning. The `have_vid` array is computed but only used in the fallback `placeholder_timevarying_outputs` function which is never called in the final code path. The `context_labels` array of strings is computed but only used for a diagnostic sample in `info`.

ii.
```python
'tm': np.asarray(group.get('tm', [])[i]).astype(np.float32),  # never used
# ...
have_vid = np.asarray(obj.get('trials', {}).get('bp', {}).get('haveVid', ...))  # barely used
```

iii. No documentation on unnecessary processing.
