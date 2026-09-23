# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each session is loaded from a `data_structure_<ANM>_<DATE>.mat` file using `pymatreader.read_mat()`, which handles both MATLAB v7 and v7.3 formats transparently. Motion energy is loaded separately from `motionEnergy_<ANM>_<DATE>.mat` files using the same library. The 44 sessions and their ALM probe numbers are hard-coded in `SESSIONS` (as `FIXED_DELAY` + `RANDOMIZED_DELAY` lists), transcribed from the authors' `load<ANM>_ALMVideo.m` scripts. Data directories are split by task type (`Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior`). Sessions are processed in parallel using `ProcessPoolExecutor`.

ii.
```python
FIXED_DELAY = [
    ("JEB6", "2021-04-18", [2]),
    ...
]
RANDOMIZED_DELAY = (
    [("JEB11", "2022-05-10", [1]), ...]
    + [("JEB23", d, [1]) for d in (...)]
    + [("JEB24", d, [1]) for d in (...)]
)
SESSIONS = ([(a, d, p, "fixed") for a, d, p in FIXED_DELAY]
            + [(a, d, p, "randomized") for a, d, p in RANDOMIZED_DELAY])

def load_obj(anm, date, task):
    from pymatreader import read_mat
    return read_mat(data_path(anm, date, task))["obj"]
```

iii. The session list is taken verbatim from the authors' `DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m` scripts, excluding sessions the authors commented out and 3 extra files on disk not in the load scripts. This reproduces the paper's n=25 fixed-delay and n=19 randomized-delay sessions exactly. `pymatreader` was chosen because it reads both MAT v7 and v7.3 with a single interface.

## 1-b. How are the data split into subjects?

i. The animal name is the first element of each session tuple (e.g., `"JEB19"`). At assembly, `subjects` is the sorted set of unique animal names, and `subject_idx` maps each session to its index in that list. The 44 sessions come from 14 animals.

ii.
```python
subjects = sorted({r["info"]["anm"] for _, r in kept})
data = {
    ...
    "subjects": subjects,
    "subject_idx": np.array([subjects.index(r["info"]["anm"]) for _, r in kept],
                            dtype=np.int64),
    ...
}
```

iii. The animal name is stored directly in the session tuple, which comes from the load scripts. This is consistent with how the authors identify animals.

## 1-c. How are the data split into sessions?

i. One session = one entry in the `SESSIONS` list = one `.mat` file on disk. Each session is identified by `(anm, date, probes, task)`. The task type determines which data directory to look in. Each session becomes one element in the `neural`, `input`, and `output` lists. The result is 44 sessions: 25 fixed-delay and 19 randomized-delay.

ii.
```python
DATA_DIR = {"fixed": "/app/data/Ephys_Behavior",
            "randomized": "/app/data/RandomizedDelay_Ephys_Behavior"}

def data_path(anm, date, task):
    return os.path.join(DATA_DIR[task], f"data_structure_{anm}_{date}.mat")
```

iii. Sessions match the authors' load scripts exactly.

## 1-d. How are the data split into trials?

i. Trials are defined by the Bpod structure (`obj.bp`), with `Ntrials` giving the count. Each per-trial field (hit, miss, R, L, etc.) has one entry per trial. Spike times carry a `trial` field (1-based) indicating which trial each spike belongs to. No trial boundaries need to be reconstructed.

ii.
```python
def select_trials(bp, obj=None, probes=()):
    n = int(vec(bp["Ntrials"])[0])
    early = vec(bp["early"])[:n] > 0
    stim = vec(bp["stim"]["enable"])[:n] > 0
    keep = ~(early | stim)
    if obj is not None:
        keep &= trials_with_ephys(obj, probes, n)
    return np.flatnonzero(keep), n
```

iii. The Bpod table directly defines trials with one go cue per trial, so no inference is needed.

## 1-e. How are trials filtered based on quality controls?

i. Three filters are applied: (1) Early-lick trials (`bp.early`) are dropped, following the paper's Methods ("omitted from analyses"). (2) Photostimulation trials (`bp.stim.enable`) are dropped, as in every `params.condition` in the reference code. (3) Trials with no electrophysiology data are dropped via `trials_with_ephys()`, which detects trials where no cluster on the ALM probe has any spike (64 trials across the dataset, concentrated in two JEB24 sessions where the SpikeGLX recording stopped early). This keeps 13,762 of 14,972 raw trials.

ii.
```python
def trials_with_ephys(obj, probes, ntrials):
    ...
    has = np.zeros(ntrials, dtype=bool)
    for p in probes:
        c = clu[p - 1]
        ...
        for t in trialno:
            ...
            has[a] = True
    return has

def select_trials(bp, obj=None, probes=()):
    n = int(vec(bp["Ntrials"])[0])
    early = vec(bp["early"])[:n] > 0
    stim = vec(bp["stim"]["enable"])[:n] > 0
    keep = ~(early | stim)
    if obj is not None:
        keep &= trials_with_ephys(obj, probes, n)
    return np.flatnonzero(keep), n
```

iii. Early-lick and photostim removal follows the paper's Methods and every `params.condition`. The no-ephys filter is needed because some trials have no neural data (recording ended), which would create all-zero neural matrices. The AI's CONVERSION_NOTES document this was added after discovering 61 warnings in the first full conversion run.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` provides the spike-sorted clusters. Each cluster has `trial` (1-based trial index), `trialtm` (spike time relative to trial start), and `quality` (manual curation label). `bp.ev.goCue` provides the alignment times.

ii.
```python
trialtm = c["trialtm"] if isinstance(c["trialtm"], list) else [c["trialtm"]]
trialno = c["trial"] if isinstance(c["trial"], list) else [c["trial"]]
tm = np.ravel(np.asarray(trialtm[i], dtype=np.float64))
tr = np.ravel(np.asarray(trialno[i], dtype=np.float64)).astype(np.int64) - 1
...
aligned = tm - align_times[tr]   # alignSpikes.m
```

iii. These are the same variables used by `alignSpikes.m` and `getSeq.m` in the reference code.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to the go cue (`trialtm - goCue`), binned into 10 ms bins over [-2.5, 2.5] s (500 bins), converted to firing rates (counts / dt), and smoothed with a causal Gaussian kernel that exactly reproduces the reference `mySmooth.m`: `gausswin(15)` with the first 7 taps zeroed and a `'reflect'` boundary condition. The result is firing rates in spikes/s, matching `obj.trialdat` from the MATLAB pipeline. Units from both probes of a two-probe session are concatenated.

ii.
```python
TMIN, TMAX, DT = -2.5, 2.5, 1.0 / 100.0   # params.tmin / params.tmax / params.dt
SMOOTH_N = 15

def causal_gaussian(n=SMOOTH_N, alpha=2.5):
    k = np.arange(n)
    w = np.exp(-0.5 * (alpha * (k - (n - 1) / 2.0) / ((n - 1) / 2.0)) ** 2)
    w[: n // 2] = 0.0
    return w / w.sum()

def bin_and_smooth(obj, cluster_ids, align_times, ntrials):
    ...
    aligned = tm - align_times[tr]
    b = bin_index(aligned)
    ...
    flat = tr[ok] * NT + b[ok]
    counts[ci] = np.bincount(flat, minlength=ntrials * NT).reshape(ntrials, NT)
    rate = counts / DT
    m = rate.reshape(-1, NT).T
    m = my_smooth(m)
    return m.T.reshape(len(cluster_ids), ntrials, NT)
```

iii. The AI explicitly verified that `my_smooth` reproduces `mySmooth.m` to machine precision (2.2e-16) and that `causal_gaussian` matches `gausswin(15)` with the first 7 taps zeroed. The 10 ms bin size and causal Gaussian smoothing match `params.dt = 1/100` and `params.smooth = 15` as used in every figure script. The AI noted that `getDefaultParams.m` has `dt=1/200` but every actual analysis script overrides to `1/100`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) Quality filter (`select_clusters`): clusters with quality label matching `garbage`, `gabrga`, `noisy`, or `real?` are dropped, with case-sensitive, trimmed comparison exactly as in `findClusters.m`. (2) Firing rate filter: units with mean rate <= 1 Hz over the analysis window are dropped (`removeLowFRClusters`). Additionally, sessions with < 10 curated units are dropped (Methods).

ii.
```python
QUALITY_REJECT = ("garbage", "gabrga", "noisy", "real?")
LOW_FR = 1.0
MIN_UNITS = 10

def select_clusters(obj, probes):
    ...
    for i, q in enumerate(quals):
        q = str(q).strip() if isinstance(q, (str, np.str_)) else ""
        if q not in QUALITY_REJECT:
            keep.append((p - 1, i))
    return keep

# In process_session:
    mean_fr = rates.reshape(n_quality, -1).mean(axis=1)
    keep_unit = mean_fr > LOW_FR
    rates = rates[keep_unit]
```

iii. The quality drop set exactly matches `findClusters.m` with `quality={'all'}`. The comparison is case-sensitive as in MATLAB's `ismember`. The 1 Hz threshold matches `params.lowFR = 1` used in all figure scripts and the paper's Methods. The >=10 units session filter comes from the Methods.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue by subtracting `bp.ev.goCue[trial]` from each spike's `trialtm`. This is the same as `alignSpikes.m`: `clu.trialtm_aligned = clu.trialtm - ev.(alignEvent)(clu.trial)`.

ii.
```python
aligned = tm - align_times[tr]   # alignSpikes.m
```

iii. This exactly reproduces `alignSpikes.m` with `params.alignEvent = 'goCue'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 10 ms bins, 500 bins spanning [-2.5, 2.5] s from the go cue. The bin edges are `np.arange(-2.5, 2.5 + dt/2, dt)` (501 edges, 500 bins) and the bin centres are the midpoints. No temporal rebinning is applied — spikes are counted directly into these bins.

ii.
```python
TMIN, TMAX, DT = -2.5, 2.5, 1.0 / 100.0
EDGES = np.round(np.arange(TMIN, TMAX + DT / 2.0, DT), 10)   # 501 bin edges
TIME = (EDGES[:-1] + DT / 2.0).astype(np.float64)            # 500 bin centres
NT = TIME.size
```

iii. The AI explicitly checked that `getDefaultParams.m` has `dt=1/200` (5ms) but every actual analysis script (`Scripts/Figure*`, `Scripts/EDFigure*`, etc.) overrides this to `params.dt = 1/100` (10ms). The AI chose 10ms as the actual processing used by the paper. The window [-2.5, 2.5] matches `params.tmin`/`params.tmax`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Not derived from raw data variables. It is computed as the bin centres of the analysis window: evenly spaced values from -2.495 to +2.495 s in 10 ms steps.

ii.
```python
TIME = (EDGES[:-1] + DT / 2.0).astype(np.float64)  # 500 bin centres
input_trial = np.ascontiguousarray(TIME[None, :], dtype=np.float32)
```

iii. The input is the time axis itself, as specified by the task instructions.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing beyond computing the bin centres from the edge array. The same time array is shared across all trials.

ii.
```python
input_trial = np.ascontiguousarray(TIME[None, :], dtype=np.float32)   # shared, read-only
for k, it in enumerate(trials):
    ...
    inputs.append(input_trial)
```

iii. The time axis is defined by the analysis window parameters.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time values are the bin centres of the same binning grid used for the neural data. Bin k in the neural data and bin k in the input correspond to the same time interval.

ii. Neural binning uses `EDGES`, and the input is `TIME = EDGES[:-1] + DT/2`, so they share the same grid.

iii. N/A — they are the same grid by construction.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial fields: `bp.R` (instructed right), `bp.L` (instructed left), `bp.hit` (correct response), and `bp.miss` (incorrect response). Additionally `bp.no` (ignore trials) is used.

ii.
```python
R = vec(bp["R"])[:ntrials_all] > 0
L = vec(bp["L"])[:ntrials_all] > 0
hit = vec(bp["hit"])[:ntrials_all] > 0
miss = vec(bp["miss"])[:ntrials_all] > 0
no = vec(bp["no"])[:ntrials_all] > 0
```

iii. The lick direction is not directly recorded but can be inferred from the instructed side and outcome, following `getPrevChoice.m`.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Left lick = `(L & hit) | (R & miss)` (code 0), Right lick = `(R & hit) | (L & miss)` (code 1), None = `no` (ignore trials, code 2). This follows the formula in `getPrevChoice.m`.

ii.
```python
lick = np.full(ntrials_all, 2, dtype=np.int8)          # 2 = none
lick[(L & hit) | (R & miss)] = 0                        # left
lick[(R & hit) | (L & miss)] = 1                        # right
lick[no] = 2                                            # getPrevChoice: NaN on ignore
```

iii. A hit means the animal licked the correct (instructed) side, a miss means it licked the wrong side. This is the same logic as `getPrevChoice.m` where `choice = (R & hit) | (L & miss)`.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial field: `bp.autowater`. Autowater trials are the water-cued (WC) context; all others are delayed-response (DR).

ii.
```python
autowater = vec(bp["autowater"])[:ntrials_all] > 0
context = np.where(autowater, 0, 1).astype(np.int8)   # 0 = WC, 1 = DR
```

iii. The reference code uses `autowater` as the proxy for context in every `params.condition`. The AI verified this against the block structure described in the Methods.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct relabelling: autowater → WC (0), otherwise → DR (1).

ii.
```python
context = np.where(autowater, 0, 1).astype(np.int8)
```

iii. Matches the reference code's use of `autowater` in condition strings.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Three per-trial fields: `bp.hit`, `bp.miss`, and `bp.no`.

ii.
```python
hit = vec(bp["hit"])[:ntrials_all] > 0
miss = vec(bp["miss"])[:ntrials_all] > 0
no = vec(bp["no"])[:ntrials_all] > 0
outcome = np.full(ntrials_all, 0, dtype=np.int8)  # 0 = incorrect
outcome[hit] = 1                                    # 1 = correct
outcome[no] = 2                                     # 2 = ignore
```

iii. Follows `getOutcome.m`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Relabelling: miss → incorrect (0), hit → correct (1), no → ignore (2). The reference code sets NaN for ignore trials; here they are an explicit class as required by the task spec.

ii. Same as 6-a.

iii. Codes follow the task specification.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DLC tracking in `obj.traj` — specifically the side camera (view index 0), feature `"tongue"`. The tracking provides `ts` (frames x [x,y,confidence] x bodypart), `frameTimes`, and `featNames`. `bp.ev.goCue` and `obj.sglx` bitcode fields are needed for the video-to-behavior clock alignment.

ii.
```python
TONGUE_VIEW, TONGUE_FEAT = 0, "tongue"   # side camera
...
i_tongue = feature_index(obj, TONGUE_VIEW, TONGUE_FEAT)
...
if ft_side is not None:
    ts = np.asarray(ts_side[it], dtype=np.float64)
    if ts.ndim == 3 and ts.shape[0] == ft_side.size:
        x, y = interp_feature(ts, i_tongue, ft_side, align, taxis)
        tongue_v[k] = feature_speed(x, y, subtract_baseline=False)
```

iii. The AI uses only the side camera's tongue feature, matching `params.traj_features{1}{1}` in the reference MATLAB code. CONVERSION_NOTES Step 5: "Tongue: side-camera `tongue` feature (`params.traj_features{1}{1}`), the canonical tongue-tip marker."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Three steps: (1) Video clock alignment: `frameTimes - vidshift - goCue` puts frames on the go-cue-relative time axis. (2) Position interpolation: x, y coordinates are linearly interpolated from the frame times onto the 10 ms neural time base (`interp_trace`). NaN outside the source range. (3) Velocity: `feature_speed` computes `np.gradient` on each contiguous visible segment separately (never across gaps), then takes `np.hypot(vx, vy)` for speed. No baseline subtraction for tongue (`subtract_baseline=False`). NaN where the feature is not visible.

ii.
```python
def interp_feature(ts, featix, ft, align_time, taxis):
    src_t = ft - align_time
    x = interp_trace(src_t, ts[:, 0, featix], taxis)
    y = interp_trace(src_t, ts[:, 1, featix], taxis)
    return x, y

def feature_speed(x, y, subtract_baseline):
    visible = np.isfinite(x) & np.isfinite(y)
    ...
    for s, e in zip(starts, stops):
        seg = idx[s:e + 1]
        if seg.size == 1:
            vx[seg] = 0.0; vy[seg] = 0.0
        else:
            vx[seg] = np.gradient(x[seg])
            vy[seg] = np.gradient(y[seg])
    ...
    return np.hypot(vx, vy)
```

iii. The AI mirrors the MATLAB pipeline: `findPosition.m` interpolates position to `obj.time`, then `findVelocity.m` computes `gradient()`. CONVERSION_NOTES: "exactly reproduces `obj.trialdat` in the reference MATLAB pipeline". No smoothing of the interpolated position (tongue is not smoothed in `findPosition.m`).

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per-session median (50th percentile) of all valid (non-NaN) tongue velocity values across all kept trials. Class 0 = below median, class 1 = above or equal to median, class 2 = not visible (NaN).

ii.
```python
def discretise(v):
    valid = np.isfinite(v)
    out = np.full(v.shape, 2, dtype=np.int8)
    if valid.any():
        thr = float(np.median(v[valid]))
        out[valid] = (v[valid] >= thr).astype(np.int8)
    ...
    return out, thr

tongue_c, thr_tongue = discretise(tongue_v)
```

iii. The 50th percentile threshold follows the task specification. Valid-only computation excludes "not visible" bins from affecting the threshold.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The video offset is computed from `findVideoOffset.m`: `mode(sglx.bitcode.bitstart)/fs - mode(bp.ev.bitStart)`. Frame times are then corrected: `frameTimes - vidshift - goCue[trial]`. The corrected frame times are used to interpolate the tongue position directly onto the neural time base (`TIME`), so both share the same time axis.

ii.
```python
def video_shift(obj):
    bit_start = mode_of(vec(obj["bp"]["ev"]["bitStart"]))
    fs = float(vec(obj["sglx"]["fs"])[0])
    vid_file_offset = mode_of(vec(obj["sglx"]["bitcode"]["bitstart"])) / fs
    return vid_file_offset - bit_start

# In process_session:
    align = gocue[it] + vidshift
    ...
    x, y = interp_feature(ts, i_tongue, ft_side, align, taxis)
```

iii. This is the reference's `findVideoOffset.m`. The interpolation onto `TIME` ensures exact alignment with the neural bins.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The DLC tracking from the bottom camera (view index 1), using both `top_paw` and `bottom_paw` features. `frameTimes`, `bp.ev.goCue`, and `obj.sglx` bitcode fields for alignment.

ii.
```python
PAW_VIEW, PAW_FEATS = 1, ("top_paw", "bottom_paw")
...
i_paws = [feature_index(obj, PAW_VIEW, f) for f in PAW_FEATS]
...
for ip in i_paws:
    x, y = interp_feature(ts, ip, ft_bot, align, taxis)
    sp.append(feature_speed(x, y, subtract_baseline=True))
sp = np.stack(sp, axis=0)
paw_v[k] = np.nanmean(sp, axis=0)   # mean over the visible paw(s)
```

iii. The AI uses both paws from the bottom camera and averages over whichever are visible, citing that a single paw marker has high "not visible" rates (up to 88-96% in some sessions) while the union reduces this to 0-24%.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same as tongue: interpolate position to the neural time base, compute per-segment velocity via `np.gradient` + `np.hypot`. Unlike tongue, baseline drift subtraction is applied (`subtract_baseline=True`), following `findVelocity.m`. The speeds from both paws are averaged per time bin (using whichever is visible via `np.nanmean`).

ii.
```python
sp.append(feature_speed(x, y, subtract_baseline=True))
...
paw_v[k] = np.nanmean(sp, axis=0)
```

Where `feature_speed` with `subtract_baseline=True`:
```python
if subtract_baseline:
    d = np.diff(np.stack([x, y], axis=1), axis=0)
    base = np.nanmedian(d, axis=0)
    if np.isfinite(base[0]):
        vx = vx - base[0]
        vy = vy - base[0]
```

iii. The baseline subtraction reproduces `findVelocity.m`'s `basederiv = median(diff(pos),'omitnan'); xvel -= basederiv(1)` including what the AI identifies as "an apparent typo" in the reference code (using `basederiv(1)` for both x and y velocity).

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: per-session median of valid values. Class 0 = below, 1 = at or above, 2 = not visible.

ii.
```python
paw_c, thr_paw = discretise(paw_v)
```

iii. Follows the task specification (50th percentile threshold).

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same alignment as tongue: video offset subtracted from frame times, then `goCue` subtracted, then linear interpolation onto the neural time base `TIME`. The bottom camera's own frame times are used.

ii.
```python
ft_bot = frame_times(obj, PAW_VIEW, it)
...
x, y = interp_feature(ts, ip, ft_bot, align, taxis)
```

iii. Same offset and time grid as every other stream.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Separate `motionEnergy_<ANM>_<DATE>.mat` files, containing one trace per trial (one value per camera frame). Falls back to `obj.me` if the external file doesn't exist (though all 44 sessions have the external file).

ii.
```python
def load_motion_energy(anm, date, task, obj, ntrials):
    from pymatreader import read_mat
    me = None
    p = me_path(anm, date, task)
    if os.path.exists(p):
        me = read_mat(p)["me"]
    elif "me" in obj:
        me = obj["me"]
    ...
    if isinstance(me, dict):
        me = me["data"]
        if isinstance(me, dict):
            me = me["data"]
    ...
```

iii. Handles the three different file layouts found across sessions (bare cell array, struct with `data`, and nested `data.data`), matching `loadMotionEnergy.m`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The motion energy is already one value per frame (the paper computes it as |median(next 5 frames) - median(prev 5 frames)| with 99th percentile across pixels). The AI linearly interpolates the per-frame values onto the 10 ms neural time base using `interp_trace`, then discretizes at the session median.

ii.
```python
tr = me_traces[it]
if tr is not None and ft_side is not None and tr.size == ft_side.size:
    me_v[k] = interp_trace(ft_side - align, tr, taxis)
```

iii. Matches `loadMotionEnergy.m`'s `interp1(frameTimes - vidshift - alignTime, me, obj.time)`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same as tongue/paw: per-session median of valid values. Class 0 = below, 1 = at or above, 2 = no video.

ii.
```python
me_c, thr_me = discretise(me_v)
```

iii. Follows the task specification.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy has one value per frame of the side camera. Frame times are corrected by the video offset and go cue, then the trace is linearly interpolated onto the neural time base `TIME`.

ii.
```python
me_v[k] = interp_trace(ft_side - align, tr, taxis)
```

iii. Same alignment as `loadMotionEnergy.m`.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases: (1) **Trials with no ephys**: 64 trials where SpikeGLX stopped early are dropped entirely via `trials_with_ephys()`. (2) **Unusable frame times**: 3 trials have all-NaN `frameTimes`; the `frame_times()` function returns None, and video outputs become entirely class 2. (3) **Untracked frames**: Where DLC position is NaN (feature not visible), the velocity is NaN and discretized to class 2. (4) **Single-element MATLAB cell arrays**: `as_cell_list()` re-wraps unwrapped single elements. (5) **Missing/empty quality labels**: Treated as empty string (not in reject list, so kept). (6) **Motion energy trace length mismatch**: Falls back to class 2.

ii.
```python
def frame_times(obj, view, itrial):
    ...
    a = np.ravel(np.asarray(ft, dtype=np.float64))
    if a.size < 2 or not np.all(np.isfinite(a)):
        return None
    return a
```

iii. Nothing is interpolated or fabricated. Missing video/tracking data results in class 2 ("not visible" / "no video"), which is an honest representation. Trials with no neural data are dropped entirely since an all-zero neural matrix is not real data.

## 11-a. What are the most time-consuming steps of the code?

i. File loading via `pymatreader` dominates: 4-8 seconds per session. The full conversion runs in ~17 seconds wall clock with 16-worker parallelism (or ~250s serial).

ii.
```python
def load_obj(anm, date, task):
    from pymatreader import read_mat
    return read_mat(data_path(anm, date, task))["obj"]
```

iii. CONVERSION_NOTES Step 7: "pymatreader reads the whole obj, including clu.spkWavs, which dominates the runtime (6-8 s and ~1.4 GB for the largest 290 MB session)."

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial video processing loop (iterating over trials for tongue, paw, and motion energy interpolation and velocity computation) could potentially be vectorized if all trials had the same number of frames, but they don't. The per-cluster spike binning uses `np.bincount` for vectorization. The smoothing is applied as a single session-wide FIR operation.

ii.
```python
for k, it in enumerate(trials):
    align = gocue[it] + vidshift
    ...
    tongue_v[k] = feature_speed(x, y, subtract_baseline=False)
    ...
    paw_v[k] = np.nanmean(sp, axis=0)
    ...
    me_v[k] = interp_trace(ft_side - align, tr, taxis)
```

iii. Variable frame counts across trials prevent straightforward vectorization of the video processing. The AI vectorized the spike binning and smoothing steps.

## 11-c. What processing does the code repeat multiple times?

i. The video offset is computed once per session and reused. The frame times are read once per trial per camera view. The causal Gaussian kernel is precomputed at module level. No significant processing is repeated.

ii.
```python
KERNEL = causal_gaussian()  # module-level
...
vidshift = video_shift(obj)  # once per session
```

iii. The AI explicitly designed the code to avoid redundant computation.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `pymatreader.read_mat` loads the entire `obj` structure including fields never used (e.g., `clu.spkWavs`, `clu.tm`, unused DLC features, `obj.ex`/`obj.meta`). This accounts for most of the per-session load time. The `trials_with_ephys()` function reads spike data from quality-rejected clusters (to detect ephys-less trials), which is extra processing but necessary for correctness.

ii.
```python
def load_obj(anm, date, task):
    from pymatreader import read_mat
    return read_mat(data_path(anm, date, task))["obj"]  # reads everything
```

iii. Selective loading would require HDF5-specific code (only possible for v7.3 files), losing the uniform-interface benefit of pymatreader.
