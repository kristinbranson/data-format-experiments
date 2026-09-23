# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the analyzed session list in `SESSIONS`, with each entry specifying animal, date, ALM probe(s), and task type. It loads each `data_structure_<anm>_<date>.mat` with `pymatreader`, and loads motion energy separately from `motionEnergy_<anm>_<date>.mat`, falling back to `obj["me"]` if needed.

ii.
```python
SESSIONS = ([(a, d, p, "fixed") for a, d, p in FIXED_DELAY]
            + [(a, d, p, "randomized") for a, d, p in RANDOMIZED_DELAY])

def load_obj(anm, date, task):
    from pymatreader import read_mat
    return read_mat(data_path(anm, date, task))["obj"]

def load_motion_energy(anm, date, task, obj, ntrials):
    from pymatreader import read_mat

    me = None
    p = me_path(anm, date, task)
    if os.path.exists(p):
        me = read_mat(p)["me"]
    elif "me" in obj:
        me = obj["me"]
```

iii. In `CONVERSION_NOTES.md`, the agent says the session table was transcribed from the authors' `load*_ALMVideo.m` files, and that only the ephys+video directories were eligible for a neural decoder.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are taken directly from the session tuples and later deduplicated into a sorted `subjects` list. Each kept session stores its mouse in `info["anm"]`, and `subject_idx` is built by indexing that mouse into the sorted subject list.

ii.
```python
info = dict(
    anm=anm, date=date, task=task, probes=list(probes),
    session=f"{anm}_{date}",
    ...
)

subjects = sorted({r["info"]["anm"] for _, r in kept})
...
"subject_idx": np.array([subjects.index(r["info"]["anm"]) for _, r in kept],
                        dtype=np.int64),
```

iii. The notes justify this by saying the load-script session list is authoritative and already encodes animal identity cleanly.

## 1-c. How are the data split into sessions?

i. One hard-coded entry in `SESSIONS` is treated as one session. Each entry is routed to either `/app/data/Ephys_Behavior` or `/app/data/RandomizedDelay_Ephys_Behavior`, processed once, and then becomes one element in the session lists under `neural`, `input`, and `output`.

ii.
```python
DATA_DIR = {"fixed": "/app/data/Ephys_Behavior",
            "randomized": "/app/data/RandomizedDelay_Ephys_Behavior"}

def data_path(anm, date, task):
    return os.path.join(DATA_DIR[task], f"data_structure_{anm}_{date}.mat")

sessions = SESSIONS
...
for r in ex.map(process_session, sessions):
    results.append(r)
```

iii. The notes say the two task folders are treated uniformly and that sessions not present in the authors' load scripts were intentionally excluded.

## 1-d. How are the data split into trials?

i. Trials are defined by the first `Ntrials` entries of Bpod arrays inside `obj["bp"]`. The script keeps trial indices as raw 0-based row indices into those arrays, and uses those indices consistently for neural, behavioral, and video-derived outputs.

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

for k, it in enumerate(trials):
    neural.append(np.ascontiguousarray(rates[:, k, :], dtype=np.float32))
    ...
    out[0] = lick[it]
```

iii. The notes describe Bpod trial rows as the canonical trial structure, so the agent did not attempt to reconstruct trials from spike or video timing.

## 1-e. How are trials filtered based on quality controls?

i. The script removes early-lick trials, photostimulation trials, and trials with no electrophysiology on the designated probe(s). Its "no ephys" test is broader than the human reference: it checks whether any sorted spike exists on those probe(s), even from clusters that will later fail unit-quality filtering.

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

iii. In the notes, the agent says dropping early and photostim trials follows the paper, and dropping no-ephys trials was added because all-zero neural trials are invalid for the decoder harness.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from `obj["clu"]`, specifically each kept cluster's `trialtm` and `trial` arrays, together with per-trial go-cue times from `obj["bp"]["ev"]["goCue"]`.

ii.
```python
trialtm = c["trialtm"] if isinstance(c["trialtm"], list) else [c["trialtm"]]
trialno = c["trial"] if isinstance(c["trial"], list) else [c["trial"]]
tm = np.ravel(np.asarray(trialtm[i], dtype=np.float64))
tr = np.ravel(np.asarray(trialno[i], dtype=np.float64)).astype(np.int64) - 1
...
aligned = tm - align_times[tr]
```

iii. The notes map this directly to `alignSpikes.m` and `getSeq.m`, and describe the stored neural signal as go-cue-aligned spiking.

## 2-b. How is the `neural` data processed?

i. The agent bins spikes from -2.5 s to +2.5 s around the go cue in 10 ms bins, converts counts to spikes/s by dividing by `DT`, and then smooths with an exact causal `gausswin(15)`-style kernel implemented in `my_smooth`.

ii.
```python
ALIGN_EVENT = "goCue"
TMIN, TMAX, DT = -2.5, 2.5, 1.0 / 100.0
...
rate = counts / DT
...
m = rate.reshape(-1, NT).T
m = my_smooth(m)
return m.T.reshape(len(cluster_ids), ntrials, NT)
```

iii. The notes explicitly justify this as reproducing the paper's shipped MATLAB analysis scripts rather than the default parameter file, and claim it is identical to `obj.trialdat`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cluster filtering happens in two stages. First, `select_clusters` drops only clusters whose trimmed quality label is exactly one of `garbage`, `gabrga`, `noisy`, or `real?`; it does not lowercase labels and does not exclude `poor`. Second, clusters whose mean firing rate over the full analysis window is not greater than 1 Hz are removed.

ii.
```python
QUALITY_REJECT = ("garbage", "gabrga", "noisy", "real?")
LOW_FR = 1.0

def select_clusters(obj, probes):
    ...
    for i, q in enumerate(quals):
        q = str(q).strip() if isinstance(q, (str, np.str_)) else ""
        if q not in QUALITY_REJECT:
            keep.append((p - 1, i))
    return keep
...
mean_fr = rates.reshape(n_quality, -1).mean(axis=1)
keep_unit = mean_fr > LOW_FR
```

iii. In `CONVERSION_NOTES.md`, the agent argues this matches `findClusters(..., {'all'})` exactly, including keeping the single `Noisy` label because MATLAB's matching is case-sensitive.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned by subtracting each spike's trial-specific go-cue time from `trialtm`, so all neural data are expressed relative to go-cue onset.

ii.
```python
gocue = vec(bp["ev"][ALIGN_EVENT])[:ntrials_all]
...
aligned = tm - align_times[tr]
```

iii. The notes explicitly map this to `alignSpikes.m` with `params.alignEvent = 'goCue'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 10 ms bins over a 5 s window, for 500 bins per trial. The script does not do any later temporal rebinning; the 10 ms grid is the stored grid.

ii.
```python
TMIN, TMAX, DT = -2.5, 2.5, 1.0 / 100.0
EDGES = np.round(np.arange(TMIN, TMAX + DT / 2.0, DT), 10)
TIME = (EDGES[:-1] + DT / 2.0).astype(np.float64)
NT = TIME.size
```

iii. The notes justify 10 ms by citing the figure scripts and treat comments or defaults implying 5 ms as stale or unused.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This input is not read from a dedicated raw variable. It is derived from the analysis time axis centered on the go cue, i.e. from the chosen go-cue-aligned window and bin centers.

ii.
```python
TIME = (EDGES[:-1] + DT / 2.0).astype(np.float64)
...
input_trial = np.ascontiguousarray(TIME[None, :], dtype=np.float32)
```

iii. The notes say this is the task-mandated decoder input and should be the continuous time relative to the alignment event.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The script uses the centers of the fixed 10 ms go-cue-aligned bins as the sole input channel. Every trial receives the same 1 by 500 array of bin-center times.

ii.
```python
TIME = (EDGES[:-1] + DT / 2.0).astype(np.float64)
...
input_trial = np.ascontiguousarray(TIME[None, :], dtype=np.float32)
for k, it in enumerate(trials):
    inputs.append(input_trial)
```

iii. The agent's notes justify this as the simplest direct encoding of "time from go cue" required by the task.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is the neural time base itself. The same `TIME` vector that defines the spike bin centers is stored as the per-trial decoder input.

ii.
```python
TIME = (EDGES[:-1] + DT / 2.0).astype(np.float64)
...
aligned = tm - align_times[tr]
...
input_trial = np.ascontiguousarray(TIME[None, :], dtype=np.float32)
```

iii. The notes treat this as exact alignment because both neural data and decoder input are defined on the same fixed go-cue-centered grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the Bpod trial flags `R`, `L`, `hit`, `miss`, and `no`.

ii.
```python
R = vec(bp["R"])[:ntrials_all] > 0
L = vec(bp["L"])[:ntrials_all] > 0
hit = vec(bp["hit"])[:ntrials_all] > 0
miss = vec(bp["miss"])[:ntrials_all] > 0
no = vec(bp["no"])[:ntrials_all] > 0
```

iii. The notes say this follows the paper's `getPrevChoice.m`, except that ignore trials are retained as an explicit third class to satisfy the decoder spec.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The code assigns left on `(L & hit) | (R & miss)`, right on `(R & hit) | (L & miss)`, and none on `no` trials. The resulting per-trial value is repeated across all time bins.

ii.
```python
lick = np.full(ntrials_all, 2, dtype=np.int8)
lick[(L & hit) | (R & miss)] = 0
lick[(R & hit) | (L & miss)] = 1
lick[no] = 2
...
out[0] = lick[it]
```

iii. The notes justify this as the explicit categorical replacement for the paper's NaN-on-ignore choice variable.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `bp["autowater"]`.

ii.
```python
autowater = vec(bp["autowater"])[:ntrials_all] > 0
```

iii. The notes say `autowater` is the reference code's proxy for water-cued versus delayed-response context.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The script relabels `autowater == True` as WC (`0`) and everything else as DR (`1`), then repeats that per-trial value across time bins.

ii.
```python
context = np.where(autowater, 0, 1).astype(np.int8)
...
out[1] = context[it]
```

iii. The notes justify this with the block structure described in the paper and the reference code's use of `autowater` in trial conditions.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the Bpod trial flags `hit`, `miss`, and `no`.

ii.
```python
hit = vec(bp["hit"])[:ntrials_all] > 0
miss = vec(bp["miss"])[:ntrials_all] > 0
no = vec(bp["no"])[:ntrials_all] > 0
```

iii. The notes say this is the explicit-class version of the paper's outcome coding, which otherwise leaves ignore trials as NaN.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code starts with all trials labeled incorrect (`0`), then overwrites hits as correct (`1`) and `no` trials as ignore (`2`). As with the other per-trial outputs, the value is repeated across all bins.

ii.
```python
outcome = np.full(ntrials_all, 0, dtype=np.int8)
outcome[hit] = 1
outcome[no] = 2
...
out[2] = outcome[it]
```

iii. The notes justify keeping ignore as a third explicit class because the decoder task requires it.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the side-camera DeepLabCut feature named `"tongue"` in `obj["traj"][0]["ts"]`, together with side-camera `frameTimes`, go-cue times, and the session video offset from `obj["sglx"]` and `bp.ev.bitStart`. The agent does not use the bottom-camera tongue feature.

ii.
```python
TONGUE_VIEW, TONGUE_FEAT = 0, "tongue"
...
ft_side = frame_times(obj, TONGUE_VIEW, it)
...
i_tongue = feature_index(obj, TONGUE_VIEW, TONGUE_FEAT)
...
x, y = interp_feature(ts, i_tongue, ft_side, align, taxis)
```

iii. The notes justify this as using the canonical tongue-tip marker from the side camera, and explicitly say the bottom-camera tongue view was not used.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each kept trial, the script linearly interpolates side-camera tongue x/y onto the 10 ms neural time base, computes velocity magnitude with `np.gradient` over contiguous visible segments, leaves missing bins as NaN, and later discretizes the result. It does not smooth the position, does not differentiate with respect to actual frame times, and does not combine the two tongue camera views.

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
            vx[seg] = 0.0
            vy[seg] = 0.0
        else:
            vx[seg] = np.gradient(x[seg])
            vy[seg] = np.gradient(y[seg])
    return np.hypot(vx, vy)
...
tongue_v[k] = feature_speed(x, y, subtract_baseline=False)
```

iii. The notes argue this was a deliberate adaptation: because "not visible" is its own decoder class, the agent chose to differentiate only on visible segments and not to fill or smooth across visibility gaps.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Tongue velocity is thresholded per session at the median of all valid tongue-velocity values from kept trials. Values below the median become class `0`, values at or above the median become class `1`, and NaNs become class `2` (`not_visible`).

ii.
```python
def discretise(v):
    valid = np.isfinite(v)
    out = np.full(v.shape, 2, dtype=np.int8)
    if valid.any():
        thr = float(np.median(v[valid]))
        out[valid] = (v[valid] >= thr).astype(np.int8)
    else:
        thr = np.nan
    return out, thr

tongue_c, thr_tongue = discretise(tongue_v)
```

iii. The notes explicitly justify excluding invalid bins from the percentile because otherwise the large number of invisible tongue bins would collapse the threshold.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The script computes `align = goCue + vidshift`, which is equivalent to subtracting `videoOffset` and then subtracting the go cue from raw frame times. It then interpolates the aligned tongue trace onto the same 10 ms `TIME` grid used for neural data.

ii.
```python
def video_shift(obj):
    bit_start = mode_of(vec(obj["bp"]["ev"]["bitStart"]))
    fs = float(vec(obj["sglx"]["fs"])[0])
    vid_file_offset = mode_of(vec(obj["sglx"]["bitcode"]["bitstart"])) / fs
    return vid_file_offset - bit_start
...
align = gocue[it] + vidshift
...
x, y = interp_feature(ts, i_tongue, ft_side, align, taxis)
```

iii. The notes explicitly map this to `findVideoOffset.m` and say there is no additional neural/video lag.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom-camera DeepLabCut features `"top_paw"` and `"bottom_paw"` in `obj["traj"][1]["ts"]`, along with bottom-camera `frameTimes`, go-cue timing, and video offset.

ii.
```python
PAW_VIEW, PAW_FEATS = 1, ("top_paw", "bottom_paw")
...
ft_bot = frame_times(obj, PAW_VIEW, it)
...
i_paws = [feature_index(obj, PAW_VIEW, f) for f in PAW_FEATS]
```

iii. The notes say the agent intentionally used the visible subset of the two bottom-camera paws because either single paw could disappear for large parts of a session.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The script interpolates each paw's x/y trace onto the 10 ms neural grid, computes speed with `feature_speed(..., subtract_baseline=True)`, and then averages the two paw speeds with `np.nanmean`, so bins are retained if either paw is visible.

ii.
```python
for ip in i_paws:
    x, y = interp_feature(ts, ip, ft_bot, align, taxis)
    sp.append(feature_speed(x, y, subtract_baseline=True))
sp = np.stack(sp, axis=0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    paw_v[k] = np.nanmean(sp, axis=0)
```

iii. The notes justify this as a deliberate departure from single-marker tracking, arguing that averaging the visible paws avoids turning tracking dropout into the dominant class.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity uses the same per-session median split as tongue velocity: class `0` below threshold, class `1` at or above threshold, and class `2` where neither paw is visible after interpolation.

ii.
```python
paw_c, thr_paw = discretise(paw_v)
...
out[4] = paw_c[k]
```

iii. The notes describe the threshold as the session median over valid timepoints only.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw features use the same session video offset and go-cue alignment as tongue features, but use bottom-camera frame times. The aligned traces are interpolated onto the same 10 ms neural grid.

ii.
```python
vidshift = video_shift(obj)
...
ft_bot = frame_times(obj, PAW_VIEW, it)
...
align = gocue[it] + vidshift
...
x, y = interp_feature(ts, ip, ft_bot, align, taxis)
```

iii. The notes say all video-derived signals use `frameTimes - videoOffset - goCue` with `params.advance_movement = 0`.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motionEnergy_<anm>_<date>.mat` when present, or from `obj["me"]` otherwise. The per-trial traces are paired with side-camera `frameTimes` to align them to the neural time base.

ii.
```python
me_traces = load_motion_energy(anm, date, task, obj, ntrials_all)
...
tr = me_traces[it]
if tr is not None and ft_side is not None and tr.size == ft_side.size:
    me_v[k] = interp_trace(ft_side - align, tr, taxis)
```

iii. The notes justify the fallback because some session objects embed motion energy in a different layout than the standalone files.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The script does not recompute motion energy from video pixels. Instead, it takes the provided per-frame motion-energy trace and linearly interpolates it onto the 10 ms neural grid before discretizing it.

ii.
```python
def interp_trace(src_t, src_y, taxis):
    return np.interp(taxis, src_t, src_y, left=np.nan, right=np.nan)
...
if tr is not None and ft_side is not None and tr.size == ft_side.size:
    me_v[k] = interp_trace(ft_side - align, tr, taxis)
```

iii. The notes say motion energy was already precomputed by the authors, so the agent treated the released trace as the authoritative raw variable and only resampled it onto the neural time base.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized per session using the same `discretise` helper as tongue and paw velocity: below-median valid values become class `0`, at-or-above-median valid values become class `1`, and NaNs become class `2` (`no_video`).

ii.
```python
me_c, thr_me = discretise(me_v)
...
out[5] = me_c[k]
```

iii. The notes justify class `2` as covering both missing video coverage and completely unusable video timing.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy uses side-camera frame times, subtracts the session video offset and trial go-cue time, and then interpolates onto the 10 ms neural time grid.

ii.
```python
ft_side = frame_times(obj, TONGUE_VIEW, it)
align = gocue[it] + vidshift
...
me_v[k] = interp_trace(ft_side - align, tr, taxis)
```

iii. The notes group motion energy with the other video-derived signals under the same `findVideoOffset.m`-style alignment rule.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or unusable `frameTimes` are turned into `None`, which leaves the corresponding tongue/paw/motion-energy arrays as NaN and therefore class `2` after discretization. Missing standalone motion-energy files fall back to `obj["me"]`. Bins outside the source time range also remain NaN. The code does not explicitly threshold by DLC likelihood; missingness is whatever survives interpolation as non-finite values.

ii.
```python
def frame_times(obj, view, itrial):
    ft = obj["traj"][view]["frameTimes"]
    ...
    if a.size < 2 or not np.all(np.isfinite(a)):
        return None
    return a

tongue_v = np.full((ntr, NT), np.nan)
paw_v = np.full((ntr, NT), np.nan)
me_v = np.full((ntr, NT), np.nan)
...
out = np.full(v.shape, 2, dtype=np.int8)
```

iii. The notes say class `2` is intended to honestly encode tracking dropout and missing video rather than fabricating values there.

## 11-a. What are the most time-consuming steps of the code?

i. The main bottleneck is reading and unpacking the MATLAB session files. The script records load, neural, and video timing separately, and the conversion log shows file loading dominates per-session runtime.

ii.
```python
t0 = time.time()
obj = load_obj(anm, date, task)
t_load = time.time() - t0
...
timing=dict(load=t_load, neural=t_neural, video=t_video,
            total=time.time() - t0),
```

iii. The notes and `conversion_full_out.txt` both emphasize that loading is much slower than the numerical processing.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The script still loops over clusters in `bin_and_smooth` and over kept trials when building tongue, paw, and motion-energy outputs. Those per-trial video loops could in principle be refactored, but the agent left them scalar because every trial can have different frame timing and missingness.

ii.
```python
for ci, (p, i) in enumerate(cluster_ids):
    ...
for k, it in enumerate(trials):
    align = gocue[it] + vidshift
    ...
    if ft_side is not None:
        ...
    if ft_bot is not None:
        ...
```

iii. The notes explicitly say variable-length video streams limited vectorization opportunities, while session-level parallelism via `ProcessPoolExecutor` provided the main speedup.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats some bookkeeping work. `trials_with_ephys` is run once inside `select_trials` and again when filling `info["n_trials_no_ephys"]`. It also re-reads some Bpod vectors for metadata after trial selection, and within each trial it separately uses side-camera timing for tongue and motion energy.

ii.
```python
def select_trials(bp, obj=None, probes=()):
    ...
    if obj is not None:
        keep &= trials_with_ephys(obj, probes, n)
...
info = dict(
    ...
    n_trials_no_ephys=int(np.sum(~trials_with_ephys(obj, probes, ntrials_all))),
    ...
)
```

iii. The notes mostly emphasize parallelism and correctness rather than eliminating repeated bookkeeping.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The conversion computes and stores substantial metadata that the downstream decoder does not use, including `two_context`, raw `trial_index`, `cluster_index`, per-session thresholds, and detailed timing breakdowns. The script also contains optional diagnostic plotting code (`show_processing`) that is outside the core conversion path.

ii.
```python
info = dict(
    anm=anm, date=date, task=task, probes=list(probes),
    session=f"{anm}_{date}",
    two_context=(anm, date) in TWO_CONTEXT,
    ...
    trial_index=trials.astype(np.int32),
    cluster_index=np.array([(p + 1, i) for p, i in cluster_ids], dtype=np.int32)
                           [keep_unit] if n_quality else np.zeros((0, 2), np.int32),
    ...
    thr_tongue=thr_tongue, thr_paw=thr_paw, thr_me=thr_me,
    timing=dict(load=t_load, neural=t_neural, video=t_video,
                total=time.time() - t0),
)

def show_processing(args, result, outdir="/app"):
    ...
```

iii. The notes frame these extras as verification and traceability features rather than requirements of the decoder dataset itself.
