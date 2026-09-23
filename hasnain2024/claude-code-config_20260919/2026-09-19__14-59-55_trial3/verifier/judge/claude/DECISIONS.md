# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each session is one MATLAB `.mat` file (`data_structure_<anm>_<date>.mat`) located in either `Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior`. The 44 sessions and their probes are hardcoded in a `SESSIONS` list, transcribed from the authors' `load<ANM>_ALMVideo.m` scripts. Two file format readers are implemented (`_H5Session` for v7.3/HDF5 and `_V7Session` for v7/MAT5) since the data files come in both formats. Motion energy is loaded separately from `motionEnergy_<anm>_<date>.mat` files.

ii.
```python
SESSIONS = [
    # (animal, date, probes (1-based), data directory, task)
    ('EKH1',  '2021-08-07', [2],    DATA_FIXED, 'DR/WC fixed delay'),
    ...
    ('JEB24', '2023-11-03', [1],    DATA_RAND,  'DR randomized delay'),
]

def open_session(path):
    return _H5Session(path) if _is_hdf5(path) else _V7Session(path)
```

iii. The session list follows the authors' own loading scripts exactly, including which sessions are commented out. Both MATLAB formats are handled because 11 of the 47 data files are v7 rather than v7.3.

## 1-b. How are the data split into subjects?

i. The animal name is the first element of each SESSIONS tuple (e.g., `'EKH1'`). A list of unique subjects is built during assembly by appending new animal names as they appear. `subject_idx` maps each session to its subject index.

ii.
```python
for k, (anm, date, probes, datadir, task) in enumerate(sessions):
    ...
    if anm not in subjects:
        subjects.append(anm)
    ...
    data['subject_idx'].append(subjects.index(anm))
```

iii. The animal name is taken from the session metadata tuple, which was transcribed from the reference loading scripts. The 44 sessions span 14 animals.

## 1-c. How are the data split into sessions?

i. One session is one entry in the `SESSIONS` list, keyed by `(animal, date, probes, datadir, task)`. The `process_session` function loads and converts one session at a time. Fixed-delay (25) and randomized-delay (19) sessions are processed uniformly.

ii.
```python
fn = os.path.join(datadir, f'data_structure_{anm}_{date}.mat')
obj = open_session(fn)
```

iii. Sessions follow the reference `load<ANM>_ALMVideo.m` scripts exactly, including the probe assignments.

## 1-d. How are the data split into trials?

i. Trials are defined by the Bpod trial table (`obj.bp`). The number of trials is `obj.bp.Ntrials`. Each per-trial field (hit, miss, R, L, etc.) has `Ntrials` entries. Spike times carry per-spike trial indices (`clu.trial`), so trial boundaries don't need to be reconstructed.

ii.
```python
ntrials = obj.Ntrials  # int(np.array(self.o['bp']['Ntrials']).ravel()[0])
gocue = obj.bp('ev.goCue')
hit, miss, no = obj.bp('hit'), obj.bp('miss'), obj.bp('no')
```

iii. The Bpod table directly defines trials with one go cue per trial. No inference of trial boundaries is needed.

## 1-e. How are trials filtered based on quality controls?

i. Three filters are applied: (1) Early-lick trials (`bp.early`) are excluded; (2) Photostimulation trials (`bp.stim.enable`) are excluded; (3) Trials with no spikes on any unit (ephys recording ended before the behavioral session) are excluded. Additionally, trials with non-finite go cue times are excluded. Across the dataset, 13,762 of ~15,000 trials are retained.

ii.
```python
keep = (early < 0.5) & (stim < 0.5) & np.isfinite(gocue)
spk_per_trial = trialdat.sum(axis=(0, 1))
no_ephys = spk_per_trial <= 0
keep &= ~no_ephys
```

iii. Early-lick and photostim removal follows the reference condition strings (`~stim.enable & ~early`). The no-ephys filter addresses sessions where the recording stopped before the behavioral session ended (64 trials in 2 sessions).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `clu{probe}(i).trialtm` (spike times relative to trial start), `clu{probe}(i).trial` (trial index per spike, 1-based), and `bp.ev.goCue` (go cue times for alignment). Quality labels `clu{probe}(i).quality` are used for filtering.

ii.
```python
tt, tr = obj.spikes(prb, i)  # trialtm, trial
al = tt - gocue[tr - 1]      # trialtm_aligned
```

iii. These are the same variables used by the reference's `alignSpikes.m`.

## 2-b. How is the `neural` data processed?

i. Aligned spike times are binned into 10 ms non-overlapping bins over [-2.5, 2.5] s (500 bins), divided by the bin width to get spikes/s, then smoothed with a causal Gaussian kernel ported from `mySmooth.m`: a `gausswin(15)` with first 7 taps zeroed and normalized, applied as a causal FIR filter using `scipy.signal.lfilter`. The boundary condition is 'reflect' (prepend reflected samples). Units from both probes of dual-probe sessions are concatenated.

ii.
```python
DT = 0.01  # params.dt = 1/100 s  -> 10 ms bins
SMOOTH_N = 15
EDGES, TAXIS = time_axis()  # edges = np.arange(TMIN, TMAX + DT/2, DT)

def _causal_kernel(N=SMOOTH_N):
    k = gausswin(N)
    k[:N // 2] = 0.0
    k = k / k.sum()
    return k[N // 2:]

out /= DT  # spikes / s
sm = my_smooth(out.reshape(NT, -1)).astype(np.float32)
```

iii. The AI explicitly ports `mySmooth.m` and uses `params.dt = 1/100` (10 ms) from `getDefaultParams.m`/`WorkingWithDataObjs.m`. The AI notes that 5 ms is mentioned in a tutorial comment but chose 10 ms as the main parameter value.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) Manual quality labels are checked case-insensitively against `{garbage, gabrga, noisy, real?}` — clusters with these labels are dropped. (2) Units whose mean firing rate over the analysis window across all trials is <= 1 Hz are dropped. This yields 2,457 units across 44 sessions.

ii.
```python
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}
keep_clu[prb] = [i for i, qq in enumerate(q) if qq.lower() not in BAD_QUALITY]
# ...
fr = mean_firing_rates(obj, probes, keep_clu, gocue, ntrials)
use = fr > LOW_FR  # LOW_FR = 1.0
```

iii. The quality label set matches `findClusters.m` exactly. The 1 Hz threshold matches `removeLowFRClusters.m` and the paper statement "All units with firing rates exceeding 1 Hz were included."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue by subtracting `bp.ev.goCue[trial]` from `clu.trialtm`. This follows `alignSpikes.m`: `trialtm_aligned = trialtm - ev.(alignEvent)(trial)`.

ii.
```python
al = tt - gocue[tr - 1]  # trialtm_aligned
```

iii. This is a direct port of the reference alignment code, with `alignEvent = 'goCue'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 10 ms bins (DT = 0.01), yielding 500 timepoints per trial over [-2.5, 2.5] s. No rebinning is applied — this is the native bin size used throughout.

ii.
```python
DT = 0.01  # params.dt = 1/100 s  -> 10 ms bins
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
```

iii. The AI cites `params.dt = 1/100` from the reference MATLAB code. The AI notes that 5 ms is mentioned in a tutorial comment and that the reference decoding analyses re-bin to 75 ms, but chose 10 ms as the canonical parameter value.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Not derived from raw data variables per se — it is the bin centers of the time axis, computed from the window parameters and bin size.

ii.
```python
EDGES, TAXIS = time_axis()
# TAXIS = edges + DT/2, from -2.495 to 2.495 s in 10 ms steps
inp = TAXIS.astype(np.float32)[None, :]
```

iii. The time axis is defined by the analysis parameters (alignment, window, bin size) and is the same for every trial and session.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Bin centers are computed as `edges + dt/2`, following `getSeq.m`'s `obj.time = edges + dt/2`. No other processing.

ii.
```python
def time_axis():
    edges = np.arange(TMIN, TMAX + DT / 2, DT)
    edges = edges[edges <= TMAX + 1e-9]
    t = edges + DT / 2
    return edges, t[:-1]
```

iii. Follows the reference's time axis definition.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input IS the time axis of the neural binning grid. Spikes are binned into `EDGES`, and the input is the center of each bin. They share the same time axis by construction.

ii.
```python
b = np.floor((al - TMIN) / DT).astype(np.int64)  # spike -> bin index
# ...
inp = TAXIS.astype(np.float32)[None, :]           # same time axis
```

iii. Both are defined by the same `EDGES`/`TAXIS` arrays.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Three per-trial fields: `bp.R` (right-instructed), `bp.L` (left-instructed), `bp.hit` (correct response), `bp.miss` (incorrect response), and `bp.no` (ignore/no response).

ii.
```python
hit, miss, no = obj.bp('hit'), obj.bp('miss'), obj.bp('no')
R, L = obj.bp('R'), obj.bp('L')
```

iii. The lick direction is not directly recorded but is derived from the combination of instructed side and outcome.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Hit + right-instructed = right lick; hit + left-instructed = left lick; miss + right-instructed = left lick (wrong side); miss + left-instructed = right lick (wrong side); no-response = none. Codes: left=0, right=1, none=2.

ii.
```python
lickdir = np.full(ntrials, 2, dtype=np.int8)                 # none
lickdir[((R > 0.5) & (hit > 0.5)) | ((L > 0.5) & (miss > 0.5))] = 1   # right
lickdir[((L > 0.5) & (hit > 0.5)) | ((R > 0.5) & (miss > 0.5))] = 0   # left
lickdir[no > 0.5] = 2
```

iii. This logic follows `getPrevChoice.m` which defines choice as `(R&hit) | (L&miss)` for right. The AI verified this against actual lickport contacts and found 100% agreement (within the response window).

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `bp.autowater` — a per-trial flag that is 1 for water-cued (WC) trials and 0 for delayed-response (DR) trials.

ii.
```python
autowater = obj.bp('autowater')
```

iii. The reference code uses `autowater` as the WC proxy throughout.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct relabelling: autowater > 0.5 → WC (0), otherwise → DR (1).

ii.
```python
context = np.where(autowater > 0.5, 0, 1).astype(np.int8)  # 0=WC, 1=DR
```

iii. Matches the decoder task specification (WC, DR).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Three per-trial flags: `bp.hit`, `bp.miss`, and `bp.no`. All three are used explicitly.

ii.
```python
hit, miss, no = obj.bp('hit'), obj.bp('miss'), obj.bp('no')
```

iii. The three flags are mutually exclusive and exhaustive.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct mapping: miss → incorrect (0), hit → correct (1), no → ignore (2).

ii.
```python
outcome = np.full(ntrials, -1, dtype=np.int8)
outcome[miss > 0.5] = 0   # incorrect
outcome[hit > 0.5] = 1    # correct
outcome[no > 0.5] = 2     # ignore
```

iii. Matches the decoder task specification. A runtime check ensures no trial is left with -1 (unlabelled).

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`, specifically the side camera (view 0), feature `'tongue'`. The x,y coordinates and frame times are extracted per trial. `bp.ev.goCue` and `sglx.bitcode.bitstart`/`sglx.fs`/`bp.ev.bitStart` are used for clock alignment.

ii.
```python
TONGUE_VIEW, TONGUE_FEAT = 0, 'tongue'
# ...
tongue_speed, tongue_vis = kinematic_speed(obj, TONGUE_VIEW, TONGUE_FEAT,
                                           gocue, ntrials, vidshift)
```

iii. The AI uses only the side camera for tongue, consistent with the reference `findPosition.m` which processes features from their native camera view.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. (1) DLC x,y positions are interpolated onto the neural time axis using `interp1` semantics (NaN outside data range). (2) Speed is computed as `sqrt(gradient(x)^2 + gradient(y)^2)` using a NaN-aware gradient. (3) Visibility is determined by whether both x and y are finite after interpolation. (4) The speed is split at the per-session 50th percentile of valid samples. (5) Invisible timepoints get class 2.

ii.
```python
x = interp_matlab(taxis, tv, xy[0])
y = interp_matlab(taxis, tv, xy[1])
vis = np.isfinite(x) & np.isfinite(y)
vx = nan_gradient(x)
vy = nan_gradient(y)
speed[t] = np.sqrt(vx ** 2 + vy ** 2)
visible[t] = vis & np.isfinite(speed[t])
# ...
tng_d, tng_thr = discretize(tongue_speed[keep], tongue_vis[keep])
```

iii. The interpolation approach follows `findPosition.m` which interpolates DLC traces onto the neural time axis. The velocity computation follows `findVelocity.m`. The discretization follows the decoder task specification.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per-session 50th percentile (median) of all valid (visible) tongue speed samples from retained trials. Below median = 0, at or above median = 1, not visible = 2.

ii.
```python
def discretize(values, valid, thresh=None):
    v = np.full(values.shape, 2, dtype=np.int8)
    if thresh is None:
        thresh = np.nanpercentile(values[valid], 50) if valid.any() else np.nan
    if valid.any():
        v[valid] = (values[valid] >= thresh).astype(np.int8)
    return v, thresh
```

iii. Follows the decoder task specification exactly (50th percentile, per-session).

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected from the video clock to the behavior/ephys clock using `findVideoOffset`: `vidshift = mode(sglx.bitcode.bitstart)/sglx.fs - mode(bp.ev.bitStart)`. Aligned frame time = `frameTimes - vidshift - goCue[trial]`. The DLC position is then interpolated onto the neural time axis (`TAXIS`), so the velocity shares the same time grid.

ii.
```python
def video_offset(obj):
    bitstart = obj.sglx_bitstart()
    fs = obj.sglx_fs()
    bp_bitstart = obj.bp('ev.bitStart')
    return (stats.mode(bitstart, keepdims=False).mode / fs
            - stats.mode(bp_bitstart, keepdims=False).mode)

tv = ft - vidshift - gocue[t]
x = interp_matlab(taxis, tv, xy[0])
```

iii. This is a direct port of `findVideoOffset.m` and `findPosition.m`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The DLC tracking from the bottom camera (view 1), feature `'top_paw'`.

ii.
```python
PAW_VIEW, PAW_FEAT = 1, 'top_paw'
paw_speed, paw_vis = kinematic_speed(obj, PAW_VIEW, PAW_FEAT,
                                     gocue, ntrials, vidshift)
```

iii. `top_paw` from the bottom camera is the feature used in the reference's Figure 1e (`top_paw_yvel_view2`).

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same pipeline as tongue velocity: interpolate x,y onto neural time axis, compute speed as magnitude of NaN-aware gradient, split at session median.

ii.
```python
paw_speed, paw_vis = kinematic_speed(obj, PAW_VIEW, PAW_FEAT, ...)
paw_d, paw_thr = discretize(paw_speed[keep], paw_vis[keep])
```

iii. Same `kinematic_speed` function is used for both tongue and paw.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: per-session 50th percentile of valid samples. 0 = below, 1 = at or above, 2 = not visible.

ii.
```python
paw_d, paw_thr = discretize(paw_speed[keep], paw_vis[keep])
```

iii. Follows the decoder task specification.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same alignment as tongue: video offset correction, go cue subtraction, interpolation onto the neural time axis.

ii.
```python
tv = ft - vidshift - gocue[t]
x = interp_matlab(taxis, tv, xy[0])
```

iii. Same `kinematic_speed` function handles alignment for all DLC features.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. A separate file `motionEnergy_<anm>_<date>.mat` containing per-trial traces at 400 Hz (one value per camera frame). Three layouts are handled (bare cell array, `me.data` cell, `me.data.data` struct).

ii.
```python
def load_motion_energy(path):
    d = sio.loadmat(path, struct_as_record=False, squeeze_me=False)
    me = d['me']
    # handles three layouts...
    return [np.asarray(x, float).ravel() for x in cells.ravel()]
```

iii. Mirrors `loadMotionEnergy.m` which handles the same layout variations (`if isstruct(me.data), me.data = me.data.data`).

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The raw motion energy trace (already one scalar per frame) is interpolated onto the neural time axis using the video-corrected frame times. No additional smoothing or spatial processing. Split at the session 50th percentile.

ii.
```python
def motion_energy_trace(obj, me_cells, gocue, ntrials, vidshift):
    for t in range(min(ntrials, len(me_cells))):
        ft = obj.frame_times(0, t)
        # fallback to 400 Hz nominal clock if needed
        out[t] = interp_matlab(taxis, ft - vidshift - gocue[t], m)
    return out
```

iii. The spatial reduction to one value per frame was done upstream by the paper's pipeline. The AI follows `loadMotionEnergy.m` for the interpolation.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session 50th percentile of valid (finite) motion energy samples. 0 = below, 1 = at or above, 2 = no video.

ii.
```python
me_d, me_thr = discretize(me[keep], me_vis[keep])
```

iii. Follows the decoder task specification.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same as tongue/paw: video offset + go cue subtraction on frame times, then interpolation onto the neural time axis. Motion energy uses the side camera frame times.

ii.
```python
ft = obj.frame_times(0, t)  # side camera
out[t] = interp_matlab(taxis, ft - vidshift - gocue[t], m)
```

iii. Same alignment pipeline as the DLC features, following `loadMotionEnergy.m`.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases: (1) Trials where `NdroppedFrames` is NaN are skipped entirely (no video data used) — following `findPosition.m`. (2) Where `frameTimes` is missing or not all finite, a fallback 400 Hz nominal clock is used. (3) DLC positions that are NaN (untracked) propagate through interpolation, resulting in NaN speed that becomes the "not visible" class. (4) Trials with no spikes at all are excluded via the no-ephys filter. (5) Motion energy files with fewer trials than `Ntrials` are handled by `min(ntrials, len(me_cells))`.

ii.
```python
if xy is None or not np.isfinite(obj.ndropped(view, t)):
    continue                                   # no usable video
if ft is None or not np.all(np.isfinite(ft)) or ft.size != xy.shape[1]:
    ft = np.arange(1, xy.shape[1] + 1) / 400.0 - 0.5 + vidshift  # fallback
```

iii. The handling follows the reference MATLAB code patterns (e.g., `findPosition.m` skips trials with `isnan(NdroppedFrames)`).

## 11-a. What are the most time-consuming steps of the code?

i. Video interpolation (DLC kinematics + motion energy) takes ~2.3 s per session, dominating the ~4 s per-session processing time. The full 44-session conversion runs in ~107 s. File loading and spike binning are secondary.

ii.
```python
info['timing'] = {k: round(v, 2) for k, v in timing.items()}
# timing breakdown: curation, neural, video, total
```

iii. The AI documented timing per stage and estimated total time before running the full conversion.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `kinematic_speed` and `motion_energy_trace` could not easily be vectorized because each trial has a different number of camera frames (ragged arrays). The per-unit spike binning uses `np.bincount` on a flattened index, which is already vectorized across trials.

ii.
```python
for t in range(ntrials):
    xy, ft = obj.traj_xy(view, t, featix)
    # ... per-trial interpolation
```

iii. The ragged frame counts prevent rectangular array operations. Since file I/O and video processing dominate, further vectorization would yield minimal improvement.

## 11-c. What processing does the code repeat multiple times?

i. The video offset is computed once per session. The DLC feature lookup (`featnames`, `featix`) is done once per feature per session. The causal FIR kernel is precomputed at module level. The smoothing is applied in one `lfilter` call over all units and trials simultaneously.

ii.
```python
_FIR = _causal_kernel()  # module-level precomputation
vidshift = video_offset(obj)  # once per session
sm = my_smooth(out.reshape(NT, -1))  # all units+trials in one call
```

iii. No redundant recomputation was identified.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads and processes probe location metadata (`ex.probe.loc`) for brain region labeling. It also computes detailed per-session statistics (fractions, thresholds, quality counts) that are stored in `metadata.session_info` but not used by the decoder. The diagnostic plotting code (`_plot_processing`) is only run on request.

ii.
```python
unit_region.append(region)  # ALM or tjM1 from ex.probe.loc
info = dict(animal=anm, ..., frac_context_WC=..., frac_outcome=...)
```

iii. The extra metadata is useful for documentation and debugging but is not consumed by the decoder.
