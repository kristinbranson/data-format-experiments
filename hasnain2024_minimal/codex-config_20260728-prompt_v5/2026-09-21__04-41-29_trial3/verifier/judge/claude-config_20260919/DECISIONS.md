# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a list of 12 sessions (subject, date, probe) and reads each one directly from `/app/data/Ephys_Behavior/data_structure_<anm>_<date>.mat` with `h5py`, using a hand-written recursive MATLAB-v7.3 reader (`read_matlab_any`, `matlab_class`, `decode_char`, `make_object_array`). Motion energy is read from `motionEnergy_<anm>_<date>.mat` with `scipy.io.loadmat(..., simplify_cells=True)`. Only the `Ephys_Behavior` folder is ever touched: the 19 sessions in `RandomizedDelay_Ephys_Behavior`, and the `JEB13`/`JEB14`/`JEB15` sessions inside `Ephys_Behavior`, are not loaded. There is no v5/`scipy.io` fallback for the data-structure files and no glob of the data directories. Each session's data-structure file is opened twice (once in `convert_session` for trials/spikes/behaviour, once again in `compute_video_outputs` for the video streams).

ii.
```python
DATA_DIR = Path("/app/data/Ephys_Behavior")

# Published DR+WC ALM-video cohort from the paper code:
# JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, JEB19 (12 sessions total).
SESSIONS = [
    {"subject": "JEB6", "date": "2021-04-18", "probe": 2},
    ...
    {"subject": "JEB19", "date": "2023-04-18", "probe": 1},
]
```

```python
def read_matlab_any(h5file, obj):
    if isinstance(obj, h5py.Reference):
        if not obj:
            return None
        obj = h5file[obj]
    cls = matlab_class(obj)
    if isinstance(obj, h5py.Group):
        return {key: read_matlab_any(h5file, obj[key]) for key in obj.keys()}
    arr = obj[()]
    if cls == "char":
        return decode_char(obj)
    ...
```

```python
    data_path = DATA_DIR / f"data_structure_{subject}_{date}.mat"
    with h5py.File(data_path, "r") as h5file:
        ...
    me_raw = loadmat(me_path, simplify_cells=True)["me"]["data"]
```

iii. The AI traced session inclusion through the authors' own analysis scripts and settled on the cohort used by `code/Scripts/Figure 8/Figure8d.m`, which calls exactly `loadJEB6/JEB7/EKH1/EKH3/JGR2/JGR3/JEB19_ALMVideo` (12 sessions). It stated: *"I've pinned down the cohort split from the authors' own scripts: the first 12 `Ephys_Behavior` ALM-video sessions are the DR+WC recordings, and later sessions are the DR-only cohort."* It supported this by printing per-session autowater counts (step 56), which show `JEB13_2022-09-24/25` and `JEB14_2022-08-22` with 0 WC trials and `JEB13_2022-09-21`/`JEB14_2022-08-23` with 3/4 — i.e. essentially DR-only. It did read all 14 `load<ANM>_ALMVideo.m` files (including JEB11, JEB12, JEB23, JEB24 for the randomized-delay task and JEB15, which has 22–45 WC trials per session) but gave no explicit reason for excluding them beyond "not the published DR+WC cohort". The HDF5 reader was written after discovering that `scipy.io.loadmat` fails on these v7.3 files: *"the files aren't readable with `scipy.io.loadmat`, so I'm building against the raw HDF5 layout directly."*

## 1-b. How are the data split into subjects (mice)?

i. The subject is a literal field of each hard-coded session entry (`{"subject": "JEB6", ...}`), equivalent to the animal prefix of the filename. `subjects` is built in first-encounter order while iterating `SESSIONS`, and `subject_idx` is the index of each session's subject into that list. The result is 7 subjects over 12 sessions (JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, JEB19).

ii.
```python
    for session_info in SESSIONS:
        session_data, session_meta = convert_session(session_info)
        subject = str(session_meta["subject"])
        if subject not in subject_to_idx:
            subject_to_idx[subject] = len(subjects)
            subjects.append(subject)
        ...
        all_subject_idx.append(subject_to_idx[subject])
```
```python
        "subjects": subjects,
        "subject_idx": np.asarray(all_subject_idx, dtype=np.int64),
```

iii. Not discussed explicitly. The animal identity comes from the authors' `meta.anm` field as transcribed from the `load<ANM>_ALMVideo.m` scripts, which is also how filenames are built; the AI never relies on `obj.meta.anm` inside the files.

## 1-c. How are the data split into sessions?

i. One entry of `SESSIONS` = one `data_structure_*.mat` file = one element of `neural`/`input`/`output`/`brain_region_idx`, in the hard-coded order. The folder is fixed to `Ephys_Behavior`, so only the fixed-delay task is represented; the randomized-delay sessions are never assigned a session slot. A single probe number is stored per session and only that probe's clusters are used (no two-probe session is in the cohort, so probe concatenation is never needed or implemented). A session is rejected if fewer than 2 trials survive filtering or fewer than 10 units survive QC.

ii.
```python
def convert_session(session_info):
    subject = str(session_info["subject"]); date = str(session_info["date"])
    probe_num = int(session_info["probe"])
    data_path = DATA_DIR / f"data_structure_{subject}_{date}.mat"
```
```python
        if keep_trials.size < 2:
            raise RuntimeError(f"{subject} {date} has fewer than two usable trials.")
```
```python
    if len(kept_unit_data) < 10:
        raise RuntimeError(f"Session failed unit inclusion: only {len(kept_unit_data)} units after filtering.")
```

iii. Same justification as 1-a: the session list is the Figure 8d cohort, and the probe number per session is copied from the authors' `load<ANM>_ALMVideo.m` entries.

## 1-d. How are the data split into trials?

i. A trial is one index into the per-trial fields of `obj.bp`. The AI builds a boolean keep mask over the full length of `bp.early` / `bp.stim.enable` / `bp.autolearn` and takes `keep_trials = np.flatnonzero(keep_mask)`; those original trial numbers index `bp.ev.goCue`, all behavioural flags, the per-trial `traj` cells and the per-trial motion-energy cells. Spikes carry their own 1-based `clu.trial`, which is converted to 0-based and mapped through a lookup table onto the kept-trial axis. The arrays are not truncated to `bp.Ntrials` (the reference does), but for all 12 sessions in the cohort the field lengths equal `Ntrials`, so no mismatch occurs.

ii.
```python
        keep_mask = (~stim_enable) & (~early) & (~autolearn)
        keep_trials = np.flatnonzero(keep_mask)
```
```python
    trial_lookup = np.full(go_cue.size, -1, dtype=np.int32)
    trial_lookup[keep_trials] = np.arange(keep_trials.size, dtype=np.int32)
    ...
        clu_trials = as_1d_float(read_matlab_any(h5file, trials[clu_idx, 0])).astype(np.int64) - 1
        mapped_trials = trial_lookup[clu_trials]
```

iii. Not discussed explicitly. The Bpod table defines trials directly and each trial has exactly one `bp.ev.goCue`, so no trial boundary has to be inferred; this mirrors `findTrials.m`, which also evaluates conditions as boolean masks over `obj.bp` fields.

## 1-e. How are trials filtered based on quality controls?

i. Three per-trial exclusions, all applied up front: photostimulation trials (`bp.stim.enable`), early-lick trials (`bp.early`), and — where the field exists — `bp.autolearn` trials. `autolearn` is present only in the four JEB19 sessions and removes 15, 18, 12 and 15 extra trials there; for the other eight sessions the kept-trial counts are identical to the expert's. 3,056 of 3,117 candidate trials (3,617 raw trials before filtering, see `n_trials_total`) are kept. No trial is dropped for running past the end of the ephys recording, and ignore (`no`) trials are kept.

ii.
```python
        stim_enable = as_bool_1d(bp["stim/enable"])
        early = as_bool_1d(bp["early"])
        autolearn = (
            as_bool_1d(bp["autolearn"]) if "autolearn" in bp.keys() else np.zeros_like(early, dtype=bool)
        )
        keep_mask = (~stim_enable) & (~early) & (~autolearn)
```
```python
        "trial_filter": "~stim.enable & ~early & ~autolearn",
```

iii. The AI's plan step says: *"Build decoder inputs/outputs from the filtered control trials (`~stim`, `~early`)"*, which follows every `params.condition` string in the authors' scripts (`'hit&~stim.enable&~autowater&~early'` etc.). The `~autolearn` term is added silently — it appears in the code and in the metadata string but the AI never states a reason for it in its messages. Ignore trials are deliberately kept because the decoder task asks for an `ignore` outcome class and a `none` lick class.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` for the one probe listed for that session: the per-cluster cell arrays `quality`, `trial` (1-based trial of each spike) and `trialtm` (spike time relative to that trial's start). `obj.bp.ev.goCue` supplies the alignment time.

ii.
```python
    clu_group = h5file[h5file["obj/clu"][probe_num - 1, 0]]
    qualities = clu_group["quality"]
    trials = clu_group["trial"]
    trial_times = clu_group["trialtm"]
    ...
        clu_trials = as_1d_float(read_matlab_any(h5file, trials[clu_idx, 0])).astype(np.int64) - 1
        clu_trial_times = as_1d_float(read_matlab_any(h5file, trial_times[clu_idx, 0]))
        aligned_times = clu_trial_times - go_cue[clu_trials]
```

iii. Taken from `alignSpikes.m` (`obj.clu{prb}(clu).trialtm_aligned = obj.clu{prb}(clu).trialtm - event`) and `getSeq.m`, which histograms `trialtm_aligned` per trial. The probe index comes from the authors' meta entries.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 5 ms bins per (trial, bin) with `np.add.at`, divided by the bin width to give spikes/s, then smoothed along time with a **causal** Gaussian kernel of 15 taps whose first `floor(15/2) = 7` taps are zeroed and which is renormalised to sum to 1 — i.e. an 8-tap causal filter spanning 0–35 ms of past activity. Boundary handling copies the first 15 samples of each trial in front of the signal and trims them afterwards. There is no normalisation, baseline subtraction or z-scoring; stored values are firing rates in Hz as `float32`. The kernel's width is set from `sigma = np.std(1..15) ≈ 4.32` bins (≈21.6 ms), which is the `kernsd` value *returned* by `mySmooth.m`, not the width of MATLAB's `gausswin(15)` (σ = 2.8 bins ≈ 14 ms); the AI's kernel therefore has the same 8-tap support but a flatter weight profile than the authors' (weights 0.183…0.049 vs 0.251…0.011).

ii.
```python
def build_smoothing_kernel(n: int = SMOOTH_N) -> np.ndarray:
    x = np.arange(1, n + 1, dtype=np.float64)
    sigma = np.std(x)
    kernel = np.exp(-0.5 * ((x - (n + 1) / 2.0) / sigma) ** 2)
    kernel[: n // 2] = 0.0
    kernel /= kernel.sum()
    return kernel

def smooth_causal_reflect(x, kernel=SMOOTH_KERNEL):
    n = kernel.size
    padded = np.concatenate([x[:, :n], x], axis=1)
    out = np.empty_like(padded)
    for i in range(padded.shape[0]):
        out[i] = np.convolve(padded[i], kernel, mode="same")
    out = out[:, n:]
```
```python
        bin_idx = np.floor((aligned_times - TMIN) / DT).astype(np.int32)
        counts = np.zeros((keep_trials.size, NT), dtype=np.float64)
        if mapped_trials.size:
            np.add.at(counts, (mapped_trials, bin_idx), 1.0)
        rates = smooth_causal_reflect(counts / DT).astype(np.float32)
```

iii. Directly modelled on `utils/mySmooth.m`, which the AI read (step 70): `kern = gausswin(N); kernsd = std(1:N); kern(1:floor(numel(kern)/2)) = 0; %causal; kern = kern./sum(kern); conv(...,'same')`, with the `'reflect'` boundary branch `x_filt = cat(1,x(1:N,:),x); trim = N+1`. `getSeq.m` divides counts by `params.dt` before smoothing, which the AI replicates. The AI describes it in the metadata as *"causal Gaussian smoothing (N=15) with reflected leading padding"* and noted that `Figure8d.m` sets `params.bctype = 'reflect'`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two unit filters plus one session filter. (1) Cluster quality: the label is stripped of whitespace and dropped if it exactly equals `garbage`, `gabrga`, `noisy` or `real?` — case-sensitively, so `Poor`, `Fair`, `Good`, `Multi` are all kept. (2) After smoothing, a unit is kept only if its mean rate over all kept trials and all 1100 bins exceeds 1 Hz. (3) A session with fewer than 10 surviving units raises an error (never triggered). This keeps 519 units over the 12 sessions (27–67 per session), against 364 for the same 12 sessions in the expert solution; the whole difference is `Poor`-labelled clusters (e.g. EKH1 has 23 `Poor`, 12 `Fair`, 7 `Good`, 6 `Multi`; the AI keeps 48, the expert 25).

ii.
```python
        quality = str(read_matlab_any(h5file, qualities[clu_idx, 0]) or "").strip()
        if quality in {"garbage", "gabrga", "noisy", "real?"}:
            continue
```
```python
        mean_fr = float(rates.mean())
        if mean_fr > LOW_FR_HZ:      # LOW_FR_HZ = 1.0
            kept_unit_data.append(rates)
```

iii. The quality drop list is copied verbatim from `findClusters.m`'s `params.quality = {'all'}` branch, which the AI read at step 71: `~ismember(qualityList,'garbage') & ~ismember(qualityList,'gabrga') & ~ismember(qualityList,'noisy') & ~ismember(qualityList,'real?')` after `strtrim` — including MATLAB's case sensitivity. The 1 Hz threshold comes from `Figure8d.m` (`params.lowFR = 1`, versus 0.5 in `getDefaultParams.m`) and from the paper's statement that units above 1 Hz were included; `removeLowFRClusters.m` likewise tests `meanFRs > lowFR`. The AI summarises this as *"the paper's unit filter (`FR > 1 Hz`, all curated qualities)"*.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. One subtraction: each spike's `trialtm` (already relative to its own trial's start, on the behaviour clock) minus `goCue` of that same trial. Spikes falling outside `[-3.0, 2.5)` s are discarded, and the remainder are binned by `floor((t − TMIN)/DT)`. No interpolation or per-session clock correction is applied to spikes (the video streams get one; see 7-d).

ii.
```python
        aligned_times = clu_trial_times - go_cue[clu_trials]
        mapped_trials = trial_lookup[clu_trials]
        valid = (mapped_trials >= 0) & (aligned_times >= TMIN) & (aligned_times < TMAX)
        mapped_trials = mapped_trials[valid]; aligned_times = aligned_times[valid]
        bin_idx = np.floor((aligned_times - TMIN) / DT).astype(np.int32)
```

iii. This is `alignSpikes.m` with `params.alignEvent = 'goCue'`, as read at step 15, combined with the `histc(trialtm_aligned, edges)` of `getSeq.m`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 5 ms bins (`DT = 1/200`), 1100 bins spanning −3.0 to +2.5 s relative to the go cue, identical for every trial and every session and shared by the neural, input and all three video output streams. No rebinning or resampling of the binned data is done afterwards: the spikes are histogrammed straight onto this grid and the video streams are interpolated straight onto its bin centres. In the metadata, `time_bin_size` is written as `0.005` (seconds) with a separate `time_bin_size_ms: 5.0`, although the target format specifies `time_bin_size` in ms.

ii.
```python
TMIN = -3.0
TMAX = 2.5
DT = 1.0 / 200.0
...
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_BINS = TIME_EDGES[:-1] + DT / 2.0
NT = TIME_BINS.size          # 1100
```
```python
            "time_bin_size": float(DT),
            "time_bin_size_ms": float(DT * 1000.0),
            "off_start": float(TMIN),
            "off_end": float(TMAX),
```

iii. `params.dt = 1/200` is the bin size in both `getDefaultParams.m` and every figure script. The window comes from `Figure8d.m` specifically (`params.tmin = -3; params.tmax = 2.5;`) rather than from `getDefaultParams.m` (`-2.5`/`2.5`), consistent with the AI's choice of the Figure 8d cohort. The centre convention (`edges + dt/2`, drop last) is copied from `getSeq.m`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Nothing from the raw files — it is the bin-centre vector of the analysis window defined by the AI (`TMIN`, `TMAX`, `DT`), which is implicitly the go-cue time of each trial because the whole window is defined relative to `bp.ev.goCue`. The same `TIME_BINS` vector is used for every trial and session.

ii.
```python
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_BINS = TIME_EDGES[:-1] + DT / 2.0
...
INPUT_NAMES = ["time_from_go_cue"]
```

iii. Defined by the decoder-task specification ("Time from go cue onset in seconds, continuous, time-varying"); the window itself is the authors' `params.tmin`/`params.tmax` from `Figure8d.m`.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond building the bin-centre vector once at module import and casting it to `float32` per trial. It is not recomputed per trial or per session.

ii.
```python
    input_trials = [TIME_BINS[np.newaxis, :].astype(np.float32) for _ in range(keep_trials.size)]
```

iii. N/A — the axis is defined by the analyst, so there is nothing to derive.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid. Spikes are assigned to bin `floor((t_spike − goCue − TMIN)/DT)` of the same `TIME_EDGES`, and the input value in bin *k* is the centre of that same interval, so bin *k* means the same interval in the input, the neural data and all outputs. Verified in the output summary: the input spans −2.9975 to 2.4975 s, i.e. the first and last bin centres.

ii.
```python
        bin_idx = np.floor((aligned_times - TMIN) / DT).astype(np.int32)
```
```python
TIME_BINS = TIME_EDGES[:-1] + DT / 2.0
```

iii. N/A.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial flags of `obj.bp`: the instructed side `R` and `L`, and the outcome flags `hit` and `miss`. (`bp.no` is also read but not used for this output.) The realised lick side is not stored in the files, so it is inferred from instruction × outcome.

ii.
```python
    r = as_bool_1d(bp["R"]); l = as_bool_1d(bp["L"])
    hit = as_bool_1d(bp["hit"]); miss = as_bool_1d(bp["miss"]); no = as_bool_1d(bp["no"])
```

iii. The AI flagged this explicitly as the main behavioural ambiguity: *"the files encode `R/L`, `hit/miss/no`, `autowater`, and `early`, and I need to confirm whether `R/L` is trial instruction or actual response. That matters for the requested 'lick direction' output."* It then checked the authors' usage (`getPrevChoice.m`, `plotLickRaster.m`, the `'R&hit'`/`'L&miss'` condition strings) and the raw flag combinations before concluding that `R`/`L` is the instructed side.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A hit means the animal licked the instructed port; a miss means it licked the other one; anything else (an ignore trial) is `none`. Codes: left 0, right 1, none 2 (the default fill). The value is constant within a trial and broadcast across all 1100 bins.

ii.
```python
    lick_direction = np.full(keep_trials.size, 2, dtype=np.int64)
    right_choice = (r & hit) | (l & miss)
    left_choice = (l & hit) | (r & miss)
    lick_direction[left_choice[keep_trials]] = 0
    lick_direction[right_choice[keep_trials]] = 1
```
```python
OUTPUT_VALUES = [["left", "right", "none"], ...]
```

iii. Follows from the AI's check that `R`/`L` are the instructed side: *"deriving actual lick direction from `R/L` with `hit/miss/no`"*. The third class is required by the decoder task's `(left, right, none)` specification.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial flag, `obj.bp.autowater`, which marks trials where water was delivered from a random port with no cue — the water-cued (WC) context.

ii.
```python
    autowater = as_bool_1d(bp["autowater"])
```

iii. Every context-related condition string in the authors' code uses `autowater` as the WC marker, and `Figure8d.m` states it outright: *"Throughout whole script, where it says '2afc' or 'afc' this means 'DR'; 'aw' = 'WC'"*. The AI also verified per-session autowater counts (step 56) to confirm both contexts occur in the chosen sessions.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling: `autowater` → WC = 0, otherwise DR = 1; constant per trial, broadcast across bins.

ii.
```python
    context = np.where(autowater[keep_trials], 0, 1).astype(np.int64)  # WC, DR
```
```python
OUTPUT_VALUES = [..., ["WC", "DR"], ...]
```

iii. The code order (WC 0, DR 1) follows the decoder task's `(WC, DR)` listing.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Two per-trial flags of `obj.bp`: `hit` and `miss`. `bp.no` is read into a local variable but never used — a trial that is neither a hit nor a miss falls into the `ignore` default.

ii.
```python
    hit = as_bool_1d(bp["hit"]); miss = as_bool_1d(bp["miss"]); no = as_bool_1d(bp["no"])
```

iii. Same two flags already needed for lick direction; the authors' condition strings use `hit`, `miss` and `(hit|miss|no)` interchangeably, and `~hit&~miss` is their own definition of an ignore trial (`params.condition` in `getDefaultParams.m`).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Relabelling into three classes: incorrect 0 on miss trials, correct 1 on hits, ignore 2 as the default; constant per trial, broadcast across bins.

ii.
```python
    outcome = np.full(keep_trials.size, 2, dtype=np.int64)
    outcome[miss[keep_trials]] = 0
    outcome[hit[keep_trials]] = 1
```
```python
OUTPUT_VALUES = [..., ["incorrect", "correct", "ignore"], ...]
```

iii. The code order follows the decoder task's `(incorrect, correct, ignore)`. Ignore trials are retained rather than dropped (the paper omits them from most analyses) so that the third class exists and so that those trials still contribute to the other five outputs.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`, both cameras. From the side view (`traj{1}`) it takes `tongue`, `left_tongue`, `right_tongue`; from the bottom view (`traj{2}`) `top_tongue`, `topleft_tongue`, `bottom_tongue`, `bottomleft_tongue` — i.e. every tongue keypoint in the authors' `params.traj_features`. Per trial and view it reads `featNames`, `ts` (frames × [x, y, likelihood] × features after transposing the h5py layout) and `frameTimes`. `obj.bp.ev.goCue`, `obj.bp.ev.bitStart`, `obj.sglx.bitcode.bitstart` and `obj.sglx.fs` are also needed for the clock correction. Features absent from a view are silently skipped.

ii.
```python
            side_tongue_speed, side_tongue_visible, _ = get_feature_positions(
                h5file, side_view, int(trial_idx),
                ["tongue", "left_tongue", "right_tongue"], align_time, video_shift)
            bottom_tongue_speed, bottom_tongue_visible, _ = get_feature_positions(
                h5file, bottom_view, int(trial_idx),
                ["top_tongue", "topleft_tongue", "bottom_tongue", "bottomleft_tongue"],
                align_time, video_shift)
```
```python
    ts = np.asarray(read_matlab_any(h5file, view_group["ts"][trial_idx, 0]), dtype=np.float64)
    # MATLAB arrays come through as feat x coord x time in h5py.
    ts = np.transpose(ts, (2, 1, 0))
    ...
        xy = ts[:, :2, feature_idx]
```

iii. The feature lists are copied from `params.traj_features` in `getDefaultParams.m` (`{{'tongue','left_tongue','right_tongue',...},{'top_tongue','topleft_tongue','bottom_tongue','bottomleft_tongue',...}}`), and the per-view indexing follows `findDLCFeatIndex.m` / `findPosition.m` (`traj(trix).ts(:,1:2,featix)`). Using both cameras is implicit in the authors' two-view feature specification.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps per trial and feature. (1) The frames' x and y are linearly interpolated onto the 1100 bin centres, but only inside each contiguous run of frames where both x and y are finite (a run of one frame is placed at its nearest bin); bins outside any run stay NaN, and the `visible` mask for that feature is exactly "interpolation produced a finite x and y here". (2) The leading and trailing NaN stretches of the interpolated position are filled with the nearest finite value (`fill_nearest_1d`); interior gaps are left NaN. No smoothing is applied to the positions. (3) Speed is `hypot` of `np.gradient(x)` and `np.gradient(y)` along the binned time axis — so the units are pixels per 5 ms bin, not pixels/s, and no baseline-derivative subtraction is done for tongue features. (4) The speeds of all tongue features are averaged over just those features marked visible in that bin, first within a view and then across the two views, with no per-camera scale normalisation; a bin is visible if any tongue keypoint in any view was visible. There is no explicit likelihood cut — the AI relies on the authors having already set x/y to NaN for untracked frames.

ii.
```python
def interpolate_visible_segments(src_t, src_xy, target_t):
    out = np.full((target_t.size, src_xy.shape[1]), np.nan, dtype=np.float64)
    valid = np.all(np.isfinite(src_xy), axis=1) & np.isfinite(src_t)
    ...
    for grp in groups:
        if grp.size == 1:
            nearest = int(np.argmin(np.abs(target_t - src_t[grp[0]])))
            out[nearest] = src_xy[grp[0]]; continue
        seg_t = src_t[grp]
        seg_mask = (target_t >= seg_t[0]) & (target_t <= seg_t[-1])
        for dim in range(src_xy.shape[1]):
            out[seg_mask, dim] = np.interp(target_t[seg_mask], seg_t, src_xy[grp, dim])
    return out
```
```python
        interp_xy = interpolate_visible_segments(frame_times, xy, TIME_BINS)
        visible = np.all(np.isfinite(interp_xy), axis=1)
        filled_xy = interp_xy.copy()
        filled_xy[:, 0] = fill_nearest_1d(filled_xy[:, 0])
        filled_xy[:, 1] = fill_nearest_1d(filled_xy[:, 1])
        ...
            vel = np.gradient(filled_xy, axis=0)
            baseline_deriv = np.nanmedian(np.diff(filled_xy, axis=0), axis=0)
            if "tongue" not in feature_name:
                vel[:, 0] = vel[:, 0] - baseline_deriv[0]
                vel[:, 1] = vel[:, 1] - baseline_deriv[0]
            speed = np.sqrt((vel**2).sum(axis=1))
```
```python
def average_over_visible(values, visible):
    any_visible = visible.any(axis=0)
    mean_values = np.full(values.shape[1], np.nan, dtype=np.float64)
    if np.any(any_visible):
        counts = visible[:, any_visible].sum(axis=0)
        summed = np.where(visible[:, any_visible], values[:, any_visible], 0.0).sum(axis=0)
        mean_values[any_visible] = summed / counts
    return mean_values, any_visible
```

iii. Steps 1–3 are a faithful port of `findPosition.m` and `findVelocity.m`: `interp1(traj(trix).frameTimes - vidshift - obj.bp.ev.(alignEv)(trix), ts, taxis)`, `if ~contains(feat,'tongue') ts = mySmooth(ts,1,'reflect')` (a no-op at N=1, hence no smoothing anywhere), `xvel = gradient(tsinterp(:,1))` with no `dt`, the `basederiv` subtraction skipped for tongue, and `fillmissing(...,'nearest')` for non-tongue. The AI's metadata describes it as *"feature positions linearly interpolated onto the neural grid, visibility preserved for tongue/paw categorization"*. Aggregating the keypoints into one scalar speed is the AI's own step (the authors keep every feature as a separate regressor); the AI gives no discussion of the two cameras being on different pixel scales.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. One threshold per session: the 50th percentile of the combined tongue speed over all kept trials × bins that were marked visible. Bins below it get 0, bins at or above it get 1, and every bin not marked visible gets 2 (`not_visible`, the default fill). In the delivered dataset the classes come out at 4.06 % / 4.06 % / 91.9 %.

ii.
```python
    tongue_threshold = float(np.nanpercentile(tongue_speed[tongue_visible], 50))
    ...
    tongue_disc = np.full((keep_trials.size, NT), 2, dtype=np.int64)
    tongue_disc[tongue_visible & (tongue_speed < tongue_threshold)] = 0
    tongue_disc[tongue_visible & (tongue_speed >= tongue_threshold)] = 1
```

iii. Straight from the decoder task specification (0: < 50th percentile, 1: >= 50th percentile, 2: not visible, per-session threshold). The thresholds are recorded per session in `metadata.session_info[...]['thresholds']`.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The camera clock leads the behaviour clock, so a session-constant offset is computed once from the bitcode pulse — `median(sglx.bitcode.bitstart / sglx.fs) − median(bp.ev.bitStart)` — and each trial's frame times become `frameTimes − video_shift − goCue[trial]`. Those corrected times are the `src_t` of the interpolation onto the shared 1100-bin grid, so tongue bins and spike bins are the same intervals. (The AI uses the median where the authors use the mode; on these sessions the two are numerically identical, e.g. 0.49004 s for EKH1 and 0.99002 s for JEB19_2023-04-19.)

ii.
```python
def get_video_shift(h5file: h5py.File, bit_start: np.ndarray) -> float:
    bitcode_start = as_1d_float(h5file["obj/sglx/bitcode/bitstart"])
    fs = float(np.asarray(h5file["obj/sglx/fs"]).squeeze())
    return float(np.nanmedian(bitcode_start / fs) - np.nanmedian(bit_start))
```
```python
    frame_times = as_1d_float(read_matlab_any(h5file, view_group["frameTimes"][trial_idx, 0]))
    frame_times = frame_times - video_shift - align_time
    ...
        interp_xy = interpolate_visible_segments(frame_times, xy, TIME_BINS)
```

iii. This is `findVideoOffset.m` — `vidshift = mode(obj.sglx.bitcode.bitstart)/obj.sglx.fs - mode(obj.bp.ev.bitStart)` — which the AI read at step 53, combined with `findPosition.m`'s `interp1(traj.frameTimes - vidshift - obj.bp.ev.(alignEv)(trix), ts, taxis)`. The AI describes it as *"video trajectories aligned with the paper's video offset"*.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The bottom camera's (`traj{2}`) DeepLabCut tracking of `top_paw` **and** `bottom_paw`, plus the same `frameTimes`, `goCue` and bitcode fields used for the tongue.

ii.
```python
            paw_speed_trial, paw_visible_trial, _ = get_feature_positions(
                h5file, bottom_view, int(trial_idx),
                ["top_paw", "bottom_paw"], align_time, video_shift)
```

iii. Both paw keypoints appear in the authors' `params.traj_features` bottom-view list, and both are on the same camera, so the AI treats them as two samples of the same behaviour. No explicit discussion of tracking reliability appears in the trajectory.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Identical pipeline to the tongue (interpolate visible runs onto the 5 ms grid → fill leading/trailing NaNs with the nearest value → `hypot` of the two `np.gradient`s → average over the keypoints visible in each bin), with one addition: because the feature name does not contain `tongue`, the per-trial median frame-to-frame displacement (`baseline_deriv`) is subtracted from the velocity before the magnitude is taken. As in the authors' MATLAB, `baseline_deriv[0]` (the x component) is subtracted from *both* the x and the y velocity. No cross-view normalisation is needed (one camera only), so values stay in pixels per bin.

ii.
```python
            vel = np.gradient(filled_xy, axis=0)
            baseline_deriv = np.nanmedian(np.diff(filled_xy, axis=0), axis=0)
            if "tongue" not in feature_name:
                vel[:, 0] = vel[:, 0] - baseline_deriv[0]
                vel[:, 1] = vel[:, 1] - baseline_deriv[0]
            speed = np.sqrt((vel**2).sum(axis=1))
```

iii. A line-for-line port of `funcs/kinematics/findVelocity.m`, which the AI read at step 52:
```matlab
basederiv = median(diff(tsinterp),'omitnan');
xvel(:,i) = gradient(tsinterp(:,1));  yvel(:,i) = gradient(tsinterp(:,2));
if ~contains(feat,'tongue')
    xvel(:,i) = xvel(:,i) - basederiv(1);
    yvel(:,i) = yvel(:,i) - basederiv(1);
end
```
including the authors' use of `basederiv(1)` for the y velocity. The `fillmissing(...,'nearest')` for non-tongue features is likewise from `findPosition.m`/`findVelocity.m`.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Per-session 50th percentile of the combined paw speed over all visible (trial, bin) entries; below → 0, at/above → 1, not visible → 2. Delivered class fractions: 44.7 % / 44.7 % / 10.6 %.

ii.
```python
    paw_threshold = float(np.nanpercentile(paw_speed[paw_visible], 50))
    paw_disc = np.full((keep_trials.size, NT), 2, dtype=np.int64)
    paw_disc[paw_visible & (paw_speed < paw_threshold)] = 0
    paw_disc[paw_visible & (paw_speed >= paw_threshold)] = 1
```

iii. Directly from the decoder task specification.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Exactly as the tongue: the session's bitcode-derived `video_shift` and the trial's `goCue` are subtracted from the bottom camera's `frameTimes`, and the positions are interpolated onto the same 1100-bin grid the spikes are histogrammed into. The paw uses the bottom view's own `frameTimes`, read from that view's `traj` entry.

ii.
```python
    frame_times = as_1d_float(read_matlab_any(h5file, view_group["frameTimes"][trial_idx, 0]))
    frame_times = frame_times - video_shift - align_time
```

iii. Same `findVideoOffset.m` + `findPosition.m` logic as 7-d; no separate treatment is needed because both views are corrected by the same session-wide offset.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file beside each data structure, read as `me['data']` — one cell per trial holding one value per camera frame. The side camera's `frameTimes` (and, as a fallback, the side camera's `ts` frame count) supply the time base. `obj.me`, where present, is not used.

ii.
```python
    me_path = DATA_DIR / f"motionEnergy_{subject}_{date}.mat"
    ...
        me_raw = loadmat(me_path, simplify_cells=True)["me"]["data"]
        ...
            me_trial = np.asarray(me_raw[trial_idx], dtype=np.float64).reshape(-1)
```

iii. From `loadMotionEnergy.m`, which loads `motionEnergy*<date>.mat` from disk and uses `me.data` as a per-trial cell array at 400 Hz. The AI probed the file format at steps 43–46 and found it to be a v5 MATLAB file, hence the `scipy.io.loadmat` path rather than `h5py`. It does not implement the `if isstruct(me.data), me.data = me.data.data; end` double-unwrap guard, which is not needed for any of the 12 files in its cohort.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond resampling: the value is already one scalar per frame. The trace is linearly interpolated from the corrected frame times onto the 1100 bin centres with NaN outside the frame range, and then the leading/trailing NaNs are filled with the nearest finite value, so nearly every bin of nearly every trial ends up defined. A bin is `available` iff the interpolated value is finite; trials whose frame times and motion-energy vector cannot be reconciled at all stay entirely unavailable. In the delivered data only 0.03 % of bins fall in the `no_video` class.

ii.
```python
def interp_motion(src_t, src_y, target_t):
    valid = np.isfinite(src_t) & np.isfinite(src_y)
    if valid.sum() < 2:
        return np.full(target_t.shape, np.nan, dtype=np.float64)
    out = np.interp(target_t, src_t[valid], src_y[valid], left=np.nan, right=np.nan)
    return fill_nearest_1d(out)
```
```python
            if frame_times.size == me_trial.size and frame_times.size >= 2:
                aligned_me = interp_motion(frame_times, me_trial, TIME_BINS)
                available = np.isfinite(aligned_me)
                motion_energy[out_trial_idx] = aligned_me
                motion_available[out_trial_idx] = available
```

iii. A port of `loadMotionEnergy.m`: `me.newdata(:,trix) = interp1(obj.traj{1}(trix).frameTimes - vidshift - alignTimes(trix), me.data{trix}, taxis)` followed by `me.data = fillmissing(me.data,'nearest')` — the AI read this at step 19 and notes in the metadata that *"motion energy [is] aligned as in the reference code"*. The spatial reduction (99th percentile across pixels) was already done upstream by the authors.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session 50th percentile over all available (trial, bin) entries; below → 0, at/above → 1, unavailable → 2 (`no_video`). Delivered fractions: 49.9 % / 50.0 % / 0.03 %. The authors' own `me.moveThresh` is not used.

ii.
```python
    me_threshold = float(np.nanpercentile(motion_energy[motion_available], 50))
    me_disc = np.full((keep_trials.size, NT), 2, dtype=np.int64)
    me_disc[motion_available & (motion_energy < me_threshold)] = 0
    me_disc[motion_available & (motion_energy >= me_threshold)] = 1
```

iii. Directly from the decoder task specification, which prescribes a per-session 50th-percentile split rather than the paper's `moveThresh`.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same session offset and same grid as the tracking, using the side camera's frame times (motion energy has one value per side-camera frame): `frameTimes − video_shift − goCue[trial]`, then `np.interp` onto the bin centres. If the frame-time vector is the wrong length, shorter than 2, or has fewer than 2 finite entries, the code falls back to synthetic 400 Hz frame times minus a fixed 0.5 s: `(1..n)/400 − 0.5 − goCue[trial]`.

ii.
```python
            frame_times = as_1d_float(read_matlab_any(h5file, side_view["frameTimes"][trial_idx, 0]))
            frame_times = frame_times - video_shift - align_time
            me_trial = np.asarray(me_raw[trial_idx], dtype=np.float64).reshape(-1)
            if (frame_times.size != me_trial.size
                or frame_times.size < 2
                or np.isfinite(frame_times).sum() < 2):
                side_ts = np.asarray(read_matlab_any(h5file, side_view["ts"][trial_idx, 0]), dtype=np.float64)
                side_ts = np.transpose(side_ts, (2, 1, 0))
                fallback_times = np.arange(1, side_ts.shape[0] + 1, dtype=np.float64) / 400.0
                fallback_times = fallback_times - 0.5 - align_time
                if fallback_times.size == me_trial.size and fallback_times.size >= 2:
                    frame_times = fallback_times
```

iii. Both the primary path and the fallback are the two branches of `loadMotionEnergy.m`'s `try`/`catch`: `frameTimes = (1:size(obj.traj{1}(trix).ts,1))./400; interp1(frameTimes-0.5-alignTimes(trix), me.data{trix}, taxis)`. The AI added the fallback after tracing a single residual `no_video` trial: *"I found the exact cause: that trial's `frameTimes` vector exists, but it is entirely `NaN`. The MATLAB loader explicitly falls back in that case, so I'm adding the same check."*

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases, all handled by keeping the trial and degrading the affected output rather than dropping data or fabricating values:
- **All-NaN or mismatched `frameTimes` (motion energy):** fall back to synthetic 400 Hz frame times offset by 0.5 s, mirroring `loadMotionEnergy.m`; if even that does not match, the trial's motion energy stays NaN and is emitted as `no_video`.
- **All-NaN `frameTimes` (tongue/paw):** `interpolate_visible_segments` finds no valid frames, the whole trial comes out NaN, and every bin becomes `not visible`. The `findPosition.m` fallback is deliberately not applied here (MATLAB also skips those trials).
- **Untracked frames:** DeepLabCut coordinates are already NaN where likelihood is low; those frames are excluded from the interpolation and the corresponding bins become `not visible`. For non-tongue features the leading/trailing gaps are nearest-filled, following `fillmissing(...,'nearest')` in the authors' code.
- **Missing fields:** `bp.autolearn` is replaced by all-False when absent; a feature name not present in a view's `featNames` is skipped; an empty HDF5 reference returns `None` and an empty quality label becomes `''`.
- **Empty sessions:** a session with < 2 usable trials or < 10 surviving units raises rather than being silently emitted.
- One consequence the AI does not handle: a bin that the code itself marks `visible` but whose speed is NaN (because `np.gradient` propagates NaN across an interior tracking gap) silently falls through to the `not visible` class. Measured on `JEB19_2023-04-19`, this affects **54 % of visible tongue bins** and ~2 % of visible paw bins.

ii.
```python
        autolearn = (
            as_bool_1d(bp["autolearn"]) if "autolearn" in bp.keys() else np.zeros_like(early, dtype=bool)
        )
```
```python
    for feature_name in feature_names:
        if feature_name not in names:
            continue
```
```python
    if not all_speed:
        return (np.full(TIME_BINS.shape, np.nan, dtype=np.float64),
                np.zeros(TIME_BINS.shape, dtype=bool),
                np.zeros(TIME_BINS.shape, dtype=bool))
```
```python
def fill_nearest_1d(y):
    y = np.asarray(y, dtype=np.float64).copy()
    good = np.flatnonzero(np.isfinite(y))
    if good.size == 0:
        return y
    y[: good[0]] = y[good[0]]
    y[good[-1] + 1 :] = y[good[-1]]
    return y
```

iii. The AI investigated the one anomalous trial it found rather than patching around it: *"One edge case remains: a single kept trial in `JEB19 2023-04-19` is still falling into `motion_energy = no_video`. I'm tracing that raw trial now so I can decide whether it's a genuine missing-video case or just a recoverable frame-count mismatch"*, and then mirrored the MATLAB fallback. The general principle is to reproduce whatever the authors' loaders do with bad frames and otherwise let the `not visible` / `no_video` class carry the information.

## 11-a. What are the most time-consuming steps of the code?

i. The conversion takes about 60 s for the 12 sessions (~5 s/session). The dominant costs are (1) the per-trial, per-feature HDF5 dereferencing in `compute_video_outputs` — for each trial `featNames`, `ts` and `frameTimes` are re-read through `read_matlab_any` once per feature group (3 calls) plus an extra `frameTimes` read for motion energy — and (2) `smooth_causal_reflect`, which runs one `np.convolve` per (cluster, trial): roughly 14 k convolutions of length 1115 per session, for every cluster including the ones later discarded by the 1 Hz filter. Reading each session file is also charged twice because `h5py.File` is opened once in `convert_session` and again in `compute_video_outputs`.

ii.
```python
    with h5py.File(data_path, "r") as h5file:          # convert_session
        ...
    tongue_disc, paw_disc, me_disc, thresholds = compute_video_outputs(session_info, keep_trials)
```
```python
    with h5py.File(data_path, "r") as h5file:          # compute_video_outputs, same file again
```
```python
    for i in range(padded.shape[0]):
        out[i] = np.convolve(padded[i], kernel, mode="same")
```

iii. Not discussed. The AI only remarked that *"the conversion is still running, which is expected here because each session is rebinning spikes and resampling the video streams onto the 5 ms neural grid."*

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Four:
- `smooth_causal_reflect`'s per-row `np.convolve` loop — a single `scipy.ndimage.convolve1d(..., axis=1)` or an FFT convolution would smooth all trials at once.
- The per-cluster loop in `compute_neural_session`, which rebuilds a `(n_trials, 1100)` count matrix per cluster; a single `np.histogram2d`/`np.add.at` over (cluster, trial, bin) would do all clusters at once.
- The per-dimension loop inside `interpolate_visible_segments` (two `np.interp` calls where x and y could be handled together).
- The per-trial loop in `compute_video_outputs`. This one is hard to vectorise because each trial has a different number of camera frames, so there is no rectangular array to work on.

ii.
```python
    for clu_idx in range(qualities.shape[0]):
        ...
        counts = np.zeros((keep_trials.size, NT), dtype=np.float64)
        if mapped_trials.size:
            np.add.at(counts, (mapped_trials, bin_idx), 1.0)
        rates = smooth_causal_reflect(counts / DT).astype(np.float32)
```
```python
        for dim in range(src_xy.shape[1]):
            out[seg_mask, dim] = np.interp(target_t[seg_mask], seg_t, src_xy[grp, dim])
```

iii. Not discussed. Within each cluster the spike binning itself is already vectorised via `np.add.at` rather than a per-trial loop.

## 11-c. What processing does the code repeat multiple times?

i. Some, all in the reading layer:
- Each session's `data_structure_*.mat` is opened and parsed twice (behaviour/spikes, then video).
- Per trial, `featNames` is decoded once per `get_feature_positions` call (3 calls for tongue side, tongue bottom, paw) even though two of them target the same bottom-view entry, and the bottom view's `ts` array is read twice per trial for the same reason.
- The side camera's `frameTimes` is read once inside `get_feature_positions` and again for motion energy, and its `ts` a third time when the fallback triggers.
What is *not* repeated: `video_shift` is computed once per session, the smoothing kernel and the bin grid are built once at module import, and the per-session thresholds are computed once from the fully assembled arrays.

ii.
```python
SMOOTH_KERNEL = build_smoothing_kernel()
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_BINS = TIME_EDGES[:-1] + DT / 2.0
```
```python
        video_shift = get_video_shift(h5file, bit_start)   # once per session
```
```python
    feat_names = np.asarray(read_matlab_any(h5file, view_group["featNames"][trial_idx, 0])).reshape(-1)
    ts = np.asarray(read_matlab_any(h5file, view_group["ts"][trial_idx, 0]), dtype=np.float64)
```

iii. Not discussed.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A few items:
- Firing rates are smoothed for **every** quality-passing cluster before the > 1 Hz test, so all the convolution work for the rejected units is thrown away. Testing the mean of the raw counts first would be exactly equivalent (the kernel is normalised) and would skip that work.
- `get_feature_positions` always returns a third value, `np.ones(TIME_BINS.shape, dtype=bool)`, which every caller discards with `_`.
- `bp.no` is read and never used; `right_choice`/`left_choice` are computed over all trials and then immediately subset to the kept trials.
- `read_matlab_any` is a fully generic recursive reader, so where it is pointed at a group it materialises every field of that group whether or not it is needed.
- `fill_nearest_1d` is applied to tongue positions even though the MATLAB code deliberately does not fill tongue gaps; the filled edge values then get zero velocity in bins that are already labelled `not visible`, so the work has no effect on the output.

ii.
```python
        rates = smooth_causal_reflect(counts / DT).astype(np.float32)
        mean_fr = float(rates.mean())
        if mean_fr > LOW_FR_HZ:
            kept_unit_data.append(rates)
```
```python
    return mean_speed, any_visible, np.ones(TIME_BINS.shape, dtype=bool)
```
```python
    no = as_bool_1d(bp["no"])
```

iii. Not discussed. Everything else computed after loading ends up in the saved pickle.
