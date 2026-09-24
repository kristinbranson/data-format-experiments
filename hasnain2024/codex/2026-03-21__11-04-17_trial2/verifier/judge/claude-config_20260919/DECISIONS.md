# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** load all of the data. It hard-codes a list of **12 sessions** (`CONTEXT_SESSION_SPECS`), transcribed from the paper's Figure 8 loader scripts (`loadJEB6/JEB7/EKH1/EKH3/JGR2/JGR3/JEB19_ALMVideo.m`), all from `data/Ephys_Behavior/`, each with the probe index those loaders specify. The 22 randomized-delay ephys sessions, the other 13 fixed-delay ephys sessions, and all behavior-only (inhibition) sessions are excluded. Each session file is read with `mat73.loadmat` (MATLAB v7.3/HDF5 only); the companion `motionEnergy_<anm>_<date>.mat` is read with `scipy.io.loadmat`. Paths are relative (`data/Ephys_Behavior/...`), so the script must be run from `/app`. Result: 12 sessions, 7 subjects, 2,415 trials, 519 units.

ii.
```python
CONTEXT_SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", 1),
    SessionSpec("JEB7", "2021-04-29", 0),
    ...
    SessionSpec("JEB19", "2023-04-18", 0),
]

@property
def data_path(self) -> Path:
    return Path("data/Ephys_Behavior") / f"data_structure_{self.session_id}.mat"
```
```python
obj = mat73.loadmat(spec.data_path)["obj"]
me = load_motion_energy(spec)
...
def load_motion_energy(spec: SessionSpec) -> dict:
    me_mat = loadmat(spec.motion_energy_path, squeeze_me=True, struct_as_record=False)
    me = me_mat["me"]
    return {"data": np.atleast_1d(me.data), "moveThresh": float(me.moveThresh)}
```

iii. From CONVERSION_NOTES Step 4/5: "Because the decoder outputs include behavioral context (`WC` vs `DR`), the most paper-consistent primary dataset is the two-context ALM subset loaded by the context-analysis scripts rather than the entire raw ephys archive." Key Decision 1–2: the 12-session Figure 8 list "is the clearest code path corresponding to the paper's context analyses", and the paper reports "12 sessions, six mice, 522 units" for that dataset, which the AI matched (519 units, 43.25 units/session).

## 1-b. How are the data split into subjects (mice)?

i. The subject is the animal field of the hard-coded `SessionSpec` (identical to the filename prefix, e.g. `JEB19_2023-04-19` → `JEB19`). At assembly, `subjects` is the sorted unique set and `subject_idx` indexes it per session. 7 subjects: EKH1, EKH3, JEB19, JEB6, JEB7, JGR2, JGR3.

ii.
```python
@dataclass(frozen=True)
class SessionSpec:
    animal: str
    date: str
    probe_index: int
...
subjects = sorted({sess["subject"] for sess in converted_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
"subject_idx": np.asarray([subject_to_idx[sess["subject"]] for sess in converted_sessions], dtype=np.int64),
```

iii. Step 10 Check 4: the paper text says six mice for the two-context dataset, but "the Figure 8 loader code names 7 animals ... and the selected raw files also contain these same 7 IDs. I retained 7 because code + data agree." The animal id is taken from the loader metadata / filename rather than `obj.meta.anm`, which is not present in every file.

## 1-c. How are the data split into sessions?

i. One `.mat` file = one session = one `SessionSpec` = one element of `neural`/`input`/`output`. Each session uses exactly one probe, the one named in the reference loader (`spec.probe_index`, 0-based); no session in the chosen subset has two probes. Only the 12 fixed-delay two-context sessions become sessions; the reference solution instead produces 44 sessions (25 fixed-delay + 19 randomized-delay).

ii.
```python
for sess_idx, spec in enumerate(session_specs):
    converted_sessions.append(convert_session(spec, make_plot=make_plot))
...
clu = obj["clu"][spec.probe_index]
if clu is None:
    raise RuntimeError(f"{spec.session_id}: selected probe {spec.probe_index + 1} has no neural data")
```

iii. Same rationale as 1-a: the two-context subset is the code path corresponding to the paper's context analyses, and its unit count (519) reproduces the paper's reported 522. The randomized-delay sessions were explicitly set aside: "These sessions are less suitable for the requested context decoder because context does not vary there" (Step 4 table).

## 1-d. How are the data split into trials?

i. Trials are the rows of the per-trial Bpod fields in `obj.bp`; `bp.Ntrials` gives the count, and every per-trial vector (`hit`, `miss`, `early`, `no`, `autowater`, `stim.enable`, `ev.goCue`, `ev.lickL/lickR`) is indexed by the same raw trial index. Camera trajectories (`obj.traj[view][field][trial]`) and motion energy (`me.data[trial]`) are indexed with the same raw index; spikes carry their trial number in `clu.trial` (1-based). The AI keeps the raw trial index throughout and maps it to a "local" kept-trial index only for spike binning.

ii.
```python
n_trials = int(obj["bp"]["Ntrials"])
...
def normalize_trial_struct(view_dict: dict, trial_index: int) -> dict:
    return {key: view_dict[key][trial_index] for key in view_dict.keys()}
...
local_map = np.full(align_times.size + 1, -1, dtype=np.int64)
local_map[kept_trials_0based + 1] = np.arange(kept_trials_0based.size, dtype=np.int64)
spike_local_trial = local_map[trial_numbers_1based]
```

iii. Not discussed at length; the Bpod table defines trials directly and there is one `goCue` per trial, so no reconstruction was needed. (Unlike the reference, the AI does not truncate per-trial fields to `Ntrials`; I verified that in all 12 selected sessions every per-trial field already has exactly `Ntrials` entries, so this does not bite here.)

## 1-e. How are trials filtered based on quality controls?

i. Four filters, all before any computation: drop early-lick trials (`bp.early`), drop photostimulation trials (`bp.stim.enable`), **drop ignore/no-response trials** (`bp.no`, equivalently keep only `hit | miss`), and then **drop any remaining trial with no lick event after the alignment time**. Sessions with <2 surviving trials would raise. 2,415 of 3,626 raw trials (67%) survive. No filter for trials that run past the end of the recording (I verified no all-zero neural trials exist in the 12 selected sessions, so this did not matter here).

ii.
```python
def select_valid_trials(obj: dict) -> np.ndarray:
    bp = obj["bp"]
    early = ensure_1d_numeric(bp["early"]) != 0
    no = ensure_1d_numeric(bp["no"]) != 0
    hit = ensure_1d_numeric(bp["hit"]) != 0
    miss = ensure_1d_numeric(bp["miss"]) != 0
    stim_enable = ensure_1d_numeric(bp["stim"]["enable"]) != 0
    valid = (~early) & (~no) & (~stim_enable) & (hit | miss)
    return np.flatnonzero(valid)
```
```python
for trial_idx in valid_trials:
    lick_dir = get_first_lick_direction(obj, int(trial_idx), align_times[trial_idx])
    if lick_dir is None:
        continue
    kept_trials.append(int(trial_idx))
```

iii. Key Decision 3: "Keep non-stim, non-early, non-ignore neural trials with a valid lick outcome (`hit` or `miss`). Rationale: this matches the paper's omission of early/ignore trials and keeps outcome well-defined." Step 3 records the paper's rules: "Early lick trials were omitted from analyses. Ignore / no-response trials were omitted from behavioral analyses."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu[probe].trialtm` (spike time relative to its trial's start), `obj.clu[probe].trial` (1-based trial of each spike), `obj.clu[probe].quality` (curation label), and `bp.ev.goCue` (alignment times). Only the single probe named by the reference loader is used.

ii.
```python
clu = obj["clu"][spec.probe_index]
unit_quality = [str(q).strip() if q is not None else "" for q in clu["quality"]]
...
trialdat = bin_unit_spikes(
    clu["trialtm"][unit_idx],
    np.asarray(clu["trial"][unit_idx], dtype=np.int64),
    align_times,
    kept_trials,
)
```
```python
align_times = ensure_1d_numeric(bp["ev"][ALIGN_EVENT])   # ALIGN_EVENT = "goCue"
```

iii. Step 1/5: the reference path is `alignSpikes` (`trialtm_aligned = trialtm - event`, `params.alignEvent = 'goCue'`) followed by `getSeq`, and "reference decoding scripts ... decode from `obj.trialdat`, confirming that single-trial smoothed firing rates are the intended neural representation rather than raw spikes."

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 5 ms bins with `np.add.at`, divided by `dt` to get Hz, and smoothed along time with a **causal** half-Gaussian: `gausswin(15)` with the first 7 taps zeroed, normalised to sum 1, convolved `'same'`, with the reference's `'reflect'` boundary handling (prepend the first 15 samples, then trim them). Stored as `float32`. No normalisation, baseline subtraction, or z-scoring.

ii.
```python
DT = 0.005; SMOOTH_N = 15; BCTYPE = "reflect"

def gaussian_window(n, alpha=2.5):
    half = (n - 1) / 2.0
    k = np.arange(0, n, dtype=np.float64) - half
    return np.exp(-0.5 * (alpha * k / half) ** 2)

def my_smooth(x, n, bctype="none"):
    if bctype.lower() == "reflect":
        x_filt = np.concatenate([x[:n, :], x], axis=0); trim = n
    ...
    kern = gaussian_window(n)
    kern[: len(kern) // 2] = 0.0          # causal
    kern /= np.sum(kern)
    for col in range(x_filt.shape[1]):
        out[:, col] = np.convolve(x_filt[:, col], kern, mode="same")
    out = out[trim:, :]
```
```python
    np.add.at(aligned_counts, (spike_local_trial[valid_bins], bin_idx[valid_bins]), 1.0)
    rates = aligned_counts / DT
    rates = my_smooth(rates.T, SMOOTH_N, BCTYPE).T
```

iii. This is a line-by-line port of the reference `getSeq.m` (`obj.trialdat{prb}(:,i,j) = mySmooth(N./params.dt, params.smooth, params.bctype)`) and `mySmooth.m` (which builds `gausswin(N)`, zeroes `kern(1:floor(N/2))` to make it causal, and trims after `'reflect'` padding). Step 6 notes: "causal Gaussian smoothing matching the reference `mySmooth` behavior".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. (1) Quality label, stripped of whitespace and compared **case-sensitively** against `{garbage, gabrga, noisy, real?}` — everything else, including `multi`, `poor`, `fair`, is kept. (2) Mean firing rate over the ±2.5 s window across *all* session trials must exceed 1 Hz. Result: 519 units of 606 quality-passing units, 27–67 per session.

ii.
```python
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}
LOW_FR_HZ = 1.0

def keep_quality(quality: str) -> bool:
    quality = str(quality).strip()
    return quality not in QUALITY_EXCLUDE
```
```python
def mean_firing_rate_window(trialtm, trial_numbers_1based, align_times) -> float:
    aligned = trialtm - align_times[trial_numbers_1based - 1]
    in_window = (aligned >= TMIN) & (aligned < TMAX)
    return float(np.sum(in_window) / (align_times.size * (TMAX - TMIN)))
...
    if mean_fr <= LOW_FR_HZ:
        continue
```

iii. Step 10 Check 3: "reference: `findClusters(..., {'all'})` excludes exact labels `garbage`, `gabrga`, `noisy`, `real?`; Figure 8 sets `params.lowFR = 1`; conversion: same label exclusions and `LOW_FR_HZ = 1.0`; result: matched." Step 4 resolves the 0.5 vs 1 Hz conflict "in favor of the paper and figure scripts", quoting the paper's "units with firing rates exceeding 1 Hz".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. One subtraction per spike: `trialtm − goCue[trial]`, since `trialtm` is already on the behaviour clock relative to its own trial's start and `bp.ev.goCue` is on the same clock. The same `bp.ev.goCue` field is used for WC trials as for DR trials.

ii.
```python
aligned = trialtm[valid_spikes] - align_times[trial_numbers_1based[valid_spikes] - 1]
bin_idx = np.floor((aligned - TMIN) / DT).astype(np.int64)
valid_bins = (bin_idx >= 0) & (bin_idx < TIME_AXIS.size)
```

iii. Key Decision 4 / Step 4: `goCue` is the reference default (`alignSpikes.m`, `params.alignEvent='goCue'`) and is required by the task. For WC trials, "direct raw-data inspection showed populated `goCue` values on WC trials ... the stored `goCue` appears to serve as the water-presentation-equivalent event, which resolves the cross-context alignment requirement."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 5 ms bins, 1000 bins spanning −2.5 to +2.5 s from the go cue, identical for every trial, session, and data stream (neural, input, all three camera outputs). No rebinning or resampling afterwards; `metadata['time_bin_size'] = 5.0` ms.

ii.
```python
DT = 0.005
TMIN = -2.5
TMAX = 2.5
TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0
EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
```

iii. Step 4: "Tutorial example uses `dt = 1/100` (10 ms); `getDefaultParams.m` and paper methods indicate 5 ms ... Resolve in favor of 5 ms base binning." Step 10 notes Figure 8 itself used `tmin=-3, tmax=2.5, dt=10 ms` and explains the deviation: "the user requested go-cue-centered decoder input and a uniform compact window, while 5 ms is still paper-consistent."

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. No raw variable. It is the analysis grid itself: the centre of each of the 1000 5 ms bins of the go-cue-aligned window, i.e. −2.4975 … +2.4975 s, identical for every trial and session.

ii.
```python
TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0
...
time_input = TIME_AXIS[None, :].astype(np.float32)
session_input.append(time_input)
```

iii. Step 5 mapping table: "Common aligned time axis → `input[0]`: continuous time-from-go-cue vector repeated for every trial ... Single decoder input requested by user."

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond tiling: the same `(1, 1000)` float32 vector is stored for every trial (a fresh array per trial, not a shared view).

ii.
```python
time_input = TIME_AXIS[None, :].astype(np.float32)
```

iii. N/A — the axis is defined by the conversion, not derived from data.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is literally the binning grid used for the spikes. Spike bins are computed as `floor((t_aligned − TMIN)/DT)` over the same `TMIN/DT` grid, and the input is the centre of those same bins, so bin *k* is the same 5 ms interval in both streams. The camera streams are interpolated onto this same `TIME_AXIS`.

ii.
```python
bin_idx = np.floor((aligned - TMIN) / DT).astype(np.int64)     # neural
...
xpos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 0], ...)   # video
```

iii. Step 10 Check 2: "converted `time_from_go_cue_seconds` matched the independently reconstructed 5 ms time axis exactly (`np.allclose = True`)", and the neural spot-check "matched exactly after direct histogramming against the same `[-2.5, 2.5)` bin edges".

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The lick-port event times `bp.ev.lickL` and `bp.ev.lickR` (not the instructed side `bp.R`/`bp.L`, and not `hit`/`miss`). The alignment time `bp.ev.goCue` defines "post-alignment".

ii.
```python
def get_first_lick_direction(obj: dict, trial_idx: int, go_time: float) -> int | None:
    lick_l = event_list_to_array(obj["bp"]["ev"]["lickL"][trial_idx])
    lick_r = event_list_to_array(obj["bp"]["ev"]["lickR"][trial_idx])
    lick_l = lick_l[lick_l >= go_time]
    lick_r = lick_r[lick_r >= go_time]
```

iii. Key Decision 8: "Use actual first post-alignment lick side from lickport events, not the instructed side (`R`/`L`). Rationale: the user explicitly requested 'lick direction'; for error trials this differs from the cue/target side." Validated in Step 10 ("first-lick direction matched direct raw-event reconstruction, `LICK_MATCH = True`").

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The earliest lick time at or after the go cue on each port is compared; left wins ties/earlier → 0, right → 1. Trials with no lick on either port after the go cue return `None` and are **dropped from the dataset**, so the output has only **two** classes — the `none` class required by the Decoder Task spec does not exist. The per-trial label is tiled constant across all 1000 bins. Overall distribution: left 0.514, right 0.486.

ii.
```python
    first_l = lick_l[0] if lick_l.size else math.inf
    first_r = lick_r[0] if lick_r.size else math.inf
    if math.isinf(first_l) and math.isinf(first_r):
        return None
    return 0 if first_l < first_r else 1
```
```python
output_trial = np.vstack([
    np.full(TIME_AXIS.size, lick_dir, dtype=np.int64),
    ...
])
...
"output_values": [["left", "right"], ...]
```

iii. Key Decision 3 and 9: ignore/no-lick trials are excluded to keep "outcome well-defined", following the paper's omission of ignore trials; per-trial categoricals are "represented as constant binary traces across the full trial window" to keep a uniform `(n_output, n_timepoints)` format. The notes never discuss the `none` class named in the Decoder Task specification.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial flag, `bp.autowater`.

ii.
```python
autowater = ensure_1d_numeric(bp["autowater"])
...
0 if autowater[trial_idx] != 0 else 1,
```

iii. Step 4: "Behavioral context is represented by `autowater` in both paper and code (`~autowater` = DR, `autowater` = WC)." Step 10: "`behavioral_context`: `autowater` exactly matches reference DR/WC splitting."

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling: autowater → WC = 0, otherwise DR = 1, tiled constant across the 1000 bins. Distribution 0.289 WC / 0.711 DR.

ii.
```python
trial_labels.append((lick_dir,
                     0 if autowater[trial_idx] != 0 else 1,
                     1 if hit[trial_idx] != 0 else 0))
...
np.full(TIME_AXIS.size, context_label, dtype=np.int64),
...
"output_values": [..., ["WC", "DR"], ...]
```

iii. Codes follow the prompt's ordering (WC, DR); the mapping is the reference code's condition split.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `bp.hit` for the label, with `bp.miss` and `bp.no` used in the trial filter (only hit-or-miss, non-ignore trials reach the labelling step, so `hit != 0` ⇔ correct and `hit == 0` ⇔ incorrect).

ii.
```python
hit = ensure_1d_numeric(bp["hit"])
...
1 if hit[trial_idx] != 0 else 0,
```

iii. Step 5 mapping: "`bp.hit`, `bp.miss` → `output[2]` ... Reference `getOutcome`. Restrict to hit/miss trials; exclude ignore/no-response trials." `getOutcome.m` returns correct/error from `bp.hit` and sets ignore trials to NaN.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Two classes only: incorrect = 0 (miss), correct = 1 (hit), tiled constant across bins. Ignore trials were removed by the trial filter, so the **`ignore` class required by the Decoder Task spec is absent**. Distribution 0.136 incorrect / 0.864 correct.

ii.
```python
np.full(TIME_AXIS.size, outcome_label, dtype=np.int64),
...
"output_values": [..., ["incorrect", "correct"], ...]
```

iii. Same as 1-e/4-b: the paper omits ignore trials from behavioural analyses, and dropping them "keeps outcome well-defined" (Key Decision 3). No discussion of the specified third class.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj[0]` (side camera) only, feature `tongue`: `ts[:, 0:2, feat]` (x, y — already NaN where the authors' likelihood cut failed), plus `frameTimes`, `NdroppedFrames`, and the clock-correction fields `bp.ev.bitStart`, `sglx.bitcode.bitstart`, `sglx.fs`, and `bp.ev.goCue`. The bottom-camera tongue features (`top_tongue`, etc.) are not used.

ii.
```python
tongue_x, tongue_y = align_feature_positions(obj, 0, "tongue", align_times, TIME_AXIS, vidshift)
tongue_visible = np.isfinite(tongue_x) & np.isfinite(tongue_y)
```
```python
    view_dict = obj["traj"][view_index]
    feat_index = find_feature_index(view_dict, feat_name)
    ...
    xy = ts[:, 0:2, feat_index]
```

iii. Step 5 mapping table: "DLC kinematics for side-view tongue feature → `output[3]` ... Planned scalar = `sqrt(xvel^2 + yvel^2)` for `tongue_*_view1`", following `getKinematicsFromVideo`/`findVelocity`. Step 6 notes only the needed features (`tongue`, `top_paw`, `bottom_paw`) are extracted, for speed.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. A close port of the reference `findPosition.m` + `findVelocity.m`: (1) skip trials whose `NdroppedFrames` is NaN; (2) shift frame times by the session video offset and the trial's go cue; (3) **linearly interpolate x and y onto the 5 ms `TIME_AXIS`** (`left/right = NaN`), with no smoothing for tongue (the reference's `mySmooth(ts,1,...)` is a no-op and is skipped for tongue anyway) and **no nearest-fill** for tongue; (4) `np.gradient` of the interpolated x and y per bin — NaNs (tongue invisible) are set to **0**, as in the reference; (5) speed = `hypot(xvel, yvel)`. Velocity is therefore in pixels per 5 ms bin, not per second (a constant scale factor, irrelevant to a percentile split).

ii.
```python
        xpos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 0], left=np.nan, right=np.nan)
        ypos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 1], left=np.nan, right=np.nan)
        if "tongue" not in feat_name:
            xpos[:, trial_idx] = fill_nearest_1d(xpos[:, trial_idx])
```
```python
        xv = np.gradient(tsinterp[:, 0]); yv = np.gradient(tsinterp[:, 1])
        if "tongue" not in feat_name:
            xv = xv - basederiv[0]; yv = yv - basederiv[0]
            xv = fill_nearest_1d(xv); yv = fill_nearest_1d(yv)
        else:
            xv = np.where(np.isfinite(xv), xv, 0.0)     # "set tongue velocity to 0 if not visible"
            yv = np.where(np.isfinite(yv), yv, 0.0)
...
tongue_speed = np.sqrt(tongue_vx**2 + tongue_vy**2)
```

iii. Key Decision 11: "Reduce multi-axis DLC velocity to a scalar speed magnitude before thresholding ... scalar speed is the most defensible collapse of x/y velocity." Step 6: "DLC-based tongue and paw speed extraction using reference-style position interpolation and velocity calculation." The NaN→0 rule and the no-smoothing/no-fill treatment of the tongue are copied verbatim from the reference MATLAB comments ("set tongue velocity to 0 if not visible", "fill missing values for all features except tongue").

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Two classes only. The threshold is the **50th percentile of tongue speed over the bins in which the tongue is visible**, pooled over all kept trials of the session. A bin is class 1 if `speed >= threshold` **and** the tongue is visible, else class 0. Because the tongue is invisible in ~82% of bins, class 0 is mostly "tongue not visible" rather than "slow". The `2: not visible` class required by the Decoder Task spec is not produced. Resulting distribution: 0.824 / 0.176.

ii.
```python
tongue_visible_kept = tongue_visible[:, kept_trials]
if np.any(tongue_visible_kept):
    tongue_threshold = float(np.nanpercentile(tongue_speed[:, kept_trials][tongue_visible_kept], 50))
else:
    tongue_threshold = 0.0
...
tongue_bin = ((tongue_speed[:, trial_idx] >= tongue_threshold) & tongue_visible[:, trial_idx]).astype(np.int64)
```

iii. Key Decision 10: "Compute per-session medians on the aligned traces across all kept trials/timepoints and bin each timepoint into `<50th percentile` vs `>=50th percentile`." Step 7: "Initial tongue discretization collapsed in one session because invisible tongue periods dominated the median; fixed by computing the session median from tongue-visible timepoints only and assigning invisible periods to the low bin." Step 10: "the conversion follows the reference tongue rule by treating missing tongue velocity as zero and keeping invisible periods in the low bin."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The video clock is corrected once per session by `vidshift = mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)` (the reference `findVideoOffset.m`), then each trial's frame times become `frameTimes − vidshift − goCue[trial]`, and x/y are interpolated onto the shared 5 ms `TIME_AXIS`, so the tongue stream sits on exactly the same time base as the spikes.

ii.
```python
def compute_vidshift(obj: dict) -> float:
    bit_start = mode_value(obj["bp"]["ev"]["bitStart"])
    vid_file_offset = mode_value(obj["sglx"]["bitcode"]["bitstart"]) / float(obj["sglx"]["fs"])
    return float(vid_file_offset - bit_start)
...
        shifted_time = frame_times - vidshift - align_times[trial_idx]
        xpos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 0], left=np.nan, right=np.nan)
```

iii. Ported from `findPosition.m` (`interp1(traj(trix).frameTimes - vidshift - obj.bp.ev.(alignEv)(trix), ts, taxis)`). Step 7: "Alignment visually centers all traces at the stored `goCue` event"; the `--show-processing` plots were used as the check.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The bottom camera (`obj.traj[1]`), **both** paw features `top_paw` and `bottom_paw`, same fields as the tongue (`ts[:, 0:2, feat]`, `frameTimes`, `NdroppedFrames`) plus the same clock-correction fields.

ii.
```python
paw_speeds = []
for paw_feat in ("top_paw", "bottom_paw"):
    paw_x, paw_y = align_feature_positions(obj, 1, paw_feat, align_times, TIME_AXIS, vidshift)
    paw_vx, paw_vy = compute_velocity(paw_x, paw_y, paw_feat)
    paw_speeds.append(np.sqrt(paw_vx**2 + paw_vy**2))
```

iii. Step 5 mapping: "DLC kinematics for bottom-view paw feature(s) → `output[4]` ... Planned scalar = mean speed magnitude across `top_paw` and `bottom_paw` in view 2." No separate discussion of tracking reliability of the two paws.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same pipeline as the tongue, but with the reference's non-tongue branches active: positions are interpolated onto the 5 ms axis and then **nearest-filled** (linear interpolation across interior gaps, edge-hold at the ends); the per-trial median frame-to-frame displacement (`basederiv`) is subtracted from the gradient; velocities are nearest-filled again; speed = `hypot`. The two paws' speeds are then averaged bin-wise over whichever paws are finite.

ii.
```python
        basederiv = np.nanmedian(diffs, axis=0)
        xv = np.gradient(tsinterp[:, 0]); yv = np.gradient(tsinterp[:, 1])
        if "tongue" not in feat_name:
            xv = xv - basederiv[0]
            yv = yv - basederiv[0]
            xv = fill_nearest_1d(xv); yv = fill_nearest_1d(yv)
```
```python
paw_stack = np.stack(paw_speeds, axis=0)
paw_counts = np.sum(np.isfinite(paw_stack), axis=0)
paw_speed = np.full(paw_counts.shape, np.nan, dtype=np.float64)
np.divide(np.nansum(paw_stack, axis=0), paw_counts, out=paw_speed, where=paw_counts > 0)
```

iii. A deliberate port of `findVelocity.m`, including its idiosyncrasy of subtracting `basederiv(1)` (the x baseline) from **both** x and y velocity — the AI reproduced the reference's own quirk rather than "fixing" it. Nearest-filling matches the paper methods as recorded in Step 3: "missing values filled with nearest available value for all features except tongue."

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Two classes: the session's 50th percentile of the combined paw speed over all kept trials and bins, `speed >= threshold` → 1 else 0. Bins/trials where the paw speed is NaN (video missing for the whole trial) evaluate `NaN >= thr` as `False` and silently become class 0. No `not visible` class. Distribution is exactly 0.500/0.500.

ii.
```python
paw_threshold = float(np.nanpercentile(paw_speed[:, kept_trials], 50))
...
(paw_speed[:, trial_idx] >= paw_threshold).astype(np.int64),
```

iii. Key Decision 10 (the prompt's median split). Step 9: "`paw_velocity` distribution ... Raw selected subset supports balanced median thresholding." Step 12 confirms the exactly balanced split was intentional ("exactly balanced by construction").

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue: the same session `vidshift`, then `frameTimes − vidshift − goCue[trial]`, then interpolation onto the shared 5 ms grid — but using the **bottom** camera's own `frameTimes` (`obj.traj[1]`), since that is the view the paws are tracked in.

ii.
```python
paw_x, paw_y = align_feature_positions(obj, 1, paw_feat, align_times, TIME_AXIS, vidshift)
...
        shifted_time = frame_times - vidshift - align_times[trial_idx]
```

iii. Same as 7-d — one offset and one grid for every video stream; verified in Step 12 by reconstructing `paw_velocity` bins directly from raw DLC trajectories for three trials of `JEB6_2021-04-18` ("all three matched the converted output exactly").

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file (`me.data`, one trace per trial at camera frame rate; `me.moveThresh` is read but never used), together with the **side** camera's `frameTimes` from `obj.traj[0]` and the same clock-correction fields. The embedded `obj.me` copy is not used.

ii.
```python
def load_motion_energy(spec: SessionSpec) -> dict:
    me_mat = loadmat(spec.motion_energy_path, squeeze_me=True, struct_as_record=False)
    me = me_mat["me"]
    return {"data": np.atleast_1d(me.data), "moveThresh": float(me.moveThresh)}
```
```python
    view_dict = obj["traj"][0]
    ...
        frame_times = view_dict["frameTimes"][trial_idx]
        me_trial = np.asarray(me["data"][trial_idx], dtype=np.float64).reshape(-1)
```

iii. Step 5: "`loadMotionEnergy` ... Use aligned motion-energy stream after interpolation onto neural time base." Step 2 notes that ephys sessions carry companion motion-energy files while behavior-only sessions embed `obj.me`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond alignment and resampling: the per-frame scalar (already a 99th-percentile-across-pixels value computed upstream by the authors) is truncated to the common length of `frameTimes`/`me.data`, linearly interpolated onto the 5 ms axis, and then nearest-filled so that no NaNs remain (including edge extrapolation by hold).

ii.
```python
        n = min(frame_times.size, me_trial.size)
        frame_times = frame_times[:n]; me_trial = me_trial[:n]
        valid = np.isfinite(frame_times) & np.isfinite(me_trial)
        shifted_time = frame_times[valid] - vidshift - align_times[trial_idx]
        aligned[:, trial_idx] = np.interp(time_axis, shifted_time, me_trial[valid], left=np.nan, right=np.nan)
        aligned[:, trial_idx] = fill_nearest_1d(aligned[:, trial_idx])
```

iii. A direct port of `loadMotionEnergy.m`, which does `interp1(frameTimes - vidshift - alignTimes(trix), me.data{trix}, taxis)` followed by `fillmissing(me.data,'nearest')` "(there are some nans at the start of each trial)". Verified in Step 10: "motion-energy output for the same trial matched direct raw interpolation + session-median thresholding (`ME_BIN_ALLCLOSE = True`)".

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Two classes: session 50th percentile over kept trials/bins, `>= threshold` → 1, else 0. Bins with no video at all (whole-trial failure) fall to class 0 via `NaN >= thr == False`; there is no `no video` class. Distribution 0.500/0.500. The authors' own per-session `me.moveThresh` is loaded but deliberately unused.

ii.
```python
me_threshold = float(np.nanpercentile(motion_energy[:, kept_trials], 50))
...
(motion_energy[:, trial_idx] >= me_threshold).astype(np.int64),
```

iii. Key Decision 10 — the prompt's required median split — is applied in place of the paper's manually chosen bimodal `moveThresh` (Step 3 records the paper's rule: "threshold chosen manually on a per-session basis at the separation between two modes of the motion-energy distribution"). Step 10: "movement variables: derived from reference-style aligned video/motion-energy traces, then discretized per the user's required median split."

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same clock correction and same grid as the tracking streams, using the side camera's frame times (motion energy has one value per side-camera frame): `frameTimes − vidshift − goCue[trial]`, interpolated onto `TIME_AXIS`. `vidshift` is recomputed inside `align_motion_energy` (a duplicate of the value already computed in `convert_session`).

ii.
```python
def align_motion_energy(obj, me, align_times, time_axis):
    vidshift = compute_vidshift(obj)
    view_dict = obj["traj"][0]
    ...
        aligned[:, trial_idx] = np.interp(time_axis, shifted_time, me_trial[valid], left=np.nan, right=np.nan)
```

iii. Matches `loadMotionEnergy.m`, which also indexes `obj.traj{1}(trix).frameTimes`. Checked in Step 10 against a direct raw-file reconstruction.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Four cases. (1) **Trials flagged bad by `NdroppedFrames = NaN`** are skipped, leaving that trial's kinematics all-NaN. (2) **Missing/all-NaN `frameTimes`**: the AI *fabricates* a nominal 400 Hz frame grid `(1..n)/400` and aligns it with the session `vidshift` (the reference's `catch` branch does the same with a fixed 0.5 s shift); 2 such trials were found. (3) **Untracked frames** (DLC likelihood failure, already NaN in `ts`): for the tongue the velocity is set to 0 and the bin marked not-visible → class 0; for paws and motion energy the gaps are nearest-filled, so nothing is marked missing. (4) **Whole-trial missing video** (nothing interpolatable): paw/motion-energy values stay NaN, and because the discretisation is a bare `>=` comparison, `NaN >= thr` is `False`, so those bins are silently emitted as class 0 ("below threshold") rather than as a missing/not-visible category. Missing-unit or missing-field cases raise (`RuntimeError`) rather than being skipped.

ii.
```python
        dropped_arr = ensure_1d_numeric([] if dropped is None else dropped)
        if dropped_arr.size and np.all(np.isnan(dropped_arr)):
            continue
        ...
        if frame_times.size == 0 or np.all(np.isnan(frame_times)):
            frame_times = (np.arange(ts.shape[0], dtype=np.float64) + 1.0) / 400.0
        ...
        valid = np.isfinite(shifted_time) & np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
        if np.sum(valid) < 2:
            continue
```
```python
def fill_nearest_1d(x):
    good = np.isfinite(x)
    if not np.any(good):
        return x
    idx = np.arange(x.size)
    x[~good] = np.interp(idx[~good], idx[good], x[good])
    return x
```

iii. Step 10 edge-case review: "`2` raw trials with empty/all-NaN `frameTimes`; the conversion already handles this by falling back to the nominal 400 Hz frame grid, matching the reference behavior of reconstructing time when timestamps are unavailable"; "`527` raw trials in the selected sessions had the tongue completely invisible; the conversion follows the reference tongue rule by treating missing tongue velocity as zero and keeping invisible periods in the low bin." Nearest-filling non-tongue features is justified by the paper methods quote in Step 3.

## 11-a. What are the most time-consuming steps of the code?

i. The AI identifies loading: `mat73.loadmat` reads the entire session object (including fields it never touches, such as spike waveforms). Measured runtime is ~3.8–7.5 s per session, 72 s for all 12 sessions, well inside the 15-minute budget, so no further optimisation was pursued. Timing is instrumented per session (`time.perf_counter()` around `convert_session`), not per processing step, so the attribution to loading is inferred rather than measured. The other substantial cost is the per-unit × per-trial `np.convolve` smoothing loop (~100k 1-D convolutions across the dataset).

ii.
```python
def convert_session(spec, make_plot=False) -> dict:
    session_start = time.perf_counter()
    print(f"[session] loading {spec.session_id}")
    obj = mat73.loadmat(spec.data_path)["obj"]
...
    elapsed = time.perf_counter() - session_start
    print(f"[session] {spec.session_id} done in {elapsed:.2f}s | ...")
```

iii. Step 6: "Full MATLAB session objects are loaded into memory per session via `mat73`, which is simpler and reliable but not maximally lean." Step 7 estimated "~7.4 s/session → ~1.5–2 min for 12 sessions", which matched the 72 s actual.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Remaining Python loops: (a) the per-trial loops in `align_feature_positions`, `compute_velocity`, and `align_motion_energy` (run over **all** raw trials, three features per session); (b) the per-unit loop in `convert_session`, which calls `bin_unit_spikes` once per unit; (c) inside `my_smooth`, a per-column `np.convolve` — i.e. one convolution per trial per unit, the single most vectorizable hot spot (a 2-D `scipy.ndimage.convolve1d`/FFT along the time axis would replace it). The AI identified (a) but not (b)/(c). Spike counting itself is already vectorized with `np.add.at`.

ii.
```python
    out = np.empty_like(x_filt, dtype=np.float64)
    for col in range(x_filt.shape[1]):
        out[:, col] = np.convolve(x_filt[:, col], kern, mode="same")
```
```python
    for trial_idx in range(xpos.shape[1]):
        tsinterp = np.column_stack([xpos[:, trial_idx], ypos[:, trial_idx]])
```

iii. Step 6: "Kinematic alignment currently interpolates each requested feature trial-by-trial in Python loops"; speed-ups implemented were "Neural spike binning is vectorized within each unit using `np.add.at` rather than nested time-bin loops" and extracting only the three needed kinematic features. Since total runtime was ~72 s, no further vectorization was judged necessary.

## 11-c. What processing does the code repeat multiple times?

i. The AI does not document any repeated processing; the code does contain some. `compute_vidshift` is computed twice per session (once in `convert_session`, once inside `align_motion_energy`). Spike alignment (`trialtm − goCue`) is computed twice for every unit: once in `mean_firing_rate_window` and again in `bin_unit_spikes`. `find_feature_index` and the per-trial struct repacking (`normalize_trial_struct`) are redone for each of the three features. All are cheap relative to file loading.

ii.
```python
    vidshift = compute_vidshift(obj)                     # convert_session
    ...
def align_motion_energy(obj, me, align_times, time_axis):
    vidshift = compute_vidshift(obj)                     # recomputed
```
```python
        mean_fr = mean_firing_rate_window(clu["trialtm"][unit_idx], ...)   # aligns spikes
        ...
        trialdat = bin_unit_spikes(clu["trialtm"][unit_idx], ...)          # aligns again
```

iii. Not discussed in CONVERSION_NOTES. The session-level quantities that matter for correctness (the video offset and the three discretisation thresholds) are computed once per session from pooled data, which is the important non-repetition.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items. (1) `mat73.loadmat` materialises the whole `obj`, including `clu.spkWavs`, `clu.tm`, `sglx` index arrays, and all untracked DLC features. (2) All kinematics and motion energy are computed for **every** raw trial (3,626) and only then subset to the 2,415 kept trials — ~33% of the video work is thrown away. (3) `me["moveThresh"]` is parsed but never used (the median split replaces it). (4) `my_smooth(xy, 1, "reflect")` in `align_feature_positions` is a no-op copied from the reference (`mySmooth` returns immediately for `N <= 1`). (5) `basederiv` is computed even for the tongue, where it is unused. (6) Per-session `thresholds`/`summary` dicts and the `--show-processing` figures are reporting artefacts, not part of the output.

ii.
```python
    xpos = np.full((time_axis.size, n_trials), np.nan, dtype=np.float64)   # n_trials = ALL raw trials
    for trial_idx in range(n_trials):
        ...
```
```python
        xy = ts[:, 0:2, feat_index]
        if "tongue" not in feat_name:
            xy = my_smooth(xy, 1, "reflect")        # no-op: my_smooth returns x.copy() for n <= 1
```
```python
    return {"data": np.atleast_1d(me.data), "moveThresh": float(me.moveThresh)}   # moveThresh unused
```

iii. Step 6 acknowledges (1): loading full objects "is simpler and reliable but not maximally lean", and claims a mitigation for the feature set: "Only the exact kinematic features needed for the requested outputs are extracted (`tongue`, `top_paw`, `bottom_paw`) instead of the full feature set." Items (2)–(5) are not documented; the AI's position is that with a 72 s total runtime, simplicity and fidelity to the reference MATLAB were worth more than trimming them.
