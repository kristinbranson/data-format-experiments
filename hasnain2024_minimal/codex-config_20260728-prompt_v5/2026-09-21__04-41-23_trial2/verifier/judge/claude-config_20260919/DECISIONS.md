# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not glob the data directory. It hard-codes a list of 12 sessions (`SESSION_SPECS`), each an `{subject, date, probe}` triple, and only ever looks in the single sub-folder `Ephys_Behavior` (`DATA_SUBDIR`). That list is transcribed from the authors' `Scripts/Figure 8/Figure8a_thru_c.m`, which loads `loadJEB6/JEB7/EKH1/EKH3/JGR2/JGR3/JEB19_ALMVideo.m` — the animals used for the paper's context analyses. The sister folder `RandomizedDelay_Ephys_Behavior` (JEB11, JEB12, JEB23, JEB24 — 19 sessions) and the fixed-delay animals JEB13/JEB14/JEB15 (14 sessions) are never opened, even though the AI saw that `load<ANM>_ALMVideo.m` files exist for them (trajectory step 23). Each session is one MATLAB v7.3 file read with `h5py`; the companion `motionEnergy_<tag>.mat` is read with `scipy.io.loadmat`. No `scipy.io` fallback exists for the data structure, because all 12 chosen files are v7.3.

ii.
```python
DATA_SUBDIR = "Ephys_Behavior"
SESSION_SPECS = [
    {"subject": "JEB6", "date": "2021-04-18", "probe": 2},
    {"subject": "JEB7", "date": "2021-04-29", "probe": 1},
    ...
    {"subject": "JEB19", "date": "2023-04-21", "probe": 1},
]
```
```python
def process_session(data_root: Path, spec: dict) -> dict:
    session_tag = f"{spec['subject']}_{spec['date']}"
    session_path = data_root / DATA_SUBDIR / f"data_structure_{session_tag}.mat"
    motion_path = data_root / DATA_SUBDIR / f"motionEnergy_{session_tag}.mat"

    with h5py.File(session_path, "r") as mat:
        beh = load_basic_behavior(mat)
```
```python
def build_dataset(data_root: Path) -> dict:
    sessions = [process_session(data_root, spec) for spec in SESSION_SPECS]
```

iii. From the trajectory: "the decoder targets only make sense for the two-context electrophysiology sessions where both `DR` and `WC` appear" (step 7) and "the authors use a specific hand-written session list rather than 'all files in a folder', so I'm checking those session loader files directly" (step 32). The AI then tabulated DR/WC trial counts for every file in `Ephys_Behavior` (step 70) and confirmed that JEB13/JEB14/JEB15 have 0–51 WC trials (three of them exactly 0) while the 12 chosen sessions all have 57–111. It also counted units per session and concluded (step 75) that "the hand-picked context session list used in the MATLAB scripts lands almost exactly on the paper's reported 522 ALM units after the same quality filtering, which is a strong signal that this is the right session set to export." The final message states: "it uses the 12 hand-picked two-context ALM sessions from the MATLAB context scripts." No justification is ever given for skipping the `RandomizedDelay_Ephys_Behavior` folder — it is simply never examined.

## 1-b. How are the data split into subjects (mice)?

i. The subject is read straight off the hard-coded `spec["subject"]` field (equivalently, the part of the filename before the underscore). `subjects` is built in order of first appearance across `SESSION_SPECS`, and `subject_idx` is that index per session. The result is 7 subjects (EKH1, EKH3, JEB6, JEB7, JEB19, JGR2, JGR3) over 12 sessions.

ii.
```python
subjects = []
subject_to_idx = {}
subject_idx = []
for sess in sessions:
    if sess["subject"] not in subject_to_idx:
        subject_to_idx[sess["subject"]] = len(subjects)
        subjects.append(sess["subject"])
    subject_idx.append(subject_to_idx[sess["subject"]])
```
```python
"subjects": subjects,
"subject_idx": np.asarray(subject_idx, dtype=np.int64),
```

iii. Not discussed explicitly in the trajectory. Implicitly, the animal id is part of the session key the authors' own loading scripts use (`load<ANM>_ALMVideo.m`), which the AI transcribed as `{"subject": ..., "date": ...}`; the AI never reads `obj.meta.anm`, which it inspected (step 53/54) and found unreliable.

## 1-c. How are the data split into sessions?

i. One session = one entry of `SESSION_SPECS` = one `data_structure_<anm>_<date>.mat` file in `Ephys_Behavior`, processed independently by `process_session` and appended as one element of `neural`, `input`, `output`, `brain_region_idx`. Every session uses exactly one probe, taken from the `probe` field transcribed from the authors' meta scripts. There is no two-probe concatenation path. 12 sessions total, 3,116 exported trials.

ii.
```python
sessions = [process_session(data_root, spec) for spec in SESSION_SPECS]
...
"neural": [sess["neural"] for sess in sessions],
"input":  [sess["input"]  for sess in sessions],
"output": [sess["output"] for sess in sessions],
```
```python
clu_ref = mat["obj"]["clu"][()][probe - 1, 0]
clu = mat[clu_ref]
if isinstance(clu, h5py.Dataset):
    raise ValueError(f"Probe {probe} is empty")
```

iii. Same as 1-a: the sessions are the ones in the authors' `Figure8a_thru_c.m` meta list, chosen because they are the two-context (DR + WC) recordings and because the surviving unit count reproduces the paper's reported ALM population.

## 1-d. How are the data split into trials?

i. `obj.bp.Ntrials` defines the trial count, and every per-trial field of `obj.bp` (`R`, `L`, `hit`, `miss`, `no`, `early`, `autowater`, `stim.enable`, `ev.goCue`, `ev.bitStart`, `ev.lickL`, `ev.lickR`) is read as a vector of that length. Spikes carry their own `clu.trial` index and camera data are stored as one cell per trial in `obj.traj{view}`, so trial boundaries are never reconstructed. The AI does not truncate the `bp` vectors to `Ntrials` (it assumes they already have that length); on the 12 selected sessions I verified all these fields are exactly `Ntrials` long and `goCue.size == Ntrials`, so nothing is mis-indexed.

ii.
```python
data = {
    "ntrials": int(read_h5_numeric(bp["Ntrials"])),
    "R": read_h5_numeric(bp["R"]).astype(bool),
    ...
    "goCue": read_h5_numeric(bp["ev"]["goCue"]).astype(np.float64),
}
```
```python
spike_trials = read_trial_ref_numeric(mat, trial_refs[unit_idx]).astype(np.int64)
spike_trialtm = read_trial_ref_numeric(mat, trialtm_refs[unit_idx]).astype(np.float64)
...
spike_trials = spike_trials - 1
```
```python
for trial_idx in range(ntrials):
    bundles[view] = load_trial_video_bundle(mat, traj_groups[view - 1], trial_idx)
```

iii. Not argued in prose. The AI verified the layout empirically before writing the code — it printed `bp` field shapes, `clu.trial`/`clu.trialtm` contents, and the per-trial `traj` cell structure (steps 50, 51, 59, 60, 82) and confirmed spike trial numbers are 1-based.

## 1-e. How are trials filtered based on quality controls?

i. Exactly two filters, applied together as one boolean mask: early-lick trials (`bp.early`) and photostimulation trials (`bp.stim.enable`) are dropped. Everything else is kept, including `no`-response (ignore) trials. If a session has fewer than two surviving trials the script raises. There is no filter on trials past the end of the recording; I checked and none of the 12 selected sessions needs one (the last trial containing spikes is `Ntrials` in every case). Sessions in which `bp.stim` is absent are treated as having no photostim.

ii.
```python
valid_trials = ~beh["early"] & ~beh["stim_enable"]
valid_idx = np.flatnonzero(valid_trials)
if valid_idx.size < 2:
    raise ValueError(f"{session_tag} has fewer than two valid trials after filtering")
```
```python
if "stim" in bp and isinstance(bp["stim"], h5py.Group) and "enable" in bp["stim"]:
    data["stim_enable"] = read_h5_numeric(bp["stim"]["enable"]).astype(bool)
else:
    data["stim_enable"] = np.zeros((data["ntrials"],), dtype=bool)
```

iii. From the metadata the AI wrote: `"trial_filter": "Kept trials with ~early and ~stim.enable; retained hit, miss, and ignore trials."`, and from the final message: "exports non-early, non-stim trials while keeping correct, incorrect, and ignore outcomes." The `~early & ~stim.enable` conjunction is lifted from the authors' `params.condition` strings, which the AI read in the figure scripts (steps 29–31) and transcribed verbatim into `get_condition_masks`. Ignore trials are kept because the decoder spec requires an `ignore` outcome class and a `none` lick class.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` for the one probe named in `SESSION_SPECS`: the per-cluster cell arrays `trial` (1-based trial of each spike), `trialtm` (spike time relative to trial start, behaviour clock) and `quality` (manual curation label). `obj.bp.ev.goCue` supplies the alignment time. `clu.site` (or `clu.channel` where `site` is absent) is also read, but its value is discarded — `brain_region_idx` is a zero vector because every recording targets ALM.

ii.
```python
clu_ref = mat["obj"]["clu"][()][probe - 1, 0]
clu = mat[clu_ref]
quality_refs = np.asarray(clu["quality"][()]).squeeze()
trial_refs   = np.asarray(clu["trial"][()]).squeeze()
trialtm_refs = np.asarray(clu["trialtm"][()]).squeeze()
if "site" in clu:
    site_refs = np.asarray(clu["site"][()]).squeeze()
elif "channel" in clu:
    site_refs = np.asarray(clu["channel"][()]).squeeze()
```
```python
"brain_regions": ["ALM"],
"brain_region_idx": [sess["brain_region_idx"] for sess in sessions],
```

iii. The AI read `alignSpikes.m` and `getSeq.m` (steps 17, 18) and dumped the raw `clu` fields for one probe (steps 60–61, 82) to establish that `trial` and `trialtm` are the only fields needed. The `site`/`channel` fallback was added after the first run crashed: "some sessions store unit location under `channel` instead of `site`, exactly as the tutorial hints" (step 97).

## 2-b. How is the `neural` data processed?

i. Per unit and per trial: spike times aligned to the go cue are histogrammed into the 10 ms bin grid, divided by `DT` to give spikes/s, and smoothed along time with a re-implementation of the authors' `mySmooth.m` — a `gausswin(15)` (MATLAB alpha = 2.5) whose first half is zeroed to make it **causal**, renormalised to sum to 1, convolved with `mode='same'`, with `'reflect'` boundary handling (prepend the first 15 samples, trim them afterwards). No normalisation, baseline subtraction, or z-scoring. Stored as `float32` Hz.

ii.
```python
def gausswin(n: int, alpha: float = 2.5) -> np.ndarray:
    idx = np.arange(n, dtype=np.float64) - (n - 1.0) / 2.0
    sigma = (n - 1.0) / (2.0 * alpha)
    return np.exp(-0.5 * (idx / sigma) ** 2)

def my_smooth(x, n, bctype):
    ...
    if bctype.lower() == "reflect":
        arr_filt = np.concatenate([arr[:n, :], arr], axis=0)
        trim = n
    ...
    kern = gausswin(n)
    kern[: len(kern) // 2] = 0.0            # causal
    kern = kern / np.sum(kern)
    out = np.zeros_like(arr_filt)
    for col in range(arr_filt.shape[1]):
        out[:, col] = np.convolve(arr_filt[:, col], kern, mode="same")
    out = out[trim:, :]
```
```python
for tr in np.unique(spike_trials):
    spk = aligned_times[spike_trials == tr]
    counts, _ = np.histogram(spk, bins=edges)
    rates = counts.astype(np.float64) / DT
    trialdat[out_idx, tr, :] = my_smooth(rates, SMOOTH, BOUNDARY).astype(np.float32)
```

iii. The AI opened `utils/mySmooth.m` (step 79) specifically to port it, and `getSeq.m` (step 18) for the `N./params.dt` → `mySmooth(...)` order. `SMOOTH = 15` and `BOUNDARY = "reflect"` are `params.smooth = 15` and `params.bctype = 'reflect'` from `Figure8a_thru_c.m`. Metadata: "smoothed with the paper's causal Gaussian filter (window 15, reflect boundary)."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. (1) The manual curation label `clu.quality` is stripped, lower-cased, and rejected if it is one of `garbage`, `gabrga`, `noisy`, `real?`. Everything else — including `poor`, `multi`, `fair` and unlabelled clusters — is kept. (2) A firing-rate criterion reproducing `removeLowFRClusters.m`: condition PSTHs are built for the seven `params.condition` masks of `Figure8a_thru_c.m` by averaging `trialdat` over the trials of each condition, then the mean over time *and* conditions must exceed 1 Hz. The result is 520 units over 12 sessions (27–67 per session).

ii.
```python
def quality_is_usable(label: str) -> bool:
    label = label.strip().lower()
    return label not in {"garbage", "gabrga", "noisy", "real?"}
```
```python
def get_condition_masks(beh):
    return [
        hit | miss | no,
        hit & ~stim_enable & ~autowater,
        hit & ~stim_enable & autowater,
        miss & ~stim_enable & ~autowater,
        miss & ~stim_enable & autowater,
        hit & ~stim_enable & ~autowater & ~early,
        hit & ~stim_enable & autowater & ~early,
    ]
```
```python
psth_by_cond = []
for mask in condition_masks:
    psth = trialdat[:, mask, :].mean(axis=1) if np.any(mask) else np.zeros(...)
    psth_by_cond.append(psth)
psth_by_cond = np.stack(psth_by_cond, axis=2)

mean_frs = psth_by_cond.mean(axis=(1, 2))
use = mean_frs > 1.0
return trialdat[use], sites[use]
```

iii. The AI read `findClusters.m` (step 66) and copied its `all`-quality branch exactly — `~ismember('garbage') & ~ismember('gabrga') & ~ismember('noisy') & ~ismember('real?')` — matching `params.quality = {'all'}` in the figure scripts. It lower-cases because it had enumerated the labels per session (step 74) and seen both `'Poor'`/`'poor'` and `'Multi'`/`'multi'` casings. It read `removeLowFRClusters.m` (step 65) and reproduced `meanFRs = mean(mean(obj.psth{prb},3,'omitnan'),'omitnan')` as a mean over the time and condition axes, with `params.lowFR = 1`. Step 75: the resulting unit count "lands almost exactly on the paper's reported 522 ALM units."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. A single subtraction, done once per cluster and vectorised over that cluster's spikes: `trialtm − goCue[trial of that spike]`. `clu.trialtm` is already on the behaviour clock and already relative to its own trial's start, and `bp.ev.goCue` is on the same clock, so no offset or interpolation is needed. The aligned times are then histogrammed into `edges`, which run from −3.0 to +2.5 s relative to the go cue.

ii.
```python
spike_trials = spike_trials - 1
aligned_times = spike_trialtm - go_cue[spike_trials]
for tr in np.unique(spike_trials):
    spk = aligned_times[spike_trials == tr]
    counts, _ = np.histogram(spk, bins=edges)
```

iii. Directly from `alignSpikes.m`, which the AI read at step 17: `obj.clu{prb}(clu).trialtm_aligned = obj.clu{prb}(clu).trialtm - event`, with `params.alignEvent = 'goCue'` in every figure script it inspected. The AI also confirmed empirically (steps 76, 77) that `bp.ev.goCue` is populated (2.5 s) on WC/autowater trials too, so the same alignment applies in both contexts. Metadata: `"temporal_alignment_event": "goCue onset"`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 10 ms bins spanning −3.0 to +2.5 s from the go cue: 550 bins per trial, identical for every trial and session. The grid is built once by `get_edges()` / `get_time_axis()` and is shared by the neural data, the time input, and all three video streams, so no rebinning or resampling between streams is ever needed. Spikes are histogrammed directly at 10 ms; video (which is sampled at ~400 Hz) is **linearly interpolated** onto the 10 ms bin centres rather than averaged into bins. `time_bin_size` is reported as 10.0 ms, `off_start = -3.0`, `off_end = 2.5`.

Consequence worth noting: the camera's first frame sits at about −2.47 s from the go cue, so the leading ~53 bins (−3.0 to −2.47 s, 9.6 % of every trial) contain no video at all and come out as the trailing class for paw velocity and motion energy on 100 % of trials. Checking the produced pickle, `paw_velocity == 2` on 9.99 % of bins and `motion_energy == 2` on 9.96 % of bins — i.e. essentially the whole "not visible"/"no video" class is this structural pre-video block, not genuine tracking or video failure. Neural data is unaffected (`clu.trialtm` does extend before the nominal trial start; mean rate in the first bins is ~3–5 Hz, not zero).

ii.
```python
TIME_MIN = -3.0
TIME_MAX = 2.5
DT = 1.0 / 100.0

def get_edges() -> np.ndarray:
    return np.arange(TIME_MIN, TIME_MAX + DT, DT, dtype=np.float64)

def get_time_axis() -> np.ndarray:
    edges = np.arange(TIME_MIN, TIME_MAX + DT, DT, dtype=np.float64)
    return edges[:-1] + DT / 2.0
```
```python
"time_bin_size": float(DT * 1000.0),
"off_start": float(TIME_MIN),
"off_end": float(TIME_MAX),
```

iii. `params.tmin = -3`, `params.tmax = 2.5`, `params.dt = 1/100` are copied verbatim from `Scripts/Figure 8/Figure8a_thru_c.m` (step 29) — the same script the session list came from. `get_time_axis` reproduces `getSeq.m`'s `obj.time = edges + params.dt/2; obj.time = obj.time(1:end-1)`. The AI did not comment on the empty pre-video window.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Nothing from the raw files — it is the centre of each of the 550 bins of the analysis grid, which is itself defined relative to `bp.ev.goCue` by the alignment in 2-d. `params.advance_movement = 0` is carried as `ADVANCE_MOVEMENT` and added to the axis (a no-op). The same 1×550 vector is used for every trial of every session.

ii.
```python
INPUT_NAMES / "input_names": ["time_from_go_cue_s"]
```
```python
taxis = get_time_axis().astype(np.float32)
input_template = taxis[None, :]
```

iii. Defined by the AI to satisfy the decoder spec ("Time from go cue onset in seconds"); the grid itself comes from the authors' `params.tmin/tmax/dt`. Not otherwise discussed.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None. `get_time_axis()` returns `edges[:-1] + DT/2`, cast to `float32`, and a `.copy()` of the same `(1, 550)` row is appended for each trial. Range is −2.995 to +2.495 s.

ii.
```python
for kept_trial_pos, trial_idx in enumerate(valid_idx):
    ...
    input_trials.append(input_template.copy())
```

iii. N/A — the axis is defined by construction.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid. `get_edges()` supplies the histogram edges for the spikes and `get_time_axis()` returns the centres of those same edges, so input element *k* and neural column *k* are the same 10 ms interval by construction. The video streams are interpolated onto the identical axis, so all four streams share one time base.

ii.
```python
edges = get_edges()                      # used to histogram spikes
...
counts, _ = np.histogram(spk, bins=edges)
```
```python
taxis = get_time_axis() + ADVANCE_MOVEMENT   # used for input and for all video interpolation
```

iii. N/A.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The recorded lick times themselves: `obj.bp.ev.lickL` and `obj.bp.ev.lickR`, one cell of timestamps per trial, together with `bp.ev.goCue`. The instructed side (`R`/`L`) and the outcome flags are **not** used to infer direction, although `R` is loaded for the condition masks.

ii.
```python
data["lickL_refs"] = np.asarray(bp["ev"]["lickL"][()]).squeeze()
data["lickR_refs"] = np.asarray(bp["ev"]["lickR"][()]).squeeze()
```
```python
lick_dir = first_post_go_lick_direction(mat, beh["lickL_refs"], beh["lickR_refs"], beh["goCue"])
```

iii. The AI dumped the lick cells for `Rmiss`/`Lmiss`/`Rhit`/`Lhit`/`no` trials (step 78) and confirmed the first post-go lick is on the instructed port for hits, on the opposite port for misses, and absent for `no` trials — i.e. it cross-validated the direct measurement against the `hit`/`miss`/`R` inference before choosing the direct one. Final message: "lick direction is the first post-go lick."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. For each trial, take all left and all right lick times at or after the go cue (with a 1 ns tolerance), compare the earliest of each, and label the trial by whichever side came first: left = 0, right = 1. Trials with no lick on either port after the go cue get `none` = 2. The label is constant across the 550 bins.

ii.
```python
def first_post_go_lick_direction(mat, lick_l_refs, lick_r_refs, go_cue):
    lick_dir = np.full((go_cue.size,), 2, dtype=np.int64)
    for trial_idx in range(go_cue.size):
        lick_l = read_trial_ref_numeric(mat, lick_l_refs[trial_idx]).astype(np.float64)
        lick_r = read_trial_ref_numeric(mat, lick_r_refs[trial_idx]).astype(np.float64)
        post_l = lick_l[lick_l >= go_cue[trial_idx] - 1e-9]
        post_r = lick_r[lick_r >= go_cue[trial_idx] - 1e-9]
        first_l = post_l[0] if post_l.size else np.inf
        first_r = post_r[0] if post_r.size else np.inf
        if first_l < first_r:
            lick_dir[trial_idx] = 0
        elif first_r < first_l:
            lick_dir[trial_idx] = 1
    return lick_dir
```
```python
"output_values": [["left", "right", "none"], ...]
...
out[0, :] = lick_dir[trial_idx]
```

iii. Same as 4-a — the AI verified the rule reproduces the expected hit/miss pattern on real trials before adopting it. Across the exported data `none` occurs on 22.3 % of trials against 22.5 % ignore trials, so the two definitions nearly but not exactly coincide.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial flag, `obj.bp.autowater`, which marks the water-cued block in which reward is delivered at a random port with no auditory cues.

ii.
```python
"autowater": read_h5_numeric(bp["autowater"]).astype(bool),
```

iii. The AI read the methods text describing the DR/WC block structure (step 6) and confirmed empirically (steps 76–77) that `autowater` trials are the ones with WC structure, and tabulated DR/WC counts per session from `~autowater`/`autowater` (step 70). The paper's own `params.condition` strings use `autowater` the same way.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling with no further processing: `autowater → WC = 0`, everything else `→ DR = 1`. Constant across the 550 bins of a trial.

ii.
```python
context = np.where(beh["autowater"], 0, 1).astype(np.int64)
...
out[1, :] = context[trial_idx]
```
```python
"output_values": [..., ["WC", "DR"], ...]
```

iii. Code order follows the decoder spec's "(WC, DR)". Not further discussed.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The three mutually exclusive per-trial flags `obj.bp.hit`, `obj.bp.miss` and `obj.bp.no`. `no` is loaded (and used in the `hit|miss|no` condition mask) but the ignore class is actually assigned as the default, so `no` is not strictly required for the outcome itself.

ii.
```python
"hit":  read_h5_numeric(bp["hit"]).astype(bool),
"miss": read_h5_numeric(bp["miss"]).astype(bool),
"no":   read_h5_numeric(bp["no"]).astype(bool),
```

iii. The AI read `funcs/getOutcome.m` (step 62) and the figure scripts' condition strings, which use exactly `hit`, `miss`, `no`. Final message: "outcome comes from `hit/miss/no`."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Initialise everything to `ignore = 2`, then set `miss → incorrect = 0` and `hit → correct = 1`. Constant across the 550 bins.

ii.
```python
outcome = np.full((beh["ntrials"],), 2, dtype=np.int64)
outcome[beh["miss"]] = 0
outcome[beh["hit"]] = 1
...
out[2, :] = outcome[trial_idx]
```
```python
"output_values": [..., ["incorrect", "correct", "ignore"], ...]
```

iii. Codes follow the decoder spec's "(incorrect, correct, ignore)". Ignore trials are retained rather than dropped (as the paper does) so that all six outputs keep the same trial set.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj{view}`: `ts` (frames × [x, y, likelihood] × features), `featNames`, `frameTimes`, and `NdroppedFrames`, for **seven** tongue features across **both** cameras — `tongue`, `left_tongue`, `right_tongue` on the side view and `top_tongue`, `topleft_tongue`, `bottom_tongue`, `bottomleft_tongue` on the bottom view. Features not present in a session's `featNames` are silently skipped. `obj.sglx.fs`, `obj.sglx.bitcode.bitstart`, `obj.bp.ev.bitStart` and `bp.ev.goCue` are also needed, for the clock correction. Only the x and y columns of `ts` are used; the likelihood column is never read, because the authors already set x/y to NaN where the likelihood is low.

ii.
```python
TONGUE_FEATURES = [
    (1, "tongue"), (1, "left_tongue"), (1, "right_tongue"),
    (2, "top_tongue"), (2, "topleft_tongue"),
    (2, "bottom_tongue"), (2, "bottomleft_tongue"),
]
```
```python
def load_trial_ts(mat, ref):
    ts = np.asarray(mat[ref][()])
    return np.transpose(ts, (2, 1, 0))       # -> (frames, xyl, features)
...
x = ts[:, 0, feat_idx]
y = ts[:, 1, feat_idx]
```
```python
tongue_speed, tongue_visible = compute_composite_speed(mat, beh, TONGUE_FEATURES)
```

iii. The feature list is `params.traj_features` from the authors' figure scripts (the union of the two views' tongue entries, steps 29–31). The AI dumped the `traj` structure per view to confirm the `ts` ordering and the `featNames` (step 59), and wrote `load_trial_ts` to undo h5py's axis reversal.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Per feature and per trial, a direct port of the authors' `findPosition.m` + `findVelocity.m`: (1) skip the trial if `NdroppedFrames` is NaN; (2) convert frame times to go-cue time (see 7-d); (3) **linearly interpolate** x and y onto the 550-bin axis, with NaN outside the frame range and NaN propagating through untracked stretches — no smoothing, because the authors explicitly do not smooth tongue features; (4) `np.gradient` of the interpolated x and y (per-sample, so units are pixels per 10 ms bin), with NaN velocities set to 0 exactly as `findVelocity.m` does for the tongue; (5) speed = `hypot(xvel, yvel)`. Visibility per feature is recorded separately as "interpolated x and y are both finite".

The composite tongue velocity is then the **unweighted mean of the speeds of whichever of the seven features are visible in that bin**, and the composite is visible if at least one feature is. No per-camera or per-feature rescaling is applied before averaging, even though the two camera views are on different pixel scales.

ii.
```python
x_interp = interp_with_nans(old_t, x, taxis)
y_interp = interp_with_nans(old_t, y, taxis)
visible = np.isfinite(x_interp) & np.isfinite(y_interp)

is_tongue = "tongue" in feat_name
x_proc, y_proc = x_interp.copy(), y_interp.copy()
if not is_tongue:
    x_proc = fill_nearest_1d(x_proc)
    y_proc = fill_nearest_1d(y_proc)

xvel, yvel = compute_velocity(x_proc, y_proc, is_tongue=is_tongue)
speed = np.sqrt(xvel**2 + yvel**2)
```
```python
def compute_velocity(xpos, ypos, is_tongue):
    xvel = np.gradient(xpos)
    yvel = np.gradient(ypos)
    if not is_tongue:
        ...
    else:
        xvel[np.isnan(xvel)] = 0.0
        yvel[np.isnan(yvel)] = 0.0
    return xvel, yvel
```
```python
visible_counts = per_feature_visible.sum(axis=0)
speed_sum = np.where(per_feature_visible, np.nan_to_num(per_feature_speed), 0.0).sum(axis=0)
composite_speed = np.divide(speed_sum, np.maximum(visible_counts, 1), ...)
composite_visible = visible_counts > 0
```

iii. The AI read `getKinematicsFromVideo.m`, `findPosition.m` and `findVelocity.m` (steps 56–58) and reproduced each branch, including "if tongue, don't smooth" and "set tongue velocity to 0 if not visible". It even probed scipy's `interp1d` NaN behaviour against MATLAB's `interp1` (step 85) to make sure NaNs propagate rather than bridge gaps. The averaging of multiple features into one scalar is the AI's own addition (the paper keeps features separate and z-scores each before PCA); it is not justified anywhere in the trajectory. Final message: "tongue/paw/motion signals use the same video alignment and feature-velocity logic as the MATLAB code."

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per session: the 50th percentile is taken over the composite speed **of the visible bins only**, pooled across all retained trials and all 550 bins. Visible bins at or above it become 1, visible bins below it become 0, and every non-visible bin becomes 2. The thresholds are stored in the metadata per session. In the produced file the classes come out as 4.767 % / 4.767 % / 90.47 %.

ii.
```python
def discretize_with_visibility(values, visible, keep_mask, absent_code):
    if values is None or visible is None:
        classes = np.full((int(np.sum(keep_mask)), get_time_axis().size), absent_code, dtype=np.int64)
        return classes, None
    keep_values = values[keep_mask]
    keep_visible = visible[keep_mask]
    threshold = float(np.nanpercentile(keep_values[keep_visible], 50)) if np.any(keep_visible) else None
    classes = np.full(keep_values.shape, absent_code, dtype=np.int64)
    if threshold is not None:
        present = keep_visible
        classes[present] = (keep_values[present] >= threshold).astype(np.int64)
    return classes, threshold
```
```python
"output_values": [..., ["<50th_percentile", ">=50th_percentile", "not_visible"], ...]
```

iii. Straight from the decoder spec ("0: < 50th percentile, 1: >= 50th percentile, 2: not visible", "per-session threshold"). Taking the percentile over visible bins only is what makes the two visible classes come out balanced; the AI logs the per-session thresholds in `source_session_info` so the choice is auditable.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A per-session video-clock offset is computed once from the bitcode that both streams record — `mode(sglx.bitcode.bitstart) / sglx.fs − mode(bp.ev.bitStart)`, a direct port of `findVideoOffset.m` — and each trial's frame times become `frameTimes − vidshift − goCue[trial]`. The result is linearly interpolated onto the same 550-bin axis the spikes were histogrammed into, so bin *k* is the same interval in both streams. Frames outside the window yield NaN and hence the `not visible` class. The offset is recomputed (identically) inside each of the three video passes rather than cached.

ii.
```python
def get_video_offset(mat, bit_start):
    fs = float(read_h5_numeric(mat["obj"]["sglx"]["fs"]))
    bitcode_starts = read_h5_numeric(mat["obj"]["sglx"]["bitcode"]["bitstart"]).astype(np.float64)
    return matlab_mode(bitcode_starts) / fs - matlab_mode(bit_start)
```
```python
old_t = frame_times - vidshift - align_time
x_interp = interp_with_nans(old_t, x, taxis)
```

iii. `findVideoOffset.m`, read at step 21, and reproduced including MATLAB's `mode` semantics via the hand-written `matlab_mode`. The AI checked the value numerically on one session (step 84: `vidshift = 0.49004` s) before building it into the script.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The same `obj.traj` tracking, restricted to the bottom camera (view 2) and to **both** paw features, `top_paw` and `bottom_paw`. Same auxiliary fields as the tongue (`frameTimes`, `NdroppedFrames`, `sglx` bitcode, `goCue`).

ii.
```python
PAW_FEATURES = [
    (2, "top_paw"),
    (2, "bottom_paw"),
]
...
paw_speed, paw_visible = compute_composite_speed(mat, beh, PAW_FEATURES)
```

iii. `top_paw` and `bottom_paw` are the view-2 paw entries of `params.traj_features` in the authors' `Figure1e.m` list (step 30). No explicit justification for using both; because the two are on the same camera they share a pixel scale, and the composite falls back to whichever is tracked.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The same pipeline as the tongue but through the **non-tongue** branch of the authors' code, which differs in three places: (1) after interpolation, gaps in x and y are filled with the nearest valid sample (`fillmissing(...,'nearest')` in `findPosition.m`); (2) after `np.gradient`, a baseline derivative `basederiv = median(diff(x))` is subtracted from *both* `xvel` and `yvel` — faithfully reproducing the authors' `xvel - basederiv(1)` / `yvel - basederiv(1)` including the fact that the x-baseline is applied to y; (3) the velocities are again nearest-filled. Speed is `hypot(xvel, yvel)` in pixels per 10 ms bin, and the two paw features are averaged over whichever are visible. Visibility is still taken from the pre-fill interpolated positions, so the filled stretches are labelled `not visible` and never enter the threshold.

ii.
```python
def compute_velocity(xpos, ypos, is_tongue):
    xvel = np.gradient(xpos)
    yvel = np.gradient(ypos)
    if not is_tongue:
        if np.any(np.isfinite(xpos)):
            basederiv_x = np.nanmedian(np.diff(np.column_stack([xpos, ypos]), axis=0)[:, 0])
            if not np.isfinite(basederiv_x):
                basederiv_x = 0.0
        else:
            basederiv_x = 0.0
        xvel = xvel - basederiv_x
        yvel = yvel - basederiv_x
        xvel = fill_nearest_1d(xvel)
        yvel = fill_nearest_1d(yvel)
```
```python
def fill_nearest_1d(x):
    ...
    use_prev = (prev_idx >= 0) & ((next_idx == out.size) | ((idx - prev_idx) <= (next_idx - idx)))
    nearest = np.where(use_prev, prev_idx, next_idx)
```

iii. A line-by-line port of `findVelocity.m` and `findPosition.m` (steps 57, 58), including the `basederiv(1)` quirk. `fill_nearest_1d` is a hand-written `fillmissing(...,'nearest')`. The AI added the `np.any(np.isfinite(xpos))` guard after the first run hit an all-NaN trial: "guarding the all-NaN velocity edge case before rerunning end to end" (step 97).

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Identically to the tongue: `discretize_with_visibility(paw_speed, paw_visible, valid_trials, absent_code=2)` — 50th percentile of the visible bins pooled over the whole session, `>=` → 1, `<` → 0, non-visible → 2. Produced fractions are 44.75 % / 44.76 % / 10.49 %. Note that ~96 % of the class-2 bins are the structural pre-video window described in 2-e, not genuine tracking failure.

ii.
```python
paw_classes, paw_threshold = discretize_with_visibility(
    paw_speed, paw_visible, valid_trials, absent_code=2
)
...
out[4, :] = paw_classes[kept_trial_pos]
```
```python
"output_values": [..., ["<50th_percentile", ">=50th_percentile", "not_visible"], ...]
```

iii. Straight from the decoder spec; the per-session threshold is logged in `source_session_info`.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue and to every other video stream: `frameTimes − vidshift − goCue[trial]`, then linear interpolation onto the shared 550-bin axis. Because the paw is only on view 2, its own camera's `frameTimes` are used (the code fetches a bundle per requested view, so the frame times always come from the camera that carries the feature).

ii.
```python
views_needed = sorted({view for view, _ in feature_specs})
for trial_idx in range(ntrials):
    for view in views_needed:
        bundles[view] = load_trial_video_bundle(mat, traj_groups[view - 1], trial_idx)
    align_time = beh["goCue"][trial_idx]
    for feat_out_idx, (view, feat_name) in enumerate(feature_specs):
        ts, frame_times = bundles[view]
        ...
        old_t = frame_times - vidshift - align_time
```

iii. Same source as 7-d (`findVideoOffset.m`, `findPosition.m`); the paw needs no special treatment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The separate `motionEnergy_<anm>_<date>.mat` file beside the data structure, read with `scipy.io.loadmat`; `me.data` is a cell array with one trace per trial, one value per camera frame. The side camera's `frameTimes` (`obj.traj{1}`) supply the time base. `obj.me` inside the data structure is not used. If the motion-energy file is missing, the whole output is set to the `no_video` class.

ii.
```python
def load_motion_energy_aligned(mat, motion_path, beh):
    if not motion_path.exists():
        return None, None
    motion_file = sio.loadmat(motion_path, squeeze_me=True, struct_as_record=False)
    me = motion_file["me"]
    motion_trials = np.asarray(me.data, dtype=object).reshape(-1)
```

iii. The AI read `loadMotionEnergy.m` (step 19) and inspected the file layout directly (steps 67, 68) to confirm `me.data` is a per-trial cell of per-frame values and `me.moveThresh` is a scalar it does not need. Note it did **not** port `loadMotionEnergy.m`'s `if isstruct(me.data), me.data = me.data.data; end` guard; none of the 12 chosen files needs it, but the omission would break on the doubly-wrapped files that exist elsewhere in the dataset.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond alignment and resampling — the value is already one scalar per frame (the paper reduces each frame to the 99th percentile across pixels of the frame-difference image upstream). The trace is linearly interpolated onto the 550-bin axis, visibility is recorded as "interpolated value is finite", and then `fill_nearest_1d` is applied to the interpolated trace (reproducing `fillmissing(me.data,'nearest')` in `loadMotionEnergy.m`). The fill has no effect on the result, because the threshold and the class assignment both use only the bins that were visible before filling.

ii.
```python
old_t = frame_times - vidshift - beh["goCue"][trial_idx]
trial_aligned = interp_with_nans(old_t, trial_motion, taxis)
visible[trial_idx, :] = np.isfinite(trial_aligned)
if np.any(visible[trial_idx, :]):
    aligned[trial_idx, :] = fill_nearest_1d(trial_aligned).astype(np.float32)
```

iii. A port of `loadMotionEnergy.m`'s `interp1(obj.traj{1}(trix).frameTimes - vidshift - alignTimes(trix), me.data{trix}, taxis)` followed by `fillmissing(...,'nearest')` (step 19). The separate visibility mask is the AI's addition, needed because the decoder spec requires a `no video` class that the paper's fill would have erased.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The same `discretize_with_visibility` call: per-session 50th percentile over the visible bins, `>=` → 1, `<` → 0, not-visible → 2 (labelled `no_video`). Produced fractions are 45.01 % / 45.22 % / 9.77 %. As with the paw, essentially all of the class-2 mass is the −3.0 to −2.47 s pre-video window rather than sessions or trials that genuinely lack video.

ii.
```python
motion_classes, motion_threshold = discretize_with_visibility(
    motion_energy, motion_visible, valid_trials, absent_code=2
)
...
out[5, :] = motion_classes[kept_trial_pos]
```
```python
"output_values": [..., ["<50th_percentile", ">=50th_percentile", "no_video"]]
```

iii. Straight from the decoder spec. `me.moveThresh` (the authors' own movement threshold) was deliberately not used, since the spec asks for a median split.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same offset and same grid as the tracking: motion energy has exactly one value per frame of the **side** camera, so `traj_groups[0]`'s `frameTimes` are used, corrected by the session `vidshift` and the trial's `goCue`, then linearly interpolated onto the shared 550-bin axis. If the side camera's bundle is unavailable for a trial (NaN `NdroppedFrames`) the whole trial stays NaN and becomes `no_video`.

ii.
```python
for trial_idx in range(ntrials):
    bundle = load_trial_video_bundle(mat, traj_groups[0], trial_idx)
    ts, frame_times = bundle
    if ts is None or frame_times is None:
        continue
    ...
    old_t = frame_times - vidshift - beh["goCue"][trial_idx]
    trial_aligned = interp_with_nans(old_t, trial_motion, taxis)
```

iii. `loadMotionEnergy.m` uses `obj.traj{1}` frame times for exactly this reason (step 19); the AI copied the choice of view 1.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Five cases, all handled by keeping the trial and degrading gracefully rather than dropping anything:

- **Video marked bad for a trial** — `NdroppedFrames` is NaN: the bundle is `None` and that trial/view contributes nothing, so its features come out `not visible` for all 550 bins. This is `findPosition.m`'s own guard. (It never fires on the 12 selected sessions.)
- **Missing/all-NaN `frameTimes`** — a synthetic 400 Hz clock `(arange(n_frames)+1)/400` is substituted. This reproduces `findPosition.m`'s fallback in spirit, but it fabricates an alignment: after subtracting `vidshift` and `goCue`, those frames land at roughly −3.0 to −0.5 s regardless of when they were actually recorded. It fires on only 4 of 7,252 trial-view pairs in the exported data.
- **Untracked frames** — DeepLabCut's low-likelihood samples are already NaN in `ts`; interpolation propagates the NaN, `visible` is False, and the bin gets the trailing class. For non-tongue features the position/velocity are nearest-filled (as in the MATLAB) but the pre-fill visibility mask is what decides the class, so no fabricated value is ever thresholded.
- **Missing features** — `feat_map.get(feat_name)` returns `None` and the feature is skipped, so sessions tracking fewer tongue/paw features still work.
- **Missing `clu.site`** — falls back to `clu.channel`, then to zeros; missing motion-energy file → the whole `motion_energy` output becomes `no_video`; missing `bp.stim` → treated as no photostim. Empty/null HDF5 references return empty arrays.

No NaN ever reaches the pickle (the verifier reported no warnings).

ii.
```python
def load_trial_video_bundle(mat, traj_group, trial_idx):
    nd = read_trial_ref_numeric(mat, traj_group["NdroppedFrames"][trial_idx, 0])
    if np.size(nd) == 1 and np.isnan(float(nd)):
        return None, None
    ts = load_trial_ts(mat, traj_group["ts"][trial_idx, 0])
    frame_times = read_trial_ref_numeric(mat, traj_group["frameTimes"][trial_idx, 0]).astype(np.float64)
    if frame_times.size == 0 or not np.any(np.isfinite(frame_times)):
        frame_times = (np.arange(ts.shape[0], dtype=np.float64) + 1.0) / 400.0
    return ts, frame_times
```
```python
def interp_with_nans(old_t, y, new_t):
    if old_t.size < 2 or y.size < 2:
        return np.full(new_t.shape, np.nan, dtype=np.float64)
    finite_t = np.isfinite(old_t)
    old_t, y = old_t[finite_t], y[finite_t]
    if old_t.size < 2:
        return np.full(new_t.shape, np.nan, dtype=np.float64)
```
```python
if "site" in clu:   ...
elif "channel" in clu:  ...
else: site_refs = None
```

iii. The guards were mostly transcribed from the authors' MATLAB (`NdroppedFrames`, the 400 Hz frame-time fallback, `fillmissing`). Two were added reactively after the first run failed: the `channel` fallback and the all-NaN `basederiv` guard (step 97: "some sessions store unit location under `channel` instead of `site` ... I'm patching that and also guarding the all-NaN velocity edge case"). The decision to represent missing video as its own class rather than interpolate across it is forced by the decoder spec's `not visible` / `no video` categories.

## 11-a. What are the most time-consuming steps of the code?

i. Reading the per-trial video arrays out of the HDF5 file. Profiling one session (JEB6, 382 trials) gives 5.9 s total, of which `load_trial_video_bundle` accounts for 4.0 s (68 %), essentially all of it `h5py` dataset reads of `traj{view}(trial).ts` (10,132 `h5py` `__getitem__` calls, 3.2 s). `load_motion_energy_aligned` adds 1.0 s and `load_neural_session` (histogram + `my_smooth` over units × trials) 0.85 s. The whole 12-session conversion runs in ~55 s wall clock.

ii.
```python
def load_trial_video_bundle(mat, traj_group, trial_idx):
    ...
    ts = load_trial_ts(mat, ts_ref)      # full (frames, 3, features) array read per call
```
```python
def load_trial_ts(mat, ref):
    ts = np.asarray(mat[ref][()])
    return np.transpose(ts, (2, 1, 0))
```

iii. Never discussed in the trajectory. The AI noted only that the run was slow because "the script is rebuilding smoothed trial-by-trial firing rates and aligned video features session by session rather than relying on pre-exported arrays" (step 94) — which the profile does not bear out; the cost is video I/O, not the neural computation.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Four.
- The nested `for unit: for tr in np.unique(spike_trials): np.histogram(...)` in `load_neural_session` — this is ~11,300 one-dimensional histograms per session where a single `np.histogram2d(spike_trial, aligned_times, bins=[trial_edges, edges])` would do the whole unit at once.
- `my_smooth`'s `for col in range(arr_filt.shape[1])` — it is always called with a single column here, but even so the whole `(trials, bins)` block could be filtered in one `scipy.ndimage.convolve1d`.
- The `for trial_idx in range(ntrials)` / `for feat_out_idx, ...` double loop in `compute_composite_speed`, and the matching one in `load_motion_energy_aligned`. These are harder to vectorise because each trial has a different number of frames, but the interpolation of all seven features of a view could be done in one `interp1d` call with a 2-D `y` instead of 14 separate calls.
- `first_post_go_lick_direction`'s per-trial loop, which is unavoidable given the ragged cell arrays but is cheap.

ii.
```python
for out_idx, unit_idx in enumerate(keep_unit_indices):
    ...
    for tr in np.unique(spike_trials):
        spk = aligned_times[spike_trials == tr]
        counts, _ = np.histogram(spk, bins=edges)
        rates = counts.astype(np.float64) / DT
        trialdat[out_idx, tr, :] = my_smooth(rates, SMOOTH, BOUNDARY).astype(np.float32)
```
```python
x_interp = interp_with_nans(old_t, x, taxis)
y_interp = interp_with_nans(old_t, y, taxis)
```

iii. Not discussed. The neural loop is also quadratic in a hidden way — `aligned_times[spike_trials == tr]` rescans the whole spike vector once per trial — but at 14 % of runtime it is not what dominates.

## 11-c. What processing does the code repeat multiple times?

i. Substantial repetition, all of it in the video path:
- `load_trial_video_bundle` is called **four times per trial**: views 1 and 2 for the tongue pass, view 2 again for the paw pass, view 1 again for the motion-energy pass. Each call re-reads the full `ts` array from disk. Halving this (one read per view per trial) would cut roughly a third off total runtime.
- `get_video_offset` — a session constant — is recomputed three times per session, each time re-reading `sglx.fs`, `sglx.bitcode.bitstart` and `bp.ev.bitStart` and re-running `matlab_mode` on them.
- `load_traj_groups` is called three times per session, and `get_feature_maps` (which scans trials until it finds a non-empty `featNames`) twice.
- `get_time_axis()` / `get_edges()` rebuild the same `np.arange` on every call (several thousand per session, including once inside `discretize_with_visibility`).
- `fill_nearest_1d` is applied twice along the paw path: once to the positions in `compute_composite_speed` and again to the velocities inside `compute_velocity`.

ii.
```python
tongue_speed, tongue_visible = compute_composite_speed(mat, beh, TONGUE_FEATURES)   # reads views 1 and 2
paw_speed, paw_visible       = compute_composite_speed(mat, beh, PAW_FEATURES)      # reads view 2 again
motion_energy, motion_visible = load_motion_energy_aligned(mat, motion_path, beh)   # reads view 1 again
```
```python
def compute_composite_speed(mat, beh, feature_specs):
    taxis = get_time_axis() + ADVANCE_MOVEMENT
    vidshift = get_video_offset(mat, beh["bitStart"])     # recomputed per call
    traj_groups = load_traj_groups(mat)                   # reloaded per call
    feature_maps = get_feature_maps(mat, traj_groups, ntrials)
```

iii. Not discussed. The structure follows from treating the three outputs as three independent passes over the session rather than one pass that extracts all features at once.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small items, none expensive:
- `clu.site` / `clu.channel` is read for every kept unit, filtered by the firing-rate mask, returned from `load_neural_session` — and then thrown away. `brain_region_idx` is a hard-coded zero vector because every recording is ALM.
- `bp.L`, `bp.ev.sample` and `bp.ev.delay` are loaded and never used; `bp.no` is loaded but the ignore class is assigned as the default so it is redundant; `bitStart` is used for the offset and then again to compute a `bit_start_offset_s` metadata field that nothing consumes.
- `fill_nearest_1d` on the aligned motion energy — the filled values are never read, since both the percentile and the class assignment use only the pre-fill visible bins.
- Seven condition PSTHs are built purely to feed a scalar mean-firing-rate criterion; conditions 2/6 and 3/7 differ only by `~early` and contribute almost the same information.
- Firing rates and video features are computed for **all** trials including the early-lick and photostim trials that are dropped immediately afterwards (necessary for the PSTH conditions, but not for the video streams, where ~9 % of trials are computed and discarded).
- The likelihood column of `ts` is transposed into memory on every read and never used.

ii.
```python
sites[out_idx] = int(read_trial_ref_numeric(mat, site_refs[unit_idx]))
...
return trialdat[use], sites[use]
```
```python
neural_all_trials, sites = load_neural_session(...)   # `sites` never referenced again
...
"brain_region_idx": np.zeros((valid_neural.shape[0],), dtype=np.int64),
```
```python
if np.any(visible[trial_idx, :]):
    aligned[trial_idx, :] = fill_nearest_1d(trial_aligned).astype(np.float32)   # fill unused
```

iii. Not discussed. The `sites` plumbing looks like an anticipated depth/region annotation that the single-region `brain_regions = ["ALM"]` decision made unnecessary.
