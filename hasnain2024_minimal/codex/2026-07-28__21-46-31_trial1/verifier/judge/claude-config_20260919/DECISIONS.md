# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a list of 12 sessions (`SESSION_SPECS`), each an `{animal, date, probes}` record, and loads only those. All 12 live in `/app/data/Ephys_Behavior`; the `RandomizedDelay_Ephys_Behavior`, `DelayInhibition_BilatMC_Behavior` and `GoCueInhibition_BilatMC_Behavior` folders are never opened. Each session is one `data_structure_<anm>_<date>.mat`, opened once with `h5py` (v7.3/HDF5 only — there is no `scipy.io` fallback for the v5 files that exist in the other folders), and its motion energy is read from `motionEnergy_<anm>_<date>.mat` with `scipy.io.loadmat`. Fields are pulled out of the HDF5 tree explicitly rather than by a generic recursive reader: `load_behavior` reads the per-trial `bp` flags and `bp.ev` event times, `load_probe_clusters` dereferences `quality`/`trial`/`trialtm` for the one probe named in the spec, and the trajectory cell arrays are dereferenced trial by trial. The 12 sessions were chosen because they are exactly the sessions loaded by the authors' `Scripts/Figure 8/Figure8a_thru_c.m` / `Figure8d.m` (`loadJEB6/JEB7/EKH1/EKH3/JGR2/JGR3/JEB19_ALMVideo`), i.e. the paper's two-context ALM cohort.

ii.
```python
SESSION_SPECS = [
    {"animal": "JEB6", "date": "2021-04-18", "probes": [2]},
    ...
    {"animal": "JEB19", "date": "2023-04-18", "probes": [1]},
]
DATA_DIR = "/app/data"
EPHYS_DIR = os.path.join(DATA_DIR, "Ephys_Behavior")
```
```python
def build_session_payload(spec: dict) -> dict:
    data_path = os.path.join(EPHYS_DIR, f"data_structure_{spec['animal']}_{spec['date']}.mat")
    print(f"Loading {os.path.basename(data_path)}")
    with h5py.File(data_path, "r") as f:
        behavior = load_behavior(f)
        probes = [load_probe_clusters(f, probe_number) for probe_number in spec["probes"]]
        neural_all, kept_quality, single_flags = bin_spikes_for_session(behavior, probes)
        raw_motion_energy, manual_motion_thresh = load_motion_energy(...)
```
```python
def load_motion_energy(animal, date, behavior):
    path = os.path.join(EPHYS_DIR, f"motionEnergy_{animal}_{date}.mat")
    dat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["me"]
    raw = dat.data
    if not isinstance(raw, np.ndarray) and hasattr(raw, "data"):
        raw = raw.data
```

iii. From the trajectory (step 12): *"The paper statistics say the main dataset should be 12 two-context ephys sessions from 6 mice, so I'm checking the loader scripts and session metadata to recover that exact inclusion set."* The methods text states "two-context paradigm: 12 sessions, six mice, 522 units, including 214 well-isolated single units", and the AI verified its cohort reproduces 521 units / 213 single-quality units (CONVERSION_NOTES.md). It explicitly anchored the cohort to the Figure 8 loader combination rather than to the folder contents: *"I kept the implementation anchored to the repository's explicit session loaders and filtering logic rather than modifying the cohort to force an exact paper-text match."* It also noted the discrepancy that its cohort spans 7 animal IDs while the paper says six mice.

## 1-b. How are the data split into subjects (mice)?

i. The subject is the `animal` field of each session spec, which is the animal prefix of the filename (`JEB19_2023-04-19` → `JEB19`). Subjects are collected in first-encounter order into an `OrderedDict`, and `subject_idx` is that index per session. The 12 sessions come from 7 animals (JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, JEB19); the animal id inside the file (`obj.meta.anm`) is never read.

ii.
```python
    for spec in SESSION_SPECS:
        payload = build_session_payload(spec)
        sid = spec["animal"]
        if sid not in subject_lookup:
            subject_lookup[sid] = len(subject_lookup)
            subjects.append(sid)
        ...
        data["subject_idx"].append(subject_lookup[sid])
    data["subject_idx"] = np.asarray(data["subject_idx"], dtype=np.int64)
```

iii. Not discussed explicitly in the trajectory. Implicitly, the animal id is part of the session key used by the authors' own loaders (`load<ANM>_ALMVideo.m`) and of every filename, so it is the natural subject label. The AI did flag in CONVERSION_NOTES.md that this yields 7 animal IDs against the paper's "six mice", and chose to keep the code's answer rather than force the paper's number.

## 1-c. How are the data split into sessions?

i. One session = one `(animal, date)` entry of `SESSION_SPECS` = one `data_structure` file = one element of `neural`/`input`/`output`/`brain_region_idx`/`subject_idx`. 12 sessions total, all from the fixed-delay `Ephys_Behavior` folder. The probe to use is fixed per session in the spec (all 12 are single-probe); the code supports concatenating several probes but never needs to. Sessions from the randomized-delay task and from the additional fixed-delay animals present in `/app/data` (JEB11, JEB12, JEB13, JEB14, JEB15, JEB23, JEB24 — 32 further sessions that the authors' `load<ANM>_ALMVideo.m` scripts also list) are not loaded at all.

ii.
```python
    for spec in SESSION_SPECS:
        payload = build_session_payload(spec)
        ...
        data["neural"].append(payload["neural_trials"])
        data["input"].append(payload["input_trials"])
        data["output"].append(payload["output_trials"])
        data["brain_region_idx"].append(payload["brain_region_idx"])
```
```python
        probes = [load_probe_clusters(f, probe_number) for probe_number in spec["probes"]]
```

iii. CONVERSION_NOTES.md: *"This conversion targets the alternating delayed-response / water-cued ALM electrophysiology cohort used for the context analyses in the reference repository. The session list follows the loader combination used in `code/Scripts/Figure 8/Figure8a_thru_c.m` and `code/Scripts/Figure 8/Figure8d.m`."* The AI cross-checked the resulting counts against the paper's headline numbers (12 sessions / 522 units / 214 single units) and got 12 / 521 / 213, and recorded the remaining one-unit gap as a known discrepancy rather than tuning the pipeline to close it.

## 1-d. How are the data split into trials?

i. Trials are the rows of the Bpod table: `bp.Ntrials` is read once, and every per-trial flag (`R`, `L`, `hit`, `miss`, `no`, `early`, `autowater`, `stim.enable`) and every event time (`bp.ev.goCue`, `bitStart`, `sample`, `delay`, `reward`) is read as one entry per trial. Spikes carry their own 1-based `trial` index (converted to 0-based and range-checked against `Ntrials`); camera frames and motion-energy traces are stored per trial in the file, so no trial boundaries have to be reconstructed. Unlike the reference, the per-trial vectors are not truncated to `Ntrials`; the code relies on them already having that length (true for these 12 files). `bp.stim.enable` is defaulted to all-false if absent.

ii.
```python
def load_behavior(f: h5py.File) -> dict:
    bp = f["obj"]["bp"]
    out = {
        "Ntrials": int(np.asarray(read_numeric_dataset(bp["Ntrials"])).item()),
        "R": ..., "L": ..., "hit": ..., "miss": ..., "no": ..., "early": ..., "autowater": ...,
    }
    if "stim" in bp and "enable" in bp["stim"]:
        out["stim_enable"] = np.asarray(read_numeric_dataset(bp["stim"]["enable"]), dtype=bool)
    else:
        out["stim_enable"] = np.zeros(out["Ntrials"], dtype=bool)
```
```python
            trial_idx = clu["trial"]
            valid_trial_spikes = (trial_idx >= 0) & (trial_idx < ntrials)
```

iii. Not discussed explicitly. The AI inspected the HDF5 schema at length (steps 32–61) and confirmed that "the behavioral event fields are directly available per trial, and `goCue` is present even in the alternating-context sessions" (step 55), so the Bpod table was taken as the trial definition, as in the authors' MATLAB code.

## 1-e. How are trials filtered based on quality controls?

i. One mask, `analysis_trial_mask`, keeps a trial only if it is `(hit | miss) & ~early & ~no & ~stim.enable`. So three classes of trial are dropped: early-lick trials, photostimulation trials, **and no-response ("ignore") trials**. Across the 12 sessions this keeps 2,415 of 3,626 trials; the ignore-trial exclusion alone removes 701 trials that the paper's own early/stim criteria would have kept. No filter is applied for trials that run past the end of the ephys recording (verified above: none of these 12 sessions has trials beyond the last spike, so the omission is harmless here). No session-level or minimum-trial-count filter is applied beyond a guard that raises if fewer than two trials survive.

ii.
```python
def analysis_trial_mask(behavior: dict) -> np.ndarray:
    return (behavior["hit"] | behavior["miss"]) & (~behavior["early"]) & (~behavior["no"]) & (~behavior["stim_enable"])
...
    use_trials = np.flatnonzero(analysis_trial_mask(behavior))
    if use_trials.size < 2:
        raise RuntimeError(f"{spec['animal']} {spec['date']} has fewer than 2 usable trials")
```

iii. CONVERSION_NOTES.md: *"The decoder dataset excludes `early` trials, `no` response trials, `stim.enable` trials. This matches the recurring `~early`, `~no`, and `~stim.enable` filtering used in the repository analyses."* The paper's methods likewise say early-lick and ignore trials "were omitted from all analyses". The AI did not weigh this against the decoder specification, which lists "none" as a lick-direction category and "ignore" as an outcome category.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}`, restricted to the single probe named in the session spec. For each cluster the code dereferences `quality` (the manual curation string), `trial` (1-based trial of each spike, converted to 0-based) and `trialtm` (spike time relative to trial start). The go cue times `obj.bp.ev.goCue` are the second input, since they define the alignment. `obj.bp.hit/miss/no/early/stim.enable/autowater` also enter the neural path indirectly, because they define the seven PSTH conditions used by the low-firing-rate filter.

ii.
```python
def load_probe_clusters(f: h5py.File, probe_number: int):
    clu_cells = np.array(f["obj"]["clu"]).reshape(-1, order="F")
    probe_group = f[clu_cells[probe_number - 1]]
    qualities = deref_string_list(f, probe_group["quality"])
    trials = deref_numeric_list(f, probe_group["trial"])
    trialtm = deref_numeric_list(f, probe_group["trialtm"])
    return [{"quality": qualities[i],
             "trial": np.asarray(trials[i], dtype=np.int64) - 1,
             "trialtm": np.asarray(trialtm[i], dtype=np.float64)} for i in range(len(qualities))]
```

iii. Step 25: *"I have the MATLAB processing path now: trials are aligned to `goCue`, binned at `dt=0.01` s, causally smoothed with `smooth=15`, and low-rate units are removed using mean PSTH firing rate >1 Hz."* This mirrors `alignSpikes.m` (which uses `clu.trial`/`clu.trialtm`) and `findClusters.m` (which uses `clu.quality`).

## 2-b. How is the `neural` data processed?

i. Per cluster and per trial, spikes are histogrammed into the 550 fixed 10 ms bins, divided by the bin width to give Hz, and smoothed along time with a re-implementation of the authors' `mySmooth(x, 15, 'reflect')`: a `gausswin(15, alpha=2.5)` whose first `floor(15/2)=7` taps are zeroed to make it **causal**, normalised to sum 1, convolved in `'same'` mode after prepending the first 15 samples as the boundary condition and then trimming them off. No normalisation, baseline subtraction or z-scoring; stored values are firing rates in Hz (`float32`, observed range 0–169 Hz). Rates are computed for all `Ntrials` trials and the kept trials are selected afterwards.

ii.
```python
def gausswin(n, alpha=2.5):
    idx = np.arange(n, dtype=np.float64) - (n - 1) / 2
    half = (n - 1) / 2
    return np.exp(-0.5 * (alpha * idx / half) ** 2)

def my_smooth(x, n, bctype="none"):
    ...
    if bctype.lower() == "reflect":
        x_filt = np.concatenate([x[:n], x], axis=0); trim = n
    kern = gausswin(n)
    kern[: n // 2] = 0          # causal
    kern /= kern.sum()
    out[:, col] = np.convolve(x_filt[:, col], kern, mode="same")
    out = out[trim:]
```
```python
                for t in unique_trials:
                    mask = trial_idx == t
                    counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
                    trial_counts[:, t] = my_smooth(counts / DT, SMOOTH, BCTYPE)
```

iii. CONVERSION_NOTES.md: *"Smoothing: causal Gaussian kernel with `N=15`, matching `mySmooth(..., 15, 'reflect')`."* The AI read `utils/mySmooth.m` (step 82) and `getSeq.m` (step 19) and reproduced them line for line, including MATLAB's non-mirrored "reflect" padding (`cat(1, x(1:N,:), x)`) and the trim.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, in the same order as the MATLAB. (1) Curation label: a cluster is dropped if its stripped `quality` string is exactly one of `garbage`, `gabrga`, `noisy`, `real?` — the exact set in `findClusters.m` under `params.quality = {'all'}`. The comparison is case-sensitive (the reference lower-cases first) and `poor` is kept. (2) Low firing rate: for each surviving cluster the code builds the seven condition-averaged PSTHs used by the Figure 8 scripts, takes the mean over time and conditions, and keeps the unit only if that mean is `> 1 Hz`. Result: 521 units across 12 sessions (27–67 per session), of which 213 carry a single-unit-style label. No session is dropped for having too few units (all have ≥27).

ii.
```python
def cluster_quality_is_kept(quality: str) -> bool:
    quality = str(quality).strip()
    return quality not in {"garbage", "gabrga", "noisy", "real?"}
```
```python
    conditions = get_context_conditions(behavior)   # the 7 params.condition entries
...
            psth = np.zeros((TIME_AXIS.size, len(conditions)), dtype=np.float64)
            for ci, mask in enumerate(conditions):
                trix = np.flatnonzero(mask)
                if trix.size:
                    psth[:, ci] = np.mean(trial_counts[:, trix], axis=1)
            mean_fr = float(np.mean(psth))
            if mean_fr > LOW_FR_HZ:
                kept_neural.append(trial_counts)
```

iii. CONVERSION_NOTES.md: *"The low-FR filter was matched to the MATLAB path: build single-trial smoothed firing rates …, build the same 7 context conditions used by the Figure 8 scripts, compute the condition-averaged PSTHs, drop units whose mean across time and conditions is not above 1 Hz."* The AI treated the paper's 522 units / 214 single units as the validation target and reported reaching 521 / 213, attributing the gap to "one borderline unit at the 1 Hz cutoff or a loader-era annotation difference" and declining to force a match.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to the go cue is a single subtraction, exactly as in `alignSpikes.m`: each spike's `trialtm` has the `goCue` time of its own trial subtracted, and the result is histogrammed into the fixed window. `ALIGN_EVENT = "goCue"` is a module constant; both the neural stream and the video streams use it, so bin *k* means the same interval in both.

ii.
```python
ALIGN_EVENT = "goCue"
...
            aligned = clu["trialtm"][valid_trial_spikes] - behavior["ev"][ALIGN_EVENT][trial_idx]
```

iii. CONVERSION_NOTES.md: *"Alignment event: `goCue`"*, and metadata `'temporal_alignment_event': 'Go cue onset (reference code alignEvent = goCue)'`. The AI read `alignSpikes.m` (step 20), whose body is `obj.clu{prb}(clu).trialtm_aligned = obj.clu{prb}(clu).trialtm - event`, and confirmed `goCue` is present in every alternating-context session (step 55).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 10 ms bins (`DT = 1/100`) over the window −3.0 s to +2.5 s from the go cue, i.e. 550 bins per trial, identical for every trial and session. Bin edges are built once at module level and bin centres are the decoder input. No rebinning is applied after the initial histogram: spikes are counted straight into the 10 ms grid, and the video/motion-energy streams (recorded at 400 Hz, 2.5 ms) are *interpolated* — not averaged — onto the same grid, so the video is effectively subsampled 4:1. `metadata['time_bin_size'] = 10.0` ms, `off_start = -3.0`, `off_end = 2.5`.

ii.
```python
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_AXIS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
```

iii. These are `params.tmin = -3`, `params.tmax = 2.5`, `params.dt = 1/100`, read verbatim off `Scripts/Figure 8/Figure8a_thru_c.m` (step 77), the script whose loader set defines the AI's cohort. (`Figure8d.m` uses `dt = 1/200`; the AI took the 10 ms value from the primary figure script.) CONVERSION_NOTES.md lists "Time window: `[-3.0 s, 2.5 s]`, Bin size: `10 ms`" under "Reference Processing Reproduced".

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Nothing from the raw files — it is the bin-centre vector `TIME_AXIS` defined by the window and bin size, which is itself fixed relative to `bp.ev.goCue` through the alignment. The same `(1, 550)` array is stored for every trial of every session, cast to `float32`, and named `time_from_go_cue_sec`.

ii.
```python
INPUT_NAMES = ["time_from_go_cue_sec"]
...
        input_trials.append(TIME_AXIS[None, :].astype(np.float32))
```

iii. Not separately justified; it follows from the decoder specification ("Time from go cue onset in seconds (continuous, time-varying)") and from the alignment choice in 2-d. Verified range in the saved file: −2.995 … 2.495 s.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None. The vector is `(edges[:-1] + edges[1:]) / 2`, computed once at import and tiled across trials with no per-trial computation, scaling or offset.

ii.
```python
TIME_AXIS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
```

iii. N/A — no processing to justify.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural grid. `TIME_EDGES` is used both to histogram the spikes and to define the input, so input sample *k* is the centre of the same 10 ms interval that neural column *k* counts. The same axis is also the interpolation target for tongue, paw and motion energy, so all five streams are on one time base.

ii.
```python
                    counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
...
        input_trials.append(TIME_AXIS[None, :].astype(np.float32))
...
            vals = interp(time_axis)     # video features interpolated onto the same axis
```

iii. N/A — by construction.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial Bpod flags: `bp.R` and `bp.L` (the instructed port) together with `bp.hit` and `bp.miss` (whether the animal licked the instructed one). The actual licked side is not recorded, so it is inferred from the pair, exactly as in the reference.

ii.
```python
def actual_lick_direction(behavior: dict) -> np.ndarray:
    return ((behavior["R"] & behavior["hit"]) | (behavior["L"] & behavior["miss"])).astype(np.int64)
```

iii. CONVERSION_NOTES.md: *"Defined as actual lick direction, not instructed side: right if `R & hit` or `L & miss`; left if `L & hit` or `R & miss`."*

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A two-class relabelling: 1 = right lick (`R&hit` or `L&miss`), 0 = left lick (everything else among the kept trials). The per-trial value is tiled across all 550 bins. There is **no third "none" class**: trials on which the animal did not lick were already removed by the trial mask (1-e), so `output_values[0] = ['left', 'right']` and the saved data contains only 0/1 (verified in `converted_data.pkl`). Overall split 51.4% left / 48.6% right.

ii.
```python
OUTPUT_VALUES = [["left", "right"], ...]
...
    lick_dir = actual_lick_direction(behavior)[use_trials]
...
                repeat_labels(np.asarray([lick_dir[local_idx]], dtype=np.int64), TIME_AXIS.size),
```
```python
def repeat_labels(values: np.ndarray, n_time: int) -> np.ndarray:
    return np.repeat(values[:, None], n_time, axis=1)
```

iii. The binary coding follows from the trial filter, which the AI justified as "the recurring `~early`, `~no`, and `~stim.enable` filtering used in the repository analyses" (CONVERSION_NOTES.md). The trajectory contains no discussion of the "none" category named in the decoder specification. The AI's stated rationale for tiling: *"Behavioral outputs are tiled across time so the decoder can be trained in the provided timepoint-wise format."*

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial flag, `obj.bp.autowater`, read as a boolean vector.

ii.
```python
        "autowater": np.asarray(read_numeric_dataset(bp["autowater"]), dtype=bool),
```

iii. Not separately argued; the AI verified early on (step 62) that `autowater` is present and non-degenerate in these sessions (`print('autowater unique', ..., 'sum', ...)`), and the authors' condition strings (`hit&~stim.enable&autowater` = "all AW hits") identify autowater with the water-cued context.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling of the flag, tiled over time: autowater → WC (0), everything else → DR (1). Across the dataset 28.9% WC / 71.1% DR.

ii.
```python
    context = (~behavior["autowater"][use_trials]).astype(np.int64)
...
OUTPUT_VALUES = [..., ["WC", "DR"], ...]
```

iii. CONVERSION_NOTES.md: *"`behavioral_context`: `0 = WC`, `1 = DR`"* — the coding required by the decoder specification.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `obj.bp.hit` alone for the label, with `bp.miss` and `bp.no` entering through the trial mask (a kept trial is either a hit or a miss, so `~hit` ⇒ miss).

ii.
```python
    outcome = behavior["hit"][use_trials].astype(np.int64)
```

iii. Not separately argued. It is the minimal reading given the trial filter, and `hit`/`miss`/`no` are the flags the authors' own condition strings use.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A binary relabelling tiled over time: 1 = correct (hit), 0 = incorrect (miss). The **"ignore" class does not exist** — `no`-response trials were dropped in 1-e — so `output_values[2] = ['incorrect', 'correct']` and the saved values are only 0/1. Class balance 13.6% incorrect / 86.4% correct.

ii.
```python
OUTPUT_VALUES = [..., ["incorrect", "correct"], ...]
...
                repeat_labels(np.asarray([outcome[local_idx]], dtype=np.int64), TIME_AXIS.size),
```

iii. Same justification as 4-b: the repository's analysis filters exclude no-response trials, and the AI followed those filters. The decoder specification's third category ("ignore") is not addressed anywhere in the notes or the trajectory.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj{1}` — the **side camera only** — feature `tongue`. Per trial the code reads `featNames` (to locate the feature), `frameTimes`, `ts` (frames × [x, y, likelihood] × features) and `NdroppedFrames`. Only the x and y columns are used; the likelihood column is never read (the authors already NaN-out x/y below their likelihood cut, so untracked frames arrive as NaN). The bottom-camera tongue (`top_tongue`) is not used. Alignment additionally needs `obj.sglx.bitcode.bitstart`, `obj.sglx.fs` and `bp.ev.bitStart` for the video offset, plus `bp.ev.goCue`.

ii.
```python
        tongue_xpos, tongue_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, "tongue", 1)
...
        ts = normalize_traj_array(ts, len(feature_names_by_trial[trial_idx]))
        coords = ts[:, 0:2, feat_idx]
```
```python
def normalize_traj_array(ts: np.ndarray, nfeatures: int) -> np.ndarray:
    if ts.shape[2] == nfeatures:
        return ts
    if ts.shape[0] == nfeatures:
        return np.transpose(ts, (2, 1, 0))
```

iii. Step 89: *"the reference code aligns video by `frameTimes - video_offset - goCue`, interpolates onto the neural time axis, fills non-tongue gaps with nearest values, and sets invisible tongue velocity to zero. I'm now checking which scalar tongue and paw velocity summaries the authors actually use."* The AI read `findPosition.m`/`findVelocity.m`/`getKinematicsFromVideo.m` (steps 84–87), where each view/feature pair is handled separately, and took the side-view `tongue` as the tongue channel. The orientation-normalisation step was added after hitting real files: *"The trajectory arrays change orientation across recording vintages. I'm adding a normalization step so both old and new files get converted to the same layout"* (step 127).

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps, modelled on `findPosition.m` + `findVelocity.m`. (1) Trials whose `NdroppedFrames` is all-NaN are skipped (left as NaN). (2) x and y are linearly interpolated from frame times onto the 550-bin axis — **but the interpolator is fitted only to the finite samples**, so it bridges the gaps where the tongue is invisible rather than propagating NaN as MATLAB's `interp1` would. No smoothing is applied to the tongue (matching `if ~contains(feat,'tongue')` in the MATLAB) and no nearest-fill. (3) Velocity is `np.gradient` of the interpolated x and y (units of pixels per 10 ms bin, not per second), with **no** baseline-derivative subtraction for the tongue, and remaining non-finite values set to 0 ("tongue velocity 0 if not visible", as in `findVelocity.m`). (4) Speed is `sqrt(xvel² + yvel²)`. Measured on JEB6_2021-04-18: only ~7.8% of raw frames have the tongue tracked in an example trial, yet 28.6% of output bins end up with a finite interpolated tongue position, so roughly two thirds of the "moving tongue" samples are interpolated through invisibility gaps; 71.7% of bins have exactly zero speed.

ii.
```python
        for dim in range(2):
            valid = np.isfinite(shifted_t) & np.isfinite(coords[:, dim])
            if valid.sum() < 2:
                continue
            interp = interp1d(shifted_t[valid], coords[valid, dim], kind="linear",
                              bounds_error=False, fill_value=np.nan, assume_sorted=True)
            vals = interp(time_axis)
```
```python
        xvel[:, trial_idx] = np.gradient(tsinterp[:, 0])
        yvel[:, trial_idx] = np.gradient(tsinterp[:, 1])
        if "tongue" not in feature_name:
            ...
        else:
            xvel[:, trial_idx][~np.isfinite(xvel[:, trial_idx])] = 0.0
            yvel[:, trial_idx][~np.isfinite(yvel[:, trial_idx])] = 0.0
...
    tongue_speed = np.sqrt(tongue_xvel**2 + tongue_yvel**2)
```

iii. CONVERSION_NOTES.md: *"Position interpolation and velocity computation follow the repository logic: non-tongue features are nearest-filled; tongue invisibility periods become zero velocity."* The AI's aim throughout was literal fidelity to the MATLAB kinematics functions rather than an independent velocity estimate. The gap-bridging side effect of filtering to finite samples before interpolating is not mentioned anywhere.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Two classes, split at a per-session threshold computed over the kept trials only: `0 = lt_p50`, `1 = ge_p50`. There is **no "not visible" class** — invisible bins carry speed 0 and therefore fall in class 0. Because more than half of all bins are exactly 0, the plain 50th percentile is 0 and every bin would be labelled `>= threshold`; the AI therefore added a fallback used only for the tongue: if the all-sample median is ≤ 0, the threshold is recomputed as the median of the strictly positive speeds. Resulting class balance 82.6% / 17.4% (one session sits at exactly 50/50, where the fallback did not trigger).

ii.
```python
def percentile_threshold(values, drop_zeros_if_needed=False):
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    thresh = float(np.nanpercentile(vals, 50))
    if drop_zeros_if_needed and thresh <= 0:
        pos = vals[vals > 0]
        if pos.size:
            thresh = float(np.nanpercentile(pos, 50))
    return thresh
...
    tongue_thresh = percentile_threshold(tongue_selected, drop_zeros_if_needed=True)
...
                (tongue_selected[:, local_idx] >= tongue_thresh).astype(np.int64)[None, :],
```

iii. Step 138: *"The verifier found a real issue: tongue velocity became effectively constant in some sessions because more than half the bins are zero from tongue invisibility, so the median threshold collapses to zero. I'm changing the tongue binning rule to estimate the 50th percentile from positive tongue-speed samples only when the all-sample median is zero, which preserves the reference preprocessing while avoiding an artifact from missing visibility."* CONVERSION_NOTES.md repeats this: *"because the repository sets invisible-tongue velocity to zero, if the all-sample median is zero the threshold is recomputed from positive tongue-speed samples only; otherwise several sessions collapse to a trivial constant label."*

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are converted to seconds from the go cue with `frameTimes − vidshift − goCue[trial]`, where `vidshift` is the session video offset computed exactly as in `findVideoOffset.m`: `mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)` (falling back to 0.5 s if `sglx.bitcode` is absent, mirroring the MATLAB `catch`). The corrected times are then the abscissa for linear interpolation onto `TIME_AXIS`, so the tongue channel lands on exactly the same 550-bin grid as the spikes. Points outside the frame range become NaN and (for the tongue) then 0.

ii.
```python
def get_video_offset(f: h5py.File, behavior: dict) -> float:
    if "sglx" not in f["obj"] or "bitcode" not in f["obj"]["sglx"]:
        return 0.5
    bitstart = np.asarray(read_numeric_dataset(sglx["bitcode"]["bitstart"]), dtype=np.float64)
    fs = float(np.asarray(read_numeric_dataset(sglx["fs"])).item())
    return matlab_mode(bitstart) / fs - matlab_mode(behavior["ev"]["bitStart"])
...
        shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
```

iii. CONVERSION_NOTES.md: *"Video alignment matches the reference code: `frameTimes - video_offset - goCue`; `video_offset` matches `findVideoOffset`: `mode(obj.sglx.bitcode.bitstart) / obj.sglx.fs - mode(obj.bp.ev.bitStart)`."* The AI read `findVideoOffset.m` at step 88.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. `obj.traj{2}` — the bottom camera — feature `top_paw` (with `bottom_paw` as a fallback if `top_paw` is absent), same fields as the tongue (`featNames`, `frameTimes`, `ts`, `NdroppedFrames`) plus the same offset/go-cue variables.

ii.
```python
def choose_paw_feature(f: h5py.File) -> str:
    view_group = f[traj_refs[1]]
    first_feats = deref_string_list(f, f[np.array(view_group["featNames"]).reshape(-1, order="F")[0]])
    for name in ("top_paw", "bottom_paw"):
        if name in first_feats:
            return name
    raise KeyError("No paw feature found in bottom-view trajectory data")
...
        paw_xpos, paw_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, paw_feature, 2)
```

iii. CONVERSION_NOTES.md: *"scalar paw speed from the `top_paw` bottom-view marker"*. The methods text the AI read states the paws "were tracked using only the bottom view", so the view is forced; `top_paw` is preferred over `bottom_paw` by the lookup order.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The MATLAB non-tongue path. (1) Trials with all-NaN `NdroppedFrames` are skipped. (2) `mySmooth(ts, 1, 'reflect')` is called on the raw coordinates — a no-op, since N=1 returns the input, faithfully reproducing the MATLAB call. (3) x and y are linearly interpolated onto the 550-bin axis (again fitted only to finite samples, so tracking gaps are bridged), then `fillmissing(...,'nearest')` is reproduced by `fill_nearest` on the positions, which also extrapolates flat values beyond the ends of the video. (4) Velocity is `np.gradient` of x and y, minus `basederiv[0]` — the median frame-to-frame x displacement — **from both components**, reproducing what looks like a bug in `findVelocity.m` (MATLAB subtracts `basederiv(1)` from `yvel` too). (5) `fill_nearest` again on the velocities (all-NaN trials become all zeros). (6) Speed = `sqrt(xvel² + yvel²)`. The result has no missing values at all.

ii.
```python
        if "tongue" not in feature_name:
            coords = my_smooth(coords, 1, "reflect")
...
        if "tongue" not in feature_name:
            xpos[:, trial_idx] = fill_nearest(xpos[:, trial_idx])
            ypos[:, trial_idx] = fill_nearest(ypos[:, trial_idx])
```
```python
        basederiv = np.nanmedian(np.diff(tsinterp, axis=0), axis=0)
        xvel[:, trial_idx] = np.gradient(tsinterp[:, 0])
        yvel[:, trial_idx] = np.gradient(tsinterp[:, 1])
        if "tongue" not in feature_name:
            xvel[:, trial_idx] = xvel[:, trial_idx] - basederiv[0]
            yvel[:, trial_idx] = yvel[:, trial_idx] - basederiv[0]
            xvel[:, trial_idx] = fill_nearest(xvel[:, trial_idx])
            yvel[:, trial_idx] = fill_nearest(yvel[:, trial_idx])
```

iii. CONVERSION_NOTES.md: *"non-tongue features are nearest-filled … non-tongue baseline subtraction reproduces the MATLAB implementation, including subtracting `basederiv(1)` from both `xvel` and `yvel`."* The AI deliberately copied the quirk rather than "fixing" it, on the grounds of matching the reference pipeline.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Two classes at the per-session 50th percentile of the kept trials' paw speed (`0 = lt_p50`, `1 = ge_p50`), with no positive-only fallback and **no "not visible" class**. Because nearest-filling removes every NaN, the split is exactly 50.0% / 50.0% in every one of the 12 sessions — including bins where the paw was never tracked, which are assigned whichever class the carried-forward value falls into.

ii.
```python
    paw_thresh = percentile_threshold(paw_selected)
...
                (paw_selected[:, local_idx] >= paw_thresh).astype(np.int64)[None, :],
```

iii. CONVERSION_NOTES.md: *"binarized with the per-session 50th percentile"*. The absence of a third class follows from the nearest-fill decision in 8-b; it is not separately discussed.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue, through the same `load_traj_feature_series` code path: the bottom camera's own `frameTimes`, minus the session video offset, minus the trial's go cue, then linear interpolation onto the shared 550-bin axis. Each view uses its own frame times, so a mismatch in frame count between cameras is harmless.

ii.
```python
        shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
        ...
            vals = interp(time_axis)
```

iii. Same justification as 7-d — one video-offset rule from `findVideoOffset.m`, applied to whichever camera supplies the feature.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file beside the data structure: `me.data`, one trace per trial with one value per side-camera frame, unwrapped one level if the file stores `me.data.data` (three files do). `me.moveThresh` is also read but only recorded in metadata — it is not used to label anything. The side camera's `frameTimes` (from `obj.traj{1}`) supply the time base. The copy that some sessions carry in `obj.me` is not used.

ii.
```python
def load_motion_energy(animal, date, behavior):
    dat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["me"]
    raw = dat.data
    if not isinstance(raw, np.ndarray) and hasattr(raw, "data"):
        raw = raw.data
    move_thresh = float(dat.moveThresh)
    return raw, move_thresh
```

iii. The unwrap guard mirrors `loadMotionEnergy.m`'s `if isstruct(me.data), me.data = me.data.data; end`, which the AI read at step 83. The guard's exact form was arrived at by debugging: *"I found the issue: I was treating any object with a `.data` attribute like a MATLAB struct, and NumPy arrays also expose `.data`. I'm tightening that loader so only MATLAB struct wrappers get unwrapped"* (step 122).

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond resampling: the per-frame trace is linearly interpolated onto the 550-bin axis and then `fill_nearest` is applied per trial, which carries edge values outward so that bins before the first or after the last video frame receive the nearest real value rather than NaN. No smoothing, no normalisation, no baseline subtraction (the MATLAB baseline subtraction is commented out in `loadMotionEnergy.m` and is likewise omitted here). If fewer than two finite samples exist for a trial, the column is left NaN.

ii.
```python
        interp = interp1d(shifted_t[valid], me_trial[valid], kind="linear",
                          bounds_error=False, fill_value=np.nan, assume_sorted=True)
        aligned[:, trial_idx] = interp(TIME_AXIS)
        aligned[:, trial_idx] = fill_nearest(aligned[:, trial_idx])
```

iii. CONVERSION_NOTES.md: *"Motion energy comes from `motionEnergy_Animal_Date.mat`, is resampled onto the neural time axis, and is nearest-filled at the edges."* This reproduces `loadMotionEnergy.m`, which does `interp1(frameTimes-vidshift-alignTimes, me.data{trix}, taxis)` followed by `fillmissing(me.data,'nearest')` with the comment "there are some nans at the start of each trial". The paper already reduces motion energy to one value per frame (99th percentile across pixels), so no spatial processing is needed.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Two classes at the per-session 50th percentile over the kept trials (`0 = lt_p50`, `1 = ge_p50`); there is **no "no video" class**. Because of the nearest-fill the split is 50/50 in essentially every session (49.9/50.1 overall). A trial whose motion energy could not be interpolated at all stays NaN, and `NaN >= threshold` evaluates to `False`, so it is silently labelled class 0 rather than flagged.

ii.
```python
    me_thresh = percentile_threshold(me_selected)
...
                (me_selected[:, local_idx] >= me_thresh).astype(np.int64)[None, :],
```

iii. CONVERSION_NOTES.md: *"aligned motion energy, binarized with the per-session 50th percentile"*. The AI chose the percentile split from the decoder specification rather than `me.moveThresh`, but kept the latter in `session_stats` as `manual_motion_energy_move_thresh` for reference.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. With the side camera's frame times (`obj.traj{1}[trial].frameTimes`), the same session video offset, and the trial's go cue — `frameTimes − vidshift − goCue[trial]` — then interpolated onto the shared 550-bin axis. If `frameTimes` cannot be read, the code falls back to a synthetic 400 Hz axis, `(1..n)/400`, mirroring the MATLAB fallback (though it omits the MATLAB's `-0.5` offset in that branch).

ii.
```python
def align_motion_energy(f, behavior, raw_motion_energy):
    view_group = f[traj_refs[0]]                      # side camera
    ...
        try:
            frame_times = np.asarray(read_numeric_dataset(f[frame_refs[trial_idx]]), dtype=np.float64)
        except Exception:
            ts = np.asarray(read_numeric_dataset(f[ts_refs[trial_idx]]), dtype=np.float64)
            frame_times = np.arange(1, ts.shape[0] + 1, dtype=np.float64) / 400.0
        shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
```

iii. Same as 9-b: a direct transcription of `loadMotionEnergy.m`, which indexes `obj.traj{1}(trix).frameTimes` for exactly this purpose.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several guards, all of which keep the trial and fill or zero the gap rather than marking it. (a) Trials whose `NdroppedFrames` is entirely NaN, or whose `ts` is not 3-D, or whose feature is missing from that trial's `featNames`, are skipped in the trajectory loader and stay NaN — which for the tongue becomes speed 0 (class 0) and for the paw becomes `fill_nearest` of an all-NaN column, i.e. exactly 0.0 (also class 0). (b) Untracked frames (NaN x/y) are dropped before interpolation, so the interpolator bridges them instead of propagating NaN. (c) Non-tongue gaps and edges are nearest-filled, as in MATLAB `fillmissing`. (d) Missing `frameTimes` for motion energy falls back to a synthetic 400 Hz axis; missing `sglx.bitcode` falls back to a 0.5 s video offset. (e) Missing `bp.stim.enable` is treated as no photostimulation. (f) Spikes whose trial index falls outside `[0, Ntrials)` are discarded. (g) A session with fewer than two usable trials raises. (h) `np.nanmedian` over an all-NaN slice emits a RuntimeWarning that is left visible in `conversion_full_out.txt`. Net effect: no NaN reaches the saved file, but missing video is represented as a genuine "low movement" label rather than as missing.

ii.
```python
        ndropped = np.asarray(read_numeric_dataset(f[ndropped_refs[trial_idx]]), dtype=np.float64)
        if ndropped.size and np.isnan(ndropped).all():
            continue
        if trial_idx not in feature_indices:
            continue
        ...
        if ts.ndim != 3:
            continue
```
```python
def fill_nearest(x: np.ndarray) -> np.ndarray:
    mask = np.isfinite(x)
    if mask.all():
        return x
    if not mask.any():
        return np.zeros_like(x)
    interp = interp1d(idx[mask], x[mask], kind="nearest", bounds_error=False,
                      fill_value=(x[mask][0], x[mask][-1]), assume_sorted=True)
    return interp(idx)
```

iii. Step 153: *"I'm cleaning one remaining nuisance before I write the final logs: a warning on completely missing trajectory trials. It doesn't break the conversion, but I want the saved output files to reflect deliberate handling rather than runtime noise."* The general principle stated in CONVERSION_NOTES.md is fidelity to the repository's own missing-data conventions (`fillmissing(...,'nearest')`, tongue → 0, synthetic 400 Hz frame times), which is what these guards reproduce.

## 11-a. What are the most time-consuming steps of the code?

i. The AI never profiled the code and makes no claim about this. Measured here on one session (JEB6_2021-04-18, 382 trials, 33 clusters on the probe): trajectory loading + interpolation dominates — 1.0 s for the tongue and 1.4 s for the paw, against 0.5 s for spike binning and smoothing, 0.2 s for motion-energy alignment and <0.1 s for behaviour and cluster reads. So roughly 3 s per session, ~40 s for all 12, plus the cost of `pickle.dump` for the 307 MB output (and a further 104 MB for the optional sample file). The video cost is per trial per feature: an HDF5 dereference of `featNames`, `frameTimes`, `ts` and `NdroppedFrames` plus two `interp1d` constructions. The spike cost is a Python loop over clusters × trials, each doing one `np.histogram` and one `np.convolve` over 550 samples.

ii.
```python
    for trial_idx in range(ntrials):
        ndropped = ...; frame_times = ...; ts = ...
        for dim in range(2):
            interp = interp1d(...); vals = interp(time_axis)
```
```python
                for t in unique_trials:
                    mask = trial_idx == t
                    counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
                    trial_counts[:, t] = my_smooth(counts / DT, SMOOTH, BCTYPE)
```

iii. No justification offered; the cohort is small enough (12 sessions) that the whole conversion finishes in under a minute, so the AI never had reason to optimise.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Four. (1) The spike-binning double loop over clusters and trials: the whole cluster could be counted in one `np.histogram2d(spike_trial, aligned, bins=[trial_edges, TIME_EDGES])` (which is what the human reference does), and the smoothing could be applied to the whole `(550, n_trials)` matrix at once — `my_smooth` already accepts 2-D input, but it is called per trial with a single column, and internally loops over columns with `np.convolve` instead of using an FFT or `scipy.ndimage`. (2) The per-condition PSTH loop, which is seven `np.mean` calls over overlapping trial subsets, only to take the grand mean afterwards. (3) `compute_velocity_from_position`, which loops over trials to call `np.gradient` and `np.nanmedian` per column; both accept an `axis` argument and could run on the full matrix. (4) The per-trial output-assembly loop, which calls `repeat_labels` and `np.vstack` 2,415 times to build arrays that could be produced once per session and sliced. The trajectory loop itself resists full vectorisation because each trial has a different number of frames, though its two `interp1d` calls per trial could be one (`interp1d` supports `axis=`).

ii.
```python
            for t in unique_trials:
                ...
            for ci, mask in enumerate(conditions):
                ...
    for trial_idx in range(xpos.shape[1]):
        ...
    for local_idx, trial_idx in enumerate(use_trials):
        output = np.vstack([repeat_labels(...), ..., ])
```

iii. Not discussed. The implementation mirrors the MATLAB loop structure (`getSeq.m` loops over clusters and trials in exactly this way), which is presumably why the loops survived.

## 11-c. What processing does the code repeat multiple times?

i. Four repetitions. (1) `get_video_offset` is recomputed three times per session — once in `align_motion_energy` and once in each of the two `load_traj_feature_series` calls — each time re-reading `sglx.bitcode.bitstart` and re-running `matlab_mode` over the behaviour vector, even though it is a session constant. (2) `load_traj_feature_series` dereferences and decodes `featNames` for **every trial** (`deref_string_list` inside the trial loop) although the feature index is effectively constant, and it does so twice per session because it is called once for the tongue and once for the paw; `choose_paw_feature` reads `featNames` a third time. (3) Firing rates are computed for all `Ntrials` trials and then subset to the ~2/3 that survive the trial mask, and rates are computed for every quality-passing cluster before the 1 Hz filter discards most of them. (4) `matlab_mode` is run again on `bitStart`/`sample`/`delay`/`goCue` purely to fill the `event_times_sec_relative_to_go_cue` diagnostic block. In `--dataset both` mode the entire 12-session dictionary is additionally `copy.deepcopy`-ed to build the 4-session sample.

ii.
```python
    vidshift = get_video_offset(f, behavior)          # in load_traj_feature_series
...
    vidshift = get_video_offset(f, behavior)          # again in align_motion_energy
```
```python
    for trial_idx, feat_ref in enumerate(feature_refs):
        feats = deref_string_list(f, f[feat_ref])
```
```python
def make_sample_dataset(full_data: dict, session_count: int) -> dict:
    sample = copy.deepcopy(full_data)
```

iii. Not discussed. Computing the rates before filtering is forced by the low-FR rule itself (it needs the PSTHs), and matches `processData.m`, where `getSeq` runs before `removeLowFRClusters`.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several things. (1) The seven condition PSTHs exist only to produce one scalar per cluster for the 1 Hz test; the same test could use the mean over all trials directly. (2) Firing rates for the ~1,200 trials removed by the trial mask are computed and thrown away. (3) `my_smooth(coords, 1, "reflect")` on non-tongue positions is a literal no-op (the function returns its input when `n == 1`), faithfully reproducing a MATLAB call that also does nothing. (4) `basederiv` is computed for the tongue as well, then used only for non-tongue features, and it is the source of the `All-NaN slice` warning. (5) `me.moveThresh` is loaded and stored but never used to label anything. (6) `fill_nearest` is applied to positions and then again to the velocities derived from them. (7) `load_behavior` reads `sample`, `delay`, `reward` and `bitStart` although only `goCue` and `bitStart` are needed, and `single_unit_flags` / `quality_labels_kept` / `usable_trial_indices_0based` / `event_times_sec_relative_to_go_cue` are diagnostics only. (8) A second 104 MB `sample_data.pkl` is built by deep-copying the full dataset, which the task never asked for. (9) The 550-sample time axis is duplicated into every one of the 2,415 `input` entries, and the three per-trial labels are tiled across 550 bins each.

ii.
```python
        if "tongue" not in feature_name:
            coords = my_smooth(coords, 1, "reflect")     # N=1 -> returns x unchanged
```
```python
        basederiv = np.nanmedian(np.diff(tsinterp, axis=0), axis=0)   # unused for tongue
```
```python
    move_thresh = float(dat.moveThresh)                  # stored in stats only
```

iii. Not discussed. Most of these are the price of transcribing the MATLAB pipeline literally, which was the AI's stated goal: *"It will mirror the MATLAB pipeline directly for the chosen cohort instead of relying on precomputed exports"* (step 116).
