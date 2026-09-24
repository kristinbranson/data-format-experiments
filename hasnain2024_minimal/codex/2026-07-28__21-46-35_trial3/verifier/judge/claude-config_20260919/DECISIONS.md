# Decisions

> Note on provenance: the instructions actually delivered to the AI (trajectory step 3) specified **two-class** movement bins and **two-class** lick direction / outcome (`left = 0, right = 1`; `incorrect = 0, correct = 1`; velocity `0: < 50th pct`, `1: >= 50th pct`). The judge-side `instruction_reference.md` is a later revision that adds a third class (`none` / `ignore` / `not visible`). Where a decision below is driven by that difference, it is called out explicitly.

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not glob the data folders. It hard-codes a roster of **12 sessions** (animal, date, probe) transcribed from the paper's own MATLAB context-analysis scripts (`Scripts/Figure 8/Figure8a_thru_c.m` lines 64–70, which call `loadJEB6/JEB7/EKH1/EKH3/JGR2/JGR3/JEB19_ALMVideo.m`). Only the `data/Ephys_Behavior` folder is used; the entire `RandomizedDelay_Ephys_Behavior` folder and the 13 remaining fixed-delay sessions are not loaded. Each session is read with `h5py` only (no scipy/v5 fallback) from `data_structure_<anm>_<date>.mat`, and its motion energy from the sibling `motionEnergy_<anm>_<date>.mat` via `scipy.io.loadmat`. Reads are lazy and field-by-field (`obj/bp`, `obj/sglx`, `obj/traj`, `obj/clu`) rather than materialising the whole `obj` tree.

ii.
```python
DATA_DIR = ROOT / "data" / "Ephys_Behavior"

# Figure 8 context-task roster from the reference MATLAB code.
SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", 2),
    SessionSpec("JEB7", "2021-04-29", 1),
    ...
    SessionSpec("JEB19", "2023-04-18", 1),
]

def load_session(spec: SessionSpec) -> dict:
    ...
    with h5py.File(spec.data_path, "r") as f:
        bp_group = f["obj/bp"]
        ...
def build_dataset() -> dict:
    sessions = [load_session(spec) for spec in SESSION_SPECS]
```

iii. From `CONVERSION_NOTES.md` and trajectory steps 67/79: "The decoder outputs require behavioral context (`WC` vs `DR`), so I converted the ALM two-context electrophysiology cohort rather than the fixed-delay-only or randomized-delay cohorts." and "I'm also using the figure-script session roster rather than 'all files with autowater trials', because the paper's context analyses clearly operate on a curated 12-session subset." The AI validated the roster against the paper text it had read in `methods.txt` ("two-context paradigm: 12 sessions, six mice, 522 units"), reporting 12 sessions and 520 retained units (delta −2), and explicitly flagged the unresolved mismatch of 7 packaged animal IDs versus the paper's "six mice".

## 1-b. How are the data split into subjects?

i. The subject is the animal field of the hard-coded roster entry, i.e. the part of the filename stem before the underscore. `subjects` is the list of unique animals in first-appearance order and `subject_idx` indexes into it per session. Result: 7 subjects over 12 sessions (`JEB6` 1, `JEB7` 2, `EKH1` 1, `EKH3` 1, `JGR2` 2, `JGR3` 1, `JEB19` 4).

ii.
```python
@dataclass(frozen=True)
class SessionSpec:
    animal: str
    date: str
    probe: int
    @property
    def stem(self) -> str:
        return f"{self.animal}_{self.date}"
...
subjects = list(OrderedDict.fromkeys(sess["subject"] for sess in sessions))
subject_to_idx = {subj: i for i, subj in enumerate(subjects)}
...
"subject_idx": np.asarray([subject_to_idx[sess["subject"]] for sess in sessions], dtype=np.int64),
```

iii. `CONVERSION_NOTES.md`: "The converted data contain `7` unique subject IDs … The manuscript text states `6` mice for the context cohort. I preserved the subject IDs present in the provided files rather than collapsing or renaming them without evidence."

## 1-c. How are the data split into sessions?

i. One roster entry = one `data_structure_*.mat` file = one element of `neural` / `input` / `output` / `brain_region_idx`, in roster order. Exactly one probe is processed per session (`spec.probe`, taken from the MATLAB `meta(end).probe` field), so units from a second probe are never concatenated. There is no per-session inclusion test at runtime (no minimum-unit or minimum-trial criterion beyond a hard failure if a session has <2 usable trials or 0 surviving units).

ii.
```python
    @property
    def data_path(self) -> Path:
        return DATA_DIR / f"data_structure_{self.stem}.mat"
...
        clu_group = f[h5_ref_at(f["obj/clu"], spec.probe - 1)]
...
        if included_trials.size < 2:
            raise RuntimeError(f"{spec.stem}: fewer than 2 usable trials after filtering")
```

iii. Same justification as 1-a: the session/probe pairs are the ones named in the paper's Figure 8 loader scripts, and the AI used the paper's "12 sessions … 522 units" as the sanity check.

## 1-d. How are the data split into trials?

i. A trial is one index into the per-trial vectors of `obj.bp` (`hit`, `miss`, `no`, `early`, `autowater`, `R`, `L`, `stim.enable`, `ev.goCue`, `ev.bitStart`), with `obj.bp.Ntrials` as the trial count used to drive the per-trial video loops. Spikes carry their own 1-based `trial` index and a within-trial time `trialtm`; video is already stored per trial as a cell of `frameTimes`/`ts`; motion energy is a cell with one trace per trial. No trial boundaries are reconstructed. The bp vectors are read at full length rather than truncated to `Ntrials` (I verified that for all 12 selected sessions every field length equals `Ntrials`, so this is not an active problem here).

ii.
```python
        bp = {
            "Ntrials": int(round(read_h5_scalar(bp_group["Ntrials"]))),
            "hit": read_h5_vector(bp_group["hit"]).astype(np.int16),
            ...
            "goCue": read_h5_vector(ev_group["goCue"]),
        }
...
        session_trial = np.asarray(trial_ds[()], dtype=np.float64).reshape(-1).astype(np.int64) - 1
        trialtm = np.asarray(trialtm_ds[()], dtype=np.float64).reshape(-1)
        aligned = trialtm - align_times[session_trial]
```

iii. Not explicitly argued; the AI mapped the raw fields by direct HDF5 inspection (trajectory steps 37–43) and noted at step 44 that "event times live in `obj.bp.ev`, spike times are per-cluster `trial`/`trialtm`, video is per-trial `frameTimes` plus `ts`".

## 1-e. How are trials filtered based on quality controls?

i. A single mask, applied before anything else is computed: keep trials that are `(hit | miss) & ~early & ~stim.enable`. This drops early-lick trials and photostimulation trials (as the paper does) **and also drops all ignore / no-response trials** (`bp.no`). Across the 12 sessions this keeps 2,415 of 3,626 trials. No filter is applied for trials that run past the end of the ephys recording (I checked the output: no all-zero neural trial exists in this 12-session subset, so that omission does not bite here).

ii.
```python
def trial_selector(bp: dict[str, np.ndarray]) -> np.ndarray:
    hit_or_miss = bp["hit"].astype(bool) | bp["miss"].astype(bool)
    return hit_or_miss & ~bp["early"].astype(bool) & ~bp["stim_enable"].astype(bool)
...
        included_mask = trial_selector(bp)
        included_trials = np.flatnonzero(included_mask)
```

iii. `CONVERSION_NOTES.md`: "Trial inclusion for the exported decoder dataset: `hit | miss`, `stim.enable == 0`, `early == 0`. Ignore / no-response trials were excluded because the requested `outcome` target is binary correct/incorrect and the paper methods state ignore trials were omitted from analyses." The early/stim exclusions mirror the repo's `params.condition` strings (`~stim.enable&~early`).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` — the spike-sorted clusters of the single roster-specified probe — using each cluster's `trial` (1-based trial index of every spike), `trialtm` (spike time relative to trial start) and `quality` (manual curation label). `obj.bp.ev.goCue` supplies the alignment times. `obj.clu` fields such as `spkWavs`/`tm` are never read.

ii.
```python
        clu_group = f[h5_ref_at(f["obj/clu"], spec.probe - 1)]
        qds = clu_group["quality"]
...
def load_trial_spikes(f, clu_group, clu_idx, align_times):
    trial_ds = f[h5_ref_at(clu_group["trial"], clu_idx)]
    trialtm_ds = f[h5_ref_at(clu_group["trialtm"], clu_idx)]
    session_trial = np.asarray(trial_ds[()], dtype=np.float64).reshape(-1).astype(np.int64) - 1
    trialtm = np.asarray(trialtm_ds[()], dtype=np.float64).reshape(-1)
    aligned = trialtm - align_times[session_trial]
    return session_trial, trialtm, aligned
```

iii. Directly follows the repo's `alignSpikes.m` / `getSeq.m`, which the AI read at trajectory steps 19 and 58.

## 2-b. How is the `neural` data processed?

i. Per unit and per included trial, spikes are histogrammed into the fixed bin edges, divided by the bin width `DT` to give spikes/s, and smoothed along time with a **causal** Gaussian kernel — a Python port of the repo's `mySmooth.m`: `gausswin(15, alpha=2.5)`, the first `floor(15/2)` taps zeroed, normalised to unit sum, convolved with `mode="same"`. The boundary condition is `'reflect'` (prepend the first 15 samples, then trim them), whereas the repo's default is `'none'`. No normalisation, baseline subtraction or z-scoring. Stored as `float32` firing rates in Hz, transposed to `(n_neurons, 550)` per trial.

ii.
```python
def gausswin(length: int, alpha: float = 2.5) -> np.ndarray:
    std = (length - 1) / (2 * alpha)
    return gaussian(length, std=std, sym=True)

def my_smooth(x, n, bctype="none"):
    """Port of `mySmooth.m` operating on axis 0."""
    ...
    if bctype == "reflect":
        x_filt = np.concatenate([x[:n, :], x], axis=0); trim = n
    ...
    kern = gausswin(n)
    kern[: math.floor(kern.size / 2)] = 0.0      # causal
    kern /= kern.sum()
    out[:, j] = np.convolve(x_filt[:, j], kern, mode="same")
    out = out[trim:, :]

def build_neural_trials(session_trial, aligned_spikes, included_trials, edges):
    for col, tr in enumerate(included_trials):
        counts, _ = np.histogram(aligned_spikes[session_trial == tr], bins=edges)
        out[:, col] = my_smooth(counts / DT, SMOOTH, "reflect")
```

iii. `CONVERSION_NOTES.md`: "Spike smoothing: causal Gaussian smoothing with window `15` and reflect padding, matching `mySmooth.m`"; the AI had read `mySmooth.m` (step 68) and `getSeq.m` (step 58), where `obj.trialdat = mySmooth(N./params.dt, params.smooth, params.bctype)`. The `'reflect'` choice is not separately argued.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. (1) The manual curation string `clu.quality` is lower-cased/stripped and dropped if it is one of `garbage`, `gabrga`, `noisy`, `real?` — exactly the repo's `findClusters.m` drop list (the expert additionally drops `poor`). Everything else, including multi-units, is kept. (2) A low-firing-rate cut that reproduces `removeLowFRClusters.m`: for each of the 7 Figure-8 conditions a trial-averaged, smoothed PSTH is built, the unit's mean over all conditions and time bins is taken, and units with mean ≤ 1 Hz are dropped. Result: 520 units over 12 sessions (27–67 per session).

ii.
```python
BAD_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}
LOW_FR = 1.0

def compute_condition_masks(bp):
    return [hit | miss | no,
            hit & ~stim_enable & ~autowater,
            hit & ~stim_enable & autowater,
            miss & ~stim_enable & ~autowater,
            miss & ~stim_enable & autowater,
            hit & ~stim_enable & ~autowater & ~early,
            hit & ~stim_enable & autowater & ~early]

def compute_psth_mean_fr(session_trial, aligned_spikes, condition_masks, edges):
    for cond_mask in condition_masks:
        ...
        psth = my_smooth(counts / trix.size / DT, SMOOTH, "reflect")
    return float(np.nanmean(np.column_stack(psths)))
...
            quality = read_h5_string(f, h5_ref_at(qds, clu_idx)).strip().lower()
            if quality in BAD_QUALITIES:
                continue
            ...
            if mean_fr <= LOW_FR:
                continue
```

iii. `CONVERSION_NOTES.md`: "Quality filter: same as `findClusters(..., {'all'})`, which excludes `garbage`, `gabrga`, `noisy`, and `real?`" and "Low firing-rate filter: `> 1 Hz`, matching Figure 8 / context scripts" (`params.lowFR = 1` in `Figure8a_thru_c.m`, and the paper's "all units with firing rates exceeding 1 Hz were included"). The AI used the resulting count as its headline sanity check: "520 retained units … paper reports 522 … delta −2 … most likely comes from a minor mismatch between the packaged data here and the exact analysis snapshot used for the manuscript".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. One subtraction. `clu.trialtm` is already on the behaviour clock relative to its own trial's start, and `bp.ev.goCue` is on the same clock, so `trialtm − goCue[trial]` gives seconds from go-cue onset with no interpolation or per-session offset. Spikes outside the window simply fall outside the histogram edges.

ii.
```python
ALIGN_EVENT = "goCue"
...
    aligned = trialtm - align_times[session_trial]
...
        counts, _ = np.histogram(aligned_spikes[spike_mask], bins=edges)
```

iii. `CONVERSION_NOTES.md`: "Alignment event: `goCue`", matching `params.alignEvent = 'goCue'` and `alignSpikes.m`'s `trialtm_aligned = trialtm - event`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **10 ms bins** (`DT = 1/100`) over the window **[−3.0, +2.5] s** from the go cue → **550 bins** per trial, identical for every trial and session. `metadata.time_bin_size = 10.0`, `off_start = -3.0`, `off_end = 2.5`. Spikes are histogrammed straight into these bins (no finer binning followed by rebinning); the 400 Hz video and motion-energy streams are *linearly interpolated* onto the same bin-centre axis (not block-averaged). These numbers are the `Figure8a_thru_c.m` parameters, not the repo's `getDefaultParams.m` values (`tmin = −2.5`, `dt = 1/200`).

ii.
```python
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
...
def build_edges_and_time():
    edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
    if not np.isclose(edges[-1], TMAX):
        edges = np.append(edges, TMAX)
    time = edges[:-1] + DT / 2
    return edges, time
```
(bin centres run −2.995 … 2.495; verified from the saved pickle.)

iii. `CONVERSION_NOTES.md`: "Neural time window: `[-3.0, 2.5]` s; Time bin: `dt = 1/100` s", listed under "Implemented directly from the repository". Trajectory step 67: "the Figure 8 scripts use 12 ALM ephys sessions … with `goCue` alignment, `dt=10 ms`, `t=[-3.0, 2.5] s`, and `lowFR=1 Hz`."

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Nothing from the raw files — it is the bin-centre axis of the conversion window itself, defined by `TMIN`, `TMAX`, `DT`. It is implicitly the go cue (`bp.ev.goCue`) because that is the event everything is aligned to.

ii.
```python
    edges, time = build_edges_and_time()
    taxis = time + ADVANCE_MOVEMENT          # ADVANCE_MOVEMENT = 0.0
...
    session_input = [
        np.asarray(time[None, :], dtype=np.float32)
        for _ in range(neural_arr.shape[2])
    ]
...
        "input_names": ["time_from_go_cue_seconds"],
```

iii. Not separately argued; it follows from the alignment decision (2-d) and the window (2-e). `ADVANCE_MOVEMENT = 0.0` mirrors the repo's `params.advance_movement`, which is added to `obj.time` to build the video axis.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond taking bin centres (`edges[:-1] + DT/2`) and broadcasting the same `(1, 550)` float32 row to every trial of every session. It is continuous-valued, not binarised.

ii.
```python
    time = edges[:-1] + DT / 2
...
    session_input = [np.asarray(time[None, :], dtype=np.float32) for _ in range(...)]
```

iii. N/A — no justification needed or given.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid. `edges` is used both for the spike histogram and to derive `time`, so input sample *k* covers exactly the same 10 ms interval as neural bin *k*. The same `taxis` is also the interpolation target for motion energy and all DLC features, so every stream shares one time axis.

ii.
```python
    edges, time = build_edges_and_time()
    taxis = time + ADVANCE_MOVEMENT
...
        counts, _ = np.histogram(aligned_spikes[spike_mask], bins=edges)   # neural
...
        out[:, tr] = interp_with_nan(frame_times - vidshift - align_times[tr], me_trial, taxis)  # video
```

iii. N/A.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Two per-trial `obj.bp` fields: the instructed side `R` and the outcome flag `hit`. (`L` and `miss` are read into the `bp` dict but the lick-direction rule only needs `R` and `hit`, because within the retained trial set "not hit" implies "miss".)

ii.
```python
def lick_direction_from_trial(bp, trial_idx):
    hit = bool(bp["hit"][trial_idx])
    is_right_trial = bool(bp["R"][trial_idx])
    if hit:
        return int(is_right_trial)
    return int(not is_right_trial)
```

iii. `CONVERSION_NOTES.md`: "`lick_direction`: actual response direction, not instructed side. Correct right trials and incorrect left trials were labeled right; correct left trials and incorrect right trials were labeled left." The direction licked is not stored directly, so it is inferred from instructed side × outcome.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Two classes only: `left = 0`, `right = 1`. On a hit the animal licked the instructed port; on a miss it licked the other one. The scalar is then tiled across all 550 bins so the output is time-varying in shape but constant within a trial. There is **no third "none" class**, because ignore trials were already removed in 1-e.

ii.
```python
        lick_dir = lick_direction_from_trial(bp, tr)
        tr_out = np.vstack([
                np.full(tongue_speed.shape[0], lick_dir, dtype=np.int16),
                ...
        ])
...
        "output_values": [["left", "right"], ...],
```

iii. The AI's delivered instructions specified `Lick direction (left = 0, right = 1, per-trial)` with no third class, and it excluded ignore trials for the same reason (see 1-e). Verified in the saved pickle: `np.unique` of dimension 0 is `[0, 1]`.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial field, `obj.bp.autowater`, which marks water-cued (WC) trials.

ii.
```python
            "autowater": read_h5_vector(bp_group["autowater"]).astype(np.int16),
...
        context = 0 if bp["autowater"][tr] else 1
```

iii. Not separately argued beyond the notes' "`behavioral_context`: `WC = 0`, `DR = 1`"; `autowater` is the flag the repo's own condition strings use to separate the two contexts.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabel of the flag (autowater → WC = 0, otherwise DR = 1), tiled across the 550 bins. Observed distribution: 28.9% WC / 71.1% DR.

ii.
```python
        context = 0 if bp["autowater"][tr] else 1
...
                np.full(tongue_speed.shape[0], context, dtype=np.int16),
...
        "output_values": [..., ["WC", "DR"], ...],
```

iii. Codes follow the instruction's `WC = 0, DR = 1`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. One per-trial flag, `obj.bp.hit`. `bp.miss` and `bp.no` are read and used in the trial mask / condition masks but the outcome label itself only consults `hit`, since within the retained set "not hit" ⇒ "miss".

ii.
```python
            "hit": read_h5_vector(bp_group["hit"]).astype(np.int16),
            "miss": read_h5_vector(bp_group["miss"]).astype(np.int16),
            "no": read_h5_vector(bp_group["no"]).astype(np.int16),
...
        outcome = 1 if bp["hit"][tr] else 0
```

iii. Not separately argued; the notes only state "`outcome`: `incorrect = 0`, `correct = 1`".

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Two classes, `incorrect = 0` / `correct = 1`, tiled across the 550 bins. No `ignore` class exists because those trials were dropped in 1-e. Observed distribution: 13.6% incorrect / 86.4% correct.

ii.
```python
        outcome = 1 if bp["hit"][tr] else 0
...
                np.full(tongue_speed.shape[0], outcome, dtype=np.int16),
...
        "output_values": [..., ["incorrect", "correct"], ...],
```

iii. `CONVERSION_NOTES.md`: "Ignore / no-response trials were excluded because the requested `outcome` target is binary correct/incorrect and the paper methods state ignore trials were omitted from analyses."

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`, **bottom camera only** (`traj{2}`), features `top_tongue` and `bottom_tongue` — two landmarks on the same view, whose velocities are averaged. The side camera's `tongue` feature is not used. `obj.traj{*}.frameTimes`, `NdroppedFrames`, `featNames` and `ts[:, 0:2, :]` (x, y; the likelihood channel is not read) are used, plus `bp.ev.goCue`, `bp.ev.bitStart` and `obj.sglx.fs` / `obj.sglx.bitcode.bitstart` for the clock correction.

ii.
```python
        feat_indices = {
            "top_tongue": find_feat_index(f, bottom_group, "top_tongue", bp["Ntrials"]),
            "bottom_tongue": find_feat_index(f, bottom_group, "bottom_tongue", bp["Ntrials"]),
            "top_paw": ...,
            "bottom_paw": ...,
        }
...
        top_tongue_xvel, top_tongue_yvel = velocities["top_tongue"]
        bottom_tongue_xvel, bottom_tongue_yvel = velocities["bottom_tongue"]
        tongue_tip_xvel = 0.5 * (top_tongue_xvel + bottom_tongue_xvel)
        tongue_tip_yvel = 0.5 * (top_tongue_yvel + bottom_tongue_yvel)
```

iii. Trajectory step 73: "the main remaining design choice is how to collapse multiple tracked tongue/paw landmarks into single velocity scalars. I'm using the same aligned DLC processing as their MATLAB code, then reducing to tongue-tip speed and mean paw speed … which is the least arbitrary way to satisfy the decoder spec without discarding their tracking geometry." `CONVERSION_NOTES.md`: "Tongue speed was computed from the bottom-view tongue-tip velocity using the average of `top_tongue` and `bottom_tongue`."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. A port of the repo's `findPosition.m` → `findVelocity.m` chain. (1) Trials whose `NdroppedFrames` is NaN are skipped (positions stay NaN). (2) x and y are **linearly interpolated** from the corrected frame times onto the 10 ms `taxis`, with NaN outside the frame range; no smoothing is applied to tongue features (matching `if ~contains(feat,'tongue')` in the repo). (3) Velocity is `np.gradient` of the interpolated position — i.e. pixels **per bin**, not per second — with no baseline-derivative subtraction for tongue and no nearest-fill; NaN velocities are set to **0** ("tongue velocity 0 if not visible", exactly as `findVelocity.m` does). (4) The two landmarks' x- and y-velocities are averaged, then speed = `hypot` of the averaged components, then `nan_to_num`. No cross-camera normalisation (only one view is used, so there is no scale mismatch to reconcile).

ii.
```python
        if "tongue" not in feat_name:
            ts = mySmooth-equivalent / fillmissing ...      # not applied to tongue
        interp_xy = interp_with_nan(frame_times - vidshift - align_times[tr], xy, taxis)
...
def aligned_feature_velocity(xpos, ypos, feat_name):
    for tr in range(xpos.shape[1]):
        tsinterp = np.column_stack([xpos[:, tr], ypos[:, tr]])
        basederiv = np.nanmedian(np.diff(tsinterp, axis=0), axis=0)
        xvel[:, tr] = np.gradient(tsinterp[:, 0])
        yvel[:, tr] = np.gradient(tsinterp[:, 1])
        if "tongue" not in feat_name:
            xvel[:, tr] -= basederiv[0]
            yvel[:, tr] -= basederiv[0]          # repo's findVelocity.m uses basederiv(1) for both
            xvel[:, tr] = fill_nearest_1d(xvel[:, tr])
            yvel[:, tr] = fill_nearest_1d(yvel[:, tr])
        else:
            xvel[:, tr][np.isnan(xvel[:, tr])] = 0.0
            yvel[:, tr][np.isnan(yvel[:, tr])] = 0.0
...
        tongue_speed = np.sqrt(tongue_tip_xvel ** 2 + tongue_tip_yvel ** 2)
        tongue_speed = np.nan_to_num(tongue_speed, nan=0.0)
```

iii. `CONVERSION_NOTES.md`: "Tongue and paw velocities were derived from the same aligned DLC trajectories used by the repository's kinematics code." The AI read `findPosition.m` and `findVelocity.m` at trajectory steps 48–49 and reproduced them line for line, including the repo's own `basederiv(1)`-for-both-axes quirk (harmless for tongue, where the branch is not taken).

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Two classes per session: `1` if speed ≥ threshold, else `0`. The threshold is the **50th percentile of the strictly positive tongue-speed values** over the session's included trials and bins — *not* the 50th percentile of all bins. There is no "not visible" class; invisible-tongue bins are 0 and therefore land in class 0. Resulting split is ~85.8% low / 14.2% high (not 50/50).

ii.
```python
    positive_tongue = tongue_use[tongue_use > 0]
    if positive_tongue.size:
        tongue_thresh = float(np.nanpercentile(positive_tongue, 50))
    else:
        tongue_thresh = 0.0
...
                (tongue_speed[:, tr] >= tongue_thresh).astype(np.int16),
```

iii. Trajectory step 105 and `CONVERSION_NOTES.md`: "The repository sets invisible-tongue velocity samples to zero. Using the median over all bins collapses the threshold to zero in every session. With the requested `>= threshold` rule, that makes the tongue target a constant all-ones label, which is not meaningful for decoding. Restricting the threshold computation to positive tongue-speed bins preserves the reference-aligned trace while yielding a usable categorical target." The AI discovered this from the validator, not a priori (step 105: "The validator found one real issue: the tongue-velocity target collapsed to a constant class").

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are converted to the go-cue clock with a per-session video offset before interpolation: `frameTimes − vidshift − goCue[trial]`, where `vidshift = mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)` (a port of `findVideoOffset.m`, computed once per session). The corrected times are then the source axis for `interp1`-style interpolation onto `taxis`, the same grid as the spike histogram, so bin *k* is the same interval in both. Frame times are taken from the camera that carries the feature (bottom view for the tongue landmarks used).

ii.
```python
def matlab_mode(x):
    vals = np.asarray(x).reshape(-1); vals = vals[~np.isnan(vals)]
    uniq, counts = np.unique(vals, return_counts=True)
    return float(uniq[np.argmax(counts)])

def find_video_offset(bit_start, bitcode_bitstart, fs):
    return matlab_mode(bitcode_bitstart) / fs - matlab_mode(bit_start)
...
        vidshift = find_video_offset(bp["bitStart"], bitcode_bitstart, fs)
...
        interp_xy = interp_with_nan(frame_times - vidshift - align_times[tr], xy, taxis)
```

iii. `CONVERSION_NOTES.md`: "Video alignment: `frameTimes - vidshift - goCue`, with `vidshift = mode(bitcode.bitstart)/fs - mode(bp.ev.bitStart)`, matching `findVideoOffset.m` and `loadMotionEnergy.m`."

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The same `obj.traj` bottom-camera tracking, features **`top_paw` and `bottom_paw`** (both forepaws), x and y channels only.

ii.
```python
PAW features in feat_indices:
            "top_paw": find_feat_index(f, bottom_group, "top_paw", bp["Ntrials"]),
            "bottom_paw": find_feat_index(f, bottom_group, "bottom_paw", bp["Ntrials"]),
...
        top_paw_xvel, top_paw_yvel = velocities["top_paw"]
        bottom_paw_xvel, bottom_paw_yvel = velocities["bottom_paw"]
```

iii. `CONVERSION_NOTES.md`: "Paw speed was computed as the mean speed of `top_paw` and `bottom_paw`." Both are in the repo's `params.traj_features` list for view 2, so the AI treated them as an equally valid pair. No reliability check between the two paw features was performed.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same `findPosition`/`findVelocity` port as the tongue, but taking the **non-tongue branch**: after interpolation onto `taxis`, missing positions are filled by `fillmissing(..., 'nearest')` (implemented as `np.interp` over valid indices, which also extrapolates flat at the ends); the per-trial median frame-to-frame derivative `basederiv` is subtracted from both x and y velocity (reproducing the repo's use of `basederiv(1)` for both axes); the velocities are nearest-filled again. Then speed is computed per paw as `hypot(xvel, yvel)` and the **two paws' speeds are averaged** (speeds, not vectors). No normalisation; units are pixels per 10 ms bin.

ii.
```python
        if "tongue" not in feat_name:
            xpos[:, tr] = fill_nearest_1d(xpos[:, tr])
            ypos[:, tr] = fill_nearest_1d(ypos[:, tr])
...
        top_paw_speed = np.sqrt(top_paw_xvel ** 2 + top_paw_yvel ** 2)
        bottom_paw_speed = np.sqrt(bottom_paw_xvel ** 2 + bottom_paw_yvel ** 2)
        paw_speed = 0.5 * (top_paw_speed + bottom_paw_speed)
        paw_speed = np.nan_to_num(paw_speed, nan=0.0)
```

iii. Same as 7-a/7-b: "derived from the same aligned DLC trajectories used by the repository's kinematics code", with the mean over the two paws chosen as "the least arbitrary way to satisfy the decoder spec without discarding their tracking geometry" (step 73).

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Literal application of the spec: the per-session threshold is `np.nanpercentile(paw_speed_over_included_trials, 50)`, class `1` if `>=`, else `0`. Two classes only; there is no "not visible" class, because nearest-filling means the paw trace is never NaN. Resulting split is exactly 50.0% / 50.0% in every session.

ii.
```python
    paw_thresh = float(np.nanpercentile(paw_use, 50))
...
                (paw_speed[:, tr] >= paw_thresh).astype(np.int16),
```

iii. `CONVERSION_NOTES.md`: "`paw_velocity_bin` and `motion_energy_bin` use the literal per-session 50th percentile over all exported bins."

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identical to the tongue: bottom-camera `frameTimes − vidshift − goCue[trial]`, interpolated onto the shared `taxis`, so it lands on the same 550-bin grid as the spikes.

ii.
```python
            xpos, ypos = aligned_feature_position(
                f=f, view_group=bottom_group, feat_idx=feat_idx, feat_name=feat_name,
                n_trials=bp["Ntrials"], align_times=bp[ALIGN_EVENT],
                vidshift=vidshift, taxis=taxis,
            )
...
        interp_xy = interp_with_nan(frame_times - vidshift - align_times[tr], xy, taxis)
```

iii. Same as 7-d — one session-constant offset, one shared grid, no per-stream special casing.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file beside each data structure, field `me.data` (a cell with one trace per trial, one value per camera frame). `me.moveThresh` is also read and recorded in metadata but never used to label anything. The embedded `obj.me` copy is deliberately not used. Frame times come from the **side** camera (`obj.traj{1}`).

ii.
```python
def load_motion_energy(path):
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    me = mat["me"]
    trial_arrays = np.asarray(me.data, dtype=object).reshape(-1)
    return trial_arrays, float(me.moveThresh)
...
        frame_times_by_trial = [read_frame_times(f, side_group, tr) for tr in range(bp["Ntrials"])]
        raw_motion_energy, raw_move_thresh = load_motion_energy(spec.motion_energy_path)
```

iii. Trajectory step 79: "The embedded motion-energy field is not consistently present, so I'm following the paper code and reading the separate `motionEnergy_*.mat` files."

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond alignment: the per-frame trace is linearly interpolated onto `taxis` (NaN outside the frame range) and then nearest-filled, a direct port of `loadMotionEnergy.m`'s `interp1` + `fillmissing(me.data,'nearest')`. No smoothing, rescaling or re-derivation. Note that `fill_nearest_1d` returns **zeros** when the whole trial is NaN, which differs from MATLAB `fillmissing` (which would leave NaN).

ii.
```python
def aligned_motion_energy(frame_times_by_trial, raw_motion_energy, align_times, vidshift, taxis):
    for tr in range(n_trials):
        out[:, tr] = interp_with_nan(frame_times[tr] - vidshift - align_times[tr], me_trial, taxis)
        out[:, tr] = fill_nearest_1d(out[:, tr])

def fill_nearest_1d(x):
    idx = np.flatnonzero(~np.isnan(x))
    if idx.size == 0:
        return np.zeros_like(x)
    return np.interp(np.arange(x.size, dtype=np.float64), idx.astype(float), x[idx])
```

iii. `CONVERSION_NOTES.md`: "Motion energy interpolation: onto the neural time axis, then nearest-fill of edge NaNs, matching `loadMotionEnergy.m`" and "Motion energy followed the repository exactly up to aligned continuous values."

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Literal per-session 50th percentile over all included trials and bins, `1` if `>=`, else `0`. Two classes; no "no video" class. Resulting split is 49.9–50.0% / 50.0–50.1% per session.

ii.
```python
    me_thresh = float(np.nanpercentile(me_use, 50))
...
                (motion_energy[:, tr] >= me_thresh).astype(np.int16),
```

iii. Same as 8-c: "the literal per-session 50th percentile over all exported bins". The AI also loaded the repo's own `me.moveThresh` (10.0 in most sessions) and recorded it per session as a cross-check, but chose the instruction's percentile rule over it.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. `frameTimes(side camera) − vidshift − goCue[trial]`, interpolated onto the same `taxis` as the neural data, exactly as `loadMotionEnergy.m` does (`obj.traj{1}(trix).frameTimes`). The AI did not implement the repo's `catch` fallback (`frameTimes = (1:nframes)/400` with a 0.5 s shift) for trials whose frame times are missing.

ii.
```python
        side_group = f[h5_ref_at(traj_root, 0)]
        frame_times_by_trial = [read_frame_times(f, side_group, tr) for tr in range(bp["Ntrials"])]
...
        out[:, tr] = interp_with_nan(frame_times - vidshift - align_times[tr], me_trial, taxis)
```

iii. Same as 7-d/8-d — one session-constant offset and the shared grid.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Five mechanisms, all of which keep the trial and substitute a value rather than dropping or marking it:
- **Bad video trials:** `NdroppedFrames` NaN → the trial's positions are left NaN (repo behaviour). (I checked: 0 such trials in these 12 sessions.)
- **Missing / all-NaN frame times:** `interp_with_nan` returns all-NaN when fewer than 2 valid source points exist; `fill_nearest_1d` then converts an all-NaN trace to **zeros**. (2 trials in these sessions have all-NaN side-camera frame times, so their motion energy becomes 0 → class "low".)
- **Untracked DLC frames:** tongue NaN → velocity 0 (repo behaviour); non-tongue NaN → nearest-fill (repo behaviour).
- **All-NaN derivative:** `basederiv` falls back to zeros, suppressing the `RuntimeWarning` the AI saw at step 94/95.
- **Session-level guards:** a session with <2 included trials or 0 surviving units raises rather than emitting a malformed entry.
No NaN survives into the pickle, and no data-quality flag is carried into the outputs — a bin with no video is indistinguishable from a genuinely slow bin.

ii.
```python
        ndropped = read_ndropped_frames(f, view_group, tr)
        if np.isnan(ndropped):
            continue
...
        if valid.sum() < 2:
            return np.full_like(x_new, np.nan, dtype=np.float64)
...
    if idx.size == 0:
        return np.zeros_like(x)
...
        if np.all(np.isnan(diff_xy)):
            basederiv = np.zeros(2, dtype=np.float64)
...
        if included_trials.size < 2:
            raise RuntimeError(f"{spec.stem}: fewer than 2 usable trials after filtering")
        if not neural_by_trial:
            raise RuntimeError(f"{spec.stem}: no units survived quality and low-FR filtering")
```

iii. Trajectory step 95: "I've already hit one edge case from the raw video data: some aligned feature traces are entirely missing on some trials, which produces all-NaN velocity baselines; I'll either tolerate that with zero-fill or patch it if it contaminates outputs." The choice of zero-fill follows the repo's own convention of setting invisible-tongue velocity to zero. Not documented in `CONVERSION_NOTES.md`.

## 11-a. What are the most time-consuming steps of the code?

i. I profiled one session (`JEB6_2021-04-18`): 6.2 s total, so ~75 s for the full 12-session conversion. The breakdown is dominated by **per-trial HDF5 reads inside `aligned_feature_position`: 4.68 s of 6.23 s (75%)**, of which `h5py.Dataset.__getitem__` alone is 3.75 s across 10,192 calls — `read_feature_xy` 3.08 s (1,528 calls = 4 features × 382 trials), `read_frame_times` 0.81 s (1,910 calls), `read_ndropped_frames` 0.66 s. Everything downstream is cheap: `aligned_feature_velocity` 0.59 s, `build_neural_trials` 0.54 s, `nanmedian` for `basederiv` 0.44 s, `my_smooth` 0.24 s over 7,754 calls, `np.histogram` 0.21 s. The neural pipeline — the part with the most arithmetic — is a small fraction of the runtime.

ii.
```python
def aligned_feature_position(f, view_group, feat_idx, feat_name, n_trials, align_times, vidshift, taxis):
    for tr in range(n_trials):
        ndropped = read_ndropped_frames(f, view_group, tr)      # 1 HDF5 deref per trial per feature
        if np.isnan(ndropped):
            continue
        frame_times = read_frame_times(f, view_group, tr)       # 1 more
        xy = read_feature_xy(f, view_group, tr, feat_idx)       # reads the FULL (feat, xyz, frame) array
        ...

def read_feature_xy(f, view_group, trial_idx, feat_idx):
    ts_ds = f[h5_ref_at(view_group["ts"], trial_idx)]
    arr = np.asarray(ts_ds[()], dtype=np.float64)               # whole array materialised
    return arr[feat_idx, 0:2, :].T                              # 2 of ~30 columns kept
```

iii. No efficiency discussion appears anywhere in `CONVERSION_NOTES.md`, `README.md` or the trajectory; the AI never profiled or commented on runtime.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Four:
- `build_neural_trials` loops over included trials calling `np.histogram` once each; a single `np.histogram2d(session_trial, aligned_spikes, bins=[trial_edges, edges])` would produce the whole trial × bin matrix in one call (this is exactly what the expert solution does).
- `compute_psth_mean_fr` loops over 7 conditions with an `np.isin` over all spikes each time; the same `histogram2d` result could be reduced per condition with matrix multiplication.
- `my_smooth` loops over columns with `np.convolve`; `scipy.ndimage.convolve1d` / `scipy.signal.fftconvolve` along an axis would do all neurons or trials at once. It is called 7,754 times for one session.
- `aligned_feature_velocity` loops over trials to call `np.gradient` and `np.nanmedian` per column; both accept an `axis` argument and could operate on the full `(time, trials)` array.

`aligned_feature_position` and `aligned_motion_energy` genuinely need a per-trial loop (ragged frame counts), but see 11-c for the redundant work inside them.

ii.
```python
    for col, tr in enumerate(included_trials):
        spike_mask = session_trial == tr
        counts, _ = np.histogram(aligned_spikes[spike_mask], bins=edges)
        out[:, col] = my_smooth(counts / DT, SMOOTH, "reflect")
...
    out = np.zeros_like(x_filt)
    for j in range(x_filt.shape[1]):
        out[:, j] = np.convolve(x_filt[:, j], kern, mode="same")
...
    for tr in range(xpos.shape[1]):
        basederiv = np.nanmedian(diff_xy, axis=0)
        xvel[:, tr] = np.gradient(tsinterp[:, 0])
```

iii. Not discussed by the AI.

## 11-c. What processing does the code repeat multiple times?

i. Several things:
- **Frame times and `NdroppedFrames` are re-read from HDF5 once per (feature, trial)** instead of once per (view, trial). With 4 bottom-camera features that is a 4× redundancy — 1,528 `NdroppedFrames` reads and 1,528 `frameTimes` reads where 382 of each would do, ~1.1 s per session.
- **`read_feature_xy` materialises the entire `(n_features, 3, n_frames)` `ts` array for every feature**, so the same array is read 4 times per trial and >90% of each read is discarded. This is the single largest cost in the program (3.1 s of 6.2 s).
- **`h5_ref_at` rebuilds the full object-reference array on every call** (`np.asarray(dataset[()], dtype=object).reshape(-1)`) just to index one element — 5,070 calls per session.
- **`basederiv` (`np.nanmedian` over the diff) is computed for every trial of every feature including tongue features**, where the result is immediately discarded by the `if "tongue" not in feat_name` guard — 0.44 s per session of pure waste.
- **`fill_nearest_1d` is applied twice** to non-tongue data: once to positions in `aligned_feature_position` and again to the velocities derived from those already-filled positions (this mirrors the repo, but the second pass can never find a NaN except where the whole trial is NaN).
- `find_feat_index` re-reads `featNames` per trial until a match is found, once per feature.

ii.
```python
def h5_ref_at(dataset: h5py.Dataset, index: int):
    refs = np.asarray(dataset[()], dtype=object).reshape(-1)   # whole array, every call
    return refs[index]
...
        basederiv = np.nanmedian(diff_xy, axis=0)              # computed before the tongue check
        ...
        if "tongue" not in feat_name:
            xvel[:, tr] = xvel[:, tr] - basederiv[0]
```

iii. Not discussed by the AI.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Both dead computation and dead data:
- `SIDE_FEATURES` and `BOTTOM_FEATURES` (the repo's `params.traj_features` lists) are defined at module scope and **never referenced**.
- `bp["L"]`, `bp["ev"]["sample"]`, `bp["ev"]["delay"]` are read and never used; `bp["no"]` is used only inside condition mask 0.
- `me.moveThresh` is loaded and stored in `session_info` but never used to threshold anything.
- `compute_condition_masks` builds 7 masks, of which conditions 5 and 6 (`hit&~stim&(~)autowater&~early`) are strict subsets of conditions 1 and 2, so the low-FR statistic double-counts them; all 7 exist only to reproduce a mean that a single `(hit|miss|no)` PSTH would approximate.
- `load_trial_spikes` returns `trialtm` which the caller throws away (`_trialtm`).
- `kept_cluster_indices`, `cluster_ids` and `prefilter_unit_count` are accumulated and never read (`prefilter_unit_count` is not even reported).
- `basederiv` for tongue features (see 11-c).
- `taxis = time + ADVANCE_MOVEMENT` with `ADVANCE_MOVEMENT = 0.0` — a no-op kept for fidelity to `params.advance_movement`.
- `interp_with_nan`'s 2-D branch runs a per-column loop for the `xy` case where both columns share a validity mask most of the time.
- The whole `subset_data` / `sample_data.pkl` path (a second 87 MB pickle) is auxiliary to the deliverable.

ii.
```python
SIDE_FEATURES = ["tongue", "left_tongue", "right_tongue", "jaw", "trident", "nose"]   # never used
...
            "sample": read_h5_vector(ev_group["sample"]),   # never used
            "delay": read_h5_vector(ev_group["delay"]),     # never used
...
        kept_cluster_indices = []
        prefilter_unit_count = 0
        cluster_ids = []                                   # all written, none read
...
        session_trial, _trialtm, aligned = load_trial_spikes(...)
```

iii. Not discussed by the AI; the unused feature lists and extra `bp` fields look like leftovers from transcribing `getDefaultParams.m`.
