# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not glob the data folders. It hard-codes a list of **12 sessions** as `SessionSpec(animal, date, folder, probe)` records, all of them in `data/Ephys_Behavior` (the fixed-delay folder), and each with the single probe number transcribed from the authors' `load<ANM>_ALMVideo.m` files. The 12 sessions are exactly the cohort loaded by the paper's `Scripts/Figure 8/Figure8a_thru_c.m`, i.e. the animals JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, JEB19. The remaining 13 fixed-delay sessions in the same folder (JEB13 ×5, JEB14 ×4, JEB15 ×4) and all 22 sessions in `RandomizedDelay_Ephys_Behavior` are not loaded at all.

Each session's `data_structure_<anm>_<date>.mat` is read with `h5py` only (no v5/`scipy.io` fallback reader for the data structure — all 12 files in this cohort are MATLAB v7.3). References inside the cell arrays are dereferenced by hand (`read_ref_array`, `read_ref_string`, `read_cell_string_list`). The companion `motionEnergy_<anm>_<date>.mat` is read separately with `scipy.io.loadmat`. The file is opened twice per session: once in `process_session` (behaviour + spikes) and again in `compute_behavioral_outputs` (video).

ii.
```python
SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", "Ephys_Behavior", 2),
    SessionSpec("JEB7", "2021-04-29", "Ephys_Behavior", 1),
    ...
    SessionSpec("JEB19", "2023-04-21", "Ephys_Behavior", 1),
]
```

```python
def process_session(spec: SessionSpec):
    with h5py.File(spec.data_path, "r") as h5:
        bp = load_bp_fields(h5)
        ...
        clusters = load_probe_clusters(h5, spec.probe)
```

```python
def read_motion_energy(path: Path):
    loaded = loadmat(path, squeeze_me=True, struct_as_record=False)
    me = loaded["me"]
    data = me.data
    if not isinstance(data, np.ndarray) and hasattr(data, "data"):
        data = data.data
    return data, float(me.moveThresh)
```

```python
def build_dataset():
    sessions = [process_session(spec) for spec in SESSION_SPECS]
```

iii. From the trajectory (steps 49–62): the AI read `Figure8a_thru_c.m`, `Figure3h.m` and `Figure3i.m` and observed that the repository splits into "a two-context cohort used for Figure 8 and a broader DR cohort used for Figure 3". Because *behavioral context (WC vs DR)* is one of the required decoder outputs, it chose the Figure 8 cohort: *"The context cohort is now clear: the Figure 8 loader set gives exactly 12 sessions, and those are the ones with substantial WC blocks rather than a few incidental `autowater` trials. I'm following the Figure 8 preprocessing path from here … instead of mixing in DR-only or randomized-delay sessions."* It supported this with a per-session `autowater` census (step 56), which showed JEB13/JEB14 sessions with 0–55 autowater trials against 78–122 for the Figure 8 animals. Note that the same census script crashed on the first `RandomizedDelay` file (v7.3 reader on a v5 file), so the randomized-delay sessions were never actually inspected before being excluded. Probe numbers were justified as "the exact probe assignments from the paper's loader scripts" (step 62).

## 1-b. How are the data split into subjects?

i. The subject is the `animal` field of the hard-coded `SessionSpec`, which is also the prefix of the session id. `subjects` is built in order of first appearance across `SESSION_SPECS`, and `subject_idx` indexes into it per session. This yields 7 subjects (`['JEB6', 'JEB7', 'EKH1', 'EKH3', 'JGR2', 'JGR3', 'JEB19']`) over the 12 sessions, with `subject_idx = [0,1,1,2,3,4,4,5,6,6,6,6]`.

ii.
```python
@dataclass(frozen=True)
class SessionSpec:
    animal: str
    date: str
    ...
    @property
    def session_id(self) -> str:
        return f"{self.animal}_{self.date}"
```

```python
subjects = []
subject_lookup = {}
subject_idx = []
for spec in SESSION_SPECS:
    if spec.animal not in subject_lookup:
        subject_lookup[spec.animal] = len(subjects)
        subjects.append(spec.animal)
    subject_idx.append(subject_lookup[spec.animal])
```

iii. Not discussed explicitly in the trajectory beyond the final summary ("12 sessions, 7 mice"). The animal id is carried in the loader-script metadata (`meta(end).anm`) and in the filename, which is how the authors' own scripts identify animals; the AI mirrors that by storing it in the spec rather than reading `obj.meta.anm`.

## 1-c. How are the data split into sessions?

i. One entry of `SESSION_SPECS` = one `data_structure_*.mat` file = one element of `neural`/`input`/`output`. The folder is stored explicitly in the spec (all 12 are `Ephys_Behavior`), so no search across folders is needed. Exactly one probe is used per session (`spec.probe`); no session in this cohort was recorded with two probes, so no probe concatenation logic exists. Sessions are processed independently by `process_session` and appended in the fixed list order.

ii.
```python
    @property
    def data_path(self) -> Path:
        return DATA_ROOT / self.folder / f"data_structure_{self.session_id}.mat"

    @property
    def motion_energy_path(self) -> Path:
        return DATA_ROOT / self.folder / f"motionEnergy_{self.session_id}.mat"
```

```python
    clusters = load_probe_clusters(h5, spec.probe)
```

iii. Same justification as 1-a: the session/probe pairs are lifted from the authors' `load<ANM>_ALMVideo.m` loader scripts that `Figure8a_thru_c.m` calls.

## 1-d. How are the data split into trials?

i. A trial is one index into the per-trial fields of `obj.bp`. `bp.Ntrials` gives the count, and every behavioural flag (`hit`, `miss`, `no`, `early`, `R`, `L`, `autowater`, `stim.enable`) and every event time (`ev.goCue`, `ev.sample`, `ev.delay`, `ev.bitStart`) is read as a flat vector of that length. Spikes carry `clu.trial` (1-based trial number) and `clu.trialtm` (time within trial), so trial boundaries never have to be reconstructed. Video is stored per trial as one cell entry per trial in `obj.traj{view}`, and motion energy as one vector per trial. The arrays are **not** explicitly truncated to `Ntrials` (I verified that in all 12 sessions of this cohort every field already has exactly `Ntrials` entries, and `obj.traj` has exactly `Ntrials` cells, so no truncation was needed).

ii.
```python
def load_bp_fields(h5: h5py.File) -> dict[str, np.ndarray]:
    bp = h5["obj"]["bp"]
    ev = bp["ev"]
    return {
        "Ntrials": int(np.asarray(bp["Ntrials"])[0, 0]),
        "hit": np.asarray(bp["hit"]).reshape(-1).astype(bool),
        ...
        "goCue": np.asarray(ev["goCue"]).reshape(-1),
    }
```

```python
def align_and_histogram(trial_ids, aligned_times, n_trials):
    counts = np.zeros((n_trials, TIME.size), dtype=np.float64)
    ...
    valid = (bins >= 0) & (bins < TIME.size) & (trial_ids >= 1) & (trial_ids <= n_trials)
    np.add.at(counts, (trial_ids[valid] - 1, bins[valid]), 1.0)
```

iii. Implicit: the Bpod table defines trials directly, one go cue per trial. The AI inspected the HDF5 layout at length (steps 33–48) to confirm that `bp`, `clu`, and `traj` are all indexed by trial. The explicit `trial_ids >= 1 & trial_ids <= n_trials` guard in the histogram shows it deliberately protected against spike trial numbers outside the Bpod table.

## 1-e. How are trials filtered based on quality controls?

i. Two filters, both from the paper's analysis conditions: **early-lick** trials (`bp.early`) and **photostimulation** trials (`bp.stim.enable`) are dropped. The surviving indices (`valid_trials`) are used identically for neural, input and output, so all three streams stay in register. Two session-level guards also exist: a session with fewer than 10 surviving units or fewer than 2 surviving trials raises an error (neither triggers on this cohort). No neural-recording-length filter is applied. Across the 12 sessions this keeps **3,116 of 3,626 trials**.

ii.
```python
valid_mask = (~bp["early"]) & (~bp["stim_enable"])
valid_trials = np.flatnonzero(valid_mask)
```

```python
if n_neurons < 10:
    raise ValueError(f"{spec.session_id}: only {n_neurons} neurons after filtering")
if len(valid_trials) < 2:
    raise ValueError(f"{spec.session_id}: fewer than 2 valid trials after filtering")
```

iii. From step 31: *"The repository treats `autowater` as the WC-context flag, excludes `early` trials in the main analyses."* At step 89 the AI explicitly checked whether more filtering was needed: *"Before I edit files, I'm checking the per-trial `haveEphys` and `haveVid` flags so I can decide whether to drop missing-neural trials and keep missing-video trials with the requested 'not visible / no video' labels."* The check (step 90) showed `haveEphys missing 0` and `haveVid missing 0` for all 12 sessions, so no further trial filter was added. Trials with no video are kept and labelled rather than dropped, per the prompt's three-class movement outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` — the spike-sorted clusters of the single probe named in the session spec. Three fields per cluster are read: `quality` (curation label, a char array behind an HDF5 reference), `trial` (1-based trial of each spike), and `trialtm` (spike time relative to that trial's start). `obj.bp.ev.goCue` supplies the alignment time, and the behavioural flags supply the condition masks used by the firing-rate filter.

ii.
```python
def load_probe_clusters(h5: h5py.File, probe_index: int):
    clu_refs = h5["obj"]["clu"][()]
    probe_group = h5[clu_refs[probe_index - 1, 0]]
    n_clusters = probe_group["quality"].shape[0]

    clusters = []
    for clu_idx in range(n_clusters):
        quality = read_ref_string(h5, probe_group["quality"][clu_idx, 0]).strip()
        clusters.append({
            "quality": quality,
            "trial": read_ref_array(h5, probe_group["trial"][clu_idx, 0]).astype(np.int64),
            "trialtm": read_ref_array(h5, probe_group["trialtm"][clu_idx, 0]).astype(np.float64),
        })
    return clusters
```

iii. Driven by the authors' `alignSpikes.m` and `getSeq.m`, which the AI read at steps 22–23. Step 25: *"The core pipeline bins spikes at `dt = 0.01 s`, aligns with `obj.bp.ev.goCue`, and keeps all non-garbage units above `1 Hz`."* `trialtm` is already on the behaviour clock relative to trial start, which is why nothing else is needed.

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed into the 10 ms bin grid, divided by the bin width to give spikes/s, and then smoothed along time with a **causal** Gaussian: `gausswin(15)` with the first half of the kernel zeroed and the remainder renormalised, applied with `'reflect'` boundary handling (the first 15 samples of the series are prepended before convolution and trimmed afterwards). This is a line-by-line reimplementation of the authors' `mySmooth.m` with `params.smooth = 15` and `params.bctype = 'reflect'`. Values are stored as `float32` firing rates in Hz. No normalisation, baseline subtraction or z-scoring. Only one probe contributes per session.

ii.
```python
def gausswin(n: int, alpha: float = 2.5) -> np.ndarray:
    idx = np.arange(n) - (n - 1) / 2.0
    denom = (n - 1) / 2.0
    return np.exp(-0.5 * (alpha * idx / denom) ** 2)


def my_smooth(x, n, bctype="none"):
    ...
    if bctype == "reflect":
        x_filt = np.concatenate([x[:n], x], axis=0)
        trim = n
    ...
    kernel = gausswin(n)
    kernel[: len(kernel) // 2] = 0.0          # causal
    kernel /= kernel.sum()
    out = np.stack([np.convolve(x_filt[:, j], kernel, mode="same")
                    for j in range(x_filt.shape[1])], axis=1)
    return out[trim:]
```

```python
cluster_trials = counts[valid_mask] / DT
cluster_trials = my_smooth(cluster_trials.T, SMOOTH_WINDOW, BCTYPE).T
good_trial_mats.append(cluster_trials.astype(np.float32))
```

iii. Step 62: *"The remaining piece is reproducing the MATLAB spike smoothing and feature extraction closely enough in Python, so I'm reading those helper functions before I implement the converter"*, followed by reading `mySmooth.m` (step 63). Final summary: *"I matched the repository's `goCue` alignment, 10 ms bins over `[-3.0, 2.5] s`, causal Gaussian smoothing … and the `>1 Hz` unit filter."*

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both copied from the authors' pipeline.
1. **Curation label**: the cluster's `quality` string is stripped and lower-cased and dropped if it is one of `garbage`, `gabrga`, `noisy`, `real?` — exactly the four labels excluded by `findClusters.m` under `params.quality = {'all'}`. Multi-units and unlabelled clusters are kept.
2. **Mean firing rate > 1 Hz**, computed the way `removeLowFRClusters.m` does it: PSTHs are first built for the seven Figure 8 trial conditions (all trials; DR/WC × hit/miss; DR/WC hits without early licks), each smoothed, and the unit is kept only if the grand mean over conditions and time exceeds 1 Hz.

Across the 12 sessions this leaves **520 units** (27–67 per session).

ii.
```python
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}
LOW_FR_HZ = 1.0
```

```python
def make_condition_masks(bp):
    ...
    return [hit | miss | no,
            hit & ~stim & ~autowater,
            hit & ~stim & autowater,
            miss & ~stim & ~autowater,
            miss & ~stim & autowater,
            hit & ~stim & ~autowater & ~early,
            hit & ~stim & autowater & ~early]
```

```python
quality = str(cluster["quality"]).strip().lower()
if quality in QUALITY_EXCLUDE:
    continue
...
for cond_mask in condition_masks:
    psth = counts[cond_mask].sum(axis=0) / n_cond / DT
    psth = my_smooth(psth, SMOOTH_WINDOW, BCTYPE)
    psths.append(psth)
mean_fr = float(np.mean(np.stack(psths, axis=1)))
if mean_fr <= LOW_FR_HZ:
    continue
```

iii. Read directly off `findClusters.m` (step 65) and `removeLowFRClusters.m` (step 64); the condition list is transcribed from `params.condition` in `Figure8a_thru_c.m`. Step 72: *"I have the neural side pinned down: 10 ms bins, causal Gaussian smoothing, Figure 8 trial conditions, and the same low-FR filter."*

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. A single subtraction. For every spike, `trialtm` (already relative to its own trial's start, on the behaviour clock) minus `goCue` of that same trial gives seconds from go-cue onset. The resulting times are histogrammed into the fixed bin grid; spikes falling outside `[-3.0, 2.5]` s are discarded by the bin-range mask. No interpolation or per-trial offset is involved. The camera streams need an extra clock correction (see 7-d); the spikes do not.

ii.
```python
trial_ids = np.asarray(cluster["trial"], dtype=np.int64)
trialtm = np.asarray(cluster["trialtm"], dtype=np.float64)
aligned_times = trialtm - bp["goCue"][trial_ids - 1]
counts = align_and_histogram(trial_ids, aligned_times, n_trials)
```

```python
bins = np.searchsorted(EDGES, aligned_times, side="right") - 1
valid = (bins >= 0) & (bins < TIME.size) & (trial_ids >= 1) & (trial_ids <= n_trials)
np.add.at(counts, (trial_ids[valid] - 1, bins[valid]), 1.0)
```

iii. This is `alignSpikes.m` with `params.alignEvent = 'goCue'`, which the AI read at step 23 and which is also what the task instructions require ("Temporally align based on **Go cue** onset"). Metadata records `'temporal_alignment_event': "Go cue / water delivery onset (obj.bp.ev.goCue)"` — noting that in WC trials the same `goCue` field marks water delivery.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **10 ms bins** (`DT = 0.01`) spanning **−3.0 to +2.5 s** from the go cue, i.e. **550 bins** per trial, identical for every trial and session. These are exactly the authors' Figure 8 parameters (`params.dt = 1/100`, `params.tmin = -3`, `params.tmax = 2.5`). Spikes are counted directly into this grid — there is no rebinning of an intermediate resolution. The camera streams (400 fps) are brought onto the same grid by *linear interpolation at bin centres* rather than by averaging the frames inside each bin, so roughly 3 of every 4 video frames are discarded; this is what `findPosition.m` does (`interp1(frameTimes - vidshift - alignEv, ts, taxis)`). The grid is built once at module level and shared by neural, input and all outputs.

ii.
```python
DT = 0.01
TMIN = -3.0
TMAX = 2.5
EDGES = np.arange(TMIN, TMAX + 1e-9, DT)
TIME = EDGES[:-1] + (DT / 2.0)
```

```python
"time_bin_size": DT * 1000.0,
"off_start": TMIN,
"off_end": TMAX,
```

iii. Step 25: *"The core pipeline bins spikes at `dt = 0.01 s`"* — taken from `params.dt = 1/100` in `Figure8a_thru_c.m`, with `tmin`/`tmax` from the same block. The AI deliberately used the Figure 8 parameter set throughout rather than a generic default.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Nothing in the raw data — it is defined by the conversion. It is the vector of bin centres of the shared time grid, which is itself defined by the go-cue alignment (2-d) and the window/bin size (2-e). The same vector is used for every trial of every session.

ii.
```python
EDGES = np.arange(TMIN, TMAX + 1e-9, DT)
TIME = EDGES[:-1] + (DT / 2.0)
```

```python
INPUT_NAMES = ["time_from_go_cue"]
```

iii. Implicit: the prompt asks for "Time from go cue onset in seconds (continuous, time-varying)", and the go cue is the alignment event, so the bin-centre axis is the variable by construction.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond building the axis. `TIME` is cast to `float32`, reshaped to `(1, 550)` and copied once per trial, giving `input` arrays of shape `(n_input=1, n_timepoints=550)` running from −2.995 to +2.495 s.

ii.
```python
inputs = [TIME[np.newaxis, :].astype(np.float32).copy() for _ in valid_trials]
```

iii. N/A — no processing to justify.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid. `EDGES` is used both to histogram the go-cue-aligned spike times and to define `TIME = EDGES[:-1] + DT/2`, so bin *k* of the input covers exactly the same interval as bin *k* of the neural matrix and of every output channel. Alignment is exact by construction, with no resampling.

ii.
```python
bins = np.searchsorted(EDGES, aligned_times, side="right") - 1   # neural
...
TIME = EDGES[:-1] + (DT / 2.0)                                   # input
...
return np.asarray(interp(TIME), dtype=np.float64)                # video outputs
```

iii. N/A.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial flags of `obj.bp`: the instructed side `R` and `L`, and the outcome flags `hit` and `miss`, plus `no` to identify trials with no lick. The licked side is not recorded directly, so it is inferred from instructed side × outcome.

ii.
```python
"hit": np.asarray(bp["hit"]).reshape(-1).astype(bool),
"miss": np.asarray(bp["miss"]).reshape(-1).astype(bool),
"no": np.asarray(bp["no"]).reshape(-1).astype(bool),
"R": np.asarray(bp["R"]).reshape(-1).astype(bool),
"L": np.asarray(bp["L"]).reshape(-1).astype(bool),
```

iii. Follows the authors' `getOutcome.m` and the `R&hit` / `L&miss` style condition strings used throughout the repository (read at steps 53 and 67), which encode the same inference.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A three-way relabelling, evaluated per trial in a short cascade: a `no` trial is `none` (2); `R&hit` or `L&miss` is `right` (1); `L&hit` or `R&miss` is `left` (0); anything else raises. The value is then broadcast across all 550 bins so that the per-trial labels share the same time-varying output tensor as the movement channels. Resulting distribution: left 39.8%, right 37.7%, none 22.5%.

ii.
```python
def trial_choice_code(bp, trial_idx: int) -> int:
    if bp["no"][trial_idx]:
        return 2
    if (bp["R"][trial_idx] and bp["hit"][trial_idx]) or (bp["L"][trial_idx] and bp["miss"][trial_idx]):
        return 1
    if (bp["L"][trial_idx] and bp["hit"][trial_idx]) or (bp["R"][trial_idx] and bp["miss"][trial_idx]):
        return 0
    raise ValueError(f"Unable to infer lick direction for trial {trial_idx}")
```

```python
trial_output = np.empty((len(OUTPUT_NAMES), TIME.size), dtype=np.int64)
trial_output[0] = lick_code
```

```python
OUTPUT_VALUES = [["left", "right", "none"], ...]
```

iii. Step 96: *"emit repeated per-trial labels plus time-varying kinematic categories so the decoder sees a single consistent output tensor per trial."* The `raise` is a deliberate assertion that the three outcome flags are exhaustive; it never fired.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial flag, `obj.bp.autowater`. It is true on water-cued (WC) trials, where water is delivered from a random port without any cue; all other trials are delayed-response (DR).

ii.
```python
"autowater": np.asarray(bp["autowater"]).reshape(-1).astype(bool),
```

iii. Step 31: *"The repository treats `autowater` as the WC-context flag."* The AI confirmed this by reading the Figure 8 condition strings (`hit&~stim.enable&autowater` = "all AW hits") and `getBlockNum_AltContextTask.m` (step 68), and by the per-session `autowater` census at step 56.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling with the prompt's codes: `autowater → 0` (WC), otherwise `1` (DR), broadcast across all 550 bins. Because the cohort is restricted to the two-context sessions, every session contains both classes; overall 31.5% WC / 68.5% DR.

ii.
```python
context_code = 0 if bp["autowater"][trial_idx] else 1
...
trial_output[1] = context_code
```

```python
OUTPUT_VALUES = [..., ["WC", "DR"], ...]
```

iii. The prompt specifies the order "(WC, DR)", and the AI's cohort choice (1-a) was made precisely so this output would be well populated in every session: *"those are the ones with substantial WC blocks rather than a few incidental `autowater` trials."* `session_info` records `wc_trials_kept` / `dr_trials_kept` per session.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Three mutually exclusive per-trial flags of `obj.bp`: `hit`, `miss`, and `no`. Unlike the reference, `no` is read explicitly rather than inferred as the complement.

ii.
```python
"hit": ..., "miss": ..., "no": np.asarray(bp["no"]).reshape(-1).astype(bool),
```

iii. Read off `getOutcome.m` (step 67) and the repository's condition strings, which treat `hit|miss|no` as the full trial set.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A three-way relabelling in the prompt's order: `miss → 0` (incorrect), `hit → 1` (correct), `no → 2` (ignore), raising if none applies; broadcast across all 550 bins. Ignore trials are kept as their own class rather than dropped, unlike the paper's analyses. Distribution: 10.6% incorrect, 66.9% correct, 22.5% ignore.

ii.
```python
def trial_outcome_code(bp, trial_idx: int) -> int:
    if bp["miss"][trial_idx]:
        return 0
    if bp["hit"][trial_idx]:
        return 1
    if bp["no"][trial_idx]:
        return 2
    raise ValueError(f"Unable to infer outcome for trial {trial_idx}")
```

iii. The prompt fixes the class order `(incorrect, correct, ignore)`. Keeping ignore trials is required for the "none" class of lick direction and for the three-class outcome the prompt asks for.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`, using **all tongue features of both cameras**: on the side view (`traj{1}`) `tongue`, `left_tongue`, `right_tongue`; on the bottom view (`traj{2}`) `top_tongue`, `topleft_tongue`, `bottom_tongue`, `bottomleft_tongue` — whichever of these are present in that session's `featNames`. Per trial and view it reads `ts` (frames × [x, y, likelihood] × features), `frameTimes`, and `NdroppedFrames`. `obj.sglx.fs`, `obj.sglx.bitcode.bitstart` and `bp.ev.bitStart` are also needed, for the video clock correction.

ii.
```python
tongue_indices = {
    0: [feat_names[0].index(name) for name in ["tongue", "left_tongue", "right_tongue"]
        if name in feat_names[0]],
    1: [feat_names[1].index(name) for name in
        ["top_tongue", "topleft_tongue", "bottom_tongue", "bottomleft_tongue"]
        if name in feat_names[1]],
}
```

```python
for feat_idx in tongue_indices[0]:
    coords = side_ts[:, :2, feat_idx]
    coords_interp = interpolate_coords(coords, side_ft, align_time, vidshift)
    speed, visible = feature_speed(coords_interp, tongue_feature=True)
```

iii. The feature lists are taken verbatim from `params.traj_features` in `Figure8a_thru_c.m`. Step 78: *"The paper never reduces 'tongue' or 'paw' to a single canonical feature for this decoder task, so I'm inspecting the actual DLC feature names in these sessions. I want to choose a scalar summary that's consistent across all 12 sessions and preserves the paper's visibility handling, rather than guessing at one landmark."*

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Per trial and per tongue feature: (1) the raw x, y traces are **linearly interpolated** from the camera's frame times onto the 550 bin centres, with NaN outside the frame range; (2) **no smoothing** is applied, following `findPosition.m`'s "if tongue, don't smooth"; (3) `np.gradient` along time gives x and y velocity in pixels per bin, with non-finite entries set to 0 (the `findVelocity.m` rule "set tongue velocity to 0 if not visible"); **no baseline-derivative subtraction** is applied to the tongue, again per `findVelocity.m`; (4) speed is `hypot(vx, vy)`; (5) a per-bin `visible` mask is defined as "both x and y finite after interpolation" — i.e. the DeepLabCut likelihood gate the authors already baked in as NaNs; (6) the per-feature speeds are averaged **only over the features visible in that bin**, via a per-timepoint Python loop, and the bin is visible if any feature is.

Notably the two camera views are averaged together **without any per-view rescaling**, so side-view pixels and bottom-view pixels enter the same average and the same session threshold.

ii.
```python
def interpolate_coords(coords, frame_times, align_time, vidshift):
    interp = interp1d(frame_times - vidshift - align_time, coords, axis=0,
                      kind="linear", bounds_error=False, fill_value=np.nan,
                      assume_sorted=True)
    return np.asarray(interp(TIME), dtype=np.float64)
```

```python
def feature_speed(coords_interp, tongue_feature: bool):
    visible = np.isfinite(coords_interp).all(axis=1)
    if tongue_feature:
        vel = np.gradient(coords_interp, axis=0)
        vel[~np.isfinite(vel)] = 0.0
    ...
    speed = np.sqrt((vel ** 2).sum(axis=1))
    return speed, visible
```

```python
def aggregate_speeds(speeds, visibles):
    speed_stack = np.stack(speeds, axis=0)
    vis_stack = np.stack(visibles, axis=0)
    visible_any = vis_stack.any(axis=0)
    out = np.full(TIME.size, np.nan)
    for t in range(TIME.size):
        mask = vis_stack[:, t]
        if np.any(mask):
            out[t] = float(np.mean(speed_stack[mask, t]))
    return out, visible_any
```

iii. The whole path mirrors `findPosition.m` / `findVelocity.m`, which the AI read at steps 41–42, including the tongue-specific exceptions (no smoothing, no baseline subtraction, NaN→0). Step 78 explains the choice to pool the DLC tongue features rather than pick one landmark. The trajectory contains no discussion of the differing pixel scales of the two cameras and no normalisation step.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A single **session-wide 50th percentile** of the combined tongue speed, computed over the visible bins of all kept trials pooled together. Bins at or above it are 1, below it are 0, and bins where no tongue feature was visible are 2 (`not visible`). The realised distribution is 4.8% / 4.8% / 90.5% — the tongue is out of view in the great majority of bins, close to the reference's 87.5%.

ii.
```python
tongue_thresh = float(np.nanpercentile(tongue_series[tongue_visible], 50))
...
trial_output[3] = np.where(~tongue_visible[out_idx], 2,
                           (tongue_series[out_idx] >= tongue_thresh).astype(np.int64))
```

```python
OUTPUT_VALUES = [..., ["<50th percentile", ">=50th percentile", "not visible"], ...]
```

iii. Directly from the prompt's specification ("discretized with per-session threshold: 0: < 50th percentile, 1: >= 50th percentile, 2: not visible"). Step 116: *"The only potentially surprising statistic is that tongue is labeled 'not visible' most of the time, which is plausible for this task"* — the AI checked this against the trained decoder rather than changing the rule.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Two corrections, then the shared grid. The camera and behaviour clocks differ by a session constant recovered from the bitcode pulse both streams record: `mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)`. Per trial, frame times become `frameTimes − vidshift − goCue[trial]`, and the x/y traces are interpolated onto the same 550 bin centres used by the spikes. The offset is computed once per session. If a trial's `frameTimes` is empty or entirely non-finite, a synthetic axis `(1..n)/400` is substituted.

ii.
```python
def get_video_shift(h5, bp) -> float:
    fs = float(np.asarray(h5["obj"]["sglx"]["fs"]).reshape(-1)[0])
    bitstart = np.asarray(h5["obj"]["sglx"]["bitcode"]["bitstart"]).reshape(-1)
    return matlab_mode(bitstart) / fs - matlab_mode(bp["bitStart"])
```

```python
interp = interp1d(frame_times - vidshift - align_time, coords, axis=0, ...)
return np.asarray(interp(TIME), dtype=np.float64)
```

```python
if frame_times.size == 0 or not np.any(np.isfinite(frame_times)):
    frame_times = np.arange(1, ts.shape[0] + 1, dtype=np.float64) / 400.0
```

iii. A direct port of `findVideoOffset.m` (read at step 43) and the `interp1(traj.frameTimes - vidshift - obj.bp.ev.(alignEv)(trix), ts, taxis)` line of `findPosition.m`; the `(1:n)/400` fallback is also lifted from `findPosition.m`. Step 31: *"[the repository] aligns video with a … camera offset before interpolation."* `matlab_mode` reproduces MATLAB's tie-breaking (smallest value wins).

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The bottom camera's DeepLabCut tracking only (`obj.traj{2}`), using **both** paw features, `top_paw` and `bottom_paw`, whichever are present in `featNames`. Same `ts` / `frameTimes` / `NdroppedFrames` fields as the tongue, and the same session video offset.

ii.
```python
paw_indices = {
    1: [feat_names[1].index(name) for name in ["top_paw", "bottom_paw"]
        if name in feat_names[1]],
}
```

```python
for feat_idx in paw_indices[1]:
    coords = bottom_ts[:, :2, feat_idx]
    coords_interp = interpolate_coords(coords, bottom_ft, align_time, vidshift)
    speed, visible = feature_speed(coords_interp, tongue_feature=False)
    trial_paw_speeds.append(speed)
```

iii. The paw is not in the authors' `params.traj_features` list at all (step 80: `rg -n "paw" /app/code -g'*.m'`), so the AI had to choose; it enumerated the bottom-camera feature names in the data (step 79) and took both paw landmarks, consistent with its "pool the available landmarks rather than guess one" reasoning at step 78.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The **non-tongue** branch of the authors' kinematics code: (1) linear interpolation of x, y onto the bin centres; (2) `fill_nearest` — nearest-neighbour fill of every NaN, including constant extrapolation beyond the ends of the video; (3) `np.gradient` for x and y velocity; (4) subtraction of the trial's baseline derivative, `median(diff(coords))`, from both components; (5) a second `fill_nearest` on the velocity; (6) speed = `hypot(vx, vy)`; (7) the two paw features are averaged unconditionally. Positions are not smoothed (the authors call `mySmooth(ts, 1, ...)`, and `N=1` is a no-op).

Two things follow from step (2)/(5). First, because the window starts at −3.0 s but the camera frames only reach back to about −2.48 s, roughly the first 50 bins of **every** trial contain no video frame at all and are filled by constant extrapolation. Second, the AI faithfully reproduces the authors' apparent index slip in `findVelocity.m`, where the *x* baseline derivative is subtracted from *both* components (`yvel(:,i) = yvel(:,i) - basederiv(1)`).

ii.
```python
    else:
        coords_filled = fill_nearest(coords_interp)
        vel = np.gradient(coords_filled, axis=0)
        base_deriv = np.nanmedian(np.diff(coords_filled, axis=0), axis=0)
        vel[:, 0] = vel[:, 0] - base_deriv[0]
        vel[:, 1] = vel[:, 1] - base_deriv[0]     # matches basederiv(1) in findVelocity.m
        vel = fill_nearest(vel)
```

```python
def fill_nearest(x):
    ...
    interp = interp1d(idx[valid], x[valid, col], kind="nearest",
                      bounds_error=False,
                      fill_value=(x[valid, col][0], x[valid, col][-1]))
    out[:, col] = interp(idx)
```

```python
if trial_paw_speeds:
    agg_paw = np.mean(np.stack(trial_paw_speeds, axis=0), axis=0)
```

iii. A deliberate port of `findVelocity.m` and `findPosition.m`'s non-tongue branch, both read at steps 41–42 — including `fillmissing(...,'nearest')` and the `basederiv(1)` subtraction. The trajectory shows no separate reasoning about the paw specifically beyond the feature-selection discussion at step 78.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. A session-wide 50th percentile of the pooled paw speed, with bins at or above it coded 1 and below 0. The third class, `not visible`, is declared in `output_values` but is assigned only when the **entire trial's bottom-camera video is missing** (`NdroppedFrames` is NaN): as soon as the video exists, the per-bin visibility mask is overwritten with all-True, discarding the `visible` mask that `feature_speed` computed. No trial in this cohort has missing video, so the realised distribution is **52.3% / 47.7% / 0%** — class 2 never occurs, against 18.6% in the reference. In particular, the ~50 bins per trial before any camera frame exists are labelled 0 ("below threshold") on every trial rather than "not visible"; I confirmed this in `converted_data.pkl` (the fraction of class-1 paw bins is exactly 0 for bins 0–52 and ~0.5 thereafter). The threshold itself is therefore also computed over those extrapolated bins.

ii.
```python
if trial_paw_speeds:
    agg_paw = np.mean(np.stack(trial_paw_speeds, axis=0), axis=0)
    agg_paw_visible = np.ones(TIME.size, dtype=bool)     # per-bin `visible` mask discarded
else:
    agg_paw = np.full(TIME.size, np.nan, dtype=np.float64)
    agg_paw_visible = np.zeros(TIME.size, dtype=bool)
```

```python
paw_thresh = float(np.nanpercentile(paw_series[paw_visible], 50))
...
trial_output[4] = np.where(~paw_visible[out_idx], 2,
                           (paw_series[out_idx] >= paw_thresh).astype(np.int64))
```

iii. The AI's stated intent (step 89) was *"keep missing-video trials with the requested 'not visible / no video' labels"* — i.e. visibility was meant to be a whole-trial property for the paw, since `fillmissing(...,'nearest')` in the authors' code leaves no NaNs to mark. The consequence that the pre-video portion of the window is scored as genuine below-threshold movement is not discussed anywhere in the trajectory, and the `--verify-only` check the AI ran does not flag unused categories.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue: the session's `vidshift` and the trial's `goCue` are subtracted from the bottom camera's `frameTimes`, and the traces are linearly interpolated onto the same 550 bin centres as the spikes. The paw always uses the bottom camera's own frame times, which is correct even if the two views recorded different numbers of frames.

ii.
```python
side_ts, side_ft = get_trial_video(h5, view_groups[0], trial_idx)
bottom_ts, bottom_ft = get_trial_video(h5, view_groups[1], trial_idx)
...
coords_interp = interpolate_coords(coords, bottom_ft, align_time, vidshift)
```

iii. Same `findVideoOffset.m` / `findPosition.m` port as 7-d; one offset and one grid serve every stream.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file beside the data structure, read with `scipy.io.loadmat`. `me.data` is a cell array with one trace per trial, one value per camera frame; `me.moveThresh` is also read but never used. The side camera's `frameTimes` provide the time base.

ii.
```python
def read_motion_energy(path: Path):
    loaded = loadmat(path, squeeze_me=True, struct_as_record=False)
    me = loaded["me"]
    data = me.data
    if not isinstance(data, np.ndarray) and hasattr(data, "data"):
        data = data.data
    return data, float(me.moveThresh)
```

```python
me_trace = np.asarray(motion_energy_data[trial_idx]).reshape(-1)
```

iii. Follows `loadMotionEnergy.m` (step 59), whose `if isstruct(me.data), me.data = me.data.data; end` guard is reproduced by the `hasattr(data, "data")` test. Steps 100–103 record a bug fix here: the first version unwrapped unconditionally and accidentally hit NumPy's own `.data` buffer — *"I found the issue: I was accidentally unwrapping NumPy's own `.data` buffer instead of only unwrapping MATLAB struct wrappers."*

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Almost none — the value is already one scalar per frame. The trace is linearly interpolated from the (offset-corrected) side-camera frame times onto the 550 bin centres, and then `fill_nearest` is applied, so any bin outside the frame range takes the nearest in-range value rather than NaN. No smoothing, no normalisation. If the frame-time vector and the trace differ in length, the frame times are replaced by `(1..n)/400`.

ii.
```python
def interpolate_motion_energy(me_trace, frame_times, align_time, vidshift):
    if frame_times.size != me_trace.size:
        frame_times = np.arange(1, me_trace.size + 1, dtype=np.float64) / 400.0
    interp = interp1d(frame_times - vidshift - align_time, me_trace,
                      kind="linear", bounds_error=False, fill_value=np.nan,
                      assume_sorted=True)
    return fill_nearest(np.asarray(interp(TIME), dtype=np.float64))
```

iii. The paper computes motion energy per pixel and reduces each frame to a scalar upstream, so there is nothing left to derive; the interpolation-onto-`taxis` step mirrors the kinematics path. The `fill_nearest` is the AI's own addition (the authors apply `fillmissing` to positions and velocities, not to motion energy) and is not justified anywhere in the trajectory.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. A session-wide 50th percentile over the pooled traces; bins at or above it are 1, below 0. The third class, `no video`, is assigned only when the whole trial's side-camera video is missing (`NdroppedFrames` NaN) — otherwise `me_visible = np.isfinite(motion)`, which is always True after `fill_nearest`. No trial in this cohort has missing video, so the realised distribution is **49.8% / 50.2% / 0%**, against 47.9% / 48.3% / 3.8% in the reference. As with the paw, the ~50 pre-video bins of every trial are coded 0 rather than 2.

ii.
```python
me_thresh = float(np.nanpercentile(me_series[me_visible], 50))
...
trial_output[5] = np.where(~me_visible[out_idx], 2,
                           (me_series[out_idx] >= me_thresh).astype(np.int64))
```

```python
if side_ts is None:
    motion = np.full(TIME.size, np.nan, dtype=np.float64)
    motion_visible = np.zeros(TIME.size, dtype=bool)
else:
    motion = interpolate_motion_energy(me_trace, side_ft, align_time, vidshift)
    motion_visible = np.isfinite(motion)
```

```python
OUTPUT_VALUES = [..., ["<50th percentile", ">=50th percentile", "no video"]]
```

iii. Threshold rule and class names come straight from the prompt. The whole-trial-only treatment of `no video` follows the AI's plan at step 89 to *"keep missing-video trials with the requested 'not visible / no video' labels"*; the interaction with the `fill_nearest` extrapolation and the −3.0 s window start is not discussed.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same offset and same grid as the tracking, using the **side** camera's frame times, since motion energy has one value per side-camera frame. `frameTimes − vidshift − goCue[trial]`, then linear interpolation onto the 550 bin centres.

ii.
```python
motion = interpolate_motion_energy(me_trace, side_ft, align_time, vidshift)
```

```python
interp = interp1d(frame_times - vidshift - align_time, me_trace, ...)
```

iii. One video offset serves every camera-derived stream; the side camera is the one motion energy is computed from, matching `loadMotionEnergy.m`. The length-mismatch fallback mirrors `findPosition.m`'s missing-`frameTimes` handling.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Four cases.
- **Whole-trial missing video** (`NdroppedFrames` NaN): `get_trial_video` returns `(None, None)`, the trial's tongue/paw/motion-energy channels become all-2 (`not visible` / `no video`), and the trial is otherwise kept. This never triggers in the 12 sessions (the AI verified `haveVid missing 0` for all of them).
- **Missing or all-NaN `frameTimes`**: replaced by a synthetic `(1..n)/400` axis, which is then still shifted by `vidshift` and the go cue. This keeps the trial rather than dropping it, but the substituted axis is not clock-calibrated.
- **Untracked frames** (DeepLabCut likelihood below the authors' cut, stored as NaN x/y): for the **tongue** these produce `visible = False` and the `not visible` class; for the **paw** and **motion energy** they are `fill_nearest`-filled and the bin is still reported as visible.
- **No valid landmark at all in a trial**: `feature_speed` returns an all-NaN speed and an all-False visibility mask; `aggregate_speeds` returns NaN where no feature is visible. A later cleanup pass (step 130) suppressed the resulting all-NaN RuntimeWarning.

Session-level guards raise rather than skip: fewer than 10 units or fewer than 2 trials aborts the whole conversion.

ii.
```python
def get_trial_video(h5, view_group, trial_idx):
    ndropped = read_ref_array(h5, view_group["NdroppedFrames"][trial_idx, 0])
    if np.asarray(ndropped).size == 0 or not np.isfinite(np.asarray(ndropped).reshape(-1)[0]):
        return None, None
```

```python
def feature_speed(coords_interp, tongue_feature: bool):
    visible = np.isfinite(coords_interp).all(axis=1)
    if not np.any(visible):
        return np.full(TIME.size, np.nan, dtype=np.float64), visible
```

```python
def aggregate_speeds(speeds, visibles):
    if not speeds:
        return np.full(TIME.size, np.nan), np.zeros(TIME.size, dtype=bool)
```

iii. Step 89: *"I'm checking the per-trial `haveEphys` and `haveVid` flags so I can decide whether to drop missing-neural trials and keep missing-video trials with the requested 'not visible / no video' labels."* Step 111: *"I've only seen a benign all-NaN visibility warning so far."* Step 129: *"I'm doing one small cleanup pass on the converter to suppress the all-NaN velocity warning we saw during generation."* The nearest-fill behaviour for non-tongue features is inherited from `findPosition.m` / `findVelocity.m` rather than chosen independently.

## 11-a. What are the most time-consuming steps of the code?

i. Two things dominate. (1) **Per-trial video processing** — for each of 3,116 trials the code dereferences and transposes both cameras' `ts` arrays, runs up to 10 separate `interp1d` calls (7 tongue + 2 paw + 1 motion energy), several `fill_nearest` calls that each build a fresh `interp1d` per coordinate column, and then a 550-iteration pure-Python loop inside `aggregate_speeds`. The AI itself observed this: *"The pass is still executing, which is expected with per-trial video interpolation across 12 sessions."* (2) **Neural processing** — per cluster, one `np.add.at` histogram plus eight `my_smooth` calls (seven condition PSTHs plus the trial matrix), each of which convolves column by column in a Python loop.

ii.
```python
    out = np.full(TIME.size, np.nan, dtype=np.float64)
    for t in range(TIME.size):
        mask = vis_stack[:, t]
        if np.any(mask):
            out[t] = float(np.mean(speed_stack[mask, t]))
```

```python
    out = np.stack(
        [np.convolve(x_filt[:, j], kernel, mode="same") for j in range(x_filt.shape[1])],
        axis=1,
    )
```

iii. No explicit reasoning about runtime in the trajectory beyond the observations above; the AI polled the running process repeatedly (steps 106–112, 131–137) rather than optimising, and noted *"if the run stalls materially I'll tighten that path"* — it did not stall, so nothing was tightened.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Three clear candidates.
- `aggregate_speeds`'s `for t in range(TIME.size)` loop is a masked mean over an axis; `np.nansum`/`np.isfinite(...).sum(axis=0)` over the stacked arrays would do the same in one expression (this is exactly the reference's `_mean_over_available`).
- `my_smooth`'s per-column `np.convolve` loop could be a single `scipy.ndimage.convolve1d` or FFT convolution along the time axis, which would also remove the need to transpose the trial matrix.
- `fill_nearest` builds an `interp1d` per column inside a Python loop; a cumulative-maximum index trick fills all columns at once.

In addition, the per-trial loop in `compute_behavioral_outputs` re-does the same feature bookkeeping 3,116 times; the reference avoids the analogous cost only for spikes, by using a single `histogram2d` over all trials.

ii.
```python
    for col in range(x.shape[1]):
        valid = np.isfinite(x[:, col])
        ...
        interp = interp1d(idx[valid], x[valid, col], kind="nearest", ...)
        out[:, col] = interp(idx)
```

```python
        for cond_mask in condition_masks:
            ...
            psth = my_smooth(psth, SMOOTH_WINDOW, BCTYPE)
```

iii. Not discussed. The spike histogram *is* vectorised across all trials and spikes at once (`np.searchsorted` + `np.add.at`), which suggests the AI vectorised where it was easy and left the video path loop-based because trials have differing frame counts.

## 11-c. What processing does the code repeat multiple times?

i. Four repetitions.
- **Each session's HDF5 file is opened twice** — once in `process_session` for `bp` and clusters, once again in `compute_behavioral_outputs` for the video — so `bp`/`sglx` metadata are re-read.
- **Seven condition PSTHs are computed and smoothed for every cluster** solely to apply the >1 Hz filter, then thrown away; the trial matrix is smoothed an eighth time.
- **`fill_nearest` is applied twice per paw feature**, once to the positions and once to the resulting velocities.
- **`interpolate_coords` is called separately for every tongue/paw feature** even though all features of one view share the same frame-time vector, so the same `frame_times - vidshift - align_time` subtraction and the same knot placement are recomputed up to seven times per trial per view.

By contrast the video offset is correctly computed once per session, and the bin grid once at module level.

ii.
```python
def process_session(spec):
    with h5py.File(spec.data_path, "r") as h5:
        ...
    outputs, thresholds = compute_behavioral_outputs(spec, bp, valid_trials)   # opens the file again
```

```python
def compute_behavioral_outputs(spec, bp, valid_trials):
    ...
    with h5py.File(spec.data_path, "r") as h5:
        vidshift = get_video_shift(h5, bp)
```

```python
        vel = np.gradient(coords_filled, axis=0)
        ...
        vel = fill_nearest(vel)
```

iii. Not discussed in the trajectory. The double file open is a structural consequence of splitting the neural and behavioural paths into separate functions; the repeated PSTH computation is required to reproduce `removeLowFRClusters.m` faithfully, so it is deliberate rather than accidental.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small items.
- The **seven condition PSTHs** exist only to produce one scalar per cluster (`mean_fr`); smoothing them before averaging cannot change the ranking much and is pure overhead.
- **`me.moveThresh`** is parsed out of every motion-energy file and returned, then dropped at the call site (`motion_energy_data, _ = read_motion_energy(...)`).
- **`bp.ev.sample` and `bp.ev.delay`** are loaded for every session and never used.
- The **per-bin `visible` mask returned by `feature_speed` for the paw** is computed and then discarded in favour of `np.ones(...)` — this is not merely wasted work, it is what removes the `not visible` class from the paw output (8-c).
- The **likelihood column of `ts`** is loaded (the full `(frames, 3, feats)` array is read and transposed) but only `[:, :2, :]` is used; visibility is inferred from NaNs instead.
- The per-session **`thresholds` dict** is stored in metadata but is not needed downstream.

ii.
```python
motion_energy_data, _ = read_motion_energy(spec.motion_energy_path)
```

```python
        "sample": np.asarray(ev["sample"]).reshape(-1),
        "delay": np.asarray(ev["delay"]).reshape(-1),
```

```python
speed, visible = feature_speed(coords_interp, tongue_feature=False)
trial_paw_speeds.append(speed)
trial_paw_visible.append(visible)          # collected, then never used for the paw
...
agg_paw_visible = np.ones(TIME.size, dtype=bool)
```

iii. Not discussed. Most of these are the residue of porting MATLAB helpers wholesale (`removeLowFRClusters.m`, `loadMotionEnergy.m`) and of keeping a single `feature_speed` signature for both the tongue and the paw.
