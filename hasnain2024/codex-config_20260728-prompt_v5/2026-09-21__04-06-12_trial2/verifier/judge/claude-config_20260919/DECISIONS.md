# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not glob the data folders. It reconstructs the authors' own cohort definition by *parsing the MATLAB source*: it reads `code/Scripts/Figure 3/Figure3h.m` to find which `load<ANM>_ALMVideo(...)` loaders are called for the fixed-delay (`fixmeta`) and randomized-delay (`randmeta`) cohorts, skipping commented-out (`%`) lines, and then parses each `load<ANM>_ALMVideo.m` to extract every `date = '...'` / `probe = ...` pair (again skipping commented lines). This yields 25 fixed-delay sessions (folder `Ephys_Behavior`) + 19 randomized-delay sessions (folder `RandomizedDelay_Ephys_Behavior`) = **44 sessions**, each with its loader-specified probe list. Each session's `data_structure_<anm>_<date>.mat` is opened once with `mat73.loadmat` (MATLAB v7.3/HDF5), falling back to `scipy.io.loadmat(..., simplify_cells=True)` for older v5 files. The companion `motionEnergy_<anm>_<date>.mat` is loaded separately by the same helper.

ii.
```python
def get_session_specs(sample: bool) -> list[SessionSpec]:
    fixed_animals = parse_loader_animals(FIXED_SCRIPT, marker="fixmeta")
    randomized_animals = parse_loader_animals(RANDOMIZED_SCRIPT, marker="randmeta")
    specs: list[SessionSpec] = []
    for animal in fixed_animals:
        specs.extend(parse_loader_sessions(animal, cohort="fixed", folder="Ephys_Behavior"))
    for animal in randomized_animals:
        specs.extend(parse_loader_sessions(animal, cohort="randomized",
                                           folder="RandomizedDelay_Ephys_Behavior"))
```
```python
def parse_loader_sessions(animal: str, cohort: str, folder: str) -> list[SessionSpec]:
    text = (LOADER_DIR / f"load{animal}_ALMVideo.m").read_text()
    ...
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("%"):
            continue
        m_date = re.search(r"date = '([^']+)'", line)
        ...
        m_probe = re.search(r"probe = (\[[^\]]+\]|\d+);", line)
```
```python
def load_mat_file(path: Path) -> dict[str, Any]:
    try:
        return mat73.loadmat(str(path))
    except Exception:
        return scipy.io.loadmat(str(path), simplify_cells=True)
```

iii. From CONVERSION_NOTES Step 4/5: "The released paper code defines the paper-comparable cohorts explicitly through the `load*_ALMVideo.m` scripts. The conversion will therefore use the union of the fixed-delay and randomized-delay ephys loader sessions (44 sessions total ...), rather than every raw `.mat` file with neural data." The AI explicitly noted that the raw folders contain extra files the authors excluded (`JEB23_2023-10-20` commented out; `JEB24_2023-10-03/04` have no `obj.clu`), so filesystem discovery would over-count. It also documented that both MATLAB file formats occur in the release, hence the two readers.

## 1-b. How are the data split into subjects?

i. The subject is the animal id carried in the `SessionSpec` (taken from the `load<ANM>_ALMVideo.m` filename / loader name, equivalently the prefix of the session id). Subjects are registered in first-appearance order in a dict and `subject_idx` is that index per session; `subjects` is the key list ordered by index. Result: 14 subjects over 44 sessions (10 fixed-delay animals + 4 randomized-delay animals).

ii.
```python
@dataclass(frozen=True)
class SessionSpec:
    cohort: str; folder: str; animal: str; date: str; probes: tuple[int, ...]
    @property
    def session_id(self) -> str:
        return f"{self.animal}_{self.date}"
```
```python
subject_idx = subject_to_idx.setdefault(spec.animal, len(subject_to_idx))
...
converted["subjects"] = [subject for subject, _ in sorted(subject_to_idx.items(), key=lambda kv: kv[1])]
converted["subject_idx"] = np.asarray(converted["subject_idx"], dtype=np.int64)
```

iii. The AI documented that the animal ids come from the loader scripts and raw filenames ("Raw file names match the loader animal IDs"), and flagged that the paper's own mouse counts (9 fixed + 4 randomized) disagree with the loader lists (10 + 4); it decided to "follow the explicit loader session lists from the released code" and treat the paper text as a manuscript counting inconsistency.

## 1-c. How are the data split into sessions?

i. One session = one `SessionSpec` = one `(animal, date, probe(s), folder)` entry parsed from a loader script = one `data_structure_*.mat` file. Each session becomes one element of `neural`, `input`, `output`, `brain_region_idx`, `subject_idx` and one `session_info` record. Fixed-delay and randomized-delay sessions are pooled into a single 44-session list (the cohort tag is retained only in `metadata.session_info`). A session is dropped if it has no usable `clu`, fewer than 10 surviving units, or fewer than 2 surviving trials; in practice no session was dropped.

ii.
```python
for sess_num, spec in enumerate(specs, start=1):
    obj = load_mat_file(spec.data_path)["obj"]
    processed = process_session(spec, obj, show_processing=...)
    if processed is None:
        print(f"  skipped {spec.session_id}")
        continue
    converted["neural"].append(processed["neural_trials"])
    ...
    converted["metadata"]["session_info"].append(processed["session_info"])
```
```python
    @property
    def data_path(self) -> Path:
        return DATA / self.folder / f"data_structure_{self.animal}_{self.date}.mat"
```

iii. Same justification as 1-a: the session list is the authors' own. The AI kept both task variants in one dataset because both are ALM recordings aligned to the same go cue, and it verified the counts against the paper ("25 fixed sessions + 19 randomized sessions = 44 saved sessions", Step 10 Check 4).

## 1-d. How are the data split into trials?

i. Trials are the rows of the Bpod table: `n_trials = int(bp.Ntrials)`, and every per-trial quantity is read as a flat length-`Ntrials` vector (`hit`, `miss`, `no`, `R`, `L`, `autowater`, `early`, `stim.enable`, `ev.goCue`, `ev.lickL`, `ev.lickR`). Spikes carry their trial number (`clu.trial`, 1-based, converted to 0-based and range-checked), and the camera trajectory struct array has one entry per trial, so trial boundaries are never reconstructed. Each surviving trial becomes one `(n_neurons, 500)` neural array, one `(1, 500)` input array and one `(6, 500)` output array.

ii.
```python
n_trials = int(gget(bp, "Ntrials", 0) or 0)
if n_trials == 0:
    return None
```
```python
def bool_array(bp: Any, field: str, n_trials: int) -> np.ndarray:
    values = gget(bp, field, None)
    if values is None:
        return np.zeros(n_trials, dtype=bool)
    values = np.asarray(values).reshape(-1)
    return values.astype(bool)
```
```python
trial_ids = np.asarray(unit["trial"], dtype=np.int64).reshape(-1) - 1
valid = (trial_ids >= 0) & (trial_ids < n_trials) & np.isfinite(trial_times)
```

iii. The AI relied on the tutorial file `WorkingWithDataObjs.m` ("obj.bp contains trial/task information such as the number of trials...") and the reference loaders, which index everything by trial number. It documented spot checks mapping saved trial indices back to raw trial numbers (e.g. "saved trial indices 0, 50, and 200 (raw trials 0, 55, and 261)"). Note it does *not* truncate per-trial fields to `Ntrials` the way the human reference does; this happens to be safe on this release because all the fields it reads have exactly `Ntrials` entries (a length mismatch would raise a broadcast error, and none occurred).

## 1-e. How are trials filtered based on quality controls?

i. Four conjunctive criteria, all applied per trial before the arrays are written: (1) the trial must be one of the three behavioural outcomes, `hit | miss | no`; (2) not an early-lick trial (`~early`); (3) not a photostimulation trial (`~stim.enable`); (4) `neural_covered` — at least one quality-passing unit must have fired at least one spike assigned to that trial. Criterion (4) was added after the first full run, when the verifier warned about all-zero neural trials in `JEB24_2023-10-23` and `JEB24_2023-11-03`, where behaviour keeps being logged after the probe recording stops. Sessions with fewer than 2 surviving trials are dropped. Result: 13,762 of 15,324 raw trials kept.

ii.
```python
keep_trials = np.flatnonzero((hit | miss | no) & ~early & ~stim & neural_covered)
if keep_trials.size < MIN_TRIALS_PER_SESSION:
    return None
```
```python
    valid = (trial_ids >= 0) & (trial_ids < n_trials) & np.isfinite(trial_times)
    trial_ids = trial_ids[valid]
    ...
    neural_covered[trial_ids] = True
```

iii. CONVERSION_NOTES Step 5 Key Decision 2: "Reference trial definitions routinely exclude perturbation/stimulation and early-lick trials. Ignore/no-response trials are retained because the decoder task requires the `ignore` outcome class." The `(hit|miss|no)` term is copied from `Figure3h.m`'s first PSTH condition `'(hit|miss|no)'`, and `~stim.enable&~early` from its other conditions. For `neural_covered` the AI documented: "some raw sessions continue to log behavior after the neural recording stops ... add `neural_covered` to the saved-trial mask", and verified the two affected sessions directly against `obj.clu[*].trial` ("`JEB24_2023-10-23`: raw neural coverage stops at trial 314/343").

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu`, the spike-sorted clusters of the loader-specified probe(s). From each cluster it reads `quality` (curation label), `trial` (1-based trial of each spike) and `trialtm` (spike time relative to that trial's start); `site`/`tm` are extracted but unused. The alignment times come from `obj.bp.ev.goCue`. Two on-disk layouts are normalised into one flat list of unit dicts: v7.3 files store one struct per probe whose fields are per-cluster cell arrays; older v5 files store a flat list of per-cluster structs for a single probe.

ii.
```python
def extract_selected_units(obj, probes):
    clu = gget(obj, "clu")
    if is_flat_unit_list(clu):                       # old v5 layout: one entry per cluster
        return [unit for unit in clu if normalize_string(gget(unit, "quality")).lower() not in BAD_QUALITIES]
    if isinstance(clu, dict):
        units = split_probe_struct(clu)
        return [unit for unit in units if normalize_string(unit["quality"]).lower() not in BAD_QUALITIES]
    clu_list = as_list(clu)
    units = []
    for probe_num in probes:                          # loader-specified probes, concatenated
        probe = clu_list[probe_num - 1]
        for unit in split_probe_struct(probe):
            if normalize_string(unit["quality"]).lower() in BAD_QUALITIES:
                continue
            units.append(unit)
    return units
```
```python
go_times = event_array(ev, ALIGN_EVENT)     # ALIGN_EVENT = "goCue"
trial_ids = np.asarray(unit["trial"], dtype=np.int64).reshape(-1) - 1
trial_times = np.asarray(unit["trialtm"], dtype=np.float64).reshape(-1)
```

iii. Mirrors `loadSessionData`/`processData`/`alignSpikes`/`getSeq`, which operate on `obj.clu{prbnum}(clu).trial` and `.trialtm`. The AI documented the dual `obj.clu` layout as a "real data-loading edge case [that] must be handled explicitly in Python" and listed the affected sessions (`JEB23_2023-10-18`, `JEB23_2023-10-21`, all `JEB24`).

## 2-b. How is the `neural` data processed?

i. For each unit and each trial with spikes: spike times are aligned to the go cue, histogrammed into 10 ms bins spanning [−2.5, +2.5] s, divided by the bin width to get Hz, and convolved with a **causal** Gaussian kernel that reproduces the authors' `mySmooth(x, 15, 'reflect')` — `gausswin(15)` (std = (15−1)/(2·2.5) = 2.8 bins = 28 ms) with the first 7 taps zeroed, normalised to unit sum, applied after reflect-padding the first 15 samples and trimming them off. Units from all loader-specified probes are concatenated. No z-scoring, baseline subtraction or normalisation; values stay as firing rates in Hz, stored `float32`. Trials in which a unit never fires stay exactly zero (as in `getSeq`, which `continue`s).

ii.
```python
def make_causal_kernel(window_bins: int) -> np.ndarray:
    std = (window_bins - 1) / (2.0 * 2.5)
    kernel = gaussian(window_bins, std=std).astype(np.float64)
    kernel[: window_bins // 2] = 0.0          # causal
    kernel /= kernel.sum()
    return kernel
```
```python
def my_smooth_1d(values, boundary_condition="reflect"):
    if boundary_condition == "reflect":
        padded = np.concatenate([values[:SMOOTH_BINS], values]); trim = SMOOTH_BINS
    ...
    smoothed = np.convolve(padded, GAUSSIAN_KERNEL, mode="same")
    return smoothed[trim:].astype(np.float32)
```
```python
        aligned_times = trial_times - go_times[trial_ids]
        for tr in np.unique(trial_ids):
            tr_mask = trial_ids == tr
            counts, _ = np.histogram(aligned_times[tr_mask], bins=edges)
            rates = counts.astype(np.float64) / DT
            trialdat[unit_idx, tr] = my_smooth_1d(rates, boundary_condition="reflect")
```

iii. CONVERSION_NOTES Step 5 Key Decision 5: "Keep neural activity as smoothed firing rates, not z-scored residuals. The reference loading path and decoder examples operate directly on `obj.trialdat` after spike alignment/binning/smoothing. Additional z-scoring or subspace projection would make the saved dataset less general and less faithful to the primary loader output." The smoothing choice is justified as a direct port of `utils/mySmooth.m` (which zeroes the first half of `gausswin(N)` "%causal" and reflect-pads `N` samples), invoked as `mySmooth(N./params.dt, params.smooth, params.bctype)` in `getSeq.m` with `params.smooth = 15`, `params.bctype = 'reflect'`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two unit filters plus one session filter. (1) Manual curation label: a cluster is dropped if `strip().lower()` of `clu.quality` is in `{garbage, gabrga, noisy, real?}` — exactly the exclusion list hard-coded in `findClusters.m` for `quality = {'all'}`; multi-units and unlabelled clusters are kept. (2) Low firing rate: condition-averaged PSTHs are built for the eight `Figure3h.m` conditions, a unit's score is the mean of its PSTH over time *and* conditions, and units with score ≤ 1 Hz are dropped. (3) A session with fewer than 10 surviving units would be dropped (no session actually was; minimum is 15). Result: 2,443 units over 44 sessions, 15–141 per session.

ii.
```python
BAD_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}
LOW_FR = 1.0
MIN_UNITS_PER_SESSION = 10
```
```python
def low_fr_condition_indices(bp, n_trials):
    masks = [
        hit | miss | no,
        right & hit  & ~stim & ~autowater & ~early,
        left  & hit  & ~stim & ~autowater & ~early,
        right & miss & ~stim & ~autowater & ~early,
        left  & miss & ~stim & ~autowater & ~early,
        right & no   & ~stim & ~autowater & ~early,
        left  & no   & ~stim & ~autowater & ~early,
        hit          & ~stim & ~autowater & ~early,
    ]
    return [np.flatnonzero(mask) for mask in masks]
```
```python
    psth = np.zeros((trialdat.shape[0], taxis.size, len(condition_indices)), dtype=np.float32)
    for cond_idx, trial_idx in enumerate(condition_indices):
        psth[:, :, cond_idx] = np.nanmean(trialdat[:, trial_idx, :], axis=1)
    mean_fr = np.nanmean(psth, axis=(1, 2))
    keep_units = mean_fr > LOW_FR
    trialdat = trialdat[keep_units]
    if trialdat.shape[0] < MIN_UNITS_PER_SESSION:
        return None
```

iii. Step 10 Check 3: "`extract_selected_units()` mirrors `findClusters('all')` by excluding only `garbage`, `gabrga`, `noisy`, and `real?`. `process_session()` then mirrors the published `>1 Hz` criterion using condition-averaged smoothed firing rates before saving units." The `> 1 Hz` value comes from `Figure3h.m` (`randparams.lowFR = 1`, `fixparams.lowFR = 1`) and the paper, in preference to `getDefaultParams.m`'s `0.5`; the mean-over-conditions-then-time rule is a direct port of `removeLowFRClusters.m` (`meanFRs = mean(mean(obj.psth{prbnum},3,'omitnan'),'omitnan')`). The 10-unit session rule is justified in Step 5 as matching "the paper's session-inclusion rule" — the released inclusion rule (`UseInclusionCritera.m`) is actually ">40 R-hit and >40 L-hit trials", so that particular justification is not supported, though the filter is inert here.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to go-cue onset is a single subtraction per spike: `clu.trialtm` is already on the behaviour clock relative to its own trial's start, and `bp.ev.goCue` is on the same clock, so `trialtm − goCue[trial]` gives seconds from the go cue with no interpolation or per-session offset. The aligned times are then histogrammed into the fixed [−2.5, 2.5] s grid, so spikes outside the window simply fall outside the bin edges. `metadata.temporal_alignment_event = "Go cue onset"` and `alignment_event_field = "bp.ev.goCue"`.

ii.
```python
ALIGN_EVENT = "goCue"
...
go_times = event_array(ev, ALIGN_EVENT)
...
        aligned_times = trial_times - go_times[trial_ids]
        for tr in np.unique(trial_ids):
            counts, _ = np.histogram(aligned_times[trial_ids == tr], bins=edges)
```

iii. This is `alignSpikes.m` (`trialtm_aligned = trialtm - event`) with `params.alignEvent = 'goCue'`. The AI additionally checked (Step 4) that go-cue alignment is valid for water-cued trials too: "In raw context sessions, WC trials still have valid `bp.ev.goCue` values ... Align all trials to `goCue` exactly as requested by the decoder task and as done in the released context-analysis code. Do not substitute reward time for WC trials."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **10 ms bins** (`DT = 0.01`), 500 bins spanning [−2.5, +2.5] s from the go cue, identical for every trial and every session. The input axis is the bin centres (−2.495 … 2.495 s). There is no rebinning: spikes are histogrammed once directly onto this grid, and all camera streams are interpolated once directly onto the same bin centres, so neural, input and all six outputs share one time axis. `metadata.time_bin_size = 10.0` ms, `off_start = −2.5`, `off_end = 2.5`.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 0.01

def session_time_axis() -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(TMIN, TMAX + DT * 0.5, DT)
    time_axis = edges[:-1] + DT / 2.0
    return edges, time_axis
```

iii. Step 5 Key Decision 3: "Align everything to `goCue`, use `dt=0.01 s`, and use a common window `[-2.5, 2.5] s`. This matches the released figure scripts and tutorial path closely while avoiding unnecessary extrapolation at very early times." Step 10: "the converter uses `dt = 0.01 s` ... This matches the released figure-script configuration more closely than the generic 5 ms / 200 Hz defaults described elsewhere in the repository and paper." Indeed `Figure3h.m` (both `fixparams` and `randparams`), `Figure3i.m`, `Figure8*` and `WorkingWithDataObjs.m` all set `dt = 1/100`, whereas `getDefaultParams.m` sets `1/200`; the AI also recorded the paper's statement that "single-trial neural activity was first binned in 5-ms intervals" in its Step 3 table and chose the figure-script value anyway. The randomized-delay script's `tmin = -2.7` was normalised to −2.5 so that all sessions share one grid.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Nothing is read from the raw files: the input is the analysis time axis itself, defined by the constants `TMIN`, `TMAX`, `DT` around the alignment event `bp.ev.goCue`. It is the vector of bin centres of the neural binning grid, identical for every trial and every session, stored as a `(1, 500)` `float32` array per trial and named `time_from_go_cue_s`.

ii.
```python
edges, taxis = session_time_axis()
...
        input_trials.append(taxis[np.newaxis, :].astype(np.float32, copy=False))
```
```python
"input_names": ["time_from_go_cue_s"],
```

iii. Step 5 mapping table: "`obj.time` after neural binning → `input[0]`: Use the neural time vector relative to go cue, repeated identically for every trial in the session as a `(1, T)` float32 array. Reference: `getSeq`. Decoder task specifies time from go cue onset as the only decoder input." This mirrors `getSeq.m`'s `obj.time = edges + params.dt/2; obj.time = obj.time(1:end-1)`.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond constructing the bin centres: `edges = arange(-2.5, 2.5+dt/2, dt)` then `centres = edges[:-1] + dt/2`. The axis is computed once per session (the helper is called once in `process_session`) and the same array object is referenced by every trial. It is not represented as a binary event series because it is a continuous, monotonically increasing clock, which is what the Decoder Task specifies.

ii.
```python
def session_time_axis() -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(TMIN, TMAX + DT * 0.5, DT)
    time_axis = edges[:-1] + DT / 2.0
    return edges, time_axis
```

iii. Same as 3-a; the AI's Step 10 sanity check independently rebuilt the axis from scratch and confirmed `np.allclose(manual_taxis, saved_input[0]) == True`.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is the neural grid by construction. The same `edges` array is used both to histogram the go-cue-aligned spike times and to derive the input bin centres, so input sample *k* is exactly the centre of the interval in which the spikes of neural bin *k* were counted. The camera streams are interpolated onto the same `taxis`, so every stream in the file shares one time index.

ii.
```python
    edges, taxis = session_time_axis()
    ...
            counts, _ = np.histogram(aligned_times[tr_mask], bins=edges)
    ...
            input_trials.append(taxis[np.newaxis, :].astype(np.float32, copy=False))
```

iii. N/A — alignment is exact by construction; the AI verified there was "no off-by-one discrepancy at the first or last saved time bin" (Step 10 Check 5).

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The raw lick-contact event times `bp.ev.lickL` and `bp.ev.lickR` (cell arrays with one list of times per trial), together with `bp.ev.goCue` for the cut-off. The instructed side (`bp.R`/`bp.L`) and the outcome flags are deliberately **not** used for this variable.

ii.
```python
    lick_l = as_list(gget(ev, "lickL"))
    lick_r = as_list(gget(ev, "lickR"))
    lick_direction = np.array(
        [post_go_choice(lick_l[tr], lick_r[tr], go_times[tr]) for tr in range(n_trials)],
        dtype=np.int64,
    )
```

iii. Step 5 Key Decision 7: "**Use trial-event licks for lick direction instead of `bp.L` / `bp.R`**: `bp.L` and `bp.R` encode the trial side, not the animal's realized first post-go-cue lick. Actual lick direction should come from `bp.ev.lickL` and `bp.ev.lickR`." The mapping table adds: "Uses actual first response side, not instructed side. Miss trials therefore flip relative to `bp.L`/`bp.R` as expected." The lick-raster example in `WorkingWithDataObjs.m` is cited as the precedent for reading these fields.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. For each trial, all left and all right lick times strictly greater than that trial's go-cue time are collected, the earliest of each is taken, and the side with the earlier first lick wins: left = 0, right = 1. If there is no post-go-cue lick on either port the class is `none` = 2. Values are coerced element-wise with a `float()` guard so ragged/empty MATLAB cells do not crash. The per-trial scalar is then broadcast across all 500 bins. Resulting distribution: left 0.407 / right 0.462 / none 0.130.

ii.
```python
def post_go_choice(lick_l: Any, lick_r: Any, go_time: float) -> int:
    left = event_list_after(lick_l, go_time)
    right = event_list_after(lick_r, go_time)
    t_left = left.min() if left.size else np.inf
    t_right = right.min() if right.size else np.inf
    if np.isinf(t_left) and np.isinf(t_right):
        return 2
    return 0 if t_left < t_right else 1
```
```python
        trial_output = np.vstack([
            np.full(taxis.size, lick_direction[tr], dtype=np.int64),
            ...
```

iii. Step 5 Key Decision 4: trial-level variables "will be repeated across time bins so they can coexist with time-varying kinematic outputs in a single consistent output array per trial." The first-lick rule is the AI's reading of the task's response window; it verified it in Step 10 by recomputing lick side from raw `lickL`/`lickR` for three trials of `JEB6_2021-04-18` and matching the saved rows with `np.allclose()`.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The single per-trial Bpod flag `bp.autowater`, which is 1 on trials where water was delivered regardless of the animal's choice.

ii.
```python
    autowater = bool_array(bp, "autowater", n_trials)
```

iii. Step 5 mapping table cites `WorkingWithDataObjs.m`: "obj.bp.autowater=1 when water was delivered regardless of animal choice (0 otherwise). this field can be used as a proxy for obtaining water-cued blocks and delayed-response blocks of trials", and the context conditions in `Figure8a_thru_c.m`. Step 4 adds the decision to "Treat `bp.autowater` as the authoritative trial-wise context label. The 12-session context cohort is a stricter analysis subset, not evidence that other sessions lack context information" — i.e. the AI deliberately labels context in all 44 sessions rather than only the paper's 12-session two-context cohort.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling: `autowater → WC = 0`, otherwise `DR = 1`, broadcast across all 500 bins. Output distribution: WC 0.097 / DR 0.903.

ii.
```python
    context = np.where(autowater, 0, 1).astype(np.int64)
```
```python
OUTPUT_VALUES = [..., ["WC", "DR"], ...]
```

iii. The coding follows the Decoder Task's "(WC, DR)" ordering. The AI's Step 10 spot check recomputed context from raw `autowater` for three trials and matched.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The three mutually exclusive per-trial Bpod flags `bp.hit`, `bp.miss` and `bp.no`. Unlike the human reference, which infers "ignore" as the complement, the AI reads `bp.no` explicitly.

ii.
```python
    hit = bool_array(bp, "hit", n_trials)
    miss = bool_array(bp, "miss", n_trials)
    no = bool_array(bp, "no", n_trials)
```

iii. Step 5 mapping table: "`bp.hit`, `bp.miss`, `bp.no` → `output[2]` (`outcome`) ... standard trial fields in `bp`." These are also the flags used in the `Figure3h.m` PSTH conditions (`'(hit|miss|no)'`, `R&no&...`, etc.).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A relabelling into three classes, written in an order that leaves the flags authoritative: initialise everything to `ignore` = 2, set `miss → incorrect = 0`, then `hit → correct = 1`, then `no → ignore = 2`. Broadcast across all 500 bins. Ignore trials are kept in the dataset rather than dropped. Distribution: incorrect 0.120 / correct 0.749 / ignore 0.131.

ii.
```python
    outcome = np.full(n_trials, 2, dtype=np.int64)
    outcome[miss] = 0
    outcome[hit] = 1
    outcome[no] = 2
```

iii. Step 5 mapping table: "Keep ignore trials because the decoder task explicitly requires them, even though some behavioral analyses omitted them." Class codes follow the Decoder Task's "(incorrect, correct, ignore)" ordering.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut output in `obj.traj{1}` (camera view 0, the **side** camera). Per trial it uses `featNames` to locate the feature, `ts[:, :2, featix]` for the x/y position, and `frameTimes` for the frame clock; `NdroppedFrames` flags unusable trials. The feature is `tongue`, with `left_tongue` then `right_tongue` as fallbacks, chosen once per session from trial 0's `featNames`. The bottom camera's `top_tongue` is **not** used. `bp.ev.goCue` plus `obj.sglx.bitcode.bitstart`/`obj.sglx.fs`/`bp.ev.bitStart` are also needed, for the clock correction (see 7-d).

ii.
```python
    tongue_speed, tongue_visible, tongue_feature = build_speed_trace(
        obj, view_index=0, feature_names=["tongue", "left_tongue", "right_tongue"],
        go_times=go_times, taxis=taxis
    )
```
```python
    for candidate in feature_names:
        first_idx = feature_index(trials_view[0], candidate)
        if first_idx is not None:
            chosen_feature = candidate
            break
```
```python
        feat_idx = feature_index(trial_view, chosen_feature)
        ts = np.asarray(gget(trial_view, "ts", None), dtype=np.float64)
        if feat_idx is None or ts.ndim != 3 or ts.shape[2] <= feat_idx:
            continue
        coords = ts[:, :2, feat_idx]
```

iii. Step 5 mapping table cites `getKinematicsFromVideo`, `findPosition`, `findVelocity`. `tongue` is the first entry of the authors' side-camera feature list in `params.traj_features` in every figure script, so the AI treated it as the canonical tongue marker; the fallbacks cover sessions whose DLC model names the marker differently. The AI noted the resulting `not_visible` fraction was high (86.6 % in the sample, 92.7 % overall) and decided in Step 7 that it "may reflect real tongue occlusion/sparsity rather than a bug", to be confirmed by decoder performance.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Per trial: (1) skip the trial entirely if `NdroppedFrames` is all-NaN; (2) build the go-cue-aligned frame clock (7-d); (3) take x and y for the chosen feature — DeepLabCut coordinates are already NaN wherever the authors' likelihood criterion failed, so `isfinite(x) & isfinite(y)` *is* the visibility mask, and no explicit likelihood threshold is applied; (4) linearly interpolate x and y onto the 500-bin axis and nearest-interpolate the visibility mask, re-imposing NaN on bins that interpolate to "not visible"; (5) because the feature is a tongue, take `np.gradient` of the interpolated position **without** nearest-filling and **without** baseline-derivative subtraction, then replace non-finite velocities with 0 (the reference's "set tongue velocity to 0 if not visible"); (6) speed is `sqrt(x_vel² + y_vel²)`, in pixels per bin. There is no smoothing of the position (the reference's `mySmooth(ts, 1, 'reflect')` is a no-op for tongue features, and is skipped for tongue anyway), and no cross-camera normalisation since only one view is used.

ii.
```python
        x_interp = linear_interp(frame_times, x, taxis)
        y_interp = linear_interp(frame_times, y, taxis)
        vis_interp = nearest_interp(frame_times, frame_visible.astype(float), taxis) >= 0.5
        x_interp[~vis_interp] = np.nan
        y_interp[~vis_interp] = np.nan

        if feat_is_tongue:
            x_vel = np.gradient(x_interp)
            y_vel = np.gradient(y_interp)
            x_vel[~np.isfinite(x_vel)] = 0.0
            y_vel[~np.isfinite(y_vel)] = 0.0
        else:
            ...
        speed[trial_idx] = np.sqrt(x_vel ** 2 + y_vel ** 2).astype(np.float32)
        visible[trial_idx] = vis_interp
```

iii. Step 5 mapping table: "Reuse the reference interpolation/alignment logic from `findPosition`/`findVelocity`, but retain the pre-fill visibility mask ... The reference code sets missing tongue velocity to 0 after interpolation; for the decoder task we preserve a separate 'not visible' class before that fill step." This is a close port of `findVelocity.m`, which computes `gradient` of the interpolated position, skips the baseline-derivative subtraction and the `fillmissing` for any feature whose name contains `tongue`, and sets NaN velocities to 0.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. One threshold per session: the 50th percentile of the speed over all finite, visible bins of all kept trials in that session. Bins with `speed < p50` get 0, visible bins with `speed >= p50` get 1, and every bin whose visibility mask is false (or whose speed is NaN, or whose whole trial was skipped) gets 2 = `not_visible`. If a session has no visible tongue bins at all, the whole stream is 2 and the threshold is NaN.

ii.
```python
def discretize_session_signal(signal, visible, missing_code, kept_trials):
    out = np.full(signal.shape, missing_code, dtype=np.int64)
    values = signal[kept_trials][visible[kept_trials]]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return out, float("nan")
    threshold = float(np.nanpercentile(values, 50))
    visible_and_low = visible & np.isfinite(signal) & (signal < threshold)
    visible_and_high = visible & np.isfinite(signal) & ~visible_and_low
    out[visible_and_low] = 0
    out[visible_and_high] = 1
    return out, threshold
```
```python
    tongue_disc, tongue_thresh = discretize_session_signal(tongue_speed, tongue_visible, 2, keep_trials)
```

iii. Directly implements the Decoder Task specification ("discretized with per-session threshold: 0: < 50th percentile, 1: >= 50th percentile, 2: not visible"). The threshold is computed only over kept trials so that discarded (early/stim/uncovered) trials cannot shift it, and only over *visible* bins so that the missing class does not contaminate the percentile. The per-session thresholds are recorded in `metadata.session_info[i]['thresholds']`.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The camera clock is corrected once per session with the bit-code offset — mode of `sglx.bitcode.bitstart / sglx.fs` minus mode of `bp.ev.bitStart` — and then each trial's frames are expressed as `frameTimes − vidshift − goCue[trial]`. Those times are the abscissa for the linear interpolation onto the shared 500-bin `taxis`, so the tongue output lands on exactly the same bins as the spikes. If `sglx`, `bitcode` or `fs` is missing the offset silently falls back to 0.

ii.
```python
def find_video_offset(obj: Any) -> float:
    ...
    if ev is None or bitcode is None or fs in (None, 0):
        return 0.0
    bit_start = event_array(ev, "bitStart")
    bitcode_start = np.asarray(gget(bitcode, "bitstart")).reshape(-1).astype(float)
    return matlab_mode(bitcode_start) / float(fs) - matlab_mode(bit_start)
```
```python
        frame_times = trial_frame_times(trial_view, n_frames) - vidshift - go_times[trial_idx]
        x_interp = linear_interp(frame_times, x, taxis)
```

iii. A port of `funcs/findVideoOffset.m` and the `interp1(traj(trix).frameTimes - vidshift - obj.bp.ev.(alignEv)(trix), ts, taxis)` line in `findPosition.m`; `matlab_mode` reproduces MATLAB's `mode` (with rounding to 9 decimals before counting). The offset is computed per call rather than cached, but is a session constant.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The same `obj.traj` DeepLabCut output, but camera view 1 (the **bottom** camera) and the feature `top_paw`, with `bottom_paw` as a fallback. Same fields: `featNames`, `ts[:, :2, featix]`, `frameTimes`, `NdroppedFrames`, plus the go cue and bit-code fields for the clock.

ii.
```python
    paw_speed, paw_visible, paw_feature = build_speed_trace(
        obj, view_index=1, feature_names=["top_paw", "bottom_paw"],
        go_times=go_times, taxis=taxis
    )
```

iii. Step 5 mapping table: "The shared code explicitly analyzes `top_paw_yvel_view2`; using `top_paw` as the primary paw marker is the closest published precedent" (citing `Figure1e.m`). `bottom_paw` is kept only as a fallback for sessions whose DLC model lacks `top_paw`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Identical pipeline to the tongue up to the interpolation, then the **non-tongue** branch of `findVelocity.m`: the interpolated x and y are nearest-filled across gaps (`fill_nearest_1d`, implemented as `np.interp` over the valid indices), the median frame-to-frame difference is taken as a baseline drift, `basederiv[0]` is subtracted from *both* the x and y gradients (faithfully reproducing the reference MATLAB, which also uses `basederiv(1)` for both axes), the velocities are nearest-filled again, and speed is the Euclidean magnitude in pixels per bin. The original (pre-fill) visibility mask is kept separately so filled bins can still be labelled `not_visible`. No normalisation is applied.

ii.
```python
        else:
            x_filled = fill_nearest_1d(x_interp)
            y_filled = fill_nearest_1d(y_interp)
            if not np.isfinite(x_filled).any() or not np.isfinite(y_filled).any():
                continue
            stacked = np.column_stack([x_filled, y_filled])
            basederiv = np.nanmedian(np.diff(stacked, axis=0), axis=0)
            baseline = basederiv[0] if np.isfinite(basederiv[0]) else 0.0
            x_vel = np.gradient(x_filled) - baseline
            y_vel = np.gradient(y_filled) - baseline
            x_vel = fill_nearest_1d(x_vel)
            y_vel = fill_nearest_1d(y_vel)
```

iii. Step 5 mapping table: "Use the same interpolation/alignment logic as the reference video code, compute speed magnitude from aligned x/y velocity ... Reference: `findPosition`, `findVelocity`." `findVelocity.m` reads:
`basederiv = median(diff(tsinterp),'omitnan'); xvel = gradient(...); if ~contains(feat,'tongue'), xvel = xvel - basederiv(1); yvel = yvel - basederiv(1); end` followed by `fillmissing(...,'nearest')` — the AI ported this line for line, including the `basederiv(1)`-for-both-axes quirk.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Exactly as the tongue: `discretize_session_signal` with the same per-session 50th percentile over finite, visible bins of kept trials, `not_visible = 2` elsewhere. Overall distribution 0.311 / 0.311 / 0.378.

ii.
```python
    paw_disc, paw_thresh = discretize_session_signal(paw_speed, paw_visible, 2, keep_trials)
```

iii. Same as 7-c — a direct implementation of the Decoder Task's per-session median split, with the untracked class preserved from the pre-fill visibility mask ("threshold visible samples at the session 50th percentile, and encode `not visible -> 2`").

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue, but using the bottom camera's own `frameTimes`: `frameTimes − vidshift − goCue[trial]`, then linear interpolation onto the shared 500-bin axis. The session bit-code offset is recomputed inside `build_speed_trace` for this call.

ii.
```python
    vidshift = find_video_offset(obj)
    ...
        frame_times = trial_frame_times(trial_view, n_frames) - vidshift - go_times[trial_idx]
```

iii. Same as 7-d; using each view's own `frameTimes` (rather than the side camera's for both) follows `findPosition.m`, which indexes `obj.traj{view}` for the view being processed.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file sitting beside the data structure, which holds one trace per trial with one value per camera frame. If that file is absent the code falls back to `obj.me` inside the data structure. Three on-disk wrappers are unwrapped (`me` as a bare array, `me.data`, or `me.data.data`). The frame clock comes from the **side** camera (`obj.traj{1}.frameTimes`) plus the bit-code offset and the trial's go cue.

ii.
```python
def load_motion_energy_source(obj, spec):
    if spec.motion_path.exists():
        me_struct = load_mat_file(spec.motion_path).get("me")
        return motion_energy_data_list(me_struct), True
    me_struct = gget(obj, "me", None)
    if me_struct is None:
        return None, False
    return motion_energy_data_list(me_struct), True

def motion_energy_data_list(me_struct):
    ...
    data = gget(me_struct, "data", me_struct)
    if isinstance(data, dict):
        data = gget(data, "data", data)
    return as_list(data)
```
```python
    side_trials = get_trials_view(obj, 0)
    vidshift = find_video_offset(obj)
```

iii. Step 5 mapping table: "Motion energy file `me.data` / `obj.me.data` and aligned video timing → `output[5]`. Load and align motion energy on the same neural time base as `loadMotionEnergy`." Step 10 Check 5 records that "the separate motion-energy files also come in mixed layouts (struct-wrapped vs bare arrays) and are both handled", matching `loadMotionEnergy.m`'s own `if isstruct(me.data), me.data = me.data.data; end` guard.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Almost none — the value is already one scalar per frame. The per-trial trace is linearly interpolated from the aligned frame times onto the 500-bin axis (with constant edge extrapolation, since `linear_interp` uses `fill_value=(first, last)`), then `fill_nearest_1d` fills any residual gaps, and every finite bin is marked valid. Trials are skipped (leaving them all-NaN → class 2) if the side-camera trial is flagged by `NdroppedFrames`, if the trace has fewer than 2 samples, or if it is entirely NaN.

ii.
```python
    for trial_idx in range(min(n_trials, len(motion_trials))):
        if trial_idx >= len(side_trials):
            continue
        if trial_invalid_video(side_trials[trial_idx]):
            continue
        raw = np.asarray(motion_trials[trial_idx], dtype=np.float64).reshape(-1)
        if raw.size < 2 or np.isnan(raw).all():
            continue
        frame_times = trial_frame_times(side_trials[trial_idx], raw.size) - vidshift - go_times[trial_idx]
        interp = linear_interp(frame_times, raw, taxis)
        interp = fill_nearest_1d(interp)
        aligned[trial_idx] = interp.astype(np.float32)
        valid[trial_idx] = np.isfinite(interp)
```

iii. The AI treated motion energy as a finished per-frame signal needing only resampling, following `loadMotionEnergy.m`, and noted that the paper's own manual bimodal movement threshold is deliberately replaced: "This intentionally differs from the paper's manual bimodal movement threshold because the decoder task explicitly requests a per-session median split."

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The same `discretize_session_signal` helper: per-session 50th percentile of the aligned motion energy over all valid, finite bins of kept trials; `< p50 → 0`, `>= p50 → 1`, everything else → 2, labelled `no_video` for this output. Because the trace is interpolated and gap-filled, essentially every bin of a trial with video is valid, so within video-bearing sessions the split is almost exactly 50/50 (0.500 / 0.500 per session).

ii.
```python
    me_disc, me_thresh = discretize_session_signal(motion_energy, motion_valid, 2, keep_trials)
```
```python
OUTPUT_VALUES = [..., ["low", "high", "no_video"]]
```

iii. Step 5: "On trials with usable video, threshold aligned motion-energy samples at the session 50th percentile: `<p50 -> 0`, `>=p50 -> 1`. If the trial/session lacks usable video or motion-energy data, encode `2` for the affected bins" — a direct implementation of the Decoder Task's third class, `2: no video`. The AI verified one trial end to end in Step 10: it reloaded `motionEnergy_JEB6_2021-04-18.mat`, rebuilt the aligned trace, discretised it with the saved session threshold (15.5) and matched the saved row with `np.allclose()`.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy has one value per side-camera frame, so it inherits the side camera's `frameTimes`, corrected by the same session bit-code offset and the trial's go cue, and is then interpolated onto the shared 500-bin axis — the same treatment and the same grid as the tongue, paw, spikes and input.

ii.
```python
        frame_times = trial_frame_times(side_trials[trial_idx], raw.size) - vidshift - go_times[trial_idx]
        interp = linear_interp(frame_times, raw, taxis)
```
```python
def trial_frame_times(trial_view, n_frames):
    frame_times = gget(trial_view, "frameTimes", None)
    if frame_times is None:
        return np.arange(1, n_frames + 1, dtype=np.float64) / 400.0
    ...
```

iii. Same reference basis as 7-d/8-d (`findVideoOffset.m`, `loadMotionEnergy.m`). The `n/400` fallback reproduces `findPosition.m`'s `traj(trix).frameTimes = (1:size(traj(trix).ts,1)) ./ 400` for trials where the field is absent.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases, handled in different ways. **Mixed file formats:** `mat73` with a `scipy.io` fallback; two `obj.clu` layouts and two `obj.traj` layouts are normalised; three `motionEnergy` wrappers are unwrapped. **Trials flagged as bad video** (`NdroppedFrames` all-NaN) are skipped and become `not_visible`/`no_video`. **Untracked frames**: DLC's NaN coordinates define the visibility mask; tongue velocities there are set to 0 (but labelled class 2), paw positions and velocities are nearest-filled (and still labelled class 2). **Missing `frameTimes`** (field absent, empty, or entirely NaN) are *fabricated* as `(1..n)/400` s — note this goes beyond `findPosition.m`, which only synthesises frame times when the field is missing and otherwise skips all-NaN trials. **Bins outside the recorded video** are filled by constant edge extrapolation and counted as valid rather than missing. **Missing bit-code/`sglx` fields** silently yield a zero video offset. **Trials after the end of the neural recording** are dropped via `neural_covered`. **Missing/short arrays** are guarded throughout by `gget` defaults, `as_list`, `flatten_strings` and `np.isfinite` masks, so no session aborts.

ii.
```python
def trial_invalid_video(trial_view: dict[str, Any]) -> bool:
    dropped = gget(trial_view, "NdroppedFrames", None)
    if dropped is None:
        return False
    arr = np.asarray(dropped).reshape(-1)
    return arr.size != 0 and np.isnan(arr.astype(float)).all()
```
```python
def trial_frame_times(trial_view, n_frames):
    frame_times = gget(trial_view, "frameTimes", None)
    if frame_times is None:
        return np.arange(1, n_frames + 1, dtype=np.float64) / 400.0
    frame_times = np.asarray(frame_times, dtype=np.float64).reshape(-1)
    if frame_times.size == 0 or np.isnan(frame_times).all():
        return np.arange(1, n_frames + 1, dtype=np.float64) / 400.0
    return frame_times
```
```python
def get_trials_view(obj: Any, view_index: int) -> list[dict[str, Any]]:
    view = as_list(traj)[view_index]
    if isinstance(view, dict):        # v7.3 (mat73): dict of trial-indexed fields
        ...
    return flatten_dicts(view)        # anything else
```
```python
    if len(trials_view) < n_trials:
        return speed, visible, None   # silently gives up on the whole session
```

iii. Step 5 Key Decision 6: "The shared scripts already define the correct video synchronization, interpolation, smoothing, and missing-data handling. The only deliberate extensions are ... preserving missing-visibility masks for the required `not visible` / `no video` classes." Step 10 Check 5 claims the mixed `obj.traj` layouts were verified as handled. **That claim is wrong.** In the 10 sessions stored in the old MATLAB v5 format (`JEB23_2023-10-18`, `JEB23_2023-10-21` and all eight `JEB24` sessions), `scipy.io.loadmat(..., simplify_cells=True)` returns `obj['traj'][view]` as an object array of `mat_struct` instances; `flatten_dicts` recognises only `dict`/`list`/object-array-of-dicts, so it returns an empty list, `len(trials_view) < n_trials` triggers, and `build_speed_trace`/`build_motion_energy_trace` return all-NaN. I confirmed this directly against the raw files: `data_structure_JEB24_2023-10-23.mat` has 343 trials of side-camera tracking with `featNames = ['tongue', 'left_tongue', ...]`, `ts` of shape (2153, 3, 7) and finite `frameTimes`, and a valid `motionEnergy_JEB24_2023-10-23.mat`, yet the agent's `get_trials_view` yields 0 trials. The consequence is that all three video-derived outputs are 100 % `not_visible`/`no_video` in those 10 sessions — 3,373 of 13,762 trials (24.5 %) — which is exactly the `no_video` fraction reported in the AI's own verification output. The AI documented this as a property of the data ("Some randomized sessions lack separate motion-energy files / usable video ... missing-video fraction is explained by raw data availability") rather than investigating it; all of its `np.allclose` sanity checks were run on `JEB6_2021-04-18`, a v7.3 session, so none of them could catch it.

## 11-a. What are the most time-consuming steps of the code?

i. The AI reports no per-step profiling, only per-session wall time (`print(f"  kept ... in {time.perf_counter() - session_t0:.1f}s")`) and a total. The full conversion took 212.6 s for 44 sessions (1.5–9.1 s per session). From the code structure the two dominant costs are (1) reading each `.mat` file — `mat73.loadmat` materialises the whole `obj` tree including `clu.spkWavs` and `clu.tm`, which are never used — and (2) the nested `for unit → for trial` loop that calls `np.histogram` and `np.convolve` once per (unit, trial) pair, i.e. roughly 2,700 units × 300 trials ≈ 0.8 M small NumPy calls across the dataset. Per-session time tracks unit count × trial count, confirming the second loop is a real contributor alongside I/O.

ii.
```python
    for unit_idx, unit in enumerate(units):
        ...
        for tr in np.unique(trial_ids):
            tr_mask = trial_ids == tr
            counts, _ = np.histogram(aligned_times[tr_mask], bins=edges)
            rates = counts.astype(np.float64) / DT
            trialdat[unit_idx, tr] = my_smooth_1d(rates, boundary_condition="reflect")
```
```python
        print(f"[{sess_num:02d}/{len(specs):02d}] Processing {spec.session_id} ({spec.cohort})")
        ...
        print(f"  kept {len(processed['neural_trials'])} trials, ... in {time.perf_counter() - session_t0:.1f}s")
```

iii. CONVERSION_NOTES Step 6: "Code inefficiencies identified: MATLAB-style nested structs/cells produce expensive Python-side branching if normalized repeatedly inside inner loops. Video alignment can become expensive if feature-name parsing or trajectory-layout parsing happens per trial instead of once per session." Step 7: "The sample run indicates roughly 5-6 seconds per session ... short enough that no parallelization is required yet" (estimated ~4 min for 44 sessions; actual 3.5 min), so the AI stopped optimising once it was comfortably under the 15-minute budget.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest one is the spike-binning double loop in `process_session`: all of a unit's spikes could be binned for every trial at once with a single `np.histogram2d(trial_index, aligned_time, bins=[trial_edges, edges])`, and the causal smoothing could then be one `scipy.ndimage.convolve1d`/`np.apply_along_axis` over the `(n_trials, 500)` matrix instead of one `np.convolve` per trial. The `tr_mask = trial_ids == tr` comparison inside the loop is additionally O(n_spikes) per trial, making the loop quadratic in spike count per unit. Other loops that remain per-trial are `build_speed_trace` and `build_motion_energy_trace` (harder to vectorise, since each trial has a different number of frames) and `post_go_choice` over trials (trivial cost). The `get_trials_view` reconstruction also builds a Python dict per trial per view.

ii.
```python
        for tr in np.unique(trial_ids):
            tr_mask = trial_ids == tr                      # O(n_spikes) inside the loop
            counts, _ = np.histogram(aligned_times[tr_mask], bins=edges)
            trialdat[unit_idx, tr] = my_smooth_1d(rates, boundary_condition="reflect")
```
```python
    for trial_idx in range(n_trials):
        trial_view = trials_view[trial_idx]
        ...
        x_interp = linear_interp(frame_times, x, taxis)
```

iii. The AI did not identify the spike-binning loop as vectorisable. Its stated speed-ups (Step 6) were structural rather than vectorisation: "Session lists and probe selections are parsed once ...; Trajectory and cluster layouts are normalized once per session before inner-loop processing; Neural trial tensors are preallocated per session and filled in place." Its justification for stopping there was the runtime estimate: ~5.4 s/session, "~4.0 min for 44 sessions", well inside the instruction's 15-minute threshold.

## 11-c. What processing does the code repeat multiple times?

i. Four repetitions, all small but avoidable. (1) `find_video_offset(obj)` is recomputed three times per session — once in each `build_speed_trace` call and once in `build_motion_energy_trace` — although it is a session constant. (2) `get_trials_view(obj, 0)` is rebuilt twice per session (once for the tongue, once for motion energy), each time materialising one dict per trial for the whole view. (3) `feature_index(trial_view, chosen_feature)` re-flattens and re-scans `featNames` on every trial, even though the feature index is fixed within a session (it is already resolved once from trial 0 to pick the feature name). (4) The behavioural masks `hit`, `miss`, `no`, `early`, `stim`, `autowater` are computed once inside `low_fr_condition_indices` and then a second time in `process_session`. Additionally `np.unique(trial_ids)` and the boolean comparison are recomputed per trial inside the binning loop.

ii.
```python
    vidshift = find_video_offset(obj)          # in build_speed_trace, called twice
...
    side_trials = get_trials_view(obj, 0)      # already built for the tongue
    vidshift = find_video_offset(obj)          # third time
```
```python
    condition_indices = low_fr_condition_indices(bp, n_trials)   # computes hit/miss/no/early/stim/autowater
    ...
    hit = bool_array(bp, "hit", n_trials)                        # recomputed here
    miss = bool_array(bp, "miss", n_trials)
    no = bool_array(bp, "no", n_trials)
    early = bool_array(bp, "early", n_trials)
    stim = stim_enable_array(bp, n_trials)
    autowater = bool_array(bp, "autowater", n_trials)
```

iii. The AI asserted the opposite in Step 6 — "Trajectory and cluster layouts are normalized once per session before inner-loop processing" — which is true for the cluster layout and for the *feature-name choice*, but not for the trajectory view (rebuilt per stream) or the video offset. None of these repetitions affects correctness, and each is cheap relative to file I/O, which is presumably why they were not caught.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) **Neural rates for discarded trials**: `trialdat` is built and smoothed for all `n_trials` raw trials, including early-lick, photostim and uncovered trials (~10 % of trials), before `keep_trials` subsets it. (2) **Velocity and motion-energy traces for discarded trials**: `build_speed_trace` and `build_motion_energy_trace` also run over all raw trials. (3) **The eight condition PSTHs** are computed solely to produce one scalar per unit for the low-FR filter and are then thrown away (this one is unavoidable given the chosen filter, which is a port of `removeLowFRClusters.m`). (4) **Unused raw fields**: `mat73`/`scipy` load the entire `obj` tree — `clu.spkWavs` (38×120 per cluster), `clu.tm`, `clu.channel`, `sglx` index arrays, and all the tracked features other than the tongue and the paw — and `split_probe_struct` explicitly copies `tm` and `site` into every unit dict although neither is ever read. (5) **Dead code**: `import math`, `from typing import Iterable`, and the `"zeropad"`/`"none"` branches of `my_smooth_1d` are never exercised. (6) The continuous speed/energy traces themselves are discarded after discretisation (only thresholds survive, in `session_info`), and the diagnostic PNGs are optional output.

ii.
```python
    trialdat = np.zeros((len(units), n_trials, taxis.size), dtype=np.float32)   # all raw trials
    ...
    for tr in keep_trials:                                                      # only a subset saved
        neural_trials.append(trialdat[:, tr, :].astype(np.float32, copy=False))
```
```python
        units.append({
            "quality": ..., "tm": np.asarray(tm_list[i]).reshape(-1),   # never used
            "trial": ..., "trialtm": ..., "site": site_list[i],         # never used
        })
```
```python
    psth = np.zeros((trialdat.shape[0], taxis.size, len(condition_indices)), dtype=np.float32)
    ...
    mean_fr = np.nanmean(psth, axis=(1, 2))      # only this scalar per unit survives
```

iii. The AI did not enumerate these; its efficiency notes focus on avoiding repeated structure traversal and on preallocation. The design is defensible in that filtering trials *before* binning would require reshuffling trial indices used by `neural_covered` and by the condition masks (which are defined over raw trial numbers), and the PSTH pass is required by the reference's own low-FR rule. The waste was accepted because the total runtime (212.6 s) was already well inside budget.
