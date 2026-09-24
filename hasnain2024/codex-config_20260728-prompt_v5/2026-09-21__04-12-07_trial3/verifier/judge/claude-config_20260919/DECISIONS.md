# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not glob the data folders. It hard-codes a manifest of 44 sessions — 25 in `/app/data/Ephys_Behavior` (fixed delay) and 19 in `/app/data/RandomizedDelay_Ephys_Behavior` (randomized delay) — transcribed from the authors' `code/DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m` loaders, each entry carrying the animal, the date, the cohort (which fixes the folder) and the ALM probe number(s) to use. Two sessions are two-probe (`JEB15_2022-07-26/27/28`, probes 1 and 2). `JEB23_2023-10-20` is excluded because it is commented out in the reference loader, and the two `JEB24` randomized-delay files with no `clu` field (`2023-10-03`, `2023-10-04`) are excluded because they contain no neural data. The behaviour-only folders (`DelayInhibition_BilatMC_Behavior`, `GoCueInhibition_BilatMC_Behavior`) are not used at all. Each session file is read once by `load_any_mat`, which dispatches on the MATLAB container format: `mat73.loadmat` for v7.3/HDF5 files and `scipy.io.loadmat` for the 11 v5 files, followed by `normalize_obj`/`convert_mat_struct` which flatten both into one common nested-dict/list representation for `bp`, `traj` and `clu`. Motion energy is read from the sibling `motionEnergy_<ANM>_<DATE>.mat` file by the same loader.

ii.
```python
@dataclass(frozen=True)
class SessionSpec:
    subject: str
    date: str
    cohort: str
    probes: tuple[int, ...]
    ...
    @property
    def data_dir(self) -> Path:
        return EPHYS_DIR if self.cohort == "fixed" else RAND_DIR

FIXED_SPECS = [
    SessionSpec("EKH1", "2021-08-07", "fixed", (2,)),
    ...
    SessionSpec("JGR3", "2021-11-18", "fixed", (1,)),
]
RANDOMIZED_SPECS = [
    SessionSpec("JEB11", "2022-05-10", "randomized", (1,)),
    ...
    SessionSpec("JEB24", "2023-11-03", "randomized", (1,)),
]
ALL_SPECS = FIXED_SPECS + RANDOMIZED_SPECS
```

```python
def load_any_mat(path: Path) -> dict[str, Any]:
    if h5py.is_hdf5(path):
        return mat73.loadmat(str(path))
    return convert_mat_struct(loadmat(path, squeeze_me=True, struct_as_record=False))
```

```python
for sess_idx, spec in enumerate(session_specs, start=1):
    print(f"[{sess_idx:02d}/{len(session_specs):02d}] Processing {spec.session_id}")
    session_data, _ = process_session(spec, show_processing=show_processing and sess_idx <= 2)
```

iii. From CONVERSION_NOTES Step 5/6: *"Use the reference ALM-session selection when available: Follow the paper code's `Recording and video/load*_ALMVideo.m` session/probe logic"*, and *"Include only sessions with neural data … exclude the behavior-only directories and any file with no `clu`, because the decoder requires neural input."* The dual loader is justified in Step 2: *"Most `data_structure_*.mat` files are MATLAB v7.3 / HDF5. 11 files in `RandomizedDelay_Ephys_Behavior` are MATLAB v5."* The resulting 25 + 19 = 44 sessions are cross-checked against the paper's reported 25 fixed-delay and 19 randomized-delay recording sessions (Step 4 / Step 9 consistency table).

## 1-b. How are the data split into subjects (mice)?

i. The subject is the animal id carried in the `SessionSpec` (which is also the prefix of the filename, e.g. `JEB19_2023-04-19` → `JEB19`); the id inside the file (`obj.meta.anm`) is not used. During assembly, `subjects` is built in order of first appearance across the session list and `subject_idx` records each session's index into that list. This yields 14 subjects over 44 sessions (EKH1, EKH3, JEB6, JEB7, JEB11–JEB15, JEB19, JEB23, JEB24, JGR2, JGR3), with 1–8 sessions per subject.

ii.
```python
    @property
    def session_id(self) -> str:
        return f"{self.subject}_{self.date}"
```

```python
        if spec.subject not in subject_lookup:
            subject_lookup[spec.subject] = len(subjects)
            subjects.append(spec.subject)
        subject_idx.append(subject_lookup[spec.subject])
...
    "subjects": subjects,
    "subject_idx": np.asarray(subject_idx, dtype=np.int64),
```

iii. CONVERSION_NOTES Step 5 maps *"Animal/session identity from filename / `obj.ex`"* to `subjects`/`subject_idx`, noting *"Session order will be deterministic and documented."* Step 9 records the resulting count (14) and explicitly flags that this differs from one sentence of the paper (9 fixed-delay mice) but matches the session loaders and the files actually present: *"Prioritize the explicit code/data session list over the one-number mouse-count discrepancy in text."*

## 1-c. How are the data split into sessions?

i. One `SessionSpec` = one `data_structure_<ANM>_<DATE>.mat` file = one element of `neural`, `input`, `output`, `brain_region_idx` and `metadata['session_info']`. The cohort field selects the folder, so fixed-delay and randomized-delay sessions are handled uniformly in a single list rather than as two datasets, and the two-probe sessions are merged into a single session by concatenating the kept units from both probes. Session order is the fixed order of `ALL_SPECS` (25 fixed-delay then 19 randomized-delay).

ii.
```python
    @property
    def data_path(self) -> Path:
        return self.data_dir / f"data_structure_{self.subject}_{self.date}.mat"
```

```python
    kept_rates: list[np.ndarray] = []
    for probe_num in spec.probes:
        probe_idx = probe_num - 1
        if probe_idx < 0 or probe_idx >= len(obj["clu"]):
            continue
        kept_rates.extend(process_probe(obj["clu"][probe_idx], go_cue, valid_mask, condition_positions))
```

iii. Step 6: *"Hard-coded the reference ALM session/probe manifest from `Recording and video/load*_ALMVideo.m`: 25 fixed-delay sessions, 19 randomized-delay sessions, randomized-delay exclusion of commented-out `JEB23_2023-10-20`."* Step 9 reports cohort summaries (fixed: 25 sessions / 10 subjects / 7,426 trials / 1,494 units; randomized: 19 / 4 / 6,336 / 870).

## 1-d. How are the data split into trials?

i. A trial is one row of the Bpod table: every per-trial field of `obj.bp` is read as a length-`Ntrials` vector by `get_bp_array`, and `bp.ev.goCue` provides exactly one alignment time per trial. Spikes carry their own 1-based `trial` index (`clu.trial`) and a within-trial time (`clu.trialtm`), and camera frames are already stored per trial in `obj.traj{view}(trial)`, so no trial boundaries need to be reconstructed. The subset of trials that survives curation is `valid_idx = np.flatnonzero(valid_mask)`, and that same index vector is applied to every stream (neural, motion energy, tongue, paw, per-trial labels), so all streams stay row-aligned. A session with fewer than 2 surviving trials is rejected.

ii.
```python
def get_bp_array(bp: dict[str, Any], key: str, ntrials: int, dtype=np.float64) -> np.ndarray:
    if key not in bp:
        if np.issubdtype(np.dtype(dtype), np.bool_):
            return np.zeros(ntrials, dtype=bool)
        return np.full(ntrials, np.nan, dtype=dtype)
    arr = as_1d_numeric(bp[key], dtype=np.float64)
    if arr.size != ntrials:
        arr = np.broadcast_to(arr, (ntrials,))
    ...
```

```python
    ntrials = int(float(bp["Ntrials"]))
    go_cue = get_bp_array(bp["ev"], "goCue", ntrials, dtype=np.float64)
    valid_mask = build_valid_mask(bp, ntrials, go_cue)
    ...
    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size < 2:
        raise RuntimeError(f"{spec.session_id}: fewer than 2 valid trials after filtering.")
```

iii. Implicit in Step 5's variable mapping (all per-trial fields are indexed by trial). The 2-trial floor is the format requirement from the instructions (*"There needs to be at least two trials within each session in order to evaluate the decoder performance"*).

## 1-e. How are trials filtered based on quality controls?

i. Four conditions, all combined in one mask before anything is computed:
1. `~bp.early` — early-lick trials dropped, following the paper;
2. `~bp.stim.enable` — photostimulation trials dropped (sessions without a `stim` struct are treated as no-stim);
3. `bp.hit | bp.miss | bp.no` — the trial must carry one of the three outcome flags;
4. `np.isfinite(goCue)` — the trial must have a usable alignment time.

Then a fifth, data-driven cut: trials whose 1-based index exceeds the largest trial index appearing in any spike train of the selected probe(s) are dropped, because in two `JEB24` randomized-delay sessions the behaviour continued after the probe stopped and those trials would otherwise enter the dataset as 1,000 bins of exactly zero firing for every neuron. Across the dataset 13,762 of 14,972 trials are kept (312.8 per session on average).

ii.
```python
def build_valid_mask(bp: dict[str, Any], ntrials: int, go_cue: np.ndarray) -> np.ndarray:
    early = get_bp_array(bp, "early", ntrials, dtype=bool)
    stim = get_stim_enable(bp, ntrials)
    hit = get_bp_array(bp, "hit", ntrials, dtype=bool)
    miss = get_bp_array(bp, "miss", ntrials, dtype=bool)
    no = get_bp_array(bp, "no", ntrials, dtype=bool)
    outcome = hit | miss | no
    return (~early) & (~stim) & outcome & np.isfinite(go_cue)
```

```python
def max_recorded_trial(obj: dict[str, Any], probes: tuple[int, ...]) -> int:
    max_trial = 0
    for probe_num in probes:
        ...
        for trial_arr in probe["trial"]:
            arr = as_1d_numeric(trial_arr, dtype=np.float64)
            if arr.size:
                max_trial = max(max_trial, int(np.nanmax(arr)))
    return max_trial
...
    last_neural_trial = max_recorded_trial(obj, spec.probes)
    if last_neural_trial > 0 and last_neural_trial < ntrials:
        valid_mask &= (np.arange(1, ntrials + 1) <= last_neural_trial)
```

iii. Step 5 Key Decision 3: *"Filter trials conservatively but keep decoder-relevant outcomes: Exclude `early` trials and `stim.enable` trials to match the paper/reference conditions. Keep `hit`, `miss`, and `no` trials so the required `outcome` output can include correct / incorrect / ignore."* The recording-coverage cut is documented as a Step 9/10 fix: *"Two JEB24 randomized-delay sessions (`2023-10-23`, `2023-11-03`) contained more behavioral trials than were covered by the neural cluster trial indices … Initial full conversion therefore produced blocks of all-zero neural trials at the session tail, which `train_decoder.py --verify-only` warned about. Fix: added an explicit `last_neural_trial` filter."* (kept trials changed 320→292 and 334→301).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The spike-sorted clusters of the ALM probe(s) named in the session manifest: `obj.clu{probe}(i).trial` (1-based trial index of each spike), `obj.clu{probe}(i).trialtm` (spike time relative to that trial's start, on the behaviour clock) and `obj.clu{probe}(i).quality` (manual curation label). `obj.bp.ev.goCue` is the second input, since it defines the alignment. No other cluster fields (waveforms, channels, `tm`) are used.

ii.
```python
    for clu_idx in np.flatnonzero(good_clusters):
        trial_ids = as_1d_numeric(probe["trial"][clu_idx], dtype=np.float64)
        trial_times = as_1d_numeric(probe["trialtm"][clu_idx], dtype=np.float64)
        rates = bin_cluster_rates(trial_ids, trial_times, go_cue, valid_mask)
```

iii. Step 5 maps *"`obj.clu` spike times from ALM probe(s)"* → `neural`, citing the reference functions `loadSessionData`, `processData`, `alignSpikes`, `getSeq`, `removeLowFRClusters`, `mySmooth`. Step 1 records that `alignSpikes` works on `obj.clu{prb}(clu).trialtm`.

## 2-b. How is the `neural` data processed?

i. Three steps, in a direct port of the reference MATLAB `getSeq.m` + `mySmooth.m`:
1. **Count** spikes into the 1000 × 5 ms grid spanning −2.5 to +2.5 s from the go cue, vectorised with a single `np.bincount` over a flattened `(time_bin, trial)` index;
2. **Convert to rate** by dividing counts by the bin width (`/ DT`), giving spikes/s;
3. **Smooth** along time with `my_smooth(..., N=15, 'reflect')`, a line-by-line Python port of the authors' `mySmooth.m`: `gausswin(15)` (alpha = 2.5), the first `floor(15/2) = 7` taps **zeroed to make the kernel causal**, normalised to sum 1, convolved with `mode='same'`, and with the authors' "reflect" boundary handling (prepend the first 15 samples, then trim them off).

No normalisation, baseline subtraction or z-scoring is applied, so stored values are firing rates in Hz (`float32`). Units from both probes of a two-probe session are concatenated into one population.

ii.
```python
def matlab_gausswin(n: int, alpha: float = 2.5) -> np.ndarray:
    idx = np.arange(n, dtype=np.float64) - (n - 1.0) / 2.0
    sigma = (n - 1.0) / (2.0 * alpha)
    return np.exp(-0.5 * (idx / sigma) ** 2)


def my_smooth(x, n, bctype="none"):
    ...
    if bctype.lower() == "reflect":
        x_filt = np.concatenate([x[:n, :], x], axis=0)
        trim = n
    ...
    kern = matlab_gausswin(n)
    kern[: math.floor(kern.size / 2)] = 0.0
    kern /= kern.sum()
    for col in range(x_filt.shape[1]):
        out[:, col] = np.convolve(x_filt[:, col], kern, mode="same")
    out = out[trim:, :]
```

```python
    bin_idx = np.floor((aligned - TMIN) / DT).astype(np.int64)
    flat_idx = bin_idx * n_valid + kept_pos
    counts_flat = np.bincount(flat_idx, minlength=TIME_CENTERS.size * n_valid)
    counts = counts_flat.reshape(TIME_CENTERS.size, n_valid).astype(np.float64)
    rates = my_smooth(counts / DT, SMOOTH, "reflect")
    return rates.astype(np.float32)
```

iii. Step 5 Key Decision 5: *"Use 5 ms neural bins and causal smoothing matching the reference pipeline."* Step 10 Check 3: *"Reference: `getSeq` bins from `tmin=-2.5` to `tmax=2.5` in `dt=1/200`; `mySmooth(...,15,'reflect')`. Converter: same time window, same bin width, same causal Gaussian kernel and prepend-style reflect boundary handling."* Step 3 notes the paper's own wording: *"Single-trial data are binned at 5 ms and smoothed with a causal Gaussian kernel."* The port was checked against a from-scratch reconstruction of one neuron/trial from the raw `.mat` file with `np.allclose(...) == True` (Step 10 Check 2).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, matching the reference's two-stage curation:
1. **Quality label**: the cluster's `quality` string is stripped and lower-cased and the cluster is dropped if it is one of `garbage`, `gabrga`, `noisy`, `real?` — exactly the drop list of `findClusters.m` under `params.quality = {'all'}`. Unlabelled clusters (non-string) become `''` and are kept. Multi-units are kept.
2. **Low firing rate**: a reimplementation of `removeLowFRClusters.m`. Four condition PSTHs are formed by averaging the already-smoothed single-trial rates over the four conditions of `getDefaultParams.m` (`R&hit&~autowater`, `L&hit&~autowater`, `R&hit&autowater`, `L&hit&autowater`, all restricted to the surviving trials); conditions with no trials contribute an all-zero PSTH, as in MATLAB. The unit is kept only if the grand mean of that (time × condition) stack exceeds 1 Hz.

Sessions retaining fewer than 10 units would be rejected outright (none were). This leaves 2,364 units over the 44 sessions (mean 53.7, min 16, max 142).

ii.
```python
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}
LOW_FR_HZ = 1.0
MIN_UNITS_PER_SESSION = 10

def cluster_good_mask(qualities: list[Any]) -> np.ndarray:
    labels = [str(q).strip().lower() if q is not None else "" for q in qualities]
    return np.array([label not in QUALITY_EXCLUDE for label in labels], dtype=bool)
```

```python
        psth_stack = []
        for cond_pos in condition_positions:
            if cond_pos.size == 0:
                psth_stack.append(np.zeros(TIME_CENTERS.size, dtype=np.float32))
            else:
                psth_stack.append(rates[:, cond_pos].mean(axis=1))
        mean_fr = float(np.mean(np.stack(psth_stack, axis=1)))
        if mean_fr > LOW_FR_HZ:
            kept_rates.append(rates)
```

```python
    conds = [
        right & hit & (~autowater),
        left & hit & (~autowater),
        right & hit & autowater,
        left & hit & autowater,
    ]
```

iii. Step 4 resolves the code-internal discrepancy on the threshold: *"`getDefaultParams.m` uses `lowFR=0.5`, but figure/analysis scripts overwhelmingly set `lowFR=1` … Methods say analyses generally used units with firing rates exceeding 1 Hz. Use `>1 Hz`."* Step 5 Key Decision 6: *"Keep all ALM units above 1 Hz, not only single units: The paper explicitly uses all >1 Hz units for most population analyses."* Step 3 records the ≥10-units-per-session inclusion rule from Methods. Step 9 explains the residual gap to the paper's 2,496 recorded units: *"converted count is expected to differ from paper's recorded-unit count because the converter applies the reference low-FR analysis filter."*

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. A single subtraction, exactly as in `alignSpikes.m`. `clu.trialtm` is already on the behaviour clock and already relative to its own trial's start, and `bp.ev.goCue` is on the same clock, so `trialtm − goCue[trial]` gives seconds from go-cue onset with no offset or interpolation. Spikes falling outside `[−2.5, 2.5)` are discarded, and spikes whose trial index is out of range or not in the kept set are discarded via a lookup table. No clock correction is needed for the neural stream (unlike the camera streams, which need the bitcode offset).

ii.
```python
    original_idx = trial_ids.astype(np.int64) - 1
    in_range = (original_idx >= 0) & (original_idx < go_cue.size)
    original_idx = original_idx[in_range]
    trial_times = trial_times[in_range]

    trial_lookup = np.full(valid_mask.size, -1, dtype=np.int32)
    trial_lookup[np.flatnonzero(valid_mask)] = np.arange(n_valid, dtype=np.int32)
    kept_pos = trial_lookup[original_idx]

    aligned = trial_times - go_cue[original_idx]
    keep = (kept_pos >= 0) & np.isfinite(aligned) & (aligned >= TMIN) & (aligned < TMAX)
```

iii. Step 10 Check 3: *"Reference: `alignSpikes` subtracts `bp.ev.goCue` per trial … Converter: same go-cue subtraction."* Step 4 also justifies aligning WC (autowater) trials to the go cue: *"Raw event fields include non-NaN `goCue` timestamps even on `autowater` trials … For this project, align all trials to `goCue` as instructed by the task."*

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 5 ms (`DT = 1/200`), with 1,000 non-overlapping bins spanning −2.5 to +2.5 s around the go cue. The edge vector and bin centres are built once at module level and are the same for every trial, session and stream, so the neural data, the time input, and the three camera-derived outputs all live on one common time axis. No rebinning or resampling is applied after the initial binning: spikes are histogrammed straight into the 5 ms grid, and the video streams (400 Hz ≈ 2.5 ms frames) are brought onto the same grid by linear interpolation at the bin centres rather than by averaging then rebinning. `metadata['time_bin_size']` is recorded as 5.0 ms.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 200.0
EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_CENTERS = (EDGES[:-1] + EDGES[1:]) / 2.0
```

```python
            "time_bin_size": 1000.0 * DT,
            "temporal_alignment_event": "Go cue onset",
            "off_start": TMIN,
            "off_end": TMAX,
```

iii. Step 4: *"`getDefaultParams.m` uses `dt=1/200` (5 ms); tutorial script in `WorkingWithDataObjs.m` uses `dt=1/100` (10 ms) … Methods explicitly state single-trial neural activity was binned in 5 ms intervals. Use 5 ms bins in the converter. Treat the 10 ms tutorial as illustrative, not authoritative."*

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Nothing in the raw data beyond the go cue itself, which defines the origin of the axis. The input is the vector of bin centres of the shared −2.5 to +2.5 s grid, i.e. `[-2.4975, -2.4925, …, 2.4975]`, identical for every trial and every session. It corresponds to `obj.time` in the reference `getSeq.m`, which is also built as `edges + dt/2` with the last element dropped.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_CENTERS = (EDGES[:-1] + EDGES[1:]) / 2.0
...
    session_input = [TIME_CENTERS[None, :].astype(np.float32).copy() for _ in range(valid_idx.size)]
...
    "input_names": ["time_from_go_cue_s"],
```

iii. Step 5 maps *"Shared time grid relative to go cue"* → `input[0]`, reference function `getSeq` (`obj.time`), with the note *"Decoder input is the required continuous time-from-go-cue signal."*

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None. The axis is defined by the conversion, not derived from data: `EDGES` is an `arange` from −2.5 to 2.5 in 5 ms steps and `TIME_CENTERS` is the midpoint of consecutive edges. It is cast to `float32` and stored with shape `(1, 1000)` per trial (a fresh copy per trial, not a shared view).

ii.
```python
    session_input = [TIME_CENTERS[None, :].astype(np.float32).copy() for _ in range(valid_idx.size)]
```

iii. N/A — the axis is constructed, not measured. Step 10 Check 2 verified it against the expected vector: *"Manual reconstruction: expected shared time axis `[-2.4975, ..., 2.4975]` … Result: `np.allclose(...) == True`."*

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid. Spike times are expressed relative to their trial's go cue and assigned to bin `floor((t − TMIN)/DT)`, i.e. the half-open interval `[EDGES[k], EDGES[k+1])`, and the input value at index `k` is the centre of that same interval. So bin *k* denotes the same 5 ms window in the neural array and in the input array, and the same is true of the three camera outputs, which are interpolated at `TIME_CENTERS`. The only caveat is that the neural rates are smoothed with a *causal* kernel, so the rate at bin *k* reflects the ~35 ms of spiking up to and including bin *k* (this is the reference pipeline's behaviour, not a misalignment of the axes).

ii.
```python
    bin_idx = np.floor((aligned - TMIN) / DT).astype(np.int64)
```
```python
    session_input = [TIME_CENTERS[None, :].astype(np.float32).copy() for _ in range(valid_idx.size)]
```

iii. Step 10 Check 3: *"Reference: `obj.time` is the shared bin-center axis. Converter: uses that shared time axis directly as the only decoder input."*

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The recorded lick-port contact times, `obj.bp.ev.lickL` and `obj.bp.ev.lickR` (one variable-length array of times per trial), together with `obj.bp.ev.goCue`. Notably the AI does **not** infer direction from the instructed side (`bp.R`/`bp.L`) plus the outcome flags; it reads the animal's actual licks.

ii.
```python
    lick_l = ensure_event_list(bp["ev"].get("lickL", []), ntrials)
    lick_r = ensure_event_list(bp["ev"].get("lickR", []), ntrials)
    lick_direction = first_post_go_lick_direction(lick_l, lick_r, go_cue)[valid_idx]
```

```python
def ensure_event_list(events: Any, ntrials: int) -> list[np.ndarray]:
    ...
    for item in events:
        if item is None:
            out.append(np.empty(0, dtype=np.float64)); continue
        arr = np.asarray(item, dtype=np.float64)
        if arr.ndim == 0:
            out.append(np.empty(0) if np.isnan(arr) else arr.reshape(1))
        else:
            out.append(arr.reshape(-1))
```

iii. Step 5 Key Decision 8: *"Define lick direction from actual behavior, not instructed side: First post-go-cue lick side is the cleanest behavioral output and naturally gives `none` for ignore trials."*

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. For each trial, the left and right lick times are restricted to the response window `[goCue, goCue + 3 s)`; the side whose earliest lick comes first wins (a tie, which cannot really occur, resolves to left); if neither port was licked in that window the trial is labelled `none`. Codes are left 0, right 1, none 2. The label is per trial and is then broadcast across all 1,000 time bins so that all six outputs share one `(6, 1000)` array. The 3 s window is the paper's own definition of an ignore trial.

ii.
```python
RESPONSE_WINDOW_S = 3.0

def first_post_go_lick_direction(lick_l, lick_r, go_cue) -> np.ndarray:
    direction = np.full(go_cue.shape, 2, dtype=np.int8)
    for trix in range(go_cue.size):
        left = lick_l[trix]; right = lick_r[trix]
        left = left[(left >= go_cue[trix]) & (left < go_cue[trix] + RESPONSE_WINDOW_S)]
        right = right[(right >= go_cue[trix]) & (right < go_cue[trix] + RESPONSE_WINDOW_S)]
        left_first = left[0] if left.size else np.inf
        right_first = right[0] if right.size else np.inf
        if not np.isfinite(left_first) and not np.isfinite(right_first):
            direction[trix] = 2
        elif left_first <= right_first:
            direction[trix] = 0
        else:
            direction[trix] = 1
    return direction
```

```python
            np.full(TIME_CENTERS.size, lick_direction[trix], dtype=np.int8),
```

iii. Step 3 records *"Ignore trials are trials without a response within 3 s of the go cue"*, which sets the window. Step 5 Key Decision 7: *"Lick direction, context, and outcome are conceptually per-trial, yet storing them as constant 1 x T traces keeps output shapes uniform alongside time-varying kinematic outputs."* Step 10 Check 2 verified the labels for the first 5 kept trials of `JEB13_2022-09-13` against a from-raw recomputation.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. A single per-trial flag, `obj.bp.autowater`, which marks the water-cued (WC) trials in which water is delivered at a random port with no auditory cues. Sessions without WC blocks simply carry a constant DR label.

ii.
```python
    autowater = get_bp_array(bp, "autowater", ntrials, dtype=bool)[valid_idx]
```

iii. Step 5 maps `bp.autowater` → `output[1]`, reference function `findTrials` condition logic, with the note *"DR-only sessions remain valid with constant `DR` label."*

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling: autowater → WC (0), otherwise DR (1), matching the code ordering required by the task spec (`WC, DR`). The label is per trial and broadcast across the 1,000 bins. Over the full dataset this gives 9.7% WC and 90.3% DR.

ii.
```python
    context = np.where(autowater, 0, 1).astype(np.int8)
...
            np.full(TIME_CENTERS.size, context[trix], dtype=np.int8),
...
            ["WC", "DR"],
```

iii. Step 5 Variable Mapping: *"Per-trial categorical label repeated across time bins: `DR` if 0, `WC` if 1."* Step 4 notes the reference `findTrials` conditions distinguish DR (`~autowater`) from WC (`autowater`).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Three per-trial flags of `obj.bp`: `hit`, `miss` and `no`. Unlike the reference, which derives ignore as "neither hit nor miss", the AI reads `bp.no` explicitly (and also uses the three-way OR as a trial-validity requirement, see 1-e).

ii.
```python
    hit = get_bp_array(bp, "hit", ntrials, dtype=bool)[valid_idx]
    miss = get_bp_array(bp, "miss", ntrials, dtype=bool)[valid_idx]
    no = get_bp_array(bp, "no", ntrials, dtype=bool)[valid_idx]
```

iii. Step 5 maps *"`bp.hit`, `bp.miss`, `bp.no`"* → `output[2]` with the note *"Early trials are excluded entirely rather than encoded as an outcome."*

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A three-way relabelling into the codes required by the task: incorrect 0 (miss), correct 1 (hit), ignore 2 (no response). The array is initialised to 2, then miss, hit and no are written in that order; since the three flags are mutually exclusive in these files the write order is immaterial. The label is per trial and broadcast across all bins. Ignore trials are kept as a third class rather than dropped, so those trials remain available for the other five outputs. Dataset-wide: 12.0% incorrect, 74.9% correct, 13.1% ignore.

ii.
```python
    outcome = np.full(valid_idx.size, 2, dtype=np.int8)
    outcome[miss] = 0
    outcome[hit] = 1
    outcome[no] = 2
...
            ["incorrect", "correct", "ignore"],
```

iii. Step 5 Key Decision 3: *"Keep `hit`, `miss`, and `no` trials so the required `outcome` output can include correct / incorrect / ignore."* Ordering follows the Decoder Task spec (`incorrect, correct, ignore`).

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj{1}` — the **side camera only** — feature `tongue`. From it the AI takes `ts[:, 0:2, featIdx]` (x and y per frame; the likelihood channel `ts[:,2,:]` is not read, since the authors already set x/y to NaN where tracking failed), `frameTimes`, `featNames` (to locate the feature index) and `NdroppedFrames` (to skip bad trials). The bottom camera's `top_tongue`/`bottom_tongue` views are **not** used. `obj.sglx.bitcode.bitstart`, `obj.sglx.fs` and `obj.bp.ev.bitStart` are also needed, to put frames on the behaviour clock, plus `bp.ev.goCue`.

ii.
```python
    tongue_x, tongue_y, tongue_visible = extract_feature_traces(obj, "tongue", 0, go_cue)
```

```python
def find_feat_index(view: dict[str, list[Any]], feat_name: str) -> int:
    feat_names_all = view.get("featNames", [])
    for names in feat_names_all:
        if not names: continue
        lowered = [str(name).lower() for name in names]
        if feat_name.lower() in lowered:
            return lowered.index(feat_name.lower())
    raise KeyError(f"Could not find feature {feat_name}")
```

```python
        ts = np.asarray(view["ts"][trix], dtype=np.float64)
        if ts.ndim != 3 or feat_idx >= ts.shape[2]:
            continue
        feat_xy = ts[:, :2, feat_idx]
```

iii. Step 5 maps *"Raw DLC tongue trajectories (`obj.traj`)"* → `output[3]` citing `findPosition`, `findVelocity`, `getKinematicsFromVideo`, with *"Visibility comes from raw NaNs before tongue-specific filling; output must explicitly preserve 'not visible'."* The notes do not explain why only the side view was used; `params.traj_features` in the reference lists `tongue` on view 1 and `top_tongue` on view 2 as separate features rather than as views to be pooled.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. A port of `findPosition.m` + `findVelocity.m`:
1. Trials whose `NdroppedFrames` is NaN are skipped entirely (left as NaN → not visible).
2. Frame times are put on the go-cue clock (see 7-d) and x, y are **linearly interpolated onto the 1,000 bin centres**. No smoothing is applied to the tongue (`findPosition` only smooths non-tongue features, and with `N=1`, i.e. not at all).
3. A visibility mask is taken as "the interpolated x and y are both finite".
4. Velocity is `np.gradient` of the interpolated x and y with unit spacing (i.e. pixels per 5 ms bin, matching MATLAB's `gradient(tsinterp(:,1))`); no baseline-derivative subtraction and no nearest-fill for the tongue; residual non-finite velocities are set to 0, as in `findVelocity.m`.
5. Speed is `hypot(xvel, yvel)`. No cross-camera normalisation is needed because only one camera is used.

**Important implementation detail:** `interp_feature` drops the NaN samples *before* calling `np.interp`, so the interpolation draws a straight line across every interval in which the tongue was untracked, instead of propagating NaN the way MATLAB's `interp1` does. Checked directly on trial 6 of `JEB13_2022-09-13`: only 2.9% of in-window frames actually have a tracked tongue, MATLAB-style interpolation marks 2.7% of bins visible, but the AI's marks 11.7% visible. Dataset-wide the AI reports the tongue as visible in 34.3% of bins, versus 12.5% in the human reference.

ii.
```python
def interp_feature(frame_times, values, taxis) -> np.ndarray:
    out = np.full((taxis.size, values.shape[1]), np.nan, dtype=np.float64)
    for col in range(values.shape[1]):
        valid = np.isfinite(frame_times) & np.isfinite(values[:, col])
        if valid.sum() < 2:
            continue
        out[:, col] = np.interp(taxis, frame_times[valid], values[valid, col],
                                left=np.nan, right=np.nan)
    return out
```

```python
        interp_xy = interp_feature(aligned_times, feat_xy, taxis)
        visible[:, trix] = np.isfinite(interp_xy[:, 0]) & np.isfinite(interp_xy[:, 1])
        xpos[:, trix] = interp_xy[:, 0]
        ypos[:, trix] = interp_xy[:, 1]

        if "tongue" not in feat_name.lower():
            xpos[:, trix] = nearest_fill(xpos[:, trix])
            ypos[:, trix] = nearest_fill(ypos[:, trix])
```

```python
        xv = np.gradient(trial_xy[:, 0]); yv = np.gradient(trial_xy[:, 1])
        if "tongue" not in feat_name.lower():
            xv = xv - basederiv[0]; yv = yv - basederiv[0]
            xv = nearest_fill(xv); yv = nearest_fill(yv)
        else:
            xv[~np.isfinite(xv)] = 0.0
            yv[~np.isfinite(yv)] = 0.0
...
    tongue_speed = np.sqrt(tongue_vx**2 + tongue_vy**2)
```

iii. Step 1: *"`findVelocity`: Compute per-frame gradients for x/y position; for tongue features, missing values are set to zero velocity, while non-tongue features are nearest-filled."* Step 4: *"Treat 'tongue not visible' explicitly in decoder outputs; use the raw visibility/missingness rather than pretending tongue velocity is observed during invisibility."* Step 10 lists the tongue as an explicitly preserved missingness case. The gap-bridging behaviour of `interp_feature` is not mentioned anywhere in the notes, and is inconsistent with that stated intent; the Step 7 plot review reads the high missing fraction as *"expected high missingness"* without quantifying it against the raw tracking.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per session, in `compute_speed_categories`: the 50th percentile is taken over the speed values at all (trial, bin) positions that are marked visible and finite; bins at or above that threshold get class 1 (`high`), bins below get class 0 (`low`), and every bin not marked visible gets class 2 (`not_visible`). The threshold is a single session-wide scalar (stored in `session_info['tongue_threshold']`), so it is pooled across trials and bins within a session, as the task requires. Dataset-wide the classes come out at 0.172 / 0.172 / 0.657 (the human reference gets 0.062 / 0.062 / 0.875, the difference being driven entirely by the inflated visibility described in 7-b).

ii.
```python
def compute_speed_categories(speed, visible) -> tuple[np.ndarray, float]:
    out = np.full(speed.shape, 2, dtype=np.int8)
    valid_speed = speed[visible & np.isfinite(speed)]
    if valid_speed.size == 0:
        return out, float("nan")
    threshold = float(np.nanpercentile(valid_speed, 50.0))
    out[visible] = (speed[visible] >= threshold).astype(np.int8)
    return out, threshold
```

iii. Step 5 Key Decision 10: *"Threshold continuous movement outputs by per-session medians on valid samples: This follows the task specification exactly and avoids cross-session scale differences."* Key Decision 9: *"Preserve 'not visible' / 'no video' explicitly as class 2: This is required by the task and is preferable to silently imputing those outputs into low/high velocity classes."*

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The camera clock leads the behaviour clock, so a session-constant offset is removed first, exactly as in `findVideoOffset.m`: `vidshift = bitcode.bitstart / sglx.fs − bp.ev.bitStart`, except that the AI reduces both per-trial vectors with `np.nanmedian` rather than MATLAB's `mode`. Each frame's time on the go-cue axis is then `frameTimes − vidshift − goCue[trial]`, and x/y are interpolated at the 1,000 bin centres of the same grid the spikes were binned into, so bin *k* means the same 5 ms window in both streams. Frames outside the window contribute nothing because `np.interp` is called with `left=np.nan, right=np.nan`. If the session has no `sglx` struct the offset falls back to 0.

ii.
```python
def get_vidshift(obj: dict[str, Any]) -> float:
    bp = obj["bp"]
    bit_start = get_bp_array(bp["ev"], "bitStart", int(bp["Ntrials"]), dtype=np.float64)
    if "sglx" not in obj or not isinstance(obj["sglx"], dict):
        return 0.0
    ...
    bit_file_offset = as_1d_numeric(bitcode.get("bitstart"), dtype=np.float64)
    fs_val = float(np.asarray(fs).reshape(-1)[0])
    return float(np.nanmedian(bit_file_offset) / fs_val - np.nanmedian(bit_start))
```

```python
        aligned_times = frame_times - vidshift - go_cue[trix]
        interp_xy = interp_feature(aligned_times, feat_xy, taxis)
```

iii. Step 10 Check 3: *"Reference: `alignSpikes` subtracts `bp.ev.goCue` per trial; `findVideoOffset` aligns video to neural start time. Converter: same go-cue subtraction and same `vidshift` formula."* The mode→median substitution is not called out in the notes; I checked it on `JEB13_2022-09-13` and the two reductions give the identical value (0.49002784 s) because both underlying vectors are essentially constant across trials.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The DeepLabCut tracking in `obj.traj{2}` — the **bottom camera only**, which is the only view that tracks paws — for **both** paw features, `top_paw` and `bottom_paw`. The same per-trial fields are read as for the tongue (`ts[:, :2, idx]`, `frameTimes`, `featNames`, `NdroppedFrames`), plus the bitcode fields for the clock correction. If a feature name is absent from the view it is silently skipped, and if neither is found the whole output becomes class 2.

ii.
```python
    paw_features = []
    paw_visible_masks = []
    for feat_name in ("top_paw", "bottom_paw"):
        try:
            paw_x, paw_y, paw_visible = extract_feature_traces(obj, feat_name, 1, go_cue)
        except KeyError:
            continue
```

iii. Step 3 notes *"paws only in bottom view"*, and Step 5 maps *"Raw DLC paw trajectories (`obj.traj`)"* → `output[4]` with *"Build aligned paw speed trace from paw-tracked bottom-view points."* The notes do not give a rationale for pooling the two paw markers rather than choosing one; the reference `params.traj_features` lists `top_paw` and `bottom_paw` as two separate features of view 2.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The same `findPosition`/`findVelocity` port as the tongue, but taking the non-tongue branch at every fork:
1. Positions are linearly interpolated onto the 1,000 bin centres; the visibility mask is captured **before** filling.
2. Because the feature name does not contain "tongue", the interpolated x and y are then nearest-filled (`nearest_fill`, the port of MATLAB `fillmissing(...,'nearest')`).
3. Velocity is `np.gradient` of the filled positions, minus a per-trial baseline derivative `basederiv = nanmedian(diff(xy))`. The port faithfully reproduces the reference's own quirk of subtracting `basederiv(1)` — the *x* baseline — from **both** the x and y velocities. Velocities are then nearest-filled as well.
4. Speed is `hypot(xvel, yvel)` per paw marker, and the two markers are combined bin-by-bin by averaging over whichever markers are visible there (a marker that is invisible contributes nothing rather than a zero); a bin is visible if either marker is.

No cross-view normalisation is applied (there is only one camera), so the values stay in pixels per bin.

ii.
```python
        basederiv = np.array([
            np.nanmedian(diffs[:, 0]) if np.any(np.isfinite(diffs[:, 0])) else 0.0,
            np.nanmedian(diffs[:, 1]) if np.any(np.isfinite(diffs[:, 1])) else 0.0,
        ], dtype=np.float64)
        xv = np.gradient(trial_xy[:, 0]); yv = np.gradient(trial_xy[:, 1])
        if "tongue" not in feat_name.lower():
            xv = xv - basederiv[0]
            yv = yv - basederiv[0]          # reference findVelocity.m also uses basederiv(1) here
            xv = nearest_fill(xv); yv = nearest_fill(yv)
```

```python
        paw_stack = np.stack(paw_features, axis=2)
        paw_vis_stack = np.stack(paw_visible_masks, axis=2)
        paw_visible = paw_vis_stack.any(axis=2)
        paw_weighted = np.where(paw_vis_stack, paw_stack, 0.0)
        paw_counts = paw_vis_stack.sum(axis=2)
        paw_speed = np.divide(paw_weighted.sum(axis=2), paw_counts,
                              out=np.full(paw_visible.shape, np.nan, dtype=np.float64),
                              where=paw_counts > 0)
```

iii. Step 1 records the reference behaviour being ported (`findVelocity`: nearest-fill for non-tongue features). Step 7 documents the bug fix that produced the current visibility logic: *"Paw not-visible class was initially lost because nearest-filled trajectories were being used to infer visibility; fixed by preserving a pre-fill visibility mask and only using reference-style filled traces for velocity computation."* Step 7 also records replacing `nanmean` with explicit sum/count logic to silence an empty-slice warning when averaging the two markers.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Identically to the tongue: one session-wide 50th percentile computed over all visible, finite speed samples, class 1 at or above it, class 0 below, class 2 wherever the pre-fill visibility mask is false. Because the reference-style nearest-fill precedes the velocity computation and because `interp_feature` bridges tracking gaps (see 7-b), class 2 ends up covering only the edges of trials where the camera record does not span the window: 0.042 of all bins, versus 0.186 in the human reference. The remaining bins split 0.479 / 0.479.

ii.
```python
        paw_cat, paw_thresh = compute_speed_categories(paw_speed, paw_visible)
```
(same `compute_speed_categories` shown in 7-c)

iii. Same justification as 7-c — Step 5 Key Decisions 9 and 10 (per-session median split; explicit class 2 for missing).

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Exactly as for the tongue: the session-constant `vidshift` is subtracted from `frameTimes`, then the trial's `goCue`, and the result is interpolated at the shared 1,000 bin centres. Because `extract_feature_traces` is called with `view_idx=1`, the frame times used are the bottom camera's own, not the side camera's — which matters because the two views can record different numbers of frames on a given trial.

ii.
```python
            paw_x, paw_y, paw_visible = extract_feature_traces(obj, feat_name, 1, go_cue)
...
        frame_times = view.get("frameTimes", [None] * ntrials)[trix]
        ...
        aligned_times = frame_times - vidshift - go_cue[trix]
        interp_xy = interp_feature(aligned_times, feat_xy, taxis)
```

iii. Same as 7-d: Step 10 Check 3 records the `vidshift` + go-cue subtraction as matching `findVideoOffset` / `findPosition`.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<ANM>_<DATE>.mat` file sitting beside the session's data structure, which holds one trace per trial with one value per camera frame. The copy sometimes stored inside `obj.me` is not used. The loader accepts all three layouts found across the 44 files — a bare cell array, a struct `{data, moveThresh}`, and a doubly-wrapped `{data: {data, moveThresh}}` — and returns all-NaN (→ class 2 everywhere) if no file exists. The side camera's `frameTimes` (`obj.traj{1}`) supply the time base, and `bp.ev.goCue` plus the bitcode fields supply the alignment.

ii.
```python
def load_motion_energy(spec: SessionSpec) -> dict[str, Any] | None:
    if not spec.motion_path.exists():
        return None
    payload = load_any_mat(spec.motion_path)
    me = payload.get("me")
    if me is None: return None
    if isinstance(me, dict): return me
    if isinstance(me, list): return {"data": me, "moveThresh": np.nan}
    if isinstance(me, np.ndarray): return {"data": me.tolist(), "moveThresh": np.nan}
    raise TypeError(...)
```

```python
    me_data = me["data"]
    if isinstance(me_data, dict) and "data" in me_data:
        me_data = me_data["data"]
```

iii. Step 1 documents `loadMotionEnergy.m`. Step 10 records the two container fixes: *"top-level MATLAB v5 dict recursion bug for motion-energy files: fixed"* and *"randomized-delay motion-energy files stored as raw cell arrays instead of structs: fixed."* The reference MATLAB has the same single unwrap guard (`if isstruct(me.data), me.data = me.data.data; end`).

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond alignment and interpolation. The value is already one scalar per frame (the paper computes a per-pixel temporal-median difference over ±5 frames and reduces each frame to its 99th percentile across pixels), so the AI only linearly interpolates the trace onto the 1,000 bin centres, leaving NaN outside the frames' coverage. Deliberately, the reference's final `fillmissing(me.data,'nearest')` is **not** ported, so bins with no video coverage stay NaN and become the `no_video` class. Trials whose trace is empty, or which have fewer than 2 finite samples, stay all-NaN.

ii.
```python
        valid = np.isfinite(frame_times) & np.isfinite(trace)
        if valid.sum() < 2:
            continue
        aligned[:, trix] = np.interp(
            TIME_CENTERS,
            frame_times[valid] - vidshift - go_cue[trix],
            trace[valid],
            left=np.nan,
            right=np.nan,
        )
```

iii. Step 5 Key Decision 9: *"Preserve 'not visible' / 'no video' explicitly as class 2: This is required by the task and is preferable to silently imputing those outputs into low/high velocity classes."* Step 4 also notes that the original per-session manual `moveThresh` is superseded here: *"Use provided per-session motion-energy values directly and create decoder classes from per-session medians as required by the task, while retaining note that original move/not-move threshold was manual."*

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The same `compute_speed_categories` used for the two kinematic streams: one session-wide 50th percentile over all finite aligned samples (the "visible" mask here is simply `np.isfinite(motion)`), class 1 at or above, class 0 below, class 2 for NaN bins. Dataset-wide this gives 0.481 / 0.481 / 0.038, closely matching the human reference's 0.479 / 0.483 / 0.038.

ii.
```python
    motion = align_motion_energy(obj, spec, go_cue)
    motion = choose_trials(motion, valid_idx)
    motion_visible = np.isfinite(motion)
    motion_cat, motion_thresh = compute_speed_categories(motion, motion_visible)
```

iii. Step 5 Key Decision 10 (per-session median split, per the task spec). Step 10 Check 2 verified the full 1,000-bin class trace of the first kept trial of `JEB23_2023-10-10` against an independent reconstruction from the raw `.mat` files: *"Result: `np.allclose(...) == True`."*

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same offset and same grid as the tracking streams. Motion energy has one value per frame of the side camera, so the AI takes `obj.traj{1}(trial).frameTimes`, subtracts the session `vidshift` and the trial's `goCue`, and interpolates onto the shared bin centres. If a trial's `frameTimes` is missing or all-NaN, a synthetic 400 Hz time base `(1..N)/400` is substituted (the reference's fallback) before the same shift is applied.

ii.
```python
    vidshift = get_vidshift(obj)
    view0 = obj["traj"][0] if obj["traj"] else {}
    for trix in range(ntrials):
        ...
        frame_times = view0.get("frameTimes", [None] * ntrials)[trix] if view0 else None
        if frame_times is None:
            frame_times = np.arange(1, trace.size + 1, dtype=np.float64) / 400.0
        else:
            frame_times = np.asarray(frame_times, dtype=np.float64).reshape(-1)
            if frame_times.size == 1 and np.isnan(frame_times[0]):
                frame_times = np.arange(1, trace.size + 1, dtype=np.float64) / 400.0
        ...
        aligned[:, trix] = np.interp(TIME_CENTERS, frame_times[valid] - vidshift - go_cue[trix],
                                     trace[valid], left=np.nan, right=np.nan)
```

iii. Step 1: *"Motion energy is aligned to the same time axis as neural data by interpolating each trial to `obj.time + params.advance_movement` after compensating for the video offset"* (the reference's `advance_movement` is 0 in `getDefaultParams.m`, so it is omitted). Step 10 Check 2 documents the raw-data verification of this alignment.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several distinct cases, handled by keeping the trial and marking the gap rather than dropping it:
- **Two MATLAB container formats** (v7.3 vs v5) and three motion-energy layouts: normalised by `load_any_mat` / `normalize_obj` / `load_motion_energy` into one representation. Two loader bugs here (top-level v5 dict recursion; bare-cell-array motion energy) were found and fixed during Steps 7 and 9.
- **Non-standard `clu` layout** (e.g. `JEB6_2021-04-18`, probe 1 empty): `normalize_probe` returns `{}` and `process_probe`/`max_recorded_trial` skip empty probes.
- **Missing `bp` fields**: `get_bp_array` returns all-NaN (or all-False for boolean fields) instead of raising; a missing `stim` struct is treated as no photostim.
- **Missing per-trial event arrays**: `ensure_event_list` converts `None`/scalar-NaN entries to empty arrays, so a trial with no recorded licks becomes `none` rather than an error.
- **Bad/absent video for a trial**: trials with NaN `NdroppedFrames` are skipped; trials with missing or all-NaN `frameTimes` get a synthetic 400 Hz time base (the reference's own fallback); trials with fewer than 2 finite samples are left NaN. All of these surface as class 2.
- **Missing motion-energy file**: the whole session's motion output becomes class 2.
- **Behaviour running past the end of the recording**: trials beyond the last spike-bearing trial are dropped (see 1-e), instead of entering the dataset as all-zero neural trials.
- **Sessions too small**: fewer than 2 valid trials or fewer than 10 kept units raises rather than silently emitting a degenerate session.

Two places where missing data is filled rather than flagged: `nearest_fill` on non-tongue positions and velocities (a faithful port of `fillmissing(...,'nearest')`), and — not documented by the AI — `interp_feature`, which drops NaN samples before interpolating and therefore draws a straight line across every untracked interval, for the tongue as well as the paw (see 7-b). One robustness gap: `get_bp_array` handles a *missing* field but calls `np.broadcast_to(arr, (ntrials,))` for a field whose length merely disagrees with `Ntrials`, which raises for any length > 1 (the reference truncates instead). I verified that none of the fields the script reads is mis-sized in any of the 44 sessions, so this path is never taken here.

ii.
```python
        ndropped = view.get("NdroppedFrames", [np.nan] * ntrials)[trix]
        ndropped_arr = np.asarray(ndropped, dtype=np.float64)
        if ndropped_arr.ndim == 0 and np.isnan(ndropped_arr):
            continue
```

```python
        frame_times = view.get("frameTimes", [None] * ntrials)[trix]
        if frame_times is None:
            frame_times = np.arange(1, feat_xy.shape[0] + 1, dtype=np.float64) / 400.0
        else:
            frame_times = np.asarray(frame_times, dtype=np.float64).reshape(-1)
            if frame_times.size == 1 and np.isnan(frame_times[0]):
                frame_times = np.arange(1, feat_xy.shape[0] + 1, dtype=np.float64) / 400.0
```

```python
def nearest_fill(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(x))
    if valid.size == 0:
        return x.copy()
    fn = interp1d(valid, x[valid], kind="nearest", bounds_error=False,
                  fill_value=(x[valid[0]], x[valid[-1]]), assume_sorted=True)
    return fn(np.arange(x.size))
```

iii. Step 5 Key Decision 9 covers the general policy: *"Preserve 'not visible' / 'no video' explicitly as class 2 … preferable to silently imputing those outputs into low/high velocity classes."* Step 10's "Issues Found and Resolved" list documents all four data-shape bugs and their fixes, and the edge-case review states *"paw visibility now uses the raw pre-fill visibility mask so class 2 reflects real invisibility; tongue missingness remains explicit and is not silently converted into a low/high observed-velocity class"* — a claim that the gap-bridging interpolation in fact undercuts.

## 11-a. What are the most time-consuming steps of the code?

i. Reading and normalising the `.mat` files dominates. The full conversion took 214.6 s for 44 sessions (1.6–8.9 s per session); the v7.3/HDF5 fixed-delay files, read through `mat73.loadmat` plus the recursive `normalize_obj`/`normalize_view` pass that converts every per-trial `ts` array into Python lists, are consistently the slowest (6–9 s), while the small v5 randomized-delay files finish in 1.6–3 s. After loading, the per-cluster work — one `np.bincount` plus one `my_smooth` over a (1000 × n_trials) array, which is itself a Python loop over trials calling `np.convolve` — is the largest compute cost, scaling with unit count (the 134- and 142-unit sessions are the slow ones in their cohort). The camera path (three `extract_feature_traces` passes, each a Python loop over all trials) is next. The AI printed per-session timing but not per-step timing, so bottleneck attribution in the notes is at session granularity only.

ii.
```python
def load_any_mat(path: Path) -> dict[str, Any]:
    if h5py.is_hdf5(path):
        return mat73.loadmat(str(path))
    return convert_mat_struct(loadmat(path, squeeze_me=True, struct_as_record=False))
```

```python
        print(f"    kept {session_data['session_info']['n_neurons_kept']} units, "
              f"{session_data['session_info']['n_trials_kept']} trials "
              f"in {time.time() - session_start:.1f}s")
```

iii. Step 7 Run Time Estimates: *"Full conversion estimate … approximately 4-6 minutes for 44 sessions"* (actual 214.6 s, inside the 15-minute budget set by the instructions), with per-session timings tabulated for the two sample sessions (8.4 s fixed-delay, 3.2 s randomized-delay).

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI vectorised the one loop that mattered most — spike binning — replacing a per-trial/per-spike histogram with a single `np.bincount` over flattened `(bin, trial)` indices. Loops that remain and could be vectorised:
- `my_smooth`'s `for col in range(x_filt.shape[1])` calling `np.convolve` per trial; this could be a single `scipy.ndimage.convolve1d` or FFT convolution along one axis, and it runs once per cluster (2,364+ times over the dataset, each over ~300 columns).
- `build_session_neural`'s double loop, which copies `rates[:, trix]` neuron by neuron into a fresh `(n_neurons, 1000)` array — about 600,000 Python-level iterations across the dataset, replaceable by one `np.stack(kept_rates, axis=...).transpose(...)` followed by slicing.
- `extract_feature_traces` and `align_motion_energy` loop over all `Ntrials` (including trials that will be discarded — see 11-d) and `interp_feature` loops over the two coordinate columns; the per-trial loop is hard to remove because frame counts differ per trial, but the column loop is trivially unrollable.
- `extract_velocity` loops over trials to call `np.gradient` and `nanmedian`, both of which accept an `axis` argument and would work on the whole `(1000, n_trials)` matrix at once (the only per-trial piece is `nearest_fill`).
- `process_probe` loops over clusters; the binning and smoothing of all clusters of a probe could be done as one 3-D operation.

ii.
```python
    out = np.empty_like(x_filt, dtype=np.float64)
    for col in range(x_filt.shape[1]):
        out[:, col] = np.convolve(x_filt[:, col], kern, mode="same")
```

```python
    for trix in range(ntrials):
        trial_mat = np.empty((nneurons, TIME_CENTERS.size), dtype=np.float32)
        for neuron_idx, rates in enumerate(kept_rates):
            trial_mat[neuron_idx, :] = rates[:, trix]
        session_trials.append(trial_mat)
```

```python
    counts_flat = np.bincount(flat_idx, minlength=TIME_CENTERS.size * n_valid)   # the one that was vectorised
```

iii. Step 6: *"A naive cluster-by-trial histogram loop would have been too slow and would not scale to full-dataset conversion … Spike binning is vectorized with `np.bincount` over flattened `(time_bin, trial)` indices instead of nested Python loops over spikes/trials."* The remaining loops are not identified in the notes; since the total runtime (215 s) was already well under the 15-minute budget, no further optimisation was pursued.

## 11-c. What processing does the code repeat multiple times?

i. Little, but not nothing:
- `get_vidshift(obj)` is recomputed from scratch on each of the four video passes per session (tongue, `top_paw`, `bottom_paw`, motion energy) instead of being computed once and passed down. It is cheap (two `nanmedian` calls) but it is literally the same constant each time.
- `find_feat_index` rescans every trial's `featNames` list on each pass, and `normalize_view` has already rebuilt those lists once per session.
- `np.isfinite(xpos) & np.isfinite(ypos)` is recomputed inside `extract_velocity` as a default, although in every actual call the mask produced by `extract_feature_traces` is passed in and the default is dead.
- The per-cluster rate matrix is computed once and then *reused* for both the low-FR condition PSTHs and the final output — an explicitly avoided repeat.
- The bin grid (`EDGES`, `TIME_CENTERS`) is built once at module level and shared by every trial, session and stream.

ii.
```python
    vidshift = get_vidshift(obj)          # in extract_feature_traces, called once per feature
...
    vidshift = get_vidshift(obj)          # and again in align_motion_energy
```

```python
        rates = bin_cluster_rates(trial_ids, trial_times, go_cue, valid_mask)
        psth_stack = []
        for cond_pos in condition_positions:
            ...
            psth_stack.append(rates[:, cond_pos].mean(axis=1))   # reuses `rates`, no second binning pass
```

iii. Step 6 Code speedups: *"Firing-rate filtering reuses already computed single-trial neural matrices to build condition PSTHs instead of re-binning spikes a second time"* and *"Trial-level motion / kinematic alignment is done once per session per feature and then reused for discretization and plotting."* The repeated `get_vidshift` and `find_feat_index` calls are not mentioned.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Four things:
- **The plot payload is always built.** `process_session` constructs `plot_payload` — including `np.stack([trial.mean(axis=0) for trial in session_neural])` over every trial of the session, plus transposes of all three speed matrices — before checking `show_processing`, and the returned payload is discarded by `build_dataset` (`session_data, _ = process_session(...)`). For the full run this work was thrown away for all 44 sessions.
- **Video streams are computed for all `Ntrials`, then subset.** `extract_feature_traces` and `align_motion_energy` interpolate every trial in the session, including early-lick, photostim and post-recording trials, and only afterwards does `choose_trials(matrix, valid_idx)` throw ~8% of them away. Passing `valid_idx` down would skip that work.
- **Positions themselves are never output.** `xpos`/`ypos` (and the nearest-filling applied to them) exist only to feed `np.gradient`; only the speed magnitude reaches the pickle. Likewise `basederiv` is computed for the tongue as well, where the code path that uses it is never taken, and `xvel`/`yvel` are collapsed to `hypot` immediately.
- **Loading materialises unused fields.** `mat73.loadmat` reads the entire `obj` tree and `normalize_obj` converts all of it, so cluster waveforms (`spkWavs`), absolute spike times (`tm`), channel/site metadata, the `sglx` per-trial index arrays, and every tracked feature other than `tongue`, `top_paw` and `bottom_paw` are all built in memory and never read. This is the largest single piece of wasted work, and it is the direct cause of the loading cost in 11-a.

ii.
```python
    plot_payload = {
        "session_id": spec.session_id,
        ...
        "neural_mean": np.stack([trial.mean(axis=0) for trial in session_neural], axis=0),
        "tongue_speed": tongue_speed.T,
        ...
    }
    if show_processing:
        plot_processing(plot_payload)
```

```python
        session_data, _ = process_session(spec, show_processing=show_processing and sess_idx <= 2)
```

```python
    for trix in range(ntrials):        # all trials, not just valid_idx
        ...
    return xpos, ypos, visible
...
    tongue_x = choose_trials(tongue_x, valid_idx)
```

iii. Not discussed in CONVERSION_NOTES; Step 6 only claims the positive speedups (vectorised binning, PSTH reuse, single-pass interpolation). None of these items was a practical problem, since the whole conversion finished in 214.6 s against a 15-minute budget.
