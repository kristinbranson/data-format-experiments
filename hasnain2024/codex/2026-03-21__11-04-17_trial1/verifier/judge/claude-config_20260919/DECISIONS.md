# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not glob the data folders. It re-derives the analysed session list *programmatically* by parsing the authors' own loader scripts `code/DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m`: it walks each file line by line, skips comment lines (`%`), and regex-extracts `anm = '...'`, `date = '...'` and `probe = [..]` / `probe = N` triples. Each complete triple is kept only if a matching `data_structure_<anm>_<date>.mat` exists under `data/*/`, so commented-out sessions and the JEB4/JEB5 sessions absent from `data/` are excluded. This yields exactly 44 sessions (25 fixed-delay + 19 randomized-delay), with the probe number(s) each loader specifies (JEB15 sessions use two probes, which are concatenated).
Each session file is then read with one of two readers chosen by probing the file with `h5py`: a *targeted* HDF5 reader (`load_session_hdf5`, reads only `bp`, the selected `clu` probe slots, the two `traj` views restricted to the needed features, and `sglx`) or a `scipy.io.loadmat` reader for the older v5 files (`load_session_mat`). Motion energy is read from the sibling `motionEnergy_<anm>_<date>.mat` file.

ii.
```python
def parse_reference_session_specs(code_dir: Path, data_dir: Path) -> list[SessionSpec]:
    data_files = find_data_files(data_dir)
    specs: list[SessionSpec] = []
    loader_dir = code_dir / "DataLoadingScripts" / "Recording and video"
    for loader in sorted(loader_dir.glob("load*_ALMVideo.m")):
        ...
        for line in loader.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            m = re.search(r"anm = '([^']+)'", s)
            ...
            m = re.search(r"probe = \[([0-9 ]+)\]", s)
            ...
            if {"subject", "date", "probes"} <= current.keys():
                key = (current["subject"], current["date"])
                if key in data_files:
                    ...
                    specs.append(SessionSpec(...))
```

```python
def is_hdf5_mat(path: Path) -> bool:
    try:
        with h5py.File(path, "r"):
            return True
    except OSError:
        return False

def load_session(spec: SessionSpec) -> dict:
    return load_session_hdf5(spec) if is_hdf5_mat(spec.session_path) else load_session_mat(spec)
```

iii. From CONVERSION_NOTES Step 4/5: "Loader scripts list 50 ALM sessions total, but 6 of those are JEB4/JEB5 sessions not present in this project's `data/`… Use the intersection of reference-code session lists and available data files. This yields 44 code-selected, paper-consistent ephys sessions", matching the paper's "25 fixed-delay + 19 randomized-delay = 44". Dual readers are needed because "`data/` mixes HDF5/v7.3 and older MATLAB files". Targeted HDF5 field loading was introduced as a speed-up because "full recursive HDF5-to-Python loading was too slow and memory-heavy".

## 1-b. How are the data split into subjects?

i. The subject is the animal id parsed out of the loader script (`anm = '...'`), which is also the first field of the file name; it is stored on `SessionSpec.subject` and carried through each session's result. At assembly, `subjects` is built in order of first appearance across the processed sessions and `subject_idx` indexes into it. Result: 14 subjects over 44 sessions (EKH1 1, EKH3 1, JEB6 1, JEB7 2, JEB11 2, JEB12 2, JEB13 5, JEB14 4, JEB15 4, JEB19 4, JEB23 7, JEB24 8, JGR2 2, JGR3 1).

ii.
```python
@dataclass(frozen=True)
class SessionSpec:
    subject: str
    date: str
    ...
    @property
    def session_id(self) -> str:
        return f"{self.subject}_{self.date}"
```
```python
if session["subject"] not in subject_names:
    subject_names.append(session["subject"])
subject_index.append(subject_names.index(session["subject"]))
...
"subjects": subject_names,
"subject_idx": np.asarray(subject_index, dtype=np.int64),
```

iii. CONVERSION_NOTES Step 5 mapping table: "Animal IDs from filenames / loader scripts → `subjects`, `subject_idx`… Expected 14 unique neural subjects in the available dataset." The AI noted separately that `obj.meta` is not present in every session, so the filename/loader id is the reliable source.

## 1-c. How are the data split into sessions?

i. One session = one `(anm, date)` pair from the loader scripts = one `data_structure_*.mat` file = one element of `neural`/`input`/`output`. `find_data_files` scans both `data/Ephys_Behavior/` and `data/RandomizedDelay_Ephys_Behavior/` (and the two behaviour-only folders, which never match because they contain no `obj.clu`-bearing sessions in the loader lists), so fixed-delay and randomized-delay sessions are handled uniformly in one list. Sessions present in `data/` but not referenced by the loaders (JEB23 2023-10-20, JEB24 2023-10-03/10-04) are excluded. A session is additionally dropped if it ends up with fewer than 10 units or fewer than 2 usable trials (neither triggered: all 44 sessions were kept).

ii.
```python
def find_data_files(data_dir: Path) -> dict[tuple[str, str], Path]:
    out = {}
    for path in sorted(data_dir.glob("*/*.mat")):
        if not path.name.startswith("data_structure_"):
            continue
        parts = path.stem.split("_")
        out[(parts[2], parts[3])] = path
    return out
```
```python
if selected_trials.size < 2:
    log(f"SKIP {spec.session_id}: only {selected_trials.size} valid hit/miss non-stim non-early trials")
    return None
...
if kept_units < 10:
    log(f"SKIP {spec.session_id}: only {kept_units} units after quality/FR filtering")
    return None
```

iii. Step 4 discrepancy table: exclude "raw sessions not referenced by the loader scripts". The ≥10-unit rule comes from the methods text the AI transcribed in Step 3: "Recording sessions included only if they contained at least 10 units." The ≥2-trial rule is from the target-format requirement that "there needs to be at least two trials within each session in order to evaluate the decoder performance".

## 1-d. How are the data split into trials?

i. Trials are the rows of the per-trial `obj.bp` arrays (`R`, `L`, `hit`, `miss`, `no`, `early`, `autowater`, `stim.enable`) and `obj.bp.ev.goCue`, one go cue per trial; the video `obj.traj{view}` entries and the motion-energy cell array are indexed by the same trial number, and spikes carry their trial in `clu.trial` (1-based). No trial boundaries are reconstructed. The AI reads the full length of each `bp` field rather than truncating to `bp.Ntrials` (I verified that in all 120 data files every one of these fields has exactly `Ntrials` entries, so the two are equivalent here).

ii.
```python
"R": np.asarray(bp.R, dtype=np.float64).reshape(-1),
"L": np.asarray(bp.L, dtype=np.float64).reshape(-1),
"hit": np.asarray(bp.hit, dtype=np.float64).reshape(-1),
...
"events": {"bitStart": ..., "sample": ..., "delay": ...,
           "goCue": np.asarray(bp.ev.goCue, dtype=np.float64).reshape(-1), ...},
```
```python
trial_to_pos = np.full(raw["R"].size, -1, dtype=np.int64)
trial_to_pos[selected_trials] = np.arange(selected_trials.size, dtype=np.int64)
...
aligned = unit["trialtm"] - go_cue[unit["trial"] - 1]
trial_pos = trial_to_pos[unit["trial"] - 1]
```

iii. Step 2 notes: "Behavioral trial fields in `obj.bp` include `Ntrials`, `hit`, `miss`, `no`, `early`, `autowater`, `R`, `L`, `stim`, and event timings in `obj.bp.ev`." The Bpod table defines the trials directly; the AI's Step 10 sanity checks confirmed per-trial label round-trips from the raw files.

## 1-e. How are trials filtered based on quality controls?

i. Four filters, all before any computation:
1. `stim.enable == 0` — photostimulation trials dropped.
2. `early == 0` — early-lick trials dropped.
3. `hit == 1 | miss == 1` — **ignore / no-response trials dropped**, and `R == 1 | L == 1`.
4. Trials whose 1-based index exceeds the last trial in which any quality-passing unit fired are dropped ("behavioural trials continued after the last neural `unit.trial` index" in two JEB24 sessions, which otherwise produced all-zero neural trials).

Result: 11,955 trials of ~15.1k (mean 271.7/session, min 137, max 472). The reference keeps ignore trials as a third outcome/lick class and so retains 13,762.

ii.
```python
def session_valid_trial_mask(raw: dict) -> np.ndarray:
    return (
        (raw["stim_enable"] == 0)
        & (raw["early"] == 0)
        & ((raw["hit"] == 1) | (raw["miss"] == 1))
        & ((raw["R"] == 1) | (raw["L"] == 1))
    )
```
```python
covered_trial_max = [
    int(np.nanmax(unit["trial"]))
    for unit in raw["units"]
    if good_quality(unit["quality"]) and np.asarray(unit["trial"]).size
]
if covered_trial_max:
    max_neural_trial = min(raw["R"].size, max(covered_trial_max))
    selected_trials = selected_trials[selected_trials + 1 <= max_neural_trial]
```

iii. Step 5 Key Decision 3: "Exclude stimulation, early-lick, and ignore/no-response trials: Reference analyses consistently use `~stim.enable`, `~early`, and usually hit/miss conditions; ignore trials are omitted in the paper." (The paper's methods indeed state early licks and ignore trials are omitted, and `getDefaultParams.m` conditions are all `&~stim.enable&~early`.) Step 5 mapping table: "Keep only non-early hit/miss trials so direction is well defined"; "Exclude `no` and `early` trials rather than inventing a label for ignores". The recording-length cut is documented in Step 9/10 as a fix for the `verification_full_out.txt` all-zero-trial warning in `JEB24_2023-10-23` and `JEB24_2023-11-03`.
Note: the decoder-output spec the agent was actually handed (visible in `/logs/agent/trajectory.json`, step 2) listed **two** classes for lick direction and outcome, with no `none`/`ignore` class, which is what motivated dropping ignore trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` for the probe(s) named by the session's `load<ANM>_ALMVideo.m` entry — per-unit `trialtm` (spike time relative to trial start, behaviour clock), `trial` (1-based trial of each spike), and `quality` (manual curation string). The alignment input is `obj.bp.ev.goCue`. Non-selected probes (e.g. M1TJ) are never read. Empty HDF5 probe slots are indexed correctly (`JEB6_2021-04-18` has an empty slot 1).

ii.
```python
for probe_idx in spec.probes:
    probe = f[probe_refs[probe_idx - 1]]
    if not isinstance(probe, h5py.Group):
        raise IndexError(f"{spec.session_id}: probe slot {probe_idx} is empty in HDF5 file")
    for i in range(probe["trialtm"].shape[0]):
        units.append({
            "quality": h5_read_string(f, probe["quality"][i, 0]),
            "trialtm": np.asarray(h5_read_numeric(f, probe["trialtm"][i, 0]), ...),
            "trial":   np.asarray(h5_read_numeric(f, probe["trial"][i, 0]), dtype=np.int64)...,
        })
```

iii. Step 4: "Raw files contain all clusters from all probes, including non-ALM probes and empty probe slots… Use only the code-specified probe for each session, treat that probe as ALM". Step 1 notes: "Neural data are sorted spike times in `obj.clu{probe}(cluster)` with per-spike session time (`tm`), per-spike within-trial time (`trialtm`), and trial index (`trial`)."

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed into the fixed 5 ms grid with `np.add.at`, accumulating `1/DT` per spike so the matrix is already in spikes/s, then smoothed along time with a **faithful Python port of the authors' `mySmooth.m`**: `gausswin(15, alpha=2.5)` with the first `floor(N/2)` taps zeroed (causal), renormalised to sum 1, applied with `'reflect'` boundary handling (prepend the first N samples, convolve `'same'`, trim N). Output is stored as `float32` in Hz. No normalisation, baseline subtraction or z-scoring. Two-probe sessions are concatenated into one population.

ii.
```python
def matlab_gausswin(n, alpha=2.5):
    k = np.arange(0, n) - (n - 1.0) / 2.0
    return np.exp(-0.5 * (alpha * k / ((n - 1.0) / 2.0)) ** 2)

def my_smooth(x, n, bctype="none"):
    ...
    if bctype.lower() == "reflect":
        pad = x[:n, :]; x_filt = np.concatenate([pad, x], axis=0); trim = n
    ...
    kern = matlab_gausswin(n)
    kern[: n // 2] = 0.0          # causal, as in mySmooth.m
    kern /= kern.sum()
    for j in range(x_filt.shape[1]):
        out[:, j] = np.convolve(x_filt[:, j], kern, mode="same")
    out = out[trim:, :]
```
```python
mat = np.zeros((n_sel, time_edges.size - 1), dtype=np.float64)
bins = np.floor((aligned[keep] - time_edges[0]) / DT).astype(np.int64)
np.add.at(mat, (trial_pos[keep], bins), 1.0 / DT)
mat = my_smooth(mat.T, SMOOTH, BCTYPE).T
```

iii. Step 10 reference-code comparison: "Reference: `mySmooth.m` with a causal Gaussian-like kernel and reflected padding. Converter: Python reimplementation of the same kernel/padding behavior." `getDefaultParams.m` sets `params.smooth = 15` under the comment "smooth with causal gaussian kernel"; `getSeq.m` applies `mySmooth(N./params.dt, params.smooth, params.bctype)`. Step 10 sanity check: independently rebuilding one unit's trace from the raw `.mat` gave `np.allclose = True`, max abs diff 1.4e-5.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, exactly the pair used by the reference MATLAB:
1. The manual `clu.quality` string, stripped and lower-cased, is rejected if it is one of `garbage`, `gabrga`, `noisy`, `real?` — the literal exclusion list of `findClusters(..., {'all'})`. Multiunits and blank labels are kept.
2. Units whose **mean smoothed rate over the whole window and all kept trials** is `<= 1.0 Hz` are dropped.
Result: 2,455 units from the selected probes (mean 55.8/session, min 17, max 142); the paper's summary totals are 1,651 + 845 = 2,496.

ii.
```python
def good_quality(label: str) -> bool:
    label = label.strip().lower()
    return label not in {"garbage", "gabrga", "noisy", "real?"}
```
```python
for unit in raw["units"]:
    if not good_quality(unit["quality"]):
        continue
    unit_mat = compute_unit_trial_matrix(unit, raw["events"]["goCue"], trial_to_pos,
                                         selected_trials.size, time_edges)
    if float(unit_mat.mean()) <= LOW_FR_HZ:      # LOW_FR_HZ = 1.0
        continue
    kept_units += 1
```

iii. Step 4 resolved two conflicts explicitly: `getDefaultParams.m` uses `lowFR = 0.5` but `WorkingWithDataObjs.m` and the Fig. 8 script use `lowFR = 1`, and the methods say "all units with firing rates exceeding 1 Hz were included in all other analyses" → "Resolve in favor of the methods text and figure scripts: use a 1 Hz low-FR threshold". And: "use all non-garbage/non-noisy manually curated units on the selected ALM probe, then apply the FR threshold… Reserve single-unit-only filtering only for analyses where the paper explicitly did so." Step 10 notes the residual 41-unit (1.6%) gap to the paper's 2,496.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. One subtraction per spike: `trialtm` is already on the behaviour clock relative to its own trial's start, and `bp.ev.goCue` is on the same clock, so `aligned = trialtm − goCue[trial−1]` gives seconds from go cue. Spikes outside `[-2.5, 2.5)` and spikes belonging to unselected trials are dropped by the `keep` mask before binning. No interpolation or per-trial offset is applied to the neural stream.

ii.
```python
def compute_unit_trial_matrix(unit, go_cue, trial_to_pos, n_sel, time_edges):
    aligned = unit["trialtm"] - go_cue[unit["trial"] - 1]
    trial_pos = trial_to_pos[unit["trial"] - 1]
    keep = ((trial_pos >= 0) & np.isfinite(aligned)
            & (aligned >= time_edges[0]) & (aligned < time_edges[-1]))
```

iii. Step 4: "Use the stored `bp.ev.goCue` field for all trials as the universal alignment event", matching both the user request and `params.alignEvent = 'goCue'` in `getDefaultParams.m`; Step 1 identifies `alignSpikes.m` as doing `trialtm_aligned = trialtm - event`. Step 10's independent raw-file rebuild of a unit's aligned trace matched the converted data.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 5 ms bins (`DT = 1/200`) on a fixed window of −2.5 s to +2.5 s → exactly 1,000 bins for every trial of every session. The grid is built once per session from the same `time_edges`/`time_vec` used by the neural, input and all three video streams, so nothing is rebinned afterwards. `metadata['time_bin_size'] = 5.0` ms, `off_start = -2.5`, `off_end = 2.5`. Verification confirms T = 1000 for every trial.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 200.0
...
time_edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time_vec = time_edges[:-1] + DT / 2.0
```

iii. Step 4: "`WorkingWithDataObjs.m` example uses `dt = 1/100` (10 ms); `getDefaultParams.m` uses `dt = 1/200` (5 ms)… prefer the paper/code-aligned 5 ms default." Step 5 Key Decision 5: "Use a fixed window of `[-2.5, 2.5] s` around `goCue` with a 5 ms bin (`dt = 1/200`)… gives a shared time axis for neural/video streams." This reproduces `getSeq.m`'s `edges = params.tmin:params.dt:params.tmax; obj.time = edges + params.dt/2` (bin centres).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not read from the raw data; it is the analysis grid the AI defines, i.e. the centres of the 1,000 bins spanning −2.5…2.5 s around each trial's `bp.ev.goCue`. Values run −2.4975 … 2.4975 and are identical for every trial and session.

ii.
```python
time_edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time_vec = time_edges[:-1] + DT / 2.0
...
input_trials.append(time_vec[None, :].astype(np.float32))
...
"input_names": ["time_from_go_cue_s"],
```

iii. Step 5 mapping table: "`obj.time` / aligned bin centers relative to `goCue` → `input[0]`: Continuous time-from-go-cue vector repeated for every trial: shape `(1, n_timepoints)`. Reference code function: `getSeq`. Decoder input requested by user."

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond constructing the bin-centre vector and broadcasting it to `(1, 1000)` float32 for each trial. It is kept as a continuous ramp (as the Decoder Task specifies "continuous, time-varying") rather than a binary event indicator.

ii.
```python
input_trials.append(time_vec[None, :].astype(np.float32))
```

iii. The AI's Step 10 input sanity check: "Independently constructed the go-cue-centered time axis `[-2.4975, ..., 2.4975]` from the reference window/bin definition and compared against `input[0]`. Result: `np.allclose = True`, max absolute difference `0.0`."

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. By construction it *is* the neural binning grid. `time_edges` defines the spike histogram bins and `time_vec = time_edges[:-1] + DT/2` is the input, so input sample *k* is the centre of neural bin *k*. The same `time_vec` is also the interpolation target for the tongue, paw and motion-energy streams, so all five streams share one axis.

ii.
```python
time_edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time_vec = time_edges[:-1] + DT / 2.0
...
bins = np.floor((aligned[keep] - time_edges[0]) / DT).astype(np.int64)   # neural
...
tongue_pos = feature_xy(raw, 0, "tongue", raw["events"]["goCue"], time_vec)   # video
```

iii. Step 5 planned sanity check: "Check that all trials in the converted dataset share the same time vector and that `input[0]` matches the neural bin centers exactly" — reported as passing in Step 10.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. `bp.R` alone (with `bp.L` used only in the trial mask as a consistency requirement). `bp.hit`/`bp.miss` are **not** used for the direction. Encoded left = 0, right = 1, broadcast as a constant time series over the 1,000 bins.

ii.
```python
lick_direction = np.int64(1 if raw["R"][trial_idx] == 1 else 0)
```
```python
& ((raw["R"] == 1) | (raw["L"] == 1))     # in session_valid_trial_mask
```

iii. Step 5 mapping table: "`bp.R`, `bp.L` → `output[0]`: Lick direction: left `0`, right `1`… Reference code: `findTrials` conditions use `R` and `L` throughout. Keep only non-early hit/miss trials so direction is well defined." The AI treated `R`/`L` as the trial's lick direction; it never discusses that on error trials the mouse licks the *other* port.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A direct relabelling of `bp.R` (1 → class 1 "right", otherwise class 0 "left"), tiled across all 1,000 time bins so the per-trial label is time-varying in shape. Only two classes exist; there is no `none`/`no lick` class because ignore trials were removed upstream (1-e). The delivered distribution is 0.498 / 0.502.
This means every **error (miss) trial is labelled with the instructed port rather than the port the animal licked**. I verified on `JEB23_2023-10-18` that `bp.R` is the *instructed* side: for 200 hit trials the first post-go-cue lick side matched `bp.R` 200/200, while for 33 miss trials it matched 0/33. With misses at 13.8% of the retained trials, ~1 in 7 trials carries the wrong direction label.

ii.
```python
lick_direction = np.int64(1 if raw["R"][trial_idx] == 1 else 0)
output_arr = np.vstack([
    np.full(time_vec.size, lick_direction, dtype=np.int64),
    ...
])
```
```python
"output_values": [
    ["left", "right"],
    ...
]
```

iii. Step 5 Key Decision 7: "Represent trial-level categorical outputs as constant time series: this keeps all outputs on the same `(n_output, n_timepoints)` grid while preserving the per-trial labels requested by the user." The two-class encoding follows the spec the agent was given ("Lick direction (left = 0, right = 1, per-trial)"). No justification is offered anywhere for using the cued side on error trials — the issue is not raised in the notes or the trajectory.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The single per-trial flag `bp.autowater`: autowater trials are the water-cued (WC) context, all others the delayed-response (DR) context.

ii.
```python
"autowater": np.asarray(bp.autowater, dtype=np.float64).reshape(-1),
...
context = np.int64(1 if raw["autowater"][trial_idx] == 0 else 0)
```

iii. Step 5 mapping table: "`bp.autowater` → `output[1]`: Behavioral context: WC `0`, DR `1` via `1 - autowater`… Stored `autowater=1` corresponds to WC in the reference code/paper." This is the same field and polarity the reference uses.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling, `context = 1 - autowater` (WC = 0, DR = 1), broadcast across all 1,000 bins. Delivered distribution 0.084 WC / 0.916 DR (reference: 0.097 / 0.903 — the small difference follows from the different trial set). 20 of 44 sessions are pure DR (range shown as `[1.0, 1.0]` in the verification output), which is expected since only the two-context sessions contain WC blocks.

ii.
```python
context = np.int64(1 if raw["autowater"][trial_idx] == 0 else 0)
...
"output_values": [..., ["WC", "DR"], ...]
```

iii. Codes follow the prompt's WC = 0, DR = 1. Step 10 output sanity check loaded raw `bp.autowater` for one trial of `JEB23_2023-10-18` and compared against the converted row: `np.allclose = True`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `bp.hit` and `bp.miss`. `bp.miss` enters only through the trial mask (`hit | miss`), after which `hit == 1` → correct and `hit == 0` → incorrect. `bp.no` is loaded but not used for the label — ignore trials were already removed.

ii.
```python
"hit": np.asarray(bp.hit, dtype=np.float64).reshape(-1),
"miss": np.asarray(bp.miss, dtype=np.float64).reshape(-1),
"no": np.asarray(bp.no, dtype=np.float64).reshape(-1),
...
outcome = np.int64(1 if raw["hit"][trial_idx] == 1 else 0)
```

iii. Step 5 mapping table: "`bp.hit`, `bp.miss` → `output[2]`: Outcome: miss `0`, hit `1`… Exclude `no` and `early` trials rather than inventing a label for ignores."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Relabelling into two classes — incorrect (0) on miss trials, correct (1) on hits — tiled across all 1,000 bins. Ignore trials are absent from the dataset entirely, so there is no third class. Delivered distribution 0.138 incorrect / 0.862 correct; the reference's three-class version is 0.120 / 0.749 / 0.131.

ii.
```python
outcome = np.int64(1 if raw["hit"][trial_idx] == 1 else 0)
...
"output_values": [..., ["incorrect", "correct"], ...]
```

iii. As in 1-e: the AI's spec listed "Outcome (incorrect = 0, correct = 1, per-trial)" with no ignore class, and the paper "omits ignore trials from analyses", so the AI dropped them rather than inventing a label. Step 12 notes `outcome` "is somewhat imbalanced (0.138 / 0.862) but still decoded well above chance under balanced accuracy" (0.646).

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj{1}` (side camera) only, feature name `tongue`: per-trial `ts` (x, y, likelihood per frame), `frameTimes`, and `NdroppedFrames`. The bottom camera's `top_tongue` is *not* used. Also required: `bp.ev.goCue` and, for the clock correction, `obj.sglx.bitcode.bitstart`, `obj.sglx.fs` and `bp.ev.bitStart`. Only the x and y columns are read (`ts[:, :2, feat]`); the likelihood column is not thresholded explicitly — the AI relies on the authors having already set x/y to NaN below likelihood 0.9 (I confirmed on `EKH1_2021-08-07` trial 0 that the NaN fraction, 0.955, is exactly the complement of the likelihood > 0.9 fraction, 0.045).

ii.
```python
needed_by_view = [["tongue"], ["top_paw", "bottom_paw"]]
...
tongue_pos = feature_xy(raw, 0, "tongue", raw["events"]["goCue"], time_vec)
tongue_speed = feature_speed(*tongue_pos, "tongue")
```
```python
feat_idx = trial["feat_names"].index(feature_name)
ts = np.asarray(trial["ts"][:, :2, feat_idx], dtype=np.float64)
```

iii. Step 5 Key Decision 8: "Define tongue velocity from the side-view `tongue` marker speed magnitude: this uses a direct reference-processed kinematic channel without inventing an unreferenced cross-camera combination." (The reference solution instead averages both views after per-view percentile normalisation.)

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. A port of the authors' `findPosition.m` + `findVelocity.m`:
1. Trials whose `NdroppedFrames` is NaN are skipped entirely (as MATLAB's `continue`), and if `frameTimes` is missing/all-NaN a synthetic 400 Hz axis is used.
2. Frame times are put on the go-cue clock (7-d) and x, y are **linearly interpolated onto the 5 ms grid** (`np.interp`).
3. Speed is `hypot(gradient(x), gradient(y))` on the interpolated grid (units are pixels per 5 ms bin; the baseline-derivative subtraction and nearest-fill of `findVelocity.m` are correctly applied only to non-tongue features).
4. Tongue NaNs become 0 velocity (MATLAB: "set tongue velocity to 0 if not visible"), then the AI re-masks to NaN wherever the interpolated position is non-finite so that untracked bins can be handled separately.
No smoothing is applied to the tongue (matching `findPosition.m`, which smooths only non-tongue features), and no cross-view normalisation is needed since only one view is used.

**Deviation found:** `interp_to_taxis` drops the NaN samples before interpolating, so `np.interp` **bridges the gaps** where the tongue is invisible, whereas MATLAB's `interp1` propagates NaN through them. On `EKH1_2021-08-07` the tongue is genuinely tracked in 10.8% of raw frames, but 38.0% of the converted bins come out finite, and 19.3% of all bins are labelled "at or above median" in the full dataset versus ~6% in the reference. Tongue position and velocity are therefore fabricated by linear interpolation across a large fraction of the invisible periods.

ii.
```python
def interp_to_taxis(old_t, values, new_t):
    mask = np.isfinite(old_t) & np.isfinite(values)
    if mask.sum() < 2:
        return np.full_like(new_t, np.nan, dtype=np.float64)
    return np.interp(new_t, old_t[mask], values[mask], left=np.nan, right=np.nan)
```
```python
    if not math.isfinite(trial["n_dropped"]):
        continue
    ...
    if frame_times is None or frame_times.size == 0 or np.all(~np.isfinite(frame_times)):
        frame_times = (np.arange(ts.shape[0], dtype=np.float64) + 1.0) / 400.0
    old_t = frame_times - vidshift - align_times[trix]
    xpos[:, trix] = interp_to_taxis(old_t, ts[:, 0], taxis)
    ypos[:, trix] = interp_to_taxis(old_t, ts[:, 1], taxis)
    if "tongue" not in feature_name:
        xpos[:, trix] = nearest_fill_1d(xpos[:, trix])
        ypos[:, trix] = nearest_fill_1d(ypos[:, trix])
```
```python
        xv = np.gradient(tsinterp[:, 0]); yv = np.gradient(tsinterp[:, 1])
        if "tongue" not in feature_name:
            xv = xv - basederiv[0]; yv = yv - basederiv[1]
            xv = nearest_fill_1d(xv);  yv = nearest_fill_1d(yv)
        else:
            xv = np.nan_to_num(xv, nan=0.0); yv = np.nan_to_num(yv, nan=0.0)
    return np.sqrt(xvel**2 + yvel**2)
```
```python
tongue_valid = np.isfinite(tongue_pos[0]) & np.isfinite(tongue_pos[1])
tongue_speed[~tongue_valid] = np.nan
```

iii. Step 10 reference-code comparison: "Time-varying kinematic/motion outputs follow the same interpolation/video-offset logic as `findPosition`, `findVelocity`, and `loadMotionEnergy`, then apply the decoder-task-specific median split requested by the user." Step 1: "Missing position values filled with nearest values for all features except tongue. Velocity is the first derivative of position." No justification is given for bridging NaN gaps in the tongue interpolation — the notes assert the reference rule was followed.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A single per-session threshold: the 50th percentile of the finite tongue speeds pooled across all kept trials and bins of that session. Bins at or above it get class 1, bins below get class 0, and bins where the tongue is not tracked also get class 0. There are only two classes — there is no separate "not visible" class. Delivered distribution 0.807 / 0.193 (reference, three-class: 0.062 / 0.062 / 0.875).

ii.
```python
def summarize_threshold(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.nanpercentile(values, 50.0))
...
tongue_thr = summarize_threshold(tongue_sel)
...
np.where(np.isfinite(tongue_sel[:, local_idx]), tongue_sel[:, local_idx] >= tongue_thr, 0).astype(np.int64),
```

iii. Step 5 Key Decision 10: "Use per-session 50th-percentile thresholds for tongue velocity, paw velocity, and motion energy: this follows the decoder task even where it differs from the paper's manual movement threshold" (the paper's `me.moveThresh` is loaded but deliberately unused). The absence of a third class follows the two-bin spec the agent was handed.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Via the authors' `findVideoOffset.m`: `vidshift = mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)`, one constant per session. Each trial's frames are then placed at `frameTimes − vidshift − goCue[trial]` and interpolated directly onto `time_vec`, the neural bin centres, so the two streams share one axis by construction. If bitcode metadata is missing the offset falls back to 0.5 s (the constant MATLAB uses in its own `catch` branch).

ii.
```python
def find_video_offset(raw: dict) -> float:
    bitstart = np.asarray(raw["bitcode_bitstart"], dtype=np.float64).reshape(-1)
    fs = float(raw["fs"])
    if bitstart.size == 0 or not np.isfinite(fs) or fs <= 0:
        return 0.5
    return robust_mode(bitstart) / fs - robust_mode(raw["events"]["bitStart"])
```
```python
old_t = frame_times - vidshift - align_times[trix]
xpos[:, trix] = interp_to_taxis(old_t, ts[:, 0], taxis)
```

iii. Step 1/Step 10: "DLC/video alignment uses `frameTimes - vidshift - alignTimes(trial)`, where `vidshift` is computed from bitcode synchronization metadata" — a direct transcription of `findVideoOffset.m` and `findPosition.m`. `robust_mode` reproduces MATLAB's `mode`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. `obj.traj{2}` (bottom camera), **both** `top_paw` and `bottom_paw`, x and y only, plus the same `frameTimes`, `NdroppedFrames`, `bp.ev.goCue` and bitcode fields. The reference solution uses `top_paw` alone.

ii.
```python
needed_by_view = [["tongue"], ["top_paw", "bottom_paw"]]
...
paw_speeds = []
for paw_name in ("top_paw", "bottom_paw"):
    paw_pos = feature_xy(raw, 1, paw_name, raw["events"]["goCue"], time_vec)
    paw_speeds.append(feature_speed(*paw_pos, paw_name))
```

iii. Step 5 Key Decision 9: "Define paw velocity from the average of top- and bottom-paw speed magnitudes in the bottom view: this captures overall paw movement while remaining close to the tracked features described in the paper/code." Step 3 from the methods: "Tongue, jaw and nose tracked in both views; paws tracked in bottom view only."

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The same `findPosition`/`findVelocity` port as the tongue, but with the non-tongue branches active: after interpolation onto the 5 ms grid, missing positions are nearest-filled (`fillmissing(...,'nearest')`), the per-trial median frame-to-frame displacement (`basederiv`) is subtracted from the x and y derivatives to remove tracking drift, and the resulting velocities are nearest-filled again. Speed is `hypot(xvel, yvel)` per paw; the two paws' speeds are then averaged bin-by-bin over whichever is finite, and any remaining NaN becomes 0. No smoothing and no normalisation (one camera, one pixel scale). Units are pixels per bin.

ii.
```python
        basederiv = np.nanmedian(deriv, axis=0)
        ...
        if "tongue" not in feature_name:
            xv = xv - basederiv[0]
            yv = yv - basederiv[1]
            xv = nearest_fill_1d(xv)
            yv = nearest_fill_1d(yv)
```
```python
paw_stack = np.stack(paw_speeds, axis=0)
paw_count = np.sum(np.isfinite(paw_stack), axis=0)
paw_sum = np.nansum(paw_stack, axis=0)
paw_speed = np.divide(paw_sum, np.maximum(paw_count, 1), where=np.maximum(paw_count, 1) > 0)
paw_speed = np.nan_to_num(paw_speed, nan=0.0)
```

iii. Step 1 function table describes `findVelocity` as "computes per-trial x/y velocity from interpolated positions; tongue NaNs become zero velocity, non-tongue NaNs filled by nearest", and Step 10 states the converter "follow[s] the same interpolation/video-offset logic". (Note the AI silently corrects a bug in the MATLAB, which subtracts `basederiv(1)` from *both* x and y velocity; the AI subtracts `basederiv[0]` from x and `basederiv[1]` from y.)

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Per-session 50th percentile of the pooled paw speeds over all kept trials and bins; `>=` → class 1, `<` → class 0. Because missing paw data has already been nearest-filled and NaN-to-zeroed, every bin has a finite value and there is no "not visible" class; the split is therefore exactly 0.500 / 0.500 in every session. (Reference, three-class: 0.407 / 0.407 / 0.186.)

ii.
```python
paw_thr = summarize_threshold(paw_sel)
...
(paw_sel[:, local_idx] >= paw_thr).astype(np.int64),
```

iii. Step 5 Key Decision 10 (per-session 50th-percentile thresholds); Step 9 lists the resulting `[0.500, 0.500]` distribution as matching expectation, and Step 12 notes `paw_velocity_bin` is "near-perfectly balanced by construction".

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue and using the bottom camera's own `frameTimes`: `frameTimes − vidshift − goCue[trial]`, then `np.interp` onto `time_vec`, the neural bin centres. The session's `vidshift` is recomputed inside each `feature_xy` call but is the same constant.

ii.
```python
def feature_xy(raw, view_idx, feature_name, align_times, time_vec):
    trials = raw["traj"][view_idx]
    taxis = time_vec.copy()
    vidshift = find_video_offset(raw)
    ...
        old_t = frame_times - vidshift - align_times[trix]
        xpos[:, trix] = interp_to_taxis(old_t, ts[:, 0], taxis)
```

iii. Same as 7-d: a transcription of `findPosition.m`/`findVideoOffset.m`. Using `raw["traj"][view_idx]`'s own frame times means each feature is timed by the camera that recorded it.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file next to each session file (never `obj.me`), read with `scipy.io.loadmat`. Only `me.data` — one trace per trial, one value per side-camera frame — is used; `me.moveThresh` is loaded but deliberately not used. The side camera's `frameTimes` supply the time base.

ii.
```python
def load_motion_energy(path: Path) -> tuple[list[np.ndarray], float]:
    me = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["me"]
    me_root = me
    if isinstance(me_root, np.ndarray) and me_root.dtype == object and me_root.size == 1:
        item = me_root.reshape(-1)[0]
        if hasattr(item, "data") or hasattr(item, "moveThresh"):
            me_root = item
    if hasattr(me_root, "data") and not isinstance(me_root, np.ndarray):
        raw_data = unwrap_motion_energy_container(me_root.data)
        thresh_obj = getattr(me_root, "moveThresh", float("nan"))
    else:
        raw_data = unwrap_motion_energy_container(me_root)
        thresh_obj = float("nan")
```
```python
me_path = spec.session_path.parent / f"motionEnergy_{spec.subject}_{spec.date}.mat"
motion_energy, motion_thresh = load_motion_energy(me_path)
```

iii. Step 9/10: the AI found three file layouts and documented the fixes — "Nested `motionEnergy` MATLAB structs in `JEB15` sessions required recursive unwrapping of `.data`" (the same guard as `loadMotionEnergy.m`'s `if isstruct(me.data), me.data = me.data.data; end`) and "Randomized-delay `motionEnergy` files store only the per-trial traces in `me` and omit `moveThresh`; loader now accepts that form and records `NaN`".

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond re-timing: the value is already one scalar per frame (the paper's per-pixel median difference reduced to the 99th percentile across pixels). Each trial's trace is interpolated linearly onto the 5 ms grid, then nearest-filled (`fillmissing(me.data,'nearest')` in `loadMotionEnergy.m`), and finally any residual NaN — e.g. a trial with no motion-energy trace at all — is set to 0. No smoothing, no baseline subtraction (the reference's baseline-subtraction block is commented out).

ii.
```python
def aligned_motion_energy(raw, align_times, time_vec):
    me_trials = raw["motion_energy"]
    side_trials = raw["traj"][0]
    vidshift = find_video_offset(raw)
    out = np.full((time_vec.size, len(me_trials)), np.nan, dtype=np.float64)
    for trix, me in enumerate(me_trials):
        ...
        out[:, trix] = interp_to_taxis(old_t, me, time_vec)
        out[:, trix] = nearest_fill_1d(out[:, trix])
    return out
...
motion = np.nan_to_num(motion, nan=0.0)
```

iii. Step 1: "Motion energy loader expects one motion-energy file per ephys session and converts it to `me.data(time, trial)` on the neural time base." The spatial reduction is already done upstream, so only re-timing is required.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session 50th percentile of the pooled aligned motion energy over all kept trials and bins; `>=` → class 1, `<` → class 0. The paper's manual `me.moveThresh` is explicitly not used. Because NaNs were zero-filled first there is no "no video" class, and the split is 0.500 / 0.500 in every session (reference, three-class: 0.479 / 0.483 / 0.038).

ii.
```python
motion_thr = summarize_threshold(motion_sel)
...
(motion_sel[:, local_idx] >= motion_thr).astype(np.int64),
```

iii. Step 5 mapping table: "The paper uses a manual movement threshold for move/non-move analyses; the user explicitly requests a 50th-percentile discretization for decoder output."

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same session `vidshift` and same grid as the tracking, using the **side camera's** frame times, since motion energy has one value per side-camera frame: `frameTimes − vidshift − goCue[trial]`, then interpolation onto `time_vec`. If the side camera's `frameTimes` are missing or all-NaN, a synthetic 400 Hz axis shifted by a fixed −0.5 s is used, reproducing the `catch` branch of `loadMotionEnergy.m`.

ii.
```python
frame_times = side_trials[trix]["frame_times"] if trix < len(side_trials) else None
if frame_times is None or frame_times.size == 0 or np.all(~np.isfinite(frame_times)):
    old_t = (np.arange(me.size, dtype=np.float64) + 1.0) / 400.0 - 0.5 - align_times[trix]
else:
    old_t = frame_times - vidshift - align_times[trix]
out[:, trix] = interp_to_taxis(old_t, me, time_vec)
```

iii. A line-for-line port of `loadMotionEnergy.m`'s `interp1(obj.traj{1}(trix).frameTimes-vidshift-alignTimes(trix), me.data{trix}, taxis)` and its fallback `interp1(frameTimes-0.5-alignTimes(trix), ...)`.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. A long list of cases, all handled by keeping the trial/session and substituting a value:
- **Video trials flagged bad** (`NdroppedFrames` NaN) are skipped in `feature_xy`, leaving that trial's positions all NaN → tongue speed NaN → class 0; paw speed 0 → class 0.
- **Missing/all-NaN `frameTimes`**: a synthetic 400 Hz axis is built (`(1:n)/400`), with a −0.5 s shift for motion energy — both directly from the MATLAB fallbacks.
- **Missing bitcode metadata**: `find_video_offset` returns 0.5.
- **Untracked frames** (likelihood ≤ 0.9, already NaN in the files): nearest-filled for paw, mapped to 0 velocity for tongue, per `findVelocity.m`.
- **Residual NaNs** in paw and motion energy are replaced with 0 (`np.nan_to_num`).
- **Trailing behavioural trials with no neural coverage** (two JEB24 sessions) are dropped.
- **Empty HDF5 probe slots** (`JEB6_2021-04-18`) are handled by indexing raw slots and raising if the requested slot is empty.
- **Three motion-energy file layouts** are unwrapped recursively; a missing `moveThresh` becomes NaN.
- **Two MATLAB file formats and two `ts` axis orders** are handled by separate readers plus a transpose.
- **Degenerate sessions** (<10 units or <2 trials) would be skipped; none were.
The key consequence is that "no data" and "no movement" are conflated: untracked bins land in class 0 rather than a dedicated class.

ii.
```python
    if not math.isfinite(trial["n_dropped"]):
        continue
```
```python
if bitstart.size == 0 or not np.isfinite(fs) or fs <= 0:
    return 0.5
```
```python
paw_speed = np.nan_to_num(paw_speed, nan=0.0)
motion = np.nan_to_num(motion, nan=0.0)
```
```python
covered_trial_max = [int(np.nanmax(unit["trial"])) for unit in raw["units"]
                     if good_quality(unit["quality"]) and np.asarray(unit["trial"]).size]
if covered_trial_max:
    max_neural_trial = min(raw["R"].size, max(covered_trial_max))
    selected_trials = selected_trials[selected_trials + 1 <= max_neural_trial]
```

iii. Step 9 and Step 10 "Issues Found and Resolved" document each case with the session that exposed it. The general principle stated is to reproduce the reference MATLAB's own missing-data handling (skip bad video trials, synthesize frame times, nearest-fill non-tongue features, zero tongue velocity when invisible) rather than to drop trials, since "their neural and behavioural data are unaffected".

## 11-a. What are the most time-consuming steps of the code?

i. File reading dominates. The full conversion of 44 sessions takes 135.2 s wall clock, 1.5–5.1 s per session, scaling with file size and trial count; the AI reports that "full recursive HDF5-to-Python loading was too slow and memory-heavy" and replaced it with targeted field reads. The heaviest compute after loading is the per-unit smoothing in `my_smooth`, which runs one `np.convolve` per trial per unit (~660k 1,015-sample convolutions over the dataset), followed by the per-trial `np.interp` calls for the four video features.

ii.
```python
    t0 = time.time()
    raw = load_session(spec)
    ...
    log(f"SESSION {spec.session_id}: kept {session_out['n_trials']} trials, "
        f"{session_out['n_units']} units, format={raw['format']}, "
        f"time={time.time() - t0:.2f}s")
```
```python
    out = np.empty_like(x_filt, dtype=np.float64)
    for j in range(x_filt.shape[1]):
        out[:, j] = np.convolve(x_filt[:, j], kern, mode="same")
```

iii. Step 6: "Full recursive HDF5-to-Python loading was too slow and memory-heavy for the large session files… Implemented targeted field loading instead of full-session recursive decoding for HDF5 files." Step 7 recorded per-session timing and the extrapolated total, which came in well under the 15-minute budget.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain. (1) `my_smooth`'s column loop — this is a same-kernel 1-D convolution over a rectangular `(1000, n_trials)` array and could be one `scipy.ndimage.convolve1d` or `fftconvolve` call; it is the single largest avoidable cost. (2) `feature_xy`'s and `aligned_motion_energy`'s per-trial `np.interp` loops — hard to vectorise, since each trial has a different number of camera frames. (3) `feature_speed`'s per-trial loop over an already-rectangular `(1000, n_trials)` array — `np.gradient`, `np.nanmedian` and the baseline subtraction could all be done on the whole array at once; only `nearest_fill_1d` needs per-column work. The spike binning is already vectorised with `np.add.at` instead of a per-trial histogram.

ii.
```python
    for j in range(x_filt.shape[1]):
        out[:, j] = np.convolve(x_filt[:, j], kern, mode="same")
```
```python
    for i in range(n_trials):
        tsinterp = np.column_stack([xpos[:, i], ypos[:, i]])
        ...
        xv = np.gradient(tsinterp[:, 0])
        yv = np.gradient(tsinterp[:, 1])
```
```python
    if np.any(keep):
        bins = np.floor((aligned[keep] - time_edges[0]) / DT).astype(np.int64)
        np.add.at(mat, (trial_pos[keep], bins), 1.0 / DT)
```

iii. Step 6: "Per-spike/per-trial histogram loops would likely be a bottleneck if implemented naively… Implemented vectorized per-unit spike accumulation with `np.add.at` instead of calling a histogram inside nested trial loops." The AI did not identify the remaining smoothing/velocity loops as vectorisation targets, presumably because the measured runtime was already acceptable.

## 11-c. What processing does the code repeat multiple times?

i. A few small redundancies, none material at 135 s total:
- `find_video_offset(raw)` is recomputed inside every `feature_xy` call (tongue, top_paw, bottom_paw) and again in `aligned_motion_energy` — four times per session for a session-wide constant.
- `is_hdf5_mat` opens each `.mat` file once purely to sniff its format, and `load_session` then opens it again; in `--sample` mode `choose_sample_specs` sniffs many files a second time.
- For v5 sessions, `scipy.io.loadmat` reads the entire `obj` (all probes, all DLC features, all `bp.ev` fields) even though only a subset is used, whereas the HDF5 path is targeted.
- For HDF5 sessions, the feature-name list is scanned per trial until the needed indices are found (early-exits after the first hit).
Each session's `obj` is dropped when `convert_one_session` returns, and the bin grid, trial mask and per-session thresholds are each computed once.

ii.
```python
def feature_xy(raw, view_idx, feature_name, align_times, time_vec):
    ...
    vidshift = find_video_offset(raw)
```
```python
def load_session(spec: SessionSpec) -> dict:
    return load_session_hdf5(spec) if is_hdf5_mat(spec.session_path) else load_session_mat(spec)
```

iii. Step 6 states only that the script "processes one session at a time to keep memory bounded"; the notes do not enumerate these repeats, and the AI's timing analysis concluded the conversion was already fast enough that no further optimisation was warranted.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items are computed or loaded and never reach the pickle:
- The `continuous` block (float32 tongue/paw/motion arrays plus the three thresholds) is built for every session but is only consumed by `plot_processing`; `build_dataset` never copies it into `data`.
- Raw fields loaded and unused: `bp.L` (only used in the trial mask), `bp.no`, `bp.ev.sample`, `bp.ev.delay`, `bp.ev.reward`, and `me.moveThresh` (loaded, threaded through the loader, then discarded because the task mandates a percentile split).
- `nearest_fill_2d`, `h5_is_empty` and `h5_nonempty_probe_groups` are defined but never called.
- The whole-`obj` `scipy.io.loadmat` on v5 sessions materialises every DLC feature and every cluster field (`tm`, waveforms, channels) although only `tongue`, the two paws, `trialtm`, `trial` and `quality` are used.
- `bottom_paw` is fully processed and then averaged into a signal whose partner, `top_paw`, is the reliably tracked one.
Everything else computed after loading ends up in the output.

ii.
```python
        "continuous": {
            "tongue": tongue_sel.astype(np.float32),
            "paw": paw_sel.astype(np.float32),
            "motion": motion_sel.astype(np.float32),
            "tongue_thr": tongue_thr, "paw_thr": paw_thr, "motion_thr": motion_thr,
        },
```
```python
        "events": {
            "bitStart": ..., "sample": ..., "delay": ...,
            "goCue": ..., "reward": ...,
        },
        ...
        "motion_thresh": motion_thresh,
```
```python
def nearest_fill_2d(x: np.ndarray) -> np.ndarray:      # never called
```

iii. Not discussed in CONVERSION_NOTES. The `continuous` block and the extra `bp.ev` fields were evidently kept for the `--show-processing` plots and for the Step 10 sanity checks, and `me.moveThresh` was loaded because `loadMotionEnergy.m` uses it, then intentionally left unused: "The paper uses a manual movement threshold for move/non-move analyses; the user explicitly requests a 50th-percentile discretization."
