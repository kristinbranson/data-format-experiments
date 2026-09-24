# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not glob the data folders. It reconstructs the authors' curated session list *programmatically* by parsing the reference MATLAB manifests `code/DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m`: it strips commented-out lines (`%`), regex-extracts every `meta(end).date = '...'` and the matching `meta(end).probe = ...`, and then locates `data_structure_<anm>_<date>.mat` in either `Ephys_Behavior/` or `RandomizedDelay_Ephys_Behavior/`. This yields exactly 44 sessions (25 fixed-delay + 19 randomized-delay), 14 mice. Each session file is opened once; MATLAB v7.3 files are read with `h5py` (with explicit cell-reference dereferencing) and v5 files with `scipy.io.loadmat`, both normalised into one internal dict (`bp`, `sglx_fs`, `sglx_bitstart`, `traj`, `clu_probes`, `probe_locs`). Motion energy is read from the sibling `motionEnergy_<anm>_<date>.mat`. Behaviour-only cohorts (`DelayInhibition_BilatMC_Behavior`, `GoCueInhibition_BilatMC_Behavior`) are never touched.

ii.
```python
def parse_manifest_sessions() -> list[SessionSpec]:
    sessions: list[SessionSpec] = []
    for manifest in sorted(MANIFEST_ROOT.glob("load*_ALMVideo.m")):
        subject = manifest.stem.replace("load", "").replace("_ALMVideo", "")
        lines = [line for line in manifest.read_text().splitlines() if not line.lstrip().startswith("%")]
        text = "\n".join(lines)
        dates = re.findall(r"meta\(end\)\.date = '([^']+)';", text)
        probes = re.findall(r"meta\(end\)\.probe = ([^;]+);", text)
        ...
        for date, probe_text in zip(dates, probes):
            probe_list = parse_probe_list(probe_text)
            for cohort in ("Ephys_Behavior", "RandomizedDelay_Ephys_Behavior"):
                data_path = DATA_ROOT / cohort / f"data_structure_{subject}_{date}.mat"
                if data_path.exists():
                    sessions.append(SessionSpec(subject=subject, date=date, probes=probe_list,
                                                data_path=data_path, cohort=cohort))
                    break
    return sessions
```

```python
def load_raw_session(path: Path) -> dict:
    if h5py.is_hdf5(path):
        return load_session_v73(path)
    return load_session_v5(path)
```

iii. From CONVERSION_NOTES.md Step 4/Step 5: "`/app/data` contains 120 sessions across behavior-only and ephys cohorts, including sessions not used in the paper analyses… Final conversion should target the paper/code ALM ephys subset, not all downloaded sessions." Key Decision 1–3: use only the ALM manifest sessions, exclude behaviour-only cohorts (no neural data), and note that the 12 "two-context" sessions are a subset of the 25 fixed-delay sessions rather than extra files. Step 4 also records that the raw files "mix MATLAB v7.3 HDF5 and MATLAB v5 layouts", hence the dual loader.

## 1-b. How are the data split into subjects?

i. The subject is the animal id taken from the manifest file name (`loadJEB19_ALMVideo.m` → `JEB19`), which is also the prefix of the data file name. It is stored on the `SessionSpec` and carried through to each session result. At assembly, `subjects` is the sorted set of unique animals and `subject_idx` is each session's index into that list. Result: 14 subjects, with 1–8 sessions each — identical to the human reference.

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
subjects = sorted({sess["subject"] for sess in processed_sessions})
subject_to_idx = {subj: i for i, subj in enumerate(subjects)}
...
"subjects": subjects,
"subject_idx": np.array([subject_to_idx[sess["subject"]] for sess in processed_sessions], dtype=np.int64),
```

iii. CONVERSION_NOTES.md Step 5 variable mapping: "Mouse/session identity from filename + manifest … Use manifest-ordered sessions to match paper/code selection", citing `load*_ALMVideo.m` as the authoritative record of which animal each session belongs to. The AI also observed (Step 2) that `obj.ex`/`meta` fields are not uniformly present, so file/manifest naming is the reliable source.

## 1-c. How are the data split into sessions?

i. One session = one `(subject, date)` entry in a manifest = one `data_structure_*.mat` file, searched for in both ephys cohort folders so that fixed-delay and randomized-delay sessions are handled uniformly. Each session becomes one element of `neural`, `input`, `output`, and one entry in `metadata['session_info']` (with `cohort`, `raw_trials`, `valid_trials`, `units_pre_filter`, `units_post_filter`, thresholds). Sessions with fewer than 2 usable trials or fewer than 10 surviving units would be dropped; in practice none were.

ii.
```python
for cohort in ("Ephys_Behavior", "RandomizedDelay_Ephys_Behavior"):
    data_path = DATA_ROOT / cohort / f"data_structure_{subject}_{date}.mat"
    if data_path.exists():
        sessions.append(SessionSpec(...)); break
```
```python
if len(neural_trials) < 2 or len(keep_unit_idx) < 10:
    info["skipped"] = True
    return None, info
```

iii. Step 4: the manifests define 25 fixed-delay + 19 randomized-delay sessions, with `JEB23 2023-10-20` commented out and `JEB24 2023-10-03/04` lacking `clu`; "Use the 19-session manifest from code." The ≥10-unit rule comes from the paper's curation note recorded in Step 3: "Recording sessions included only if they had at least 10 units."

## 1-d. How are the data split into trials?

i. A trial is one index into the per-trial fields of `obj.bp` (`hit`, `miss`, `no`, `early`, `autowater`, `R`, `L`, `stim.enable`, `ev.goCue`, …), with `obj.bp.Ntrials` giving the count. Camera trajectories (`obj.traj{view}`) are read as one entry per trial, motion energy as one cell per trial, and spikes already carry `trial` (1-based) plus `trialtm` (time within trial), so trial boundaries never have to be reconstructed. All per-trial arrays are read with `reshape(-1)`; they are indexed by trial number so no truncation to `Ntrials` is applied (I verified that no field in any of the 44 sessions is longer than `Ntrials`, so this is harmless here).

ii.
```python
ntrials = int(np.asarray(bp["Ntrials"][()], dtype=float).reshape(-1)[0])
bp_out = {"Ntrials": ntrials,
          "hit": bp["hit"][()].reshape(-1).astype(bool), ...
          "ev": {"goCue": ev["goCue"][()].reshape(-1), ...}}
...
for tr_idx in range(ntrials):
    trial = raw["traj"][view_idx][tr_idx]
```
```python
trial_idx = unit["trial"] - 1                      # spike -> trial, 1-based in file
aligned = unit["trialtm"] - align_times[trial_idx]
```

iii. Step 2 of CONVERSION_NOTES.md documents the Bpod trial table (`obj.bp` fields and `obj.bp.ev` event times) and the per-trial structure of `obj.traj` and `obj.clu`; the mapping table in Step 5 uses those fields directly. No separate justification was needed because the trial structure is explicit in the files.

## 1-e. How are trials filtered based on quality controls?

i. Three filters, and one notable omission.
1. **Early-lick trials dropped** (`~bp.early`).
2. **Trials with a non-finite go cue dropped** (`np.isfinite(goCue)`).
3. **Trials whose entire neural matrix is zero dropped**, applied after binning. This removed 28 trials from `JEB24_2023-10-23` and 33 from `JEB24_2023-11-03`, where behaviour continued past the end of the probe recording (max spike trial 314/312 vs 343/346 behavioural trials).
**Photostimulation trials are NOT excluded.** `bp.stim.enable` is loaded into the internal representation but never used for filtering; `metadata['trial_exclusion'] = 'early_lick_only'`. I verified this leaves 173 photoinactivation trials in the dataset, concentrated in 5 sessions (`JEB15_2022-07-29` 52/245 = 21%, `JEB6_2021-04-18` 55/357 = 15%, `JEB7_2021-04-29` 39/284 = 14%, `JEB7_2021-04-30` 26, `JGR3_2021-11-18` 1). Net: 13,935 of 14,972 trials kept.

ii.
```python
candidate_trials = np.flatnonzero(~raw["bp"]["early"] & np.isfinite(raw["bp"]["ev"]["goCue"]))
...
nonzero_trial_mask = np.array([np.any(trial != 0) for trial in neural_trials], dtype=bool)
if nonzero_trial_mask.size and not np.all(nonzero_trial_mask):
    candidate_trials = candidate_trials[nonzero_trial_mask]
    neural_trials = [trial for trial, keep in zip(neural_trials, nonzero_trial_mask.tolist()) if keep]
valid_trials = candidate_trials
```
```python
"trial_exclusion": "early_lick_only",
```

iii. Key Decision 4: "Exclude early-lick trials but retain ignore trials: Early trials are omitted throughout the paper's analyses and can have abnormal within-trial structure; ignore trials are explicitly requested as an output class." For the zero-neural trials (Step 10): "valid behavioral trials extended beyond the maximum spike trial index stored in the selected units (314 < 343 and 312 < 346)… drop any trial whose full `(n_neurons, n_timepoints)` neural matrix is zero." No justification is given anywhere for keeping photostim trials — Step 1 of the notes actually records that the reference "commonly exclude `stim.enable` and early trials", and the Step 10 reference-comparison table nevertheless asserts "Same base exclusion of early trials and finite go-cue requirement", i.e. the omission appears to be an oversight rather than a reasoned choice.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu` — the spike-sorted clusters of the manifest-selected ALM probe(s). Per unit the AI reads `quality` (manual curation label), `trial` (1-based trial of each spike), `trialtm` (spike time relative to trial start), plus `tm` and `site`/`channel` (loaded but unused downstream). For two-probe manifest entries (e.g. `JEB15` `[1 2]`) the units of both probes are concatenated. `obj.bp.ev.goCue` supplies the alignment times and `obj.ex.probe.loc` the region label.

ii.
```python
units.append({"quality": quality, "tm": tm, "trialtm": trialtm, "trial": trial,
              "site": int(site[0]) if site.size else -1})
```
```python
def selected_units(raw, probes):
    units, regions = [], []
    for probe in probes:
        probe_idx = probe - 1
        probe_units = raw["clu_probes"][probe_idx] if probe_idx < len(raw["clu_probes"]) else []
        probe_loc = raw["probe_locs"][probe_idx] if probe_idx < len(raw["probe_locs"]) else "ALM"
        for unit in probe_units:
            if matlab_quality_ok(unit["quality"]):
                units.append(unit); regions.append(normalize_region(probe_loc))
    return units, regions
```

iii. Step 1/Step 5: the reference pipeline is `findTrials → findClusters → alignSpikes → getSeq → removeLowFRClusters`, operating on `obj.clu{prb}(clu).trial/.trialtm/.quality`; "Probe selection is encoded in per-animal manifest files… dual-probe sessions are concatenated at load time only for the probe(s) marked as ALM."

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 10 ms bins over [−2.5, 2.5) s from the go cue, divided by the bin width to give Hz, and smoothed along time with a **causal** Gaussian — a direct Python transcription of the reference `mySmooth(x, N=15, 'reflect')`: pad the first 15 samples in front, build `gausswin(15)` (σ = (N−1)/2/2.5 = 2.8 samples = 28 ms at 10 ms bins), zero the first half of the kernel to make it causal, normalise, convolve `'same'`, then trim the pad. No normalisation, baseline subtraction or z-scoring. Stored as `float32` in Hz, one `(n_neurons, 500)` matrix per trial.

ii.
```python
def smooth_causal_reflect(x, n, bctype="reflect"):
    ...
    padded = np.concatenate([arr[:n, :], arr], axis=0); trim = n
    std = ((n - 1) / 2) / 2.5
    kern = gaussian(n, std=std)
    kern[: len(kern) // 2] = 0.0          # causal
    kern = kern / kern.sum()
    for col in range(padded.shape[1]):
        out[:, col] = np.convolve(padded[:, col], kern, mode="same")
    return out[trim:, :]
```
```python
bin_idx = np.floor((aligned[mask] - TMIN) / DT).astype(int)
np.add.at(counts, (mapped_trials, bin_idx), 1.0)
fr = smooth_causal_reflect((counts / DT).T, SMOOTH_N, BCTYPE).T
```

iii. Step 1: "Smoothing is applied with `mySmooth` after converting counts to firing rate (`N / dt`). Tutorial code uses a causal Gaussian kernel with window `15` and boundary condition `reflect`." Key Decision 6: "Use reference-style smoothed firing-rate trial matrices rather than raw spike counts: the paper's pipeline works on aligned/smoothed `trialdat`." (Independently verified: re-deriving the rates from the raw HDF5 file for `JEB6_2021-04-18` reproduces the stored matrices, max abs diff 7.6e-06.)

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three layers. (1) Probe selection: only the manifest ALM probe(s). (2) Quality label: the unit is kept unless its stripped, lower-cased `quality` is in `{garbage, gabrga, noisy, real?}` — exactly the exclusion list in the reference `findClusters.m` for `quality = {'all'}` (so `poor`, `fair`, `multi`, unlabeled units are all kept, i.e. multi-units included). (3) Mean firing rate: the unit is kept only if the mean of its smoothed rate over all kept trials and all 500 bins exceeds 1 Hz. Result: 2,456 units over 44 sessions (17–141 per session); the paper reports 1,651 + 845 = 2,496 recorded units for the same cohorts, and the AI's pre-1 Hz count was 2,513.

ii.
```python
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}
LOW_FR_HZ = 1.0

def matlab_quality_ok(quality: str) -> bool:
    return quality.strip().lower() not in QUALITY_EXCLUDE
```
```python
mean_fr = float(np.nanmean(fr))
if mean_fr > LOW_FR_HZ:
    keep_idx.append(unit_idx); keep_mats.append(fr.astype(np.float32))
```

iii. Step 4 discrepancy table: "`WorkingWithDataObjs.m` uses `params.lowFR = 1`; `getDefaultParams.m` uses `0.5`… Methods state 'All units with firing rates exceeding 1 Hz were included in all other analyses' → Use a 1 Hz firing-rate threshold." And: "For the decoder conversion, include all non-garbage/non-noisy ALM units that pass >1 Hz, not single-units-only, because the decoder task is closer to 'all other analyses' than to the single-unit selectivity analyses" (Key Decisions 7–8).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. One subtraction. `clu.trialtm` is already on the behaviour clock and relative to its own trial's start, and `bp.ev.goCue` is on the same clock, so `trialtm − goCue[trial]` gives seconds from go cue. Spikes outside [−2.5, +2.5) are dropped, and spikes belonging to trials that were filtered out are dropped via a trial map. No interpolation or additional offset. (The camera streams get a separate clock correction — see 7-d.)

ii.
```python
align_times = raw["bp"]["ev"]["goCue"]
trial_idx = unit["trial"] - 1
aligned = unit["trialtm"] - align_times[trial_idx]
valid_mask = np.array([t in trial_map for t in trial_idx], dtype=bool)
in_win = (aligned >= TMIN) & (aligned < TMAX)
mask = valid_mask & in_win
```

iii. Step 1: "`alignSpikes` aligns spike times to the chosen event (`goCue`, …); for this task the relevant event is `goCue`", and Key Decision 5: "Align everything to go cue and use a common 10 ms grid from −2.5 to 2.5 s." The decoder task itself specifies go-cue alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 10 ms bins (`DT = 1/100`), 500 bins spanning [−2.5, +2.5) s, identical for every trial and session; `metadata['time_bin_size'] = 10.0`. The same grid is used for neural data, the input, and all three camera outputs, so no rebinning or resampling between streams is ever needed. Spike times are histogrammed directly onto this grid (no finer intermediate binning).

ii.
```python
TMIN, TMAX = -2.5, 2.5
DT = 1 / 100  # 10 ms; matches most figure/decoder scripts

def build_time_axis() -> np.ndarray:
    edges = np.arange(TMIN, TMAX + DT, DT)
    return edges[:-1] + DT / 2
```

iii. Step 4 discrepancy table: "`getDefaultParams.m` sets `dt = 1/200` (5 ms), but most figure/decoder scripts and `WorkingWithDataObjs.m` set `dt = 1/100` (10 ms); choice/context decoders then aggregate to 75 ms bins for classification… Use 10 ms bins for the converted neural time series because this matches the majority of task-analysis scripts, including the fixed-delay/randomized-delay figure scripts and the choice/context decoding pipeline." (Confirmed against the reference code: `Figure 1/3/8`, `EDFigure 2/3`, `ParallelAnalysis`, `Behavior/*` and `WorkingWithDataObjs.m` all set `params.dt = 1/100`; `tmin/tmax = ±2.5` comes from `getDefaultParams.m`.)

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Nothing in the raw data — it is defined by the conversion as the centre of each of the 500 bins of the go-cue-aligned window, i.e. −2.495 … +2.495 s. The same `(1, 500)` float32 vector is stored for every trial of every session. It is the only decoder input.

ii.
```python
INPUT_NAMES = ["time_from_go_cue"]
...
def build_time_axis() -> np.ndarray:
    edges = np.arange(TMIN, TMAX + DT, DT)
    return edges[:-1] + DT / 2
...
inp = time_axis[np.newaxis, :].astype(np.float32)
```

iii. Step 5 mapping table: "Bin centers relative to go cue → `input[0]` (`time_from_go_cue`): continuous 1×T vector of bin-center times in seconds, identical for all trials after alignment… `getSeq`/`obj.time` convention. Only decoder input requested by task." The reference `getSeq.m` builds `obj.time = edges + dt/2` the same way.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond constructing the bin-centre grid; the values are the same for all trials so the array is simply materialised per trial.

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT)
return edges[:-1] + DT / 2
```

iii. N/A — the axis is defined by the conversion, matching the reference's `obj.time` convention (Step 5).

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid: spike times are expressed relative to the trial's go cue and assigned to bin `floor((t − TMIN)/DT)` of the same edges whose centres form the input, so bin *k* denotes the same interval in both streams. The AI's own sanity check compared the stored input vector with an independently built centred 10 ms grid (`np.allclose`, max diff 1.14e-07).

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT)   # in build_time_axis and again in bin_session_neural
bin_idx = np.floor((aligned[mask] - TMIN) / DT).astype(int)
```

iii. Key Decision 5 (common grid for all streams) and Step 10 check 2 ("Input check … Result: `np.allclose(...) == True`").

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Five per-trial Bpod flags: `obj.bp.no`, `obj.bp.R`, `obj.bp.L`, `obj.bp.hit`, `obj.bp.miss`. The direction the animal actually licked is not recorded, so it is reconstructed from the instructed side plus the outcome.

ii.
```python
"hit": bp["hit"][()].reshape(-1).astype(bool),
"miss": bp["miss"][()].reshape(-1).astype(bool),
"no": bp["no"][()].reshape(-1).astype(bool),
"R": bp["R"][()].reshape(-1).astype(bool),
"L": bp["L"][()].reshape(-1).astype(bool),
```

iii. Key Decision 10: "Define lick direction as actual lick side, not instructed/rewarded side: Use `R/L` combined with `hit/miss/no` to recover the animal's actual left/right/none behavioral output." Step 5 cites the reference `findTrials`/`getPrevChoice` conventions (`obj.bp.hit & obj.bp.R`, etc.) from `WorkingWithDataObjs.m`.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A per-trial relabelling: an ignore trial (`no`) is class 2 (`none`); otherwise a hit on an R trial or a miss on an L trial means the animal licked right (1), a hit on an L trial or a miss on an R trial means it licked left (0); anything else falls back to 2. The scalar is then broadcast across all 500 bins so the output is time-varying in shape. Resulting distribution 0.423 / 0.444 / 0.134 (human reference: 0.423 / 0.446 / 0.131).

ii.
```python
def lick_direction_value(bp: dict, trial_idx: int) -> int:
    if bp["no"][trial_idx]:
        return 2
    right_choice = (bp["R"][trial_idx] and bp["hit"][trial_idx]) or (bp["L"][trial_idx] and bp["miss"][trial_idx])
    left_choice = (bp["L"][trial_idx] and bp["hit"][trial_idx]) or (bp["R"][trial_idx] and bp["miss"][trial_idx])
    if left_choice:  return 0
    if right_choice: return 1
    return 2
```
```python
out[0, :] = lick_direction_value(raw["bp"], raw_trial_idx)
```
`OUTPUT_VALUES[0] = ["left", "right", "none"]`

iii. Key Decisions 9–10: outputs are represented as time-varying matrices by repeating per-trial values across bins "so all outputs share shape `(n_output, n_timepoints)` and can be decoded jointly with time-varying kinematic outputs"; lick direction is the animal's actual choice rather than the instructed side. (I re-derived the labels from the raw file for all 357 kept trials of `JEB6_2021-04-18`; they match the stored values exactly.)

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. A single per-trial flag, `obj.bp.autowater`, which marks trials where water was delivered without any auditory cue (the water-cued block).

ii.
```python
"autowater": bp["autowater"][()].reshape(-1).astype(bool),
```

iii. Step 1: "`obj.bp.autowater=1` when water was delivered regardless of animal choice… this field can be used as a proxy for obtaining water-cued blocks and delayed-response blocks of trials" (from the authors' `WorkingWithDataObjs.m` tutorial); Step 5 maps `obj.bp.autowater → output[1]`.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling, broadcast across bins: `autowater == 0 → DR (0)`, `autowater == 1 → WC (1)`. Note the code order is the reverse of the human reference (which uses WC = 0, DR = 1), but `output_values[1] = ["DR", "WC"]` matches the codes, so the labelling is self-consistent. Sessions without a water-cued block are simply all-DR. Overall distribution DR 0.901 / WC 0.099 (human reference: DR 0.903 / WC 0.097).

ii.
```python
def context_value(bp: dict, trial_idx: int) -> int:
    return 1 if bp["autowater"][trial_idx] else 0
...
out[1, :] = context_value(raw["bp"], raw_trial_idx)
```
`OUTPUT_VALUES[1] = ["DR", "WC"]`

iii. Step 5 mapping: "`autowater==0` → DR; `autowater==1` → WC; repeat across time bins… DR-only and randomized-delay sessions become all-DR trials." Step 4 notes the paper's two-context analyses use a 12-session subset but that "DR-only sessions can still be labeled as DR context for decoder purposes."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Two per-trial flags, `obj.bp.miss` and `obj.bp.hit`; everything else (i.e. `bp.no`) is treated as an ignore by construction.

ii.
```python
"hit": ..., "miss": ..., "no": ...   # all loaded; outcome uses hit and miss
```

iii. Step 5 mapping: "`obj.bp.hit`, `obj.bp.miss`, `obj.bp.no` → `output[2]` (`outcome`)… `getOutcome`, `findTrials`"; ignore trials are retained deliberately because the decoder spec asks for an `ignore` class (Key Decision 4).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A three-way relabelling broadcast across bins: miss → incorrect (0), hit → correct (1), otherwise → ignore (2). Because early trials are removed, no separate "early" class is created. Distribution 0.121 / 0.746 / 0.134 (human reference: 0.120 / 0.749 / 0.131).

ii.
```python
def outcome_value(bp: dict, trial_idx: int) -> int:
    if bp["miss"][trial_idx]: return 0
    if bp["hit"][trial_idx]:  return 1
    return 2
...
out[2, :] = outcome_value(raw["bp"], raw_trial_idx)
```
`OUTPUT_VALUES[2] = ["incorrect", "correct", "ignore"]`

iii. Step 5 mapping: "`miss` → incorrect; `hit` → correct; `no` → ignore; repeat across time bins… Early trials will be excluded entirely, so no separate 'early' class is needed." The class codes follow the decoder-task ordering (incorrect, correct, ignore).

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj{1}` — the **side camera only** — feature `tongue`: its x, y and likelihood columns (`ts[:, 0:2, featix]`), plus that view's `frameTimes` and `NdroppedFrames`. The bottom camera's tongue markers (`top_tongue`, `bottom_tongue`, …) are not used. Go cue times and the bitcode fields of `obj.sglx`/`obj.bp.ev.bitStart` are needed for the clock correction. Note the AI never applies a likelihood threshold; it relies on x/y already being NaN where tracking failed — I confirmed on the raw files that `isnan(x)` is exactly `likelihood <= 0.9`, so this is equivalent to the reference's likelihood cut.

ii.
```python
tongue_x, tongue_y, tongue_vis = align_feature(raw, time_axis, align_times, 0, "tongue")
```
```python
ts = np.asarray(trial["ts"][:, :2, feat_idx], dtype=float)
frame_times = np.asarray(trial["frameTimes"], dtype=float).reshape(-1)
...
vis = np.isfinite(x) & np.isfinite(y)
```

iii. Step 5 mapping: "Raw tongue DLC trajectories from side view (`tongue`) → `output[3]`… Use raw visibility before any tongue NaN replacement so 'not visible' is preserved" (Key Decision 11). The side-view `tongue` is the first feature in the reference `params.traj_features`. No justification is given for not also using the bottom-camera tongue markers.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Three steps, transcribed from the reference `findPosition.m` / `findVelocity.m`. (1) Frame times are put on the go-cue clock and x, y are **linearly interpolated onto the 500 bin centres**, interpolating only within contiguous runs of tracked frames so that gaps stay NaN (`interp_preserve_nan`). Tongue coordinates are *not* smoothed (the reference explicitly skips smoothing for tongue features). (2) Speed = `hypot(gradient(x), gradient(y))`, computed on the resampled grid with unit spacing (no division by dt — same as the MATLAB `gradient(...)`), with **no** baseline subtraction and **no** NaN filling, because those are applied only to non-tongue features in the reference. (3) The per-bin speed is discretised (7-c).

Implementation defect: because `np.gradient` is a centred difference, a NaN neighbour poisons the sample next to it, so the first and last bin of every visible run of the tongue loses its speed and is reported as `not visible`. Measured on `JEB6_2021-04-18`: 11.34 % of bins have a visible tongue, but only 7.55 % survive with a finite speed — a third of the tracked tongue bins are mislabelled. Dataset-wide the tongue is `not_visible` in 95.6 % of bins, versus 87.5 % for the human reference (single camera + this edge loss).

ii.
```python
x = interp_preserve_nan(frame_times - vidshift - align_times[tr_idx], ts[:, 0], time_axis + ADVANCE_MOVEMENT)
y = interp_preserve_nan(frame_times - vidshift - align_times[tr_idx], ts[:, 1], time_axis + ADVANCE_MOVEMENT)
vis = np.isfinite(x) & np.isfinite(y)
if not is_tongue:                 # tongue keeps its NaNs
    x = fill_nearest_1d(x); y = fill_nearest_1d(y)
```
```python
xvel = np.gradient(xx); yvel = np.gradient(yy)
if not is_tongue:
    ...  # baseline subtraction + nearest fill, non-tongue only
speed[:, tr_idx] = np.sqrt(xvel ** 2 + yvel ** 2)
```

iii. Step 1: "Non-tongue DLC features are smoothed and nearest-filled when coordinates are missing. Tongue coordinates preserve invisibility periods; tongue position NaNs are later replaced with a baseline position while tongue velocity NaNs are set to zero. This is relevant for translating 'not visible' into decoder outputs." Key Decision 11: keep the raw visibility instead of the reference's NaN→0 substitution, because the decoder spec requires an explicit `not visible` class.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per session, the 50th percentile of the tongue speed is taken over all bins of all kept trials where the tongue is visible and the speed is finite; bins below it get class 0 (`low`), bins at or above get class 1 (`high`), and every other bin — not visible, or visible but with a NaN speed — gets class 2 (`not_visible`). The threshold is stored in `metadata['session_info'][i]['thresholds']`. Dataset-wide: 0.022 / 0.022 / 0.956.

ii.
```python
def discretize_visible_signal(values, visible):
    out = np.full(values.shape, 2, dtype=np.int64)
    valid_vals = values[visible & np.isfinite(values)]
    if valid_vals.size == 0:
        return out, float("nan")
    thresh = float(np.nanpercentile(valid_vals, 50))
    low_mask = visible & np.isfinite(values) & (values < thresh)
    high_mask = visible & np.isfinite(values) & ~low_mask
    out[low_mask] = 0; out[high_mask] = 1
    return out, thresh
```

iii. Key Decision 12: "Use session-specific 50th-percentile thresholds for tongue velocity, paw velocity, and motion energy: This follows the decoder-task specification and is applied after reference-style temporal alignment." The 0/1/2 coding follows the decoder-task spec verbatim.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The camera clock leads the behaviour clock, so a session-constant offset is removed first: `vidshift = mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)`, exactly the reference `findVideoOffset.m`. Frame times then become `frameTimes − vidshift − goCue[trial]`, and the tracked coordinates are interpolated onto the same 500 bin centres used for the spikes (`+ ADVANCE_MOVEMENT`, set to 0.0 as in `getDefaultParams.m`). Sanity check on the converted data: across trials the fraction of bins with a visible tongue peaks 0.14–0.23 s *after* the go cue, as expected for a lick response.

ii.
```python
def find_video_offset(raw: dict) -> float:
    bit_start = mode_float(raw["bp"]["ev"]["bitStart"])
    vid_file_offset = mode_float(raw["sglx_bitstart"]) / raw["sglx_fs"]
    return vid_file_offset - bit_start
```
```python
x = interp_preserve_nan(frame_times - vidshift - align_times[tr_idx], ts[:, 0], time_axis + ADVANCE_MOVEMENT)
```

iii. Step 1: "Alignment to video is explicit: trajectories use `frameTimes - vidshift - alignTimes(trial)`, where `vidshift = mode(obj.sglx.bitcode.bitstart)/obj.sglx.fs - mode(obj.bp.ev.bitStart)`." Step 10's reference-comparison table records "Same go-cue alignment and same video offset logic."

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The DeepLabCut tracking of the **bottom camera** (`obj.traj{2}`), using **both** paw markers, `top_paw` and `bottom_paw` (x, y, likelihood), plus that view's `frameTimes`/`NdroppedFrames` and the same bitcode/go-cue timing fields.

ii.
```python
paw_feats = ["top_paw", "bottom_paw"]
paw_speeds, paw_vis = [], []
for feat in paw_feats:
    x, y, vis = align_feature(raw, time_axis, align_times, 1, feat)
    paw_speeds.append(compute_speed(x, y, feat)); paw_vis.append(vis)
```

iii. Step 5 mapping: "Raw bottom-view paw DLC trajectories (`top_paw`, `bottom_paw`) → `output[4]`… aggregate as max visible paw speed… Paws are only tracked in bottom view per methods." Both markers are in the reference `params.traj_features` for camera 1.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The non-tongue branch of the reference kinematics, per marker: interpolate x, y onto the 500 bin centres, nearest-fill missing coordinates, take `gradient` of each, subtract the trial's baseline drift (`median(diff(...))`, using the x-component for both axes — faithfully reproducing the quirk in the MATLAB `findVelocity.m`), nearest-fill the velocities, then speed = `hypot(xvel, yvel)`. The two markers are then combined by taking the element-wise **maximum** of their speeds, with the bin marked visible if *either* marker was tracked. No cross-camera normalisation is needed (single view). `not_visible` ends up at 5.9 % of bins (human reference 18.6 %, which uses `top_paw` alone).

ii.
```python
xvel = np.gradient(xx); yvel = np.gradient(yy)
if not is_tongue:
    diffs = np.column_stack([np.diff(xx), np.diff(yy)])
    basederiv = np.nanmedian(diffs, axis=0) if np.isfinite(diffs).any() else np.array([0.0, 0.0])
    baseline = basederiv[0] if np.isfinite(basederiv[0]) else 0.0
    xvel = fill_nearest_1d(xvel - baseline); yvel = fill_nearest_1d(yvel - baseline)
speed[:, tr_idx] = np.sqrt(xvel ** 2 + yvel ** 2)
```
```python
paw_stack = np.stack(paw_speeds, axis=0)
paw_finite = np.isfinite(paw_stack)
paw_speed = np.max(np.where(paw_finite, paw_stack, -np.inf), axis=0)
paw_speed[~paw_finite.any(axis=0)] = np.nan
paw_visible = np.any(np.stack(paw_vis, axis=0), axis=0)
```

iii. Step 1: "`findVelocity` computes per-feature x/y velocities; non-tongue NaNs are filled nearest, tongue NaNs are converted to zero velocity when not visible." Step 5: aggregate the two markers "as max visible paw speed", with class 2 only "when neither paw marker visible".

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The same `discretize_visible_signal` as the tongue: per-session 50th percentile of the combined paw speed over all visible, finite bins; `< threshold` → 0, `>= threshold` → 1, everything else → 2 (`not_visible`). Threshold recorded per session. Dataset-wide: 0.473 / 0.468 / 0.059.

ii.
```python
paw_cat, paw_thr = discretize_visible_signal(paw_speed, paw_visible)
...
thresholds = {"tongue_velocity_median": tongue_thr, "paw_velocity_median": paw_thr,
              "motion_energy_median": me_thr}
```

iii. Key Decision 12 (session-specific 50th-percentile thresholds, per the decoder-task spec) and Step 5's "class 2 when neither paw marker visible".

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue, but using the bottom camera's own `frameTimes`: subtract the session video offset and the trial's go cue, then interpolate onto the same 500 bin centres as the spikes. Because each feature is aligned through the frame times of the view it belongs to, the two cameras' frame counts do not have to agree.

ii.
```python
x, y, vis = align_feature(raw, time_axis, align_times, 1, feat)   # view 1 = bottom camera
...
x = interp_preserve_nan(frame_times - vidshift - align_times[tr_idx], ts[:, 0], time_axis + ADVANCE_MOVEMENT)
```

iii. Same as 7-d — Step 1's `frameTimes - vidshift - alignTimes(trial)` rule and Step 10's "Same go-cue alignment and same video offset logic".

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` sitting beside each session file: one trace per trial at the camera frame rate. The MATLAB wrapper struct is unwrapped recursively (`me` → `me.data` → `me.data.data`) until the per-trial cell array is reached; `moveThresh` is read but not used. The embedded `obj.me` copy is not read. If the file were missing, motion energy would be all class 2 — this never happens, all 44 sessions have one.

ii.
```python
def load_motion_energy(spec: SessionSpec):
    me_path = spec.data_path.with_name(f"motionEnergy_{spec.subject}_{spec.date}.mat")
    if not me_path.exists():
        return None, None
    me = sio.loadmat(me_path, struct_as_record=False, squeeze_me=True)["me"]
    payload = me
    while hasattr(payload, "_fieldnames") and "data" in payload._fieldnames:
        if hasattr(payload, "moveThresh"):
            move_thresh = float(np.asarray(payload.moveThresh).reshape(-1)[0])
        next_payload = payload.data
        if next_payload is payload: break
        payload = next_payload
    return [np.asarray(t, dtype=float).reshape(-1) for t in np.ravel(payload)], move_thresh
```

iii. Step 1: "`loadMotionEnergy` loads motion-energy traces, aligns them to the session time axis with interpolation, fills edge NaNs, and thresholds movement epochs"; Step 10 edge-case list: "Handled nested `motionEnergy.me.data.data` struct variants" (the reference has the same guard, `if isstruct(me.data), me.data = me.data.data; end`). Step 4 had planned to also fall back to the embedded `obj.me`; the script only reads the standalone files, which is moot because every session has one.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond resampling: the value is already one scalar per camera frame (the paper computes it per pixel as a median difference over neighbouring frames, reduced to the 99th percentile across pixels). The AI linearly interpolates the trace onto the 500 bin centres and then nearest-fills the residual NaNs at the trial edges — a direct copy of the reference's `interp1(...)` + `fillmissing(me.data,'nearest')`. As a consequence, the `no_video` class is never used (0 % of bins; the human reference has 3.8 %, keeping edge bins uncovered by frames as class 2).

ii.
```python
sig = interp_numeric(tt, y, time_axis + ADVANCE_MOVEMENT)
aligned[:, tr_idx] = fill_nearest_1d(sig)
```
```python
def fill_nearest_1d(x):
    idx = np.flatnonzero(np.isfinite(x)); missing = np.flatnonzero(~np.isfinite(x))
    if missing.size: x[missing] = np.interp(missing, idx, x[idx])
    return x
```

iii. Step 1: "Motion energy is resampled onto the same aligned time grid as neural data and then `fillmissing(...,'nearest')` is used for edge NaNs." Step 5: "Use raw aligned motion energy, not the paper's manual movement threshold, because task specifies a 50th-percentile discretization."

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per session, the 50th percentile over all finite aligned samples; `< threshold` → 0 (`low`), `>= threshold` → 1 (`high`), NaN → 2 (`no_video`). Because of the nearest-fill in 9-b, nothing is NaN in practice, so only classes 0 and 1 occur (0.508 / 0.492); class 2 would only appear for a session with no motion-energy file at all. The session's manually chosen `moveThresh` from the file is deliberately ignored in favour of the task-specified median split.

ii.
```python
def discretize_motion_energy(me_aligned):
    if me_aligned is None: return None, float("nan")
    out = np.full(me_aligned.shape, 2, dtype=np.int64)
    valid = np.isfinite(me_aligned); vals = me_aligned[valid]
    if vals.size == 0: return out, float("nan")
    thresh = float(np.nanpercentile(vals, 50))
    out[valid & (me_aligned < thresh)] = 0
    out[valid & (me_aligned >= thresh)] = 1
    return out, thresh
...
if me_cat is None:
    me_cat = np.full((time_axis.size, nraw), 2, dtype=np.int64)
```
`OUTPUT_VALUES[5] = ["low", "high", "no_video"]`

iii. Key Decision 12 and the Step 5 note above: the decoder task prescribes a 50th-percentile split, so the paper's manually chosen bimodal movement threshold is not used; class 2 is reserved for "sessions lacking video" (Step 7: "the reserved class `2` is still present in `output_values` for sessions lacking video").

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same offset and same grid as the tracking, using the **side camera's** frame times (motion energy has one value per side-camera frame): `frameTimes − vidshift − goCue[trial]`, interpolated onto the 500 bin centres. If a trial's `frameTimes` are absent or contain NaNs, the AI falls back to synthetic frame times `(1:n)/400` with a fixed 0.5 s shift — a literal transcription of the `catch` branch in the reference `loadMotionEnergy.m`. Sanity check on the converted data: the fraction of `high` motion-energy bins peaks ~0.4 s after the go cue and is low during the delay.

ii.
```python
trial = raw["traj"][0][tr_idx]                       # side camera
frame_times = np.asarray(trial["frameTimes"], dtype=float).reshape(-1)
if frame_times.size == 0 or not np.all(np.isfinite(frame_times)):
    frame_times = (np.arange(y.size) + 1) / 400.0
    tt = frame_times - 0.5 - align_times[tr_idx]
else:
    tt = frame_times - vidshift - align_times[tr_idx]
sig = interp_numeric(tt, y, time_axis + ADVANCE_MOVEMENT)
```

iii. Step 1's description of `loadMotionEnergy`, and Step 10's raw-file sanity check: "Independently reimplemented motion-energy alignment, nearest-fill, session-median thresholding, and class assignment for the first kept trial. Result: `np.allclose(...) == True`."

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Every case is handled by keeping the trial and marking or filling the gap; no trial is dropped for video problems.
- **Bad video trial** (`NdroppedFrames` is NaN): the trial is skipped in `align_feature`, leaving its tongue and paw as all-NaN → 500 bins of `not visible` (same rule as the reference `findPosition.m`).
- **Missing/NaN frame times**: synthetic 400 Hz frame times are generated (`(arange(n)+1)/400`), matching the reference fallbacks.
- **Untracked frames** (DLC x/y NaN, equivalently likelihood ≤ 0.9): non-tongue features are nearest-filled (positions and velocities) as in the reference; the tongue keeps its NaNs and those bins become `not visible`.
- **Missing motion-energy samples**: nearest-filled; a missing motion-energy *file* would give class 2 throughout.
- **Trials past the end of the recording**: detected as all-zero neural matrices and dropped (61 trials in 2 sessions).
- **File-format heterogeneity**: v7.3 vs v5 layouts, `(n,1)` vs `(1,n)` cell arrays, missing `obj.ex`, `ex.probe.loc` stored as raw char, `site` vs `channel`, nested `me.data.data` — all handled with explicit fallbacks.

ii.
```python
ndropped = trial["NdroppedFrames"]
if np.isnan(ndropped):
    continue                       # leaves this trial NaN -> 'not visible'
frame_times = np.asarray(trial["frameTimes"], dtype=float).reshape(-1)
if frame_times.size == 0:
    frame_times = (np.arange(ts.shape[0]) + 1) / 400.0
```
```python
site_field = "site" if "site" in probe_target else ("channel" if "channel" in probe_target else None)
...
if len(probe_locs) < len(clu_probes):
    probe_locs.extend(["ALM"] * (len(clu_probes) - len(probe_locs)))
```

iii. Step 10 "Edge-case review" enumerates each of these fixes, and Key Decision 11 explains why tongue/paw gaps are preserved rather than filled: the decoder task needs an explicit invisibility class, so "Derive 'not visible' from raw DLC visibility, not from post-filled trajectories."

## 11-a. What are the most time-consuming steps of the code?

i. The AI reports only session-level timing: 3.69 s per session on average, 162 s for the full 44-session conversion, which it judged well inside the 15-minute budget, so no profiling of individual steps was done. In fact the dominant cost is reading the `.mat` files — `load_session_v73` dereferences one HDF5 cell per trial per field and materialises *every* DLC feature of *both* cameras for all ~300 trials, plus every cluster's `tm`/`trialtm`/`trial` arrays. Everything afterwards (binning, smoothing, interpolation, discretisation) is comparatively cheap. Sessions with more trials/units take proportionally longer (1.6 s–6.7 s range).

ii.
```python
t0 = time.perf_counter()
raw = load_raw_session(spec.data_path)
...
info["elapsed_sec"] = time.perf_counter() - t0
```
```python
print(f"Mean session processing time: {np.mean(timings):.2f}s")
print(f"Estimated total time at this rate for full curated set: {np.mean(timings) * len(parse_manifest_sessions()):.2f}s")
```

iii. Step 6/Step 7: "Conversion runtime on the sample run after fixes: 3.58 s for `JEB6_2021-04-18`, 2.37 s for `JEB23_2023-10-18`… Estimated full curated conversion time at current speed: ~131 s for 44 sessions." No step-level bottleneck analysis is documented.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain, and the notes overstate how vectorised the code is ("Kept neural and video processing vectorized within trials wherever possible, with only session-level loops"). The remaining loops are: the per-unit loop in `bin_session_neural` (a single `np.add.at`/`histogram2d` over all units at once would be possible); the per-column `np.convolve` loop in `smooth_causal_reflect` (replaceable by one `scipy.ndimage.convolve1d` on the 2-D array); the per-trial loops in `align_feature`, `compute_speed`, and `align_motion_energy`; the per-trial Python membership test `np.array([t in trial_map for t in trial_idx])` (`np.isin` would do this in C); and the per-trial/per-field HDF5 dereferencing loops in the loader. The per-trial video loops are hard to vectorise because each trial has a different number of frames — the same reason the human reference kept them. Given that I/O dominates, none of these materially change the 162 s runtime.

ii.
```python
for unit_idx, unit in enumerate(units):
    ...
    valid_mask = np.array([t in trial_map for t in trial_idx], dtype=bool)
    ...
    fr = smooth_causal_reflect((counts / DT).T, SMOOTH_N, BCTYPE).T
```
```python
out = np.zeros_like(padded, dtype=float)
for col in range(padded.shape[1]):
    out[:, col] = np.convolve(padded[:, col], kern, mode="same")
```

iii. Step 6: "Code speedups added: … Kept neural and video processing vectorized within trials wherever possible, with only session-level loops." The AI judged further optimisation unnecessary once the estimated full runtime (~131 s) was far below the 15-minute threshold in the instructions.

## 11-c. What processing does the code repeat multiple times?

i. A few cheap recomputations: `find_video_offset(raw)` is recomputed on every call of `align_feature` (three times: tongue, `top_paw`, `bottom_paw`) and again in `align_motion_energy`, i.e. four times per session instead of once; `parse_manifest_sessions()` is called three times in `main` (once for the session list, twice more inside print statements); `build_time_axis()` is rebuilt per session and the bin edges are rebuilt again inside `bin_session_neural`; and the kinematics/motion energy are computed for **all** raw trials of a session (`range(ntrials)`) even though only the surviving trials are exported. None of these is expensive relative to file loading, and no data is loaded twice.

ii.
```python
def align_feature(raw, time_axis, align_times, view_idx, feat_name):
    ntrials = raw["bp"]["Ntrials"]
    vidshift = find_video_offset(raw)          # recomputed on every call
    ...
    for tr_idx in range(ntrials):              # all trials, including dropped ones
```
```python
print(f"Curated sessions available: {len(parse_manifest_sessions())}")
...
print(f"Estimated total time at this rate for full curated set: {np.mean(timings) * len(parse_manifest_sessions()):.2f}s")
```

iii. Not discussed in CONVERSION_NOTES.md. The AI's efficiency notes (Step 6/7) only cover the loader-orientation fix and the overall runtime estimate.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Mostly in loading. `load_session_v73`/`load_session_v5` materialise the full DLC tensor, `featNames` and `frameTimes` for **every** feature of **both** cameras on every trial, though only `tongue`, `top_paw`, `bottom_paw` and the frame times are used; they also read every cluster's whole-session spike times `tm`, the `site`/`channel` index, and the behavioural fields `ev.sample`, `ev.delay`, `ev.reward`, `ev.lickL`, `ev.lickR`, `bp.L`, `bp.no` (partly used) — none of which reach the output. `moveThresh` is parsed from the motion-energy file and never used. `selected_units` builds a per-unit `unit_regions` list which `build_dataset` then ignores in favour of a hard-coded `["ALM"]`. Kinematics and motion energy are computed for trials that are subsequently dropped. Everything computed after that point does end up in the pickle.

ii.
```python
for tr_idx in range(ntrials):
    feat_outer = h5_deref(f, h5_cell_ref(view["featNames"], tr_idx))
    ...
    ts = np.asarray(h5_deref(f, h5_cell_ref(view["ts"], tr_idx))[()], dtype=float)   # all features
```
```python
unit_regions = [unit_regions[i] for i in keep_unit_idx.tolist()]   # stored per session...
...
brain_regions = ["ALM"]                                            # ...but never used
"brain_region_idx": [np.zeros(sess["neural"][0].shape[0], dtype=np.int64) for sess in processed_sessions],
```

iii. Not discussed in CONVERSION_NOTES.md. The region hard-coding is implicitly justified by Step 3 ("ALM recordings only for the electrophysiology analyses described in the paper") and Step 2 ("Brain region metadata is stored in `obj.ex.probe.loc`; representative ephys sessions indicate `R ALM`"), which the AI normalises to `ALM`.
