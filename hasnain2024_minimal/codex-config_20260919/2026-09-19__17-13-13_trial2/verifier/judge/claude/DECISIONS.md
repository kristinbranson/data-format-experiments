# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not load the whole shared dataset. It hard-codes a tuple of **12 sessions** (animal, date, ALM probe number) transcribed from the authors' `load<ANM>_ALMVideo.m` scripts *as they are called from* `Scripts/Figure 8/Figure8a_thru_c.m` — i.e. only the seven animals used in the paper's two-context (DR/WC) figure (JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, JEB19). Only `/app/data/Ephys_Behavior` is searched (the `--data-dir` default); the 19–21 `RandomizedDelay_Ephys_Behavior` sessions and the 13 other fixed-delay sessions (JEB13/14/15) are never opened, and neither are the `*_BilatMC_Behavior` photoinhibition folders. Each session is one `data_structure_<anm>_<date>.mat`, opened once with `h5py` (all 12 of the chosen files are MATLAB v7.3); the companion `motionEnergy_<anm>_<date>.mat` is a v5 file and is read with `scipy.io.loadmat`. There is no `scipy.io` fallback for the data structure itself, so the v5 `data_structure` files that exist in the randomized-delay folder could not have been read by this code even if they were listed. Within a session, one probe's clusters are used (`obj/clu{probe}`), the behaviour table is `obj/bp`, the DeepLabCut tracking is `obj/traj`, and the clock alignment comes from `obj/sglx`.

ii.
```python
SESSIONS = (
    ("JEB6", "2021-04-18", 2),
    ("JEB7", "2021-04-29", 1),
    ...
    ("JEB19", "2023-04-18", 1),
)

for session_index, (animal, date, probe) in enumerate(SESSIONS, start=1):
    stem = f"{animal}_{date}"
    data_path = data_dir / f"data_structure_{stem}.mat"
    motion_path = data_dir / f"motionEnergy_{stem}.mat"
    if not data_path.exists() or not motion_path.exists():
        raise FileNotFoundError(f"Missing source data for {stem}")

    with h5py.File(data_path, "r") as handle:
        behavior = _load_behavior(handle)
        neural_all, qualities = _load_neural(handle, probe, behavior, edges)
```

```python
parser.add_argument(
    "--data-dir",
    type=Path,
    default=Path(__file__).resolve().parent / "data" / "Ephys_Behavior",
)
```

iii. From the trajectory (step 9): *"The archive contains several experiment families, but only the two-context electrophysiology sessions jointly provide neural activity, WC/DR labels, kinematics, and motion energy. I'm narrowing to those sessions…"* and (step 28): *"The repository's Figure 8 pipeline resolves the key ambiguity: its 12 two-context sessions, `[-3.0, 2.5)` s window, 10 ms bins, causal 15-bin Gaussian smoothing, quality exclusions, and >1 Hz filter reproduce the paper's reported 522 ALM units exactly."* The AI reasoned that because *behavioral context* (WC/DR) is one of the required decoder outputs, the correct reference analysis is Figure 8 (the CDContext analysis), and it adopted that script's session list, probe assignment, window, bin size, smoothing and curation wholesale. It confirmed the choice numerically: the paper states "two-context paradigm: 12 sessions, six mice, 522 units", and its pipeline yields exactly 12 sessions and 522 units.

## 1-b. How are the data split into subjects?

i. The subject is the animal string in the hard-coded session tuple (equivalently, the part of the file stem before the underscore). `subjects` is the list of unique animals in first-appearance order, and `subject_idx` maps each session onto that list. This gives **7 subjects** over 12 sessions (JEB19 contributes 4 sessions, JEB7 and JGR2 two each, the rest one each). The animal id inside the file (`obj.meta.anm`) is never read.

ii.
```python
subjects = list(dict.fromkeys(session[0] for session in SESSIONS))
...
"subjects": subjects,
"subject_idx": np.asarray(
    [subjects.index(animal) for animal, _, _ in SESSIONS], dtype=np.int64
),
```
and per session:
```python
session_info.append({"subject": animal, "date": date, "source_file": data_path.name, ...})
```

iii. Not discussed explicitly in the trajectory. Implicitly, the animal is already an explicit field of the session table the AI transcribed from the authors' `load<ANM>_ALMVideo.m` files, so no parsing or lookup inside the `.mat` was needed.

## 1-c. How are the data split into sessions?

i. One session = one entry of `SESSIONS` = one `data_structure_*.mat` file = one element of `neural`/`input`/`output`/`brain_region_idx`. Twelve sessions in total, all fixed-delay two-context recordings, each contributing 210–390 trials (3,116 trials total). Exactly one probe per session is used — the probe number given by the authors' loading scripts (probe 2 for JEB6/EKH1/EKH3, probe 1 for the rest); the second probe of two-probe sessions is not concatenated, matching the authors' `params.probe`. Sessions are never merged or split.

ii.
```python
for session_index, (animal, date, probe) in enumerate(SESSIONS, start=1):
    ...
    neural_sessions.append(neural_trials)
    input_sessions.append(input_trials)
    output_sessions.append(output_trials)
    region_indices.append(np.zeros(neural_all.shape[1], dtype=np.int64))
```
and the probe selection:
```python
probe_group = _deref(handle, handle["obj/clu"], probe - 1)
```

iii. Step 28: the 12 sessions and their probes come from the Figure 8 loading scripts. The AI verified its session/probe list against `code/DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m` (step 22), which is the same record the authors use, and cross-checked the result against the paper's stated "12 sessions … 522 units".

## 1-d. How are the data split into trials?

i. Trials are the rows of the per-trial `obj.bp` table. The trial count is taken as the length of `obj.bp.ev.goCue`, and every behavioural flag (`R`, `L`, `hit`, `miss`, `no`, `early`, `autowater`, `stim.enable`) is read as a same-length vector. Spikes carry their own 1-based trial index (`clu.trial`), and the DLC/motion-energy streams are stored as one cell per trial, so no trial boundaries have to be reconstructed. `obj.bp.Ntrials` is not read; the go-cue length is used instead (these agree in all 12 sessions, and no bp field is stored longer than `Ntrials` in these files).

ii.
```python
def _load_behavior(handle: h5py.File) -> dict[str, np.ndarray]:
    bp = handle["obj/bp"]
    result = {
        name: _array(bp[name]).astype(bool)
        for name in ("R", "L", "hit", "miss", "no", "early", "autowater")
    }
    result["stim"] = _array(bp["stim/enable"]).astype(bool)
    result["go_cue"] = _array(bp["ev/goCue"]).astype(np.float64)
    result["ntrials"] = np.asarray([len(result["go_cue"])], dtype=np.int64)
    return result
```
Per-trial indexing of the neural and video streams:
```python
aligned = spike_time - go_cue[spike_trial - 1]
...
for trial in range(ntrials):
    frames = _array(_deref(handle, group["frameTimes"], trial))...
```

iii. Not discussed explicitly. The Bpod table has one entry per trial and exactly one go cue per trial, so the trial axis is given directly by the data; the AI's exploration (step 22) printed per-session `Ntrials` alongside the per-flag totals and confirmed consistency before writing the converter.

## 1-e. How are trials filtered based on quality controls?

i. Two filters, applied jointly as one boolean `keep` mask: **early-lick trials** (`bp.early`) and **photostimulation trials** (`bp.stim.enable`) are dropped. Ignore (no-response) trials are deliberately **retained**, because "ignore" is a required class of the `outcome` output. Nothing else is filtered — no minimum-trial-count, no reaction-time cut, and no check that the trial falls inside the electrophysiology recording. 3,116 of 3,926 trials survive (210–390 per session). The per-session kept-trial counts are identical to the expert's for the 12 shared sessions (e.g. EKH1 252, EKH3 390, JEB6 302). The `keep` mask is also what defines the population used for the per-session movement thresholds; the neural rates and the velocity traces are computed for *all* trials and only subset at the end.

ii.
```python
# Paper analyses omit early licks and use no-photostimulation trials.
# Ignore trials are deliberately retained for the required output.
keep = ~behavior["early"] & ~behavior["stim"]
...
trial_ids = np.flatnonzero(keep)
for trial in trial_ids:
    neural_trials.append(neural_all[trial].copy())
```
and in the metadata:
```python
"trial_filter": "no photostimulation and no early lick; ignore trials retained",
```

iii. Step 28: *"exclude photostimulation and early-lick trials as the analyses do, while retaining ignore trials because the requested outcome explicitly requires that class."* The repository's Figure 8 conditions (`hit&~stim.enable&~autowater&~early`, etc.) are the source of the two exclusions; the retention of ignore trials is an explicit, stated departure driven by the decoder specification.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` for the single ALM probe of each session: `clu.trial` (1-based trial index of each spike), `clu.trialtm` (spike time relative to that trial's start, on the behaviour clock) and `clu.quality` (the manual curation string). `obj.bp.ev.goCue` supplies the alignment times, and the seven Figure-8 trial-condition masks (built from `hit`, `miss`, `no`, `stim.enable`, `autowater`, `early`) are used for the low-firing-rate criterion.

ii.
```python
probe_group = _deref(handle, handle["obj/clu"], probe - 1)
...
for unit in range(probe_group["quality"].shape[0]):
    quality = _string(_deref(handle, probe_group["quality"], unit))
    ...
    spike_trial = _array(_deref(handle, probe_group["trial"], unit)).astype(np.int64)
    spike_time = _array(_deref(handle, probe_group["trialtm"], unit)).astype(np.float64)
    aligned = spike_time - go_cue[spike_trial - 1]
```

iii. Not stated as prose, but the AI traced `alignSpikes.m` / `getSeq.m` (steps 10, 17, 20) and reproduced the fields they use. `trialtm` is already relative to trial start on the behaviour clock, so it is directly comparable to `bp.ev.goCue`.

## 2-b. How is the `neural` data processed?

i. Per unit, spikes are histogrammed into 10 ms bins spanning −3 to +2.5 s from the go cue for every trial, divided by the bin width to give spikes/s, and then smoothed along time with the repository's **causal** Gaussian kernel: `gausswin(15)` with its first 7 taps zeroed and renormalised, applied with the repository's "reflect" boundary handling (which in `mySmooth.m` actually *prepends a copy of the first 15 samples*, not a mirror). No normalisation, baseline subtraction or z-scoring. Rates are stored as `float32` in Hz, shape `(n_neurons, 550)` per trial. Only one probe per session contributes.

ii.
```python
def _causal_smooth(rates: np.ndarray) -> np.ndarray:
    """Match utils/mySmooth.m with N=15 and boundary type 'reflect'."""
    # MATLAB gausswin(N) uses alpha=2.5, equivalent to std=N/(2*alpha).
    kernel = gaussian(SMOOTH_BINS, std=SMOOTH_BINS / 5.0)
    kernel[: SMOOTH_BINS // 2] = 0.0  # causal operation in mySmooth.m
    kernel /= kernel.sum()

    # Despite its name, the repository's 'reflect' option prepends a copy of
    # the first N bins. Convolution is along time (axis 1) for all trials.
    padded = np.concatenate((rates[:, :SMOOTH_BINS], rates), axis=1)
    smoothed = signal.convolve(padded, kernel[None, :], mode="same")
    return smoothed[:, SMOOTH_BINS:]
```
```python
bin_index = np.floor((aligned - TMIN) / DT).astype(np.int64)
valid = (bin_index >= 0) & (bin_index < len(edges) - 1)
counts = np.zeros((ntrials, len(edges) - 1), dtype=np.float64)
np.add.at(counts, (spike_trial[valid] - 1, bin_index[valid]), 1.0)
rates = _causal_smooth(counts / DT)
```

iii. The parameters are copied from `Scripts/Figure 8/Figure8a_thru_c.m` (`params.smooth = 15`, `params.bctype = 'reflect'`, `params.dt = 1/100`, `params.tmin = -3`, `params.tmax = 2.5`). The AI read `utils/mySmooth.m` directly (step 17) and reproduced both the causal zeroing of the kernel's first half and the non-mirroring "reflect" padding, documenting the latter in a comment because the option's name is misleading. Step 28 records that this pipeline reproduces the paper's 522 units exactly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both copied from the repository. **(1)** Cluster quality: the label is read, whitespace-stripped, and **case-sensitively** compared against `{'garbage','gabrga','noisy','real?'}` — exactly the exclusion list in the `'all'` branch of `findClusters.m`. Every other label survives, including `Poor`/`poor`, `Multi`, and the handful of units whose label is empty/`\x00`. (Because the comparison is case-sensitive, the one unit labelled `Noisy` in JEB6 is kept — which is also what the MATLAB `ismember` does.) **(2)** Low firing rate: for each surviving unit, a PSTH is built for each of the seven Figure-8 conditions, and the unit is kept only if the **mean of those condition PSTHs over conditions and time exceeds 1 Hz** — i.e. `removeLowFRClusters.m`'s `mean(mean(psth,3,'omitnan'),'omitnan') > lowFR`, with conditions weighted equally rather than by trial count. Conditions with no trials are skipped. The result is **522 units** over 12 sessions (27–67 per session), matching the unit count the paper reports for the two-context dataset.

ii.
```python
exclusions = {"garbage", "gabrga", "noisy", "real?"}
...
quality = _string(_deref(handle, probe_group["quality"], unit))
# Match findClusters.m exactly: labels are stripped but case-sensitive.
if quality in exclusions:
    continue
...
# removeLowFRClusters.m averages condition PSTHs equally (rather than
# weighting conditions by their trial counts) and ignores empty cells.
condition_means = [rates[mask].mean(axis=0) for mask in conditions if np.any(mask)]
mean_rate = float(np.mean(np.stack(condition_means)))
if mean_rate > LOW_FR_HZ:
    neurons.append(rates.astype(np.float32))
```
```python
def _figure8_conditions(behavior):
    return [
        hit | miss | no,
        hit & ~stim & ~wc,
        hit & ~stim & wc,
        miss & ~stim & ~wc,
        miss & ~stim & wc,
        hit & ~stim & ~wc & ~early,
        hit & ~stim & wc & ~early,
    ]
```

iii. Step 28: *"…quality exclusions, and >1 Hz filter reproduce the paper's reported 522 ALM units exactly."* The AI first tried a naive "mean rate over the window > 1 Hz" criterion (step 22/23), which gave 486 units, then traced `removeLowFRClusters.m` and found that the criterion is computed on the *condition-averaged PSTHs*; re-running with that definition (step 25) gave exactly 522, matching the paper's "In total, 522 units … were recorded in these sessions". It used that agreement as the validation of both the curation rule and the −3 s / 10 ms window.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. A single subtraction, per spike: `trialtm − goCue[trial]`. Both are on the behaviour clock and `trialtm` is already relative to its own trial's start, so no interpolation or offset correction is needed on the neural side (only the camera streams need the bitcode clock correction). Bin 0 of every trial is therefore −3.0 to −2.99 s from the go cue, and bin 300 straddles the go cue.

ii.
```python
spike_trial = _array(_deref(handle, probe_group["trial"], unit)).astype(np.int64)
spike_time  = _array(_deref(handle, probe_group["trialtm"], unit)).astype(np.float64)
aligned = spike_time - go_cue[spike_trial - 1]
bin_index = np.floor((aligned - TMIN) / DT).astype(np.int64)
```
and in metadata:
```python
"temporal_alignment_event": "go cue onset (DR) or matched water-delivery/go-cue event (WC)",
```

iii. This is `alignSpikes.m` with `params.alignEvent = 'goCue'`, which the AI read while tracing `processData.m`/`loadSessionData.m` (step 10). The instructions also mandate go-cue alignment. The metadata note reflects that in WC trials `bp.ev.goCue` marks the matched water-delivery event rather than an audible chirp.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **10 ms bins** (`DT = 0.01`, i.e. the Figure-8 script's `params.dt = 1/100`), over the window **−3.0 to +2.5 s** relative to the go cue, giving **550 bins per trial** for every trial in every session. Bin centres run from −2.995 to +2.495 s. Spikes are binned once at this resolution, so there is **no rebinning** — the 10 ms grid is the native grid of the conversion. All other streams (the time input, the DLC velocities and motion energy) are put on exactly the same 550-bin grid, so a bin index means the same interval in every stream.

ii.
```python
TMIN = -3.0
TMAX = 2.5
DT = 0.01
...
edges = np.arange(TMIN, TMAX + DT / 2, DT, dtype=np.float64)
time = (edges[:-1] + DT / 2).astype(np.float32)
```
```python
"time_bin_size": DT * 1000.0,
"off_start": TMIN,
"off_end": TMAX,
```

iii. Step 28 cites the Figure 8 `[-3.0, 2.5)` s window and 10 ms bins as directly adopted from `Figure8a_thru_c.m`. The AI also tested the four combinations of `tmin ∈ {−3, −2.5}` and `dt ∈ {0.01, 0.005}` (steps 23/25) and observed that the surviving unit count is insensitive to `dt` but that `tmin = −3` yields the paper's exact 522 (vs 521 at −2.5), which it used as a tie-breaker in favour of the Figure-8 values.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not derived from a raw variable — it is the conversion's own time axis, the centres of the 550 bins of the −3 to +2.5 s window around each trial's `obj.bp.ev.goCue`. The same 550-value vector is used for every trial of every session.

ii.
```python
edges = np.arange(TMIN, TMAX + DT / 2, DT, dtype=np.float64)
time = (edges[:-1] + DT / 2).astype(np.float32)
...
"input_names": ["time from go cue onset (s)"],
```

iii. Not discussed; the decoder specification asks for "Time from go cue onset in seconds (continuous, time-varying)", and the go cue is the alignment event, so the axis is defined by the window rather than read from the data.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond taking the bin centres. The values are stored in seconds as a `float32` ramp from −2.995 to +2.495, given to the decoder as a `(1, 550)` array per trial. It is kept continuous rather than binarised, as the decoder-input specification requires.

ii.
```python
for trial in trial_ids:
    ...
    # Time is the sole decoder input and is continuous/time-varying.
    input_trials.append(time[None, :].copy())
```

iii. Not discussed. (One copy of the identical array is materialised per trial rather than shared.)

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural grid. The same `edges` array is passed into `_load_neural`, which floors `(spike_time − goCue) − TMIN` by `DT` to get the bin index, and the input is the centre of exactly those bins. So input bin *k* and neural bin *k* cover the same 10 ms interval relative to that trial's go cue, by construction, for every trial and session.

ii.
```python
edges = np.arange(TMIN, TMAX + DT / 2, DT, dtype=np.float64)
time  = (edges[:-1] + DT / 2).astype(np.float32)
...
neural_all, qualities = _load_neural(handle, probe, behavior, edges)
...
bin_index = np.floor((aligned - TMIN) / DT).astype(np.int64)
```
The same `time` array is also the interpolation target for all three behavioural streams:
```python
tongue_speed, tongue_visible = _load_velocity(handle, behavior, time, camera=0, feature="tongue")
```

iii. Not discussed explicitly; alignment is guaranteed by passing one grid object to every stream.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial flags of `obj.bp`: `R` and `L` (the instructed/rewarded port) and `hit`, `miss`, `no` (the outcome). The licks themselves (`bp.ev.lickL` / `bp.ev.lickR`) are not read; direction is inferred from instructed side × outcome, because the port the animal actually chose is not stored as its own field.

ii.
```python
result = {
    name: _array(bp[name]).astype(bool)
    for name in ("R", "L", "hit", "miss", "no", "early", "autowater")
}
```

iii. Step 28: *"Actual lick choice will be derived from target side plus hit/miss, not merely copied from the instructed side."* The AI found the authors' own definition in `funcs/getPrevChoice.m` (step 26): `choice = (bp.R & bp.hit) | (bp.L & bp.miss)`, with ignore trials set to NaN, and adopted it verbatim.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Three classes, constant within a trial and broadcast across all 550 bins: **right (1)** where `(R & hit) | (L & miss)`, **none (2)** on ignore trials (`bp.no`), **left (0)** otherwise (i.e. `(L & hit) | (R & miss)`). `left` is the array default, so the correctness of the `left` class relies on `hit`, `miss` and `no` being mutually exclusive and exhaustive — which holds for every trial of all 12 sessions. Resulting distribution: left 0.398, right 0.377, none 0.225.

ii.
```python
# getPrevChoice.m defines right choice as R-hit or L-miss. An ignore
# trial has no choice; all remaining response trials are left choice.
lick = np.zeros(len(keep), dtype=np.int8)  # left
lick[(behavior["R"] & behavior["hit"]) | (behavior["L"] & behavior["miss"])] = 1
lick[behavior["no"]] = 2
...
trial_output[0] = lick[trial]
```
```python
"output_values": [["left", "right", "none"], ...]
```

iii. Step 26 shows the AI reading `getPrevChoice.m`; the in-code comment states the derivation. The "none" class is an addition required by the decoder specification (ignore trials are NaN in the authors' code, and dropped from the paper's analyses).

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial flag, `obj.bp.autowater`, which marks the water-cued (WC) block trials in which water is delivered at a random port with no sample tone, delay or go cue.

ii.
```python
result = {name: _array(bp[name]).astype(bool) for name in (..., "autowater")}
```

iii. Not spelled out, but the AI's condition list transcribed from `Figure8a_thru_c.m` uses `autowater` as the WC/DR discriminator throughout (`hit&~stim.enable&autowater` = "all AW hits"), so the field is the repository's own context label.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct cast of the boolean flag: **DR = 0, WC = 1**, constant within a trial and broadcast across all 550 bins. Note that this is the *opposite* code ordering from the human reference (which uses WC = 0, DR = 1) and from the order in which the instructions list the classes; the AI's `output_values` entry is declared consistently as `["DR", "WC"]`, so the labelling is internally correct. Distribution: DR 0.685, WC 0.315 — a much higher WC share than the expert's 0.097, because the AI kept only the 12 genuine two-context sessions (22–39% autowater) and excluded the DR-dominated sessions.

ii.
```python
context = behavior["autowater"].astype(np.int8)  # 0 DR, 1 WC
...
trial_output[1] = context[trial]
...
"output_values": [..., ["DR", "WC"], ...]
```

iii. Not discussed beyond the code comment. Step 9's rationale — that the two-context sessions are the ones that "jointly provide … WC/DR labels" — is the reason the class is well populated.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Two per-trial flags of `obj.bp`: `hit` and `no`. `miss` is not read for this output; a trial that is neither a hit nor an ignore is an incorrect trial by construction (the three flags sum to exactly 1 on every trial of all 12 sessions).

ii.
```python
outcome = np.zeros(len(keep), dtype=np.int8)  # incorrect
outcome[behavior["hit"]] = 1
outcome[behavior["no"]] = 2
```

iii. The AI read `funcs/getOutcome.m` (step 26), which defines outcome as `bp.hit` with ignore trials set to NaN; the third class replaces that NaN, as required by the decoder specification.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A relabelling into three per-trial classes broadcast over time: **incorrect 0** (default, i.e. `miss`), **correct 1** (`hit`), **ignore 2** (`no`). Codes follow the order given in the instructions. Distribution: incorrect 0.106, correct 0.669, ignore 0.225.

ii.
```python
outcome = np.zeros(len(keep), dtype=np.int8)  # incorrect
outcome[behavior["hit"]] = 1
outcome[behavior["no"]] = 2
...
trial_output[2] = outcome[trial]
...
"output_values": [..., ["incorrect", "correct", "ignore"], ...]
```

iii. Step 28: ignore trials are retained *"because the requested outcome explicitly requires that class"* — a stated departure from the paper, which excludes them.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj{1}` — the **side camera only** — feature `tongue`, taking its `ts[:, 0, :]` (x) and `ts[:, 1, :]` (y) columns, plus that camera's `frameTimes` and `NdroppedFrames`. The bottom camera's `top_tongue` (which the expert also uses) is not read. Clock alignment additionally uses `obj.bp.ev.bitStart`, `obj.sglx.bitcode.bitstart` and `obj.sglx.fs`, and `obj.bp.ev.goCue`. The likelihood column (`ts[:, 2, :]`) is not read explicitly — visibility is taken from x/y being finite, which is equivalent, since the authors already set x and y to NaN wherever likelihood ≤ 0.9 (verified: on the example trial, x is NaN for 100% of frames with likelihood ≤ 0.9 and 0% of frames above it).

ii.
```python
tongue_speed, tongue_visible = _load_velocity(
    handle, behavior, time, camera=0, feature="tongue"
)
...
group = _video_group(handle, camera)
feature_index = _feature_index(handle, group, feature)
...
frames = _array(_deref(handle, group["frameTimes"], trial)).astype(np.float64).reshape(-1)
tracking = _array(_deref(handle, group["ts"], trial)).astype(np.float64)
```

iii. Not discussed in the narrative. `tongue` is the first entry of the authors' `params.traj_features{1}` (side view), i.e. the repository's primary tongue feature; the AI checked the feature names of both cameras (step 27) before choosing.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps, following `findPosition.m` + `findVelocity.m`. **(1)** The trial is skipped entirely if `NdroppedFrames` is NaN (the repository's own video-quality gate). **(2)** x and y are **linearly interpolated** from the (≈400 Hz) frame times onto the 550-bin, 10 ms grid; `interp1`-style semantics mean any target bin whose bracketing frames include a NaN, or that lies outside the video coverage, becomes NaN. **(3)** The visibility mask is recorded at this point (`isfinite(x) & isfinite(y)`). **(4)** Because the feature name contains "tongue", the repository's `fillmissing(...,'nearest')` and baseline-drift subtraction are **skipped** (matching `findVelocity.m`, which excludes the tongue from both), and speed is `hypot(np.gradient(x), np.gradient(y))`. No smoothing is applied to the tongue, matching `findPosition.m`'s `if ~contains(feat,'tongue')` guard. Where the authors set the tongue's NaN velocity to 0, the AI instead leaves NaN so the bin can be labelled "not visible". Note `np.gradient` is called without a spacing argument, so the units are pixels per bin rather than pixels per second — irrelevant for a within-session percentile split. Because `gradient` propagates NaN to neighbours, the first and last bin of each visible run also become "not visible". Result: only 3.9% of bins are classified as visible tongue (vs the expert's 12.5%, which pools both camera views).

ii.
```python
dropped = _array(_deref(handle, group["NdroppedFrames"], trial))
if np.size(dropped) and not np.all(np.isfinite(dropped)):
    continue
...
aligned_frames = frames - offset - behavior["go_cue"][trial]
xpos = _interp_trace(aligned_frames, tracking[:, 0, feature_index], target_time)
ypos = _interp_trace(aligned_frames, tracking[:, 1, feature_index], target_time)
visible[trial] = np.isfinite(xpos) & np.isfinite(ypos)

# The paper fills non-tongue coordinates with the nearest observation
# before differentiating. Tongue gaps remain gaps; invalid velocity is
# later represented by the explicit not-visible class.
if "tongue" not in feature:
    xpos, ypos = _nearest_fill(xpos), _nearest_fill(ypos)
xvel, yvel = np.gradient(xpos), np.gradient(ypos)
...
speed[trial] = np.hypot(xvel, yvel)
```

iii. The in-code comments state the rationale: the repository fills gaps for every feature except the tongue, and the AI keeps tongue gaps as gaps precisely so that the required "not visible" class has a meaning. Step 54: *"tongue is visible only during brief lick epochs, while paw/video coverage spans most of the aligned window"* — i.e. the AI checked the resulting visibility pattern for plausibility rather than accepting it blindly.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per session, over the pooled visible, finite speed samples of the **kept** trials only: the threshold is the **50th percentile**; bins below it are **0**, bins at or above it are **1**, and bins that are not visible (or where the speed is NaN) are **2**. If a session has no usable sample at all the whole session would be class 2 (never triggered). By construction classes 0 and 1 are equal in size (0.019 / 0.019 of all bins), with 0.961 not visible.

ii.
```python
def _median_discretize(values, available, keep):
    """Map below/equal-above session median to 0/1 and unavailable to 2."""
    usable = keep[:, None] & available & np.isfinite(values)
    if not np.any(usable):
        return np.full(values.shape, 2, dtype=np.int8), float("nan")
    threshold = float(np.percentile(values[usable], 50))
    output = np.full(values.shape, 2, dtype=np.int8)
    output[usable & (values < threshold)] = 0
    output[usable & (values >= threshold)] = 1
    return output, threshold
...
tongue_class, tongue_threshold = _median_discretize(tongue_speed, tongue_visible, keep)
```
The threshold is also recorded per session:
```python
"tongue_velocity_median": tongue_threshold,
```

iii. Directly from the decoder specification ("discretized with per-session threshold: 0 < 50th percentile, 1 ≥ 50th percentile, 2 not visible"). Step 54: *"each per-session median splits available samples evenly"* — the AI verified the split empirically.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Via the video-clock offset and then the same 550-bin grid. The offset between the video/neural file clock and the behaviour clock is computed once per call as `bitcode.bitstart / sglx.fs − bp.ev.bitStart`, using the **median** of each in place of `findVideoOffset.m`'s `mode`. Each trial's frame times then become `frameTimes − offset − goCue[trial]`, and the tracking is linearly interpolated onto the shared bin centres, so tongue bin *k* is the same 10 ms interval as neural bin *k*. (Verified: median and mode of both quantities are identical in all 12 sessions, giving offsets of 0.49004 s for the 2021 sessions and 0.990016 s for the JEB19 sessions.)

ii.
```python
def _video_offset(handle, behavior) -> float:
    bit_start = _array(handle["obj/bp/ev/bitStart"]).astype(float)
    video_bit_start = _array(handle["obj/sglx/bitcode/bitstart"]).astype(float)
    sampling_rate = float(_array(handle["obj/sglx/fs"]))
    # All values are effectively constant; median is the robust equivalent of
    # MATLAB mode here and avoids floating-point uniqueness issues.
    return float(np.median(video_bit_start) / sampling_rate - np.median(bit_start))
...
aligned_frames = frames - offset - behavior["go_cue"][trial]
xpos = _interp_trace(aligned_frames, tracking[:, 0, feature_index], target_time)
```

iii. The code comment gives the reason for median-over-mode (robustness and floating-point uniqueness). The AI read `funcs/findVideoOffset.m` (step 18) and reproduced its formula, including dividing `bitstart` by `sglx.fs`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. `obj.traj{2}` — the **bottom camera** — feature `top_paw`, x and y columns of `ts`, with that camera's `frameTimes` and `NdroppedFrames`, plus the same session video offset and per-trial go cue. `bottom_paw` is not used.

ii.
```python
# Primary side-view tongue and top paw are repository-standard
# features (Figure 1 uses top_paw_yvel_view2).
paw_speed, paw_visible = _load_velocity(
    handle, behavior, time, camera=1, feature="top_paw"
)
```

iii. The in-code comment: `top_paw` (view 2) is the feature the repository's own figures use (`top_paw_yvel_view2` in the Figure 1 kinematic plots). The AI listed both cameras' feature names and grepped for `top_paw|bottom_paw` usages in the repository (step 27) before choosing.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The same `_load_velocity` routine as the tongue, but with the two non-tongue branches active: after interpolation onto the 10 ms grid and after the visibility mask has been recorded, x and y are **nearest-filled** (`fillmissing(...,'nearest')`), then differentiated with `np.gradient`, then the **baseline drift** is removed by subtracting the median frame-to-frame difference of each coordinate, and speed is the hypotenuse. No cross-camera normalisation (only one view is used) and no smoothing (the repository calls `mySmooth(ts, 1, ...)`, which is a no-op). The visibility mask is deliberately the *pre-fill* one, so bins that only exist because of the nearest-fill are still labelled "not visible" (21.0% of bins). Note the AI deviates from `findVelocity.m` in one place, deliberately: the MATLAB subtracts `basederiv(1)` (the x median) from *both* xvel and yvel, which the AI treats as a typo and replaces with each coordinate's own median.

ii.
```python
if "tongue" not in feature:
    xpos, ypos = _nearest_fill(xpos), _nearest_fill(ypos)
xvel, yvel = np.gradient(xpos), np.gradient(ypos)
if "tongue" not in feature:
    # Remove slow tracking drift, following findVelocity.m. Subtracting
    # each coordinate's own median derivative fixes an apparent y/x typo
    # in that helper without changing visibility or threshold semantics.
    if np.any(np.isfinite(xvel)):
        xvel -= np.nanmedian(np.diff(xpos))
    if np.any(np.isfinite(yvel)):
        yvel -= np.nanmedian(np.diff(ypos))
speed[trial] = np.hypot(xvel, yvel)
```
```python
def _nearest_fill(values):
    """Match MATLAB fillmissing(..., 'nearest') for a one-dimensional trace."""
```

iii. The comments state both the source (`findVelocity.m`) and the one deliberate correction, noting it does not change visibility or threshold semantics.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Identically to the tongue: the 50th percentile of the visible, finite paw speeds pooled over the kept trials of that session; `< threshold → 0`, `≥ threshold → 1`, not-visible/NaN → 2. Resulting fractions 0.395 / 0.395 / 0.210, close to the expert's 0.407 / 0.407 / 0.186.

ii.
```python
paw_class, paw_threshold = _median_discretize(paw_speed, paw_visible, keep)
...
"output_values": [..., ["< session median", ">= session median", "not visible"], ...]
```

iii. Same as 7-c — the split is the decoder specification's.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue, but using the **bottom camera's own** `frameTimes` (so a mismatch in frame count between the two views cannot misalign it): `frameTimes − offset − goCue[trial]`, then linear interpolation onto the shared 550-bin grid. The session offset is recomputed inside this second `_load_velocity` call, but from the same fields and with the same result.

ii.
```python
group = _video_group(handle, camera)          # camera = 1 for the paw
offset = _video_offset(handle, behavior)
...
frames = _array(_deref(handle, group["frameTimes"], trial)).astype(np.float64).reshape(-1)
aligned_frames = frames - offset - behavior["go_cue"][trial]
xpos = _interp_trace(aligned_frames, tracking[:, 0, feature_index], target_time)
```

iii. Not discussed separately; alignment is shared code with the tongue, and the per-camera `frameTimes` fall out of passing `camera` through.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file beside the data structure, field `me.data` — a per-trial cell of one motion-energy value per side-camera frame. The copy inside `obj.me` is not used. The **side camera's** (`obj.traj{1}`) `frameTimes` and `NdroppedFrames` supply the time base, together with the session video offset and the per-trial go cue. `me.moveThresh` is read but ignored (the task requires a median split, not the authors' movement threshold).

ii.
```python
mat = io.loadmat(path, simplify_cells=True, variable_names=["me"])
raw = mat["me"]["data"]
raw_trials = list(raw) if isinstance(raw, np.ndarray) and raw.dtype == object else [raw]
group = _video_group(handle, 0)
offset = _video_offset(handle, behavior)
```

iii. Not narrated. `loadMotionEnergy.m` (read at step 10) indexes `obj.traj{1}(trix).frameTimes` for the motion-energy time base, which is why the side camera is used. All 12 of the AI's motion-energy files store `me` as `{data, moveThresh}` with `data` a cell array, so the single unwrap suffices (the doubly-wrapped `me.data.data` layout that `loadMotionEnergy.m` also guards against does not occur in these 12 files, and the AI does not handle it).

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond re-timing: the per-frame trace is linearly interpolated from the offset-corrected frame times onto the 550-bin grid. The value is already a single scalar per frame upstream, so there is nothing to reduce, smooth or differentiate. Trials whose `NdroppedFrames` is NaN are skipped, and — unlike `loadMotionEnergy.m`, which ends with `fillmissing(me.data,'nearest')` — the AI deliberately does **not** fill, so bins outside the video coverage (mostly the early part of the −3 s window, before the camera starts) stay NaN and become the "no video" class (9.8% of bins, higher than the expert's 3.8% mainly because the window extends 0.5 s further back).

ii.
```python
for trial in range(min(ntrials, len(raw_trials))):
    dropped = _array(_deref(handle, group["NdroppedFrames"], trial))
    if np.size(dropped) and not np.all(np.isfinite(dropped)):
        continue
    frames = _array(_deref(handle, group["frameTimes"], trial)).astype(np.float64).reshape(-1)
    aligned_frames = frames - offset - behavior["go_cue"][trial]
    trace = np.asarray(raw_trials[trial], dtype=np.float64).reshape(-1)
    values[trial] = _interp_trace(aligned_frames, trace, target_time)
    available[trial] = np.isfinite(values[trial])
```

iii. Not narrated; the "no video" class name in `output_values` records the intent, and leaving the gaps unfilled is what makes that class meaningful.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The same `_median_discretize`: per session, the 50th percentile of the finite motion-energy values over the kept trials; `< threshold → 0`, `≥ threshold → 1`, missing → 2 ("no video"). Fractions 0.450 / 0.452 / 0.098.

ii.
```python
motion_class, motion_threshold = _median_discretize(motion_energy, motion_available, keep)
...
["< session median", ">= session median", "no video"],
```

iii. From the decoder specification. The authors' own `me.moveThresh` was available but was not used, since the task prescribes the percentile split.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same offset and same grid as the DLC streams, using the **side camera's** frame times because motion energy has exactly one value per side-camera frame: `frameTimes − offset − goCue[trial]`, then `interp1` onto the shared 550 bin centres.

ii.
```python
group = _video_group(handle, 0)
offset = _video_offset(handle, behavior)
...
aligned_frames = frames - offset - behavior["go_cue"][trial]
values[trial] = _interp_trace(aligned_frames, trace, target_time)
```

iii. This reproduces `loadMotionEnergy.m`'s `interp1(obj.traj{1}(trix).frameTimes - vidshift - alignTimes(trix), me.data{trix}, taxis)` line, which the AI read at step 10.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Every anomaly is handled by **keeping the trial and marking the gap**, never by dropping a trial or inventing a value. Six guards: **(1)** `NdroppedFrames` NaN → the whole trial's video stream is skipped and comes out as 550 "not visible"/"no video" bins (this is the repository's own gate in `findPosition.m`; none of the 12 sessions actually trigger it). **(2)** Frame times or coordinates that cannot support interpolation (`n < 2` usable samples, e.g. all-NaN `frameTimes`) → the whole trial's trace is NaN. **(3)** Untracked frames — the authors already store NaN wherever the DLC likelihood ≤ 0.9 — propagate through `interp1` and become class 2. **(4)** Bins outside the video's temporal coverage get `fill_value=np.nan` and become class 2, rather than being extrapolated. **(5)** A malformed or short `ts` array (`ndim != 3`, or the feature index out of range) skips that trial. **(6)** A motion-energy cell array shorter than the trial count is truncated with `min(ntrials, len(raw_trials))`. Within `_median_discretize`, any remaining NaN is folded into the trailing class, and a session with no usable sample at all would return all-2 plus a NaN threshold rather than raising. The one hard failure the code allows is a session in which no unit survives curation (`RuntimeError`) or a missing input file (`FileNotFoundError`).

ii.
```python
dropped = _array(_deref(handle, group["NdroppedFrames"], trial))
if np.size(dropped) and not np.all(np.isfinite(dropped)):
    continue
...
if tracking.ndim != 3 or feature_index >= tracking.shape[2]:
    continue
```
```python
n = min(len(source_time), len(values))
if n < 2:
    return np.full(target_time.shape, np.nan, dtype=np.float64)
...
fun = interpolate.interp1d(source_time, values, kind="linear",
                           bounds_error=False, fill_value=np.nan, assume_sorted=True)
```
```python
usable = keep[:, None] & available & np.isfinite(values)
if not np.any(usable):
    return np.full(values.shape, 2, dtype=np.int8), float("nan")
```

iii. Partly from the repository (the `NdroppedFrames` gate and the `interp1` semantics are copied from `findPosition.m`), partly forced by the output format: the format forbids NaN, and the "not visible"/"no video" classes exist precisely so a missing camera sample can be represented honestly. Step 39 records one gap-handling fix made after the first run: *"silencing drift estimation on wholly missing video trials"* — the `np.any(np.isfinite(...))` guards around the drift subtraction were added because `np.nanmedian` of an all-NaN trace warned.

## 11-a. What are the most time-consuming steps of the code?

i. The whole conversion takes roughly 40 s for the 12 sessions (≈3 s per session), so nothing is expensive in absolute terms, but the ranking inside a session is: **(1)** `_load_neural` — a Python loop over every cluster on the probe (≈860 clusters over 12 sessions) that, for each one, dereferences two HDF5 datasets, scatter-adds spikes with `np.add.at` (the slowest available way to do a 2-D histogram in NumPy — it takes the un-buffered path), runs a 2-D `signal.convolve` over the full (n_trials × 550) matrix, and then computes seven condition means. **(2)** Reading the HDF5 files themselves — the per-trial `frameTimes`/`ts`/`NdroppedFrames` cell arrays are dereferenced one trial at a time, three separate times per session (side camera for the tongue, bottom camera for the paw, side camera again for motion energy), i.e. ≈9,300 individual dataset reads. **(3)** `scipy.io.loadmat` on the motion-energy file, which is mitigated by `variable_names=["me"]`. **(4)** Pickling the 311 MiB result.

ii.
```python
for unit in range(probe_group["quality"].shape[0]):
    ...
    np.add.at(counts, (spike_trial[valid] - 1, bin_index[valid]), 1.0)
    rates = _causal_smooth(counts / DT)
    condition_means = [rates[mask].mean(axis=0) for mask in conditions if np.any(mask)]
```
```python
for trial in range(ntrials):
    dropped = _array(_deref(handle, group["NdroppedFrames"], trial))
    ...
    frames = _array(_deref(handle, group["frameTimes"], trial))...
    tracking = _array(_deref(handle, group["ts"], trial))...
```

iii. Not discussed in the trajectory — the AI never profiled or commented on runtime. The conversion is fast enough (≈40 s) that the inefficiencies never mattered, largely *because* the AI restricted itself to 12 of the 44 sessions.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Four. **(1)** The cluster loop in `_load_neural`: `np.add.at` should be `np.bincount` on a flattened `(trial, bin)` index or a single `np.histogram2d` over all trials at once (which is what the expert's code does), and the smoothing of all units could be batched into one convolution instead of one per unit. **(2)** The seven-condition mean inside that loop: `[rates[mask].mean(axis=0) for mask in conditions]` re-slices the same `(n_trials, 550)` matrix seven times per unit; it is a single `(7, n_trials) @ (n_trials, 550)` matrix product, and it could be done once for all units rather than once per unit. **(3)** The final assembly loop `for trial in trial_ids`, which builds 3,116 separate `(6, 550)` arrays and 3,116 identical copies of the time vector — the per-trial outputs could be assembled by broadcasting the per-trial scalars against the already-computed `(n_trials, 550)` class matrices, and the input could be one shared array. **(4)** The trial loops in `_load_velocity` and `_load_motion_energy` are the genuinely hard case, since each trial has a different number of camera frames; they cannot be fully vectorized, but the three passes could at least be merged so the per-trial HDF5 dereferences happen once instead of three times.

ii.
```python
np.add.at(counts, (spike_trial[valid] - 1, bin_index[valid]), 1.0)
```
```python
condition_means = [rates[mask].mean(axis=0) for mask in conditions if np.any(mask)]
```
```python
for trial in trial_ids:
    neural_trials.append(neural_all[trial].copy())
    input_trials.append(time[None, :].copy())
    trial_output = np.empty((6, len(time)), dtype=np.int8)
    trial_output[0] = lick[trial]
    ...
```

iii. Not discussed. The output format itself demands a list of per-trial arrays, so loop (3) cannot be removed entirely, only made cheaper.

## 11-c. What processing does the code repeat multiple times?

i. Three genuine repeats, all small. **(1)** `_video_offset` is recomputed **three times per session** — once in each `_load_velocity` call and once in `_load_motion_energy` — re-reading `bp.ev.bitStart`, `sglx.bitcode.bitstart` and `sglx.fs` each time, even though it is a session constant (the expert computes it once in `Camera.__init__`). **(2)** The side camera's `frameTimes` and `NdroppedFrames` are dereferenced and decoded twice per trial, once for the tongue and once for motion energy. **(3)** `_feature_index` re-reads and decodes the camera's `featNames` cell on every call. Against that, the genuinely expensive things are done once: each `.mat` file is opened once, each unit's spikes are binned and smoothed once (the condition means reuse that one `rates` matrix), and the bin grid is built once in `convert` and threaded through every stream.

ii.
```python
def _load_velocity(handle, behavior, target_time, camera, feature):
    group = _video_group(handle, camera)
    feature_index = _feature_index(handle, group, feature)
    ...
    offset = _video_offset(handle, behavior)      # once per feature, not per session
```
```python
def _load_motion_energy(path, handle, behavior, target_time):
    ...
    group = _video_group(handle, 0)
    offset = _video_offset(handle, behavior)      # third time for this session
```

iii. Not discussed. The repeats are cheap (three tiny dataset reads per session) and reading them is idempotent, so they change correctness not at all and runtime negligibly.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Five things. **(1)** Every behavioural stream is computed for **all** trials and only subset at the end, so the full velocity, motion-energy and firing-rate pipelines run on the ~20% of trials (810 of 3,926) that the early-lick/photostim filter then discards. For the neural data this is necessary — the condition PSTHs used by the low-FR filter are defined over unfiltered trials — but for the three camera streams it is pure waste. **(2)** For the paw, `_nearest_fill` plus the drift subtraction produce a velocity in exactly the bins that the pre-fill visibility mask then relabels "not visible", so that work is computed and thrown away for 21% of bins. **(3)** `_load_behavior` reads `bp.L`, and `bp.miss` is loaded but only used via the `_figure8_conditions` masks; `me.moveThresh` is loaded by `scipy.io` and never used. **(4)** `metadata['time_bin_centers_s']` stores a 550-element list that duplicates the per-trial `input` exactly, and `input_trials` stores 3,116 identical copies of that same vector (≈6.8 MB of redundancy). **(5)** The `qualities` list is accumulated per session only to produce a `unit_quality_counts` dictionary in the metadata that no downstream analysis reads. Nothing in the output is wrong because of any of this; it is all pure overhead.

ii.
```python
tongue_speed, tongue_visible = _load_velocity(handle, behavior, time, camera=0, feature="tongue")
# computed for all ntrials, then:
trial_ids = np.flatnonzero(keep)
```
```python
if "tongue" not in feature:
    xpos, ypos = _nearest_fill(xpos), _nearest_fill(ypos)   # filled bins are later class 2
```
```python
# A plain list keeps the optional --stats-json path in the supplied
# validator JSON-serializable.
"time_bin_centers_s": time.tolist(),
...
"unit_quality_counts": {quality: qualities.count(quality) for quality in sorted(set(qualities))},
```

iii. The only related note in the trajectory is step 47, where `time_bin_centers_s` was converted from a NumPy array to a list so that the validator's `--stats-json` output stays JSON-serializable — i.e. the redundant field was kept and patched rather than removed. The other items are not discussed.
