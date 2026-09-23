# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a list of 12 sessions (animal, date, probe tuple) taken from the authors' two-context analysis scripts (`Scripts/Figure 8/Figure8a_thru_c.m`, which loads JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, JEB19), and reads only `/app/data/Ephys_Behavior`. Each session is one MATLAB v7.3 file opened once with `h5py`; only the fields actually needed are read (`obj/bp/*`, `obj/clu`, `obj/traj`, `obj/sglx`), rather than materialising the whole `obj` tree. Motion energy comes from the matching `motionEnergy_<anm>_<date>.mat`, read with `scipy.io.loadmat(..., simplify_cells=True)`. The other 13 fixed-delay sessions (JEB13, JEB14, JEB15) and all 22 randomized-delay sessions are deliberately not converted; the `DelayInhibition`/`GoCueInhibition` folders are excluded because they have no ephys. No v5/MAT-file reader is implemented, so the 11 v5 files in `RandomizedDelay_Ephys_Behavior` could not have been loaded by this script anyway.

ii.
```python
DATA_DIR = APP / "data" / "Ephys_Behavior"
...
# Exact order and ALM probes in the supplied two-context analysis scripts.
SESSIONS = [
    ("JEB6", "2021-04-18", (2,)),
    ("JEB7", "2021-04-29", (1,)),
    ...
    ("JEB19", "2023-04-18", (1,)),
]
```
```python
def process_session(animal, date, probes, show_plot):
    session_id = f"{animal}_{date}"
    data_path = DATA_DIR / f"data_structure_{session_id}.mat"
    motion_path = DATA_DIR / f"motionEnergy_{session_id}.mat"
    with h5py.File(data_path, "r") as f:
        n_trials = int(np.asarray(f["obj/bp/Ntrials"])[0, 0])
        fields = ("R", "L", "hit", "miss", "no", "early", "autowater")
        bp = {name: np.asarray(f[f"obj/bp/{name}"]).ravel().astype(bool) for name in fields}
```
```python
def load_motion_file(path, n_trials):
    loaded = scipy_io.loadmat(path, simplify_cells=True)["me"]
    raw = loaded["data"]
    if isinstance(raw, dict) and "data" in raw:
        raw = raw["data"]
```

iii. From CONVERSION_NOTES Step 4: "The paper's two-context code selects 12 sessions (3,626 native trials)"; the paper states "In 12 sessions from six mice, animals performed the two-context task. In total, 522 units". The AI argues the decoder must predict WC vs DR context, so only the cohort that actually alternates WC and DR blocks is applicable; it claims the other 13 fixed-delay sessions "do not contain genuine alternating WC blocks; small `autowater` counts there are assistance trials, not a WC context", and that randomized-delay sessions "test a different task variant and do not provide the alternating WC/DR context required here". It reports 518 retained units against the paper's 522 as its cohort sanity check.

## 1-b. How are the data split into subjects (mice)?

i. The subject is the animal token in the session tuple (the filename prefix). `subjects` is built in first-appearance order over the selected sessions and `subject_idx` indexes into it. The result is 7 subjects (JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, JEB19) for 12 sessions; the AI keeps 7 distinct IDs even though the paper reports six mice for this cohort, and documents that discrepancy rather than merging IDs.

ii.
```python
for index, (animal, date, probes) in enumerate(selected):
    if animal not in subjects:
        subjects.append(animal)
...
subject_idx = np.asarray([subjects.index(animal) for animal, _, _ in selected], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 4: "the supplied files contain seven subject IDs even though the paper reports six … Files contain seven distinct subject IDs, which is preserved rather than relabeling animals." The animal id is taken from the filename because that is how the authors' `load<ANM>_ALMVideo.m` scripts identify animals.

## 1-c. How are the data split into sessions?

i. One entry of `SESSIONS` = one `data_structure_*.mat` file = one element of `neural`/`input`/`output`/`brain_region_idx`. Session order is the fixed order of the hard-coded list, and every per-session quantity (video clock offset, the three discretisation thresholds, unit set) is computed within that one file. Only the probe(s) named by the authors' loader for that session are used; the code supports concatenating several probes, although all 12 selected sessions are single-probe. Sample mode (`--sample`) takes the first two sessions.

ii.
```python
selected = SESSIONS[:2] if args.sample else SESSIONS
...
for index, (animal, date, probes) in enumerate(selected):
    n, i, o, info = process_session(animal, date, probes, show)
    neural.append(n); inputs.append(i); outputs.append(o); session_info.append(info)
```
```python
for probe_index, probe_ref in enumerate(f["obj/clu"][:, 0], start=1):
    if probe_index not in probes:
        continue
```

iii. Sessions are defined by the authors' loader scripts, which also specify the ALM probe per session ("Exact order and ALM probes in the supplied two-context analysis scripts"). Everything that must be session-wide (thresholds, video offset) is therefore computed inside `process_session`.

## 1-d. How are the data split into trials?

i. A trial is one row of the Bpod table: `obj.bp.Ntrials` sets the count, and every per-trial flag/event array is read and explicitly checked to have exactly `Ntrials` entries (raising otherwise). Spikes carry their trial number in `clu.trial` (1-based) and their within-trial time in `clu.trialtm`; video frames are stored per trial in `obj.traj{view}(trial)`; motion energy is one vector per trial. So no trial boundary has to be reconstructed.

ii.
```python
n_trials = int(np.asarray(f["obj/bp/Ntrials"])[0, 0])
fields = ("R", "L", "hit", "miss", "no", "early", "autowater")
bp = {name: np.asarray(f[f"obj/bp/{name}"]).ravel().astype(bool) for name in fields}
bp["stim"] = np.asarray(f["obj/bp/stim/enable"]).ravel().astype(bool)
go = np.asarray(f["obj/bp/ev/goCue"]).ravel().astype(np.float64)
for name, values in bp.items():
    if values.size != n_trials:
        raise ValueError(f"{session_id}: {name} has {values.size}, expected {n_trials}")
if go.size != n_trials:
    raise ValueError(f"{session_id}: goCue length mismatch")
```

iii. CONVERSION_NOTES Step 2 documents that `bp` holds one entry per trial for all labels and events and that `traj` holds "one structure per trial"; the length assertions are listed under the Step 6 "validate shapes and types at each step" and Step 10 edge-case audit ("all trial arrays agree").

## 1-e. How are trials filtered based on quality controls?

i. A trial is kept when it is not an early-lick trial, not a photostimulation trial, has a finite go-cue time, has exactly one of `hit`/`miss`/`no` set, and exactly one of `R`/`L` set. Ignore (`no`) trials are deliberately kept because the decoder spec requires an `ignore` outcome class, even though the paper omits them from its analyses. Trials with missing/broken video are also kept (their movement outputs become the "not visible" class) rather than dropped. Sessions with fewer than two surviving trials would raise. Across the 12 sessions this keeps 3,116 of 3,626 native trials. There is no check for trials that fall after the end of the ephys recording (I verified independently that no such trials exist in these 12 sessions, so the omission is inert for this cohort).

ii.
```python
outcome_sum = bp["hit"].astype(int) + bp["miss"].astype(int) + bp["no"].astype(int)
side_sum = bp["R"].astype(int) + bp["L"].astype(int)
retained = np.flatnonzero(~bp["early"] & ~bp["stim"] & np.isfinite(go)
                          & (outcome_sum == 1) & (side_sum == 1))
if retained.size < 2:
    raise ValueError(f"{session_id}: fewer than two retained trials")
```

iii. CONVERSION_NOTES Step 3/4: "Early-lick trials are omitted from analyses. Reference task conditions typically also exclude stimulation-enabled trials … Behavioral analyses omit ignore trials, while the requested decoder explicitly requires an `ignore` outcome, so valid ignore trials must be retained as a task-required exception." The methods text quoted by the AI says early-lick trials "were omitted from analyses", and every `params.condition` expression in the reference code contains `~stim.enable`. The flag-consistency requirements are described as guarding against malformed rows.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` for the probe(s) named by the authors' loader for that session: `quality` (curation label), `trial` (1-based trial index of each spike) and `trialtm` (spike time relative to that trial's start). `obj.bp.ev.goCue` supplies the alignment time. `obj.bp.Ntrials` bounds the trial axis. Spike waveforms and absolute times (`spkWavs`, `tm`) are never read.

ii.
```python
for cluster_index in range(group["quality"].shape[0]):
    quality = h5_string(f, group["quality"][cluster_index, 0]).strip()
    ...
    trials = h5_vector(f, group["trial"][cluster_index, 0], np.float64)
    trial_times = h5_vector(f, group["trialtm"][cluster_index, 0], np.float64)
    if trials.size != trial_times.size:
        raise ValueError(...)
    counts = bin_one_unit(trials, trial_times, go, n_trials)
```

iii. Step 5 mapping table: "Selected `obj.clu{probe}.trialtm`, `.trial`, `bp.ev.goCue` → `neural`", with reference functions `alignSpikes`, `getSeq`, `mySmooth`. `trialtm` is already on the behaviour clock relative to trial start, which is what `alignSpikes.m` subtracts the event time from.

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed into 500 × 10 ms bins per trial using explicit right-side `searchsorted` edges (to reproduce MATLAB `histc` semantics at exact bin edges), divided by `DT` to give spikes/s, then smoothed along time with the reference's **causal** 15-sample Gaussian (`mySmooth(x, 15, 'reflect')`): a `gausswin(15)` whose first half is zeroed and which is renormalised, with the reference's boundary trick of prepending the first 15 samples and trimming them afterwards. No normalisation, baseline subtraction or z-scoring. Stored as `float32`, one `(n_neurons, 500)` matrix per trial. Smoothing is applied to all trials of a unit at once via a sliding-window/`einsum` convolution.

ii.
```python
def matlab_gausswin_causal(n=SMOOTH_N, alpha=2.5):
    x = np.arange(n, dtype=np.float64) - (n - 1) / 2
    kernel = np.exp(-0.5 * (alpha * x / (n / 2)) ** 2)
    kernel[: n // 2] = 0          # causal
    kernel /= kernel.sum()
    return kernel

def smooth_reference(counts):
    """Reference mySmooth(x, 15, 'reflect') along the last dimension."""
    prefix = counts[..., :SMOOTH_N]
    padded = np.concatenate((prefix, counts), axis=-1)
    ...
    smoothed = np.einsum("...k,k->...", windows, KERNEL[::-1], optimize=True)
    return smoothed[..., SMOOTH_N:]
```
```python
bins = np.searchsorted(EDGES, aligned, side="right") - 1
...
rates = smooth_reference(counts / DT).astype(np.float32)
```

iii. Step 5/Step 10: "histogram in 10-ms bins on `[-2.5,2.5)`; divide by 0.01 s; apply the reference 15-sample causal Gaussian smoother with reflected leading boundary", matching `getSeq.m` + `mySmooth.m`. The `searchsorted` edge handling was introduced in Critical Review 1 after an independent check showed floor arithmetic could place spikes sitting exactly on a decimal edge in the wrong bin relative to MATLAB `histc`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. (1) The manual curation label `clu.quality`, stripped and lower-cased, is rejected if it is one of `garbage`, `gabrga`, `noisy`, `real?` — exactly the drop list of `findClusters.m` with `quality={'all'}`. Everything else is kept, including `multi`, `poor` and the four empty labels. (2) After binning and smoothing, any unit whose mean rate over all trials and bins is not strictly greater than 1 Hz is dropped. This yields 528 quality-passing → 518 retained units (27–67 per session, mean 43.2), against the paper's 522 for this cohort. Only the probe(s) named by the authors' loader contribute.

ii.
```python
BAD_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}
...
if quality.lower() in BAD_QUALITIES:
    continue
...
mean_rate = float(rates.mean())
if mean_rate > 1.0:
    all_rates.append(rates)
```

iii. Step 3 curation rules: "The standard loader with `quality={'all'}` excludes garbage/gabrga/noisy/real? and then retains all acceptable units whose mean firing rate is strictly >1 Hz. Well-isolated-only filtering is reserved for single-unit selectivity/subspace analyses." The paper: "All units with firing rates exceeding 1 Hz were included in all other analyses." `params.lowFR = 1` is used by every figure script (the 0.5 in `getDefaultParams.m` is overridden). Step 4 records the 518-vs-522 gap and explicitly refuses to tune the threshold to match the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. One subtraction: each spike's `trialtm` minus the `goCue` of its own trial, both on the behaviour clock. No interpolation or extra offset. Spikes falling outside [-2.5, 2.5) s drop off the histogram edges. Trials are indexed 1-based in the raw file and converted with `- 1`. For WC trials the same `bp.ev.goCue` field holds the water-presentation time, which is documented in the metadata as the alignment event.

ii.
```python
valid_trial = (trials >= 1) & (trials <= n_trials) & np.isfinite(trial_times)
tr0 = trials[valid_trial].astype(np.int64) - 1
aligned = trial_times[valid_trial] - go[tr0]
bins = np.searchsorted(EDGES, aligned, side="right") - 1
valid = np.isfinite(aligned) & (bins >= 0) & (bins < N_TIME)
np.add.at(counts, (tr0[valid], bins[valid]), 1)
```
```python
"temporal_alignment_event": "go cue onset (water presentation in WC trials)",
```

iii. Step 10 Check 3(c): "spike `trialtm-goCue` matches `alignSpikes`", i.e. `obj.clu{prb}(clu).trialtm_aligned = obj.clu{prb}(clu).trialtm - event` with `params.alignEvent='goCue'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 10 ms bins spanning [-2.5, +2.5) s from the go cue = 500 bins, identical for every trial, session and data stream. Spikes are binned directly at that resolution (no rebinning from a finer grid); the 400 Hz video streams are resampled onto the same axis by interpolating at the bin centres. `metadata['time_bin_size'] = 10.0` ms, `off_start = -2.5`, `off_end = 2.5`.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 0.010
TIME = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2   # bin centres
N_TIME = TIME.size                                            # 500
EDGES = TMIN + np.arange(N_TIME + 1, dtype=np.float64) * DT
```

iii. Step 4: "Choice/kinematics code uses `[-2.5,2.5)`, 10 ms; context Figure 8 uses `[-3,2.5)`, 10 ms … Use the common decoder/kinematics window `[-2.5,2.5)` and 10 ms; it matches both the tutorial and choice decoding and avoids context-specific extra ITI." `params.dt = 1/100` is indeed the value used by almost every analysis script in `/app/code` (only `getDefaultParams.m`, `Figure1e`, `Figure8d` and the RGB overlay script use 1/200). Bin centres rather than edges are stored "to match `obj.time` in `getSeq`".

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not derived from a raw variable: it is the bin-centre time axis defined by the conversion, i.e. the same grid the spikes are binned on, expressed relative to `bp.ev.goCue` of each trial. Every trial gets the identical `(1, 500)` float32 vector from −2.495 to +2.495 s.

ii.
```python
TIME = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2
...
time_input = TIME.astype(np.float32)[None, :]
...
input_trials.append(time_input.copy())
```
```python
"input_names": ["time from go cue onset (s)"],
```

iii. Step 5 mapping table: "Bin centers `-2.495 ... 2.495` s → `input[0]`; store as a 1 × 500 float32 time series on every trial; reference `getSeq`; decoder input is only time from go cue." The decoder spec asks for a continuous, time-varying "time from go cue onset in seconds".

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond constructing the axis: `arange(-2.5, 2.5, 0.01) + 0.005`, cast to float32 and copied per trial. The AI verified independently (`cache/sanity_checks.py`) that all 3,116 stored input vectors equal an independently recreated bin-centre axis to within float32 quantisation (`atol=2e-7`).

ii.
```python
input_checks = []
for session in converted["input"]:
    for trial in session:
        input_checks.append(np.allclose(trial[0], TIME, rtol=0, atol=2e-7))
assert all(input_checks)
```

iii. Step 10 Check 2: "Input: independently recreated bin centers and matched all 3,116 converted trials (`atol=2e-7`, the float32 quantization bound)."

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is the neural binning grid itself. Spike times are expressed relative to the trial's go cue and counted into `EDGES`; the input is the centre of those same bins, so column *k* of `input` and column *k* of `neural` describe the same 10 ms interval, and the same axis is used as the interpolation target for the three video-derived outputs.

ii.
```python
EDGES = TMIN + np.arange(N_TIME + 1, dtype=np.float64) * DT   # neural bin edges
TIME  = np.arange(TMIN, TMAX, DT) + DT / 2                    # input = bin centres
...
bins = np.searchsorted(EDGES, aligned, side="right") - 1      # spikes onto that grid
```

iii. Step 10 Check 3(e): "Input: bin centers match `obj.time`; the target specifically limits input to time." Since the axis is shared by construction there is no alignment step to get wrong; the `--show-processing` plots draw the go cue at t=0 on both the neural raster and the behavioural traces.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial Bpod flags: `bp.R` and `bp.L` (the instructed/rewarded port) together with `bp.hit`, `bp.miss` and `bp.no`. The lick direction itself is not recorded, so it is inferred from instructed side × outcome.

ii.
```python
def trial_labels(bp, trial):
    right, left = bool(bp["R"][trial]), bool(bp["L"][trial])
    hit, miss, no = bool(bp["hit"][trial]), bool(bp["miss"][trial]), bool(bp["no"][trial])
```

iii. Step 5 mapping table: "`bp.R/L`, `bp.hit/miss/no` → `output[0]` lick direction. Actual response: hit uses instructed/reward side, miss uses opposite side, no-response maps to none."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A three-way relabelling: a `no` trial is `none` (2); a hit is the instructed side (right→1, left→0); anything else (a miss) is the opposite of the instructed side. The scalar is broadcast across all 500 bins so that all six outputs share one `(6, 500)` integer array. Codes/names are `["left", "right", "none"]`, matching the order in the decoder spec. Resulting distribution: left 0.398, right 0.377, none 0.225.

ii.
```python
if no:
    lick = 2
elif hit:
    lick = 1 if right else 0
else:  # incorrect response is opposite the instructed/reward side
    lick = 0 if right else 1
...
output[0] = lick
```

iii. Step 5 Key Decision 3: "Store all six outputs as a 6 × 500 int64 array. The first three rows are constant within trial (therefore still per-trial) … The provided decoder requires a single common dimensionality." Step 10 Check 2 reports that lick/context/outcome rows were re-derived from the raw Bpod flags for the first/middle/last retained trial of every session (36 trials) and matched.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial flag, `obj.bp.autowater`, which marks the water-cued block trials.

ii.
```python
fields = ("R", "L", "hit", "miss", "no", "early", "autowater")
bp = {name: np.asarray(f[f"obj/bp/{name}"]).ravel().astype(bool) for name in fields}
...
context = 0 if bool(bp["autowater"][trial]) else 1
```

iii. Step 3: "The task alternates DR and WC blocks … `autowater` is used by the supplied code as the WC indicator" — the reference `params.condition` strings separate contexts with `autowater` / `~autowater`.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling broadcast over time: `autowater` → WC = 0, otherwise DR = 1, with `output_values` `["WC", "DR"]` matching the order in the decoder spec. Distribution 0.315 WC / 0.685 DR, and every one of the 12 sessions contains both classes (per-session WC fraction 0.226–0.391).

ii.
```python
context = 0 if bool(bp["autowater"][trial]) else 1
...
output[1] = context
```

iii. Step 5 mapping table: "`bp.autowater` → `output[1]` behavioral context; 1 → WC, 0 → DR; broadcast; codes WC=0, DR=1." Step 12 additionally audits the block structure of one session (JEB6: first WC trial at retained position 141, seven block transitions) to confirm the labels reproduce alternating blocks.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The three mutually exclusive per-trial flags `bp.hit`, `bp.miss` and `bp.no`. The code additionally asserts that exactly one of them is set on every retained trial.

ii.
```python
hit, miss, no = bool(bp["hit"][trial]), bool(bp["miss"][trial]), bool(bp["no"][trial])
...
outcome_sum = bp["hit"].astype(int) + bp["miss"].astype(int) + bp["no"].astype(int)
retained = np.flatnonzero(... & (outcome_sum == 1) & ...)
```

iii. Step 1 notes `getOutcome.m` ("Use hit as correctness and replace `bp.no` (ignore) with NaN"); Step 5 maps "`bp.hit/miss/no` → `output[2]` outcome … miss → incorrect, hit → correct, no → ignore".

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Relabelling into `["incorrect", "correct", "ignore"]` = 0/1/2 and broadcasting across the 500 bins. Ignore trials are kept as their own class instead of being dropped as in the paper. Distribution: incorrect 0.106, correct 0.669, ignore 0.225.

ii.
```python
outcome = 1 if hit else (0 if miss else 2)
...
output[2] = outcome
```

iii. Step 4: "Paper task conditions exclude early and stimulation trials and behavioral analyses omit ignores … Requested output explicitly includes `ignore` → Exclude early and stimulation trials; retain valid ignore trials because predicting ignore is explicitly required." Codes follow the spec's "incorrect, correct, ignore" ordering.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj{1}` (side camera) only: the `tongue` feature's x and y columns of `ts` (frames × [x, y, likelihood] × features), plus that view's `frameTimes`. The bottom-camera tongue markers (`top_tongue`, `bottom_tongue`, …) are not used. `obj.sglx.bitcode.bitstart`, `obj.sglx.fs`, `obj.bp.ev.bitStart` (for the video clock offset) and `bp.ev.goCue` are the other inputs. Missing tracking is identified as NaN in x/y — which I verified is exactly equivalent to DeepLabCut likelihood ≤ 0.9 in these files.

ii.
```python
side = get_traj_group(f, 0)
side_names = traj_feature_names(f, side) if side is not None else []
if "tongue" not in side_names:
    raise ValueError(f"Side-view tongue feature unavailable: {side_names}")
tongue_ix = side_names.index("tongue")
...
x = interp_matlab(aligned_ft, side_ts[:, 0, tongue_ix], target_absolute)
y = interp_matlab(aligned_ft, side_ts[:, 1, tongue_ix], target_absolute)
tongue[trial], _ = velocity_from_xy(x, y, tongue=True)
```

iii. Step 5 mapping: "Side-camera DLC `tongue` x/y → `output[3]` tongue velocity", with reference functions `findVideoOffset`, `findPosition`, `findVelocity`. The reference `params.traj_features` lists `tongue` as the side-view tongue feature and treats each view's features separately, so the AI follows the reference code's per-view convention rather than fusing the two views.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps, deliberately mirroring `findPosition.m`/`findVelocity.m`: (1) trials whose `NdroppedFrames` is NaN or whose `frameTimes` are missing/all-NaN are skipped entirely; (2) the raw x and y traces are linearly interpolated from the clock-corrected frame times onto the 500 bin centres, with NaN outside the covered range (`interp1` semantics) — the tongue is explicitly **not** smoothed, as in the reference; (3) `np.gradient` of the interpolated x and y (units: pixels per 10 ms sample, not pixels/s — a monotone rescaling that does not affect percentile discretisation), with NaN gradients set to 0 exactly as the reference does ("set tongue velocity to 0 if not visible"); speed is the Euclidean norm; (4) the pre-interpolation visibility mask is then re-imposed so that bins where the tongue was untracked become NaN and later take the "not visible" class. No position filling for the tongue.

ii.
```python
def velocity_from_xy(x, y, tongue):
    visible = np.isfinite(x) & np.isfinite(y)
    if tongue:
        xf, yf = x.copy(), y.copy()
    else:
        xf, yf = fill_nearest(x), fill_nearest(y)
    ...
    xv = np.gradient(xf)
    yv = np.gradient(yf)
    if tongue:
        xv[~np.isfinite(xv)] = 0
        yv[~np.isfinite(yv)] = 0
    ...
    speed = np.hypot(xv, yv)
    speed[~visible] = np.nan
    return speed, visible
```

iii. Step 4: "Reference nearest-fills non-tongue features and preserves tongue missingness … Compute reference-style velocity on filled coordinates, but retain the pre-fill visibility mask and assign class 2 where raw tracking is unavailable." Step 5 Key Decision 5: "Use missingness before reference nearest-fill to assign class 2. Filling remains necessary to reproduce derivative processing, but must not erase the requested not-visible state." Note the side effect the AI reports in Step 10: because a visible sample adjacent to an invisible one gets a NaN gradient that is zeroed, many visible bins have speed exactly 0, and in 6 of 12 sessions the session median is therefore 0.0 — "mathematically required by `< median` versus `>= median`, not a conversion error".

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per session: the 50th percentile of all finite (i.e. visible) tongue-speed samples over the retained trials and the whole 500-bin window. Samples `< threshold` → 0, `>= threshold` → 1 (ties go to the high class, exactly as "< 50th" vs ">= 50th" requires), NaN (untracked) → 2. If nothing is finite the whole session is class 2. Realised distribution: 0.029 / 0.052 / 0.919; in the 6 sessions whose median is 0.0 the class-0 fraction is exactly 0.000, so those sessions have only two effective classes.

ii.
```python
def discretize_session(values, retained):
    subset = values[retained]
    finite = np.isfinite(subset)
    if not finite.any():
        return np.full(subset.shape, 2, dtype=np.int64), np.nan
    threshold = float(np.nanpercentile(subset, 50))
    labels = np.full(subset.shape, 2, dtype=np.int64)
    labels[finite & (subset < threshold)] = 0
    labels[finite & (subset >= threshold)] = 1
    return labels, threshold
```

iii. Step 5 Key Decision 7: "Compute each median once per session from all finite visible samples among retained trials. Values equal to the median are class 1, exactly matching `<50th` versus `>=50th`." Step 10 documents the empty class-0 sessions as an arithmetic consequence of the zero-inflated tongue speed rather than a bug.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A single session-wide video clock offset is computed exactly as `findVideoOffset.m` — `mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)` — and each trial's side-camera `frameTimes` are shifted by that offset and by the trial's `goCue`. The resulting frame times are the x-axis for the `interp1` onto the shared 500-bin centre grid, so tongue bins and neural bins are the same intervals. Realised offsets are 0.490 s (8 sessions) and 0.990 s (the 4 JEB19 sessions).

ii.
```python
bit_start = np.asarray(f["obj/bp/ev/bitStart"]).ravel()
sglx_start = np.asarray(f["obj/sglx/bitcode/bitstart"]).ravel()
fs = float(np.asarray(f["obj/sglx/fs"])[0, 0])
vidshift = matlab_mode(sglx_start) / fs - matlab_mode(bit_start)
```
```python
aligned_ft = side_ft - vidshift - go[trial]
x = interp_matlab(aligned_ft, side_ts[:, 0, tongue_ix], target_absolute)
```

iii. Step 10 Check 3(c): "video `frameTimes-vidshift-goCue` matches `findPosition`/`loadMotionEnergy` and `findVideoOffset`." The `--show-processing` plots were inspected for alignment: "Tongue visibility begins primarily after the cue as expected … No temporal shift or threshold anomaly was found."

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The bottom-camera (`obj.traj{2}`) DeepLabCut features `top_paw` **and** `bottom_paw` (whichever of the two exist in `featNames`), their x/y columns of `ts`, and that view's `frameTimes`; plus the same video offset and go cue. Absence of both paw features raises an error.

ii.
```python
paw_indices = [bottom_names.index(x) for x in ("top_paw", "bottom_paw") if x in bottom_names]
if not paw_indices:
    raise ValueError(f"Bottom-view paw features unavailable: {bottom_names}")
...
for feature_ix in paw_indices:
    x = interp_matlab(aligned_ft, bottom_ts[:, 0, feature_ix], target_absolute)
    y = interp_matlab(aligned_ft, bottom_ts[:, 1, feature_ix], target_absolute)
    speed, _ = velocity_from_xy(x, y, tongue=False)
    paw_speeds.append(speed)
```

iii. Step 5 Key Decision 6: "Average speed magnitude of the two bottom-view paw landmarks using whichever are visible; class 2 only when neither is visible. This avoids arbitrarily favoring a landmark while remaining invariant to marker naming." Both markers are in the reference `params.traj_features` bottom-view list.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same pipeline as the tongue but with the reference's non-tongue branch: clock-correct, interpolate x/y to the 500 bin centres, **nearest-fill** missing samples (`fillmissing(...,'nearest')`), take `np.gradient` of both coordinates and subtract the baseline drift term `median(diff(xy))[0]` from *both* gradients (faithfully reproducing the reference's quirk of subtracting the x-component from the y-gradient too), take the Euclidean norm, then re-impose the pre-fill visibility mask as NaN. The per-marker speeds are averaged bin-wise over whichever markers are visible. Units are pixels per 10 ms sample; no cross-view normalisation is needed since only one camera is used.

ii.
```python
else:
    # Match findVelocity.m, including its subtraction of basederiv(1)
    # from both coordinate gradients.
    baseline_x = np.nanmedian(np.diff(np.column_stack((xf, yf)), axis=0), axis=0)[0]
    xv -= baseline_x
    yv -= baseline_x
```
```python
stack = np.stack(paw_speeds)
count = np.sum(np.isfinite(stack), axis=0)
summed = np.nansum(stack, axis=0)
paw[trial] = np.divide(summed, count, out=np.full(N_TIME, np.nan), where=count > 0)
```

iii. Step 5 mapping: "Clock correct/interpolate; reference nearest-fill for velocity; baseline-gradient correction; average available marker speeds; session median over visible retained samples; restore visibility mask", citing `findPosition`/`findVelocity`.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The same `discretize_session`: per-session 50th percentile of the finite paw-speed samples over retained trials and the full window; `<` → 0, `>=` → 1, untracked → 2. Session thresholds range 0.246–0.675 px/sample and the realised distribution is well balanced: 0.492 / 0.492 / 0.016.

ii.
```python
paw_labels, paw_threshold = discretize_session(paw, retained)
```

iii. Same rationale as 7-c (Step 5 Key Decision 7); the decoder spec mandates the 50th-percentile per-session split and a "not visible" third class.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue, but with the **bottom** camera's own `frameTimes` (the two views can have different frame counts), shifted by the same session-wide video offset and the trial's go cue, then interpolated onto the shared 500-bin centre grid.

ii.
```python
bottom_ts, bottom_ft = load_trial_traj(f, bottom, trial)
if bottom_ts is not None:
    aligned_ft = bottom_ft - vidshift - go[trial]
```

iii. Step 10 Check 3(c): the same `frameTimes − vidshift − alignEvent` rule as `findPosition.m`; the processing plots show paw and motion traces "share the corrected time base".

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file beside each session, whose `me.data` is a cell array with one 400 Hz trace per trial (one value per camera frame). The copy sometimes present in `obj.me` is not used, and the file's manual `me.moveThresh` is deliberately ignored. One level of extra wrapping (`me.data.data`) is unwrapped.

ii.
```python
def load_motion_file(path, n_trials):
    loaded = scipy_io.loadmat(path, simplify_cells=True)["me"]
    raw = loaded["data"]
    if isinstance(raw, dict) and "data" in raw:
        raw = raw["data"]
    raw = np.atleast_1d(raw)
    out = []
    for trial in range(n_trials):
        if trial >= raw.size:
            out.append(None); continue
        arr = np.asarray(raw[trial], dtype=np.float64).ravel()
        out.append(arr if arr.size else None)
    return out
```

iii. Step 2: "Each `motionEnergy_*.mat` is MATLAB v5 and contains `me.data`, an object array with one 400-Hz vector per trial, and a session movement threshold (`moveThresh`)." The unwrap mirrors `loadMotionEnergy.m`'s `if isstruct(me.data), me.data = me.data.data; end`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond resampling: the per-frame value is already the 99th-percentile-across-pixels scalar computed upstream, so it is only linearly interpolated onto the 500 bin centres (NaN outside coverage) and then discretised. Unlike `loadMotionEnergy.m` the AI does not `fillmissing(...,'nearest')` the leading edge NaNs; those bins become the "no video" class instead (0.7% of bins).

ii.
```python
me = motion_trials[trial]
# Reference uses side-camera frame times for motion-energy alignment.
if me is not None and side_ft is not None:
    aligned_ft = side_ft - vidshift - go[trial]
    motion[trial] = interp_matlab(aligned_ft, me, target_absolute)
```

iii. Step 5 mapping: "`motionEnergy_*.mat: me.data` → `output[5]` motion energy; clock correct and linearly interpolate exactly as `loadMotionEnergy`; session median over available retained samples"; the spec's third class is "no video", so unavailable samples are marked rather than filled.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Again `discretize_session`: per-session 50th percentile of the finite interpolated samples over retained trials, `<` → 0, `>=` → 1, missing → 2. The session thresholds (7.5–19.4 a.u.) replace the authors' manually chosen bimodal `me.moveThresh`. Distribution 0.495 / 0.497 / 0.007.

ii.
```python
motion_labels, motion_threshold = discretize_session(motion, retained)
```

iii. Step 4 discrepancy table: "Paper manually sets a bimodal movement threshold; separate files provide `moveThresh`; requested output explicitly mandates 50th percentile → Use the task-mandated per-session median over aligned valid motion-energy samples, not `moveThresh`; reserve class 2 for missing video."

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy has one value per **side**-camera frame, so it is timed with the side view's `frameTimes`, corrected by the same session video offset and the trial's go cue, and interpolated onto the shared bin-centre grid. If the side-camera frame times are unusable the code falls back to a synthetic 400 Hz axis shifted by 0.5 s — exactly the `catch` branch of `loadMotionEnergy.m`.

ii.
```python
elif me is not None:
    fallback = np.arange(1, me.size + 1, dtype=np.float64) / 400.0
    motion[trial] = interp_matlab(fallback - 0.5 - go[trial], me, target_absolute)
```

iii. Step 10 Check 2 verifies this end to end: "Motion output: independently loaded the raw motion file and frame timestamps, recomputed clock correction/alignment and the session median, and matched all 500 categorical samples for session 0/trial 0; threshold matched to `1e-12`."

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Everything is handled by keeping the trial and marking the gap, never by inventing data: (a) trials flagged with NaN `NdroppedFrames`, with an empty `ts`, or with missing/all-NaN `frameTimes` are skipped for that camera, leaving 500 "not visible"/"no video" bins; (b) a non-monotonic or <2-sample time base makes `interp_matlab` return all-NaN instead of producing garbage; (c) mismatched lengths between `frameTimes` and `ts`/motion traces are truncated to the shorter; (d) missing DeepLabCut samples are nearest-filled only for non-tongue features (as the reference does) but the pre-fill mask always decides the "not visible" class; (e) motion-energy traces absent for a trial index become class 2; (f) spikes with non-finite times or out-of-range trial numbers are dropped; (g) per-trial Bpod arrays whose length disagrees with `Ntrials`, unlabelled cluster qualities, and non-finite converted neural values are all checked, the last two by explicit `raise`.

ii.
```python
def load_trial_traj(f, traj, trial):
    if traj is None or trial >= traj["ts"].shape[0]:
        return None, None
    if "NdroppedFrames" in traj:
        dropped = h5_vector(f, traj["NdroppedFrames"][trial, 0])
        if dropped.size and np.isnan(dropped[0]):
            return None, None
    ...
    if frame_times.size == 0 or np.all(~np.isfinite(frame_times)):
        return None, None
```
```python
def interp_matlab(x, y, target):
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    ...
    if x.size < 2 or np.any(np.diff(x) <= 0):
        return np.full(target.shape, np.nan, dtype=np.float64)
```
```python
if not np.isfinite(neural_trial).all():
    raise ValueError(f"{session_id}: non-finite neural data")
```

iii. Step 5 Key Decision 2: "Do not drop missing-video trials; encode their behavior outputs as class 2. This preserves neural/outcome data while truthfully marking behavior availability." Step 10 Check 5 lists the edge-case audit (histc edge semantics, 1-based indices, median ties, minimum retained unit rate 1.00577 Hz).

## 11-a. What are the most time-consuming steps of the code?

i. The AI reports per-session timings (1.8–3.3 s, 27.9 s total for 12 sessions) and attributes the cost to file reading, noting it avoided `mat73`-style whole-object loading. I profiled `convert_data.py` directly on JEB6: `load_behavior_streams` takes 2.24 s of the 2.53 s session total, versus 0.25 s for `load_neural` and 0.02 s for the motion-energy file — i.e. ~90% of the runtime is the per-trial loop that reads each camera's `ts`/`frameTimes` out of HDF5 and interpolates them. Conversion is fast enough that the AI never needed further optimisation.

ii.
```python
for trial in range(n_trials):
    side_ts, side_ft = load_trial_traj(f, side, trial)      # 2 HDF5 dereferences/trial
    ...
    bottom_ts, bottom_ft = load_trial_traj(f, bottom, trial)
```
```python
print(f"[{session_id}] native trials={n_trials}, retained={retained.size}, "
      f"units={neural_all.shape[1]}, thresholds={thresholds}, time={elapsed:.2f}s", flush=True)
```

iii. Step 6: "Loading full 100-300 MB MATLAB objects via `mat73` would materialize waveforms and all DLC features, and smoothing one spike train/trial at a time would create excessive Python overhead"; Step 7 estimated ~57 s for the full run and Step 9 recorded 40.78 s, later 27.91 s.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI vectorised the two loops that were vectorisable: spike binning is a single `np.add.at` over all spikes of a unit across all trials, and smoothing is one sliding-window `einsum` over the whole (trials × time) matrix. What remains are (1) the loop over clusters in `load_neural`, (2) the loop over trials in `load_behavior_streams`, which does two HDF5 reads and four `np.interp` calls per trial, and (3) the loop over retained trials that packs the output arrays. Loop (2) is the one that dominates runtime; it is hard to vectorise because each trial has a different number of camera frames, but the per-trial HDF5 dereferencing could have been batched, and the loops over trials/clusters are independent so the sessions could have been processed in parallel (the AI runs them sequentially).

ii.
```python
bins = np.searchsorted(EDGES, aligned, side="right") - 1
np.add.at(counts, (tr0[valid], bins[valid]), 1)     # all trials at once
```
```python
windows = np.lib.stride_tricks.sliding_window_view(
    np.pad(padded, pad_width, mode="constant"), KERNEL.size, axis=-1)
smoothed = np.einsum("...k,k->...", windows, KERNEL[::-1], optimize=True)
```

iii. Step 6 "Code speedups added: Use HDF5 field/reference access, vectorized spike binning (`np.add.at`), 2-D convolution across all trials, single-pass per-view video reads, float32 neural storage, and sequential session release."

## 11-c. What processing does the code repeat multiple times?

i. Very little. Each data file is opened once, the video offset is computed once per session, feature-name lists are resolved once per view, the bin grid and smoothing kernel are module-level constants, and the three session thresholds are computed once each. The genuine repeats are small: the side-camera aligned frame times are recomputed for motion energy after having been computed for the tongue in the same loop iteration; `velocity_from_xy` recomputes the `visible` mask that the caller then discards; and the smoothed rates are computed for every quality-passing cluster before the >1 Hz test rejects 10 of 528. Nothing is loaded twice.

ii.
```python
aligned_ft = side_ft - vidshift - go[trial]        # for the tongue
...
if me is not None and side_ft is not None:
    aligned_ft = side_ft - vidshift - go[trial]    # recomputed for motion energy
```
```python
vidshift = matlab_mode(sglx_start) / fs - matlab_mode(bit_start)   # once per session
```

iii. Step 6 claims "single-pass per-view video reads"; the repeated arithmetic above is trivial relative to the HDF5 reads it sits between and was not flagged in CONVERSION_NOTES.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main waste is that the whole pipeline runs over **all native trials** and only then selects the retained ones: `bin_one_unit`/`smooth_reference` build `(n_trials, 500)` rate matrices for every cluster, and the video loop computes tongue/paw/motion for every trial, but 510 of 3,626 trials (14%) are thrown away by the early-lick/photostim filter. Similarly, the rate is computed from *smoothed* data for all 528 quality-passing clusters before 10 of them are rejected (the mean could have been taken on raw counts). Smaller items: the `visible` mask returned by `velocity_from_xy` is discarded at every call site, the `quality_counts` Counter and the full per-unit `unit_info` list exist only for metadata, and `bottom_paw` speed is computed even where `top_paw` alone would decide the class. None of these is significant at 28 s total runtime.

ii.
```python
counts = bin_one_unit(trials, trial_times, go, n_trials)   # all native trials
rates = smooth_reference(counts / DT).astype(np.float32)
...
for trial in range(n_trials):                              # all native trials
    side_ts, side_ft = load_trial_traj(f, side, trial)
...
subset = values[retained]                                  # only now are 14% dropped
```
```python
tongue[trial], _ = velocity_from_xy(x, y, tongue=True)     # visibility mask discarded
```

iii. Not documented in CONVERSION_NOTES; the AI's efficiency discussion (Step 6/7) concentrates on avoiding whole-object MATLAB loads and on vectorising the spike binning and smoothing, and judged the resulting 27.9 s runtime comfortably inside the 15-minute budget, so it did not pursue the remaining waste.
