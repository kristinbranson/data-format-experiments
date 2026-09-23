# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI restricts the conversion to a single folder, `data/Ephys_Behavior`, and within it to the seven animals it hard-codes in `ALM_PROBE`. `session_files` globs `data_structure_*.mat`, parses `<anm>_<date>` out of the filename with a regex, keeps a file only if its animal is a key of `ALM_PROBE`, and then asserts that exactly 12 files survived. Each session file is opened once with `h5py` (v7.3 only — there is no `scipy.io` fallback for the data structure), and its motion-energy sibling `motionEnergy_<anm>_<date>.mat` is opened with `scipy.io.loadmat`. The probe number is looked up per *animal*, not per session. `data/RandomizedDelay_Ephys_Behavior` is never touched, and the 13 fixed-delay sessions of JEB13/JEB14/JEB15 are excluded. Result: 12 sessions, 7 subjects, 3,116 trials, 520 units.

ii.
```python
DATA_DIR = Path(__file__).resolve().parent / "data" / "Ephys_Behavior"

# ALM probe selection copied from code/DataLoadingScripts/Recording and video.
# These are the sessions loaded by both supplied Figure 8 two-context scripts.
ALM_PROBE = {
    "JEB6": 2, "JEB7": 1, "EKH1": 2, "EKH3": 2,
    "JGR2": 1, "JGR3": 1, "JEB19": 1,
}

def session_files(data_dir: Path) -> list[tuple[Path, str, str]]:
    sessions = []
    pattern = re.compile(r"data_structure_([^_]+)_(\d{4}-\d{2}-\d{2})\.mat$")
    for path in sorted(data_dir.glob("data_structure_*.mat")):
        match = pattern.match(path.name)
        if match and match.group(1) in ALM_PROBE:
            sessions.append((path, match.group(1), match.group(2)))
    if len(sessions) != 12:
        raise RuntimeError(f"Expected 12 two-context sessions, found {len(sessions)}")
    return sessions
```
```python
    with h5py.File(path, "r") as h5:
        bp = h5["obj/bp"]
```

iii. From the trajectory (step 10): *"The repository separates three experimental collections. The requested WC/DR context output points to the fixed-delay electrophysiology collection (`Ephys_Behavior`), not the randomized-delay or optogenetic behavior-only cohorts."* And step 21: *"The paper's exact two-context neural cohort is identifiable from the figure scripts: 12 sessions from six mice (the reported 522-unit cohort)."* The seven animals are exactly the ones loaded by `Scripts/Figure 8/Figure8a_thru_c.m` and `Figure8d.m`, and the paper (`methods.txt`) does state "In 12 sessions from six mice, animals performed the two-context task. In total, 522 units … were recorded". The AI tried to confirm the cohort text directly from `paper.pdf` at step 42 but `pdftotext` was not installed, so the claim was never actually verified against the paper; it relied on the figure scripts alone. No rationale was recorded for keying the probe by animal rather than by session, nor for discovering sessions by globbing rather than transcribing the `load<ANM>_ALMVideo.m` manifests.

## 1-b. How are the data split into subjects?

i. The subject is the `<anm>` capture group of the filename regex. `convert` collects the sorted unique set as `subjects` and stores a per-session index into it as `subject_idx`. Within its 12-session cohort this yields 7 subjects (EKH1, EKH3, JEB19, JEB6, JEB7, JGR2, JGR3).

ii.
```python
    pattern = re.compile(r"data_structure_([^_]+)_(\d{4}-\d{2}-\d{2})\.mat$")
    ...
            sessions.append((path, match.group(1), match.group(2)))
```
```python
    subjects = sorted({animal for _, animal, _ in files})
    subject_lookup = {subject: i for i, subject in enumerate(subjects)}
    ...
        subject_idx.append(subject_lookup[animal])
    ...
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int64),
```

iii. No explicit justification is recorded. The animal id is implicitly taken from the filename, which is also how the authors' `load<ANM>_ALMVideo.m` scripts identify animals and is the only place the animal is reliably stored. The AI did not notice that its 7 subjects disagree with the "six mice" its own step-21 message attributed to the paper.

## 1-c. How are the data split into sessions?

i. One `data_structure_*.mat` file is one session, and each session becomes one element of `neural`, `input`, `output`, `brain_region_idx`, `subject_idx`, and `metadata['session_info']`. Sessions are processed one at a time by `process_session`, in sorted filename order. Because only `Ephys_Behavior` is scanned and only seven animals are accepted, the session axis has length 12.

ii.
```python
    for number, (path, animal, date) in enumerate(files, start=1):
        print(f"[{number:02d}/{len(files)}] {animal} {date}", flush=True)
        result = process_session(path, animal, date)
        neural.append(result.pop("neural"))
        inputs.append(result.pop("input"))
        outputs.append(result.pop("output"))
        subject_idx.append(subject_lookup[animal])
        brain_region_idx.append(np.zeros(result["n_units"], dtype=np.int64))
        ...
        session_info.append({"subject": animal, "date": date, "source_file": path.name, **result})
```

iii. Same as 1-a: the session set is the Figure 8 two-context cohort. The metadata field records it as `"cohort": "12-session two-context electrophysiology cohort loaded by the supplied Figure 8 scripts"`.

## 1-d. How are the data split into trials?

i. A trial is one entry of the per-trial `obj.bp` arrays. The trial count is taken as `len(bp['ev/goCue'])` (not `bp.Ntrials`), and every other per-trial flag (`hit`, `miss`, `no`, `early`, `autowater`, `R`, `stim/enable`) is read as a flat array of the same length. Spikes carry their trial number in `clu.trial` (1-based, converted to 0-based and range-checked), and camera frames are already stored one cell per trial, so no trial boundaries are reconstructed. Surviving trials are kept as `kept_trials` (original indices) with `original_to_kept` mapping originals to positions in the output arrays.

ii.
```python
        go = np.asarray(bp["ev/goCue"]).ravel().astype(np.float64)
        n_trials = go.size
        hit = np.asarray(bp["hit"]).ravel().astype(bool)
        ...
        kept_trials = np.flatnonzero(trial_mask)
        ...
        original_to_kept = np.full(n_trials, -1, dtype=np.int64)
        original_to_kept[kept_trials] = np.arange(kept_trials.size)
```
```python
            spike_trial = referenced_array(h5, cluster_group["trial"][unit, 0]).ravel().astype(np.int64) - 1
            valid = (spike_trial >= 0) & (spike_trial < n_trials)
```

iii. No explicit justification recorded. Using `goCue`'s length instead of `bp.Ntrials` is safe here: I verified that for all 12 sessions `Ntrials`, `len(goCue)`, all `bp` flags and both cameras' trial counts are identical.

## 1-e. How are trials filtered based on quality controls?

i. One mask, applied before anything is computed: drop early-lick trials (`bp.early`), drop photostimulation trials (`bp.stim.enable`), require a finite go cue, and require the trial to be one of hit/miss/ignore. Ignore trials are deliberately kept because "ignore" is a requested outcome class. A session with fewer than two surviving trials raises. There is no cut for trials that run past the end of the ephys recording. Across the 12 sessions this keeps 3,116 of 3,626 trials.

ii.
```python
        # Paper analyses use control trials and omit early licks.  Ignore trials
        # remain because ignore is a required decoder output in this task.
        trial_mask = ~early & ~stim & np.isfinite(go) & (hit | miss | ignore)
        kept_trials = np.flatnonzero(trial_mask)
        if kept_trials.size < 2:
            raise RuntimeError(f"{animal} {date} has fewer than two usable trials")
```

iii. Step 28: *"Two additional paper-matching curation rules are now confirmed: only non-photostimulation control trials are used, and ALM units are restricted to the manifest-selected probe, accepted quality labels, and firing rate above 1 Hz."* The module docstring states *"trials with early licks or optogenetic stimulation are excluded"*, and the methods text says early-lick trials "were omitted from analyses". Retention of ignore trials is justified in the docstring: *"The requested ignore outcome is deliberately retained (unlike analyses in the paper that only compare hits/misses)."* The `isfinite(go)` and `(hit|miss|ignore)` terms are no-ops on this cohort (I verified there are no NaN go cues and the three flags are mutually exclusive and exhaustive on every trial). No recording-length check was considered; I verified it is not needed here — none of the 3,116 trials is all-zero across all units.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` on the single probe selected for that animal by `ALM_PROBE`, using three per-cluster fields: `quality` (manual curation string), `trial` (1-based trial of each spike) and `trialtm` (spike time relative to that trial's start). `obj.bp.ev.goCue` supplies the alignment times. Only one probe is ever read per session.

ii.
```python
        probe_number = ALM_PROBE[animal]
        cluster_group = h5[h5["obj/clu"][probe_number - 1, 0]]
        for unit in range(cluster_group["quality"].shape[0]):
            quality = matlab_char(h5, cluster_group["quality"][unit, 0])
            ...
            spike_trial = referenced_array(h5, cluster_group["trial"][unit, 0]).ravel().astype(np.int64) - 1
            trial_time = referenced_array(h5, cluster_group["trialtm"][unit, 0]).ravel().astype(np.float64)
            ...
            aligned_time = trial_time - go[spike_trial]
```

iii. The docstring says the probe numbers "come from the Figure 8 scripts". `trialtm` is already on the behaviour clock and relative to trial start, so it needs only the go-cue subtraction. No session in this cohort is listed with two probes in the authors' manifests, so the single-probe lookup is sufficient here.

## 2-b. How is the `neural` data processed?

i. Aligned spike times are counted into 10 ms bins over −2.5 to +2.5 s (`histogram_unit`), assembled into a `(trials, units, 500)` count array, and then passed through `smooth_rates`, which reproduces the repository's `mySmooth(x, 15, 'reflect')`: a `gausswin(15)` (alpha 2.5) whose first 7 taps are zeroed to make it causal, renormalised, applied as a causal FIR filter, with the "reflect" boundary handling of `mySmooth` (which actually prepends the first 15 samples unreversed and then trims them). The result is divided by the bin width, so stored values are firing rates in Hz, stored as `float32`. No normalisation, baseline subtraction or z-scoring. Only one probe contributes, so no concatenation across probes.

ii.
```python
def gaussian_causal_kernel(n: int = SMOOTH_BINS) -> np.ndarray:
    """Reproduce MATLAB gausswin(n), then the causalization in mySmooth.m."""
    # MATLAB gausswin's default alpha is 2.5.
    x = np.arange(n, dtype=np.float64) - (n - 1) / 2
    kernel = np.exp(-0.5 * (2.5 * x / ((n - 1) / 2)) ** 2)
    kernel[: n // 2] = 0
    kernel /= kernel.sum()
    # With conv(..., 'same'), the center coefficient acts on the current bin.
    return kernel[n // 2 :].astype(np.float32)


def smooth_rates(counts: np.ndarray) -> np.ndarray:
    """Apply mySmooth(..., 15, 'reflect') along the final dimension."""
    # mySmooth prepends the first N samples in their original order.  Although
    # called "reflect" in the source, this exact behavior is what we reproduce.
    padded = np.concatenate((counts[..., :SMOOTH_BINS], counts), axis=-1)
    filtered = lfilter(CAUSAL_KERNEL, [1.0], padded, axis=-1)
    return (filtered[..., SMOOTH_BINS:] / DT).astype(np.float32)
```
```python
def histogram_unit(aligned_time, spike_trial, n_trials):
    result = np.zeros((n_trials, N_TIME), dtype=np.float32)
    bins = np.floor((aligned_time - TMIN) / DT).astype(np.int64)
    valid = ((spike_trial >= 0) & (spike_trial < n_trials)
             & (bins >= 0) & (bins < N_TIME))
    np.add.at(result, (spike_trial[valid], bins[valid]), 1.0)
    return result
```

iii. Docstring: *"spikes are aligned to `bp.ev.goCue`, binned at 10 ms, and smoothed with the repository's 15-bin causal Gaussian kernel and reflect boundary mode"*; metadata: `"neural_representation": "10 ms firing rates smoothed with the source 15-bin causal Gaussian kernel"`. The repository's `getDefaultParams.m`, `Figure8a_thru_c.m` and `Figure8d.m` all set `params.smooth = 15` and `params.bctype = 'reflect'`. The AI also noted in a code comment that `mySmooth`'s "reflect" is not a true reflection and chose to copy the actual behaviour rather than the name.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three filters. (1) The curation label is compared **case-sensitively and exactly** against `{garbage, gabrga, noisy, real?}`, matching `findClusters.m`'s `ismember`; the AI comments that this is deliberate. (2) A >1 Hz mean firing rate criterion computed the way `removeLowFRClusters.m` does it: build a smoothed PSTH in Hz for each of seven Figure 8 conditions, average each over time, then average across conditions, and keep the unit if that scalar exceeds 1. (3) A session-level rule: fewer than 10 curated units raises an error (the paper's "Recording sessions were included for analysis only if they had at least 10 units"). This leaves 520 of the cohort's 2,314 clusters, 27–67 per session.

ii.
```python
REJECTED_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}
...
            quality = matlab_char(h5, cluster_group["quality"][unit, 0])
            # Deliberately case-sensitive, matching findClusters.m/isMember.
            if quality in REJECTED_QUALITIES:
                continue
            ...
            mean_rate = unit_mean_rate(aligned_time, spike_trial, condition_masks)
            if mean_rate > 1.0:
                selected_units.append((aligned_time, spike_trial, quality, mean_rate))

        if len(selected_units) < 10:
            raise RuntimeError(f"{animal} {date} has only {len(selected_units)} curated units")
```
```python
def unit_mean_rate(aligned_time, spike_trial, condition_masks) -> float:
    """Match removeLowFRClusters using the Figure 8 condition PSTHs."""
    condition_means = []
    edges = np.arange(TMIN, TMAX + DT / 2, DT)
    for mask in condition_masks:
        n_trials = int(mask.sum())
        selected = mask[spike_trial] if spike_trial.size else np.zeros(0, bool)
        hist = np.histogram(aligned_time[selected], bins=edges)[0].astype(np.float32)
        if n_trials:
            psth = smooth_rates(hist[None, :])[0] / n_trials
        else:
            psth = np.zeros(N_TIME, dtype=np.float32)
        condition_means.append(float(psth.mean()))
    return float(np.mean(condition_means))
```
```python
        # Figure 8 condition definitions used by the source low-FR filter.
        condition_masks = [
            hit | miss | ignore,
            hit & ~autowater, hit & autowater,
            miss & ~autowater, miss & autowater,
            hit & ~autowater & ~early, hit & autowater & ~early,
        ]
```

iii. Step 28: *"ALM units are restricted to the manifest-selected probe, accepted quality labels, and firing rate above 1 Hz."* Metadata: `"manifest-selected ALM probe; exclude exact quality labels garbage, gabrga, noisy, real?; retain units with source-style mean firing rate >1 Hz"`. The AI checked at step 43 that the differently-cased `'Poor'` labels present in this cohort all fall below 1 Hz and are removed by the rate criterion anyway. It did not check `'Noisy'`: JEB6_2021-04-18 carries one capitalised `Noisy` cluster which the case-sensitive filter lets through (MATLAB's `ismember` in `findClusters.m` is likewise case-sensitive, so this faithfully reproduces the source's own behaviour).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. A single subtraction: each spike's `trialtm` minus the `goCue` of its own trial. Nothing else — no interpolation, no per-session clock correction (the neural stream is already on the behaviour clock; only the camera streams get the video offset).

ii.
```python
            spike_trial = referenced_array(h5, cluster_group["trial"][unit, 0]).ravel().astype(np.int64) - 1
            trial_time = referenced_array(h5, cluster_group["trialtm"][unit, 0]).ravel().astype(np.float64)
            valid = (spike_trial >= 0) & (spike_trial < n_trials)
            spike_trial, trial_time = spike_trial[valid], trial_time[valid]
            aligned_time = trial_time - go[spike_trial]
```

iii. Docstring: *"spikes are aligned to `bp.ev.goCue`"*. Metadata: `"temporal_alignment_event": "Bpod goCue onset (auditory go cue in DR; corresponding water-delivery event in WC), using obj.bp.ev.goCue"`. This is what `alignSpikes.m` does with `params.alignEvent = 'goCue'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 10 ms bins (`DT = 0.01`), 500 non-overlapping bins spanning −2.5 to +2.5 s from the go cue, identical for every trial and session. `TIME` holds the bin centres (−2.495 … 2.495 s). No rebinning is applied after the initial binning — spikes are histogrammed directly at 10 ms, and the camera streams are interpolated directly onto the same 500-point axis. `metadata['time_bin_size'] = 10.0` ms, `off_start = -2.5`, `off_end = 2.5`.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 0.01
SMOOTH_BINS = 15
TIME = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2
N_TIME = TIME.size
```

iii. Step 28: *"Neural activity will use the repository's 10 ms bins and 15-bin causal Gaussian smoothing over −2.5 to +2.5 s."* `Figure8a_thru_c.m` sets `params.dt = 1/100`; the window −2.5 to 2.5 is `getDefaultParams.m`'s `params.tmin`/`tmax` (the Figure 8 scripts themselves use `tmin = -3`). The repository is internally inconsistent about `dt` (`getDefaultParams.m` and `Figure8d.m` use `1/200`), and the AI picked the Figure 8a–c value without commenting on the discrepancy.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Not derived from any raw variable. It is the analysis time axis itself, defined by the constants `TMIN`, `TMAX`, `DT`; its meaning comes from the alignment event, `bp.ev.goCue`, which was subtracted from every stream.

ii.
```python
TIME = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2
N_TIME = TIME.size
```

iii. The decoder task specifies "Time from go cue onset in seconds (continuous, time-varying)" as the sole input, and the alignment event is the go cue. `input_names` is `["time_from_go_cue_s"]`.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond taking the bin centres and casting to `float32`. The same 1×500 vector is copied for every trial of every session, giving `input` shape `(1, 500)` per trial.

ii.
```python
    time_input = TIME.astype(np.float32)[None, :]
    for local_trial, original_trial in enumerate(kept_trials):
        neural_trials.append(np.ascontiguousarray(rates[local_trial], dtype=np.float32))
        input_trials.append(time_input.copy())
```

iii. No justification recorded; none needed — the axis is defined by the conversion, not read from data. Note the input is left continuous rather than converted to a binary indicator, which is what the instructions ask for ("continuous, time-varying").

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. By construction it *is* the neural time axis. `histogram_unit` assigns each spike to bin `floor((t − TMIN)/DT)`, and `TIME[k]` is the centre of that same bin, so index *k* of the input and index *k* of the neural matrix refer to the same 10 ms interval. The camera streams are interpolated onto `TIME` as well, so all four streams share one axis.

ii.
```python
    bins = np.floor((aligned_time - TMIN) / DT).astype(np.int64)
```
```python
TIME = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2
```

iii. No separate justification; the shared grid is the mechanism.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial `obj.bp` flags: `R` (the instructed/rewarded side), and the outcome flags `hit`, `miss`, `no`. The actual lick direction is not recorded anywhere, so it is inferred from instructed side × outcome.

ii.
```python
        hit = np.asarray(bp["hit"]).ravel().astype(bool)
        miss = np.asarray(bp["miss"]).ravel().astype(bool)
        ignore = np.asarray(bp["no"]).ravel().astype(bool)
        right_target = np.asarray(bp["R"]).ravel().astype(bool)
```

iii. Step 21: *"derive lick direction from target side plus outcome, so incorrect trials are labeled by the animal's actual opposite-side response."*

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. On an ignore trial the animal did not lick → class 2 (`none`). On a hit it licked the instructed port → `right` (1) if `R` else `left` (0). On a miss it licked the other port → `left` (0) if `R` else `right` (1). The scalar is tiled across all 500 bins so it can live in the same array as the time-varying outputs. `output_values[0] = ["left", "right", "none"]`.

ii.
```python
        if ignore[original_trial]:
            lick_direction = 2  # none
            outcome = 2
        elif hit[original_trial]:
            lick_direction = 1 if right_target[original_trial] else 0
            outcome = 1
        else:
            # R/L identifies instructed/reward side, so an incorrect response is
            # the opposite side and is the animal's actual lick direction.
            lick_direction = 0 if right_target[original_trial] else 1
            outcome = 0
        ...
        output = np.empty((6, N_TIME), dtype=np.int64)
        output[0] = lick_direction
```

iii. Inline comment above, plus the docstring: *"Trial-level labels are repeated in time so they can coexist with the requested time-varying movement outputs."* Codes follow the prompt's ordering (left, right, none).

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial flag, `obj.bp.autowater`, which marks the water-cued block.

ii.
```python
        autowater = np.asarray(bp["autowater"]).ravel().astype(bool)
```

iii. No explicit justification recorded; the field maps directly onto the WC/DR distinction described in the methods ("a drop of water was presented at a random point in time at a randomly selected reward port").

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct relabelling: `autowater` → WC (0), otherwise DR (1), tiled across the 500 bins. `output_values[1] = ["WC", "DR"]`.

ii.
```python
        context = 0 if autowater[original_trial] else 1
        ...
        output[1] = context
```

iii. Codes follow the prompt's `WC, DR` ordering. The whole cohort selection (1-a) was driven by this output: the AI chose the two-context sessions precisely so that both classes are present.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The three mutually exclusive `obj.bp` flags `hit`, `miss` and `no`. Unlike the reference, `no` is read explicitly rather than inferred as the complement.

ii.
```python
        hit = np.asarray(bp["hit"]).ravel().astype(bool)
        miss = np.asarray(bp["miss"]).ravel().astype(bool)
        ignore = np.asarray(bp["no"]).ravel().astype(bool)
```

iii. No separate justification; the same branch that assigns lick direction assigns outcome, so the flags are read once.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Relabelling into three classes in the same `if/elif/else` as lick direction: ignore → 2, hit → 1 (`correct`), miss → 0 (`incorrect`), tiled across the 500 bins. `output_values[2] = ["incorrect", "correct", "ignore"]`.

ii.
```python
        if ignore[original_trial]:
            lick_direction = 2  # none
            outcome = 2
        elif hit[original_trial]:
            ...
            outcome = 1
        else:
            ...
            outcome = 0
        ...
        output[2] = outcome
```

iii. Docstring: *"The requested ignore outcome is deliberately retained (unlike analyses in the paper that only compare hits/misses)."* Codes follow the prompt's `incorrect, correct, ignore` ordering.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`, **side camera only** (`obj.traj{1}`), feature `tongue`. For each trial it reads `NdroppedFrames`, `frameTimes`, and `ts` (indexed `[feature, xyl, frame]` after the HDF5 transpose), taking the x and y rows. The bottom camera's `top_tongue` is never used. The go cue and the session video offset (`obj.sglx.bitcode.bitstart`, `obj.sglx.fs`, `obj.bp.ev.bitStart`) are the other inputs.

ii.
```python
        trajectory_refs = np.asarray(h5["obj/traj"]).ravel()
        side = h5[trajectory_refs[0]]
        bottom = h5[trajectory_refs[1]]
        tongue_index = find_feature_indices(h5, side, "tongue")
```
```python
    raw = referenced_array(h5, trajectory_group["ts"][trial, 0])
    ...
    # HDF5 dimensions are reversed relative to MATLAB: feature x xyz x frame.
    x_raw = raw[feature_index, 0, :]
    y_raw = raw[feature_index, 1, :]
```

iii. No justification is recorded anywhere for using one camera rather than both. The AI did enumerate both cameras' `featNames` at step 26 and so knew that `top_tongue`/`bottom_tongue` exist on the bottom view. Metadata only states `"DLC tongue (side camera) and top-paw (bottom camera) …"`. The consequence, visible in the verification output, is that the tongue is "not visible" in 95.7% of bins.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps. (1) If the trial's `NdroppedFrames` is empty or non-finite, the trial is skipped entirely (a whole-trial NaN), reproducing `findPosition.m`'s guard. (2) Frame times are put on the go-cue clock (7-d) and x and y are linearly interpolated onto the 500-bin axis with `left=nan, right=nan`, so NaN stretches — which in this data are exactly the frames where DeepLabCut likelihood ≤ 0.9 — stay NaN and the window outside camera coverage stays NaN. No smoothing is applied to the tongue, matching `findPosition.m` (`mySmooth(ts, 1, …)` is a no-op anyway) and no nearest-fill, matching the paper's "Missing values were filled in with the nearest available value for all features, except for the tongue." (3) Speed is `hypot(gradient(x), gradient(y))` on the interpolated grid, i.e. pixels per bin, matching `findVelocity.m`'s bare `gradient`. (4) Any bin where x or y is non-finite is forced back to NaN.

ii.
```python
def interp_preserving_missing(source_time, values, target_time):
    """Linear interpolation that preserves NaN intervals and extrapolates NaN."""
    ...
    return np.interp(target_time, source_time, values, left=np.nan, right=np.nan)


def speed_from_position(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    speed = np.hypot(np.gradient(x), np.gradient(y))
    speed[~(np.isfinite(x) & np.isfinite(y))] = np.nan
    return speed
```
```python
    dropped = referenced_array(h5, trajectory_group["NdroppedFrames"][trial, 0]).squeeze()
    if np.size(dropped) == 0 or not np.all(np.isfinite(dropped)):
        return None
    ...
            tongue_velocity.append(
                np.full(N_TIME, np.nan) if tongue is None else speed_from_position(tongue[0], tongue[1])
            )
```

iii. Docstring: *"camera and motion-energy streams use the repository's per-session video offset before interpolation onto the neural time axis"*; metadata: `"DLC tongue (side camera) and top-paw (bottom camera) Euclidean frame-to-frame velocity"`. The AI never applies an explicit likelihood threshold; I verified that in this data x/y are NaN exactly where likelihood ≤ 0.9, so the implicit rule is equivalent to the reference's explicit cut.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. `categorize_session_signal` pools every finite speed value across all trials of the session, takes the 50th percentile as the threshold, and assigns 0 below it, 1 at or above it, and 2 to every non-finite bin. If nothing is finite in the whole session the entire output is class 2. The threshold is stored per session in `metadata['session_info']`. `output_values[3] = ["below_session_median", "at_or_above_session_median", "not_visible"]`.

ii.
```python
def categorize_session_signal(signals):
    finite_parts = [x[np.isfinite(x)] for x in signals if np.any(np.isfinite(x))]
    if not finite_parts:
        return [np.full(N_TIME, 2, dtype=np.int64) for _ in signals], None
    threshold = float(np.percentile(np.concatenate(finite_parts), 50))
    categorical = []
    for signal in signals:
        out = np.full(N_TIME, 2, dtype=np.int64)
        visible = np.isfinite(signal)
        out[visible] = (signal[visible] >= threshold).astype(np.int64)
        categorical.append(out)
    return categorical, threshold
```

iii. This is directly the prompt's specification (per-session 50th percentile; 2 for "not visible"). No further justification recorded.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The camera clock leads the behaviour clock, so a per-session offset is computed once as `median(sglx.bitcode.bitstart / sglx.fs) − median(bp.ev.bitStart)`, reproducing `findVideoOffset.m` (which uses `mode` instead of `median`). Each trial's frame times become `frameTimes − offset − goCue[trial]`, and x/y are then interpolated onto the shared 500-bin axis, so tongue bin *k* is the same interval as neural bin *k*.

ii.
```python
def video_offset(h5: h5py.File) -> float:
    bit_start = np.asarray(h5["obj/bp/ev/bitStart"]).ravel()
    video_bit_start = np.asarray(h5["obj/sglx/bitcode/bitstart"]).ravel()
    fs = float(np.asarray(h5["obj/sglx/fs"]).squeeze())
    return float(np.nanmedian(video_bit_start / fs) - np.nanmedian(bit_start))
```
```python
    aligned_frame_time = frame_time[:n] - offset - go_cue
    x = interp_preserving_missing(aligned_frame_time, x_raw[:n], TIME)
    y = interp_preserving_missing(aligned_frame_time, y_raw[:n], TIME)
```

iii. Metadata: `"aligned with the source video offset"`. The offset is computed once per session in `process_session` and reused for all trials and both cameras. I verified that median and mode give bit-identical offsets on all 12 sessions (0.490040 s or 0.990016 s), so the `median`-for-`mode` substitution has no effect here.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The same `obj.traj` tracking, **bottom camera only** (`obj.traj{2}`), feature `top_paw`. `bottom_paw` is not used.

ii.
```python
        paw_index = find_feature_indices(h5, bottom, "top_paw")
        ...
            paw = trajectory_signal(h5, bottom, int(trial), paw_index, go[trial], offset)
```

iii. No justification recorded for preferring `top_paw` over `bottom_paw`. The methods state "the paws were tracked using only the bottom view", which the code respects.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Identical to the tongue: `NdroppedFrames` guard, frame times shifted by offset and go cue, x/y linearly interpolated onto the 500-bin axis with NaN preserved and NaN outside camera coverage, speed `hypot(gradient(x), gradient(y))` in pixels per bin, NaN forced back wherever x or y is missing. The reference MATLAB's extra non-tongue steps — subtracting the baseline median derivative and `fillmissing(…, 'nearest')` — are *not* applied, so untracked paw bins remain NaN and become the "not visible" class. No cross-camera normalisation (only one view is used).

ii.
```python
            paw = trajectory_signal(h5, bottom, int(trial), paw_index, go[trial], offset)
            ...
            paw_velocity.append(
                np.full(N_TIME, np.nan) if paw is None else speed_from_position(paw[0], paw[1])
            )
```

iii. No explicit justification. Keeping NaN rather than nearest-filling is required by the decoder spec, which asks for a distinct "not visible" class. The resulting "not visible" fraction is 14.5% of bins.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same `categorize_session_signal` call: session-wide 50th percentile over all finite bins, 0 below / 1 at-or-above, 2 for NaN. Threshold recorded per session (e.g. 0.387 for EKH1, 0.217 for JEB19_2023-04-18).

ii.
```python
    paw_cat, paw_threshold = categorize_session_signal(paw_velocity)
```

iii. The prompt's specification, applied identically to all three movement variables.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue, but using the bottom camera's own `frameTimes` (read from `bottom`, not `side`), with the same session offset and the same interpolation onto the shared 500-bin axis.

ii.
```python
    frame_time = referenced_array(h5, trajectory_group["frameTimes"][trial, 0]).ravel()
    ...
    aligned_frame_time = frame_time[:n] - offset - go_cue
```
```python
            paw = trajectory_signal(h5, bottom, int(trial), paw_index, go[trial], offset)
```

iii. The video offset is a whole-session constant shared by both cameras, so no per-camera treatment is needed; taking `frameTimes` from the same group the feature was read from means the two views can have different frame counts without breaking anything.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. A separate file next to the data structure, `motionEnergy_<anm>_<date>.mat`, loaded with `scipy.io.loadmat(..., simplify_cells=True)` and unwrapped via `me['data']` plus one optional further `['data']`. It holds one trace per trial with one value per side-camera frame. `obj.me` is never consulted. The side camera's aligned frame times (reused from the tongue call) provide the time base.

ii.
```python
def load_motion_energy(path: Path) -> list[np.ndarray]:
    mat = loadmat(path, simplify_cells=True)
    raw = mat["me"]["data"]
    if isinstance(raw, dict):
        raw = raw["data"]
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        return [np.asarray(item, dtype=np.float64).ravel() for item in raw.ravel()]
    if isinstance(raw, np.ndarray) and raw.ndim == 2:
        return [np.asarray(raw[:, i], dtype=np.float64) for i in range(raw.shape[1])]
    raise RuntimeError(f"Unsupported motion-energy layout in {path}")
```

iii. No prose justification; the single re-unwrap mirrors `loadMotionEnergy.m`'s `if isstruct(me.data), me.data = me.data.data; end`. Two array layouts (cell array and 2-D matrix) are handled, with an explicit error for anything else.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Almost none — the value is already one number per frame. The trace is truncated to `min(len(me), len(frame_times))`, linearly interpolated onto the 500-bin axis with NaN outside coverage, and then **nearest-filled** across the remaining NaNs, exactly as `loadMotionEnergy.m` does with `fillmissing(me.data, 'nearest')`. A trial whose side-camera tracking was skipped, or with fewer than two frames, or with no finite motion-energy values, becomes an all-NaN trace. The consequence of the nearest fill is that "no video" ends up covering only 0.03% of bins.

ii.
```python
                    me = interp_preserving_missing(tongue[2][:n], raw_me[:n], TIME)
                    motion_energy.append(nearest_fill(me))
```
```python
def nearest_fill(values: np.ndarray) -> np.ndarray:
    """One-dimensional equivalent of MATLAB fillmissing(..., 'nearest')."""
    ...
    choose_right = np.abs(right - bad) < np.abs(bad - left)
    nearest = np.where(choose_right, right, left)
    values[bad] = values[nearest]
    return values
```

iii. The docstring's claim that "camera and motion-energy streams use the repository's per-session video offset before interpolation onto the neural time axis", and the function's own docstring naming `fillmissing(..., 'nearest')` as the thing being reproduced. The paper's per-pixel median-difference and 99th-percentile reduction are already applied upstream in the stored file, so nothing is recomputed.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The same `categorize_session_signal`: session-wide 50th percentile of finite values, 0 below, 1 at or above, 2 where non-finite. `output_values[5] = ["below_session_median", "at_or_above_session_median", "no_video"]`. The authors' own manual per-session `moveThresh` in the motion-energy file is deliberately not used.

ii.
```python
    motion_cat, motion_threshold = categorize_session_signal(motion_energy)
```

iii. The prompt mandates the 50th-percentile split, which overrides the paper's bimodality-based manual threshold.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy has one value per side-camera frame, so it inherits the side camera's aligned frame times — literally the third element returned by the tongue's `trajectory_signal` call — and is interpolated onto the same 500-bin axis. If the side camera trial was skipped, motion energy for that trial is all NaN.

ii.
```python
            if tongue is None or trial >= len(motion_trials):
                motion_energy.append(np.full(N_TIME, np.nan))
            else:
                raw_me = motion_trials[int(trial)]
                # Motion energy and the side camera have one value per frame.
                n = min(raw_me.size, tongue[2].size)
                if n < 2 or not np.any(np.isfinite(raw_me[:n])):
                    motion_energy.append(np.full(N_TIME, np.nan))
                else:
                    me = interp_preserving_missing(tongue[2][:n], raw_me[:n], TIME)
                    motion_energy.append(nearest_fill(me))
```

iii. Inline comment: *"Motion energy and the side camera have one value per frame."* `loadMotionEnergy.m` likewise indexes `obj.traj{1}(trix).frameTimes`.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Six guards, all of which keep the trial and mark the gap rather than drop or invent data. (1) `NdroppedFrames` empty or non-finite → the whole trial's tracking is `None` → 500 bins of "not visible" for that feature (and all-NaN motion energy). (2) A `ts` array with the wrong rank, too few coordinate rows, or a missing feature index → same. (3) Frame times, x and y truncated to their common length, so mismatched counts do not crash. (4) Fewer than two finite source times → all-NaN output. (5) Untracked frames (DeepLabCut likelihood ≤ 0.9, stored as NaN x/y) → NaN through the interpolation → "not visible". (6) Motion energy NaNs are nearest-filled, following the source. Two hard failures are raised rather than worked around: a session with fewer than two usable trials, and a session with fewer than 10 curated units — either aborts the whole conversion instead of skipping the session.

ii.
```python
    dropped = referenced_array(h5, trajectory_group["NdroppedFrames"][trial, 0]).squeeze()
    if np.size(dropped) == 0 or not np.all(np.isfinite(dropped)):
        return None
    ...
    if raw.ndim != 3 or raw.shape[1] < 2 or feature_index >= raw.shape[0]:
        return None
    n = min(frame_time.size, x_raw.size, y_raw.size)
```
```python
    valid_time = np.isfinite(source_time)
    source_time, values = source_time[valid_time], values[valid_time]
    if source_time.size < 2:
        return np.full(target_time.shape, np.nan, dtype=np.float64)
```
```python
        if kept_trials.size < 2:
            raise RuntimeError(f"{animal} {date} has fewer than two usable trials")
        ...
        if len(selected_units) < 10:
            raise RuntimeError(f"{animal} {date} has only {len(selected_units)} curated units")
```

iii. The `NdroppedFrames` guard is copied from `findPosition.m`. The "not visible"/"no video" classes exist precisely so missing bins can be represented without NaN, which the output format forbids. The 10-unit session rule comes from the methods text ("Recording sessions were included for analysis only if they had at least 10 units").

## 11-a. What are the most time-consuming steps of the code?

i. HDF5 reading, specifically the per-trial DeepLabCut arrays. Profiling one session (JEB19_2023-04-20, 297 trials, 67 units, 2.72 s total) attributes 1.85 s (68%) to `trajectory_signal`, of which 1.63 s is `referenced_array` → `h5py.read_direct`: each call dereferences and reads the *entire* `ts` array for all features of that camera (e.g. 10 features × 3 × ~2,700 frames) in order to use a single feature's two rows, and it is called once per trial per feature. Spike-time reads are the next cost, then `smooth_rates` (0.20 s, of which most is the 484 `lfilter` calls used by `unit_mean_rate`). Everything else — histogramming, discretisation, assembly — is negligible. The whole 12-session conversion takes about 11 s, plus pickling a 345 MiB file.

ii.
```python
def referenced_array(h5: h5py.File, ref: h5py.Reference) -> np.ndarray:
    return np.asarray(h5[ref])
...
    raw = referenced_array(h5, trajectory_group["ts"][trial, 0])
    ...
    x_raw = raw[feature_index, 0, :]
    y_raw = raw[feature_index, 1, :]
```

iii. No performance discussion appears in the code or the trajectory; the AI never profiled. The cost is partly reducible — slicing the HDF5 dataset (`raw[feature_index, :2, :]`) instead of materialising it would avoid reading the unused features — but the data must be read once regardless.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Three. (1) The per-unit spike-binning loop in `process_session` calls `histogram_unit` once per unit, each call allocating a fresh `(n_trials, 500)` array and running a separate `np.add.at`; a single `np.histogramdd`/`histogram2d` over the concatenated (unit, trial, time) triples would do it in one pass — this is exactly the loop the reference solution vectorised. (2) The seven-condition loop inside `unit_mean_rate`, run for every surviving unit (≈480 `np.histogram` + `lfilter` calls per session), could be one `histogram2d` over condition × time. (3) The per-trial video loop is genuinely awkward to vectorise because each trial has a different frame count, but the two `np.interp` calls per feature could at least be replaced by one call on a stacked array.

ii.
```python
        for unit_index, (aligned_time, spike_trial, _, _) in enumerate(selected_units):
            mapped = original_to_kept[spike_trial]
            use = mapped >= 0
            unit_counts = histogram_unit(aligned_time[use], mapped[use], kept_trials.size)
            counts[:, unit_index, :] = unit_counts
```
```python
    for mask in condition_masks:
        n_trials = int(mask.sum())
        selected = mask[spike_trial] if spike_trial.size else np.zeros(0, bool)
        hist = np.histogram(aligned_time[selected], bins=edges)[0].astype(np.float32)
```

iii. Not discussed by the AI. In practice the penalty is small because I/O dominates, but the two spike loops are the clearest missed vectorisations.

## 11-c. What processing does the code repeat multiple times?

i. Three genuine repetitions. (1) **Spikes are binned and smoothed twice.** `unit_mean_rate` histograms and smooths every surviving unit's spikes at 10 ms to apply the >1 Hz criterion; the survivors' spikes are then histogrammed again by `histogram_unit` and smoothed again by `smooth_rates`. (2) **The per-trial `ts` dataset is re-read per feature**: the side camera's array is read for the tongue and the bottom camera's for the paw, each time materialising all features and all three coordinate rows. (3) `categorize_session_signal` computes `np.isfinite` over the session twice — once to pool the finite values for the percentile, once per trial to place the "not visible" class. Things the code correctly does *not* repeat: the video offset is computed once per session and reused for both cameras and all trials; `find_feature_indices` is called once per feature per session; the motion-energy file is loaded once; the side camera's aligned frame times are computed once and shared between the tongue and motion energy.

ii.
```python
            mean_rate = unit_mean_rate(aligned_time, spike_trial, condition_masks)
            if mean_rate > 1.0:
                selected_units.append((aligned_time, spike_trial, quality, mean_rate))
        ...
        for unit_index, (aligned_time, spike_trial, _, _) in enumerate(selected_units):
            ...
            unit_counts = histogram_unit(aligned_time[use], mapped[use], kept_trials.size)
        rates = smooth_rates(counts)
```
```python
        offset = video_offset(h5)          # once per session
        ...
                    me = interp_preserving_missing(tongue[2][:n], raw_me[:n], TIME)
```

iii. The double binning is the direct price of reproducing `removeLowFRClusters.m` faithfully — the criterion is defined on condition-averaged PSTHs, not on the raw rate — so it is deliberate rather than accidental, but it was not flagged or reused.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Four things. (1) `referenced_array` materialises every DeepLabCut feature of a camera (up to 10 features × 3 × ~2,700 frames) on every trial, and all but one feature's x and y rows are thrown away — the single largest wasted cost in the script. (2) `unit_mean_rate` builds seven full 500-bin smoothed PSTHs per unit and reduces each to one scalar; the PSTHs are discarded. Its `condition_masks` list is also partly redundant: masks 6 and 7 (`hit & ~autowater & ~early`, `hit & autowater & ~early`) are near-duplicates of masks 2 and 3, and mask 1 (`hit|miss|ignore`) is all trials. (3) Per-unit `quality` strings and `mean_rates_hz` are collected through the whole session and then popped off before the result is saved, so they never reach the pickle. (4) `input_trials.append(time_input.copy())` stores 3,116 byte-identical copies of the same 500-element vector instead of sharing one array. Smaller items: `trial_mask`'s `np.isfinite(go)` and `(hit|miss|ignore)` terms are no-ops on this cohort, `ignore` is read from `bp.no` although it is the complement of `hit|miss`, and `histogram_unit` allocates a temporary per-unit array that is immediately copied into `counts`.

ii.
```python
    raw = referenced_array(h5, trajectory_group["ts"][trial, 0])   # reads all features
```
```python
        result.pop("qualities")
        result.pop("mean_rates_hz")
```
```python
        input_trials.append(time_input.copy())
```
```python
        trial_mask = ~early & ~stim & np.isfinite(go) & (hit | miss | ignore)
```

iii. Not discussed by the AI. None of it changes the output; the wasted work is bounded by the ~11 s total runtime, and the redundant input copies add roughly 6 MB to a 345 MiB file.
