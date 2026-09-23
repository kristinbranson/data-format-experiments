# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. One MATLAB file per session, `data_structure_<anm>_<date>.mat`, plus a companion `motionEnergy_<anm>_<date>.mat` in the same folder. The two ephys folders (`Ephys_Behavior`, `RandomizedDelay_Ephys_Behavior`) are globbed for `data_structure_*.mat`, but the glob is only used to *resolve paths*: the set of sessions actually converted is the hard-coded `ALM_PROBES` (25 fixed-delay) and `RANDOMIZED_DELAY_PROBES` (19 randomized-delay) dictionaries, transcribed from the authors' `DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m` files, which also fix the probe(s) used per session. If any listed session is absent from disk the converter raises rather than silently shrinking the dataset. Two readers are used because the shared data mixes MATLAB formats: `h5py` for v7.3 files and `scipy.io.loadmat(squeeze_me=True, struct_as_record=False)` for v5/v7 files, selected by `h5py.is_hdf5`. Motion energy is always read with `scipy.io.loadmat`. The result is 44 sessions, 13,762 trials, 2,457 units.

ii.
```python
ALM_PROBES = {
    "EKH1_2021-08-07": (2,),
    ...
    "JGR3_2021-11-18": (1,),
}
RANDOMIZED_DELAY_PROBES = {
    "JEB11_2022-05-10": (1,),
    ...
    "JEB24_2023-11-03": (1,),
}
SESSION_GROUPS = (
    ("Ephys_Behavior", "fixed_delay", ALM_PROBES),
    ("RandomizedDelay_Ephys_Behavior", "randomized_delay", RANDOMIZED_DELAY_PROBES),
)
```

```python
for directory_name, task_variant, probe_map in SESSION_GROUPS:
    directory = data_root / directory_name
    available = {_session_key(path): path for path in directory.glob("data_structure_*.mat")}
    missing = sorted(set(probe_map) - set(available))
    if missing:
        raise ValueError(f"Missing selected sessions in {directory}: {missing}")
    for key in sorted(probe_map):
        sessions.append((available[key], directory / f"motionEnergy_{key}.mat",
                         probe_map[key], task_variant))
```

```python
if h5py.is_hdf5(data_path):
    with h5py.File(data_path, "r") as matfile:
        ...
        neural_all, neural_info = _load_neural(matfile, probes)
else:
    obj = loadmat(data_path, squeeze_me=True, struct_as_record=False,
                  variable_names=["obj"])["obj"]
    ...
    neural_all, neural_info = _load_neural_v5(obj, probes)
```

iii. From the trajectory (step 8, 12, 26): "The source data separates the 25 electrophysiology sessions from the optogenetic-only datasets, matching the paper's reported ALM cohort"; "probe selection is session-specific (including both probes for three JEB15 sessions)"; "A broader inventory found a second neural cohort that is relevant: the paper's 19 randomized-delay ALM sessions. The repository explicitly selects 19 of the 22 files (three are commented out and lack matching motion-energy data) … the optogenetic directories contain behavior only and remain out of scope because the decoder requires neural activity." The v5 reader was added at step 51 after discovering that "a later randomized session uses MATLAB v7 (not HDF5/v7.3) … I'm adding a small compatibility path for those legacy files rather than dropping published sessions."

## 1-b. How are the data split into subjects?

i. The subject is the filename stem before the first underscore (`JEB19_2023-04-19` → `JEB19`). Each session records that string; `subjects` is the list of unique subjects in order of first appearance (not sorted), and `subject_idx` is each session's index into it. 14 subjects across the 44 sessions.

ii.
```python
key = _session_key(data_path)
subject, date = key.split("_", 1)
...
session_subjects.append(subject)
```
```python
subjects = list(dict.fromkeys(session_subjects))
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
...
"subjects": subjects,
"subject_idx": np.asarray([subject_lookup[x] for x in session_subjects], dtype=np.int64),
```

iii. Not discussed explicitly in the trajectory; the filename convention is the same one the authors' `load<ANM>_ALMVideo.m` scripts use to identify animals (`meta.anm`/`meta.date`), and the AI's `_session_key` regex enforces it.

## 1-c. How are the data split into sessions?

i. One `data_structure_*.mat` file = one session = one element of `neural`/`input`/`output`/`brain_region_idx`. Fixed-delay and randomized-delay sessions are concatenated into one flat session list (fixed-delay first, each group alphabetically sorted), with the task variant recorded per session in `metadata.session_info['task_variant']` rather than split into two datasets. 25 fixed-delay + 19 randomized-delay = 44.

ii.
```python
for key in sorted(probe_map):
    sessions.append((available[key], directory / f"motionEnergy_{key}.mat",
                     probe_map[key], task_variant))
...
session_info.append({
    "session_id": key, "subject": subject, "date": date,
    "task_variant": task_variant,
    "source_file": data_path.name, "source_directory": data_path.parent.name,
    "alm_probes": list(probes), ...})
```

iii. Step 26: the randomized-delay cohort is "the paper's 19 randomized-delay ALM sessions", included "alongside the 25 fixed-delay sessions". Both cohorts are go-cue aligned DR/WC sessions with ALM recordings, so they enter the same decoder dataset; the variant is kept as metadata for auditability.

## 1-d. How are the data split into trials?

i. The trial is the Bpod trial index. Every per-trial field of `obj.bp` (`hit`, `miss`, `no`, `R`, `autowater`, `early`, `stim.enable`, `ev.goCue`) is read as a flat vector and indexed by trial number; `obj.bp.Ntrials` gives the count used to allocate the spike-count grid. Spikes carry their own 1-based `clu.trial` label, so no trial boundaries are reconstructed; camera tracking (`obj.traj{view}`) and motion energy are per-trial cell arrays indexed by the same number. The motion-energy file's trial count is asserted equal to `Ntrials`. No re-binning of trials occurs — the arrays converted are `selected`, the retained subset of trial indices.

ii.
```python
ntrials = int(_vector(matfile["obj/bp/Ntrials"])[0])
...
spike_trials = _deref_vector(matfile, cluster_group["trial"], cluster_idx).astype(np.int64) - 1
go_cue = _vector(matfile["obj/bp/ev/goCue"]).astype(np.float64)
valid_trial = (spike_trials >= 0) & (spike_trials < ntrials)
```
```python
def _motion_trials(motion_path: Path, ntrials: int) -> np.ndarray:
    ...
    trials = np.asarray(motion_data, dtype=object).reshape(-1)
    if trials.size != ntrials:
        raise ValueError(f"Motion-energy trials ({trials.size}) != Bpod trials ({ntrials})")
    return trials
```
```python
for out_trial, source_trial in enumerate(selected):
    if not have_video[source_trial]:
        continue
```

iii. The Bpod table defines trials directly and there is one `goCue` per trial, so no inference is needed. The AI's only explicit comment on trial bookkeeping is `_fit_trial_mask`, added at step 87/93 after finding that one randomized session stores 319 `trials.bp.haveEphys` entries for 318 Bpod trials: "The trial mapping confirms the extra entry is trailing (`BPnum` and file numbers run 1…319 while the retained Bpod object ends at trial 318), so the compatibility rule trims only trailing bookkeeping excess and pads missing coverage as false."

## 1-e. How are trials filtered based on quality controls?

i. Four conditions, combined in `convert`. (1) early-lick trials (`bp.early`) are dropped; (2) photostimulation trials (`bp.stim.enable`) are dropped; (3) trials without ephys coverage (`obj.trials.bp.haveEphys`, length-corrected by `_fit_trial_mask`) are dropped — in practice this removes 0 trials; (4) any remaining trial in which **no retained unit fires a single spike anywhere in the 5 s window** is dropped, which removes 61 trials in two sessions where behaviour continued after the probe stopped. Ignore trials are deliberately *kept*, because "ignore" is a requested outcome class and "none" a requested lick-direction class. Sessions with fewer than two candidate trials would raise. Final count: 13,762 trials, and the per-session trial counts are identical to the human reference's for all 44 sessions.

ii.
```python
early = _vector(matfile["obj/bp/early"]).astype(bool)
stimulated = _vector(matfile["obj/bp/stim/enable"]).astype(bool)
have_ephys = _fit_trial_mask(_vector(matfile["obj/trials/bp/haveEphys"]), early.size)
candidate_trials = np.flatnonzero(~early & ~stimulated & have_ephys)
if candidate_trials.size < 2:
    raise ValueError(f"{key}: fewer than two usable trials")
neural_all, neural_info = _load_neural(matfile, probes)
neural_present = np.any(neural_all != 0, axis=(1, 2))
selected = candidate_trials[neural_present[candidate_trials]]
```
with per-session counts recorded:
```python
"n_early_lick_excluded": int(early.sum()),
"n_photostim_excluded": int((stimulated & ~early).sum()),
"n_no_ephys_excluded": int((~have_ephys & ~early & ~stimulated).sum()),
"n_all_zero_neural_excluded": int(candidate_trials.size - selected.size),
```

iii. Docstring: "Early-lick and photostimulation trials are removed; ignore trials are retained because ignore is a requested decoder outcome." Step 76: "two recordings continued behavioral trials after electrophysiology stopped, producing all-zero neural trials. The source exposes `haveEphys` specifically for this boundary." Step 104, after the `haveEphys` fix failed: "The source `haveEphys` flag itself is incorrectly true for those tails … The reliable boundary is the neural data: after curation, those trials contain no spikes in any retained unit across the full 5 s window. I'm switching the final inclusion test to that direct evidence … a genuine silent trial across 17–45 active units for five seconds is effectively impossible." Step 110 notes the movement medians are recomputed after this cut "so the movement thresholds remain internally consistent with the retained trials".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}`, the spike-sorted clusters of the probe(s) listed for that session, using three fields per cluster: `trial` (1-based trial of each spike), `trialtm` (spike time relative to that trial's start, behaviour clock), and `quality` (manual curation label, a MATLAB char array). `obj.bp.ev.goCue` supplies the alignment times and `obj.bp.Ntrials` the grid size. Clusters from both probes of a two-probe session are concatenated into one population.

ii.
```python
for probe in probes:
    cluster_group = matfile[np.asarray(matfile["obj/clu"])[probe - 1, 0]]
    for cluster_idx in range(cluster_group["trial"].shape[0]):
        quality_obj = matfile[np.asarray(cluster_group["quality"])[cluster_idx, 0]]
        ...
        spike_trials = _deref_vector(matfile, cluster_group["trial"], cluster_idx).astype(np.int64) - 1
        spike_times = _deref_vector(matfile, cluster_group["trialtm"], cluster_idx).astype(np.float64)
        go_cue = _vector(matfile["obj/bp/ev/goCue"]).astype(np.float64)
```

iii. Step 12: "The repository confirms the core neural preprocessing: spikes aligned to each trial's `goCue`, binned at 5 ms from −2.5 to +2.5 s, converted to Hz, then Gaussian-smoothed with the paper's 15-bin kernel." These are the same fields `alignSpikes.m`/`getSeq.m` read.

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed into the 1000 × 5 ms bins with `np.floor((t − TMIN)/DT)` (the half-open `histc` assignment), accumulated per trial with `np.add.at`, then smoothed and converted to Hz. The smoother is a deliberate reimplementation of the repository's `mySmooth(x, 15, 'reflect')`: a `gausswin(15)`-shaped kernel whose **first floor(15/2)=7 weights are zeroed to make it causal**, renormalised to sum 1; the boundary handling prepends the *first* 15 samples (which is what `mySmooth.m` actually does, despite calling itself "reflect") and then trims them off. The smoothed counts are divided by `DT` to give spikes/s. No normalisation, baseline subtraction, or z-scoring. Stored as `float32`.

ii.
```python
def _gaussian_kernel(length: int) -> np.ndarray:
    # MATLAB gausswin(length) uses alpha=2.5. This is its defining formula.
    n = np.arange(length, dtype=np.float64) - (length - 1.0) / 2.0
    kernel = np.exp(-0.5 * (2.5 * n / (length / 2.0)) ** 2)
    # mySmooth.m zeros the first floor(N/2) weights to make the filter causal.
    kernel[: length // 2] = 0.0
    kernel /= kernel.sum()
    return kernel.astype(np.float32)

def _smooth_counts(counts: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of mySmooth(counts, 15, 'reflect')."""
    # Despite its name, repository code prepends the first N samples rather
    # than reversing them. Preserve that exact behavior.
    padded = np.concatenate((counts[:, :SMOOTH_BINS], counts), axis=1)
    smoothed = convolve(padded, SMOOTH_KERNEL[None, :], mode="same", method="direct")
    return smoothed[:, SMOOTH_BINS:] / np.float32(DT)
```
```python
time_idx = np.floor((aligned - TMIN) / DT).astype(np.int64)
counts = np.zeros((ntrials, TIME.size), dtype=np.float32)
np.add.at(counts, (spike_trials, time_idx), 1.0)
units.append(_smooth_counts(counts))
...
neural = np.stack(units, axis=1).astype(np.float32, copy=False)
```

iii. Step 12: the repository's pipeline is "binned at 5 ms … converted to Hz, then Gaussian-smoothed with the paper's 15-bin kernel." The code comments record the two non-obvious details of `mySmooth.m` (the causal zeroing of the first half of the kernel, and the fact that `'reflect'` prepends rather than reflects), and the metadata field `neural_measurement` reads "single-trial firing rate (spikes/s), causal Gaussian smoothed" with `neural_smoothing_bins: 15`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. First the manual curation label: `clu.quality` is decoded from chars, stripped, lower-cased, and rejected if it is one of `garbage`, `gabrga`, `noisy`, `real?` — exactly the drop list in `findClusters.m` for `params.quality = {'all'}`. Multi-unit, `poor`, `fair`, `good`, `great`, `excellent` and unlabelled clusters are all kept. Second, units whose mean firing rate over the whole −2.5…2.5 s window, averaged over *all* `Ntrials` (not only the retained trials), is not above 1 Hz are dropped. This leaves 2,513 curated → 2,457 retained units (17–141 per session). The per-session label tallies are stored in metadata.

ii.
```python
REJECTED_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}
...
quality_counts[quality or "unlabelled"] = quality_counts.get(quality or "unlabelled", 0) + 1
if quality.lower() in REJECTED_QUALITIES:
    continue
...
mean_fr = neural.mean(axis=(0, 2), dtype=np.float64)
keep = mean_fr > LOW_FR_HZ            # LOW_FR_HZ = 1.0
return neural[:, keep, :], {
    "n_curated_before_fr_filter": int(len(units)),
    "n_units_after_fr_filter": int(keep.sum()),
    "mean_fr_hz_min_retained": float(mean_fr[keep].min()) if keep.any() else None,
    "quality_counts_before_curation": quality_counts,
}
```

iii. Docstring: "manual cluster curation, and a 1 Hz firing-rate cutoff"; metadata: "Repository-curated units excluding garbage/noisy/real? labels, with mean go-cue-window firing rate > 1 Hz; published ALM probe selections." Step 12: "The published inclusion rule is all curated units above 1 Hz." The lower-casing is the AI's own addition, needed because the labels are free text written in both cases (`Noisy` appears once, which `findClusters.m`'s case-sensitive `ismember` would have let through).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. One subtraction. `clu.trialtm` is already on the behaviour clock and relative to its own trial's start, and `bp.ev.goCue` is on the same clock, so `trialtm − goCue[trial]` gives seconds from go cue with no interpolation or offset. Spikes outside the trial range or outside [−2.5, 2.5) are discarded before binning. This is `alignSpikes.m` with `params.alignEvent = 'goCue'`. (The camera streams need an additional clock correction; the spikes do not.)

ii.
```python
valid_trial = (spike_trials >= 0) & (spike_trials < ntrials)
spike_trials = spike_trials[valid_trial]
aligned = spike_times[valid_trial] - go_cue[spike_trials]
valid_time = (aligned >= TMIN) & (aligned < TMAX)
spike_trials = spike_trials[valid_time]
aligned = aligned[valid_time]
```

iii. Step 12: "spikes aligned to each trial's `goCue`". Metadata: `temporal_alignment_event = "auditory go cue onset (water delivery onset in WC trials)"`, which reflects that in the water-cued context there is no auditory cue and `bp.ev.goCue` marks water delivery.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 5 ms (`DT = 1/200`), 1000 non-overlapping bins spanning −2.5 to +2.5 s from the go cue, identical for every trial and session. These are `params.dt`, `params.tmin`, `params.tmax` from `getDefaultParams.m`. Spikes go straight into this grid (no finer intermediate binning, no rebinning afterwards); the video streams are interpolated onto the *centres* of the same bins, so all streams share one time axis. `metadata.time_bin_size = 5.0` ms, `off_start = −2.5`, `off_end = 2.5`.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 200.0
...
TIME = (np.arange(int(round((TMAX - TMIN) / DT)), dtype=np.float32) * DT
        + TMIN + DT / 2.0)
EDGES = TMIN + np.arange(TIME.size + 1, dtype=np.float64) * DT
```
```python
"time_bin_size": DT * 1000.0,
"off_start": TMIN,
"off_end": TMAX,
```

iii. Step 12: "binned at 5 ms from −2.5 to +2.5 s". These are the repository defaults verified at step 13/15 by grepping for `params.dt`, `tmin`, `tmax` across the analysis scripts.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. No raw variable. It is the analysis window we define: the centre of each of the 1000 bins of the go-cue-aligned grid, i.e. −2.4975 … +2.4975 s. Because every trial is aligned to its own `bp.ev.goCue`, this vector is by construction "time from go cue onset" for every trial, and is identical across trials and sessions.

ii.
```python
TIME = (np.arange(int(round((TMAX - TMIN) / DT)), dtype=np.float32) * DT
        + TMIN + DT / 2.0)
...
"input_names": ["time_from_go_cue_s"],
```

iii. Implicit in step 12's framing of the window; the code comment at the assembly site reads "Time from the go cue is a continuous, time-varying decoder input."

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None. The bin-centre vector is computed once at module level and the same `float32` row of shape `(1, 1000)` is copied into every trial of every session.

ii.
```python
# Time from the go cue is a continuous, time-varying decoder input.
input_trials = [TIME[None, :].copy() for _ in range(selected.size)]
```

iii. N/A — the axis is defined by the conversion, not derived from data.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural time axis. Spikes are placed into `EDGES = TMIN + k·DT` by `floor((t − TMIN)/DT)`, and `TIME` holds the centres of exactly those bins, so element *k* of the input is the same 5 ms interval as column *k* of the neural matrix. The video streams are interpolated onto `TIME` as well, so all three streams are index-aligned.

ii.
```python
time_idx = np.floor((aligned - TMIN) / DT).astype(np.int64)
```
```python
TIME = (np.arange(int(round((TMAX - TMIN) / DT)), dtype=np.float32) * DT
        + TMIN + DT / 2.0)
```
```python
x = _interp(source_time, x_raw, aligned_time)   # aligned_time is TIME
```

iii. N/A.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial flags of `obj.bp`: `hit`, `miss`, `no`, and `R` (the instructed/rewarded side). The direction the animal actually licked is not recorded, so it is inferred from the instructed side combined with the outcome.

ii.
```python
hit = _vector(matfile["obj/bp/hit"]).astype(bool)
miss = _vector(matfile["obj/bp/miss"]).astype(bool)
ignore = _vector(matfile["obj/bp/no"]).astype(bool)
right_target = _vector(matfile["obj/bp/R"]).astype(bool)
```

iii. Step 20: "lick direction is the actual choice (target side on correct trials, opposite side on incorrect trials, and `none` on ignores)".

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A three-way relabelling, evaluated per trial and broadcast across all 1000 bins (the value is constant within a trial): ignore → `none` (2); hit → the instructed side (right=1 if `R`, else left=0); miss → the opposite side. A trial that is none of hit/miss/ignore raises. Codes and names follow the prompt's order: `["left", "right", "none"]`.

ii.
```python
if ignore[source_trial]:
    lick_direction, outcome = 2, 2  # none, ignore
elif hit[source_trial]:
    lick_direction = 1 if right_target[source_trial] else 0
    outcome = 1  # correct
elif miss[source_trial]:
    lick_direction = 0 if right_target[source_trial] else 1
    outcome = 0  # incorrect
else:
    raise ValueError(f"Trial {source_trial + 1} has no hit/miss/ignore outcome")
...
output[out_trial, 0:3, :] = np.asarray(
    (lick_direction, context, outcome), dtype=np.int8)[:, None]
```

iii. Step 20: "I've resolved the label semantics as well: lick direction is the actual choice (target side on correct trials, opposite side on incorrect trials, and `none` on ignores)." Ignore trials are retained specifically so the `none` class exists.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial flag, `obj.bp.autowater`, which marks the water-cued (WC) block where water is delivered at a random port with no sample tone, delay, or go cue.

ii.
```python
water_cued = _vector(matfile["obj/bp/autowater"]).astype(bool)
```

iii. Step 20: "context comes from the repository's `autowater` flag." This is the same flag the authors' `params.condition` strings use to separate the two contexts.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling of the flag, broadcast across all 1000 bins: autowater → WC (0), otherwise DR (1). Names `["WC", "DR"]`, following the prompt's order.

ii.
```python
context = 0 if water_cued[source_trial] else 1  # WC, DR
output[out_trial, 0:3, :] = np.asarray(
    (lick_direction, context, outcome), dtype=np.int8)[:, None]
```

iii. Codes follow the prompt's "WC, DR" ordering; no further processing is possible or needed for a boolean block flag.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The three mutually-exclusive per-trial flags of `obj.bp`: `hit`, `miss`, and `no`. Unlike the human reference, the AI reads `bp.no` explicitly rather than treating "neither hit nor miss" as ignore, and raises if a trial matches none of the three.

ii.
```python
hit = _vector(matfile["obj/bp/hit"]).astype(bool)
miss = _vector(matfile["obj/bp/miss"]).astype(bool)
ignore = _vector(matfile["obj/bp/no"]).astype(bool)
```

iii. The same flags already determine lick direction, so outcome is derived in the same branch. Reading `bp.no` explicitly makes the "no outcome flag set" case an error instead of a silent ignore.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Relabelling into three classes broadcast across all bins: miss → incorrect (0), hit → correct (1), `no` → ignore (2). Names `["incorrect", "correct", "ignore"]`, matching the prompt's order. Ignore trials are kept rather than dropped (the paper excludes them from its own analyses).

ii.
```python
if ignore[source_trial]:
    lick_direction, outcome = 2, 2  # none, ignore
elif hit[source_trial]:
    ...
    outcome = 1  # correct
elif miss[source_trial]:
    ...
    outcome = 0  # incorrect
```

iii. Docstring: "ignore trials are retained because ignore is a requested decoder outcome." Metadata: "All hit, miss, and ignore trials except early-lick and photostimulation trials … ignore trials are retained to satisfy the requested outcome/none classes."

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj{1}` — the **side camera only** — feature `tongue`. From that per-trial struct it reads `ts` (x, y, likelihood per frame), `frameTimes`, `featNames`, and `NdroppedFrames`. The bottom-camera tongue features (`top_tongue` etc.) are not used. Alignment additionally needs `obj.sglx.fs`, `obj.sglx.bitcode.bitstart`, `obj.bp.ev.bitStart`, `obj.bp.ev.goCue`, and the per-trial `obj.trials.bp.haveVid` gate.

ii.
```python
tx, ty, tongue_vis = _trajectory_xy(
    matfile, trajectory_side, source_trial, "tongue", TIME,
    video_shift, go_cue[source_trial])
tongue_speed[out_trial] = _speed(tx, ty, tongue=True)
tongue_visible[out_trial] = tongue_vis
```
```python
feature_idx = _feature_index(matfile, trajectory_group, trial, feature)
# HDF5 dimensions are reversed from MATLAB: feature x coordinate x frame.
x_raw = ts[feature_idx, 0, :]
y_raw = ts[feature_idx, 1, :]
```

iii. Step 12: "I'm now resolving video visibility and timing fields so the three movement labels use the same interpolation conventions as the authors." `tongue` is the canonical side-view feature in the authors' `params.traj_features{1}`, and their `findPosition.m`/`findVelocity.m` operate one feature at a time; the AI took the single side-view feature rather than combining views.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. A direct reimplementation of `findPosition.m` + `findVelocity.m`'s tongue branch. (1) If `NdroppedFrames` is NaN/empty, or `frameTimes` is empty, or `haveVid` is false, the whole trial is left NaN. (2) Otherwise raw x and y are **linearly interpolated** from frame times onto the 1000 bin centres, with NaN outside the frame range (MATLAB `interp1` semantics). No positional smoothing is applied — matching the authors, whose `mySmooth(ts, 1, 'reflect')` is a no-op for N=1 and which they skip for the tongue anyway. (3) Visibility is taken as `isfinite(x) & isfinite(y)` *on the interpolated trace*; this is equivalent to the likelihood > 0.9 rule because the authors already store x and y as NaN wherever DLC likelihood ≤ 0.9 (verified: on one trial, 1711/1792 side-tongue frames are NaN and exactly the 81 frames with likelihood > 0.9 are finite). (4) Speed is `hypot(gradient(x), gradient(y))` on the bin grid, with non-finite derivatives set to 0 — the authors' "set tongue velocity to 0 if not visible". No baseline subtraction for the tongue and no cross-camera normalisation (only one view is used). Units are pixels per 5 ms bin, which is immaterial because the class boundary is a percentile.

ii.
```python
def _speed(x, y, tongue: bool) -> np.ndarray:
    """Match findVelocity.m (including its shared x/y baseline subtraction)."""
    xy = np.column_stack((x, y))
    differences = np.diff(xy, axis=0)
    baseline = np.asarray([...])
    if tongue:
        xvel = np.gradient(x)
        yvel = np.gradient(y)
        xvel[~np.isfinite(xvel)] = 0.0
        yvel[~np.isfinite(yvel)] = 0.0
    else:
        ...
    return np.hypot(xvel, yvel)
```
```python
def _interp(source_time, values, target_time):
    """Linear interpolation with MATLAB interp1-style NaNs out of bounds."""
    if source_time.size < 2 or values.size != source_time.size:
        return np.full(target_time.shape, np.nan, dtype=np.float64)
    order = np.argsort(source_time)
    return np.interp(target_time, source_time[order], values[order],
                     left=np.nan, right=np.nan)
```

iii. Step 20: "Visibility classes will be derived before the authors' nearest-value filling, so 'not visible/no video' remains meaningful." The `_speed` docstring names `findVelocity.m` as the target and flags its quirks. The AI chose to follow the authors' functions literally rather than re-derive a velocity estimate.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per session. All bins that are both visible and finite, pooled across the session's retained trials, define a 50th percentile; bins at or above it get 1, visible bins below it get 0, and every non-visible bin gets 2 (`not_visible`). The array is pre-filled with 2 and only the valid mask is overwritten, so any trial with no video at all comes out as 1000 `not_visible` bins. Threshold values are written into `metadata.session_info['tongue_velocity_median']`. In the delivered dataset 90.9% of bins are `not_visible`, 4.3% below and 4.9% at/above.

ii.
```python
tongue_values = tongue_speed[tongue_visible & np.isfinite(tongue_speed)]
...
thresholds = {"tongue_velocity_median": float(np.percentile(tongue_values, 50)), ...}
...
output[out_trial, 3:6, :] = 2
valid = tongue_visible[out_trial] & np.isfinite(tongue_speed[out_trial])
output[out_trial, 3, valid] = (
    tongue_speed[out_trial, valid] >= thresholds["tongue_velocity_median"])
```
```python
"output_values": [..., ["below_session_median", "at_or_above_session_median", "not_visible"], ...]
```

iii. Metadata: "Per-session 50th percentile over visible retained samples; class 1 is >= median." This is the prompt's rule (`<` 50th pct → 0, `>=` → 1, not visible → 2), and step 110 records that the medians are recomputed after the zero-neural trials are dropped so the thresholds stay consistent with the retained trials.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The video clock leads the behaviour clock, so `frameTimes` is corrected by a session-constant offset before being aligned. The offset is `mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)`, i.e. the repository's `findVideoOffset.m` verbatim, computed once per session. Frame time from the go cue is then `frameTimes − video_shift − goCue[trial]`, and x/y are interpolated from that onto the same 1000 bin centres the spikes were binned into, so the two streams are index-aligned. `params.advance_movement = 0` in the repository, so no extra lead is applied.

ii.
```python
fs = float(_vector(matfile["obj/sglx/fs"])[0])
video_shift = (_mode(matfile["obj/sglx/bitcode/bitstart"]) / fs
               - _mode(matfile["obj/bp/ev/bitStart"]))
```
```python
source_time = frame_times - video_shift - go_cue
x = _interp(source_time, x_raw, aligned_time)
y = _interp(source_time, y_raw, aligned_time)
```

iii. Step 19 verified the offset numerically against `fs`, `bitStart`, and `goCue` on the first three sessions before it was used. The construction mirrors `findVideoOffset.m` (`vidshift = mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)`) and `findPosition.m`'s `interp1(frameTimes − vidshift − alignEv(trix), ts, taxis)`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. `obj.traj{2}`, the bottom camera, using **both** tracked paws, `top_paw` and `bottom_paw` — the two features the authors list in `params.traj_features{2}`. Same per-trial fields as the tongue (`ts`, `frameTimes`, `featNames`, `NdroppedFrames`), same clock fields for alignment.

ii.
```python
PAW features, iterated:
for feature in ("top_paw", "bottom_paw"):
    px, py, paw_vis = _trajectory_xy(
        matfile, trajectory_bottom, source_trial, feature, TIME,
        video_shift, go_cue[source_trial])
    paw_speeds.append(_speed(px, py, tongue=False))
    paw_masks.append(paw_vis)
```

iii. Code comment: "Both paws were tracked in the bottom view. Average their speed where available; this gives one requested paw-velocity stream without privileging either paw." Metadata: "Mean Euclidean image-plane speed of the two bottom-view paw keypoints when visible."

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same interpolation onto the bin centres as the tongue, then the **non-tongue** branch of `findVelocity.m`: x and y are nearest-filled across gaps, differentiated with `np.gradient`, and the median of the frame-to-frame x differences (`basederiv(1)`) is subtracted from *both* the x and the y derivative — reproducing the authors' apparent copy-paste in `findVelocity.m`, as the code comment records. Speed is the magnitude of the two components. The two paws' speeds are then averaged **only over the bins where that paw is genuinely visible** (visibility is captured before the nearest-fill, so filled-in gap values never enter the average); a bin is `not_visible` only if neither paw was tracked. No normalisation, since both features come from the same camera.

ii.
```python
else:
    x_filled = _nearest_fill(x)
    y_filled = _nearest_fill(y)
    xvel = np.gradient(x_filled) - baseline[0]
    # The repository subtracts basederiv(1) from both components.
    yvel = np.gradient(y_filled) - baseline[0]
return np.hypot(xvel, yvel)
```
```python
paw_stack = np.stack(paw_speeds)
mask_stack = np.stack(paw_masks)
count = mask_stack.sum(axis=0)
summed = np.where(mask_stack, paw_stack, 0.0).sum(axis=0)
paw_speed[out_trial] = np.divide(summed, count, out=np.full(TIME.shape, np.nan), where=count > 0)
paw_visible[out_trial] = count > 0
```

iii. Step 20: "Visibility classes will be derived before the authors' nearest-value filling, so 'not visible/no video' remains meaningful" — i.e. the AI deliberately kept the authors' fill for the velocity computation while refusing to let it manufacture a visible class. The `_speed` docstring names `findVelocity.m` "(including its shared x/y baseline subtraction)".

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Identically to the tongue: a per-session 50th percentile over all visible, finite paw-speed samples pooled across retained trials; `>=` → 1, `<` → 0, no paw visible → 2. The threshold is recorded per session. In the delivered dataset the split is 47.1% / 47.1% / 5.8% `not_visible` — the low not-visible fraction is a consequence of using both paws and of `top_paw` being tracked in essentially every frame.

ii.
```python
paw_values = paw_speed[paw_visible & np.isfinite(paw_speed)]
...
"paw_velocity_median": float(np.percentile(paw_values, 50)),
...
valid = paw_visible[out_trial] & np.isfinite(paw_speed[out_trial])
output[out_trial, 4, valid] = (
    paw_speed[out_trial, valid] >= thresholds["paw_velocity_median"])
```

iii. Same rule and same justification as 7-c; the prompt specifies the 50th-percentile split and the `not visible` class.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Exactly as the tongue, but with the bottom camera's own `frameTimes`: `frameTimes − video_shift − goCue[trial]`, then linear interpolation onto the same 1000 bin centres. The offset is the same session constant, since it is a property of the recording, not of the camera.

ii.
```python
px, py, paw_vis = _trajectory_xy(
    matfile, trajectory_bottom, source_trial, feature, TIME,
    video_shift, go_cue[source_trial])
```
```python
source_time = frame_times - video_shift - go_cue
x = _interp(source_time, x_raw, aligned_time)
```

iii. Same as 7-d; the paw needs no separate treatment because every stream is interpolated onto one shared grid, and each feature is timed by the `frameTimes` of the camera that recorded it.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file next to each data structure, variable `me`, field `me.data` — a cell array with one trace per trial at camera frame rate. `obj.me` (present in only some sessions) is not used. The side camera's `frameTimes` provides the time base, since motion energy has one value per side-camera frame.

ii.
```python
def _motion_trials(motion_path: Path, ntrials: int) -> np.ndarray:
    me_obj = loadmat(motion_path, squeeze_me=True, struct_as_record=False,
                     variable_names=["me"])["me"]
    motion_data = me_obj.data
    # Some released files wrap the actual data in a second ``me`` struct.
    # This is the same ``if isstruct(me.data); me.data=me.data.data`` case
    # handled by DataLoadingScripts/loadMotionEnergy.m.
    if hasattr(motion_data, "data"):
        motion_data = motion_data.data
    trials = np.asarray(motion_data, dtype=object).reshape(-1)
    if trials.size != ntrials:
        raise ValueError(f"Motion-energy trials ({trials.size}) != Bpod trials ({ntrials})")
    return trials
```

iii. Step 38: "One source-format variant appeared in JEB15: its motion-energy field has an extra MATLAB struct wrapper, which the authors' loader explicitly unwraps. I've stopped before writing the final pickle and am adding the same unwrapping rule." Step 47: "The wrapper case now matches `loadMotionEnergy.m`." The trial-count assertion is the AI's own guard against a silent misalignment.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond resampling — the value is already one scalar per frame (the paper reduces each frame's per-pixel motion energy to its 99th percentile). The per-trial trace is linearly interpolated from side-camera frame times onto the 1000 bin centres, and the resulting edge NaNs are then **nearest-filled**, following `loadMotionEnergy.m`'s `fillmissing(me.data,'nearest')`. A trial is only marked as having video if the interpolation produced at least one finite sample and the trace length matches the frame count. Because the fill removes all NaNs on trials that have video, and every retained trial does have video, the declared `no_video` class ends up with zero instances in the delivered dataset.

ii.
```python
raw_motion = np.asarray(motion_trials[source_trial], dtype=np.float64).reshape(-1)
if frame_times.size >= 2 and raw_motion.size == frame_times.size:
    source_time = frame_times - video_shift - go_cue[source_trial]
    interp_motion = _interp(source_time, raw_motion, TIME)
    if np.isfinite(interp_motion).any():
        # loadMotionEnergy.m fills edge NaNs with the nearest value.
        interp_motion = _nearest_fill(interp_motion)
        motion[out_trial] = interp_motion
        motion_video[out_trial] = True
```

iii. The inline comment cites `loadMotionEnergy.m`, whose comment reads "fill nans with nearest value (there are some nans at the start of each trial)". Step 20's rule — "Visibility classes will be derived before the authors' nearest-value filling" — is applied at the trial level here (`motion_video`), rather than the bin level, since the authors treat the whole-trial trace as valid once it exists.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per session, same machinery: the 50th percentile of all finite motion-energy samples over the retained trials; `>=` → 1, `<` → 0, and 2 reserved for `no_video`. The delivered dataset is 50.0% / 50.0% / 0% — class 2 is declared in `output_values` but never occurs, because every retained trial has video and the nearest-fill leaves no NaN bins.

ii.
```python
motion_values = motion[motion_video & np.isfinite(motion)]
...
"motion_energy_median": float(np.percentile(motion_values, 50)),
...
valid = motion_video[out_trial] & np.isfinite(motion[out_trial])
output[out_trial, 5, valid] = (
    motion[out_trial, valid] >= thresholds["motion_energy_median"])
```
```python
["below_session_median", "at_or_above_session_median", "no_video"],
```

iii. Metadata: "Per-session 50th percentile over visible retained samples; class 1 is >= median." The `no_video` class is kept in `output_values` because the prompt asks for it, even though the retained trials never trigger it.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same offset and same grid as the tracking, using the **side camera's** `frameTimes` because motion energy is computed from that camera: `frameTimes − video_shift − goCue[trial]`, then `interp1` onto the 1000 bin centres. This is `loadMotionEnergy.m`'s `interp1(obj.traj{1}(trix).frameTimes − vidshift − alignTimes(trix), me.data{trix}, taxis)` exactly.

ii.
```python
frame_obj = matfile[np.asarray(trajectory_side["frameTimes"])[source_trial, 0]]
frame_times = np.asarray(frame_obj).reshape(-1, order="F")
raw_motion = np.asarray(motion_trials[source_trial], dtype=np.float64).reshape(-1)
if frame_times.size >= 2 and raw_motion.size == frame_times.size:
    source_time = frame_times - video_shift - go_cue[source_trial]
    interp_motion = _interp(source_time, raw_motion, TIME)
```

iii. Same rationale as 7-d/8-d — one clock correction, one shared bin grid for all streams.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Seven cases, handled by degrading gracefully and marking the gap rather than filling or dropping:
- **Two MATLAB file formats** — `h5py.is_hdf5` selects an HDF5 reader or a `scipy.io` reader; both paths implement the same processing.
- **Three motion-energy layouts** — `me.data` is unwrapped once more if it is itself a struct, matching `loadMotionEnergy.m`.
- **Bookkeeping arrays longer than the Bpod trial table** — `_fit_trial_mask` trims trailing excess (verified against `BPnum`/`sglxFileNum` to be trailing) and pads short masks with `False`.
- **Trials with no video** (`trials.bp.haveVid` false) or with `NdroppedFrames` NaN, or with empty `frameTimes` — the trial is kept, and its tongue/paw/motion outputs become 1000 bins of the trailing class.
- **Untracked frames** — DLC already stores x/y as NaN below likelihood 0.9; the NaNs propagate through the interpolation and become `not_visible` bins.
- **Trials outside the interpolation range** — `_interp` returns NaN outside the frame range (MATLAB `interp1` semantics) rather than extrapolating.
- **Unlabelled or oddly-typed cluster `quality`** — non-char labels become `""` and are kept, as `findClusters.m` does.
Two cases are hard failures instead: a motion-energy trial count that disagrees with `Ntrials`, and a session with no curated units or fewer than two usable trials.

ii.
```python
def _fit_trial_mask(values: np.ndarray, ntrials: int) -> np.ndarray:
    """Align bookkeeping masks to the Bpod trial count. ..."""
    values = np.asarray(values).reshape(-1).astype(bool)
    if values.size >= ntrials:
        return values[:ntrials]
    return np.pad(values, (0, ntrials - values.size), constant_values=False)
```
```python
dropped = np.asarray(dropped_obj).reshape(-1, order="F")
if dropped.size == 0 or not np.isfinite(dropped[0]):
    nan = np.full(aligned_time.shape, np.nan, dtype=np.float64)
    return nan, nan.copy(), np.zeros(aligned_time.shape, dtype=bool)
```
```python
for out_trial, source_trial in enumerate(selected):
    if not have_video[source_trial]:
        continue                       # leaves NaN -> class 2 for all three streams
```

iii. Step 87/93 on the bookkeeping overhang: "This is a source synchronization artifact, not missing data. I'm checking its trial-number mapping before trimming it so the mask stays aligned rather than assuming which end is extra"; the mapping was then verified before the rule was written. Step 20 on visibility: the classes are "derived before the authors' nearest-value filling, so 'not visible/no video' remains meaningful". Step 51 on the legacy format: "I'm adding a small compatibility path for those legacy files rather than dropping published sessions."

## 11-a. What are the most time-consuming steps of the code?

i. Three things, in roughly this order. (1) Reading the `.mat` files — each session's HDF5 object is opened once, but every cluster's `trial`/`trialtm`/`quality` is a separate object reference that has to be dereferenced individually, and the DLC `ts` array is read per trial per feature. (2) Spike binning and smoothing: for each of ~10,300 clusters (before curation drops ~7,800 garbage units) the code allocates a fresh `(Ntrials, 1000)` `float32` array and fills it with `np.add.at`, which falls back to an unbuffered scatter-add and is the single slowest array operation in the file; the direct-method convolution then runs over that same array. (3) Writing the 3.1 GB pickle at the end. The full conversion takes roughly 5 minutes (≈10 polling intervals of 30 s in the final run), against ~135 s for the human reference. The video path — four `_interp` calls plus two `_speed` calls per trial — is a per-trial Python loop but operates on ~2,000-element arrays, so it is not dominant.

ii.
```python
counts = np.zeros((ntrials, TIME.size), dtype=np.float32)
np.add.at(counts, (spike_trials, time_idx), 1.0)
units.append(_smooth_counts(counts))
```
```python
smoothed = convolve(padded, SMOOTH_KERNEL[None, :], mode="same", method="direct")
```
```python
with output_path.open("wb") as stream:
    pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The AI never profiled or discussed runtime; step 133 only distinguishes decoder-training GPU time from conversion time ("the quiet period is the per-session SVD initialization, not a stall"). The design priority stated throughout is fidelity to the repository's functions, not speed.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Two are genuinely vectorizable and were not. (1) **Per-cluster spike accumulation.** `np.add.at` on a 2-D `(trial, bin)` array is the slow unbuffered path; `np.bincount(spike_trials * 1000 + time_idx, minlength=ntrials*1000).reshape(...)` would be several times faster for the same result, and all clusters of a probe could be histogrammed in one pass (the human reference does exactly this with a single `np.histogram2d` per cluster over all trials). (2) **Per-trial, per-feature video loop.** The three streams are computed trial by trial with four `_interp` calls and two `_speed` calls each; because every trial is interpolated onto the *same* 1000-point grid, the output is rectangular, so the whole session's features could be stacked and `_speed` applied with `np.gradient(..., axis=)`. Only the ragged raw frame arrays would still need a loop.
Loops that are already vectorized: smoothing runs over all trials of a cluster at once, the per-paw average is a masked `np.stack`/`sum`, and the thresholding is a masked boolean comparison over the full session array rather than a per-bin loop.

ii. The two hot loops:
```python
for cluster_idx in range(cluster_group["trial"].shape[0]):
    ...
    counts = np.zeros((ntrials, TIME.size), dtype=np.float32)
    np.add.at(counts, (spike_trials, time_idx), 1.0)
```
```python
for out_trial, source_trial in enumerate(selected):
    ...
    for feature in ("top_paw", "bottom_paw"):
        px, py, paw_vis = _trajectory_xy(...)
```
Already vectorized:
```python
padded = np.concatenate((counts[:, :SMOOTH_BINS], counts), axis=1)
smoothed = convolve(padded, SMOOTH_KERNEL[None, :], mode="same", method="direct")
```

iii. Not discussed. The per-trial video structure follows the shape of `findPosition.m`/`findVelocity.m`, which are themselves written as `for i = 1:nTrials` loops, so the loop structure is inherited from the reference implementation rather than chosen for performance.

## 11-c. What processing does the code repeat multiple times?

i. Four repetitions, all cheap-to-moderate:
- `obj/bp/ev/goCue` is re-read from the HDF5 file **inside the per-cluster loop**, so it is fetched once per cluster (thousands of times per session) instead of once. The v5 path correctly hoists it out of the loop.
- `_feature_index` re-decodes the per-trial `featNames` cell array from scratch for every trial and every feature (3 features × ~300 trials per session), even though the names are constant within a session.
- The side camera's `frameTimes` is read twice per trial: once inside `_trajectory_xy` for the tongue and again in the motion-energy block.
- `_speed` computes `baseline` (the median of the x/y frame differences) unconditionally, including on the tongue path where it is never used.
Correctly done once: the video offset is computed once per session in `_load_behavior` (not per trial), the motion-energy file is loaded once per session, the bin grid and the smoothing kernel are built once at module level, and the per-session medians are computed once over the whole session array.

ii.
```python
for cluster_idx in range(cluster_group["trial"].shape[0]):
    ...
    go_cue = _vector(matfile["obj/bp/ev/goCue"]).astype(np.float64)   # loop-invariant
```
```python
def _feature_index(matfile, trajectory_group, trial, name) -> int:
    names_obj = matfile[np.asarray(trajectory_group["featNames"])[trial, 0]]
    names = _decode_cellstr(matfile, names_obj)
    return names.index(name)
```
```python
video_shift = (_mode(matfile["obj/sglx/bitcode/bitstart"]) / fs
               - _mode(matfile["obj/bp/ev/bitStart"]))     # once per session — correct
```

iii. Not discussed. None of these change the output; they are straightforward hoisting opportunities.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Five items.
- **Neural rates are computed for every trial, then subset.** `_load_neural` bins and smooths all `Ntrials` trials of every curated unit, builds the full `(Ntrials, n_units, 1000)` stack, and only afterwards does `neural_all[selected]` discard the early-lick, photostim and zero-neural trials — about 9% of the work. It also has to hold that full stack in memory before the firing-rate filter prunes columns.
- **The `haveEphys` machinery is dead.** `_fit_trial_mask` plus the `have_ephys` term excluded exactly 0 trials across all 44 sessions (`n_no_ephys_excluded` sums to 0); the zero-neural test that superseded it does all the work.
- **`_speed` computes `baseline` on the tongue path** and then never uses it.
- **`_load_behavior` returns `water_cued[selected]`** as its second value, which the caller discards into `_`.
- **Paw x/y are nearest-filled over gaps** whose bins are then masked out of both the average and the percentile, so the filled values only matter at the one-bin boundary of each gap.
Also retained but harmless: the per-session `quality_counts_before_curation` tallies and other diagnostics, which are small and are written into the metadata rather than discarded.

ii.
```python
neural_all, neural_info = _load_neural(matfile, probes)   # all Ntrials
neural_present = np.any(neural_all != 0, axis=(1, 2))
selected = candidate_trials[neural_present[candidate_trials]]
...
neural_selected = neural_all[selected]                    # the rest is thrown away
```
```python
output_all, _, thresholds = _load_behavior(matfile, motion_path, selected)
```
```python
differences = np.diff(xy, axis=0)
baseline = np.asarray([...])       # unused when tongue=True
```

iii. Not discussed. The all-trials-then-subset ordering is in fact *required* by the AI's own trial filter — the zero-neural test needs the neural data for the candidate trials before it can decide which to keep — so only the fully-unselected trials are genuine waste.
