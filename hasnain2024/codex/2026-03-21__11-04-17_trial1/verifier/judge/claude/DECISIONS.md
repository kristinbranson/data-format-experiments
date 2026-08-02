# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI parses the reference MATLAB loader scripts (`load*_ALMVideo.m`) in the `code/DataLoadingScripts/Recording and video/` directory to recover the analyzed session list and probe selections. It then intersects these with available data files in `data/` (matching by subject and date). Each session's `.mat` file is loaded using either `h5py` (for MATLAB v7.3/HDF5 files) or `scipy.io.loadmat` (for older `.mat` files). Motion energy files are loaded separately from `motionEnergy_<subject>_<date>.mat` files.

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

def parse_reference_session_specs(code_dir: Path, data_dir: Path) -> list[SessionSpec]:
    data_files = find_data_files(data_dir)
    specs: list[SessionSpec] = []
    loader_dir = code_dir / "DataLoadingScripts" / "Recording and video"
    for loader in sorted(loader_dir.glob("load*_ALMVideo.m")):
        # ... parses anm, date, probe from loader scripts
        if key in data_files:
            specs.append(SessionSpec(...))
    return specs
```

iii. The AI justified this approach in CONVERSION_NOTES.md Step 4: "Use the intersection of reference-code session lists and available data files. This yields 44 code-selected, paper-consistent ephys sessions." This avoids including raw sessions not referenced by the loader scripts.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the animal name (`anm`) parsed from each loader script and the filenames. Unique subject names are collected across retained sessions and stored in a list. Each session maps to a subject via index.

ii.
```python
if session["subject"] not in subject_names:
    subject_names.append(session["subject"])
subject_index.append(subject_names.index(session["subject"]))
```

iii. The AI documented 14 unique subjects across the available dataset intersection. The animal IDs come directly from the reference loader scripts.

## 1-c. How are the data split into sessions?

i. Each session is a unique (subject, date) combination corresponding to one data file. The 44 sessions come from the intersection of reference loader scripts and available data files. Sessions are processed one-at-a-time in a loop.

ii.
```python
for spec in session_specs:
    session = convert_one_session(spec, show_processing=show_processing, outdir=outdir)
    if session is None:
        continue
    neural.append(session["neural"])
    inputs.append(session["input"])
    outputs.append(session["output"])
```

iii. CONVERSION_NOTES Step 4: "Use the intersection of reference-code session lists and available data files." The AI verified this yields 25 fixed-delay + 19 randomized-delay = 44 sessions, matching the paper.

## 1-d. How are the data split into trials?

i. Trials are the behavioral trials within each session, indexed by the `obj.bp` fields (R, L, hit, miss, etc.). Each trial has associated neural spike data, video tracking data, and motion energy. Trials are split per session using the indices from the behavioral data arrays.

ii.
```python
valid = session_valid_trial_mask(raw)
selected_trials = np.flatnonzero(valid)
```

iii. The AI uses the raw trial structure from the session files, where each element of the behavioral arrays corresponds to one trial.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) no optogenetic stimulation (`stim_enable == 0`), (2) no early lick (`early == 0`), (3) must be a hit or miss trial (`hit == 1 | miss == 1`), and (4) must have a defined lick direction (`R == 1 | L == 1`). Additionally, trials whose raw index exceeds the last neural unit trial index are excluded.

ii.
```python
def session_valid_trial_mask(raw: dict) -> np.ndarray:
    return (
        (raw["stim_enable"] == 0)
        & (raw["early"] == 0)
        & ((raw["hit"] == 1) | (raw["miss"] == 1))
        & ((raw["R"] == 1) | (raw["L"] == 1))
    )

# Additional neural coverage filter:
covered_trial_max = [int(np.nanmax(unit["trial"])) for unit in raw["units"]
                     if good_quality(unit["quality"]) and np.asarray(unit["trial"]).size]
if covered_trial_max:
    max_neural_trial = min(raw["R"].size, max(covered_trial_max))
    selected_trials = selected_trials[selected_trials + 1 <= max_neural_trial]
```

iii. CONVERSION_NOTES Step 5: "Exclude stimulation, early-lick, and ignore/no-response trials: Reference analyses consistently use `~stim.enable`, `~early`, and usually hit/miss conditions." The AI included miss trials (unlike the default reference conditions which only use hit) because the decoder task requires predicting outcome (correct vs incorrect), which necessitates both hit and miss trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `obj.clu{probe}` spike data - specifically the `trialtm` (within-trial spike times) and `trial` (trial assignment) fields for each unit on the ALM-designated probe.

ii.
```python
units.append({
    "quality": mat_to_str(getattr(unit, "quality", "")),
    "trialtm": np.asarray(getattr(unit, "trialtm"), dtype=np.float64).reshape(-1),
    "trial": np.asarray(getattr(unit, "trial"), dtype=np.int64).reshape(-1),
})
```

iii. CONVERSION_NOTES Step 1: "Neural data are sorted spike times in `obj.clu{probe}(cluster)` with per-spike session time (`tm`), per-spike within-trial time (`trialtm`), and trial index (`trial`)."

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue event, then binned into 5 ms bins over a [-2.5, 2.5] s window. Binned spike counts are divided by dt to convert to firing rates, then smoothed with a causal Gaussian kernel (window size 15 bins, reflected boundary condition).

ii.
```python
TMIN = -2.5; TMAX = 2.5; DT = 1.0 / 200.0; SMOOTH = 15; BCTYPE = "reflect"

def compute_unit_trial_matrix(unit, go_cue, trial_to_pos, n_sel, time_edges):
    aligned = unit["trialtm"] - go_cue[unit["trial"] - 1]
    # ... bin spikes
    bins = np.floor((aligned[keep] - time_edges[0]) / DT).astype(np.int64)
    np.add.at(mat, (trial_pos[keep], bins), 1.0 / DT)
    mat = my_smooth(mat.T, SMOOTH, BCTYPE).T
    return mat
```

iii. CONVERSION_NOTES Step 5: "Use a fixed window of [-2.5, 2.5] s around goCue with a 5 ms bin (dt = 1/200): This matches the default reference processing pipeline."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) Quality-based: exclude units labeled `garbage`, `gabrga`, `noisy`, or `real?`. (2) Firing-rate-based: exclude units with mean firing rate <= 1 Hz. Sessions with fewer than 10 kept units are skipped.

ii.
```python
def good_quality(label: str) -> bool:
    label = label.strip().lower()
    return label not in {"garbage", "gabrga", "noisy", "real?"}

LOW_FR_HZ = 1.0
# In convert_one_session:
if not good_quality(unit["quality"]):
    continue
# ...
if float(unit_mat.mean()) <= LOW_FR_HZ:
    continue
# ...
if kept_units < 10:
    log(f"SKIP {spec.session_id}: only {kept_units} units after quality/FR filtering")
    return None
```

iii. CONVERSION_NOTES Step 4: "Resolve in favor of the methods text and figure scripts: use a 1 Hz low-FR threshold when matching the analyzed neural population." The quality labels match `findClusters.m` with the `'all'` quality setting.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is aligned to the go cue by subtracting the go cue time for the spike's trial: `aligned = trialtm - goCue[trial - 1]`.

ii.
```python
aligned = unit["trialtm"] - go_cue[unit["trial"] - 1]
```

iii. CONVERSION_NOTES Step 5: "Use `goCue` as the universal alignment event: This matches the user request and the default reference alignment."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 5 ms (dt = 1/200 s). No additional temporal rebinning is applied after the initial binning. The time axis spans [-2.5, 2.5] s yielding 1000 time bins.

ii.
```python
DT = 1.0 / 200.0
time_edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time_vec = time_edges[:-1] + DT / 2.0
```

iii. CONVERSION_NOTES Step 4: "Tentatively prefer the paper/code-aligned 5 ms default." The metadata stores `time_bin_size: 5.0` (ms).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is derived from the time axis itself (the bin centers of the [-2.5, 2.5] s window), not from any specific raw data variable. It represents the time relative to the go cue alignment event.

ii.
```python
input_trials.append(time_vec[None, :].astype(np.float32))
```

iii. This is a direct consequence of the alignment: the decoder input is the time coordinate of each bin center relative to the go cue.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time vector is computed as bin centers: `time_edges[:-1] + DT / 2.0`, where `time_edges = np.arange(-2.5, 2.5 + DT, DT)`. The same time vector is repeated for every trial, shaped as `(1, n_timepoints)`.

ii.
```python
time_vec = time_edges[:-1] + DT / 2.0
# ...
input_trials.append(time_vec[None, :].astype(np.float32))
```

iii. The time vector construction matches the reference `getSeq.m`: `obj.time = edges + params.dt/2; obj.time = obj.time(1:end-1);`

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time vector is identical to the neural time axis by construction - both use the same bin centers. No additional alignment is needed.

ii.
```python
# Same time_vec used for both neural binning and input:
time_edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time_vec = time_edges[:-1] + DT / 2.0
```

iii. The AI verified in sanity checks (Step 10) that `input[0]` matches the neural bin centers exactly (`np.allclose = True, max absolute difference 0.0`).

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `obj.bp.R` and `obj.bp.L` - the right and left lick indicators for each trial.

ii.
```python
lick_direction = np.int64(1 if raw["R"][trial_idx] == 1 else 0)
```

iii. CONVERSION_NOTES Step 5: "Lick direction: left 0, right 1; encode as a constant time series over the trial window." R=1 maps to right (1), otherwise left (0).

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. It's a direct binary mapping: R==1 yields 1 (right), otherwise 0 (left). The per-trial scalar is broadcast to a constant time series matching the neural time axis.

ii.
```python
lick_direction = np.int64(1 if raw["R"][trial_idx] == 1 else 0)
# ...
np.full(time_vec.size, lick_direction, dtype=np.int64),
```

iii. The instruction specifies left=0, right=1. The AI maps R==1 to right (1) and L==1 (the else case) to left (0).

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj.bp.autowater` - the autowater flag for each trial.

ii.
```python
context = np.int64(1 if raw["autowater"][trial_idx] == 0 else 0)
```

iii. CONVERSION_NOTES Step 5: "Behavioral context: WC 0, DR 1 via 1 - autowater."

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct binary mapping: autowater==0 (DR trial) maps to 1, autowater==1 (WC trial) maps to 0. This is equivalent to `1 - autowater`. The per-trial scalar is broadcast to a constant time series.

ii.
```python
context = np.int64(1 if raw["autowater"][trial_idx] == 0 else 0)
# ...
np.full(time_vec.size, context, dtype=np.int64),
```

iii. The instructions specify WC=0, DR=1. In the reference code, `autowater=1` corresponds to WC. The AI's mapping correctly inverts this.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit` - the hit indicator for each trial.

ii.
```python
outcome = np.int64(1 if raw["hit"][trial_idx] == 1 else 0)
```

iii. CONVERSION_NOTES Step 5: "Outcome: miss 0, hit 1."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A direct binary mapping: hit==1 maps to 1 (correct), otherwise 0 (incorrect, i.e. miss). The per-trial scalar is broadcast to a constant time series.

ii.
```python
outcome = np.int64(1 if raw["hit"][trial_idx] == 1 else 0)
# ...
np.full(time_vec.size, outcome, dtype=np.int64),
```

iii. Only hit and miss trials are included (no/ignore trials are filtered out), so the else case corresponds to miss (incorrect).

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from `obj.traj{1}` (side-view camera), specifically the `tongue` feature's x,y position time series (`ts`), frame times (`frameTimes`), and the video synchronization data (`sglx.bitcode.bitstart`, `sglx.fs`, `bp.ev.bitStart`).

ii.
```python
tongue_pos = feature_xy(raw, 0, "tongue", raw["events"]["goCue"], time_vec)
tongue_speed = feature_speed(*tongue_pos, "tongue")
```

iii. CONVERSION_NOTES Step 5: "Define tongue velocity from the side-view `tongue` marker speed magnitude."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The processing pipeline:
1. Compute video offset from bitcode synchronization metadata.
2. Extract tongue x,y positions from the side-view DLC tracking data.
3. Interpolate positions onto the neural time axis using `old_t = frameTimes - vidshift - alignTimes[trix]`.
4. For tongue: do NOT fill missing (NaN) values (unlike other features).
5. Compute x and y velocity using `np.gradient()`.
6. For tongue: set NaN velocities to 0 (rather than nearest-fill).
7. Compute speed magnitude: `sqrt(xvel^2 + yvel^2)`.
8. Mask speed as NaN where original positions were NaN.

ii.
```python
def feature_speed(xpos, ypos, feature_name):
    # ...
    xv = np.gradient(tsinterp[:, 0])
    yv = np.gradient(tsinterp[:, 1])
    if "tongue" not in feature_name:
        xv = xv - basederiv[0]
        yv = yv - basederiv[1]
        xv = nearest_fill_1d(xv); yv = nearest_fill_1d(yv)
    else:
        xv = np.nan_to_num(xv, nan=0.0)
        yv = np.nan_to_num(yv, nan=0.0)
    return np.sqrt(xvel**2 + yvel**2)
```

iii. The processing follows the reference `findPosition.m` and `findVelocity.m` pipeline, with the difference that the AI computes a scalar speed magnitude rather than keeping x/y velocity components separate.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Tongue speed is discretized into two bins using the per-session 50th percentile (median) as the threshold. Values below the median are 0, values at or above are 1. NaN values (where tongue was not visible) are assigned to bin 0.

ii.
```python
tongue_thr = summarize_threshold(tongue_sel)
# ...
np.where(
    np.isfinite(tongue_sel[:, local_idx]),
    tongue_sel[:, local_idx] >= tongue_thr,
    0,
).astype(np.int64),
```

iii. The instructions specify: "Tongue velocity: discretized into two bins with per-session threshold: 0: < 50th percentile, 1: >= 50th percentile."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue positions are interpolated from the video frame times onto the same neural time axis (`time_vec`) using linear interpolation. The video-to-neural timing offset is corrected using the bitcode synchronization metadata. Go cue alignment is applied by subtracting the go cue time from the frame times.

ii.
```python
old_t = frame_times - vidshift - align_times[trix]
xpos[:, trix] = interp_to_taxis(old_t, ts[:, 0], taxis)
```

iii. This matches the reference `findPosition.m`: `interp1(traj(trix).frameTimes-vidshift-obj.bp.ev.(alignEv)(trix), ts, taxis)`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from `obj.traj{2}` (bottom-view camera), specifically the `top_paw` and `bottom_paw` features' x,y position time series.

ii.
```python
for paw_name in ("top_paw", "bottom_paw"):
    paw_pos = feature_xy(raw, 1, paw_name, raw["events"]["goCue"], time_vec)
    paw_speeds.append(feature_speed(*paw_pos, paw_name))
```

iii. CONVERSION_NOTES Step 5: "Define paw velocity from the average of top- and bottom-paw speed magnitudes in the bottom view."

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The processing pipeline:
1. Compute video offset from bitcode synchronization.
2. Extract x,y positions for both `top_paw` and `bottom_paw` from the bottom-view DLC data.
3. Interpolate positions onto the neural time axis.
4. For paws: fill missing values with nearest available value.
5. Compute x and y velocity using `np.gradient()`.
6. Subtract baseline velocity (median of diff) from both x and y components.
7. Fill missing velocities with nearest value.
8. Compute speed magnitude for each paw: `sqrt(xvel^2 + yvel^2)`.
9. Average the two paw speeds per timepoint.

ii.
```python
paw_stack = np.stack(paw_speeds, axis=0)
paw_count = np.sum(np.isfinite(paw_stack), axis=0)
paw_sum = np.nansum(paw_stack, axis=0)
paw_speed = np.divide(paw_sum, np.maximum(paw_count, 1), ...)
paw_speed = np.nan_to_num(paw_speed, nan=0.0)
```

iii. CONVERSION_NOTES Step 5: "Compute top- and bottom-paw speed magnitudes from x/y velocity pairs, average them per timepoint." The AI noted paws are only tracked in the bottom view per the methods.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw speed is discretized into two bins using the per-session 50th percentile (median). Values below the median are 0, values at or above are 1.

ii.
```python
paw_thr = summarize_threshold(paw_sel)
# ...
(paw_sel[:, local_idx] >= paw_thr).astype(np.int64),
```

iii. Follows the instruction: "Paw velocity: discretized into two bins with per-session threshold: 0: < 50th percentile, 1: >= 50th percentile."

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue - paw positions are interpolated from video frame times onto the neural time axis using linear interpolation, with video offset correction and go cue alignment.

ii.
```python
paw_pos = feature_xy(raw, 1, paw_name, raw["events"]["goCue"], time_vec)
```

iii. The alignment uses the same `interp_to_taxis` function and video offset correction as tongue, matching the reference `findPosition.m` approach.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_<subject>_<date>.mat` files, which contain `me.data` (per-trial motion energy traces at 400 Hz). Frame times for alignment come from the side-view camera's `frameTimes`.

ii.
```python
me_path = spec.session_path.parent / f"motionEnergy_{spec.subject}_{spec.date}.mat"
motion_energy, motion_thresh = load_motion_energy(me_path)
```

iii. CONVERSION_NOTES Step 1: "Motion energy loader expects one motion-energy file per ephys session."

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The processing pipeline:
1. Load per-trial motion energy traces from the `.mat` file.
2. Compute video offset from bitcode synchronization.
3. Interpolate motion energy onto the neural time axis using the side-view camera frame times (or a fallback of `(arange + 1) / 400 - 0.5` if frame times are unavailable).
4. Fill NaN values with nearest available value.

ii.
```python
def aligned_motion_energy(raw, align_times, time_vec):
    for trix, me in enumerate(me_trials):
        frame_times = side_trials[trix]["frame_times"]
        if frame_times is None or ...:
            old_t = (np.arange(me.size) + 1.0) / 400.0 - 0.5 - align_times[trix]
        else:
            old_t = frame_times - vidshift - align_times[trix]
        out[:, trix] = interp_to_taxis(old_t, me, time_vec)
        out[:, trix] = nearest_fill_1d(out[:, trix])
    return out
```

iii. This closely follows the reference `loadMotionEnergy.m`, including the fallback for missing frame times.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized into two bins using the per-session 50th percentile (median). Values below the median are 0, values at or above are 1.

ii.
```python
motion_thr = summarize_threshold(motion_sel)
# ...
(motion_sel[:, local_idx] >= motion_thr).astype(np.int64),
```

iii. The instructions specify median-based thresholding. The AI noted that this differs from the reference paper's manual movement threshold but follows the decoder task specification.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is interpolated from video frame times onto the neural time axis using linear interpolation, with video offset correction and go cue alignment. This uses the same frame-time-based alignment as the kinematic features.

ii.
```python
old_t = frame_times - vidshift - align_times[trix]
out[:, trix] = interp_to_taxis(old_t, me, time_vec)
```

iii. This matches the reference `loadMotionEnergy.m`: `interp1(obj.traj{1}(trix).frameTimes-vidshift-alignTimes(trix), me.data{trix}, taxis)`.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several data issues:
- Empty HDF5 probe slots: Indexes raw probe slots directly rather than a compressed non-empty list.
- Nested motion energy structs: Recursively unwraps `.data` containers for JEB15 sessions.
- Missing `moveThresh` in randomized-delay motion energy files: Records NaN for the unused manual threshold.
- Behavioral trials exceeding neural coverage: Drops valid behavioral trials whose raw trial index exceeds the last neural `unit.trial` index.
- Missing video frame times: Falls back to synthetic frame times at 400 Hz with a 0.5 s offset.
- Missing/NaN tongue positions: Tongue speed set to 0 where position was NaN; tongue bin assigned 0 for non-visible periods.
- Missing/NaN paw/motion data: Filled with nearest available value.
- NaN-dropped-frames indicator: Skips trials with NaN `NdroppedFrames` (indicates bad video data).

ii.
```python
# Missing frame times fallback:
if frame_times is None or frame_times.size == 0:
    frame_times = (np.arange(ts.shape[0]) + 1.0) / 400.0

# Neural coverage filter:
if covered_trial_max:
    max_neural_trial = min(raw["R"].size, max(covered_trial_max))
    selected_trials = selected_trials[selected_trials + 1 <= max_neural_trial]

# Tongue NaN handling:
np.where(np.isfinite(tongue_sel[:, local_idx]),
         tongue_sel[:, local_idx] >= tongue_thr, 0).astype(np.int64)
```

iii. CONVERSION_NOTES Step 9 documents several edge cases found and fixed during full conversion, including JEB15 motion energy wrapping, JEB6 empty probe slot, JEB24 trailing behavioral trials, and randomized-delay motion energy format differences.

## 11-a. What are the most time-consuming steps of the code?

i. According to CONVERSION_NOTES Step 7, the full conversion ran in ~135 seconds for 44 sessions (~3 s/session average). The most time-consuming steps are: (1) loading each session's `.mat` file (especially large HDF5 files), (2) computing per-unit spike binning and smoothing across all trials, and (3) interpolating video/kinematic features onto the neural time axis for all trials.

ii.
```python
# Per-session timing logged:
log(f"SESSION {spec.session_id}: ... time={time.time() - t0:.2f}s")
```

iii. CONVERSION_NOTES Step 7: "sample conversion finished in 8.12 s for 2 sessions" and "Full-run estimate for 44 retained sessions: ~3.5-5 minutes."

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain that could potentially be vectorized:
- The per-unit loop in `compute_unit_trial_matrix` processes one unit at a time. While individual spike binning uses vectorized `np.add.at`, the outer loop over units is serial.
- The per-trial loop in `feature_speed` computes velocity for each trial independently.
- The per-trial loop in `aligned_motion_energy` interpolates motion energy for each trial independently.
- The per-column loop in `my_smooth` applies convolution column-by-column.

ii.
```python
# Per-unit loop:
for unit in raw["units"]:
    if not good_quality(unit["quality"]): continue
    unit_mat = compute_unit_trial_matrix(...)

# Per-trial velocity loop:
for i in range(n_trials):
    tsinterp = np.column_stack([xpos[:, i], ypos[:, i]])
    xv = np.gradient(tsinterp[:, 0])
    yv = np.gradient(tsinterp[:, 1])

# Per-column smoothing loop:
for j in range(x_filt.shape[1]):
    out[:, j] = np.convolve(x_filt[:, j], kern, mode="same")
```

iii. CONVERSION_NOTES Step 6 notes: "Implemented vectorized per-unit spike accumulation with `np.add.at` instead of calling a histogram inside nested trial loops." The smoothing loop could use `scipy.ndimage.convolve1d` or `scipy.signal.fftconvolve` for batch processing.

## 11-c. What processing does the code repeat multiple times?

i. The video offset computation (`find_video_offset`) is called multiple times per session: once for tongue position, once for each paw position, and once for motion energy alignment. It could be computed once and reused.

ii.
```python
# Called in feature_xy for each feature:
vidshift = find_video_offset(raw)

# Called again in aligned_motion_energy:
vidshift = find_video_offset(raw)
```

iii. While the result is deterministic for a given session, the function is called 4 times per session (tongue, top_paw, bottom_paw, motion_energy). This is a minor inefficiency.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes continuous speed values for tongue, paw, and motion energy and stores them in `session_out["continuous"]` for plotting purposes, even though only the binarized versions are used in the final output. This doubles the memory for these variables during processing.

The code also computes features for ALL trials (before filtering to selected_trials) in some functions. For example, `feature_xy` and `aligned_motion_energy` compute values for all trials, then index into `selected_trials` afterwards.

ii.
```python
# Continuous values computed but only used for plotting:
session_out = {
    # ...
    "continuous": {
        "tongue": tongue_sel.astype(np.float32),
        "paw": paw_sel.astype(np.float32),
        "motion": motion_sel.astype(np.float32),
        "tongue_thr": tongue_thr,
        "paw_thr": paw_thr,
        "motion_thr": motion_thr,
    },
}

# Features computed for all trials, then subsetted:
tongue_sel = tongue_speed[:, selected_trials]
paw_sel = paw_speed[:, selected_trials]
motion_sel = motion[:, selected_trials]
```

iii. The continuous values are used in `plot_processing` when `--show-processing` is enabled, but are discarded when building the final dataset. Computing features for all trials before filtering is a minor inefficiency - the kinematic and motion energy computations run over all behavioral trials even though only the selected subset is needed.
