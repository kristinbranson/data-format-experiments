# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Every NWB file under `data/sub-*/` is globbed and converted; nothing is sub-sampled or skipped. Files are sorted by numeric mouse and session id parsed from the file path. Each file is opened directly with `h5py` (raw HDF5 access) rather than `pynwb`, and the datasets are read by their NWB group paths (`processing/behavior/BehavioralTimeSeries/...`, `processing/ophys/Fluorescence|Neuropil|ImageSegmentation`). The result is 152 sessions / 11 mice / 12,216 original trials. Session-level metadata (`identifier`, `general/session_id`, `general/subject/subject_id`) is read from the same file.

ii.
```python
def natural_key(path: Path) -> tuple[int, int]:
    """Sort files by numeric mouse and session identifiers."""
    match = re.search(r"sub-m(\d+).*ses-(\d+)", str(path))
    if match is None:
        raise ValueError(f"Cannot parse mouse/session from {path}")
    return int(match.group(1)), int(match.group(2))


def build_dataset(data_root: Path) -> dict:
    files = sorted(data_root.glob("sub-*/*.nwb"), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No NWB files found under {data_root}")
```
```python
    with h5py.File(path, "r") as h5:
        behavior = h5["processing/behavior/BehavioralTimeSeries"]
        ...
        identifier = h5["identifier"][()].decode()
        subject = h5["general/subject/subject_id"][()].decode()
        session_id = h5["general/session_id"][()].decode()
```

iii. The AI first dumped the full HDF5 tree of an example file (`f.visititems`) and the dandiset metadata to learn the layout, then confirmed the file inventory (152 files, 11 mice, 12,216 trials, 138,678 curated ROIs) with a survey pass before writing the converter. It states in the module docstring that "A session is one NWB recording (one mouse/day); every available session is used." `h5py` is used instead of `pynwb` because the converter only needs a handful of named datasets and reads large fluorescence arrays in time slabs, which is substantially faster than the `pynwb` object layer (an explicit code comment makes this point about slab reads).

## 1-b. How are the data split into subjects?

i. Subjects are the mouse ids parsed out of the NWB file paths (`sub-m<N>`), de-duplicated and sorted numerically, giving `['m3','m4','m7','m11',...,'m19']` (11 mice). The per-session subject index is looked up from the `general/subject/subject_id` field read inside each file, so the filename-derived subject list and the in-file subject id must agree.

ii.
```python
    subjects = sorted({f"m{natural_key(path)[0]}" for path in files},
                      key=lambda value: int(value[1:]))
    subject_lookup = {subject: index for index, subject in enumerate(subjects)}
...
        subject_idx.append(subject_lookup[info["subject"]])   # info["subject"] from general/subject/subject_id
```

iii. The AI verified that the subject directory names, the filename prefixes, and the in-file `subject_id` are consistent, and that the count (11) matches the 11 switch mice reported in the paper. Numeric sorting is used so that `m3` precedes `m11` rather than sorting lexically.

## 1-c. How are the data split into sessions?

i. One NWB file = one session (one mouse-day); all 152 files become sessions in the order (mouse number, session number). No cross-day ROI alignment is attempted; each session keeps its own neuron set. Sessions with fewer than two retained trials would raise (none do; the minimum retained is 40 trials).

ii.
```python
    for number, path in enumerate(files, start=1):
        session_neural, session_input, session_output, info = convert_session(path)
        neural.append(session_neural)
        ...
        if len(neural_trials) < 2:
            raise ValueError(f"Fewer than two retained trials in {path}")
```

iii. The file naming (`sub-m3_ses-03_behavior+ophys.nwb`) and the `general/session_id` field both identify the session, and the AI checked they agree. The target format asks for per-session neuron dimensions and at least two trials per session, which the one-file-per-session mapping satisfies.

## 1-d. How are the data split into trials?

i. A trial runs from a `trial_start` sample (inclusive) to the matching `teleport` sample (exclusive), i.e. the running lap only; the teleport/inter-trial interval is not included. Boundaries are the nonzero samples of the `trial_start` and `teleport` behavioral time series. A sanity check requires equal numbers of starts and stops and `stop > start` for every pair.

ii.
```python
        starts = np.flatnonzero(behavior["trial_start/data"][:] > 0)
        stops = np.flatnonzero(behavior["teleport/data"][:] > 0)
        if len(starts) != len(stops) or np.any(stops <= starts):
            raise ValueError(f"Invalid trial boundaries in {path}")
...
        for trial, (start, stop) in enumerate(zip(starts, stops)):
```
Module docstring: "Trials run from each `trial_start` sample up to (not including) its matching `teleport` sample. Teleport periods are not included, matching the main paper analyses."

iii. This is the trial definition used throughout the paper's repository (`sess.trial_start_inds[trial]`, `sess.teleport_inds[trial]` in `behavior.get_trial_types`, and the `start_inds`/`stop_inds` pairs in `preprocessing.dff`). The AI read those functions before choosing the boundaries and used the identical convention so that the dF/F baseline windows and trial labels are computed over the same samples the paper uses.

## 1-e. How are trials filtered based on quality controls?

i. One trial-level filter is applied: trials with a stuck/damaged lick sensor — defined exactly as in the Methods and in the repo's `correct_lick_sensor_error`, i.e. more than 30% of the trial's samples have a cumulative lick count > 2 — are dropped entirely. This removes 81 of 12,216 trials (12,135 retained), exactly the count the paper reports. No minimum-duration filter is applied (the shortest surviving trial is 96 samples). Sessions with < 2 retained trials would raise; none do. Excluded trials are still used for the neural interneuron statistics and still count as the "previous trial" for the previous-outcome input.

ii.
```python
LICK_ERROR_FRACTION = 0.30
...
            trial_lick = lick[start:stop]
            bad_lick_sensor = np.mean(trial_lick > 2) > LICK_ERROR_FRACTION
...
            if bad_lick_sensor:
                excluded_lick_trials += 1
                continue
            trial_events.append(events)
            retained_trial_indices.append(trial)
```

iii. From the module docstring: "The Methods report 81 lick-detector failures as trials on which >30% of samples had cumulative lick count >2. Those trials are excluded because lick is a required categorical decoder target and NaN labels cannot be trained on." The AI swept the threshold (0.30 → 81 trials, 0.35 → 69, 0.50 → 44) and picked 0.30 because it reproduces the paper's reported 81 trials exactly. The paper NaNs those lick values rather than deleting the trial, but a NaN is not representable in a categorical decoder target, so the whole trial is dropped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From the raw Suite2p traces: `processing/ophys/Fluorescence/plane*/data` (F) and `processing/ophys/Neuropil/plane*/data` (Fneu), restricted to manually curated ROIs (`ImageSegmentation/PlaneSegmentation/iscell[:,0] == 1`) and pooled across imaging planes in the segmentation table's ROI order. The NWB `Deconvolved` field is deliberately **not** used.

ii.
```python
    series = h5[f"processing/ophys/{series_name}"]          # "Fluorescence" or "Neuropil"
    for plane_name in sorted(series.keys(), key=lambda value: int(value.removeprefix("plane"))):
        plane = int(plane_name.removeprefix("plane"))
        plane_global = np.flatnonzero(plane_idx == plane)
        local_keep = np.flatnonzero(iscell[plane_global])
        slab = np.asarray(series[plane_name]["data"][start:stop, :], dtype=np.float32)
        pieces.append(slab[:, local_keep])
        global_indices.append(plane_global[local_keep])
...
            f = read_curated_trial(h5, "Fluorescence", start, stop, iscell, plane_idx)
            f_neu = read_curated_trial(h5, "Neuropil", start, stop, iscell, plane_idx)
            dff, events = compute_dff_and_events(f, f_neu)
```

iii. The AI initially considered using the stored `Deconvolved` stream, then compared it against re-running the repository's own pipeline and found "a substantially different scale from the NWB's generic Suite2p `Deconvolved` export" (step 39), including different dF/F–speed correlation structure (39 cells over r = 0.5 from recomputed dF/F vs 0 from the stored deconvolution). It concluded that the paper analyses the signal produced by its own `preprocessing.dff(..., deconvolve=True)` and recomputed it from F/Fneu.

## 2-b. How is the `neural` data processed?

i. The repository's `preprocessing.dff` pipeline is re-implemented and applied **independently within each trial**: subtract 0.7 × neuropil, add back 0.7 × the trial-mean neuropil (so the ratio is a true dF/F), compute a maximin baseline (Gaussian smoothing σ = 15 samples, then a 300-sample running minimum followed by a 300-sample running maximum ≈ the Methods' 20 s window), form `(F − baseline)/|baseline|`, smooth with a 2-sample Gaussian, and deconvolve with Suite2p's OASIS at τ = 0.7 s and 15.5078 Hz. Non-finite values from degenerate baselines are zeroed. Work is done in float32; the stored `neural` array per trial is the deconvolved events matrix (n_neurons × n_timepoints). The teleport period is always excluded from the baseline window (the per-mouse/day `teleport_metadata.py` "imaged teleport" exception is not applied).

ii.
```python
NEUROPIL_COEF = 0.7
CALCIUM_TAU_S = 0.7
BASELINE_WINDOW_SAMPLES = 300  # paper/code: 20 s at approximately 15 Hz

def compute_dff_and_events(f, f_neu):
    corrected = f - NEUROPIL_COEF * f_neu
    corrected += NEUROPIL_COEF * np.mean(f_neu, axis=1, keepdims=True)

    baseline = gaussian_filter1d(corrected, 15, axis=1)
    baseline = minimum_filter1d(baseline, BASELINE_WINDOW_SAMPLES, axis=1)
    baseline = maximum_filter1d(baseline, BASELINE_WINDOW_SAMPLES, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dff = (corrected - baseline) / np.abs(baseline)
    dff = gaussian_filter1d(dff, 2, axis=1)
    if not np.isfinite(dff).all():
        dff = np.nan_to_num(dff, copy=False)
    events = dcnv.oasis(dff, 2000, CALCIUM_TAU_S, FRAME_RATE_HZ)
    return dff, np.asarray(events, dtype=np.float32)
```

iii. Docstring: "Neural activity is recomputed from ROI and neuropil fluorescence with the paper's pipeline: 0.7 neuropil subtraction, trial-local 20 s maximin baseline, dF/F, Gaussian smoothing (2 imaging samples), and Suite2p OASIS deconvolution with a 0.7 s calcium time constant." The AI read `preprocessing.dff` and the notebooks to take `neu_coef = 0.7`, `baseline_method = 'maximin'`, the 300-sample filters, and `tau = 0.7`, and it matched the Methods text ("baseline fluorescence was calculated within each trial independently using a maximin procedure with a 20 s sliding window … smoothed with a two-sample s.d. Gaussian kernel … deconvolving dF/F … using the OASIS algorithm as used in Suite2p"). Computing everything per trial reproduces the source function, whose baseline, smoothing and OASIS calls are all inside a per-trial loop.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both the paper's. (1) Only manually curated ROIs are read (`iscell[:, 0] == 1`), which keeps 138,678 of the ROIs in the release. (2) Putative interneurons are then removed: for each cell the Pearson correlation between its trial-period dF/F and the animal's running speed is computed over all valid trials of the session (accumulated with streaming sufficient statistics, including trials whose lick sensor failed), and cells with r > 0.5 are dropped. 402 cells (0.35 ± 0.61% per session) are removed, leaving 138,276 neurons.

ii.
```python
        iscell_table = h5["processing/ophys/ImageSegmentation/PlaneSegmentation/iscell"][:, 0]
        iscell = np.asarray(iscell_table == 1)
...
            # Interneuron curation is a neural-quality step in the paper and uses
            # every valid neural trial, including trials whose lick channel failed.
            trial_speed = np.asarray(speed[start:stop], dtype=np.float64)
            dff64 = np.asarray(dff, dtype=np.float64)
            sx += np.sum(dff64, axis=1); sx2 += np.sum(dff64 * dff64, axis=1)
            sxy += dff64 @ trial_speed
            sy += float(np.sum(trial_speed)); sy2 += float(trial_speed @ trial_speed)
            n_samples += len(trial_speed)

        speed_correlation = correlation_from_sums(n_samples, sx, sx2, sy, sy2, sxy)
        keep_neuron = speed_correlation <= 0.5
        neural_trials = [np.ascontiguousarray(events[keep_neuron], dtype=np.float32)
                         for events in trial_events]
```

iii. Docstring: "Only manually curated Suite2p ROIs (`iscell[:, 0] == 1`) are retained … Putative interneurons whose trial-period dF/F has Pearson r > 0.5 with speed are then removed, as described in Methods." The AI read `spatial.is_putative_interneuron` (method `'speed'`, correlation over the non-NaN trial samples) and `dayData` (`int_thresh = 0.5`) and reproduced both. It sanity-checked the outcome against the Methods' reported exclusion rate of 0.42 ± 0.85% of cells, obtaining 0.35 ± 0.61%. A late self-audit (step 67) moved the correlation accumulation before the bad-lick `continue` so that valid neural data from lick-failure trials still contributes to the neuron test, as it would in the paper's session-wide computation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to trial start, which is exactly the left edge of each trial slice, so no extra work is needed: sample 0 of every neural trial matrix is the `trial_start` sample. `metadata['temporal_alignment_event']` is set accordingly with `off_start = 0.0` and `off_end = None` (trials have variable length). Neural and behavioral streams share one sample index, so no resampling or shifting is performed.

ii.
```python
            f = read_curated_trial(h5, "Fluorescence", start, stop, iscell, plane_idx)
...
            "temporal_alignment_event": "start of trial at entry to the 0 cm end of the virtual corridor",
            "off_start": 0.0,
            "off_end": None,
```

iii. The behavioral series and the per-plane fluorescence arrays have the same length and the same sample grid in this release (the AI checked array lengths against the behavior timestamps in its survey), so slicing both with the same `[start:stop]` indices aligns them; and because the slice begins at the `trial_start` sample, alignment to the trial start is automatic.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native acquisition grid is kept: 15.5078125 Hz → 64.4836 ms per bin, recorded in `metadata['time_bin_size']`. No rebinning, resampling, padding or truncation is done, so trials have different numbers of samples (96 to 3,359; mean 216). The same bin size holds for every session, including the two-plane mice, whose per-plane series advertise 31.02 Hz but store one sample per plane frame at the 15.5 Hz effective rate.

ii.
```python
FRAME_RATE_HZ = 15.5078125
...
            "time_bin_size": 1000.0 / FRAME_RATE_HZ,
```
Docstring: "The dual-plane mice have scanner metadata at 62.03 Hz and per-plane series metadata at 31.02 Hz, but their arrays and aligned behavior have one sample per plane frame. The paper states the effective sampling rate is ~15.5 Hz per plane, which also matches the behavioral timestamp spacing. Thus all sessions share the same native time-bin duration."

iii. The AI checked the plane count, the stored `rate` attribute, the array lengths, and the median behavioral timestamp spacing across files (step 17), finding a median Δt of 0.06448 s everywhere, and used the effective per-plane rate both for the time-bin metadata and for the OASIS deconvolution kernel. Keeping the native grid avoids interpolating either the neural or the behavioral stream.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. From `processing/behavior/BehavioralTimeSeries/position/timestamps`, sliced to the trial.

ii.
```python
        timestamps = behavior["position/timestamps"][:]
...
            time_from_start = np.asarray(timestamps[start:stop] - timestamps[start], dtype=np.float32)
```

iii. All behavioral series in the file carry the same timestamp vector (the AI verified "ts equal" across streams in step 22), so any of them works; `position` was used since position is needed anyway. These timestamps are also the ones used for the reward-event matching, keeping one clock for the whole conversion.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Subtract the trial's first timestamp, so each trial starts at 0 s; values stored as float32 (0 to 216.5 s).

ii.
```python
            time_from_start = np.asarray(timestamps[start:stop] - timestamps[start], dtype=np.float32)
            inputs = np.vstack((
                time_from_start,
                ...
```

iii. Straightforward implementation of the instruction "Time from start of trial in seconds", using the trial start as the zero point, consistent with the stated alignment event.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. No alignment step is needed: the timestamps and the fluorescence arrays are indexed by the same sample number, and both are sliced with the identical `[start:stop]` range, so element k of the time vector corresponds to column k of the neural matrix.

ii.
```python
            f = read_curated_trial(h5, "Fluorescence", start, stop, iscell, plane_idx)
...
            time_from_start = np.asarray(timestamps[start:stop] - timestamps[start], dtype=np.float32)
```

iii. The behavior in this NWB release is already resampled onto the imaging frame grid by the authors (one behavioral sample per imaging frame; median Δt = 64.48 ms matching the imaging rate), which the AI verified before writing the converter, so common indexing is sufficient.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Primarily from the session's VR scene name, taken from the NWB `identifier` (e.g. `.../Env1_LocationC_to_A`, `.../Env1_B_to_Env2_C`), parsed into a per-trial environment code (Env1 → 0, Env2 → 1), with the switch occurring at trial 30 on environment-switch days. This is cross-checked on every trial against the `environment` behavioral time series, and a mismatch aborts the conversion.

ii.
```python
    static = re.fullmatch(r"Env([12])_Location([ABC])", scene)
    same_env_switch = re.fullmatch(r"Env([12])_Location([ABC])_to_([ABC])", scene)
    env_switch = re.fullmatch(r"Env([12])_([ABC])_to_Env([12])_([ABC])", scene)

    if static:
        environments = np.full(n_trials, int(static.group(1)) - 1, dtype=np.int8)
    elif same_env_switch:
        environments = np.full(n_trials, int(same_env_switch.group(1)) - 1, dtype=np.int8)
    elif env_switch:
        environments = np.where(np.arange(n_trials) < 30,
                                int(env_switch.group(1)) - 1,
                                int(env_switch.group(3)) - 1).astype(np.int8)
```
```python
            # The environment stream is a direct source check on scene parsing.
            stream_environment = int(round(float(np.median(env_stream[start:stop]))))
            if stream_environment != int(environments[trial]):
                raise ValueError(f"Scene/environment mismatch in {path}, trial {trial}")
```

iii. The scene name is the paper's own condition label (`behavior.get_reward_zones` switches on `sess.scene`, and `env_morph_dict` maps `Env1 → 0`, `Env2 → 1`). The AI verified that all 152 scene strings match one of its three regexes ("unmatched scenes []", step 29) and inspected the `environment` stream around the switch trial (step 22) before wiring in the assertion; because the assertion passed for all 12,216 trials, the scene-derived value is identical to the recorded environment everywhere.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Only the Env1/Env2 → 0/1 mapping and broadcasting the per-trial scalar across the trial's timepoints (the decoder input block is time-varying by construction).

ii.
```python
            environment = float(environments[trial])
            inputs = np.vstack((
                time_from_start,
                np.full(len(pos), environment, dtype=np.float32),
                ...
```

iii. The instructions call for a binary ENV1 vs ENV2 input; the environment is constant within a trial, so it is simply repeated over the trial's samples to give a uniform `(n_input, n_timepoints)` input block.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. From the `trial number` behavioral time series, taken as the median value over the trial's samples (0-based, 0–99 within a session).

ii.
```python
        trial_number_stream = behavior["trial number/data"][:]
...
            trial_number = float(np.median(trial_number_stream[start:stop]))
```

iii. The AI inspected the stream at trials 0, 29, 30 and 79 in two example sessions (step 22) and found a single constant value per trial equal to the chronological trial index (`0 trialnum [0.]`, `29 trialnum [29.]`, …), so it used the recorded stream rather than a loop counter. (Checking all 152 sessions confirms the stream is constant within every trial and exactly equals the `trial_start`-derived index, so the two choices coincide.) The median makes the extraction robust to any single-sample glitch at a trial edge.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond the median and broadcasting the constant across the trial's timepoints; the raw (unnormalized) trial index is stored. Note the index counts *original* trials, so removing a bad-lick trial leaves a gap rather than renumbering.

ii.
```python
                np.full(len(pos), trial_number, dtype=np.float32),
```

iii. The instruction asks for trial number as a continuous per-trial input; keeping the original index preserves the true experience/ordering information (in particular the position of each trial relative to the 30-trial switch point).

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the reward outcome computed for the preceding trial, which comes from `Reward/timestamps` (reward delivery events) together with the `reward_zone` stream: a trial is rewarded if at least one reward timestamp falls inside the trial's time window and the reward zone was active during the trial. Outcomes are computed for **all original trials** before any trial filtering.

ii.
```python
        reward_times = behavior["Reward/timestamps"][:]
...
        # Determine reward outcomes on all original trials before applying the lick
        # quality filter, so "previous trial" retains its literal meaning.
        outcomes = np.zeros(len(starts), dtype=np.int8)
        for trial, (start, stop) in enumerate(zip(starts, stops)):
            left = np.searchsorted(reward_times, timestamps[start], side="left")
            right = np.searchsorted(reward_times, timestamps[stop], side="left")
            outcomes[trial] = int(right > left and np.any(rzone_stream[start:stop] > 0))
```

iii. Docstring: "Reward outcome requires both a reward delivery timestamp and an active reward-zone sample, matching `behavior.get_trial_types`. Previous outcome refers to the immediately preceding original trial (not merely the preceding retained trial)." The repo's `get_trial_types` uses exactly `np.any(reward > 0) and np.any(rzone > 0)`. The AI also quantified the two criteria across the whole dataset (step 24): 10,342 trials with a reward event, 10,394 with reward-zone samples, 10,342 satisfying both — i.e. every reward event occurs in a reward-zone-positive trial, so the conjunction is well behaved.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial *t* the value is `outcomes[t-1]` (binary, omitted = 0, rewarded = 1); for the first trial of a session it is 0. The scalar is broadcast across the trial's timepoints. Because `outcomes` is indexed by original trial number, a retained trial that follows an excluded bad-lick trial still reports that excluded trial's true outcome.

ii.
```python
            previous_outcome = float(outcomes[trial - 1]) if trial > 0 else 0.0
...
                np.full(len(pos), previous_outcome, dtype=np.float32),
```

iii. Matches the instruction ("Previous trial outcome (binary, omitted = 0, rewarded = 1, per trial)"). Defaulting the first trial to 0 is the only available convention since no preceding trial exists, and keeping the original trial indexing avoids silently mislabeling the neighbours of the 81 excluded trials.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the `position` behavioral stream and the reward-zone boundaries for that trial. The zone identity per trial comes from the session's scene name (with the 30-trial switch), and the zone coordinates are the paper's: A = 80–130 cm, B = 200–250 cm, C = 320–370 cm.

ii.
```python
ZONE_BOUNDS = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}
...
        position = behavior["position/data"][:]
...
            zone = str(zones[trial])
            zone_start, zone_stop = ZONE_BOUNDS[zone]
            outputs = np.vstack((
                discretize_distance(pos, zone_start, zone_stop),
                ...
```
with the per-trial zone labels from the scene:
```python
    elif same_env_switch:
        zones = np.where(np.arange(n_trials) < 30,
                         same_env_switch.group(2), same_env_switch.group(3))
    elif env_switch:
        zones = np.where(np.arange(n_trials) < 30,
                         env_switch.group(2), env_switch.group(4))
```

iii. This reproduces the paper's own `behavior.get_reward_zones(sess, change_trial=30)`, which assigns zones from `sess.scene` using `rz_dict['X'] = [80,130]`, `['Y'] = [200,250]`, `['Z'] = [320,370]` with the `map_labels` A→X, B→Y, C→Z, and the Methods statement "Each switch occurred after 30 trials." The AI read that function and verified the reward-zone positions in the data around trials 29/30 (step 22) before adopting the rule.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance to the nearest point of the zone interval: negative before the zone (`position − zone_start`), exactly 0 anywhere inside the zone, positive after it (`position − zone_stop`). The continuous distance is then discretized (see 7-c); only the discrete label is stored.

ii.
```python
def discretize_distance(position, start, stop):
    """Categorize signed distance to the nearest point in an interval."""
    distance = np.where(position < start, position - start,
                        np.where(position > stop, position - stop, 0.0))
```

iii. Docstring: "Signed distance to the reward zone is negative before the zone, zero anywhere inside it, and positive after it (distance to the nearest zone boundary)." This matches the instruction's phrasing "Distance to any location in the reward zone" and gives the category-3 ("0 cm") bin the meaning "inside the zone", which is how the discretization table is written.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Into the seven categories from the instructions: 0: < −50 cm; 1: [−50, −10); 2: [−10, 0); 3: exactly 0 (in the zone); 4: (0, +10]; 5: (+10, +50]; 6: > +50 cm. Stored as int8, with the value names recorded in `output_values`.

ii.
```python
    result = np.empty(distance.shape, dtype=np.int8)
    result[distance < -50] = 0
    result[(distance >= -50) & (distance < -10)] = 1
    result[(distance >= -10) & (distance < 0)] = 2
    result[distance == 0] = 3
    result[(distance > 0) & (distance <= 10)] = 4
    result[(distance > 10) & (distance <= 50)] = 5
    result[distance > 50] = 6
```
```python
            ["< -50 cm", "-50 to < -10 cm", "-10 to < 0 cm", "inside reward zone",
             "> 0 to +10 cm", "> +10 to +50 cm", "> +50 cm"],
```

iii. The bin edges are taken directly from the Decoder Task specification. The masks are mutually exclusive and exhaustive, which the AI verified by checking that all seven classes occur and that the observed class range is 0–6 in the converted data.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. No separate alignment: position is sliced with the same `[start:stop]` sample range as the fluorescence, so the distance label at column k corresponds to the neural sample at column k.

ii.
```python
            pos = np.asarray(position[start:stop], dtype=np.float32)
...
                discretize_distance(pos, zone_start, zone_stop),
```

iii. Same rationale as 2-d/3-c: behavior in this release is already on the imaging frame grid, so identical indexing gives sample-by-sample alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Directly from the `position` behavioral time series (VR corridor position in cm), sliced per trial.

ii.
```python
        position = behavior["position/data"][:]
...
            pos = np.asarray(position[start:stop], dtype=np.float32)
```

iii. `position` is the animal's corridor position used throughout the paper; the AI checked its per-trial range (≈0–450 cm) in step 22 before using it.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond slicing and discretization into 5 equal 90 cm bins over the 450 cm track (see 8-c). Because the trial window stops at the teleport sample, the negative positions of the teleport/jitter zone are not included; the residual few samples marginally outside [0, 450] fall into the open end bins.

ii.
```python
                np.digitize(pos, [90.0, 180.0, 270.0, 360.0]).astype(np.int8),
```

iii. The track is 450 cm (Methods), so "5 equal-sized bins spanning the 450 cm track" is 90 cm per bin, and the raw values need no transformation.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize` with interior edges [90, 180, 270, 360]: 0: < 90; 1: [90, 180); 2: [180, 270); 3: [270, 360); 4: ≥ 360. End bins are open so that samples slightly outside the nominal track are absorbed rather than creating extra classes.

ii.
```python
                np.digitize(pos, [90.0, 180.0, 270.0, 360.0]).astype(np.int8),
...
            ["< 90 cm", "90 to < 180 cm", "180 to < 270 cm",
             "270 to < 360 cm", ">= 360 cm"],
```

iii. Exactly the discretization listed in the Decoder Task instructions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `[start:stop]` slice as the neural data; no shift or resampling.

ii.
```python
            pos = np.asarray(position[start:stop], dtype=np.float32)
```

iii. Behavior and imaging share one sample grid in the NWB release (verified from timestamps and array lengths).

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. From the `lick` behavioral time series (cumulative lick counts per imaging frame), sliced per trial. The same stream is used for the bad-sensor trial filter.

ii.
```python
        lick = behavior["lick/data"][:]
...
                (lick[start:stop] > 0).astype(np.int8),
```

iii. `lick` is the capacitive lick-sensor signal the paper uses; it is a count per sample, so a threshold is needed to obtain the required binary target.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarization: any sample with lick count > 0 becomes 1, otherwise 0. No smoothing, no spatial binning, no rate conversion. Trials whose sensor was stuck (81) were already removed, so no NaN/garbage lick labels remain.

ii.
```python
                (lick[start:stop] > 0).astype(np.int8),
...
            ["no lick", "lick"],
```

iii. The instruction requires a binary lick output (0 = no, 1 = yes). The paper likewise converts the remaining lick counts to a binary vector after the sensor-error correction; the AI reproduced the error-detection rule (see 1-e) so the binarized channel only contains trials with a trustworthy sensor.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[start:stop]` sample range as the neural data; sample-by-sample correspondence with no offset.

ii.
```python
                (lick[start:stop] > 0).astype(np.int8),
```

iii. Same reasoning as 2-d: one shared sample grid for behavior and imaging.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. From the session's VR scene name in the NWB `identifier`, parsed into a per-trial zone label A/B/C with the switch at trial 30 (see 7-a), then mapped to 0/1/2. The `reward_zone` stream is used only for the reward-outcome conjunction, not to infer the zone identity.

ii.
```python
    if static:
        zones = np.full(n_trials, static.group(2), dtype="U1")
    elif same_env_switch:
        zones = np.where(np.arange(n_trials) < 30, same_env_switch.group(2), same_env_switch.group(3))
    elif env_switch:
        zones = np.where(np.arange(n_trials) < 30, env_switch.group(2), env_switch.group(4))
...
        zone_to_class = {"A": 0, "B": 1, "C": 2}
...
                np.full(len(pos), zone_to_class[zone], dtype=np.int8),
```

iii. This is the paper's own mapping from scene to reward zone (`behavior.get_reward_zones`, `change_trial = 30`) plus the Methods' "Each switch occurred after 30 trials". The AI confirmed every scene string parses and validated the switch trial through the environment-stream assertion on environment-switch days. The labelling also covers the ~1,800 trials on which the mouse never entered an active reward zone (reward omitted / no zone samples), for which the stream alone would be uninformative.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Label → class mapping A→0, B→1, C→2, broadcast across the trial's timepoints as a time-varying (constant within trial) output; class names are recorded with their coordinates in `output_values`.

ii.
```python
                np.full(len(pos), zone_to_class[zone], dtype=np.int8),
...
            ["A (80-130 cm)", "B (200-250 cm)", "C (320-370 cm)"],
```

iii. The instructions specify "Reward zone location, per-trial. 0 = A, 1 = B, 2 = C"; the format guidance asks for time-varying outputs where possible, so the per-trial label is repeated over samples. The resulting class balance is near-uniform (0.332/0.336/0.333).

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. From `Reward/timestamps` (reward delivery events, on their own clock) combined with the `reward_zone` stream, evaluated per original trial — the same `outcomes` vector used for the previous-trial input.

ii.
```python
        reward_times = behavior["Reward/timestamps"][:]
        rzone_stream = behavior["reward_zone/data"][:]
...
            left = np.searchsorted(reward_times, timestamps[start], side="left")
            right = np.searchsorted(reward_times, timestamps[stop], side="left")
            outcomes[trial] = int(right > left and np.any(rzone_stream[start:stop] > 0))
```

iii. This mirrors `behavior.get_trial_types`, which returns `isreward = np.any(reward > 0) and np.any(rzone > 0)` per trial. Using the reward event timestamps rather than a per-sample reward channel avoids depending on how the delivery is rendered on the frame grid.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Binary search of the reward timestamps against the trial's start and end times; the trial is rewarded (1) if at least one reward timestamp lies in `[t_start, t_teleport)` and the reward zone was active in that trial, else 0. The per-trial scalar is broadcast across the trial's timepoints. The resulting rewarded fraction is 0.84, consistent with the paper's ~15% omission rate.

ii.
```python
            left = np.searchsorted(reward_times, timestamps[start], side="left")
            right = np.searchsorted(reward_times, timestamps[stop], side="left")
            outcomes[trial] = int(right > left and np.any(rzone_stream[start:stop] > 0))
...
                np.full(len(pos), outcomes[trial], dtype=np.int8),
...
            ["no reward", "rewarded"],
```

iii. The instruction asks for a binary per-trial reward outcome. The AI checked the two candidate criteria against the whole dataset before committing (10,342 reward-event trials, 10,394 reward-zone trials, 10,342 both), showing that adding the reward-zone requirement changes nothing but makes the definition identical to the paper's.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The converter is deliberately strict — it raises rather than silently patching — except where the paper defines a correction:
- **Trial boundaries**: mismatched counts of `trial_start`/`teleport` pulses, or a teleport at/ before its start, raise `ValueError`.
- **Condition metadata**: the scene-derived environment is checked against the `environment` stream on every trial; a mismatch raises.
- **Damaged lick sensor**: the paper's detection rule is applied and those 81 trials are dropped (see 1-e).
- **Degenerate baselines**: non-finite dF/F values (division by a ~zero baseline) are replaced by zeros with an explanatory comment instead of propagating NaN/Inf into the decoder.
- **Too-small sessions**: a session left with fewer than two retained trials raises, since the decoder needs at least two.
- **Missing reward-zone samples** (trials where the animal never triggered the zone): handled implicitly — the zone identity comes from the scene metadata, and reward outcome is 0.
- Behavior/neural length mismatches are handled implicitly by indexing both streams with the behavioral sample indices (the per-plane arrays are equal to, or one sample longer than, the behavior stream in this release); there is no explicit check that the imaging array covers the last trial.

ii.
```python
        if len(starts) != len(stops) or np.any(stops <= starts):
            raise ValueError(f"Invalid trial boundaries in {path}")
...
            if stream_environment != int(environments[trial]):
                raise ValueError(f"Scene/environment mismatch in {path}, trial {trial}")
...
    if not np.isfinite(dff).all():
        # Degenerate zero baselines are not expected in curated ROIs. Preserve the
        # usable samples without allowing NaNs/Infs into the decoder.
        dff = np.nan_to_num(dff, copy=False)
...
        if len(neural_trials) < 2:
            raise ValueError(f"Fewer than two retained trials in {path}")
```

iii. The AI's stated approach is to fail loudly on anything that would indicate a misread of the source layout (trial pairing, condition metadata) while reproducing the paper's own data-quality corrections where they exist (lick sensor, `iscell`, interneurons). It confirmed after the full run that no assertion fired and that the provided verifier reported "no errors and no warnings", 12,135 trials and 138,276 neurons.

## 13-a. What are the most time-consuming steps of the code?

i. The dominant costs are (1) reading the fluorescence and neuropil slabs out of HDF5 for every trial of every session — this is the I/O bound part, and it reads the full ROI dimension (all Suite2p ROIs, not just curated ones) for each time slab; (2) the per-trial dF/F filtering (Gaussian + min/max filters over cells × time) and the OASIS deconvolution, which are the CPU-bound parts (a single 80-trial session takes ~6.5 s wall but ~2 min of CPU across threads); and (3) writing the ~9.5 GB pickle at the end. The whole 152-session conversion runs in roughly 5–10 minutes.

ii.
```python
        # Reading a contiguous time slab before selecting columns is substantially
        # faster in h5py than a large two-dimensional fancy selection.
        slab = np.asarray(series[plane_name]["data"][start:stop, :], dtype=np.float32)
        pieces.append(slab[:, local_keep])
...
    events = dcnv.oasis(dff, 2000, CALCIUM_TAU_S, FRAME_RATE_HZ)
...
    with temporary.open("wb") as stream:
        pickle.dump(dataset, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The AI profiled a single session end-to-end (steps 34–36) before launching the full run, and estimated the output size in advance ("neural trial bytes f32 9.672 GB; full-session curated bytes 13.32 GB", step 21), which is why it keeps the neural arrays in float32, stores only the trial windows rather than whole sessions, and reads contiguous slabs instead of fancy-indexing HDF5.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Three loops remain: the per-trial outcome loop, the main per-trial neural loop (read → dF/F → OASIS → correlation statistics), and the per-trial input/output construction loop. The outcome loop is trivially vectorizable (`np.searchsorted` accepts arrays, and the reward-zone test could be done with a `np.add.reduceat`-style segment reduction). The per-trial behavioral discretizations (`discretize_distance`, the two `np.digitize` calls, the lick binarization) could be computed once on the full session vectors and then sliced, instead of once per trial. The neural loop is intrinsically per-trial because the paper's baseline and deconvolution are defined within a trial, though one could batch trials of similar length. The per-cell correlation is already vectorized via streaming sufficient statistics rather than a Python loop over cells.

ii.
```python
        for trial, (start, stop) in enumerate(zip(starts, stops)):
            left = np.searchsorted(reward_times, timestamps[start], side="left")
            right = np.searchsorted(reward_times, timestamps[stop], side="left")
            outcomes[trial] = int(right > left and np.any(rzone_stream[start:stop] > 0))
...
        for trial in retained_trial_indices:
            ...
            outputs = np.vstack((
                discretize_distance(pos, zone_start, zone_stop),
                np.digitize(pos, [90.0, 180.0, 270.0, 360.0]).astype(np.int8),
```

iii. These loops run at most ~100 iterations per session over short vectors, so they are negligible next to the HDF5 reads and OASIS; the AI's optimization effort went into the parts it measured as expensive (slab reads, float32, streaming correlation statistics that avoid holding the whole session's dF/F in memory).

## 13-c. What processing does the code repeat multiple times?

i. Each NWB file is opened and read exactly once — there is no separate survey pass — but within a session some work is repeated: the trial loop is executed twice (once for neural processing, once to build inputs/outputs), so `starts`/`stops` are re-indexed and `lick[start:stop]` is sliced a second time (once for the sensor test, once for the lick output). `read_curated_trial` recomputes the per-plane `iscell`/`planeIdx` index maps and the ROI reordering `argsort` for every trial and for both Fluorescence and Neuropil, when they could be computed once per session. Behavioral streams are read whole into memory once and then sliced, which is not repeated work.

ii.
```python
        for plane_name in sorted(series.keys(), key=...):
            plane_global = np.flatnonzero(plane_idx == plane)
            local_keep = np.flatnonzero(iscell[plane_global])
            ...
    order = np.argsort(np.concatenate(global_indices))
```
```python
        for trial, (start, stop) in enumerate(zip(starts, stops)):   # neural pass
            ...
        for trial in retained_trial_indices:                          # behavior pass
```

iii. The two-pass structure inside a session is deliberate: the interneuron mask can only be applied once the speed correlation has been accumulated over all trials, so the neural pass must finish before the final per-trial arrays are assembled. The repeated index bookkeeping in `read_curated_trial` is cheap relative to the slab read it accompanies.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A small amount:
- dF/F **and** OASIS events are computed for the 81 bad-lick trials, but their events are then discarded (only the dF/F is genuinely needed, for the speed correlation).
- Events are deconvolved for all curated cells, including the 402 later dropped as putative interneurons.
- Each HDF5 slab read pulls in every ROI column (e.g. 2,929 ROIs) before keeping only the curated ones (e.g. 1,052), i.e. roughly 2–3× more bytes than strictly needed — a deliberate speed-vs-bytes trade documented in a comment.
- The `dff` return value of `compute_dff_and_events` is used only for the interneuron statistics and never stored.
- The `environment` stream is read and median-reduced per trial purely as a consistency assertion.

ii.
```python
            dff, events = compute_dff_and_events(f, f_neu)   # events unused if bad_lick_sensor
            ...
            if bad_lick_sensor:
                excluded_lick_trials += 1
                continue
...
        keep_neuron = speed_correlation <= 0.5
        neural_trials = [np.ascontiguousarray(events[keep_neuron], dtype=np.float32)
                         for events in trial_events]
```

iii. Each of these is a consequence of an ordering constraint or a deliberate trade-off that the AI documents: the interneuron test must see all valid neural trials (the fix applied at step 67), so dF/F must be computed for excluded trials, and the neuron mask is only known after the whole session, so deconvolution cannot be restricted to the surviving cells in a single pass. The full-width slab read is justified in a code comment as being faster in h5py than a two-dimensional fancy selection, and the environment assertion is cheap insurance against a scene-parsing error.
