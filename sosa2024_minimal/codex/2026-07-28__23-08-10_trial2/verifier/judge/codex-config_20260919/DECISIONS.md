# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively globs every matching NWB file under `/app/data/sub-*`, sorts them by numeric mouse/session, reads identifying metadata with `h5py`, and processes every discovered session.

ii.
```python
for file_path in sorted(data_root.glob("sub-*/sub-*_behavior+ophys.nwb")):
    with h5py.File(file_path, "r") as f:
        subject = decode_scalar(f["general/subject/subject_id"][()])
...
for meta in session_meta:
    session_data, stats = process_session(meta, block_size=block_size)
```

iii. The trajectory says the agent bulk-scanned all 154 sessions and chose direct `h5py` access, one session at a time, for memory safety.

## 1-b. How are the data split into subjects?

i. Subject IDs are read from NWB metadata, deduplicated, numerically sorted, and mapped to each session through `subject_idx`.

ii.
```python
subjects = sorted({meta.subject for meta in session_meta}, key=lambda s: int(s[1:]))
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
dataset["subject_idx"].append(subject_to_idx[meta.subject])
```

iii. The agent inspected both file structure and NWB metadata, using metadata rather than merely trusting directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session; its number is parsed from `ses-NN`, and one nested trial list is appended per file.

ii.
```python
session_num = int(re.search(r"ses-(\d+)", file_path.name).group(1))
dataset["neural"].append(session_data["neural"])
```

iii. The trajectory’s 154-session scan verified the file/session correspondence before conversion.

## 1-d. How are the data split into trials?

i. Trial starts are every positive `trial_start` sample and stops are every positive `teleport` sample; slices are half-open `[start:stop)`.

ii.
```python
trial_start_idx = np.where(trial_start_series > 0)[0]
trial_stop_idx = np.where(teleport_series > 0)[0]
...
neural_trial = kept_events[:, start:stop]
```

iii. The agent inspected trial-start/teleport consistency across all sessions and treated teleport as the trial endpoint described by the dataset.

## 1-e. How are trials filtered based on quality controls?

i. It drops a trial when more than 30% of its samples have raw lick count greater than 2, the paper’s lick-sensor-error rule. It does not apply the reference converter’s short-trial filter.

ii.
```python
if np.mean(lick_counts[start:stop] > 2.0) > 0.3:
    mask[trial_idx] = True
kept_trial_indices = np.where(~lick_error_mask)[0]
```

iii. The agent explicitly states in metadata and runtime summaries that this matches the paper’s lick-sensor quality criterion; the full run removed 81 trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural output is derived from each plane’s Suite2p `Fluorescence` and `Neuropil` arrays, plus ROI `iscell` metadata; it does not use stored deconvolved traces.

ii.
```python
fluorescence_ds = ophys["Fluorescence"][plane_key]["data"]
neuropil_ds = ophys["Neuropil"][plane_key]["data"]
iscell = imgseg["iscell"][:]
```

iii. The trajectory says the agent traced the original repository’s dF/F procedure and determined that the paper recomputed events from fluorescence and neuropil.

## 2-b. How is the `neural` data processed?

i. Per plane and cell block, it keeps trial samples, subtracts `0.7*neuropil`, adds back trial mean neuropil, computes a trialwise Gaussian/maximin baseline (sigma 15, 300-sample min then max), forms dF/F, smooths with sigma 2, and OASIS-deconvolves with tau 0.7. Unlike the reference, it never conditionally extends baseline windows through teleports.

ii.
```python
f -= neu_coef * f_neu
baseline[:, sl] = nansmooth_2d(f[:, sl], sigma=15)
baseline[:, sl] = minimum_filter1d(baseline[:, sl], size=300, axis=1, mode="nearest")
baseline[:, sl] = maximum_filter1d(baseline[:, sl], size=300, axis=1, mode="nearest")
dff[:, valid] = (f[:, valid] - baseline[:, valid]) / np.abs(baseline[:, valid])
dff[:, sl] = nansmooth_2d(dff[:, sl], sigma=2)
events[:, sl] = dcnv.oasis(dff[:, sl], 2000, tau, frame_rate_hz)
```

iii. The agent justified the parameters from the paper/repository and smoke-tested the interneuron rate before scaling up. The trajectory does not mention discovering or implementing the paper’s session-specific `keep_teleports` rule.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It retains manually curated `iscell` ROIs, then removes cells whose dF/F–speed Pearson correlation is finite and greater than 0.5.

ii.
```python
plane_iscell = iscell[plane_slice, 0] == 1
block_corr = np.array([np.corrcoef(trace, speed_valid)[0, 1] ...])
block_interneuron = np.isfinite(block_corr) & (block_corr > 0.5)
```

iii. The agent identified both filters in the paper and checked the exclusion fraction during a smoke test and the full conversion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is by slicing deconvolved events at each `trial_start`; sample zero of each trial is the alignment event.

ii.
```python
start = trial_start_idx[trial_idx]
stop = trial_stop_idx[trial_idx]
kept_events[:, start:stop]
```

iii. The agent regarded the NWB behavior and imaging samples as already index-aligned, so no resampling or shift was added.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. Per-session rate is inferred from median positive position-timestamp differences; metadata stores `1000 / mean_frame_rate` (about 64.5 ms).

ii.
```python
return float(1.0 / np.median(diffs))
...
dataset["metadata"]["time_bin_size"] = 1000.0 / mean_frame_rate
```

iii. The agent used the observed behavior sampling interval because neural and behavior arrays share samples, and validated frame-rate statistics across sessions.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the raw `position` timestamps.

ii.
```python
timestamps = behavior["position"]["timestamps"][:].astype(np.float64)
```

iii. The agent found the behavioral streams aligned and selected one timestamp stream as the common clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at trial start is subtracted from every timestamp in that trial and cast to float32.

ii.
```python
trial_times = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. This directly implements time relative to the requested trial-start event.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The same `[start:stop)` indices are used for timestamps and neural events, producing equal-length arrays.

ii.
```python
neural_trial = np.concatenate(block_list, axis=0)
trial_times = timestamps[start:stop] - timestamps[start]
```

iii. The agent relied on the released arrays’ common sampling/indexing and verified the converted structure.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The saved value is derived from the NWB `identifier` scene string (`Env1`/`Env2`), not directly from the environment time series; the latter is only used to validate it.

ii.
```python
scene = parts[-1]
env = int(match.group("env")) - 1
...
np.full(trial_times.shape, float(label.env), dtype=np.float32)
```

iii. The agent inspected environment coding and scene names across all sessions and added a runtime mismatch check against raw environment values.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Regex parsing converts scene environment numbers from one-based to binary zero-based labels; environment-switch sessions switch at trial index 30, then the scalar is repeated over the trial.

ii.
```python
env0 = int(match.group("env0")) - 1
env1 = int(match.group("env1")) - 1
env=env0 if trial_idx < switch_trial_count else env1
```

iii. The agent used known protocol structure and validates the result against the raw stream, failing on disagreement.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It uses the raw `trial number` behavior series, taking the value at the trial’s first sample.

ii.
```python
trial_number_series = behavior["trial number"]["data"][:].astype(np.float32)
np.full(trial_times.shape, trial_number_series[start], dtype=np.float32)
```

iii. The trajectory shows the agent inspected the encoding but gives no specific justification for preferring it over the loop index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transformation is performed; the start-sample scalar is repeated over all timepoints.

ii.
```python
np.full(trial_times.shape, trial_number_series[start], dtype=np.float32)
```

iii. The agent treated trial number as an already encoded per-trial covariate.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward` event timestamps and trial start/stop timestamps.

ii.
```python
reward_timestamps = behavior["Reward"]["timestamps"][:].astype(np.float64)
reward_outcomes = reward_outcome_per_trial(reward_timestamps, timestamps[trial_start_idx], timestamps[trial_stop_idx])
```

iii. The agent interpreted any reward event inside a trial interval as a rewarded outcome.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Each trial is labeled 1 if a reward timestamp falls in `[start, stop)`, outcomes are shifted by one trial with the first set to 0, and the result is repeated over time. Filtering does not change which physical trial is considered previous.

ii.
```python
prev_reward_outcomes = np.concatenate([[0], reward_outcomes[:-1]]).astype(np.float32)
np.full(trial_times.shape, prev_reward_outcomes[trial_idx], dtype=np.float32)
```

iii. This directly follows the binary omitted/rewarded specification and preserves original trial history even when a bad current/previous trial is omitted from the saved list.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw `position` plus reward-zone A/B/C inferred from the NWB identifier’s scene name and a fixed switch at trial 30; fixed bounds are A=80–130, B=200–250, C=320–370 cm.

ii.
```python
ZONE_TO_BOUNDS_CM = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}
label = trial_labels[trial_idx]
zone_bounds = ZONE_TO_BOUNDS_CM[label.zone]
```

iii. The agent inspected reward-zone streams and scene coding, then chose protocol/identifier labels. This avoids noisy raw-zone inference but assumes all switches happen exactly at trial 30.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position is clipped to 0–450 cm. Signed distance is position minus the near edge before the zone, zero inside it, and position minus the far edge after it; this is then categorized.

ii.
```python
signed_distance = np.where(position_cm < zone_start, position_cm - zone_start,
    np.where(position_cm > zone_stop, position_cm - zone_stop, 0.0))
```

iii. The calculation follows the paper’s reward-relative coordinate definition and the task’s zero-inside-zone requirement.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit masks implement seven bins. Notably, -10 belongs to bin 1, +10 to bin 4, and +50 to bin 5, matching the literal inclusive boundaries in the task.

ii.
```python
bins[signed_distance < -50.0] = 0
bins[(signed_distance >= -50.0) & (signed_distance <= -10.0)] = 1
bins[(signed_distance > -10.0) & (signed_distance < 0.0)] = 2
bins[signed_distance == 0.0] = 3
bins[(signed_distance > 0.0) & (signed_distance <= 10.0)] = 4
bins[(signed_distance > 10.0) & (signed_distance <= 50.0)] = 5
bins[signed_distance > 50.0] = 6
```

iii. The agent encoded the instruction’s inequalities directly rather than relying on `np.digitize` edge conventions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural events use the same trial slice and therefore the same number/order of samples.

ii.
```python
neural_trial = kept_events[:, start:stop]
trial_pos = position[start:stop]
```

iii. The agent relied on common NWB sample indexing and validated converted dimensions.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from the raw `position` behavior series.

ii.
```python
position = behavior["position"]["data"][:].astype(np.float32)
```

iii. This is the direct corridor-position measurement.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Trial position is clipped to 0–450 cm, then clipped again below 450 for safe floor division by 90.

ii.
```python
trial_pos = np.clip(position[start:stop], 0.0, 450.0)
clipped = np.clip(position_cm, 0.0, 449.999999)
bins = np.floor(clipped / 90.0).astype(np.int8)
```

iii. The agent used the stated 450 cm track extent and prevented marginal out-of-range samples from creating invalid classes.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Floor division into 90 cm intervals yields classes 0–4, with exact 90/180/270/360 boundaries entering the higher bin.

ii.
```python
bins = np.floor(clipped / 90.0).astype(np.int8)
bins[bins > 4] = 4
```

iii. Five equal bins across 450 cm directly implement the instructions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is sliced with exactly the neural trial’s `[start:stop)` indices.

ii.
```python
trial_pos = position[start:stop]
```

iii. The common sample index provides direct alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from the raw `lick` behavior data.

ii.
```python
lick_counts = behavior["lick"]["data"][:].astype(np.float32)
```

iii. The agent inspected lick coding and also used it for the paper’s sensor-error QC.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Any positive count is 1; zero/nonpositive is 0.

ii.
```python
trial_lick = (lick_counts[start:stop] > 0).astype(np.int8)
```

iii. This implements the required binary no/yes output while retaining all valid lick events.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It uses the same `[start:stop)` sample slice as neural activity.

ii.
```python
trial_lick = (lick_counts[start:stop] > 0).astype(np.int8)
```

iii. The common behavior/neural indexing supplies alignment without interpolation.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is inferred from the scene string in the NWB `identifier`, session protocol, and trial index—not the raw `reward_zone` series.

ii.
```python
parts = identifier.strip("/").split("/")
scene = parts[-1]
trial_labels = parse_scene(session_meta.scene, len(trial_start_idx))
```

iii. The agent’s bulk inspection identified stable scene naming and used the raw environment stream as a partial consistency check.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Regexes parse constant-location, location-switch, and environment-switch scenes. Switch sessions use the first label before trial 30 and the second afterward; A/B/C maps to 0/1/2 and is repeated over time.

ii.
```python
zone=zone0 if trial_idx < switch_trial_count else zone1
zone_idx = ZONE_TO_INDEX[label.zone]
np.full(trial_times.shape, zone_idx, dtype=np.int8)
```

iii. The agent used the known 30-trial switch protocol to obtain clean labels, trading away the reference’s data-driven Viterbi handling of missing/noisy zone samples.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward` event timestamps and the position timestamp clock at trial boundaries.

ii.
```python
reward_timestamps = behavior["Reward"]["timestamps"][:].astype(np.float64)
```

iii. The agent treated reward delivery events as the authoritative outcome source.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is 1 if any reward timestamp is at or after its start and before its stop, otherwise 0; the scalar is repeated for all samples.

ii.
```python
while probe_idx < len(reward_timestamps) and reward_timestamps[probe_idx] < stop_time:
    outcomes[trial_idx] = 1
...
np.full(trial_times.shape, reward_outcomes[trial_idx], dtype=np.int8)
```

iii. This directly realizes per-trial binary reward/no-reward without needing to snap events to frames.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. All behavior and imaging streams are truncated to their shared minimum session length. Nonfinite/nonpositive timestamp differences are ignored for rate inference; NaN-aware smoothing is used; zero-variance cell correlations become NaN and are retained. Environment disagreement and absence of retained neurons raise errors. There is no explicit assertion pairing start/stop counts, and no reference-style reward timing tolerance check.

ii.
```python
session_len = min(len(position), ..., *plane_lengths)
diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
if env_mode != label.env:
    raise RuntimeError(...)
```

iii. The trajectory emphasizes robust bulk scanning and memory-safe processing. The agent chose silent common-length truncation and targeted runtime checks based on issues found during inspection.

## 13-a. What are the most time-consuming steps of the code?

i. Reading full fluorescence/neuropil blocks, trialwise baseline filtering, and OASIS deconvolution dominate. Full-dataset pickle writing and optional decoder validation are also substantial.

ii.
```python
fluorescence = fluorescence_ds[:session_len, block_local].T.astype(np.float32)
dff, events = compute_dff_and_events(...)
events[:, sl] = dcnv.oasis(...)
```

iii. The trajectory shows the full conversion ran in the background and was monitored; the code’s block design specifically addresses the large neural arrays.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-cell correlation list, per-trial lick-QC and reward-outcome loops, repeated trial slicing, and some per-trial baseline operations could be vectorized or batched. Variable trial lengths make complete vectorization awkward.

ii.
```python
block_corr = np.array([np.corrcoef(trace, speed_valid)[0, 1] ... for trace in dff_valid])
for trial_idx, (start, stop) in enumerate(zip(start_idx, stop_idx)):
```

iii. The agent prioritized bounded memory and clarity over maximum vectorization; cell blocks were the main scalability optimization.

## 13-c. What processing does the code repeat multiple times?

i. Trial intervals are traversed for fluorescence masking, neuropil correction/baselines, smoothing/deconvolution, lick QC, reward assignment, neural extraction, and construction of each behavioral output. Every neuron block also rebuilds the same trial masks/baseline windows.

ii.
```python
for start, stop in zip(start_idx, stop_idx):  # repeated in compute_dff_and_events
...
for out_idx, trial_idx in enumerate(kept_trial_indices):
```

iii. Blocking keeps peak memory low but deliberately repeats trial-window bookkeeping for every cell block.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads `planeIdx` but never uses it, computes/stores extensive conversion statistics and interneuron flags only for summaries, validates the raw environment stream although labels come from scenes, and builds an optional sample dataset unrelated to the required full output.

ii.
```python
plane_idx = imgseg["planeIdx"][:].astype(np.int16)
interneuron_flags.append(block_interneuron)
env_in_trial = env_timeseries[start:stop]
sample = make_sample_dataset(...)
```

iii. These operations support validation, diagnostics, and a representative sample; they are not used by downstream decoder features in the required full dataset.
