# Visual Behavior 2P — decoder-ready dataset

`converted_data.pkl` holds the Allen Brain Observatory **Visual Behavior 2P** dataset
reformatted for neural decoding: 2-photon calcium activity in mouse primary visual
cortex (V1) recorded while head-fixed mice performed a go/no-go visual change-detection
task, segmented into the task's own trials and paired with five categorical variables to
decode.

---

## 1. The experiment in one paragraph

Natural images are flashed for 250 ms every 750 ms (500 ms grey inter-stimulus interval);
5 % of flashes are omitted. Occasionally the image identity changes, and the mouse earns
water by licking within 150–750 ms of the change. Each trial is either a **GO** trial (the
image really changes → `hit` or `miss`) or a **CATCH** trial (a sham change, the same image
is re-presented → `false_alarm` or `correct_reject`). Trials in which the mouse licked
before the change (**aborted**) and the free-reward (**auto-rewarded**) trials are excluded,
as specified for this decoding task. dF/F is imaged at 30.94 Hz; running speed (60 Hz) and
eye tracking (30 Hz) are recorded simultaneously on the same hardware clock.

Sources: `whitepaper.pdf` (Allen Brain Observatory: Visual Behavior 2P Technical
Whitepaper) and `paper.pdf` (Behavioral strategy shapes activation of the Vip-Sst
disinhibitory circuit in visual cortex).

## 2. Key statistics

| | |
|---|---|
| Sessions | **165** (single-plane `VisualBehavior` project, active behaviour, eye tracking present) |
| Mice | **37** (2–9 sessions each) |
| Neurons | **28,821** (6–666 per session, mean 174.7) |
| Brain region | VISp (mouse V1) — the only region targeted by this project |
| Cre lines | Slc17a7 (excitatory), Sst, Vip — see `metadata['session_info'][i]['cre_line']` |
| Trials | **42,410** go/catch trials (39–409 per session, mean 257) |
| Timepoints | 11,177,298 (217–389 per trial, mean 262) |
| Time bin | **32.32 ms** (one 2-photon frame, 30.9406 Hz), identical for every trial and session |
| Trial window | `[trials.start_time, trials.stop_time)` — 7.26–12.56 s, the experiment's own trial |
| Neural signal | released dF/F (`dff_traces.dff`), float32 |
| File size | 8.4 GB |

Distribution of each decoded variable over all timepoints:

| Output | Classes | Distribution |
|---|---|---|
| `image_identity` | 17 | grey 0.669; each of the 16 images ≈0.020–0.021 |
| `image_change` | 2 | no_change 0.974, change 0.026 |
| `running_speed_bin` | 5 | 0.200 each (per-session quintiles) |
| `pupil_diameter_bin` | 5 | 0.200 each (per-session quintiles) |
| `trial_outcome` | 4 | hit 0.316, miss 0.559, false_alarm 0.018, correct_reject 0.107 |

## 3. Loading and using the data

```python
import pickle, numpy as np

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

session, trial = 0, 5
X = data['neural'][session][trial]     # (n_neurons, T)  float32 dF/F
Y = data['output'][session][trial]     # (5, T)          int16 class labels
U = data['input'][session][trial]      # (0, T)          no decoder inputs for this task

# what output row 0 means at timepoint 100
name  = data['output_names'][0]                          # 'image_identity'
value = data['output_values'][0][Y[0, 100]]              # e.g. 'im065' or 'grey'

# which mouse / brain region
mouse  = data['subjects'][data['subject_idx'][session]]
region = [data['brain_regions'][i] for i in data['brain_region_idx'][session]]  # per neuron

# time axis of a trial, in seconds from the start of the trial
t = np.arange(X.shape[1]) * data['metadata']['time_bin_size'] / 1000
```

Train and evaluate the reference decoder:

```bash
python train_decoder.py /app/converted_data.pkl              # train + validate
python train_decoder.py /app/converted_data.pkl --verify-only  # format check + summary
```

Rebuild the dataset from the NWB files (~65 s on 16 cores):

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
```

## 4. Format specification

```python
data = {
  'neural':  [ [ (n_neurons, T) float32, ... ],  ... ],   # per session, per trial: dF/F
  'input':   [ [ (0, T)        float32, ... ],  ... ],    # no decoder inputs
  'output':  [ [ (5, T)        int16,   ... ],  ... ],    # 5 categorical variables
  'subjects':          list[str],        # 37 mouse ids
  'subject_idx':       (165,) int64,     # index into 'subjects' for each session
  'brain_regions':     ['VISp'],
  'brain_region_idx':  [ (n_neurons,) int64, ... ],   # index into 'brain_regions' per neuron
  'input_names':       [],
  'output_names':      ['image_identity', 'image_change', 'running_speed_bin',
                        'pupil_diameter_bin', 'trial_outcome'],
  'output_values':     [ [...17...], [...2...], [...5...], [...5...], [...4...] ],
  'metadata':          {...},
}
```

### The five outputs

| # | Name | Classes | Definition |
|---|------|---------|------------|
| 0 | `image_identity` | `grey`, `im000`, `im031`, `im035`, `im045`, `im054`, `im061`, `im062`, `im063`, `im065`, `im066`, `im069`, `im073`, `im075`, `im077`, `im085`, `im106` | the natural image on the screen at that 32 ms frame; `grey` during the 500 ms inter-stimulus interval and during omitted flashes (the screen stays grey). A session uses one 8-image set (A or B); the label space is global so images are comparable across sessions. |
| 1 | `image_change` | `no_change`, `change` | 1 on the frames of the 250 ms flash at which the image identity changed (`stimulus_presentations.is_change`). Catch trials present a *sham* change (the same image) and are therefore 0. |
| 2 | `running_speed_bin` | `running_quintile_0…4` | SDK running speed (cm/s, already 10 Hz low-passed) linearly interpolated onto the ophys frames, then cut at the **session's own** 20/40/60/80th percentiles of the timepoints in the dataset. |
| 3 | `pupil_diameter_bin` | `pupil_quintile_0…4` | effective pupil diameter `2·sqrt(pupil_area/π)` in camera pixels, blinks removed and linearly interpolated, resampled onto the ophys frames, then cut at the **session's own** quintiles. Per-session bins are essential here: pupil size is in camera pixels and is not comparable between rigs/mice. |
| 4 | `trial_outcome` | `hit`, `miss`, `false_alarm`, `correct_reject` | one label per trial from the `trials` table, repeated across the trial's timepoints so that all five outputs share one `(5, T)` array. |

### `metadata` keys

`dataset`, `project_code`, `task_description`, `time_bin_size` (ms),
`time_bin_size_range_ms`, `temporal_alignment_event`, `off_start` (0.0),
`off_end` (`None` — trials keep their natural variable length),
`trial_duration_s_note`, `neural_signal`, `neural_signal_description`,
`sampling_rate_hz`, `n_sessions`, `n_subjects`, `n_trials`, `n_neurons`,
`n_timepoints`, `running_speed_units`, `pupil_units`, `exclusions`,
`skipped_sessions`, and `session_info` — a list with one dict per session holding
`ophys_experiment_id`, `ophys_session_id`, `behavior_session_id`, `mouse_id`,
`cre_line`, `session_type`, `targeted_structure`, `imaging_depth`, `equipment_name`,
`frame_period_s`, `cell_specimen_ids`, `trial_ids`, trial counts, the quintile edges
actually used, and the blink fraction. `cell_specimen_ids` and `trial_ids` make every
row and every trial traceable back to the original NWB file.

## 5. What was included and excluded

Included: `project_code == 'VisualBehavior'` (the complete single-plane project present
on disk), active behaviour sessions of all experience levels (familiar and novel images),
all `valid_roi` neurons, all go and catch trials.

Excluded, with reasons:
- **Multiscope experiments** (45 files, 1 mouse): imaged at 10.73 Hz instead of 30.94 Hz,
  which would break the requirement that time bins be the same size in every session.
- **Passive sessions** (71): the lick spout is retracted, so there are no hits and no
  false alarms and `trial_outcome` is degenerate; the reference paper likewise does not
  analyse them.
- **3 sessions with no eye-tracking data** (oeid 795953296, 806456687, 833631914):
  pupil diameter is a required output.
- **60 trials (0.14 %)** that contain no non-blink pupil sample; their pupil trace would
  be pure extrapolation.
- **Aborted and auto-rewarded trials**, as specified for this decoding task.

## 6. Validation

- `train_decoder.py --verify-only` reports **no errors and no warnings**.
- 95 independent checks re-read the original NWB files with **h5py directly** and
  reproduce the dF/F values (`np.allclose`), the running and pupil quintile bins, the
  image identity and change traces, the trial outcomes and the trial lengths — all exact
  (`cache/sanity_checks.py`, 0 failures).
- Whitepaper cross-checks reproduced from the converted data: catch fraction 0.1254
  (~12.5 % expected), mean change latency 4.23 s (~4.2 s expected), flash duration
  258.6 ms (250 ms expected), image on-screen fraction 0.336 (1/3 expected),
  30.94 Hz frame rate (31 Hz expected).

Reference decoder, validation balanced accuracy (165 sessions):

| Output | Chance | Validation |
|---|---|---|
| image_identity | 0.059 | **0.405** |
| image_change | 0.500 | **0.625** |
| running_speed_bin | 0.200 | **0.283** |
| pupil_diameter_bin | 0.200 | **0.266** |
| trial_outcome | 0.250 | **0.295** |

See `CONVERSION_NOTES.md` for the full decision log, every consistency check, and the
analysis of why the last four numbers are what they are.

## 7. Files

| File | Contents |
|---|---|
| `converted_data.pkl` | the full dataset (165 sessions, 8.4 GB) |
| `sample_data.pkl` | 2 sessions, for quick tests |
| `convert_data.py` | the conversion script |
| `CONVERSION_NOTES.md` | decisions, validation and review log |
| `processing_<oeid>.png` | per-step diagnostic plots (`--show-processing`) |
| `conversion_{sample,full}_out.txt` | conversion logs |
| `verification_{sample,full}_out.txt` | format-verifier output |
| `train_decoder_{sample,full}_out.txt` | decoder training logs |
| `cache/` | exploration, sanity-check and diagnostic scripts |
