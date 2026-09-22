# Allen Visual Behavior 2P → neural-decoder dataset

`/app/converted_data.pkl` (8.7 GB) holds the Allen Brain Observatory **Visual Behavior 2P**
dataset reshaped for the decoder in `/app/train_decoder.py`.

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full          # rebuild (~50 s)
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only  # validate
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples # train + score
```

Every decision made during the conversion, together with its justification and the checks
that back it, is recorded in `CONVERSION_NOTES.md`.

## Dataset

Head-fixed mice perform a go/no-go **visual change-detection task** while 2-photon calcium
imaging is performed in primary visual cortex (VISp). A stream of natural images is flashed
(250 ms image / 500 ms gray screen; 5 % of eligible flashes omitted) and the mouse earns a
water reward by licking 150–750 ms after the image identity changes. Source: AllenSDK local
cache `visual-behavior-ophys-1.1.0`, read exclusively through
`VisualBehaviorOphysProjectCache`.

**Selection** — locally available experiments with `project_code == "VisualBehavior"`
(single-plane Scientifica rigs, constant 30.95 Hz frame rate, VISp) and `passive == False`
(active behaviour). 3 sessions with an empty `eye_tracking` table are dropped.

**Trials** — the experiment's own trial table, keeping `go | catch` and excluding
`aborted` and `auto_rewarded`. A trial spans `[trials.start_time, trials.stop_time)`.

## Key statistics

| | |
|---|---|
| sessions | 165 |
| mice | 37 (1–9 sessions each) |
| brain region | VISp |
| neurons | 28 821 (mean 175/session, range 6–666) |
| trials | 42 470 (mean 257/session, range 39–409) |
| timepoints | 11 192 974 |
| time bin | 32.319 ms (one 2-photon frame; sd 0.006 ms across sessions) |
| trial length | 217–389 bins (7.0–12.6 s), mean 264 |
| neural signal | detrended dF/F (`dff_traces`) |
| decoder inputs | none (`d_input = 0`) |
| decoder outputs | 5 categorical variables |

Output distributions over all 11.2 M timepoints:

| output | classes | distribution |
|---|---|---|
| `image_identity` | 16 image names | 0.060–0.065 each (8 per session) |
| `image_change` | `no_change`, `change` | 0.923 / 0.077 |
| `running_speed_bin` | `Q1_lowest`…`Q5_highest` | 0.200 × 5 |
| `pupil_diameter_bin` | `Q1_lowest`…`Q5_highest` | 0.200 × 5 |
| `trial_outcome` | hit, miss, false_alarm, correct_reject | 0.316 / 0.559 / 0.018 / 0.107 |

## Format

```python
import pickle
data = pickle.load(open('/app/converted_data.pkl', 'rb'))

data['neural'][session][trial]   # float32 (n_neurons, T)   dF/F, one column per 2p frame
data['input'][session][trial]    # float32 (0, T)           no decoder inputs for this task
data['output'][session][trial]   # int64   (5, T)           the five categorical outputs
data['subjects']                 # list[str], 37 mouse ids
data['subject_idx']              # int64 (165,)  index into subjects, one per session
data['brain_regions']            # ['VISp']
data['brain_region_idx'][session]# int64 (n_neurons,)
data['input_names']              # []
data['output_names']             # ['image_identity', 'image_change',
                                 #  'running_speed_bin', 'pupil_diameter_bin',
                                 #  'trial_outcome']
data['output_values'][i]         # names of the classes of output i
data['metadata']                 # see below
```

### Output specification

| # | name | definition |
|---|---|---|
| 0 | `image_identity` | which of the 16 natural images is currently being shown. Value is constant over the 750 ms image-presentation interval (flash + following gray screen) and is carried through omitted flashes, so the gray screen keeps the identity of the image the mouse is currently seeing repeated. |
| 1 | `image_change` | 1 for every bin of the image-presentation interval that begins with a change in image identity, 0 elsewhere. Go trials contain exactly one such interval; catch (sham-change) trials contain none. |
| 2 | `running_speed_bin` | running speed (cm/s, 60 Hz encoder trace linearly interpolated onto the ophys frame times) discretised into 5 equal-percentile (quintile) bins computed **per session**. |
| 3 | `pupil_diameter_bin` | pupil diameter `2·max(pupil_width, pupil_height)` in camera pixels, blink/outlier frames removed and interpolated, resampled onto the ophys frame times, discretised into 5 per-session quintile bins. |
| 4 | `trial_outcome` | hit / miss / false_alarm / correct_reject, constant within a trial (broadcast across its bins so that all trials share one output dimension). |

### Metadata

`data['metadata']` carries `task_description`, `time_bin_size` (ms), `temporal_alignment_event`,
`off_start` (0.0, trial start), `off_end` (`None`, trials have variable length), the trial and
selection definitions, aggregate counts, and:

- `session_info`: one dict per session with `ophys_experiment_id`, `ophys_session_id`,
  `behavior_session_id`, `mouse_id`, `cre_line`, `targeted_structure`, `imaging_depth`,
  `session_type`, `equipment_name`, neuron/trial counts, the per-session quintile edges for
  running speed and pupil diameter, the pupil blink fraction and the outcome fractions.
- `skipped_sessions`: the sessions that were selected but could not be converted, with the reason.

### Temporal alignment

All streams in an NWB file already share one session clock (100 kHz sync board). The
**ophys frame timestamps** are the time base: dF/F is used as recorded, running speed and pupil
diameter are linearly interpolated onto those frame times, and the stimulus/trial variables are
evaluated at them. Bin *k* of a trial is the ophys frame at `ophys_timestamps[a + k]` where
`a` is the first frame at or after `trials.start_time`.

## Decoder performance (validation, balanced accuracy)

| output | chance | validation |
|---|---|---|
| `image_identity` | 0.0625 | 0.418 |
| `image_change` | 0.500 | 0.609 |
| `running_speed_bin` | 0.200 | 0.282 |
| `pupil_diameter_bin` | 0.200 | 0.267 |
| `trial_outcome` | 0.250 | 0.296 |

Every output is above chance. See `CONVERSION_NOTES.md` §11–12 for the comparison with the
accuracies reported in the reference paper.

## Files

| file | contents |
|---|---|
| `convert_data.py` | the conversion script |
| `converted_data.pkl` | full converted dataset (165 sessions) |
| `sample_data.pkl` | 2-session sample |
| `CONVERSION_NOTES.md` | decisions, rationale, validation, checks |
| `conversion_{sample,full}_out.txt` | conversion logs |
| `verification_{sample,full}_out.txt` | `--verify-only` output |
| `train_decoder_{sample,full}_out.txt` | decoder training logs |
| `processing_<eid>.png` | per-step conversion diagnostics |
| `sample_trials.png`, `predictions.png` | decoder plots |
| `cache/` | exploration/analysis scripts and their outputs (see `cache/README_CACHE.md`) |
