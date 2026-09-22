# Visual Behavior 2P -> neural-decoder dataset

`converted_data.pkl` contains the Allen Institute **Visual Behavior 2P** dataset
(release `visual-behavior-ophys-1.1.0`, the 284 experiments present in the local
AllenSDK cache at `/app/data`) reformatted for training a neural decoder.

## Dataset description

Head-fixed mice perform a go/no-go **visual change-detection** task: a continuous stream of
natural images is flashed (250 ms image, 500 ms grey inter-stimulus interval, 8 images per
session) and the mouse licks a spout when the image identity changes, earning water.
Two-photon calcium imaging (GCaMP6f) is recorded simultaneously from excitatory (Slc17a7),
Sst and Vip neurons in VISp and VISl, on a single-plane Scientifica rig (31 Hz) or a
multi-plane Mesoscope (11 Hz per plane).

Only **active behaviour sessions** (OPHYS_1/3/4/6) are included; passive sessions
(lick spout retracted, no rewards) are excluded. Trials are the **go** (real change) and
**catch** (sham change) trials; aborted and auto-rewarded trials are excluded.

## Key statistics

| Quantity | Value |
|---|---|
| Sessions (ophys sessions; simultaneous planes merged) | 171 |
| Mice | 38 |
| Neurons | 29,168 (VISp 29,006, VISl 162) |
| Trials (go + catch) | 43,975 (go fraction 0.875) |
| Time bins per trial | 24 |
| Bin size | 250 ms |
| Trial window | -3.0 s to +3.0 s around the image change |
| Outcomes | hit 13,940 / miss 24,520 / false alarm 834 / correct reject 4,681 |

## How to load

```python
import pickle, numpy as np
data = pickle.load(open('converted_data.pkl', 'rb'))

neural  = data['neural'][s][t]   # (n_neurons, 24) float32, mean dF/F per 250 ms bin
outputs = data['output'][s][t]   # (5, 24) int64 class labels
inputs  = data['input'][s][t]    # (0, 24)  -- this task has no decoder inputs
mouse   = data['subjects'][data['subject_idx'][s]]
regions = [data['brain_regions'][i] for i in data['brain_region_idx'][s]]
```

## Output format specification

| idx | `output_names` | classes (`output_values`) | description |
|---|---|---|---|
| 0 | `image_identity` | 16 image names (`im000`...`im106`) + `omitted` | the image of the 750 ms image-presentation interval containing the bin |
| 1 | `image_change` | `no_change`, `change` | 1 during the presentation of a changed image (3 bins after the change on go trials) |
| 2 | `running_speed` | `quintile_0..4` | bin-averaged running speed, discretised into 5 equal-percentile bins per session |
| 3 | `pupil_diameter` | `quintile_0..4` | bin-averaged pupil diameter (`2*sqrt(pupil_area/pi)`, blinks interpolated), 5 equal-percentile bins per session |
| 4 | `trial_outcome` | `hit`, `miss`, `false_alarm`, `correct_reject` | static per trial, repeated over the 24 bins |

`metadata` holds the task description, `time_bin_size` (250.0 ms),
`temporal_alignment_event` (the image-change time), `off_start`/`off_end` (-3.0 / +3.0 s)
and a `session_info` list with per-session provenance (ophys_session_id, experiment ids,
mouse, session type, rig, cre line, frame rate, trial/outcome counts, blink fraction,
running and pupil ranges).

## Reproducing

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full          # ~1 min, 24 workers
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```
Options: `--neural {dff,events,filtered_events}` (default `dff`), `--bin <seconds>`,
`--zscore`, `--nsample N`, `--workers N`.

## Decoder performance (held-out trials, balanced accuracy)

| Output | Accuracy | Chance |
|---|---|---|
| image_identity | 0.446 | 0.059 |
| image_change | 0.656 | 0.500 |
| running_speed | 0.307 | 0.200 |
| pupil_diameter | 0.281 | 0.200 |
| trial_outcome | 0.306 | 0.250 |

See `CONVERSION_NOTES.md` for every processing decision, validation check and comparison
with the reference papers and code.
