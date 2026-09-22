# Visual Behavior 2P -> neural-decoder dataset

Converted from the **Allen Brain Observatory: Visual Behavior 2P** release
(`visual-behavior-ophys` v1.1.0, local cache in `/app/data`) into the decoder
format described in the task specification.

## Dataset description

Head-fixed mice perform a go/no-go **visual change-detection task** while
2-photon calcium imaging is performed in visual cortex (VISp, VISl).  Natural
images are flashed for 250 ms every 750 ms; the mouse earns water by licking
when the image identity changes.  5% of flashes are omitted.  Trials are
**go** (a real change) or **catch** (a sham change); trials in which the mouse
licked before the change (*aborted*) and free-reward (*auto-rewarded*) trials
are excluded.

| Statistic | Value |
|---|---|
| Sessions | 171 (active/behaving ophys sessions) |
| Subjects | 38 mice |
| Neurons | 29,168 (mean 170.6 per session, range 6-666) |
| Trials | 43,975 (mean 257 per session, range 39-409) |
| go / catch | 38,460 / 5,515 (catch fraction 12.5%) |
| hit / miss / FA / CR | 13,940 / 24,520 / 834 / 4,681 (hit rate 0.36, FA rate 0.15) |
| Brain regions | VISp (29,006 neurons), VISl (162) |
| Neural signal | dF/F (Allen pipeline), mean per 100 ms bin |
| Time bins | 60 per trial (100 ms), window [-2, +4] s around the image change |
| Alignment event | `trials.change_time` (sham change time on catch trials) |

## How to load

```python
import pickle, numpy as np
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][0][0]      # session 0, trial 0 -> (n_neurons, 60)
output = data['output'][0][0]      # (5, 60) integer class labels
print(data['output_names'])        # ['image_identity', 'image_change',
                                   #  'running_speed_bin', 'pupil_diameter_bin',
                                   #  'trial_outcome']
print(data['output_values'][0])    # image class names, 'omitted' last
print(data['metadata']['session_info'][0])   # per-session provenance
```

## Format

| Key | Contents |
|---|---|
| `neural` | list of sessions -> list of trials -> `(n_neurons, 60)` float32 mean dF/F per 100 ms bin |
| `input` | list of sessions -> list of trials -> `(0, 60)` float32 (this task has **no decoder inputs**) |
| `output` | list of sessions -> list of trials -> `(5, 60)` int64 class labels |
| `subjects` / `subject_idx` | 38 mouse ids; index of the mouse for each session |
| `brain_regions` / `brain_region_idx` | `['VISl', 'VISp']`; region index for each neuron of each session |
| `input_names` | `[]` |
| `output_names` | `image_identity`, `image_change`, `running_speed_bin`, `pupil_diameter_bin`, `trial_outcome` |
| `output_values` | class-name lists for each output |
| `metadata` | task description, `time_bin_size` (100.0 ms), `temporal_alignment_event`, `off_start` (-2.0), `off_end` (4.0), neural-signal description, curation rules, and `session_info` (per-session ids, mouse, session type, cre line, experience level, rig, neuron/trial counts, outcome counts, the running/pupil quintile thresholds and the blink fraction) |

### Outputs

0. **image_identity** - 17 classes: the 16 natural images of image sets A and B
   plus `omitted`.  The label is the image of the 750 ms image-presentation
   interval containing the bin (i.e. the image currently or most recently
   flashed), so it is defined during the grey inter-stimulus period as well.
1. **image_change** - 1 during the 750 ms interval of the change flash of a go
   trial (8 bins starting at t = 0), 0 otherwise; always 0 on catch trials.
2. **running_speed_bin** - quintile (0-4) of the binned running speed, computed
   **within each session** (speeds are not comparable across sessions/rigs).
3. **pupil_diameter_bin** - quintile (0-4) of the binned pupil diameter
   (`2*sqrt(pupil_area/pi)`), blink gaps linearly interpolated, per-session
   quintiles.
4. **trial_outcome** - `hit`, `miss`, `false_alarm`, `correct_reject`; one value
   per trial, broadcast over the 60 bins.

## Reproducing

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full          # ~70 s, 16 workers
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```
`--neural-signal {dff,events,filtered_events}` selects the neural stream
(default `dff`; see CONVERSION_NOTES.md Step 8 for the empirical comparison).

## Decoder performance (reference decoder, held-out validation)

| Output | Balanced accuracy | Chance |
|---|---|---|
| image_identity | 0.421 | 0.059 |
| image_change | 0.620 | 0.500 |
| running_speed_bin | 0.296 | 0.200 |
| pupil_diameter_bin | 0.275 | 0.200 |
| trial_outcome | 0.306 | 0.250 |

See `CONVERSION_NOTES.md` for every processing decision, all validation checks
and the comparison with the reference papers.
