# Visual Behavior 2P -> Neural Decoder Dataset

Converted from the **Allen Brain Observatory: Visual Behavior 2P** release
(`visual-behavior-ophys-1.1.0`) into the decoder format described in `train_decoder.py`.

## Dataset description

Head-fixed mice perform a go/no-go **visual change detection** task while 2-photon calcium
imaging is performed in visual cortex (VISp / VISl). Natural images are flashed for 250 ms
every 750 ms (250 ms image + 500 ms gray); the mouse earns water by licking when the image
identity changes. 5% of image repeats are omitted (changes and the flash before a change are
never omitted).

Neural activity is the **detected calcium event** magnitude (the signal used by Piet et al.),
summed within each 750 ms image presentation interval.

## Key statistics

| Statistic | Value |
|---|---|
| Sessions | 171 |
| Subjects (mice) | 38 |
| Neurons | 29,168 (VISp 29,006; VISl 162) |
| Trials | 43,975 (go 38,460; catch 5,515) |
| Trials / session | mean 257, min 39, max 409 |
| Time bins per trial | 8 (750 ms each) |
| Alignment | image change (go) / sham change (catch), `trials.change_time` |
| Window | -2.25 s to +3.0 s relative to the change (start of last bin) |
| Outcomes | hit 13,940; miss 24,520; false alarm 834; correct reject 4,681 |

## How to load

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][session][trial]   # (n_neurons, 8) float32, summed calcium events
output = data['output'][session][trial]   # (5, 8) int64, categorical labels per bin
input_ = data['input'][session][trial]    # (0, 8) -- this task has no decoder inputs
```

## Output format

| idx | name | type | values |
|---|---|---|---|
| 0 | `image_identity` | time-varying | 16 image names + `omitted` (17 classes) |
| 1 | `image_change` | time-varying | `no_change`, `change` (1 only in the bin starting at a real change) |
| 2 | `running_speed` | time-varying | `speed_q1..q5` (within-session quintiles) |
| 3 | `pupil_diameter` | time-varying | `pupil_q1..q5` (within-session quintiles) |
| 4 | `trial_outcome` | per trial (broadcast over bins) | `hit`, `miss`, `false_alarm`, `correct_reject` |

Catch trials are sham changes, so their `image_change` row is all zeros and the image does
not change at the aligned bin. That is the intended behaviour, not a bug.

Other keys: `subjects`, `subject_idx`, `brain_regions`, `brain_region_idx`, `input_names`,
`output_names`, `output_values`, `metadata` (includes a per-session `session_info` list with
session ids, experiment ids, mouse, session type, cre line and frame rate).

## Curation applied

- Only **active** behavior sessions (passive OPHYS_2/OPHYS_5 sessions have no licks or
  rewards, so trial outcome would be degenerate).
- Only **go** and **catch** trials (aborted and auto-rewarded excluded).
- Only **valid ROIs** (AllenSDK `exclude_invalid_rois=True`, the whitepaper's ROI filtering).
- 3 sessions dropped because their eye-tracking table is empty (pupil output would be NaN).
- Multiscope sessions: all simultaneously recorded imaging planes are **merged** into one
  session, concatenating neurons.

## Reproducing

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl
```

`--sample` converts 2 sessions; `--show-processing` writes per-session diagnostic figures.

## Decoder performance (validation balanced accuracy)

| Output | Accuracy | Chance |
|---|---|---|
| image_identity | 0.490 | 0.059 |
| image_change | 0.658 | 0.500 |
| running_speed | 0.316 | 0.200 |
| pupil_diameter | 0.283 | 0.200 |
| trial_outcome | 0.318 | 0.250 |

See `CONVERSION_NOTES.md` for full validation, sanity checks and design rationale.
