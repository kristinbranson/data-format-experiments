# Visual Behavior 2P -> neural-decoder dataset

Converted from the **Allen Brain Observatory Visual Behavior 2P** release
(`visual-behavior-ophys-1.1.0`, read exclusively through the AllenSDK
`VisualBehaviorOphysProjectCache`) into the decoder-benchmark pickle format.

## Dataset description

Head-fixed mice perform a visual **change-detection** task while 2-photon calcium
imaging is performed in visual cortex (V1 = VISp, LM = VISl).  A series of natural
images is flashed (250 ms image, 500 ms gray, i.e. a 750 ms cycle, 5 % of repeats
omitted); the mouse earns water by licking when the image identity changes.
Trials are **go** (real change), **catch** (sham change), *aborted* (premature lick)
and *auto-rewarded*; only **go and catch** trials are kept here.

| Statistic | Value |
|---|---|
| Sessions | 171 (active behaviour + imaging sessions) |
| Subjects (mice) | 38 (2-9 sessions each) |
| Neurons | 29,168 (mean 171/session, range 6-666) |
| Trials | 43,975 (mean 257/session, range 39-409) |
| Timepoints per trial | 20 bins of 250 ms, -2 s .. +3 s around the change |
| Brain regions | VISp (29,006 neurons), VISl (162 neurons) |
| Neural signal | pipeline dF/F, averaged within each 250 ms bin |
| go / catch | 38,460 / 5,515 (catch fraction 0.125) |
| Outcomes | hit 31.7 %, miss 55.8 %, false alarm 1.9 %, correct reject 10.6 % |

## How to load

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][session][trial]   # (n_neurons, 20) float32, dF/F bin means
inputs = data['input'][session][trial]    # (0, 20) - this task has no decoder inputs
output = data['output'][session][trial]   # (5, 20) int64 class indices
```

Train/evaluate the reference decoder:

```bash
python /app/train_decoder.py /app/converted_data.pkl            # train
python /app/train_decoder.py /app/converted_data.pkl --verify-only
```

Re-create the dataset:

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
```
Useful options: `--neural-signal {dff,events,filtered_events}`,
`--quantile-scope {session,global}`, `--change-window {bin,interval}`,
`--off-start/--off-end`, `--subset N`, `--workers N`.

## Output format specification

| idx | `output_names` | classes (`output_values`) | description |
|---|---|---|---|
| 0 | `image_identity` | 16 image names (`im000` ... `im106`) | identity of the image whose 750 ms presentation interval contains the bin; omitted flashes inherit the previous image |
| 1 | `image_change` | `no_change`, `change` | 1 in the single 250 ms bin that starts with an image change (go trials only) |
| 2 | `running_speed_quintile` | `q1` ... `q5` | mean running speed per bin, per-session equal-percentile bins |
| 3 | `pupil_diameter_quintile` | `q1` ... `q5` | mean pupil diameter (2*max(width,height), blinks interpolated) per bin, per-session equal-percentile bins |
| 4 | `trial_outcome` | `hit`, `miss`, `false_alarm`, `correct_reject` | static per trial, broadcast over the 20 bins |

All outputs are time-varying arrays of shape `(5, 20)`; `input` is empty
(`(0, 20)`) because the decoder task specifies no inputs.

`data['metadata']` records the bin size (250 ms), the alignment event
(`trials.change_time`), `off_start`/`off_end` (-2 / +3 s), the definitions above,
the quantile edges, and a `session_info` list with the ophys/behaviour ids,
mouse, cre line, session type, equipment and per-session neuron/trial counts.

## Decoder accuracy (reference decoder, validation balanced accuracy)

| Output | Chance | Accuracy |
|---|---|---|
| image_identity | 0.063 | 0.449 |
| image_change | 0.500 | 0.693 |
| running_speed_quintile | 0.200 | 0.311 |
| pupil_diameter_quintile | 0.200 | 0.280 |
| trial_outcome | 0.250 | 0.310 |

## Files

| File | Content |
|---|---|
| `convert_data.py` | the conversion script |
| `converted_data.pkl` | full dataset (674 MB) |
| `sample_data.pkl` | 2-session sample |
| `CONVERSION_NOTES.md` | every decision, check and validation result |
| `sanity_checks.py`, `sanity_multiplane.py` | independent re-derivation of the data from the SDK |
| `analyze_accuracy.py` | per-class recall / confusion analysis |
| `processing_*.png` | per-step processing figures |
| `conversion_*_out.txt`, `verification_*_out.txt`, `train_decoder_*_out.txt` | run logs |
