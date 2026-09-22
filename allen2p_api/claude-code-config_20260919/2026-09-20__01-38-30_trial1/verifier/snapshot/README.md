# Visual Behavior 2P → neural-decoder dataset

`converted_data.pkl` holds the Allen Brain Observatory **Visual Behavior 2-photon** dataset
reformatted for the decoder in `train_decoder.py`: trial-segmented, 250 ms-binned calcium imaging
together with five categorical task/behaviour variables to decode.

## Dataset description

Head-fixed mice perform a go/no-go **change-detection task** while 2-photon calcium imaging is
performed in visual cortex (VISp / VISl). Natural images are flashed for 250 ms every 750 ms
(500 ms gray inter-stimulus interval; ~4 % of flashes are omitted). The mouse earns water by licking
within 150–750 ms of a change in image identity.

- **go** trial: the image identity changes → `hit` (licked) or `miss` (did not lick)
- **catch** trial: a *sham* change (the same image repeats) → `false_alarm` (licked) or
  `correct_reject` (did not lick)
- **aborted** (licked before the change) and **auto-rewarded** (free reward) trials are excluded,
  as specified by the decoder task.

Source: AllenSDK cache `visual-behavior-ophys-1.1.0`, read exclusively through
`VisualBehaviorOphysProjectCache` (no NWB file is opened directly).

## Key statistics

| | |
|---|---|
| Sessions | 171 (one *ophys session*; simultaneously recorded imaging planes are merged) |
| Mice | 38 |
| Neurons | 29,168 (mean 171 / session, range 6–666) — VISp 29,006, VISl 162 |
| Trials | 43,966 (mean 257 / session, range 39–409) |
| Timepoints per trial | 21 bins of 250 ms, spanning −2.25 s … +3.0 s around the image change |
| Neural signal | mean dF/F per bin (Allen pipeline `dff_traces`) |
| Decoder inputs | none (`input_names == []`, arrays of shape `(0, 21)`) |
| go / catch | 87.5 % / 12.5 % (matches the technical whitepaper) |
| Outcomes | hit 31.7 %, miss 55.8 %, false alarm 1.9 %, correct reject 10.6 % |

## Loading and using the data

```python
import pickle
import numpy as np

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

session, trial = 0, 0
X = data['neural'][session][trial]      # (n_neurons, 21) float32, mean dF/F per 250 ms bin
U = data['input'][session][trial]       # (0, 21) float32 — this task has no decoder inputs
Y = data['output'][session][trial]      # (5, 21) int64 — the five variables to decode

t = data['metadata']['off_start'] + 0.25 * (np.arange(21) + 0.5)   # bin centres re: the change

# which mouse / which brain region is each neuron in
mouse = data['subjects'][data['subject_idx'][session]]
regions = [data['brain_regions'][i] for i in data['brain_region_idx'][session]]

# decode a label back to its name
name = data['output_values'][0][Y[0, 10]]      # e.g. 'im085'
```

Validate and train the reference decoder:

```bash
python train_decoder.py converted_data.pkl --verify-only     # structure + summary statistics
python train_decoder.py converted_data.pkl --plot-samples    # trains and plots predictions
```

Rebuild the dataset from the AllenSDK cache:

```bash
python -u convert_data.py converted_data.pkl --full                      # ~70 s with 16 workers
python -u convert_data.py sample_data.pkl --sample --show-processing     # 2 sessions + check plots
```

## Output format specification

`data` is a dict with the keys `neural`, `input`, `output`, `subjects`, `subject_idx`,
`brain_regions`, `brain_region_idx`, `input_names`, `output_names`, `output_values`, `metadata`.
`neural`, `input` and `output` are lists over sessions of lists over trials.

| # | `output_names` | type | values (`output_values`) |
|---|---|---|---|
| 0 | `image_identity` | time-varying | 16 image names (`im000` … `im106`) + `omitted`; the image presented in the 750 ms flash interval containing the bin |
| 1 | `image_change` | time-varying | `no_change`, `change`; 1 in the 500 ms after a change of image identity (go trials only) |
| 2 | `running_speed_bin` | time-varying | `0-20%` … `80-100%`; quintiles of the binned running speed, computed per session |
| 3 | `pupil_diameter_bin` | time-varying | `0-20%` … `80-100%`; quintiles of the binned pupil diameter (`2*sqrt(pupil_area/pi)`, blinks interpolated), per session |
| 4 | `trial_outcome` | constant within a trial | `hit`, `miss`, `false_alarm`, `correct_reject` |

`metadata` records `time_bin_size` (250 ms), `temporal_alignment_event` (the stimulus change time),
`off_start` (−2.25 s), `off_end` (+3.0 s), how each stream was selected and processed, and a
`session_info` list with, per session, the ophys session / experiment ids, mouse, session type,
cre line, imaging depths, frame rate, neuron and trial counts, and the running-speed and
pupil-diameter quintile edges in physical units (cm/s and camera pixels).

## Decoder performance (reference decoder, held-out trials)

| Output | Validation balanced accuracy | Chance |
|---|---|---|
| image_identity (17 classes) | 0.441 | 0.059 |
| image_change (2) | 0.703 | 0.500 |
| running_speed_bin (5) | 0.309 | 0.200 |
| pupil_diameter_bin (5) | 0.280 | 0.200 |
| trial_outcome (4) | 0.307 | 0.250 |

## Files

- `convert_data.py` — the conversion script
- `converted_data.pkl` / `sample_data.pkl` — full and 2-session datasets
- `CONVERSION_NOTES.md` — every decision, check and validation result
- `conversion_*_out.txt`, `verification_*_out.txt`, `train_decoder_*_out.txt` — run logs
- `processing_<session_id>.png` — per-step verification plots for two sessions
- `cache/` — analysis and sanity-check scripts (see `cache/README_CACHE.md`)
