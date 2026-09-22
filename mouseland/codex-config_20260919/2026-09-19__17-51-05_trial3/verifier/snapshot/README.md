# Neural Decoder Dataset

This directory contains a decoder-ready conversion of the two-photon calcium-imaging dataset from *Unsupervised pretraining in biological neural networks*. The conversion follows the released processing conventions where applicable and aligns each trial to corridor entry.

## Main files

- `converted_data.pkl`: complete converted dataset (89 sessions, 38,110 trials; approximately 141 GiB).
- `sample_data.pkl`: two-session test conversion.
- `convert_data.py`: reproducible converter.
- `CONVERSION_NOTES.md`: detailed source audit, decisions, checks, statistics, and decoder results.
- `verification_full_out.txt` and `train_decoder_full_out.txt`: complete validation and training logs.
- `processing_*.png`, `sample_trials.png`, and `predictions.png`: processing/alignment and prediction checks.

## Loading

```python
import pickle

with open("/app/converted_data.pkl", "rb") as f:
    data = pickle.load(f)

trial_neural = data["neural"][0][0]  # neurons x time
trial_input = data["input"][0][0]    # 4 x time
trial_output = data["output"][0][0]  # 4 x time
```

Loading the complete pickle requires substantial RAM because it contains about 141 GiB of float32 neural arrays.

## Variables

Decoder inputs, in row order:

1. `time_to_sound_cue_s`: signed seconds until the sound cue (positive before cue).
2. `day_of_training`: per-trial training day.
3. `time_since_trial_start_s`: seconds since corridor entry.
4. `reward_available`: whether the trial's corridor identity is rewarded.

Categorical decoder outputs, in row order:

1. `visual_stimulus_category`: circle, leaf, rock, or wood.
2. `licking`: no lick or lick at each imaging frame.
3. `position_1m_bin`: four equal 1-m bins across the 4-m corridor.
4. `running_speed_quartile`: four global quartile bins using edges 12.4224, 25.3526, and 40.8546.

Neural values are the released Suite2p deconvolved `spks` traces. Cells assigned outside visual cortex (raw retinotopy codes `-1` and `7`) are excluded. Retained frames are native imaging samples in the textured corridor with positive VR movement, matching the reference neural-analysis mask. Frame timestamps remain exact even when removed stationary frames create gaps.

## Key statistics

| Statistic | Value |
|-----------|------:|
| Subjects | 19 |
| Sessions | 89 |
| Trials | 38,110 |
| Retained neuron-session units | 4,105,393 |
| Mean neurons/session | 46,128.01 |
| Retained timepoints | 821,579 |
| Nominal imaging interval | 315.4574 ms |
| Brain regions | V1, mHV, lHV, aHV |

Full output fractions are: stimulus `[0.311, 0.470, 0.085, 0.135]`, licking `[0.963, 0.037]`, position `[0.250, 0.249, 0.250, 0.252]`, and speed `[0.250, 0.250, 0.250, 0.250]` (rounded).

## Reproduction and validation

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

Full validation reported no errors or warnings. Full validation balanced accuracies were 0.8131 for visual category, 0.8482 for licking, 0.3104 for position, and 0.3487 for speed; chance levels were 0.25, 0.50, 0.25, and 0.25. See `CONVERSION_NOTES.md` for the raw-file `np.allclose` audit and interpretation.
