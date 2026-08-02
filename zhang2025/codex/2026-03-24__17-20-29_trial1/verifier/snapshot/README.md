# IBL BWM Decoder Conversion

This directory contains a conversion of the International Brain Laboratory brain-wide map (BWM) electrophysiology release into the decoder-ready pickle format required by `train_decoder.py`.

## Dataset Summary

- Source release: local IBL BWM cache in `data/one_cache/2025_Q3_IBL_et_al_BWM`
- Exported sessions: 438
- Exported subjects: 135
- Exported trials: 186,245
- Exported well-isolated neurons: 72,757
- Alignment event: stimulus onset
- Time window: `[-0.5, 1.5]` s relative to stimulus onset
- Bin size: `20 ms`

## Exported Files

- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: 2-session sample export
- `convert_data.py`: conversion script
- `CONVERSION_NOTES.md`: step-by-step conversion log, checks, and validation
- `conversion_sample_out.txt`, `verification_sample_out.txt`, `train_decoder_sample_out.txt`: sample-run logs
- `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt`: full-run logs

## Inputs and Outputs

The exported pickle contains:

- Inputs:
  - `time_since_stimulus_onset`
  - `trial_number_in_block`
- Outputs:
  - `choice`: `left=0`, `right=1`
  - `prior_probability_of_left`: `0.2->0`, `0.5->1`, `0.8->2`
  - `wheel_speed_bin`: `low=0`, `medium=1`, `high=2`
  - `whisker_motion_energy_bin`: `low=0`, `medium=1`, `high=2`

Neural data are stored as per-trial `(n_neurons, 100)` spike-count matrices. Inputs and outputs are stored as per-trial `(d, 100)` arrays.

## Loading the Converted Data

```python
import pickle

with open("converted_data.pkl", "rb") as f:
    data = pickle.load(f)

print(data.keys())
print(data["metadata"])
print(len(data["neural"]), "sessions")
```

## Reproducing the Conversion

Full export:

```bash
python -u convert_data.py converted_data.pkl --full
```

Sample export with processing plots:

```bash
python -u convert_data.py sample_data.pkl --sample --show-processing
```

Verify format only:

```bash
python -u train_decoder.py converted_data.pkl --verify-only
```

Train decoder:

```bash
python -u train_decoder.py converted_data.pkl --plot-samples --cpu
```

## Notes

- Neurons are filtered using `clusters.metrics.label >= 1`, matching the 75,708 well-isolated neurons described in the BWM release and yielding 72,757 neurons in the kept exported sessions.
- Sessions lacking required wheel or whisker streams are excluded.
- Trials with missing required behavior or all-zero neural windows are excluded.
- Dynamic behaviors are discretized into three global tertile bins, as required by the decoder task.
