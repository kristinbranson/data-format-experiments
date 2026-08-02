# Converted Decoder Dataset

This repository now contains a decoder-ready conversion of the Zhong et al. imaging dataset from *Unsupervised pretraining in biological neural networks*.

## Files

- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: 2-session sample dataset for quick checks
- `convert_data.py`: conversion script
- `CONVERSION_NOTES.md`: detailed conversion log, checks, and rationale
- `train_decoder_full_out.txt`: full decoder training log
- `verification_full_out.txt`: full format-verification log

## Dataset Summary

- Subjects: `19`
- Sessions: `89`
- Trials: `38,110`
- Curated neurons: `357,328`
- Brain regions: `V1`, `mHV`, `lHV`, `aHV`, `other`
- Temporal alignment: corridor entry (`StartFr`)
- Trial end: gray-space entry (`GrayFr`)
- Frame selection: running-only frames within the texture corridor

## Conversion Choices

- Neural data use the paper’s deconvolved imaging traces from `data/spk`.
- Sessions are deduplicated to the `89` unique imaging recordings in `Imaging_Exp_info.npy`.
- Trials keep only running frames between corridor entry and gray-space entry.
- Stimulus-selective neurons use running-corridor `|d'| >= 0.3`.
- Reward-prediction neurons use the paper-style aHV criterion: cue-frame stimulus selectivity plus late-vs-early cue `d'` above the session 95th percentile.

## Decoder Variables

Inputs:
- `time_to_sound_cue_sec`
- `day_of_training`
- `time_since_trial_start_sec`
- `reward_available`

Outputs:
- `visual_stimulus_category`
- `licking`
- `position_bin`
- `running_speed_bin`

## Loading

```python
import pickle

with open("converted_data.pkl", "rb") as f:
    data = pickle.load(f)

print(data.keys())
print(len(data["neural"]), "sessions")
print(data["input_names"])
print(data["output_names"])
```

Each session is stored as a list of trials. Each neural trial is an array of shape `(n_neurons, n_timepoints)`. Each input/output trial is an array of shape `(n_variables, n_timepoints)`.

## Re-running

Full conversion:

```bash
python3 -u convert_data.py converted_data.pkl --full
```

Sample conversion:

```bash
python3 -u convert_data.py sample_data.pkl --sample --show-processing
```

Verify format:

```bash
python3 -u train_decoder.py converted_data.pkl --verify-only
```

Train decoder:

```bash
python3 -u train_decoder.py converted_data.pkl --plot-samples
```
