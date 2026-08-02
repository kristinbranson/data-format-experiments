# IBL Neural Decoder Conversion

This directory contains a stimulus-aligned conversion of the IBL Brain-Wide Map release into the decoder format expected by `train_decoder.py`.

## Dataset summary
- Source: IBL Brain-Wide Map 2025 Q3 release
- Retained sessions: 444
- Retained subjects: 136
- Retained trials: 188,925
- Retained neurons: 73,044
- Time bin size: 20 ms
- Alignment event: stimulus onset
- Trial window: `[-0.5, 1.5]` s

## Neural and behavioral content
- Neural data: good units only (`clusters.metrics.label >= 1`), merged across probes within each session
- Decoder inputs:
  - `time_since_stimulus_onset_s`
  - `trial_number_in_block`
- Decoder outputs:
  - `choice` with `left=0`, `right=1`
  - `prior_left_probability` with `0.2->0`, `0.5->1`, `0.8->2`
  - `wheel_speed_bin` with 3 global tertile bins
  - `whisker_motion_energy_bin` with 3 global tertile bins

## Files
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: 2-session sample conversion
- `convert_data.py`: conversion script
- `CONVERSION_NOTES.md`: detailed conversion log and validation notes
- `conversion_sample_out.txt`, `verification_sample_out.txt`, `train_decoder_sample_out.txt`: sample-run logs
- `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt`: full-run logs

## How to use

Load the converted dataset:

```python
import pickle

with open("converted_data.pkl", "rb") as f:
    data = pickle.load(f)
```

Validate format only:

```bash
python3 train_decoder.py converted_data.pkl --verify-only
```

Run decoder training:

```bash
python3 train_decoder.py converted_data.pkl --plot-samples --cpu
```

Rebuild the dataset:

```bash
python3 -u convert_data.py converted_data.pkl --full
```

## Format notes
- `neural[session][trial]` is shaped `(n_neurons, 100)`.
- `input[session][trial]` is shaped `(2, 100)`.
- `output[session][trial]` is shaped `(4, 100)`.
- Neural counts are stored compactly as `uint8` to keep the full dataset trainable; `train_decoder.py` converts them during training.

## Important exclusions
- 15 release sessions were excluded from the final converted set:
  - 14 lacked whisker motion energy traces
  - 1 had zero valid wheel+whisker-overlap trials after filtering
