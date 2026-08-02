# IBL Decoder Conversion

This directory contains a decoder-ready conversion of the IBL Brain-Wide Map ephys release for the stimulus-onset-aligned benchmark described in the task.

## Dataset Summary

- Sessions converted: 438
- Subjects: 135
- Trials: 186,261
- QC-passing neurons: 72,757
- Brain regions: 266 Beryl-mapped labels
- Temporal alignment: stimulus onset (`stimOn_times`)
- Trial window: `[-0.5, 1.5]` s relative to stimulus onset
- Time bin size: 20 ms
- Time bins per trial: 100

## Decoder Variables

- Inputs:
  - `time_since_stimulus_onset_s`
  - `trial_number_in_block`
- Outputs:
  - `choice` with values `left=0`, `right=1`
  - `prior_probability_left` with values `0.2=0`, `0.5=1`, `0.8=2`
  - `wheel_speed_bin` with values `low=0`, `medium=1`, `high=2`
  - `whisker_motion_energy_bin` with values `low=0`, `medium=1`, `high=2`

## Conversion Rules

- Neural activity is spike-count data from QC-passing clusters (`label >= 1`), merged across probes within each session.
- Brain-region labels are exported in the Beryl ontology to match the reference code.
- Trials are filtered to require valid stimulus onset, feedback, first movement timing, valid choice, valid block probability, and usable whisker and wheel data within the aligned window.
- Wheel speed and whisker motion energy are aligned to the same stimulus-onset trial bins and discretized into three global quantile bins.

## Files

- `convert_data.py`: conversion script
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: 2-session sample dataset
- `CONVERSION_NOTES.md`: full audit trail of decisions, checks, and validation
- `conversion_*_out.txt`, `verification_*_out.txt`, `train_decoder_*_out.txt`: captured logs from conversion, validation, and decoder training

## Usage

Run the full conversion:

```bash
python3 -u convert_data.py converted_data.pkl --full
```

Run a 2-session sample conversion with processing plots:

```bash
python3 -u convert_data.py sample_data.pkl --sample --show-processing
```

Verify the converted file format:

```bash
python3 -u train_decoder.py converted_data.pkl --verify-only
```

Train the benchmark decoder:

```bash
python3 -u train_decoder.py converted_data.pkl --plot-samples
```

Use `--cpu` if GPU memory is insufficient.

## Loading The Pickle

```python
import pickle

with open("converted_data.pkl", "rb") as f:
    data = pickle.load(f)

print(data.keys())
print(data["input_names"])
print(data["output_names"])
```

The pickle contains:

- `neural`: session -> trial -> `(n_neurons, n_timepoints)` arrays
- `input`: session -> trial -> `(2, n_timepoints)` arrays
- `output`: session -> trial -> `(4, n_timepoints)` arrays
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Validation Snapshot

Full decoder training completed successfully on CPU after a GPU memory failure during the initial full run.

- Choice validation balanced accuracy: 0.6183
- Prior validation balanced accuracy: 0.6670
- Wheel validation balanced accuracy: 0.6392
- Whisker validation balanced accuracy: 0.7383
