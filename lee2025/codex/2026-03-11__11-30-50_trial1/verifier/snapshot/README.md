# CA1 Geometry Remapping Conversion

This project converts the CA1 calcium imaging dataset from the paper "Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping" into the decoder-ready format expected by `train_decoder.py`.

## Dataset Summary

- Subjects: 7 mice
- Sessions: 207
- Session length: 40 minutes at 30 Hz
- Neural signal: binarized calcium event traces from the source dataset
- Brain region: CA1
- Converted trialing: 40 one-minute trials per session
- Converted time bin: 100 ms non-overlapping bins (3 frames at 30 Hz)
- Decoder input: 9-dimensional static arena-geometry vector indicating which 3x3 partitions are open
- Decoder output: mouse position discretized into 9 classes on a 3x3 grid

## Files

- `convert_data.py`: conversion script
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: 2-session sample conversion
- `CONVERSION_NOTES.md`: full audit trail of processing, checks, and validation

## Run Conversion

```bash
python3 -u convert_data.py converted_data.pkl --full
python3 -u convert_data.py sample_data.pkl --sample --show-processing
```

## Validate

```bash
python3 -u train_decoder.py converted_data.pkl --verify-only
python3 -u train_decoder.py converted_data.pkl --plot-samples
```

## Converted Format

The pickle stores a dictionary with:

- `neural`: session list of trial matrices shaped `(n_neurons, n_timepoints)`
- `input`: session list of trial inputs shaped `(9,)`
- `output`: session list of trial outputs shaped `(1, n_timepoints)` with values `0..8`
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Loading Example

```python
import pickle

with open("converted_data.pkl", "rb") as f:
    data = pickle.load(f)

trial_neural = data["neural"][0][0]
trial_input = data["input"][0][0]
trial_output = data["output"][0][0]
print(trial_neural.shape, trial_input.shape, trial_output.shape)
```

## Key Processing Choices

- Reused the source dataset's binarized calcium event traces; no dF/F recomputation.
- Kept session-valid neurons and excluded neuron-session entries marked invalid by NaNs in the source traces.
- Matched the reference temporal binning used by the paper's decoder: 3-frame bins at 30 Hz.
- Constructed position classes from raw x-y trajectories using session-wise max-based 3x3 discretization.
- Converted arena geometry from the source `blocked` representation into a 9-element open/closed vector aligned to the same class ordering as the output bins.

## Validation Summary

- `train_decoder.py --verify-only` passed on sample and full datasets without errors or warnings.
- Full decoder training completed successfully.
- Full-data balanced accuracy for `position_bin_3x3`: training `0.6378`, validation `0.5637` (chance `0.1111`).
