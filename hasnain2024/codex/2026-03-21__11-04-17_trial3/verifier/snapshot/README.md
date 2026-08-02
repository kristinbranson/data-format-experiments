# Neural Decoder Conversion

This directory contains a decoder-ready conversion of the dataset from *Separating cognitive and motor processes in the behaving mouse*.

## Outputs

- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: 2-session sample conversion used for debugging/validation
- `convert_data.py`: conversion script
- `CONVERSION_NOTES.md`: full audit trail of loading, curation, checks, and decoder results
- `conversion_sample_out.txt`, `verification_sample_out.txt`, `train_decoder_sample_out.txt`: sample-run logs
- `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt`: full-run logs

## Conversion Summary

- Data source: local files in `data/`
- Reference materials: `paper.pdf`, `methods.txt`, and MATLAB code in `code/`
- Alignment: auditory go cue onset
- Neural representation: go-cue-aligned spike rates, 5 ms bins, causal Gaussian smoothing
- Trial curation: exclude `stim.enable`, `early`, and `no` trials; additionally drop behavior-valid trials with no neural recording coverage
- Neuron curation: released-code quality filter (`all` except garbage/noisy-like labels) plus `>1 Hz` low-FR filter
- Outputs:
  - `lick_direction`: left/right
  - `behavioral_context`: WC/DR
  - `outcome`: incorrect/correct
  - `tongue_velocity_bin`, `paw_velocity_bin`, `motion_energy_bin`: below/above per-session median

## Data Format

The pickle stores a Python dictionary with:

- `neural`: session -> trial -> `(n_neurons, n_timepoints)` array
- `input`: session -> trial -> `(1, n_timepoints)` time-from-go-cue array
- `output`: session -> trial -> `(6, n_timepoints)` categorical targets
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`: task description, timing, curation settings, and per-session metadata

## Loading

```python
import pickle

with open("converted_data.pkl", "rb") as f:
    data = pickle.load(f)

print(len(data["neural"]))          # sessions
print(data["input_names"])          # ['time_from_go_cue']
print(data["output_names"])         # decoder targets
print(data["neural"][0][0].shape)   # (n_neurons, 1000)
```

## Re-running

```bash
python -u convert_data.py converted_data.pkl --full
python -u train_decoder.py converted_data.pkl --verify-only
python -u train_decoder.py converted_data.pkl --plot-samples
```

## Key Statistics

- Sessions: 44
- Subjects: 14
- Trials: 11,955
- Neurons: 2,455
- Brain regions observed in selected probes: `R ALM`, `ALM`, `L ALM`, `L M1TJ`

See `CONVERSION_NOTES.md` for the full step-by-step rationale and validation record.
