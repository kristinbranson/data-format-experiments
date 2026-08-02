# Track2p Motion Decoder Conversion

This directory contains a decoder-ready conversion of the Track2p developmental barrel-cortex dataset from Majnik et al. (2025).

## Dataset summary

- Source data: longitudinal 2-photon calcium imaging from 6 mice in barrel cortex, with matched neurons across daily sessions.
- Converted file: `converted_data.pkl`
- Sessions: 41
- Pseudo-trials: 545
- Subjects: 6
- Brain region label: `barrel cortex L2/3`
- Neural trace used: Suite2p-style baseline-corrected fluorescence reconstructed from `F.npy`, `Fneu.npy`, and per-session `ops.npy`

## Conversion choices

- Continuous sessions were denoised exactly as described in the paper for decoding:
  - neural and behavior streams were averaged in non-overlapping 10-frame bins
  - this produces a time bin size of `333.333 ms`
- Because the decoder expects trials, each continuous session was segmented into consecutive 2-minute blocks:
  - 20-minute sessions become 10 trials
  - 30-minute sessions become 15 trials
- Motion-energy outputs were reconstructed to imaging length when camera frames were missing, then linearly interpolated at the missing frame positions before binning.
- The benchmark requires categorical outputs, so binned motion energy was globally min-max normalized and discretized into 5 equal-percentile bins.

## File format

`converted_data.pkl` is a Python dictionary with the required keys:

- `neural`: list of sessions, each a list of trials, each trial shaped `(n_neurons, n_timepoints)`
- `input`: list of sessions/trials, with one time-varying input `elapsed_time_sec`
- `output`: list of sessions/trials, with one time-varying categorical output `motion_energy_quintile`
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Load example

```python
import pickle

with open("converted_data.pkl", "rb") as f:
    data = pickle.load(f)

print(data["metadata"]["session_ids"][:3])
print(data["neural"][0][0].shape)
print(data["input_names"])
print(data["output_names"], data["output_values"])
```

## Re-running conversion

Sample run:

```bash
python3 -u convert_data.py sample_data.pkl --sample --show-processing
```

Full run:

```bash
python3 -u convert_data.py converted_data.pkl --full
```

Validation:

```bash
python3 -u train_decoder.py converted_data.pkl --verify-only
python3 -u train_decoder.py converted_data.pkl --plot-samples
```
