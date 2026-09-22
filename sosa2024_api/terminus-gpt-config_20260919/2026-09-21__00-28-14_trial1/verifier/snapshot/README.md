# Sosa et al. Hippocampal Decoder Dataset

This directory contains a decoder-ready conversion of the NWB data associated with **“A flexible hippocampal population code for experience relative to reward”**.

## Main files

- `converted_data.pkl` — full converted dataset (152 sessions)
- `sample_data.pkl` — two-session test dataset (one single-plane and one two-plane session)
- `convert_data.py` — reproducible pynwb-based converter
- `CONVERSION_NOTES.md` — detailed decisions, reference comparisons, checks, and decoder results
- `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt` — full run logs
- `conversion_sample_out.txt`, `verification_sample_out.txt`, `train_decoder_sample_out.txt` — sample run logs

## Dataset summary

- 11 mice
- 152 available imaging sessions: 77 reward-switch and 75 stay sessions
- 12,216 valid trials
- 138,678 session-neurons after Suite2p `iscell` curation
- 73,512 cells in the 77 switch sessions, exactly matching the paper
- Hippocampal CA1 two-photon calcium imaging
- Common temporal bin: 64.4836 ms (15.5078125 Hz)
- Trials aligned to entry onto the linear track (`trial_start`) and ending before teleport/ITI entry

## Loading

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(data.keys())
print(len(data['neural']))       # 152 sessions
print(data['neural'][0][0].shape)  # neurons x time
print(data['input'][0][0].shape)   # 4 x time
print(data['output'][0][0].shape)  # 6 x time
```

Each of `neural`, `input`, and `output` is a list of sessions, where each session is a list of trials. Trial lengths vary naturally, but neural/input/output lengths agree within every trial.

## Inputs

All inputs have shape `(4, time)`:

1. Time from trial start, seconds
2. Environment type: ENV1=0, ENV2=1
3. Zero-based trial number
4. Previous trial outcome: omitted=0, rewarded=1

Per-trial values are repeated over time because the supplied decoder requires time-varying matrices.

## Outputs

All outputs are categorical and have shape `(6, time)`:

1. Signed distance to the nearest point in the current reward-zone interval, 7 classes
2. Absolute corridor position, 5 classes
3. Speed, 5 classes
4. Lick, binary
5. Reward-zone location: A=0, B=1, C=2
6. Reward outcome: omitted=0, rewarded=1

Reward zones are the 50 cm intervals A=[80,130], B=[200,250], and C=[320,370] cm. On switch sessions, source-zone trials are 0–29 and the destination begins at trial 30.

## Neural processing

The converter reads NWB files exclusively through `pynwb.NWBHDF5IO`. It uses exported Suite2p deconvolved calcium activity, iterates over every imaging plane, maps each response series through its `DynamicTableRegion`, and retains ROIs with `iscell[:, 0] == 1`. No place-cell-only or running-speed filter is applied because the requested decoder requires complete trial behavior, including stationary periods.

Two-plane RoiResponseSeries metadata report 31.015625 Hz, but their rows align one-to-one with behavior timestamps at 15.5078125 Hz. The explicit behavior timestamps are therefore used as the authoritative clock; planes are concatenated neuron-wise without temporal downsampling.

## Reproducing conversion and validation

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/sample_data.pkl --verify-only
python -u /app/train_decoder.py /app/sample_data.pkl

python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

## Full decoder validation accuracy

| Output | Balanced accuracy | Chance |
|---|---:|---:|
| Distance to reward zone | 0.4131 | 0.1429 |
| Absolute position | 0.5360 | 0.2000 |
| Speed | 0.4283 | 0.2000 |
| Lick | 0.6000 | 0.5000 |
| Reward-zone location | 0.8007 | 0.3333 |
| Reward outcome | 0.5058 | 0.5000 |

Training loss decreased from 167.40 to 1.1966. See `CONVERSION_NOTES.md` for detailed interpretation, raw-data sanity checks, edge-case review, and reference-paper comparisons.
