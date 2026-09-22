# MAP Neural Decoder Dataset

This directory contains a decoder-ready conversion of the Mesoscale Activity Map dataset (DANDI 000363, version 0.230822.0128), originally presented in *Brain-wide neural activity underlying memory-guided movement*.

## Main files

- `converted_data.pkl`: full converted dataset (173 sessions, 28 mice)
- `sample_data.pkl`: two-session development sample
- `convert_data.py`: reproducible PyNWB-only converter
- `CONVERSION_NOTES.md`: complete decisions, reference comparisons, checks, and decoder results
- `conversion_*_out.txt`, `verification_*_out.txt`, `train_decoder_*_out.txt`: required run logs

## Reproduce

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

The converter reads all NWBs through `pynwb.NWBHDF5IO`; it does not use `h5py`.

## Data structure

Load with:

```python
import pickle

with open("/app/converted_data.pkl", "rb") as stream:
    data = pickle.load(stream)
```

`data['neural'][session][trial]` is float32 firing rate in Hz with shape `(n_neurons, 80)`. Bins are non-overlapping, 50 ms wide, half-open, and span -2.5 to +1.5 s relative to auditory go-cue onset.

`data['input'][session][trial]` has shape `(2, 80)`:

1. continuous seconds since the final tone/sample onset preceding go;
2. binary photostimulation state at each bin center.

`data['output'][session][trial]` has shape `(4, 80)` and integer categorical values:

1. choice: left, right, no lick;
2. outcome: ignore, miss, hit;
3. early lick: no, yes;
4. tongue y: below session 40th percentile, 40th–60th, above 60th, not visible.

Choice/outcome/early labels repeat across a trial; tongue class varies over time. Tongue visibility uses side-camera DeepLabCut likelihood >=0.9, and percentiles use all visible frames within that session.

Other fields provide subject indices, detailed Allen CCF region indices, names/value labels, and extensive metadata including bin centers and per-session conversion statistics.

## Key statistics

| Statistic | Value |
|---|---:|
| Sessions | 173 |
| Subjects | 28 |
| Retained trials | 90,859 |
| Classifier-good session-neurons | 69,453 |
| Neurons/session | 90–923 (mean 401.46) |
| Timepoints/trial | 80 |
| Brain-region labels | 293 |
| Full pickle size | 11.042 GiB |

Trials with no spike across the entire classifier-good population in the requested window were excluded; all other requested categories, including early lick, ignore, and photostimulation, were retained. The full reference verifier reports no errors or warnings.

## Full decoder validation

| Output | Train balanced accuracy | Validation balanced accuracy | Chance |
|---|---:|---:|---:|
| Lick direction choice | 0.7075 | 0.6779 | 0.3333 |
| Outcome | 0.7029 | 0.6662 | 0.3333 |
| Early lick | 0.7992 | 0.7557 | 0.5000 |
| Tongue y-position | 0.6718 | 0.6305 | 0.2500 |

Training loss decreased from 15.5517 to 0.6581 over 200 epochs. See `CONVERSION_NOTES.md` for the full methodological rationale and audits.
