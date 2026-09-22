# Reward-Relative CA1 Dataset Conversion

## Dataset

This project converts the released NWB data associated with **“A flexible hippocampal population code for experience relative to reward”** into a trial-organized format for neural decoding.

The complete conversion contains:

- 11 subjects
- 152 recording sessions
- 12,215 fully imaged trials
- 138,678 curated neuron-session recordings
- 450 cm virtual linear-corridor behavior
- CA1 two-photon deconvolved calcium activity
- 100 ms common temporal bins aligned to explicit trial start

## Files

- `converted_data.pkl`: complete converted dataset (about 5.9 GiB)
- `sample_data.pkl`: two-session test dataset
- `convert_data.py`: reproducible converter using the `pynwb` API
- `CONVERSION_NOTES.md`: detailed decisions, reference comparisons, checks, and decoder results
- `conversion_*_out.txt`: conversion logs
- `verification_*_out.txt`: format-validation logs
- `train_decoder_*_out.txt`: sample and full decoder logs
- `processing_*.png`: sample conversion visualizations

## Reproducing the Conversion

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
```

The converter accesses NWB files exclusively with `pynwb.NWBHDF5IO`.

## Loading

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# First trial of first session
neural = data['neural'][0][0]  # neurons x time
inputs = data['input'][0][0]   # 4 x time
outputs = data['output'][0][0] # 6 x time
```

## Data Structure

The top-level dictionary contains:

- `neural[session][trial]`: curated deconvolved activity, `(n_neurons, n_timepoints)`
- `input[session][trial]`: `(4, n_timepoints)`
- `output[session][trial]`: `(6, n_timepoints)`
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`, including session details and reward-zone coordinates

### Decoder Inputs

1. Time from trial start (seconds)
2. Environment type (ENV1=0, ENV2=1)
3. Session-local trial number
4. Previous trial outcome (omitted=0, rewarded=1; first trial=0)

### Decoder Outputs

1. Signed distance to the active reward zone, discretized into 7 classes
2. Absolute corridor position, discretized into 5 classes
3. Speed, discretized into 5 classes
4. Lick (native count > 0)
5. Reward-zone location (A/B/C)
6. Reward outcome (omitted/rewarded)

Per-trial values are repeated across time so all trial arrays have a uniform variable-by-time representation.

## Processing Summary

- Trials run from each explicit `trial_start` pulse through the following `teleport` pulse.
- Inter-trial periods and their -500 cm position sentinel are excluded.
- Neural activity is the NWB `Deconvolved` RoiResponseSeries, retaining Suite2p rows with `iscell[:, 0] > 0`.
- All planes in multi-plane sessions are curated through each series' NWB DynamicTableRegion and concatenated. For m17/m18 interleaved acquisition, the methods-supported effective rate is 15.5 Hz per plane.
- Trials must be fully covered from start through teleport by every neural plane; one terminal trial lacking complete imaging was excluded.
- Neural values are averaged in 100 ms bins; continuous behavior is timestamp-interpolated to bin centers.
- Fixed zones are A=80–100 cm, B=200–220 cm, and C=320–340 cm.
- Reward outcome is determined from sparse NWB Reward-event timestamps inside each trial.

## Full-Dataset Output Fractions

| Output | Fractions by class |
|--------|--------------------|
| Distance | 0.2518, 0.1019, 0.0739, 0.1676, 0.0269, 0.0832, 0.2946 |
| Position | 0.2092, 0.1784, 0.2318, 0.2273, 0.1533 |
| Speed | 0.1224, 0.0888, 0.1343, 0.3179, 0.3366 |
| Lick | 0.7689, 0.2311 |
| Reward zone | 0.3283, 0.3369, 0.3349 |
| Reward outcome | 0.1572, 0.8428 |

Trial-level zone counts are A=4,183, B=4,008, C=4,024. There are 1,874 omission and 10,341 rewarded trials.

## Decoder Validation

Full validation balanced accuracies were:

| Output | Accuracy | Chance |
|--------|----------|--------|
| Distance to reward zone | 0.4123 | 0.1429 |
| Absolute position | 0.5307 | 0.2000 |
| Speed | 0.4608 | 0.2000 |
| Lick | 0.6466 | 0.5000 |
| Reward-zone location | 0.8110 | 0.3333 |
| Reward outcome | 0.5162 | 0.5000 |

The validator reported a valid format with no errors or warnings. See `CONVERSION_NOTES.md` for raw-NWB `np.allclose` checks, reference-code comparisons, edge-case review, and interpretation.
