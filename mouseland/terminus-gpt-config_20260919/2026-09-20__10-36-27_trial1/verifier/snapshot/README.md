# Converted Neural Decoder Dataset

## Dataset
This conversion contains two-photon mesoscope recordings from **“Unsupervised pretraining in biological neural networks.”** It includes 89 recordings from 19 mice and preserves all 4,691,034 Suite2p-classified neurons (20,547–89,577 per session).

The converted file is `/app/converted_data.pkl` (about 161.6 GiB). Neural activity is nonnegative Suite2p-deconvolved fluorescence. Three imaging planes are concatenated along the neuron dimension. Trials contain synchronized running frames from the 4 m visual corridor; gray-space and stationary frames are excluded following the reference analysis.

## Loading
```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

Loading requires substantial RAM because the full pickle is approximately 162 GiB.

## Structure
- `data['neural'][session][trial]`: float32 `(n_neurons, n_timepoints)`
- `data['input'][session][trial]`: float32 `(4, n_timepoints)`
  1. time to sound cue (seconds; positive before cue)
  2. day/session of training
  3. time since corridor entry (seconds)
  4. reward availability (0/1)
- `data['output'][session][trial]`: int16 `(4, n_timepoints)`
  1. visual stimulus category (15 physical wall identities)
  2. licking (0/1)
  3. position class: 0–1, 1–2, 2–3, or 3–4 m
  4. global running-speed quartile
- `subjects`, `subject_idx`: subject lookup
- `brain_regions`, `brain_region_idx`: per-neuron region lookup
- `metadata['session_info']`: source session IDs, aliases, source trial IDs, frame interval, training day, and processing details

## Key Statistics
| Statistic | Value |
|-----------|-------|
| Subjects | 19 |
| Sessions | 89 |
| Trials | 38,110 |
| Retained timepoints | 821,579 |
| Neurons | 4,691,034 |
| Nominal time bin | 314.804 ms |
| Position fractions | 0.2500, 0.2487, 0.2496, 0.2517 |
| Speed fractions | 0.2493, 0.2502, 0.2501, 0.2504 |
| Licking-positive fraction | 0.0347 |

## Reproduction and Validation
```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

Full validation reported no errors or warnings. Full validation balanced accuracies were:
- visual stimulus: 0.5921 (chance 0.0667)
- licking: 0.8395 (chance 0.5)
- position: 0.3452 (chance 0.25)
- speed quartile: 0.3212 (chance 0.25)

See `/app/CONVERSION_NOTES.md` for source mapping, decisions, exact checks, issues, and rationale.
