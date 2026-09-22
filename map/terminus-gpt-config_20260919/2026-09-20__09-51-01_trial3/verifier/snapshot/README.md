# MAP Neural Decoder Dataset Conversion

## Dataset

This directory contains a decoder-ready conversion of the Mesoscale Activity Map dataset from **“Brain-wide neural activity underlying memory-guided movement”**, using processing guidance from **“Brain-wide analysis reveals movement encoding structured across and within brain areas”** and the supplied spike-sorting/QC white paper.

The source data are NWB extracellular electrophysiology recordings with synchronized task events and side-camera DeepLabCut tongue tracking.

## Main Files

- `converted_data.pkl` — full converted dataset
- `sample_data.pkl` — two-session test dataset
- `convert_data.py` — reproducible conversion script
- `CONVERSION_NOTES.md` — complete processing decisions, statistics, checks, and decoder results
- `conversion_full_out.txt`, `verification_full_out.txt` — full conversion and format-validation logs
- `train_decoder_full_out.txt` — completed full decoder training log

## Reproducing the Conversion

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full
```

For a two-session test with diagnostic plots:

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
```

Validate or train:

```bash
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

## Loading

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

trial_neural = data['neural'][0][0]  # neurons x 80 time bins
trial_input = data['input'][0][0]    # 2 x 80
trial_output = data['output'][0][0]  # 4 x 80
```

## Temporal Processing

- Alignment: go-cue onset
- Window: -2.5 to +1.5 seconds
- Bins: 80 non-overlapping 50-ms bins
- Neural values: firing rates in Hz
- Spike counting: half-open `[left, right)` bins

## Variables

### Decoder inputs

1. `time from tone onset` — seconds from the most recent sample/tone onset before go
2. `photostimulation on` — binary time series from laser on/off events

### Decoder outputs

1. `lick direction choice`: left, right, no lick
2. `outcome`: ignore, miss, hit
3. `early lick`: no, yes
4. `tongue y-position`: below session 40th percentile, 40th–60th percentile, above 60th percentile, not visible

Tongue visibility uses DeepLabCut likelihood ≥0.9. Percentiles are calculated per session using visible frames only.

## Curation

- 173 sessions with the paper's custom classifier labels are retained; one NWB lacking classifier output is excluded.
- Neurons require `classification == 'good'` and valid `is_good_trials` across recorded trials.
- Behavioral trials are mapped to actual electrophysiology observation intervals in `units/obs_intervals`.
- Trials lacking synchronized video or any usable neural activity in the requested window are excluded.
- Early-lick, ignore, free-water, and photostimulation trials are retained because they are required by this decoder task.
- Native Allen CCF `anno_name` labels are preserved as brain regions.

## Full Dataset Summary

- Subjects: 28
- Sessions: 173
- Trials: 90,094
- Stable classifier-good neurons across sessions: 68,888
- Neurons/session: mean 398.2, range 90–923
- Native brain-region labels: 293
- Time bins/trial: 80
- Pickle size: approximately 11.6 GB

## Full Decoder Results

Balanced validation accuracy:

| Output | Accuracy | Chance |
|--------|----------|--------|
| Lick direction choice | 0.6766 | 0.3333 |
| Outcome | 0.6640 | 0.3333 |
| Early lick | 0.7476 | 0.5000 |
| Tongue y-position | 0.6085 | 0.2500 |

Training completed for 200 epochs; loss decreased from 19.4850 to 0.6731 and test loss was 0.6699.

See `CONVERSION_NOTES.md` for detailed rationale, source comparisons, edge cases, independent `np.allclose` checks, and issue-resolution history.
