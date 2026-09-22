# IBL Brain-Wide Map Decoder Dataset

## Overview

`converted_data.pkl` is a stimulus-aligned conversion of the IBL Brain-Wide Map Neuropixels release for neural decoding. It contains well-isolated spike-count activity and the requested task/behavior variables for 445 sessions from 136 mice.

All source scientific data were loaded through `one.api.ONE` and brainbox. The reproducible conversion is implemented in `convert_data.py`; detailed decisions, paper comparisons, checks, and decoder results are in `CONVERSION_NOTES.md`.

## Key statistics

- Sessions: **445** of 459 ephys release sessions; 14 lacked a required paired whisker-motion-energy stream
- Subjects: **136**
- Curated trials: **190,239**
- Neurons summed across sessions: **62,761**
- Brain regions: **264** Beryl acronyms
- Time bins: **100 × 20 ms**, from **-0.5 to +1.5 s** around visual stimulus onset
- Neural representation: float32 spike counts per bin
- Choice distribution: 49.25% left, 50.75% right
- Prior distribution: approximately 41.8% / 14.0% / 44.2% for 0.2 / 0.5 / 0.8
- Wheel and whisker classes: global tertiles, approximately one-third each

## Loading

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(len(data['neural']))       # 445 sessions
print(data['neural'][0][0].shape)  # (n_neurons, 100)
print(data['input'][0][0].shape)   # (2, 100)
print(data['output'][0][0].shape)  # (4, 100)
```

The pickle is approximately 11.28 GB and may require substantial memory when loaded.

## Dictionary format

- `neural[session][trial]`: `(n_neurons, 100)` float32 spike counts
- `input[session][trial]`: `(2, 100)` float32
  1. time since stimulus onset in seconds
  2. zero-based trial number in the current prior block, broadcast over time
- `output[session][trial]`: `(4, 100)` categorical arrays
  1. choice: left=0, right=1, broadcast over time
  2. prior probability left: 0.2→0, 0.5→1, 0.8→2, broadcast over time
  3. wheel-speed tertile: low/medium/high = 0/1/2
  4. whisker-motion-energy tertile: low/medium/high = 0/1/2
- `subjects`, `subject_idx`: subject vocabulary and session mapping
- `brain_regions`, `brain_region_idx`: Beryl region vocabulary and per-neuron mapping
- `input_names`, `output_names`, `output_values`: dimension descriptions
- `metadata`: alignment, bin size, filtering, tertile thresholds, EIDs, source trial indices, cameras, and neuron UUIDs

## Processing summary

1. Start from the 459-session BWM ephys release.
2. Require trials, wheel, spikes/clusters, and paired camera motion energy/timestamps.
3. Merge probes within each session.
4. Keep well-isolated clusters (`label >= 1`), remap Allen acronyms to Beryl, and exclude `root`/`void`.
5. Exclude trials missing required events, no-choice trials, invalid priors, and reaction times outside 0.08–2.00 s.
6. Bin spikes in 20 ms bins around `stimOn_times`.
7. Compute absolute filtered wheel velocity with brainbox; use left-camera whisker motion energy with right fallback.
8. Discretize dynamic behavior using global tertile thresholds stored in metadata.
9. Remove trials with no spikes from any retained neuron.

## Reproducing and validating

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

Final validation balanced accuracies:

| Output | Accuracy | Chance |
|---|---:|---:|
| Choice | 0.6156 | 0.5000 |
| Prior probability left | 0.6600 | 0.3333 |
| Wheel speed tertile | 0.6312 | 0.3333 |
| Whisker motion energy tertile | 0.7279 | 0.3333 |
