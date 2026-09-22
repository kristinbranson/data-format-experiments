# Track2p Barrel-Cortex Decoder Dataset

## Dataset

`converted_data.pkl` contains longitudinal two-photon calcium-imaging and motion data from six mice recorded in layer 2/3 barrel cortex. The source is associated with Majnik et al., *Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p*.

The converted decoder task predicts a time-varying, five-class motion-energy quintile from neural population activity. Decoder contextual input is elapsed time from the beginning of each session.

## Key statistics

- 6 subjects: `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`
- 41 sessions (6–7 per subject)
- 2,998 unique longitudinally matched neurons
- 20,445 neuron-session observations
- 1,081 non-overlapping complete 60-second trials
- 19–30 trials per session
- 180 timepoints per trial
- 333.333 ms bins (10 source frames averaged at 30 Hz)
- Motion classes are essentially exactly balanced at 20% each
- Full decoder validation balanced accuracy: 0.3072 (chance 0.2000)

## Processing summary

1. Load curated, longitudinally matched Suite2p `F`, `Fneu`, `iscell`, and `ops` arrays.
2. Apply Suite2p neuropil correction: `F - neucoeff * Fneu`.
3. Estimate and subtract the Suite2p maximin baseline using session-specific ops.
4. Average neural and motion signals over aligned non-overlapping blocks of 10 frames.
5. Use the shared valid neural/motion prefix and retain only complete 60-second intervals.
6. Compute motion-energy 20th, 40th, 60th, and 80th percentiles separately per session and encode labels 0–4.
7. Represent elapsed session time at the center of each 10-frame bin.

No missing behavior is padded or interpolated. In nine sessions with terminal video-frame deficits, only the incomplete final minute is excluded.

## Loading

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# First session, first trial
neural = data['neural'][0][0]  # (n_neurons, 180), float32
inputs = data['input'][0][0]   # (1, 180), float32 elapsed seconds
output = data['output'][0][0]  # (1, 180), int64 labels 0..4
```

## Structure

The pickle is a dictionary with:

- `neural[session][trial]`: baseline-corrected neural activity, `(n_neurons, 180)`
- `input[session][trial]`: elapsed session time in seconds, `(1, 180)`
- `output[session][trial]`: motion-energy quintile, `(1, 180)`
- `subjects` and `subject_idx`: session-to-mouse mapping
- `brain_regions` and `brain_region_idx`: all neurons are barrel cortex L2/3
- `input_names`, `output_names`, `output_values`: variable/class labels
- `metadata`: timing, processing provenance, and detailed per-session source/retention statistics

Class labels are:

0. quintile 1 (lowest motion)
1. quintile 2
2. quintile 3
3. quintile 4
4. quintile 5 (highest motion)

## Reproducing conversion and validation

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

See `CONVERSION_NOTES.md` for complete decisions, reference comparisons, sanity checks, edge cases, and decoder results.
