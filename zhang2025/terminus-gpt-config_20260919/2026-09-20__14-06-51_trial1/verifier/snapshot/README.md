# IBL Neural Decoder Dataset

This project converts a deterministic 10-session subset of the International Brain Laboratory Brain-Wide Map electrophysiology release into stimulus-aligned decoder tensors.

## Files

- `converted_data.pkl`: full converted dataset (10 sessions)
- `sample_data.pkl`: two-session test dataset
- `convert_data.py`: reproducible converter
- `CONVERSION_NOTES.md`: detailed decisions, paper/code comparisons, checks, and decoder results
- `verification_full_out.txt`: final format validation
- `train_decoder_full_out.txt`: final decoder training log

## Reproduce

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

Source payloads are downloaded from the public IBL BWM S3 repository and cached under `data/raw_cache/`. Selection uses the reference code's seed 42 and one session from each of 10 subjects.

## Data Format

Load with:

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

Each `data['neural'][session][trial]` is a float32 neuron-by-time spike-count matrix. Inputs are `(2, 105)` and outputs are `(4, 105)`.

### Alignment and processing

- Alignment: visual stimulus onset (`trials.stimOn_times`)
- Window: −0.6 to +1.5 seconds
- Bin size: 20 ms (105 bins)
- Units: pykilosort clusters with `clusters.metrics.label >= 1`
- Simultaneous probes are concatenated within each session
- Wheel speed: absolute filtered IBL wheel velocity
- Whisker motion: official left-camera ROI motion energy (right fallback supported)
- Dynamic outputs: pooled dataset-wide tertiles

### Variables

Inputs:
1. Time since stimulus onset
2. Zero-based trial number in the current probability-left block

Outputs:
1. Choice: left=0, right=1
2. Prior probability left: 0.2=0, 0.5=1, 0.8=2
3. Wheel speed: low/medium/high
4. Whisker motion energy: low/medium/high

Choice and prior are broadcast over time because they are per-trial variables.

## Final Statistics

- 10 subjects and 10 sessions
- 6,684 retained trials
- 1,207 quality-filtered session-neurons (44–294/session)
- 108 source anatomical acronyms
- Choice distribution: 45.72% left, 54.28% right
- Prior distribution: 41.59% / 13.41% / 45.00%
- Wheel and whisker classes: exactly one-third each pooled
- Pickle size: approximately 386 MB

Final validation reported no errors or warnings. Validation balanced accuracies were choice 0.6028, prior 0.6414, wheel 0.6356, and whisker 0.7090.
