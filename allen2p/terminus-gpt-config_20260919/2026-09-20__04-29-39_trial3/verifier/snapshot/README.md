# Allen Visual Behavior Ophys Decoder Dataset

## Description
This directory contains a decoder-ready conversion of the local Allen Brain Observatory Visual Behavior 2P release 1.1.0. Each target session is one ophys experiment/imaging plane. Neural activity is the Allen-released dF/F signal, aligned on absolute ophys timestamps and sampled at uniform 100 ms grid centers within native trials.

The conversion keeps Go and Catch trials and excludes Aborted and Auto-rewarded trials. Three experiments without any pupil stream are excluded because pupil diameter is a required output; all eligible trials in the remaining 281 experiments are retained.

## Files
- `converted_data.pkl`: complete converted dataset
- `sample_data.pkl`: two-session validation sample spanning native fast/slow ophys rates
- `convert_data.py`: reproducible converter
- `CONVERSION_NOTES.md`: detailed decisions, source comparisons, checks, and decoder results
- `conversion_*_out.txt`, `verification_*_out.txt`, `train_decoder_*_out.txt`: execution logs
- `processing_*.png`: conversion diagnostics
- `cache/`: investigation scripts and extracted reference text

## Loading
```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural_trial = data['neural'][0][0]  # neurons x time
output_trial = data['output'][0][0]  # 5 x time
```

## Structure
The pickle follows the requested dictionary format:
- `neural[session][trial]`: float32 dF/F, shape `(n_neurons, T)`
- `input[session][trial]`: empty float32 array, shape `(0, T)` because there are no decoder inputs
- `output[session][trial]`: integer categorical matrix, shape `(5, T)`
- subject and brain-region lookup/index fields
- output names/value labels and detailed metadata/session information

Output rows are:
1. `image_identity`: gray plus 16 image identities across image sets A/B
2. `image_change`: one-bin pulse immediately after a true identity change
3. `running_speed_quintile`: session-level equal-frequency bins
4. `pupil_diameter_quintile`: session-level equal-frequency bins; diameter is `2*sqrt(processed_pupil_area/pi)`
5. `trial_outcome`: hit, miss, false alarm, or correct reject; repeated over T to represent a static per-trial label in a rectangular output matrix

## Key Statistics
- 281 ophys experiments/sessions
- 38 mice
- 84,313 trials
- 41,871 neurons
- Brain regions: VISp and VISl
- 100 ms time bins; trial lengths 70–126 bins
- No validator errors or warnings

Full validation balanced accuracies:
- image identity: 0.3634 (chance 0.0588)
- image change: 0.6281 (chance 0.5000)
- running quintile: 0.2698 (chance 0.2000)
- pupil quintile: 0.2702 (chance 0.2000)
- trial outcome: 0.3219 (chance 0.2500)

## Reproduction
```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
```

See `CONVERSION_NOTES.md` for curation rationale, source-to-target mappings, raw-data `np.allclose` audits, and reference-paper comparisons.
