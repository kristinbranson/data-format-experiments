# MAP Neural Decoder Dataset

This project converts the Mesoscale Activity Map NWB dataset (DANDI:000363/0.230822.0128) into a trial-aligned neural-decoding dataset. It contains 173 usable recording sessions from 28 mice, 69,453 classifier-approved units, and 90,363 valid trials.

## Files

- `converted_data.pkl`: full converted dataset (11.17 GiB)
- `sample_data.pkl`: two-session test dataset
- `convert_data.py`: reproducible converter
- `CONVERSION_NOTES.md`: processing rationale, paper/code comparison, checks, and decoder results
- `conversion_*_out.txt`, `verification_*_out.txt`, `train_decoder_*_out.txt`: complete run logs
- `processing_*.png`, `sample_trials.png`, `predictions.png`, `early_lick_alignment.png`: processing and decoder diagnostics
- `cache/`: independent investigation scripts and logs

## Loading

```python
import pickle

with open("/app/converted_data.pkl", "rb") as f:
    data = pickle.load(f)

# First trial of first session
neural = data["neural"][0][0]   # (n_neurons, 80), firing rate in Hz
inputs = data["input"][0][0]    # (2, 80)
outputs = data["output"][0][0]  # (4, 80), categorical integer labels
```

All trials have 80 adjacent 50-ms bins spanning `[-2.5, +1.5)` seconds around go-cue onset. Bin centers run from -2.475 to +1.475 seconds.

Inputs:

1. Signed time from the final instruction-tone onset, seconds
2. Photostimulation state at the bin center (0/1)

Outputs:

1. Lick choice: left, right, no lick
2. Outcome: ignore, miss, hit
3. Early lick: no, yes
4. Tongue y-position: below session p40, p40–p60, above p60, not visible

Trial-level outputs 1–3 are broadcast across time. Tongue position varies across bins. Full class names and ordering are in `output_values`.

## Conversion decisions

- Units are retained only when the authors' NWB classifier label is `good`; no ad hoc metric threshold is substituted.
- Native absolute spikes, behavioral events, and side-camera tracking are aligned through each trial's go timestamp.
- NWB observation intervals and per-unit trial-validity flags are enforced. Partial recordings, incomplete requested windows, and population-all-zero truncated windows are excluded.
- Actual choice is derived from instructed side plus hit/miss/ignore outcome, matching task semantics.
- Tongue frames require DeepLabCut likelihood >=0.9. High-confidence velocity outliers are interpolated following the method paper; missing/occluded samples remain the required `not visible` class.
- Full Allen CCF annotation strings are preserved as brain regions.

See `CONVERSION_NOTES.md` for the complete justification and raw-data comparisons.

## Reproducing and validating

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

The final verifier reports no errors or warnings. Full validation balanced accuracies were 0.6789 choice, 0.6614 outcome, 0.7484 early lick, and 0.6166 tongue position.

