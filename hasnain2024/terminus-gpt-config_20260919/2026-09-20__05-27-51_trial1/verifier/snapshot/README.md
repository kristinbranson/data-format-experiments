# Two-Context ALM Neural Decoder Dataset

## Dataset

`converted_data.pkl` contains go-cue-aligned ALM electrophysiology and behavior from the two-context subset of **“Separating cognitive and motor processes in the behaving mouse.”** It includes 12 recording sessions selected by the paper's Figure 8 loaders.

Key statistics:

- 12 sessions, 7 released subject IDs
- 3,491 non-stimulation trials
- 515 post-curation, >1 Hz session-neurons
- 500 time bins per trial, 10 ms/bin
- Window: -2.5 to +2.5 s around Bpod go-cue onset
- Brain region: ALM

The manuscript states six mice, while the released loaders and data use seven IDs for the exact 12 available sessions. This and all processing decisions are documented in `CONVERSION_NOTES.md`.

## Files

- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: two-session test dataset
- `convert_data.py`: reproducible conversion script
- `CONVERSION_NOTES.md`: detailed audit trail, decisions, checks, and decoder results
- `conversion_*_out.txt`, `verification_*_out.txt`, `train_decoder_*_out.txt`: run logs
- `processing_*.png`: sample conversion diagnostics
- `cache/`: investigation scripts and summaries

## Reproduce

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

## Load

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

trial_neural = data['neural'][0][0]  # neurons x 500
trial_input  = data['input'][0][0]   # 1 x 500
trial_output = data['output'][0][0]  # 6 x 500
```

Top-level lists are organized as session → trial. Neural values are smoothed firing rates in Hz. The input is continuous time from go cue in seconds.

## Outputs

Rows of each output matrix follow `output_names`:

1. **lick direction**: left, right, none
2. **behavioral context**: WC, DR
3. **outcome**: incorrect, correct, ignore
4. **tongue velocity**: below session median, at/above median, not visible
5. **paw velocity**: below session median, at/above median, not visible
6. **motion energy**: below session median, at/above median, no video

Per-trial outputs (lick, context, outcome) are repeated over time. Video outputs vary over time. Missing tongue/paw visibility and absent video are preserved as category 2 rather than treated as low velocity.

## Processing Summary

- Sessions/probes follow the reference Figure 8 ALM/video loaders.
- Photostimulation trials are excluded, matching reference context conditions.
- Spikes are aligned by subtracting each trial's `bp.ev.goCue`, binned at 10 ms over ±2.5 s, converted to Hz, and smoothed with the reference width of 15 bins.
- Clear artifact clusters are excluded and retained units must have mean firing rate >1 Hz.
- DLC tongue and top-paw velocities and per-trial motion energy are aligned with camera timestamps and discretized using per-session medians.

See `CONVERSION_NOTES.md` for exact mappings, source comparisons, edge cases, and validation tables.
