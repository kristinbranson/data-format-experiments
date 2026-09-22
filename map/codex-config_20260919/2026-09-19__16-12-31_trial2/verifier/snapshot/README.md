# Brain-wide Neural Decoder Dataset

This directory contains a decoder-ready conversion of the mouse electrophysiology dataset from *Brain-wide neural activity underlying memory-guided movement* (DANDI 000363, version 0.230822.0128). Trials are aligned to go-cue onset, span -2.5 to +1.5 seconds, and contain 80 nonoverlapping 50-ms firing-rate bins.

## Load the data

```python
import pickle

with open("/app/converted_data.pkl", "rb") as f:
    data = pickle.load(f)

# First session, first trial
neural = data["neural"][0][0]  # (neurons, 80), float32 Hz
inputs = data["input"][0][0]   # (2, 80), float32
outputs = data["output"][0][0] # (4, 80), int8 categories
```

The top-level fields are `neural`, `input`, `output`, `subjects`, `subject_idx`, `brain_regions`, `brain_region_idx`, `input_names`, `output_names`, `output_values`, and `metadata`. Session and trial ordering is identical across the three data lists. Scalar trial outputs are broadcast over 80 bins so they share an array with time-varying tongue position.

Inputs:

1. Time from the first within-trial tone onset, in seconds.
2. Binary photostimulation state at each bin center.

Outputs:

1. Lick choice: left, right, no lick (`0,1,2`).
2. Outcome: ignore, miss, hit (`0,1,2`).
3. Early lick: no, yes (`0,1`).
4. Session-discretized tongue y: below 40th percentile, 40th–60th, above 60th, not visible (`0,1,2,3`).

## Dataset summary

| Item | Value |
|------|-------|
| Subjects | 28 |
| Sessions | 173 |
| Trials | 90,253 |
| Classifier-curated session-units | 69,453 |
| Fine CCF region labels | 293 |
| Neurons/session | mean 401.46, range 90–923 |
| Trials/session | mean 521.69, range 159–800 |
| Choice fractions | left .428, right .422, no lick .150 |
| Outcome fractions | ignore .150, miss .166, hit .684 |
| Early-lick fractions | no .884, yes .116 |
| Tongue fractions | low .062, middle .032, high .065, not visible .841 |

Units use the published regional QC classifier (`classification == "good"`). Trials require ephys coverage, stability for every retained unit, a finite tone event, and population spikes in the requested window. Photostimulation, early-lick, ignore, miss, auto-water, and free-water trials are retained because the requested decoder targets require the broader behavior distribution. Missing/low-confidence side-camera tongue frames map to “not visible.”

The paper reports 69,943 good units, whereas this exact NWB release exports 69,453 good units across the same 173 usable sessions; the release value is used without synthesizing missing units. See [CONVERSION_NOTES.md](/app/CONVERSION_NOTES.md) for the full reconciliation and raw-file tests.

## Reproduce and validate

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

Full validation reports no errors or warnings. Validation balanced accuracies are 0.6760 choice, 0.6624 outcome, 0.7521 early lick, and 0.6241 tongue position. Detailed logs are retained in the corresponding `conversion_*`, `verification_*`, and `train_decoder_*` text files.

