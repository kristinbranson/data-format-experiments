# Visual Behavior Neural Decoder Dataset

This directory contains a decoder-ready conversion of the locally cached Allen Brain Observatory **VisualBehavior** two-photon dataset. It includes active go/catch change-detection sessions, release-QC-valid VISp cells, and the reference paper's unfiltered L0 calcium-event magnitudes sampled on native synchronized ophys timestamps.

## Files

- `converted_data.pkl`: full conversion (165 sessions, 42,470 trials, 37 mice, 28,821 session-neurons)
- `sample_data.pkl`: two cell-rich sessions, one from each image set
- `convert_data.py`: reproducible AllenSDK-only converter
- `CONVERSION_NOTES.md`: decisions, source comparisons, statistics, checks, and decoder results
- `conversion_*_out.txt`, `verification_*_out.txt`, `train_decoder_*_out.txt`: requested run logs
- `processing_<id>.png`: sample processing diagnostics
- `cache/`: auxiliary review artifacts

## Reproduce

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
```

The converter accesses experiments exclusively through `VisualBehaviorOphysProjectCache`; it does not open NWB files directly.

## Load and format

```python
import pickle

with open("/app/converted_data.pkl", "rb") as f:
    data = pickle.load(f)

neural_trial = data["neural"][0][0]  # neurons × ophys frames
output_trial = data["output"][0][0]  # 5 × ophys frames
```

`input` has shape `(0, T)` because the task specifies no decoder inputs. Output rows are:

1. image identity: gray plus 16 natural-image classes;
2. image change: one on the first ophys frame at/after a real identity change;
3. running-speed quintile;
4. blink-clean pupil diameter-equivalent quintile;
5. trial outcome: hit, miss, false alarm, or correct reject, repeated across time to coexist with time-varying rows.

Trials use the half-open SDK interval `[start_time, stop_time)` and exclude aborted and auto-rewarded trials. Three otherwise eligible sessions are excluded because they contain fewer than two finite blink-clean pupil samples.

## Key validation results

The full file passes format validation with no errors. Validator warnings identify valid trials containing no inferred events in any recorded cell; these sparse zero-event trials are retained for reference fidelity. Full validation balanced accuracies were 0.1581 image identity, 0.5197 image change, 0.2201 running, 0.2131 pupil, and 0.2609 outcome, all above uniform chance.
