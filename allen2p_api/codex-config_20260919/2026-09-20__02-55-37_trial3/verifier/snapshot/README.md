# Allen Visual Behavior Neural Decoder Dataset

This directory contains a decoder-ready conversion of the locally cached Allen Brain Observatory Visual Behavior 2P release 1.1.0. The conversion uses the AllenSDK `VisualBehaviorOphysProjectCache` exclusively. It includes active go and catch trials, excludes aborted and auto-rewarded trials, and omits three active experiments that have no processed pupil stream.

## Files

- `converted_data.pkl`: full dataset (199 imaging-plane sessions, 38 mice, 51,075 trials, 29,168 experiment-cell entries)
- `sample_data.pkl`: two-session test conversion
- `convert_data.py`: reproducible converter
- `CONVERSION_NOTES.md`: complete decisions, paper/code comparisons, checks, and decoder results
- `conversion_*_out.txt`, `verification_*_out.txt`, `train_decoder_*_out.txt`: run logs
- `processing_<id>.png`, `sample_trials.png`, `predictions.png`: processing/training diagnostics

## Load and inspect

```python
import pickle

with open("/app/converted_data.pkl", "rb") as stream:
    data = pickle.load(stream)

print(len(data["neural"]), data["output_names"])
neural = data["neural"][0][0]   # cells x time, float32 event magnitudes
outputs = data["output"][0][0]  # 5 x time, categorical integer codes
```

Each target session is one ophys experiment/imaging plane. Trials have variable duration but use uniform 100 ms bins. `input` has shape `(0,T)` because the task specifies no decoder inputs. Output rows are image identity, changed-image interval, running-speed quintile, pupil-diameter quintile, and trial outcome. Interpret integer codes with `output_values`; inspect thresholds and per-session source IDs in `metadata`.

Neural data are SDK-provided unfiltered FastLZero inferred calcium-event magnitudes summed in each bin. Running uses SDK-filtered speed. Pupil is equivalent circular diameter derived from the processed/masked ellipse area. Image and change labels use active display intervals on the synchronized clock.

## Reproduce and validate

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python /app/train_decoder.py /app/converted_data.pkl --verify-only
python /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

Full validation has no structural errors. It warns that 4.90% of trials contain no inferred events; these are genuine sparse-event trials, concentrated in low-cell experiments, and are retained to avoid neural-activity-dependent trial selection.

Full validation balanced accuracies are 0.2557 image identity, 0.6112 image change, 0.2883 running quintile, 0.3025 pupil quintile, and 0.2760 trial outcome; all exceed balanced chance.
