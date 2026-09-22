# Allen Visual Behavior 2P Decoder Dataset

This directory contains a decoder-ready conversion of the supplied Allen Brain Observatory Visual Behavior 2P release. Each target session is one imaging plane/NWB experiment. The dataset contains active-task go and catch trials; aborted and auto-rewarded trials are excluded.

## Main Files

- `converted_data.pkl`: complete converted dataset (199 sessions, 48,909 trials, 29,168 session-neurons, 38 mice)
- `sample_data.pkl`: two-session test conversion
- `convert_data.py`: reproducible converter
- `CONVERSION_NOTES.md`: source review, decisions, statistics, and validation record
- `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt`: complete-run logs
- `processing_*.png`, `sample_trials.png`, `predictions.png`: alignment and decoder diagnostics

## Loading

```python
import pickle

with open("/app/converted_data.pkl", "rb") as stream:
    data = pickle.load(stream)

neural_trial = data["neural"][0][0]   # (neurons, time), float32
input_trial = data["input"][0][0]     # (0, time): no decoder inputs requested
output_trial = data["output"][0][0]   # (5, time), int16 categorical codes
```

The full pickle is approximately 7.35 GiB, so loading it requires correspondingly large RAM. Trials retain their native variable durations but share complete 33.333-ms bins. Bin centers are aligned on the hardware-synchronized ophys clock.

## Output Rows

| Row | Name | Codes |
|-----|------|-------|
| 0 | Image identity | 0 = gray/omission; 1--16 correspond to names in `output_values[0]` |
| 1 | Image change | 1 for the first bin center at/after a true image-identity change; otherwise 0 |
| 2 | Running speed quintile | 0--4, using per-session percentiles of released processed speed |
| 3 | Pupil diameter quintile | 0--4, using per-session percentiles of blink-filtered major-axis diameter |
| 4 | Trial outcome | 0 hit, 1 miss, 2 false alarm, 3 correct reject; constant within a trial |

Neural values are released FastLZero detected calcium-event magnitudes linearly interpolated to the common 30-Hz grid. Released valid ROIs are retained. `brain_region_idx` maps cells to VISp or VISl. Detailed per-session IDs, raw trial IDs, exclusions, percentile edges, and acquisition metadata are in `metadata["session_info"]`.

## Key Statistics

| Statistic | Value |
|-----------|-------|
| Sessions | 199 imaging planes from 171 unique recordings |
| Subjects | 38 |
| Session-neurons | 29,168 (4--666/session; mean 146.57) |
| Retained trials | 48,909 (39--403/session) |
| Outcomes | 15,209 hit; 27,582 miss; 886 false alarm; 5,232 correct reject |
| Brain regions | VISp 29,006 cells; VISl 162 cells |
| Trial duration | 210--376 bins (7.00--12.53 s) |

Three active experiments lacking eye-tracking data are excluded because pupil diameter is a required target. A further 2,166 go/catch trials intersect irrecoverable long or edge pupil gaps and are documented by raw ID. The verifier reports 2,429 all-zero neural trials; independent raw-NWB reconstruction confirms these are genuine silent intervals in sparse event traces, not missing conversion data.

## Reproducing

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/sample_data.pkl --verify-only
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

Full validation balanced accuracies were 0.1921 (image identity), 0.5309 (image change), 0.2226 (running), 0.2207 (pupil), and 0.2654 (outcome), all above their respective chance levels. See `CONVERSION_NOTES.md` for interpretation and direct raw-data audits.
