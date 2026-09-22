# Allen Visual Behavior Ophys Decoder Dataset

## Overview

`converted_data.pkl` contains neural and behavioral data from the locally supplied Allen Brain Observatory Visual Behavior 2P cache, converted for neural decoding. Data were loaded exclusively through AllenSDK's `VisualBehaviorOphysProjectCache`; NWB files were not opened directly.

The retained cohort contains active go/no-go change-detection recordings only. Passive replay sessions were excluded because they do not have meaningful behavioral outcomes. Three active experiments without eye tracking were also excluded because pupil diameter is a required output.

## Key Statistics

- 199 ophys experiments (imaging planes) from 171 unique ophys sessions
- 38 mice
- 29,168 valid experiment-cell records
- 51,075 plane-trials
- 15,682 hit, 28,990 miss, 920 false-alarm, and 5,483 correct-reject trials
- Brain regions: VISp (29,006 cells) and VISl (162 cells)
- 100 ms bins; variable trial duration from AllenSDK `start_time` to `stop_time`
- Native source rates include ~31 Hz single-plane and ~10.7 Hz multiplane recordings

## Loading

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(len(data['neural']))          # 199 sessions
print(data['output_names'])
print(data['neural'][0][0].shape)   # neurons x timepoints
print(data['output'][0][0].shape)   # 5 x timepoints
```

The pickle is approximately 2.9 GB, so loading requires sufficient RAM.

## Data Structure

- `neural[session][trial]`: `float32` matrix `(n_neurons, n_timepoints)`. Values are AllenSDK filtered calcium-event magnitudes summed in 100 ms bins.
- `input[session][trial]`: empty `float32` matrix `(0, n_timepoints)`, because this decoder task specifies no inputs.
- `output[session][trial]`: integer matrix `(5, n_timepoints)`:
  1. `image_identity`: 0 for gray/omitted; 1-16 for named natural images.
  2. `image_change`: one-bin pulse immediately at a real image-identity change.
  3. `running_speed_quintile`: 0-4, using within-experiment percentile edges.
  4. `pupil_diameter_quintile`: 0-4, using within-experiment percentile edges.
  5. `trial_outcome`: 0 hit, 1 miss, 2 false alarm, 3 correct reject. This static value is repeated across time for a uniform output matrix.
- `subjects` and `subject_idx`: mouse identifiers and per-session lookup.
- `brain_regions` and `brain_region_idx`: region labels and per-neuron lookup.
- `metadata['session_info']`: source experiment/session IDs, frame rate, trial/cell counts, and behavior percentile edges.

## Processing Summary

1. Enumerate locally available experiments through cache metadata.
2. Keep active session types and exclude files lacking mandatory eye tracking.
3. Keep SDK `go | catch` trials, thereby excluding aborted and auto-rewarded attempts.
4. Use SDK-valid ROIs and SDK `filtered_events` to match the paper's detected-event analysis.
5. Align all streams by absolute timestamps to 100 ms bins anchored at trial start.
6. Assign image identity by stimulus interval membership and image-change pulses by exact onset.
7. Interpolate running speed and blink-cleaned pupil diameter to bin centers, then discretize into five within-experiment percentile bins.

See `CONVERSION_NOTES.md` for detailed rationale, source statistics, validation, independent `np.allclose()` checks, and decoder results. Reproduce conversion with:

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full
```

For a two-session test with processing plots:

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
```

## Validation and Decoder Results

`train_decoder.py --verify-only` completed without errors. Its only warnings are source-faithful all-zero event trials from sparse neural populations; these were retained rather than fabricating activity or filtering trials based on neural response.

Full validation balanced accuracies:

| Output | Accuracy | Chance |
|--------|----------|--------|
| Image identity | 0.2909 | 0.0588 |
| Image change | 0.5630 | 0.5000 |
| Running quintile | 0.2531 | 0.2000 |
| Pupil quintile | 0.2573 | 0.2000 |
| Trial outcome | 0.2907 | 0.2500 |

Training completed successfully and loss decreased over 200 epochs.
