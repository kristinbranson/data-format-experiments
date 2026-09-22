# Allen Visual Behavior Neural Decoder Dataset

This project converts the supplied Allen Brain Observatory Visual Behavior 2-photon release into trial-organized data for neural decoding. The final dataset is [`converted_data.pkl`](converted_data.pkl), generated reproducibly by [`convert_data.py`](convert_data.py).

## Dataset scope

- Exact project: `VisualBehavior` (single-plane VISp)
- Active behavior only; passive replay excluded
- Published-QC ROIs and released raw FastLZero L0 calcium events
- Go and catch trials retained; aborted and auto-rewarded trials excluded
- Three active sessions lacking required pupil data excluded
- 165 sessions, 37 mice, 42,470 trials, 28,821 session-neuron channels
- Native ophys timestamp grid, nominally 31 Hz (median bin about 32.31 ms)
- Variable SDK-defined half-open trial windows `[start_time, stop_time)`

## Load the data

```python
import pickle

with open("/app/converted_data.pkl", "rb") as stream:
    data = pickle.load(stream)

# First trial in the first session
neural = data["neural"][0][0]   # (neurons, time)
inputs = data["input"][0][0]    # (0, time): this task specifies no inputs
outputs = data["output"][0][0]  # (5, time)
```

Top-level fields follow the requested schema: `neural`, `input`, and `output` are nested session/trial lists; `subjects`/`subject_idx` identify mice; `brain_regions`/`brain_region_idx` identify VISp; and `metadata` records curation, alignment, processing, and per-session source IDs.

## Decoder outputs

Rows of every `(5, T)` output array are:

1. `image_identity`: `0=gray`; `1-16` are the sorted natural-image names in `output_values[0]`. Images are labeled only during their 250-ms display; ISIs and omissions are gray.
2. `image_change`: `1` throughout the changed-image display, otherwise `0`. Catch sham changes remain zero.
3. `running_speed_bin`: per-session quintiles `0-4`, derived from released filtered running speed interpolated to ophys timestamps.
4. `pupil_diameter_bin`: per-session quintiles `0-4`, derived from processed blink-masked pupil area as `2*sqrt(area/pi)` and interpolated to ophys timestamps.
5. `trial_outcome`: `0=hit`, `1=miss`, `2=false_alarm`, `3=correct_reject`; the static trial label is repeated through time.

There are intentionally no decoder inputs, so every input array has shape `(0, T)` and `input_names` is empty.

## Key statistics and validation

| Statistic | Value |
|-----------|-------|
| Sessions / subjects | 165 / 37 |
| Trials | 42,470 |
| Session-neuron channels | 28,821 |
| Neurons/session | mean 174.67, range 6-666 |
| Trial timepoints | mean 262.02, range 217-389 |
| Trial outcomes | 13,569 hit; 23,574 miss; 814 false alarm; 4,513 correct reject |
| Final pickle size | 7.816 GiB |

The provided validator completed successfully. It warns about 1,729 all-zero event trials (4.07%), all in low-cell-count sessions; independent raw-NWB checks confirm these are genuine sparse L0 event slices, so they are retained rather than applying an unsupported activity filter.

Full validation balanced accuracies were: image identity 0.1606 (chance 0.0588), image change 0.5832 (0.5), running bin 0.2183 (0.2), pupil bin 0.2143 (0.2), and trial outcome 0.2607 (0.25). Training loss decreased from 1.6332 to 1.5812.

See [`CONVERSION_NOTES.md`](CONVERSION_NOTES.md) for source reconciliation, decisions, raw `np.allclose` audits, warning analysis, and accuracy review.

## Reproduce

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/sample_data.pkl --verify-only
python -u /app/train_decoder.py /app/sample_data.pkl

python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

Required run logs are retained in `/app`. Processing audits for the final two-session sample are `processing_844395446.png` and `processing_845037476.png`; full decoder visualizations are `sample_trials.png` and `predictions.png`.
