# Conversion Notes

## Goal

Convert the CA1 miniscope dataset from Lee, Keinath, Cianfarano, and Brandon into the decoder format used by `train_decoder.py`, while preserving the paper/code preprocessing and only changing what was necessary for the requested decoder task.

## Reference-aligned choices

1. Session definition
   - I treated each original recording day as one decoder session.
   - This matches the paper code, which iterates over day/session axes in `trace`, `position`, and `envs`.

2. Neural signal
   - I used the provided binary rise-event traces directly from `data/<animal>`.
   - I did not re-deconvolve, smooth, or re-threshold the calcium traces.
   - This matches the paper/methods description: all analyses use the binarized rising phase of calcium transients.

3. Neuron inclusion
   - For each session, I kept only neurons registered on that day.
   - In the source files, unregistered neurons are all-`NaN` for that day; these were dropped from that session.
   - No additional place-cell or activity-threshold filtering was applied to the exported neural arrays.
   - This keeps the exported dataset faithful to the recorded session content while avoiding `NaN`s, which the decoder format rejects.

4. Subjects and region labels
   - Subjects are the seven animal IDs:
     - `QLAK-CA1-08`
     - `QLAK-CA1-30`
     - `QLAK-CA1-50`
     - `QLAK-CA1-51`
     - `QLAK-CA1-56`
     - `QLAK-CA1-74`
     - `QLAK-CA1-75`
   - `brain_regions = ["CA1"]` and every per-session `brain_region_idx` is all zeros.

## Decoder-specific adaptations

1. Trialization
   - The source recordings are continuous long sessions rather than pre-segmented trials.
   - I split each session into contiguous full 60 s windows.
   - With 30 Hz sampling, each trial is `1800` time bins.
   - Sessions produced either `39` or `40` complete one-minute trials depending on raw recording length.
   - Any trailing incomplete remainder was discarded.

2. Decoder input
   - Input is a static 9D vector per trial: a flattened `3 x 3` open-bin mask.
   - `1` means the arena partition is open, `0` means blocked.
   - I derived this from the `blocked` metadata, not only from the environment name.
   - This matters because the `env` string alone does not fully specify orientation in this dataset, while `blocked` does.

3. Decoder output
   - Output is a single categorical variable `position_bin` with values `0..8`.
   - Label definition is `x_bin * 3 + y_bin`.
   - Output labels are:
     - `x0_y0`, `x0_y1`, `x0_y2`,
     - `x1_y0`, `x1_y1`, `x1_y2`,
     - `x2_y0`, `x2_y1`, `x2_y2`

4. Position discretization
   - I matched the paper code’s binning style: no min subtraction, divide raw positions by per-axis maxima, then floor.
   - Concretely:
     - `floor(position / ((per-axis max + eps) / 3))`
   - This is the same scaling logic used by the reference code for spatial binning before rate-map construction.

5. Coarse-bin cleanup
   - After 3x3 binning, some time points fell into blocked coarse bins.
   - For those frames only, I reassigned the label to the nearest valid open bin in the same `3 x 3` geometry.
   - This is a decoder-task adaptation, but it is in the spirit of the paper’s within-session decoder code, which also snaps decoded/actual positions to valid bins for geometry-aware error measurement.

## Important dataset detail discovered during conversion

The authoritative geometry/orientation information for this decoder task is the `blocked` field.

Observed `blocked` patterns per environment:

- `square`: `[]`
- `o`: `[4]`
- `t`: `[3, 5, 6, 8]`
- `u`: `[4, 5]`
- `rectangle`: `[0, 3, 6]`
- `+`: `[0, 2, 6, 8]`
- `i`: `[3, 5]`
- `l`: `[1, 2, 4, 5]`
- `bit donut`: `[0, 4]`
- `glenn`: `[0, 8]`

This was the main non-obvious point in the conversion. Using only canonical environment names would have mis-specified some geometries.

## Sanity checks

### Paper-level count checks

These matched exactly:

- Sessions: `207`
- Unique neurons: `5413`
- Session-by-neuron rate maps: `69744`

These three values are the strongest sanity checks that session loading and per-day neuron selection match the reference dataset.

### Session/frame checks

- Unique raw session frame lengths:
  - `71866` frames: `93` sessions
  - `72060` frames: `31` sessions
  - `72071` frames: `31` sessions
  - `72091` frames: `31` sessions
  - `72219` frames: `21` sessions
- Trial length after conversion: always `1800` frames
- Trials per session: `39` or `40`
- Total converted trials: `8187`
- Mean discarded tail per session: `26.80` s

### Environment counts

- `square`: `27`
- each non-square geometry: `20`

### Verification checks

Both exported files passed `train_decoder.py --verify-only` with no errors or warnings.

### Decoder performance

Sample dataset:

- Training balanced accuracy: `0.5528`
- Validation balanced accuracy: `0.4748`
- Uniform chance: `0.1111`
- Majority-class chance: `0.2161`

Full dataset:

- Training balanced accuracy: `0.6395`
- Validation balanced accuracy: `0.5729`
- Uniform chance: `0.1111`
- Majority-class chance: `0.2146`

These are comfortably above chance and are consistent with the converted geometry/position structure being usable by the provided decoder.

## Exported files

- `converted_data.pkl`
  - Full dataset.
- `sample_data.pkl`
  - First `10` sessions from `QLAK-CA1-08` and first `10` sessions from `QLAK-CA1-30`.
  - `20` sessions total, `780` trials total.
- `convert_data.py`
  - Reproducible conversion script.

## Notes on artifacts

- `geometry_x2_y1` is always `1` in the exported input vectors. This reflects the source geometry definitions rather than a conversion bug.
- `total_frames_reassigned_to_valid_bins = 2555747`. This sounds large in absolute terms because the dataset is large; across plausible normalization schemes the invalid-bin rate was effectively unchanged, so this is a property of coarse `3 x 3` discretization rather than a hidden loading error.
