# Conversion Notes

## Source and scope

- Paper: `paper.pdf`
- Methods excerpt: `methods.txt`
- Reference code: `code/georepca1/src/utils.py`
- Source data used for conversion: the animal-level joblib files in `data/QLAK-CA1-*`

The joblib files are the repository's own converted form of the original MATLAB data, loaded by the reference helper `load_dat(..., format="joblib")`. They preserve the paper fields directly: `envs`, `blocked`, `position`, `trace`, `maps`, `SFPs`, and `centroids`.

## Reference-processing choices mirrored here

- Sessions are recording days. This yields 207 sessions total, matching the paper and repository summaries.
- Neural activity uses the paper's rise-extracted binary calcium event traces, not deconvolved or re-smoothed traces.
- Unregistered neurons are removed session-by-session by dropping rows that are all `NaN` on that day.
- Subject ordering follows the repository animal IDs:
  - `QLAK-CA1-08`
  - `QLAK-CA1-30`
  - `QLAK-CA1-50`
  - `QLAK-CA1-51`
  - `QLAK-CA1-56`
  - `QLAK-CA1-74`
  - `QLAK-CA1-75`

## Decoder-specific formatting decisions

These steps are required by the target decoder task rather than directly provided by the paper.

### Trialization

- The recordings are nominally 40 min at 30 Hz.
- I split each day into 40 consecutive nominal 1 min trials of 1800 frames each.
- Sessions shorter than 72000 frames keep a shorter final trial.
- Sessions longer than 72000 frames are truncated at the nominal 40 min boundary.

Observed consequences:

- 93 sessions have a short final trial.
- Minimum trial length: 1666 frames.
- Maximum trial length: 1800 frames.
- Total dropped trailing frames from recordings longer than 40 nominal minutes: 11,481.

### Geometry input

- Decoder input is a 9D static blocked-partition mask per trial.
- The mask uses the dataset's bottom-up row-major partition numbering, which is the same convention encoded by `blocked`.
- Input name order is `blocked_bin_0` through `blocked_bin_8`.

### Spatial output

- Decoder output is one categorical variable, `spatial_bin`, with 9 classes.
- Class IDs are bottom-up row-major: `bin_id = y_bin * 3 + x_bin`.
- I calibrated each animal's absolute 3x3 arena frame from that animal's square sessions:
  - arena origin: minimum square-session position on each axis
  - arena side length: maximum square-session extent
  - resulting side length: exactly `75.0` for all seven animals
  - resulting bin size: `25.0`

This preserves translated geometries such as `rectangle`, which would be misaligned by per-session min-shifting.

### Geometry-consistency cleaning

- After absolute 3x3 binning, any frame that landed in a blocked partition was snapped to the nearest open partition center.
- This mirrors the reference code's geometry masking of impossible spatial bins in rate maps.
- Total corrected frames: 448 across the full dataset.

## Sanity checks against the paper and code

### Dataset-wide counts

- Unique neurons across animals: 5,413
- Sessions: 207
- Session-cell observations after per-day registration filtering: 69,744

These match the paper/methods text and the repository outputs:

- `methods.txt` reports 5,413 unique neurons across 207 sessions forming 69,744 rate maps.
- The repository's `df_shr_pvals` file has shape `(69744, 4)`, consistent with 69,744 session-cell observations.

### Session structure

- Sessions per subject:
  - `QLAK-CA1-08`: 31
  - `QLAK-CA1-30`: 31
  - `QLAK-CA1-50`: 31
  - `QLAK-CA1-51`: 21
  - `QLAK-CA1-56`: 31
  - `QLAK-CA1-74`: 31
  - `QLAK-CA1-75`: 31
- Trials per session: always 40
- Mean neurons per session after removing unregistered rows: 336.93
- Range of session neuron counts: 113 to 564

### Geometry checks

- The environment counts in the converted dataset are:
  - `square`: 27
  - every other geometry (`o`, `t`, `u`, `rectangle`, `+`, `i`, `l`, `bit donut`, `glenn`): 20
- `blocked` values were checked against the environment labels and matched the expected geometry masks.
- The input feature `blocked_bin_7` is always zero because none of the 10 experimental geometries occlude that partition under the dataset's numbering convention.

### Reference decoding cross-check

- The repository ships within-session decoding results in `data/precomputed_results/within_decoding`.
- Animal-level mean decoding errors from that file are:
  - `QLAK-CA1-08`: 16.486
  - `QLAK-CA1-30`: 12.833
  - `QLAK-CA1-50`: 11.425
  - `QLAK-CA1-51`: 17.567
  - `QLAK-CA1-56`: 15.160
  - `QLAK-CA1-74`: 11.385
  - `QLAK-CA1-75`: 10.893

I did not try to reproduce those exact numbers because the target task changes the readout to 9 categorical 3x3 bins and introduces artificial 1 min trials, but I used the same underlying traces, session definitions, and geometry conventions.

## Validation results

### Conversion

`conversion_full_out.txt` reports:

- animals: 7
- sessions: 207
- trials: 8,280
- unique neurons across animals: 5,413
- session neuron observations: 69,744
- trial lengths: 1666 to 1800
- short final trials: 93
- dropped trailing frames: 11,481
- snapped blocked-bin frames: 448

### Official format verification

- Full dataset: `verification_full_out.txt`
  - `train_decoder.py --verify-only` passed with no errors or warnings.
- Sample dataset: `verification_sample_out.txt`
  - `train_decoder.py --verify-only` passed with no errors or warnings.

### Sample decoder training

`train_decoder_sample_out.txt`:

- Train trials: 640
- Validation trials: 160
- Balanced accuracy, train: `0.4464`
- Balanced accuracy, validation: `0.4032`
- Chance level for 9 classes: `0.1111`

This is a strong signal that the neural/activity alignment, geometry input, and spatial output formatting are coherent.

### Full decoder training log

- `train_decoder_full_out.txt` was created by launching `python /app/train_decoder.py /app/converted_data.pkl`.
- The run successfully completed format verification and entered model training.
- At the time these notes were finalized, the log had reached SVD initialization through session 118 and had not yet emitted epoch-loss lines.
- This does not affect the converted datasets or the completed verification passes above; it only means the long full-dataset training job had not finished within the interactive work window.

## Sample dataset

- `sample_data.pkl` contains the first 10 sessions from `QLAK-CA1-08` and the first 10 sessions from `QLAK-CA1-51`.
- Total sample sessions: 20
- Total sample trials: 800

## Files created

- `convert_data.py`
- `converted_data.pkl`
- `sample_data.pkl`
- `README.md`
- `CONVERSION_NOTES.md`
- `conversion_full_out.txt`
- `conversion_sample_out.txt`
- `verification_full_out.txt`
- `verification_sample_out.txt`
- `train_decoder_sample_out.txt`
- `train_decoder_full_out.txt`
