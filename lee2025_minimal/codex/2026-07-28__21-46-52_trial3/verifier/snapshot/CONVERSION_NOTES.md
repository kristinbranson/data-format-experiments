# Conversion Notes

## Source alignment

Reference materials used:
- `paper.pdf`
- `methods.txt`
- `code/georepca1/src/utils.py`
- `code/README.md`

Key reference facts matched by the converted dataset:
- 7 subjects
- 207 sessions total
- 5,413 unique tracked neurons across animals
- 69,744 session-specific CA1 neuron maps / registered cell-session instances
- 10 geometries with counts:
  - `square`: 27
  - every other geometry: 20

These counts were verified directly from the source `.mat` files and again after conversion.

## Conversion choices

Session ordering:
- Sessions are ordered by animal in the repository order:
  - `QLAK-CA1-08`, `QLAK-CA1-30`, `QLAK-CA1-50`, `QLAK-CA1-51`, `QLAK-CA1-56`, `QLAK-CA1-74`, `QLAK-CA1-75`
- Within each animal, session order is preserved exactly as stored in the source files.

Neural data:
- Source traces are the paper's rise-extracted binary calcium-event vectors.
- Cells not registered on a given session are dropped from that session.
- I kept all registered cells so the converted data preserves the published `69,744` session-level cell count.
- For decoder-ready time series, traces are:
  - restricted to movement frames,
  - Gaussian-smoothed with `sigma=3` frames,
  - average pooled in non-overlapping 3-frame windows.

Behavior / outputs:
- Position uses the stored session-aligned DLC trajectories.
- Movement frames are selected with the same rule used by the paper's within-session position decoder:
  - Gaussian-smoothed speed threshold `> 5 cm/s`
- Position is converted to a coarse 3x3 grid using a session-wise common spatial scale based on the session's maximum coordinate extent.
- Output is stored as one categorical variable:
  - `position_bin` with 9 values `r0c0 ... r2c2`

Inputs:
- The decoder input is a static 3x3 environment-open mask.
- I used the source `blocked` field rather than a geometry-name template, because asymmetric environments can appear in session-specific orientations.
- Input values are `1=open`, `0=blocked`.

Trialization:
- Sessions are split into contiguous 1-minute windows from session start.
- Because source frame counts are slightly short of exactly 72,000 frames, the final raw window can be shorter than 60 s.
- Trials with no movement frames or fewer than 10 pooled time bins are dropped.
- This still leaves at least 24 trials in every session, and usually about 40.

Blocked-bin remapping:
- Temporal pooling can occasionally push a pooled sample into a blocked coarse bin.
- Those samples are remapped to the nearest valid open bin.
- This mirrors the spirit of the paper's decoder cleanup, which snaps position estimates to valid occupied bins.

## Dataset summary

Full dataset:
- Sessions: 207
- Trials: 8,266
- Session-level neuron count total: 69,744
- Trial timepoints after movement selection and pooling:
  - min: 10
  - median: 322
  - max: 560
- Time bin size: 100 ms
- Total kept moving frames before pooling: 7,717,198
- Remapped pooled samples from blocked to open bins: 441,410

Sample dataset:
- Sessions: 11
- Trials: 438
- Session-level neuron count total: 2,061
- Built from the first full geometry sequence of `QLAK-CA1-08`

Output distribution on full dataset:
- `r0c0`: 386,924
- `r0c1`: 189,095
- `r0c2`: 368,565
- `r1c0`: 179,748
- `r1c1`: 218,022
- `r1c2`: 230,155
- `r2c0`: 207,445
- `r2c1`: 308,904
- `r2c2`: 480,526

## Validation

Format verification:
- Sample: passed with no warnings
- Full: passed with no warnings

Decoder training:
- Sample validation balanced accuracy:
  - `position_bin`: `0.5929`
  - uniform chance: `0.1111`
- Full validation balanced accuracy:
  - `position_bin`: `0.7084`
  - uniform chance: `0.1111`

The corresponding command outputs are saved in:
- `verification_sample_out.txt`
- `train_decoder_sample_out.txt`
- `verification_full_out.txt`
- `train_decoder_full_out.txt`

## Sanity checks performed

- Confirmed the session count mismatch suspicion was false:
  - `QLAK-CA1-51` truly contains 21 sessions, giving the paper's total of 207.
- Confirmed the sum of animal-level tracked cell counts is exactly 5,413.
- Confirmed the sum of per-session registered neurons is exactly 69,744.
- Confirmed geometry histogram matches the paper-level totals.
- Confirmed full and sample data pass `train_decoder.py --verify-only`.
- Confirmed the provided decoder trains successfully and greatly exceeds chance on both sample and full datasets.

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
- `train_decoder_full_out.txt`
- `train_decoder_sample_out.txt`
