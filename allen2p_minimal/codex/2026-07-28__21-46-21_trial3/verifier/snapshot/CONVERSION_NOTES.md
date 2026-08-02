# Conversion Notes

## Scope

This conversion targets the Allen Visual Behavior two-photon ophys data available locally in `/app/data`, using the AllenSDK code vendored in `/app/code` plus the experiment/task descriptions in `whitepaper.pdf`, `paper.pdf`, and `methods.txt`.

The requested decoder dataset was built from the Visual Behavior change-detection task only. Temporal alignment is based on ophys timestamps, and trials are segmented using the AllenSDK `trials` table.

## Reference-Matched Decisions

### Session unit

Each decoder "session" is one Allen `BehaviorOphysExperiment` NWB file, not one unique behavior session. This matches the AllenSDK data model and avoids incorrectly merging different imaging planes from multiscope sessions into one neuron matrix. As a result:

- 199 converted decoder sessions correspond to 171 unique `ophys_session_id` values.
- The local active cache contains 202 experiment files across 174 unique behavior sessions.

### Local data subset

Only NWB files physically present under:

`/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments`

were used. The local cache contains:

- 284 total experiment files
- 202 active experiment files
- 174 unique active behavior/ophys sessions
- 38 mice

This is smaller than the full dataset described in the paper, so paper-level totals such as 376 imaging sessions / 82 mice cannot be matched exactly from the local copy.

### Active-task filtering

The whitepaper and paper distinguish active change-detection sessions from passive viewing. The converter keeps only experiments with `passive == False`, which locally yields:

- `OPHYS_1_images_A`: 54 converted sessions
- `OPHYS_3_images_A`: 55 converted sessions
- `OPHYS_4_images_B`: 45 converted sessions
- `OPHYS_6_images_B`: 45 converted sessions

Project-code coverage after conversion:

- `VisualBehavior`: 165 sessions
- `VisualBehaviorMultiscope`: 34 sessions

Experience-level coverage after conversion:

- `Familiar`: 109 sessions
- `Novel 1`: 37 sessions
- `Novel >1`: 53 sessions

### Trial inclusion

Per instructions and Allen trial definitions, trials are kept only when:

- `go == True` or `catch == True`
- `aborted == False`
- `auto_rewarded == False`

This retains the experimentally meaningful go/catch trials while excluding premature-lick resets and free-reward trials.

### Neural signal

Neural activity uses AllenSDK discrete calcium events from `dataset.events["events"]`.

This matches the paper's statement that neural analyses were performed on detected calcium events rather than raw fluorescence. The converter intentionally does not use dF/F traces and does not use `filtered_events`.

AllenSDK default ROI validity filtering is left intact, so invalid ROIs are excluded automatically when loading each `BehaviorOphysExperiment`.

### Temporal alignment and binning

Trials are segmented from `trial.start_time` to `trial.stop_time`. Within each trial:

- a common 100 ms bin size is used for every session
- bin timestamps are bin centers
- neural samples are aligned by nearest ophys timestamp
- running speed is linearly interpolated to those timestamps
- pupil diameter is linearly interpolated to those timestamps

The local dataset mixes single-plane and multiscope experiments. Their native frame intervals differ:

- median native frame interval: 32.32 ms
- min: 32.31 ms
- max: 93.23 ms

Using 100 ms bins preserves compatibility across both acquisition modes while staying close to the slower multiscope sampling interval.

### Stimulus labels

Stimulus labels come only from `stimulus_presentations` rows whose `stimulus_block_name` contains `change_detection`.

For each trial:

- `image_identity` is the image shown during flashed image epochs
- all other times are labeled `gray`
- omissions remain `gray`
- `image_change` is `1` only during flashed epochs with `is_change == True`, else `0`

This matches the task structure described in the references: 250 ms image flash followed by 500 ms gray, with 5% omissions and no omissions on the change image or the immediately preceding image.

### Behavioral outputs

The decoder outputs are:

1. `image_identity`
2. `image_change`
3. `running_speed_bin`
4. `pupil_diameter_bin`
5. `trial_outcome`

`trial_outcome` is static within each trial and encoded as:

- `0`: `hit`
- `1`: `miss`
- `2`: `false_alarm`
- `3`: `correct_reject`

Running and pupil are discretized into global quintiles over all kept trial time bins in the converted dataset.

Pupil diameter is computed as equivalent diameter from AllenSDK `pupil_area`:

`diameter = 2 * sqrt(area / pi)`

## Exclusions

No sessions were dropped for missing events or too few valid trials. Three experiment files were excluded because pupil data were entirely invalid:

- `795953296`
- `806456687`
- `833631914`

Final converted dataset:

- 199 experiment-level sessions
- 171 unique behavior/ophys sessions
- 38 subjects
- 51,075 trials
- 29,168 neurons

Brain-region composition of the local subset is heavily V1-skewed:

- `VISp`: 29,006 neurons
- `VISl`: 162 neurons

## Sanity Checks

### Structural checks

`train_decoder.py --verify-only` passes on both the full and sample datasets.

- `/app/verification_full_out.txt`
- `/app/verification_sample_out.txt`

The decoder-format structure is valid, with:

- empty decoder inputs represented as `(0, T)` arrays
- 5 categorical outputs
- at least 2 trials per kept session

### Distribution checks

Full-dataset output fractions:

- `image_identity`: `gray` = 0.668
- `image_change`: `change` = 0.027
- `running_speed_bin`: each bin = 0.200
- `pupil_diameter_bin`: each bin = 0.200
- `trial_outcome`: `hit` 0.303, `miss` 0.571, `false_alarm` 0.017, `correct_reject` 0.108

Important consistency observations:

- The `gray` fraction is close to the expected 2/3 from the 250 ms image + 500 ms gray task cadence.
- `image_change` is rare, as expected for the roving-baseline change-detection design.
- Running and pupil bins are exactly balanced because they are defined by global percentiles.

### Sample decoder training

The sample dataset is a fixed representative subset spanning both project codes, both local brain regions, and all four active session types. It contains 11 sessions, 1,986 trials, and 1,133 neurons.

Sample experiment IDs:

- `775614751`
- `792813858`
- `794381992`
- `795073741`
- `795076128`
- `788490510`
- `796105304`
- `951980475`
- `958527485`
- `957759566`
- `960410038`

Sample training results from `/app/train_decoder_sample_out.txt`:

- validation `image_identity`: 0.1705 vs 0.0588 chance
- validation `image_change`: 0.5947 vs 0.5000 chance
- validation `running_speed_bin`: 0.2258 vs 0.2000 chance
- validation `pupil_diameter_bin`: 0.2352 vs 0.2000 chance
- validation `trial_outcome`: 0.2468 vs 0.2500 chance

This indicates that stimulus identity, stimulus change, running, and pupil carry usable signal in the converted neural data. Trial outcome is near chance on the small representative subset.

### Full decoder training

Full training on `/app/converted_data.pkl` completed successfully. Final validation balanced accuracy from `/app/train_decoder_full_out.txt`:

- validation `image_identity`: 0.1488 vs 0.0588 chance
- validation `image_change`: 0.5694 vs 0.5000 chance
- validation `running_speed_bin`: 0.2295 vs 0.2000 chance
- validation `pupil_diameter_bin`: 0.2467 vs 0.2000 chance
- validation `trial_outcome`: 0.2582 vs 0.2500 chance

The full dataset therefore preserves decodable information for the main time-varying stimulus and behavior variables, with trial-outcome decoding only slightly above chance.

## Warnings and Caveats

`train_decoder.py` reports warnings for some trials whose neural event matrices are entirely zero. These are not format errors. They arise when valid event traces contain no detected events during a given trial window, which is plausible for sparse event representations, especially in low-neuron multiscope planes.

The converted dataset is intentionally experiment-plane based, not cross-plane merged. That choice is important for correctness and is the main reason the number of converted decoder sessions exceeds the number of unique behavior sessions.

## Artifacts

Created files:

- `/app/convert_data.py`
- `/app/converted_data.pkl`
- `/app/sample_data.pkl`
- `/app/conversion_full_out.txt`
- `/app/conversion_sample_out.txt`
- `/app/verification_full_out.txt`
- `/app/verification_sample_out.txt`
- `/app/train_decoder_sample_out.txt`
- `/app/train_decoder_full_out.txt`
- `/app/CONVERSION_NOTES.md`
- `/app/README.md`

The full decoder training run is recorded in `/app/train_decoder_full_out.txt`.
