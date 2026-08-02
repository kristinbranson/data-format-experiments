# Conversion Notes

## References Used

- `whitepaper.pdf`: Allen Brain Observatory Visual Behavior 2P technical whitepaper.
- `paper.pdf` and `methods.txt`: especially the description that behavior is analyzed by assigning events to each `750 ms` image-presentation interval.
- AllenSDK code under `code/`, using `BehaviorOphysExperiment`.

## Scope and Local Data Caveat

The local workspace does not contain the full Allen release. It contains:

- 284 NWB experiment files present locally
- 202 active-behavior experiments selected from those files
- 38 mice
- 2 recorded areas in the local subset: `VISp` and `VISl`

Because the local subset is incomplete, final counts here do not match paper-wide counts from the full public release.

## Core Processing Decisions

### Session definition

One decoder session is one `ophys_experiment_id`.

Reason:

- An `ophys_experiment_id` is one imaging plane with one ophys timestamp stream.
- This keeps neural timestamps native and avoids merging planes with different neuron sets.

### Trial selection

Included trials satisfy all of:

- `go` or `catch`
- not `aborted`
- not `auto_rewarded`
- finite `change_time`

This matches the prompt and the Allen trial table semantics.

### Time axis

Each trial is represented as a variable-length sequence of native `750 ms` image-presentation intervals.

Reason:

- The whitepaper and paper describe the task as `250 ms` flashed images plus `500 ms` gray.
- The paper’s behavioral processing explicitly assigns events to each `750 ms` image-presentation interval.
- AllenSDK provides trial boundaries and stimulus presentation onset times directly, so interval-based binning is the closest match to the references while still preserving trial structure.

Implementation:

- For each kept trial, stimulus intervals were taken from `stimulus_presentations` rows in the change-detection block whose `start_time` falls within the AllenSDK trial `start_time` to `stop_time`.
- Bin edges are the interval start times plus one final edge at `last_start_time + 0.75`.
- All neural and behavioral streams are reduced onto these interval bins using ophys / behavior timestamps.

### Neural signal

- Used `dataset.events["events"]` from AllenSDK.
- Invalid ROIs are excluded by loading with `exclude_invalid_rois=True`.
- Trials with non-finite binned values or all-zero neural activity were dropped.

This follows the paper’s use of inferred/discrete calcium events rather than raw fluorescence.

### Running speed

- Used AllenSDK `running_speed["speed"]`.
- Binned by interval mean.
- Discretized into 5 global quantile bins across the full kept dataset.

### Pupil diameter

- Used AllenSDK `eye_tracking["pupil_width"]` as the pupil-size measure.
- Missing values were linearly interpolated in timestamp space before interval binning.
- Binned by interval mean.
- Discretized into 5 global quantile bins across the full kept dataset.

### Image identity output

- Output is interval-wise.
- For non-omitted intervals, the label is the flashed image name.
- For omitted intervals, the label is `gray`.

Final image label set:

- `gray`
- `im000`, `im031`, `im035`, `im045`, `im054`, `im061`, `im062`, `im063`, `im065`, `im066`, `im069`, `im073`, `im075`, `im077`, `im085`, `im106`

### Image change output

- Binary interval-wise output from the Allen stimulus table `is_change`.
- `1` only on true change intervals.
- Catch trials therefore contain no positive change interval.

### Trial outcome

- Static per trial, repeated over all intervals in that trial.
- Categories: `hit`, `miss`, `false_alarm`, `correct_reject`.

## Filtering Outcomes

Skipped sessions in the final full conversion:

- `missing_eye_tracking`: 3

No additional sessions were dropped for malformed change flags in kept trials.

## Final Artifact Statistics

### Full dataset

- Sessions: 199
- Subjects: 38
- Trials: 48,655
- Image intervals: 567,895
- Trial count per session: 39 to 387
- Interval count per trial: 10 to 17
- Neurons per session: 4 to 666
- Go trials: 42,669
- Catch trials: 5,986

Outcome counts:

- `hit`: 15,217
- `miss`: 27,452
- `false_alarm`: 864
- `correct_reject`: 5,122

### Sample dataset

- Sessions: 11
- Subjects: 4
- Trials: 1,708
- Image intervals: 19,270

## Sanity Checks

### Structural checks

- `verification_full_out.txt`: passes format validation with no errors or warnings.
- `verification_sample_out.txt`: passes format validation with no errors or warnings.

### Quantile-bin checks

Full dataset interval counts are exactly balanced across quintiles:

- running bins: `113,579` each
- pupil bins: `113,579` each

### Omission checks

Omissions are part of the task and should appear in trials, but they should not occur:

- on true image-change intervals
- on the image immediately preceding a true change

Observed on the full dataset:

- omitted intervals in kept trials: `19,798`
- omitted change intervals: `0`
- omitted pre-change intervals: `0`
- change-flag mismatches: `0`

This matches the whitepaper statement that the change image and immediately preceding image are never omitted.

### Decoder performance

Sample dataset validation balanced accuracy:

- `image_identity`: `0.2729` vs chance `0.0588`
- `image_change`: `0.6431` vs chance `0.5000`
- `running_speed_bin`: `0.3633` vs chance `0.2000`
- `pupil_diameter_bin`: `0.3645` vs chance `0.2000`
- `trial_outcome`: `0.2613` vs chance `0.2500`

Full dataset validation balanced accuracy:

- `image_identity`: `0.2447` vs chance `0.0588`
- `image_change`: `0.5876` vs chance `0.5000`
- `running_speed_bin`: `0.3024` vs chance `0.2000`
- `pupil_diameter_bin`: `0.4297` vs chance `0.2000`
- `trial_outcome`: `0.3217` vs chance `0.2500`

These are above chance on all requested outputs, which is consistent with the conversion carrying real task and behavior information.

## Important Files

- `convert_data.py`: conversion implementation
- `converted_data.pkl`: final full dataset
- `sample_data.pkl`: smaller subset for quick verification
- `conversion_full_out.txt`
- `conversion_sample_out.txt`
- `verification_full_out.txt`
- `verification_sample_out.txt`
- `train_decoder_full_out.txt`
- `train_decoder_sample_out.txt`
