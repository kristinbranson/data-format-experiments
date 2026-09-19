# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not scan the raw `.mat` files. Instead, it hard-codes the seven animal IDs, loads one `joblib` file per animal from `data/`, then reads nested arrays from the loaded dict. Trials are created later from each session/day after temporal binning.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
...
for a_idx, animal in enumerate(animals_to_process):
    print(f"\nLoading {animal}...")
    dat = joblib.load(os.path.join(data_dir, animal))
    animal_data = dat[animal]
    n_days = animal_data['trace'].shape[0]
```

iii. In `CONVERSION_NOTES.md`, the AI says each animal has a `joblib` file with `trace`, `position`, `envs`, `blocked`, and other fields, and frames the joblib structure as the dataset entry point. The notes justify this by saying it matches the reference code path it explored, and the trajectory summary says the dataset was converted from these seven mice with per-day sessions.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the hard-coded `ANIMALS` list. Each loaded joblib file corresponds to one mouse, and `subjects` is just `list(animals_to_process)`.

ii.
```python
subjects = list(animals_to_process)
...
for a_idx, animal in enumerate(animals_to_process):
    dat = joblib.load(os.path.join(data_dir, animal))
    animal_data = dat[animal]
```

iii. The notes identify seven mice by these exact IDs and report subject-level counts, so the AI’s justification is that one file corresponds to one animal and the animal name should be the subject ID.

## 1-c. How are the data split into sessions?

i. The AI treats each recording day as one session. It uses the first axis of `animal_data['trace']` as the session/day index and appends one converted session per day.

ii.
```python
n_days = animal_data['trace'].shape[0]
...
for day in range(n_days):
    neural_trials, input_trials, output_trials, n_valid = process_session(
        animal_data, day,
        show_processing=do_plot,
        animal_name=animal,
        fig_axes=fig_axes
    )
...
    all_neural.append(neural_trials)
    all_input.append(input_trials)
    all_output.append(output_trials)
```

iii. `CONVERSION_NOTES.md` explicitly says “Session = day” and describes the dataset as `(n_days, n_cells, n_frames)` for `trace`, so the AI justified using each day as one output session.

## 1-d. How are the data split into trials?

i. Each session is split into non-overlapping 1-minute trials after temporal binning. Because the AI first rebins time into 100 ms bins, each trial contains 600 bins. Any remainder shorter than a full minute is discarded.

ii.
```python
TRIAL_DURATION_SEC = 60
FRAMES_PER_TRIAL = TRIAL_DURATION_SEC * FPS
BINS_PER_TRIAL = FRAMES_PER_TRIAL // TEMPORAL_BIN_SIZE
...
n_trials = n_total_bins // BINS_PER_TRIAL
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    trial_neural = binned_trace[:, start:end]
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. The notes say “Trial = 1-minute segment” and the trajectory explicitly states “39 trials per session ... dropping the last partial trial,” which is the AI’s rationale for the trial split.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial quality-control filter. The AI keeps all full 1-minute trials produced by integer division. It does skip any session that ends up with fewer than two trials.

ii.
```python
n_trials = n_total_bins // BINS_PER_TRIAL
...
if len(neural_trials) < 2:
    print(f"  Day {day}: skipped (< 2 trials)")
    ...
    continue
```

iii. The notes say there is “No explicit trial filtering in the reference,” and the code only enforces the target-format requirement that a kept session must have at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from the `trace` field in each animal’s loaded joblib object, indexed by day.

ii.
```python
trace = animal_data['trace'][day_idx]  # (n_cells, n_frames)
```

iii. The notes describe `trace` as binary calcium events, already preprocessed by the original pipeline, and repeatedly state that neural data comes from `trace`.

## 2-b. How is the `neural` data processed?

i. The AI filters to valid cells, then smooths each cell’s trace over time with a Gaussian kernel (`sigma=3` frames), average-pools into 3-frame bins, and casts to `float32`. It does not transpose because its source array is already `(n_cells, n_frames)`.

ii.
```python
def temporal_bin_trace(trace_2d, sigma=TEMPORAL_BIN_SIZE, bin_size=TEMPORAL_BIN_SIZE):
    smoothed = gaussian_filter1d(trace_2d.astype(np.float64), sigma=sigma, axis=1)
    n_cells, n_frames = smoothed.shape
    n_bins = n_frames // bin_size
    trimmed = smoothed[:, :n_bins * bin_size]
    binned = trimmed.reshape(n_cells, n_bins, bin_size).mean(axis=2)
    return binned.astype(np.float32)
...
valid_trace = trace[valid_mask]
binned_trace = temporal_bin_trace(valid_trace)
```

iii. The notes explicitly justify this as matching the reference decoder path: “same temporal binning approach,” “Gaussian smoothing of trace,” and “reference `fit_decoder` applies `gaussian_filter1d` ... before binning.” The README repeats that neural data are Gaussian smoothed and temporally binned to 100 ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only cells whose first frame is not `NaN`, interpreting that as “registered cells for this day.” It does not apply place-cell filtering, movement filtering, or an event-count threshold during conversion.

ii.
```python
valid_mask = ~np.isnan(trace[:, 0])
valid_trace = trace[valid_mask]
n_valid = valid_mask.sum()

if n_valid == 0:
    return [], [], [], 0
```

iii. `CONVERSION_NOTES.md` says “Include only registered cells per session (not NaN)” and “No place cell filtering.” It also notes that velocity filtering is decoder-specific and should not be done in conversion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does not do event-based alignment. It treats session start as the alignment point and cuts the continuous recording into consecutive 1-minute windows.

ii.
```python
'metadata': {
    'temporal_alignment_event': 'Start of recording session',
    'off_start': 0.0,
    'off_end': float(TRIAL_DURATION_SEC),
    ...
}
...
start = t * BINS_PER_TRIAL
end = (t + 1) * BINS_PER_TRIAL
trial_neural = binned_trace[:, start:end]
```

iii. The notes explicitly say “Temporal alignment: Trials start at beginning of session recording. Align to session start.” This is also reflected in metadata.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 100 ms bins. The AI rebins from 30 Hz frames by smoothing and averaging each non-overlapping group of 3 frames.

ii.
```python
FPS = 30
TEMPORAL_BIN_SIZE = 3
TIME_BIN_MS = (TEMPORAL_BIN_SIZE / FPS) * 1000  # 100 ms
...
binned_trace = temporal_bin_trace(valid_trace)
```

iii. The notes justify this by citing the reference decoder’s `temporal_bin_size=3` and state that 100 ms bins “match reference decoder code.”

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The AI derives environment geometry from the per-session `envs` label, not from `blocked`. It maps the environment name to a hard-coded 3x3 accessibility matrix.

ii.
```python
env_name = str(animal_data['envs'][day_idx].item() if hasattr(animal_data['envs'][day_idx], 'item')
               else animal_data['envs'][day_idx])
...
ENV_MATRICES = {
    'square':    np.array([[1,1,1],[1,1,1],[1,1,1]]),
    'o':         np.array([[1,1,1],[1,0,1],[1,1,1]]),
    ...
}
```

iii. The notes say the input should come from `envs[day] -> get_env_mat()` and describe this as using the reference function for environment geometry.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI looks up a 3x3 binary matrix for the session’s environment name, flattens it to length 9, casts to `float32`, and uses the same static vector for every trial in that session.

ii.
```python
def get_env_input(env_name):
    mat = ENV_MATRICES.get(env_name)
    if mat is None:
        raise ValueError(f"Unknown environment: {env_name}")
    return mat.flatten().astype(np.float32)
...
env_input = get_env_input(env_name)
...
input_trials.append(env_input)
```

iii. The notes justify this as the environment geometry required by the decoder task: “3x3 binary matrix ... flattened to 9 values,” static per trial, matching the reference `get_env_mat` idea.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. The AI derives output position from the `position` field for each day/session.

ii.
```python
position = animal_data['position'][day_idx]  # (2, n_frames)
```

iii. The notes describe `position` as `(n_days, 2, n_frames)` with x,y coordinates in centimeters and state that decoder output is derived from that array.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first temporally bins continuous x,y position by averaging every 3 frames. It then discretizes each binned x,y sample into a 3x3 grid using 25 cm bin widths and stores the result as a single categorical time series.

ii.
```python
def bin_position(position_2d, bin_size=TEMPORAL_BIN_SIZE):
    n_frames = position_2d.shape[1]
    n_bins = n_frames // bin_size
    trimmed = position_2d[:, :n_bins * bin_size]
    binned = trimmed.reshape(2, n_bins, bin_size).mean(axis=2)
    return binned
...
binned_pos = bin_position(position)
pos_bins = discretize_position(binned_pos)
```

iii. The notes say “Position ... temporally binned by 3 frames” and treat this as matching the reference decoder behavior before converting to 3x3 bins.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. The AI computes `x_bin = floor(x / 25)` and `y_bin = floor(y / 25)`, clamps each to `[0, 2]`, and combines them as `x_bin * 3 + y_bin` to produce class labels `0..8`.

ii.
```python
def discretize_position(position_2d, bin_size=SPATIAL_BIN_SIZE, n_bins=N_SPATIAL_BINS):
    x_bin = np.clip(np.floor(position_2d[0] / bin_size).astype(int), 0, n_bins - 1)
    y_bin = np.clip(np.floor(position_2d[1] / bin_size).astype(int), 0, n_bins - 1)
    return x_bin * n_bins + y_bin
```

iii. The notes justify this as 3x3 spatial bins over a 75 cm arena, with combined index values `0-8`. The README presents these as the output position bins.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI aligns output to neural data by applying the same 3-frame temporal binning scheme to both streams and then slicing them into trials with the same start/end indices.

ii.
```python
binned_trace = temporal_bin_trace(valid_trace)
binned_pos = bin_position(position)
pos_bins = discretize_position(binned_pos)
...
start = t * BINS_PER_TRIAL
end = (t + 1) * BINS_PER_TRIAL
trial_neural = binned_trace[:, start:end]
trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)
```

iii. The notes say the converted data should use the “same temporal binning approach” for decoder inputs/outputs, and the metadata names session start as the alignment event.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missingness mainly by removing cells marked `NaN` at the first frame of a session and by dropping trailing partial trials via integer division. If a day has no valid cells, it returns no trials for that session.

ii.
```python
valid_mask = ~np.isnan(trace[:, 0])
valid_trace = trace[valid_mask]
...
if n_valid == 0:
    return [], [], [], 0
...
n_trials = n_total_bins // BINS_PER_TRIAL
```

iii. The notes describe NaN values as unregistered cells and explicitly plan to include “only registered cells per session.” The trajectory also mentions dropping the final partial trial after temporal binning.

## 6-a. What are the most time-consuming steps of the code?

i. The most expensive processing in the AI’s pipeline is per-session loading plus the Gaussian smoothing and temporal binning over large cell-by-time matrices. Optional plotting can add more overhead when enabled.

ii.
```python
dat = joblib.load(os.path.join(data_dir, animal))
...
smoothed = gaussian_filter1d(trace_2d.astype(np.float64), sigma=sigma, axis=1)
...
if do_plot:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
```

iii. The notes report per-animal runtimes and emphasize that the pipeline matches a decoder-style temporal binning path, which explains why smoothing and large-array reshaping are central runtime costs.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorizable loop is the per-trial Python loop that repeatedly slices arrays and appends identical static inputs. Session/day iteration is structurally necessary, but trial packaging could have been batched more efficiently.

ii.
```python
neural_trials = []
input_trials = []
output_trials = []

n_trials = n_total_bins // BINS_PER_TRIAL
for t in range(n_trials):
    start = t * BINS_PER_TRIAL
    end = (t + 1) * BINS_PER_TRIAL
    trial_neural = binned_trace[:, start:end]
    trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)

    neural_trials.append(trial_neural)
    input_trials.append(env_input)
    output_trials.append(trial_output)
```

iii. The AI did not explicitly discuss vectorization in the notes, but the code structure shows trial creation is handled in Python lists rather than via a batched reshape/split.

## 6-c. What processing does the code repeat multiple times?

i. The code repeatedly converts environment labels to strings, repeatedly appends the same static environment vector once per trial, and repeats session-level figure setup/teardown when plotting is enabled.

ii.
```python
env_name = str(animal_data['envs'][day_idx].item() if hasattr(animal_data['envs'][day_idx], 'item')
               else animal_data['envs'][day_idx])
...
input_trials.append(env_input)
...
env_name = str(animal_data['envs'][day].squeeze())
...
if do_plot:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
```

iii. The notes justify using static per-trial environment inputs and optional visualization for inspection, but the implementation repeats some session metadata handling and trial-level packaging work instead of sharing it.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code includes optional plotting and extensive runtime/accounting output that are not part of the saved dataset. It also computes continuous `binned_pos` only to immediately discard it after converting to categorical bins.

ii.
```python
binned_pos = bin_position(position)  # (2, n_total_bins)
pos_bins = discretize_position(binned_pos)  # (n_total_bins,)
...
if do_plot and fig is not None:
    fig.suptitle(f'{animal} Day {day} ({env_name}) - Processing Overview', fontsize=14)
    fig.tight_layout()
    fig.savefig(f'processing_{animal}_day{day}.png', dpi=100)
...
print(f"\n=== Conversion Summary ===")
print(f"Subjects: {len(subjects)}")
```

iii. The notes describe `--show-processing` as a validation aid and emphasize summary statistics and sanity checks, so these extra branches were included for inspection rather than for downstream decoder consumption.
