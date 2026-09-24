# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script uses a hard-coded list of seven animal IDs. For each ID it loads the corresponding extensionless joblib file from the relative `data` directory, selects the nested animal dictionary, and reads the full `trace`, `position`, and `envs` arrays. Full mode processes all seven; sample mode processes the first two animals (all their sessions).

ii.
```python
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
    "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
dat = joblib.load(os.path.join(data_dir, animal))
d = dat[animal]
trace = d['trace']
position = d['position']
envs = d['envs']
```

iii. The notes say the reference repository's `load_dat` supports joblib and that exploration found seven joblib animal files with the required arrays. They report checks of 7 subjects, 207 sessions, and 69,744 active neuron-sessions.

## 1-b. How are the data split into subjects?

i. Each hard-coded animal ID is one subject. Its position in `animals_to_process` is stored as the subject index for every session loaded from that animal.

ii.
```python
for a_idx, animal in enumerate(animals_to_process):
    sessions = process_animal(animal, data_dir=DATA_DIR, ...)
    for sess in sessions:
        all_subject_idx.append(a_idx)
...
'subjects': [a for a in animals_to_process],
```

iii. The notes identify seven animal-specific files and verify session counts per animal as 31, 31, 31, 21, 31, 31, and 31.

## 1-c. How are the data split into sessions?

i. Axis 0 of each animal's `trace`, `position`, and `envs` arrays is treated as recording day/session. Every day becomes one output session.

ii.
```python
n_sessions = trace.shape[0]
for day in range(n_sessions):
    env_name = envs[day, 0]
    tr = trace[day]
    pos = position[day]
```

iii. The agent states that each day/recording is one session and confirms a total of 207 sessions, matching the paper.

## 1-d. How are the data split into trials?

i. Each continuous session is divided into consecutive, non-overlapping 1-minute trials of 1,800 frames at 30 Hz. Any final incomplete minute is dropped.

ii.
```python
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_SEC
n_trials = n_timepoints // FRAMES_PER_TRIAL
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
```

iii. The notes cite the explicit instruction to create one-minute trials, report 39–40 trials per session, and document boundary and slicing checks.

## 1-e. How are trials filtered based on quality controls?

i. No complete one-minute trial is filtered for quality. Only an incomplete trailing segment is omitted. In particular, low-velocity frames are retained.

ii.
```python
n_trials = n_timepoints // FRAMES_PER_TRIAL
# only range(n_trials) is emitted
```

iii. The notes explicitly decide not to apply the reference decoder's velocity filtering during conversion because the downstream decoder is different. No trial-level rejection criterion was found in the source analysis.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `dat[animal]['trace']`, a session-by-neuron-by-time array of already binarized rising-phase calcium transients.

ii.
```python
trace = d['trace']       # (n_sessions, n_neurons, n_timepoints)
tr = trace[day]          # (n_neurons, n_timepoints)
```

iii. The notes and methods identify `trace` as the already processed binary rising-phase vector, so the agent did not recompute fluorescence or deconvolution.

## 2-b. How is the `neural` data processed?

i. For each session, all-NaN neuron rows are removed, remaining NaNs are changed to zero, and each one-minute slice is cast to float32. No smoothing, normalization, or temporal rebinning is performed.

ii.
```python
active_mask = ~np.all(np.isnan(tr), axis=1)
active_neurons = tr[active_mask]
active_neurons = np.nan_to_num(active_neurons, nan=0.0)
trial_neural.append(active_neurons[:, start:end].astype(np.float32))
```

iii. The agent argues that the stored traces are already binary transients and that the requested decoder can consume native frames. `nan_to_num` is described as a safety measure that “shouldn't happen.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only neurons whose entire session trace is NaN are excluded. The paper/reference decoder's activity threshold and place-cell criteria are not applied.

ii.
```python
active_mask = ~np.all(np.isnan(tr), axis=1)
active_neurons = tr[active_mask]
```

iii. The notes interpret all-NaN rows as cells not tracked that day. They explicitly retain all detected cells, reasoning that activity filtering belongs to the original decoding routine rather than conversion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no experimental event alignment. Trial zero begins at session frame zero, and later trials begin at successive 1,800-frame boundaries. Metadata calls the alignment event `Start of recording session`, even though every stored trial is a different segment.

ii.
```python
start = t * FRAMES_PER_TRIAL
end = (t + 1) * FRAMES_PER_TRIAL
...
'temporal_alignment_event': 'Start of recording session',
'off_start': 0.0,
'off_end': TRIAL_DURATION_SEC,
```

iii. The notes treat the recordings as continuous data with artificial one-minute trials; no stimulus-alignment event exists. They report checking exact source/converted frame correspondence.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native 30 Hz samples are retained, giving 33.33 ms per time bin. No temporal rebinning is applied.

ii.
```python
FPS = 30
...
'time_bin_size': 1000.0 / FPS,
```

iii. The agent notes that the original decoder pools three frames, but deliberately keeps native frames because conversion specifications and the new decoder do not require that pooling.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. The input is derived from the session's string in `d['envs']`, not from the raw `blocked` variable. The string is looked up in a hard-coded environment-name-to-mask dictionary.

ii.
```python
envs = d['envs']
env_name = envs[day, 0]
env_mat = get_env_mat(env_name).flatten()
```

iii. The notes say the repository's canonical `get_env_mat` masks were chosen because `blocked` indices appeared to use a different orientation.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. `get_env_mat` maps each of ten known names to a binary 3×3 accessible-space mask, flattens it to nine float32 values, and repeats that static vector for every trial in the session. Unknown names silently produce nine zeros.

ii.
```python
return np.array(envs.get(env, [[0,0,0],[0,0,0],[0,0,0]])).astype(float)
...
env_mat = get_env_mat(env_name).flatten()
trial_input.append(env_mat.astype(np.float32))
```

iii. The agent copied the helper from the repository and describes it as the canonical environment representation. It verified masks for the first five session environments.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from the two coordinate streams in `dat[animal]['position']` for each session.

ii.
```python
position = d['position'] # (n_sessions, 2, n_timepoints)
pos = position[day]
pos_bins = position_to_bin(pos)
```

iii. The notes identify these as DeepLabCut x/y positions in centimeters over a 75×75 cm arena.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Each axis is divided by 25 cm and floored, clipped into 0–2, then combined into one of nine integer labels. Each trial slice is reshaped to `(1, 1800)` and cast to int64.

ii.
```python
x_bins = np.clip(np.floor(pos_xy[0] / bin_size).astype(int), 0, n_bins - 1)
y_bins = np.clip(np.floor(pos_xy[1] / bin_size).astype(int), 0, n_bins - 1)
bin_indices = x_bins * n_bins + y_bins
...
trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```

iii. The task requires a 3×3 categorical output. The notes say positions span 0–75 cm and document tests at arena boundaries and against recomputed source bins.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Thresholds are 25 and 50 cm on both axes (implemented by floor division by 25), with out-of-range values clipped. Labels use `x_bin * 3 + y_bin`, so x is the major index.

ii.
```python
bin_size = env_size / n_bins
x_bins = np.clip(np.floor(pos_xy[0] / bin_size).astype(int), 0, n_bins - 1)
y_bins = np.clip(np.floor(pos_xy[1] / bin_size).astype(int), 0, n_bins - 1)
bin_indices = x_bins * n_bins + y_bins
```

iii. The agent describes equal 25 cm cells and a row-times-three-plus-column convention, although the implemented choice makes the first (`x`) coordinate the row/major coordinate.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and trace are assumed to share frame indices and 30 Hz sampling. Both are sliced with the identical `start:end` boundaries for each trial.

ii.
```python
trial_neural.append(active_neurons[:, start:end].astype(np.float32))
trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```

iii. The agent reports direct checks of several sessions/trials and selected frame positions against the original arrays, with no offset or resampling required.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. All-NaN neurons are removed; any remaining neural NaNs are replaced by zero. Coordinate values outside the arena are clipped to valid categories. Incomplete final trial data is dropped. Unknown environment names become an all-zero mask rather than raising an error.

ii.
```python
active_mask = ~np.all(np.isnan(tr), axis=1)
active_neurons = np.nan_to_num(tr[active_mask], nan=0.0)
x_bins = np.clip(..., 0, n_bins - 1)
n_trials = n_timepoints // FRAMES_PER_TRIAL
```

iii. The notes explain all-NaN rows as absent cells and remaining-NaN replacement as defensive. They report explicit boundary checks and accept the small loss from incomplete one-minute tails.

## 6-a. What are the most time-consuming steps of the code?

i. Loading large joblib animal arrays and serializing the roughly 20 GB pickle dominate. Optional plotting and decoder training are outside the core conversion. Per-animal/session timings are instrumented.

ii.
```python
t0 = time.time()
dat = joblib.load(os.path.join(data_dir, animal))
...
with open(args.output, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. The notes measured loading at roughly 12–23 seconds per animal and estimated saving at about 40 seconds for the full dataset; processing was estimated at 8–15 seconds per animal.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial creation loops over every trial to slice/cast three streams; trial arrays could instead be truncated and reshaped in bulk. The summary distribution also loops through all session/trial outputs before concatenation. Animal and session loops are structurally useful because neuron counts and stored session lists vary.

ii.
```python
for t in range(n_trials):
    start = t * FRAMES_PER_TRIAL
    end = (t + 1) * FRAMES_PER_TRIAL
    trial_neural.append(active_neurons[:, start:end].astype(np.float32))
...
for sess_outputs in all_output:
    for trial_out in sess_outputs:
        all_bins.append(trial_out[0])
```

iii. The agent did not explicitly discuss vectorization. Its timings imply these loops were acceptable relative to loading and saving.

## 6-c. What processing does the code repeat multiple times?

i. The environment mask is recast to float32 once per trial; neural data is cast once per trial even though a whole session could be cast once. Output slices likewise reshape and cast repeatedly. Summary traversal rereads all output labels after conversion.

ii.
```python
trial_neural.append(active_neurons[:, start:end].astype(np.float32))
trial_input.append(env_mat.astype(np.float32))
trial_output.append(pos_bins[start:end].reshape(1, -1).astype(np.int64))
```

iii. No explicit justification for the repeated casts is given. The notes emphasize simple per-trial construction and validation rather than optimization.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It builds `session_info` metadata and scans all output samples to print a global histogram; these do not affect decoder tensors. With `--show-processing`, it also generates figures solely for manual verification. It retains extensive descriptive metadata, while `env_name` and `animal` in intermediate session dictionaries are used only to build metadata.

ii.
```python
session_info.append({'animal': sess['animal'], 'env_name': sess['env_name'], ...})
...
all_bins = np.concatenate(all_bins)
counts = np.bincount(all_bins.astype(int), minlength=9)
...
if show_processing:
    plot_processing(d, animal, sessions)
```

iii. The notes frame the plots, histogram, and metadata as sanity checks and provenance. They are useful for verification but are not consumed by the decoder itself.
