# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all data from the seven extensionless per-animal `joblib` files in `/app/data`, not from the `.mat` files. It treats those files as the primary source because the paper notebooks load them directly. Each animal file is loaded one at a time, then each day/session inside that file is processed and appended to the output lists.

ii.
```python
def animal_files():
    """Return primary joblib animal files, excluding larger .mat duplicates."""
    return sorted(p for p in DATA_DIR.iterdir()
                  if p.is_file() and p.name.startswith('QLAK-CA1-') and p.suffix != '.mat')

...

for si,f in enumerate(files):
    load_t=time.time(); raw=joblib.load(f)[f.name]
    print(f'Loaded {f.name} in {time.time()-load_t:.2f}s',flush=True)
```

iii. In `CONVERSION_NOTES.md`, the AI says `/app/data` contains both joblib animal files and larger MATLAB duplicates, and explicitly chooses the compact joblib files because "these are what the reference notebooks load and they contain all required aligned streams."

## 1-b. How are the data split into subjects?

i. Each per-animal file is treated as one subject. The subject identifier is the filename, and the sorted list of those filenames becomes `subjects`.

ii.
```python
files=animal_files(); subjects=[p.name for p in files]

...

for si,f in enumerate(files):
    ...
    subject_idx.append(si)
```

iii. The notes say the dataset has seven primary animal files and that "each joblib object is `{subject_id: data_dict}`," so one file naturally maps to one mouse.

## 1-c. How are the data split into sessions?

i. Within each subject file, each day index in the `trace`/`position` arrays is treated as one session. The AI appends one session per animal-day to the final lists.

ii.
```python
for day in range(raw['trace'].shape[0]):
    ntr,itr,otr,info,payload=process_session(
        raw['position'][day],raw['trace'][day],raw['blocked'][day],f.name,day,raw['envs'].ravel()[day])
    neural.append(ntr); inputs.append(itr); outputs.append(otr); subject_idx.append(si)
```

iii. In the notes, the AI states that "a target session corresponds to one animal-day/environment" and that the source arrays have a leading day axis.

## 1-d. How are the data split into trials?

i. Sessions are continuous recordings that the AI splits into non-overlapping 60-second trials. It computes the number of complete 1800-frame source windows (`30 Hz * 60 s`), discards any incomplete tail, smooths/pools continuously across the whole session, and then slices the pooled data into 600-bin trials.

ii.
```python
RAW_TRIAL_FRAMES = FPS * TRIAL_SECONDS
TRIAL_BINS = int(TRIAL_SECONDS * 1000 / BIN_MS)

...

n_trials = position.shape[1] // RAW_TRIAL_FRAMES
used_raw = n_trials * RAW_TRIAL_FRAMES

...

for tr in range(n_trials):
    sl = slice(tr * TRIAL_BINS, (tr + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(neural_pooled[:, sl], dtype=np.float32))
    input_trials.append(geometry.copy())
    output_trials.append(np.ascontiguousarray(labels[sl][None, :], dtype=np.int64))
```

iii. The notes say there are "no native trials" and that required one-minute trials should be constructed from the session start into complete contiguous windows, dropping only the incomplete tail.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply a substantive trial-quality filter. It only drops incomplete trailing data that cannot fill a full 60-second window, and it raises an error if a session would have fewer than two complete trials.

ii.
```python
n_trials = position.shape[1] // RAW_TRIAL_FRAMES
used_raw = n_trials * RAW_TRIAL_FRAMES
if n_trials < 2:
    raise ValueError(f'{subject} day {source_day}: fewer than two complete trials')
```

iii. In the notes, the AI explicitly says there are no native failed trials and that contiguous timing should be preserved. It frames the incomplete-tail drop as required to enforce equal-length trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the raw per-day `trace` arrays in the source animal files.

ii.
```python
ntr,itr,otr,info,payload=process_session(
    raw['position'][day],raw['trace'][day],raw['blocked'][day],f.name,day,raw['envs'].ravel()[day])
```

iii. The notes say `trace[day, cell, frame]` is the source for `neural`, and describe it as already paper-curated calcium-event activity rather than raw fluorescence.

## 2-b. How is the `neural` data processed?

i. The AI keeps selected cells, casts to `float32`, Gaussian-smooths each cell's activity across time with `sigma=3` source frames, then average-pools non-overlapping groups of 3 frames to produce 100 ms bins. It does not keep the raw 30 Hz trace values.

ii.
```python
selected = np.asarray(trace[keep], dtype=np.float32)

...

smooth = gaussian_filter1d(selected, sigma=POOL, axis=1).astype(np.float32, copy=False)
neural_pooled = smooth[:, :used_raw].reshape(selected.shape[0], -1, POOL).mean(axis=2,
                                                                                   dtype=np.float32)
```

iii. The justification in the notes is that this matches the paper code's `fit_decoder`/`test_decoder` temporal processing: "Gaussian smoothing and non-overlapping 3-frame mean pooling," giving a 100 ms bin size aligned with the reference decoder.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI first defines "present" cells as those with any finite sample on that day. It then applies a second decoder-style activity filter: compute speed from position, smooth speed with a Gaussian (`sigma=5` frames), mark frames with speed `> 5 cm/s` as moving, and keep only cells with summed activity `> 5` on those moving frames.

ii.
```python
def select_cells(position, trace):
    present = np.isfinite(trace).any(axis=1)
    displacement_speed = np.linalg.norm(np.diff(position.T, axis=0) * FPS, axis=1)
    moving = np.zeros(position.shape[1], dtype=bool)
    moving[1:] = gaussian_filter1d(displacement_speed, sigma=5, axis=0) > 5.0
    event_sum = np.nansum(trace[:, moving], axis=1)
    keep = present & (event_sum > 5)
    return keep, moving, event_sum
```

iii. The notes say this was chosen to "match `decode_position_within`" from the paper code, while still preserving all timepoints in the final trials. The AI explicitly rejected place-cell-only filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no stimulus- or behavior-locked event. The AI defines the alignment event as the start of each artificial non-overlapping 1-minute segment cut from a continuous session.

ii.
```python
'temporal_alignment_event':'Start of each non-overlapping 1-minute segment within a continuous animal-day recording',
'off_start':0.0,
'off_end':60.0,
```

iii. The notes repeatedly state that the source data are continuous ~40-minute sessions with no native trials, so the start of each constructed 60-second segment is used as the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 100 ms bins. Yes: the AI rebins from 30 Hz source frames by smoothing the neural data and then mean-pooling every 3 frames.

ii.
```python
FPS = 30
POOL = 3
BIN_MS = 100.0

...

'time_bin_size':BIN_MS,
...
neural_pooled = smooth[:, :used_raw].reshape(selected.shape[0], -1, POOL).mean(axis=2,
                                                                                   dtype=np.float32)
```

iii. The notes justify this as matching the paper decoder's 3-frame pooling and call 100 ms "the most directly applicable common neural/behavior time bin."

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. `input` environment geometry is derived from the raw per-day `blocked` variable.

ii.
```python
ntr,itr,otr,info,payload=process_session(
    raw['position'][day],raw['trace'][day],raw['blocked'][day],f.name,day,raw['envs'].ravel()[day])

...

geometry = blocked_mask(blocked)
```

iii. The notes say `blocked[day]` contains row-major 3x3 blocked partition indices, with `-1` meaning the open square environment.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. The AI flattens the blocked-index list, removes `-1`, validates that remaining indices are between 0 and 8, and converts the result into a static length-9 float32 binary mask. That same mask is copied into every trial for the session.

ii.
```python
def blocked_mask(raw):
    idx = np.asarray(raw).ravel().astype(np.int64)
    mask = np.zeros(9, dtype=np.float32)
    idx = idx[idx >= 0]  # -1 denotes no blocked partition
    if idx.size:
        if np.any(idx > 8):
            raise ValueError(f'Invalid blocked indices: {idx}')
        mask[idx] = 1.0
    return mask

...

input_trials.append(geometry.copy())
```

iii. The notes justify this as an explicit decoder-compatible representation of the environment geometry, and say it matches the source README's row-major blocked-index convention.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. `output` mouse position is derived from the raw per-day `position` array.

ii.
```python
ntr,itr,otr,info,payload=process_session(
    raw['position'][day],raw['trace'][day],raw['blocked'][day],f.name,day,raw['envs'].ravel()[day])
```

iii. The notes describe `position[day, x/y, frame]` as the aligned source for the decoder output.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. The AI first mean-pools `x` and `y` over the same non-overlapping 3-frame bins used for the neural data, then converts the pooled coordinates into 3x3 position labels. The final per-trial output is stored as shape `(1, 600)` integer labels.

ii.
```python
pos_pooled = position[:, :used_raw].reshape(2, -1, POOL).mean(axis=2)
xybin = np.clip(np.floor(pos_pooled / 25.0).astype(np.int64), 0, 2)
labels = (xybin[1] * 3 + xybin[0]).astype(np.int64)

...

output_trials.append(np.ascontiguousarray(labels[sl][None, :], dtype=np.int64))
```

iii. The notes justify the shared 3-frame pooling as matching the reference decoder's behavior pooling, then replacing the paper's finer 15x15 decoding grid with the task-required 3x3 experimental partitions.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each pooled coordinate is divided by 25 cm, floored to get one of three bins on each axis, clipped to stay within `0..2`, and then converted to a row-major class index `y_bin * 3 + x_bin`, yielding 9 categories.

ii.
```python
xybin = np.clip(np.floor(pos_pooled / 25.0).astype(np.int64), 0, 2)
labels = (xybin[1] * 3 + xybin[0]).astype(np.int64)
OUTPUT_VALUES = ['NW','N','NE','W','center','E','SW','S','SE']
```

iii. The notes say the bin edges are the fixed physical 25 cm boundaries of the 75 cm arena and that clipping handles exact 75 cm boundary values.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. The AI keeps neural and position aligned by operating on the same source frames, pooling both streams in the same 3-frame windows, and slicing both into trials with identical pooled-time indices.

ii.
```python
smooth = gaussian_filter1d(selected, sigma=POOL, axis=1).astype(np.float32, copy=False)
neural_pooled = smooth[:, :used_raw].reshape(selected.shape[0], -1, POOL).mean(axis=2,
                                                                                   dtype=np.float32)
pos_pooled = position[:, :used_raw].reshape(2, -1, POOL).mean(axis=2)

...

sl = slice(tr * TRIAL_BINS, (tr + 1) * TRIAL_BINS)
neural_trials.append(np.ascontiguousarray(neural_pooled[:, sl], dtype=np.float32))
output_trials.append(np.ascontiguousarray(labels[sl][None, :], dtype=np.int64))
```

iii. The notes say the source `trace` and `position` streams are already frame-aligned, and that pooling/slicing is intentionally done on identical frame groups to preserve that alignment.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing neural data are handled by treating cells with no finite samples as absent, then filtering cells further with the moving-frame activity criterion. Incomplete session tails shorter than 60 seconds are discarded. Position is required to be finite; exact upper-edge coordinates are clipped into the last spatial bin; invalid blocked indices raise an error.

ii.
```python
if not np.isfinite(position).all():
    raise ValueError(f'{subject} day {source_day}: nonfinite position')

...

present = np.isfinite(trace).any(axis=1)

...

used_raw = n_trials * RAW_TRIAL_FRAMES

...

xybin = np.clip(np.floor(pos_pooled / 25.0).astype(np.int64), 0, 2)
```

iii. The notes say NaN traces mark absent cells, positions are fully finite in the dataset, and the incomplete-tail drop is the chosen way to enforce consistent trial length without padding.

## 6-a. What are the most time-consuming steps of the code?

i. The AI identifies loading/decompressing the per-animal joblib files as the main unavoidable cost, with full-session smoothing/pooling and final pickle serialization as the other notable costs.

ii.
```python
raw=joblib.load(f)[f.name]

...

smooth = gaussian_filter1d(selected, sigma=POOL, axis=1).astype(np.float32, copy=False)

...

with open(tmp,'wb') as fh: pickle.dump(data,fh,protocol=pickle.HIGHEST_PROTOCOL)
```

iii. In the notes, the AI says "Source joblib files are compressed and must each be decompressed once," and gives runtime estimates that separate source loading, processing, and serialization.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The heavy per-frame work is already vectorized. The main remaining loop that could only partly be reduced is the per-trial packaging loop that slices pooled arrays into Python lists and copies the static input vector once per trial.

ii.
```python
neural_trials, input_trials, output_trials = [], [], []
for tr in range(n_trials):
    sl = slice(tr * TRIAL_BINS, (tr + 1) * TRIAL_BINS)
    neural_trials.append(np.ascontiguousarray(neural_pooled[:, sl], dtype=np.float32))
    input_trials.append(geometry.copy())
    output_trials.append(np.ascontiguousarray(labels[sl][None, :], dtype=np.int64))
```

iii. The notes say "Vectorized Gaussian filtering, reshape-based pooling, and label construction replace per-frame loops," implying that most of the expensive computation was already vectorized.

## 6-c. What processing does the code repeat multiple times?

i. The code repeats per-session bookkeeping and some small computations: it recomputes present-neuron counts for `info`, repeatedly concatenates trial outputs to accumulate `class_counts`, and copies the same geometry vector into every trial in a session.

ii.
```python
'source_present_neurons': int(np.isfinite(trace).any(axis=1).sum()),

...

class_counts += np.bincount(np.concatenate([x.ravel() for x in otr]),minlength=9)

...

input_trials.append(geometry.copy())
```

iii. The notes acknowledge some unavoidable copying from the target "list-of-trial-arrays format" and describe the geometry replication as part of the requested static-per-trial representation.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and returns extra diagnostic information that is not needed by the downstream decoder: `moving`, `event_sum`, `plot_payload`, `session_info`, timing/class summary stats, and optional plots. `plot_payload` is constructed even when plots are not requested.

ii.
```python
keep, moving, event_sum = select_cells(position, trace)

...

info = {
    ...
    'moving_frame_fraction': float(moving.mean()),
}
plot_payload = (pos_pooled[:, :TRIAL_BINS], labels[:TRIAL_BINS],
                neural_pooled[:min(30, selected.shape[0]), :TRIAL_BINS], geometry)
return neural_trials, input_trials, output_trials, info, plot_payload
```

iii. The notes describe these as sanity-check and documentation aids: raw-source spot checks, plots, runtime summaries, and rich `session_info` metadata for auditing edge cases.
