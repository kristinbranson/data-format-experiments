# CA1 Geometric Deformation Dataset - Decoder-Ready Conversion

Converted from **Lee, Keinath, Cianfarano & Brandon (2025), "Identifying representational structure in CA1 to
benchmark theoretical models of cognitive mapping", Neuron 113(2):307-320**
(data: https://doi.org/10.5281/zenodo.13993254, code: https://github.com/jquinnlee/georepca1).

## Dataset description

Seven mice freely foraged for 40 min/day in a 75 x 75 cm arena that was partitioned into a 3 x 3 grid of
25 x 25 cm compartments. On each day a different subset of the 9 compartments was walled off, producing 10
distinct geometries (`square, o, t, u, rectangle, +, i, l, bit donut, glenn`); the sequence started and ended
with the square and was repeated up to three times per mouse. CA1 activity was recorded with a one-photon
miniscope at 30 Hz and preprocessed by the authors into **binarized calcium-transient rising-phase events**
(1 = event), which the paper treats as the firing rate. Head position came from DeepLabCut and is sampled on the
same 30 Hz frame clock.

**Decoder task**: from CA1 activity plus the static environment geometry, predict which of the 9 arena
partitions the mouse occupies in each 100 ms time bin.

## Key statistics

| Statistic | Value |
|-----------|-------|
| Subjects (mice) | 7 (`QLAK-CA1-08/30/50/51/56/74/75`) |
| Sessions | 207 (31 per mouse, 21 for QLAK-CA1-51) |
| Trials | 8,280 (40 per session; non-overlapping 1-minute blocks) |
| Time bin | 100 ms (3 frames at 30 Hz); T = 600 per trial (555-600 for the last trial of a session) |
| Neurons | 5,413 unique cells; 69,744 registered cell-sessions (paper: "69,744 rate maps") |
| Neurons per session | mean 336.9, min 113, max 564 |
| Brain region | CA1 (all neurons) |
| Neural units | calcium events / s (mean 0.29 Hz, range 0-30 Hz) |
| Inputs | 9 binary values: is partition p blocked in this session (static per trial) |
| Output | `position_partition`, 9 categories, time-varying |
| Output class fractions | 0.099, 0.098, 0.135, 0.075, 0.057, 0.077, 0.116, 0.141, 0.201 |
| Decoder validation balanced accuracy | 0.605 (chance 0.111) |

## How to load and use

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][session][trial]   # (n_neurons, T) float32, events/s
inp    = data['input'][session][trial]    # (9,) float32, 1 = partition blocked
out    = data['output'][session][trial]   # (1, T) int64, partition index 0-8
subject = data['subjects'][data['subject_idx'][session]]
env     = data['metadata']['session_info'][session]['environment']
```

Regenerate with:
```bash
python -u convert_data.py converted_data.pkl --full          # all 207 sessions (~3 min, 6.7 GB)
python -u convert_data.py sample_data.pkl --sample --show-processing   # 2 sessions + diagnostic plots
python -u train_decoder.py converted_data.pkl                # train/validate the decoder
```

## Output format specification

```
data = {
  'neural':  [session][trial] -> (n_neurons, T) float32   # calcium event rate, events/s
  'input':   [session][trial] -> (9,) float32             # blocked_partition_0..8 (static)
  'output':  [session][trial] -> (1, T) int64             # position_partition, 0..8
  'subjects': list of 7 mouse IDs
  'subject_idx': (207,) int64 index into 'subjects'
  'brain_regions': ['CA1']
  'brain_region_idx': [session] -> (n_neurons,) zeros
  'input_names': ['blocked_partition_0', ..., 'blocked_partition_8']
  'output_names': ['position_partition']
  'output_values': [[ 'p0 (x bin 0, y bin 0)', ..., 'p8 (x bin 2, y bin 2)' ]]
  'metadata': { task_description, time_bin_size (100.0 ms), temporal_alignment_event,
                off_start (0.0 s), off_end (60.0 s), session_info (per-session id, subject, day,
                environment, blocked partitions, n_neurons, n_trials, n_time_bins),
                neural_units, neural_preprocessing, output_discretization,
                spatial_bin_size_cm (25), arena_size_cm (75), fps (30), source }
}
```

**Partition indexing**: `p = 3*floor(y/25) + floor(x/25)` with x, y in cm, i.e.
`[[0,1,2],[3,4,5],[6,7,8]]` with p0 at the (x=0, y=0) corner. This is the same indexing used by the `blocked`
field of the original dataset, so the input and output share one labelling.

## Processing summary (matches the reference code base)

1. Load each animal's joblib file (`load_dat` equivalent) and keep only cells registered that session (non-NaN).
2. Smooth the binary event trains along time with `gaussian_filter1d(sigma = 3 frames)` and average-pool
   3 frames (`AvgPool1d(3, 3)`), exactly as the authors' `fit_decoder`; multiply by 30 to get events/s.
3. Average the x,y position over the same 100 ms bins and assign it to a 25 cm partition. The 0.003% of bins
   tracked inside a walled-off partition are snapped to the nearest open partition.
4. Cut the session into non-overlapping 1-minute trials (smoothing is done before cutting, so trials have no
   filter edge artefacts).

See `CONVERSION_NOTES.md` for the full decision log, sanity checks and validation results.
