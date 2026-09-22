# CA1 geometric-deformation dataset, converted for neural decoding

Converted from **Lee, Keinath, Cianfarano & Brandon (2025). Identifying representational
structure in CA1 to benchmark theoretical models of cognitive mapping. Neuron 113(2):307-320**
(code: `georepca1`; data: Zenodo 10.5281/zenodo.13993254).

## Dataset description

Seven mice freely foraged for 40 min/day in a 75 x 75 cm arena partitioned into a 3 x 3 grid of
25 cm partitions. Across days the environment geometry was changed by blocking subsets of the
9 partitions, producing 10 distinct geometries presented in a mouse-specific random order and
repeated up to three times, always starting and ending with the open square. CA1 population
activity was recorded with miniscope calcium imaging at 30 Hz; position was tracked at 30 Hz
with DeepLabCut on a synchronised behavioural camera.

**Decoder task**: predict which of the 9 spatial partitions the mouse occupies, from CA1
activity, given the environment geometry (which partitions are blocked) as a static input.

## Key statistics

| Statistic | Value |
|-----------|-------|
| Subjects (mice) | 7 |
| Sessions (recording days) | 207 (31/31/31/21/31/31/31) |
| Unique neurons | 5,413 |
| Registered cell-sessions (rate maps) | 69,744 (matches the paper) |
| Neurons after curation | 68,862 cell-sessions, mean 332.7/session (112-562) |
| Trials | 8,056 one-minute trials (mean 38.9/session) |
| Timepoints | 2,551,604 bins of 100 ms (4,253 min of locomotion) |
| Brain region | CA1 (single region) |
| Decoder input | 9 binary flags: is partition p blocked? (static per trial) |
| Decoder output | position_bin, 9 classes (3*ybin + xbin), time-varying |
| Validation balanced accuracy | 0.663 (chance 0.111) |

## Processing summary

1. **Neural**: the released rise-extracted **binary** event traces (1 = significant transient
   rising phase) are treated as firing rate, exactly as in the paper. Per session the trace is
   smoothed with `gaussian_filter1d(sigma = 3 frames)` and average-pooled over 3 frames
   (100 ms bins), reproducing the reference `fit_decoder`, then scaled by 30 to give Hz.
2. **Neuron curation**: cells registered on that day (non-NaN trace) with more than 5 events
   during locomotion, i.e. the reference `decode_position_within` cell_threshold = 5 rule.
3. **Timepoint curation**: bins whose mean speed is <= 5 cm/s are dropped (reference
   v_thresh = 5 with speed smoothed by `gaussian_filter1d(sigma = 5 frames)`).
4. **Output**: mean position within each 100 ms bin, floor-divided by 25 cm -> class
   3*ybin + xbin in 0..8. The bin edges coincide with the physical partition walls.
5. **Input**: the per-session `blocked` field as a 9-d binary vector (1 = blocked).
6. **Trials**: consecutive 1-minute windows of session time (600 bins); trailing partial
   window dropped; windows with fewer than 30 moving bins dropped.

## How to load and use

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][session][trial]    # (n_neurons, T) float32, event rate in Hz
geom   = data['input'][session][trial]     # (9,) float32, 1 = partition blocked
pos    = data['output'][session][trial]    # (1, T) int64, spatial-bin class 0..8
animal = data['subjects'][data['subject_idx'][session]]
info   = data['metadata']['session_info'][session]
```

Train/evaluate the reference decoder:

```bash
python /app/train_decoder.py /app/converted_data.pkl
```

## Output format specification

```
data['neural'][session][trial]   : (n_neurons, T) float32  CA1 event rate (Hz), 100 ms bins
data['input'][session][trial]    : (9,) float32            blocked_partition_0 .. _8
data['output'][session][trial]   : (1, T) int64            position_bin, values 0..8
data['subjects']                 : 7 animal IDs
data['subject_idx']              : (207,) int
data['brain_regions']            : ['CA1']
data['brain_region_idx'][session]: (n_neurons,) int, all zeros
data['input_names']              : ['blocked_partition_0', ..., 'blocked_partition_8']
data['output_names']             : ['position_bin']
data['output_values']            : [['x0y0','x1y0','x2y0','x0y1', ..., 'x2y2']]
data['metadata']                 : task_description, time_bin_size (100.0 ms),
                                   temporal_alignment_event, off_start (0.0 s), off_end (60.0 s),
                                   session_info, curation rules, neural_data_type, arena sizes
```

Files: `convert_data.py` (conversion script), `CONVERSION_NOTES.md` (all decisions,
validation and checks), `cache/` (exploration and sanity-check scripts).
