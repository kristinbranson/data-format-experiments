# CA1 geometric-deformation dataset, converted for neural decoding

Converted version of the dataset from

> Lee, J.Q., Keinath, A.T., Cianfarano, E. & Brandon, M.P. (2025).
> *Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping.*
> Neuron 113(2), 307-320. https://doi.org/10.1016/j.neuron.2024.10.027
> Data: https://doi.org/10.5281/zenodo.13993254 - Code: https://github.com/jquinnlee/georepca1

## Dataset description

Seven mice ran freely for one 40-minute session per day in a 75 x 75 cm arena that was treated as a 3 x 3
grid of 25 x 25 cm partitions. On each day a different subset of partitions was blocked off with 25 cm
walls, producing **10 distinct geometries** (square, o, t, u, rectangle, +, i, l, bit donut, glenn)
presented in a mouse-specific random order and repeated for up to three sequences (31 days; one mouse
completed 21 days). Dorsal CA1 populations were imaged with a UCLA miniscope (GCaMP6f) at 30 Hz while head
position was tracked from an overhead camera at the same rate. The provided traces are the **binarised
rising phases of calcium transients**, which the authors treat as the firing rate.

**207 sessions, 7 mice, 5,413 unique neurons, 69,744 registered cell-sessions.**

## Decoding task

| | |
|---|---|
| **neural** | CA1 population activity, events/s in 100 ms bins |
| **input** | environment geometry: 9-d binary vector, 1 = that partition is blocked (static per trial) |
| **output** | which of the 9 partitions the mouse occupies, per time bin (9 categories) |
| **trials** | consecutive 1-minute blocks of each session |

## How to load and use

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][session][trial]   # (n_neurons, T)  float32, events/s
geom   = data['input'][session][trial]    # (9,)            float32, 1 = partition blocked
pos    = data['output'][session][trial]   # (1, T)          int64,   partition index 0..8

data['subjects'][data['subject_idx'][session]]        # mouse id, e.g. 'QLAK-CA1-50'
data['output_values'][0][pos[0, t]]                   # 'r0c0' ... 'r2c2'
data['metadata']['session_info'][session]             # animal, day, geometry, cell ids, counts
```

Validate and train the reference decoder with

```bash
python train_decoder.py converted_data.pkl --verify-only     # format check + summary
python train_decoder.py converted_data.pkl --plot-samples    # train + evaluate
```

Regenerate the dataset (41 s on 7 cores) with

```bash
python -u convert_data.py converted_data.pkl --full
python -u convert_data.py sample_data.pkl --sample --show-processing   # 2 sessions + diagnostic plots
```

## Output format specification

```
data = {
  'neural':  [nsessions][ntrials] -> (n_neurons[session], T[trial]) float32   events/s
  'input':   [nsessions][ntrials] -> (9,)                           float32   1 = partition blocked
  'output':  [nsessions][ntrials] -> (1, T[trial])                  int64     partition 0..8
  'subjects':          ['QLAK-CA1-08', ... ]                        7 mice
  'subject_idx':       (nsessions,) int64
  'brain_regions':     ['CA1']
  'brain_region_idx':  [nsessions] -> (n_neurons[session],) int64   (all zeros)
  'input_names':       ['blocked_r0c0', ... 'blocked_r2c2']
  'output_names':      ['position_bin']
  'output_values':     [['r0c0', 'r0c1', 'r0c2', 'r1c0', ... 'r2c2']]
  'metadata':          {...}
}
```

Partition index = `3 * row + column`, with `column = floor(x / 25 cm)` (West to East) and
`row = floor(y / 25 cm)` (North to South) - the same ordering used for the `blocked` field of the source
data and for the partitions in the paper.

`metadata` records the task description, `time_bin_size` (100 ms), the alignment convention
(`temporal_alignment_event`, `off_start` = 0 s, `off_end` = 60 s), the processing and curation rules, and
`session_info`, a per-session record of animal, day, geometry, blocked partitions, neuron counts and the
ids of the cells that were kept.

## Processing

Follows the paper's own position-decoding analysis (`decode_position_within`, `fit_decoder`,
`test_decoder` in `georepca1/src/utils.py`):

1. **100 ms time bins**: traces are smoothed with a 3-frame Gaussian and averaged over 3 frames at 30 Hz,
   then expressed in events/s. Position is averaged over the same 3 frames.
2. **Speed filter**: bins in which the mouse moved at <= 5 cm/s are discarded (speed smoothed with a
   5-frame Gaussian). 51.9% of bins survive, so a 1-min trial holds on average 315 of 600 bins.
3. **Cell curation**: cells registered on that session (non-NaN trace) that emit more than 5 events during
   the retained bins. 68,860 of the 69,744 registered cell-sessions are kept.
4. **Spatial discretisation**: `3 * floor(y/25) + floor(x/25)`, clipped to the 3 x 3 grid. Validated
   against the source data's `blocked` field and against the authors' precomputed occupancy maps
   (correlation 1.0000 in all 207 sessions).
5. **Trials**: consecutive 1-min blocks; blocks retaining fewer than 30 bins (3 s of movement) are dropped.

See `CONVERSION_NOTES.md` for the full rationale, the reference-code comparison and all validation results.

## Key statistics

| | |
|---|---|
| Sessions | 207 (31 per mouse, 21 for QLAK-CA1-51) |
| Subjects | 7 |
| Trials | 8,163 (mean 39.4 per session, range 22-41) |
| Neurons per session | mean 332.7 (112-562); 68,860 cell-sessions; 5,374 unique cells |
| Time bin | 100 ms |
| T per trial | mean 315 bins (30-561) |
| Total timepoints | 2,574,536 (71.5 h of running) |
| Output classes | 9, fractions 0.070-0.171 |
| File size | 3.44 GB |

## Decoder performance

| Output | Chance | Training balanced acc. | Validation balanced acc. |
|--------|--------|------------------------|--------------------------|
| position_bin (9 partitions) | 0.111 | 0.826 | **0.662** |

87% of validation predictions are the correct or an adjacent partition. Expressed as the paper's metric,
the mean Euclidean error between the decoded and the true spatial bin is 11.4 cm (the paper reports
13.5 cm with 5 x 5 cm bins). Per-session accuracy correlates at r = -0.93 with the authors' own per-session
Bayesian decoding error, and reproduces both their per-animal ordering and the improvement in decoding
across the month of recording.
