# CA1 geometric-deformation dataset — decoder-ready conversion

`converted_data.pkl` holds the CA1 miniscope recordings of **Lee, Keinath, Cianfarano & Brandon (2025),
*Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping*,
Neuron 113(2):307-320** ([doi](https://doi.org/10.1016/j.neuron.2024.10.027);
data [doi:10.5281/zenodo.13993254](https://doi.org/10.5281/zenodo.13993254)), reformatted for the neural
decoder in `train_decoder.py`.

## Dataset description

Seven mice foraged freely for 40 min a day in a 75 × 75 cm arena divided into a 3 × 3 grid. Different
subsets of the nine partitions were walled off with 25 cm inserts to create **10 distinct geometries**
(`square, o, t, u, rectangle, +, i, l, bit donut, glenn`). One geometry was presented per day; each
animal's randomly ordered sequence started and ended with the square and was repeated up to three times
(31 days, 21 for QLAK-CA1-51). Dorsal CA1 populations were imaged with a UCLA miniscope (GCaMP6f, GRIN
lens) at 30 Hz, simultaneously with overhead behavioural video on the same DAQ.

**Decoding task**: from CA1 population activity plus the arena geometry, predict which of the nine
spatial bins the animal occupies.

## Key statistics

| | |
|---|---|
| Subjects | 7 |
| Sessions (recording days) | 207 |
| Trials (1-min blocks) | 8,148 (mean 39.4 per session) |
| Unique neurons | 5,413 |
| Registered cell-sessions | 69,744 → **68,862** after curation |
| Neurons per session | mean 332.7 (112–562) |
| Time bin | 100 ms (3 frames at 30 Hz) |
| Timepoints per trial | mean 315 (30–561) |
| Total retained time | 71.4 h of running |
| Brain region | CA1 |
| File size | 3.44 GB |

Decoder performance (`train_decoder.py`, full dataset): **validation balanced accuracy 0.676** for the
9-way position bin, versus 0.111 chance (training 0.768).

## Loading and using the data

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# neural[session][trial] : (n_neurons, T) float32 -- event rate per 100 ms bin
# input[session][trial]  : (9,)        float32 -- 1 = partition blocked this session
# output[session][trial] : (1, T)      int64   -- spatial bin 0-8 at each timepoint

x = data['neural'][0][0]          # session 0, trial 0
geom = data['input'][0][0]
ybin = data['output'][0][0][0]

print(data['subjects'][data['subject_idx'][0]])          # 'QLAK-CA1-08'
print(data['metadata']['session_info'][0]['environment']) # 'square'
```

```bash
python train_decoder.py converted_data.pkl              # train and evaluate
python train_decoder.py converted_data.pkl --verify-only # structure + summary only
python convert_data.py out.pkl --full                   # regenerate from /app/data (~45 s)
python convert_data.py out.pkl --sample --show-processing  # 2 sessions + diagnostic plots
```

## Output format specification

```python
data = {
  'neural': [[ (n_neurons, T) float32, ... ], ...],   # per session, per trial
  'input':  [[ (9,) float32, ... ], ...],
  'output': [[ (1, T) int64,  ... ], ...],
  'subjects': ['QLAK-CA1-08', ... ],                  # 7 mouse IDs
  'subject_idx': (207,) int64,
  'brain_regions': ['CA1'],
  'brain_region_idx': [ (n_neurons,) int64, ... ],    # all zeros: every cell is CA1
  'input_names':  ['blocked_x0y0', ..., 'blocked_x2y2'],
  'output_names': ['position_bin'],
  'output_values': [['x0y0', ..., 'x2y2']],           # 9 categories
  'metadata': {...},
}
```

**Spatial bins.** `bin = 3 * floor(y / 25 cm) + floor(x / 25 cm)`, i.e. the `[[0,1,2],[3,4,5],[6,7,8]]`
layout the source dataset uses for its `blocked` field. `input_names[i]` and `output_values[0][i]` refer
to the *same* partition, so the input tells the decoder which output classes are reachable.

**Neural units.** The distributed `trace` is the binarised rising phase of each calcium transient — the
vector the paper treats as the firing rate. It is smoothed with a gaussian of σ = 3 frames and
average-pooled over 3 frames, exactly as in the paper's own decoder
(`georepca1/src/utils.py:fit_decoder`), giving mean event occupancy per 100 ms bin (grand mean
0.101 events/s/cell).

**Trials.** Contiguous 60 s blocks of each 40 min session; a trailing partial block is kept when it spans
at least 10 s. Only bins in which the animal ran faster than 5 cm/s are written, so trials have unequal
lengths. `metadata['session_info']` records the geometry, blocked partitions, cell counts, trial count and
running fraction of every session.

**Curation.** Cells must be registered that session and have more than 5 events while running
(`decode_position_within`'s `cell_threshold=5`); no place-cell selection, since the paper explicitly
includes all cells. Timepoints are restricted to running (`v_thresh=5` cm/s, velocity smoothed with
σ = 5 frames). Trials left with under 3 s are dropped. All 7 animals and all 207 sessions are retained.

## Files

| File | |
|---|---|
| `convert_data.py` | the conversion script |
| `converted_data.pkl` | full converted dataset |
| `sample_data.pkl` | 2-session sample |
| `CONVERSION_NOTES.md` | every decision, check and validation result |
| `processing_*.png` | per-step diagnostic figures |
| `sample_trials.png`, `predictions.png` | decoder input/output and held-out predictions |
| `cache/` | verification scripts and logs (see `cache/README_CACHE.md`) |
