# CA1 geometric-deformation dataset - decoder-ready conversion

Converted from **Lee, Keinath, Cianfarano & Brandon (2025), "Identifying representational
structure in CA1 to benchmark theoretical models of cognitive mapping", Neuron 113(2):307-320**
(data: Zenodo doi:10.5281/zenodo.13993254; code: `github.com/jquinnlee/georepca1`).

## Dataset description

Seven mice (4 male, 3 female, C57Bl/6) freely explored a 75 x 75 cm arena that was divided into a
3 x 3 grid of 25 cm partitions. On each of up to 31 daily 40-minute sessions the arena took one of
**10 geometries** produced by blocking a subset of the partitions (`square`, `o`, `t`, `u`,
`rectangle`, `+`, `i`, `l`, `bit donut`, `glenn`). Each mouse ran the 10-geometry sequence up to
three times in a randomised order, always starting and ending with the square.

CA1 populations were recorded with a UCLA miniscope (one-photon calcium imaging) at 30 Hz,
simultaneously with head-tracked position (DeepLabCut). The neural signal distributed by the
authors, and used here, is the **binarised rising phase of the calcium transients**, which the
paper treats as the firing rate.

| Statistic | Value |
|-----------|-------|
| Subjects | 7 |
| Sessions | 207 (31 per animal, 21 for QLAK-CA1-51) |
| Unique neurons | 5,413 |
| Neuron-sessions in the converted data | 68,862 (98.7% of the 69,744 registered cell-sessions) |
| Neurons per session | 112 - 562 (mean 333) |
| Trials | 8,147 (1-minute blocks; 22 - 40 per session, median 40) |
| Time bin | 100 ms |
| Timepoints per trial | 30 - 561 (mean 316; bins with speed <= 5 cm/s are removed) |
| Brain region | CA1 |

## Decoder task

- **Input** (9 values, static per trial): `blocked_partition_0` ... `blocked_partition_8`,
  1 if that partition of the 3 x 3 grid is blocked in this session's geometry, 0 otherwise.
- **Output** (1 variable, 9 classes, time-varying): `position_3x3`, the partition the mouse
  occupies, encoded as `class = 3 * ybin + xbin` with `xbin, ybin` in {0, 1, 2} - the same
  partition indexing as the dataset's own `blocked` field.

Performance of the provided decoder (`train_decoder.py`) on the full dataset:

| Output | Training balanced acc. | Validation balanced acc. | Chance |
|--------|-----------------------|--------------------------|--------|
| position_3x3 | 0.825 | **0.662** | 0.111 |

## How to load and use the converted data

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][0][0]     # session 0, trial 0: (n_neurons, n_timepoints) float32, events/s
geometry = data['input'][0][0]    # (9,) float32, 1 = partition blocked
position = data['output'][0][0]   # (1, n_timepoints) int64, values 0-8

print(data['subjects'][data['subject_idx'][0]])          # mouse id of session 0
print(data['metadata']['session_info'][0]['environment']) # geometry name of session 0
```

## Output format specification

```
data = {
  'neural':   [session][trial] -> (n_neurons, T) float32, calcium event rate in events/s
  'input':    [session][trial] -> (9,) float32, blocked partitions (static per trial)
  'output':   [session][trial] -> (1, T) int64, 3x3 partition index of the mouse
  'subjects': ['QLAK-CA1-08', ..., 'QLAK-CA1-75']          # 7 mice
  'subject_idx':      (207,) int64
  'brain_regions':    ['CA1']
  'brain_region_idx': [session] -> (n_neurons,) int64, all zeros
  'input_names':  ['blocked_partition_0', ..., 'blocked_partition_8']
  'output_names': ['position_3x3']
  'output_values': [['partition_0_(x0,y0)', ..., 'partition_8_(x2,y2)']]
  'metadata': {task_description, time_bin_size (100.0 ms), temporal_alignment_event,
               off_start (0.0), off_end (60.0), neural_signal, speed_threshold_cm_s,
               neuron_inclusion, trial_definition, geometries, session_info, ...}
}
```

`metadata['session_info'][s]` gives, per session: `session_id`, `animal`, `day`, `environment`,
`sequence`, `n_neurons`, `n_cells_registered`, `n_cells_file`, `n_trials`, `n_frames_raw`,
`n_bins_kept`.

## Processing summary

All processing follows the reference code (`/app/code/georepca1/src/utils.py`):

1. **Load** `trace` (binary events, 30 Hz) and `position` (cm) per session from the MATLAB files.
2. **Speed**: `gaussian_filter1d(|diff(position)| * 30, sigma = 5 frames)`; only periods faster
   than 5 cm/s are kept (`v_thresh` of `decode_position_within`).
3. **Neuron curation**: keep cells registered on that day with more than 5 events during
   locomotion (`cell_threshold` of `decode_position_within`). No place-cell selection - the paper
   includes all cells.
4. **Temporal binning**: `gaussian_filter1d(sigma = 3 frames)` then average pooling over 3 frames
   (the `AvgPool1d(3)` of `fit_decoder`) -> 100 ms bins; scaled to events/s. Position and speed are
   pooled on the identical grid.
5. **Spatial discretisation**: `floor(position / 25 cm)`, class = `3 * ybin + xbin`. The 0.005% of
   samples that tracking noise places inside a blocked partition are snapped to the nearest open
   partition, the 3 x 3 analogue of the cleaning step in `decode_position_within`.
6. **Trials**: consecutive 60 s blocks; a final block shorter than 30 s is discarded, and a trial
   with less than 3 s of locomotion is discarded.

**Validation**: re-running the paper's own Gaussian Naive Bayes decoder on the converted neural
data reproduces the published 15 x 15 Bayesian position-decoding error to within 0.7 cm
(13.71 cm vs 12.99 cm in `precomputed_results/within_decoding`), session by session.

## Files

| File | Contents |
|------|----------|
| `convert_data.py` | the conversion script (`--full`, `--sample`, `--show-processing`) |
| `converted_data.pkl` | the full converted dataset (3.4 GB) |
| `sample_data.pkl` | 2 sessions, for quick tests |
| `CONVERSION_NOTES.md` | every decision, check and validation result |
| `conversion_*_out.txt`, `verification_*_out.txt`, `train_decoder_*_out.txt` | logs |
| `processing_<session>.png` | step-by-step processing figures |
| `cache/` | analysis and verification scripts used during the conversion |
