# Visual Behavior 2P -> neural-decoder dataset

`/app/converted_data.pkl` contains the Allen Brain Observatory **Visual Behavior 2-photon** dataset
(AllenSDK `VisualBehaviorOphysProjectCache`, manifest v1.1.0, local cache in `/app/data`) reformatted for
trial-based neural decoding.

## Dataset description

Head-fixed mice perform a go/no-go **visual change detection** task while 2-photon calcium imaging is performed in
visual cortex (VISp / VISl). Natural images are flashed for 250 ms every 750 ms (500 ms gray inter-stimulus interval);
mice earn water by licking when the image identity changes. Trials are **go** (real change) or **catch** (sham change);
aborted (early lick) and auto-rewarded (free reward) trials are excluded.

| Quantity | Value |
|---|---|
| Sessions | 171 (active behavior ophys sessions; Mesoscope planes of a session merged) |
| Mice | 38 |
| Neurons | 29,168 (VISp 29,006, VISl 162); mean 171/session (6-666) |
| Trials | 41,904 go+catch trials (mean 245/session, 39-393) |
| Time bins | 24 x 250 ms per trial, spanning [-2.25 s, +3.75 s] around the stimulus change |
| Neural signal | dF/F (Allen pipeline), averaged within each 250 ms bin |
| Cre lines | Slc17a7 (excitatory), Sst, Vip |

## Loading

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][session][trial]   # (n_neurons, 24) float32, dF/F per 250 ms bin
out    = data['output'][session][trial]   # (5, 24) int64 class labels
inp    = data['input'][session][trial]    # (0, 24) - this task has no decoder inputs
```

## Output format

| idx | `output_names` | Type | Classes (`output_values`) |
|---|---|---|---|
| 0 | `image_identity` | time-varying | 16 natural images (`im000` ... `im106`; 8 per session, image set A or B) - identity of the image currently being presented, held through the gray interval and through omissions |
| 1 | `image_change` | time-varying | `no_change`, `change` - 1 in the three bins (750 ms) following a real image change |
| 2 | `running_speed_quintile` | time-varying | `speed_q1..q5` - mean running speed per bin, binned at the dataset-wide 20/40/60/80th percentiles (edges 0.019, 2.52, 19.26, 33.85 cm/s) |
| 3 | `pupil_diameter_quintile` | time-varying | `pupil_q1..q5` - mean pupil diameter (2*sqrt(area/pi), px) per bin, dataset-wide percentile edges 74.7, 84.2, 93.1, 105.8 |
| 4 | `trial_outcome` | static per trial (broadcast over the 24 bins) | `hit`, `miss`, `false_alarm`, `correct_reject` |

Other fields: `subjects` / `subject_idx` (mouse ids per session), `brain_regions` / `brain_region_idx`
(`VISp`/`VISl` per neuron), `input_names` (empty), and `metadata` (task description, `time_bin_size` = 250 ms,
`temporal_alignment_event`, `off_start` = -2.25, `off_end` = 3.75, quantile edges, per-session info, excluded sessions).

## Key statistics

- catch trials 12.5% of trials (whitepaper: ~12.5%), hit rate 0.37, false-alarm rate 0.15
- `image_change` is 1 in 10.9% of bins (= 0.875 go x 3/24 bins)
- running and pupil quintiles are each exactly 20% of bins
- decoder (provided `train_decoder.py`, validation balanced accuracy):
  image identity 0.453 (chance 0.0625), image change 0.662 (0.5), running 0.401 (0.2), pupil 0.468 (0.2), outcome 0.307 (0.25)
- replicating the source paper's random-forest decoders on this data gives 0.770 correct for change-vs-repeat and
  0.771 for hit-vs-miss, matching the values reported in the paper

## Reproducing

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full            # ~90 s with 16 workers
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```
Options: `--neural-signal {dff,events,filtered_events}`, `--quantile-scope {global,session}`, `--limit N`, `--nproc N`.

See `/app/CONVERSION_NOTES.md` for every conversion decision, consistency check and validation result.
