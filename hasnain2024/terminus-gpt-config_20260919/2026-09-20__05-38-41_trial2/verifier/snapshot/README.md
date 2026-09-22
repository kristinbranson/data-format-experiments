# Two-Context ALM Neural Decoder Dataset

## Overview

`converted_data.pkl` contains trial-aligned ALM spiking activity and behavioral/video labels from the two-context subset of **“Separating cognitive and motor processes in the behaving mouse.”** It includes the exact 12 sessions selected by the reference Figure 8 metadata loaders.

- **Alignment:** go-cue onset (water drop in WC)
- **Window:** −3.0 to +2.5 s
- **Bin size:** 10 ms (550 bins)
- **Sessions:** 12
- **Released subject IDs:** 7
- **Trials:** 3,116 after curation
- **Neurons:** 515 curated ALM units
- **Brain region:** ALM

## Loading

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

trial_neural = data['neural'][0][0]  # neurons x 550
trial_input = data['input'][0][0]    # 1 x 550
trial_output = data['output'][0][0]  # 6 x 550
```

## Data Fields

- `neural`: sessions → trials → float32 firing-rate matrices, `(n_neurons, 550)`.
- `input`: signed time from go cue in seconds, `(1, 550)`.
- `output`: six categorical series, `(6, 550)`.
- `subjects`, `subject_idx`: released subject IDs and session mapping.
- `brain_regions`, `brain_region_idx`: all retained units map to ALM.
- `metadata`: timing, alignment, curation, and per-session thresholds/statistics.

### Output dimensions

| Index | Name | Values |
|---:|---|---|
| 0 | lick direction | 0 left, 1 right, 2 none |
| 1 | behavioral context | 0 DR, 1 WC |
| 2 | outcome | 0 incorrect, 1 correct, 2 ignore |
| 3 | tongue velocity | 0 below session median, 1 at/above median, 2 not visible |
| 4 | paw velocity | 0 below session median, 1 at/above median, 2 not visible |
| 5 | motion energy | 0 below session median, 1 at/above median, 2 no video |

Trial-level labels are repeated over time. Velocity and motion-energy labels are time-varying.

## Processing Summary

1. Use the 12 two-context sessions and ALM probes listed by the Figure 8 metadata loaders.
2. Align spikes by subtracting each trial's go-cue timestamp.
3. Bin at 10 ms from −3 to +2.5 s and convert counts to Hz.
4. Apply the reference causal `gausswin(15)` smoother with reflected boundaries.
5. Retain manually curated quality labels (`poor`, `fair`, `good`, `great`, `excellent`, `multi`) with mean firing rate strictly above 1 Hz; require at least 10 units/session.
6. Exclude early, stimulated, invalid-ephys, or invalid-outcome trials. Ignore/no-response trials are retained because they are explicitly required decoder classes.
7. Align video-derived speed and motion energy to the same bins. Split finite values at each session's 50th percentile and preserve missing visibility/video as class 2.

The final count is 515 units versus 522 reported for the paper's two-context set. This 1.3% discrepancy is documented in `CONVERSION_NOTES.md`; forcing the count would require retaining explicit garbage/unknown labels or violating the strict >1 Hz rule. The well-isolated single-unit count of 214 is reproduced exactly.

## Validation

`verification_full_out.txt` reports no errors or warnings. Full decoder validation balanced accuracies were:

| Output | Validation balanced accuracy | Chance |
|---|---:|---:|
| Lick direction | 0.5841 | 0.3333 |
| Behavioral context | 0.7465 | 0.5000 |
| Outcome | 0.5740 | 0.3333 |
| Tongue velocity | 0.5696 | 0.3333 |
| Paw velocity | 0.5496 | 0.3333 |
| Motion energy | 0.6545 | 0.3333 |

See `CONVERSION_NOTES.md` for decisions, source comparisons, independent `np.allclose` checks, and critical reviews.

## Reproduction

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```
