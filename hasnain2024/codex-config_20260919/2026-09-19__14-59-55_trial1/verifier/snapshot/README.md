# Neural Decoder Dataset

This workspace contains a decoder-ready conversion of the two-context ALM electrophysiology cohort from *Separating cognitive and motor processes in the behaving mouse*. Trials are aligned to go cue onset (the matched response/water-drop event in WC), sampled in 5-ms bins from −2.5 to +2.5 s, and retain the paper's spike curation, causal Gaussian smoothing, and video synchronization.

## Quick start

```python
import pickle

with open("/app/converted_data.pkl", "rb") as stream:
    data = pickle.load(stream)

# First session, first trial
neural = data["neural"][0][0]  # (neurons, 1000), float32 spikes/s
decoder_input = data["input"][0][0]  # (1, 1000), float32 seconds from go cue
targets = data["output"][0][0]  # (6, 1000), int8 categories
```

Reproduce the conversion or validate/train:

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

## Dataset summary

| Item | Value |
|---|---:|
| Sessions | 12 |
| Explicit source subject IDs | 7 |
| Retained trials | 3,116 |
| Retained ALM session-units | 521 |
| Units/session | 27–67 (mean 43.42) |
| Trials/session | 210–390 (mean 259.67) |
| Time points/trial | 1,000 |
| Bin size/window | 5 ms; `[−2.5,+2.5)` s |

The paper reports 522 total units and 214 well-isolated single units from six mice. Applying its executable quality and strict >1-Hz rule to the released files yields 521 total units and exactly 214 well-isolated units. The files contain seven distinct animal IDs, which are preserved rather than collapsed. These release/paper differences are analyzed in [CONVERSION_NOTES.md](/app/CONVERSION_NOTES.md).

## Format and labels

The pickle follows the requested nested session/trial dictionary. `neural`, `input`, and `output` have parallel session and trial lists. `brain_region_idx` maps every retained unit to the sole region, `ALM`; `subject_idx` maps sessions into `subjects`.

Output rows are:

| Row | Name | Categories |
|---:|---|---|
| 0 | Lick direction | `0 left`, `1 right`, `2 none` |
| 1 | Behavioral context | `0 WC`, `1 DR` |
| 2 | Outcome | `0 incorrect`, `1 correct`, `2 ignore` |
| 3 | Tongue velocity | `0 below median`, `1 at/above median`, `2 not visible` |
| 4 | Paw velocity | `0 below median`, `1 at/above median`, `2 not visible` |
| 5 | Motion energy | `0 below median`, `1 at/above median`, `2 no video` |

Rows 0–2 are per-trial labels broadcast across time. Rows 3–5 vary over time. Velocity is Euclidean x/y derivative magnitude; thresholds are session medians computed only over visible/valid retained samples. There are no absent-video traces after applying the reference fallback, so motion class 2 is defined but not observed.

Early-lick and stimulation-enabled trials are excluded. Hit, miss, and ignore trials are retained because all requested outcome and lick classes must be represented. Actual lick direction is used: a miss is opposite the instructed side and an ignore is `none`.

## Validation and decoder results

The official verifier reports no errors or warnings. Independent raw-file audits exactly reconstruct selected neural, input, behavior, tongue, paw, and motion traces with `np.allclose`; all 3,116 static labels are also checked directly.

| Target | Full validation balanced accuracy |
|---|---:|
| Lick direction | 0.5566 |
| Behavioral context | 0.7143 |
| Outcome | 0.5420 |
| Tongue velocity | 0.5627 |
| Paw velocity | 0.5228 |
| Motion energy | 0.7496 |

Training loss decreases from 7.8210 to 0.7821. A supplemental paper-style time-resolved check peaks at 0.9396 for choice and 0.9624 for context, with shuffled controls near 0.50. Complete processing decisions, distributions, edge-case fixes, and review results are in [CONVERSION_NOTES.md](/app/CONVERSION_NOTES.md); auxiliary audits are described in [cache/README_CACHE.md](/app/cache/README_CACHE.md).

