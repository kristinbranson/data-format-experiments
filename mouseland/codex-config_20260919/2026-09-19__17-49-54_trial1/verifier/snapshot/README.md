# Neural Decoder Dataset

This directory contains a decoder-ready conversion of the two-photon imaging dataset accompanying *Unsupervised pretraining in biological neural networks*. The conversion preserves the supplied Suite2p deconvolved activity at native imaging samples and follows the paper's running/0–4 m corridor selection.

## Main files

- `converted_data.pkl`: complete converted dataset (172.61 GB)
- `sample_data.pkl`: two-session test subset (4.69 GB)
- `convert_data.py`: reproducible converter
- `CONVERSION_NOTES.md`: processing rationale, source comparisons, audits, and decoder results
- `verification_full_out.txt`: format-validation output
- `train_decoder_full_out.txt`: complete 200-epoch decoder log

The full pickle requires substantially more than 173 GB of available RAM when loaded because it contains Python container overhead in addition to the 172.58 GB neural payload.

## Dataset summary

| Statistic | Value |
|-----------|------:|
| Subjects | 19 |
| Sessions | 89 |
| Trials | 37,801 |
| Neural samples | 815,506 |
| Neurons across sessions | 4,691,034 |
| Neurons/session | 20,547–89,577 (mean 52,708.25) |
| Nominal native sampling interval | 314.804 ms |

Each trial contains only active-running (`ft_move > 0`) samples in the valid textured corridor (`ft_CorrSpc`, 0–4 m). Trials without one of the seven canonical stimulus labels are excluded. Neural values are the supplied non-negative Suite2p deconvolved fluorescence traces; no dF/F recomputation or normalization is applied.

## Loading and structure

```python
import pickle

with open("/app/converted_data.pkl", "rb") as handle:
    data = pickle.load(handle)

neural_trial = data["neural"][0][0]  # (neurons, time)
input_trial = data["input"][0][0]    # (4, time), float32
output_trial = data["output"][0][0]  # (4, time), int16
```

Decoder inputs, in row order, are signed time to sound cue (seconds; positive before cue), elapsed training day, time since corridor entry (seconds), and binary reward availability.

Decoder outputs are visual stimulus category (7 classes), binary licking, corridor position (`0–1`, `1–2`, `2–3`, `3–4 m`), and global running-speed quartile. The exact names and category labels are stored in `input_names`, `output_names`, and `output_values`.

The remaining fields provide subject/session mappings, per-neuron brain-region mappings (`V1`, `mHV`, `lHV`, `aHV`, or `unmapped/non-visual`), and detailed metadata.

## Reproduction and validation

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/sample_data.pkl --verify-only
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

Full validation reported no format errors or warnings. Full validation balanced accuracies were 0.4753 (visual category), 0.8462 (licking), 0.3426 (position), and 0.3890 (speed), all above uniform chance. See `CONVERSION_NOTES.md` for source-level `np.allclose` audits and interpretation.

