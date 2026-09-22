# MAP dataset → neural-decoder format

Converted version of the **Mesoscale Activity Map** dataset (DANDI:000363) for training neural
decoders that predict behaviour from brain-wide Neuropixels activity.

- **Data paper**: Chen, S. *et al.* "Brain-wide neural activity underlying memory-guided movement", *Cell* 187 (2024).
- **Analysis paper**: Wang, Z. A.\*, Kurgyis, B.\* *et al.* "Brain-wide analysis reveals movement encoding structured across and within brain areas", *Nat. Neurosci.* (2025). Reference code in `/app/code`.
- **Source files**: 174 NWB files in `/app/data`, read with `pynwb`.

## Dataset description

Head-fixed mice perform an **auditory delayed-response task**: a tone (3 kHz or 12 kHz, three
150 ms pips) presented during the sample epoch instructs a lick to the left or right port; after a
1.2 s delay an auditory go cue (6 kHz, 0.1 s) allows the response (1.5 s answer period). In ~20 %
of trials ALM is photoinhibited (5.5 mW, 0.5 s, ending before the go cue). Orofacial movements are
tracked with DeepLabCut from a 300 Hz side-view camera.

## Key statistics of the converted data

| Quantity | Value |
|---|---|
| Sessions | 173 (of 174 files; one has no QC-passing unit) |
| Subjects (mice) | 28 |
| Neurons (QC-classifier `good` units) | 69,453 (mean 401/session, range 90–923) |
| Trials | 89,544 (mean 518/session, range 159–796) |
| Time bins | 80 × 50 ms, −2.5 s → +1.5 s around the go cue |
| Brain regions | 28 labels = hemisphere × 14 coarse CCF groups |
| Choice | left 42.9 %, right 42.2 %, no lick 14.8 % |
| Outcome | hit 68.5 %, miss 16.7 %, ignore 14.8 % |
| Early lick | 11.6 % |
| Tongue visible | 25.1 % of bins |
| Photostim trials | 20.0 % |

## How to load

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][session][trial]   # (n_neurons, 80) firing rate in Hz
inputs = data['input'][session][trial]    # (2, 80)
outputs = data['output'][session][trial]  # (4, 80) integer class labels
```

## Format specification

| Key | Type / shape | Meaning |
|---|---|---|
| `neural` | list[session] of list[trial] of `float32 (n_neurons, 80)` | firing rate (Hz) = spike count in each 50 ms bin / 0.05 s, aligned to the go cue |
| `input` | list[session] of list[trial] of `float32 (2, 80)` | `input_names = ['time_from_tone_onset', 'photostim_on']`; row 0 = seconds since the tone (sample epoch) onset at each bin centre; row 1 = 1 if ALM photoinhibition overlaps the bin |
| `output` | list[session] of list[trial] of `int64 (4, 80)` | `output_names = ['choice', 'outcome', 'early_lick', 'tongue_y']` |
| `output_values` | list of lists | `choice`: left / right / no lick; `outcome`: ignore / miss / hit; `early_lick`: no / yes; `tongue_y`: `<40th pct` / `40–60th pct` / `>60th pct` / `not visible` |
| `subjects`, `subject_idx` | list[str] (28), `int64 (173,)` | mouse ids and per-session index |
| `brain_regions`, `brain_region_idx` | list[str] (28), list of `int64 (n_neurons,)` | e.g. `left ALM`, `right Thalamus`; hemisphere from CCF ML coordinate (midline 5,700 µm) |
| `metadata` | dict | task description, binning, alignment, curation rules and a per-session `session_info` record (identifier, subject, neuron/trial counts, dropped-trial counts, tongue percentiles, photostim counts, window-coverage fractions, timings) |

## Processing summary

1. **Neurons**: only units with `units.classification == 'good'` (the output of the region-specific
   spike-sorting QC classifiers described in the accompanying white paper and used by the
   reference pipeline). Per-area totals reproduce the published counts exactly
   (striatum 7,664; midbrain 7,495; thalamus 12,808; medulla 2,928).
2. **Trials**: auto-water and free-water trials removed (reward independent of the animal's action,
   as in the reference `get_regular_trial_mask`); trials without any ephys (acquisition gaps)
   removed. Early-lick, error (miss), no-response (ignore) and photostimulation trials are kept
   because the decoder task requires them.
3. **Alignment**: every stream is aligned to `BehavioralEvents/go_start_times`.
4. **Binning**: 80 non-overlapping 50 ms bins from −2.5 s to +1.5 s; rates in Hz.
5. **Tongue**: DLC side-camera y position averaged per bin over frames with likelihood > 0.5 after
   5σ velocity-outlier rejection; discretised with per-session 40th/60th percentiles; bins with no
   visible frame are class 3.

## Reproducing

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full          # ~45 s, 11.9 GB output
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

## Decoder performance (validation balanced accuracy, full dataset)

| Output | Chance | Accuracy |
|---|---|---|
| choice | 0.333 | 0.683 |
| outcome | 0.333 | 0.660 |
| early_lick | 0.500 | 0.752 |
| tongue_y | 0.250 | 0.652 |

See `CONVERSION_NOTES.md` for the full decision log, validation and consistency checks.
