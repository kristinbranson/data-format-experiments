# MAP Dataset Conversion: NWB to Decoder Format

Converts MAP (Mesoscale Activity Project) brain-wide neural recordings from NWB format into a Python dictionary structure for neural decoder training.

## Source Data
- **Dataset**: DANDI:000363 (Chen et al. 2024)
- **Method Paper**: Wang, Kurgyis, Chen et al. 2025
- **Task**: Auditory delayed response (tone instruction -> 1.2s delay -> go cue -> lick response)
- **Data**: 28 subjects, 173 sessions, 69,453 good units, 90,605 trials

## Files
| File | Description |
|------|-------------|
| `convert_data.py` | Main conversion script |
| `converted_data.pkl` | Full converted dataset (~12 GB) |
| `sample_data.pkl` | Sample (5 sessions, ~280 MB) |
| `train_decoder.py` | Decoder training script (provided) |
| `decoder.py` | Decoder model (provided) |
| `CONVERSION_NOTES.md` | Detailed conversion notes and decisions |

## Output Format
```python
{
    'neural': [session_list of [trial_list of ndarray(n_neurons, 80)]],
    'input': [session_list of [trial_list of ndarray(2, 80)]],
    'output': [session_list of [trial_list of ndarray(4, 80)]],
    'subjects': ['440956', '440957', ...],
    'subject_idx': [0, 0, 1, ...],  # per session
    'brain_regions': ['ACA', 'ACB', 'AI', ...],
    'brain_region_idx': [session_list of [region_idx per neuron]],
    'input_names': ['time_from_tone_onset', 'photostim_active'],
    'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position'],
    'output_values': {
        'choice': {0: 'left', 1: 'right', 2: 'no_lick'},
        'outcome': {0: 'ignore', 1: 'miss', 2: 'hit'},
        'early_lick': {0: 'no', 1: 'yes'},
        'tongue_y_position': {0: 'low', 1: 'mid', 2: 'high', 3: 'not_visible'}
    },
    'metadata': {...}
}
```

## Temporal Alignment
- Aligned to go cue onset (t=0)
- Window: -2.5s to +1.5s (80 bins x 50ms)
- Tone onset: -1.85s (fixed task protocol)
- Delay onset: -1.2s
- Go cue: 0s

## Usage
```bash
# Full conversion
python -u convert_data.py converted_data.pkl --full

# Sample conversion (5 sessions)
python -u convert_data.py sample_data.pkl --sample

# Verify data format
python -u train_decoder.py converted_data.pkl --verify-only

# Train decoder
python -u train_decoder.py converted_data.pkl --plot-samples
```

## Decoder Results
| Output | Val Balanced Acc | Chance | Above Chance |
|--------|-----------------|--------|-------------|
| choice | 0.686 | 0.333 | 2.06x |
| outcome | 0.661 | 0.333 | 1.98x |
| early_lick | 0.748 | 0.500 | 1.50x |
| tongue_y | 0.659 | 0.250 | 2.64x |

## Key Decisions
1. **Trial filtering**: Keep all trials except auto_water/free_water (early lick, ignore, photostim kept as decoder outputs/inputs)
2. **Session filtering**: None needed (DANDI archive already curated)
3. **Tone onset**: Fixed at -1.85s (protocol-defined, not event-based, due to early lick replay artifacts)
4. **Tongue y discretization**: Session-level percentiles (40th/60th) on visible tongue data
5. **Brain regions**: 44 regions from CCF annotations via keyword matching
