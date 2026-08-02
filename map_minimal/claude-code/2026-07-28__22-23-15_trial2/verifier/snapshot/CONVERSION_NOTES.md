# Conversion Notes: MAP Dataset to Decoder Format

## Source Data
- **Dataset**: "Brain-wide neural activity underlying memory-guided movement" (MAP dataset)
- **Format**: 174 NWB files (Neurodata Without Borders)
- **Task**: Auditory delayed response task with 3kHz/12kHz tones, 1.2s delay, Go cue, lick left/right response

## Processing Parameters
- **Temporal alignment**: Go cue onset (t=0)
- **Time window**: -2.5s to +1.5s (4s total)
- **Bin size**: 50ms non-overlapping bins -> 80 time bins
- **Neural data**: Spike counts per bin, converted to firing rates (spikes/s)

## Filtering Criteria

### Session Filtering
- Behavioral performance > 65% correct on control (non-photostimulation) trials
- At least 50 correct lick-left AND 50 correct lick-right control trials
- Performance computed on ALL session trials (not just recorded subset)
- **Result**: 144 of 174 sessions passed (30 skipped: 21 low performance, 6 insufficient correct trials per side, 1 no good units, 2 other)

### Unit Filtering
- Only units with `classification == 'good'` (from quality control classifiers)
- Units must have a valid CCF annotation (`anno_name`) that maps to one of 14 brain regions
- **Result**: 57,935 good units across 144 sessions (paper reports 69,943 across 173 sessions; difference due to session filtering and NWB file count)

### Trial Filtering
- Exclude `auto_water` and `free_water` trials
- Keep all other trials including early lick and no-response (ignore) trials
- Only include trials covered by neural recordings (via `obs_intervals`)

## Key Decisions

### Tone Onset Timing
Tone onset is consistently 1.85s before Go cue across all sessions. This was determined by comparing `sample_start_times` to `go_start_times` in the NWB trial tables. The value matches the task structure: sample epoch (3 tones x 150ms + 2 gaps x 100ms = 650ms) + delay epoch (1200ms) = 1850ms.

### obs_intervals Alignment
Each unit's `obs_intervals` defines which trials have valid neural coverage. The first obs_interval does NOT always correspond to trial 0 — some sessions start recording mid-session. We match by finding the trial whose `start_time` is closest to the first `obs_interval` start time, then assign sequential trial indices from there.

### Brain Region Mapping
CCF (Allen Brain Atlas Common Coordinate Framework) annotations are mapped to 14 broad regions matching the original paper's analysis:
- **ALM**: Secondary motor area
- **OtherCortex**: Primary motor, somatosensory, visual, auditory, cingulate, etc.
- **Orbital**: Orbital areas, Frontal pole
- **Striatum**: Caudoputamen, nucleus accumbens
- **Pallidum**: Globus pallidus, septal nuclei, bed nuclei, substantia innominata
- **Thalamus**: All thalamic nuclei including paracentral, parafascicular, anterodorsal, anteromedial, rhomboid, perireunensis, habenula, geniculate
- **Hypothalamus**: Hypothalamic areas, zona incerta, subthalamic nucleus
- **Hippocampus**: CA fields, dentate gyrus, subiculum, entorhinal
- **Olfactory**: Olfactory areas, piriform, taenia tecta
- **CorticalSubplate**: Amygdala, claustrum, endopiriform
- **Midbrain**: Superior/inferior colliculus, substantia nigra, VTA, PAG
- **Pons**: Pontine nuclei, parabrachial, locus ceruleus
- **Medulla**: Reticular nuclei, trigeminal, vestibular, facial motor
- **Cerebellum**: Lobules, deep cerebellar nuclei

An initial run identified 835 thalamic units whose CCF names lacked the word "thalamus" (e.g., "Paracentral nucleus", "Parafascicular nucleus"). These were added as explicit matches.

### Decoder Inputs
1. **time_from_tone_onset** (continuous): Time of each bin center relative to tone onset. Range: [-0.6, 3.3]s. Same for all trials (deterministic from bin structure).
2. **photostimulation** (binary): 1 if photostimulation trial, 0 otherwise. From NWB `photo_stim_type` column.

### Decoder Outputs
1. **choice** (binary): 0=left, 1=right. From NWB `trial_instruction` column.
2. **outcome** (3 classes): 0=ignore (no response), 1=miss (wrong lick), 2=hit (correct). Derived from `outcome` and `early_lick` columns.
3. **early_lick** (binary): 0=no early lick, 1=early lick. From NWB `early_lick` column.
4. **tongue_y_position** (3 classes): Discretized tongue vertical position from DeepLabCut tracking. 0=low, 1=mid, 2=high. Computed per-session using 33rd/67th percentile thresholds on valid (likelihood > 0.9) tongue_y values. Trials with no valid tongue data default to class 0 (low).

## Validation Results

### Full Dataset (144 sessions, 74,769 trials, 57,935 units)
Decoder validation balanced accuracy (chance level in parentheses):
- **choice**: 0.7221 (0.5000)
- **outcome**: 0.6597 (0.3333)
- **early_lick**: 0.7484 (0.5000)
- **tongue_y_position**: 0.5370 (0.3333)

All outputs decode well above chance, confirming meaningful neural signal in the converted data.

### Sample Dataset (5 sessions, 1,931 trials, 2,163 units)
- **choice**: 0.7723 (0.5000)
- **outcome**: 0.5664 (0.3333)
- **early_lick**: 0.6314 (0.5000)
- **tongue_y_position**: 0.4668 (0.3333)

### Data Format Warnings
- Session 0, trial 159: all neural data is zero (last trial of a recording subset, expected edge case)
- Session 131: tongue_y_position distribution is (0.991, 0.003, 0.006) — nearly all "low", suggesting poor tongue tracking in this session

## Output Distribution
- **choice**: ~50/50 left/right (balanced by design)
- **outcome**: 10.8% ignore, 15.3% miss, 73.9% hit
- **early_lick**: 88.5% no, 11.5% yes
- **tongue_y_position**: 38.5% low, 19.0% mid, 42.4% high

## Region Distribution
| Region | Units |
|--------|-------|
| Thalamus | 11,107 |
| Orbital | 10,530 |
| Midbrain | 6,253 |
| ALM | 6,046 |
| Striatum | 5,851 |
| OtherCortex | 5,801 |
| Olfactory | 3,767 |
| Medulla | 2,635 |
| Hippocampus | 1,652 |
| Cerebellum | 1,543 |
| Pallidum | 1,039 |
| Hypothalamus | 749 |
| CorticalSubplate | 623 |
| Pons | 339 |
