# Conversion Notes: MAP Dataset NWB to Decoder Format

## Source Data
- **Dataset**: MAP (Mesoscale Activity Project), Li et al.
- **Format**: NWB (Neurodata Without Borders) files
- **Location**: `/app/data/` (28 subjects, 174 sessions)
- **Task**: Auditory delayed response task with Neuropixels recordings

## Conversion Parameters
| Parameter | Value |
|-----------|-------|
| Bin width | 50 ms |
| Time window | -2.5 to +1.5 s (relative to go cue) |
| Time bins per trial | 80 |
| Min performance | >65% correct on control trials |
| Min correct per side | >= 50 lick-left and lick-right correct trials |
| QC method | Classifier-based ('good' classification in NWB) |
| Trial filtering | Exclude early lick and no-response (ignore) trials |

## Output Summary
| Metric | Full Dataset | Sample Dataset |
|--------|-------------|----------------|
| Sessions | 105 | 5 |
| Subjects | 25 | 5 |
| Total trials | 48,356 | 250 |
| Total neurons | 41,197 | 1,864 |
| File size | ~5.95 GB | ~29 MB |

## Key Decisions

### 1. Temporal Alignment
All neural data is aligned to go cue onset (time 0). The time window spans -2.5s to +1.5s, covering:
- Pre-sample baseline
- Sample epoch (tone presentation, ~1.85s before go cue)
- Delay epoch
- Go cue and response

### 2. Firing Rate Computation
Spike counts are histogrammed into 50ms bins and divided by bin width (0.05s) to get firing rates in Hz. This is simpler than the sliding kernel approach in the original preprocessing code but appropriate for the decoder task.

### 3. obs_intervals Handling
Some NWB files have units with `obs_intervals` that cover only a subset of trials (e.g., 160 obs_intervals for 480 trials). This occurs when recording was started/stopped within a session. The conversion builds a mapping from obs_intervals indices to trial table indices based on temporal overlap between observation windows and trial events. Only trials with neural data coverage are included.

### 4. Trial Filtering
Following the paper's methodology:
- **Early lick trials**: Excluded (animals licked before go cue)
- **No-response (ignore) trials**: Excluded (no lick within response window)
- After filtering, `early_lick` output is always 0 and `outcome` never has value 0 (ignore). The output_values definitions still list all possible values for completeness.

### 5. Session Selection Criteria
Applied to control (non-photostimulation) trials only:
- Performance must exceed 65%
- At least 50 correct lick-left and 50 correct lick-right trials required
- 69 sessions were excluded (skipped); 105 sessions passed

### 6. Choice Determination
- **Hit trials**: choice = instruction side (the instructed lick direction)
- **Miss trials**: choice = opposite of instruction side (animal chose wrong side)

### 7. Tone Onset (time_from_tone_onset)
The tone onset time is identified as the last `sample_start_time` before the go cue. For most sessions, this produces `time_from_tone_onset` values ranging from ~-0.6 to ~3.3 (relative to each bin center).

Notable exceptions:
- **Session 13**: One trial has unusual tone onset (~7.4s range), likely from a replay/repeated stimulus
- **Sessions 40-104** (subject SC038 and later): Range is [-1.5, 3.9], indicating a different sample-to-go-cue timing (~1.85s vs ~0.65s earlier sessions). This reflects genuine experimental variation across subjects.

### 8. Photostimulation
Binary input (0=off, 1=on). Photostim onset times from NWB are converted from absolute times to relative-to-go-cue times. If any photostim occurs within the trial window [-2.5, 1.5], photostimulation is set to 1.

### 9. Brain Region Mapping
Detailed CCF (Common Coordinate Framework) annotations are mapped to 15 major regions:
ALM, Striatum, Thalamus, Orbital, OtherCortex, Midbrain, Olfactory, Cerebellum, Hippocampus, CorticalSubplate, Pallidum, Medulla, Pons, Hypothalamus, FiberTract.

### 10. Tongue Y-Position Discretization
- Extracted from DeepLabCut tracking (side camera, Camera0)
- Only positions with tracking likelihood > 0.5 are used
- Per-session discretization using 40th and 60th percentiles:
  - 0 (low): below 40th percentile
  - 1 (mid): between 40th and 60th percentiles
  - 2 (high): above 60th percentile
- Time bins without valid tracking data default to 0 (low)

## Known Issues / Warnings

### All-Zero Neural Data
Some trials in sessions 40-42 (and a few others) have all-zero neural data across all neurons and time bins. This occurs because the obs_intervals for those trials overlap very little with the actual firing window, resulting in no spikes being captured. These trials are retained in the data but flagged as warnings during verification.

## Decoder Training Results

### Full Dataset (105 sessions, 48,356 trials)
| Output | Balanced Accuracy | Chance |
|--------|------------------|--------|
| Choice | 72.4% | 50.0% |
| Outcome | 67.9% | 33.3% |
| Early lick | 100% (trivial) | 50.0% |
| Tongue y-position | 71.5% | 33.3% |

### Sample Dataset (5 sessions, 250 trials)
| Output | Balanced Accuracy | Chance |
|--------|------------------|--------|
| Choice | 67.5% | 50.0% |
| Outcome | 64.5% | 33.3% |
| Early lick | 100% (trivial) | 50.0% |
| Tongue y-position | 56.9% | 33.3% |
