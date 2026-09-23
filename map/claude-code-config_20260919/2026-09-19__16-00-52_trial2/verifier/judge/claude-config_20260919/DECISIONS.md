# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is one NWB file per session under `/app/data/sub-<id>/`. The AI finds every session with a single sorted glob over that layout (174 files), and processes each file independently in a `ProcessPoolExecutor` (24 workers). Each file is opened **once** with raw `h5py` (not `pynwb`) and every needed array is pulled into plain numpy in a single pass (`read_session`): the trials table (`intervals/trials`), behavioural events (`acquisition/BehavioralEvents`), the side-camera tongue tracking (`acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`), the units table (`units/classification`, `anno_name`, `electrodes`, `obs_intervals`, `spike_times` + `spike_times_index`), the electrode table (`general/extracellular_ephys/electrodes`) and `general/subject/subject_id`. Only the spike-time ranges of QC-passing units are sliced out of the ragged `spike_times` buffer. Results are re-sorted by session id after the parallel map so output order is deterministic.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
print(f'found {len(files)} NWB files')
...
with ProcessPoolExecutor(max_workers=nworkers) as ex:
    for k, (res, info) in enumerate(ex.map(_worker, jobs)):
```

```python
def read_session(path):
    """Read everything we need from one NWB file into plain numpy arrays."""
    with h5py.File(path, 'r') as f:
        trials = f['intervals/trials']
        ev = f['acquisition/BehavioralEvents']
        d = {}
        d['session_id'] = session_id_from_path(path)
        d['subject'] = f['general/subject/subject_id'][()].decode()
        u = f['units']
        ...
        st_index = u['spike_times_index'][:]
        starts = np.concatenate([[0], st_index[:-1]])
        spike_times = u['spike_times'][:]
        unit_spikes = [spike_times[starts[i]:st_index[i]] for i in idx]
```

iii. From CONVERSION_NOTES Step 2/6: NWB is the published format, one file per session, so the directory listing is the complete session set. `h5py` is used instead of `pynwb` for speed ("read NWB 0.45 s / session"), and the whole archive is processed in 19 s of wall-clock with 24 workers. The AI verified its file inventory against `data/dandiset.yaml` and the papers (174 files, 28 subjects, 210,932 Kilosort clusters, 69,453 `classification == 'good'` units = 25.5 % of clusters vs the paper's 25.9 %).

## 1-b. How are the data split into subjects?

i. The animal is read from `general/subject/subject_id` in each file (a numeric string such as `'440956'`, matching the `sub-440956` folder name). At assembly, `subjects` is the sorted set of unique ids and `subject_idx` is each session's index into that list. 28 subjects result, matching the papers.

ii.
```python
d['subject'] = f['general/subject/subject_id'][()].decode()
```

```python
subjects = sorted({r['subject'] for r in results})
sub_index = {s: i for i, s in enumerate(subjects)}
...
'subjects': subjects,
'subject_idx': np.array([sub_index[r['subject']] for r in results], dtype=np.int64),
```

iii. CONVERSION_NOTES Step 2 records that `subject_id` is the canonical animal identifier in the NWB file and that the per-subject folder name is derived from it, so no extra grouping is needed. Sanity check #14 ("`subjects[subject_idx[s]]` == `general/subject/subject_id`") verifies the mapping directly against the raw files. The AI notes the count (28) matches "Mice (n = 28, Table S1)".

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no splitting is inferred. The session id is built from the filename as `<subject>_<ses-timestamp>` (e.g. `440956_20190209T150135`). **In addition, the AI rejects whole sessions by quality criteria** taken from the data paper's STAR Methods: control-trial (no-photostim, non-early-lick) performance must be > 65 %, and there must be ≥ 50 correct lick-left and ≥ 50 correct lick-right trials; a third, AI-invented criterion rejects a session whose DeepLabCut tongue likelihood is high on > 50 % of frames (tracking failure). Performance is computed **on the surviving trials only** (after the auto/free-water and video-coverage trial filters). 38 of 174 sessions are rejected → **136 sessions kept** (70,949 trials, 54,629 neurons). Five of those rejections are a cascade: sessions `440959_20190224`, `440959_20190225`, `440959_20190226`, `442571_20190227`, `442571_20190228`, `484676_20210413` lost essentially all their trials to the video-coverage filter and were then reported as failing "behavioural performance"/"too few correct" with perf = 0.0, L = 0, R = 0, despite ~78 % hit rates in the raw trials table.

ii.
```python
def session_id_from_path(path):
    base = os.path.basename(path)
    sub = base.split('_')[0].replace('sub-', '')
    ses = base.split('_')[1].replace('ses-', '')
    return f'{sub}_{ses}'
```

```python
MIN_PERFORMANCE = 0.65
MIN_CORRECT_PER_DIRECTION = 50
MAX_TONGUE_VISIBLE_FRACTION = 0.5
```

```python
def session_performance(d, sel=slice(None)):
    """Correct rate on control trials excluding early licks (datapaper definition)."""
    ps = d['photostim_power'][sel] != 'N/A'
    ctrl = (~ps) & (d['early_lick'][sel] == 'no early')
    hit = d['outcome'][sel] == 'hit'
    miss = d['outcome'][sel] == 'miss'
    m = ctrl & (hit | miss)
    perf = hit[m].sum() / m.sum() if m.sum() > 0 else 0.0
    n_left = int((hit & ctrl & (d['instruction'][sel] == 'left')).sum())
    n_right = int((hit & ctrl & (d['instruction'][sel] == 'right')).sum())
    return float(perf), n_left, n_right
```

```python
perf, n_left, n_right = session_performance(d, trial_idx)     # NB: post-filter subset
vis_frac = float((d['tongue_lik'] > TONGUE_LIKELIHOOD_THRESH).mean())
if perf <= MIN_PERFORMANCE:
    info['rejected'] = 'behavioural performance'
    return None, info
if n_left < MIN_CORRECT_PER_DIRECTION or n_right < MIN_CORRECT_PER_DIRECTION:
    info['rejected'] = 'too few correct lick-left / lick-right trials'
    return None, info
if vis_frac > MAX_TONGUE_VISIBLE_FRACTION:
    info['rejected'] = 'tongue tracking failure (implausible visible fraction)'
    return None, info
```

iii. The AI quotes the data paper verbatim (methods.txt): "We selected experimental sessions for analysis based on following criteria: overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each." Its stated evidence (Step 4/9) is that the correct rate over all 174 files is 80.6 %, below the paper's reported "84 %, range 65–99 %", while after applying the criteria it becomes 83.8 % (range 65.8–98.9 %) — which it treats as confirmation that the paper's criteria must be re-applied. It separately documents the 174 vs 173 session discrepancy and the 69,453 vs 69,943 unit discrepancy as "a discrepancy in the published archive, not in this conversion". The tongue-tracking rejection is justified by an explicit audit of `455220_20190803T150200`: likelihood > 0.5 on 96 % of frames with all y within 1 px, so the 40th/60th percentiles were 0.6 px apart.

## 1-d. Are the data correctly split into trials?

i. Trials come from the NWB trials table, with one go-cue event per row. The trial set is first restricted to trials the ephys actually observed, recovered by matching `units/obs_intervals` start times against `trials/start_time` (asserted with `np.allclose`, and asserted that all units share the same observed-interval count and that `is_good_trials` has the same width). Every per-trial field (`start_time`, `stop_time`, `outcome`, `trial_instruction`, `early_lick`, `auto_water`, `free_water`, `photostim_power`, `task`, `go_start_times`) is subset by the same index array `et`, keeping all streams aligned. `raw_trial_index` (row indices into the source trials table) is stored in metadata for provenance.

ii.
```python
oi_index = u['obs_intervals_index'][:]
n_int = np.diff(np.concatenate([[0], oi_index]))
assert n_int.min() == n_int.max(), 'units disagree on observed trials'
obs = u['obs_intervals'][0:oi_index[0]]
ephys_trials = np.searchsorted(trial_start_all, obs[:, 0])
assert np.allclose(trial_start_all[ephys_trials], obs[:, 0]), \
    'obs_intervals do not line up with the trials table'
assert u['is_good_trials'].shape[1] == len(ephys_trials)
d['ephys_trials'] = ephys_trials
...
et = ephys_trials
d['trial_start'] = trial_start_all[et]
d['outcome'] = _decode(trials['outcome'][:])[et]
...
d['go_time'] = ev['go_start_times']['timestamps'][:][et]
```

```python
go = d['go_time']
assert len(go) == len(d['trial_start'])
assert np.all(np.diff(go) > (T_STOP - T_START)), 'trial windows overlap'
```

iii. CONVERSION_NOTES Step 10 Check 5 records how this was found: "321/480 trials of one session came out with zero spikes", traced to the ephys covering only a prefix of the behavioural session in 9 of 174 files. The fix ("only the observed trials are converted") was verified afterwards — "no converted trial has zero total spikes". The non-overlap assertion is also what licenses the single-`bincount` binning trick.

## 1-e. How are trials filtered based on quality controls?

i. Three filters, in this order: (1) trials with no spike data (not covered by `units/obs_intervals`) are removed at load time (1-d); (2) `auto_water` and `free_water` trials are dropped, because reward is not contingent on the animal's choice so `outcome` is not a behavioural report; (3) trials whose [-2.5, +1.5] s window is not covered by at least 90 % of the expected 1,176 video frames (300 Hz) are dropped, so the tongue output is determinable. Photostimulation, early-lick and no-response (`ignore`) trials are deliberately **kept** because they are required decoder inputs/outputs. Accounting over the kept sessions: 74,689 behavioural trials → 73,953 with ephys coverage → −2,759 water → −254 video → **70,949 converted trials**.

ii.
```python
# (a) reward not contingent on the animal's choice -> `outcome` is not a
#     behavioural report on these trials (reference: get_regular_trial_mask)
keep = (~d['auto_water']) & (~d['free_water'])
info['n_dropped_water'] = int((~keep).sum())
# (b) the behavioural video must cover the whole extracted window, otherwise
#     the tongue output cannot be determined (reference: get_bad_trial_inds)
cov = video_coverage(d, win_start)
keep &= cov >= MIN_VIDEO_COVERAGE
info['n_dropped_video'] = int(((cov < MIN_VIDEO_COVERAGE)).sum())
trial_idx = np.where(keep)[0]
```

```python
def video_coverage(d, win_start):
    """Fraction of the [-2.5, +1.5] s window covered by video frames, per trial."""
    ts = d['tongue_t']
    lo = np.searchsorted(ts, win_start)
    hi = np.searchsorted(ts, win_start + N_BINS * BIN_WIDTH)
    expected = (N_BINS * BIN_WIDTH) / VIDEO_DT
    return (hi - lo) / expected
```

iii. CONVERSION_NOTES Step 4/5: the reference `get_regular_trial_mask` excludes "no early lick, no auto water, no free water, no no-response trials, no stimulation"; the AI keeps early-lick, no-response and photostim trials as a "**Required deviation**… photostimulation is a *specified decoder input*, and early-lick / ignore(no-lick) are *specified decoder outputs*", while retaining the water exclusions since those trials' `outcome` is not a behavioural report. The video filter is justified as the analogue of the reference's `get_bad_trial_inds`, which "drops trials whose frame count does not match the trial duration" ("Same intent"). The AI also documents the downstream consequence itself: "4 of those sessions consequently fall below the ≥50-correct-trials criterion and are rejected outright."

## 2-a. What variables in the raw data is the final `neural` data derived from?

i. `units/spike_times` together with the ragged index `units/spike_times_index`, restricted to the units that pass QC (2-c), plus `acquisition/BehavioralEvents/go_start_times/timestamps` which defines each trial's window. `units/anno_name` and the electrode `x` coordinate are used only to label each neuron's region/hemisphere.

ii.
```python
st_index = u['spike_times_index'][:]
starts = np.concatenate([[0], st_index[:-1]])
spike_times = u['spike_times'][:]
unit_spikes = [spike_times[starts[i]:st_index[i]] for i in idx]
d['unit_spikes'] = unit_spikes
```

```python
d['go_time'] = ev['go_start_times']['timestamps'][:][et]
```

iii. Step 4's variable mapping table equates `units/spike_times` (+`spike_times_index`) with the reference `.mat` field `neuron_single_units`, and `go_start_times` with `task_cue_time[0]`; spike times are the only neural representation in the file.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin firing rates in Hz: spike counts in 80 non-overlapping 50 ms bins divided by the bin width. The implementation is fully vectorised over units, trials and bins — because the 4 s windows are guaranteed non-overlapping, each spike belongs to at most one (unit, trial, bin) cell, so one `searchsorted` for the trial index, integer division for the bin index and a single `np.bincount` produce the whole (n_units, n_trials, 80) tensor. No smoothing, normalisation or baseline subtraction. Stored as `float32`.

ii.
```python
ti = np.searchsorted(win_start, t, side='right') - 1
valid = (ti >= 0) & (ti < n_trials)
ti_c = np.where(valid, ti, 0)
valid &= t < win_end[ti_c]
t, uidx, ti = t[valid], uidx[valid], ti[valid]
b = ((t - win_start[ti]) / BIN_WIDTH).astype(np.int64)
np.clip(b, 0, N_BINS - 1, out=b)

flat = (uidx * n_trials + ti) * N_BINS + b
counts = np.bincount(flat, minlength=n_units * n_trials * N_BINS)
counts = counts.reshape(n_units, n_trials, N_BINS)
return (counts / BIN_WIDTH).astype(np.float32)
```

iii. "Neural: firing rate in Hz (spike count / bin width), as in the reference `sliding_histogram(..., rate=True)`" (module docstring; Step 5 mapping table). The single-`bincount` formulation is justified in Step 6 as an ≈100× speed-up over per-(unit, trial) `searchsorted`, licensed by the asserted ≥4.58 s spacing of go cues. Sanity checks #3 and #4 re-derive individual (trial, neuron, bin) rates and a whole trial's total spike count from the raw ragged `spike_times` and match exactly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three conditions on units: `units/classification == 'good'` (the region-specific QC classifier), an `anno_name` CCF annotation that maps to one of the 14 major regions, and non-zero spiking somewhere in the extracted windows of the session. In practice only the first bites: over the 136 kept sessions the number of `classification == 'good'` units (54,629) equals the number of converted neurons exactly, so the annotation and activity filters removed zero units. Across all 174 files `classification == 'good'` gives 69,453 units (25.5 % of clusters); within the kept sessions the good fraction is 25.90 %, matching the paper's 25.9 %. No individual quality-metric thresholds and no 2 Hz firing-rate cut are applied. Session-level rejection (1-c) is what removes the remaining ~14.8 k good units.

ii.
```python
classification = _decode(u['classification'][:])
anno = _decode(u['anno_name'][:])
good = classification == 'good'
...
regions = np.array([annotation_to_region(a) or '' for a in anno])
keep_unit = good & (regions != '')
idx = np.where(keep_unit)[0]
```

```python
# drop units that are completely silent in every extracted window
active = fr.any(axis=(1, 2))
fr = fr[active]
unit_region = d['unit_region'][active]
```

iii. Step 5 Key Decision 2: "`classification == 'good'` (the QC-classifier 'good units' used throughout both papers), plus a mappable CCF annotation. Neurons with zero spikes in the extracted windows across the whole session are dropped (they carry no information and match `check_fr`'s intent)." Step 4 explains why the methods paper's 2 Hz cut is *not* applied: "it is a stability criterion for *per-neuron regression*, not a data-quality criterion, and the reference population-decoding code (`population_decoding_utils.load_session`) does not apply it. Discarding ~half the units would throw away real population information." The AI also verified that all 293 distinct annotations of good units map to one of its 14 regions (0 unmapped) and that four of the five per-region unit counts reproduce the paper to ≤0.25 % (ALM differs because the reference used an undistributed CCF voxel mask).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to go-cue onset (`go_start_times`). All NWB streams share one session-absolute clock, so no resampling or offset correction is needed: each trial's window start is `go - 2.5 s`, and spikes are assigned to (trial, bin) by their offset from that window start. The same `win_start` vector is reused for the tongue output, and the same go cue defines the bin centres used for both inputs.

ii.
```python
go = d['go_time']
assert np.all(np.diff(go) > (T_STOP - T_START)), 'trial windows overlap'
win_start = go + T_START
...
fr = bin_spikes(d['unit_spikes'], win_start)          # (n_units, n_trials, 80)
```

```python
b = ((t - win_start[ti]) / BIN_WIDTH).astype(np.int64)
```

iii. Decoder spec ("Temporally align based on Go cue onset"), and Step 10 Check 3(c): "reference: go cue = 0 for spikes, licks and photostimulation (`process_one_sess` subtracts `task_cue_time[0]`) … identical: `go_start_times` is the origin for spikes, tone onset, photostimulation and video — Yes". The `--show-processing` PSTH panel is offered as visual proof: flat baseline, a bump beginning at the median tone onset (−1.85 s) and a large transient starting exactly at t = 0.

## 2-e. How is the `neural` data temporally binned/resampled?

i. 80 non-overlapping 50 ms bins spanning [−2.5, +1.5) s relative to the go cue, defined once at module level as 81 edges / 80 centres and reused for every trial and session, so T = 80 everywhere. No further rebinning, no sliding window, no smoothing. `metadata['time_bin_size'] = 50.0` ms.

ii.
```python
BIN_WIDTH = 0.05          # s, decoder spec: 50 ms bins
T_START = -2.5            # s relative to go cue, decoder spec
T_STOP = 1.5              # s relative to go cue, decoder spec
N_BINS = int(round((T_STOP - T_START) / BIN_WIDTH))   # 80

BIN_EDGES = T_START + BIN_WIDTH * np.arange(N_BINS + 1)   # (81,)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2              # (80,)
```

iii. The window and bin width are set by the instructions. Step 4 records this as a deliberate, required deviation from the reference's 40 ms width / 3.4 ms stride sliding histogram and [−3, +3] s window: "the task specifies 50 ms bins. We use non-overlapping 50 ms bins (stride = width), which is the natural reading of '50-ms-width bins' and avoids leaking information across bins."

## 3-a. What variables in the raw data is `input` *time from tone onset in seconds* derived from?

i. `acquisition/BehavioralEvents/sample_start_times/timestamps` (sample-epoch = instruction-tone onsets, read for the **whole** session, not the trial-subset) together with each trial's go cue. The tone of a trial is the **last** sample onset at or before its go cue.

ii.
```python
d['sample_start'] = ev['sample_start_times']['timestamps'][:]
```

```python
def tone_onset_rel(d):
    """Time of the last sample-epoch (tone) onset at or before each go cue,
    expressed relative to the go cue (so always <= 0)."""
    si = np.searchsorted(d['sample_start'], d['go_time'], side='right') - 1
    assert (si >= 0).all(), 'a trial has no preceding sample epoch'
    return d['sample_start'][si] - d['go_time']
```

iii. Step 5 Key Decision 6: "Tone onset = the last sample-epoch onset at or before the go cue. Early-lick trials replay the sample epoch, so this is the last time the instruction tones were actually played before the go cue." Step 10 Check 5 adds the verification that this epoch always has the full 0.65 s duration in all 94,990 trials, i.e. it is never a truncated/aborted presentation.

## 3-b. What processing is involved in computing `input` *time from tone onset in seconds*?

i. A continuous, time-varying value per bin: the bin centre (expressed relative to the go cue) minus the tone-onset time (also relative to the go cue), i.e. seconds elapsed since tone onset, negative before it. It is stored as `float32` row 0 of the (2, 80) input array. For an ordinary trial (0.65 s sample + 1.2 s delay) the go-cue bin therefore reads ≈1.875 s, which the AI verified for 87.4 % of trials; trials with replayed sample epochs or non-standard delays get larger values (observed range [−1.525, 11.894] s).

ii.
```python
tone_rel = tone_onset_rel(d)[trial_idx]               # <= 0
time_from_tone = BIN_CENTERS[None, :] - tone_rel[:, None]     # (n_tr, 80)
```

```python
inputs = [np.stack([time_from_tone[i], stim[i]]).astype(np.float32)
          for i in range(n_tr)]
```

iii. Step 9's consistency table checks the derived values against the paper's task structure ("Sample epoch duration 0.65 s … tone onset at −1.85 s for 87.5 % of trials"; "Trials with a 0.3 s or 1.8 s delay instead of 1.2 s … not special-cased: `time_from_tone_onset_s` is computed per trial from the actual events, so it reflects the true delay"). Sanity check #5 recomputes the value from the raw NWB for random trials (max abs error 9.5e-8).

## 3-c. How is `input` *time from tone onset in seconds* aligned with the neural data?

i. By construction: the input uses the same go-cue-relative grid (`BIN_CENTERS`) that defines the neural bin edges (`win_start = go + T_START`), so input bin *k* covers exactly the same interval as firing-rate bin *k*. Only a per-trial scalar offset (the tone-to-go-cue gap) is added.

ii.
```python
BIN_EDGES = T_START + BIN_WIDTH * np.arange(N_BINS + 1)   # (81,)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2              # (80,)
```
```python
win_start = go + T_START
fr = bin_spikes(d['unit_spikes'], win_start)
...
time_from_tone = BIN_CENTERS[None, :] - tone_rel[:, None]
```

iii. Step 10 Check 3(c): all streams share the go cue as origin. The `--show-processing` panel "input 0" is described as "a straight ramp of slope 1 crossing zero at the tone onset", used as visual confirmation of alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The **event timestamps** `acquisition/BehavioralEvents/photostim_start_times` and `photostim_stop_times` (guarded for sessions that lack them), mapped to trials by `searchsorted` over the trial start times and then expressed relative to each trial's go cue. `trials/photostim_power != 'N/A'` is used separately, only to define control trials for the session-performance computation.

ii.
```python
d['stim_on'] = ev['photostim_start_times']['timestamps'][:] \
    if 'photostim_start_times' in ev else np.zeros(0)
d['stim_off'] = ev['photostim_stop_times']['timestamps'][:] \
    if 'photostim_stop_times' in ev else np.zeros(0)
```

```python
def photostim_windows(d):
    n = len(d['go_time'])
    on = np.full(n, np.nan); off = np.full(n, np.nan)
    if len(d['stim_on']) == 0:
        return on, off
    ti = np.searchsorted(d['trial_start'], d['stim_on'], side='right') - 1
    ok = (ti >= 0) & (ti < n)
    ti, s_on, s_off = ti[ok], d['stim_on'][ok], d['stim_off'][ok]
    # keep the first stimulation event of each trial (there is exactly one)
    on[ti] = s_on - d['go_time'][ti]
    off[ti] = s_off - d['go_time'][ti]
    return on, off
```

iii. Step 4's mapping table equates the reference `.mat` `task_stimulation` (power, type, on, off) with `photostim_power`, `photostim_onset`, `photostim_duration` and `photostim_start/stop_times`, and records the check "onset field == event time − trial start ✓" — i.e. the AI verified that the event stream and the trials-table onset/duration columns describe the same windows, then used the event stream ("we derive the stimulation window per-trial from the recorded event times rather than assuming a fixed epoch", because only a subset of mice used late-delay stimulation). Step 9 verifies 0.5 s duration for all 18,588 events.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series per trial: bin = 1 when the laser is on **at the bin centre**, i.e. `on <= centre <= off`, else 0; trials without stimulation keep NaN bounds, which are replaced by ±inf so all bins are 0. Stored as `float32` row 1 of the input array. This yields exactly 10 on-bins (0.5 s) for 14,368 of 14,382 stimulation trials, and photostim is on in 2.53 % of all time bins (20.3 % of trials have stimulation).

ii.
```python
# A bin is "on" when the laser is on at the bin centre, i.e. the binary
# laser signal is sampled at the same instants as the bin centres.  Using
# interval overlap instead would light up the bin containing the go cue for
# a third of the stimulation trials, because the recorded laser-off
# timestamp overshoots the go cue by ~1 ms; the photoinhibition always ended
# before the go cue (datapaper STAR Methods).  With the centre rule each
# 0.5 s stimulation covers exactly 10 bins.
stim = np.zeros((len(trial_idx), N_BINS), dtype=np.float32)
if has_stim.any():
    o = np.where(has_stim, on, np.inf)[:, None]
    f_ = np.where(has_stim, off, -np.inf)[:, None]
    stim = ((BIN_CENTERS[None, :] >= o) & (BIN_CENTERS[None, :] <= f_)
            ).astype(np.float32)
```

iii. Step 10 Check 5 documents the bug this rule fixed: "**Photostim off-timestamps overshoot the go cue by ~1 ms** in 35 % of stimulation events, which lit up the go-cue bin under an interval-overlap rule. → The binary laser signal is sampled at the bin centres instead. Each 0.5 s stimulation now covers exactly 10 bins and never reaches the go-cue bin, as the papers state." Sanity check #6 recomputes the full photostim trace for all 1,535 trials of a session directly from the raw event times with 0 mismatches.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The stimulation on/off times are converted to go-cue-relative seconds (`s_on - go[ti]`) and compared against the same `BIN_CENTERS` grid used for the firing rates, so bin *k* of the photostim input is the same interval as bin *k* of the neural data.

ii.
```python
on[ti] = s_on - d['go_time'][ti]
off[ti] = s_off - d['go_time'][ti]
```
```python
stim = ((BIN_CENTERS[None, :] >= o) & (BIN_CENTERS[None, :] <= f_)).astype(np.float32)
```

iii. Step 9: "Photostim always ends before go cue … latest 'on' bin centre = −0.025 s ✅", and the `--show-processing` panels ("photostimulation confined to the delay epoch, always ending at or before the go cue"; "raw photostim event histogram: on at −1.2 s or −0.5 s, off 0.5 s later") are presented as alignment evidence.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. There is no lick-direction column, so choice is derived from `trials/outcome` (`hit`/`miss`/`ignore`) crossed with `trials/trial_instruction` (`left`/`right`): a hit means the animal licked the instructed side, a miss the opposite side, an `ignore` means no lick.

ii.
```python
d['outcome'] = _decode(trials['outcome'][:])[et]
d['instruction'] = _decode(trials['trial_instruction'][:])[et]
```

```python
def choice_from_trials(d):
    """0 = left, 1 = right, 2 = no lick."""
    ch = np.full(len(d['outcome']), 2, dtype=np.int64)
    hit = d['outcome'] == 'hit'
    miss = d['outcome'] == 'miss'
    left = d['instruction'] == 'left'
    ch[hit & left] = 0
    ch[hit & ~left] = 1
    ch[miss & left] = 1      # error on a lick-left instruction -> licked right
    ch[miss & ~left] = 0
    return ch
```

iii. Step 5 mapping table: "hit ⇒ instruction; miss ⇒ opposite of instruction; ignore ⇒ 'no lick'… validated against first post-go lick". Sanity check #8 cross-validates the derived choice against the direction of the first lick inside the 1.5 s answer period using `left_lick_times`/`right_lick_times`: 1,318/1,319 = 99.92 % agreement, the single disagreement being a left lick 2.7 ms after the go cue ("a lick already in flight") before the scored right lick, where "the rig's `outcome` field is authoritative".

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded 0 = left, 1 = right, 2 = no lick, named in `output_values[0] = ['left', 'right', 'no lick']`, and broadcast across all 80 bins so that all four outputs share one (4, 80) `int64` array per trial. Resulting distribution: left 0.448, right 0.442, no lick 0.110.

ii.
```python
OUTPUT_VALUES = [
    ['left', 'right', 'no lick'],
    ...
]
```
```python
choice = choice_from_trials(d)[trial_idx]
...
outputs = [np.stack([np.full(N_BINS, choice[i]), np.full(N_BINS, outcome[i]),
                     np.full(N_BINS, early[i]), tongue[i]]).astype(np.int64)
           for i in range(n_tr)]
```

iii. Step 5 Key Decision 9: "Per-trial outputs are broadcast over time so that all four outputs share one (4, 80) array, as required by the format (tongue position is time-varying)." Value order follows the Decoder Task listing (left, right, no lick). Sanity checks #7 and #11 verify the recomputed choice for all trials and that outputs 0–2 are constant across the 80 bins.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from `intervals/trials/outcome`, which already stores exactly the three categories asked for (`ignore`, `miss`, `hit`).

ii.
```python
d['outcome'] = _decode(trials['outcome'][:])[et]
```

iii. Step 4's mapping table equates it with the reference `.mat` `behavior_report` (1 correct / 0 error / −1 no response). No derivation needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped through a fixed dictionary to 0 = ignore, 1 = miss, 2 = hit (matching `output_values[1]`), and broadcast across the 80 bins as row 1 of the output array. Distribution: ignore 0.110, miss 0.153, hit 0.737.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = np.array([outcome_map[o] for o in d['outcome']])[trial_idx]
```
```python
OUTPUT_VALUES[1] = ['ignore', 'miss', 'hit']
```

iii. The code order follows the Decoder Task listing ("Outcome (ignore, miss, hit, per-trial)"). Because the map is a strict dictionary lookup, an unexpected string would raise rather than be silently mis-coded. Sanity check #9 recomputes `outcome` for all trials of the checked sessions; Step 9 compares the converted hit/miss/ignore fractions with those computed directly from the raw files.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From `intervals/trials/early_lick`, which holds `'early'` / `'no early'`.

ii.
```python
d['early_lick'] = _decode(trials['early_lick'][:])[et]
```

iii. Step 4 maps it to the reference `.mat` `behavior_early_report`; the flag is recorded explicitly by the rig, so no derivation is needed. Step 12 verifies the event is inside the extracted window: "99.1 % of early-lick trials contain at least one recorded lick inside the extracted window, and their pre-go-cue tongue-visible fraction is 26.3 % versus 4.7 % on non-early trials."

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `'early'` → 1, anything else → 0 (`output_values[2] = ['no', 'yes']`), broadcast across the 80 bins as row 2. Distribution: no 0.884, yes 0.116, matching the 0.114 early fraction computed directly from the raw files.

ii.
```python
early = (d['early_lick'] == 'early').astype(np.int64)[trial_idx]
```
```python
OUTPUT_VALUES[2] = ['no', 'yes']
```

iii. Value order follows the Decoder Task listing ("Early lick (no, yes, per-trial)"). Sanity check #10 recomputes `early_lick` for all trials of the checked sessions and passes.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: `data[:, 1]` is the DeepLabCut tongue y-coordinate, `data[:, 2]` its likelihood, and `timestamps` the ~294 Hz frame times on the session clock. Column 0 (tongue x) is not used.

ii.
```python
bts = f['acquisition/BehavioralTimeSeries']
tk = bts['Camera0_side_TongueTracking']
tongue = tk['data'][:]                    # (n_frames, 3): x, y, likelihood
d['tongue_t'] = tk['timestamps'][:]
d['tongue_y'] = tongue[:, 1]
d['tongue_lik'] = tongue[:, 2]
```

iii. Step 4's mapping table equates these with the reference `tracking.camera_0_side.tongue_x/y/likelihood`; the reference video analysis also uses the side view only ("DeepLabCut markers for tongue, jaw and nose from the *side* view only", Step 3).

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. A frame counts as "tongue visible" when its likelihood exceeds 0.5. Visible frames are assigned to (trial, bin) cells on the same go-cue-anchored grid as the spikes, with one `bincount` for the count and one weighted `bincount` for the sum, giving the mean visible y per bin plus a boolean `has` mask. Bins with no visible frame get the dedicated class 3 ("not visible") — 73.4 % of all bins. Occluded frames are excluded from the average rather than imputed.

ii.
```python
visible = d['tongue_lik'] > TONGUE_LIKELIHOOD_THRESH
vt = d['tongue_t'][visible]; vy = d['tongue_y'][visible]
ti = np.searchsorted(win_start, vt, side='right') - 1
...
b = ((vt - win_start[ti]) / BIN_WIDTH).astype(np.int64)
np.clip(b, 0, N_BINS - 1, out=b)
flat = ti * N_BINS + b
nvis = np.bincount(flat, minlength=n_trials * N_BINS).reshape(n_trials, N_BINS)
ysum = np.bincount(flat, weights=vy, minlength=n_trials * N_BINS).reshape(n_trials, N_BINS)
has = nvis > 0
ybar[has] = ysum[has] / nvis[has]
```
```python
out = np.full(ybar.shape, 3, dtype=np.int64)   # 3 = not visible
```

iii. Step 5 Key Decision 7: the likelihood distribution is strongly bimodal ("median ≈ 6e-5 for occluded frames vs ≈ 1.0 when visible"), so any threshold in [0.1, 0.99] gives essentially the same labels (visible fraction 0.1045 at 0.99 vs 0.1059 at 0.5). Step 4 records the deviation from the reference's imputation rule as required by the spec: the reference "set the tongue position to its mean value" when occluded, but "the task defines a separate 'not visible' class (3), so occlusion is encoded rather than imputed."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per session, the 40th and 60th percentiles are taken over the **per-bin mean tongue y of the session's converted trials, restricted to bins where the tongue was visible**; bins are then labelled 0 (< 40th pct), 1 (40th–60th pct), 2 (> 60th pct), 3 (no visible frame). If fewer than two visible bins exist, everything stays class 3. Over the whole dataset this gives 0.106 / 0.053 / 0.106 / 0.734, i.e. exactly 40/20/40 among visible bins. Note that two documentation locations (CONVERSION_NOTES Step 5 Key Decision 8 and `metadata['output_descriptions'][3]` in the shipped pickle) still describe the *earlier* rule — percentiles "of all visible frames of the session" — which is not what the final code does; the Step 10 iteration log records the change.

ii.
```python
TONGUE_LIKELIHOOD_THRESH = 0.5
TONGUE_LOW_PCT = 40.0
TONGUE_HIGH_PCT = 60.0
```
```python
def tongue_classes(d, win_start):
    ybar, has = tongue_bin_position(d, win_start)
    out = np.full(ybar.shape, 3, dtype=np.int64)
    if has.sum() < 2:
        return out, np.nan, np.nan
    p_lo, p_hi = np.percentile(ybar[has], [TONGUE_LOW_PCT, TONGUE_HIGH_PCT])
    cls = np.where(ybar < p_lo, 0, np.where(ybar > p_hi, 2, 1))
    out[has] = cls[has]
    return out, float(p_lo), float(p_hi)
```
```python
tongue, p_lo, p_hi = tongue_classes(d, win_start[trial_idx])
```

iii. Step 10 "Iterations performed" #2: "**Tongue percentiles computed over whole-session raw frames** gave an unbalanced 44/31/24 split of the visible bins → switched to percentiles of the per-bin tongue position over the session's converted trials → exactly 40/20/40." The per-session scope and the 40/60 split follow the Decoder Task spec. Sanity check #12 recomputes the class of **every** (trial, bin) from the raw 300 Hz stream including the session percentiles, with 0 mismatches.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps are on the same session clock as spikes and events, so frames are binned with exactly the same construction as the spikes: trial index from `searchsorted` over `win_start` (= go − 2.5 s), bin index from the offset divided by 50 ms, clipped to [0, 79], with the `vt < win_end` mask applied. Bin *k* of the tongue output therefore covers the same interval as bin *k* of the firing rates. No interpolation or offset correction.

ii.
```python
win_end = win_start + N_BINS * BIN_WIDTH
ti = np.searchsorted(win_start, vt, side='right') - 1
ok = (ti >= 0) & (ti < n_trials)
ti_c = np.where(ok, ti, 0)
ok &= vt < win_end[ti_c]
vt, vy, ti = vt[ok], vy[ok], ti[ok]
b = ((vt - win_start[ti]) / BIN_WIDTH).astype(np.int64)
```

iii. Step 7's plot review uses the cross-stream timing as the alignment check: "tongue class raster / class fractions over time: ≈0 visible before the go cue, rising abruptly at t = 0 — the licking response. Correct temporal alignment of the video stream against the neural stream." Trials whose video does not cover the whole window are removed beforehand (1-e), so the leading bins are not silently filled with "not visible".

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Documented cases and their handling:
- **Ephys covering only part of the behavioural session** (9/174 files): trials outside `units/obs_intervals` are dropped (1-d), verified afterwards that no converted trial has zero total spikes.
- **Free/auto-water trials** (reward not contingent on choice): dropped.
- **Spikes only inside `[trial.start_time, trial.stop_time]`**: on error trials the interval ends 0.2–0.5 s after the go cue, so trailing bins are genuinely 0 Hz. No imputation; recorded in `metadata['known_limitations']` and visualised.
- **Missing/short video** (7 sessions): trials with < 90 % window coverage dropped; the sessions that thereby lose nearly all trials fall out via the session criteria.
- **Degenerate DeepLabCut tracking** (`455220_20190803T150200`, likelihood > 0.5 on 96 % of frames): session rejected.
- **Occluded tongue frames**: excluded from the bin mean; a bin with no visible frame becomes the explicit class 3.
- **Sessions with no QC labels** (`classification` = NaN): `_decode` turns them into the string `'nan'`, so no unit is `'good'`; there is no explicit "drop the session" guard — the one such session (`440958_20190216T162508`, 1,852 clusters, 0 good) happened to be rejected first by the behavioural criteria, so a 0-neuron session was never emitted.
- **Sessions/trials with no photostim events, missing lick streams**: guarded with `if ... in ev else np.zeros(0)`.
- **Missing CCF coordinates**: a fall-back to the probe's targeted ML sign is implemented (never needed among good units); units with an unmappable annotation would be dropped (0 occur).
- **Float edge cases**: `np.clip(b, 0, 79)` after an explicit `t < win_end` mask.

ii.
```python
d['stim_on'] = ev['photostim_start_times']['timestamps'][:] \
    if 'photostim_start_times' in ev else np.zeros(0)
```
```python
def _decode(arr):
    """bytes ndarray -> list of str."""
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])
```
```python
if np.isfinite(x):
    hemi[j] = 'left' if x >= ML_MIDLINE else 'right'
else:
    ml = probe_ml.get(gnames[i], np.nan)
    hemi[j] = 'left' if (np.isfinite(ml) and ml < 0) else 'right'
```
```python
if has.sum() < 2:
    return out, np.nan, np.nan
```
```python
except Exception as e:  # keep one bad file from killing the whole run
    import traceback
    traceback.print_exc()
    return None, dict(session_id=session_id_from_path(path), path=path,
                      rejected=f'error: {e}')
```

iii. The governing principle in CONVERSION_NOTES is: where nothing was recorded, exclude (trials without spikes, sessions without usable video/tracking); where the measurement legitimately has no value (retracted tongue), encode it as an explicit category rather than impute; where the published archive is simply truncated (spikes only inside the trial interval), document the limitation because "no imputation is possible and masking is not representable in the format" (Step 5 Key Decision 10). Every case above is listed in the Step 10 Check 5 edge-case table with the evidence that led to it.

## 10-a. What are the most time-consuming steps of the code?

i. The script instruments itself (`t_read`, `t_neural`, `t_io`, `t_pack` per session, printed per file). Reading the NWB file (~0.45 s/session, dominated by pulling the `spike_times` buffer and the ~1 M × 3 tongue array) and binning the spikes (~0.50 s/session) dominate; input/output construction and packing are ~0.02 s each. With 24 worker processes the whole 174-file archive takes 19 s wall-clock (0.11 s/file), plus ~17 s to pickle the 9.45 GB result — 36 s total, far inside the 15-minute budget.

ii.
```python
info.update(n_trials=n_tr, n_neurons=int(fr.shape[0]),
            ...
            t_read=t_read, t_neural=t_neural, t_io=t_io, t_pack=t_pack,
            t_total=time.time() - t0)
```
```python
print(f'[{k+1}/{len(files)}] {i["session_id"]}: {i["n_neurons"]} neurons, '
      f'{i["n_trials"]} trials, perf={i["performance"]:.3f}, '
      f'{i["t_total"]:.1f}s (read {i["t_read"]:.1f} / bin {i["t_neural"]:.1f} '
      f'/ io {i["t_io"]:.2f} / pack {i["t_pack"]:.2f})', flush=True)
```

iii. Step 7's timing table ("read NWB 0.45 s → 78 s CPU; bin spikes 0.50 s → 87 s CPU; inputs+outputs 0.02 s; pack 0.02 s; total 0.11 s wall → 19 s") is used to conclude "the full run (36 s) confirmed the estimate. No further optimisation needed."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The two expensive loops are already gone: neither spike binning nor video binning loops over units, trials or bins — each is a single `bincount` over a flattened (unit, trial, bin) index, which the AI measured as ≈100× and ≈50× faster than the naive per-(unit, trial) `searchsorted` implementation. The loops that remain are all cheap and per-unit or per-trial Python-level list building: slicing the ragged spike buffer per unit (`[spike_times[starts[i]:st_index[i]] for i in idx]`), the per-unit hemisphere assignment loop (which also does a dict lookup per unit), the element-wise `_decode` of text columns, the `outcome_map` list comprehension, and the three per-trial packing comprehensions. The hemisphere loop is the only one that is straightforwardly vectorisable (`np.where(ccf_x >= ML_MIDLINE, 'left', 'right')` with a masked fall-back); the others build the ragged per-trial lists the target format requires.

ii.
```python
hemi = np.empty(len(idx), dtype=object)
for j, i in enumerate(idx):
    x = ccf_x[i]
    if np.isfinite(x):
        hemi[j] = 'left' if x >= ML_MIDLINE else 'right'
    else:
        ml = probe_ml.get(gnames[i], np.nan)
        hemi[j] = 'left' if (np.isfinite(ml) and ml < 0) else 'right'
```
```python
neural = [np.ascontiguousarray(fr[:, i, :]) for i in range(n_tr)]
inputs = [np.stack([time_from_tone[i], stim[i]]).astype(np.float32)
          for i in range(n_tr)]
outputs = [np.stack([...]) for i in range(n_tr)]
```

iii. Step 6 "Code inefficiencies identified / Code speedups added": "A naive implementation bins spikes per (unit, trial) with `np.searchsorted` over 81 bin edges — ~87 M searchsorted probes per session… Single-pass histogram for the whole session. Because the go cues are ≥4.58 s apart (asserted) the 4 s trial windows never overlap, so every spike belongs to at most one (trial, bin)… O(n_spikes) instead of O(n_units · n_trials · n_bins · log n_spikes). The same trick bins the video frames." The remaining loops are not discussed, consistent with them being negligible next to I/O.

## 10-c. What processing does the code repeat multiple times?

i. Very little. Each NWB file is opened exactly once and every quantity derived in one pass; the bin grid is built once at module level and reused for every trial and session; the per-session tongue percentiles are computed inside the same pass. The small repeats are: `d['tongue_lik'] > TONGUE_LIKELIHOOD_THRESH` is computed twice (once for the session-level `tongue_visible_fraction`, once inside `tongue_bin_position`); `_decode` is applied to whole text columns which are then immediately subset to the ephys trials; and the firing rates are computed for **all** ephys trials and only afterwards subset to the kept trials, so ~4 % of the binning work is thrown away.

ii.
```python
vis_frac = float((d['tongue_lik'] > TONGUE_LIKELIHOOD_THRESH).mean())
```
```python
visible = d['tongue_lik'] > TONGUE_LIKELIHOOD_THRESH
```
```python
fr = bin_spikes(d['unit_spikes'], win_start)          # all ephys trials
fr = fr[:, trial_idx, :]                              # then subset
```

iii. Not discussed explicitly in CONVERSION_NOTES beyond the general design ("`read_session` — one pass over the NWB file"); the repeats are O(n_frames) boolean work and a few percent of the binning, negligible against the 0.45 s file read.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A handful of loads and computations never reach the output: `left_lick_times` / `right_lick_times` and `trials/task` are read on every session and never used anywhere in the converter (the lick streams are used only by the separate `cache/sanity_checks.py`); `trial_stop` and `unit_anno` are used only when `--show-processing` debug info is requested; `probe_ml` parses the JSON `location` attribute of every probe group in every file for a hemisphere fall-back that never fires (the AI verified no good unit has a NaN CCF coordinate); firing rates are binned for trials that are subsequently dropped; and a `<outfile>_session_info.json` sidecar is written in addition to `metadata['session_info']`. All of this is small relative to the file read.

ii.
```python
d['lick_left'] = ev['left_lick_times']['timestamps'][:] \
    if 'left_lick_times' in ev else np.zeros(0)
d['lick_right'] = ev['right_lick_times']['timestamps'][:] \
    if 'right_lick_times' in ev else np.zeros(0)
d['task'] = _decode(trials['task'][:])[et]
```
```python
probe_ml = {}
for gname, grp in f['general/extracellular_ephys'].items():
    if gname == 'electrodes':
        continue
    try:
        probe_ml[gname] = float(json.loads(grp.attrs['location'])['ml_location'])
    except Exception:
        probe_ml[gname] = np.nan
```

iii. CONVERSION_NOTES does not flag these; the hemisphere fall-back is described as "rarely needed" in a code comment and Step 10 Check 5 confirms it is in fact never needed among good units ("Neurons with a NaN CCF coordinate. None occur among good units (checked over the whole archive); a fall-back to the probe's targeted ML sign is implemented anyway"). The unused lick times appear to be leftovers from the choice-validation sanity check.
