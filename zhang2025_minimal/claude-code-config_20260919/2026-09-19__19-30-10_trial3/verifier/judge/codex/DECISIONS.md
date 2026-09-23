# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the session roster from the frozen release table `code_zhang2025/data/bwm_release.csv`, optionally restricts that list with `DATALIMIT_SUBSET.csv`, then uses a cached `ONE` client plus `SessionLoader` and `SpikeSortingLoader` to load trials, wheel, motion-energy, and spike-sorting data for each selected `eid` and probe. It does not use `one.search(...)` to discover sessions.

ii.
```python
ONE_CACHE_DIR = '/app/data/one_cache'
ONE_BASE_URL = 'https://openalyx.internationalbrainlab.org'
BWM_RELEASE_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'
DATALIMIT_SUBSET_CSV = '/app/DATALIMIT_SUBSET.csv'
```

```python
def get_one():
    global _ONE
    if _ONE is None:
        from one.api import ONE
        _ONE = ONE(base_url=ONE_BASE_URL, silent=True, cache_dir=ONE_CACHE_DIR)
    return _ONE
```

```python
bwm_df = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
eids = list(dict.fromkeys(bwm_df['eid'].tolist()))
if os.path.exists(DATALIMIT_SUBSET_CSV):
    subset = pd.read_csv(DATALIMIT_SUBSET_CSV)
    col = 'eid' if 'eid' in subset.columns else subset.columns[0]
    allowed = set(subset[col].astype(str))
    eids = [e for e in eids if e in allowed]
```

```python
trials, trials_mask = load_trials_and_mask(one, eid)
traces = load_behaviour_traces(one, eid)
spike_times, spike_units, acronyms = load_session_units(
    one, eid, probe_rows, brain_regions)
```

iii. In the trajectory the agent inspected `0_data_caching.py` and `ibl_data_utils.py`, inspected `bwm_release.csv`, verified that a cached-token `ONE` client could still resolve `eid2pid`, and then wrote the top-of-file design summary stating that sessions come from the BWM release CSV and contents are loaded through `ONE` loaders.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are taken directly from the `subject` column of `bwm_release.csv`. Each session carries one subject label, and after conversion the agent builds sorted unique `subjects` plus `subject_idx` for each retained session.

ii.
```python
for eid in eids:
    rows = by_eid.get_group(eid)
    tasks.append((eid, rows[['pid', 'probe_name']].to_dict('records'),
                  str(rows['subject'].iloc[0]), str(rows['lab'].iloc[0])))
```

```python
subjects = sorted({r['subject'] for r in results})
subject_index = {s: i for i, s in enumerate(subjects)}
...
'subject_idx': np.array([subject_index[r['subject']] for r in results],
                        dtype=np.int64),
```

iii. In the trajectory the agent inspected the release CSV schema and counts, including the `subject` column, and then propagated `subject` through the conversion task tuples and final assembly.

## 1-c. How are the data split into sessions?

i. Sessions are identified by unique `eid` values from the release CSV. Each `eid` is processed independently and becomes one output session.

ii.
```python
eids = list(dict.fromkeys(bwm_df['eid'].tolist()))
```

```python
for eid in eids:
    rows = by_eid.get_group(eid)
    tasks.append((eid, rows[['pid', 'probe_name']].to_dict('records'),
                  str(rows['subject'].iloc[0]), str(rows['lab'].iloc[0])))
```

iii. In the trajectory the agent examined `bwm_release.csv`, noted that it contains 459 unique `eid`s, and used those `eid`s as the session-level work items.

## 1-d. How are the data split into trials?

i. Trials are taken from the IBL trials table loaded by `SessionLoader`; the code treats that table as already being one row per trial and creates trial-level outputs by masking rows of that table.

ii.
```python
sess_loader = SessionLoader(one=one, eid=eid)
sess_loader.load_trials()
trials = sess_loader.trials
```

```python
keep = trials_mask & np.isfinite(interval_begs_all) & (prior_all >= 0)
...
n_trials = int(keep.sum())
...
'neural': [binned_spikes[k] for k in range(n_trials)],
'input': [inputs[k] for k in range(n_trials)],
'output': [outputs[k] for k in range(n_trials)],
```

iii. In the trajectory the agent read the reference code and then implemented trial handling entirely around the session trials table and a boolean keep-mask, without deriving trials from any other stream.

## 1-e. How are trials filtered based on quality controls?

i. The agent reimplements the reference repository’s `load_trials_and_mask(..., max_trial_len=10.)` logic: it drops trials with reaction times outside 0.08-2.0 s, trials with no response, trials with NaNs in several key columns, and trials with `feedback_times - goCue_times > 10 s`. It then further drops trials whose alignment time is not finite, whose `probabilityLeft` is not one of `{0.2, 0.5, 0.8}`, or whose wheel/whisker traces do not cover the full decoding window or contain NaNs after interpolation.

ii.
```python
query = f'(firstMovement_times - stimOn_times < {MIN_RT})'
query += f' | (firstMovement_times - stimOn_times > {MAX_RT})'
query += f' | (feedback_times - goCue_times > {MAX_TRIAL_LEN})'
for event in nan_exclude:
    query += f' | {event}.isnull()'
query += ' | (choice == 0)'          # no response

mask = ~trials.eval(query)
```

```python
prior_all = np.full(len(trials), -1, dtype=np.int64)
for value, code in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior_all[np.isclose(p_left_all, value)] = code

keep = trials_mask & np.isfinite(interval_begs_all) & (prior_all >= 0)
```

```python
for name, (t, v) in traces.items():
    binned, good = bin_behaviour(t, v, safe_begs)
    beh_binned[name] = binned
    keep &= good
```

iii. In the trajectory the agent explicitly said it was following `load_trials_and_mask(..., max_trial_len=10.)`, diagnosed trial attrition by printing counts before and after wheel/whisker coverage, and kept that stricter mask in the final script.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural tensor is derived from `spikes['times']` and `spikes['clusters']`, with `clusters['label']` and `clusters['acronym']` used to decide which clusters survive and how they are labeled anatomically.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
...
clusters = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
...
sc = np.asarray(spikes['clusters'], dtype=np.int64)
st = np.asarray(spikes['times'], dtype=np.float64)
```

iii. In the trajectory the agent inspected the spike-sorting load on a sample session, looked at the returned cluster labels and spike counts, and then wrote `load_session_units` around the spike time and cluster arrays.

## 2-b. How is the `neural` data processed?

i. The agent merges all retained probes in a session into one unit population, renumbers units consecutively across probes, and bins spikes into 20 ms non-overlapping bins over a 2 s stimulus-aligned window. The saved neural values are spike counts per bin, not firing rates.

ii.
```python
lut = np.full(int(clusters.shape[0]), -1, dtype=np.int64)
lut[keep_ids] = np.arange(keep_ids.size) + offset
...
offset += keep_ids.size
```

```python
out = np.zeros((n_trials, n_units, N_BINS), dtype=np.float32)
...
b = np.floor((t - interval_begs[k]) / BINSIZE).astype(np.int64)
np.clip(b, 0, N_BINS - 1, out=b)
counts = np.bincount(u * N_BINS + b, minlength=flat)
out[k] = counts.reshape(n_units, N_BINS)
```

```python
'neural': [binned_spikes[k] for k in range(n_trials)],
...
'neural_units': 'spike count per 20 ms bin',
```

iii. In the trajectory the agent read the reference binning code, implemented its own `bin_spikes`, tested a few sessions, and kept the count representation that it documented in metadata as “spike count per 20 ms bin.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent keeps clusters with `label >= 1` and drops clusters whose Beryl-mapped acronym is `root` or `void`, treating both as non-grey-matter / unassigned locations. It also drops spikes attached to discarded clusters and spikes with non-finite timestamps.

ii.
```python
NON_GREY_MATTER = ('root', 'void')
```

```python
beryl = np.asarray(brain_regions.acronym2acronym(
    clusters['acronym'].to_numpy(), mapping='Beryl'))
...
keep = (clusters['label'].to_numpy() >= 1) & ~np.isin(beryl, NON_GREY_MATTER)
```

```python
mapped = lut[sc]
ok = mapped >= 0
ok &= np.isfinite(st)
```

iii. In the trajectory the agent inspected cluster labels from a sample spike-sorting load, adopted `label == 1` as the well-isolated-unit cutoff, and described the unit filter in the script header as grey-matter only with `{root, void}` excluded.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to `stimOn_times`. The code defines each trial window by `stimOn_times - 0.5 s`, then bins spikes for the subsequent 2.0 s interval, so the neural tensor spans `[-0.5, 1.5)` relative to stimulus onset.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
```

```python
align_times = trials[ALIGN_TIME].to_numpy(dtype=np.float64)
interval_begs_all = align_times + TIME_WINDOW[0]
...
interval_begs = interval_begs_all[keep]
binned_spikes = bin_spikes(spike_times, spike_units, n_units, interval_begs)
```

iii. In the trajectory the agent read the reference caching parameters, copied `stimOn_times` and the `(-0.5, 1.5)` window into the script configuration, and described that choice in the file header.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins and 100 timepoints per trial. There is no extra temporal rebinning beyond the initial spike binning onto that grid.

ii.
```python
BINSIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100
```

```python
b = np.floor((t - interval_begs[k]) / BINSIZE).astype(np.int64)
```

iii. In the trajectory the agent copied the reference caching parameters from `0_data_caching.py`, tested converted sessions with the decoder verifier, and confirmed they all had `T = 100`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the trial alignment variable `trials['stimOn_times']` plus the fixed decoding window and bin geometry. The raw event is the stimulus-onset timestamp; the actual feature values are relative times computed from that alignment choice.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
```

```python
align_times = trials[ALIGN_TIME].to_numpy(dtype=np.float64)
interval_begs_all = align_times + TIME_WINDOW[0]
```

iii. In the trajectory the agent repeatedly framed the whole conversion around `stimOn_times`, both when reading the reference code and when writing the alignment summary in the script header.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The agent computes the centres of the 20 ms neural bins and reuses that same vector for every retained trial. It is not read from a raw continuous trace.

ii.
```python
bin_centres = TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE
...
inputs = np.empty((n_trials, 2, N_BINS), dtype=np.float32)
inputs[:, 0, :] = bin_centres[None, :]
```

iii. In the trajectory the agent unit-tested the helper values and printed the first and last bin centres, confirming the intended `[-0.49, ..., 1.49]` grid.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. It is aligned by construction to the neural bins: the feature equals the centre of each 20 ms neural bin in the stimulus-aligned window.

ii.
```python
binned_spikes = bin_spikes(spike_times, spike_units, n_units, interval_begs)
...
bin_centres = TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE
inputs[:, 0, :] = bin_centres[None, :]
```

iii. In the trajectory the agent treated `time_from_stimulus_onset` as the neural bin grid itself and verified that all converted sessions had exactly 100 aligned timepoints.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `trials['probabilityLeft']`. The agent infers a new block whenever that per-trial prior changes.

ii.
```python
def trial_number_in_block(trials):
    p_left = trials['probabilityLeft'].to_numpy(dtype=float)
    new_block = np.ones(len(p_left), dtype=bool)
    new_block[1:] = ~(p_left[1:] == p_left[:-1])
```

iii. In the trajectory the agent followed the reference idea that block identity is encoded in `probabilityLeft`, and later explicitly unit-tested the helper on a synthetic sequence of block probabilities.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code counts trials from zero within each contiguous run of constant `probabilityLeft`, and it does so on the full trials table before filtering so excluded trials still advance the count.

ii.
```python
block_id = np.cumsum(new_block) - 1
block_start = np.flatnonzero(new_block)
idx_in_block = np.arange(len(p_left)) - block_start[block_id]
return idx_in_block.astype(np.float32)
```

```python
n_in_block = trial_number_in_block(trials)[keep]
inputs[:, 1, :] = n_in_block[:, None]
```

iii. In the trajectory the agent noticed a bug in its first block-index implementation, unit-tested the helper, then fixed it so the final code truly returns a 0-based within-block count computed before filtering.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from `trials['choice']`.

ii.
```python
choice = (trials['choice'].to_numpy()[keep] == -1).astype(np.int64)
```

iii. In the trajectory the agent verified the IBL sign convention on sample data and documented the left/right interpretation directly above the choice mapping.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The code converts IBL’s signed coding into the requested binary coding: `+1` becomes left=`0`, `-1` becomes right=`1`. No-response trials (`choice == 0`) are removed earlier by the trial mask, and the resulting scalar choice is broadcast across all 100 time bins of the trial.

ii.
```python
query += ' | (choice == 0)'          # no response
```

```python
choice = (trials['choice'].to_numpy()[keep] == -1).astype(np.int64)
...
outputs[:, 0, :] = choice[:, None]
```

iii. In the trajectory the agent said it verified the sign convention and then encoded the requested left/right remapping in the final script.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from `trials['probabilityLeft']`.

ii.
```python
p_left_all = trials['probabilityLeft'].to_numpy(dtype=float)
prior_all = np.full(len(trials), -1, dtype=np.int64)
for value, code in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior_all[np.isclose(p_left_all, value)] = code
```

iii. In the trajectory the agent treated `probabilityLeft` as both the block identifier and the source of the decoder target, matching the task specification.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code recodes `probabilityLeft` from `{0.2, 0.5, 0.8}` to category ids `{0, 1, 2}`, drops trials with any other value, and broadcasts the retained category across all 100 time bins of the trial.

ii.
```python
for value, code in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior_all[np.isclose(p_left_all, value)] = code

keep = trials_mask & np.isfinite(interval_begs_all) & (prior_all >= 0)
```

```python
prior = prior_all[keep]
...
outputs[:, 1, :] = prior[:, None]
```

iii. In the trajectory the agent explicitly added the `prior_all >= 0` filter while cleaning up the session conversion logic, so the code and the stated mapping stayed consistent.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed comes from the wheel stream loaded by `SessionLoader`, specifically the wheel timestamps and the wheel velocity that `SessionLoader.load_wheel()` computes from the raw wheel position / timestamp files.

ii.
```python
sl = SessionLoader(one=one, eid=eid)
sl.load_wheel()
traces['wheel-speed'] = (sl.wheel['times'].to_numpy(dtype=np.float64),
                         np.abs(sl.wheel['velocity'].to_numpy(dtype=np.float64)))
```

iii. In the trajectory the agent inspected the ibllib code and the reference repository utilities, then documented wheel speed as `|velocity|` from the ibllib wheel loader.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The code takes the absolute value of the loader-provided wheel velocity, interpolates it onto the per-trial stimulus-aligned decoding bins, and then discretizes the full session trace into three equal-frequency levels.

ii.
```python
traces['wheel-speed'] = (sl.wheel['times'].to_numpy(dtype=np.float64),
                         np.abs(sl.wheel['velocity'].to_numpy(dtype=np.float64)))
```

```python
binned = np.interp(sample_times.ravel(), times, values).reshape(n_trials, N_BINS)
```

```python
wheel_lvl, wheel_thr = discretize_tertiles(beh_binned['wheel-speed'][keep])
```

iii. In the trajectory the agent described wheel speed in the file header as `|velocity|` from `SessionLoader`, tested behaviour coverage/attrition on sample sessions, and kept the interpolation plus tertile split in the final script.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The code uses per-session tertiles: it computes the 1/3 and 2/3 quantiles of all retained binned wheel-speed values in that session, then assigns bins with `np.digitize` into `low`, `medium`, and `high`.

ii.
```python
def discretize_tertiles(values):
    lo, hi = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    if not hi > lo:
        hi = np.nextafter(lo, np.inf)
    return np.digitize(values, [lo, hi]).astype(np.int64), (float(lo), float(hi))
```

```python
wheel_lvl, wheel_thr = discretize_tertiles(beh_binned['wheel-speed'][keep])
```

iii. In the trajectory the agent justified per-session tertiles in the function docstring as a way to balance classes and to avoid confounding session-specific scale differences.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is interpolated onto one sample per neural bin, but the sample point is the right edge of each 20 ms bin (`beg + 0.02, ..., beg + 2.0`) rather than the bin centre.

ii.
```python
offsets = (np.arange(N_BINS) + 1) * BINSIZE
sample_times = interval_begs[:, None] + offsets[None, :]

binned = np.interp(sample_times.ravel(), times, values).reshape(n_trials, N_BINS)
```

iii. In the trajectory the agent explicitly wrote that behaviour is “linearly interpolated onto the right edge of each 20 ms bin,” citing the repository’s `get_behavior_per_interval` logic as its justification.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from the camera motion-energy stream: the code tries `leftCamera` first and falls back to `rightCamera`, using the corresponding frame times and `whiskerMotionEnergy` column.

ii.
```python
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sl_me = SessionLoader(one=one, eid=eid)
        sl_me.load_motion_energy(views=[view])
        df = sl_me.motion_energy[cam]
        whisker = (df['times'].to_numpy(dtype=np.float64),
                   df['whiskerMotionEnergy'].to_numpy(dtype=np.float64))
        break
    except Exception:
        continue
```

iii. In the trajectory the agent inspected the session files and revisions, saw that camera times and motion-energy arrays live in separate released datasets, and encoded a left-preferred, right-fallback policy in the final script.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The code uses the released whisker motion-energy trace as-is, linearly interpolates it onto the per-trial decoding bins, and discretizes the full session trace into three equal-frequency levels.

ii.
```python
df = sl_me.motion_energy[cam]
whisker = (df['times'].to_numpy(dtype=np.float64),
           df['whiskerMotionEnergy'].to_numpy(dtype=np.float64))
```

```python
binned = np.interp(sample_times.ravel(), times, values).reshape(n_trials, N_BINS)
```

```python
whisk_lvl, whisk_thr = discretize_tertiles(beh_binned['whisker-motion-energy'][keep])
```

iii. In the trajectory the agent treated whisker motion energy analogously to wheel speed, diagnosed missing-whisker sessions during the full run, and kept the interpolation plus tertile discretization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is thresholded exactly like wheel speed: the 1/3 and 2/3 quantiles of the session’s retained binned whisker values define `low`, `medium`, and `high`.

ii.
```python
def discretize_tertiles(values):
    lo, hi = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    if not hi > lo:
        hi = np.nextafter(lo, np.inf)
    return np.digitize(values, [lo, hi]).astype(np.int64), (float(lo), float(hi))
```

```python
whisk_lvl, whisk_thr = discretize_tertiles(beh_binned['whisker-motion-energy'][keep])
```

iii. In the trajectory the agent justified per-session tertiles for behaviour generally and stored the resulting whisker thresholds in session metadata.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Like wheel speed, whisker motion energy is interpolated onto one sample per neural bin at the right edge of each 20 ms bin, not at the bin centre.

ii.
```python
offsets = (np.arange(N_BINS) + 1) * BINSIZE
sample_times = interval_begs[:, None] + offsets[None, :]

binned = np.interp(sample_times.ravel(), times, values).reshape(n_trials, N_BINS)
```

iii. In the trajectory the agent’s written rationale for behavioural alignment was the same for wheel and whisker: match `get_behavior_per_interval` by sampling at the right bin edges.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent mostly handles missing or malformed data by dropping it. Trials are removed for NaNs in key trial columns, non-finite alignment times, missing valid block-probability labels, and incomplete or NaN wheel/whisker windows. Sessions are skipped if they end up with fewer than two usable trials, no whisker motion-energy stream, or no retained units. Empty probe insertions and non-finite spike timestamps are also ignored.

ii.
```python
for event in nan_exclude:
    query += f' | {event}.isnull()'
...
keep = trials_mask & np.isfinite(interval_begs_all) & (prior_all >= 0)
```

```python
if whisker is None:
    raise RuntimeError('no whisker motion energy available')
```

```python
if keep.sum() < MIN_TRIALS_PER_SESSION:
    return None, 'too few trials with usable behaviour'
...
if spike_times is None:
    return None, 'no well-isolated grey-matter units'
```

```python
ok = mapped >= 0
ok &= np.isfinite(st)
```

iii. In the trajectory the agent diagnosed missing-whisker sessions during the full run, printed skipped-session reasons, and also found and fixed a bug in the block-index helper rather than leaving that data issue unresolved.

## 10-a. What are the most time-consuming steps of the code?

i. The code is dominated by loading and merging spike-sorting outputs and then binning spikes trial by trial. Behaviour interpolation is cheaper and largely vectorized.

ii.
```python
spikes, clusters, channels = ssl.load_spike_sorting()
clusters = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
```

```python
for k in range(n_trials):
    t = spike_times[i0[k]:i1[k]]
    ...
    counts = np.bincount(u * N_BINS + b, minlength=flat)
```

iii. In the trajectory the agent timed a sample spike-sorting load, saw that it took about 15 seconds for one probe and involved tens of millions of spikes, then ran the full conversion where the long-running work was clearly session-by-session spike loading and binning.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining candidate is the per-trial loop in `bin_spikes`. Behaviour interpolation is already vectorized across all trials by flattening all sample times into one `np.interp` call, so the AI avoided one of the trial loops present in the reference solution.

ii.
```python
for k in range(n_trials):
    t = spike_times[i0[k]:i1[k]]
    if t.size == 0:
        continue
    u = spike_units[i0[k]:i1[k]]
    b = np.floor((t - interval_begs[k]) / BINSIZE).astype(np.int64)
    ...
```

```python
sample_times = interval_begs[:, None] + offsets[None, :]
binned = np.interp(sample_times.ravel(), times, values).reshape(n_trials, N_BINS)
```

iii. In the trajectory the agent emphasized throughput testing and ended up keeping the per-trial spike loop while already rewriting behaviour resampling in a more vectorized form than the reference.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats some setup and session I/O: it constructs `BrainRegions()` inside every session conversion, creates separate `SessionLoader` instances for wheel and motion-energy loading, and may attempt motion-energy loading twice per session when falling back from left to right camera.

ii.
```python
def convert_session(eid, probe_rows, subject, lab):
    from iblatlas.regions import BrainRegions
    ...
    brain_regions = BrainRegions()
```

```python
sl = SessionLoader(one=one, eid=eid)
sl.load_wheel()
...
for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
    try:
        sl_me = SessionLoader(one=one, eid=eid)
        sl_me.load_motion_energy(views=[view])
```

iii. In the trajectory the agent focused on correctness and end-to-end throughput rather than eliminating repeated setup, and the final script retains these repeated loader/setup steps.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores several extras that the downstream decoder does not use: `lab`, `n_trials_total`, `n_probes`, per-session wheel/whisker thresholds, lists of skipped sessions, and verbose metadata. It also bins wheel and whisker traces on all trials before the final keep-mask removes some of those trials.

ii.
```python
return {
    'eid': eid,
    'subject': subject,
    'lab': lab,
    ...
    'n_trials_total': int(len(trials)),
    'n_probes': int(len(probe_rows)),
    'wheel_speed_thresholds': wheel_thr,
    'whisker_motion_energy_thresholds': whisk_thr,
}, None
```

```python
'metadata': {
    ...
    'output_discretization': (
        'wheel_speed and whisker_motion_energy are split at the 1/3 and 2/3 '
        'quantiles of that session; thresholds are in session_info.'),
    'session_info': [
        {'eid': r['eid'], 'subject': r['subject'], 'lab': r['lab'],
         'n_probes': r['n_probes'], 'n_neurons': r['n_units'],
         'n_trials': r['n_trials'], 'n_trials_recorded': r['n_trials_total'],
         'brain_regions': sorted(set(r['acronyms'].tolist())),
         'wheel_speed_thresholds': r['wheel_speed_thresholds'],
         'whisker_motion_energy_thresholds':
             r['whisker_motion_energy_thresholds']}
        for r in results],
    'skipped_sessions': [{'eid': e, 'reason': str(m).splitlines()[0]}
                         for e, m in sorted(skipped)],
},
```

iii. In the trajectory the agent spent effort on rich metadata, skip logging, and diagnostic verification, so the final artifact keeps more bookkeeping than the decoder itself needs.
