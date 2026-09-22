# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything is read through the IBL **ONE** API against the staged cache at `/app/data/one_cache`; no file is ever opened directly. The client is built with `ONE(base_url='https://openalyx.internationalbrainlab.org', silent=True)` with **no password and no explicit mode**, so the staged Alyx token plus the cached REST responses under `<cache>/.rest` answer every query from disk (the agent established experimentally that `mode='local'` could not resolve the revision folders — `#2025-03-03#` for the trials table, `#2025-05-29#` for motion energy — while the default remote-with-cached-REST mode could).

The list of sessions is **not** obtained from `one.search`; it is read from the reference repository's own release freeze, `/app/code/code_zhang2025/data/bwm_release.csv`, which holds one row per probe insertion with columns `pid, eid, probe_name, session_number, date, subject, lab` (699 rows, 459 eids, 139 subjects). This is exactly the file `0_data_caching.py` iterates over. `DATALIMIT_SUBSET.csv` is honoured if present (it is not, on this dataset).

Per session, three loaders are used:
* `SpikeSortingLoader(pid=..., eid=..., pname=...)` once per probe row — pids come straight from the csv, so `one.eid2pid` (which needs a live Alyx) is never called;
* `SessionLoader.load_trials()` for the trials table;
* `SessionLoader.load_wheel()` and `SessionLoader.load_motion_energy(views=[...])` for the behavioural streams.

Sessions are processed in parallel (`multiprocessing` `spawn` pool, 16 workers), each worker writing a per-session pickle to `/app/work/sessions/<eid>.pkl`; a second pass re-reads those and assembles the final dictionary, which makes the run resumable.

ii.
```python
def get_one():
    from one.api import ONE
    # No password / no explicit mode: the staged Alyx token plus the cached REST
    # responses under <cache>/.rest let ONE answer every query from disk.
    return ONE(base_url='https://openalyx.internationalbrainlab.org', silent=True)
```
```python
BWM_FREEZE = '/app/code/code_zhang2025/data/bwm_release.csv'
DATALIMIT  = '/app/data/DATALIMIT_SUBSET.csv'
...
bwm = pd.read_csv(BWM_FREEZE, index_col=0)
if os.path.exists(DATALIMIT):
    sub = pd.read_csv(DATALIMIT)
    col = 'eid' if 'eid' in sub.columns else sub.columns[0]
    bwm = bwm[bwm.eid.isin(sub[col].astype(str))]
eids = list(dict.fromkeys(bwm.eid.tolist()))
jobs = [(eid, bwm[bwm.eid == eid]) for eid in eids
        if not os.path.exists(os.path.join(args.tmp, eid + '.pkl'))]
```
```python
for _, r in rows.iterrows():
    ssl = SpikeSortingLoader(pid=r.pid, one=one, eid=eid, pname=r.probe_name)
    sp, cl, ch = ssl.load_spike_sorting()
...
sl = SessionLoader(one=one, eid=eid)
trials, trials_mask = load_trials_and_mask(sl)
sl.load_wheel()
sl.load_motion_energy(views=[view])
```

iii. From the trajectory: "Sessions are the 459 eids of the public BWM release freeze (`code_zhang2025/data/bwm_release.csv`) — exactly what the reference caching script iterates over." The agent spent ~15 steps (steps 40–49) establishing that ONE could be made to work offline: it found that `load_cache(tag='Brainwidemap')` in local mode returned trials tables with only one column and failed on motion energy, traced this to the on-disk revision folders missing from the release parquet tables, noticed the `.rest` cache of Alyx responses, and concluded "the reference used ONE in default 'auto'/remote mode where REST queries are served from the cache." It also noted that `eid2pid` requires remote access but "bwm_release.csv provides pid/eid/probe_name", which is why pids are taken from the csv.

## 1-b. How are the data split into subjects?

i. The subject name is taken from the `subject` column of `bwm_release.csv` for the session's rows — no path parsing, no extra query. During assembly the `subjects` list is built in order of first appearance and `subject_idx` records each session's index into it. The result is 136 subjects over 444 sessions.

ii.
```python
    return {
        'eid': eid,
        ...
        'subject': str(rows.subject.iloc[0]),
        'lab': str(rows.lab.iloc[0]),
```
```python
        if res['subject'] not in subjects:
            subjects.append(res['subject'])
        data['subject_idx'].append(subjects.index(res['subject']))
...
data['subjects'] = subjects
data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. Not discussed explicitly in the trajectory; the freeze file already carries a unique subject id per session, so nothing has to be derived. The agent reported the outcome as "136 mice" in its summary.

## 1-c. How are the data split into sessions?

i. A session is the unit the release freeze is organised by. The distinct `eid`s of `bwm_release.csv` are taken in file order (`dict.fromkeys` preserves order and de-duplicates the 699 probe rows into 459 eids), and the probe rows belonging to an eid are grouped into one job. One job = one session = one worker call = one output entry in every top-level list.

ii.
```python
eids = list(dict.fromkeys(bwm.eid.tolist()))
...
jobs = [(eid, bwm[bwm.eid == eid]) for eid in eids ...]
```
```python
def _process_session(eid, rows):
    ...
    for _, r in rows.iterrows():          # the probes of this session
```

iii. The agent's summary: the eids of the freeze are "exactly what the reference caching script iterates over"; the reference loops `for eid_idx, eid in enumerate(include_eids)`. No split has to be invented.

## 1-d. How are the data split into trials?

i. The trials table returned by `SessionLoader.load_trials()` has one row per trial, so the split is given by the data. Each retained row becomes one entry in the per-session `neural`/`input`/`output` lists, with its window defined as `stimOn_times + (-0.5, 1.5)` s.

ii.
```python
    if sess_loader.trials.empty:
        sess_loader.load_trials()
    trials = sess_loader.trials
...
    align = trials[ALIGN_TIME].to_numpy(dtype=float)
    valid = trials_mask & np.isfinite(align)
    ...
    interval_begs = align + TIME_WINDOW[0]
```

iii. No explicit reasoning; the trials table is already one row per trial, and the agent simply reproduced the reference code's `load_trials_and_mask` + `trials_df[align_time]` idiom.

## 1-e. How are trials filtered based on quality controls?

i. Two stages, ANDed together.

**(1) The standard BWM trial mask**, a verbatim re-implementation of `ibl_data_utils.load_trials_and_mask` with the arguments `prepare_data` uses (`min_rt=0.08`, `max_rt=2.0`, `max_trial_len=10.0`, default `nan_exclude`, `exclude_nochoice=True`). A trial is dropped if: reaction time (`firstMovement_times - stimOn_times`) is `< 0.08 s` or `> 2 s`; trial length (`feedback_times - goCue_times`) `> 10 s`; any of `stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType` is NaN; or `choice == 0` (no response). A non-finite `stimOn_times` is re-checked separately.

**(2) Behavioural coverage**, a re-implementation of the `get_behavior_per_interval` skip rules: within the trial window the wheel and the whisker trace must exist, contain no NaN, start no more than one bin (20 ms) after the window opens and end no more than one bin before it closes. Both the wheel and the whisker mask must pass.

Sessions are then dropped if fewer than 2 trials survive (the decoder's minimum), if there are no well-isolated neurons, if no spike sorting is present, or if there is no whisker motion energy at all.

ii.
```python
def load_trials_and_mask(sess_loader, min_rt=MIN_RT, max_rt=MAX_RT,
                         max_trial_len=MAX_TRIAL_LEN):
    nan_exclude = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
                   'firstMovement_times', 'feedbackType']
    ...
    query = f'(firstMovement_times - stimOn_times < {min_rt})'
    query += f' | (firstMovement_times - stimOn_times > {max_rt})'
    if max_trial_len is not None:
        query += f' | (feedback_times - goCue_times > {max_trial_len})'
    for event in nan_exclude:
        query += f' | {event}.isnull()'
    query += ' | (choice == 0)'
    mask = ~trials.eval(query)
    return trials, mask.to_numpy()
```
```python
        if len(tv) == 0:
            continue
        if np.any(np.isnan(tv)):
            continue
        if abs(interval_begs[k] - tt[0]) > BINSIZE:      # data starts too late
            continue
        if abs(interval_ends[k] - tt[-1]) > BINSIZE:     # data ends too early
            continue
```
```python
    valid = trials_mask & np.isfinite(align)
    ...
    valid = valid & ws_good & wm_good
    if valid.sum() < MIN_TRIALS_PER_SESSION:
        return {'eid': eid, 'skip': f'only {int(valid.sum())} trials with complete behaviour'}
```

iii. Module docstring: "the standard BWM trial mask (`load_trials_and_mask`) with `max_trial_len=10 s`, exactly as `prepare_data` calls it … Trials for which the behavioural traces do not cover the decoding window are dropped as well (`get_behavior_per_interval`)." Final summary: "Standard BWM trial mask (`load_trials_and_mask` with `max_trial_len=10`)". The agent had read the data paper's matching statement ("trials were excluded if … choice, probabilityLeft, feedbackType, feedback times, stimOn times and firstMovement times [could not be detected] … outside the range of 0.08–2.00 s").

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes['times']` and `spikes['clusters']` from `SpikeSortingLoader.load_spike_sorting()`, one pair per probe. The merged cluster table (`SpikeSortingLoader.merge_clusters(sp, cl, ch).to_df()`) supplies only two columns that are used: `label` (the IBL quality score) and `acronym` (the histological location, later mapped to Beryl). The counted array itself is built from the two spike arrays alone.

ii.
```python
    for _, r in rows.iterrows():
        ssl = SpikeSortingLoader(pid=r.pid, one=one, eid=eid, pname=r.probe_name)
        sp, cl, ch = ssl.load_spike_sorting()
        if sp is None or len(sp) == 0:
            continue
        cld = SpikeSortingLoader.merge_clusters(sp, cl, ch).to_df()
        spikes_list.append(sp)
        clusters_list.append(cld)
```
```python
    spike_times = np.concatenate([s['times'] for s in ms])
    spike_clusters = np.concatenate([s['clusters'] for s in ms])
...
    beryl = np.asarray(br.acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl'))
    label = clusters['label'].to_numpy()
```

iii. Implicit — this is what `load_spiking_data`/`prepare_data` in the reference code produce (`neural_dict = {'spike_times': ..., 'spike_clusters': ..., 'cluster_regions': ...}`). The agent verified the loader works offline before writing the script (step 46: "SpikeSortingLoader works offline").

## 2-b. How is the `neural` data processed?

i. Three steps.

**Merge probes.** All probes of a session are concatenated into one population, with the second probe's cluster ids offset by the first probe's cluster count — a line-for-line copy of `ibl_data_utils.merge_probes` (including its `cluster_max = clusters.index.max() + 1` bookkeeping, which is correct for the ≤2 probes any BWM session has).

**Select neurons and renumber.** See 2-c; survivors are remapped to `0 … n_neurons-1` through a lookup table.

**Bin.** Spikes are counted into 100 non-overlapping 20 ms bins spanning `stimOn_times + (-0.5, 1.5)` s. The implementation sorts the merged spike train once, uses `searchsorted` to slice each trial's spikes, computes `floor((t - beg)/0.02)` clipped to `[0, 99]`, and fills the whole unit×bin grid with one `np.bincount` on a flat index. The result is **raw spike counts per 20 ms bin**, stored as `float32`; there is **no** conversion to firing rate, no smoothing, and no normalisation (`'neural_units': 'spike counts per 20 ms bin'`). A spot-check of one session's output confirms integer-valued counts with max 12 and mean 0.435 per bin.

ii.
```python
    cmax = 0
    ms, mc = [], []
    for cl, sp in zip(clusters_list, spikes_list):
        sp = dict(sp)
        sp['clusters'] = sp['clusters'] + cmax
        cmax = cl.index.max() + 1
        ms.append(sp)
        mc.append(cl)
    clusters = pd.concat(mc, ignore_index=True)
```
```python
def bin_spikes(spike_times, spike_clusters, n_clusters, interval_begs):
    """... Equivalent to ibl_data_utils.get_spike_data_per_interval (bincount2D with
    xlim=[t_beg, t_end], keeping the first N_BINS bins) but vectorised."""
    out = np.zeros((n_trials, n_clusters, N_BINS), dtype=np.float32)
    order = np.argsort(spike_times, kind='stable')
    st, sc = spike_times[order], spike_clusters[order]
    beg = np.searchsorted(st, interval_begs, side='left')
    end = np.searchsorted(st, interval_begs + N_BINS * BINSIZE, side='left')
    for k in range(n_trials):
        b, e = beg[k], end[k]
        if e <= b: continue
        tb = np.floor((st[b:e] - interval_begs[k]) / BINSIZE).astype(np.int64)
        np.clip(tb, 0, N_BINS - 1, out=tb)
        flat = sc[b:e] * N_BINS + tb
        out[k] = np.bincount(flat, minlength=n_clusters * N_BINS).reshape(n_clusters, N_BINS)
    return out
```

iii. Docstring: "all probes of a session are merged into one population (`merge_probes`), as in the reference code and in the data paper ('neurons in the same session … were combined across probes')" and "Alignment: stimulus onset (`stimOn_times`), window (-0.5, +1.5) s, 20 ms non-overlapping bins -> T = 100 — the exact `params` dict of `0_data_caching.py`." Keeping counts rather than rates follows the method paper, which decodes from "temporally binned spike counts".

## 2-c. How is the `neural` data filtered based on quality controls?

i. A cluster is kept only if `label >= 1` **and** its Beryl acronym is neither `root` nor `void`. `label` is the IBL pipeline's quality score (0, 1/3, 2/3, 1) formed from the three RIGOR single-unit metrics; requiring 1 selects the data paper's "well-isolated neurons". Excluding `root`/`void` implements the paper's additional restriction to grey-matter regions (`void` = outside the brain; `root` = a site whose Allen label is not one of the Beryl summary structures). Spikes from rejected clusters are dropped and the survivors renumbered contiguously.

The agent explicitly *declined* to also apply the data paper's "at least five well-isolated neurons per region per session" criterion.

Outcome: 62,763 neurons over 444 sessions (mean 141/session, min 1, max 516). The 75.8k well-isolated units the agent measured drop to ~62.8k after the `root`/`void` cut.

ii.
```python
    beryl = np.asarray(br.acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl'))
    label = clusters['label'].to_numpy()
    keep = (label >= 1) & ~np.isin(beryl, ['root', 'void'])
    keep_ids = np.nonzero(keep)[0]
    n_neurons = len(keep_ids)
    if n_neurons < MIN_NEURONS_PER_SESSION:
        return {'eid': eid, 'skip': f'only {n_neurons} well-isolated neurons'}

    sel = np.isin(spike_clusters, keep_ids)
    spike_times = spike_times[sel]
    remap = np.full(int(clusters.index.max()) + 1, -1, dtype=np.int64)
    remap[keep_ids] = np.arange(n_neurons)
    spike_clusters = remap[spike_clusters[sel]]
    regions = beryl[keep_ids]
```

iii. Two justifications are given, one scientific and one practical. Docstring: "well-isolated neurons only, i.e. clusters with IBL label == 1, which is the conjunction of the three RIGOR single-unit metrics used by the data paper … Additionally restricted to grey matter (Beryl acronym not 'root'/'void'), again as in the data paper. The reference caching script keeps every Kilosort unit; that is impossible here because 622k units x 197k trials x 100 bins is ~164 GB, while the well-isolated set (75.7k neurons, the number quoted by the data paper) is ~11 GB."

Final summary: "One deliberate deviation from the reference caching script … I instead keep the well-isolated neurons (cluster `label == 1`, i.e. the three RIGOR single-unit metrics) in grey matter — the exact set the data paper uses for all of its analyses, and whose size I verified reproduces the paper's quoted 75,708 units. I checked the paper's additional per-region criterion (≥5 neurons/region/session) and did *not* apply it: it exists for the paper's region-wise decoding, whereas here we decode from the whole session population, so it would discard 2,280 usable neurons for no reason."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All IBL streams (spike times, trial event times, wheel timestamps, camera frame times) are already expressed in seconds on one synchronised session clock, so alignment is a subtraction. Each trial's window begins at `interval_begs = stimOn_times - 0.5` and spans 2 s; the bin index of a spike is `floor((t - interval_begs)/0.02)`, i.e. the spike time expressed relative to that trial's stimulus onset. Trials with non-finite `stimOn_times` are removed before any binning.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)          # seconds relative to the alignment event
...
    align = trials[ALIGN_TIME].to_numpy(dtype=float)
    valid = trials_mask & np.isfinite(align)
    interval_begs = align + TIME_WINDOW[0]
```
```python
        tb = np.floor((st[b:e] - interval_begs[k]) / BINSIZE).astype(np.int64)
```
```python
        'temporal_alignment_event': 'stimulus onset (stimOn_times)',
        'off_start': TIME_WINDOW[0],
        'off_end': TIME_WINDOW[1],
```

iii. The window is taken from the reference `params` dict, which the docstring reproduces verbatim. The agent additionally sanity-checked the alignment empirically: "Alignment sanity-checked — wheel speed and whisker energy are low pre-stimulus and jump after onset."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 per trial over the 2 s window, identical for every trial and session; `metadata['time_bin_size'] = 20.0` (ms). Spikes are counted directly into these bins — there is no intermediate binning and no rebinning or resampling of the neural data. (The *behavioural* traces are resampled onto the same grid; see 7-b/8-b.) Note the method paper's text mentions 50 ms bins, but the reference code's `params` dict uses `binsize: 0.02`, and the agent followed the code.

ii.
```python
BINSIZE = 0.02                     # seconds
N_BINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # 100
...
        'time_bin_size': BINSIZE * 1000.0,
        'n_timepoints': N_BINS,
```

iii. Docstring: "20 ms non-overlapping bins -> T = 100 — the exact `params` dict of `0_data_caching.py`." The method paper likewise states "each divided into 20-ms bins, producing T = 100 time steps".

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not read from the raw data at all; it is defined by the alignment. Because every trial uses the same window and bin grid around its own `stimOn_times`, the same 100-value vector serves every trial of every session. The values are the **right edges** of the 100 bins: `-0.48, -0.46, …, 1.48, 1.50` s.

ii.
```python
# time stamp of each bin, relative to the alignment event.  These are the right edges of
# the bins, i.e. the sample times used by `get_behavior_per_interval` in the reference
# code (x_interp = linspace(beg + binsize, end, n_bins)).
BIN_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
```
```python
    inputs[:, 0, :] = BIN_TIMES[None, :]
```

iii. The inline comment is the justification: the grid is chosen to be the one `get_behavior_per_interval` interpolates onto in the reference code, `x_interp = np.linspace(interval_begs + binsize, interval_ends, n_bins)`. The full vector is also written into `metadata['bin_times_s']`.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the grid once at import and broadcasting it to every trial. It is a continuous ramp (as the decoder-task spec requires: "Time since stimulus onset, continuous, time-varying"), cast to `float32`.

ii.
```python
BIN_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
...
    inputs = np.empty((n_trials, 2, N_BINS), dtype=np.float32)
    inputs[:, 0, :] = BIN_TIMES[None, :]
```
```python
    data['input_names'] = ['time_from_stimulus_onset', 'trial_number_in_block']
```

iii. N/A — the variable is defined by the alignment, not measured.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It *is* the neural bin grid. Bin `k` of the neural array covers `[-0.5 + 0.02k, -0.5 + 0.02(k+1))` s from that trial's stimulus onset, and `BIN_TIMES[k]` is that bin's closing edge, so element `k` of the input and column `k` of the neural matrix describe the same 20 ms of the same trial. The same grid is used for the two behavioural outputs, so all four streams share one time axis.

ii.
```python
        tb = np.floor((st[b:e] - interval_begs[k]) / BINSIZE).astype(np.int64)   # neural
```
```python
        x = np.linspace(interval_begs[k] + BINSIZE, interval_ends[k], N_BINS)    # behaviour
```
```python
BIN_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)        # input
```

iii. N/A — by construction. The agent chose right edges rather than centres specifically so the input grid coincides with the sample points the reference code uses for behaviour.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. The trials table carries no block identifier, so blocks are recovered as maximal runs of constant `probabilityLeft`; a change of value starts a new block.

ii.
```python
    pleft = trials['probabilityLeft'].to_numpy(dtype=float)
    new_block = np.ones(len(pleft), dtype=bool)
    new_block[1:] = pleft[1:] != pleft[:-1]
    block_id = np.cumsum(new_block) - 1
```

iii. Not discussed separately in the trajectory; the agent's plan (step 52) simply lists "inputs = [time since stim onset (bin centres), trial number in block]". The construction follows directly from the task description the agent read ("in subsequent trials, it appears predominantly on one side in blocks"), with `probabilityLeft` the only column that marks them.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Within each block the trials are numbered `0, 1, 2, …` in order. Crucially the count is computed over the **full, unfiltered** trials table and only then subset to the kept trials (`trial_in_block[tidx]`), so a trial removed by the quality mask still advances the counter and the number reported is the animal's true position in the block. The per-trial scalar is broadcast across all 100 bins and stored as `float32`. (Verified on a converted session: a trial's input row 1 is a constant, e.g. 7.0, across all bins.)

ii.
```python
    trial_in_block = np.zeros(len(pleft), dtype=np.int64)
    for b in np.unique(block_id):
        idx = np.nonzero(block_id == b)[0]
        trial_in_block[idx] = np.arange(len(idx))
```
```python
    tidx = np.nonzero(valid)[0]
    ...
    tinb = trial_in_block[tidx]
    ...
    inputs[:, 1, :] = tinb[:, None].astype(np.float32)
```

iii. No explicit statement in the trajectory; the ordering of the code (block counting on the raw table, subsetting afterwards) is the decision.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which is `+1` (leftward wheel turn / left choice), `-1` (rightward) or `0` (no response). No-response trials have already been removed by the mask (`choice == 0` is one of its clauses), so only `±1` reach this point.

ii.
```python
    choice = trials['choice'].to_numpy()[tidx]          # +1 = left, -1 = right
```
```python
    query += ' | (choice == 0)'
```

iii. The agent verified the sign convention empirically rather than assuming it (steps 48–49): "Choice convention confirmed: choice=+1 corresponds to left stimulus (left choice), -1 to right", and again "Choice convention confirmed empirically: choice=+1 on correct left-contrast trials => 'left'."

## 5-b. What processing is involved in computing `output` *Choice*?

i. A single recoding to the mapping the instructions require, left = 0 and right = 1, done as `(choice < 0)`. The per-trial value is broadcast across all 100 bins (the instructions ask for time-varying outputs "if at all possible") and stored as `int64`. `output_values[0] = ['left', 'right']`.

ii.
```python
    out_choice = (choice < 0).astype(np.int64)          # left = 0, right = 1
...
    outputs = np.empty((n_trials, 4, N_BINS), dtype=np.int64)
    outputs[:, 0, :] = out_choice[:, None]
```
```python
    data['output_names'] = ['choice', 'prior', 'wheel_speed', 'whisker_motion_energy']
    data['output_values'] = [
        ['left', 'right'], ...
```

iii. The mapping is dictated by the decoder-task spec ("Choice, binary, per-trial, left = 0, right = 1"); the sign convention behind it was confirmed empirically as quoted in 5-a.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table — the block prior the task holds constant within a block. Trials with NaN `probabilityLeft` are already removed by the mask.

ii.
```python
    pleft = trials['probabilityLeft'].to_numpy(dtype=float)
    ...
    pleft_t = pleft[tidx]
```

iii. The agent checked the value set empirically: "pLeft in {0.2,0.5,0.8}" (step 49).

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. A recoding to the mapping the instructions require: `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`, applied through a dictionary after rounding to one decimal to guard against float representation. The per-trial value is broadcast across the 100 bins; `output_values[1] = ['p(left)=0.2', 'p(left)=0.5', 'p(left)=0.8']`. Note the code has no fallback for an unexpected `probabilityLeft`: a stray value would raise `KeyError` and the whole session would be recorded as skipped. In practice this never fired — the conversion log shows no exception skips.

ii.
```python
    prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
    out_prior = np.array([prior_map[round(float(p), 1)] for p in pleft_t], dtype=np.int64)
...
    outputs[:, 1, :] = out_prior[:, None]
```

iii. The mapping is dictated by the decoder-task spec; the agent's empirical check that pLeft only takes those three values is the justification for the hard lookup.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `SessionLoader.load_wheel()`, i.e. `_ibl_wheel.position` and `_ibl_wheel.timestamps`, from which the loader derives a velocity; speed is its absolute value, in rad/s. This is exactly the reference code's `'wheel-speed'`.

ii.
```python
    sl.load_wheel()
    wheel_speed_t = sl.wheel['times'].to_numpy()
    wheel_speed_v = np.abs(sl.wheel['velocity'].to_numpy())
```

iii. Follows `ibl_data_utils.load_anytime_behaviors`/`bin_behaviors`, which the agent read in full (steps 7–9) and where `'wheel-speed'` is `np.abs` of `SessionLoader`'s velocity.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three stages.
1. **Inside `SessionLoader.load_wheel()` (defaults, untouched):** the irregularly sampled wheel position is interpolated to a uniform 1000 Hz grid and differentiated into velocity, with an 8th-order 20 Hz Butterworth low-pass applied during differentiation.
2. **Resampling onto the trial grid:** for each trial the samples strictly inside the window are sliced (`searchsorted` with `side='right'`/`'left'`, matching the reference), then `scipy.interpolate.interp1d(..., kind='linear', fill_value='extrapolate')` is evaluated at the 100 bin right edges `linspace(beg + 0.02, end, 100)`. This is a verbatim re-implementation of `get_behavior_per_interval`, including its four skip conditions (see 1-e). Speed is computed for **all** trials and only afterwards subset to the kept ones.
3. **Discretisation into 3 classes** — see 7-c.

ii.
```python
def bin_behavior(target_times, target_vals, interval_begs):
    """Interpolate a continuous behavioural trace onto the trial bins. ..."""
    ib = np.searchsorted(target_times, interval_begs, side='right')
    ie = np.searchsorted(target_times, interval_ends, side='left')
    for k in range(n_trials):
        tt = target_times[ib[k]:ie[k]]
        tv = target_vals[ib[k]:ie[k]]
        ...
        x = np.linspace(interval_begs[k] + BINSIZE, interval_ends[k], N_BINS)
        vals[k] = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x)
        good[k] = True
    return vals, good
```
```python
    ws, ws_good = bin_behavior(wheel_speed_t, wheel_speed_v, interval_begs)
    ...
    ws = ws[tidx]
```

iii. Docstring: "Behavioural traces are interpolated onto the bin right-edges exactly as `get_behavior_per_interval` does, and trials whose traces don't cover the window are dropped." The 1000 Hz interpolation and Butterworth filter are not the agent's choices — they are `SessionLoader`'s defaults, which the reference code also relies on.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into three classes at the **tertiles of that session's own kept trials**, pooling all trials × all 100 bins: values below the 33.3rd percentile → 0 (`low`), between → 1 (`medium`), above the 66.7th → 2 (`high`). Duplicate quantiles are collapsed with `np.unique` first, so a session whose trace is degenerate simply yields fewer than three occupied classes rather than an error (the agent saw exactly one such session: "there's one wheel_speed class with fraction 0.000 in a session").

ii.
```python
def discretize3(x):
    """Discretize a continuous signal into 3 bins by its tertiles.

    The tertiles are computed per session: whisker motion energy is in camera-specific
    arbitrary units (left and right cameras differ in resolution and gain) and wheel
    speed also varies in scale across rigs, so a session-wise split is the only one that
    means the same thing (low / medium / high for this mouse in this session) everywhere.
    """
    q = np.quantile(x, [1. / 3., 2. / 3.])
    edges = np.unique(q)
    return np.searchsorted(edges, x, side='right').astype(np.int64)
```
```python
    out_ws = discretize3(ws)
    ...
    outputs[:, 2, :] = out_ws
```

iii. The docstring above is the justification. The agent also flagged the trade-off unprompted in its final summary: "I discretized wheel speed and whisker motion energy into 3 bins using per-session tertiles, on the grounds that whisker energy is in camera-specific arbitrary units … and wheel scale varies by rig. The cost is that this bakes a uniform 1/3-1/3-1/3 marginal into every session, so the chance level for those two outputs is exactly uniform by construction and no cross-session amplitude information survives; a global or fixed-threshold split would trade that away for comparability across sessions."

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is evaluated at the same 100 sample points, measured from the same `stimOn_times`, that define the neural bins — `linspace(stimOn - 0.5 + 0.02, stimOn + 1.5, 100)`, the right edge of each neural bin. Because the wheel timestamps are on the same session clock as the spikes, no further correction is needed. Trials where the wheel trace does not span the window (within one bin at each end) are dropped from *all* streams, so the per-trial lists stay in register.

ii.
```python
        x = np.linspace(interval_begs[k] + BINSIZE, interval_ends[k], N_BINS)
        vals[k] = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x)
```
```python
    valid = valid & ws_good & wm_good
    ...
    tidx = np.nonzero(valid)[0]
    binned = bin_spikes(spike_times, spike_clusters, n_neurons, interval_begs[tidx])
    ws = ws[tidx]
```

iii. "Behavioural traces are interpolated onto the bin right-edges exactly as `get_behavior_per_interval` does." The agent also verified it visually: "Alignment sanity-checked — wheel speed and whisker energy are low pre-stimulus and jump after onset" (it produced `sample_trials.png` for this).

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `SessionLoader.load_motion_energy(views=[view])`, which reads `<view>Camera.ROIMotionEnergy.npy` and `_ibl_<view>Camera.times.npy` and returns a DataFrame with columns `times` and `whiskerMotionEnergy` — IBL's mean absolute frame-to-frame difference over a box on the whisker pad. The **left** camera is tried first and the **right** used only if the left raises; a session with neither is dropped entirely.

ii.
```python
    whisker_t = whisker_v = None
    for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
        try:
            sl.load_motion_energy(views=[view])
            me = sl.motion_energy[cam]
            whisker_t = me['times'].to_numpy()
            whisker_v = me['whiskerMotionEnergy'].to_numpy()
            break
        except Exception:
            continue
    if whisker_t is None:
        return {'eid': eid, 'skip': 'no whisker motion energy'}
```

iii. Follows the reference code's `'whisker-motion-energy'` behaviour name and the method paper's description, which the agent had read in `methods.txt`: "Whisker motion energy is computed as the mean absolute difference between adjacent video frames within a bounding box anchored between nose tip and eye." 14 of 459 sessions were dropped by the fallback failing on both cameras.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is — no filtering, no normalisation, no rescaling. It goes through the **same** `bin_behavior` routine as the wheel: slice the window, reject if empty / contains NaN / starts >1 bin late / ends >1 bin early, then linear-interpolate onto the 100 bin right edges. Then it is discretised into three classes (8-c). As with the wheel, it is computed for all trials and subset afterwards.

ii.
```python
    wm, wm_good = bin_behavior(whisker_t, whisker_v, interval_begs)
    valid = valid & ws_good & wm_good
    ...
    wm = wm[tidx]
    ...
    out_wm = discretize3(wm)
    outputs[:, 3, :] = out_wm
```

iii. Same as 7-b — the agent re-implemented `get_behavior_per_interval` and applied it uniformly to both behavioural streams, which is what the reference code does (`bin_behaviors(one, eid, beh_names[3:], ...)` handles wheel speed and whisker motion energy together).

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: per-session tertiles over all kept trials × bins, via the same `discretize3`. `output_values[3] = ['low', 'medium', 'high']`.

ii.
```python
    out_wm = discretize3(wm)
```
```python
    q = np.quantile(x, [1. / 3., 2. / 3.])
    edges = np.unique(q)
    return np.searchsorted(edges, x, side='right').astype(np.int64)
```

iii. The `discretize3` docstring names whisker motion energy first as the reason for the per-session split: "whisker motion energy is in camera-specific arbitrary units (left and right cameras differ in resolution and gain) … so a session-wise split is the only one that means the same thing (low / medium / high for this mouse in this session) everywhere." The same caveat the agent raised in its summary (uniform marginal by construction) applies here.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same mechanism as the wheel: the camera frame times are on the same session clock as the spikes, so the trace is simply interpolated at the 100 bin right edges measured from that trial's `stimOn_times`. Trials whose camera coverage has a gap of more than one bin at either edge are dropped from every stream.

ii.
```python
        x = np.linspace(interval_begs[k] + BINSIZE, interval_ends[k], N_BINS)
        vals[k] = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x)
```
```python
    valid = valid & ws_good & wm_good
```

iii. Same as 7-d, including the visual sanity check that whisker energy is low pre-stimulus and rises after onset.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing data is dropped, at whichever level it is missing, and the reason is recorded.

**Per probe:** an insertion whose spike sorting is absent is skipped (`if sp is None or len(sp) == 0: continue`); the session continues with its other probe.

**Per trial:** NaNs in any of the six required trial events, and a non-finite `stimOn_times`, remove the trial via the mask; a wheel or camera trace that is absent, contains a NaN inside the window, or does not span the window to within one bin removes the trial via `ws_good`/`wm_good`.

**Per session:** no spike sorting at all → skipped; 0 well-isolated neurons → skipped; fewer than 2 trials passing the mask, or fewer than 2 with complete behaviour → skipped; no whisker motion energy on either camera → skipped. Every skip returns a `{'eid': ..., 'skip': <reason>}` record that is still pickled, printed, and counted in `metadata['n_sessions_skipped']`, so nothing disappears silently.

**Catch-all:** every session is wrapped in a `try/except` that converts any unexpected exception into a skip record carrying the last traceback line, so one bad session cannot abort the 459-session run. (In the actual run no session hit this path: 14 were dropped for missing whisker motion energy and 1 for having no trial with complete behaviour, leaving 444.)

Degenerate-but-not-missing cases are tolerated rather than dropped: a tertile split that collapses (constant trace) yields fewer than 3 classes instead of an error, and a trial in which a small population happens to emit no spikes is kept as an all-zero matrix (the verifier reported a handful of such warnings).

ii.
```python
def process_session(arg):
    eid, rows = arg
    try:
        return _process_session(eid, rows)
    except Exception:
        return {'eid': eid, 'skip': 'exception: ' + traceback.format_exc().splitlines()[-1]}
```
```python
        if sp is None or len(sp) == 0:
            continue
    if len(spikes_list) == 0:
        return {'eid': eid, 'skip': 'no spike sorting'}
```
```python
    if valid.sum() < MIN_TRIALS_PER_SESSION:
        return {'eid': eid, 'skip': f'only {int(valid.sum())} trials pass the mask'}
    ...
    if valid.sum() < MIN_TRIALS_PER_SESSION:
        return {'eid': eid, 'skip': f'only {int(valid.sum())} trials with complete behaviour'}
```
```python
        if res['skip']:
            n_skipped += 1
            continue
```

iii. The agent audited the skips rather than accepting the count: "444/459 sessions kept. 13 skipped for missing whisker motion energy (required by the decoder task), 1 for no trials with complete behaviour. Let me find the 15th", then "444 kept + 13 skipped = 457; 2 eids in the freeze had no session dir." On the zero-spike trials: "Format verifies with only one minor warning (one trial with no spikes)" and later "only benign 'all neural data is zero' warnings for a handful of trials in very small populations".

## 10-a. What are the most time-consuming steps of the code?

i. Reading the spike sorting off disk dominates: `spikes.times` and `spikes.clusters` run to hundreds of MB per probe, and the code makes this heavier than it needs to be in two ways — it leaves `SPIKES_ATTRIBUTES` at the ibllib default `['clusters', 'times', 'amps', 'depths']`, so it reads roughly twice the data it uses, and it leaves `check_hash` at its default `True`, so ONE re-reads each spike file to verify its md5. After that, the two per-trial Python loops (`bin_spikes`, `bin_behavior` ×2) are the next cost, followed by `np.isin(spike_clusters, keep_ids)` over the full merged spike train. Per-session setup (`get_one()` and `BrainRegions()`, both re-created on every job) adds a fixed overhead. Measured: ~2.3 s per session per worker; the whole 459-session run took ~300 s wall-clock with 16 workers, and writing/re-reading the 11.7 GB of intermediate and final pickles is a comparable share of that.

ii.
```python
        ssl = SpikeSortingLoader(pid=r.pid, one=one, eid=eid, pname=r.probe_name)
        sp, cl, ch = ssl.load_spike_sorting()          # no spike_attributes, no check_hash=False
```
```python
    sel = np.isin(spike_clusters, keep_ids)
```
```python
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=args.n_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(process_session, jobs)):
```

iii. The agent measured rather than guessed — its prototype printed `load 2.67` s for the spike-sorting call alone — and sized the run from it: "Timing is ~2.3 s per session per worker, so the full 459-session conversion should take only a few minutes." It chose parallelism plus per-session caching on that basis; it did not comment on the redundant spike attributes or the hash check.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four.
1. `bin_spikes`'s `for k in range(n_trials)` — could be one `bincount` over all trials by adding `trial * n_clusters * N_BINS` to the flat index. (The docstring already calls the function "vectorised", which is only true *within* a trial.)
2. `bin_behavior`'s `for k in range(n_trials)` — one `np.interp` call over a concatenated query vector would replace both the loop and the per-trial `interp1d` object construction, which is the more expensive part.
3. The block-numbering loop `for b in np.unique(block_id)` — `pandas.groupby(block).cumcount()`, or `arange(n) - maximum.accumulate(where(new_block, arange(n), 0))`, is a one-liner; as written it is O(n_blocks × n_trials).
4. In `main`, `subjects.index(res['subject'])` and `regions_all.index(r)` are linear scans inside a per-neuron loop — a dict would make them O(1). With 62,763 neurons over 263 regions this is the only quadratic term in the assembly.

None of these matter at this scale: the whole conversion is ~300 s and is I/O-bound.

ii.
```python
    for k in range(n_trials):
        b, e = beg[k], end[k]
        ...
        out[k] = np.bincount(flat, minlength=n_clusters * N_BINS).reshape(n_clusters, N_BINS)
```
```python
    for k in range(n_trials):
        tt = target_times[ib[k]:ie[k]]
        ...
        vals[k] = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x)
```
```python
    for b in np.unique(block_id):
        idx = np.nonzero(block_id == b)[0]
        trial_in_block[idx] = np.arange(len(idx))
```
```python
        data['brain_region_idx'].append(
            np.array([regions_all.index(r) for r in res['regions']], dtype=np.int64))
```

iii. The agent partially vectorised the spike binning deliberately, and documented it: "Equivalent to `ibl_data_utils.get_spike_data_per_interval` (`bincount2D` with `xlim=[t_beg, t_end]`, keeping the first `N_BINS` bins) but vectorised" — i.e. it replaced the reference's per-trial `bincount2D` + `intersect1d` with a single flat `bincount`. It gave no reasoning for leaving the outer loops, and did not discuss the assembly-time `list.index` scans.

## 10-c. What processing does the code repeat multiple times?

i. Four kinds of repetition.
1. **Per-session client construction.** `get_one()` and `BrainRegions()` are called inside `_process_session`, so a new ONE client and a fresh Allen/Beryl atlas table are built for each of the 459 jobs rather than once per worker. A pool `initializer` would do this 16 times instead.
2. **Intermediate pickles written then re-read.** Every session is pickled to `/app/work/sessions/<eid>.pkl` and then unpickled in the assembly pass, so the full ~11.7 GB is serialised twice and deserialised once. This is a deliberate trade for resumability (`jobs` skips eids whose pickle exists, and `--assemble-only` re-runs the second pass alone).
3. **Redundant reads within the loader.** `amps` and `depths` are read for every probe and never used; `merge_clusters(...).to_df()` builds the full metrics DataFrame when only `label` and `acronym` are consumed; `check_hash` re-reads each spike file to verify its md5.
4. **Repeated array rewrites.** Each trial is copied once more by `np.ascontiguousarray` when the stacked `(n_trials, ...)` arrays are split into per-trial lists, and `np.isin(spike_clusters, keep_ids)` sorts `keep_ids` for a membership test that a boolean lookup table (`keep[spike_clusters]`, which the code already has in `keep`) would answer directly.

ii.
```python
def _process_session(eid, rows):
    from brainbox.io.one import SpikeSortingLoader, SessionLoader
    from iblatlas.regions import BrainRegions

    one = get_one()
    br = BrainRegions()
```
```python
            for i, res in enumerate(pool.imap_unordered(process_session, jobs)):
                with open(os.path.join(args.tmp, res['eid'] + '.pkl'), 'wb') as f:
                    pickle.dump(res, f, protocol=4)
...
        with open(p, 'rb') as f:
            res = pickle.load(f)
```
```python
        data['neural'].append([np.ascontiguousarray(res['neural'][k]) for k in range(nt)])
```

iii. The resumability half is explicit in the design (the `jobs` filter and the `--assemble-only` flag), and the agent's summary describes the script as "parallel over sessions, resumable". The per-session `ONE`/`BrainRegions` construction and the unused spike attributes are not mentioned anywhere in the trajectory.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Five items, all small.
1. **Behavioural traces are interpolated for every trial in the session, then thrown away for the rejected ones** (`ws = ws[tidx]`, `wm = wm[tidx]`). Since ~1/3 of trials fail the mask on a typical session, roughly a third of the `interp1d` work is wasted. The ordering is forced, though: `ws_good`/`wm_good` are themselves part of the mask, so coverage must be tested before `tidx` is known — only the interpolation itself (not the coverage test) could be deferred.
2. **`spikes.amps` and `spikes.depths` are loaded and never touched**, roughly doubling the dominant I/O cost; `check_hash=True` re-reads the same files again.
3. **The full merged cluster DataFrame** (metrics, uuids, channels, depths, …) is materialised per probe when only `label` and `acronym` survive.
4. **Metadata that the decoder never reads**: `lab`, `date`, `n_probes`, `n_trials_total` per session, and the full 100-element `bin_times_s` vector, which duplicates input row 0 of every trial.
5. **Per-trial list copies**: the compact `(n_trials, n_neurons, 100)` array is split into 444 × ~425 separate `np.ascontiguousarray` views-turned-copies to satisfy the target format, and `decoder.py` immediately re-concatenates them per session.

Nothing here changes the output; the total cost is a modest constant factor on a run that already finishes in ~5 minutes.

ii.
```python
    ws, ws_good = bin_behavior(wheel_speed_t, wheel_speed_v, interval_begs)   # all trials
    wm, wm_good = bin_behavior(whisker_t, whisker_v, interval_begs)
    ...
    ws = ws[tidx]
    wm = wm[tidx]
```
```python
        cld = SpikeSortingLoader.merge_clusters(sp, cl, ch).to_df()
```
```python
        'bin_times_s': [float(t) for t in BIN_TIMES],
        ...
        session_info.append({'eid': eid, 'subject': res['subject'], 'lab': res['lab'],
                             'date': res['date'], 'n_probes': res['n_probes'], ...})
```

iii. Not discussed in the trajectory. The extra metadata is plainly intentional (the instructions invite "other relevant fields, e.g. `session_info`"), and the agent did re-serialise the final pickle once to make the metadata fully JSON-safe. The wasted interpolation and the unused spike attributes appear to be oversights rather than choices.
