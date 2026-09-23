# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything is read through the IBL ONE API against the locally staged cache at
`/app/data/one_cache`, never by opening ALF files directly. The session list comes from the
reference repository's own release freeze, `/app/code/code_zhang2025/data/bwm_release.csv`
(699 insertions / 459 sessions / 139 subjects / 12 labs); the unique `eid`s of that table are
the work list. For each `eid`, `SessionLoader` supplies the trials table (via the reference
`load_trials_and_mask`, which is given the same loader), the wheel, and the camera motion
energy; `SpikeSortingLoader` is instantiated once per probe insertion, taking `pid` and
`probe_name` from the same CSV rather than from `one.eid2pid` (which needs a live Alyx
connection). Sessions are processed in a `multiprocessing` spawn Pool (24 workers), each
worker building its own `ONE` instance, and the results are re-ordered back into
`bwm_release.csv` order afterwards.

A practical prerequisite: the cache tables shipped with the dataset list dataset paths
*without* the revision folders that are actually on disk, so ONE in local mode cannot resolve
trials/wheel/motion-energy. The agent rebuilt the index from the filesystem with a separate
script (`/app/cache/build_one_cache.py`, using `one.alf.cache.make_parquet_db` per lab and
re-mapping the hashed session UUIDs back to the true eids via `(lab, subject, date, number)`),
and `convert_data.py` points ONE at those rebuilt tables.

ii.
```python
BWM_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'
ONE_KWARGS = dict(cache_dir='/app/data/one_cache',
                  tables_dir='/app/cache/one_tables',
                  mode='local', silent=True)

def get_one():
    global _ONE
    if _ONE is None:
        from one.api import ONE
        _ONE = ONE(**ONE_KWARGS)
    return _ONE
```

```python
bwm = pd.read_csv(BWM_CSV, index_col=0)
eids = sorted(bwm.eid.unique())
...
tasks = [(eid, bwm[bwm.eid == eid][['pid', 'probe_name']].copy(),
          i < n_show, out_dir, not args.no_zscore) for i, eid in enumerate(eids)]
```

```python
sess_loader = SessionLoader(one=one, eid=eid)
trials, mask = load_trials_and_mask(one=one, eid=eid, max_trial_len=MAX_TRIAL_LEN,
                                    sess_loader=sess_loader)
...
ssl = SpikeSortingLoader(pid=row.pid, one=one, eid=eid, pname=row.probe_name)
spikes, clusters, channels = ssl.load_spike_sorting()
```

iii. From CONVERSION_NOTES Step 1/Step 4: "`one.eid2pid` needs a network connection: replaced
with the `pid`/`probe_name` columns of `bwm_release.csv`" — and "`bwm_release.csv` is the same
freeze the reference loads and contains `pid`/`probe_name` columns", so the probe list is
identical to what the reference pipeline would obtain. The rebuilt ONE index is justified as
the only way to make offline loading work: "the cache tables shipped in
`/app/data/one_cache/...` list dataset paths WITHOUT the revision folders actually staged on
disk ... consequently `SessionLoader.load_trials()` and `load_motion_energy()` raise
`ALFObjectNotFound`". The agent verified the rebuilt tables contain all 459 BWM-release eids,
and separately reproduced the published 459 sessions / 699 insertions / 139 subjects / 12 labs /
621,733 units / 75,708 well-isolated neurons from them. Processes rather than threads were used
because "a shared ONE instance is not thread-safe: 76 of 459 sessions silently failed to load
under `ThreadPoolExecutor`".

## 1-b. How are the data split into subjects?

i. No parsing is done: the subject name is read off the `subject` column of
`bwm_release.csv`, one value per `eid`. At assembly the subject list is the sorted unique set
of subject names over the sessions that survived conversion, and `subject_idx` is each
session's index into that list. Result: 136 subjects over 445 sessions (3 of the 139 released
subjects are lost because all of their sessions lack whisker motion energy).

ii.
```python
eid2subject = bwm.drop_duplicates('eid').set_index('eid')['subject'].to_dict()
subjects = sorted({eid2subject[r['eid']] for r in ok})
subject_index = {s: i for i, s in enumerate(subjects)}
...
'subject_idx': np.array([subject_index[eid2subject[r['eid']]] for r in ok], dtype=np.int64),
```

iii. The release table already carries a unique subject id per session, so nothing has to be
derived. CONVERSION_NOTES Step 9 records and explains the 136-vs-139 difference: "3 subjects
only had sessions without whisker ME".

## 1-c. How are the data split into sessions?

i. A session *is* an `eid`; the release table has one row per (session, probe insertion), so
`bwm.eid.unique()` gives the 459 sessions directly and nothing has to be split. Sessions with
two insertions are handled by merging the probes into one population (see 2-b), so one
converted session = one `eid`, never one probe.

ii.
```python
eids = sorted(bwm.eid.unique())
...
print('bwm_release.csv: %d insertions, %d sessions, %d subjects, %d labs'
      % (len(bwm), bwm.eid.nunique(), bwm.subject.nunique(), bwm.lab.nunique()))
```

iii. No decision to make — the release freeze is organised by session. The agent's Step 4
consistency table confirms "bwm_release.csv lists 459 sessions / 459 sessions staged, all
loadable", matching the data paper's 459.

## 1-d. How are the data split into trials?

i. The ALF trials table has one row per trial, so the split is given by the data. The agent
takes the trials table exactly as the reference `load_trials_and_mask` returns it, and a
"trial" in the converted data is one row of that table (after curation) turned into a 2 s
window around its `stimOn_times`.

ii.
```python
trials, mask = load_trials_and_mask(one=one, eid=eid, max_trial_len=MAX_TRIAL_LEN,
                                    sess_loader=sess_loader)
n_trials_raw = len(trials)
...
align_times = trials[ALIGN_EVENT].values[mask]
```

iii. Nothing to decide. The agent sanity-checked the raw split against the data paper:
"Trials/session mean 645 / median 602 / range 401–1,525 (paper)" vs "645.08 / 601 /
401–1,525 (measured)".

## 1-e. How are trials filtered based on quality controls?

i. Two stages, applied together.

**(a) The reference mask, used verbatim.** Rather than re-implement it, `convert_data.py`
imports `utils.ibl_data_utils.load_trials_and_mask` from the reference repository and calls it
with `max_trial_len=10.0`, the same argument `prepare_data` uses. That mask drops trials with
reaction time (`firstMovement_times - stimOn_times`) outside [0.08, 2.0] s; trials with NaN in
any of `stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times,
feedbackType`; trials with `feedback_times - goCue_times > 10 s`; and no-response trials
(`choice == 0`). `exclude_unbiased` is left at its default `False`, so the 90-trial unbiased
`pL = 0.5` block is kept — required, because the prior output has three classes.

**(b) Behavioural coverage.** A trial is additionally dropped if its 2 s window is not covered
by the wheel and the camera streams, or if either interpolated trace contains a non-finite
value.

A session is dropped entirely if fewer than 2 trials survive (none did), if it has no
well-isolated grey-matter neuron (none did), or if it has no whisker motion energy from either
camera (14 sessions). Result: 189,057 trials over 445 sessions, mean 424.8 per session.

ii.
```python
MAX_TRIAL_LEN = 10.0           # reference `load_trials_and_mask(..., max_trial_len=10.0)`
...
trials, mask = load_trials_and_mask(one=one, eid=eid, max_trial_len=MAX_TRIAL_LEN,
                                    sess_loader=sess_loader)
mask = np.asarray(mask, dtype=bool)
if mask.sum() < MIN_TRIALS_PER_SESSION:
    return dict(eid=eid, skip='fewer than %d trials pass curation' % MIN_TRIALS_PER_SESSION, ...)
```

```python
grid = align_times[:, None] + BIN_RIGHT_EDGES[None, :]
valid = (grid[:, 0] >= beh_times[0] - BIN_SIZE) & (grid[:, -1] <= beh_times[-1] + BIN_SIZE)
...
valid &= np.isfinite(vals).all(axis=1)
return vals, valid
```

```python
keep = valid_wheel & valid_me
if keep.sum() < MIN_TRIALS_PER_SESSION:
    return dict(eid=eid, skip='fewer than %d trials with complete behaviour' % MIN_TRIALS_PER_SESSION, ...)
```

iii. From Step 4/Step 10: "USE the reference code mask verbatim (it is a superset of the paper
criteria and is what the decoding paper actually ran)" — and the mask is obtained by importing
the reference function itself "so trial curation is guaranteed identical to the reference
pipeline rather than re-implemented." The coverage rule is justified as mirroring the reference
`get_behavior_per_interval` checks ("target data starts too late" / "ends too early" / "nans in
target data"): "Trials at the very start/end of a recording where the wheel or camera stream
does not cover the whole 2 s window are dropped ... rather than being silently extrapolated."
Keeping the unbiased block is justified because "the decoder must distinguish pL = 0.5 from
0.2 and 0.8".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times` and `spikes.clusters` of each probe insertion of the session. The merged
cluster table (`SpikeSortingLoader.merge_clusters(spikes, clusters, channels)`) supplies two
further fields that are used only for curation and labelling, never for the values themselves:
`label` (the IBL single-unit QC score) and `acronym` (the Allen location of the cluster's peak
channel, mapped to Beryl for `brain_regions`).

ii.
```python
ssl = SpikeSortingLoader(pid=row.pid, one=one, eid=eid, pname=row.probe_name)
spikes, clusters, channels = ssl.load_spike_sorting()
if spikes is None or len(spikes) == 0 or 'times' not in spikes:
    continue
clu = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
...
times_list.append(np.asarray(spikes['times'])[sel])
clusters_list.append(remap[np.asarray(spikes['clusters'])[sel]])
```

iii. These are the same calls the reference `load_spiking_data` makes; the agent's Step 10
Check 3 table records "(a) Data loading ... Identical calls". The one reference call it does
not make is `spike_loader.raw_electrophysiology(band='ap', stream=True).fs`, skipped because it
"needs the raw AP binaries, which are not staged; it only supplies a metadata field".

## 2-b. How is the `neural` data processed?

i. Four steps.
1. Per probe, quality-filter the clusters (2-c), renumber the survivors 0..n-1 continuing the
   count from the previous probe, concatenate, and sort the merged spike train by time — the
   `merge_probes` behaviour, so a two-probe session yields one pooled population.
2. Count spikes into 100 non-overlapping 20 ms bins over `[stimOn - 0.5, stimOn + 1.5)` for
   each surviving trial. Implemented as one `np.searchsorted` for all trial boundaries plus a
   per-trial `np.add.at`, which the agent verified bit-identical to the reference
   `bincount2D` path.
3. **Standardise**: for each of the 100 time bins, a *single scalar* mean and std pooled over
   all neurons and all trials of the session is subtracted/divided — i.e. exactly what the
   reference decoding loader `data_loader_utils.standardize_spike_data` does. Statistics are
   accumulated in float64 and the result cast to float32.
4. Store as one `(n_neurons, 100)` float32 array per trial.

Note: the stored `neural` is therefore a standardised quantity, **not** a firing rate in Hz and
not raw counts. Also note that `metadata['neural_data_type']`, the Step 5 plan text and the
`--show-processing` plot title all still say "z-scored **per neuron**", which is not what the
shipped code does (verified directly on `sample_data.pkl`: per-*bin* mean ≈ 3e-5 and std ≈
1.0000, while per-*neuron* means range −0.52 to +2.85). The correct description is the one in
the Step 12 write-up and the inline comment.

ii.
```python
remap = np.full(len(clu), -1, dtype=np.int64)
remap[keep_idx] = np.arange(keep_idx.size) + n_offset
...
order = np.argsort(spike_times, kind='stable')   # `merge_probes` sorts by spike time
```

```python
def bin_spikes(spike_times, spike_clusters, n_neurons, align_times):
    begs = align_times + TIME_WINDOW[0]
    ends = align_times + TIME_WINDOW[1]
    i0 = np.searchsorted(spike_times, begs, side='left')
    i1 = np.searchsorted(spike_times, ends, side='left')
    out = np.zeros((n_trials, n_neurons, N_BINS), dtype=np.float32)
    for k in range(n_trials):
        a, b = i0[k], i1[k]
        bin_idx = ((spike_times[a:b] - begs[k]) / BIN_SIZE).astype(np.int64)
        np.clip(bin_idx, 0, N_BINS - 1, out=bin_idx)
        np.add.at(out[k], (spike_clusters[a:b], bin_idx), 1.0)
    return out
```

```python
# Reference `data_loader_utils.standardize_spike_data`: for each time bin it takes a
# SINGLE scalar mean and std pooled over all neurons and trials ...
mu = spikes_binned.mean(axis=(0, 1), keepdims=True, dtype=np.float64)
sd = spikes_binned.std(axis=(0, 1), keepdims=True, dtype=np.float64)
neural = ((spikes_binned.astype(np.float64) - mu)
          / np.where(sd > 0, sd, 1.0)).astype(np.float32)
```

iii. Binning parameters are the reference `params` (`binsize: 0.02`, `time_window: (-.5, 1.5)`).
Probe merging is justified by the BWM paper ("neurons in the same session and region were
combined across probes for our decoding analysis") and by `merge_probes`. The standardisation
is justified twice over in Step 4/Step 12: (a) "the provided `/app/decoder.py` does NO
normalisation (it runs `torch.svd` straight on the stored values), so normalisation has to
happen in the conversion", and (b) the agent first implemented a per-neuron z-score, then
re-read `standardize_spike_data` line by line, found the reference pools per time bin, and
**measured** all three variants on 40 sessions (per-neuron 0.5987/0.6401 choice/prior; raw
counts 0.6202/0.6753; reference transform 0.6196/0.6697), adopting the reference transform.
A follow-up check found float32 accumulation error (max diff 1.59e-3 vs a 1.9e-6 round-off
bound) and moved the statistics to float64, reducing it to 4.8e-7.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two cuts, applied per probe before any spike is kept:
- `label >= 1`. `label` is the mean of the three RIGOR single-unit metrics (amplitude > 50 µV,
  noise cut-off < 20 µV, no refractory-period violation), so `>= 1` means all three passed —
  the BWM paper's "well-isolated neuron". The agent confirmed this reproduces the published
  621,733 → 75,708 exactly over the full release.
- Beryl acronym not in `{root, void}`, i.e. grey matter only. `void` is outside the brain;
  `root` is a site the atlas could not assign to a summary structure. This is an extra cut
  relative to the reference caching code, which keeps every sorted cluster.

Result: 62,779 neurons over the 445 kept sessions (mean 141.1, median 123, range 1–516), out of
73,060 label≥1 clusters in those sessions.

ii.
```python
NON_GREY = ('root', 'void')    # Beryl acronyms that are not grey matter
...
# IBL single-unit QC: label is the mean of the three RIGOR criteria, so
# label >= 1 means all three passed -> "well-isolated neuron" in the BWM paper.
good = (clu['label'].values >= 1)
beryl = np.asarray(br.acronym2acronym(clu['acronym'].to_numpy(), mapping='Beryl'))
keep = good & ~np.isin(beryl, NON_GREY)
keep_idx = np.flatnonzero(keep)
```

iii. Step 4 records the discrepancy explicitly and resolves it against the paper:
"`load_spiking_data(qc=None)` keeps ALL clusters ... APPLY the QC filter (`label >= 1`). The
paper headline neuron count (75,708) is the published statistic we are asked to match, and MUA
clusters add noise rather than signal. The reference caching code keeps all clusters only
because it defers the decision downstream." For grey matter: "BWM paper: *Final analyses were
additionally restricted to regions that were designated grey matter* ... DROP root/void units.
They are not assigned to any anatomical region, so they cannot be given a meaningful
`brain_regions` label in the target format." The BWM region criterion (≥5 neurons/session,
≥2 sessions) was deliberately **not** applied: "our decoder pools all neurons of a session, so
discarding neurons from sparsely sampled regions would only throw information away."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All IBL streams (spike times, trial events, wheel timestamps, camera frame times) are
already on one session clock in seconds, so alignment is a subtraction: each trial's window is
`stimOn_times + (-0.5, +1.5)` and the bin index of a spike is computed relative to that trial's
window start. `t = 0` therefore falls at the boundary between bin 24 and bin 25, i.e. bin 25 is
the first post-stimulus bin.

ii.
```python
ALIGN_EVENT = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)      # seconds relative to the alignment event
...
align_times = trials[ALIGN_EVENT].values[mask]
...
begs = align_times + TIME_WINDOW[0]
bin_idx = ((spike_times[a:b] - begs[k]) / BIN_SIZE).astype(np.int64)
```

iii. Step 3/Step 4: the reference `params` are `align_time='stimOn_times'`,
`time_window=(-.5, 1.5)`, and the methods paper says "For choice, we align trials to the
stimulus onset, considering neural activity from 0.5 s before to 1.5 s post-onset"; the Decoder
Task also specifies stimulus-onset alignment. The agent noted one deliberate difference: "the
methods paper aligns wheel speed / whisker ME to FIRST MOVEMENT onset ... USE STIMULUS ONSET
FOR EVERYTHING, because the Decoder Task requires a single alignment event and all four outputs
must share one time base." Alignment was then verified empirically: a per-bin cross-validated
choice decode is at chance (0.503–0.522) in every pre-stimulus bin and jumps to 0.696 at bin 30
and 0.812 at +0.3 s — "a misalignment would smear or shift this transition".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins per trial, identical for every trial of every session. No rebinning,
resampling, smoothing or overlap is applied to the neural data — spikes are counted once,
directly into the final grid. `metadata['time_bin_size'] = 20.0` (ms).

ii.
```python
BIN_SIZE = 0.02                # seconds
N_BINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BIN_SIZE))   # = 100
BIN_CENTRES = TIME_WINDOW[0] + BIN_SIZE * (np.arange(N_BINS) + 0.5)
BIN_RIGHT_EDGES = TIME_WINDOW[0] + BIN_SIZE * (np.arange(N_BINS) + 1.0)
STIM_ONSET_BIN = int(np.floor((0.0 - TIME_WINDOW[0]) / BIN_SIZE))
```

iii. The reference caching params set `'binsize': 0.02` and the methods paper main text says
trials are "each divided into 20-ms bins, producing T = 100 time steps". The agent found and
resolved a conflict: "its STAR Methods says 50-ms non-overlapping time bins for choice/prior
... USE 20 ms, T = 100, following the runnable reference code and the main text. The STAR-
Methods 50 ms sentence is internally inconsistent with the same paper's T = 100 over a 2 s
trial. 20 ms is also needed to resolve the time-varying wheel/whisker outputs."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. From `trials.stimOn_times`, the event every trial is aligned to, combined with the fixed
`(-0.5, 1.5)` / 20 ms grid. The value stored is the bin-centre time, so the same 100 numbers
(−0.49 … +1.49) serve every trial of every session. The agent also emits a second, related
input, `stim_onset`, a binary indicator that is 1 in bin 25 (the bin containing `t = 0`) and 0
elsewhere.

ii.
```python
INPUT_NAMES = ['time_from_stim_onset', 'stim_onset', 'trial_num_in_block']
BIN_CENTRES = TIME_WINDOW[0] + BIN_SIZE * (np.arange(N_BINS) + 0.5)
STIM_ONSET_BIN = int(np.floor((0.0 - TIME_WINDOW[0]) / BIN_SIZE))
```

iii. "Alignment event: `trials.stimOn_times` (stimulus onset). Window: off_start = −0.5 s,
off_end = +1.5 s ... input[0] = bin centre time in seconds relative to stimOn". The extra
binary channel is justified by an explicit instruction: "Decoder-Task rule: if an input is a
time such as onset of some stimulus, represent it as a binary time series."

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. Essentially none — the variable is defined by the binning grid, not measured. The bin
centres are computed once at module level and broadcast into every trial's input array as
float32. The `stim_onset` channel is likewise a constant 100-element vector with a single 1 at
index 25, identical in every trial and every session (so it carries no information the decoder
can use to discriminate between trials).

ii.
```python
stim_indicator = np.zeros(N_BINS, dtype=np.float32)
stim_indicator[STIM_ONSET_BIN] = 1.0
time_axis = BIN_CENTRES.astype(np.float32)
inputs = []
for k in range(n_trials):
    arr = np.empty((3, N_BINS), dtype=np.float32)
    arr[0] = time_axis
    arr[1] = stim_indicator
    arr[2] = tnb[k]
    inputs.append(arr)
```

iii. N/A — defined by us. The agent verified the result: "`time_from_stim_onset` is
`np.allclose` to the bin centres −0.49 … +1.49" and "`stim_onset` sums to exactly 1 per trial
with its 1 at bin 25 (the bin containing t = 0)", checked on 4 sessions without using any
conversion code.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It *is* the neural grid. The spikes are binned on `[stimOn - 0.5 + 0.02k, ... + 0.02(k+1))`
and the input is the centre of that same bin `k`, so index-for-index the two describe the same
20 ms of the same trial. No interpolation or offset is involved.

ii.
```python
# bin centres relative to the alignment event: -0.49, -0.47, ..., +1.49
BIN_CENTRES = TIME_WINDOW[0] + BIN_SIZE * (np.arange(N_BINS) + 0.5)
...
bin_idx = ((spike_times[a:b] - begs[k]) / BIN_SIZE).astype(np.int64)
```

iii. "Bin k covers `[stimOn − 0.5 + 0.02k, stimOn − 0.5 + 0.02(k+1))`", so `t = 0` falls in bin
25 and the last bin ends exactly at +1.5 s; this was listed as an explicit off-by-one edge case
in Step 10 Check 5 and verified on every trial of every session.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `trials.probabilityLeft`. The trials table carries no block identifier, so blocks are
recovered from the fact that `probabilityLeft` is constant within a block: a change of value
starts a new block.

ii.
```python
def trial_number_in_block(prob_left):
    pl = np.asarray(prob_left, dtype=float)
    new_block = np.ones(len(pl), dtype=bool)
    new_block[1:] = pl[1:] != pl[:-1]
    ...
```

iii. "input[2] = trial_in_block | 0-based count of trials since the last change of
`probabilityLeft`". The agent cross-checked the block structure against the paper: the first
block is `pL = 0.5` for 90 trials in 459/459 sessions, and biased blocks are 20–100 trials, so
the observed range `[0, 98]` is as expected.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A 0-based running count of the trial's position within its block, computed as
`arange - block_start` via a `np.maximum.accumulate` over the indices where the block changes.
Crucially it is computed on the **unmasked** trials table, before any quality-control trial is
dropped, so a dropped trial still advances the counter and the stored number is the animal's
true position in the block. The per-trial scalar is then broadcast across all 100 bins.

ii.
```python
def trial_number_in_block(prob_left):
    """0-based index of each trial within its block of constant `probabilityLeft`.

    Computed on the *unmasked* trials table, because the position of a trial within its
    block is a property of the experiment and must not change when trials are dropped.
    """
    pl = np.asarray(prob_left, dtype=float)
    new_block = np.ones(len(pl), dtype=bool)
    new_block[1:] = pl[1:] != pl[:-1]
    block_start = np.maximum.accumulate(np.where(new_block, np.arange(len(pl)), 0))
    return np.arange(len(pl)) - block_start
```

```python
# `trial_in_block` must be computed before the mask is applied
tnb_all = trial_number_in_block(trials['probabilityLeft'].values)
...
tnb = tnb_all[mask]
```

iii. Key decision 9: "`trial_in_block` computed on the unmasked trials table: the true trial
index within the block is a property of the experiment, so it must be counted before trials are
dropped." Verified in Step 10 Check 2 against "an independent re-computation of block
boundaries from `probabilityLeft`".

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. One column, `trials.choice`, which is +1, −1 or 0. `+1` is remapped to 0 (left) and `−1` to
1 (right); `choice == 0` (no response) trials never reach this point because the reference mask
removes them (`exclude_nochoice=True`).

ii.
```python
# choice: ALF +1 == mouse reported LEFT, -1 == reported RIGHT (verified in Step 3)
choice = np.where(choice_raw > 0, 0, 1).astype(np.int8)
```

iii. The sign convention is not stated in the provided text, so the agent established it
empirically over all 459 sessions: "On every one of the 459 sessions, 100% of CORRECT trials
with the stimulus on the LEFT have `choice == +1`, and 100% of correct trials with the stimulus
on the RIGHT have `choice == −1`. Independently, in right-biased blocks (pL = 0.2) 75.6% of
choices are −1, while in left-biased blocks (pL = 0.8) only 25.5% are −1." It also flagged the
trap: "the wheel turn direction is the opposite of the reported side, because the mouse moves
the stimulus toward the centre; the ALF `choice` field encodes the reported side." The Decoder
Task specifies left = 0, right = 1.

## 5-b. What processing is involved in computing `output` *Choice*?

i. None beyond the +1/−1 → 0/1 recode and the broadcast of the per-trial scalar across all 100
bins (the target format prefers time-varying outputs, and the decoder broadcasts per-trial
values anyway). Stored as int8.

ii.
```python
outputs = []
for k in range(n_trials):
    arr = np.empty((4, N_BINS), dtype=np.int8)
    arr[0] = choice[k]
    arr[1] = prior[k]
    arr[2] = wheel_lab[k]
    arr[3] = me_lab[k]
    outputs.append(arr)
```

iii. N/A. Verified: "`choice` equals `where(trials.choice > 0, 0, 1)` element-by-element" and
"`choice` and `prior` are constant within each trial"; the converted class fractions are
[0.508, 0.492], i.e. near balanced as expected.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. One column, `trials.probabilityLeft`, which takes the three values 0.2, 0.5 and 0.8,
remapped to 0, 1 and 2 exactly as the Decoder Task specifies.

ii.
```python
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
prior = np.array([prior_map[round(float(p), 4)] for p in prob_left_raw], dtype=np.int8)
```

iii. "output[1] = prior_prob_left | 0.2 → 0, 0.5 → 1, 0.8 → 2 | exactly as specified in the
Decoder Task." The unbiased `pL = 0.5` block is deliberately kept: "KEEP them, required because
prior has three classes 0.2/0.5/0.8 in the Decoder Output spec."

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. None beyond the recode (with a `round(..., 4)` guard against float representation) and the
broadcast across the 100 bins. The resulting distribution is [0.418, 0.141, 0.442], i.e. about
14% of trials in the unbiased block — consistent with 90 unbiased trials out of ~645 per
session.

ii.
```python
prior = np.array([prior_map[round(float(p), 4)] for p in prob_left_raw], dtype=np.int8)
...
arr[1] = prior[k]
```

iii. Step 9 consistency table: "prior_prob_left distribution | 90 unbiased trials of about 645
→ about 0.14 at pL = 0.5 | [0.418, 0.141, 0.442] | yes." Verified independently: "`prior_prob_left`
equals the 0.2/0.5/0.8 → 0/1/2 map of `trials.probabilityLeft`."

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `_ibl_wheel.position` and `_ibl_wheel.timestamps`, loaded through
`SessionLoader.load_wheel()`, which returns an evenly-sampled `times`/`position`/`velocity`
table. Wheel speed is `np.abs(velocity)`.

ii.
```python
sess_loader.load_wheel()
wheel_speed_raw, valid_wheel = interp_behavior(
    sess_loader.wheel['times'].to_numpy(),
    np.abs(sess_loader.wheel['velocity'].to_numpy()), align_times)
```

iii. Step 10 Check 3: "(f) wheel speed | reference: `abs(SessionLoader.wheel['velocity'])` |
mine: Identical". Step 3: "Wheel speed: absolute wheel velocity, where velocity is the
smoothed derivative of wheel position computed by `SessionLoader.load_wheel()`."

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three stages. (1) Inside `SessionLoader.load_wheel`, the irregularly sampled wheel position
is interpolated onto a regular 1 kHz grid and differentiated into a velocity with a 20 Hz
Butterworth low pass — the IBL default, not a choice made here. (2) `np.abs` gives speed in
rad/s. (3) The trace is linearly interpolated (`scipy.interpolate.interp1d`, one vectorised
call for the whole `(n_trials, 100)` grid) onto the **right edge** of each of the 100 bins of
each trial, i.e. `np.linspace(t_beg + binsize, t_end, 100)`, which is the grid the reference
`get_behavior_per_interval` uses. Non-finite samples are removed and the trace is sorted before
interpolation; trials outside the stream's time range are marked invalid rather than
extrapolated. Discretisation is then applied (7-c).

ii.
```python
def interp_behavior(beh_times, beh_values, align_times):
    finite = np.isfinite(beh_times) & np.isfinite(beh_values)
    beh_times, beh_values = beh_times[finite], beh_values[finite]
    order = np.argsort(beh_times, kind='stable')
    beh_times, beh_values = beh_times[order], beh_values[order]

    grid = align_times[:, None] + BIN_RIGHT_EDGES[None, :]
    valid = (grid[:, 0] >= beh_times[0] - BIN_SIZE) & (grid[:, -1] <= beh_times[-1] + BIN_SIZE)
    vals = np.full(grid.shape, np.nan, dtype=np.float64)
    if valid.any():
        f = interp1d(beh_times, beh_values, kind='linear', bounds_error=False,
                     fill_value=(beh_values[0], beh_values[-1]))
        vals[valid] = f(grid[valid])
    valid &= np.isfinite(vals).all(axis=1)
    return vals, valid
```

iii. The velocity computation is the IBL default, so it is identical to what the reference
obtains. The interpolation grid is deliberately the reference's: "right edge of each bin, used
for interpolating the continuous behaviours. This matches the reference
`get_behavior_per_interval`, which evaluates the behaviour on
`np.linspace(t_beg + binsize, t_end, n_bins)`." Step 10 Check 3 records this stage as
"Identical grid (`BIN_RIGHT_EDGES`) and identical validity rules".

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Per-session terciles. The 33.3rd and 66.7th percentiles are computed over **all** (trial,
bin) samples of that session that survived curation, and `np.searchsorted(..., side='right')`
assigns each sample to class 0/1/2 = low/medium/high. Duplicate edges (from a zero-inflated
trace) are nudged apart with `np.nextafter` so three classes remain available. The resulting
distribution is [0.3333, 0.3333, 0.3333] by construction.

ii.
```python
def discretize(values, n_bins=N_DISCRETE_BINS):
    """Discretise a continuous (n_trials, N_BINS) signal into `n_bins` equal-count classes.

    Thresholds are the per-session quantiles of all (trial, bin) samples, so the classes
    are balanced within each session and invariant to the arbitrary per-session units ...
    """
    flat = values.ravel()
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.quantile(flat, qs)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    labels = np.searchsorted(edges, flat, side='right').astype(np.int8)
    return labels.reshape(values.shape), edges
```

iii. "Both wheel speed and whisker motion energy are continuous, strongly right-skewed, and in
SESSION-SPECIFIC ARBITRARY UNITS ... Fixed global thresholds would therefore put nearly all
trials of some sessions into a single class. Decision: PER-SESSION TERCILE binning ...
guarantees three roughly balanced classes in every session (about 1/3 each), so the balanced
accuracy of the decoder is interpretable and chance is 1/3; is invariant to the arbitrary
per-session scale; preserves the ordering." Verified: the stored edges equal an independent
recomputation, the labels match an independent `np.searchsorted`, and the classes are ordered
(`max(value | class 0) <= min(value | class 1) <= ...`).

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Both are cut out of the session on the same clock relative to the same `stimOn_times`, and
the wheel value stored in bin `k` is the trace evaluated at the **right edge** of neural bin
`k` (i.e. `stimOn − 0.5 + 0.02(k+1)`), which is the reference pipeline's convention. So output
index `k` corresponds to neural bin `k`, with the behaviour sampled at the end rather than the
centre of that 20 ms — a 10 ms offset relative to the `time_from_stim_onset` input.

ii.
```python
BIN_RIGHT_EDGES = TIME_WINDOW[0] + BIN_SIZE * (np.arange(N_BINS) + 1.0)
...
grid = align_times[:, None] + BIN_RIGHT_EDGES[None, :]
```

iii. "This matches the reference `get_behavior_per_interval`, which evaluates the behaviour on
`np.linspace(t_beg + binsize, t_end, n_bins)`." The `--show-processing` figure overlays the raw
wheel-velocity trace of trial 0 with the interpolated bin values and the tercile thresholds
specifically to make this visible: "the interpolated points sit exactly on the source traces,
and the discretised classes change exactly when the continuous signal crosses a threshold."

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `<view>Camera.ROIMotionEnergy` with its frame times `_ibl_<view>Camera.times`, loaded via
`SessionLoader.load_motion_energy(views=[view])` and read from the `whiskerMotionEnergy`
column. The left camera (60 Hz) is preferred and the right camera (150 Hz) is used as a
fallback; a session with neither is dropped (14 sessions).

ii.
```python
# Reference `bin_behaviors`: prefer the left camera, fall back to the right one.
me_raw, valid_me, me_view = None, None, None
for view in ('left', 'right'):
    try:
        sess_loader.load_motion_energy(views=[view])
        me = sess_loader.motion_energy[view + 'Camera']
        me_raw, valid_me = interp_behavior(me['times'].to_numpy(),
                                           me['whiskerMotionEnergy'].to_numpy(),
                                           align_times)
        me_view = view
        break
    except Exception:
        continue
if me_raw is None:
    return dict(eid=eid, skip='no whisker motion energy from either camera', ...)
```

iii. "PREFER LEFT, FALL BACK TO RIGHT, exactly as the reference code does" — the reference
`bin_behaviors` tries `left-whisker-motion-energy` and falls back to `right-`. The agent
surveyed availability first: "left ME on 437/459 sessions, right on 420/459, neither on 14",
and the camera actually used is recorded per session in
`metadata['session_info'][i]['motion_energy_camera']`.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. None on the values themselves — the released `ROIMotionEnergy` trace (mean absolute frame
difference over a whisker-pad ROI) is used as published, with no filtering or normalisation.
It goes through the same `interp_behavior` as the wheel: non-finite samples removed, sorted,
linearly interpolated onto the 100 bin right edges of each trial, with the same coverage
validity rule.

ii.
```python
me_raw, valid_me = interp_behavior(me['times'].to_numpy(),
                                   me['whiskerMotionEnergy'].to_numpy(),
                                   align_times)
```

iii. Step 3: "Whisker motion energy: mean across pixels of the absolute frame difference in a
whisker-pad bounding box anchored between nose tip and eye (precomputed as
`*Camera.ROIMotionEnergy`)", i.e. the processing is already done upstream by IBL and the
reference simply consumes it.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: per-session terciles over all (trial, bin) samples, via the same
`discretize()` function, giving low/medium/high with fractions [0.3326, 0.3343, 0.3331]. The
edges are recorded per session in `metadata['session_info'][i]['whisker_me_tercile_edges']`.

ii.
```python
wheel_lab, wheel_edges = discretize(wheel_speed_raw)
me_lab, me_edges = discretize(me_raw)
```

iii. Same rationale as 7-c, and the per-session normalisation matters more here: "whisker ME
depends on camera, illumination and ROI size", and the fallback to a 150 Hz right camera on
some sessions makes any global threshold meaningless.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as the wheel: the camera frame times are on the same session clock as the spikes, and
the trace is evaluated at the right edge of each of the 100 neural bins measured from the same
`stimOn_times`, so output bin `k` and neural bin `k` describe the same 20 ms.

ii.
```python
grid = align_times[:, None] + BIN_RIGHT_EDGES[None, :]
vals[valid] = f(grid[valid])
```

iii. Same justification as 7-d; the `--show-processing` panel 7 overlays the raw camera trace
of trial 0 with the interpolated bin values for visual confirmation, and the agent notes the
camera is on the same synchronised clock so no further alignment is needed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or unusable data is dropped at the appropriate granularity and the reason is
recorded, rather than being patched or extrapolated:
- **Trial events missing** (NaN in any of the six curation events) — trial dropped by the
  reference mask.
- **Behavioural stream not covering the trial window, or non-finite after interpolation** —
  trial dropped (`valid_wheel & valid_me`).
- **A probe with no released spike sorting** — that insertion is skipped, the other is kept.
- **A session with no whisker motion energy from either camera** — session skipped (14
  sessions), reason recorded in `metadata['skipped_sessions']`.
- **A session with 0 kept neurons, or < 2 kept trials** — session skipped (none occurred).
- **Silent / zero-variance neurons** — the standardisation divides by
  `np.where(sd > 0, sd, 1.0)`, so they become 0 rather than NaN.
- **Degenerate tercile edges** on a zero-inflated trace — nudged apart with `np.nextafter`.
- **Anything else** — the per-session worker wraps the whole conversion in `try/except`,
  returns a `skip` with the traceback, and the traceback is printed. (In the full run no
  session hit this path; all 14 skips were missing motion energy.)
- **Sessions with only 1–3 neurons** are deliberately kept, not dropped.

ii.
```python
def _worker(args):
    eid, probe_rows, show_processing, out_dir, zscore = args
    try:
        return convert_session(eid, probe_rows, show_processing, out_dir, zscore)
    except Exception as exc:                                   # noqa: BLE001
        return dict(eid=eid, skip='exception: %s' % exc,
                    traceback=traceback.format_exc())
```

```python
if spikes is None or len(spikes) == 0 or 'times' not in spikes:
    continue
...
neural = ((spikes_binned.astype(np.float64) - mu)
          / np.where(sd > 0, sd, 1.0)).astype(np.float32)
```

```python
'skipped_sessions': [{'eid': r['eid'], 'reason': r['skip']} for r in skipped],
```

iii. Step 10 Check 5 enumerates each edge case and its handling, with the reasoning that
dropping is safer than imputing: trials at recording edges "are dropped ... rather than being
silently extrapolated". Sessions with 1–3 neurons are kept because "they are legitimate
recordings that passed every published criterion; the BWM paper's 'at least 5 neurons'
threshold applies to a *region within a session* for its region-level statistics, not to whole
sessions, and our decoder pools all of a session's neurons." The 14 skipped sessions are
justified as "a property of the released data ... not a bug", cross-checked against an
independent availability survey of all 459 sessions.

## 10-a. What are the most time-consuming steps of the code?

i. The script times every stage per session and prints an aggregated profile. Spike-sorting
I/O dominates at 76% of worker time (1,413 s of 1,859 s), followed by wheel loading at 18%
(329 s — `SessionLoader.load_wheel` interpolates the whole session onto a 1 kHz grid and
filters it), then binning 2%, trials 1%, motion energy 1%. Wall-clock for the full 459-session
run was 91 s of conversion + 17 s of pickling with 24 workers.

ii.
```python
t0 = time.time()
spike_times, spike_clusters, acronyms, counts = load_session_neurons(eid, probe_rows)
timing['spikes'] = time.time() - t0
...
if ok and 'timing' in ok[0]:
    agg = {}
    for r in ok:
        for k, v in r['timing'].items():
            agg[k] = agg.get(k, 0.0) + v
    tot = sum(r['seconds'] for r in ok)
    print('\ntiming (summed over sessions, %.0f s of worker time):' % tot)
```

iii. Step 9: "Timing profile: spike loading 78%, wheel loading 17%, binning 2%, trials 1%,
motion energy 1%." Step 7 estimated 2.4 s/session serial → ~1,100 s serial → "about 70–120 s
plus pickling" with 16 workers, and concluded "well under the 15 minute budget, so no further
optimisation is needed". The estimate held (91 s actual).

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent vectorised the two loops the reference code runs per trial, and left one.
- **Vectorised**: behaviour interpolation is a single `interp1d` call over the whole
  `(n_trials, 100)` grid instead of the reference's per-trial pool; and the trial-window
  boundaries are found with two `np.searchsorted` calls over all trials at once instead of the
  reference's per-trial boolean re-scan of the full spike vector (O(n_spikes + n_trials·n_bins)
  instead of O(n_trials·n_spikes)).
- **Not vectorised**: the per-trial loop inside `bin_spikes`, which does one `np.add.at` per
  trial. This could be a single `np.bincount` over a flat `(trial, unit, bin)` index, but at 2%
  of runtime there is nothing to gain. Likewise the two small Python loops that materialise the
  per-trial `input` and `output` arrays, and the list comprehension that splits `neural` into
  per-trial views — these are required by the target format (a list of per-trial arrays) rather
  than by the computation.

ii.
```python
i0 = np.searchsorted(spike_times, begs, side='left')
i1 = np.searchsorted(spike_times, ends, side='left')
out = np.zeros((n_trials, n_neurons, N_BINS), dtype=np.float32)
for k in range(n_trials):
    a, b = i0[k], i1[k]
    ...
    np.add.at(out[k], (spike_clusters[a:b], bin_idx), 1.0)
```

```python
grid = align_times[:, None] + BIN_RIGHT_EDGES[None, :]
vals[valid] = f(grid[valid])
```

iii. Step 6: "Spike binning uses one `np.searchsorted` for all trial boundaries and a single
`np.add.at` per trial, i.e. O(n_spikes + n_trials × n_bins) instead of O(n_trials × n_spikes).
A 407-trial, 61-neuron session binned in 0.03 s. Behaviour interpolation is a single vectorised
`interp1d` call over the whole (n_trials, 100) grid. Parallelism is at the session level (16
processes), not inside a session, which avoids the repeated pool creation and gives near-linear
scaling."

## 10-c. What processing does the code repeat multiple times?

i. The agent did not identify repeated processing in its own script (it identified it in the
reference: "the reference `get_spike_data_per_interval` spawns a multiprocessing pool **per
session** and calls `bincount2D` once per trial, each time re-scanning the full spike vector
with a boolean mask", and "`get_behavior_per_interval` likewise builds a pool per behaviour").
Reviewing the delivered script, there is essentially no repeated *processing* either; the few
minor duplications are:
- `spikes['clusters']` is traversed twice per probe, once by `np.isin(..., keep_idx)` to build
  the keep mask and once by the `remap[...]` fancy-index (the reference does this in one pass
  with a boolean array indexed directly by cluster id).
- `spikes_binned` is materialised in float32 and then up-cast to a second float64 copy for the
  standardisation before being cast back to float32 — three copies of the largest array.
- `BrainRegions.acronym2acronym` and the `ONE`/`BrainRegions` construction are per-process
  singletons, so those are *not* repeated.

ii.
```python
sel = np.isin(spikes['clusters'], keep_idx)
times_list.append(np.asarray(spikes['times'])[sel])
clusters_list.append(remap[np.asarray(spikes['clusters'])[sel]])
```

```python
neural = ((spikes_binned.astype(np.float64) - mu)
          / np.where(sd > 0, sd, 1.0)).astype(np.float32)
```

iii. Not discussed as such in CONVERSION_NOTES; the float64 copy is a deliberate, measured
choice, not an oversight: "the mean and std were being accumulated IN FLOAT32 over the roughly
1e5 values that each time bin pools, and that summation loses precision ... compute the
statistics with `dtype=np.float64` and do the subtraction/division in float64 before casting
the result to float32. This reduced the maximum difference to 4.77e-7."

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The agent explicitly removed one such step but did not enumerate the rest. Removed: the
reference's `spike_loader.raw_electrophysiology(band='ap', stream=True).fs` call, "skipped;
only recorded in metadata, never used in processing". Remaining items that are computed and
then not used by the decoder:
- **`input[1] = stim_onset`**, a binary indicator that is byte-for-byte identical in every
  trial of every session (1 at bin 25). It carries zero discriminative information, costs a
  third of the `input` array, and exists only to satisfy the instruction about representing
  onset times as binary series.
- **Whole-session wheel processing**: `SessionLoader.load_wheel()` interpolates and filters the
  entire multi-million-sample session at 1 kHz (18% of total runtime) when only the 445 × ~425
  two-second windows are ever read.
- **Diagnostics**: `mean_rate` (a full mean over the binned array), `n_all` / `n_good` cluster
  counts, and the per-session tercile edges are computed and stored in `metadata` but never
  used downstream — though they are what makes the consistency checks against the papers
  possible.
- **Storage**: `neural` is stored as dense standardised float32 (11.28 GB) rather than sparse
  counts; the standardisation could equally have been left to the decoder.

ii.
```python
stim_indicator = np.zeros(N_BINS, dtype=np.float32)
stim_indicator[STIM_ONSET_BIN] = 1.0
...
arr[1] = stim_indicator
```

```python
mean_rate=float(spikes_binned.mean() / BIN_SIZE),
```

iii. The binary onset channel is justified by the instruction "if an input is a time such as
onset of some stimulus, represent it as a binary time series", and the agent acknowledges it is
fixed by construction ("the `stim_onset` indicator has exactly one 1, at index 25, in every
trial of every session"). The diagnostics are justified as the basis of the consistency checks
in Steps 9–10. The remaining items are not discussed.
