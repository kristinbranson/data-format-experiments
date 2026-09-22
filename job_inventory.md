## Inventory of benchmark job runs

### Job data roots

- `/groups/branson/home/bransonk/behavioranalysis/code/ScienceBenchmark/data-format/harbor-jobs`: Results of record, git repo, origin `github.com/kristinbranson/data-format-experiments`
- `/groups/branson/home/bransonk/behavioranalysis/code/ScienceBenchmark/data-format/`: Staging for new results before putting them in git repo, the default destination of `collect_cluster_results.py` (Columns, below), empty now
- `/groups/branson/home/bransonk/harbor-cluster-jobs`: raw cluster output
- `/groups/branson/home/bransonk/harbor-cluster-jobs-config_20260728-prompt_v5` raw cluster output for old agents v5 prompt, empty
- `/groups/branson/home/bransonk/harbor-cluster-jobs-superseded`: trash

All data have been merged into `harbor-jobs`

### Columns

- **Agents**: old = `config_20260728` (claude-code / claude-opus-4-6, codex / gpt-5.4, terminus-2
  on those models, harbor 0.1.45); new = `config_20260919` (claude-opus-5, gpt-5.6-sol,
  harbor 0.23.0, terminus capped at 800 turns).
- **Refs**: reference solutions and reference statistics from before or after the 09-17
  merge-back (`harbor-tasks/changes_since_preprint.md`).
- **Judges**: old = claude-opus-4-6 + gpt-5.4; new = claude-opus-5 + gpt-5.6-sol.
- **Prompt**: v4 = `prompt_v4/` and minimal v1; v5 = `prompt_v5/` and minimal v2.
- **Collected**: moved by `harbor-scripts/collect_cluster_results.py` in the data-format repo
  (`/groups/branson/home/bransonk/behavioranalysis/code/ScienceBenchmark/data-format`, which
  contains this one) into an analysis tree.

### Table

| # | Location prefix | Tasks | Agents | Refs | Judges | Prompt | Detail |
|---|---|---|---|---|---|---|---|
| 0 | `<task>/<agent>`¹ commit 0ffb910 | ALL | old¹ | old | old | v4 | maximal |
| 1 | `<task>_minimal/<agent>`¹ commit 0ffb910 | ALL | old | old | old | v4 | minimal |
| 2 | `<task>/`¹ | ALL | old¹ | new² | old | v4 | maximal |
| 3 | `<task>_minimal/<agent>` | ALL | old | new² | old | v4 | minimal |
| 4 | `<task>/<agent>-config_20260919` | ALL | new | new | new | v5³ | maximal |
| 5 | `<task>_minimal/<agent>-config_20260919` | ALL | new | new | new | v5 | minimal |
| 6 | `<task>/<terminus>-config_20260919` | ALL | new | new | new | v5 | maximal |
| 7 | `<task>_minimal/<terminus>-config_20260919` | ALL | new | new | new | v5 | minimal |
| 8 | `<task>_api/<agent>-config_20260919` | API | new | new | new | v5⁴ | api |
| 9 | `<task>/<agent>-config_20260728-prompt_v5` | ΔPROMPT | old | new | old | v5 | maximal |
| 10 | `<task>_minimal/<agent>-config_20260728-prompt_v5` | ΔPROMPT | old | new | old | v5 | minimal |
| 11 | `allen2p/<agent>/<trial>/judge_replicates/config_20260728/judgerep<k>`⁶ | allen2p | old¹ | new⁶ | old | v4 | maximal |
| 12 | `allen2p/<agent>/<trial>/judge_replicates/config_20260919/judgerep<k>`⁶ | allen2p | old¹ | new⁶ | new | v4 | maximal |

Task sets in the Tasks column:

- **ALL** = allen2p, hasnain2024, lee2025, majnik2025, map, mouseland, sosa2024, zhang2025
- **API** = allen2p_api, map_api, sosa2024_api, zhang2025_api
- **ΔPROMPT** = map, hasnain2024, majnik2025, sosa2024, mouseland: the tasks whose prompts
  changed substantively since the preprint

<agent> in {claude,claude-code,codex,terminus-opus,terminus-gpt}; <terminus> in {terminus-opus,terminus-gpt}

Rows 4–8 and 9–10 were collected from the raw job folders in `harbor-cluster-jobs`
(`hb_<task>_<agent>_t<N>`) and `harbor-cluster-jobs-config_20260728-prompt_v5`
(`hb728v5_<task>_<agent>_t<N>`) respectively. Those raw folders were deleted on 2026-09-21 once
both sweeps were collected; the stuck-terminal cases among them were moved to
`harbor-cluster-jobs-superseded` first.

### Footnotes

0. Rows 0 and 1 are not separate sets of trials. They are the same v4 trial folders on disk in harbor-jobs (rows 2 and 3), but with the scores they had in the preprint. Re-scoring overwrote those scores on disk, so they now exist only in harbor-jobs git history. `0ffb910` (2026-08-03) is the last commit before the re-scoring commit `7c974fd`. At that point each trial's `verifier/` held `metrics.json` and `reward.txt` (a 0/1 pass-fail); partial-credit `reward.json` did not exist yet. To read one:
   ```
   git -C /groups/branson/home/bransonk/behavioranalysis/code/ScienceBenchmark/data-format/harbor-jobs \
       show 0ffb910:lee2025_minimal/claude-code/2026-07-28__21-46-39_trial1/verifier/metrics.json
   ```
   That trial's `reward.txt` reads `1` at `0ffb910`; its current `reward.json` has reward 0.921.
1. Applies to the claude and codex folders only. Those maximal v4 trials ran in March–April,
   before harness versions were pinned: the models are the same (claude-opus-4-6, gpt-5.4), but
   the CLI versions span claude-code 2.1.72–2.1.97 and codex 0.114–0.118, not exactly 2.1.81 and
   0.116.0. The terminus folders ran 07-28 to 07-30 on the `config_20260728` pins. The claude
   folder is `claude` for hasnain2024, lee2025, majnik2025, map and sosa2024, and `claude-code`
   for allen2p, mouseland and zhang2025; no task has both.
2. Re-scored 09-18 between 23:19 and 23:58, mouseland 09-19 at 10:11, and committed in harbor-jobs as
   `7c974fd` and `6e6ec13`. The judges were re-run with the old models but the current judge
   instructions. The preprint's original scores are in harbor-jobs's history at `0ffb910` (rows 0 and 1,
   footnote 0).
3. These prompts still contained "Do NOT look at any files outside this directory", which
   `a69fcb7` removed at 22:07 on 09-19, after these trials ran. The v5 maximal prompts in rows 6
   and 9 do not have it. The minimal prompts never had it.
4. The API tasks were collected in one pass on 2026-09-21, all four arms together. The sosa2024_api
   claude/codex prompts have the older wording "Do not directly load and parse the `hdf5`
   files."; the terminus ones have "… with `h5py`." (`8b689da`, 09-20).
5. Committed in harbor-jobs on 2026-09-21, one commit per sweep: `56d42a9` holds the September sweep
   (240 trials, rows 4–8) and `7accb45b` holds the v5 sweep (90 trials, rows 9–10). Alongside them
   are `cc94999`, which dropped eight duplicate READMEs, and `2e1c8115`, the judge variability
   replicates (footnote 6). As with every trial in harbor-jobs, only the allow-listed light files are tracked --
   `metrics.json`, the judge output, `DECISIONS.md`, `trial.log`, `trajectory.json`, the agent's
   `convert_data.py` and the like -- never `converted_data.pkl` or the snapshots.
6. Rows 11 and 12 are the judge variability experiment: judges-only reruns of the six allen2p
   trials in row 2 run by claude-code and codex, three each, so `<agent>` is claude-code or
   codex. Each trial was rejudged five times, `<k>` from 1 to 5, with each set of judges: row 11
   with the old judges and row 12 with the new, 30 replicates apiece. Only the two LLM judges
   were re-run; nothing was re-scored. The judges grade each trial against the human reference
   solution in the task's `tests/` (`reference_convert_data.py`, `reference_DECISIONS.md`), and
   both checkouts the replicates ran from hold the post-merge-back version, byte-identical
   between them, hence new Refs for both rows. These are not trials: each
   sits inside the trial it rejudges, and `trial_metrics.py` does not count it. Both rows are
   committed together as `2e1c8115`; `judge_variability.md` in the data-format repo describes
   the experiment.

### Status and other contents

- **Every row is complete and collected.** Each `*-config_20260919` arm in harbor-jobs holds exactly three
  trials, and so does each of the 30 `*-config_20260728-prompt_v5` arms.
- **Four stuck trials were superseded, not collected** (row 6). They got stuck, ran to the
  800-turn cap, and were scored on nothing: terminus-opus `allen2p` t3, `hasnain2024` t1,
  `lee2025` t1 and `map` t1, all started 2026-09-20. Their 2026-09-21 reruns were collected in
  their place.
- **`harbor-cluster-jobs-superseded` holds all 13 stuck-terminal cases**, each keeping its
  sub-path, e.g. `hb_allen2p_terminus-opus_t3/allen2p/terminus-2/2026-09-20__05-17-38_trial1`:
  those four, the eight terminus-opus trials killed while stuck, and the `mouseland_minimal`
  terminus-gpt t1 trial killed by mistake. None is part of any arm; their reruns are what
  `harbor-jobs` holds.
- A collected `trial<K>` is numbered in collection order, not by the original job's `t<N>`: the
  `allen2p` terminus-opus rerun of job t3 is `trial2`. The `hb_..._t<N>` path in each trial's
  `config.json` names the job it came from.
- **`harbor-cluster-jobs` was emptied on 2026-09-21** except `refstats/` and
  `refstats_zhang2025_datalimit/`, the oracle runs behind the reference stats, which were kept.
  Every other raw job folder went: the unscored attempts, the resubmissions that failed at image
  build in the Launchpad outage, the failed 07-28 runs, and the 2026-09-19 bring-up and gate
  runs. `harbor-cluster-jobs-config_20260728-prompt_v5` is empty.
- harbor-jobs also contains `oracle` (lee2025, majnik2025, sosa2024) and `debug`; `badtrial3` directories
  under `lee2025/claude` and `zhang2025/claude-code`, which were never re-scored and still carry
  old-reference scores; and a `trial4` under `zhang2025/codex`.
