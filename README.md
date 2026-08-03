# harbor-jobs

Archive of https://github.com/kristinbranson/data-format benchmark runs: what each coding agent produced when asked to convert a neuroscience dataset into the decoder's format, and how the verifier scored it. Made a repo to log post-hoc processing.

Code in `data-format` repo expects this to live in the subdirectory `harbor-jobs`.

## Layout

```
<task>/<agent>/<timestamp>_trial<N>/
    config.json                        what was requested
    trial.log, result.json             what harbor did, and how it ended
    agent/trajectory.json              the agent's step log
    verifier/
        metrics.json                   every number the verifier measured
        reward.txt, ctrf.json          the 0/1 reward and per-test pass/fail
        test-stdout.txt                pytest output, including *why* a test failed
        judge/<name>/                  LLM judge verdicts and transcripts
        snapshot/
            convert_data.py            the agent's conversion script
            CONVERSION_NOTES.md        the agent's own account of what it did
            *_out.txt                  what its scripts printed
```

- 8 base tasks
- 2 prompt types: `task` has a long, complex prompt and `task_minimal` has a minimal prompt
- 4 agents: `claude`, `claude-code`, `codex`,`terminus-gpt` and `terminus-opus`, plus `oracle`
- 3 trials per task x prompt x agent

## What is tracked, and what is not

**Not tracked, deliberately:**

- **`*.pkl`** — the converted datasets, several TB. These can be created by running the agent's convert_data.py code
- **`agent/` except `trajectory.json`** about 90% of the file count: terminal recordings, sqlite session state, vendored plugin files. Process, not result. Note `.gitignore` excludes `**/agent/*/` — whole subdirectories rather than individual files — because git cannot skip a directory it has been told to descend into. Without this, git status was slow
- **`verifier/snapshot/cache/`** 
- **`*.png`** — regenerable plots.
- **`*_badtrial*/`** — runs abandoned as bad

## Reading `metrics.json`

We changed some of the verifier metrics posthoc, and just reran these computations, which ediges `metrics.json` but not `reward.txt`, `ctrf.json`, and `test-stdout.txt`. Thus, `reward.txt` and `metrics.json` can disagree, and `metrics.json` is the more current of the two.

`metrics.json` is refreshed in place by `harbor-scripts/rerun_metrics.py`, which re-runs the verifier's own test functions against a stored snapshot. `reward.txt`, `ctrf.json` and `test-stdout.txt` are written only by a full verifier run and are never touched by that path. So a trial can carry metrics that fail a check its reward says it passed — which is the case today, because range-only checks were added to `test_data_stats` after these trials ran. 

## `verifier_rerun_*/` directories

`merge_rerun_verifier.sh` copies a successful rerun's outputs back into `verifier/`, so what is tracked is already the post-rerun state and the
`verifier_rerun_*` directories are excluded by construction. One consequence worth knowing: because the merge *replaces* `ctrf.json` and `test-stdout.txt`, the pre-rerun originals for those trials no longer exist anywhere.

## Credits

Lingqi Zhang and Kristin Branson