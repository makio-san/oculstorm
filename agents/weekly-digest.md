# weekly-digest

Every Monday you write one page for the owner on how the week went. You change no files.
Follow [RULES.md](RULES.md), except that the digest itself is written in Turkish, plain and
short, because the owner reads it. Still no em or en dashes.

## Each run

1. Set up your worktree (`weekly-digest`), `git fetch -q origin live-archive`.
2. Gather, for the last 7 days:
   - live: `$PY agents/bin/live_score.py --days 7` and for the whole log; missing hours;
     storms seen (`agents/reports/storms/`);
   - fresh holdout: `reports/holdout.json` if present;
   - merged PRs: `gh pr list --state merged --search "merged:>=<date>" --json number,title,headRefName`;
   - open PRs and their review state, rejected ones and why;
   - experiments: issues with `result:*` labels changed this week, with their validation
     numbers; open experiment and feature issues;
   - retrain issues and test ledger lines;
   - live workflow runs: how many ran and failed;
   - anything labelled `agents:needs-human`.
3. Open an issue labelled `agents:digest`, titled `Haftalık özet: <date>`, with these parts:
   `Canlı tahmin`, `Bu hafta birleşenler`, `Deneyler`, `Model eğitimi`, `Sorunlar`,
   `Senin kararını bekleyenler`. Numbers only from what you ran or read; name the source.
   Explain metrics in one plain sentence each the first time they appear (RMSE in nT, skill
   against persistence, 0 means no better than "the next hour equals this hour").
4. Close last week's digest issue.
5. Finish with the issue link.
