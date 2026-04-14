# CLAUDE.md

## What this project does
`tocify` is a weekly journal table-of-contents digest pipeline. A GitHub Action pulls new entries from a list of RSS feeds, locally prefilters them against user keywords, sends the survivors to the OpenAI Responses API for scored triage against the user's research interests, and commits a ranked markdown digest (`digest.md`) back to the repo.

## Tech stack
- **Python 3.11** (pinned in the workflow)
- **Key deps** (`requirements.txt`): `openai>=1.0.0`, `feedparser>=6.0.0`, `python-dateutil>=2.9.0`. The workflow also upgrades `httpx` and `certifi`.
- **Runtime**: GitHub Actions (`ubuntu-latest`). No local server, no database.

## Project structure
- `digest.py` — the entire pipeline: `load_feeds` → `fetch_rss_items` → `keyword_prefilter` → `triage_in_batches` (OpenAI) → `render_digest_md`.
- `feeds.txt` — RSS feed list. Supports `#` comments and optional `Name | URL` format.
- `interests.md` — user keywords + narrative. Parsed by `parse_interests_md`.
- `prompt.txt` — triage prompt template with `{{KEYWORDS}}`, `{{NARRATIVE}}`, `{{ITEMS}}` placeholders.
- `digest.md` — generated output, auto-committed by the workflow bot.
- `.github/workflows/weekly-digest.yml` — the runner.
- `tests/` — pytest characterization tests.

## Conventions
- **Configuration** is entirely env-var driven (see the block of `os.getenv(...)` constants at the top of `digest.py`: `MODEL`, `LOOKBACK_DAYS`, `PREFILTER_KEEP_TOP`, `BATCH_SIZE`, `MIN_SCORE_READ`, `MAX_RETURNED`, etc.). The workflow sets production values inline under `env:`; defaults in code are reasonable for local runs.
- **Secrets** live in GitHub Actions secrets. Only `OPENAI_API_KEY` is required; it is injected into the `Run digest` step. Never commit keys.
- **Triggering**: currently `workflow_dispatch` only — the `schedule:` cron is commented out in `weekly-digest.yml`. Re-enable it to get the Monday run back.
- **Output commits**: a bot identity (`toc-digest-bot`) stages only `digest.md` and pushes. `git commit ... || exit 0` makes an empty-diff run a no-op.

## Gotchas
- `interests.md` is parsed by looking for markdown H2 sections titled exactly `Keywords` and `Narrative` (case-insensitive via regex in `section()`). Renaming those headings silently yields empty keywords / narrative. Keyword list items may be plain lines or bullet-prefixed (`- `, `* `, `+ `).
- `prompt.txt` placeholders are literal string-replaces, not a template engine — don't rename them without updating `call_openai_triage`.
- `keyword_prefilter` has a fallback branch: if fewer than `min(50, keep_top)` items match, it returns `items[:keep_top]` *unfiltered* (the model sees everything). Characterization tests in `tests/test_keyword_prefilter.py` pin this behavior — update them deliberately.
- The OpenAI call uses the **Responses API** (`client.responses.create`) with a strict JSON schema (`SCHEMA` in `digest.py`). Switching to Chat Completions would require reworking both the call and the response parsing.
- The workflow explicitly clears `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` before running — left in after a past networking issue. Don't re-add proxy envs without reason.
