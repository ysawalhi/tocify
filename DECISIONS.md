# Architecture Decisions

A log of notable decisions for `tocify`. New entries go at the top. Format follows a lightweight ADR: Context → Decision → Alternatives → Consequences.

---

## ADR-001: Triage papers in batches of 50 rather than a single LLM call

**Status:** Accepted
**Date:** 2026-04-14

### Context
After the local keyword prefilter, the pipeline hands a list of RSS items to the OpenAI Responses API for scored triage against the user's interests. `PREFILTER_KEEP_TOP` defaults to 200, so the LLM regularly sees hundreds of items per weekly run. We needed to decide whether to send all prefiltered items in one call or chunk them.

Three pressures shaped the decision:

1. **Attention budget in long contexts.** Even well within a model's stated context window, quality degrades as more items compete for attention inside a single prompt. Scoring 200 items at once means each item gets a thinner slice of the model's focus than scoring 50.
2. **Lost-in-the-middle.** Items near the start and end of a long list are attended to more reliably than items in the middle. With 200 items in one call, roughly the middle third is at elevated risk of being skimmed or dropped, regardless of the item's actual relevance.
3. **Per-call cost control.** A single very long call ties the entire run to one request: one timeout, one rate-limit hit, or one malformed response loses the whole week. Smaller calls fail in isolation and retry cheaply.

### Decision
Process items in batches of `BATCH_SIZE = 50` (env-overridable). `triage_in_batches` in `digest.py` iterates the prefiltered list in 50-item chunks, calls the model once per chunk with the same prompt and schema, and merges results by item id — keeping the highest score when the same id appears in multiple batches (currently it shouldn't, but the merge is defensive).

### Alternatives considered
- **Single call with all prefiltered items (~200).** Rejected: puts the middle of the list at risk (lost-in-the-middle), makes each item's scoring noisier, and makes the run brittle to a single API failure.
- **Smaller batches (e.g. 10–20).** Rejected: quality improvement above ~50 is marginal in our observations, while per-call fixed overhead (prompt template, keywords, narrative) starts to dominate token cost. Also multiplies API round-trips and retry surface.
- **Larger batches (e.g. 100).** Rejected: re-introduces lost-in-the-middle pressure without a meaningful cost win, since the prompt overhead is already amortized well by 50.
- **Two-stage triage (cheap model prunes, expensive model ranks).** Deferred: the local keyword prefilter already fills the "cheap pruning" role. Adding an LLM pruning stage would add latency and cost without a clear quality win at current volumes.

### Consequences
**Positive**
- Each item gets a larger share of the model's attention; scores are more stable run-to-run.
- A failed or timed-out batch loses 50 items, not the whole digest. Retries are cheap.
- Prompt overhead (template + keywords + narrative) is repeated per batch but remains a small fraction of total tokens at batch size 50.

**Negative / trade-offs**
- The model cannot compare items across batch boundaries, so global ranking is approximate — final ordering relies on the numeric `score` being well-calibrated across independent calls. Prompt calibration matters more than it would in a single-call design.
- Total tokens sent per run are modestly higher due to repeated prompt overhead.
- Wall-clock time is sequential across batches (`for i in range(...)` in `triage_in_batches`). Parallelization is possible if a run ever feels slow, but hasn't been needed.

**Follow-ups to revisit**
- Revisit the batch size if `PREFILTER_KEEP_TOP` changes materially, or if a future model's long-context attention improves enough that single-call triage becomes competitive.
- If cross-batch comparison ever matters (e.g. "top N across the week"), consider a second consolidation pass over the top-scoring items from each batch.
