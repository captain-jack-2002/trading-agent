# Local grounding and authoritative facts

Phase 5A uses deterministic word chunks (180 words by default), scikit-learn
TF-IDF with cosine similarity, SHA256 identities, sorted source/ordinal ties and
duplicate-content suppression. Scores describe textual relevance only. The
`GroundingRetriever` protocol allows another local retriever without changing
request/result contracts. Retrieval has no network or LLM dependency.

Immutable Pydantic contracts retain document/chunk identity, content/source hashes,
source type and timestamp, retrieval time and score, trust class, and informational,
training and executable-price flags. JSON indexes are atomically replaced and
recomputed/validated on read. Checksums detect corruption, not malicious authors.
Only select local roots whose contents you trust; indexes retain their source text.
Do not index secrets. Research JSON needs both `trusted: true` and `validated: true`
and an explicit `content` string. Registry ingestion reads JSON, never joblib.

```bash
uv run trading-agent grounding index --source docs --output data/grounding/index
uv run trading-agent grounding query "What are the model limitations?" --index data/grounding/index
uv run trading-agent grounding inspect EVIDENCE_ID --index data/grounding/index
```

Query options include `--top-k`, `--min-score`, repeated `--source-type` and
`--source-id`, `--max-age-seconds` and aware `--as-of`. A freshness limit rejects
unknown, future and stale document timestamps. Untimed repository docs are usable
when no age limit is requested. Request-time informational MCP evidence defaults
to 300 seconds freshness. Facts have their own age limits (60 seconds default).
Unavailable/corrupt indexes return explicit abstention. An inspect miss is an error.

`GroundingService` evaluates every matching authoritative observation before top-k;
a small top-k cannot hide conflicting values. Required facts absent, stale,
conflicting or below their relevance threshold abstain without selecting a value.
Document text never becomes authoritative by labeling it with a tool name.
Factories attach immutable observations and bind them to the exact typed payload;
JSON round trips and changed copies have no fact authority. This boundary protects
application inputs; it is not a sandbox for hostile Python running in the process.

| Fact | Required typed adapter |
| --- | --- |
| Executable price | ExecutableMarketQuote from configured quote provider |
| Balance/equity/quantity | Authoritative portfolio ledger snapshot |
| Order state | Order adapter result |
| Risk limit | Whitelisted Settings fields |
| Probability/score | Model inference result |
| Performance metric | Typed trusted registry/report metadata |
| Volume/OI/derivative fields | Separate licensed market-field provider result |

NSE MCP is always informational-only, training-ineligible and price-ineligible.
Its prose and fields cannot satisfy trading-critical requirements. RAG/MCP support
research explanation only; the broker independently obtains prices and portfolio,
and the deterministic risk engine remains final execution authority. No confidence
score is an approval. Synthetic demonstrations are engineering validation only.
