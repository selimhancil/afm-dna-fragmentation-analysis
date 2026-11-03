# Knowledge base

- `library.csv`: papers (title, authors, year, DOI/URL, language, access, read status).
- `evidence.csv`: one row per fact. Each row has the exact quote, the source, and a `verified` flag.
  New rows are `unverified`. Only the researcher or supervisor changes this to
  `user_verified` or `supervisor_verified` after checking the original paper.
- Rows were read through a page-reading tool. Quotes must be checked against the original PDF before use in a paper.
- Legal sources only: OpenAlex, Semantic Scholar, PubMed Central, arXiv/bioRxiv, DergiPark, TR Dizin, YOK Tez.
  Paywalled PDFs stay in a private archive, not in git.
- `python scripts/10_knowledge.py` checks the tables. `python scripts/11_collect_openalex.py "query"` lists new candidates (needs internet).
