# RFP / Job Description Synthetic Dataset

Built from the 62 PDFs in `../pdfdocs/` (public-sector RFPs, RFQs and job descriptions for
construction, program and project management services). See `SCHEMA.md` for every column.

| File | Rows | What it is |
|---|---:|---|
| `document_catalog.csv` | 62 | **Real** metadata extracted from each PDF (org, sector, values, dates, criteria, certs, insurance, salary). Hub table. |
| `solicitations.csv` | 45,000 | Simulated procurements, 1,000 per solicitation-type PDF, seeded from that PDF's attributes. |
| `job_postings.csv` | 17,000 | Simulated job requisitions, 1,000 per job-description PDF. |
| `evaluation_criteria.csv` | ~280,000 | Weighted criteria per solicitation (weights sum to 100). |
| `proposals.csv` | ~316,000 | Competing proposals per solicitation with per-criterion scores, rank, award, protest. |
| `source_profiles.json` | 62 | Full structured profile per PDF used to seed the generator. |
| `generate_dataset.py` | | Deterministic generator (seed 42). Re-run: `python generate_dataset.py source_profiles.json .` |

Keys: `source_doc_id` links every synthetic row to `document_catalog.doc_id`;
`solicitation_id` links criteria and proposals to solicitations.

Notes
- Dates span 2009-2024 to match the corpus. Four source files are manuals/forms/memos
  (CTDOT, USACE, Army Reserve D-B manual, Oklahoma A305 form); their 1,000 rows are
  solicitations generated in the style of that agency and service type.
- About 3% of solicitation rows carry deliberate real-world quirks (RFQ labelled as RFP,
  award date before due date) because the corpus contains them.
- All firm names, agency names and figures are synthetic.
