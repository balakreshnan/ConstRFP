# Proposed Synthetic Dataset Schema

Source: 62 PDFs in `pdfdocs/` (45 solicitations / manuals, 17 job descriptions) covering
construction management, program management, project management and construction
inspection services for public owners.

Design: a small relational star. `document_catalog.csv` holds the **real** metadata extracted
from each PDF. Every synthetic table links back to it through `source_doc_id`, so you can
always filter "all simulated records derived from the LAUSD RFQ".

Minimum volume: **1,000 primary rows per source document** (62,000+ in the two primary
tables), plus child tables that fan out from those rows.

```
document_catalog (62, real)
   ├── solicitations (45 docs x 1,000 = 45,000)      primary table for RFP analytics
   │      ├── evaluation_criteria (~5 per solicitation = ~225,000)
   │      ├── proposals (~5 per solicitation = ~225,000)
   │      │      └── proposal_key_personnel (~5 per proposal = ~1.1M)   [optional]
   │      └── contract_rate_schedule (~8 labor categories per award = ~360,000) [optional]
   └── job_postings (17 docs x 1,000 = 17,000)      primary table for staffing analytics
          └── job_applicants (~15 per posting = ~255,000) [optional]
```

---

## 0. document_catalog.csv  (62 rows, REAL extracted data, already written)

One row per PDF: doc_id, file_name, doc_type, title, solicitation_number, issuing_org,
org_type, state, city, service_category, sector, delivery_method, values, dates, submission
rules, selection method, compensation type, evaluation criteria with weights, required
certifications, key personnel roles, software, insurance limits, DBE/MBE/WBE goals,
salary (job descriptions), scope summary, notes.

---

## 1. solicitations.csv  (45,000 rows: 1,000 per solicitation-type document)

Each row is a simulated procurement "in the style of" its source document: same org type,
sector, service category and selection style, with randomized but realistic agency names,
values, dates, weights and outcomes.

| Column | Type | Notes / distribution source |
|---|---|---|
| solicitation_id | string PK | `SOL-000001` |
| source_doc_id | FK | document_catalog.doc_id |
| source_file | string | PDF file name |
| solicitation_number | string | agency-style, e.g. `RFP 17-18-075`, `PIN 8502016VP0085P` |
| doc_type | enum | RFP, RFQ, RFQ/RFP two-step, IFB |
| title | string | templated from source title |
| issuing_org | string | synthetic agency name matched to org_type/state |
| org_type | enum | City, County, State DOT, State Agency, Transit Agency, K-12 School District, University/College, Federal, Healthcare, Housing Authority, Airport/Port, Utility/Water Authority, Nonprofit |
| state | string | 2-letter, weighted to CA/VA/NY/TX heavy corpus |
| region | enum | West, South, Northeast, Midwest |
| city | string | |
| sector | enum | Transportation/Highway, Transit/Rail, Aviation, K-12 Education, Higher Education, Water/Wastewater, Healthcare, Public Buildings/Facilities, Parks/Recreation, Housing, Telecom/Fiber, Defense/Federal, Mixed/Multiple |
| service_category | enum | Construction Management, Program Management, Project Management, CM-at-Risk, Owner's Representative, Construction Inspection, Design-Build |
| delivery_method | enum | Design-Bid-Build, CMAR, Design-Build, Multiple, Staff Augmentation (task order / on-call) |
| contract_vehicle | enum | Single Project, On-Call/IDIQ, Program-wide |
| project_name | string | |
| scope_summary | text | 1-2 templated sentences |
| funding_source | enum | Local, State, Federal-FTA, Federal-FHWA, Federal-Other, Bond, Grant, Mixed |
| federally_funded | bool | drives DBE / Davis-Bacon flags |
| est_construction_value_usd | number | log-normal by sector; $0.5M to $8B |
| est_fee_usd | number | 1-6% of construction value (type-dependent) or annual NTE cap |
| fee_pct_of_construction | number | |
| base_term_months | int | 12, 18, 24, 36, 60 |
| option_years | int | 0-4 |
| compensation_type | enum | Hourly Rate Schedule, Time & Materials NTE, Lump Sum, Cost Plus Fixed Fee, Percentage Fee |
| selection_method | enum | QBS, Two-step shortlist+interview, Best Value, Lowest Price Technically Acceptable |
| issue_date | date | |
| pre_proposal_date | date | nullable |
| pre_proposal_mandatory | bool | |
| questions_due | date | ~7 days before due |
| proposal_due | date | 14-45 days after issue |
| anticipated_award_date | date | |
| response_window_days | int | derived |
| submission_format | enum | Hard copy, Electronic, Both |
| hard_copies_required | int | 0-7 |
| page_limit | int | nullable; 8-50 |
| sf330_required | bool | federal/some state |
| num_addenda | int | 0-5 |
| num_questions_received | int | 0-60 |
| weight_firm_experience | int | eval weights, sum = 100 |
| weight_key_personnel | int | |
| weight_approach | int | |
| weight_references_past_perf | int | |
| weight_capacity_local | int | |
| weight_price | int | 0 in most QBS |
| weight_dbe_participation | int | |
| weight_interview | int | |
| min_years_lead_pm | int | 5-20, mode 10 |
| ccm_requirement | enum | Required, Preferred, Not mentioned |
| pe_requirement | enum | Required, Preferred, Not mentioned |
| pmp_requirement | enum | same |
| leed_requirement | enum | same |
| osha_requirement | enum | same |
| state_inspector_cert_required | bool | inspection contracts |
| key_roles_required | pipe list | e.g. `Project Manager | Resident Engineer | Inspector | Scheduler | Cost Estimator` |
| est_fte | number | staffing level |
| software_required | pipe list | Primavera P6, e-Builder, Procore, PMWeb, Bluebeam, MS Project, SiteManager |
| gl_limit_usd | number | 0.5M-2M |
| pl_limit_usd | number | 1M-5M |
| auto_limit_usd | number | |
| umbrella_limit_usd | number | nullable |
| dbe_goal_pct | number | nullable |
| mbe_goal_pct | number | nullable |
| wbe_goal_pct | number | nullable |
| local_preference_pct | number | nullable |
| small_business_setaside | enum | None, SB, SDVOSB, 8(a), HUBZone |
| prevailing_wage | bool | |
| bonding_required | bool | |
| protest_period_days | int | |
| num_proposals_received | int | 2-15 |
| num_shortlisted | int | |
| interviews_held | bool | |
| awarded_firm | string | synthetic firm (consistent across tables) |
| awarded_firm_size | enum | Small, Mid, Large, National |
| awarded_amount_usd | number | |
| award_vs_estimate_pct | number | |
| incumbent_awarded | bool | |
| status | enum | Open, Shortlisting, Awarded, Cancelled, Protested, Re-bid |
| days_issue_to_award | int | |

## 2. job_postings.csv  (17,000 rows: 1,000 per job-description document)

| Column | Type | Notes |
|---|---|---|
| posting_id | string PK | |
| source_doc_id | FK | |
| source_file | string | |
| job_title | string | templated: Construction Manager, Senior Project Manager, Resident Engineer, ... |
| job_family | enum | Construction Management, Project Management, Program Management, Engineering, Inspection |
| level | enum | Entry, Mid, Senior, Manager, Director, Executive |
| employer_name | string | synthetic |
| employer_type | enum | same enum as org_type plus Private Company |
| sector | enum | |
| state / city | string | |
| employment_type | enum | Full-time, Part-time, Contract, Bench/On-call |
| flsa_exempt | bool | |
| union | bool | |
| work_mode | enum | Onsite, Field, Hybrid |
| travel_pct | int | |
| salary_min_usd / salary_max_usd | number | annualized |
| pay_basis | enum | Annual, Hourly, Biweekly |
| hourly_bill_rate_usd | number | nullable (bench roles) |
| min_years_experience | int | |
| education_required | enum | HS, Associate, Bachelor's, Master's |
| degree_field | string | Civil Eng, Construction Mgmt, Architecture, ... |
| ccm_requirement / pe_requirement / pmp_requirement / leed_requirement / osha_requirement | enum | Required, Preferred, Not mentioned |
| supervises_headcount | int | |
| reports_to | string | |
| program_value_managed_usd | number | |
| posting_date / close_date | date | |
| open_until_filled | bool | |
| residency_required | bool | |
| background_check / drug_screen | bool | |
| benefits_flags | pipe list | Pension, 401k, Tuition, Transit pass, ... |
| num_applicants | int | |
| days_to_fill | int | nullable |
| filled | bool | |

## 3. evaluation_criteria.csv  (~225,000 rows)

| Column | Notes |
|---|---|
| criterion_id, solicitation_id (FK) | |
| criterion_order | 1..n |
| criterion_name | Firm Experience, Key Personnel, Approach & Work Plan, References/Past Performance, Capacity & Local Presence, Price/Fee, DBE Participation, Interview, QA/QC Plan, Schedule Control |
| weight_points | sums to 100 per solicitation |
| scoring_phase | Written, Interview, Both |
| pass_fail | bool |

## 4. proposals.csv  (~225,000 rows, 2-15 per solicitation)

| Column | Notes |
|---|---|
| proposal_id, solicitation_id (FK) | |
| firm_name, firm_size, firm_hq_state, is_local, dbe_certified, is_incumbent | |
| team_subconsultants | int |
| proposed_dbe_pct | |
| proposed_pm_years, proposed_pm_has_ccm, proposed_pm_has_pe | |
| proposed_fte | |
| proposed_fee_usd | nullable for QBS |
| score_firm_experience, score_key_personnel, score_approach, score_references, score_capacity, score_price, score_dbe, score_interview | |
| technical_score_total, final_score | |
| rank, shortlisted, interviewed, awarded | |
| protest_filed | bool |
| responsive | bool (late / missing forms) |

## 5. proposal_key_personnel.csv  (optional, ~1.1M rows)

proposal_id (FK), role, years_experience, certifications (pipe list), degree, pct_time_committed,
hourly_rate_usd, is_subconsultant, prior_projects_similar.

## 6. contract_rate_schedule.csv  (optional, ~360,000 rows)

solicitation_id (FK), awarded_firm, labor_category, min_years, raw_hourly_rate_usd,
multiplier, billing_rate_usd, escalation_pct_per_year.

## 7. job_applicants.csv  (optional, ~255,000 rows)

posting_id (FK), applicant_id, years_experience, certifications, education, current_employer_type,
salary_expectation_usd, stage_reached (Applied/Screened/Interviewed/Offered/Hired), days_in_process.

---

## Realism rules baked into the generator

- Dates span 2009-2024 by default to match the corpus (switchable to 2022-2026).
- Federal funding forces DBE goal, Davis-Bacon flag and SF330 more often.
- QBS and two-step selection set `weight_price` to 0 or 5; Best Value gives 10-40.
- CCM requirement probability by service category: CM 70%, Inspection 40%, PM 55%.
- Inspection contracts use annual NTE caps ($250K-$5M) rather than construction value.
- Program management records have construction values of $100M-$8B and 36-60 month terms.
- Winning proposals have higher personnel scores more often than lower price.
- ~3% of records carry data quirks seen in the corpus (RFQ titled as RFP, due date before award date, duplicate templated scope text).
