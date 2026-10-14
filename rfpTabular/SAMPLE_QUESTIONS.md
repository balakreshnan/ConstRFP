# Sample Questions for the RFP / Job Description Dataset

Reusable question bank for the two systems holding this data:

- **Microsoft Fabric Data Agent**: the five CSV tables (`document_catalog`, `solicitations`,
  `job_postings`, `evaluation_criteria`, `proposals`). Structured, SQL-style questions.
- **Azure AI Search**: the 62 source PDFs in `pdfdocs/`. Unstructured, retrieval and
  summarization questions.
- **Orchestrated / multi-agent**: questions that need both systems.

Key relationships the agents must know:

```
document_catalog.doc_id  <-  solicitations.source_doc_id
document_catalog.doc_id  <-  job_postings.source_doc_id
solicitations.solicitation_id  <-  evaluation_criteria.solicitation_id
solicitations.solicitation_id  <-  proposals.solicitation_id
proposals.firm_name  =  solicitations.awarded_firm   (for the winning proposal)
```

---

## 1. Microsoft Fabric Data Agent (structured)

### 1.1 Simple aggregates

| # | Question | Tables |
|---|---|---|
| 1 | How many solicitations were issued per sector and per year? | solicitations |
| 2 | What is the median estimated construction value by service category? | solicitations |
| 3 | What share of solicitations require CCM versus prefer it, by org type? | solicitations |
| 4 | What is the average response window in days for federally funded versus locally funded solicitations? | solicitations |
| 5 | How many job postings per state have a salary maximum above 150,000 dollars? | job_postings |
| 6 | What is the distribution of selection methods (QBS, two-step, best value, LPTA)? | solicitations |
| 7 | How many solicitations are open-call or IDIQ versus single project versus program-wide? | solicitations |
| 8 | What is the average number of proposals received per solicitation, by sector? | solicitations |

### 1.2 Grouped and filtered

| # | Question | Tables |
|---|---|---|
| 9 | For California K-12 solicitations, what are the average evaluation weights for key personnel, firm experience and price? | solicitations |
| 10 | Which agencies issued the most solicitations with mandatory pre-proposal meetings? | solicitations |
| 11 | Compare DBE goal percentages for FTA-funded versus FHWA-funded solicitations. | solicitations |
| 12 | Which states have the highest average professional liability insurance limits? | solicitations |
| 13 | List the top ten software tools required, with the count of solicitations for each. | solicitations |
| 14 | What are the most common key personnel roles required in construction inspection solicitations? | solicitations |
| 15 | Which org types most often impose page limits, and what is the median page limit? | solicitations |
| 16 | How many transit agency solicitations require prevailing wage and bonding? | solicitations |
| 17 | What is the average minimum years of experience for the lead project manager, by service category? | solicitations |
| 18 | Which job families and levels most often require a PMP? | job_postings |

### 1.3 Joins across tables

| # | Question | Tables |
|---|---|---|
| 19 | Which firms have the highest win rate across proposals, considering only firms with at least 50 bids? | proposals |
| 20 | Do proposals where the project manager holds a CCM score higher on key personnel than those without? | proposals |
| 21 | For awarded solicitations, what is the average gap between the winner's final score and the second-ranked score, by selection method? | proposals, solicitations |
| 22 | Which source documents seeded the solicitations with the most protests? | solicitations, document_catalog |
| 23 | What is the average awarded amount versus estimated fee by contract vehicle, with percent variance? | solicitations |
| 24 | How many shortlisted small firms lost to national firms in best value selections? | proposals, solicitations |
| 25 | Which evaluation criterion carries the highest average weight in healthcare solicitations? | evaluation_criteria, solicitations |
| 26 | For each source PDF, how many simulated solicitations, criteria rows and proposals exist? | all |
| 27 | Which firms win most often when they are local to the issuing state? | proposals, solicitations |
| 28 | What is the average proposed DBE percentage of winning proposals versus losing proposals on federally funded work? | proposals, solicitations |
| 29 | Which sectors have the largest gap between the median RFP-required years of experience and the median job posting minimum years? | solicitations, job_postings |
| 30 | For job postings seeded by transit agency documents, what is the salary range by level compared with city documents? | job_postings, document_catalog |

### 1.4 Trend and outcome

| # | Question | Tables |
|---|---|---|
| 31 | How has the share of electronic submissions changed year over year? | solicitations |
| 32 | Is time from issue to award longer when more addenda are issued? | solicitations |
| 33 | Which sector has the highest cancellation rate? | solicitations |
| 34 | What is the median days to fill for senior-level construction management jobs, by region? | job_postings |
| 35 | Are protests more common when the price weight is above 20 points? | solicitations |
| 36 | How does the number of questions received relate to the number of proposals received? | solicitations |
| 37 | What percentage of awards go to incumbents, by org type and year? | solicitations |
| 38 | Which year had the highest median construction value for program management solicitations? | solicitations |

### 1.5 Data-quality checks

| # | Question | Tables |
|---|---|---|
| 39 | List solicitations whose anticipated award date is before the proposal due date. | solicitations |
| 40 | Find solicitations with doc type RFQ but a non-zero price weight. | solicitations |
| 41 | Which solicitations have more shortlisted firms than proposals received? | solicitations |
| 42 | Are there any solicitations whose evaluation weights do not sum to 100? | solicitations |

---

## 2. Azure AI Search (unstructured, 62 PDFs)

### 2.1 Fact lookup inside a single document

| # | Question | Source PDF |
|---|---|---|
| 43 | What are the evaluation criteria and weights in the DeKalb program management RFP? | DeKalb rfp_no._17-752-026 |
| 44 | What insurance limits does the Moscone Center RFQ require? | moscone_small |
| 45 | What is the M/WBE requirement in the NYC DDC construction management RFP? | nycddccm_small |
| 46 | What qualifications does Clark County require for the Chief Project Representative? | Clark County 2014 |
| 47 | What is the salary range for the MBTA Assistant General Manager for Capital Delivery? | MBTA - AsstGMForCapitalDelivery |
| 48 | What is the total IDIQ ceiling and term in the FTA PMO RFP? | FTA+PMO+RFP_Final_11.09.23 |
| 49 | What is the estimated construction value of the Sacramento EchoWater projects? | Sacramento RFP |
| 50 | What is the DBE goal in the Port Columbus Concourse C solicitation? | Port of Columbus Intl Airpot |
| 51 | How many hard copies and what page limit does the Kuspuk School District RFP require? | Kuspuk School District |
| 52 | What are the staffing tiers and NTE caps in the NYC DDC requirements contracts? | nycddccm_small |

### 2.2 Cross-document semantic search

| # | Question |
|---|---|
| 53 | Which RFPs require the construction manager to hold a CCM certification? |
| 54 | Find solicitations that mention Primavera P6 scheduling requirements. |
| 55 | Which documents describe work on occupied hospital or school campuses? |
| 56 | Which RFPs are funded by FTA or FHWA and reference Davis-Bacon or DBE? |
| 57 | Which job descriptions require a Professional Engineer license? |
| 58 | Which solicitations use a two-step shortlist and interview process? |
| 59 | Which documents mention e-Builder, PMWeb or Procore? |
| 60 | Which RFPs include a conflict-of-interest clause barring the CM from later design or construction work? |
| 61 | Which solicitations mention prevailing wage or California DSA requirements? |
| 62 | Which job descriptions mention 24/7 on-call duty or night and weekend shifts? |

### 2.3 Summarization and comparison

| # | Question |
|---|---|
| 63 | Summarize the scope of services in the Sacramento EchoWater RFP. |
| 64 | Compare the submission requirements of the Loudoun County and Prince William County RFPs. |
| 65 | What page limits and copy counts do the Virginia solicitations impose? |
| 66 | List the key personnel roles named in the Maryland SHA construction inspection RFP. |
| 67 | Draft an evaluation criteria section modeled on the Hampton and Nassau County RFPs. |
| 68 | Compare the minimum qualifications for a project manager across the Arlington, Austin and Parkrose school district documents. |
| 69 | Summarize how the Penn State and St. Louis Science Center RFPs differ, given they share a scope template. |
| 70 | What insurance limits are common across the California city RFPs (Carlsbad, Coronado, El Cajon, East Palo Alto)? |
| 71 | Produce a one-paragraph summary of each federal document (GSA, VA, Smithsonian, USACE, FTA, Army Reserve). |
| 72 | Extract every protest procedure described across the corpus and compare the time limits. |

---

## 3. Orchestrated questions (Fabric + Azure AI Search)

| # | Question | Flow |
|---|---|---|
| 73 | The data shows K-12 solicitations weight personnel at about 30 percent. Which actual school district PDFs set a different weight, and what did they use instead? | Fabric aggregate, then search for the source wording |
| 74 | Which agencies in the data award most often to incumbents, and what do their source RFPs say about conflict of interest or prior work restrictions? | Fabric ranking, join to document_catalog, then search |
| 75 | What insurance limits are typical in the data for water and wastewater CM contracts, and does the Sacramento RFP match that pattern? | Fabric median, then search the PDF |
| 76 | Find the real RFP closest to this profile: California, transit, construction management, two-step selection. Then show the synthetic award statistics derived from it. | Search or catalog filter, then Fabric by source_doc_id |
| 77 | Build a draft RFP for a county public works inspection contract in Virginia. Pull structure and language from the Hampton, Danville and Christiansburg PDFs and fill weights, insurance limits and DBE goal with median values from the solicitations table. | Search for templates, Fabric for numbers |
| 78 | For the roles most often required in transit RFPs, what salaries do agencies post, and what do the BART and MBTA job descriptions actually state? | Fabric join, then search |
| 79 | Which source PDFs produced synthetic solicitations with the highest protest rate, and do those PDFs contain unusually strict or vague evaluation language? | Fabric, then search |
| 80 | Compare the typical federal small business set-aside mix in the data with what the VA and GSA solicitations require. | Fabric, then search |
| 81 | Generate a bid/no-bid summary for a hypothetical new solicitation: show win rates of competing firms in that sector from Fabric and the typical scope and submission burden from the closest PDFs. | Both |
| 82 | Write a key personnel section for a program management proposal using role and certification frequency from Fabric and the role definitions in the LAUSD and LACCD documents. | Both |

---

## 4. Configuration tips

**Fabric Data Agent instructions**
- State that `source_doc_id` is the join key from `solicitations` and `job_postings` to `document_catalog`, and `solicitation_id` joins `evaluation_criteria` and `proposals` to `solicitations`.
- Explain enum values: `ccm_requirement`, `pe_requirement`, `pmp_requirement`, `leed_requirement`, `osha_requirement` hold `Required`, `Preferred`, `Not mentioned`.
- Note that `salary_min_usd` and `salary_max_usd` are already annualized regardless of `pay_basis`.
- Note that `weight_*` columns on `solicitations` sum to 100 and mirror rows in `evaluation_criteria`.
- Note that boolean columns are stored as the strings `True` / `False`.
- Add questions 1, 9, 19, 21 and 31 as example queries so the agent learns the join patterns.

**Azure AI Search**
- Index `document_catalog.csv` fields (org_type, sector, state, doc_type, service_category) as filterable metadata on each PDF so semantic queries can be scoped, for example sector eq 'K-12 Education'.
- Chunk the large PDFs (NYC DDC 289 pages, FTA 212 pages, Moscone 102 pages) by section header; evaluation criteria and insurance sections are small and easy to lose in page-size chunks.
- The Clark County PDF is a scanned image and needs OCR enabled in the indexer.


---

# Part 2: Procurement and Supply Chain Questions

Tables: `vendors`, `purchase_orders`, `invoices_payments`, `shipments_deliveries`, `vendor_scorecard`.
Links: `vendors.firm_name = proposals.firm_name`; `purchase_orders.solicitation_id = solicitations.solicitation_id`;
`invoices_payments.po_id` and `shipments_deliveries.po_id` = `purchase_orders.po_id`.

## 5. Purchase history (Fabric)

| # | Question | Tables |
|---|---|---|
| 83 | What is the total spend per vendor over the last five fiscal years, and who are the top 20? | purchase_orders |
| 84 | How is spend split between on-contract task orders and off-contract purchases, by vendor size? | purchase_orders, vendors |
| 85 | Which categories make up most of the spend with each vendor? | purchase_orders |
| 86 | How many change orders were issued per vendor and what share of PO value do they add? | purchase_orders |
| 87 | Which buyer agencies issue the most POs to DBE-certified vendors? | purchase_orders, vendors |
| 88 | What is the average PO amount by procurement method and buyer org type? | purchase_orders |
| 89 | Which vendors have the highest cancellation rate on POs? | purchase_orders |
| 90 | For each awarded solicitation, how much has actually been ordered against the contract compared with the awarded amount? | purchase_orders, solicitations |
| 91 | Which contracts have PO spend exceeding 110 percent of the awarded amount? | purchase_orders, solicitations |
| 92 | How does spend per vendor trend year over year, and which vendors are growing fastest? | purchase_orders |

## 6. Payment history and on-time payment (Fabric)

| # | Question | Tables |
|---|---|---|
| 93 | What percentage of invoices are paid on time, by buyer org type? | invoices_payments, purchase_orders |
| 94 | What is the average days-to-pay for each vendor, and which vendors wait longest? | invoices_payments |
| 95 | Which buyer agencies pay latest on average, and how much late fee have they incurred? | invoices_payments |
| 96 | What share of invoices are disputed per vendor, and what are the top dispute reasons? | invoices_payments |
| 97 | How much early-payment discount was captured per fiscal year, and how much was offered but missed? | invoices_payments, purchase_orders |
| 98 | What is the total retention currently withheld per vendor? | invoices_payments |
| 99 | Are Net 60 invoices paid later relative to due date than Net 30 invoices? | invoices_payments |
| 100 | Which payment methods are associated with the fastest payment? | invoices_payments |
| 101 | Do federally funded POs get paid faster than locally funded ones? | invoices_payments, purchase_orders |
| 102 | List the pending invoices older than 60 days with their vendor and buyer. | invoices_payments |
| 103 | Which vendors have the most short-paid invoices, and what is the total short-paid amount? | invoices_payments |
| 104 | How many approval cycles do disputed invoices go through compared with clean invoices? | invoices_payments |

## 7. Supply chain, delivery and availability (Fabric)

| # | Question | Tables |
|---|---|---|
| 105 | What is the on-time delivery rate per vendor for physical goods versus service deliverables? | shipments_deliveries |
| 106 | What is the average lead time by item category and availability status at order? | shipments_deliveries |
| 107 | Which destination states or regions experience the longest delivery delays? | shipments_deliveries |
| 108 | Which carriers and transport modes have the best on-time and lowest damage rates? | shipments_deliveries |
| 109 | How often are items on backorder at the time of ordering, by category and vendor? | shipments_deliveries |
| 110 | What are the most common delay reasons, by region and season? | shipments_deliveries |
| 111 | What is the average freight cost per mile by transport mode? | shipments_deliveries |
| 112 | Which vendors ship from the farthest warehouses relative to project sites? | shipments_deliveries, vendors |
| 113 | What percentage of deliveries are accepted with exceptions or rejected, by vendor? | shipments_deliveries |
| 114 | Which items are most often returned or damaged? | shipments_deliveries |
| 115 | For older solicitations (before 2015), what were the delivery times and on-time rates compared with recent ones? | shipments_deliveries, solicitations |
| 116 | Which project sites (destination city) received the most equipment deliveries? | shipments_deliveries |
| 117 | How many shipments are currently in transit or scheduled, by vendor? | shipments_deliveries |
| 118 | What is the average stock level at order for items that ended up late? | shipments_deliveries |

## 8. Vendor performance and cross-dataset (Fabric)

| # | Question | Tables |
|---|---|---|
| 119 | Rank vendors by performance score for the latest fiscal year and show their win rate in proposals. | vendor_scorecard, proposals |
| 120 | Do vendors with higher proposal technical scores also deliver on time more often? | proposals, vendor_scorecard |
| 121 | Which vendors won contracts but have a high dispute rate or low in-full delivery rate? | solicitations, vendor_scorecard |
| 122 | For a given issuing agency, list every vendor it has awarded, total spend, on-time payment rate and on-time delivery rate. | solicitations, purchase_orders, invoices_payments, shipments_deliveries |
| 123 | Which sectors have the worst delivery performance and the slowest payments? | purchase_orders, invoices_payments, shipments_deliveries |
| 124 | Which small or DBE vendors outperform national vendors on both delivery and dispute rate? | vendors, vendor_scorecard |
| 125 | Has vendor performance improved or declined over the fiscal years for the top 10 vendors by spend? | vendor_scorecard |
| 126 | Which vendors have an expired insurance certificate or high risk rating but still receive open POs? | vendors, purchase_orders |
| 127 | Build a vendor 360 for one firm: contracts won, bid history, spend, payment behavior, delivery KPIs and scorecard trend. | all |
| 128 | For upcoming RFPs in the water sector, which incumbent vendors have the best combined payment-dispute and delivery record? | solicitations, vendor_scorecard |

## 9. Orchestrated (Fabric + Azure AI Search)

| # | Question | Flow |
|---|---|---|
| 129 | The Sacramento EchoWater RFP sets specific insurance and retention terms. Do the invoices under contracts seeded by that PDF show retention withheld at that rate? | Search PDF terms, Fabric on invoices via source_doc_id |
| 130 | Which RFPs require Primavera P6, and how much have the winning vendors spent on P6 licenses under those contracts? | Search, then Fabric on purchase_orders software category |
| 131 | For contracts seeded by federal PDFs (GSA, VA, FTA), compare on-time payment against the prompt-payment language in those documents. | Both |
| 132 | Draft a vendor performance clause for a new RFP using the delivery and dispute statistics of past winners in the same sector, citing the equivalent clauses from the Loudoun and Hampton PDFs. | Both |

## Configuration tips for the procurement tables

- Tell the Fabric agent that `paid_on_time`, `on_time_delivery` and `delivered_in_full` are empty strings when the invoice is unpaid or the shipment is not yet delivered, so rates should divide by non-empty counts.
- `days_late` and `delivery_delay_days` are negative when early. Clarify that "late" means greater than zero.
- `fiscal_year` runs July to June, so FY2020 is July 2019 through June 2020.
- `solicitation_id` on purchase orders is empty for off-contract purchases; use it only for on-contract analysis.
- `vendor_scorecard` is already aggregated. Prefer it for year-level KPI questions and the detail tables for drill-down.
