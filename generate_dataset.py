"""
Synthetic dataset generator for the RFP / job-description corpus in pdfdocs/.

Seeds every synthetic record from the real metadata of one source PDF
(profiles extracted into all_profiles.json), so each source document yields
exactly ROWS_PER_DOC primary rows. Deterministic: SEED fixes the output.

Usage:  python generate_dataset.py <all_profiles.json> <output_dir>
"""
import csv, json, math, os, random, sys
from datetime import date, timedelta

SEED = 42
ROWS_PER_DOC = 1000
YEAR_MIN, YEAR_MAX = 2009, 2024
rng = random.Random(SEED)

# --------------------------------------------------------------------------- lookups
JOB_TYPES = {"Job Description"}
STATE_REGION = {
    "CA": "West", "WA": "West", "OR": "West", "NV": "West", "AZ": "West", "UT": "West", "NM": "West", "CO": "West", "AK": "West",
    "TX": "South", "VA": "South", "GA": "South", "FL": "South", "LA": "South", "MD": "South", "DC": "South", "WV": "South", "OK": "South", "NC": "South", "TN": "South",
    "NY": "Northeast", "MA": "Northeast", "PA": "Northeast", "CT": "Northeast", "NJ": "Northeast",
    "OH": "Midwest", "MO": "Midwest", "IL": "Midwest", "MI": "Midwest", "MN": "Midwest",
}
CITIES = {
    "CA": ["Fresno", "Oakland", "Riverside", "San Jose", "Bakersfield", "Santa Rosa", "Stockton", "Irvine", "Chula Vista", "Pasadena"],
    "VA": ["Richmond", "Norfolk", "Roanoke", "Chesapeake", "Lynchburg", "Alexandria", "Fredericksburg", "Charlottesville", "Leesburg", "Suffolk"],
    "NY": ["Buffalo", "Rochester", "Albany", "Syracuse", "Yonkers", "White Plains", "Brooklyn", "Queens", "Hempstead", "Schenectady"],
    "TX": ["Dallas", "San Antonio", "El Paso", "Arlington", "Plano", "Corpus Christi", "Laredo", "Lubbock", "Frisco", "Round Rock"],
    "OH": ["Columbus", "Cleveland", "Cincinnati", "Toledo", "Akron", "Dayton", "Parma", "Canton", "Lorain", "Elyria"],
    "WA": ["Tacoma", "Spokane", "Bellevue", "Everett", "Kent", "Renton", "Olympia", "Vancouver", "Bellingham", "Redmond"],
    "MA": ["Boston", "Worcester", "Springfield", "Cambridge", "Lowell", "Quincy", "Newton", "Somerville", "Framingham", "Brockton"],
    "PA": ["Pittsburgh", "Harrisburg", "Allentown", "Erie", "Scranton", "Lancaster", "Reading", "State College", "Bethlehem", "York"],
    "MD": ["Baltimore", "Rockville", "Frederick", "Annapolis", "Gaithersburg", "Bowie", "Hagerstown", "Towson", "Columbia", "Silver Spring"],
    "DC": ["Washington"] * 10,
    "AZ": ["Phoenix", "Tempe", "Mesa", "Tucson", "Chandler", "Scottsdale", "Glendale", "Gilbert", "Flagstaff", "Peoria"],
    "CT": ["Hartford", "New Haven", "Stamford", "Bridgeport", "Waterbury", "Norwalk", "Newington", "Danbury", "Meriden", "Bristol"],
    "LA": ["New Orleans", "Baton Rouge", "Shreveport", "Lafayette", "Lake Charles", "Kenner", "Metairie", "Bossier City", "Monroe", "Alexandria"],
    "NV": ["Las Vegas", "Henderson", "Reno", "North Las Vegas", "Sparks", "Carson City", "Boulder City", "Mesquite", "Elko", "Fernley"],
    "GA": ["Atlanta", "Decatur", "Savannah", "Augusta", "Macon", "Athens", "Marietta", "Roswell", "Albany", "Columbus"],
    "CO": ["Denver", "Aurora", "Colorado Springs", "Fort Collins", "Lakewood", "Boulder", "Pueblo", "Greeley", "Longmont", "Loveland"],
    "FL": ["Fort Lauderdale", "Miami", "Tampa", "Orlando", "Jacksonville", "St. Petersburg", "Hialeah", "Tallahassee", "Hollywood", "Gainesville"],
    "AK": ["Anchorage", "Fairbanks", "Juneau", "Wasilla", "Sitka", "Kenai", "Ketchikan", "Bethel", "Palmer", "Kodiak"],
    "NM": ["Albuquerque", "Santa Fe", "Las Cruces", "Rio Rancho", "Roswell", "Farmington", "Clovis", "Hobbs", "Carlsbad", "Alamogordo"],
    "UT": ["Salt Lake City", "Ogden", "Provo", "West Valley City", "Sandy", "Orem", "St. George", "Layton", "Logan", "Lehi"],
    "OK": ["Oklahoma City", "Tulsa", "Norman", "Edmond", "Lawton", "Broken Arrow", "Moore", "Stillwater", "Enid", "Midwest City"],
    "OR": ["Portland", "Salem", "Eugene", "Gresham", "Hillsboro", "Beaverton", "Bend", "Medford", "Springfield", "Corvallis"],
    "MO": ["St. Louis", "Kansas City", "Springfield", "Columbia", "Independence", "Lee's Summit", "O'Fallon", "St. Joseph", "Chesterfield", "Joplin"],
    "WV": ["Charleston", "Huntington", "Morgantown", "Parkersburg", "Wheeling", "Weirton", "Fairmont", "Martinsburg", "Beckley", "Clarksburg"],
}
ALL_STATES = list(CITIES.keys())
COUNTY_NAMES = ["Washington", "Jefferson", "Franklin", "Lincoln", "Madison", "Clark", "Union", "Jackson", "Montgomery", "Marion", "Fairfield", "Greene", "Monroe", "Wayne", "Lake", "Orange", "Summit", "Douglas", "Prince George", "Loudon"]
AGENCY_SUFFIX = {
    "City": ["City of {c}", "City of {c} Public Works Department", "City of {c} Capital Projects Division", "{c} Department of Engineering"],
    "County": ["{cty} County", "{cty} County Department of Public Works", "{cty} County Facilities Management", "{cty} County Transportation Department"],
    "State DOT": ["{st} Department of Transportation", "{st} DOT District {n}", "{st} State Highway Administration", "{st} Turnpike Authority"],
    "State Agency": ["{st} Department of General Services", "{st} School Building Authority", "{st} Facilities Construction Division", "{st} Capital Asset Management Agency"],
    "Transit Agency": ["{c} Metropolitan Transit Authority", "{c} Regional Transit District", "{c} Area Rapid Transit", "{st} Railway Express"],
    "K-12 School District": ["{c} Unified School District", "{c} Independent School District", "{c} Public Schools", "{cty} County School District"],
    "University/College": ["{st} State University", "University of {st} at {c}", "{c} Community College District", "{c} Technical College"],
    "Federal": ["U.S. General Services Administration Region {n}", "U.S. Army Corps of Engineers {c} District", "Federal Transit Administration", "Department of Veterans Affairs VISN {n}", "Smithsonian Institution", "U.S. Air Force Civil Engineer Center"],
    "Healthcare": ["{c} Health System", "{cty} County Hospital District", "{c} Health + Hospitals", "{c} Medical Center"],
    "Housing Authority": ["{c} Housing Authority", "Housing Authority of the City of {c}", "{cty} County Housing Authority"],
    "Airport/Port": ["{c} Airport Authority", "Port of {c}", "{c} International Airport", "{c} Port Authority"],
    "Utility/Water Authority": ["{c} Water District", "{cty} County Service Authority", "{c} Regional Sanitation District", "{c} Utilities Department"],
    "Nonprofit": ["{c} Science Center", "{c} Museum Foundation", "{c} Children's Hospital Foundation", "{c} Community Foundation"],
    "Private Company": ["{c} Energy", "{c} Data Centers LLC", "Northfield Aerospace", "Summit Grid Services", "Meridian Logistics", "Pacific Semiconductor Holdings"],
    "Other": ["{c} Recreation and Park District", "{c} Convention Facilities Authority", "{c} Redevelopment Agency"],
}
SECTOR_PROJECTS = {
    "Transportation/Highway": ["Route {n} Widening", "Bridge Rehabilitation Program", "Interchange Improvements at {c}", "Districtwide Roadway Resurfacing", "{c} Corridor Safety Improvements", "Pedestrian Bridge over Route {n}"],
    "Transit/Rail": ["Light Rail Extension Phase {p}", "Bus Rapid Transit Corridor", "Station Accessibility Upgrades", "Rail Yard Modernization", "Commuter Rail Platform Extensions", "Signal and Train Control Replacement"],
    "Aviation": ["Concourse {L} Expansion", "Runway {n}L/{n}R Rehabilitation", "Terminal Modernization Program", "Airfield Lighting Replacement", "Consolidated Rental Car Facility", "Spaceport Horizontal Launch Facility"],
    "K-12 Education": ["{c} High School Modernization", "Bond Program Phase {p} ({k} schools)", "New Elementary School No. {n}", "Districtwide HVAC Replacement", "Middle School Replacement", "Career Technical Education Center"],
    "Higher Education": ["Engineering Research Building", "Student Union Renovation", "Campus Utility Infrastructure", "Residence Hall Phase {p}", "Health Sciences Building", "Athletics Complex"],
    "Water/Wastewater": ["Wastewater Treatment Plant Upgrade", "{c} Water Reclamation Facility", "Sewer Main Rehabilitation Program", "Pump Station No. {n} Replacement", "Advanced Water Purification Facility", "Reservoir Seismic Retrofit"],
    "Healthcare": ["Hospital Tower Expansion", "Emergency Department Renovation", "Ambulatory Care Center", "Central Utility Plant Replacement", "Behavioral Health Pavilion", "Campus Transformation Program"],
    "Public Buildings/Facilities": ["Family Services Building and Parking Garage", "Public Safety Headquarters", "Convention Center Expansion", "Civic Center Renovation", "Courthouse Annex", "Fire Station No. {n}", "Library Modernization"],
    "Parks/Recreation": ["{c} Community Park Phase {p}", "Regional Aquatic Center", "Bay Trail Segment {n}", "Sports Complex", "Nature Center Renovation", "Trail and Bridge Connector"],
    "Housing": ["Public Housing Modernization Program", "{c} Houses Roof and Facade Repair", "Senior Housing Development", "Elevator Modernization at {k} Developments"],
    "Telecom/Fiber": ["Rural Fiber-to-the-Home Build", "Middle-Mile Fiber Backbone", "Data Center Campus Phase {p}", "Smart Grid Communications Network"],
    "Defense/Federal": ["Federal Office Building Repair and Alteration", "Reserve Center Design-Build", "VA Medical Center Renovation", "Courthouse Security Upgrades", "Base Infrastructure Program"],
    "Mixed/Multiple": ["Capital Improvement Program FY{y}", "On-Call Capital Projects", "Citywide Facilities Program", "Multi-Site Infrastructure Program"],
    "Other": ["Capital Projects Program", "Facilities Renewal Program", "Generation Facilities Upgrades"],
}
SERVICE_TITLES = {
    "Construction Management": ["Construction Management Services", "Construction Management and Inspection Services", "Professional Construction Management Services", "On-Call Construction Management Services"],
    "Program Management": ["Program Management Services", "Program Management Consultant", "Program Management Office (PMO) Support Services", "Bond Program Management Services"],
    "Project Management": ["Project Management Services", "Owner's Project Management Services", "Project Management and Oversight Services", "Capital Project Management Support"],
    "CM-at-Risk": ["Construction Manager at Risk Services", "CM/GC Preconstruction and Construction Services", "Construction Manager at Risk"],
    "Owner's Representative": ["Owner's Representative Services", "Owner's Agent Services", "Owner's Project Representative"],
    "Construction Inspection": ["Construction Engineering and Inspection Services", "Construction Inspection Services", "Construction Management and Inspection (CMI) Services", "Resident Engineering and Inspection Services"],
    "Design-Build": ["Design-Build Services", "Progressive Design-Build Services"],
    "Engineering Staffing": ["Engineering Staff Augmentation Services"],
    "Other": ["Professional Construction Services", "Architectural and Engineering Services"],
}
ROLE_POOL = {
    "Construction Management": ["Project Manager", "Construction Manager", "Resident Engineer", "Office Engineer", "Construction Inspector", "Scheduler", "Cost Estimator", "Document Control Specialist", "QA/QC Manager", "Safety Manager", "Claims Specialist", "Commissioning Agent"],
    "Program Management": ["Program Director", "Deputy Program Manager", "Program Controls Manager", "Senior Project Manager", "Scheduler", "Cost Estimator", "Risk Manager", "Community Outreach Liaison", "Document Control Specialist", "QA Manager", "BIM Manager"],
    "Project Management": ["Project Manager", "Assistant Project Manager", "Project Engineer", "Scheduler", "Cost Estimator", "Contract Administrator", "Project Coordinator"],
    "CM-at-Risk": ["Project Executive", "Project Manager", "Preconstruction Manager", "Superintendent", "Cost Estimator", "Scheduler", "Safety Manager", "MEP Coordinator"],
    "Owner's Representative": ["Principal-in-Charge", "Owner's Representative", "Project Manager", "Cost Estimator", "Scheduler", "Commissioning Agent"],
    "Construction Inspection": ["Responsible Charge Engineer", "Resident Engineer", "Senior Construction Inspector", "Construction Inspector", "Inspector Trainee", "Materials Technician", "Office Engineer", "Scheduler", "Administrative Assistant"],
    "Design-Build": ["Design-Build Project Manager", "Design Manager", "Construction Manager", "Superintendent", "Quality Control Manager", "Safety Manager", "Scheduler"],
    "Engineering Staffing": ["Senior Civil Engineer", "Civil Engineer", "Engineering Technician", "Construction Inspector"],
    "Other": ["Project Manager", "Construction Manager", "Scheduler", "Cost Estimator"],
}
SOFTWARE_POOL = ["Primavera P6", "MS Project", "e-Builder", "Procore", "PMWeb", "Bluebeam", "SiteManager", "Proliance", "Unifier", "Revit/BIM", "AutoCAD", "ArcGIS", "Kahua", "Aconex", "SharePoint", "Excel"]
CRITERIA_FIELDS = [
    ("weight_firm_experience", "Firm Experience and Past Performance"),
    ("weight_key_personnel", "Qualifications of Key Personnel"),
    ("weight_approach", "Project Understanding and Approach"),
    ("weight_references_past_perf", "Client References"),
    ("weight_capacity_local", "Capacity, Availability and Local Presence"),
    ("weight_price", "Price / Fee Proposal"),
    ("weight_dbe_participation", "DBE / Small Business Participation"),
    ("weight_interview", "Oral Interview / Presentation"),
]
BASE_WEIGHTS = {
    "QBS": [25, 35, 20, 10, 10, 0, 0, 0],
    "Two-step shortlist+interview": [20, 30, 20, 10, 5, 0, 0, 15],
    "Best Value": [20, 25, 20, 10, 5, 15, 5, 0],
    "Lowest Price Technically Acceptable": [15, 25, 20, 0, 0, 40, 0, 0],
}
FIRM_A = ["Harland", "Beckett", "Northgate", "Alder", "Stonebridge", "Meridian", "Vantage", "Kessler", "Pinnacle", "Castillo", "Rowan", "Ironwood", "Summit", "Keystone", "Granite", "Blueline", "Cardinal", "Tidewater", "Sierra", "Redwood", "Lakeshore", "Horizon", "Atlas", "Crescent", "Everett", "Falcon", "Greenfield", "Hollis", "Juniper", "Kinsley", "Linden", "Marlow", "Norwood", "Oakridge", "Prescott", "Quarry", "Ridgeline", "Sutton", "Tremont", "Upland", "Vera", "Westbrook", "Yardley", "Zephyr", "Ashford", "Brightwater", "Coastal", "Delta", "Eastman", "Foxhall"]
FIRM_B = ["Engineering", "Construction Services", "Program Management", "Consulting Engineers", "Infrastructure Group", "& Associates", "Partners", "Project Services", "CM Group", "Technical Services"]
FIRM_SUFFIX = ["Inc.", "LLC", "PC", "Corp.", "LLP", ""]
JOB_LEVEL_SALARY = {  # annual USD midpoint ranges
    "Entry": (55000, 75000), "Mid": (70000, 100000), "Senior": (95000, 140000),
    "Manager": (115000, 165000), "Director": (150000, 200000), "Executive": (190000, 260000),
}
BENEFITS = ["Medical/Dental/Vision", "Pension", "401k/457 Match", "Tuition Assistance", "Transit Pass", "Paid Parental Leave", "Relocation", "Vehicle Allowance", "Flexible Schedule", "EAP"]
DEGREE_FIELDS = ["Civil Engineering", "Construction Management", "Architecture", "Engineering (any)", "Business Administration", "Construction Engineering", "Mechanical Engineering", "Electrical Engineering"]


# --------------------------------------------------------------------------- helpers
def rnd_date(y0=YEAR_MIN, y1=YEAR_MAX):
    start = date(y0, 1, 1)
    end = date(y1, 12, 31)
    return start + timedelta(days=rng.randint(0, (end - start).days))


def lognormal_between(lo, hi):
    mu = (math.log(lo) + math.log(hi)) / 2
    sigma = (math.log(hi) - math.log(lo)) / 4
    v = math.exp(rng.gauss(mu, sigma))
    return max(lo, min(hi, v))


def round_money(v):
    if v >= 1e8: return round(v, -6)
    if v >= 1e7: return round(v, -5)
    if v >= 1e6: return round(v, -4)
    return round(v, -3)


def pick(lst):
    return rng.choice(lst)


def norm100(ws):
    ws = [max(0.0, w) for w in ws]
    s = sum(ws) or 1
    ints = [int(round(w / s * 100)) for w in ws]
    diff = 100 - sum(ints)
    ints[ints.index(max(ints))] += diff
    return ints


def cert_level(certs, key, p_required, p_pref):
    txt = " ".join(certs).upper()
    if key in txt:
        if "PREFER" in txt or "DESIR" in txt:
            return "Preferred" if rng.random() < 0.75 else "Required"
        return "Required" if rng.random() < 0.7 else "Preferred"
    r = rng.random()
    return "Required" if r < p_required else ("Preferred" if r < p_required + p_pref else "Not mentioned")


def nz(v, default):
    return default if v in (None, "", "Not stated") else v


class FirmPool:
    def __init__(self):
        self.firms = {}
        names = set()
        while len(names) < 220:
            names.add(f"{pick(FIRM_A)} {pick(FIRM_B)} {pick(FIRM_SUFFIX)}".strip())
        for n in names:
            size = rng.choices(["Small", "Mid", "Large", "National"], [0.38, 0.32, 0.2, 0.1])[0]
            self.firms[n] = {"size": size, "hq": pick(ALL_STATES), "dbe": rng.random() < (0.45 if size == "Small" else 0.08)}
        self.names = list(self.firms)

    def sample(self, k):
        return rng.sample(self.names, k)


# --------------------------------------------------------------------------- generators
def agency_name(org_type, state, city):
    tpl = pick(AGENCY_SUFFIX.get(org_type, AGENCY_SUFFIX["Other"]))
    return tpl.format(c=city, cty=pick(COUNTY_NAMES), st=state, n=rng.randint(1, 12))


def project_name(sector, city, year):
    tpl = pick(SECTOR_PROJECTS.get(sector, SECTOR_PROJECTS["Other"]))
    return tpl.format(c=city, n=rng.randint(1, 99), p=pick(["I", "II", "III", "IV"]), L=pick("ABCDE"), k=rng.randint(3, 40), y=str(year)[2:])


def sol_number(org_type, year):
    f = rng.random()
    if org_type == "Federal":
        return f"{rng.choice(['GS', 'W912', 'VA', '36C'])}{rng.randint(10, 99)}-{year % 100}-R-{rng.randint(1000, 9999)}"
    if f < 0.4: return f"RFP {year % 100}-{rng.randint(1, 99):02d}-{rng.randint(100, 999)}"
    if f < 0.7: return f"RFQ No. {rng.randint(1000, 9999)}-{year % 100}"
    if f < 0.85: return f"BCS {year}-{rng.randint(1, 60):02d}"
    return f"PIN {rng.randint(8500000, 8600000)}{pick('PQRV')}{rng.randint(1, 99):04d}"


def gen_solicitations(doc, doc_id, firms, sol_rows, crit_rows, prop_rows, start_idx):
    org_type = nz(doc.get("org_type"), "City")
    if org_type == "Private Company": org_type = "Other"
    state0 = doc.get("state") if doc.get("state") in CITIES else None
    sector = nz(doc.get("sector"), "Public Buildings/Facilities")
    service = nz(doc.get("service_category"), "Construction Management")
    if service == "Engineering Staffing": service = "Construction Inspection"
    doc_type0 = doc.get("doc_type") or "RFP"
    if doc_type0 not in ("RFP", "RFQ", "RFQ/RFP (two-step)", "IFB"): doc_type0 = None
    sel0 = (doc.get("selection_method") or "")
    sel0 = "QBS" if sel0.startswith("QBS") else "Two-step shortlist+interview" if sel0.startswith("Two-step") else "Best Value" if sel0.startswith("Best") else None
    comp0 = doc.get("compensation_type") if doc.get("compensation_type") in ("Hourly Rate Schedule", "Lump Sum", "Time & Materials", "Cost Plus Fixed Fee") else None
    deliv0 = doc.get("delivery_method") if doc.get("delivery_method") in ("Design-Bid-Build", "CMAR", "Design-Build", "Multiple", "Staff Augmentation") else None
    certs = doc.get("required_licenses_certs") or []
    src_roles = [r for r in (doc.get("key_personnel_roles") or []) if r]
    src_sw = [s for s in (doc.get("software_tools_mentioned") or []) if s in SOFTWARE_POOL]
    src_cv = doc.get("estimated_construction_value_usd")
    src_fee = doc.get("estimated_contract_fee_usd")
    ins = doc.get("insurance_requirements") or {}
    sb = doc.get("small_business_goals") or {}
    src_weights = None
    ec = [c for c in (doc.get("evaluation_criteria") or []) if isinstance(c.get("weight_pct_or_points"), (int, float))]
    if len(ec) >= 3:
        # map source criteria loosely onto the 8 standard buckets
        src_weights = [0.0] * 8
        for c in ec:
            n = (c["criterion"] or "").lower(); w = float(c["weight_pct_or_points"])
            i = 1 if any(k in n for k in ("personnel", "staff", "team", "manager")) else \
                5 if any(k in n for k in ("price", "fee", "cost")) else \
                7 if "interview" in n or "oral" in n else \
                6 if any(k in n for k in ("dbe", "mbe", "small", "local business")) else \
                3 if "reference" in n or "past perf" in n else \
                4 if any(k in n for k in ("capacity", "availab", "local", "workload")) else \
                2 if any(k in n for k in ("approach", "understand", "plan", "method", "capab")) else 0
            src_weights[i] += w
    min_years0 = doc.get("years_experience_required")
    issue0 = (doc.get("dates") or {}).get("issue_date") or (doc.get("dates") or {}).get("proposal_due")
    year0 = int(issue0[:4]) if issue0 and issue0[:4].isdigit() and YEAR_MIN <= int(issue0[:4]) <= YEAR_MAX else None

    for i in range(ROWS_PER_DOC):
        idx = start_idx + i
        sid = f"SOL-{idx:06d}"
        state = state0 if (state0 and rng.random() < 0.7) else pick(ALL_STATES)
        city = pick(CITIES[state])
        # year: cluster around the source year
        year = year0 if year0 and rng.random() < 0.35 else max(YEAR_MIN, min(YEAR_MAX, (year0 or 2016) + rng.randint(-6, 6)))
        issue = rnd_date(year, year)
        doc_type = doc_type0 if doc_type0 and rng.random() < 0.8 else rng.choices(["RFP", "RFQ", "RFQ/RFP (two-step)", "IFB"], [0.55, 0.25, 0.17, 0.03])[0]
        sel = sel0 if sel0 and rng.random() < 0.75 else rng.choices(list(BASE_WEIGHTS), [0.3, 0.4, 0.25, 0.05])[0]
        if doc_type == "IFB": sel = "Lowest Price Technically Acceptable"
        comp = comp0 if comp0 and rng.random() < 0.7 else rng.choices(["Hourly Rate Schedule", "Time & Materials NTE", "Lump Sum", "Cost Plus Fixed Fee", "Percentage Fee"], [0.35, 0.25, 0.18, 0.14, 0.08])[0]
        if comp == "Time & Materials": comp = "Time & Materials NTE"
        deliv = deliv0 if deliv0 and rng.random() < 0.75 else rng.choices(["Design-Bid-Build", "CMAR", "Design-Build", "Multiple", "Staff Augmentation"], [0.4, 0.15, 0.1, 0.2, 0.15])[0]
        vehicle = "On-Call/IDIQ" if deliv == "Staff Augmentation" or (service == "Construction Inspection" and rng.random() < 0.6) else \
                  "Program-wide" if service == "Program Management" or rng.random() < 0.08 else "Single Project"
        fed = org_type == "Federal" or (sector in ("Transportation/Highway", "Transit/Rail", "Aviation") and rng.random() < 0.6) or rng.random() < 0.15
        funding = "Federal-Other" if org_type == "Federal" else \
                  rng.choices(["Federal-FTA", "Federal-FHWA", "Mixed"], [0.4, 0.4, 0.2])[0] if fed and sector in ("Transportation/Highway", "Transit/Rail") else \
                  rng.choices(["Federal-Other", "Mixed"], [0.5, 0.5])[0] if fed else \
                  rng.choices(["Local", "State", "Bond", "Grant"], [0.4, 0.2, 0.3, 0.1])[0]
        if sector in ("K-12 Education", "Higher Education") and not fed: funding = rng.choices(["Bond", "Local", "State"], [0.6, 0.2, 0.2])[0]
        # value
        if vehicle == "Program-wide":
            cv = lognormal_between(1e8, 8e9)
        elif src_cv and rng.random() < 0.6:
            cv = src_cv * rng.uniform(0.5, 1.8)
        else:
            lo, hi = {"Transportation/Highway": (1e6, 2e8), "Transit/Rail": (1e7, 2e9), "Aviation": (5e6, 1e9), "K-12 Education": (5e6, 6e8),
                      "Higher Education": (5e6, 4e8), "Water/Wastewater": (2e6, 1e9), "Healthcare": (5e6, 1.5e9), "Public Buildings/Facilities": (2e6, 5e8),
                      "Parks/Recreation": (5e5, 6e7), "Housing": (2e6, 2e8), "Telecom/Fiber": (5e6, 4e8), "Defense/Federal": (2e6, 3e8),
                      "Mixed/Multiple": (5e6, 5e8), "Other": (1e6, 2e8)}[sector]
            cv = lognormal_between(lo, hi)
        if vehicle != "Program-wide":
            cv = min(cv, 1.5e9 if sector in ("Transit/Rail", "Healthcare", "Aviation", "Water/Wastewater") else 4e8)
        cv = round_money(cv)
        if vehicle == "On-Call/IDIQ" and service == "Construction Inspection":
            fee = round_money((src_fee or 1e6) * rng.uniform(0.4, 2.5)) if src_fee else round_money(lognormal_between(2.5e5, 5e6))
            fee_pct = None
        else:
            fee_pct = round(rng.uniform({"Program Management": 2.0, "Construction Inspection": 4.0, "CM-at-Risk": 2.5}.get(service, 1.5),
                                        {"Program Management": 5.0, "Construction Inspection": 10.0, "CM-at-Risk": 5.0}.get(service, 4.5)), 2)
            fee = round_money(cv * fee_pct / 100)
        base_term = rng.choices([12, 18, 24, 36, 48, 60], [0.25, 0.1, 0.2, 0.3, 0.05, 0.1])[0]
        if vehicle == "Program-wide": base_term = rng.choice([36, 48, 60])
        options = rng.choices([0, 1, 2, 3, 4], [0.3, 0.2, 0.25, 0.1, 0.15])[0]
        # dates
        pre_prop = issue + timedelta(days=rng.randint(7, 14)) if rng.random() < 0.75 else None
        pre_mand = bool(pre_prop and rng.random() < 0.3)
        due = issue + timedelta(days=rng.randint(14, 45))
        qdue = due - timedelta(days=rng.randint(5, 10))
        award = due + timedelta(days=rng.randint(30, 150))
        # weights
        bw = src_weights if (src_weights and rng.random() < 0.65) else BASE_WEIGHTS[sel]
        ws = norm100([w * rng.uniform(0.7, 1.3) if w > 0 else (rng.uniform(0, 6) if rng.random() < 0.15 else 0) for w in bw])
        if sel in ("QBS", "Two-step shortlist+interview") and rng.random() < 0.85: ws[5] = 0; ws = norm100(ws)
        # certs
        ccm = cert_level(certs, "CCM", *{"Construction Management": (0.3, 0.4), "Construction Inspection": (0.1, 0.3), "Program Management": (0.2, 0.35)}.get(service, (0.15, 0.35)))
        pe = cert_level(certs, "PE", 0.25 if service == "Construction Inspection" else 0.12, 0.3)
        pmp = cert_level(certs, "PMP", 0.05, 0.25)
        leed = cert_level(certs, "LEED", 0.03, 0.2)
        osha = cert_level(certs, "OSHA", 0.15, 0.2)
        st_cert = service == "Construction Inspection" and rng.random() < 0.8
        roles = list(dict.fromkeys((src_roles if rng.random() < 0.5 else []) + rng.sample(ROLE_POOL[service], k=rng.randint(3, min(7, len(ROLE_POOL[service]))))))[:8]
        sw = list(dict.fromkeys(src_sw + rng.sample(SOFTWARE_POOL, k=rng.randint(0, 3))))[:4]
        fte = round(lognormal_between(1.5, 40 if vehicle != "Program-wide" else 120), 1)
        gl = nz(ins.get("general_liability_usd"), None) if rng.random() < 0.6 else None
        gl = gl or rng.choices([500000, 1000000, 2000000], [0.1, 0.6, 0.3])[0]
        pl = (nz(ins.get("professional_liability_usd"), None) if rng.random() < 0.6 else None) or rng.choices([1000000, 2000000, 3000000, 5000000], [0.5, 0.3, 0.1, 0.1])[0]
        auto = rng.choices([500000, 1000000, 2000000], [0.1, 0.7, 0.2])[0]
        umb = rng.choices([None, 1000000, 2000000, 5000000, 10000000], [0.45, 0.15, 0.2, 0.15, 0.05])[0]
        dbe = (sb.get("dbe_pct") if isinstance(sb.get("dbe_pct"), (int, float)) and rng.random() < 0.5 else None)
        if dbe is None and fed and rng.random() < 0.7: dbe = rng.choice([5, 7, 8, 10, 12, 15, 18, 20, 23])
        mbe = rng.choice([None, None, 10, 15, 20, 25, 30]) if not fed and org_type in ("City", "County", "Housing Authority", "Transit Agency") else None
        wbe = rng.choice([None, 5, 10, 15]) if mbe else None
        local_pref = rng.choice([None, None, None, 5, 10, 15]) if org_type in ("City", "County") else None
        setaside = rng.choices(["None", "SB", "SDVOSB", "8(a)", "HUBZone"], [0.6, 0.2, 0.1, 0.05, 0.05])[0] if org_type == "Federal" else "None"
        pw = bool(fed or state in ("CA", "NY", "WA", "OR", "MA", "NJ", "IL") or rng.random() < 0.3)
        bond = deliv in ("CMAR", "Design-Build") or rng.random() < 0.1
        # outcomes
        n_prop = rng.choices(range(2, 16), [3, 6, 9, 11, 12, 11, 9, 8, 6, 4, 3, 2, 1, 1])[0]
        n_short = min(n_prop, rng.choice([3, 3, 4, 5])) if sel in ("Two-step shortlist+interview", "Best Value") else (n_prop if sel == "QBS" and rng.random() < 0.5 else min(n_prop, 3))
        interviews = sel == "Two-step shortlist+interview" or rng.random() < 0.3
        status = rng.choices(["Awarded", "Cancelled", "Protested", "Re-bid", "Open", "Shortlisting"], [0.78, 0.06, 0.04, 0.04, 0.05, 0.03])[0]
        if issue.year >= YEAR_MAX and rng.random() < 0.5: status = rng.choice(["Open", "Shortlisting"])
        firm_names = firms.sample(n_prop)
        awarded_firm = None; award_amt = None; size = None; incumbent = None; avp = None
        if status in ("Awarded", "Protested"):
            awarded_firm = firm_names[0]; size = firms.firms[awarded_firm]["size"]
            award_amt = round_money(fee * rng.uniform(0.82, 1.18)); avp = round((award_amt / fee - 1) * 100, 1)
            incumbent = rng.random() < 0.3
        days_award = (award - issue).days if awarded_firm else None
        pname = (pick(["Capital Improvement Program FY{y}", "{s} Bond Program Phase {p}", "Capital Delivery Program ({k} projects)", "Systemwide {s} Modernization Program", "Multi-Year {s} Investment Program"]).format(y=str(year)[2:], p=pick(["I", "II", "III"]), k=rng.randint(8, 120), s=sector.split("/")[0])
                 if vehicle == "Program-wide" else project_name(sector, city, year))
        agency = agency_name(org_type, state, city)
        svc_title = pick(SERVICE_TITLES.get(service, SERVICE_TITLES['Other']))
        title = f"{svc_title} for {pname}"
        # quirks (~3%)
        quirk = rng.random() < 0.03
        if quirk:
            q = rng.random()
            if q < 0.4 and doc_type == "RFP": doc_type = "RFQ"
            elif q < 0.7: award = due - timedelta(days=rng.randint(1, 10))
            else: title = title.replace(" for ", " for the ")
        sol_rows.append([sid, doc_id, doc["file"], sol_number(org_type, year), doc_type, title, agency, org_type, state,
                         STATE_REGION.get(state, "West"), city, sector, service, deliv, vehicle, pname,
                         f"{agency} seeks a qualified firm to provide {svc_title.lower()} for the {pname}" + (f" with an estimated construction value of ${cv/1e6:,.1f}M." if vehicle != "On-Call/IDIQ" else f" on a task-order basis with an annual not-to-exceed amount of ${fee/1e6:,.2f}M."),
                         funding, fed, int(cv), int(fee), fee_pct, base_term, options, comp, sel, issue, pre_prop, pre_mand, qdue, due, award, (due - issue).days,
                         rng.choices(["Hard copy", "Electronic", "Both"], [0.35 if year < 2018 else 0.1, 0.2 if year < 2018 else 0.6, 0.45 if year < 2018 else 0.3])[0],
                         rng.choice([0, 0, 1, 3, 4, 5, 6, 7]), rng.choice([None, None, 8, 10, 15, 20, 25, 30, 40, 50]), org_type == "Federal" or rng.random() < 0.15,
                         rng.choices([0, 1, 2, 3, 4, 5], [0.25, 0.3, 0.2, 0.12, 0.08, 0.05])[0], rng.randint(0, 60), *ws, (min_years0 if isinstance(min_years0, (int, float)) and rng.random() < 0.5 else rng.choices([5, 7, 8, 10, 12, 15, 20], [0.1, 0.1, 0.1, 0.4, 0.1, 0.15, 0.05])[0]),
                         ccm, pe, pmp, leed, osha, st_cert, " | ".join(roles), fte, " | ".join(sw), gl, pl, auto, umb, dbe, mbe, wbe, local_pref, setaside, pw, bond,
                         rng.choice([5, 7, 10, 10, 14]), n_prop, n_short, interviews, awarded_firm, size, int(award_amt) if award_amt else None, avp, incumbent, status, days_award])
        # criteria
        for k, ((fld, name), w) in enumerate(zip(CRITERIA_FIELDS, ws), 1):
            if w > 0:
                crit_rows.append([f"{sid}-C{k}", sid, k, name, w, "Interview" if fld == "weight_interview" else ("Both" if interviews and fld in ("weight_key_personnel", "weight_approach") else "Written"), False])
        if rng.random() < 0.5:
            crit_rows.append([f"{sid}-C9", sid, 9, pick(["Responsiveness and Completeness", "Insurance and Licensing Compliance", "Conflict of Interest Certification", "Financial Capacity"]), 0, "Written", True])
        # proposals
        props = []
        for j, fn in enumerate(firm_names):
            f = firms.firms[fn]
            pm_years = max(3, int(rng.gauss(12, 4)))
            pm_ccm = rng.random() < 0.55; pm_pe = rng.random() < 0.45
            quality = rng.gauss(0.72, 0.1) + (0.05 if pm_ccm else 0) + (0.04 if f["size"] in ("Large", "National") else 0) + (0.02 if f["hq"] == state else 0)
            quality = max(0.35, min(0.98, quality))
            tech_scores = [round(w * max(0.3, min(1.0, rng.gauss(quality, 0.08))), 1) if w > 0 else 0.0 for w in ws]
            prop_fee = round_money(fee * rng.uniform(0.8, 1.25)) if (ws[5] > 0 or sel == "Best Value" or rng.random() < 0.3) else None
            props.append([f"{sid}-P{j+1:02d}", sid, fn, f["size"], f["hq"], f["hq"] == state, f["dbe"], False, rng.randint(0, 6),
                          round(rng.uniform(0, max(dbe or 0, 10) * 1.3), 1) if (dbe or mbe) else None, pm_years, pm_ccm, pm_pe, round(fte * rng.uniform(0.8, 1.2), 1),
                          int(prop_fee) if prop_fee else None, tech_scores, rng.random() < 0.96])
        # price score: cheapest gets full points
        fees = [p[14] for p in props if p[14]]
        for p in props:
            ts = p[15]
            if ws[5] > 0 and p[14] and fees:
                ts[5] = round(ws[5] * min(fees) / p[14], 1)
            if not p[16]:
                ts = [0.0] * 8
            tech_total = round(sum(ts[k] for k in range(8) if k != 5), 1)
            final = round(tech_total + ts[5], 1)
            p[15] = ts; p.append(tech_total); p.append(final)
        props.sort(key=lambda p: -p[-1])
        if awarded_firm:  # make the pre-chosen awardee rank first
            w_i = next(k for k, p in enumerate(props) if p[2] == awarded_firm)
            props.insert(0, props.pop(w_i))
            if props[0][-1] < props[1][-1]:
                props[0][-1] = round(props[1][-1] + rng.uniform(0.5, 4), 1); props[0][-2] = round(props[0][-1] - props[0][15][5], 1)
        for r, p in enumerate(props, 1):
            short = r <= n_short and p[16]
            p[7] = bool(incumbent) and r == 1
            prop_rows.append(p[:15] + p[15] + [p[-2], p[-1], r, short, short and interviews, awarded_firm == p[2], status == "Protested" and r == 2, p[16]])
    return start_idx + ROWS_PER_DOC


def gen_jobs(doc, doc_id, job_rows, start_idx):
    jp = doc.get("job_posting") or {}
    title0 = doc.get("title") or "Project Manager"
    t = title0.lower()
    level0 = "Executive" if any(k in t for k in ("assistant general manager", "chief", "vice president")) else \
             "Director" if "director" in t or "manager of" in t or "field manager" in t else \
             "Manager" if ("manager" in t and "project manager" not in t and "construction manager" not in t) else \
             "Senior" if any(k in t for k in ("senior", "sr.", "sr ", " ii", "staff")) else "Mid"
    fam0 = {"Construction Management": "Construction Management", "Program Management": "Program Management", "Project Management": "Project Management",
            "Engineering Staffing": "Engineering", "Construction Inspection": "Inspection", "Owner's Representative": "Project Management"}.get(doc.get("service_category"), "Construction Management")
    org0 = nz(doc.get("org_type"), "City"); sector0 = nz(doc.get("sector"), "Public Buildings/Facilities")
    state0 = doc.get("state") if doc.get("state") in CITIES else None
    smin, smax = jp.get("salary_min_usd"), jp.get("salary_max_usd")
    basis = (jp.get("pay_basis") or "").lower()
    if smin and smax:
        if "hour" in basis: smin, smax = smin * 2080, smax * 2080
        elif "biweek" in basis: smin, smax = smin * 26, smax * 26
    certs = doc.get("required_licenses_certs") or []
    yrs0 = doc.get("years_experience_required")
    issue0 = (doc.get("dates") or {}).get("issue_date") or (doc.get("dates") or {}).get("proposal_due")
    year0 = int(issue0[:4]) if issue0 and issue0[:4].isdigit() and YEAR_MIN <= int(issue0[:4]) <= YEAR_MAX else None
    TITLES = {"Construction Management": ["Construction Manager", "Senior Construction Manager", "Construction Project Manager", "Field Construction Manager", "Construction Services Manager"],
              "Program Management": ["Program Manager", "Manager of Engineering Programs", "Director of Capital Programs", "Assistant General Manager, Capital Delivery", "Program Controls Manager"],
              "Project Management": ["Project Manager", "Project Manager II", "Senior Project Manager", "Capital Projects Manager", "Project Manager, Sr."],
              "Engineering": ["Senior Civil Engineer", "Civil Engineer III", "Staff Civil Engineer", "Engineering Supervisor", "Resident Engineer"],
              "Inspection": ["Construction Inspector", "Senior Construction Inspector", "Chief Inspector", "Field Engineer", "Resident Inspector"]}
    for i in range(ROWS_PER_DOC):
        pid = f"JOB-{start_idx + i:06d}"
        state = state0 if (state0 and rng.random() < 0.7) else pick(ALL_STATES)
        city = pick(CITIES[state])
        year = year0 if year0 and rng.random() < 0.35 else max(YEAR_MIN, min(YEAR_MAX, (year0 or 2018) + rng.randint(-5, 5)))
        fam = fam0 if rng.random() < 0.8 else pick(list(TITLES))
        level = level0 if rng.random() < 0.6 else rng.choices(["Entry", "Mid", "Senior", "Manager", "Director", "Executive"], [0.08, 0.3, 0.3, 0.17, 0.1, 0.05])[0]
        title = title0.split(" - ")[0].strip() if rng.random() < 0.3 else pick(TITLES[fam])
        org = org0 if rng.random() < 0.7 else rng.choices(["City", "County", "Transit Agency", "University/College", "Healthcare", "Private Company", "Utility/Water Authority", "Airport/Port", "State DOT"], [0.25, 0.1, 0.12, 0.1, 0.1, 0.15, 0.08, 0.05, 0.05])[0]
        sector = sector0 if org == org0 else {"Transit Agency": "Transit/Rail", "Airport/Port": "Aviation", "Healthcare": "Healthcare", "University/College": "Higher Education", "Utility/Water Authority": "Water/Wastewater", "State DOT": "Transportation/Highway", "K-12 School District": "K-12 Education"}.get(org, pick(["Public Buildings/Facilities", "Transportation/Highway", "Mixed/Multiple", "Parks/Recreation", "Water/Wastewater"]))
        lo, hi = JOB_LEVEL_SALARY[level]
        if smin and smax and rng.random() < 0.5:
            s_lo, s_hi = smin * rng.uniform(0.9, 1.1), smax * rng.uniform(0.9, 1.1)
        else:
            s_lo = rng.uniform(lo, hi * 0.85); s_hi = s_lo * rng.uniform(1.25, 1.6)
        cola = {"CA": 1.15, "NY": 1.15, "MA": 1.12, "WA": 1.1, "DC": 1.12, "MD": 1.05, "VA": 1.03}.get(state, 0.95)
        s_lo, s_hi = round(s_lo * cola, -2), round(s_hi * cola, -2)
        if org == "Private Company": s_lo, s_hi = round(s_lo * 1.1, -2), round(s_hi * 1.15, -2)
        pay_basis = rng.choices(["Annual", "Hourly", "Biweekly"], [0.75, 0.15, 0.1])[0]
        emp = rng.choices(["Full-time", "Part-time", "Contract", "Bench/On-call"], [0.85, 0.02, 0.08, 0.05])[0]
        bill = round(s_hi / 2080 * rng.uniform(2.6, 3.2), 0) if emp in ("Contract", "Bench/On-call") else None
        min_yrs = yrs0 if isinstance(yrs0, (int, float)) and rng.random() < 0.5 else {"Entry": rng.randint(0, 2), "Mid": rng.randint(3, 6), "Senior": rng.randint(6, 10), "Manager": rng.randint(8, 12), "Director": rng.randint(10, 15), "Executive": rng.randint(15, 20)}[level]
        post = rnd_date(year, year)
        close = post + timedelta(days=rng.randint(14, 60))
        ouf = rng.random() < 0.35
        filled = rng.random() < 0.8 and post.year < YEAR_MAX
        job_rows.append([pid, doc_id, doc["file"], title, fam, level, agency_name(org, state, city), org, sector, state, STATE_REGION.get(state, "West"), city, emp,
                         level != "Entry", org in ("City", "County", "Transit Agency", "Utility/Water Authority") and rng.random() < 0.4,
                         rng.choices(["Onsite", "Field", "Hybrid"], [0.55, 0.3, 0.15])[0], rng.choice([0, 0, 10, 25, 25, 50]), int(s_lo), int(s_hi), pay_basis, bill, int(min_yrs),
                         rng.choices(["HS", "Associate", "Bachelor's", "Master's"], {"Entry": [0.15, 0.25, 0.6, 0.0], "Mid": [0.05, 0.1, 0.8, 0.05], "Senior": [0.0, 0.05, 0.8, 0.15], "Manager": [0.0, 0.02, 0.73, 0.25], "Director": [0.0, 0.0, 0.65, 0.35], "Executive": [0.0, 0.0, 0.5, 0.5]}[level])[0], pick(DEGREE_FIELDS),
                         cert_level(certs, "CCM", 0.1, 0.4), cert_level(certs, "PE", 0.2 if fam == "Engineering" else 0.08, 0.3), cert_level(certs, "PMP", 0.05, 0.3),
                         cert_level(certs, "LEED", 0.02, 0.2), cert_level(certs, "OSHA", 0.1, 0.25),
                         {"Entry": 0, "Mid": rng.randint(0, 2), "Senior": rng.randint(0, 5), "Manager": rng.randint(3, 12), "Director": rng.randint(8, 40), "Executive": rng.randint(50, 300)}[level],
                         pick(["Director of Capital Projects", "Chief Engineer", "Deputy Director, Public Works", "Program Manager", "Director of Construction Management", "Chief of Capital Programs", "City Engineer"]),
                         int(round_money(lognormal_between(2e6, 8e9 if level == "Executive" else 5e8 if level in ("Director", "Manager") else 1e8))),
                         post, None if ouf else close, ouf, rng.random() < 0.1, rng.random() < 0.85, rng.random() < 0.6,
                         " | ".join(rng.sample(BENEFITS, k=rng.randint(2, 6))), rng.randint(5, 180), rng.randint(30, 180) if filled else None, filled])
    return start_idx + ROWS_PER_DOC


# --------------------------------------------------------------------------- main
SOL_COLS = ["solicitation_id", "source_doc_id", "source_file", "solicitation_number", "doc_type", "title", "issuing_org", "org_type", "state", "region", "city", "sector", "service_category", "delivery_method", "contract_vehicle", "project_name", "scope_summary", "funding_source", "federally_funded", "est_construction_value_usd", "est_fee_usd", "fee_pct_of_construction", "base_term_months", "option_years", "compensation_type", "selection_method", "issue_date", "pre_proposal_date", "pre_proposal_mandatory", "questions_due", "proposal_due", "anticipated_award_date", "response_window_days", "submission_format", "hard_copies_required", "page_limit", "sf330_required", "num_addenda", "num_questions_received", "weight_firm_experience", "weight_key_personnel", "weight_approach", "weight_references_past_perf", "weight_capacity_local", "weight_price", "weight_dbe_participation", "weight_interview", "min_years_lead_pm", "ccm_requirement", "pe_requirement", "pmp_requirement", "leed_requirement", "osha_requirement", "state_inspector_cert_required", "key_roles_required", "est_fte", "software_required", "gl_limit_usd", "pl_limit_usd", "auto_limit_usd", "umbrella_limit_usd", "dbe_goal_pct", "mbe_goal_pct", "wbe_goal_pct", "local_preference_pct", "small_business_setaside", "prevailing_wage", "bonding_required", "protest_period_days", "num_proposals_received", "num_shortlisted", "interviews_held", "awarded_firm", "awarded_firm_size", "awarded_amount_usd", "award_vs_estimate_pct", "incumbent_awarded", "status", "days_issue_to_award"]
CRIT_COLS = ["criterion_id", "solicitation_id", "criterion_order", "criterion_name", "weight_points", "scoring_phase", "pass_fail"]
PROP_COLS = ["proposal_id", "solicitation_id", "firm_name", "firm_size", "firm_hq_state", "is_local", "dbe_certified", "is_incumbent", "team_subconsultants", "proposed_dbe_pct", "proposed_pm_years", "proposed_pm_has_ccm", "proposed_pm_has_pe", "proposed_fte", "proposed_fee_usd", "score_firm_experience", "score_key_personnel", "score_approach", "score_references", "score_capacity", "score_price", "score_dbe", "score_interview", "technical_score_total", "final_score", "rank", "shortlisted", "interviewed", "awarded", "protest_filed", "responsive"]
JOB_COLS = ["posting_id", "source_doc_id", "source_file", "job_title", "job_family", "level", "employer_name", "employer_type", "sector", "state", "region", "city", "employment_type", "flsa_exempt", "union", "work_mode", "travel_pct", "salary_min_usd", "salary_max_usd", "pay_basis", "hourly_bill_rate_usd", "min_years_experience", "education_required", "degree_field", "ccm_requirement", "pe_requirement", "pmp_requirement", "leed_requirement", "osha_requirement", "supervises_headcount", "reports_to", "program_value_managed_usd", "posting_date", "close_date", "open_until_filled", "residency_required", "background_check", "drug_screen", "benefits_flags", "num_applicants", "days_to_fill", "filled"]


def write(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for r in rows:
            w.writerow(["" if v is None else v for v in r])
    print(f"{os.path.basename(path):28s} {len(rows):>9,} rows")


def main():
    profiles = json.load(open(sys.argv[1], encoding="utf-8"))
    out = sys.argv[2]; os.makedirs(out, exist_ok=True)
    profiles.sort(key=lambda r: r["file"].lower())
    firms = FirmPool()
    sol, crit, prop, jobs = [], [], [], []
    si = ji = 1
    for n, doc in enumerate(profiles, 1):
        doc_id = f"DOC-{n:03d}"
        if doc.get("doc_type") in JOB_TYPES:
            ji = gen_jobs(doc, doc_id, jobs, ji)
        else:
            si = gen_solicitations(doc, doc_id, firms, sol, crit, prop, si)
    write(os.path.join(out, "solicitations.csv"), SOL_COLS, sol)
    write(os.path.join(out, "evaluation_criteria.csv"), CRIT_COLS, crit)
    write(os.path.join(out, "proposals.csv"), PROP_COLS, prop)
    write(os.path.join(out, "job_postings.csv"), JOB_COLS, jobs)


if __name__ == "__main__":
    main()
