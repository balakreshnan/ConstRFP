"""
Procurement + supply chain synthetic data, linked to the RFP dataset.

Vendors   = the 220 firms that bid in proposals.csv
Contracts = solicitations they won (solicitations.awarded_firm)
Buyers    = the issuing agencies on those solicitations

Outputs (all in <output_dir>):
  vendors.csv                 one row per firm
  purchase_orders.csv         >= 1,000 POs per vendor
  invoices_payments.csv       1-3 invoices per PO with payment timing
  shipments_deliveries.csv    1-3 deliveries per PO (goods or service deliverables)
  vendor_scorecard.csv        per vendor per fiscal year, derived from the tables above

Usage: python generate_procurement.py <dataset_dir>
"""
import csv, math, os, random, sys, collections
from datetime import date, timedelta

SEED = 7
PO_PER_VENDOR_MIN = 1000
rng = random.Random(SEED)

CITIES = {
    "CA": ["Fresno", "Oakland", "Riverside", "San Jose", "Sacramento", "Los Angeles", "San Diego"], "VA": ["Richmond", "Norfolk", "Roanoke", "Chesapeake", "Alexandria"],
    "NY": ["Buffalo", "Rochester", "Albany", "Brooklyn", "Queens"], "TX": ["Dallas", "Houston", "Austin", "San Antonio", "El Paso"], "OH": ["Columbus", "Cleveland", "Cincinnati", "Toledo"],
    "WA": ["Seattle", "Tacoma", "Spokane", "Bellevue"], "MA": ["Boston", "Worcester", "Springfield"], "PA": ["Philadelphia", "Pittsburgh", "Harrisburg"], "MD": ["Baltimore", "Rockville", "Frederick"],
    "DC": ["Washington"], "AZ": ["Phoenix", "Tempe", "Tucson"], "CT": ["Hartford", "New Haven"], "LA": ["New Orleans", "Baton Rouge"], "NV": ["Las Vegas", "Reno"], "GA": ["Atlanta", "Savannah"],
    "CO": ["Denver", "Colorado Springs"], "FL": ["Fort Lauderdale", "Miami", "Tampa", "Orlando"], "AK": ["Anchorage", "Fairbanks"], "NM": ["Albuquerque", "Santa Fe"], "UT": ["Salt Lake City", "Ogden"],
    "OK": ["Oklahoma City", "Tulsa"], "OR": ["Portland", "Salem"], "MO": ["St. Louis", "Kansas City"], "WV": ["Charleston", "Huntington"],
}
# rough centroid lat/lon per state for distance estimates
STATE_LL = {"CA": (36.8, -119.4), "VA": (37.5, -78.9), "NY": (42.2, -74.9), "TX": (31.1, -97.6), "OH": (40.4, -82.8), "WA": (47.4, -121.5), "MA": (42.2, -71.5), "PA": (40.6, -77.2), "MD": (39.1, -76.8),
            "DC": (38.9, -77.0), "AZ": (33.7, -111.4), "CT": (41.6, -72.8), "LA": (31.2, -91.9), "NV": (38.3, -117.1), "GA": (33.0, -83.6), "CO": (39.1, -105.4), "FL": (27.8, -81.7), "AK": (61.4, -152.3),
            "NM": (34.8, -106.2), "UT": (40.2, -111.9), "OK": (35.6, -96.9), "OR": (44.6, -122.1), "MO": (38.5, -92.3), "WV": (38.5, -81.0)}

# PO catalog: (category, item, unit, unit price range, is_goods, lead time days range)
CATALOG = [
    ("Professional Services", "Construction Management Labor - Monthly Billing", "month", (15000, 400000), False, (0, 0)),
    ("Professional Services", "Construction Inspection Labor - Monthly Billing", "month", (8000, 150000), False, (0, 0)),
    ("Professional Services", "Scheduling & Cost Control Services", "month", (5000, 60000), False, (0, 0)),
    ("Professional Services", "Program Controls Support", "month", (10000, 120000), False, (0, 0)),
    ("Professional Services", "Commissioning Services", "lot", (20000, 250000), False, (0, 0)),
    ("Professional Services", "Constructability Review", "each", (8000, 60000), False, (0, 0)),
    ("Professional Services", "Claims Analysis Support", "hour", (165, 285), False, (0, 0)),
    ("Professional Services", "Staff Augmentation - Resident Engineer", "hour", (120, 210), False, (0, 0)),
    ("Professional Services", "Staff Augmentation - Inspector", "hour", (75, 135), False, (0, 0)),
    ("Testing & Inspection", "Materials Testing Services (Concrete/Soils)", "lot", (3000, 45000), False, (0, 0)),
    ("Testing & Inspection", "Special Inspection - Structural Steel", "lot", (4000, 40000), False, (0, 0)),
    ("Testing & Inspection", "Environmental Sampling", "lot", (2500, 30000), False, (0, 0)),
    ("Field Office", "Field Office Trailer Rental", "month", (900, 3500), True, (7, 21)),
    ("Field Office", "Field Office Furniture Package", "lot", (3000, 15000), True, (10, 30)),
    ("Field Office", "Temporary Utilities & Connectivity", "month", (400, 2500), False, (0, 0)),
    ("Equipment", "Survey Total Station", "each", (12000, 35000), True, (14, 60)),
    ("Equipment", "GNSS Rover Receiver", "each", (8000, 22000), True, (14, 45)),
    ("Equipment", "Drone (UAS) Survey Kit", "each", (3000, 18000), True, (7, 30)),
    ("Equipment", "Concrete Test Cylinder Molds (case)", "case", (60, 140), True, (3, 14)),
    ("Equipment", "Nuclear Density Gauge", "each", (7000, 12000), True, (21, 60)),
    ("Equipment", "Laser Scanner", "each", (30000, 80000), True, (30, 90)),
    ("Equipment", "Laptop Workstation (ruggedized)", "each", (1800, 4200), True, (5, 21)),
    ("Equipment", "Tablet for Field Inspection", "each", (600, 1500), True, (3, 14)),
    ("Equipment", "Construction Camera System", "each", (2500, 9000), True, (10, 30)),
    ("Safety & PPE", "PPE Kit (hard hat, vest, boots, glasses)", "kit", (120, 350), True, (2, 10)),
    ("Safety & PPE", "Gas Monitor (4-gas)", "each", (500, 1400), True, (5, 21)),
    ("Safety & PPE", "Fall Protection Harness Set", "each", (150, 450), True, (3, 14)),
    ("Safety & PPE", "First Aid / Trauma Kits", "each", (80, 300), True, (2, 10)),
    ("Software", "Primavera P6 License (annual)", "seat", (2200, 3200), False, (0, 0)),
    ("Software", "Procore / e-Builder Subscription", "month", (800, 6000), False, (0, 0)),
    ("Software", "Bluebeam Revu License", "seat", (300, 450), False, (0, 0)),
    ("Software", "BIM Coordination Software", "seat", (1500, 4500), False, (0, 0)),
    ("Vehicles & Fuel", "Pickup Truck Lease", "month", (650, 1200), True, (14, 45)),
    ("Vehicles & Fuel", "Fleet Fuel Card Charges", "month", (300, 4000), False, (0, 0)),
    ("Printing & Documents", "Plan Reprographics", "lot", (200, 5000), True, (1, 5)),
    ("Printing & Documents", "Document Control Scanning Services", "lot", (500, 8000), False, (0, 0)),
    ("Materials", "Signage & Traffic Control Devices", "lot", (800, 12000), True, (5, 21)),
    ("Materials", "Survey Stakes, Paint & Flagging", "lot", (150, 1500), True, (2, 7)),
    ("Materials", "Erosion Control Supplies (silt fence, wattles)", "lot", (500, 9000), True, (3, 14)),
    ("Materials", "Concrete Repair Mortar (pallet)", "pallet", (900, 2200), True, (5, 21)),
    ("Training", "OSHA 30 Training Session", "seat", (180, 400), False, (0, 0)),
    ("Training", "CCM Exam Prep Course", "seat", (900, 1800), False, (0, 0)),
    ("Travel & Per Diem", "Travel and Per Diem Reimbursement", "month", (500, 12000), False, (0, 0)),
    ("Subconsultant", "Subconsultant - Scheduling Specialist", "month", (6000, 40000), False, (0, 0)),
    ("Subconsultant", "Subconsultant - DBE Inspection Firm", "month", (5000, 60000), False, (0, 0)),
    ("Subconsultant", "Subconsultant - Community Outreach", "month", (3000, 20000), False, (0, 0)),
]
GOODS_IDX = [i for i, c in enumerate(CATALOG) if c[4]]
SERVICE_DELIVERABLES = ["Monthly Progress Report", "Updated CPM Schedule", "Cost Report", "Inspection Daily Reports (batch)", "Submittal Log", "RFI Log", "Change Order Package", "Commissioning Report", "Quality Audit Report", "Safety Audit Report", "Payment Application Review", "Constructability Report"]
CARRIERS = ["FedEx Freight", "UPS", "XPO Logistics", "Old Dominion", "Estes", "Saia", "R+L Carriers", "Vendor Fleet", "YRC Freight", "DHL Express", "USPS", "Local Courier"]
WAREHOUSES = {"West": ["Reno, NV", "Ontario, CA", "Kent, WA", "Phoenix, AZ"], "South": ["Richmond, VA", "Dallas, TX", "Atlanta, GA", "Jacksonville, FL"],
              "Northeast": ["Edison, NJ", "Allentown, PA", "Worcester, MA"], "Midwest": ["Columbus, OH", "Indianapolis, IN", "Kansas City, MO"]}
STATE_REGION = {"CA": "West", "WA": "West", "OR": "West", "NV": "West", "AZ": "West", "UT": "West", "NM": "West", "CO": "West", "AK": "West", "TX": "South", "VA": "South", "GA": "South", "FL": "South", "LA": "South",
                "MD": "South", "DC": "South", "WV": "South", "OK": "South", "NY": "Northeast", "MA": "Northeast", "PA": "Northeast", "CT": "Northeast", "OH": "Midwest", "MO": "Midwest"}
COST_CENTERS = {"City": ["Public Works-Construction", "CIP-Engineering", "Parks-Capital", "Facilities-Capital"], "County": ["Public Works-Construction", "CIP-Engineering", "Facilities-Capital"],
                "State DOT": ["District Construction", "Statewide CMI Program", "Bridge Program"], "State Agency": ["Capital Asset Management", "Facilities-Capital"], "Transit Agency": ["Transit-Capital Delivery", "Rail Programs", "Facilities-Capital"],
                "K-12 School District": ["Bond Program Office", "Facilities-Capital"], "University/College": ["Capital Projects Office", "Facilities-Capital"], "Federal": ["Capital Construction Program", "Facilities-Capital"],
                "Healthcare": ["Facilities Planning & Construction"], "Housing Authority": ["Capital Projects Division"], "Airport/Port": ["Aviation-Development", "Airfield Capital Program"],
                "Utility/Water Authority": ["Water-Capital Program", "Wastewater CIP"], "Nonprofit": ["Capital Campaign"], "Other": ["Capital Projects"], "Private Company": ["Construction Operations"]}
APPROVERS = ["J. Alvarez", "M. Chen", "R. Patel", "S. Okafor", "L. Nguyen", "D. Brooks", "K. Romero", "A. Haddad", "T. Lindqvist", "P. Mensah"]
DISPUTE_REASONS = ["Quantity mismatch", "Rate not per contract", "Missing timesheets", "Duplicate invoice", "Work not accepted", "Retention miscalculated", "Missing DBE report", "Incorrect PO reference"]
PAY_METHODS = ["ACH", "ACH", "ACH", "Check", "Virtual Card", "Wire"]


def d(s):
    return date.fromisoformat(s)


def miles(s1, s2):
    (a1, o1), (a2, o2) = STATE_LL.get(s1, (39, -98)), STATE_LL.get(s2, (39, -98))
    dx = (o2 - o1) * math.cos(math.radians((a1 + a2) / 2)) * 69.0; dy = (a2 - a1) * 69.0
    if s1 == s2: return rng.randint(20, 260)
    return max(40, int(math.hypot(dx, dy) * rng.uniform(1.05, 1.3)))


def write(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for r in rows: w.writerow(["" if v is None else v for v in r])
    print(f"{os.path.basename(path):28s} {len(rows):>9,} rows")


def main():
    ds = sys.argv[1]
    sols = list(csv.DictReader(open(os.path.join(ds, "solicitations.csv"), encoding="utf-8")))
    props = csv.DictReader(open(os.path.join(ds, "proposals.csv"), encoding="utf-8"))
    firms = {}
    for r in props:
        if r["firm_name"] not in firms:
            firms[r["firm_name"]] = {"size": r["firm_size"], "hq": r["firm_hq_state"], "dbe": r["dbe_certified"] == "True"}
    awards = collections.defaultdict(list)
    agencies = []
    for s in sols:
        agencies.append((s["issuing_org"], s["org_type"], s["state"], s["city"]))
        if s["awarded_firm"]:
            awards[s["awarded_firm"]].append(s)

    # ------------------------------------------------------------------ vendors
    vendors, vrows = {}, []
    for n, (name, f) in enumerate(sorted(firms.items()), 1):
        vid = f"VEN-{n:04d}"
        emp = {"Small": rng.randint(8, 60), "Mid": rng.randint(60, 400), "Large": rng.randint(400, 3000), "National": rng.randint(3000, 25000)}[f["size"]]
        # vendor behaviour profile drives payment & delivery realism
        prof = {"reliability": max(0.5, min(0.99, rng.gauss(0.86, 0.08))),      # on-time delivery propensity
                "invoice_quality": max(0.5, min(0.99, rng.gauss(0.9, 0.06))),   # fewer disputes
                "terms": rng.choices([30, 30, 45, 60], [0.5, 0.2, 0.2, 0.1])[0]}
        vendors[name] = {"id": vid, **f, **prof, "city": rng.choice(CITIES[f["hq"]])}
        vrows.append([vid, name, f["size"], f["hq"], vendors[name]["city"], STATE_REGION.get(f["hq"], "West"), f["dbe"], rng.random() < 0.15, rng.random() < 0.1,
                      rng.randint(1958, 2012), emp, rng.choice(["541330", "541330", "236220", "541611", "541618", "541380"]), f"Net {prof['terms']}", rng.random() < 0.85,
                      rng.random() < 0.9, rng.choices(["Low", "Medium", "High"], [0.6, 0.3, 0.1])[0], round(rng.uniform(0.6, 1.4), 2), rng.random() < 0.92,
                      rng.choice(["Platinum", "Gold", "Silver", "Bronze", "Unrated"]), len(awards[name]), rng.choice(["A", "A", "B", "C"]), f"{rng.randint(10, 99)}-{rng.randint(1000000, 9999999)}"])

    po_rows, inv_rows, ship_rows = [], [], []
    score = collections.defaultdict(lambda: collections.defaultdict(lambda: {"spend": 0.0, "pos": 0, "inv": 0, "inv_on_time": 0, "days_pay": 0, "disputed": 0, "ships": 0, "otd": 0, "if": 0, "dmg": 0, "lead": 0}))
    po_n = inv_n = sh_n = 0

    for name, v in vendors.items():
        vid = v["id"]; won = awards[name]
        n_po = PO_PER_VENDOR_MIN + rng.randint(0, 250)
        for _ in range(n_po):
            on_contract = rng.random() < 0.72
            if on_contract:
                s = rng.choice(won)
                award_d = d(s["anticipated_award_date"]); months = int(s["base_term_months"]) + 12 * int(s["option_years"])
                po_date = award_d + timedelta(days=rng.randint(0, max(30, months * 30)))
                buyer, org_type, st, city = s["issuing_org"], s["org_type"], s["state"], s["city"]
                sol_id, contract_id, sector, fund = s["solicitation_id"], f"CTR-{s['solicitation_id'][4:]}", s["sector"], s["funding_source"]
                fee = float(s["est_fee_usd"])
                # service-heavy mix on contract
                ci = rng.choices(range(len(CATALOG)), [6 if c[0] in ("Professional Services", "Subconsultant") else 1 for c in CATALOG])[0]
            else:
                buyer, org_type, st, city = rng.choice(agencies)
                po_date = date(2009, 1, 1) + timedelta(days=rng.randint(0, 365 * 16))
                sol_id = contract_id = None; sector = rng.choice(["Public Buildings/Facilities", "Transportation/Highway", "Water/Wastewater", "K-12 Education", "Mixed/Multiple"]); fund = "Local"
                fee = 500000.0
                ci = rng.choices(range(len(CATALOG)), [1 if c[0] in ("Professional Services", "Subconsultant") else 3 for c in CATALOG])[0]
            po_date = min(po_date, date(2025, 9, 30))
            cat, item, unit, (plo, phi), goods, (llo, lhi) = CATALOG[ci]
            if unit == "month" and cat == "Professional Services":
                unit_price = round(min(phi, max(plo, fee / max(6, rng.randint(12, 48)) * rng.uniform(0.6, 1.4))), 2); qty = rng.choice([1, 1, 1, 2, 3])
            elif unit == "hour":
                unit_price = round(rng.uniform(plo, phi), 2); qty = rng.choice([80, 160, 320, 480, 640])
            else:
                unit_price = round(rng.uniform(plo, phi), 2); qty = rng.choice([1, 1, 1, 2, 2, 3, 4, 5, 6, 10, 12, 20]) if goods else rng.choice([1, 1, 1, 2, 3])
            amount = round(unit_price * qty, 2)
            po_type = "Change Order" if rng.random() < 0.06 else ("Task Order Release" if on_contract else ("Blanket PO Release" if rng.random() < 0.3 else "Standard PO"))
            if po_type == "Change Order": amount = round(amount * rng.uniform(0.05, 0.4), 2)
            fy = po_date.year + (1 if po_date.month >= 7 else 0)
            lead = rng.randint(llo, lhi) if goods else rng.randint(14, 45)
            req_date = po_date + timedelta(days=lead + rng.randint(0, 10))
            status = rng.choices(["Closed", "Fulfilled", "Partially Received", "Open", "Cancelled"], [0.62, 0.2, 0.06, 0.08, 0.04])[0]
            if po_date > date(2025, 3, 1) and status in ("Closed", "Fulfilled"): status = rng.choice(["Open", "Fulfilled", "Partially Received"])
            po_n += 1; po_id = f"PO-{po_n:07d}"
            terms = v["terms"] if rng.random() < 0.8 else rng.choice([30, 45, 60])
            po_rows.append([po_id, vid, name, f"REQ-{rng.randint(100000, 999999)}", sol_id, contract_id, buyer, org_type, st, city, sector, fund, po_date, fy, po_type, cat, item, qty, unit, unit_price, amount, "USD",
                            f"Net {terms}", rng.random() < 0.25 and terms == 30, req_date, status, rng.random() < 0.35, round(amount * 0.05, 2) if (cat in ("Professional Services", "Subconsultant") and rng.random() < 0.4) else 0.0,
                            rng.choice(COST_CENTERS.get(org_type, COST_CENTERS["Other"])), rng.choice(APPROVERS), rng.choice(["Competitive Award", "Task Order", "Sole Source", "Cooperative Contract", "Small Purchase"]) if not on_contract else "Task Order", rng.random() < 0.08])
            sc = score[vid][fy]; sc["spend"] += amount if status != "Cancelled" else 0; sc["pos"] += 1
            if status == "Cancelled":
                continue
            # ----------------------------------------------------------- invoices
            n_inv = 1 if (goods or qty == 1 or rng.random() < 0.5) else min(3, qty)
            remaining = amount
            for k in range(n_inv):
                inv_amt = round(remaining if k == n_inv - 1 else amount / n_inv * rng.uniform(0.8, 1.2), 2); remaining = round(remaining - inv_amt, 2)
                inv_date = po_date + timedelta(days=rng.randint(5, 40) + 30 * k)
                due = inv_date + timedelta(days=terms)
                disputed = rng.random() > v["invoice_quality"]
                mean = {"Federal": -11, "Private Company": -9, "State DOT": -7, "State Agency": -7, "Transit Agency": -6, "University/College": -6, "Utility/Water Authority": -6, "Airport/Port": -6}.get(org_type, -3)
                base_delay = rng.gauss(mean, 9) + (rng.randint(15, 60) if disputed else 0) + (rng.randint(10, 40) if rng.random() < 0.05 else 0)
                pay_date = due + timedelta(days=int(base_delay))
                if pay_date < inv_date + timedelta(days=5): pay_date = inv_date + timedelta(days=5)
                paid = pay_date <= date(2025, 10, 1) and status != "Open"
                days_late = (pay_date - due).days if paid else None
                on_time = (days_late <= 0) if paid else None
                disc = rng.random() < 0.3 and paid and (pay_date - inv_date).days <= 10 and terms == 30
                short = round(inv_amt * rng.uniform(0.02, 0.15), 2) if disputed and rng.random() < 0.5 else 0.0
                ret = round(inv_amt * 0.05, 2) if cat in ("Professional Services", "Subconsultant") and rng.random() < 0.4 else 0.0
                inv_n += 1
                inv_rows.append([f"INV-{inv_n:07d}", po_id, vid, name, buyer, inv_date, f"{name[:3].upper()}-{inv_date.year}-{rng.randint(1000, 99999)}", inv_amt, ret, round(inv_amt - ret - short, 2), f"Net {terms}", due,
                                 pay_date if paid else None, (pay_date - inv_date).days if paid else None, days_late, on_time, max(0, days_late) if paid else None, "Paid" if paid else ("Disputed" if disputed else "Pending"),
                                 rng.choice(PAY_METHODS) if paid else None, disc, round(inv_amt * 0.02, 2) if disc else 0.0, disputed, rng.choice(DISPUTE_REASONS) if disputed else None, short,
                                 round(inv_amt * 0.015 * max(0, (days_late or 0)) / 30, 2) if paid and (days_late or 0) > 30 and rng.random() < 0.3 else 0.0, rng.randint(1, 4) if paid else None])
                if paid:
                    sc["inv"] += 1; sc["inv_on_time"] += int(on_time); sc["days_pay"] += (pay_date - inv_date).days; sc["disputed"] += int(disputed)
            # ----------------------------------------------------------- shipments / deliveries
            n_ship = 1 if not goods or qty <= 2 else rng.choice([1, 1, 2, 3])
            qty_left = qty
            for k in range(n_ship):
                q = qty_left if k == n_ship - 1 else max(1, int(qty_left * rng.uniform(0.3, 0.7))); qty_left -= q
                avail = rng.choices(["In Stock", "Low Stock", "Backorder", "Made to Order"], [0.6, 0.15, 0.15, 0.1])[0] if goods else "N/A (service)"
                extra = {"In Stock": 0, "Low Stock": rng.randint(0, 5), "Backorder": rng.randint(7, 45), "Made to Order": rng.randint(10, 60)}.get(avail, 0)
                ship_date = po_date + timedelta(days=(rng.randint(1, max(2, lead // 2)) + extra) if goods else rng.randint(10, 40) + 30 * k)
                promised = req_date + timedelta(days=(30 * k) if not goods else int(extra * 0.6))
                origin = rng.choice(WAREHOUSES[STATE_REGION.get(v["hq"], "West")]) if goods else f"{v['city']}, {v['hq']}"
                o_state = origin.split(", ")[-1]
                dist = miles(o_state, st) if goods else 0
                mode = ("Courier" if dist < 60 else rng.choices(["Parcel", "LTL Truck", "FTL Truck", "Air Freight", "Rail Intermodal"], [0.35, 0.35, 0.1, 0.12, 0.08])[0]) if goods else "Electronic Submission"
                transit = 0 if not goods else {"Courier": 1, "Parcel": rng.randint(1, 5), "LTL Truck": rng.randint(2, 8), "FTL Truck": rng.randint(1, 6), "Air Freight": rng.randint(1, 3), "Rail Intermodal": rng.randint(5, 14)}[mode] + dist // 900
                late = (rng.random() > v["reliability"])
                actual = ship_date + timedelta(days=transit + (rng.randint(1, 20) if late else 0))
                delivered = actual <= date(2025, 10, 1) and status not in ("Open",)
                delay = (actual - promised).days if delivered else None
                dmg = rng.randint(1, max(1, q // 4)) if goods and delivered and rng.random() < 0.04 else 0
                in_full = delivered and (q - dmg) >= q and status != "Partially Received"
                sh_n += 1
                ship_rows.append([f"SHP-{sh_n:07d}", po_id, vid, name, sol_id, "Physical Goods" if goods else "Service Deliverable", cat, item if goods else rng.choice(SERVICE_DELIVERABLES), q, (q - dmg) if delivered else 0, unit,
                                  origin, f"{city}, {st}", st, STATE_REGION.get(st, "West"), dist, mode, rng.choice(CARRIERS) if goods else "N/A", f"1Z{rng.randint(10**9, 10**10-1)}" if goods and mode == "Parcel" else (f"PRO{rng.randint(10**7, 10**8-1)}" if goods else None),
                                  avail, rng.randint(0, 100) if goods else None, ship_date, promised, actual if delivered else None, (actual - po_date).days if delivered else None, transit if goods else None, delay,
                                  (delay <= 0) if delivered else None, in_full if delivered else None, dmg, avail == "Backorder", round(dist * rng.uniform(0.9, 2.4) + (25 if mode == "Parcel" else 120), 2) if goods else 0.0,
                                  rng.choice(["FOB Destination", "FOB Origin", "DDP"]) if goods else "N/A", rng.choices(["Accepted", "Accepted", "Accepted", "Accepted w/ Exceptions", "Rejected"], [0.7, 0.15, 0.08, 0.05, 0.02])[0] if delivered else "Pending",
                                  dmg > 0 and rng.random() < 0.7, "Delivered" if delivered else ("In Transit" if ship_date <= date(2025, 10, 1) else "Scheduled"), rng.choice(["Weather", "Carrier capacity", "Customs/Documentation", "Stockout", "Address issue", "Site access"]) if (delivered and delay and delay > 0) else None])
                if delivered:
                    sc["ships"] += 1; sc["otd"] += int(delay <= 0); sc["if"] += int(bool(in_full)); sc["dmg"] += dmg; sc["lead"] += (actual - po_date).days

    # ------------------------------------------------------------------ scorecard
    sc_rows = []
    for name, v in vendors.items():
        for fy, s in sorted(score[v["id"]].items()):
            sc_rows.append([v["id"], name, fy, s["pos"], round(s["spend"], 2), s["inv"], round(100 * s["inv_on_time"] / s["inv"], 1) if s["inv"] else None, round(s["days_pay"] / s["inv"], 1) if s["inv"] else None,
                            round(100 * s["disputed"] / s["inv"], 1) if s["inv"] else None, s["ships"], round(100 * s["otd"] / s["ships"], 1) if s["ships"] else None, round(100 * s["if"] / s["ships"], 1) if s["ships"] else None,
                            s["dmg"], round(s["lead"] / s["ships"], 1) if s["ships"] else None,
                            round(min(100, 0.4 * (100 * s["otd"] / s["ships"] if s["ships"] else 80) + 0.3 * (100 * s["if"] / s["ships"] if s["ships"] else 80) + 0.3 * (100 - (100 * s["disputed"] / s["inv"] if s["inv"] else 10))), 1)])

    write(os.path.join(ds, "vendors.csv"), ["vendor_id", "firm_name", "firm_size", "hq_state", "hq_city", "region", "dbe_certified", "mbe_certified", "wbe_certified", "year_founded", "employees", "naics_code", "default_payment_terms", "ach_enrolled", "w9_on_file", "risk_rating", "emr_safety_rating", "insurance_certificate_current", "prequalification_tier", "contracts_won", "credit_rating", "tax_id_masked"], vrows)
    write(os.path.join(ds, "purchase_orders.csv"), ["po_id", "vendor_id", "firm_name", "requisition_id", "solicitation_id", "contract_id", "buyer_agency", "buyer_org_type", "buyer_state", "buyer_city", "sector", "funding_source", "po_date", "fiscal_year", "po_type", "category", "item_description", "quantity", "unit", "unit_price_usd", "po_amount_usd", "currency", "payment_terms", "early_pay_discount_offered", "required_delivery_date", "po_status", "three_way_match_required", "retention_pct_amount_usd", "cost_center", "approver", "procurement_method", "expedited"], po_rows)
    write(os.path.join(ds, "invoices_payments.csv"), ["invoice_id", "po_id", "vendor_id", "firm_name", "buyer_agency", "invoice_date", "vendor_invoice_number", "invoice_amount_usd", "retention_withheld_usd", "net_paid_usd", "payment_terms", "due_date", "payment_date", "days_to_pay", "days_late", "paid_on_time", "days_past_due", "payment_status", "payment_method", "early_pay_discount_taken", "discount_amount_usd", "disputed", "dispute_reason", "short_paid_usd", "late_fee_usd", "approval_cycles"], inv_rows)
    write(os.path.join(ds, "shipments_deliveries.csv"), ["shipment_id", "po_id", "vendor_id", "firm_name", "solicitation_id", "delivery_type", "category", "item_or_deliverable", "qty_shipped", "qty_accepted", "unit", "origin_location", "destination_location", "destination_state", "destination_region", "distance_miles", "transport_mode", "carrier", "tracking_number", "availability_at_order", "stock_level_pct_at_order", "ship_date", "promised_delivery_date", "actual_delivery_date", "lead_time_days", "transit_days", "delivery_delay_days", "on_time_delivery", "delivered_in_full", "damaged_qty", "backordered", "freight_cost_usd", "incoterms", "inspection_result", "return_initiated", "shipment_status", "delay_reason"], ship_rows)
    write(os.path.join(ds, "vendor_scorecard.csv"), ["vendor_id", "firm_name", "fiscal_year", "po_count", "total_spend_usd", "invoices_paid", "pct_invoices_paid_on_time", "avg_days_to_pay", "pct_invoices_disputed", "deliveries", "pct_on_time_delivery", "pct_in_full", "damaged_units", "avg_lead_time_days", "vendor_performance_score"], sc_rows)


if __name__ == "__main__":
    main()
