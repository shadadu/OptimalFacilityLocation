# parcel_sources.py
# Python 3.10+
from __future__ import annotations
import os, time, urllib.parse, requests

HEADERS = {"User-Agent": "parcel-research/1.0 (+contact: you@example.com)"}
REQ_TIMEOUT = 25

# ---------------------------
# Helpers
# ---------------------------

def _get(url: str, params: dict | None = None):
    r = requests.get(url, params=params, headers=HEADERS, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r.json()

def _arcgis_query(base: str, layer: int = 0, **params):
    """Generic ArcGIS REST /query. base = .../MapServer or .../FeatureServer root."""
    url = base.rstrip("/") + f"/{layer}/query"
    default = {
        "f": "json",
        "outFields": "*",
        "returnGeometry": "true",
    }
    default.update(params)
    return _get(url, default)

# ======================================================================
# 1) LOS ANGELES COUNTY, CA  (Assessor PAIS + Map Books via ArcGIS)
# REST folder index: https://assessor.gis.lacounty.gov/assessor/rest/services
# PAIS sales/parcel layer: https://assessor.gis.lacounty.gov/assessor/rest/services/PAIS/pais_sales_parcels/MapServer
# Map Books: https://egispais.gis.lacounty.gov/pais/rest/services/MAPPING/AssessorMapBooks_AMP/MapServer
# ======================================================================

LA_PARCELS_BASE = "https://assessor.gis.lacounty.gov/assessor/rest/services/PAIS/pais_sales_parcels/MapServer"
LA_MAPBOOKS_BASE = "https://egispais.gis.lacounty.gov/pais/rest/services/MAPPING/AssessorMapBooks_AMP/MapServer"

def la_parcels_by_intersect(x_3857: float, y_3857: float, tol_m: float = 1.0):
    """Point-in-parcel query using PAIS Sales Parcels (layer 0)."""
    geom = {"x": x_3857, "y": y_3857, "spatialReference": {"wkid": 3857}}
    return _arcgis_query(
        LA_PARCELS_BASE, 0,
        geometry=urllib.parse.quote(str(geom).replace("'", '"')),
        geometryType="esriGeometryPoint",
        spatialRel="esriSpatialRelIntersects",
        inSR=3857, outSR=3857
    )

def la_parcels_by_attribute(where: str):
    """Attribute search (e.g., AIN='1234-567-890'). Field names vary; inspect layer fields first."""
    return _arcgis_query(LA_PARCELS_BASE, 0, where=where)

def la_mapbook_by_tile(where: str = "1=1"):
    """Lookup Assessor Map Book tiles (helpful for indexing)."""
    return _arcgis_query(LA_MAPBOOKS_BASE, 0, where=where)

# ======================================================================
# 2) COOK COUNTY (CHICAGO), IL  (Assessor via Socrata Open Data)
# Assessed values API: https://dev.socrata.com/foundry/datacatalog.cookcountyil.gov/uzyt-m557
# Parcel universe: https://datacatalog.cookcountyil.gov/Property-Taxation/Assessor-Parcel-Universe/nj4t-kc8j
# ======================================================================

SOCRATA_BASE = "https://datacatalog.cookcountyil.gov/resource"

def cook_assessed_values(pin14: str, app_token: str | None = None):
    """
    Fetch assessed values for a 14-digit PIN.
    Dataset: uzyt-m557
    """
    url = f"{SOCRATA_BASE}/uzyt-m557.json"
    params = {"pin": pin14}
    headers = HEADERS.copy()
    if app_token:
        headers["X-App-Token"] = app_token
    r = requests.get(url, params=params, headers=headers, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r.json()

def cook_parcel_universe(pin14: str, app_token: str | None = None):
    """
    Core parcel metadata snapshot for a PIN.
    Dataset: nj4t-kc8j
    """
    url = f"{SOCRATA_BASE}/nj4t-kc8j.json"
    params = {"pin": pin14}
    headers = HEADERS.copy()
    if app_token:
        headers["X-App-Token"] = app_token
    r = requests.get(url, params=params, headers=headers, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r.json()

# Note: Cook County “Property Info” portal is an HTML app; rely on the open APIs above rather than scraping.

# ======================================================================
# 3) HARRIS COUNTY (HOUSTON), TX (HCAD ArcGIS)
# Parcels layer: https://www.gis.hctx.net/arcgis/rest/services/HCAD/Parcels/MapServer
# ======================================================================

HCAD_PARCELS_BASE = "https://www.gis.hctx.net/arcgis/rest/services/HCAD/Parcels/MapServer"

def hcad_by_account(account: str):
    """
    Query by account number (field names sometimes 'ACCOUNT' or 'ACCOUNT_NUM').
    Try both; fall back to contains-search.
    """
    for field in ("ACCOUNT", "ACCOUNT_NUM", "ACCOUNTNO"):
        res = _arcgis_query(HCAD_PARCELS_BASE, 0, where=f"{field}='{account}'")
        if res.get("features"):
            return res
    # fallback: LIKE match
    return _arcgis_query(HCAD_PARCELS_BASE, 0, where=f"ACCOUNT LIKE '%{account}%'")

def hcad_by_point(x_3857: float, y_3857: float):
    geom = {"x": x_3857, "y": y_3857, "spatialReference": {"wkid": 3857}}
    return _arcgis_query(
        HCAD_PARCELS_BASE, 0,
        geometry=urllib.parse.quote(str(geom).replace("'", '"')),
        geometryType="esriGeometryPoint",
        spatialRel="esriSpatialRelIntersects",
        inSR=3857, outSR=3857
    )

# ======================================================================
# 4) MARICOPA COUNTY (PHOENIX), AZ (Assessor + Treasurer via ArcGIS/URLs)
# Parcels MapServer: https://gis.mcassessor.maricopa.gov/arcgis/rest/services/Parcels/MapServer
# Parcel outline: https://gis.mcassessor.maricopa.gov/arcgis/rest/services/ParcelOutline/MapServer
# Treasurer parcel page: https://treasurer.maricopa.gov/Parcel/?Parcel={APN}
# ======================================================================

MARICOPA_PARCELS_BASE = "https://gis.mcassessor.maricopa.gov/arcgis/rest/services/Parcels/MapServer"

def maricopa_by_apn(apn: str):
    """Query by APN (field typically 'APN' in Maricopa services)."""
    return _arcgis_query(MARICOPA_PARCELS_BASE, 0, where=f"APN='{apn}'")

def maricopa_by_point(x_3857: float, y_3857: float):
    geom = {"x": x_3857, "y": y_3857, "spatialReference": {"wkid": 3857}}
    return _arcgis_query(
        MARICOPA_PARCELS_BASE, 0,
        geometry=urllib.parse.quote(str(geom).replace("'", '"')),
        geometryType="esriGeometryPoint",
        spatialRel="esriSpatialRelIntersects",
        inSR=3857, outSR=3857
    )

def maricopa_treasurer_link(apn: str) -> str:
    """Direct treasurer detail page (HTML) for taxes/bills."""
    return f"https://treasurer.maricopa.gov/Parcel/?Parcel={urllib.parse.quote(apn)}"

# ======================================================================
# 5) KING COUNTY (SEATTLE), WA (Open Data + Parcel Viewer)
# Open data API (ArcGIS Hub/FeatureService): see dataset/API page.
# Example dataset: “Parcels for King County with address...” exposed via API.
# ======================================================================

# Example FeatureService root from Open Data “API Explorer” page:
KING_CO_PARCELS_FEATURESERVICE = (
    "https://services7.arcgis.com/8pQ9YqQGIOdVZ8Ny/arcgis/rest/services/"
    "Parcels_for_King_County_with_address_with_property_information/FeatureServer"
)

def king_county_parcels_where(where: str, layer: int = 0, out_fields: str = "*", result_record_count: int = 2000, result_offset: int = 0):
    """Generic where-query against the published parcels FeatureService."""
    return _arcgis_query(
        KING_CO_PARCELS_FEATURESERVICE, layer,
        where=where,
        outFields=out_fields,
        resultRecordCount=result_record_count,
        resultOffset=result_offset
    )

def king_county_by_parcel_id(parid: str):
    """Try common parcel id fields."""
    for field in ("PIN", "PARCELID", "PARCEL_ID", "TAXPARCELNUMBER"):
        res = king_county_parcels_where(f"{field}='{parid}'")
        if res.get("features"):
            return res
    return {"features": []}

# ======================================================================
# 6) NEW YORK CITY, NY (NYC Open Data / DOF)
# Property valuation/assessment (DOF) dataset on NYC Open Data:
#   Property Valuation and Assessment Data: https://data.cityofnewyork.us/City-Government/Property-Valuation-and-Assessment-Data/yjxr-fw8i
# DOF Property Tax portal (HTML): https://a836-pts-access.nyc.gov/care/
# BBL lookup jump page: https://www.nyc.gov/assets/finance/jump/bbl_lookup.html
# ======================================================================

NYC_OD_BASE = "https://data.cityofnewyork.us/resource"

def nyc_dof_assessment_by_bbl(boro: int, block: int, lot: int, app_token: str | None = None):
    """
    Query the DOF Property Valuation & Assessment dataset (yjxr-fw8i) by BBL.
    boro: 1=Manhattan, 2=Bronx, 3=Brooklyn, 4=Queens, 5=Staten Island
    """
    url = f"{NYC_OD_BASE}/yjxr-fw8i.json"
    # Typical fields in this dataset include boro, block, lot, taxclass, bldgclass, marketvalue, assessedvalue, etc.
    params = {"boro": str(boro), "block": str(block), "lot": str(lot)}
    headers = HEADERS.copy()
    if app_token:
        headers["X-App-Token"] = app_token
    r = requests.get(url, params=params, headers=headers, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r.json()

def nyc_property_tax_portal_link(boro: int, block: int, lot: int) -> str:
    """Direct link to DOF portal search-by-BBL (user must click through)."""
    return "https://a836-pts-access.nyc.gov/care/"

# ---------------------------
# Example usage (remove or guard with __main__)
# ---------------------------
if __name__ == "__main__":
    # LA example: attribute search (you must know a valid field & value)
    print(la_parcels_by_attribute("AIN='2125-036-901'"))

    # Cook County example:
    # print(cook_assessed_values("16014300070000"))

    # HCAD example:
    # print(hcad_by_account("1234567890"))

    # Maricopa example:
    # print(maricopa_by_apn("123-45-678"))

    # King County example:
    # print(king_county_by_parcel_id("1234567890"))

    # NYC example:
    print(nyc_dof_assessment_by_bbl(2, 1, 1))
    pass
