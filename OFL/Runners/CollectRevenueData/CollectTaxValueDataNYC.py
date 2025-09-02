import requests
import csv
import json



def query_point(lat, lon, extra_fields=None):
    """
    Query MapPLUTO for a given lat/lon.
    Returns dict with bbl and assesstot, or indicates not assigned.
    """
    PLUTO_URL = (
        "https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/ArcGIS/rest/services/"
        "MAPPLUTO/FeatureServer/0/query"
    )
    fields = ["bbl", "assesstot"] + (extra_fields or [])
    params = {
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": ",".join(fields),
        "f": "json"
    }
    resp = requests.get(PLUTO_URL, params=params)
    resp.raise_for_status()
    res = resp.json()
    # print(f'response {res}')
    feats = res.get("features", [])
    # print(f'feats {feats}')
    # assessed_val = feats[0]["attributes"]#["AssessTot"]
    # print(f'assessed_val {assessed_val}')
    if not feats:
        return {"bbl": None, "assesstot": None, "status": "No tax value assigned"}
    attr = feats[0].get("attributes", {})
    return {
        "bbl": attr.get("BBL"),
        "assesstot": attr.get("AssessTot"),
        "status": ("Tax value assigned" if attr.get("AssessTot") is not None else "No tax value assigned")
    }

def batch_process(points, output_csv=None, output_geojson=None):
    results = []
    for lat, lon in points:
        info = query_point(lat, lon)
        info.update({"latitude": lat, "longitude": lon})
        results.append(info)

    if output_csv:
        with open(output_csv, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["latitude","longitude","bbl","assesstot","status"])
            writer.writeheader()
            writer.writerows(results)

    if output_geojson:
        geo = {"type": "FeatureCollection", "features": []}
        for rec in results:
            feat = {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [rec["longitude"], rec["latitude"]]},
                "properties": {
                    "bbl": rec["bbl"],
                    "assesstot": rec["assesstot"],
                    "status": rec["status"]
                }
            }
            geo["features"].append(feat)
        with open(output_geojson, "w") as f:
            json.dump(geo, f, indent=2)

    return results