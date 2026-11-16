"""Fetch each SPARCS scene's sun elevation from Landsat Collection 2 STAC metadata.

Usage: python -I fetch_sparcs_sun.py SPARCS_DIR OUT_DIR
SPARCS_DIR holds the extracted `*_data.tif` files; OUT_DIR is a new, empty directory.
For each scene ID (LC8PPPRRRYYYYDDD...), the Planetary Computer STAC API is searched
for the Landsat 8 Collection 2 Level-2 item of the same path, row and date. The raw
responses are kept, and OUT_DIR/sun.csv lists scene, subscene, datetime, sun
elevation and the STAC item ID ("missing" when no item is found).
"""
import datetime, glob, json, os, sys, urllib.request

sparcs_dir, out_dir = sys.argv[1], sys.argv[2]
names = sorted(os.path.basename(p)[:-len("_data.tif")] for p in glob.glob(f"{sparcs_dir}/*_data.tif"))
rows = []
for name in names:
    scene = name.split("_")[0]
    path, row, year, doy = scene[3:6], scene[6:9], int(scene[9:13]), int(scene[13:16])
    day = datetime.date(year, 1, 1) + datetime.timedelta(days=doy - 1)
    raw = os.path.join(out_dir, f"{scene}.json")
    if not os.path.exists(raw):
        body = {"collections": ["landsat-c2-l2"],
                "datetime": f"{day}T00:00:00Z/{day}T23:59:59Z",
                "query": {"landsat:wrs_path": {"eq": path}, "landsat:wrs_row": {"eq": row},
                          "platform": {"eq": "landsat-8"}},
                "limit": 5}
        req = urllib.request.Request("https://planetarycomputer.microsoft.com/api/stac/v1/search",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r, open(raw, "wb") as f:
            f.write(r.read())
    feats = json.load(open(raw)).get("features", [])
    if len(feats) >= 1:
        p = feats[0]["properties"]
        rows.append((name, scene, p["datetime"], f"{p['view:sun_elevation']:.6f}", feats[0]["id"], len(feats)))
    else:
        rows.append((name, scene, "", "", "missing", 0))
with open(os.path.join(out_dir, "sun.csv"), "w") as f:
    f.write("subscene,scene,datetime,sun_elevation,stac_id,n_items\n")
    for r in rows:
        f.write(",".join(map(str, r)) + "\n")
print(len(rows), "subscenes;", sum(r[4] == "missing" for r in rows), "missing;", sum(r[5] > 1 for r in rows), "with several items")
