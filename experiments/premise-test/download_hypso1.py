"""Download the labelled HYPSO-1 cubes and label files from DataverseNO.

Usage: python -I download_hypso1.py FILES_JSON OUT_DIR
FILES_JSON is the dataset's API file listing; OUT_DIR must be a new, empty directory
(or one this script filled earlier). Only `*-l1a.nc` and `*-slc-labels.bin` files of
the folder "1-Data with Labels" are fetched; each is checked against its MD5.
"""
import hashlib, json, os, sys, urllib.request

files_json, out_dir = sys.argv[1], sys.argv[2]
entries = json.load(open(files_json))["data"]
want = [e for e in entries
        if e.get("directoryLabel", "").startswith("1-Data with Labels/")
        and e["dataFile"]["filename"].endswith(("-l1a.nc", "-slc-labels.bin"))]
assert len(want) == 76, len(want)

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

for i, e in enumerate(want, 1):
    df = e["dataFile"]
    folder = e["directoryLabel"].split("/", 1)[1]
    dest_dir = os.path.join(out_dir, folder)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, df["filename"])
    if os.path.exists(dest) and md5(dest) == df["md5"]:
        continue
    url = f"https://dataverse.no/api/access/datafile/{df['id']}"
    tmp = dest + ".part"
    if os.path.exists(tmp):
        os.remove(tmp)
    for attempt in range(1, 21):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                mode = "ab" if have and r.status == 206 else "wb"
                with open(tmp, mode) as f:
                    while block := r.read(1 << 20):
                        f.write(block)
        except OSError as err:
            print(f"retry {attempt} {folder}/{df['filename']}: {err}", flush=True)
            continue
        size = os.path.getsize(tmp)
        if size == df["filesize"]:
            if md5(tmp) == df["md5"]:
                break
            os.remove(tmp)
            print(f"retry {attempt} {folder}/{df['filename']}: MD5 mismatch, restarting", flush=True)
        else:
            print(f"resume {attempt} {folder}/{df['filename']}: {size} of {df['filesize']} bytes", flush=True)
    else:
        sys.exit(f"failed after 20 attempts: {folder}/{df['filename']}")
    os.replace(tmp, dest)
    print(f"{i}/{len(want)} {folder}/{df['filename']} {df['filesize']}", flush=True)
print("done", len(want))
