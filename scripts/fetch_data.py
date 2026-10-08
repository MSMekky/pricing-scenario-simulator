"""Download the UCI Online Retail dataset (CC BY 4.0) into data/raw/."""
from pathlib import Path
import io
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "Online Retail.xlsx"
    if target.exists():
        print(f"already present: {target}")
        return
    print(f"downloading {URL}")
    with urllib.request.urlopen(URL, timeout=120) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(OUT)
    print(f"saved to {target}")


if __name__ == "__main__":
    main()
