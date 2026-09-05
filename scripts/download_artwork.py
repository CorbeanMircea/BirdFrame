"""
Download public-domain vintage bird illustrations from Wikimedia Commons.

Sources:
  - "Nederlandsche vogelen" by Nozeman & Sepp (1770-1829) — public domain
  - Naumann "Naturgeschichte der Vögel Mitteleuropas" (1905) — public domain

Run from the BirdFrame root:
    python scripts/download_artwork.py

Already-downloaded files are skipped. Safe to re-run after rate limiting.
"""

import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config

# ---------------------------------------------------------------------------
# Species list with confirmed Wikimedia Commons filenames.
#
# Nederlandsche vogelen filenames confirmed from:
# https://commons.wikimedia.org/wiki/Category:Nederlandsche_vogelen_van_Nozeman_en_Sepp
#
# Sylvia atricapilla uses Naumann plate (also fully public domain).
# ---------------------------------------------------------------------------

SPECIES_ARTWORK = [
    (
        "Erithacus rubecula",
        "European Robin",
        "Nederlandsche_vogelen_(KB)_-_Erithacus_rubecula_(086b).jpg",
    ),
    (
        "Turdus merula",
        "Eurasian Blackbird",
        "Nederlandsche_vogelen_(KB)_-_Turdus_merula_(016f).jpg",
    ),
    (
        "Parus major",
        "Great Tit",
        "Nederlandsche_vogelen_(KB)_-_Parus_major_(112b).jpg",
    ),
    (
        "Cyanistes caeruleus",
        "Blue Tit",
        "Nederlandsche_vogelen_(KB)_-_Cyanistes_caeruleus_(044b).jpg",
    ),
    (
        "Fringilla coelebs",
        "Common Chaffinch",
        "Nederlandsche_vogelen_(KB)_-_Fringilla_coelebs_(140b).jpg",
    ),
    (
        "Passer domesticus",
        "House Sparrow",
        "Nederlandsche_vogelen_(KB)_-_Passer_domesticus_(076d).jpg",
    ),
    (
        "Hirundo rustica",
        "Barn Swallow",
        "Nederlandsche_vogelen_(KB)_-_Hirundo_rustica_(030b).jpg",
    ),
    (
        "Columba palumbus",
        "Common Wood Pigeon",
        "Nederlandsche_vogelen_(KB)_-_Columba_palumbus_(008b).jpg",
    ),
    (
        "Garrulus glandarius",
        "Eurasian Jay",
        "Nederlandsche_vogelen_(KB)_-_Garrulus_glandarius_(03va).jpg",
    ),
    (
        # Naumann plate — also fully public domain (1905, author died 1857)
        "Sylvia atricapilla",
        "Eurasian Blackcap",
        "Nederlandsche_vogelen_(KB)_-_Sylvia_atricapilla_(424b).jpg",
    ),
]

WIKIMEDIA_FILEPATH_URL = (
    "https://commons.wikimedia.org/wiki/Special:FilePath/{filename}?width=800"
)

HEADERS = {
    "User-Agent": (
        "BirdFrame/1.0 (bird sound artwork downloader; "
        "educational/research use; public domain images only)"
    )
}


def _local_filename(scientific_name: str) -> str:
    return scientific_name.lower().replace(" ", "_") + ".jpg"


def download_one(
    scientific_name: str,
    common_name: str,
    wikimedia_filename: str,
    dest_dir: Path,
    delay: float = 3.0,
) -> bool:
    dest = dest_dir / _local_filename(scientific_name)

    if dest.exists() and dest.stat().st_size > 5000:
        print(f"  ✓ Already have: {common_name}")
        return True

    url = WIKIMEDIA_FILEPATH_URL.format(filename=wikimedia_filename)
    print(f"  ↓ {common_name} ({scientific_name})")

    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()

        if len(data) < 5000:
            print(f"    ✗ Response too small ({len(data)} bytes)")
            return False

        dest.write_bytes(data)
        print(f"    ✓ Saved {dest.name} ({len(data):,} bytes)")
        time.sleep(delay)
        return True

    except urllib.error.HTTPError as exc:
        print(f"    ✗ HTTP {exc.code}: {exc.reason}")
        if exc.code == 429:
            print(f"    Rate limited — waiting 20 seconds…")
            time.sleep(20)
        return False
    except urllib.error.URLError as exc:
        print(f"    ✗ Network error: {exc.reason}")
        return False
    except OSError as exc:
        print(f"    ✗ Write error: {exc}")
        return False


def main() -> None:
    dest_dir = config.ASSETS_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBirdFrame Artwork Downloader")
    print(f"Sources: Nederlandsche vogelen (1770-1829) + Naumann (1905)")
    print(f"License: Public domain")
    print(f"Destination: {dest_dir}")
    print(f"Species: {len(SPECIES_ARTWORK)}\n")

    succeeded = 0
    failed = []

    for sci, common, wiki_file in SPECIES_ARTWORK:
        ok = download_one(sci, common, wiki_file, dest_dir)
        if ok:
            succeeded += 1
        else:
            failed.append((sci, common, wiki_file))

    print(f"\n{'='*50}")
    print(f"Result: {succeeded}/{len(SPECIES_ARTWORK)} downloaded\n")

    if failed:
        print(f"Failed ({len(failed)}) — re-run to retry (skips existing files):")
        for sci, common, _ in failed:
            print(f"  {common} ({sci})")
    else:
        print("All artwork downloaded successfully.")

    print(f"\nVerifying via StaticArtworkProvider…")
    from backend.artwork.static_provider import StaticArtworkProvider
    provider = StaticArtworkProvider(assets_dir=dest_dir)
    count = provider.artwork_count()
    print(f"Provider sees {count} species with artwork:")
    for name in sorted(provider.supported_species):
        path = provider.get_artwork(name)
        size_kb = path.stat().st_size // 1024 if path else 0
        print(f"  {name}: {path.name if path else 'MISSING'} ({size_kb} KB)")

    print(f"""
Note on artwork coverage
------------------------
These {len(SPECIES_ARTWORK)} species are pre-sourced illustrations for common
Romanian/European birds. They are NOT a detection whitelist.

BirdNET detects 6,000+ species. For any species without artwork,
get_artwork() returns None and the collage skips it gracefully.

To add artwork for any species, place a file named:
  <scientific_name_underscored>.jpg
in: {dest_dir}
""")


if __name__ == "__main__":
    main()