"""Download the EDP onshore wind farm SCADA dataset (Mendeley DOI 10.17632/zjxjnjp3xs.2).

Source: Kijanowski, Barszcz, Staszewski, Dao — AGH University of Krakow.
4 turbines (T01, T06, T07, T11), 2016-2017, 10-minute SCADA + met mast + event
and failure logs. F99 (99.9% CI) filtering variant is used for the per-turbine
SCADA files rather than F95, so genuine deviation isn't stripped out by a
filtering choice made for a different research question before it reaches
our models.
"""

import pathlib
import urllib.request

RAW_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"

FILES = {
    "T01_scada.xlsx": "https://data.mendeley.com/public-files/datasets/zjxjnjp3xs/files/6c3a6348-959d-4ed6-8028-e587055441b2/file_downloaded",
    "T06_scada.xlsx": "https://data.mendeley.com/public-files/datasets/zjxjnjp3xs/files/8dfb1277-d77c-4389-9b01-f6c4b6d61ed5/file_downloaded",
    "T07_scada.xlsx": "https://data.mendeley.com/public-files/datasets/zjxjnjp3xs/files/bf728fca-cbea-4efc-8065-53c81c293f6d/file_downloaded",
    "T11_scada.xlsx": "https://data.mendeley.com/public-files/datasets/zjxjnjp3xs/files/a8be1f9d-c864-4246-85af-2515024b9b4b/file_downloaded",
    "met_mast_scada_combined.xlsx": "https://data.mendeley.com/public-files/datasets/zjxjnjp3xs/files/e9488959-c6b6-4478-81b3-a64990a57d83/file_downloaded",
    "failure_logbook.xlsx": "https://data.mendeley.com/public-files/datasets/zjxjnjp3xs/files/034dd515-8eae-44ee-8d99-537de6b9c4f4/file_downloaded",
    "event_logs_2016.xlsx": "https://data.mendeley.com/public-files/datasets/zjxjnjp3xs/files/ae510451-18dd-4eaf-9cc2-2783a2e51809/file_downloaded",
    "event_logs_2017.xlsx": "https://data.mendeley.com/public-files/datasets/zjxjnjp3xs/files/a1e69df8-4e18-4047-b2ff-006fb89cf3e9/file_downloaded",
}


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in FILES.items():
        dest = RAW_DIR / filename
        if dest.exists():
            print(f"skip (exists): {filename}")
            continue
        print(f"downloading: {filename}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, open(dest, "wb") as out_file:
            out_file.write(response.read())
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  done: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
