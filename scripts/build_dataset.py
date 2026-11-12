"""NASA OMNI2 saatlik veriyi (omni2_all_years.dat) temiz bir Parquet tablosuna çevirir.

Kullanım: python scripts/build_dataset.py
Girdi : data/raw/omni2_all_years.dat  (kaynak: https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/)
Çıktı : data/kaggle/omni2_hourly.parquet
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "omni2_all_years.dat"
OUT = ROOT / "data" / "kaggle" / "omni2_hourly.parquet"

# (sütun adı, eksik-veri dolgu değeri) — sıra omni2.text'teki 55 kelimeyle birebir aynı.
COLUMNS = [
    ("year", None), ("doy", None), ("hour", None),
    ("bartels_rotation", 9999),
    ("imf_spacecraft_id", 99), ("plasma_spacecraft_id", 99),
    ("imf_n_points", 999), ("plasma_n_points", 999),
    ("b_mag_avg", 999.9), ("b_vector_mag", 999.9),
    ("b_lat_gse", 999.9), ("b_lon_gse", 999.9),
    ("bx_gse", 999.9), ("by_gse", 999.9), ("bz_gse", 999.9),
    ("by_gsm", 999.9), ("bz_gsm", 999.9),
    ("sigma_b_mag", 999.9), ("sigma_b", 999.9),
    ("sigma_bx", 999.9), ("sigma_by", 999.9), ("sigma_bz", 999.9),
    ("proton_temp", 9999999.0), ("proton_density", 999.9),
    ("flow_speed", 9999.0), ("flow_lon", 999.9), ("flow_lat", 999.9),
    ("alpha_proton_ratio", 9.999), ("flow_pressure", 99.99),
    ("sigma_temp", 9999999.0), ("sigma_density", 999.9), ("sigma_speed", 9999.0),
    ("sigma_flow_lon", 999.9), ("sigma_flow_lat", 999.9),
    ("sigma_alpha_proton_ratio", 9.999),
    ("electric_field", 999.99), ("plasma_beta", 999.99), ("alfven_mach", 999.9),
    ("kp_x10", 99), ("sunspot_number", 999),
    ("dst", 99999), ("ae", 9999),
    ("proton_flux_1mev", 999999.99), ("proton_flux_2mev", 99999.99),
    ("proton_flux_4mev", 99999.99), ("proton_flux_10mev", 99999.99),
    ("proton_flux_30mev", 99999.99), ("proton_flux_60mev", 99999.99),
    ("proton_flux_flag", None),
    ("ap", 999), ("f107", 999.9), ("pc_n", 999.9),
    ("al", 99999), ("au", 99999),
    ("magnetosonic_mach", 99.9),
]


def main() -> None:
    names = [c for c, _ in COLUMNS]
    df = pd.read_csv(RAW, sep=r"\s+", header=None, names=names)

    for name, fill in COLUMNS:
        if fill is not None:
            df[name] = df[name].astype("float64").mask(df[name] == fill).astype("float32")

    df.insert(
        0,
        "time",
        pd.to_datetime(df["year"].astype(str) + df["doy"].astype(str).str.zfill(3), format="%Y%j")
        + pd.to_timedelta(df["hour"], unit="h"),
    )
    df[["year", "doy", "hour"]] = df[["year", "doy", "hour"]].astype("int16")
    df["proton_flux_flag"] = df["proton_flux_flag"].astype("int8")

    # Dosya, yılın sonuna kadar dolgu satırlarıyla uzuyor; Dst'nin bittiği yerden sonrasını at.
    df = df.loc[: df["dst"].last_valid_index()].reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False, compression="zstd")

    print(f"{len(df):,} satır, {df.shape[1]} sütun -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    print(f"zaman aralığı: {df['time'].min()} .. {df['time'].max()}")
    print(f"zaman adımı tek tip mi (1 saat): {(df['time'].diff().dropna() == pd.Timedelta('1h')).all()}")
    cov = df.assign(y=df["year"] // 10 * 10).groupby("y")[["dst", "bz_gsm", "flow_speed", "proton_density"]]
    print("\nOnyıllara göre doluluk oranı (%):")
    print((cov.apply(lambda g: g.notna().mean()) * 100).round(1).to_string())
    print("\nDst özeti:")
    print(df["dst"].describe().round(1).to_string())


if __name__ == "__main__":
    main()
