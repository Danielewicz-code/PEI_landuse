"""Final layer: IDs, area, county, provenance -> GeoPackage for ArcGIS."""
from __future__ import annotations

import json
from pathlib import Path

from .config import log


def finalize(cfg, fields_gdf, run_dir):
    import geopandas as gpd

    gdf = fields_gdf.copy()
    target_crs = cfg["project"].get("crs")
    if target_crs and gdf.crs and gdf.crs.to_string() != target_crs:
        gdf = gdf.to_crs(target_crs)

    year = cfg["project"]["year"]
    gdf["year"] = year
    gdf["area_m2"] = gdf.geometry.area.round(1)

    counties_path = cfg["paths"].get("counties")
    if counties_path and Path(counties_path).exists():
        counties = gpd.read_file(counties_path).to_crs(gdf.crs)
        name_col = cfg["project"].get("county_field", "NAME")
        pts = gdf.copy(); pts["geometry"] = gdf.geometry.representative_point()
        j = gpd.sjoin(pts, counties[[name_col, "geometry"]], how="left", predicate="within")
        gdf["county"] = j[name_col].values

    run_id = cfg["run"]["id"]
    gdf["run_id"] = run_id
    gdf.insert(0, "field_id", [f"{year}_{i:07d}" for i in range(1, len(gdf) + 1)])

    out = Path(run_dir) / f"clear_land_{year}_{cfg['run']['name']}.gpkg"
    gdf.to_file(out, layer="clear_land", driver="GPKG")
    summary = {"n_fields": int(len(gdf)), "total_area_ha": round(float(gdf.area_m2.sum()) / 1e4, 1)}
    if "county" in gdf:
        summary["by_county"] = gdf.groupby("county").agg(n=("field_id", "size"), area_ha=("area_m2", lambda a: round(a.sum() / 1e4, 1))).to_dict("index")
    (Path(run_dir) / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(run_dir, f"Saved {out.name}: {summary}")
    return out
