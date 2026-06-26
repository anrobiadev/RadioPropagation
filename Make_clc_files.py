#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_clc_tiles.py - Cut a CORINE Land Cover GeoTIFF into 1-degree binary tiles
                    compatible with the Radio Coverage Android app (ClcManager).

Output format (per tile, matches ClcManager.kt exactly):
  - Filename:  {lat}_{lon}.bin     e.g. 45_24.bin  (lat/lon = SW corner, integer)
  - Size:      1200 x 1200 = 1,440,000 bytes
  - Layout:    flat uint8 array, row 0 = NORTH edge, each row = WEST -> EAST
  - Value:     CLC code 0-44  (0 = no data / ocean / outside)

Usage:
  pip install rasterio numpy
  python make_clc_tiles.py INPUT.tif OUTPUT_DIR [--gzip] [--bbox N E S W]

  # All tiles covered by the GeoTIFF:
  python make_clc_tiles.py CLC2018_EU.tif ./clc_tiles

  # Only Romania-ish area, gzip-compressed for GitHub:
  python make_clc_tiles.py CLC2018_EU.tif ./clc_tiles --gzip --bbox 48 30 43 20

Notes:
  - CORINE rasters use a 44-class GRID_CODE (1-44). If your GeoTIFF stores the
    raw CLC_CODE (111, 112, ... 523) instead of 1-44, pass --remap-clc to convert.
  - The script reprojects/reads in EPSG:4326. If your TIFF is EPSG:3035 (ETRS89-LAEA,
    the CORINE default), install rasterio with GDAL and it will warp on the fly.
"""
import os
import sys
import gzip
import argparse
import numpy as np

try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.windows import from_bounds
except ImportError:
    sys.exit("ERROR: pip install rasterio numpy")

TILE_SIZE = 1200  # must match ClcManager.TILE_SIZE

# CORINE CLC_CODE (3-digit) -> GRID_CODE (1-44) mapping
CLC_CODE_TO_GRID = {
    111: 1, 112: 2, 121: 3, 122: 4, 123: 5, 124: 6, 131: 7, 132: 8, 133: 9,
    141: 10, 142: 11, 211: 12, 212: 13, 213: 14, 221: 15, 222: 16, 223: 17,
    231: 18, 241: 19, 242: 20, 243: 21, 244: 22, 311: 23, 312: 24, 313: 25,
    321: 26, 322: 27, 323: 28, 324: 29, 331: 30, 332: 31, 333: 32, 334: 33,
    335: 34, 411: 35, 412: 36, 421: 37, 422: 38, 423: 39, 511: 40, 512: 41,
    521: 42, 522: 43, 523: 44,
}


def build_remap_lut(remap_clc: bool) -> np.ndarray:
    """Build a 0..1023 lookup table to normalize pixel values to 0-44."""
    lut = np.zeros(1024, dtype=np.uint8)
    if remap_clc:
        for code, grid in CLC_CODE_TO_GRID.items():
            if code < 1024:
                lut[code] = grid
    else:
        # Already 1-44; pass through, clamp anything else to 0
        for v in range(1, 45):
            lut[v] = v
    return lut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_tif")
    ap.add_argument("output_dir")
    ap.add_argument("--gzip", action="store_true",
                    help="write .bin.gz instead of .bin (smaller for GitHub)")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("N", "E", "S", "W"),
                    help="limit output to this bounding box (degrees)")
    ap.add_argument("--remap-clc", action="store_true",
                    help="input stores 3-digit CLC_CODE (111..523); convert to 1-44")
    ap.add_argument("--nodata", type=int, default=0,
                    help="value to write where there is no data (default 0)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    lut = build_remap_lut(args.remap_clc)

    with rasterio.open(args.input_tif) as src:
        print(f"Input: {src.width}x{src.height}, CRS={src.crs}, dtype={src.dtypes[0]}")

        # Work in EPSG:4326. If source differs, set up a warped VRT view.
        dst_crs = "EPSG:4326"
        need_warp = (src.crs is None) or (src.crs.to_string() != dst_crs)

        if need_warp:
            print(f"Reprojecting {src.crs} -> {dst_crs} (this may take a while)...")
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)
            # Read the whole band reprojected into memory (uint8)
            dst = np.zeros((height, width), dtype=np.uint8)
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=transform, dst_crs=dst_crs,
                resampling=Resampling.nearest)
            arr = dst
            west  = transform.c
            north = transform.f
            pxw   = transform.a
            pxh   = transform.e  # negative
        else:
            arr = src.read(1)
            west  = src.transform.c
            north = src.transform.f
            pxw   = src.transform.a
            pxh   = src.transform.e

        H, W = arr.shape
        east  = west + pxw * W
        south = north + pxh * H
        print(f"Geo extent: N={north:.3f} S={south:.3f} W={west:.3f} E={east:.3f}")

        # Determine tile range
        if args.bbox:
            bN, bE, bS, bW = args.bbox
        else:
            bN, bE, bS, bW = north, east, south, west

        lat_min = int(np.floor(min(bS, bN)))
        lat_max = int(np.floor(max(bS, bN)))
        lon_min = int(np.floor(min(bW, bE)))
        lon_max = int(np.floor(max(bW, bE)))

        print(f"Tiles: lat {lat_min}..{lat_max}, lon {lon_min}..{lon_max}")

        written = 0
        for latI in range(lat_min, lat_max + 1):
            for lonI in range(lon_min, lon_max + 1):
                tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
                if args.nodata != 0:
                    tile[:] = args.nodata

                # For each tile pixel, sample the source array (nearest)
                # Tile row 0 = north edge (latI+1), row increases southward
                # Tile col 0 = west edge (lonI),   col increases eastward
                any_data = False
                for r in range(TILE_SIZE):
                    lat = (latI + 1) - (r / (TILE_SIZE - 1))
                    sy = int(round((lat - north) / pxh))
                    if sy < 0 or sy >= H:
                        continue
                    # Vectorized over columns
                    cols = np.arange(TILE_SIZE)
                    lons = lonI + cols / (TILE_SIZE - 1)
                    sx = np.round((lons - west) / pxw).astype(np.int64)
                    valid = (sx >= 0) & (sx < W)
                    if not valid.any():
                        continue
                    raw = np.zeros(TILE_SIZE, dtype=np.uint16)
                    raw[valid] = arr[sy, sx[valid]].astype(np.uint16)
                    # Normalize via LUT (clamp >1023 to 0)
                    raw[raw >= 1024] = 0
                    tile[r] = lut[raw]
                    if tile[r].any():
                        any_data = True

                if not any_data:
                    continue  # skip empty tiles (ocean / outside coverage)

                base = f"{latI}_{lonI}.bin"
                if args.gzip:
                    path = os.path.join(args.output_dir, base + ".gz")
                    with gzip.open(path, "wb") as f:
                        f.write(tile.tobytes())
                else:
                    path = os.path.join(args.output_dir, base)
                    with open(path, "wb") as f:
                        f.write(tile.tobytes())
                written += 1
                print(f"  wrote {base}  ({tile.astype(bool).sum()} non-zero px)")

        print(f"\nDone. {written} tile(s) written to {args.output_dir}")
        if args.gzip:
            print("Files are gzip-compressed (.bin.gz). The app decompresses on import.")


if __name__ == "__main__":
    main()