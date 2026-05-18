"""파출소 접근 취약도 (police_void) — 인프라 공백 지수 두 번째 변수 (가중치 30%)

처리 목적
- 행정동 중심점(centroid)에서 가장 가까운 파출소·지구대까지의 직선 최단거리 산출.
- Min-Max 정규화로 [0, 1] 변환. 1-x 변환 없음 (거리 자체가 이미 취약도 방향).
- 값이 클수록 경찰 접근성 취약 (인프라 공백).

학술 근거 (가중치 30%)
- Sherman & Weisburd (1995): Minneapolis Hot Spots Patrol — place-specific micro-deterrence
- Braga et al. (2019) 78개 핫스팟 메타분석: Cohen's d = 0.110
- Turchan & Braga (2024) 32개 연구: 24% 폭력 범죄 감소, POP 35% vs 전통 16%

거리 기반 채택 이유 (시설 수 기반 X)
- 426개 행정동 중 201개(47%)가 시설을 한 개도 보유하지 않음
- 시설 수 기반이면 절반이 동일하게 "0"으로 분류 → 차별화 불가능
- 거리 기반은 시설 0개 동도 인접 시설까지의 거리로 차별화 가능

분석 대상 시설
- 파출소 144개 + 지구대 99개 = 243개 (서울 31개 경찰서 관할)

산식
- C_d = centroid(polygon_d)
- dist_m_d = min_i ||C_d - P_i||  (직선 최단거리, EPSG:5179 미터 단위)
- police_void_d = (dist_m_d - min) / (max - min)  (Min-Max, 1-x 변환 없음)

산출 (out_dir)
- police_distance_dong.csv (426행, 4컬럼): ADM_CD, ADM_NM, dist_m, police_void
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


# -----------------------------------------------------------------------------
# 입력 로딩
# -----------------------------------------------------------------------------

def load_police(
    path: Path,
    lat_col: str = "위도",
    lon_col: str = "경도",
    encoding: str = "utf-8-sig",
    target_crs: str = "EPSG:5179",
) -> gpd.GeoDataFrame:
    """파출소·지구대 위치 csv 로드 → GeoDataFrame.

    원본: WGS84 (EPSG:4326) → 타겟 CRS (EPSG:5179, 미터 단위)로 변환.
    """
    df = pd.read_csv(path, encoding=encoding)
    if lat_col not in df.columns or lon_col not in df.columns:
        raise KeyError(
            f"위도·경도 컬럼 누락: {df.columns.tolist()}. "
            f"--lat-col / --lon-col 인자로 지정 필요."
        )

    df = df.dropna(subset=[lat_col, lon_col]).copy()
    df["geometry"] = df.apply(
        lambda r: Point(r[lon_col], r[lat_col]), axis=1
    )
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    return gdf.to_crs(target_crs)


def load_dong_shp(
    path: Path,
    encoding: str = "cp949",
    target_crs: str = "EPSG:5179",
) -> gpd.GeoDataFrame:
    """행정동 경계 SHP."""
    gdf = gpd.read_file(path, encoding=encoding)
    if gdf.crs is None:
        gdf = gdf.set_crs(target_crs, allow_override=True)
    else:
        gdf = gdf.to_crs(target_crs)
    gdf["ADM_CD"] = gdf["ADM_CD"].astype(str).str.zfill(8)
    return gdf


# -----------------------------------------------------------------------------
# 핵심 계산
# -----------------------------------------------------------------------------

def compute_min_distance(
    dong_gdf: gpd.GeoDataFrame,
    police_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """행정동 centroid → 가장 가까운 시설까지의 직선 최단거리 (m).

    GeoPandas `.centroid` 사용 (면적 가중 중심).
    각 동에 대해 모든 시설과의 거리를 계산 후 최솟값.
    """
    # centroid 산출
    centroids = dong_gdf.copy()
    centroids["centroid"] = dong_gdf.geometry.centroid

    police_geoms = police_gdf.geometry.values
    print(f"  시설 수: {len(police_geoms)}개")
    print(f"  행정동 수: {len(centroids)}개")

    rows = []
    for _, row in centroids.iterrows():
        c = row["centroid"]
        min_dist = min(c.distance(p) for p in police_geoms)
        rows.append({
            "ADM_CD": row["ADM_CD"],
            "ADM_NM": row["ADM_NM"],
            "dist_m": min_dist,
        })

    return pd.DataFrame(rows)


def compute_void(dist_df: pd.DataFrame) -> pd.DataFrame:
    """Min-Max 정규화 → police_void (1−x 변환 없음)."""
    out = dist_df.copy()
    d_min = out["dist_m"].min()
    d_max = out["dist_m"].max()
    out["police_void"] = (out["dist_m"] - d_min) / (d_max - d_min)
    # 거리 자체가 이미 취약도 방향 → 1−x 변환 없음
    return out


# -----------------------------------------------------------------------------
# 진입점
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="파출소 접근 취약도 (police_void) 산출")

    parser.add_argument(
        "--police-csv", type=Path,
        default=Path("data/raw/police/seoul_police_geo.csv"),
        help="파출소·지구대 위치 csv (위도·경도 컬럼 포함)",
    )
    parser.add_argument("--lat-col", type=str, default="위도")
    parser.add_argument("--lon-col", type=str, default="경도")
    parser.add_argument("--police-encoding", type=str, default="utf-8-sig")

    parser.add_argument(
        "--dong-shp", type=Path,
        default=Path("data/raw/bnd_dong_11_2025_2Q/bnd_dong_11_2025_2Q.shp"),
    )
    parser.add_argument("--dong-encoding", type=str, default="cp949")
    parser.add_argument("--target-crs", type=str, default="EPSG:5179")

    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("data/processed/infra_inputs"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] 파출소·지구대 위치 csv 로드")
    police = load_police(
        args.police_csv,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
        encoding=args.police_encoding,
        target_crs=args.target_crs,
    )
    print(f"  로드: {len(police)} 시설, CRS: {police.crs}")

    print("\n[2] 행정동 경계 SHP 로드")
    dong_gdf = load_dong_shp(args.dong_shp, encoding=args.dong_encoding,
                              target_crs=args.target_crs)
    print(f"  로드: {len(dong_gdf)}동")

    print("\n[3] 행정동 centroid → 시설 최단거리 계산")
    dist_df = compute_min_distance(dong_gdf, police)

    print("\n[4] Min-Max 정규화 → police_void")
    result = compute_void(dist_df)

    out_path = args.out_dir / "police_distance_dong.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_path} ({len(result)}행)")

    # 검증 출력
    print("\n분포 통계:")
    print(f"  dist_m 평균: {result['dist_m'].mean():.1f}m / 중앙값: {result['dist_m'].median():.1f}m")
    print(f"  dist_m 범위: {result['dist_m'].min():.1f}m ~ {result['dist_m'].max():.1f}m")
    print(f"  police_void 평균: {result['police_void'].mean():.4f}")
    print(f"  police_void 중앙값: {result['police_void'].median():.4f}")

    # 극단 사례 (참고)
    print("\n최단 거리 5동 (파출소가 가장 가까운 동):")
    print(result.nsmallest(5, "dist_m")[["ADM_NM", "dist_m"]].to_string(index=False))
    print("\n최장 거리 5동 (파출소가 가장 먼 동):")
    print(result.nlargest(5, "dist_m")[["ADM_NM", "dist_m"]].to_string(index=False))


if __name__ == "__main__":
    # 실행 예시:
    # import sys
    # sys.argv = [
    #     "police_void.py",
    #     "--police-csv", "data/raw/police/seoul_police_geo.csv",
    #     "--dong-shp", "data/raw/bnd_dong_11_2025_2Q/bnd_dong_11_2025_2Q.shp",
    #     "--out-dir", "data/processed/infra_inputs",
    # ]
    # main()
    pass
