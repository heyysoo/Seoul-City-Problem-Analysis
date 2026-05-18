"""CCTV 공백도 (cctv_void) — 인프라 공백 지수 첫 번째 변수 (가중치 45%)

처리 목적
- 행정동별 CCTV 밀도(대/km²) 산출 → Min-Max 정규화 → 1-x 변환으로 공백도 산출.
- 값이 클수록 CCTV 인프라 공백이 큼 (취약 지역).
- 인프라 공백 지수의 3개 구성 변수 중 가장 큰 가중치(45%).

학술 근거 (가중치 45%)
- Welsh-Farrington (2009) 메타분석: CCTV 16% 범죄 감소
- Piza et al. (2019) 80개 평가 연구: 한국·영국에서 가장 강한 효과
- Alexandrie (2017): 공공 거리·지하철역에서 24-28% 범죄 감소

데이터 출처 (4종)
- 서울시 CCTV 통합 정보 (17개 자치구, SHP)
- 종로구 별도 SHP (983대)
- 도봉구 별도 SHP (2,988대)
- 광진·서대문·금천: 자치구 통계 수치만 (위치정보 없음)

처리 방식 (mapping_method 3종)
- spatial_join (386동): SHP 위치정보 기반 공간 조인
- area_proportional (39동): 자치구 총계 → 면적 비례 배분 (광진 15 + 서대문 14 + 금천 10)
- gu_avg_density (1동): 수궁동(11170690), 구로구 평균 밀도 111.91대/km² 적용

INSTL_SE 필터
- 교통단속·교통정보수집 제외 (생활방범·범죄예방·공원·국가유산방범 등 안전 관련 CCTV만)
- 약 3,228건 제외 (전체 40,014건 중)

산식
- cctv_density = cctv_count / area_km2  (대/km²)
- cctv_density_norm = (density - min) / (max - min)  (Min-Max 정규화)
- cctv_void = 1 - cctv_density_norm  (방향 반전: 풍부→공백)

산출 (out_dir)
- cctv_void_dong.csv (426행, 8컬럼): ADM_CD, ADM_NM, area_km2, cctv_count,
  cctv_density, cctv_density_norm, cctv_void, mapping_method
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


# -----------------------------------------------------------------------------
# 상수
# -----------------------------------------------------------------------------

# 광진·서대문·금천 자치구 단위 CCTV 통계 (위치정보 없음 → 면적 비례 배분)
# 「서울시 자치구(목적별) CCTV 설치현황」(2025.12.31 기준) "범죄예방 및 수사" 항목
GU_TOTALS = {
    "11215": 4590,  # 광진구
    "11410": 3658,  # 서대문구
    "11545": 3672,  # 금천구
}

# 수궁동 — 단일 누락 동 (구로구 평균 밀도 적용)
SUGUNG_DONG_CODE = "11170690"
GURO_GU_CODE = "11170"

# INSTL_SE 필터에서 제외할 카테고리 (교통 목적 CCTV)
EXCLUDED_INSTL_SE = frozenset({"교통단속", "교통정보수집"})


# -----------------------------------------------------------------------------
# 입력 로딩
# -----------------------------------------------------------------------------

def load_cctv_shps(
    main_shp: Path,
    jongno_shp: Path | None,
    dobong_shp: Path | None,
    source_crs: str = "EPSG:5186",
) -> gpd.GeoDataFrame:
    """3개 CCTV SHP 통합 (17개 구 + 종로 + 도봉)."""
    frames = []

    main = gpd.read_file(main_shp)
    if main.crs is None:
        main = main.set_crs(source_crs, allow_override=True)
    frames.append(main)
    print(f"  메인 SHP: {len(main)} 행 (17개 자치구 통합)")

    if jongno_shp and jongno_shp.exists():
        jongno = gpd.read_file(jongno_shp)
        if jongno.crs is None:
            jongno = jongno.set_crs(source_crs, allow_override=True)
        jongno = jongno.to_crs(main.crs)
        frames.append(jongno)
        print(f"  종로구 SHP: {len(jongno)} 행")

    if dobong_shp and dobong_shp.exists():
        dobong = gpd.read_file(dobong_shp)
        if dobong.crs is None:
            dobong = dobong.set_crs(source_crs, allow_override=True)
        dobong = dobong.to_crs(main.crs)
        frames.append(dobong)
        print(f"  도봉구 SHP: {len(dobong)} 행")

    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=main.crs)


def load_dong_shp(
    path: Path,
    encoding: str = "cp949",
    target_crs: str = "EPSG:5179",
) -> gpd.GeoDataFrame:
    """행정동 경계 SHP (EPSG:5179)."""
    gdf = gpd.read_file(path, encoding=encoding)
    if gdf.crs is None:
        gdf = gdf.set_crs(target_crs, allow_override=True)
    else:
        gdf = gdf.to_crs(target_crs)
    gdf["ADM_CD"] = gdf["ADM_CD"].astype(str).str.zfill(8)
    return gdf


# -----------------------------------------------------------------------------
# Step 1 — CCTV → 행정동 spatial join + 필터
# -----------------------------------------------------------------------------

def spatial_join_cctv(
    cctv_gdf: gpd.GeoDataFrame,
    dong_gdf: gpd.GeoDataFrame,
    instl_se_col: str = "INSTL_SE",
) -> gpd.GeoDataFrame:
    """CCTV 점 → 행정동 spatial join + INSTL_SE 필터."""
    cctv_5179 = cctv_gdf.to_crs(dong_gdf.crs)

    joined = gpd.sjoin(
        cctv_5179,
        dong_gdf[["ADM_CD", "ADM_NM", "geometry"]],
        how="left",
        predicate="within",
    )

    # INSTL_SE 필터 (생활방범 위주, 교통 목적 제외)
    if instl_se_col in joined.columns:
        before = len(joined)
        joined = joined[~joined[instl_se_col].isin(EXCLUDED_INSTL_SE)]
        after = len(joined)
        print(f"  INSTL_SE 필터: {before:,} → {after:,} (제외 {before - after:,})")
    else:
        print(f"  '{instl_se_col}' 컬럼 없음 — INSTL_SE 필터 건너뜀")

    return joined


def aggregate_count_per_dong(joined: gpd.GeoDataFrame) -> pd.DataFrame:
    """행정동별 CCTV 개수 집계 (spatial_join 방식만)."""
    counts = (
        joined.dropna(subset=["ADM_CD"])
        .groupby(["ADM_CD", "ADM_NM"], as_index=False)
        .size()
        .rename(columns={"size": "cctv_count"})
    )
    counts["ADM_CD"] = counts["ADM_CD"].astype(str).str.zfill(8)
    counts["mapping_method"] = "spatial_join"
    return counts


# -----------------------------------------------------------------------------
# Step 2 — 광진·서대문·금천 면적 비례 배분
# -----------------------------------------------------------------------------

def append_area_proportional(
    counts: pd.DataFrame,
    dong_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """광진·서대문·금천 자치구 총계 → 면적 비례 배분.

    위치정보 없음 → 자치구 총계 보존 + 면적 비례 분배가 가장 합리적.
    0 처리는 11,920대를 모두 취약 분류하는 큰 오류 발생.
    """
    rows = []
    for gu_code, total in GU_TOTALS.items():
        gu_dongs = dong_gdf[dong_gdf["ADM_CD"].str.startswith(gu_code)].copy()
        if len(gu_dongs) == 0:
            print(f"  자치구 {gu_code} 행정동 없음 — 건너뜀")
            continue

        gu_dongs["area_km2"] = gu_dongs.geometry.area / 1e6
        total_area = gu_dongs["area_km2"].sum()

        for _, row in gu_dongs.iterrows():
            ratio = row["area_km2"] / total_area
            rows.append({
                "ADM_CD": row["ADM_CD"],
                "ADM_NM": row["ADM_NM"],
                "cctv_count": total * ratio,
                "mapping_method": "area_proportional",
            })
        print(f"  자치구 {gu_code}: {len(gu_dongs)}동에 {total:,}대 면적 비례 배분")

    if rows:
        return pd.concat([counts, pd.DataFrame(rows)], ignore_index=True)
    return counts


# -----------------------------------------------------------------------------
# Step 3 — 수궁동 특수 처리 (구로구 평균 밀도)
# -----------------------------------------------------------------------------

def append_sugung_estimate(
    counts: pd.DataFrame,
    dong_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """수궁동(11170690) — 구로구 평균 밀도 적용.

    구로구 내 단일 동만 SHP 매핑 누락 → 자치구 동질성 가정으로 평균 밀도 적용.
    """
    # 구로구 내 spatial_join으로 매핑된 동들의 평균 밀도 계산
    guro_mask = counts["ADM_CD"].str.startswith(GURO_GU_CODE) & (counts["mapping_method"] == "spatial_join")
    guro_dongs = counts[guro_mask].merge(
        dong_gdf[["ADM_CD", "geometry"]], on="ADM_CD", how="left"
    )
    if len(guro_dongs) == 0:
        print("  구로구 spatial_join 동 없음 — 수궁동 추정 건너뜀")
        return counts

    guro_gdf = gpd.GeoDataFrame(guro_dongs, geometry="geometry", crs=dong_gdf.crs)
    guro_gdf["area_km2"] = guro_gdf.geometry.area / 1e6
    avg_density = guro_gdf["cctv_count"].sum() / guro_gdf["area_km2"].sum()
    print(f"  구로구 평균 밀도: {avg_density:.2f} 대/km²")

    sugung_row = dong_gdf[dong_gdf["ADM_CD"] == SUGUNG_DONG_CODE]
    if len(sugung_row) == 0:
        print(f"  수궁동({SUGUNG_DONG_CODE}) SHP 없음 — 건너뜀")
        return counts

    sugung_area = sugung_row.geometry.area.iloc[0] / 1e6
    sugung_estimated = avg_density * sugung_area
    sugung_name = sugung_row["ADM_NM"].iloc[0]
    print(f"  수궁동 추정: {sugung_area:.3f} km² * {avg_density:.2f} = {sugung_estimated:.1f}대")

    new_row = pd.DataFrame([{
        "ADM_CD": SUGUNG_DONG_CODE,
        "ADM_NM": sugung_name,
        "cctv_count": sugung_estimated,
        "mapping_method": "gu_avg_density",
    }])
    return pd.concat([counts, new_row], ignore_index=True)


# -----------------------------------------------------------------------------
# Step 4 — 밀도 + 정규화 + 공백도
# -----------------------------------------------------------------------------

def compute_void(
    counts: pd.DataFrame,
    dong_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """행정동 면적 결합 → 밀도 → Min-Max 정규화 → 1−x 변환."""
    area_df = dong_gdf[["ADM_CD", "ADM_NM"]].copy()
    area_df["area_km2"] = dong_gdf.geometry.area / 1e6
    area_df["ADM_CD"] = area_df["ADM_CD"].astype(str).str.zfill(8)

    # 426동 기준 LEFT JOIN
    counts["ADM_CD"] = counts["ADM_CD"].astype(str).str.zfill(8)
    result = area_df.merge(
        counts[["ADM_CD", "cctv_count", "mapping_method"]],
        on="ADM_CD", how="left",
    )
    result["cctv_count"] = result["cctv_count"].fillna(0)
    result["mapping_method"] = result["mapping_method"].fillna("no_data")

    # 밀도
    result["cctv_density"] = result["cctv_count"] / result["area_km2"]

    # Min-Max 정규화
    d_min = result["cctv_density"].min()
    d_max = result["cctv_density"].max()
    result["cctv_density_norm"] = (result["cctv_density"] - d_min) / (d_max - d_min)

    # 1−x 변환 (방향: 풍부 → 공백)
    result["cctv_void"] = 1 - result["cctv_density_norm"]

    return result[[
        "ADM_CD", "ADM_NM", "area_km2", "cctv_count",
        "cctv_density", "cctv_density_norm", "cctv_void", "mapping_method",
    ]]


# -----------------------------------------------------------------------------
# 진입점
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CCTV 공백도 (cctv_void) 산출")

    parser.add_argument(
        "--cctv-shp", type=Path,
        default=Path("data/raw/cctv/SEOUL_CCTV_DATA.shp"),
        help="서울시 CCTV 통합 SHP (17개 자치구, EPSG:5186)",
    )
    parser.add_argument(
        "--jongno-shp", type=Path,
        default=Path("data/raw/cctv/jongno_cctv.shp"),
        help="종로구 별도 SHP",
    )
    parser.add_argument(
        "--dobong-shp", type=Path,
        default=Path("data/raw/cctv/dobong_cctv.shp"),
        help="도봉구 별도 SHP",
    )
    parser.add_argument("--cctv-source-crs", type=str, default="EPSG:5186")
    parser.add_argument("--instl-se-col", type=str, default="INSTL_SE",
                        help="설치 목적 컬럼 (교통단속·교통정보수집 제외 필터용)")

    parser.add_argument(
        "--dong-shp", type=Path,
        default=Path("data/raw/bnd_dong_11_2025_2Q/bnd_dong_11_2025_2Q.shp"),
        help="서울 행정동 경계 SHP",
    )
    parser.add_argument("--dong-encoding", type=str, default="cp949")
    parser.add_argument("--dong-target-crs", type=str, default="EPSG:5179")

    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("data/processed/infra_inputs"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] 행정동 SHP 로드")
    dong_gdf = load_dong_shp(args.dong_shp, encoding=args.dong_encoding,
                              target_crs=args.dong_target_crs)
    print(f"  로드: {len(dong_gdf)}동, CRS: {dong_gdf.crs}")

    print("\n[2] CCTV SHP 로드 (메인 + 종로 + 도봉)")
    cctv_gdf = load_cctv_shps(
        args.cctv_shp,
        args.jongno_shp if args.jongno_shp.exists() else None,
        args.dobong_shp if args.dobong_shp.exists() else None,
        source_crs=args.cctv_source_crs,
    )
    print(f"  통합: {len(cctv_gdf):,} 행")

    print("\n[3] Spatial join + INSTL_SE 필터")
    joined = spatial_join_cctv(cctv_gdf, dong_gdf, instl_se_col=args.instl_se_col)

    print("\n[4] 행정동별 CCTV 개수 집계 (spatial_join)")
    counts = aggregate_count_per_dong(joined)
    print(f"  매핑된 동: {len(counts)}개")

    print("\n[5] 광진·서대문·금천 면적 비례 배분")
    counts = append_area_proportional(counts, dong_gdf)

    print("\n[6] 수궁동(11170690) 구로구 평균 밀도 적용")
    counts = append_sugung_estimate(counts, dong_gdf)

    print("\n[7] 밀도 → Min-Max 정규화 → 공백도 산출")
    result = compute_void(counts, dong_gdf)

    out_path = args.out_dir / "cctv_void_dong.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_path} ({len(result)}행 * {len(result.columns)}열)")

    # 검증 출력
    print("\n분포 통계:")
    print(f"  cctv_void 평균: {result['cctv_void'].mean():.4f}")
    print(f"  cctv_void 중앙값: {result['cctv_void'].median():.4f}")
    print(f"  mapping_method 분포:")
    for method, n in result["mapping_method"].value_counts().items():
        print(f"    {method:25s}: {n}동")


if __name__ == "__main__":
    # 실행 예시:
    # import sys
    # sys.argv = [
    #     "cctv_void.py",
    #     "--cctv-shp", "data/raw/cctv/SEOUL_CCTV_DATA.shp",
    #     "--jongno-shp", "data/raw/cctv/jongno_cctv.shp",
    #     "--dobong-shp", "data/raw/cctv/dobong_cctv.shp",
    #     "--dong-shp", "data/raw/bnd_dong_11_2025_2Q/bnd_dong_11_2025_2Q.shp",
    #     "--out-dir", "data/processed/infra_inputs",
    # ]
    # main()
    pass
