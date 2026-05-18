"""1차 필터링: 거주지 필터 (Anderson 1976 67% + 자연영역 분모 보정)

처리 목적
- 서울 426개 행정동에서 거주지가 거의 확정된 동을 외부 토지이용 SHP 기반으로 사전 제외한다.
- 매출·인구 같은 결과 변수를 필터로 쓰지 않아 동어반복(circular reasoning)을 회피한다.
- 학술 표준 기반 컷오프: Anderson 1976 USGS Professional Paper 964의
  "predominant land use 2/3 = 67%" 기준 채택.

핵심 산식
- 자연영역 = 보전산지 U 개발제한구역 U 도시자연공원구역  (unary_union)
- 도시활용 면적 = 행정동 면적 - (행정동 ∩ 자연영역)
- 거주 비율_NEW = (행정동 ∩ 주거지역) / 도시활용 면적
- 통과 = (거주 비율_NEW < 0.67) AND (ADM_CD ≠ 공항동 11160690)

분모 보정 사유
- 자연영역(산지·공원·개발제한)이 큰 동에서 단순 행정동 면적 분모를 쓰면
  거주 비율이 과소평가됨. 예: 도봉1동 단순 8.7% → 보정 후 93.5% (산지 위주 거주지)
- 자연영역은 「국토의 계획 및 이용에 관한 법률」, 「산지관리법」 등 공식 분류 기반

공항동 수동 제외
- 11160690(공항동): 동 면적의 78.5%가 김포공항 부지. 토지이용분류상 주거도 자연도 아님.
- 단일 사례라 수동 제외로 처리.

폐기된 시도
- v6 (한강·하천 추가 보정): OpenStreetMap river polygon overlay 시도 후 폐기.
  변경된 109→90동 결과 대신 이전 방법 유지.

산출물
- dong_residential_filter_v4.csv (426행): 동별 거주비율·통과여부 마스터
- residential_filter_pass_v4.csv (108행): 통과 동 명단

학술 근거
- Anderson, J. R. et al. (1976), USGS Professional Paper 964,
  A Land Use and Land Cover Classification System for Use with Remote Sensor Data.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from shapely.validation import make_valid


# -----------------------------------------------------------------------------
# 상수
# -----------------------------------------------------------------------------

# Anderson 1976 67% (= 2/3) 컷오프
RESIDENTIAL_THRESHOLD = 0.67

# 수동 제외 동 (단일 케이스: 김포공항)
MANUAL_EXCLUDE_CODES = ("11160690",)

# UPIS_C_UQ111 (용도지역) 내 주거지역 SCLAS_CL 코드 9종
# (전용주거 1·2종, 미분류, 일반주거 1·2·3종, 준주거, 미분류, 그 외 주거)
RESIDENTIAL_SCLAS_CODES = (
    "UQA111", "UQA112", "UQA119",
    "UQA121", "UQA122", "UQA123", "UQA124", "UQA129",
    "UQA130",
)

# 좌표계 정의
DONG_CRS = 5179      # 행정동 SHP 기본 (UTM-K, GRS80)
ZONING_CRS = 5174    # UPIS SHP 기본
WORK_CRS = 5179      # overlay 작업 통일 좌표계


# -----------------------------------------------------------------------------
# SHP 로딩 유틸
# -----------------------------------------------------------------------------

def load_shp_with_crs(path: Path, fallback_crs: int, encoding: str = "cp949") -> gpd.GeoDataFrame:
    """SHP 파일을 로드한다. .prj 가 없는 경우 fallback_crs 를 적용한다."""
    gdf = gpd.read_file(path, encoding=encoding)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=fallback_crs)
    return gdf.to_crs(epsg=WORK_CRS)


def ensure_valid(geoseries: gpd.GeoSeries) -> list:
    """무효 geometry를 make_valid로 보정한 리스트 반환."""
    return [make_valid(g) if not g.is_valid else g for g in geoseries]


# -----------------------------------------------------------------------------
# 핵심 처리
# -----------------------------------------------------------------------------

def build_natural_union(
    forest_path: Path,
    greenbelt_path: Path,
    park_path: Path,
) -> object:
    """보전산지 + 개발제한구역 + 도시자연공원구역의 unary_union 결과를 반환.

    세 영역은 일부 중복되므로 unary_union으로 중복 제거된 합집합을 생성한다.
    """
    forest = load_shp_with_crs(forest_path, fallback_crs=ZONING_CRS)
    greenbelt = load_shp_with_crs(greenbelt_path, fallback_crs=ZONING_CRS)
    park = load_shp_with_crs(park_path, fallback_crs=ZONING_CRS)

    raw_sum_km2 = (
        forest.geometry.area.sum()
        + greenbelt.geometry.area.sum()
        + park.geometry.area.sum()
    ) / 1_000_000
    print(f"  보전산지: {len(forest)} polygon")
    print(f"  개발제한구역: {len(greenbelt)} polygon")
    print(f"  도시자연공원구역: {len(park)} polygon")
    print(f"  단순 합산 (중복 포함): {raw_sum_km2:.1f} km²")

    all_geoms = (
        ensure_valid(forest.geometry)
        + ensure_valid(greenbelt.geometry)
        + ensure_valid(park.geometry)
    )
    natural_union = unary_union(all_geoms)
    print(f"  unary_union 후 자연영역: {natural_union.area / 1_000_000:.1f} km²")
    return natural_union


def build_residential_union(zoning_path: Path) -> object:
    """UPIS_C_UQ111의 주거지역 SCLAS_CL 코드를 필터링해 unary_union."""
    zoning = load_shp_with_crs(zoning_path, fallback_crs=ZONING_CRS)

    if "SCLAS_CL" not in zoning.columns:
        raise KeyError(
            "UPIS_C_UQ111 SHP에 SCLAS_CL 컬럼이 없습니다. "
            f"현재 컬럼: {zoning.columns.tolist()}"
        )

    res = zoning[zoning["SCLAS_CL"].isin(RESIDENTIAL_SCLAS_CODES)].copy()
    print(f"  주거지역 폴리곤: {len(res)} (필터 후)")

    res_geoms = ensure_valid(res.geometry)
    res_union = unary_union(res_geoms)
    print(f"  주거영역 unary_union 면적: {res_union.area / 1_000_000:.1f} km²")
    return res_union


def compute_dong_metrics(
    dong: gpd.GeoDataFrame,
    natural_union: object,
    residential_union: object,
) -> pd.DataFrame:
    """행정동별 면적·자연영역·주거영역 intersection 면적을 산출한다."""
    dong = dong.copy()
    dong["dong_area_m2"] = dong.geometry.area

    natural_gdf = gpd.GeoDataFrame(geometry=[natural_union], crs=f"EPSG:{WORK_CRS}")
    res_gdf = gpd.GeoDataFrame(geometry=[residential_union], crs=f"EPSG:{WORK_CRS}")

    # 동 × 자연영역
    nat_inter = gpd.overlay(
        natural_gdf,
        dong[["ADM_CD", "ADM_NM", "dong_area_m2", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    nat_inter["natural_area_m2"] = nat_inter.geometry.area
    nat_agg = nat_inter.groupby("ADM_CD", as_index=False)["natural_area_m2"].sum()

    # 동 × 주거영역
    res_inter = gpd.overlay(
        res_gdf,
        dong[["ADM_CD", "ADM_NM", "dong_area_m2", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    res_inter["res_area_m2"] = res_inter.geometry.area
    res_agg = res_inter.groupby("ADM_CD", as_index=False)["res_area_m2"].sum()

    # 동 마스터에 병합 (없는 동은 0)
    result = dong[["ADM_CD", "ADM_NM", "dong_area_m2"]].copy()
    result = result.merge(nat_agg, on="ADM_CD", how="left")
    result = result.merge(res_agg, on="ADM_CD", how="left")
    result["natural_area_m2"] = result["natural_area_m2"].fillna(0)
    result["res_area_m2"] = result["res_area_m2"].fillna(0)

    # 도시활용 면적 = 동 면적 - 자연영역
    result["urban_area_m2"] = result["dong_area_m2"] - result["natural_area_m2"]
    result["urban_area_m2"] = result["urban_area_m2"].clip(lower=0)

    # 거주 비율 산식 (OLD: 단순, NEW: 자연영역 분모 보정)
    result["res_ratio_OLD"] = result["res_area_m2"] / result["dong_area_m2"]
    result["res_ratio_NEW"] = result["res_area_m2"] / result["urban_area_m2"].replace(0, pd.NA)
    result["res_ratio_NEW"] = result["res_ratio_NEW"].fillna(0)

    return result


def apply_filter(metrics: pd.DataFrame) -> pd.DataFrame:
    """Anderson 67% 컷오프 + 공항동 수동 제외 적용."""
    out = metrics.copy()
    out["ADM_CD"] = out["ADM_CD"].astype(str)
    out["manual_exclude"] = out["ADM_CD"].isin(MANUAL_EXCLUDE_CODES)
    out["pass_threshold"] = out["res_ratio_NEW"] < RESIDENTIAL_THRESHOLD
    out["pass_v4"] = out["pass_threshold"] & ~out["manual_exclude"]

    out["exclude_reason"] = "PASS"
    out.loc[~out["pass_threshold"], "exclude_reason"] = (
        f"residential_ratio>={RESIDENTIAL_THRESHOLD:.2f}"
    )
    out.loc[out["manual_exclude"], "exclude_reason"] = "manual_exclude_airport"

    return out.sort_values(["pass_v4", "res_ratio_NEW", "ADM_CD"], ascending=[False, True, True])


def write_summary(decision: pd.DataFrame, out_dir: Path) -> None:
    """필터 결과 요약 보고."""
    passed = decision[decision["pass_v4"]]
    excluded = decision[~decision["pass_v4"]]
    excluded_by_threshold = excluded[~excluded["manual_exclude"]]
    excluded_by_manual = excluded[excluded["manual_exclude"]]

    report = f"""# 1차 거주지 필터 산출 보고 (v4)

## 필터 결과

| 항목 | 결과 |
| --- | ---: |
| 시작 풀 | {len(decision)} |
| Anderson 67% 컷오프 | {RESIDENTIAL_THRESHOLD:.2f} |
| 자연영역 분모 보정 적용 | 보전산지 + 개발제한구역 + 도시자연공원구역 |
| 컷오프 기준 제외 | {len(excluded_by_threshold)} |
| 수동 제외 (김포공항) | {len(excluded_by_manual)} |
| 최종 통과 | {len(passed)} |

## 알려진 야간상권 동 검증 (모두 통과 기대)

명동, 을지로동, 문래동, 여의동, 종로1·2·3·4가동, 광희동, 상암동, 소공동, 회현동

→ 자세한 검증 결과는 노션 페이지 §정합성 검증 참조.
"""
    (out_dir / "residential_filter_v4_report.md").write_text(report, encoding="utf-8")


# -----------------------------------------------------------------------------
# 진입점
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="1차 거주지 필터 (Anderson 1976 67% + 자연영역 보정)")
    parser.add_argument(
        "--dong-shp",
        type=Path,
        default=Path("data/raw/boundary/bnd_dong_11_2025_2Q.shp"),
        help="서울 행정동 경계 SHP (EPSG:5179)",
    )
    parser.add_argument(
        "--zoning-shp",
        type=Path,
        default=Path("data/raw/UPIS/UPIS_C_UQ111.shp"),
        help="용도지역 SHP (주거지역 추출용)",
    )
    parser.add_argument(
        "--forest-shp",
        type=Path,
        default=Path("data/raw/UPIS/UPIS_SHP_CUL220.shp"),
        help="보전산지 SHP",
    )
    parser.add_argument(
        "--greenbelt-shp",
        type=Path,
        default=Path("data/raw/UPIS/UPIS_C_UQ141.shp"),
        help="개발제한구역 SHP",
    )
    parser.add_argument(
        "--park-shp",
        type=Path,
        default=Path("data/raw/UPIS/UPIS_C_UQ142.shp"),
        help="도시자연공원구역 SHP",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/residential_filter"),
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] SHP 로딩")
    dong = gpd.read_file(args.dong_shp)
    if dong.crs is None:
        dong = dong.set_crs(epsg=DONG_CRS)
    dong = dong.to_crs(epsg=WORK_CRS)
    print(f"  행정동 폴리곤: {len(dong)}")

    print("\n[2] 자연영역 unary_union")
    natural_union = build_natural_union(args.forest_shp, args.greenbelt_shp, args.park_shp)

    print("\n[3] 주거영역 unary_union")
    residential_union = build_residential_union(args.zoning_shp)

    print("\n[4] 동별 면적 산출 (overlay)")
    metrics = compute_dong_metrics(dong, natural_union, residential_union)
    print(f"  산출 완료: {len(metrics)} 행")

    print(f"\n[5] Anderson {RESIDENTIAL_THRESHOLD:.2f} 컷오프 + 공항동 수동 제외 적용")
    decision = apply_filter(metrics)
    passed = decision[decision["pass_v4"]]
    print(f"  통과: {len(passed)} / 전체: {len(decision)}")

    print("\n[6] 산출물 저장")
    master_path = args.out_dir / "dong_residential_filter_v4.csv"
    pass_path = args.out_dir / "residential_filter_pass_v4.csv"
    decision.to_csv(master_path, index=False, encoding="utf-8-sig")
    passed.to_csv(pass_path, index=False, encoding="utf-8-sig")
    write_summary(decision, args.out_dir)

    print(f"  마스터: {master_path}")
    print(f"  통과 명단: {pass_path}")


if __name__ == "__main__":
    # 실행 예시 (필요 시 주석 해제):
    # import sys
    # sys.argv = [
    #     "residential.py",
    #     "--dong-shp", "data/raw/boundary/bnd_dong_11_2025_2Q.shp",
    #     "--zoning-shp", "data/raw/UPIS/UPIS_C_UQ111.shp",
    #     "--forest-shp", "data/raw/UPIS/UPIS_SHP_CUL220.shp",
    #     "--greenbelt-shp", "data/raw/UPIS/UPIS_C_UQ141.shp",
    #     "--park-shp", "data/raw/UPIS/UPIS_C_UQ142.shp",
    #     "--out-dir", "data/processed/residential_filter",
    # ]
    # main()
    pass
