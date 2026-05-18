"""보안등 밀도 역수 (light_void) — 인프라 공백 지수 세 번째 변수 (가중치 25%)

처리 목적
- 서울시 25개 자치구 원본 보안등 CSV에서 전체 파이프라인 자체 실행.
- 좌표 정리 → spatial join → 행정동 집계 → 결측 보정 → Min-Max 정규화 → 1-x 변환.
- 값이 클수록 보안등 인프라 공백 (취약 지역).

학술 근거 (가중치 25%)
- Farrington & Welsh (2002) 13개 평가 연구: 거리 조명 20% 범죄 감소
- Welsh, Farrington & Douglas (2022) 21개 평가 연구: 14% 범죄 감소
- Doleac & Sanders (2015) DST 자연 실험: 일조 1시간 ↑ → 강도 7% ↓

처리 흐름 (5단계, 원자료 → 최종 산출, 노션 §3 정통 12단계 통합)
1. 25개 자치구 CSV 통합 + 좌표 컬럼 정규화 (노션 Step 1-2)
2. 한국 영역 좌표 필터 + WGS84(EPSG:4326) → 미터 단위 CRS(EPSG:5179)
   ※ 좌표 없는 4구 (동대문·마포·송파·용산) → light_geocode.py 모듈에서 처리 (노션 Step 5-7)
3. Two-pass spatial join: within → sjoin_nearest (중복 인덱스 dedup) (노션 Step 8-10)
4. 행정동×보안등 개수 집계 + 면적·밀도 계산 (노션 Step 11)
5. 결측 보정 4가지 케이스 (도봉·서대문·송파·성동) → Min-Max → 1-x 변환 (노션 Step 12)

결측 보정 4가지 케이스
- measured (383동): 실측 보안등 수 기반
- gu_avg_density (26동): 자치구 평균 밀도 적용 (도봉·서대문·송파)
- adj_gu_avg (17동): 성동구 — 인접 4구(광진·동대문·중구·용산) 평균
  (성동 원본 3.42 개/km²로 서울 평균 522의 0.7% — 데이터 오류 판단)

산식
- light_density = streetlight_count / area_km2
- light_density_norm = (density - min) / (max - min)  (Min-Max 정규화)
- light_void = 1 - light_density_norm  (CCTV와 동일 방향 반전)

산출
- streetlight_count_dong.csv (검증용 중간 산출): ADM_CD, ADM_NM, count, mapping_method
- streetlight_void_dong.csv (최종, 426행 * 9열): dong_code, dong_name, gu_name,
  area_km2, light_density, light_density_norm, light_void, coverage_issue,
  mapping_method

주의
- 25개 자치구 CSV는 컬럼 스키마가 통일되어 있지 않음 → COLUMN_ALIASES로 매핑
- 좌표 컬럼이 없거나 잘못된 자치구는 좌표 미보유 자치구로 처리 (총계만 유지)
- 행정동 SHP가 면적·매핑 기준
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


# -----------------------------------------------------------------------------
# 상수
# -----------------------------------------------------------------------------

# 좌표 유효성 (한국 영역)
LAT_RANGE = (33.0, 39.0)
LON_RANGE = (124.0, 132.0)

# 컬럼명 후보 (자치구별로 다양함 → 표준명으로 통일)
COLUMN_ALIASES = {
    "lat": ["위도", "WGS84위도", "lat", "LAT", "Latitude", "Y좌표", "Y_COORD", "Y"],
    "lon": ["경도", "WGS84경도", "lon", "LON", "lng", "Longitude", "X좌표", "X_COORD", "X"],
    "address": ["주소", "도로명주소", "지번주소", "address", "ADDRESS", "ADDR"],
    "gu": ["자치구", "자치구명", "구", "구명", "GU", "관할구"],
}

# 결측 처리 대상
GU_NAN_PROCESS = ("도봉구", "서대문구", "송파구")
SEONGDONG_GU = "성동구"
ADJ_4_GU = ("광진구", "동대문구", "중구", "용산구")


# -----------------------------------------------------------------------------
# Step 1 — 25개 자치구 CSV 통합
# -----------------------------------------------------------------------------

def resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """후보 컬럼명 중 실제 존재하는 것 반환."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_one_gu_csv(
    path: Path,
    encoding_candidates: tuple[str, ...] = ("utf-8-sig", "cp949", "utf-8"),
) -> pd.DataFrame:
    """단일 자치구 CSV 로드 + 컬럼명 표준화.

    encoding 추측: utf-8-sig → cp949 → utf-8 순서.
    """
    for enc in encoding_candidates:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError(
            "utf-8", b"", 0, 1, f"{path.name}: 인코딩 후보 모두 실패"
        )

    out = pd.DataFrame({"_source_file": [path.name] * len(df)})
    lat_col = resolve_column(df, COLUMN_ALIASES["lat"])
    lon_col = resolve_column(df, COLUMN_ALIASES["lon"])
    addr_col = resolve_column(df, COLUMN_ALIASES["address"])
    gu_col = resolve_column(df, COLUMN_ALIASES["gu"])

    out["lat"] = pd.to_numeric(df[lat_col], errors="coerce") if lat_col else pd.NA
    out["lon"] = pd.to_numeric(df[lon_col], errors="coerce") if lon_col else pd.NA
    out["address"] = df[addr_col].astype(str) if addr_col else ""
    out["gu"] = df[gu_col].astype(str) if gu_col else ""

    return out


def load_all_gu_csvs(raw_dir: Path) -> pd.DataFrame:
    """raw_dir 안의 모든 csv 파일 로드 + 통합."""
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"{raw_dir}에 csv 파일 없음")

    frames = []
    for fp in csv_files:
        try:
            df = load_one_gu_csv(fp)
            print(f"  로드: {fp.name} ({len(df):,}건)")
            frames.append(df)
        except Exception as e:
            print(f" {fp.name} 실패: {e}")

    merged = pd.concat(frames, ignore_index=True)
    print(f"  통합: {len(merged):,}건 (25개 자치구)")
    return merged


# -----------------------------------------------------------------------------
# Step 2 — 좌표 검증
# -----------------------------------------------------------------------------

def validate_coords(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """좌표 유효성 분리.

    Returns:
        (valid_df, invalid_df)
        - valid_df: lat·lon 모두 유효한 한국 영역 좌표
        - invalid_df: 좌표 결측 또는 한국 영역 밖 (geocoding 필요 → 일단 제외)
    """
    out = df.copy()
    has_coords = out["lat"].notna() & out["lon"].notna()
    in_korea = (
        out["lat"].between(*LAT_RANGE) & out["lon"].between(*LON_RANGE)
    )
    valid_mask = has_coords & in_korea

    valid = out[valid_mask].copy()
    invalid = out[~valid_mask].copy()

    print(f"  좌표 유효: {len(valid):,}건")
    print(f"  좌표 무효: {len(invalid):,}건 (별도 geocoding 필요, 본 산출에서는 제외)")
    return valid, invalid


# -----------------------------------------------------------------------------
# Step 3 — Spatial join (two-pass: within → nearest)
# -----------------------------------------------------------------------------

def build_geodataframe(
    df: pd.DataFrame,
    target_crs: str = "EPSG:5179",
) -> gpd.GeoDataFrame:
    """좌표 valid DataFrame → GeoDataFrame (WGS84 → 미터 CRS 변환)."""
    geom = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")
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


def spatial_join_two_pass(
    points: gpd.GeoDataFrame,
    dong: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Within → Nearest fallback. nearest 중복 인덱스 dedup 포함."""
    joined = gpd.sjoin(
        points,
        dong[["ADM_CD", "ADM_NM", "geometry"]],
        how="left",
        predicate="within",
    )
    joined["mapping_method"] = "within_dong"

    unmatched_idx = joined[joined["ADM_CD"].isna()].index
    n_unmatched = len(unmatched_idx)
    if n_unmatched > 0:
        print(f"  within 매핑 실패 {n_unmatched:,}건 → sjoin_nearest fallback")

        nearest = gpd.sjoin_nearest(
            points.loc[unmatched_idx],
            dong[["ADM_CD", "ADM_NM", "geometry"]],
            how="left",
            distance_col="nearest_dist_m",
        )[["ADM_CD", "ADM_NM", "nearest_dist_m"]]

        # 등거리 폴리곤 다수 시 같은 포인트 행 중복 → 첫 번째만
        nearest = nearest[~nearest.index.duplicated(keep="first")]

        joined.loc[unmatched_idx, "ADM_CD"] = nearest["ADM_CD"]
        joined.loc[unmatched_idx, "ADM_NM"] = nearest["ADM_NM"]
        joined.loc[unmatched_idx, "mapping_method"] = "nearest_dong"

    # index_right 정리
    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])

    return joined


# -----------------------------------------------------------------------------
# Step 4 — 행정동 집계 + 밀도
# -----------------------------------------------------------------------------

def aggregate_per_dong(
    joined: pd.DataFrame,
    dong_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """행정동별 보안등 개수 집계 + 426동 LEFT JOIN + 면적·밀도 산출."""
    counts = (
        joined.dropna(subset=["ADM_CD"])
        .groupby(["ADM_CD"], as_index=False)
        .size()
        .rename(columns={"size": "streetlight_count"})
    )
    counts["ADM_CD"] = counts["ADM_CD"].astype(str).str.zfill(8)

    # 면적 산출
    area_df = dong_gdf[["ADM_CD", "ADM_NM"]].copy()
    area_df["area_km2"] = dong_gdf.geometry.area / 1e6
    area_df["ADM_CD"] = area_df["ADM_CD"].astype(str).str.zfill(8)

    # 자치구명 (ADM_NM 앞 글자 또는 SGG_NM 컬럼 사용)
    if "SGG_NM" in dong_gdf.columns:
        area_df["gu_name"] = dong_gdf["SGG_NM"].values
    else:
        # ADM_NM에서 자치구 추출 어렵다면 NaN으로
        area_df["gu_name"] = ""

    # 426동 LEFT JOIN
    result = area_df.merge(counts, on="ADM_CD", how="left")
    result["streetlight_count"] = result["streetlight_count"].fillna(0).astype(int)
    result["streetlight_count"] = result["streetlight_count"].where(
        result["streetlight_count"] > 0,
    )  # 0 → NaN (보정 대상으로 표시)

    result["light_density"] = result["streetlight_count"] / result["area_km2"]

    return result


# -----------------------------------------------------------------------------
# Step 5 — 결측 보정 4가지 케이스
# -----------------------------------------------------------------------------

def apply_gu_avg_density(df: pd.DataFrame) -> pd.DataFrame:
    """도봉·서대문·송파구 NaN 동 → 자치구 평균 밀도 적용."""
    out = df.copy()
    out["mapping_method"] = "measured"
    out["light_density_final"] = out["light_density"]

    nan_mask = out["light_density"].isna() & out["gu_name"].isin(GU_NAN_PROCESS)
    out.loc[nan_mask, "mapping_method"] = "gu_avg_density"

    for gu in GU_NAN_PROCESS:
        gu_mask = out["gu_name"] == gu
        valid_mask = gu_mask & out["light_density"].notna()
        target_mask = gu_mask & out["light_density"].isna()

        n_valid = valid_mask.sum()
        n_target = target_mask.sum()
        if n_target == 0:
            continue
        if n_valid == 0:
            print(f" {gu}: 실측 동 없음 — 건너뜀")
            continue

        gu_avg = out.loc[valid_mask, "light_density"].mean()
        out.loc[target_mask, "light_density_final"] = gu_avg
        print(f"  {gu}: {n_target}동 → 구 평균 {gu_avg:.2f} 개/km² 적용 (실측 {n_valid}동)")

    return out


def apply_adj_gu_avg(df: pd.DataFrame) -> pd.DataFrame:
    """성동구 17동 전체 → 인접 4구 평균 밀도 (데이터 오류 특수 처리)."""
    out = df.copy()

    adj_mask = out["gu_name"].isin(ADJ_4_GU) & out["light_density"].notna()
    if adj_mask.sum() == 0:
        print(" 인접 4구 실측 동 없음 — 성동 보정 건너뜀")
        return out
    adj_avg = out.loc[adj_mask, "light_density"].mean()

    sd_mask = out["gu_name"] == SEONGDONG_GU
    n_seongdong = sd_mask.sum()
    out.loc[sd_mask, "light_density_final"] = adj_avg
    out.loc[sd_mask, "mapping_method"] = "adj_gu_avg"
    print(f"  성동구: {n_seongdong}동 전체에 인접 4구 평균 {adj_avg:.2f} 개/km² 적용")
    return out


def compute_void(df: pd.DataFrame) -> pd.DataFrame:
    """Min-Max 정규화 → 1−x 변환."""
    out = df.copy()
    d_min = out["light_density_final"].min()
    d_max = out["light_density_final"].max()
    out["light_density_norm"] = (out["light_density_final"] - d_min) / (d_max - d_min)
    out["light_void"] = 1 - out["light_density_norm"]
    return out


def prepare_output(df: pd.DataFrame) -> pd.DataFrame:
    """최종 산출 컬럼 정리."""
    out = df[[
        "ADM_CD", "ADM_NM", "gu_name", "area_km2",
        "light_density_final", "light_density_norm", "light_void",
        "mapping_method",
    ]].copy()
    out = out.rename(columns={
        "ADM_CD": "dong_code",
        "ADM_NM": "dong_name",
        "light_density_final": "light_density",
    })
    out["coverage_issue"] = (out["mapping_method"] != "measured").astype(int)
    return out


# -----------------------------------------------------------------------------
# 진입점
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="보안등 밀도 역수 (light_void) 산출 — 전체 파이프라인")

    parser.add_argument(
        "--raw-dir", type=Path,
        default=Path("data/raw/streetlight/raw_gu"),
        help="25개 자치구 원본 CSV 디렉터리",
    )
    parser.add_argument(
        "--geocoded-supplement", type=Path,
        default=None,
        help="좌표 없는 4구(동대문·마포·송파·용산) geocoding 결과 csv (선택). "
             "light_geocode.py 산출물. 미지정 시 좌표 있는 데이터로만 처리.",
    )
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


def load_geocoded_supplement(
    path: Path,
    encoding_candidates: tuple[str, ...] = ("utf-8-sig", "cp949", "utf-8"),
) -> pd.DataFrame:
    """좌표 없는 4구의 geocoding 결과 csv 로드.

    좌표 없는 4구(동대문·마포·송파·용산)는 light_geocode.py에서 주소→위경도 geocoding 처리한 산출물.
    산출 파일 ("서울시_보안등_좌표0개구_행정동매핑_용산반영.csv") 형식 가정:
    - 좌표 부족한 보안등에 행정동 코드(ADM_CD)가 부여된 결과
    - 컬럼: ADM_CD, ADM_NM, count 또는 행별 데이터
    """
    for enc in encoding_candidates:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError(
            "utf-8", b"", 0, 1, f"{path.name}: 인코딩 후보 모두 실패"
        )

    # ADM_CD 컬럼이 있는지 확인
    if "ADM_CD" in df.columns:
        df["ADM_CD"] = df["ADM_CD"].astype(str).str.zfill(8)
        print(f"  geocoding 보강 csv 로드: {len(df)}건 (4구 geocoded)")
        return df
    else:
        print(f"  geocoding csv에 ADM_CD 컬럼 없음 — 건너뜀")
        return pd.DataFrame()


def merge_geocoded_into_joined(
    joined: pd.DataFrame,
    geocoded: pd.DataFrame,
) -> pd.DataFrame:
    """spatial join 결과에 geocoding 결과를 추가 행으로 결합.

    geocoded는 이미 ADM_CD가 부여된 상태. mapping_method='geocoded_dong'으로 표시.
    """
    if len(geocoded) == 0:
        return joined

    if "ADM_CD" not in geocoded.columns:
        print(" geocoded csv에 ADM_CD 컬럼 없음 — 결합 건너뜀")
        return joined

    # 필요 컬럼만 추출 + mapping_method 부여
    aux = geocoded[["ADM_CD"]].copy()
    if "ADM_NM" in geocoded.columns:
        aux["ADM_NM"] = geocoded["ADM_NM"]
    aux["mapping_method"] = "geocoded_dong"

    print(f"  geocoding 결과 결합: +{len(aux)}건 (총 {len(joined) + len(aux)}건)")
    return pd.concat([joined, aux], ignore_index=True)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] 25개 자치구 CSV 통합")
    merged = load_all_gu_csvs(args.raw_dir)

    print("\n[2] 좌표 검증")
    valid, invalid = validate_coords(merged)

    if len(invalid) > 0:
        invalid_path = args.out_dir / "streetlight_invalid_coords.csv"
        invalid.to_csv(invalid_path, index=False, encoding="utf-8-sig")
        print(f"  무효 좌표 저장: {invalid_path}")
        print(f"  → light_geocode.py 또는 --geocoded-supplement 활용")

    print("\n[3] 행정동 SHP 로드 + Spatial join")
    dong_gdf = load_dong_shp(args.dong_shp, encoding=args.dong_encoding,
                              target_crs=args.target_crs)
    print(f"  행정동: {len(dong_gdf)}개")

    points = build_geodataframe(valid, target_crs=args.target_crs)
    joined = spatial_join_two_pass(points, dong_gdf)

    print("\n[3.5] 좌표 없는 4구 geocoding 결과 추가 결합 (--geocoded-supplement 제공 시)")
    if args.geocoded_supplement and args.geocoded_supplement.exists():
        geocoded = load_geocoded_supplement(args.geocoded_supplement)
        joined = merge_geocoded_into_joined(joined, geocoded)
    else:
        print("  --geocoded-supplement 미제공 — 좌표 있는 데이터로만 처리")
        print("  동대문·마포·송파·용산 일부 동의 보안등 수 과소 가능성")

    print("\n[4] 행정동 집계 + 면적·밀도 계산")
    aggregated = aggregate_per_dong(joined, dong_gdf)

    # 중간 산출물 저장
    count_path = args.out_dir / "streetlight_count_dong.csv"
    aggregated.to_csv(count_path, index=False, encoding="utf-8-sig")
    print(f"  중간 산출 저장: {count_path}")

    print("\n[5] 결측 보정 (도봉·서대문·송파·성동)")
    boosted = apply_gu_avg_density(aggregated)
    boosted = apply_adj_gu_avg(boosted)

    print("\n[6] Min-Max 정규화 → 공백도 산출")
    voided = compute_void(boosted)
    result = prepare_output(voided)

    out_path = args.out_dir / "streetlight_void_dong.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_path} ({len(result)}행 × {len(result.columns)}열)")

    # 검증
    print("\n분포 통계:")
    print(f"  light_void 평균: {result['light_void'].mean():.4f}")
    print(f"  light_void 중앙값: {result['light_void'].median():.4f}")
    print(f"  mapping_method 분포:")
    for method, n in result["mapping_method"].value_counts().items():
        print(f"    {method:20s}: {n}동")


if __name__ == "__main__":
    # 실행 예시:
    # import sys
    # sys.argv = [
    #     "light_void.py",
    #     "--raw-dir", "data/raw/streetlight/raw_gu",
    #     "--dong-shp", "data/raw/bnd_dong_11_2025_2Q/bnd_dong_11_2025_2Q.shp",
    #     "--out-dir", "data/processed/infra_inputs",
    # ]
    # main()
    pass
