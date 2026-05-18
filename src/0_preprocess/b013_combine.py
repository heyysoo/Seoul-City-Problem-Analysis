"""B013 통합 변환 (B013.py 출력 → transit_index.py 입력 형식)

처리 목적
- B013.py 산출물(`b013_bus_2023_2025_all.csv`, `b013_subway_2023_2025_all.csv`)을 받아
  transit_index.py 입력 형식(`B013_행정동별_교통이원트랙유입_월별상세.csv`)으로 변환.
- 누락 단계 4가지 채움:
  1. 지하철 역 → 행정동 매핑 (역 좌표 + bnd_dong SHP, 500m 반경 면적 가중)
  2. "지표" 컬럼 파싱 (저녁/심야 * 지하철/버스 * 유입/유출)
  3. 유입만 추출 (저녁_지하철_유입, 심야_지하철_유입, 저녁_버스_유입, 심야_버스_유입)
  4. 행정동×사용년월 wide pivot + 교통 종합(버스+지하철 합산) 산출

설계 배경
- B013.py는 캠퍼스 환경(E:/)에서 raw 거래내역 → ckpt → all → YoY 산출.
- transit_index.py는 캠퍼스 외부에서 월별 상세 데이터를 받아 YoY+클리핑+Min-Max 수행.
- 두 모듈 사이 형식이 다르므로 본 모듈이 가교 역할.

입력
- bus_all_csv:    b013_bus_2023_2025_all.csv
                  컬럼: 사용년월, 행정동ID, 지표, 승객수합계
                  ('지표' 예: "저녁_일반버스_유입", "심야_N버스_유출" 등)
- subway_all_csv: b013_subway_2023_2025_all.csv
                  컬럼: 사용년월, 역ID, 역명, 호선명, 지표, 승객수합계
                  ('지표' 예: "저녁_지하철_유입", "심야_지하철_유출" 등)
- station_csv:    지하철 역 좌표 (서울 열린데이터광장 OA-12914 또는 동일 데이터)
                  컬럼 가정: 역ID, 역명, 위도(WGS84), 경도(WGS84)
- dong_shp:       bnd_dong_11_2025_2Q.shp (행정동 폴리곤, raw)

처리 단계
1. B013.py 두 출력 로드 (long format)
2. 지하철 역ID → 행정동 매핑
   - 역 좌표 → 500m 반경 buffer → 행정동 폴리곤 intersection
   - 면적 비율로 승객수 가중 배분 (한 역이 여러 행정동에 걸치면 면적 비율 분할)
   - 기본 반경은 500m이며, 필요하면 --radius-m으로 조정
3. 지표 컬럼 파싱: "{트랙}_{수단}_{방향}" → (track, mode, direction)
4. 유입만 필터 (direction == "유입")
5. 행정동×사용년월 그룹별로 합산 + wide pivot:
   버스_저녁유입, 버스_심야유입, 지하철_저녁유입, 지하철_심야유입
6. 교통 종합 산출:
   교통_저녁유입 = 버스_저녁유입 + 지하철_저녁유입
   교통_심야유입 = 버스_심야유입 + 지하철_심야유입
   교통_종합유입 = 교통_저녁유입 + 교통_심야유입
7. 행정동명·자치구명 결합 (bnd_dong SHP 속성 또는 별도 마스터)

출력
- B013_행정동별_교통이원트랙유입_월별상세.csv (transit_index.py 입력)
  컬럼: 사용년월, 행정동ID(통계청), 행정동명, 자치구명,
        버스_저녁유입, 버스_심야유입, 지하철_저녁유입, 지하철_심야유입,
        교통_저녁유입, 교통_심야유입, 교통_종합유입

주의
- B013.py의 버스 지표는 `저녁_일반버스_유입`, `심야_N버스_유입`처럼 일반버스/N버스가 분리되어 있다.
  본 변환 단계에서는 두 값을 모두 `버스`로 정규화해 저녁·심야 유입 합계에 포함한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# -----------------------------------------------------------------------------
# 상수
# -----------------------------------------------------------------------------

SUBWAY_INFLUENCE_RADIUS_M = 500

# 지표 파싱 패턴: "{트랙}_{수단}_{방향}"
INDICATOR_TRACKS = ("저녁", "심야")
INDICATOR_MODES = ("버스", "지하철")
INDICATOR_DIRECTIONS = ("유입", "유출")


# -----------------------------------------------------------------------------
# Step 1 — 입력 로드
# -----------------------------------------------------------------------------

def load_b013_long(path: Path, encoding: str = "utf-8-sig") -> pd.DataFrame:
    """B013.py long format csv 로드 (버스 또는 지하철)."""
    df = pd.read_csv(path, encoding=encoding, dtype=str)
    df["사용년월"] = df["사용년월"].astype(str).str.strip()
    df["승객수합계"] = pd.to_numeric(df["승객수합계"], errors="coerce").fillna(0)
    if "행정동ID" in df.columns:
        df["행정동ID"] = df["행정동ID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    if "역ID" in df.columns:
        df["역ID"] = df["역ID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    print(f"  로드 {path.name}: {len(df):,}행")
    return df


# -----------------------------------------------------------------------------
# Step 2 — 지하철 역 → 행정동 매핑 (면적 가중 500m 반경)
# -----------------------------------------------------------------------------

def map_subway_to_dong(
    subway_long: pd.DataFrame,
    station_csv: Path,
    dong_shp: Path,
    radius_m: int = SUBWAY_INFLUENCE_RADIUS_M,
    dong_shp_encoding: str = "cp949",
) -> pd.DataFrame:
    """지하철 역ID → 행정동 매핑 (면적 가중 배분).

    1) 역 좌표 → 500m 반경 buffer (EPSG:5179 미터 단위)
    2) buffer ∩ 행정동 폴리곤 → 면적 비율 산출
    3) 승객수합계를 면적 비율로 분배 → 한 역이 여러 행정동에 split
    """
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("지하철 행정동 매핑은 geopandas 필요") from exc

    # 1) 역 좌표 로드 + Point geometry
    stations = pd.read_csv(station_csv, dtype={"역ID": str})
    required_cols = {"역ID", "위도", "경도"}
    missing = required_cols - set(stations.columns)
    if missing:
        raise KeyError(f"역 좌표 csv 필수 컬럼 없음: {missing}")

    stations["역ID"] = stations["역ID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    stations["위도"] = pd.to_numeric(stations["위도"], errors="coerce")
    stations["경도"] = pd.to_numeric(stations["경도"], errors="coerce")
    stations = stations.dropna(subset=["역ID", "위도", "경도"])

    station_gdf = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations["경도"], stations["위도"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:5179")

    # 2) 500m buffer
    station_gdf["buffer"] = station_gdf.geometry.buffer(radius_m)
    station_gdf = station_gdf.set_geometry("buffer")
    print(f"  지하철 역 {len(station_gdf)}개에 {radius_m}m buffer 생성")

    # 3) 행정동 SHP 로드
    dong = gpd.read_file(dong_shp, encoding=dong_shp_encoding).to_crs("EPSG:5179")
    print(f"  행정동 SHP {len(dong)}개 로드")

    # 4) buffer ∩ 행정동 intersection
    inter = gpd.overlay(
        station_gdf[["역ID", "buffer"]].rename(columns={"buffer": "geometry"}).set_geometry("geometry"),
        dong[["ADM_CD", "geometry"]],
        how="intersection",
    )
    inter["intersect_area_m2"] = inter.geometry.area

    # 5) 역별 buffer 총 면적 + 각 행정동 면적 비율
    station_total_area = (
        inter.groupby("역ID")["intersect_area_m2"].sum().rename("total_buffer_area")
    )
    inter = inter.merge(station_total_area, on="역ID")
    inter["area_ratio"] = inter["intersect_area_m2"] / inter["total_buffer_area"]
    inter["ADM_CD"] = inter["ADM_CD"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(8)
    print(f"  intersection: {len(inter):,}건 (역×행정동)")

    # 6) 승객수합계 면적 가중 배분
    subway_long = subway_long.copy()
    subway_long["역ID"] = subway_long["역ID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    subway_long["승객수합계"] = pd.to_numeric(subway_long["승객수합계"], errors="coerce").fillna(0)
    weighted = subway_long.merge(
        inter[["역ID", "ADM_CD", "area_ratio"]],
        on="역ID", how="inner",
    )
    if weighted.empty:
        raise ValueError("지하철 역ID와 역 좌표/행정동 매핑 결과가 매칭되지 않았습니다.")
    weighted["승객수합계_가중"] = weighted["승객수합계"] * weighted["area_ratio"]

    # 7) 행정동×사용년월×지표 단위 합산
    agg = (
        weighted.groupby(["사용년월", "ADM_CD", "지표"], as_index=False)["승객수합계_가중"]
        .sum()
        .rename(columns={"ADM_CD": "행정동ID", "승객수합계_가중": "승객수합계"})
    )
    print(f"  지하철 행정동 단위 집계: {len(agg):,}행")
    return agg


# -----------------------------------------------------------------------------
# Step 3 — 지표 컬럼 파싱 + 트랙·수단·방향 분리
# -----------------------------------------------------------------------------

def parse_indicator(df: pd.DataFrame, indicator_col: str = "지표") -> pd.DataFrame:
    """지표 컬럼 ("{트랙}_{수단}_{방향}") 파싱 → 3개 컬럼 분리."""
    out = df.copy()
    parts = out[indicator_col].astype(str).str.split("_", expand=True)
    if parts.shape[1] < 3:
        raise ValueError(f"지표 컬럼 파싱 실패: {out[indicator_col].dropna().unique()[:10]}")
    out["track"] = parts[0]
    raw_mode = parts[1]
    out["direction"] = parts[parts.shape[1] - 1]
    out["mode"] = raw_mode.replace({"일반버스": "버스", "N버스": "버스"})

    invalid = out[
        ~out["track"].isin(INDICATOR_TRACKS)
        | ~out["mode"].isin(INDICATOR_MODES)
        | ~out["direction"].isin(INDICATOR_DIRECTIONS)
    ]
    if not invalid.empty:
        sample = invalid[indicator_col].dropna().unique()[:10]
        raise ValueError(f"알 수 없는 지표 형식: {sample}")
    return out


# -----------------------------------------------------------------------------
# Step 4 — 유입만 필터 + wide pivot
# -----------------------------------------------------------------------------

def pivot_to_wide(
    bus_df: pd.DataFrame,
    subway_df: pd.DataFrame,
) -> pd.DataFrame:
    """버스+지하철 결합 후 wide pivot (행정동×사용년월 단위).

    출력 컬럼:
      사용년월, 행정동ID,
      버스_저녁유입, 버스_심야유입, 지하철_저녁유입, 지하철_심야유입,
      교통_저녁유입, 교통_심야유입, 교통_종합유입
    """
    # 유입만 필터
    bus_in = bus_df[bus_df["direction"] == "유입"].copy()
    sub_in = subway_df[subway_df["direction"] == "유입"].copy()
    print(f"  버스 유입: {len(bus_in):,}행 / 지하철 유입: {len(sub_in):,}행")

    # 결합 (양쪽 mode 컬럼 일치하므로 concat)
    combined = pd.concat([bus_in, sub_in], ignore_index=True)
    combined["행정동ID"] = combined["행정동ID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(8)
    combined["승객수합계"] = pd.to_numeric(combined["승객수합계"], errors="coerce").fillna(0)

    # 행정동×사용년월×mode×track 그룹 합산
    grouped = (
        combined.groupby(["사용년월", "행정동ID", "mode", "track"], as_index=False)
        ["승객수합계"].sum()
    )

    # wide pivot: ({mode}_{track}유입) 컬럼 형식
    grouped["col_name"] = grouped["mode"] + "_" + grouped["track"] + "유입"
    wide = grouped.pivot_table(
        index=["사용년월", "행정동ID"],
        columns="col_name",
        values="승객수합계",
        fill_value=0,
    ).reset_index()
    wide.columns.name = None
    for col in ["버스_저녁유입", "버스_심야유입", "지하철_저녁유입", "지하철_심야유입"]:
        if col not in wide.columns:
            wide[col] = 0

    # 교통 종합 산출
    wide["교통_저녁유입"] = wide.get("버스_저녁유입", 0) + wide.get("지하철_저녁유입", 0)
    wide["교통_심야유입"] = wide.get("버스_심야유입", 0) + wide.get("지하철_심야유입", 0)
    wide["교통_종합유입"] = wide["교통_저녁유입"] + wide["교통_심야유입"]

    return wide


# -----------------------------------------------------------------------------
# Step 5 — 행정동명·자치구명 결합 (bnd_dong SHP attribute에서)
# -----------------------------------------------------------------------------

def attach_dong_metadata(
    wide: pd.DataFrame,
    dong_shp: Path,
    encoding: str = "cp949",
) -> pd.DataFrame:
    """행정동ID에 행정동명·자치구명 추가 (bnd_dong SHP attribute)."""
    import geopandas as gpd
    GU_CODE_MAP = {
        "11110": "종로구", "11140": "중구", "11170": "용산구", "11200": "성동구",
        "11215": "광진구", "11230": "동대문구", "11260": "중랑구", "11290": "성북구",
        "11305": "강북구", "11320": "도봉구", "11350": "노원구", "11380": "은평구",
        "11410": "서대문구", "11440": "마포구", "11470": "양천구", "11500": "강서구",
        "11530": "구로구", "11545": "금천구", "11560": "영등포구", "11590": "동작구",
        "11620": "관악구", "11650": "서초구", "11680": "강남구", "11710": "송파구",
        "11740": "강동구",
    }
    gdf = gpd.read_file(dong_shp, encoding=encoding)
    master = gdf[["ADM_CD", "ADM_NM"]].drop_duplicates().copy()
    master["행정동ID"] = master["ADM_CD"].astype(str).str.zfill(8)
    master["행정동명"] = master["ADM_NM"]
    master["자치구명"] = master["행정동ID"].str[:5].map(GU_CODE_MAP)
    return wide.merge(
        master[["행정동ID", "행정동명", "자치구명"]],
        on="행정동ID", how="left",
    )


# -----------------------------------------------------------------------------
# 진입점
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="B013 통합 변환 (B013.py 출력 → transit_index.py 입력 형식)"
    )

    parser.add_argument(
        "--bus-all-csv", type=Path,
        default=Path("data/processed/b013/b013_bus_2023_2025_all.csv"),
        help="B013.py 산출물 (버스 long format)",
    )
    parser.add_argument(
        "--subway-all-csv", type=Path,
        default=Path("data/processed/b013/b013_subway_2023_2025_all.csv"),
        help="B013.py 산출물 (지하철 long format)",
    )
    parser.add_argument(
        "--station-csv", type=Path,
        default=Path("data/raw/b013_subway_station/seoul_subway_station_coords.csv"),
        help="지하철 역 좌표 csv (서울 열린데이터광장, raw). 컬럼: 역ID, 위도, 경도",
    )
    parser.add_argument(
        "--dong-shp", type=Path,
        default=Path("data/raw/bnd_dong_11_2025_2Q/bnd_dong_11_2025_2Q.shp"),
        help="행정동 SHP (raw)",
    )
    parser.add_argument("--dong-shp-encoding", type=str, default="cp949")
    parser.add_argument(
        "--radius-m", type=int, default=SUBWAY_INFLUENCE_RADIUS_M,
        help="지하철 영향권 반경(m). 기본값 500",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("data/raw/b013"),
        help="transit_index.py가 기대하는 raw 위치에 저장",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] B013.py long format 두 csv 로드")
    bus_long = load_b013_long(args.bus_all_csv)
    subway_long = load_b013_long(args.subway_all_csv)

    print(f"\n[2] 지하철 역 → 행정동 매핑 ({args.radius_m}m 반경, 면적 가중)")
    subway_dong = map_subway_to_dong(
        subway_long, args.station_csv, args.dong_shp,
        radius_m=args.radius_m, dong_shp_encoding=args.dong_shp_encoding,
    )

    print("\n[3] 지표 컬럼 파싱 (트랙·수단·방향 분리)")
    bus_parsed = parse_indicator(bus_long)
    subway_parsed = parse_indicator(subway_dong)

    print("\n[4] 유입 필터 + 행정동×월 wide pivot + 교통 종합 산출")
    wide = pivot_to_wide(bus_parsed, subway_parsed)
    print(f"  wide pivot 결과: {len(wide):,}행 (행정동×월)")

    print("\n[5] 행정동명·자치구명 결합")
    final = attach_dong_metadata(wide, args.dong_shp, encoding=args.dong_shp_encoding)

    # 최종 컬럼 순서
    column_order = [
        "사용년월", "행정동ID", "행정동명", "자치구명",
        "버스_저녁유입", "버스_심야유입",
        "지하철_저녁유입", "지하철_심야유입",
        "교통_저녁유입", "교통_심야유입", "교통_종합유입",
    ]
    available = [c for c in column_order if c in final.columns]
    final = final[available]

    out_path = args.out_dir / "B013_행정동별_교통이원트랙유입_월별상세.csv"
    final.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_path} ({len(final):,}행 × {len(final.columns)}컬럼)")
    print("      → transit_index.py 입력으로 바로 사용 가능")


if __name__ == "__main__":
    # 실행 예시 (필요 시 주석 해제):
    # import sys
    # sys.argv = [
    #     "b013_combine.py",
    #     "--bus-all-csv", "data/processed/b013/b013_bus_2023_2025_all.csv",
    #     "--subway-all-csv", "data/processed/b013/b013_subway_2023_2025_all.csv",
    #     "--station-csv", "data/raw/b013_subway_station/seoul_subway_station_coords.csv",
    #     "--dong-shp", "data/raw/bnd_dong_11_2025_2Q/bnd_dong_11_2025_2Q.shp",
    #     "--out-dir", "data/raw/b013",
    # ]
    # main()
    pass
