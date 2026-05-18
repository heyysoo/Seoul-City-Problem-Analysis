"""B079 카드매출 데이터 전처리 (캠퍼스 내부 처리)

처리 목적
- 빅데이터캠퍼스 B079 SEOUL_SIMIN_05_VF 일별 카드매출 데이터를
  행정동×월별 합계 → YoY 증감률 → 서울 중앙값 대비 상대 차이로 변환.
- 트랙별(저녁·심야) 별도 실행 → potential.py 입력으로 사용.

출처
- B079 SEOUL_SIMIN_05_VF (빅데이터캠퍼스 핵심 113종 중 1종)
- 일별 csv, 격자_50 단위, 시간대·업종대분류·카드이용금액계 컬럼 포함

산출물
- monthly/{YYYY}_격자별카드합_{MM}.csv (월별 격자 합계)
- total_card.csv (2023~2025 36개월 통합)
- lookup_grid_to_adm.csv (격자→행정동 룩업)
- 행정동_카드매출.csv (둔촌1동 제외 + 신사동 분리 후)
- growth_{track}.csv ← potential.py 입력
- yoy_full_{track}.csv (전체 YoY 디테일)

트랙별 시간대 블록 (SIMIN_05 시간대 컬럼 4시간 단위 블록 가정)
- 저녁 (evening): block [5] = 17~21시 
- 심야 (late_night): block [1, 6] = 00-06 + 21-24시 → 21-04시 정합
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 상수
# -----------------------------------------------------------------------------

# CSV 파싱 시 카테고리명에 들어있는 쉼표 정규화
# (SIMIN_05 원본의 일부 행에서 카테고리명 안 쉼표가 필드 구분자와 충돌)
CATEGORY_COMMA_FIX = {
    "모텔,여관,기타숙박": "모텔",
    "통신요금(이동,시내전화)": "통신",
    "통신요금(PC통신,무선호출)": "통신2",
}

# 야간 부적합 업종 (외식·소매·여가와 무관 — 도메인 판단)
# 분석결과서 페이지 30 [카드 매출 야간 업종 필터링 로직] 정통:
#   박스 1 (시점 불일치 / 구분 불가):
#     - 요금/금융: 공과금·보험료·금융 서비스
#     - 숙박/여행/교통: 호텔·콘도·항공·철도 등 (사전 예약 위주)
#     - 교육/학원: 학원비·온라인 강의 (결제와 학습 시점 괴리)
#   박스 2 (야간 정책 적용 부적합):
#     - 주유 / 자동차 판매 / 가전·가구 / 의료 / 생활·업무 서비스
#   박스 3 (분석 데이터 포함 예외):
#     - 세탁소·주차장·택시·면세점 → EXCLUDE에 없음 (자동 포함)
# evening_sales.py의 EXCLUDE_CODES와 카테고리 정합성 일치 (다만 OA-22175에는
# 요금/금융·주유 코드 자체가 없어서 코드 갯수 차이 발생, 데이터 구조 차이는 정상).
EXCLUDE_CATEGORIES = (
    # 요금/금융
    "상품권/복권", "보험", "금융/보험", "결제대행(PG)", "전자상거래(다품목취급)",
    "세금공과금", "통신", "통신2", "요금",
    # 자동차·주유
    "주유", "주유소", "LPG가스",
    "자동차서비스", "자동차용품", "자동차서비스/용품", "자동차판매",
    "중고차판매", "신차판매", "수입자동차", "오토바이",
    # 가전·가구
    "사무기기/문구용품", "컴퓨터/소프트웨어", "가전", "가구", "가전/가구",
    "인테리어/건축자재/주방기구",
    # 교육
    "학교등록금", "학원/학습자", "독서실", "유치원",
    # 의료
    "일반병원", "치과병원", "동물병원", "보건소", "한의원", "약국",
    "장례식장/묘지/장의사", "기타의료",
    # 업무·서비스
    "부동산중개", "예식장/결혼서비스",
    "법률/사무서비스", "회계/변리서비스", "연구/번역서비스", "사무기기/컴퓨터",
    # 숙박·교통
    "호텔/콘도", "고속버스/철도/여객선", "모텔", "여행사/항공사",
)

# 트랙별 시간대 블록 (SIMIN_05 시간대 컬럼 코드 기준)
TRACK_TIME_KEEP = {
    "evening": (5,),           # 17-21시 블록
    "late_night": (1, 6),      # 00-06 + 21-24시 블록
}

# 컬럼 매핑 (SIMIN_05 원본 컬럼명)
COL_TIME = "시간대"
COL_CATEGORY = "업종대분류"
COL_GRID = "격자_50"
COL_AMOUNT = "카드이용금액계"

# 행정동 분리·제외 처리
DUNCHON_1_DONG_CODE = "1174069000"  # 둔촌제1동 (재건축으로 카드매출 미관측)
SINSA_GANGNAM_CODE = "1168051000"
SINSA_GWANAK_CODE = "1162068500"


# -----------------------------------------------------------------------------
# 1. 일별 csv 로딩 (쉼표 깨짐 보정)
# -----------------------------------------------------------------------------

def fix_csv_commas(filepath: Path, encoding: str = "utf-8") -> pd.DataFrame:
    """카테고리명 내 쉼표를 정규화한 뒤 CSV 파싱.

    `통신요금(이동,시내전화)` 같이 필드 안에 쉼표가 들어간 카테고리는
    CSV 파서가 필드 경계로 오인하므로, 사전 치환 후 io.StringIO로 읽는다.
    """
    with open(filepath, encoding=encoding) as f:
        text = f.read()
    for old, new in CATEGORY_COMMA_FIX.items():
        text = text.replace(old, new)
    df = pd.read_csv(io.StringIO(text))
    # 일부 셀의 인용부호(') 제거
    if COL_CATEGORY in df.columns:
        df[COL_CATEGORY] = df[COL_CATEGORY].astype(str).str.replace("'", "", regex=False)
    return df


# -----------------------------------------------------------------------------
# 2. 월별 격자 집계
# -----------------------------------------------------------------------------

def extract_month(
    year: int,
    month: int,
    base_dir: Path,
    file_prefix: str,
    time_keep: tuple[int, ...],
    output_dir: Path,
) -> Path | None:
    """단일 월의 일별 csv를 통합해 격자×월 합계 csv 산출."""
    folder = base_dir / str(year) / f"{year}{month:02d}"
    if not folder.exists():
        return None

    frames = []
    for day in range(1, 32):
        date_str = f"{year}{month:02d}{day:02d}"
        filename = f"{file_prefix}{date_str}.csv"
        filepath = folder / filename
        if not filepath.exists():
            continue

        df = fix_csv_commas(filepath)

        # 필터링: 시간대, 업종, 격자 정보없음 제외
        df = df[df[COL_TIME].isin(time_keep)]
        df = df[~df[COL_CATEGORY].isin(EXCLUDE_CATEGORIES)]
        df = df[df[COL_GRID] != "정보없음"]

        frames.append(df)

    if not frames:
        return None

    merged = pd.concat(frames, ignore_index=True)
    result = merged.groupby(COL_GRID, as_index=False)[COL_AMOUNT].sum()
    result["date"] = int(f"{year}{month:02d}")
    result = result.rename(columns={COL_AMOUNT: "sum_card"})

    out_path = output_dir / f"{year}_격자별카드합_{month:02d}.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def extract_all_months(
    years: tuple[int, ...],
    base_dir: Path,
    file_prefix: str,
    time_keep: tuple[int, ...],
    output_dir: Path,
) -> None:
    """연도×월 루프 전체 추출."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        for month in range(1, 13):
            extract_month(year, month, base_dir, file_prefix, time_keep, output_dir)


# -----------------------------------------------------------------------------
# 3. 월별 csv → 전체 통합
# -----------------------------------------------------------------------------

def union_monthly_files(monthly_dir: Path, years: tuple[int, ...], out_path: Path) -> pd.DataFrame:
    """월별 격자 합계 csv들을 단일 데이터프레임으로 통합."""
    frames = []
    for year in years:
        for month in range(1, 13):
            filepath = monthly_dir / f"{year}_격자별카드합_{month:02d}.csv"
            if filepath.exists():
                frames.append(pd.read_csv(filepath, encoding="utf-8"))

    if not frames:
        raise FileNotFoundError(f"{monthly_dir}에 월별 csv 없음")

    merged = pd.concat(frames, ignore_index=True)
    result = merged.sort_values(["date", COL_GRID]).reset_index(drop=True)
    result = result[["date", COL_GRID, "sum_card"]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    return result


# -----------------------------------------------------------------------------
# 4. 격자 → 행정동 룩업 (spatial join, 한 번만 만들고 재사용)
# -----------------------------------------------------------------------------

def build_grid_lookup(
    grid_shp: Path,
    adm_shp: Path,
    grid_id_col: str,
    adm_code_col: str,
    out_path: Path,
) -> pd.DataFrame:
    """격자 폴리곤 중심점 → 행정동 spatial join → 룩업 테이블 생성.

    Point-in-Polygon 으로 1:1 매핑 보장. 경계 위 점은 nearest로 보완.
    """
    grid = gpd.read_file(grid_shp, encoding="utf8")
    adm = gpd.read_file(adm_shp)

    if adm.crs is None:
        adm = adm.set_crs("EPSG:5179")
    if grid.crs != adm.crs:
        grid = grid.to_crs(adm.crs)

    # 폴리곤 → 중심점 (N:N 매핑 회피)
    grid_pts = grid.copy()
    grid_pts["geometry"] = grid.geometry.centroid

    joined = gpd.sjoin(
        grid_pts[[grid_id_col, "geometry"]],
        adm[[adm_code_col, "geometry"]],
        how="left",
        predicate="within",
    )

    # 경계 위 점 nearest 보완
    unmatched = joined[adm_code_col].isna()
    if unmatched.any():
        nearest = gpd.sjoin_nearest(
            grid_pts.loc[unmatched, [grid_id_col, "geometry"]],
            adm[[adm_code_col, "geometry"]],
            how="left",
        )
        # 중복 인덱스 dedup (등거리 폴리곤 다수일 경우)
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        joined.loc[unmatched, adm_code_col] = nearest[adm_code_col].values

    lookup = (
        joined[[grid_id_col, adm_code_col]]
        .drop_duplicates(subset=[grid_id_col])
        .reset_index(drop=True)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lookup.to_csv(out_path, index=False, encoding="utf-8-sig")
    return lookup


# -----------------------------------------------------------------------------
# 5. 격자 매출 → 행정동 매출
# -----------------------------------------------------------------------------

def map_to_adm(
    total_card: pd.DataFrame,
    lookup: pd.DataFrame,
    grid_id_col: str,
    adm_code_col: str,
) -> pd.DataFrame:
    """격자별 카드매출에 행정동 코드 부여 후 동 단위 집계."""
    merged = total_card.merge(
        lookup,
        left_on=COL_GRID,
        right_on=grid_id_col,
        how="left",
    )
    merged = merged[["date", adm_code_col, "sum_card"]].rename(columns={adm_code_col: "amd_code"})

    result = (
        merged.dropna(subset=["amd_code"])
        .groupby(["date", "amd_code"], as_index=False)["sum_card"]
        .sum()
        .sort_values(["date", "amd_code"])
        .reset_index(drop=True)
    )
    result["amd_code"] = result["amd_code"].astype(str)
    return result


# -----------------------------------------------------------------------------
# 6. Cleanup: 둔촌1동 제외 + 신사동 분리
# -----------------------------------------------------------------------------

def apply_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """둔촌1동 제외 (재건축으로 카드매출 미관측) + 신사동 두 곳 자치구 분리."""
    out = df.copy()
    out["amd_code"] = out["amd_code"].astype(str)

    # 둔촌1동 제외
    out = out[out["amd_code"] != DUNCHON_1_DONG_CODE]

    # 신사동 분리 — amd_name 컬럼 생성 (다른 동은 amd_code만 사용)
    out["amd_name"] = None
    out.loc[out["amd_code"] == SINSA_GWANAK_CODE, "amd_name"] = "신사(관악구)"
    out.loc[out["amd_code"] == SINSA_GANGNAM_CODE, "amd_name"] = "신사(강남구)"

    return out.reset_index(drop=True)


# -----------------------------------------------------------------------------
# 7. YoY 계산
# -----------------------------------------------------------------------------

def calc_yoy(
    df: pd.DataFrame,
    region_col: str = "amd_code",
    date_col: str = "date",
    value_col: str = "sum_card",
) -> pd.DataFrame:
    """전년 동월 대비 증감률 계산."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col].astype(str), format="%Y%m")
    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month

    # 전년 테이블 (year+1로 시프트 후 self-join)
    prev = (
        out[[region_col, "year", "month", value_col]]
        .assign(year=lambda x: x["year"] + 1)
        .rename(columns={value_col: f"{value_col}_prev"})
    )
    merged = out.merge(prev, on=[region_col, "year", "month"], how="left")

    # 0 분모 회피
    denom = merged[f"{value_col}_prev"].replace(0, np.nan)
    merged["yoy_diff"] = merged[value_col] - merged[f"{value_col}_prev"]
    merged["yoy_pct"] = (merged[value_col] / denom - 1) * 100

    return merged


def summarize_yoy(
    df_yoy: pd.DataFrame,
    region_col: str = "amd_code",
    track_name: str = "evening",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """동별 YoY 평균/표준편차 + 서울 중앙값 대비 상대 차이.

    Returns:
        (summary_df, growth_df)
        - summary_df: 동별 YoY 통계 전체 컬럼
        - growth_df: potential.py 입력용 (amd_code + growth_{track_name})
    """
    summary = (
        df_yoy.groupby(region_col)["yoy_pct"]
        .agg(YoY_평균="mean", YoY_표준편차="std", 계산수="count")
        .reset_index()
        .sort_values("YoY_평균")
    )

    seoul_median = round(df_yoy["yoy_pct"].median(), 2)
    summary["seoul_yoy"] = seoul_median
    summary["YoY_상대_차이"] = summary["YoY_평균"] - summary["seoul_yoy"]
    summary = summary.round(2)

    # potential.py 입력용 (트랙별 컬럼명)
    growth_col = f"growth_{track_name}"
    growth = summary[[region_col, "YoY_상대_차이"]].rename(
        columns={"YoY_상대_차이": growth_col}
    )

    return summary, growth


# -----------------------------------------------------------------------------
# 진입점
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B079 카드매출 전처리 + YoY 산출")

    parser.add_argument(
        "--track",
        type=str,
        choices=["evening", "late_night"],
        required=True,
        help="저녁 또는 심야 트랙",
    )

    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(r"E:/카드"),
        help="B079 일별 csv 루트 디렉터리 (캠퍼스)",
    )
    parser.add_argument("--file-prefix", type=str, default="SEOUL_SIMIN_05_VF_")
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2023, 2024, 2025],
    )

    # SHP
    parser.add_argument(
        "--grid-shp",
        type=Path,
        default=Path(r"E:/50m격자 shp"),
        help="격자 50m SHP (행안부 기준)",
    )
    parser.add_argument(
        "--adm-shp",
        type=Path,
        default=Path(r"E:/서울시_행정동경계_2023년10월"),
        help="서울시 행정동 경계 SHP",
    )
    parser.add_argument("--grid-id-col", type=str, default="SPO_NO_CD")
    parser.add_argument("--adm-code-col", type=str, default="EMD_CD")

    # 출력
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="산출 디렉터리. 미지정 시 base_dir/b079_processed/{track} 으로 생성.",
    )

    # 단계별 스킵 (기존 산출물 재사용)
    parser.add_argument("--skip-extract", action="store_true", help="월별 추출 단계 건너뛰기")
    parser.add_argument("--skip-union", action="store_true", help="월별 통합 단계 건너뛰기")
    parser.add_argument("--skip-lookup", action="store_true", help="격자 룩업 생성 건너뛰기")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = args.out_dir or (args.base_dir.parent / "b079_processed" / args.track)
    out_dir.mkdir(parents=True, exist_ok=True)
    monthly_dir = out_dir / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)

    time_keep = TRACK_TIME_KEEP[args.track]

    # [1] 일별 → 월별 격자 합계
    if not args.skip_extract:
        extract_all_months(
            years=tuple(args.years),
            base_dir=args.base_dir,
            file_prefix=args.file_prefix,
            time_keep=time_keep,
            output_dir=monthly_dir,
        )

    # [2] 월별 → 전체 통합
    total_card_path = out_dir / "total_card.csv"
    if not args.skip_union:
        total_card = union_monthly_files(monthly_dir, tuple(args.years), total_card_path)
    else:
        total_card = pd.read_csv(total_card_path)

    # [3] 격자 → 행정동 룩업 (한 번만 생성)
    lookup_path = out_dir / "lookup_grid_to_adm.csv"
    if not args.skip_lookup:
        lookup = build_grid_lookup(
            grid_shp=args.grid_shp,
            adm_shp=args.adm_shp,
            grid_id_col=args.grid_id_col,
            adm_code_col=args.adm_code_col,
            out_path=lookup_path,
        )
    else:
        lookup = pd.read_csv(lookup_path)

    # [4] 격자 매출 → 행정동 매출
    by_dong = map_to_adm(total_card, lookup, args.grid_id_col, args.adm_code_col)

    # [5] Cleanup: 둔촌1동 제외 + 신사동 분리
    by_dong = apply_cleanup(by_dong)
    cleaned_path = out_dir / "행정동_카드매출.csv"
    by_dong.to_csv(cleaned_path, index=False, encoding="utf-8-sig")

    # [6] YoY 계산
    df_yoy = calc_yoy(by_dong, region_col="amd_code", date_col="date", value_col="sum_card")
    yoy_full_path = out_dir / f"yoy_full_{args.track}.csv"
    df_yoy.to_csv(yoy_full_path, index=False, encoding="utf-8-sig")

    # [7] 동별 YoY 요약 + potential.py 입력용 growth csv
    summary, growth = summarize_yoy(df_yoy, region_col="amd_code", track_name=args.track)
    summary_path = out_dir / f"카드매출YoY_{args.track}.csv"
    growth_path = out_dir / f"growth_{args.track}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    growth.to_csv(growth_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    # 실행 예시 (필요 시 주석 해제, 캠퍼스 환경에서 트랙별 2회 실행):
    # import sys
    # sys.argv = [
    #     "b079.py",
    #     "--track", "evening",
    #     "--base-dir", r"E:/카드",
    #     "--grid-shp", r"E:/50m격자 shp",
    #     "--adm-shp", r"E:/서울시_행정동경계_2023년10월",
    #     "--out-dir", r"E:/b079_processed/evening",
    # ]
    # main()
    #
    # sys.argv[2] = "late_night"
    # sys.argv[-1] = r"E:/b079_processed/late_night"
    # main()
    pass
