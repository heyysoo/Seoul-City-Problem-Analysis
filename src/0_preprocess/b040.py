"""B040 내국인 생활인구 전처리 (캠퍼스 streaming)

처리 목적
- 빅데이터캠퍼스 B040 일별 텍스트 파일에서 행정동×시간대 생활인구를 streaming 집계.
- 트랙별(저녁·심야) 시간대 필터 → 행정동×월별 평균 인구 → gap_index 입력 산출.
- 분석 연도 2023~2025 (36개월).

출처
- B040 내국인 생활인구 (빅데이터캠퍼스 핵심 113종 중 1종, 오프라인 환경)
- 파일 구조: E:/YYYYMM/YYYYMM/TN_PF_PPS_SPOP_LOCAL_RESD_YYYYMMDD.txt
- 형식: pipe(|) delimited, backtick(`) wrapped 값
- 컬럼 인덱스: 0=날짜(stdr_de_id), 1=시간대(tmzon_pd_se),
              2=행정동코드(adstrd_code_se), 4=총생활인구(tot_lvpop_co)

트랙별 시간대 (4시간 블록이 아닌 hourly 데이터 가정)
- evening:    17, 18, 19, 20시        (저녁 트랙 17~20)
- late_night: 21, 22, 23, 0~5시        (심야 트랙 21~05)

처리 흐름
1. 일별 텍스트 파일 streaming (메모리 안전)
2. 트랙 시간대 필터 적용
3. (날짜, 행정동) 단위로 시간대 인구 합 누적
4. 월별로 (행정동, 일수) → 평균
5. 산출:
   - b040_monthly_{track}_pop.csv (행정동×월별 평균 인구, 캠퍼스 internal)
   - b040_monthly_{track}_share.csv (응용집계 비율, exportable)

캠퍼스 반출 정책
- 절대값(평균 인구 등)은 반출 불가 (감수 정책)
- 비율·증감률·범주만 응용집계로 반출 가능
- 본 모듈은 두 산출물 분리 저장 — share만 반출 가능

결측일 (이전 세션에서 확인)
- 20230512, 20230825, 20230922, 20230924 (B040 원본에 부재)
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


# -----------------------------------------------------------------------------
# 상수
# -----------------------------------------------------------------------------

# B040 컬럼 인덱스 (pipe-delimited 텍스트)
IDX_DATE = 0
IDX_HOUR = 1
IDX_DONG = 2
IDX_POP = 4

# 트랙별 시간대 (문자열 비교)
TRACK_HOURS = {
    "evening": {"17", "18", "19", "20"},
    "late_night": {
        "21", "22", "23",
        "00", "01", "02", "03", "04", "05",
    },
}

# 결측일 (B040 원본 부재)
KNOWN_MISSING_DATES = frozenset({
    "20230512", "20230825", "20230922", "20230924",
})

FILE_PREFIX = "TN_PF_PPS_SPOP_LOCAL_RESD_"


# -----------------------------------------------------------------------------
# 헬퍼
# -----------------------------------------------------------------------------

def strip_backtick(raw: str) -> str:
    """백틱 래핑 + 공백 제거."""
    return raw.strip().strip("`").strip()


def collect_files(base_dir: Path, years: tuple[int, ...]) -> list[Path]:
    """E:/YYYYMM/YYYYMM/TN_PF_*.txt 패턴 파일 목록 수집."""
    files: list[Path] = []
    for year in years:
        for month in range(1, 13):
            yyyymm = f"{year}{month:02d}"
            folder = base_dir / yyyymm / yyyymm
            if not folder.exists():
                continue
            for f in sorted(folder.iterdir()):
                if f.name.startswith(FILE_PREFIX) and f.suffix == ".txt":
                    files.append(f)
    return files


def extract_date_from_filename(filename: str) -> str | None:
    """파일명에서 YYYYMMDD 추출."""
    # TN_PF_PPS_SPOP_LOCAL_RESD_20230101.txt → 20230101
    stem = filename.replace(FILE_PREFIX, "").replace(".txt", "")
    return stem if (len(stem) == 8 and stem.isdigit()) else None


# -----------------------------------------------------------------------------
# 핵심 streaming
# -----------------------------------------------------------------------------

def stream_single_file(
    filepath: Path,
    target_hours: set[str],
    encoding: str = "utf-8",
) -> dict[str, float]:
    """단일 파일 streaming → 행정동별 시간대 인구 합 dict 반환.

    헤더 자동 판별: 첫 줄의 첫 컬럼이 isdigit + len==8 이면 데이터 행, 아니면 헤더.
    """
    accum: dict[str, float] = defaultdict(float)

    with open(filepath, encoding=encoding) as f:
        first = True
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) <= IDX_POP:
                continue

            date_val = strip_backtick(parts[IDX_DATE])

            # 첫 줄이 헤더면 skip
            if first:
                first = False
                if not (len(date_val) == 8 and date_val.isdigit()):
                    continue

            hour = strip_backtick(parts[IDX_HOUR]).zfill(2)
            if hour not in target_hours:
                continue

            dong = strip_backtick(parts[IDX_DONG])
            try:
                pop = float(strip_backtick(parts[IDX_POP]))
            except ValueError:
                continue

            accum[dong] += pop

    return dict(accum)


def stream_all(
    files: list[Path],
    target_hours: set[str],
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """전체 파일 streaming → (date, dong) → 시간대 인구 합 누적."""
    # {(YYYYMMDD, dong): sum_pop_in_hours}
    records: dict[tuple[str, str], float] = defaultdict(float)
    processed = 0
    missing_dates: list[str] = []

    for fp in files:
        date = extract_date_from_filename(fp.name)
        if date is None:
            continue
        if date in KNOWN_MISSING_DATES:
            missing_dates.append(date)
            continue

        daily = stream_single_file(fp, target_hours, encoding=encoding)
        for dong, pop in daily.items():
            records[(date, dong)] += pop
        processed += 1

    rows = [
        {"date": date, "adm_code": dong, "pop_sum_hours": pop}
        for (date, dong), pop in records.items()
    ]
    df = pd.DataFrame(rows)
    print(f"  처리: {processed} 일자 / 결측: {len(missing_dates)} 일자")
    return df


# -----------------------------------------------------------------------------
# 일별 → 월별 평균
# -----------------------------------------------------------------------------

def aggregate_to_monthly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """(date, adm_code) → (yyyymm, adm_code) 월별 평균."""
    out = df_daily.copy()
    out["yyyymm"] = out["date"].astype(str).str[:6].astype(int)
    monthly = (
        out.groupby(["yyyymm", "adm_code"], as_index=False)["pop_sum_hours"]
        .mean()
        .rename(columns={"pop_sum_hours": "pop_avg_per_day"})
    )
    return monthly.sort_values(["yyyymm", "adm_code"]).reset_index(drop=True)


def compute_monthly_share(monthly: pd.DataFrame) -> pd.DataFrame:
    """월별 서울 전체 대비 행정동 비율 (응용집계 비율, 반출 가능)."""
    out = monthly.copy()
    monthly_total = out.groupby("yyyymm")["pop_avg_per_day"].transform("sum")
    out["dong_share_in_month"] = out["pop_avg_per_day"] / monthly_total
    return out[["yyyymm", "adm_code", "dong_share_in_month"]]


# -----------------------------------------------------------------------------
# 야간 특화 지수 (옛 정통, K-means 입력용 — 메인 파이프라인 미사용)
# -----------------------------------------------------------------------------
# 본 함수는 옛 대화 8a79670f·898f898b에서 Ojiro·Claude가 함께 만든 산식.
# 2026-05-11 결정으로 K-means/야간특화지수는 메인 파이프라인에서 제외됐지만,
# 코드 자체는 보고서 §11 한계 인정 + 부록 코드로 남기기 위해 보존.

DAY_HOURS = {"06", "07", "08", "09", "10", "11", "12",
             "13", "14", "15", "16", "17", "18", "19", "20"}
NIGHT_HOURS = {"21", "22", "23", "00", "01", "02", "03", "04", "05"}

# K-means 사전 제외 이상치 3동 (옛 대화 정통)
KMEANS_OUTLIER_DONGS = frozenset({"11710631", "11740520", "11710647"})


def compute_night_special_index(
    df_daily_night: pd.DataFrame,
    df_daily_day: pd.DataFrame,
) -> pd.DataFrame:
    """야간 특화 지수 산출 (옛 대화 8a79670f·898f898b 정통).

    산식:
    1) night_pop = Σ tot_lvpop_co WHERE tmzon IN {21,22,23,0~5}
    2) day_pop   = Σ tot_lvpop_co WHERE tmzon IN {6~20}
    3) night_day_ratio = night_pop / day_pop
    4) night_special_index = night_day_ratio / 서울 평균(night_day_ratio)

    Args:
        df_daily_night: 야간 시간대만 streaming 한 일별 (date, adm_code, pop_sum_hours)
        df_daily_day:   주간 시간대만 streaming 한 일별 (date, adm_code, pop_sum_hours)

    Returns:
        DataFrame[dong_code, night_pop, day_pop, night_day_ratio, night_special_index]

    Notes:
        - 결과 평균이 정확히 1.000 이 되도록 서울 평균으로 재정규화.
        - 1보다 크면 야간 특화, 1보다 작으면 주간 중심.
        - 응용집계 (비율 ÷ 평균비율) 형태로 캠퍼스 반출 가능.
        - 산출 csv: dong_night_index_final.csv (424행, 5컬럼).
    """
    night_dong = (
        df_daily_night.groupby("adm_code", as_index=False)["pop_sum_hours"]
        .sum().rename(columns={"pop_sum_hours": "night_pop"})
    )
    day_dong = (
        df_daily_day.groupby("adm_code", as_index=False)["pop_sum_hours"]
        .sum().rename(columns={"pop_sum_hours": "day_pop"})
    )
    merged = night_dong.merge(day_dong, on="adm_code", how="inner")
    merged = merged[merged["day_pop"] > 0].copy()
    merged["night_day_ratio"] = merged["night_pop"] / merged["day_pop"]

    # 서울 평균 대비 상대화 (응용집계 — 반출 가능 형태)
    seoul_mean_ratio = merged["night_day_ratio"].mean()
    merged["night_special_index"] = merged["night_day_ratio"] / seoul_mean_ratio
    print(f"  서울 평균 night/day ratio: {seoul_mean_ratio:.4f}")
    print(f"  night_special_index 평균: {merged['night_special_index'].mean():.4f}")
    print(f"  min {merged['night_special_index'].min():.3f} / "
          f"max {merged['night_special_index'].max():.3f}")
    return merged.rename(columns={"adm_code": "dong_code"})


# -----------------------------------------------------------------------------
# 진입점
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B040 생활인구 streaming → 월별 행정동 야간 인구")

    parser.add_argument(
        "--track",
        type=str,
        choices=["evening", "late_night"],
        required=True,
        help="저녁(17~20) 또는 심야(21~05) 트랙",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(r"E:/"),
        help="B040 원본 루트 (이 아래에 YYYYMM/YYYYMM/TN_PF_*.txt 구조)",
    )
    parser.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--encoding", type=str, default="utf-8")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="산출 디렉터리. 미지정 시 cwd/b040_processed/{track}/",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (Path.cwd() / "b040_processed" / args.track)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_hours = TRACK_HOURS[args.track]
    print(f"[설정] 트랙={args.track}, 시간대={sorted(target_hours)}, 연도={args.years}")

    print("\n[1] 파일 목록 수집")
    files = collect_files(args.base_dir, tuple(args.years))
    print(f"  발견: {len(files)} 파일")

    print("\n[2] 일별 streaming + 시간대 필터 + 행정동 누적")
    start = datetime.now()
    df_daily = stream_all(files, target_hours, encoding=args.encoding)
    elapsed = (datetime.now() - start).total_seconds()
    print(f"  완료: {len(df_daily)} 행, {elapsed:.1f}초")

    print("\n[3] 월별 평균 집계 (캠퍼스 internal — 반출 불가)")
    monthly = aggregate_to_monthly(df_daily)
    internal_path = out_dir / f"b040_monthly_{args.track}_pop.csv"
    monthly.to_csv(internal_path, index=False, encoding="utf-8-sig")
    print(f"  저장 (internal): {internal_path} ({len(monthly)} 행)")

    print("\n[4] 월별 행정동 비율 (응용집계 — 반출 가능)")
    share = compute_monthly_share(monthly)
    share_path = out_dir / f"b040_monthly_{args.track}_share.csv"
    share.to_csv(share_path, index=False, encoding="utf-8-sig")
    print(f"  저장 (반출용): {share_path} ({len(share)} 행)")


if __name__ == "__main__":
    # 실행 예시 (캠퍼스 환경에서 트랙별 별도 실행):
    # import sys
    # sys.argv = [
    #     "b040.py",
    #     "--track", "late_night",
    #     "--base-dir", r"E:/",
    #     "--years", "2023", "2024", "2025",
    #     "--out-dir", r"C:/야간경제/b040_processed/late_night",
    # ]
    # main()
    pass
