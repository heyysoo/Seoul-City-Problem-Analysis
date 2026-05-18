"""교통접근성 지수 (transit_norm) — 잠재력 지수 세 번째 변수 (가중치 20%)

처리 목적
- B013 캠퍼스 산출물(`B013_행정동별_교통이원트랙유입_월별상세.csv`)을 입력으로 받아
  저녁/심야 트랙별 transit_norm 산출.
- 월별 YoY → 행정동 평균 → 서울 중앙값 차이 → 1·99 clipping → Min-Max.
- 통계청 코드 → 행자부 코드 변환 + EXCEPTION 매핑 10동 (426/426 100%).

학술 근거 (가중치 20%)
- Cervero & Duncan (2002): LA 라이트 레일 인접 상업 매출 +24%
- Cervero & Kang (2011, Transport Policy): 서울 BRT 토지 가치 영향
- Pizzol et al. (2025): 대중교통 접근성 > 사적 교통 접근성 (소매 밀도 상관)
- TOD 학술 표준 + 한국 환경 직접 정합

5가지 핵심 결정
1. 버스+지하철 절대량 합산 후 YoY (단위 동일 → 합산 정통)
2. 전년 동월 비교 (계절성 제거)
3. 2024 YoY + 2025 YoY 평균 (단일 연도 변동 완화)
4. 서울 중앙값 차이 (Applebaum 1966, gap_norm·growth와 정합)
5. 1·99 percentile 클리핑 + Min-Max (극단값 영향 제한)

산식
- 트랙별 월별 유입 = 버스 + 지하철 (4시간 또는 9시간 블록 합)
- YoY_{y,m} = (유입_{y,m} - 유입_{y-1,m}) / 유입_{y-1,m}
- 동별 평균 = mean(YoY_2024, YoY_2025) over m=1..12
- yoy_diff = 동별 평균 - 서울 중앙값
- yoy_diff_clip = clip(yoy_diff, q01, q99)
- transit_norm = (yoy_diff_clip - min) / (max - min) ∈ [0, 1]

EXCEPTION 매핑 10동 (통계청 → 행자부)
- 강북 6쌍 (번1·2·3·수유1·2·3동)
- 강남·강동 4동 (개포1·3동·상일1·2동)

입력
- B013_행정동별_교통이원트랙유입_월별상세.csv (15,336행)
  컬럼: 사용년월, 행정동ID(통계청), 행정동명, 자치구명,
        버스_저녁유입, 버스_심야유입, 지하철_저녁유입, 지하철_심야유입,
        교통_저녁유입, 교통_심야유입, 교통_종합유입
- opening_rate.csv (행정동 매핑 테이블 활용, 통계청↔행자부 코드 변환용)

산출 (out_dir 하위)
- transit_yoy_monthly.csv (월별 YoY 상세, 검증용)
- transit_yoy_dong_avg.csv (행정동별 YoY 평균, 검증용)
- transit_norm_final.csv (426행 * 11컬럼): 통계청_코드, 행자부_코드, 행자부_행정동명,
  gu_name, 저녁_yoy_mean, 저녁_yoy_diff, transit_norm_evening,
  심야_yoy_mean, 심야_yoy_diff, transit_norm_late, matching_method
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 상수
# -----------------------------------------------------------------------------

CLIP_LOWER_PCT = 0.01
CLIP_UPPER_PCT = 0.99

# EXCEPTION 매핑: 통계청 → 행자부 (opening 매핑에 없거나 일반 규칙과 다른 동)
# 노션 §3.3 정통 (10개 동, 426/426 100% 매칭)
EXCEPTION_MAPPING = {
    # 강북구 6쌍 (B040·B079 패턴과 동일)
    "11305590": "11090600",  # 번1동
    "11305600": "11090610",  # 번2동
    "11305606": "11090620",  # 번3동
    "11305610": "11090630",  # 수유1동
    "11305620": "11090640",  # 수유2동
    "11305630": "11090650",  # 수유3동
    # 강남·강동 4동 (opening 매핑 미포함)
    "11680660": "11230680",  # 개포1동
    "11680675": "11230511",  # 개포3동
    "11740525": "11250760",  # 상일1동
    "11740526": "11250770",  # 상일2동
}


# -----------------------------------------------------------------------------
# Step 1 — 월별 데이터 로드
# -----------------------------------------------------------------------------

def load_monthly(path: Path, encoding: str = "utf-8-sig") -> pd.DataFrame:
    """B013 캠퍼스 산출물 로드. 통계청 코드 8자리 통일."""
    df = pd.read_csv(path, encoding=encoding, dtype={"행정동ID": str})
    df["행정동ID"] = df["행정동ID"].astype(str).str.strip().str.zfill(8)

    # 연월 분리
    df["연도"] = df["사용년월"] // 100
    df["월"] = df["사용년월"] % 100

    print(f"  로드: {len(df):,}행, {df['행정동ID'].nunique()}개 통계청 코드, "
          f"{df['연도'].unique().tolist()}년")
    return df


# -----------------------------------------------------------------------------
# Step 2 — 전년 동월 YoY 계산 (트랙별)
# -----------------------------------------------------------------------------

def calc_yoy(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """행정동×월별 전년 동월 YoY 계산.

    pivot 후 연도 컬럼 간 차이 계산 → 2024 YoY (vs 2023), 2025 YoY (vs 2024).
    """
    pivot = df.pivot_table(
        index=["행정동ID", "행정동명", "자치구명", "월"],
        columns="연도",
        values=value_col,
        aggfunc="first",
    ).reset_index()

    # 분모 0 회피 (replace 후 division)
    if 2023 in pivot.columns:
        denom_2023 = pivot[2023].replace(0, np.nan)
        pivot["YoY_2024"] = (pivot[2024] - pivot[2023]) / denom_2023
    if 2024 in pivot.columns:
        denom_2024 = pivot[2024].replace(0, np.nan)
        pivot["YoY_2025"] = (pivot[2025] - pivot[2024]) / denom_2024

    pivot = pivot.replace([np.inf, -np.inf], np.nan)
    return pivot


def aggregate_yoy(pivot: pd.DataFrame, label: str) -> pd.DataFrame:
    """행정동별 2년 YoY 평균 (모든 월 평균 × 2년 평균)."""
    agg = (
        pivot.groupby(["행정동ID", "행정동명", "자치구명"], as_index=False)
        .agg(
            YoY_2024_평균=("YoY_2024", "mean"),
            YoY_2025_평균=("YoY_2025", "mean"),
        )
    )
    agg[f"{label}_yoy_mean"] = (agg["YoY_2024_평균"] + agg["YoY_2025_평균"]) / 2
    return agg


# -----------------------------------------------------------------------------
# Step 3 — 서울 중앙값 차이 (Median 정통)
# -----------------------------------------------------------------------------

def apply_median_diff(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """서울 행정동 전체 YoY 평균의 중앙값을 기준으로 차이 산출."""
    out = df.copy()
    mean_col = f"{label}_yoy_mean"
    diff_col = f"{label}_yoy_diff"

    seoul_median = out[mean_col].median()
    out[diff_col] = out[mean_col] - seoul_median

    print(f"  {label} 서울 중앙값: {seoul_median * 100:.2f}%")
    print(f"  {label} 평균/중앙값 비율: {out[mean_col].mean() / seoul_median:.2f}x")
    return out


# -----------------------------------------------------------------------------
# Step 4 — 클리핑 + Min-Max 정규화
# -----------------------------------------------------------------------------

def clip_and_minmax(
    series: pd.Series,
    lower_pct: float = CLIP_LOWER_PCT,
    upper_pct: float = CLIP_UPPER_PCT,
) -> pd.Series:
    """1·99 percentile 클리핑 후 Min-Max 정규화 → [0, 1]."""
    s = pd.to_numeric(series, errors="coerce")
    lo = s.quantile(lower_pct)
    hi = s.quantile(upper_pct)
    clipped = s.clip(lower=lo, upper=hi)
    rng = clipped.max() - clipped.min()
    if rng == 0:
        return pd.Series(0.5, index=s.index)
    return (clipped - clipped.min()) / rng


# -----------------------------------------------------------------------------
# Step 5 — 통계청 → 행자부 코드 변환
# -----------------------------------------------------------------------------

def load_code_mapping(opening_csv: Path | None) -> dict[str, str]:
    """opening_rate.py 산출물 또는 별도 매핑 csv에서 통계청→행자부 매핑 추출.

    opening은 통계청 코드 기반이고, 별도 정합 정리 단계에서 행자부 코드를 부여한
    매핑 테이블을 활용. EXCEPTION 매핑에서 다루지 않는 416동 자동 매핑.

    매핑 csv 없을 시 빈 dict 반환 (EXCEPTION만 적용).
    """
    if opening_csv is None or not opening_csv.exists():
        print(" opening 매핑 csv 없음 — EXCEPTION 매핑만 적용")
        return {}

    df = pd.read_csv(opening_csv, dtype=str)
    # 컬럼명 후보 (opening_rate 산출 컬럼 명세 다양성 대응)
    src_candidates = ["통계청_코드", "stat_code", "행정동_코드"]
    tgt_candidates = ["행자부_코드", "haengjabu_code", "adm_code"]

    src_col = next((c for c in src_candidates if c in df.columns), None)
    tgt_col = next((c for c in tgt_candidates if c in df.columns), None)

    if src_col is None or tgt_col is None:
        print(f" 코드 매핑 컬럼 없음 (가능 컬럼: {df.columns.tolist()})")
        return {}

    df[src_col] = df[src_col].astype(str).str.zfill(8)
    df[tgt_col] = df[tgt_col].astype(str).str.zfill(8)
    return dict(zip(df[src_col], df[tgt_col]))


def map_to_haengjabu(stat_code: str, mapping: dict[str, str]) -> tuple[str, str]:
    """통계청 코드 → 행자부 코드 (EXCEPTION 우선, 그다음 자동 매핑).

    Returns:
        (haengjabu_code, matching_method)
    """
    code = str(stat_code).zfill(8)
    if code in EXCEPTION_MAPPING:
        return EXCEPTION_MAPPING[code], "EXCEPTION_매핑"
    if code in mapping:
        return mapping[code], "코드매핑_성공"
    return code, "매칭_실패"  # fallback: 통계청 그대로 (수동 검토 필요)


# -----------------------------------------------------------------------------
# 통합 파이프라인
# -----------------------------------------------------------------------------

def build_track_norm(
    df_monthly: pd.DataFrame,
    value_col: str,
    label: str,
) -> pd.DataFrame:
    """단일 트랙 (저녁 또는 심야) transit_norm 산출."""
    pivot = calc_yoy(df_monthly, value_col)
    agg = aggregate_yoy(pivot, label)
    agg = apply_median_diff(agg, label)

    diff_col = f"{label}_yoy_diff"
    norm_col = f"transit_norm_{label}"
    agg[norm_col] = clip_and_minmax(agg[diff_col])

    n_clipped = ((agg[diff_col] <= agg[diff_col].quantile(CLIP_LOWER_PCT))
                 | (agg[diff_col] >= agg[diff_col].quantile(CLIP_UPPER_PCT))).sum()
    print(f"  {label} 클리핑 적용 동: {n_clipped}개 (양 극단 1%)")
    return agg


def merge_tracks(
    evening_df: pd.DataFrame,
    late_df: pd.DataFrame,
) -> pd.DataFrame:
    """저녁·심야 트랙 결과 결합 (행정동 키 기준)."""
    merged = evening_df.merge(
        late_df[[
            "행정동ID", "심야_yoy_mean", "심야_yoy_diff", "transit_norm_late",
        ]],
        on="행정동ID",
        how="outer",
    )
    return merged


def apply_code_conversion(
    df: pd.DataFrame,
    code_mapping: dict[str, str],
) -> pd.DataFrame:
    """통계청 → 행자부 코드 변환 + matching_method 컬럼 추가."""
    out = df.copy()
    out["행정동ID"] = out["행정동ID"].astype(str).str.zfill(8)
    out.rename(columns={"행정동ID": "통계청_코드"}, inplace=True)

    converted = out["통계청_코드"].apply(lambda c: map_to_haengjabu(c, code_mapping))
    out["행자부_코드"] = converted.apply(lambda x: x[0])
    out["matching_method"] = converted.apply(lambda x: x[1])

    method_counts = out["matching_method"].value_counts()
    print("  매핑 결과:")
    for method, n in method_counts.items():
        print(f"    {method:25s}: {n}동")
    return out


# -----------------------------------------------------------------------------
# 진입점
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="교통접근성 (transit_norm) 산출")

    parser.add_argument(
        "--monthly-csv", type=Path,
        default=Path("data/raw/b013/B013_행정동별_교통이원트랙유입_월별상세.csv"),
        help="B013 캠퍼스 산출물 (월별 트랙별 유입)",
    )
    parser.add_argument(
        "--opening-csv", type=Path,
        default=Path("data/processed/opening_rate/opening_rate.csv"),
        help="opening_rate.py 산출물 (통계청↔행자부 매핑 테이블 활용)",
    )
    parser.add_argument("--encoding", type=str, default="utf-8-sig")

    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("data/processed/transit"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] B013 월별 데이터 로드")
    monthly = load_monthly(args.monthly_csv, encoding=args.encoding)

    print("\n[2] 저녁 트랙 YoY 산출")
    evening = build_track_norm(monthly, "교통_저녁유입", "저녁")
    print(f"  저녁: {len(evening)}동")

    print("\n[3] 심야 트랙 YoY 산출")
    late = build_track_norm(monthly, "교통_심야유입", "심야")
    print(f"  심야: {len(late)}동")

    print("\n[4] 트랙 결합")
    merged = merge_tracks(evening, late)
    print(f"  결합: {len(merged)}동")

    print("\n[5] 통계청 → 행자부 코드 변환 (EXCEPTION 매핑 10동 포함)")
    code_mapping = load_code_mapping(args.opening_csv)
    final = apply_code_conversion(merged, code_mapping)

    # 컬럼 순서 정리
    final = final.rename(columns={"행정동명": "행자부_행정동명", "자치구명": "gu_name"})
    output_cols = [
        "통계청_코드", "행자부_코드", "행자부_행정동명", "gu_name",
        "저녁_yoy_mean", "저녁_yoy_diff", "transit_norm_evening",
        "심야_yoy_mean", "심야_yoy_diff", "transit_norm_late",
        "matching_method",
    ]
    final = final[[c for c in output_cols if c in final.columns]]

    out_path = args.out_dir / "transit_norm_final.csv"
    final.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_path} ({len(final)}행 × {len(final.columns)}열)")

    # 검증 요약
    print("\n분포 통계 (csv 직접 검증):")
    for col, label in [("transit_norm_evening", "저녁"), ("transit_norm_late", "심야")]:
        if col in final.columns:
            print(f"  {label}: 평균 {final[col].mean():.3f} / "
                  f"중앙값 {final[col].median():.3f}")


if __name__ == "__main__":
    # 실행 예시:
    # import sys
    # sys.argv = [
    #     "transit_index.py",
    #     "--monthly-csv", "data/raw/b013/B013_행정동별_교통이원트랙유입_월별상세.csv",
    #     "--opening-csv", "data/processed/opening_rate/opening_rate.csv",
    #     "--out-dir", "data/processed/transit",
    # ]
    # main()
    pass
