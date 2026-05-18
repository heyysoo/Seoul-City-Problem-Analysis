"""점포 개점률 지수 (opening_norm) — 잠재력 지수 두 번째 변수 (가중치 30%)

처리 목적
- opening_rate.py 산출물(opening_rate.csv)을 입력으로 받아
  imputed 처리 + Min-Max 정규화 → 잠재력 지수 입력 형태로 변환.
- 점포수 5개 미만 동 = 서울 평균 대체 + is_imputed=1 플래그.
- 일원2동(행자부 통합 폐지) = 마스터 단계 자연 제외 또는 imputed 대체.

학술 근거 (가중치 30%)
- World Bank Entrepreneurship Database: new business entry density rate 표준
- Glaeser, Kerr, & Ponzetto (2010) JUE: 도시 entrepreneurship 클러스터 분석
- 12분기 평균으로 단기 변동 평탄화 + 야간 10개 업종 필터로 정합성

5가지 핵심 결정
1. 점포수 5개 미만 imputed 처리 (옵션 D, 서울 평균 3.58% 대체)
2. is_imputed 플래그로 추적성 확보
3. 통계청 → 행자부 코드 변환 (별도 매핑 csv 활용)
4. 일원2동 통합 폐지 처리 (마스터 단계 명시)
5. Min-Max → [0, 1] 정규화 (잠재력 지수 합산용)

산식
- imputed_value = opening_rate.csv 의 서울 평균 (avg_opening_rate.mean())
- avg_opening_rate_used = avg_opening_rate if (avg_store_count >= 5) else imputed_value
- is_imputed = (avg_store_count < 5)
- opening_norm = (avg_opening_rate_used - min) / (max - min)
  → Min-Max만, 클리핑 없음 (분포 안정)

코드 매핑 (통계청 → 행자부)
- 자동 매핑 (별도 매핑 csv 활용) — 416/426 동
- EXCEPTION 매핑 10동 (강북 6쌍 + 강남·강동 4동) — 매핑 csv에 포함

입력
- opening_rate.csv: opening_rate.py 산출물 (행정동, 평균 개업률, 점포수, 분기수)

산출 (out_dir 하위)
- opening_rate_final.csv (~427행): 행자부_코드, opening_rate_used, opening_norm,
  is_imputed, avg_store_count, matching_method

주의
- 본 모듈은 opening_rate.py 산출물을 입력으로 받음 (분기 집계는 별도 모듈에서 처리).
- 행자부 코드 변환 시 별도 매핑 csv 또는 EXCEPTION 매핑이 필요 (자세히는 코드 내 주석 참조).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# -----------------------------------------------------------------------------
# 상수
# -----------------------------------------------------------------------------

# 점포수 5개 미만 imputed 처리 (서울 평균 대체)
STORE_COUNT_THRESHOLD = 5

# 일원2동 — 행자부 통합 폐지 (마스터 단계에서 자연 제외 또는 imputed)
ILWON_2_DONG_STAT = "11680660"  # 통계청 코드 (참고)


# -----------------------------------------------------------------------------
# Step 1 — 입력 로딩
# -----------------------------------------------------------------------------

def load_opening_summary(
    path: Path,
    encoding: str = "utf-8-sig",
) -> pd.DataFrame:
    """opening_rate.py 산출물 로드.

    필요 컬럼: 행정동_코드, avg_opening_rate, avg_store_count
    """
    df = pd.read_csv(path, encoding=encoding, dtype={"행정동_코드": str})
    required = ["행정동_코드", "avg_opening_rate", "avg_store_count"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"필수 컬럼 누락: {missing}. 현재: {df.columns.tolist()}")

    df["행정동_코드"] = df["행정동_코드"].astype(str).str.zfill(8)
    print(f"  로드: {len(df)}동")
    print(f"  점포수 5개 미만 동: {(df['avg_store_count'] < STORE_COUNT_THRESHOLD).sum()}개")
    return df


# -----------------------------------------------------------------------------
# Step 2 — Imputed 처리
# -----------------------------------------------------------------------------

def apply_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """점포수 5개 미만 동 → 서울 평균 대체 + is_imputed 플래그.

    옵션 D (노션 §3.2 정통):
    - imputed_value = 정상 동들의 avg_opening_rate 평균 (5개 이상 동 기준)
    - target 동들에만 imputed_value 적용 + is_imputed=1
    """
    out = df.copy()
    valid_mask = out["avg_store_count"] >= STORE_COUNT_THRESHOLD
    target_mask = ~valid_mask

    imputed_value = out.loc[valid_mask, "avg_opening_rate"].mean()
    print(f"  서울 평균 (점포수 ≥{STORE_COUNT_THRESHOLD} 동 기준): {imputed_value:.4f}")

    out["opening_rate_used"] = out["avg_opening_rate"]
    out.loc[target_mask, "opening_rate_used"] = imputed_value
    out["is_imputed"] = target_mask.astype(int)

    n_imputed = target_mask.sum()
    print(f"  imputed 적용: {n_imputed}동")
    return out


# -----------------------------------------------------------------------------
# Step 3 — 통계청 → 행자부 코드 변환
# -----------------------------------------------------------------------------

def apply_code_mapping(
    df: pd.DataFrame,
    mapping_csv: Path | None,
) -> pd.DataFrame:
    """통계청 → 행자부 코드 변환.

    별도 매핑 csv 활용. 없으면 통계청 코드 그대로 사용 (마스터 단계에서 처리 권장).
    매핑 csv는 transit_index.py의 EXCEPTION_MAPPING과 동일하게 강북 6쌍 + 강남·강동 4동
    포함해야 함.
    """
    out = df.copy()
    out["matching_method"] = "코드_변환_미적용"

    if mapping_csv is None or not mapping_csv.exists():
        out["행자부_코드"] = out["행정동_코드"]
        print("  ⚠️ 매핑 csv 없음 — 통계청 코드 그대로 사용 (마스터 단계에서 처리)")
        return out

    mapping_df = pd.read_csv(mapping_csv, dtype=str)
    # 컬럼명 자동 추출
    src_candidates = ["통계청_코드", "stat_code", "행정동_코드"]
    tgt_candidates = ["행자부_코드", "haengjabu_code", "adm_code"]
    src_col = next((c for c in src_candidates if c in mapping_df.columns), None)
    tgt_col = next((c for c in tgt_candidates if c in mapping_df.columns), None)

    if src_col is None or tgt_col is None:
        print(f"  ⚠️ 매핑 csv 컬럼 식별 실패 ({mapping_df.columns.tolist()})")
        out["행자부_코드"] = out["행정동_코드"]
        return out

    mapping_df[src_col] = mapping_df[src_col].astype(str).str.zfill(8)
    mapping_df[tgt_col] = mapping_df[tgt_col].astype(str).str.zfill(8)
    mapping = dict(zip(mapping_df[src_col], mapping_df[tgt_col]))

    out["행자부_코드"] = out["행정동_코드"].apply(
        lambda c: mapping.get(c, c)
    )
    out["matching_method"] = out.apply(
        lambda r: "코드매핑_성공" if r["행정동_코드"] in mapping else "매핑_없음",
        axis=1,
    )
    n_mapped = (out["matching_method"] == "코드매핑_성공").sum()
    print(f"  매핑 성공: {n_mapped}/{len(out)}동")
    return out


# -----------------------------------------------------------------------------
# Step 4 — Min-Max 정규화
# -----------------------------------------------------------------------------

def normalize_opening(df: pd.DataFrame) -> pd.DataFrame:
    """Min-Max 정규화 → [0, 1] (클리핑 없음, 분포 안정).

    opening_rate_used 컬럼 기준으로 정규화.
    """
    out = df.copy()
    s = pd.to_numeric(out["opening_rate_used"], errors="coerce")
    rng = s.max() - s.min()
    if rng == 0:
        out["opening_norm"] = 0.5
    else:
        out["opening_norm"] = (s - s.min()) / rng
    print(f"  opening_norm 평균 {out['opening_norm'].mean():.3f} / "
          f"중앙값 {out['opening_norm'].median():.3f}")
    return out


# -----------------------------------------------------------------------------
# 진입점
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="점포 개점률 (opening_norm) 산출 — opening_rate.py 산출물 후처리"
    )

    parser.add_argument(
        "--input-csv", type=Path,
        default=Path("data/processed/opening_rate/opening_rate.csv"),
        help="opening_rate.py 산출물 (행정동별 12분기 평균 개업률)",
    )
    parser.add_argument(
        "--mapping-csv", type=Path,
        default=None,
        help="통계청↔행자부 매핑 csv (선택, 없으면 통계청 코드 그대로 유지)",
    )
    parser.add_argument("--encoding", type=str, default="utf-8-sig")

    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("data/processed/opening"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] opening_rate.py 산출물 로드")
    df = load_opening_summary(args.input_csv, encoding=args.encoding)

    print("\n[2] 점포수 5개 미만 imputed 처리 (서울 평균 대체)")
    df = apply_imputation(df)

    print("\n[3] 통계청 → 행자부 코드 변환")
    df = apply_code_mapping(df, mapping_csv=args.mapping_csv)

    print("\n[4] Min-Max 정규화")
    df = normalize_opening(df)

    # 출력 정리
    output_cols = [
        "행정동_코드", "행자부_코드", "행정동_코드_명" if "행정동_코드_명" in df.columns else None,
        "avg_opening_rate", "avg_store_count",
        "opening_rate_used", "opening_norm", "is_imputed", "matching_method",
    ]
    output_cols = [c for c in output_cols if c is not None and c in df.columns]
    result = df[output_cols].copy()

    out_path = args.out_dir / "opening_rate_final.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_path} ({len(result)}행 × {len(result.columns)}열)")


if __name__ == "__main__":
    # 실행 예시:
    # import sys
    # sys.argv = [
    #     "opening_index.py",
    #     "--input-csv", "data/processed/opening_rate/opening_rate.csv",
    #     "--mapping-csv", "data/raw/codes/통계청_행자부_매핑.csv",
    #     "--out-dir", "data/processed/opening",
    # ]
    # main()
    pass
