"""매출 미실현도 (gap_norm) — B040 + B079 결합

처리 목적
- 심야 트랙의 보조 변수 매출 미실현도(gap_norm)를 산출.
- 도메인 의미: "인구 대비 매출이 부족한가" (저평가된 야간경제 후보 식별).
- 심야 매트릭스 점 크기 + 4_priority 4요소 중 하나로 사용.

산식 
[캠퍼스 처리 4단계]
1. B040 야간 평균 생활인구 (행정동별)
2. B079 야간 카드매출 합계 (행정동별)
3. 결합 + 1인당 야간 매출 + 서울 중앙값 대비 비율
   - per_capita = sum_card / pop_avg
   - per_capita_rel = per_capita / Seoul_median(per_capita)  ← Median 정통
   - gap_index = 1 - per_capita_rel
4. 캠퍼스 반출 (응용집계 비율만)

[운영 보강 2단계, 마스터 JOIN 사전 작업]
5. gap_norm = PercentileRank(gap_index) ∈ [0, 1]
6. 통계청 → 행자부 코드 변환 (opening 매핑 + EXCEPTION 7동)

산식 결정 사유
- 가중평균 폐기: 명동(per_capita 576배) 등 극단값으로 분포 왜곡
- Median 정통 (Applebaum 1966): per_capita 평균 5.36 vs 중앙값 1.0 (5.36배 차이)
- PercentileRank: gap_index의 음수 값(매출 우세 동) 자연 처리

이상치 제외 (5개 동, 산식 안정성 위해)
- 11710631, 11710647, 11740520: 옛 K-means 이상치 (생활인구 극단값)
  → 메인 파이프라인에서 K-means 제거됐으나 gap_index 산식 안정성 보존 위해 유지
- 11740525, 11740526: 상일동 분리 (B079에 분할 코드로 존재, B040 8자리에 미존재)

코드 매핑 — B040(8자리) ↔ B079(10자리) ↔ 행자부(8자리)
- B079 10자리 → 앞 8자리만 추출이 기본 매핑 (통계청 8자리)
- 강북 6쌍 (번동·수유동): 단순 절단으로 불일치 → 수동 매핑
- 마지막에 통계청 → 행자부 변환 (opening_rate.py 매핑 csv + EXCEPTION 7동)

EXCEPTION_MAPPING (통계청 → 행자부, 노션 §3.3 정통)
- 강북 6쌍: 11305590/11305600/11305606/11305610/11305620/11305630 → 11090600/.../11090650
- 강남 개포1동: 11680660 → 11230680 (opening 매핑 미포함)

마스터 JOIN 결측 7동 (노션 §6.6 — Grey out 정책)
- 항동(11170740), 개포3동(11230511), 가락1동(11240660), 위례동(11240820),
  둔촌1동(11250700), 상일1동(11250760), 상일2동(11250770)
- 마스터 단계에서 중앙값(약 0.498) 대체 + is_gap_imputed 플래그 + 시각화 Grey out

입력
- b040: 캠퍼스 산출 b040_monthly_late_night_pop.csv (행정동×월별 평균 인구)
- b079: 캠퍼스 산출 행정동_카드매출.csv (행정동×월별 카드 매출)
- mapping (선택): 통계청↔행자부 매핑 csv (opening_rate_final.csv 활용 가능)

산출물 (out_dir 하위)
- b040_b079_gap.csv (419행, 캠퍼스 반출용): amd_code(통계청), per_capita_rel, gap_index, gap_norm
- b040_b079_gap_final.csv (419행, 마스터 JOIN용): + 행자부_코드, 행자부_행정동명,
  gu_name, matching_method

캠퍼스 반출 정책
- per_capita, sum_card 등 절대값은 반출 불가
- per_capita_rel, gap_index, gap_norm 등 비율은 응용집계로 반출 가능
- 본 모듈의 산출 csv는 반출 가능 형태로만 컬럼 유지
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 상수
# -----------------------------------------------------------------------------

# 이상치 제외 (5개 동, 산식 안정성)
OUTLIER_CODES = frozenset({
    "11710631", "11710647", "11740520",  # 옛 K-means 이상치 (생활인구 극단값)
    "11740525", "11740526",              # 상일동 분리 (B079 분할 코드)
})

# 강북구 번동·수유동 6쌍 수동 매핑 (B079 10자리 → B040 8자리)
# 노션 정통: 옛 대화 692f8ee6 (Ojiro + Claude) 검증 완료, 419개 동 매칭 0 failures
# 앞 8자리 절단 규칙과 다르게 별도 부여된 코드. growth_index.py와 동일 매핑 유지.
GANGBUK_CODE_MAPPING = {
    "1130559500": "11305590",   # 번1동
    "1130560300": "11305600",   # 번2동
    "1130560800": "11305606",   # 번3동
    "1130561500": "11305610",   # 수유1동
    "1130562500": "11305620",   # 수유2동
    "1130563500": "11305630",   # 수유3동
}

# EXCEPTION_MAPPING — 통계청 8자리 → 행자부 8자리 (노션 §3.3 정통, 7개 동)
# opening 매핑 csv에서 다루지 않거나 일반 규칙과 다른 동들.
EXCEPTION_MAPPING = {
    # 강북구 6쌍 (B079 출력의 통계청 코드는 일반 자치구와 다른 형식)
    "11305590": ("11090600", "번1동", "강북구"),
    "11305600": ("11090610", "번2동", "강북구"),
    "11305606": ("11090620", "번3동", "강북구"),
    "11305610": ("11090630", "수유1동", "강북구"),
    "11305620": ("11090640", "수유2동", "강북구"),
    "11305630": ("11090650", "수유3동", "강북구"),
    # 강남구 개포1동 (opening 매핑에 없음, 도메인 검증 완료)
    "11680660": ("11230680", "개포1동", "강남구"),
}


# -----------------------------------------------------------------------------
# 코드 매핑
# -----------------------------------------------------------------------------

def map_b079_to_b040(amd_code_10: str) -> str:
    """B079 10자리 코드 → B040 8자리 코드.

    기본: 앞 8자리 절단. 6쌍 예외(강북구 번동·수유동)는 수동 매핑 우선 적용.
    """
    code = str(amd_code_10).strip()
    if code in GANGBUK_CODE_MAPPING:
        return GANGBUK_CODE_MAPPING[code]
    return code[:8] if len(code) >= 8 else code


# -----------------------------------------------------------------------------
# 핵심 산식
# -----------------------------------------------------------------------------

def aggregate_to_dong(
    monthly_b040: pd.DataFrame,
    monthly_b079: pd.DataFrame,
) -> pd.DataFrame:
    """행정동 단위 인구·매출 합계 (36개월 통합)."""
    pop = monthly_b040.copy()
    pop["adm_code"] = pop["adm_code"].astype(str).str.strip()
    pop_dong = pop.groupby("adm_code", as_index=False)["pop_avg_per_day"].sum()
    pop_dong = pop_dong.rename(columns={"pop_avg_per_day": "pop_total"})

    sales = monthly_b079.copy()
    sales["amd_code"] = sales["amd_code"].astype(str).str.strip()
    sales["adm_code"] = sales["amd_code"].map(map_b079_to_b040)
    sales_dong = sales.groupby("adm_code", as_index=False)["sum_card"].sum()
    sales_dong = sales_dong.rename(columns={"sum_card": "sales_total"})

    merged = pop_dong.merge(sales_dong, on="adm_code", how="inner")
    return merged


def compute_gap(merged: pd.DataFrame) -> pd.DataFrame:
    """행정동별 per_capita → 중앙값 정규화 → gap_index → gap_norm."""
    out = merged.copy()
    out = out[out["pop_total"] > 0].copy()
    out["per_capita"] = out["sales_total"] / out["pop_total"]

    seoul_median = out["per_capita"].median()
    print(f"  서울 행정동 per_capita 중앙값: {seoul_median:,.2f}")
    out["per_capita_rel"] = out["per_capita"] / seoul_median

    out["gap_index"] = 1 - out["per_capita_rel"]

    # gap_norm: PercentileRank → [0, 1]
    out["gap_norm"] = out["gap_index"].rank(pct=True)

    return out


def apply_outlier_exclusion(df: pd.DataFrame) -> pd.DataFrame:
    """이상치 5개 동 제외."""
    out = df.copy()
    out["adm_code"] = out["adm_code"].astype(str)
    before = len(out)
    out = out[~out["adm_code"].isin(OUTLIER_CODES)]
    after = len(out)
    print(f"  이상치 제외: {before} → {after} (제외 {before - after}개)")
    return out


# -----------------------------------------------------------------------------
# 통계청 → 행자부 코드 변환 (마스터 JOIN 사전 작업, 노션 §3.2 STEP 6)
# -----------------------------------------------------------------------------

def load_haengjabu_mapping(mapping_csv: Path | None) -> dict[str, tuple[str, str, str]]:
    """opening_rate_final.csv 또는 별도 매핑 csv에서 통계청 → 행자부 매핑 추출.

    매핑 csv 없을 시 빈 dict 반환 (EXCEPTION_MAPPING만 적용 — 7동).
    """
    if mapping_csv is None or not mapping_csv.exists():
        print("  매핑 csv 없음 — EXCEPTION 매핑 7동만 적용")
        return {}

    df = pd.read_csv(mapping_csv, dtype=str)
    src_candidates = ["통계청_코드", "stat_code", "행정동_코드"]
    tgt_candidates = ["행자부_코드", "haengjabu_code", "adm_code"]
    name_candidates = ["행자부_행정동명", "행정동_코드_명", "ADM_NM"]
    gu_candidates = ["gu_name", "자치구", "자치구명"]

    src_col = next((c for c in src_candidates if c in df.columns), None)
    tgt_col = next((c for c in tgt_candidates if c in df.columns), None)
    if src_col is None or tgt_col is None:
        print(f"  매핑 csv 컬럼 식별 실패: {df.columns.tolist()}")
        return {}

    name_col = next((c for c in name_candidates if c in df.columns), None)
    gu_col = next((c for c in gu_candidates if c in df.columns), None)

    df[src_col] = df[src_col].astype(str).str.zfill(8)
    df[tgt_col] = df[tgt_col].astype(str).str.zfill(8)

    mapping = {}
    for _, row in df.iterrows():
        src = row[src_col]
        tgt = row[tgt_col]
        nm = row[name_col] if name_col else ""
        gu = row[gu_col] if gu_col else ""
        mapping[src] = (tgt, nm, gu)
    print(f"  매핑 로드: {len(mapping)}동 (통계청 → 행자부)")
    return mapping


def apply_haengjabu_conversion(
    df: pd.DataFrame,
    haengjabu_mapping: dict[str, tuple[str, str, str]],
) -> pd.DataFrame:
    """통계청 → 행자부 변환 + matching_method 추적.

    우선순위: EXCEPTION 매핑 → opening 매핑 → 매칭_실패.
    """
    out = df.copy()
    out["amd_code"] = out["amd_code"].astype(str).str.zfill(8)

    def _resolve(code: str) -> tuple[str, str, str, str]:
        if code in EXCEPTION_MAPPING:
            tgt, nm, gu = EXCEPTION_MAPPING[code]
            return tgt, nm, gu, "강북_EXCEPTION_매핑" if gu == "강북구" else "개포1동_매핑"
        if code in haengjabu_mapping:
            tgt, nm, gu = haengjabu_mapping[code]
            return tgt, nm, gu, "코드매핑_성공"
        return code, "", "", "매칭_실패"

    resolved = out["amd_code"].apply(_resolve)
    out["행자부_코드"] = resolved.apply(lambda x: x[0])
    out["행자부_행정동명"] = resolved.apply(lambda x: x[1])
    out["gu_name"] = resolved.apply(lambda x: x[2])
    out["matching_method"] = resolved.apply(lambda x: x[3])

    methods = out["matching_method"].value_counts()
    print("  매핑 결과:")
    for m, n in methods.items():
        print(f"    {m:25s}: {n}동")
    return out


# -----------------------------------------------------------------------------
# 진입점
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="매출 미실현도 (gap_norm) 산출")

    parser.add_argument(
        "--b040", type=Path,
        default=Path("data/processed/b040/late_night/b040_monthly_late_night_pop.csv"),
        help="B040 월별 평균 인구 (캠퍼스 산출)",
    )
    parser.add_argument(
        "--b079", type=Path,
        default=Path("data/processed/b079/late_night/행정동_카드매출.csv"),
        help="B079 월별 카드매출 (팀원 산출)",
    )
    parser.add_argument(
        "--mapping-csv", type=Path,
        default=None,
        help="통계청↔행자부 매핑 csv (opening_rate_final.csv 활용). 없으면 EXCEPTION 7동만 적용.",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("data/processed/gap_index"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] 캠퍼스 산출물 로드")
    pop_monthly = pd.read_csv(args.b040, dtype={"adm_code": str})
    sales_monthly = pd.read_csv(args.b079, dtype={"amd_code": str})
    print(f"  B040: {len(pop_monthly)} 행 (월별 인구)")
    print(f"  B079: {len(sales_monthly)} 행 (월별 매출)")

    print("\n[2] 행정동 단위 인구·매출 통합 (B079 10자리 → B040 8자리 매핑)")
    merged = aggregate_to_dong(pop_monthly, sales_monthly)
    print(f"  inner-merge 후: {len(merged)}동")

    print("\n[3] 이상치 5개 동 제외")
    cleaned = apply_outlier_exclusion(merged)

    print("\n[4] gap_index 산출 (per_capita → 중앙값 정규화 → PercentileRank)")
    final = compute_gap(cleaned)

    # 반출용 컬럼만 남김 (절대값 제외)
    export = final[["adm_code", "per_capita_rel", "gap_index", "gap_norm"]].copy()
    export["per_capita_rel"] = export["per_capita_rel"].round(6)
    export["gap_index"] = export["gap_index"].round(6)
    export["gap_norm"] = export["gap_norm"].round(6)
    export = export.rename(columns={"adm_code": "amd_code"})

    # [4a] 캠퍼스 반출용 (통계청 코드 그대로) — 응용집계 형태
    out_path = args.out_dir / "b040_b079_gap.csv"
    export.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장 (캠퍼스 반출용): {out_path} ({len(export)} 행)")

    print("\n[5] 통계청 → 행자부 코드 변환 (EXCEPTION 7동 + opening 매핑)")
    mapping = load_haengjabu_mapping(args.mapping_csv)
    final_with_haengjabu = apply_haengjabu_conversion(export, mapping)

    # [5a] 마스터 JOIN용 — 행자부 코드 포함
    out_path_final = args.out_dir / "b040_b079_gap_final.csv"
    final_with_haengjabu.to_csv(out_path_final, index=False, encoding="utf-8-sig")
    print(f"\n저장 (마스터 JOIN용): {out_path_final} ({len(final_with_haengjabu)} 행)")

    # 검증 요약
    top5 = export.nlargest(5, "gap_norm")[["amd_code", "gap_norm"]]
    bot5 = export.nsmallest(5, "gap_norm")[["amd_code", "gap_norm"]]
    print("\n매출 미실현도 상위 5 (인구 대비 매출 부족 — 정책 후보):")
    print(top5.to_string(index=False))
    print("\n매출 미실현도 하위 5 (이미 인구 대비 매출 활성):")
    print(bot5.to_string(index=False))


if __name__ == "__main__":
    # 실행 예시 (필요 시 주석 해제):
    # import sys
    # sys.argv = [
    #     "gap_index.py",
    #     "--b040", "data/processed/b040/late_night/b040_monthly_late_night_pop.csv",
    #     "--b079", "data/processed/b079/late_night/행정동_카드매출.csv",
    #     "--out-dir", "data/processed/gap_index",
    # ]
    # main()
    pass
