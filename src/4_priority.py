"""정책 적용 우선순위 산출 (4단계 정합성 점검 후 20동 대상)

처리 목적
- 3단계 50/50 매트릭스 + 4단계 정합성 점검을 통과한 정책 적용 후보 20개 동에 대해
  정량 데이터 기반 정책 적용 우선순위 산출.
- 발표에서 대표 6동 선정 + 나머지 14동 부록 매핑 표 작성에 직접 활용.

핵심 결정사항 4가지
- 기준 집단: 80동 (2차 필터 통과 분석대상)
  → 426동 기준 percentile은 2차 필터·매트릭스 맥락과 동떨어짐
- 심야 산정: 4요소 (잠재력 + 인프라공백 + 범죄 + 매출미실현)
  → 후보 추출은 2축이지만 정책 적용 우선순위는 시점·강도 판단 위해 4요소 종합 필요
- 메인 방식: 순위합산 (분포 차이 영향 제거, 모든 변수 동등 가중치)
- 보조 방식: 단순합산 (민감도 검증, 절대값 영향 직접 반영)

산식 (노션 §1.3)
- 저녁 트랙 (2축):
  - 메인: rank_80(잠재력 높은 순) + rank_80(저녁 매출 낮은 순)
  - 보조: potential_evening_pct + (1 - evening_sales_pct)
- 심야 트랙 (4요소):
  - 메인: rank_80(잠재력↑) + rank_80(인프라공백↑) + rank_80(범죄↑) + rank_80(매출미실현↑)
  - 보조: potential_late_pct + infrastructure_gap_pct + crime_percentile + gap_norm

대표 6동 선정 원칙 (노션 §1.4)
- 저녁 트랙 상위 2동 (양 트랙 후보는 공통 슬롯으로 분리)
- 양 트랙 공통 후보 상위 2동
- 심야 트랙 상위 2동 (양 트랙 후보는 공통 슬롯으로 분리)

입력
- master csv: 80동 * 6개 변수 컬럼 (잠재력_저녁·잠재력_심야·매출·인프라공백·범죄·매출미실현)
- candidates_evening csv: 저녁 트랙 정책 후보 동 코드 list (도봉2 F 보류 후 10동)
- candidates_late csv: 심야 트랙 정책 후보 동 코드 list (13동)

산출물 (out_dir 하위)
- priority_evening.csv: 저녁 트랙 우선순위 (10동, 메인·보조 순위 + 차이)
- priority_late.csv: 심야 트랙 우선순위 (13동, 메인·보조 순위 + 차이)
- priority_common.csv: 양 트랙 공통 후보 비교 (저녁/심야/평균 순위)
- priority_top6.csv: 대표 6동 추천 (저녁 2 + 공통 2 + 심야 2)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# -----------------------------------------------------------------------------
# 상수
# -----------------------------------------------------------------------------

KEY_COL_DEFAULT = "amd_code"

# 기본 컬럼명 (마스터 테이블 컨벤션 — 3_matrix.py 산출)
DEFAULT_COLS = {
    "potential_evening_pct": "potential_evening_pct_no_manual_keep",
    "potential_late_pct": "potential_late_pct_no_manual_keep",
    "evening_sales_pct": "evening_sales_pct_no_manual_keep",
    "infrastructure_gap_pct": "infrastructure_gap_pct_no_manual_keep",
    "crime_percentile": "crime_percentile",
    "gap_norm": "gap_norm",
    "dong_name": "dong_name",
}


# -----------------------------------------------------------------------------
# 입력 로딩 + 후보 식별
# -----------------------------------------------------------------------------

def load_master(path: Path, key_col: str) -> pd.DataFrame:
    """80동 마스터 csv 로드."""
    df = pd.read_csv(path, dtype={key_col: str})
    df[key_col] = df[key_col].astype(str).str.strip()
    if len(df) != 80:
        print(f"  master 행수 {len(df)} (정상: 80). 후속 rank가 80동 기준이 아닐 수 있음.")
    return df


def load_candidates(path: Path, key_col: str) -> list[str]:
    """후보 동 코드 csv 로드 (key_col 단일 컬럼 또는 첫 컬럼 사용)."""
    df = pd.read_csv(path, dtype=str)
    col = key_col if key_col in df.columns else df.columns[0]
    return df[col].astype(str).str.strip().tolist()


# -----------------------------------------------------------------------------
# 핵심: 트랙별 우선순위 산출
# -----------------------------------------------------------------------------

def compute_evening_priority(
    master: pd.DataFrame,
    candidates: list[str],
    key_col: str,
    cols: dict,
) -> pd.DataFrame:
    """저녁 트랙 우선순위 산출 (2축).

    절차:
    1) 80동 마스터 기준 rank 산출 — 잠재력 높은 순 / 매출 낮은 순
    2) 메인 = rank 합산, 보조 = pct + (1 - pct) 합산
    3) 후보 동만 필터링 + 정렬
    """
    out = master.copy()

    # 80동 기준 rank
    out["rank_pot_eve"] = out[cols["potential_evening_pct"]].rank(ascending=False, method="min").astype(int)
    out["rank_sales_low"] = out[cols["evening_sales_pct"]].rank(ascending=True, method="min").astype(int)

    # 메인: 순위합
    out["main_score"] = out["rank_pot_eve"] + out["rank_sales_low"]

    # 보조: 단순합 (pct + (1 - pct))
    out["aux_score"] = out[cols["potential_evening_pct"]] + (1 - out[cols["evening_sales_pct"]])

    # 후보 동만 필터링
    candidates_set = set(candidates)
    out = out[out[key_col].isin(candidates_set)].copy()

    # 순위 부여 (후보 내)
    out["main_rank"] = out["main_score"].rank(ascending=True, method="min").astype(int)
    out["aux_rank"] = out["aux_score"].rank(ascending=False, method="min").astype(int)
    out["rank_diff"] = out["aux_rank"] - out["main_rank"]

    result_cols = [
        key_col, cols["dong_name"],
        "rank_pot_eve", "rank_sales_low",
        "main_score", "main_rank",
        "aux_score", "aux_rank", "rank_diff",
    ]
    result_cols = [c for c in result_cols if c in out.columns]
    return out[result_cols].sort_values("main_rank").reset_index(drop=True)


def compute_late_priority(
    master: pd.DataFrame,
    candidates: list[str],
    key_col: str,
    cols: dict,
) -> pd.DataFrame:
    """심야 트랙 우선순위 산출 (4요소).

    절차:
    1) 80동 마스터 기준 rank 산출 — 4개 변수 모두 높은 순
    2) 메인 = 4개 rank 합산, 보조 = 4개 pct 단순합
    3) 후보 동만 필터링 + 정렬
    """
    out = master.copy()

    # 80동 기준 rank (4 변수 모두 큰 값이 우선순위 높음)
    out["rank_pot_late"] = out[cols["potential_late_pct"]].rank(ascending=False, method="min").astype(int)
    out["rank_infra"] = out[cols["infrastructure_gap_pct"]].rank(ascending=False, method="min").astype(int)
    out["rank_crime"] = out[cols["crime_percentile"]].rank(ascending=False, method="min").astype(int)
    out["rank_gap_norm"] = out[cols["gap_norm"]].rank(ascending=False, method="min").astype(int)

    # 메인: 4개 rank 합산
    out["main_score"] = (
        out["rank_pot_late"]
        + out["rank_infra"]
        + out["rank_crime"]
        + out["rank_gap_norm"]
    )

    # 보조: 4개 pct 단순합
    out["aux_score"] = (
        out[cols["potential_late_pct"]]
        + out[cols["infrastructure_gap_pct"]]
        + out[cols["crime_percentile"]]
        + out[cols["gap_norm"]]
    )

    # 후보 동만 필터링
    candidates_set = set(candidates)
    out = out[out[key_col].isin(candidates_set)].copy()

    # 순위 부여 (후보 내)
    out["main_rank"] = out["main_score"].rank(ascending=True, method="min").astype(int)
    out["aux_rank"] = out["aux_score"].rank(ascending=False, method="min").astype(int)
    out["rank_diff"] = out["aux_rank"] - out["main_rank"]

    result_cols = [
        key_col, cols["dong_name"],
        "rank_pot_late", "rank_infra", "rank_crime", "rank_gap_norm",
        "main_score", "main_rank",
        "aux_score", "aux_rank", "rank_diff",
    ]
    result_cols = [c for c in result_cols if c in out.columns]
    return out[result_cols].sort_values("main_rank").reset_index(drop=True)


# -----------------------------------------------------------------------------
# 양 트랙 공통 후보 비교
# -----------------------------------------------------------------------------

def compare_common(
    evening_priority: pd.DataFrame,
    late_priority: pd.DataFrame,
    key_col: str,
    dong_name_col: str,
) -> pd.DataFrame:
    """양 트랙 모두에 등장한 후보 동의 저녁·심야 순위 비교."""
    common_codes = set(evening_priority[key_col]) & set(late_priority[key_col])

    eve = evening_priority[evening_priority[key_col].isin(common_codes)][
        [key_col, dong_name_col, "main_rank"]
    ].rename(columns={"main_rank": "evening_rank"})

    lat = late_priority[late_priority[key_col].isin(common_codes)][
        [key_col, "main_rank"]
    ].rename(columns={"main_rank": "late_rank"})

    merged = eve.merge(lat, on=key_col, how="inner")
    merged["avg_rank"] = (merged["evening_rank"] + merged["late_rank"]) / 2
    return merged.sort_values("avg_rank").reset_index(drop=True)


# -----------------------------------------------------------------------------
# 대표 6동 선정
# -----------------------------------------------------------------------------

def select_top6(
    evening_priority: pd.DataFrame,
    late_priority: pd.DataFrame,
    common: pd.DataFrame,
    key_col: str,
    dong_name_col: str,
) -> pd.DataFrame:
    """대표 6동 선정: 저녁 2 + 공통 2 + 심야 2 (양 트랙 후보는 공통 슬롯으로 분리).

    선정 규칙:
    1) 공통 슬롯: 양 트랙 후보 중 평균 순위 상위 2동
    2) 저녁 슬롯: 저녁 후보 (양 트랙 후보 제외) 중 main_rank 상위 2동
    3) 심야 슬롯: 심야 후보 (양 트랙 후보 제외) 중 main_rank 상위 2동
    """
    common_codes = set(common[key_col])

    # 공통 2동: 평균 순위 1·2위
    common_top2 = common.head(2).copy()
    common_top2["slot"] = ["common_1", "common_2"]
    common_top2 = common_top2.rename(columns={"avg_rank": "rank_value"})
    common_top2 = common_top2[[key_col, dong_name_col, "slot", "rank_value"]]

    # 저녁 2동 (양 트랙 후보 제외)
    eve_only = evening_priority[~evening_priority[key_col].isin(common_codes)].head(2).copy()
    eve_only["slot"] = ["evening_1", "evening_2"]
    eve_only = eve_only.rename(columns={"main_rank": "rank_value"})
    eve_only = eve_only[[key_col, dong_name_col, "slot", "rank_value"]]

    # 심야 2동 (양 트랙 후보 제외)
    lat_only = late_priority[~late_priority[key_col].isin(common_codes)].head(2).copy()
    lat_only["slot"] = ["late_1", "late_2"]
    lat_only = lat_only.rename(columns={"main_rank": "rank_value"})
    lat_only = lat_only[[key_col, dong_name_col, "slot", "rank_value"]]

    return pd.concat([eve_only, common_top2, lat_only], ignore_index=True)


# -----------------------------------------------------------------------------
# 진입점
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="정책 적용 우선순위 산출 (저녁·심야 트랙)")

    parser.add_argument("--key-col", type=str, default=KEY_COL_DEFAULT)

    # 입력
    parser.add_argument(
        "--master", type=Path,
        default=Path("data/processed/matrix/master_80.csv"),
        help="80동 마스터 (모든 변수 포함)",
    )
    parser.add_argument(
        "--candidates-evening", type=Path,
        default=Path("data/processed/matrix/candidates_evening.csv"),
        help="저녁 트랙 정책 후보 동 코드 list",
    )
    parser.add_argument(
        "--candidates-late", type=Path,
        default=Path("data/processed/matrix/candidates_late.csv"),
        help="심야 트랙 정책 후보 동 코드 list",
    )

    # 컬럼명 (마스터 컨벤션과 다를 경우 조정)
    parser.add_argument("--col-potential-evening", type=str, default=DEFAULT_COLS["potential_evening_pct"])
    parser.add_argument("--col-potential-late", type=str, default=DEFAULT_COLS["potential_late_pct"])
    parser.add_argument("--col-evening-sales", type=str, default=DEFAULT_COLS["evening_sales_pct"])
    parser.add_argument("--col-infrastructure-gap", type=str, default=DEFAULT_COLS["infrastructure_gap_pct"])
    parser.add_argument("--col-crime", type=str, default=DEFAULT_COLS["crime_percentile"])
    parser.add_argument("--col-gap-norm", type=str, default=DEFAULT_COLS["gap_norm"])
    parser.add_argument("--col-dong-name", type=str, default=DEFAULT_COLS["dong_name"])

    # 출력
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/priority"))

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cols = {
        "potential_evening_pct": args.col_potential_evening,
        "potential_late_pct": args.col_potential_late,
        "evening_sales_pct": args.col_evening_sales,
        "infrastructure_gap_pct": args.col_infrastructure_gap,
        "crime_percentile": args.col_crime,
        "gap_norm": args.col_gap_norm,
        "dong_name": args.col_dong_name,
    }

    print("[1] 입력 로딩")
    master = load_master(args.master, args.key_col)
    cands_eve = load_candidates(args.candidates_evening, args.key_col)
    cands_lat = load_candidates(args.candidates_late, args.key_col)
    print(f"  master: {len(master)}동 / 저녁 후보: {len(cands_eve)}동 / 심야 후보: {len(cands_lat)}동")

    print("\n[2] 저녁 트랙 우선순위 산출 (2축)")
    pri_eve = compute_evening_priority(master, cands_eve, args.key_col, cols)
    eve_path = args.out_dir / "priority_evening.csv"
    pri_eve.to_csv(eve_path, index=False, encoding="utf-8-sig")
    print(f"  저장: {eve_path} ({len(pri_eve)}동)")

    print("\n[3] 심야 트랙 우선순위 산출 (4요소)")
    pri_lat = compute_late_priority(master, cands_lat, args.key_col, cols)
    lat_path = args.out_dir / "priority_late.csv"
    pri_lat.to_csv(lat_path, index=False, encoding="utf-8-sig")
    print(f"  저장: {lat_path} ({len(pri_lat)}동)")

    print("\n[4] 양 트랙 공통 후보 비교")
    common = compare_common(pri_eve, pri_lat, args.key_col, args.col_dong_name)
    common_path = args.out_dir / "priority_common.csv"
    common.to_csv(common_path, index=False, encoding="utf-8-sig")
    print(f"  저장: {common_path} ({len(common)}동)")

    print("\n[5] 대표 6동 선정")
    top6 = select_top6(pri_eve, pri_lat, common, args.key_col, args.col_dong_name)
    top6_path = args.out_dir / "priority_top6.csv"
    top6.to_csv(top6_path, index=False, encoding="utf-8-sig")
    print(f"  저장: {top6_path}")
    print(top6.to_string(index=False))


if __name__ == "__main__":
    # 실행 예시 (필요 시 주석 해제):
    # import sys
    # sys.argv = [
    #     "4_priority.py",
    #     "--master", "data/processed/matrix/master_80.csv",
    #     "--candidates-evening", "data/processed/matrix/candidates_evening.csv",
    #     "--candidates-late", "data/processed/matrix/candidates_late.csv",
    #     "--out-dir", "data/processed/priority",
    # ]
    # main()
    pass
