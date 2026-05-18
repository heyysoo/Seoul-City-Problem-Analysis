"""50/50 매트릭스 분류: 저녁·심야 트랙

처리 목적
- 2차 필터 통과 80동 분석대상을 입력으로 받아 두 트랙의 사분면을 산출한다.
- 저녁 트랙: X = 잠재력_저녁 (percentile), Y = evening_sales_amt (percentile)
- 심야 트랙: X = 잠재력_심야 (percentile), Y = infrastructure_gap_3var (percentile)

분류 기준 (50/50 median split)
- 저녁 후보 = 우하_잠재력높음_매출낮음 (x>=0.5 & y<0.5)
- 심야 후보 = 우상_잠재력높음_인프라공백높음 (x>=0.5 & y>=0.5)

산출물
- evening_quadrants_2차{N}_lodging_included_no_manual_keep_50_50.csv
- late_quadrants_2차{N}_lodging_included_no_manual_keep_50_50.csv
- evening_candidates_2차{N}_lodging_included_no_manual_keep_50_50.csv
- late_candidates_2차{N}_lodging_included_no_manual_keep_50_50.csv
- matrix_report.md

입력
- analysis_targets CSV  (1_filter/commerce.py 출력)
- master_evening CSV    (2_index/ 출력; 행자부_행정동명, potential_evening, evening_sales_amt 필요)
- master_late CSV       (2_index/ 출력; 행자부_행정동명, potential_late, infrastructure_gap_3var 필요)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def norm_name(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace("·", ".", regex=False)


def add_key(df: pd.DataFrame, name_col: str, gu_col: str = "gu_name") -> pd.DataFrame:
    out = df.copy()
    out["_join_key"] = out[gu_col].astype(str).str.strip() + "|" + norm_name(out[name_col])
    return out


def q_evening(x: float, y: float) -> str:
    """저녁 트랙 사분면 라벨.

    X = 잠재력_저녁 (percentile), Y = evening_sales_amt (percentile).
    저녁 후보 = 우하_잠재력높음_매출낮음.
    """
    if x >= 0.5 and y < 0.5:
        return "우하_잠재력높음_매출낮음"
    if x >= 0.5 and y >= 0.5:
        return "우상_잠재력높음_매출높음"
    if x < 0.5 and y >= 0.5:
        return "좌상_잠재력낮음_매출높음"
    return "좌하_잠재력낮음_매출낮음"


def q_late(x: float, y: float) -> str:
    """심야 트랙 사분면 라벨.

    X = 잠재력_심야 (percentile), Y = infrastructure_gap_3var (percentile).
    심야 후보 = 우상_잠재력높음_인프라공백높음.
    """
    if x >= 0.5 and y >= 0.5:
        return "우상_잠재력높음_인프라공백높음"
    if x >= 0.5 and y < 0.5:
        return "우하_잠재력높음_인프라공백낮음"
    if x < 0.5 and y >= 0.5:
        return "좌상_잠재력낮음_인프라공백높음"
    return "좌하_잠재력낮음_인프라공백낮음"


def build_quadrants(
    analysis_targets: Path,
    master_evening: Path,
    master_late: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    target = pd.read_csv(analysis_targets, dtype=str)
    # secondary_exclude 컬럼이 있더라도 이미 PASS 풀이므로 그대로 사용
    target = add_key(target, "ADM_NM")
    target_keys = set(target["_join_key"])
    merge_cols = [
        "_join_key",
        "secondary_decision_no_manual_keep",
        "decision_reason_no_manual_keep",
        "commerce_score",
        "contamination_score",
        "lodging_count",
        "lodging_share",
        "lodging_score",
        "sales_aligned_count",
        "contamination_count",
        "res_ratio_NEW",
        "top1_addr_share",
        "top2_addr_share",
        "top2_addrs",
    ]
    merge_cols = [c for c in merge_cols if c in target.columns]

    evening = add_key(pd.read_csv(master_evening), "행자부_행정동명")
    e = evening[evening["_join_key"].isin(target_keys)].copy()
    e = e.merge(target[merge_cols], on="_join_key", how="left")
    e["potential_evening_pct_no_manual_keep"] = e["potential_evening"].rank(pct=True)
    e["evening_sales_pct_no_manual_keep"] = e["evening_sales_amt"].fillna(0).rank(pct=True)
    e["evening_quadrant_50"] = [
        q_evening(x, y)
        for x, y in zip(
            e["potential_evening_pct_no_manual_keep"],
            e["evening_sales_pct_no_manual_keep"],
        )
    ]
    e["evening_candidate_50"] = e["evening_quadrant_50"].eq("우하_잠재력높음_매출낮음")

    late = add_key(pd.read_csv(master_late), "행자부_행정동명")
    l = late[late["_join_key"].isin(target_keys)].copy()
    l = l.merge(target[merge_cols], on="_join_key", how="left")
    l["potential_late_pct_no_manual_keep"] = l["potential_late"].rank(pct=True)
    l["infrastructure_gap_pct_no_manual_keep"] = l["infrastructure_gap_3var"].rank(pct=True)
    l["late_quadrant_50"] = [
        q_late(x, y)
        for x, y in zip(
            l["potential_late_pct_no_manual_keep"],
            l["infrastructure_gap_pct_no_manual_keep"],
        )
    ]
    l["late_candidate_50"] = l["late_quadrant_50"].eq("우상_잠재력높음_인프라공백높음")

    meta = {
        "evening_rows": int(len(e)),
        "late_rows": int(len(l)),
        "evening_missing_axis": {
            k: int(v) for k, v in e[["potential_evening", "evening_sales_amt"]].isna().sum().items()
        },
        "late_missing_axis": {
            k: int(v) for k, v in l[["potential_late", "infrastructure_gap_3var"]].isna().sum().items()
        },
        "late_gap_norm_missing": int(l["gap_norm"].isna().sum()) if "gap_norm" in l.columns else None,
    }
    return e, l, meta


def write_matrix_report(out_dir: Path, qmeta: dict[str, object], e: pd.DataFrame, l: pd.DataFrame) -> None:
    e_cand = e[e["evening_candidate_50"]]
    l_cand = l[l["late_candidate_50"]]
    report = f"""# 50/50 매트릭스 분류 결과

## 사분면 검산

| 항목 | 결과 |
| --- | ---: |
| 저녁 master 매칭 | {qmeta['evening_rows']} |
| 심야 master 매칭 | {qmeta['late_rows']} |
| 저녁 후보 | {len(e_cand)} |
| 심야 후보 | {len(l_cand)} |

저녁 후보: {", ".join(e_cand.sort_values(["gu_name", "행자부_행정동명"])["행자부_행정동명"].tolist())}

심야 후보: {", ".join(l_cand.sort_values(["gu_name", "행자부_행정동명"])["행자부_행정동명"].tolist())}
"""
    (out_dir / "matrix_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="50/50 매트릭스 분류 (저녁·심야 트랙)")
    parser.add_argument(
        "--analysis-targets",
        type=Path,
        default=Path("data/processed/secondary_filter/secondary_filter_analysis_targets_80_lodging_included_no_manual_keep.csv"),
    )
    parser.add_argument("--master-evening", type=Path,
                        default=Path("data/processed/master/master_table_evening.csv"))
    parser.add_argument("--master-late", type=Path,
                        default=Path("data/processed/master/master_table_late.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/matrix"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    e, l, qmeta = build_quadrants(args.analysis_targets, args.master_evening, args.master_late)
    n = qmeta["evening_rows"]

    e.to_csv(
        args.out_dir / f"evening_quadrants_2차{n}_lodging_included_no_manual_keep_50_50.csv",
        index=False, encoding="utf-8-sig",
    )
    l.to_csv(
        args.out_dir / f"late_quadrants_2차{n}_lodging_included_no_manual_keep_50_50.csv",
        index=False, encoding="utf-8-sig",
    )
    e[e["evening_candidate_50"]].to_csv(
        args.out_dir / f"evening_candidates_2차{n}_lodging_included_no_manual_keep_50_50.csv",
        index=False, encoding="utf-8-sig",
    )
    l[l["late_candidate_50"]].to_csv(
        args.out_dir / f"late_candidates_2차{n}_lodging_included_no_manual_keep_50_50.csv",
        index=False, encoding="utf-8-sig",
    )

    # downstream (4_priority.py) 용 단순 별칭 + 통합 master_80
    # 위의 긴 파일명은 노션·자료실 정합용으로 유지하되 후속 파이프라인은 단순명으로 받음
    e.to_csv(args.out_dir / "evening_quadrants.csv", index=False, encoding="utf-8-sig")
    l.to_csv(args.out_dir / "late_quadrants.csv", index=False, encoding="utf-8-sig")
    e[e["evening_candidate_50"]].to_csv(
        args.out_dir / "candidates_evening.csv", index=False, encoding="utf-8-sig",
    )
    l[l["late_candidate_50"]].to_csv(
        args.out_dir / "candidates_late.csv", index=False, encoding="utf-8-sig",
    )

    # 통합 master_80 (저녁·심야 매트릭스 결과 결합, 4_priority 입력용)
    eve_keep = ["행자부_행정동명", "potential_evening", "evening_sales_norm",
                "evening_quadrant_50", "evening_candidate_50"]
    lat_keep = ["행자부_행정동명", "potential_late", "infrastructure_gap_3var",
                "late_quadrant_50", "late_candidate_50"]
    eve_subset = e[[c for c in eve_keep if c in e.columns]]
    lat_subset = l[[c for c in lat_keep if c in l.columns]]
    master_80 = eve_subset.merge(lat_subset, on="행자부_행정동명", how="outer")
    master_80.to_csv(args.out_dir / "master_80.csv", index=False, encoding="utf-8-sig")
    write_matrix_report(args.out_dir, qmeta, e, l)

    print("QMETA", qmeta)
    print("저장 폴더:", args.out_dir)


if __name__ == "__main__":
    # 실행 예시 (필요 시 주석 해제):
    # import sys
    # sys.argv = [
    #     "matrix.py",
    #     "--analysis-targets", "data/processed/secondary_filter/secondary_filter_analysis_targets_80_lodging_included_no_manual_keep.csv",
    #     "--master-evening", "data/processed/index/master_evening.csv",
    #     "--master-late", "data/processed/index/master_late.csv",
    # ]
    # main()
    pass
