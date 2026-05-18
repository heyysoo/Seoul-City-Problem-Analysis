"""
2차 필터링(상권성 필터) 제출용 코드.

처리 목적
- 1차 거주지 필터 통과 108동을 대상으로 상권성 점수를 산출한다.
- 숙박업을 저녁·심야 체류 수요의 positive signal에 포함한다.
- 상권성 하위 28%를 저상권성 검토 후보로 탐지한다.
- 야간 소비 관련 점포 수(sales_aligned_count)가 400개 이상이면 규모 예외로 유지한다.
- 수동 유지/제외 플래그 없이 80동 분석대상 풀을 확정하고 저녁·심야 사분면을 재계산한다.

핵심 산식
- sales_aligned_density = sales_aligned_count / urban_area_km2
- sales_aligned_share = sales_aligned_count / total_stores
- commerce_score = density_percentile * 0.70 + share_percentile * 0.30
- low_commerce_exclude = commerce_score <= q28 AND sales_aligned_count < 400
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COMMERCE_QUANTILE = 0.28
SCALE_EXCEPTION_MIN_SALES_ALIGNED = 400

BASE_SALES_KEYWORDS = (
    "음식",
    "한식",
    "중식",
    "일식",
    "양식",
    "분식",
    "패스트푸드",
    "치킨",
    "피자",
    "주점",
    "호프",
    "맥주",
    "술집",
    "포차",
    "카페",
    "커피",
    "제과",
    "제빵",
    "디저트",
    "편의점",
    "오락",
    "노래",
    "PC",
    "게임",
    "문화",
    "영화",
    "공연",
    "서점",
)
LODGING_KEYWORDS = ("숙박", "호텔", "모텔", "여관", "게스트하우스")
SALES_KEYWORDS_LODGING_INCLUDED = BASE_SALES_KEYWORDS + LODGING_KEYWORDS
CONTAMINATION_KEYWORDS_WITHOUT_LODGING = (
    "병원",
    "의원",
    "약국",
    "한의원",
    "치과",
    "의료",
    "고시원",
    "고시텔",
    "독서실",
    "스터디",
    "헬스",
    "피트니스",
    "요가",
    "필라테스",
    "체육",
    "운동",
    "세탁",
    "빨래",
    "마사지",
    "피부",
    "미용",
)


def normalize_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def contains_any(value: object, keywords: tuple[str, ...]) -> bool:
    text = "" if pd.isna(value) else str(value)
    return any(keyword in text for keyword in keywords)


def norm_name(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace("·", ".", regex=False)


def add_key(df: pd.DataFrame, name_col: str, gu_col: str = "gu_name") -> pd.DataFrame:
    out = df.copy()
    out["_join_key"] = out[gu_col].astype(str).str.strip() + "|" + norm_name(out[name_col])
    return out


def q_evening(x: float, y: float) -> str:
    if x >= 0.5 and y < 0.5:
        return "우하_잠재력높음_매출낮음"
    if x >= 0.5 and y >= 0.5:
        return "우상_잠재력높음_매출높음"
    if x < 0.5 and y >= 0.5:
        return "좌상_잠재력낮음_매출높음"
    return "좌하_잠재력낮음_매출낮음"


def q_late(x: float, y: float) -> str:
    if x >= 0.5 and y >= 0.5:
        return "우상_잠재력높음_인프라공백높음"
    if x >= 0.5 and y < 0.5:
        return "우하_잠재력높음_인프라공백낮음"
    if x < 0.5 and y >= 0.5:
        return "좌상_잠재력낮음_인프라공백높음"
    return "좌하_잠재력낮음_인프라공백낮음"


def address_concentration(stores: pd.DataFrame, dong_name: str) -> dict[str, object]:
    if "rdnmAdr" not in stores.columns:
        return {"store_rows": 0, "top1_addr_share": 0.0, "top2_addr_share": 0.0, "top2_addrs": ""}
    rows = stores[stores["adongNm"].astype(str).eq(dong_name)]
    if rows.empty:
        return {"store_rows": 0, "top1_addr_share": 0.0, "top2_addr_share": 0.0, "top2_addrs": ""}
    vc = rows["rdnmAdr"].value_counts(dropna=False)
    return {
        "store_rows": int(len(rows)),
        "top1_addr_share": float(vc.iloc[0] / len(rows)),
        "top2_addr_share": float(vc.head(2).sum() / len(rows)),
        "top2_addrs": " / ".join(map(str, vc.head(2).index.tolist())),
    }


def build_commerce_summary(pass_api_codes: Path, stores_raw: Path) -> pd.DataFrame:
    pass_df = pd.read_csv(pass_api_codes, dtype=str)
    stores = pd.read_csv(stores_raw, dtype=str, low_memory=False)

    name_cols = [col for col in ["indsLclsNm", "indsMclsNm", "indsSclsNm", "ksicNm", "bizesNm"] if col in stores.columns]
    if not name_cols:
        raise KeyError("상가업소 원자료에 업종/상호명 텍스트 컬럼이 없습니다.")

    stores["adongCd"] = normalize_code(stores["adongCd"])
    stores["category_text"] = stores[name_cols].fillna("").agg(" ".join, axis=1)
    stores["is_lodging"] = stores["category_text"].map(lambda x: contains_any(x, LODGING_KEYWORDS))
    stores["is_sales_aligned"] = stores["category_text"].map(lambda x: contains_any(x, SALES_KEYWORDS_LODGING_INCLUDED))
    stores["is_contamination"] = stores["category_text"].map(lambda x: contains_any(x, CONTAMINATION_KEYWORDS_WITHOUT_LODGING))
    stores["is_food"] = stores["category_text"].str.contains("음식|한식|중식|일식|양식|분식|치킨|피자|패스트푸드", regex=True, na=False)
    stores["is_cafe"] = stores["category_text"].str.contains("카페|커피|제과|제빵|디저트", regex=True, na=False)
    stores["is_pub"] = stores["category_text"].str.contains("주점|호프|맥주|술집|포차", regex=True, na=False)
    stores["is_entertainment_culture"] = stores["category_text"].str.contains("오락|노래|PC|게임|문화|영화|공연|서점", regex=True, na=False)
    stores["is_convenience"] = stores["category_text"].str.contains("편의점", regex=True, na=False)

    grouped = stores.groupby("adongCd").agg(
        total_stores=("bizesId", "count"),
        sales_aligned_count=("is_sales_aligned", "sum"),
        lodging_count=("is_lodging", "sum"),
        contamination_count=("is_contamination", "sum"),
        food_count=("is_food", "sum"),
        cafe_count=("is_cafe", "sum"),
        pub_count=("is_pub", "sum"),
        entertainment_culture_count=("is_entertainment_culture", "sum"),
        convenience_count=("is_convenience", "sum"),
    )

    base_cols = ["ADM_CD", "ADM_CD_source", "ADM_NM", "gu_name", "urban_area_m2", "res_ratio_NEW"]
    missing = [col for col in base_cols if col not in pass_df.columns]
    if missing:
        raise KeyError(f"1차 필터 통과 코드 파일 필수 컬럼 없음: {missing}")
    base = pass_df[base_cols].copy()
    base["ADM_CD"] = normalize_code(base["ADM_CD"])
    base["urban_area_m2"] = pd.to_numeric(base["urban_area_m2"], errors="coerce")

    out = base.merge(grouped, left_on="ADM_CD", right_index=True, how="left")
    count_cols = [
        "total_stores",
        "sales_aligned_count",
        "lodging_count",
        "contamination_count",
        "food_count",
        "cafe_count",
        "pub_count",
        "entertainment_culture_count",
        "convenience_count",
    ]
    out[count_cols] = out[count_cols].fillna(0).astype(int)
    out["urban_area_km2"] = out["urban_area_m2"] / 1_000_000
    out["sales_aligned_density"] = out["sales_aligned_count"] / out["urban_area_km2"]
    out["contamination_density"] = out["contamination_count"] / out["urban_area_km2"]
    out["lodging_density"] = out["lodging_count"] / out["urban_area_km2"]
    out["sales_aligned_share"] = out["sales_aligned_count"] / out["total_stores"].replace(0, pd.NA)
    out["contamination_share"] = out["contamination_count"] / out["total_stores"].replace(0, pd.NA)
    out["lodging_share"] = out["lodging_count"] / out["total_stores"].replace(0, pd.NA)
    out[["sales_aligned_share", "contamination_share", "lodging_share"]] = out[
        ["sales_aligned_share", "contamination_share", "lodging_share"]
    ].fillna(0)

    out["sales_aligned_density_pct"] = out["sales_aligned_density"].rank(pct=True)
    out["sales_aligned_share_pct"] = out["sales_aligned_share"].rank(pct=True)
    out["commerce_score"] = out["sales_aligned_density_pct"] * 0.70 + out["sales_aligned_share_pct"] * 0.30
    out["contamination_density_pct"] = out["contamination_density"].rank(pct=True)
    out["contamination_share_pct"] = out["contamination_share"].rank(pct=True)
    out["contamination_score"] = out["contamination_density_pct"] * 0.50 + out["contamination_share_pct"] * 0.50
    out["lodging_density_pct"] = out["lodging_density"].rank(pct=True)
    out["lodging_share_pct"] = out["lodging_share"].rank(pct=True)
    out["lodging_score"] = out["lodging_density_pct"] * 0.50 + out["lodging_share_pct"] * 0.50
    return out


def build_decision(summary: pd.DataFrame, stores_raw: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    stores = pd.read_csv(stores_raw, dtype=str, low_memory=False)
    decision = summary.copy()
    commerce_cut = float(decision["commerce_score"].quantile(COMMERCE_QUANTILE))
    decision["low_commerce_candidate"] = decision["commerce_score"] <= commerce_cut
    decision["scale_exception"] = decision["sales_aligned_count"] >= SCALE_EXCEPTION_MIN_SALES_ALIGNED
    decision["low_commerce_exclude"] = decision["low_commerce_candidate"] & ~decision["scale_exception"]
    decision["secondary_exclude_no_manual_keep"] = decision["low_commerce_exclude"]
    decision["secondary_decision_no_manual_keep"] = "ANALYSIS_TARGET"
    decision.loc[decision["secondary_exclude_no_manual_keep"], "secondary_decision_no_manual_keep"] = "SECONDARY_EXCLUDE"

    def reason(row: pd.Series) -> str:
        if row["low_commerce_exclude"]:
            return f"low_commerce: commerce_score <= q28({commerce_cut:.6f}), no scale exception"
        if row["scale_exception"] and row["low_commerce_candidate"]:
            return f"kept_by_scale_exception: sales_aligned_count >= {SCALE_EXCEPTION_MIN_SALES_ALIGNED}"
        return "pass_secondary_commerce_filter"

    decision["decision_reason_no_manual_keep"] = decision.apply(reason, axis=1)
    for col in ["store_rows", "top1_addr_share", "top2_addr_share", "top2_addrs"]:
        decision[col] = None
    for idx, row in decision.iterrows():
        info = address_concentration(stores, row["ADM_NM"])
        for col, value in info.items():
            decision.at[idx, col] = value

    decision["exclude_category"] = "PASS"
    decision.loc[decision["low_commerce_exclude"], "exclude_category"] = "LOW_COMMERCE"
    meta = {
        "total_108": int(len(decision)),
        "commerce_cut": commerce_cut,
        "low_commerce_candidates": int(decision["low_commerce_candidate"].sum()),
        "scale_exceptions": int((decision["low_commerce_candidate"] & decision["scale_exception"]).sum()),
        "final_excluded": int(decision["secondary_exclude_no_manual_keep"].sum()),
        "final_targets": int((~decision["secondary_exclude_no_manual_keep"]).sum()),
    }
    return decision.sort_values(
        ["secondary_exclude_no_manual_keep", "exclude_category", "commerce_score", "ADM_NM"],
        ascending=[False, True, True, True],
    ), meta


def build_quadrants(
    decision: pd.DataFrame,
    master_evening: Path,
    master_late: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    target = decision[~decision["secondary_exclude_no_manual_keep"]].copy()
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

    evening = add_key(pd.read_csv(master_evening), "행자부_행정동명")
    e = evening[evening["_join_key"].isin(target_keys)].copy()
    e = e.merge(target[merge_cols], on="_join_key", how="left")
    e["potential_evening_pct_no_manual_keep"] = e["potential_evening"].rank(pct=True)
    e["evening_sales_pct_no_manual_keep"] = e["evening_sales_amt"].fillna(0).rank(pct=True)
    e["evening_quadrant_50"] = [
        q_evening(x, y) for x, y in zip(e["potential_evening_pct_no_manual_keep"], e["evening_sales_pct_no_manual_keep"])
    ]
    e["evening_candidate_50"] = e["evening_quadrant_50"].eq("우하_잠재력높음_매출낮음")

    late = add_key(pd.read_csv(master_late), "행자부_행정동명")
    l = late[late["_join_key"].isin(target_keys)].copy()
    l = l.merge(target[merge_cols], on="_join_key", how="left")
    l["potential_late_pct_no_manual_keep"] = l["potential_late"].rank(pct=True)
    l["infrastructure_gap_pct_no_manual_keep"] = l["infrastructure_gap_3var"].rank(pct=True)
    l["late_quadrant_50"] = [
        q_late(x, y) for x, y in zip(l["potential_late_pct_no_manual_keep"], l["infrastructure_gap_pct_no_manual_keep"])
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


def write_report(out_dir: Path, meta: dict[str, object], qmeta: dict[str, object], decision: pd.DataFrame, e: pd.DataFrame, l: pd.DataFrame) -> None:
    excluded = decision[decision["secondary_exclude_no_manual_keep"]]
    scale_kept = decision[decision["low_commerce_candidate"] & decision["scale_exception"]]
    e_cand = e[e["evening_candidate_50"]]
    l_cand = l[l["late_candidate_50"]]
    report = f"""# 2차 필터링 상권성 산출 보고

## 필터 결과

| 항목 | 결과 |
| --- | ---: |
| 시작 풀 | {meta['total_108']} |
| commerce_score 컷오프 | {meta['commerce_cut']:.6f} |
| 저상권성 검토 후보 | {meta['low_commerce_candidates']} |
| 규모 예외 유지 | {meta['scale_exceptions']} |
| 최종 제외 | {meta['final_excluded']} |
| 최종 분석대상 | {meta['final_targets']} |

규모 예외 유지 동: {", ".join(scale_kept["ADM_NM"].tolist()) or "없음"}

최종 제외 동: {", ".join(excluded.sort_values(["gu_name", "ADM_NM"])["ADM_NM"].tolist())}

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
    (out_dir / "secondary_filter_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="숙박 포함 무예외 2차 상권성 필터")
    parser.add_argument("--pass-api-codes", type=Path, default=Path("data/interim/residential_filter_pass_v4_api_codes.csv"))
    parser.add_argument("--stores-raw", type=Path, default=Path("data/interim/sdsc_stores_108_raw.csv"))
    parser.add_argument("--master-evening", type=Path, default=Path("data/interim/master_evening.csv"))
    parser.add_argument("--master-late", type=Path, default=Path("data/interim/master_late.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/secondary_filter"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_commerce_summary(args.pass_api_codes, args.stores_raw)
    decision, meta = build_decision(summary, args.stores_raw)
    e, l, qmeta = build_quadrants(decision, args.master_evening, args.master_late)
    n = meta["final_targets"]

    summary.to_csv(args.out_dir / "secondary_filter_sdsc_summary_lodging_included_no_manual_keep.csv", index=False, encoding="utf-8-sig")
    decision.to_csv(args.out_dir / "secondary_filter_decision_lodging_included_no_manual_keep_108.csv", index=False, encoding="utf-8-sig")
    decision[~decision["secondary_exclude_no_manual_keep"]].to_csv(
        args.out_dir / f"secondary_filter_analysis_targets_{n}_lodging_included_no_manual_keep.csv",
        index=False,
        encoding="utf-8-sig",
    )
    decision[decision["secondary_exclude_no_manual_keep"]].to_csv(
        args.out_dir / "secondary_filter_excluded_lodging_included_no_manual_keep.csv",
        index=False,
        encoding="utf-8-sig",
    )
    e.to_csv(args.out_dir / f"evening_quadrants_2차{n}_lodging_included_no_manual_keep_50_50.csv", index=False, encoding="utf-8-sig")
    l.to_csv(args.out_dir / f"late_quadrants_2차{n}_lodging_included_no_manual_keep_50_50.csv", index=False, encoding="utf-8-sig")
    e[e["evening_candidate_50"]].to_csv(
        args.out_dir / f"evening_candidates_2차{n}_lodging_included_no_manual_keep_50_50.csv",
        index=False,
        encoding="utf-8-sig",
    )
    l[l["late_candidate_50"]].to_csv(
        args.out_dir / f"late_candidates_2차{n}_lodging_included_no_manual_keep_50_50.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_report(args.out_dir, meta, qmeta, decision, e, l)

    print("META", meta)
    print("QMETA", qmeta)
    print("저장 폴더:", args.out_dir)


if __name__ == "__main__":
    main()
