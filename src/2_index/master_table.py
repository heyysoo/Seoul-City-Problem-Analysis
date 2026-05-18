"""마스터 테이블 JOIN — 5개 레이어 통합 + 4종 결측 플래그

처리 목적
- 잠재력 지수 + 인프라 공백 지수 + 범죄 분위수 + 매출 미실현도 + 저녁 매출 5개 레이어 통합.
- 행자부 코드 기준 LEFT JOIN, 결측은 중앙값 대체 + is_*_imputed 플래그로 추적.
- 마스터 노션 §3 (NEW_TO_OLD_MAP + 4종 결측 플래그)와 §6.6 (Grey out 정책) 정통.

기준 (JOIN baseline)
- 인프라 base 426동을 기준으로 LEFT JOIN
- 각 변수의 행자부 코드 컬럼으로 매칭

JOIN 흐름
1. infra_gap.py 산출물 (infrastructure_gap_final.csv) — 426동
2. potential.py 산출물 (potential_evening.csv, potential_late.csv) ← 잠재력 지수
3. gap_index.py 산출물 (b040_b079_gap_final.csv) ← 매출 미실현도
4. evening_sales.py 산출물 (evening_sales_final.csv) ← 저녁 매출 절대값
5. crime_context.py 산출물 (crime_percentile.csv) ← 범죄 분위수 (자치구 단위 broadcast)

NEW_TO_OLD_MAP (잠재력 신코드 → gap.csv 구코드)
- 강북 6쌍: 11305595/11305603/11305608/11305615/11305625/11305635
  → 11305590/11305600/11305606/11305610/11305620/11305630
- 잠재력 지수는 행정안전부 신코드 (행자부 최신 개편 반영)
- gap.csv는 행정안전부 구코드 (B040·B079 처리 시점 코드)
- 마스터 JOIN 시 신 → 구 매핑 후 결합

4종 결측 플래그
- is_gap_imputed: gap_norm 결측 (7동: 항동·개포3동·가락1동·위례동·둔촌1동·상일1·2동)
- is_evening_imputed: evening_sales_norm 결측 (1동: 개포3동)
- is_potential_imputed: 잠재력 지수 결측 (잠재력 입력 변수 모두 NaN)
- is_crime_imputed: 범죄 분위수 결측 (자치구 매핑 실패)

결측 처리 정책
- 단계 1: LEFT JOIN 시 NaN 발생
- 단계 2: 각 변수의 전체 중앙값으로 대체
- 단계 3: is_*_imputed=True 플래그 부여
- 단계 4: 시각화에서 Grey out (회색 X 마커 / 빗금 패치)

산출
- master_table_evening.csv: 저녁 트랙 마스터 (잠재력 저녁 + 저녁 매출 + 인프라 + 범죄 + gap)
- master_table_late.csv: 심야 트랙 마스터 (잠재력 심야 + 인프라 + 범죄 + gap)
- master_table_full.csv: 전체 통합 (양 트랙 + 모든 보조 변수, 검증·시각화용)

컬럼 구조 (master_table_full.csv, 27개)
- 식별자 (4): amd_code, amd_name, gu_name, hanb_gu_code
- 정규화 입력 (4): growth_norm, opening_norm, transit_norm, gap_norm
- 잠재력 (5): potential_evening, potential_late, momentum_equal, momentum_entropy, momentum_pca
- 인프라 (6): cctv_void, police_void, light_void, police_dist_m, safety_vulnerability, crime_percentile
- 저녁 매출 (1): evening_sales_norm
- 플래그 (4): is_gap_imputed, is_evening_imputed, is_potential_imputed, is_crime_imputed
- 매핑 추적 (3): matching_method_gap, matching_method_transit, matching_method_evening
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 상수
# -----------------------------------------------------------------------------

# 잠재력 신코드 → gap.csv 구코드 (노션 M5 §2)
# 잠재력 산출 시점에 행자부 신코드를 사용하고, gap_index는 구코드를 사용해서 차이 발생.
NEW_TO_OLD_MAP = {
    "11305595": "11305590",  # 번1동
    "11305603": "11305600",  # 번2동
    "11305608": "11305606",  # 번3동
    "11305615": "11305610",  # 수유1동
    "11305625": "11305620",  # 수유2동
    "11305635": "11305630",  # 수유3동
}

# 키 컬럼 표준 이름
KEY_COL = "amd_code"


# -----------------------------------------------------------------------------
# 입력 로딩 (5개 레이어)
# -----------------------------------------------------------------------------

def load_layer(
    path: Path,
    key_col: str,
    rename_to: str | None = None,
    encoding: str = "utf-8-sig",
) -> pd.DataFrame:
    """단일 레이어 csv 로드 + 키 컬럼명 표준화."""
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음")
    df = pd.read_csv(path, encoding=encoding, dtype={key_col: str})
    df[key_col] = df[key_col].astype(str).str.zfill(8)
    if rename_to and rename_to != key_col:
        df = df.rename(columns={key_col: rename_to})
    print(f"  로드 {path.name}: {len(df)}행")
    return df


# -----------------------------------------------------------------------------
# Step 1 — 잠재력 신코드 → 구코드 보정 (gap JOIN 사전 작업)
# -----------------------------------------------------------------------------

def apply_new_to_old_mapping(
    potential_df: pd.DataFrame,
    key_col: str = KEY_COL,
) -> pd.DataFrame:
    """잠재력 코드의 신코드 → 구코드 변환 (gap.csv와 JOIN하기 전).

    강북 6쌍은 잠재력 산출 시점에 새로운 행자부 코드(개편 후)를 사용했지만,
    gap.csv는 옛 행자부 코드(개편 전)를 사용해서 코드 보정 필요.
    """
    out = potential_df.copy()
    out["amd_code_for_join"] = out[key_col].replace(NEW_TO_OLD_MAP)
    n_remapped = (out[key_col] != out["amd_code_for_join"]).sum()
    print(f"  강북 6쌍 신코드→구코드 보정: {n_remapped}동")
    return out


# -----------------------------------------------------------------------------
# Step 2 — 5개 레이어 LEFT JOIN
# -----------------------------------------------------------------------------

def join_layers(
    infra: pd.DataFrame,
    potential_eve: pd.DataFrame,
    potential_lat: pd.DataFrame,
    gap: pd.DataFrame,
    evening_sales: pd.DataFrame,
    crime: pd.DataFrame,
    key_col: str = KEY_COL,
) -> pd.DataFrame:
    """인프라 426동 기준 LEFT JOIN.

    잠재력 신코드는 NEW_TO_OLD_MAP 적용 후 gap과 결합.
    """
    print("\n[JOIN 1/5] 인프라 베이스 (426동)")
    master = infra.copy()
    print(f"  인프라: {len(master)}동")

    print("\n[JOIN 2/5] 잠재력 저녁 트랙")
    eve = apply_new_to_old_mapping(potential_eve, key_col)
    master = master.merge(
        eve[[key_col, "potential_evening"] + (
            ["growth_norm", "opening_norm", "transit_norm"] if "growth_norm" in eve.columns else []
        )],
        on=key_col, how="left",
    )

    print("\n[JOIN 3/5] 잠재력 심야 트랙")
    lat = apply_new_to_old_mapping(potential_lat, key_col)
    join_cols_lat = [key_col, "potential_late"]
    if "growth_norm_late" in lat.columns:
        join_cols_lat.append("growth_norm_late")
    if "transit_norm_late" in lat.columns:
        join_cols_lat.append("transit_norm_late")
    master = master.merge(lat[join_cols_lat], on=key_col, how="left")

    print("\n[JOIN 4/5] 매출 미실현도 (행자부 코드 기준)")
    # gap_index.py 산출은 행자부 코드를 따로 별도 컬럼으로 가짐
    gap_col = "행자부_코드" if "행자부_코드" in gap.columns else key_col
    gap_renamed = gap.rename(columns={gap_col: key_col})
    master = master.merge(
        gap_renamed[[key_col, "gap_norm"]].drop_duplicates(subset=[key_col]),
        on=key_col, how="left",
    )

    print("\n[JOIN 5/5] 저녁 매출 + 범죄 분위수")
    eve_col = "행자부_코드" if "행자부_코드" in evening_sales.columns else key_col
    eve_sales_renamed = evening_sales.rename(columns={eve_col: key_col})
    master = master.merge(
        eve_sales_renamed[[key_col, "evening_sales_norm"]].drop_duplicates(subset=[key_col]),
        on=key_col, how="left",
    )

    # 범죄는 자치구 단위라 gu_name 또는 자치구 코드 기준 broadcast
    if "gu_name" in master.columns and "gu_name" in crime.columns:
        master = master.merge(
            crime[["gu_name", "crime_percentile"]].drop_duplicates(subset=["gu_name"]),
            on="gu_name", how="left",
        )
    elif "hanb_gu_code" in master.columns and "hanb_gu_code" in crime.columns:
        master = master.merge(
            crime[["hanb_gu_code", "crime_percentile"]].drop_duplicates(subset=["hanb_gu_code"]),
            on="hanb_gu_code", how="left",
        )
    else:
        print("  범죄 매핑 키 미발견 — crime_percentile 미결합 (master에 컬럼 없음)")
        master["crime_percentile"] = np.nan

    print(f"\n최종: {len(master)}동 × {len(master.columns)}열")
    return master


# -----------------------------------------------------------------------------
# Step 3 — 결측 처리 + 4종 플래그 (Grey out 정책)
# -----------------------------------------------------------------------------

def apply_imputation_flags(df: pd.DataFrame) -> pd.DataFrame:
    """4종 결측 플래그 부여 + 중앙값 대체 (노션 §6.6 정통)."""
    out = df.copy()

    flag_specs = [
        ("gap_norm", "is_gap_imputed"),
        ("evening_sales_norm", "is_evening_imputed"),
        ("crime_percentile", "is_crime_imputed"),
    ]

    # 잠재력은 두 트랙 별도로 추적
    if "potential_evening" in out.columns:
        out["is_potential_evening_imputed"] = out["potential_evening"].isna().astype(int)
    if "potential_late" in out.columns:
        out["is_potential_late_imputed"] = out["potential_late"].isna().astype(int)

    for col, flag in flag_specs:
        if col not in out.columns:
            continue
        out[flag] = out[col].isna().astype(int)
        n_imputed = out[flag].sum()
        if n_imputed > 0:
            median_val = out[col].median()
            out[col] = out[col].fillna(median_val)
            print(f"  {col} 결측 {n_imputed}동 → 중앙값 {median_val:.4f} 대체")

    # 잠재력 결측도 중앙값 대체
    for pot_col in ["potential_evening", "potential_late"]:
        if pot_col in out.columns and out[pot_col].isna().any():
            median_val = out[pot_col].median()
            n_imputed = out[pot_col].isna().sum()
            out[pot_col] = out[pot_col].fillna(median_val)
            print(f"  {pot_col} 결측 {n_imputed}동 → 중앙값 {median_val:.4f} 대체")

    return out


# -----------------------------------------------------------------------------
# Step 4 — 트랙별 마스터 분리 (저녁·심야)
# -----------------------------------------------------------------------------

def split_tracks(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """저녁·심야 트랙별 컬럼 분리."""
    common_cols = [c for c in [
        KEY_COL, "amd_name", "dong_name", "gu_name", "hanb_gu_code",
        "cctv_void", "police_void", "light_void", "police_dist_m",
        "safety_vulnerability", "crime_percentile",
    ] if c in master.columns]

    evening_cols = common_cols + [c for c in [
        "potential_evening", "evening_sales_norm",
        "is_potential_evening_imputed", "is_evening_imputed", "is_crime_imputed",
    ] if c in master.columns]

    late_cols = common_cols + [c for c in [
        "potential_late", "gap_norm",
        "is_potential_late_imputed", "is_gap_imputed", "is_crime_imputed",
    ] if c in master.columns]

    evening = master[evening_cols].copy()
    late = master[late_cols].copy()
    return evening, late


# -----------------------------------------------------------------------------
# 진입점
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="마스터 테이블 JOIN (5개 레이어 + 4종 플래그)")

    parser.add_argument(
        "--infra-csv", type=Path,
        default=Path("data/processed/infra_gap/infrastructure_gap_final.csv"),
        help="인프라 공백 지수 (베이스, 426동)",
    )
    parser.add_argument(
        "--potential-evening-csv", type=Path,
        default=Path("data/processed/potential/potential_evening.csv"),
        help="잠재력 저녁 트랙",
    )
    parser.add_argument(
        "--potential-late-csv", type=Path,
        default=Path("data/processed/potential/potential_late.csv"),
        help="잠재력 심야 트랙",
    )
    parser.add_argument(
        "--gap-csv", type=Path,
        default=Path("data/processed/gap_index/b040_b079_gap_final.csv"),
        help="매출 미실현도 (행자부 코드 포함)",
    )
    parser.add_argument(
        "--evening-sales-csv", type=Path,
        default=Path("data/processed/evening_sales/evening_sales_final.csv"),
        help="저녁 매출 절대값 (행자부 코드)",
    )
    parser.add_argument(
        "--crime-csv", type=Path,
        default=Path("data/processed/crime_context/seoul_dong_crime_context_2023_2024.csv"),
        help="범죄 분위수 (crime_context.py 산출, 자치구 단위 broadcast)",
    )

    parser.add_argument("--key-col", type=str, default=KEY_COL)
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("data/processed/master"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] 5개 레이어 로드")
    infra = load_layer(args.infra_csv, args.key_col)
    potential_eve = load_layer(args.potential_evening_csv, args.key_col)
    potential_lat = load_layer(args.potential_late_csv, args.key_col)
    gap = load_layer(args.gap_csv, "행자부_코드" if "_gap_final" in str(args.gap_csv) else args.key_col)
    evening_sales = load_layer(
        args.evening_sales_csv,
        "행자부_코드" if "_final" in str(args.evening_sales_csv) else args.key_col,
    )
    crime = load_layer(args.crime_csv, "gu_name" if args.crime_csv.exists() else args.key_col)

    # crime_context.py 출력 컬럼명 → master_table 내부 표준명 변환
    # (crime_context.py는 한글 컬럼 산출, 본 모듈 join_layers는 영문 컬럼 가정)
    # v1 9.1 시간대 결합 방식: 야간_범죄맥락분위수_권장 우선 사용
    # (시간대 결합 없으면 fallback으로 전체 범죄맥락분위수_권장 사용)
    crime_value_col = (
        "야간_범죄맥락분위수_권장" if "야간_범죄맥락분위수_권장" in crime.columns
        else "범죄맥락분위수_권장"
    )
    crime_rename = {
        "구명": "gu_name",
        "행정동코드": args.key_col,
        crime_value_col: "crime_percentile",
    }
    crime = crime.rename(columns={k: v for k, v in crime_rename.items() if k in crime.columns})
    print(f"  crime 입력 컬럼: {crime_value_col} → crime_percentile")
    if "연도" in crime.columns:
        # 2023+2024 평균 산출 (분위수는 연도별로 산출됨)
        crime = (
            crime.groupby(["gu_name"], as_index=False)["crime_percentile"]
            .mean()
        )
        print(f"  crime: 2023+2024 평균 산출 → {len(crime)}구")

    print("\n[2] 5개 레이어 LEFT JOIN (인프라 426동 기준)")
    master = join_layers(infra, potential_eve, potential_lat, gap, evening_sales, crime,
                        key_col=args.key_col)

    print("\n[3] 결측 처리 + 4종 플래그 (Grey out 정책)")
    master = apply_imputation_flags(master)

    out_full = args.out_dir / "master_table_full.csv"
    master.to_csv(out_full, index=False, encoding="utf-8-sig")
    print(f"\n저장 (전체): {out_full} ({len(master)}행 × {len(master.columns)}열)")

    print("\n[4] 트랙별 마스터 분리 (저녁·심야)")
    evening, late = split_tracks(master)

    out_eve = args.out_dir / "master_table_evening.csv"
    evening.to_csv(out_eve, index=False, encoding="utf-8-sig")
    print(f"  저녁 트랙: {out_eve} ({len(evening)}행 × {len(evening.columns)}열)")

    out_lat = args.out_dir / "master_table_late.csv"
    late.to_csv(out_lat, index=False, encoding="utf-8-sig")
    print(f"  심야 트랙: {out_lat} ({len(late)}행 × {len(late.columns)}열)")


if __name__ == "__main__":
    # 실행 예시:
    # import sys
    # sys.argv = [
    #     "master_table.py",
    #     "--infra-csv", "data/processed/infra_gap/infrastructure_gap_final.csv",
    #     "--potential-evening-csv", "data/processed/potential/potential_evening.csv",
    #     "--potential-late-csv", "data/processed/potential/potential_late.csv",
    #     "--gap-csv", "data/processed/gap_index/b040_b079_gap_final.csv",
    #     "--evening-sales-csv", "data/processed/evening_sales/evening_sales_final.csv",
    #     "--crime-csv", "data/processed/crime/crime_percentile.csv",
    #     "--out-dir", "data/processed/master",
    # ]
    # main()
    pass
