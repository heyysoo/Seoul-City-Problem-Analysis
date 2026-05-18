"""인프라 공백 지수 (Y축, 심야 트랙 메인)

처리 목적
- 심야 트랙 2*2 매트릭스의 메인 Y축 변수 infrastructure_gap_3var 산출.
- 도메인 의미: "야간 안전 인프라가 얼마나 부족한가" (공급 측 공백 측정).
- 범죄는 합산 공식에서 분리 → 매트릭스에서 별도 색상 레이어로 병기 (WHAT/WHEN 분리).

산식
- infrastructure_gap_3var = 0.45 * cctv_void + 0.30 * police_void + 0.25 * light_void

가중치 학술 근거 (Welsh-Farrington 메타분석 효과크기 순서)
- CCTV (0.45): Welsh & Farrington (2009), Piza et al. (2019) — 범죄 16% 감소
- 파출소 접근성 (0.30): Braga et al. (2019), Turchan & Braga (2024) — Cohen's d = 0.132
- 보안등 (0.25): Welsh, Farrington & Douglas (2022) — 범죄 14% 감소
- 7:5:4 비율, 합계 100% 재정규화 (극단값 회피)
- OECD/JRC (2008) BAL(Budget Allocation Process) 표준 부합

범죄 변수 분리 근거 (메인 합산 공식에서 제외)
- 측정 층위 상이: 인프라(공급 측) vs 범죄(결과 측) → 합산 시 개념 혼란
- 데이터 해상도 한계: 자치구 단위 분위수를 동에 동일 부여 → 구 내 동 간 차이 반영 불가
- 팀 결정: 공백 지수/ 범죄 레이어 분리

입력 (CSV 3종, 컬럼명은 --col-* 인자로 조정 가능)
- cctv:    cctv_void_dong.csv (cctv_void 컬럼)
- police:  police_distance_dong.csv (police_void 컬럼)
- light:   streetlight_void_dong.csv (light_void 컬럼)
- crime (선택): crime_percentile_dong.csv (crime_percentile 컬럼)

각 입력은 행자부 8자리 행정동 코드 컬럼 + 변수값 컬럼으로 구성.
정규화는 이미 적용된 [0, 1] 값으로 가정 (높을수록 공백 큼).

산출물
- infrastructure_gap_final.csv: 마스터 (dong_code + 3 변수 + crime_percentile + infrastructure_gap_3var + rank)
- infra_gap_robustness.csv (선택, --run-robustness 시): 4-Method 강건성 검증
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 상수
# -----------------------------------------------------------------------------

# v5 정통 가중치 (CCTV · 파출소 · 보안등, 7:5:4 비율 → 합계 100%)
WEIGHTS_MAIN = {"cctv": 0.45, "police": 0.30, "light": 0.25}

KEY_COL_DEFAULT = "ADM_CD"


# -----------------------------------------------------------------------------
# 입력 로딩
# -----------------------------------------------------------------------------

def load_variable(
    path: Path,
    key_col: str,
    value_col: str,
    out_key: str,
    out_name: str,
) -> pd.DataFrame:
    """단일 변수 csv 로드 → 공통 키 컬럼명 + 표준 변수명으로 정리."""
    df = pd.read_csv(path, dtype={key_col: str})
    if key_col not in df.columns:
        raise KeyError(
            f"{path.name}에 key 컬럼 '{key_col}' 없음. 현재 컬럼: {df.columns.tolist()}"
        )
    if value_col not in df.columns:
        raise KeyError(
            f"{path.name}에 value 컬럼 '{value_col}' 없음. 현재 컬럼: {df.columns.tolist()}"
        )
    out = df[[key_col, value_col]].copy()
    out = out.rename(columns={key_col: out_key, value_col: out_name})
    out[out_key] = out[out_key].astype(str).str.strip()
    return out


# -----------------------------------------------------------------------------
# 핵심: 가중 합산
# -----------------------------------------------------------------------------

def compute_gap_3var(
    cctv: pd.DataFrame,
    police: pd.DataFrame,
    light: pd.DataFrame,
    key_col: str,
    weights: dict,
) -> pd.DataFrame:
    """3 변수 outer-merge 후 가중 합산.

    outer-merge 선택 이유: 인프라 변수별로 서울 426동 커버하나, 일부 결측 가능 →
    결측은 NaN 유지하여 마스터 단계에서 처리 (Grey out 등).
    """
    merged = cctv.merge(police, on=key_col, how="outer")
    merged = merged.merge(light, on=key_col, how="outer")

    print(f"  병합 결과: {len(merged)} 행 (CCTV {len(cctv)} / police {len(police)} / light {len(light)})")
    missing = merged[["cctv_void", "police_void", "light_void"]].isna().sum()
    if missing.sum() > 0:
        print(f"  결측: {missing.to_dict()}")

    # 가중 합산 (결측은 NaN 유지)
    merged["infrastructure_gap_3var"] = (
        weights["cctv"] * merged["cctv_void"]
        + weights["police"] * merged["police_void"]
        + weights["light"] * merged["light_void"]
    )

    return merged.sort_values("infrastructure_gap_3var", ascending=False).reset_index(drop=True)


# -----------------------------------------------------------------------------
# 강건성 검증 (4-Method) — 선택 산출
# -----------------------------------------------------------------------------

def weights_equal() -> dict:
    return {"cctv": 1 / 3, "police": 1 / 3, "light": 1 / 3}


def weights_entropy(df: pd.DataFrame, cols: list[str]) -> dict:
    """Entropy 기반 가중치 (Shannon, m = 동수).

    노션 §4 기준 검증: police_void 분포 특성으로 Entropy가 파출소에 치우침 (메모리상 ~0.74).
    """
    m = len(df)
    if m == 0:
        return weights_equal()

    diversities = []
    for col in cols:
        x = df[col].to_numpy(dtype=float)
        x_pos = x - np.nanmin(x) + 1e-12
        p = x_pos / np.nansum(x_pos)
        entropy = -np.nansum(p * np.log(p + 1e-12)) / np.log(m)
        diversities.append(1.0 - entropy)

    total = sum(diversities)
    if total <= 0:
        return weights_equal()

    name_map = {"cctv_void": "cctv", "police_void": "police", "light_void": "light"}
    return {name_map[col]: d / total for col, d in zip(cols, diversities)}


def weights_pca(df: pd.DataFrame, cols: list[str]) -> tuple[dict, float]:
    """PCA PC1 적재량 절댓값 비례 가중치 + PC1 설명 분산.

    노션 §4.2 기준: PC1 설명 분산 49.5% (50% 기준 근소 미달).
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise ImportError("PCA 검증을 위해 scikit-learn 설치 필요: pip install scikit-learn") from exc

    valid = df[cols].dropna()
    X = valid.to_numpy(dtype=float)
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    pca = PCA(n_components=3)
    pca.fit(X_std)

    loadings = np.abs(pca.components_[0])
    explained_var_pc1 = float(pca.explained_variance_ratio_[0])
    total = loadings.sum()

    name_map = {"cctv_void": "cctv", "police_void": "police", "light_void": "light"}
    weights = {name_map[col]: float(loadings[i] / total) for i, col in enumerate(cols)}
    return weights, explained_var_pc1


def verify_robustness(df: pd.DataFrame, main_weights: dict) -> pd.DataFrame:
    """Entropy/PCA/Equal 가중치로 재산출 + 메인과 Spearman ρ 비교."""
    from scipy.stats import spearmanr

    cols = ["cctv_void", "police_void", "light_void"]
    pca_weights, pca_var = weights_pca(df, cols)

    methods = {
        "main_45_30_25": main_weights,
        "entropy": weights_entropy(df, cols),
        "pca": pca_weights,
        "equal": weights_equal(),
    }

    valid = df.dropna(subset=cols).copy()
    scores = {}
    for name, w in methods.items():
        scores[name] = (
            w["cctv"] * valid["cctv_void"]
            + w["police"] * valid["police_void"]
            + w["light"] * valid["light_void"]
        )

    main_score = scores["main_45_30_25"]
    rho_rows = []
    for name, score in scores.items():
        rho, _ = spearmanr(main_score, score)
        w = methods[name]
        rho_rows.append({
            "method": name,
            "weight_cctv": w["cctv"],
            "weight_police": w["police"],
            "weight_light": w["light"],
            "spearman_rho_vs_main": rho,
            "pca_pc1_variance": pca_var if name == "pca" else None,
        })
    return pd.DataFrame(rho_rows)


# -----------------------------------------------------------------------------
# 선택: 범죄 레이어 결합 (메인 산식 외)
# -----------------------------------------------------------------------------

def attach_crime_percentile(
    gap_df: pd.DataFrame,
    crime_path: Path,
    crime_key_col: str,
    crime_value_col: str,
    target_key: str,
) -> pd.DataFrame:
    """범죄 분위수 컬럼을 결합 (보조 레이어, 합산 공식에는 영향 없음)."""
    crime = pd.read_csv(crime_path, dtype={crime_key_col: str})
    crime = crime[[crime_key_col, crime_value_col]].rename(
        columns={crime_key_col: target_key, crime_value_col: "crime_percentile"}
    )
    crime[target_key] = crime[target_key].astype(str).str.strip()
    return gap_df.merge(crime, on=target_key, how="left")


# -----------------------------------------------------------------------------
# 진입점
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="인프라 공백 지수 산출 (심야 트랙 메인 Y축)")

    # 공통 키 컬럼
    parser.add_argument("--key-col", type=str, default=KEY_COL_DEFAULT, help="행정동 코드 컬럼명 (3개 입력 csv 공통)")

    # 입력 3종 (인프라)
    parser.add_argument(
        "--cctv-csv",
        type=Path,
        default=Path("data/processed/infra_inputs/cctv_void_dong.csv"),
    )
    parser.add_argument("--col-cctv", type=str, default="cctv_void")

    parser.add_argument(
        "--police-csv",
        type=Path,
        default=Path("data/processed/infra_inputs/police_distance_dong.csv"),
    )
    parser.add_argument("--col-police", type=str, default="police_void")

    parser.add_argument(
        "--light-csv",
        type=Path,
        default=Path("data/processed/infra_inputs/streetlight_void_dong.csv"),
    )
    parser.add_argument("--col-light", type=str, default="light_void")

    # 선택: 범죄 레이어 (보조)
    parser.add_argument(
        "--crime-csv",
        type=Path,
        default=Path("data/processed/crime_context/seoul_dong_crime_context_2023_2024.csv"),
        help="범죄 분위수 csv (crime_context.py 산출, 보조 레이어 — 합산 공식 미영향)",
    )
    parser.add_argument("--crime-key-col", type=str, default="행정동코드")
    parser.add_argument("--col-crime", type=str, default="범죄맥락분위수_권장")
    parser.add_argument("--skip-crime", action="store_true", help="범죄 레이어 결합 건너뛰기")

    # 출력
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/infra_gap"),
    )
    parser.add_argument("--run-robustness", action="store_true", help="4-Method 강건성 검증 실행")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] 인프라 변수 3종 로드")
    cctv = load_variable(args.cctv_csv, args.key_col, args.col_cctv, args.key_col, "cctv_void")
    police = load_variable(args.police_csv, args.key_col, args.col_police, args.key_col, "police_void")
    light = load_variable(args.light_csv, args.key_col, args.col_light, args.key_col, "light_void")

    print("\n[2] 가중 합산 (0.45 / 0.30 / 0.25)")
    gap = compute_gap_3var(cctv, police, light, args.key_col, WEIGHTS_MAIN)

    if not args.skip_crime and args.crime_csv.exists():
        print("\n[3] 범죄 분위수 결합 (보조 레이어)")
        gap = attach_crime_percentile(
            gap,
            args.crime_csv,
            crime_key_col=args.crime_key_col,
            crime_value_col=args.col_crime,
            target_key=args.key_col,
        )
    else:
        print("\n[3] 범죄 레이어 결합 생략")
        gap["crime_percentile"] = None

    # 순위 부여
    gap["rank_3var"] = gap["infrastructure_gap_3var"].rank(ascending=False, method="min").astype("Int64")

    final_path = args.out_dir / "infrastructure_gap_final.csv"
    gap.to_csv(final_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {final_path} ({len(gap)} 행)")

    if args.run_robustness:
        print("\n[4] 4-Method 강건성 검증")
        rob = verify_robustness(gap, WEIGHTS_MAIN)
        rob_path = args.out_dir / "infra_gap_robustness.csv"
        rob.to_csv(rob_path, index=False, encoding="utf-8-sig")
        print(f"  저장: {rob_path}")
        print(rob.to_string(index=False))


if __name__ == "__main__":
    # 실행 예시 (필요 시 주석 해제):
    # import sys
    # sys.argv = [
    #     "infra_gap.py",
    #     "--cctv-csv", "data/processed/infra_inputs/cctv_void_dong.csv",
    #     "--police-csv", "data/processed/infra_inputs/police_distance_dong.csv",
    #     "--light-csv", "data/processed/infra_inputs/streetlight_void_dong.csv",
    #     "--crime-csv", "data/processed/crime_context/crime_percentile_dong.csv",
    #     "--out-dir", "data/processed/infra_gap",
    #     "--run-robustness",
    # ]
    # main()
    pass
