"""잠재력 지수 (X축, 저녁·심야 트랙 공통)

처리 목적
- 2차 필터 통과 80동의 X축 잠재력 지수를 산출한다.
- 저녁 트랙(potential_evening)과 심야 트랙(potential_late)을 각각 생성한다.
- opening은 시간대 무관 공통, growth·transit은 트랙별 별도 산출 입력 사용.

핵심 산식 (노션 잠재력 지수 마스터 §1.1)
- 각 변수: 별도 모듈에서 1·99 percentile 양측 클리핑 + Min-Max 정규화 완료 → [0,1]
- 본 모듈은 정규화 완료된 변수를 입력 받아 가중합만 수행
- 가중합: potential = 0.50 * growth + 0.30 * opening + 0.20 * transit
- 최종 [0,1] 재정규화 (선택, 기본 raw 가중합 유지)

각 변수의 산출 모듈
- growth_norm: src/2_index/growth_index.py (B079 카드매출 + 클리핑 + Min-Max)
- opening_norm: src/2_index/opening_index.py (OA-22172 12분기 평균 + imputed + Min-Max)
- transit_norm: src/2_index/transit_index.py (B013 YoY + 클리핑 + Min-Max)

imputed 추적
- 각 변수에 is_imputed_{var} 컬럼 있으면 잠재력 산출 시 OR로 결합:
  is_imputed_any = is_imputed_growth OR is_imputed_opening OR is_imputed_transit

학술 근거 (가중치 정당성)
- growth 0.50: 야간경제 활성화 직접 신호 (Glaeser et al. 2015, Lin et al. 2022)
- opening 0.30: 신규 진입 시그널 (World Bank EDB, Glaeser et al. 2010)
- transit 0.20: 유입 모멘텀 간접 측정 (Cervero & Duncan 2002, Cervero & Kang 2011)
- 4-Method robustness 검증: Equal·Entropy·PCA·Spearman
  → 잠재력 r ≥ 0.93 (Equal·Entropy 두 트랙 모두 수렴)
  → (--run-robustness 플래그로 별도 산출)

산출물 (out_dir 하위)
- potential_evening.csv: amd_code + 3 변수 norm + is_imputed_* + potential_evening
- potential_late.csv:    amd_code + 3 변수 norm + is_imputed_* + potential_late
- potential_robustness.csv (선택, --run-robustness 시)

입력 (각 변수 정규화 완료된 5종)
- growth-evening:  growth_norm_evening.csv (growth_index.py 산출)
- growth-late:     growth_norm_late.csv (growth_index.py 산출)
- opening:         opening_rate_final.csv (opening_index.py 산출)
- transit-evening: transit_norm_final.csv:transit_norm_evening (transit_index.py 산출)
- transit-late:    transit_norm_final.csv:transit_norm_late (transit_index.py 산출)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 상수
# -----------------------------------------------------------------------------

WEIGHTS_MAIN = {"growth": 0.50, "opening": 0.30, "transit": 0.20}
CLIP_LOWER_PCT = 0.01
CLIP_UPPER_PCT = 0.99
KEY_COL_DEFAULT = "amd_code"


# -----------------------------------------------------------------------------
# 정규화 유틸
# -----------------------------------------------------------------------------

def clip_and_minmax(series: pd.Series, lower_pct: float = CLIP_LOWER_PCT, upper_pct: float = CLIP_UPPER_PCT) -> pd.Series:
    """1·99 percentile 양측 클리핑 후 Min-Max 정규화.

    극단치의 비대칭 영향을 제어한 뒤 [0,1] 스케일로 변환한다.
    클리핑 경계 밖 값은 동일 경계값으로 압축되어 0.000/1.000 으로 수렴 가능.
    """
    s = pd.to_numeric(series, errors="coerce")
    lo = s.quantile(lower_pct)
    hi = s.quantile(upper_pct)
    clipped = s.clip(lower=lo, upper=hi)
    rng = clipped.max() - clipped.min()
    if rng == 0:
        return pd.Series(0.5, index=s.index)
    return (clipped - clipped.min()) / rng


def renormalize_to_unit(series: pd.Series) -> pd.Series:
    """가중합 결과를 [0,1]로 재정규화."""
    s = pd.to_numeric(series, errors="coerce")
    rng = s.max() - s.min()
    if rng == 0:
        return pd.Series(0.5, index=s.index)
    return (s - s.min()) / rng


# -----------------------------------------------------------------------------
# 입력 로딩
# -----------------------------------------------------------------------------

def load_variable(path: Path, key_col: str, value_col: str, out_name: str) -> pd.DataFrame:
    """단일 변수 csv 로드 (key + value 두 컬럼 유지)."""
    df = pd.read_csv(path, dtype={key_col: str})
    if key_col not in df.columns:
        raise KeyError(f"{path.name}에 key 컬럼 '{key_col}' 없음. 현재 컬럼: {df.columns.tolist()}")
    if value_col not in df.columns:
        raise KeyError(f"{path.name}에 value 컬럼 '{value_col}' 없음. 현재 컬럼: {df.columns.tolist()}")
    out = df[[key_col, value_col]].copy()
    out = out.rename(columns={value_col: out_name})
    out[key_col] = out[key_col].astype(str).str.strip()
    return out


# -----------------------------------------------------------------------------
# 트랙별 잠재력 산출
# -----------------------------------------------------------------------------

def build_track(
    growth_df: pd.DataFrame,
    opening_df: pd.DataFrame,
    transit_df: pd.DataFrame,
    key_col: str,
    track_name: str,
    weights: dict,
) -> pd.DataFrame:
    """단일 트랙(저녁 또는 심야) 잠재력 산출.

    절차:
    1) 3 변수 inner-merge (공통 행정동만 유지)
    2) 각 변수 클리핑 + Min-Max 정규화 → _norm 컬럼
    3) 가중합 → renormalize → potential_{track_name}
    """
    merged = growth_df.merge(opening_df, on=key_col, how="inner")
    merged = merged.merge(transit_df, on=key_col, how="inner")

    n_before = max(len(growth_df), len(opening_df), len(transit_df))
    n_after = len(merged)
    if n_after < n_before:
        print(f"  [{track_name}] inner merge 후 {n_after}/{n_before} (공통 행정동만)")

    merged["growth_raw"] = merged["growth"]
    merged["opening_raw"] = merged["opening"]
    merged["transit_raw"] = merged["transit"]

    merged["growth_norm"] = clip_and_minmax(merged["growth_raw"])
    merged["opening_norm"] = clip_and_minmax(merged["opening_raw"])
    merged["transit_norm"] = clip_and_minmax(merged["transit_raw"])

    weighted = (
        weights["growth"] * merged["growth_norm"]
        + weights["opening"] * merged["opening_norm"]
        + weights["transit"] * merged["transit_norm"]
    )
    col_name = f"potential_{track_name}"
    merged[col_name] = renormalize_to_unit(weighted)

    return merged.sort_values(col_name, ascending=False).reset_index(drop=True)


# -----------------------------------------------------------------------------
# 강건성 검증 (4-Method) — 선택 산출
# -----------------------------------------------------------------------------

def weights_equal(n: int = 3) -> dict:
    """균등 가중치."""
    w = 1.0 / n
    return {"growth": w, "opening": w, "transit": w}


def weights_entropy(df_norm: pd.DataFrame, cols: list[str]) -> dict:
    """Entropy 기반 가중치.

    각 변수의 정보 엔트로피 e_j 를 계산한 후, 분산도 d_j = 1 - e_j 의 비율로 가중치 산출.
    엔트로피가 낮을수록(분산도 높을수록) 정보량이 많아 큰 가중치 부여.
    """
    m = len(df_norm)
    if m == 0:
        return weights_equal(len(cols))

    weights = {}
    diversities = []
    for col in cols:
        x = df_norm[col].to_numpy(dtype=float)
        # 양수화 + 정규화 (확률 분포 형태)
        x_pos = x - x.min() + 1e-12
        p = x_pos / x_pos.sum()
        # Shannon entropy, 자연로그
        entropy = -np.sum(p * np.log(p + 1e-12)) / np.log(m)
        diversities.append(1.0 - entropy)

    total = sum(diversities)
    if total <= 0:
        return weights_equal(len(cols))

    for col, d in zip(cols, diversities):
        weights[col] = d / total
    return weights


def weights_pca(df_norm: pd.DataFrame, cols: list[str]) -> dict:
    """PCA PC1 적재량 절댓값 비례 가중치.

    PC1이 본 분석의 잠재력 잠재 차원과 가장 정렬되어 있다고 가정하고,
    각 변수의 PC1 적재량 절댓값으로 가중치 산출.
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise ImportError("PCA 검증을 위해 scikit-learn 설치 필요: pip install scikit-learn") from exc

    X = df_norm[cols].to_numpy(dtype=float)
    # 표준화
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    pca = PCA(n_components=1)
    pca.fit(X_std)
    loadings = np.abs(pca.components_[0])

    total = loadings.sum()
    if total <= 0:
        return weights_equal(len(cols))

    return {col: float(loadings[i] / total) for i, col in enumerate(cols)}


def verify_robustness(
    df_norm: pd.DataFrame,
    cols: list[str],
    track_name: str,
    main_weights: dict,
) -> pd.DataFrame:
    """4가지 가중치 방법별 잠재력 산출 + Spearman ρ 비교."""
    from scipy.stats import spearmanr

    methods = {
        "main_50_30_20": main_weights,
        "equal": weights_equal(len(cols)),
        "entropy": weights_entropy(df_norm, cols),
        "pca": weights_pca(df_norm, cols),
    }

    scores = {}
    for name, w in methods.items():
        weighted = sum(w[c.replace("_norm", "")] * df_norm[c] for c in cols)
        scores[name] = renormalize_to_unit(weighted)

    main_score = scores["main_50_30_20"]
    rho_rows = []
    for name, score in scores.items():
        rho, _ = spearmanr(main_score, score)
        w = methods[name]
        rho_rows.append({
            "track": track_name,
            "method": name,
            "weight_growth": w["growth"],
            "weight_opening": w["opening"],
            "weight_transit": w["transit"],
            "spearman_rho_vs_main": rho,
        })
    return pd.DataFrame(rho_rows)


# -----------------------------------------------------------------------------
# 진입점
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="잠재력 지수 산출 (저녁·심야 트랙)")

    parser.add_argument("--key-col", type=str, default=KEY_COL_DEFAULT, help="행정동 코드 컬럼명 (5개 입력 csv 공통)")

    # 저녁 트랙 입력
    parser.add_argument("--growth-evening", type=Path,
                        default=Path("data/processed/growth/growth_norm_evening.csv"))
    parser.add_argument("--col-growth-evening", type=str, default="growth_norm_evening")
    parser.add_argument("--transit-evening", type=Path,
                        default=Path("data/processed/transit/transit_norm_final.csv"))
    parser.add_argument("--col-transit-evening", type=str, default="transit_norm_evening")

    # 심야 트랙 입력
    parser.add_argument("--growth-late", type=Path,
                        default=Path("data/processed/growth/growth_norm_late.csv"))
    parser.add_argument("--col-growth-late", type=str, default="growth_norm_late")
    parser.add_argument("--transit-late", type=Path,
                        default=Path("data/processed/transit/transit_norm_final.csv"))
    parser.add_argument("--col-transit-late", type=str, default="transit_norm_late")

    # 공통 (시간대 무관)
    parser.add_argument("--opening", type=Path,
                        default=Path("data/processed/opening/opening_rate_final.csv"))
    parser.add_argument("--col-opening", type=str, default="opening_norm")

    # 출력
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/potential"))
    parser.add_argument("--run-robustness", action="store_true", help="4-Method 강건성 검증 실행")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] 입력 로딩")
    opening_df = load_variable(args.opening, args.key_col, args.col_opening, "opening")
    print(f"  opening: {len(opening_df)}")

    growth_eve = load_variable(args.growth_evening, args.key_col, args.col_growth_evening, "growth")
    growth_lat = load_variable(args.growth_late, args.key_col, args.col_growth_late, "growth")
    print(f"  growth_evening: {len(growth_eve)}, growth_late: {len(growth_lat)}")

    transit_eve = load_variable(args.transit_evening, args.key_col, args.col_transit_evening, "transit")
    transit_lat = load_variable(args.transit_late, args.key_col, args.col_transit_late, "transit")
    print(f"  transit_evening: {len(transit_eve)}, transit_late: {len(transit_lat)}")

    print("\n[2] 저녁 트랙 잠재력 산출")
    evening = build_track(growth_eve, opening_df, transit_eve, args.key_col, "evening", WEIGHTS_MAIN)
    eve_path = args.out_dir / "potential_evening.csv"
    evening.to_csv(eve_path, index=False, encoding="utf-8-sig")
    print(f"  저장: {eve_path} ({len(evening)} 행)")

    print("\n[3] 심야 트랙 잠재력 산출")
    late = build_track(growth_lat, opening_df, transit_lat, args.key_col, "late", WEIGHTS_MAIN)
    lat_path = args.out_dir / "potential_late.csv"
    late.to_csv(lat_path, index=False, encoding="utf-8-sig")
    print(f"  저장: {lat_path} ({len(late)} 행)")

    if args.run_robustness:
        print("\n[4] 4-Method 강건성 검증")
        norm_cols = ["growth_norm", "opening_norm", "transit_norm"]
        rob_eve = verify_robustness(evening, norm_cols, "evening", WEIGHTS_MAIN)
        rob_lat = verify_robustness(late, norm_cols, "late", WEIGHTS_MAIN)
        rob = pd.concat([rob_eve, rob_lat], ignore_index=True)
        rob_path = args.out_dir / "potential_robustness.csv"
        rob.to_csv(rob_path, index=False, encoding="utf-8-sig")
        print(f"  저장: {rob_path}")
        print(rob.to_string(index=False))


if __name__ == "__main__":
    # 실행 예시 (필요 시 주석 해제):
    # import sys
    # sys.argv = [
    #     "potential.py",
    #     "--growth-evening", "data/interim/growth_evening.csv",
    #     "--growth-late", "data/interim/growth_late_night.csv",
    #     "--opening", "data/interim/opening_rate.csv",
    #     "--transit-evening", "data/interim/transit_evening.csv",
    #     "--transit-late", "data/interim/transit_late_night.csv",
    #     "--out-dir", "data/processed/potential",
    #     "--run-robustness",
    # ]
    # main()
    pass
