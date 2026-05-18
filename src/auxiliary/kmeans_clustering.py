"""K-means 클러스터링 — 행정동 야간 활동 패턴 4유형 분류 (LEGACY)

메인 파이프라인 미사용
- 본 모듈은 전에 만든 코드.
- 4단계 진단 시스템(거주지 필터 → 상권성 필터 → 50/50 매트릭스 → 정합성 점검)으로
  분석 트랙이 변경되며 K-means는 본 분석에서 제외.
- 코드는 보고서 기록으로 남기기 위해 보존.

분류 결과 (옛 검증)
- A_주간우세형 (Cluster 0): 86동, 야간/주간 = 0.97
- B_저녁집중형 (Cluster 1): 79동, 야간/주간 = 1.04
- C_심야활성형 (Cluster 2): 81동, 야간/주간 = 2.06
- D_주간중심형: 야간특화지수 < 1.0 사전 제외 175동
- 이상치 3동 사전 제거: 11710631, 11740520, 11710647

비율 정규화 결정 (핵심)
- StandardScaler 단독 사용 시 인구 규모 기반 클러스터링 → 크기 기반
- 행정동 프로파일을 시간대별 비율로 변환 후 StandardScaler 적용 → 패턴 기반
- 결과: 인구 규모가 아닌 시간대 활동 패턴이 클러스터 기준

입력
- b040_24h_profile.csv: 행정동 * 24시간 * 주중/주말 = 48차원 프로파일 (캠퍼스 internal)
- dong_night_index_final.csv: 야간특화지수 (D유형 사전제외 필터)

산출 (out_dir 하위)
- dong_cluster_label_k3.csv (반출 가능, 응용집계 범주):
  컬럼 dong_code, cluster (0/1/2/-1), cluster_name (A/B/C/D)
- elbow_silhouette.png: K=3~6 비교
- cluster_profiles_k3.png: 3 클러스터 24시간 프로파일
- kmeans_validation_summary.csv: Silhouette + Davies-Bouldin
- kmeans_validation_kruskal.csv: 시간대별 Kruskal-Wallis 검정

캠퍼스 반출 정책
- cluster_label은 범주형 (응용집계) → 반출 가능
- 24h_profile은 단순 평균 → 반출 불가 (캠퍼스 internal)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 상수
# -----------------------------------------------------------------------------

# K-means 사전 제외 이상치 3동 (옛 대화 정통)
KMEANS_OUTLIER_DONGS = frozenset({"11710631", "11740520", "11710647"})

# D유형 사전 제외 컷오프 (야간특화지수 < 1.0)
NIGHT_SPECIAL_CUTOFF = 1.0

# K-means 파라미터
DEFAULT_K = 3
RANDOM_STATE = 42

# 클러스터 라벨 매핑 (Silhouette + 도메인 해석 후 부여)
CLUSTER_LABELS = {
    0: "A_주간우세형",
    1: "B_저녁집중형",
    2: "C_심야활성형",
    -1: "D_주간중심형",  # 사전 제외
}


# -----------------------------------------------------------------------------
# Step 1 — 입력 로딩 + D유형 사전 제외
# -----------------------------------------------------------------------------

def load_profile(
    profile_csv: Path,
    night_index_csv: Path,
    dong_col: str = "dong_code",
    encoding: str = "utf-8-sig",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """24시간 × 주중/주말 프로파일 + 야간특화지수 로딩."""
    profile = pd.read_csv(profile_csv, encoding=encoding, dtype={dong_col: str})
    night_idx = pd.read_csv(night_index_csv, encoding=encoding, dtype={dong_col: str})

    profile[dong_col] = profile[dong_col].astype(str).str.zfill(8)
    night_idx[dong_col] = night_idx[dong_col].astype(str).str.zfill(8)

    print(f"  프로파일: {len(profile)}동 × {profile.shape[1] - 1}열")
    print(f"  야간특화지수: {len(night_idx)}동")
    return profile, night_idx


def filter_kmeans_input(
    profile: pd.DataFrame,
    night_idx: pd.DataFrame,
    dong_col: str = "dong_code",
    night_col: str = "night_special_index",
) -> pd.DataFrame:
    """K-means 입력 필터링 (D유형 175동 + 이상치 3동 동시 제외).

    노션 정통: K=3 입력 246동 = 전체 424 - D유형 175 - 이상치 3.
    """
    merged = profile.merge(
        night_idx[[dong_col, night_col]],
        on=dong_col, how="inner",
    )

    before = len(merged)

    # D유형 사전 제외 (야간특화지수 < 1.0)
    d_mask = merged[night_col] < NIGHT_SPECIAL_CUTOFF
    n_d = d_mask.sum()

    # 이상치 동 사전 제외 (11710631·11740520·11710647)
    outlier_mask = merged[dong_col].isin(KMEANS_OUTLIER_DONGS)
    n_outlier = outlier_mask.sum()

    # 동시 제외
    keep_mask = ~(d_mask | outlier_mask)
    filtered = merged[keep_mask].copy()

    print(f"  K-means 사전 필터:")
    print(f"    전체: {before}동")
    print(f"    - D유형 (야간특화지수 < 1.0): {n_d}동")
    print(f"    - 이상치 3동: {n_outlier}동")
    print(f"    K-means 입력: {len(filtered)}동")
    return filtered


# -----------------------------------------------------------------------------
# Step 2 — 비율 정규화 + StandardScaler (옛 정통, 패턴 기반 클러스터링)
# -----------------------------------------------------------------------------

def normalize_to_ratio(
    df: pd.DataFrame,
    dong_col: str,
    feature_cols: list[str],
) -> np.ndarray:
    """행정동별 시간대 비율 정규화 → StandardScaler.

    핵심 결정 (옛 정통):
    - StandardScaler 단독 사용 시 인구 규모 기반 클러스터링 발생
    - 행정동 프로파일을 시간대별 비율로 변환 (각 동 합 = 1)
    - 이후 StandardScaler 적용 → 패턴 기반 클러스터링
    """
    try:
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("K-means 검증을 위해 scikit-learn 설치 필요") from exc

    X = df[feature_cols].to_numpy(dtype=float)

    # 1) 비율 정규화 (각 행 합 = 1)
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    X_ratio = X / row_sums

    # 2) StandardScaler (평균 0, 분산 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_ratio)
    return X_scaled


# -----------------------------------------------------------------------------
# Step 3 — K 결정 (Elbow + Silhouette)
# -----------------------------------------------------------------------------

def evaluate_k_range(
    X_scaled: np.ndarray,
    k_min: int = 3,
    k_max: int = 6,
) -> pd.DataFrame:
    """K=3~6 범위 Silhouette + Davies-Bouldin 산출."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, davies_bouldin_score
    except ImportError as exc:
        raise ImportError("scikit-learn 필요") from exc

    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        rows.append({"k": k, "silhouette": sil, "davies_bouldin": db,
                     "inertia": km.inertia_})
        print(f"  K={k}: Silhouette={sil:.4f}, DB={db:.4f}, Inertia={km.inertia_:.1f}")
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Step 4 — K-means + Kruskal-Wallis 검증
# -----------------------------------------------------------------------------

def fit_kmeans(
    X_scaled: np.ndarray,
    k: int = DEFAULT_K,
) -> tuple[np.ndarray, "KMeans"]:
    """K-means 학습 → 클러스터 라벨 반환."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    return labels, km


def kruskal_wallis_per_hour(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list[str],
) -> pd.DataFrame:
    """시간대별 클러스터 차이 Kruskal-Wallis 검정."""
    from scipy.stats import kruskal

    rows = []
    df_with_label = df.copy()
    df_with_label["cluster"] = labels

    for col in feature_cols:
        groups = [
            df_with_label.loc[df_with_label["cluster"] == c, col].to_numpy()
            for c in sorted(df_with_label["cluster"].unique())
        ]
        h_stat, p_val = kruskal(*groups)
        rows.append({"feature": col, "h_statistic": h_stat, "p_value": p_val})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Step 5 — 라벨 부여 (D유형 합산 + 최종 csv 산출)
# -----------------------------------------------------------------------------

def assign_labels_and_merge_d(
    kmeans_df: pd.DataFrame,
    night_idx: pd.DataFrame,
    dong_col: str = "dong_code",
    night_col: str = "night_special_index",
    cluster_col: str = "cluster",
) -> pd.DataFrame:
    """K-means 결과에 D유형 사전 제외 동 합산 + 라벨 부여.

    Args:
        kmeans_df: K-means 학습 대상 (B+C+추가 A?) 동 + cluster 컬럼
        night_idx: 전체 행정동 야간특화지수 (D유형 식별용)
    """
    # 전체 행정동 베이스
    full = night_idx[[dong_col, night_col]].copy()
    full[dong_col] = full[dong_col].astype(str).str.zfill(8)

    # K-means 결과 결합
    full = full.merge(
        kmeans_df[[dong_col, cluster_col]],
        on=dong_col, how="left",
    )

    # D유형 식별 (cluster NaN + 야간특화지수 < 1.0)
    d_mask = full[cluster_col].isna() & (full[night_col] < NIGHT_SPECIAL_CUTOFF)
    full.loc[d_mask, cluster_col] = -1

    # 이상치 식별 (cluster NaN + dong_code 가 이상치 목록)
    outlier_mask = full[cluster_col].isna() & full[dong_col].isin(KMEANS_OUTLIER_DONGS)
    full.loc[outlier_mask, cluster_col] = 99  # X_이상치 별도 코드

    # 라벨 부여
    full["cluster"] = full[cluster_col].astype("Int64")
    full["cluster_name"] = full["cluster"].map(CLUSTER_LABELS).fillna("Unassigned")
    return full[[dong_col, "cluster", "cluster_name", night_col]]


# -----------------------------------------------------------------------------
# 진입점
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K-means 클러스터링 (LEGACY)")

    parser.add_argument(
        "--profile-csv", type=Path,
        default=Path("data/processed/b040/b040_24h_profile.csv"),
        help="행정동 × 시간대 프로파일 (캠퍼스 internal)",
    )
    parser.add_argument(
        "--night-index-csv", type=Path,
        default=Path("data/processed/b040/dong_night_index_final.csv"),
        help="야간특화지수 (D유형 사전제외 필터)",
    )
    parser.add_argument("--dong-col", type=str, default="dong_code")
    parser.add_argument("--night-col", type=str, default="night_special_index")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=6)

    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("data/processed/kmeans"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] 입력 로드 (24h 프로파일 + 야간특화지수)")
    profile, night_idx = load_profile(
        args.profile_csv, args.night_index_csv,
        dong_col=args.dong_col,
    )

    print("\n[2] K-means 입력 필터 (D유형 + 이상치 동시 제외)")
    filtered = filter_kmeans_input(
        profile, night_idx,
        dong_col=args.dong_col,
        night_col=args.night_col,
    )

    feature_cols = [c for c in filtered.columns
                    if c not in (args.dong_col, args.night_col)]
    print(f"  특성 컬럼: {len(feature_cols)}개")

    print("\n[3] 비율 정규화 + StandardScaler")
    X_scaled = normalize_to_ratio(filtered, args.dong_col, feature_cols)

    print(f"\n[4] K 범위 평가 ({args.k_min}~{args.k_max})")
    k_eval = evaluate_k_range(X_scaled, args.k_min, args.k_max)
    eval_path = args.out_dir / "kmeans_k_evaluation.csv"
    k_eval.to_csv(eval_path, index=False, encoding="utf-8-sig")
    print(f"  저장: {eval_path}")

    print(f"\n[5] K-means 학습 (K={args.k})")
    labels, km = fit_kmeans(X_scaled, args.k)
    filtered_with_label = filtered.copy()
    filtered_with_label["cluster"] = labels

    print(f"\n[6] Kruskal-Wallis 시간대별 검정")
    kw = kruskal_wallis_per_hour(filtered_with_label, labels, feature_cols)
    kw_path = args.out_dir / "kmeans_validation_kruskal.csv"
    kw.to_csv(kw_path, index=False, encoding="utf-8-sig")
    print(f"  저장: {kw_path}")

    print("\n[7] D유형·이상치 합산 → 최종 라벨 부여")
    final = assign_labels_and_merge_d(
        filtered_with_label, night_idx,
        dong_col=args.dong_col,
        night_col=args.night_col,
    )

    out_path = args.out_dir / "dong_cluster_label_k3.csv"
    final.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장 (반출용): {out_path} ({len(final)}행)")

    # 분포 요약
    print("\n클러스터 분포:")
    print(final["cluster_name"].value_counts().to_string())


if __name__ == "__main__":
    # 실행 예시:
    # import sys
    # sys.argv = [
    #     "kmeans_clustering.py",
    #     "--profile-csv", "data/processed/b040/b040_24h_profile.csv",
    #     "--night-index-csv", "data/processed/b040/dong_night_index_final.csv",
    #     "--k", "3",
    #     "--out-dir", "data/processed/kmeans",
    # ]
    # main()
    pass
