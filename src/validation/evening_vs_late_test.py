"""저녁·심야 YoY 분포 차이 검정 (방법론 정당화)

위상
- 본 검정은 §방법론에서 "왜 저녁 트랙과 심야 트랙을 별도로 분석해야 하는가"를
  정당화하기 위한 분포 차이 검증이다.
- 지수 산출의 견고성 검증(Equal/Entropy/PCA/Spearman)과는 별개의 항목이다.

가설
- H0: 행정동별 저녁 YoY 평균과 심야 YoY 평균은 동일하다.
- H1: 두 평균은 다르다.

검정 절차
1. 대응표본 t검정 (paired t-test): 정규성 가정 기반
2. Wilcoxon signed-rank test: 비모수 (정규성 가정 불필요)
3. 효과 크기: Cohen's d, 비모수 r = |Z|/√N, rank-biserial r
4. 분포 형태 비교: Kolmogorov-Smirnov 양표본 검정
5. IQR 1.5배 기준 이상치 제거 후 재검정 (강건성 확인)

산출물
- result_yoy_evening_vs_night.csv      (원본 데이터 결과)
- result_yoy_outlier_comparison.csv    (원본 vs 이상치 제거 비교)
- result_yoy_qq_plot.png               (차이값 정규성 Q-Q)
- result_yoy_outlier_comparison.png    (제거 전후 분포 비교)

해석 (기존 분석 결과)
- 정제 전: 차이값 skew=16.28, kurtosis=310 (극심한 양의 왜도)
- IQR 1.5배 기준 이상치 37개(8.7%) 제거 후 정규성 충족 (Shapiro p=0.148, skew=-0.17)
- paired t-test, Wilcoxon 모두 p<0.001, 효과 크기 중간 수준 (Cohen's d=0.44, r=0.48)
- 정제 전후 본질적 패턴 일관 → 저녁 우세 결론 강건
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 헤드리스 환경 안전
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


COL_DONG_DEFAULT = "amd_name"
COL_EVE_DEFAULT = "YoY_평균_eve"
COL_NGT_DEFAULT = "YoY_평균_ngt"


def configure_korean_font() -> None:
    """플랫폼별 한글 폰트 설정 (그래프 한글 표시용)."""
    import platform

    system = platform.system()
    if system == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif system == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"
    else:
        plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False


def load_yoy_data(
    eve_path: Path,
    ngt_path: Path,
    col_dong: str,
    col_eve: str,
    col_ngt: str,
    exclude_dong: list[str] | None = None,
) -> pd.DataFrame:
    """저녁·심야 YoY 데이터를 로드하고 동 키로 머지."""
    df_eve = pd.read_csv(eve_path)
    df_ngt = pd.read_csv(ngt_path)

    if exclude_dong:
        df_eve = df_eve[~df_eve[col_dong].isin(exclude_dong)].reset_index(drop=True)
        df_ngt = df_ngt[~df_ngt[col_dong].isin(exclude_dong)].reset_index(drop=True)

    # 컬럼 이름이 양쪽에서 같으면 suffix로 구분되도록 그대로 머지
    df_yoy = df_eve.merge(df_ngt, on=col_dong, suffixes=("_eve", "_ngt"))
    return df_yoy


def safe_shapiro(x: np.ndarray) -> tuple[float, float]:
    if 3 <= len(x) <= 5000:
        return stats.shapiro(x)
    return (np.nan, np.nan)


def run_paired_test(
    paired: pd.DataFrame,
    col_eve: str,
    col_ngt: str,
    label: str = "",
) -> pd.DataFrame:
    """대응표본 검정 + 효과 크기 + 분포 형태 비교 (단일 데이터셋)."""
    paired = paired.dropna(subset=[col_eve, col_ngt]).reset_index(drop=True)
    x_eve = paired[col_eve].to_numpy(dtype=float)
    x_ngt = paired[col_ngt].to_numpy(dtype=float)
    diff = x_eve - x_ngt
    n = len(diff)

    # 정규성
    sw_eve_stat, sw_eve_p = safe_shapiro(x_eve)
    sw_ngt_stat, sw_ngt_p = safe_shapiro(x_ngt)
    sw_diff_stat, sw_diff_p = safe_shapiro(diff)

    skew_diff = stats.skew(diff)
    kurt_diff = stats.kurtosis(diff)

    # 본 검정
    t_stat, t_p = stats.ttest_rel(x_eve, x_ngt)
    w_stat, w_p = stats.wilcoxon(x_eve, x_ngt)

    # 효과 크기
    cohens_d = abs(diff.mean()) / diff.std(ddof=1)
    mean_W = n * (n + 1) / 4
    sd_W = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z_w = (w_stat - mean_W) / sd_W
    r_effect = abs(z_w) / np.sqrt(n)
    r_rb = abs(1 - (2 * w_stat) / (n * (n + 1) / 2))

    # 부호 카운트 + KS
    n_eve_bigger = int((diff > 0).sum())
    n_ngt_bigger = int((diff < 0).sum())
    n_tie = int((diff == 0).sum())
    ks_stat, ks_p = stats.ks_2samp(x_eve, x_ngt)

    col_label = f"값({label})" if label else "값"
    return pd.DataFrame(
        {
            col_label: [
                n,
                x_eve.mean(), x_ngt.mean(),
                diff.mean(), diff.std(ddof=1), skew_diff, kurt_diff,
                n_eve_bigger, n_ngt_bigger, n_tie,
                sw_eve_p, sw_ngt_p, sw_diff_p,
                (sw_diff_p > 0.05) if not np.isnan(sw_diff_p) else None,
                t_stat, t_p,
                w_stat, w_p,
                cohens_d, r_effect, r_rb,
                ks_stat, ks_p,
            ]
        },
        index=[
            "쌍 표본수 N",
            "저녁 yoy 평균", "심야 yoy 평균",
            "차이 평균", "차이 SD", "차이 왜도", "차이 첨도",
            "저녁>심야 케이스", "심야>저녁 케이스", "동일",
            "저녁 Shapiro p", "심야 Shapiro p", "차이값 Shapiro p",
            "차이값 정규성(p>0.05)",
            "Paired t 통계량", "Paired t p값",
            "Wilcoxon 통계량", "Wilcoxon p값",
            "Cohen's d", "비모수 r (|Z|/√N)", "rank-biserial r",
            "KS 통계량", "KS p값",
        ],
    )


def remove_outliers_iqr(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    k: float = 1.5,
    mode: str = "diff",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """차이값 또는 두 컬럼 동시 기준 IQR 이상치 제거.

    mode:
      'diff' → 차이값(col_a - col_b) 기준
      'both' → 두 컬럼 모두 IQR 범위 내인 행만 유지 (보수적)
    """
    work = df.dropna(subset=[col_a, col_b]).copy()
    work["_diff"] = work[col_a] - work[col_b]

    if mode == "diff":
        q1, q3 = work["_diff"].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        mask = work["_diff"].between(lo, hi)
    elif mode == "both":
        mask = pd.Series(True, index=work.index)
        for col in [col_a, col_b]:
            q1, q3 = work[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lo, hi = q1 - k * iqr, q3 + k * iqr
            mask &= work[col].between(lo, hi)
    else:
        raise ValueError(f"알 수 없는 mode: {mode}")

    cleaned = work[mask].drop(columns=["_diff"]).reset_index(drop=True)
    removed = work[~mask].drop(columns=["_diff"]).reset_index(drop=True)
    return cleaned, removed


def plot_diff_diagnostics(
    diff: np.ndarray,
    title_prefix: str,
    out_path: Path,
) -> None:
    """단일 차이값 분포의 Q-Q + 히스토그램."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sw_p = safe_shapiro(diff)[1]
    skew_v = stats.skew(diff)

    stats.probplot(diff, dist="norm", plot=axes[0])
    axes[0].set_title(f"Q-Q: {title_prefix}\nskew={skew_v:.2f}, Shapiro p={sw_p:.4f}")
    axes[0].grid(alpha=0.3)

    axes[1].hist(diff, bins=30, density=True, alpha=0.6, edgecolor="black")
    axes[1].axvline(0, color="gray", linestyle=":", alpha=0.7, label="Zero")
    axes[1].axvline(diff.mean(), color="red", linestyle="--", alpha=0.7, label="Mean")
    axes[1].set_title(f"Histogram: {title_prefix}")
    axes[1].set_xlabel("Evening yoy - Night yoy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_outlier_comparison(
    diff_orig: pd.Series,
    diff_clean: pd.Series,
    out_path: Path,
) -> None:
    """이상치 제거 전후 Q-Q + 히스토그램 비교."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    stats.probplot(diff_orig, dist="norm", plot=axes[0, 0])
    axes[0, 0].set_title(
        f"Q-Q: Original (n={len(diff_orig)}, skew={stats.skew(diff_orig):.2f})"
    )
    axes[0, 0].grid(alpha=0.3)

    stats.probplot(diff_clean, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title(
        f"Q-Q: Cleaned (n={len(diff_clean)}, skew={stats.skew(diff_clean):.2f})"
    )
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].hist(diff_orig, bins=40, alpha=0.6, edgecolor="black")
    axes[1, 0].axvline(0, color="gray", linestyle=":")
    axes[1, 0].set_title("Histogram: Original")
    axes[1, 0].set_xlabel("Evening yoy - Night yoy")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].hist(diff_clean, bins=40, alpha=0.6, edgecolor="black", color="orange")
    axes[1, 1].axvline(0, color="gray", linestyle=":")
    axes[1, 1].set_title("Histogram: Cleaned")
    axes[1, 1].set_xlabel("Evening yoy - Night yoy")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="저녁·심야 YoY 분포 차이 검정")
    parser.add_argument(
        "--eve-path",
        type=Path,
        default=Path("data/processed/index/카드매출YoY(저녁).csv"),
    )
    parser.add_argument(
        "--ngt-path",
        type=Path,
        default=Path("data/processed/index/카드매출YoY(심야_문래동제외).csv"),
    )
    parser.add_argument("--col-dong", type=str, default=COL_DONG_DEFAULT)
    parser.add_argument("--col-eve", type=str, default=COL_EVE_DEFAULT)
    parser.add_argument("--col-ngt", type=str, default=COL_NGT_DEFAULT)
    parser.add_argument(
        "--exclude-dong",
        nargs="*",
        default=["문래동"],
        help="저녁 데이터에서 제외할 동 이름 (심야는 파일명상 이미 제외됨)",
    )
    parser.add_argument("--iqr-k", type=float, default=1.5)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/validation"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_korean_font()

    df_yoy = load_yoy_data(
        args.eve_path, args.ngt_path,
        args.col_dong, args.col_eve, args.col_ngt,
        exclude_dong=args.exclude_dong,
    )

    # 1) 원본 데이터 검정
    result_orig = run_paired_test(df_yoy, args.col_eve, args.col_ngt, label="원본")
    result_orig.to_csv(
        args.out_dir / "result_yoy_evening_vs_night.csv",
        encoding="utf-8-sig",
    )

    # Q-Q 진단 (원본)
    diff_orig = (df_yoy[args.col_eve] - df_yoy[args.col_ngt]).dropna()
    plot_diff_diagnostics(
        diff_orig.to_numpy(),
        title_prefix="yoy diff (original)",
        out_path=args.out_dir / "result_yoy_qq_plot.png",
    )

    # 2) IQR 이상치 제거 후 검정
    df_clean, df_removed = remove_outliers_iqr(
        df_yoy, args.col_eve, args.col_ngt, k=args.iqr_k, mode="diff",
    )
    print(f"원본: {len(df_yoy)}개 → 정제 후: {len(df_clean)}개")
    print(f"제거된 행정동: {len(df_removed)}개")

    if len(df_removed) > 0:
        removed_sorted = (
            df_removed.assign(diff=lambda d: d[args.col_eve] - d[args.col_ngt])
            .reindex(columns=[args.col_dong, args.col_eve, args.col_ngt, "diff"])
            .sort_values("diff", key=abs, ascending=False)
        )
        print("\n[제거된 행정동 상위 10]")
        print(removed_sorted.head(10).to_string(index=False))

    result_clean = run_paired_test(df_clean, args.col_eve, args.col_ngt, label="이상치제거")
    result_compare = pd.concat([result_orig, result_clean], axis=1)
    result_compare.to_csv(
        args.out_dir / "result_yoy_outlier_comparison.csv",
        encoding="utf-8-sig",
    )
    print("\n[원본 vs 이상치 제거 비교]")
    print(result_compare.to_string())

    # 비교 시각화
    diff_clean = df_clean[args.col_eve] - df_clean[args.col_ngt]
    plot_outlier_comparison(
        diff_orig, diff_clean,
        out_path=args.out_dir / "result_yoy_outlier_comparison.png",
    )

    print(f"\n저장 폴더: {args.out_dir}")


if __name__ == "__main__":
    # 실행 예시 (필요 시 주석 해제):
    # import sys
    # sys.argv = [
    #     "evening_vs_late_test.py",
    #     "--eve-path", "data/processed/index/카드매출YoY(저녁).csv",
    #     "--ngt-path", "data/processed/index/카드매출YoY(심야_문래동제외).csv",
    #     "--out-dir", "outputs/validation",
    # ]
    # main()
    pass
