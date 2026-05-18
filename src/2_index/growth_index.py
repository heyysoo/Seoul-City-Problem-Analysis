"""매출 성장률 지수 (growth_norm) — 잠재력 지수 첫 번째 변수 (가중치 50%)

처리 목적
- b079.py 산출물(`yoy_full_{track}.csv` 또는 `행정동_카드매출.csv`)을 입력으로 받아
  저녁/심야 트랙별 growth_norm 산출.
- 동별 YoY 평균 → 서울 중앙값 차이 → 1·99 percentile 클리핑 → Min-Max → [0, 1].
- 잠재력 지수 합산 공식의 50% 가중치 (직접 신호).

학술 근거 (가중치 50%)
- Glaeser, Kerr, & Kerr (2015) RES: 도시 1인당 기업 수 ↑ → 고용 성장
- Lin et al. (2022) Tourism Economics: NTEVI 6 sub-indices 핵심 변수로 매출 채택
- Jeong & Jun (2022) Buildings: 서울 NTE 카드매출 PLS-SEM
- Choi et al. (2024) arXiv: 서울 BC카드 (2018-2023) 학술 분석

5가지 핵심 결정
1. 24개월 평균 (단년 변동 평탄화) — b079.py 단계 적용
2. 서울 중앙값 차이 (Median 정통, Applebaum 1966)
3. 둔촌1동 자연 제외 (B079에 재건축으로 카드매출 미관측)
4. 1·99 percentile 클리핑 (문래동 등 극단값 영향 제한)
5. Min-Max → [0, 1] 정규화

산식
- yoy_diff = 동별 YoY 평균 - 서울 중앙값  (b079.py에서 산출)
- yoy_diff_clip = clip(yoy_diff, q01, q99)
- growth_norm = (yoy_diff_clip - min) / (max - min) ∈ [0, 1]

코드 매핑
- B079는 10자리 코드 → 8자리 행자부 코드로 변환
- 강북구 6쌍 수동 매핑 (b079.py와 동일)
- 신사동 강남(11680510)/관악(11620685) 분리는 b079.py 단계에서 이미 처리

입력 (트랙별 별도 실행)
- yoy_full_{track}.csv: b079.py 산출물 (YoY 계산 완료 + 행정동별 평균)
  → 자세히는 카드매출YoY_{track}.csv 와 동일 구조

산출 (out_dir 하위)
- growth_norm_evening.csv, growth_norm_late.csv: 트랙별 growth_norm
- 컬럼: amd_code (행자부 8자리), amd_name, yoy_mean, yoy_diff, growth_norm

마스터 노션 페이지: 잠재력 지수 마스터 참조 (358b941b-9e4a-8140-9451-ff76f227fecd)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# -----------------------------------------------------------------------------
# 상수
# -----------------------------------------------------------------------------

CLIP_LOWER_PCT = 0.01
CLIP_UPPER_PCT = 0.99

# 강북구 6쌍 매핑 (B079 10자리 → B040 8자리)
# 노션 정통: 옛 대화 692f8ee6 (Ojiro + Claude) 검증 완료, 419개 동 매칭 0 failures
# 앞 8자리 절단 규칙과 다르게 별도 부여된 코드. b079.py·gap_index.py와 동일 매핑 유지.
GANGBUK_CODE_MAPPING = {
    "1130559500": "11305590",   # 번1동
    "1130560300": "11305600",   # 번2동
    "1130560800": "11305606",   # 번3동
    "1130561500": "11305610",   # 수유1동
    "1130562500": "11305620",   # 수유2동
    "1130563500": "11305630",   # 수유3동
}

# 둔촌1동 (재건축으로 자연 제외 — b079.py에서 이미 처리되지만 안전망)
DUNCHON_1_DONG_CODES = frozenset({"1174069000", "11740690"})


# -----------------------------------------------------------------------------
# Step 1 — 입력 로딩
# -----------------------------------------------------------------------------

def load_yoy_summary(
    path: Path,
    region_col: str = "amd_code",
    yoy_col: str = "YoY_상대_차이",
    encoding: str = "utf-8-sig",
) -> pd.DataFrame:
    """b079.py 산출물 (카드매출YoY_{track}.csv) 로드.

    필요 컬럼: amd_code, YoY_평균, YoY_상대_차이
    """
    df = pd.read_csv(path, encoding=encoding, dtype={region_col: str})
    df[region_col] = df[region_col].astype(str).str.strip()

    if yoy_col not in df.columns:
        # 대체 컬럼명 (csv 버전별 차이 대응)
        for alt in ["yoy_diff", "YoY_차이", "yoy_상대차이"]:
            if alt in df.columns:
                yoy_col = alt
                break
        else:
            raise KeyError(f"YoY 차이 컬럼 없음. 현재: {df.columns.tolist()}")

    out = df.copy()
    out["yoy_diff"] = pd.to_numeric(out[yoy_col], errors="coerce")
    return out


# -----------------------------------------------------------------------------
# Step 2 — 코드 매핑 (B079 10자리 → 행자부 8자리)
# -----------------------------------------------------------------------------

def map_b079_to_haengjabu(amd_code_10: str) -> str:
    """B079 10자리 → 행자부 8자리.

    기본: 앞 8자리 절단. 강북 6쌍 수동 매핑 우선.
    """
    code = str(amd_code_10).strip()
    if code in GANGBUK_CODE_MAPPING:
        return GANGBUK_CODE_MAPPING[code]
    return code[:8] if len(code) >= 8 else code.zfill(8)


def apply_code_mapping(df: pd.DataFrame, region_col: str) -> pd.DataFrame:
    """전체 row에 코드 매핑 적용."""
    out = df.copy()
    out["amd_code_8"] = out[region_col].apply(map_b079_to_haengjabu)
    return out


# -----------------------------------------------------------------------------
# Step 3 — 둔촌1동 자연 제외
# -----------------------------------------------------------------------------

def exclude_dunchon(df: pd.DataFrame, region_col: str) -> pd.DataFrame:
    """둔촌1동 자연 제외 (재건축 카드매출 미관측)."""
    out = df.copy()
    before = len(out)
    mask = out[region_col].isin(DUNCHON_1_DONG_CODES) | out["amd_code_8"].isin(DUNCHON_1_DONG_CODES)
    out = out[~mask].copy()
    if before - len(out) > 0:
        print(f"  둔촌1동 제외: {before} → {len(out)} ({before - len(out)}개)")
    return out


# -----------------------------------------------------------------------------
# Step 4 — 1·99 클리핑 + Min-Max 정규화
# -----------------------------------------------------------------------------

def clip_and_minmax(
    series: pd.Series,
    lower_pct: float = CLIP_LOWER_PCT,
    upper_pct: float = CLIP_UPPER_PCT,
) -> pd.Series:
    """1·99 percentile 클리핑 후 Min-Max 정규화 → [0, 1]."""
    s = pd.to_numeric(series, errors="coerce")
    lo = s.quantile(lower_pct)
    hi = s.quantile(upper_pct)
    clipped = s.clip(lower=lo, upper=hi)
    rng = clipped.max() - clipped.min()
    if rng == 0:
        return pd.Series(0.5, index=s.index)
    return (clipped - clipped.min()) / rng


# -----------------------------------------------------------------------------
# 통합 파이프라인
# -----------------------------------------------------------------------------

def build_growth_norm(
    path: Path,
    region_col: str,
    yoy_col: str,
    track_name: str,
    encoding: str = "utf-8-sig",
) -> pd.DataFrame:
    """트랙별 growth_norm 산출 (4단계)."""
    print(f"\n[{track_name}] 입력 로드")
    df = load_yoy_summary(path, region_col=region_col, yoy_col=yoy_col, encoding=encoding)
    print(f"  로드: {len(df)}행")

    print(f"[{track_name}] B079 10자리 → 행자부 8자리 코드 매핑")
    df = apply_code_mapping(df, region_col=region_col)

    print(f"[{track_name}] 둔촌1동 자연 제외")
    df = exclude_dunchon(df, region_col=region_col)

    print(f"[{track_name}] 1·99 클리핑 + Min-Max 정규화")
    df["growth_norm"] = clip_and_minmax(df["yoy_diff"])

    n_clipped = ((df["yoy_diff"] <= df["yoy_diff"].quantile(CLIP_LOWER_PCT))
                 | (df["yoy_diff"] >= df["yoy_diff"].quantile(CLIP_UPPER_PCT))).sum()
    print(f"  클리핑 적용 동: {n_clipped}개")
    print(f"  growth_norm 평균 {df['growth_norm'].mean():.3f} / "
          f"중앙값 {df['growth_norm'].median():.3f}")

    # 출력 정리
    out_cols = ["amd_code_8", "yoy_diff", "growth_norm"]
    if "amd_name" in df.columns:
        out_cols.insert(1, "amd_name")
    if "YoY_평균" in df.columns:
        df["yoy_mean"] = df["YoY_평균"]
        out_cols.insert(out_cols.index("yoy_diff"), "yoy_mean")

    return df[out_cols].rename(columns={"amd_code_8": "amd_code"})


# -----------------------------------------------------------------------------
# 진입점
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="매출 성장률 (growth_norm) 산출")

    parser.add_argument(
        "--evening-csv", type=Path,
        default=Path("data/processed/b079/evening/카드매출YoY_evening.csv"),
        help="저녁 트랙 b079 산출물 (YoY 평균 + 상대 차이 포함)",
    )
    parser.add_argument(
        "--late-csv", type=Path,
        default=Path("data/processed/b079/late_night/카드매출YoY_late_night.csv"),
        help="심야 트랙 b079 산출물",
    )
    parser.add_argument("--region-col", type=str, default="amd_code")
    parser.add_argument("--yoy-col", type=str, default="YoY_상대_차이")
    parser.add_argument("--encoding", type=str, default="utf-8-sig")

    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("data/processed/growth"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.evening_csv.exists():
        evening = build_growth_norm(
            args.evening_csv,
            region_col=args.region_col,
            yoy_col=args.yoy_col,
            track_name="evening",
            encoding=args.encoding,
        )
        eve_path = args.out_dir / "growth_norm_evening.csv"
        evening.to_csv(eve_path, index=False, encoding="utf-8-sig")
        print(f"\n저장 (저녁): {eve_path} ({len(evening)}행)")
    else:
        print(f"{args.evening_csv} 없음 — 저녁 트랙 건너뜀")

    if args.late_csv.exists():
        late = build_growth_norm(
            args.late_csv,
            region_col=args.region_col,
            yoy_col=args.yoy_col,
            track_name="late_night",
            encoding=args.encoding,
        )
        lat_path = args.out_dir / "growth_norm_late.csv"
        late.to_csv(lat_path, index=False, encoding="utf-8-sig")
        print(f"\n저장 (심야): {lat_path} ({len(late)}행)")
    else:
        print(f"{args.late_csv} 없음 — 심야 트랙 건너뜀")


if __name__ == "__main__":
    # 실행 예시:
    # import sys
    # sys.argv = [
    #     "growth_index.py",
    #     "--evening-csv", "data/processed/b079/evening/카드매출YoY_evening.csv",
    #     "--late-csv", "data/processed/b079/late_night/카드매출YoY_late_night.csv",
    #     "--out-dir", "data/processed/growth",
    # ]
    # main()
    pass
