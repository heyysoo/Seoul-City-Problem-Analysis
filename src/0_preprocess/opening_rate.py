"""야간 특화 업종 신규 개업률 (opening) — 잠재력 지수 두 번째 변수 (가중치 30%)

처리 목적
- 서울시 상권분석서비스 점포-행정동 OpenAPI(OA-22172)에서 분기별 점포 데이터 수집
- 야간 특화 10개 업종 필터 → 행정동×분기 집계 → 분기별 개업률 → 12분기 평균
- 잠재력 지수 X축의 세 변수 중 두 번째 (저녁/심야 무관 단일 값)

산식 (4단계)
1. 야간 특화 10개 업종 필터 (전체 100개 업종 중 야간 영업 위주)
   - 423,454행 → 32,585행
2. 행정동 * 기준년분기별 집계 (점포수·개업·폐업 합계)
3. 분기별 개업률 = 개업점포수_합계 / 점포수_합계
4. 행정동별 12분기 산술평균 → avg_opening_rate

야간 특화 10개 업종 (CS 코드)
- CS100009 호프-간이주점 / CS200016 당구장 / CS200019 PC방 / CS200020 전자게임장
- CS200021 기타오락장 / CS200034 여관 / CS200035 게스트하우스 / CS200036 고시원
- CS200037 노래방 / CS200039 DVD방

데이터 출처
- 서울시 상권분석서비스 점포-행정동 (OA-22172)
- https://data.seoul.go.kr/dataList/OA-22172/S/1/datasetView.do
- 인코딩: cp949 (CSV 다운로드 시) / json (OpenAPI 시)
- 시간 범위: 2023 Q1 ~ 2025 Q4 (12개 분기)

입력 모드 두 가지
- API 모드 (--mode api): OpenAPI에서 직접 수집 (인증키 필요)
- CSV 모드 (--mode csv): 다운로드 받은 cp949 CSV 파일 처리

산출물
- raw_stores_yearly.csv: API/CSV 통합 원자료 (전체 423,454행, 검증용)
- opening_rate.csv (~431행, 6컬럼): 행정동_코드, 행정동_코드_명,
  avg_opening_rate, avg_store_count, total_new_store, total_cls_store

처리 노트: 5개 미만 점포수 동의 보정(is_imputed 처리) + 통계청↔행자부
코드 변환은 마스터 테이블 단계(potential.py 입력 결합 시)에서 적용.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests


# -----------------------------------------------------------------------------
# 상수
# -----------------------------------------------------------------------------

# 야간 특화 10개 업종 (서비스_업종_코드)
NIGHT_INDUSTRIES = (
    "CS100009",   # 호프-간이주점
    "CS200016",   # 당구장
    "CS200019",   # PC방
    "CS200020",   # 전자게임장
    "CS200021",   # 기타오락장
    "CS200034",   # 여관
    "CS200035",   # 게스트하우스
    "CS200036",   # 고시원
    "CS200037",   # 노래방
    "CS200039",   # DVD방
)

# API 페이지 크기 (서울 OpenAPI 최대 1,000건/호출)
PAGE_SIZE = 1000


# -----------------------------------------------------------------------------
# Step 0 — API/CSV 입력 로딩
# -----------------------------------------------------------------------------

def fetch_total_count(base_url: str) -> tuple[int, str]:
    """전체 데이터 건수와 응답 최상위 키 확인."""
    url = f"{base_url}/1/1"
    res = requests.get(url, timeout=30)
    data = res.json()
    key = list(data.keys())[0]
    total = data[key]["list_total_count"]
    print(f"  전체 건수: {total:,}")
    return total, key


def fetch_via_api(
    api_key: str,
    service_name: str = "VwsmAdstrdStorW",
    page_size: int = PAGE_SIZE,
    sleep_sec: float = 0.5,
) -> pd.DataFrame:
    """서울 OpenAPI 페이징 수집 → DataFrame."""
    base_url = f"http://openapi.seoul.go.kr:8088/{api_key}/json/{service_name}"
    total, key = fetch_total_count(base_url)

    all_rows: list[dict] = []
    for start in range(1, total + 1, page_size):
        end = min(start + page_size - 1, total)
        url = f"{base_url}/{start}/{end}"
        res = requests.get(url, timeout=30)
        data = res.json()
        rows = data[key].get("row", [])
        all_rows.extend(rows)
        if end % 10000 == 1 or end == total:
            print(f"  수집: {end:,} / {total:,}")
        time.sleep(sleep_sec)

    return pd.DataFrame(all_rows)


def load_from_csv(
    csv_paths: list[Path],
    encoding: str = "cp949",
) -> pd.DataFrame:
    """수동 다운로드 CSV 파일들 통합 로드."""
    frames = []
    for p in csv_paths:
        if not p.exists():
            print(f"  ⚠️ {p} 없음 — 건너뜀")
            continue
        df = pd.read_csv(p, encoding=encoding)
        print(f"  로드: {p.name} ({len(df):,}행)")
        frames.append(df)
    if not frames:
        raise FileNotFoundError("로드된 CSV 파일이 없습니다.")
    return pd.concat(frames, ignore_index=True)


# -----------------------------------------------------------------------------
# 컬럼명 정규화 (API JSON 키 vs CSV 한글 컬럼 매핑)
# -----------------------------------------------------------------------------

# API JSON 키 → 표준 한글 컬럼
API_COLUMN_MAP = {
    "STDR_YY_CD": "기준_년_코드",
    "STDR_QU_CD": "기준_분기_코드",
    "STDR_YYQU_CD": "기준_년분기_코드",
    "ADSTRD_CD": "행정동_코드",
    "ADSTRD_CD_NM": "행정동_코드_명",
    "SVC_INDUTY_CD": "서비스_업종_코드",
    "SVC_INDUTY_CD_NM": "서비스_업종_코드_명",
    "STOR_CO": "점포_수",
    "SIMILR_INDUTY_STOR_CO": "유사_업종_점포_수",
    "OPBIZ_RT": "개업_률",
    "OPBIZ_STOR_CO": "개업_점포_수",
    "CLSBIZ_RT": "폐업_률",
    "CLSBIZ_STOR_CO": "폐업_점포_수",
    "FRC_STOR_CO": "프랜차이즈_점포_수",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """API 키 → 한글 컬럼명 통일 + 기준_년분기_코드 보강."""
    out = df.copy()
    for src, tgt in API_COLUMN_MAP.items():
        if src in out.columns and tgt not in out.columns:
            out = out.rename(columns={src: tgt})

    # 기준_년분기_코드 보강 (API에 없으면 yy+q 결합)
    if "기준_년분기_코드" not in out.columns:
        if "기준_년_코드" in out.columns and "기준_분기_코드" in out.columns:
            out["기준_년분기_코드"] = (
                out["기준_년_코드"].astype(str) + out["기준_분기_코드"].astype(str)
            )
    return out


# -----------------------------------------------------------------------------
# Step 1 — 야간 특화 10개 업종 필터
# -----------------------------------------------------------------------------

def filter_night_industries(df: pd.DataFrame) -> pd.DataFrame:
    """서비스_업종_코드 기준 10개 야간 업종만 유지."""
    if "서비스_업종_코드" not in df.columns:
        raise KeyError(f"'서비스_업종_코드' 컬럼 없음. 현재: {df.columns.tolist()}")

    before = len(df)
    filtered = df[df["서비스_업종_코드"].isin(NIGHT_INDUSTRIES)].copy()
    after = len(filtered)
    print(f"  야간 10개 업종 필터: {before:,} → {after:,} (유지 {after / before:.1%})")
    return filtered


# -----------------------------------------------------------------------------
# Step 2 — 행정동 × 분기 집계
# -----------------------------------------------------------------------------

def aggregate_quarterly(df: pd.DataFrame) -> pd.DataFrame:
    """행정동 × 기준년분기 단위로 점포수·개업·폐업 합계."""
    numeric_cols = ["점포_수", "개업_점포_수", "폐업_점포_수"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    quarterly = df.groupby(
        ["행정동_코드", "행정동_코드_명", "기준_년분기_코드"],
        as_index=False,
    ).agg(
        점포수_합계=("점포_수", "sum"),
        개업점포수_합계=("개업_점포_수", "sum"),
        폐업점포수_합계=("폐업_점포_수", "sum"),
    )

    # 분기별 개업률 (분모 0 회피)
    denom = quarterly["점포수_합계"].replace(0, pd.NA)
    quarterly["분기별_개업률"] = quarterly["개업점포수_합계"] / denom

    print(f"  분기 집계: {len(quarterly):,}행 (행정동×분기)")
    return quarterly


# -----------------------------------------------------------------------------
# Step 3 — 행정동별 12분기 평균
# -----------------------------------------------------------------------------

def aggregate_yearly_avg(quarterly: pd.DataFrame) -> pd.DataFrame:
    """행정동별 12분기 산술평균 개업률."""
    result = quarterly.groupby(
        ["행정동_코드", "행정동_코드_명"], as_index=False,
    ).agg(
        avg_opening_rate=("분기별_개업률", "mean"),
        avg_store_count=("점포수_합계", "mean"),
        total_new_store=("개업점포수_합계", "sum"),
        total_cls_store=("폐업점포수_합계", "sum"),
        n_quarters=("기준_년분기_코드", "nunique"),
    )

    # 8자리 코드 통일
    result["행정동_코드"] = (
        result["행정동_코드"].astype(int).astype(str).str.zfill(8)
    )

    print(f"  12분기 평균: {len(result)}개 행정동")
    return result


# -----------------------------------------------------------------------------
# 진입점
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="야간 특화 업종 신규 개업률 산출 (OA-22172 전체 파이프라인)"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["api", "csv"],
        default="csv",
        help="입력 모드: api(서울 OpenAPI 수집) 또는 csv(다운로드 파일 처리)",
    )

    # API 모드
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("SEOUL_OPENAPI_KEY", ""),
        help="서울 OpenAPI 키 (--mode api). 환경변수 SEOUL_OPENAPI_KEY도 사용 가능.",
    )
    parser.add_argument("--service-name", type=str, default="VwsmAdstrdStorW",
                        help="OpenAPI 서비스명")
    parser.add_argument("--page-size", type=int, default=PAGE_SIZE)
    parser.add_argument("--sleep-sec", type=float, default=0.5)

    # CSV 모드
    parser.add_argument(
        "--csv-paths",
        type=Path,
        nargs="+",
        default=[
            Path("data/raw/opening/서울시_상권분석서비스_점포_2023.csv"),
            Path("data/raw/opening/서울시_상권분석서비스_점포_2024.csv"),
            Path("data/raw/opening/서울시_상권분석서비스_점포_2025.csv"),
        ],
        help="수동 다운로드 CSV 경로들 (--mode csv)",
    )
    parser.add_argument("--csv-encoding", type=str, default="cp949")

    # 출력
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("data/processed/opening_rate"),
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="필터 전 원자료(raw_stores_yearly.csv) 저장",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # [0] 입력 로딩
    print(f"[0] 입력 로딩 (mode={args.mode})")
    if args.mode == "api":
        if not args.api_key:
            raise SystemExit("API 키 없음. --api-key 또는 SEOUL_OPENAPI_KEY 환경변수 설정 필요.")
        df = fetch_via_api(
            api_key=args.api_key,
            service_name=args.service_name,
            page_size=args.page_size,
            sleep_sec=args.sleep_sec,
        )
    else:
        df = load_from_csv(args.csv_paths, encoding=args.csv_encoding)

    print(f"  원자료: {len(df):,}행")
    df = normalize_columns(df)

    if args.save_raw:
        raw_path = args.out_dir / "raw_stores_yearly.csv"
        df.to_csv(raw_path, index=False, encoding="utf-8-sig")
        print(f"  원자료 저장: {raw_path}")

    # [1] 야간 10개 업종 필터
    print("\n[1] 야간 특화 10개 업종 필터")
    df_night = filter_night_industries(df)

    # [2] 행정동 × 분기 집계
    print("\n[2] 행정동 × 분기 집계 → 분기별 개업률")
    quarterly = aggregate_quarterly(df_night)
    quarterly_path = args.out_dir / "quarterly_opening_rate.csv"
    quarterly.to_csv(quarterly_path, index=False, encoding="utf-8-sig")
    print(f"  저장 (분기): {quarterly_path}")

    # [3] 12분기 평균
    print("\n[3] 행정동별 12분기 평균")
    result = aggregate_yearly_avg(quarterly)

    out_path = args.out_dir / "opening_rate.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_path} ({len(result)}개 행정동)")

    # 검증 통계
    print("\n분포 통계:")
    print(f"  avg_opening_rate 평균: {result['avg_opening_rate'].mean():.4f}")
    print(f"  avg_opening_rate 중앙값: {result['avg_opening_rate'].median():.4f}")
    print(f"  점포수 5개 미만 행정동: {(result['avg_store_count'] < 5).sum()}개")
    print(f"    → 마스터 테이블 단계에서 imputed 보정 권장 (서울 평균 또는 별도 처리)")


if __name__ == "__main__":
    # 실행 예시:
    # API 모드:
    #   python opening_rate.py --mode api --api-key YOUR_KEY --save-raw
    # CSV 모드 (수동 다운로드):
    #   python opening_rate.py --mode csv \
    #       --csv-paths data/raw/opening/2023.csv data/raw/opening/2024.csv data/raw/opening/2025.csv
    pass
