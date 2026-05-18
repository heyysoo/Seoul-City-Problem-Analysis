"""보안등 4구 geocoding 보강 (노션 정통 12단계 중 Step 5-7)

처리 목적
- 좌표 없는 4구 (동대문·마포·송파·용산)의 보안등 데이터를 주소 기반 geocoding으로 보강.
- light_void.py의 `--geocoded-supplement` 입력으로 산출.

- Step 5: 4구 주소 기반 geocoding (동대문·마포·송파·용산)
- Step 6: 용산구 주소 정규화 보완 (geocoding 성공률 향상)
- Step 7: 4구 행정동 집계 → light_void.py 입력 형식

처리 흐름
1. 좌표 없는 4구 raw csv 로드
2. 주소 정규화 (특히 용산구: "한남대로 N길" 등 도로명 정규화)
3. geocoding API 호출 (카카오 또는 네이버 로컬 API)
4. 위경도 → 행정동 SHP spatial join
5. 4구 행정동별 집계 → light_void.py 입력 csv 산출

입력
- 4구 raw csv (또는 25구 통합 csv에서 4구 필터링)
- 행정동 SHP (bnd_dong_11_2025_2Q.shp)
- geocoding API 키 (환경변수 또는 인자)

산출
- 서울시_보안등_좌표0개구_행정동매핑_용산반영.csv
  (light_void.py의 --geocoded-supplement 입력)
- 용산구_geocode_final.csv (용산구 보완 산출, 검증용)

API 선택지
- 카카오 로컬 API: 무료 한도 일 300,000회, 정확도 높음
- 네이버 Geocoding API: 일 100,000회 무료
- 두 API 모두 도로명·지번 주소 입력 → 위경도 반환

주의사항
- API 키는 환경변수(KAKAO_API_KEY 또는 NAVER_CLIENT_ID/SECRET)로 받음
- API 호출 횟수 제한 → time.sleep + 배치 처리
- 일부 주소는 API 실패 가능 → fallback (수동 매핑 또는 동명 매칭)
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False


# -----------------------------------------------------------------------------
# 상수
# -----------------------------------------------------------------------------

# Step 5 대상 4구
GEOCODE_TARGET_GUS = ("동대문구", "마포구", "송파구", "용산구")

# Geocoding API
KAKAO_BASE_URL = "https://dapi.kakao.com/v2/local/search/address.json"
DEFAULT_SLEEP_SEC = 0.05  # API 호출 간 sleep (rate limit 회피)
MAX_RETRIES = 3


# -----------------------------------------------------------------------------
# Step 5a — 주소 정규화 (특히 용산구)
# -----------------------------------------------------------------------------

def normalize_address(addr: str, gu_name: str = None) -> str:
    """주소 정규화 (Step 6: 용산구 보완 핵심).

    공통:
    - 양끝 공백 제거
    - 다중 공백 → 단일 공백
    - "서울특별시" → "서울" (API에 따라 다름)

    용산구 특화:
    - "한남대로 N길" 형식 통일
    - 번지수 추출 정리
    """
    if pd.isna(addr) or not addr:
        return ""

    s = str(addr).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("서울특별시", "서울시")

    # 용산구: 도로명 정규화
    if gu_name == "용산구":
        # "한남대로N가길" → "한남대로 N가길" (공백 분리)
        s = re.sub(r"(대로|로|길)(\d)", r"\1 \2", s)
        # 번지수 정규화 (마지막 숫자 패턴 정리)
        s = re.sub(r"(\d+)\s*[-]\s*(\d+)", r"\1-\2", s)

    return s


# -----------------------------------------------------------------------------
# Step 5b — Kakao Local API geocoding
# -----------------------------------------------------------------------------

def geocode_address_kakao(
    addr: str,
    api_key: str,
    sleep_sec: float = DEFAULT_SLEEP_SEC,
    max_retries: int = MAX_RETRIES,
) -> tuple[float, float] | tuple[None, None]:
    """카카오 로컬 API로 주소 → 위경도 변환.

    Returns: (lat, lon) 또는 (None, None) (실패 시)
    """
    if not addr:
        return None, None

    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {"query": addr}

    for attempt in range(max_retries):
        try:
            resp = requests.get(KAKAO_BASE_URL, headers=headers, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                docs = data.get("documents", [])
                if docs:
                    doc = docs[0]
                    return float(doc["y"]), float(doc["x"])  # y=lat, x=lon
                return None, None
            elif resp.status_code == 429:  # rate limit
                time.sleep(1.0)
                continue
            else:
                return None, None
        except (requests.RequestException, ValueError):
            time.sleep(sleep_sec * (attempt + 1))

    return None, None


# -----------------------------------------------------------------------------
# Step 5c — 4구 raw csv → geocoded
# -----------------------------------------------------------------------------

def geocode_4gu(
    df: pd.DataFrame,
    api_key: str,
    addr_col: str = "주소",
    gu_col: str = "자치구",
    sleep_sec: float = DEFAULT_SLEEP_SEC,
) -> pd.DataFrame:
    """4구 raw csv → 행별 geocoding 적용."""
    out = df.copy()
    out["주소_정규화"] = out.apply(
        lambda r: normalize_address(r.get(addr_col, ""), r.get(gu_col, "")),
        axis=1,
    )

    n_total = len(out)
    lats, lons = [], []
    n_success = 0
    for i, addr in enumerate(out["주소_정규화"]):
        lat, lon = geocode_address_kakao(addr, api_key, sleep_sec=sleep_sec)
        lats.append(lat)
        lons.append(lon)
        if lat is not None:
            n_success += 1
        if (i + 1) % 500 == 0:
            print(f"  진행: {i+1}/{n_total} (성공률: {n_success/(i+1)*100:.1f}%)")
        time.sleep(sleep_sec)

    out["lat_geocoded"] = lats
    out["lon_geocoded"] = lons
    print(f"  최종 geocoding 성공: {n_success}/{n_total} ({n_success/n_total*100:.1f}%)")
    return out


# -----------------------------------------------------------------------------
# Step 7 — 4구 geocoded → 행정동 매핑 + 집계
# -----------------------------------------------------------------------------

def map_geocoded_to_dong(
    df: pd.DataFrame,
    dong_shp: Path,
    encoding: str = "cp949",
    target_crs: str = "EPSG:5179",
) -> pd.DataFrame:
    """geocoded 4구 데이터 → 행정동 spatial join."""
    if not HAS_GEOPANDAS:
        raise ImportError("geopandas 필요")

    valid = df.dropna(subset=["lat_geocoded", "lon_geocoded"]).copy()
    print(f"  spatial join 대상: {len(valid)}건 (geocoding 성공만)")

    geom = [Point(lon, lat) for lat, lon in zip(valid["lat_geocoded"], valid["lon_geocoded"])]
    points = gpd.GeoDataFrame(valid, geometry=geom, crs="EPSG:4326").to_crs(target_crs)

    dong = gpd.read_file(dong_shp, encoding=encoding).to_crs(target_crs)
    dong["ADM_CD"] = dong["ADM_CD"].astype(str).str.zfill(8)

    joined = gpd.sjoin(
        points, dong[["ADM_CD", "ADM_NM", "geometry"]],
        how="left", predicate="within",
    )

    # nearest fallback
    unmatched = joined[joined["ADM_CD"].isna()].index
    if len(unmatched) > 0:
        print(f"  within 매핑 실패 {len(unmatched)}건 → nearest fallback")
        near = gpd.sjoin_nearest(
            points.loc[unmatched], dong, how="left", distance_col="dist_m",
        )[["ADM_CD", "ADM_NM"]]
        near = near[~near.index.duplicated(keep="first")]
        joined.loc[unmatched, "ADM_CD"] = near["ADM_CD"]
        joined.loc[unmatched, "ADM_NM"] = near["ADM_NM"]

    return joined.drop(columns=["geometry", "index_right"], errors="ignore")


# -----------------------------------------------------------------------------
# 진입점
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="보안등 4구 geocoding 보강 (노션 Step 5-7)")
    parser.add_argument(
        "--input-csv", type=Path,
        default=Path("data/raw/streetlight/raw_gu/서울시_보안등_4개구_좌표없음.csv"),
        help="좌표 없는 4구 보안등 raw csv (동대문·마포·송파·용산)",
    )
    parser.add_argument("--addr-col", type=str, default="주소")
    parser.add_argument("--gu-col", type=str, default="자치구")
    parser.add_argument(
        "--api-key", type=str,
        default=os.environ.get("KAKAO_API_KEY", ""),
        help="카카오 로컬 API 키 (환경변수 KAKAO_API_KEY 도 가능)",
    )
    parser.add_argument(
        "--dong-shp", type=Path,
        default=Path("data/raw/bnd_dong_11_2025_2Q/bnd_dong_11_2025_2Q.shp"),
    )
    parser.add_argument("--sleep-sec", type=float, default=DEFAULT_SLEEP_SEC)
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("data/processed/streetlight_geocode"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.api_key:
        raise SystemExit(
            "API 키가 비어 있습니다. --api-key 또는 환경변수 KAKAO_API_KEY 설정 필요."
        )

    print("[1] 4구 좌표 없는 보안등 raw csv 로드")
    df = pd.read_csv(args.input_csv, dtype=str, low_memory=False)
    print(f"  총 {len(df)}건 (4구: {args.input_csv.name})")

    print("\n[2] 주소 정규화 (Step 6: 용산구 포함)")
    print("\n[3] Kakao Local API geocoding 호출 (Step 5)")
    geocoded = geocode_4gu(
        df, args.api_key,
        addr_col=args.addr_col, gu_col=args.gu_col,
        sleep_sec=args.sleep_sec,
    )

    # 중간 산출 — 용산구 보완 검증용
    yongsan = geocoded[geocoded[args.gu_col] == "용산구"]
    if len(yongsan) > 0:
        yongsan_path = args.out_dir / "용산구_geocode_final.csv"
        yongsan.to_csv(yongsan_path, index=False, encoding="utf-8-sig")
        print(f"\n  용산구 보완 산출: {yongsan_path} ({len(yongsan)}건)")

    print("\n[4] 행정동 spatial join (Step 7)")
    joined = map_geocoded_to_dong(geocoded, args.dong_shp)

    out_path = args.out_dir / "서울시_보안등_좌표0개구_행정동매핑_용산반영.csv"
    joined.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_path} ({len(joined)}건)")
    print(f"      → light_void.py의 --geocoded-supplement 입력으로 사용")


if __name__ == "__main__":
    # 실행 예시 (필요 시 주석 해제):
    # import sys
    # sys.argv = [
    #     "light_geocode.py",
    #     "--input-csv", "data/raw/streetlight/raw_gu/서울시_보안등_4개구_좌표없음.csv",
    #     "--api-key", "<KAKAO_API_KEY>",
    #     "--dong-shp", "data/raw/bnd_dong_11_2025_2Q/bnd_dong_11_2025_2Q.shp",
    #     "--out-dir", "data/processed/streetlight_geocode",
    # ]
    # main()
    pass
