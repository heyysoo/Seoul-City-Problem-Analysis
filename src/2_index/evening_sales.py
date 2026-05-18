"""저녁 매출 절대값 (Y축, 저녁 트랙 메인 — OA-22175 기반)

처리 목적
- 저녁 트랙 2*2 매트릭스의 메인 Y축 변수 evening_sales_norm 산출.
- 도메인 의미: "저녁(17~21시)에 이미 얼마나 활성화되었는가" (절대 매출 수준).
- 심야 트랙의 매출 미실현도(gap_norm)와 정반대 개념.

데이터 출처
- 서울시 상권분석서비스 (추정매출-행정동) — OA-22175
- OpenAPI 서비스명: VwsmAdstrdSelngW
- 17~21시 매출 전용 컬럼 TMZON_17_21_SELNG_AMT 제공 (별도 시간 슬라이싱 불필요)
- B079 카드매출은 캠퍼스 외부 반출 불가이므로 외부 OA-22175로 대체

처리 단계
1. OpenAPI 페이지네이션 호출 (또는 --from-csv 로 기존 raw csv 로드)
2. 24개 야간 부적합 업종 제외 (외식·소매·여가 핵심 업종만 유지)
3. 동명 충돌 분리: 신사동 강남·관악 두 곳을 자치구 코드로 분리
4. 행정동*17-21시 매출 합계 (groupby)
5. 행자부 동명 매칭 + 상일1·2동 분리 처리 + 일원2동 통합 폐지 처리
6. PercentileRank 정규화 → evening_sales_norm ∈ [0, 1]

산출물 (out_dir 하위)
- evening_sales_raw_by_dong.csv: OA-22175 원본 그룹바이 결과 (425행, 신사동 분리 반영)
- evening_sales_final.csv: 마스터 JOIN용 최종 (426행, 행자부 코드, evening_sales_norm)

마스터 테이블 결측 처리 (3_matrix.py에서 처리)
- 개포3동(11230511): OA-22175에 부재 → 중앙값 대체 + is_evening_imputed 플래그 + Grey out
- 일원2동: 행자부 통합 폐지로 자연 제외

학술 정당화 (보고서 §6 카드)
- PercentileRank 채택: 노량진2동 9,925억 등 극단값으로 Min-Max 시 분포 압축 → 순위 정보 보존
- 24개 업종 제외: 교육·의료·자동차 등 저녁 소비와 무관한 업종 (도메인 정확성)
- B079 미사용: 캠퍼스 반출 정책 + OA-22175가 17-21 전용 컬럼 제공 → 정합성↑
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

DEFAULT_SERVICE = "VwsmAdstrdSelngW"  # OA-22175 OpenAPI 서비스명
DEFAULT_PAGE_SIZE = 1000
DEFAULT_SLEEP_SEC = 0.5

# 17-21시 매출 컬럼 (OA-22175 전용 컬럼)
COL_AMT_17_21 = "TMZON_17_21_SELNG_AMT"

# 원본 컬럼 (페이지네이션 결과)
KEEP_COLS = [
    "STDR_YYQU_CD",       # 분기 코드 (YYYYQ)
    "ADSTRD_CD",          # 행정동 코드 (10자리, OA 통계청 코드)
    "ADSTRD_CD_NM",       # 행정동명
    "SVC_INDUTY_CD",      # 업종 코드 (CS-)
    "SVC_INDUTY_CD_NM",   # 업종명
    COL_AMT_17_21,        # 17~21시 매출
]

# 24개 야간 부적합 업종 (외식·소매·여가 무관 — 도메인 판단)
# 분석결과서 페이지 30 [카드 매출 야간 업종 필터링 로직] 박스 1·2 정통:
#   박스 1 (시점 불일치 / 구분 불가): 교육/학원·숙박/여행/교통
#   박스 2 (야간 정책 적용 부적합): 자동차·가전·가구·의료·생활/업무 서비스
#   박스 3 (예외 포함): 세탁소·주차장·택시·면세점 → EXCLUDE에 없음 (자동 포함)
# b079.py의 EXCLUDE_CATEGORIES와 카테고리 정합성 동일 (OA-22175에는 요금/금융·
# 주유 코드가 없어서 자연스럽게 EXCLUDE에 미포함).
# 노션 정통 (자료실 "카드 데이터 업종 제외 로직과 결과" §2 OA-22175 EXCLUDE_CODES).
# CS 코드 기반 정확한 매칭 (업종명 부분일치로 인한 오탐 방지).
EXCLUDE_CODES = {
    # 교육·학원 (4)
    "CS200001": "일반교습학원",
    "CS200002": "외국어학원",
    "CS200003": "예술학원",
    "CS200005": "스포츠 강습",
    # 의료 (5)
    "CS200006": "일반의원",
    "CS200007": "치과의원",
    "CS200008": "한의원",
    "CS300018": "의약품",
    "CS300019": "의료기기",
    # 자동차 (2)
    "CS200025": "자동차수리",
    "CS200026": "자동차미용",
    # 생활·업무 (2) — 세탁소 예외 포함
    "CS200032": "가전제품수리",
    "CS200033": "부동산중개업",
    # 숙박·장기거주 (2)
    "CS200034": "여관",
    "CS200036": "고시원",
    # 가전·가구 (8)
    "CS300003": "컴퓨터및주변장치판매",
    "CS300004": "핸드폰",
    "CS300031": "가구",
    "CS300032": "가전제품",
    "CS300033": "철물점",
    "CS300035": "인테리어",
    "CS300036": "조명용품",
    "CS300043": "전자상거래업",
}

# 신사동 두 곳 자치구 분리 (OA 통계청 코드 기준)
SINSA_CODE_GANGNAM = "11680510"   # 가로수길 (강남구)
SINSA_CODE_GWANAK = "11620685"    # 서울대 인근 (관악구)


# -----------------------------------------------------------------------------
# 1. OpenAPI 호출
# -----------------------------------------------------------------------------

def fetch_oa22175(
    api_key: str,
    service_name: str = DEFAULT_SERVICE,
    page_size: int = DEFAULT_PAGE_SIZE,
    sleep_sec: float = DEFAULT_SLEEP_SEC,
) -> pd.DataFrame:
    """서울시 OA-22175 추정매출-행정동 전체 수집 (페이지네이션)."""
    base_url = f"http://openapi.seoul.go.kr:8088/{api_key}/json/{service_name}"

    # 전체 건수 확인
    res = requests.get(f"{base_url}/1/1", timeout=30)
    data = res.json()
    key = list(data.keys())[0]
    total = data[key]["list_total_count"]
    print(f"  전체 건수: {total:,}")

    all_rows: list[dict] = []
    for start in range(1, total + 1, page_size):
        end = min(start + page_size - 1, total)
        url = f"{base_url}/{start}/{end}"
        res = requests.get(url, timeout=60)
        data = res.json()
        rows = data[key].get("row", [])
        all_rows.extend(rows)
        print(f"  수집: {start}~{end} ({len(all_rows):,}/{total:,})")
        time.sleep(sleep_sec)

    df = pd.DataFrame(all_rows)
    # 필요 컬럼만 유지 (없으면 KeyError 회피)
    keep = [c for c in KEEP_COLS if c in df.columns]
    return df[keep]


# -----------------------------------------------------------------------------
# 2. 업종 필터 + 신사동 분리 + groupby
# -----------------------------------------------------------------------------

def filter_industries(df: pd.DataFrame) -> pd.DataFrame:
    """24개 야간 부적합 업종 제외 (CS 코드 기반 정확 매칭, 노션 정통).

    EXCLUDE_CODES dict 의 24개 코드와 정확히 일치하는 행 제거.
    업종명 부분일치보다 안전 (예: '여관' 키워드가 '여관 한식' 같은 가짜 매칭 방지).
    """
    if "SVC_INDUTY_CD" not in df.columns:
        raise KeyError(
            "SVC_INDUTY_CD 컬럼이 필요합니다 (CS 코드 기반 필터링).\n"
            f"현재 컬럼: {df.columns.tolist()}"
        )

    code = df["SVC_INDUTY_CD"].astype(str)
    mask = code.isin(EXCLUDE_CODES.keys())

    before = len(df)
    out = df[~mask].copy()
    print(f"  업종 필터 (CS 코드): {before:,} → {len(out):,} 행 ({before - len(out):,} 제외)")

    # 검증 로그: 어떤 코드가 어느 정도 제외됐는지
    if mask.sum() > 0:
        excluded = (
            df.loc[mask, ["SVC_INDUTY_CD", "SVC_INDUTY_CD_NM"]]
            .value_counts()
            .head(10)
        )
        print("  제외 분포 (상위 10):")
        for (cd, nm), n in excluded.items():
            print(f"    {cd:10s} {nm:25s}: {n:,}")
    return out


def split_sinsa_dong(df: pd.DataFrame) -> pd.DataFrame:
    """신사동 강남·관악 두 곳을 자치구 코드로 분리."""
    out = df.copy()
    out["ADSTRD_CD"] = out["ADSTRD_CD"].astype(str)
    out.loc[out["ADSTRD_CD"] == SINSA_CODE_GANGNAM, "ADSTRD_CD_NM"] = "신사(강남구)"
    out.loc[out["ADSTRD_CD"] == SINSA_CODE_GWANAK, "ADSTRD_CD_NM"] = "신사(관악구)"
    return out


def aggregate_by_dong(df: pd.DataFrame) -> pd.DataFrame:
    """행정동×17-21시 매출 합계 (전 분기 통합)."""
    out = df.copy()
    out[COL_AMT_17_21] = pd.to_numeric(out[COL_AMT_17_21], errors="coerce").fillna(0)

    agg = out.groupby("ADSTRD_CD_NM", as_index=False).agg(
        evening_sales_amt=(COL_AMT_17_21, "sum"),
        ADSTRD_CD_first=("ADSTRD_CD", "first"),  # 추적용
    )
    agg = agg.rename(columns={"ADSTRD_CD_first": "ADSTRD_CD_sample"})
    return agg.sort_values("evening_sales_amt", ascending=False).reset_index(drop=True)


# -----------------------------------------------------------------------------
# 3. 행자부 코드 매칭 + 분리/통합 처리
# -----------------------------------------------------------------------------

# 신사동 수동 매핑 (OA 통계청 코드 → 행자부 코드)
SINSA_MAPPING = {
    "신사(강남구)": {"행자부_코드": "11230510", "행자부_행정동명": "신사동", "gu_name": "강남구"},
    "신사(관악구)": {"행자부_코드": "11210680", "행자부_행정동명": "신사동", "gu_name": "관악구"},
}

# 상일동 분리 (OA의 상일동 → 행자부 상일1·2동에 동일값 분배)
SANGIL_SPLIT = (
    ("11250760", "상일1동"),
    ("11250770", "상일2동"),
)

# 일원2동: 행자부 통합 폐지 (OA에는 있으나 행자부 코드 없음)
ILWON_2_NAME = "일원2동"


def normalize_dong_name(name: str) -> str:
    """동명 정규화 (앞뒤 공백·중점 통일)."""
    if pd.isna(name):
        return ""
    return str(name).strip().replace("·", ".")


def load_dong_master_from_shp(shp_path: Path, encoding: str = "cp949") -> pd.DataFrame:
    """raw bnd_dong SHP에서 행정동 마스터 추출 (geopandas attribute만 사용).

    SHP attribute 컬럼: ADM_CD (행자부 코드, 8자리), ADM_NM (동명)
    gu_name 은 ADM_NM 첫 부분에서 추출 (서울 명명 규칙).
    """
    import geopandas as gpd
    gdf = gpd.read_file(shp_path, encoding=encoding)
    master = gdf[["ADM_CD", "ADM_NM"]].drop_duplicates().copy()
    master["amd_code"] = master["ADM_CD"].astype(str).str.zfill(8)
    master["dong_name"] = master["ADM_NM"].astype(str).str.strip()
    # 자치구명: ADM_CD 앞 5자리 → 자치구 코드 → 매핑 또는 ADM_NM에서 추출 시도
    # 간단 접근: 자치구 코드를 별도 매핑으로 직접 산출
    GU_CODE_MAP = {
        "11110": "종로구", "11140": "중구", "11170": "용산구", "11200": "성동구",
        "11215": "광진구", "11230": "동대문구", "11260": "중랑구", "11290": "성북구",
        "11305": "강북구", "11320": "도봉구", "11350": "노원구", "11380": "은평구",
        "11410": "서대문구", "11440": "마포구", "11470": "양천구", "11500": "강서구",
        "11530": "구로구", "11545": "금천구", "11560": "영등포구", "11590": "동작구",
        "11620": "관악구", "11650": "서초구", "11680": "강남구", "11710": "송파구",
        "11740": "강동구",
    }
    master["gu_code"] = master["amd_code"].str[:5]
    master["gu_name"] = master["gu_code"].map(GU_CODE_MAP)
    return master[["amd_code", "dong_name", "gu_name"]]


def match_with_infra(
    agg: pd.DataFrame,
    infra_path: Path,
    infra_name_col: str = "dong_name",
    infra_code_col: str = "amd_code",
    infra_gu_col: str = "gu_name",
) -> pd.DataFrame:
    """행정동 마스터의 행자부 동명과 매칭. 신사동·상일동·일원2동 특수 케이스 분기.

    입력 경로 확장자에 따라 모드 자동 결정:
    - .shp → raw bnd_dong SHP 에서 attribute 만 추출
    - .csv → 사전 산출된 행정동 마스터 csv (이전 버전 호환)
    """
    if str(infra_path).lower().endswith(".shp"):
        infra = load_dong_master_from_shp(infra_path)
    else:
        infra = pd.read_csv(infra_path, dtype=str)
    required = [infra_name_col, infra_code_col, infra_gu_col]
    missing = [c for c in required if c not in infra.columns]
    if missing:
        raise KeyError(
            f"행정동 마스터 ({infra_path.name})에 필수 컬럼 없음: {missing}\n"
            f"현재 컬럼: {infra.columns.tolist()}"
        )

    infra["_name_norm"] = infra[infra_name_col].map(normalize_dong_name)
    agg = agg.copy()
    agg["_name_norm"] = agg["ADSTRD_CD_NM"].map(normalize_dong_name)

    rows: list[dict] = []

    for _, row in agg.iterrows():
        oa_name = row["ADSTRD_CD_NM"]
        amt = row["evening_sales_amt"]

        # Case 1: 신사동 수동 매핑
        if oa_name in SINSA_MAPPING:
            m = SINSA_MAPPING[oa_name]
            rows.append({
                "원본_행정동명": oa_name,
                "행자부_코드": m["행자부_코드"],
                "행자부_행정동명": m["행자부_행정동명"],
                "gu_name": m["gu_name"],
                "evening_sales_amt": amt,
                "matching_method": "신사동_수동매핑",
            })
            continue

        # Case 2: 상일동 분리 (동일값 분배)
        if oa_name.strip() == "상일동":
            for code, name in SANGIL_SPLIT:
                rows.append({
                    "원본_행정동명": oa_name,
                    "행자부_코드": code,
                    "행자부_행정동명": name,
                    "gu_name": "강동구",
                    "evening_sales_amt": amt,  # 동일값 분배 (분할 아님)
                    "matching_method": "행자부_분리_동일값분배",
                })
            continue

        # Case 3: 일원2동 통합 폐지
        if oa_name.strip() == ILWON_2_NAME:
            rows.append({
                "원본_행정동명": oa_name,
                "행자부_코드": None,
                "행자부_행정동명": None,
                "gu_name": None,
                "evening_sales_amt": amt,
                "matching_method": "행자부_통합폐지",
            })
            continue

        # Case 4: 일반 동명 매칭
        match = infra[infra["_name_norm"] == row["_name_norm"]]
        if len(match) == 1:
            m = match.iloc[0]
            rows.append({
                "원본_행정동명": oa_name,
                "행자부_코드": m[infra_code_col],
                "행자부_행정동명": m[infra_name_col],
                "gu_name": m[infra_gu_col],
                "evening_sales_amt": amt,
                "matching_method": "동명_매칭_성공",
            })
        elif len(match) > 1:
            # 다중 매칭: gu_name으로 disambiguation 필요한 경우
            rows.append({
                "원본_행정동명": oa_name,
                "행자부_코드": None,
                "행자부_행정동명": None,
                "gu_name": None,
                "evening_sales_amt": amt,
                "matching_method": f"다중매칭_{len(match)}건_확인필요",
            })
        else:
            rows.append({
                "원본_행정동명": oa_name,
                "행자부_코드": None,
                "행자부_행정동명": None,
                "gu_name": None,
                "evening_sales_amt": amt,
                "matching_method": "매칭실패",
            })

    out = pd.DataFrame(rows)
    print(f"  매칭 결과 분포: {out['matching_method'].value_counts().to_dict()}")
    return out


# -----------------------------------------------------------------------------
# 4. PercentileRank 정규화
# -----------------------------------------------------------------------------

def add_percentile_rank(df: pd.DataFrame) -> pd.DataFrame:
    """행자부 코드가 있는 행에 대해서만 PercentileRank 정규화."""
    out = df.copy()
    matched_mask = out["행자부_코드"].notna()
    out["evening_sales_norm"] = None
    out.loc[matched_mask, "evening_sales_norm"] = (
        out.loc[matched_mask, "evening_sales_amt"].rank(pct=True)
    )
    return out


# -----------------------------------------------------------------------------
# 진입점
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="저녁 매출 절대값 산출 (OA-22175)")

    # 데이터 소스 (둘 중 하나)
    parser.add_argument(
        "--api-key", type=str, default=os.environ.get("SEOUL_OPENAPI_KEY", ""),
        help="서울시 OpenAPI 키. 환경변수 SEOUL_OPENAPI_KEY 도 가능.",
    )
    parser.add_argument("--service-name", type=str, default=DEFAULT_SERVICE)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument(
        "--from-csv", type=Path, default=None,
        help="이미 받은 OA-22175 raw csv 경로. 지정 시 API 호출 생략.",
    )

    # 행정동 마스터 (raw bnd_dong SHP 또는 사전 산출 csv)
    parser.add_argument(
        "--infra-master", type=Path,
        default=Path("data/raw/bnd_dong_11_2025_2Q/bnd_dong_11_2025_2Q.shp"),
        help="행정동 코드·동명·자치구 마스터. "
             ".shp(권장, raw에서 직접 시작) 또는 .csv (사전 산출물) 모두 허용.",
    )
    parser.add_argument("--infra-name-col", type=str, default="dong_name")
    parser.add_argument("--infra-code-col", type=str, default="amd_code")
    parser.add_argument("--infra-gu-col", type=str, default="gu_name")

    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("data/processed/evening_sales"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 원본 데이터 수집
    if args.from_csv is not None:
        print(f"[1] 기존 csv 로드: {args.from_csv}")
        df = pd.read_csv(args.from_csv, dtype=str)
    else:
        if not args.api_key:
            raise SystemExit(
                "API 키가 비어 있습니다. --api-key 또는 환경변수 SEOUL_OPENAPI_KEY 설정 필요.\n"
                "또는 --from-csv 로 기존 raw csv 사용."
            )
        print("[1] OA-22175 OpenAPI 호출")
        df = fetch_oa22175(args.api_key, args.service_name, args.page_size)
        raw_path = args.out_dir / "oa22175_raw.csv"
        df.to_csv(raw_path, index=False, encoding="utf-8-sig")
        print(f"  원본 저장: {raw_path} ({len(df):,} 행)")

    print("\n[2] 업종 필터 (24개 야간 부적합 업종 제외)")
    df = filter_industries(df)

    print("\n[3] 신사동 자치구 분리")
    df = split_sinsa_dong(df)

    print("\n[4] 행정동×17-21시 매출 groupby")
    agg = aggregate_by_dong(df)
    raw_dong_path = args.out_dir / "evening_sales_raw_by_dong.csv"
    agg.to_csv(raw_dong_path, index=False, encoding="utf-8-sig")
    print(f"  저장: {raw_dong_path} ({len(agg)} 행)")

    print("\n[5] 행자부 코드 매칭")
    matched = match_with_infra(
        agg, args.infra_master,
        infra_name_col=args.infra_name_col,
        infra_code_col=args.infra_code_col,
        infra_gu_col=args.infra_gu_col,
    )

    print("\n[6] PercentileRank 정규화")
    final = add_percentile_rank(matched)

    final_path = args.out_dir / "evening_sales_final.csv"
    final.to_csv(final_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {final_path} ({len(final)} 행, 매칭 {final['행자부_코드'].notna().sum()}개)")


if __name__ == "__main__":
    # 실행 예시 (필요 시 주석 해제):
    # import sys
    # sys.argv = [
    #     "evening_sales.py",
    #     "--api-key", "YOUR_KEY_HERE",
    #     "--infra-master", "data/processed/infra_gap/infrastructure_gap_final.csv",
    #     "--out-dir", "data/processed/evening_sales",
    # ]
    # main()
    pass
