"""범죄데이터 자치구 맥락변수 처리 (옛 v1 9.1 시간대 결합 방식)

처리 목적
- 두 데이터 결합으로 자치구별 야간 강력+폭력 추정치 산출:
  ① 전국_범죄발생지.csv: 자치구별 범죄 대분류 건수
  ② 전국_범죄발생시간.csv: 전국 범죄 유형별 시간대 분포
- 자치구별 강력 * 야간_강력_비율 + 폭력 * 야간_폭력_비율 = 자치구 야간 추정
- 25구 rank percentile + 행정동에 자치구 분위수 동일 부여

처리 흐름 (옛 v1 9.1 정통)
1. 전국_범죄발생지 raw → 서울 25구 대분류별 건수 (강력·폭력 별도)
2. 전국_범죄발생시간 raw → 강력·폭력 야간 시간대 비율 산출
   야간 = 18-21 + 21-24 + 00-03 + 03-06 (4개 시간대, 12시간)
3. 자치구 야간 추정 = 강력 * 야간_강력_비율 + 폭력 * 야간_폭력_비율
4. 자치구별 야간 분위수 (rank percentile, 25개 구 기준, 연도별)
5. 행정동에 자치구 야간 분위수 동일 부여 (Area Proportional Assignment)

가정 (한계 명시)
- 서울 자치구별 야간 범죄 비율이 전국 평균과 유사하다고 가정
- 시간대 데이터에 자치구 분리 없음 → 전국 비율 일괄 적용

해석 주의
- 본 결과는 자치구별 야간 강력+폭력 추정 분위수.
- 같은 자치구 안의 모든 행정동에 같은 야간 분위수 부여 (동 단위 변별 X).

산출물
- seoul_gu_crime_context_2023_2024.csv (구 단위, 전체 및 야간 추정)
- seoul_dong_crime_context_2023_2024.csv (동 단위 부여)
- 핵심 컬럼: 범죄맥락값_권장(전체 강력+폭력), 범죄맥락분위수_권장(전체 분위수),
            야간_범죄맥락값_권장(야간 추정 강력+폭력),
            야간_범죄맥락분위수_권장(야간 추정 분위수) ← 본 분석 메인 사용
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import pandas as pd


ENCODINGS = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]


def read_csv_auto(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"CSV 읽기 실패: {path}") from last_error


def resolve_existing_path(path: Path) -> Path:
    """macOS 한글 파일명의 NFC/NFD 차이를 흡수한다."""
    if path.exists():
        return path

    parent = path.parent if str(path.parent) else Path(".")
    target_name = unicodedata.normalize("NFC", path.name)
    if parent.exists():
        for candidate in parent.iterdir():
            if unicodedata.normalize("NFC", candidate.name) == target_name:
                return candidate

    matches = []
    for candidate in Path(".").rglob("*"):
        if unicodedata.normalize("NFC", candidate.name) == target_name:
            matches.append(candidate)
    if matches:
        return sorted(matches, key=lambda p: len(str(p)))[0]

    return path


def build_seoul_gu_crime_context(crime_place_path: Path) -> pd.DataFrame:
    crime_place_path = resolve_existing_path(crime_place_path)
    raw = read_csv_auto(crime_place_path)
    metro_row = raw.iloc[1]
    gu_row = raw.iloc[2]

    body = raw.iloc[3:].reset_index(drop=True).copy()
    body = body.rename(columns={"죄종별(1)": "대분류", "죄종별(2)": "소분류"})

    seoul_cols_2023 = [col for col in raw.columns[2:102] if metro_row[col] == "서울"]
    seoul_cols_2024 = [col for col in raw.columns[102:] if metro_row[col] == "서울"]
    seoul_map_2023 = {col: gu_row[col] for col in seoul_cols_2023}
    seoul_map_2024 = {col: gu_row[col] for col in seoul_cols_2024}

    for col in seoul_cols_2023 + seoul_cols_2024:
        body[col] = pd.to_numeric(body[col].replace("-", 0), errors="coerce").fillna(0)

    crime_2023 = body[["대분류", "소분류"] + seoul_cols_2023].melt(
        id_vars=["대분류", "소분류"],
        var_name="원본컬럼",
        value_name="발생건수",
    )
    crime_2023["연도"] = 2023
    crime_2023["구명"] = crime_2023["원본컬럼"].map(seoul_map_2023)

    crime_2024 = body[["대분류", "소분류"] + seoul_cols_2024].melt(
        id_vars=["대분류", "소분류"],
        var_name="원본컬럼",
        value_name="발생건수",
    )
    crime_2024["연도"] = 2024
    crime_2024["구명"] = crime_2024["원본컬럼"].map(seoul_map_2024)

    crime_long = pd.concat([crime_2023, crime_2024], ignore_index=True)
    gu_by_major = crime_long.groupby(["연도", "구명", "대분류"], as_index=False)["발생건수"].sum()
    gu_wide = gu_by_major.pivot_table(
        index=["연도", "구명"],
        columns="대분류",
        values="발생건수",
        fill_value=0,
    ).reset_index()
    gu_wide.columns.name = None

    for col in ["강력범죄", "폭력범죄", "풍속범죄", "지능범죄"]:
        if col not in gu_wide.columns:
            gu_wide[col] = 0

    gu_wide["강력폭력합계"] = gu_wide["강력범죄"] + gu_wide["폭력범죄"]
    gu_wide["강력폭력풍속합계"] = gu_wide["강력범죄"] + gu_wide["폭력범죄"] + gu_wide["풍속범죄"]
    gu_wide["범죄맥락값_권장"] = gu_wide["강력폭력합계"]
    gu_wide["범죄맥락분위수_권장"] = gu_wide.groupby("연도")["범죄맥락값_권장"].rank(pct=True)

    return gu_wide.sort_values(
        ["연도", "범죄맥락값_권장"], ascending=[True, False]
    ).reset_index(drop=True)


# -----------------------------------------------------------------------------
# 옛 v1 9.1 시간대 결합 — 야간 범죄 추정
# -----------------------------------------------------------------------------

# 야간 시간대 매핑 (.csv 컬럼 → 시간대)
# 전국_범죄발생시간.csv 컬럼 구조: 죄종별(1)·죄종별(2) + 2023.1~8 + 2024.1~8
# .1=00-03, .2=03-06, .3=06-09, .4=09-12, .5=12-15, .6=15-18, .7=18-21, .8=21-24
# 야간(18-06) = .1 + .2 + .7 + .8 (4개 시간대, 12시간)
NIGHT_TIME_COLS = {
    2023: {
        "00~03": "2023.1", "03~06": "2023.2",
        "18~21": "2023.7", "21~00": "2023.8",
    },
    2024: {
        "00~03": "2024.1", "03~06": "2024.2",
        "18~21": "2024.7", "21~00": "2024.8",
    },
}
# 전체 8개 시간대 (24시간 / 3시간 단위)
ALL_TIME_COLS = {
    2023: [f"2023.{i}" for i in range(1, 9)],
    2024: [f"2024.{i}" for i in range(1, 9)],
}


def compute_night_ratio_by_category(crime_time_path: Path) -> pd.DataFrame:
    """전국_범죄발생시간.csv → 범죄 유형별 야간 비율 산출.

    야간 = 18-06 (12시간 = 4개 3시간 시간대): 18-21 + 21-24 + 00-03 + 03-06
    전체 = 24시간 (8개 3시간 시간대)

    범죄 유형별 야간 비율 = 야간 시간대 건수 합 / 전체 시간대 건수 합
    → 강력범죄·폭력범죄 각각 별도 비율 산출 (v1 9.1 정통)

    Returns:
        DataFrame[연도, 대분류, 야간_비율, 전체_건수, 야간_건수]
    """
    crime_time_path = resolve_existing_path(crime_time_path)
    df = read_csv_auto(crime_time_path)
    df = df.rename(columns={"죄종별(1)": "대분류", "죄종별(2)": "소분류"})

    # 강력범죄·폭력범죄 — 소분류 소계 제외 (세부만 합산)
    target_majors = ["강력범죄", "폭력범죄"]
    target = df[df["대분류"].isin(target_majors) & (df["소분류"] != "소계")].copy()

    results = []
    for year in [2023, 2024]:
        all_cols = [c for c in ALL_TIME_COLS[year] if c in target.columns]
        night_cols = [c for c in NIGHT_TIME_COLS[year].values() if c in target.columns]
        if not all_cols or not night_cols:
            print(f"  ⚠️ {year}년 시간대 컬럼 누락 — 야간 비율 산출 불가")
            continue

        # 숫자 변환
        for c in all_cols:
            target[c] = pd.to_numeric(target[c].replace("-", 0), errors="coerce").fillna(0)

        for major in target_majors:
            subset = target[target["대분류"] == major]
            if subset.empty:
                continue
            total = float(subset[all_cols].sum().sum())
            night = float(subset[night_cols].sum().sum())
            ratio = night / total if total > 0 else 0.0
            results.append({
                "연도": year,
                "대분류": major,
                "전체_건수": total,
                "야간_건수": night,
                "야간_비율": ratio,
            })

    out = pd.DataFrame(results)
    print(f"  야간 비율 (연도×대분류, n={len(out)}):")
    for _, r in out.iterrows():
        print(f"    {int(r['연도'])} {r['대분류']}: {r['야간_비율']*100:.1f}% "
              f"({int(r['야간_건수']):,}/{int(r['전체_건수']):,})")
    return out


def estimate_night_crime_by_gu(
    gu_context: pd.DataFrame,
    night_ratio: pd.DataFrame,
) -> pd.DataFrame:
    """자치구 야간 강력+폭력 추정 + 분위수 산출.

    v1 9.1 정통: 자치구 강력 × 야간_강력_비율 + 자치구 폭력 × 야간_폭력_비율
    """
    # pivot night_ratio: 연도×대분류 → wide
    ratio_wide = night_ratio.pivot_table(
        index="연도", columns="대분류", values="야간_비율"
    ).reset_index()
    ratio_wide.columns.name = None
    ratio_wide = ratio_wide.rename(
        columns={"강력범죄": "야간비율_강력", "폭력범죄": "야간비율_폭력"},
    )

    out = gu_context.merge(ratio_wide, on="연도", how="left")

    # 자치구 야간 강력 추정 = 자치구 강력 × 야간_강력_비율
    out["야간_강력_추정"] = out["강력범죄"] * out["야간비율_강력"]
    out["야간_폭력_추정"] = out["폭력범죄"] * out["야간비율_폭력"]
    out["야간_범죄맥락값_권장"] = out["야간_강력_추정"] + out["야간_폭력_추정"]

    # 연도별 25구 분위수 (rank percentile)
    out["야간_범죄맥락분위수_권장"] = out.groupby("연도")["야간_범죄맥락값_권장"].rank(pct=True)

    return out


def read_seoul_dong_master(region_code_path: Path, sheet_name: str, year: int) -> pd.DataFrame:
    region_code_path = resolve_existing_path(region_code_path)
    raw = pd.read_excel(region_code_path, sheet_name=sheet_name)
    columns = raw.iloc[0].tolist()
    df = raw.iloc[1:].copy()
    df.columns = columns

    df["시도코드"] = df["시도코드"].astype(str).str.zfill(2)
    df["시군구코드"] = df["시군구코드"].astype(str).str.zfill(3)
    df["읍면동코드"] = df["읍면동코드"].astype(str).str.zfill(3)

    seoul = df[df["시도명칭"] == "서울특별시"].copy()
    seoul["연도"] = year
    seoul["구코드"] = seoul["시도코드"] + seoul["시군구코드"]
    seoul["행정동코드"] = seoul["구코드"] + seoul["읍면동코드"]
    seoul["구명"] = seoul["시군구명칭"]
    seoul["행정동명"] = seoul["읍면동명칭"]
    return seoul[["연도", "구코드", "구명", "행정동코드", "행정동명"]].drop_duplicates()


def build_seoul_dong_crime_context(gu_context: pd.DataFrame, region_code_path: Path) -> pd.DataFrame:
    dong_2023 = read_seoul_dong_master(region_code_path, "2023년 12월", 2023)
    dong_2024 = read_seoul_dong_master(region_code_path, "2024년 6월", 2024)
    dong_master = pd.concat([dong_2023, dong_2024], ignore_index=True)

    use_cols = [
        "연도",
        "구명",
        "강력범죄",
        "폭력범죄",
        "풍속범죄",
        "지능범죄",
        "강력폭력합계",
        "강력폭력풍속합계",
        "범죄맥락값_권장",
        "범죄맥락분위수_권장",
        # v1 9.1 시간대 결합 추가 컬럼
        "야간비율_강력",
        "야간비율_폭력",
        "야간_강력_추정",
        "야간_폭력_추정",
        "야간_범죄맥락값_권장",
        "야간_범죄맥락분위수_권장",
    ]
    use_cols = [c for c in use_cols if c in gu_context.columns]
    merged = dong_master.merge(gu_context[use_cols], on=["연도", "구명"], how="left")
    return merged.sort_values(["연도", "구코드", "행정동코드"]).reset_index(drop=True)


def write_readme(out_dir: Path) -> None:
    text = """# 범죄 맥락변수 산출 설명 (v1 9.1 시간대 결합 정통)

## 사용 원본

- `전국_범죄발생지.csv` — 서울 25구 자치구별 범죄 대분류 건수
- `전국_범죄발생시간.csv` — 전국 범죄 유형별 시간대 분포 (8개 3시간 시간대)
- `센서스 공간정보 지역 코드.xlsx` — 행정동 매핑

## 처리 방식

1. 전국 범죄발생지 원자료의 다중 헤더에서 서울 25개 구 컬럼만 추출
2. 2023년, 2024년을 분리해 구 단위 범죄 대분류별 발생건수 집계
3. 전국 범죄발생시간 raw → 범죄 유형별 야간 비율 산출
   야간 = 18-21 + 21-24 + 00-03 + 03-06 (12시간)
   야간 비율 = 야간 시간대 합 / 전체 24시간 (범죄 유형별 별도)
4. v1 9.1 결합 — 자치구 야간 강력+폭력 추정:
   야간_강력_추정 = 자치구 강력 * 야간_비율_강력
   야간_폭력_추정 = 자치구 폭력 * 야간_비율_폭력
   야간_범죄맥락값_권장 = 야간_강력_추정 + 야간_폭력_추정
5. 연도별 자치구 야간 분위수(`야간_범죄맥락분위수_권장`) 계산 (25구 rank percentile)
6. 센서스 행정동 코드표에 자치구 야간 분위수 동일 부여

## 가정 (한계)

- 시간대 데이터는 전국 단위 → 서울 자치구별 야간 비율이 전국 평균과 유사하다고 가정
- 시간대 데이터 자치구 분리 불가 → 전국 비율 일괄 적용

## 산출 컬럼

- `야간_범죄맥락분위수_권장` (메인) — master_table.py crime_percentile 자동 매핑
- `범죄맥락분위수_권장` (보조) — 시간대 결합 없는 전체 분위수 (검증·비교용)

## 해석 주의

이 변수는 행정동별 실제 범죄 발생건수가 아니라,
소속 자치구의 야간 범죄 추정 수준을 행정동에 동일하게 부여한 구 단위 치안 맥락변수다.
"""
    (out_dir / "README_crime_context.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="서울 자치구 범죄 맥락변수 처리 (v1 9.1 시간대 결합)")
    parser.add_argument("--crime-place", type=Path,
                        default=Path("data/raw/범죄데이터/전국_범죄발생지.csv"),
                        help="자치구별 범죄 건수 raw (대분류·소분류 형태)")
    parser.add_argument("--crime-time", type=Path,
                        default=Path("data/raw/범죄데이터/전국_범죄발생시간.csv"),
                        help="전국 범죄 시간대 raw (24시간 8개 시간대 분포)")
    parser.add_argument("--region-code", type=Path,
                        default=Path("data/raw/범죄데이터/센서스 공간정보 지역 코드.xlsx"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/crime_context"))
    parser.add_argument("--skip-night", action="store_true",
                        help="시간대 결합 건너뛰기 (전체 분위수만 산출)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] 전국_범죄발생지 → 서울 25구 강력+폭력 추출")
    gu_context = build_seoul_gu_crime_context(args.crime_place)

    if not args.skip_night:
        print("\n[2] 전국_범죄발생시간 → 범죄 유형별 야간 비율 산출")
        night_ratio = compute_night_ratio_by_category(args.crime_time)

        print("\n[3] v1 9.1 결합 — 자치구 야간 강력+폭력 추정 + 분위수")
        gu_context = estimate_night_crime_by_gu(gu_context, night_ratio)

    print("\n[4] 행정동에 자치구 분위수 부여 (Area Proportional Assignment)")
    dong_context = build_seoul_dong_crime_context(gu_context, args.region_code)

    gu_path = args.out_dir / "seoul_gu_crime_context_2023_2024.csv"
    dong_path = args.out_dir / "seoul_dong_crime_context_2023_2024.csv"
    gu_context.to_csv(gu_path, index=False, encoding="utf-8-sig")
    dong_context.to_csv(dong_path, index=False, encoding="utf-8-sig")
    write_readme(args.out_dir)

    print(f"\n구 파일: {gu_path} 행수={len(gu_context)}")
    print(f"동 파일: {dong_path} 행수={len(dong_context)}")
    if not args.skip_night:
        print(f"\n→ 메인 사용 컬럼: 야간_범죄맥락분위수_권장")
        print(f"   (master_table.py crime_percentile 매핑 대상)")


if __name__ == "__main__":
    # 실행 예시 (필요 시 주석 해제):
    # import sys
    # sys.argv = [
    #     "crime_context.py",
    #     "--crime-place", "data/raw/범죄데이터/전국_범죄발생지.csv",
    #     "--region-code", "data/raw/범죄데이터/센서스 공간정보 지역 코드.xlsx",
    #     "--out-dir", "data/processed/crime_context",
    # ]
    # main()
    pass
