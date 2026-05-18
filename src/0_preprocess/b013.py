"""B013 교통데이터 버스·지하철 통합 처리

캠퍼스 환경(E:/ 드라이브)에서 B013 거래내역(2023~2025)을 스트리밍으로 읽어
저녁(17:00~20:59) / 심야(21:00~05:59) 트랙별로 행정동·역 단위 승객수 합계와
연속 연도 YoY 증감률을 산출한다.

산출물
- ckpt_bus_{YYYYMM}.csv, ckpt_subway_{YYYYMM}.csv
- b013_bus_2023_2025_all.csv, b013_subway_2023_2025_all.csv
- b013_bus_yoy_2023_2025.csv, b013_subway_yoy_2023_2025.csv
"""

import argparse
import re
import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR = Path("E:/")

DEAL_BASE_DIRS = [
    ROOT_DIR / "B013 공유폴더" / "거래내역",
    ROOT_DIR / "B013공유폴더" / "거래내역",
    ROOT_DIR / "거래내역",
]

BUS_MASTER_DIRS = [
    ROOT_DIR / "B013공유폴더" / "버스정류장",
    ROOT_DIR / "버스정류장",
]

SUBWAY_MASTER_DIRS = [
    ROOT_DIR / "B013공유폴더" / "지하철역",
    ROOT_DIR / "지하철역",
    ROOT_DIR / "코드",
]

OUT_DIR = ROOT_DIR / "b013_ckpt_2023_2025"

CHUNKSIZE = 200_000

RUN_JOBS = [
    (2023, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (2024, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (2025, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
]

RUN_YEARS = sorted({year for year, _ in RUN_JOBS})
YOY_PAIRS = list(zip(RUN_YEARS[:-1], RUN_YEARS[1:]))
YEAR_LABEL = f"{RUN_YEARS[0]}_{RUN_YEARS[-1]}"

# 저녁: 17:00 이상 21:00 미만
# 심야: 21:00 이상 또는 06:00 미만
EVENING_START_MIN = 17 * 60
EVENING_END_MIN = 21 * 60
LATE_START_MIN = 21 * 60
LATE_END_MIN = 6 * 60

SUBWAY_CODES = {"201"}
NBUS_CODES = {"131"}

ENCODINGS = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]


def detect_compression(path):
    path = Path(path)
    with open(path, "rb") as f:
        sig = f.read(2)
    return "gzip" if sig == b"\x1f\x8b" else None


def read_csv_auto(path, usecols=None, nrows=None, chunksize=None):
    path = Path(path)
    compression = detect_compression(path)
    last_error = None

    for enc in ENCODINGS:
        try:
            return pd.read_csv(
                path,
                encoding=enc,
                compression=compression,
                dtype=str,
                usecols=usecols,
                nrows=nrows,
                chunksize=chunksize,
                low_memory=False,
                on_bad_lines="skip"
            )
        except Exception as e:
            last_error = e

    raise last_error


def clean_key(col):
    col = str(col).strip().replace("\ufeff", "")
    col = col.lstrip("#").strip()
    col = re.sub(r"\([^)]*\)", "", col)
    col = col.replace(" ", "").replace("_", "").replace("-", "")
    return col.lower()


COL_ALIAS = {
    "교통수단코드": "transport_code",
    "교통수단cd": "transport_code",
    "sudancd": "transport_code",

    "노선id": "line_id",
    "버스노선id": "line_id",
    "lineid": "line_id",

    "승차일시": "geton_time",
    "getondatetime": "geton_time",

    "하차일시": "getoff_time",
    "getoffdatetime": "getoff_time",

    "승차정류장id": "geton_station_id",
    "getonstationid": "geton_station_id",

    "하차정류장id": "getoff_station_id",
    "getoffstationid": "getoff_station_id",

    "승객수1": "passenger1",
    "passncnt1": "passenger1",

    "승객수2": "passenger2",
    "passncnt2": "passenger2",

    "승객수3": "passenger3",
    "passncnt3": "passenger3",

    "정류장id": "station_id",
    "stationid": "station_id",

    "역id": "station_id",
    "지하철역id": "station_id",

    "행정동id": "adm_id",
    "행정동코드": "adm_id",
    "adstrdcd": "adm_id",

    # fallback. 실제 행정동ID가 있으면 그걸 우선 사용하고, 없을 때만 법정동ID를 쓴다.
    "법정동id": "adm_id",
    "법정동코드": "adm_id",

    "노선명": "line_name",
    "버스노선명": "line_name",
    "linenm": "line_name",

    "역명": "station_name",
    "지하철역": "station_name",
    "sttn": "station_name",

    "호선명": "subway_line_name",
    "sbwyroutlnnm": "subway_line_name",
}


def normalize_columns(df):
    direct_rename = {
        "교통수단코드": "transport_code",
        "교통수단CD": "transport_code",
        "SUDAN_CD": "transport_code",
        "노선ID": "line_id",
        "버스노선ID": "line_id",
        "LINE_ID": "line_id",
        "승차일시": "geton_time",
        "GETON_DATETIME": "geton_time",
        "하차일시": "getoff_time",
        "GETOFF_DATETIME": "getoff_time",
        "승차정류장ID": "geton_station_id",
        "GETON_STATION_ID": "geton_station_id",
        "하차정류장ID": "getoff_station_id",
        "GETOFF_STATION_ID": "getoff_station_id",
        "승객수1": "passenger1",
        "PASSN_CNT1": "passenger1",
        "승객수2": "passenger2",
        "PASSN_CNT2": "passenger2",
        "승객수3": "passenger3",
        "PASSN_CNT3": "passenger3",
    }

    df = df.rename(
        columns={
            c: direct_rename[str(c).strip().replace("\ufeff", "")]
            for c in df.columns
            if str(c).strip().replace("\ufeff", "") in direct_rename
        }
    )

    rename = {}
    for c in df.columns:
        k = clean_key(c)
        if k in COL_ALIAS:
            rename[c] = COL_ALIAS[k]

    df = df.rename(columns=rename)

    # B013 파일은 연도/반출형태에 따라 한글 헤더가 미묘하게 달라질 수 있으므로
    # 필수 컬럼은 부분 문자열 기준으로 한 번 더 강제 매핑한다.
    fallback = {}
    for c in df.columns:
        if c in {
            "transport_code", "line_id",
            "geton_time", "geton_station_id",
            "getoff_time", "getoff_station_id",
            "passenger1", "passenger2", "passenger3",
        }:
            continue

        k = clean_key(c)

        if "교통수단" in k and "코드" in k:
            fallback[c] = "transport_code"
        elif "노선" in k and "id" in k:
            fallback[c] = "line_id"
        elif "승차일시" in k or "getondatetime" in k:
            fallback[c] = "geton_time"
        elif "하차일시" in k or "getoffdatetime" in k:
            fallback[c] = "getoff_time"
        elif "승차정류장" in k and "id" in k:
            fallback[c] = "geton_station_id"
        elif "하차정류장" in k and "id" in k:
            fallback[c] = "getoff_station_id"
        elif "passncnt1" in k or "승객수1" in k:
            fallback[c] = "passenger1"
        elif "passncnt2" in k or "승객수2" in k:
            fallback[c] = "passenger2"
        elif "passncnt3" in k or "승객수3" in k:
            fallback[c] = "passenger3"

    return df.rename(columns=fallback)


def normalize_id(s):
    return (
        s.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
    )


def to_number(s):
    return pd.to_numeric(s, errors="coerce").fillna(0)


def extract_hour(s):
    raw = s.astype(str).str.replace(r"\D", "", regex=True)
    # YYYYMMDDHHMMSS 기준
    hour = pd.to_numeric(raw.str.slice(8, 10), errors="coerce")
    return hour


def extract_minute_of_day(s):
    raw = s.astype(str).str.replace(r"\D", "", regex=True)
    # YYYYMMDDHHMMSS 기준
    hour = pd.to_numeric(raw.str.slice(8, 10), errors="coerce")
    minute = pd.to_numeric(raw.str.slice(10, 12), errors="coerce")
    return hour * 60 + minute


def time_window_mask(minute_of_day, window):
    if window == "evening":
        return (minute_of_day >= EVENING_START_MIN) & (minute_of_day < EVENING_END_MIN)
    if window == "late":
        return (minute_of_day >= LATE_START_MIN) | (minute_of_day < LATE_END_MIN)
    raise ValueError(f"알 수 없는 시간대: {window}")


def safe_percent_change(new, old):
    old = pd.to_numeric(old, errors="coerce")
    new = pd.to_numeric(new, errors="coerce")
    return np.where(old > 0, (new - old) / old, np.nan)


def find_cp01_files(year, month):
    yy = str(year)[2:]
    mm = f"{month:02d}"
    ym = f"{year}{mm}"

    # cp01240101, cp01240101.gz, cp01240101.csv 같은 형태까지 허용
    pattern = re.compile(rf"^cp01{yy}{mm}\d{{2}}(?:\.[A-Za-z0-9]+)?$", re.IGNORECASE)

    roots = []
    for base_dir in DEAL_BASE_DIRS:
        roots.extend([
            base_dir / str(year) / ym,
            base_dir / str(year),
            base_dir / ym,
            base_dir,
        ])

    files = []
    for root in roots:
        if not root.exists():
            continue

        for p in root.rglob("*"):
            if p.is_file() and pattern.match(p.name):
                files.append(p)

    return sorted(set(files))


def find_latest_master_file(prefix, dirs):
    pattern = re.compile(rf"^{prefix}\d+(\.gz)?$", re.IGNORECASE)
    files = []

    for d in dirs:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file() and pattern.match(p.name):
                files.append(p)

    if not files:
        return None

    return sorted(files)[-1]


def read_bus_master():
    path = find_latest_master_file("cp12", BUS_MASTER_DIRS)
    if path is None:
        raise FileNotFoundError("버스정류장 마스터 cp12 파일을 찾지 못했습니다.")

    print("버스 마스터:", path)

    df = read_csv_auto(path)
    df = normalize_columns(df)

    required = ["line_id", "station_id", "adm_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"버스 마스터 필수 컬럼 없음: {missing}\n현재 컬럼: {df.columns.tolist()}")

    if "line_name" not in df.columns:
        df["line_name"] = ""

    df["line_id"] = normalize_id(df["line_id"])
    df["station_id"] = normalize_id(df["station_id"])
    df["adm_id"] = normalize_id(df["adm_id"])
    df["line_name"] = df["line_name"].astype(str).str.strip()

    df = df.dropna(subset=["station_id", "adm_id"])
    df = df[df["adm_id"].str.startswith("11", na=False)]

    return df[["line_id", "station_id", "adm_id", "line_name"]].drop_duplicates()


def read_subway_master_optional():
    path = find_latest_master_file("cp13", SUBWAY_MASTER_DIRS)

    if path is None:
        print("지하철 마스터 cp13 파일 없음: 역ID만 사용")
        return None

    print("지하철 마스터:", path)

    try:
        df = read_csv_auto(path)
        df = normalize_columns(df)

        if "station_id" not in df.columns:
            print("지하철 마스터에 station_id 없음: 역ID만 사용")
            return None

        df["station_id"] = normalize_id(df["station_id"])

        if "station_name" not in df.columns:
            df["station_name"] = ""

        if "subway_line_name" not in df.columns:
            if "line_name" in df.columns:
                df["subway_line_name"] = df["line_name"]
            else:
                df["subway_line_name"] = ""

        return (
            df[["station_id", "station_name", "subway_line_name"]]
            .dropna(subset=["station_id"])
            .drop_duplicates()
        )

    except Exception as e:
        print("지하철 마스터 읽기 실패. 역ID만 사용:", e)
        return None


BUS_MASTER = None
SUBWAY_MASTER = None


def prepare_transaction_chunk(chunk):
    chunk = normalize_columns(chunk)

    required = [
        "transport_code", "line_id",
        "geton_time", "geton_station_id",
        "getoff_time", "getoff_station_id",
        "passenger1", "passenger2", "passenger3",
    ]

    missing = [c for c in required if c not in chunk.columns]
    if missing:
        raise KeyError(f"거래내역 필수 컬럼 없음: {missing}\n현재 컬럼: {chunk.columns.tolist()}")

    chunk = chunk[required].copy()

    chunk["transport_code"] = normalize_id(chunk["transport_code"])
    chunk["line_id"] = normalize_id(chunk["line_id"])
    chunk["geton_station_id"] = normalize_id(chunk["geton_station_id"])
    chunk["getoff_station_id"] = normalize_id(chunk["getoff_station_id"])

    chunk["passenger_sum"] = (
        to_number(chunk["passenger1"])
        + to_number(chunk["passenger2"])
        + to_number(chunk["passenger3"])
    )

    return chunk


def aggregate_bus_direction(chunk, ym, direction, window, metric_prefix):
    if direction == "in":
        time_col = "getoff_time"
        station_col = "getoff_station_id"
        metric_suffix = "유입"
    else:
        time_col = "geton_time"
        station_col = "geton_station_id"
        metric_suffix = "유출"

    tmp = chunk.copy()
    tmp["minute_of_day"] = extract_minute_of_day(tmp[time_col])
    tmp = tmp[time_window_mask(tmp["minute_of_day"], window)].copy()

    if tmp.empty:
        return pd.DataFrame(columns=["사용년월", "행정동ID", "지표", "승객수합계"])

    # 지하철 코드는 버스 처리에서 제외
    tmp = tmp[~tmp["transport_code"].isin(SUBWAY_CODES)].copy()

    if tmp.empty:
        return pd.DataFrame(columns=["사용년월", "행정동ID", "지표", "승객수합계"])

    merged = tmp.merge(
        BUS_MASTER,
        left_on=["line_id", station_col],
        right_on=["line_id", "station_id"],
        how="left"
    )

    merged = merged.dropna(subset=["adm_id"]).copy()

    if merged.empty:
        return pd.DataFrame(columns=["사용년월", "행정동ID", "지표", "승객수합계"])

    line_upper = merged["line_name"].astype(str).str.upper().str.strip()
    is_nbus = (
        merged["transport_code"].isin(NBUS_CODES)
        | line_upper.str.startswith("N")
        | line_upper.str.contains("심야", na=False)
    )

    if "N버스" in metric_prefix:
        merged = merged[is_nbus].copy()
    else:
        merged = merged[~is_nbus].copy()

    if merged.empty:
        return pd.DataFrame(columns=["사용년월", "행정동ID", "지표", "승객수합계"])

    out = (
        merged.groupby("adm_id", as_index=False)["passenger_sum"]
        .sum()
        .rename(columns={"adm_id": "행정동ID", "passenger_sum": "승객수합계"})
    )

    out["사용년월"] = ym
    out["지표"] = f"{metric_prefix}_{metric_suffix}"

    return out[["사용년월", "행정동ID", "지표", "승객수합계"]]


def process_bus_month(year, month, files):
    ym = f"{year}{month:02d}"
    parts = []

    for i, path in enumerate(files, start=1):
        print(f"[버스 {ym}] {i}/{len(files)} 읽는 중: {path.name}")

        reader = read_csv_auto(path, chunksize=CHUNKSIZE)

        for chunk in reader:
            try:
                tx = prepare_transaction_chunk(chunk)
            except Exception as e:
                print(f"[버스 {ym}] chunk skip:", e)
                continue

            parts.append(aggregate_bus_direction(tx, ym, "in", "evening", "저녁_일반버스"))
            parts.append(aggregate_bus_direction(tx, ym, "out", "evening", "저녁_일반버스"))
            parts.append(aggregate_bus_direction(tx, ym, "in", "late", "심야_일반버스"))
            parts.append(aggregate_bus_direction(tx, ym, "out", "late", "심야_일반버스"))
            parts.append(aggregate_bus_direction(tx, ym, "in", "late", "심야_N버스"))
            parts.append(aggregate_bus_direction(tx, ym, "out", "late", "심야_N버스"))

    if not parts:
        return pd.DataFrame(columns=["사용년월", "행정동ID", "지표", "승객수합계"])

    result = pd.concat(parts, ignore_index=True)
    result["승객수합계"] = pd.to_numeric(result["승객수합계"], errors="coerce").fillna(0)

    result = (
        result.groupby(["사용년월", "행정동ID", "지표"], as_index=False)["승객수합계"]
        .sum()
        .sort_values(["사용년월", "행정동ID", "지표"])
    )

    out_path = OUT_DIR / f"ckpt_bus_{ym}.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[버스 {ym}] 저장 완료:", out_path, "행수:", len(result))

    return result


def aggregate_subway_direction(chunk, ym, direction, window, track_prefix):
    if direction == "in":
        time_col = "getoff_time"
        station_col = "getoff_station_id"
        metric = f"{track_prefix}_지하철_유입"
    else:
        time_col = "geton_time"
        station_col = "geton_station_id"
        metric = f"{track_prefix}_지하철_유출"

    tmp = chunk.copy()
    tmp = tmp[tmp["transport_code"].isin(SUBWAY_CODES)].copy()

    if tmp.empty:
        return pd.DataFrame(columns=["사용년월", "역ID", "역명", "호선명", "지표", "승객수합계"])

    tmp["minute_of_day"] = extract_minute_of_day(tmp[time_col])
    tmp = tmp[time_window_mask(tmp["minute_of_day"], window)].copy()

    if tmp.empty:
        return pd.DataFrame(columns=["사용년월", "역ID", "역명", "호선명", "지표", "승객수합계"])

    tmp["역ID"] = tmp[station_col]

    out = (
        tmp.groupby("역ID", as_index=False)["passenger_sum"]
        .sum()
        .rename(columns={"passenger_sum": "승객수합계"})
    )

    if SUBWAY_MASTER is not None:
        out = out.merge(
            SUBWAY_MASTER,
            left_on="역ID",
            right_on="station_id",
            how="left"
        )
        out["역명"] = out["station_name"].fillna("")
        out["호선명"] = out["subway_line_name"].fillna("")
    else:
        out["역명"] = ""
        out["호선명"] = ""

    out["사용년월"] = ym
    out["지표"] = metric

    return out[["사용년월", "역ID", "역명", "호선명", "지표", "승객수합계"]]


def process_subway_month(year, month, files):
    ym = f"{year}{month:02d}"
    parts = []

    for i, path in enumerate(files, start=1):
        print(f"[지하철 {ym}] {i}/{len(files)} 읽는 중: {path.name}")

        reader = read_csv_auto(path, chunksize=CHUNKSIZE)

        for chunk in reader:
            try:
                tx = prepare_transaction_chunk(chunk)
            except Exception as e:
                print(f"[지하철 {ym}] chunk skip:", e)
                continue

            parts.append(aggregate_subway_direction(tx, ym, "in", "evening", "저녁"))
            parts.append(aggregate_subway_direction(tx, ym, "out", "evening", "저녁"))
            parts.append(aggregate_subway_direction(tx, ym, "in", "late", "심야"))
            parts.append(aggregate_subway_direction(tx, ym, "out", "late", "심야"))

    if not parts:
        return pd.DataFrame(columns=["사용년월", "역ID", "역명", "호선명", "지표", "승객수합계"])

    result = pd.concat(parts, ignore_index=True)
    result["승객수합계"] = pd.to_numeric(result["승객수합계"], errors="coerce").fillna(0)

    group_cols = ["사용년월", "역ID", "역명", "호선명", "지표"]
    result = (
        result.groupby(group_cols, as_index=False)["승객수합계"]
        .sum()
        .sort_values(["사용년월", "역ID", "지표"])
    )

    out_path = OUT_DIR / f"ckpt_subway_{ym}.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[지하철 {ym}] 저장 완료:", out_path, "행수:", len(result))

    return result


def process_month(year, month):
    ym = f"{year}{month:02d}"
    files = find_cp01_files(year, month)

    print("-" * 80)
    print(f"[{ym}] 거래내역 파일 수:", len(files))

    if not files:
        print(f"[{ym}] 파일 없음. skip")
        return

    process_bus_month(year, month, files)
    process_subway_month(year, month, files)


def combine_ckpts(kind):
    files = sorted(OUT_DIR.glob(f"ckpt_{kind}_*.csv"))

    if not files:
        print(f"{kind} ckpt 없음")
        return None

    df = pd.concat(
        [pd.read_csv(f, encoding="utf-8-sig", dtype=str) for f in files],
        ignore_index=True
    )

    df["승객수합계"] = pd.to_numeric(df["승객수합계"], errors="coerce").fillna(0)
    df["연도"] = df["사용년월"].astype(str).str[:4].astype(int)
    df["월"] = df["사용년월"].astype(str).str[4:6].astype(int)

    all_path = OUT_DIR / f"b013_{kind}_{YEAR_LABEL}_all.csv"
    df.to_csv(all_path, index=False, encoding="utf-8-sig")
    print(f"{kind} 전체 저장:", all_path, "행수:", len(df))

    return df


def make_bus_yoy(df):
    if df is None or df.empty:
        return

    bus = df.copy()
    bus["행정동ID"] = bus["행정동ID"].astype(str).str.strip()

    out = bus.pivot_table(
        index=["행정동ID", "월", "지표"],
        columns="연도",
        values="승객수합계",
        aggfunc="sum",
    ).reset_index()

    for year in RUN_YEARS:
        if year not in out.columns:
            out[year] = np.nan

    for old_year, new_year in YOY_PAIRS:
        out[f"{old_year}대비{new_year}"] = safe_percent_change(out[new_year], out[old_year])

    out = out.rename(columns={year: str(year) for year in RUN_YEARS})
    out = out.sort_values(["행정동ID", "월", "지표"])

    out_path = OUT_DIR / f"b013_bus_yoy_{YEAR_LABEL}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("버스 YoY 저장:", out_path, "행수:", len(out))


def make_subway_yoy(df):
    if df is None or df.empty:
        return

    sub = df.copy()
    sub["역ID"] = sub["역ID"].astype(str).str.strip()

    key_cols = ["역ID", "역명", "호선명", "월", "지표"]

    out = sub.pivot_table(
        index=key_cols,
        columns="연도",
        values="승객수합계",
        aggfunc="sum",
    ).reset_index()

    for year in RUN_YEARS:
        if year not in out.columns:
            out[year] = np.nan

    for old_year, new_year in YOY_PAIRS:
        out[f"{old_year}대비{new_year}"] = safe_percent_change(out[new_year], out[old_year])

    out = out.rename(columns={year: str(year) for year in RUN_YEARS})
    out = out.sort_values(["역ID", "월", "지표"])

    out_path = OUT_DIR / f"b013_subway_yoy_{YEAR_LABEL}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("지하철 YoY 저장:", out_path, "행수:", len(out))


def parse_args():
    parser = argparse.ArgumentParser(description="B013 버스·지하철 거래내역 통합 처리")
    parser.add_argument("--root-dir", type=Path, default=Path("E:/"), help="B013 원자료 루트 디렉터리")
    parser.add_argument("--out-dir", type=Path, default=None, help="산출물 저장 디렉터리")
    parser.add_argument("--chunksize", type=int, default=200_000, help="거래내역 chunk 크기")
    return parser.parse_args()


def configure_runtime(root_dir, out_dir, chunksize):
    global ROOT_DIR, DEAL_BASE_DIRS, BUS_MASTER_DIRS, SUBWAY_MASTER_DIRS
    global OUT_DIR, CHUNKSIZE, BUS_MASTER, SUBWAY_MASTER

    ROOT_DIR = Path(root_dir)
    DEAL_BASE_DIRS = [
        ROOT_DIR / "B013 공유폴더" / "거래내역",
        ROOT_DIR / "B013공유폴더" / "거래내역",
        ROOT_DIR / "거래내역",
    ]
    BUS_MASTER_DIRS = [
        ROOT_DIR / "B013공유폴더" / "버스정류장",
        ROOT_DIR / "버스정류장",
    ]
    SUBWAY_MASTER_DIRS = [
        ROOT_DIR / "B013공유폴더" / "지하철역",
        ROOT_DIR / "지하철역",
        ROOT_DIR / "코드",
    ]
    OUT_DIR = Path(out_dir) if out_dir is not None else ROOT_DIR / "b013_ckpt_2023_2025"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKSIZE = chunksize
    BUS_MASTER = read_bus_master()
    SUBWAY_MASTER = read_subway_master_optional()


def main():
    args = parse_args()
    configure_runtime(args.root_dir, args.out_dir, args.chunksize)

    for year, months in RUN_JOBS:
        print("=" * 80)
        print(f"{year}년 처리 시작: {months[0]:02d}월 ~ {months[-1]:02d}월")
        print("=" * 80)
        for month in months:
            process_month(year, month)

    bus_all = combine_ckpts("bus")
    subway_all = combine_ckpts("subway")
    make_bus_yoy(bus_all)
    make_subway_yoy(subway_all)

    print("=" * 80)
    print("전체 완료")
    print("저장 폴더:", OUT_DIR)
    print("=" * 80)


if __name__ == "__main__":
    main()
