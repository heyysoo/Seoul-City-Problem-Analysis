"""서울시 야간경제 정책도구 적용 후보 20동 Folium 지도.

처리 목적
- 발표용 화면에 맞춰 지도와 우측 설명 패널을 하나의 HTML로 생성한다.
- 후보 20동은 저녁 7동, 공통 3동, 심야 10동으로 색상 구분한다.
- 대표 6동은 굵은 경계와 번호 마커로 강조한다.
- 보류·비대상·특수상권 11동은 회색 사선 후보로 함께 표시한다.

입력
- 서울 행정동 경계 SHP: bnd_dong_11_2025_2Q.shp

산출물
- seoul_night_policy_candidates_20.html

실행 예시
python 5_viz.py \
  --shp data/raw/bnd_dong_11_2025_2Q/bnd_dong_11_2025_2Q.shp \
  --out data/processed/viz/seoul_night_policy_candidates_20.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
from branca.element import Element


# -----------------------------------------------------------------------------
# 후보 목록
# -----------------------------------------------------------------------------

TRACKS = {
    "evening": {
        "label": "저녁",
        "count_label": "저녁 7",
        "color": "#2563eb",
        "stroke": "#1d4ed8",
        "dongs": ["숭인2동", "창1동", "황학동", "창신1동", "구로2동", "자양3동", "자양4동"],
    },
    "common": {
        "label": "공통",
        "count_label": "공통 3",
        "color": "#facc15",
        "stroke": "#b58900",
        "dongs": ["용산2가동", "망원1동", "고척1동"],
    },
    "late": {
        "label": "심야",
        "count_label": "심야 10",
        "color": "#7c3aed",
        "stroke": "#5b21b6",
        "dongs": [
            "문정2동", "가양1동", "발산1동", "잠원동", "한강로동",
            "남영동", "사당2동", "구로3동", "성수1가2동", "종로1·2·3·4가동",
        ],
    },
}

REPRESENTATIVE_DONGS = [
    {"rank": 1, "dong": "창1동", "track": "evening"},
    {"rank": 2, "dong": "숭인2동", "track": "evening"},
    {"rank": 3, "dong": "용산2가동", "track": "common"},
    {"rank": 4, "dong": "망원1동", "track": "common"},
    {"rank": 5, "dong": "문정2동", "track": "late"},
    {"rank": 6, "dong": "가양1동", "track": "late"},
]

SPECIAL_DONGS = [
    "수서동", "여의동", "가락1동", "염창동", "소공동", "잠실6동",
    "고덕2동", "자양2동", "도봉2동", "창4동", "천호3동",
]

REGION_LABELS = [
    ("서북권", 37.560, 126.905),
    ("동북권", 37.642, 127.066),
    ("도심권", 37.557, 126.988),
    ("서남권", 37.485, 126.893),
    ("동남권", 37.505, 127.078),
]

GU_CODE_MAP = {
    "11110": "종로구", "11140": "중구", "11170": "용산구", "11200": "성동구",
    "11215": "광진구", "11230": "동대문구", "11260": "중랑구", "11290": "성북구",
    "11305": "강북구", "11320": "도봉구", "11350": "노원구", "11380": "은평구",
    "11410": "서대문구", "11440": "마포구", "11470": "양천구", "11500": "강서구",
    "11530": "구로구", "11545": "금천구", "11560": "영등포구", "11590": "동작구",
    "11620": "관악구", "11650": "서초구", "11680": "강남구", "11710": "송파구",
    "11740": "강동구",
}


# -----------------------------------------------------------------------------
# 공통 유틸
# -----------------------------------------------------------------------------

def normalize_dong_name(value: object) -> str:
    return str(value).strip().replace(".", "·")


def short_dong_name(value: str) -> str:
    name = normalize_dong_name(value)
    return name[:-1] if name.endswith("동") else name


def join_names(names: list[str]) -> str:
    return " · ".join(short_dong_name(name) for name in names)


def load_dong_boundary(path: Path, name_col: str, encoding: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, encoding=encoding)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:5179", allow_override=True)
    if name_col not in gdf.columns:
        raise KeyError(f"SHP에 동명 컬럼이 없습니다: {name_col}")
    if "ADM_CD" not in gdf.columns:
        raise KeyError("SHP에 ADM_CD 컬럼이 없습니다.")

    projected = gdf.to_crs("EPSG:5179")
    rep_points = projected.geometry.representative_point()
    rep_points = gpd.GeoSeries(rep_points, crs=projected.crs).to_crs("EPSG:4326")

    out = gdf.to_crs("EPSG:4326").copy()
    out["_dong_norm"] = out[name_col].map(normalize_dong_name)
    out["_label_lat"] = rep_points.y.values
    out["_label_lon"] = rep_points.x.values
    out["ADM_CD"] = out["ADM_CD"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)

    if "SGG_NM" in out.columns:
        out["_gu_name"] = out["SGG_NM"].astype(str)
    else:
        out["_gu_name"] = out["ADM_CD"].str[:5].map(GU_CODE_MAP).fillna("")

    return out


def select_dong(gdf: gpd.GeoDataFrame, dong_name: str) -> gpd.GeoDataFrame:
    target = normalize_dong_name(dong_name)
    return gdf[gdf["_dong_norm"] == target]


def dong_center(gdf: gpd.GeoDataFrame, dong_name: str) -> tuple[float, float] | None:
    feat = select_dong(gdf, dong_name)
    if feat.empty:
        return None
    row = feat.iloc[0]
    return float(row["_label_lat"]), float(row["_label_lon"])


def missing_dongs(gdf: gpd.GeoDataFrame) -> list[str]:
    required = []
    for meta in TRACKS.values():
        required.extend(meta["dongs"])
    required.extend(SPECIAL_DONGS)
    required.extend(item["dong"] for item in REPRESENTATIVE_DONGS)
    existing = set(gdf["_dong_norm"])
    return sorted({name for name in required if normalize_dong_name(name) not in existing})


# -----------------------------------------------------------------------------
# 지도 구성 요소
# -----------------------------------------------------------------------------

def popup_html(dong_name: str, gu_name: str, track_label: str, note: str = "") -> folium.Popup:
    rows = ""
    if note:
        rows = f"""
        <tr>
          <th style="padding:4px 8px; color:#64748b; text-align:left;">비고</th>
          <td style="padding:4px 8px;">{note}</td>
        </tr>
        """
    html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Noto Sans KR', sans-serif; min-width:220px;">
      <h3 style="margin:0 0 8px 0; color:#0f172a;">{short_dong_name(dong_name)}</h3>
      <table style="font-size:12px; border-collapse:collapse;">
        <tr>
          <th style="padding:4px 8px; color:#64748b; text-align:left;">자치구</th>
          <td style="padding:4px 8px;">{gu_name}</td>
        </tr>
        <tr>
          <th style="padding:4px 8px; color:#64748b; text-align:left;">분류</th>
          <td style="padding:4px 8px;"><b>{track_label}</b></td>
        </tr>
        {rows}
      </table>
    </div>
    """
    return folium.Popup(html, max_width=320)


def add_background(m: folium.Map, gdf: gpd.GeoDataFrame, name_col: str) -> None:
    folium.GeoJson(
        gdf,
        name="서울 전체 행정동",
        style_function=lambda _: {
            "fillColor": "#fbfdff",
            "color": "#d9e2ec",
            "weight": 0.55,
            "fillOpacity": 0.36,
        },
        highlight_function=lambda _: {"weight": 1.2, "color": "#94a3b8", "fillOpacity": 0.45},
        tooltip=folium.GeoJsonTooltip(fields=[name_col], aliases=["행정동"], sticky=False),
    ).add_to(m)


def add_special_dongs(m: folium.Map, gdf: gpd.GeoDataFrame, name_col: str) -> None:
    for dong_name in SPECIAL_DONGS:
        feat = select_dong(gdf, dong_name)
        if feat.empty:
            continue
        row = feat.iloc[0]
        folium.GeoJson(
            feat,
            name=f"보류·비대상 {short_dong_name(dong_name)}",
            style_function=lambda _: {
                "fillColor": "#e2e8f0",
                "color": "#9aa8ba",
                "weight": 1.2,
                "fillOpacity": 0.55,
                "dashArray": "6,4",
                "className": "night-hatch",
            },
            popup=popup_html(row[name_col], row["_gu_name"], "보류·비대상·특수상권"),
        ).add_to(m)


def add_candidate_dongs(m: folium.Map, gdf: gpd.GeoDataFrame, name_col: str) -> None:
    for track_key, meta in TRACKS.items():
        for dong_name in meta["dongs"]:
            feat = select_dong(gdf, dong_name)
            if feat.empty:
                continue
            row = feat.iloc[0]
            folium.GeoJson(
                feat,
                name=f"{meta['label']} {short_dong_name(dong_name)}",
                style_function=lambda _, c=meta["color"], s=meta["stroke"]: {
                    "fillColor": c,
                    "color": s,
                    "weight": 1.5,
                    "fillOpacity": 0.76,
                },
                highlight_function=lambda _, s=meta["stroke"]: {
                    "weight": 3.2,
                    "color": s,
                    "fillOpacity": 0.9,
                },
                popup=popup_html(row[name_col], row["_gu_name"], f"{meta['label']} 후보"),
                tooltip=f"{short_dong_name(dong_name)} · {meta['label']} 후보",
            ).add_to(m)


def add_representative_outlines(m: folium.Map, gdf: gpd.GeoDataFrame) -> None:
    for item in REPRESENTATIVE_DONGS:
        feat = select_dong(gdf, item["dong"])
        if feat.empty:
            continue
        folium.GeoJson(
            feat,
            name=f"대표 {item['rank']} {short_dong_name(item['dong'])}",
            style_function=lambda _: {
                "fillColor": "transparent",
                "color": "#0f172a",
                "weight": 4.2,
                "fillOpacity": 0,
                "opacity": 1,
            },
        ).add_to(m)


def add_rank_markers(m: folium.Map, gdf: gpd.GeoDataFrame) -> None:
    for item in REPRESENTATIVE_DONGS:
        center = dong_center(gdf, item["dong"])
        if center is None:
            continue
        track = TRACKS[item["track"]]
        html = f"""
        <div class="rank-marker">
          <span class="rank-dot">{item['rank']}</span>
          <span class="rank-name">{short_dong_name(item['dong'])}</span>
        </div>
        """
        folium.Marker(
            location=center,
            icon=folium.DivIcon(
                html=html,
                icon_size=(112, 42),
                icon_anchor=(18, 20),
                class_name=f"rank-wrap rank-{item['track']}",
            ),
            tooltip=f"대표 {item['rank']} · {track['label']} · {short_dong_name(item['dong'])}",
        ).add_to(m)


def add_region_labels(m: folium.Map) -> None:
    for label, lat, lon in REGION_LABELS:
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=f'<div class="region-label">{label}</div>',
                icon_size=(62, 24),
                icon_anchor=(31, 12),
            ),
        ).add_to(m)


# -----------------------------------------------------------------------------
# 고정 레이아웃 HTML
# -----------------------------------------------------------------------------

def build_side_panel() -> str:
    reps = []
    for item in REPRESENTATIVE_DONGS:
        track = TRACKS[item["track"]]
        reps.append(
            f"""
            <div class="rep-row">
              <span class="rep-no">{item['rank']}</span>
              <span class="track-badge {item['track']}">{track['label']}</span>
              <span class="rep-name">{short_dong_name(item['dong'])}</span>
            </div>
            """
        )

    return f"""
    <div class="title-block">
      <h1>서울시 야간경제 정책도구 적용 후보 20동</h1>
      <p>저녁 후보 7동 · 공통 후보 3동 · 심야 후보 10동 / 대표 6동은 번호 마커와 굵은 경계로 강조</p>
    </div>

    <aside class="info-panel">
      <div class="panel-head">
        <h2>트랙별 후보</h2>
        <div class="mini-badges">
          <span class="track-badge evening">저녁</span>
          <span class="track-badge common">공통</span>
          <span class="track-badge late">심야</span>
        </div>
      </div>

      <div class="track-list">
        <div class="track-row">
          <span class="track-pill evening">저녁 7</span>
          <span class="track-text">{join_names(TRACKS["evening"]["dongs"])}</span>
        </div>
        <div class="track-row">
          <span class="track-pill common">공통 3</span>
          <span class="track-text">{join_names(TRACKS["common"]["dongs"])}</span>
        </div>
        <div class="track-row">
          <span class="track-pill late">심야 10</span>
          <span class="track-text">{join_names(TRACKS["late"]["dongs"])}</span>
        </div>
      </div>

      <h2 class="rep-title">대표 6동</h2>
      <div class="rep-list">
        {''.join(reps)}
      </div>

      <div class="special-block">
        <strong>보류·비대상·특수상권 11동</strong>
        <p>{join_names(SPECIAL_DONGS)}</p>
      </div>

      <p class="panel-note">색상은 발표 기준 트랙을 의미하며, 회색 사선은 보류/Q&amp;A 처리 후보입니다.</p>
    </aside>
    """


def add_page_layout(m: folium.Map) -> None:
    css = """
    <style>
      html, body {
        margin: 0;
        width: 100%;
        height: 100%;
        background: #ffffff;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Noto Sans KR", "Apple SD Gothic Neo", sans-serif;
      }
      .folium-map {
        position: fixed !important;
        left: 72px !important;
        right: 520px !important;
        top: 112px !important;
        bottom: 42px !important;
        width: auto !important;
        height: auto !important;
        background: #ffffff !important;
      }
      .leaflet-control-attribution,
      .leaflet-control-zoom {
        display: none !important;
      }
      .title-block {
        position: fixed;
        left: 78px;
        top: 22px;
        z-index: 9999;
      }
      .title-block h1 {
        margin: 0;
        font-size: 39px;
        line-height: 1.1;
        letter-spacing: 0;
        font-weight: 500;
        color: #111827;
      }
      .title-block p {
        margin: 8px 0 0 2px;
        font-size: 17px;
        color: #475569;
      }
      .info-panel {
        position: fixed;
        top: 92px;
        right: 74px;
        width: 430px;
        bottom: 42px;
        z-index: 9999;
        background: #f5f7fa;
        padding: 42px 38px;
        box-sizing: border-box;
      }
      .panel-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 54px;
      }
      .panel-head h2,
      .rep-title {
        margin: 0;
        font-size: 26px;
        font-weight: 500;
        letter-spacing: 0;
      }
      .mini-badges {
        display: flex;
        gap: 7px;
      }
      .track-badge,
      .track-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 45px;
        height: 44px;
        padding: 0 13px;
        box-sizing: border-box;
        border-radius: 7px;
        color: #fff;
        font-weight: 600;
        font-size: 14px;
        white-space: nowrap;
      }
      .track-badge.common,
      .track-pill.common {
        color: #111827;
      }
      .evening { background: #2563eb; }
      .common { background: #facc15; }
      .late { background: #7c3aed; }
      .track-list {
        display: grid;
        gap: 52px;
        margin-bottom: 82px;
      }
      .track-row {
        display: grid;
        grid-template-columns: 104px 1fr;
        align-items: center;
        column-gap: 10px;
      }
      .track-pill {
        width: 104px;
        height: 56px;
        font-size: 16px;
      }
      .track-text {
        font-size: 15px;
        line-height: 1.35;
        color: #111827;
      }
      .rep-title {
        margin-bottom: 28px;
      }
      .rep-list {
        display: grid;
        gap: 9px;
        margin-bottom: 17px;
      }
      .rep-row {
        display: grid;
        grid-template-columns: 34px 78px 1fr;
        align-items: center;
        column-gap: 10px;
        min-height: 42px;
      }
      .rep-no {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 24px;
        height: 24px;
        border-radius: 999px;
        background: #0f172a;
        color: #fff;
        font-size: 13px;
        font-weight: 700;
      }
      .rep-row .track-badge {
        height: 40px;
        min-width: 78px;
        font-size: 14px;
      }
      .rep-name {
        font-size: 20px;
        color: #111827;
      }
      .special-block {
        margin-top: 8px;
        color: #475569;
      }
      .special-block strong {
        display: block;
        font-size: 17px;
        font-weight: 500;
        margin-bottom: 14px;
      }
      .special-block p {
        margin: 0;
        font-size: 13px;
        line-height: 1.55;
      }
      .panel-note {
        position: absolute;
        left: 38px;
        right: 38px;
        bottom: 22px;
        margin: 0;
        font-size: 12px;
        color: #64748b;
      }
      .rank-marker {
        display: inline-flex;
        align-items: center;
        position: relative;
        gap: 4px;
      }
      .rank-dot {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        border-radius: 999px;
        background: #0f172a;
        color: #fff;
        border: 3px solid #fff;
        box-shadow: 0 0 0 3px #0f172a;
        font-size: 17px;
        font-weight: 700;
        line-height: 1;
      }
      .rank-name {
        display: inline-flex;
        align-items: center;
        height: 22px;
        padding: 0 5px;
        border: 1.5px solid #111827;
        border-radius: 3px;
        background: rgba(255,255,255,0.94);
        color: #111827;
        font-size: 14px;
        font-weight: 500;
        white-space: nowrap;
        transform: translateY(-19px);
      }
      .region-label {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        height: 26px;
        padding: 0 8px;
        border-radius: 5px;
        background: rgba(75, 85, 99, 0.88);
        color: white;
        font-size: 15px;
        font-weight: 600;
        white-space: nowrap;
      }
      path.night-hatch {
        fill: url(#nightDiagonalHatch) !important;
        fill-opacity: 0.72 !important;
      }
      @media (max-width: 1200px) {
        .folium-map { right: 410px !important; left: 34px !important; }
        .info-panel { right: 30px; width: 360px; padding: 32px 28px; }
        .title-block { left: 42px; }
        .title-block h1 { font-size: 31px; }
      }
    </style>
    """

    script = """
    <script>
      function installNightHatchPattern() {
        document.querySelectorAll(".leaflet-overlay-pane svg").forEach(function(svg) {
          if (svg.querySelector("#nightDiagonalHatch")) return;
          var defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
          var pattern = document.createElementNS("http://www.w3.org/2000/svg", "pattern");
          pattern.setAttribute("id", "nightDiagonalHatch");
          pattern.setAttribute("patternUnits", "userSpaceOnUse");
          pattern.setAttribute("width", "8");
          pattern.setAttribute("height", "8");
          var rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
          rect.setAttribute("width", "8");
          rect.setAttribute("height", "8");
          rect.setAttribute("fill", "#edf2f7");
          var line = document.createElementNS("http://www.w3.org/2000/svg", "path");
          line.setAttribute("d", "M-2,8 l10,-10 M0,10 l10,-10 M6,12 l10,-10");
          line.setAttribute("stroke", "#9aa8ba");
          line.setAttribute("stroke-width", "1.2");
          defs.appendChild(pattern);
          pattern.appendChild(rect);
          pattern.appendChild(line);
          svg.insertBefore(defs, svg.firstChild);
        });
      }
      document.addEventListener("DOMContentLoaded", function() {
        installNightHatchPattern();
        setTimeout(installNightHatchPattern, 400);
        setTimeout(installNightHatchPattern, 1200);
      });
    </script>
    """

    m.get_root().header.add_child(Element(css))
    m.get_root().html.add_child(Element(build_side_panel()))
    m.get_root().html.add_child(Element(script))


# -----------------------------------------------------------------------------
# 진입점
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="서울시 야간경제 정책도구 적용 후보 20동 지도")
    parser.add_argument(
        "--shp",
        type=Path,
        default=Path("data/raw/bnd_dong_11_2025_2Q/bnd_dong_11_2025_2Q.shp"),
        help="서울 행정동 경계 SHP",
    )
    parser.add_argument("--shp-encoding", type=str, default="cp949")
    parser.add_argument("--shp-name-col", type=str, default="ADM_NM")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/viz/seoul_night_policy_candidates_20.html"),
    )
    parser.add_argument("--zoom-start", type=int, default=11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("[1] 서울 행정동 경계 로드")
    gdf = load_dong_boundary(args.shp, args.shp_name_col, args.shp_encoding)
    print(f"  행정동: {len(gdf):,}개")

    missing = missing_dongs(gdf)
    if missing:
        print("  경고: SHP에서 찾지 못한 동명:", ", ".join(missing))

    print("[2] Folium 지도 생성")
    m = folium.Map(
        location=[37.558, 126.995],
        zoom_start=args.zoom_start,
        tiles="CartoDB positron",
        control_scale=False,
        zoom_control=False,
        prefer_canvas=True,
    )
    m.fit_bounds([[37.415, 126.755], [37.715, 127.190]])

    add_background(m, gdf, args.shp_name_col)
    add_special_dongs(m, gdf, args.shp_name_col)
    add_candidate_dongs(m, gdf, args.shp_name_col)
    add_representative_outlines(m, gdf)
    add_rank_markers(m, gdf)
    add_region_labels(m)
    add_page_layout(m)

    m.save(str(args.out))
    print(f"[3] 저장 완료: {args.out}")
    print("  저녁 후보 7동 / 공통 후보 3동 / 심야 후보 10동 / 대표 6동 / 보류·비대상 11동")


if __name__ == "__main__":
    main()
