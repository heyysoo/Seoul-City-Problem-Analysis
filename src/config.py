"""분석 전반의 핵심 상수 단일 참조 문서.

---

## 시간 정의 (Evans 2012 야간경제 정의 기반)

이론적 야간경제 범위: 18:00 ~ 06:00
분석 적용 범위: 17:00 ~ 06:00 (B040 데이터 블록 구조 제약)

저녁 트랙: 17:00 이상 ~ 21:00 미만
심야 트랙: 21:00 이상 ~ 다음날 06:00 미만

17-18시 진입부 포함 정당화: 행태 정합성 + 트랙 분리 견고성 + 분석 투명성

→ src/0_preprocess/b013.py: EVENING_START_MIN, EVENING_END_MIN,
                            LATE_START_MIN, LATE_END_MIN
→ src/0_preprocess/b040.py: TRACK_HOURS (evening, late_night)
→ src/0_preprocess/b079.py: TRACK_TIME_KEEP (저녁=[5], 심야=[1,6])

---

## 1차 필터: 거주지 필터 (Anderson 1976 USGS)

거주비율 컷오프: 67%
- 자연지역 분모 보정 적용 (보전산지∪개발제한∪도시자연공원 제외 시가화면적 기준)
- 공항동(11160690) 수동 제외
- 426동 → 108동 (v6 한강 보정 폐기, v5 정통 결과 유지)

→ src/1_filter/residential.py

---

## 2차 필터: 상권성 필터

commerce_score = density_percentile * 0.70 + share_percentile * 0.30
COMMERCE_QUANTILE = 0.28 (하위 28% 저상권성 검토)
결과 commerce_score 컷오프 값: ≈ 0.339185
규모 예외: sales_aligned_count ≥ 400 → 후보 중 2동 유지 (양평2동·발산1동)

108동 → 80동 (제외 28동)

→ src/1_filter/commerce.py
   raw 입력: data/raw/sbiz/sdsc_stores_raw.csv (소상공인진흥공단 상가 API)
            LEFT JOIN 으로 108동만 자동 필터링 (서울 전체 점포 ~600,000개 또는 사전 필터링분 모두 허용)

---

## X축 잠재력 가중치 (저녁·심야 공통)

potential = growth * 0.50 + opening * 0.30 + transit * 0.20

- growth (매출성장률): 저녁/심야 별도 산출 ← B079 (b079.py)
- opening (점포개점률): 시간대 무관 단일 버전 ← OA-22172 (opening_rate.py)
- transit (교통접근성): 저녁/심야 별도 산출 ← B013 (b013.py → b013_combine.py)

→ src/2_index/potential.py
   1·99 clipping → Min-Max → 가중합 → [0, 1]
   4-Method 강건성 (Equal/Entropy/PCA/Spearman) 검증 포함
   잠재력 r = 0.97 / 0.93 (Equal·Entropy 두 트랙 모두 수렴)

---

## B013 처리 체인 (3단계, raw → transit_norm)

Step 1 → src/0_preprocess/b013.py (캠퍼스)
   raw 거래내역 → 정류장·역 단위 streaming → 행정동(버스)/역(지하철) 월별 + YoY 산출
   출력: b013_bus_2023_2025_all.csv, b013_subway_2023_2025_all.csv,
        b013_bus_yoy_2023_2025.csv, b013_subway_yoy_2023_2025.csv

Step 2 → src/0_preprocess/b013_combine.py
   b013.py 출력 두 개 → transit_index.py 입력 형식 변환
   - 지하철 역 → 행정동 매핑 (Cervero & Duncan 2002, 500m 반경 면적 가중)
   - "지표" 컬럼 파싱 (트랙·수단·방향 분리)
   - 유입 필터 + 행정동×사용년월 wide pivot + 교통 종합 산출
   추가 raw 필요: data/raw/b013_subway_station/seoul_subway_station_coords.csv
   출력: data/raw/b013/B013_행정동별_교통이원트랙유입_월별상세.csv

Step 3 → src/2_index/transit_index.py
   행정동별 2년 YoY 평균 (모든 월 평균 * 2년 평균) → 클리핑 + Min-Max + 코드변환
   통계청 → 행자부 코드 EXCEPTION 매핑 10동 (강북 6쌍 + 강남·강동 4동)

---

## Y축 인프라 공백 지수 (심야 트랙)

infrastructure_gap_3var = 0.45 * cctv_void + 0.30 * police_void + 0.25 * light_void

학술 근거 (Welsh-Farrington 메타분석 효과크기 순서):
- CCTV (0.45): Welsh & Farrington (2009), Piza et al. (2019)
- 파출소 (0.30): Braga et al. (2019), Turchan & Braga (2024)
- 보안등 (0.25): Welsh, Farrington & Douglas (2022)
- 7:5:4 비율, 합계 100% 재정규화

→ src/2_index/infra_gap.py
   범죄 분위수는 합산 공식에서 분리, 보조 레이어로만 결합 (방안 B)

입력 3종 (0_preprocess/):
- cctv_void.py:    SHP → 공간조인 → 면적비례(광진·서대문·금천) + 수궁동 보정 → 1-x 변환
- police_void.py:  위경도 → centroid 최단거리 → Min-Max (1-x 변환 없음)
- light_void.py:   25개 자치구 csv → 도봉·서대문·송파·성동 4가지 결측보정 → 1-x 변환
                   (노션 정통 12단계 중 Step 1-4, 8-12)

---

## 보안등 처리 체인 (raw → light_void)

노션 정통 12단계 (보안등 데이터 구축 및 처리 로그):
- Step 1-4 (25구 통합 + 좌표 분리)        → light_void.py 내부
- Step 5-7 (4구 좌표 없는 데이터 geocoding) → light_geocode.py 별도 모듈
- Step 8-12 (spatial join + 결측 보정 + 밀도) → light_void.py 내부

→ src/0_preprocess/light_geocode.py
   좌표 없는 4구 (동대문·마포·송파·용산)의 주소 → Kakao 로컬 API → 위경도 → 행정동 매핑
   용산구: 주소 정규화 적용 (geocoding 성공률 향상)
   출력: 서울시_보안등_좌표0개구_행정동매핑_용산반영.csv
        → light_void.py의 --geocoded-supplement 입력

→ src/0_preprocess/light_void.py
   25개 자치구 CSV 통합 + spatial join + 4구 geocoding 결합 + 결측 4구 보정

결측 보정 (Step 12):
- 도봉구 (10동): 구 평균 193.84개/km² (실측 4동 기반)
- 서대문구 (12동): 구 평균 79.13개/km² (실측 2동 기반)
- 송파구 (4동): 구 평균 251.69개/km² (실측 23동 기반)
- 성동구 (17동, 전체): 인접 4구 평균 831.36개/km² (광진·동대문·중구·용산)

---

## Y축 저녁 매출 (저녁 트랙 메인)

evening_sales_norm = OA-22175 (추정매출-행정동) 17~21시 매출의 PercentileRank

24개 야간 부적합 업종 CS 코드 제외 (분석결과서 페이지 30 박스 1·2 정합)
신사동 강남(11680510)/관악(11620685) 분리
상일동 행자부 분리 (상일1·2동 동일값 분배)
B079 카드매출 원본은 캠퍼스 외부 반출 불가하므로 동등 시간대별 추정매출 OA-22175 외부 사용

→ src/2_index/evening_sales.py
   행정동 마스터: raw bnd_dong_11_2025_2Q.shp 직접 읽기 (geopandas attribute)
   API 키: 환경변수 SEOUL_OPENAPI_KEY 또는 --api-key 인자

---

## 매출 미실현도 (gap_norm) — 심야 보조

per_capita = sum_card / pop_avg
per_capita_rel = per_capita / Seoul_median(per_capita)
gap_index = 1 - per_capita_rel
gap_norm = PercentileRank(gap_index) ∈ [0, 1]

코드 매핑: B079 10자리 → B040 8자리 (강북 6쌍 수동 매핑 예외)
이상치 5동 제외: 11710631·11710647·11740520 (산식 안정성),
              11740525·11740526 (상일동 분할)
EXCEPTION 통계청→행자부 7동 매핑 (강북 6쌍 + 개포1동)

→ src/2_index/gap_index.py

---

## 마스터 테이블 — 5개 레이어 통합

→ src/2_index/master_table.py
   입력 5개:
   1. infra_gap → infrastructure_gap_final.csv (인프라 공백, 426동 기준 베이스)
   2. potential → potential_evening.csv + potential_late.csv (잠재력 트랙별)
   3. gap_index → b040_b079_gap_final.csv (매출 미실현도, 행자부 코드)
   4. evening_sales → evening_sales_final.csv (저녁 매출 절대값, 행자부 코드)
   5. crime_context → seoul_dong_crime_context_2023_2024.csv (자치구 범죄 분위수)

   NEW_TO_OLD_MAP: 잠재력 신코드 → gap.csv 구코드 (강북 6쌍)
   4종 결측 플래그: is_gap_imputed, is_evening_imputed, is_potential_imputed, is_crime_imputed
   둔촌1동·개포3동 사전 제거 + 신축 부문 중앙값 0.501 대체

   출력 3개: master_table_full.csv, master_table_evening.csv, master_table_late.csv

---

## 범죄 맥락 변수 처리 체인 (v1 9.1 시간대 결합 정통)

→ src/0_preprocess/crime_context.py
   두 raw 데이터 결합으로 자치구별 야간 강력+폭력 추정:

   입력 1: data/raw/범죄데이터/전국_범죄발생지.csv
           → 서울 25구 강력·폭력 별도 건수 (연도별)
   입력 2: data/raw/범죄데이터/전국_범죄발생시간.csv
           → 전국 범죄 유형별 시간대 분포 (8개 3시간 시간대)

   처리 흐름 (v1 9.1 정통):
   1. 자치구별 강력·폭력 건수 추출 (연도별)
   2. 야간 비율 산출 = (18-21 + 21-24 + 00-03 + 03-06) / 전체 24시간
      → 강력범죄·폭력범죄 각각 별도 비율
   3. 자치구 야간 추정 = 강력 * 야간_강력_비율 + 폭력 * 야간_폭력_비율
   4. 자치구별 야간 분위수 (rank percentile, 25개 구 기준, 연도별)
   5. 행정동에 자치구 야간 분위수 동일 부여 (Area Proportional Assignment)

   가정 (한계 명시):
   - 서울 자치구별 야간 범죄 비율이 전국 평균과 유사
   - 시간대 데이터에 자치구 분리 없음 → 전국 비율 일괄 적용

   메인 사용 컬럼: 야간_범죄맥락분위수_권장
   (master_table.py crime_percentile 컬럼에 매핑, 자동 우선 사용)

   보조 컬럼 (참조용, 검증·비교): 범죄맥락분위수_권장 (시간대 결합 없는 전체 분위수)

   --skip-night 인자로 시간대 결합 생략 가능 (전체 분위수만 산출)

---

## 3단계: 50/50 매트릭스 (median split)

저녁 트랙:
- X = potential_evening (percentile)
- Y = evening_sales_amt (percentile)  ← 메인 Y축, 보조 아님
- 후보 = 우하_잠재력높음_매출낮음
- 결과: 16동

심야 트랙:
- X = potential_late (percentile)
- Y = infrastructure_gap_3var (percentile)  ← 메인 Y축
- 점 크기 = gap_norm (보조)
- 점 색상 = crime_percentile (보조)
- 후보 = 우상_잠재력높음_인프라공백높음
- 결과: 21동

공통 6동, 통합 31동.

→ src/3_matrix.py
   출력: master_80.csv, candidates_{evening,late}.csv, {evening,late}_quadrants.csv

---

## 4단계: 후보지 정합성 점검

로드뷰·검색 기반 단일 점검 (야간경제 인프라 실재 여부).
- 정책 대상: 20동 (양트랙 3 + 저녁만 7 + 심야만 10)
- 보류: 5동 (자양2, 창4, 잠실6, 천호3, 도봉2)
- 부적합: 6동 (5유형: 업무·교통결절점, 신축·개발, 관광·경관,
              시장단일의존, 대형시설의존)

공통 6동 중 50%(3동)만 정합성 통과 — false positive 식별 가치 입증

---

## 정책 우선순위 산출

기준: 80동 분석대상 (2차 필터 통과 동 전체)
- 426동 기준 percentile은 2차 필터·매트릭스 맥락과 동떨어짐

저녁 트랙 (2축):
- 메인: rank_80(잠재력↑) + rank_80(저녁 매출↓)
- 보조: potential_evening_pct + (1 - evening_sales_pct)

심야 트랙 (4요소):
- 메인: rank_80(잠재력↑) + rank_80(인프라공백↑) + rank_80(범죄↑) + rank_80(매출미실현↑)
- 보조: potential_late_pct + infrastructure_gap_pct + crime_percentile + gap_norm

대표 6동: 숭인2 (저녁1) · 창1 (저녁2) · 용산2가 (공통1) · 망원1 (공통2)
        · 문정2 (심야1) · 가양1 (심야2)
도봉2 보류 후 자동 승급 반영 (숭인2→저녁1, 창1→저녁2)

범죄 레이어 = WHEN (예산 배정 순서, 합산엔 영향 없음)
- Tier 1 (즉시 투자): 집중투자형 + 범죄 최상위
- Tier 2 (단계 투자): 집중투자형 + 범죄 중간
- Tier 3 (확장 투자): 집중투자형 + 범죄 하위 또는 잔여

→ src/4_priority.py

---

## 5단계: 인터랙티브 지도 시각화

단일 인터랙티브 HTML + Layer Control:
- 배경: 서울 전체 426동
- 양트랙 후보 3동: "#facc15"
- 저녁만 후보 7동: "#2563eb"
- 심야만 후보 10동: "#7c3aed"
- 보류 5동: 회색
- 부적합 6동: 회색
- 대표 6동: 라벨

SHP: bnd_dong_11_2025_2Q (통계청 2025 2분기 행정동 경계, EPSG:5179)

→ src/5_viz.py

---

## 학문적 앵커

- Anderson 1976: USGS 67% 거주비율 임계
- Welsh-Farrington 메타분석: 안전 인프라 가중치 근거
- Evans 2012 (TBR): 야간경제 이론적 정의
- Cervero & Duncan 2002: 지하철 500m 영향권 (b013_combine.py 면적 가중)
- 4-method robustness verification:
    Equal weight / Entropy / PCA / Spearman
    잠재력 r = 0.97 / 0.93
    인프라 공백 r = 0.97 / 0.98

---

## 데이터셋

캠퍼스 데이터 (반출 제약):
- B013: 대중교통 1회권 (b013.py 캠퍼스 실행 + b013_combine.py 외부 결합)
- B040: 내국인 생활인구 (streaming 처리)
- B079: 카드매출 (캠퍼스 내 growth/gap_norm 산출만)

외부 공개 데이터:
- OA-22172: 상가업소 점포 단위 → opening 산출 (opening_rate.py)
- OA-22175: 추정매출-행정동 → evening_sales_norm 산출 (B079 반출 불가 대체)
- CCTV SHP: 서울시 CCTV 통합 정보 + 종로·도봉 별도 + 광진·서대문·금천 면적 비례 배분
- 파출소·지구대 csv: 공공데이터포털 경찰청 위치 정보 (243개 시설)
- 보안등 csv: 서울시 25개 자치구 통합 (자치구별 CSV → 좌표 spatial join + 4구 geocoding 보강)
- 범죄 데이터: 자치구 단위 분위수 (행정동별 발생건수 아님)
- 지하철 역 좌표: 서울 열린데이터광장 (b013_combine.py 행정동 매핑용)
- bnd_dong_11_2025_2Q SHP: 통계청 2025 2분기 행정동 경계 (EPSG:5179)

---

## 폐기 항목 (보고서 §11 한계에서만 언급)

- K-means / 야간특화지수 (2026-05-11 결정)
  → src/auxiliary/kmeans_clustering.py 에 LEGACY 코드만 보존
- D유형 사전 제외 개념 (K-means와 함께)
- Moran's I / LISA 공간자기상관
- 변수 합산 (cctv 35% + 파출소 25% + 보안등 20% + 범죄 20%)
  → crime 분리, 3변수 (45/30/25)
"""
