# Serendipity Engine

사용자의 최근 관심사를 분석해 **낯설지만 연결 가능한 개념**을 추천하는 실험적 세렌디피티(Serendipity) 탐색 프로젝트입니다.  
한국어 문장을 기반으로 핵심 개념을 추출하고, 위키백과 표제어 임베딩 공간에서 의미적으로 먼 영역을 찾은 뒤, 중간 연결 경로를 만들어 새로운 탐색 지점을 제안합니다.

## 주요 기능

- 한국어 입력 문장에서 핵심 개념(명사/복합명사) 추출
- KeyBERT + KoSBERT 임베딩 기반 관심사 벡터(Interest Nebula) 생성
- 전체 개념 맵에서 의미적으로 먼 영역(Antipode) 탐색
- 클러스터링(K-Means)으로 낯선 테마 후보 도출
- 시작 관심사와 낯선 테마 사이의 연결 경로(Bridge) 생성
- Streamlit UI를 통한 상호작용형 결과 확인

## 프로젝트 구조

- `main.py`  
  Streamlit 앱 본체. 모델 로드, 키워드 추출, 반대편 탐색, 연결 경로 생성, UI 렌더링 로직 포함
- `build_semantic_map.py`  
  한국어 위키백과 덤프에서 유효 표제어를 추출하고 임베딩 벡터를 생성/저장
- `test_model.py`  
  임베딩 모델 및 KeyBERT 로딩/동작 간단 점검 스크립트
- `text_vector.py`  
  코사인 유사도 기반 벡터 연산 로직 간단 점검 스크립트
- `test_sparql.py`  
  Wikidata SPARQL 엔드포인트 연결 점검 스크립트

## 실행 환경

Python 3.10+ 권장

주요 의존성:

- `streamlit`
- `numpy`
- `scikit-learn`
- `sentence-transformers`
- `keybert`
- `SPARQLWrapper`
- `kiwipiepy`
- `tqdm`

예시 설치:

```bash
pip install streamlit numpy scikit-learn sentence-transformers keybert SPARQLWrapper kiwipiepy tqdm
```

## 빠른 시작

### 1) 의미 지도 데이터 준비 (선택이지만 앱 핵심 기능에 필요)

`main.py`는 아래 파일을 읽어 전체 개념 맵을 사용합니다.

- `korean_wiki_titles.json`
- `korean_wiki_vectors.npy`

해당 파일이 없다면 `build_semantic_map.py`를 실행해 생성할 수 있습니다.

```bash
python build_semantic_map.py
```

> 참고: 스크립트는 `kowiki-latest-pages-articles.xml.bz2` 파일이 같은 디렉터리에 있다고 가정합니다.
> 덤프 파일은 Wikimedia Dumps(https://dumps.wikimedia.org/kowiki/latest/)에서 받을 수 있습니다.

### 2) 앱 실행

```bash
streamlit run main.py
```

브라우저에서 열리는 UI에 최근 관심사(줄바꿈 구분)를 입력하면 추천 경로를 확인할 수 있습니다.

## 점검용 스크립트 실행

```bash
python test_model.py
python text_vector.py
python test_sparql.py
```

각 파일은 간단한 동작 점검용 스크립트 형태이며, 표준 테스트 프레임워크 기반의 단위 테스트 세트는 아닙니다.
