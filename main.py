import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from SPARQLWrapper import SPARQLWrapper, JSON
from sklearn.metrics.pairwise import cosine_similarity
import json

# 모델 및 데이터 로딩 (Streamlit 캐싱으로 앱 재실행 시 모델을 다시 로드하지 않도록 최적화)
@st.cache_resource
def load_models():
    """사전 학습된 모델들을 로드합니다."""
    print("임베딩 모델 및 키워드 모델 로딩 중...")
    embedding_model = SentenceTransformer('jhgan/ko-sbert-sts')
    kw_model = KeyBERT(embedding_model)
    print("모델 로딩 완료.")
    return embedding_model, kw_model

@st.cache_data
def load_concept_map(_embedding_model):
    """생성된 위키백과 의미 지도 파일들을 로드합니다."""
    title_path = 'korean_wiki_titles.json'
    vector_path = 'korean_wiki_vectors.npy'
    
    try:
        print("대규모 의미 지도 로딩 중 (약간의 시간이 소요됩니다)...")
        
        # 1. 타이틀 리스트 로드
        with open(title_path, 'r', encoding='utf-8') as f:
            all_concepts = json.load(f)
            
        # 2. 벡터 행렬 로드
        all_concept_vectors = np.load(vector_path)
        
        print(f"로딩 완료! (총 {len(all_concepts)}개의 개념)")
        return all_concepts, all_concept_vectors
        
    except FileNotFoundError:
        st.error("'의미 지도' 데이터 파일이 없습니다. 'build_semantic_map.py'를 먼저 실행해주세요.")
        # 파일이 없을 경우를 대비한 비상용 더미 데이터
        dummy_concepts = ['데이터 없음', '스크립트 실행 필요']
        dummy_vectors = _embedding_model.encode(dummy_concepts)
        return dummy_concepts, dummy_vectors
    
embedding_model, kw_model = load_models()
ALL_CONCEPTS, ALL_CONCEPT_VECTORS = load_concept_map(embedding_model)

# --- 함수 정의 ---

def extract_key_concepts(search_history):
    """자연어 검색 기록 리스트에서 핵심 개념들을 추출합니다."""
    if not search_history:
        return []
    
    full_text = " ".join(search_history)
    
    keywords_with_scores = kw_model.extract_keywords(
        full_text, 
        keyphrase_ngram_range=(1, 2), 
        stop_words=None, 
        top_n=10
    )
    
    # (키워드, 점수) 튜플에서 키워드(문자열)만 추출하여 반환
    concepts = [keyword for keyword, score in keywords_with_scores]
    return concepts

def get_interest_nebula_vector(concepts, embedding_model):
    """핵심 개념 리스트를 바탕으로 사용자의 '관심 성운' 중심 벡터를 계산합니다."""
    if not concepts:
        return None
    
    concept_vectors = embedding_model.encode(concepts)
    weights = np.linspace(0.5, 1.5, len(concepts)) # 최신 검색어에 더 높은 가중치 부여
    weighted_avg_vector = np.average(concept_vectors, axis=0, weights=weights)
    
    return weighted_avg_vector

def find_semantic_antipode(interest_vector, concept_vectors, all_concepts, top_n=5):
    """관심 벡터와 가장 거리가 먼 개념(의미적 반대편)을 찾습니다."""
    if interest_vector is None:
        return []

    similarities = cosine_similarity(interest_vector.reshape(1, -1), concept_vectors)
    
    # 2D 배열인 similarities에서 첫 번째 행(1D 배열)을 선택하여 정렬
    antipode_indices = np.argsort(similarities[0])[:top_n]
    
    # 가장 이질적인 개념들을 반환합니다.
    antipode_concepts = [all_concepts[i] for i in antipode_indices]
    return antipode_concepts

def find_bridge_keywords(concept1, concept2):
    """Wikidata에서 두 개념 사이의 연결 경로를 탐색하여 키워드를 찾습니다."""
    endpoint_url = "https://query.wikidata.org/sparql"
    
    # 두 개념을 잇는 중간 개념(들)을 찾는 예시 쿼리
    # P31(instance of) 또는 P279(subclass of) 속성을 따라 최대 5단계까지 탐색
    query = f"""
    SELECT ?bridgeLabel WHERE {{
      SERVICE wikibase:mwapi {{
        bd:serviceParam wikibase:api "EntitySearch".
        bd:serviceParam wikibase:endpoint "www.wikidata.org".
        bd:serviceParam mwapi:search "{concept1}".
        bd:serviceParam mwapi:language "ko".
        ?concept1 wikibase:apiOutputItem mwapi:item.
      }}
      SERVICE wikibase:mwapi {{
        bd:serviceParam wikibase:api "EntitySearch".
        bd:serviceParam wikibase:endpoint "www.wikidata.org".
        bd:serviceParam mwapi:search "{concept2}".
        bd:serviceParam mwapi:language "ko".
        ?concept2 wikibase:apiOutputItem mwapi:item.
      }}
      
      ?concept1 (wdt:P31|wdt:P279)* ?bridge.
      ?concept2 (wdt:P31|wdt:P279)* ?bridge.
      
      FILTER(?concept1 != ?concept2 && ?bridge != wd:Q35120) # Entity(최상위 클래스)는 제외
      
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "ko". }}
    }} LIMIT 5
    """
    
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    try:
        results = sparql.query().convert()
        keywords = [result["bridgeLabel"]["value"] for result in results["results"]["bindings"]]
        return list(set(keywords)) # 중복 제거
    except Exception as e:
        print(f"SPARQL 쿼리 오류: {e}")
        return []

# Streamlit UI 구성 

st.set_page_config(layout="wide")
st.title("세렌디피티 검색 엔진")
st.write("당신의 생각의 감옥을 깨뜨릴 의외의 연결고리를 찾아드립니다.")

# 사용자 입력
search_history_input = st.text_area(
    "최근 관심있게 찾아본 주제나 질문들을 한 줄에 하나씩 입력해주세요.", 
    height=150,
    placeholder="예시:\n최근 AI 윤리 문제가 심각한데 스토아 철학으로 해결할 수 있을까?\n자율주행차의 트롤리 딜레마 사례 분석"
)

if st.button("새로운 발견 시작하기"):
    if search_history_input:
        history_list = [line.strip() for line in search_history_input.split('\n') if line.strip()]
        
        with st.spinner("당신의 지적 성운을 분석 중입니다..."):
            # Phase 1 실행
            concepts = extract_key_concepts(history_list)
            if not concepts:
                st.error("입력에서 핵심 개념을 추출하지 못했습니다. 더 자세히 입력해주세요.")
            else:
                st.info(f"**분석된 핵심 관심사:** {', '.join(concepts)}")
                
                interest_vector = get_interest_nebula_vector(concepts, embedding_model)
                
                antipode_concepts = find_semantic_antipode(interest_vector, ALL_CONCEPT_VECTORS, ALL_CONCEPTS, top_n=1)
                
                if not antipode_concepts:
                    st.warning("의미적 반대편을 찾지 못했습니다.")
                else:
                    # 리스트의 첫 번째 항목을 명확히 선택
                    main_concept = concepts[0]
                    antipode_concept = antipode_concepts[0] 
                    st.info(f"**새로운 탐험 영역 제안:** #{antipode_concept}")

                    # Phase 2 실행
                    with st.spinner(f"'{main_concept}'와(과) '#{antipode_concept}' 사이의 의외의 연결고리를 찾는 중..."):
                        bridge_keywords = find_bridge_keywords(main_concept, antipode_concept)
                        
                        if bridge_keywords:
                            st.success("💡 **연결고리 키워드를 발견했습니다!**")
                            st.write("다음 키워드들로 탐험을 시작해보세요:")
                            
                            # 키워드를 클릭 가능한 링크처럼 보이게 만듬
                            keyword_md = " -> ".join([f"`{kw}`" for kw in bridge_keywords])
                            st.markdown(f"**`{main_concept}`** -> {keyword_md} -> **`{antipode_concept}`**")
                        else:
                            st.warning("두 개념을 잇는 직접적인 연결고리를 찾지 못했습니다. 하지만 이것 자체로 새로운 발견일 수 있습니다!")
    else:
        st.error("검색 기록을 입력해주세요.")
