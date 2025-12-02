import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from SPARQLWrapper import SPARQLWrapper, JSON
from sklearn.metrics.pairwise import cosine_similarity
import random
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

def find_semantic_antipode(interest_vector, concept_vectors, all_concepts, top_n=30, diversity=0.4):
    """
    관심 벡터와 의미적으로 멀면서, 동시에 서로 다양한 개념들을 찾습니다.
    """
    if interest_vector is None:
        return []

    # 1. 모든 개념과의 코사인 유사도 계산
    user_sims = cosine_similarity(interest_vector.reshape(1, -1), concept_vectors)[0]
   
    # 2. 초기 후보군 선정
    pool_size = 500
    sorted_indices = np.argsort(user_sims) # 오름차순(유사도 낮은 순)
    candidate_indices = sorted_indices[:pool_size]
    
    selected_indices = []
    
    # 3. Greedy Selection (MMR 알고리즘)
    for _ in range(top_n):
        best_idx = -1
        best_score = -float('inf')

        for idx in candidate_indices:
            if idx in selected_indices:
                continue
            
            # A. 사용자 관심사와의 거리 (멀수록 좋음)
            dist_to_user = 1 - user_sims[idx]
            
            # B. 이미 선택된 개념들과의 거리 (다양성)
            if not selected_indices:
                dist_to_selected = 1.0 
            else:
                selected_vectors = concept_vectors[selected_indices]
                current_vector = concept_vectors[idx].reshape(1, -1)
                sims_to_selected = cosine_similarity(current_vector, selected_vectors)[0]
                dist_to_selected = 1 - np.max(sims_to_selected)
            
            # 점수 계산 (다양성 반영)
            score = (1 - diversity) * dist_to_user + (diversity * dist_to_selected)
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        if best_idx != -1:
            selected_indices.append(best_idx)
            
    return [all_concepts[i] for i in selected_indices]


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
    sparql.setTimeout(10)
            
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
        
        if len(history_list) < 2:
            st.warning("더 정확한 분석을 위해 2개 이상의 관심사를 입력해주시면 좋습니다.")

        with st.spinner("1. 당신의 지적 성운(Interest Nebula)을 분석 중입니다..."):
            # Phase 1: 핵심 개념 추출 및 벡터화
            concepts = extract_key_concepts(history_list)
            
            if not concepts:
                st.error("입력에서 유의미한 핵심 개념을 찾지 못했습니다. 조금 더 자세히 적어주세요.")
            else:
                st.info(f"**🔍 분석된 핵심 키워드:** {', '.join(concepts)}")
                interest_vector = get_interest_nebula_vector(concepts, embedding_model)
                
                # Phase 2: 의미적 반대편 탐색 (MMR 적용)
                with st.spinner("2. 의미의 우주를 탐색하여 낯선 행성(Antipode)을 찾는 중입니다..."):
                    # 넉넉하게 30개를 뽑습니다 (필터링 및 연결성 검증을 위해)
                    candidates = find_semantic_antipode(interest_vector, ALL_CONCEPT_VECTORS, ALL_CONCEPTS, top_n=30, diversity=0.4)
                
                # Phase 3: 필터링 및 연결 고리 검증 (기존의 단순 random.choice 대신 검증 루프 사용)
                with st.spinner("3. 논리적 연결 고리(Bridge)를 건설 중입니다..."):
                    main_concept = concepts[0] # 가장 비중 있는 키워드
                    final_antipode = None
                    final_bridges = []
                    
                    progress_bar = st.progress(0)
                    
                    for i, candidate in enumerate(candidates):
                        progress_bar.progress((i + 1) / len(candidates))
                        
                        # 연결 고리 존재 여부 확인 (SPARQL)
                        bridges = find_bridge_keywords(main_concept, candidate)
                        if bridges:
                            final_antipode = candidate
                            final_bridges = bridges
                            break
                    
                    progress_bar.empty() # 진행바 숨기기

                # 최종 결과 출력
                if final_antipode:
                    st.success(f"🎯 **새로운 탐험 영역 발견:** #{final_antipode}")
                    st.markdown("---")
                    
                    # 연결 고리 시각화
                    path_steps = [f"**{main_concept}**"] + [f"`{b}`" for b in final_bridges] + [f"**{final_antipode}**"]
                    path_md = " ➡️ ".join(path_steps)
                    
                    st.write("다음의 논리적 경로를 통해 당신의 관심사와 연결됩니다:")
                    st.info(path_md)
                    
                    st.caption(f"💡 '{main_concept}'와(과) '{final_antipode}' 사이의 관계를 Wikidata 지식 그래프에서 찾았습니다.")
                else:
                    st.warning("아쉽게도 논리적으로 연결 가능한 '의미적 반대편'을 찾지 못했습니다.")
                    st.write("관심사와 너무 동떨어진 개념만 남았거나, 지식 그래프 연결이 끊겨있을 수 있습니다. 다른 주제로 다시 시도해보세요!")
                    if candidates:
                        st.write(f"(참고: 후보로 '{candidates[0]}' 등이 발견되었으나 연결 고리가 부족했습니다.)")
                        print(f"Debug: Candidates were {candidates}")

    else:
        st.error("검색 기록을 입력해주세요.")
