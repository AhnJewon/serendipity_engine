import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from SPARQLWrapper import SPARQLWrapper, JSON
from sklearn.metrics.pairwise import cosine_similarity
from kiwipiepy import Kiwi
import random
import json

# 모델 및 데이터 로딩 (Streamlit 캐싱으로 앱 재실행 시 모델을 다시 로드하지 않도록 최적화)
@st.cache_resource
def load_models():
    """사전 학습된 모델들을 로드합니다."""
    print("임베딩 모델 및 키워드 모델 로딩 중...")
    embedding_model = SentenceTransformer('jhgan/ko-sbert-sts')
    kw_model = KeyBERT(embedding_model)
    kiwi = Kiwi()
    print("모델 로딩 완료.")
    return embedding_model, kw_model, kiwi

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
    
embedding_model, kw_model, kiwi = load_models()
ALL_CONCEPTS, ALL_CONCEPT_VECTORS = load_concept_map(embedding_model)

# --- 함수 정의 ---

def extract_noun_candidates(text):
    """주어진 텍스트에서 명사 후보들을 추출합니다."""
    tokens = kiwi.tokenize(text)
    candidates = []
    for t in tokens:
        # NNG(일반명사), NNP(고유명사), SL(외국어)만 추출
        if t.tag in ['NNG', 'NNP', 'SL']:
            candidates.append(t.form)
    
    # 중복 제거 및 2글자 이상만 남김 (너무 짧은 단어 제외)
    return list(set([c for c in candidates if len(c) >= 2]))

def extract_key_concepts(search_history):
    """자연어 검색 기록 리스트에서 핵심 개념들을 추출합니다."""
    if not search_history:
        return []
    
    full_text = " ".join(search_history)

    # 형태소 분석으로 명사 후보군 추출 (조사 제거됨)
    noun_candidates = extract_noun_candidates(full_text)

    if not noun_candidates:
        return []
    
    keywords_with_scores = kw_model.extract_keywords(
        full_text, 
        candidates=noun_candidates, 
        top_n=5
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

def find_semantic_antipode(interest_vector, top_n=30, diversity=0.4):
    """
    관심 벡터와 의미적으로 멀면서, 동시에 서로 다양한 개념들을 찾습니다.
    """
    if interest_vector is None:
        return []

    # 1. 모든 개념과의 코사인 유사도 계산
    user_sims = cosine_similarity(interest_vector.reshape(1, -1), ALL_CONCEPT_VECTORS)[0]
   
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
                dist_to_selected = 1.0  # 이미 뽑힌 게 없으면 거리는 최대(1.0)
            else:
                # 현재 후보와 이미 뽑힌 애들 간의 유사도 계산
                selected_vectors = ALL_CONCEPT_VECTORS[selected_indices]
                current_vector = ALL_CONCEPT_VECTORS[idx].reshape(1, -1)
                sims_to_selected = cosine_similarity(current_vector, selected_vectors)[0]
                # 가장 유사한 놈과의 거리를 구함
                dist_to_selected = 1 - np.max(sims_to_selected)
            
            # 점수 계산 (다양성 반영)
            score = (1 - diversity) * dist_to_user + (diversity * dist_to_selected)
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        if best_idx != -1:
            selected_indices.append(best_idx)
            
    return [ALL_CONCEPTS[i] for i in selected_indices]

def check_and_promote_concept(concept):
    
    # 추천된 개념이 '영화', '책' 등 지엽적 인스턴스라면, 그 작품의 '장르'나 '주제'로 승격시킴.
    
    endpoint_url = "https://query.wikidata.org/sparql"
    sparql = SPARQLWrapper(endpoint_url)
    
    # P31: instance of, P136: genre, P921: main subject
    # 영화(Q11424), 책(Q571), 소설(Q7725) 인지 확인하고, 맞다면 장르나 주제를 가져옴
    query = f"""
    SELECT ?type ?promotionLabel WHERE {{
      SERVICE wikibase:mwapi {{
        bd:serviceParam wikibase:api "EntitySearch".
        bd:serviceParam wikibase:endpoint "www.wikidata.org".
        bd:serviceParam mwapi:search "{concept}".
        bd:serviceParam mwapi:language "ko".
        ?item wikibase:apiOutputItem mwapi:item.
      }}
      
      # 아이템이 영화, 책, 소설 중 하나인지 확인
      VALUES ?targetClass {{ wd:Q11424 wd:Q571 wd:Q7725 }}
      ?item wdt:P31 ?targetClass.
      
      # 승격할 상위 개념 찾기 (장르 또는 주제)
      OPTIONAL {{ ?item wdt:P136 ?genre. }}
      OPTIONAL {{ ?item wdt:P921 ?subject. }}
      
      BIND(COALESCE(?subject, ?genre) AS ?promotion)
      
      SERVICE wikibase:label {{ 
        bd:serviceParam wikibase:language "ko". 
        ?promotion rdfs:label ?promotionLabel.
      }}
    }} LIMIT 1
    """
    
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(5) # 짧게 설정
    
    try:
        results = sparql.query().convert()
        bindings = results["results"]["bindings"]
        
        if bindings:
            # 승격된 개념이 있으면 반환 (예: '매트릭스' -> '사이버펑크')
            promoted = bindings[0].get("promotionLabel", {}).get("value")
            if promoted:
                return promoted, True # (변경된 이름, 변경됨 여부)
        
        return concept, False # 변경 없음
        
    except Exception:
        return concept, False

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

      # 양쪽에서 공통된 연결 고리 탐색 (최대 1단계)
      ?concept1 (wdt:P31|wdt:P279)* ?bridge.
      ?concept2 (wdt:P31|wdt:P279)* ?bridge.
      
      # 연결 고리는 위키데이터 엔티티여야 함
      FILTER(isIRI(?bridge))
      
      SERVICE wikibase:label {{ 
        bd:serviceParam wikibase:language "ko". 
        ?bridge rdfs:label ?bridgeLabel.
      }}
    }} LIMIT 1
    """
    
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(5)
            
    try:
        results = sparql.query().convert()
        keywords = results["results"]["bindings"]
        if keywords:
            return keywords[0]["bridgeLabel"]["value"]
        return None
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
                st.info(f"**분석된 핵심 키워드:** {', '.join(concepts)}")
                interest_vector = get_interest_nebula_vector(concepts, embedding_model)
                
                # Phase 2: 의미적 반대편 탐색 (MMR 적용)
                with st.spinner("2. 의미의 우주를 탐색하여 낯선 행성(Antipode)을 찾는 중입니다..."):
                    # 넉넉하게 30개를 뽑습니다 (필터링 및 연결성 검증을 위해)
                    candidates = find_semantic_antipode(interest_vector, top_n=30, diversity=0.4)
                
                # Phase 3: 필터링 및 연결 고리 검증
                with st.spinner("3. 논리적 연결 고리(Bridge)를 건설 중입니다..."):
                    main_concept = concepts[0] # 가장 비중 있는 키워드
                    final_antipode = None
                    
                    progress_bar = st.progress(0)
                    
                    for i, candidate in enumerate(candidates[:10]): # 시간 관계상 상위 10개만 검사
                        progress_bar.progress((i + 1) / len(candidates[:10]))
                        # A. 개념 승격 체크 (영화면 장르로 바꿈)
                        promoted_cand, is_promoted = check_and_promote_concept(candidate)

                        if is_promoted:
                            print(f"Promotion: {candidate} -> {promoted_cand}")
                        
                        # 연결 고리 존재 여부 확인 (SPARQL)
                        bridges = find_bridge_keywords(main_concept, promoted_cand)

                        if  bridges:
                            final_antipode = {
                                "start": main_concept,
                                "bridge": bridges,
                                "end": promoted_cand,
                                "original": candidate if is_promoted else None
                            }
                            break
                    
                    progress_bar.empty() # 진행바 숨기기

                # 최종 결과 출력
                if final_antipode:
                    st.success(f"**새로운 탐험 영역 발견:** #{final_antipode['end']}")
                    st.markdown("---")
                    
                    if final_antipode['original']:
                        st.caption(f"(원래 발견된 '{final_antipode['original']}'에서 더 깊은 주제로 확장되었습니다.)")
                    
                    st.info(f"**논리적 경로:** {final_antipode['start']} ➡️ ({final_antipode['bridge']}) ➡️ {final_antipode['end']}")
                    st.markdown(f"**제안:** `{final_antipode['start']}`에 익숙하시다면, 공통점 `{final_antipode['bridge']}`을 공유하는 `{final_antipode['end']}`의 관점에서도 생각해보세요.")
                else:
                    st.warning("적절한 연결 고리를 찾지 못했습니다. 조금 더 다양한 관심사를 입력해보세요.")
