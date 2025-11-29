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
    # 사전 학습된 모델들을 로드
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


# 함수 정의

def extract_key_concepts(search_history):
    #자연어 검색 기록 리스트에서 핵심 개념들을 추출
    if not search_history:
        return []
    
    full_text = " ".join(search_history)
    
    keywords_with_scores = kw_model.extract_keywords(
        full_text, 
        keyphrase_ngram_range=(1, 2), 
        stop_words=None, 
        top_n=10
    )
    
    # 튜플에서 키워드만 추출하여 반환
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
    # 관심 벡터와 가장 거리가 먼 개념 탐색
    if interest_vector is None:
        return []

    similarities = cosine_s함
                            keyword_md = " ➡️ ".join([f"`{kw}`" for kw in bridge_keywords])
                            st.markdown(f"**`{main_concept}`** ➡️ {keyword_md} ➡️ **`{antipode_concept}`**")
                        else:
                            st.warning("두 개념을 잇는 직접적인 연결고리를 찾지 못했습니다. 하지만 이것 자체로 새로운 발견일 수 있습니다!")
    else:
        st.error("검색 기록을 입력해주세요.")
