import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from SPARQLWrapper import SPARQLWrapper, JSON
from sklearn.metrics.pairwise import cosine_similarity

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
    # 의미 지도 개념 목록과 벡터를 로드
    print("의미 지도 로딩 중...")
    # 위키백과 표제어 사용전 테스트 코드
    all_concepts = ['인공지능', '딥러닝', '자율주행', '도시 계획', '스토아 철학', '고대사', '미술사', '조류학', '양자역학', '분자생물학', '패션 디자인', '건축학', '경제학', '사회학', '심리학']
    all_concept_vectors = _embedding_model.encode(all_concepts)
    print("의미 지도 로딩 완료.")
    return all_concepts, all_concept_vectors

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
