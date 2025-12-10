import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from SPARQLWrapper import SPARQLWrapper, JSON
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from kiwipiepy import Kiwi
import random
import json
from difflib import SequenceMatcher

# 모델 및 데이터 로딩 (Streamlit 캐싱으로 앱 재실행 시 모델을 다시 로드하지 않도록 최적화)
@st.cache_resource
def load_models():
    """사전 학습된 모델들을 로드"""
    print("임베딩 모델 및 키워드 모델 로딩 중...")
    embedding_model = SentenceTransformer('jhgan/ko-sbert-sts')
    kw_model = KeyBERT(embedding_model)
    kiwi = Kiwi()
    print("모델 로딩 완료.")
    return embedding_model, kw_model, kiwi

@st.cache_data
def load_concept_map(_embedding_model):
    """생성된 위키백과 의미 지도 파일들을 로드"""
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
    """주어진 텍스트에서 명사 후보들을 추출"""
    tokens = kiwi.tokenize(text)
    candidates = []
    for t in tokens:
        # NNG(일반명사), NNP(고유명사), SL(외국어)만 추출
        if t.tag in ['NNG', 'NNP', 'SL']:
            candidates.append(t.form)
    
    # 중복 제거 및 2글자 이상만 남김 (너무 짧은 단어 제외)
    return list(set([c for c in candidates if len(c) >= 2]))

def extract_complex_candidates(text, valid_concepts_set):
    """
    텍스트에서 명사 및 복합 명사를 추출하되, 
    '위키백과 표제어(valid_concepts_set)'에 존재하는 개념만 추출출
    """
    print("\n[DEBUG]\textract_complex_candidates 시작")
    print(f"[DEBUG]\t입력 텍스트 길이: {len(text)} 문자")
    
    # 1. 텍스트를 줄바꿈 기준으로 분리
    lines = text.split('\n')
    candidates = set()
    all_extracted_nouns = []
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        tokens = kiwi.tokenize(line)
        
        nouns = []
        for t in tokens:
            # 명사(NNG, NNP, NR, SL)만 일단 수집
            if t.tag in ['NNG', 'NNP', 'NR', 'SL']:
                nouns.append(t.form)
        
        if nouns:
            all_extracted_nouns.extend(nouns)
        
        if not nouns: continue
        
        n = len(nouns)
        
        # --- N-gram 생성 및 Whitelist 검증 ---
        
        # 1. 단일 명사 검증
        for noun in nouns:
            if len(noun) < 2: continue # 1글자는 제외 (선택사항)
            
            # 위키백과 표제어에 있는 경우에만 후보로 등록
            if noun in valid_concepts_set:
                candidates.add(noun)
                print(f"[DEBUG]\t단일 명사 매칭: '{noun}'")
        
        # 2. 복합 명사(2단어, 3단어) 검증
        for i in range(n):
            # 2단어 결합
            if i + 1 < n:
                bigram = f"{nouns[i]} {nouns[i+1]}" 
                bigram_nospace = f"{nouns[i]}{nouns[i+1]}"
                
                if (bigram in valid_concepts_set) or (bigram_nospace in valid_concepts_set):
                    candidates.add(bigram)
                    print(f"[DEBUG]\t복합명사(2) 매칭: '{bigram}'")
            
            # 3단어 결합
            if i + 2 < n:
                trigram = f"{nouns[i]} {nouns[i+1]} {nouns[i+2]}"
                trigram_nospace = f"{nouns[i]}{nouns[i+1]}{nouns[i+2]}"
                
                if (trigram in valid_concepts_set) or (trigram_nospace in valid_concepts_set):
                    candidates.add(trigram)
                    print(f"[DEBUG]\t복합명사(3) 매칭: '{trigram}'")

    print(f"[DEBUG]\t전체 추출된 명사: {set(all_extracted_nouns)}")
    print(f"[DEBUG]\t최종 후보 개수: {len(candidates)}")
    print(f"[DEBUG]\t최종 후보 목록: {list(candidates)}\n")
    return list(candidates)

def extract_key_concepts(search_history):
    """자연어 검색 기록 리스트에서 핵심 개념들을 추출"""
    if not search_history:
        return []
    
    # 1. 원본 텍스트 합치기
    full_text = "\n".join(search_history)
    
    valid_concepts_set = set(ALL_CONCEPTS) 
    
    # 2. Kiwi로 복합 명사 후보군 추출
    complex_candidates = extract_complex_candidates(full_text, valid_concepts_set)
    print(f"추출된 후보군: {complex_candidates}")
    
    if not complex_candidates:
        return []

    # 본문 자체를 형태소 단위로 띄어쓰기
    tokenized_tokens = kiwi.tokenize(full_text)
    tokenized_text = " ".join([t.form for t in tokenized_tokens])
    print(f"\n[DEBUG]\t형태소 분리 텍스트 (처음 200자): {tokenized_text[:200]}...")

    # 3. 모델 실행
    print(f"[DEBUG]\tKeyBERT 실행 - 후보군 개수: {len(complex_candidates)}")
    keywords_with_scores = kw_model.extract_keywords(
        tokenized_text, 
        candidates=complex_candidates, 
        keyphrase_ngram_range=(1, 3),
        top_n=5,
        use_mmr=True, 
        diversity=0.3
    )
    
    print(f"\n[DEBUG]\tKeyBERT 결과 (키워드, 점수):")
    for kw, score in keywords_with_scores:
        print(f"  - {kw}: {score:.4f}")
    print()

    return [keyword for keyword, score in keywords_with_scores]

def get_interest_nebula_vector(concepts, embedding_model):
    """추출된 키워드들의 벡터를 가중 평균하여 '관심 성운' 벡터 생성"""
    if not concepts:
        return None
    
    print(f"\n[DEBUG]\t관심 성운 벡터 생성")
    print(f"[DEBUG]\t입력 개념: {concepts}")
    
    concept_vectors = embedding_model.encode(concepts)
    # 최신(또는 상위) 키워드에 가중치 부여
    weights = np.linspace(0.8, 1.2, len(concepts))
    print(f"[DEBUG]\t가중치: {weights}")
    
    weighted_avg_vector = np.average(concept_vectors, axis=0, weights=weights)
    print(f"[DEBUG]\t생성된 벡터 shape: {weighted_avg_vector.shape}\n")
    
    return weighted_avg_vector

def find_semantic_antipode_by_clustering(interest_vector, n_clusters=5, pool_size=1000):
    """
    1. 관심사와 가장 거리가 먼 pool_size개의 개념을 1차 선별
    2. 선별된 개념들을 K-Means로 n_clusters개로 군집화
    3. 각 군집의 중심에 가장 가까운 단어를 대표 키워드로 반환
    """
    if interest_vector is None:
        return []

    # 1. 전체 개념과의 코사인 유사도 계산
    print(f"\n[DEBUG]\t반대편 탐색 시작 (pool_size={pool_size}, n_clusters={n_clusters})")
    user_sims = cosine_similarity(interest_vector.reshape(1, -1), ALL_CONCEPT_VECTORS)[0]
    print(f"[DEBUG]\t유사도 범위: {user_sims.min():.4f} ~ {user_sims.max():.4f}")

    # 2. 유사도가 가장 낮은 하위 pool_size개 인덱스 추출 (오름차순 정렬)
    far_indices = np.argsort(user_sims)[:pool_size]
    print(f"[DEBUG]\t가장 먼 {pool_size}개 개념 선별 완료")
    print(f"[DEBUG]\t최하위 유사도 샘플 (5개): {user_sims[far_indices[:5]]}")
    
    far_vectors = ALL_CONCEPT_VECTORS[far_indices]
    far_concepts = [ALL_CONCEPTS[i] for i in far_indices]
    print(f"[DEBUG]\t샘플 개념들: {far_concepts[:10]}")
    
    # 3. K-Means 군집화 수행
    print(f"[DEBUG]\tK-Means 군집화 시작...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=None, n_init='auto')
    kmeans.fit(far_vectors)
    print(f"[DEBUG]\t군집화 완료 (inertia: {kmeans.inertia_:.2f})")
    
    centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    
    representative_concepts = []
    
    # 4. 각 군집별 대표 단어 추출
    print(f"\n[DEBUG]\t각 군집의 대표 단어 선정:")
    for i in range(n_clusters):
        # 해당 클러스터에 속하는 벡터들 마스킹
        mask = (cluster_labels == i)
        if not np.any(mask):
            print(f"  [군집 {i}] 비어있음")
            continue
            
        cluster_vectors = far_vectors[mask]
        cluster_concepts_list = np.array(far_concepts)[mask]
        print(f"  [군집 {i}] 크기: {len(cluster_concepts_list)}개")
        
        # 클러스터 중심과 해당 클러스터 내 단어들 간의 유사도 계산
        sims_to_center = cosine_similarity(centers[i].reshape(1, -1), cluster_vectors)[0]
        
        # 중심과 가장 가까운 단어 선택
        best_idx = np.argmax(sims_to_center)
        representative = cluster_concepts_list[best_idx]
        representative_concepts.append(representative)
        print(f"  [군집 {i}] 대표: '{representative}' (중심 유사도: {sims_to_center[best_idx]:.4f})")
        print(f"  [군집 {i}] 샘플: {list(cluster_concepts_list[:5])}")
    
    print(f"\n[DEBUG]\t최종 대표 개념들: {representative_concepts}\n")
    return representative_concepts

def check_and_promote_concept(concept):
    """
    개념 승격 시, '목록', '동음이의어' 같은 메타 데이터를 제외
    """
    print(f"\n[DEBUG]\t개념 승격(추상화) 시도: '{concept}'")
    endpoint_url = "https://query.wikidata.org/sparql"
    sparql = SPARQLWrapper(endpoint_url)
    
    query = f"""
    SELECT ?parentLabel WHERE {{
      SERVICE wikibase:mwapi {{
        bd:serviceParam wikibase:api "EntitySearch".
        bd:serviceParam wikibase:endpoint "www.wikidata.org".
        bd:serviceParam mwapi:search "{concept}".
        bd:serviceParam mwapi:language "ko".
        ?item wikibase:apiOutputItem mwapi:item.
      }}
      
      ?item wdt:P31|wdt:P279|wdt:P136 ?parent.
      
      # 무의미한 메타 데이터 ID 블랙리스트 필터링
      # Q13406463: 위키미디어 목록 (Wikimedia list article)
      # Q4167410: 동음이의어 문서 (Wikipedia disambiguation page)
      # Q11266439: 템플릿 (Template)
      FILTER(?parent NOT IN (wd:Q13406463, wd:Q4167410, wd:Q11266439))

      SERVICE wikibase:label {{ 
        bd:serviceParam wikibase:language "ko". 
        ?parent rdfs:label ?parentLabel. 
      }}
    }} LIMIT 1
    """
    
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(3) 
    
    try:
        results = sparql.query().convert()
        bindings = results["results"]["bindings"]
        
        if bindings:
            promoted = bindings[0].get("parentLabel", {}).get("value")
            # 영어 이름이 그대로 나오거나 입력과 같으면 패스
            if promoted and promoted != concept: 
                print(f"[DEBUG]\t승격 성공: '{concept}' -> '{promoted}'")
                return promoted, True
                
        print(f"[DEBUG]\t- 승격 실패 (유효한 상위 개념 없음)")
        return concept, False
        
    except Exception as e:
        print(f"[DEBUG]\t! SPARQL 에러: {e}")
        return concept, False
        
    except Exception as e:
        print(f"[DEBUG]\t! SPARQL 에러: {e}")
        return concept, False

def is_string_too_similar(s1, s2):
    """두 단어의 글자 구성이 너무 비슷하면 True 반환"""
    s1_clean = s1.replace(" ", "")
    s2_clean = s2.replace(" ", "")
    
    # 1. 포함 관계
    if s1_clean in s2_clean or s2_clean in s1_clean:
        return True
        
    # 2. 텍스트 유사도
    # 40% 이상 글자가 겹치면 차단
    if SequenceMatcher(None, s1_clean, s2_clean).ratio() > 0.4:
        return True
        
    return False

# 전역 변수를 직접 참조하여 연결고리 찾기
def find_multi_step_bridge(start_concept, end_concept):
    """
    ALL_CONCEPTS, ALL_CONCEPT_VECTORS, embedding_model 전역 변수를 사용합니다.
    """
    print(f"\n[DEBUG]\t벡터 연결고리 생성: '{start_concept}' -> ... -> '{end_concept}'")
    
    # 1. 시작점 벡터 찾기
    try:
        idx1 = ALL_CONCEPTS.index(start_concept)
        v1 = ALL_CONCEPT_VECTORS[idx1]
    except ValueError:
        # 리스트에 없으면 전역 모델로 즉시 인코딩
        v1 = embedding_model.encode([start_concept])[0]

    # 2. 끝점 벡터 찾기
    try:
        idx2 = ALL_CONCEPTS.index(end_concept)
        v2 = ALL_CONCEPT_VECTORS[idx2]
    except ValueError:
        v2 = embedding_model.encode([end_concept])[0]

    path = []
    # 중복 방지 집합
    seen_concepts = {start_concept, end_concept}
    
    # 3. 3단계 보간 (25%, 50%, 75%)
    steps = [0.25, 0.50, 0.75]
    
    for t in steps:
        interpolated_vector = (1 - t) * v1 + t * v2
        
        # 전역 벡터 행렬과 유사도 계산
        sims = cosine_similarity(interpolated_vector.reshape(1, -1), ALL_CONCEPT_VECTORS)[0]
        
        top_indices = np.argsort(sims)[::-1][:100]
        
        found_step = None
        for idx in top_indices:
            candidate = ALL_CONCEPTS[idx]
            
            # [필터링 1] 이미 나온 단어 제외
            if candidate in seen_concepts:
                continue
            
            # [필터링 2] 너무 짧은 단어 제외
            if len(candidate) < 2:
                continue

            # [필터링 3] 글자 중복/유사도 체크
            if is_string_too_similar(start_concept, candidate):
                continue
            if is_string_too_similar(end_concept, candidate):
                continue
            
            # 통과
            found_step = candidate
            seen_concepts.add(candidate)
            break
        
        if found_step:
            path.append(found_step)
            print(f"[DEBUG]\t{int(t*100)}% 지점 발견: '{found_step}'")
        else:
            path.append("연관 개념")

    return path

def get_vector(word):
    """단어의 벡터를 안전하게 가져오는 헬퍼 함수"""
    try:
        idx = ALL_CONCEPTS.index(word)
        return ALL_CONCEPT_VECTORS[idx]
    except ValueError:
        return embedding_model.encode([word])[0]

def fill_gap_between(w1, w2, exclude_list=None):
    if exclude_list is None: exclude_list = set()
    
    v1 = get_vector(w1)
    v2 = get_vector(w2)
    mid_vector = (v1 + v2) / 2
    
    sims = cosine_similarity(mid_vector.reshape(1, -1), ALL_CONCEPT_VECTORS)[0]
    top_indices = np.argsort(sims)[::-1][:30]
    
    for idx in top_indices:
        candidate = ALL_CONCEPTS[idx]
        
        # 제외 목록에 있거나, 글자가 너무 비슷하면 패스
        if candidate in exclude_list: continue
        if candidate in [w1, w2]: continue
        if is_string_too_similar(w1, candidate): continue
        if is_string_too_similar(w2, candidate): continue
            
        return candidate
            
    return None

def smooth_path_recursively(full_path, threshold=0.5):
    print(f"[DEBUG]\t경로 평탄화 시작: {full_path}")
    refined_path = [full_path[0]] 
    
    for i in range(len(full_path) - 1):
        curr_word = full_path[i]
        next_word = full_path[i+1]
        
        v1 = get_vector(curr_word)
        v2 = get_vector(next_word)
        
        similarity = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]
        
        # Gap 발견 시 보수 공사
        if similarity < threshold:
            print(f"[DEBUG]\tGap 발견 ({curr_word} <-> {next_word}, sim={similarity:.2f})")
            
            # 이미 나온 단어들을 피하도록 함
            # 현재까지 확정된 경로(refined_path)와 다음 목적지(next_word)를 배제 목록으로 전달
            exclude_words = set(refined_path + [next_word])
            
            gap_filler = fill_gap_between(curr_word, next_word, exclude_words)
            
            if gap_filler:
                print(f"[DEBUG]\t보강 완료: {gap_filler}")
                refined_path.append(gap_filler)
            else:
                print(f"[DEBUG]\t보강 실패")
        
        refined_path.append(next_word)
        
    return refined_path

# --- 3. Streamlit UI 구성 ---

st.set_page_config(layout="wide", page_title="Serendipity Engine")

st.title("세렌디피티 검색 엔진")
st.markdown("""
> *"우리는 우리가 무엇을 모르는지 모릅니다."* 당신의 관심사 너머, 완전히 새로운 영감의 세계로 안내합니다.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔍 당신의 관심사 입력")
    search_history_input = st.text_area(
        "최근 관심있게 찾아본 주제나 고민들을 자유롭게 적어주세요.", 
        height=200,
        placeholder="예시:\n- 최근 생성형 AI의 저작권 논란에 대해 찾아봄\n- 맛있는 커피를 내리는 브루잉 레시피\n- 조선왕조실록 중 세종대왕의 업적"
    )
    start_btn = st.button("새로운 발견 시작하기", use_container_width=True)

with col2:
    st.subheader("💡 발견 결과")
    result_container = st.container()

if start_btn:
    if not search_history_input.strip():
        st.warning("관심사를 입력해야 분석이 가능합니다.")
    else:
        history_list = [line.strip() for line in search_history_input.split('\n') if line.strip()]
        
        with result_container:
            # 1. 관심 성운 분석
            with st.status("1. 지적 성운(Interest Nebula) 분석 중...", expanded=True) as status:
                concepts = extract_key_concepts(history_list)
                
                if not concepts:
                    status.update(label="핵심 개념 추출 실패! 문장을 더 구체적으로 적어주세요.", state="error")
                else:
                    st.write(f"**추출된 핵심 키워드:** {', '.join(concepts)}")
                    interest_vector = get_interest_nebula_vector(concepts, embedding_model)
                    
                    # 2. 동적 군집화를 통한 반대편 탐색
                    status.update(label="2. 미지의 영역(Antipode) 군집화 및 탐사 중...", state="running")
                    
                    # 5개의 낯선 테마를 도출
                    antipode_themes = find_semantic_antipode_by_clustering(interest_vector, n_clusters=5)
                    st.write(f"**발견된 낯선 테마들:** {', '.join(antipode_themes)}")
                    
                    # 3. 연결 고리 건설
                    status.update(label="3. 논리적 연결 고리(Bridge) 건설 중...", state="running")
                    
                    main_concept = concepts[0] # 가장 비중 있는 내 관심사
                    final_path = None
                    
                    progress_text = st.empty()
                    prog_bar = st.progress(0)
                    
                    for i, candidate in enumerate(antipode_themes):
                        prog_bar.progress((i + 1) / len(antipode_themes))
                        
                        # 1. 개념 승격
                        promoted_cand, is_promoted = check_and_promote_concept(candidate)
                        
                        # 2. 브릿지 탐색 (출발지 <-> 승격된 상위 개념)
                        initial_bridges = find_multi_step_bridge(main_concept, promoted_cand)
                        
                        # 3. 경로가 유효하면 채택
                        if len(initial_bridges) == 3:
                            # 2. 전체 경로 조립
                            raw_path = [main_concept] + initial_bridges + [promoted_cand]

                            # 3. Gap Filling 수행
                            # threshold=0.5: 유사도가 0.5 미만이면 중간에 하나 더 끼워넣음
                            smoothed_path_list = smooth_path_recursively(raw_path, threshold=0.5)

                            # 4. 결과 저장
                            final_path = {
                                "full_chain": smoothed_path_list, # 전체 경로 리스트
                                "start": main_concept,
                                "end": candidate,
                                "context": promoted_cand
                            }
                            break
                    
                    prog_bar.empty()
                    status.update(label="탐사 완료!", state="complete", expanded=False)

            # --- 최종 결과 카드 표시 (UI 개선) ---
            if final_path:
                st.balloons()
                st.success(f"###새로운 탐험지 발견: [{final_path['end']}]")
                st.markdown("---")

                # 전체 경로 시각화
                display_chain = final_path['full_chain'] + [f"**{final_path['end']}**"]

                path_str = " ➔ ".join(display_chain)

                st.markdown("### 🔗 연결 경로")
                st.info(path_str)

                st.markdown("---")
                mid_concepts = final_path['full_chain'][1:-1] # 중간 단계들만 추출
                mid_concepts_str = ", ".join([f"**[{c}]**" for c in mid_concepts])
