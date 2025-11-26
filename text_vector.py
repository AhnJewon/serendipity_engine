import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def test_vector_logic():
    print(">>> [Test 2] 벡터 연산 및 의미적 반대편 논리 테스트 시작")

    try:
        # 1. 가상의 데이터 생성 (3개의 개념 벡터, 5차원)
        # 가정: A는 B와 비슷하고, C는 아주 다름
        vec_a = np.array([1, 1, 1, 0, 0])
        vec_b = np.array([1, 1, 0, 0, 0]) 
        vec_c = np.array([0, 0, 0, 1, 1]) # A와 정반대 성향
        
        concept_vectors = np.array([vec_a, vec_b])
        all_vectors = np.array([vec_a, vec_b, vec_c])
        all_concepts = ["개념A", "개념B", "개념C(반대편)"]

        # 2. 관심 성운 벡터 계산 (가중 평균 테스트)
        print("1. 가중 평균 벡터 계산 중...")
        weights = [0.5, 1.5] # 최근 검색어(B)에 가중치
        interest_vector = np.average(concept_vectors, axis=0, weights=weights)
        print(f"   - 계산된 관심 벡터: {interest_vector}")

        # 3. 의미적 반대편 찾기 (코사인 유사도)
        print("2. 의미적 반대편 탐색 중...")
        
        # 유사도 계산 (reshape(1,-1)은 1차원 배열을 2차원 행렬로 변환)
        similarities = cosine_similarity(interest_vector.reshape(1, -1), all_vectors)
        
        # 가장 낮은 유사도 인덱스 찾기 (argsort의 첫 번째 요소)
        antipode_index = np.argsort(similarities[0])[0]
        antipode_concept = all_concepts[antipode_index]
        
        print(f"   - 유사도 점수: {similarities[0]}")
        print(f"   - 가장 낮은 유사도 인덱스: {antipode_index}")
        print(f"   - 결과 개념: {antipode_concept}")

        if antipode_concept == "개념C(반대편)":
            print(">>> [Test 2] 성공: 논리적으로 가장 먼 개념을 찾았습니다.\n")
        else:
            print(">>> [Test 2] 실패: 예상치 못한 개념이 선택되었습니다.\n")

    except Exception as e:
        print(f">>> [Test 2] 실패: 오류가 발생했습니다.\n{e}")

if __name__ == "__main__":
    test_vector_logic()
