import sys
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

def test_model_loading():
    print(">>> [Test 1] 모델 로딩 및 키워드 추출 테스트 시작")
    
    try:
        # 1. 임베딩 모델 로드
        print("1. 임베딩 모델 로딩 중... (jhgan/ko-sbert-sts)")
        embedding_model = SentenceTransformer('jhgan/ko-sbert-sts')
        print("   - 임베딩 모델 로드 성공")

        # 2. KeyBERT 모델 로드
        print("2. KeyBERT 모델 로딩 중...")
        kw_model = KeyBERT(embedding_model)
        print("   - KeyBERT 모델 로드 성공")

        # 3. 키워드 추출 및 벡터화 테스트
        test_text = "인공지능과 딥러닝은 현대 기술의 핵심입니다."
        print(f"3. 테스트 문장: '{test_text}'")
        
        # 키워드 추출
        keywords = kw_model.extract_keywords(test_text, keyphrase_ngram_range=(1, 1), top_n=3)
        print(f"   - 추출된 키워드: {keywords}")

        # 벡터화
        vector = embedding_model.encode(test_text)
        print(f"   - 문장 벡터 차원: {vector.shape}")

        print(">>> [Test 1] 성공: 모델이 정상적으로 작동합니다.\n")

    except Exception as e:
        print(f">>> [Test 1] 실패: 오류가 발생했습니다.\n{e}")

if __name__ == "__main__":
    test_model_loading()
