import bz2
import xml.etree.ElementTree as ET
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
from kiwipiepy import Kiwi

# 설정
DUMP_FILE = 'kowiki-latest-pages-articles.xml.bz2'
OUTPUT_TITLES = 'korean_wiki_titles.json'
OUTPUT_VECTORS = 'korean_wiki_vectors.npy'
MODEL_NAME = 'jhgan/ko-sbert-sts'
BATCH_SIZE = 2048  # GPU 메모리에 따라 조절 
MIN_LENGTH = 2    # 2글자 미만 표제어 제외

# Kiwi 형태소 분석기 초기화 (속도를 위해 전역 사용)
kiwi = Kiwi()

# 제외할 네임스페이스 접두어
EXCLUDE_PREFIXES = (
    "위키백과:", "틀:", "분류:", "파일:", "미디어위키:", 
    "도움말:", "포털:", "초안:", "모듈:", "Wikipedia:", 
    "Template:", "Category:", "File:", "MediaWiki:", 
    "Help:", "Portal:", "Draft:", "Module:"
)

def is_valid_title(title):
    
    # 1. 기본 문자열 필터링
    if title.startswith(EXCLUDE_PREFIXES): return False
    if len(title) < MIN_LENGTH or len(title) > 15: return False # 너무 긴 제목 제외


    # 금지 문자 포함 여부
    if any(char in title for char in ['(', ')', '!', '?', ':', '。', '，', '、', '~', '"', "'", '[', ']', '%', '+', '=', '#', '@', ';']):
        return False
    
   # 2. 형태소 분석 기반 필터링
    try:
        # 제목을 분석
        tokens = kiwi.tokenize(title)
        
        # A. 명사(NNG:일반명사, NNP:고유명사, NR:수사, NP:대명사, SL:외국어)가 하나라도 없으면 개념이 아님
        has_noun = any(t.tag in ['NNG', 'NNP', 'NR', 'NP', 'SL'] for t in tokens)
        if not has_noun:
            return False
            
        # B. 조사가 붙어있는 경우 제외 (위키 표제어는 보통 조사가 없지만, 노이즈 방지)
        # 마지막 토큰이 조사(J..)나 어미(E..)면 제외
        if tokens[-1].tag.startswith('J') or tokens[-1].tag.startswith('E'):
            return False
            
    except Exception:
        return False # 분석 실패 시 안전하게 제외
        
    # 4. 숫자로 시작하는 것 제외
    if title[0].isdigit(): 
        return False
        
    return True

def extract_titles(dump_file):
    
    titles = []
    print(f"위키백과 덤프({dump_file}) 읽는 중...")
    
    # bz2 파일을 스트림으로 염
    with bz2.open(dump_file, 'rt', encoding='utf-8') as f:
        context = ET.iterparse(f, events=('end',))
        
        for event, elem in tqdm(context, desc="XML 파싱 및 타이틀 추출"):
            if elem.tag.endswith('title'):
                title = elem.text
                if title:
                    # 특수 문서 및 너무 짧은 단어 필터링
                    if is_valid_title(title):
                        titles.append(title)
                
                # 메모리 관리를 위해 요소 삭제
                elem.clear()
                
    print(f"총 {len(titles)}개의 유효한 표제어를 추출완료")
    return titles

def build_vectors(titles, model_name):
    
    print(f"임베딩 모델 로딩 중... ({model_name})")
    # device='cuda'는 GPU가 있으면 자동으로 사용 없으면 cpu로 동작
    model = SentenceTransformer(model_name)
    
    print("표제어 벡터화 진행 중...")
    # show_progress_bar=True로 진행 상황 확인
    vectors = model.encode(titles, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True)
    
    return vectors

def main():
    if not os.path.exists(DUMP_FILE):
        print(f"오류: '{DUMP_FILE}' 파일을 찾을 수 없음. 같은 폴더에 위치.")
        return

    # 1. 타이틀 추출
    titles = extract_titles(DUMP_FILE)
    
    # 2. 벡터 생성
    vectors = build_vectors(titles, MODEL_NAME)
    
    # 3. 저장
    print("데이터 저장 중...")
    
    # JSON으로 타이틀 저장 (한글 깨짐 방지 ensure_ascii=False)
    with open(OUTPUT_TITLES, 'w', encoding='utf-8') as f:
        json.dump(titles, f, ensure_ascii=False)
        
    # Numpy로 벡터 저장
    np.save(OUTPUT_VECTORS, vectors)
    
    print("모든 작업 완료")
    print(f"   - 생성된 파일 1: {OUTPUT_TITLES}")
    print(f"   - 생성된 파일 2: {OUTPUT_VECTORS}")

if __name__ == "__main__":
    main()
