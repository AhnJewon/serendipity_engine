import bz2
import xml.etree.ElementTree as ET
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

# 설정
DUMP_FILE = 'kowiki-latest-pages-articles.xml.bz2'
OUTPUT_TITLES = 'korean_wiki_titles.json'
OUTPUT_VECTORS = 'korean_wiki_vectors.npy'
MODEL_NAME = 'jhgan/ko-sbert-sts'
BATCH_SIZE = 2048  # GPU 메모리에 따라 조절 
MIN_LENGTH = 2    # 2글자 미만 표제어 제외

# 제외할 네임스페이스 접두어 (특수 문서 제외)
EXCLUDE_PREFIXES = (
    "위키백과:", "틀:", "분류:", "파일:", "미디어위키:", 
    "도움말:", "포털:", "초안:", "모듈:", "Wikipedia:", 
    "Template:", "Category:", "File:", "MediaWiki:", 
    "Help:", "Portal:", "Draft:", "Module:"
)

def is_valid_title(title):
    
    # 서술형/명령형 제목, 외국어, 특수 기호 등을 차단 '순수 개념'만 남김
    
    # 0. 기본 필터 (길이 및 네임스페이스)
    if title.startswith(EXCLUDE_PREFIXES): return False
    # 개념은 보통 짧음 12글자 이상은 문장형 제목일 확률이 매우 높음
    if len(title) < 2 or len(title) > 12: return False 

    # 1. 서술형/명령형 어미 차단 (노래, 영화, 책 제목 등)
    # '돌아오라', '놀자', '없어', '지마', '하리', '이다' 등 문장 끝 글자 패턴
    bad_endings = [
        '다', '요', '라', '해', '까', '나', '줘', '서', '게', 
        '자', '니', '오', '네', '마', '어', '지', '든', '는', '고', 
        '함', '음', '기', '리' # 명사형 종결이나 '~하리' 같은 고어체
    ]
    if title[-1] in bad_endings:
        return False

    # 2. 특수문자 및 외국어 알파벳 단독 사용 차단
    # %, +, = 등이 포함되면 기술 코드거나 노이즈일 확률 높음
    # 한글 제목인데 중간에 영어가 너무 많이 섞인 경우도 제외 (예: ABC Radio)
    if any(char in title for char in ['(', ')', '!', '?', ':', '。', '，', '、', '~', '"', "'", '[', ']', '%', '+', '=', '#', '@', ';']):
        return False
    
    # 3. 금지어 목록 (구체적 고유명사, 회사, 브랜드 등)
    boring_keywords = [
        # 행정/법률/정치/통계
        "법률", "규정", "조례", "규칙", "목록", "연표", "협약", "선언", "결의", "조약", "헌장",
        "선거구", "행정 구역", "분기점", "나들목", "도로", "국도", "지방도", "노선", "고속도로", "대로", "길", "대교", "터널",
        "위원회", "조직", "협회", "은행", "정부", "내각", "부", "청", "국", "공사", "그룹", "클럽", "연맹", "당", "건설",
        "인구", "관계", "경제", "지리", "역사", "문화", "사회", # "가나의 인구" 등 통계 문서
        
        # 지리/장소
        "산", "섬", "강", "평원", "호", "동", "리", "구", "군", "시", "도", "주",
        "공원", "병원", "학교", "대학", "공항", "역", "경기장", "도서관", "빌딩", "타워",
        
        # 스포츠/대회
        "월드컵", "올림픽", "선수권", "리그", "대회", "축구", "농구", "야구", "배구", "단", "팀", "선수단",
        "제1", "제2", "제3", "제4", "제5", "제6", "제7", "제8", "제9", "세기", "년대",
        
        # 엔터테인먼트/미디어 (가장 많은 노이즈!)
        "음반", "노래", "싱글", "드라마", "영화", "만화", "애니메이션", "소설", "시리즈", "방송", "프로그램", "뉴스", "비디오", "극장", "오페라", "곡",
        "게임", "캐릭터", "에피소드", "라디오", "채널", "시즌", "화", "Radio", "TV", "FM",
        
        # 인물/직위
        "선수", "감독", "배우", "가수", "의원", "장관", "대통령", "총리", "회장", "사장", "교수", "박사", "씨", "왕", "황제", "부인",
        
        # 기술/제품/상업
        "국기", "문장", "국가", 
        "지하철", "버스", "차", "트럭", "모델", "디스플레이", "소프트웨어", "엔진", "시스템", "폰", "패드", "항공", "노선", "편", "파트", "Part",
        "주식회사", "Corp", "Inc", "Limited",
        
        # 생물/기타
        "개미", "지빠귀", "나비", "꽃", "나무", "가족", "아이", "소녀", "소년"
    ]
    
    if any(bk in title for bk in boring_keywords):
        return False
        
    # 4. 숫자로 시작하는 것 제외
    if title[0].isdigit(): 
        return False
        
    return True

def extract_titles(dump_file):
    """.xml.bz2 파일에서 유효한 표제어만 추출합니다."""
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
                
    print(f"총 {len(titles)}개의 유효한 표제어를 추출했습니다.")
    return titles

def build_vectors(titles, model_name):
    """추출된 표제어들을 벡터화합니다."""
    print(f"임베딩 모델 로딩 중... ({model_name})")
    # device='cuda'는 GPU가 있으면 자동으로 사용 없으면 cpu로 동작
    model = SentenceTransformer(model_name)
    
    print("표제어 벡터화 진행 중 (시간이 좀 걸립니다)...")
    # show_progress_bar=True로 진행 상황 확인
    vectors = model.encode(titles, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True)
    
    return vectors

def main():
    if not os.path.exists(DUMP_FILE):
        print(f"오류: '{DUMP_FILE}' 파일을 찾을 수 없습니다. 같은 폴더에 위치시켜주세요.")
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
