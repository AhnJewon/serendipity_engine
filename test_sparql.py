from SPARQLWrapper import SPARQLWrapper, JSON

def test_knowledge_graph():
    print(">>> [Test 3] Wikidata SPARQL 연결 테스트 시작")

    endpoint_url = "https://query.wikidata.org/sparql"
    sparql = SPARQLWrapper(endpoint_url)

    # 테스트 시나리오: '인공지능'(AI)과 '컴퓨터 과학'(CS) 사이의 관계 찾기
    concept1 = "인공지능"
    concept2 = "컴퓨터 과학"

    print(f"1. '{concept1}'과 '{concept2}' 사이의 연결 고리 검색 중...")

    query = f"""
    SELECT ?bridgeLabel WHERE {{
      SERVICE wikibase:mwapi {{
        bd:serviceParam wikibase:api "EntitySearch".
        bd:serviceParam wikibase:endpoint "www.wikidata.org".
        bd:serviceParam mwapi:search "{concept1}".
        bd:serviceParam mwapi:language "ko".
        ?item1 wikibase:apiOutputItem mwapi:item.
      }}
      SERVICE wikibase:mwapi {{
        bd:serviceParam wikibase:api "EntitySearch".
        bd:serviceParam wikibase:endpoint "www.wikidata.org".
        bd:serviceParam mwapi:search "{concept2}".
        bd:serviceParam mwapi:language "ko".
        ?item2 wikibase:apiOutputItem mwapi:item.
      }}
      
      ?item1 (wdt:P31|wdt:P279)* ?bridge .
      ?item2 (wdt:P31|wdt:P279)* ?bridge .
      
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "ko". }}
    }} LIMIT 3
    """

    try:
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        bindings = results["results"]["bindings"]
        
        if bindings:
            print("2. 결과 수신 성공:")
            for result in bindings:
                print(f"   - 연결 고리(Bridge): {result['bridgeLabel']['value']}")
            print(">>> [Test 3] 성공: Wikidata와 정상적으로 통신했습니다.\n")
        else:
            print("2. 결과 없음 (통신은 성공했으나 연결 고리를 못 찾음)")
            print(">>> [Test 3] 성공: 쿼리 실행에는 문제가 없습니다.\n")

    except Exception as e:
        print(f">>> [Test 3] 실패: SPARQL 쿼리 중 오류가 발생했습니다.\n{e}")

if __name__ == "__main__":
    test_knowledge_graph()
