#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETRI 위키백과 QA API 데이터 수집 스크립트

이 스크립트는 ETRI 위키백과 QA API를 사용하여 외부 QA 데이터셋을 수집하고 저장합니다.

사용법:
    python etri_data_collection.py --num_questions 2000 --api_delay 0.2

환경 변수:
    ETRI_ACCESS_KEY: ETRI API 접근 키 (필수)

⚠️ 중요:
    - API 일일 호출 제한: 5,000건/일
    - 수집된 데이터는 `data/etri_qa_dataset.json`에 저장됩니다
"""

import os
import sys
import json
import urllib3
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from tqdm.auto import tqdm


# 프로젝트 루트 경로 설정
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# sys.path에 프로젝트 루트 추가
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ETRI API 설정
ETRI_API_URL = "http://epretx.etri.re.kr:8000/api/WikiQA/"


class ETRIWikiQA:
    """ETRI 위키백과 QA API 클라이언트"""
    
    def __init__(self, access_key: str, timeout: float = 30.0):
        """
        Args:
            access_key: ETRI API 접근 키
            timeout: API 호출 타임아웃 (초)
        """
        self.api_url = ETRI_API_URL
        self.access_key = access_key
        self.timeout = timeout
        # ETRI 공식 예제와 동일하게 설정
        self.http = urllib3.PoolManager()
    
    def query(self, question: str, engine_type: str = "hybridqa") -> Optional[Dict]:
        """
        위키백과 QA API 호출
        
        Args:
            question: 질문 텍스트
            engine_type: 엔진 타입 (irqa, kbqa, hybridqa)
        
        Returns:
            API 응답 결과 또는 None
        """
        request_json = {
            "argument": {
                "question": question,
                "type": engine_type
            }
        }
        
        try:
            # ETRI 공식 예제와 동일한 방식
            response = self.http.request(
                "POST",
                self.api_url,
                headers={
                    "Content-Type": "application/json; charset=UTF-8",
                    "Authorization": self.access_key
                },
                body=json.dumps(request_json)
            )
            
            print(f"  [응답코드] {response.status}")
            
            if response.status == 200:
                return json.loads(str(response.data, "utf-8"))
            else:
                print(f"  [응답본문] {str(response.data, 'utf-8')[:200]}")
                return None
        
        except urllib3.exceptions.MaxRetryError as e:
            print(f"  연결 실패: {e}")
            return None
        except Exception as e:
            print(f"  요청 실패: {type(e).__name__}: {e}")
            return None
    
    def extract_qa_data(self, response: Dict) -> Optional[Dict]:
        """
        API 응답에서 QA 데이터 추출
        
        Args:
            response: API 응답
            
        Returns:
            추출된 QA 데이터
        """
        if not response or response.get('result') != 0:
            return None
        
        try:
            return_object = response.get('return_object', {})
            wiki_info = return_object.get('WiKiInfo', {})
            
            answer_info = wiki_info.get('AnswerInfo', [])
            if not answer_info:
                return None
            
            best_answer = answer_info[0]
            answer = best_answer.get('answer', '')
            confidence = best_answer.get('confidence', 0)
            
            ir_info = wiki_info.get('IRInfo', [])
            context = ""
            wiki_title = ""
            if ir_info:
                context = ir_info[0].get('sent', '')
                wiki_title = ir_info[0].get('wiki_title', '')
            
            return {
                'answer': answer,
                'confidence': confidence,
                'context': context,
                'wiki_title': wiki_title
            }
            
        except Exception as e:
            print(f"데이터 추출 실패: {e}")
            return None


def collect_etri_qa_data(
    questions: List[str],
    etri_client: ETRIWikiQA,
    num_questions: int = 1000,
    delay: float = 0.2
) -> List[Dict]:
    """
    ETRI API를 사용하여 QA 데이터 수집
    모든 질문-정답 페어를 저장합니다 (필터링 없음)
    
    Args:
        questions: 질문 리스트
        etri_client: ETRI API 클라이언트
        num_questions: 수집할 질문 수
        delay: API 호출 간 딜레이 (초)
    
    Returns:
        수집된 QA 데이터 리스트
    """
    collected_data = []
    failed_count = 0
    
    sample_questions = questions[:num_questions]
    total = len(sample_questions)
    
    print(f"총 {total}개의 질문 처리 시작...")
    print("-" * 60)
    
    start_time = time.time()
    
    for idx, question in enumerate(sample_questions):
        # 진행률 계산
        progress = (idx + 1) / total * 100
        elapsed = time.time() - start_time
        avg_time = elapsed / (idx + 1) if idx > 0 else 0
        remaining = avg_time * (total - idx - 1)
        
        # 진행 상황 출력 (매번)
        print(f"\n[{idx+1}/{total}] ({progress:.1f}%) 남은 시간: {remaining/60:.1f}분")
        print(f"  질문: {question[:50]}{'...' if len(question) > 50 else ''}")
        sys.stdout.flush()
        
        try:
            response = etri_client.query(question, "hybridqa")
            
            if response:
                qa_data = etri_client.extract_qa_data(response)
                
                if qa_data and qa_data.get('answer'):
                    answer = qa_data['answer']
                    context = qa_data.get('context', '')
                    confidence = qa_data.get('confidence', 0.0)
                    
                    # answer_start 찾기 (없으면 -1로 설정)
                    answer_start = context.find(answer) if context and answer else -1
                    
                    collected_data.append({
                        'id': f"etri-{idx:05d}",
                        'question': question,
                        'context': context,
                        'answers': {
                            'text': [answer] if answer else [],
                            'answer_start': [answer_start] if answer_start != -1 else []
                        },
                        'title': qa_data.get('wiki_title', ''),
                        'confidence': confidence
                    })
                    print(f"  ✅ 답변: {answer[:40]}{'...' if len(answer) > 40 else ''}")
                    print(f"     신뢰도: {confidence:.4f} | 누적 성공: {len(collected_data)}개")
                else:
                    failed_count += 1
                    print(f"  ❌ 실패: 답변 추출 불가")
            else:
                failed_count += 1
                print(f"  ❌ 실패: API 응답 없음")
            
            time.sleep(delay)
            
        except KeyboardInterrupt:
            print(f"\n\n⚠️ 사용자 중단! 현재까지 수집된 데이터: {len(collected_data)}개")
            break
        except Exception as e:
            failed_count += 1
            print(f"  ❌ 에러: {type(e).__name__}: {e}")
            continue
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✅ 수집 완료!")
    print(f"   - 성공: {len(collected_data)}개")
    print(f"   - 실패/스킵: {failed_count}개")
    print(f"   - 성공률: {len(collected_data)/(len(collected_data)+failed_count)*100:.1f}%")
    print(f"   - 총 소요 시간: {total_time/60:.1f}분")
    print("=" * 60)
    
    return collected_data


def test_api_connection(etri_client: ETRIWikiQA) -> bool:
    """API 연결 테스트 - ETRI 공식 예제 방식"""
    test_question = "대한민국의 수도는 어디인가요?"
    print(f"테스트 질문: {test_question}")
    print("-" * 50)
    
    # 직접 API 호출 테스트 (공식 예제 방식)
    request_json = {
        "argument": {
            "question": test_question,
            "type": "hybridqa"
        }
    }
    
    try:
        http = urllib3.PoolManager()
        response = http.request(
            "POST",
            ETRI_API_URL,
            headers={
                "Content-Type": "application/json; charset=UTF-8",
                "Authorization": etri_client.access_key
            },
            body=json.dumps(request_json)
        )
        
        print(f"[responseCode] {response.status}")
        print(f"[responseBody]")
        response_text = str(response.data, "utf-8")
        print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
        
        if response.status == 200:
            data = json.loads(response_text)
            qa_data = etri_client.extract_qa_data(data)
            if qa_data:
                print("\n✅ API 연결 성공!")
                print(f"정답: {qa_data['answer']}")
                print(f"신뢰도: {qa_data['confidence']:.4f}")
                return True
        
        print("❌ API 응답 실패")
        return False
        
    except Exception as e:
        print(f"❌ API 연결 실패: {type(e).__name__}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="ETRI 위키백과 QA API 데이터 수집")
    parser.add_argument("--num_questions", type=int, default=2000,
                        help="수집할 질문 수 (기본값: 2000)")
    parser.add_argument("--api_delay", type=float, default=0.2,
                        help="API 호출 간 딜레이 (초, 기본값: 0.2)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="출력 디렉토리 (기본값: notebooks/external_data_set/data)")
    parser.add_argument("--test_only", action="store_true",
                        help="API 연결 테스트만 수행")
    
    args = parser.parse_args()
    
    # API 키 확인
    ETRI_ACCESS_KEY = os.getenv("ETRI_ACCESS_KEY", "")
    if not ETRI_ACCESS_KEY:
        print("❌ ETRI_ACCESS_KEY가 설정되지 않았습니다!")
        print("   환경 변수로 설정: export ETRI_ACCESS_KEY='your-key'")
        sys.exit(1)
    
    # API 클라이언트 초기화
    etri_qa = ETRIWikiQA(ETRI_ACCESS_KEY)
    print("✅ ETRI API 클라이언트 초기화 완료")
    print(f"API URL: {ETRI_API_URL}")
    
    # API 연결 테스트
    print("\n=== API 연결 테스트 ===")
    if not test_api_connection(etri_qa):
        sys.exit(1)
    
    if args.test_only:
        print("\n테스트 완료. 데이터 수집을 건너뜁니다.")
        sys.exit(0)
    
    # 출력 디렉토리 설정
    if args.output_dir:
        data_dir = Path(args.output_dir)
    else:
        data_dir = SCRIPT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 기존 데이터셋에서 질문 로드
    print("\n=== 기존 데이터셋에서 질문 로드 ===")
    original_data_path = PROJECT_ROOT / "data" / "train_dataset"
    print(f"데이터셋 경로: {original_data_path}")
    
    if original_data_path.exists():
        print("데이터셋 로드 중...")
        sys.stdout.flush()
        
        from datasets import load_from_disk
        original_datasets = load_from_disk(str(original_data_path))
        original_questions = list(original_datasets['train']['question'])  # 리스트로 변환
        print(f"✅ 원본 데이터셋에서 {len(original_questions)}개의 질문 로드")
    else:
        print("❌ 원본 데이터셋을 찾을 수 없습니다.")
        print(f"   경로: {original_data_path}")
        sys.exit(1)
    
    # 데이터 수집 실행
    print(f"\n=== 데이터 수집 시작 ===")
    print(f"수집할 질문 수: {args.num_questions}")
    print(f"예상 소요 시간: 약 {args.num_questions * args.api_delay / 60:.1f}분")
    print(f"⚠️ 모든 질문-정답 페어를 저장합니다 (필터링 없음)\n")
    
    etri_qa_data = collect_etri_qa_data(
        questions=original_questions,
        etri_client=etri_qa,
        num_questions=args.num_questions,
        delay=args.api_delay
    )
    
    # 수집된 데이터 저장
    if etri_qa_data:
        etri_data_path = data_dir / "etri_qa_dataset.json"
        with open(etri_data_path, 'w', encoding='utf-8') as f:
            json.dump(etri_qa_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 데이터 저장 완료!")
        print(f"   - 저장 경로: {etri_data_path}")
        print(f"   - 총 샘플 수: {len(etri_qa_data)}개")
        
        # 샘플 출력
        print("\n=== 수집된 데이터 샘플 ===")
        for i, sample in enumerate(etri_qa_data[:3]):
            print(f"\n[샘플 {i+1}]")
            print(f"  ID: {sample['id']}")
            print(f"  Question: {sample['question']}")
            print(f"  Answer: {sample['answers']['text'][0] if sample['answers']['text'] else 'N/A'}")
            print(f"  Confidence: {sample['confidence']:.4f}")
    else:
        print("❌ 저장할 데이터가 없습니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()

