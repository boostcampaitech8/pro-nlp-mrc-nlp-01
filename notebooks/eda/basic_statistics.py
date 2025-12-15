"""
MRC 데이터셋 및 Corpus 기초 통계 분석 스크립트

본 스크립트는 MRC 데이터셋과 Corpus 전체에 대한 기초 통계 분석(EDA)을 수행합니다.
데이터 품질 평가나 오류 검출이 아닌, 순수하게 데이터의 구조적 특징과 분포적 특성을 파악합니다.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk

# 프로젝트 루트 경로 설정
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from notebooks.utils import setup_notebook_environment, load_dataset_safely, load_json_safely

# 한글 폰트 설정 (Linux 환경 대응)
import matplotlib.font_manager as fm
import platform
import warnings
import os

# 전역 폰트 변수
KOREAN_FONT = None

def setup_korean_font():
    """Linux 환경에 맞는 한글 폰트 설정 - 폰트 파일 경로 직접 지정"""
    global KOREAN_FONT
    system = platform.system()
    
    # 한글 폰트 경고 무시
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    if system == 'Windows':
        # Windows: 맑은 고딕 사용
        plt.rcParams['font.family'] = 'Malgun Gothic'
        KOREAN_FONT = None  # 기본 폰트 사용
    elif system == 'Darwin':
        # macOS: AppleGothic 사용
        plt.rcParams['font.family'] = 'AppleGothic'
        KOREAN_FONT = None  # 기본 폰트 사용
    else:
        # Linux: 시스템에 설치된 한글 폰트 파일 경로 직접 찾기
        font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        
        # 한글 폰트 파일 이름 패턴 (우선순위 순)
        korean_font_patterns = [
            'NanumGothic', 'NanumGothicRegular', 'NanumGothicBold', 'NanumGothicExtraBold',
            'NotoSansCJK-Regular', 'NotoSansKR-Regular', 'NotoSansCJK',
            'NanumBarunGothic', 'NanumBarunGothicRegular'
        ]
        
        font_found = False
        selected_font_path = None
        
        for pattern in korean_font_patterns:
            for font_path in font_paths:
                font_name = os.path.basename(font_path)
                if pattern.lower() in font_name.lower():
                    selected_font_path = font_path
                    font_found = True
                    break
            if font_found:
                break
        
        if font_found and selected_font_path:
            try:
                # 폰트 파일을 직접 로드하여 전역 변수에 저장
                KOREAN_FONT = fm.FontProperties(fname=selected_font_path)
                font_family = KOREAN_FONT.get_name()
                
                # matplotlib 폰트 캐시에 추가
                try:
                    fm.fontManager.addfont(selected_font_path)
                except Exception:
                    pass
                
                # 폰트 설정
                plt.rcParams['font.family'] = font_family
                plt.rcParams['font.sans-serif'] = [font_family, 'DejaVu Sans']
                
                print(f"✅ 한글 폰트 설정 완료: {font_family}")
                print(f"   폰트 경로: {selected_font_path}")
            except Exception as e:
                print(f"⚠️ 폰트 로드 실패: {e}")
                font_found = False
                KOREAN_FONT = None
        
        if not font_found:
            # 한글 폰트가 없으면 기본 폰트 사용 (경고 무시)
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            KOREAN_FONT = None
            print("⚠️ 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
            print("   한글이 깨질 수 있지만 그래프는 정상적으로 생성됩니다.")
            print("   한글 폰트 설치 방법:")
            print("   sudo apt-get install fonts-nanum fonts-noto-cjk")
            print("   설치 후 matplotlib 폰트 캐시 삭제: rm -rf ~/.cache/matplotlib")
    
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
    sns.set_palette("husl")

# 폰트 설정 실행
setup_korean_font()

def get_font_prop():
    """한글 폰트 FontProperties 반환 (없으면 None)"""
    return KOREAN_FONT


class MRCStatistics:
    """MRC 데이터셋 통계 분석 클래스"""
    
    def __init__(self, train_dataset_path: Path, test_dataset_path: Path = None):
        """
        Args:
            train_dataset_path: train_dataset 디렉토리 경로
            test_dataset_path: test_dataset 디렉토리 경로 (선택)
        """
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.train_dataset = None
        self.test_dataset = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        
    def load_data(self):
        """데이터셋 로드"""
        print("=" * 60)
        print("MRC 데이터셋 로드 중...")
        print("=" * 60)
        
        # Train dataset 로드
        self.train_dataset = load_from_disk(str(self.train_dataset_path))
        self.train_df = pd.DataFrame(self.train_dataset["train"])
        self.val_df = pd.DataFrame(self.train_dataset["validation"])
        
        print(f"✅ Train 데이터: {len(self.train_df):,}개")
        print(f"✅ Validation 데이터: {len(self.val_df):,}개")
        print(f"✅ 컬럼: {self.train_df.columns.tolist()}")
        
        # Test dataset 로드 (선택)
        if self.test_dataset_path and self.test_dataset_path.exists():
            self.test_dataset = load_from_disk(str(self.test_dataset_path))
            if "validation" in self.test_dataset:
                self.test_df = pd.DataFrame(self.test_dataset["validation"])
                print(f"✅ Test 데이터: {len(self.test_df):,}개")
                print(f"   (Public: 240개, Private: 360개 예상)")
        
        print()
    
    def analyze_question(self) -> Dict:
        """Question 통계 분석"""
        print("=" * 60)
        print("2.1 Question 통계 분석")
        print("=" * 60)
        
        stats = {}
        
        for split_name, df in [("train", self.train_df), ("validation", self.val_df)]:
            # 문자 수 계산
            char_lengths = df['question'].str.len()
            # 단어 수 계산 (공백 기준)
            word_counts = df['question'].str.split().str.len()
            
            stats[split_name] = {
                'count': len(df),
                'char_length': {
                    'mean': float(char_lengths.mean()),
                    'median': float(char_lengths.median()),
                    'std': float(char_lengths.std()),
                    'min': int(char_lengths.min()),
                    'max': int(char_lengths.max()),
                    'q25': float(char_lengths.quantile(0.25)),
                    'q75': float(char_lengths.quantile(0.75)),
                },
                'word_count': {
                    'mean': float(word_counts.mean()),
                    'median': float(word_counts.median()),
                    'std': float(word_counts.std()),
                    'min': int(word_counts.min()),
                    'max': int(word_counts.max()),
                    'q25': float(word_counts.quantile(0.25)),
                    'q75': float(word_counts.quantile(0.75)),
                }
            }
            
            print(f"\n[{split_name.upper()}]")
            print(f"  샘플 수: {stats[split_name]['count']:,}개")
            print(f"  질문 길이 (문자 수):")
            print(f"    평균: {stats[split_name]['char_length']['mean']:.2f}자")
            print(f"    중앙값: {stats[split_name]['char_length']['median']:.2f}자")
            print(f"    표준편차: {stats[split_name]['char_length']['std']:.2f}자")
            print(f"    범위: {stats[split_name]['char_length']['min']} ~ {stats[split_name]['char_length']['max']}자")
            print(f"    IQR: {stats[split_name]['char_length']['q25']:.2f} ~ {stats[split_name]['char_length']['q75']:.2f}자")
            print(f"  질문 단어 수:")
            print(f"    평균: {stats[split_name]['word_count']['mean']:.2f}개")
            print(f"    중앙값: {stats[split_name]['word_count']['median']:.2f}개")
            print(f"    범위: {stats[split_name]['word_count']['min']} ~ {stats[split_name]['word_count']['max']}개")
        
        return stats
    
    def plot_question_distribution(self, output_dir: Path):
        """Question 길이 분포 시각화"""
        font_prop = get_font_prop()
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Question 길이 분포', fontsize=16, fontweight='bold', fontproperties=font_prop)
        
        for idx, (split_name, df) in enumerate([("train", self.train_df), ("validation", self.val_df)]):
            char_lengths = df['question'].str.len()
            word_counts = df['question'].str.split().str.len()
            
            # 문자 수 분포
            axes[0, idx].hist(char_lengths, bins=50, alpha=0.7, edgecolor='black')
            axes[0, idx].axvline(char_lengths.mean(), color='red', linestyle='--', label=f'평균: {char_lengths.mean():.1f}')
            axes[0, idx].axvline(char_lengths.median(), color='green', linestyle='--', label=f'중앙값: {char_lengths.median():.1f}')
            axes[0, idx].set_xlabel('문자 수', fontproperties=font_prop)
            axes[0, idx].set_ylabel('빈도', fontproperties=font_prop)
            axes[0, idx].set_title(f'{split_name.upper()} - 질문 길이 (문자 수)', fontproperties=font_prop)
            axes[0, idx].legend(prop=font_prop)
            axes[0, idx].grid(True, alpha=0.3)
            
            # 단어 수 분포
            axes[1, idx].hist(word_counts, bins=30, alpha=0.7, edgecolor='black', color='orange')
            axes[1, idx].axvline(word_counts.mean(), color='red', linestyle='--', label=f'평균: {word_counts.mean():.1f}')
            axes[1, idx].axvline(word_counts.median(), color='green', linestyle='--', label=f'중앙값: {word_counts.median():.1f}')
            axes[1, idx].set_xlabel('단어 수', fontproperties=font_prop)
            axes[1, idx].set_ylabel('빈도', fontproperties=font_prop)
            axes[1, idx].set_title(f'{split_name.upper()} - 질문 단어 수', fontproperties=font_prop)
            axes[1, idx].legend(prop=font_prop)
            axes[1, idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / 'question_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ 질문 분포 그래프 저장: {output_path}")
        plt.close()
    
    def analyze_context(self) -> Dict:
        """Context 통계 분석"""
        print("\n" + "=" * 60)
        print("2.2 Context 통계 분석")
        print("=" * 60)
        
        stats = {}
        
        for split_name, df in [("train", self.train_df), ("validation", self.val_df)]:
            # 문자 수 계산
            char_lengths = df['context'].str.len()
            # 단어 수 계산 (공백 기준)
            word_counts = df['context'].str.split().str.len()
            
            stats[split_name] = {
                'count': len(df),
                'char_length': {
                    'mean': float(char_lengths.mean()),
                    'median': float(char_lengths.median()),
                    'std': float(char_lengths.std()),
                    'min': int(char_lengths.min()),
                    'max': int(char_lengths.max()),
                    'q25': float(char_lengths.quantile(0.25)),
                    'q75': float(char_lengths.quantile(0.75)),
                },
                'word_count': {
                    'mean': float(word_counts.mean()),
                    'median': float(word_counts.median()),
                    'std': float(word_counts.std()),
                    'min': int(word_counts.min()),
                    'max': int(word_counts.max()),
                    'q25': float(word_counts.quantile(0.25)),
                    'q75': float(word_counts.quantile(0.75)),
                }
            }
            
            print(f"\n[{split_name.upper()}]")
            print(f"  샘플 수: {stats[split_name]['count']:,}개")
            print(f"  Context 길이 (문자 수):")
            print(f"    평균: {stats[split_name]['char_length']['mean']:.2f}자")
            print(f"    중앙값: {stats[split_name]['char_length']['median']:.2f}자")
            print(f"    표준편차: {stats[split_name]['char_length']['std']:.2f}자")
            print(f"    범위: {stats[split_name]['char_length']['min']:,} ~ {stats[split_name]['char_length']['max']:,}자")
            print(f"    IQR: {stats[split_name]['char_length']['q25']:.2f} ~ {stats[split_name]['char_length']['q75']:.2f}자")
            print(f"  Context 단어 수:")
            print(f"    평균: {stats[split_name]['word_count']['mean']:.2f}개")
            print(f"    중앙값: {stats[split_name]['word_count']['median']:.2f}개")
            print(f"    범위: {stats[split_name]['word_count']['min']:,} ~ {stats[split_name]['word_count']['max']:,}개")
        
        return stats
    
    def plot_context_distribution(self, output_dir: Path):
        """Context 길이 분포 시각화"""
        font_prop = get_font_prop()
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Context 길이 분포', fontsize=16, fontweight='bold', fontproperties=font_prop)
        
        for idx, (split_name, df) in enumerate([("train", self.train_df), ("validation", self.val_df)]):
            char_lengths = df['context'].str.len()
            word_counts = df['context'].str.split().str.len()
            
            # 문자 수 분포
            axes[0, idx].hist(char_lengths, bins=50, alpha=0.7, edgecolor='black')
            axes[0, idx].axvline(char_lengths.mean(), color='red', linestyle='--', label=f'평균: {char_lengths.mean():.0f}')
            axes[0, idx].axvline(char_lengths.median(), color='green', linestyle='--', label=f'중앙값: {char_lengths.median():.0f}')
            axes[0, idx].set_xlabel('문자 수', fontproperties=font_prop)
            axes[0, idx].set_ylabel('빈도', fontproperties=font_prop)
            axes[0, idx].set_title(f'{split_name.upper()} - Context 길이 (문자 수)', fontproperties=font_prop)
            axes[0, idx].legend(prop=font_prop)
            axes[0, idx].grid(True, alpha=0.3)
            
            # 단어 수 분포
            axes[1, idx].hist(word_counts, bins=50, alpha=0.7, edgecolor='black', color='orange')
            axes[1, idx].axvline(word_counts.mean(), color='red', linestyle='--', label=f'평균: {word_counts.mean():.0f}')
            axes[1, idx].axvline(word_counts.median(), color='green', linestyle='--', label=f'중앙값: {word_counts.median():.0f}')
            axes[1, idx].set_xlabel('단어 수', fontproperties=font_prop)
            axes[1, idx].set_ylabel('빈도', fontproperties=font_prop)
            axes[1, idx].set_title(f'{split_name.upper()} - Context 단어 수', fontproperties=font_prop)
            axes[1, idx].legend(prop=font_prop)
            axes[1, idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / 'context_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Context 분포 그래프 저장: {output_path}")
        plt.close()
    
    def analyze_answer(self) -> Dict:
        """Answer 통계 분석"""
        print("\n" + "=" * 60)
        print("2.3 Answer 통계 분석")
        print("=" * 60)
        
        stats = {}
        
        for split_name, df in [("train", self.train_df), ("validation", self.val_df)]:
            # Answer 텍스트 추출
            answer_texts = []
            answer_starts = []
            
            for answers in df['answers']:
                if isinstance(answers, dict):
                    if 'text' in answers and len(answers['text']) > 0:
                        answer_texts.append(answers['text'][0])
                        if 'answer_start' in answers and len(answers['answer_start']) > 0:
                            answer_starts.append(answers['answer_start'][0])
                    else:
                        answer_texts.append("")
                        answer_starts.append(-1)
                else:
                    answer_texts.append("")
                    answer_starts.append(-1)
            
            answer_df = pd.DataFrame({
                'text': answer_texts,
                'start': answer_starts
            })
            
            # 문자 수 계산
            char_lengths = answer_df['text'].str.len()
            # 단어 수 계산
            word_counts = answer_df['text'].str.split().str.len()
            
            stats[split_name] = {
                'count': len(df),
                'char_length': {
                    'mean': float(char_lengths.mean()),
                    'median': float(char_lengths.median()),
                    'std': float(char_lengths.std()),
                    'min': int(char_lengths.min()),
                    'max': int(char_lengths.max()),
                    'q25': float(char_lengths.quantile(0.25)),
                    'q75': float(char_lengths.quantile(0.75)),
                },
                'word_count': {
                    'mean': float(word_counts.mean()),
                    'median': float(word_counts.median()),
                    'std': float(word_counts.std()),
                    'min': int(word_counts.min()),
                    'max': int(word_counts.max()),
                    'q25': float(word_counts.quantile(0.25)),
                    'q75': float(word_counts.quantile(0.75)),
                },
                'answer_start': {
                    'mean': float(answer_df['start'].mean()) if (answer_df['start'] >= 0).any() else None,
                    'median': float(answer_df['start'].median()) if (answer_df['start'] >= 0).any() else None,
                }
            }
            
            print(f"\n[{split_name.upper()}]")
            print(f"  샘플 수: {stats[split_name]['count']:,}개")
            print(f"  Answer 길이 (문자 수):")
            print(f"    평균: {stats[split_name]['char_length']['mean']:.2f}자")
            print(f"    중앙값: {stats[split_name]['char_length']['median']:.2f}자")
            print(f"    표준편차: {stats[split_name]['char_length']['std']:.2f}자")
            print(f"    범위: {stats[split_name]['char_length']['min']} ~ {stats[split_name]['char_length']['max']}자")
            print(f"    IQR: {stats[split_name]['char_length']['q25']:.2f} ~ {stats[split_name]['char_length']['q75']:.2f}자")
            print(f"  Answer 단어 수:")
            print(f"    평균: {stats[split_name]['word_count']['mean']:.2f}개")
            print(f"    중앙값: {stats[split_name]['word_count']['median']:.2f}개")
            print(f"    범위: {stats[split_name]['word_count']['min']} ~ {stats[split_name]['word_count']['max']}개")
        
        return stats
    
    def plot_answer_distribution(self, output_dir: Path):
        """Answer 길이 분포 시각화"""
        font_prop = get_font_prop()
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Answer 길이 분포', fontsize=16, fontweight='bold', fontproperties=font_prop)
        
        for idx, (split_name, df) in enumerate([("train", self.train_df), ("validation", self.val_df)]):
            # Answer 텍스트 추출
            answer_texts = []
            for answers in df['answers']:
                if isinstance(answers, dict) and 'text' in answers and len(answers['text']) > 0:
                    answer_texts.append(answers['text'][0])
                else:
                    answer_texts.append("")
            
            answer_series = pd.Series(answer_texts)
            char_lengths = answer_series.str.len()
            word_counts = answer_series.str.split().str.len()
            
            # 문자 수 분포
            axes[0, idx].hist(char_lengths, bins=50, alpha=0.7, edgecolor='black', color='green')
            axes[0, idx].axvline(char_lengths.mean(), color='red', linestyle='--', label=f'평균: {char_lengths.mean():.1f}')
            axes[0, idx].axvline(char_lengths.median(), color='blue', linestyle='--', label=f'중앙값: {char_lengths.median():.1f}')
            axes[0, idx].set_xlabel('문자 수', fontproperties=font_prop)
            axes[0, idx].set_ylabel('빈도', fontproperties=font_prop)
            axes[0, idx].set_title(f'{split_name.upper()} - Answer 길이 (문자 수)', fontproperties=font_prop)
            axes[0, idx].legend(prop=font_prop)
            axes[0, idx].grid(True, alpha=0.3)
            
            # 단어 수 분포
            axes[1, idx].hist(word_counts, bins=30, alpha=0.7, edgecolor='black', color='purple')
            axes[1, idx].axvline(word_counts.mean(), color='red', linestyle='--', label=f'평균: {word_counts.mean():.1f}')
            axes[1, idx].axvline(word_counts.median(), color='blue', linestyle='--', label=f'중앙값: {word_counts.median():.1f}')
            axes[1, idx].set_xlabel('단어 수', fontproperties=font_prop)
            axes[1, idx].set_ylabel('빈도', fontproperties=font_prop)
            axes[1, idx].set_title(f'{split_name.upper()} - Answer 단어 수', fontproperties=font_prop)
            axes[1, idx].legend(prop=font_prop)
            axes[1, idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / 'answer_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Answer 분포 그래프 저장: {output_path}")
        plt.close()


class CorpusStatistics:
    """Corpus 통계 분석 클래스"""
    
    def __init__(self, corpus_json_path: Path):
        """
        Args:
            corpus_json_path: wikipedia_documents.json 파일 경로
        """
        self.corpus_json_path = corpus_json_path
        self.corpus_df = None
        
    def load_data(self):
        """Corpus 데이터 로드"""
        print("=" * 60)
        print("Corpus 데이터 로드 중...")
        print("=" * 60)
        
        corpus_data = load_json_safely(self.corpus_json_path)
        
        # 딕셔너리를 리스트로 변환 (document_id 순서대로 정렬)
        corpus_list = [v for k, v in sorted(corpus_data.items(), key=lambda x: int(x[0]))]
        self.corpus_df = pd.DataFrame(corpus_list)
        
        print(f"✅ Corpus 문서 개수: {len(self.corpus_df):,}개")
        print(f"✅ 컬럼: {self.corpus_df.columns.tolist()}")
        print()
    
    def analyze_text(self) -> Dict:
        """Text 통계 분석"""
        print("=" * 60)
        print("3.1 Text 통계 분석")
        print("=" * 60)
        
        # 문자 수 계산
        char_lengths = self.corpus_df['text'].str.len()
        # 단어 수 계산 (공백 기준)
        word_counts = self.corpus_df['text'].str.split().str.len()
        
        stats = {
            'count': len(self.corpus_df),
            'char_length': {
                'mean': float(char_lengths.mean()),
                'median': float(char_lengths.median()),
                'std': float(char_lengths.std()),
                'min': int(char_lengths.min()),
                'max': int(char_lengths.max()),
                'q25': float(char_lengths.quantile(0.25)),
                'q75': float(char_lengths.quantile(0.75)),
            },
            'word_count': {
                'mean': float(word_counts.mean()),
                'median': float(word_counts.median()),
                'std': float(word_counts.std()),
                'min': int(word_counts.min()),
                'max': int(word_counts.max()),
                'q25': float(word_counts.quantile(0.25)),
                'q75': float(word_counts.quantile(0.75)),
            }
        }
        
        print(f"\n문서 수: {stats['count']:,}개")
        print(f"\nText 길이 (문자 수):")
        print(f"  평균: {stats['char_length']['mean']:.2f}자")
        print(f"  중앙값: {stats['char_length']['median']:.2f}자")
        print(f"  표준편차: {stats['char_length']['std']:.2f}자")
        print(f"  범위: {stats['char_length']['min']:,} ~ {stats['char_length']['max']:,}자")
        print(f"  IQR: {stats['char_length']['q25']:.2f} ~ {stats['char_length']['q75']:.2f}자")
        print(f"\nText 단어 수:")
        print(f"  평균: {stats['word_count']['mean']:.2f}개")
        print(f"  중앙값: {stats['word_count']['median']:.2f}개")
        print(f"  표준편차: {stats['word_count']['std']:.2f}개")
        print(f"  범위: {stats['word_count']['min']:,} ~ {stats['word_count']['max']:,}개")
        print(f"  IQR: {stats['word_count']['q25']:.2f} ~ {stats['word_count']['q75']:.2f}개")
        
        return stats
    
    def plot_text_distribution(self, output_dir: Path):
        """Text 길이 분포 시각화"""
        font_prop = get_font_prop()
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Corpus Text 길이 분포', fontsize=16, fontweight='bold', fontproperties=font_prop)
        
        char_lengths = self.corpus_df['text'].str.len()
        word_counts = self.corpus_df['text'].str.split().str.len()
        
        # 문자 수 분포
        axes[0].hist(char_lengths, bins=100, alpha=0.7, edgecolor='black')
        axes[0].axvline(char_lengths.mean(), color='red', linestyle='--', label=f'평균: {char_lengths.mean():.0f}')
        axes[0].axvline(char_lengths.median(), color='green', linestyle='--', label=f'중앙값: {char_lengths.median():.0f}')
        axes[0].set_xlabel('문자 수', fontproperties=font_prop)
        axes[0].set_ylabel('빈도', fontproperties=font_prop)
        axes[0].set_title('Text 길이 (문자 수)', fontproperties=font_prop)
        axes[0].legend(prop=font_prop)
        axes[0].grid(True, alpha=0.3)
        
        # 단어 수 분포
        axes[1].hist(word_counts, bins=100, alpha=0.7, edgecolor='black', color='orange')
        axes[1].axvline(word_counts.mean(), color='red', linestyle='--', label=f'평균: {word_counts.mean():.0f}')
        axes[1].axvline(word_counts.median(), color='green', linestyle='--', label=f'중앙값: {word_counts.median():.0f}')
        axes[1].set_xlabel('단어 수', fontproperties=font_prop)
        axes[1].set_ylabel('빈도', fontproperties=font_prop)
        axes[1].set_title('Text 단어 수', fontproperties=font_prop)
        axes[1].legend(prop=font_prop)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / 'corpus_text_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Corpus Text 분포 그래프 저장: {output_path}")
        plt.close()
    
    def analyze_title(self) -> Dict:
        """Title 통계 분석"""
        print("\n" + "=" * 60)
        print("3.2 Title 통계 분석")
        print("=" * 60)
        
        # 문자 수 계산
        char_lengths = self.corpus_df['title'].str.len()
        # 단어 수 계산 (공백 기준)
        word_counts = self.corpus_df['title'].str.split().str.len()
        
        stats = {
            'count': len(self.corpus_df),
            'char_length': {
                'mean': float(char_lengths.mean()),
                'median': float(char_lengths.median()),
                'std': float(char_lengths.std()),
                'min': int(char_lengths.min()),
                'max': int(char_lengths.max()),
                'q25': float(char_lengths.quantile(0.25)),
                'q75': float(char_lengths.quantile(0.75)),
            },
            'word_count': {
                'mean': float(word_counts.mean()),
                'median': float(word_counts.median()),
                'std': float(word_counts.std()),
                'min': int(word_counts.min()),
                'max': int(word_counts.max()),
                'q25': float(word_counts.quantile(0.25)),
                'q75': float(word_counts.quantile(0.75)),
            }
        }
        
        print(f"\n문서 수: {stats['count']:,}개")
        print(f"\nTitle 길이 (문자 수):")
        print(f"  평균: {stats['char_length']['mean']:.2f}자")
        print(f"  중앙값: {stats['char_length']['median']:.2f}자")
        print(f"  표준편차: {stats['char_length']['std']:.2f}자")
        print(f"  범위: {stats['char_length']['min']} ~ {stats['char_length']['max']}자")
        print(f"  IQR: {stats['char_length']['q25']:.2f} ~ {stats['char_length']['q75']:.2f}자")
        print(f"\nTitle 단어 수:")
        print(f"  평균: {stats['word_count']['mean']:.2f}개")
        print(f"  중앙값: {stats['word_count']['median']:.2f}개")
        print(f"  표준편차: {stats['word_count']['std']:.2f}개")
        print(f"  범위: {stats['word_count']['min']} ~ {stats['word_count']['max']}개")
        print(f"  IQR: {stats['word_count']['q25']:.2f} ~ {stats['word_count']['q75']:.2f}개")
        
        return stats
    
    def plot_title_distribution(self, output_dir: Path):
        """Title 길이 분포 시각화"""
        font_prop = get_font_prop()
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Corpus Title 길이 분포', fontsize=16, fontweight='bold', fontproperties=font_prop)
        
        char_lengths = self.corpus_df['title'].str.len()
        word_counts = self.corpus_df['title'].str.split().str.len()
        
        # 문자 수 분포
        axes[0].hist(char_lengths, bins=50, alpha=0.7, edgecolor='black', color='green')
        axes[0].axvline(char_lengths.mean(), color='red', linestyle='--', label=f'평균: {char_lengths.mean():.1f}')
        axes[0].axvline(char_lengths.median(), color='blue', linestyle='--', label=f'중앙값: {char_lengths.median():.1f}')
        axes[0].set_xlabel('문자 수', fontproperties=font_prop)
        axes[0].set_ylabel('빈도', fontproperties=font_prop)
        axes[0].set_title('Title 길이 (문자 수)', fontproperties=font_prop)
        axes[0].legend(prop=font_prop)
        axes[0].grid(True, alpha=0.3)
        
        # 단어 수 분포
        axes[1].hist(word_counts, bins=30, alpha=0.7, edgecolor='black', color='purple')
        axes[1].axvline(word_counts.mean(), color='red', linestyle='--', label=f'평균: {word_counts.mean():.1f}')
        axes[1].axvline(word_counts.median(), color='blue', linestyle='--', label=f'중앙값: {word_counts.median():.1f}')
        axes[1].set_xlabel('단어 수', fontproperties=font_prop)
        axes[1].set_ylabel('빈도', fontproperties=font_prop)
        axes[1].set_title('Title 단어 수', fontproperties=font_prop)
        axes[1].legend(prop=font_prop)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / 'corpus_title_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Corpus Title 분포 그래프 저장: {output_path}")
        plt.close()


def main():
    """메인 실행 함수"""
    # 환경 설정
    paths = setup_notebook_environment()
    data_dir = paths['data_dir']
    output_dir = Path(__file__).parent / 'statistics_output'
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("MRC 데이터셋 및 Corpus 기초 통계 분석")
    print("=" * 60)
    print(f"데이터 디렉토리: {data_dir}")
    print(f"출력 디렉토리: {output_dir}")
    print()
    
    # 경로 설정
    train_dataset_path = data_dir / "train_dataset"
    test_dataset_path = data_dir / "test_dataset"
    corpus_json_path = data_dir / "wikipedia_documents.json"
    
    # 결과 저장용 딕셔너리
    results = {
        'mrc_statistics': {},
        'corpus_statistics': {}
    }
    
    # 1. MRC 데이터셋 통계 분석
    print("\n" + "=" * 60)
    print("# 2. MRC 데이터셋 통계")
    print("=" * 60)
    
    mrc_stats = MRCStatistics(train_dataset_path, test_dataset_path)
    mrc_stats.load_data()
    
    # Question 통계
    results['mrc_statistics']['question'] = mrc_stats.analyze_question()
    mrc_stats.plot_question_distribution(output_dir)
    
    # Context 통계
    results['mrc_statistics']['context'] = mrc_stats.analyze_context()
    mrc_stats.plot_context_distribution(output_dir)
    
    # Answer 통계
    results['mrc_statistics']['answer'] = mrc_stats.analyze_answer()
    mrc_stats.plot_answer_distribution(output_dir)
    
    # 2. Corpus 통계 분석
    print("\n" + "=" * 60)
    print("# 3. Corpus 통계")
    print("=" * 60)
    
    corpus_stats = CorpusStatistics(corpus_json_path)
    corpus_stats.load_data()
    
    # Text 통계
    results['corpus_statistics']['text'] = corpus_stats.analyze_text()
    corpus_stats.plot_text_distribution(output_dir)
    
    # Title 통계
    results['corpus_statistics']['title'] = corpus_stats.analyze_title()
    corpus_stats.plot_title_distribution(output_dir)
    
    # 결과를 JSON으로 저장
    output_json_path = output_dir / 'statistics_results.json'
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("분석 완료!")
    print("=" * 60)
    print(f"✅ 통계 결과 JSON 저장: {output_json_path}")
    print(f"✅ 시각화 그래프 저장 위치: {output_dir}")
    print()


if __name__ == "__main__":
    main()

