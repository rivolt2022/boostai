import pickle
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec

# 모델 파일 로드
with open('model_300dim.pkl', 'rb') as f:
    model_data = pickle.load(f)

print("모델 타입:", type(model_data))
print("모델 클래스:", model_data.__class__.__name__)

# Word2Vec 모델 정보
if isinstance(model_data, Word2Vec):
    print("\nWord2Vec 모델 정보:")
    
    # wv 속성 확인
    wv = model_data.wv
    print(f"wv 타입: {type(wv)}")
    
    # 사용 가능한 속성들 확인
    print("\n사용 가능한 속성들:")
    for attr in dir(wv):
        if not attr.startswith('_'):
            print(f"- {attr}")
    
    # 벡터 크기 확인
    if hasattr(wv, 'vector_size'):
        print(f"\n벡터 크기: {wv.vector_size}")
    
    # 어휘 크기 확인 (다양한 방법 시도)
    vocab_size = None
    if hasattr(wv, 'key_to_index'):
        vocab_size = len(wv.key_to_index)
    elif hasattr(wv, 'vocab'):
        vocab_size = len(wv.vocab)
    elif hasattr(wv, 'index2word'):
        vocab_size = len(wv.index2word)
    
    if vocab_size:
        print(f"어휘 크기: {vocab_size}")
    
    # 처음 몇 개의 키 확인
    print("\n처음 10개 키:")
    keys = []
    if hasattr(wv, 'key_to_index'):
        keys = list(wv.key_to_index.keys())[:10]
    elif hasattr(wv, 'vocab'):
        keys = list(wv.vocab.keys())[:10]
    elif hasattr(wv, 'index2word'):
        keys = wv.index2word[:10]
    
    for key in keys:
        print(f"- {key}")
    
    # 벡터 예시
    if keys:
        try:
            example_vector = wv[keys[0]]
            print(f"\n첫 번째 키 '{keys[0]}'의 벡터 shape: {example_vector.shape}")
            print(f"벡터 값 (처음 10개): {example_vector[:10]}")
        except Exception as e:
            print(f"벡터 추출 오류: {e}") 