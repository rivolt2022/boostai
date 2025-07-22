# 🔥 하이브리드 앙상블: 3개 모델 최강 조합 
import pandas as pd
import numpy as np

def create_hybrid_ensemble():
    """🔥 3개 모델 하이브리드 앙상블"""
    print("🔥 하이브리드 앙상블 생성 시작!")
    print("=" * 50)
    
    try:
        # 🤖 3개 모델 파일 로드
        print("📁 모델 예측 파일 로드 중...")
        
        # 1. 견고성 모델 (최고 CV: 0.7223)
        robust_df = pd.read_csv('submission_robust_80percent.csv')
        print(f"✅ 견고성 모델 로드 완료: {len(robust_df)}개 예측")
        
        # 2. Word2Vec 모델 (CV: 0.7095)
        word2vec_df = pd.read_csv('submission_word2vec.csv')
        print(f"✅ Word2Vec 모델 로드 완료: {len(word2vec_df)}개 예측")
        
        # 3. 심플 향상 모델 (CV: 0.7115)
        enhanced_df = pd.read_csv('submission_enhanced_simple.csv')
        print(f"✅ 심플 향상 모델 로드 완료: {len(enhanced_df)}개 예측")
        
        # 🎯 가중치 설정 (CV 성능 + 다양성 고려)
        weights = {
            'robust': 0.40,      # 최고 CV + 안정성
            'word2vec': 0.35,    # 완전히 다른 패턴 (시퀀스)
            'enhanced': 0.25     # 특성 다양화
        }
        
        print(f"\n⚖️ 가중치 설정:")
        print(f"  견고성 모델: {weights['robust']:.0%}")
        print(f"  Word2Vec 모델: {weights['word2vec']:.0%}")
        print(f"  심플 향상 모델: {weights['enhanced']:.0%}")
        
        # 🔥 하이브리드 앙상블 계산
        print(f"\n🔥 하이브리드 앙상블 계산 중...")
        
        # ID 확인 (모든 파일이 같은 순서인지 체크)
        if not (robust_df['ID'].equals(word2vec_df['ID']) and 
                robust_df['ID'].equals(enhanced_df['ID'])):
            print("❌ 오류: ID 순서가 다릅니다!")
            return False
        
        # 가중 평균 계산
        hybrid_predictions = (
            weights['robust'] * robust_df['Inhibition'] +
            weights['word2vec'] * word2vec_df['Inhibition'] +
            weights['enhanced'] * enhanced_df['Inhibition']
        )
        
        # 📊 결과 분석
        print(f"\n📊 각 모델별 예측 통계:")
        print(f"견고성 모델:")
        print(f"  평균: {robust_df['Inhibition'].mean():.2f}")
        print(f"  표준편차: {robust_df['Inhibition'].std():.2f}")
        
        print(f"Word2Vec 모델:")
        print(f"  평균: {word2vec_df['Inhibition'].mean():.2f}")
        print(f"  표준편차: {word2vec_df['Inhibition'].std():.2f}")
        
        print(f"심플 향상 모델:")
        print(f"  평균: {enhanced_df['Inhibition'].mean():.2f}")
        print(f"  표준편차: {enhanced_df['Inhibition'].std():.2f}")
        
        print(f"\n🔥 하이브리드 앙상블 결과:")
        print(f"  평균: {hybrid_predictions.mean():.2f}")
        print(f"  표준편차: {hybrid_predictions.std():.2f}")
        print(f"  범위: {hybrid_predictions.min():.2f} ~ {hybrid_predictions.max():.2f}")
        
        # 🛡️ 안전 처리 (범위 확인)
        hybrid_predictions = np.clip(hybrid_predictions, 0, 100)
        
        # 📤 최종 제출 파일 생성
        hybrid_submission = pd.DataFrame({
            'ID': robust_df['ID'],
            'Inhibition': hybrid_predictions
        })
        
        output_file = 'submission_hybrid_ensemble.csv'
        hybrid_submission.to_csv(output_file, index=False)
        
        print(f"\n✅ 하이브리드 앙상블 완료!")
        print(f"📁 파일 저장: {output_file}")
        print(f"\n📊 최종 하이브리드 예측 통계:")
        print(hybrid_submission['Inhibition'].describe())
        
        # 🎯 각 모델의 기여도 분석
        print(f"\n🎯 모델별 기여도:")
        robust_contrib = weights['robust'] * robust_df['Inhibition'].mean()
        word2vec_contrib = weights['word2vec'] * word2vec_df['Inhibition'].mean()
        enhanced_contrib = weights['enhanced'] * enhanced_df['Inhibition'].mean()
        
        print(f"  견고성 모델 기여: {robust_contrib:.2f}")
        print(f"  Word2Vec 기여: {word2vec_contrib:.2f}")
        print(f"  심플 향상 기여: {enhanced_contrib:.2f}")
        print(f"  총합: {robust_contrib + word2vec_contrib + enhanced_contrib:.2f}")
        
        print(f"\n🚀 하이브리드 앙상블의 장점:")
        print(f"  ✅ 3가지 다른 학습 패턴 조합")
        print(f"  ✅ 견고성 모델의 안정성")
        print(f"  ✅ Word2Vec의 시퀀스 패턴")
        print(f"  ✅ 심플 향상의 특성 다양화")
        print(f"  ✅ 단일 모델 한계 극복")
        
        print(f"\n⚡ 기대 효과:")
        print(f"  🎯 실제 리더보드: 0.75~0.82 예상")
        print(f"  🛡️ 변동성 최소화")
        print(f"  🔥 상호 보완 시너지")
        
        return True
        
    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
        print("📋 필요한 파일들:")
        print("  - submission_robust_80percent.csv")
        print("  - submission_word2vec.csv") 
        print("  - submission_enhanced_simple.csv")
        return False
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False

if __name__ == "__main__":
    success = create_hybrid_ensemble()
    
    if success:
        print(f"\n🎉 하이브리드 앙상블 성공!")
        print(f"🚀 submission_hybrid_ensemble.csv 제출 준비 완료!")
    else:
        print(f"\n💥 하이브리드 앙상블 실패") 