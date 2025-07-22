# 🤖 Word2Vec 기반 CYP3A4 예측 모델 (0.85+ 도전)
import pandas as pd
import numpy as np
import os
import random
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
import optuna
import warnings

warnings.filterwarnings('ignore')

# 🤖 Word2Vec 기반 설정 (0.85+ 목표)
CFG = {
    'NBITS': 2048,      # Morgan 지문 비트 수 (1024 사용)
    'SEEDS': [42, 123, 456, 789, 999],  # 🛡️ 다중 시드로 안정성 확보
    'N_SPLITS': 15,     # K-폴드 증가 (안정성)
    'N_TRIALS': 50,     # 빠른 실행을 위해 축소
    'SIMULATE_80_PERCENT': True,  # 🎲 무작위 80% 시뮬레이션
    'N_SIMULATIONS': 20,  # 80% 샘플링 시뮬레이션 횟수
    'TARGET_TRANSFORM': True,  # 🎯 타겟 변환 활성화
    'WORD2VEC_DIM': 300,  # 🤖 Word2Vec 벡터 차원
}

def seed_everything(seed):
    """모든 랜덤 시드를 설정하여 재현성 보장"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def simulate_random_80_percent(y_true, y_pred, n_simulations=20):
    """🎲 무작위 80% 샘플링 시뮬레이션 (리더보드 평가 방식 모방)"""
    scores = []
    n_samples = len(y_true)
    
    for _ in range(n_simulations):
        # 무작위 80% 샘플링
        indices = np.random.choice(n_samples, size=int(0.8 * n_samples), replace=False)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
        
        # 스코어 계산
        rmse = np.sqrt(mean_squared_error(y_true_sample, y_pred_sample))
        y_range = y_true_sample.max() - y_true_sample.min()
        nrmse = rmse / y_range if y_range > 0 else 0
        
        correlation = np.corrcoef(y_true_sample, y_pred_sample)[0, 1]
        if np.isnan(correlation):
            correlation = 0
            
        score = 0.5 * (1 - nrmse) + 0.5 * correlation
        scores.append(score)
    
    return np.mean(scores), np.std(scores)

# 첫 번째 시드로 초기 설정
seed_everything(CFG['SEEDS'][0])

class Word2VecCYP3A4Predictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.best_params = None
        # 모든 RDKit 설명자 이름
        self.descriptor_names = [desc_name for desc_name, _ in Descriptors._descList]
        # 🎯 타겟 변환 추가
        self.target_transformer = None
        self.use_target_transform = True
        # 🤖 Word2Vec 모델 로드
        self.word2vec_model = None
        self.load_word2vec_model()
    
    def load_word2vec_model(self):
        """🤖 Word2Vec 모델 로드"""
        try:
            with open('model_300dim.pkl', 'rb') as f:
                self.word2vec_model = pickle.load(f)
            print("✅ Word2Vec 모델 로드 성공 (300차원)")
        except Exception as e:
            print(f"❌ Word2Vec 모델 로드 실패: {e}")
            print("Word2Vec 없이 Morgan 기반으로 진행합니다.")
            self.word2vec_model = None
    
    def tokenize_smiles(self, smiles):
        """🧬 SMILES를 토큰으로 분리"""
        tokens = []
        i = 0
        while i < len(smiles):
            # 두 글자 토큰들 먼저 확인
            if i < len(smiles) - 1:
                two_char = smiles[i:i+2]
                if two_char in ['Cl', 'Br', 'Si', 'Na', 'Mg', 'Al', 'Ca', 'Fe', 'Cu', 'Zn']:
                    tokens.append(two_char)
                    i += 2
                    continue
            
            # 한 글자 토큰
            tokens.append(smiles[i])
            i += 1
        
        return tokens
    
    def smiles_to_word2vec(self, smiles):
        """🤖 SMILES를 Word2Vec 벡터로 변환"""
        if self.word2vec_model is None:
            return np.zeros(300)  # 기본 300차원 영벡터
        
        tokens = self.tokenize_smiles(smiles)
        vectors = []
        
        for token in tokens:
            try:
                # Word2Vec 벡터 추출
                vector = self.word2vec_model.wv[token]
                vectors.append(vector)
            except (KeyError, AttributeError):
                # 토큰이 없으면 무시
                continue
        
        if vectors:
            # 평균 벡터 계산 (분자 전체 표현)
            return np.mean(vectors, axis=0)
        else:
            # 벡터가 없으면 영벡터
            return np.zeros(300)

    def smiles_to_features(self, smiles):
        """🤖 Word2Vec + Morgan 조합 특성 추출"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # 🤖 1. Word2Vec 벡터 (300차원) - 핵심 혁신!
        word2vec_features = self.smiles_to_word2vec(smiles)
        
        # 🧬 2. Morgan Fingerprint (단일 radius=2, 1024 bits)
        fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        arr_morgan = np.zeros((1024,))
        DataStructs.ConvertToNumpyArray(fp_morgan, arr_morgan)
        
        # 3. MACCS Keys (167 bits)
        fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
        arr_maccs = np.zeros((167,))
        DataStructs.ConvertToNumpyArray(fp_maccs, arr_maccs)
        
        # 🎯 4. 핵심 RDKit Descriptors (12개)
        core_descriptors = []
        important_descs = [
            'MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors', 'TPSA',
            'NumRotatableBonds', 'NumAromaticRings', 'NumAliphaticRings',
            'NumHeteroatoms', 'FractionCsp3', 'LabuteASA', 'BertzCT'
        ]
        
        for desc_name in important_descs:
            try:
                desc_func = getattr(Descriptors, desc_name)
                desc_value = desc_func(mol)
                core_descriptors.append(desc_value if desc_value is not None else 0)
            except:
                core_descriptors.append(0)
        
        # 🤖 모든 특성 결합: Word2Vec + Morgan + MACCS + Descriptors
        features = np.concatenate([word2vec_features, arr_morgan, arr_maccs, core_descriptors])
        return features

    def prepare_data(self, df, is_training=True):
        """📊 데이터 전처리 + 타겟 변환"""
        smiles_col = 'Canonical_Smiles' if 'Canonical_Smiles' in df.columns else 'SMILES'
        
        print(f"특성 추출 중... ({len(df)}개 분자)")
        features_list = []
        valid_indices = []
        
        for idx, smiles in enumerate(df[smiles_col]):
            features = self.smiles_to_features(smiles)
            if features is not None:
                features_list.append(features)
                valid_indices.append(idx)
            
            if (idx + 1) % 100 == 0:
                print(f"  진행: {idx + 1}/{len(df)}")
        
        if not features_list:
            raise ValueError("유효한 분자가 없습니다!")
        
        X = np.array(features_list)
        print(f"✅ 특성 추출 완료: {X.shape[1]:,}개 특성")
        
        # NaN/Inf 처리
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        # 스케일링
        if is_training:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        return X, valid_indices
    
    def transform_target(self, y, fit=False):
        """🎯 타겟 변환 (sqrt로 분포 개선)"""
        if not self.use_target_transform:
            return y
            
        if fit:
            # sqrt 변환 (0에 가까운 값들 처리)
            y_transformed = np.sqrt(y + 1)  # +1로 0 처리
            self.target_mean = np.mean(y_transformed)
            self.target_std = np.std(y_transformed)
            return y_transformed
        else:
            # 기존 변환 적용
            return np.sqrt(y + 1)
    
    def inverse_transform_target(self, y_transformed):
        """🎯 타겟 역변환"""
        if not self.use_target_transform:
            return y_transformed
            
        # sqrt 역변환
        y_original = np.square(y_transformed) - 1
        return np.clip(y_original, 0, 100)  # 범위 보정

    def get_score(self, y_true, y_pred):
        """리더보드 스코어 계산: 0.5 * (1 - NRMSE) + 0.5 * Pearson_Correlation"""
        # RMSE 계산
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Normalized RMSE
        y_range = y_true.max() - y_true.min()
        nrmse = rmse / y_range if y_range > 0 else 0
        
        # Pearson 상관관계
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        if np.isnan(correlation):
            correlation = 0
        
        # 최종 스코어
        score = 0.5 * (1 - nrmse) + 0.5 * correlation
        return score

    def objective(self, trial, X, y):
        """🤖 Word2Vec + 견고성 Optuna 최적화"""
        # 🎯 타겟 변환 적용
        y_transformed = self.transform_target(y, fit=True)
        
        # 🎯 약간 더 공격적인 하이퍼파라미터 (0.85+ 목표)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1,
            'n_jobs': -1,
            'seed': CFG['SEEDS'][0],
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 800, 2000),  # 범위 확대
            'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.12),  # 범위 확대
            'num_leaves': trial.suggest_int('num_leaves', 15, 80),  # 범위 확대
            'max_depth': trial.suggest_int('max_depth', 4, 10),  # 범위 확대
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.95),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 0.95),
            'bagging_freq': trial.suggest_int('bagging_freq', 3, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 15, 45),  # 범위 조정
            'lambda_l1': trial.suggest_float('lambda_l1', 0.001, 0.5),  # 정규화 완화
            'lambda_l2': trial.suggest_float('lambda_l2', 0.001, 0.5),
        }
        
        # K-폴드 교차 검증
        kf = KFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=CFG['SEEDS'][0])
        all_y_true = []
        all_y_pred = []
        
        for train_idx, val_idx in kf.split(X, y_transformed):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_transformed[train_idx], y_transformed[val_idx]
            
            # 모델 훈련
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
            # 예측 및 역변환
            y_pred_transformed = model.predict(X_val)
            y_pred = self.inverse_transform_target(y_pred_transformed)
            y_true = y[val_idx]  # 원본 타겟 사용
            
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
        
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        
        # 🎲 무작위 80% 시뮬레이션 스코어 (리더보드 모방)
        if CFG['SIMULATE_80_PERCENT']:
            robust_score, score_std = simulate_random_80_percent(
                all_y_true, all_y_pred, CFG['N_SIMULATIONS']
            )
            # 안정성 보너스: 표준편차가 낮을수록 좋음
            stability_bonus = max(0, 0.1 - score_std)
            return robust_score + stability_bonus
        else:
            # 기본 스코어
            return self.get_score(all_y_true, all_y_pred)

    def train(self, X_train, y_train):
        """🤖 Word2Vec 기반 모델 훈련 (분자 시퀀스 학습 + Optuna)"""
        print("\n🤖 Word2Vec 기반 Optuna 최적화...")
        print("🤖 SMILES 시퀀스 패턴 학습 (300차원)")
        print("🧬 Morgan + MACCS + 핵심 Descriptors")
        print("🎯 타겟 변환 (sqrt) 적용")
        if CFG['SIMULATE_80_PERCENT']:
            print(f"🎲 무작위 80% 샘플링 시뮬레이션 활성화 ({CFG['N_SIMULATIONS']}회)")
        
        # Optuna 최적화
        study = optuna.create_study(direction='maximize', study_name='word2vec_lgbm')
        study.optimize(lambda trial: self.objective(trial, X_train, y_train), n_trials=CFG['N_TRIALS'])
        
        print(f"✅ Word2Vec 최적화 완료! 최고 스코어: {study.best_value:.4f}")
        print(f"최적 파라미터: {study.best_params}")
        
        # 최적 파라미터 저장
        self.best_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1,
            'n_jobs': -1,
            'seed': CFG['SEEDS'][0],
            'boosting_type': 'gbdt',
            'n_estimators': 1500  # 기본값
        }
            
        self.best_params.update(study.best_params)
        
        return study.best_value

    def predict(self, X_test):
        """🤖 Word2Vec + 다중 시드 견고 앙상블 예측"""
        print("🤖 Word2Vec 견고 앙상블 예측...")
        print(f"🔢 시드 수: {len(CFG['SEEDS'])}, 폴드 수: {CFG['N_SPLITS']}")
        print(f"🎯 총 모델 수: {len(CFG['SEEDS']) * CFG['N_SPLITS']}")
        
        # 훈련 데이터
        X_train = self.X_train_stored
        y_train = self.y_train_stored
        y_train_transformed = self.transform_target(y_train, fit=False)  # 타겟 변환
        
        all_predictions = []
        
        # 다중 시드 앙상블
        for seed_idx, seed in enumerate(CFG['SEEDS']):
            print(f"\n🔄 시드 {seed} ({seed_idx + 1}/{len(CFG['SEEDS'])}) 처리 중...")
            
            # 시드별 K-폴드
            kf = KFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=seed)
            seed_preds = np.zeros(len(X_test))
            
            for fold, (train_idx, _) in enumerate(kf.split(X_train, y_train_transformed)):
                if fold % 5 == 0:  # 5개 폴드마다 출력
                    print(f"  시드 {seed} - 폴드 {fold + 1}/{CFG['N_SPLITS']}")
                
                X_fold_train = X_train[train_idx]
                y_fold_train = y_train_transformed[train_idx]  # 변환된 타겟 사용
                
                # 모델 훈련 (시드별 파라미터)
                params = self.best_params.copy()
                params['seed'] = seed
                
                model = lgb.LGBMRegressor(**params)
                model.fit(X_fold_train, y_fold_train)
                
                # 테스트 예측 (변환된 타겟 공간)
                fold_pred_transformed = model.predict(X_test)
                # 역변환
                fold_pred = self.inverse_transform_target(fold_pred_transformed)
                seed_preds += fold_pred / CFG['N_SPLITS']
            
            all_predictions.append(seed_preds)
        
        # 🛡️ 견고한 앙상블: 모든 시드 예측의 평균
        final_predictions = np.mean(all_predictions, axis=0)
        
        # 📊 예측 스무딩 (극값 방지)
        final_predictions = self.smooth_predictions(final_predictions)
        
        print(f"✅ Word2Vec 견고 앙상블 완료! ({len(CFG['SEEDS']) * CFG['N_SPLITS']}개 모델)")
        return final_predictions
    
    def smooth_predictions(self, predictions):
        """📊 예측값 스무딩 (극값 방지)"""
        # 1. 기본 클리핑
        predictions = np.clip(predictions, 0, 100)
        
        # 2. 부드러운 조정 (극값 억제)
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # 3σ 이상 극값들을 부드럽게 조정
        upper_bound = mean_pred + 2.5 * std_pred
        lower_bound = mean_pred - 2.5 * std_pred
        
        predictions = np.where(predictions > upper_bound, 
                              upper_bound + 0.3 * (predictions - upper_bound),
                              predictions)
        predictions = np.where(predictions < lower_bound,
                              lower_bound + 0.3 * (predictions - lower_bound), 
                              predictions)
        
        # 최종 클리핑
        return np.clip(predictions, 0, 100)

    def store_training_data(self, X_train, y_train):
        """훈련 데이터 저장 (예측 시 사용)"""
        self.X_train_stored = X_train
        self.y_train_stored = y_train

def main():
    print("🤖 Word2Vec 기반 CYP3A4 예측 모델 (0.85+ 목표) 🤖")
    print("=" * 70)
    print("🎯 전략: Word2Vec 분자표현 + Morgan + 견고 앙상블")
    print("🤖 SMILES Word2Vec 벡터 (300차원)")
    print("🧬 Morgan Fingerprint + MACCS + 핵심 Descriptors")
    print("📊 타겟 변환 (sqrt) + 다중 시드 앙상블")
    print("🎲 무작위 80% 샘플링 대응")
    print("⚡ 분자 시퀀스 패턴 학습으로 0.85+ 도전!")
    
    try:
        # 데이터 로드
        print("\n📁 데이터 로드...")
        train_df = pd.read_csv('data/train.csv')
        test_df = pd.read_csv('data/test.csv')
        sample_submission = pd.read_csv('data/sample_submission.csv')
        
        print(f"훈련: {len(train_df)}개, 테스트: {len(test_df)}개")
        print(f"Inhibition 범위: {train_df['Inhibition'].min():.1f} ~ {train_df['Inhibition'].max():.1f}")
        
        # 모델 초기화
        predictor = Word2VecCYP3A4Predictor()
        
        # 특성 추출
        print("\n🔬 특성 추출...")
        X_train, train_valid_idx = predictor.prepare_data(train_df, is_training=True)
        X_test, test_valid_idx = predictor.prepare_data(test_df, is_training=False)
        
        # 유효한 훈련 데이터만 사용
        y_train = train_df.iloc[train_valid_idx]['Inhibition'].values
        
        print(f"✅ 최종 훈련 데이터: {X_train.shape}")
        print(f"✅ 최종 테스트 데이터: {X_test.shape}")
        
        # 훈련 데이터 저장
        predictor.store_training_data(X_train, y_train)
        
        # 모델 훈련
        print("\n🎯 모델 훈련...")
        best_score = predictor.train(X_train, y_train)
        
        # 예측
        print("\n🚀 최종 예측...")
        predictions = predictor.predict(X_test)
        
        # 예측 후처리 (범위 제한)
        predictions = np.clip(predictions, 0, 100)
        
        # 제출 파일 생성
        submission = sample_submission.copy()
        submission.iloc[test_valid_idx, submission.columns.get_loc('Inhibition')] = predictions
        
        # 예측하지 못한 부분은 평균값으로 채움
        mean_inhibition = train_df['Inhibition'].mean()
        submission['Inhibition'].fillna(mean_inhibition, inplace=True)
        
        # 저장
        output_file = 'submission_word2vec.csv'
        submission.to_csv(output_file, index=False)
        
        print(f"\n✅ Word2Vec 기반 제출 파일 저장: {output_file}")
        print(f"🤖 최고 Word2Vec 스코어: {best_score:.4f}")
        print(f"🔢 총 앙상블 모델 수: {len(CFG['SEEDS']) * CFG['N_SPLITS']}개")
        print(f"\n📊 Word2Vec 예측 결과 요약:")
        print(submission['Inhibition'].describe())
        
        print(f"\n🤖 Word2Vec 기반 모델 완료!")
        print(f"🤖 SMILES 시퀀스 패턴 학습 (300차원)")
        print(f"🧬 Morgan + MACCS + 핵심 Descriptors 보완")
        print(f"📊 타겟 변환으로 분포 최적화")
        print(f"🛡️ 견고 앙상블로 안정성 유지")
        print(f"⚡ 기대 효과: 분자 표현 혁신으로 0.85+ 달성!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 