import pandas as pd
import numpy as np
import os
import random
import pickle
import warnings
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdMolDescriptors
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, ShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge, HuberRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

# 🛡️ 무작위 80% 샘플링 견고성에 특화된 설정
CFG = {
    'NBITS': 2048,
    'SEEDS': [42, 123, 456, 789, 999, 2023, 2024, 777, 888],  # 🔥 9개 시드로 극대 다양성
    'N_SPLITS': 20,        # 더 많은 분할로 모든 케이스 커버
    'N_REPEATS': 3,        # 반복 교차검증
    'BOOTSTRAP_SAMPLES': 50,  # 🔥 Bootstrap 샘플링으로 견고성 극대화
    'CONSENSUS_THRESHOLD': 0.7,  # 합의 기반 예측
    'UNCERTAINTY_WEIGHT': 0.3,   # 불확실성 가중치
    'DIVERSITY_PENALTY': 0.1,    # 다양성 페널티
    'ROBUST_LOSS': True,         # 견고한 손실함수 사용
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

class UltraRobustPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.imputer = SimpleImputer(strategy='median')
        self.uncertainty_models = []
        
    def get_core_descriptors(self, mol):
        """핵심 분자 설명자"""
        try:
            desc_dict = {
                'MolWt': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'NumHDonors': Descriptors.NumHDonors(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                'NumAromaticRings': Descriptors.NumAromaticRings(mol),
                'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
                'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
                'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
                'RingCount': Descriptors.RingCount(mol),
                'BertzCT': Descriptors.BertzCT(mol),
            }
            return desc_dict
        except:
            return {key: 0 for key in ['MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 
                                      'NumRotatableBonds', 'NumAromaticRings', 'NumSaturatedRings',
                                      'NumAliphaticRings', 'HeavyAtomCount', 'RingCount', 'BertzCT']}

    def smiles_to_robust_features(self, smiles):
        """견고한 특성 추출"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        try:
            # 1. Multiple Morgan Fingerprints (다양한 반지름으로 견고성 확보)
            morgan_features = []
            for radius in [1, 2, 3]:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=CFG['NBITS'])
                arr = np.zeros((CFG['NBITS'],))
                DataStructs.ConvertToNumpyArray(fp, arr)
                morgan_features.append(arr)

            # 2. MACCS Keys
            fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
            arr_maccs = np.zeros((167,))
            DataStructs.ConvertToNumpyArray(fp_maccs, arr_maccs)

            # 3. 핵심 분자 설명자
            descriptors = self.get_core_descriptors(mol)

            # 4. CYP3A4 핵심 구조 알림
            structural_alerts = {
                'HasBenzene': int(mol.HasSubstructMatch(Chem.MolFromSmarts('c1ccccc1'))),
                'HasPyridine': int(mol.HasSubstructMatch(Chem.MolFromSmarts('c1ccncc1'))),
                'HasImidazole': int(mol.HasSubstructMatch(Chem.MolFromSmarts('c1cnc[nH]1'))),
                'HasAmide': int(mol.HasSubstructMatch(Chem.MolFromSmarts('C(=O)N'))),
                'HasEster': int(mol.HasSubstructMatch(Chem.MolFromSmarts('C(=O)O'))),
                'HasFluorine': int(mol.HasSubstructMatch(Chem.MolFromSmarts('[F]'))),
                'HasChlorine': int(mol.HasSubstructMatch(Chem.MolFromSmarts('[Cl]'))),
                'HasNitro': int(mol.HasSubstructMatch(Chem.MolFromSmarts('[N+](=O)[O-]'))),
                'HasTrifluoromethyl': int(mol.HasSubstructMatch(Chem.MolFromSmarts('C(F)(F)F'))),
                'HasIndole': int(mol.HasSubstructMatch(Chem.MolFromSmarts('c1ccc2[nH]ccc2c1'))),
            }

            return morgan_features, arr_maccs, descriptors, structural_alerts
            
        except Exception as e:
            return None

    def prepare_robust_data(self, df, is_training=True):
        """견고한 데이터 준비"""
        print("견고한 특성 추출 중...")
        
        all_features = []
        failed_count = 0
        
        for i, smiles in enumerate(df['Canonical_Smiles']):
            if i % 200 == 0:
                print(f"처리 중: {i}/{len(df)}")
            
            result = self.smiles_to_robust_features(smiles)
            
            if result is None:
                failed_count += 1
                morgan_features = [np.zeros(CFG['NBITS']) for _ in range(3)]
                arr_maccs = np.zeros(167)
                descriptors = {key: 0 for key in ['MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 
                                                 'NumRotatableBonds', 'NumAromaticRings', 'NumSaturatedRings',
                                                 'NumAliphaticRings', 'HeavyAtomCount', 'RingCount', 'BertzCT']}
                structural_alerts = {key: 0 for key in ['HasBenzene', 'HasPyridine', 'HasImidazole', 
                                                       'HasAmide', 'HasEster', 'HasFluorine', 
                                                       'HasChlorine', 'HasNitro', 'HasTrifluoromethyl', 'HasIndole']}
            else:
                morgan_features, arr_maccs, descriptors, structural_alerts = result

            all_features.append((morgan_features, arr_maccs, list(descriptors.values()), 
                               list(structural_alerts.values())))

        # 특성 결합
        morgan_all = np.hstack([np.array([item[0][i] for item in all_features]) for i in range(3)])
        maccs_all = np.array([item[1] for item in all_features])
        desc_all = np.array([item[2] for item in all_features])
        alert_all = np.array([item[3] for item in all_features])

        X = np.hstack([morgan_all, maccs_all, desc_all, alert_all])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if is_training:
            X = self.imputer.fit_transform(X)
        else:
            X = self.imputer.transform(X)
        
        print(f"✅ 견고한 특성 추출 완료: {X.shape[1]:,}개 특성")
        return X

    def get_leaderboard_score(self, y_true, y_pred):
        """정확한 리더보드 평가 지표"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        y_range = np.max(y_true) - np.min(y_true)
        normalized_rmse = rmse / y_range
        A = min(normalized_rmse, 1)
        
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        if np.isnan(correlation):
            correlation = 0
        B = np.clip(correlation, 0, 1)
        
        score = 0.5 * (1 - A) + 0.5 * B
        return score, A, B, correlation

    def simulate_random_80_percent(self, y_true, y_pred, n_simulations=1000):
        """🎯 무작위 80% 샘플링 시뮬레이션"""
        scores = []
        indices = np.arange(len(y_true))
        
        for _ in range(n_simulations):
            # 무작위 80% 선택
            sample_size = int(len(indices) * 0.8)
            random_indices = np.random.choice(indices, size=sample_size, replace=False)
            
            y_true_sample = y_true.iloc[random_indices] if hasattr(y_true, 'iloc') else y_true[random_indices]
            y_pred_sample = y_pred[random_indices]
            
            score, _, _, _ = self.get_leaderboard_score(y_true_sample, y_pred_sample)
            scores.append(score)
        
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'q25': np.percentile(scores, 25),
            'q75': np.percentile(scores, 75)
        }

    def robust_ensemble_predict(self, models, X_test, uncertainty_weight=0.3):
        """🛡️ 불확실성을 고려한 견고한 앙상블 예측"""
        predictions = []
        uncertainties = []
        
        for model_group in models:
            group_preds = []
            for model in model_group:
                pred = model.predict(X_test)
                group_preds.append(pred)
            
            # 그룹 내 예측 분산 = 불확실성
            group_preds = np.array(group_preds)
            group_mean = np.mean(group_preds, axis=0)
            group_std = np.std(group_preds, axis=0)
            
            predictions.append(group_mean)
            uncertainties.append(group_std)
        
        predictions = np.array(predictions)
        uncertainties = np.array(uncertainties)
        
        # 불확실성 가중 평균
        weights = 1.0 / (uncertainties + 1e-8)
        weights = weights / np.sum(weights, axis=0)
        
        final_pred = np.sum(predictions * weights, axis=0)
        final_uncertainty = np.mean(uncertainties, axis=0)
        
        return final_pred, final_uncertainty

    def train_ultra_robust_ensemble(self, X_train_full, y_train_full, X_test_full):
        """🛡️ 무작위 80% 샘플링에 최적화된 초견고 앙상블"""
        print("🛡️ 무작위 80% 샘플링 견고 앙상블 훈련 시작...")
        
        all_models = []
        all_predictions = []
        oof_predictions = np.zeros(len(X_train_full))
        
        # 🔥 Bootstrap + Multiple CV 전략으로 극대 견고성
        cv_strategies = [
            ('KFold', KFold(n_splits=CFG['N_SPLITS'], shuffle=True)),
            ('RepeatedKFold', RepeatedKFold(n_splits=10, n_repeats=CFG['N_REPEATS'])),
            ('ShuffleSplit', ShuffleSplit(n_splits=15, test_size=0.2)),
        ]
        
        for seed_idx, seed in enumerate(CFG['SEEDS']):
            print(f"\n🔄 시드 {seed} ({seed_idx+1}/{len(CFG['SEEDS'])}) 처리 중...")
            seed_everything(seed)
            
            seed_models = []
            seed_test_preds = []
            
            for cv_name, cv_splitter in cv_strategies:
                print(f"  CV 전략: {cv_name}")
                cv_splitter.random_state = seed
                
                for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train_full, y_train_full)):
                    if fold_idx >= 5:  # 시간 절약을 위해 5개 폴드만
                        break
                        
                    X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
                    y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
                    
                    # 🔥 Bootstrap 샘플링으로 추가 견고성
                    bootstrap_indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
                    X_train_bootstrap = X_train[bootstrap_indices]
                    y_train_bootstrap = y_train.iloc[bootstrap_indices]
                    
                    # 다양한 스케일러 사용
                    scalers = [RobustScaler(), StandardScaler()]
                    scaler = scalers[fold_idx % len(scalers)]
                    
                    X_train_scaled = scaler.fit_transform(X_train_bootstrap)
                    X_val_scaled = scaler.transform(X_val)
                    X_test_scaled = scaler.transform(X_test_full)
                    
                    # 🎯 견고성에 특화된 모델들
                    robust_models = [
                        ('lgb', lgb.LGBMRegressor(
                            objective='regression', metric='rmse', verbose=-1, n_jobs=-1,
                            n_estimators=2000, learning_rate=0.01, num_leaves=15,
                            max_depth=6, feature_fraction=0.8, bagging_fraction=0.8,
                            bagging_freq=5, min_child_samples=20, lambda_l1=0.1, lambda_l2=0.1,
                            seed=seed
                        )),
                        ('xgb', xgb.XGBRegressor(
                            n_estimators=1500, learning_rate=0.01, max_depth=6,
                            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                            random_state=seed, n_jobs=-1
                        )),
                        ('rf', RandomForestRegressor(
                            n_estimators=800, max_depth=8, min_samples_split=10,
                            min_samples_leaf=5, max_features=0.8, bootstrap=True,
                            random_state=seed, n_jobs=-1
                        )),
                        ('et', ExtraTreesRegressor(
                            n_estimators=800, max_depth=8, min_samples_split=10,
                            min_samples_leaf=5, max_features=0.8, bootstrap=True,
                            random_state=seed, n_jobs=-1
                        )),
                    ]
                    
                    fold_models = []
                    fold_test_preds = []
                    
                    for model_name, model in robust_models:
                        try:
                            if model_name in ['lgb', 'xgb']:
                                if model_name == 'lgb':
                                    model.fit(X_train_scaled, y_train_bootstrap, 
                                            eval_set=[(X_val_scaled, y_val)],
                                            callbacks=[lgb.early_stopping(100, verbose=False)])
                                else:  # xgb
                                    try:
                                        model.fit(X_train_scaled, y_train_bootstrap, 
                                                eval_set=[(X_val_scaled, y_val)],
                                                callbacks=[xgb.callback.EarlyStopping(rounds=100)])
                                    except:
                                        model.fit(X_train_scaled, y_train_bootstrap)
                            else:
                                model.fit(X_train_scaled, y_train_bootstrap)
                            
                            test_pred = model.predict(X_test_scaled)
                            val_pred = model.predict(X_val_scaled)
                            
                            fold_models.append(model)
                            fold_test_preds.append(test_pred)
                            
                            # OOF 예측 누적
                            oof_predictions[val_idx] += val_pred / (len(CFG['SEEDS']) * len(cv_strategies) * 5 * len(robust_models))
                            
                        except Exception as e:
                            print(f"    {model_name} 실패: {e}")
                            continue
                    
                    if fold_models:
                        seed_models.append(fold_models)
                        seed_test_preds.append(np.mean(fold_test_preds, axis=0))
            
            if seed_models:
                all_models.append(seed_models)
                all_predictions.append(np.mean(seed_test_preds, axis=0))
        
        # 🎯 불확실성 고려 최종 앙상블
        if all_predictions:
            final_test_preds = np.mean(all_predictions, axis=0)
            final_uncertainty = np.std(all_predictions, axis=0)
        else:
            final_test_preds = np.full(len(X_test_full), y_train_full.mean())
            final_uncertainty = np.zeros(len(X_test_full))
        
        # 성능 평가
        final_score, A, B, corr = self.get_leaderboard_score(y_train_full, oof_predictions)
        
        # 🎯 무작위 80% 샘플링 시뮬레이션
        random_80_stats = self.simulate_random_80_percent(y_train_full, oof_predictions)
        
        print(f"\n🏆 최종 초견고 앙상블 성능:")
        print(f"전체 데이터 스코어: {final_score:.4f}")
        print(f"무작위 80% 평균: {random_80_stats['mean']:.4f} ± {random_80_stats['std']:.4f}")
        print(f"무작위 80% 범위: {random_80_stats['min']:.4f} ~ {random_80_stats['max']:.4f}")
        print(f"무작위 80% Q1-Q3: {random_80_stats['q25']:.4f} ~ {random_80_stats['q75']:.4f}")
        print(f"상관관계 (B): {B:.4f}")
        print(f"예측 불확실성 평균: {np.mean(final_uncertainty):.4f}")
        
        # 🛡️ 견고성 기반 후처리
        final_test_preds = self.robust_post_process(final_test_preds, final_uncertainty, y_train_full)
        
        return final_test_preds

    def robust_post_process(self, predictions, uncertainties, y_train):
        """🛡️ 견고성 기반 후처리"""
        # 1. 기본 클리핑
        predictions = np.clip(predictions, 0, 100)
        
        # 2. 불확실성이 높은 예측을 보수적으로 조정
        high_uncertainty_mask = uncertainties > np.percentile(uncertainties, 75)
        conservative_adjustment = 0.9  # 불확실성이 높으면 조금 더 보수적으로
        
        predictions[high_uncertainty_mask] = (
            predictions[high_uncertainty_mask] * conservative_adjustment + 
            y_train.mean() * (1 - conservative_adjustment)
        )
        
        # 3. 최종 클리핑
        predictions = np.clip(predictions, 0, 100)
        
        return predictions

def main():
    print("🛡️ 무작위 80% 샘플링 견고 CYP3A4 예측 모델 🛡️")
    print("=" * 70)
    print("🎯 전략: 극대 견고성 + 불확실성 정량화 + 다양성 앙상블")
    print("⚡ 예상 실행 시간: 15-20분")
    
    try:
        # 데이터 로드
        print("\n데이터 로드 중...")
        train_df = pd.read_csv('data/train.csv')
        test_df = pd.read_csv('data/test.csv')
        sample_submission = pd.read_csv('data/sample_submission.csv')
        
        print(f"훈련 데이터: {len(train_df)}개")
        print(f"테스트 데이터: {len(test_df)}개")
        print(f"0 라벨 개수: {(train_df['Inhibition'] == 0).sum()}개 (모두 유지)")
        
        # 모델 초기화
        predictor = UltraRobustPredictor()
        
        # 견고한 특성 추출
        X_train_full = predictor.prepare_robust_data(train_df, is_training=True)
        X_test_full = predictor.prepare_robust_data(test_df, is_training=False)
        y_train_full = train_df['Inhibition']
        
        print(f"\n🚀 견고한 특성 수: {X_train_full.shape[1]:,}")
        print(f"🛡️ 시드 개수: {len(CFG['SEEDS'])}개")
        print(f"📊 Bootstrap 샘플: {CFG['BOOTSTRAP_SAMPLES']}개")
        
        # 초견고 앙상블 훈련
        test_preds = predictor.train_ultra_robust_ensemble(X_train_full, y_train_full, X_test_full)
        
        # 제출 파일 생성
        submission = sample_submission.copy()
        submission['Inhibition'] = test_preds
        submission['Inhibition'] = np.clip(submission['Inhibition'], 0, 100)
        
        submission.to_csv('submission_ultra_robust.csv', index=False)
        print(f"\n✅ 제출 파일이 'submission_ultra_robust.csv'로 저장되었습니다.")
        
        print("\n🎯 최종 예측 결과 요약:")
        print(submission['Inhibition'].describe())
        print(f"\n🏆 목표: 무작위 80% 샘플링에 견고한 일관된 성능!")
        print(f"🛡️ 핵심: 예측 안정성 > RMSE 최적화")

    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 