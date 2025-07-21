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
    'SEEDS': [42, 123, 456, 789, 999],  # 5개 시드로 균형
    'N_SPLITS': 10,        # 최적화 시 빠른 실행
    'N_REPEATS': 2,        # 반복 교차검증
    'OPTIMIZATION_TRIALS': 100,  # 🔥 Optuna 최적화 시행 수
    'ENSEMBLE_TRIALS': 200,      # 앙상블 시 더 많은 시행
    'ENABLE_OPTIMIZATION': True,  # 🎯 Optuna 최적화 활성화
    'OPTIMIZATION_TIMEOUT': 3600,  # 1시간 최적화 타임아웃
    'RANDOM_SAMPLING_WEIGHT': 0.8,  # 무작위 샘플링 가중치
    'STABILITY_WEIGHT': 0.2,         # 안정성 가중치
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

class OptimizedRobustPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.imputer = SimpleImputer(strategy='median')
        self.optimized_params = {}
        
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
            # 1. Multiple Morgan Fingerprints
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

    def simulate_random_80_percent_cv(self, y_true, y_pred, n_simulations=100):
        """🎯 교차검증용 빠른 무작위 80% 샘플링 시뮬레이션"""
        scores = []
        indices = np.arange(len(y_true))
        
        for _ in range(n_simulations):
            sample_size = int(len(indices) * 0.8)
            random_indices = np.random.choice(indices, size=sample_size, replace=False)
            
            y_true_sample = y_true.iloc[random_indices] if hasattr(y_true, 'iloc') else y_true[random_indices]
            y_pred_sample = y_pred[random_indices]
            
            score, _, _, _ = self.get_leaderboard_score(y_true_sample, y_pred_sample)
            scores.append(score)
        
        return np.mean(scores), np.std(scores)

    def objective_lgb(self, trial, X, y):
        """🔥 LightGBM 무작위 80% 샘플링 최적화 목적함수"""
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1,
            'n_jobs': -1,
            'seed': trial.suggest_categorical('seed', CFG['SEEDS']),
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 1000, 4000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 8, 40),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 0.9),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 0.9),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 1.0, log=True),
        }

        # 🎯 무작위 80% 샘플링 견고성 평가
        cv_scores = []
        random_sampling_scores = []
        
        # 다양한 CV 전략으로 견고성 확인
        cv_strategies = [
            KFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=42),
            RepeatedKFold(n_splits=5, n_repeats=CFG['N_REPEATS'], random_state=42),
            ShuffleSplit(n_splits=8, test_size=0.2, random_state=42)
        ]
        
        for cv_strategy in cv_strategies:
            for train_idx, val_idx in cv_strategy.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                model = lgb.LGBMRegressor(**params)
                model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
                
                y_pred = model.predict(X_val_scaled)
                
                # 기본 스코어
                score, _, _, _ = self.get_leaderboard_score(y_val, y_pred)
                cv_scores.append(score)
                
                # 🎯 무작위 80% 샘플링 시뮬레이션
                random_mean, random_std = self.simulate_random_80_percent_cv(y_val, y_pred)
                random_sampling_scores.append(random_mean)
                
                # 시간 절약을 위해 일부만 평가
                if len(cv_scores) >= 10:
                    break
            if len(cv_scores) >= 10:
                break

        # 🛡️ 견고성 점수 계산
        base_score = np.mean(cv_scores)
        random_score = np.mean(random_sampling_scores)
        stability_score = 1.0 / (1.0 + np.std(cv_scores))  # 안정성 점수
        
        # 🎯 최종 목적함수: 무작위 샘플링 성능 + 안정성
        final_score = (CFG['RANDOM_SAMPLING_WEIGHT'] * random_score + 
                      CFG['STABILITY_WEIGHT'] * stability_score)
        
        return final_score

    def objective_xgb(self, trial, X, y):
        """🔥 XGBoost 무작위 80% 샘플링 최적화 목적함수"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 1000, 4000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.4, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'random_state': trial.suggest_categorical('random_state', CFG['SEEDS']),
            'n_jobs': -1,
        }

        cv_scores = []
        random_sampling_scores = []
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model = xgb.XGBRegressor(**params)
            try:
                model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)],
                         callbacks=[xgb.callback.EarlyStopping(rounds=200)])
            except:
                model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_val_scaled)
            
            score, _, _, _ = self.get_leaderboard_score(y_val, y_pred)
            cv_scores.append(score)
            
            random_mean, _ = self.simulate_random_80_percent_cv(y_val, y_pred)
            random_sampling_scores.append(random_mean)

        base_score = np.mean(cv_scores)
        random_score = np.mean(random_sampling_scores)
        stability_score = 1.0 / (1.0 + np.std(cv_scores))
        
        final_score = (CFG['RANDOM_SAMPLING_WEIGHT'] * random_score + 
                      CFG['STABILITY_WEIGHT'] * stability_score)
        
        return final_score

    def objective_rf(self, trial, X, y):
        """🔥 RandomForest 무작위 80% 샘플링 최적화 목적함수"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 50),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 20),
            'max_features': trial.suggest_float('max_features', 0.4, 1.0),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': trial.suggest_categorical('random_state', CFG['SEEDS']),
            'n_jobs': -1,
        }

        cv_scores = []
        random_sampling_scores = []
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model = RandomForestRegressor(**params)
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_val_scaled)
            
            score, _, _, _ = self.get_leaderboard_score(y_val, y_pred)
            cv_scores.append(score)
            
            random_mean, _ = self.simulate_random_80_percent_cv(y_val, y_pred)
            random_sampling_scores.append(random_mean)

        base_score = np.mean(cv_scores)
        random_score = np.mean(random_sampling_scores)
        stability_score = 1.0 / (1.0 + np.std(cv_scores))
        
        final_score = (CFG['RANDOM_SAMPLING_WEIGHT'] * random_score + 
                      CFG['STABILITY_WEIGHT'] * stability_score)
        
        return final_score

    def optimize_models(self, X_train, y_train):
        """🔥 다중 모델 Optuna 최적화"""
        print("🔥 Optuna 하이퍼파라미터 최적화 시작...")
        
        optimized_params = {}
        
        # 1. LightGBM 최적화
        print("\n🎯 LightGBM 최적화 중...")
        study_lgb = optuna.create_study(direction='maximize', 
                                       sampler=optuna.samplers.TPESampler(seed=42),
                                       pruner=optuna.pruners.MedianPruner())
        study_lgb.optimize(lambda trial: self.objective_lgb(trial, X_train, y_train), 
                          n_trials=CFG['OPTIMIZATION_TRIALS'],
                          timeout=CFG['OPTIMIZATION_TIMEOUT']//3)
        
        optimized_params['lgb'] = study_lgb.best_params
        print(f"✅ LightGBM 최적화 완료 - 최고 스코어: {study_lgb.best_value:.4f}")
        print(f"최적 파라미터: {study_lgb.best_params}")
        
        # 2. XGBoost 최적화
        print("\n🎯 XGBoost 최적화 중...")
        study_xgb = optuna.create_study(direction='maximize', 
                                       sampler=optuna.samplers.TPESampler(seed=123),
                                       pruner=optuna.pruners.MedianPruner())
        study_xgb.optimize(lambda trial: self.objective_xgb(trial, X_train, y_train), 
                          n_trials=CFG['OPTIMIZATION_TRIALS'],
                          timeout=CFG['OPTIMIZATION_TIMEOUT']//3)
        
        optimized_params['xgb'] = study_xgb.best_params
        print(f"✅ XGBoost 최적화 완료 - 최고 스코어: {study_xgb.best_value:.4f}")
        print(f"최적 파라미터: {study_xgb.best_params}")
        
        # 3. RandomForest 최적화
        print("\n🎯 RandomForest 최적화 중...")
        study_rf = optuna.create_study(direction='maximize', 
                                      sampler=optuna.samplers.TPESampler(seed=456),
                                      pruner=optuna.pruners.MedianPruner())
        study_rf.optimize(lambda trial: self.objective_rf(trial, X_train, y_train), 
                         n_trials=CFG['OPTIMIZATION_TRIALS'],
                         timeout=CFG['OPTIMIZATION_TIMEOUT']//3)
        
        optimized_params['rf'] = study_rf.best_params
        print(f"✅ RandomForest 최적화 완료 - 최고 스코어: {study_rf.best_value:.4f}")
        print(f"최적 파라미터: {study_rf.best_params}")
        
        self.optimized_params = optimized_params
        
        # 최적화 결과 저장
        with open('optimized_params.pkl', 'wb') as f:
            pickle.dump(optimized_params, f)
        print("\n💾 최적화된 파라미터가 'optimized_params.pkl'에 저장되었습니다.")
        
        return optimized_params

    def train_optimized_ensemble(self, X_train_full, y_train_full, X_test_full):
        """🎯 최적화된 파라미터로 견고한 앙상블 훈련"""
        print("🎯 최적화된 파라미터로 견고한 앙상블 훈련 시작...")
        
        if CFG['ENABLE_OPTIMIZATION']:
            # Optuna 최적화 실행
            optimized_params = self.optimize_models(X_train_full, y_train_full)
        else:
            # 저장된 파라미터 로드
            try:
                with open('optimized_params.pkl', 'rb') as f:
                    optimized_params = pickle.load(f)
                print("💾 저장된 최적화 파라미터를 로드했습니다.")
            except FileNotFoundError:
                print("❌ 저장된 파라미터가 없습니다. 최적화를 실행합니다.")
                optimized_params = self.optimize_models(X_train_full, y_train_full)
        
        # 🛡️ 최적화된 파라미터로 견고한 앙상블 훈련
        all_predictions = []
        oof_predictions = np.zeros(len(X_train_full))
        
        for seed in CFG['SEEDS']:
            print(f"\n🔄 시드 {seed} 앙상블 훈련 중...")
            seed_everything(seed)
            
            seed_test_preds = []
            
            # 다양한 CV 전략
            cv_strategies = [
                ('KFold', KFold(n_splits=15, shuffle=True, random_state=seed)),
                ('RepeatedKFold', RepeatedKFold(n_splits=8, n_repeats=2, random_state=seed)),
                ('ShuffleSplit', ShuffleSplit(n_splits=10, test_size=0.25, random_state=seed))
            ]
            
            for cv_name, cv_splitter in cv_strategies:
                fold_count = 0
                for train_idx, val_idx in cv_splitter.split(X_train_full, y_train_full):
                    fold_count += 1
                    if fold_count > 3:  # 시간 절약
                        break
                        
                    X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
                    y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
                    
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    X_test_scaled = scaler.transform(X_test_full)
                    
                    # 최적화된 모델들 훈련
                    models = [
                        ('lgb', lgb.LGBMRegressor(**{**{'objective': 'regression', 'metric': 'rmse', 
                                                      'verbose': -1, 'n_jobs': -1}, 
                                                   **optimized_params['lgb']})),
                        ('xgb', xgb.XGBRegressor(**optimized_params['xgb'])),
                        ('rf', RandomForestRegressor(**optimized_params['rf'])),
                        ('et', ExtraTreesRegressor(n_estimators=500, max_depth=8, random_state=seed, n_jobs=-1))
                    ]
                    
                    fold_test_preds = []
                    
                    for model_name, model in models:
                        try:
                            if model_name == 'lgb':
                                model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)],
                                         callbacks=[lgb.early_stopping(100, verbose=False)])
                            elif model_name == 'xgb':
                                try:
                                    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)],
                                             callbacks=[xgb.callback.EarlyStopping(rounds=100)])
                                except:
                                    model.fit(X_train_scaled, y_train)
                            else:
                                model.fit(X_train_scaled, y_train)
                            
                            val_pred = model.predict(X_val_scaled)
                            test_pred = model.predict(X_test_scaled)
                            
                            # OOF 예측 누적
                            oof_predictions[val_idx] += val_pred / (len(CFG['SEEDS']) * len(cv_strategies) * 3 * len(models))
                            fold_test_preds.append(test_pred)
                            
                        except Exception as e:
                            print(f"    {model_name} 실패: {e}")
                            continue
                    
                    if fold_test_preds:
                        seed_test_preds.append(np.mean(fold_test_preds, axis=0))
            
            if seed_test_preds:
                all_predictions.append(np.mean(seed_test_preds, axis=0))
        
        # 최종 앙상블
        if all_predictions:
            final_test_preds = np.mean(all_predictions, axis=0)
        else:
            final_test_preds = np.full(len(X_test_full), y_train_full.mean())
        
        # 성능 평가
        final_score, A, B, corr = self.get_leaderboard_score(y_train_full, oof_predictions)
        
        # 무작위 80% 샘플링 시뮬레이션
        random_scores = []
        for _ in range(1000):
            sample_size = int(len(y_train_full) * 0.8)
            random_indices = np.random.choice(len(y_train_full), size=sample_size, replace=False)
            score, _, _, _ = self.get_leaderboard_score(y_train_full.iloc[random_indices], 
                                                       oof_predictions[random_indices])
            random_scores.append(score)
        
        print(f"\n🏆 최종 최적화된 앙상블 성능:")
        print(f"전체 데이터 스코어: {final_score:.4f}")
        print(f"무작위 80% 평균: {np.mean(random_scores):.4f} ± {np.std(random_scores):.4f}")
        print(f"무작위 80% 범위: {np.min(random_scores):.4f} ~ {np.max(random_scores):.4f}")
        print(f"상관관계 (B): {B:.4f}")
        
        # 후처리
        final_test_preds = np.clip(final_test_preds, 0, 100)
        
        return final_test_preds

def main():
    print("🔥 Optuna 최적화 + 무작위 80% 샘플링 견고 모델 🔥")
    print("=" * 80)
    print(f"🎯 Optuna 최적화: {'활성화' if CFG['ENABLE_OPTIMIZATION'] else '비활성화'}")
    print(f"🔥 최적화 시행 수: {CFG['OPTIMIZATION_TRIALS']}")
    print(f"⏰ 최적화 타임아웃: {CFG['OPTIMIZATION_TIMEOUT']//60}분")
    print(f"⚡ 예상 실행 시간: {'60-90분' if CFG['ENABLE_OPTIMIZATION'] else '20-30분'}")
    
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
        predictor = OptimizedRobustPredictor()
        
        # 견고한 특성 추출
        X_train_full = predictor.prepare_robust_data(train_df, is_training=True)
        X_test_full = predictor.prepare_robust_data(test_df, is_training=False)
        y_train_full = train_df['Inhibition']
        
        print(f"\n🚀 견고한 특성 수: {X_train_full.shape[1]:,}")
        print(f"🛡️ 시드 개수: {len(CFG['SEEDS'])}개")
        
        # 최적화된 앙상블 훈련
        test_preds = predictor.train_optimized_ensemble(X_train_full, y_train_full, X_test_full)
        
        # 제출 파일 생성
        submission = sample_submission.copy()
        submission['Inhibition'] = test_preds
        submission['Inhibition'] = np.clip(submission['Inhibition'], 0, 100)
        
        submission.to_csv('submission_optimized_robust.csv', index=False)
        print(f"\n✅ 제출 파일이 'submission_optimized_robust.csv'로 저장되었습니다.")
        
        print("\n🎯 최종 예측 결과 요약:")
        print(submission['Inhibition'].describe())
        print(f"\n🏆 목표: Optuna 최적화로 무작위 80% 샘플링에 견고한 최고 성능!")

    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 