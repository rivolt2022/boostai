# 🔥 다중 알고리즘 CYP3A4 예측 모델 (0.70+ 돌파!)
import pandas as pd
import numpy as np
import os
import random
import pickle
import warnings
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import optuna

# 알고리즘별 임포트
import lightgbm as lgb
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    
try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings('ignore')

# 🔥 다중 알고리즘 설정
CFG = {
    'SEEDS': [42, 123, 456, 789, 999, 2023, 2024],  # 7개 시드로 확대
    'N_SPLITS': 10,     # 10-fold CV
    'N_TRIALS': 75,     # 각 알고리즘별 최적화 시도
    'ALGORITHMS': ['xgb', 'catboost', 'neural', 'lgb', 'rf'],  # 5개 알고리즘
    'ENSEMBLE_METHOD': 'weighted',  # weighted, stacking
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG['SEEDS'][0])

class MultiAlgorithmPredictor:
    def __init__(self):
        self.algorithms = {}
        self.best_params = {}
        self.scaler = StandardScaler()
        self.cv_scores = {}
        self.algorithm_weights = {}
        
    def extract_features(self, smiles):
        """🔬 핵심 특성 추출 (속도 최적화)"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # 1. Morgan Fingerprint (radius=2, 1024 bits)
        fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        arr_morgan = np.zeros((1024,))
        DataStructs.ConvertToNumpyArray(fp_morgan, arr_morgan)
        
        # 2. MACCS Keys
        fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
        arr_maccs = np.zeros((167,))
        DataStructs.ConvertToNumpyArray(fp_maccs, arr_maccs)
        
        # 3. 핵심 RDKit Descriptors (속도 중시)
        core_descriptors = []
        important_descs = [
            'MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors', 'TPSA',
            'NumRotatableBonds', 'NumAromaticRings', 'NumHeteroatoms'
        ]
        
        for desc_name in important_descs:
            try:
                desc_func = getattr(Descriptors, desc_name)
                desc_value = desc_func(mol)
                core_descriptors.append(desc_value if desc_value is not None else 0)
            except:
                core_descriptors.append(0)
        
        # 특성 결합
        features = np.concatenate([arr_morgan, arr_maccs, core_descriptors])
        return features
    
    def prepare_data(self, df, is_training=True):
        """📊 데이터 전처리"""
        smiles_col = 'Canonical_Smiles' if 'Canonical_Smiles' in df.columns else 'SMILES'
        
        print(f"특성 추출 중... ({len(df)}개 분자)")
        features_list = []
        valid_indices = []
        
        for idx, smiles in enumerate(df[smiles_col]):
            features = self.extract_features(smiles)
            if features is not None:
                features_list.append(features)
                valid_indices.append(idx)
            
            if (idx + 1) % 100 == 0:
                print(f"  진행: {idx + 1}/{len(df)}")
        
        X = np.array(features_list)
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        # 스케일링
        if is_training:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        print(f"✅ 특성 추출 완료: {X.shape[1]:,}개 특성")
        return X, valid_indices

    def get_score(self, y_true, y_pred):
        """리더보드 스코어 계산"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        y_range = y_true.max() - y_true.min()
        nrmse = rmse / y_range if y_range > 0 else 0
        
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        if np.isnan(correlation):
            correlation = 0
        
        score = 0.5 * (1 - nrmse) + 0.5 * correlation
        return score

    def optimize_xgboost(self, X, y):
        """🚀 XGBoost 최적화"""
        if not HAS_XGB:
            print("❌ XGBoost를 사용할 수 없습니다.")
            return None
            
        print("🚀 XGBoost 하이퍼파라미터 최적화...")
        
        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'subsample': trial.suggest_float('subsample', 0.6, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 1.0),
                'random_state': CFG['SEEDS'][0],
                'n_jobs': -1,
                'verbosity': 0,
            }
            
            kf = KFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=CFG['SEEDS'][0])
            scores = []
            
            for train_idx, val_idx in kf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_val)
                score = self.get_score(y_val, y_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize', study_name='xgb_opt')
        study.optimize(objective, n_trials=CFG['N_TRIALS'])
        
        print(f"✅ XGBoost 최적화 완료! 스코어: {study.best_value:.4f}")
        return study.best_params, study.best_value

    def optimize_catboost(self, X, y):
        """🐱 CatBoost 최적화"""
        if not HAS_CATBOOST:
            print("❌ CatBoost를 사용할 수 없습니다.")
            return None
            
        print("🐱 CatBoost 하이퍼파라미터 최적화...")
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli']),
                'random_seed': CFG['SEEDS'][0],
                'thread_count': -1,
                'verbose': False,
            }
            
            if params['bootstrap_type'] == 'Bayesian':
                params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 1)
            else:
                params['subsample'] = trial.suggest_float('subsample', 0.6, 0.95)
            
            kf = KFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=CFG['SEEDS'][0])
            scores = []
            
            for train_idx, val_idx in kf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = cb.CatBoostRegressor(**params)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_val)
                score = self.get_score(y_val, y_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize', study_name='cat_opt')
        study.optimize(objective, n_trials=CFG['N_TRIALS'])
        
        print(f"✅ CatBoost 최적화 완료! 스코어: {study.best_value:.4f}")
        return study.best_params, study.best_value

    def optimize_neural_network(self, X, y):
        """🧠 Neural Network 최적화"""
        print("🧠 Neural Network 하이퍼파라미터 최적화...")
        
        def objective(trial):
            # 히든 레이어 구조 최적화
            n_layers = trial.suggest_int('n_layers', 2, 4)
            hidden_sizes = []
            
            for i in range(n_layers):
                size = trial.suggest_int(f'hidden_size_{i}', 50, 500)
                hidden_sizes.append(size)
            
            params = {
                'hidden_layer_sizes': tuple(hidden_sizes),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs']),
                'alpha': trial.suggest_float('alpha', 0.0001, 0.1, log=True),
                'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
                'max_iter': trial.suggest_int('max_iter', 500, 2000),
                'early_stopping': True,
                'validation_fraction': 0.15,
                'n_iter_no_change': 50,
                'random_state': CFG['SEEDS'][0],
            }
            
            if params['solver'] == 'adam':
                params['learning_rate_init'] = trial.suggest_float('learning_rate_init', 0.0001, 0.01, log=True)
            
            kf = KFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=CFG['SEEDS'][0])
            scores = []
            
            for train_idx, val_idx in kf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                try:
                    model = MLPRegressor(**params)
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_val)
                    score = self.get_score(y_val, y_pred)
                    scores.append(score)
                except:
                    scores.append(0.0)  # 실패시 낮은 점수
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize', study_name='nn_opt')
        study.optimize(objective, n_trials=CFG['N_TRIALS'])
        
        print(f"✅ Neural Network 최적화 완료! 스코어: {study.best_value:.4f}")
        return study.best_params, study.best_value

    def optimize_lightgbm(self, X, y):
        """💡 LightGBM 최적화 (참고용)"""
        print("💡 LightGBM 하이퍼파라미터 최적화...")
        
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'verbose': -1,
                'n_jobs': -1,
                'seed': CFG['SEEDS'][0],
                'boosting_type': 'gbdt',
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
                'num_leaves': trial.suggest_int('num_leaves', 15, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.95),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'lambda_l1': trial.suggest_float('lambda_l1', 0.001, 1.0),
                'lambda_l2': trial.suggest_float('lambda_l2', 0.001, 1.0),
            }
            
            kf = KFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=CFG['SEEDS'][0])
            scores = []
            
            for train_idx, val_idx in kf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_val)
                score = self.get_score(y_val, y_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize', study_name='lgb_opt')
        study.optimize(objective, n_trials=CFG['N_TRIALS'])
        
        print(f"✅ LightGBM 최적화 완료! 스코어: {study.best_value:.4f}")
        return study.best_params, study.best_value

    def optimize_random_forest(self, X, y):
        """🌲 Random Forest 최적화"""
        print("🌲 Random Forest 하이퍼파라미터 최적화...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': CFG['SEEDS'][0],
                'n_jobs': -1,
            }
            
            kf = KFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=CFG['SEEDS'][0])
            scores = []
            
            for train_idx, val_idx in kf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = RandomForestRegressor(**params)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_val)
                score = self.get_score(y_val, y_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize', study_name='rf_opt')
        study.optimize(objective, n_trials=CFG['N_TRIALS'])
        
        print(f"✅ Random Forest 최적화 완료! 스코어: {study.best_value:.4f}")
        return study.best_params, study.best_value

    def train_all_algorithms(self, X_train, y_train):
        """🔥 모든 알고리즘 훈련 및 최적화"""
        print("🔥 다중 알고리즘 최적화 시작!")
        print(f"📊 알고리즘: {CFG['ALGORITHMS']}")
        print(f"🎯 각 알고리즘별 {CFG['N_TRIALS']}회 최적화")
        
        results = {}
        
        # 각 알고리즘별 최적화
        for algorithm in CFG['ALGORITHMS']:
            print(f"\n{'='*50}")
            print(f"🚀 {algorithm.upper()} 최적화 시작...")
            
            try:
                if algorithm == 'xgb' and HAS_XGB:
                    params, score = self.optimize_xgboost(X_train, y_train)
                elif algorithm == 'catboost' and HAS_CATBOOST:
                    params, score = self.optimize_catboost(X_train, y_train)
                elif algorithm == 'neural':
                    params, score = self.optimize_neural_network(X_train, y_train)
                elif algorithm == 'lgb':
                    params, score = self.optimize_lightgbm(X_train, y_train)
                elif algorithm == 'rf':
                    params, score = self.optimize_random_forest(X_train, y_train)
                else:
                    print(f"❌ {algorithm} 사용 불가능")
                    continue
                    
                if params is not None:
                    results[algorithm] = {
                        'params': params,
                        'score': score
                    }
                    print(f"✅ {algorithm} 완료: {score:.4f}")
                    
            except Exception as e:
                print(f"❌ {algorithm} 오류: {e}")
                continue
        
        # 결과 정리
        if results:
            print(f"\n🏆 알고리즘별 성능 순위:")
            sorted_results = sorted(results.items(), key=lambda x: x[1]['score'], reverse=True)
            
            for i, (alg, result) in enumerate(sorted_results, 1):
                print(f"  {i}. {alg.upper()}: {result['score']:.4f}")
                
            # 가중치 계산 (성능 기반)
            total_score = sum(result['score'] for result in results.values())
            self.algorithm_weights = {
                alg: result['score'] / total_score 
                for alg, result in results.items()
            }
            
            print(f"\n⚖️ 성능 기반 가중치:")
            for alg, weight in self.algorithm_weights.items():
                print(f"  {alg.upper()}: {weight:.3f}")
        
        self.best_params = results
        return results

    def predict_ensemble(self, X_test):
        """🎯 다중 알고리즘 앙상블 예측"""
        print("🎯 다중 알고리즘 앙상블 예측...")
        
        all_predictions = {}
        X_train = self.X_train_stored
        y_train = self.y_train_stored
        
        # 각 알고리즘별 예측
        for algorithm in self.best_params.keys():
            print(f"🔄 {algorithm.upper()} 앙상블 예측...")
            
            algorithm_preds = []
            
            # 다중 시드 앙상블
            for seed in CFG['SEEDS']:
                kf = KFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=seed)
                seed_preds = np.zeros(len(X_test))
                
                params = self.best_params[algorithm]['params'].copy()
                
                for fold, (train_idx, _) in enumerate(kf.split(X_train, y_train)):
                    X_fold_train = X_train[train_idx]
                    y_fold_train = y_train[train_idx]
                    
                    # 알고리즘별 모델 생성
                    if algorithm == 'xgb':
                        params['random_state'] = seed
                        model = xgb.XGBRegressor(**params)
                    elif algorithm == 'catboost':
                        params['random_seed'] = seed
                        model = cb.CatBoostRegressor(**params)
                    elif algorithm == 'neural':
                        params['random_state'] = seed
                        model = MLPRegressor(**params)
                    elif algorithm == 'lgb':
                        params['seed'] = seed
                        model = lgb.LGBMRegressor(**params)
                    elif algorithm == 'rf':
                        params['random_state'] = seed
                        model = RandomForestRegressor(**params)
                    
                    model.fit(X_fold_train, y_fold_train)
                    fold_pred = model.predict(X_test)
                    seed_preds += fold_pred / CFG['N_SPLITS']
                
                algorithm_preds.append(seed_preds)
            
            # 시드별 평균
            all_predictions[algorithm] = np.mean(algorithm_preds, axis=0)
        
        # 가중 평균 앙상블
        final_predictions = np.zeros(len(X_test))
        for algorithm, predictions in all_predictions.items():
            weight = self.algorithm_weights[algorithm]
            final_predictions += weight * predictions
        
        # 후처리
        final_predictions = np.clip(final_predictions, 0, 100)
        
        print(f"✅ 다중 알고리즘 앙상블 완료!")
        return final_predictions

    def store_training_data(self, X_train, y_train):
        self.X_train_stored = X_train
        self.y_train_stored = y_train

def main():
    print("🔥 다중 알고리즘 CYP3A4 예측 모델 🔥")
    print("=" * 70)
    print("🚀 전략: XGBoost + CatBoost + Neural Network + LightGBM + RandomForest")
    print("🎯 목표: 0.70+ 돌파 (완전히 다른 접근법)")
    print("⚡ 각 알고리즘 독립 최적화 → 성능 기반 가중 앙상블")
    
    try:
        # 데이터 로드
        print("\n📁 데이터 로드...")
        train_df = pd.read_csv('data/train.csv')
        test_df = pd.read_csv('data/test.csv')
        sample_submission = pd.read_csv('data/sample_submission.csv')
        
        print(f"훈련: {len(train_df)}개, 테스트: {len(test_df)}개")
        print(f"Inhibition 범위: {train_df['Inhibition'].min():.1f} ~ {train_df['Inhibition'].max():.1f}")
        
        # 모델 초기화
        predictor = MultiAlgorithmPredictor()
        
        # 특성 추출
        X_train, train_valid_idx = predictor.prepare_data(train_df, is_training=True)
        X_test, test_valid_idx = predictor.prepare_data(test_df, is_training=False)
        
        y_train = train_df.iloc[train_valid_idx]['Inhibition'].values
        
        print(f"✅ 최종 데이터: 훈련 {X_train.shape}, 테스트 {X_test.shape}")
        
        # 훈련 데이터 저장
        predictor.store_training_data(X_train, y_train)
        
        # 모든 알고리즘 훈련
        print("\n🔥 다중 알고리즘 훈련 시작...")
        results = predictor.train_all_algorithms(X_train, y_train)
        
        if not results:
            print("❌ 모든 알고리즘 실패!")
            return
        
        # 앙상블 예측
        print("\n🎯 다중 알고리즘 앙상블 예측...")
        predictions = predictor.predict_ensemble(X_test)
        
        # 제출 파일 생성
        submission = sample_submission.copy()
        submission.iloc[test_valid_idx, submission.columns.get_loc('Inhibition')] = predictions
        
        # 예측하지 못한 부분은 평균값으로 채움
        mean_inhibition = train_df['Inhibition'].mean()
        submission['Inhibition'].fillna(mean_inhibition, inplace=True)
        
        # 저장
        output_file = 'submission_multi_algorithm.csv'
        submission.to_csv(output_file, index=False)
        
        print(f"\n✅ 다중 알고리즘 제출 파일 저장: {output_file}")
        print(f"🔥 최고 성능 알고리즘: {max(results.items(), key=lambda x: x[1]['score'])}")
        print(f"\n📊 최종 예측 결과:")
        print(submission['Inhibition'].describe())
        
        print(f"\n🔥 다중 알고리즘 모델 완료!")
        print(f"🚀 XGBoost + CatBoost + Neural Network + LightGBM + RandomForest")
        print(f"⚖️ 성능 기반 가중 앙상블")
        print(f"🎯 기대 효과: 0.70+ 돌파!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 