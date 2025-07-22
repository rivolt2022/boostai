import pandas as pd
import numpy as np
import os
import random
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import lightgbm as lgb
import optuna

# 전역 설정
CFG = {
    'NBITS': 2048,      # Morgan 지문의 비트 수
    'FP_RADIUS': 3,     # Morgan 지문 반지름
    'SEED': 42,         # 재현성을 위한 랜덤 시드
    'N_SPLITS': 10,     # K-폴드 교차 검증 폴드 수 (5 -> 10)
    'N_TRIALS': 100     # Optuna 시도 횟수 (50 -> 100)
}

def seed_everything(seed):
    """
    재현성을 위해 모든 랜덤 시드를 설정하는 함수
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG['SEED'])

def load_data():
    """
    train.csv와 test.csv 데이터를 로드하는 함수
    """
    try:
        data_dir = Path("./data")
        train_df = pd.read_csv(data_dir / "train.csv")
        test_df = pd.read_csv(data_dir / "test.csv")
        return train_df, test_df
    except FileNotFoundError as e:
        print(f"오류: {e}. 'data' 디렉토리에 파일이 있는지 확인하세요.")
        return None, None

def smiles_to_fingerprint(smiles):
    """
    SMILES 문자열을 Morgan 지문으로 변환하는 함수
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, CFG['FP_RADIUS'], nBits=CFG['NBITS'])
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    return None

def calculate_rdkit_descriptors(smiles):
    """
    SMILES 문자열로부터 RDKit 분자 설명자들을 계산하는 함수
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.full((len(Descriptors._descList),), np.nan)
    descriptors = [desc_func(mol) for _, desc_func in Descriptors._descList]
    return np.array(descriptors)

def get_score(y_true, y_pred):
    """
    대회 평가 산식에 따라 점수를 계산하는 함수
    Score = 0.5 * (1 - min(A, 1)) + 0.5 * B
    A = NRMSE
    B = Pearson Correlation Coefficient (clipped)
    """
    # A 계산: Normalized RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    y_true_range = np.max(y_true) - np.min(y_true)
    if y_true_range == 0:
        nrmse = 0 if rmse == 0 else np.inf
    else:
        nrmse = rmse / y_true_range
    A = nrmse
    
    # B 계산: Pearson Correlation Coefficient
    if np.std(y_true) < 1e-6 or np.std(y_pred) < 1e-6:
        correlation = 0.0
    else:
        correlation, _ = pearsonr(y_true, y_pred)
    B = np.clip(correlation, 0, 1)

    score = 0.5 * (1 - min(A, 1)) + 0.5 * B
    return score

def lgbm_score_metric(y_true, y_pred):
    """
    LightGBM을 위한 커스텀 평가 지표 함수
    """
    score = get_score(y_true, y_pred)
    return 'custom_score', score, True # is_higher_better=True

def objective(trial, X, y):
    """
    Optuna 하이퍼파라미터 최적화를 위한 목적 함수
    """
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbose': -1,
        'n_jobs': -1,
        'seed': CFG['SEED'],
        'boosting_type': 'gbdt',
        'n_estimators': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
    }

    kf = KFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=CFG['SEED'])
    oof_preds = np.zeros(len(X))

    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  eval_metric=lgbm_score_metric, # 'rmse' -> custom metric
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        # 예측값 범위 보정
        oof_preds[val_idx] = np.clip(model.predict(X_val), 0, 100)

    score = get_score(y, oof_preds)
    return score

if __name__ == "__main__":
    print("1. 데이터 로딩...")
    train_df, test_df = load_data()

    if train_df is not None and test_df is not None:
        print("\n2. 특징 공학(Feature Engineering)...")
        
        train_df['fingerprint'] = train_df['Canonical_Smiles'].apply(smiles_to_fingerprint)
        train_df['descriptors'] = train_df['Canonical_Smiles'].apply(calculate_rdkit_descriptors)
        train_df.dropna(subset=['fingerprint'], inplace=True)

        desc_stack = np.stack(train_df['descriptors'].values)
        desc_mean = np.nanmean(desc_stack, axis=0)
        desc_stack = np.nan_to_num(desc_stack, nan=desc_mean)

        scaler = StandardScaler()
        desc_scaled = scaler.fit_transform(desc_stack)
        fp_stack = np.stack(train_df['fingerprint'].values)
        X = np.hstack([fp_stack, desc_scaled])
        y = train_df['Inhibition'].values

        fp_feature_names = [f"fp_{i}" for i in range(CFG['NBITS'])]
        desc_feature_names = [name for name, _ in Descriptors._descList]
        all_feature_names = fp_feature_names + desc_feature_names
        X = pd.DataFrame(X, columns=all_feature_names)

        print("\n3. Optuna를 사용한 하이퍼파라미터 최적화...")
        study = optuna.create_study(direction='maximize', study_name='lgbm_inhibition_tuning')
        study.optimize(lambda trial: objective(trial, X, y), n_trials=CFG['N_TRIALS'])

        print(f"\n최적화 완료. 최고 점수: {study.best_value:.4f}")
        print("최적 파라미터:", study.best_params)

        best_params = {
            'objective': 'regression', 'metric': 'rmse', 'verbose': -1, 'n_jobs': -1,
            'seed': CFG['SEED'], 'boosting_type': 'gbdt', 'n_estimators': 2000
        }
        best_params.update(study.best_params)

        print("\n4. 최적 파라미터로 최종 모델 훈련...")
        test_df['fingerprint'] = test_df['Canonical_Smiles'].apply(smiles_to_fingerprint)
        test_df['descriptors'] = test_df['Canonical_Smiles'].apply(calculate_rdkit_descriptors)
        
        valid_test_mask = test_df['fingerprint'].notna()
        fp_test_stack = np.stack(test_df.loc[valid_test_mask, 'fingerprint'].values)
        desc_test_stack = np.stack(test_df.loc[valid_test_mask, 'descriptors'].values)
        desc_test_stack = np.nan_to_num(desc_test_stack, nan=desc_mean)
        desc_test_scaled = scaler.transform(desc_test_stack)
        X_test = np.hstack([fp_test_stack, desc_test_scaled])
        X_test = pd.DataFrame(X_test, columns=all_feature_names)

        kf = KFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=CFG['SEED'])
        test_preds = np.zeros(len(X_test))

        for fold, (train_idx, _) in enumerate(kf.split(X, y)):
            print(f"--- 훈련 폴드 {fold+1}/{CFG['N_SPLITS']} ---")
            X_train, y_train = X.iloc[train_idx], y[train_idx]
            model = lgb.LGBMRegressor(**best_params)
            model.fit(X_train, y_train)
            test_preds += model.predict(X_test) / CFG['N_SPLITS']

        # 최종 예측값 범위 보정
        test_preds = np.clip(test_preds, 0, 100)

        print("\n5. 제출 파일 생성...")
        data_dir = Path("./data")
        sample_submission = pd.read_csv(data_dir / "sample_submission.csv")
        pred_df = pd.DataFrame({'ID': test_df.loc[valid_test_mask, 'ID'], 'Inhibition': test_preds})
        
        submission_df = sample_submission[['ID']].merge(pred_df, on='ID', how='left')
        submission_df['Inhibition'].fillna(train_df['Inhibition'].mean(), inplace=True)
        
        submission_path = Path("submission.csv")
        submission_df.to_csv(submission_path, index=False)
        print(f"제출 파일 저장 완료: {submission_path}")
        
        print(f"\n--- 예측 결과 통계 ---")
        print(f"총 예측 수: {len(submission_df)}")
        print(f"유효한 예측 수: {len(pred_df)}")
        print(f"Inhibition 범위: {submission_df['Inhibition'].min():.2f}% ~ {submission_df['Inhibition'].max():.2f}%")
        print(f"평균 Inhibition: {submission_df['Inhibition'].mean():.2f}%")
        print(f"중앙값 Inhibition: {submission_df['Inhibition'].median():.2f}%")
    else:
        print("데이터 로드 실패. 파일 경로를 확인하세요.")
