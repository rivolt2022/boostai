# 필요한 라이브러리들을 임포트
import pandas as pd
import numpy as np
import os
import random
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 전역 설정 변수들
CFG = {
    'NBITS': 2048,      # Morgan 지문의 비트 수
    'SEED': 42,         # 재현성을 위한 랜덤 시드
    'N_SPLITS': 5,      # K-폴드 교차 검증에서 사용할 폴드 수
    'N_TRIALS': 100     # Optuna 하이퍼파라미터 최적화 시도 횟수
}

def seed_everything(seed):
    """모든 랜덤 시드를 설정하여 실험의 재현성을 보장하는 함수"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# 전역 시드 설정
seed_everything(CFG['SEED'])

def load_data():
    """데이터를 로드하는 함수"""
    data_dir = Path("data")
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")
    return train_df, test_df, sample_submission

def smiles_to_fingerprint(smiles):
    """SMILES 문자열을 Morgan 지문으로 변환하는 함수"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=CFG['NBITS'])
        arr = np.zeros((CFG['NBITS'],))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    return None

def calculate_rdkit_descriptors(smiles):
    """SMILES 문자열로부터 RDKit 분자 설명자들을 계산하는 함수"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.full((len(Descriptors._descList),), np.nan)
    descriptors = [desc_func(mol) for _, desc_func in Descriptors._descList]
    return np.array(descriptors)

def get_score(y_true, y_pred):
    """평가 지표에 따른 스코어 계산 함수 (NRMSE와 상관계수 기반)"""
    # NRMSE 계산
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    A = 1 - min(nrmse, 1)
    
    # 피어슨 상관계수 계산
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    B = np.clip(corr, 0, 1)  # 0과 1 사이로 클리핑
    
    # 최종 스코어 계산
    score = 0.5 * A + 0.5 * B
    return score

def create_stratified_folds(y, n_splits=5):
    """회귀 문제에서 stratified 폴드를 생성하는 함수"""
    # 타겟을 구간으로 나누어 stratify 수행
    bins = np.percentile(y, np.linspace(0, 100, 11))  # 10개 구간으로 나누기
    y_binned = np.digitize(y, bins)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CFG['SEED'])
    return skf.split(np.zeros(len(y)), y_binned)

def objective(trial, X, y):
    """Optuna 하이퍼파라미터 최적화를 위한 목적 함수"""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbose': -1,
        'n_jobs': -1,
        'seed': CFG['SEED'],
        'boosting_type': 'gbdt',
        'n_estimators': 3000,
        # 최적화할 하이퍼파라미터들
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
    }

    # Stratified K-fold 교차 검증
    fold_scores = []
    oof_preds = np.zeros(len(X))
    
    for train_idx, val_idx in create_stratified_folds(y, CFG['N_SPLITS']):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)],
                  eval_metric='rmse', 
                  callbacks=[lgb.early_stopping(200, verbose=False)])
        
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        
        # 각 폴드의 스코어 계산
        fold_score = get_score(y_val, val_pred)
        fold_scores.append(fold_score)
    
    # 전체 OOF 스코어 계산
    oof_score = get_score(y, oof_preds)
    return oof_score

def main():
    print("=== CYP3A4 효소 저해율 예측 모델 ===")
    print("1. 데이터 로딩 및 전처리...")
    
    # 데이터 로드
    train_df, test_df, sample_submission = load_data()
    print(f"Train 데이터: {len(train_df)}개")
    print(f"Test 데이터: {len(test_df)}개")
    print(f"Inhibition 평균: {train_df['Inhibition'].mean():.2f}%")
    print(f"Inhibition 범위: {train_df['Inhibition'].min():.2f}% ~ {train_df['Inhibition'].max():.2f}%")
    print(f"0값 개수: {sum(train_df['Inhibition'] == 0)}개 (전체의 {sum(train_df['Inhibition'] == 0)/len(train_df)*100:.1f}%)")
    
    print("\n2. 분자 특성 추출...")
    
    # 훈련 데이터 특성 추출
    print("훈련 데이터 Morgan 지문 계산 중...")
    train_df['fingerprint'] = train_df['Canonical_Smiles'].apply(smiles_to_fingerprint)
    print("훈련 데이터 RDKit 설명자 계산 중...")
    train_df['descriptors'] = train_df['Canonical_Smiles'].apply(calculate_rdkit_descriptors)
    
    # 테스트 데이터 특성 추출
    print("테스트 데이터 Morgan 지문 계산 중...")
    test_df['fingerprint'] = test_df['Canonical_Smiles'].apply(smiles_to_fingerprint)
    print("테스트 데이터 RDKit 설명자 계산 중...")
    test_df['descriptors'] = test_df['Canonical_Smiles'].apply(calculate_rdkit_descriptors)
    
    # 유효한 데이터만 필터링
    train_valid = train_df.dropna(subset=['fingerprint', 'descriptors'])
    test_valid_mask = test_df['fingerprint'].notna() & test_df['descriptors'].notna()
    test_valid = test_df[test_valid_mask].copy()
    
    print(f"유효한 훈련 데이터: {len(train_valid)}개 (전체의 {len(train_valid)/len(train_df)*100:.1f}%)")
    print(f"유효한 테스트 데이터: {sum(test_valid_mask)}개 (전체의 {sum(test_valid_mask)/len(test_df)*100:.1f}%)")
    
    # 설명자 데이터 처리
    desc_train = np.stack(train_valid['descriptors'].values)
    desc_test = np.stack(test_valid['descriptors'].values)
    
    # NaN 값을 평균으로 대체
    desc_mean = np.nanmean(desc_train, axis=0)
    desc_train = np.nan_to_num(desc_train, nan=desc_mean)
    desc_test = np.nan_to_num(desc_test, nan=desc_mean)
    
    # 설명자 정규화
    scaler = StandardScaler()
    desc_train_scaled = scaler.fit_transform(desc_train)
    desc_test_scaled = scaler.transform(desc_test)
    
    # 지문과 설명자 결합
    fp_train = np.stack(train_valid['fingerprint'].values)
    fp_test = np.stack(test_valid['fingerprint'].values)
    
    X_train = np.hstack([fp_train, desc_train_scaled])
    X_test = np.hstack([fp_test, desc_test_scaled])
    y_train = train_valid['Inhibition'].values
    
    print(f"최종 특성 차원: {X_train.shape[1]}개")
    
    print("\n3. 하이퍼파라미터 최적화...")
    study = optuna.create_study(direction='maximize', study_name='cyp3a4_inhibition')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=CFG['N_TRIALS'])
    
    print(f"최적화 완료. 최고 스코어: {study.best_value:.4f}")
    print("최적 파라미터:", study.best_params)
    
    print("\n4. 최종 모델 훈련 및 예측...")
    
    # 최적 파라미터로 모델 설정
    best_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbose': -1,
        'n_jobs': -1,
        'seed': CFG['SEED'],
        'boosting_type': 'gbdt',
        'n_estimators': 3000
    }
    best_params.update(study.best_params)
    
    # 앙상블 예측
    test_preds = np.zeros(len(X_test))
    oof_preds = np.zeros(len(X_train))
    feature_importance = np.zeros(X_train.shape[1])
    
    for fold, (train_idx, val_idx) in enumerate(create_stratified_folds(y_train, CFG['N_SPLITS'])):
        print(f"폴드 {fold+1}/{CFG['N_SPLITS']} 훈련 중...")
        
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        model = lgb.LGBMRegressor(**best_params)
        model.fit(X_fold_train, y_fold_train,
                  eval_set=[(X_fold_val, y_fold_val)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(200, verbose=False)])
        
        # OOF 예측
        oof_preds[val_idx] = model.predict(X_fold_val)
        
        # 테스트 예측
        test_preds += model.predict(X_test) / CFG['N_SPLITS']
        
        # 특성 중요도 누적
        feature_importance += model.feature_importances_ / CFG['N_SPLITS']
        
        # 폴드 스코어 출력
        fold_score = get_score(y_fold_val, oof_preds[val_idx])
        print(f"폴드 {fold+1} 스코어: {fold_score:.4f}")
    
    # 최종 OOF 스코어
    final_oof_score = get_score(y_train, oof_preds)
    print(f"\n최종 OOF 스코어: {final_oof_score:.4f}")
    
    print("\n5. 제출 파일 생성...")
    
    # 제출 파일 생성
    submission = sample_submission.copy()
    
    # 예측 결과를 데이터프레임으로 만들기
    pred_df = pd.DataFrame({
        'ID': test_valid['ID'].values,
        'Inhibition': test_preds
    })
    
    # 예측값을 0 이상으로 클리핑
    pred_df['Inhibition'] = np.clip(pred_df['Inhibition'], 0, None)
    
    # 제출 파일에 예측값 병합
    submission = submission.merge(pred_df, on='ID', how='left', suffixes=('', '_pred'))
    submission['Inhibition'] = submission['Inhibition_pred'].fillna(train_df['Inhibition'].mean())
    submission = submission[['ID', 'Inhibition']]
    
    # 제출 파일 저장
    submission.to_csv('submission.csv', index=False)
    print("제출 파일 'submission.csv' 생성 완료!")
    
    # 예측 통계
    print(f"\n=== 예측 결과 통계 ===")
    print(f"전체 예측 수: {len(submission)}개")
    print(f"유효한 예측 수: {len(pred_df)}개")
    print(f"예측값 범위: {submission['Inhibition'].min():.2f}% ~ {submission['Inhibition'].max():.2f}%")
    print(f"예측값 평균: {submission['Inhibition'].mean():.2f}%")
    print(f"예측값 중앙값: {submission['Inhibition'].median():.2f}%")
    
    # 특성 중요도 상위 10개
    print(f"\n=== 상위 특성 중요도 ===")
    top_features = np.argsort(feature_importance)[-10:][::-1]
    for i, feat_idx in enumerate(top_features):
        if feat_idx < CFG['NBITS']:
            feat_name = f"Morgan_bit_{feat_idx}"
        else:
            desc_idx = feat_idx - CFG['NBITS']
            feat_name = f"Descriptor_{desc_idx}"
        print(f"{i+1:2d}. {feat_name}: {feature_importance[feat_idx]:.1f}")

if __name__ == "__main__":
    main()
