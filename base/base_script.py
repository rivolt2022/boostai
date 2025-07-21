

# 필요한 라이브러리들을 임포트
import pandas as pd  # 데이터 처리 및 분석을 위한 라이브러리
import numpy as np   # 수치 계산을 위한 라이브러리
import os           # 운영체제 관련 기능
import random       # 난수 생성
from rdkit import Chem  # 화학 정보 처리를 위한 RDKit 라이브러리
from rdkit.Chem import AllChem, DataStructs, Descriptors  # RDKit의 화학 계산 기능들
from sklearn.preprocessing import StandardScaler  # 데이터 정규화
from sklearn.model_selection import KFold  # 교차 검증을 위한 K-폴드 분할
import lightgbm as lgb  # LightGBM 머신러닝 모델
from sklearn.metrics import r2_score, mean_squared_error  # 모델 평가 지표
import optuna  # 하이퍼파라미터 최적화 라이브러리
from pathlib import Path  # 경로 처리를 위한 라이브러리

# 전역 설정 변수들
CFG = {
    'NBITS': 2048,      # Morgan 지문의 비트 수 (분자 특성을 나타내는 벡터의 차원)
    'SEED': 42,         # 재현성을 위한 랜덤 시드
    'N_SPLITS': 5,      # K-폴드 교차 검증에서 사용할 폴드 수
    'N_TRIALS': 50      # Optuna 하이퍼파라미터 최적화 시도 횟수
}

def seed_everything(seed):
    """
    모든 랜덤 시드를 설정하여 실험의 재현성을 보장하는 함수
    
    Args:
        seed (int): 설정할 랜덤 시드 값
    """
    random.seed(seed)  # Python 내장 random 모듈 시드 설정
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python 해시 시드 설정
    np.random.seed(seed)  # NumPy 랜덤 시드 설정

# 전역 시드 설정
seed_everything(CFG['SEED'])

def load_and_preprocess_data():
    """
    ChEMBL과 PubChem 데이터를 로드하고 전처리하는 함수
    
    Returns:
        pandas.DataFrame: 전처리된 훈련 데이터프레임 또는 None (파일 로드 실패 시)
    """
    try:
        # 현재 디렉토리 구조에 맞게 경로 수정
        data_dir = Path("../data")  # 데이터 디렉토리 경로 설정
        # ChEMBL 데이터 로드 (세미콜론으로 구분된 CSV)
        chembl = pd.read_csv(data_dir / "ChEMBL_ASK1(IC50).csv", sep=';')
        # PubChem 데이터 로드
        pubchem = pd.read_csv(data_dir / "Pubchem_ASK1.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure data files are in the correct directory.")
        return None

    # ChEMBL 데이터 전처리
    chembl.columns = chembl.columns.str.strip().str.replace('"', '')  # 컬럼명 정리
    chembl = chembl[chembl['Standard Type'] == 'IC50']  # IC50 타입 데이터만 필터링
    # 필요한 컬럼만 선택하고 이름 변경
    chembl = chembl[['Smiles', 'Standard Value']].rename(columns={'Smiles': 'smiles', 'Standard Value': 'ic50_nM'})
    chembl['ic50_nM'] = pd.to_numeric(chembl['ic50_nM'], errors='coerce')  # 숫자형으로 변환

    # PubChem 데이터 전처리
    pubchem = pubchem[['SMILES', 'Activity_Value']].rename(columns={'SMILES': 'smiles', 'Activity_Value': 'ic50_nM'})
    pubchem['ic50_nM'] = pd.to_numeric(pubchem['ic50_nM'], errors='coerce')

    # 두 데이터셋 결합 및 정리
    df = pd.concat([chembl, pubchem], ignore_index=True).dropna(subset=['smiles', 'ic50_nM'])  # 결합 후 결측값 제거
    df = df.drop_duplicates(subset='smiles').reset_index(drop=True)  # 중복 SMILES 제거
    df = df[df['ic50_nM'] > 0]  # 양수 IC50 값만 유지

    return df

def smiles_to_fingerprint(smiles):
    """
    SMILES 문자열을 Morgan 지문(Morgan fingerprint)으로 변환하는 함수
    
    Args:
        smiles (str): 분자의 SMILES 문자열
        
    Returns:
        numpy.ndarray: Morgan 지문 벡터 또는 None (분자 생성 실패 시)
    """
    mol = Chem.MolFromSmiles(smiles)  # SMILES를 RDKit 분자 객체로 변환
    if mol is not None:
        # Morgan 지문 생성 (반지름 2, 2048 비트)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=CFG['NBITS'])
        arr = np.zeros((1,))  # NumPy 배열 초기화
        DataStructs.ConvertToNumpyArray(fp, arr)  # RDKit 지문을 NumPy 배열로 변환
        return arr
    return None

def calculate_rdkit_descriptors(smiles):
    """
    SMILES 문자열로부터 RDKit 분자 설명자들을 계산하는 함수
    
    Args:
        smiles (str): 분자의 SMILES 문자열
        
    Returns:
        numpy.ndarray: RDKit 분자 설명자들의 배열
    """
    mol = Chem.MolFromSmiles(smiles)  # SMILES를 RDKit 분자 객체로 변환
    if mol is None: 
        # 분자 생성 실패 시 모든 설명자를 NaN으로 채움
        return np.full((len(Descriptors._descList),), np.nan)
    # 모든 RDKit 설명자 계산
    descriptors = [desc_func(mol) for _, desc_func in Descriptors._descList]
    return np.array(descriptors)

def IC50_to_pIC50(ic50_nM): 
    """
    IC50 값을 pIC50 값으로 변환하는 함수
    pIC50 = -log10(IC50) = 9 - log10(IC50_nM)
    
    Args:
        ic50_nM (float): 나노몰 단위의 IC50 값
        
    Returns:
        float: pIC50 값
    """
    return 9 - np.log10(ic50_nM)

def pIC50_to_IC50(pIC50): 
    """
    pIC50 값을 IC50 값으로 변환하는 함수
    IC50 = 10^(9 - pIC50)
    
    Args:
        pIC50 (float): pIC50 값
        
    Returns:
        float: 나노몰 단위의 IC50 값
    """
    return 10**(9 - pIC50)

def get_score(y_true_ic50, y_pred_ic50, y_true_pic50, y_pred_pic50):
    """
    모델 성능을 평가하는 커스텀 스코어 함수
    스코어 = 0.4 * A + 0.6 * B
    A = 1 - min(NRMSE, 1), B = R² score
    
    Args:
        y_true_ic50 (array): 실제 IC50 값들
        y_pred_ic50 (array): 예측된 IC50 값들
        y_true_pic50 (array): 실제 pIC50 값들
        y_pred_pic50 (array): 예측된 pIC50 값들
        
    Returns:
        float: 계산된 스코어
    """
    mse = mean_squared_error(y_true_ic50, y_pred_ic50)  # 평균 제곱 오차
    rmse = np.sqrt(mse)  # 평균 제곱근 오차
    # 정규화된 평균 제곱근 오차 (NRMSE)
    nrmse = rmse / (np.max(y_true_ic50) - np.min(y_true_ic50))
    A = 1 - min(nrmse, 1)  # NRMSE 기반 점수 (최대 1로 제한)
    B = r2_score(y_true_pic50, y_pred_pic50)  # R² 점수
    score = 0.4 * A + 0.6 * B  # 최종 스코어 계산
    return score

def objective(trial, X, y):
    """
    Optuna 하이퍼파라미터 최적화를 위한 목적 함수
    
    Args:
        trial: Optuna trial 객체
        X (array): 특성 행렬
        y (array): 타겟 변수
        
    Returns:
        float: 교차 검증을 통한 모델 성능 스코어
    """
    # LightGBM 하이퍼파라미터 설정
    params = {
        'objective': 'regression',  # 회귀 문제
        'metric': 'rmse',          # 평가 지표
        'verbose': -1,             # 출력 억제
        'n_jobs': -1,              # 모든 CPU 코어 사용
        'seed': CFG['SEED'],       # 랜덤 시드
        'boosting_type': 'gbdt',   # 그래디언트 부스팅
        'n_estimators': 2000,      # 트리 개수
        # Optuna가 최적화할 하이퍼파라미터들
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
    }

    # K-폴드 교차 검증 설정
    kf = KFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=CFG['SEED'])
    oof_preds = np.zeros(len(X))  # Out-of-fold 예측값 저장 배열

    # 각 폴드에 대해 모델 훈련 및 예측
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]  # 훈련/검증 데이터 분할
        y_train, y_val = y[train_idx], y[val_idx]
        model = lgb.LGBMRegressor(**params)  # LightGBM 모델 생성
        # 모델 훈련 (조기 종료 포함)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  eval_metric='rmse', callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_preds[val_idx] = model.predict(X_val)  # 검증 데이터 예측

    # IC50 단위로 변환하여 스코어 계산
    y_ic50_true = pIC50_to_IC50(y)  # 실제 pIC50을 IC50으로 변환
    oof_ic50_preds = pIC50_to_IC50(oof_preds)  # 예측 pIC50을 IC50으로 변환
    score = get_score(y_ic50_true, oof_ic50_preds, y, oof_preds)  # 커스텀 스코어 계산
    return score

if __name__ == "__main__":
    print("1. Loading and preprocessing data...")
    train_df = load_and_preprocess_data()  # 훈련 데이터 로드 및 전처리

    if train_df is not None:
        # pIC50 값 계산 (모델링에 사용할 타겟 변수)
        train_df['pIC50'] = IC50_to_pIC50(train_df['ic50_nM'])
        print("\n--- Feature Engineering ---")
        
        # 분자 지문 및 설명자 계산
        train_df['fingerprint'] = train_df['smiles'].apply(smiles_to_fingerprint)  # Morgan 지문
        train_df['descriptors'] = train_df['smiles'].apply(calculate_rdkit_descriptors)  # RDKit 설명자
        train_df.dropna(subset=['fingerprint', 'descriptors'], inplace=True)  # 결측값 제거

        # 설명자 데이터 전처리
        desc_stack = np.stack(train_df['descriptors'].values)  # 설명자들을 배열로 변환
        desc_mean = np.nanmean(desc_stack, axis=0)  # 각 설명자의 평균값 계산
        desc_stack = np.nan_to_num(desc_stack, nan=desc_mean)  # NaN 값을 평균으로 대체

        # 특성 정규화 및 결합
        scaler = StandardScaler()  # 표준화 스케일러
        desc_scaled = scaler.fit_transform(desc_stack)  # 설명자 정규화
        fp_stack = np.stack(train_df['fingerprint'].values)  # 지문들을 배열로 변환
        X = np.hstack([fp_stack, desc_scaled])  # 지문과 설명자를 수평으로 결합
        y = train_df['pIC50'].values  # 타겟 변수

        print("\n--- Starting Hyperparameter Optimization with Optuna ---")
        # Optuna를 사용한 하이퍼파라미터 최적화
        study = optuna.create_study(direction='maximize', study_name='lgbm_tuning')
        study.optimize(lambda trial: objective(trial, X, y), n_trials=CFG['N_TRIALS'])

        print(f"\nOptimization Finished. Best Score: {study.best_value:.4f}")
        print("Best Parameters:", study.best_params)

        # 최적 하이퍼파라미터로 최종 모델 파라미터 설정
        best_params = { 'objective': 'regression', 'metric': 'rmse', 'verbose': -1, 'n_jobs': -1,
                        'seed': CFG['SEED'], 'boosting_type': 'gbdt', 'n_estimators': 2000 }
        best_params.update(study.best_params)  # 최적화된 파라미터 추가

        print("\n--- Training Final Model with Best Parameters ---")
        # 테스트 데이터 로드 및 전처리
        data_dir = Path("../data")
        test_df = pd.read_csv(data_dir / "test.csv")
        test_df['fingerprint'] = test_df['Smiles'].apply(smiles_to_fingerprint)  # 테스트 데이터 지문
        test_df['descriptors'] = test_df['Smiles'].apply(calculate_rdkit_descriptors)  # 테스트 데이터 설명자

        # 유효한 테스트 데이터만 필터링
        valid_test_mask = test_df['fingerprint'].notna() & test_df['descriptors'].notna()
        fp_test_stack = np.stack(test_df.loc[valid_test_mask, 'fingerprint'].values)  # 테스트 지문
        desc_test_stack = np.stack(test_df.loc[valid_test_mask, 'descriptors'].values)  # 테스트 설명자
        desc_test_stack = np.nan_to_num(desc_test_stack, nan=desc_mean)  # NaN 값 처리
        desc_test_scaled = scaler.transform(desc_test_stack)  # 테스트 설명자 정규화
        X_test = np.hstack([fp_test_stack, desc_test_scaled])  # 테스트 특성 결합

        # 앙상블 예측을 위한 K-폴드 훈련
        kf = KFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=CFG['SEED'])
        test_preds = np.zeros(len(X_test))  # 테스트 예측값 초기화

        # 각 폴드에서 모델을 훈련하고 테스트 데이터에 대해 예측
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f"--- Training Fold {fold+1}/{CFG['N_SPLITS']} ---")
            X_train, y_train = X[train_idx], y[train_idx]  # 해당 폴드의 훈련 데이터
            model = lgb.LGBMRegressor(**best_params)  # 최적 파라미터로 모델 생성
            model.fit(X_train, y_train)  # 모델 훈련
            test_preds += model.predict(X_test) / CFG['N_SPLITS']  # 예측값을 폴드 수로 나누어 평균

        print("\n3. Generating submission file...")
        # 샘플 제출 파일 로드
        sample_submission = pd.read_csv(data_dir / "sample_submission.csv")
        # 예측 결과를 IC50 단위로 변환하여 데이터프레임 생성
        pred_df = pd.DataFrame({'ID': test_df.loc[valid_test_mask, 'ID'], 'ASK1_IC50_nM': pIC50_to_IC50(test_preds)})
        # 샘플 제출 파일과 예측 결과를 병합
        submission_df = sample_submission[['ID']].merge(pred_df, on='ID', how='left')
        # 예측하지 못한 ID에 대해서는 훈련 데이터의 평균 IC50 값으로 채움
        submission_df['ASK1_IC50_nM'].fillna(train_df['ic50_nM'].mean(), inplace=True)
        
        # 제출 파일 저장
        submission_path = Path("lgbm_tuned_submission.csv")
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission file saved to: {submission_path}")
        
        # 예측 결과 통계 출력
        print(f"\n--- Prediction Statistics ---")
        print(f"Total predictions: {len(submission_df)}")  # 전체 예측 수
        print(f"Valid predictions: {len(pred_df)}")  # 유효한 예측 수
        print(f"IC50 range: {submission_df['ASK1_IC50_nM'].min():.2f} ~ {submission_df['ASK1_IC50_nM'].max():.2f} nM")  # IC50 범위
        print(f"Mean IC50: {submission_df['ASK1_IC50_nM'].mean():.2f} nM")  # 평균 IC50
        print(f"Median IC50: {submission_df['ASK1_IC50_nM'].median():.2f} nM")  # 중앙값 IC50
    else:
        print("Failed to load training data. Please check the file paths.")