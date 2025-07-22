# --- 라이브러리 임포트 ---
# 데이터 분석 및 처리를 위한 필수 라이브러리
import pandas as pd  # 데이터프레임(표 형태의 데이터)을 다루기 위한 라이브러리
import numpy as np   # 수치 계산, 특히 배열(행렬) 연산을 위한 라이브러리

# 파이썬 기본 내장 라이브러리
import os            # 운영체제와 상호작용하기 위한 라이브러리 (예: 환경 변수 설정)
import random        # 무작위 수를 생성하기 위한 라이브러리
from pathlib import Path  # 파일 및 디렉토리 경로를 객체 지향적으로 다루기 위한 라이브러리

# 화학 정보학 및 특징 공학을 위한 라이브러리
from rdkit import Chem  # 분자 구조를 다루고 화학 계산을 수행하기 위한 핵심 라이브러리
from rdkit.Chem import AllChem, DataStructs, Descriptors  # Morgan Fingerprint, 분자 설명자 등 계산 기능

# 머신러닝 모델링 및 평가를 위한 라이브러리
from sklearn.preprocessing import StandardScaler   # 데이터의 스케일을 조정(정규화)하기 위한 도구
from sklearn.model_selection import KFold          # 교차 검증을 위해 데이터를 여러 부분으로 나누는 도구
from sklearn.metrics import mean_squared_error   # 모델의 예측 오차(MSE)를 계산하는 함수
from scipy.stats import pearsonr                 # 두 변수 간의 피어슨 상관 계수를 계산하는 함수
import lightgbm as lgb                           # 빠르고 효율적인 그래디언트 부스팅 머신러닝 모델
import optuna                                    # 하이퍼파라미터 최적화를 자동화하는 라이브러리

# --- 신경망 모델 및 라이브러리 추가 ---
import torch
from transformers import AutoTokenizer, AutoModel

# --- 전역 설정 (Global Configuration) ---
# 실험의 주요 파라미터들을 코드 상단에 모아두어 관리하기 쉽게 함
CFG = {
    'NBITS': 2048,      # Morgan Fingerprint를 생성할 때 사용할 비트(차원)의 수
    'FP_RADIUS': 3,     # Morgan Fingerprint 계산 시 고려할 원자의 반경. 클수록 더 넓은 구조 정보를 포함.
    'SEEDS': [42, 2024, 101, 7, 99], # 시드 앙상블에 사용할 여러 개의 랜덤 시드 목록
    'N_SPLITS': 10,     # K-Fold 교차 검증 시 데이터를 나눌 폴드(Fold)의 수
    'N_TRIALS': 50,     # Optuna가 하이퍼파라미터 최적화를 위해 시도할 횟수 (시간 관계상 축소)
    'CHEMBERTA_MODEL': 'seyonec/ChemBERTa-zinc-base-v1' # 사용할 사전 훈련 모델
}

# --- 함수 정의 ---

def seed_everything(seed):
    """
    재현성을 위해 모든 종류의 랜덤 시드를 고정하는 함수.
    이 함수를 호출하면 코드를 여러 번 실행해도 항상 동일한 결과를 얻을 수 있음.
    """
    random.seed(seed)  # 파이썬 내장 random 모듈의 시드 고정
    os.environ['PYTHONHASHSEED'] = str(seed)  # 파이썬 해시 함수의 시드 고정
    np.random.seed(seed)  # NumPy 라이브러리의 시드 고정

# Optuna 튜닝 및 초기 데이터 분할의 일관성을 위해 첫 번째 시드로 초기화
seed_everything(CFG['SEEDS'][0])

def load_data():
    """
    대회에서 제공된 'train.csv'와 'test.csv' 데이터를 로드하는 함수
    """
    try:
        data_dir = Path("./data")  # 데이터 파일이 있는 디렉토리 경로
        train_df = pd.read_csv(data_dir / "train.csv") # 훈련 데이터 로드
        test_df = pd.read_csv(data_dir / "test.csv")   # 테스트 데이터 로드
        return train_df, test_df
    except FileNotFoundError as e:
        # 파일이 없을 경우 에러 메시지를 출력하고 프로그램을 안전하게 종료
        print(f"오류: {e}. 'data' 디렉토리에 파일이 있는지 확인하세요.")
        return None, None

def get_chemberta_embeddings(smiles_list, model_name, batch_size=32):
    """
    SMILES 리스트로부터 사전 훈련된 ChemBERTa 모델을 사용하여 임베딩을 추출하는 함수.
    """
    print(f"'{model_name}' 모델을 사용하여 임베딩 추출 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i+batch_size]
            inputs = tokenizer(batch_smiles, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            outputs = model(**inputs)
            # [CLS] 토큰의 임베딩을 사용 (분자 전체의 대표 벡터)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.extend(cls_embeddings)
            if (i // batch_size) % 10 == 0:
                print(f"  {i+len(batch_smiles)} / {len(smiles_list)} 처리 완료...")

    print("임베딩 추출 완료.")
    return np.array(all_embeddings)

def smiles_to_fingerprint(smiles):
    """
    분자의 구조 정보(SMILES 문자열)를 숫자 벡터(Morgan Fingerprint)로 변환하는 함수.
    모델이 학습할 수 있도록 텍스트 정보를 숫자 정보로 바꾸는 과정.
    """
    mol = Chem.MolFromSmiles(smiles)  # SMILES 문자열을 RDKit 분자 객체로 변환
    if mol is not None:
        # Morgan Fingerprint 생성. 분자 내 각 원자 주변의 구조적 특징을 요약한 것.
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, CFG['FP_RADIUS'], nBits=CFG['NBITS'])
        arr = np.zeros((1,))  # 결과를 담을 NumPy 배열 초기화
        DataStructs.ConvertToNumpyArray(fp, arr)  # Fingerprint를 NumPy 배열로 변환
        return arr
    return None  # 분자 객체 생성 실패 시 None 반환

def calculate_rdkit_descriptors(smiles):
    """
    SMILES 문자열로부터 약 200여 개의 물리화학적 특성(분자 설명자)을 계산하는 함수.
    예: 분자량(MolWt), 로그 P(MolLogP) 등.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # 분자 객체 생성 실패 시, 모든 설명자 값을 NaN(Not a Number)으로 채운 배열 반환
        return np.full((len(Descriptors._descList),), np.nan)
    # RDKit에서 제공하는 모든 설명자 함수를 호출하여 값을 계산
    descriptors = [desc_func(mol) for _, desc_func in Descriptors._descList]
    return np.array(descriptors)

def get_score(y_true, y_pred):
    """
    대회 평가 산식에 따라 모델의 성능 점수를 계산하는 함수.
    Score = 0.5 * (1 - min(A, 1)) + 0.5 * B
    A = Normalized RMSE (정규화된 평균 제곱근 오차)
    B = Pearson Correlation Coefficient (피어슨 상관 계수)
    """
    # --- A 계산: Normalized RMSE (NRMSE) ---
    # 예측값과 실제값의 차이(오차)를 측정하는 지표. 작을수록 좋음.
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # 실제값의 범위 (최대값 - 최소값)
    y_true_range = np.max(y_true) - np.min(y_true)
    # 분모가 0이 되는 극단적인 경우를 방지
    if y_true_range == 0:
        nrmse = 0 if rmse == 0 else np.inf
    else:
        # RMSE를 실제값의 범위로 나누어 스케일에 무관하게 만들어 줌
        nrmse = rmse / y_true_range
    A = nrmse
    
    # --- B 계산: Pearson Correlation Coefficient ---
    # 예측값과 실제값 사이의 '선형 관계'의 강도를 측정. 1에 가까울수록 좋음.
    # 즉, 예측값이 실제값의 변화 경향성(오르내림)을 얼마나 잘 따라가는지를 나타냄.
    if np.std(y_true) < 1e-6 or np.std(y_pred) < 1e-6:
        # 데이터의 모든 값이 거의 동일하여 분산이 0에 가까우면 상관계수 계산이 불가능
        correlation = 0.0
    else:
        correlation, _ = pearsonr(y_true, y_pred)
    # 평가 산식에 따라 상관계수 값을 0과 1 사이로 제한(clip)
    B = np.clip(correlation, 0, 1)

    # 최종 점수 계산 (두 지표를 0.5씩 가중 평균)
    score = 0.5 * (1 - min(A, 1)) + 0.5 * B
    return score

def lgbm_score_metric(y_true, y_pred):
    """
    LightGBM 모델 훈련 시, 조기 종료(Early Stopping) 기준으로 사용하기 위한 커스텀 평가지표 함수.
    """
    score = get_score(y_true, y_pred)
    # LightGBM이 인식할 수 있는 형태로 반환: (평가지표 이름, 점수, 높은 점수가 좋은지 여부)
    return 'custom_score', score, True # is_higher_better=True. True이므로 점수가 높아지는 방향으로 학습.

def objective(trial, X, y):
    """
    Optuna 라이브러리가 하이퍼파라미터를 최적화하기 위해 호출하는 목적 함수.
    이 함수의 반환값(score)을 최대화하는 방향으로 최적의 파라미터 조합을 탐색.
    """
    # Optuna가 탐색할 하이퍼파라미터들의 이름과 범위를 정의
    params = {
        'objective': 'regression',          # 목표: 회귀(숫자 예측)
        'metric': 'rmse',                   # 기본 평가지표 (실제로는 커스텀 지표로 덮어쓰므로 큰 의미 없음)
        'verbose': -1,                      # 훈련 과정의 로그를 출력하지 않음
        'n_jobs': -1,                       # 컴퓨터의 모든 CPU 코어를 사용하여 훈련 속도 향상
        'seed': CFG['SEEDS'][0],            # 튜닝 과정의 재현성을 위해 시드를 고정
        'boosting_type': 'gbdt',            # 전통적인 그래디언트 부스팅 결정 트리 방식 사용
        'n_estimators': 2000,               # 앙상블할 트리의 최대 개수 (조기 종료로 최적 개수 자동 탐색)
        
        # --- Optuna가 값을 제안(suggest)하여 최적화할 하이퍼파라미터들 ---
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True), # 학습률. 너무 크면 최적점을 지나치고, 작으면 훈련이 느림.
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),                     # 하나의 트리가 가질 수 있는 최대 리프(터미널) 노드의 수. 모델의 복잡도와 관련.
        'max_depth': trial.suggest_int('max_depth', 3, 10),                         # 트리의 최대 깊이. 과적합 제어.
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),      # 각 트리를 훈련할 때 무작위로 선택할 특징(feature)의 비율.
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),      # 각 트리를 훈련할 때 무작위로 선택할 데이터(row)의 비율.
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),                    # 몇 번의 이터레이션마다 Bagging을 수행할지 결정.
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),         # 리프 노드가 되기 위해 필요한 최소한의 데이터 샘플 수. 과적합 제어.
    }

    # 교차 검증을 위한 데이터 분할기 설정 (Optuna 튜닝 시에는 고정된 시드 사용)
    kf = KFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=CFG['SEEDS'][0])
    # Out-of-Fold (OOF) 예측값을 저장하기 위한 배열 초기화.
    # OOF 예측: 각 데이터 포인트가 '검증용'으로 사용될 때의 예측값을 모은 것. 모델의 일반화 성능을 평가하는 좋은 척도.
    oof_preds = np.zeros(len(X))

    # K-Fold 교차 검증 수행
    for train_idx, val_idx in kf.split(X, y):
        # 훈련 데이터와 검증 데이터 분할
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 정의된 파라미터로 LightGBM 모델 생성
        model = lgb.LGBMRegressor(**params)
        
        # 모델 훈련
        model.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)],           # 훈련 중 성능을 모니터링할 검증 데이터셋 지정
                  eval_metric=lgbm_score_metric,      # 조기 종료의 기준으로 커스텀 평가 함수 사용
                  callbacks=[lgb.early_stopping(100, verbose=False)]) # 100번의 이터레이션 동안 검증 점수가 향상되지 않으면 훈련을 조기 종료
        
        # 검증 데이터에 대한 예측 및 결과 저장
        # Inhibition 값은 0~100 사이의 퍼센트 값이므로, 예측값도 해당 범위로 클리핑하여 안정성 확보
        oof_preds[val_idx] = np.clip(model.predict(X_val), 0, 100)

    # 모든 폴드의 OOF 예측값을 사용하여 최종 성능 점수 계산
    score = get_score(y, oof_preds)
    return score

# --- 메인 실행 블록 ---
# 이 스크립트가 직접 실행될 때만 아래 코드가 동작하도록 함
if __name__ == "__main__":
    # === 1. 데이터 로딩 ===
    print("1. 데이터 로딩...")
    train_df, test_df = load_data()

    if train_df is not None and test_df is not None:
        # === 2. 특징 공학 (Feature Engineering) ===
        # 분자 구조(SMILES)로부터 모델이 학습할 수 있는 유의미한 숫자 형태의 특징들을 추출하고 가공하는 과정
        print("\n2. 특징 공학(Feature Engineering)...")
        
        # --- 2a. ChemBERTa 임베딩 추출 ---
        train_embeddings = get_chemberta_embeddings(train_df['Canonical_Smiles'].tolist(), CFG['CHEMBERTA_MODEL'])
        embedding_feature_names = [f"emb_{i}" for i in range(train_embeddings.shape[1])]
        embedding_df = pd.DataFrame(train_embeddings, columns=embedding_feature_names, index=train_df.index)

        # --- 2b. RDKit 분자 설명자 특징 추출 ---
        train_df['descriptors'] = train_df['Canonical_Smiles'].apply(calculate_rdkit_descriptors)
        
        # 임베딩과 설명자 특징 결합
        train_df = pd.concat([train_df, embedding_df], axis=1)
        train_df.dropna(subset=['descriptors'], inplace=True) # 설명자 계산 실패한 경우 제외

        # 특징들을 수평으로 결합하기 위해 NumPy 배열 형태로 변환
        desc_stack = np.stack(train_df['descriptors'].values)
        
        # 분자 설명자의 결측값(NaN) 처리
        # RDKit이 특정 분자에 대해 설명자를 계산하지 못하는 경우 발생.
        # 훈련 데이터 전체의 각 설명자별 평균값으로 이 결측값을 대체.
        desc_mean = np.nanmean(desc_stack, axis=0)
        desc_stack = np.nan_to_num(desc_stack, nan=desc_mean)

        # 분자 설명자 정규화 (Standard Scaling)
        # 각 특징(열)의 평균을 0, 표준편차를 1로 만들어줌.
        # 스케일이 다른 특징들이 모델 학습에 미치는 영향을 균등하게 만들어 성능 향상에 도움.
        scaler = StandardScaler()
        desc_scaled = scaler.fit_transform(desc_stack)
        
        # === 최종 훈련 데이터셋(X, y) 생성 ===
        # ChemBERTa 임베딩과 정규화된 분자 설명자를 수평으로 결합하여 최종 특징 행렬(X) 생성
        embedding_features = train_df[embedding_feature_names].values
        X = np.hstack([embedding_features, desc_scaled])
        # 예측해야 할 목표 변수(y)를 'Inhibition' 컬럼으로 지정
        y = train_df['Inhibition'].values

        # 특징 이름 생성 (LightGBM 실행 시 발생하는 경고 메시지를 방지하고, 나중에 특징 중요도 분석을 용이하게 함)
        desc_feature_names = [name for name, _ in Descriptors._descList]
        all_feature_names = embedding_feature_names + desc_feature_names
        # 숫자만 있던 NumPy 배열을 특징 이름이 있는 Pandas DataFrame으로 변환
        X = pd.DataFrame(X, columns=all_feature_names)

        # === 3. 하이퍼파라미터 최적화 (Optuna) ===
        print("\n3. Optuna를 사용한 하이퍼파라미터 최적화...")
        # Optuna 스터디 객체 생성 (목표: objective 함수의 점수를 '최대화(maximize)')
        study = optuna.create_study(direction='maximize', study_name='lgbm_inhibition_tuning')
        # 정의된 횟수(N_TRIALS)만큼 최적화 수행
        study.optimize(lambda trial: objective(trial, X, y), n_trials=CFG['N_TRIALS'])

        print(f"\n최적화 완료. 최고 점수: {study.best_value:.4f}")
        print("최적 파라미터:", study.best_params)

        # Optuna가 찾은 최적의 하이퍼파라미터를 기본 모델 파라미터에 업데이트
        best_params = {
            'objective': 'regression', 
            'metric': 'rmse', 
            'verbose': -1, 
            'n_jobs': -1,
            'boosting_type': 'gbdt', 
            'n_estimators': 2000
        }
        best_params.update(study.best_params)

        # === 4. 최종 모델 훈련 및 예측 (시드 앙상블) ===
        print("\n4. 시드 앙상블을 사용한 최종 모델 훈련 및 예측...")
        
        # --- 테스트 데이터에 대해서도 훈련 데이터와 '동일한' 특징 공학 과정 수행 ---
        # --- 4a. ChemBERTa 임베딩 추출 ---
        test_embeddings = get_chemberta_embeddings(test_df['Canonical_Smiles'].tolist(), CFG['CHEMBERTA_MODEL'])
        test_embedding_df = pd.DataFrame(test_embeddings, columns=embedding_feature_names, index=test_df.index)

        # --- 4b. RDKit 분자 설명자 특징 추출 ---
        test_df['descriptors'] = test_df['Canonical_Smiles'].apply(calculate_rdkit_descriptors)
        
        # 임베딩과 설명자 특징 결합
        test_df = pd.concat([test_df, test_embedding_df], axis=1)

        # 특징 추출에 성공한 유효한 테스트 데이터만 선택 (임베딩은 항상 성공한다고 가정)
        valid_test_mask = test_df['descriptors'].notna()
        
        # 특징들을 NumPy 배열로 변환
        embedding_test_features = test_df.loc[valid_test_mask, embedding_feature_names].values
        desc_test_stack = np.stack(test_df.loc[valid_test_mask, 'descriptors'].values)
        
        # 결측값 처리 (중요: 테스트 데이터의 평균이 아닌, '훈련 데이터'에서 계산한 평균(desc_mean)으로 채워야 함)
        desc_test_stack = np.nan_to_num(desc_test_stack, nan=desc_mean)
        # 정규화 (중요: 테스트 데이터로 새로 학습하는 것이 아닌, '훈련 데이터'로 학습된 스케일러(scaler)를 그대로 사용)
        desc_test_scaled = scaler.transform(desc_test_stack)
        
        # 최종 테스트 데이터셋(X_test) 생성
        X_test = np.hstack([embedding_test_features, desc_test_scaled])
        X_test = pd.DataFrame(X_test, columns=all_feature_names)

        # 시드 앙상블의 전체 예측값을 저장할 배열 초기화
        ensembled_test_preds = np.zeros(len(X_test))
        
        # 설정된 시드 목록을 하나씩 순회하며 훈련 및 예측 반복
        for seed in CFG['SEEDS']:
            print(f"--- 훈련 시작 (시드: {seed}) ---")
            seed_everything(seed)             # 현재 시드로 모든 랜덤 상태 고정
            best_params['seed'] = seed        # 모델 파라미터에도 현재 시드 설정
            
            # K-Fold 분할기 (현재 시드로 초기화하여 매번 다른 방식으로 데이터를 분할)
            kf = KFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=seed)
            # 현재 시드에서의 예측값을 저장할 배열 초기화 (폴드별 예측을 평균내기 위함)
            seed_test_preds = np.zeros(len(X_test))

            # 교차 검증 루프 (주의: 여기서는 검증 데이터를 사용하지 않고, 각 폴드를 전체 데이터로 사용하여 훈련)
            # 이는 최종 예측 시, 가능한 한 많은 데이터로 훈련된 모델을 만들기 위함. K-Fold는 데이터 분할 방식의 다양성을 위해 사용.
            for fold, (train_idx, _) in enumerate(kf.split(X, y)):
                print(f"--- 폴드 {fold+1}/{CFG['N_SPLITS']} (시드: {seed}) ---")
                X_train, y_train = X.iloc[train_idx], y[train_idx]
                
                # 최적의 하이퍼파라미터로 모델 생성
                model = lgb.LGBMRegressor(**best_params)
                # 현재 폴드의 훈련 데이터로 모델 훈련
                model.fit(X_train, y_train)
                # 테스트 데이터에 대한 예측을 수행하고, 폴드 수로 나누어 누적 (평균 계산)
                seed_test_preds += model.predict(X_test) / CFG['N_SPLITS']
            
            # 현재 시드의 예측 결과를 전체 앙상블 예측값에 더함
            ensembled_test_preds += seed_test_preds
            
        # 모든 시드의 예측값을 합한 결과를 시드의 개수로 나누어 최종 평균 예측값 계산
        final_preds = ensembled_test_preds / len(CFG['SEEDS'])
        # 안정성을 위해 최종 예측값도 0~100 사이로 클리핑
        final_preds = np.clip(final_preds, 0, 100)

        # === 5. 제출 파일 생성 ===
        print("\n5. 제출 파일 생성...")
        data_dir = Path("./data")
        sample_submission = pd.read_csv(data_dir / "sample_submission.csv")
        
        # 예측 결과를 'ID'와 'Inhibition' 컬럼을 가진 데이터프레임으로 변환
        pred_df = pd.DataFrame({'ID': test_df.loc[valid_test_mask, 'ID'], 'Inhibition': final_preds})
        
        # 대회 제출 양식(sample_submission)에 나의 예측값을 ID 기준으로 병합
        submission_df = sample_submission[['ID']].merge(pred_df, on='ID', how='left')
        # 특징 추출 실패 등으로 예측하지 못한 값이 있다면, 훈련 데이터의 전체 평균값으로 채움
        submission_df['Inhibition'] = submission_df['Inhibition'].fillna(train_df['Inhibition'].mean())
        
        # 최종 제출 파일을 'submission.csv'로 저장 (인덱스는 제외)
        submission_path = Path("submission.csv")
        submission_df.to_csv(submission_path, index=False)
        print(f"제출 파일 저장 완료: {submission_path}")
        
        # --- 예측 결과 통계 출력 ---
        print(f"\n--- 예측 결과 통계 ---")
        print(f"총 예측 수: {len(submission_df)}")
        print(f"유효한 예측 수: {len(pred_df)}")
        print(f"Inhibition 범위: {submission_df['Inhibition'].min():.2f}% ~ {submission_df['Inhibition'].max():.2f}%")
        print(f"평균 Inhibition: {submission_df['Inhibition'].mean():.2f}%")
        print(f"중앙값 Inhibition: {submission_df['Inhibition'].median():.2f}%")
    else:
        print("데이터 로드 실패. 파일 경로를 확인하세요.")
