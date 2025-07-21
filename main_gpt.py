import pandas as pd
import numpy as np
import os
import random
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 시드 고정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# 2. 데이터 로딩
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sample_submission = pd.read_csv('data/sample_submission.csv')

# 3. Word2Vec 모델 로딩 (gensim.models.Word2Vec)
with open('model_300dim.pkl', 'rb') as f:
    w2v_model = pickle.load(f)

def smiles_to_emb_matrix(smiles_list, w2v_model):
    zero_vec = np.zeros(w2v_model.vector_size)
    embs = []
    for s in smiles_list:
        try:
            # get_vector 메서드를 사용하여 더 안전하게 벡터 추출
            vec = w2v_model.wv.get_vector(s)
        except (KeyError, AttributeError):
            vec = zero_vec
        embs.append(vec)
    return np.vstack(embs)

# 4. 특성 생성 (300차원 임베딩)
X_train = smiles_to_emb_matrix(train['Canonical_Smiles'], w2v_model)
X_test = smiles_to_emb_matrix(test['Canonical_Smiles'], w2v_model)
y_train = train['Inhibition']

# 특성 이름 설정
feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

# 5. 스케일링
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 특성 이름을 DataFrame으로 변환하여 설정
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)

# 6. 커스텀 스코어
def custom_score(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    nrmse = rmse / 100  # Inhibition 값 0~100 기준
    A = 1 - min(nrmse, 1)
    B = r2_score(y_true, y_pred)
    return 0.4 * A + 0.6 * B

# 7. Optuna 튜닝 함수
def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'random_state': SEED,
        'n_jobs': -1,
        'verbose': -1,  # verbose를 params에 포함
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'n_estimators': 1000
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X_train_scaled_df))
    for train_idx, val_idx in kf.split(X_train_scaled_df, y_train):
        X_tr, X_val = X_train_scaled_df.iloc[train_idx], X_train_scaled_df.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(100, verbose=False)])  # verbose 파라미터 제거
        oof_preds[val_idx] = model.predict(X_val)
    score = custom_score(y_train, oof_preds)
    return score

print("Optuna 탐색 시작...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print("Optuna Best Score:", study.best_value)
print("Optuna Best Params:", study.best_params)

# 8. 최적 파라미터로 앙상블 훈련 및 예측
final_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'random_state': SEED,
    'n_jobs': -1,
    'verbose': -1,  # verbose를 params에 포함
    'n_estimators': 1000,
}
final_params.update(study.best_params)

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(X_train_scaled_df))
test_preds = np.zeros(len(X_test_scaled_df))
models = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled_df, y_train)):
    print(f'Fold {fold+1}/5')
    X_tr, X_val = X_train_scaled_df.iloc[train_idx], X_train_scaled_df.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    model = lgb.LGBMRegressor(**final_params)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)])  # verbose 파라미터 제거
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test_scaled_df) / 5
    models.append(model)

# 9. OOF 평가
print("\n[KFold OOF Performance]")
print(f"R2: {r2_score(y_train, oof_preds):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_train, oof_preds)):.4f}")
print(f"MAE: {mean_absolute_error(y_train, oof_preds):.4f}")
print(f"Custom Score: {custom_score(y_train, oof_preds):.4f}")

# 10. 결과 분석/시각화
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
sns.scatterplot(x=y_train, y=oof_preds, alpha=0.5)
plt.xlabel('실제 값')
plt.ylabel('예측 값')
plt.title('OOF: 실제 vs 예측')
plt.subplot(1,2,2)
sns.histplot(y_train-oof_preds, bins=30, kde=True)
plt.title('잔차 분포')
plt.tight_layout()
plt.show()

# 11. 제출 파일 저장
submission = sample_submission.copy()
submission['Inhibition'] = test_preds
submission['Inhibition'] = np.clip(submission['Inhibition'], 0, 100)
submission.to_csv('submission_embedding.csv', index=False)
print("\n제출 파일 저장 완료: submission_embedding.csv")
print("예측 통계:")
print(submission['Inhibition'].describe())
