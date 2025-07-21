import pandas as pd
import numpy as np
import os
import random
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdMolDescriptors
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 전역 설정 변수들
CFG = {
    'NBITS': 2048,      # Morgan 지문의 비트 수
    'SEED': 42,         # 재현성을 위한 랜덤 시드
    'N_SPLITS': 10,     # K-폴드 교차 검증에서 사용할 폴드 수 (늘려서 안정성 확보)
    'N_TRIALS': 50      # Optuna 하이퍼파라미터 최적화 시도 횟수 (탐색 기회 증가)
}

def seed_everything(seed):
    """모든 랜덤 시드를 설정하여 실험의 재현성을 보장하는 함수"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# 전역 시드 설정
seed_everything(CFG['SEED'])

class CYP3A4InhibitionPredictor:
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler() # 이상치에 강한 RobustScaler 유지
        self.feature_names = None
        self.best_params = None
        # RDKit에서 계산 가능한 모든 설명자 리스트
        self.descriptor_names = [desc_name for desc_name, _ in Descriptors._descList]

    def get_all_descriptors(self, mol):
        """RDKit의 모든 분자 설명자를 계산하는 함수"""
        desc_dict = {}
        for name in self.descriptor_names:
            try:
                desc_func = getattr(Descriptors, name)
                desc_dict[name] = desc_func(mol)
            except:
                # 계산 실패 시 0으로 채움
                desc_dict[name] = 0
        return desc_dict

    def smiles_to_features(self, smiles):
        """SMILES 문자열에서 Morgan 지문, MACCS 키, 분자 설명자를 추출하는 함수"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, None

        # 1. Morgan Fingerprint
        fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=CFG['NBITS'])
        arr_morgan = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp_morgan, arr_morgan)

        # 2. MACCS Keys
        fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
        arr_maccs = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp_maccs, arr_maccs)

        # 3. RDKit Descriptors
        descriptors = self.get_all_descriptors(mol)

        return arr_morgan, arr_maccs, descriptors

    def prepare_data(self, df):
        """데이터프레임에서 모든 특성을 병렬로 추출"""
        print("분자 특성 추출 중 (Morgan, MACCS, RDKit Descriptors)...")
        
        all_features = []
        for i, smiles in enumerate(df['Canonical_Smiles']):
            if i % 200 == 0:
                print(f"처리 중: {i}/{len(df)}")
            
            morgan_fp, maccs_fp, descriptors = self.smiles_to_features(smiles)
            
            if morgan_fp is None:
                # SMILES 파싱 실패 시
                morgan_fp = np.zeros(CFG['NBITS'])
                maccs_fp = np.zeros(167) # MACCS 키는 167 비트
                descriptors = {name: 0 for name in self.descriptor_names}

            all_features.append((morgan_fp, maccs_fp, list(descriptors.values())))

        # 특성별로 분리
        morgan_fps = np.array([item[0] for item in all_features])
        maccs_fps = np.array([item[1] for item in all_features])
        desc_df = pd.DataFrame([item[2] for item in all_features], columns=self.descriptor_names)

        # 특성 이름 저장 (나중에 특성 중요도 시각화를 위해)
        morgan_names = [f'Morgan_{i}' for i in range(CFG['NBITS'])]
        maccs_names = [f'MACCS_{i}' for i in range(maccs_fps.shape[1])]
        self.feature_names = morgan_names + maccs_names + self.descriptor_names
        
        # 모든 특성을 하나의 numpy 배열로 결합
        return np.hstack([morgan_fps, maccs_fps, desc_df.values])

    def get_score(self, y_true, y_pred):
        """커스텀 스코어 함수"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        # y_true의 범위가 0~100으로 고정되어 있으므로 분모를 상수로 사용 가능
        nrmse = rmse / (100 - 0) 
        A = 1 - min(nrmse, 1)
        B = r2_score(y_true, y_pred)
        score = 0.4 * A + 0.6 * B
        return score

    def objective(self, trial, X, y):
        """Optuna 목적 함수"""
        params = {
            'objective': 'regression_l1', # MAE objective, 이상치에 더 강함
            'metric': 'rmse',
            'verbose': -1,
            'n_jobs': -1,
            'seed': CFG['SEED'],
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
            'num_leaves': trial.suggest_int('num_leaves', 20, 60),
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        }

        kf = KFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=CFG['SEED'])
        oof_preds = np.zeros(len(X))
        y_array = y.values if hasattr(y, 'values') else np.array(y)

        for train_idx, val_idx in kf.split(X, y_array):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_array[train_idx], y_array[val_idx]

            # 데이터 스케일링
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model = lgb.LGBMRegressor(**params)
            model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)],
                      eval_metric='rmse', callbacks=[lgb.early_stopping(100, verbose=False)])
            
            oof_preds[val_idx] = model.predict(X_val_scaled)

        score = self.get_score(y_array, oof_preds)
        return score

    def train_and_predict(self, X_train_full, y_train_full, X_test_full):
        """모델 훈련 및 테스트 데이터 예측"""
        print("하이퍼파라미터 최적화 중...")
        study = optuna.create_study(direction='maximize', study_name='lgbm_tuning')
        study.optimize(lambda trial: self.objective(trial, X_train_full, y_train_full), n_trials=CFG['N_TRIALS'])

        print(f"최적화 완료. 최고 스코어: {study.best_value:.4f}")
        self.best_params = study.best_params
        print("최적 파라미터:", self.best_params)

        # K-Fold 앙상블 훈련 및 예측
        print("\n앙상블 예측을 위한 K-폴드 훈련 중...")
        kf = KFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=CFG['SEED'])
        test_preds = np.zeros(len(X_test_full))
        oof_preds = np.zeros(len(X_train_full))
        
        final_model_params = {
            'objective': 'regression_l1',
            'metric': 'rmse', 'verbose': -1, 'n_jobs': -1,
            'seed': CFG['SEED'], 'boosting_type': 'gbdt'
        }
        final_model_params.update(self.best_params)

        models = [] # 특성 중요도 시각화를 위해 모델 저장

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full, y_train_full)):
            print(f"--- 훈련 폴드 {fold+1}/{CFG['N_SPLITS']} ---")
            X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
            y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

            # ❗ 중요: 폴드마다 스케일러를 새로 fit_transform 해야 데이터 누수를 막을 수 있음
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test_full)

            model = lgb.LGBMRegressor(**final_model_params)
            model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)],
                      eval_metric='rmse', callbacks=[lgb.early_stopping(100, verbose=False)])

            oof_preds[val_idx] = model.predict(X_val_scaled)
            test_preds += model.predict(X_test_scaled) / CFG['N_SPLITS']
            models.append(model)
        
        self.model = models # 마지막 폴드의 모델을 대표로 저장 (혹은 평균 모델)
        
        # OOF 예측 결과로 전체 훈련 데이터에 대한 성능 평가
        final_score = self.get_score(y_train_full, oof_preds)
        print(f"\nK-Fold OOF Custom Score: {final_score:.4f}")
        
        self.plot_results(y_train_full, oof_preds, "K-Fold OOF Predictions")

        return test_preds

    def plot_results(self, y_true, y_pred, title="예측 vs 실제"):
        """결과 시각화"""
        plt.figure(figsize=(14, 8))
        plt.rcParams['font.family'] = 'Malgun Gothic' # 한글 폰트 설정
        plt.rcParams['axes.unicode_minus'] = False # 마이너스 깨짐 방지
        
        # 산점도
        plt.subplot(2, 2, 1)
        sns.regplot(x=y_true, y=y_pred, scatter_kws={'alpha':0.4, 'color': 'royalblue'}, line_kws={'color':'red', 'linestyle':'--'})
        plt.xlabel('실제 값 (Inhibition %)')
        plt.ylabel('예측 값 (Inhibition %)')
        plt.title(f'{title}\nR² = {r2_score(y_true, y_pred):.4f}')
        
        # 잔차 플롯
        plt.subplot(2, 2, 2)
        residuals = y_true - y_pred
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, color='forestgreen')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('예측 값')
        plt.ylabel('잔차')
        plt.title('잔차 플롯')
        
        # 특성 중요도
        plt.subplot(2, 1, 2)
        if self.model and self.feature_names:
            # 모든 폴드의 특성 중요도를 평균내어 사용
            importances = np.mean([m.feature_importances_ for m in self.model], axis=0)
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(20)
            
            sns.barplot(x='importance', y='feature', data=feature_importance_df)
            plt.title('상위 20개 특성 중요도 (K-Fold 평균)')
        
        plt.tight_layout()
        plt.savefig('model_results_improved.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    print("🚀 CYP3A4 효소 저해 예측 모델 개선 시작 🚀")
    print("=" * 70)
    
    try:
        # 데이터 로드
        print("데이터 로드 중...")
        train_df = pd.read_csv('data/train.csv')
        test_df = pd.read_csv('data/test.csv')
        sample_submission = pd.read_csv('data/sample_submission.csv')
        
        # 모델 초기화
        predictor = CYP3A4InhibitionPredictor()
        
        # 특성 추출
        X_train_full = predictor.prepare_data(train_df)
        X_test_full = predictor.prepare_data(test_df)
        y_train_full = train_df['Inhibition']
        
        print(f"\n최종 특성 수: {X_train_full.shape[1]}")
        
        # 모델 훈련 및 예측
        test_preds = predictor.train_and_predict(X_train_full, y_train_full, X_test_full)
        
        # 제출 파일 생성
        submission = sample_submission.copy()
        submission['Inhibition'] = test_preds
        submission['Inhibition'] = np.clip(submission['Inhibition'], 0, 100)
        
        submission.to_csv('submission_improved.csv', index=False)
        print(f"\n✅ 제출 파일이 'submission_improved.csv'로 저장되었습니다.")
        
        print("\n예측 결과 요약:")
        print(submission['Inhibition'].describe())

    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()