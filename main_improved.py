import pandas as pd
import numpy as np
import os
import random
import pickle
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
from gensim.models.word2vec import Word2Vec

warnings.filterwarnings('ignore')

# 전역 설정 변수들
CFG = {
    'NBITS': 2048,      # Morgan 지문의 비트 수
    'SEED': 42,         # 재현성을 위한 랜덤 시드
    'N_SPLITS': 10,     # K-폴드 교차 검증에서 사용할 폴드 수
    'N_TRIALS': 100,    # Optuna 하이퍼파라미터 최적화 시도 횟수 (증가)
    'USE_WORD2VEC': True,  # Word2Vec 모델 사용 여부
    'WORD2VEC_DIM': 300,   # Word2Vec 벡터 차원
    'USE_ENSEMBLE': True,  # 앙상블 모델 사용
    'ENSEMBLE_MODELS': ['lgb']  # LightGBM만 사용
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
        self.models = {}
        self.scaler = RobustScaler()
        self.feature_names = None
        self.best_params = {}
        self.word2vec_model = None
        self.descriptor_names = [desc_name for desc_name, _ in Descriptors._descList]
        
        # Word2Vec 모델 로드
        if CFG['USE_WORD2VEC']:
            try:
                with open('model_300dim.pkl', 'rb') as f:
                    self.word2vec_model = pickle.load(f)
                print("✅ Word2Vec 모델 로드 성공")
            except Exception as e:
                print(f"❌ Word2Vec 모델 로드 실패: {e}")
                print("Word2Vec 없이 진행합니다.")
                CFG['USE_WORD2VEC'] = False

    def tokenize_smiles(self, smiles):
        """SMILES를 토큰으로 분리하는 함수"""
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
        """SMILES를 Word2Vec 벡터로 변환하는 함수"""
        if not CFG['USE_WORD2VEC'] or self.word2vec_model is None:
            return np.zeros(CFG['WORD2VEC_DIM'])
        
        try:
            tokens = self.tokenize_smiles(smiles)
            vectors = []
            
            for token in tokens:
                try:
                    # 다양한 방법으로 벡터 가져오기 시도
                    vec = None
                    if hasattr(self.word2vec_model.wv, 'word_vec'):
                        vec = self.word2vec_model.wv.word_vec(token)
                    elif hasattr(self.word2vec_model.wv, 'get_vector'):
                        vec = self.word2vec_model.wv.get_vector(token)
                    elif hasattr(self.word2vec_model.wv, '__getitem__'):
                        vec = self.word2vec_model.wv[token]
                    else:
                        # vocab에서 직접 접근 시도
                        if hasattr(self.word2vec_model.wv, 'vocab') and token in self.word2vec_model.wv.vocab:
                            idx = self.word2vec_model.wv.vocab[token].index
                            vec = self.word2vec_model.wv.syn0[idx]
                        elif hasattr(self.word2vec_model.wv, 'index2word'):
                            # index2word를 사용해서 찾기
                            try:
                                idx = self.word2vec_model.wv.index2word.index(token)
                                vec = self.word2vec_model.wv.syn0[idx]
                            except ValueError:
                                continue
                    
                    if vec is not None:
                        vectors.append(vec)
                        
                except (KeyError, ValueError, AttributeError):
                    # 토큰이 어휘에 없는 경우 건너뛰기
                    continue
            
            if vectors:
                # 모든 토큰 벡터의 평균 계산
                return np.mean(vectors, axis=0)
            else:
                return np.zeros(CFG['WORD2VEC_DIM'])
                
        except Exception as e:
            print(f"Word2Vec 변환 오류 ({smiles}): {e}")
            return np.zeros(CFG['WORD2VEC_DIM'])

    def get_all_descriptors(self, mol):
        """RDKit의 모든 분자 설명자를 계산하는 함수"""
        desc_dict = {}
        for name in self.descriptor_names:
            try:
                desc_func = getattr(Descriptors, name)
                desc_dict[name] = desc_func(mol)
            except:
                desc_dict[name] = 0
        return desc_dict

    def smiles_to_features(self, smiles):
        """SMILES 문자열에서 모든 특성을 추출하는 함수"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, None, None

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

        # 4. Word2Vec 벡터
        word2vec_vec = self.smiles_to_word2vec(smiles)

        return arr_morgan, arr_maccs, descriptors, word2vec_vec

    def prepare_data(self, df):
        """데이터프레임에서 모든 특성을 병렬로 추출"""
        print("분자 특성 추출 중 (Morgan, MACCS, RDKit Descriptors, Word2Vec)...")
        
        all_features = []
        for i, smiles in enumerate(df['Canonical_Smiles']):
            if i % 200 == 0:
                print(f"처리 중: {i}/{len(df)}")
            
            morgan_fp, maccs_fp, descriptors, word2vec_vec = self.smiles_to_features(smiles)
            
            if morgan_fp is None:
                # SMILES 파싱 실패 시
                morgan_fp = np.zeros(CFG['NBITS'])
                maccs_fp = np.zeros(167)
                descriptors = {name: 0 for name in self.descriptor_names}
                word2vec_vec = np.zeros(CFG['WORD2VEC_DIM'])

            all_features.append((morgan_fp, maccs_fp, list(descriptors.values()), word2vec_vec))

        # 특성별로 분리
        morgan_fps = np.array([item[0] for item in all_features])
        maccs_fps = np.array([item[1] for item in all_features])
        desc_df = pd.DataFrame([item[2] for item in all_features], columns=self.descriptor_names)
        word2vec_vecs = np.array([item[3] for item in all_features])

        # 특성 이름 저장
        morgan_names = [f'Morgan_{i}' for i in range(CFG['NBITS'])]
        maccs_names = [f'MACCS_{i}' for i in range(maccs_fps.shape[1])]
        word2vec_names = [f'Word2Vec_{i}' for i in range(CFG['WORD2VEC_DIM'])]
        
        self.feature_names = morgan_names + maccs_names + self.descriptor_names + word2vec_names
        
        # 모든 특성을 하나의 numpy 배열로 결합
        return np.hstack([morgan_fps, maccs_fps, desc_df.values, word2vec_vecs])

    def get_score(self, y_true, y_pred):
        """리더보드 평가 지표에 맞는 스코어 함수
        Score = 0.5 * (1 - min(A, 1)) + 0.5 * B
        A = Normalized RMSE = RMSE / (max(y) - min(y))
        B = Pearson Correlation Coefficient (clipped to [0, 1])
        """
        # A: Normalized RMSE
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        y_range = np.max(y_true) - np.min(y_true)
        normalized_rmse = rmse / y_range
        A = min(normalized_rmse, 1)  # 1로 클리핑
        
        # B: Pearson Correlation Coefficient (clipped to [0, 1])
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        if np.isnan(correlation):
            correlation = 0
        B = np.clip(correlation, 0, 1)
        
        # 최종 스코어 계산
        score = 0.5 * (1 - A) + 0.5 * B
        
        return score

    def objective_lgb(self, trial, X, y):
        """LightGBM Optuna 목적 함수"""
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1,
            'n_jobs': -1,
            'seed': CFG['SEED'],
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=CFG['SEED'])
        oof_preds = np.zeros(len(X))
        y_array = y.values if hasattr(y, 'values') else np.array(y)

        for train_idx, val_idx in kf.split(X, y_array):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_array[train_idx], y_array[val_idx]

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
        """앙상블 모델 훈련 및 테스트 데이터 예측"""
        print("앙상블 모델 훈련 시작...")
        
        # 각 모델별 최적화
        for model_name in CFG['ENSEMBLE_MODELS']:
            print(f"\n{model_name.upper()} 모델 최적화 중...")
            
            if model_name == 'lgb':
                study = optuna.create_study(direction='maximize', study_name=f'{model_name}_tuning')
                study.optimize(lambda trial: self.objective_lgb(trial, X_train_full, y_train_full), 
                              n_trials=CFG['N_TRIALS']//3)
                self.best_params[model_name] = study.best_params
                print(f"{model_name} 최고 스코어: {study.best_value:.4f}")
                
        # K-Fold 앙상블 훈련 및 예측
        print("\n앙상블 예측을 위한 K-폴드 훈련 중...")
        kf = KFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=CFG['SEED'])
        test_preds_ensemble = np.zeros(len(X_test_full))
        oof_preds_ensemble = np.zeros(len(X_train_full))
        
        # 각 모델별 예측 저장
        test_preds_models = {model: np.zeros(len(X_test_full)) for model in CFG['ENSEMBLE_MODELS']}
        oof_preds_models = {model: np.zeros(len(X_train_full)) for model in CFG['ENSEMBLE_MODELS']}

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full, y_train_full)):
            print(f"--- 훈련 폴드 {fold+1}/{CFG['N_SPLITS']} ---")
            X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
            y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test_full)

            # 각 모델 훈련
            fold_models = {}
            
            # LightGBM
            if 'lgb' in CFG['ENSEMBLE_MODELS']:
                lgb_params = {
                    'objective': 'regression', 'metric': 'rmse', 'verbose': -1, 'n_jobs': -1,
                    'seed': CFG['SEED'], 'boosting_type': 'gbdt'
                }
                lgb_params.update(self.best_params['lgb'])
                lgb_model = lgb.LGBMRegressor(**lgb_params)
                lgb_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)],
                             eval_metric='rmse', callbacks=[lgb.early_stopping(100, verbose=False)])
                fold_models['lgb'] = lgb_model
                
                oof_preds_models['lgb'][val_idx] = lgb_model.predict(X_val_scaled)
                test_preds_models['lgb'] += lgb_model.predict(X_test_scaled) / CFG['N_SPLITS']

            # 앙상블 예측 (가중 평균)
            weights = {'lgb': 1.0}  # LightGBM만 사용
            
            for model_name in CFG['ENSEMBLE_MODELS']:
                if model_name in fold_models:
                    oof_preds_ensemble[val_idx] += weights[model_name] * oof_preds_models[model_name][val_idx]
                    test_preds_ensemble += weights[model_name] * test_preds_models[model_name] / CFG['N_SPLITS']

        # 최종 성능 평가
        final_score = self.get_score(y_train_full, oof_preds_ensemble)
        print(f"\n앙상블 모델 최종 스코어: {final_score:.4f}")
        
        # 개별 모델 성능도 출력
        for model_name in CFG['ENSEMBLE_MODELS']:
            if model_name in oof_preds_models:
                model_score = self.get_score(y_train_full, oof_preds_models[model_name])
                print(f"{model_name.upper()} 개별 스코어: {model_score:.4f}")
        
        self.plot_results(y_train_full, oof_preds_ensemble, "앙상블 모델 예측 결과")

        return test_preds_ensemble

    def plot_results(self, y_true, y_pred, title="예측 vs 실제"):
        """결과 시각화"""
        plt.figure(figsize=(14, 8))
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        
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
        
        # 특성 중요도 (LightGBM 기준)
        plt.subplot(2, 1, 2)
        if self.feature_names and 'lgb' in self.models:
            importances = self.models['lgb'].feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(20)
            
            sns.barplot(x='importance', y='feature', data=feature_importance_df)
            plt.title('상위 20개 특성 중요도 (LightGBM 기준)')
        
        plt.tight_layout()
        plt.savefig('model_results_ensemble.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    print("🚀 CYP3A4 효소 저해 예측 모델 (앙상블 + Word2Vec) 시작 🚀")
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
        print(f"- Morgan Fingerprint: {CFG['NBITS']}")
        print(f"- MACCS Keys: 167")
        print(f"- RDKit Descriptors: {len(predictor.descriptor_names)}")
        print(f"- Word2Vec Vectors: {CFG['WORD2VEC_DIM']}")
        
        # 모델 훈련 및 예측
        test_preds = predictor.train_and_predict(X_train_full, y_train_full, X_test_full)
        
        # 제출 파일 생성
        submission = sample_submission.copy()
        submission['Inhibition'] = test_preds
        submission['Inhibition'] = np.clip(submission['Inhibition'], 0, 100)
        
        submission.to_csv('submission_ensemble.csv', index=False)
        print(f"\n✅ 제출 파일이 'submission_ensemble.csv'로 저장되었습니다.")
        
        print("\n예측 결과 요약:")
        print(submission['Inhibition'].describe())

    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 