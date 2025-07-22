import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import CalcNumAromaticRings, CalcNumAliphaticRings
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class SimpleBestModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        
        # 간단하고 효과적인 모델들만 선택
        self.base_models = {
            'rf': RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1
            ),
            'extra': ExtraTreesRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1
            ),
            'xgb': xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1
            ),
            'lgb': lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1
            )
        }
    
    def extract_core_features(self, smiles_list):
        """핵심적인 분자 특성만 추출"""
        
        # 핵심 분자 서술자만 선별
        core_descriptors = [
            ('MolWt', Descriptors.MolWt),
            ('LogP', Descriptors.MolLogP),
            ('NumHDonors', Descriptors.NumHDonors),
            ('NumHAcceptors', Descriptors.NumHAcceptors),
            ('TPSA', Descriptors.TPSA),
            ('NumRotatableBonds', Descriptors.NumRotatableBonds),
            ('NumAromaticRings', CalcNumAromaticRings),
            ('NumAliphaticRings', CalcNumAliphaticRings),
            ('FractionCSP3', Descriptors.FractionCSP3),
            ('NumHeteroatoms', Descriptors.NumHeteroatoms),
            ('BertzCT', Descriptors.BertzCT),
            ('Chi0v', Descriptors.Chi0v),
            ('Chi1v', Descriptors.Chi1v),
            ('Kappa1', Descriptors.Kappa1),
            ('Kappa2', Descriptors.Kappa2),
            ('LabuteASA', Descriptors.LabuteASA),
            ('PEOE_VSA1', Descriptors.PEOE_VSA1),
            ('PEOE_VSA2', Descriptors.PEOE_VSA2),
            ('SMR_VSA1', Descriptors.SMR_VSA1),
            ('SlogP_VSA1', Descriptors.SlogP_VSA1)
        ]
        
        # Morgan 지문 (적당한 크기)
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
        
        all_features = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                # 잘못된 SMILES의 경우 기본값
                mol_features = [0] * len(core_descriptors)
                morgan_fp = [0] * 512
            else:
                # 분자 서술자 계산
                mol_features = []
                for name, func in core_descriptors:
                    try:
                        value = func(mol)
                        if np.isnan(value) or np.isinf(value):
                            value = 0
                        mol_features.append(value)
                    except:
                        mol_features.append(0)
                
                # Morgan 지문 계산
                try:
                    fp = morgan_gen.GetFingerprint(mol)
                    morgan_fp = list(fp.ToBitString())
                    morgan_fp = [int(bit) for bit in morgan_fp]
                except:
                    morgan_fp = [0] * 512
            
            # 특성 결합
            combined_features = mol_features + morgan_fp
            all_features.append(combined_features)
        
        return np.array(all_features)
    
    def fit(self, X_train, y_train):
        """간단한 학습 방법"""
        print("핵심 분자 특성 추출 중...")
        X_features = self.extract_core_features(X_train)
        
        print(f"학습 데이터 크기: {X_features.shape}")
        print(f"타겟 범위: {y_train.min():.2f} ~ {y_train.max():.2f}")
        
        # 스케일링
        X_scaled = self.scaler.fit_transform(X_features)
        
        # 각 모델 학습 및 평가
        cv_scores = {}
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for model_name, model in self.base_models.items():
            print(f"\n{model_name.upper()} 모델 학습 중...")
            
            # Cross-validation 점수 계산
            cv_score = cross_val_score(
                model, X_scaled, y_train, 
                cv=kf,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            cv_scores[model_name] = -cv_score.mean()
            print(f"{model_name} CV RMSE: {np.sqrt(cv_scores[model_name]):.4f} (+/- {np.sqrt(cv_score.std()):.4f})")
            
            # 전체 데이터로 모델 학습
            model.fit(X_scaled, y_train)
            self.models[model_name] = model
        
        # 성능 기반 가중치 계산
        weights = []
        for model_name in self.base_models.keys():
            weight = 1 / (cv_scores[model_name] + 1e-8)
            weights.append(weight)
        
        total_weight = sum(weights)
        self.model_weights = {name: w/total_weight for name, w in zip(self.base_models.keys(), weights)}
        
        print("\n모델 가중치:")
        for name, weight in self.model_weights.items():
            print(f"{name}: {weight:.4f}")
        
        return self
    
    def predict(self, X_test):
        """간단한 앙상블 예측"""
        print("테스트 데이터 특성 추출 중...")
        X_features = self.extract_core_features(X_test)
        X_scaled = self.scaler.transform(X_features)
        
        # 각 모델의 예측값 수집
        predictions = []
        weights = []
        
        for model_name, model in self.models.items():
            pred = model.predict(X_scaled)
            predictions.append(pred)
            weights.append(self.model_weights[model_name])
        
        # 가중 평균으로 최종 예측
        final_pred = np.average(predictions, axis=0, weights=weights)
        
        # 간단한 후처리
        final_pred = np.clip(final_pred, 0, 100)
        
        return final_pred

def main():
    print("=== 간단하고 효과적인 CYP3A4 예측 모델 ===")
    
    # 데이터 로드
    print("데이터 로딩 중...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    print(f"훈련 데이터: {len(train_df)}개")
    print(f"테스트 데이터: {len(test_df)}개")
    print(f"Inhibition 통계: 평균={train_df['Inhibition'].mean():.2f}, 표준편차={train_df['Inhibition'].std():.2f}")
    print(f"Inhibition 범위: {train_df['Inhibition'].min():.2f} ~ {train_df['Inhibition'].max():.2f}")
    
    # 데이터 준비 (모든 데이터 사용 - 0값 포함)
    X_train = train_df['Canonical_Smiles'].values
    y_train = train_df['Inhibition'].values
    X_test = test_df['Canonical_Smiles'].values
    
    # 모델 학습
    print("\n=== 모델 학습 시작 ===")
    model = SimpleBestModel(random_state=42)
    model.fit(X_train, y_train)
    
    # 예측
    print("\n=== 예측 시작 ===")
    predictions = model.predict(X_test)
    
    # 결과 저장
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'Inhibition': predictions
    })
    
    output_file = 'submission_simple_best.csv'
    submission.to_csv(output_file, index=False)
    
    print(f"\n=== 완료 ===")
    print(f"예측 결과 저장: {output_file}")
    print(f"예측값 통계:")
    print(f"  평균: {predictions.mean():.4f}")
    print(f"  표준편차: {predictions.std():.4f}")
    print(f"  최솟값: {predictions.min():.4f}")
    print(f"  최댓값: {predictions.max():.4f}")
    print(f"  중간값: {np.median(predictions):.4f}")

if __name__ == "__main__":
    main() 