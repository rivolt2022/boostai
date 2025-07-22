import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski, rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import CalcNumAromaticRings, CalcNumAliphaticRings
from rdkit.Chem import rdMolDescriptors as rdMD
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
# import catboost as cb  # 제거 - 너무 느림
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class AdvancedMolecularModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scalers = {}
        self.models = {}
        self.meta_model = None
        self.feature_names = []
        self.best_threshold = None
        
        # 더 다양하고 정교한 모델들
        self.base_models = {
            'rf_deep': RandomForestRegressor(
                n_estimators=1000,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=random_state,
                n_jobs=-1
            ),
            'extra_deep': ExtraTreesRegressor(
                n_estimators=1000,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=random_state,
                n_jobs=-1
            ),
            'xgb_tuned': xgb.XGBRegressor(
                n_estimators=1000,
                max_depth=10,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=random_state,
                n_jobs=-1
            ),
            'lgb_tuned': lgb.LGBMRegressor(
                n_estimators=1000,
                max_depth=10,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1
            ),

            'gbm': GradientBoostingRegressor(
                n_estimators=1000,
                max_depth=8,
                learning_rate=0.03,
                subsample=0.85,
                random_state=random_state
            ),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(512, 256, 128),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=random_state
            )
        }
    
    def extract_advanced_molecular_features(self, smiles_list):
        """고급 분자 특성 추출 - 더 많은 descriptor와 도메인 특화 특성"""
        
        # 기본 분자 서술자 (확장)
        descriptor_funcs = [
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
            ('Chi2v', Descriptors.Chi2v),
            ('Chi3v', Descriptors.Chi3v),
            ('Chi4v', Descriptors.Chi4v),
            ('Kappa1', Descriptors.Kappa1),
            ('Kappa2', Descriptors.Kappa2),
            ('Kappa3', Descriptors.Kappa3),
            ('LabuteASA', Descriptors.LabuteASA),
            ('PEOE_VSA1', Descriptors.PEOE_VSA1),
            ('PEOE_VSA2', Descriptors.PEOE_VSA2),
            ('PEOE_VSA3', Descriptors.PEOE_VSA3),
            ('SMR_VSA1', Descriptors.SMR_VSA1),
            ('SMR_VSA2', Descriptors.SMR_VSA2),
            ('SMR_VSA3', Descriptors.SMR_VSA3),
            ('SlogP_VSA1', Descriptors.SlogP_VSA1),
            ('SlogP_VSA2', Descriptors.SlogP_VSA2),
            ('SlogP_VSA3', Descriptors.SlogP_VSA3),
            ('EState_VSA1', Descriptors.EState_VSA1),
            ('EState_VSA2', Descriptors.EState_VSA2),
            ('EState_VSA3', Descriptors.EState_VSA3),
            ('VSA_EState1', Descriptors.VSA_EState1),
            ('VSA_EState2', Descriptors.VSA_EState2),
            ('VSA_EState3', Descriptors.VSA_EState3),
            # 추가 고급 descriptor들
            ('MaxEStateIndex', Descriptors.MaxEStateIndex),
            ('MinEStateIndex', Descriptors.MinEStateIndex),
            ('MaxAbsEStateIndex', Descriptors.MaxAbsEStateIndex),
            ('MinAbsEStateIndex', Descriptors.MinAbsEStateIndex),
            ('MolMR', Descriptors.MolMR),
            ('HallKierAlpha', Descriptors.HallKierAlpha),
            ('BalabanJ', Descriptors.BalabanJ),
            ('Ipc', Descriptors.Ipc),
            ('NumSaturatedCarbocycles', Descriptors.NumSaturatedCarbocycles),
            ('NumSaturatedHeterocycles', Descriptors.NumSaturatedHeterocycles),
            ('NumSaturatedRings', Descriptors.NumSaturatedRings),
            ('RingCount', Descriptors.RingCount),
            ('FpDensityMorgan1', Descriptors.FpDensityMorgan1),
            ('FpDensityMorgan2', Descriptors.FpDensityMorgan2),
            ('FpDensityMorgan3', Descriptors.FpDensityMorgan3)
        ]
        
        # 다양한 지문 생성기들
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
        rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)
        atom_pair_gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
        
        all_features = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                # 잘못된 SMILES의 경우 기본값으로 채움
                mol_features = [0] * len(descriptor_funcs)
                morgan_fp = [0] * 2048
                rdkit_fp = [0] * 2048
                atom_pair_fp = [0] * 2048
                custom_features = [0] * 10  # 사용자 정의 특성들
            else:
                # 분자 서술자 계산
                mol_features = []
                for name, func in descriptor_funcs:
                    try:
                        value = func(mol)
                        if np.isnan(value) or np.isinf(value):
                            value = 0
                        mol_features.append(value)
                    except:
                        mol_features.append(0)
                
                # 다양한 지문 계산
                try:
                    morgan_fp = list(morgan_gen.GetFingerprint(mol).ToBitString())
                    morgan_fp = [int(bit) for bit in morgan_fp]
                except:
                    morgan_fp = [0] * 2048
                
                try:
                    rdkit_fp = list(rdkit_gen.GetFingerprint(mol).ToBitString())
                    rdkit_fp = [int(bit) for bit in rdkit_fp]
                except:
                    rdkit_fp = [0] * 2048
                
                try:
                    atom_pair_fp = list(atom_pair_gen.GetFingerprint(mol).ToBitString())
                    atom_pair_fp = [int(bit) for bit in atom_pair_fp]
                except:
                    atom_pair_fp = [0] * 2048
                
                # 사용자 정의 특성들 (CYP3A4 관련)
                custom_features = []
                try:
                    # 분자량 관련 특성
                    mw = Descriptors.MolWt(mol)
                    custom_features.append(1 if 300 <= mw <= 600 else 0)  # 적정 분자량 범위
                    
                    # LogP 관련 특성
                    logp = Descriptors.MolLogP(mol)
                    custom_features.append(1 if 1 <= logp <= 5 else 0)  # 적정 지질친화성
                    
                    # 방향족 고리 비율
                    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
                    total_atoms = mol.GetNumAtoms()
                    aromatic_ratio = aromatic_atoms / total_atoms if total_atoms > 0 else 0
                    custom_features.append(aromatic_ratio)
                    
                    # 헤테로원자 비율
                    hetero_ratio = Descriptors.NumHeteroatoms(mol) / total_atoms if total_atoms > 0 else 0
                    custom_features.append(hetero_ratio)
                    
                    # TPSA/MW 비율
                    tpsa = Descriptors.TPSA(mol)
                    custom_features.append(tpsa / mw if mw > 0 else 0)
                    
                    # 회전 가능한 결합 비율
                    rotatable_ratio = Descriptors.NumRotatableBonds(mol) / mol.GetNumBonds() if mol.GetNumBonds() > 0 else 0
                    custom_features.append(rotatable_ratio)
                    
                    # 특정 원소 존재 여부
                    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
                    custom_features.append(1 if 'N' in atom_symbols else 0)  # 질소
                    custom_features.append(1 if 'O' in atom_symbols else 0)  # 산소
                    custom_features.append(1 if 'S' in atom_symbols else 0)  # 황
                    custom_features.append(1 if any(symbol in ['F', 'Cl', 'Br', 'I'] for symbol in atom_symbols) else 0)  # 할로겐
                    
                except:
                    custom_features = [0] * 10
            
            # 모든 특성 결합
            combined_features = mol_features + morgan_fp + rdkit_fp + atom_pair_fp + custom_features
            all_features.append(combined_features)
        
        # 특성 이름 생성
        if not self.feature_names:
            self.feature_names = ([name for name, _ in descriptor_funcs] + 
                                [f'Morgan_{i}' for i in range(2048)] +
                                [f'RDKit_{i}' for i in range(2048)] +
                                [f'AtomPair_{i}' for i in range(2048)] +
                                ['MW_Range', 'LogP_Range', 'Aromatic_Ratio', 'Hetero_Ratio', 
                                 'TPSA_MW_Ratio', 'Rotatable_Ratio', 'Has_N', 'Has_O', 'Has_S', 'Has_Halogen'])
        
        return np.array(all_features)
    
    def optimize_threshold(self, y_true, y_pred):
        """최적 임계값 찾기 (후처리용)"""
        def objective(threshold):
            adjusted_pred = y_pred * threshold
            rmse = np.sqrt(mean_squared_error(y_true, adjusted_pred))
            return rmse
        
        result = minimize(objective, x0=1.0, bounds=[(0.5, 2.0)], method='L-BFGS-B')
        return result.x[0]
    
    def remove_outliers_advanced(self, X, y, method='isolation_forest'):
        """고급 특이값 제거"""
        if method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=self.random_state)
            outlier_mask = iso_forest.fit_predict(X) == 1
        elif method == 'local_outlier':
            from sklearn.neighbors import LocalOutlierFactor
            lof = LocalOutlierFactor(contamination=0.1)
            outlier_mask = lof.fit_predict(X) == 1
        else:  # IQR method
            Q1 = np.percentile(y, 25)
            Q3 = np.percentile(y, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (y >= lower_bound) & (y <= upper_bound)
        
        return X[outlier_mask], y[outlier_mask]
    
    def create_bins(self, y, n_bins=10):
        """더 세밀한 구간 분할"""
        return pd.cut(y, bins=n_bins, labels=False)
    
    def fit(self, X_train, y_train, use_stacking=True):
        """고급 학습 방법"""
        print("고급 분자 특성 추출 중...")
        X_features = self.extract_advanced_molecular_features(X_train)
        
        # 고급 특이값 제거
        print("고급 특이값 제거 중...")
        X_features, y_train = self.remove_outliers_advanced(X_features, y_train, method='isolation_forest')
        
        # 타겟 변환 (Box-Cox 또는 Yeo-Johnson)
        print("타겟 변환 적용 중...")
        self.target_transformer = PowerTransformer(method='yeo-johnson')
        y_transformed = self.target_transformer.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        # 타겟 변수 구간화 (더 세밀하게)
        y_bins = self.create_bins(y_train, n_bins=10)
        
        print(f"학습 데이터 크기: {X_features.shape}")
        print(f"타겟 범위: {y_train.min():.2f} ~ {y_train.max():.2f}")
        
        # 각 모델별로 다른 스케일러와 전처리 사용
        scaler_types = {
            'rf_deep': StandardScaler(),
            'extra_deep': StandardScaler(),
            'xgb_tuned': RobustScaler(),
            'lgb_tuned': RobustScaler(),
            'gbm': StandardScaler(),
            'ridge': StandardScaler(),
            'lasso': StandardScaler(),
            'elastic': StandardScaler(),
            'mlp': PowerTransformer()
        }
        
        # Stacking을 위한 준비
        if use_stacking:
            # 1차 모델들의 예측값을 저장할 배열
            stacking_features = np.zeros((len(X_features), len(self.base_models)))
            
        # Cross-validation으로 각 모델 평가 및 학습
        cv_scores = {}
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state)
        
        for idx, (model_name, model) in enumerate(self.base_models.items()):
            print(f"\n{model_name.upper()} 모델 학습 중...")
            
            # 스케일러 적용
            scaler = scaler_types[model_name]
            X_scaled = scaler.fit_transform(X_features)
            self.scalers[model_name] = scaler
            
            # Cross-validation으로 stacking 특성 생성
            if use_stacking:
                cv_preds = np.zeros(len(X_features))
                for train_idx, val_idx in skf.split(X_scaled, y_bins):
                    X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
                    y_train_fold, y_val_fold = y_transformed[train_idx], y_transformed[val_idx]
                    
                    model_copy = type(model)(**model.get_params())
                    model_copy.fit(X_train_fold, y_train_fold)
                    cv_preds[val_idx] = model_copy.predict(X_val_fold)
                
                stacking_features[:, idx] = cv_preds
            
            # CV 점수 계산
            cv_score = cross_val_score(
                model, X_scaled, y_transformed, 
                cv=skf.split(X_scaled, y_bins),
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            cv_scores[model_name] = -cv_score.mean()
            print(f"{model_name} CV RMSE: {np.sqrt(cv_scores[model_name]):.4f} (+/- {np.sqrt(cv_score.std()):.4f})")
            
            # 전체 데이터로 모델 학습
            model.fit(X_scaled, y_transformed)
            self.models[model_name] = model
        
        # Stacking 메타 모델 학습
        if use_stacking:
            print("\nStacking 메타 모델 학습 중...")
            self.meta_model = Ridge(alpha=1.0)
            self.meta_model.fit(stacking_features, y_transformed)
            self.use_stacking = True
        else:
            self.use_stacking = False
        
        # 모델 가중치 계산
        weights = []
        for model_name in self.base_models.keys():
            weight = 1 / (cv_scores[model_name] + 1e-8)
            weights.append(weight)
        
        total_weight = sum(weights)
        self.model_weights = {name: w/total_weight for name, w in zip(self.base_models.keys(), weights)}
        
        # 최적 후처리 임계값 찾기
        print("\n최적 후처리 임계값 계산 중...")
        if use_stacking:
            meta_pred = self.meta_model.predict(stacking_features)
        else:
            ensemble_pred = np.average([self.models[name].predict(self.scalers[name].transform(X_features)) 
                                      for name in self.models.keys()], 
                                     weights=list(self.model_weights.values()), axis=0)
            meta_pred = ensemble_pred
        
        # 타겟 역변환
        meta_pred_original = self.target_transformer.inverse_transform(meta_pred.reshape(-1, 1)).ravel()
        self.best_threshold = self.optimize_threshold(y_train, meta_pred_original)
        
        print(f"\n최적 임계값: {self.best_threshold:.4f}")
        print("\n모델 가중치:")
        for name, weight in self.model_weights.items():
            print(f"{name}: {weight:.4f}")
        
        return self
    
    def predict(self, X_test):
        """고급 예측"""
        print("테스트 데이터 고급 특성 추출 중...")
        X_features = self.extract_advanced_molecular_features(X_test)
        
        if self.use_stacking:
            # Stacking 예측
            stacking_features = np.zeros((len(X_features), len(self.base_models)))
            
            for idx, (model_name, model) in enumerate(self.models.items()):
                X_scaled = self.scalers[model_name].transform(X_features)
                pred = model.predict(X_scaled)
                stacking_features[:, idx] = pred
            
            # 메타 모델로 최종 예측
            final_pred = self.meta_model.predict(stacking_features)
        else:
            # 일반 앙상블 예측
            predictions = []
            weights = []
            
            for model_name, model in self.models.items():
                X_scaled = self.scalers[model_name].transform(X_features)
                pred = model.predict(X_scaled)
                predictions.append(pred)
                weights.append(self.model_weights[model_name])
            
            final_pred = np.average(predictions, axis=0, weights=weights)
        
        # 타겟 역변환
        final_pred = self.target_transformer.inverse_transform(final_pred.reshape(-1, 1)).ravel()
        
        # 최적 임계값 적용
        final_pred = final_pred * self.best_threshold
        
        # 최종 후처리
        final_pred = np.clip(final_pred, 0, 100)
        
        # 추가 스무딩 (극값 완화)
        final_pred = np.where(final_pred > 80, final_pred * 0.95, final_pred)
        final_pred = np.where(final_pred < 5, final_pred * 1.1, final_pred)
        
        return final_pred

def main():
    print("=== 고급 CYP3A4 효소 저해 예측 모델 (리더보드 1등 도전) ===")
    
    # 데이터 로드
    print("데이터 로딩 중...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    print(f"훈련 데이터: {len(train_df)}개")
    print(f"테스트 데이터: {len(test_df)}개")
    print(f"Inhibition 통계: 평균={train_df['Inhibition'].mean():.2f}, 표준편차={train_df['Inhibition'].std():.2f}")
    print(f"Inhibition 범위: {train_df['Inhibition'].min():.2f} ~ {train_df['Inhibition'].max():.2f}")
    
    # 데이터 준비
    X_train = train_df['Canonical_Smiles'].values
    y_train = train_df['Inhibition'].values
    X_test = test_df['Canonical_Smiles'].values
    
    # 고급 모델 학습
    print("\n=== 고급 모델 학습 시작 ===")
    model = AdvancedMolecularModel(random_state=42)
    model.fit(X_train, y_train, use_stacking=True)
    
    # 예측
    print("\n=== 고급 예측 시작 ===")
    predictions = model.predict(X_test)
    
    # 결과 저장
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'Inhibition': predictions
    })
    
    output_file = 'submission_advanced_stacking.csv'
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