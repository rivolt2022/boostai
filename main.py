import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import warnings
warnings.filterwarnings('ignore')

class CYP3A4InhibitionPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def calculate_molecular_descriptors(self, smiles_list):
        """SMILES 문자열에서 분자 설명자 계산"""
        descriptors = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # 기본 분자 설명자들
                    desc = {
                        'MolWt': Descriptors.MolWt(mol),
                        'LogP': Descriptors.MolLogP(mol),
                        'NumHDonors': Descriptors.NumHDonors(mol),
                        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                        'TPSA': Descriptors.TPSA(mol),
                        'NumAtoms': mol.GetNumAtoms(),
                        'NumBonds': mol.GetNumBonds(),
                        'NumRings': Descriptors.RingCount(mol),
                        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
                        'FractionCsp3': Descriptors.FractionCsp3(mol),
                        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
                        'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
                        'NumSpiroAtoms': Descriptors.NumSpiroAtoms(mol),
                        'NumBridgeheadAtoms': Descriptors.NumBridgeheadAtoms(mol),
                        'NumAmideBonds': Descriptors.NumAmideBonds(mol),
                        'NumAromaticHeterocycles': Descriptors.NumAromaticHeterocycles(mol),
                        'NumSaturatedHeterocycles': Descriptors.NumSaturatedHeterocycles(mol),
                        'NumAliphaticHeterocycles': Descriptors.NumAliphaticHeterocycles(mol),
                        'NumAromaticCarbocycles': Descriptors.NumAromaticCarbocycles(mol),
                        'NumSaturatedCarbocycles': Descriptors.NumSaturatedCarbocycles(mol),
                        'NumAliphaticCarbocycles': Descriptors.NumAliphaticCarbocycles(mol),
                        'NumStereocenters': Descriptors.NumStereocenters(mol),
                        'NumUnspecifiedAtomStereoCenters': Descriptors.NumUnspecifiedAtomStereoCenters(mol),
                        'NumRadicalElectrons': Descriptors.NumRadicalElectrons(mol),
                        'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
                        'MaxPartialCharge': Descriptors.MaxPartialCharge(mol),
                        'MinPartialCharge': Descriptors.MinPartialCharge(mol),
                        'MaxEStateIndex': Descriptors.MaxEStateIndex(mol),
                        'MinEStateIndex': Descriptors.MinEStateIndex(mol),
                        'MaxAbsEStateIndex': Descriptors.MaxAbsEStateIndex(mol),
                        'MinAbsEStateIndex': Descriptors.MinAbsEStateIndex(mol),
                        'qed': Descriptors.qed(mol),
                        'MolMR': Descriptors.MolMR(mol),
                        'ExactMolWt': Descriptors.ExactMolWt(mol),
                        'NumHeavyAtoms': mol.GetNumHeavyAtoms(),
                    }
                    
                    # 추가적인 분자 설명자들
                    try:
                        desc['LabuteASA'] = Descriptors.LabuteASA(mol)
                    except:
                        desc['LabuteASA'] = 0
                        
                    try:
                        desc['PEOE_VSA'] = sum(rdMolDescriptors.PEOE_VSA_(mol))
                    except:
                        desc['PEOE_VSA'] = 0
                        
                    try:
                        desc['SMR_VSA'] = sum(rdMolDescriptors.SMR_VSA_(mol))
                    except:
                        desc['SMR_VSA'] = 0
                        
                    try:
                        desc['SlogP_VSA'] = sum(rdMolDescriptors.SlogP_VSA_(mol))
                    except:
                        desc['SlogP_VSA'] = 0
                        
                    try:
                        desc['EState_VSA'] = sum(rdMolDescriptors.EState_VSA_(mol))
                    except:
                        desc['EState_VSA'] = 0
                        
                    try:
                        desc['VSA_EState'] = sum(rdMolDescriptors.VSA_EState_(mol))
                    except:
                        desc['VSA_EState'] = 0
                        
                    descriptors.append(desc)
                else:
                    # SMILES 파싱 실패 시 기본값으로 채움
                    descriptors.append({key: 0 for key in [
                        'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
                        'TPSA', 'NumAtoms', 'NumBonds', 'NumRings', 'NumAromaticRings',
                        'FractionCsp3', 'HeavyAtomCount', 'NumHeteroatoms', 'NumSpiroAtoms',
                        'NumBridgeheadAtoms', 'NumAmideBonds', 'NumAromaticHeterocycles',
                        'NumSaturatedHeterocycles', 'NumAliphaticHeterocycles',
                        'NumAromaticCarbocycles', 'NumSaturatedCarbocycles',
                        'NumAliphaticCarbocycles', 'NumStereocenters',
                        'NumUnspecifiedAtomStereoCenters', 'NumRadicalElectrons',
                        'NumValenceElectrons', 'MaxPartialCharge', 'MinPartialCharge',
                        'MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex',
                        'MinAbsEStateIndex', 'qed', 'MolMR', 'ExactMolWt', 'NumHeavyAtoms',
                        'LabuteASA', 'PEOE_VSA', 'SMR_VSA', 'SlogP_VSA', 'EState_VSA', 'VSA_EState'
                    ]})
            except:
                # 예외 발생 시 기본값으로 채움
                descriptors.append({key: 0 for key in [
                    'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
                    'TPSA', 'NumAtoms', 'NumBonds', 'NumRings', 'NumAromaticRings',
                    'FractionCsp3', 'HeavyAtomCount', 'NumHeteroatoms', 'NumSpiroAtoms',
                    'NumBridgeheadAtoms', 'NumAmideBonds', 'NumAromaticHeterocycles',
                    'NumSaturatedHeterocycles', 'NumAliphaticHeterocycles',
                    'NumAromaticCarbocycles', 'NumSaturatedCarbocycles',
                    'NumAliphaticCarbocycles', 'NumStereocenters',
                    'NumUnspecifiedAtomStereoCenters', 'NumRadicalElectrons',
                    'NumValenceElectrons', 'MaxPartialCharge', 'MinPartialCharge',
                    'MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex',
                    'MinAbsEStateIndex', 'qed', 'MolMR', 'ExactMolWt', 'NumHeavyAtoms',
                    'LabuteASA', 'PEOE_VSA', 'SMR_VSA', 'SlogP_VSA', 'EState_VSA', 'VSA_EState'
                ]})
        
        return pd.DataFrame(descriptors)
    
    def prepare_data(self, train_data, test_data=None):
        """데이터 준비 및 특성 추출"""
        print("훈련 데이터에서 분자 설명자 계산 중...")
        train_features = self.calculate_molecular_descriptors(train_data['Canonical_Smiles'])
        
        if test_data is not None:
            print("테스트 데이터에서 분자 설명자 계산 중...")
            test_features = self.calculate_molecular_descriptors(test_data['Canonical_Smiles'])
            return train_features, test_features
        
        return train_features
    
    def train(self, X_train, y_train):
        """모델 훈련"""
        print("모델 훈련 중...")
        
        # 여러 모델 시도
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=200, random_state=42)
        }
        
        best_score = -np.inf
        best_model = None
        best_model_name = None
        
        for name, model in models.items():
            print(f"{name} 모델 훈련 중...")
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            mean_score = scores.mean()
            print(f"{name} CV R² Score: {mean_score:.4f} (+/- {scores.std() * 2:.4f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
                best_model_name = name
        
        print(f"\n최적 모델: {best_model_name}")
        
        # 최적 모델로 전체 데이터 훈련
        self.model = best_model
        self.model.fit(X_train, y_train)
        
        # 훈련 성능 평가
        y_pred = self.model.predict(X_train)
        train_r2 = r2_score(y_train, y_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        train_mae = mean_absolute_error(y_train, y_pred)
        
        print(f"\n훈련 성능:")
        print(f"R² Score: {train_r2:.4f}")
        print(f"RMSE: {train_rmse:.4f}")
        print(f"MAE: {train_mae:.4f}")
        
        return self.model
    
    def predict(self, X):
        """예측 수행"""
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다. train() 메서드를 먼저 호출하세요.")
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """모델 평가"""
        y_pred = self.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"\n테스트 성능:")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        return r2, rmse, mae
    
    def plot_results(self, y_true, y_pred, title="예측 vs 실제"):
        """결과 시각화"""
        plt.figure(figsize=(10, 8))
        
        # 산점도
        plt.subplot(2, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('실제 값')
        plt.ylabel('예측 값')
        plt.title(f'{title}\nR² = {r2_score(y_true, y_pred):.4f}')
        
        # 잔차 플롯
        plt.subplot(2, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('예측 값')
        plt.ylabel('잔차')
        plt.title('잔차 플롯')
        
        # 히스토그램
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=30, alpha=0.7)
        plt.xlabel('잔차')
        plt.ylabel('빈도')
        plt.title('잔차 분포')
        
        # 특성 중요도 (RandomForest인 경우)
        if hasattr(self.model, 'feature_importances_'):
            plt.subplot(2, 2, 4)
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)
            
            plt.barh(range(len(feature_importance)), feature_importance['importance'])
            plt.yticks(range(len(feature_importance)), feature_importance['feature'])
            plt.xlabel('특성 중요도')
            plt.title('상위 15개 특성 중요도')
        
        plt.tight_layout()
        plt.show()

def main():
    print("CYP3A4 효소 저해 예측 모델 개발")
    print("=" * 50)
    
    # 데이터 로드
    print("데이터 로드 중...")
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    sample_submission = pd.read_csv('data/sample_submission.csv')
    
    print(f"훈련 데이터: {train_data.shape}")
    print(f"테스트 데이터: {test_data.shape}")
    print(f"샘플 제출 파일: {sample_submission.shape}")
    
    # 데이터 기본 정보
    print("\n훈련 데이터 정보:")
    print(train_data.info())
    print(f"\n저해율 통계:")
    print(train_data['Inhibition'].describe())
    
    # 모델 초기화
    predictor = CYP3A4InhibitionPredictor()
    
    # 특성 추출
    train_features, test_features = predictor.prepare_data(train_data, test_data)
    predictor.feature_names = train_features.columns.tolist()
    
    print(f"\n추출된 특성 수: {len(predictor.feature_names)}")
    print("주요 특성들:", predictor.feature_names[:10])
    
    # 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(
        train_features, train_data['Inhibition'], 
        test_size=0.2, random_state=42
    )
    
    print(f"\n훈련 세트: {X_train.shape}")
    print(f"검증 세트: {X_val.shape}")
    
    # 모델 훈련
    predictor.train(X_train, y_train)
    
    # 검증 세트에서 평가
    val_r2, val_rmse, val_mae = predictor.evaluate(X_val, y_val)
    
    # 결과 시각화
    y_val_pred = predictor.predict(X_val)
    predictor.plot_results(y_val, y_val_pred, "검증 세트 예측 결과")
    
    # 테스트 데이터 예측
    print("\n테스트 데이터 예측 중...")
    test_predictions = predictor.predict(test_features)
    
    # 제출 파일 생성
    submission = sample_submission.copy()
    submission['Inhibition'] = test_predictions
    
    # 예측값 범위 조정 (0-100% 범위로)
    submission['Inhibition'] = np.clip(submission['Inhibition'], 0, 100)
    
    # 제출 파일 저장
    submission.to_csv('submission.csv', index=False)
    print(f"\n제출 파일이 'submission.csv'로 저장되었습니다.")
    
    # 예측 결과 요약
    print(f"\n예측 결과 요약:")
    print(f"평균 예측 저해율: {submission['Inhibition'].mean():.2f}%")
    print(f"최소 예측 저해율: {submission['Inhibition'].min():.2f}%")
    print(f"최대 예측 저해율: {submission['Inhibition'].max():.2f}%")
    
    # 상위 예측값들
    print(f"\n상위 10개 예측값:")
    top_predictions = submission.nlargest(10, 'Inhibition')
    for idx, row in top_predictions.iterrows():
        print(f"{row['ID']}: {row['Inhibition']:.2f}%")
    
    return predictor, submission

if __name__ == "__main__":
    model, submission = main()