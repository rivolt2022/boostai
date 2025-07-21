import pandas as pd
import numpy as np
import os
import random
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem.EState import EState_VSA
from rdkit.Chem.Fragments import fr_Al_OH, fr_Ar_OH, fr_benzene, fr_ether, fr_halogen, fr_ketone, fr_ketone_Topliss
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from gensim.models.word2vec import Word2Vec
import re
from collections import Counter
import networkx as nx
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

warnings.filterwarnings('ignore')

# 전역 설정 변수들
CFG = {
    'SEED': 42,
    'N_SPLITS': 10,
    'N_TRIALS': 200,  # 더 많은 최적화 시도
    'USE_ADVANCED_FEATURES': True,
    'USE_GRAPH_FEATURES': True,
    'USE_SEQUENCE_FEATURES': True,
    'USE_ENSEMBLE': True,
    'ENSEMBLE_MODELS': ['lgb', 'xgb', 'catboost', 'rf', 'svr', 'mlp']
}

def seed_everything(seed):
    """모든 랜덤 시드를 설정"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG['SEED'])

class AdvancedMolecularPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.best_params = {}
        self.word2vec_model = None
        
        # 고급 분자 특성 계산기들
        self.fp_generators = {
            'morgan': rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048),
            'rdkit': rdFingerprintGenerator.GetRDKitFPGenerator(minPath=1, maxPath=7, fpSize=2048),
            'atom_pair': rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048),
            'torsion': rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)
        }
        
        # Word2Vec 모델 로드
        self.load_word2vec_model()
    
    def load_word2vec_model(self):
        """Word2Vec 모델 로드"""
        try:
            with open('model_300dim.pkl', 'rb') as f:
                self.word2vec_model = pickle.load(f)
            print("✅ Word2Vec 모델 로드 성공")
        except Exception as e:
            print(f"❌ Word2Vec 모델 로드 실패: {e}")
            self.word2vec_model = None
    
    def extract_advanced_fingerprints(self, mol):
        """고급 분자 지문 추출"""
        fps = {}
        
        # 새로운 지문 생성기들
        for name, generator in self.fp_generators.items():
            try:
                fp = generator.GetFingerprint(mol)
                arr = np.zeros((1,))
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps[name] = arr
            except:
                fps[name] = np.zeros(2048)
        
        # MACCS 키 (기존 방식 사용)
        try:
            fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
            arr_maccs = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp_maccs, arr_maccs)
            fps['maccs'] = arr_maccs
        except:
            fps['maccs'] = np.zeros(167)
        
        return fps
    
    def extract_graph_features(self, mol):
        """그래프 기반 특성 추출"""
        features = {}
        
        try:
            # 분자 그래프 생성
            G = nx.Graph()
            
            # 원자 추가
            for atom in mol.GetAtoms():
                G.add_node(atom.GetIdx(), 
                          symbol=atom.GetSymbol(),
                          degree=atom.GetDegree(),
                          valence=atom.GetTotalValence(),
                          aromatic=atom.GetIsAromatic(),
                          hybridization=atom.GetHybridization())
            
            # 결합 추가
            for bond in mol.GetBonds():
                G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                          type=bond.GetBondType(),
                          aromatic=bond.GetIsAromatic())
            
            # 그래프 특성 계산
            features['num_nodes'] = G.number_of_nodes()
            features['num_edges'] = G.number_of_edges()
            features['density'] = nx.density(G)
            features['avg_clustering'] = nx.average_clustering(G)
            features['avg_shortest_path'] = nx.average_shortest_path_length(G) if nx.is_connected(G) else 0
            features['diameter'] = nx.diameter(G) if nx.is_connected(G) else 0
            features['radius'] = nx.radius(G) if nx.is_connected(G) else 0
            
            # 중심성 측정
            if G.number_of_nodes() > 1:
                features['avg_degree_centrality'] = np.mean(list(nx.degree_centrality(G).values()))
                features['avg_betweenness_centrality'] = np.mean(list(nx.betweenness_centrality(G).values()))
                features['avg_closeness_centrality'] = np.mean(list(nx.closeness_centrality(G).values()))
            else:
                features['avg_degree_centrality'] = 0
                features['avg_betweenness_centrality'] = 0
                features['avg_closeness_centrality'] = 0
            
            # 순환 구조 특성
            cycles = list(nx.simple_cycles(G))
            features['num_cycles'] = len(cycles)
            features['avg_cycle_length'] = np.mean([len(cycle) for cycle in cycles]) if cycles else 0
            
        except Exception as e:
            # 실패 시 기본값
            features = {
                'num_nodes': 0, 'num_edges': 0, 'density': 0, 'avg_clustering': 0,
                'avg_shortest_path': 0, 'diameter': 0, 'radius': 0,
                'avg_degree_centrality': 0, 'avg_betweenness_centrality': 0, 'avg_closeness_centrality': 0,
                'num_cycles': 0, 'avg_cycle_length': 0
            }
        
        return features
    
    def extract_sequence_features(self, smiles):
        """SMILES 시퀀스 특성 추출"""
        features = {}
        
        # 기본 통계
        features['length'] = len(smiles)
        features['num_atoms'] = smiles.count('C') + smiles.count('N') + smiles.count('O') + smiles.count('S') + smiles.count('F') + smiles.count('Cl') + smiles.count('Br') + smiles.count('I')
        features['num_bonds'] = smiles.count('=') + smiles.count('#') + smiles.count('-') + smiles.count(':')
        
        # 특수 문자 패턴
        features['num_brackets'] = smiles.count('[') + smiles.count(']')
        features['num_parentheses'] = smiles.count('(') + smiles.count(')')
        features['num_dots'] = smiles.count('.')
        features['num_plus'] = smiles.count('+')
        features['num_minus'] = smiles.count('-')
        
        # 원자별 비율
        total_atoms = features['num_atoms']
        if total_atoms > 0:
            features['c_ratio'] = smiles.count('C') / total_atoms
            features['n_ratio'] = smiles.count('N') / total_atoms
            features['o_ratio'] = smiles.count('O') / total_atoms
            features['s_ratio'] = smiles.count('S') / total_atoms
        else:
            features['c_ratio'] = features['n_ratio'] = features['o_ratio'] = features['s_ratio'] = 0
        
        # Word2Vec 특성
        if self.word2vec_model:
            try:
                # SMILES를 토큰으로 분할
                tokens = re.findall(r'\[[^\]]+\]|[A-Z][a-z]?|\d+|[()=#\-:.]', smiles)
                vectors = []
                
                for token in tokens:
                    try:
                        if hasattr(self.word2vec_model.wv, 'word_vec'):
                            vec = self.word2vec_model.wv.word_vec(token)
                        elif hasattr(self.word2vec_model.wv, 'get_vector'):
                            vec = self.word2vec_model.wv.get_vector(token)
                        else:
                            vec = self.word2vec_model.wv[token]
                        vectors.append(vec)
                    except:
                        continue
                
                if vectors:
                    vectors = np.array(vectors)
                    features['w2v_mean'] = np.mean(vectors, axis=0)
                    features['w2v_std'] = np.std(vectors, axis=0)
                    features['w2v_max'] = np.max(vectors, axis=0)
                    features['w2v_min'] = np.min(vectors, axis=0)
                else:
                    features['w2v_mean'] = np.zeros(300)
                    features['w2v_std'] = np.zeros(300)
                    features['w2v_max'] = np.zeros(300)
                    features['w2v_min'] = np.zeros(300)
            except:
                features['w2v_mean'] = np.zeros(300)
                features['w2v_std'] = np.zeros(300)
                features['w2v_max'] = np.zeros(300)
                features['w2v_min'] = np.zeros(300)
        
        return features
    
    def extract_advanced_descriptors(self, mol):
        """고급 분자 설명자 추출"""
        descriptors = {}
        
        try:
            # 기본 RDKit 설명자
            for name, func in Descriptors._descList:
                try:
                    descriptors[f'rdkit_{name}'] = func(mol)
                except:
                    descriptors[f'rdkit_{name}'] = 0
            
            # 고급 설명자들
            descriptors['molecular_weight'] = Descriptors.MolWt(mol)
            descriptors['logp'] = Descriptors.MolLogP(mol)
            descriptors['hbd'] = Descriptors.NumHDonors(mol)
            descriptors['hba'] = Descriptors.NumHAcceptors(mol)
            descriptors['rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
            descriptors['aromatic_rings'] = Descriptors.NumAromaticRings(mol)
            descriptors['saturated_rings'] = Descriptors.NumSaturatedRings(mol)
            descriptors['heteroatoms'] = Descriptors.NumHeteroatoms(mol)
            
            # EState VSA 설명자
            estate_vsa = EState_VSA.EState_VSA_(mol)
            for i, val in enumerate(estate_vsa):
                descriptors[f'estate_vsa_{i}'] = val
            
            # 분자 조각 설명자
            fragment_funcs = [fr_Al_OH, fr_Ar_OH, fr_benzene, fr_ether, fr_halogen, fr_ketone, fr_ketone_Topliss]
            fragment_names = ['fr_Al_OH', 'fr_Ar_OH', 'fr_benzene', 'fr_ether', 'fr_halogen', 'fr_ketone', 'fr_ketone_Topliss']
            
            for name, func in zip(fragment_names, fragment_funcs):
                try:
                    descriptors[name] = func(mol)
                except:
                    descriptors[name] = 0
            
            # 3D 설명자 (가능한 경우)
            try:
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
                descriptors['asphericity'] = Descriptors.Asphericity(mol)
                descriptors['eccentricity'] = Descriptors.Eccentricity(mol)
                descriptors['inertial_shape_factor'] = Descriptors.InertialShapeFactor(mol)
                descriptors['spherocity_index'] = Descriptors.SpherocityIndex(mol)
            except:
                descriptors['asphericity'] = 0
                descriptors['eccentricity'] = 0
                descriptors['inertial_shape_factor'] = 0
                descriptors['spherocity_index'] = 0
            
        except Exception as e:
            # 실패 시 기본값들
            descriptors = {f'rdkit_{name}': 0 for name, _ in Descriptors._descList}
            descriptors.update({
                'molecular_weight': 0, 'logp': 0, 'hbd': 0, 'hba': 0,
                'rotatable_bonds': 0, 'aromatic_rings': 0, 'saturated_rings': 0, 'heteroatoms': 0,
                'asphericity': 0, 'eccentricity': 0, 'inertial_shape_factor': 0, 'spherocity_index': 0
            })
        
        return descriptors
    
    def prepare_advanced_features(self, df):
        """고급 특성 추출"""
        print("고급 분자 특성 추출 중...")
        
        all_features = []
        
        for i, row in enumerate(df.iterrows()):
            if i % 100 == 0:
                print(f"처리 중: {i}/{len(df)}")
            
            smiles = row[1]['Canonical_Smiles']
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                # SMILES 파싱 실패 시 기본값들
                features = self.get_default_features()
            else:
                features = {}
                
                # 1. 고급 분자 지문
                fps = self.extract_advanced_fingerprints(mol)
                for name, fp in fps.items():
                    features[f'fp_{name}'] = fp
                
                # 2. 그래프 특성
                if CFG['USE_GRAPH_FEATURES']:
                    graph_features = self.extract_graph_features(mol)
                    features.update(graph_features)
                
                # 3. 시퀀스 특성
                if CFG['USE_SEQUENCE_FEATURES']:
                    seq_features = self.extract_sequence_features(smiles)
                    features.update(seq_features)
                
                # 4. 고급 설명자
                if CFG['USE_ADVANCED_FEATURES']:
                    adv_descriptors = self.extract_advanced_descriptors(mol)
                    features.update(adv_descriptors)
            
            all_features.append(features)
        
        # 특성을 데이터프레임으로 변환
        feature_df = pd.DataFrame(all_features)
        
        # 특성 선택
        feature_df = self.select_features(feature_df)
        
        return feature_df.values
    
    def get_default_features(self):
        """기본 특성값 반환"""
        features = {}
        
        # 기본 지문
        for name in ['morgan', 'rdkit', 'atom_pair', 'torsion']:
            features[f'fp_{name}'] = np.zeros(2048)
        features['fp_maccs'] = np.zeros(167)
        
        # 그래프 특성
        if CFG['USE_GRAPH_FEATURES']:
            features.update({
                'num_nodes': 0, 'num_edges': 0, 'density': 0, 'avg_clustering': 0,
                'avg_shortest_path': 0, 'diameter': 0, 'radius': 0,
                'avg_degree_centrality': 0, 'avg_betweenness_centrality': 0, 'avg_closeness_centrality': 0,
                'num_cycles': 0, 'avg_cycle_length': 0
            })
        
        # 시퀀스 특성
        if CFG['USE_SEQUENCE_FEATURES']:
            features.update({
                'length': 0, 'num_atoms': 0, 'num_bonds': 0, 'num_brackets': 0,
                'num_parentheses': 0, 'num_dots': 0, 'num_plus': 0, 'num_minus': 0,
                'c_ratio': 0, 'n_ratio': 0, 'o_ratio': 0, 's_ratio': 0,
                'w2v_mean': np.zeros(300), 'w2v_std': np.zeros(300),
                'w2v_max': np.zeros(300), 'w2v_min': np.zeros(300)
            })
        
        # 고급 설명자
        if CFG['USE_ADVANCED_FEATURES']:
            features.update({f'rdkit_{name}': 0 for name, _ in Descriptors._descList})
            features.update({
                'molecular_weight': 0, 'logp': 0, 'hbd': 0, 'hba': 0,
                'rotatable_bonds': 0, 'aromatic_rings': 0, 'saturated_rings': 0, 'heteroatoms': 0,
                'asphericity': 0, 'eccentricity': 0, 'inertial_shape_factor': 0, 'spherocity_index': 0
            })
        
        return features
    
    def select_features(self, feature_df):
        """특성 선택"""
        print("특성 선택 중...")
        
        # 수치형 특성만 선택
        numeric_features = feature_df.select_dtypes(include=[np.number])
        
        # 상수 특성 제거
        constant_features = numeric_features.columns[numeric_features.std() == 0]
        numeric_features = numeric_features.drop(columns=constant_features)
        
        # 상관관계가 높은 특성 제거
        corr_matrix = numeric_features.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        numeric_features = numeric_features.drop(columns=high_corr_features)
        
        print(f"선택된 특성 수: {numeric_features.shape[1]}")
        return numeric_features
    
    def get_score(self, y_true, y_pred):
        """리더보드 평가 지표"""
        # A: Normalized RMSE
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        y_range = np.max(y_true) - np.min(y_true)
        normalized_rmse = rmse / y_range
        A = min(normalized_rmse, 1)
        
        # B: Pearson Correlation Coefficient
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        if np.isnan(correlation):
            correlation = 0
        B = np.clip(correlation, 0, 1)
        
        # 최종 스코어
        score = 0.5 * (1 - A) + 0.5 * B
        return score
    
    def objective_lgb(self, trial, X, y):
        """LightGBM 목적 함수"""
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1,
            'n_jobs': -1,
            'random_state': CFG['SEED'],
            'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 0.9),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.3, 0.9),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 1.0, log=True),
        }
        
        kf = KFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=CFG['SEED'])
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
    
    def train_ensemble(self, X_train_full, y_train_full, X_test_full):
        """앙상블 모델 훈련"""
        print("앙상블 모델 훈련 시작...")
        
        # LightGBM 최적화
        print("LightGBM 최적화 중...")
        study = optuna.create_study(direction='maximize', study_name='lgb_advanced')
        study.optimize(lambda trial: self.objective_lgb(trial, X_train_full, y_train_full), 
                      n_trials=CFG['N_TRIALS'])
        
        self.best_params['lgb'] = study.best_params
        print(f"LightGBM 최고 스코어: {study.best_value:.4f}")
        
        # K-Fold 앙상블 훈련
        print("앙상블 예측을 위한 K-폴드 훈련 중...")
        kf = KFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=CFG['SEED'])
        
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
            
            # LightGBM
            lgb_params = {
                'objective': 'regression', 'metric': 'rmse', 'verbose': -1, 'n_jobs': -1,
                'random_state': CFG['SEED']
            }
            lgb_params.update(self.best_params['lgb'])
            
            lgb_model = lgb.LGBMRegressor(**lgb_params)
            lgb_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)],
                         eval_metric='rmse', callbacks=[lgb.early_stopping(100, verbose=False)])
            
            oof_preds_models['lgb'][val_idx] = lgb_model.predict(X_val_scaled)
            test_preds_models['lgb'] += lgb_model.predict(X_test_scaled) / CFG['N_SPLITS']
            
            # XGBoost
            xgb_model = xgb.XGBRegressor(
                n_estimators=1000, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=CFG['SEED']
            )
            xgb_model.fit(X_train_scaled, y_train)
            
            oof_preds_models['xgb'][val_idx] = xgb_model.predict(X_val_scaled)
            test_preds_models['xgb'] += xgb_model.predict(X_test_scaled) / CFG['N_SPLITS']
            
            # CatBoost
            cb_model = cb.CatBoostRegressor(
                iterations=1000, learning_rate=0.05, depth=6,
                l2_leaf_reg=3, random_seed=CFG['SEED'], verbose=False
            )
            cb_model.fit(X_train_scaled, y_train)
            
            oof_preds_models['catboost'][val_idx] = cb_model.predict(X_val_scaled)
            test_preds_models['catboost'] += cb_model.predict(X_test_scaled) / CFG['N_SPLITS']
            
            # Random Forest
            rf_model = RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=CFG['SEED'], n_jobs=-1
            )
            rf_model.fit(X_train_scaled, y_train)
            
            oof_preds_models['rf'][val_idx] = rf_model.predict(X_val_scaled)
            test_preds_models['rf'] += rf_model.predict(X_test_scaled) / CFG['N_SPLITS']
            
            # SVR
            svr_model = SVR(kernel='rbf', C=1.0, gamma='scale')
            svr_model.fit(X_train_scaled, y_train)
            
            oof_preds_models['svr'][val_idx] = svr_model.predict(X_val_scaled)
            test_preds_models['svr'] += svr_model.predict(X_test_scaled) / CFG['N_SPLITS']
            
            # MLP
            mlp_model = MLPRegressor(
                hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
                alpha=0.001, max_iter=500, random_state=CFG['SEED']
            )
            mlp_model.fit(X_train_scaled, y_train)
            
            oof_preds_models['mlp'][val_idx] = mlp_model.predict(X_val_scaled)
            test_preds_models['mlp'] += mlp_model.predict(X_test_scaled) / CFG['N_SPLITS']
        
        # 앙상블 가중 평균
        weights = {
            'lgb': 0.3, 'xgb': 0.2, 'catboost': 0.2, 
            'rf': 0.15, 'svr': 0.1, 'mlp': 0.05
        }
        
        final_test_preds = np.zeros(len(X_test_full))
        final_oof_preds = np.zeros(len(X_train_full))
        
        for model_name in CFG['ENSEMBLE_MODELS']:
            final_test_preds += weights[model_name] * test_preds_models[model_name]
            final_oof_preds += weights[model_name] * oof_preds_models[model_name]
        
        # 최종 스코어 계산
        final_score = self.get_score(y_train_full, final_oof_preds)
        print(f"\n앙상블 모델 최종 스코어: {final_score:.4f}")
        
        # 개별 모델 스코어
        for model_name in CFG['ENSEMBLE_MODELS']:
            model_score = self.get_score(y_train_full, oof_preds_models[model_name])
            print(f"{model_name.upper()} 개별 스코어: {model_score:.4f}")
        
        return final_test_preds

def main():
    print("🚀 고급 분자 예측 모델 시작 🚀")
    print("=" * 70)
    
    try:
        # 데이터 로드
        print("데이터 로드 중...")
        train_df = pd.read_csv('data/train.csv')
        test_df = pd.read_csv('data/test.csv')
        sample_submission = pd.read_csv('data/sample_submission.csv')
        
        # 모델 초기화
        predictor = AdvancedMolecularPredictor()
        
        # 고급 특성 추출
        X_train_full = predictor.prepare_advanced_features(train_df)
        X_test_full = predictor.prepare_advanced_features(test_df)
        y_train_full = train_df['Inhibition']
        
        print(f"\n최종 특성 수: {X_train_full.shape[1]}")
        
        # 앙상블 모델 훈련 및 예측
        test_preds = predictor.train_ensemble(X_train_full, y_train_full, X_test_full)
        
        # 제출 파일 생성
        submission = sample_submission.copy()
        submission['Inhibition'] = test_preds
        submission['Inhibition'] = np.clip(submission['Inhibition'], 0, 100)
        
        submission.to_csv('submission_advanced.csv', index=False)
        print(f"\n✅ 제출 파일이 'submission_advanced.csv'로 저장되었습니다.")
        
        print("\n예측 결과 요약:")
        print(submission['Inhibition'].describe())
        
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 