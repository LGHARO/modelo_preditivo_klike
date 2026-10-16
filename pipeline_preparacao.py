import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

# 1. Carregamento dos dados
df = pd.read_csv('base_de_dados/klike_challenge_dataset.csv')

# 2. Feature Engineering: Taxa de Retenção do Usuário (% do vídeo assistida)
# Proporção entre tempo médio assistido e duração total (limitada entre 0 e 100%)
df['retention_rate'] = (df['avg_watch_time_s'] / df['video_duration_s']).clip(0, 1)

# 3. Separação de X e y (descartando apenas ID e data)
X = df.drop(columns=['klike_score', 'campaign_id', 'date'])
y = df['klike_score']

# 4. Divisão em Treino (80%) e Teste (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Mapeamento dos grupos conforme sinalizado na EDA
# Features com cauda longa / alta assimetria (log1p + padronização)
cols_log = ['impressions', 'clicks', 'spend', 'conversions', 'revenue', 'roas']

# Features contínuas com presença de outliers (RobustScaler)
cols_robust = ['video_duration_s', 'avg_watch_time_s', 'cpc']

# Taxas limitadas entre 0 e 1 (MinMaxScaler) - agora incluindo retention_rate!
cols_taxas = ['ctr', 'engagement_rate', 'music_voice_ratio', 'retention_rate']

# Categóricas nominais (One-Hot Encoding)
cols_nominais = ['platform', 'category', 'objective', 'format']

# Categóricas ordinais (Ordinal Encoding preservando hierarquia)
cols_ordinais = ['target_audience_age', 'text_density']

# Flags binárias / booleanas
cols_binarias = ['is_retargeting', 'has_subtitle', 'has_cta', 'has_hook', 'has_face']

# 6. Construção do Pipeline de Pré-processamento
preprocessor = ColumnTransformer(
    transformers=[
        (
            'log',
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('log1p', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
                ('scaler', StandardScaler()),
            ]),
            cols_log,
        ),
        (
            'robust',
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler()),
            ]),
            cols_robust,
        ),
        (
            'taxas',
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', MinMaxScaler()),
            ]),
            cols_taxas,
        ),
        (
            'nominais',
            Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                (
                    'ohe',
                    OneHotEncoder(
                        drop='first', sparse_output=False, handle_unknown='ignore'
                    ),
                ),
            ]),
            cols_nominais,
        ),
        (
            'ordinais',
            Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                (
                    'ord',
                    OrdinalEncoder(
                        categories=[
                            ['18-24', '25-34', '35-44', '45+'],
                            ['low', 'medium', 'high'],
                        ],
                        handle_unknown='use_encoded_value',
                        unknown_value=-1,
                    ),
                ),
                ('scaler', MinMaxScaler()),
            ]),
            cols_ordinais,
        ),
        (
            'binarias',
            Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                (
                    'to_int',
                    FunctionTransformer(
                        lambda x: x.astype(int), feature_names_out='one-to-one'
                    ),
                ),
            ]),
            cols_binarias,
        ),
    ]
)

# 7. Ajuste no treino e transformação no teste (sem vazamento de dados)
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# Nomes das colunas resultantes
feature_names = preprocessor.get_feature_names_out()

if __name__ == '__main__':
    print("=== PIPELINE EXECUTADO COM SUCESSO (COM RETENTION_RATE) ===")
    print(f"X_train preparado: {X_train_prep.shape} (amostras, features)")
    print(f"X_test preparado : {X_test_prep.shape} (amostras, features)")
    print(f"Nulos em X_train : {np.isnan(X_train_prep).sum()}")
    print(f"Nulos em X_test  : {np.isnan(X_test_prep).sum()}")
    print(f"\nTotal de features geradas: {len(feature_names)}")
    print("Nova feature incluída: taxas__retention_rate")
