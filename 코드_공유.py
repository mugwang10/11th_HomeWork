# 라이브러리 및 데이터 불러오기

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split, GridSearchCV

# xgb 모델
from xgboost import XGBClassifier

import matplotlib.pyplot as plt

wine = load_wine()

# feature로 사용할 데이터에서는 'target' 컬럼을 drop합니다.
# target은 'target' 컬럼만을 대상으로 합니다.
# X, y 데이터를 test size는 0.2, random_state 값은 42로 하여 train 데이터와 test 데이터로 분할합니다.

''' 코드 작성 바랍니다 '''
df = pd.DataFrame(data=wine.data, columns= wine.feature_names)
df['target'] = wine.target
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)


####### A 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''



####### B 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''
# 하이퍼 파라미터
xgb_params = {
    "max_depth" : [3, 5, 7, 9, 15],
    "learning_rate" : [0.1, 0.01, 0.001],
    "n_estimators": [50, 100, 200, 300]
}

# 모델 정의
xgb_model = XGBClassifier(random_state=42)

# GridSearchCV로 튜닝
xgb_grid_search = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='accuracy')

# 학습
xgb_grid_search.fit(X_train, y_train)
