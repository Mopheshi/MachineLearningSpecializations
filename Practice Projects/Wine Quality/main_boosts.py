import numpy as np
from sklearn.metrics import mean_squared_error

from main import X_train_scaled, X_test_scaled, y_train, y_test

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train_scaled, y_train)

y_pred_xgb = xgb_model.predict(X_test_scaled)

rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print(f'XGBoost RMSE: {rmse_xgb}')

lgbm_model = LGBMRegressor(random_state=42)
lgbm_model.fit(X_train_scaled, y_train)

y_pred_lgbm = lgbm_model.predict(X_test_scaled)

rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
print(f'LightGBM RMSE: {rmse_lgbm}')
