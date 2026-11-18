"""Machine learning, 'ship as a wave buoy': estimate the buoy's Hm0 from the ship's own motion per 10-min block."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_predict

FEATURES = ['heave_hs_m', 'roll_std', 'pitch_std', 'speed_kn']
MODELS = {'Mean (baseline)': DummyRegressor(),
          'Linear regression': LinearRegression(),
          'Random forest': RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0)}
AS_IS = 'Ship heave 4σ as-is'   # no fitting: the ship's heave 4σ read directly as Hm0


def evaluate(df, features, target='Hm0', leaky=False):
    """Leave-one-day-out predictions for every model: blocks within a day are autocorrelated, so a random
    split leaks (leaky=True shows how much). Returns (metrics, predictions in long form, RF permutation importance)."""
    X, y, days = df[features], df[target], df['day']
    cv, groups = (KFold(5, shuffle=True, random_state=0), None) if leaky else (LeaveOneGroupOut(), days)
    pred = {name: cross_val_predict(m, X, y, groups=groups, cv=cv) for name, m in MODELS.items()}
    pred[AS_IS] = df.heave_hs_m.values
    long = pd.concat([df.assign(model=name, predicted=p) for name, p in pred.items()], ignore_index=True)
    rows = []
    for name, g in long.groupby('model', sort=False):
        err = g.predicted - g[target]
        rows.append({'model': name, 'MAE_m': err.abs().mean(), 'RMSE_m': np.sqrt((err ** 2).mean()),
                     'R2': r2_score(g[target], g.predicted),
                     **{f'MAE_m {d}': e.abs().mean() for d, e in err.groupby(g.day)}})
    # ponytail: importance on the training fit is optimistic; compute it per held-out day if it gets argued over
    rf = clone(MODELS['Random forest']).fit(X, y)
    imp = permutation_importance(rf, X, y, n_repeats=20, random_state=0)
    importance = pd.DataFrame({'feature': features, 'importance': imp.importances_mean, 'std': imp.importances_std})
    return pd.DataFrame(rows), long, importance
