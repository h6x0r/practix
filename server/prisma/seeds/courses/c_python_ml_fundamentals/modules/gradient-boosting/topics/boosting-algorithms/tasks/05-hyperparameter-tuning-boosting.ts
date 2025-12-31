import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'boosting-hyperparameter-tuning',
	title: 'Boosting Hyperparameter Tuning',
	difficulty: 'hard',
	tags: ['tuning', 'optuna', 'boosting'],
	estimatedTime: '20m',
	isPremium: true,
	order: 5,
	description: `# Boosting Hyperparameter Tuning

Optimize gradient boosting hyperparameters with Optuna.

## Task

Implement three functions:
1. \`tune_xgboost(X, y, n_trials)\` - Tune XGBoost with Optuna
2. \`tune_lightgbm(X, y, n_trials)\` - Tune LightGBM with Optuna
3. \`get_best_model(study, X, y)\` - Train model with best params

## Example

\`\`\`python
import optuna

study = tune_xgboost(X, y, n_trials=50)
print(study.best_params)

best_model = get_best_model(study, X, y)
\`\`\``,

	initialCode: `import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

def tune_xgboost(X: np.ndarray, y: np.ndarray, n_trials: int = 50):
    """Tune XGBoost hyperparameters with Optuna. Return study object."""
    # Your code here
    pass

def tune_lightgbm(X: np.ndarray, y: np.ndarray, n_trials: int = 50):
    """Tune LightGBM hyperparameters with Optuna. Return study object."""
    # Your code here
    pass

def get_best_model(study, X: np.ndarray, y: np.ndarray):
    """Train model with best parameters from study. Return fitted model."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

def tune_xgboost(X: np.ndarray, y: np.ndarray, n_trials: int = 50):
    """Tune XGBoost hyperparameters with Optuna. Return study object."""
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }
        model = xgb.XGBClassifier(random_state=42, **params)
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study

def tune_lightgbm(X: np.ndarray, y: np.ndarray, n_trials: int = 50):
    """Tune LightGBM hyperparameters with Optuna. Return study object."""
    def objective(trial):
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        }
        model = lgb.LGBMClassifier(random_state=42, verbose=-1, **params)
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study

def get_best_model(study, X: np.ndarray, y: np.ndarray):
    """Train model with best parameters from study. Return fitted model."""
    best_params = study.best_params
    if 'num_leaves' in best_params:
        model = lgb.LGBMClassifier(random_state=42, verbose=-1, **best_params)
    else:
        model = xgb.XGBClassifier(random_state=42, **best_params)
    model.fit(X, y)
    return model
`,

	testCode: `import numpy as np
import unittest
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

class TestBoostingTuning(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.array([0]*50 + [1]*50)

    def test_tune_xgboost(self):
        study = tune_xgboost(self.X, self.y, n_trials=3)
        self.assertIsNotNone(study.best_params)

    def test_tune_lightgbm(self):
        study = tune_lightgbm(self.X, self.y, n_trials=3)
        self.assertIsNotNone(study.best_params)

    def test_get_best_model_xgb(self):
        study = tune_xgboost(self.X, self.y, n_trials=3)
        model = get_best_model(study, self.X, self.y)
        preds = model.predict(self.X[:5])
        self.assertEqual(len(preds), 5)

    def test_get_best_model_lgb(self):
        study = tune_lightgbm(self.X, self.y, n_trials=3)
        model = get_best_model(study, self.X, self.y)
        preds = model.predict(self.X[:5])
        self.assertEqual(len(preds), 5)

    def test_xgboost_study_has_best_value(self):
        study = tune_xgboost(self.X, self.y, n_trials=3)
        self.assertIsNotNone(study.best_value)

    def test_lightgbm_study_has_best_value(self):
        study = tune_lightgbm(self.X, self.y, n_trials=3)
        self.assertIsNotNone(study.best_value)

    def test_xgboost_study_returns_optuna_study(self):
        study = tune_xgboost(self.X, self.y, n_trials=3)
        self.assertTrue(hasattr(study, 'best_trial'))

    def test_lightgbm_study_returns_optuna_study(self):
        study = tune_lightgbm(self.X, self.y, n_trials=3)
        self.assertTrue(hasattr(study, 'best_trial'))

    def test_best_model_can_predict_proba(self):
        study = tune_xgboost(self.X, self.y, n_trials=3)
        model = get_best_model(study, self.X, self.y)
        probs = model.predict_proba(self.X[:5])
        self.assertEqual(probs.shape[0], 5)

    def test_best_model_lgb_can_predict_proba(self):
        study = tune_lightgbm(self.X, self.y, n_trials=3)
        model = get_best_model(study, self.X, self.y)
        probs = model.predict_proba(self.X[:5])
        self.assertEqual(probs.shape[0], 5)
`,

	hint1: 'Use trial.suggest_int() and trial.suggest_float() for hyperparameters',
	hint2: 'Access study.best_params to get optimal parameters after optimization',

	whyItMatters: `Hyperparameter tuning with Optuna:

- **Bayesian optimization**: Smarter than grid search
- **Pruning**: Early stopping of bad trials
- **Visualization**: Built-in plotting functions
- **Reproducibility**: Save and load studies

Modern approach to ML optimization.`,

	translations: {
		ru: {
			title: 'Подбор гиперпараметров бустинга',
			description: `# Подбор гиперпараметров бустинга

Оптимизируйте гиперпараметры градиентного бустинга с Optuna.

## Задача

Реализуйте три функции:
1. \`tune_xgboost(X, y, n_trials)\` - Подбор XGBoost с Optuna
2. \`tune_lightgbm(X, y, n_trials)\` - Подбор LightGBM с Optuna
3. \`get_best_model(study, X, y)\` - Обучить модель с лучшими параметрами

## Пример

\`\`\`python
import optuna

study = tune_xgboost(X, y, n_trials=50)
print(study.best_params)

best_model = get_best_model(study, X, y)
\`\`\``,
			hint1: 'Используйте trial.suggest_int() и trial.suggest_float() для гиперпараметров',
			hint2: 'study.best_params содержит оптимальные параметры',
			whyItMatters: `Подбор гиперпараметров с Optuna:

- **Байесовская оптимизация**: Умнее grid search
- **Обрезка**: Ранняя остановка плохих экспериментов
- **Визуализация**: Встроенные графики`,
		},
		uz: {
			title: 'Boosting giperparametrlarini sozlash',
			description: `# Boosting giperparametrlarini sozlash

Optuna bilan gradient boosting giperparametrlarini optimallashtiring.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`tune_xgboost(X, y, n_trials)\` - Optuna bilan XGBoost ni sozlash
2. \`tune_lightgbm(X, y, n_trials)\` - Optuna bilan LightGBM ni sozlash
3. \`get_best_model(study, X, y)\` - Eng yaxshi parametrlar bilan modelni o'rgatish

## Misol

\`\`\`python
import optuna

study = tune_xgboost(X, y, n_trials=50)
print(study.best_params)

best_model = get_best_model(study, X, y)
\`\`\``,
			hint1: "Giperparametrlar uchun trial.suggest_int() va trial.suggest_float() dan foydalaning",
			hint2: "Optimallashtirshdan keyin study.best_params ga kiring",
			whyItMatters: `Optuna bilan giperparametr sozlash:

- **Bayes optimizatsiyasi**: Grid searchdan aqlliroq
- **Kesish**: Yomon sinovlarni erta to'xtatish
- **Vizualizatsiya**: O'rnatilgan chizish funksiyalari`,
		},
	},
};

export default task;
