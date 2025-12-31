import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'xgboost-basics',
	title: 'XGBoost',
	difficulty: 'medium',
	tags: ['xgboost', 'boosting', 'ml'],
	estimatedTime: '15m',
	isPremium: false,
	order: 2,
	description: `# XGBoost

Master XGBoost - the most popular gradient boosting library.

## Task

Implement three functions:
1. \`train_xgb_classifier(X, y, params)\` - Train XGBClassifier with custom params
2. \`train_xgb_regressor(X, y, params)\` - Train XGBRegressor
3. \`early_stopping_train(X_train, y_train, X_val, y_val)\` - Train with early stopping

## Example

\`\`\`python
import xgboost as xgb

params = {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100}
model = train_xgb_classifier(X_train, y_train, params)

# With early stopping
model = early_stopping_train(X_train, y_train, X_val, y_val)
\`\`\``,

	initialCode: `import numpy as np
import xgboost as xgb

def train_xgb_classifier(X: np.ndarray, y: np.ndarray, params: dict = None):
    """Train XGBClassifier with given params. Return fitted model."""
    # Your code here
    pass

def train_xgb_regressor(X: np.ndarray, y: np.ndarray, params: dict = None):
    """Train XGBRegressor with given params. Return fitted model."""
    # Your code here
    pass

def early_stopping_train(X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray, n_estimators: int = 1000):
    """Train with early stopping. Return fitted model."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
import xgboost as xgb

def train_xgb_classifier(X: np.ndarray, y: np.ndarray, params: dict = None):
    """Train XGBClassifier with given params. Return fitted model."""
    if params is None:
        params = {}
    model = xgb.XGBClassifier(random_state=42, **params)
    model.fit(X, y)
    return model

def train_xgb_regressor(X: np.ndarray, y: np.ndarray, params: dict = None):
    """Train XGBRegressor with given params. Return fitted model."""
    if params is None:
        params = {}
    model = xgb.XGBRegressor(random_state=42, **params)
    model.fit(X, y)
    return model

def early_stopping_train(X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray, n_estimators: int = 1000):
    """Train with early stopping. Return fitted model."""
    model = xgb.XGBClassifier(n_estimators=n_estimators, random_state=42,
                               early_stopping_rounds=50, eval_metric='logloss')
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model
`,

	testCode: `import numpy as np
import unittest

class TestXGBoost(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y_class = np.array([0]*50 + [1]*50)
        self.y_reg = np.random.randn(100)

    def test_classifier_default_params(self):
        model = train_xgb_classifier(self.X, self.y_class)
        self.assertIsNotNone(model)

    def test_classifier_custom_params(self):
        params = {'max_depth': 3, 'n_estimators': 10}
        model = train_xgb_classifier(self.X, self.y_class, params)
        preds = model.predict(self.X[:5])
        self.assertEqual(len(preds), 5)

    def test_regressor(self):
        model = train_xgb_regressor(self.X, self.y_reg, {'n_estimators': 10})
        preds = model.predict(self.X[:5])
        self.assertEqual(len(preds), 5)

    def test_early_stopping(self):
        X_train, X_val = self.X[:80], self.X[80:]
        y_train, y_val = self.y_class[:80], self.y_class[80:]
        model = early_stopping_train(X_train, y_train, X_val, y_val, n_estimators=100)
        self.assertIsNotNone(model)

    def test_classifier_has_predict_proba(self):
        model = train_xgb_classifier(self.X, self.y_class)
        probs = model.predict_proba(self.X[:5])
        self.assertEqual(probs.shape[0], 5)

    def test_regressor_default(self):
        model = train_xgb_regressor(self.X, self.y_reg)
        self.assertIsNotNone(model)

    def test_classifier_has_feature_importances(self):
        model = train_xgb_classifier(self.X, self.y_class, {'n_estimators': 10})
        self.assertTrue(hasattr(model, 'feature_importances_'))

    def test_early_stopping_can_predict(self):
        X_train, X_val = self.X[:80], self.X[80:]
        y_train, y_val = self.y_class[:80], self.y_class[80:]
        model = early_stopping_train(X_train, y_train, X_val, y_val, n_estimators=100)
        preds = model.predict(self.X[:5])
        self.assertEqual(len(preds), 5)

    def test_regressor_predictions_are_floats(self):
        model = train_xgb_regressor(self.X, self.y_reg, {'n_estimators': 10})
        preds = model.predict(self.X[:5])
        self.assertTrue(all(isinstance(p, (float, np.floating)) for p in preds))
`,

	hint1: 'xgb.XGBClassifier(**params).fit(X, y)',
	hint2: 'For early stopping: fit(X, y, eval_set=[(X_val, y_val)])',

	whyItMatters: `XGBoost dominates because:

- **Speed**: Optimized C++ implementation with parallelization
- **Regularization**: L1/L2 built-in to prevent overfitting
- **Missing values**: Native handling of missing data
- **Competition winner**: Most used in Kaggle competitions

Industry standard for tabular data.`,

	translations: {
		ru: {
			title: 'XGBoost',
			description: `# XGBoost

Освойте XGBoost - самую популярную библиотеку градиентного бустинга.

## Задача

Реализуйте три функции:
1. \`train_xgb_classifier(X, y, params)\` - Обучить XGBClassifier
2. \`train_xgb_regressor(X, y, params)\` - Обучить XGBRegressor
3. \`early_stopping_train(X_train, y_train, X_val, y_val)\` - Обучение с ранней остановкой

## Пример

\`\`\`python
import xgboost as xgb

params = {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100}
model = train_xgb_classifier(X_train, y_train, params)

# With early stopping
model = early_stopping_train(X_train, y_train, X_val, y_val)
\`\`\``,
			hint1: 'xgb.XGBClassifier(**params).fit(X, y)',
			hint2: 'Для ранней остановки: fit(X, y, eval_set=[(X_val, y_val)])',
			whyItMatters: `XGBoost доминирует потому что:

- **Скорость**: Оптимизированная C++ реализация с параллелизацией
- **Регуляризация**: Встроенные L1/L2 против переобучения
- **Пропущенные значения**: Нативная обработка пропусков`,
		},
		uz: {
			title: 'XGBoost',
			description: `# XGBoost

XGBoostni o'rganing - eng mashhur gradient boosting kutubxonasi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`train_xgb_classifier(X, y, params)\` - XGBClassifier ni o'rgatish
2. \`train_xgb_regressor(X, y, params)\` - XGBRegressor ni o'rgatish
3. \`early_stopping_train(X_train, y_train, X_val, y_val)\` - Erta to'xtatish bilan o'rgatish

## Misol

\`\`\`python
import xgboost as xgb

params = {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100}
model = train_xgb_classifier(X_train, y_train, params)

# With early stopping
model = early_stopping_train(X_train, y_train, X_val, y_val)
\`\`\``,
			hint1: "xgb.XGBClassifier(**params).fit(X, y)",
			hint2: "Erta to'xtatish uchun: fit(X, y, eval_set=[(X_val, y_val)])",
			whyItMatters: `XGBoost ustunlik qiladi chunki:

- **Tezlik**: Parallellashtirish bilan optimallashtirilgan C++ realizatsiyasi
- **Regulyarizatsiya**: Ortiqcha moslanishni oldini olish uchun o'rnatilgan L1/L2
- **Yo'qolgan qiymatlar**: Yo'qolgan ma'lumotlarni tabiiy ishlov berish`,
		},
	},
};

export default task;
