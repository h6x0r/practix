import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'boosting-shap-importance',
	title: 'Feature Importance with SHAP',
	difficulty: 'hard',
	tags: ['shap', 'interpretability', 'boosting'],
	estimatedTime: '18m',
	isPremium: true,
	order: 6,
	description: `# Feature Importance with SHAP

Explain model predictions using SHAP values.

## Task

Implement three functions:
1. \`compute_shap_values(model, X)\` - Calculate SHAP values for predictions
2. \`get_feature_importance_shap(shap_values)\` - Get mean absolute SHAP importance
3. \`explain_single_prediction(model, X, idx)\` - Explain one prediction

## Example

\`\`\`python
import shap

model = xgb.XGBClassifier().fit(X_train, y_train)
shap_values = compute_shap_values(model, X_test)

importance = get_feature_importance_shap(shap_values)
explanation = explain_single_prediction(model, X_test, idx=0)
\`\`\``,

	initialCode: `import numpy as np
import shap
import xgboost as xgb

def compute_shap_values(model, X: np.ndarray) -> np.ndarray:
    """Compute SHAP values for model predictions. Return shap values array."""
    # Your code here
    pass

def get_feature_importance_shap(shap_values: np.ndarray) -> np.ndarray:
    """Get mean absolute SHAP importance per feature. Return array."""
    # Your code here
    pass

def explain_single_prediction(model, X: np.ndarray, idx: int) -> dict:
    """Explain prediction at index. Return dict with base_value and shap_values."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
import shap
import xgboost as xgb

def compute_shap_values(model, X: np.ndarray) -> np.ndarray:
    """Compute SHAP values for model predictions. Return shap values array."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return shap_values

def get_feature_importance_shap(shap_values: np.ndarray) -> np.ndarray:
    """Get mean absolute SHAP importance per feature. Return array."""
    return np.abs(shap_values).mean(axis=0)

def explain_single_prediction(model, X: np.ndarray, idx: int) -> dict:
    """Explain prediction at index. Return dict with base_value and shap_values."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[idx:idx+1])
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return {
        'base_value': explainer.expected_value if not isinstance(explainer.expected_value, list)
                      else explainer.expected_value[1],
        'shap_values': shap_values[0]
    }
`,

	testCode: `import numpy as np
import unittest
import xgboost as xgb

class TestSHAP(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.array([0]*50 + [1]*50)
        self.model = xgb.XGBClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X, self.y)

    def test_compute_shap_values_shape(self):
        shap_vals = compute_shap_values(self.model, self.X[:10])
        self.assertEqual(shap_vals.shape[0], 10)
        self.assertEqual(shap_vals.shape[1], 5)

    def test_feature_importance_length(self):
        shap_vals = compute_shap_values(self.model, self.X[:10])
        importance = get_feature_importance_shap(shap_vals)
        self.assertEqual(len(importance), 5)

    def test_importance_non_negative(self):
        shap_vals = compute_shap_values(self.model, self.X[:10])
        importance = get_feature_importance_shap(shap_vals)
        self.assertTrue(all(i >= 0 for i in importance))

    def test_explain_single_returns_dict(self):
        explanation = explain_single_prediction(self.model, self.X, idx=0)
        self.assertIn('base_value', explanation)
        self.assertIn('shap_values', explanation)

    def test_shap_values_returns_numpy(self):
        shap_vals = compute_shap_values(self.model, self.X[:10])
        self.assertIsInstance(shap_vals, np.ndarray)

    def test_importance_returns_numpy(self):
        shap_vals = compute_shap_values(self.model, self.X[:10])
        importance = get_feature_importance_shap(shap_vals)
        self.assertIsInstance(importance, np.ndarray)

    def test_explain_shap_values_length(self):
        explanation = explain_single_prediction(self.model, self.X, idx=0)
        self.assertEqual(len(explanation['shap_values']), 5)

    def test_explain_base_value_is_numeric(self):
        explanation = explain_single_prediction(self.model, self.X, idx=0)
        self.assertIsInstance(explanation['base_value'], (int, float, np.floating))

    def test_shap_values_different_idx(self):
        exp1 = explain_single_prediction(self.model, self.X, idx=0)
        exp2 = explain_single_prediction(self.model, self.X, idx=1)
        self.assertFalse(np.allclose(exp1['shap_values'], exp2['shap_values']))

    def test_importance_sums_not_zero(self):
        shap_vals = compute_shap_values(self.model, self.X[:10])
        importance = get_feature_importance_shap(shap_vals)
        self.assertGreater(sum(importance), 0)
`,

	hint1: 'shap.TreeExplainer(model).shap_values(X) for tree-based models',
	hint2: 'Mean absolute SHAP: np.abs(shap_values).mean(axis=0)',

	whyItMatters: `SHAP provides:

- **Model interpretability**: Understand why predictions are made
- **Feature importance**: More accurate than built-in importance
- **Local explanations**: Explain individual predictions
- **Debugging**: Find model issues and biases

Essential for trustworthy ML in production.`,

	translations: {
		ru: {
			title: 'Важность признаков с SHAP',
			description: `# Важность признаков с SHAP

Объясняйте предсказания модели с помощью SHAP значений.

## Задача

Реализуйте три функции:
1. \`compute_shap_values(model, X)\` - Вычислить SHAP значения
2. \`get_feature_importance_shap(shap_values)\` - Средняя абсолютная важность
3. \`explain_single_prediction(model, X, idx)\` - Объяснить одно предсказание

## Пример

\`\`\`python
import shap

model = xgb.XGBClassifier().fit(X_train, y_train)
shap_values = compute_shap_values(model, X_test)

importance = get_feature_importance_shap(shap_values)
explanation = explain_single_prediction(model, X_test, idx=0)
\`\`\``,
			hint1: 'shap.TreeExplainer(model).shap_values(X) для древовидных моделей',
			hint2: 'Средний абсолютный SHAP: np.abs(shap_values).mean(axis=0)',
			whyItMatters: `SHAP предоставляет:

- **Интерпретируемость**: Понять почему делаются предсказания
- **Важность признаков**: Точнее встроенной важности
- **Локальные объяснения**: Объяснение отдельных предсказаний`,
		},
		uz: {
			title: 'SHAP bilan xususiyat muhimligi',
			description: `# SHAP bilan xususiyat muhimligi

SHAP qiymatlari yordamida model bashoratlarini tushuntiring.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`compute_shap_values(model, X)\` - SHAP qiymatlarini hisoblash
2. \`get_feature_importance_shap(shap_values)\` - O'rtacha mutlaq SHAP muhimligini olish
3. \`explain_single_prediction(model, X, idx)\` - Bitta bashoratni tushuntirish

## Misol

\`\`\`python
import shap

model = xgb.XGBClassifier().fit(X_train, y_train)
shap_values = compute_shap_values(model, X_test)

importance = get_feature_importance_shap(shap_values)
explanation = explain_single_prediction(model, X_test, idx=0)
\`\`\``,
			hint1: "Daraxt asosidagi modellar uchun shap.TreeExplainer(model).shap_values(X)",
			hint2: "O'rtacha mutlaq SHAP: np.abs(shap_values).mean(axis=0)",
			whyItMatters: `SHAP quyidagilarni ta'minlaydi:

- **Model interpretatsiyligi**: Nima uchun bashoratlar qilinishini tushunish
- **Xususiyat muhimligi**: O'rnatilgan muhimlikdan aniqroq
- **Mahalliy tushuntirishlar**: Individual bashoratlarni tushuntirish`,
		},
	},
};

export default task;
