import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'sklearn-feature-selection',
	title: 'Feature Selection',
	difficulty: 'hard',
	tags: ['sklearn', 'features', 'selection'],
	estimatedTime: '15m',
	isPremium: true,
	order: 8,
	description: `# Feature Selection

Select the most important features to improve model performance.

## Task

Implement three functions:
1. \`select_k_best(X, y, k)\` - Select k best features using f_classif
2. \`select_by_variance(X, threshold)\` - Remove low variance features
3. \`recursive_feature_elimination(model, X, y, n_features)\` - RFE selection

## Example

\`\`\`python
from sklearn.feature_selection import SelectKBest, VarianceThreshold, RFE

X, y = np.random.randn(100, 10), np.array([0]*50 + [1]*50)

X_selected = select_k_best(X, y, k=5)  # (100, 5)
X_high_var = select_by_variance(X, threshold=0.1)
\`\`\``,

	initialCode: `import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, RFE

def select_k_best(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Select k best features using f_classif. Return transformed X."""
    # Your code here
    pass

def select_by_variance(X: np.ndarray, threshold: float) -> np.ndarray:
    """Remove features with variance below threshold. Return transformed X."""
    # Your code here
    pass

def recursive_feature_elimination(model, X: np.ndarray, y: np.ndarray, n_features: int) -> np.ndarray:
    """Use RFE to select n_features. Return transformed X."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, RFE

def select_k_best(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Select k best features using f_classif. Return transformed X."""
    selector = SelectKBest(score_func=f_classif, k=k)
    return selector.fit_transform(X, y)

def select_by_variance(X: np.ndarray, threshold: float) -> np.ndarray:
    """Remove features with variance below threshold. Return transformed X."""
    selector = VarianceThreshold(threshold=threshold)
    return selector.fit_transform(X)

def recursive_feature_elimination(model, X: np.ndarray, y: np.ndarray, n_features: int) -> np.ndarray:
    """Use RFE to select n_features. Return transformed X."""
    rfe = RFE(estimator=model, n_features_to_select=n_features)
    return rfe.fit_transform(X, y)
`,

	testCode: `import numpy as np
import unittest
from sklearn.linear_model import LogisticRegression

class TestFeatureSelection(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 10)
        self.y = np.array([0]*50 + [1]*50)

    def test_select_k_best_shape(self):
        X_new = select_k_best(self.X, self.y, k=5)
        self.assertEqual(X_new.shape[1], 5)
        self.assertEqual(X_new.shape[0], 100)

    def test_variance_threshold(self):
        X_low_var = np.hstack([self.X, np.ones((100, 2))])
        X_new = select_by_variance(X_low_var, threshold=0.01)
        self.assertLess(X_new.shape[1], X_low_var.shape[1])

    def test_rfe_shape(self):
        model = LogisticRegression(max_iter=1000)
        X_new = recursive_feature_elimination(model, self.X, self.y, n_features=3)
        self.assertEqual(X_new.shape[1], 3)

    def test_select_reduces_features(self):
        X_new = select_k_best(self.X, self.y, k=3)
        self.assertLess(X_new.shape[1], self.X.shape[1])

    def test_select_k_best_returns_numpy(self):
        X_new = select_k_best(self.X, self.y, k=5)
        self.assertIsInstance(X_new, np.ndarray)

    def test_variance_preserves_samples(self):
        X_new = select_by_variance(self.X, threshold=0.1)
        self.assertEqual(X_new.shape[0], 100)

    def test_rfe_returns_numpy(self):
        model = LogisticRegression(max_iter=1000)
        X_new = recursive_feature_elimination(model, self.X, self.y, n_features=4)
        self.assertIsInstance(X_new, np.ndarray)

    def test_select_k_best_different_k(self):
        X_new_2 = select_k_best(self.X, self.y, k=2)
        X_new_7 = select_k_best(self.X, self.y, k=7)
        self.assertEqual(X_new_2.shape[1], 2)
        self.assertEqual(X_new_7.shape[1], 7)

    def test_variance_all_pass(self):
        X_new = select_by_variance(self.X, threshold=0.0)
        self.assertEqual(X_new.shape[1], self.X.shape[1])

    def test_rfe_preserves_samples(self):
        model = LogisticRegression(max_iter=1000)
        X_new = recursive_feature_elimination(model, self.X, self.y, n_features=5)
        self.assertEqual(X_new.shape[0], 100)
`,

	hint1: 'SelectKBest(score_func=f_classif, k=k).fit_transform(X, y)',
	hint2: 'RFE(estimator=model, n_features_to_select=n).fit_transform(X, y)',

	whyItMatters: `Feature selection improves models by:

- **Reducing overfitting**: Fewer features, less noise
- **Faster training**: Smaller feature space
- **Interpretability**: Focus on important features
- **Curse of dimensionality**: Combat high dimensions

Essential for high-dimensional datasets.`,

	translations: {
		ru: {
			title: 'Отбор признаков',
			description: `# Отбор признаков

Выберите наиболее важные признаки для улучшения производительности модели.

## Задача

Реализуйте три функции:
1. \`select_k_best(X, y, k)\` - Выбрать k лучших признаков
2. \`select_by_variance(X, threshold)\` - Удалить признаки с низкой дисперсией
3. \`recursive_feature_elimination(model, X, y, n_features)\` - RFE отбор

## Пример

\`\`\`python
from sklearn.feature_selection import SelectKBest, VarianceThreshold, RFE

X, y = np.random.randn(100, 10), np.array([0]*50 + [1]*50)

X_selected = select_k_best(X, y, k=5)  # (100, 5)
X_high_var = select_by_variance(X, threshold=0.1)
\`\`\``,
			hint1: 'SelectKBest(score_func=f_classif, k=k).fit_transform(X, y)',
			hint2: 'RFE(estimator=model, n_features_to_select=n).fit_transform(X, y)',
			whyItMatters: `Отбор признаков улучшает модели:

- **Уменьшение переобучения**: Меньше признаков, меньше шума
- **Быстрое обучение**: Меньшее пространство признаков
- **Интерпретируемость**: Фокус на важных признаках`,
		},
		uz: {
			title: 'Xususiyat tanlash',
			description: `# Xususiyat tanlash

Model samaradorligini oshirish uchun eng muhim xususiyatlarni tanlang.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`select_k_best(X, y, k)\` - f_classif yordamida k eng yaxshi xususiyatlarni tanlash
2. \`select_by_variance(X, threshold)\` - Past dispersiyali xususiyatlarni olib tashlash
3. \`recursive_feature_elimination(model, X, y, n_features)\` - RFE tanlash

## Misol

\`\`\`python
from sklearn.feature_selection import SelectKBest, VarianceThreshold, RFE

X, y = np.random.randn(100, 10), np.array([0]*50 + [1]*50)

X_selected = select_k_best(X, y, k=5)  # (100, 5)
X_high_var = select_by_variance(X, threshold=0.1)
\`\`\``,
			hint1: "SelectKBest(score_func=f_classif, k=k).fit_transform(X, y)",
			hint2: "RFE(estimator=model, n_features_to_select=n).fit_transform(X, y)",
			whyItMatters: `Xususiyat tanlash modellarni yaxshilaydi:

- **Haddan tashqari moslanishni kamaytirish**: Kamroq xususiyatlar, kamroq shovqin
- **Tezroq o'qitish**: Kichikroq xususiyat maydoni
- **Interpretatsiya**: Muhim xususiyatlarga e'tibor`,
		},
	},
};

export default task;
