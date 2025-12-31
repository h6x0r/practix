import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'sklearn-decision-trees',
	title: 'Decision Trees',
	difficulty: 'medium',
	tags: ['sklearn', 'decision-tree', 'classification'],
	estimatedTime: '15m',
	isPremium: false,
	order: 4,
	description: `# Decision Trees

Decision trees split data based on feature thresholds for interpretable predictions.

## Task

Implement three functions:
1. \`train_decision_tree(X, y, max_depth)\` - Train tree with depth limit
2. \`get_feature_importance(model)\` - Return feature importances
3. \`predict_with_path(model, X_sample)\` - Predict and return decision path

## Example

\`\`\`python
from sklearn.tree import DecisionTreeClassifier

X = np.array([[1, 2], [2, 3], [3, 1], [4, 5]])
y = np.array([0, 0, 1, 1])

model = train_decision_tree(X, y, max_depth=3)
importances = get_feature_importance(model)
pred, path = predict_with_path(model, X[[0]])
\`\`\``,

	initialCode: `import numpy as np
from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X: np.ndarray, y: np.ndarray, max_depth: int = None):
    """Train decision tree classifier. Return fitted model."""
    # Your code here
    pass

def get_feature_importance(model) -> np.ndarray:
    """Return feature importances array."""
    # Your code here
    pass

def predict_with_path(model, X_sample: np.ndarray) -> tuple:
    """Predict and return (prediction, decision_path_matrix)."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X: np.ndarray, y: np.ndarray, max_depth: int = None):
    """Train decision tree classifier. Return fitted model."""
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X, y)
    return model

def get_feature_importance(model) -> np.ndarray:
    """Return feature importances array."""
    return model.feature_importances_

def predict_with_path(model, X_sample: np.ndarray) -> tuple:
    """Predict and return (prediction, decision_path_matrix)."""
    prediction = model.predict(X_sample)
    path = model.decision_path(X_sample)
    return prediction, path
`,

	testCode: `import numpy as np
import unittest

class TestDecisionTrees(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 2], [2, 3], [3, 1], [4, 5], [5, 6], [6, 2]])
        self.y = np.array([0, 0, 0, 1, 1, 1])

    def test_train_returns_model(self):
        model = train_decision_tree(self.X, self.y, max_depth=3)
        self.assertIsNotNone(model)

    def test_model_respects_depth(self):
        model = train_decision_tree(self.X, self.y, max_depth=2)
        self.assertLessEqual(model.get_depth(), 2)

    def test_feature_importance_shape(self):
        model = train_decision_tree(self.X, self.y)
        importances = get_feature_importance(model)
        self.assertEqual(len(importances), 2)

    def test_feature_importance_sums_to_one(self):
        model = train_decision_tree(self.X, self.y)
        importances = get_feature_importance(model)
        self.assertAlmostEqual(sum(importances), 1.0, places=5)

    def test_predict_with_path_returns_tuple(self):
        model = train_decision_tree(self.X, self.y)
        result = predict_with_path(model, self.X[[0]])
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_prediction_shape(self):
        model = train_decision_tree(self.X, self.y)
        pred, _ = predict_with_path(model, self.X[[0]])
        self.assertEqual(len(pred), 1)

    def test_feature_importance_returns_numpy(self):
        model = train_decision_tree(self.X, self.y)
        importances = get_feature_importance(model)
        self.assertIsInstance(importances, np.ndarray)

    def test_model_can_predict(self):
        model = train_decision_tree(self.X, self.y, max_depth=3)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.y))

    def test_importances_non_negative(self):
        model = train_decision_tree(self.X, self.y)
        importances = get_feature_importance(model)
        self.assertTrue(np.all(importances >= 0))

    def test_predict_with_path_on_multiple(self):
        model = train_decision_tree(self.X, self.y)
        pred, path = predict_with_path(model, self.X[:3])
        self.assertEqual(len(pred), 3)
`,

	hint1: 'Use DecisionTreeClassifier(max_depth=n).fit(X, y) for training',
	hint2: 'Use model.feature_importances_ and model.decision_path(X) for interpretation',

	whyItMatters: `Decision trees are valuable for:

- **Interpretability**: Visualize and explain decisions
- **Feature selection**: Built-in importance scores
- **Non-linear**: Capture complex decision boundaries
- **Foundation**: Basis for ensemble methods

The most interpretable ML algorithm.`,

	translations: {
		ru: {
			title: 'Деревья решений',
			description: `# Деревья решений

Деревья решений разделяют данные на основе пороговых значений признаков для интерпретируемых предсказаний.

## Задача

Реализуйте три функции:
1. \`train_decision_tree(X, y, max_depth)\` - Обучить дерево с ограничением глубины
2. \`get_feature_importance(model)\` - Вернуть важности признаков
3. \`predict_with_path(model, X_sample)\` - Предсказать и вернуть путь решения

## Пример

\`\`\`python
from sklearn.tree import DecisionTreeClassifier

X = np.array([[1, 2], [2, 3], [3, 1], [4, 5]])
y = np.array([0, 0, 1, 1])

model = train_decision_tree(X, y, max_depth=3)
importances = get_feature_importance(model)
pred, path = predict_with_path(model, X[[0]])
\`\`\``,
			hint1: 'Используйте DecisionTreeClassifier(max_depth=n).fit(X, y)',
			hint2: 'Используйте model.feature_importances_ и model.decision_path(X)',
			whyItMatters: `Деревья решений ценны для:

- **Интерпретируемость**: Визуализация и объяснение решений
- **Отбор признаков**: Встроенные оценки важности
- **Нелинейность**: Улавливание сложных границ решений`,
		},
		uz: {
			title: 'Qaror daraxtlari',
			description: `# Qaror daraxtlari

Qaror daraxtlari interpretatsiya qilinadigan bashoratlar uchun xususiyat chegaralari asosida ma'lumotlarni ajratadi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`train_decision_tree(X, y, max_depth)\` - Chuqurlik cheklovi bilan daraxtni o'rgatish
2. \`get_feature_importance(model)\` - Xususiyat ahamiyatlarini qaytarish
3. \`predict_with_path(model, X_sample)\` - Bashorat va qaror yo'lini qaytarish

## Misol

\`\`\`python
from sklearn.tree import DecisionTreeClassifier

X = np.array([[1, 2], [2, 3], [3, 1], [4, 5]])
y = np.array([0, 0, 1, 1])

model = train_decision_tree(X, y, max_depth=3)
importances = get_feature_importance(model)
pred, path = predict_with_path(model, X[[0]])
\`\`\``,
			hint1: "DecisionTreeClassifier(max_depth=n).fit(X, y) dan foydalaning",
			hint2: "model.feature_importances_ va model.decision_path(X) dan foydalaning",
			whyItMatters: `Qaror daraxtlari quyidagilar uchun qimmatli:

- **Interpretatsiya**: Qarorlarni vizualizatsiya qilish va tushuntirish
- **Xususiyat tanlash**: O'rnatilgan ahamiyat ballari
- **Nochiziqlilik**: Murakkab qaror chegaralarini ushlash`,
		},
	},
};

export default task;
