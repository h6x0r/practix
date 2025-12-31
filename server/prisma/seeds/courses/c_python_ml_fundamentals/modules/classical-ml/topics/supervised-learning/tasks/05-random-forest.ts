import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'sklearn-random-forest',
	title: 'Random Forest',
	difficulty: 'medium',
	tags: ['sklearn', 'ensemble', 'random-forest'],
	estimatedTime: '15m',
	isPremium: false,
	order: 5,
	description: `# Random Forest

Random Forest combines multiple decision trees for robust predictions.

## Task

Implement three functions:
1. \`train_random_forest(X, y, n_estimators)\` - Train forest with n trees
2. \`get_oob_score(X, y, n_estimators)\` - Train with OOB and return score
3. \`compare_with_single_tree(X, y)\` - Compare RF vs single tree accuracy

## Example

\`\`\`python
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(n_samples=100, n_features=10)

model = train_random_forest(X, y, n_estimators=100)
oob = get_oob_score(X, y, n_estimators=100)  # ~0.85
comparison = compare_with_single_tree(X, y)  # {'tree': 0.8, 'forest': 0.9}
\`\`\``,

	initialCode: `import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

def train_random_forest(X: np.ndarray, y: np.ndarray, n_estimators: int = 100):
    """Train random forest classifier. Return fitted model."""
    # Your code here
    pass

def get_oob_score(X: np.ndarray, y: np.ndarray, n_estimators: int = 100) -> float:
    """Train with OOB and return out-of-bag score."""
    # Your code here
    pass

def compare_with_single_tree(X: np.ndarray, y: np.ndarray) -> dict:
    """Compare RF vs single tree using cross-validation. Return {'tree': score, 'forest': score}."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

def train_random_forest(X: np.ndarray, y: np.ndarray, n_estimators: int = 100):
    """Train random forest classifier. Return fitted model."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X, y)
    return model

def get_oob_score(X: np.ndarray, y: np.ndarray, n_estimators: int = 100) -> float:
    """Train with OOB and return out-of-bag score."""
    model = RandomForestClassifier(n_estimators=n_estimators, oob_score=True, random_state=42)
    model.fit(X, y)
    return model.oob_score_

def compare_with_single_tree(X: np.ndarray, y: np.ndarray) -> dict:
    """Compare RF vs single tree using cross-validation. Return {'tree': score, 'forest': score}."""
    tree = DecisionTreeClassifier(random_state=42)
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    tree_score = cross_val_score(tree, X, y, cv=5).mean()
    forest_score = cross_val_score(forest, X, y, cv=5).mean()
    return {'tree': tree_score, 'forest': forest_score}
`,

	testCode: `import numpy as np
from sklearn.datasets import make_classification
import unittest

class TestRandomForest(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification(n_samples=100, n_features=10, random_state=42)

    def test_train_returns_model(self):
        model = train_random_forest(self.X, self.y, 10)
        self.assertIsNotNone(model)

    def test_model_has_estimators(self):
        model = train_random_forest(self.X, self.y, 10)
        self.assertEqual(len(model.estimators_), 10)

    def test_oob_score_in_range(self):
        score = get_oob_score(self.X, self.y, 50)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_compare_returns_dict(self):
        result = compare_with_single_tree(self.X, self.y)
        self.assertIsInstance(result, dict)
        self.assertIn('tree', result)
        self.assertIn('forest', result)

    def test_forest_usually_better(self):
        np.random.seed(42)
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        result = compare_with_single_tree(X, y)
        # Forest should generally be >= tree
        self.assertGreaterEqual(result['forest'], result['tree'] - 0.1)

    def test_model_has_feature_importances(self):
        model = train_random_forest(self.X, self.y, 10)
        self.assertTrue(hasattr(model, 'feature_importances_'))
        self.assertEqual(len(model.feature_importances_), 10)

    def test_oob_score_returns_float(self):
        score = get_oob_score(self.X, self.y, 50)
        self.assertIsInstance(score, float)

    def test_compare_scores_in_range(self):
        result = compare_with_single_tree(self.X, self.y)
        self.assertTrue(0 <= result['tree'] <= 1)
        self.assertTrue(0 <= result['forest'] <= 1)

    def test_model_can_predict(self):
        model = train_random_forest(self.X, self.y, 10)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.y))

    def test_different_n_estimators(self):
        model_10 = train_random_forest(self.X, self.y, 10)
        model_50 = train_random_forest(self.X, self.y, 50)
        self.assertEqual(len(model_10.estimators_), 10)
        self.assertEqual(len(model_50.estimators_), 50)
`,

	hint1: 'Use RandomForestClassifier(n_estimators=n, oob_score=True) for OOB',
	hint2: 'Use cross_val_score for fair comparison between models',

	whyItMatters: `Random Forest is powerful because:

- **Robustness**: Reduces overfitting through averaging
- **Feature importance**: Aggregated across all trees
- **Out-of-bag**: Built-in validation without holdout
- **Parallelizable**: Fast training on multi-core

The go-to ensemble method for tabular data.`,

	translations: {
		ru: {
			title: 'Случайный лес',
			description: `# Случайный лес

Случайный лес объединяет несколько деревьев решений для устойчивых предсказаний.

## Задача

Реализуйте три функции:
1. \`train_random_forest(X, y, n_estimators)\` - Обучить лес с n деревьями
2. \`get_oob_score(X, y, n_estimators)\` - Обучить с OOB и вернуть score
3. \`compare_with_single_tree(X, y)\` - Сравнить RF и одно дерево

## Пример

\`\`\`python
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(n_samples=100, n_features=10)

model = train_random_forest(X, y, n_estimators=100)
oob = get_oob_score(X, y, n_estimators=100)  # ~0.85
comparison = compare_with_single_tree(X, y)  # {'tree': 0.8, 'forest': 0.9}
\`\`\``,
			hint1: 'Используйте RandomForestClassifier(n_estimators=n, oob_score=True)',
			hint2: 'Используйте cross_val_score для честного сравнения моделей',
			whyItMatters: `Случайный лес мощный потому что:

- **Устойчивость**: Снижает переобучение через усреднение
- **Важность признаков**: Агрегированная по всем деревьям
- **Out-of-bag**: Встроенная валидация без holdout`,
		},
		uz: {
			title: "Tasodifiy o'rmon",
			description: `# Tasodifiy o'rmon

Tasodifiy o'rmon mustahkam bashoratlar uchun bir nechta qaror daraxtlarini birlashtiradi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`train_random_forest(X, y, n_estimators)\` - n ta daraxt bilan o'rmonni o'rgatish
2. \`get_oob_score(X, y, n_estimators)\` - OOB bilan o'rgatish va ballni qaytarish
3. \`compare_with_single_tree(X, y)\` - RF va bitta daraxtni taqqoslash

## Misol

\`\`\`python
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(n_samples=100, n_features=10)

model = train_random_forest(X, y, n_estimators=100)
oob = get_oob_score(X, y, n_estimators=100)  # ~0.85
comparison = compare_with_single_tree(X, y)  # {'tree': 0.8, 'forest': 0.9}
\`\`\``,
			hint1: "RandomForestClassifier(n_estimators=n, oob_score=True) dan foydalaning",
			hint2: "Modellarni adolatli taqqoslash uchun cross_val_score dan foydalaning",
			whyItMatters: `Tasodifiy o'rmon kuchli chunki:

- **Mustahkamlik**: O'rtachalash orqali ortiqcha moslashishni kamaytiradi
- **Xususiyat ahamiyati**: Barcha daraxtlar bo'yicha jamlangan
- **Out-of-bag**: Holdout siz o'rnatilgan validatsiya`,
		},
	},
};

export default task;
