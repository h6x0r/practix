import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'sklearn-logistic-regression',
	title: 'Logistic Regression',
	difficulty: 'medium',
	tags: ['sklearn', 'classification', 'logistic'],
	estimatedTime: '15m',
	isPremium: false,
	order: 3,
	description: `# Logistic Regression

Logistic regression is the fundamental algorithm for binary classification.

## Task

Implement three functions:
1. \`train_logistic(X, y)\` - Train logistic regression classifier
2. \`predict_proba(model, X)\` - Get class probabilities
3. \`evaluate_classifier(model, X, y)\` - Return accuracy, precision, recall

## Example

\`\`\`python
from sklearn.linear_model import LogisticRegression

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 0, 1, 1, 1])

model = train_logistic(X, y)
probas = predict_proba(model, X)  # [[0.9, 0.1], [0.7, 0.3], ...]
metrics = evaluate_classifier(model, X, y)  # {'accuracy': 0.8, ...}
\`\`\``,

	initialCode: `import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

def train_logistic(X: np.ndarray, y: np.ndarray):
    """Train logistic regression. Return fitted model."""
    # Your code here
    pass

def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """Get class probabilities. Return probabilities array."""
    # Your code here
    pass

def evaluate_classifier(model, X: np.ndarray, y: np.ndarray) -> dict:
    """Return {'accuracy': float, 'precision': float, 'recall': float}."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

def train_logistic(X: np.ndarray, y: np.ndarray):
    """Train logistic regression. Return fitted model."""
    model = LogisticRegression()
    model.fit(X, y)
    return model

def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """Get class probabilities. Return probabilities array."""
    return model.predict_proba(X)

def evaluate_classifier(model, X: np.ndarray, y: np.ndarray) -> dict:
    """Return {'accuracy': float, 'precision': float, 'recall': float}."""
    y_pred = model.predict(X)
    return {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0)
    }
`,

	testCode: `import numpy as np
import unittest

class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.array([[1, 2], [2, 3], [3, 1], [4, 5], [5, 6], [6, 2]])
        self.y = np.array([0, 0, 0, 1, 1, 1])

    def test_train_returns_model(self):
        model = train_logistic(self.X, self.y)
        self.assertIsNotNone(model)

    def test_model_has_classes(self):
        model = train_logistic(self.X, self.y)
        self.assertTrue(hasattr(model, 'classes_'))

    def test_predict_proba_shape(self):
        model = train_logistic(self.X, self.y)
        probas = predict_proba(model, self.X)
        self.assertEqual(probas.shape, (6, 2))

    def test_predict_proba_sums_to_one(self):
        model = train_logistic(self.X, self.y)
        probas = predict_proba(model, self.X)
        for row in probas:
            self.assertAlmostEqual(sum(row), 1.0, places=5)

    def test_evaluate_returns_dict(self):
        model = train_logistic(self.X, self.y)
        metrics = evaluate_classifier(model, self.X, self.y)
        self.assertIsInstance(metrics, dict)

    def test_evaluate_has_all_metrics(self):
        model = train_logistic(self.X, self.y)
        metrics = evaluate_classifier(model, self.X, self.y)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)

    def test_metrics_in_range(self):
        model = train_logistic(self.X, self.y)
        metrics = evaluate_classifier(model, self.X, self.y)
        for v in metrics.values():
            self.assertGreaterEqual(v, 0)
            self.assertLessEqual(v, 1)

    def test_predict_proba_returns_numpy(self):
        model = train_logistic(self.X, self.y)
        probas = predict_proba(model, self.X)
        self.assertIsInstance(probas, np.ndarray)

    def test_model_can_predict(self):
        model = train_logistic(self.X, self.y)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.y))

    def test_probas_in_valid_range(self):
        model = train_logistic(self.X, self.y)
        probas = predict_proba(model, self.X)
        self.assertTrue(np.all(probas >= 0))
        self.assertTrue(np.all(probas <= 1))
`,

	hint1: 'Use LogisticRegression().fit(X, y) for training',
	hint2: 'Use model.predict_proba(X) for probabilities, sklearn.metrics for evaluation',

	whyItMatters: `Logistic regression is essential because:

- **Probability outputs**: Get confidence scores, not just labels
- **Interpretability**: Coefficients show feature importance
- **Baseline**: Standard comparison for classification
- **Production ready**: Fast, simple, reliable

The workhorse of binary classification.`,

	translations: {
		ru: {
			title: 'Логистическая регрессия',
			description: `# Логистическая регрессия

Логистическая регрессия - фундаментальный алгоритм для бинарной классификации.

## Задача

Реализуйте три функции:
1. \`train_logistic(X, y)\` - Обучить классификатор логистической регрессии
2. \`predict_proba(model, X)\` - Получить вероятности классов
3. \`evaluate_classifier(model, X, y)\` - Вернуть accuracy, precision, recall

## Пример

\`\`\`python
from sklearn.linear_model import LogisticRegression

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 0, 1, 1, 1])

model = train_logistic(X, y)
probas = predict_proba(model, X)  # [[0.9, 0.1], [0.7, 0.3], ...]
metrics = evaluate_classifier(model, X, y)  # {'accuracy': 0.8, ...}
\`\`\``,
			hint1: 'Используйте LogisticRegression().fit(X, y) для обучения',
			hint2: 'Используйте model.predict_proba(X) для вероятностей',
			whyItMatters: `Логистическая регрессия необходима потому что:

- **Вероятностные выходы**: Получайте уверенность, а не только метки
- **Интерпретируемость**: Коэффициенты показывают важность признаков
- **Базовая модель**: Стандарт для сравнения классификаторов`,
		},
		uz: {
			title: 'Logistik regressiya',
			description: `# Logistik regressiya

Logistik regressiya ikkilik klassifikatsiya uchun asosiy algoritm.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`train_logistic(X, y)\` - Logistik regressiya klassifikatorini o'rgatish
2. \`predict_proba(model, X)\` - Klass ehtimolliklarini olish
3. \`evaluate_classifier(model, X, y)\` - accuracy, precision, recall qaytarish

## Misol

\`\`\`python
from sklearn.linear_model import LogisticRegression

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 0, 1, 1, 1])

model = train_logistic(X, y)
probas = predict_proba(model, X)  # [[0.9, 0.1], [0.7, 0.3], ...]
metrics = evaluate_classifier(model, X, y)  # {'accuracy': 0.8, ...}
\`\`\``,
			hint1: "O'rgatish uchun LogisticRegression().fit(X, y) dan foydalaning",
			hint2: "Ehtimolliklar uchun model.predict_proba(X) dan foydalaning",
			whyItMatters: `Logistik regressiya zarur chunki:

- **Ehtimollik chiqishlari**: Faqat teglar emas, ishonch ballari oling
- **Interpretatsiya**: Koeffitsientlar xususiyat ahamiyatini ko'rsatadi
- **Bazaviy model**: Klassifikatsiya uchun standart taqqoslash`,
		},
	},
};

export default task;
