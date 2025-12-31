import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'sklearn-svm',
	title: 'Support Vector Machines',
	difficulty: 'hard',
	tags: ['sklearn', 'svm', 'classification'],
	estimatedTime: '18m',
	isPremium: true,
	order: 6,
	description: `# Support Vector Machines

SVM finds optimal decision boundaries using kernel transformations.

## Task

Implement three functions:
1. \`train_svm(X, y, kernel)\` - Train SVM with specified kernel
2. \`get_support_vectors(model)\` - Return support vectors
3. \`compare_kernels(X, y, kernels)\` - Compare different kernels

## Example

\`\`\`python
from sklearn.svm import SVC

X, y = make_classification(n_samples=100)

model = train_svm(X, y, kernel='rbf')
sv = get_support_vectors(model)
comparison = compare_kernels(X, y, ['linear', 'rbf', 'poly'])
\`\`\``,

	initialCode: `import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

def train_svm(X: np.ndarray, y: np.ndarray, kernel: str = 'rbf'):
    """Train SVM classifier. Return fitted model."""
    # Your code here
    pass

def get_support_vectors(model) -> np.ndarray:
    """Return support vectors array."""
    # Your code here
    pass

def compare_kernels(X: np.ndarray, y: np.ndarray, kernels: list) -> dict:
    """Compare kernels using cross-validation. Return {kernel: score}."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

def train_svm(X: np.ndarray, y: np.ndarray, kernel: str = 'rbf'):
    """Train SVM classifier. Return fitted model."""
    model = SVC(kernel=kernel, random_state=42)
    model.fit(X, y)
    return model

def get_support_vectors(model) -> np.ndarray:
    """Return support vectors array."""
    return model.support_vectors_

def compare_kernels(X: np.ndarray, y: np.ndarray, kernels: list) -> dict:
    """Compare kernels using cross-validation. Return {kernel: score}."""
    scores = {}
    for kernel in kernels:
        model = SVC(kernel=kernel, random_state=42)
        score = cross_val_score(model, X, y, cv=5).mean()
        scores[kernel] = score
    return scores
`,

	testCode: `import numpy as np
from sklearn.datasets import make_classification
import unittest

class TestSVM(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification(n_samples=100, n_features=10, random_state=42)

    def test_train_returns_model(self):
        model = train_svm(self.X, self.y, 'rbf')
        self.assertIsNotNone(model)

    def test_model_has_support_vectors(self):
        model = train_svm(self.X, self.y, 'rbf')
        self.assertTrue(hasattr(model, 'support_vectors_'))

    def test_get_support_vectors_shape(self):
        model = train_svm(self.X, self.y, 'rbf')
        sv = get_support_vectors(model)
        self.assertEqual(sv.shape[1], 10)

    def test_compare_kernels_returns_dict(self):
        result = compare_kernels(self.X, self.y, ['linear', 'rbf'])
        self.assertIsInstance(result, dict)
        self.assertIn('linear', result)
        self.assertIn('rbf', result)

    def test_scores_in_range(self):
        result = compare_kernels(self.X, self.y, ['linear'])
        for score in result.values():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)

    def test_support_vectors_returns_numpy(self):
        model = train_svm(self.X, self.y, 'rbf')
        sv = get_support_vectors(model)
        self.assertIsInstance(sv, np.ndarray)

    def test_model_can_predict(self):
        model = train_svm(self.X, self.y, 'linear')
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.y))

    def test_different_kernels(self):
        model_linear = train_svm(self.X, self.y, 'linear')
        model_rbf = train_svm(self.X, self.y, 'rbf')
        self.assertIsNotNone(model_linear)
        self.assertIsNotNone(model_rbf)

    def test_compare_returns_all_kernels(self):
        kernels = ['linear', 'rbf', 'poly']
        result = compare_kernels(self.X, self.y, kernels)
        for kernel in kernels:
            self.assertIn(kernel, result)

    def test_model_has_classes(self):
        model = train_svm(self.X, self.y, 'rbf')
        self.assertTrue(hasattr(model, 'classes_'))
`,

	hint1: 'Use SVC(kernel=kernel).fit(X, y) for training',
	hint2: 'Access model.support_vectors_ for support vectors',

	whyItMatters: `SVM is valuable for:

- **Kernel trick**: Handle non-linear boundaries elegantly
- **Margin optimization**: Maximum separation between classes
- **Sparse solution**: Only support vectors matter
- **High dimensions**: Works well when features > samples

Powerful for complex classification problems.`,

	translations: {
		ru: {
			title: 'Метод опорных векторов',
			description: `# Метод опорных векторов

SVM находит оптимальные границы решений с помощью ядерных преобразований.

## Задача

Реализуйте три функции:
1. \`train_svm(X, y, kernel)\` - Обучить SVM с указанным ядром
2. \`get_support_vectors(model)\` - Вернуть опорные векторы
3. \`compare_kernels(X, y, kernels)\` - Сравнить разные ядра

## Пример

\`\`\`python
from sklearn.svm import SVC

X, y = make_classification(n_samples=100)

model = train_svm(X, y, kernel='rbf')
sv = get_support_vectors(model)
comparison = compare_kernels(X, y, ['linear', 'rbf', 'poly'])
\`\`\``,
			hint1: 'Используйте SVC(kernel=kernel).fit(X, y) для обучения',
			hint2: 'Доступ к model.support_vectors_ для опорных векторов',
			whyItMatters: `SVM ценен для:

- **Ядерный трюк**: Элегантная обработка нелинейных границ
- **Оптимизация отступа**: Максимальное разделение классов
- **Разреженное решение**: Важны только опорные векторы`,
		},
		uz: {
			title: "Qo'llab-quvvatlash vektorlari mashinasi",
			description: `# Qo'llab-quvvatlash vektorlari mashinasi

SVM yadro almashtirishlari yordamida optimal qaror chegaralarini topadi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`train_svm(X, y, kernel)\` - Ko'rsatilgan yadro bilan SVM o'rgatish
2. \`get_support_vectors(model)\` - Qo'llab-quvvatlash vektorlarini qaytarish
3. \`compare_kernels(X, y, kernels)\` - Turli yadrolarni taqqoslash

## Misol

\`\`\`python
from sklearn.svm import SVC

X, y = make_classification(n_samples=100)

model = train_svm(X, y, kernel='rbf')
sv = get_support_vectors(model)
comparison = compare_kernels(X, y, ['linear', 'rbf', 'poly'])
\`\`\``,
			hint1: "O'rgatish uchun SVC(kernel=kernel).fit(X, y) dan foydalaning",
			hint2: "Qo'llab-quvvatlash vektorlari uchun model.support_vectors_ ga kiring",
			whyItMatters: `SVM quyidagilar uchun qimmatli:

- **Yadro hiylasi**: Nochiziqli chegaralarni nafis qayta ishlash
- **Margin optimallashtirish**: Sinflar orasida maksimal ajratish
- **Siyrak yechim**: Faqat qo'llab-quvvatlash vektorlari muhim`,
		},
	},
};

export default task;
