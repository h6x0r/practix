import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'sklearn-naive-bayes',
	title: 'Naive Bayes',
	difficulty: 'medium',
	tags: ['sklearn', 'naive-bayes', 'probability'],
	estimatedTime: '12m',
	isPremium: false,
	order: 8,
	description: `# Naive Bayes

Naive Bayes uses Bayes theorem with feature independence assumption.

## Task

Implement three functions:
1. \`train_gaussian_nb(X, y)\` - Train Gaussian Naive Bayes
2. \`get_class_priors(model)\` - Return prior probabilities per class
3. \`compare_nb_types(X, y)\` - Compare Gaussian vs Multinomial NB

## Example

\`\`\`python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

X = np.array([[1, 2], [2, 3], [3, 1], [4, 5]])
y = np.array([0, 0, 1, 1])

model = train_gaussian_nb(X, y)
priors = get_class_priors(model)  # [0.5, 0.5]
comparison = compare_nb_types(X, y)  # {'gaussian': 0.9, 'multinomial': 0.8}
\`\`\``,

	initialCode: `import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

def train_gaussian_nb(X: np.ndarray, y: np.ndarray):
    """Train Gaussian Naive Bayes. Return fitted model."""
    # Your code here
    pass

def get_class_priors(model) -> np.ndarray:
    """Return prior probabilities per class."""
    # Your code here
    pass

def compare_nb_types(X: np.ndarray, y: np.ndarray) -> dict:
    """Compare Gaussian vs Multinomial NB. Return {'gaussian': score, 'multinomial': score}."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

def train_gaussian_nb(X: np.ndarray, y: np.ndarray):
    """Train Gaussian Naive Bayes. Return fitted model."""
    model = GaussianNB()
    model.fit(X, y)
    return model

def get_class_priors(model) -> np.ndarray:
    """Return prior probabilities per class."""
    return model.class_prior_

def compare_nb_types(X: np.ndarray, y: np.ndarray) -> dict:
    """Compare Gaussian vs Multinomial NB. Return {'gaussian': score, 'multinomial': score}."""
    gaussian = GaussianNB()
    gaussian_score = cross_val_score(gaussian, X, y, cv=5).mean()
    
    # Multinomial needs non-negative data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    multinomial = MultinomialNB()
    multinomial_score = cross_val_score(multinomial, X_scaled, y, cv=5).mean()
    
    return {'gaussian': gaussian_score, 'multinomial': multinomial_score}
`,

	testCode: `import numpy as np
import unittest

class TestNaiveBayes(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.abs(np.random.randn(100, 5))
        self.y = np.array([0] * 50 + [1] * 50)

    def test_train_returns_model(self):
        model = train_gaussian_nb(self.X, self.y)
        self.assertIsNotNone(model)

    def test_model_can_predict(self):
        model = train_gaussian_nb(self.X, self.y)
        pred = model.predict(self.X[[0]])
        self.assertEqual(len(pred), 1)

    def test_get_class_priors_shape(self):
        model = train_gaussian_nb(self.X, self.y)
        priors = get_class_priors(model)
        self.assertEqual(len(priors), 2)

    def test_priors_sum_to_one(self):
        model = train_gaussian_nb(self.X, self.y)
        priors = get_class_priors(model)
        self.assertAlmostEqual(sum(priors), 1.0, places=5)

    def test_compare_returns_dict(self):
        result = compare_nb_types(self.X, self.y)
        self.assertIsInstance(result, dict)
        self.assertIn('gaussian', result)
        self.assertIn('multinomial', result)

    def test_scores_in_range(self):
        result = compare_nb_types(self.X, self.y)
        for score in result.values():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)

    def test_priors_returns_numpy(self):
        model = train_gaussian_nb(self.X, self.y)
        priors = get_class_priors(model)
        self.assertIsInstance(priors, np.ndarray)

    def test_model_has_classes(self):
        model = train_gaussian_nb(self.X, self.y)
        self.assertTrue(hasattr(model, 'classes_'))
        self.assertEqual(len(model.classes_), 2)

    def test_predictions_shape(self):
        model = train_gaussian_nb(self.X, self.y)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.y))

    def test_balanced_priors(self):
        model = train_gaussian_nb(self.X, self.y)
        priors = get_class_priors(model)
        self.assertAlmostEqual(priors[0], 0.5, places=2)
        self.assertAlmostEqual(priors[1], 0.5, places=2)
`,

	hint1: 'Use GaussianNB().fit(X, y) for continuous data',
	hint2: 'MultinomialNB needs non-negative data, use MinMaxScaler',

	whyItMatters: `Naive Bayes is useful for:

- **Speed**: Very fast training and inference
- **Text classification**: Standard for spam detection
- **Probabilistic**: Outputs calibrated probabilities
- **Small data**: Works well with limited samples

Foundation for probabilistic machine learning.`,

	translations: {
		ru: {
			title: 'Наивный Байес',
			description: `# Наивный Байес

Наивный Байес использует теорему Байеса с предположением о независимости признаков.

## Задача

Реализуйте три функции:
1. \`train_gaussian_nb(X, y)\` - Обучить Gaussian Naive Bayes
2. \`get_class_priors(model)\` - Вернуть априорные вероятности классов
3. \`compare_nb_types(X, y)\` - Сравнить Gaussian и Multinomial NB

## Пример

\`\`\`python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

X = np.array([[1, 2], [2, 3], [3, 1], [4, 5]])
y = np.array([0, 0, 1, 1])

model = train_gaussian_nb(X, y)
priors = get_class_priors(model)  # [0.5, 0.5]
comparison = compare_nb_types(X, y)  # {'gaussian': 0.9, 'multinomial': 0.8}
\`\`\``,
			hint1: 'Используйте GaussianNB().fit(X, y) для непрерывных данных',
			hint2: 'MultinomialNB требует неотрицательные данные, используйте MinMaxScaler',
			whyItMatters: `Наивный Байес полезен для:

- **Скорость**: Очень быстрое обучение и инференс
- **Классификация текста**: Стандарт для обнаружения спама
- **Вероятностный**: Выдаёт калиброванные вероятности`,
		},
		uz: {
			title: 'Sodda Bayes',
			description: `# Sodda Bayes

Sodda Bayes xususiyatlarning mustaqilligi haqidagi taxmin bilan Bayes teoremasidan foydalanadi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`train_gaussian_nb(X, y)\` - Gaussian Sodda Bayesni o'rgatish
2. \`get_class_priors(model)\` - Sinflar uchun oldingi ehtimolliklarni qaytarish
3. \`compare_nb_types(X, y)\` - Gaussian va Multinomial NB ni taqqoslash

## Misol

\`\`\`python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

X = np.array([[1, 2], [2, 3], [3, 1], [4, 5]])
y = np.array([0, 0, 1, 1])

model = train_gaussian_nb(X, y)
priors = get_class_priors(model)  # [0.5, 0.5]
comparison = compare_nb_types(X, y)  # {'gaussian': 0.9, 'multinomial': 0.8}
\`\`\``,
			hint1: "Uzluksiz ma'lumotlar uchun GaussianNB().fit(X, y) dan foydalaning",
			hint2: "MultinomialNB manfiy bo'lmagan ma'lumotlarni talab qiladi, MinMaxScaler dan foydalaning",
			whyItMatters: `Sodda Bayes quyidagilar uchun foydali:

- **Tezlik**: Juda tez o'qitish va inferensiya
- **Matn klassifikatsiyasi**: Spam aniqlash uchun standart
- **Ehtimollik**: Kalibrlangan ehtimolliklarni chiqaradi`,
		},
	},
};

export default task;
