import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'sklearn-learning-curves',
	title: 'Learning Curves',
	difficulty: 'medium',
	tags: ['sklearn', 'diagnostics', 'curves'],
	estimatedTime: '12m',
	isPremium: false,
	order: 9,
	description: `# Learning Curves

Diagnose model performance with learning and validation curves.

## Task

Implement three functions:
1. \`compute_learning_curve(model, X, y)\` - Return train_sizes, train_scores, test_scores
2. \`compute_validation_curve(model, X, y, param_name, param_range)\` - Validation curve data
3. \`diagnose_bias_variance(train_scores, test_scores)\` - Return 'high_bias', 'high_variance', or 'good_fit'

## Example

\`\`\`python
from sklearn.model_selection import learning_curve, validation_curve

model = SVC()
train_sizes, train_scores, test_scores = compute_learning_curve(model, X, y)

# Diagnose: high bias (underfitting) or high variance (overfitting)
diagnosis = diagnose_bias_variance(train_scores, test_scores)
\`\`\``,

	initialCode: `import numpy as np
from sklearn.model_selection import learning_curve, validation_curve

def compute_learning_curve(model, X: np.ndarray, y: np.ndarray):
    """Compute learning curve. Return train_sizes, train_scores, test_scores."""
    # Your code here
    pass

def compute_validation_curve(model, X: np.ndarray, y: np.ndarray, 
                             param_name: str, param_range):
    """Compute validation curve. Return train_scores, test_scores."""
    # Your code here
    pass

def diagnose_bias_variance(train_scores: np.ndarray, test_scores: np.ndarray) -> str:
    """Diagnose model. Return 'high_bias', 'high_variance', or 'good_fit'."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
from sklearn.model_selection import learning_curve, validation_curve

def compute_learning_curve(model, X: np.ndarray, y: np.ndarray):
    """Compute learning curve. Return train_sizes, train_scores, test_scores."""
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    return train_sizes, train_scores, test_scores

def compute_validation_curve(model, X: np.ndarray, y: np.ndarray, 
                             param_name: str, param_range):
    """Compute validation curve. Return train_scores, test_scores."""
    train_scores, test_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range,
        cv=5, scoring='accuracy'
    )
    return train_scores, test_scores

def diagnose_bias_variance(train_scores: np.ndarray, test_scores: np.ndarray) -> str:
    """Diagnose model. Return 'high_bias', 'high_variance', or 'good_fit'."""
    train_mean = train_scores.mean()
    test_mean = test_scores.mean()
    gap = train_mean - test_mean
    
    if train_mean < 0.7 and test_mean < 0.7:
        return 'high_bias'
    elif gap > 0.15:
        return 'high_variance'
    else:
        return 'good_fit'
`,

	testCode: `import numpy as np
import unittest
from sklearn.svm import SVC

class TestLearningCurves(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.array([0]*50 + [1]*50)
        self.model = SVC()

    def test_learning_curve_returns_three(self):
        result = compute_learning_curve(self.model, self.X, self.y)
        self.assertEqual(len(result), 3)

    def test_learning_curve_sizes_match(self):
        sizes, train_s, test_s = compute_learning_curve(self.model, self.X, self.y)
        self.assertEqual(len(sizes), train_s.shape[0])

    def test_validation_curve_returns_two(self):
        train_s, test_s = compute_validation_curve(
            self.model, self.X, self.y, 'C', [0.1, 1, 10]
        )
        self.assertEqual(train_s.shape[0], 3)

    def test_diagnose_returns_string(self):
        train_s = np.array([[0.9, 0.85, 0.88]])
        test_s = np.array([[0.75, 0.78, 0.72]])
        result = diagnose_bias_variance(train_s, test_s)
        self.assertIn(result, ['high_bias', 'high_variance', 'good_fit'])

    def test_diagnose_high_bias(self):
        train_s = np.array([[0.5, 0.55]])
        test_s = np.array([[0.48, 0.52]])
        result = diagnose_bias_variance(train_s, test_s)
        self.assertEqual(result, 'high_bias')

    def test_diagnose_high_variance(self):
        train_s = np.array([[0.98, 0.99, 0.97]])
        test_s = np.array([[0.65, 0.68, 0.62]])
        result = diagnose_bias_variance(train_s, test_s)
        self.assertEqual(result, 'high_variance')

    def test_diagnose_good_fit(self):
        train_s = np.array([[0.88, 0.90, 0.87]])
        test_s = np.array([[0.82, 0.85, 0.80]])
        result = diagnose_bias_variance(train_s, test_s)
        self.assertEqual(result, 'good_fit')

    def test_learning_curve_scores_are_numpy(self):
        sizes, train_s, test_s = compute_learning_curve(self.model, self.X, self.y)
        self.assertIsInstance(train_s, np.ndarray)
        self.assertIsInstance(test_s, np.ndarray)

    def test_validation_curve_shapes_match(self):
        train_s, test_s = compute_validation_curve(
            self.model, self.X, self.y, 'C', [0.1, 1, 10, 100]
        )
        self.assertEqual(train_s.shape[0], test_s.shape[0])

    def test_learning_curve_train_scores_2d(self):
        sizes, train_s, test_s = compute_learning_curve(self.model, self.X, self.y)
        self.assertEqual(len(train_s.shape), 2)
`,

	hint1: 'learning_curve returns (train_sizes, train_scores, test_scores)',
	hint2: 'High bias: both scores low. High variance: large gap between train and test',

	whyItMatters: `Learning curves are diagnostic tools for:

- **Bias-variance tradeoff**: Identify underfitting vs overfitting
- **Data requirements**: Do you need more training data?
- **Model complexity**: Is your model too simple or complex?
- **Hyperparameter tuning**: Validation curves show optimal values

Essential for systematic model improvement.`,

	translations: {
		ru: {
			title: 'Кривые обучения',
			description: `# Кривые обучения

Диагностируйте производительность модели с помощью кривых обучения и валидации.

## Задача

Реализуйте три функции:
1. \`compute_learning_curve(model, X, y)\` - Вернуть train_sizes, train_scores, test_scores
2. \`compute_validation_curve(model, X, y, param_name, param_range)\` - Кривая валидации
3. \`diagnose_bias_variance(train_scores, test_scores)\` - Вернуть диагноз

## Пример

\`\`\`python
from sklearn.model_selection import learning_curve, validation_curve

model = SVC()
train_sizes, train_scores, test_scores = compute_learning_curve(model, X, y)

# Diagnose: high bias (underfitting) or high variance (overfitting)
diagnosis = diagnose_bias_variance(train_scores, test_scores)
\`\`\``,
			hint1: 'learning_curve возвращает (train_sizes, train_scores, test_scores)',
			hint2: 'Высокое смещение: обе оценки низкие. Высокая дисперсия: большой разрыв',
			whyItMatters: `Кривые обучения - диагностические инструменты для:

- **Компромисс смещение-дисперсия**: Определить недо/переобучение
- **Требования к данным**: Нужно ли больше данных?
- **Сложность модели**: Модель слишком простая или сложная?`,
		},
		uz: {
			title: "O'rganish egri chiziqlari",
			description: `# O'rganish egri chiziqlari

O'rganish va validatsiya egri chiziqlari yordamida model samaradorligini diagnostika qiling.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`compute_learning_curve(model, X, y)\` - train_sizes, train_scores, test_scores qaytarish
2. \`compute_validation_curve(model, X, y, param_name, param_range)\` - Validatsiya egri chizig'i
3. \`diagnose_bias_variance(train_scores, test_scores)\` - Diagnostikani qaytarish

## Misol

\`\`\`python
from sklearn.model_selection import learning_curve, validation_curve

model = SVC()
train_sizes, train_scores, test_scores = compute_learning_curve(model, X, y)

# Diagnose: high bias (underfitting) or high variance (overfitting)
diagnosis = diagnose_bias_variance(train_scores, test_scores)
\`\`\``,
			hint1: "learning_curve (train_sizes, train_scores, test_scores) qaytaradi",
			hint2: "Yuqori bias: ikkala ball past. Yuqori variance: train va test orasida katta farq",
			whyItMatters: `O'rganish egri chiziqlari quyidagilar uchun diagnostika vositalari:

- **Bias-variance o'zaro munosabati**: Kam/ortiqcha moslanishni aniqlash
- **Ma'lumot talablari**: Ko'proq o'qitish ma'lumotlari kerakmi?
- **Model murakkabligi**: Model juda oddiy yoki murakkabmi?`,
		},
	},
};

export default task;
