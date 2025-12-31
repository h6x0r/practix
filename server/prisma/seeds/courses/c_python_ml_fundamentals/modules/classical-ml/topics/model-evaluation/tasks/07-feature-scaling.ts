import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'sklearn-feature-scaling',
	title: 'Feature Scaling',
	difficulty: 'easy',
	tags: ['sklearn', 'preprocessing', 'scaling'],
	estimatedTime: '10m',
	isPremium: false,
	order: 7,
	description: `# Feature Scaling

Normalize and standardize features for better model performance.

## Task

Implement three functions:
1. \`standardize(X_train, X_test)\` - Zero mean, unit variance (StandardScaler)
2. \`normalize(X_train, X_test)\` - Scale to [0, 1] range (MinMaxScaler)
3. \`robust_scale(X_train, X_test)\` - Robust to outliers (RobustScaler)

## Example

\`\`\`python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

X_train = np.array([[1, 2], [3, 4], [5, 6]])
X_test = np.array([[2, 3]])

X_train_scaled, X_test_scaled = standardize(X_train, X_test)
\`\`\``,

	initialCode: `import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def standardize(X_train: np.ndarray, X_test: np.ndarray):
    """StandardScaler: zero mean, unit variance. Return scaled train and test."""
    # Your code here
    pass

def normalize(X_train: np.ndarray, X_test: np.ndarray):
    """MinMaxScaler: scale to [0, 1]. Return scaled train and test."""
    # Your code here
    pass

def robust_scale(X_train: np.ndarray, X_test: np.ndarray):
    """RobustScaler: robust to outliers. Return scaled train and test."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def standardize(X_train: np.ndarray, X_test: np.ndarray):
    """StandardScaler: zero mean, unit variance. Return scaled train and test."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def normalize(X_train: np.ndarray, X_test: np.ndarray):
    """MinMaxScaler: scale to [0, 1]. Return scaled train and test."""
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def robust_scale(X_train: np.ndarray, X_test: np.ndarray):
    """RobustScaler: robust to outliers. Return scaled train and test."""
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
`,

	testCode: `import numpy as np
import unittest

class TestFeatureScaling(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X_train = np.random.randn(100, 5) * 10 + 50
        self.X_test = np.random.randn(20, 5) * 10 + 50

    def test_standardize_zero_mean(self):
        X_train_s, X_test_s = standardize(self.X_train, self.X_test)
        self.assertAlmostEqual(X_train_s.mean(), 0, places=1)

    def test_standardize_unit_variance(self):
        X_train_s, X_test_s = standardize(self.X_train, self.X_test)
        self.assertAlmostEqual(X_train_s.std(), 1, places=1)

    def test_normalize_range(self):
        X_train_s, X_test_s = normalize(self.X_train, self.X_test)
        self.assertGreaterEqual(X_train_s.min(), 0)
        self.assertLessEqual(X_train_s.max(), 1)

    def test_robust_scale_shape(self):
        X_train_s, X_test_s = robust_scale(self.X_train, self.X_test)
        self.assertEqual(X_train_s.shape, self.X_train.shape)
        self.assertEqual(X_test_s.shape, self.X_test.shape)

    def test_standardize_returns_numpy(self):
        X_train_s, X_test_s = standardize(self.X_train, self.X_test)
        self.assertIsInstance(X_train_s, np.ndarray)
        self.assertIsInstance(X_test_s, np.ndarray)

    def test_normalize_returns_numpy(self):
        X_train_s, X_test_s = normalize(self.X_train, self.X_test)
        self.assertIsInstance(X_train_s, np.ndarray)

    def test_standardize_shape_preserved(self):
        X_train_s, X_test_s = standardize(self.X_train, self.X_test)
        self.assertEqual(X_train_s.shape, self.X_train.shape)

    def test_normalize_test_may_exceed_range(self):
        X_train_s, X_test_s = normalize(self.X_train, self.X_test)
        self.assertEqual(X_test_s.shape, self.X_test.shape)

    def test_robust_scale_handles_outliers(self):
        X_with_outlier = self.X_train.copy()
        X_with_outlier[0, 0] = 1000
        X_train_s, X_test_s = robust_scale(X_with_outlier, self.X_test)
        self.assertIsNotNone(X_train_s)

    def test_all_scalers_same_feature_count(self):
        X1, _ = standardize(self.X_train, self.X_test)
        X2, _ = normalize(self.X_train, self.X_test)
        X3, _ = robust_scale(self.X_train, self.X_test)
        self.assertEqual(X1.shape[1], X2.shape[1])
        self.assertEqual(X2.shape[1], X3.shape[1])
`,

	hint1: 'Use fit_transform on training data, transform only on test data',
	hint2: 'StandardScaler(), MinMaxScaler(), RobustScaler() - same API pattern',

	whyItMatters: `Feature scaling is crucial for:

- **Distance-based algorithms**: KNN, SVM, K-means need scaled features
- **Gradient descent**: Faster convergence with normalized features
- **Regularization**: L1/L2 penalties work properly
- **Interpretability**: Compare feature importance fairly

Always fit on train, transform on test to avoid data leakage.`,

	translations: {
		ru: {
			title: 'Масштабирование признаков',
			description: `# Масштабирование признаков

Нормализуйте и стандартизируйте признаки для лучшей производительности модели.

## Задача

Реализуйте три функции:
1. \`standardize(X_train, X_test)\` - Нулевое среднее, единичная дисперсия
2. \`normalize(X_train, X_test)\` - Масштаб в диапазон [0, 1]
3. \`robust_scale(X_train, X_test)\` - Устойчив к выбросам

## Пример

\`\`\`python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

X_train = np.array([[1, 2], [3, 4], [5, 6]])
X_test = np.array([[2, 3]])

X_train_scaled, X_test_scaled = standardize(X_train, X_test)
\`\`\``,
			hint1: 'Используйте fit_transform на обучающих данных, transform на тестовых',
			hint2: 'StandardScaler(), MinMaxScaler(), RobustScaler() - одинаковый API',
			whyItMatters: `Масштабирование признаков важно для:

- **Алгоритмы на расстояниях**: KNN, SVM, K-means требуют масштабирования
- **Градиентный спуск**: Быстрая сходимость с нормализацией
- **Регуляризация**: L1/L2 работают корректно`,
		},
		uz: {
			title: "Xususiyatlarni masshtablash",
			description: `# Xususiyatlarni masshtablash

Model samaradorligini oshirish uchun xususiyatlarni normallash va standartlashtirish.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`standardize(X_train, X_test)\` - Nol o'rtacha, birlik dispersiya
2. \`normalize(X_train, X_test)\` - [0, 1] oralig'iga masshtablash
3. \`robust_scale(X_train, X_test)\` - Chekinishlarga chidamli

## Misol

\`\`\`python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

X_train = np.array([[1, 2], [3, 4], [5, 6]])
X_test = np.array([[2, 3]])

X_train_scaled, X_test_scaled = standardize(X_train, X_test)
\`\`\``,
			hint1: "O'qitish ma'lumotlarida fit_transform, testda faqat transform ishlating",
			hint2: "StandardScaler(), MinMaxScaler(), RobustScaler() - bir xil API namunasi",
			whyItMatters: `Xususiyatlarni masshtablash quyidagilar uchun muhim:

- **Masofaga asoslangan algoritmlar**: KNN, SVM, K-means masshtablangan xususiyatlarni talab qiladi
- **Gradient tushish**: Normallashtirilgan xususiyatlar bilan tezroq yaqinlashish
- **Regulyarizatsiya**: L1/L2 jarimalar to'g'ri ishlaydi`,
		},
	},
};

export default task;
