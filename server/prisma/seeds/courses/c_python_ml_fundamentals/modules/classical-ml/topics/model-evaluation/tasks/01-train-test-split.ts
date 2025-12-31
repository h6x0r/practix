import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'sklearn-train-test-split',
	title: 'Train-Test Split',
	difficulty: 'easy',
	tags: ['sklearn', 'validation', 'split'],
	estimatedTime: '10m',
	isPremium: false,
	order: 1,
	description: `# Train-Test Split

Split data into training and testing sets for unbiased evaluation.

## Task

Implement three functions:
1. \`basic_split(X, y, test_size)\` - Simple train-test split
2. \`stratified_split(X, y, test_size)\` - Stratified split for classification
3. \`time_series_split(X, y, test_size)\` - Preserve temporal order

## Example

\`\`\`python
from sklearn.model_selection import train_test_split

X, y = np.random.randn(100, 5), np.array([0]*50 + [1]*50)

X_train, X_test, y_train, y_test = basic_split(X, y, 0.2)
X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2)
\`\`\``,

	initialCode: `import numpy as np
from sklearn.model_selection import train_test_split

def basic_split(X: np.ndarray, y: np.ndarray, test_size: float):
    """Simple train-test split. Return X_train, X_test, y_train, y_test."""
    # Your code here
    pass

def stratified_split(X: np.ndarray, y: np.ndarray, test_size: float):
    """Stratified split preserving class proportions. Return splits."""
    # Your code here
    pass

def time_series_split(X: np.ndarray, y: np.ndarray, test_size: float):
    """Split preserving temporal order (no shuffle). Return splits."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
from sklearn.model_selection import train_test_split

def basic_split(X: np.ndarray, y: np.ndarray, test_size: float):
    """Simple train-test split. Return X_train, X_test, y_train, y_test."""
    return train_test_split(X, y, test_size=test_size, random_state=42)

def stratified_split(X: np.ndarray, y: np.ndarray, test_size: float):
    """Stratified split preserving class proportions. Return splits."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

def time_series_split(X: np.ndarray, y: np.ndarray, test_size: float):
    """Split preserving temporal order (no shuffle). Return splits."""
    return train_test_split(X, y, test_size=test_size, shuffle=False)
`,

	testCode: `import numpy as np
import unittest

class TestTrainTestSplit(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.array([0]*50 + [1]*50)

    def test_basic_split_sizes(self):
        X_train, X_test, y_train, y_test = basic_split(self.X, self.y, 0.2)
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)

    def test_stratified_preserves_proportions(self):
        X_train, X_test, y_train, y_test = stratified_split(self.X, self.y, 0.2)
        test_ratio = sum(y_test == 1) / len(y_test)
        self.assertAlmostEqual(test_ratio, 0.5, places=1)

    def test_time_series_no_shuffle(self):
        X_train, X_test, y_train, y_test = time_series_split(self.X, self.y, 0.2)
        # Last 20% should be test set
        self.assertEqual(len(X_test), 20)

    def test_basic_split_returns_four_arrays(self):
        result = basic_split(self.X, self.y, 0.2)
        self.assertEqual(len(result), 4)

    def test_basic_split_30_percent(self):
        X_train, X_test, y_train, y_test = basic_split(self.X, self.y, 0.3)
        self.assertEqual(len(X_test), 30)
        self.assertEqual(len(X_train), 70)

    def test_stratified_train_proportions(self):
        X_train, X_test, y_train, y_test = stratified_split(self.X, self.y, 0.2)
        train_ratio = sum(y_train == 1) / len(y_train)
        self.assertAlmostEqual(train_ratio, 0.5, places=1)

    def test_time_series_preserves_order(self):
        X_ordered = np.arange(100).reshape(-1, 1)
        y_ordered = np.arange(100)
        X_train, X_test, y_train, y_test = time_series_split(X_ordered, y_ordered, 0.2)
        self.assertTrue(np.all(X_test[:, 0] >= X_train[:, 0].max()))

    def test_basic_split_preserves_features(self):
        X_train, X_test, y_train, y_test = basic_split(self.X, self.y, 0.2)
        self.assertEqual(X_train.shape[1], 5)
        self.assertEqual(X_test.shape[1], 5)

    def test_stratified_with_imbalanced(self):
        y_imbalanced = np.array([0]*90 + [1]*10)
        X_train, X_test, y_train, y_test = stratified_split(self.X, y_imbalanced, 0.2)
        test_ratio = sum(y_test == 1) / len(y_test)
        self.assertAlmostEqual(test_ratio, 0.1, places=1)

    def test_returns_numpy_arrays(self):
        X_train, X_test, y_train, y_test = basic_split(self.X, self.y, 0.2)
        self.assertIsInstance(X_train, np.ndarray)
        self.assertIsInstance(y_test, np.ndarray)
`,

	hint1: 'Use train_test_split(X, y, test_size=0.2) from sklearn.model_selection',
	hint2: 'Use stratify=y for stratified, shuffle=False for time series',

	whyItMatters: `Proper splitting is essential for:

- **Unbiased evaluation**: Test on unseen data
- **Class balance**: Stratified for imbalanced data
- **Time series**: Prevent data leakage
- **Reproducibility**: Random state for consistency

Foundation of reliable ML evaluation.`,

	translations: {
		ru: {
			title: 'Разделение данных',
			description: `# Разделение данных

Разделите данные на обучающую и тестовую выборки для объективной оценки.

## Задача

Реализуйте три функции:
1. \`basic_split(X, y, test_size)\` - Простое разделение
2. \`stratified_split(X, y, test_size)\` - Стратифицированное для классификации
3. \`time_series_split(X, y, test_size)\` - С сохранением порядка времени

## Пример

\`\`\`python
from sklearn.model_selection import train_test_split

X, y = np.random.randn(100, 5), np.array([0]*50 + [1]*50)

X_train, X_test, y_train, y_test = basic_split(X, y, 0.2)
X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2)
\`\`\``,
			hint1: 'Используйте train_test_split(X, y, test_size=0.2)',
			hint2: 'Используйте stratify=y для стратифицированного, shuffle=False для временных рядов',
			whyItMatters: `Правильное разделение необходимо для:

- **Объективная оценка**: Тестирование на невиденных данных
- **Баланс классов**: Стратификация для несбалансированных данных
- **Временные ряды**: Предотвращение утечки данных`,
		},
		uz: {
			title: "Ma'lumotlarni ajratish",
			description: `# Ma'lumotlarni ajratish

Xolisona baholash uchun ma'lumotlarni o'rgatish va test to'plamlariga ajrating.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`basic_split(X, y, test_size)\` - Oddiy ajratish
2. \`stratified_split(X, y, test_size)\` - Klassifikatsiya uchun stratifitsiyalangan
3. \`time_series_split(X, y, test_size)\` - Vaqt tartibini saqlash

## Misol

\`\`\`python
from sklearn.model_selection import train_test_split

X, y = np.random.randn(100, 5), np.array([0]*50 + [1]*50)

X_train, X_test, y_train, y_test = basic_split(X, y, 0.2)
X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2)
\`\`\``,
			hint1: "train_test_split(X, y, test_size=0.2) dan foydalaning",
			hint2: "Stratifitsiyalangan uchun stratify=y, vaqt qatorlari uchun shuffle=False dan foydalaning",
			whyItMatters: `To'g'ri ajratish quyidagilar uchun zarur:

- **Xolisona baholash**: Ko'rilmagan ma'lumotlarda test qilish
- **Sinf muvozanati**: Nomutanosib ma'lumotlar uchun stratifikatsiya
- **Vaqt qatorlari**: Ma'lumotlar oqishini oldini olish`,
		},
	},
};

export default task;
