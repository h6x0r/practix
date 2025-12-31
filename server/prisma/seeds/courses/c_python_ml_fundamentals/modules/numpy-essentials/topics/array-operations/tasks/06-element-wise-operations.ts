import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'numpy-element-wise-operations',
	title: 'Element-wise Operations',
	difficulty: 'easy',
	tags: ['numpy', 'operations', 'vectorization'],
	estimatedTime: '12m',
	isPremium: false,
	order: 6,
	description: `# Element-wise Operations

Element-wise operations apply functions to each element independently. This is the basis of vectorized computing.

## Task

Implement four functions:
1. \`apply_relu(arr)\` - Apply ReLU activation (max(0, x))
2. \`apply_sigmoid(arr)\` - Apply sigmoid: 1 / (1 + exp(-x))
3. \`clip_values(arr, min_val, max_val)\` - Clip values to range
4. \`compute_softmax(arr)\` - Compute softmax along last axis

## Example

\`\`\`python
arr = np.array([-2, -1, 0, 1, 2])

apply_relu(arr)  # [0, 0, 0, 1, 2]
apply_sigmoid(arr)  # [0.12, 0.27, 0.5, 0.73, 0.88]
clip_values(arr, -1, 1)  # [-1, -1, 0, 1, 1]

logits = np.array([1.0, 2.0, 3.0])
compute_softmax(logits)  # [0.09, 0.24, 0.67] (sums to 1)
\`\`\``,

	initialCode: `import numpy as np

def apply_relu(arr: np.ndarray) -> np.ndarray:
    """Apply ReLU activation (max(0, x))."""
    # Your code here
    pass

def apply_sigmoid(arr: np.ndarray) -> np.ndarray:
    """Apply sigmoid: 1 / (1 + exp(-x))."""
    # Your code here
    pass

def clip_values(arr: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Clip values to range [min_val, max_val]."""
    # Your code here
    pass

def compute_softmax(arr: np.ndarray) -> np.ndarray:
    """Compute softmax along last axis."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def apply_relu(arr: np.ndarray) -> np.ndarray:
    """Apply ReLU activation (max(0, x))."""
    return np.maximum(0, arr)

def apply_sigmoid(arr: np.ndarray) -> np.ndarray:
    """Apply sigmoid: 1 / (1 + exp(-x))."""
    return 1 / (1 + np.exp(-arr))

def clip_values(arr: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Clip values to range [min_val, max_val]."""
    return np.clip(arr, min_val, max_val)

def compute_softmax(arr: np.ndarray) -> np.ndarray:
    """Compute softmax along last axis."""
    exp_arr = np.exp(arr - np.max(arr, axis=-1, keepdims=True))
    return exp_arr / np.sum(exp_arr, axis=-1, keepdims=True)
`,

	testCode: `import numpy as np
import unittest

class TestElementWiseOperations(unittest.TestCase):
    def test_relu_positive(self):
        arr = np.array([1, 2, 3])
        result = apply_relu(arr)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_relu_negative(self):
        arr = np.array([-1, -2, -3])
        result = apply_relu(arr)
        np.testing.assert_array_equal(result, [0, 0, 0])

    def test_relu_mixed(self):
        arr = np.array([-2, -1, 0, 1, 2])
        result = apply_relu(arr)
        np.testing.assert_array_equal(result, [0, 0, 0, 1, 2])

    def test_sigmoid_zero(self):
        arr = np.array([0.0])
        result = apply_sigmoid(arr)
        self.assertAlmostEqual(result[0], 0.5)

    def test_sigmoid_range(self):
        arr = np.array([-10.0, 0.0, 10.0])
        result = apply_sigmoid(arr)
        self.assertTrue(all(0 < r < 1 for r in result))

    def test_sigmoid_symmetry(self):
        result_pos = apply_sigmoid(np.array([2.0]))[0]
        result_neg = apply_sigmoid(np.array([-2.0]))[0]
        self.assertAlmostEqual(result_pos + result_neg, 1.0)

    def test_clip_basic(self):
        arr = np.array([-5, 0, 5, 10])
        result = clip_values(arr, 0, 5)
        np.testing.assert_array_equal(result, [0, 0, 5, 5])

    def test_clip_no_change(self):
        arr = np.array([1, 2, 3])
        result = clip_values(arr, 0, 10)
        np.testing.assert_array_equal(result, arr)

    def test_softmax_sum_to_one(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = compute_softmax(arr)
        self.assertAlmostEqual(np.sum(result), 1.0)

    def test_softmax_all_positive(self):
        arr = np.array([-1.0, 0.0, 1.0])
        result = compute_softmax(arr)
        self.assertTrue(all(r > 0 for r in result))

    def test_softmax_max_largest(self):
        arr = np.array([1.0, 2.0, 5.0])
        result = compute_softmax(arr)
        self.assertEqual(np.argmax(result), 2)

    def test_softmax_2d(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = compute_softmax(arr)
        self.assertEqual(result.shape, (2, 2))
        np.testing.assert_almost_equal(result.sum(axis=-1), [1.0, 1.0])
`,

	hint1: 'Use np.maximum(0, arr) for ReLU, np.clip for clipping',
	hint2: 'For numerical stability in softmax, subtract max before exp',

	whyItMatters: `Element-wise operations are the building blocks of neural networks:

- **Activation functions**: ReLU, sigmoid, tanh applied to layer outputs
- **Loss computation**: Element-wise comparisons and transformations
- **Gradient clipping**: Prevent exploding gradients
- **Probability outputs**: Softmax for classification

Vectorized implementations are 100x faster than Python loops.`,

	translations: {
		ru: {
			title: 'Поэлементные операции',
			description: `# Поэлементные операции

Поэлементные операции применяют функции к каждому элементу независимо. Это основа векторизованных вычислений.

## Задача

Реализуйте четыре функции:
1. \`apply_relu(arr)\` - Применить ReLU активацию (max(0, x))
2. \`apply_sigmoid(arr)\` - Применить sigmoid: 1 / (1 + exp(-x))
3. \`clip_values(arr, min_val, max_val)\` - Ограничить значения диапазоном
4. \`compute_softmax(arr)\` - Вычислить softmax по последней оси

## Пример

\`\`\`python
arr = np.array([-2, -1, 0, 1, 2])

apply_relu(arr)  # [0, 0, 0, 1, 2]
apply_sigmoid(arr)  # [0.12, 0.27, 0.5, 0.73, 0.88]
clip_values(arr, -1, 1)  # [-1, -1, 0, 1, 1]

logits = np.array([1.0, 2.0, 3.0])
compute_softmax(logits)  # [0.09, 0.24, 0.67] (сумма = 1)
\`\`\``,
			hint1: 'Используйте np.maximum(0, arr) для ReLU, np.clip для ограничения',
			hint2: 'Для численной стабильности softmax вычтите max перед exp',
			whyItMatters: `Поэлементные операции — строительные блоки нейросетей:

- **Функции активации**: ReLU, sigmoid, tanh применяются к выходам слоёв
- **Вычисление loss**: Поэлементные сравнения и преобразования
- **Gradient clipping**: Предотвращение взрывающихся градиентов
- **Вероятностные выходы**: Softmax для классификации`,
		},
		uz: {
			title: "Elementli operatsiyalar",
			description: `# Elementli operatsiyalar

Elementli operatsiyalar funksiyalarni har bir elementga mustaqil ravishda qo'llaydi. Bu vektorlashtirilgan hisoblashning asosidir.

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`apply_relu(arr)\` - ReLU aktivatsiyasini qo'llash (max(0, x))
2. \`apply_sigmoid(arr)\` - Sigmoid qo'llash: 1 / (1 + exp(-x))
3. \`clip_values(arr, min_val, max_val)\` - Qiymatlarni diapazon bilan cheklash
4. \`compute_softmax(arr)\` - Oxirgi o'q bo'ylab softmax hisoblash

## Misol

\`\`\`python
arr = np.array([-2, -1, 0, 1, 2])

apply_relu(arr)  # [0, 0, 0, 1, 2]
apply_sigmoid(arr)  # [0.12, 0.27, 0.5, 0.73, 0.88]
clip_values(arr, -1, 1)  # [-1, -1, 0, 1, 1]

logits = np.array([1.0, 2.0, 3.0])
compute_softmax(logits)  # [0.09, 0.24, 0.67] (yig'indisi 1)
\`\`\``,
			hint1: "ReLU uchun np.maximum(0, arr), cheklash uchun np.clip dan foydalaning",
			hint2: "Softmax da raqamli barqarorlik uchun exp dan oldin max ni ayiring",
			whyItMatters: `Elementli operatsiyalar neyron tarmoqlarning qurilish bloklaridir:

- **Aktivatsiya funksiyalari**: Qatlam chiqishlariga ReLU, sigmoid, tanh qo'llanadi
- **Loss hisoblash**: Elementli taqqoslashlar va o'zgartirishlar
- **Gradient clipping**: Portlovchi gradientlarni oldini olish
- **Ehtimollik chiqishlari**: Klassifikatsiya uchun softmax`,
		},
	},
};

export default task;
