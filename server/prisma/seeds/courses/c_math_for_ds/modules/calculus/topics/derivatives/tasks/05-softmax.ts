import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'math-softmax',
	title: 'Softmax Function',
	difficulty: 'medium',
	tags: ['python', 'math', 'calculus', 'softmax', 'classification', 'numpy'],
	estimatedTime: '20m',
	isPremium: true,
	order: 5,
	description: `# Softmax Function

Implement the softmax function that converts logits to probabilities.

## Background

Softmax(xᵢ) = e^xᵢ / Σⱼ e^xⱼ

- Output: probability distribution (sums to 1)
- Input: raw scores (logits)
- Numerical stability: subtract max(x) before exp

## Example

\`\`\`python
logits = [2.0, 1.0, 0.1]
softmax(logits)  # [0.659, 0.242, 0.099]
\`\`\`
`,
	initialCode: `import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities.

    Args:
        x: Input logits (1D array)

    Returns:
        Probability distribution (sums to 1)
    """
    # Your code here
    pass
`,
	solutionCode: `import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities.

    Args:
        x: Input logits (1D array)

    Returns:
        Probability distribution (sums to 1)
    """
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)
`,
	testCode: `import unittest
import numpy as np

def assert_array_close(actual, expected, msg=""):
    """Helper for array comparison with clear error message"""
    if not np.allclose(actual, expected):
        raise AssertionError(f"Expected {expected.tolist()}, got {actual.tolist()}")

def assert_close(actual, expected, places=5, msg=""):
    """Helper for scalar comparison with clear error message"""
    if abs(actual - expected) > 10**(-places):
        raise AssertionError(f"Expected {expected}, got {actual}")

class TestSoftmax(unittest.TestCase):
    def test_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        self.assertAlmostEqual(np.sum(result), 1.0)

    def test_all_positive(self):
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        assert np.all(result > 0)

    def test_ordering_preserved(self):
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        assert result[0] < result[1] < result[2]

    def test_equal_inputs(self):
        x = np.array([1.0, 1.0, 1.0])
        result = softmax(x)
        assert_array_close(result, [1/3, 1/3, 1/3])

    def test_numerical_stability(self):
        x = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(x)
        self.assertAlmostEqual(np.sum(result), 1.0)
        self.assertFalse(np.any(np.isnan(result)))

    def test_negative_inputs(self):
        x = np.array([-1.0, -2.0, -3.0])
        result = softmax(x)
        self.assertAlmostEqual(np.sum(result), 1.0)

    def test_single_element(self):
        x = np.array([5.0])
        result = softmax(x)
        assert_close(result[0], 1.0)

    def test_large_difference(self):
        x = np.array([0.0, 100.0])
        result = softmax(x)
        assert_close(result[1], 1.0)

    def test_returns_numpy_array(self):
        x = np.array([1.0, 2.0])
        result = softmax(x)
        assert isinstance(result, np.ndarray), f"Expected numpy.ndarray, got {type(result).__name__}"

    def test_example_values(self):
        x = np.array([2.0, 1.0, 0.1])
        result = softmax(x)
        assert 0.6 < result[0] < 0.7
        assert 0.2 < result[1] < 0.3

`,
	hint1: 'Use exp(x - max(x)) for numerical stability before dividing by sum.',
	hint2: 'Subtracting max(x) prevents overflow while giving same result.',
	whyItMatters: `Softmax is the output layer of every classification neural network. It converts raw scores to probabilities. **Production Pattern:** Combined with cross-entropy loss, softmax enables multi-class classification in transformers, CNNs, and all modern architectures.`,
	translations: {
		ru: {
			title: 'Функция Softmax',
			description: `# Функция Softmax

Реализуйте softmax, преобразующий логиты в вероятности.

## Теория

Softmax(xᵢ) = e^xᵢ / Σⱼ e^xⱼ

- Выход: распределение вероятностей (сумма = 1)
- Вход: сырые оценки (логиты)
- Численная стабильность: вычтите max(x) перед exp
`,
			hint1: 'Используйте exp(x - max(x)) для численной стабильности.',
			hint2: 'Вычитание max(x) предотвращает переполнение.',
			whyItMatters: `Softmax - выходной слой каждой классификационной нейросети. **Production Pattern:** В комбинации с cross-entropy loss softmax обеспечивает многоклассовую классификацию.`,
		},
		uz: {
			title: 'Softmax funksiyasi',
			description: `# Softmax funksiyasi

Logitlarni ehtimolliklarga aylantiradigan softmax ni amalga oshiring.
`,
			hint1: 'Raqamli barqarorlik uchun exp(x - max(x)) dan foydalaning.',
			hint2: 'max(x) ni ayirish to\'lib ketishni oldini oladi.',
			whyItMatters: `Softmax har bir klassifikatsiya neyron tarmog'ining chiqish qatlami. Cross-entropy yo'qotish bilan birgalikda ko'p sinfli klassifikatsiyani ta'minlaydi.`,
		},
	},
};

export default task;
