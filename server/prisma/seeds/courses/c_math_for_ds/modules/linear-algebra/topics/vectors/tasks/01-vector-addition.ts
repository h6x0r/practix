import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'math-vector-addition',
	title: 'Vector Addition',
	difficulty: 'easy',
	tags: ['python', 'math', 'linear-algebra', 'vectors', 'numpy'],
	estimatedTime: '15m',
	isPremium: false,
	order: 1,
	description: `# Vector Addition

Implement a function that adds two vectors element-wise using NumPy.

## Background

Vector addition is a fundamental operation in linear algebra. When you add two vectors, you add their corresponding elements:

\`\`\`
[a₁, a₂, a₃] + [b₁, b₂, b₃] = [a₁+b₁, a₂+b₂, a₃+b₃]
\`\`\`

## Requirements

- Implement \`add_vectors(v1, v2)\` that returns the sum of two vectors
- Input vectors are NumPy arrays of the same shape
- Return a NumPy array

## Example

\`\`\`python
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
result = add_vectors(v1, v2)  # [5, 7, 9]
\`\`\`
`,
	initialCode: `import numpy as np

def add_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Add two vectors element-wise.

    Args:
        v1: First vector (numpy array)
        v2: Second vector (numpy array)

    Returns:
        Sum of the two vectors
    """
    # Your code here
    pass
`,
	solutionCode: `import numpy as np

def add_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Add two vectors element-wise.

    Args:
        v1: First vector (numpy array)
        v2: Second vector (numpy array)

    Returns:
        Sum of the two vectors
    """
    return v1 + v2
`,
	testCode: `import unittest
import numpy as np

def assert_array_close(actual, expected, msg=""):
    """Helper for array comparison with clear error message"""
    if not np.allclose(actual, expected):
        raise AssertionError(f"Expected {expected.tolist()}, got {actual.tolist()}")

class TestVectorAddition(unittest.TestCase):
    def test_simple_addition(self):
        # v1 = [1, 2, 3], v2 = [4, 5, 6]
        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6])
        result = add_vectors(v1, v2)
        expected = np.array([5, 7, 9])
        assert_array_close(result, expected)

    def test_zeros(self):
        # v1 = [1, 2, 3], v2 = [0, 0, 0]
        v1 = np.array([1, 2, 3])
        v2 = np.array([0, 0, 0])
        result = add_vectors(v1, v2)
        expected = np.array([1, 2, 3])
        assert_array_close(result, expected)

    def test_negative_values(self):
        # v1 = [1, -2, 3], v2 = [-1, 2, -3]
        v1 = np.array([1, -2, 3])
        v2 = np.array([-1, 2, -3])
        result = add_vectors(v1, v2)
        expected = np.array([0, 0, 0])
        assert_array_close(result, expected)

    def test_float_vectors(self):
        # v1 = [1.5, 2.5, 3.5], v2 = [0.5, 0.5, 0.5]
        v1 = np.array([1.5, 2.5, 3.5])
        v2 = np.array([0.5, 0.5, 0.5])
        result = add_vectors(v1, v2)
        expected = np.array([2.0, 3.0, 4.0])
        assert_array_close(result, expected)

    def test_large_vectors(self):
        # v1 = [100, 200, 300, 400, 500], v2 = [1, 2, 3, 4, 5]
        v1 = np.array([100, 200, 300, 400, 500])
        v2 = np.array([1, 2, 3, 4, 5])
        result = add_vectors(v1, v2)
        expected = np.array([101, 202, 303, 404, 505])
        assert_array_close(result, expected)

    def test_2d_vector(self):
        # v1 = [10, 20], v2 = [5, 15]
        v1 = np.array([10, 20])
        v2 = np.array([5, 15])
        result = add_vectors(v1, v2)
        expected = np.array([15, 35])
        assert_array_close(result, expected)

    def test_single_element(self):
        # v1 = [42], v2 = [8]
        v1 = np.array([42])
        v2 = np.array([8])
        result = add_vectors(v1, v2)
        expected = np.array([50])
        assert_array_close(result, expected)

    def test_returns_numpy_array(self):
        # v1 = [1, 2], v2 = [3, 4]
        v1 = np.array([1, 2])
        v2 = np.array([3, 4])
        result = add_vectors(v1, v2)
        assert isinstance(result, np.ndarray), f"Expected numpy.ndarray, got {type(result).__name__}"

    def test_identity_property(self):
        # v1 = [7, 8, 9], v2 = [0, 0, 0]
        v1 = np.array([7, 8, 9])
        v2 = np.zeros(3)
        result = add_vectors(v1, v2)
        expected = v1
        assert_array_close(result, expected)

    def test_commutative_property(self):
        # Test that add_vectors(v1, v2) == add_vectors(v2, v1)
        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6])
        result1 = add_vectors(v1, v2)
        result2 = add_vectors(v2, v1)
        assert_array_close(result1, result2)

`,
	hint1: 'NumPy arrays support element-wise addition using the + operator.',
	hint2: 'Simply return v1 + v2 - NumPy handles the element-wise operation automatically.',
	whyItMatters: `Vector addition is everywhere in ML. When updating model weights during training, you add gradient vectors to weight vectors. In neural networks, the weighted sum of inputs is essentially vector addition followed by an activation function. **Production Pattern:** In recommendation systems, user preference vectors are often combined through addition to create aggregate profiles.`,
	translations: {
		ru: {
			title: 'Сложение векторов',
			description: `# Сложение векторов

Реализуйте функцию, которая складывает два вектора поэлементно с помощью NumPy.

## Теория

Сложение векторов - фундаментальная операция линейной алгебры. При сложении двух векторов складываются соответствующие элементы:

\`\`\`
[a₁, a₂, a₃] + [b₁, b₂, b₃] = [a₁+b₁, a₂+b₂, a₃+b₃]
\`\`\`

## Требования

- Реализуйте \`add_vectors(v1, v2)\`, возвращающую сумму двух векторов
- Входные векторы - NumPy массивы одинаковой формы
- Верните NumPy массив

## Пример

\`\`\`python
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
result = add_vectors(v1, v2)  # [5, 7, 9]
\`\`\`
`,
			hint1: 'NumPy массивы поддерживают поэлементное сложение с помощью оператора +.',
			hint2: 'Просто верните v1 + v2 - NumPy автоматически выполнит поэлементную операцию.',
			whyItMatters: `Сложение векторов повсюду в ML. При обновлении весов модели во время обучения вы добавляете векторы градиентов к векторам весов. **Production Pattern:** В рекомендательных системах векторы предпочтений пользователей часто объединяются через сложение.`,
		},
		uz: {
			title: 'Vektorlarni qo\'shish',
			description: `# Vektorlarni qo'shish

NumPy yordamida ikkita vektorni elementma-element qo'shadigan funksiyani amalga oshiring.

## Nazariya

Vektorlarni qo'shish chiziqli algebraning asosiy operatsiyasi. Ikki vektorni qo'shganda ularning mos elementlari qo'shiladi.

## Talablar

- \`add_vectors(v1, v2)\` funksiyasini amalga oshiring
- Kirish vektorlari bir xil shakldagi NumPy massivlari
- NumPy massivini qaytaring
`,
			hint1: 'NumPy massivlari + operatori yordamida elementma-element qo\'shishni qo\'llab-quvvatlaydi.',
			hint2: 'Oddiy v1 + v2 qaytaring - NumPy avtomatik ravishda elementma-element operatsiyani bajaradi.',
			whyItMatters: `Vektorlarni qo'shish ML ning hamma joyida uchraydi. Model og'irliklarini yangilashda gradient vektorlari og'irlik vektorlariga qo'shiladi.`,
		},
	},
};

export default task;
