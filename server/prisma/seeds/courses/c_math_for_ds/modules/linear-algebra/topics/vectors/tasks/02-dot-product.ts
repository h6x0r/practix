import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'math-dot-product',
	title: 'Dot Product',
	difficulty: 'easy',
	tags: ['python', 'math', 'linear-algebra', 'vectors', 'numpy'],
	estimatedTime: '15m',
	isPremium: false,
	order: 2,
	description: `# Dot Product

Implement a function that computes the dot product of two vectors.

## Background

The dot product (scalar product) of two vectors is:
- Sum of products of corresponding elements
- Result is a scalar (single number), not a vector

\`\`\`
a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ
\`\`\`

## Use Cases in ML

- **Similarity**: Dot product measures how similar two vectors are
- **Neural Networks**: Weighted sum of inputs uses dot product
- **Attention Mechanisms**: Query-key dot products in transformers

## Example

\`\`\`python
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
result = dot_product(v1, v2)  # 1*4 + 2*5 + 3*6 = 32
\`\`\`
`,
	initialCode: `import numpy as np

def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the dot product of two vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Dot product (scalar value)
    """
    # Your code here
    pass
`,
	solutionCode: `import numpy as np

def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the dot product of two vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Dot product (scalar value)
    """
    return np.dot(v1, v2)
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

class TestDotProduct(unittest.TestCase):
    def test_simple_dot_product(self):
        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6])
        result = dot_product(v1, v2)
        assert_close(result, 32)

    def test_orthogonal_vectors(self):
        v1 = np.array([1, 0])
        v2 = np.array([0, 1])
        result = dot_product(v1, v2)
        assert_close(result, 0)

    def test_parallel_vectors(self):
        v1 = np.array([1, 2, 3])
        v2 = np.array([2, 4, 6])
        result = dot_product(v1, v2)
        assert_close(result, 28)

    def test_negative_values(self):
        v1 = np.array([1, -2, 3])
        v2 = np.array([-1, 2, -3])
        result = dot_product(v1, v2)
        assert_close(result, -14)

    def test_float_vectors(self):
        v1 = np.array([0.5, 1.5, 2.5])
        v2 = np.array([2.0, 2.0, 2.0])
        result = dot_product(v1, v2)
        assert_close(result, 9.0)

    def test_unit_vectors(self):
        v1 = np.array([1, 0, 0])
        v2 = np.array([1, 0, 0])
        result = dot_product(v1, v2)
        assert_close(result, 1)

    def test_same_vector(self):
        v = np.array([3, 4])
        result = dot_product(v, v)
        assert_close(result, 25)

    def test_zero_vector(self):
        v1 = np.array([1, 2, 3])
        v2 = np.array([0, 0, 0])
        result = dot_product(v1, v2)
        assert_close(result, 0)

    def test_large_vectors(self):
        v1 = np.array([1, 2, 3, 4, 5])
        v2 = np.array([5, 4, 3, 2, 1])
        result = dot_product(v1, v2)
        assert_close(result, 35)

    def test_returns_scalar(self):
        v1 = np.array([1, 2])
        v2 = np.array([3, 4])
        result = dot_product(v1, v2)
        assert np.isscalar(result or result.shape == ())

`,
	hint1: 'NumPy provides np.dot() function for computing dot products.',
	hint2: 'Alternatively, you can use (v1 * v2).sum() or the @ operator for matrix multiplication.',
	whyItMatters: `The dot product is the workhorse of neural networks. Every neuron computes a dot product between its weights and inputs. In attention mechanisms (like in transformers), dot products measure similarity between queries and keys. **Production Pattern:** Recommendation systems use dot products between user and item embeddings to predict preferences.`,
	translations: {
		ru: {
			title: 'Скалярное произведение',
			description: `# Скалярное произведение

Реализуйте функцию вычисления скалярного произведения двух векторов.

## Теория

Скалярное произведение двух векторов:
- Сумма произведений соответствующих элементов
- Результат - скаляр (одно число), не вектор

\`\`\`
a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ
\`\`\`

## Применение в ML

- **Схожесть**: Скалярное произведение измеряет похожесть векторов
- **Нейросети**: Взвешенная сумма входов использует скалярное произведение
- **Attention**: Query-key скалярные произведения в трансформерах
`,
			hint1: 'NumPy предоставляет функцию np.dot() для вычисления скалярных произведений.',
			hint2: 'Альтернативно можно использовать (v1 * v2).sum() или оператор @ для матричного умножения.',
			whyItMatters: `Скалярное произведение - рабочая лошадка нейросетей. Каждый нейрон вычисляет скалярное произведение между весами и входами. **Production Pattern:** Рекомендательные системы используют скалярные произведения эмбеддингов пользователей и товаров.`,
		},
		uz: {
			title: 'Skalyar ko\'paytma',
			description: `# Skalyar ko'paytma

Ikki vektorning skalyar ko'paytmasini hisoblaydigan funksiyani amalga oshiring.

## Nazariya

Ikki vektorning skalyar ko'paytmasi - mos elementlar ko'paytmalarining yig'indisi.

## ML da qo'llanilishi

- Vektorlar o'xshashligini o'lchash
- Neyron tarmoqlarda og'irlangan yig'indi
- Transformer attention mexanizmlari
`,
			hint1: 'NumPy skalyar ko\'paytmani hisoblash uchun np.dot() funksiyasini taqdim etadi.',
			hint2: 'Shuningdek (v1 * v2).sum() yoki @ operatoridan foydalanishingiz mumkin.',
			whyItMatters: `Skalyar ko'paytma neyron tarmoqlarning asosiy operatsiyasi. Har bir neyron og'irliklari va kirish qiymatlari orasida skalyar ko'paytma hisoblaydi.`,
		},
	},
};

export default task;
