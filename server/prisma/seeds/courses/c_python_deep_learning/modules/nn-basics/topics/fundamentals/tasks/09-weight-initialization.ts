import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-nn-weight-initialization',
	title: 'Weight Initialization',
	difficulty: 'easy',
	tags: ['numpy', 'initialization', 'neural-networks'],
	estimatedTime: '10m',
	isPremium: false,
	order: 9,
	description: `# Weight Initialization

Implement proper weight initialization strategies.

## Task

Implement four functions:
1. \`xavier_uniform(fan_in, fan_out)\` - Xavier/Glorot uniform initialization
2. \`xavier_normal(fan_in, fan_out)\` - Xavier/Glorot normal initialization
3. \`he_uniform(fan_in, fan_out)\` - He/Kaiming uniform (for ReLU)
4. \`he_normal(fan_in, fan_out)\` - He/Kaiming normal (for ReLU)

## Example

\`\`\`python
# For layer with 784 inputs and 256 outputs
W = xavier_uniform(784, 256)  # Good for tanh/sigmoid
W = he_normal(784, 256)       # Good for ReLU
\`\`\``,

	initialCode: `import numpy as np

def xavier_uniform(fan_in: int, fan_out: int) -> np.ndarray:
    """Xavier uniform: U(-limit, limit) where limit = sqrt(6/(fan_in+fan_out))."""
    # Your code here
    pass

def xavier_normal(fan_in: int, fan_out: int) -> np.ndarray:
    """Xavier normal: N(0, std) where std = sqrt(2/(fan_in+fan_out))."""
    # Your code here
    pass

def he_uniform(fan_in: int, fan_out: int) -> np.ndarray:
    """He uniform for ReLU: U(-limit, limit) where limit = sqrt(6/fan_in)."""
    # Your code here
    pass

def he_normal(fan_in: int, fan_out: int) -> np.ndarray:
    """He normal for ReLU: N(0, std) where std = sqrt(2/fan_in)."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def xavier_uniform(fan_in: int, fan_out: int) -> np.ndarray:
    """Xavier uniform: U(-limit, limit) where limit = sqrt(6/(fan_in+fan_out))."""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

def xavier_normal(fan_in: int, fan_out: int) -> np.ndarray:
    """Xavier normal: N(0, std) where std = sqrt(2/(fan_in+fan_out))."""
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(fan_in, fan_out) * std

def he_uniform(fan_in: int, fan_out: int) -> np.ndarray:
    """He uniform for ReLU: U(-limit, limit) where limit = sqrt(6/fan_in)."""
    limit = np.sqrt(6.0 / fan_in)
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

def he_normal(fan_in: int, fan_out: int) -> np.ndarray:
    """He normal for ReLU: N(0, std) where std = sqrt(2/fan_in)."""
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out) * std
`,

	testCode: `import numpy as np
import unittest

class TestWeightInitialization(unittest.TestCase):
    def test_xavier_uniform_shape(self):
        W = xavier_uniform(784, 256)
        self.assertEqual(W.shape, (784, 256))

    def test_xavier_uniform_range(self):
        W = xavier_uniform(100, 100)
        limit = np.sqrt(6.0 / 200)
        self.assertTrue(np.all(W >= -limit))
        self.assertTrue(np.all(W <= limit))

    def test_xavier_normal_mean(self):
        W = xavier_normal(1000, 1000)
        self.assertAlmostEqual(np.mean(W), 0, places=1)

    def test_he_normal_variance(self):
        W = he_normal(1000, 256)
        expected_std = np.sqrt(2.0 / 1000)
        self.assertAlmostEqual(np.std(W), expected_std, places=1)

    def test_he_uniform_shape(self):
        W = he_uniform(512, 128)
        self.assertEqual(W.shape, (512, 128))

    def test_xavier_normal_shape(self):
        W = xavier_normal(256, 64)
        self.assertEqual(W.shape, (256, 64))

    def test_he_uniform_range(self):
        W = he_uniform(100, 50)
        limit = np.sqrt(6.0 / 100)
        self.assertTrue(np.all(W >= -limit))
        self.assertTrue(np.all(W <= limit))

    def test_he_normal_mean(self):
        W = he_normal(1000, 500)
        self.assertAlmostEqual(np.mean(W), 0, places=1)

    def test_xavier_uniform_returns_numpy(self):
        W = xavier_uniform(64, 32)
        self.assertIsInstance(W, np.ndarray)

    def test_all_return_different_values(self):
        W1 = xavier_uniform(100, 50)
        W2 = xavier_uniform(100, 50)
        self.assertFalse(np.allclose(W1, W2))
`,

	hint1: 'Xavier: scale by sqrt(2/(fan_in + fan_out))',
	hint2: 'He: scale by sqrt(2/fan_in), better for ReLU networks',

	whyItMatters: `Proper initialization is critical:

- **Training stability**: Prevents vanishing/exploding gradients at start
- **Faster convergence**: Good initialization speeds up training
- **Activation-aware**: He init matches ReLU statistics
- **Deep networks**: Enables training very deep architectures

Bad initialization can completely prevent learning.`,

	translations: {
		ru: {
			title: 'Инициализация весов',
			description: `# Инициализация весов

Реализуйте правильные стратегии инициализации весов.

## Задача

Реализуйте четыре функции:
1. \`xavier_uniform(fan_in, fan_out)\` - Xavier/Glorot равномерная
2. \`xavier_normal(fan_in, fan_out)\` - Xavier/Glorot нормальная
3. \`he_uniform(fan_in, fan_out)\` - He/Kaiming для ReLU
4. \`he_normal(fan_in, fan_out)\` - He/Kaiming нормальная для ReLU

## Пример

\`\`\`python
# For layer with 784 inputs and 256 outputs
W = xavier_uniform(784, 256)  # Good for tanh/sigmoid
W = he_normal(784, 256)       # Good for ReLU
\`\`\``,
			hint1: 'Xavier: масштаб sqrt(2/(fan_in + fan_out))',
			hint2: 'He: масштаб sqrt(2/fan_in), лучше для ReLU сетей',
			whyItMatters: `Правильная инициализация критична:

- **Стабильность обучения**: Предотвращает затухание/взрыв градиентов
- **Быстрая сходимость**: Хорошая инициализация ускоряет обучение
- **Учёт активации**: He init соответствует статистике ReLU`,
		},
		uz: {
			title: "Og'irliklarni ishga tushirish",
			description: `# Og'irliklarni ishga tushirish

To'g'ri og'irliklarni ishga tushirish strategiyalarini amalga oshiring.

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`xavier_uniform(fan_in, fan_out)\` - Xavier/Glorot uniform ishga tushirish
2. \`xavier_normal(fan_in, fan_out)\` - Xavier/Glorot normal ishga tushirish
3. \`he_uniform(fan_in, fan_out)\` - ReLU uchun He/Kaiming uniform
4. \`he_normal(fan_in, fan_out)\` - ReLU uchun He/Kaiming normal

## Misol

\`\`\`python
# For layer with 784 inputs and 256 outputs
W = xavier_uniform(784, 256)  # Good for tanh/sigmoid
W = he_normal(784, 256)       # Good for ReLU
\`\`\``,
			hint1: "Xavier: sqrt(2/(fan_in + fan_out)) bilan masshtablash",
			hint2: "He: sqrt(2/fan_in) bilan masshtablash, ReLU tarmoqlar uchun yaxshiroq",
			whyItMatters: `To'g'ri ishga tushirish muhim:

- **O'qitish barqarorligi**: Boshida so'nuvchi/portlovchi gradientlarni oldini oladi
- **Tezroq yaqinlashish**: Yaxshi ishga tushirish o'qitishni tezlashtiradi
- **Aktivatsiyadan xabardor**: He init ReLU statistikasiga mos keladi`,
		},
	},
};

export default task;
