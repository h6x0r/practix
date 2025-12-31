import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'numpy-random-arrays',
	title: 'Random Array Generation',
	difficulty: 'easy',
	tags: ['numpy', 'random', 'initialization'],
	estimatedTime: '15m',
	isPremium: false,
	order: 4,
	description: `# Random Array Generation

Random arrays are essential for initializing neural network weights, data augmentation, and Monte Carlo simulations.

## Task

Implement three functions:
1. \`random_uniform(shape)\` - Random values between 0 and 1
2. \`random_integers(low, high, size)\` - Random integers in range [low, high)
3. \`random_normal(mean, std, shape)\` - Random values from normal distribution

## Example

\`\`\`python
uniform = random_uniform((2, 3))
# [[0.123 0.456 0.789]
#  [0.234 0.567 0.890]]

integers = random_integers(1, 10, (3,))
# [5 2 8]

normal = random_normal(0, 1, (2, 2))
# [[-0.5  1.2]
#  [ 0.3 -0.8]]
\`\`\`

## Requirements

- Use \`np.random.rand()\`, \`np.random.randint()\`, \`np.random.normal()\``,

	initialCode: `import numpy as np

def random_uniform(shape: tuple) -> np.ndarray:
    """Create array with random values between 0 and 1."""
    # Your code here
    pass

def random_integers(low: int, high: int, size: tuple) -> np.ndarray:
    """Create array with random integers in [low, high)."""
    # Your code here
    pass

def random_normal(mean: float, std: float, shape: tuple) -> np.ndarray:
    """Create array with values from normal distribution."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def random_uniform(shape: tuple) -> np.ndarray:
    """Create array with random values between 0 and 1."""
    return np.random.rand(*shape)

def random_integers(low: int, high: int, size: tuple) -> np.ndarray:
    """Create array with random integers in [low, high)."""
    return np.random.randint(low, high, size)

def random_normal(mean: float, std: float, shape: tuple) -> np.ndarray:
    """Create array with values from normal distribution."""
    return np.random.normal(mean, std, shape)
`,

	testCode: `import numpy as np
import unittest

class TestRandomArrays(unittest.TestCase):
    def test_uniform_shape(self):
        result = random_uniform((3, 4))
        self.assertEqual(result.shape, (3, 4))

    def test_uniform_range(self):
        result = random_uniform((100,))
        self.assertTrue(np.all(result >= 0) and np.all(result < 1))

    def test_uniform_1d(self):
        result = random_uniform((5,))
        self.assertEqual(result.shape, (5,))

    def test_integers_shape(self):
        result = random_integers(0, 10, (2, 3))
        self.assertEqual(result.shape, (2, 3))

    def test_integers_range(self):
        result = random_integers(5, 15, (100,))
        self.assertTrue(np.all(result >= 5) and np.all(result < 15))

    def test_integers_dtype(self):
        result = random_integers(0, 10, (5,))
        self.assertTrue(np.issubdtype(result.dtype, np.integer))

    def test_normal_shape(self):
        result = random_normal(0, 1, (4, 5))
        self.assertEqual(result.shape, (4, 5))

    def test_normal_mean_approx(self):
        result = random_normal(10, 1, (10000,))
        self.assertAlmostEqual(np.mean(result), 10, delta=0.1)

    def test_normal_std_approx(self):
        result = random_normal(0, 5, (10000,))
        self.assertAlmostEqual(np.std(result), 5, delta=0.2)

    def test_uniform_different_values(self):
        result = random_uniform((10,))
        self.assertTrue(len(set(result)) > 1)
`,

	hint1: 'Use np.random.rand(), randint(), normal()',
	hint2: 'rand() unpacks shape with *, randint needs size param',

	whyItMatters: `Random initialization is critical in ML:

- **Weight initialization**: Neural networks need random starting weights to break symmetry
- **Data augmentation**: Add random noise to training data
- **Dropout**: Randomly zero out neurons during training
- **Monte Carlo methods**: Simulate random processes for uncertainty estimation

Poor random initialization can cause vanishing/exploding gradients and training failures.`,

	translations: {
		ru: {
			title: 'Генерация случайных массивов',
			description: `# Генерация случайных массивов

Случайные массивы необходимы для инициализации весов нейросетей, аугментации данных и симуляций Монте-Карло.

## Задача

Реализуйте три функции:
1. \`random_uniform(shape)\` - Случайные значения от 0 до 1
2. \`random_integers(low, high, size)\` - Случайные целые числа в диапазоне [low, high)
3. \`random_normal(mean, std, shape)\` - Случайные значения из нормального распределения

## Пример

\`\`\`python
uniform = random_uniform((2, 3))
# [[0.123 0.456 0.789]
#  [0.234 0.567 0.890]]

integers = random_integers(1, 10, (3,))
# [5 2 8]

normal = random_normal(0, 1, (2, 2))
# [[-0.5  1.2]
#  [ 0.3 -0.8]]
\`\`\`

## Требования

- Используйте \`np.random.rand()\`, \`np.random.randint()\`, \`np.random.normal()\``,
			hint1: 'Используйте np.random.rand(), randint(), normal()',
			hint2: 'rand() распаковывает shape через *, randint нужен параметр size',
			whyItMatters: `Случайная инициализация критична в ML:

- **Инициализация весов**: Нейросетям нужны случайные начальные веса для нарушения симметрии
- **Аугментация данных**: Добавление случайного шума к обучающим данным
- **Dropout**: Случайное обнуление нейронов при обучении
- **Методы Монте-Карло**: Симуляция случайных процессов для оценки неопределённости`,
		},
		uz: {
			title: "Tasodifiy massivlar generatsiyasi",
			description: `# Tasodifiy massivlar generatsiyasi

Tasodifiy massivlar neyron tarmoq vaznlarini initsializatsiya qilish, ma'lumotlarni augmentatsiya qilish va Monte-Karlo simulyatsiyalari uchun zarur.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`random_uniform(shape)\` - 0 dan 1 gacha tasodifiy qiymatlar
2. \`random_integers(low, high, size)\` - [low, high) oralig'ida tasodifiy butun sonlar
3. \`random_normal(mean, std, shape)\` - Normal taqsimotdan tasodifiy qiymatlar

## Misol

\`\`\`python
uniform = random_uniform((2, 3))
# [[0.123 0.456 0.789]
#  [0.234 0.567 0.890]]

integers = random_integers(1, 10, (3,))
# [5 2 8]

normal = random_normal(0, 1, (2, 2))
# [[-0.5  1.2]
#  [ 0.3 -0.8]]
\`\`\`

## Talablar

- \`np.random.rand()\`, \`np.random.randint()\`, \`np.random.normal()\` dan foydalaning`,
			hint1: "np.random.rand(), randint(), normal() dan foydalaning",
			hint2: "rand() shape ni * bilan ochadi, randint size parametrini talab qiladi",
			whyItMatters: `Tasodifiy initsializatsiya ML da juda muhim:

- **Vazn initsializatsiyasi**: Neyron tarmoqlarga simmetriyani buzish uchun tasodifiy boshlang'ich vaznlar kerak
- **Ma'lumotlar augmentatsiyasi**: O'quv ma'lumotlariga tasodifiy shovqin qo'shish
- **Dropout**: O'qitish paytida neyronlarni tasodifiy nolga tenglashtirish`,
		},
	},
};

export default task;
