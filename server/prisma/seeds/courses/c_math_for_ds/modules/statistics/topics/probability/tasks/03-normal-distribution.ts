import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'math-normal-distribution',
	title: 'Normal Distribution',
	difficulty: 'medium',
	tags: ['python', 'math', 'statistics', 'normal', 'gaussian', 'numpy'],
	estimatedTime: '20m',
	isPremium: true,
	order: 3,
	description: `# Normal Distribution

Implement the probability density function (PDF) of the normal distribution.

## Formula

f(x) = (1 / √(2πσ²)) × exp(-(x-μ)² / 2σ²)

The bell curve - most important distribution in statistics!
`,
	initialCode: `import numpy as np

def normal_pdf(x: np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """Compute normal distribution PDF."""
    # Your code here
    pass
`,
	solutionCode: `import numpy as np

def normal_pdf(x: np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """Compute normal distribution PDF."""
    coefficient = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    return coefficient * np.exp(exponent)
`,
	testCode: `import unittest
import numpy as np
from scipy import stats

class TestNormalPDF(unittest.TestCase):
    def test_standard_normal_at_zero(self):
        result = normal_pdf(np.array([0.0]))
        expected = 1 / np.sqrt(2 * np.pi)
        assert_close(result[0], expected)

    def test_standard_normal_symmetry(self):
        result_pos = normal_pdf(np.array([1.0]))
        result_neg = normal_pdf(np.array([-1.0]))
        assert_close(result_pos[0], result_neg[0])

    def test_peak_at_mean(self):
        x = np.linspace(-3, 3, 100)
        pdf = normal_pdf(x)
        max_idx = np.argmax(pdf)
        assert 45 < max_idx < 55

    def test_custom_mean(self):
        result = normal_pdf(np.array([5.0]), mu=5.0)
        expected = 1 / np.sqrt(2 * np.pi)
        assert_close(result[0], expected)

    def test_custom_sigma(self):
        result = normal_pdf(np.array([0.0]), sigma=2.0)
        expected = 1 / (2 * np.sqrt(2 * np.pi))
        assert_close(result[0], expected)

    def test_all_positive(self):
        x = np.linspace(-5, 5, 100)
        pdf = normal_pdf(x)
        assert np.all(pdf > 0)

    def test_tails_approach_zero(self):
        result = normal_pdf(np.array([10.0, -10.0]))
        assert np.all(result < 0.001)

    def test_scipy_comparison(self):
        x = np.array([0, 1, 2])
        result = normal_pdf(x)
        expected = stats.norm.pdf(x)
        assert_array_close(result, expected)

    def test_returns_array(self):
        result = normal_pdf(np.array([1, 2, 3]))
        assert isinstance(result, np.ndarray), f"Expected numpy.ndarray, got {type(result).__name__}"

    def test_shape_preserved(self):
        x = np.array([1, 2, 3, 4])
        result = normal_pdf(x)
        assert result.shape == x.shape, f"Expected x.shape, got {result.shape}"

`,
	hint1: 'Coefficient is 1/(sigma * sqrt(2*pi)), exponent is -(x-mu)²/(2*sigma²)',
	hint2: 'Use np.exp() for the exponential and np.sqrt(2*np.pi) for the constant.',
	whyItMatters: `The normal distribution is assumed in many ML algorithms. Linear regression assumes normal errors. Many natural phenomena follow normal distribution. **Production Pattern:** Weight initialization in neural networks often uses normal distribution (Glorot, He initialization).`,
	translations: {
		ru: {
			title: 'Нормальное распределение',
			description: `# Нормальное распределение

Реализуйте функцию плотности вероятности (PDF) нормального распределения.

Колоколообразная кривая - важнейшее распределение в статистике!
`,
			hint1: 'Коэффициент 1/(sigma * sqrt(2*pi)), экспонента -(x-mu)²/(2*sigma²)',
			hint2: 'Используйте np.exp() и np.sqrt(2*np.pi)',
			whyItMatters: `Нормальное распределение предполагается во многих алгоритмах ML. **Production Pattern:** Инициализация весов нейросетей часто использует нормальное распределение.`,
		},
		uz: {
			title: 'Normal taqsimot',
			description: `# Normal taqsimot

Normal taqsimotning ehtimollik zichligi funksiyasini (PDF) amalga oshiring.
`,
			hint1: 'Koeffitsient 1/(sigma * sqrt(2*pi))',
			hint2: 'np.exp() va np.sqrt(2*np.pi) dan foydalaning',
			whyItMatters: `Normal taqsimot ko'plab ML algoritmlarida qabul qilinadi. Neyron tarmoqlarning og'irlik initsializatsiyasi ko'pincha normal taqsimotdan foydalanadi.`,
		},
	},
};

export default task;
