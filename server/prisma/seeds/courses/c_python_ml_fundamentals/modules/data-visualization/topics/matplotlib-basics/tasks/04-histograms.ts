import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'matplotlib-histograms',
	title: 'Histograms',
	difficulty: 'easy',
	tags: ['matplotlib', 'histogram', 'distribution'],
	estimatedTime: '10m',
	isPremium: false,
	order: 4,
	description: `# Histograms

Histograms show the distribution of continuous data, essential for understanding data spread.

## Task

Implement three functions:
1. \`simple_histogram(data, bins)\` - Create histogram with specified bins
2. \`histogram_with_kde(data, bins)\` - Histogram with kernel density estimate overlay
3. \`stacked_histogram(data1, data2, bins, labels)\` - Stacked histogram for comparison

## Example

\`\`\`python
data = np.random.normal(0, 1, 1000)

fig = simple_histogram(data, 30)

# With KDE (density=True to normalize)
fig = histogram_with_kde(data, 30)

# Stacked comparison
data2 = np.random.normal(2, 1, 1000)
fig = stacked_histogram(data, data2, 30, ['Group A', 'Group B'])
\`\`\``,

	initialCode: `import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def simple_histogram(data: np.ndarray, bins: int):
    """Create histogram with specified bins. Return figure."""
    # Your code here
    pass

def histogram_with_kde(data: np.ndarray, bins: int):
    """Histogram with KDE overlay. Return figure."""
    # Your code here
    pass

def stacked_histogram(data1: np.ndarray, data2: np.ndarray, bins: int, labels: list):
    """Stacked histogram for comparison. Return figure."""
    # Your code here
    pass
`,

	solutionCode: `import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def simple_histogram(data: np.ndarray, bins: int):
    """Create histogram with specified bins. Return figure."""
    fig, ax = plt.subplots()
    ax.hist(data, bins=bins)
    return fig

def histogram_with_kde(data: np.ndarray, bins: int):
    """Histogram with KDE overlay. Return figure."""
    fig, ax = plt.subplots()
    ax.hist(data, bins=bins, density=True, alpha=0.7)
    kde = stats.gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 100)
    ax.plot(x, kde(x), 'r-', linewidth=2)
    return fig

def stacked_histogram(data1: np.ndarray, data2: np.ndarray, bins: int, labels: list):
    """Stacked histogram for comparison. Return figure."""
    fig, ax = plt.subplots()
    ax.hist([data1, data2], bins=bins, stacked=True, label=labels)
    ax.legend()
    return fig
`,

	testCode: `import matplotlib.pyplot as plt
import numpy as np
import unittest

class TestHistograms(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.data = np.random.normal(0, 1, 100)
        self.data2 = np.random.normal(2, 1, 100)
        plt.close('all')

    def test_simple_histogram_returns_figure(self):
        fig = simple_histogram(self.data, 20)
        self.assertIsInstance(fig, plt.Figure)

    def test_simple_histogram_has_patches(self):
        fig = simple_histogram(self.data, 20)
        ax = fig.axes[0]
        self.assertGreater(len(ax.patches), 0)

    def test_histogram_with_kde_returns_figure(self):
        fig = histogram_with_kde(self.data, 20)
        self.assertIsInstance(fig, plt.Figure)

    def test_histogram_with_kde_has_line(self):
        fig = histogram_with_kde(self.data, 20)
        ax = fig.axes[0]
        self.assertGreater(len(ax.lines), 0)

    def test_stacked_histogram_returns_figure(self):
        fig = stacked_histogram(self.data, self.data2, 20, ['A', 'B'])
        self.assertIsInstance(fig, plt.Figure)

    def test_stacked_histogram_has_legend(self):
        fig = stacked_histogram(self.data, self.data2, 20, ['A', 'B'])
        ax = fig.axes[0]
        self.assertIsNotNone(ax.get_legend())

    def test_stacked_histogram_has_patches(self):
        fig = stacked_histogram(self.data, self.data2, 20, ['A', 'B'])
        ax = fig.axes[0]
        self.assertGreater(len(ax.patches), 0)

    def test_simple_histogram_has_axes(self):
        fig = simple_histogram(self.data, 20)
        self.assertEqual(len(fig.axes), 1)

    def test_histogram_kde_has_patches(self):
        fig = histogram_with_kde(self.data, 20)
        ax = fig.axes[0]
        self.assertGreater(len(ax.patches), 0)

    def test_stacked_histogram_legend_labels(self):
        fig = stacked_histogram(self.data, self.data2, 20, ['Group A', 'Group B'])
        ax = fig.axes[0]
        legend = ax.get_legend()
        texts = [t.get_text() for t in legend.get_texts()]
        self.assertIn('Group A', texts)
        self.assertIn('Group B', texts)

    def tearDown(self):
        plt.close('all')
`,

	hint1: 'Use ax.hist(data, bins=n) for histogram, density=True for normalized',
	hint2: 'Use scipy.stats.gaussian_kde for KDE, pass list of arrays for stacked',

	whyItMatters: `Histograms are crucial for:

- **Data understanding**: See the shape of your data distribution
- **Feature engineering**: Identify transformations needed
- **Outlier detection**: Spot unusual values in tails
- **Model assumptions**: Check if data meets distribution requirements

Understanding distributions is fundamental to statistical ML.`,

	translations: {
		ru: {
			title: 'Гистограммы',
			description: `# Гистограммы

Гистограммы показывают распределение непрерывных данных, необходимы для понимания разброса данных.

## Задача

Реализуйте три функции:
1. \`simple_histogram(data, bins)\` - Создать гистограмму с указанным числом бинов
2. \`histogram_with_kde(data, bins)\` - Гистограмма с наложением ядерной оценки плотности
3. \`stacked_histogram(data1, data2, bins, labels)\` - Составная гистограмма для сравнения

## Пример

\`\`\`python
data = np.random.normal(0, 1, 1000)

fig = simple_histogram(data, 30)

# With KDE (density=True to normalize)
fig = histogram_with_kde(data, 30)

# Stacked comparison
data2 = np.random.normal(2, 1, 1000)
fig = stacked_histogram(data, data2, 30, ['Group A', 'Group B'])
\`\`\``,
			hint1: 'Используйте ax.hist(data, bins=n) для гистограммы, density=True для нормализованной',
			hint2: 'Используйте scipy.stats.gaussian_kde для KDE, передайте список массивов для составной',
			whyItMatters: `Гистограммы критичны для:

- **Понимание данных**: Видеть форму распределения данных
- **Feature engineering**: Определить необходимые преобразования
- **Обнаружение выбросов**: Заметить необычные значения в хвостах`,
		},
		uz: {
			title: 'Gistogrammalar',
			description: `# Gistogrammalar

Gistogrammalar uzluksiz ma'lumotlarning taqsimlanishini ko'rsatadi, ma'lumotlar tarqalishini tushunish uchun zarur.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`simple_histogram(data, bins)\` - Ko'rsatilgan binlar bilan gistogramma yaratish
2. \`histogram_with_kde(data, bins)\` - KDE qo'shimchasi bilan gistogramma
3. \`stacked_histogram(data1, data2, bins, labels)\` - Taqqoslash uchun qatlamli gistogramma

## Misol

\`\`\`python
data = np.random.normal(0, 1, 1000)

fig = simple_histogram(data, 30)

# With KDE (density=True to normalize)
fig = histogram_with_kde(data, 30)

# Stacked comparison
data2 = np.random.normal(2, 1, 1000)
fig = stacked_histogram(data, data2, 30, ['Group A', 'Group B'])
\`\`\``,
			hint1: "Gistogramma uchun ax.hist(data, bins=n), normallashtirilgan uchun density=True dan foydalaning",
			hint2: "KDE uchun scipy.stats.gaussian_kde dan, qatlamli uchun massivlar ro'yxatini bering",
			whyItMatters: `Gistogrammalar quyidagilar uchun muhim:

- **Ma'lumotlarni tushunish**: Ma'lumotlar taqsimlanishi shaklini ko'rish
- **Feature engineering**: Zarur o'zgartirishlarni aniqlash
- **Outlier aniqlash**: Chetlardagi g'ayrioddiy qiymatlarni topish`,
		},
	},
};

export default task;
