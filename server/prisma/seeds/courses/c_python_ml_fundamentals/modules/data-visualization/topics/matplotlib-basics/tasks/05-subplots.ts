import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'matplotlib-subplots',
	title: 'Subplots',
	difficulty: 'medium',
	tags: ['matplotlib', 'subplots', 'layout'],
	estimatedTime: '15m',
	isPremium: false,
	order: 5,
	description: `# Subplots

Subplots allow multiple plots in one figure for side-by-side comparison.

## Task

Implement three functions:
1. \`grid_subplots(data_list, nrows, ncols)\` - Create grid of line plots
2. \`shared_axis_subplots(x, y1, y2)\` - Two plots sharing x-axis
3. \`unequal_subplots(data1, data2, data3)\` - Different sized subplots

## Example

\`\`\`python
data_list = [np.random.randn(100) for _ in range(4)]
fig = grid_subplots(data_list, 2, 2)

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
fig = shared_axis_subplots(x, y1, y2)

# Large plot on left, two stacked on right
fig = unequal_subplots(data1, data2, data3)
\`\`\``,

	initialCode: `import matplotlib.pyplot as plt
import numpy as np

def grid_subplots(data_list: list, nrows: int, ncols: int):
    """Create grid of line plots. Return figure."""
    # Your code here
    pass

def shared_axis_subplots(x: np.ndarray, y1: np.ndarray, y2: np.ndarray):
    """Two plots sharing x-axis. Return figure."""
    # Your code here
    pass

def unequal_subplots(data1: np.ndarray, data2: np.ndarray, data3: np.ndarray):
    """Different sized subplots using GridSpec. Return figure."""
    # Your code here
    pass
`,

	solutionCode: `import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def grid_subplots(data_list: list, nrows: int, ncols: int):
    """Create grid of line plots. Return figure."""
    fig, axes = plt.subplots(nrows, ncols)
    axes = axes.flatten()
    for i, data in enumerate(data_list):
        axes[i].plot(data)
    return fig

def shared_axis_subplots(x: np.ndarray, y1: np.ndarray, y2: np.ndarray):
    """Two plots sharing x-axis. Return figure."""
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(x, y1)
    ax2.plot(x, y2)
    return fig

def unequal_subplots(data1: np.ndarray, data2: np.ndarray, data3: np.ndarray):
    """Different sized subplots using GridSpec. Return figure."""
    fig = plt.figure()
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[:, 0])  # Left column, both rows
    ax2 = fig.add_subplot(gs[0, 1])  # Top right
    ax3 = fig.add_subplot(gs[1, 1])  # Bottom right
    ax1.plot(data1)
    ax2.plot(data2)
    ax3.plot(data3)
    return fig
`,

	testCode: `import matplotlib.pyplot as plt
import numpy as np
import unittest

class TestSubplots(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.data_list = [np.random.randn(50) for _ in range(4)]
        self.x = np.linspace(0, 10, 50)
        self.y1 = np.sin(self.x)
        self.y2 = np.cos(self.x)
        plt.close('all')

    def test_grid_subplots_returns_figure(self):
        fig = grid_subplots(self.data_list, 2, 2)
        self.assertIsInstance(fig, plt.Figure)

    def test_grid_subplots_has_correct_axes(self):
        fig = grid_subplots(self.data_list, 2, 2)
        self.assertEqual(len(fig.axes), 4)

    def test_shared_axis_subplots_returns_figure(self):
        fig = shared_axis_subplots(self.x, self.y1, self.y2)
        self.assertIsInstance(fig, plt.Figure)

    def test_shared_axis_subplots_has_two_axes(self):
        fig = shared_axis_subplots(self.x, self.y1, self.y2)
        self.assertEqual(len(fig.axes), 2)

    def test_unequal_subplots_returns_figure(self):
        fig = unequal_subplots(self.y1, self.y2, self.y1)
        self.assertIsInstance(fig, plt.Figure)

    def test_unequal_subplots_has_three_axes(self):
        fig = unequal_subplots(self.y1, self.y2, self.y1)
        self.assertEqual(len(fig.axes), 3)

    def test_grid_subplots_each_has_line(self):
        fig = grid_subplots(self.data_list, 2, 2)
        for ax in fig.axes:
            self.assertEqual(len(ax.lines), 1)

    def test_shared_axis_each_has_line(self):
        fig = shared_axis_subplots(self.x, self.y1, self.y2)
        for ax in fig.axes:
            self.assertEqual(len(ax.lines), 1)

    def test_unequal_subplots_each_has_line(self):
        fig = unequal_subplots(self.y1, self.y2, self.y1)
        for ax in fig.axes:
            self.assertEqual(len(ax.lines), 1)

    def test_grid_different_size(self):
        data = [np.random.randn(50) for _ in range(6)]
        fig = grid_subplots(data, 2, 3)
        self.assertEqual(len(fig.axes), 6)

    def tearDown(self):
        plt.close('all')
`,

	hint1: 'Use plt.subplots(nrows, ncols) and flatten() to iterate over axes',
	hint2: 'Use GridSpec for complex layouts, sharex=True to share x-axis',

	whyItMatters: `Subplots are essential for:

- **Model comparison**: Show performance across datasets
- **Feature analysis**: Compare distributions side by side
- **Time series**: Multiple metrics on shared timeline
- **Reports**: Combine related visualizations

Professional dashboards require mastering subplot layouts.`,

	translations: {
		ru: {
			title: 'Подграфики',
			description: `# Подграфики

Подграфики позволяют размещать несколько графиков на одной фигуре для сравнения.

## Задача

Реализуйте три функции:
1. \`grid_subplots(data_list, nrows, ncols)\` - Создать сетку линейных графиков
2. \`shared_axis_subplots(x, y1, y2)\` - Два графика с общей осью X
3. \`unequal_subplots(data1, data2, data3)\` - Подграфики разного размера

## Пример

\`\`\`python
data_list = [np.random.randn(100) for _ in range(4)]
fig = grid_subplots(data_list, 2, 2)

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
fig = shared_axis_subplots(x, y1, y2)

# Large plot on left, two stacked on right
fig = unequal_subplots(data1, data2, data3)
\`\`\``,
			hint1: 'Используйте plt.subplots(nrows, ncols) и flatten() для итерации по осям',
			hint2: 'Используйте GridSpec для сложных макетов, sharex=True для общей оси X',
			whyItMatters: `Подграфики необходимы для:

- **Сравнение моделей**: Показать производительность на разных датасетах
- **Анализ признаков**: Сравнить распределения рядом
- **Временные ряды**: Несколько метрик на общей временной шкале`,
		},
		uz: {
			title: 'Subplotlar',
			description: `# Subplotlar

Subplotlar bir figurada bir nechta grafiklarni joylashtirish imkonini beradi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`grid_subplots(data_list, nrows, ncols)\` - Chiziqli grafiklar panjarasi yaratish
2. \`shared_axis_subplots(x, y1, y2)\` - Umumiy X o'qi bilan ikkita grafik
3. \`unequal_subplots(data1, data2, data3)\` - Turli o'lchamdagi subplotlar

## Misol

\`\`\`python
data_list = [np.random.randn(100) for _ in range(4)]
fig = grid_subplots(data_list, 2, 2)

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
fig = shared_axis_subplots(x, y1, y2)

# Large plot on left, two stacked on right
fig = unequal_subplots(data1, data2, data3)
\`\`\``,
			hint1: "plt.subplots(nrows, ncols) va o'qlar bo'yicha iteratsiya uchun flatten() dan foydalaning",
			hint2: "Murakkab joylashuvlar uchun GridSpec, umumiy X o'qi uchun sharex=True dan foydalaning",
			whyItMatters: `Subplotlar quyidagilar uchun zarur:

- **Model taqqoslash**: Turli datasetlarda ishlashni ko'rsatish
- **Xususiyat tahlili**: Taqsimlanishlarni yonma-yon taqqoslash
- **Vaqt qatorlari**: Umumiy vaqt shkalasida bir nechta metrikalar`,
		},
	},
};

export default task;
