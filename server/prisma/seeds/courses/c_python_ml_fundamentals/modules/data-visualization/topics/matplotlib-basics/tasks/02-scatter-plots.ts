import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'matplotlib-scatter-plots',
	title: 'Scatter Plots',
	difficulty: 'easy',
	tags: ['matplotlib', 'scatter', 'visualization'],
	estimatedTime: '12m',
	isPremium: false,
	order: 2,
	description: `# Scatter Plots

Scatter plots reveal relationships between two variables and are essential for correlation analysis.

## Task

Implement three functions:
1. \`simple_scatter(x, y)\` - Create basic scatter plot
2. \`scatter_with_size(x, y, sizes)\` - Scatter with variable point sizes
3. \`scatter_with_colors(x, y, colors, colormap)\` - Colored scatter with colorbar

## Example

\`\`\`python
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 1, 5, 3])

fig = simple_scatter(x, y)

# Variable sizes
sizes = np.array([20, 50, 100, 200, 300])
fig = scatter_with_size(x, y, sizes)

# Colors based on values
colors = np.array([1, 2, 3, 4, 5])
fig = scatter_with_colors(x, y, colors, 'viridis')
\`\`\``,

	initialCode: `import matplotlib.pyplot as plt
import numpy as np

def simple_scatter(x: np.ndarray, y: np.ndarray):
    """Create basic scatter plot. Return figure."""
    # Your code here
    pass

def scatter_with_size(x: np.ndarray, y: np.ndarray, sizes: np.ndarray):
    """Scatter with variable point sizes. Return figure."""
    # Your code here
    pass

def scatter_with_colors(x: np.ndarray, y: np.ndarray, colors: np.ndarray, colormap: str):
    """Colored scatter with colorbar. Return figure."""
    # Your code here
    pass
`,

	solutionCode: `import matplotlib.pyplot as plt
import numpy as np

def simple_scatter(x: np.ndarray, y: np.ndarray):
    """Create basic scatter plot. Return figure."""
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    return fig

def scatter_with_size(x: np.ndarray, y: np.ndarray, sizes: np.ndarray):
    """Scatter with variable point sizes. Return figure."""
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=sizes)
    return fig

def scatter_with_colors(x: np.ndarray, y: np.ndarray, colors: np.ndarray, colormap: str):
    """Colored scatter with colorbar. Return figure."""
    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y, c=colors, cmap=colormap)
    fig.colorbar(scatter)
    return fig
`,

	testCode: `import matplotlib.pyplot as plt
import numpy as np
import unittest

class TestScatterPlots(unittest.TestCase):
    def setUp(self):
        self.x = np.array([1, 2, 3, 4, 5])
        self.y = np.array([2, 4, 1, 5, 3])
        plt.close('all')

    def test_simple_scatter_returns_figure(self):
        fig = simple_scatter(self.x, self.y)
        self.assertIsInstance(fig, plt.Figure)

    def test_simple_scatter_has_collection(self):
        fig = simple_scatter(self.x, self.y)
        ax = fig.axes[0]
        self.assertEqual(len(ax.collections), 1)

    def test_scatter_with_size_returns_figure(self):
        sizes = np.array([20, 50, 100, 200, 300])
        fig = scatter_with_size(self.x, self.y, sizes)
        self.assertIsInstance(fig, plt.Figure)

    def test_scatter_with_size_has_points(self):
        sizes = np.array([20, 50, 100, 200, 300])
        fig = scatter_with_size(self.x, self.y, sizes)
        ax = fig.axes[0]
        self.assertEqual(len(ax.collections), 1)

    def test_scatter_with_colors_returns_figure(self):
        colors = np.array([1, 2, 3, 4, 5])
        fig = scatter_with_colors(self.x, self.y, colors, 'viridis')
        self.assertIsInstance(fig, plt.Figure)

    def test_scatter_with_colors_has_colorbar(self):
        colors = np.array([1, 2, 3, 4, 5])
        fig = scatter_with_colors(self.x, self.y, colors, 'viridis')
        # Figure should have 2 axes: main plot + colorbar
        self.assertEqual(len(fig.axes), 2)

    def test_point_count(self):
        fig = simple_scatter(self.x, self.y)
        ax = fig.axes[0]
        offsets = ax.collections[0].get_offsets()
        self.assertEqual(len(offsets), 5)

    def test_scatter_with_size_point_count(self):
        sizes = np.array([20, 50, 100, 200, 300])
        fig = scatter_with_size(self.x, self.y, sizes)
        ax = fig.axes[0]
        offsets = ax.collections[0].get_offsets()
        self.assertEqual(len(offsets), 5)

    def test_scatter_with_colors_point_count(self):
        colors = np.array([1, 2, 3, 4, 5])
        fig = scatter_with_colors(self.x, self.y, colors, 'viridis')
        ax = fig.axes[0]
        offsets = ax.collections[0].get_offsets()
        self.assertEqual(len(offsets), 5)

    def test_different_colormap(self):
        colors = np.array([1, 2, 3, 4, 5])
        fig = scatter_with_colors(self.x, self.y, colors, 'plasma')
        self.assertIsInstance(fig, plt.Figure)

    def tearDown(self):
        plt.close('all')
`,

	hint1: 'Use ax.scatter(x, y) for basic scatter, s= for sizes, c= for colors',
	hint2: 'Use fig.colorbar(scatter) to add colorbar, pass cmap= for colormap',

	whyItMatters: `Scatter plots are crucial for:

- **Correlation analysis**: See relationships between variables
- **Outlier detection**: Identify unusual data points visually
- **Clustering visualization**: Visualize cluster assignments
- **Feature relationships**: Understand how features interact

Essential for exploratory data analysis in ML.`,

	translations: {
		ru: {
			title: 'Точечные диаграммы',
			description: `# Точечные диаграммы

Точечные диаграммы показывают связи между двумя переменными и необходимы для корреляционного анализа.

## Задача

Реализуйте три функции:
1. \`simple_scatter(x, y)\` - Создать базовую точечную диаграмму
2. \`scatter_with_size(x, y, sizes)\` - Диаграмма с переменными размерами точек
3. \`scatter_with_colors(x, y, colors, colormap)\` - Цветная диаграмма с цветовой шкалой

## Пример

\`\`\`python
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 1, 5, 3])

fig = simple_scatter(x, y)

# Variable sizes
sizes = np.array([20, 50, 100, 200, 300])
fig = scatter_with_size(x, y, sizes)

# Colors based on values
colors = np.array([1, 2, 3, 4, 5])
fig = scatter_with_colors(x, y, colors, 'viridis')
\`\`\``,
			hint1: 'Используйте ax.scatter(x, y) для базовой диаграммы, s= для размеров, c= для цветов',
			hint2: 'Используйте fig.colorbar(scatter) для добавления цветовой шкалы',
			whyItMatters: `Точечные диаграммы критичны для:

- **Корреляционный анализ**: Видеть связи между переменными
- **Обнаружение выбросов**: Визуально определять необычные точки
- **Визуализация кластеров**: Показывать принадлежность к кластерам`,
		},
		uz: {
			title: 'Nuqtali diagrammalar',
			description: `# Nuqtali diagrammalar

Nuqtali diagrammalar ikki o'zgaruvchi orasidagi munosabatlarni ko'rsatadi va korrelyatsiya tahlili uchun zarur.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`simple_scatter(x, y)\` - Oddiy nuqtali diagramma yaratish
2. \`scatter_with_size(x, y, sizes)\` - O'zgaruvchan nuqta o'lchamlari bilan diagramma
3. \`scatter_with_colors(x, y, colors, colormap)\` - Rang shkalasi bilan rangli diagramma

## Misol

\`\`\`python
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 1, 5, 3])

fig = simple_scatter(x, y)

# Variable sizes
sizes = np.array([20, 50, 100, 200, 300])
fig = scatter_with_size(x, y, sizes)

# Colors based on values
colors = np.array([1, 2, 3, 4, 5])
fig = scatter_with_colors(x, y, colors, 'viridis')
\`\`\``,
			hint1: "Oddiy diagramma uchun ax.scatter(x, y), o'lchamlar uchun s=, ranglar uchun c= dan foydalaning",
			hint2: "Rang shkalasi qo'shish uchun fig.colorbar(scatter) dan foydalaning",
			whyItMatters: `Nuqtali diagrammalar quyidagilar uchun muhim:

- **Korrelyatsiya tahlili**: O'zgaruvchilar orasidagi munosabatlarni ko'rish
- **Outlier aniqlash**: G'ayrioddiy nuqtalarni vizual aniqlash
- **Klasterlarni vizualizatsiya qilish**: Klaster tayinlanishlarini ko'rsatish`,
		},
	},
};

export default task;
