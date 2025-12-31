import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'seaborn-basics',
	title: 'Seaborn Basics',
	difficulty: 'medium',
	tags: ['seaborn', 'statistical', 'visualization'],
	estimatedTime: '15m',
	isPremium: false,
	order: 2,
	description: `# Seaborn Basics

Seaborn provides high-level statistical visualization built on Matplotlib.

## Task

Implement three functions:
1. \`seaborn_scatter(df, x, y, hue)\` - Scatter plot with color grouping
2. \`seaborn_line(df, x, y, hue)\` - Line plot with confidence intervals
3. \`seaborn_dist(df, column, hue)\` - Distribution plot with KDE

## Example

\`\`\`python
import seaborn as sns

df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'group': ['A'] * 50 + ['B'] * 50
})

fig = seaborn_scatter(df, 'x', 'y', 'group')
fig = seaborn_line(df, 'x', 'y', 'group')
fig = seaborn_dist(df, 'x', 'group')
\`\`\``,

	initialCode: `import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def seaborn_scatter(df: pd.DataFrame, x: str, y: str, hue: str):
    """Scatter plot with color grouping. Return figure."""
    # Your code here
    pass

def seaborn_line(df: pd.DataFrame, x: str, y: str, hue: str):
    """Line plot with grouping. Return figure."""
    # Your code here
    pass

def seaborn_dist(df: pd.DataFrame, column: str, hue: str):
    """Distribution plot with KDE. Return figure."""
    # Your code here
    pass
`,

	solutionCode: `import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def seaborn_scatter(df: pd.DataFrame, x: str, y: str, hue: str):
    """Scatter plot with color grouping. Return figure."""
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax)
    return fig

def seaborn_line(df: pd.DataFrame, x: str, y: str, hue: str):
    """Line plot with grouping. Return figure."""
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax)
    return fig

def seaborn_dist(df: pd.DataFrame, column: str, hue: str):
    """Distribution plot with KDE. Return figure."""
    fig, ax = plt.subplots()
    sns.histplot(data=df, x=column, hue=hue, kde=True, ax=ax)
    return fig
`,

	testCode: `import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import unittest

class TestSeabornBasics(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.df = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
            'group': ['A'] * 50 + ['B'] * 50
        })
        plt.close('all')

    def test_seaborn_scatter_returns_figure(self):
        fig = seaborn_scatter(self.df, 'x', 'y', 'group')
        self.assertIsInstance(fig, plt.Figure)

    def test_seaborn_scatter_has_legend(self):
        fig = seaborn_scatter(self.df, 'x', 'y', 'group')
        ax = fig.axes[0]
        self.assertIsNotNone(ax.get_legend())

    def test_seaborn_line_returns_figure(self):
        fig = seaborn_line(self.df, 'x', 'y', 'group')
        self.assertIsInstance(fig, plt.Figure)

    def test_seaborn_line_has_lines(self):
        fig = seaborn_line(self.df, 'x', 'y', 'group')
        ax = fig.axes[0]
        self.assertGreater(len(ax.lines), 0)

    def test_seaborn_dist_returns_figure(self):
        fig = seaborn_dist(self.df, 'x', 'group')
        self.assertIsInstance(fig, plt.Figure)

    def test_seaborn_dist_has_patches(self):
        fig = seaborn_dist(self.df, 'x', 'group')
        ax = fig.axes[0]
        self.assertGreater(len(ax.patches), 0)

    def test_seaborn_scatter_has_axes(self):
        fig = seaborn_scatter(self.df, 'x', 'y', 'group')
        self.assertEqual(len(fig.axes), 1)

    def test_seaborn_line_has_legend(self):
        fig = seaborn_line(self.df, 'x', 'y', 'group')
        ax = fig.axes[0]
        self.assertIsNotNone(ax.get_legend())

    def test_seaborn_dist_has_axes(self):
        fig = seaborn_dist(self.df, 'x', 'group')
        self.assertEqual(len(fig.axes), 1)

    def test_seaborn_line_has_axes(self):
        fig = seaborn_line(self.df, 'x', 'y', 'group')
        self.assertEqual(len(fig.axes), 1)

    def tearDown(self):
        plt.close('all')
`,

	hint1: 'Use sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax)',
	hint2: 'Use sns.histplot with kde=True for distribution with KDE overlay',

	whyItMatters: `Seaborn simplifies:

- **Statistical visualization**: Built-in confidence intervals and aggregations
- **Categorical data**: Automatic handling of groups
- **Aesthetics**: Beautiful defaults out of the box
- **DataFrame integration**: Works directly with pandas

The go-to library for exploratory data analysis.`,

	translations: {
		ru: {
			title: 'Основы Seaborn',
			description: `# Основы Seaborn

Seaborn предоставляет высокоуровневую статистическую визуализацию на основе Matplotlib.

## Задача

Реализуйте три функции:
1. \`seaborn_scatter(df, x, y, hue)\` - Точечная диаграмма с группировкой по цвету
2. \`seaborn_line(df, x, y, hue)\` - Линейный график с доверительными интервалами
3. \`seaborn_dist(df, column, hue)\` - Диаграмма распределения с KDE

## Пример

\`\`\`python
import seaborn as sns

df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'group': ['A'] * 50 + ['B'] * 50
})

fig = seaborn_scatter(df, 'x', 'y', 'group')
fig = seaborn_line(df, 'x', 'y', 'group')
fig = seaborn_dist(df, 'x', 'group')
\`\`\``,
			hint1: 'Используйте sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax)',
			hint2: 'Используйте sns.histplot с kde=True для распределения с KDE',
			whyItMatters: `Seaborn упрощает:

- **Статистическая визуализация**: Встроенные доверительные интервалы
- **Категориальные данные**: Автоматическая обработка групп
- **Эстетика**: Красивые настройки по умолчанию`,
		},
		uz: {
			title: 'Seaborn asoslari',
			description: `# Seaborn asoslari

Seaborn Matplotlib asosida yuqori darajadagi statistik vizualizatsiyani taqdim etadi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`seaborn_scatter(df, x, y, hue)\` - Rang guruhlash bilan nuqtali diagramma
2. \`seaborn_line(df, x, y, hue)\` - Ishonch intervallari bilan chiziqli grafik
3. \`seaborn_dist(df, column, hue)\` - KDE bilan taqsimlanish diagrammasi

## Misol

\`\`\`python
import seaborn as sns

df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'group': ['A'] * 50 + ['B'] * 50
})

fig = seaborn_scatter(df, 'x', 'y', 'group')
fig = seaborn_line(df, 'x', 'y', 'group')
fig = seaborn_dist(df, 'x', 'group')
\`\`\``,
			hint1: "sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax) dan foydalaning",
			hint2: "KDE bilan taqsimlanish uchun kde=True bilan sns.histplot dan foydalaning",
			whyItMatters: `Seaborn quyidagilarni soddalashtiradi:

- **Statistik vizualizatsiya**: O'rnatilgan ishonch intervallari
- **Kategorik ma'lumotlar**: Guruhlarni avtomatik qayta ishlash
- **Estetika**: Chiroyli standart sozlamalar`,
		},
	},
};

export default task;
