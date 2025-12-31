import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'seaborn-statistical-plots',
	title: 'Statistical Plots',
	difficulty: 'medium',
	tags: ['seaborn', 'boxplot', 'violin'],
	estimatedTime: '15m',
	isPremium: false,
	order: 3,
	description: `# Statistical Plots

Box plots and violin plots show distribution statistics for categorical comparison.

## Task

Implement three functions:
1. \`box_plot(df, x, y)\` - Box plot for distribution comparison
2. \`violin_plot(df, x, y)\` - Violin plot showing full distribution shape
3. \`swarm_plot(df, x, y)\` - Swarm plot showing individual data points

## Example

\`\`\`python
df = pd.DataFrame({
    'category': ['A'] * 50 + ['B'] * 50,
    'value': np.concatenate([np.random.normal(0, 1, 50), np.random.normal(2, 1, 50)])
})

fig = box_plot(df, 'category', 'value')
fig = violin_plot(df, 'category', 'value')
fig = swarm_plot(df, 'category', 'value')
\`\`\``,

	initialCode: `import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def box_plot(df: pd.DataFrame, x: str, y: str):
    """Box plot for distribution comparison. Return figure."""
    # Your code here
    pass

def violin_plot(df: pd.DataFrame, x: str, y: str):
    """Violin plot showing distribution shape. Return figure."""
    # Your code here
    pass

def swarm_plot(df: pd.DataFrame, x: str, y: str):
    """Swarm plot showing individual points. Return figure."""
    # Your code here
    pass
`,

	solutionCode: `import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def box_plot(df: pd.DataFrame, x: str, y: str):
    """Box plot for distribution comparison. Return figure."""
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x=x, y=y, ax=ax)
    return fig

def violin_plot(df: pd.DataFrame, x: str, y: str):
    """Violin plot showing distribution shape. Return figure."""
    fig, ax = plt.subplots()
    sns.violinplot(data=df, x=x, y=y, ax=ax)
    return fig

def swarm_plot(df: pd.DataFrame, x: str, y: str):
    """Swarm plot showing individual points. Return figure."""
    fig, ax = plt.subplots()
    sns.swarmplot(data=df, x=x, y=y, ax=ax)
    return fig
`,

	testCode: `import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import unittest

class TestStatisticalPlots(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.df = pd.DataFrame({
            'category': ['A'] * 30 + ['B'] * 30,
            'value': np.concatenate([np.random.normal(0, 1, 30), np.random.normal(2, 1, 30)])
        })
        plt.close('all')

    def test_box_plot_returns_figure(self):
        fig = box_plot(self.df, 'category', 'value')
        self.assertIsInstance(fig, plt.Figure)

    def test_box_plot_has_patches(self):
        fig = box_plot(self.df, 'category', 'value')
        ax = fig.axes[0]
        self.assertGreater(len(ax.patches), 0)

    def test_violin_plot_returns_figure(self):
        fig = violin_plot(self.df, 'category', 'value')
        self.assertIsInstance(fig, plt.Figure)

    def test_violin_plot_has_collections(self):
        fig = violin_plot(self.df, 'category', 'value')
        ax = fig.axes[0]
        self.assertGreater(len(ax.collections), 0)

    def test_swarm_plot_returns_figure(self):
        fig = swarm_plot(self.df, 'category', 'value')
        self.assertIsInstance(fig, plt.Figure)

    def test_swarm_plot_has_collections(self):
        fig = swarm_plot(self.df, 'category', 'value')
        ax = fig.axes[0]
        self.assertGreater(len(ax.collections), 0)

    def test_box_plot_has_axes(self):
        fig = box_plot(self.df, 'category', 'value')
        self.assertEqual(len(fig.axes), 1)

    def test_violin_plot_has_axes(self):
        fig = violin_plot(self.df, 'category', 'value')
        self.assertEqual(len(fig.axes), 1)

    def test_swarm_plot_has_axes(self):
        fig = swarm_plot(self.df, 'category', 'value')
        self.assertEqual(len(fig.axes), 1)

    def test_box_plot_has_lines(self):
        fig = box_plot(self.df, 'category', 'value')
        ax = fig.axes[0]
        self.assertGreater(len(ax.lines), 0)

    def tearDown(self):
        plt.close('all')
`,

	hint1: 'Use sns.boxplot(data=df, x=x, y=y, ax=ax) for box plots',
	hint2: 'Use sns.violinplot for distribution shape, sns.swarmplot for individual points',

	whyItMatters: `Statistical plots are essential for:

- **Distribution comparison**: Compare groups at a glance
- **Outlier detection**: Box plots clearly show outliers
- **Model evaluation**: Compare performance across conditions
- **Feature analysis**: Understand feature distributions by target

Core tools for statistical analysis in ML.`,

	translations: {
		ru: {
			title: 'Статистические графики',
			description: `# Статистические графики

Диаграммы размаха и скрипичные диаграммы показывают статистики распределения для сравнения категорий.

## Задача

Реализуйте три функции:
1. \`box_plot(df, x, y)\` - Диаграмма размаха для сравнения распределений
2. \`violin_plot(df, x, y)\` - Скрипичная диаграмма показывающая форму распределения
3. \`swarm_plot(df, x, y)\` - Роевая диаграмма показывающая отдельные точки

## Пример

\`\`\`python
df = pd.DataFrame({
    'category': ['A'] * 50 + ['B'] * 50,
    'value': np.concatenate([np.random.normal(0, 1, 50), np.random.normal(2, 1, 50)])
})

fig = box_plot(df, 'category', 'value')
fig = violin_plot(df, 'category', 'value')
fig = swarm_plot(df, 'category', 'value')
\`\`\``,
			hint1: 'Используйте sns.boxplot(data=df, x=x, y=y, ax=ax) для диаграмм размаха',
			hint2: 'Используйте sns.violinplot для формы распределения, sns.swarmplot для отдельных точек',
			whyItMatters: `Статистические графики необходимы для:

- **Сравнение распределений**: Сравнить группы одним взглядом
- **Обнаружение выбросов**: Диаграммы размаха чётко показывают выбросы
- **Оценка моделей**: Сравнение производительности`,
		},
		uz: {
			title: 'Statistik grafiklar',
			description: `# Statistik grafiklar

Box va violin diagrammalari kategoriyalarni taqqoslash uchun taqsimlanish statistikasini ko'rsatadi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`box_plot(df, x, y)\` - Taqsimlanishni taqqoslash uchun box diagramma
2. \`violin_plot(df, x, y)\` - Taqsimlanish shaklini ko'rsatadigan violin diagramma
3. \`swarm_plot(df, x, y)\` - Alohida nuqtalarni ko'rsatadigan swarm diagramma

## Misol

\`\`\`python
df = pd.DataFrame({
    'category': ['A'] * 50 + ['B'] * 50,
    'value': np.concatenate([np.random.normal(0, 1, 50), np.random.normal(2, 1, 50)])
})

fig = box_plot(df, 'category', 'value')
fig = violin_plot(df, 'category', 'value')
fig = swarm_plot(df, 'category', 'value')
\`\`\``,
			hint1: "Box diagrammalar uchun sns.boxplot(data=df, x=x, y=y, ax=ax) dan foydalaning",
			hint2: "Taqsimlanish shakli uchun sns.violinplot, alohida nuqtalar uchun sns.swarmplot dan foydalaning",
			whyItMatters: `Statistik grafiklar quyidagilar uchun zarur:

- **Taqsimlanishni taqqoslash**: Guruhlarni bir qarashda taqqoslash
- **Outlier aniqlash**: Box diagrammalar outlierlarni aniq ko'rsatadi
- **Model baholash**: Shartlar bo'yicha ishlashni taqqoslash`,
		},
	},
};

export default task;
