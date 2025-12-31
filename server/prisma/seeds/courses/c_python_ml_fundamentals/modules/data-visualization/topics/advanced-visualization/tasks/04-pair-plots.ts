import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'seaborn-pair-plots',
	title: 'Pair Plots and Correlation',
	difficulty: 'medium',
	tags: ['seaborn', 'pairplot', 'correlation'],
	estimatedTime: '12m',
	isPremium: false,
	order: 4,
	description: `# Pair Plots and Correlation

Pair plots show pairwise relationships between all numeric features at once.

## Task

Implement three functions:
1. \`simple_pairplot(df)\` - Create basic pair plot for all numeric columns
2. \`pairplot_with_hue(df, hue)\` - Pair plot colored by category
3. \`correlation_clustermap(df)\` - Hierarchically clustered correlation heatmap

## Example

\`\`\`python
df = pd.DataFrame({
    'a': np.random.randn(100),
    'b': np.random.randn(100),
    'c': np.random.randn(100),
    'group': ['X'] * 50 + ['Y'] * 50
})

fig = simple_pairplot(df[['a', 'b', 'c']])
fig = pairplot_with_hue(df, 'group')
fig = correlation_clustermap(df[['a', 'b', 'c']])
\`\`\``,

	initialCode: `import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def simple_pairplot(df: pd.DataFrame):
    """Create basic pair plot. Return figure."""
    # Your code here
    pass

def pairplot_with_hue(df: pd.DataFrame, hue: str):
    """Pair plot colored by category. Return figure."""
    # Your code here
    pass

def correlation_clustermap(df: pd.DataFrame):
    """Hierarchically clustered correlation heatmap. Return figure."""
    # Your code here
    pass
`,

	solutionCode: `import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def simple_pairplot(df: pd.DataFrame):
    """Create basic pair plot. Return figure."""
    g = sns.pairplot(df)
    return g.figure

def pairplot_with_hue(df: pd.DataFrame, hue: str):
    """Pair plot colored by category. Return figure."""
    g = sns.pairplot(df, hue=hue)
    return g.figure

def correlation_clustermap(df: pd.DataFrame):
    """Hierarchically clustered correlation heatmap. Return figure."""
    corr = df.corr()
    g = sns.clustermap(corr, cmap='coolwarm', vmin=-1, vmax=1, annot=True)
    return g.figure
`,

	testCode: `import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import unittest

class TestPairPlots(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.df = pd.DataFrame({
            'a': np.random.randn(50),
            'b': np.random.randn(50),
            'c': np.random.randn(50),
            'group': ['X'] * 25 + ['Y'] * 25
        })
        plt.close('all')

    def test_simple_pairplot_returns_figure(self):
        fig = simple_pairplot(self.df[['a', 'b', 'c']])
        self.assertIsInstance(fig, plt.Figure)

    def test_simple_pairplot_has_axes(self):
        fig = simple_pairplot(self.df[['a', 'b', 'c']])
        self.assertGreater(len(fig.axes), 0)

    def test_pairplot_with_hue_returns_figure(self):
        fig = pairplot_with_hue(self.df, 'group')
        self.assertIsInstance(fig, plt.Figure)

    def test_pairplot_with_hue_has_axes(self):
        fig = pairplot_with_hue(self.df, 'group')
        self.assertGreater(len(fig.axes), 0)

    def test_correlation_clustermap_returns_figure(self):
        fig = correlation_clustermap(self.df[['a', 'b', 'c']])
        self.assertIsInstance(fig, plt.Figure)

    def test_correlation_clustermap_has_axes(self):
        fig = correlation_clustermap(self.df[['a', 'b', 'c']])
        self.assertGreater(len(fig.axes), 0)

    def test_simple_pairplot_axes_count(self):
        fig = simple_pairplot(self.df[['a', 'b', 'c']])
        self.assertGreaterEqual(len(fig.axes), 9)

    def test_pairplot_with_hue_axes_count(self):
        fig = pairplot_with_hue(self.df, 'group')
        self.assertGreater(len(fig.axes), 0)

    def test_correlation_clustermap_multiple_axes(self):
        fig = correlation_clustermap(self.df[['a', 'b', 'c']])
        self.assertGreater(len(fig.axes), 1)

    def test_pairplot_with_different_hue(self):
        df = self.df.copy()
        df['category'] = ['A'] * 25 + ['B'] * 25
        fig = pairplot_with_hue(df, 'category')
        self.assertIsInstance(fig, plt.Figure)

    def tearDown(self):
        plt.close('all')
`,

	hint1: 'Use sns.pairplot(df) for pair plots, returns PairGrid with .figure attribute',
	hint2: 'Use sns.clustermap(corr) for clustered heatmap with hierarchical clustering',

	whyItMatters: `Pair plots are powerful for:

- **Feature exploration**: See all relationships at once
- **Cluster identification**: Spot natural groupings
- **Correlation discovery**: Find linear relationships
- **Feature selection**: Identify redundant features

Essential first step in exploratory data analysis.`,

	translations: {
		ru: {
			title: 'Парные графики и корреляция',
			description: `# Парные графики и корреляция

Парные графики показывают попарные отношения между всеми числовыми признаками сразу.

## Задача

Реализуйте три функции:
1. \`simple_pairplot(df)\` - Создать базовый парный график
2. \`pairplot_with_hue(df, hue)\` - Парный график с цветовой группировкой
3. \`correlation_clustermap(df)\` - Иерархически кластеризованная корреляционная карта

## Пример

\`\`\`python
df = pd.DataFrame({
    'a': np.random.randn(100),
    'b': np.random.randn(100),
    'c': np.random.randn(100),
    'group': ['X'] * 50 + ['Y'] * 50
})

fig = simple_pairplot(df[['a', 'b', 'c']])
fig = pairplot_with_hue(df, 'group')
fig = correlation_clustermap(df[['a', 'b', 'c']])
\`\`\``,
			hint1: 'Используйте sns.pairplot(df) для парных графиков',
			hint2: 'Используйте sns.clustermap(corr) для кластеризованной тепловой карты',
			whyItMatters: `Парные графики мощный инструмент для:

- **Исследование признаков**: Видеть все отношения сразу
- **Идентификация кластеров**: Замечать естественные группировки
- **Обнаружение корреляций**: Находить линейные связи`,
		},
		uz: {
			title: "Juft grafiklar va korrelyatsiya",
			description: `# Juft grafiklar va korrelyatsiya

Juft grafiklar barcha raqamli xususiyatlar orasidagi juft munosabatlarni bir vaqtda ko'rsatadi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`simple_pairplot(df)\` - Oddiy juft grafik yaratish
2. \`pairplot_with_hue(df, hue)\` - Kategoriya bo'yicha ranglangan juft grafik
3. \`correlation_clustermap(df)\` - Ierarxik klasterlangan korrelyatsiya xaritasi

## Misol

\`\`\`python
df = pd.DataFrame({
    'a': np.random.randn(100),
    'b': np.random.randn(100),
    'c': np.random.randn(100),
    'group': ['X'] * 50 + ['Y'] * 50
})

fig = simple_pairplot(df[['a', 'b', 'c']])
fig = pairplot_with_hue(df, 'group')
fig = correlation_clustermap(df[['a', 'b', 'c']])
\`\`\``,
			hint1: "Juft grafiklar uchun sns.pairplot(df) dan foydalaning",
			hint2: "Klasterlangan issiqlik xaritasi uchun sns.clustermap(corr) dan foydalaning",
			whyItMatters: `Juft grafiklar quyidagilar uchun kuchli:

- **Xususiyatlarni o'rganish**: Barcha munosabatlarni bir vaqtda ko'rish
- **Klaster aniqlash**: Tabiiy guruhlanishlarni topish
- **Korrelyatsiya kashfiyoti**: Chiziqli munosabatlarni topish`,
		},
	},
};

export default task;
