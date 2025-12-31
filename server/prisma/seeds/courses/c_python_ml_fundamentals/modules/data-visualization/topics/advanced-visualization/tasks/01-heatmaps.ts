import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'matplotlib-heatmaps',
	title: 'Heatmaps',
	difficulty: 'medium',
	tags: ['matplotlib', 'heatmap', 'correlation'],
	estimatedTime: '12m',
	isPremium: false,
	order: 1,
	description: `# Heatmaps

Heatmaps visualize matrix data using color gradients, essential for correlation analysis.

## Task

Implement three functions:
1. \`simple_heatmap(data)\` - Create basic heatmap with colorbar
2. \`annotated_heatmap(data, labels)\` - Heatmap with value annotations
3. \`correlation_heatmap(df)\` - Correlation matrix heatmap from DataFrame

## Example

\`\`\`python
data = np.random.rand(5, 5)
fig = simple_heatmap(data)

# With annotations
labels = ['A', 'B', 'C', 'D', 'E']
fig = annotated_heatmap(data, labels)

# Correlation matrix
df = pd.DataFrame(np.random.randn(100, 4), columns=['a', 'b', 'c', 'd'])
fig = correlation_heatmap(df)
\`\`\``,

	initialCode: `import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def simple_heatmap(data: np.ndarray):
    """Create basic heatmap with colorbar. Return figure."""
    # Your code here
    pass

def annotated_heatmap(data: np.ndarray, labels: list):
    """Heatmap with value annotations. Return figure."""
    # Your code here
    pass

def correlation_heatmap(df: pd.DataFrame):
    """Correlation matrix heatmap from DataFrame. Return figure."""
    # Your code here
    pass
`,

	solutionCode: `import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def simple_heatmap(data: np.ndarray):
    """Create basic heatmap with colorbar. Return figure."""
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='viridis')
    fig.colorbar(im)
    return fig

def annotated_heatmap(data: np.ndarray, labels: list):
    """Heatmap with value annotations. Return figure."""
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='viridis')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='white')
    fig.colorbar(im)
    return fig

def correlation_heatmap(df: pd.DataFrame):
    """Correlation matrix heatmap from DataFrame. Return figure."""
    corr = df.corr()
    fig, ax = plt.subplots()
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im)
    return fig
`,

	testCode: `import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import unittest

class TestHeatmaps(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.data = np.random.rand(5, 5)
        self.labels = ['A', 'B', 'C', 'D', 'E']
        self.df = pd.DataFrame(np.random.randn(50, 4), columns=['a', 'b', 'c', 'd'])
        plt.close('all')

    def test_simple_heatmap_returns_figure(self):
        fig = simple_heatmap(self.data)
        self.assertIsInstance(fig, plt.Figure)

    def test_simple_heatmap_has_colorbar(self):
        fig = simple_heatmap(self.data)
        self.assertEqual(len(fig.axes), 2)

    def test_annotated_heatmap_returns_figure(self):
        fig = annotated_heatmap(self.data, self.labels)
        self.assertIsInstance(fig, plt.Figure)

    def test_annotated_heatmap_has_texts(self):
        fig = annotated_heatmap(self.data, self.labels)
        ax = fig.axes[0]
        self.assertGreater(len(ax.texts), 0)

    def test_correlation_heatmap_returns_figure(self):
        fig = correlation_heatmap(self.df)
        self.assertIsInstance(fig, plt.Figure)

    def test_correlation_heatmap_has_colorbar(self):
        fig = correlation_heatmap(self.df)
        self.assertEqual(len(fig.axes), 2)

    def test_correlation_heatmap_has_labels(self):
        fig = correlation_heatmap(self.df)
        ax = fig.axes[0]
        self.assertEqual(len(ax.get_xticklabels()), 4)

    def test_simple_heatmap_has_axes(self):
        fig = simple_heatmap(self.data)
        self.assertGreater(len(fig.axes), 0)

    def test_annotated_heatmap_has_colorbar(self):
        fig = annotated_heatmap(self.data, self.labels)
        self.assertEqual(len(fig.axes), 2)

    def test_annotated_heatmap_text_count(self):
        fig = annotated_heatmap(self.data, self.labels)
        ax = fig.axes[0]
        self.assertEqual(len(ax.texts), 25)

    def tearDown(self):
        plt.close('all')
`,

	hint1: 'Use ax.imshow(data, cmap="viridis") for heatmap, fig.colorbar(im) for colorbar',
	hint2: 'Use df.corr() to compute correlation matrix, ax.text() for annotations',

	whyItMatters: `Heatmaps are essential for:

- **Correlation analysis**: See relationships between all features at once
- **Confusion matrices**: Evaluate classification performance
- **Feature selection**: Identify highly correlated features
- **Pattern discovery**: Spot clusters and anomalies in matrix data

Critical for exploratory data analysis in ML.`,

	translations: {
		ru: {
			title: 'Тепловые карты',
			description: `# Тепловые карты

Тепловые карты визуализируют матричные данные с помощью цветовых градиентов, необходимы для корреляционного анализа.

## Задача

Реализуйте три функции:
1. \`simple_heatmap(data)\` - Создать базовую тепловую карту с цветовой шкалой
2. \`annotated_heatmap(data, labels)\` - Тепловая карта с аннотациями значений
3. \`correlation_heatmap(df)\` - Тепловая карта корреляционной матрицы из DataFrame

## Пример

\`\`\`python
data = np.random.rand(5, 5)
fig = simple_heatmap(data)

# With annotations
labels = ['A', 'B', 'C', 'D', 'E']
fig = annotated_heatmap(data, labels)

# Correlation matrix
df = pd.DataFrame(np.random.randn(100, 4), columns=['a', 'b', 'c', 'd'])
fig = correlation_heatmap(df)
\`\`\``,
			hint1: 'Используйте ax.imshow(data, cmap="viridis") для тепловой карты',
			hint2: 'Используйте df.corr() для вычисления корреляционной матрицы',
			whyItMatters: `Тепловые карты необходимы для:

- **Корреляционный анализ**: Видеть связи между всеми признаками сразу
- **Матрицы ошибок**: Оценка качества классификации
- **Отбор признаков**: Определение сильно коррелированных признаков`,
		},
		uz: {
			title: 'Issiqlik xaritalari',
			description: `# Issiqlik xaritalari

Issiqlik xaritalari rang gradientlari yordamida matritsa ma'lumotlarini vizualizatsiya qiladi, korrelyatsiya tahlili uchun zarur.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`simple_heatmap(data)\` - Rang shkalasi bilan oddiy issiqlik xaritasi yaratish
2. \`annotated_heatmap(data, labels)\` - Qiymat annotatsiyalari bilan issiqlik xaritasi
3. \`correlation_heatmap(df)\` - DataFrame dan korrelyatsiya matritsasi xaritasi

## Misol

\`\`\`python
data = np.random.rand(5, 5)
fig = simple_heatmap(data)

# With annotations
labels = ['A', 'B', 'C', 'D', 'E']
fig = annotated_heatmap(data, labels)

# Correlation matrix
df = pd.DataFrame(np.random.randn(100, 4), columns=['a', 'b', 'c', 'd'])
fig = correlation_heatmap(df)
\`\`\``,
			hint1: 'Issiqlik xaritasi uchun ax.imshow(data, cmap="viridis") dan foydalaning',
			hint2: "Korrelyatsiya matritsasini hisoblash uchun df.corr() dan foydalaning",
			whyItMatters: `Issiqlik xaritalari quyidagilar uchun zarur:

- **Korrelyatsiya tahlili**: Barcha xususiyatlar orasidagi munosabatlarni bir vaqtda ko'rish
- **Chalkashlik matritsalari**: Klassifikatsiya ishlashini baholash
- **Xususiyat tanlash**: Yuqori darajada bog'langan xususiyatlarni aniqlash`,
		},
	},
};

export default task;
