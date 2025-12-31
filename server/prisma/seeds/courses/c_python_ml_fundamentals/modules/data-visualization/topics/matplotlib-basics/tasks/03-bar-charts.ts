import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'matplotlib-bar-charts',
	title: 'Bar Charts',
	difficulty: 'easy',
	tags: ['matplotlib', 'bar-chart', 'visualization'],
	estimatedTime: '12m',
	isPremium: false,
	order: 3,
	description: `# Bar Charts

Bar charts compare categorical data and are perfect for showing discrete values.

## Task

Implement three functions:
1. \`simple_bar(categories, values)\` - Create vertical bar chart
2. \`horizontal_bar(categories, values)\` - Create horizontal bar chart
3. \`grouped_bar(categories, values1, values2, labels)\` - Grouped bar chart

## Example

\`\`\`python
categories = ['A', 'B', 'C', 'D']
values = [25, 40, 30, 55]

fig = simple_bar(categories, values)
fig = horizontal_bar(categories, values)

# Grouped bars
values2 = [30, 25, 35, 45]
fig = grouped_bar(categories, values, values2, ['2023', '2024'])
\`\`\``,

	initialCode: `import matplotlib.pyplot as plt
import numpy as np

def simple_bar(categories: list, values: list):
    """Create vertical bar chart. Return figure."""
    # Your code here
    pass

def horizontal_bar(categories: list, values: list):
    """Create horizontal bar chart. Return figure."""
    # Your code here
    pass

def grouped_bar(categories: list, values1: list, values2: list, labels: list):
    """Create grouped bar chart. Return figure."""
    # Your code here
    pass
`,

	solutionCode: `import matplotlib.pyplot as plt
import numpy as np

def simple_bar(categories: list, values: list):
    """Create vertical bar chart. Return figure."""
    fig, ax = plt.subplots()
    ax.bar(categories, values)
    return fig

def horizontal_bar(categories: list, values: list):
    """Create horizontal bar chart. Return figure."""
    fig, ax = plt.subplots()
    ax.barh(categories, values)
    return fig

def grouped_bar(categories: list, values1: list, values2: list, labels: list):
    """Create grouped bar chart. Return figure."""
    fig, ax = plt.subplots()
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, values1, width, label=labels[0])
    ax.bar(x + width/2, values2, width, label=labels[1])
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    return fig
`,

	testCode: `import matplotlib.pyplot as plt
import numpy as np
import unittest

class TestBarCharts(unittest.TestCase):
    def setUp(self):
        self.categories = ['A', 'B', 'C', 'D']
        self.values = [25, 40, 30, 55]
        self.values2 = [30, 25, 35, 45]
        plt.close('all')

    def test_simple_bar_returns_figure(self):
        fig = simple_bar(self.categories, self.values)
        self.assertIsInstance(fig, plt.Figure)

    def test_simple_bar_has_patches(self):
        fig = simple_bar(self.categories, self.values)
        ax = fig.axes[0]
        self.assertEqual(len(ax.patches), 4)

    def test_horizontal_bar_returns_figure(self):
        fig = horizontal_bar(self.categories, self.values)
        self.assertIsInstance(fig, plt.Figure)

    def test_horizontal_bar_has_patches(self):
        fig = horizontal_bar(self.categories, self.values)
        ax = fig.axes[0]
        self.assertEqual(len(ax.patches), 4)

    def test_grouped_bar_returns_figure(self):
        fig = grouped_bar(self.categories, self.values, self.values2, ['a', 'b'])
        self.assertIsInstance(fig, plt.Figure)

    def test_grouped_bar_has_patches(self):
        fig = grouped_bar(self.categories, self.values, self.values2, ['a', 'b'])
        ax = fig.axes[0]
        self.assertEqual(len(ax.patches), 8)

    def test_grouped_bar_has_legend(self):
        fig = grouped_bar(self.categories, self.values, self.values2, ['a', 'b'])
        ax = fig.axes[0]
        self.assertIsNotNone(ax.get_legend())

    def test_simple_bar_has_axes(self):
        fig = simple_bar(self.categories, self.values)
        self.assertEqual(len(fig.axes), 1)

    def test_horizontal_bar_has_axes(self):
        fig = horizontal_bar(self.categories, self.values)
        self.assertEqual(len(fig.axes), 1)

    def test_grouped_bar_legend_labels(self):
        fig = grouped_bar(self.categories, self.values, self.values2, ['2023', '2024'])
        ax = fig.axes[0]
        legend = ax.get_legend()
        texts = [t.get_text() for t in legend.get_texts()]
        self.assertIn('2023', texts)
        self.assertIn('2024', texts)

    def tearDown(self):
        plt.close('all')
`,

	hint1: 'Use ax.bar() for vertical bars, ax.barh() for horizontal bars',
	hint2: 'For grouped bars, use np.arange() and offset positions by bar width',

	whyItMatters: `Bar charts are fundamental for:

- **Category comparison**: Compare values across groups
- **Model performance**: Compare accuracy across models
- **Feature importance**: Visualize feature rankings
- **A/B testing**: Show results across variants

Clear categorical comparisons are essential for presentations.`,

	translations: {
		ru: {
			title: 'Столбчатые диаграммы',
			description: `# Столбчатые диаграммы

Столбчатые диаграммы сравнивают категориальные данные и идеальны для показа дискретных значений.

## Задача

Реализуйте три функции:
1. \`simple_bar(categories, values)\` - Создать вертикальную столбчатую диаграмму
2. \`horizontal_bar(categories, values)\` - Создать горизонтальную диаграмму
3. \`grouped_bar(categories, values1, values2, labels)\` - Сгруппированная диаграмма

## Пример

\`\`\`python
categories = ['A', 'B', 'C', 'D']
values = [25, 40, 30, 55]

fig = simple_bar(categories, values)
fig = horizontal_bar(categories, values)

# Grouped bars
values2 = [30, 25, 35, 45]
fig = grouped_bar(categories, values, values2, ['2023', '2024'])
\`\`\``,
			hint1: 'Используйте ax.bar() для вертикальных столбцов, ax.barh() для горизонтальных',
			hint2: 'Для группированных столбцов используйте np.arange() и смещайте позиции на ширину столбца',
			whyItMatters: `Столбчатые диаграммы фундаментальны для:

- **Сравнение категорий**: Сравнение значений между группами
- **Производительность моделей**: Сравнение точности моделей
- **Важность признаков**: Визуализация рейтинга признаков`,
		},
		uz: {
			title: "Ustunli diagrammalar",
			description: `# Ustunli diagrammalar

Ustunli diagrammalar kategorik ma'lumotlarni taqqoslaydi va diskret qiymatlarni ko'rsatish uchun ideal.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`simple_bar(categories, values)\` - Vertikal ustunli diagramma yaratish
2. \`horizontal_bar(categories, values)\` - Gorizontal diagramma yaratish
3. \`grouped_bar(categories, values1, values2, labels)\` - Guruhlangan diagramma

## Misol

\`\`\`python
categories = ['A', 'B', 'C', 'D']
values = [25, 40, 30, 55]

fig = simple_bar(categories, values)
fig = horizontal_bar(categories, values)

# Grouped bars
values2 = [30, 25, 35, 45]
fig = grouped_bar(categories, values, values2, ['2023', '2024'])
\`\`\``,
			hint1: "Vertikal ustunlar uchun ax.bar(), gorizontal uchun ax.barh() dan foydalaning",
			hint2: "Guruhlangan ustunlar uchun np.arange() dan foydalaning va pozitsiyalarni ustun kengligi bo'yicha siljiting",
			whyItMatters: `Ustunli diagrammalar quyidagilar uchun asosiy:

- **Kategoriyalarni taqqoslash**: Guruhlar bo'yicha qiymatlarni taqqoslash
- **Model ishlashi**: Modellar aniqligini taqqoslash
- **Xususiyat ahamiyati**: Xususiyatlar reytingini vizualizatsiya qilish`,
		},
	},
};

export default task;
