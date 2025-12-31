import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'matplotlib-plot-customization',
	title: 'Plot Customization',
	difficulty: 'medium',
	tags: ['matplotlib', 'styling', 'customization'],
	estimatedTime: '15m',
	isPremium: false,
	order: 6,
	description: `# Plot Customization

Customize plots with titles, labels, colors, and styles for professional visualizations.

## Task

Implement three functions:
1. \`add_labels(fig, title, xlabel, ylabel)\` - Add title and axis labels
2. \`set_style(x, y, style)\` - Apply matplotlib style ('seaborn', 'ggplot', etc.)
3. \`customize_colors(x, y, facecolor, edgecolor, linecolor)\` - Full color customization

## Example

\`\`\`python
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
fig = add_labels(fig, 'My Plot', 'X Axis', 'Y Axis')

# Apply style
fig = set_style(x, y, 'seaborn-v0_8')

# Custom colors
fig = customize_colors(x, y, 'lightgray', 'black', 'red')
\`\`\``,

	initialCode: `import matplotlib.pyplot as plt
import numpy as np

def add_labels(fig, title: str, xlabel: str, ylabel: str):
    """Add title and axis labels to figure. Return figure."""
    # Your code here
    pass

def set_style(x: np.ndarray, y: np.ndarray, style: str):
    """Create plot with matplotlib style. Return figure."""
    # Your code here
    pass

def customize_colors(x: np.ndarray, y: np.ndarray, facecolor: str, edgecolor: str, linecolor: str):
    """Create plot with custom colors. Return figure."""
    # Your code here
    pass
`,

	solutionCode: `import matplotlib.pyplot as plt
import numpy as np

def add_labels(fig, title: str, xlabel: str, ylabel: str):
    """Add title and axis labels to figure. Return figure."""
    ax = fig.axes[0]
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig

def set_style(x: np.ndarray, y: np.ndarray, style: str):
    """Create plot with matplotlib style. Return figure."""
    with plt.style.context(style):
        fig, ax = plt.subplots()
        ax.plot(x, y)
    return fig

def customize_colors(x: np.ndarray, y: np.ndarray, facecolor: str, edgecolor: str, linecolor: str):
    """Create plot with custom colors. Return figure."""
    fig, ax = plt.subplots()
    ax.set_facecolor(facecolor)
    for spine in ax.spines.values():
        spine.set_edgecolor(edgecolor)
    ax.plot(x, y, color=linecolor)
    return fig
`,

	testCode: `import matplotlib.pyplot as plt
import numpy as np
import unittest

class TestPlotCustomization(unittest.TestCase):
    def setUp(self):
        self.x = np.array([1, 2, 3, 4, 5])
        self.y = np.array([1, 4, 9, 16, 25])
        plt.close('all')

    def test_add_labels_returns_figure(self):
        fig, ax = plt.subplots()
        ax.plot(self.x, self.y)
        fig = add_labels(fig, 'Title', 'X', 'Y')
        self.assertIsInstance(fig, plt.Figure)

    def test_add_labels_has_title(self):
        fig, ax = plt.subplots()
        ax.plot(self.x, self.y)
        fig = add_labels(fig, 'My Title', 'X', 'Y')
        self.assertEqual(fig.axes[0].get_title(), 'My Title')

    def test_add_labels_has_xlabel(self):
        fig, ax = plt.subplots()
        ax.plot(self.x, self.y)
        fig = add_labels(fig, 'T', 'X Label', 'Y')
        self.assertEqual(fig.axes[0].get_xlabel(), 'X Label')

    def test_add_labels_has_ylabel(self):
        fig, ax = plt.subplots()
        ax.plot(self.x, self.y)
        fig = add_labels(fig, 'T', 'X', 'Y Label')
        self.assertEqual(fig.axes[0].get_ylabel(), 'Y Label')

    def test_set_style_returns_figure(self):
        fig = set_style(self.x, self.y, 'default')
        self.assertIsInstance(fig, plt.Figure)

    def test_set_style_has_line(self):
        fig = set_style(self.x, self.y, 'default')
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 1)

    def test_customize_colors_returns_figure(self):
        fig = customize_colors(self.x, self.y, 'white', 'black', 'blue')
        self.assertIsInstance(fig, plt.Figure)

    def test_customize_colors_line_color(self):
        fig = customize_colors(self.x, self.y, 'white', 'black', 'red')
        ax = fig.axes[0]
        self.assertEqual(ax.lines[0].get_color(), 'red')

    def test_customize_colors_has_line(self):
        fig = customize_colors(self.x, self.y, 'white', 'black', 'blue')
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 1)

    def test_set_style_has_axes(self):
        fig = set_style(self.x, self.y, 'default')
        self.assertEqual(len(fig.axes), 1)

    def tearDown(self):
        plt.close('all')
`,

	hint1: 'Use ax.set_title(), ax.set_xlabel(), ax.set_ylabel() for labels',
	hint2: 'Use plt.style.context(style) for temporary style, ax.set_facecolor() for background',

	whyItMatters: `Customization matters for:

- **Professional reports**: Publication-ready figures
- **Branding**: Company color schemes
- **Accessibility**: High contrast for visibility
- **Clarity**: Clear labeling for understanding

Well-styled visualizations communicate more effectively.`,

	translations: {
		ru: {
			title: 'Настройка графиков',
			description: `# Настройка графиков

Настройте графики с заголовками, подписями, цветами и стилями для профессиональных визуализаций.

## Задача

Реализуйте три функции:
1. \`add_labels(fig, title, xlabel, ylabel)\` - Добавить заголовок и подписи осей
2. \`set_style(x, y, style)\` - Применить стиль matplotlib
3. \`customize_colors(x, y, facecolor, edgecolor, linecolor)\` - Полная настройка цветов

## Пример

\`\`\`python
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
fig = add_labels(fig, 'My Plot', 'X Axis', 'Y Axis')

# Apply style
fig = set_style(x, y, 'seaborn-v0_8')

# Custom colors
fig = customize_colors(x, y, 'lightgray', 'black', 'red')
\`\`\``,
			hint1: 'Используйте ax.set_title(), ax.set_xlabel(), ax.set_ylabel() для подписей',
			hint2: 'Используйте plt.style.context(style) для временного стиля',
			whyItMatters: `Настройка важна для:

- **Профессиональные отчёты**: Фигуры готовые к публикации
- **Брендинг**: Фирменные цветовые схемы
- **Доступность**: Высокий контраст для видимости`,
		},
		uz: {
			title: 'Grafik sozlamalari',
			description: `# Grafik sozlamalari

Grafiklarni sarlavhalar, teglar, ranglar va stillar bilan professional vizualizatsiyalar uchun sozlang.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`add_labels(fig, title, xlabel, ylabel)\` - Sarlavha va o'q teglarini qo'shish
2. \`set_style(x, y, style)\` - Matplotlib stilini qo'llash
3. \`customize_colors(x, y, facecolor, edgecolor, linecolor)\` - To'liq rang sozlamalari

## Misol

\`\`\`python
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
fig = add_labels(fig, 'My Plot', 'X Axis', 'Y Axis')

# Apply style
fig = set_style(x, y, 'seaborn-v0_8')

# Custom colors
fig = customize_colors(x, y, 'lightgray', 'black', 'red')
\`\`\``,
			hint1: "Teglar uchun ax.set_title(), ax.set_xlabel(), ax.set_ylabel() dan foydalaning",
			hint2: "Vaqtinchalik stil uchun plt.style.context(style) dan foydalaning",
			whyItMatters: `Sozlamalar quyidagilar uchun muhim:

- **Professional hisobotlar**: Nashrga tayyor figuralar
- **Brending**: Kompaniya rang sxemalari
- **Kirish imkoniyati**: Ko'rinish uchun yuqori kontrast`,
		},
	},
};

export default task;
