import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'matplotlib-line-plots',
	title: 'Line Plots',
	difficulty: 'easy',
	tags: ['matplotlib', 'line-plot', 'visualization'],
	estimatedTime: '10m',
	isPremium: false,
	order: 1,
	description: `# Line Plots

Line plots are fundamental for visualizing trends and continuous data over time.

## Task

Implement three functions:
1. \`simple_line(x, y)\` - Create basic line plot
2. \`multiple_lines(x, y1, y2, labels)\` - Plot multiple lines with labels
3. \`styled_line(x, y, color, linestyle, marker)\` - Customized line plot

## Example

\`\`\`python
import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 4, 9, 16, 25])

fig = simple_line(x, y)  # Returns figure

# Multiple lines
y2 = np.array([1, 2, 3, 4, 5])
fig = multiple_lines(x, y, y2, ['squared', 'linear'])

# Styled line
fig = styled_line(x, y, 'red', '--', 'o')
\`\`\``,

	initialCode: `import matplotlib.pyplot as plt
import numpy as np

def simple_line(x: np.ndarray, y: np.ndarray):
    """Create basic line plot. Return figure."""
    # Your code here
    pass

def multiple_lines(x: np.ndarray, y1: np.ndarray, y2: np.ndarray, labels: list):
    """Plot multiple lines with labels. Return figure."""
    # Your code here
    pass

def styled_line(x: np.ndarray, y: np.ndarray, color: str, linestyle: str, marker: str):
    """Create styled line plot. Return figure."""
    # Your code here
    pass
`,

	solutionCode: `import matplotlib.pyplot as plt
import numpy as np

def simple_line(x: np.ndarray, y: np.ndarray):
    """Create basic line plot. Return figure."""
    fig, ax = plt.subplots()
    ax.plot(x, y)
    return fig

def multiple_lines(x: np.ndarray, y1: np.ndarray, y2: np.ndarray, labels: list):
    """Plot multiple lines with labels. Return figure."""
    fig, ax = plt.subplots()
    ax.plot(x, y1, label=labels[0])
    ax.plot(x, y2, label=labels[1])
    ax.legend()
    return fig

def styled_line(x: np.ndarray, y: np.ndarray, color: str, linestyle: str, marker: str):
    """Create styled line plot. Return figure."""
    fig, ax = plt.subplots()
    ax.plot(x, y, color=color, linestyle=linestyle, marker=marker)
    return fig
`,

	testCode: `import matplotlib.pyplot as plt
import numpy as np
import unittest

class TestLinePlots(unittest.TestCase):
    def setUp(self):
        self.x = np.array([1, 2, 3, 4, 5])
        self.y = np.array([1, 4, 9, 16, 25])
        self.y2 = np.array([1, 2, 3, 4, 5])
        plt.close('all')

    def test_simple_line_returns_figure(self):
        fig = simple_line(self.x, self.y)
        self.assertIsInstance(fig, plt.Figure)

    def test_simple_line_has_axes(self):
        fig = simple_line(self.x, self.y)
        self.assertEqual(len(fig.axes), 1)

    def test_simple_line_has_line(self):
        fig = simple_line(self.x, self.y)
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 1)

    def test_multiple_lines_count(self):
        fig = multiple_lines(self.x, self.y, self.y2, ['a', 'b'])
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 2)

    def test_multiple_lines_has_legend(self):
        fig = multiple_lines(self.x, self.y, self.y2, ['a', 'b'])
        ax = fig.axes[0]
        self.assertIsNotNone(ax.get_legend())

    def test_styled_line_color(self):
        fig = styled_line(self.x, self.y, 'red', '-', 'o')
        ax = fig.axes[0]
        self.assertEqual(ax.lines[0].get_color(), 'red')

    def test_styled_line_marker(self):
        fig = styled_line(self.x, self.y, 'blue', '--', 's')
        ax = fig.axes[0]
        self.assertEqual(ax.lines[0].get_marker(), 's')

    def test_styled_line_linestyle(self):
        fig = styled_line(self.x, self.y, 'green', '--', 'o')
        ax = fig.axes[0]
        self.assertEqual(ax.lines[0].get_linestyle(), '--')

    def test_multiple_lines_returns_figure(self):
        fig = multiple_lines(self.x, self.y, self.y2, ['a', 'b'])
        self.assertIsInstance(fig, plt.Figure)

    def test_styled_line_returns_figure(self):
        fig = styled_line(self.x, self.y, 'blue', '-', 'x')
        self.assertIsInstance(fig, plt.Figure)

    def tearDown(self):
        plt.close('all')
`,

	hint1: 'Use plt.subplots() to create figure and axes, then ax.plot() to draw lines',
	hint2: 'Pass color, linestyle, marker as keyword arguments to ax.plot()',

	whyItMatters: `Line plots are essential for:

- **Time series analysis**: Visualize trends over time
- **Model training**: Track loss and accuracy curves
- **Comparison**: Compare multiple metrics side by side
- **Communication**: Present findings clearly to stakeholders

This is the most fundamental visualization skill for data science.`,

	translations: {
		ru: {
			title: 'Линейные графики',
			description: `# Линейные графики

Линейные графики фундаментальны для визуализации трендов и непрерывных данных во времени.

## Задача

Реализуйте три функции:
1. \`simple_line(x, y)\` - Создать базовый линейный график
2. \`multiple_lines(x, y1, y2, labels)\` - Построить несколько линий с подписями
3. \`styled_line(x, y, color, linestyle, marker)\` - Стилизованный линейный график

## Пример

\`\`\`python
import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 4, 9, 16, 25])

fig = simple_line(x, y)  # Returns figure

# Multiple lines
y2 = np.array([1, 2, 3, 4, 5])
fig = multiple_lines(x, y, y2, ['squared', 'linear'])

# Styled line
fig = styled_line(x, y, 'red', '--', 'o')
\`\`\``,
			hint1: 'Используйте plt.subplots() для создания фигуры и осей, затем ax.plot() для рисования линий',
			hint2: 'Передайте color, linestyle, marker как именованные аргументы в ax.plot()',
			whyItMatters: `Линейные графики необходимы для:

- **Анализ временных рядов**: Визуализация трендов во времени
- **Обучение моделей**: Отслеживание кривых потерь и точности
- **Сравнение**: Сопоставление нескольких метрик
- **Коммуникация**: Ясное представление результатов`,
		},
		uz: {
			title: 'Chiziqli grafiklar',
			description: `# Chiziqli grafiklar

Chiziqli grafiklar trendlarni va uzluksiz ma'lumotlarni vaqt bo'yicha vizualizatsiya qilish uchun asosiy hisoblanadi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`simple_line(x, y)\` - Oddiy chiziqli grafik yaratish
2. \`multiple_lines(x, y1, y2, labels)\` - Belgilar bilan bir nechta chiziq chizish
3. \`styled_line(x, y, color, linestyle, marker)\` - Stillashtirilgan chiziqli grafik

## Misol

\`\`\`python
import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 4, 9, 16, 25])

fig = simple_line(x, y)  # Returns figure

# Multiple lines
y2 = np.array([1, 2, 3, 4, 5])
fig = multiple_lines(x, y, y2, ['squared', 'linear'])

# Styled line
fig = styled_line(x, y, 'red', '--', 'o')
\`\`\``,
			hint1: "Figura va o'qlarni yaratish uchun plt.subplots(), chiziqlarni chizish uchun ax.plot() dan foydalaning",
			hint2: "ax.plot() ga color, linestyle, marker ni nomlangan argument sifatida bering",
			whyItMatters: `Chiziqli grafiklar quyidagilar uchun zarur:

- **Vaqt qatorlari tahlili**: Vaqt bo'yicha trendlarni vizualizatsiya qilish
- **Model o'qitish**: Loss va accuracy egri chiziqlarini kuzatish
- **Taqqoslash**: Bir nechta metrikalarni yonma-yon solishtirish`,
		},
	},
};

export default task;
