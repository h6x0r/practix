import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'matplotlib-legends-annotations',
	title: 'Legends and Annotations',
	difficulty: 'medium',
	tags: ['matplotlib', 'legend', 'annotation'],
	estimatedTime: '12m',
	isPremium: false,
	order: 7,
	description: `# Legends and Annotations

Legends and annotations add context and highlight important data points.

## Task

Implement three functions:
1. \`plot_with_legend(x, y1, y2, labels, location)\` - Plot with positioned legend
2. \`annotate_point(fig, x, y, text)\` - Annotate specific point with arrow
3. \`add_text_box(fig, text, position)\` - Add text box to figure

## Example

\`\`\`python
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

fig = plot_with_legend(x, y1, y2, ['sin', 'cos'], 'upper right')

# Annotate maximum point
fig = annotate_point(fig, np.pi/2, 1, 'Maximum')

# Add info text box
fig = add_text_box(fig, 'Period: 2π', (0.7, 0.9))
\`\`\``,

	initialCode: `import matplotlib.pyplot as plt
import numpy as np

def plot_with_legend(x: np.ndarray, y1: np.ndarray, y2: np.ndarray, labels: list, location: str):
    """Create plot with positioned legend. Return figure."""
    # Your code here
    pass

def annotate_point(fig, x: float, y: float, text: str):
    """Annotate specific point with arrow. Return figure."""
    # Your code here
    pass

def add_text_box(fig, text: str, position: tuple):
    """Add text box to figure. position is (x, y) in axes coordinates. Return figure."""
    # Your code here
    pass
`,

	solutionCode: `import matplotlib.pyplot as plt
import numpy as np

def plot_with_legend(x: np.ndarray, y1: np.ndarray, y2: np.ndarray, labels: list, location: str):
    """Create plot with positioned legend. Return figure."""
    fig, ax = plt.subplots()
    ax.plot(x, y1, label=labels[0])
    ax.plot(x, y2, label=labels[1])
    ax.legend(loc=location)
    return fig

def annotate_point(fig, x: float, y: float, text: str):
    """Annotate specific point with arrow. Return figure."""
    ax = fig.axes[0]
    ax.annotate(text, xy=(x, y), xytext=(x + 1, y + 0.2),
                arrowprops=dict(arrowstyle='->', color='black'))
    return fig

def add_text_box(fig, text: str, position: tuple):
    """Add text box to figure. position is (x, y) in axes coordinates. Return figure."""
    ax = fig.axes[0]
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(position[0], position[1], text, transform=ax.transAxes,
            verticalalignment='top', bbox=props)
    return fig
`,

	testCode: `import matplotlib.pyplot as plt
import numpy as np
import unittest

class TestLegendsAnnotations(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(0, 10, 50)
        self.y1 = np.sin(self.x)
        self.y2 = np.cos(self.x)
        plt.close('all')

    def test_plot_with_legend_returns_figure(self):
        fig = plot_with_legend(self.x, self.y1, self.y2, ['a', 'b'], 'upper right')
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_with_legend_has_legend(self):
        fig = plot_with_legend(self.x, self.y1, self.y2, ['a', 'b'], 'upper right')
        ax = fig.axes[0]
        self.assertIsNotNone(ax.get_legend())

    def test_plot_with_legend_has_two_lines(self):
        fig = plot_with_legend(self.x, self.y1, self.y2, ['a', 'b'], 'upper right')
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 2)

    def test_annotate_point_returns_figure(self):
        fig, ax = plt.subplots()
        ax.plot(self.x, self.y1)
        fig = annotate_point(fig, 1.57, 1.0, 'Max')
        self.assertIsInstance(fig, plt.Figure)

    def test_annotate_point_has_annotation(self):
        fig, ax = plt.subplots()
        ax.plot(self.x, self.y1)
        fig = annotate_point(fig, 1.57, 1.0, 'Max')
        ax = fig.axes[0]
        self.assertGreater(len(ax.texts), 0)

    def test_add_text_box_returns_figure(self):
        fig, ax = plt.subplots()
        ax.plot(self.x, self.y1)
        fig = add_text_box(fig, 'Info', (0.5, 0.9))
        self.assertIsInstance(fig, plt.Figure)

    def test_add_text_box_has_text(self):
        fig, ax = plt.subplots()
        ax.plot(self.x, self.y1)
        fig = add_text_box(fig, 'Info Text', (0.5, 0.9))
        ax = fig.axes[0]
        self.assertGreater(len(ax.texts), 0)

    def test_legend_labels_correct(self):
        fig = plot_with_legend(self.x, self.y1, self.y2, ['sin', 'cos'], 'upper left')
        ax = fig.axes[0]
        legend = ax.get_legend()
        texts = [t.get_text() for t in legend.get_texts()]
        self.assertIn('sin', texts)
        self.assertIn('cos', texts)

    def test_annotate_point_has_axes(self):
        fig, ax = plt.subplots()
        ax.plot(self.x, self.y1)
        fig = annotate_point(fig, 1.57, 1.0, 'Peak')
        self.assertEqual(len(fig.axes), 1)

    def test_add_text_box_has_axes(self):
        fig, ax = plt.subplots()
        ax.plot(self.x, self.y1)
        fig = add_text_box(fig, 'Note', (0.1, 0.1))
        self.assertEqual(len(fig.axes), 1)

    def tearDown(self):
        plt.close('all')
`,

	hint1: 'Use ax.legend(loc=location) for positioned legend',
	hint2: 'Use ax.annotate() with arrowprops for arrows, ax.text() with bbox for text boxes',

	whyItMatters: `Legends and annotations are crucial for:

- **Clarity**: Help readers understand what each element represents
- **Highlighting**: Draw attention to important data points
- **Context**: Provide additional information inline
- **Publications**: Required for scientific papers

Well-annotated plots tell a complete story.`,

	translations: {
		ru: {
			title: 'Легенды и аннотации',
			description: `# Легенды и аннотации

Легенды и аннотации добавляют контекст и выделяют важные точки данных.

## Задача

Реализуйте три функции:
1. \`plot_with_legend(x, y1, y2, labels, location)\` - График с позиционированной легендой
2. \`annotate_point(fig, x, y, text)\` - Аннотировать точку со стрелкой
3. \`add_text_box(fig, text, position)\` - Добавить текстовый блок

## Пример

\`\`\`python
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

fig = plot_with_legend(x, y1, y2, ['sin', 'cos'], 'upper right')

# Annotate maximum point
fig = annotate_point(fig, np.pi/2, 1, 'Maximum')

# Add info text box
fig = add_text_box(fig, 'Period: 2π', (0.7, 0.9))
\`\`\``,
			hint1: 'Используйте ax.legend(loc=location) для позиционированной легенды',
			hint2: 'Используйте ax.annotate() с arrowprops для стрелок, ax.text() с bbox для текстовых блоков',
			whyItMatters: `Легенды и аннотации критичны для:

- **Ясность**: Помочь читателям понять что представляет каждый элемент
- **Выделение**: Привлечь внимание к важным точкам
- **Контекст**: Предоставить дополнительную информацию`,
		},
		uz: {
			title: 'Legendalar va annotatsiyalar',
			description: `# Legendalar va annotatsiyalar

Legendalar va annotatsiyalar kontekst qo'shadi va muhim ma'lumotlar nuqtalarini ajratib ko'rsatadi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`plot_with_legend(x, y1, y2, labels, location)\` - Joylashtirilgan legenda bilan grafik
2. \`annotate_point(fig, x, y, text)\` - O'q bilan nuqtani annotatsiya qilish
3. \`add_text_box(fig, text, position)\` - Matn qutisini qo'shish

## Misol

\`\`\`python
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

fig = plot_with_legend(x, y1, y2, ['sin', 'cos'], 'upper right')

# Annotate maximum point
fig = annotate_point(fig, np.pi/2, 1, 'Maximum')

# Add info text box
fig = add_text_box(fig, 'Period: 2π', (0.7, 0.9))
\`\`\``,
			hint1: "Joylashtirilgan legenda uchun ax.legend(loc=location) dan foydalaning",
			hint2: "O'qlar uchun ax.annotate() bilan arrowprops, matn qutilari uchun ax.text() bilan bbox dan foydalaning",
			whyItMatters: `Legendalar va annotatsiyalar quyidagilar uchun muhim:

- **Aniqlik**: O'quvchilarga har bir element nimani ifodalashini tushunishga yordam berish
- **Ajratib ko'rsatish**: Muhim nuqtalarga e'tiborni jalb qilish
- **Kontekst**: Qo'shimcha ma'lumotlarni taqdim etish`,
		},
	},
};

export default task;
