import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'matplotlib-saving-figures',
	title: 'Saving Figures',
	difficulty: 'easy',
	tags: ['matplotlib', 'save', 'export'],
	estimatedTime: '10m',
	isPremium: false,
	order: 8,
	description: `# Saving Figures

Save figures to various formats with quality settings for reports and presentations.

## Task

Implement three functions:
1. \`save_figure(fig, filename, dpi)\` - Save figure with specified DPI
2. \`save_transparent(fig, filename)\` - Save with transparent background
3. \`save_tight(fig, filename)\` - Save with tight bounding box (no whitespace)

## Example

\`\`\`python
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])

# Save with high resolution
save_figure(fig, 'plot.png', 300)

# Save with transparent background
save_transparent(fig, 'plot_transparent.png')

# Save with tight bounds
save_tight(fig, 'plot_tight.pdf')
\`\`\``,

	initialCode: `import matplotlib.pyplot as plt
import numpy as np
import io

def save_figure(fig, filename: str, dpi: int) -> bytes:
    """Save figure with specified DPI. Return bytes for testing."""
    # Your code here
    pass

def save_transparent(fig, filename: str) -> bytes:
    """Save with transparent background. Return bytes for testing."""
    # Your code here
    pass

def save_tight(fig, filename: str) -> bytes:
    """Save with tight bounding box. Return bytes for testing."""
    # Your code here
    pass
`,

	solutionCode: `import matplotlib.pyplot as plt
import numpy as np
import io

def save_figure(fig, filename: str, dpi: int) -> bytes:
    """Save figure with specified DPI. Return bytes for testing."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi)
    buf.seek(0)
    return buf.getvalue()

def save_transparent(fig, filename: str) -> bytes:
    """Save with transparent background. Return bytes for testing."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=True)
    buf.seek(0)
    return buf.getvalue()

def save_tight(fig, filename: str) -> bytes:
    """Save with tight bounding box. Return bytes for testing."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()
`,

	testCode: `import matplotlib.pyplot as plt
import numpy as np
import unittest

class TestSavingFigures(unittest.TestCase):
    def setUp(self):
        self.fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

    def test_save_figure_returns_bytes(self):
        result = save_figure(self.fig, 'test.png', 100)
        self.assertIsInstance(result, bytes)

    def test_save_figure_not_empty(self):
        result = save_figure(self.fig, 'test.png', 100)
        self.assertGreater(len(result), 0)

    def test_save_figure_is_png(self):
        result = save_figure(self.fig, 'test.png', 100)
        # PNG files start with specific bytes
        self.assertTrue(result.startswith(b'\\x89PNG'))

    def test_save_transparent_returns_bytes(self):
        result = save_transparent(self.fig, 'test.png')
        self.assertIsInstance(result, bytes)

    def test_save_transparent_not_empty(self):
        result = save_transparent(self.fig, 'test.png')
        self.assertGreater(len(result), 0)

    def test_save_tight_returns_bytes(self):
        result = save_tight(self.fig, 'test.png')
        self.assertIsInstance(result, bytes)

    def test_save_tight_not_empty(self):
        result = save_tight(self.fig, 'test.png')
        self.assertGreater(len(result), 0)

    def test_higher_dpi_larger_file(self):
        low_dpi = save_figure(self.fig, 'test.png', 50)
        high_dpi = save_figure(self.fig, 'test.png', 200)
        self.assertGreater(len(high_dpi), len(low_dpi))

    def test_save_transparent_is_png(self):
        result = save_transparent(self.fig, 'test.png')
        self.assertTrue(result.startswith(b'\\x89PNG'))

    def test_save_tight_is_png(self):
        result = save_tight(self.fig, 'test.png')
        self.assertTrue(result.startswith(b'\\x89PNG'))

    def tearDown(self):
        plt.close('all')
`,

	hint1: 'Use fig.savefig(filename, dpi=dpi) to save with resolution',
	hint2: 'Use transparent=True for transparent, bbox_inches="tight" for tight bounds',

	whyItMatters: `Saving figures properly is essential for:

- **Reports**: High-resolution images for documents
- **Presentations**: Correct format and transparency
- **Web**: Optimized file sizes
- **Publications**: Vector formats (PDF, SVG) for journals

Proper export ensures your visualizations look great everywhere.`,

	translations: {
		ru: {
			title: 'Сохранение фигур',
			description: `# Сохранение фигур

Сохраняйте фигуры в различных форматах с настройками качества для отчётов и презентаций.

## Задача

Реализуйте три функции:
1. \`save_figure(fig, filename, dpi)\` - Сохранить фигуру с указанным DPI
2. \`save_transparent(fig, filename)\` - Сохранить с прозрачным фоном
3. \`save_tight(fig, filename)\` - Сохранить с плотными границами

## Пример

\`\`\`python
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])

# Save with high resolution
save_figure(fig, 'plot.png', 300)

# Save with transparent background
save_transparent(fig, 'plot_transparent.png')

# Save with tight bounds
save_tight(fig, 'plot_tight.pdf')
\`\`\``,
			hint1: 'Используйте fig.savefig(filename, dpi=dpi) для сохранения с разрешением',
			hint2: 'Используйте transparent=True для прозрачности, bbox_inches="tight" для плотных границ',
			whyItMatters: `Правильное сохранение фигур необходимо для:

- **Отчёты**: Изображения высокого разрешения для документов
- **Презентации**: Правильный формат и прозрачность
- **Веб**: Оптимизированные размеры файлов`,
		},
		uz: {
			title: 'Figuralarni saqlash',
			description: `# Figuralarni saqlash

Hisobotlar va taqdimotlar uchun sifat sozlamalari bilan turli formatlarda figuralarni saqlang.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`save_figure(fig, filename, dpi)\` - Ko'rsatilgan DPI bilan figurani saqlash
2. \`save_transparent(fig, filename)\` - Shaffof fon bilan saqlash
3. \`save_tight(fig, filename)\` - Zich chegaralar bilan saqlash

## Misol

\`\`\`python
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])

# Save with high resolution
save_figure(fig, 'plot.png', 300)

# Save with transparent background
save_transparent(fig, 'plot_transparent.png')

# Save with tight bounds
save_tight(fig, 'plot_tight.pdf')
\`\`\``,
			hint1: "Ruxsat bilan saqlash uchun fig.savefig(filename, dpi=dpi) dan foydalaning",
			hint2: 'Shaffoflik uchun transparent=True, zich chegaralar uchun bbox_inches="tight" dan foydalaning',
			whyItMatters: `Figuralarni to'g'ri saqlash quyidagilar uchun zarur:

- **Hisobotlar**: Hujjatlar uchun yuqori aniqlikdagi rasmlar
- **Taqdimotlar**: To'g'ri format va shaffoflik
- **Veb**: Optimallashtirilgan fayl o'lchamlari`,
		},
	},
};

export default task;
