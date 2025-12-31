import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'matplotlib-3d-plots',
	title: '3D Plots',
	difficulty: 'hard',
	tags: ['matplotlib', '3d', 'surface'],
	estimatedTime: '15m',
	isPremium: true,
	order: 5,
	description: `# 3D Plots

3D plots visualize three-dimensional data for surface and spatial analysis.

## Task

Implement three functions:
1. \`scatter_3d(x, y, z)\` - 3D scatter plot
2. \`surface_3d(X, Y, Z)\` - 3D surface plot from meshgrid
3. \`wireframe_3d(X, Y, Z)\` - 3D wireframe plot

## Example

\`\`\`python
from mpl_toolkits.mplot3d import Axes3D

# 3D scatter
x = np.random.randn(100)
y = np.random.randn(100)
z = np.random.randn(100)
fig = scatter_3d(x, y, z)

# Surface plot
X, Y = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = np.sin(np.sqrt(X**2 + Y**2))
fig = surface_3d(X, Y, Z)
\`\`\``,

	initialCode: `import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def scatter_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """3D scatter plot. Return figure."""
    # Your code here
    pass

def surface_3d(X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
    """3D surface plot from meshgrid. Return figure."""
    # Your code here
    pass

def wireframe_3d(X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
    """3D wireframe plot. Return figure."""
    # Your code here
    pass
`,

	solutionCode: `import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def scatter_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """3D scatter plot. Return figure."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    return fig

def surface_3d(X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
    """3D surface plot from meshgrid. Return figure."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    return fig

def wireframe_3d(X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
    """3D wireframe plot. Return figure."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z)
    return fig
`,

	testCode: `import matplotlib.pyplot as plt
import numpy as np
import unittest

class Test3DPlots(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.x = np.random.randn(50)
        self.y = np.random.randn(50)
        self.z = np.random.randn(50)
        self.X, self.Y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
        self.Z = np.sin(self.X) * np.cos(self.Y)
        plt.close('all')

    def test_scatter_3d_returns_figure(self):
        fig = scatter_3d(self.x, self.y, self.z)
        self.assertIsInstance(fig, plt.Figure)

    def test_scatter_3d_has_3d_axes(self):
        fig = scatter_3d(self.x, self.y, self.z)
        ax = fig.axes[0]
        self.assertEqual(ax.name, '3d')

    def test_surface_3d_returns_figure(self):
        fig = surface_3d(self.X, self.Y, self.Z)
        self.assertIsInstance(fig, plt.Figure)

    def test_surface_3d_has_3d_axes(self):
        fig = surface_3d(self.X, self.Y, self.Z)
        ax = fig.axes[0]
        self.assertEqual(ax.name, '3d')

    def test_wireframe_3d_returns_figure(self):
        fig = wireframe_3d(self.X, self.Y, self.Z)
        self.assertIsInstance(fig, plt.Figure)

    def test_wireframe_3d_has_3d_axes(self):
        fig = wireframe_3d(self.X, self.Y, self.Z)
        ax = fig.axes[0]
        self.assertEqual(ax.name, '3d')

    def test_scatter_3d_has_one_axes(self):
        fig = scatter_3d(self.x, self.y, self.z)
        self.assertEqual(len(fig.axes), 1)

    def test_surface_3d_has_one_axes(self):
        fig = surface_3d(self.X, self.Y, self.Z)
        self.assertEqual(len(fig.axes), 1)

    def test_wireframe_3d_has_one_axes(self):
        fig = wireframe_3d(self.X, self.Y, self.Z)
        self.assertEqual(len(fig.axes), 1)

    def test_scatter_3d_with_different_data(self):
        x = np.random.randn(20)
        y = np.random.randn(20)
        z = np.random.randn(20)
        fig = scatter_3d(x, y, z)
        self.assertIsInstance(fig, plt.Figure)

    def tearDown(self):
        plt.close('all')
`,

	hint1: 'Use fig.add_subplot(111, projection="3d") to create 3D axes',
	hint2: 'Use ax.scatter(), ax.plot_surface(), ax.plot_wireframe() for different 3D plots',

	whyItMatters: `3D plots are useful for:

- **Loss landscapes**: Visualize optimization surfaces
- **Feature spaces**: See 3D embeddings (t-SNE, UMAP)
- **Scientific data**: Physical simulations and measurements
- **Decision boundaries**: Visualize classifier decisions

Advanced visualization for complex ML problems.`,

	translations: {
		ru: {
			title: '3D графики',
			description: `# 3D графики

3D графики визуализируют трёхмерные данные для анализа поверхностей и пространственного анализа.

## Задача

Реализуйте три функции:
1. \`scatter_3d(x, y, z)\` - 3D точечная диаграмма
2. \`surface_3d(X, Y, Z)\` - 3D поверхность из meshgrid
3. \`wireframe_3d(X, Y, Z)\` - 3D каркасный график

## Пример

\`\`\`python
from mpl_toolkits.mplot3d import Axes3D

# 3D scatter
x = np.random.randn(100)
y = np.random.randn(100)
z = np.random.randn(100)
fig = scatter_3d(x, y, z)

# Surface plot
X, Y = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = np.sin(np.sqrt(X**2 + Y**2))
fig = surface_3d(X, Y, Z)
\`\`\``,
			hint1: 'Используйте fig.add_subplot(111, projection="3d") для создания 3D осей',
			hint2: 'Используйте ax.scatter(), ax.plot_surface(), ax.plot_wireframe()',
			whyItMatters: `3D графики полезны для:

- **Ландшафты потерь**: Визуализация поверхностей оптимизации
- **Пространства признаков**: 3D вложения (t-SNE, UMAP)
- **Научные данные**: Физические симуляции`,
		},
		uz: {
			title: '3D grafiklar',
			description: `# 3D grafiklar

3D grafiklar uch o'lchovli ma'lumotlarni sirt va fazoviy tahlil uchun vizualizatsiya qiladi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`scatter_3d(x, y, z)\` - 3D nuqtali diagramma
2. \`surface_3d(X, Y, Z)\` - Meshgrid dan 3D sirt
3. \`wireframe_3d(X, Y, Z)\` - 3D simkash grafik

## Misol

\`\`\`python
from mpl_toolkits.mplot3d import Axes3D

# 3D scatter
x = np.random.randn(100)
y = np.random.randn(100)
z = np.random.randn(100)
fig = scatter_3d(x, y, z)

# Surface plot
X, Y = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = np.sin(np.sqrt(X**2 + Y**2))
fig = surface_3d(X, Y, Z)
\`\`\``,
			hint1: '3D o\'qlarni yaratish uchun fig.add_subplot(111, projection="3d") dan foydalaning',
			hint2: "ax.scatter(), ax.plot_surface(), ax.plot_wireframe() dan foydalaning",
			whyItMatters: `3D grafiklar quyidagilar uchun foydali:

- **Loss landshaftlari**: Optimallashtirish sirtlarini vizualizatsiya qilish
- **Xususiyat fazolari**: 3D embeddinglarni ko'rish (t-SNE, UMAP)
- **Ilmiy ma'lumotlar**: Fizik simulyatsiyalar`,
		},
	},
};

export default task;
