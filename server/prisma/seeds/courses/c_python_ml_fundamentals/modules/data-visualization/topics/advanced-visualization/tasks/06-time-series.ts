import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'matplotlib-time-series',
	title: 'Time Series Visualization',
	difficulty: 'medium',
	tags: ['matplotlib', 'time-series', 'dates'],
	estimatedTime: '15m',
	isPremium: false,
	order: 6,
	description: `# Time Series Visualization

Visualize temporal data with proper date formatting and trend analysis.

## Task

Implement three functions:
1. \`plot_time_series(dates, values)\` - Basic time series with date axis
2. \`plot_with_trend(dates, values, window)\` - Time series with rolling average
3. \`plot_seasonal_decomposition(dates, values)\` - Show trend, seasonal, residual

## Example

\`\`\`python
dates = pd.date_range('2023-01-01', periods=365, freq='D')
values = np.cumsum(np.random.randn(365))

fig = plot_time_series(dates, values)
fig = plot_with_trend(dates, values, 30)  # 30-day rolling average
fig = plot_seasonal_decomposition(dates, values)
\`\`\``,

	initialCode: `import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

def plot_time_series(dates, values: np.ndarray):
    """Basic time series with date axis. Return figure."""
    # Your code here
    pass

def plot_with_trend(dates, values: np.ndarray, window: int):
    """Time series with rolling average trend. Return figure."""
    # Your code here
    pass

def plot_seasonal_decomposition(dates, values: np.ndarray):
    """Show trend, seasonal, residual components. Return figure."""
    # Your code here
    pass
`,

	solutionCode: `import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

def plot_time_series(dates, values: np.ndarray):
    """Basic time series with date axis. Return figure."""
    fig, ax = plt.subplots()
    ax.plot(dates, values)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    fig.autofmt_xdate()
    return fig

def plot_with_trend(dates, values: np.ndarray, window: int):
    """Time series with rolling average trend. Return figure."""
    fig, ax = plt.subplots()
    series = pd.Series(values, index=dates)
    ax.plot(dates, values, alpha=0.5, label='Original')
    ax.plot(dates, series.rolling(window).mean(), 'r-', linewidth=2, label=f'{window}-day MA')
    ax.legend()
    fig.autofmt_xdate()
    return fig

def plot_seasonal_decomposition(dates, values: np.ndarray):
    """Show trend, seasonal, residual components. Return figure."""
    series = pd.Series(values, index=dates)
    decomposition = seasonal_decompose(series, model='additive', period=30)
    fig, axes = plt.subplots(4, 1, figsize=(10, 8))
    decomposition.observed.plot(ax=axes[0], title='Observed')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
    decomposition.resid.plot(ax=axes[3], title='Residual')
    fig.tight_layout()
    return fig
`,

	testCode: `import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import unittest

class TestTimeSeriesVisualization(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.values = np.cumsum(np.random.randn(100))
        plt.close('all')

    def test_plot_time_series_returns_figure(self):
        fig = plot_time_series(self.dates, self.values)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_time_series_has_line(self):
        fig = plot_time_series(self.dates, self.values)
        ax = fig.axes[0]
        self.assertGreater(len(ax.lines), 0)

    def test_plot_with_trend_returns_figure(self):
        fig = plot_with_trend(self.dates, self.values, 7)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_with_trend_has_two_lines(self):
        fig = plot_with_trend(self.dates, self.values, 7)
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 2)

    def test_plot_with_trend_has_legend(self):
        fig = plot_with_trend(self.dates, self.values, 7)
        ax = fig.axes[0]
        self.assertIsNotNone(ax.get_legend())

    def test_plot_seasonal_decomposition_returns_figure(self):
        fig = plot_seasonal_decomposition(self.dates, self.values)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_seasonal_decomposition_has_four_axes(self):
        fig = plot_seasonal_decomposition(self.dates, self.values)
        self.assertEqual(len(fig.axes), 4)

    def test_plot_time_series_has_axes(self):
        fig = plot_time_series(self.dates, self.values)
        self.assertEqual(len(fig.axes), 1)

    def test_plot_with_trend_has_axes(self):
        fig = plot_with_trend(self.dates, self.values, 7)
        self.assertEqual(len(fig.axes), 1)

    def test_plot_with_different_window(self):
        fig = plot_with_trend(self.dates, self.values, 14)
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 2)

    def tearDown(self):
        plt.close('all')
`,

	hint1: 'Use fig.autofmt_xdate() to format date labels, pd.Series.rolling() for moving average',
	hint2: 'Use statsmodels.tsa.seasonal.seasonal_decompose for decomposition',

	whyItMatters: `Time series visualization is critical for:

- **Trend analysis**: Identify long-term patterns
- **Seasonality**: Detect periodic patterns
- **Anomaly detection**: Spot unusual events
- **Forecasting**: Understand data before modeling

Foundation for time series machine learning.`,

	translations: {
		ru: {
			title: 'Визуализация временных рядов',
			description: `# Визуализация временных рядов

Визуализируйте временные данные с правильным форматированием дат и анализом трендов.

## Задача

Реализуйте три функции:
1. \`plot_time_series(dates, values)\` - Базовый временной ряд с осью дат
2. \`plot_with_trend(dates, values, window)\` - Временной ряд со скользящим средним
3. \`plot_seasonal_decomposition(dates, values)\` - Показать тренд, сезонность, остаток

## Пример

\`\`\`python
dates = pd.date_range('2023-01-01', periods=365, freq='D')
values = np.cumsum(np.random.randn(365))

fig = plot_time_series(dates, values)
fig = plot_with_trend(dates, values, 30)  # 30-day rolling average
fig = plot_seasonal_decomposition(dates, values)
\`\`\``,
			hint1: 'Используйте fig.autofmt_xdate() для форматирования дат',
			hint2: 'Используйте statsmodels.tsa.seasonal.seasonal_decompose для декомпозиции',
			whyItMatters: `Визуализация временных рядов критична для:

- **Анализ трендов**: Определение долгосрочных паттернов
- **Сезонность**: Обнаружение периодических паттернов
- **Обнаружение аномалий**: Выявление необычных событий`,
		},
		uz: {
			title: 'Vaqt qatorlarini vizualizatsiya qilish',
			description: `# Vaqt qatorlarini vizualizatsiya qilish

Vaqtinchalik ma'lumotlarni to'g'ri sana formatlash va trend tahlili bilan vizualizatsiya qiling.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`plot_time_series(dates, values)\` - Sana o'qi bilan oddiy vaqt qatori
2. \`plot_with_trend(dates, values, window)\` - Sirpanuvchi o'rtacha bilan vaqt qatori
3. \`plot_seasonal_decomposition(dates, values)\` - Trend, mavsumiy, qoldiq ko'rsatish

## Misol

\`\`\`python
dates = pd.date_range('2023-01-01', periods=365, freq='D')
values = np.cumsum(np.random.randn(365))

fig = plot_time_series(dates, values)
fig = plot_with_trend(dates, values, 30)  # 30-day rolling average
fig = plot_seasonal_decomposition(dates, values)
\`\`\``,
			hint1: "Sana teglarini formatlash uchun fig.autofmt_xdate() dan foydalaning",
			hint2: "Dekompozitsiya uchun statsmodels.tsa.seasonal.seasonal_decompose dan foydalaning",
			whyItMatters: `Vaqt qatorlarini vizualizatsiya qilish quyidagilar uchun muhim:

- **Trend tahlili**: Uzoq muddatli naqshlarni aniqlash
- **Mavsumiylik**: Davriy naqshlarni aniqlash
- **Anomaliya aniqlash**: G'ayrioddiy hodisalarni topish`,
		},
	},
};

export default task;
