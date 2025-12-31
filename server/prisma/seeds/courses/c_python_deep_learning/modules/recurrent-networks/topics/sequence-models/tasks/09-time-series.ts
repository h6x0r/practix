import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-time-series-forecasting',
	title: 'Time Series Forecasting',
	difficulty: 'medium',
	tags: ['pytorch', 'lstm', 'time-series'],
	estimatedTime: '18m',
	isPremium: false,
	order: 9,
	description: `# Time Series Forecasting

Build an LSTM model for predicting future values in time series.

## Task

Implement a \`TimeSeriesLSTM\` class that:
- Takes a sequence of past values
- Predicts the next N values
- Supports multi-step forecasting

## Example

\`\`\`python
model = TimeSeriesLSTM(
    input_size=1,
    hidden_size=64,
    forecast_horizon=5
)

# Past 30 values -> predict next 5
x = torch.randn(4, 30, 1)  # batch, seq_len, features
predictions = model(x)
# predictions.shape = (4, 5)
\`\`\``,

	initialCode: `import torch
import torch.nn as nn

class TimeSeriesLSTM(nn.Module):
    """LSTM for time series forecasting."""

    def __init__(self, input_size: int, hidden_size: int,
                 forecast_horizon: int, num_layers: int = 2):
        super().__init__()
        # Your code here
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict next forecast_horizon values."""
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn as nn

class TimeSeriesLSTM(nn.Module):
    """LSTM for time series forecasting."""

    def __init__(self, input_size: int, hidden_size: int,
                 forecast_horizon: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, forecast_horizon)
        self.forecast_horizon = forecast_horizon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict next forecast_horizon values."""
        # Process sequence
        _, (hidden, _) = self.lstm(x)
        # Use last layer hidden state
        out = hidden[-1]
        # Predict all future steps at once
        predictions = self.fc(out)
        return predictions
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class TestTimeSeries(unittest.TestCase):
    def test_output_shape(self):
        model = TimeSeriesLSTM(1, 64, 5)
        x = torch.randn(4, 30, 1)
        out = model(x)
        self.assertEqual(out.shape, (4, 5))

    def test_different_horizon(self):
        model = TimeSeriesLSTM(1, 32, 10)
        x = torch.randn(2, 50, 1)
        out = model(x)
        self.assertEqual(out.shape, (2, 10))

    def test_multivariate(self):
        model = TimeSeriesLSTM(3, 64, 5)
        x = torch.randn(4, 30, 3)  # 3 features
        out = model(x)
        self.assertEqual(out.shape, (4, 5))

    def test_is_nn_module(self):
        model = TimeSeriesLSTM(1, 64, 5)
        self.assertIsInstance(model, nn.Module)

    def test_has_lstm(self):
        model = TimeSeriesLSTM(1, 64, 5)
        self.assertTrue(hasattr(model, 'lstm'))

    def test_has_fc(self):
        model = TimeSeriesLSTM(1, 64, 5)
        self.assertTrue(hasattr(model, 'fc'))

    def test_single_sample(self):
        model = TimeSeriesLSTM(1, 64, 5)
        x = torch.randn(1, 30, 1)
        out = model(x)
        self.assertEqual(out.shape, (1, 5))

    def test_different_seq_len(self):
        model = TimeSeriesLSTM(1, 64, 5)
        for seq_len in [10, 50, 100]:
            x = torch.randn(2, seq_len, 1)
            out = model(x)
            self.assertEqual(out.shape, (2, 5))

    def test_output_not_nan(self):
        model = TimeSeriesLSTM(1, 64, 5)
        x = torch.randn(4, 30, 1)
        out = model(x)
        self.assertFalse(torch.isnan(out).any())

    def test_has_forecast_horizon(self):
        model = TimeSeriesLSTM(1, 64, 7)
        self.assertEqual(model.forecast_horizon, 7)
`,

	hint1: 'Use hidden state of last layer for prediction',
	hint2: 'FC layer outputs all forecast_horizon steps at once',

	whyItMatters: `Time series forecasting is widely used:

- **Stock prediction**: Forecast prices and trends
- **Weather forecasting**: Predict temperature, rain
- **Demand planning**: Forecast product demand
- **Anomaly detection**: Identify unusual patterns

LSTMs are still competitive for time series despite transformers.`,

	translations: {
		ru: {
			title: 'Прогнозирование временных рядов',
			description: `# Прогнозирование временных рядов

Создайте LSTM модель для прогнозирования будущих значений временных рядов.

## Задача

Реализуйте класс \`TimeSeriesLSTM\`, который:
- Принимает последовательность прошлых значений
- Предсказывает следующие N значений
- Поддерживает многошаговое прогнозирование

## Пример

\`\`\`python
model = TimeSeriesLSTM(
    input_size=1,
    hidden_size=64,
    forecast_horizon=5
)

# Past 30 values -> predict next 5
x = torch.randn(4, 30, 1)  # batch, seq_len, features
predictions = model(x)
# predictions.shape = (4, 5)
\`\`\``,
			hint1: 'Используйте hidden state последнего слоя для предсказания',
			hint2: 'FC слой выдает все forecast_horizon шагов сразу',
			whyItMatters: `Прогнозирование временных рядов широко используется:

- **Прогноз акций**: Предсказание цен и трендов
- **Прогноз погоды**: Предсказание температуры, осадков
- **Планирование спроса**: Прогноз спроса на продукцию
- **Детекция аномалий**: Выявление необычных паттернов`,
		},
		uz: {
			title: "Vaqt qatorlarini bashorat qilish",
			description: `# Vaqt qatorlarini bashorat qilish

Vaqt qatorlaridagi kelajak qiymatlarni bashorat qilish uchun LSTM modelini yarating.

## Topshiriq

\`TimeSeriesLSTM\` sinfini amalga oshiring:
- O'tgan qiymatlar ketma-ketligini qabul qiladi
- Keyingi N qiymatlarni bashorat qiladi
- Ko'p qadamli bashoratni qo'llab-quvvatlaydi

## Misol

\`\`\`python
model = TimeSeriesLSTM(
    input_size=1,
    hidden_size=64,
    forecast_horizon=5
)

# Past 30 values -> predict next 5
x = torch.randn(4, 30, 1)  # batch, seq_len, features
predictions = model(x)
# predictions.shape = (4, 5)
\`\`\``,
			hint1: "Bashorat uchun oxirgi qatlam hidden state dan foydalaning",
			hint2: "FC qatlam barcha forecast_horizon qadamlarni bir vaqtda chiqaradi",
			whyItMatters: `Vaqt qatorlarini bashorat qilish keng qo'llaniladi:

- **Aksiya bashorati**: Narxlar va trendlarni bashorat qilish
- **Ob-havo bashorati**: Harorat, yog'ingarchilikni bashorat qilish
- **Talab rejalashtirish**: Mahsulot talabini bashorat qilish
- **Anomaliya aniqlash**: G'ayrioddiy naqshlarni aniqlash`,
		},
	},
};

export default task;
