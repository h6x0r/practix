import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pandas-window-functions',
	title: 'Window Functions',
	difficulty: 'medium',
	tags: ['pandas', 'window', 'rolling', 'time-series'],
	estimatedTime: '15m',
	isPremium: false,
	order: 3,
	description: `# Window Functions

Window functions compute values over a sliding window of rows - essential for time series analysis.

## Task

Implement four functions:
1. \`rolling_mean(df, column, window)\` - Calculate rolling mean
2. \`rolling_sum(df, column, window)\` - Calculate rolling sum
3. \`expanding_mean(df, column)\` - Calculate expanding (cumulative) mean
4. \`percent_change(df, column, periods)\` - Calculate percent change over periods

## Example

\`\`\`python
df = pd.DataFrame({'value': [10, 20, 30, 40, 50]})

rolling_mean(df, 'value', 3)  # NaN, NaN, 20, 30, 40
rolling_sum(df, 'value', 2)   # NaN, 30, 50, 70, 90
expanding_mean(df, 'value')   # 10, 15, 20, 25, 30
percent_change(df, 'value', 1)  # NaN, 100%, 50%, 33%, 25%
\`\`\``,

	initialCode: `import pandas as pd

def rolling_mean(df: pd.DataFrame, column: str, window: int) -> pd.DataFrame:
    """Calculate rolling mean."""
    # Your code here
    pass

def rolling_sum(df: pd.DataFrame, column: str, window: int) -> pd.DataFrame:
    """Calculate rolling sum."""
    # Your code here
    pass

def expanding_mean(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Calculate expanding (cumulative) mean."""
    # Your code here
    pass

def percent_change(df: pd.DataFrame, column: str, periods: int = 1) -> pd.DataFrame:
    """Calculate percent change over periods."""
    # Your code here
    pass
`,

	solutionCode: `import pandas as pd

def rolling_mean(df: pd.DataFrame, column: str, window: int) -> pd.DataFrame:
    """Calculate rolling mean."""
    df = df.copy()
    df[f'{column}_rolling_mean'] = df[column].rolling(window=window).mean()
    return df

def rolling_sum(df: pd.DataFrame, column: str, window: int) -> pd.DataFrame:
    """Calculate rolling sum."""
    df = df.copy()
    df[f'{column}_rolling_sum'] = df[column].rolling(window=window).sum()
    return df

def expanding_mean(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Calculate expanding (cumulative) mean."""
    df = df.copy()
    df[f'{column}_expanding_mean'] = df[column].expanding().mean()
    return df

def percent_change(df: pd.DataFrame, column: str, periods: int = 1) -> pd.DataFrame:
    """Calculate percent change over periods."""
    df = df.copy()
    df[f'{column}_pct_change'] = df[column].pct_change(periods=periods)
    return df
`,

	testCode: `import pandas as pd
import numpy as np
import unittest

class TestWindowFunctions(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'value': [10.0, 20.0, 30.0, 40.0, 50.0]})

    def test_rolling_mean_basic(self):
        result = rolling_mean(self.df, 'value', 3)
        self.assertIn('value_rolling_mean', result.columns)
        self.assertTrue(np.isnan(result['value_rolling_mean'].iloc[0]))
        self.assertAlmostEqual(result['value_rolling_mean'].iloc[2], 20.0)

    def test_rolling_mean_values(self):
        result = rolling_mean(self.df, 'value', 2)
        self.assertAlmostEqual(result['value_rolling_mean'].iloc[1], 15.0)
        self.assertAlmostEqual(result['value_rolling_mean'].iloc[4], 45.0)

    def test_rolling_sum_basic(self):
        result = rolling_sum(self.df, 'value', 2)
        self.assertIn('value_rolling_sum', result.columns)
        self.assertAlmostEqual(result['value_rolling_sum'].iloc[1], 30.0)

    def test_rolling_sum_window3(self):
        result = rolling_sum(self.df, 'value', 3)
        self.assertAlmostEqual(result['value_rolling_sum'].iloc[2], 60.0)

    def test_expanding_mean_basic(self):
        result = expanding_mean(self.df, 'value')
        self.assertIn('value_expanding_mean', result.columns)
        self.assertAlmostEqual(result['value_expanding_mean'].iloc[0], 10.0)
        self.assertAlmostEqual(result['value_expanding_mean'].iloc[1], 15.0)

    def test_expanding_mean_cumulative(self):
        result = expanding_mean(self.df, 'value')
        self.assertAlmostEqual(result['value_expanding_mean'].iloc[4], 30.0)

    def test_percent_change_basic(self):
        result = percent_change(self.df, 'value', 1)
        self.assertIn('value_pct_change', result.columns)
        self.assertTrue(np.isnan(result['value_pct_change'].iloc[0]))
        self.assertAlmostEqual(result['value_pct_change'].iloc[1], 1.0)  # 100%

    def test_percent_change_periods(self):
        result = percent_change(self.df, 'value', 2)
        self.assertTrue(np.isnan(result['value_pct_change'].iloc[1]))
        self.assertAlmostEqual(result['value_pct_change'].iloc[2], 2.0)

    def test_preserves_original(self):
        result = rolling_mean(self.df, 'value', 2)
        self.assertEqual(result['value'].tolist(), self.df['value'].tolist())

    def test_no_modify_original(self):
        _ = rolling_mean(self.df, 'value', 2)
        self.assertNotIn('value_rolling_mean', self.df.columns)
`,

	hint1: 'Use df["col"].rolling(window).mean() for rolling calculations',
	hint2: 'Use .expanding().mean() for cumulative mean, .pct_change() for percent change',

	whyItMatters: `Window functions are essential for time series:

- **Moving averages**: Smooth noisy data for trend analysis
- **Technical indicators**: Stock trading signals (SMA, EMA)
- **Anomaly detection**: Compare current vs historical values
- **Feature engineering**: Create lagged features for prediction

These are fundamental to financial and IoT data analysis.`,

	translations: {
		ru: {
			title: 'Оконные функции',
			description: `# Оконные функции

Оконные функции вычисляют значения над скользящим окном строк — необходимы для анализа временных рядов.

## Задача

Реализуйте четыре функции:
1. \`rolling_mean(df, column, window)\` - Вычислить скользящее среднее
2. \`rolling_sum(df, column, window)\` - Вычислить скользящую сумму
3. \`expanding_mean(df, column)\` - Вычислить расширяющееся (кумулятивное) среднее
4. \`percent_change(df, column, periods)\` - Вычислить процентное изменение

## Пример

\`\`\`python
df = pd.DataFrame({'value': [10, 20, 30, 40, 50]})

rolling_mean(df, 'value', 3)  # NaN, NaN, 20, 30, 40
rolling_sum(df, 'value', 2)   # NaN, 30, 50, 70, 90
expanding_mean(df, 'value')   # 10, 15, 20, 25, 30
percent_change(df, 'value', 1)  # NaN, 100%, 50%, 33%, 25%
\`\`\``,
			hint1: 'Используйте df["col"].rolling(window).mean() для скользящих вычислений',
			hint2: 'Используйте .expanding().mean() для кумулятивного среднего, .pct_change() для процентного изменения',
			whyItMatters: `Оконные функции необходимы для временных рядов:

- **Скользящие средние**: Сглаживание шумных данных для анализа трендов
- **Технические индикаторы**: Торговые сигналы (SMA, EMA)
- **Обнаружение аномалий**: Сравнение текущих и исторических значений
- **Feature engineering**: Создание лаговых признаков для прогнозирования`,
		},
		uz: {
			title: "Window funksiyalari",
			description: `# Window funksiyalari

Window funksiyalari sirpanuvchi qatorlar oynasi ustida qiymatlarni hisoblab chiqadi - vaqt qatorlari tahlili uchun zarur.

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`rolling_mean(df, column, window)\` - Sirpanuvchi o'rtachani hisoblash
2. \`rolling_sum(df, column, window)\` - Sirpanuvchi yig'indini hisoblash
3. \`expanding_mean(df, column)\` - Kengayuvchi (kumulyativ) o'rtachani hisoblash
4. \`percent_change(df, column, periods)\` - Foiz o'zgarishini hisoblash

## Misol

\`\`\`python
df = pd.DataFrame({'value': [10, 20, 30, 40, 50]})

rolling_mean(df, 'value', 3)  # NaN, NaN, 20, 30, 40
rolling_sum(df, 'value', 2)   # NaN, 30, 50, 70, 90
expanding_mean(df, 'value')   # 10, 15, 20, 25, 30
percent_change(df, 'value', 1)  # NaN, 100%, 50%, 33%, 25%
\`\`\``,
			hint1: "Sirpanuvchi hisob-kitoblar uchun df['col'].rolling(window).mean() dan foydalaning",
			hint2: "Kumulyativ o'rtacha uchun .expanding().mean(), foiz o'zgarishi uchun .pct_change() dan foydalaning",
			whyItMatters: `Window funksiyalari vaqt qatorlari uchun zarur:

- **Sirpanuvchi o'rtachalar**: Trend tahlili uchun shovqinli ma'lumotlarni tekislash
- **Texnik ko'rsatkichlar**: Savdo signallari (SMA, EMA)
- **Anomaliya aniqlash**: Joriy va tarixiy qiymatlarni taqqoslash`,
		},
	},
};

export default task;
