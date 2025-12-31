import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'sklearn-regression-metrics',
	title: 'Regression Metrics',
	difficulty: 'medium',
	tags: ['sklearn', 'metrics', 'regression'],
	estimatedTime: '12m',
	isPremium: false,
	order: 4,
	description: `# Regression Metrics

Evaluate regression models with MSE, RMSE, MAE, and R-squared.

## Task

Implement four functions:
1. \`compute_mse(y_true, y_pred)\` - Mean Squared Error
2. \`compute_rmse(y_true, y_pred)\` - Root Mean Squared Error
3. \`compute_mae(y_true, y_pred)\` - Mean Absolute Error
4. \`compute_r2(y_true, y_pred)\` - R-squared score

## Example

\`\`\`python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_true = [3.0, 5.0, 2.5, 7.0]
y_pred = [2.5, 5.0, 3.0, 6.5]

mse = compute_mse(y_true, y_pred)  # 0.1875
r2 = compute_r2(y_true, y_pred)    # 0.95
\`\`\``,

	initialCode: `import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_mse(y_true, y_pred) -> float:
    """Calculate Mean Squared Error."""
    # Your code here
    pass

def compute_rmse(y_true, y_pred) -> float:
    """Calculate Root Mean Squared Error."""
    # Your code here
    pass

def compute_mae(y_true, y_pred) -> float:
    """Calculate Mean Absolute Error."""
    # Your code here
    pass

def compute_r2(y_true, y_pred) -> float:
    """Calculate R-squared score."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_mse(y_true, y_pred) -> float:
    """Calculate Mean Squared Error."""
    return mean_squared_error(y_true, y_pred)

def compute_rmse(y_true, y_pred) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def compute_mae(y_true, y_pred) -> float:
    """Calculate Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)

def compute_r2(y_true, y_pred) -> float:
    """Calculate R-squared score."""
    return r2_score(y_true, y_pred)
`,

	testCode: `import numpy as np
import unittest

class TestRegressionMetrics(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([3.0, 5.0, 2.5, 7.0, 4.0])
        self.y_pred = np.array([2.5, 5.0, 3.0, 6.5, 4.5])

    def test_mse_non_negative(self):
        mse = compute_mse(self.y_true, self.y_pred)
        self.assertGreaterEqual(mse, 0)

    def test_rmse_is_sqrt_mse(self):
        mse = compute_mse(self.y_true, self.y_pred)
        rmse = compute_rmse(self.y_true, self.y_pred)
        self.assertAlmostEqual(rmse, np.sqrt(mse), places=5)

    def test_mae_non_negative(self):
        mae = compute_mae(self.y_true, self.y_pred)
        self.assertGreaterEqual(mae, 0)

    def test_r2_perfect_prediction(self):
        r2 = compute_r2(self.y_true, self.y_true)
        self.assertAlmostEqual(r2, 1.0, places=5)

    def test_r2_reasonable(self):
        r2 = compute_r2(self.y_true, self.y_pred)
        self.assertTrue(-1 <= r2 <= 1)

    def test_mse_returns_float(self):
        mse = compute_mse(self.y_true, self.y_pred)
        self.assertIsInstance(mse, float)

    def test_rmse_returns_float(self):
        rmse = compute_rmse(self.y_true, self.y_pred)
        self.assertIsInstance(rmse, float)

    def test_mae_returns_float(self):
        mae = compute_mae(self.y_true, self.y_pred)
        self.assertIsInstance(mae, float)

    def test_mse_zero_for_perfect(self):
        mse = compute_mse(self.y_true, self.y_true)
        self.assertEqual(mse, 0.0)

    def test_mae_less_than_rmse(self):
        mae = compute_mae(self.y_true, self.y_pred)
        rmse = compute_rmse(self.y_true, self.y_pred)
        self.assertLessEqual(mae, rmse)
`,

	hint1: 'Use mean_squared_error, mean_absolute_error, r2_score from sklearn.metrics',
	hint2: 'RMSE is np.sqrt(mean_squared_error(y_true, y_pred))',

	whyItMatters: `Regression metrics help you:

- **MSE/RMSE**: Penalize large errors more heavily
- **MAE**: Robust to outliers, interpretable units
- **R-squared**: Explained variance proportion
- **Model selection**: Compare different models objectively

Choose based on error tolerance and outlier sensitivity.`,

	translations: {
		ru: {
			title: 'Метрики регрессии',
			description: `# Метрики регрессии

Оценивайте модели регрессии с помощью MSE, RMSE, MAE и R-squared.

## Задача

Реализуйте четыре функции:
1. \`compute_mse(y_true, y_pred)\` - Среднеквадратичная ошибка
2. \`compute_rmse(y_true, y_pred)\` - Корень из MSE
3. \`compute_mae(y_true, y_pred)\` - Средняя абсолютная ошибка
4. \`compute_r2(y_true, y_pred)\` - Коэффициент детерминации

## Пример

\`\`\`python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_true = [3.0, 5.0, 2.5, 7.0]
y_pred = [2.5, 5.0, 3.0, 6.5]

mse = compute_mse(y_true, y_pred)  # 0.1875
r2 = compute_r2(y_true, y_pred)    # 0.95
\`\`\``,
			hint1: 'Используйте mean_squared_error, mean_absolute_error, r2_score',
			hint2: 'RMSE это np.sqrt(mean_squared_error(y_true, y_pred))',
			whyItMatters: `Метрики регрессии помогают:

- **MSE/RMSE**: Штрафуют большие ошибки сильнее
- **MAE**: Устойчива к выбросам, интерпретируемые единицы
- **R-squared**: Доля объясненной дисперсии`,
		},
		uz: {
			title: 'Regressiya metrikalari',
			description: `# Regressiya metrikalari

Regressiya modellarini MSE, RMSE, MAE va R-squared bilan baholang.

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`compute_mse(y_true, y_pred)\` - O'rtacha kvadrat xato
2. \`compute_rmse(y_true, y_pred)\` - MSE ning ildizi
3. \`compute_mae(y_true, y_pred)\` - O'rtacha mutlaq xato
4. \`compute_r2(y_true, y_pred)\` - R-squared bali

## Misol

\`\`\`python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_true = [3.0, 5.0, 2.5, 7.0]
y_pred = [2.5, 5.0, 3.0, 6.5]

mse = compute_mse(y_true, y_pred)  # 0.1875
r2 = compute_r2(y_true, y_pred)    # 0.95
\`\`\``,
			hint1: "mean_squared_error, mean_absolute_error, r2_score dan foydalaning",
			hint2: "RMSE bu np.sqrt(mean_squared_error(y_true, y_pred))",
			whyItMatters: `Regressiya metrikalari yordam beradi:

- **MSE/RMSE**: Katta xatolarni ko'proq jazoylaydi
- **MAE**: Chekinishlarga chidamli, tushunarli birliklar
- **R-squared**: Tushuntirilgan dispersiya ulushi`,
		},
	},
};

export default task;
