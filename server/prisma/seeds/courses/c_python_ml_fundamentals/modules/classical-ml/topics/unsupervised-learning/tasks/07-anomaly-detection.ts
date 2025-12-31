import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'sklearn-anomaly-detection',
	title: 'Anomaly Detection',
	difficulty: 'hard',
	tags: ['sklearn', 'anomaly', 'outlier'],
	estimatedTime: '18m',
	isPremium: true,
	order: 7,
	description: `# Anomaly Detection

Identify unusual patterns using Isolation Forest and Local Outlier Factor.

## Task

Implement three functions:
1. \`isolation_forest(X, contamination)\` - Train Isolation Forest
2. \`local_outlier_factor(X, n_neighbors)\` - Apply LOF
3. \`compare_detectors(X)\` - Compare IF vs LOF predictions

## Example

\`\`\`python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

X = np.vstack([np.random.randn(100, 2), [[10, 10], [-10, -10]]])

if_labels = isolation_forest(X, contamination=0.02)
lof_labels = local_outlier_factor(X, n_neighbors=20)
comparison = compare_detectors(X)  # {'isolation_forest': [...], 'lof': [...]}
\`\`\``,

	initialCode: `import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def isolation_forest(X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
    """Train Isolation Forest. Return labels (1=normal, -1=anomaly)."""
    # Your code here
    pass

def local_outlier_factor(X: np.ndarray, n_neighbors: int = 20) -> np.ndarray:
    """Apply Local Outlier Factor. Return labels (1=normal, -1=anomaly)."""
    # Your code here
    pass

def compare_detectors(X: np.ndarray) -> dict:
    """Compare IF vs LOF. Return {'isolation_forest': labels, 'lof': labels}."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def isolation_forest(X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
    """Train Isolation Forest. Return labels (1=normal, -1=anomaly)."""
    clf = IsolationForest(contamination=contamination, random_state=42)
    return clf.fit_predict(X)

def local_outlier_factor(X: np.ndarray, n_neighbors: int = 20) -> np.ndarray:
    """Apply Local Outlier Factor. Return labels (1=normal, -1=anomaly)."""
    clf = LocalOutlierFactor(n_neighbors=n_neighbors)
    return clf.fit_predict(X)

def compare_detectors(X: np.ndarray) -> dict:
    """Compare IF vs LOF. Return {'isolation_forest': labels, 'lof': labels}."""
    return {
        'isolation_forest': isolation_forest(X, contamination=0.1),
        'lof': local_outlier_factor(X, n_neighbors=20)
    }
`,

	testCode: `import numpy as np
import unittest

class TestAnomalyDetection(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        normal = np.random.randn(100, 2)
        anomalies = np.array([[10, 10], [-10, -10]])
        self.X = np.vstack([normal, anomalies])

    def test_isolation_forest_shape(self):
        labels = isolation_forest(self.X, 0.02)
        self.assertEqual(len(labels), 102)

    def test_isolation_forest_finds_anomalies(self):
        labels = isolation_forest(self.X, 0.02)
        anomaly_count = np.sum(labels == -1)
        self.assertGreaterEqual(anomaly_count, 1)

    def test_lof_shape(self):
        labels = local_outlier_factor(self.X, 10)
        self.assertEqual(len(labels), 102)

    def test_lof_finds_anomalies(self):
        labels = local_outlier_factor(self.X, 10)
        anomaly_count = np.sum(labels == -1)
        self.assertGreaterEqual(anomaly_count, 1)

    def test_compare_returns_dict(self):
        result = compare_detectors(self.X)
        self.assertIsInstance(result, dict)
        self.assertIn('isolation_forest', result)
        self.assertIn('lof', result)

    def test_isolation_forest_returns_numpy(self):
        labels = isolation_forest(self.X, 0.02)
        self.assertIsInstance(labels, np.ndarray)

    def test_lof_returns_numpy(self):
        labels = local_outlier_factor(self.X, 10)
        self.assertIsInstance(labels, np.ndarray)

    def test_labels_are_1_or_minus1(self):
        labels = isolation_forest(self.X, 0.02)
        unique = set(labels)
        self.assertTrue(unique.issubset({1, -1}))

    def test_compare_shapes_match(self):
        result = compare_detectors(self.X)
        self.assertEqual(len(result['isolation_forest']), 102)
        self.assertEqual(len(result['lof']), 102)

    def test_different_contamination(self):
        labels_02 = isolation_forest(self.X, 0.02)
        labels_10 = isolation_forest(self.X, 0.1)
        anomalies_02 = np.sum(labels_02 == -1)
        anomalies_10 = np.sum(labels_10 == -1)
        self.assertLessEqual(anomalies_02, anomalies_10)
`,

	hint1: 'Use IsolationForest(contamination=c).fit_predict(X)',
	hint2: 'Use LocalOutlierFactor(n_neighbors=n).fit_predict(X)',

	whyItMatters: `Anomaly detection is crucial for:

- **Fraud detection**: Identify suspicious transactions
- **System monitoring**: Detect server anomalies
- **Quality control**: Find manufacturing defects
- **Data cleaning**: Identify corrupt data points

Core ML application in industry.`,

	translations: {
		ru: {
			title: 'Обнаружение аномалий',
			description: `# Обнаружение аномалий

Определение необычных паттернов с помощью Isolation Forest и Local Outlier Factor.

## Задача

Реализуйте три функции:
1. \`isolation_forest(X, contamination)\` - Обучить Isolation Forest
2. \`local_outlier_factor(X, n_neighbors)\` - Применить LOF
3. \`compare_detectors(X)\` - Сравнить IF и LOF предсказания

## Пример

\`\`\`python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

X = np.vstack([np.random.randn(100, 2), [[10, 10], [-10, -10]]])

if_labels = isolation_forest(X, contamination=0.02)
lof_labels = local_outlier_factor(X, n_neighbors=20)
comparison = compare_detectors(X)  # {'isolation_forest': [...], 'lof': [...]}
\`\`\``,
			hint1: 'Используйте IsolationForest(contamination=c).fit_predict(X)',
			hint2: 'Используйте LocalOutlierFactor(n_neighbors=n).fit_predict(X)',
			whyItMatters: `Обнаружение аномалий критично для:

- **Обнаружение мошенничества**: Выявление подозрительных транзакций
- **Мониторинг систем**: Обнаружение аномалий серверов
- **Контроль качества**: Поиск производственных дефектов`,
		},
		uz: {
			title: 'Anomaliya aniqlash',
			description: `# Anomaliya aniqlash

Isolation Forest va Local Outlier Factor yordamida g'ayrioddiy naqshlarni aniqlash.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`isolation_forest(X, contamination)\` - Isolation Forest ni o'rgatish
2. \`local_outlier_factor(X, n_neighbors)\` - LOF ni qo'llash
3. \`compare_detectors(X)\` - IF va LOF bashoratlarini taqqoslash

## Misol

\`\`\`python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

X = np.vstack([np.random.randn(100, 2), [[10, 10], [-10, -10]]])

if_labels = isolation_forest(X, contamination=0.02)
lof_labels = local_outlier_factor(X, n_neighbors=20)
comparison = compare_detectors(X)  # {'isolation_forest': [...], 'lof': [...]}
\`\`\``,
			hint1: "IsolationForest(contamination=c).fit_predict(X) dan foydalaning",
			hint2: "LocalOutlierFactor(n_neighbors=n).fit_predict(X) dan foydalaning",
			whyItMatters: `Anomaliya aniqlash quyidagilar uchun muhim:

- **Firibgarlikni aniqlash**: Shubhali tranzaksiyalarni aniqlash
- **Tizim monitoringi**: Server anomaliyalarini aniqlash
- **Sifat nazorati**: Ishlab chiqarish nuqsonlarini topish`,
		},
	},
};

export default task;
