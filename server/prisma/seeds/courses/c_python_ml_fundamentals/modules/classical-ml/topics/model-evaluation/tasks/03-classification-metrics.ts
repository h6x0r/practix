import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'sklearn-classification-metrics',
	title: 'Classification Metrics',
	difficulty: 'medium',
	tags: ['sklearn', 'metrics', 'classification'],
	estimatedTime: '15m',
	isPremium: false,
	order: 3,
	description: `# Classification Metrics

Evaluate classification models with precision, recall, F1-score, and confusion matrix.

## Task

Implement four functions:
1. \`compute_accuracy(y_true, y_pred)\` - Calculate accuracy
2. \`compute_precision_recall_f1(y_true, y_pred)\` - Return dict with all three
3. \`get_confusion_matrix(y_true, y_pred)\` - Return confusion matrix
4. \`compute_roc_auc(y_true, y_proba)\` - Calculate ROC-AUC score

## Example

\`\`\`python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

acc = compute_accuracy(y_true, y_pred)  # 0.8
metrics = compute_precision_recall_f1(y_true, y_pred)
# {'precision': 1.0, 'recall': 0.67, 'f1': 0.8}
\`\`\``,

	initialCode: `import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score)

def compute_accuracy(y_true, y_pred) -> float:
    """Calculate accuracy score."""
    # Your code here
    pass

def compute_precision_recall_f1(y_true, y_pred) -> dict:
    """Return dict with precision, recall, f1 keys."""
    # Your code here
    pass

def get_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Return confusion matrix as 2D array."""
    # Your code here
    pass

def compute_roc_auc(y_true, y_proba) -> float:
    """Calculate ROC-AUC score from probability predictions."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score)

def compute_accuracy(y_true, y_pred) -> float:
    """Calculate accuracy score."""
    return accuracy_score(y_true, y_pred)

def compute_precision_recall_f1(y_true, y_pred) -> dict:
    """Return dict with precision, recall, f1 keys."""
    return {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

def get_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Return confusion matrix as 2D array."""
    return confusion_matrix(y_true, y_pred)

def compute_roc_auc(y_true, y_proba) -> float:
    """Calculate ROC-AUC score from probability predictions."""
    return roc_auc_score(y_true, y_proba)
`,

	testCode: `import numpy as np
import unittest

class TestClassificationMetrics(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        self.y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1])
        self.y_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.6, 0.7, 0.95])

    def test_accuracy_returns_float(self):
        acc = compute_accuracy(self.y_true, self.y_pred)
        self.assertIsInstance(acc, float)

    def test_accuracy_range(self):
        acc = compute_accuracy(self.y_true, self.y_pred)
        self.assertTrue(0 <= acc <= 1)

    def test_precision_recall_f1_keys(self):
        metrics = compute_precision_recall_f1(self.y_true, self.y_pred)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)

    def test_confusion_matrix_shape(self):
        cm = get_confusion_matrix(self.y_true, self.y_pred)
        self.assertEqual(cm.shape, (2, 2))

    def test_roc_auc_range(self):
        auc = compute_roc_auc(self.y_true, self.y_proba)
        self.assertTrue(0 <= auc <= 1)

    def test_precision_recall_f1_values_range(self):
        metrics = compute_precision_recall_f1(self.y_true, self.y_pred)
        self.assertTrue(0 <= metrics['precision'] <= 1)
        self.assertTrue(0 <= metrics['recall'] <= 1)
        self.assertTrue(0 <= metrics['f1'] <= 1)

    def test_confusion_matrix_values(self):
        cm = get_confusion_matrix(self.y_true, self.y_pred)
        self.assertEqual(cm.sum(), len(self.y_true))

    def test_perfect_accuracy(self):
        y_perfect = np.array([0, 1, 0, 1])
        acc = compute_accuracy(y_perfect, y_perfect)
        self.assertEqual(acc, 1.0)

    def test_roc_auc_perfect_prediction(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        auc = compute_roc_auc(y_true, y_proba)
        self.assertEqual(auc, 1.0)

    def test_confusion_matrix_returns_numpy(self):
        cm = get_confusion_matrix(self.y_true, self.y_pred)
        self.assertIsInstance(cm, np.ndarray)
`,

	hint1: 'Use accuracy_score, precision_score, recall_score, f1_score from sklearn.metrics',
	hint2: 'For ROC-AUC, use roc_auc_score(y_true, y_proba) with probability predictions',

	whyItMatters: `Classification metrics are essential for:

- **Imbalanced data**: Accuracy alone is misleading
- **Business context**: Precision vs recall trade-off
- **Model comparison**: AUC for ranking models
- **Error analysis**: Confusion matrix shows error types

Choose metrics based on business requirements.`,

	translations: {
		ru: {
			title: 'Метрики классификации',
			description: `# Метрики классификации

Оценивайте модели классификации с помощью precision, recall, F1-score и матрицы ошибок.

## Задача

Реализуйте четыре функции:
1. \`compute_accuracy(y_true, y_pred)\` - Вычислить accuracy
2. \`compute_precision_recall_f1(y_true, y_pred)\` - Вернуть словарь с тремя метриками
3. \`get_confusion_matrix(y_true, y_pred)\` - Вернуть матрицу ошибок
4. \`compute_roc_auc(y_true, y_proba)\` - Вычислить ROC-AUC

## Пример

\`\`\`python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

acc = compute_accuracy(y_true, y_pred)  # 0.8
metrics = compute_precision_recall_f1(y_true, y_pred)
# {'precision': 1.0, 'recall': 0.67, 'f1': 0.8}
\`\`\``,
			hint1: 'Используйте accuracy_score, precision_score, recall_score из sklearn.metrics',
			hint2: 'Для ROC-AUC используйте roc_auc_score(y_true, y_proba) с вероятностями',
			whyItMatters: `Метрики классификации важны для:

- **Несбалансированные данные**: Accuracy обманчива
- **Бизнес-контекст**: Баланс precision и recall
- **Сравнение моделей**: AUC для ранжирования`,
		},
		uz: {
			title: 'Klassifikatsiya metrikalari',
			description: `# Klassifikatsiya metrikalari

Klassifikatsiya modellarini precision, recall, F1-score va chalkashlik matritsasi bilan baholang.

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`compute_accuracy(y_true, y_pred)\` - Accuracy hisoblash
2. \`compute_precision_recall_f1(y_true, y_pred)\` - Uchta metrika bilan lug'at qaytarish
3. \`get_confusion_matrix(y_true, y_pred)\` - Chalkashlik matritsasini qaytarish
4. \`compute_roc_auc(y_true, y_proba)\` - ROC-AUC hisoblash

## Misol

\`\`\`python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

acc = compute_accuracy(y_true, y_pred)  # 0.8
metrics = compute_precision_recall_f1(y_true, y_pred)
# {'precision': 1.0, 'recall': 0.67, 'f1': 0.8}
\`\`\``,
			hint1: "sklearn.metrics dan accuracy_score, precision_score, recall_score dan foydalaning",
			hint2: "ROC-AUC uchun roc_auc_score(y_true, y_proba) ehtimolliklar bilan foydalaning",
			whyItMatters: `Klassifikatsiya metrikalari quyidagilar uchun muhim:

- **Nomutanosib ma'lumotlar**: Accuracy yolg'on tasavvur beradi
- **Biznes konteksti**: Precision va recall balansi
- **Model taqqoslash**: Modellarni tartiblash uchun AUC`,
		},
	},
};

export default task;
