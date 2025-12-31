import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'matplotlib-ml-plots',
	title: 'ML Evaluation Plots',
	difficulty: 'hard',
	tags: ['matplotlib', 'confusion-matrix', 'roc', 'ml'],
	estimatedTime: '18m',
	isPremium: true,
	order: 7,
	description: `# ML Evaluation Plots

Essential plots for evaluating machine learning model performance.

## Task

Implement three functions:
1. \`plot_confusion_matrix(y_true, y_pred, labels)\` - Confusion matrix heatmap
2. \`plot_roc_curve(y_true, y_scores)\` - ROC curve with AUC
3. \`plot_feature_importance(names, importances)\` - Feature importance bar chart

## Example

\`\`\`python
from sklearn.metrics import confusion_matrix, roc_curve, auc

y_true = [0, 0, 1, 1, 1, 0, 1]
y_pred = [0, 1, 1, 1, 0, 0, 1]

fig = plot_confusion_matrix(y_true, y_pred, ['Negative', 'Positive'])

# ROC curve
y_scores = [0.1, 0.4, 0.35, 0.8, 0.6, 0.2, 0.9]
fig = plot_roc_curve(y_true, y_scores)

# Feature importance
names = ['feature_a', 'feature_b', 'feature_c']
importances = [0.5, 0.3, 0.2]
fig = plot_feature_importance(names, importances)
\`\`\``,

	initialCode: `import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_confusion_matrix(y_true: list, y_pred: list, labels: list):
    """Confusion matrix heatmap. Return figure."""
    # Your code here
    pass

def plot_roc_curve(y_true: list, y_scores: list):
    """ROC curve with AUC. Return figure."""
    # Your code here
    pass

def plot_feature_importance(names: list, importances: list):
    """Feature importance bar chart (sorted). Return figure."""
    # Your code here
    pass
`,

	solutionCode: `import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_confusion_matrix(y_true: list, y_pred: list, labels: list):
    """Confusion matrix heatmap. Return figure."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center')
    fig.colorbar(im)
    return fig

def plot_roc_curve(y_true: list, y_scores: list):
    """ROC curve with AUC. Return figure."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'r--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    return fig

def plot_feature_importance(names: list, importances: list):
    """Feature importance bar chart (sorted). Return figure."""
    indices = np.argsort(importances)[::-1]
    sorted_names = [names[i] for i in indices]
    sorted_importances = [importances[i] for i in indices]
    fig, ax = plt.subplots()
    ax.barh(sorted_names, sorted_importances)
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    ax.invert_yaxis()
    return fig
`,

	testCode: `import matplotlib.pyplot as plt
import numpy as np
import unittest

class TestMLPlots(unittest.TestCase):
    def setUp(self):
        self.y_true = [0, 0, 1, 1, 1, 0, 1]
        self.y_pred = [0, 1, 1, 1, 0, 0, 1]
        self.y_scores = [0.1, 0.4, 0.35, 0.8, 0.6, 0.2, 0.9]
        self.names = ['a', 'b', 'c']
        self.importances = [0.5, 0.3, 0.2]
        plt.close('all')

    def test_plot_confusion_matrix_returns_figure(self):
        fig = plot_confusion_matrix(self.y_true, self.y_pred, ['Neg', 'Pos'])
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_confusion_matrix_has_colorbar(self):
        fig = plot_confusion_matrix(self.y_true, self.y_pred, ['Neg', 'Pos'])
        self.assertEqual(len(fig.axes), 2)

    def test_plot_confusion_matrix_has_annotations(self):
        fig = plot_confusion_matrix(self.y_true, self.y_pred, ['Neg', 'Pos'])
        ax = fig.axes[0]
        self.assertGreater(len(ax.texts), 0)

    def test_plot_roc_curve_returns_figure(self):
        fig = plot_roc_curve(self.y_true, self.y_scores)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_roc_curve_has_lines(self):
        fig = plot_roc_curve(self.y_true, self.y_scores)
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 2)

    def test_plot_roc_curve_has_legend(self):
        fig = plot_roc_curve(self.y_true, self.y_scores)
        ax = fig.axes[0]
        self.assertIsNotNone(ax.get_legend())

    def test_plot_feature_importance_returns_figure(self):
        fig = plot_feature_importance(self.names, self.importances)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_feature_importance_has_bars(self):
        fig = plot_feature_importance(self.names, self.importances)
        ax = fig.axes[0]
        self.assertEqual(len(ax.patches), 3)

    def test_plot_confusion_matrix_has_axes(self):
        fig = plot_confusion_matrix(self.y_true, self.y_pred, ['Neg', 'Pos'])
        self.assertGreater(len(fig.axes), 0)

    def test_plot_feature_importance_has_axes(self):
        fig = plot_feature_importance(self.names, self.importances)
        self.assertEqual(len(fig.axes), 1)

    def tearDown(self):
        plt.close('all')
`,

	hint1: 'Use sklearn.metrics.confusion_matrix and roc_curve for calculations',
	hint2: 'Sort feature importances with np.argsort, use barh for horizontal bars',

	whyItMatters: `ML evaluation plots are essential for:

- **Model selection**: Compare models using ROC/AUC
- **Error analysis**: Understand misclassification patterns
- **Interpretability**: Explain model decisions
- **Reporting**: Communicate results to stakeholders

Core skills for any ML practitioner.`,

	translations: {
		ru: {
			title: 'Графики оценки ML',
			description: `# Графики оценки ML

Необходимые графики для оценки производительности моделей машинного обучения.

## Задача

Реализуйте три функции:
1. \`plot_confusion_matrix(y_true, y_pred, labels)\` - Тепловая карта матрицы ошибок
2. \`plot_roc_curve(y_true, y_scores)\` - ROC кривая с AUC
3. \`plot_feature_importance(names, importances)\` - Столбчатая диаграмма важности признаков

## Пример

\`\`\`python
from sklearn.metrics import confusion_matrix, roc_curve, auc

y_true = [0, 0, 1, 1, 1, 0, 1]
y_pred = [0, 1, 1, 1, 0, 0, 1]

fig = plot_confusion_matrix(y_true, y_pred, ['Negative', 'Positive'])

# ROC curve
y_scores = [0.1, 0.4, 0.35, 0.8, 0.6, 0.2, 0.9]
fig = plot_roc_curve(y_true, y_scores)

# Feature importance
names = ['feature_a', 'feature_b', 'feature_c']
importances = [0.5, 0.3, 0.2]
fig = plot_feature_importance(names, importances)
\`\`\``,
			hint1: 'Используйте sklearn.metrics.confusion_matrix и roc_curve для расчётов',
			hint2: 'Сортируйте важности с np.argsort, используйте barh для горизонтальных столбцов',
			whyItMatters: `Графики оценки ML необходимы для:

- **Выбор модели**: Сравнение моделей используя ROC/AUC
- **Анализ ошибок**: Понимание паттернов неправильной классификации
- **Интерпретируемость**: Объяснение решений модели`,
		},
		uz: {
			title: "ML baholash grafiklari",
			description: `# ML baholash grafiklari

Mashinani o'rganish modellarining ishlashini baholash uchun zarur grafiklar.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`plot_confusion_matrix(y_true, y_pred, labels)\` - Chalkashlik matritsasi issiqlik xaritasi
2. \`plot_roc_curve(y_true, y_scores)\` - AUC bilan ROC egri chizig'i
3. \`plot_feature_importance(names, importances)\` - Xususiyat ahamiyati ustunli diagrammasi

## Misol

\`\`\`python
from sklearn.metrics import confusion_matrix, roc_curve, auc

y_true = [0, 0, 1, 1, 1, 0, 1]
y_pred = [0, 1, 1, 1, 0, 0, 1]

fig = plot_confusion_matrix(y_true, y_pred, ['Negative', 'Positive'])

# ROC curve
y_scores = [0.1, 0.4, 0.35, 0.8, 0.6, 0.2, 0.9]
fig = plot_roc_curve(y_true, y_scores)

# Feature importance
names = ['feature_a', 'feature_b', 'feature_c']
importances = [0.5, 0.3, 0.2]
fig = plot_feature_importance(names, importances)
\`\`\``,
			hint1: "Hisob-kitoblar uchun sklearn.metrics.confusion_matrix va roc_curve dan foydalaning",
			hint2: "Ahamiyatlarni np.argsort bilan saralang, gorizontal ustunlar uchun barh dan foydalaning",
			whyItMatters: `ML baholash grafiklari quyidagilar uchun zarur:

- **Model tanlash**: ROC/AUC yordamida modellarni taqqoslash
- **Xato tahlili**: Noto'g'ri klassifikatsiya naqshlarini tushunish
- **Interpretatsiya**: Model qarorlarini tushuntirish`,
		},
	},
};

export default task;
