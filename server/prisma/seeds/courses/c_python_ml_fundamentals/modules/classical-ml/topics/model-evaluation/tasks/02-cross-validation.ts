import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'sklearn-cross-validation',
	title: 'Cross-Validation',
	difficulty: 'easy',
	tags: ['sklearn', 'validation', 'cv'],
	estimatedTime: '12m',
	isPremium: false,
	order: 2,
	description: `# Cross-Validation

Evaluate model performance using k-fold cross-validation.

## Task

Implement three functions:
1. \`kfold_cv_score(model, X, y, k)\` - K-fold cross-validation scores
2. \`stratified_cv_score(model, X, y, k)\` - Stratified K-fold for classification
3. \`leave_one_out_score(model, X, y)\` - Leave-one-out cross-validation

## Example

\`\`\`python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, LeaveOneOut
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
X, y = np.random.randn(100, 5), np.array([0]*50 + [1]*50)

scores = kfold_cv_score(model, X, y, k=5)  # 5 scores
scores = stratified_cv_score(model, X, y, k=5)
\`\`\``,

	initialCode: `import numpy as np
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, LeaveOneOut

def kfold_cv_score(model, X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """K-fold cross-validation. Return array of k scores."""
    # Your code here
    pass

def stratified_cv_score(model, X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Stratified K-fold for classification. Return array of k scores."""
    # Your code here
    pass

def leave_one_out_score(model, X: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out CV. Return mean score."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, LeaveOneOut

def kfold_cv_score(model, X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """K-fold cross-validation. Return array of k scores."""
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    return cross_val_score(model, X, y, cv=cv)

def stratified_cv_score(model, X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Stratified K-fold for classification. Return array of k scores."""
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    return cross_val_score(model, X, y, cv=cv)

def leave_one_out_score(model, X: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out CV. Return mean score."""
    cv = LeaveOneOut()
    scores = cross_val_score(model, X, y, cv=cv)
    return scores.mean()
`,

	testCode: `import numpy as np
import unittest
from sklearn.linear_model import LogisticRegression

class TestCrossValidation(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.array([0]*50 + [1]*50)
        self.model = LogisticRegression(max_iter=1000)

    def test_kfold_returns_k_scores(self):
        scores = kfold_cv_score(self.model, self.X, self.y, k=5)
        self.assertEqual(len(scores), 5)

    def test_kfold_scores_reasonable(self):
        scores = kfold_cv_score(self.model, self.X, self.y, k=5)
        self.assertTrue(all(0 <= s <= 1 for s in scores))

    def test_stratified_returns_k_scores(self):
        scores = stratified_cv_score(self.model, self.X, self.y, k=5)
        self.assertEqual(len(scores), 5)

    def test_loo_returns_float(self):
        X_small = self.X[:20]
        y_small = self.y[:20]
        score = leave_one_out_score(self.model, X_small, y_small)
        self.assertIsInstance(score, float)

    def test_kfold_with_10_folds(self):
        scores = kfold_cv_score(self.model, self.X, self.y, k=10)
        self.assertEqual(len(scores), 10)

    def test_stratified_scores_reasonable(self):
        scores = stratified_cv_score(self.model, self.X, self.y, k=5)
        self.assertTrue(all(0 <= s <= 1 for s in scores))

    def test_loo_score_range(self):
        X_small = self.X[:20]
        y_small = self.y[:20]
        score = leave_one_out_score(self.model, X_small, y_small)
        self.assertTrue(0 <= score <= 1)

    def test_kfold_returns_numpy_array(self):
        scores = kfold_cv_score(self.model, self.X, self.y, k=5)
        self.assertIsInstance(scores, np.ndarray)

    def test_stratified_with_3_folds(self):
        scores = stratified_cv_score(self.model, self.X, self.y, k=3)
        self.assertEqual(len(scores), 3)

    def test_kfold_mean_reasonable(self):
        scores = kfold_cv_score(self.model, self.X, self.y, k=5)
        mean_score = scores.mean()
        self.assertTrue(0.3 <= mean_score <= 1.0)
`,

	hint1: 'Use KFold(n_splits=k) or StratifiedKFold(n_splits=k) as cv parameter',
	hint2: 'cross_val_score(model, X, y, cv=cv_object) returns array of scores',

	whyItMatters: `Cross-validation is crucial for:

- **Robust evaluation**: Multiple train/test splits
- **Variance estimation**: Understand model stability
- **Hyperparameter tuning**: Reliable performance metrics
- **Small datasets**: Make most of limited data

Gold standard for model evaluation.`,

	translations: {
		ru: {
			title: 'Кросс-валидация',
			description: `# Кросс-валидация

Оценивайте производительность модели с помощью k-fold кросс-валидации.

## Задача

Реализуйте три функции:
1. \`kfold_cv_score(model, X, y, k)\` - K-fold кросс-валидация
2. \`stratified_cv_score(model, X, y, k)\` - Стратифицированная K-fold
3. \`leave_one_out_score(model, X, y)\` - Leave-one-out кросс-валидация

## Пример

\`\`\`python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, LeaveOneOut
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
X, y = np.random.randn(100, 5), np.array([0]*50 + [1]*50)

scores = kfold_cv_score(model, X, y, k=5)  # 5 scores
scores = stratified_cv_score(model, X, y, k=5)
\`\`\``,
			hint1: 'Используйте KFold(n_splits=k) как параметр cv',
			hint2: 'cross_val_score(model, X, y, cv=cv_object) возвращает массив оценок',
			whyItMatters: `Кросс-валидация важна для:

- **Надежная оценка**: Множественные разбиения данных
- **Оценка дисперсии**: Понимание стабильности модели
- **Подбор гиперпараметров**: Надежные метрики`,
		},
		uz: {
			title: 'Kross-validatsiya',
			description: `# Kross-validatsiya

K-fold kross-validatsiya yordamida model samaradorligini baholang.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`kfold_cv_score(model, X, y, k)\` - K-fold kross-validatsiya
2. \`stratified_cv_score(model, X, y, k)\` - Stratifitsiyalangan K-fold
3. \`leave_one_out_score(model, X, y)\` - Leave-one-out kross-validatsiya

## Misol

\`\`\`python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, LeaveOneOut
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
X, y = np.random.randn(100, 5), np.array([0]*50 + [1]*50)

scores = kfold_cv_score(model, X, y, k=5)  # 5 scores
scores = stratified_cv_score(model, X, y, k=5)
\`\`\``,
			hint1: "cv parametri sifatida KFold(n_splits=k) dan foydalaning",
			hint2: "cross_val_score(model, X, y, cv=cv_object) ballar massivini qaytaradi",
			whyItMatters: `Kross-validatsiya quyidagilar uchun muhim:

- **Ishonchli baholash**: Ko'p train/test ajratishlari
- **Dispersiya baholash**: Model barqarorligini tushunish
- **Giperparametr sozlash**: Ishonchli ko'rsatkichlar`,
		},
	},
};

export default task;
