import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'sklearn-hyperparameter-tuning',
	title: 'Hyperparameter Tuning',
	difficulty: 'medium',
	tags: ['sklearn', 'tuning', 'gridsearch'],
	estimatedTime: '15m',
	isPremium: false,
	order: 5,
	description: `# Hyperparameter Tuning

Optimize model hyperparameters using GridSearchCV and RandomizedSearchCV.

## Task

Implement three functions:
1. \`grid_search(model, param_grid, X, y)\` - GridSearchCV optimization
2. \`random_search(model, param_dist, X, y, n_iter)\` - RandomizedSearchCV
3. \`get_best_params(search_result)\` - Extract best parameters

## Example

\`\`\`python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC

model = SVC()
param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}

result = grid_search(model, param_grid, X, y)
best = get_best_params(result)  # {'C': 1, 'kernel': 'rbf'}
\`\`\``,

	initialCode: `import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def grid_search(model, param_grid: dict, X: np.ndarray, y: np.ndarray):
    """Perform GridSearchCV. Return fitted GridSearchCV object."""
    # Your code here
    pass

def random_search(model, param_dist: dict, X: np.ndarray, y: np.ndarray, n_iter: int = 10):
    """Perform RandomizedSearchCV. Return fitted object."""
    # Your code here
    pass

def get_best_params(search_result) -> dict:
    """Extract best parameters from search result."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def grid_search(model, param_grid: dict, X: np.ndarray, y: np.ndarray):
    """Perform GridSearchCV. Return fitted GridSearchCV object."""
    gs = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    gs.fit(X, y)
    return gs

def random_search(model, param_dist: dict, X: np.ndarray, y: np.ndarray, n_iter: int = 10):
    """Perform RandomizedSearchCV. Return fitted object."""
    rs = RandomizedSearchCV(model, param_dist, n_iter=n_iter, cv=5, 
                            scoring='accuracy', random_state=42)
    rs.fit(X, y)
    return rs

def get_best_params(search_result) -> dict:
    """Extract best parameters from search result."""
    return search_result.best_params_
`,

	testCode: `import numpy as np
import unittest
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

class TestHyperparameterTuning(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.array([0]*50 + [1]*50)

    def test_grid_search_returns_object(self):
        model = SVC()
        param_grid = {'C': [0.1, 1], 'kernel': ['rbf']}
        result = grid_search(model, param_grid, self.X, self.y)
        self.assertIsNotNone(result)

    def test_grid_search_has_best_params(self):
        model = SVC()
        param_grid = {'C': [0.1, 1], 'kernel': ['rbf']}
        result = grid_search(model, param_grid, self.X, self.y)
        self.assertTrue(hasattr(result, 'best_params_'))

    def test_random_search_returns_object(self):
        model = RandomForestClassifier(n_estimators=10)
        param_dist = {'max_depth': [3, 5, 7]}
        result = random_search(model, param_dist, self.X, self.y, n_iter=3)
        self.assertIsNotNone(result)

    def test_get_best_params_returns_dict(self):
        model = SVC()
        param_grid = {'C': [0.1, 1], 'kernel': ['rbf']}
        result = grid_search(model, param_grid, self.X, self.y)
        params = get_best_params(result)
        self.assertIsInstance(params, dict)

    def test_grid_search_has_best_score(self):
        model = SVC()
        param_grid = {'C': [0.1, 1], 'kernel': ['rbf']}
        result = grid_search(model, param_grid, self.X, self.y)
        self.assertTrue(hasattr(result, 'best_score_'))

    def test_random_search_has_best_params(self):
        model = RandomForestClassifier(n_estimators=10)
        param_dist = {'max_depth': [3, 5, 7]}
        result = random_search(model, param_dist, self.X, self.y, n_iter=3)
        self.assertTrue(hasattr(result, 'best_params_'))

    def test_best_score_in_valid_range(self):
        model = SVC()
        param_grid = {'C': [0.1, 1], 'kernel': ['rbf']}
        result = grid_search(model, param_grid, self.X, self.y)
        self.assertTrue(0 <= result.best_score_ <= 1)

    def test_best_params_contains_searched_keys(self):
        model = SVC()
        param_grid = {'C': [0.1, 1], 'kernel': ['rbf']}
        result = grid_search(model, param_grid, self.X, self.y)
        params = get_best_params(result)
        self.assertIn('C', params)

    def test_random_search_with_distributions(self):
        model = RandomForestClassifier(n_estimators=10)
        param_dist = {'max_depth': [3, 5, 7, 10]}
        result = random_search(model, param_dist, self.X, self.y, n_iter=2)
        self.assertTrue(hasattr(result, 'best_score_'))

    def test_grid_search_can_predict(self):
        model = SVC()
        param_grid = {'C': [0.1, 1], 'kernel': ['rbf']}
        result = grid_search(model, param_grid, self.X, self.y)
        predictions = result.predict(self.X[:5])
        self.assertEqual(len(predictions), 5)
`,

	hint1: 'GridSearchCV(model, param_grid, cv=5).fit(X, y)',
	hint2: 'Access best_params_ attribute to get the optimal parameters',

	whyItMatters: `Hyperparameter tuning is critical for:

- **Performance optimization**: Find optimal model configuration
- **Avoiding overfitting**: Cross-validation ensures generalization
- **Automation**: Systematic search beats manual tuning
- **Reproducibility**: Document optimal settings

GridSearch for small spaces, RandomSearch for large spaces.`,

	translations: {
		ru: {
			title: 'Подбор гиперпараметров',
			description: `# Подбор гиперпараметров

Оптимизируйте гиперпараметры модели с помощью GridSearchCV и RandomizedSearchCV.

## Задача

Реализуйте три функции:
1. \`grid_search(model, param_grid, X, y)\` - Оптимизация GridSearchCV
2. \`random_search(model, param_dist, X, y, n_iter)\` - RandomizedSearchCV
3. \`get_best_params(search_result)\` - Извлечь лучшие параметры

## Пример

\`\`\`python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC

model = SVC()
param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}

result = grid_search(model, param_grid, X, y)
best = get_best_params(result)  # {'C': 1, 'kernel': 'rbf'}
\`\`\``,
			hint1: 'GridSearchCV(model, param_grid, cv=5).fit(X, y)',
			hint2: 'Используйте атрибут best_params_ для получения оптимальных параметров',
			whyItMatters: `Подбор гиперпараметров важен для:

- **Оптимизация производительности**: Найти оптимальную конфигурацию
- **Избежание переобучения**: Кросс-валидация обеспечивает обобщение
- **Автоматизация**: Систематический поиск лучше ручного`,
		},
		uz: {
			title: 'Giperparametrlarni sozlash',
			description: `# Giperparametrlarni sozlash

GridSearchCV va RandomizedSearchCV yordamida model giperparametrlarini optimallashtiring.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`grid_search(model, param_grid, X, y)\` - GridSearchCV optimallashtirish
2. \`random_search(model, param_dist, X, y, n_iter)\` - RandomizedSearchCV
3. \`get_best_params(search_result)\` - Eng yaxshi parametrlarni olish

## Misol

\`\`\`python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC

model = SVC()
param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}

result = grid_search(model, param_grid, X, y)
best = get_best_params(result)  # {'C': 1, 'kernel': 'rbf'}
\`\`\``,
			hint1: "GridSearchCV(model, param_grid, cv=5).fit(X, y)",
			hint2: "Optimal parametrlarni olish uchun best_params_ atributidan foydalaning",
			whyItMatters: `Giperparametr sozlash quyidagilar uchun muhim:

- **Samaradorlik optimallashtirish**: Optimal konfiguratsiyani topish
- **Haddan tashqari moslanishdan qochish**: Kross-validatsiya umumlashtirishni ta'minlaydi
- **Avtomatlashtirish**: Tizimli qidiruv qo'lda sozlashdan yaxshi`,
		},
	},
};

export default task;
