import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'sklearn-gradient-boosting',
	title: 'Gradient Boosting Basics',
	difficulty: 'easy',
	tags: ['sklearn', 'boosting', 'ensemble'],
	estimatedTime: '12m',
	isPremium: false,
	order: 1,
	description: `# Gradient Boosting Basics

Learn the fundamentals of gradient boosting with sklearn's GradientBoostingClassifier.

## Task

Implement three functions:
1. \`train_gb_classifier(X, y, n_estimators)\` - Train GradientBoostingClassifier
2. \`train_gb_regressor(X, y, n_estimators)\` - Train GradientBoostingRegressor
3. \`get_feature_importance(model)\` - Extract feature importances

## Example

\`\`\`python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

model = train_gb_classifier(X_train, y_train, n_estimators=100)
predictions = model.predict(X_test)
importances = get_feature_importance(model)
\`\`\``,

	initialCode: `import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

def train_gb_classifier(X: np.ndarray, y: np.ndarray, n_estimators: int = 100):
    """Train GradientBoostingClassifier. Return fitted model."""
    # Your code here
    pass

def train_gb_regressor(X: np.ndarray, y: np.ndarray, n_estimators: int = 100):
    """Train GradientBoostingRegressor. Return fitted model."""
    # Your code here
    pass

def get_feature_importance(model) -> np.ndarray:
    """Extract feature importances from trained model."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

def train_gb_classifier(X: np.ndarray, y: np.ndarray, n_estimators: int = 100):
    """Train GradientBoostingClassifier. Return fitted model."""
    model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X, y)
    return model

def train_gb_regressor(X: np.ndarray, y: np.ndarray, n_estimators: int = 100):
    """Train GradientBoostingRegressor. Return fitted model."""
    model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X, y)
    return model

def get_feature_importance(model) -> np.ndarray:
    """Extract feature importances from trained model."""
    return model.feature_importances_
`,

	testCode: `import numpy as np
import unittest

class TestGradientBoosting(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y_class = np.array([0]*50 + [1]*50)
        self.y_reg = np.random.randn(100)

    def test_classifier_returns_model(self):
        model = train_gb_classifier(self.X, self.y_class, 10)
        self.assertIsNotNone(model)

    def test_classifier_can_predict(self):
        model = train_gb_classifier(self.X, self.y_class, 10)
        preds = model.predict(self.X[:5])
        self.assertEqual(len(preds), 5)

    def test_regressor_returns_model(self):
        model = train_gb_regressor(self.X, self.y_reg, 10)
        self.assertIsNotNone(model)

    def test_feature_importance_shape(self):
        model = train_gb_classifier(self.X, self.y_class, 10)
        imp = get_feature_importance(model)
        self.assertEqual(len(imp), 5)

    def test_feature_importance_sums_to_one(self):
        model = train_gb_classifier(self.X, self.y_class, 10)
        imp = get_feature_importance(model)
        self.assertAlmostEqual(sum(imp), 1.0, places=5)

    def test_regressor_can_predict(self):
        model = train_gb_regressor(self.X, self.y_reg, 10)
        preds = model.predict(self.X[:5])
        self.assertEqual(len(preds), 5)

    def test_feature_importance_returns_numpy(self):
        model = train_gb_classifier(self.X, self.y_class, 10)
        imp = get_feature_importance(model)
        self.assertIsInstance(imp, np.ndarray)

    def test_classifier_has_estimators(self):
        model = train_gb_classifier(self.X, self.y_class, 10)
        self.assertTrue(hasattr(model, 'n_estimators'))

    def test_regressor_has_estimators(self):
        model = train_gb_regressor(self.X, self.y_reg, 10)
        self.assertTrue(hasattr(model, 'n_estimators'))

    def test_importance_all_positive(self):
        model = train_gb_classifier(self.X, self.y_class, 10)
        imp = get_feature_importance(model)
        self.assertTrue(all(i >= 0 for i in imp))
`,

	hint1: 'GradientBoostingClassifier(n_estimators=n).fit(X, y)',
	hint2: 'Access feature_importances_ attribute after fitting',

	whyItMatters: `Gradient Boosting is powerful because:

- **Sequential learning**: Each tree corrects previous errors
- **Flexibility**: Works for classification and regression
- **Feature importance**: Built-in feature ranking
- **Industry standard**: Widely used in competitions and production

Foundation for XGBoost, LightGBM, and CatBoost.`,

	translations: {
		ru: {
			title: 'Основы градиентного бустинга',
			description: `# Основы градиентного бустинга

Изучите основы градиентного бустинга с GradientBoostingClassifier из sklearn.

## Задача

Реализуйте три функции:
1. \`train_gb_classifier(X, y, n_estimators)\` - Обучить классификатор
2. \`train_gb_regressor(X, y, n_estimators)\` - Обучить регрессор
3. \`get_feature_importance(model)\` - Извлечь важность признаков

## Пример

\`\`\`python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

model = train_gb_classifier(X_train, y_train, n_estimators=100)
predictions = model.predict(X_test)
importances = get_feature_importance(model)
\`\`\``,
			hint1: 'GradientBoostingClassifier(n_estimators=n).fit(X, y)',
			hint2: 'Используйте атрибут feature_importances_ после обучения',
			whyItMatters: `Градиентный бустинг мощный потому что:

- **Последовательное обучение**: Каждое дерево исправляет предыдущие ошибки
- **Гибкость**: Работает для классификации и регрессии
- **Важность признаков**: Встроенное ранжирование признаков`,
		},
		uz: {
			title: 'Gradient Boosting asoslari',
			description: `# Gradient Boosting asoslari

sklearn ning GradientBoostingClassifier bilan gradient boosting asoslarini o'rganing.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`train_gb_classifier(X, y, n_estimators)\` - Klassifikatorni o'rgatish
2. \`train_gb_regressor(X, y, n_estimators)\` - Regressorni o'rgatish
3. \`get_feature_importance(model)\` - Xususiyat muhimligini olish

## Misol

\`\`\`python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

model = train_gb_classifier(X_train, y_train, n_estimators=100)
predictions = model.predict(X_test)
importances = get_feature_importance(model)
\`\`\``,
			hint1: "GradientBoostingClassifier(n_estimators=n).fit(X, y)",
			hint2: "O'rgatishdan keyin feature_importances_ atributidan foydalaning",
			whyItMatters: `Gradient Boosting kuchli chunki:

- **Ketma-ket o'rganish**: Har bir daraxt oldingi xatolarni tuzatadi
- **Moslashuvchanlik**: Klassifikatsiya va regressiya uchun ishlaydi
- **Xususiyat muhimligi**: O'rnatilgan xususiyatlarni tartiblash`,
		},
	},
};

export default task;
