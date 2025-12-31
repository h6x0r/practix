import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'lightgbm-basics',
	title: 'LightGBM',
	difficulty: 'medium',
	tags: ['lightgbm', 'boosting', 'ml'],
	estimatedTime: '15m',
	isPremium: false,
	order: 3,
	description: `# LightGBM

Master LightGBM - faster training with leaf-wise tree growth.

## Task

Implement three functions:
1. \`train_lgb_classifier(X, y, params)\` - Train LGBMClassifier
2. \`train_lgb_regressor(X, y, params)\` - Train LGBMRegressor
3. \`train_with_categorical(X, y, cat_features)\` - Handle categorical features natively

## Example

\`\`\`python
import lightgbm as lgb

params = {'num_leaves': 31, 'learning_rate': 0.05, 'n_estimators': 100}
model = train_lgb_classifier(X_train, y_train, params)

# With categorical features
model = train_with_categorical(X, y, cat_features=[0, 2])
\`\`\``,

	initialCode: `import numpy as np
import lightgbm as lgb

def train_lgb_classifier(X: np.ndarray, y: np.ndarray, params: dict = None):
    """Train LGBMClassifier with given params. Return fitted model."""
    # Your code here
    pass

def train_lgb_regressor(X: np.ndarray, y: np.ndarray, params: dict = None):
    """Train LGBMRegressor with given params. Return fitted model."""
    # Your code here
    pass

def train_with_categorical(X: np.ndarray, y: np.ndarray, cat_features: list):
    """Train LGBMClassifier with categorical features. Return fitted model."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
import lightgbm as lgb

def train_lgb_classifier(X: np.ndarray, y: np.ndarray, params: dict = None):
    """Train LGBMClassifier with given params. Return fitted model."""
    if params is None:
        params = {}
    model = lgb.LGBMClassifier(random_state=42, verbose=-1, **params)
    model.fit(X, y)
    return model

def train_lgb_regressor(X: np.ndarray, y: np.ndarray, params: dict = None):
    """Train LGBMRegressor with given params. Return fitted model."""
    if params is None:
        params = {}
    model = lgb.LGBMRegressor(random_state=42, verbose=-1, **params)
    model.fit(X, y)
    return model

def train_with_categorical(X: np.ndarray, y: np.ndarray, cat_features: list):
    """Train LGBMClassifier with categorical features. Return fitted model."""
    model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X, y, categorical_feature=cat_features)
    return model
`,

	testCode: `import numpy as np
import unittest

class TestLightGBM(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y_class = np.array([0]*50 + [1]*50)
        self.y_reg = np.random.randn(100)

    def test_classifier_default(self):
        model = train_lgb_classifier(self.X, self.y_class)
        self.assertIsNotNone(model)

    def test_classifier_custom_params(self):
        params = {'num_leaves': 15, 'n_estimators': 10}
        model = train_lgb_classifier(self.X, self.y_class, params)
        preds = model.predict(self.X[:5])
        self.assertEqual(len(preds), 5)

    def test_regressor(self):
        model = train_lgb_regressor(self.X, self.y_reg, {'n_estimators': 10})
        preds = model.predict(self.X[:5])
        self.assertEqual(len(preds), 5)

    def test_categorical_features(self):
        X_cat = np.column_stack([self.X, np.random.randint(0, 3, 100)])
        model = train_with_categorical(X_cat, self.y_class, cat_features=[5])
        self.assertIsNotNone(model)

    def test_classifier_has_predict_proba(self):
        model = train_lgb_classifier(self.X, self.y_class)
        probs = model.predict_proba(self.X[:5])
        self.assertEqual(probs.shape[0], 5)

    def test_regressor_default(self):
        model = train_lgb_regressor(self.X, self.y_reg)
        self.assertIsNotNone(model)

    def test_classifier_has_feature_importances(self):
        model = train_lgb_classifier(self.X, self.y_class, {'n_estimators': 10})
        self.assertTrue(hasattr(model, 'feature_importances_'))

    def test_categorical_can_predict(self):
        X_cat = np.column_stack([self.X, np.random.randint(0, 3, 100)])
        model = train_with_categorical(X_cat, self.y_class, cat_features=[5])
        preds = model.predict(X_cat[:5])
        self.assertEqual(len(preds), 5)

    def test_regressor_predictions_are_floats(self):
        model = train_lgb_regressor(self.X, self.y_reg, {'n_estimators': 10})
        preds = model.predict(self.X[:5])
        self.assertTrue(all(isinstance(p, (float, np.floating)) for p in preds))
`,

	hint1: 'lgb.LGBMClassifier(**params).fit(X, y)',
	hint2: 'For categorical: fit(X, y, categorical_feature=[indices])',

	whyItMatters: `LightGBM excels at:

- **Speed**: Leaf-wise growth is faster than level-wise
- **Memory**: Histogram-based algorithm uses less memory
- **Large datasets**: Handles millions of rows efficiently
- **Categorical features**: Native support without encoding

Best for large-scale ML pipelines.`,

	translations: {
		ru: {
			title: 'LightGBM',
			description: `# LightGBM

Освойте LightGBM - более быстрое обучение с leaf-wise ростом деревьев.

## Задача

Реализуйте три функции:
1. \`train_lgb_classifier(X, y, params)\` - Обучить LGBMClassifier
2. \`train_lgb_regressor(X, y, params)\` - Обучить LGBMRegressor
3. \`train_with_categorical(X, y, cat_features)\` - Нативная работа с категориальными

## Пример

\`\`\`python
import lightgbm as lgb

params = {'num_leaves': 31, 'learning_rate': 0.05, 'n_estimators': 100}
model = train_lgb_classifier(X_train, y_train, params)

# With categorical features
model = train_with_categorical(X, y, cat_features=[0, 2])
\`\`\``,
			hint1: 'lgb.LGBMClassifier(**params).fit(X, y)',
			hint2: 'Для категориальных: fit(X, y, categorical_feature=[индексы])',
			whyItMatters: `LightGBM превосходит в:

- **Скорость**: Leaf-wise рост быстрее level-wise
- **Память**: Гистограммный алгоритм использует меньше памяти
- **Большие данные**: Эффективно обрабатывает миллионы строк`,
		},
		uz: {
			title: 'LightGBM',
			description: `# LightGBM

LightGBMni o'rganing - leaf-wise daraxt o'sishi bilan tezroq o'rganish.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`train_lgb_classifier(X, y, params)\` - LGBMClassifier ni o'rgatish
2. \`train_lgb_regressor(X, y, params)\` - LGBMRegressor ni o'rgatish
3. \`train_with_categorical(X, y, cat_features)\` - Kategorik xususiyatlarni tabiiy ishlov berish

## Misol

\`\`\`python
import lightgbm as lgb

params = {'num_leaves': 31, 'learning_rate': 0.05, 'n_estimators': 100}
model = train_lgb_classifier(X_train, y_train, params)

# With categorical features
model = train_with_categorical(X, y, cat_features=[0, 2])
\`\`\``,
			hint1: "lgb.LGBMClassifier(**params).fit(X, y)",
			hint2: "Kategorik uchun: fit(X, y, categorical_feature=[indekslar])",
			whyItMatters: `LightGBM quyidagilarda ustun:

- **Tezlik**: Leaf-wise o'sish level-wise dan tezroq
- **Xotira**: Gistogramma algoritmi kamroq xotira ishlatadi
- **Katta ma'lumotlar**: Millionlab qatorlarni samarali ishlaydi`,
		},
	},
};

export default task;
