import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'ensemble-stacking',
	title: 'Ensemble Stacking',
	difficulty: 'hard',
	tags: ['ensemble', 'stacking', 'boosting'],
	estimatedTime: '18m',
	isPremium: true,
	order: 7,
	description: `# Ensemble Stacking

Combine multiple boosting models using stacking.

## Task

Implement three functions:
1. \`create_stacking_classifier(base_models, final_model)\` - Create stacking ensemble
2. \`train_stacking(stacking_clf, X, y)\` - Train stacking classifier
3. \`blend_predictions(models, X, weights)\` - Weighted average of predictions

## Example

\`\`\`python
from sklearn.ensemble import StackingClassifier

base_models = [
    ('xgb', XGBClassifier()),
    ('lgb', LGBMClassifier()),
    ('cat', CatBoostClassifier())
]
stacking = create_stacking_classifier(base_models, LogisticRegression())
stacking.fit(X_train, y_train)
\`\`\``,

	initialCode: `import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

def create_stacking_classifier(base_models: list, final_model):
    """Create StackingClassifier with base models and meta-learner."""
    # Your code here
    pass

def train_stacking(stacking_clf, X: np.ndarray, y: np.ndarray):
    """Train stacking classifier. Return fitted model."""
    # Your code here
    pass

def blend_predictions(models: list, X: np.ndarray, weights: list) -> np.ndarray:
    """Weighted average of model probability predictions. Return blended probabilities."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

def create_stacking_classifier(base_models: list, final_model):
    """Create StackingClassifier with base models and meta-learner."""
    return StackingClassifier(
        estimators=base_models,
        final_estimator=final_model,
        cv=5,
        stack_method='predict_proba'
    )

def train_stacking(stacking_clf, X: np.ndarray, y: np.ndarray):
    """Train stacking classifier. Return fitted model."""
    stacking_clf.fit(X, y)
    return stacking_clf

def blend_predictions(models: list, X: np.ndarray, weights: list) -> np.ndarray:
    """Weighted average of model probability predictions. Return blended probabilities."""
    weights = np.array(weights) / sum(weights)
    predictions = []
    for model in models:
        proba = model.predict_proba(X)[:, 1]
        predictions.append(proba)
    predictions = np.array(predictions)
    return np.average(predictions, axis=0, weights=weights)
`,

	testCode: `import numpy as np
import unittest
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

class TestEnsembleStacking(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.array([0]*50 + [1]*50)

    def test_create_stacking(self):
        base = [('xgb', xgb.XGBClassifier(n_estimators=10))]
        stacking = create_stacking_classifier(base, LogisticRegression())
        self.assertIsNotNone(stacking)

    def test_train_stacking(self):
        base = [('xgb', xgb.XGBClassifier(n_estimators=10))]
        stacking = create_stacking_classifier(base, LogisticRegression())
        trained = train_stacking(stacking, self.X, self.y)
        preds = trained.predict(self.X[:5])
        self.assertEqual(len(preds), 5)

    def test_blend_predictions(self):
        m1 = xgb.XGBClassifier(n_estimators=10).fit(self.X, self.y)
        m2 = lgb.LGBMClassifier(n_estimators=10, verbose=-1).fit(self.X, self.y)
        blended = blend_predictions([m1, m2], self.X[:10], [0.6, 0.4])
        self.assertEqual(len(blended), 10)

    def test_blend_range(self):
        m1 = xgb.XGBClassifier(n_estimators=10).fit(self.X, self.y)
        blended = blend_predictions([m1], self.X[:10], [1.0])
        self.assertTrue(all(0 <= p <= 1 for p in blended))

    def test_stacking_is_stacking_classifier(self):
        from sklearn.ensemble import StackingClassifier
        base = [('xgb', xgb.XGBClassifier(n_estimators=10))]
        stacking = create_stacking_classifier(base, LogisticRegression())
        self.assertIsInstance(stacking, StackingClassifier)

    def test_stacking_has_predict_proba(self):
        base = [('xgb', xgb.XGBClassifier(n_estimators=10))]
        stacking = create_stacking_classifier(base, LogisticRegression())
        trained = train_stacking(stacking, self.X, self.y)
        probs = trained.predict_proba(self.X[:5])
        self.assertEqual(probs.shape[0], 5)

    def test_blend_returns_numpy(self):
        m1 = xgb.XGBClassifier(n_estimators=10).fit(self.X, self.y)
        blended = blend_predictions([m1], self.X[:10], [1.0])
        self.assertIsInstance(blended, np.ndarray)

    def test_blend_multiple_models(self):
        m1 = xgb.XGBClassifier(n_estimators=10).fit(self.X, self.y)
        m2 = lgb.LGBMClassifier(n_estimators=10, verbose=-1).fit(self.X, self.y)
        m3 = xgb.XGBClassifier(n_estimators=5).fit(self.X, self.y)
        blended = blend_predictions([m1, m2, m3], self.X[:10], [0.5, 0.3, 0.2])
        self.assertEqual(len(blended), 10)

    def test_stacking_multiple_base_models(self):
        base = [
            ('xgb', xgb.XGBClassifier(n_estimators=10)),
            ('lgb', lgb.LGBMClassifier(n_estimators=10, verbose=-1))
        ]
        stacking = create_stacking_classifier(base, LogisticRegression())
        trained = train_stacking(stacking, self.X, self.y)
        preds = trained.predict(self.X[:5])
        self.assertEqual(len(preds), 5)

    def test_train_stacking_returns_fitted(self):
        base = [('xgb', xgb.XGBClassifier(n_estimators=10))]
        stacking = create_stacking_classifier(base, LogisticRegression())
        trained = train_stacking(stacking, self.X, self.y)
        self.assertTrue(hasattr(trained, 'final_estimator_'))
`,

	hint1: 'StackingClassifier(estimators=base_models, final_estimator=meta)',
	hint2: 'For blending: np.average(predictions, axis=0, weights=weights)',

	whyItMatters: `Stacking and blending:

- **Model diversity**: Combine strengths of different algorithms
- **Reduced variance**: Average out individual model errors
- **Competition winning**: Used in almost all Kaggle top solutions
- **Production ensembles**: Robust predictions in real applications

Advanced technique for maximum performance.`,

	translations: {
		ru: {
			title: 'Стекинг ансамблей',
			description: `# Стекинг ансамблей

Комбинируйте несколько моделей бустинга с помощью стекинга.

## Задача

Реализуйте три функции:
1. \`create_stacking_classifier(base_models, final_model)\` - Создать стекинг
2. \`train_stacking(stacking_clf, X, y)\` - Обучить стекинг
3. \`blend_predictions(models, X, weights)\` - Взвешенное среднее предсказаний

## Пример

\`\`\`python
from sklearn.ensemble import StackingClassifier

base_models = [
    ('xgb', XGBClassifier()),
    ('lgb', LGBMClassifier()),
    ('cat', CatBoostClassifier())
]
stacking = create_stacking_classifier(base_models, LogisticRegression())
stacking.fit(X_train, y_train)
\`\`\``,
			hint1: 'StackingClassifier(estimators=base_models, final_estimator=meta)',
			hint2: 'Для блендинга: np.average(predictions, axis=0, weights=weights)',
			whyItMatters: `Стекинг и блендинг:

- **Разнообразие моделей**: Объединение сильных сторон алгоритмов
- **Уменьшение дисперсии**: Усреднение ошибок отдельных моделей
- **Победа в соревнованиях**: Используется почти во всех топ решениях Kaggle`,
		},
		uz: {
			title: 'Ansambl stacking',
			description: `# Ansambl stacking

Stacking yordamida bir nechta boosting modellarini birlashtiring.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`create_stacking_classifier(base_models, final_model)\` - Stacking ansamblini yaratish
2. \`train_stacking(stacking_clf, X, y)\` - Stacking klassifikatorini o'rgatish
3. \`blend_predictions(models, X, weights)\` - Bashoratlarning og'irlikli o'rtachasi

## Misol

\`\`\`python
from sklearn.ensemble import StackingClassifier

base_models = [
    ('xgb', XGBClassifier()),
    ('lgb', LGBMClassifier()),
    ('cat', CatBoostClassifier())
]
stacking = create_stacking_classifier(base_models, LogisticRegression())
stacking.fit(X_train, y_train)
\`\`\``,
			hint1: "StackingClassifier(estimators=base_models, final_estimator=meta)",
			hint2: "Blending uchun: np.average(predictions, axis=0, weights=weights)",
			whyItMatters: `Stacking va blending:

- **Model xilma-xilligi**: Turli algoritmlarning kuchli tomonlarini birlashtirish
- **Kamaytirilgan dispersiya**: Individual model xatolarini o'rtacha qilish
- **Musobaqa g'oliblari**: Deyarli barcha Kaggle top yechimlarida ishlatiladi`,
		},
	},
};

export default task;
