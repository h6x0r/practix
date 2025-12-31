import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'sklearn-pipeline',
	title: 'ML Pipelines',
	difficulty: 'medium',
	tags: ['sklearn', 'pipeline', 'workflow'],
	estimatedTime: '15m',
	isPremium: true,
	order: 6,
	description: `# ML Pipelines

Chain preprocessing and modeling steps into a single pipeline.

## Task

Implement three functions:
1. \`create_pipeline(steps)\` - Create pipeline from list of (name, transformer) tuples
2. \`create_preprocessing_pipeline()\` - Pipeline with scaling and model
3. \`pipeline_with_gridsearch(pipeline, param_grid, X, y)\` - Tune pipeline hyperparameters

## Example

\`\`\`python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

steps = [('scaler', StandardScaler()), ('svc', SVC())]
pipe = create_pipeline(steps)
pipe.fit(X_train, y_train)
\`\`\``,

	initialCode: `import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def create_pipeline(steps: list):
    """Create pipeline from list of (name, transformer) tuples."""
    # Your code here
    pass

def create_preprocessing_pipeline():
    """Create pipeline with StandardScaler and SVC."""
    # Your code here
    pass

def pipeline_with_gridsearch(pipeline, param_grid: dict, X: np.ndarray, y: np.ndarray):
    """Tune pipeline with GridSearchCV. Return fitted GridSearchCV."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def create_pipeline(steps: list):
    """Create pipeline from list of (name, transformer) tuples."""
    return Pipeline(steps)

def create_preprocessing_pipeline():
    """Create pipeline with StandardScaler and SVC."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC())
    ])

def pipeline_with_gridsearch(pipeline, param_grid: dict, X: np.ndarray, y: np.ndarray):
    """Tune pipeline with GridSearchCV. Return fitted GridSearchCV."""
    gs = GridSearchCV(pipeline, param_grid, cv=5)
    gs.fit(X, y)
    return gs
`,

	testCode: `import numpy as np
import unittest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class TestPipelines(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.array([0]*50 + [1]*50)

    def test_create_pipeline(self):
        steps = [('scaler', StandardScaler()), ('svc', SVC())]
        pipe = create_pipeline(steps)
        self.assertIsNotNone(pipe)

    def test_pipeline_can_fit(self):
        steps = [('scaler', StandardScaler()), ('svc', SVC())]
        pipe = create_pipeline(steps)
        pipe.fit(self.X, self.y)
        self.assertTrue(hasattr(pipe, 'predict'))

    def test_preprocessing_pipeline(self):
        pipe = create_preprocessing_pipeline()
        pipe.fit(self.X, self.y)
        preds = pipe.predict(self.X[:5])
        self.assertEqual(len(preds), 5)

    def test_gridsearch_pipeline(self):
        pipe = create_preprocessing_pipeline()
        param_grid = {'svc__C': [0.1, 1]}
        result = pipeline_with_gridsearch(pipe, param_grid, self.X, self.y)
        self.assertTrue(hasattr(result, 'best_params_'))

    def test_pipeline_is_pipeline_object(self):
        from sklearn.pipeline import Pipeline
        steps = [('scaler', StandardScaler()), ('svc', SVC())]
        pipe = create_pipeline(steps)
        self.assertIsInstance(pipe, Pipeline)

    def test_pipeline_predict_shape(self):
        steps = [('scaler', StandardScaler()), ('svc', SVC())]
        pipe = create_pipeline(steps)
        pipe.fit(self.X, self.y)
        preds = pipe.predict(self.X)
        self.assertEqual(len(preds), len(self.y))

    def test_preprocessing_pipeline_has_steps(self):
        pipe = create_preprocessing_pipeline()
        self.assertTrue(len(pipe.steps) >= 2)

    def test_gridsearch_has_best_score(self):
        pipe = create_preprocessing_pipeline()
        param_grid = {'svc__C': [0.1, 1]}
        result = pipeline_with_gridsearch(pipe, param_grid, self.X, self.y)
        self.assertTrue(hasattr(result, 'best_score_'))

    def test_pipeline_transform_and_predict(self):
        pipe = create_preprocessing_pipeline()
        pipe.fit(self.X, self.y)
        self.assertTrue(hasattr(pipe, 'predict'))

    def test_gridsearch_best_score_valid(self):
        pipe = create_preprocessing_pipeline()
        param_grid = {'svc__C': [0.1, 1]}
        result = pipeline_with_gridsearch(pipe, param_grid, self.X, self.y)
        self.assertTrue(0 <= result.best_score_ <= 1)
`,

	hint1: 'Pipeline takes a list of (name, transformer) tuples',
	hint2: 'For GridSearchCV params use step__param format (e.g., svc__C)',

	whyItMatters: `Pipelines are essential for:

- **Prevent data leakage**: Fit preprocessing only on training data
- **Clean code**: Single object for entire workflow
- **Reproducibility**: Serialize complete pipeline
- **Cross-validation**: Proper preprocessing in each fold

Standard practice in production ML systems.`,

	translations: {
		ru: {
			title: 'ML Пайплайны',
			description: `# ML Пайплайны

Объедините препроцессинг и модель в единый пайплайн.

## Задача

Реализуйте три функции:
1. \`create_pipeline(steps)\` - Создать пайплайн из списка кортежей
2. \`create_preprocessing_pipeline()\` - Пайплайн со скейлингом и моделью
3. \`pipeline_with_gridsearch(pipeline, param_grid, X, y)\` - Подбор гиперпараметров

## Пример

\`\`\`python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

steps = [('scaler', StandardScaler()), ('svc', SVC())]
pipe = create_pipeline(steps)
pipe.fit(X_train, y_train)
\`\`\``,
			hint1: 'Pipeline принимает список кортежей (имя, трансформер)',
			hint2: 'Для GridSearchCV используйте формат step__param (например, svc__C)',
			whyItMatters: `Пайплайны важны для:

- **Предотвращение утечки данных**: Препроцессинг только на обучающих данных
- **Чистый код**: Один объект для всего процесса
- **Воспроизводимость**: Сериализация полного пайплайна`,
		},
		uz: {
			title: 'ML Pipelinelar',
			description: `# ML Pipelinelar

Oldindan ishlov berish va modellashtirish bosqichlarini bitta pipelineda birlashtiring.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`create_pipeline(steps)\` - Kortejlar ro'yxatidan pipeline yaratish
2. \`create_preprocessing_pipeline()\` - Scaling va model bilan pipeline
3. \`pipeline_with_gridsearch(pipeline, param_grid, X, y)\` - Giperparametrlarni sozlash

## Misol

\`\`\`python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

steps = [('scaler', StandardScaler()), ('svc', SVC())]
pipe = create_pipeline(steps)
pipe.fit(X_train, y_train)
\`\`\``,
			hint1: "Pipeline (nom, transformer) kortejlari ro'yxatini qabul qiladi",
			hint2: "GridSearchCV uchun step__param formatidan foydalaning (masalan, svc__C)",
			whyItMatters: `Pipelinelar quyidagilar uchun muhim:

- **Ma'lumot oqishini oldini olish**: Faqat o'qitish ma'lumotlarida preprotsessing
- **Toza kod**: Butun jarayon uchun bitta obyekt
- **Takrorlanish**: To'liq pipelineni seriyalashtirish`,
		},
	},
};

export default task;
