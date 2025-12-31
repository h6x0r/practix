import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'boosting-model-serialization',
	title: 'Model Serialization',
	difficulty: 'easy',
	tags: ['serialization', 'joblib', 'production'],
	estimatedTime: '10m',
	isPremium: false,
	order: 8,
	description: `# Model Serialization

Save and load trained boosting models for production deployment.

## Task

Implement three functions:
1. \`save_model_joblib(model, filepath)\` - Save model using joblib
2. \`load_model_joblib(filepath)\` - Load model from file
3. \`save_xgboost_native(model, filepath)\` - Save XGBoost in native format

## Example

\`\`\`python
import joblib

# Save trained model
save_model_joblib(model, 'model.joblib')

# Load for inference
loaded_model = load_model_joblib('model.joblib')
predictions = loaded_model.predict(X_new)
\`\`\``,

	initialCode: `import numpy as np
import joblib
import xgboost as xgb

def save_model_joblib(model, filepath: str) -> None:
    """Save model using joblib. No return value."""
    # Your code here
    pass

def load_model_joblib(filepath: str):
    """Load model from joblib file. Return model object."""
    # Your code here
    pass

def save_xgboost_native(model, filepath: str) -> None:
    """Save XGBoost model in native JSON format. No return value."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
import joblib
import xgboost as xgb

def save_model_joblib(model, filepath: str) -> None:
    """Save model using joblib. No return value."""
    joblib.dump(model, filepath)

def load_model_joblib(filepath: str):
    """Load model from joblib file. Return model object."""
    return joblib.load(filepath)

def save_xgboost_native(model, filepath: str) -> None:
    """Save XGBoost model in native JSON format. No return value."""
    model.save_model(filepath)
`,

	testCode: `import numpy as np
import unittest
import os
import tempfile
import xgboost as xgb

class TestModelSerialization(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(50, 5)
        self.y = np.array([0]*25 + [1]*25)
        self.model = xgb.XGBClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X, self.y)
        self.temp_dir = tempfile.mkdtemp()

    def test_save_joblib(self):
        filepath = os.path.join(self.temp_dir, 'model.joblib')
        save_model_joblib(self.model, filepath)
        self.assertTrue(os.path.exists(filepath))

    def test_load_joblib(self):
        filepath = os.path.join(self.temp_dir, 'model2.joblib')
        save_model_joblib(self.model, filepath)
        loaded = load_model_joblib(filepath)
        preds = loaded.predict(self.X[:5])
        self.assertEqual(len(preds), 5)

    def test_predictions_match(self):
        filepath = os.path.join(self.temp_dir, 'model3.joblib')
        original_preds = self.model.predict(self.X[:5])
        save_model_joblib(self.model, filepath)
        loaded = load_model_joblib(filepath)
        loaded_preds = loaded.predict(self.X[:5])
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_xgboost_native(self):
        filepath = os.path.join(self.temp_dir, 'model.json')
        save_xgboost_native(self.model, filepath)
        self.assertTrue(os.path.exists(filepath))

    def test_xgboost_native_loadable(self):
        filepath = os.path.join(self.temp_dir, 'model4.json')
        save_xgboost_native(self.model, filepath)
        loaded = xgb.XGBClassifier()
        loaded.load_model(filepath)
        preds = loaded.predict(self.X[:5])
        self.assertEqual(len(preds), 5)

    def test_loaded_model_has_predict_proba(self):
        filepath = os.path.join(self.temp_dir, 'model5.joblib')
        save_model_joblib(self.model, filepath)
        loaded = load_model_joblib(filepath)
        probs = loaded.predict_proba(self.X[:5])
        self.assertEqual(probs.shape[0], 5)

    def test_joblib_file_not_empty(self):
        filepath = os.path.join(self.temp_dir, 'model6.joblib')
        save_model_joblib(self.model, filepath)
        self.assertGreater(os.path.getsize(filepath), 0)

    def test_xgboost_native_file_not_empty(self):
        filepath = os.path.join(self.temp_dir, 'model7.json')
        save_xgboost_native(self.model, filepath)
        self.assertGreater(os.path.getsize(filepath), 0)

    def test_loaded_proba_matches(self):
        filepath = os.path.join(self.temp_dir, 'model8.joblib')
        original_probs = self.model.predict_proba(self.X[:5])
        save_model_joblib(self.model, filepath)
        loaded = load_model_joblib(filepath)
        loaded_probs = loaded.predict_proba(self.X[:5])
        np.testing.assert_array_almost_equal(original_probs, loaded_probs)

    def test_multiple_save_load_cycles(self):
        filepath = os.path.join(self.temp_dir, 'model9.joblib')
        save_model_joblib(self.model, filepath)
        loaded = load_model_joblib(filepath)
        filepath2 = os.path.join(self.temp_dir, 'model10.joblib')
        save_model_joblib(loaded, filepath2)
        loaded2 = load_model_joblib(filepath2)
        preds = loaded2.predict(self.X[:5])
        self.assertEqual(len(preds), 5)
`,

	hint1: 'joblib.dump(model, filepath) to save, joblib.load(filepath) to load',
	hint2: 'XGBoost native: model.save_model(filepath)',

	whyItMatters: `Model serialization is essential for:

- **Production deployment**: Load models in serving infrastructure
- **Version control**: Save different model versions
- **Reproducibility**: Ensure consistent predictions
- **Sharing**: Distribute trained models

Critical for ML operations (MLOps).`,

	translations: {
		ru: {
			title: 'Сериализация моделей',
			description: `# Сериализация моделей

Сохраняйте и загружайте обученные модели бустинга для продакшена.

## Задача

Реализуйте три функции:
1. \`save_model_joblib(model, filepath)\` - Сохранить модель через joblib
2. \`load_model_joblib(filepath)\` - Загрузить модель из файла
3. \`save_xgboost_native(model, filepath)\` - Сохранить XGBoost в нативном формате

## Пример

\`\`\`python
import joblib

# Save trained model
save_model_joblib(model, 'model.joblib')

# Load for inference
loaded_model = load_model_joblib('model.joblib')
predictions = loaded_model.predict(X_new)
\`\`\``,
			hint1: 'joblib.dump(model, filepath) для сохранения, joblib.load(filepath) для загрузки',
			hint2: 'XGBoost нативный: model.save_model(filepath)',
			whyItMatters: `Сериализация моделей важна для:

- **Деплой в продакшен**: Загрузка моделей в инфраструктуру
- **Версионирование**: Сохранение разных версий моделей
- **Воспроизводимость**: Гарантия консистентных предсказаний`,
		},
		uz: {
			title: 'Model seriyalizatsiyasi',
			description: `# Model seriyalizatsiyasi

Ishlab chiqarishga joylashtirish uchun o'qitilgan boosting modellarini saqlang va yuklang.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`save_model_joblib(model, filepath)\` - Modelni joblib bilan saqlash
2. \`load_model_joblib(filepath)\` - Fayldan modelni yuklash
3. \`save_xgboost_native(model, filepath)\` - XGBoostni tabiiy formatda saqlash

## Misol

\`\`\`python
import joblib

# Save trained model
save_model_joblib(model, 'model.joblib')

# Load for inference
loaded_model = load_model_joblib('model.joblib')
predictions = loaded_model.predict(X_new)
\`\`\``,
			hint1: "Saqlash uchun joblib.dump(model, filepath), yuklash uchun joblib.load(filepath)",
			hint2: "XGBoost tabiiy: model.save_model(filepath)",
			whyItMatters: `Model seriyalizatsiyasi quyidagilar uchun muhim:

- **Ishlab chiqarishga joylashtirish**: Xizmat infratuzilmasida modellarni yuklash
- **Versiya nazorati**: Turli model versiyalarini saqlash
- **Takrorlanish**: Izchil bashoratlarni ta'minlash`,
		},
	},
};

export default task;
