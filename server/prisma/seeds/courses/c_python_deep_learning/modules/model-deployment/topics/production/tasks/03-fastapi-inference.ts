import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-fastapi-inference',
	title: 'FastAPI Inference Server',
	difficulty: 'medium',
	tags: ['pytorch', 'fastapi', 'api', 'deployment'],
	estimatedTime: '18m',
	isPremium: false,
	order: 3,
	description: `# FastAPI Inference Server

Build a REST API for model inference using FastAPI.

## Task

Implement a \`ModelServer\` class that:
- Loads a PyTorch model
- Provides a predict endpoint
- Handles input validation and preprocessing

## Example

\`\`\`python
server = ModelServer('model.pt', num_classes=10)

# FastAPI app
app = server.create_app()

# POST /predict
# Body: {"features": [1.0, 2.0, 3.0, ...]}
# Response: {"prediction": 0, "confidence": 0.95}
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
from typing import List, Dict
import numpy as np

# Simulating FastAPI for task
class Request:
    def __init__(self, features: List[float]):
        self.features = features

class ModelServer:
    """Inference server for PyTorch models."""

    def __init__(self, model_path: str = None, model: nn.Module = None):
        # Your code here
        pass

    def preprocess(self, features: List[float]) -> torch.Tensor:
        """Preprocess input features."""
        # Your code here
        pass

    def predict(self, features: List[float]) -> Dict:
        """Make prediction and return result dict."""
        # Your code here
        pass

    def batch_predict(self, batch: List[List[float]]) -> List[Dict]:
        """Make predictions for a batch of inputs."""
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import numpy as np

class Request:
    def __init__(self, features: List[float]):
        self.features = features

class ModelServer:
    """Inference server for PyTorch models."""

    def __init__(self, model_path: str = None, model: nn.Module = None):
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = torch.jit.load(model_path)
        else:
            raise ValueError("Either model_path or model must be provided")

        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def preprocess(self, features: List[float]) -> torch.Tensor:
        """Preprocess input features."""
        tensor = torch.tensor(features, dtype=torch.float32)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, features: List[float]) -> Dict:
        """Make prediction and return result dict."""
        with torch.no_grad():
            input_tensor = self.preprocess(features)
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=-1)
            confidence, prediction = probabilities.max(dim=-1)

            return {
                'prediction': prediction.item(),
                'confidence': confidence.item(),
                'probabilities': probabilities.squeeze().tolist()
            }

    def batch_predict(self, batch: List[List[float]]) -> List[Dict]:
        """Make predictions for a batch of inputs."""
        with torch.no_grad():
            tensors = [self.preprocess(f) for f in batch]
            batch_tensor = torch.cat(tensors, dim=0)
            outputs = self.model(batch_tensor)
            probabilities = F.softmax(outputs, dim=-1)
            confidences, predictions = probabilities.max(dim=-1)

            results = []
            for i in range(len(batch)):
                results.append({
                    'prediction': predictions[i].item(),
                    'confidence': confidences[i].item()
                })
            return results
`,

	testCode: `import torch
import torch.nn as nn
import unittest

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)

class TestModelServer(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.server = ModelServer(model=self.model)

    def test_predict(self):
        features = [1.0] * 10
        result = self.server.predict(features)
        self.assertIn('prediction', result)
        self.assertIn('confidence', result)
        self.assertTrue(0 <= result['confidence'] <= 1)

    def test_batch_predict(self):
        batch = [[1.0] * 10, [2.0] * 10]
        results = self.server.batch_predict(batch)
        self.assertEqual(len(results), 2)

    def test_preprocess(self):
        features = [1.0] * 10
        tensor = self.server.preprocess(features)
        self.assertEqual(tensor.shape, (1, 10))

    def test_predict_has_probabilities(self):
        features = [1.0] * 10
        result = self.server.predict(features)
        self.assertIn('probabilities', result)

    def test_prediction_is_int(self):
        features = [1.0] * 10
        result = self.server.predict(features)
        self.assertIsInstance(result['prediction'], int)

    def test_prediction_in_range(self):
        features = [1.0] * 10
        result = self.server.predict(features)
        self.assertTrue(0 <= result['prediction'] < 5)

    def test_batch_predict_all_have_prediction(self):
        batch = [[1.0] * 10, [2.0] * 10, [3.0] * 10]
        results = self.server.batch_predict(batch)
        for r in results:
            self.assertIn('prediction', r)

    def test_server_has_model(self):
        self.assertIsNotNone(self.server.model)

    def test_preprocess_returns_tensor(self):
        features = [1.0] * 10
        tensor = self.server.preprocess(features)
        self.assertIsInstance(tensor, torch.Tensor)

    def test_preprocess_different_features(self):
        features = list(range(10))
        tensor = self.server.preprocess([float(f) for f in features])
        self.assertEqual(tensor.shape, (1, 10))
`,

	hint1: 'Use torch.no_grad() for inference to save memory',
	hint2: 'Apply softmax to get probabilities from logits',

	whyItMatters: `REST APIs are the standard way to serve ML models:

- **Easy integration**: Any client can call HTTP endpoints
- **Scalable**: Deploy behind load balancer
- **FastAPI**: Automatic docs, validation, async support
- **Microservices**: Decouple model from application

This pattern is used in most production ML systems.`,

	translations: {
		ru: {
			title: 'Сервер инференса FastAPI',
			description: `# Сервер инференса FastAPI

Создайте REST API для инференса модели используя FastAPI.

## Задача

Реализуйте класс \`ModelServer\`, который:
- Загружает модель PyTorch
- Предоставляет endpoint для предсказаний
- Обрабатывает валидацию входных данных и препроцессинг

## Пример

\`\`\`python
server = ModelServer('model.pt', num_classes=10)

# FastAPI app
app = server.create_app()

# POST /predict
# Body: {"features": [1.0, 2.0, 3.0, ...]}
# Response: {"prediction": 0, "confidence": 0.95}
\`\`\``,
			hint1: 'Используйте torch.no_grad() для инференса для экономии памяти',
			hint2: 'Примените softmax для получения вероятностей из logits',
			whyItMatters: `REST API - стандартный способ обслуживания ML моделей:

- **Легкая интеграция**: Любой клиент может вызывать HTTP endpoints
- **Масштабируемость**: Развертывание за load balancer
- **FastAPI**: Автодокументация, валидация, async поддержка
- **Микросервисы**: Отделение модели от приложения`,
		},
		uz: {
			title: 'FastAPI inference serveri',
			description: `# FastAPI inference serveri

FastAPI yordamida model inference uchun REST API yarating.

## Topshiriq

\`ModelServer\` sinfini amalga oshiring:
- PyTorch modelini yuklaydi
- Bashorat qilish uchun endpoint beradi
- Kirish validatsiyasi va preprocessingni boshqaradi

## Misol

\`\`\`python
server = ModelServer('model.pt', num_classes=10)

# FastAPI app
app = server.create_app()

# POST /predict
# Body: {"features": [1.0, 2.0, 3.0, ...]}
# Response: {"prediction": 0, "confidence": 0.95}
\`\`\``,
			hint1: "Xotirani tejash uchun inference da torch.no_grad() dan foydalaning",
			hint2: "Logits dan ehtimolliklarni olish uchun softmax qo'llang",
			whyItMatters: `REST API lar ML modellarini xizmat qilishning standart usuli:

- **Oson integratsiya**: Har qanday mijoz HTTP endpointlarni chaqirishi mumkin
- **Kengayuvchanlik**: Load balancer ortida joylashtirish
- **FastAPI**: Avtohujjatlar, validatsiya, async qo'llab-quvvatlash
- **Mikroxizmatlar**: Modelni ilovadan ajratish`,
		},
	},
};

export default task;
