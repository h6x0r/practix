import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-complete-deployment-pipeline',
	title: 'Complete Deployment Pipeline',
	difficulty: 'hard',
	tags: ['pytorch', 'deployment', 'mlops', 'project'],
	estimatedTime: '25m',
	isPremium: true,
	order: 11,
	description: `# Complete Deployment Pipeline

Build a production-ready deployment pipeline integrating all concepts.

## Task

Implement a \`DeploymentPipeline\` class that:
- Exports model (TorchScript/ONNX)
- Applies optimizations (quantization)
- Creates inference server
- Includes monitoring and versioning

## Example

\`\`\`python
pipeline = DeploymentPipeline(model, config={
    'export_format': 'torchscript',
    'quantize': True,
    'server_port': 8000
})

# Deploy
pipeline.export()
pipeline.optimize()
pipeline.start_server()

# Monitor
metrics = pipeline.get_metrics()
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
from typing import Dict, Any
import time

class DeploymentPipeline:
    """Complete deployment pipeline for PyTorch models."""

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        # Your code here
        pass

    def export(self, output_path: str = 'model_exported.pt') -> str:
        """Export model to specified format."""
        # Your code here
        pass

    def optimize(self) -> nn.Module:
        """Apply optimizations (quantization, etc)."""
        # Your code here
        pass

    def predict(self, features: torch.Tensor) -> Dict:
        """Make prediction with timing."""
        # Your code here
        pass

    def get_metrics(self) -> Dict:
        """Get deployment metrics."""
        # Your code here
        pass

    def health_check(self) -> Dict:
        """Check if deployment is healthy."""
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
import time
from collections import deque

class DeploymentPipeline:
    """Complete deployment pipeline for PyTorch models."""

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.original_model = model
        self.model = model
        self.config = config
        self.model.eval()

        # Metrics tracking
        self.latencies = deque(maxlen=1000)
        self.predictions_count = 0
        self.errors_count = 0
        self.start_time = time.time()

        # Export info
        self.export_path = None
        self.is_optimized = False

    def export(self, output_path: str = 'model_exported.pt') -> str:
        """Export model to specified format."""
        export_format = self.config.get('export_format', 'torchscript')

        if export_format == 'torchscript':
            example_input = self.config.get('example_input', torch.randn(1, 10))
            traced = torch.jit.trace(self.model, example_input)
            traced.save(output_path)
        elif export_format == 'state_dict':
            torch.save(self.model.state_dict(), output_path)

        self.export_path = output_path
        return output_path

    def optimize(self) -> nn.Module:
        """Apply optimizations (quantization, etc)."""
        if self.config.get('quantize', False):
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear},
                dtype=torch.qint8
            )
            self.is_optimized = True
        return self.model

    def predict(self, features: torch.Tensor) -> Dict:
        """Make prediction with timing."""
        start = time.time()
        try:
            with torch.no_grad():
                output = self.model(features)
                probs = F.softmax(output, dim=-1)
                confidence, prediction = probs.max(dim=-1)

            latency = (time.time() - start) * 1000
            self.latencies.append(latency)
            self.predictions_count += 1

            return {
                'prediction': prediction.item() if prediction.dim() == 0 else prediction.tolist(),
                'confidence': confidence.item() if confidence.dim() == 0 else confidence.tolist(),
                'latency_ms': latency
            }
        except Exception as e:
            self.errors_count += 1
            raise e

    def get_metrics(self) -> Dict:
        """Get deployment metrics."""
        uptime = time.time() - self.start_time
        latencies = list(self.latencies)

        return {
            'uptime_seconds': uptime,
            'predictions_count': self.predictions_count,
            'errors_count': self.errors_count,
            'error_rate': self.errors_count / max(self.predictions_count, 1),
            'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0,
            'is_optimized': self.is_optimized,
            'export_path': self.export_path
        }

    def health_check(self) -> Dict:
        """Check if deployment is healthy."""
        try:
            test_input = self.config.get('example_input', torch.randn(1, 10))
            result = self.predict(test_input)
            return {
                'status': 'healthy',
                'model_responsive': True,
                'last_latency_ms': result['latency_ms']
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'model_responsive': False,
                'error': str(e)
            }
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

class TestDeploymentPipeline(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.config = {
            'export_format': 'torchscript',
            'quantize': False,
            'example_input': torch.randn(1, 10)
        }
        self.pipeline = DeploymentPipeline(self.model, self.config)

    def test_predict(self):
        features = torch.randn(1, 10)
        result = self.pipeline.predict(features)
        self.assertIn('prediction', result)
        self.assertIn('latency_ms', result)

    def test_get_metrics(self):
        self.pipeline.predict(torch.randn(1, 10))
        metrics = self.pipeline.get_metrics()
        self.assertEqual(metrics['predictions_count'], 1)

    def test_health_check(self):
        health = self.pipeline.health_check()
        self.assertEqual(health['status'], 'healthy')

    def test_optimize(self):
        self.pipeline.config['quantize'] = True
        self.pipeline.optimize()
        self.assertTrue(self.pipeline.is_optimized)

    def test_has_model(self):
        self.assertTrue(hasattr(self.pipeline, 'model'))

    def test_has_config(self):
        self.assertTrue(hasattr(self.pipeline, 'config'))

    def test_has_latencies(self):
        self.assertTrue(hasattr(self.pipeline, 'latencies'))

    def test_predict_has_confidence(self):
        features = torch.randn(1, 10)
        result = self.pipeline.predict(features)
        self.assertIn('confidence', result)

    def test_metrics_has_uptime(self):
        metrics = self.pipeline.get_metrics()
        self.assertIn('uptime_seconds', metrics)

    def test_health_check_returns_status(self):
        health = self.pipeline.health_check()
        self.assertIn('status', health)
`,

	hint1: 'Track latencies with deque for bounded memory',
	hint2: 'Include error handling and error rate tracking',

	whyItMatters: `A complete deployment pipeline brings together:

- **Model export**: Portable, optimized format
- **Optimization**: Quantization for efficiency
- **Serving**: Low-latency predictions
- **Monitoring**: Track health and performance

This is the foundation of production ML systems.`,

	translations: {
		ru: {
			title: 'Полный пайплайн развертывания',
			description: `# Полный пайплайн развертывания

Создайте production-ready пайплайн развертывания, интегрирующий все концепции.

## Задача

Реализуйте класс \`DeploymentPipeline\`, который:
- Экспортирует модель (TorchScript/ONNX)
- Применяет оптимизации (квантизация)
- Создает сервер инференса
- Включает мониторинг и версионирование

## Пример

\`\`\`python
pipeline = DeploymentPipeline(model, config={
    'export_format': 'torchscript',
    'quantize': True,
    'server_port': 8000
})

# Deploy
pipeline.export()
pipeline.optimize()
pipeline.start_server()

# Monitor
metrics = pipeline.get_metrics()
\`\`\``,
			hint1: 'Отслеживайте задержки с deque для ограниченной памяти',
			hint2: 'Включите обработку ошибок и отслеживание error rate',
			whyItMatters: `Полный пайплайн развертывания объединяет:

- **Экспорт модели**: Портативный, оптимизированный формат
- **Оптимизация**: Квантизация для эффективности
- **Обслуживание**: Предсказания с низкой задержкой
- **Мониторинг**: Отслеживание здоровья и производительности`,
		},
		uz: {
			title: "To'liq joylashtirish pipeline",
			description: `# To'liq joylashtirish pipeline

Barcha tushunchalarni integratsiya qiluvchi production-ready joylashtirish pipeline yarating.

## Topshiriq

\`DeploymentPipeline\` sinfini amalga oshiring:
- Modelni eksport qiladi (TorchScript/ONNX)
- Optimizatsiyalarni qo'llaydi (kvantizatsiya)
- Inference server yaratadi
- Monitoring va versiyalashni o'z ichiga oladi

## Misol

\`\`\`python
pipeline = DeploymentPipeline(model, config={
    'export_format': 'torchscript',
    'quantize': True,
    'server_port': 8000
})

# Deploy
pipeline.export()
pipeline.optimize()
pipeline.start_server()

# Monitor
metrics = pipeline.get_metrics()
\`\`\``,
			hint1: "Chegaralangan xotira uchun kechikishlarni deque bilan kuzating",
			hint2: "Xato boshqarish va xato stavkasini kuzatishni qo'shing",
			whyItMatters: `To'liq joylashtirish pipeline quyidagilarni birlashtiradi:

- **Model eksporti**: Portativ, optimallashtirilgan format
- **Optimallashtirish**: Samaradorlik uchun kvantizatsiya
- **Xizmat ko'rsatish**: Past kechikishli bashoratlar
- **Monitoring**: Sog'lik va ishlashni kuzatish`,
		},
	},
};

export default task;
