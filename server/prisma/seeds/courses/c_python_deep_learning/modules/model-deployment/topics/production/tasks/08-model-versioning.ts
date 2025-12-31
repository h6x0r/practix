import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-model-versioning',
	title: 'Model Versioning',
	difficulty: 'medium',
	tags: ['pytorch', 'mlops', 'versioning'],
	estimatedTime: '15m',
	isPremium: false,
	order: 8,
	description: `# Model Versioning

Implement model versioning for tracking and rollback.

## Task

Implement a \`ModelRegistry\` class that:
- Saves models with version metadata
- Loads specific versions
- Tracks model lineage and metrics

## Example

\`\`\`python
registry = ModelRegistry('./models')

# Save with metadata
registry.save(model, version='1.0.0', metrics={'accuracy': 0.95})

# Load specific version
loaded = registry.load(version='1.0.0')

# List all versions
versions = registry.list_versions()
\`\`\``,

	initialCode: `import torch
import torch.nn as nn
import json
import os
from typing import Dict, List, Optional
from datetime import datetime

class ModelRegistry:
    """Simple model registry for versioning."""

    def __init__(self, base_path: str):
        # Your code here
        pass

    def save(self, model: nn.Module, version: str,
             metrics: Dict = None, description: str = '') -> str:
        """Save model with version and metadata."""
        # Your code here
        pass

    def load(self, version: str) -> nn.Module:
        """Load model by version."""
        # Your code here
        pass

    def list_versions(self) -> List[Dict]:
        """List all saved versions with metadata."""
        # Your code here
        pass

    def get_latest(self) -> Optional[str]:
        """Get the latest version."""
        # Your code here
        pass
`,

	solutionCode: `import torch
import torch.nn as nn
import json
import os
from typing import Dict, List, Optional
from datetime import datetime

class ModelRegistry:
    """Simple model registry for versioning."""

    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        self.metadata_file = os.path.join(base_path, 'registry.json')

        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {'versions': {}}

    def _save_registry(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def save(self, model: nn.Module, version: str,
             metrics: Dict = None, description: str = '') -> str:
        """Save model with version and metadata."""
        model_path = os.path.join(self.base_path, f'model_v{version}.pt')
        torch.save(model.state_dict(), model_path)

        self.registry['versions'][version] = {
            'path': model_path,
            'metrics': metrics or {},
            'description': description,
            'timestamp': datetime.now().isoformat(),
        }
        self.registry['latest'] = version

        self._save_registry()
        return model_path

    def load(self, version: str) -> Dict:
        """Load model state dict by version."""
        if version not in self.registry['versions']:
            raise ValueError(f"Version {version} not found")

        model_path = self.registry['versions'][version]['path']
        return torch.load(model_path)

    def list_versions(self) -> List[Dict]:
        """List all saved versions with metadata."""
        versions = []
        for v, meta in self.registry['versions'].items():
            versions.append({
                'version': v,
                **meta
            })
        return sorted(versions, key=lambda x: x['timestamp'], reverse=True)

    def get_latest(self) -> Optional[str]:
        """Get the latest version."""
        return self.registry.get('latest')
`,

	testCode: `import torch
import torch.nn as nn
import tempfile
import shutil
import unittest

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)

class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(self.temp_dir)
        self.model = SimpleModel()

    def test_save_and_load(self):
        self.registry.save(self.model, '1.0.0', {'accuracy': 0.95})
        state_dict = self.registry.load('1.0.0')
        self.assertIn('fc.weight', state_dict)

    def test_list_versions(self):
        self.registry.save(self.model, '1.0.0')
        self.registry.save(self.model, '1.0.1')
        versions = self.registry.list_versions()
        self.assertEqual(len(versions), 2)

    def test_get_latest(self):
        self.registry.save(self.model, '1.0.0')
        self.registry.save(self.model, '2.0.0')
        self.assertEqual(self.registry.get_latest(), '2.0.0')

    def test_has_base_path(self):
        self.assertTrue(hasattr(self.registry, 'base_path'))

    def test_has_registry(self):
        self.assertTrue(hasattr(self.registry, 'registry'))

    def test_save_returns_path(self):
        path = self.registry.save(self.model, '1.0.0')
        self.assertIsInstance(path, str)

    def test_list_versions_empty(self):
        versions = self.registry.list_versions()
        self.assertEqual(len(versions), 0)

    def test_get_latest_empty(self):
        latest = self.registry.get_latest()
        self.assertIsNone(latest)

    def test_save_with_metrics(self):
        self.registry.save(self.model, '1.0.0', metrics={'acc': 0.9, 'loss': 0.1})
        versions = self.registry.list_versions()
        self.assertIn('metrics', versions[0])

    def test_metadata_file_exists(self):
        self.registry.save(self.model, '1.0.0')
        import os
        self.assertTrue(os.path.exists(self.registry.metadata_file))

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
`,

	hint1: 'Store metadata in a JSON file alongside models',
	hint2: 'Track timestamp for ordering versions',

	whyItMatters: `Model versioning is critical for MLOps:

- **Reproducibility**: Track exactly which model is in production
- **Rollback**: Quickly revert to previous versions
- **Comparison**: Compare metrics across versions
- **Audit trail**: Know when and why models changed

Proper versioning is essential for reliable ML systems.`,

	translations: {
		ru: {
			title: 'Версионирование моделей',
			description: `# Версионирование моделей

Реализуйте версионирование моделей для отслеживания и отката.

## Задача

Реализуйте класс \`ModelRegistry\`, который:
- Сохраняет модели с метаданными версий
- Загружает конкретные версии
- Отслеживает происхождение и метрики модели

## Пример

\`\`\`python
registry = ModelRegistry('./models')

# Save with metadata
registry.save(model, version='1.0.0', metrics={'accuracy': 0.95})

# Load specific version
loaded = registry.load(version='1.0.0')

# List all versions
versions = registry.list_versions()
\`\`\``,
			hint1: 'Храните метаданные в JSON файле рядом с моделями',
			hint2: 'Отслеживайте timestamp для упорядочивания версий',
			whyItMatters: `Версионирование моделей критично для MLOps:

- **Воспроизводимость**: Точное отслеживание модели в production
- **Откат**: Быстрый возврат к предыдущим версиям
- **Сравнение**: Сравнение метрик между версиями
- **Аудит**: Знание когда и почему модели менялись`,
		},
		uz: {
			title: 'Model versiyalash',
			description: `# Model versiyalash

Kuzatish va orqaga qaytarish uchun model versiyalashni amalga oshiring.

## Topshiriq

\`ModelRegistry\` sinfini amalga oshiring:
- Modellarni versiya metadata bilan saqlaydi
- Ma'lum versiyalarni yuklaydi
- Model kelib chiqishi va metrikalarini kuzatadi

## Misol

\`\`\`python
registry = ModelRegistry('./models')

# Save with metadata
registry.save(model, version='1.0.0', metrics={'accuracy': 0.95})

# Load specific version
loaded = registry.load(version='1.0.0')

# List all versions
versions = registry.list_versions()
\`\`\``,
			hint1: "Metadata ni modellar yonida JSON faylda saqlang",
			hint2: "Versiyalarni tartiblash uchun timestamp ni kuzating",
			whyItMatters: `Model versiyalash MLOps uchun muhim:

- **Qayta ishlab chiqarish**: Production da qaysi model borligini aniq kuzatish
- **Orqaga qaytarish**: Oldingi versiyalarga tez qaytish
- **Taqqoslash**: Versiyalar bo'ylab metrikalarni solishtirish
- **Audit izi**: Modellar qachon va nima uchun o'zgarganini bilish`,
		},
	},
};

export default task;
