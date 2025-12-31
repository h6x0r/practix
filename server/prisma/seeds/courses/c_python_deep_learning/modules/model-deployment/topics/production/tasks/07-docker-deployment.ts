import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-docker-deployment',
	title: 'Docker Deployment',
	difficulty: 'medium',
	tags: ['pytorch', 'docker', 'deployment'],
	estimatedTime: '15m',
	isPremium: false,
	order: 7,
	description: `# Docker Deployment

Containerize PyTorch models for consistent deployment.

## Task

Create helper functions for Docker deployment:
1. \`generate_dockerfile\` - Generate Dockerfile for PyTorch model
2. \`generate_requirements\` - Generate requirements.txt
3. \`create_health_check\` - Create health check endpoint

## Example

\`\`\`python
dockerfile = generate_dockerfile(
    base_image='python:3.9-slim',
    model_file='model.pt',
    port=8000
)

requirements = generate_requirements(['torch', 'fastapi', 'uvicorn'])
\`\`\``,

	initialCode: `from typing import List

def generate_dockerfile(base_image: str, model_file: str,
                        port: int = 8000) -> str:
    """Generate Dockerfile content for PyTorch model serving."""
    # Your code here
    pass

def generate_requirements(packages: List[str]) -> str:
    """Generate requirements.txt content."""
    # Your code here
    pass

def generate_docker_compose(service_name: str, port: int,
                            gpu: bool = False) -> str:
    """Generate docker-compose.yml content."""
    # Your code here
    pass
`,

	solutionCode: `from typing import List

def generate_dockerfile(base_image: str, model_file: str,
                        port: int = 8000) -> str:
    """Generate Dockerfile content for PyTorch model serving."""
    dockerfile = f'''FROM {base_image}

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and application
COPY {model_file} .
COPY app.py .

# Expose port
EXPOSE {port}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \\\\
    CMD curl -f http://localhost:{port}/health || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "{port}"]
'''
    return dockerfile

def generate_requirements(packages: List[str]) -> str:
    """Generate requirements.txt content."""
    default_packages = [
        'torch>=2.0.0',
        'fastapi>=0.100.0',
        'uvicorn>=0.22.0',
        'pydantic>=2.0.0',
    ]

    all_packages = set(default_packages)
    for pkg in packages:
        all_packages.add(pkg)

    return '\\n'.join(sorted(all_packages))

def generate_docker_compose(service_name: str, port: int,
                            gpu: bool = False) -> str:
    """Generate docker-compose.yml content."""
    compose = f'''version: '3.8'

services:
  {service_name}:
    build: .
    ports:
      - "{port}:{port}"
    environment:
      - MODEL_PATH=/app/model.pt
    volumes:
      - ./models:/app/models:ro
    restart: unless-stopped
'''
    if gpu:
        compose += '''    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
'''
    return compose
`,

	testCode: `import unittest

class TestDockerDeployment(unittest.TestCase):
    def test_generate_dockerfile(self):
        dockerfile = generate_dockerfile(
            base_image='python:3.9-slim',
            model_file='model.pt',
            port=8000
        )
        self.assertIn('FROM python:3.9-slim', dockerfile)
        self.assertIn('EXPOSE 8000', dockerfile)
        self.assertIn('model.pt', dockerfile)

    def test_generate_requirements(self):
        reqs = generate_requirements(['numpy', 'pandas'])
        self.assertIn('torch', reqs)
        self.assertIn('fastapi', reqs)

    def test_generate_docker_compose(self):
        compose = generate_docker_compose('ml-api', 8000)
        self.assertIn('ml-api', compose)
        self.assertIn('8000:8000', compose)

    def test_docker_compose_with_gpu(self):
        compose = generate_docker_compose('ml-api', 8000, gpu=True)
        self.assertIn('nvidia', compose)

    def test_dockerfile_has_workdir(self):
        dockerfile = generate_dockerfile('python:3.9', 'model.pt')
        self.assertIn('WORKDIR', dockerfile)

    def test_dockerfile_has_healthcheck(self):
        dockerfile = generate_dockerfile('python:3.9', 'model.pt')
        self.assertIn('HEALTHCHECK', dockerfile)

    def test_dockerfile_different_port(self):
        dockerfile = generate_dockerfile('python:3.9', 'model.pt', port=5000)
        self.assertIn('EXPOSE 5000', dockerfile)

    def test_requirements_returns_string(self):
        reqs = generate_requirements([])
        self.assertIsInstance(reqs, str)

    def test_docker_compose_has_version(self):
        compose = generate_docker_compose('api', 8000)
        self.assertIn('version', compose)

    def test_docker_compose_has_volumes(self):
        compose = generate_docker_compose('api', 8000)
        self.assertIn('volumes', compose)
`,

	hint1: 'Include HEALTHCHECK for container orchestration',
	hint2: 'Use --no-cache-dir with pip for smaller images',

	whyItMatters: `Docker is essential for ML deployment:

- **Reproducibility**: Same environment everywhere
- **Isolation**: Dependencies don't conflict
- **Scalability**: Easy horizontal scaling with Kubernetes
- **CI/CD**: Automated build and deploy pipelines

Containerization is the standard for production ML.`,

	translations: {
		ru: {
			title: 'Развертывание в Docker',
			description: `# Развертывание в Docker

Контейнеризируйте модели PyTorch для консистентного развертывания.

## Задача

Создайте вспомогательные функции для Docker развертывания:
1. \`generate_dockerfile\` - Генерация Dockerfile для модели PyTorch
2. \`generate_requirements\` - Генерация requirements.txt
3. \`create_health_check\` - Создание endpoint проверки здоровья

## Пример

\`\`\`python
dockerfile = generate_dockerfile(
    base_image='python:3.9-slim',
    model_file='model.pt',
    port=8000
)

requirements = generate_requirements(['torch', 'fastapi', 'uvicorn'])
\`\`\``,
			hint1: 'Включите HEALTHCHECK для оркестрации контейнеров',
			hint2: 'Используйте --no-cache-dir с pip для меньших образов',
			whyItMatters: `Docker необходим для ML развертывания:

- **Воспроизводимость**: Одинаковое окружение везде
- **Изоляция**: Зависимости не конфликтуют
- **Масштабируемость**: Легкое горизонтальное масштабирование с Kubernetes
- **CI/CD**: Автоматизированные пайплайны сборки и деплоя`,
		},
		uz: {
			title: 'Docker joylashtirish',
			description: `# Docker joylashtirish

Izchil joylashtirish uchun PyTorch modellarini konteynerlashtiring.

## Topshiriq

Docker joylashtirish uchun yordamchi funksiyalar yarating:
1. \`generate_dockerfile\` - PyTorch model uchun Dockerfile yaratish
2. \`generate_requirements\` - requirements.txt yaratish
3. \`create_health_check\` - Sog'liq tekshirish endpointini yaratish

## Misol

\`\`\`python
dockerfile = generate_dockerfile(
    base_image='python:3.9-slim',
    model_file='model.pt',
    port=8000
)

requirements = generate_requirements(['torch', 'fastapi', 'uvicorn'])
\`\`\``,
			hint1: "Konteyner orkestrasiyasi uchun HEALTHCHECK ni qo'shing",
			hint2: "Kichikroq image lar uchun pip bilan --no-cache-dir dan foydalaning",
			whyItMatters: `Docker ML joylashtirish uchun muhim:

- **Qayta ishlab chiqarish**: Hamma joyda bir xil muhit
- **Izolyatsiya**: Bog'liqliklar ziddiyat qilmaydi
- **Kengayuvchanlik**: Kubernetes bilan oson gorizontal masshtablash
- **CI/CD**: Avtomatlashtirilgan qurish va joylashtirish pipelinelar`,
		},
	},
};

export default task;
