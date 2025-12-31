import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-custom-dataset',
	title: 'Custom Image Dataset',
	difficulty: 'medium',
	tags: ['pytorch', 'dataset', 'dataloader'],
	estimatedTime: '15m',
	isPremium: false,
	order: 10,
	description: `# Custom Image Dataset

Learn to create custom datasets for image classification.

## Task

Implement an \`ImageDataset\` class that:
- Takes a list of (image_path, label) tuples
- Applies optional transforms
- Returns (image_tensor, label) pairs

## Example

\`\`\`python
data = [
    ('path/to/cat1.jpg', 0),
    ('path/to/dog1.jpg', 1),
]

dataset = ImageDataset(data, transform=transforms.ToTensor())
image, label = dataset[0]
# image: Tensor, label: int

# Use with DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True)
\`\`\``,

	initialCode: `import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple, Optional, Callable

class ImageDataset(Dataset):
    """Custom dataset for image classification."""

    def __init__(self, data: List[Tuple[str, int]],
                 transform: Optional[Callable] = None):
        # TODO: Initialize with data and optional transform
        pass

    def __len__(self) -> int:
        # Your code here
        pass

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Your code here
        pass
`,

	solutionCode: `import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple, Optional, Callable

class ImageDataset(Dataset):
    """Custom dataset for image classification."""

    def __init__(self, data: List[Tuple[str, int]],
                 transform: Optional[Callable] = None):
        """
        Args:
            data: List of (image_path, label) tuples
            transform: Optional transform to apply to images
        """
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
`,

	testCode: `import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import tempfile
import os
import unittest

class TestImageDataset(unittest.TestCase):
    def setUp(self):
        # Create temp images
        self.temp_dir = tempfile.mkdtemp()
        self.paths = []
        for i in range(3):
            path = os.path.join(self.temp_dir, f'img_{i}.png')
            img = Image.new('RGB', (32, 32), color=(i*50, i*50, i*50))
            img.save(path)
            self.paths.append((path, i % 2))

    def test_len(self):
        dataset = ImageDataset(self.paths)
        self.assertEqual(len(dataset), 3)

    def test_getitem_no_transform(self):
        transform = transforms.ToTensor()
        dataset = ImageDataset(self.paths, transform=transform)
        img, label = dataset[0]
        self.assertEqual(img.shape, (3, 32, 32))
        self.assertEqual(label, 0)

    def test_with_dataloader(self):
        transform = transforms.ToTensor()
        dataset = ImageDataset(self.paths, transform=transform)
        loader = DataLoader(dataset, batch_size=2)
        batch = next(iter(loader))
        self.assertEqual(batch[0].shape[0], 2)  # batch size

    def test_is_dataset_subclass(self):
        dataset = ImageDataset(self.paths)
        self.assertIsInstance(dataset, Dataset)

    def test_stores_data(self):
        dataset = ImageDataset(self.paths)
        self.assertEqual(dataset.data, self.paths)

    def test_stores_transform(self):
        t = transforms.ToTensor()
        dataset = ImageDataset(self.paths, transform=t)
        self.assertEqual(dataset.transform, t)

    def test_label_type(self):
        transform = transforms.ToTensor()
        dataset = ImageDataset(self.paths, transform=transform)
        _, label = dataset[0]
        self.assertIsInstance(label, int)

    def test_image_channels(self):
        transform = transforms.ToTensor()
        dataset = ImageDataset(self.paths, transform=transform)
        img, _ = dataset[0]
        self.assertEqual(img.shape[0], 3)  # RGB

    def test_all_items_accessible(self):
        transform = transforms.ToTensor()
        dataset = ImageDataset(self.paths, transform=transform)
        for i in range(len(dataset)):
            img, label = dataset[i]
            self.assertEqual(img.shape, (3, 32, 32))

    def test_with_compose_transforms(self):
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        dataset = ImageDataset(self.paths, transform=transform)
        img, _ = dataset[0]
        self.assertEqual(img.shape, (3, 64, 64))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
`,

	hint1: 'Use PIL.Image.open(path).convert("RGB") to load images',
	hint2: 'Apply transform only if it is not None',

	whyItMatters: `Custom datasets are essential for real projects:

- **Flexibility**: Handle any data format or structure
- **Preprocessing**: Apply transforms on-the-fly
- **Memory efficiency**: Load images when needed, not all at once
- **Integration**: Works seamlessly with DataLoader

Almost every real project needs custom data loading logic.`,

	translations: {
		ru: {
			title: 'Пользовательский датасет изображений',
			description: `# Пользовательский датасет изображений

Научитесь создавать пользовательские датасеты для классификации изображений.

## Задача

Реализуйте класс \`ImageDataset\`, который:
- Принимает список кортежей (путь_к_изображению, метка)
- Применяет опциональные преобразования
- Возвращает пары (тензор_изображения, метка)

## Пример

\`\`\`python
data = [
    ('path/to/cat1.jpg', 0),
    ('path/to/dog1.jpg', 1),
]

dataset = ImageDataset(data, transform=transforms.ToTensor())
image, label = dataset[0]
# image: Tensor, label: int

# Use with DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True)
\`\`\``,
			hint1: 'Используйте PIL.Image.open(path).convert("RGB") для загрузки',
			hint2: 'Применяйте transform только если он не None',
			whyItMatters: `Пользовательские датасеты необходимы для реальных проектов:

- **Гибкость**: Работа с любым форматом данных
- **Предобработка**: Применение трансформаций на лету
- **Эффективность памяти**: Загрузка изображений по требованию
- **Интеграция**: Бесшовная работа с DataLoader`,
		},
		uz: {
			title: "Maxsus tasvir ma'lumotlar to'plami",
			description: `# Maxsus tasvir ma'lumotlar to'plami

Tasvirlarni tasniflash uchun maxsus ma'lumotlar to'plamlarini yaratishni o'rganing.

## Topshiriq

\`ImageDataset\` sinfini amalga oshiring:
- (tasvir_yo'li, belgi) kortejlar ro'yxatini qabul qiladi
- Ixtiyoriy transformatsiyalarni qo'llaydi
- (tasvir_tensori, belgi) juftliklarini qaytaradi

## Misol

\`\`\`python
data = [
    ('path/to/cat1.jpg', 0),
    ('path/to/dog1.jpg', 1),
]

dataset = ImageDataset(data, transform=transforms.ToTensor())
image, label = dataset[0]
# image: Tensor, label: int

# Use with DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True)
\`\`\``,
			hint1: 'Tasvirlarni yuklash uchun PIL.Image.open(path).convert("RGB") dan foydalaning',
			hint2: "Transform ni faqat None bo'lmasa qo'llang",
			whyItMatters: `Maxsus ma'lumotlar to'plamlari real loyihalar uchun muhim:

- **Moslashuvchanlik**: Har qanday ma'lumot formati bilan ishlash
- **Oldindan qayta ishlash**: Transformatsiyalarni parvozda qo'llash
- **Xotira samaradorligi**: Tasvirlarni kerak bo'lganda yuklash
- **Integratsiya**: DataLoader bilan uzluksiz ishlash`,
		},
	},
};

export default task;
