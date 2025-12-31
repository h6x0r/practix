import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-image-folder-dataset',
	title: 'ImageFolder Dataset',
	difficulty: 'easy',
	tags: ['pytorch', 'dataset', 'images'],
	estimatedTime: '12m',
	isPremium: false,
	order: 6,
	description: `# ImageFolder Dataset

Learn to load image datasets organized in folder structure.

## Task

Implement two functions:
1. \`create_datasets\` - Create train/val datasets from folder structure
2. \`create_dataloaders\` - Create DataLoaders with proper transforms

Expected folder structure:
\`\`\`
root/
  train/
    class1/
      img1.jpg
      img2.jpg
    class2/
      ...
  val/
    class1/
      ...
\`\`\`

## Example

\`\`\`python
train_ds, val_ds = create_datasets('data/')
train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=32)
\`\`\``,

	initialCode: `import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def create_datasets(root: str, image_size: int = 224):
    """Create train and validation datasets from folder structure."""
    # Your code here
    pass

def create_dataloaders(train_ds, val_ds, batch_size: int = 32,
                       num_workers: int = 4):
    """Create DataLoaders for training and validation."""
    # Your code here
    pass
`,

	solutionCode: `import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

def create_datasets(root: str, image_size: int = 224):
    """Create train and validation datasets from folder structure."""
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(
        os.path.join(root, 'train'),
        transform=train_transform
    )
    val_ds = datasets.ImageFolder(
        os.path.join(root, 'val'),
        transform=val_transform
    )

    return train_ds, val_ds

def create_dataloaders(train_ds, val_ds, batch_size: int = 32,
                       num_workers: int = 4):
    """Create DataLoaders for training and validation."""
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
`,

	testCode: `import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tempfile
import os
from PIL import Image
import unittest

class TestImageFolder(unittest.TestCase):
    def setUp(self):
        # Create temp directory structure
        self.temp_dir = tempfile.mkdtemp()
        for split in ['train', 'val']:
            for cls in ['cat', 'dog']:
                path = os.path.join(self.temp_dir, split, cls)
                os.makedirs(path)
                img = Image.new('RGB', (100, 100))
                img.save(os.path.join(path, 'img.jpg'))

    def test_create_datasets(self):
        train_ds, val_ds = create_datasets(self.temp_dir)
        self.assertIsInstance(train_ds, datasets.ImageFolder)
        self.assertEqual(len(train_ds.classes), 2)

    def test_create_dataloaders(self):
        train_ds, val_ds = create_datasets(self.temp_dir)
        train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=1, num_workers=0)
        batch = next(iter(train_loader))
        self.assertEqual(batch[0].shape[0], 1)

    def test_val_dataset_is_image_folder(self):
        train_ds, val_ds = create_datasets(self.temp_dir)
        self.assertIsInstance(val_ds, datasets.ImageFolder)

    def test_datasets_have_classes(self):
        train_ds, val_ds = create_datasets(self.temp_dir)
        self.assertEqual(val_ds.classes, ['cat', 'dog'])

    def test_train_loader_is_dataloader(self):
        train_ds, val_ds = create_datasets(self.temp_dir)
        train_loader, _ = create_dataloaders(train_ds, val_ds, batch_size=1, num_workers=0)
        self.assertIsInstance(train_loader, DataLoader)

    def test_val_loader_is_dataloader(self):
        train_ds, val_ds = create_datasets(self.temp_dir)
        _, val_loader = create_dataloaders(train_ds, val_ds, batch_size=1, num_workers=0)
        self.assertIsInstance(val_loader, DataLoader)

    def test_batch_has_images_and_labels(self):
        train_ds, val_ds = create_datasets(self.temp_dir)
        train_loader, _ = create_dataloaders(train_ds, val_ds, batch_size=1, num_workers=0)
        batch = next(iter(train_loader))
        self.assertEqual(len(batch), 2)

    def test_image_is_tensor(self):
        train_ds, val_ds = create_datasets(self.temp_dir)
        train_loader, _ = create_dataloaders(train_ds, val_ds, batch_size=1, num_workers=0)
        images, _ = next(iter(train_loader))
        self.assertIsInstance(images, torch.Tensor)

    def test_label_is_tensor(self):
        train_ds, val_ds = create_datasets(self.temp_dir)
        train_loader, _ = create_dataloaders(train_ds, val_ds, batch_size=1, num_workers=0)
        _, labels = next(iter(train_loader))
        self.assertIsInstance(labels, torch.Tensor)

    def test_custom_image_size(self):
        train_ds, _ = create_datasets(self.temp_dir, image_size=128)
        img, _ = train_ds[0]
        self.assertEqual(img.shape[1], 128)
        self.assertEqual(img.shape[2], 128)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
`,

	hint1: 'Use datasets.ImageFolder(path, transform=transform)',
	hint2: 'Set shuffle=True for training, shuffle=False for validation',

	whyItMatters: `ImageFolder is the standard way to load image datasets:

- **Automatic labeling**: Folder names become class labels
- **Scalable**: Works with any number of classes
- **Standard format**: Used by most image datasets
- **Easy integration**: Works with transforms and DataLoader

This is the first step in any image classification project.`,

	translations: {
		ru: {
			title: 'Датасет ImageFolder',
			description: `# Датасет ImageFolder

Научитесь загружать датасеты изображений организованные в папках.

## Задача

Реализуйте две функции:
1. \`create_datasets\` - Создание train/val датасетов из структуры папок
2. \`create_dataloaders\` - Создание DataLoader с нужными трансформациями

## Пример

\`\`\`python
train_ds, val_ds = create_datasets('data/')
train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=32)
\`\`\``,
			hint1: 'Используйте datasets.ImageFolder(path, transform=transform)',
			hint2: 'Установите shuffle=True для обучения, shuffle=False для валидации',
			whyItMatters: `ImageFolder - стандартный способ загрузки датасетов изображений:

- **Автоматическая разметка**: Имена папок становятся метками классов
- **Масштабируемость**: Работает с любым числом классов
- **Стандартный формат**: Используется большинством датасетов
- **Легкая интеграция**: Работает с transforms и DataLoader`,
		},
		uz: {
			title: "ImageFolder ma'lumotlar to'plami",
			description: `# ImageFolder ma'lumotlar to'plami

Papka tuzilmasida tashkil etilgan tasvir ma'lumotlar to'plamlarini yuklashni o'rganing.

## Topshiriq

Ikki funksiya amalga oshiring:
1. \`create_datasets\` - Papka tuzilmasidan train/val ma'lumotlar to'plamlarini yaratish
2. \`create_dataloaders\` - Tegishli transformatsiyalar bilan DataLoader yaratish

## Misol

\`\`\`python
train_ds, val_ds = create_datasets('data/')
train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=32)
\`\`\``,
			hint1: 'datasets.ImageFolder(path, transform=transform) dan foydalaning',
			hint2: "O'qitish uchun shuffle=True, validatsiya uchun shuffle=False qo'ying",
			whyItMatters: `ImageFolder tasvir ma'lumotlar to'plamlarini yuklashning standart usuli:

- **Avtomatik belgilash**: Papka nomlari sinf belgilari bo'ladi
- **Kengayuvchanlik**: Har qanday sinflar soni bilan ishlaydi
- **Standart format**: Ko'pchilik ma'lumotlar to'plamlari tomonidan ishlatiladi
- **Oson integratsiya**: transforms va DataLoader bilan ishlaydi`,
		},
	},
};

export default task;
