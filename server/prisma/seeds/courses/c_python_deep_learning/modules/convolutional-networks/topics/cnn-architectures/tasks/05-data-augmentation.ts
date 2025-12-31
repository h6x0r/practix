import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-data-augmentation',
	title: 'Data Augmentation',
	difficulty: 'medium',
	tags: ['pytorch', 'cnn', 'augmentation', 'transforms'],
	estimatedTime: '12m',
	isPremium: false,
	order: 5,
	description: `# Data Augmentation

Learn to apply image transformations to improve model generalization.

## Task

Implement two functions:
1. \`get_train_transforms\` - Return a composition of training augmentations
2. \`get_test_transforms\` - Return transforms for test/validation (no augmentation)

Training transforms should include:
- Random horizontal flip
- Random rotation (up to 10 degrees)
- Color jitter
- Normalize to ImageNet stats

## Example

\`\`\`python
train_transform = get_train_transforms()
test_transform = get_test_transforms()

# Apply to PIL image
augmented = train_transform(pil_image)  # Tensor
normalized = test_transform(pil_image)  # Tensor
\`\`\``,

	initialCode: `import torch
from torchvision import transforms

def get_train_transforms(image_size: int = 224):
    """Get training transforms with augmentation."""
    # Your code here
    pass

def get_test_transforms(image_size: int = 224):
    """Get test transforms (no augmentation, just normalize)."""
    # Your code here
    pass
`,

	solutionCode: `import torch
from torchvision import transforms

def get_train_transforms(image_size: int = 224):
    """Get training transforms with augmentation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_test_transforms(image_size: int = 224):
    """Get test transforms (no augmentation, just normalize)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
`,

	testCode: `import torch
from torchvision import transforms
from PIL import Image
import unittest

class TestAugmentation(unittest.TestCase):
    def setUp(self):
        # Create a dummy RGB image
        self.img = Image.new('RGB', (256, 256), color='red')

    def test_train_transforms_output(self):
        t = get_train_transforms(224)
        out = t(self.img)
        self.assertEqual(out.shape, (3, 224, 224))

    def test_test_transforms_output(self):
        t = get_test_transforms(224)
        out = t(self.img)
        self.assertEqual(out.shape, (3, 224, 224))

    def test_output_is_normalized(self):
        t = get_test_transforms(224)
        out = t(self.img)
        # Normalized values should not be in [0, 1] range
        self.assertTrue(out.min() < 0 or out.max() > 1)

    def test_different_sizes(self):
        t = get_train_transforms(128)
        out = t(self.img)
        self.assertEqual(out.shape, (3, 128, 128))

    def test_train_returns_compose(self):
        t = get_train_transforms()
        self.assertIsInstance(t, transforms.Compose)

    def test_test_returns_compose(self):
        t = get_test_transforms()
        self.assertIsInstance(t, transforms.Compose)

    def test_output_is_tensor(self):
        t = get_train_transforms()
        out = t(self.img)
        self.assertIsInstance(out, torch.Tensor)

    def test_train_different_each_call(self):
        t = get_train_transforms()
        out1 = t(self.img)
        out2 = t(self.img)
        # Due to random augmentations, outputs should differ
        self.assertFalse(torch.allclose(out1, out2))

    def test_test_consistent(self):
        t = get_test_transforms()
        out1 = t(self.img)
        out2 = t(self.img)
        torch.testing.assert_close(out1, out2)

    def test_smaller_size(self):
        t = get_test_transforms(64)
        out = t(self.img)
        self.assertEqual(out.shape, (3, 64, 64))
`,

	hint1: 'Use transforms.Compose() to chain multiple transforms',
	hint2: 'ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]',

	whyItMatters: `Data augmentation is crucial for training robust models:

- **Prevents overfitting**: Creates virtual training examples
- **Improves generalization**: Model sees more variations
- **No extra data needed**: Augment existing samples
- **Standard practice**: Used in almost all image models

Different augmentations for train vs test ensures fair evaluation.`,

	translations: {
		ru: {
			title: 'Аугментация данных',
			description: `# Аугментация данных

Научитесь применять преобразования изображений для улучшения обобщения модели.

## Задача

Реализуйте две функции:
1. \`get_train_transforms\` - Возврат композиции аугментаций для обучения
2. \`get_test_transforms\` - Возврат преобразований для теста (без аугментации)

Преобразования обучения должны включать:
- Случайный горизонтальный переворот
- Случайный поворот (до 10 градусов)
- Изменение цвета
- Нормализация по статистике ImageNet

## Пример

\`\`\`python
train_transform = get_train_transforms()
test_transform = get_test_transforms()

# Apply to PIL image
augmented = train_transform(pil_image)  # Tensor
normalized = test_transform(pil_image)  # Tensor
\`\`\``,
			hint1: 'Используйте transforms.Compose() для объединения преобразований',
			hint2: 'Нормализация ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]',
			whyItMatters: `Аугментация данных критична для обучения робастных моделей:

- **Предотвращает переобучение**: Создает виртуальные примеры
- **Улучшает обобщение**: Модель видит больше вариаций
- **Не нужны дополнительные данные**: Аугментируем существующие
- **Стандартная практика**: Используется почти во всех моделях изображений`,
		},
		uz: {
			title: "Ma'lumotlarni kengaytirish",
			description: `# Ma'lumotlarni kengaytirish

Model umumlashtirishini yaxshilash uchun tasvir transformatsiyalarini qo'llashni o'rganing.

## Topshiriq

Ikki funksiya amalga oshiring:
1. \`get_train_transforms\` - O'qitish augmentatsiyalari kompozitsiyasini qaytarish
2. \`get_test_transforms\` - Test uchun transformatsiyalar (augmentatsiyasiz)

O'qitish transformatsiyalari quyidagilarni o'z ichiga olishi kerak:
- Tasodifiy gorizontal aylantirish
- Tasodifiy aylantirish (10 gradusgacha)
- Rang o'zgarishi
- ImageNet statistikasi bo'yicha normalizatsiya

## Misol

\`\`\`python
train_transform = get_train_transforms()
test_transform = get_test_transforms()

# Apply to PIL image
augmented = train_transform(pil_image)  # Tensor
normalized = test_transform(pil_image)  # Tensor
\`\`\``,
			hint1: "Bir nechta transformatsiyalarni birlashtirish uchun transforms.Compose() dan foydalaning",
			hint2: 'ImageNet normalizatsiyasi: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]',
			whyItMatters: `Ma'lumotlarni kengaytirish mustahkam modellarni o'qitish uchun muhim:

- **Ortiqcha moslanishni oldini oladi**: Virtual misollar yaratadi
- **Umumlashtirishni yaxshilaydi**: Model ko'proq variatsiyalarni ko'radi
- **Qo'shimcha ma'lumot kerak emas**: Mavjud namunalarni kengaytiradi
- **Standart amaliyot**: Deyarli barcha tasvir modellarida ishlatiladi`,
		},
	},
};

export default task;
