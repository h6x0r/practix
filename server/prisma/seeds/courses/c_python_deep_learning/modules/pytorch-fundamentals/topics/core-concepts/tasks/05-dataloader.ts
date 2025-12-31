import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-pytorch-dataloader',
	title: 'Dataset and DataLoader',
	difficulty: 'medium',
	tags: ['pytorch', 'data', 'dataloader'],
	estimatedTime: '15m',
	isPremium: false,
	order: 5,
	description: `# Dataset and DataLoader

Learn to create custom datasets and use DataLoaders.

## Task

Implement three classes/functions:
1. \`SimpleDataset\` - Custom Dataset class for numpy arrays
2. \`create_dataloader(dataset, batch_size, shuffle)\` - Create DataLoader
3. \`get_batch(dataloader)\` - Get first batch from dataloader

## Example

\`\`\`python
from torch.utils.data import Dataset, DataLoader

X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, 1000)

dataset = SimpleDataset(X, y)
loader = create_dataloader(dataset, batch_size=32, shuffle=True)

for batch_x, batch_y in loader:
    # batch_x: (32, 10), batch_y: (32,)
    pass
\`\`\``,

	initialCode: `import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    """Custom dataset from numpy arrays."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Your code here
        pass

    def __len__(self) -> int:
        # Your code here
        pass

    def __getitem__(self, idx: int) -> tuple:
        # Your code here - return (X[idx], y[idx]) as tensors
        pass

def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    """Create DataLoader from dataset."""
    # Your code here
    pass

def get_batch(dataloader: DataLoader) -> tuple:
    """Get first batch from dataloader. Return (X_batch, y_batch)."""
    # Your code here
    pass
`,

	solutionCode: `import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    """Custom dataset from numpy arrays."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx]

def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    """Create DataLoader from dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_batch(dataloader: DataLoader) -> tuple:
    """Get first batch from dataloader. Return (X_batch, y_batch)."""
    return next(iter(dataloader))
`,

	testCode: `import torch
import numpy as np
import unittest

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.X = np.random.randn(100, 10).astype(np.float32)
        self.y = np.random.randint(0, 2, 100)

    def test_dataset_len(self):
        dataset = SimpleDataset(self.X, self.y)
        self.assertEqual(len(dataset), 100)

    def test_dataset_getitem(self):
        dataset = SimpleDataset(self.X, self.y)
        x, y = dataset[0]
        self.assertEqual(x.shape, torch.Size([10]))

    def test_dataloader_creation(self):
        dataset = SimpleDataset(self.X, self.y)
        loader = create_dataloader(dataset, 32, True)
        self.assertIsInstance(loader, DataLoader)

    def test_batch_shape(self):
        dataset = SimpleDataset(self.X, self.y)
        loader = create_dataloader(dataset, 32, False)
        x_batch, y_batch = get_batch(loader)
        self.assertEqual(x_batch.shape, torch.Size([32, 10]))
        self.assertEqual(y_batch.shape, torch.Size([32]))

    def test_iteration(self):
        dataset = SimpleDataset(self.X, self.y)
        loader = create_dataloader(dataset, 25, False)
        batches = list(loader)
        self.assertEqual(len(batches), 4)

    def test_dataset_returns_tensors(self):
        dataset = SimpleDataset(self.X, self.y)
        x, y = dataset[0]
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)

    def test_get_batch_returns_tuple(self):
        dataset = SimpleDataset(self.X, self.y)
        loader = create_dataloader(dataset, 32, False)
        result = get_batch(loader)
        self.assertEqual(len(result), 2)

    def test_dataloader_shuffle(self):
        dataset = SimpleDataset(self.X, self.y)
        loader = create_dataloader(dataset, 100, True)
        x1, _ = get_batch(loader)
        x2, _ = get_batch(loader)
        # Shuffled loaders should produce different results
        self.assertEqual(x1.shape, x2.shape)

    def test_dataset_all_items_accessible(self):
        dataset = SimpleDataset(self.X, self.y)
        for i in range(len(dataset)):
            x, y = dataset[i]
            self.assertEqual(x.shape, torch.Size([10]))

    def test_batch_y_dtype(self):
        dataset = SimpleDataset(self.X, self.y)
        loader = create_dataloader(dataset, 32, False)
        _, y_batch = get_batch(loader)
        self.assertEqual(y_batch.dtype, torch.long)
`,

	hint1: 'Store tensors in __init__, return self.X[idx], self.y[idx] in __getitem__',
	hint2: 'DataLoader(dataset, batch_size=32, shuffle=True)',

	whyItMatters: `DataLoaders are essential for training:

- **Batching**: Efficiently group samples for parallel processing
- **Shuffling**: Randomize order each epoch for better training
- **Multi-processing**: Load data in parallel with training
- **Memory efficiency**: Load data on-demand, not all at once

Every training loop uses DataLoaders.`,

	translations: {
		ru: {
			title: 'Dataset и DataLoader',
			description: `# Dataset и DataLoader

Научитесь создавать кастомные датасеты и использовать DataLoader.

## Задача

Реализуйте три класса/функции:
1. \`SimpleDataset\` - Кастомный Dataset для numpy массивов
2. \`create_dataloader(dataset, batch_size, shuffle)\` - Создать DataLoader
3. \`get_batch(dataloader)\` - Получить первый батч

## Пример

\`\`\`python
from torch.utils.data import Dataset, DataLoader

X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, 1000)

dataset = SimpleDataset(X, y)
loader = create_dataloader(dataset, batch_size=32, shuffle=True)

for batch_x, batch_y in loader:
    # batch_x: (32, 10), batch_y: (32,)
    pass
\`\`\``,
			hint1: 'Храните тензоры в __init__, возвращайте self.X[idx], self.y[idx] в __getitem__',
			hint2: 'DataLoader(dataset, batch_size=32, shuffle=True)',
			whyItMatters: `DataLoader необходим для обучения:

- **Батчинг**: Эффективная группировка для параллельной обработки
- **Перемешивание**: Случайный порядок каждую эпоху
- **Мультипроцессинг**: Параллельная загрузка данных`,
		},
		uz: {
			title: 'Dataset va DataLoader',
			description: `# Dataset va DataLoader

Maxsus datasetlar yaratish va DataLoaderlardan foydalanishni o'rganing.

## Topshiriq

Uchta sinf/funksiyani amalga oshiring:
1. \`SimpleDataset\` - numpy massivlari uchun maxsus Dataset sinfi
2. \`create_dataloader(dataset, batch_size, shuffle)\` - DataLoader yaratish
3. \`get_batch(dataloader)\` - Dataloader dan birinchi batchni olish

## Misol

\`\`\`python
from torch.utils.data import Dataset, DataLoader

X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, 1000)

dataset = SimpleDataset(X, y)
loader = create_dataloader(dataset, batch_size=32, shuffle=True)

for batch_x, batch_y in loader:
    # batch_x: (32, 10), batch_y: (32,)
    pass
\`\`\``,
			hint1: "__init__ da tensorlarni saqlang, __getitem__ da self.X[idx], self.y[idx] qaytaring",
			hint2: "DataLoader(dataset, batch_size=32, shuffle=True)",
			whyItMatters: `DataLoaderlar o'qitish uchun muhim:

- **Batching**: Parallel ishlash uchun namunalarni samarali guruhlash
- **Aralashtirish**: Yaxshiroq o'qitish uchun har bir davrda tartibni randomlashtirish
- **Ko'p protsessing**: O'qitish bilan parallel ma'lumotlarni yuklash`,
		},
	},
};

export default task;
