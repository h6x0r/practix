import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pandas-create-dataframe',
	title: 'Creating DataFrames',
	difficulty: 'easy',
	tags: ['pandas', 'dataframe', 'basics'],
	estimatedTime: '10m',
	isPremium: false,
	order: 1,
	description: `# Creating DataFrames

A DataFrame is a 2D labeled data structure with columns of potentially different types - like a spreadsheet or SQL table.

## Task

Implement three functions:
1. \`from_dict(data)\` - Create DataFrame from dictionary
2. \`from_lists(data, columns)\` - Create DataFrame from list of lists with column names
3. \`from_numpy(arr, columns)\` - Create DataFrame from NumPy array with column names

## Example

\`\`\`python
# From dictionary
data = {'name': ['Alice', 'Bob'], 'age': [25, 30]}
from_dict(data)
#     name  age
# 0  Alice   25
# 1    Bob   30

# From lists
data = [['Alice', 25], ['Bob', 30]]
from_lists(data, ['name', 'age'])

# From numpy
arr = np.array([[1, 2], [3, 4]])
from_numpy(arr, ['a', 'b'])
\`\`\``,

	initialCode: `import pandas as pd
import numpy as np

def from_dict(data: dict) -> pd.DataFrame:
    """Create DataFrame from dictionary."""
    # Your code here
    pass

def from_lists(data: list, columns: list) -> pd.DataFrame:
    """Create DataFrame from list of lists with column names."""
    # Your code here
    pass

def from_numpy(arr: np.ndarray, columns: list) -> pd.DataFrame:
    """Create DataFrame from NumPy array with column names."""
    # Your code here
    pass
`,

	solutionCode: `import pandas as pd
import numpy as np

def from_dict(data: dict) -> pd.DataFrame:
    """Create DataFrame from dictionary."""
    return pd.DataFrame(data)

def from_lists(data: list, columns: list) -> pd.DataFrame:
    """Create DataFrame from list of lists with column names."""
    return pd.DataFrame(data, columns=columns)

def from_numpy(arr: np.ndarray, columns: list) -> pd.DataFrame:
    """Create DataFrame from NumPy array with column names."""
    return pd.DataFrame(arr, columns=columns)
`,

	testCode: `import pandas as pd
import numpy as np
import unittest

class TestCreateDataFrame(unittest.TestCase):
    def test_from_dict_basic(self):
        data = {'name': ['Alice', 'Bob'], 'age': [25, 30]}
        df = from_dict(data)
        self.assertEqual(list(df.columns), ['name', 'age'])
        self.assertEqual(len(df), 2)

    def test_from_dict_values(self):
        data = {'x': [1, 2, 3]}
        df = from_dict(data)
        self.assertEqual(df['x'].tolist(), [1, 2, 3])

    def test_from_dict_types(self):
        data = {'int_col': [1, 2], 'str_col': ['a', 'b']}
        df = from_dict(data)
        self.assertIsInstance(df, pd.DataFrame)

    def test_from_lists_basic(self):
        data = [['Alice', 25], ['Bob', 30]]
        df = from_lists(data, ['name', 'age'])
        self.assertEqual(list(df.columns), ['name', 'age'])
        self.assertEqual(len(df), 2)

    def test_from_lists_values(self):
        data = [[1, 2], [3, 4]]
        df = from_lists(data, ['a', 'b'])
        self.assertEqual(df.iloc[0, 0], 1)
        self.assertEqual(df.iloc[1, 1], 4)

    def test_from_lists_shape(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        df = from_lists(data, ['x', 'y', 'z'])
        self.assertEqual(df.shape, (3, 3))

    def test_from_numpy_basic(self):
        arr = np.array([[1, 2], [3, 4]])
        df = from_numpy(arr, ['a', 'b'])
        self.assertEqual(list(df.columns), ['a', 'b'])
        self.assertEqual(len(df), 2)

    def test_from_numpy_values(self):
        arr = np.array([[1.5, 2.5], [3.5, 4.5]])
        df = from_numpy(arr, ['x', 'y'])
        self.assertAlmostEqual(df['x'].iloc[0], 1.5)

    def test_from_numpy_shape(self):
        arr = np.zeros((5, 3))
        df = from_numpy(arr, ['a', 'b', 'c'])
        self.assertEqual(df.shape, (5, 3))

    def test_from_numpy_dtypes(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.int64)
        df = from_numpy(arr, ['x', 'y'])
        self.assertTrue(np.issubdtype(df['x'].dtype, np.integer))
`,

	hint1: 'Use pd.DataFrame(data) constructor',
	hint2: 'Pass columns parameter for custom column names',

	whyItMatters: `DataFrames are the foundation of data science:

- **Data loading**: Convert JSON, CSV, SQL results to DataFrames
- **Feature engineering**: Create new feature columns
- **Model input**: Many ML libraries accept DataFrames directly
- **Data exploration**: View, filter, and analyze tabular data

Pandas is the most-used Python library after NumPy.`,

	translations: {
		ru: {
			title: 'Создание DataFrame',
			description: `# Создание DataFrame

DataFrame — это 2D структура данных с метками, где столбцы могут иметь разные типы — как таблица в Excel или SQL.

## Задача

Реализуйте три функции:
1. \`from_dict(data)\` - Создать DataFrame из словаря
2. \`from_lists(data, columns)\` - Создать DataFrame из списка списков с именами столбцов
3. \`from_numpy(arr, columns)\` - Создать DataFrame из NumPy массива с именами столбцов

## Пример

\`\`\`python
# Из словаря
data = {'name': ['Alice', 'Bob'], 'age': [25, 30]}
from_dict(data)
#     name  age
# 0  Alice   25
# 1    Bob   30
\`\`\``,
			hint1: 'Используйте конструктор pd.DataFrame(data)',
			hint2: 'Передайте параметр columns для пользовательских имён столбцов',
			whyItMatters: `DataFrames — основа data science:

- **Загрузка данных**: Конвертация JSON, CSV, SQL результатов в DataFrames
- **Feature engineering**: Создание новых столбцов признаков
- **Вход модели**: Многие ML библиотеки принимают DataFrames напрямую
- **Исследование данных**: Просмотр, фильтрация и анализ табличных данных`,
		},
		uz: {
			title: "DataFrame yaratish",
			description: `# DataFrame yaratish

DataFrame - ustunlari turli tipdagi 2D yorliqli ma'lumotlar strukturasi - Excel yoki SQL jadvaliga o'xshash.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`from_dict(data)\` - Lug'atdan DataFrame yaratish
2. \`from_lists(data, columns)\` - Ro'yxatlar ro'yxatidan ustun nomlari bilan DataFrame yaratish
3. \`from_numpy(arr, columns)\` - NumPy massividan ustun nomlari bilan DataFrame yaratish

## Misol

\`\`\`python
data = {'name': ['Alice', 'Bob'], 'age': [25, 30]}
from_dict(data)
#     name  age
# 0  Alice   25
# 1    Bob   30
\`\`\``,
			hint1: "pd.DataFrame(data) konstruktoridan foydalaning",
			hint2: "Maxsus ustun nomlari uchun columns parametrini bering",
			whyItMatters: `DataFrames data science ning asosidir:

- **Ma'lumotlarni yuklash**: JSON, CSV, SQL natijalarini DataFrames ga aylantirish
- **Feature engineering**: Yangi xususiyat ustunlarini yaratish
- **Model kirishi**: Ko'p ML kutubxonalari DataFrames ni to'g'ridan-to'g'ri qabul qiladi`,
		},
	},
};

export default task;
