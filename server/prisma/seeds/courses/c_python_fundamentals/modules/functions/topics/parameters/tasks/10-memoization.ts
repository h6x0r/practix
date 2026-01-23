import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-memoization',
	title: 'Memoization',
	difficulty: 'medium',
	tags: ['python', 'fundamentals', 'functions', 'optimization'],
	estimatedTime: '15m',
	isPremium: false,
	order: 10,

	description: `# Memoization

Memoization caches function results to avoid redundant calculations.

## Task

Implement the function \`memoize(func)\` that creates a memoized version of any function.

## Requirements

- Return a wrapper function that caches results
- Cache key should be based on the function arguments
- Return cached result if the same arguments are passed again
- The wrapper should have a \`cache\` attribute (dictionary)

## Examples

\`\`\`python
@memoize
def expensive_calc(n):
    return n ** 2

>>> expensive_calc(5)
25  # Calculated
>>> expensive_calc(5)
25  # Returned from cache
>>> expensive_calc.cache
{(5,): 25}
\`\`\``,

	initialCode: `def memoize(func):
    """Decorator that caches function results.

    Args:
        func: Function to memoize

    Returns:
        Memoized function with a cache attribute
    """
    # TODO: Implement memoization
    pass`,

	solutionCode: `def memoize(func):
    """Decorator that caches function results.

    Args:
        func: Function to memoize

    Returns:
        Memoized function with a cache attribute
    """
    def wrapper(*args):
        # Check if result is in cache
        if args not in wrapper.cache:
            # Calculate and store result
            wrapper.cache[args] = func(*args)
        # Return cached result
        return wrapper.cache[args]

    # Initialize empty cache
    wrapper.cache = {}

    return wrapper`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Basic memoization"""
        @memoize
        def square(n):
            return n ** 2

        self.assertEqual(square(5), 25)
        self.assertEqual(square(5), 25)
        self.assertEqual(len(square.cache), 1)

    def test_2(self):
        """Multiple different calls"""
        @memoize
        def add(a, b):
            return a + b

        add(1, 2)
        add(3, 4)
        add(1, 2)  # Cached
        self.assertEqual(len(add.cache), 2)

    def test_3(self):
        """Correct cached value"""
        @memoize
        def multiply(x, y):
            return x * y

        self.assertEqual(multiply(3, 4), 12)
        self.assertIn((3, 4), multiply.cache)
        self.assertEqual(multiply.cache[(3, 4)], 12)

    def test_4(self):
        """Has cache attribute"""
        @memoize
        def func(x):
            return x

        self.assertTrue(hasattr(func, 'cache'))
        self.assertIsInstance(func.cache, dict)

    def test_5(self):
        """Empty cache initially"""
        @memoize
        def func(x):
            return x

        self.assertEqual(len(func.cache), 0)

    def test_6(self):
        """Cache hit returns same value"""
        @memoize
        def func(x):
            return {"value": x}

        result1 = func(1)
        result2 = func(1)
        self.assertIs(result1, result2)

    def test_7(self):
        """Works with no arguments"""
        call_count = 0
        @memoize
        def get_value():
            nonlocal call_count
            call_count += 1
            return 42

        get_value()
        get_value()
        self.assertEqual(call_count, 1)

    def test_8(self):
        """Different args give different results"""
        @memoize
        def identity(x):
            return x

        self.assertEqual(identity(1), 1)
        self.assertEqual(identity(2), 2)
        self.assertEqual(len(identity.cache), 2)

    def test_9(self):
        """Negative numbers work"""
        @memoize
        def abs_val(x):
            return abs(x)

        self.assertEqual(abs_val(-5), 5)
        self.assertIn((-5,), abs_val.cache)

    def test_10(self):
        """Multiple arguments cached correctly"""
        @memoize
        def concat(a, b, c):
            return f"{a}-{b}-{c}"

        self.assertEqual(concat("x", "y", "z"), "x-y-z")
        self.assertIn(("x", "y", "z"), concat.cache)

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Use a dictionary to store results. The key is the tuple of arguments (*args as tuple is hashable).',
	hint2: 'Before calling func, check if args is already in wrapper.cache. If so, return the cached value.',

	whyItMatters: `Memoization dramatically improves performance for functions with repeated calls and expensive computations.

**Production Pattern:**

\`\`\`python
from functools import lru_cache

# Built-in memoization with LRU eviction
@lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Custom memoization with TTL (time-to-live)
import time

def memoize_with_ttl(ttl_seconds: int):
    """Memoization decorator with cache expiration."""
    def decorator(func):
        cache = {}

        def wrapper(*args):
            now = time.time()
            if args in cache:
                result, timestamp = cache[args]
                if now - timestamp < ttl_seconds:
                    return result

            result = func(*args)
            cache[args] = (result, now)
            return result

        wrapper.cache = cache
        wrapper.clear_cache = lambda: cache.clear()
        return wrapper

    return decorator

# Async memoization for async functions
import asyncio

def async_memoize(func):
    """Memoization for async functions."""
    cache = {}

    async def wrapper(*args):
        if args not in cache:
            cache[args] = await func(*args)
        return cache[args]

    wrapper.cache = cache
    return wrapper
\`\`\`

**Practical Benefits:**
- Recursive algorithms (Fibonacci, tree traversal)
- API response caching
- Expensive database query results`,

	translations: {
		ru: {
			title: 'Мемоизация',
			description: `# Мемоизация

Мемоизация кэширует результаты функций для избежания повторных вычислений.

## Задача

Реализуйте функцию \`memoize(func)\`, которая создаёт мемоизированную версию любой функции.

## Требования

- Верните функцию-обёртку, которая кэширует результаты
- Ключ кэша должен основываться на аргументах функции
- Верните кэшированный результат при повторных вызовах с теми же аргументами
- Обёртка должна иметь атрибут \`cache\` (словарь)

## Примеры

\`\`\`python
@memoize
def expensive_calc(n):
    return n ** 2

>>> expensive_calc(5)
25  # Вычислено
>>> expensive_calc(5)
25  # Возвращено из кэша
>>> expensive_calc.cache
{(5,): 25}
\`\`\``,
			hint1: 'Используйте словарь для хранения результатов. Ключ — это кортеж аргументов (*args как tuple хэшируем).',
			hint2: 'Перед вызовом func проверьте, есть ли args в wrapper.cache. Если да — верните кэшированное.',
			whyItMatters: `Мемоизация резко улучшает производительность для функций с повторными вызовами.

**Продакшен паттерн:**

\`\`\`python
from functools import lru_cache

# Встроенная мемоизация с LRU вытеснением
@lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Пользовательская мемоизация с TTL
import time

def memoize_with_ttl(ttl_seconds: int):
    """Декоратор мемоизации с истечением кэша."""
    def decorator(func):
        cache = {}
        def wrapper(*args):
            now = time.time()
            if args in cache:
                result, timestamp = cache[args]
                if now - timestamp < ttl_seconds:
                    return result
            result = func(*args)
            cache[args] = (result, now)
            return result
        wrapper.cache = cache
        return wrapper
    return decorator
\`\`\`

**Практические преимущества:**
- Рекурсивные алгоритмы (Фибоначчи, обход деревьев)
- Кэширование ответов API
- Результаты дорогих запросов к БД`,
		},
		uz: {
			title: 'Memoizatsiya',
			description: `# Memoizatsiya

Memoizatsiya ortiqcha hisoblashlardan qochish uchun funksiya natijalarini keshlaydi.

## Vazifa

Har qanday funksiyaning memoizatsiyalangan versiyasini yaratuvchi \`memoize(func)\` funksiyasini amalga oshiring.

## Talablar

- Natijalarni keshlaydigan wrapper funksiya qaytaring
- Kesh kaliti funksiya argumentlariga asoslanishi kerak
- Bir xil argumentlar bilan qayta chaqiruvlarda keshlangan natijani qaytaring
- Wrapper \`cache\` atributiga (lug'at) ega bo'lishi kerak

## Misollar

\`\`\`python
@memoize
def expensive_calc(n):
    return n ** 2

>>> expensive_calc(5)
25  # Hisoblandi
>>> expensive_calc(5)
25  # Keshdan qaytarildi
>>> expensive_calc.cache
{(5,): 25}
\`\`\``,
			hint1: "Natijalarni saqlash uchun lug'at ishlating. Kalit argumentlar korteji (*args tuple sifatida xeshlanadi).",
			hint2: "func ni chaqirishdan oldin args wrapper.cache da borligini tekshiring. Bo'lsa — keshlangan ni qaytaring.",
			whyItMatters: `Memoizatsiya takroriy chaqiruvlarga ega funksiyalar uchun ish faoliyatini keskin yaxshilaydi.

**Ishlab chiqarish patterni:**

\`\`\`python
from functools import lru_cache

# LRU siqib chiqarish bilan o'rnatilgan memoizatsiya
@lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# TTL bilan maxsus memoizatsiya
import time

def memoize_with_ttl(ttl_seconds: int):
    """Kesh tugashi bilan memoizatsiya dekoratori."""
    def decorator(func):
        cache = {}
        def wrapper(*args):
            now = time.time()
            if args in cache:
                result, timestamp = cache[args]
                if now - timestamp < ttl_seconds:
                    return result
            result = func(*args)
            cache[args] = (result, now)
            return result
        wrapper.cache = cache
        return wrapper
    return decorator
\`\`\`

**Amaliy foydalari:**
- Rekursiv algoritmlar (Fibonachchi, daraxt traversal)
- API javoblarini keshlash
- Qimmat ma'lumotlar bazasi so'rovlari natijalari`,
		},
	},
};

export default task;
