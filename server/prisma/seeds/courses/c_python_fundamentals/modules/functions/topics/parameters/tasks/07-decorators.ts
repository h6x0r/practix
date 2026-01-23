import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-decorators',
	title: 'Simple Decorator',
	difficulty: 'medium',
	tags: ['python', 'fundamentals', 'functions', 'decorators'],
	estimatedTime: '20m',
	isPremium: false,
	order: 7,

	description: `# Simple Decorator

Decorators wrap functions to add behavior without modifying the original code.

## Task

Implement the function \`call_counter(func)\` that creates a decorator counting function calls.

## Requirements

- Return a wrapper function that counts how many times the original function is called
- The wrapper should have a \`call_count\` attribute
- The wrapper should call the original function and return its result
- Each decorated function should have its own independent counter

## Examples

\`\`\`python
@call_counter
def greet(name):
    return f"Hello, {name}!"

>>> greet("Alice")
"Hello, Alice!"
>>> greet("Bob")
"Hello, Bob!"
>>> greet.call_count
2
\`\`\``,

	initialCode: `def call_counter(func):
    """Decorator that counts how many times a function is called.

    Args:
        func: The function to decorate

    Returns:
        Wrapper function with a call_count attribute
    """
    # TODO: Implement the decorator
    pass`,

	solutionCode: `def call_counter(func):
    """Decorator that counts how many times a function is called.

    Args:
        func: The function to decorate

    Returns:
        Wrapper function with a call_count attribute
    """
    def wrapper(*args, **kwargs):
        # Increment the call count
        wrapper.call_count += 1
        # Call and return the original function result
        return func(*args, **kwargs)

    # Initialize the counter attribute on the wrapper function
    wrapper.call_count = 0

    return wrapper`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Basic call counting"""
        @call_counter
        def add(a, b):
            return a + b

        add(1, 2)
        add(3, 4)
        self.assertEqual(add.call_count, 2)

    def test_2(self):
        """Return value preserved"""
        @call_counter
        def greet(name):
            return f"Hello, {name}!"

        result = greet("Alice")
        self.assertEqual(result, "Hello, Alice!")

    def test_3(self):
        """Initial count is 0"""
        @call_counter
        def noop():
            pass

        self.assertEqual(noop.call_count, 0)

    def test_4(self):
        """Independent counters"""
        @call_counter
        def func1():
            pass

        @call_counter
        def func2():
            pass

        func1()
        func1()
        func2()
        self.assertEqual(func1.call_count, 2)
        self.assertEqual(func2.call_count, 1)

    def test_5(self):
        """Works with no arguments"""
        @call_counter
        def say_hi():
            return "Hi"

        self.assertEqual(say_hi(), "Hi")
        self.assertEqual(say_hi.call_count, 1)

    def test_6(self):
        """Works with kwargs"""
        @call_counter
        def func(a, b=10):
            return a + b

        result = func(5, b=20)
        self.assertEqual(result, 25)

    def test_7(self):
        """Multiple calls increment"""
        @call_counter
        def inc(x):
            return x + 1

        for _ in range(5):
            inc(1)
        self.assertEqual(inc.call_count, 5)

    def test_8(self):
        """Works with *args"""
        @call_counter
        def sum_all(*args):
            return sum(args)

        result = sum_all(1, 2, 3, 4)
        self.assertEqual(result, 10)

    def test_9(self):
        """Counter persists across calls"""
        @call_counter
        def double(x):
            return x * 2

        double(1)
        double(2)
        double(3)
        self.assertEqual(double.call_count, 3)

    def test_10(self):
        """Has call_count attribute"""
        @call_counter
        def test_func():
            pass

        self.assertTrue(hasattr(test_func, 'call_count'))

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Define a wrapper function inside call_counter. Use *args and **kwargs to accept any arguments.',
	hint2: 'Functions are objects - you can add attributes to them. Set wrapper.call_count = 0 before returning wrapper.',

	whyItMatters: `Decorators are a powerful Python pattern used extensively in frameworks like Flask, Django, and FastAPI.

**Production Pattern:**

\`\`\`python
import functools
import time
from typing import Callable, TypeVar

T = TypeVar('T')

def timing_decorator(func: Callable[..., T]) -> Callable[..., T]:
    """Measure and log function execution time."""
    @functools.wraps(func)  # Preserve function metadata
    def wrapper(*args, **kwargs) -> T:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f}s")
        return result
    return wrapper

def retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator factory for retry logic."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

def cache_result(ttl_seconds: int = 300):
    """Simple time-based cache decorator."""
    def decorator(func):
        cache = {}

        @functools.wraps(func)
        def wrapper(*args):
            now = time.time()
            if args in cache:
                result, timestamp = cache[args]
                if now - timestamp < ttl_seconds:
                    return result
            result = func(*args)
            cache[args] = (result, now)
            return result
        return wrapper
    return decorator
\`\`\`

**Practical Benefits:**
- Cross-cutting concerns (logging, caching, auth)
- Clean separation of concerns
- Reusable behavior modification`,

	translations: {
		ru: {
			title: 'Простой декоратор',
			description: `# Простой декоратор

Декораторы оборачивают функции, добавляя поведение без изменения оригинального кода.

## Задача

Реализуйте функцию \`call_counter(func)\`, которая создаёт декоратор для подсчёта вызовов.

## Требования

- Верните функцию-обёртку, которая считает вызовы оригинальной функции
- Обёртка должна иметь атрибут \`call_count\`
- Обёртка должна вызывать оригинальную функцию и возвращать её результат
- Каждая декорированная функция должна иметь независимый счётчик

## Примеры

\`\`\`python
@call_counter
def greet(name):
    return f"Hello, {name}!"

>>> greet("Alice")
"Hello, Alice!"
>>> greet("Bob")
"Hello, Bob!"
>>> greet.call_count
2
\`\`\``,
			hint1: 'Определите функцию wrapper внутри call_counter. Используйте *args и **kwargs для любых аргументов.',
			hint2: 'Функции — это объекты, к ним можно добавлять атрибуты. Установите wrapper.call_count = 0.',
			whyItMatters: `Декораторы — мощный паттерн Python, широко используемый в Flask, Django и FastAPI.

**Продакшен паттерн:**

\`\`\`python
import functools
import time

def timing_decorator(func):
    """Измерение и логирование времени выполнения."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} заняло {end - start:.4f}s")
        return result
    return wrapper

def retry(max_attempts: int = 3, delay: float = 1.0):
    """Фабрика декораторов для повторных попыток."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    time.sleep(delay)
            raise e
        return wrapper
    return decorator
\`\`\`

**Практические преимущества:**
- Сквозная функциональность (логирование, кэширование, авторизация)
- Чистое разделение ответственности`,
		},
		uz: {
			title: 'Oddiy dekorator',
			description: `# Oddiy dekorator

Dekoratorlar funksiyalarni o'rab, asl kodni o'zgartirmasdan xatti-harakat qo'shadi.

## Vazifa

Chaqiruvlarni hisoblovchi dekorator yaratuvchi \`call_counter(func)\` funksiyasini amalga oshiring.

## Talablar

- Asl funksiya necha marta chaqirilganini hisoblovchi wrapper funksiya qaytaring
- Wrapper \`call_count\` atributiga ega bo'lishi kerak
- Wrapper asl funksiyani chaqirishi va natijasini qaytarishi kerak
- Har bir dekoratsiyalangan funksiya mustaqil hisoblagichga ega bo'lishi kerak

## Misollar

\`\`\`python
@call_counter
def greet(name):
    return f"Hello, {name}!"

>>> greet("Alice")
"Hello, Alice!"
>>> greet("Bob")
"Hello, Bob!"
>>> greet.call_count
2
\`\`\``,
			hint1: "call_counter ichida wrapper funksiya belgilang. Har qanday argumentlar uchun *args va **kwargs ishlating.",
			hint2: "Funksiyalar ob'ektlar — ularga atributlar qo'shish mumkin. wrapper.call_count = 0 o'rnating.",
			whyItMatters: `Dekoratorlar Flask, Django va FastAPI da keng ishlatiladigan kuchli Python patterni.

**Ishlab chiqarish patterni:**

\`\`\`python
import functools
import time

def timing_decorator(func):
    """Bajarilish vaqtini o'lchash va qayd qilish."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} {end - start:.4f}s oldi")
        return result
    return wrapper

def retry(max_attempts: int = 3, delay: float = 1.0):
    """Qayta urinish mantiqiy uchun dekorator fabrikasi."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    time.sleep(delay)
            raise e
        return wrapper
    return decorator
\`\`\`

**Amaliy foydalari:**
- Kesuvchi funksionallik (logging, keshlash, autorizatsiya)
- Mas'uliyatlarni toza ajratish`,
		},
	},
};

export default task;
