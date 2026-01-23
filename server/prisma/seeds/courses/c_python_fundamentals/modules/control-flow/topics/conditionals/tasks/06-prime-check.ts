import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-prime-check',
	title: 'Prime Number Check',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'loops', 'math'],
	estimatedTime: '15m',
	isPremium: false,
	order: 6,

	description: `# Prime Number Check

Learn to use loops for mathematical validation.

## Task

Implement the function \`is_prime(n)\` that checks if a number is prime.

## Requirements

- A prime number is greater than 1 and only divisible by 1 and itself
- Return \`True\` if n is prime, \`False\` otherwise
- Handle edge cases: numbers less than 2 are not prime

## Examples

\`\`\`python
>>> is_prime(7)
True  # Only divisible by 1 and 7

>>> is_prime(4)
False  # Divisible by 2

>>> is_prime(1)
False  # 1 is not prime by definition

>>> is_prime(2)
True  # Smallest prime number
\`\`\``,

	initialCode: `def is_prime(n: int) -> bool:
    """Check if a number is prime.

    A prime number is:
    - Greater than 1
    - Only divisible by 1 and itself

    Args:
        n: Integer to check

    Returns:
        True if n is prime, False otherwise
    """
    # TODO: Implement this function
    pass`,

	solutionCode: `def is_prime(n: int) -> bool:
    """Check if a number is prime.

    A prime number is:
    - Greater than 1
    - Only divisible by 1 and itself

    Args:
        n: Integer to check

    Returns:
        True if n is prime, False otherwise
    """
    # Numbers less than 2 are not prime
    if n < 2:
        return False

    # 2 is the only even prime
    if n == 2:
        return True

    # All other even numbers are not prime
    if n % 2 == 0:
        return False

    # Check odd divisors up to square root of n
    # If n has a divisor larger than sqrt(n), it must also have
    # a corresponding divisor smaller than sqrt(n)
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2  # Skip even numbers

    return True`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """7 is prime"""
        self.assertTrue(is_prime(7))

    def test_2(self):
        """4 is not prime"""
        self.assertFalse(is_prime(4))

    def test_3(self):
        """1 is not prime"""
        self.assertFalse(is_prime(1))

    def test_4(self):
        """2 is prime (smallest prime)"""
        self.assertTrue(is_prime(2))

    def test_5(self):
        """0 is not prime"""
        self.assertFalse(is_prime(0))

    def test_6(self):
        """Negative numbers are not prime"""
        self.assertFalse(is_prime(-5))

    def test_7(self):
        """13 is prime"""
        self.assertTrue(is_prime(13))

    def test_8(self):
        """15 is not prime (3 * 5)"""
        self.assertFalse(is_prime(15))

    def test_9(self):
        """97 is prime"""
        self.assertTrue(is_prime(97))

    def test_10(self):
        """100 is not prime"""
        self.assertFalse(is_prime(100))

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'First handle edge cases: numbers less than 2 are not prime. 2 is the only even prime.',
	hint2: 'You only need to check divisors up to the square root of n. Use a while loop: while i * i <= n.',

	whyItMatters: `Prime number algorithms are fundamental to cryptography and security.

**Production Pattern:**

\`\`\`python
def generate_primes(limit: int) -> list[int]:
    """Sieve of Eratosthenes - efficient prime generation."""
    if limit < 2:
        return []

    # Start with all numbers marked as potentially prime
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False

    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            # Mark all multiples of i as not prime
            for j in range(i * i, limit + 1, i):
                sieve[j] = False

    return [i for i, is_prime in enumerate(sieve) if is_prime]

def find_prime_factors(n: int) -> list[int]:
    """Find all prime factors of a number."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors
\`\`\`

**Practical Benefits:**
- Cryptographic key generation uses large primes
- Hash table sizing often uses prime numbers
- Understanding factorization helps with optimization`,

	translations: {
		ru: {
			title: 'Проверка простых чисел',
			description: `# Проверка простых чисел

Научитесь использовать циклы для математической проверки.

## Задача

Реализуйте функцию \`is_prime(n)\`, которая проверяет, является ли число простым.

## Требования

- Простое число больше 1 и делится только на 1 и на себя
- Верните \`True\` если n простое, \`False\` в противном случае
- Обработайте краевые случаи: числа меньше 2 не являются простыми

## Примеры

\`\`\`python
>>> is_prime(7)
True  # Делится только на 1 и 7

>>> is_prime(4)
False  # Делится на 2

>>> is_prime(1)
False  # 1 не является простым по определению

>>> is_prime(2)
True  # Наименьшее простое число
\`\`\``,
			hint1: 'Сначала обработайте краевые случаи: числа меньше 2 не простые. 2 — единственное чётное простое.',
			hint2: 'Достаточно проверять делители до квадратного корня из n. Используйте while i * i <= n.',
			whyItMatters: `Алгоритмы простых чисел — основа криптографии и безопасности.

**Продакшен паттерн:**

\`\`\`python
def generate_primes(limit: int) -> list[int]:
    """Решето Эратосфена — эффективная генерация простых чисел."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]
\`\`\`

**Практические преимущества:**
- Генерация криптографических ключей использует большие простые числа
- Размеры хеш-таблиц часто используют простые числа`,
		},
		uz: {
			title: "Tub sonlarni tekshirish",
			description: `# Tub sonlarni tekshirish

Matematik tekshirish uchun tsikllardan foydalanishni o'rganing.

## Vazifa

Sonning tub ekanligini tekshiruvchi \`is_prime(n)\` funksiyasini amalga oshiring.

## Talablar

- Tub son 1 dan katta va faqat 1 va o'ziga bo'linadi
- n tub bo'lsa \`True\`, aks holda \`False\` qaytaring
- Chegaraviy holatlar: 2 dan kichik sonlar tub emas

## Misollar

\`\`\`python
>>> is_prime(7)
True  # Faqat 1 va 7 ga bo'linadi

>>> is_prime(4)
False  # 2 ga bo'linadi

>>> is_prime(1)
False  # 1 ta'rif bo'yicha tub emas

>>> is_prime(2)
True  # Eng kichik tub son
\`\`\``,
			hint1: "Avval chegaraviy holatlarni ko'rib chiqing: 2 dan kichik sonlar tub emas. 2 yagona juft tub son.",
			hint2: "n ning kvadrat ildizigacha bo'lgan bo'luvchilarni tekshirish kifoya. while i * i <= n ishlatang.",
			whyItMatters: `Tub sonlar algoritmlari kriptografiya va xavfsizlikning asosidir.

**Ishlab chiqarish patterni:**

\`\`\`python
def generate_primes(limit: int) -> list[int]:
    """Eratosfen elagi — samarali tub sonlar generatsiyasi."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]
\`\`\`

**Amaliy foydalari:**
- Kriptografik kalit generatsiyasi katta tub sonlardan foydalanadi
- Xesh jadval o'lchamlari ko'pincha tub sonlardan foydalanadi`,
		},
	},
};

export default task;
