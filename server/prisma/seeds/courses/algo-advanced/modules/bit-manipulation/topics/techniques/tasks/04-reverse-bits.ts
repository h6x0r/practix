import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'bit-manipulation-reverse-bits',
	title: 'Reverse Bits',
	difficulty: 'easy',
	tags: ['python', 'bit-manipulation', 'divide-and-conquer'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Reverse the bits of a given 32-bit unsigned integer.

**Problem:**

Reverse bits of a given 32 bits unsigned integer.

**Note:**
- In some languages, there is no unsigned integer type. The integer's internal binary representation is the same, whether it is signed or unsigned.
- In Python, integers have arbitrary precision, so we treat the input as a 32-bit unsigned integer.

**Examples:**

\`\`\`
Input: n = 43261596
Input binary:  00000010100101000001111010011100
Output: 964176192
Output binary: 00111001011110000010100101000000

Input: n = 4294967293
Input binary:  11111111111111111111111111111101
Output: 3221225471
Output binary: 10111111111111111111111111111111
\`\`\`

**Key Insight:**

Extract each bit from right to left, and build the result from left to right:
1. Get the rightmost bit of n using \`n & 1\`
2. Add it to result shifted left
3. Right shift n to process the next bit

**Constraints:**
- The input is a 32-bit unsigned integer

**Follow up:** If this function is called many times, how would you optimize it?

**Time Complexity:** O(32) = O(1)
**Space Complexity:** O(1)`,
	initialCode: `def reverse_bits(n: int) -> int:
    # TODO: Reverse bits of a 32-bit unsigned integer

    return 0`,
	solutionCode: `def reverse_bits(n: int) -> int:
    """
    Reverse bits by extracting and placing each bit.
    """
    result = 0
    for _ in range(32):
        # Shift result left, add rightmost bit of n
        result = (result << 1) | (n & 1)
        n >>= 1
    return result


# Approach 2: Using bit positions explicitly
def reverse_bits_positions(n: int) -> int:
    """Extract each bit and place at reversed position."""
    result = 0
    for i in range(32):
        bit = (n >> i) & 1
        result |= bit << (31 - i)
    return result


# Approach 3: Divide and conquer (swap halves recursively)
def reverse_bits_divide_conquer(n: int) -> int:
    """
    Swap halves at each level:
    - Swap 16-bit halves
    - Swap 8-bit halves within each 16-bit half
    - Swap 4-bit halves within each 8-bit half
    - Swap 2-bit halves within each 4-bit half
    - Swap individual bits within each 2-bit half
    """
    # Swap 16-bit halves
    n = (n >> 16) | (n << 16)
    # Swap 8-bit halves
    n = ((n & 0xFF00FF00) >> 8) | ((n & 0x00FF00FF) << 8)
    # Swap 4-bit halves
    n = ((n & 0xF0F0F0F0) >> 4) | ((n & 0x0F0F0F0F) << 4)
    # Swap 2-bit halves
    n = ((n & 0xCCCCCCCC) >> 2) | ((n & 0x33333333) << 2)
    # Swap individual bits
    n = ((n & 0xAAAAAAAA) >> 1) | ((n & 0x55555555) << 1)
    return n & 0xFFFFFFFF


# Approach 4: Using lookup table (optimized for multiple calls)
class ReverseBitsLookup:
    """Byte-level lookup table for O(1) reversal."""

    def __init__(self):
        # Precompute reversal for all bytes (0-255)
        self.table = [0] * 256
        for i in range(256):
            self.table[i] = self._reverse_byte(i)

    def _reverse_byte(self, b: int) -> int:
        """Reverse 8 bits."""
        result = 0
        for _ in range(8):
            result = (result << 1) | (b & 1)
            b >>= 1
        return result

    def reverse(self, n: int) -> int:
        """Reverse 32 bits using lookup table."""
        return (
            (self.table[n & 0xFF] << 24) |
            (self.table[(n >> 8) & 0xFF] << 16) |
            (self.table[(n >> 16) & 0xFF] << 8) |
            (self.table[(n >> 24) & 0xFF])
        )


# Approach 5: Python string manipulation (not recommended for interviews)
def reverse_bits_string(n: int) -> int:
    """Using string operations."""
    binary = bin(n)[2:].zfill(32)  # Remove '0b', pad to 32 bits
    return int(binary[::-1], 2)`,
	testCode: `import pytest
from solution import reverse_bits


class TestReverseBits:
    def test_example_one(self):
        """Test first example"""
        assert reverse_bits(43261596) == 964176192

    def test_example_two(self):
        """Test second example"""
        assert reverse_bits(4294967293) == 3221225471

    def test_zero(self):
        """Test zero"""
        assert reverse_bits(0) == 0

    def test_all_ones(self):
        """Test all 1s (max 32-bit value)"""
        assert reverse_bits(0xFFFFFFFF) == 0xFFFFFFFF

    def test_one(self):
        """Test 1 -> highest bit set"""
        assert reverse_bits(1) == 2147483648  # 1 << 31

    def test_highest_bit(self):
        """Test highest bit set -> 1"""
        assert reverse_bits(2147483648) == 1  # 1 << 31 -> 1

    def test_alternating_bits(self):
        """Test alternating bits"""
        # 10101010... -> 01010101...
        n = 0xAAAAAAAA  # 10101010...
        expected = 0x55555555  # 01010101...
        assert reverse_bits(n) == expected

    def test_palindrome(self):
        """Test palindrome number stays same"""
        # 11000000000000000000000000000011
        n = 0xC0000003
        assert reverse_bits(n) == n

    def test_single_byte_low(self):
        """Test single byte in low position"""
        # 00000000000000000000000011111111 -> 11111111000000000000000000000000
        assert reverse_bits(0xFF) == 0xFF000000

    def test_double_reverse(self):
        """Test that reversing twice gives original"""
        n = 12345
        assert reverse_bits(reverse_bits(n)) == n`,
	hint1: `Extract each bit from the right side of n using \`n & 1\`, then shift n right. Build the result by shifting left and adding each extracted bit.`,
	hint2: `Loop 32 times: \`result = (result << 1) | (n & 1)\` then \`n >>= 1\`. This moves each bit from position i to position 31-i.`,
	whyItMatters: `Reverse Bits demonstrates bit-by-bit manipulation and the powerful divide-and-conquer approach for bit operations that achieves O(log n) parallel swaps.

**Why This Matters:**

**1. Basic Bit Extraction**

\`\`\`python
# Extract rightmost bit: n & 1
# Shift right: n >> 1
# Shift left: result << 1
# Combine: result | bit

# These operations form the foundation of all bit manipulation
\`\`\`

**2. Divide and Conquer Approach**

\`\`\`python
# Instead of 32 iterations, use 5 parallel swaps:
# 1. Swap 16-bit halves: (n >> 16) | (n << 16)
# 2. Swap 8-bit halves: mask and shift
# 3. Swap 4-bit halves
# 4. Swap 2-bit halves
# 5. Swap individual bits

# This is how hardware implements bit reversal efficiently
\`\`\`

**3. Lookup Table Optimization**

\`\`\`python
# For frequent calls, precompute reversal for all bytes
# Then combine 4 reversed bytes
# Trade memory (256 entries) for speed

# This pattern applies to many bit operations:
# - Population count
# - Parity calculation
# - Gray code conversion
\`\`\`

**4. Applications**

\`\`\`python
# FFT (Fast Fourier Transform) uses bit reversal
# Network byte order conversion
# Graphics and image processing
# Certain encryption algorithms
\`\`\``,
	order: 4,
	translations: {
		ru: {
			title: 'Реверс битов',
			description: `Разверните биты 32-битного беззнакового целого числа.

**Задача:**

Разверните биты данного 32-битного беззнакового целого числа.

**Примеры:**

\`\`\`
Вход: n = 43261596
Двоичный вход:  00000010100101000001111010011100
Выход: 964176192
Двоичный выход: 00111001011110000010100101000000

Вход: n = 4294967293
Двоичный вход:  11111111111111111111111111111101
Выход: 3221225471
\`\`\`

**Ключевая идея:**

Извлекайте каждый бит справа налево и стройте результат слева направо:
1. Получите правый бит n через \`n & 1\`
2. Добавьте его к результату, сдвинутому влево
3. Сдвиньте n вправо для обработки следующего бита

**Ограничения:**
- Вход - 32-битное беззнаковое целое

**Временная сложность:** O(32) = O(1)
**Пространственная сложность:** O(1)`,
			hint1: `Извлекайте каждый бит справа через \`n & 1\`, затем сдвигайте n вправо. Стройте результат сдвигом влево и добавлением бита.`,
			hint2: `Цикл 32 раза: \`result = (result << 1) | (n & 1)\` затем \`n >>= 1\`. Это перемещает бит с позиции i на позицию 31-i.`,
			whyItMatters: `Реверс битов демонстрирует побитовую манипуляцию и мощный подход "разделяй и властвуй" для битовых операций.

**Почему это важно:**

**1. Базовое извлечение битов**

Извлечение правого бита, сдвиги влево/вправо, комбинирование.

**2. Подход "разделяй и властвуй"**

Вместо 32 итераций - 5 параллельных обменов: swap половин 16, 8, 4, 2 бит.

**3. Оптимизация таблицей**

Предвычисление для всех байтов для частых вызовов.

**4. Применения**

FFT, сетевой порядок байтов, графика, шифрование.`,
			solutionCode: `def reverse_bits(n: int) -> int:
    """Разворачивает биты извлечением и размещением каждого бита."""
    result = 0
    for _ in range(32):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result`
		},
		uz: {
			title: 'Bitlarni teskari aylantirish',
			description: `32-bitli ishorasiz butun sonning bitlarini teskari aylantiring.

**Masala:**

Berilgan 32-bitli ishorasiz butun sonning bitlarini teskari aylantiring.

**Misollar:**

\`\`\`
Kirish: n = 43261596
Ikkilik kirish:  00000010100101000001111010011100
Chiqish: 964176192
Ikkilik chiqish: 00111001011110000010100101000000

Kirish: n = 4294967293
Ikkilik kirish:  11111111111111111111111111111101
Chiqish: 3221225471
\`\`\`

**Asosiy tushuncha:**

Har bir bitni o'ngdan chapga ajratib oling va natijani chapdan o'ngga quring:
1. n ning o'ng bitini \`n & 1\` orqali oling
2. Uni chapga siljitilgan natijaga qo'shing
3. Keyingi bit uchun n ni o'ngga siljiting

**Cheklovlar:**
- Kirish 32-bitli ishorasiz butun son

**Vaqt murakkabligi:** O(32) = O(1)
**Xotira murakkabligi:** O(1)`,
			hint1: `Har bir bitni o'ngdan \`n & 1\` orqali ajrating, keyin n ni o'ngga siljiting. Natijani chapga siljitish va bit qo'shish orqali quring.`,
			hint2: `32 marta sikl: \`result = (result << 1) | (n & 1)\` keyin \`n >>= 1\`. Bu bitni i pozitsiyadan 31-i pozitsiyaga ko'chiradi.`,
			whyItMatters: `Bitlarni teskari aylantirish bit-by-bit manipulyatsiyasini va bit operatsiyalari uchun kuchli "bo'l va hukmronlik qil" yondashuvini ko'rsatadi.

**Bu nima uchun muhim:**

**1. Asosiy bit ajratish**

O'ng bitni ajratish, chap/o'ng siljitishlar, birlashtirish.

**2. "Bo'l va hukmronlik qil" yondashuvi**

32 iteratsiya o'rniga 5 ta parallel almashtirish: 16, 8, 4, 2 bit yarmlarini swap.

**3. Jadval bilan optimallashtirish**

Tez-tez chaqirishlar uchun barcha baytlarni oldindan hisoblash.

**4. Qo'llanishlar**

FFT, tarmoq bayt tartibi, grafika, shifrlash.`,
			solutionCode: `def reverse_bits(n: int) -> int:
    """Har bir bitni ajratib va joylashtirish orqali bitlarni teskari aylantiradi."""
    result = 0
    for _ in range(32):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result`
		}
	}
};

export default task;
