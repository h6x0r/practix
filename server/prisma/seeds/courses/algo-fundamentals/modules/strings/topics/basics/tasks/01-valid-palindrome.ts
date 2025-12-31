import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-valid-palindrome',
	title: 'Valid Palindrome',
	difficulty: 'easy',
	tags: ['python', 'strings', 'two-pointers'],
	estimatedTime: '15m',
	isPremium: false,
	youtubeUrl: '',
	description: `Check if a string is a valid palindrome, considering only alphanumeric characters.

**Problem:**

Given a string \`s\`, return \`True\` if it is a palindrome, considering only alphanumeric characters and ignoring cases.

**Examples:**

\`\`\`
Input: s = "A man, a plan, a canal: Panama"
Output: True
Explanation: "amanaplanacanalpanama" is a palindrome

Input: s = "race a car"
Output: False
Explanation: "raceacar" is not a palindrome

Input: s = " "
Output: True
Explanation: Empty string after removing non-alphanumeric is palindrome
\`\`\`

**Two Pointers Approach:**

1. Use two pointers: left (start) and right (end)
2. Skip non-alphanumeric characters
3. Compare characters (case-insensitive)
4. Move pointers towards center

\`\`\`python
while left < right:
    while left < right and not s[left].isalnum():
        left += 1
    while left < right and not s[right].isalnum():
        right -= 1
    if s[left].lower() != s[right].lower():
        return False
    left += 1
    right -= 1
\`\`\`

**Time Complexity:** O(n)
**Space Complexity:** O(1)`,
	initialCode: `def is_palindrome(s: str) -> bool:
    # TODO: Check if string is a valid palindrome (alphanumeric only, case-insensitive)

    return False`,
	solutionCode: `def is_palindrome(s: str) -> bool:
    """
    Check if string is a valid palindrome.
    Only considers alphanumeric characters, case-insensitive.

    Args:
        s: Input string

    Returns:
        True if s is a valid palindrome, False otherwise
    """
    left, right = 0, len(s) - 1

    while left < right:
        # Skip non-alphanumeric from left
        while left < right and not s[left].isalnum():
            left += 1
        # Skip non-alphanumeric from right
        while left < right and not s[right].isalnum():
            right -= 1

        # Compare characters (case-insensitive)
        if s[left].lower() != s[right].lower():
            return False

        left += 1
        right -= 1

    return True`,
	testCode: `import pytest
from solution import is_palindrome

class TestIsPalindrome:
    def test_classic_palindrome(self):
        """Test classic palindrome with spaces and punctuation"""
        assert is_palindrome("A man, a plan, a canal: Panama") == True

    def test_not_palindrome(self):
        """Test string that is not a palindrome"""
        assert is_palindrome("race a car") == False

    def test_empty_string(self):
        """Test empty/whitespace string"""
        assert is_palindrome(" ") == True

    def test_single_char(self):
        """Test single character"""
        assert is_palindrome("a") == True

    def test_numbers_palindrome(self):
        """Test palindrome with numbers"""
        assert is_palindrome("12321") == True

    def test_mixed_palindrome(self):
        """Test mixed letters and numbers"""
        assert is_palindrome("a1b2b1a") == True

    def test_case_insensitive(self):
        """Test case insensitivity"""
        assert is_palindrome("AbBa") == True

    def test_special_chars_only(self):
        """Test string with only special characters"""
        assert is_palindrome(".,") == True

    def test_not_palindrome_simple(self):
        """Test simple non-palindrome"""
        assert is_palindrome("hello") == False

    def test_alphanumeric_palindrome(self):
        """Test alphanumeric palindrome"""
        assert is_palindrome("0P") == False`,
	hint1: `Use two pointers starting from both ends. Use while loops with isalnum() to skip non-alphanumeric characters before comparing.`,
	hint2: `Python's str.isalnum() returns True for letters and digits. Use str.lower() for case-insensitive comparison.`,
	whyItMatters: `The two-pointer technique for palindromes is a foundation for many string problems.

**Why This Matters:**

**1. Two Pointer Pattern**

This pattern appears everywhere:
- Container with Most Water
- Trapping Rain Water
- 3Sum, 4Sum problems
- Linked list cycle detection

**2. Python String Methods**

Understanding built-in methods saves time:
\`\`\`python
# Character checking
'a'.isalnum()  # True - alphanumeric
'A'.isalpha()  # True - letter only
'5'.isdigit()  # True - digit only

# Case conversion
'ABC'.lower()  # 'abc'
'abc'.upper()  # 'ABC'
\`\`\`

**3. Input Sanitization**

Real-world strings are messy:
- User input has spaces, punctuation
- Need to normalize before processing
- This pattern handles edge cases gracefully

**4. Space Efficiency**

O(1) space vs O(n) for creating cleaned string:
\`\`\`python
# O(n) space - creates new string
cleaned = ''.join(c.lower() for c in s if c.isalnum())
return cleaned == cleaned[::-1]

# O(1) space - two pointers
return is_palindrome(s)  # in-place checking
\`\`\``,
	order: 1,
	translations: {
		ru: {
			title: 'Проверка палиндрома',
			description: `Проверьте, является ли строка палиндромом, учитывая только буквенно-цифровые символы.

**Задача:**

Дана строка \`s\`, верните \`True\`, если она является палиндромом, учитывая только буквенно-цифровые символы и игнорируя регистр.

**Примеры:**

\`\`\`
Вход: s = "A man, a plan, a canal: Panama"
Выход: True
Объяснение: "amanaplanacanalpanama" - палиндром

Вход: s = "race a car"
Выход: False
Объяснение: "raceacar" - не палиндром

Вход: s = " "
Выход: True
Объяснение: Пустая строка после удаления небуквенно-цифровых символов - палиндром
\`\`\`

**Подход двух указателей:**

1. Используйте два указателя: left (начало) и right (конец)
2. Пропускайте небуквенно-цифровые символы
3. Сравнивайте символы (без учёта регистра)
4. Двигайте указатели к центру

\`\`\`python
while left < right:
    while left < right and not s[left].isalnum():
        left += 1
    while left < right and not s[right].isalnum():
        right -= 1
    if s[left].lower() != s[right].lower():
        return False
    left += 1
    right -= 1
\`\`\`

**Временная сложность:** O(n)
**Пространственная сложность:** O(1)`,
			hint1: `Используйте два указателя, начиная с обоих концов. Используйте циклы while с isalnum() для пропуска небуквенно-цифровых символов перед сравнением.`,
			hint2: `Python str.isalnum() возвращает True для букв и цифр. Используйте str.lower() для сравнения без учёта регистра.`,
			whyItMatters: `Техника двух указателей для палиндромов - основа для многих задач со строками.

**Почему это важно:**

**1. Паттерн двух указателей**

Этот паттерн встречается везде:
- Container with Most Water
- Trapping Rain Water
- 3Sum, 4Sum задачи
- Обнаружение цикла в связном списке

**2. Методы строк Python**

Понимание встроенных методов экономит время:
\`\`\`python
# Проверка символов
'a'.isalnum()  # True - буквенно-цифровой
'A'.isalpha()  # True - только буква
'5'.isdigit()  # True - только цифра

# Преобразование регистра
'ABC'.lower()  # 'abc'
'abc'.upper()  # 'ABC'
\`\`\`

**3. Эффективность по памяти**

O(1) память против O(n) для создания очищенной строки:
\`\`\`python
# O(n) памяти - создаёт новую строку
cleaned = ''.join(c.lower() for c in s if c.isalnum())
return cleaned == cleaned[::-1]

# O(1) памяти - два указателя
return is_palindrome(s)  # проверка на месте
\`\`\``,
			solutionCode: `def is_palindrome(s: str) -> bool:
    """
    Проверяет, является ли строка палиндромом.
    Учитывает только буквенно-цифровые символы, без учёта регистра.

    Args:
        s: Входная строка

    Returns:
        True если s - палиндром, иначе False
    """
    left, right = 0, len(s) - 1

    while left < right:
        # Пропускаем небуквенно-цифровые слева
        while left < right and not s[left].isalnum():
            left += 1
        # Пропускаем небуквенно-цифровые справа
        while left < right and not s[right].isalnum():
            right -= 1

        # Сравниваем символы (без учёта регистра)
        if s[left].lower() != s[right].lower():
            return False

        left += 1
        right -= 1

    return True`
		},
		uz: {
			title: 'Palindromni tekshirish',
			description: `Satr palindrom ekanligini tekshiring, faqat harf-raqamli belgilarni hisobga olib.

**Masala:**

Satr \`s\` berilgan, faqat harf-raqamli belgilarni hisobga olib va registrni e'tiborsiz qoldirib, u palindrom bo'lsa \`True\` qaytaring.

**Misollar:**

\`\`\`
Kirish: s = "A man, a plan, a canal: Panama"
Chiqish: True
Tushuntirish: "amanaplanacanalpanama" palindrom

Kirish: s = "race a car"
Chiqish: False
Tushuntirish: "raceacar" palindrom emas

Kirish: s = " "
Chiqish: True
Tushuntirish: Harf-raqamli bo'lmaganlarni olib tashlashdan keyin bo'sh satr palindrom
\`\`\`

**Ikki ko'rsatkich yondashuvi:**

1. Ikki ko'rsatkich ishlating: left (boshi) va right (oxiri)
2. Harf-raqamli bo'lmagan belgilarni o'tkazib yuboring
3. Belgilarni solishtiring (registrni hisobga olmay)
4. Ko'rsatkichlarni markazga qarab siljiting

\`\`\`python
while left < right:
    while left < right and not s[left].isalnum():
        left += 1
    while left < right and not s[right].isalnum():
        right -= 1
    if s[left].lower() != s[right].lower():
        return False
    left += 1
    right -= 1
\`\`\`

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(1)`,
			hint1: `Ikki uchidan boshlanadigan ikki ko'rsatkichdan foydalaning. Solishtirishdan oldin harf-raqamli bo'lmagan belgilarni o'tkazib yuborish uchun isalnum() bilan while tsikllaridan foydalaning.`,
			hint2: `Python str.isalnum() harflar va raqamlar uchun True qaytaradi. Registrni hisobga olmaydigan solishtirish uchun str.lower() dan foydalaning.`,
			whyItMatters: `Palindromlar uchun ikki ko'rsatkich texnikasi ko'plab satr masalalari uchun asos.

**Bu nima uchun muhim:**

**1. Ikki ko'rsatkich patterni**

Bu pattern hamma joyda uchraydi:
- Container with Most Water
- Trapping Rain Water
- 3Sum, 4Sum masalalar
- Bog'langan ro'yxatda tsiklni aniqlash

**2. Python satr metodlari**

O'rnatilgan metodlarni tushunish vaqtni tejaydi:
\`\`\`python
# Belgi tekshiruvi
'a'.isalnum()  # True - harf-raqamli
'A'.isalpha()  # True - faqat harf
'5'.isdigit()  # True - faqat raqam

# Registr o'zgartirish
'ABC'.lower()  # 'abc'
'abc'.upper()  # 'ABC'
\`\`\`

**3. Xotira samaradorligi**

Tozalangan satr yaratish uchun O(n) o'rniga O(1) xotira:
\`\`\`python
# O(n) xotira - yangi satr yaratadi
cleaned = ''.join(c.lower() for c in s if c.isalnum())
return cleaned == cleaned[::-1]

# O(1) xotira - ikki ko'rsatkich
return is_palindrome(s)  # joyida tekshirish
\`\`\``,
			solutionCode: `def is_palindrome(s: str) -> bool:
    """
    Satr palindrom ekanligini tekshiradi.
    Faqat harf-raqamli belgilarni hisobga oladi, registrni hisobga olmaydi.

    Args:
        s: Kirish satri

    Returns:
        Agar s palindrom bo'lsa True, aks holda False
    """
    left, right = 0, len(s) - 1

    while left < right:
        # Chapdan harf-raqamli bo'lmaganlarni o'tkazamiz
        while left < right and not s[left].isalnum():
            left += 1
        # O'ngdan harf-raqamli bo'lmaganlarni o'tkazamiz
        while left < right and not s[right].isalnum():
            right -= 1

        # Belgilarni solishtiramiz (registrni hisobga olmay)
        if s[left].lower() != s[right].lower():
            return False

        left += 1
        right -= 1

    return True`
		}
	}
};

export default task;
