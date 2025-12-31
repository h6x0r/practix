import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'algo-longest-common-prefix',
	title: 'Longest Common Prefix',
	difficulty: 'easy',
	tags: ['python', 'strings', 'comparison'],
	estimatedTime: '15m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find the longest common prefix among an array of strings.

**Problem:**

Given an array of strings \`strs\`, find the longest common prefix string amongst all strings.

If there is no common prefix, return an empty string \`""\`.

**Examples:**

\`\`\`
Input: strs = ["flower", "flow", "flight"]
Output: "fl"

Input: strs = ["dog", "racecar", "car"]
Output: ""
Explanation: No common prefix exists

Input: strs = ["interspecies", "interstellar", "interstate"]
Output: "inters"
\`\`\`

**Vertical Scanning Approach:**

Compare characters column by column:
1. Take first string as reference
2. For each character position, compare across all strings
3. Stop when mismatch found or end of any string reached

**Time Complexity:** O(S) where S is sum of all characters
**Space Complexity:** O(1)`,
	initialCode: `from typing import List

def longest_common_prefix(strs: List[str]) -> str:
    # TODO: Find the longest common prefix among an array of strings

    return ""`,
	solutionCode: `from typing import List

def longest_common_prefix(strs: List[str]) -> str:
    """
    Find the longest common prefix among an array of strings.

    Args:
        strs: List of strings

    Returns:
        The longest common prefix string
    """
    # Edge cases
    if not strs:
        return ""
    if len(strs) == 1:
        return strs[0]

    # Use first string as reference
    prefix = strs[0]

    # Compare with each other string
    for s in strs[1:]:
        # Shorten prefix until it matches
        while prefix and not s.startswith(prefix):
            prefix = prefix[:-1]
        if not prefix:
            return ""

    return prefix`,
	testCode: `import pytest
from solution import longest_common_prefix

class TestLongestCommonPrefix:
    def test_basic(self):
        """Test basic case with common prefix"""
        assert longest_common_prefix(["flower", "flow", "flight"]) == "fl"

    def test_no_common(self):
        """Test no common prefix"""
        assert longest_common_prefix(["dog", "racecar", "car"]) == ""

    def test_full_match(self):
        """Test all strings identical"""
        assert longest_common_prefix(["test", "test", "test"]) == "test"

    def test_single(self):
        """Test single string in array"""
        assert longest_common_prefix(["alone"]) == "alone"

    def test_empty_array(self):
        """Test empty array"""
        assert longest_common_prefix([]) == ""

    def test_empty_string(self):
        """Test array with empty string"""
        assert longest_common_prefix(["", "abc"]) == ""

    def test_inter(self):
        """Test longer common prefix"""
        assert longest_common_prefix(["interspecies", "interstellar", "interstate"]) == "inters"

    def test_single_char(self):
        """Test single character prefix"""
        assert longest_common_prefix(["a", "ab", "abc"]) == "a"

    def test_two_strings(self):
        """Test with two strings"""
        assert longest_common_prefix(["hello", "help"]) == "hel"

    def test_first_string_shortest(self):
        """Test when first string is shortest"""
        assert longest_common_prefix(["ab", "abc", "abcd"]) == "ab"`,
	hint1: `Start with the first string as your prefix. Then compare it with each subsequent string, shortening the prefix whenever you find a mismatch.`,
	hint2: `For each comparison, check if the current string starts with the prefix. If not, remove the last character from the prefix and try again.`,
	whyItMatters: `String prefix matching is fundamental to many applications.

**Why This Matters:**

**1. Autocomplete Systems**

When you type in a search bar:
\`\`\`python
suggestions = ["flower", "flow", "florida"]
prefix = longest_common_prefix(suggestions)
# prefix = "flo" - shows this as autocomplete hint
\`\`\`

**2. Trie Data Structure**

Common prefix problems lead to Trie:
\`\`\`python
# Trie stores strings by shared prefixes
#      root
#     /    \\
#    f      d
#   /        \\
#  l         o
# / \\         \\
#o   i         g
\`\`\`

**3. File System Paths**

Finding common directory:
\`\`\`python
paths = [
    "/home/user/docs/file1.txt",
    "/home/user/docs/file2.txt",
    "/home/user/images/photo.jpg",
]
# Common prefix: "/home/user/"
\`\`\`

**4. URL Routing**

Web frameworks use prefix matching:
\`\`\`
/api/users/123     -> matches /api/users/:id
/api/users/profile -> matches /api/users/profile
# Common prefix: /api/users/
\`\`\``,
	order: 3,
	translations: {
		ru: {
			title: 'Наибольший общий префикс',
			description: `Найдите наибольший общий префикс среди массива строк.

**Задача:**

Дан массив строк \`strs\`, найдите наибольшую общую строку-префикс среди всех строк.

Если общего префикса нет, верните пустую строку \`""\`.

**Примеры:**

\`\`\`
Вход: strs = ["flower", "flow", "flight"]
Выход: "fl"

Вход: strs = ["dog", "racecar", "car"]
Выход: ""
Объяснение: Общего префикса не существует

Вход: strs = ["interspecies", "interstellar", "interstate"]
Выход: "inters"
\`\`\`

**Подход вертикального сканирования:**

Сравниваем символы по столбцам:
1. Берём первую строку как эталон
2. Для каждой позиции символа сравниваем по всем строкам
3. Останавливаемся при несовпадении или конце любой строки

**Временная сложность:** O(S), где S - сумма всех символов
**Пространственная сложность:** O(1)`,
			hint1: `Начните с первой строки как вашего префикса. Затем сравните её с каждой последующей строкой, укорачивая префикс при каждом несовпадении.`,
			hint2: `При каждом сравнении проверяйте, начинается ли текущая строка с префикса. Если нет, удалите последний символ из префикса и попробуйте снова.`,
			whyItMatters: `Сопоставление префиксов строк фундаментально для многих приложений.

**Почему это важно:**

**1. Системы автодополнения**

Когда вы печатаете в поисковой строке.

**2. Структура данных Trie**

Задачи на общие префиксы ведут к Trie.

**3. Пути файловой системы**

Нахождение общей директории.`,
			solutionCode: `from typing import List

def longest_common_prefix(strs: List[str]) -> str:
    """
    Находит наибольший общий префикс среди массива строк.

    Args:
        strs: Список строк

    Returns:
        Наибольший общий префикс
    """
    # Граничные случаи
    if not strs:
        return ""
    if len(strs) == 1:
        return strs[0]

    # Используем первую строку как эталон
    prefix = strs[0]

    # Сравниваем с каждой другой строкой
    for s in strs[1:]:
        # Укорачиваем префикс пока не совпадёт
        while prefix and not s.startswith(prefix):
            prefix = prefix[:-1]
        if not prefix:
            return ""

    return prefix`
		},
		uz: {
			title: 'Eng uzun umumiy prefiks',
			description: `Satrlar massivi orasidan eng uzun umumiy prefiksni toping.

**Masala:**

Satrlar massivi \`strs\` berilgan, barcha satrlar orasidagi eng uzun umumiy prefiks satrini toping.

Agar umumiy prefiks bo'lmasa, bo'sh satr \`""\` qaytaring.

**Misollar:**

\`\`\`
Kirish: strs = ["flower", "flow", "flight"]
Chiqish: "fl"

Kirish: strs = ["dog", "racecar", "car"]
Chiqish: ""
Tushuntirish: Umumiy prefiks mavjud emas
\`\`\`

**Vertikal skanerlash yondashuvi:**

Belgilarni ustun bo'yicha solishtiring:
1. Birinchi satrni etalon sifatida oling
2. Har bir belgi pozitsiyasi uchun barcha satrlarda solishtiring
3. Nomuvofiqlik topilganda yoki har qanday satr tugaganda to'xtang

**Vaqt murakkabligi:** O(S), bu yerda S barcha belgilar yig'indisi
**Xotira murakkabligi:** O(1)`,
			hint1: `Birinchi satrni prefiksingiz sifatida boshlang. Keyin har bir keyingi satr bilan solishtiring, nomuvofiqlik topilganda prefiksni qisqartirib.`,
			hint2: `Har bir solishtirishda joriy satr prefiks bilan boshlanishini tekshiring. Agar yo'q bo'lsa, prefiksdan oxirgi belgini olib tashlang va qayta urinib ko'ring.`,
			whyItMatters: `Satr prefikslarini moslashtirish ko'plab ilovalar uchun asosiy.

**Bu nima uchun muhim:**

**1. Avtoto'ldirish tizimlari**

Qidiruv satrida yozayotganingizda.

**2. Trie ma'lumotlar tuzilmasi**

Umumiy prefiks masalalari Trie ga olib keladi.`,
			solutionCode: `from typing import List

def longest_common_prefix(strs: List[str]) -> str:
    """
    Satrlar massivi orasidan eng uzun umumiy prefiksni topadi.

    Args:
        strs: Satrlar ro'yxati

    Returns:
        Eng uzun umumiy prefiks satri
    """
    # Chegara holatlari
    if not strs:
        return ""
    if len(strs) == 1:
        return strs[0]

    # Birinchi satrni etalon sifatida ishlatamiz
    prefix = strs[0]

    # Har bir boshqa satr bilan solishtiramiz
    for s in strs[1:]:
        # Mos kelguncha prefiksni qisqartiramiz
        while prefix and not s.startswith(prefix):
            prefix = prefix[:-1]
        if not prefix:
            return ""

    return prefix`
		}
	}
};

export default task;
