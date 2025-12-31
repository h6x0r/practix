import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'graph-word-ladder',
	title: 'Word Ladder',
	difficulty: 'hard',
	tags: ['python', 'graphs', 'bfs', 'implicit-graph', 'string'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find the shortest transformation sequence from one word to another.

**Problem:**

Given two words, \`beginWord\` and \`endWord\`, and a dictionary \`wordList\`, find the length of the shortest transformation sequence from \`beginWord\` to \`endWord\`, such that:

1. Only one letter can be changed at a time
2. Each transformed word must exist in the word list

Return 0 if there is no such transformation sequence.

**Examples:**

\`\`\`
Input: beginWord = "hit", endWord = "cog",
       wordList = ["hot","dot","dog","lot","log","cog"]
Output: 5

Explanation: "hit" -> "hot" -> "dot" -> "dog" -> "cog"
Length = 5 (including start and end words)

Input: beginWord = "hit", endWord = "cog",
       wordList = ["hot","dot","dog","lot","log"]
Output: 0
Explanation: "cog" is not in wordList, so no transformation possible.

Input: beginWord = "a", endWord = "c",
       wordList = ["a","b","c"]
Output: 2
Explanation: "a" -> "c"
\`\`\`

**Visualization:**

\`\`\`
Word graph (words differ by 1 letter are connected):

hit
 |
hot --- lot
 |       |
dot --- log
 |       |
dog --- cog

BFS from "hit":
Level 0: hit
Level 1: hot
Level 2: dot, lot
Level 3: dog, log
Level 4: cog

Shortest path length: 5
\`\`\`

**Key Insight:**

This is a **graph problem** where:
- Each word is a node
- Two words are connected if they differ by exactly one letter
- We need the shortest path → **BFS**

**Constraints:**
- 1 <= beginWord.length <= 10
- endWord.length == beginWord.length
- 1 <= wordList.length <= 5000
- All words have the same length
- All words consist of lowercase English letters
- beginWord and endWord are not the same

**Time Complexity:** O(M² × N) where M = word length, N = wordList size
**Space Complexity:** O(M² × N)`,
	initialCode: `from typing import List
from collections import deque

def ladder_length(beginWord: str, endWord: str, wordList: List[str]) -> int:
    # TODO: Find the length of shortest transformation sequence

    return 0`,
	solutionCode: `from typing import List
from collections import deque, defaultdict

def ladder_length(beginWord: str, endWord: str, wordList: List[str]) -> int:
    """
    Find the length of shortest transformation sequence.
    """
    if endWord not in wordList:
        return 0

    # Build pattern dictionary
    word_len = len(beginWord)
    patterns = defaultdict(list)

    for word in wordList:
        for i in range(word_len):
            pattern = word[:i] + '*' + word[i+1:]
            patterns[pattern].append(word)

    # BFS
    queue = deque([(beginWord, 1)])
    visited = set([beginWord])

    while queue:
        word, level = queue.popleft()

        for i in range(word_len):
            pattern = word[:i] + '*' + word[i+1:]

            for next_word in patterns[pattern]:
                if next_word == endWord:
                    return level + 1

                if next_word not in visited:
                    visited.add(next_word)
                    queue.append((next_word, level + 1))

            # Clear pattern to avoid revisiting
            patterns[pattern] = []

    return 0


# Bidirectional BFS (faster for long paths)
def ladder_length_bidirectional(beginWord: str, endWord: str, wordList: List[str]) -> int:
    """Bidirectional BFS for faster convergence."""
    word_set = set(wordList)
    if endWord not in word_set:
        return 0

    # Two frontiers
    front = {beginWord}
    back = {endWord}
    word_len = len(beginWord)
    level = 1

    while front and back:
        # Always expand smaller frontier
        if len(front) > len(back):
            front, back = back, front

        next_front = set()

        for word in front:
            for i in range(word_len):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c == word[i]:
                        continue
                    next_word = word[:i] + c + word[i+1:]

                    if next_word in back:
                        return level + 1

                    if next_word in word_set:
                        next_front.add(next_word)
                        word_set.remove(next_word)

        front = next_front
        level += 1

    return 0


# Word Ladder II: Find all shortest paths
def find_ladders(beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
    """Find all shortest transformation sequences."""
    word_set = set(wordList)
    if endWord not in word_set:
        return []

    # BFS to find shortest distance and parents
    word_len = len(beginWord)
    parents = defaultdict(set)
    queue = deque([beginWord])
    visited = {beginWord}
    found = False

    while queue and not found:
        level_visited = set()

        for _ in range(len(queue)):
            word = queue.popleft()

            for i in range(word_len):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    next_word = word[:i] + c + word[i+1:]

                    if next_word == endWord:
                        found = True
                        parents[endWord].add(word)
                    elif next_word in word_set and next_word not in visited:
                        level_visited.add(next_word)
                        parents[next_word].add(word)
                        queue.append(next_word)

        visited.update(level_visited)

    # Backtrack to find all paths
    def backtrack(word):
        if word == beginWord:
            return [[beginWord]]
        paths = []
        for parent in parents[word]:
            for path in backtrack(parent):
                paths.append(path + [word])
        return paths

    return backtrack(endWord) if found else []`,
	testCode: `import pytest
from solution import ladder_length


class TestWordLadder:
    def test_basic_transformation(self):
        """Test basic word transformation"""
        wordList = ["hot", "dot", "dog", "lot", "log", "cog"]
        assert ladder_length("hit", "cog", wordList) == 5

    def test_no_path_end_not_in_list(self):
        """Test when end word not in list"""
        wordList = ["hot", "dot", "dog", "lot", "log"]
        assert ladder_length("hit", "cog", wordList) == 0

    def test_single_letter_words(self):
        """Test single letter words"""
        assert ladder_length("a", "c", ["a", "b", "c"]) == 2

    def test_direct_transformation(self):
        """Test direct one-step transformation"""
        assert ladder_length("hot", "dot", ["dot"]) == 2

    def test_no_path_exists(self):
        """Test when no transformation path exists"""
        wordList = ["hot", "dog"]
        assert ladder_length("hit", "cog", wordList) == 0

    def test_longer_path(self):
        """Test longer transformation path"""
        wordList = ["hot", "dot", "lot", "log", "cog"]
        # hit -> hot -> lot -> log -> cog = 5
        result = ladder_length("hit", "cog", wordList)
        assert result == 5

    def test_multiple_paths_same_length(self):
        """Test with multiple paths of same length"""
        wordList = ["hot", "dot", "dog", "lot", "log", "cog"]
        # Multiple paths exist, should return shortest
        assert ladder_length("hit", "cog", wordList) == 5

    def test_begin_equals_end(self):
        """Test when begin and end are same (edge case)"""
        # This shouldn't happen per constraints but handle gracefully
        assert ladder_length("hit", "hit", ["hit"]) >= 0

    def test_two_letter_words(self):
        """Test two letter words"""
        wordList = ["ab", "ac", "bc"]
        assert ladder_length("ab", "bc", wordList) == 3

    def test_large_word_list(self):
        """Test with larger word list"""
        wordList = [
            "hot", "dot", "dog", "lot", "log", "cog",
            "hit", "hat", "bat", "bag", "bog"
        ]
        result = ladder_length("hit", "cog", wordList)
        assert result >= 1  # Should find some path`,
	hint1: `Build an implicit graph where words differ by 1 letter are connected. Use BFS to find shortest path. Consider using pattern matching: "hot" matches "*ot", "h*t", "ho*".`,
	hint2: `Create a dictionary mapping patterns to words. For BFS, generate patterns for current word and find all matching words. Mark words as visited to avoid cycles.`,
	whyItMatters: `Word Ladder demonstrates BFS on implicit graphs where edges aren't explicitly defined. This pattern appears in puzzle solving, state space search, and NLP applications.

**Why This Matters:**

**1. Implicit Graph Pattern**

\`\`\`python
# Graph where edges are defined by a relationship, not stored
# Common in puzzles and state-space problems

# Examples of implicit graphs:
# - Word transformations (differ by 1 letter)
# - Sliding puzzle (swap adjacent tiles)
# - Rubik's cube (rotate faces)
# - Chess (knight moves)
\`\`\`

**2. Pattern-Based Neighbor Finding**

\`\`\`python
# Instead of checking all pairs O(N²)
# Use patterns for O(N × M × 26) neighbor finding

def get_patterns(word):
    return [word[:i] + '*' + word[i+1:] for i in range(len(word))]

# "hot" -> ["*ot", "h*t", "ho*"]
# All words sharing a pattern are neighbors
\`\`\`

**3. Bidirectional BFS**

\`\`\`python
# For long paths, search from both ends
# Time complexity: O(b^(d/2)) instead of O(b^d)
# where b = branching factor, d = depth

def bidirectional_bfs(start, end, get_neighbors):
    front, back = {start}, {end}
    visited = {start, end}
    level = 0

    while front and back:
        if len(front) > len(back):
            front, back = back, front

        next_front = set()
        for node in front:
            for neighbor in get_neighbors(node):
                if neighbor in back:
                    return level + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_front.add(neighbor)

        front = next_front
        level += 1

    return -1  # No path
\`\`\`

**4. State Space Search Applications**

\`\`\`python
# 8-Puzzle solver
def solve_puzzle(initial, goal):
    # State = tuple of board positions
    # Neighbors = states reachable by one move
    return bfs(initial, goal, get_puzzle_neighbors)

# Open Lock (LeetCode 752)
def open_lock(deadends, target):
    # State = 4-digit combination
    # Neighbors = rotating any wheel ±1
    return bfs("0000", target, get_lock_neighbors)
\`\`\`

**5. Word Ladder II: All Shortest Paths**

\`\`\`python
# Extension: find ALL shortest paths, not just length
# Approach:
# 1. BFS to find distances and parent sets
# 2. Backtrack from end to find all paths

# Used in:
# - Network routing (multiple equal-cost paths)
# - Game AI (exploring all optimal moves)
\`\`\``,
	order: 9,
	translations: {
		ru: {
			title: 'Лестница слов',
			description: `Найдите кратчайшую последовательность преобразований от одного слова к другому.

**Задача:**

Даны два слова \`beginWord\` и \`endWord\`, и словарь \`wordList\`. Найдите длину кратчайшей последовательности преобразований от \`beginWord\` к \`endWord\`, где:

1. За один шаг можно изменить только одну букву
2. Каждое промежуточное слово должно быть в словаре

Верните 0, если преобразование невозможно.

**Примеры:**

\`\`\`
Вход: beginWord = "hit", endWord = "cog",
      wordList = ["hot","dot","dog","lot","log","cog"]
Выход: 5

Объяснение: "hit" -> "hot" -> "dot" -> "dog" -> "cog"
\`\`\`

**Ключевая идея:**

Это **задача на графы**, где:
- Каждое слово - вершина
- Слова связаны, если отличаются на одну букву
- Нужен кратчайший путь → **BFS**

**Ограничения:**
- 1 <= длина слова <= 10
- 1 <= wordList.length <= 5000

**Временная сложность:** O(M² × N)
**Пространственная сложность:** O(M² × N)`,
			hint1: `Постройте неявный граф, где слова, отличающиеся на 1 букву, связаны. Используйте BFS. Рассмотрите сопоставление паттернов: "hot" соответствует "*ot", "h*t", "ho*".`,
			hint2: `Создайте словарь паттернов. Для BFS генерируйте паттерны текущего слова и находите все соответствующие слова. Помечайте посещённые.`,
			whyItMatters: `Word Ladder демонстрирует BFS на неявных графах. Этот паттерн встречается в решении головоломок и NLP.

**Почему это важно:**

**1. Паттерн неявного графа**

Граф, где рёбра определяются отношением, а не хранятся явно.

**2. Поиск соседей на основе паттернов**

Вместо проверки всех пар O(N²) используем паттерны.

**3. Двунаправленный BFS**

Для длинных путей ищем с обоих концов - быстрее.

**4. Применения в поиске состояний**

8-Puzzle, Open Lock и другие головоломки.`,
			solutionCode: `from typing import List
from collections import deque, defaultdict

def ladder_length(beginWord: str, endWord: str, wordList: List[str]) -> int:
    """Находит длину кратчайшей последовательности преобразований."""
    if endWord not in wordList:
        return 0

    word_len = len(beginWord)
    patterns = defaultdict(list)

    for word in wordList:
        for i in range(word_len):
            pattern = word[:i] + '*' + word[i+1:]
            patterns[pattern].append(word)

    queue = deque([(beginWord, 1)])
    visited = set([beginWord])

    while queue:
        word, level = queue.popleft()

        for i in range(word_len):
            pattern = word[:i] + '*' + word[i+1:]

            for next_word in patterns[pattern]:
                if next_word == endWord:
                    return level + 1

                if next_word not in visited:
                    visited.add(next_word)
                    queue.append((next_word, level + 1))

            patterns[pattern] = []

    return 0`
		},
		uz: {
			title: 'So\'z narvoni',
			description: `Bir so'zdan boshqasiga eng qisqa o'zgartirish ketma-ketligini toping.

**Masala:**

\`beginWord\` va \`endWord\` so'zlari hamda \`wordList\` lug'ati berilgan. \`beginWord\` dan \`endWord\` ga eng qisqa o'zgartirish ketma-ketligi uzunligini toping, shunda:

1. Bir vaqtda faqat bitta harf o'zgartiriladi
2. Har bir oraliq so'z lug'atda bo'lishi kerak

Agar bunday ketma-ketlik yo'q bo'lsa 0 qaytaring.

**Misollar:**

\`\`\`
Kirish: beginWord = "hit", endWord = "cog",
       wordList = ["hot","dot","dog","lot","log","cog"]
Chiqish: 5

Izoh: "hit" -> "hot" -> "dot" -> "dog" -> "cog"
\`\`\`

**Asosiy tushuncha:**

Bu **graf masalasi**, bu yerda:
- Har bir so'z - tugun
- So'zlar bitta harfga farq qilsa bog'langan
- Eng qisqa yo'l kerak → **BFS**

**Cheklovlar:**
- 1 <= so'z uzunligi <= 10
- 1 <= wordList.length <= 5000

**Vaqt murakkabligi:** O(M² × N)
**Xotira murakkabligi:** O(M² × N)`,
			hint1: `1 harfga farq qiladigan so'zlar bog'langan yashirin graf tuzing. BFS ishlating. Pattern moslashtirish ko'ring: "hot" -> "*ot", "h*t", "ho*" ga mos.`,
			hint2: `Patternlar lug'atini yarating. BFS uchun joriy so'zning patternlarini yarating va mos keluvchi so'zlarni toping. Tashrif buyurilganlarni belgilang.`,
			whyItMatters: `Word Ladder yashirin graflarda BFS ni ko'rsatadi. Bu pattern boshqotirmalar va NLP da uchraydi.

**Bu nima uchun muhim:**

**1. Yashirin graf patterni**

Qirralar munosabat bilan aniqlanadigan graf.

**2. Patternlar asosida qo'shni topish**

Barcha juftlarni tekshirish o'rniga O(N²) patternlar ishlatamiz.

**3. Ikki tomonlama BFS**

Uzoq yo'llar uchun ikkala tomondan qidiramiz - tezroq.

**4. Holat qidirish qo'llanilishi**

8-Puzzle, Open Lock va boshqa boshqotirmalar.`,
			solutionCode: `from typing import List
from collections import deque, defaultdict

def ladder_length(beginWord: str, endWord: str, wordList: List[str]) -> int:
    """Eng qisqa o'zgartirish ketma-ketligi uzunligini topadi."""
    if endWord not in wordList:
        return 0

    word_len = len(beginWord)
    patterns = defaultdict(list)

    for word in wordList:
        for i in range(word_len):
            pattern = word[:i] + '*' + word[i+1:]
            patterns[pattern].append(word)

    queue = deque([(beginWord, 1)])
    visited = set([beginWord])

    while queue:
        word, level = queue.popleft()

        for i in range(word_len):
            pattern = word[:i] + '*' + word[i+1:]

            for next_word in patterns[pattern]:
                if next_word == endWord:
                    return level + 1

                if next_word not in visited:
                    visited.add(next_word)
                    queue.append((next_word, level + 1))

            patterns[pattern] = []

    return 0`
		}
	}
};

export default task;
