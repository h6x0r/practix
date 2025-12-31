import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'graph-course-schedule',
	title: 'Course Schedule',
	difficulty: 'medium',
	tags: ['python', 'graphs', 'topological-sort', 'dfs', 'bfs', 'cycle-detection'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Determine if you can finish all courses given their prerequisites.

**Problem:**

There are \`numCourses\` courses you have to take, labeled from \`0\` to \`numCourses - 1\`. You are given an array \`prerequisites\` where \`prerequisites[i] = [ai, bi]\` indicates that you must take course \`bi\` before course \`ai\`.

Return \`true\` if you can finish all courses. Otherwise, return \`false\`.

**Examples:**

\`\`\`
Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: Take course 0 first, then course 1.

Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false
Explanation: To take course 1, you need course 0.
             To take course 0, you need course 1.
             This is a cycle - impossible!

Input: numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
Output: true
Explanation: Take 0 → 1 → 2 → 3 (or 0 → 2 → 1 → 3)
\`\`\`

**Visualization:**

\`\`\`
Example with cycle (impossible):
    0 ←→ 1     (0 needs 1, 1 needs 0)

Example without cycle (possible):
    0 → 1
    ↓   ↓
    2 → 3

Valid order: 0, 1, 2, 3 or 0, 2, 1, 3
\`\`\`

**Key Insight:**

This problem is about **cycle detection** in a directed graph. If there's a cycle in the prerequisite graph, it's impossible to complete all courses.

Methods:
1. **DFS with coloring** (WHITE → GRAY → BLACK)
2. **BFS with indegree** (Kahn's algorithm for topological sort)

**Constraints:**
- 1 <= numCourses <= 2000
- 0 <= prerequisites.length <= 5000
- prerequisites[i].length == 2
- 0 <= ai, bi < numCourses
- All prerequisite pairs are unique

**Time Complexity:** O(V + E)
**Space Complexity:** O(V + E)`,
	initialCode: `from typing import List

def can_finish(numCourses: int, prerequisites: List[List[int]]) -> bool:
    # TODO: Determine if all courses can be finished (detect cycle in prerequisites)

    return True`,
	solutionCode: `from typing import List
from collections import defaultdict, deque

def can_finish(numCourses: int, prerequisites: List[List[int]]) -> bool:
    """
    Determine if all courses can be finished.

    Args:
        numCourses: Total number of courses (0 to numCourses-1)
        prerequisites: List of [course, prerequisite] pairs

    Returns:
        True if all courses can be finished, False otherwise
    """
    # Build adjacency list
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)

    # DFS with three states: 0=unvisited, 1=visiting, 2=visited
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * numCourses

    def has_cycle(node: int) -> bool:
        color[node] = GRAY  # Mark as visiting

        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                return True  # Back edge found = cycle!
            if color[neighbor] == WHITE:
                if has_cycle(neighbor):
                    return True

        color[node] = BLACK  # Mark as fully visited
        return False

    # Check each unvisited node
    for course in range(numCourses):
        if color[course] == WHITE:
            if has_cycle(course):
                return False

    return True


# Alternative: BFS with indegree (Kahn's algorithm)
def can_finish_bfs(numCourses: int, prerequisites: List[List[int]]) -> bool:
    """Topological sort using BFS (Kahn's algorithm)."""
    graph = defaultdict(list)
    indegree = [0] * numCourses

    # Build graph and count indegrees
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        indegree[course] += 1

    # Start with nodes that have no prerequisites
    queue = deque([i for i in range(numCourses) if indegree[i] == 0])
    processed = 0

    while queue:
        node = queue.popleft()
        processed += 1

        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    # If all courses processed, no cycle exists
    return processed == numCourses


# Course Schedule II: Return the order
def find_order(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    """Return valid course order, or empty list if impossible."""
    graph = defaultdict(list)
    indegree = [0] * numCourses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        indegree[course] += 1

    queue = deque([i for i in range(numCourses) if indegree[i] == 0])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)

        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    return order if len(order) == numCourses else []`,
	testCode: `import pytest
from solution import can_finish


class TestCanFinish:
    def test_simple_chain(self):
        """Test simple prerequisite chain"""
        assert can_finish(2, [[1, 0]]) == True

    def test_simple_cycle(self):
        """Test simple cycle (impossible)"""
        assert can_finish(2, [[1, 0], [0, 1]]) == False

    def test_complex_dag(self):
        """Test complex DAG without cycle"""
        assert can_finish(4, [[1, 0], [2, 0], [3, 1], [3, 2]]) == True

    def test_no_prerequisites(self):
        """Test when no prerequisites exist"""
        assert can_finish(3, []) == True

    def test_single_course(self):
        """Test single course with no prerequisites"""
        assert can_finish(1, []) == True

    def test_self_loop(self):
        """Test self-dependency (implicit cycle)"""
        assert can_finish(1, [[0, 0]]) == False

    def test_longer_cycle(self):
        """Test longer cycle"""
        assert can_finish(3, [[0, 1], [1, 2], [2, 0]]) == False

    def test_multiple_components(self):
        """Test disconnected components"""
        assert can_finish(4, [[1, 0], [3, 2]]) == True

    def test_multiple_prerequisites(self):
        """Test course with multiple prerequisites"""
        assert can_finish(4, [[3, 0], [3, 1], [3, 2]]) == True

    def test_large_dag(self):
        """Test larger DAG"""
        # Linear chain: 0 → 1 → 2 → ... → 9
        prereqs = [[i + 1, i] for i in range(9)]
        assert can_finish(10, prereqs) == True`,
	hint1: `Build a directed graph from prerequisites. If course A requires course B, add edge B → A. Then detect if there's a cycle.`,
	hint2: `Use DFS with three colors: WHITE (unvisited), GRAY (in current path), BLACK (done). If you visit a GRAY node, there's a cycle. Or use Kahn's algorithm with indegree.`,
	whyItMatters: `Course Schedule teaches cycle detection and topological sorting - fundamental graph algorithms used in build systems, task scheduling, and dependency resolution.

**Why This Matters:**

**1. Topological Sort Applications**

\`\`\`python
# Build systems (Makefile, Gradle)
# - Compile dependencies before main code

# Package managers (npm, pip)
# - Install dependencies in correct order

# Task schedulers
# - Execute tasks respecting dependencies

# Spreadsheet cell evaluation
# - Evaluate formula cells in correct order
\`\`\`

**2. Cycle Detection Techniques**

\`\`\`python
# Method 1: DFS with colors (recommended)
def has_cycle_dfs(graph, n):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n

    def dfs(node):
        color[node] = GRAY
        for neighbor in graph[node]:
            if color[neighbor] == GRAY:  # Back edge!
                return True
            if color[neighbor] == WHITE and dfs(neighbor):
                return True
        color[node] = BLACK
        return False

    return any(color[i] == WHITE and dfs(i) for i in range(n))

# Method 2: Kahn's algorithm
# If topological sort doesn't include all nodes, cycle exists
\`\`\`

**3. Kahn's Algorithm (BFS Topological Sort)**

\`\`\`python
def topological_sort_kahn(graph, n):
    indegree = [0] * n
    for node in graph:
        for neighbor in graph[node]:
            indegree[neighbor] += 1

    queue = deque([i for i in range(n) if indegree[i] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    return result if len(result) == n else []  # Empty = cycle
\`\`\`

**4. DFS Topological Sort**

\`\`\`python
def topological_sort_dfs(graph, n):
    visited = [False] * n
    result = []

    def dfs(node):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs(neighbor)
        result.append(node)  # Add after exploring all descendants

    for i in range(n):
        if not visited[i]:
            dfs(i)

    return result[::-1]  # Reverse for topological order
\`\`\`

**5. Course Schedule II: Find Valid Order**

\`\`\`python
# Extension: return actual order, not just boolean
# Use Kahn's algorithm and collect nodes in queue order
# Or use DFS and reverse the post-order
\`\`\``,
	order: 4,
	translations: {
		ru: {
			title: 'Расписание курсов',
			description: `Определите, можно ли пройти все курсы с учётом пререквизитов.

**Задача:**

Есть \`numCourses\` курсов с номерами от 0 до numCourses-1. Дан массив \`prerequisites\`, где \`prerequisites[i] = [ai, bi]\` означает, что для прохождения курса \`ai\` нужно сначала пройти курс \`bi\`.

Верните \`true\`, если можно пройти все курсы, иначе \`false\`.

**Примеры:**

\`\`\`
Вход: numCourses = 2, prerequisites = [[1,0]]
Выход: true
Объяснение: Сначала курс 0, затем курс 1.

Вход: numCourses = 2, prerequisites = [[1,0],[0,1]]
Выход: false
Объяснение: Для курса 1 нужен курс 0.
             Для курса 0 нужен курс 1.
             Это цикл - невозможно!
\`\`\`

**Ключевая идея:**

Задача сводится к **обнаружению цикла** в ориентированном графе. Если в графе пререквизитов есть цикл, невозможно завершить все курсы.

**Ограничения:**
- 1 <= numCourses <= 2000
- 0 <= prerequisites.length <= 5000

**Временная сложность:** O(V + E)
**Пространственная сложность:** O(V + E)`,
			hint1: `Постройте ориентированный граф из пререквизитов. Если курс A требует курс B, добавьте ребро B → A. Затем найдите цикл.`,
			hint2: `Используйте DFS с тремя цветами: WHITE (не посещён), GRAY (в текущем пути), BLACK (завершён). Если встречаем GRAY вершину - есть цикл.`,
			whyItMatters: `Расписание курсов учит обнаружению циклов и топологической сортировке - фундаментальным алгоритмам, используемым в системах сборки и планировании задач.

**Почему это важно:**

**1. Применения топологической сортировки**

- Системы сборки (Makefile, Gradle)
- Менеджеры пакетов (npm, pip)
- Планировщики задач

**2. Техники обнаружения циклов**

DFS с тремя цветами или алгоритм Кана.

**3. Алгоритм Кана (BFS)**

Если топологическая сортировка не включает все вершины - есть цикл.`,
			solutionCode: `from typing import List
from collections import defaultdict

def can_finish(numCourses: int, prerequisites: List[List[int]]) -> bool:
    """
    Определяет, можно ли пройти все курсы.

    Args:
        numCourses: Общее количество курсов
        prerequisites: Список пар [курс, пререквизит]

    Returns:
        True если можно пройти все курсы, иначе False
    """
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * numCourses

    def has_cycle(node: int) -> bool:
        color[node] = GRAY

        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                return True
            if color[neighbor] == WHITE:
                if has_cycle(neighbor):
                    return True

        color[node] = BLACK
        return False

    for course in range(numCourses):
        if color[course] == WHITE:
            if has_cycle(course):
                return False

    return True`
		},
		uz: {
			title: 'Kurslar jadvali',
			description: `Barcha kurslarni tugatish mumkinligini aniqlang.

**Masala:**

0 dan numCourses-1 gacha raqamlangan \`numCourses\` ta kurs bor. \`prerequisites\` massivi berilgan, bu yerda \`prerequisites[i] = [ai, bi]\` \`ai\` kursini olish uchun avval \`bi\` kursini tugatish kerakligini bildiradi.

Agar barcha kurslarni tugatish mumkin bo'lsa \`true\`, aks holda \`false\` qaytaring.

**Misollar:**

\`\`\`
Kirish: numCourses = 2, prerequisites = [[1,0]]
Chiqish: true
Izoh: Avval 0-kurs, keyin 1-kurs.

Kirish: numCourses = 2, prerequisites = [[1,0],[0,1]]
Chiqish: false
Izoh: 1-kurs uchun 0-kurs kerak.
      0-kurs uchun 1-kurs kerak.
      Bu sikl - imkonsiz!
\`\`\`

**Asosiy tushuncha:**

Masala yo'naltirilgan grafda **sikl aniqlash**ga keladi. Agar prerekvizitlar grafida sikl bo'lsa, barcha kurslarni tugatish imkonsiz.

**Cheklovlar:**
- 1 <= numCourses <= 2000
- 0 <= prerequisites.length <= 5000

**Vaqt murakkabligi:** O(V + E)
**Xotira murakkabligi:** O(V + E)`,
			hint1: `Prerekvizitlardan yo'naltirilgan graf tuzing. Agar A kurs B kursni talab qilsa, B → A qirra qo'shing. Keyin sikl toping.`,
			hint2: `Uch rang bilan DFS ishlating: WHITE (tashrif buyurilmagan), GRAY (joriy yo'lda), BLACK (tugallangan). GRAY tugun uchrasak - sikl bor.`,
			whyItMatters: `Kurslar jadvali sikl aniqlash va topologik saralashni o'rgatadi - bu build tizimlari va vazifalarni rejalashtirishda ishlatiladigan asosiy algoritmlar.

**Bu nima uchun muhim:**

**1. Topologik saralash qo'llanilishi**

- Build tizimlari (Makefile, Gradle)
- Paket menejerlari (npm, pip)
- Vazifa rejalashtirgichlari

**2. Sikl aniqlash texnikalari**

Uch rang bilan DFS yoki Kahn algoritmi.

**3. Kahn algoritmi (BFS)**

Agar topologik saralash barcha tugunlarni o'z ichiga olmasa - sikl bor.`,
			solutionCode: `from typing import List
from collections import defaultdict

def can_finish(numCourses: int, prerequisites: List[List[int]]) -> bool:
    """
    Barcha kurslarni tugatish mumkinligini aniqlaydi.

    Args:
        numCourses: Kurslarning umumiy soni
        prerequisites: [kurs, prerekvizit] juftliklari ro'yxati

    Returns:
        Barcha kurslarni tugatish mumkin bo'lsa True, aks holda False
    """
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * numCourses

    def has_cycle(node: int) -> bool:
        color[node] = GRAY

        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                return True
            if color[neighbor] == WHITE:
                if has_cycle(neighbor):
                    return True

        color[node] = BLACK
        return False

    for course in range(numCourses):
        if color[course] == WHITE:
            if has_cycle(course):
                return False

    return True`
		}
	}
};

export default task;
