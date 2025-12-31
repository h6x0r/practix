import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'backtracking-n-queens',
	title: 'N-Queens',
	difficulty: 'hard',
	tags: ['python', 'backtracking', 'recursion', 'constraint-satisfaction'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Place N queens on an N×N chessboard so that no two queens attack each other.

**Problem:**

Given an integer \`n\`, return all distinct solutions to the N-Queens puzzle.

Each solution contains a distinct board configuration where 'Q' represents a queen and '.' represents an empty square.

A queen attacks any piece in the same row, column, or diagonal.

**Examples:**

\`\`\`
Input: n = 4
Output: [
  [".Q..",
   "...Q",
   "Q...",
   "..Q."],

  ["..Q.",
   "Q...",
   "...Q",
   ".Q.."]
]

Input: n = 1
Output: [["Q"]]
\`\`\`

**Visualization:**

\`\`\`
n = 4, Solution 1:

. Q . .     Column 0: blocked by Q at (0,1)
. . . Q     Diagonal: \\
Q . . .           and /
. . Q .

Queen positions: (0,1), (1,3), (2,0), (3,2)

Attack patterns for a queen at (r, c):
- Same row: all cells in row r
- Same column: all cells in column c
- Diagonals: (r±k, c±k) for all k
\`\`\`

**Key Insight:**

Use backtracking to place queens row by row:
1. Try placing a queen in each column of the current row
2. Check if the position is valid (not attacked)
3. If valid, move to the next row
4. If no valid position exists, backtrack

**Constraints:**
- 1 <= n <= 9

**Time Complexity:** O(n!) - at most n choices for first row, n-1 for second, etc.
**Space Complexity:** O(n²) for the board`,
	initialCode: `from typing import List

def solve_n_queens(n: int) -> List[List[str]]:
    # TODO: Find all solutions to N-Queens puzzle

    return []`,
	solutionCode: `from typing import List

def solve_n_queens(n: int) -> List[List[str]]:
    """
    Find all solutions to N-Queens puzzle.
    """
    result = []

    # Track attacked positions
    cols = set()      # Columns with queens
    diag1 = set()     # Diagonals (row - col)
    diag2 = set()     # Anti-diagonals (row + col)

    # Board state: queens[row] = column where queen is placed
    queens = [-1] * n

    def is_safe(row: int, col: int) -> bool:
        return col not in cols and \\
               (row - col) not in diag1 and \\
               (row + col) not in diag2

    def place_queen(row: int, col: int) -> None:
        queens[row] = col
        cols.add(col)
        diag1.add(row - col)
        diag2.add(row + col)

    def remove_queen(row: int, col: int) -> None:
        queens[row] = -1
        cols.remove(col)
        diag1.remove(row - col)
        diag2.remove(row + col)

    def build_board() -> List[str]:
        board = []
        for row in range(n):
            line = '.' * queens[row] + 'Q' + '.' * (n - queens[row] - 1)
            board.append(line)
        return board

    def backtrack(row: int) -> None:
        if row == n:
            result.append(build_board())
            return

        for col in range(n):
            if is_safe(row, col):
                place_queen(row, col)
                backtrack(row + 1)
                remove_queen(row, col)

    backtrack(0)
    return result


# Count solutions only (N-Queens II)
def total_n_queens(n: int) -> int:
    """Count number of solutions."""
    cols = set()
    diag1 = set()
    diag2 = set()
    count = 0

    def backtrack(row: int) -> None:
        nonlocal count
        if row == n:
            count += 1
            return

        for col in range(n):
            if col not in cols and (row - col) not in diag1 and (row + col) not in diag2:
                cols.add(col)
                diag1.add(row - col)
                diag2.add(row + col)
                backtrack(row + 1)
                cols.remove(col)
                diag1.remove(row - col)
                diag2.remove(row + col)

    backtrack(0)
    return count


# Using bit manipulation for even faster checking
def solve_n_queens_bits(n: int) -> int:
    """Count solutions using bit manipulation."""
    count = 0

    def backtrack(row: int, cols: int, diag1: int, diag2: int) -> None:
        nonlocal count
        if row == n:
            count += 1
            return

        # available = positions not attacked
        available = ((1 << n) - 1) & ~(cols | diag1 | diag2)

        while available:
            # Get rightmost available position
            pos = available & -available
            available -= pos
            backtrack(row + 1, cols | pos, (diag1 | pos) << 1, (diag2 | pos) >> 1)

    backtrack(0, 0, 0, 0)
    return count`,
	testCode: `import pytest
from solution import solve_n_queens


class TestNQueens:
    def test_n4(self):
        """Test n=4 has 2 solutions"""
        result = solve_n_queens(4)
        assert len(result) == 2

    def test_n1(self):
        """Test n=1 has 1 solution"""
        result = solve_n_queens(1)
        assert result == [["Q"]]

    def test_n8(self):
        """Test n=8 has 92 solutions"""
        result = solve_n_queens(8)
        assert len(result) == 92

    def test_board_size(self):
        """Test each board has correct size"""
        for n in range(1, 6):
            result = solve_n_queens(n)
            for board in result:
                assert len(board) == n
                for row in board:
                    assert len(row) == n

    def test_one_queen_per_row(self):
        """Test exactly one queen per row"""
        result = solve_n_queens(5)
        for board in result:
            for row in board:
                assert row.count('Q') == 1

    def test_no_column_conflict(self):
        """Test no two queens in same column"""
        result = solve_n_queens(5)
        for board in result:
            for col in range(5):
                queen_count = sum(1 for row in board if row[col] == 'Q')
                assert queen_count == 1

    def test_no_diagonal_conflict(self):
        """Test no two queens on same diagonal"""
        result = solve_n_queens(4)
        for board in result:
            queens = []
            for r, row in enumerate(board):
                c = row.index('Q')
                queens.append((r, c))
            # Check all pairs
            for i in range(len(queens)):
                for j in range(i + 1, len(queens)):
                    r1, c1 = queens[i]
                    r2, c2 = queens[j]
                    # Not on same diagonal if |r1-r2| != |c1-c2|
                    assert abs(r1 - r2) != abs(c1 - c2)

    def test_valid_characters(self):
        """Test board contains only Q and ."""
        result = solve_n_queens(4)
        for board in result:
            for row in board:
                for char in row:
                    assert char in ['Q', '.']

    def test_unique_solutions(self):
        """Test all solutions are unique"""
        result = solve_n_queens(6)
        tuples = [tuple(board) for board in result]
        assert len(tuples) == len(set(tuples))

    def test_n5_solution_count(self):
        """Test n=5 has 10 solutions"""
        result = solve_n_queens(5)
        assert len(result) == 10`,
	hint1: `Place queens row by row. For each row, try each column. A position is safe if no queen attacks it from previous rows. Track attacked columns and diagonals.`,
	hint2: `Use three sets: cols (attacked columns), diag1 (row - col, attacked diagonals), diag2 (row + col, attacked anti-diagonals). Position is safe if not in any set.`,
	whyItMatters: `N-Queens is a classic constraint satisfaction problem. It demonstrates backtracking's power in pruning impossible branches early, making exponential problems tractable.

**Why This Matters:**

**1. Constraint Satisfaction Pattern**

\`\`\`python
# General CSP backtracking:
def solve_csp(variables, domains, constraints):
    if all_assigned(variables):
        return is_consistent(variables, constraints)

    var = select_unassigned_variable(variables)
    for value in domains[var]:
        if is_consistent_with(var, value, constraints):
            assign(var, value)
            if solve_csp(variables, domains, constraints):
                return True
            unassign(var)
    return False
\`\`\`

**2. Diagonal Math**

\`\`\`python
# For any diagonal (\\), row - col is constant
# (0,0), (1,1), (2,2) all have row - col = 0

# For any anti-diagonal (/), row + col is constant
# (0,2), (1,1), (2,0) all have row + col = 2

# So we track:
cols = set()          # attacked columns
diag1 = set()         # row - col values
diag2 = set()         # row + col values
\`\`\`

**3. Optimization with Bit Manipulation**

\`\`\`python
def n_queens_bits(n):
    def backtrack(row, cols, diag1, diag2):
        if row == n:
            return 1

        # Available positions = NOT (cols | diag1 | diag2)
        available = ((1 << n) - 1) & ~(cols | diag1 | diag2)
        count = 0

        while available:
            pos = available & -available  # Rightmost bit
            available -= pos
            count += backtrack(
                row + 1,
                cols | pos,
                (diag1 | pos) << 1,   # Shift diagonal
                (diag2 | pos) >> 1    # Shift anti-diagonal
            )
        return count

    return backtrack(0, 0, 0, 0)
\`\`\`

**4. Related Problems**

\`\`\`python
# Sudoku Solver
# Graph Coloring
# Knight's Tour
# Crossword Puzzles

# All follow same pattern:
# - Try an option
# - Check constraints
# - Recurse or backtrack
\`\`\`

**5. Known Solution Counts**

\`\`\`
n:  1  2  3  4  5  6   7   8    9    10
#:  1  0  0  2 10  4  40  92  352   724
\`\`\``,
	order: 4,
	translations: {
		ru: {
			title: 'N ферзей',
			description: `Расставьте N ферзей на доске N×N так, чтобы никакие два ферзя не атаковали друг друга.

**Задача:**

Дано целое число \`n\`. Верните все различные решения головоломки N ферзей.

Каждое решение содержит конфигурацию доски, где 'Q' - ферзь, '.' - пустая клетка.

Ферзь атакует любую фигуру в той же строке, столбце или диагонали.

**Примеры:**

\`\`\`
Вход: n = 4
Выход: [
  [".Q..",
   "...Q",
   "Q...",
   "..Q."],

  ["..Q.",
   "Q...",
   "...Q",
   ".Q.."]
]
\`\`\`

**Ключевая идея:**

Используйте бэктрекинг для расстановки ферзей строка за строкой:
1. Пробуем поставить ферзя в каждый столбец текущей строки
2. Проверяем безопасность позиции (не атакована)
3. Если безопасно, переходим к следующей строке
4. Если нет безопасной позиции, откатываемся

**Ограничения:**
- 1 <= n <= 9

**Временная сложность:** O(n!)
**Пространственная сложность:** O(n²)`,
			hint1: `Ставьте ферзей строка за строкой. Позиция безопасна если ни один ферзь не атакует её из предыдущих строк. Отслеживайте атакованные столбцы и диагонали.`,
			hint2: `Используйте три множества: cols (столбцы), diag1 (row - col, диагонали), diag2 (row + col, антидиагонали). Позиция безопасна если не в любом множестве.`,
			whyItMatters: `N ферзей - классическая задача удовлетворения ограничений. Она демонстрирует мощь бэктрекинга в отсечении невозможных ветвей.

**Почему это важно:**

**1. Паттерн удовлетворения ограничений (CSP)**

Общий шаблон для задач с ограничениями.

**2. Математика диагоналей**

Для диагонали row - col константа, для антидиагонали row + col константа.

**3. Оптимизация битовыми масками**

Ещё быстрее с битовыми операциями.

**4. Связанные задачи**

Sudoku Solver, Graph Coloring, Knight's Tour.`,
			solutionCode: `from typing import List

def solve_n_queens(n: int) -> List[List[str]]:
    """Находит все решения головоломки N ферзей."""
    result = []
    cols = set()
    diag1 = set()
    diag2 = set()
    queens = [-1] * n

    def is_safe(row: int, col: int) -> bool:
        return col not in cols and \\
               (row - col) not in diag1 and \\
               (row + col) not in diag2

    def place_queen(row: int, col: int) -> None:
        queens[row] = col
        cols.add(col)
        diag1.add(row - col)
        diag2.add(row + col)

    def remove_queen(row: int, col: int) -> None:
        cols.remove(col)
        diag1.remove(row - col)
        diag2.remove(row + col)

    def build_board() -> List[str]:
        return ['.' * queens[r] + 'Q' + '.' * (n - queens[r] - 1) for r in range(n)]

    def backtrack(row: int) -> None:
        if row == n:
            result.append(build_board())
            return
        for col in range(n):
            if is_safe(row, col):
                place_queen(row, col)
                backtrack(row + 1)
                remove_queen(row, col)

    backtrack(0)
    return result`
		},
		uz: {
			title: 'N farzin',
			description: `N×N shaxmat taxtasiga N ta farzinni hech qaysi ikkitasi bir-birini urmaydigandek qo'ying.

**Masala:**

Butun son \`n\` berilgan. N farzin boshqotirmasining barcha turli yechimlarini qaytaring.

Har bir yechimda 'Q' farzinni, '.' bo'sh katakni bildiradi.

Farzin bir xil qator, ustun yoki diagonaldagi har qanday figurani uradi.

**Misollar:**

\`\`\`
Kirish: n = 4
Chiqish: [
  [".Q..",
   "...Q",
   "Q...",
   "..Q."],

  ["..Q.",
   "Q...",
   "...Q",
   ".Q.."]
]
\`\`\`

**Asosiy tushuncha:**

Farzinlarni qator bo'yicha joylashtirish uchun backtracking ishlating:
1. Joriy qatorning har bir ustuniga farzin qo'yishga harakat qiling
2. Pozitsiyaning xavfsizligini tekshiring (hujum qilinmagan)
3. Xavfsiz bo'lsa, keyingi qatorga o'ting
4. Xavfsiz pozitsiya yo'q bo'lsa, orqaga qayting

**Cheklovlar:**
- 1 <= n <= 9

**Vaqt murakkabligi:** O(n!)
**Xotira murakkabligi:** O(n²)`,
			hint1: `Farzinlarni qator bo'yicha qo'ying. Oldingi qatorlardan hech qaysi farzin urmaydigan bo'lsa pozitsiya xavfsiz. Hujum qilingan ustun va diagonallarni kuzating.`,
			hint2: `Uchta to'plam ishlating: cols (ustunlar), diag1 (row - col, diagonallar), diag2 (row + col, antidiagonallar). Hech qaysi to'plamda bo'lmasa pozitsiya xavfsiz.`,
			whyItMatters: `N farzin klassik cheklovlarni qondirish masalasi. U backtrackingning imkonsiz tarmoqlarni kesish quvvatini ko'rsatadi.

**Bu nima uchun muhim:**

**1. Cheklovlarni qondirish patterni (CSP)**

Cheklovli masalalar uchun umumiy shablon.

**2. Diagonal matematikasi**

Diagonal uchun row - col o'zgarmas, antidiagonal uchun row + col o'zgarmas.

**3. Bit maskalari bilan optimallashtirish**

Bit operatsiyalari bilan yanada tezroq.

**4. Bog'liq masalalar**

Sudoku Solver, Graph Coloring, Knight's Tour.`,
			solutionCode: `from typing import List

def solve_n_queens(n: int) -> List[List[str]]:
    """N farzin boshqotirmasining barcha yechimlarini topadi."""
    result = []
    cols = set()
    diag1 = set()
    diag2 = set()
    queens = [-1] * n

    def is_safe(row: int, col: int) -> bool:
        return col not in cols and \\
               (row - col) not in diag1 and \\
               (row + col) not in diag2

    def place_queen(row: int, col: int) -> None:
        queens[row] = col
        cols.add(col)
        diag1.add(row - col)
        diag2.add(row + col)

    def remove_queen(row: int, col: int) -> None:
        cols.remove(col)
        diag1.remove(row - col)
        diag2.remove(row + col)

    def build_board() -> List[str]:
        return ['.' * queens[r] + 'Q' + '.' * (n - queens[r] - 1) for r in range(n)]

    def backtrack(row: int) -> None:
        if row == n:
            result.append(build_board())
            return
        for col in range(n):
            if is_safe(row, col):
                place_queen(row, col)
                backtrack(row + 1)
                remove_queen(row, col)

    backtrack(0)
    return result`
		}
	}
};

export default task;
