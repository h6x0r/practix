import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'greedy-gas-station',
	title: 'Gas Station',
	difficulty: 'medium',
	tags: ['python', 'greedy', 'array', 'circular'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Find the starting gas station for a circular tour.

**Problem:**

There are \`n\` gas stations along a circular route. At each station \`i\`:
- You receive \`gas[i]\` units of gas
- It costs \`cost[i]\` units of gas to travel to the next station

You start with an empty tank. Return the starting station index if you can complete the circuit once clockwise. If there's no solution, return \`-1\`.

If a solution exists, it is **guaranteed to be unique**.

**Examples:**

\`\`\`
Input: gas = [1, 2, 3, 4, 5], cost = [3, 4, 5, 1, 2]
Output: 3

Explanation:
Start at station 3: tank = 0
- Station 3: +4 gas, -1 cost -> tank = 3
- Station 4: +5 gas, -2 cost -> tank = 6
- Station 0: +1 gas, -3 cost -> tank = 4
- Station 1: +2 gas, -4 cost -> tank = 2
- Station 2: +3 gas, -5 cost -> tank = 0
Complete circuit!

Input: gas = [2, 3, 4], cost = [3, 4, 3]
Output: -1

Explanation: Total gas (9) < Total cost (10), impossible.
\`\`\`

**Key Insight:**

1. If total gas >= total cost, a solution exists
2. If we can't reach station j from station i, we can't reach j from any station between i and j
3. So if tank goes negative, start fresh from the next station

**Constraints:**
- n == gas.length == cost.length
- 1 <= n <= 10^5
- 0 <= gas[i], cost[i] <= 10^4

**Time Complexity:** O(n)
**Space Complexity:** O(1)`,
	initialCode: `from typing import List

def can_complete_circuit(gas: List[int], cost: List[int]) -> int:
    # TODO: Find starting station for complete circuit (-1 if impossible)

    return -1`,
	solutionCode: `from typing import List

def can_complete_circuit(gas: List[int], cost: List[int]) -> int:
    """
    Find starting station for complete circuit.
    """
    n = len(gas)
    total_tank = 0
    current_tank = 0
    start = 0

    for i in range(n):
        diff = gas[i] - cost[i]
        total_tank += diff
        current_tank += diff

        # If tank goes negative, can't start from any previous station
        if current_tank < 0:
            start = i + 1
            current_tank = 0

    # If total gas >= total cost, solution exists at 'start'
    return start if total_tank >= 0 else -1


# Brute force solution (for verification)
def can_complete_circuit_brute(gas: List[int], cost: List[int]) -> int:
    """Brute force O(n^2) solution."""
    n = len(gas)

    for start in range(n):
        tank = 0
        success = True

        for i in range(n):
            station = (start + i) % n
            tank += gas[station] - cost[station]

            if tank < 0:
                success = False
                break

        if success:
            return start

    return -1


# Two-pointer approach (same complexity, different perspective)
def can_complete_circuit_two_pointer(gas: List[int], cost: List[int]) -> int:
    """
    Two-pointer approach:
    - end pointer explores forward
    - start pointer moves backward when needed
    """
    n = len(gas)
    tank = 0
    start = 0
    end = 0
    count = 0  # Number of stations visited

    while count < n:
        tank += gas[end] - cost[end]
        count += 1
        end = (end + 1) % n

        while tank < 0 and count < n:
            # Move start backward (wrap around)
            start = (start - 1 + n) % n
            tank += gas[start] - cost[start]
            count += 1

    return start if tank >= 0 else -1


# Return path and tank levels
def can_complete_circuit_detailed(gas: List[int], cost: List[int]) -> dict:
    """Return detailed information about the circuit."""
    n = len(gas)
    start = can_complete_circuit(gas, cost)

    if start == -1:
        return {"start": -1, "possible": False, "path": [], "tank_levels": []}

    path = []
    tank_levels = []
    tank = 0

    for i in range(n):
        station = (start + i) % n
        path.append(station)
        tank += gas[station] - cost[station]
        tank_levels.append(tank)

    return {
        "start": start,
        "possible": True,
        "path": path,
        "tank_levels": tank_levels
    }`,
	testCode: `import pytest
from solution import can_complete_circuit


class TestGasStation:
    def test_basic_case(self):
        """Test basic case with solution"""
        gas = [1, 2, 3, 4, 5]
        cost = [3, 4, 5, 1, 2]
        assert can_complete_circuit(gas, cost) == 3

    def test_no_solution(self):
        """Test when no solution exists"""
        gas = [2, 3, 4]
        cost = [3, 4, 3]
        assert can_complete_circuit(gas, cost) == -1

    def test_single_station(self):
        """Test single station"""
        assert can_complete_circuit([5], [4]) == 0
        assert can_complete_circuit([4], [5]) == -1

    def test_start_from_zero(self):
        """Test when answer is station 0"""
        gas = [5, 1, 2, 3, 4]
        cost = [4, 4, 1, 5, 1]
        assert can_complete_circuit(gas, cost) == 0

    def test_equal_gas_cost(self):
        """Test when total gas equals total cost"""
        gas = [1, 2, 3]
        cost = [2, 3, 1]
        result = can_complete_circuit(gas, cost)
        assert result >= 0  # Solution should exist

    def test_last_station_start(self):
        """Test when start is last station"""
        gas = [1, 2, 3, 4, 5]
        cost = [2, 3, 4, 5, 1]
        assert can_complete_circuit(gas, cost) == 4

    def test_verify_solution(self):
        """Verify returned solution actually works"""
        gas = [1, 2, 3, 4, 5]
        cost = [3, 4, 5, 1, 2]
        start = can_complete_circuit(gas, cost)

        tank = 0
        n = len(gas)
        for i in range(n):
            station = (start + i) % n
            tank += gas[station] - cost[station]
            assert tank >= 0

    def test_all_zero_diff(self):
        """Test when gas equals cost at each station"""
        gas = [3, 3, 3]
        cost = [3, 3, 3]
        result = can_complete_circuit(gas, cost)
        assert result >= 0

    def test_large_difference(self):
        """Test with large differences"""
        gas = [0, 0, 0, 0, 100]
        cost = [10, 10, 10, 10, 10]
        assert can_complete_circuit(gas, cost) == 4

    def test_impossible_total(self):
        """Test when total gas < total cost"""
        gas = [1, 1, 1]
        cost = [2, 2, 2]
        assert can_complete_circuit(gas, cost) == -1`,
	hint1: `If total gas >= total cost, a solution must exist. The key insight is: if we can't reach station j starting from i, we also can't reach j from any station between i and j.`,
	hint2: `Track both total_tank (to check if solution exists) and current_tank (to find starting point). When current_tank goes negative, reset start to i+1 and current_tank to 0.`,
	whyItMatters: `Gas Station demonstrates a clever greedy insight that eliminates the need for checking every starting point. Understanding why the greedy choice works is key to solving similar circular problems.

**Why This Matters:**

**1. The Key Insight**

\`\`\`python
# If we can't reach station j from i, we can't reach j from any k where i < k < j

# Why? At each station between i and j, we would have:
# - Started with 0 gas (instead of positive tank from i)
# - Still couldn't reach j

# This lets us skip ahead to j+1 instead of trying k
\`\`\`

**2. Two Conditions**

\`\`\`python
# 1. Feasibility: sum(gas) >= sum(cost)
# 2. Starting point: first station where prefix sums stay non-negative

# If condition 1 is true, condition 2 will find a valid start
total_tank = sum(gas[i] - cost[i] for i in range(n))
if total_tank < 0:
    return -1  # Impossible
\`\`\`

**3. Circular Array Pattern**

\`\`\`python
# Common in circular array problems:
# - Gas Station
# - Circular Buffer
# - Maximum sum circular subarray
# - Rotating array operations
\`\`\`

**4. Similar Problems**

\`\`\`python
# Minimum refueling stops
# Car pooling capacity
# Meeting scheduler
# Resource allocation in cycles
\`\`\`

**5. Mathematical Proof**

\`\`\`python
# Let surplus[i] = gas[i] - cost[i]
# If sum(surplus) >= 0, solution exists

# Minimum prefix sum index + 1 = optimal start
# Starting after the minimum ensures all future prefix sums >= 0
\`\`\``,
	order: 4,
	translations: {
		ru: {
			title: 'Заправочные станции',
			description: `Найдите начальную заправку для кругового маршрута.

**Задача:**

На круговом маршруте \`n\` заправок. На каждой станции \`i\`:
- Вы получаете \`gas[i]\` единиц топлива
- Переезд до следующей станции стоит \`cost[i]\` единиц

Начинаете с пустым баком. Верните индекс начальной станции для полного круга по часовой стрелке. Если невозможно, верните \`-1\`.

**Примеры:**

\`\`\`
Вход: gas = [1, 2, 3, 4, 5], cost = [3, 4, 5, 1, 2]
Выход: 3

Объяснение: Начав со станции 3, можно проехать полный круг.
\`\`\`

**Ключевая идея:**

1. Если total gas >= total cost, решение существует
2. Если не можем доехать до j из i, то не можем и из любой станции между ними
3. Когда бак становится отрицательным, начинаем заново со следующей станции

**Ограничения:**
- 1 <= n <= 10^5

**Временная сложность:** O(n)
**Пространственная сложность:** O(1)`,
			hint1: `Если total gas >= total cost, решение существует. Ключ: если не можем доехать до j из i, не можем и из любой станции между i и j.`,
			hint2: `Отслеживайте total_tank (существует ли решение) и current_tank (стартовая точка). Когда current_tank < 0, сбросьте start = i+1 и current_tank = 0.`,
			whyItMatters: `Gas Station демонстрирует хитрый жадный подход, устраняющий необходимость проверки каждой стартовой точки.

**Почему это важно:**

**1. Ключевая идея**

Если не можем доехать до j из i, то и из любой k между i и j не сможем.

**2. Два условия**

Выполнимость: sum(gas) >= sum(cost). Стартовая точка: первая станция с неотрицательными префиксными суммами.

**3. Паттерн циклических массивов**

Часто встречается в задачах с круговыми структурами.`,
			solutionCode: `from typing import List

def can_complete_circuit(gas: List[int], cost: List[int]) -> int:
    """Находит начальную станцию для полного круга."""
    n = len(gas)
    total_tank = 0
    current_tank = 0
    start = 0

    for i in range(n):
        diff = gas[i] - cost[i]
        total_tank += diff
        current_tank += diff

        if current_tank < 0:
            start = i + 1
            current_tank = 0

    return start if total_tank >= 0 else -1`
		},
		uz: {
			title: 'Benzin stansiyalari',
			description: `Aylanma marshrut uchun boshlang'ich stansiyani toping.

**Masala:**

Aylanma marshrutda \`n\` ta benzin stansiyasi bor. Har bir stansiyada \`i\`:
- Siz \`gas[i]\` birlik yoqilg'i olasiz
- Keyingi stansiyaga borish \`cost[i]\` birlik sarflaydi

Bo'sh bak bilan boshlaysiz. Soat yo'nalishi bo'yicha to'liq aylanish uchun boshlang'ich stansiya indeksini qaytaring. Imkonsiz bo'lsa \`-1\` qaytaring.

**Misollar:**

\`\`\`
Kirish: gas = [1, 2, 3, 4, 5], cost = [3, 4, 5, 1, 2]
Chiqish: 3

Izoh: 3-stansiyadan boshlab to'liq aylanish mumkin.
\`\`\`

**Asosiy tushuncha:**

1. Agar total gas >= total cost, yechim mavjud
2. Agar i dan j ga yetib bo'lmasa, ular orasidagi biror stansiyadan ham bo'lmaydi
3. Bak manfiy bo'lganda keyingi stansiyadan yangidan boshlang

**Cheklovlar:**
- 1 <= n <= 10^5

**Vaqt murakkabligi:** O(n)
**Xotira murakkabligi:** O(1)`,
			hint1: `Agar total gas >= total cost, yechim mavjud. Kalit: agar i dan j ga yetib bo'lmasa, i va j orasidagi biror stansiyadan ham bo'lmaydi.`,
			hint2: `total_tank (yechim mavjudmi) va current_tank (boshlang'ich nuqta) kuzating. current_tank < 0 bo'lganda start = i+1 va current_tank = 0 qiling.`,
			whyItMatters: `Gas Station har bir boshlang'ich nuqtani tekshirish zaruriyatini yo'qotadigan aqlli greedy yondashuvni ko'rsatadi.

**Bu nima uchun muhim:**

**1. Asosiy tushuncha**

Agar i dan j ga yetib bo'lmasa, i va j orasidagi biror k dan ham bo'lmaydi.

**2. Ikki shart**

Amalga oshirilishi: sum(gas) >= sum(cost). Boshlang'ich nuqta: manfiy bo'lmagan prefiks yig'indilariga ega birinchi stansiya.

**3. Aylanma massivlar patterni**

Aylanma strukturali masalalarda tez-tez uchraydi.`,
			solutionCode: `from typing import List

def can_complete_circuit(gas: List[int], cost: List[int]) -> int:
    """To'liq aylanish uchun boshlang'ich stansiyani topadi."""
    n = len(gas)
    total_tank = 0
    current_tank = 0
    start = 0

    for i in range(n):
        diff = gas[i] - cost[i]
        total_tank += diff
        current_tank += diff

        if current_tank < 0:
            start = i + 1
            current_tank = 0

    return start if total_tank >= 0 else -1`
		}
	}
};

export default task;
