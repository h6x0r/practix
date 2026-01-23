import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-bank-account',
	title: 'Bank Account',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'oop', 'classes'],
	estimatedTime: '15m',
	isPremium: false,
	order: 2,

	description: `# Bank Account

Practice encapsulation with a bank account class.

## Task

Create a class \`BankAccount\` with deposit, withdraw, and balance operations.

## Requirements

- \`__init__(self, owner, balance=0)\`: Initialize with owner name and optional starting balance
- \`deposit(self, amount)\`: Add amount to balance, return new balance
- \`withdraw(self, amount)\`: Subtract from balance if sufficient funds, return new balance or None if insufficient
- \`get_balance(self)\`: Return current balance

## Examples

\`\`\`python
>>> acc = BankAccount("Alice", 100)
>>> acc.deposit(50)
150
>>> acc.withdraw(30)
120
>>> acc.withdraw(200)  # Insufficient funds
None
>>> acc.get_balance()
120
\`\`\``,

	initialCode: `class BankAccount:
    """A simple bank account class."""

    def __init__(self, owner: str, balance: float = 0):
        # TODO: Initialize owner and balance
        pass

    def deposit(self, amount: float) -> float:
        # TODO: Add to balance and return new balance
        pass

    def withdraw(self, amount: float):
        # TODO: Subtract if funds available, else return None
        pass

    def get_balance(self) -> float:
        # TODO: Return current balance
        pass`,

	solutionCode: `class BankAccount:
    """A simple bank account class."""

    def __init__(self, owner: str, balance: float = 0):
        self.owner = owner
        self._balance = balance  # Convention: underscore for "private"

    def deposit(self, amount: float) -> float:
        self._balance += amount
        return self._balance

    def withdraw(self, amount: float):
        if amount > self._balance:
            return None  # Insufficient funds
        self._balance -= amount
        return self._balance

    def get_balance(self) -> float:
        return self._balance`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        acc = BankAccount("Alice", 100)
        self.assertEqual(acc.deposit(50), 150)

    def test_2(self):
        acc = BankAccount("Alice", 100)
        self.assertEqual(acc.withdraw(30), 70)

    def test_3(self):
        acc = BankAccount("Alice", 100)
        self.assertIsNone(acc.withdraw(200))

    def test_4(self):
        acc = BankAccount("Alice", 100)
        self.assertEqual(acc.get_balance(), 100)

    def test_5(self):
        acc = BankAccount("Bob")
        self.assertEqual(acc.get_balance(), 0)

    def test_6(self):
        acc = BankAccount("Alice", 100)
        acc.withdraw(200)
        self.assertEqual(acc.get_balance(), 100)

    def test_7(self):
        acc = BankAccount("Test", 50)
        acc.deposit(50)
        self.assertEqual(acc.withdraw(100), 0)

    def test_8(self):
        acc = BankAccount("Test")
        self.assertEqual(acc.owner, "Test")

    def test_9(self):
        acc = BankAccount("Test", 100)
        self.assertEqual(acc.withdraw(100), 0)

    def test_10(self):
        acc = BankAccount("Test", 0)
        self.assertIsNone(acc.withdraw(1))

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'Use self._balance (with underscore) to indicate the balance should not be accessed directly.',
	hint2: 'In withdraw(), check if amount > self._balance before subtracting.',

	whyItMatters: `Encapsulation protects object state and ensures valid operations.

**Production Pattern:**

\`\`\`python
class BankAccount:
    def __init__(self, owner: str, balance: float = 0):
        self._owner = owner
        self._balance = balance
        self._transactions = []

    @property
    def balance(self) -> float:
        return self._balance

    def deposit(self, amount: float) -> float:
        if amount <= 0:
            raise ValueError("Deposit must be positive")
        self._balance += amount
        self._transactions.append(("deposit", amount))
        return self._balance

    def withdraw(self, amount: float) -> float:
        if amount <= 0:
            raise ValueError("Withdrawal must be positive")
        if amount > self._balance:
            raise InsufficientFundsError(f"Need {amount}, have {self._balance}")
        self._balance -= amount
        self._transactions.append(("withdraw", amount))
        return self._balance
\`\`\``,

	translations: {
		ru: {
			title: 'Банковский счёт',
			description: `# Банковский счёт

Практика инкапсуляции с классом банковского счёта.

## Задача

Создайте класс \`BankAccount\` с операциями депозита, снятия и баланса.

## Требования

- \`__init__(self, owner, balance=0)\`: Инициализация с именем владельца и опциональным начальным балансом
- \`deposit(self, amount)\`: Добавить сумму к балансу, вернуть новый баланс
- \`withdraw(self, amount)\`: Вычесть из баланса при наличии средств, вернуть новый баланс или None
- \`get_balance(self)\`: Вернуть текущий баланс`,
			hint1: 'Используйте self._balance (с подчёркиванием) для обозначения приватного атрибута.',
			hint2: 'В withdraw() проверьте amount > self._balance перед вычитанием.',
			whyItMatters: `Инкапсуляция защищает состояние объекта и обеспечивает валидные операции.`,
		},
		uz: {
			title: 'Bank hisobi',
			description: `# Bank hisobi

Bank hisobi klassi bilan inkapsulatsiya mashqi.

## Vazifa

Depozit, yechish va balans operatsiyalariga ega \`BankAccount\` klassini yarating.

## Talablar

- \`__init__(self, owner, balance=0)\`: Egasi ismi va ixtiyoriy boshlang'ich balans bilan ishga tushirish
- \`deposit(self, amount)\`: Balansga qo'shish, yangi balansni qaytarish
- \`withdraw(self, amount)\`: Mablag' yetarli bo'lsa ayirish, yangi balans yoki None qaytarish
- \`get_balance(self)\`: Joriy balansni qaytarish`,
			hint1: "Xususiy atributni ko'rsatish uchun self._balance (pastki chiziq bilan) ishlating.",
			hint2: "withdraw() da ayirishdan oldin amount > self._balance ni tekshiring.",
			whyItMatters: `Inkapsulatsiya ob'ekt holatini himoya qiladi va to'g'ri operatsiyalarni ta'minlaydi.`,
		},
	},
};

export default task;
