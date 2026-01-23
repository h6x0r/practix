import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'py-boolean-logic',
	title: 'Boolean Logic',
	difficulty: 'easy',
	tags: ['python', 'fundamentals', 'booleans', 'logic'],
	estimatedTime: '10m',
	isPremium: false,
	order: 6,

	description: `# Boolean Logic

Master logical operators in Python: \`and\`, \`or\`, and \`not\`.

## Task

Implement the function \`can_enter(age, has_ticket, is_vip)\` that determines if a person can enter an event.

## Requirements

- Entry is allowed if:
  - Person is at least 18 years old AND has a ticket, OR
  - Person is a VIP (VIPs can enter regardless of age or ticket)
- Return \`True\` if entry is allowed, \`False\` otherwise

## Examples

\`\`\`python
>>> can_enter(25, True, False)
True  # Adult with ticket

>>> can_enter(16, True, False)
False  # Minor with ticket - not allowed

>>> can_enter(16, False, True)
True  # Minor but VIP - allowed

>>> can_enter(25, False, False)
False  # Adult without ticket or VIP status
\`\`\``,

	initialCode: `def can_enter(age: int, has_ticket: bool, is_vip: bool) -> bool:
    """Determine if a person can enter an event.

    Entry rules:
    - Must be 18+ AND have a ticket, OR
    - Be a VIP (bypasses all other requirements)

    Args:
        age: Person's age in years
        has_ticket: Whether they have a valid ticket
        is_vip: Whether they have VIP status

    Returns:
        True if entry is allowed, False otherwise
    """
    # TODO: Implement the entry logic
    pass`,

	solutionCode: `def can_enter(age: int, has_ticket: bool, is_vip: bool) -> bool:
    """Determine if a person can enter an event.

    Entry rules:
    - Must be 18+ AND have a ticket, OR
    - Be a VIP (bypasses all other requirements)

    Args:
        age: Person's age in years
        has_ticket: Whether they have a valid ticket
        is_vip: Whether they have VIP status

    Returns:
        True if entry is allowed, False otherwise
    """
    # VIPs always get in - check this first (short-circuit evaluation)
    if is_vip:
        return True

    # Regular entry: must be adult AND have ticket
    # Both conditions must be True for entry
    is_adult = age >= 18
    return is_adult and has_ticket`,

	testCode: `import unittest

class Test(unittest.TestCase):
    def test_1(self):
        """Adult with ticket can enter"""
        self.assertTrue(can_enter(25, True, False))

    def test_2(self):
        """Minor with ticket cannot enter"""
        self.assertFalse(can_enter(16, True, False))

    def test_3(self):
        """VIP minor without ticket can enter"""
        self.assertTrue(can_enter(16, False, True))

    def test_4(self):
        """Adult without ticket cannot enter"""
        self.assertFalse(can_enter(25, False, False))

    def test_5(self):
        """Exactly 18 with ticket can enter"""
        self.assertTrue(can_enter(18, True, False))

    def test_6(self):
        """Just under 18 with ticket cannot enter"""
        self.assertFalse(can_enter(17, True, False))

    def test_7(self):
        """VIP adult with ticket can enter"""
        self.assertTrue(can_enter(30, True, True))

    def test_8(self):
        """VIP adult without ticket can enter"""
        self.assertTrue(can_enter(30, False, True))

    def test_9(self):
        """Age 0 VIP can enter"""
        self.assertTrue(can_enter(0, False, True))

    def test_10(self):
        """Senior without ticket cannot enter"""
        self.assertFalse(can_enter(65, False, False))

if __name__ == '__main__':
    unittest.main()`,

	hint1: 'VIPs bypass all rules - check `is_vip` first and return True immediately if so.',
	hint2: 'For non-VIPs, use `and` to combine two conditions: age check AND ticket check.',

	whyItMatters: `Boolean logic is the foundation of all decision-making in software.

**Production Pattern:**

\`\`\`python
def can_access_resource(user: dict, resource: dict) -> bool:
    """Check if user can access a protected resource."""
    # Admin bypasses all checks
    if user.get("role") == "admin":
        return True

    # Check ownership
    is_owner = resource.get("owner_id") == user.get("id")

    # Check permissions
    has_permission = user.get("role") in resource.get("allowed_roles", [])

    # Check if resource is public
    is_public = resource.get("visibility") == "public"

    return is_owner or has_permission or is_public

def validate_form(data: dict) -> tuple[bool, list[str]]:
    """Validate form data with multiple conditions."""
    errors = []

    # All fields must pass their validation
    if not data.get("email") or "@" not in data["email"]:
        errors.append("Invalid email")

    password = data.get("password", "")
    has_length = len(password) >= 8
    has_digit = any(c.isdigit() for c in password)
    has_upper = any(c.isupper() for c in password)

    if not (has_length and has_digit and has_upper):
        errors.append("Password must be 8+ chars with digit and uppercase")

    return len(errors) == 0, errors
\`\`\`

**Practical Benefits:**
- Access control systems use complex boolean logic
- Form validation requires combining multiple conditions
- Short-circuit evaluation improves performance`,

	translations: {
		ru: {
			title: 'Булева логика',
			description: `# Булева логика

Освойте логические операторы Python: \`and\`, \`or\` и \`not\`.

## Задача

Реализуйте функцию \`can_enter(age, has_ticket, is_vip)\`, которая определяет, может ли человек войти на мероприятие.

## Требования

- Вход разрешён, если:
  - Человеку не менее 18 лет И у него есть билет, ИЛИ
  - Человек — VIP (VIP могут входить без учёта возраста и билета)
- Верните \`True\` если вход разрешён, \`False\` в противном случае

## Примеры

\`\`\`python
>>> can_enter(25, True, False)
True  # Взрослый с билетом

>>> can_enter(16, True, False)
False  # Несовершеннолетний с билетом — не разрешено

>>> can_enter(16, False, True)
True  # Несовершеннолетний, но VIP — разрешено

>>> can_enter(25, False, False)
False  # Взрослый без билета и VIP статуса
\`\`\``,
			hint1: 'VIP обходят все правила — сначала проверьте `is_vip` и верните True.',
			hint2: 'Для не-VIP используйте `and` для объединения условий: проверка возраста И билета.',
			whyItMatters: `Булева логика — основа всех решений в программировании.

**Продакшен паттерн:**

\`\`\`python
def can_access_resource(user: dict, resource: dict) -> bool:
    """Проверка доступа пользователя к ресурсу."""
    if user.get("role") == "admin":
        return True

    is_owner = resource.get("owner_id") == user.get("id")
    has_permission = user.get("role") in resource.get("allowed_roles", [])

    return is_owner or has_permission
\`\`\`

**Практические преимущества:**
- Системы контроля доступа используют сложную булеву логику
- Валидация форм требует комбинации условий
- Short-circuit evaluation улучшает производительность`,
		},
		uz: {
			title: 'Mantiqiy operatorlar',
			description: `# Mantiqiy operatorlar

Python mantiqiy operatorlarini o'rganing: \`and\`, \`or\` va \`not\`.

## Vazifa

Odam tadbirga kirishi mumkinligini aniqlovchi \`can_enter(age, has_ticket, is_vip)\` funksiyasini amalga oshiring.

## Talablar

- Kirish ruxsat etiladi, agar:
  - Odam kamida 18 yoshda VA chipta bor, YOKI
  - Odam VIP (VIP lar yoshi va chiptadan qat'i nazar kirishi mumkin)
- Kirish ruxsat etilsa \`True\`, aks holda \`False\` qaytaring

## Misollar

\`\`\`python
>>> can_enter(25, True, False)
True  # Kattalar chiptali

>>> can_enter(16, True, False)
False  # Voyaga yetmagan chiptali - ruxsat yo'q

>>> can_enter(16, False, True)
True  # Voyaga yetmagan lekin VIP - ruxsat

>>> can_enter(25, False, False)
False  # Kattalar chiptasiz va VIP holatsiz
\`\`\``,
			hint1: "VIP lar barcha qoidalarni chetlab o'tadi — avval `is_vip` ni tekshiring.",
			hint2: 'VIP bolmaganlar uchun `and` dan foydalanib shartlarni birlashtiring.',
			whyItMatters: `Mantiqiy operatorlar dasturiy ta'minotdagi barcha qarorlarning asosidir.

**Ishlab chiqarish patterni:**

\`\`\`python
def can_access_resource(user: dict, resource: dict) -> bool:
    """Foydalanuvchi resursga kirish huquqini tekshirish."""
    if user.get("role") == "admin":
        return True

    is_owner = resource.get("owner_id") == user.get("id")
    has_permission = user.get("role") in resource.get("allowed_roles", [])

    return is_owner or has_permission
\`\`\`

**Amaliy foydalari:**
- Kirish nazorati tizimlari murakkab mantiqiy amallardan foydalanadi
- Forma tekshiruvi bir nechta shartlarni birlashtiradi`,
		},
	},
};

export default task;
