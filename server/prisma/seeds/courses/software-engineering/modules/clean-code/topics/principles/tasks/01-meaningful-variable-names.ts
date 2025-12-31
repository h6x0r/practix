import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-clean-code-meaningful-variables',
	title: 'Meaningful Variable Names',
	difficulty: 'easy',
	tags: ['go', 'clean-code', 'naming', 'variables'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Refactor code to use meaningful, intention-revealing variable names that make the code self-documenting.

**You will refactor:**

1. **ProcessData(data []int) int** - Process user ages and return average age of adults
2. Replace cryptic variable names (x, y, tmp, d, etc.) with clear, descriptive names
3. Make the code self-explanatory without needing comments

**Key Concepts:**
- **Intention-Revealing Names**: Names should reveal why they exist, what they do, and how they're used
- **Avoid Disinformation**: Don't use names that obscure meaning
- **Pronounceable Names**: Use names you can discuss with team members
- **Searchable Names**: Use descriptive names instead of magic numbers

**Example - Before:**

\`\`\`go
func calc(d []int) int {
    t := 0
    c := 0
    for _, x := range d {
        if x >= 18 {
            t += x
            c++
        }
    }
    if c == 0 {
        return 0
    }
    return t / c
}
\`\`\`

**Example - After:**

\`\`\`go
func ProcessData(ages []int) int {
    totalAdultAge := 0
    adultCount := 0
    const legalAdultAge = 18

    for _, age := range ages {
        if age >= legalAdultAge {
            totalAdultAge += age
            adultCount++
        }
    }

    if adultCount == 0 {
        return 0
    }

    return totalAdultAge / adultCount
}
\`\`\`

**When to use meaningful names:**
- Always! Good naming is fundamental to clean code
- Variables, functions, types, packages - everything needs clear names
- Especially important in team environments

**Constraints:**
- Keep the same logic and algorithm
- Only change variable names and add constants
- Make names reveal intention`,
	initialCode: `package principles

func ProcessData(d []int) int {
	for _, x := range d {
		if x >= 18 {
		}
	}
	if c == 0 {
		return 0
	}
	return t / c
}`,
	solutionCode: `package principles

// ProcessData calculates the average age of adults (18+) from a list of ages
func ProcessData(ages []int) int {
	const legalAdultAge = 18	// extracted magic number to named constant

	totalAdultAge := 0	// clear what we're summing
	adultCount := 0	// clear what we're counting

	for _, age := range ages {	// 'age' is more meaningful than 'x'
		if age >= legalAdultAge {	// intention-revealing: checking if adult
			totalAdultAge += age
			adultCount++
		}
	}

	if adultCount == 0 {	// prevents division by zero
		return 0
	}

	averageAdultAge := totalAdultAge / adultCount	// explicit calculation result
	return averageAdultAge
}`,
	hint1: `Replace 'd' with 'ages', 'x' with 'age', 't' with 'totalAdultAge', 'c' with 'adultCount'. Extract the magic number 18 to a constant named 'legalAdultAge'.`,
	hint2: `Use names that describe what the value represents, not just its type. 'totalAdultAge' is better than 'sum' or 'total'. Add inline comments only where the name isn't sufficient.`,
	testCode: `package principles

import "testing"

// Test1: Basic case with adults and non-adults
func Test1(t *testing.T) {
	ages := []int{20, 25, 30, 15, 10}
	result := ProcessData(ages)
	// Adults: 20, 25, 30 -> avg = 75/3 = 25
	if result != 25 {
		t.Errorf("ProcessData(%v) = %d, want 25", ages, result)
	}
}

// Test2: Empty slice returns zero
func Test2(t *testing.T) {
	ages := []int{}
	result := ProcessData(ages)
	if result != 0 {
		t.Errorf("ProcessData([]) = %d, want 0", result)
	}
}

// Test3: No adults returns zero
func Test3(t *testing.T) {
	ages := []int{5, 10, 15, 17}
	result := ProcessData(ages)
	if result != 0 {
		t.Errorf("ProcessData(%v) = %d, want 0", ages, result)
	}
}

// Test4: All adults
func Test4(t *testing.T) {
	ages := []int{20, 30, 40}
	result := ProcessData(ages)
	// Average = 90/3 = 30
	if result != 30 {
		t.Errorf("ProcessData(%v) = %d, want 30", ages, result)
	}
}

// Test5: Edge case - exactly 18
func Test5(t *testing.T) {
	ages := []int{18}
	result := ProcessData(ages)
	if result != 18 {
		t.Errorf("ProcessData([18]) = %d, want 18", result)
	}
}

// Test6: Edge case - 17 is not adult
func Test6(t *testing.T) {
	ages := []int{17}
	result := ProcessData(ages)
	if result != 0 {
		t.Errorf("ProcessData([17]) = %d, want 0", result)
	}
}

// Test7: Single adult
func Test7(t *testing.T) {
	ages := []int{25}
	result := ProcessData(ages)
	if result != 25 {
		t.Errorf("ProcessData([25]) = %d, want 25", result)
	}
}

// Test8: Large ages
func Test8(t *testing.T) {
	ages := []int{100, 80, 60}
	result := ProcessData(ages)
	// Average = 240/3 = 80
	if result != 80 {
		t.Errorf("ProcessData(%v) = %d, want 80", ages, result)
	}
}

// Test9: Mixed ages with boundary
func Test9(t *testing.T) {
	ages := []int{18, 17, 19, 16}
	result := ProcessData(ages)
	// Adults: 18, 19 -> avg = 37/2 = 18
	if result != 18 {
		t.Errorf("ProcessData(%v) = %d, want 18", ages, result)
	}
}

// Test10: Integer division truncates
func Test10(t *testing.T) {
	ages := []int{18, 19}
	result := ProcessData(ages)
	// Average = 37/2 = 18 (integer division)
	if result != 18 {
		t.Errorf("ProcessData(%v) = %d, want 18", ages, result)
	}
}
`,
	whyItMatters: `Meaningful variable names are the foundation of clean, maintainable code.

**Why Meaningful Names Matter:**

**1. Code Readability - Your Code is Read 10x More Than Written**

\`\`\`go
// BAD: Cryptic names require mental translation
func calc(d []int) int {
    t, c := 0, 0	// what are t and c?
    for _, x := range d {	// what is x?
        if x >= 18 {	// why 18?
            t += x
            c++
        }
    }
    return t / c	// what does this mean?
}

// GOOD: Self-documenting code tells a story
func CalculateAverageAdultAge(ages []int) int {
    const legalAdultAge = 18
    totalAdultAge := 0
    adultCount := 0

    for _, age := range ages {
        if age >= legalAdultAge {
            totalAdultAge += age
            adultCount++
        }
    }

    return totalAdultAge / adultCount
}
// No comments needed - the code explains itself!
\`\`\`

**2. Reduce Cognitive Load**

Every time you see a variable like 'x', 'tmp', or 'd', your brain has to:
1. Remember what it represents
2. Trace back to where it was defined
3. Context-switch from the current task

\`\`\`go
// BAD: Forces you to keep mental map
tmp := fetchData()
x := transform(tmp)
y := validate(x)
// Wait, what was tmp again?

// GOOD: Names carry their context
rawUserData := fetchData()
normalizedUserData := transform(rawUserData)
validatedUserData := validate(normalizedUserData)
// Crystal clear at every step
\`\`\`

**3. Searchability and Refactoring**

\`\`\`go
// BAD: Try searching for "x" or "d" in your codebase
for _, x := range d {
    if x > 18 { ... }
}

// GOOD: Easy to find all age-related logic
for _, age := range ages {
    if age > legalAdultAge { ... }
}
// Search "age" or "legalAdultAge" - all occurrences are relevant
\`\`\`

**4. Magic Numbers Kill Maintainability**

\`\`\`go
// BAD: What if the legal age changes? Find all "18"s in codebase?
if age >= 18 { ... }
if userAge >= 18 { ... }
if person.Age >= 18 { ... }

// GOOD: Single source of truth
const legalAdultAge = 18

if age >= legalAdultAge { ... }
if userAge >= legalAdultAge { ... }
if person.Age >= legalAdultAge { ... }
// Change once, updates everywhere
\`\`\`

**Real-World Impact:**

**Google Study:** Engineers spend 60% of their time reading code, only 40% writing it.
Every second saved in comprehension multiplies across your team.

**Example from Production Code:**

\`\`\`go
// BEFORE: 5 minutes to understand
func p(u *U) error {
    if u.S != 1 { return errors.New("invalid") }
    t := time.Now().Unix()
    if u.E < t { return errors.New("expired") }
    return nil
}

// AFTER: Instant understanding
func ValidateUserSession(user *User) error {
    const activeStatus = 1
    if user.Status != activeStatus {
        return errors.New("user session is not active")
    }

    currentTimestamp := time.Now().Unix()
    if user.ExpiresAt < currentTimestamp {
        return errors.New("user session has expired")
    }

    return nil
}
\`\`\`

**Naming Guidelines from Go Community:**

\`\`\`go
// 1. Use camelCase for multi-word names
userAge, not user_age or userage

// 2. Short names for short scopes
for i, v := range items { ... }  // OK: i is obvious in small loop

// 3. Longer names for wider scopes
var configurationManager *ConfigManager  // OK: package-level variable

// 4. Avoid redundancy with type names
var userMap map[string]*User  // BAD
var users map[string]*User    // GOOD - type already says it's a map

// 5. Acronyms should be consistent
userID, not userId or userIf
httpServer, not HttpServer
\`\`\`

**Common Anti-Patterns:**

\`\`\`go
// ❌ Single letter names (except loop counters)
n := getName()

// ✅ Descriptive names
userName := getName()

// ❌ Generic names
data := fetchData()
result := process(data)

// ✅ Specific names
userProfiles := fetchData()
validatedProfiles := process(userProfiles)

// ❌ Hungarian notation (type prefixes)
strName := "John"
intAge := 25

// ✅ Let Go's type system handle types
name := "John"
age := 25
\`\`\`

**The Name Length Rule:**

\`\`\`go
// Scope size ∝ Name length
for i := 0; i < 10; i++ { ... }           // i is fine: 2 lines
var u *User                                // BAD: u used for 50 lines
var currentUser *User                      // GOOD: clear and searchable
var httpRequestAuthenticationMiddleware    // TOO LONG: unwieldy
var authMiddleware                         // GOOD: balanced
\`\`\``,
	order: 0,
	translations: {
		ru: {
			title: 'Осмысленные имена переменных',
			description: `Выполните рефакторинг кода для использования осмысленных, раскрывающих намерения имён переменных, которые делают код самодокументируемым.

**Вы выполните рефакторинг:**

1. **ProcessData(data []int) int** - Обработка возрастов пользователей и возврат среднего возраста взрослых
2. Замените загадочные имена переменных (x, y, tmp, d, и т.д.) на чёткие, описательные имена
3. Сделайте код самообъясняющимся без необходимости комментариев

**Ключевые концепции:**
- **Раскрывающие намерения имена**: Имена должны раскрывать зачем они существуют, что делают и как используются
- **Избегайте дезинформации**: Не используйте имена, которые скрывают смысл
- **Произносимые имена**: Используйте имена, которые можно обсуждать с командой
- **Находимые имена**: Используйте описательные имена вместо магических чисел

**Пример - До:**

\`\`\`go
func calc(d []int) int {
    t := 0
    c := 0
    for _, x := range d {
        if x >= 18 {
            t += x
            c++
        }
    }
    if c == 0 {
        return 0
    }
    return t / c
}
\`\`\`

**Пример - После:**

\`\`\`go
func ProcessData(ages []int) int {
    totalAdultAge := 0
    adultCount := 0
    const legalAdultAge = 18

    for _, age := range ages {
        if age >= legalAdultAge {
            totalAdultAge += age
            adultCount++
        }
    }

    if adultCount == 0 {
        return 0
    }

    return totalAdultAge / adultCount
}
\`\`\`

**Когда использовать осмысленные имена:**
- Всегда! Хорошее именование — основа чистого кода
- Переменные, функции, типы, пакеты — всё нуждается в ясных именах
- Особенно важно в командной разработке

**Ограничения:**
- Сохраните ту же логику и алгоритм
- Изменяйте только имена переменных и добавляйте константы
- Сделайте имена раскрывающими намерения`,
			hint1: `Замените 'd' на 'ages', 'x' на 'age', 't' на 'totalAdultAge', 'c' на 'adultCount'. Извлеките магическое число 18 в константу 'legalAdultAge'.`,
			hint2: `Используйте имена, описывающие что значение представляет, а не только его тип. 'totalAdultAge' лучше чем 'sum' или 'total'. Добавляйте комментарии только там, где имени недостаточно.`,
			whyItMatters: `Осмысленные имена переменных — основа чистого, поддерживаемого кода.

**Почему осмысленные имена важны:**

**1. Читаемость кода - Код читается в 10 раз чаще, чем пишется**

Каждая секунда, сэкономленная на понимании, умножается на всю команду.

**2. Снижение когнитивной нагрузки**

Каждый раз видя переменную вроде 'x', 'tmp' или 'd', ваш мозг должен помнить что она представляет.

**3. Поиск и рефакторинг**

Попробуйте найти "x" или "d" в кодовой базе — множество ложных срабатываний.
Поиск "age" или "legalAdultAge" — все вхождения релевантны.

**4. Магические числа убивают поддерживаемость**

Что если законный возраст изменится? Искать все "18" в коде?
С константой \`legalAdultAge\` — изменяется в одном месте.`,
			solutionCode: `package principles

// ProcessData вычисляет средний возраст взрослых (18+) из списка возрастов
func ProcessData(ages []int) int {
	const legalAdultAge = 18	// извлекли магическое число в именованную константу

	totalAdultAge := 0	// ясно что мы суммируем
	adultCount := 0	// ясно что мы считаем

	for _, age := range ages {	// 'age' более осмысленно чем 'x'
		if age >= legalAdultAge {	// раскрывает намерение: проверка совершеннолетия
			totalAdultAge += age
			adultCount++
		}
	}

	if adultCount == 0 {	// предотвращает деление на ноль
		return 0
	}

	averageAdultAge := totalAdultAge / adultCount	// явный результат вычисления
	return averageAdultAge
}`
		},
		uz: {
			title: "O'zgaruvchilarning mazmunli nomlari",
			description: `Kodni mazmunli, niyatni ochib beradigan o'zgaruvchi nomlari bilan refaktoring qiling, bu kodni o'z-o'zini hujjatlaydigan qiladi.

**Siz refaktoring qilasiz:**

1. **ProcessData(data []int) int** - Foydalanuvchilar yoshini qayta ishlash va kattalarning o'rtacha yoshini qaytarish
2. Tushunarsiz o'zgaruvchi nomlarini (x, y, tmp, d, va h.k.) aniq, ta'riflovchi nomlar bilan almashtiring
3. Kodni izohsiz o'z-o'zini tushuntiradigan qiling

**Asosiy tushunchalar:**
- **Niyatni ochib beruvchi nomlar**: Nomlar nima uchun mavjud, nima qiladi va qanday ishlatilishini ko'rsatishi kerak
- **Noto'g'ri ma'lumotdan qoching**: Ma'noni yashiradigan nomlarni ishlatmang
- **Talaffuz qilinadigan nomlar**: Jamoa a'zolari bilan muhokama qilish mumkin bo'lgan nomlar ishlating
- **Qidiriladigan nomlar**: Sehrli raqamlar o'rniga ta'riflovchi nomlar ishlating

**Misol - Oldin:**

\`\`\`go
func calc(d []int) int {
    t := 0
    c := 0
    for _, x := range d {
        if x >= 18 {
            t += x
            c++
        }
    }
    if c == 0 {
        return 0
    }
    return t / c
}
\`\`\`

**Misol - Keyin:**

\`\`\`go
func ProcessData(ages []int) int {
    totalAdultAge := 0
    adultCount := 0
    const legalAdultAge = 18

    for _, age := range ages {
        if age >= legalAdultAge {
            totalAdultAge += age
            adultCount++
        }
    }

    if adultCount == 0 {
        return 0
    }

    return totalAdultAge / adultCount
}
\`\`\`

**Qachon mazmunli nomlar ishlatiladi:**
- Har doim! Yaxshi nomlash toza kodning asosidir
- O'zgaruvchilar, funksiyalar, turlar, paketlar - hamma narsaga aniq nomlar kerak
- Ayniqsa jamoaviy muhitda muhim

**Cheklovlar:**
- Bir xil mantiq va algoritmni saqlang
- Faqat o'zgaruvchi nomlarini o'zgartiring va konstantalar qo'shing
- Nomlar niyatni ochib berishi kerak`,
			hint1: `'d' ni 'ages', 'x' ni 'age', 't' ni 'totalAdultAge', 'c' ni 'adultCount' bilan almashtiring. 18 sehrli raqamini 'legalAdultAge' konstantasiga ajrating.`,
			hint2: `Qiymat nimani ifodalashini tasvirlaydigan nomlarni ishlating, faqat turini emas. 'totalAdultAge' 'sum' yoki 'total' dan yaxshiroq. Izohlarni faqat nom yetarli bo'lmagan joyda qo'shing.`,
			whyItMatters: `Mazmunli o'zgaruvchi nomlari toza, qo'llab-quvvatlanadigan kodning asosidir.

**Mazmunli nomlar nima uchun muhim:**

**1. Kodning o'qilishi - Kod yozilganidan 10 marta ko'p o'qiladi**

Tushunishda tejangan har bir soniya butun jamoaga ko'payadi.

**2. Kognitiv yukni kamaytirish**

'x', 'tmp' yoki 'd' kabi o'zgaruvchini har safar ko'rganingizda, miyangiz nimani ifodalashini eslab qolishi kerak.

**3. Qidiruv va refaktoring**

Kod bazasida "x" yoki "d" ni topishga harakat qiling — ko'plab noto'g'ri natijalar.
"age" yoki "legalAdultAge" ni qidirish — barcha natijalar tegishli.

**4. Sehrli raqamlar qo'llab-quvvatlashni o'ldiradi**

Qonuniy yosh o'zgarsa nima bo'ladi? Koddagi barcha "18" larni qidirasizmi?
\`legalAdultAge\` konstantasi bilan — bir joyda o'zgaradi.`,
			solutionCode: `package principles

// ProcessData yoshlar ro'yxatidan kattalarning (18+) o'rtacha yoshini hisoblaydi
func ProcessData(ages []int) int {
	const legalAdultAge = 18	// sehrli raqamni nomlangan konstantaga ajratdik

	totalAdultAge := 0	// nima yig'ayotganimiz aniq
	adultCount := 0	// nima sanayotganimiz aniq

	for _, age := range ages {	// 'age' 'x' dan ko'ra mazmunli
		if age >= legalAdultAge {	// niyatni ochadi: kattalarni tekshirish
			totalAdultAge += age
			adultCount++
		}
	}

	if adultCount == 0 {	// nolga bo'lishni oldini oladi
		return 0
	}

	averageAdultAge := totalAdultAge / adultCount	// aniq hisoblash natijasi
	return averageAdultAge
}`
		}
	}
};

export default task;
