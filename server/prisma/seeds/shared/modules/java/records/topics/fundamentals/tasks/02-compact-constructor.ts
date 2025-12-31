import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-compact-constructor',
    title: 'Compact Constructor and Validation',
    difficulty: 'easy',
    tags: ['java', 'records', 'validation', 'constructor', 'java16'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Compact Constructor and Validation

Records provide a special compact constructor syntax that allows you to validate or normalize data before the record is created. Unlike regular constructors, compact constructors don't repeat the parameter list and assignment code.

## Requirements:
1. Create a record with validation in compact constructor:
   1. Validate that age is positive
   2. Validate that email contains '@'
   3. Throw IllegalArgumentException for invalid data

2. Create another record with data normalization:
   1. Trim whitespace from strings
   2. Normalize email to lowercase
   3. Format phone number

3. Demonstrate validation works:
   1. Create valid records
   2. Try to create invalid records (catch exceptions)

4. Show that validation happens before field assignment

## Example Output:
\`\`\`
=== Compact Constructor Validation ===
Valid person: Person[name=Alice, age=30, email=alice@example.com]

Trying to create person with negative age...
Error: Age must be positive

Trying to create person with invalid email...
Error: Email must contain @

=== Data Normalization ===
Input: "  Bob  ", "BOB@EXAMPLE.COM", "123-456-7890"
Normalized: Contact[name=Bob, email=bob@example.com, phone=1234567890]
\`\`\``,
    initialCode: `// TODO: Create a Person record with validation in compact constructor

// TODO: Create a Contact record with data normalization

public class CompactConstructor {
    public static void main(String[] args) {
        // TODO: Create valid records

        // TODO: Try to create invalid records and handle exceptions

        // TODO: Demonstrate data normalization
    }
}`,
    solutionCode: `// Record with validation in compact constructor
record Person(String name, int age, String email) {
    // Compact constructor - parameters are implicit, no need to list them
    public Person {
        // Validation before field assignment
        if (age < 0) {
            throw new IllegalArgumentException("Age must be positive");
        }
        if (email == null || !email.contains("@")) {
            throw new IllegalArgumentException("Email must contain @");
        }
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("Name cannot be blank");
        }
    }
    // Fields are automatically assigned after the compact constructor
}

// Record with data normalization
record Contact(String name, String email, String phone) {
    public Contact {
        // Normalize data before assignment
        name = name != null ? name.trim() : "";
        email = email != null ? email.toLowerCase().trim() : "";
        phone = phone != null ? phone.replaceAll("[^0-9]", "") : "";
    }
}

public class CompactConstructor {
    public static void main(String[] args) {
        System.out.println("=== Compact Constructor Validation ===");

        // Create valid record
        Person validPerson = new Person("Alice", 30, "alice@example.com");
        System.out.println("Valid person: " + validPerson);

        // Try to create record with invalid age
        System.out.println("\\nTrying to create person with negative age...");
        try {
            Person invalidAge = new Person("Bob", -5, "bob@example.com");
        } catch (IllegalArgumentException e) {
            System.out.println("Error: " + e.getMessage());
        }

        // Try to create record with invalid email
        System.out.println("\\nTrying to create person with invalid email...");
        try {
            Person invalidEmail = new Person("Charlie", 25, "invalid-email");
        } catch (IllegalArgumentException e) {
            System.out.println("Error: " + e.getMessage());
        }

        System.out.println("\\n=== Data Normalization ===");

        // Create record with messy data - it gets normalized
        String rawName = "  Bob  ";
        String rawEmail = "BOB@EXAMPLE.COM";
        String rawPhone = "123-456-7890";

        System.out.println("Input: \\"" + rawName + "\\", \\"" +
            rawEmail + "\\", \\"" + rawPhone + "\\"");

        Contact contact = new Contact(rawName, rawEmail, rawPhone);
        System.out.println("Normalized: " + contact);
    }
}`,
    hint1: `Use compact constructor syntax: public RecordName { validation/normalization code }. No need to list parameters or assign fields - it's done automatically.`,
    hint2: `In compact constructors, you can validate parameters and throw exceptions before fields are assigned. You can also reassign parameters to normalize data (e.g., name = name.trim()).`,
    whyItMatters: `Compact constructors provide a clean way to enforce invariants and data quality in records. They ensure that invalid records can never be created, making your code more robust. Data normalization in constructors ensures consistent data format throughout your application, preventing bugs and simplifying comparisons.

**Production Pattern:**
\`\`\`java
record Money(BigDecimal amount, String currency) {
    public Money {
        if (amount.compareTo(BigDecimal.ZERO) < 0) {
            throw new IllegalArgumentException("Amount cannot be negative");
        }
        currency = currency.toUpperCase().trim();
    }
}
\`\`\`

**Practical Benefits:**
- 100% prevention of invalid data at object creation level
- Data consistency across the entire system`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Test1: Verify valid person creation
class Test1 {
    @Test
    public void testValidPersonCreation() {
        Person person = new Person("Alice", 30, "alice@example.com");
        assertNotNull(person);
        assertEquals("Alice", person.name());
        assertEquals(30, person.age());
        assertEquals("alice@example.com", person.email());
    }
}

// Test2: Verify negative age validation
class Test2 {
    @Test(expected = IllegalArgumentException.class)
    public void testNegativeAgeValidation() {
        new Person("Bob", -5, "bob@example.com");
    }
}

// Test3: Verify invalid email validation
class Test3 {
    @Test(expected = IllegalArgumentException.class)
    public void testInvalidEmailValidation() {
        new Person("Charlie", 25, "invalid-email");
    }
}

// Test4: Verify null name validation
class Test4 {
    @Test(expected = IllegalArgumentException.class)
    public void testNullNameValidation() {
        new Person(null, 30, "test@example.com");
    }
}

// Test5: Verify blank name validation
class Test5 {
    @Test(expected = IllegalArgumentException.class)
    public void testBlankNameValidation() {
        new Person("   ", 30, "test@example.com");
    }
}

// Test6: Verify null email validation
class Test6 {
    @Test(expected = IllegalArgumentException.class)
    public void testNullEmailValidation() {
        new Person("Test", 30, null);
    }
}

// Test7: Verify name trimming in Contact
class Test7 {
    @Test
    public void testNameTrimming() {
        Contact contact = new Contact("  John  ", "JOHN@EXAMPLE.COM", "123-456-7890");
        assertEquals("John", contact.name());
    }
}

// Test8: Verify email normalization in Contact
class Test8 {
    @Test
    public void testEmailNormalization() {
        Contact contact = new Contact("Jane", "  JANE@EXAMPLE.COM  ", "123-456-7890");
        assertEquals("jane@example.com", contact.email());
    }
}

// Test9: Verify phone normalization in Contact
class Test9 {
    @Test
    public void testPhoneNormalization() {
        Contact contact = new Contact("Mike", "mike@example.com", "123-456-7890");
        assertEquals("1234567890", contact.phone());
    }
}

// Test10: Verify all Contact normalizations together
class Test10 {
    @Test
    public void testAllNormalizations() {
        Contact contact = new Contact("  Sarah  ", "  SARAH@TEST.COM  ", "(555) 123-4567");
        assertEquals("Sarah", contact.name());
        assertEquals("sarah@test.com", contact.email());
        assertEquals("5551234567", contact.phone());
    }
}`,
    order: 2,
    translations: {
        ru: {
            title: 'Компактный конструктор и валидация',
            solutionCode: `// Record с валидацией в компактном конструкторе
record Person(String name, int age, String email) {
    // Компактный конструктор - параметры неявные, не нужно их перечислять
    public Person {
        // Валидация перед присваиванием полей
        if (age < 0) {
            throw new IllegalArgumentException("Age must be positive");
        }
        if (email == null || !email.contains("@")) {
            throw new IllegalArgumentException("Email must contain @");
        }
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("Name cannot be blank");
        }
    }
    // Поля автоматически присваиваются после компактного конструктора
}

// Record с нормализацией данных
record Contact(String name, String email, String phone) {
    public Contact {
        // Нормализация данных перед присваиванием
        name = name != null ? name.trim() : "";
        email = email != null ? email.toLowerCase().trim() : "";
        phone = phone != null ? phone.replaceAll("[^0-9]", "") : "";
    }
}

public class CompactConstructor {
    public static void main(String[] args) {
        System.out.println("=== Валидация компактного конструктора ===");

        // Создание валидного record
        Person validPerson = new Person("Alice", 30, "alice@example.com");
        System.out.println("Valid person: " + validPerson);

        // Попытка создать record с невалидным возрастом
        System.out.println("\\nTrying to create person with negative age...");
        try {
            Person invalidAge = new Person("Bob", -5, "bob@example.com");
        } catch (IllegalArgumentException e) {
            System.out.println("Error: " + e.getMessage());
        }

        // Попытка создать record с невалидным email
        System.out.println("\\nTrying to create person with invalid email...");
        try {
            Person invalidEmail = new Person("Charlie", 25, "invalid-email");
        } catch (IllegalArgumentException e) {
            System.out.println("Error: " + e.getMessage());
        }

        System.out.println("\\n=== Нормализация данных ===");

        // Создание record с неаккуратными данными - они нормализуются
        String rawName = "  Bob  ";
        String rawEmail = "BOB@EXAMPLE.COM";
        String rawPhone = "123-456-7890";

        System.out.println("Input: \\"" + rawName + "\\", \\"" +
            rawEmail + "\\", \\"" + rawPhone + "\\"");

        Contact contact = new Contact(rawName, rawEmail, rawPhone);
        System.out.println("Normalized: " + contact);
    }
}`,
            description: `# Компактный конструктор и валидация

Records предоставляют специальный синтаксис компактного конструктора, который позволяет проверять или нормализовать данные перед созданием record. В отличие от обычных конструкторов, компактные конструкторы не повторяют список параметров и код присваивания.

## Требования:
1. Создайте record с валидацией в компактном конструкторе:
   1. Проверьте, что возраст положительный
   2. Проверьте, что email содержит '@'
   3. Выбросите IllegalArgumentException для невалидных данных

2. Создайте другой record с нормализацией данных:
   1. Удалите пробелы из строк
   2. Нормализуйте email в нижний регистр
   3. Отформатируйте номер телефона

3. Продемонстрируйте работу валидации:
   1. Создайте валидные records
   2. Попробуйте создать невалидные records (перехватите исключения)

4. Покажите, что валидация происходит до присваивания полей

## Пример вывода:
\`\`\`
=== Compact Constructor Validation ===
Valid person: Person[name=Alice, age=30, email=alice@example.com]

Trying to create person with negative age...
Error: Age must be positive

Trying to create person with invalid email...
Error: Email must contain @

=== Data Normalization ===
Input: "  Bob  ", "BOB@EXAMPLE.COM", "123-456-7890"
Normalized: Contact[name=Bob, email=bob@example.com, phone=1234567890]
\`\`\``,
            hint1: `Используйте синтаксис компактного конструктора: public RecordName { код валидации/нормализации }. Не нужно перечислять параметры или присваивать поля - это делается автоматически.`,
            hint2: `В компактных конструкторах вы можете валидировать параметры и выбрасывать исключения до присваивания полей. Вы также можете переприсваивать параметры для нормализации данных (например, name = name.trim()).`,
            whyItMatters: `Компактные конструкторы предоставляют чистый способ обеспечения инвариантов и качества данных в records. Они гарантируют, что невалидные records никогда не могут быть созданы, делая ваш код более надежным. Нормализация данных в конструкторах обеспечивает единообразный формат данных во всем приложении, предотвращая баги и упрощая сравнения.

**Продакшен паттерн:**
\`\`\`java
record Money(BigDecimal amount, String currency) {
    public Money {
        if (amount.compareTo(BigDecimal.ZERO) < 0) {
            throw new IllegalArgumentException("Amount cannot be negative");
        }
        currency = currency.toUpperCase().trim();
    }
}
\`\`\`

**Практические преимущества:**
- Предотвращение невалидных данных на 100% на уровне создания объекта
- Единообразие данных по всей системе`
        },
        uz: {
            title: `Ixcham konstruktor va validatsiya`,
            solutionCode: `// Ixcham konstruktorda validatsiya bilan record
record Person(String name, int age, String email) {
    // Ixcham konstruktor - parametrlar yashirin, ularni sanab o'tish kerak emas
    public Person {
        // Maydonlar tayinlanishidan oldin validatsiya
        if (age < 0) {
            throw new IllegalArgumentException("Age must be positive");
        }
        if (email == null || !email.contains("@")) {
            throw new IllegalArgumentException("Email must contain @");
        }
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("Name cannot be blank");
        }
    }
    // Maydonlar ixcham konstruktordan keyin avtomatik tayinlanadi
}

// Ma'lumotlarni normalizatsiya qilish bilan record
record Contact(String name, String email, String phone) {
    public Contact {
        // Tayinlashdan oldin ma'lumotlarni normalizatsiya qilish
        name = name != null ? name.trim() : "";
        email = email != null ? email.toLowerCase().trim() : "";
        phone = phone != null ? phone.replaceAll("[^0-9]", "") : "";
    }
}

public class CompactConstructor {
    public static void main(String[] args) {
        System.out.println("=== Ixcham konstruktor validatsiyasi ===");

        // Yaroqli recordni yaratish
        Person validPerson = new Person("Alice", 30, "alice@example.com");
        System.out.println("Valid person: " + validPerson);

        // Noto'g'ri yosh bilan record yaratishga harakat
        System.out.println("\\nTrying to create person with negative age...");
        try {
            Person invalidAge = new Person("Bob", -5, "bob@example.com");
        } catch (IllegalArgumentException e) {
            System.out.println("Error: " + e.getMessage());
        }

        // Noto'g'ri email bilan record yaratishga harakat
        System.out.println("\\nTrying to create person with invalid email...");
        try {
            Person invalidEmail = new Person("Charlie", 25, "invalid-email");
        } catch (IllegalArgumentException e) {
            System.out.println("Error: " + e.getMessage());
        }

        System.out.println("\\n=== Ma'lumotlarni normalizatsiya qilish ===");

        // Tartibsiz ma'lumotlar bilan record yaratish - ular normalizatsiya qilinadi
        String rawName = "  Bob  ";
        String rawEmail = "BOB@EXAMPLE.COM";
        String rawPhone = "123-456-7890";

        System.out.println("Input: \\"" + rawName + "\\", \\"" +
            rawEmail + "\\", \\"" + rawPhone + "\\"");

        Contact contact = new Contact(rawName, rawEmail, rawPhone);
        System.out.println("Normalized: " + contact);
    }
}`,
            description: `# Ixcham konstruktor va validatsiya

Recordlar maxsus ixcham konstruktor sintaksisini taqdim etadi, bu record yaratilishidan oldin ma'lumotlarni tekshirish yoki normalizatsiya qilish imkonini beradi. Oddiy konstruktorlardan farqli o'laroq, ixcham konstruktorlar parametrlar ro'yxati va tayinlash kodini takrorlamaydi.

## Talablar:
1. Ixcham konstruktorda validatsiya bilan record yarating:
   1. Yoshning musbat ekanligini tekshiring
   2. Emailda '@' borligini tekshiring
   3. Noto'g'ri ma'lumotlar uchun IllegalArgumentException tashlang

2. Ma'lumotlarni normalizatsiya qilish bilan boshqa record yarating:
   1. Satrlardan bo'sh joylarni olib tashlang
   2. Emailni kichik harflarga normalizatsiya qiling
   3. Telefon raqamini formatlang

3. Validatsiyaning ishlashini namoyish eting:
   1. Yaroqli recordlarni yarating
   2. Noto'g'ri recordlarni yaratishga harakat qiling (istisnolarni ushlang)

4. Validatsiyaning maydonlar tayinlanishidan oldin sodir bo'lishini ko'rsating

## Chiqish namunasi:
\`\`\`
=== Compact Constructor Validation ===
Valid person: Person[name=Alice, age=30, email=alice@example.com]

Trying to create person with negative age...
Error: Age must be positive

Trying to create person with invalid email...
Error: Email must contain @

=== Data Normalization ===
Input: "  Bob  ", "BOB@EXAMPLE.COM", "123-456-7890"
Normalized: Contact[name=Bob, email=bob@example.com, phone=1234567890]
\`\`\``,
            hint1: `Ixcham konstruktor sintaksisidan foydalaning: public RecordName { validatsiya/normalizatsiya kodi }. Parametrlarni sanab o'tish yoki maydonlarni tayinlash kerak emas - bu avtomatik bajariladi.`,
            hint2: `Ixcham konstruktorlarda parametrlarni validatsiya qilishingiz va maydonlar tayinlanishidan oldin istisnolar tashlashingiz mumkin. Shuningdek, ma'lumotlarni normalizatsiya qilish uchun parametrlarni qayta tayinlashingiz mumkin (masalan, name = name.trim()).`,
            whyItMatters: `Ixcham konstruktorlar recordlarda invariantlar va ma'lumotlar sifatini ta'minlashning toza usulini taqdim etadi. Ular noto'g'ri recordlar hech qachon yaratilmasligini kafolatlaydi va kodingizni yanada ishonchli qiladi. Konstruktorlarda ma'lumotlarni normalizatsiya qilish butun ilovada izchil ma'lumot formatini ta'minlaydi, xatolarni oldini oladi va taqqoslashlarni soddalashtiradi.

**Ishlab chiqarish patterni:**
\`\`\`java
record Money(BigDecimal amount, String currency) {
    public Money {
        if (amount.compareTo(BigDecimal.ZERO) < 0) {
            throw new IllegalArgumentException("Amount cannot be negative");
        }
        currency = currency.toUpperCase().trim();
    }
}
\`\`\`

**Amaliy foydalari:**
- Ob'ekt yaratish darajasida noto'g'ri ma'lumotlarning 100% oldini olish
- Butun tizimda ma'lumotlar bir xilligi`
        }
    }
};

export default task;
