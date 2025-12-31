import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-builder-pattern',
    title: 'Builder Pattern',
    difficulty: 'medium',
    tags: ['java', 'design-patterns', 'creational', 'builder', 'fluent-api'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Builder Pattern

The Builder pattern separates the construction of a complex object from its representation, allowing the same construction process to create different representations. It's particularly useful for objects with many optional parameters and provides a fluent API.

## Requirements:
1. Create a User class with many fields:
   1.1. Required: firstName, lastName, email
   1.2. Optional: age, phone, address, role

2. Implement Builder pattern:
   2.1. Static nested Builder class
   2.2. Fluent method chaining (return this)
   2.3. build() method that creates the User
   2.4. Validation in build() method

3. Demonstrate different user constructions:
   3.1. User with only required fields
   3.2. User with all fields
   3.3. User with selected optional fields

4. Show method chaining in action

## Example Output:
\`\`\`
=== Building Users ===
Minimal User:
Name: John Doe
Email: john@example.com
Age: Not specified
Phone: Not specified

Complete User:
Name: Jane Smith
Email: jane@example.com
Age: 28
Phone: +1-555-0123
Address: 123 Main St
Role: ADMIN

Custom User:
Name: Bob Wilson
Email: bob@example.com
Age: 35
Phone: Not specified
Address: Not specified
Role: USER
\`\`\``,
    initialCode: `// TODO: Create User class with Builder pattern

public class BuilderPattern {
    public static void main(String[] args) {
        // TODO: Build users with different configurations
    }
}`,
    solutionCode: `// User class with Builder pattern
class User {
    // Required parameters
    private final String firstName;
    private final String lastName;
    private final String email;

    // Optional parameters
    private final Integer age;
    private final String phone;
    private final String address;
    private final String role;

    // Private constructor - only accessible via Builder
    private User(Builder builder) {
        this.firstName = builder.firstName;
        this.lastName = builder.lastName;
        this.email = builder.email;
        this.age = builder.age;
        this.phone = builder.phone;
        this.address = builder.address;
        this.role = builder.role;
    }

    // Static nested Builder class
    public static class Builder {
        // Required parameters
        private final String firstName;
        private final String lastName;
        private final String email;

        // Optional parameters with default values
        private Integer age;
        private String phone;
        private String address;
        private String role = "USER";

        // Constructor with required parameters
        public Builder(String firstName, String lastName, String email) {
            if (firstName == null || lastName == null || email == null) {
                throw new IllegalArgumentException("Required fields cannot be null");
            }
            this.firstName = firstName;
            this.lastName = lastName;
            this.email = email;
        }

        // Fluent methods for optional parameters (return this for chaining)
        public Builder age(int age) {
            if (age < 0 || age > 150) {
                throw new IllegalArgumentException("Invalid age");
            }
            this.age = age;
            return this;
        }

        public Builder phone(String phone) {
            this.phone = phone;
            return this;
        }

        public Builder address(String address) {
            this.address = address;
            return this;
        }

        public Builder role(String role) {
            this.role = role;
            return this;
        }

        // Build method creates the User instance
        public User build() {
            return new User(this);
        }
    }

    @Override
    public String toString() {
        return "Name: " + firstName + " " + lastName + "\\n" +
               "Email: " + email + "\\n" +
               "Age: " + (age != null ? age : "Not specified") + "\\n" +
               "Phone: " + (phone != null ? phone : "Not specified") +
               (address != null ? "\\nAddress: " + address : "") +
               (role != null ? "\\nRole: " + role : "");
    }
}

public class BuilderPattern {
    public static void main(String[] args) {
        System.out.println("=== Building Users ===");

        // Build user with only required fields
        User minimalUser = new User.Builder("John", "Doe", "john@example.com")
            .build();
        System.out.println("Minimal User:");
        System.out.println(minimalUser);

        System.out.println("\\nComplete User:");
        // Build user with all fields using fluent API
        User completeUser = new User.Builder("Jane", "Smith", "jane@example.com")
            .age(28)
            .phone("+1-555-0123")
            .address("123 Main St")
            .role("ADMIN")
            .build();
        System.out.println(completeUser);

        System.out.println("\\nCustom User:");
        // Build user with selected optional fields
        User customUser = new User.Builder("Bob", "Wilson", "bob@example.com")
            .age(35)
            .role("USER")
            .build();
        System.out.println(customUser);
    }
}`,
    hint1: `Create a static nested Builder class inside User. The Builder constructor takes required parameters, and optional parameters are set via fluent methods that return 'this'.`,
    hint2: `Make User constructor private and accept Builder as parameter. Add a build() method in Builder that creates and returns new User(this).`,
    whyItMatters: `Builder pattern is essential for creating objects with many parameters, especially when many are optional. It eliminates telescoping constructors, makes code more readable, and allows immutable objects. The fluent API style is widely used in modern Java libraries like Stream API, and understanding it is crucial for writing clean, maintainable code.

**Production Pattern:**
\`\`\`java
// Building complex objects with fluent API
User user = new User.Builder("John", "Doe", "john@example.com")
    .age(28)
    .phone("+1-555-0123")
    .address("123 Main St")
    .role("ADMIN")
    .build();

// Validation in build() method
public User build() {
    if (age != null && (age < 0 || age > 150)) {
        throw new IllegalArgumentException("Invalid age");
    }
    return new User(this);
}
\`\`\`

**Practical Benefits:**
- Avoids telescoping constructors with many parameters
- Creates immutable objects (final fields)
- Fluent API makes code self-documenting
- Used in Stream API, StringBuilder, and many frameworks`,
    order: 2,
    testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

// Test1: Builder creates User with required fields
class Test1 {
    @Test
    void testRequiredFields() {
        User user = new User.Builder("John", "Doe", "john@example.com").build();
        assertNotNull(user);
    }
}

// Test2: User has correct name in toString
class Test2 {
    @Test
    void testUserName() {
        User user = new User.Builder("John", "Doe", "john@example.com").build();
        String str = user.toString();
        assertTrue(str.contains("John"));
        assertTrue(str.contains("Doe"));
    }
}

// Test3: Builder chain works for age
class Test3 {
    @Test
    void testAgeChain() {
        User user = new User.Builder("John", "Doe", "john@example.com")
            .age(25)
            .build();
        assertTrue(user.toString().contains("25"));
    }
}

// Test4: Builder chain works for phone
class Test4 {
    @Test
    void testPhoneChain() {
        User user = new User.Builder("John", "Doe", "john@example.com")
            .phone("+1-555-0123")
            .build();
        assertTrue(user.toString().contains("+1-555-0123"));
    }
}

// Test5: Builder chain works for all fields
class Test5 {
    @Test
    void testAllFields() {
        User user = new User.Builder("Jane", "Smith", "jane@example.com")
            .age(28)
            .phone("+1-555-0123")
            .address("123 Main St")
            .role("ADMIN")
            .build();
        String str = user.toString();
        assertTrue(str.contains("Jane"));
        assertTrue(str.contains("ADMIN"));
    }
}

// Test6: Null required fields throw exception
class Test6 {
    @Test
    void testNullRequiredFields() {
        assertThrows(IllegalArgumentException.class, () -> {
            new User.Builder(null, "Doe", "email@test.com");
        });
    }
}

// Test7: Invalid age throws exception
class Test7 {
    @Test
    void testInvalidAge() {
        assertThrows(IllegalArgumentException.class, () -> {
            new User.Builder("John", "Doe", "john@example.com")
                .age(-5)
                .build();
        });
    }
}

// Test8: Default role is USER
class Test8 {
    @Test
    void testDefaultRole() {
        User user = new User.Builder("John", "Doe", "john@example.com").build();
        assertTrue(user.toString().contains("USER"));
    }
}

// Test9: Builder returns itself for chaining
class Test9 {
    @Test
    void testBuilderReturnsThis() {
        User.Builder builder = new User.Builder("John", "Doe", "john@example.com");
        User.Builder result = builder.age(25);
        assertSame(builder, result);
    }
}

// Test10: Email is included in toString
class Test10 {
    @Test
    void testEmailIncluded() {
        User user = new User.Builder("John", "Doe", "john@example.com").build();
        assertTrue(user.toString().contains("john@example.com"));
    }
}
`,
    translations: {
        ru: {
            title: 'Паттерн Builder',
            solutionCode: `// Класс User с паттерном Builder
class User {
    // Обязательные параметры
    private final String firstName;
    private final String lastName;
    private final String email;

    // Необязательные параметры
    private final Integer age;
    private final String phone;
    private final String address;
    private final String role;

    // Приватный конструктор - доступен только через Builder
    private User(Builder builder) {
        this.firstName = builder.firstName;
        this.lastName = builder.lastName;
        this.email = builder.email;
        this.age = builder.age;
        this.phone = builder.phone;
        this.address = builder.address;
        this.role = builder.role;
    }

    // Статический вложенный класс Builder
    public static class Builder {
        // Обязательные параметры
        private final String firstName;
        private final String lastName;
        private final String email;

        // Необязательные параметры со значениями по умолчанию
        private Integer age;
        private String phone;
        private String address;
        private String role = "USER";

        // Конструктор с обязательными параметрами
        public Builder(String firstName, String lastName, String email) {
            if (firstName == null || lastName == null || email == null) {
                throw new IllegalArgumentException("Required fields cannot be null");
            }
            this.firstName = firstName;
            this.lastName = lastName;
            this.email = email;
        }

        // Fluent методы для необязательных параметров (возвращают this для цепочки)
        public Builder age(int age) {
            if (age < 0 || age > 150) {
                throw new IllegalArgumentException("Invalid age");
            }
            this.age = age;
            return this;
        }

        public Builder phone(String phone) {
            this.phone = phone;
            return this;
        }

        public Builder address(String address) {
            this.address = address;
            return this;
        }

        public Builder role(String role) {
            this.role = role;
            return this;
        }

        // Метод build создает экземпляр User
        public User build() {
            return new User(this);
        }
    }

    @Override
    public String toString() {
        return "Name: " + firstName + " " + lastName + "\\n" +
               "Email: " + email + "\\n" +
               "Age: " + (age != null ? age : "Not specified") + "\\n" +
               "Phone: " + (phone != null ? phone : "Not specified") +
               (address != null ? "\\nAddress: " + address : "") +
               (role != null ? "\\nRole: " + role : "");
    }
}

public class BuilderPattern {
    public static void main(String[] args) {
        System.out.println("=== Создание пользователей ===");

        // Создание пользователя только с обязательными полями
        User minimalUser = new User.Builder("John", "Doe", "john@example.com")
            .build();
        System.out.println("Minimal User:");
        System.out.println(minimalUser);

        System.out.println("\\nComplete User:");
        // Создание пользователя со всеми полями используя fluent API
        User completeUser = new User.Builder("Jane", "Smith", "jane@example.com")
            .age(28)
            .phone("+1-555-0123")
            .address("123 Main St")
            .role("ADMIN")
            .build();
        System.out.println(completeUser);

        System.out.println("\\nCustom User:");
        // Создание пользователя с выбранными необязательными полями
        User customUser = new User.Builder("Bob", "Wilson", "bob@example.com")
            .age(35)
            .role("USER")
            .build();
        System.out.println(customUser);
    }
}`,
            description: `# Паттерн Builder

Паттерн Builder отделяет конструирование сложного объекта от его представления, позволяя одному и тому же процессу конструирования создавать разные представления. Он особенно полезен для объектов с множеством необязательных параметров и предоставляет fluent API.

## Требования:
1. Создайте класс User с множеством полей:
   1.1. Обязательные: firstName, lastName, email
   1.2. Необязательные: age, phone, address, role

2. Реализуйте паттерн Builder:
   2.1. Статический вложенный класс Builder
   2.2. Цепочка fluent методов (return this)
   2.3. Метод build(), который создает User
   2.4. Валидация в методе build()

3. Продемонстрируйте различные конструкции пользователей:
   3.1. User только с обязательными полями
   3.2. User со всеми полями
   3.3. User с выбранными необязательными полями

4. Покажите цепочку методов в действии

## Пример вывода:
\`\`\`
=== Building Users ===
Minimal User:
Name: John Doe
Email: john@example.com
Age: Not specified
Phone: Not specified

Complete User:
Name: Jane Smith
Email: jane@example.com
Age: 28
Phone: +1-555-0123
Address: 123 Main St
Role: ADMIN

Custom User:
Name: Bob Wilson
Email: bob@example.com
Age: 35
Phone: Not specified
Address: Not specified
Role: USER
\`\`\``,
            hint1: `Создайте статический вложенный класс Builder внутри User. Конструктор Builder принимает обязательные параметры, а необязательные параметры устанавливаются через fluent методы, которые возвращают 'this'.`,
            hint2: `Сделайте конструктор User приватным и принимайте Builder как параметр. Добавьте метод build() в Builder, который создает и возвращает new User(this).`,
            whyItMatters: `Паттерн Builder необходим для создания объектов с множеством параметров, особенно когда многие из них необязательны. Он устраняет телескопические конструкторы, делает код более читаемым и позволяет создавать неизменяемые объекты.

**Продакшен паттерн:**
\`\`\`java
// Создание сложных объектов с fluent API
User user = new User.Builder("John", "Doe", "john@example.com")
    .age(28)
    .phone("+1-555-0123")
    .address("123 Main St")
    .role("ADMIN")
    .build();

// Валидация в build() методе
public User build() {
    if (age != null && (age < 0 || age > 150)) {
        throw new IllegalArgumentException("Invalid age");
    }
    return new User(this);
}
\`\`\`

**Практические преимущества:**
- Избегает телескопических конструкторов с множеством параметров
- Создает неизменяемые объекты (final поля)
- Fluent API делает код самодокументирующимся
- Используется в Stream API, StringBuilder и многих фреймворках`
        },
        uz: {
            title: `Builder namunasi`,
            solutionCode: `// Builder namunasi bilan User klassi
class User {
    // Majburiy parametrlar
    private final String firstName;
    private final String lastName;
    private final String email;

    // Ixtiyoriy parametrlar
    private final Integer age;
    private final String phone;
    private final String address;
    private final String role;

    // Xususiy konstruktor - faqat Builder orqali kirish mumkin
    private User(Builder builder) {
        this.firstName = builder.firstName;
        this.lastName = builder.lastName;
        this.email = builder.email;
        this.age = builder.age;
        this.phone = builder.phone;
        this.address = builder.address;
        this.role = builder.role;
    }

    // Statik ichki Builder klassi
    public static class Builder {
        // Majburiy parametrlar
        private final String firstName;
        private final String lastName;
        private final String email;

        // Standart qiymatlari bilan ixtiyoriy parametrlar
        private Integer age;
        private String phone;
        private String address;
        private String role = "USER";

        // Majburiy parametrlar bilan konstruktor
        public Builder(String firstName, String lastName, String email) {
            if (firstName == null || lastName == null || email == null) {
                throw new IllegalArgumentException("Required fields cannot be null");
            }
            this.firstName = firstName;
            this.lastName = lastName;
            this.email = email;
        }

        // Ixtiyoriy parametrlar uchun fluent metodlar (zanjir uchun this qaytaradi)
        public Builder age(int age) {
            if (age < 0 || age > 150) {
                throw new IllegalArgumentException("Invalid age");
            }
            this.age = age;
            return this;
        }

        public Builder phone(String phone) {
            this.phone = phone;
            return this;
        }

        public Builder address(String address) {
            this.address = address;
            return this;
        }

        public Builder role(String role) {
            this.role = role;
            return this;
        }

        // Build metodi User misolini yaratadi
        public User build() {
            return new User(this);
        }
    }

    @Override
    public String toString() {
        return "Name: " + firstName + " " + lastName + "\\n" +
               "Email: " + email + "\\n" +
               "Age: " + (age != null ? age : "Not specified") + "\\n" +
               "Phone: " + (phone != null ? phone : "Not specified") +
               (address != null ? "\\nAddress: " + address : "") +
               (role != null ? "\\nRole: " + role : "");
    }
}

public class BuilderPattern {
    public static void main(String[] args) {
        System.out.println("=== Foydalanuvchilarni yaratish ===");

        // Faqat majburiy maydonlar bilan foydalanuvchi yaratish
        User minimalUser = new User.Builder("John", "Doe", "john@example.com")
            .build();
        System.out.println("Minimal User:");
        System.out.println(minimalUser);

        System.out.println("\\nComplete User:");
        // Fluent API dan foydalanib barcha maydonlar bilan foydalanuvchi yaratish
        User completeUser = new User.Builder("Jane", "Smith", "jane@example.com")
            .age(28)
            .phone("+1-555-0123")
            .address("123 Main St")
            .role("ADMIN")
            .build();
        System.out.println(completeUser);

        System.out.println("\\nCustom User:");
        // Tanlangan ixtiyoriy maydonlar bilan foydalanuvchi yaratish
        User customUser = new User.Builder("Bob", "Wilson", "bob@example.com")
            .age(35)
            .role("USER")
            .build();
        System.out.println(customUser);
    }
}`,
            description: `# Builder namunasi

Builder namunasi murakkab obyektni yaratishni uning vakilligidan ajratadi va bir xil yaratish jarayoni turli vakilliklarni yaratishga imkon beradi. U ko'plab ixtiyoriy parametrlarga ega obyektlar uchun ayniqsa foydali va fluent API taqdim etadi.

## Talablar:
1. Ko'plab maydonlar bilan User klassini yarating:
   1.1. Majburiy: firstName, lastName, email
   1.2. Ixtiyoriy: age, phone, address, role

2. Builder namunasini amalga oshiring:
   2.1. Statik ichki Builder klassi
   2.2. Fluent metod zanjiri (return this)
   2.3. User yaratadigan build() metodi
   2.4. build() metodida validatsiya

3. Turli foydalanuvchi konstruksiyalarini namoyish eting:
   3.1. Faqat majburiy maydonlar bilan User
   3.2. Barcha maydonlar bilan User
   3.3. Tanlangan ixtiyoriy maydonlar bilan User

4. Metod zanjirini amalda ko'rsating

## Chiqish namunasi:
\`\`\`
=== Building Users ===
Minimal User:
Name: John Doe
Email: john@example.com
Age: Not specified
Phone: Not specified

Complete User:
Name: Jane Smith
Email: jane@example.com
Age: 28
Phone: +1-555-0123
Address: 123 Main St
Role: ADMIN

Custom User:
Name: Bob Wilson
Email: bob@example.com
Age: 35
Phone: Not specified
Address: Not specified
Role: USER
\`\`\``,
            hint1: `User ichida statik ichki Builder klassini yarating. Builder konstruktori majburiy parametrlarni qabul qiladi va ixtiyoriy parametrlar 'this' qaytaradigan fluent metodlar orqali o'rnatiladi.`,
            hint2: `User konstruktorini xususiy qiling va Builder-ni parametr sifatida qabul qiling. Builder-da new User(this) yaratuvchi va qaytaruvchi build() metodini qo'shing.`,
            whyItMatters: `Builder namunasi ko'plab parametrlarga ega obyektlarni yaratish uchun zarur, ayniqsa ko'pchilik ixtiyoriy bo'lganda. U teleskopik konstruktorlarni yo'q qiladi, kodni o'qilishi osonroq qiladi va o'zgarmas obyektlarni yaratishga imkon beradi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Fluent API bilan murakkab obyektlarni yaratish
User user = new User.Builder("John", "Doe", "john@example.com")
    .age(28)
    .phone("+1-555-0123")
    .address("123 Main St")
    .role("ADMIN")
    .build();

// build() metodida validatsiya
public User build() {
    if (age != null && (age < 0 || age > 150)) {
        throw new IllegalArgumentException("Invalid age");
    }
    return new User(this);
}
\`\`\`

**Amaliy foydalari:**
- Ko'plab parametrlar bilan teleskopik konstruktorlardan qochadi
- O'zgarmas obyektlar yaratadi (final maydonlar)
- Fluent API kodni o'z-o'zini hujjatlashtiradigan qiladi
- Stream API, StringBuilder va ko'plab freymvorklarda ishlatiladi`
        }
    }
};

export default task;
