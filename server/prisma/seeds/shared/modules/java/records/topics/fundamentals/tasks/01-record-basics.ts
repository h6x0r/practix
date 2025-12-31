import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-record-basics',
    title: 'Record Declaration and Basics',
    difficulty: 'easy',
    tags: ['java', 'records', 'java16', 'immutable', 'data-class'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Record Declaration and Basics

Records are a special kind of class in Java (introduced in Java 16) designed to model immutable data. A record automatically generates a constructor, getters, equals(), hashCode(), and toString() methods based on its components.

## Requirements:
1. Create a simple record with multiple components:
   1. String name
   2. int age
   3. String email

2. Demonstrate auto-generated methods:
   1. toString() - prints record in a readable format
   2. equals() - compares records by value
   3. hashCode() - generates hash based on components
   4. Accessor methods (name(), age(), email())

3. Create multiple instances and compare them

4. Show that records are immutable (no setters)

## Example Output:
\`\`\`
=== Record Basics ===
Person record: Person[name=Alice, age=30, email=alice@example.com]
Name accessor: Alice
Age accessor: 30
Email accessor: alice@example.com

=== Auto-generated Methods ===
person1: Person[name=Bob, age=25, email=bob@example.com]
person2: Person[name=Bob, age=25, email=bob@example.com]
person3: Person[name=Charlie, age=30, email=charlie@example.com]

Are person1 and person2 equal? true
Are person1 and person3 equal? false

person1 hash code: 123456789
person2 hash code: 123456789
person3 hash code: 987654321
\`\`\``,
    initialCode: `// TODO: Create a Person record with name, age, and email components

public class RecordBasics {
    public static void main(String[] args) {
        // TODO: Create record instances

        // TODO: Demonstrate accessor methods

        // TODO: Demonstrate equals() and hashCode()

        // Note: Records are immutable - no setters are generated
    }
}`,
    solutionCode: `// Record declaration - concise and immutable
record Person(String name, int age, String email) {}

public class RecordBasics {
    public static void main(String[] args) {
        System.out.println("=== Record Basics ===");

        // Create a record instance using the canonical constructor
        Person person = new Person("Alice", 30, "alice@example.com");

        // Auto-generated toString() method
        System.out.println("Person record: " + person);

        // Auto-generated accessor methods (not getters - just component names)
        System.out.println("Name accessor: " + person.name());
        System.out.println("Age accessor: " + person.age());
        System.out.println("Email accessor: " + person.email());

        System.out.println("\\n=== Auto-generated Methods ===");

        // Create multiple instances for comparison
        Person person1 = new Person("Bob", 25, "bob@example.com");
        Person person2 = new Person("Bob", 25, "bob@example.com");
        Person person3 = new Person("Charlie", 30, "charlie@example.com");

        System.out.println("person1: " + person1);
        System.out.println("person2: " + person2);
        System.out.println("person3: " + person3);

        // Auto-generated equals() - compares by value
        System.out.println("\\nAre person1 and person2 equal? " +
            person1.equals(person2));
        System.out.println("Are person1 and person3 equal? " +
            person1.equals(person3));

        // Auto-generated hashCode() - consistent with equals()
        System.out.println("\\nperson1 hash code: " + person1.hashCode());
        System.out.println("person2 hash code: " + person2.hashCode());
        System.out.println("person3 hash code: " + person3.hashCode());
    }
}`,
    hint1: `Declare a record using: record PersonName(Type field1, Type field2) {}. Records automatically generate constructor, accessors, equals, hashCode, and toString.`,
    hint2: `Access record components using their names as methods: person.name(), person.age(). Records are immutable - you cannot modify their fields after creation.`,
    whyItMatters: `Records eliminate boilerplate code for immutable data classes. Instead of writing dozens of lines for constructors, getters, equals, hashCode, and toString, you get all of this automatically. Records make code more concise, readable, and less error-prone, especially for DTOs and value objects.

**Production Pattern:**
\`\`\`java
record UserDTO(Long id, String username, String email, LocalDateTime createdAt) {}

// In controller:
public UserDTO getUser(Long id) {
    User user = userRepository.findById(id);
    return new UserDTO(user.getId(), user.getUsername(),
                       user.getEmail(), user.getCreatedAt());
}
\`\`\`

**Practical Benefits:**
- Reduces DTO code by 70%
- Automatic correct equals/hashCode implementation for caching`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Test1: Verify record creation and toString
class Test1 {
    @Test
    public void testRecordCreation() {
        Person person = new Person("Alice", 30, "alice@example.com");
        assertNotNull(person);
        String str = person.toString();
        assertTrue(str.contains("Alice"));
        assertTrue(str.contains("30"));
        assertTrue(str.contains("alice@example.com"));
    }
}

// Test2: Verify accessor methods
class Test2 {
    @Test
    public void testAccessors() {
        Person person = new Person("Bob", 25, "bob@example.com");
        assertEquals("Bob", person.name());
        assertEquals(25, person.age());
        assertEquals("bob@example.com", person.email());
    }
}

// Test3: Verify equals method with same values
class Test3 {
    @Test
    public void testEqualsWithSameValues() {
        Person person1 = new Person("Charlie", 35, "charlie@example.com");
        Person person2 = new Person("Charlie", 35, "charlie@example.com");
        assertEquals(person1, person2);
    }
}

// Test4: Verify equals method with different values
class Test4 {
    @Test
    public void testEqualsWithDifferentValues() {
        Person person1 = new Person("David", 40, "david@example.com");
        Person person2 = new Person("Eve", 28, "eve@example.com");
        assertNotEquals(person1, person2);
    }
}

// Test5: Verify hashCode consistency
class Test5 {
    @Test
    public void testHashCodeConsistency() {
        Person person1 = new Person("Frank", 33, "frank@example.com");
        Person person2 = new Person("Frank", 33, "frank@example.com");
        assertEquals(person1.hashCode(), person2.hashCode());
    }
}

// Test6: Verify hashCode different for different values
class Test6 {
    @Test
    public void testHashCodeDifferentValues() {
        Person person1 = new Person("Grace", 29, "grace@example.com");
        Person person2 = new Person("Henry", 31, "henry@example.com");
        assertNotEquals(person1.hashCode(), person2.hashCode());
    }
}

// Test7: Verify immutability (no setters)
class Test7 {
    @Test
    public void testImmutability() {
        Person person = new Person("Ivy", 27, "ivy@example.com");
        // Record fields are final and cannot be modified
        assertEquals("Ivy", person.name());
        assertEquals(27, person.age());
        assertEquals("ivy@example.com", person.email());
    }
}

// Test8: Verify multiple instances independence
class Test8 {
    @Test
    public void testMultipleInstances() {
        Person person1 = new Person("Jack", 45, "jack@example.com");
        Person person2 = new Person("Kate", 38, "kate@example.com");
        assertNotEquals(person1, person2);
        assertNotEquals(person1.name(), person2.name());
    }
}

// Test9: Verify equals with null
class Test9 {
    @Test
    public void testEqualsWithNull() {
        Person person = new Person("Leo", 32, "leo@example.com");
        assertNotEquals(null, person);
    }
}

// Test10: Verify equals reflexivity
class Test10 {
    @Test
    public void testEqualsReflexivity() {
        Person person = new Person("Mike", 41, "mike@example.com");
        assertEquals(person, person);
    }
}`,
    order: 1,
    translations: {
        ru: {
            title: 'Объявление и основы записей',
            solutionCode: `// Объявление record - краткое и неизменяемое
record Person(String name, int age, String email) {}

public class RecordBasics {
    public static void main(String[] args) {
        System.out.println("=== Основы Record ===");

        // Создание экземпляра record с помощью канонического конструктора
        Person person = new Person("Alice", 30, "alice@example.com");

        // Автогенерированный метод toString()
        System.out.println("Person record: " + person);

        // Автогенерированные методы доступа (не геттеры - просто имена компонентов)
        System.out.println("Name accessor: " + person.name());
        System.out.println("Age accessor: " + person.age());
        System.out.println("Email accessor: " + person.email());

        System.out.println("\\n=== Автогенерированные методы ===");

        // Создание нескольких экземпляров для сравнения
        Person person1 = new Person("Bob", 25, "bob@example.com");
        Person person2 = new Person("Bob", 25, "bob@example.com");
        Person person3 = new Person("Charlie", 30, "charlie@example.com");

        System.out.println("person1: " + person1);
        System.out.println("person2: " + person2);
        System.out.println("person3: " + person3);

        // Автогенерированный equals() - сравнение по значению
        System.out.println("\\nAre person1 and person2 equal? " +
            person1.equals(person2));
        System.out.println("Are person1 and person3 equal? " +
            person1.equals(person3));

        // Автогенерированный hashCode() - согласован с equals()
        System.out.println("\\nperson1 hash code: " + person1.hashCode());
        System.out.println("person2 hash code: " + person2.hashCode());
        System.out.println("person3 hash code: " + person3.hashCode());
    }
}`,
            description: `# Объявление и основы записей

Records - это особый вид класса в Java (введен в Java 16), предназначенный для моделирования неизменяемых данных. Record автоматически генерирует конструктор, геттеры, методы equals(), hashCode() и toString() на основе своих компонентов.

## Требования:
1. Создайте простой record с несколькими компонентами:
   1. String name
   2. int age
   3. String email

2. Продемонстрируйте автогенерированные методы:
   1. toString() - выводит record в читаемом формате
   2. equals() - сравнивает records по значению
   3. hashCode() - генерирует хеш на основе компонентов
   4. Методы доступа (name(), age(), email())

3. Создайте несколько экземпляров и сравните их

4. Покажите, что records неизменяемы (нет сеттеров)

## Пример вывода:
\`\`\`
=== Record Basics ===
Person record: Person[name=Alice, age=30, email=alice@example.com]
Name accessor: Alice
Age accessor: 30
Email accessor: alice@example.com

=== Auto-generated Methods ===
person1: Person[name=Bob, age=25, email=bob@example.com]
person2: Person[name=Bob, age=25, email=bob@example.com]
person3: Person[name=Charlie, age=30, email=charlie@example.com]

Are person1 and person2 equal? true
Are person1 and person3 equal? false

person1 hash code: 123456789
person2 hash code: 123456789
person3 hash code: 987654321
\`\`\``,
            hint1: `Объявите record используя: record PersonName(Type field1, Type field2) {}. Records автоматически генерируют конструктор, методы доступа, equals, hashCode и toString.`,
            hint2: `Получайте доступ к компонентам record используя их имена как методы: person.name(), person.age(). Records неизменяемы - вы не можете изменить их поля после создания.`,
            whyItMatters: `Records устраняют шаблонный код для неизменяемых классов данных. Вместо написания десятков строк для конструкторов, геттеров, equals, hashCode и toString, вы получаете все это автоматически. Records делают код более кратким, читаемым и менее подверженным ошибкам, особенно для DTO и объектов-значений.

**Продакшен паттерн:**
\`\`\`java
record UserDTO(Long id, String username, String email, LocalDateTime createdAt) {}

// В контроллере:
public UserDTO getUser(Long id) {
    User user = userRepository.findById(id);
    return new UserDTO(user.getId(), user.getUsername(),
                       user.getEmail(), user.getCreatedAt());
}
\`\`\`

**Практические преимущества:**
- Сокращение кода DTO на 70%
- Автоматическая корректная реализация equals/hashCode для кэширования`
        },
        uz: {
            title: `Record e'lon qilish va asoslari`,
            solutionCode: `// Record e'lon qilish - qisqa va o'zgarmas
record Person(String name, int age, String email) {}

public class RecordBasics {
    public static void main(String[] args) {
        System.out.println("=== Record asoslari ===");

        // Kanonik konstruktor yordamida record namunasini yaratish
        Person person = new Person("Alice", 30, "alice@example.com");

        // Avtomatik yaratilgan toString() metodi
        System.out.println("Person record: " + person);

        // Avtomatik yaratilgan kirish metodlari (getterlar emas - faqat komponent nomlari)
        System.out.println("Name accessor: " + person.name());
        System.out.println("Age accessor: " + person.age());
        System.out.println("Email accessor: " + person.email());

        System.out.println("\\n=== Avtomatik yaratilgan metodlar ===");

        // Taqqoslash uchun bir nechta namunalarni yaratish
        Person person1 = new Person("Bob", 25, "bob@example.com");
        Person person2 = new Person("Bob", 25, "bob@example.com");
        Person person3 = new Person("Charlie", 30, "charlie@example.com");

        System.out.println("person1: " + person1);
        System.out.println("person2: " + person2);
        System.out.println("person3: " + person3);

        // Avtomatik yaratilgan equals() - qiymat bo'yicha taqqoslash
        System.out.println("\\nAre person1 and person2 equal? " +
            person1.equals(person2));
        System.out.println("Are person1 and person3 equal? " +
            person1.equals(person3));

        // Avtomatik yaratilgan hashCode() - equals() bilan mos
        System.out.println("\\nperson1 hash code: " + person1.hashCode());
        System.out.println("person2 hash code: " + person2.hashCode());
        System.out.println("person3 hash code: " + person3.hashCode());
    }
}`,
            description: `# Record e'lon qilish va asoslari

Recordlar Java-da maxsus klass turi (Java 16-da kiritilgan) bo'lib, o'zgarmas ma'lumotlarni modellashtirish uchun mo'ljallangan. Record avtomatik ravishda konstruktor, getterlar, equals(), hashCode() va toString() metodlarini o'z komponentlari asosida yaratadi.

## Talablar:
1. Bir nechta komponentli oddiy record yarating:
   1. String name
   2. int age
   3. String email

2. Avtomatik yaratilgan metodlarni namoyish eting:
   1. toString() - recordni o'qiladigan formatda chiqaradi
   2. equals() - recordlarni qiymat bo'yicha taqqoslaydi
   3. hashCode() - komponentlar asosida xesh yaratadi
   4. Kirish metodlari (name(), age(), email())

3. Bir nechta namunalarni yarating va ularni taqqoslang

4. Recordlar o'zgarmasligini ko'rsating (setterlar yo'q)

## Chiqish namunasi:
\`\`\`
=== Record Basics ===
Person record: Person[name=Alice, age=30, email=alice@example.com]
Name accessor: Alice
Age accessor: 30
Email accessor: alice@example.com

=== Auto-generated Methods ===
person1: Person[name=Bob, age=25, email=bob@example.com]
person2: Person[name=Bob, age=25, email=bob@example.com]
person3: Person[name=Charlie, age=30, email=charlie@example.com]

Are person1 and person2 equal? true
Are person1 and person3 equal? false

person1 hash code: 123456789
person2 hash code: 123456789
person3 hash code: 987654321
\`\`\``,
            hint1: `Recordni e'lon qiling: record PersonName(Type field1, Type field2) {}. Recordlar avtomatik ravishda konstruktor, kirish metodlari, equals, hashCode va toString yaratadi.`,
            hint2: `Record komponentlariga ularning nomlarini metodlar sifatida ishlatib kiring: person.name(), person.age(). Recordlar o'zgarmas - siz ularning maydonlarini yaratilgandan keyin o'zgartira olmaysiz.`,
            whyItMatters: `Recordlar o'zgarmas ma'lumot klasslari uchun shablon kodini yo'q qiladi. Konstruktorlar, getterlar, equals, hashCode va toString uchun o'nlab qatorlar yozish o'rniga, bularning barchasini avtomatik ravishda olasiz. Recordlar kodni qisqaroq, o'qilishi osonroq va xatolarga kamroq moyil qiladi, ayniqsa DTO va qiymat ob'ektlari uchun.

**Ishlab chiqarish patterni:**
\`\`\`java
record UserDTO(Long id, String username, String email, LocalDateTime createdAt) {}

// Kontrollerda:
public UserDTO getUser(Long id) {
    User user = userRepository.findById(id);
    return new UserDTO(user.getId(), user.getUsername(),
                       user.getEmail(), user.getCreatedAt());
}
\`\`\`

**Amaliy foydalari:**
- DTO kodini 70% qisqartiradi
- Keshlash uchun avtomatik to'g'ri equals/hashCode amalga oshirish`
        }
    }
};

export default task;
