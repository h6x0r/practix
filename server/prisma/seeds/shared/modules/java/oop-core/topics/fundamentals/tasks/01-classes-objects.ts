import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-oop-classes-objects',
    title: 'Classes and Objects',
    difficulty: 'easy',
    tags: ['java', 'oop', 'classes', 'objects', 'static'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Create a **Person** class that demonstrates the fundamental concepts of classes and objects in Java.

**Requirements:**
1. Create a Person class with the following instance fields:
   1.1. name (String)
   1.2. age (int)
   1.3. email (String)

2. Add a static field to track the total number of Person objects created:
   2.1. static int personCount

3. Implement the following methods:
   3.1. A method to display person information: displayInfo()
   3.2. A static method to get the total person count: getPersonCount()
   3.3. An instance method to update email: updateEmail(String newEmail)

4. In the main method:
   4.1. Create at least 3 Person objects
   4.2. Call displayInfo() on each person
   4.3. Update email for one person
   4.4. Display the total count of persons created

**Learning Goals:**
- Understand the difference between classes and objects
- Learn about instance vs static members
- Practice creating and using objects`,
    initialCode: `public class Person {
    // TODO: Add instance fields (name, age, email)

    // TODO: Add static field for person count

    // TODO: Implement displayInfo() method

    // TODO: Implement static getPersonCount() method

    // TODO: Implement updateEmail() method

    public static void main(String[] args) {
        // TODO: Create Person objects and test the class
    }
}`,
    solutionCode: `public class Person {
    // Instance fields - each object has its own copy
    private String name;
    private int age;
    private String email;

    // Static field - shared across all Person objects
    private static int personCount = 0;

    // Constructor
    public Person(String name, int age, String email) {
        this.name = name;
        this.age = age;
        this.email = email;
        personCount++; // Increment count when new Person is created
    }

    // Instance method - operates on a specific object
    public void displayInfo() {
        System.out.println("Name: " + name);
        System.out.println("Age: " + age);
        System.out.println("Email: " + email);
        System.out.println("---");
    }

    // Static method - belongs to the class, not objects
    public static int getPersonCount() {
        return personCount;
    }

    // Instance method to update email
    public void updateEmail(String newEmail) {
        this.email = newEmail;
        System.out.println("Email updated for " + name);
    }

    public static void main(String[] args) {
        // Creating objects - each is an instance of Person class
        Person person1 = new Person("Alice Johnson", 28, "alice@email.com");
        Person person2 = new Person("Bob Smith", 35, "bob@email.com");
        Person person3 = new Person("Carol Williams", 42, "carol@email.com");

        // Calling instance methods on specific objects
        person1.displayInfo();
        person2.displayInfo();
        person3.displayInfo();

        // Update email for one person
        person1.updateEmail("alice.johnson@newemail.com");
        person1.displayInfo();

        // Calling static method - use class name
        System.out.println("Total persons created: " + Person.getPersonCount());
    }
}`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.lang.reflect.*;

// Test1: Verify Person class has required instance fields
class Test1 {
    @Test
    public void test() throws Exception {
        Class<?> cls = Class.forName("Person");
        Field nameField = cls.getDeclaredField("name");
        Field ageField = cls.getDeclaredField("age");
        Field emailField = cls.getDeclaredField("email");
        assertNotNull("Person should have name field", nameField);
        assertNotNull("Person should have age field", ageField);
        assertNotNull("Person should have email field", emailField);
    }
}

// Test2: Verify Person class has static personCount field
class Test2 {
    @Test
    public void test() throws Exception {
        Class<?> cls = Class.forName("Person");
        Field countField = cls.getDeclaredField("personCount");
        assertTrue("personCount should be static", Modifier.isStatic(countField.getModifiers()));
    }
}

// Test3: Verify Person constructor sets fields correctly
class Test3 {
    @Test
    public void test() throws Exception {
        Class<?> cls = Class.forName("Person");
        Constructor<?> constructor = cls.getConstructor(String.class, int.class, String.class);
        Object person = constructor.newInstance("Test", 25, "test@email.com");

        Field nameField = cls.getDeclaredField("name");
        nameField.setAccessible(true);
        assertEquals("Name should be set", "Test", nameField.get(person));

        Field ageField = cls.getDeclaredField("age");
        ageField.setAccessible(true);
        assertEquals("Age should be set", 25, ageField.get(person));
    }
}

// Test4: Verify displayInfo method exists
class Test4 {
    @Test
    public void test() throws Exception {
        Class<?> cls = Class.forName("Person");
        Method method = cls.getMethod("displayInfo");
        assertNotNull("displayInfo method should exist", method);
    }
}

// Test5: Verify getPersonCount static method exists and works
class Test5 {
    @Test
    public void test() throws Exception {
        Class<?> cls = Class.forName("Person");
        Method method = cls.getMethod("getPersonCount");
        assertTrue("getPersonCount should be static", Modifier.isStatic(method.getModifiers()));
        Object count = method.invoke(null);
        assertTrue("getPersonCount should return an int", count instanceof Integer);
    }
}

// Test6: Verify updateEmail method exists and works
class Test6 {
    @Test
    public void test() throws Exception {
        Class<?> cls = Class.forName("Person");
        Constructor<?> constructor = cls.getConstructor(String.class, int.class, String.class);
        Object person = constructor.newInstance("Test", 25, "old@email.com");

        Method updateEmail = cls.getMethod("updateEmail", String.class);
        updateEmail.invoke(person, "new@email.com");

        Field emailField = cls.getDeclaredField("email");
        emailField.setAccessible(true);
        assertEquals("Email should be updated", "new@email.com", emailField.get(person));
    }
}

// Test7: Verify personCount increments with each new Person
class Test7 {
    @Test
    public void test() throws Exception {
        Class<?> cls = Class.forName("Person");
        Field countField = cls.getDeclaredField("personCount");
        countField.setAccessible(true);

        int initialCount = (int) countField.get(null);
        Constructor<?> constructor = cls.getConstructor(String.class, int.class, String.class);
        constructor.newInstance("Test1", 25, "test1@email.com");
        int newCount = (int) countField.get(null);

        assertEquals("personCount should increment", initialCount + 1, newCount);
    }
}

// Test8: Verify multiple Person objects can be created
class Test8 {
    @Test
    public void test() throws Exception {
        Class<?> cls = Class.forName("Person");
        Constructor<?> constructor = cls.getConstructor(String.class, int.class, String.class);

        Object person1 = constructor.newInstance("Alice", 28, "alice@email.com");
        Object person2 = constructor.newInstance("Bob", 35, "bob@email.com");
        Object person3 = constructor.newInstance("Carol", 42, "carol@email.com");

        assertNotNull("Person1 should be created", person1);
        assertNotNull("Person2 should be created", person2);
        assertNotNull("Person3 should be created", person3);
    }
}

// Test9: Verify each object has independent instance fields
class Test9 {
    @Test
    public void test() throws Exception {
        Class<?> cls = Class.forName("Person");
        Constructor<?> constructor = cls.getConstructor(String.class, int.class, String.class);

        Object person1 = constructor.newInstance("Alice", 28, "alice@email.com");
        Object person2 = constructor.newInstance("Bob", 35, "bob@email.com");

        Field nameField = cls.getDeclaredField("name");
        nameField.setAccessible(true);

        assertEquals("Person1 name", "Alice", nameField.get(person1));
        assertEquals("Person2 name", "Bob", nameField.get(person2));
    }
}

// Test10: Verify static method can be called without instance
class Test10 {
    @Test
    public void test() throws Exception {
        Class<?> cls = Class.forName("Person");
        Method getPersonCount = cls.getMethod("getPersonCount");
        Object count = getPersonCount.invoke(null);
        assertTrue("Should return count without creating instance", count instanceof Integer);
    }
}`,
    hint1: `Start by declaring instance variables at the top of the class. Remember that each object will have its own copy of these variables.`,
    hint2: `Static variables belong to the class itself and are shared by all objects. Initialize personCount to 0 and increment it in the constructor.`,
    whyItMatters: `Understanding classes and objects is fundamental to Java programming. Classes are blueprints that define the structure and behavior, while objects are actual instances created from those blueprints. The distinction between instance and static members is crucial for proper object-oriented design.

**Production Pattern:**
\`\`\`java
public class Person {
    // Instance fields - each object has its own copy
    private String name;
    private int age;

    // Static field - shared across all objects
    private static int personCount = 0;

    public Person(String name, int age) {
        this.name = name;
        this.age = age;
        personCount++; // Track number of objects created
    }

    public static int getPersonCount() {
        return personCount; // Static method to access static field
    }
}
\`\`\`

**Practical Benefits:**
- Instance fields allow each object to store unique state
- Static fields are ideal for shared data (counters, constants)
- Static methods can be called without creating an object (Person.getPersonCount())
- Encapsulation through access modifiers protects data from misuse`,
    order: 0,
    translations: {
        ru: {
            title: 'Классы и Объекты',
            solutionCode: `public class Person {
    // Поля экземпляра - каждый объект имеет свою копию
    private String name;
    private int age;
    private String email;

    // Статическое поле - общее для всех объектов Person
    private static int personCount = 0;

    // Конструктор
    public Person(String name, int age, String email) {
        this.name = name;
        this.age = age;
        this.email = email;
        personCount++; // Увеличиваем счетчик при создании нового Person
    }

    // Метод экземпляра - работает с конкретным объектом
    public void displayInfo() {
        System.out.println("Имя: " + name);
        System.out.println("Возраст: " + age);
        System.out.println("Email: " + email);
        System.out.println("---");
    }

    // Статический метод - принадлежит классу, а не объектам
    public static int getPersonCount() {
        return personCount;
    }

    // Метод экземпляра для обновления email
    public void updateEmail(String newEmail) {
        this.email = newEmail;
        System.out.println("Email обновлен для " + name);
    }

    public static void main(String[] args) {
        // Создание объектов - каждый является экземпляром класса Person
        Person person1 = new Person("Алиса Джонсон", 28, "alice@email.com");
        Person person2 = new Person("Боб Смит", 35, "bob@email.com");
        Person person3 = new Person("Кэрол Уильямс", 42, "carol@email.com");

        // Вызов методов экземпляра для конкретных объектов
        person1.displayInfo();
        person2.displayInfo();
        person3.displayInfo();

        // Обновление email для одного человека
        person1.updateEmail("alice.johnson@newemail.com");
        person1.displayInfo();

        // Вызов статического метода - используем имя класса
        System.out.println("Всего создано людей: " + Person.getPersonCount());
    }
}`,
            description: `Создайте класс **Person**, который демонстрирует фундаментальные концепции классов и объектов в Java.

**Требования:**
1. Создайте класс Person со следующими полями экземпляра:
   1.1. name (String)
   1.2. age (int)
   1.3. email (String)

2. Добавьте статическое поле для отслеживания общего количества созданных объектов Person:
   2.1. static int personCount

3. Реализуйте следующие методы:
   3.1. Метод для отображения информации о человеке: displayInfo()
   3.2. Статический метод для получения общего количества людей: getPersonCount()
   3.3. Метод экземпляра для обновления email: updateEmail(String newEmail)

4. В методе main:
   4.1. Создайте минимум 3 объекта Person
   4.2. Вызовите displayInfo() для каждого человека
   4.3. Обновите email для одного человека
   4.4. Отобразите общее количество созданных людей

**Цели обучения:**
- Понять разницу между классами и объектами
- Изучить разницу между членами экземпляра и статическими членами
- Практиковаться в создании и использовании объектов`,
            hint1: `Начните с объявления переменных экземпляра в верхней части класса. Помните, что каждый объект будет иметь свою копию этих переменных.`,
            hint2: `Статические переменные принадлежат самому классу и являются общими для всех объектов. Инициализируйте personCount значением 0 и увеличивайте его в конструкторе.`,
            whyItMatters: `Понимание классов и объектов является основой программирования на Java. Классы - это чертежи, которые определяют структуру и поведение, а объекты - это фактические экземпляры, созданные из этих чертежей. Различие между членами экземпляра и статическими членами имеет решающее значение для правильного объектно-ориентированного проектирования.

**Продакшен паттерн:**
\`\`\`java
public class Person {
    // Поля экземпляра - каждый объект имеет свою копию
    private String name;
    private int age;

    // Статическое поле - общее для всех объектов
    private static int personCount = 0;

    public Person(String name, int age) {
        this.name = name;
        this.age = age;
        personCount++; // Отслеживаем количество созданных объектов
    }

    public static int getPersonCount() {
        return personCount; // Статический метод для доступа к статическому полю
    }
}
\`\`\`

**Практические преимущества:**
- Поля экземпляра позволяют каждому объекту хранить уникальное состояние
- Статические поля идеально подходят для общих данных (счетчики, константы)
- Статические методы можно вызывать без создания объекта (Person.getPersonCount())
- Инкапсуляция через модификаторы доступа защищает данные от неправильного использования`
        },
        uz: {
            title: 'Sinflar va Obyektlar',
            solutionCode: `public class Person {
    // Nusxa maydonlari - har bir obyekt o'zining nusxasiga ega
    private String name;
    private int age;
    private String email;

    // Statik maydon - barcha Person obyektlari uchun umumiy
    private static int personCount = 0;

    // Konstruktor
    public Person(String name, int age, String email) {
        this.name = name;
        this.age = age;
        this.email = email;
        personCount++; // Yangi Person yaratilganda hisoblagichni oshirish
    }

    // Nusxa metodi - ma'lum bir obyekt bilan ishlaydi
    public void displayInfo() {
        System.out.println("Ism: " + name);
        System.out.println("Yosh: " + age);
        System.out.println("Email: " + email);
        System.out.println("---");
    }

    // Statik metod - sinfga tegishli, obyektlarga emas
    public static int getPersonCount() {
        return personCount;
    }

    // Email yangilash uchun nusxa metodi
    public void updateEmail(String newEmail) {
        this.email = newEmail;
        System.out.println(name + " uchun email yangilandi");
    }

    public static void main(String[] args) {
        // Obyektlarni yaratish - har biri Person sinfining nusxasi
        Person person1 = new Person("Alisa Jonson", 28, "alice@email.com");
        Person person2 = new Person("Bob Smit", 35, "bob@email.com");
        Person person3 = new Person("Kerol Vilyams", 42, "carol@email.com");

        // Ma'lum obyektlar uchun nusxa metodlarini chaqirish
        person1.displayInfo();
        person2.displayInfo();
        person3.displayInfo();

        // Bir kishi uchun emailni yangilash
        person1.updateEmail("alice.johnson@newemail.com");
        person1.displayInfo();

        // Statik metodini chaqirish - sinf nomidan foydalanish
        System.out.println("Jami yaratilgan odamlar: " + Person.getPersonCount());
    }
}`,
            description: `Java-da sinflar va obyektlarning asosiy tushunchalarini ko'rsatadigan **Person** sinfini yarating.

**Talablar:**
1. Quyidagi nusxa maydonlariga ega Person sinfini yarating:
   1.1. name (String)
   1.2. age (int)
   1.3. email (String)

2. Yaratilgan Person obyektlarining umumiy sonini kuzatish uchun statik maydon qo'shing:
   2.1. static int personCount

3. Quyidagi metodlarni amalga oshiring:
   3.1. Shaxs ma'lumotlarini ko'rsatish metodi: displayInfo()
   3.2. Umumiy odamlar sonini olish uchun statik metod: getPersonCount()
   3.3. Emailni yangilash uchun nusxa metodi: updateEmail(String newEmail)

4. Main metodida:
   4.1. Kamida 3 ta Person obyektini yarating
   4.2. Har bir odam uchun displayInfo() ni chaqiring
   4.3. Bir kishi uchun emailni yangilang
   4.4. Yaratilgan odamlarning umumiy sonini ko'rsating

**O'rganish maqsadlari:**
- Sinflar va obyektlar o'rtasidagi farqni tushunish
- Nusxa va statik a'zolar haqida bilim olish
- Obyektlar yaratish va ishlatishda amaliyot`,
            hint1: `Sinf yuqori qismida nusxa o'zgaruvchilarini e'lon qilishdan boshlang. Har bir obyekt ushbu o'zgaruvchilarning o'z nusxasiga ega bo'lishini eslang.`,
            hint2: `Statik o'zgaruvchilar sinfning o'ziga tegishli va barcha obyektlar tomonidan umumiy qo'llaniladi. personCount ni 0 ga ishga tushiring va uni konstruktorda oshiring.`,
            whyItMatters: `Sinflar va obyektlarni tushunish Java dasturlashning asosi hisoblanadi. Sinflar - bu tuzilma va xatti-harakatni belgilaydigan rejalar, obyektlar esa ushbu rejalardan yaratilgan haqiqiy nusxalardir. Nusxa va statik a'zolar o'rtasidagi farq to'g'ri obyektga yo'naltirilgan dizayn uchun juda muhimdir.

**Ishlab chiqarish patterni:**
\`\`\`java
public class Person {
    // Nusxa maydonlari - har bir obyekt o'zining nusxasiga ega
    private String name;
    private int age;

    // Statik maydon - barcha obyektlar uchun umumiy
    private static int personCount = 0;

    public Person(String name, int age) {
        this.name = name;
        this.age = age;
        personCount++; // Yaratilgan obyektlar sonini kuzatamiz
    }

    public static int getPersonCount() {
        return personCount; // Statik maydonga kirish uchun statik metod
    }
}
\`\`\`

**Amaliy foydalari:**
- Nusxa maydonlari har bir obyektga o'ziga xos holatni saqlash imkonini beradi
- Statik maydonlar umumiy ma'lumotlar uchun juda mos keladi (hisoblagichlar, konstantalar)
- Statik metodlarni obyekt yaratmasdan chaqirish mumkin (Person.getPersonCount())
- Kirish modifikatorlari orqali inkapsulyatsiya ma'lumotlarni noto'g'ri foydalanishdan himoya qiladi`
        }
    }
};

export default task;
