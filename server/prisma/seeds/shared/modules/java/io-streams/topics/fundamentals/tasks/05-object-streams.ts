import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-object-streams',
    title: 'Object Serialization with ObjectInputStream and ObjectOutputStream',
    difficulty: 'medium',
    tags: ['java', 'io', 'serialization', 'objects'],
    estimatedTime: '35m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn how to serialize and deserialize objects using ObjectInputStream and ObjectOutputStream.

**Requirements:**
1. Create a Person class that implements Serializable
2. Add fields: name (String), age (int), email (String)
3. Write Person objects to a file using ObjectOutputStream
4. Read the objects back using ObjectInputStream
5. Display the deserialized objects
6. Demonstrate transient keyword for fields that shouldn't be serialized

Object serialization converts Java objects into a byte stream for storage or transmission, while deserialization recreates the objects from bytes.`,
    initialCode: `import java.io.*;

// Create a Serializable Person class

public class ObjectStreams {
    public static void main(String[] args) {
        String fileName = "persons.ser";

        // Create Person objects

        // Write objects to file using ObjectOutputStream

        // Read objects from file using ObjectInputStream

        // Display the objects
    }
}`,
    solutionCode: `import java.io.*;

// Serializable Person class
class Person implements Serializable {
    // serialVersionUID for version control
    private static final long serialVersionUID = 1L;

    private String name;
    private int age;
    private String email;

    // transient fields are not serialized
    private transient String password;

    public Person(String name, int age, String email, String password) {
        this.name = name;
        this.age = age;
        this.email = email;
        this.password = password;
    }

    @Override
    public String toString() {
        return "Person{name='" + name + "', age=" + age +
               ", email='" + email + "', password='" + password + "'}";
    }
}

public class ObjectStreams {
    public static void main(String[] args) {
        String fileName = "persons.ser";

        // Create Person objects
        Person person1 = new Person("Alice Johnson", 28, "alice@example.com", "secret123");
        Person person2 = new Person("Bob Smith", 35, "bob@example.com", "pass456");
        Person person3 = new Person("Charlie Brown", 42, "charlie@example.com", "pwd789");

        // Write objects to file using ObjectOutputStream
        try (FileOutputStream fos = new FileOutputStream(fileName);
             ObjectOutputStream oos = new ObjectOutputStream(fos)) {

            // Write multiple objects
            oos.writeObject(person1);
            oos.writeObject(person2);
            oos.writeObject(person3);

            System.out.println("Objects serialized successfully");
            System.out.println("\\nOriginal objects:");
            System.out.println(person1);
            System.out.println(person2);
            System.out.println(person3);

        } catch (IOException e) {
            System.err.println("Error writing objects: " + e.getMessage());
        }

        // Read objects from file using ObjectInputStream
        try (FileInputStream fis = new FileInputStream(fileName);
             ObjectInputStream ois = new ObjectInputStream(fis)) {

            System.out.println("\\nDeserialized objects:");

            // Read objects in the same order they were written
            Person p1 = (Person) ois.readObject();
            Person p2 = (Person) ois.readObject();
            Person p3 = (Person) ois.readObject();

            // Display the deserialized objects
            System.out.println(p1);
            System.out.println(p2);
            System.out.println(p3);

            System.out.println("\\nNote: password field is null (transient)");

        } catch (ClassNotFoundException e) {
            System.err.println("Class not found: " + e.getMessage());
        } catch (IOException e) {
            System.err.println("Error reading objects: " + e.getMessage());
        }

        // Example with a list of objects
        serializeList();
    }

    // Demonstrate serializing a collection
    private static void serializeList() {
        String fileName = "person_list.ser";

        System.out.println("\\n=== Serializing Collections ===");

        // Create a list of persons
        java.util.List<Person> persons = new java.util.ArrayList<>();
        persons.add(new Person("David Lee", 30, "david@example.com", "pwd1"));
        persons.add(new Person("Emma Wilson", 25, "emma@example.com", "pwd2"));

        // Serialize the entire list
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new FileOutputStream(fileName))) {

            oos.writeObject(persons);
            System.out.println("List serialized: " + persons.size() + " persons");

        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
        }

        // Deserialize the list
        try (ObjectInputStream ois = new ObjectInputStream(
                new FileInputStream(fileName))) {

            @SuppressWarnings("unchecked")
            java.util.List<Person> readPersons =
                (java.util.List<Person>) ois.readObject();

            System.out.println("List deserialized: " + readPersons.size() + " persons");
            for (Person p : readPersons) {
                System.out.println("  " + p);
            }

        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Error: " + e.getMessage());
        }
    }
}`,
    hint1: `A class must implement Serializable interface to be serializable. Add 'implements Serializable' to the class declaration.`,
    hint2: `Use transient keyword for fields that shouldn't be serialized (like passwords). After deserialization, transient fields will have default values (null for objects, 0 for numbers).`,
    whyItMatters: `Object serialization is crucial for saving object state, sending objects over networks, caching, and distributed computing. It's used by frameworks like RMI, EJB, and in session management. Understanding serialization helps prevent security issues and version compatibility problems.`,
    order: 4,
    testCode: `import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;
import java.io.*;
import java.nio.file.Path;

// Test 1: Person class is Serializable
class Test1 {
    @Test
    void testPersonIsSerializable() {
        Person person = new Person("Test", 25, "test@test.com", "pwd");
        assertTrue(person instanceof Serializable);
    }
}

// Test 2: Person can be serialized to bytes
class Test2 {
    @Test
    void testPersonSerializesToBytes() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(baos);
        Person person = new Person("Alice", 30, "alice@test.com", "secret");
        oos.writeObject(person);
        oos.close();
        assertTrue(baos.size() > 0);
    }
}

// Test 3: Person can be deserialized
class Test3 {
    @Test
    void testPersonDeserializes() throws IOException, ClassNotFoundException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(baos);
        Person original = new Person("Bob", 35, "bob@test.com", "pass");
        oos.writeObject(original);
        oos.close();

        ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(baos.toByteArray()));
        Person deserialized = (Person) ois.readObject();
        assertNotNull(deserialized);
        ois.close();
    }
}

// Test 4: Name field is preserved
class Test4 {
    @Test
    void testNameFieldPreserved() throws IOException, ClassNotFoundException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(baos);
        Person original = new Person("Charlie", 40, "charlie@test.com", "pwd");
        oos.writeObject(original);
        oos.close();

        ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(baos.toByteArray()));
        Person deserialized = (Person) ois.readObject();
        assertTrue(deserialized.toString().contains("Charlie"));
        ois.close();
    }
}

// Test 5: Transient password is not serialized
class Test5 {
    @Test
    void testTransientPasswordNotSerialized() throws IOException, ClassNotFoundException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(baos);
        Person original = new Person("David", 45, "david@test.com", "secret123");
        oos.writeObject(original);
        oos.close();

        ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(baos.toByteArray()));
        Person deserialized = (Person) ois.readObject();
        assertTrue(deserialized.toString().contains("password='null'"));
        ois.close();
    }
}

// Test 6: Multiple objects can be serialized
class Test6 {
    @Test
    void testMultipleObjectsSerialized() throws IOException, ClassNotFoundException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(baos);
        oos.writeObject(new Person("A", 20, "a@test.com", "p1"));
        oos.writeObject(new Person("B", 25, "b@test.com", "p2"));
        oos.close();

        ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(baos.toByteArray()));
        Person p1 = (Person) ois.readObject();
        Person p2 = (Person) ois.readObject();
        assertNotNull(p1);
        assertNotNull(p2);
        ois.close();
    }
}

// Test 7: Age field is preserved
class Test7 {
    @Test
    void testAgeFieldPreserved() throws IOException, ClassNotFoundException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(baos);
        Person original = new Person("Eve", 28, "eve@test.com", "pwd");
        oos.writeObject(original);
        oos.close();

        ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(baos.toByteArray()));
        Person deserialized = (Person) ois.readObject();
        assertTrue(deserialized.toString().contains("age=28"));
        ois.close();
    }
}

// Test 8: Email field is preserved
class Test8 {
    @Test
    void testEmailFieldPreserved() throws IOException, ClassNotFoundException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(baos);
        Person original = new Person("Frank", 33, "frank@example.com", "pwd");
        oos.writeObject(original);
        oos.close();

        ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(baos.toByteArray()));
        Person deserialized = (Person) ois.readObject();
        assertTrue(deserialized.toString().contains("frank@example.com"));
        ois.close();
    }
}

// Test 9: Serialization works with files
class Test9 {
    @TempDir
    Path tempDir;

    @Test
    void testSerializationWithFiles() throws IOException, ClassNotFoundException {
        File file = tempDir.resolve("person.ser").toFile();

        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file))) {
            oos.writeObject(new Person("Grace", 29, "grace@test.com", "pwd"));
        }

        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
            Person p = (Person) ois.readObject();
            assertTrue(p.toString().contains("Grace"));
        }
    }
}

// Test 10: List of persons can be serialized
class Test10 {
    @Test
    void testListSerializable() throws IOException, ClassNotFoundException {
        java.util.List<Person> list = new java.util.ArrayList<>();
        list.add(new Person("H", 20, "h@t.com", "p"));
        list.add(new Person("I", 21, "i@t.com", "p"));

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(baos);
        oos.writeObject(list);
        oos.close();

        ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(baos.toByteArray()));
        @SuppressWarnings("unchecked")
        java.util.List<Person> deserialized = (java.util.List<Person>) ois.readObject();
        assertEquals(2, deserialized.size());
        ois.close();
    }
}`,
    translations: {
        ru: {
            title: 'Сериализация объектов с ObjectInputStream и ObjectOutputStream',
            solutionCode: `import java.io.*;

// Сериализуемый класс Person
class Person implements Serializable {
    // serialVersionUID для контроля версий
    private static final long serialVersionUID = 1L;

    private String name;
    private int age;
    private String email;

    // transient поля не сериализуются
    private transient String password;

    public Person(String name, int age, String email, String password) {
        this.name = name;
        this.age = age;
        this.email = email;
        this.password = password;
    }

    @Override
    public String toString() {
        return "Person{name='" + name + "', age=" + age +
               ", email='" + email + "', password='" + password + "'}";
    }
}

public class ObjectStreams {
    public static void main(String[] args) {
        String fileName = "persons.ser";

        // Создаем объекты Person
        Person person1 = new Person("Alice Johnson", 28, "alice@example.com", "secret123");
        Person person2 = new Person("Bob Smith", 35, "bob@example.com", "pass456");
        Person person3 = new Person("Charlie Brown", 42, "charlie@example.com", "pwd789");

        // Записываем объекты в файл с использованием ObjectOutputStream
        try (FileOutputStream fos = new FileOutputStream(fileName);
             ObjectOutputStream oos = new ObjectOutputStream(fos)) {

            // Записываем несколько объектов
            oos.writeObject(person1);
            oos.writeObject(person2);
            oos.writeObject(person3);

            System.out.println("Объекты успешно сериализованы");
            System.out.println("\\nИсходные объекты:");
            System.out.println(person1);
            System.out.println(person2);
            System.out.println(person3);

        } catch (IOException e) {
            System.err.println("Ошибка записи объектов: " + e.getMessage());
        }

        // Читаем объекты из файла с использованием ObjectInputStream
        try (FileInputStream fis = new FileInputStream(fileName);
             ObjectInputStream ois = new ObjectInputStream(fis)) {

            System.out.println("\\nДесериализованные объекты:");

            // Читаем объекты в том же порядке, что и записывали
            Person p1 = (Person) ois.readObject();
            Person p2 = (Person) ois.readObject();
            Person p3 = (Person) ois.readObject();

            // Отображаем десериализованные объекты
            System.out.println(p1);
            System.out.println(p2);
            System.out.println(p3);

            System.out.println("\\nПримечание: поле password равно null (transient)");

        } catch (ClassNotFoundException e) {
            System.err.println("Класс не найден: " + e.getMessage());
        } catch (IOException e) {
            System.err.println("Ошибка чтения объектов: " + e.getMessage());
        }

        // Пример со списком объектов
        serializeList();
    }

    // Демонстрация сериализации коллекции
    private static void serializeList() {
        String fileName = "person_list.ser";

        System.out.println("\\n=== Сериализация коллекций ===");

        // Создаем список персон
        java.util.List<Person> persons = new java.util.ArrayList<>();
        persons.add(new Person("David Lee", 30, "david@example.com", "pwd1"));
        persons.add(new Person("Emma Wilson", 25, "emma@example.com", "pwd2"));

        // Сериализуем весь список
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new FileOutputStream(fileName))) {

            oos.writeObject(persons);
            System.out.println("Список сериализован: " + persons.size() + " персон");

        } catch (IOException e) {
            System.err.println("Ошибка: " + e.getMessage());
        }

        // Десериализуем список
        try (ObjectInputStream ois = new ObjectInputStream(
                new FileInputStream(fileName))) {

            @SuppressWarnings("unchecked")
            java.util.List<Person> readPersons =
                (java.util.List<Person>) ois.readObject();

            System.out.println("Список десериализован: " + readPersons.size() + " персон");
            for (Person p : readPersons) {
                System.out.println("  " + p);
            }

        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Ошибка: " + e.getMessage());
        }
    }
}`,
            description: `Научитесь сериализовать и десериализовать объекты с помощью ObjectInputStream и ObjectOutputStream.

**Требования:**
1. Создайте класс Person, реализующий Serializable
2. Добавьте поля: name (String), age (int), email (String)
3. Запишите объекты Person в файл, используя ObjectOutputStream
4. Прочитайте объекты обратно, используя ObjectInputStream
5. Отобразите десериализованные объекты
6. Продемонстрируйте ключевое слово transient для полей, которые не должны сериализоваться

Сериализация объектов преобразует объекты Java в поток байтов для хранения или передачи, а десериализация воссоздает объекты из байтов.`,
            hint1: `Класс должен реализовывать интерфейс Serializable, чтобы быть сериализуемым. Добавьте 'implements Serializable' в объявление класса.`,
            hint2: `Используйте ключевое слово transient для полей, которые не должны сериализоваться (например, пароли). После десериализации transient поля будут иметь значения по умолчанию (null для объектов, 0 для чисел).`,
            whyItMatters: `Сериализация объектов имеет решающее значение для сохранения состояния объектов, отправки объектов по сетям, кэширования и распределенных вычислений. Она используется фреймворками, такими как RMI, EJB, и в управлении сеансами. Понимание сериализации помогает предотвратить проблемы безопасности и совместимости версий.`
        },
        uz: {
            title: 'ObjectInputStream va ObjectOutputStream bilan obyektlarni serializatsiya qilish',
            solutionCode: `import java.io.*;

// Serializatsiya qilinadigan Person klassi
class Person implements Serializable {
    // versiyani nazorat qilish uchun serialVersionUID
    private static final long serialVersionUID = 1L;

    private String name;
    private int age;
    private String email;

    // transient maydonlar serializatsiya qilinmaydi
    private transient String password;

    public Person(String name, int age, String email, String password) {
        this.name = name;
        this.age = age;
        this.email = email;
        this.password = password;
    }

    @Override
    public String toString() {
        return "Person{name='" + name + "', age=" + age +
               ", email='" + email + "', password='" + password + "'}";
    }
}

public class ObjectStreams {
    public static void main(String[] args) {
        String fileName = "persons.ser";

        // Person obyektlarini yaratamiz
        Person person1 = new Person("Alice Johnson", 28, "alice@example.com", "secret123");
        Person person2 = new Person("Bob Smith", 35, "bob@example.com", "pass456");
        Person person3 = new Person("Charlie Brown", 42, "charlie@example.com", "pwd789");

        // ObjectOutputStream yordamida obyektlarni faylga yozamiz
        try (FileOutputStream fos = new FileOutputStream(fileName);
             ObjectOutputStream oos = new ObjectOutputStream(fos)) {

            // Bir nechta obyektlarni yozamiz
            oos.writeObject(person1);
            oos.writeObject(person2);
            oos.writeObject(person3);

            System.out.println("Obyektlar muvaffaqiyatli serializatsiya qilindi");
            System.out.println("\\nDastlabki obyektlar:");
            System.out.println(person1);
            System.out.println(person2);
            System.out.println(person3);

        } catch (IOException e) {
            System.err.println("Obyektlarni yozishda xato: " + e.getMessage());
        }

        // ObjectInputStream yordamida fayldan obyektlarni o'qiymiz
        try (FileInputStream fis = new FileInputStream(fileName);
             ObjectInputStream ois = new ObjectInputStream(fis)) {

            System.out.println("\\nDeserializatsiya qilingan obyektlar:");

            // Obyektlarni yozilgan tartibda o'qiymiz
            Person p1 = (Person) ois.readObject();
            Person p2 = (Person) ois.readObject();
            Person p3 = (Person) ois.readObject();

            // Deserializatsiya qilingan obyektlarni ko'rsatamiz
            System.out.println(p1);
            System.out.println(p2);
            System.out.println(p3);

            System.out.println("\\nEslatma: password maydoni null (transient)");

        } catch (ClassNotFoundException e) {
            System.err.println("Klass topilmadi: " + e.getMessage());
        } catch (IOException e) {
            System.err.println("Obyektlarni o'qishda xato: " + e.getMessage());
        }

        // Obyektlar ro'yxati bilan misol
        serializeList();
    }

    // Kolleksiyani serializatsiya qilishni ko'rsatish
    private static void serializeList() {
        String fileName = "person_list.ser";

        System.out.println("\\n=== Kolleksiyalarni serializatsiya qilish ===");

        // Personlar ro'yxatini yaratamiz
        java.util.List<Person> persons = new java.util.ArrayList<>();
        persons.add(new Person("David Lee", 30, "david@example.com", "pwd1"));
        persons.add(new Person("Emma Wilson", 25, "emma@example.com", "pwd2"));

        // Butun ro'yxatni serializatsiya qilamiz
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new FileOutputStream(fileName))) {

            oos.writeObject(persons);
            System.out.println("Ro'yxat serializatsiya qilindi: " + persons.size() + " kishi");

        } catch (IOException e) {
            System.err.println("Xato: " + e.getMessage());
        }

        // Ro'yxatni deserializatsiya qilamiz
        try (ObjectInputStream ois = new ObjectInputStream(
                new FileInputStream(fileName))) {

            @SuppressWarnings("unchecked")
            java.util.List<Person> readPersons =
                (java.util.List<Person>) ois.readObject();

            System.out.println("Ro'yxat deserializatsiya qilindi: " + readPersons.size() + " kishi");
            for (Person p : readPersons) {
                System.out.println("  " + p);
            }

        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Xato: " + e.getMessage());
        }
    }
}`,
            description: `ObjectInputStream va ObjectOutputStream yordamida obyektlarni serializatsiya va deserializatsiya qilishni o'rganing.

**Talablar:**
1. Serializable ni amalga oshiradigan Person klassini yarating
2. Maydonlarni qo'shing: name (String), age (int), email (String)
3. ObjectOutputStream yordamida Person obyektlarini faylga yozing
4. ObjectInputStream yordamida obyektlarni qaytadan o'qing
5. Deserializatsiya qilingan obyektlarni ko'rsating
6. Serializatsiya qilinmasligi kerak bo'lgan maydonlar uchun transient kalit so'zini ko'rsating

Obyektlarni serializatsiya qilish Java obyektlarini saqlash yoki uzatish uchun bayt oqimiga aylantiradi, deserializatsiya esa obyektlarni baytlardan qayta yaratadi.`,
            hint1: `Klass serializatsiya qilinadigan bo'lishi uchun Serializable interfeysini amalga oshirishi kerak. Klass deklaratsiyasiga 'implements Serializable' qo'shing.`,
            hint2: `Serializatsiya qilinmasligi kerak bo'lgan maydonlar (masalan, parollar) uchun transient kalit so'zidan foydalaning. Deserializatsiya qilingandan keyin transient maydonlar standart qiymatlarga ega bo'ladi (obyektlar uchun null, raqamlar uchun 0).`,
            whyItMatters: `Obyektlarni serializatsiya qilish obyekt holatini saqlash, obyektlarni tarmoqlar orqali yuborish, keshlash va taqsimlangan hisoblash uchun juda muhimdir. U RMI, EJB kabi freymvorklar va sessiyalarni boshqarishda ishlatiladi. Serializatsiyani tushunish xavfsizlik muammolari va versiya mosligi muammolarini oldini olishga yordam beradi.`
        }
    }
};

export default task;
