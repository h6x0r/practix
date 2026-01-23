import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-multiple-interfaces',
    title: 'Multiple Interface Implementation',
    difficulty: 'easy',
    tags: ['java', 'interfaces', 'multiple-inheritance', 'oop'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Multiple Interface Implementation

Java doesn't support multiple inheritance with classes, but a class can implement multiple interfaces. This provides the benefits of multiple inheritance without the diamond problem complexity.

## Requirements:
1. Create a \`Playable\` interface with:
   1.1. \`void play()\`
   1.2. \`void pause()\`
   1.3. \`void stop()\`

2. Create a \`Downloadable\` interface with:
   2.1. \`void download()\`
   2.2. \`long getFileSize()\`

3. Create a \`StreamingVideo\` class implementing both interfaces:
   3.1. Has \`title\` and \`fileSizeInMB\` fields
   3.2. Implements all methods from both interfaces
   3.3. Print meaningful messages for each action

4. Demonstrate using the class with both interface types

## Example Output:
\`\`\`
Playing: Java Tutorial
Video paused
Downloading: Java Tutorial (250 MB)
File size: 250 MB
Video stopped
\`\`\``,
    initialCode: `// TODO: Create Playable interface

// TODO: Create Downloadable interface

// TODO: Create StreamingVideo class implementing both interfaces

public class MultipleInterfaces {
    public static void main(String[] args) {
        // TODO: Create StreamingVideo instance

        // TODO: Use as Playable type

        // TODO: Use as Downloadable type
    }
}`,
    solutionCode: `// Playable interface for media playback functionality
interface Playable {
    void play();
    void pause();
    void stop();
}

// Downloadable interface for download functionality
interface Downloadable {
    void download();
    long getFileSize();
}

// StreamingVideo implements multiple interfaces
class StreamingVideo implements Playable, Downloadable {
    private String title;
    private long fileSizeInMB;

    public StreamingVideo(String title, long fileSizeInMB) {
        this.title = title;
        this.fileSizeInMB = fileSizeInMB;
    }

    // Implementing Playable interface methods
    @Override
    public void play() {
        System.out.println("Playing: " + title);
    }

    @Override
    public void pause() {
        System.out.println("Video paused");
    }

    @Override
    public void stop() {
        System.out.println("Video stopped");
    }

    // Implementing Downloadable interface methods
    @Override
    public void download() {
        System.out.println("Downloading: " + title + " (" + fileSizeInMB + " MB)");
    }

    @Override
    public long getFileSize() {
        return fileSizeInMB;
    }
}

public class MultipleInterfaces {
    public static void main(String[] args) {
        // Create instance that implements both interfaces
        StreamingVideo video = new StreamingVideo("Java Tutorial", 250);

        // Using as Playable type
        Playable playable = video;
        playable.play();
        playable.pause();

        // Using as Downloadable type
        Downloadable downloadable = video;
        downloadable.download();
        System.out.println("File size: " + downloadable.getFileSize() + " MB");

        // Can access all methods through original reference
        video.stop();
    }
}`,
    hint1: `A class can implement multiple interfaces by separating them with commas: class MyClass implements Interface1, Interface2`,
    hint2: `You must implement ALL methods from ALL interfaces. Use @Override to ensure you're correctly implementing interface methods.`,
    whyItMatters: `Multiple interface implementation is Java's solution to multiple inheritance. It's used extensively in real-world applications - for example, a class might implement Serializable, Cloneable, and Comparable interfaces simultaneously. This pattern allows objects to take on multiple roles without the complexity of multiple class inheritance.

**Production Pattern:**

\`\`\`java
// Real-world example: database entity class
class User implements Serializable, Comparable<User>, Cloneable {
    private String username;
    private int age;

    // For saving to DB or cache
    @Override
    public Object clone() throws CloneNotSupportedException {
        return super.clone();
    }

    // For sorting users
    @Override
    public int compareTo(User other) {
        return this.username.compareTo(other.username);
    }

    // Serializable requires no methods (marker interface)
}

// Using multiple capabilities
List<User> users = new ArrayList<>();
Collections.sort(users);	// Uses Comparable
User backup = (User) user.clone();	// Uses Cloneable
saveToFile(user);	// Uses Serializable
\`\`\`

**Practical Benefits:**

1. **Multiple Roles**: Object can have several unrelated capabilities
2. **Flexible Composition**: Combine behaviors as needed
3. **Standard Practice**: Used in Java Collections, Spring, Hibernate
4. **Extensibility**: Easy to add new capabilities without changing class hierarchy`,
    order: 2,
    testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.io.*;

// Test 1: StreamingVideo implements Playable
class Test1 {
    @Test
    void testImplementsPlayable() {
        StreamingVideo video = new StreamingVideo("Test", 100);
        assertTrue(video instanceof Playable);
    }
}

// Test 2: StreamingVideo implements Downloadable
class Test2 {
    @Test
    void testImplementsDownloadable() {
        StreamingVideo video = new StreamingVideo("Test", 100);
        assertTrue(video instanceof Downloadable);
    }
}

// Test 3: Play method works
class Test3 {
    @Test
    void testPlayMethod() {
        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(outContent));

        StreamingVideo video = new StreamingVideo("Java Tutorial", 250);
        video.play();

        System.setOut(oldOut);
        String output = outContent.toString();
        assertTrue(output.contains("Play") || output.contains("Java Tutorial"));
    }
}

// Test 4: Pause method works
class Test4 {
    @Test
    void testPauseMethod() {
        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(outContent));

        StreamingVideo video = new StreamingVideo("Test Video", 100);
        video.pause();

        System.setOut(oldOut);
        String output = outContent.toString();
        assertTrue(output.toLowerCase().contains("pause"));
    }
}

// Test 5: Stop method works
class Test5 {
    @Test
    void testStopMethod() {
        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(outContent));

        StreamingVideo video = new StreamingVideo("Test Video", 100);
        video.stop();

        System.setOut(oldOut);
        String output = outContent.toString();
        assertTrue(output.toLowerCase().contains("stop"));
    }
}

// Test 6: Download method works
class Test6 {
    @Test
    void testDownloadMethod() {
        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(outContent));

        StreamingVideo video = new StreamingVideo("Java Tutorial", 250);
        video.download();

        System.setOut(oldOut);
        String output = outContent.toString();
        assertTrue(output.contains("Download") || output.contains("250"));
    }
}

// Test 7: getFileSize returns correct size
class Test7 {
    @Test
    void testGetFileSize() {
        StreamingVideo video = new StreamingVideo("Test", 250);
        assertEquals(250, video.getFileSize());
    }
}

// Test 8: Can use as Playable reference
class Test8 {
    @Test
    void testPlayableReference() {
        Playable playable = new StreamingVideo("Test", 100);
        assertDoesNotThrow(() -> playable.play());
        assertDoesNotThrow(() -> playable.pause());
        assertDoesNotThrow(() -> playable.stop());
    }
}

// Test 9: Can use as Downloadable reference
class Test9 {
    @Test
    void testDownloadableReference() {
        Downloadable downloadable = new StreamingVideo("Test", 500);
        assertDoesNotThrow(() -> downloadable.download());
        assertEquals(500, downloadable.getFileSize());
    }
}

// Test 10: Multiple videos with different sizes
class Test10 {
    @Test
    void testMultipleVideos() {
        StreamingVideo small = new StreamingVideo("Small", 50);
        StreamingVideo large = new StreamingVideo("Large", 1000);

        assertTrue(large.getFileSize() > small.getFileSize());
    }
}`,
    translations: {
        ru: {
            title: 'Реализация нескольких интерфейсов',
            solutionCode: `// Интерфейс Playable для функциональности воспроизведения медиа
interface Playable {
    void play();
    void pause();
    void stop();
}

// Интерфейс Downloadable для функциональности загрузки
interface Downloadable {
    void download();
    long getFileSize();
}

// StreamingVideo реализует несколько интерфейсов
class StreamingVideo implements Playable, Downloadable {
    private String title;
    private long fileSizeInMB;

    public StreamingVideo(String title, long fileSizeInMB) {
        this.title = title;
        this.fileSizeInMB = fileSizeInMB;
    }

    // Реализация методов интерфейса Playable
    @Override
    public void play() {
        System.out.println("Playing: " + title);
    }

    @Override
    public void pause() {
        System.out.println("Video paused");
    }

    @Override
    public void stop() {
        System.out.println("Video stopped");
    }

    // Реализация методов интерфейса Downloadable
    @Override
    public void download() {
        System.out.println("Downloading: " + title + " (" + fileSizeInMB + " MB)");
    }

    @Override
    public long getFileSize() {
        return fileSizeInMB;
    }
}

public class MultipleInterfaces {
    public static void main(String[] args) {
        // Создаем экземпляр, реализующий оба интерфейса
        StreamingVideo video = new StreamingVideo("Java Tutorial", 250);

        // Используем как тип Playable
        Playable playable = video;
        playable.play();
        playable.pause();

        // Используем как тип Downloadable
        Downloadable downloadable = video;
        downloadable.download();
        System.out.println("File size: " + downloadable.getFileSize() + " MB");

        // Можем обращаться ко всем методам через исходную ссылку
        video.stop();
    }
}`,
            description: `# Реализация нескольких интерфейсов

Java не поддерживает множественное наследование для классов, но класс может реализовывать несколько интерфейсов. Это обеспечивает преимущества множественного наследования без сложности проблемы ромба.

## Требования:
1. Создайте интерфейс \`Playable\` с методами:
   1.1. \`void play()\`
   1.2. \`void pause()\`
   1.3. \`void stop()\`

2. Создайте интерфейс \`Downloadable\` с методами:
   2.1. \`void download()\`
   2.2. \`long getFileSize()\`

3. Создайте класс \`StreamingVideo\`, реализующий оба интерфейса:
   3.1. Имеет поля \`title\` и \`fileSizeInMB\`
   3.2. Реализует все методы обоих интерфейсов
   3.3. Выводит осмысленные сообщения для каждого действия

4. Продемонстрируйте использование класса с обоими типами интерфейсов

## Пример вывода:
\`\`\`
Playing: Java Tutorial
Video paused
Downloading: Java Tutorial (250 MB)
File size: 250 MB
Video stopped
\`\`\``,
            hint1: `Класс может реализовывать несколько интерфейсов, разделяя их запятыми: class MyClass implements Interface1, Interface2`,
            hint2: `Вы должны реализовать ВСЕ методы из ВСЕХ интерфейсов. Используйте @Override, чтобы убедиться, что вы правильно реализуете методы интерфейса.`,
            whyItMatters: `Реализация нескольких интерфейсов - это решение Java для множественного наследования. Это широко используется в реальных приложениях - например, класс может одновременно реализовывать интерфейсы Serializable, Cloneable и Comparable. Этот паттерн позволяет объектам принимать несколько ролей без сложности множественного наследования классов.

**Продакшен паттерн:**

\`\`\`java
// Реальный пример: класс сущности базы данных
class User implements Serializable, Comparable<User>, Cloneable {
    private String username;
    private int age;

    // Для сохранения в БД или кэш
    @Override
    public Object clone() throws CloneNotSupportedException {
        return super.clone();
    }

    // Для сортировки пользователей
    @Override
    public int compareTo(User other) {
        return this.username.compareTo(other.username);
    }

    // Serializable не требует методов (marker interface)
}

// Использование нескольких возможностей
List<User> users = new ArrayList<>();
Collections.sort(users);	// Использует Comparable
User backup = (User) user.clone();	// Использует Cloneable
saveToFile(user);	// Использует Serializable
\`\`\`

**Практические преимущества:**

1. **Множественные роли**: Объект может иметь несколько несвязанных возможностей
2. **Гибкая композиция**: Комбинируйте поведения по необходимости
3. **Стандартная практика**: Используется в Java Collections, Spring, Hibernate
4. **Расширяемость**: Легко добавлять новые возможности без изменения иерархии классов`
        },
        uz: {
            title: `Bir nechta interfeysni amalga oshirish`,
            solutionCode: `// Playable interfeysi media ijro etish funksiyasi uchun
interface Playable {
    void play();
    void pause();
    void stop();
}

// Downloadable interfeysi yuklab olish funksiyasi uchun
interface Downloadable {
    void download();
    long getFileSize();
}

// StreamingVideo bir nechta interfeysni amalga oshiradi
class StreamingVideo implements Playable, Downloadable {
    private String title;
    private long fileSizeInMB;

    public StreamingVideo(String title, long fileSizeInMB) {
        this.title = title;
        this.fileSizeInMB = fileSizeInMB;
    }

    // Playable interfeysi metodlarini amalga oshirish
    @Override
    public void play() {
        System.out.println("Playing: " + title);
    }

    @Override
    public void pause() {
        System.out.println("Video paused");
    }

    @Override
    public void stop() {
        System.out.println("Video stopped");
    }

    // Downloadable interfeysi metodlarini amalga oshirish
    @Override
    public void download() {
        System.out.println("Downloading: " + title + " (" + fileSizeInMB + " MB)");
    }

    @Override
    public long getFileSize() {
        return fileSizeInMB;
    }
}

public class MultipleInterfaces {
    public static void main(String[] args) {
        // Ikkala interfeysni ham amalga oshiruvchi misolni yaratamiz
        StreamingVideo video = new StreamingVideo("Java Tutorial", 250);

        // Playable turi sifatida ishlatamiz
        Playable playable = video;
        playable.play();
        playable.pause();

        // Downloadable turi sifatida ishlatamiz
        Downloadable downloadable = video;
        downloadable.download();
        System.out.println("File size: " + downloadable.getFileSize() + " MB");

        // Asl havola orqali barcha metodlarga murojaat qilishimiz mumkin
        video.stop();
    }
}`,
            description: `# Bir nechta interfeysni amalga oshirish

Java klasslar uchun ko'p merosxo'rlikni qo'llab-quvvatlamaydi, lekin klass bir nechta interfeysni amalga oshirishi mumkin. Bu olmos muammosining murakkabligi bo'lmagan holda ko'p merosxo'rlik afzalliklarini beradi.

## Talablar:
1. \`Playable\` interfeysini metodlar bilan yarating:
   1.1. \`void play()\`
   1.2. \`void pause()\`
   1.3. \`void stop()\`

2. \`Downloadable\` interfeysini metodlar bilan yarating:
   2.1. \`void download()\`
   2.2. \`long getFileSize()\`

3. Ikkala interfeysni ham amalga oshiruvchi \`StreamingVideo\` klassini yarating:
   3.1. \`title\` va \`fileSizeInMB\` maydonlariga ega
   3.2. Ikkala interfeysdagi barcha metodlarni amalga oshiradi
   3.3. Har bir harakat uchun ma'noli xabarlarni chiqaradi

4. Klassdan ikkala interfeys turi bilan foydalanishni namoyish eting

## Chiqish namunasi:
\`\`\`
Playing: Java Tutorial
Video paused
Downloading: Java Tutorial (250 MB)
File size: 250 MB
Video stopped
\`\`\``,
            hint1: `Klass bir nechta interfeysni vergul bilan ajratib amalga oshirishi mumkin: class MyClass implements Interface1, Interface2`,
            hint2: `Siz BARCHA interfeyslardan BARCHA metodlarni amalga oshirishingiz kerak. Interfeys metodlarini to'g'ri amalga oshirayotganingizga ishonch hosil qilish uchun @Override dan foydalaning.`,
            whyItMatters: `Bir nechta interfeysni amalga oshirish Java-ning ko'p merosxo'rlik yechimi hisoblanadi. Bu haqiqiy ilovalarda keng qo'llaniladi - masalan, klass bir vaqtning o'zida Serializable, Cloneable va Comparable interfeyslarini amalga oshirishi mumkin. Bu namuna obyektlarga ko'p klassli merosxo'rlik murakkabligi bo'lmagan holda bir nechta rollarni o'ynashga imkon beradi.

**Ishlab chiqarish patterni:**

\`\`\`java
// Haqiqiy misol: ma'lumotlar bazasi entity klassi
class User implements Serializable, Comparable<User>, Cloneable {
    private String username;
    private int age;

    // DB yoki keshga saqlash uchun
    @Override
    public Object clone() throws CloneNotSupportedException {
        return super.clone();
    }

    // Foydalanuvchilarni saralash uchun
    @Override
    public int compareTo(User other) {
        return this.username.compareTo(other.username);
    }

    // Serializable metodlarni talab qilmaydi (marker interface)
}

// Bir nechta imkoniyatlardan foydalanish
List<User> users = new ArrayList<>();
Collections.sort(users);	// Comparable dan foydalanadi
User backup = (User) user.clone();	// Cloneable dan foydalanadi
saveToFile(user);	// Serializable dan foydalanadi
\`\`\`

**Amaliy foydalari:**

1. **Bir nechta rollar**: Obyekt bir nechta bog'lanmagan imkoniyatlarga ega bo'lishi mumkin
2. **Moslashuvchan kompozitsiya**: Kerak bo'lganda xatti-harakatlarni birlashtiring
3. **Standart amaliyot**: Java Collections, Spring, Hibernate da qo'llaniladi
4. **Kengaytirilishi**: Klass ierarxiyasini o'zgartirmasdan yangi imkoniyatlarni qo'shish oson`
        }
    }
};

export default task;
