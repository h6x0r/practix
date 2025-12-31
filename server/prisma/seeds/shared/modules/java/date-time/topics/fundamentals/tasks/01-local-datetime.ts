import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-local-datetime',
    title: 'LocalDate, LocalTime, LocalDateTime',
    difficulty: 'easy',
    tags: ['java', 'date-time', 'java8', 'temporal', 'localdate'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `# LocalDate, LocalTime, LocalDateTime

The Java Date and Time API (java.time) introduced in Java 8 provides classes for handling dates and times without timezone information. LocalDate represents a date, LocalTime represents a time, and LocalDateTime combines both.

## Requirements:
1. Create and display LocalDate objects:
   1. Get current date
   2. Create specific dates using of()
   3. Parse dates from strings

2. Work with LocalTime:
   1. Get current time
   2. Create specific times
   3. Extract hours, minutes, seconds

3. Combine with LocalDateTime:
   1. Get current date and time
   2. Create from date and time components
   3. Extract date and time parts

4. Display various date/time properties (year, month, day, hour, minute)

## Example Output:
\`\`\`
=== LocalDate Examples ===
Current date: 2024-12-10
Specific date: 2025-06-15
Parsed date: 2024-01-01
Year: 2024, Month: DECEMBER, Day: 10

=== LocalTime Examples ===
Current time: 14:30:45.123456789
Specific time: 09:30:00
Hour: 14, Minute: 30, Second: 45

=== LocalDateTime Examples ===
Current date-time: 2024-12-10T14:30:45.123456789
Specific date-time: 2025-06-15T09:30:00
Date part: 2025-06-15
Time part: 09:30:00
\`\`\``,
    initialCode: `// TODO: Import java.time classes

public class LocalDateTimeBasics {
    public static void main(String[] args) {
        // TODO: Work with LocalDate

        // TODO: Work with LocalTime

        // TODO: Work with LocalDateTime
    }
}`,
    solutionCode: `import java.time.LocalDate;
import java.time.LocalTime;
import java.time.LocalDateTime;

public class LocalDateTimeBasics {
    public static void main(String[] args) {
        System.out.println("=== LocalDate Examples ===");

        // Get current date
        LocalDate currentDate = LocalDate.now();
        System.out.println("Current date: " + currentDate);

        // Create specific date
        LocalDate specificDate = LocalDate.of(2025, 6, 15);
        System.out.println("Specific date: " + specificDate);

        // Parse date from string
        LocalDate parsedDate = LocalDate.parse("2024-01-01");
        System.out.println("Parsed date: " + parsedDate);

        // Extract date components
        System.out.println("Year: " + currentDate.getYear() +
            ", Month: " + currentDate.getMonth() +
            ", Day: " + currentDate.getDayOfMonth());

        System.out.println("\\n=== LocalTime Examples ===");

        // Get current time
        LocalTime currentTime = LocalTime.now();
        System.out.println("Current time: " + currentTime);

        // Create specific time
        LocalTime specificTime = LocalTime.of(9, 30, 0);
        System.out.println("Specific time: " + specificTime);

        // Extract time components
        System.out.println("Hour: " + currentTime.getHour() +
            ", Minute: " + currentTime.getMinute() +
            ", Second: " + currentTime.getSecond());

        System.out.println("\\n=== LocalDateTime Examples ===");

        // Get current date and time
        LocalDateTime currentDateTime = LocalDateTime.now();
        System.out.println("Current date-time: " + currentDateTime);

        // Create specific date-time
        LocalDateTime specificDateTime = LocalDateTime.of(2025, 6, 15, 9, 30, 0);
        System.out.println("Specific date-time: " + specificDateTime);

        // Combine LocalDate and LocalTime
        LocalDateTime combined = LocalDateTime.of(specificDate, specificTime);
        System.out.println("Combined date-time: " + combined);

        // Extract date and time parts
        System.out.println("Date part: " + specificDateTime.toLocalDate());
        System.out.println("Time part: " + specificDateTime.toLocalTime());
    }
}`,
    hint1: `Use LocalDate.now() for current date, LocalDate.of(year, month, day) for specific dates, and LocalDate.parse() to parse from strings.`,
    hint2: `LocalDateTime combines date and time. You can create it with LocalDateTime.of() or combine LocalDate and LocalTime objects.`,
    whyItMatters: `The java.time package provides a modern, immutable, and thread-safe API for working with dates and times. Unlike the old Date and Calendar classes, these new classes are easy to use, follow best practices, and handle common date/time operations elegantly. They are essential for any Java application dealing with temporal data.

**Production Pattern:**
\`\`\`java
// Logging with timestamp
LocalDateTime now = LocalDateTime.now();
System.out.println("[" + now + "] User login successful");

// Task scheduling
LocalDate nextMeeting = LocalDate.now().plusDays(7);
LocalTime meetingTime = LocalTime.of(14, 30);
LocalDateTime meeting = LocalDateTime.of(nextMeeting, meetingTime);
\`\`\`

**Practical Benefits:**
- Immutability prevents bugs in multithreading
- Clear separation of date, time, and date-time
- Intuitive API without unexpected behavior`,
    order: 1,
    testCode: `import java.time.LocalDate;
import java.time.LocalTime;
import java.time.LocalDateTime;
import java.time.Month;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

// Test1: LocalDate.now() returns current date
class Test1 {
    @Test
    void testLocalDateNow() {
        LocalDate date = LocalDate.now();
        assertNotNull(date);
        assertTrue(date.getYear() >= 2024);
    }
}

// Test2: LocalDate.of() creates specific date
class Test2 {
    @Test
    void testLocalDateOf() {
        LocalDate date = LocalDate.of(2025, 6, 15);
        assertEquals(2025, date.getYear());
        assertEquals(Month.JUNE, date.getMonth());
        assertEquals(15, date.getDayOfMonth());
    }
}

// Test3: LocalDate.parse() parses string
class Test3 {
    @Test
    void testLocalDateParse() {
        LocalDate date = LocalDate.parse("2024-01-01");
        assertEquals(2024, date.getYear());
        assertEquals(1, date.getMonthValue());
        assertEquals(1, date.getDayOfMonth());
    }
}

// Test4: LocalTime.now() returns current time
class Test4 {
    @Test
    void testLocalTimeNow() {
        LocalTime time = LocalTime.now();
        assertNotNull(time);
        assertTrue(time.getHour() >= 0 && time.getHour() < 24);
    }
}

// Test5: LocalTime.of() creates specific time
class Test5 {
    @Test
    void testLocalTimeOf() {
        LocalTime time = LocalTime.of(9, 30, 45);
        assertEquals(9, time.getHour());
        assertEquals(30, time.getMinute());
        assertEquals(45, time.getSecond());
    }
}

// Test6: LocalDateTime.now() returns current date-time
class Test6 {
    @Test
    void testLocalDateTimeNow() {
        LocalDateTime dateTime = LocalDateTime.now();
        assertNotNull(dateTime);
        assertNotNull(dateTime.toLocalDate());
        assertNotNull(dateTime.toLocalTime());
    }
}

// Test7: LocalDateTime.of() creates specific date-time
class Test7 {
    @Test
    void testLocalDateTimeOf() {
        LocalDateTime dateTime = LocalDateTime.of(2025, 6, 15, 9, 30, 0);
        assertEquals(2025, dateTime.getYear());
        assertEquals(6, dateTime.getMonthValue());
        assertEquals(9, dateTime.getHour());
    }
}

// Test8: LocalDateTime can be created from LocalDate and LocalTime
class Test8 {
    @Test
    void testLocalDateTimeCombined() {
        LocalDate date = LocalDate.of(2025, 6, 15);
        LocalTime time = LocalTime.of(9, 30);
        LocalDateTime dateTime = LocalDateTime.of(date, time);
        assertEquals(date, dateTime.toLocalDate());
        assertEquals(time, dateTime.toLocalTime());
    }
}

// Test9: toLocalDate() extracts date part
class Test9 {
    @Test
    void testToLocalDate() {
        LocalDateTime dateTime = LocalDateTime.of(2025, 6, 15, 9, 30, 0);
        LocalDate date = dateTime.toLocalDate();
        assertEquals(LocalDate.of(2025, 6, 15), date);
    }
}

// Test10: toLocalTime() extracts time part
class Test10 {
    @Test
    void testToLocalTime() {
        LocalDateTime dateTime = LocalDateTime.of(2025, 6, 15, 9, 30, 0);
        LocalTime time = dateTime.toLocalTime();
        assertEquals(LocalTime.of(9, 30, 0), time);
    }
}
`,
    translations: {
        ru: {
            title: 'LocalDate, LocalTime, LocalDateTime',
            solutionCode: `import java.time.LocalDate;
import java.time.LocalTime;
import java.time.LocalDateTime;

public class LocalDateTimeBasics {
    public static void main(String[] args) {
        System.out.println("=== Примеры LocalDate ===");

        // Получить текущую дату
        LocalDate currentDate = LocalDate.now();
        System.out.println("Current date: " + currentDate);

        // Создать конкретную дату
        LocalDate specificDate = LocalDate.of(2025, 6, 15);
        System.out.println("Specific date: " + specificDate);

        // Разобрать дату из строки
        LocalDate parsedDate = LocalDate.parse("2024-01-01");
        System.out.println("Parsed date: " + parsedDate);

        // Извлечь компоненты даты
        System.out.println("Year: " + currentDate.getYear() +
            ", Month: " + currentDate.getMonth() +
            ", Day: " + currentDate.getDayOfMonth());

        System.out.println("\\n=== Примеры LocalTime ===");

        // Получить текущее время
        LocalTime currentTime = LocalTime.now();
        System.out.println("Current time: " + currentTime);

        // Создать конкретное время
        LocalTime specificTime = LocalTime.of(9, 30, 0);
        System.out.println("Specific time: " + specificTime);

        // Извлечь компоненты времени
        System.out.println("Hour: " + currentTime.getHour() +
            ", Minute: " + currentTime.getMinute() +
            ", Second: " + currentTime.getSecond());

        System.out.println("\\n=== Примеры LocalDateTime ===");

        // Получить текущую дату и время
        LocalDateTime currentDateTime = LocalDateTime.now();
        System.out.println("Current date-time: " + currentDateTime);

        // Создать конкретную дату-время
        LocalDateTime specificDateTime = LocalDateTime.of(2025, 6, 15, 9, 30, 0);
        System.out.println("Specific date-time: " + specificDateTime);

        // Объединить LocalDate и LocalTime
        LocalDateTime combined = LocalDateTime.of(specificDate, specificTime);
        System.out.println("Combined date-time: " + combined);

        // Извлечь части даты и времени
        System.out.println("Date part: " + specificDateTime.toLocalDate());
        System.out.println("Time part: " + specificDateTime.toLocalTime());
    }
}`,
            description: `# LocalDate, LocalTime, LocalDateTime

API даты и времени Java (java.time), представленный в Java 8, предоставляет классы для обработки дат и времени без информации о часовом поясе. LocalDate представляет дату, LocalTime представляет время, а LocalDateTime объединяет оба.

## Требования:
1. Создайте и отобразите объекты LocalDate:
   1. Получить текущую дату
   2. Создать конкретные даты используя of()
   3. Разобрать даты из строк

2. Работа с LocalTime:
   1. Получить текущее время
   2. Создать конкретное время
   3. Извлечь часы, минуты, секунды

3. Объединить с LocalDateTime:
   1. Получить текущую дату и время
   2. Создать из компонентов даты и времени
   3. Извлечь части даты и времени

4. Отобразить различные свойства даты/времени (год, месяц, день, час, минута)

## Пример вывода:
\`\`\`
=== LocalDate Examples ===
Current date: 2024-12-10
Specific date: 2025-06-15
Parsed date: 2024-01-01
Year: 2024, Month: DECEMBER, Day: 10

=== LocalTime Examples ===
Current time: 14:30:45.123456789
Specific time: 09:30:00
Hour: 14, Minute: 30, Second: 45

=== LocalDateTime Examples ===
Current date-time: 2024-12-10T14:30:45.123456789
Specific date-time: 2025-06-15T09:30:00
Date part: 2025-06-15
Time part: 09:30:00
\`\`\``,
            hint1: `Используйте LocalDate.now() для текущей даты, LocalDate.of(год, месяц, день) для конкретных дат и LocalDate.parse() для разбора из строк.`,
            hint2: `LocalDateTime объединяет дату и время. Вы можете создать его с помощью LocalDateTime.of() или объединить объекты LocalDate и LocalTime.`,
            whyItMatters: `Пакет java.time предоставляет современный, неизменяемый и потокобезопасный API для работы с датами и временем. В отличие от старых классов Date и Calendar, эти новые классы просты в использовании, следуют лучшим практикам и элегантно обрабатывают обычные операции с датой/временем. Они необходимы для любого Java-приложения, работающего с временными данными.

**Продакшен паттерн:**
\`\`\`java
// Логирование с timestamp
LocalDateTime now = LocalDateTime.now();
System.out.println("[" + now + "] User login successful");

// Планирование задач
LocalDate nextMeeting = LocalDate.now().plusDays(7);
LocalTime meetingTime = LocalTime.of(14, 30);
LocalDateTime meeting = LocalDateTime.of(nextMeeting, meetingTime);
\`\`\`

**Практические преимущества:**
- Неизменяемость предотвращает баги при многопоточности
- Четкое разделение даты, времени и даты-времени
- Интуитивный API без неожиданного поведения`
        },
        uz: {
            title: `LocalDate, LocalTime, LocalDateTime`,
            solutionCode: `import java.time.LocalDate;
import java.time.LocalTime;
import java.time.LocalDateTime;

public class LocalDateTimeBasics {
    public static void main(String[] args) {
        System.out.println("=== LocalDate namunalari ===");

        // Joriy sanani olish
        LocalDate currentDate = LocalDate.now();
        System.out.println("Current date: " + currentDate);

        // Aniq sanani yaratish
        LocalDate specificDate = LocalDate.of(2025, 6, 15);
        System.out.println("Specific date: " + specificDate);

        // Satrdan sanani tahlil qilish
        LocalDate parsedDate = LocalDate.parse("2024-01-01");
        System.out.println("Parsed date: " + parsedDate);

        // Sana komponentlarini ajratib olish
        System.out.println("Year: " + currentDate.getYear() +
            ", Month: " + currentDate.getMonth() +
            ", Day: " + currentDate.getDayOfMonth());

        System.out.println("\\n=== LocalTime namunalari ===");

        // Joriy vaqtni olish
        LocalTime currentTime = LocalTime.now();
        System.out.println("Current time: " + currentTime);

        // Aniq vaqtni yaratish
        LocalTime specificTime = LocalTime.of(9, 30, 0);
        System.out.println("Specific time: " + specificTime);

        // Vaqt komponentlarini ajratib olish
        System.out.println("Hour: " + currentTime.getHour() +
            ", Minute: " + currentTime.getMinute() +
            ", Second: " + currentTime.getSecond());

        System.out.println("\\n=== LocalDateTime namunalari ===");

        // Joriy sana va vaqtni olish
        LocalDateTime currentDateTime = LocalDateTime.now();
        System.out.println("Current date-time: " + currentDateTime);

        // Aniq sana-vaqtni yaratish
        LocalDateTime specificDateTime = LocalDateTime.of(2025, 6, 15, 9, 30, 0);
        System.out.println("Specific date-time: " + specificDateTime);

        // LocalDate va LocalTime-ni birlashtirish
        LocalDateTime combined = LocalDateTime.of(specificDate, specificTime);
        System.out.println("Combined date-time: " + combined);

        // Sana va vaqt qismlarini ajratib olish
        System.out.println("Date part: " + specificDateTime.toLocalDate());
        System.out.println("Time part: " + specificDateTime.toLocalTime());
    }
}`,
            description: `# LocalDate, LocalTime, LocalDateTime

Java 8-da kiritilgan Java Sana va Vaqt API (java.time) vaqt mintaqasi ma'lumotisiz sana va vaqtlarni boshqarish uchun klasslarni taqdim etadi. LocalDate sanani, LocalTime vaqtni, LocalDateTime esa ikkalasini birlashtiradi.

## Talablar:
1. LocalDate obyektlarini yarating va ko'rsating:
   1. Joriy sanani oling
   2. of() yordamida aniq sanalar yarating
   3. Satrlardan sanalarni tahlil qiling

2. LocalTime bilan ishlang:
   1. Joriy vaqtni oling
   2. Aniq vaqtlar yarating
   3. Soat, daqiqa, soniyalarni ajratib oling

3. LocalDateTime bilan birlashing:
   1. Joriy sana va vaqtni oling
   2. Sana va vaqt komponentlaridan yarating
   3. Sana va vaqt qismlarini ajratib oling

4. Turli sana/vaqt xususiyatlarini ko'rsating (yil, oy, kun, soat, daqiqa)

## Chiqish namunasi:
\`\`\`
=== LocalDate Examples ===
Current date: 2024-12-10
Specific date: 2025-06-15
Parsed date: 2024-01-01
Year: 2024, Month: DECEMBER, Day: 10

=== LocalTime Examples ===
Current time: 14:30:45.123456789
Specific time: 09:30:00
Hour: 14, Minute: 30, Second: 45

=== LocalDateTime Examples ===
Current date-time: 2024-12-10T14:30:45.123456789
Specific date-time: 2025-06-15T09:30:00
Date part: 2025-06-15
Time part: 09:30:00
\`\`\``,
            hint1: `Joriy sana uchun LocalDate.now(), aniq sanalar uchun LocalDate.of(yil, oy, kun) va satrlardan tahlil qilish uchun LocalDate.parse() dan foydalaning.`,
            hint2: `LocalDateTime sana va vaqtni birlashtiradi. Uni LocalDateTime.of() bilan yaratishingiz yoki LocalDate va LocalTime obyektlarini birlashtirishingiz mumkin.`,
            whyItMatters: `java.time paketi sana va vaqt bilan ishlash uchun zamonaviy, o'zgarmas va thread-safe API ni taqdim etadi. Eski Date va Calendar klasslaridan farqli o'laroq, bu yangi klasslar ishlatish oson, eng yaxshi amaliyotlarga amal qiladi va oddiy sana/vaqt operatsiyalarini nafis tarzda boshqaradi. Ular vaqtinchalik ma'lumotlar bilan ishlaydigan har qanday Java ilovasi uchun zarurdir.

**Ishlab chiqarish patterni:**
\`\`\`java
// Timestamp bilan loglash
LocalDateTime now = LocalDateTime.now();
System.out.println("[" + now + "] User login successful");

// Vazifalarni rejalashtirish
LocalDate nextMeeting = LocalDate.now().plusDays(7);
LocalTime meetingTime = LocalTime.of(14, 30);
LocalDateTime meeting = LocalDateTime.of(nextMeeting, meetingTime);
\`\`\`

**Amaliy foydalari:**
- O'zgarmaslik ko'p oqimlilikda xatolarni oldini oladi
- Sana, vaqt va sana-vaqtning aniq bo'linishi
- Kutilmagan xatti-harakatlarsiz intuitiv API`
        }
    }
};

export default task;
