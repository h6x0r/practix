import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-date-manipulation',
    title: 'Date Manipulation - plus, minus, with, TemporalAdjusters',
    difficulty: 'medium',
    tags: ['java', 'date-time', 'manipulation', 'temporal-adjusters', 'date-operations'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Date Manipulation - plus, minus, with, TemporalAdjusters

Date-time objects are immutable, so all manipulation methods return new instances. You can add/subtract time units, replace specific fields, and use TemporalAdjusters for complex date calculations like "first day of month" or "next Monday".

## Requirements:
1. Use plus methods:
   1. plusDays(), plusMonths(), plusYears()
   2. plusHours(), plusMinutes(), plusSeconds()
   3. plus(Period), plus(Duration)

2. Use minus methods:
   1. minusDays(), minusMonths(), minusYears()
   2. minusHours(), minusMinutes()
   3. Calculate dates in the past

3. Use with methods:
   1. withYear(), withMonth(), withDayOfMonth()
   2. withHour(), withMinute(), withSecond()
   3. Replace specific components

4. Use TemporalAdjusters:
   1. firstDayOfMonth(), lastDayOfMonth()
   2. firstDayOfYear(), lastDayOfYear()
   3. next(), previous(), nextOrSame()
   4. Custom adjusters for business logic

## Example Output:
\`\`\`
=== Plus Operations ===
Original date: 2024-12-10
Plus 5 days: 2024-12-15
Plus 2 months: 2025-02-10
Plus 1 year: 2025-12-10
Plus 3 hours: 2024-12-10T17:30:45

=== Minus Operations ===
Minus 3 days: 2024-12-07
Minus 6 months: 2024-06-10
Minus 2 years: 2022-12-10
90 days ago: 2024-09-11

=== With Operations ===
With year 2025: 2025-12-10
With month 6: 2024-06-10
With day 1: 2024-12-01
Set time to noon: 2024-12-10T12:00:00

=== TemporalAdjusters ===
First day of month: 2024-12-01
Last day of month: 2024-12-31
First day of next month: 2025-01-01
Next Monday: 2024-12-16
Previous Friday: 2024-12-06
First Monday of month: 2024-12-02
\`\`\``,
    initialCode: `// TODO: Import java.time classes

public class DateManipulation {
    public static void main(String[] args) {
        // TODO: Use plus methods

        // TODO: Use minus methods

        // TODO: Use with methods

        // TODO: Use TemporalAdjusters
    }
}`,
    solutionCode: `import java.time.*;
import java.time.temporal.ChronoUnit;
import java.time.temporal.TemporalAdjusters;
import java.time.DayOfWeek;

public class DateManipulation {
    public static void main(String[] args) {
        LocalDate date = LocalDate.of(2024, 12, 10);
        LocalTime time = LocalTime.of(14, 30, 45);
        LocalDateTime dateTime = LocalDateTime.of(date, time);

        System.out.println("=== Plus Operations ===");
        System.out.println("Original date: " + date);

        // Add days, months, years
        LocalDate plusDays = date.plusDays(5);
        System.out.println("Plus 5 days: " + plusDays);

        LocalDate plusMonths = date.plusMonths(2);
        System.out.println("Plus 2 months: " + plusMonths);

        LocalDate plusYears = date.plusYears(1);
        System.out.println("Plus 1 year: " + plusYears);

        // Add time units
        LocalDateTime plusHours = dateTime.plusHours(3);
        System.out.println("Plus 3 hours: " + plusHours);

        LocalDateTime plusMinutes = dateTime.plusMinutes(45);
        System.out.println("Plus 45 minutes: " + plusMinutes);

        // Add period
        LocalDate plusPeriod = date.plus(Period.ofWeeks(2));
        System.out.println("Plus 2 weeks: " + plusPeriod);

        System.out.println("\\n=== Minus Operations ===");

        // Subtract days, months, years
        LocalDate minusDays = date.minusDays(3);
        System.out.println("Minus 3 days: " + minusDays);

        LocalDate minusMonths = date.minusMonths(6);
        System.out.println("Minus 6 months: " + minusMonths);

        LocalDate minusYears = date.minusYears(2);
        System.out.println("Minus 2 years: " + minusYears);

        // Calculate past dates
        LocalDate ninetyDaysAgo = date.minus(90, ChronoUnit.DAYS);
        System.out.println("90 days ago: " + ninetyDaysAgo);

        System.out.println("\\n=== With Operations ===");

        // Replace specific components
        LocalDate withYear = date.withYear(2025);
        System.out.println("With year 2025: " + withYear);

        LocalDate withMonth = date.withMonth(6);
        System.out.println("With month 6: " + withMonth);

        LocalDate withDay = date.withDayOfMonth(1);
        System.out.println("With day 1: " + withDay);

        // Replace time components
        LocalDateTime noonTime = dateTime.withHour(12).withMinute(0).withSecond(0);
        System.out.println("Set time to noon: " + noonTime);

        LocalDateTime midnight = dateTime.with(LocalTime.MIDNIGHT);
        System.out.println("Set to midnight: " + midnight);

        System.out.println("\\n=== TemporalAdjusters ===");

        // First and last day of month
        LocalDate firstDayOfMonth = date.with(TemporalAdjusters.firstDayOfMonth());
        System.out.println("First day of month: " + firstDayOfMonth);

        LocalDate lastDayOfMonth = date.with(TemporalAdjusters.lastDayOfMonth());
        System.out.println("Last day of month: " + lastDayOfMonth);

        LocalDate firstDayOfNextMonth = date
            .with(TemporalAdjusters.firstDayOfNextMonth());
        System.out.println("First day of next month: " + firstDayOfNextMonth);

        // First and last day of year
        LocalDate firstDayOfYear = date.with(TemporalAdjusters.firstDayOfYear());
        System.out.println("First day of year: " + firstDayOfYear);

        LocalDate lastDayOfYear = date.with(TemporalAdjusters.lastDayOfYear());
        System.out.println("Last day of year: " + lastDayOfYear);

        // Next and previous day of week
        LocalDate nextMonday = date.with(TemporalAdjusters.next(DayOfWeek.MONDAY));
        System.out.println("Next Monday: " + nextMonday);

        LocalDate previousFriday = date
            .with(TemporalAdjusters.previous(DayOfWeek.FRIDAY));
        System.out.println("Previous Friday: " + previousFriday);

        LocalDate nextOrSameMonday = date
            .with(TemporalAdjusters.nextOrSame(DayOfWeek.MONDAY));
        System.out.println("Next or same Monday: " + nextOrSameMonday);

        // First day of week in month
        LocalDate firstMondayOfMonth = date
            .with(TemporalAdjusters.firstInMonth(DayOfWeek.MONDAY));
        System.out.println("First Monday of month: " + firstMondayOfMonth);

        LocalDate lastFridayOfMonth = date
            .with(TemporalAdjusters.lastInMonth(DayOfWeek.FRIDAY));
        System.out.println("Last Friday of month: " + lastFridayOfMonth);

        // Practical examples
        System.out.println("\\n=== Practical Examples ===");

        // Calculate due date (30 days from now)
        LocalDate dueDate = LocalDate.now().plusDays(30);
        System.out.println("Due date (30 days): " + dueDate);

        // Next business day (skip weekends)
        LocalDate nextBusinessDay = LocalDate.now()
            .with(TemporalAdjusters.next(DayOfWeek.MONDAY));
        if (LocalDate.now().getDayOfWeek() == DayOfWeek.FRIDAY) {
            nextBusinessDay = LocalDate.now().plusDays(3);
        }
        System.out.println("Next business day: " + nextBusinessDay);

        // End of quarter
        LocalDate endOfQuarter = date.withMonth(12).with(TemporalAdjusters.lastDayOfMonth());
        System.out.println("End of quarter: " + endOfQuarter);
    }
}`,
    hint1: `All date-time objects are immutable. Methods like plus(), minus(), and with() return new instances - they don't modify the original.`,
    hint2: `TemporalAdjusters provide common date adjustments. Use TemporalAdjusters.next(DayOfWeek) to find the next occurrence of a specific day.`,
    whyItMatters: `Date manipulation is fundamental for business logic: calculating due dates, scheduling events, determining billing periods, and handling deadlines. TemporalAdjusters simplify complex date calculations like "first Monday of next month" that would otherwise require error-prone manual calculations. Understanding immutability prevents bugs where developers expect mutations but get new objects instead.`,
    order: 5,
    testCode: `import java.time.*;
import java.time.temporal.TemporalAdjusters;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

// Test1: plusDays() adds days
class Test1 {
    @Test
    void testPlusDays() {
        LocalDate date = LocalDate.of(2024, 12, 10);
        LocalDate result = date.plusDays(5);
        assertEquals(LocalDate.of(2024, 12, 15), result);
    }
}

// Test2: plusMonths() adds months
class Test2 {
    @Test
    void testPlusMonths() {
        LocalDate date = LocalDate.of(2024, 12, 10);
        LocalDate result = date.plusMonths(2);
        assertEquals(LocalDate.of(2025, 2, 10), result);
    }
}

// Test3: minusDays() subtracts days
class Test3 {
    @Test
    void testMinusDays() {
        LocalDate date = LocalDate.of(2024, 12, 10);
        LocalDate result = date.minusDays(3);
        assertEquals(LocalDate.of(2024, 12, 7), result);
    }
}

// Test4: withYear() replaces year
class Test4 {
    @Test
    void testWithYear() {
        LocalDate date = LocalDate.of(2024, 12, 10);
        LocalDate result = date.withYear(2025);
        assertEquals(2025, result.getYear());
        assertEquals(12, result.getMonthValue());
    }
}

// Test5: withDayOfMonth() replaces day
class Test5 {
    @Test
    void testWithDayOfMonth() {
        LocalDate date = LocalDate.of(2024, 12, 10);
        LocalDate result = date.withDayOfMonth(1);
        assertEquals(LocalDate.of(2024, 12, 1), result);
    }
}

// Test6: firstDayOfMonth() adjuster
class Test6 {
    @Test
    void testFirstDayOfMonth() {
        LocalDate date = LocalDate.of(2024, 12, 10);
        LocalDate result = date.with(TemporalAdjusters.firstDayOfMonth());
        assertEquals(LocalDate.of(2024, 12, 1), result);
    }
}

// Test7: lastDayOfMonth() adjuster
class Test7 {
    @Test
    void testLastDayOfMonth() {
        LocalDate date = LocalDate.of(2024, 12, 10);
        LocalDate result = date.with(TemporalAdjusters.lastDayOfMonth());
        assertEquals(LocalDate.of(2024, 12, 31), result);
    }
}

// Test8: next(DayOfWeek) adjuster
class Test8 {
    @Test
    void testNextDayOfWeek() {
        LocalDate date = LocalDate.of(2024, 12, 10); // Tuesday
        LocalDate result = date.with(TemporalAdjusters.next(DayOfWeek.MONDAY));
        assertEquals(DayOfWeek.MONDAY, result.getDayOfWeek());
        assertTrue(result.isAfter(date));
    }
}

// Test9: Original date is immutable
class Test9 {
    @Test
    void testImmutability() {
        LocalDate original = LocalDate.of(2024, 12, 10);
        LocalDate modified = original.plusDays(5);
        assertEquals(LocalDate.of(2024, 12, 10), original);
        assertEquals(LocalDate.of(2024, 12, 15), modified);
    }
}

// Test10: firstDayOfNextMonth() adjuster
class Test10 {
    @Test
    void testFirstDayOfNextMonth() {
        LocalDate date = LocalDate.of(2024, 12, 10);
        LocalDate result = date.with(TemporalAdjusters.firstDayOfNextMonth());
        assertEquals(LocalDate.of(2025, 1, 1), result);
    }
}
`,
    translations: {
        ru: {
            title: 'Манипуляции с датами - plus, minus, with, TemporalAdjusters',
            solutionCode: `import java.time.*;
import java.time.temporal.ChronoUnit;
import java.time.temporal.TemporalAdjusters;
import java.time.DayOfWeek;

public class DateManipulation {
    public static void main(String[] args) {
        LocalDate date = LocalDate.of(2024, 12, 10);
        LocalTime time = LocalTime.of(14, 30, 45);
        LocalDateTime dateTime = LocalDateTime.of(date, time);

        System.out.println("=== Операции Plus ===");
        System.out.println("Original date: " + date);

        // Добавить дни, месяцы, годы
        LocalDate plusDays = date.plusDays(5);
        System.out.println("Plus 5 days: " + plusDays);

        LocalDate plusMonths = date.plusMonths(2);
        System.out.println("Plus 2 months: " + plusMonths);

        LocalDate plusYears = date.plusYears(1);
        System.out.println("Plus 1 year: " + plusYears);

        // Добавить единицы времени
        LocalDateTime plusHours = dateTime.plusHours(3);
        System.out.println("Plus 3 hours: " + plusHours);

        LocalDateTime plusMinutes = dateTime.plusMinutes(45);
        System.out.println("Plus 45 minutes: " + plusMinutes);

        // Добавить период
        LocalDate plusPeriod = date.plus(Period.ofWeeks(2));
        System.out.println("Plus 2 weeks: " + plusPeriod);

        System.out.println("\\n=== Операции Minus ===");

        // Вычесть дни, месяцы, годы
        LocalDate minusDays = date.minusDays(3);
        System.out.println("Minus 3 days: " + minusDays);

        LocalDate minusMonths = date.minusMonths(6);
        System.out.println("Minus 6 months: " + minusMonths);

        LocalDate minusYears = date.minusYears(2);
        System.out.println("Minus 2 years: " + minusYears);

        // Вычислить прошлые даты
        LocalDate ninetyDaysAgo = date.minus(90, ChronoUnit.DAYS);
        System.out.println("90 days ago: " + ninetyDaysAgo);

        System.out.println("\\n=== Операции With ===");

        // Заменить конкретные компоненты
        LocalDate withYear = date.withYear(2025);
        System.out.println("With year 2025: " + withYear);

        LocalDate withMonth = date.withMonth(6);
        System.out.println("With month 6: " + withMonth);

        LocalDate withDay = date.withDayOfMonth(1);
        System.out.println("With day 1: " + withDay);

        // Заменить компоненты времени
        LocalDateTime noonTime = dateTime.withHour(12).withMinute(0).withSecond(0);
        System.out.println("Set time to noon: " + noonTime);

        LocalDateTime midnight = dateTime.with(LocalTime.MIDNIGHT);
        System.out.println("Set to midnight: " + midnight);

        System.out.println("\\n=== TemporalAdjusters ===");

        // Первый и последний день месяца
        LocalDate firstDayOfMonth = date.with(TemporalAdjusters.firstDayOfMonth());
        System.out.println("First day of month: " + firstDayOfMonth);

        LocalDate lastDayOfMonth = date.with(TemporalAdjusters.lastDayOfMonth());
        System.out.println("Last day of month: " + lastDayOfMonth);

        LocalDate firstDayOfNextMonth = date
            .with(TemporalAdjusters.firstDayOfNextMonth());
        System.out.println("First day of next month: " + firstDayOfNextMonth);

        // Первый и последний день года
        LocalDate firstDayOfYear = date.with(TemporalAdjusters.firstDayOfYear());
        System.out.println("First day of year: " + firstDayOfYear);

        LocalDate lastDayOfYear = date.with(TemporalAdjusters.lastDayOfYear());
        System.out.println("Last day of year: " + lastDayOfYear);

        // Следующий и предыдущий день недели
        LocalDate nextMonday = date.with(TemporalAdjusters.next(DayOfWeek.MONDAY));
        System.out.println("Next Monday: " + nextMonday);

        LocalDate previousFriday = date
            .with(TemporalAdjusters.previous(DayOfWeek.FRIDAY));
        System.out.println("Previous Friday: " + previousFriday);

        LocalDate nextOrSameMonday = date
            .with(TemporalAdjusters.nextOrSame(DayOfWeek.MONDAY));
        System.out.println("Next or same Monday: " + nextOrSameMonday);

        // Первый день недели в месяце
        LocalDate firstMondayOfMonth = date
            .with(TemporalAdjusters.firstInMonth(DayOfWeek.MONDAY));
        System.out.println("First Monday of month: " + firstMondayOfMonth);

        LocalDate lastFridayOfMonth = date
            .with(TemporalAdjusters.lastInMonth(DayOfWeek.FRIDAY));
        System.out.println("Last Friday of month: " + lastFridayOfMonth);

        // Практические примеры
        System.out.println("\\n=== Практические примеры ===");

        // Вычислить срок выполнения (30 дней с настоящего момента)
        LocalDate dueDate = LocalDate.now().plusDays(30);
        System.out.println("Due date (30 days): " + dueDate);

        // Следующий рабочий день (пропустить выходные)
        LocalDate nextBusinessDay = LocalDate.now()
            .with(TemporalAdjusters.next(DayOfWeek.MONDAY));
        if (LocalDate.now().getDayOfWeek() == DayOfWeek.FRIDAY) {
            nextBusinessDay = LocalDate.now().plusDays(3);
        }
        System.out.println("Next business day: " + nextBusinessDay);

        // Конец квартала
        LocalDate endOfQuarter = date.withMonth(12).with(TemporalAdjusters.lastDayOfMonth());
        System.out.println("End of quarter: " + endOfQuarter);
    }
}`,
            description: `# Манипуляции с датами - plus, minus, with, TemporalAdjusters

Объекты даты-времени неизменяемы, поэтому все методы манипуляции возвращают новые экземпляры. Вы можете добавлять/вычитать единицы времени, заменять конкретные поля и использовать TemporalAdjusters для сложных вычислений дат, таких как "первый день месяца" или "следующий понедельник".

## Требования:
1. Использовать методы plus:
   1. plusDays(), plusMonths(), plusYears()
   2. plusHours(), plusMinutes(), plusSeconds()
   3. plus(Period), plus(Duration)

2. Использовать методы minus:
   1. minusDays(), minusMonths(), minusYears()
   2. minusHours(), minusMinutes()
   3. Вычислить даты в прошлом

3. Использовать методы with:
   1. withYear(), withMonth(), withDayOfMonth()
   2. withHour(), withMinute(), withSecond()
   3. Заменить конкретные компоненты

4. Использовать TemporalAdjusters:
   1. firstDayOfMonth(), lastDayOfMonth()
   2. firstDayOfYear(), lastDayOfYear()
   3. next(), previous(), nextOrSame()
   4. Пользовательские adjusters для бизнес-логики

## Пример вывода:
\`\`\`
=== Plus Operations ===
Original date: 2024-12-10
Plus 5 days: 2024-12-15
Plus 2 months: 2025-02-10
Plus 1 year: 2025-12-10
Plus 3 hours: 2024-12-10T17:30:45

=== Minus Operations ===
Minus 3 days: 2024-12-07
Minus 6 months: 2024-06-10
Minus 2 years: 2022-12-10
90 days ago: 2024-09-11

=== With Operations ===
With year 2025: 2025-12-10
With month 6: 2024-06-10
With day 1: 2024-12-01
Set time to noon: 2024-12-10T12:00:00

=== TemporalAdjusters ===
First day of month: 2024-12-01
Last day of month: 2024-12-31
First day of next month: 2025-01-01
Next Monday: 2024-12-16
Previous Friday: 2024-12-06
First Monday of month: 2024-12-02
\`\`\``,
            hint1: `Все объекты даты-времени неизменяемы. Методы, такие как plus(), minus() и with(), возвращают новые экземпляры - они не изменяют оригинал.`,
            hint2: `TemporalAdjusters предоставляют общие корректировки дат. Используйте TemporalAdjusters.next(DayOfWeek) для поиска следующего вхождения конкретного дня.`,
            whyItMatters: `Манипуляции с датами являются фундаментальными для бизнес-логики: расчет сроков выполнения, планирование событий, определение периодов выставления счетов и обработка дедлайнов. TemporalAdjusters упрощают сложные вычисления дат, такие как "первый понедельник следующего месяца", которые в противном случае потребовали бы подверженных ошибкам ручных вычислений. Понимание неизменяемости предотвращает ошибки, когда разработчики ожидают мутаций, но получают новые объекты.`
        },
        uz: {
            title: `Sana bilan manipulyatsiya - plus, minus, with, TemporalAdjusters`,
            solutionCode: `import java.time.*;
import java.time.temporal.ChronoUnit;
import java.time.temporal.TemporalAdjusters;
import java.time.DayOfWeek;

public class DateManipulation {
    public static void main(String[] args) {
        LocalDate date = LocalDate.of(2024, 12, 10);
        LocalTime time = LocalTime.of(14, 30, 45);
        LocalDateTime dateTime = LocalDateTime.of(date, time);

        System.out.println("=== Plus operatsiyalari ===");
        System.out.println("Original date: " + date);

        // Kun, oy, yil qo'shish
        LocalDate plusDays = date.plusDays(5);
        System.out.println("Plus 5 days: " + plusDays);

        LocalDate plusMonths = date.plusMonths(2);
        System.out.println("Plus 2 months: " + plusMonths);

        LocalDate plusYears = date.plusYears(1);
        System.out.println("Plus 1 year: " + plusYears);

        // Vaqt birliklarini qo'shish
        LocalDateTime plusHours = dateTime.plusHours(3);
        System.out.println("Plus 3 hours: " + plusHours);

        LocalDateTime plusMinutes = dateTime.plusMinutes(45);
        System.out.println("Plus 45 minutes: " + plusMinutes);

        // Davr qo'shish
        LocalDate plusPeriod = date.plus(Period.ofWeeks(2));
        System.out.println("Plus 2 weeks: " + plusPeriod);

        System.out.println("\\n=== Minus operatsiyalari ===");

        // Kun, oy, yil ayirish
        LocalDate minusDays = date.minusDays(3);
        System.out.println("Minus 3 days: " + minusDays);

        LocalDate minusMonths = date.minusMonths(6);
        System.out.println("Minus 6 months: " + minusMonths);

        LocalDate minusYears = date.minusYears(2);
        System.out.println("Minus 2 years: " + minusYears);

        // O'tmish sanalarini hisoblash
        LocalDate ninetyDaysAgo = date.minus(90, ChronoUnit.DAYS);
        System.out.println("90 days ago: " + ninetyDaysAgo);

        System.out.println("\\n=== With operatsiyalari ===");

        // Aniq komponentlarni almashtirish
        LocalDate withYear = date.withYear(2025);
        System.out.println("With year 2025: " + withYear);

        LocalDate withMonth = date.withMonth(6);
        System.out.println("With month 6: " + withMonth);

        LocalDate withDay = date.withDayOfMonth(1);
        System.out.println("With day 1: " + withDay);

        // Vaqt komponentlarini almashtirish
        LocalDateTime noonTime = dateTime.withHour(12).withMinute(0).withSecond(0);
        System.out.println("Set time to noon: " + noonTime);

        LocalDateTime midnight = dateTime.with(LocalTime.MIDNIGHT);
        System.out.println("Set to midnight: " + midnight);

        System.out.println("\\n=== TemporalAdjusters ===");

        // Oyning birinchi va oxirgi kuni
        LocalDate firstDayOfMonth = date.with(TemporalAdjusters.firstDayOfMonth());
        System.out.println("First day of month: " + firstDayOfMonth);

        LocalDate lastDayOfMonth = date.with(TemporalAdjusters.lastDayOfMonth());
        System.out.println("Last day of month: " + lastDayOfMonth);

        LocalDate firstDayOfNextMonth = date
            .with(TemporalAdjusters.firstDayOfNextMonth());
        System.out.println("First day of next month: " + firstDayOfNextMonth);

        // Yilning birinchi va oxirgi kuni
        LocalDate firstDayOfYear = date.with(TemporalAdjusters.firstDayOfYear());
        System.out.println("First day of year: " + firstDayOfYear);

        LocalDate lastDayOfYear = date.with(TemporalAdjusters.lastDayOfYear());
        System.out.println("Last day of year: " + lastDayOfYear);

        // Keyingi va oldingi hafta kuni
        LocalDate nextMonday = date.with(TemporalAdjusters.next(DayOfWeek.MONDAY));
        System.out.println("Next Monday: " + nextMonday);

        LocalDate previousFriday = date
            .with(TemporalAdjusters.previous(DayOfWeek.FRIDAY));
        System.out.println("Previous Friday: " + previousFriday);

        LocalDate nextOrSameMonday = date
            .with(TemporalAdjusters.nextOrSame(DayOfWeek.MONDAY));
        System.out.println("Next or same Monday: " + nextOrSameMonday);

        // Oydagi hafta kunining birinchi kuni
        LocalDate firstMondayOfMonth = date
            .with(TemporalAdjusters.firstInMonth(DayOfWeek.MONDAY));
        System.out.println("First Monday of month: " + firstMondayOfMonth);

        LocalDate lastFridayOfMonth = date
            .with(TemporalAdjusters.lastInMonth(DayOfWeek.FRIDAY));
        System.out.println("Last Friday of month: " + lastFridayOfMonth);

        // Amaliy misollar
        System.out.println("\\n=== Amaliy misollar ===");

        // Tugash muddatini hisoblash (hozirdan 30 kun)
        LocalDate dueDate = LocalDate.now().plusDays(30);
        System.out.println("Due date (30 days): " + dueDate);

        // Keyingi ish kuni (dam olish kunlarini o'tkazib yuborish)
        LocalDate nextBusinessDay = LocalDate.now()
            .with(TemporalAdjusters.next(DayOfWeek.MONDAY));
        if (LocalDate.now().getDayOfWeek() == DayOfWeek.FRIDAY) {
            nextBusinessDay = LocalDate.now().plusDays(3);
        }
        System.out.println("Next business day: " + nextBusinessDay);

        // Chorak oxiri
        LocalDate endOfQuarter = date.withMonth(12).with(TemporalAdjusters.lastDayOfMonth());
        System.out.println("End of quarter: " + endOfQuarter);
    }
}`,
            description: `# Sana bilan manipulyatsiya - plus, minus, with, TemporalAdjusters

Sana-vaqt obyektlari o'zgarmasdir, shuning uchun barcha manipulyatsiya usullari yangi nusxalarni qaytaradi. Siz vaqt birliklarini qo'shishingiz/ayirishingiz, aniq maydonlarni almashtirishingiz va murakkab sana hisoblar uchun TemporalAdjusters dan foydalanishingiz mumkin, masalan "oyning birinchi kuni" yoki "keyingi dushanba".

## Talablar:
1. Plus usullaridan foydalaning:
   1. plusDays(), plusMonths(), plusYears()
   2. plusHours(), plusMinutes(), plusSeconds()
   3. plus(Period), plus(Duration)

2. Minus usullaridan foydalaning:
   1. minusDays(), minusMonths(), minusYears()
   2. minusHours(), minusMinutes()
   3. O'tmishdagi sanalarni hisoblash

3. With usullaridan foydalaning:
   1. withYear(), withMonth(), withDayOfMonth()
   2. withHour(), withMinute(), withSecond()
   3. Aniq komponentlarni almashtirish

4. TemporalAdjusters dan foydalaning:
   1. firstDayOfMonth(), lastDayOfMonth()
   2. firstDayOfYear(), lastDayOfYear()
   3. next(), previous(), nextOrSame()
   4. Biznes mantiq uchun maxsus adjusterlar

## Chiqish namunasi:
\`\`\`
=== Plus Operations ===
Original date: 2024-12-10
Plus 5 days: 2024-12-15
Plus 2 months: 2025-02-10
Plus 1 year: 2025-12-10
Plus 3 hours: 2024-12-10T17:30:45

=== Minus Operations ===
Minus 3 days: 2024-12-07
Minus 6 months: 2024-06-10
Minus 2 years: 2022-12-10
90 days ago: 2024-09-11

=== With Operations ===
With year 2025: 2025-12-10
With month 6: 2024-06-10
With day 1: 2024-12-01
Set time to noon: 2024-12-10T12:00:00

=== TemporalAdjusters ===
First day of month: 2024-12-01
Last day of month: 2024-12-31
First day of next month: 2025-01-01
Next Monday: 2024-12-16
Previous Friday: 2024-12-06
First Monday of month: 2024-12-02
\`\`\``,
            hint1: `Barcha sana-vaqt obyektlari o'zgarmasdir. plus(), minus() va with() kabi usullar yangi nusxalarni qaytaradi - ular asl nusxani o'zgartirmaydi.`,
            hint2: `TemporalAdjusters umumiy sana tuzatishlarini taqdim etadi. Aniq kunning keyingi sodir bo'lishini topish uchun TemporalAdjusters.next(DayOfWeek) dan foydalaning.`,
            whyItMatters: `Sana bilan manipulyatsiya biznes mantiq uchun asosiydir: tugash muddatlarini hisoblash, voqealarni rejalashtirish, hisob-kitob davrlarini aniqlash va muddatlarni boshqarish. TemporalAdjusters murakkab sana hisoblarni soddalashtiradi, masalan "keyingi oyning birinchi dushanba kuni", aks holda xatolarga moyil qo'lda hisob-kitoblarni talab qiladi. O'zgarmaslikni tushunish dasturchilar mutatsiyalarni kutganlarida, lekin yangi obyektlarni olganlarida xatolarning oldini oladi.`
        }
    }
};

export default task;
