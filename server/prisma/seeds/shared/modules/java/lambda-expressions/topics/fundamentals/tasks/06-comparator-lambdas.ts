import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-comparator-lambdas',
    title: 'Comparators with Lambdas',
    difficulty: 'easy',
    tags: ['java', 'lambda', 'comparator', 'sorting', 'collections'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Comparators with Lambdas

Comparator is a functional interface perfect for lambda expressions. Java 8 enhanced the Comparator interface with static and default methods that make sorting collections much easier. Understanding Comparator lambdas is essential for working with collections and streams.

## Requirements:
1. Basic comparator lambdas:
   1.1. Natural order sorting
   1.2. Reverse order sorting
   1.3. Custom comparison logic
   1.4. Compare by multiple fields

2. Comparator static methods:
   2.1. \`Comparator.comparing()\`: Extract and compare key
   2.2. \`Comparator.naturalOrder()\`: Natural ordering
   2.3. \`Comparator.reverseOrder()\`: Reverse natural ordering
   2.4. \`Comparator.nullsFirst()\`, \`nullsLast()\`: Null handling

3. Chaining comparators:
   3.1. \`thenComparing()\`: Secondary sort criteria
   3.2. Multiple level sorting
   3.3. Complex sorting scenarios

4. Practical examples:
   4.1. Sort list of Person objects by name, age
   4.2. Sort strings by length then alphabetically
   4.3. Handle null values in sorting

## Example Output:
\`\`\`
=== Basic Comparator Lambdas ===
Natural order: [1, 2, 3, 4, 5]
Reverse order: [5, 4, 3, 2, 1]
By length: [cat, dog, bird, elephant]

=== Comparator Static Methods ===
By name: [Alice, Bob, Charlie]
By age: [Bob(25), Alice(30), Charlie(35)]
Reverse by age: [Charlie(35), Alice(30), Bob(25)]

=== Chaining Comparators ===
By age, then name: [Bob(25), Alice(30), Charlie(30)]
By length, then alphabetically: [cat, dog, bird, apple]

=== Null Handling ===
Nulls first: [null, Apple, Banana, Cherry]
Nulls last: [Apple, Banana, Cherry, null]
\`\`\``,
    initialCode: `import java.util.*;

class Person {
    String name;
    int age;

    Person(String name, int age) {
        this.name = name;
        this.age = age;
    }

    @Override
    public String toString() {
        return name + "(" + age + ")";
    }
}

public class ComparatorLambdas {
    public static void main(String[] args) {
        // TODO: Create list of numbers and demonstrate basic sorting

        // TODO: Create list of Persons and use Comparator.comparing()

        // TODO: Demonstrate chaining with thenComparing()

        // TODO: Show null handling with nullsFirst/nullsLast
    }
}`,
    solutionCode: `import java.util.*;

class Person {
    String name;
    int age;

    Person(String name, int age) {
        this.name = name;
        this.age = age;
    }

    @Override
    public String toString() {
        return name + "(" + age + ")";
    }
}

public class ComparatorLambdas {
    public static void main(String[] args) {
        System.out.println("=== Basic Comparator Lambdas ===");

        // Natural order sorting
        List<Integer> numbers = new ArrayList<>(Arrays.asList(3, 1, 4, 1, 5, 9, 2));
        numbers.sort((a, b) -> a.compareTo(b));
        // Or simply: numbers.sort(Integer::compareTo);
        System.out.println("Natural order: " + numbers);

        // Reverse order using lambda
        List<Integer> reversed = new ArrayList<>(Arrays.asList(3, 1, 4, 1, 5, 9, 2));
        reversed.sort((a, b) -> b.compareTo(a));
        System.out.println("Reverse order: " + reversed);

        // Custom comparison - sort strings by length
        List<String> words = new ArrayList<>(Arrays.asList("elephant", "cat", "dog", "bird"));
        words.sort((s1, s2) -> Integer.compare(s1.length(), s2.length()));
        System.out.println("By length: " + words);

        System.out.println("\\n=== Comparator Static Methods ===");

        List<Person> people = Arrays.asList(
            new Person("Charlie", 35),
            new Person("Alice", 30),
            new Person("Bob", 25)
        );

        // Sort by name using Comparator.comparing()
        List<Person> byName = new ArrayList<>(people);
        byName.sort(Comparator.comparing(p -> p.name));
        // Or with method reference: byName.sort(Comparator.comparing(Person::getName));
        System.out.println("By name: " + byName);

        // Sort by age
        List<Person> byAge = new ArrayList<>(people);
        byAge.sort(Comparator.comparing(p -> p.age));
        System.out.println("By age: " + byAge);

        // Reverse order by age
        List<Person> byAgeReverse = new ArrayList<>(people);
        byAgeReverse.sort(Comparator.comparing((Person p) -> p.age).reversed());
        System.out.println("Reverse by age: " + byAgeReverse);

        System.out.println("\\n=== Chaining Comparators ===");

        List<Person> multiSort = Arrays.asList(
            new Person("Charlie", 30),
            new Person("Alice", 30),
            new Person("Bob", 25)
        );

        // Sort by age, then by name for same age
        multiSort.sort(Comparator.comparing((Person p) -> p.age)
                                .thenComparing(p -> p.name));
        System.out.println("By age, then name: " + multiSort);

        // Sort strings by length, then alphabetically
        List<String> words2 = new ArrayList<>(Arrays.asList("dog", "cat", "bird", "apple"));
        words2.sort(Comparator.comparing((String s) -> s.length())
                             .thenComparing(s -> s));
        System.out.println("By length, then alphabetically: " + words2);

        System.out.println("\\n=== Null Handling ===");

        List<String> withNulls = new ArrayList<>(Arrays.asList("Banana", null, "Apple", "Cherry"));

        // Nulls first
        List<String> nullsFirst = new ArrayList<>(withNulls);
        nullsFirst.sort(Comparator.nullsFirst(Comparator.naturalOrder()));
        System.out.println("Nulls first: " + nullsFirst);

        // Nulls last
        List<String> nullsLast = new ArrayList<>(withNulls);
        nullsLast.sort(Comparator.nullsLast(Comparator.naturalOrder()));
        System.out.println("Nulls last: " + nullsLast);
    }
}`,
    hint1: `Comparator.comparing() extracts a key and compares it. Use thenComparing() to add secondary sort criteria. This is much cleaner than writing complex comparison logic manually.`,
    hint2: `For null handling, wrap your comparator with Comparator.nullsFirst() or nullsLast(). This prevents NullPointerException and gives you control over null ordering.`,
    whyItMatters: `Comparator lambdas make sorting code concise and readable. The Comparator API with comparing(), thenComparing(), and null handling methods eliminates boilerplate code and reduces bugs. This is essential knowledge for working with collections, streams, and any scenario requiring custom sorting logic. Modern Java code relies heavily on these patterns.`,
    order: 6,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.*;

// Test1: Verify natural order sorting with lambda
class Test1 {
    @Test
    public void test() {
        List<Integer> nums = new ArrayList<>(Arrays.asList(3, 1, 4, 1, 5));
        nums.sort((a, b) -> a.compareTo(b));
        assertEquals(Integer.valueOf(1), nums.get(0));
        assertEquals(Integer.valueOf(5), nums.get(nums.size() - 1));
    }
}

// Test2: Verify reverse order sorting
class Test2 {
    @Test
    public void test() {
        List<Integer> nums = new ArrayList<>(Arrays.asList(3, 1, 4, 1, 5));
        nums.sort((a, b) -> b.compareTo(a));
        assertEquals(Integer.valueOf(5), nums.get(0));
    }
}

// Test3: Verify Comparator.comparing() with Person
class Test3 {
    @Test
    public void test() {
        List<Person> people = Arrays.asList(
            new Person("Charlie", 35),
            new Person("Alice", 30),
            new Person("Bob", 25)
        );
        people.sort(Comparator.comparing(p -> p.name));
        assertEquals("Alice", people.get(0).name);
    }
}

// Test4: Verify sorting by age
class Test4 {
    @Test
    public void test() {
        List<Person> people = Arrays.asList(
            new Person("Charlie", 35),
            new Person("Alice", 30),
            new Person("Bob", 25)
        );
        people.sort(Comparator.comparing(p -> p.age));
        assertEquals(25, people.get(0).age);
    }
}

// Test5: Verify reversed() method
class Test5 {
    @Test
    public void test() {
        List<Person> people = Arrays.asList(
            new Person("Charlie", 35),
            new Person("Alice", 30),
            new Person("Bob", 25)
        );
        people.sort(Comparator.comparing((Person p) -> p.age).reversed());
        assertEquals(35, people.get(0).age);
    }
}

// Test6: Verify thenComparing() chaining
class Test6 {
    @Test
    public void test() {
        List<Person> people = Arrays.asList(
            new Person("Charlie", 30),
            new Person("Alice", 30),
            new Person("Bob", 25)
        );
        people.sort(Comparator.comparing((Person p) -> p.age)
                             .thenComparing(p -> p.name));
        assertEquals("Alice", people.get(1).name);
    }
}

// Test7: Verify sorting strings by length
class Test7 {
    @Test
    public void test() {
        List<String> words = new ArrayList<>(Arrays.asList("elephant", "cat", "dog", "bird"));
        words.sort((s1, s2) -> Integer.compare(s1.length(), s2.length()));
        assertEquals("cat", words.get(0));
    }
}

// Test8: Verify Comparator.nullsFirst()
class Test8 {
    @Test
    public void test() {
        List<String> list = new ArrayList<>(Arrays.asList("Banana", null, "Apple"));
        list.sort(Comparator.nullsFirst(Comparator.naturalOrder()));
        assertNull(list.get(0));
    }
}

// Test9: Verify Comparator.nullsLast()
class Test9 {
    @Test
    public void test() {
        List<String> list = new ArrayList<>(Arrays.asList("Banana", null, "Apple"));
        list.sort(Comparator.nullsLast(Comparator.naturalOrder()));
        assertNull(list.get(list.size() - 1));
    }
}

// Test10: Verify custom comparison lambda
class Test10 {
    @Test
    public void test() {
        List<String> words = new ArrayList<>(Arrays.asList("dog", "cat", "bird", "apple"));
        words.sort((s1, s2) -> {
            int lenCompare = Integer.compare(s1.length(), s2.length());
            if (lenCompare != 0) return lenCompare;
            return s1.compareTo(s2);
        });
        assertEquals("cat", words.get(0));
    }
}`,
    translations: {
        ru: {
            title: 'Компараторы с лямбдами',
            solutionCode: `import java.util.*;

class Person {
    String name;
    int age;

    Person(String name, int age) {
        this.name = name;
        this.age = age;
    }

    @Override
    public String toString() {
        return name + "(" + age + ")";
    }
}

public class ComparatorLambdas {
    public static void main(String[] args) {
        System.out.println("=== Базовые лямбды компараторов ===");

        // Сортировка в естественном порядке
        List<Integer> numbers = new ArrayList<>(Arrays.asList(3, 1, 4, 1, 5, 9, 2));
        numbers.sort((a, b) -> a.compareTo(b));
        // Или просто: numbers.sort(Integer::compareTo);
        System.out.println("Natural order: " + numbers);

        // Обратный порядок используя лямбду
        List<Integer> reversed = new ArrayList<>(Arrays.asList(3, 1, 4, 1, 5, 9, 2));
        reversed.sort((a, b) -> b.compareTo(a));
        System.out.println("Reverse order: " + reversed);

        // Пользовательское сравнение - сортировка строк по длине
        List<String> words = new ArrayList<>(Arrays.asList("elephant", "cat", "dog", "bird"));
        words.sort((s1, s2) -> Integer.compare(s1.length(), s2.length()));
        System.out.println("By length: " + words);

        System.out.println("\\n=== Статические методы Comparator ===");

        List<Person> people = Arrays.asList(
            new Person("Charlie", 35),
            new Person("Alice", 30),
            new Person("Bob", 25)
        );

        // Сортировка по имени используя Comparator.comparing()
        List<Person> byName = new ArrayList<>(people);
        byName.sort(Comparator.comparing(p -> p.name));
        // Или со ссылкой на метод: byName.sort(Comparator.comparing(Person::getName));
        System.out.println("By name: " + byName);

        // Сортировка по возрасту
        List<Person> byAge = new ArrayList<>(people);
        byAge.sort(Comparator.comparing(p -> p.age));
        System.out.println("By age: " + byAge);

        // Обратный порядок по возрасту
        List<Person> byAgeReverse = new ArrayList<>(people);
        byAgeReverse.sort(Comparator.comparing((Person p) -> p.age).reversed());
        System.out.println("Reverse by age: " + byAgeReverse);

        System.out.println("\\n=== Цепочки компараторов ===");

        List<Person> multiSort = Arrays.asList(
            new Person("Charlie", 30),
            new Person("Alice", 30),
            new Person("Bob", 25)
        );

        // Сортировка по возрасту, затем по имени для одинакового возраста
        multiSort.sort(Comparator.comparing((Person p) -> p.age)
                                .thenComparing(p -> p.name));
        System.out.println("By age, then name: " + multiSort);

        // Сортировка строк по длине, затем в алфавитном порядке
        List<String> words2 = new ArrayList<>(Arrays.asList("dog", "cat", "bird", "apple"));
        words2.sort(Comparator.comparing((String s) -> s.length())
                             .thenComparing(s -> s));
        System.out.println("By length, then alphabetically: " + words2);

        System.out.println("\\n=== Обработка null ===");

        List<String> withNulls = new ArrayList<>(Arrays.asList("Banana", null, "Apple", "Cherry"));

        // Null сначала
        List<String> nullsFirst = new ArrayList<>(withNulls);
        nullsFirst.sort(Comparator.nullsFirst(Comparator.naturalOrder()));
        System.out.println("Nulls first: " + nullsFirst);

        // Null в конце
        List<String> nullsLast = new ArrayList<>(withNulls);
        nullsLast.sort(Comparator.nullsLast(Comparator.naturalOrder()));
        System.out.println("Nulls last: " + nullsLast);
    }
}`,
            description: `# Компараторы с лямбдами

Comparator - это функциональный интерфейс, идеально подходящий для лямбда-выражений. Java 8 расширила интерфейс Comparator статическими методами и методами по умолчанию, которые значительно упрощают сортировку коллекций. Понимание лямбд компараторов необходимо для работы с коллекциями и потоками.

## Требования:
1. Базовые лямбды компараторов:
   1.1. Сортировка в естественном порядке
   1.2. Сортировка в обратном порядке
   1.3. Пользовательская логика сравнения
   1.4. Сравнение по нескольким полям

2. Статические методы Comparator:
   2.1. \`Comparator.comparing()\`: Извлечь и сравнить ключ
   2.2. \`Comparator.naturalOrder()\`: Естественный порядок
   2.3. \`Comparator.reverseOrder()\`: Обратный естественный порядок
   2.4. \`Comparator.nullsFirst()\`, \`nullsLast()\`: Обработка null

3. Цепочки компараторов:
   3.1. \`thenComparing()\`: Вторичный критерий сортировки
   3.2. Многоуровневая сортировка
   3.3. Сложные сценарии сортировки

4. Практические примеры:
   4.1. Сортировка списка объектов Person по имени, возрасту
   4.2. Сортировка строк по длине, затем в алфавитном порядке
   4.3. Обработка null значений при сортировке

## Пример вывода:
\`\`\`
=== Basic Comparator Lambdas ===
Natural order: [1, 2, 3, 4, 5]
Reverse order: [5, 4, 3, 2, 1]
By length: [cat, dog, bird, elephant]

=== Comparator Static Methods ===
By name: [Alice, Bob, Charlie]
By age: [Bob(25), Alice(30), Charlie(35)]
Reverse by age: [Charlie(35), Alice(30), Bob(25)]

=== Chaining Comparators ===
By age, then name: [Bob(25), Alice(30), Charlie(30)]
By length, then alphabetically: [cat, dog, bird, apple]

=== Null Handling ===
Nulls first: [null, Apple, Banana, Cherry]
Nulls last: [Apple, Banana, Cherry, null]
\`\`\``,
            hint1: `Comparator.comparing() извлекает ключ и сравнивает его. Используйте thenComparing() для добавления вторичных критериев сортировки. Это намного чище, чем писать сложную логику сравнения вручную.`,
            hint2: `Для обработки null оберните ваш компаратор с помощью Comparator.nullsFirst() или nullsLast(). Это предотвращает NullPointerException и дает вам контроль над порядком null.`,
            whyItMatters: `Лямбды компараторов делают код сортировки кратким и читаемым. API Comparator с методами comparing(), thenComparing() и обработкой null устраняет шаблонный код и уменьшает количество ошибок. Это необходимые знания для работы с коллекциями, потоками и любым сценарием, требующим пользовательской логики сортировки. Современный Java-код в значительной степени полагается на эти паттерны.`
        },
        uz: {
            title: `Lambdalar bilan komparatorlar`,
            solutionCode: `import java.util.*;

class Person {
    String name;
    int age;

    Person(String name, int age) {
        this.name = name;
        this.age = age;
    }

    @Override
    public String toString() {
        return name + "(" + age + ")";
    }
}

public class ComparatorLambdas {
    public static void main(String[] args) {
        System.out.println("=== Asosiy komparator lambdalari ===");

        // Tabiiy tartibda saralash
        List<Integer> numbers = new ArrayList<>(Arrays.asList(3, 1, 4, 1, 5, 9, 2));
        numbers.sort((a, b) -> a.compareTo(b));
        // Yoki oddiy: numbers.sort(Integer::compareTo);
        System.out.println("Natural order: " + numbers);

        // Lambda yordamida teskari tartib
        List<Integer> reversed = new ArrayList<>(Arrays.asList(3, 1, 4, 1, 5, 9, 2));
        reversed.sort((a, b) -> b.compareTo(a));
        System.out.println("Reverse order: " + reversed);

        // Maxsus taqqoslash - satrlarni uzunligi bo'yicha saralash
        List<String> words = new ArrayList<>(Arrays.asList("elephant", "cat", "dog", "bird"));
        words.sort((s1, s2) -> Integer.compare(s1.length(), s2.length()));
        System.out.println("By length: " + words);

        System.out.println("\\n=== Comparator statik metodlari ===");

        List<Person> people = Arrays.asList(
            new Person("Charlie", 35),
            new Person("Alice", 30),
            new Person("Bob", 25)
        );

        // Comparator.comparing() yordamida ism bo'yicha saralash
        List<Person> byName = new ArrayList<>(people);
        byName.sort(Comparator.comparing(p -> p.name));
        // Yoki metod havolasi bilan: byName.sort(Comparator.comparing(Person::getName));
        System.out.println("By name: " + byName);

        // Yosh bo'yicha saralash
        List<Person> byAge = new ArrayList<>(people);
        byAge.sort(Comparator.comparing(p -> p.age));
        System.out.println("By age: " + byAge);

        // Yosh bo'yicha teskari tartib
        List<Person> byAgeReverse = new ArrayList<>(people);
        byAgeReverse.sort(Comparator.comparing((Person p) -> p.age).reversed());
        System.out.println("Reverse by age: " + byAgeReverse);

        System.out.println("\\n=== Komparatorlarni zanjir qilish ===");

        List<Person> multiSort = Arrays.asList(
            new Person("Charlie", 30),
            new Person("Alice", 30),
            new Person("Bob", 25)
        );

        // Yosh bo'yicha, keyin bir xil yosh uchun ism bo'yicha saralash
        multiSort.sort(Comparator.comparing((Person p) -> p.age)
                                .thenComparing(p -> p.name));
        System.out.println("By age, then name: " + multiSort);

        // Satrlarni uzunligi bo'yicha, keyin alifbo tartibida saralash
        List<String> words2 = new ArrayList<>(Arrays.asList("dog", "cat", "bird", "apple"));
        words2.sort(Comparator.comparing((String s) -> s.length())
                             .thenComparing(s -> s));
        System.out.println("By length, then alphabetically: " + words2);

        System.out.println("\\n=== Null bilan ishlash ===");

        List<String> withNulls = new ArrayList<>(Arrays.asList("Banana", null, "Apple", "Cherry"));

        // Nulllar birinchi
        List<String> nullsFirst = new ArrayList<>(withNulls);
        nullsFirst.sort(Comparator.nullsFirst(Comparator.naturalOrder()));
        System.out.println("Nulls first: " + nullsFirst);

        // Nulllar oxirida
        List<String> nullsLast = new ArrayList<>(withNulls);
        nullsLast.sort(Comparator.nullsLast(Comparator.naturalOrder()));
        System.out.println("Nulls last: " + nullsLast);
    }
}`,
            description: `# Lambdalar bilan komparatorlar

Comparator lambda ifodalari uchun mukammal funksional interfeys hisoblanadi. Java 8 Comparator interfeysini statik va standart metodlar bilan kengaytirdi, bu kolleksiyalarni saralashni ancha osonlashtiradi. Komparator lambdalarini tushunish kolleksiyalar va oqimlar bilan ishlash uchun zarurdir.

## Talablar:
1. Asosiy komparator lambdalari:
   1.1. Tabiiy tartibda saralash
   1.2. Teskari tartibda saralash
   1.3. Maxsus taqqoslash mantigi
   1.4. Bir nechta maydon bo'yicha taqqoslash

2. Comparator statik metodlari:
   2.1. \`Comparator.comparing()\`: Kalitni ajratib olish va taqqoslash
   2.2. \`Comparator.naturalOrder()\`: Tabiiy tartib
   2.3. \`Comparator.reverseOrder()\`: Teskari tabiiy tartib
   2.4. \`Comparator.nullsFirst()\`, \`nullsLast()\`: Null bilan ishlash

3. Komparatorlarni zanjir qilish:
   3.1. \`thenComparing()\`: Ikkilamchi saralash mezoni
   3.2. Ko'p darajali saralash
   3.3. Murakkab saralash stsenariylari

4. Amaliy misollar:
   4.1. Person ob'ektlari ro'yxatini ism, yosh bo'yicha saralash
   4.2. Satrlarni uzunligi, keyin alifbo tartibida saralash
   4.3. Saralashda null qiymatlarni boshqarish

## Chiqish namunasi:
\`\`\`
=== Basic Comparator Lambdas ===
Natural order: [1, 2, 3, 4, 5]
Reverse order: [5, 4, 3, 2, 1]
By length: [cat, dog, bird, elephant]

=== Comparator Static Methods ===
By name: [Alice, Bob, Charlie]
By age: [Bob(25), Alice(30), Charlie(35)]
Reverse by age: [Charlie(35), Alice(30), Bob(25)]

=== Chaining Comparators ===
By age, then name: [Bob(25), Alice(30), Charlie(30)]
By length, then alphabetically: [cat, dog, bird, apple]

=== Null Handling ===
Nulls first: [null, Apple, Banana, Cherry]
Nulls last: [Apple, Banana, Cherry, null]
\`\`\``,
            hint1: `Comparator.comparing() kalitni ajratib oladi va uni taqqoslaydi. Ikkilamchi saralash mezonlarini qo'shish uchun thenComparing() dan foydalaning. Bu qo'lda murakkab taqqoslash mantiqini yozishdan ancha toza.`,
            hint2: `Null bilan ishlash uchun komparatoringizni Comparator.nullsFirst() yoki nullsLast() bilan o'rang. Bu NullPointerException ning oldini oladi va sizga null tartibini boshqarish imkonini beradi.`,
            whyItMatters: `Komparator lambdalari saralash kodini qisqa va o'qilishi oson qiladi. comparing(), thenComparing() va null bilan ishlash metodlari bilan Comparator API shablon kodini yo'q qiladi va xatolarni kamaytiradi. Bu kolleksiyalar, oqimlar va maxsus saralash mantiqi talab qilinadigan har qanday stsеnariy bilan ishlash uchun zarur bilim. Zamonaviy Java kodi bu naqshlarga katta darajada tayanadi.`
        }
    }
};

export default task;
