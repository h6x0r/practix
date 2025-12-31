import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-list-sorting',
    title: 'Sorting Lists with Comparator',
    difficulty: 'medium',
    tags: ['java', 'collections', 'list', 'sorting', 'comparator'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn to sort lists using Comparator and lambda expressions.

**Requirements:**
1. Create a Person class with name, age, and city fields
2. Create a list of Person objects
3. Sort by age using Collections.sort() with a Comparator
4. Sort by name using List.sort() with Comparator.comparing()
5. Sort by city, then by name using thenComparing()
6. Sort in reverse order by age
7. Sort handling null values safely

Modern Java provides elegant sorting with Comparator.comparing() and method references.`,
    initialCode: `import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

class Person {
    // Define fields and constructor
}

public class ListSorting {
    public static void main(String[] args) {
        // Create a list of Person objects

        // Sort by age using Collections.sort()

        // Sort by name using List.sort()

        // Sort by city, then by name

        // Sort in reverse order by age

        // Handle null values
    }
}`,
    solutionCode: `import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

class Person {
    String name;
    int age;
    String city;

    Person(String name, int age, String city) {
        this.name = name;
        this.age = age;
        this.city = city;
    }

    @Override
    public String toString() {
        return name + " (" + age + ", " + city + ")";
    }
}

public class ListSorting {
    public static void main(String[] args) {
        // Create a list of Person objects
        List<Person> people = new ArrayList<>();
        people.add(new Person("Alice", 30, "New York"));
        people.add(new Person("Bob", 25, "Chicago"));
        people.add(new Person("Charlie", 35, "New York"));
        people.add(new Person("Diana", 25, "Boston"));

        System.out.println("Original list:");
        people.forEach(System.out::println);

        // Sort by age using Collections.sort() with Comparator
        Collections.sort(people, new Comparator<Person>() {
            @Override
            public int compare(Person p1, Person p2) {
                return Integer.compare(p1.age, p2.age);
            }
        });
        System.out.println("\nSorted by age (Collections.sort):");
        people.forEach(System.out::println);

        // Sort by name using List.sort() with Comparator.comparing()
        people.sort(Comparator.comparing(p -> p.name));
        System.out.println("\nSorted by name (List.sort with lambda):");
        people.forEach(System.out::println);

        // Sort by city, then by name using thenComparing()
        people.sort(Comparator.comparing((Person p) -> p.city)
                              .thenComparing(p -> p.name));
        System.out.println("\nSorted by city, then name:");
        people.forEach(System.out::println);

        // Sort in reverse order by age
        people.sort(Comparator.comparing((Person p) -> p.age).reversed());
        System.out.println("\nSorted by age (descending):");
        people.forEach(System.out::println);

        // Handle null values safely
        List<Person> peopleWithNull = new ArrayList<>(people);
        peopleWithNull.add(new Person(null, 28, "Seattle"));
        peopleWithNull.add(new Person("Eve", 28, null));

        // Sort with null-safe comparator (nulls last)
        peopleWithNull.sort(Comparator.comparing((Person p) -> p.name,
                                                 Comparator.nullsLast(String::compareTo))
                                      .thenComparing(p -> p.city,
                                                    Comparator.nullsLast(String::compareTo)));
        System.out.println("\nWith null handling (nulls last):");
        peopleWithNull.forEach(System.out::println);
    }
}`,
    hint1: `Comparator.comparing() takes a function that extracts the field to compare. Use thenComparing() to add secondary sorting.`,
    hint2: `Use reversed() to reverse the sort order, and Comparator.nullsLast() or nullsFirst() to handle null values.`,
    whyItMatters: `Sorting is one of the most common operations on collections. Modern Java's Comparator API with lambda expressions makes sorting elegant and readable, essential for real-world applications.

**Production Pattern:**
\`\`\`java
// Sorting search results by relevance, then by date
List<SearchResult> results = performSearch(query);
results.sort(Comparator.comparing(SearchResult::getRelevance).reversed()
                       .thenComparing(SearchResult::getDate).reversed());

// Handling null values in real data
List<Product> products = loadProducts();
products.sort(Comparator.comparing(Product::getPrice,
                                   Comparator.nullsLast(Double::compareTo))
                        .thenComparing(Product::getName));
\`\`\`

**Practical Benefits:**
- Multi-level sorting for complex business requirements
- Safe null handling without NullPointerException
- Readable code without anonymous classes thanks to lambdas`,
    order: 3,
    testCode: `import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class Person {
    String name;
    int age;
    String city;
    Person(String name, int age, String city) {
        this.name = name;
        this.age = age;
        this.city = city;
    }
}

// Test1: Collections.sort() with Comparator sorts by age
class Test1 {
    @Test
    void testSortByAge() {
        List<Person> people = new ArrayList<>();
        people.add(new Person("Alice", 30, "NYC"));
        people.add(new Person("Bob", 25, "LA"));
        Collections.sort(people, (p1, p2) -> Integer.compare(p1.age, p2.age));
        assertEquals(25, people.get(0).age);
        assertEquals(30, people.get(1).age);
    }
}

// Test2: List.sort() with Comparator.comparing sorts by name
class Test2 {
    @Test
    void testSortByName() {
        List<Person> people = new ArrayList<>();
        people.add(new Person("Charlie", 30, "NYC"));
        people.add(new Person("Alice", 25, "LA"));
        people.sort(Comparator.comparing(p -> p.name));
        assertEquals("Alice", people.get(0).name);
        assertEquals("Charlie", people.get(1).name);
    }
}

// Test3: thenComparing() for multi-level sorting
class Test3 {
    @Test
    void testMultiLevelSort() {
        List<Person> people = new ArrayList<>();
        people.add(new Person("Bob", 30, "NYC"));
        people.add(new Person("Alice", 30, "NYC"));
        people.sort(Comparator.comparing((Person p) -> p.city).thenComparing(p -> p.name));
        assertEquals("Alice", people.get(0).name);
        assertEquals("Bob", people.get(1).name);
    }
}

// Test4: reversed() sorts in descending order
class Test4 {
    @Test
    void testReversedSort() {
        List<Person> people = new ArrayList<>();
        people.add(new Person("Alice", 25, "NYC"));
        people.add(new Person("Bob", 35, "LA"));
        people.sort(Comparator.comparing((Person p) -> p.age).reversed());
        assertEquals(35, people.get(0).age);
        assertEquals(25, people.get(1).age);
    }
}

// Test5: Comparator.nullsLast handles null values
class Test5 {
    @Test
    void testNullsLast() {
        List<String> names = new ArrayList<>();
        names.add(null);
        names.add("Alice");
        names.add("Bob");
        names.sort(Comparator.nullsLast(String::compareTo));
        assertEquals("Alice", names.get(0));
        assertNull(names.get(2));
    }
}

// Test6: Comparator.nullsFirst puts nulls at beginning
class Test6 {
    @Test
    void testNullsFirst() {
        List<String> names = new ArrayList<>();
        names.add("Alice");
        names.add(null);
        names.add("Bob");
        names.sort(Comparator.nullsFirst(String::compareTo));
        assertNull(names.get(0));
        assertEquals("Alice", names.get(1));
    }
}

// Test7: Integer.compare for numeric comparison
class Test7 {
    @Test
    void testIntegerCompare() {
        List<Integer> nums = new ArrayList<>(List.of(3, 1, 4, 1, 5));
        Collections.sort(nums, Integer::compare);
        assertEquals(1, nums.get(0));
        assertEquals(5, nums.get(4));
    }
}

// Test8: Empty list sort doesn't throw
class Test8 {
    @Test
    void testEmptyListSort() {
        List<Person> people = new ArrayList<>();
        assertDoesNotThrow(() -> people.sort(Comparator.comparing(p -> p.name)));
        assertTrue(people.isEmpty());
    }
}

// Test9: Single element list stays unchanged
class Test9 {
    @Test
    void testSingleElementSort() {
        List<Person> people = new ArrayList<>();
        people.add(new Person("Alice", 30, "NYC"));
        people.sort(Comparator.comparing(p -> p.name));
        assertEquals("Alice", people.get(0).name);
    }
}

// Test10: Complex sort with multiple criteria
class Test10 {
    @Test
    void testComplexSort() {
        List<Person> people = new ArrayList<>();
        people.add(new Person("Alice", 30, "NYC"));
        people.add(new Person("Bob", 25, "LA"));
        people.add(new Person("Charlie", 30, "LA"));
        people.sort(Comparator.comparing((Person p) -> p.age)
                              .thenComparing(p -> p.city)
                              .thenComparing(p -> p.name));
        assertEquals("Bob", people.get(0).name);
        assertEquals(25, people.get(0).age);
    }
}
`,
    translations: {
        ru: {
            title: 'Сортировка списков с Comparator',
            solutionCode: `import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

class Person {
    String name;
    int age;
    String city;

    Person(String name, int age, String city) {
        this.name = name;
        this.age = age;
        this.city = city;
    }

    @Override
    public String toString() {
        return name + " (" + age + ", " + city + ")";
    }
}

public class ListSorting {
    public static void main(String[] args) {
        // Создаем список объектов Person
        List<Person> people = new ArrayList<>();
        people.add(new Person("Alice", 30, "New York"));
        people.add(new Person("Bob", 25, "Chicago"));
        people.add(new Person("Charlie", 35, "New York"));
        people.add(new Person("Diana", 25, "Boston"));

        System.out.println("Исходный список:");
        people.forEach(System.out::println);

        // Сортируем по возрасту используя Collections.sort() с Comparator
        Collections.sort(people, new Comparator<Person>() {
            @Override
            public int compare(Person p1, Person p2) {
                return Integer.compare(p1.age, p2.age);
            }
        });
        System.out.println("\nОтсортировано по возрасту (Collections.sort):");
        people.forEach(System.out::println);

        // Сортируем по имени используя List.sort() с Comparator.comparing()
        people.sort(Comparator.comparing(p -> p.name));
        System.out.println("\nОтсортировано по имени (List.sort с лямбдой):");
        people.forEach(System.out::println);

        // Сортируем по городу, затем по имени используя thenComparing()
        people.sort(Comparator.comparing((Person p) -> p.city)
                              .thenComparing(p -> p.name));
        System.out.println("\nОтсортировано по городу, затем по имени:");
        people.forEach(System.out::println);

        // Сортируем в обратном порядке по возрасту
        people.sort(Comparator.comparing((Person p) -> p.age).reversed());
        System.out.println("\nОтсортировано по возрасту (по убыванию):");
        people.forEach(System.out::println);

        // Безопасно обрабатываем null значения
        List<Person> peopleWithNull = new ArrayList<>(people);
        peopleWithNull.add(new Person(null, 28, "Seattle"));
        peopleWithNull.add(new Person("Eve", 28, null));

        // Сортируем с null-безопасным компаратором (null в конце)
        peopleWithNull.sort(Comparator.comparing((Person p) -> p.name,
                                                 Comparator.nullsLast(String::compareTo))
                                      .thenComparing(p -> p.city,
                                                    Comparator.nullsLast(String::compareTo)));
        System.out.println("\nС обработкой null (null в конце):");
        peopleWithNull.forEach(System.out::println);
    }
}`,
            description: `Научитесь сортировать списки используя Comparator и лямбда-выражения.

**Требования:**
1. Создайте класс Person с полями name, age и city
2. Создайте список объектов Person
3. Отсортируйте по возрасту используя Collections.sort() с Comparator
4. Отсортируйте по имени используя List.sort() с Comparator.comparing()
5. Отсортируйте по городу, затем по имени используя thenComparing()
6. Отсортируйте в обратном порядке по возрасту
7. Безопасно обработайте null значения

Современная Java предоставляет элегантную сортировку с Comparator.comparing() и ссылками на методы.`,
            hint1: `Comparator.comparing() принимает функцию, извлекающую поле для сравнения. Используйте thenComparing() для добавления вторичной сортировки.`,
            hint2: `Используйте reversed() для обратной сортировки, и Comparator.nullsLast() или nullsFirst() для обработки null значений.`,
            whyItMatters: `Сортировка - одна из самых распространенных операций над коллекциями. Современный API Comparator в Java с лямбда-выражениями делает сортировку элегантной и читаемой, что необходимо для реальных приложений.

**Продакшен паттерн:**
\`\`\`java
// Сортировка результатов поиска по релевантности, затем по дате
List<SearchResult> results = performSearch(query);
results.sort(Comparator.comparing(SearchResult::getRelevance).reversed()
                       .thenComparing(SearchResult::getDate).reversed());

// Обработка null-значений в реальных данных
List<Product> products = loadProducts();
products.sort(Comparator.comparing(Product::getPrice,
                                   Comparator.nullsLast(Double::compareTo))
                        .thenComparing(Product::getName));
\`\`\`

**Практические преимущества:**
- Многоуровневая сортировка для сложных бизнес-требований
- Безопасная обработка null без NullPointerException
- Читаемый код без анонимных классов благодаря лямбдам`
        },
        uz: {
            title: 'Comparator bilan Ro\'yxatlarni Saralash',
            solutionCode: `import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

class Person {
    String name;
    int age;
    String city;

    Person(String name, int age, String city) {
        this.name = name;
        this.age = age;
        this.city = city;
    }

    @Override
    public String toString() {
        return name + " (" + age + ", " + city + ")";
    }
}

public class ListSorting {
    public static void main(String[] args) {
        // Person obyektlari ro'yxatini yaratamiz
        List<Person> people = new ArrayList<>();
        people.add(new Person("Alice", 30, "New York"));
        people.add(new Person("Bob", 25, "Chicago"));
        people.add(new Person("Charlie", 35, "New York"));
        people.add(new Person("Diana", 25, "Boston"));

        System.out.println("Dastlabki ro'yxat:");
        people.forEach(System.out::println);

        // Yosh bo'yicha Collections.sort() va Comparator bilan saralaymiz
        Collections.sort(people, new Comparator<Person>() {
            @Override
            public int compare(Person p1, Person p2) {
                return Integer.compare(p1.age, p2.age);
            }
        });
        System.out.println("\nYosh bo'yicha saralangan (Collections.sort):");
        people.forEach(System.out::println);

        // Ism bo'yicha List.sort() va Comparator.comparing() bilan saralaymiz
        people.sort(Comparator.comparing(p -> p.name));
        System.out.println("\nIsm bo'yicha saralangan (List.sort lambda bilan):");
        people.forEach(System.out::println);

        // Shahar, keyin ism bo'yicha thenComparing() bilan saralaymiz
        people.sort(Comparator.comparing((Person p) -> p.city)
                              .thenComparing(p -> p.name));
        System.out.println("\nShahar, keyin ism bo'yicha saralangan:");
        people.forEach(System.out::println);

        // Yosh bo'yicha teskari tartibda saralaymiz
        people.sort(Comparator.comparing((Person p) -> p.age).reversed());
        System.out.println("\nYosh bo'yicha saralangan (kamayish tartibida):");
        people.forEach(System.out::println);

        // null qiymatlarni xavfsiz qayta ishlaymiz
        List<Person> peopleWithNull = new ArrayList<>(people);
        peopleWithNull.add(new Person(null, 28, "Seattle"));
        peopleWithNull.add(new Person("Eve", 28, null));

        // null-xavfsiz komparator bilan saralaymiz (null oxirida)
        peopleWithNull.sort(Comparator.comparing((Person p) -> p.name,
                                                 Comparator.nullsLast(String::compareTo))
                                      .thenComparing(p -> p.city,
                                                    Comparator.nullsLast(String::compareTo)));
        System.out.println("\nnull bilan ishlash (null oxirida):");
        peopleWithNull.forEach(System.out::println);
    }
}`,
            description: `Comparator va lambda ifodalar yordamida ro'yxatlarni saralashni o'rganing.

**Talablar:**
1. name, age va city maydonlari bilan Person klassi yarating
2. Person obyektlari ro'yxatini yarating
3. Collections.sort() va Comparator yordamida yosh bo'yicha saralang
4. List.sort() va Comparator.comparing() yordamida ism bo'yicha saralang
5. thenComparing() yordamida shahar, keyin ism bo'yicha saralang
6. Yosh bo'yicha teskari tartibda saralang
7. null qiymatlarni xavfsiz qayta ishlang

Zamonaviy Java Comparator.comparing() va metod havolalari bilan nafis saralashni taqdim etadi.`,
            hint1: `Comparator.comparing() solishtiriluvchi maydonni ajratib oladigan funksiyani qabul qiladi. Ikkinchi darajali saralash qo'shish uchun thenComparing() dan foydalaning.`,
            hint2: `Saralash tartibini teskari aylantirish uchun reversed() dan, null qiymatlarni qayta ishlash uchun Comparator.nullsLast() yoki nullsFirst() dan foydalaning.`,
            whyItMatters: `Saralash kolleksiyalar ustidagi eng keng tarqalgan operatsiyalardan biridir. Zamonaviy Java ning lambda ifodalar bilan Comparator API si saralashni nafis va o'qilishi oson qiladi, bu real dasturlar uchun zarur.

**Ishlab chiqarish patterni:**
\`\`\`java
// Qidiruv natijalarini relevantlik, keyin sana bo'yicha saralash
List<SearchResult> results = performSearch(query);
results.sort(Comparator.comparing(SearchResult::getRelevance).reversed()
                       .thenComparing(SearchResult::getDate).reversed());

// Haqiqiy ma'lumotlarda null qiymatlarni qayta ishlash
List<Product> products = loadProducts();
products.sort(Comparator.comparing(Product::getPrice,
                                   Comparator.nullsLast(Double::compareTo))
                        .thenComparing(Product::getName));
\`\`\`

**Amaliy foydalari:**
- Murakkab biznes talablari uchun ko'p darajali saralash
- NullPointerException siz xavfsiz null qayta ishlash
- Lambda tufayli anonim klassiz o'qilishi oson kod`
        }
    }
};

export default task;
