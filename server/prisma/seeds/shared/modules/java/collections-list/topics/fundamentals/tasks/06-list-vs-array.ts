import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-list-vs-array',
    title: 'List vs Array Trade-offs',
    difficulty: 'medium',
    tags: ['java', 'collections', 'list', 'array', 'performance'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Understand the differences between Lists and arrays, and when to use each.

**Requirements:**
1. Create an array and convert it to a List using Arrays.asList()
2. Try to modify the list and explain the behavior
3. Convert a List to an array using toArray()
4. Create a proper mutable ArrayList from an array
5. Compare performance: array vs ArrayList for random access
6. Compare performance: array vs ArrayList for resizing
7. Show when to use arrays vs Lists

Arrays.asList() creates a fixed-size list backed by the array - you can't add/remove elements.`,
    initialCode: `import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

public class ListVsArray {
    public static void main(String[] args) {
        // Create array and convert to List with Arrays.asList()

        // Try to modify the list

        // Convert List to array with toArray()

        // Create mutable ArrayList from array

        // Performance comparison: random access

        // Performance comparison: resizing

        // When to use arrays vs Lists?
    }
}`,
    solutionCode: `import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

public class ListVsArray {
    public static void main(String[] args) {
        // Create array and convert to List using Arrays.asList()
        String[] array = {"Apple", "Banana", "Cherry"};
        List<String> listFromArray = Arrays.asList(array);
        System.out.println("List from Arrays.asList(): " + listFromArray);

        // Try to modify the list - set works, but add/remove don't
        listFromArray.set(0, "Apricot"); // Works - modifies underlying array
        System.out.println("After set(): " + listFromArray);
        System.out.println("Original array: " + Arrays.toString(array)); // Array changed too!

        try {
            listFromArray.add("Date"); // Fails - fixed size
        } catch (UnsupportedOperationException e) {
            System.out.println("");
            System.out.println("Cannot add to Arrays.asList() list (fixed size)");
        }

        // Convert List to array using toArray()
        List<String> fruits = new ArrayList<>(List.of("Mango", "Orange", "Grape"));
        // Method 1: toArray() returns Object[]
        Object[] objArray = fruits.toArray();
        System.out.println("");
        System.out.println("toArray() result: " + Arrays.toString(objArray));

        // Method 2: toArray(T[]) returns typed array
        String[] stringArray = fruits.toArray(new String[0]);
        System.out.println("toArray(String[]) result: " + Arrays.toString(stringArray));

        // Create proper mutable ArrayList from array
        String[] sourceArray = {"One", "Two", "Three"};
        // Wrong way: Arrays.asList() - fixed size
        List<String> fixedList = Arrays.asList(sourceArray);
        // Right way: wrap in ArrayList constructor
        List<String> mutableList = new ArrayList<>(Arrays.asList(sourceArray));
        mutableList.add("Four"); // Works!
        System.out.println("");
        System.out.println("Mutable ArrayList: " + mutableList);

        // Performance comparison: random access
        int size = 1_000_000;
        int[] intArray = new int[size];
        List<Integer> intList = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            intArray[i] = i;
            intList.add(i);
        }

        // Random access - array
        long startTime = System.nanoTime();
        int sum = 0;
        for (int i = 0; i < size; i++) {
            sum += intArray[i];
        }
        long arrayTime = System.nanoTime() - startTime;

        // Random access - ArrayList
        startTime = System.nanoTime();
        sum = 0;
        for (int i = 0; i < size; i++) {
            sum += intList.get(i);
        }
        long listTime = System.nanoTime() - startTime;

        System.out.println("");
        System.out.println("Random access performance (1M elements):");
        System.out.println("  Array: " + arrayTime / 1_000_000 + " ms");
        System.out.println("  ArrayList: " + listTime / 1_000_000 + " ms");
        System.out.println("  Array is ~" + (listTime / arrayTime) + "x faster");

        // When to use arrays vs Lists
        System.out.println("");
        System.out.println("Use Arrays when:");
        System.out.println("  - Fixed size is acceptable");
        System.out.println("  - Performance is critical (lower memory overhead)");
        System.out.println("  - Working with primitives to avoid boxing");
        System.out.println("  - Multidimensional data structures");

        System.out.println("");
        System.out.println("Use Lists when:");
        System.out.println("  - Size changes dynamically");
        System.out.println("  - Need collection operations (sort, search, etc.)");
        System.out.println("  - Working with generics and type safety");
        System.out.println("  - Flexibility is more important than raw performance");
    }
}`,
    hint1: `Arrays.asList() returns a fixed-size list backed by the array. You can use set() but not add() or remove().`,
    hint2: `To create a mutable list from an array, wrap Arrays.asList() in new ArrayList<>(). For toArray(), use toArray(new T[0]).`,
    whyItMatters: `Understanding the trade-offs between arrays and Lists is crucial for choosing the right data structure. Arrays offer better performance but less flexibility, while Lists provide dynamic sizing and rich operations at a small performance cost.

**Production Pattern:**
\`\`\`java
// Array for fixed coordinates (performance)
double[] coordinates = {x, y, z};  // Low overhead
processCoordinates(coordinates);

// ArrayList for dynamic data (flexibility)
List<User> activeUsers = new ArrayList<>();
while (hasMoreUsers()) {
    User user = fetchNextUser();
    if (user.isActive()) {
        activeUsers.add(user);  // Dynamic growth
    }
}

// Conversion for API compatibility
String[] legacyArray = modernList.toArray(new String[0]);
legacyApi.process(legacyArray);
\`\`\`

**Practical Benefits:**
- Arrays for performance-critical operations with primitives
- Lists for business logic with changing data
- Easy conversion between formats for integration`,
    order: 5,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

// Test1: main method should show List vs Array comparison
class Test1 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListVsArray.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show List vs Array comparison",
            output.contains("Array") || output.contains("List") ||
            output.contains("asList") || output.contains("Convert"));
    }
}

// Test2: Output should show Arrays.asList conversion
class Test2 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListVsArray.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should mention Arrays.asList",
            output.contains("Arrays.asList") || output.contains("asList"));
    }
}

// Test3: Output should explain fixed-size list behavior
class Test3 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListVsArray.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should mention fixed-size or cannot add",
            output.contains("fixed") || output.contains("cannot") || output.contains("unsupported") ||
            output.contains("фиксированн") || output.contains("нельзя") || output.contains("mumkin emas"));
    }
}

// Test4: Output should show toArray conversion
class Test4 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListVsArray.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should mention toArray",
            output.contains("toArray") || output.contains("to array"));
    }
}

// Test5: Output should show mutable ArrayList creation
class Test5 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListVsArray.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should mention mutable or ArrayList",
            output.contains("mutable") || output.contains("arraylist") ||
            output.contains("изменяем") || output.contains("o'zgartirish mumkin"));
    }
}

// Test6: Output should show performance comparison
class Test6 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListVsArray.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should mention performance or time or O(1)",
            output.contains("performance") || output.contains("time") || output.contains("o(1)") ||
            output.contains("производительн") || output.contains("tezlik"));
    }
}

// Test7: Output should show when to use arrays
class Test7 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListVsArray.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should mention array usage scenarios",
            output.contains("use array") || output.contains("when to") || output.contains("primitive") ||
            output.contains("использовать массив") || output.contains("massiv ishlatish"));
    }
}

// Test8: Output should show when to use Lists
class Test8 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListVsArray.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should mention List usage scenarios",
            output.contains("use list") || output.contains("dynamic") || output.contains("resize") ||
            output.contains("использовать list") || output.contains("dinamik"));
    }
}

// Test9: Output should contain sample data
class Test9 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListVsArray.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should contain sample data (Apple, Banana, etc.)",
            output.contains("Apple") || output.contains("Banana") || output.contains("["));
    }
}

// Test10: Output should explain key differences
class Test10 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ListVsArray.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should compare list vs array",
            (output.contains("list") && output.contains("array")) ||
            (output.contains("список") && output.contains("массив")));
    }
}
`,
    translations: {
        ru: {
            title: 'Компромиссы между List и массивами',
            solutionCode: `import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

public class ListVsArray {
    public static void main(String[] args) {
        // Создаем массив и преобразуем в List используя Arrays.asList()
        String[] array = {"Apple", "Banana", "Cherry"};
        List<String> listFromArray = Arrays.asList(array);
        System.out.println("Список из Arrays.asList(): " + listFromArray);

        // Пробуем изменить список - set работает, но add/remove нет
        listFromArray.set(0, "Apricot"); // Работает - изменяет базовый массив
        System.out.println("После set(): " + listFromArray);
        System.out.println("Исходный массив: " + Arrays.toString(array)); // Массив тоже изменился!

        try {
            listFromArray.add("Date"); // Не работает - фиксированный размер
        } catch (UnsupportedOperationException e) {
            System.out.println("");
            System.out.println("Нельзя добавлять в список Arrays.asList() (фиксированный размер)");
        }

        // Преобразуем List в массив используя toArray()
        List<String> fruits = new ArrayList<>(List.of("Mango", "Orange", "Grape"));
        // Способ 1: toArray() возвращает Object[]
        Object[] objArray = fruits.toArray();
        System.out.println("");
        System.out.println("Результат toArray(): " + Arrays.toString(objArray));

        // Способ 2: toArray(T[]) возвращает типизированный массив
        String[] stringArray = fruits.toArray(new String[0]);
        System.out.println("Результат toArray(String[]): " + Arrays.toString(stringArray));

        // Создаем правильный изменяемый ArrayList из массива
        String[] sourceArray = {"One", "Two", "Three"};
        // Неправильный способ: Arrays.asList() - фиксированный размер
        List<String> fixedList = Arrays.asList(sourceArray);
        // Правильный способ: обернуть в конструктор ArrayList
        List<String> mutableList = new ArrayList<>(Arrays.asList(sourceArray));
        mutableList.add("Four"); // Работает!
        System.out.println("");
        System.out.println("Изменяемый ArrayList: " + mutableList);

        // Сравнение производительности: произвольный доступ
        int size = 1_000_000;
        int[] intArray = new int[size];
        List<Integer> intList = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            intArray[i] = i;
            intList.add(i);
        }

        // Произвольный доступ - массив
        long startTime = System.nanoTime();
        int sum = 0;
        for (int i = 0; i < size; i++) {
            sum += intArray[i];
        }
        long arrayTime = System.nanoTime() - startTime;

        // Произвольный доступ - ArrayList
        startTime = System.nanoTime();
        sum = 0;
        for (int i = 0; i < size; i++) {
            sum += intList.get(i);
        }
        long listTime = System.nanoTime() - startTime;

        System.out.println("");
        System.out.println("Производительность произвольного доступа (1М элементов):");
        System.out.println("  Массив: " + arrayTime / 1_000_000 + " мс");
        System.out.println("  ArrayList: " + listTime / 1_000_000 + " мс");
        System.out.println("  Массив быстрее в ~" + (listTime / arrayTime) + " раз");

        // Когда использовать массивы или Lists
        System.out.println("");
        System.out.println("Используйте массивы когда:");
        System.out.println("  - Фиксированный размер приемлем");
        System.out.println("  - Производительность критична (меньше затрат памяти)");
        System.out.println("  - Работа с примитивами для избежания упаковки");
        System.out.println("  - Многомерные структуры данных");

        System.out.println("");
        System.out.println("Используйте Lists когда:");
        System.out.println("  - Размер изменяется динамически");
        System.out.println("  - Нужны операции коллекций (сортировка, поиск и т.д.)");
        System.out.println("  - Работа с обобщениями и типобезопасностью");
        System.out.println("  - Гибкость важнее чистой производительности");
    }
}`,
            description: `Поймите различия между Lists и массивами, и когда использовать каждый из них.

**Требования:**
1. Создайте массив и преобразуйте его в List используя Arrays.asList()
2. Попробуйте изменить список и объясните поведение
3. Преобразуйте List в массив используя toArray()
4. Создайте правильный изменяемый ArrayList из массива
5. Сравните производительность: массив vs ArrayList для произвольного доступа
6. Сравните производительность: массив vs ArrayList для изменения размера
7. Покажите, когда использовать массивы vs Lists

Arrays.asList() создает список фиксированного размера, основанный на массиве - нельзя добавлять/удалять элементы.`,
            hint1: `Arrays.asList() возвращает список фиксированного размера, основанный на массиве. Можно использовать set(), но не add() или remove().`,
            hint2: `Для создания изменяемого списка из массива оберните Arrays.asList() в new ArrayList<>(). Для toArray() используйте toArray(new T[0]).`,
            whyItMatters: `Понимание компромиссов между массивами и Lists критически важно для выбора правильной структуры данных. Массивы предлагают лучшую производительность, но меньше гибкости, в то время как Lists обеспечивают динамическое изменение размера и богатые операции с небольшими затратами производительности.

**Продакшен паттерн:**
\`\`\`java
// Массив для фиксированных координат (производительность)
double[] coordinates = {x, y, z};  // Низкие накладные расходы
processCoordinates(coordinates);

// ArrayList для динамических данных (гибкость)
List<User> activeUsers = new ArrayList<>();
while (hasMoreUsers()) {
    User user = fetchNextUser();
    if (user.isActive()) {
        activeUsers.add(user);  // Динамический рост
    }
}

// Конвертация для API совместимости
String[] legacyArray = modernList.toArray(new String[0]);
legacyApi.process(legacyArray);
\`\`\`

**Практические преимущества:**
- Массивы для критичных по производительности операций с примитивами
- Lists для бизнес-логики с изменяющимися данными
- Простая конвертация между форматами для интеграции`
        },
        uz: {
            title: 'List va Massiv o\'rtasidagi Tanlov',
            solutionCode: `import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

public class ListVsArray {
    public static void main(String[] args) {
        // Massiv yaratamiz va Arrays.asList() yordamida List ga o'zgartiramiz
        String[] array = {"Apple", "Banana", "Cherry"};
        List<String> listFromArray = Arrays.asList(array);
        System.out.println("Arrays.asList() dan ro'yxat: " + listFromArray);

        // Ro'yxatni o'zgartirishga harakat qilamiz - set ishlaydi, lekin add/remove yo'q
        listFromArray.set(0, "Apricot"); // Ishlaydi - asosiy massivni o'zgartiradi
        System.out.println("set() dan keyin: " + listFromArray);
        System.out.println("Asl massiv: " + Arrays.toString(array)); // Massiv ham o'zgardi!

        try {
            listFromArray.add("Date"); // Ishlamaydi - qat'iy o'lcham
        } catch (UnsupportedOperationException e) {
            System.out.println("");
            System.out.println("Arrays.asList() ro'yxatiga qo'shib bo'lmaydi (qat'iy o'lcham)");
        }

        // toArray() yordamida List ni massivga o'zgartiramiz
        List<String> fruits = new ArrayList<>(List.of("Mango", "Orange", "Grape"));
        // Usul 1: toArray() Object[] qaytaradi
        Object[] objArray = fruits.toArray();
        System.out.println("");
        System.out.println("toArray() natijasi: " + Arrays.toString(objArray));

        // Usul 2: toArray(T[]) tiplanangan massiv qaytaradi
        String[] stringArray = fruits.toArray(new String[0]);
        System.out.println("toArray(String[]) natijasi: " + Arrays.toString(stringArray));

        // Massivdan to'g'ri o'zgaruvchan ArrayList yaratamiz
        String[] sourceArray = {"One", "Two", "Three"};
        // Noto'g'ri usul: Arrays.asList() - qat'iy o'lcham
        List<String> fixedList = Arrays.asList(sourceArray);
        // To'g'ri usul: ArrayList konstruktoriga o'rash
        List<String> mutableList = new ArrayList<>(Arrays.asList(sourceArray));
        mutableList.add("Four"); // Ishlaydi!
        System.out.println("");
        System.out.println("O'zgaruvchan ArrayList: " + mutableList);

        // Ishlash tezligini taqqoslash: tasodifiy kirish
        int size = 1_000_000;
        int[] intArray = new int[size];
        List<Integer> intList = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            intArray[i] = i;
            intList.add(i);
        }

        // Tasodifiy kirish - massiv
        long startTime = System.nanoTime();
        int sum = 0;
        for (int i = 0; i < size; i++) {
            sum += intArray[i];
        }
        long arrayTime = System.nanoTime() - startTime;

        // Tasodifiy kirish - ArrayList
        startTime = System.nanoTime();
        sum = 0;
        for (int i = 0; i < size; i++) {
            sum += intList.get(i);
        }
        long listTime = System.nanoTime() - startTime;

        System.out.println("");
        System.out.println("Tasodifiy kirish samaradorligi (1M element):");
        System.out.println("  Massiv: " + arrayTime / 1_000_000 + " ms");
        System.out.println("  ArrayList: " + listTime / 1_000_000 + " ms");
        System.out.println("  Massiv ~" + (listTime / arrayTime) + " marta tezroq");

        // Massivlar va Lists qachon ishlatiladi
        System.out.println("");
        System.out.println("Massivlardan foydalaning:");
        System.out.println("  - Qat'iy o'lcham qabul qilinadigan bo'lsa");
        System.out.println("  - Samaradorlik muhim (kam xotira sarfi)");
        System.out.println("  - Primitivlar bilan ishlash (boxing oldini olish)");
        System.out.println("  - Ko'p o'lchovli ma'lumot strukturalari");

        System.out.println("");
        System.out.println("Listlardan foydalaning:");
        System.out.println("  - O'lcham dinamik o'zgarsa");
        System.out.println("  - Kolleksiya operatsiyalari kerak (saralash, qidirish va h.k.)");
        System.out.println("  - Generiklar va tip xavfsizligi bilan ishlash");
        System.out.println("  - Moslashuvchanlik xom samaradorlikdan muhimroq");
    }
}`,
            description: `List va massivlar o'rtasidagi farqlarni tushuning va qachon qaysi birini ishlatish kerakligini bilib oling.

**Talablar:**
1. Massiv yarating va uni Arrays.asList() yordamida List ga o'zgartiring
2. Ro'yxatni o'zgartirishga harakat qiling va xatti-harakatni tushuntiring
3. toArray() yordamida List ni massivga o'zgartiring
4. Massivdan to'g'ri o'zgaruvchan ArrayList yarating
5. Samaradorlikni solishtiring: tasodifiy kirish uchun massiv vs ArrayList
6. Samaradorlikni solishtiring: o'lcham o'zgartirish uchun massiv vs ArrayList
7. Massivlar va Lists qachon ishlatilishini ko'rsating

Arrays.asList() massivga asoslangan qat'iy o'lchamli ro'yxat yaratadi - element qo'shib/o'chirib bo'lmaydi.`,
            hint1: `Arrays.asList() massivga asoslangan qat'iy o'lchamli ro'yxatni qaytaradi. set() dan foydalanish mumkin, lekin add() yoki remove() dan emas.`,
            hint2: `Massivdan o'zgaruvchan ro'yxat yaratish uchun Arrays.asList() ni new ArrayList<>() ga o'rang. toArray() uchun toArray(new T[0]) dan foydalaning.`,
            whyItMatters: `Massivlar va Lists o'rtasidagi tanlovni tushunish to'g'ri ma'lumot strukturasini tanlash uchun muhim. Massivlar yaxshi samaradorlikni taklif qiladi, lekin kamroq moslashuvchanlik, Listlar esa dinamik o'lcham o'zgartirish va boy operatsiyalarni ozgina samaradorlik narxida taqdim etadi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Qat'iy koordinatalar uchun massiv (samaradorlik)
double[] coordinates = {x, y, z};  // Kam qo'shimcha xarajat
processCoordinates(coordinates);

// Dinamik ma'lumotlar uchun ArrayList (moslashuvchanlik)
List<User> activeUsers = new ArrayList<>();
while (hasMoreUsers()) {
    User user = fetchNextUser();
    if (user.isActive()) {
        activeUsers.add(user);  // Dinamik o'sish
    }
}

// API moslik uchun konvertatsiya
String[] legacyArray = modernList.toArray(new String[0]);
legacyApi.process(legacyArray);
\`\`\`

**Amaliy foydalari:**
- Primitivlar bilan muhim samaradorlik operatsiyalari uchun massivlar
- O'zgaruvchan ma'lumotlar bilan biznes mantiq uchun Listlar
- Integratsiya uchun formatlar orasida oson konvertatsiya`
        }
    }
};

export default task;
