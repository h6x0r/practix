import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-stream-basics',
    title: 'Stream Basics',
    difficulty: 'easy',
    tags: ['java', 'stream-api', 'java8', 'functional-programming', 'collections'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Stream Basics

Streams provide a functional approach to processing collections of data. A stream is a sequence of elements that supports aggregate operations. Streams don't store data - they compute elements on-demand. The Stream API follows a pipeline pattern: source -> intermediate operations -> terminal operation.

## Requirements:
1. Create streams from various sources:
   1.1. From collections (List, Set)
   1.2. From arrays using Arrays.stream()
   1.3. Using Stream.of() for individual elements
   1.4. Using Stream.generate() and Stream.iterate()

2. Demonstrate the stream pipeline:
   2.1. Source: where the stream comes from
   2.2. Intermediate operations: transform the stream (lazy)
   2.3. Terminal operation: produces a result (triggers execution)

3. Show that streams are:
   3.1. Consumed only once
   3.2. Lazy evaluated
   3.3. Not data structures

4. Create examples with different data types

## Example Output:
\`\`\`
=== Creating Streams ===
From List: [1, 2, 3, 4, 5]
From Array: [Apple, Banana, Cherry]
Using Stream.of: [10, 20, 30]
Using generate (5 elements): [0.123, 0.456, 0.789, 0.321, 0.654]
Using iterate (5 elements): [0, 2, 4, 6, 8]

=== Stream Pipeline ===
Original list: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Even numbers doubled: [4, 8, 12, 16, 20]

=== Stream Properties ===
Stream elements consumed: [1, 2, 3]
Attempting to reuse stream will cause IllegalStateException
Intermediate operations are lazy - they don't execute until terminal operation
\`\`\``,
    initialCode: `import java.util.*;
import java.util.stream.*;

public class StreamBasics {
    public static void main(String[] args) {
        // TODO: Create streams from different sources

        // TODO: Demonstrate stream pipeline

        // TODO: Show stream properties (consume once, lazy evaluation)
    }
}`,
    solutionCode: `import java.util.*;
import java.util.stream.*;

public class StreamBasics {
    public static void main(String[] args) {
        System.out.println("=== Creating Streams ===");

        // From collection (List)
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        Stream<Integer> streamFromList = numbers.stream();
        System.out.println("From List: " + numbers);

        // From array
        String[] fruits = {"Apple", "Banana", "Cherry"};
        Stream<String> streamFromArray = Arrays.stream(fruits);
        System.out.println("From Array: " + Arrays.toString(fruits));

        // Using Stream.of()
        Stream<Integer> streamOf = Stream.of(10, 20, 30);
        System.out.println("Using Stream.of: [10, 20, 30]");

        // Using Stream.generate() - infinite stream
        Stream<Double> randomStream = Stream.generate(Math::random).limit(5);
        System.out.print("Using generate (5 elements): [");
        randomStream.forEach(n -> System.out.print(String.format("%.3f", n) + " "));
        System.out.println("]");

        // Using Stream.iterate() - infinite stream
        Stream<Integer> evenNumbers = Stream.iterate(0, n -> n + 2).limit(5);
        System.out.print("Using iterate (5 elements): [");
        evenNumbers.forEach(n -> System.out.print(n + " "));
        System.out.println("]");

        System.out.println("\\n=== Stream Pipeline ===");

        // Stream pipeline: source -> intermediate -> terminal
        List<Integer> data = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        System.out.println("Original list: " + data);

        // Pipeline: filter even numbers, multiply by 2, collect to list
        List<Integer> result = data.stream()	// Source
            .filter(n -> n % 2 == 0)	// Intermediate
            .map(n -> n * 2)	// Intermediate
            .collect(Collectors.toList());	// Terminal

        System.out.println("Even numbers doubled: " + result);

        System.out.println("\\n=== Stream Properties ===");

        // Streams can be consumed only once
        Stream<Integer> stream1 = Stream.of(1, 2, 3);
        System.out.print("Stream elements consumed: [");
        stream1.forEach(n -> System.out.print(n + " "));
        System.out.println("]");

        System.out.println("Attempting to reuse stream will cause IllegalStateException");

        // Lazy evaluation - intermediate operations don't execute until terminal operation
        System.out.println("Intermediate operations are lazy - they don't execute until terminal operation");

        Stream<Integer> lazyStream = data.stream()
            .filter(n -> {
                System.out.println("  Filtering: " + n);
                return n % 2 == 0;
            })
            .map(n -> {
                System.out.println("  Mapping: " + n);
                return n * 2;
            });

        System.out.println("Stream created but not executed yet...");
        System.out.println("Executing terminal operation:");
        List<Integer> lazyResult = lazyStream.collect(Collectors.toList());
        System.out.println("Result: " + lazyResult);
    }
}`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

// Test1: main method should show Stream basics header
class Test1 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        StreamBasics.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show Stream basics header",
            output.contains("Stream") || output.contains("Creating") ||
            output.contains("Создание") || output.contains("Yaratish"));
    }
}

// Test2: should show stream from List
class Test2 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        StreamBasics.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show stream from List", output.contains("List") || output.contains("ro'yxat") || output.contains("[1, 2, 3"));
    }
}

// Test3: should show stream from Array
class Test3 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        StreamBasics.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show stream from Array", output.contains("Array") || output.contains("massiv") || output.contains("Apple"));
    }
}

// Test4: should show Stream.of() usage
class Test4 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        StreamBasics.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show Stream.of()", output.contains("Stream.of") || output.contains("10") || output.contains("20"));
    }
}

// Test5: should demonstrate generate or iterate
class Test5 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        StreamBasics.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show generate or iterate", output.contains("generate") || output.contains("iterate") || output.contains("elements"));
    }
}

// Test6: should show stream pipeline
class Test6 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        StreamBasics.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show stream pipeline", output.contains("Pipeline") || output.contains("konveyer") || output.contains("Конвейер") || output.contains("doubled"));
    }
}

// Test7: should demonstrate filter and map
class Test7 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        StreamBasics.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show filter/map result", output.contains("Even") || output.contains("juft") || output.contains("[4") || output.contains("8"));
    }
}

// Test8: should show stream properties
class Test8 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        StreamBasics.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show stream properties", output.contains("Properties") || output.contains("xususiyat") || output.contains("Свойств") || output.contains("consumed"));
    }
}

// Test9: should demonstrate lazy evaluation
class Test9 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        StreamBasics.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show lazy evaluation", output.contains("lazy") || output.contains("Filtering") || output.contains("Mapping") || output.contains("Lazy"));
    }
}

// Test10: should have section headers
class Test10 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        StreamBasics.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        boolean hasHeaders = output.contains("===") ||
                             output.contains("Creating") || output.contains("Создание") || output.contains("yaratish");
        assertTrue("Should have section headers", hasHeaders);
    }
}
`,
    hint1: `Use Collection.stream() to create a stream from a collection. Use Arrays.stream() for arrays. Use Stream.of() for individual elements.`,
    hint2: `A stream pipeline has three parts: source (where data comes from), intermediate operations (transformations), and terminal operation (produces result).`,
    whyItMatters: `Streams are fundamental to modern Java programming. They provide a declarative way to process data, making code more readable and maintainable. Understanding stream basics is essential for working with collections efficiently and writing functional-style Java code.

**Production Pattern:**
\`\`\`java
// Processing large collections with filtering and mapping
List<Order> orders = orderRepository.findAll();
List<OrderDTO> activeOrders = orders.stream()
    .filter(order -> order.getStatus() == OrderStatus.ACTIVE)
    .map(OrderMapper::toDTO)
    .collect(Collectors.toList());
\`\`\`

**Practical Benefits:**
- Improved code readability when working with collections
- Easy switching to parallel processing (parallelStream)
- Optimization through lazy evaluation`,
    order: 1,
    translations: {
        ru: {
            title: 'Основы потоков',
            solutionCode: `import java.util.*;
import java.util.stream.*;

public class StreamBasics {
    public static void main(String[] args) {
        System.out.println("=== Создание потоков ===");

        // Из коллекции (List)
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        Stream<Integer> streamFromList = numbers.stream();
        System.out.println("From List: " + numbers);

        // Из массива
        String[] fruits = {"Apple", "Banana", "Cherry"};
        Stream<String> streamFromArray = Arrays.stream(fruits);
        System.out.println("From Array: " + Arrays.toString(fruits));

        // Используя Stream.of()
        Stream<Integer> streamOf = Stream.of(10, 20, 30);
        System.out.println("Using Stream.of: [10, 20, 30]");

        // Используя Stream.generate() - бесконечный поток
        Stream<Double> randomStream = Stream.generate(Math::random).limit(5);
        System.out.print("Using generate (5 elements): [");
        randomStream.forEach(n -> System.out.print(String.format("%.3f", n) + " "));
        System.out.println("]");

        // Используя Stream.iterate() - бесконечный поток
        Stream<Integer> evenNumbers = Stream.iterate(0, n -> n + 2).limit(5);
        System.out.print("Using iterate (5 elements): [");
        evenNumbers.forEach(n -> System.out.print(n + " "));
        System.out.println("]");

        System.out.println("\\n=== Конвейер потока ===");

        // Конвейер потока: источник -> промежуточные -> терминальная
        List<Integer> data = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        System.out.println("Original list: " + data);

        // Конвейер: фильтр четных чисел, умножение на 2, сбор в список
        List<Integer> result = data.stream()	// Источник
            .filter(n -> n % 2 == 0)	// Промежуточная
            .map(n -> n * 2)	// Промежуточная
            .collect(Collectors.toList());	// Терминальная

        System.out.println("Even numbers doubled: " + result);

        System.out.println("\\n=== Свойства потока ===");

        // Потоки можно использовать только один раз
        Stream<Integer> stream1 = Stream.of(1, 2, 3);
        System.out.print("Stream elements consumed: [");
        stream1.forEach(n -> System.out.print(n + " "));
        System.out.println("]");

        System.out.println("Attempting to reuse stream will cause IllegalStateException");

        // Ленивая оценка - промежуточные операции не выполняются до терминальной операции
        System.out.println("Intermediate operations are lazy - they don't execute until terminal operation");

        Stream<Integer> lazyStream = data.stream()
            .filter(n -> {
                System.out.println("  Filtering: " + n);
                return n % 2 == 0;
            })
            .map(n -> {
                System.out.println("  Mapping: " + n);
                return n * 2;
            });

        System.out.println("Stream created but not executed yet...");
        System.out.println("Executing terminal operation:");
        List<Integer> lazyResult = lazyStream.collect(Collectors.toList());
        System.out.println("Result: " + lazyResult);
    }
}`,
            description: `# Основы потоков

Потоки предоставляют функциональный подход к обработке коллекций данных. Поток - это последовательность элементов, которая поддерживает агрегатные операции. Потоки не хранят данные - они вычисляют элементы по требованию. Stream API следует паттерну конвейера: источник -> промежуточные операции -> терминальная операция.

## Требования:
1. Создайте потоки из различных источников:
   1.1. Из коллекций (List, Set)
   1.2. Из массивов используя Arrays.stream()
   1.3. Используя Stream.of() для отдельных элементов
   1.4. Используя Stream.generate() и Stream.iterate()

2. Продемонстрируйте конвейер потока:
   2.1. Источник: откуда приходит поток
   2.2. Промежуточные операции: преобразуют поток (ленивые)
   2.3. Терминальная операция: производит результат (запускает выполнение)

3. Покажите, что потоки:
   3.1. Используются только один раз
   3.2. Ленивые в оценке
   3.3. Не являются структурами данных

4. Создайте примеры с различными типами данных

## Пример вывода:
\`\`\`
=== Creating Streams ===
From List: [1, 2, 3, 4, 5]
From Array: [Apple, Banana, Cherry]
Using Stream.of: [10, 20, 30]
Using generate (5 elements): [0.123, 0.456, 0.789, 0.321, 0.654]
Using iterate (5 elements): [0, 2, 4, 6, 8]

=== Stream Pipeline ===
Original list: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Even numbers doubled: [4, 8, 12, 16, 20]

=== Stream Properties ===
Stream elements consumed: [1, 2, 3]
Attempting to reuse stream will cause IllegalStateException
Intermediate operations are lazy - they don't execute until terminal operation
\`\`\``,
            hint1: `Используйте Collection.stream() для создания потока из коллекции. Используйте Arrays.stream() для массивов. Используйте Stream.of() для отдельных элементов.`,
            hint2: `Конвейер потока имеет три части: источник (откуда приходят данные), промежуточные операции (преобразования) и терминальная операция (производит результат).`,
            whyItMatters: `Потоки являются основой современного программирования на Java. Они предоставляют декларативный способ обработки данных, делая код более читаемым и поддерживаемым. Понимание основ потоков необходимо для эффективной работы с коллекциями и написания Java-кода в функциональном стиле.

**Продакшен паттерн:**
\`\`\`java
// Обработка больших коллекций с фильтрацией и маппингом
List<Order> orders = orderRepository.findAll();
List<OrderDTO> activeOrders = orders.stream()
    .filter(order -> order.getStatus() == OrderStatus.ACTIVE)
    .map(OrderMapper::toDTO)
    .collect(Collectors.toList());
\`\`\`

**Практические преимущества:**
- Улучшенная читаемость кода при работе с коллекциями
- Легкость переключения на параллельную обработку (parallelStream)
- Оптимизация за счёт ленивых вычислений`
        },
        uz: {
            title: 'Stream asoslari',
            solutionCode: `import java.util.*;
import java.util.stream.*;

public class StreamBasics {
    public static void main(String[] args) {
        System.out.println("=== Streamlarni yaratish ===");

        // Kolleksiyadan (List)
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        Stream<Integer> streamFromList = numbers.stream();
        System.out.println("From List: " + numbers);

        // Massivdan
        String[] fruits = {"Apple", "Banana", "Cherry"};
        Stream<String> streamFromArray = Arrays.stream(fruits);
        System.out.println("From Array: " + Arrays.toString(fruits));

        // Stream.of() yordamida
        Stream<Integer> streamOf = Stream.of(10, 20, 30);
        System.out.println("Using Stream.of: [10, 20, 30]");

        // Stream.generate() yordamida - cheksiz stream
        Stream<Double> randomStream = Stream.generate(Math::random).limit(5);
        System.out.print("Using generate (5 elements): [");
        randomStream.forEach(n -> System.out.print(String.format("%.3f", n) + " "));
        System.out.println("]");

        // Stream.iterate() yordamida - cheksiz stream
        Stream<Integer> evenNumbers = Stream.iterate(0, n -> n + 2).limit(5);
        System.out.print("Using iterate (5 elements): [");
        evenNumbers.forEach(n -> System.out.print(n + " "));
        System.out.println("]");

        System.out.println("\\n=== Stream konveyeri ===");

        // Stream konveyeri: manba -> oraliq -> yakuniy
        List<Integer> data = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        System.out.println("Original list: " + data);

        // Konveyer: juft sonlarni filtrlash, 2 ga ko'paytirish, ro'yxatga yig'ish
        List<Integer> result = data.stream()	// Manba
            .filter(n -> n % 2 == 0)	// Oraliq
            .map(n -> n * 2)	// Oraliq
            .collect(Collectors.toList());	// Yakuniy

        System.out.println("Even numbers doubled: " + result);

        System.out.println("\\n=== Stream xususiyatlari ===");

        // Streamlar faqat bir marta ishlatiladi
        Stream<Integer> stream1 = Stream.of(1, 2, 3);
        System.out.print("Stream elements consumed: [");
        stream1.forEach(n -> System.out.print(n + " "));
        System.out.println("]");

        System.out.println("Attempting to reuse stream will cause IllegalStateException");

        // Lazy baholash - oraliq operatsiyalar yakuniy operatsiyagacha bajarilmaydi
        System.out.println("Intermediate operations are lazy - they don't execute until terminal operation");

        Stream<Integer> lazyStream = data.stream()
            .filter(n -> {
                System.out.println("  Filtering: " + n);
                return n % 2 == 0;
            })
            .map(n -> {
                System.out.println("  Mapping: " + n);
                return n * 2;
            });

        System.out.println("Stream created but not executed yet...");
        System.out.println("Executing terminal operation:");
        List<Integer> lazyResult = lazyStream.collect(Collectors.toList());
        System.out.println("Result: " + lazyResult);
    }
}`,
            description: `# Stream asoslari

Streamlar ma'lumotlar kolleksiyalarini qayta ishlashning funksional yondashuvini taqdim etadi. Stream - bu agregat operatsiyalarni qo'llab-quvvatlaydigan elementlar ketma-ketligi. Streamlar ma'lumotlarni saqlamaydi - ular elementlarni talab bo'yicha hisoblaydi. Stream API konveyer naqshiga amal qiladi: manba -> oraliq operatsiyalar -> yakuniy operatsiya.

## Talablar:
1. Turli manbalardan streamlar yarating:
   1.1. Kolleksiyalardan (List, Set)
   1.2. Massivlardan Arrays.stream() yordamida
   1.3. Alohida elementlar uchun Stream.of() yordamida
   1.4. Stream.generate() va Stream.iterate() yordamida

2. Stream konveyerini namoyish eting:
   2.1. Manba: stream qayerdan keladi
   2.2. Oraliq operatsiyalar: streamni o'zgartiradi (lazy)
   2.3. Yakuniy operatsiya: natija ishlab chiqaradi (bajarilishni boshlaydi)

3. Streamlarni ko'rsating:
   3.1. Faqat bir marta ishlatiladi
   3.2. Lazy baholanadi
   3.3. Ma'lumotlar strukturalari emas

4. Turli ma'lumot turlari bilan misollar yarating

## Chiqish namunasi:
\`\`\`
=== Creating Streams ===
From List: [1, 2, 3, 4, 5]
From Array: [Apple, Banana, Cherry]
Using Stream.of: [10, 20, 30]
Using generate (5 elements): [0.123, 0.456, 0.789, 0.321, 0.654]
Using iterate (5 elements): [0, 2, 4, 6, 8]

=== Stream Pipeline ===
Original list: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Even numbers doubled: [4, 8, 12, 16, 20]

=== Stream Properties ===
Stream elements consumed: [1, 2, 3]
Attempting to reuse stream will cause IllegalStateException
Intermediate operations are lazy - they don't execute until terminal operation
\`\`\``,
            hint1: `Kolleksiyadan stream yaratish uchun Collection.stream() ishlatiladi. Massivlar uchun Arrays.stream() ishlatiladi. Alohida elementlar uchun Stream.of() ishlatiladi.`,
            hint2: `Stream konveyeri uchta qismga ega: manba (ma'lumot qayerdan keladi), oraliq operatsiyalar (o'zgartirishlar) va yakuniy operatsiya (natija ishlab chiqaradi).`,
            whyItMatters: `Streamlar zamonaviy Java dasturlashning asosi hisoblanadi. Ular ma'lumotlarni qayta ishlashning deklarativ usulini taqdim etadi, bu kodni yanada o'qilishi va qo'llab-quvvatlashni osonlashtiradi. Stream asoslarini tushunish kolleksiyalar bilan samarali ishlash va funksional uslubdagi Java kodi yozish uchun zarur.

**Ishlab chiqarish patterni:**
\`\`\`java
// Katta kolleksiyalarni filtrlash va mapping bilan qayta ishlash
List<Order> orders = orderRepository.findAll();
List<OrderDTO> activeOrders = orders.stream()
    .filter(order -> order.getStatus() == OrderStatus.ACTIVE)
    .map(OrderMapper::toDTO)
    .collect(Collectors.toList());
\`\`\`

**Amaliy foydalari:**
- Kolleksiyalar bilan ishlashda kodning o'qilishini yaxshilash
- Parallel qayta ishlashga (parallelStream) oson o'tish
- Lazy baholash orqali optimallashtirish`
        }
    }
};

export default task;
