import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-arraylist-basics',
    title: 'ArrayList Fundamentals',
    difficulty: 'easy',
    tags: ['java', 'collections', 'list', 'arraylist'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn ArrayList fundamentals in Java.

**Requirements:**
1. Create an ArrayList to store strings
2. Add elements: "Apple", "Banana", "Cherry"
3. Get the element at index 1
4. Remove "Banana" from the list
5. Check if the list contains "Cherry"
6. Print the size and whether the list is empty
7. Add "Date" at index 1
8. Print all elements

ArrayList is the most commonly used List implementation, backed by a dynamic array that grows as needed.`,
    initialCode: `import java.util.ArrayList;

public class ArrayListBasics {
    public static void main(String[] args) {
        // Create an ArrayList to store strings

        // Add elements: "Apple", "Banana", "Cherry"

        // Get the element at index 1

        // Remove "Banana" from the list

        // Check if the list contains "Cherry"

        // Print size and isEmpty

        // Add "Date" at index 1

        // Print all elements
    }
}`,
    solutionCode: `import java.util.ArrayList;

public class ArrayListBasics {
    public static void main(String[] args) {
        // Create an ArrayList to store strings
        ArrayList<String> fruits = new ArrayList<>();

        // Add elements: "Apple", "Banana", "Cherry"
        fruits.add("Apple");
        fruits.add("Banana");
        fruits.add("Cherry");
        System.out.println("Initial list: " + fruits);

        // Get the element at index 1
        String secondFruit = fruits.get(1);
        System.out.println("Element at index 1: " + secondFruit);

        // Remove "Banana" from the list
        fruits.remove("Banana");
        System.out.println("After removing Banana: " + fruits);

        // Check if the list contains "Cherry"
        boolean hasCherry = fruits.contains("Cherry");
        System.out.println("Contains Cherry: " + hasCherry);

        // Print size and isEmpty
        System.out.println("Size: " + fruits.size());
        System.out.println("Is empty: " + fruits.isEmpty());

        // Add "Date" at index 1
        fruits.add(1, "Date");
        System.out.println("After adding Date at index 1: " + fruits);

        // Print all elements
        System.out.println("Final list:");
        for (String fruit : fruits) {
            System.out.println("  - " + fruit);
        }
    }
}`,
    hint1: `Use new ArrayList<>() to create a list. The add() method appends elements, and get(index) retrieves them.`,
    hint2: `remove() can take either an index or an object. contains() returns a boolean, and size() returns the number of elements.`,
    whyItMatters: `ArrayList is the most versatile and commonly used collection in Java. Understanding its basic operations is essential for managing dynamic lists of data efficiently.

**Production Pattern:**
\`\`\`java
// Storing query results with pre-allocated memory
List<User> users = new ArrayList<>(expectedSize);
for (Row row : resultSet) {
    users.add(parseUser(row));
}

// Filtering and processing data
List<Order> activeOrders = new ArrayList<>();
for (Order order : allOrders) {
    if (order.isActive()) {
        activeOrders.add(order);
    }
}
\`\`\`

**Practical Benefits:**
- Dynamic resizing without manual array management
- O(1) index access for fast reading
- Pre-allocating memory reduces the number of copies`,
    order: 0,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

// Test1: main method should show ArrayList demo
class Test1 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ArrayListBasics.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show ArrayList demo",
            output.contains("ArrayList") || output.contains("List") ||
            output.contains("Apple") || output.contains("Initial"));
    }
}

// Test2: Output should show initial list with Apple, Banana, Cherry
class Test2 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ArrayListBasics.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should contain Apple", output.contains("Apple"));
        assertTrue("Should contain Banana initially", output.contains("Banana"));
        assertTrue("Should contain Cherry", output.contains("Cherry"));
    }
}

// Test3: Output should show element at index 1
class Test3 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ArrayListBasics.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should mention index 1", output.contains("index 1") || output.contains("индекс 1") || output.contains("1-indeks"));
    }
}

// Test4: Output should show removing Banana
class Test4 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ArrayListBasics.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should mention removing", output.contains("remov") || output.contains("удален") || output.contains("o'chir"));
    }
}

// Test5: Output should show contains check for Cherry
class Test5 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ArrayListBasics.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should show contains check", output.contains("contains") || output.contains("содержит") || output.contains("bor"));
    }
}

// Test6: Output should show size
class Test6 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ArrayListBasics.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should show size", output.contains("size") || output.contains("размер") || output.contains("o'lcham"));
    }
}

// Test7: Output should show isEmpty check
class Test7 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ArrayListBasics.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should show empty check", output.contains("empty") || output.contains("пуст") || output.contains("bo'sh"));
    }
}

// Test8: Output should show adding Date at index 1
class Test8 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ArrayListBasics.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should contain Date", output.contains("Date"));
    }
}

// Test9: Output should show final list with all elements
class Test9 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ArrayListBasics.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should mention 'final' or 'all elements'",
            output.contains("final") || output.contains("all") ||
            output.contains("итог") || output.contains("yakuniy") ||
            output.contains("финальн"));
    }
}

// Test10: Output should show list iteration (elements printed separately)
class Test10 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ArrayListBasics.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        // After operations: Apple, Date, Cherry (Banana removed, Date inserted)
        assertTrue("Should show Apple in final output", output.contains("Apple"));
        assertTrue("Should show Cherry in final output", output.contains("Cherry"));
    }
}
`,
    translations: {
        ru: {
            title: 'Основы ArrayList',
            solutionCode: `import java.util.ArrayList;

public class ArrayListBasics {
    public static void main(String[] args) {
        // Создаем ArrayList для хранения строк
        ArrayList<String> fruits = new ArrayList<>();

        // Добавляем элементы: "Apple", "Banana", "Cherry"
        fruits.add("Apple");
        fruits.add("Banana");
        fruits.add("Cherry");
        System.out.println("Начальный список: " + fruits);

        // Получаем элемент по индексу 1
        String secondFruit = fruits.get(1);
        System.out.println("Элемент по индексу 1: " + secondFruit);

        // Удаляем "Banana" из списка
        fruits.remove("Banana");
        System.out.println("После удаления Banana: " + fruits);

        // Проверяем, содержит ли список "Cherry"
        boolean hasCherry = fruits.contains("Cherry");
        System.out.println("Содержит Cherry: " + hasCherry);

        // Выводим размер и проверяем, пуст ли список
        System.out.println("Размер: " + fruits.size());
        System.out.println("Пустой: " + fruits.isEmpty());

        // Добавляем "Date" по индексу 1
        fruits.add(1, "Date");
        System.out.println("После добавления Date по индексу 1: " + fruits);

        // Выводим все элементы
        System.out.println("Финальный список:");
        for (String fruit : fruits) {
            System.out.println("  - " + fruit);
        }
    }
}`,
            description: `Изучите основы ArrayList в Java.

**Требования:**
1. Создайте ArrayList для хранения строк
2. Добавьте элементы: "Apple", "Banana", "Cherry"
3. Получите элемент по индексу 1
4. Удалите "Banana" из списка
5. Проверьте, содержит ли список "Cherry"
6. Выведите размер и проверьте, пуст ли список
7. Добавьте "Date" по индексу 1
8. Выведите все элементы

ArrayList - наиболее часто используемая реализация List, основанная на динамическом массиве, который растет по мере необходимости.`,
            hint1: `Используйте new ArrayList<>() для создания списка. Метод add() добавляет элементы, а get(index) извлекает их.`,
            hint2: `remove() может принимать либо индекс, либо объект. contains() возвращает boolean, а size() возвращает количество элементов.`,
            whyItMatters: `ArrayList - самая универсальная и часто используемая коллекция в Java. Понимание основных операций необходимо для эффективного управления динамическими списками данных.

**Продакшен паттерн:**
\`\`\`java
// Хранение результатов запроса с предварительным выделением памяти
List<User> users = new ArrayList<>(expectedSize);
for (Row row : resultSet) {
    users.add(parseUser(row));
}

// Фильтрация и обработка данных
List<Order> activeOrders = new ArrayList<>();
for (Order order : allOrders) {
    if (order.isActive()) {
        activeOrders.add(order);
    }
}
\`\`\`

**Практические преимущества:**
- Динамическое изменение размера без ручного управления массивами
- O(1) доступ по индексу для быстрого чтения
- Предварительное выделение памяти уменьшает количество копирований`
        },
        uz: {
            title: 'ArrayList Asoslari',
            solutionCode: `import java.util.ArrayList;

public class ArrayListBasics {
    public static void main(String[] args) {
        // Satrlarni saqlash uchun ArrayList yaratamiz
        ArrayList<String> fruits = new ArrayList<>();

        // Elementlarni qo'shamiz: "Apple", "Banana", "Cherry"
        fruits.add("Apple");
        fruits.add("Banana");
        fruits.add("Cherry");
        System.out.println("Boshlang'ich ro'yxat: " + fruits);

        // 1-indeksdagi elementni olamiz
        String secondFruit = fruits.get(1);
        System.out.println("1-indeksdagi element: " + secondFruit);

        // Ro'yxatdan "Banana" ni o'chiramiz
        fruits.remove("Banana");
        System.out.println("Banana o'chirilgandan keyin: " + fruits);

        // Ro'yxatda "Cherry" borligini tekshiramiz
        boolean hasCherry = fruits.contains("Cherry");
        System.out.println("Cherry bor: " + hasCherry);

        // O'lchamni va bo'shligini tekshiramiz
        System.out.println("O'lchami: " + fruits.size());
        System.out.println("Bo'shmi: " + fruits.isEmpty());

        // 1-indeksga "Date" qo'shamiz
        fruits.add(1, "Date");
        System.out.println("1-indeksga Date qo'shgandan keyin: " + fruits);

        // Barcha elementlarni chiqaramiz
        System.out.println("Yakuniy ro'yxat:");
        for (String fruit : fruits) {
            System.out.println("  - " + fruit);
        }
    }
}`,
            description: `Java da ArrayList asoslarini o'rganing.

**Talablar:**
1. Satrlarni saqlash uchun ArrayList yarating
2. Elementlarni qo'shing: "Apple", "Banana", "Cherry"
3. 1-indeksdagi elementni oling
4. Ro'yxatdan "Banana" ni o'chiring
5. Ro'yxatda "Cherry" borligini tekshiring
6. O'lchamni va bo'shligini chiqaring
7. 1-indeksga "Date" ni qo'shing
8. Barcha elementlarni chiqaring

ArrayList - eng ko'p ishlatiladigan List implementatsiyasi bo'lib, kerak bo'lganda o'sadigan dinamik massivga asoslangan.`,
            hint1: `Ro'yxat yaratish uchun new ArrayList<>() dan foydalaning. add() metodi elementlarni qo'shadi, get(index) esa oladi.`,
            hint2: `remove() indeks yoki obyektni qabul qilishi mumkin. contains() boolean qaytaradi, size() esa elementlar sonini qaytaradi.`,
            whyItMatters: `ArrayList Java da eng ko'p qo'llaniladigan va universal kolleksiya. Uning asosiy operatsiyalarini tushunish dinamik ma'lumotlar ro'yxatlarini samarali boshqarish uchun zarur.

**Ishlab chiqarish patterni:**
\`\`\`java
// Oldindan xotira ajratish bilan so'rov natijalarini saqlash
List<User> users = new ArrayList<>(expectedSize);
for (Row row : resultSet) {
    users.add(parseUser(row));
}

// Ma'lumotlarni filtrlash va qayta ishlash
List<Order> activeOrders = new ArrayList<>();
for (Order order : allOrders) {
    if (order.isActive()) {
        activeOrders.add(order);
    }
}
\`\`\`

**Amaliy foydalari:**
- Massivlarni qo'lda boshqarishsiz dinamik o'lcham o'zgarishi
- Tez o'qish uchun indeks bo'yicha O(1) kirish
- Oldindan xotira ajratish nusxa ko'chirishlarni kamaytiradi`
        }
    }
};

export default task;
