import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-exception-hierarchy',
    title: 'Exception Hierarchy and Types',
    difficulty: 'easy',
    tags: ['java', 'exceptions', 'checked', 'unchecked', 'hierarchy'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Create an ExceptionDemo class that demonstrates different exception types and proper handling patterns.

Requirements:
1. Implement handleCheckedException() that throws and catches IOException
2. Implement handleUncheckedException() that catches NullPointerException
3. Implement demonstrateExceptionTypes() that shows multiple exception handling
4. Use specific catch blocks before general ones

Example:
\`\`\`java
ExceptionDemo demo = new ExceptionDemo();
demo.demonstrateExceptionTypes();
// Handles multiple exception types appropriately
\`\`\``,
    initialCode: `import java.io.*;

public class ExceptionDemo {

    public void handleCheckedException(String filename) {
        // TODO: Read file and handle IOException
    }

    public void handleUncheckedException(String text) {
        // TODO: Access string length and handle NullPointerException
    }

    public void demonstrateExceptionTypes() {
        // TODO: Demonstrate multiple exception types
    }

    public static void main(String[] args) {
        ExceptionDemo demo = new ExceptionDemo();
        demo.demonstrateExceptionTypes();
    }
}`,
    solutionCode: `import java.io.*;

public class ExceptionDemo {

    public void handleCheckedException(String filename) {
        // Checked exception: Must be declared or caught
        try {
            FileReader reader = new FileReader(filename);
            System.out.println("File opened successfully");
            reader.close();
        } catch (FileNotFoundException e) {
            // Specific checked exception
            System.out.println("Checked: FileNotFoundException - " + e.getMessage());
        } catch (IOException e) {
            // General checked exception
            System.out.println("Checked: IOException - " + e.getMessage());
        }
    }

    public void handleUncheckedException(String text) {
        try {
            // Unchecked exception: Not required to be caught
            int length = text.length();
            System.out.println("String length: " + length);
        } catch (NullPointerException e) {
            // Runtime exception (unchecked)
            System.out.println("Unchecked: NullPointerException - null string provided");
        }
    }

    public void demonstrateExceptionTypes() {
        System.out.println("=== Demonstrating Exception Hierarchy ===");

        // 1. Checked exception example
        System.out.println("");
        System.out.println("1. Checked Exception (IOException):");
        handleCheckedException("nonexistent.txt");

        // 2. Unchecked exception example
        System.out.println("");
        System.out.println("2. Unchecked Exception (NullPointerException):");
        handleUncheckedException(null);

        // 3. Multiple exception types in one try
        System.out.println("");
        System.out.println("3. Multiple Exception Types:");
        try {
            String[] array = {"Hello", null, "World"};
            for (String str : array) {
                System.out.println("Processing: " + str.toUpperCase());
            }
        } catch (NullPointerException e) {
            // Most specific exception first
            System.out.println("Caught NullPointerException");
        } catch (Exception e) {
            // More general exception last
            System.out.println("Caught general Exception");
        }

        System.out.println("");
        System.out.println("=== Demo Complete ===");
    }

    public static void main(String[] args) {
        ExceptionDemo demo = new ExceptionDemo();
        demo.demonstrateExceptionTypes();
    }
}`,
    hint1: `Checked exceptions (like IOException) must be declared with 'throws' or caught. Unchecked exceptions (like NullPointerException) extend RuntimeException and don't require declaration.`,
    hint2: `Always catch more specific exceptions before more general ones. For example, FileNotFoundException should be caught before IOException.`,
    whyItMatters: `Understanding Java's exception hierarchy helps you write more precise error handling code. Knowing when to use checked vs unchecked exceptions is crucial for API design and error recovery strategies.

**Production Pattern:**
\`\`\`java
public void processRequest(Request req) {
    try {
        validateRequest(req);
        processData(req.getData());
    } catch (ValidationException e) {
        logger.warn("Invalid request: {}", e.getMessage());
        throw new BadRequestException(e);
    } catch (IOException e) {
        logger.error("I/O error during processing", e);
        throw new ServiceUnavailableException(e);
    } catch (RuntimeException e) {
        logger.error("Unexpected error", e);
        throw new InternalServerException(e);
    }
}
\`\`\`

**Practical Benefits:**
- Specific handling of different error types
- Proper exception conversion for APIs
- Detailed logging for debugging`,
    order: 2,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

// Test1: main produces output with header
class Test1 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ExceptionDemo.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show exception hierarchy header",
            output.contains("Demonstrating Exception Hierarchy") ||
            output.contains("Демонстрация Иерархии Исключений") ||
            output.contains("Istisnolar Ierarxiyasini"));
    }
}

// Test2: shows checked exception handling with FileNotFoundException
class Test2 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ExceptionDemo demo = new ExceptionDemo();
        demo.handleCheckedException("nonexistent.txt");
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should catch FileNotFoundException",
            output.contains("FileNotFoundException") ||
            output.contains("Checked:") ||
            output.contains("Проверяемое:") ||
            output.contains("Tekshiriladigan:"));
    }
}

// Test3: shows unchecked exception handling with NullPointerException
class Test3 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ExceptionDemo demo = new ExceptionDemo();
        demo.handleUncheckedException(null);
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should catch NullPointerException",
            output.contains("NullPointerException") ||
            output.contains("null string") ||
            output.contains("null строка") ||
            output.contains("null satr"));
    }
}

// Test4: valid string shows length
class Test4 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ExceptionDemo demo = new ExceptionDemo();
        demo.handleUncheckedException("test");
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show string length",
            output.contains("length") || output.contains("4") ||
            output.contains("Длина") || output.contains("uzunligi"));
    }
}

// Test5: demonstrates multiple exception types section
class Test5 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ExceptionDemo.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show multiple exception types section",
            output.contains("Multiple Exception Types") ||
            output.contains("Множественные Типы") ||
            output.contains("Ko'p Istisno Turlari"));
    }
}

// Test6: catches NullPointerException in array processing
class Test6 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ExceptionDemo.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should catch NullPointerException in array",
            output.contains("Caught NullPointerException") ||
            output.contains("Перехвачено NullPointerException") ||
            output.contains("NullPointerException ushlandi"));
    }
}

// Test7: shows demo complete message
class Test7 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ExceptionDemo.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show demo complete message",
            output.contains("Demo Complete") ||
            output.contains("Демонстрация Завершена") ||
            output.contains("Namoyish Tugadi"));
    }
}

// Test8: has numbered sections (1., 2., 3.)
class Test8 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ExceptionDemo.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should have numbered sections",
            output.contains("1.") && output.contains("2.") && output.contains("3."));
    }
}

// Test9: mentions IOException in checked exception section
class Test9 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ExceptionDemo.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should mention IOException",
            output.contains("IOException") || output.contains("Checked Exception"));
    }
}

// Test10: processes first valid string before hitting null
class Test10 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        ExceptionDemo.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should process HELLO before catching null",
            output.contains("HELLO") || output.contains("ПРИВЕТ") || output.contains("SALOM") ||
            output.contains("Processing:") || output.contains("Обработка:") || output.contains("Qayta ishlash:"));
    }
}
`,
    translations: {
        ru: {
            title: 'Иерархия и Типы Исключений',
            solutionCode: `import java.io.*;

public class ExceptionDemo {

    public void handleCheckedException(String filename) {
        // Проверяемое исключение: Должно быть объявлено или перехвачено
        try {
            FileReader reader = new FileReader(filename);
            System.out.println("Файл успешно открыт");
            reader.close();
        } catch (FileNotFoundException e) {
            // Конкретное проверяемое исключение
            System.out.println("Проверяемое: FileNotFoundException - " + e.getMessage());
        } catch (IOException e) {
            // Общее проверяемое исключение
            System.out.println("Проверяемое: IOException - " + e.getMessage());
        }
    }

    public void handleUncheckedException(String text) {
        try {
            // Непроверяемое исключение: Не требуется перехватывать
            int length = text.length();
            System.out.println("Длина строки: " + length);
        } catch (NullPointerException e) {
            // Исключение времени выполнения (непроверяемое)
            System.out.println("Непроверяемое: NullPointerException - передана null строка");
        }
    }

    public void demonstrateExceptionTypes() {
        System.out.println("=== Демонстрация Иерархии Исключений ===");

        // 1. Пример проверяемого исключения
        System.out.println("");
        System.out.println("1. Проверяемое Исключение (IOException):");
        handleCheckedException("несуществующий.txt");

        // 2. Пример непроверяемого исключения
        System.out.println("");
        System.out.println("2. Непроверяемое Исключение (NullPointerException):");
        handleUncheckedException(null);

        // 3. Множественные типы исключений в одном try
        System.out.println("");
        System.out.println("3. Множественные Типы Исключений:");
        try {
            String[] array = {"Привет", null, "Мир"};
            for (String str : array) {
                System.out.println("Обработка: " + str.toUpperCase());
            }
        } catch (NullPointerException e) {
            // Сначала наиболее конкретное исключение
            System.out.println("Перехвачено NullPointerException");
        } catch (Exception e) {
            // Более общее исключение последним
            System.out.println("Перехвачено общее Exception");
        }

        System.out.println("");
        System.out.println("=== Демонстрация Завершена ===");
    }

    public static void main(String[] args) {
        ExceptionDemo demo = new ExceptionDemo();
        demo.demonstrateExceptionTypes();
    }
}`,
            description: `Создайте класс ExceptionDemo, который демонстрирует различные типы исключений и правильные паттерны их обработки.

Требования:
1. Реализуйте handleCheckedException(), который выбрасывает и перехватывает IOException
2. Реализуйте handleUncheckedException(), который перехватывает NullPointerException
3. Реализуйте demonstrateExceptionTypes(), который показывает обработку множественных исключений
4. Используйте конкретные блоки catch перед общими

Пример:
\`\`\`java
ExceptionDemo demo = new ExceptionDemo();
demo.demonstrateExceptionTypes();
// Обрабатывает множественные типы исключений соответствующим образом
\`\`\``,
            hint1: `Проверяемые исключения (как IOException) должны быть объявлены с 'throws' или перехвачены. Непроверяемые исключения (как NullPointerException) наследуются от RuntimeException и не требуют объявления.`,
            hint2: `Всегда перехватывайте более конкретные исключения перед более общими. Например, FileNotFoundException должен быть перехвачен перед IOException.`,
            whyItMatters: `Понимание иерархии исключений Java помогает писать более точный код обработки ошибок. Знание, когда использовать проверяемые и непроверяемые исключения, критически важно для проектирования API и стратегий восстановления после ошибок.

**Продакшен паттерн:**
\`\`\`java
public void processRequest(Request req) {
    try {
        validateRequest(req);
        processData(req.getData());
    } catch (ValidationException e) {
        logger.warn("Invalid request: {}", e.getMessage());
        throw new BadRequestException(e);
    } catch (IOException e) {
        logger.error("I/O error during processing", e);
        throw new ServiceUnavailableException(e);
    } catch (RuntimeException e) {
        logger.error("Unexpected error", e);
        throw new InternalServerException(e);
    }
}
\`\`\`

**Практические преимущества:**
- Специфичная обработка разных типов ошибок
- Правильное преобразование исключений для API
- Детальное логирование для отладки`
        },
        uz: {
            title: `Istisnolar Ierarxiyasi va Turlari`,
            solutionCode: `import java.io.*;

public class ExceptionDemo {

    public void handleCheckedException(String filename) {
        // Tekshiriladigan istisno: E'lon qilinishi yoki ushlanishi kerak
        try {
            FileReader reader = new FileReader(filename);
            System.out.println("Fayl muvaffaqiyatli ochildi");
            reader.close();
        } catch (FileNotFoundException e) {
            // Aniq tekshiriladigan istisno
            System.out.println("Tekshiriladigan: FileNotFoundException - " + e.getMessage());
        } catch (IOException e) {
            // Umumiy tekshiriladigan istisno
            System.out.println("Tekshiriladigan: IOException - " + e.getMessage());
        }
    }

    public void handleUncheckedException(String text) {
        try {
            // Tekshirilmaydigan istisno: Ushlash talab qilinmaydi
            int length = text.length();
            System.out.println("Satr uzunligi: " + length);
        } catch (NullPointerException e) {
            // Ish vaqti istisnosi (tekshirilmaydigan)
            System.out.println("Tekshirilmaydigan: NullPointerException - null satr berilgan");
        }
    }

    public void demonstrateExceptionTypes() {
        System.out.println("=== Istisnolar Ierarxiyasini Namoyish Qilish ===");

        // 1. Tekshiriladigan istisno misoli
        System.out.println("");
        System.out.println("1. Tekshiriladigan Istisno (IOException):");
        handleCheckedException("mavjud_emas.txt");

        // 2. Tekshirilmaydigan istisno misoli
        System.out.println("");
        System.out.println("2. Tekshirilmaydigan Istisno (NullPointerException):");
        handleUncheckedException(null);

        // 3. Bitta try da ko'p istisno turlari
        System.out.println("");
        System.out.println("3. Ko'p Istisno Turlari:");
        try {
            String[] array = {"Salom", null, "Dunyo"};
            for (String str : array) {
                System.out.println("Qayta ishlash: " + str.toUpperCase());
            }
        } catch (NullPointerException e) {
            // Avval eng aniq istisno
            System.out.println("NullPointerException ushlandi");
        } catch (Exception e) {
            // Oxirida umumiyroq istisno
            System.out.println("Umumiy Exception ushlandi");
        }

        System.out.println("");
        System.out.println("=== Namoyish Tugadi ===");
    }

    public static void main(String[] args) {
        ExceptionDemo demo = new ExceptionDemo();
        demo.demonstrateExceptionTypes();
    }
}`,
            description: `ExceptionDemo klassini yarating, u turli istisno turlarini va to'g'ri qayta ishlash naqshlarini ko'rsatadi.

Talablar:
1. handleCheckedException() metodini yarating, u IOException ni tashlaydi va ushlaydi
2. handleUncheckedException() metodini yarating, u NullPointerException ni ushlaydi
3. demonstrateExceptionTypes() metodini yarating, u ko'p istisnolarni qayta ishlashni ko'rsatadi
4. Umumiy bloklar oldidan aniq catch bloklaridan foydalaning

Misol:
\`\`\`java
ExceptionDemo demo = new ExceptionDemo();
demo.demonstrateExceptionTypes();
// Ko'p istisno turlarini tegishlicha qayta ishlaydi
\`\`\``,
            hint1: `Tekshiriladigan istisnolar (IOException kabi) 'throws' bilan e'lon qilinishi yoki ushlanishi kerak. Tekshirilmaydigan istisnolar (NullPointerException kabi) RuntimeException dan meros oladi va e'lon qilish talab qilinmaydi.`,
            hint2: `Har doim umumiyroq istisnolardan oldin aniqroq istisnolarni ushlang. Masalan, FileNotFoundException IOException dan oldin ushlanishi kerak.`,
            whyItMatters: `Java istisnolar ierarxiyasini tushunish aniqroq xatolarni qayta ishlash kodini yozishga yordam beradi. Tekshiriladigan va tekshirilmaydigan istisnolarni qachon ishlatishni bilish API dizayni va xatolardan tiklash strategiyalari uchun juda muhimdir.

**Ishlab chiqarish patterni:**
\`\`\`java
public void processRequest(Request req) {
    try {
        validateRequest(req);
        processData(req.getData());
    } catch (ValidationException e) {
        logger.warn("Noto'g'ri so'rov: {}", e.getMessage());
        throw new BadRequestException(e);
    } catch (IOException e) {
        logger.error("Qayta ishlashda I/O xatosi", e);
        throw new ServiceUnavailableException(e);
    } catch (RuntimeException e) {
        logger.error("Kutilmagan xato", e);
        throw new InternalServerException(e);
    }
}
\`\`\`

**Amaliy foydalari:**
- Turli xil xato turlarini aniq qayta ishlash
- API uchun istisnolarni to'g'ri o'zgartirish
- Disk raskadrovka qilish uchun batafsil jurnallashtirish`
        }
    }
};

export default task;
