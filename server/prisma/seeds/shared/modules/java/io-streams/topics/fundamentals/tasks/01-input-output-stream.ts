import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-input-output-stream',
    title: 'InputStream and OutputStream Basics',
    difficulty: 'easy',
    tags: ['java', 'io', 'streams', 'byte-streams'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn the fundamentals of byte streams using InputStream and OutputStream.

**Requirements:**
1. Create a ByteArrayInputStream from a byte array containing "Hello Streams"
2. Read bytes one by one using read() method
3. Print each byte as a character
4. Create a ByteArrayOutputStream
5. Write bytes to it using write() method
6. Convert the output stream to a byte array and display as a string

InputStream and OutputStream are abstract base classes for all byte-oriented I/O in Java. They work with raw bytes.`,
    initialCode: `import java.io.*;

public class InputOutputStream {
    public static void main(String[] args) {
        // Create ByteArrayInputStream from "Hello Streams"

        // Read and print bytes one by one

        // Create ByteArrayOutputStream

        // Write bytes to it

        // Convert to byte array and print as string
    }
}`,
    solutionCode: `import java.io.*;

public class InputOutputStream {
    public static void main(String[] args) {
        try {
            // Create ByteArrayInputStream from byte array
            String text = "Hello Streams";
            byte[] inputBytes = text.getBytes();
            InputStream inputStream = new ByteArrayInputStream(inputBytes);

            // Read bytes one by one
            System.out.println("Reading bytes:");
            int byteData;
            while ((byteData = inputStream.read()) != -1) {
                // Convert byte to character and print
                System.out.print((char) byteData);
            }
            System.out.println("\\n");

            // Close input stream
            inputStream.close();

            // Create ByteArrayOutputStream
            OutputStream outputStream = new ByteArrayOutputStream();

            // Write bytes to output stream
            String outputText = "Writing to stream";
            byte[] outputBytes = outputText.getBytes();
            outputStream.write(outputBytes);

            // Convert to byte array and display
            byte[] result = ((ByteArrayOutputStream) outputStream).toByteArray();
            System.out.println("Output stream content: " + new String(result));

            // Close output stream
            outputStream.close();

        } catch (IOException e) {
            System.err.println("I/O Error: " + e.getMessage());
        }
    }
}`,
    hint1: `Use ByteArrayInputStream constructor with byte array. Call getBytes() on a String to convert it to bytes.`,
    hint2: `The read() method returns -1 when the end of stream is reached. Use a while loop to read until -1.`,
    whyItMatters: `InputStream and OutputStream are the foundation of Java's byte-oriented I/O system. Understanding them is essential for working with files, network sockets, and any binary data processing.

**Production Pattern:**

\`\`\`java
// Reading data from network or file
try (InputStream input = new FileInputStream("data.bin")) {
    byte[] buffer = new byte[1024];
    int bytesRead;

    // Read data in chunks for efficiency
    while ((bytesRead = input.read(buffer)) != -1) {
        // Process the read data
        processData(buffer, bytesRead);
    }
} catch (IOException e) {
    logger.error("Read error", e);
}

// Writing data with error control
try (OutputStream output = new FileOutputStream("result.bin")) {
    byte[] data = prepareData();
    output.write(data);
    output.flush(); // Ensure data is written
} catch (IOException e) {
    logger.error("Write error", e);
}
\`\`\`

**Practical Benefits:**

1. **Universality**: Base interface for all byte I/O operations
2. **Efficiency**: Direct work with bytes without conversions
3. **Compatibility**: Works with any data sources (files, network, memory)
4. **Foundation for other streams**: BufferedInputStream, DataInputStream are built on top of them`,
    order: 0,
    testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.io.*;

// Test 1: ByteArrayInputStream reads bytes correctly
class Test1 {
    @Test
    void testByteArrayInputStreamReads() throws IOException {
        String text = "Hello";
        InputStream is = new ByteArrayInputStream(text.getBytes());
        assertEquals('H', (char) is.read());
        is.close();
    }
}

// Test 2: ByteArrayInputStream reads all bytes
class Test2 {
    @Test
    void testByteArrayInputStreamReadsAll() throws IOException {
        String text = "Test";
        InputStream is = new ByteArrayInputStream(text.getBytes());
        StringBuilder sb = new StringBuilder();
        int b;
        while ((b = is.read()) != -1) {
            sb.append((char) b);
        }
        assertEquals("Test", sb.toString());
        is.close();
    }
}

// Test 3: ByteArrayOutputStream writes bytes
class Test3 {
    @Test
    void testByteArrayOutputStreamWrites() throws IOException {
        ByteArrayOutputStream os = new ByteArrayOutputStream();
        os.write("Hello".getBytes());
        assertEquals("Hello", os.toString());
        os.close();
    }
}

// Test 4: ByteArrayOutputStream toByteArray works
class Test4 {
    @Test
    void testByteArrayOutputStreamToByteArray() throws IOException {
        ByteArrayOutputStream os = new ByteArrayOutputStream();
        os.write("Test".getBytes());
        byte[] result = os.toByteArray();
        assertEquals(4, result.length);
        os.close();
    }
}

// Test 5: InputStream returns -1 at end of stream
class Test5 {
    @Test
    void testInputStreamReturnsMinusOne() throws IOException {
        InputStream is = new ByteArrayInputStream(new byte[0]);
        assertEquals(-1, is.read());
        is.close();
    }
}

// Test 6: OutputStream write single byte
class Test6 {
    @Test
    void testOutputStreamWriteSingleByte() throws IOException {
        ByteArrayOutputStream os = new ByteArrayOutputStream();
        os.write('A');
        assertEquals("A", os.toString());
        os.close();
    }
}

// Test 7: Multiple writes to OutputStream
class Test7 {
    @Test
    void testMultipleWritesToOutputStream() throws IOException {
        ByteArrayOutputStream os = new ByteArrayOutputStream();
        os.write("Hello ".getBytes());
        os.write("World".getBytes());
        assertEquals("Hello World", os.toString());
        os.close();
    }
}

// Test 8: InputStream available returns correct count
class Test8 {
    @Test
    void testInputStreamAvailable() throws IOException {
        InputStream is = new ByteArrayInputStream("Test".getBytes());
        assertEquals(4, is.available());
        is.close();
    }
}

// Test 9: Reading bytes preserves data
class Test9 {
    @Test
    void testReadingPreservesData() throws IOException {
        byte[] original = {65, 66, 67, 68};
        InputStream is = new ByteArrayInputStream(original);
        ByteArrayOutputStream os = new ByteArrayOutputStream();
        int b;
        while ((b = is.read()) != -1) {
            os.write(b);
        }
        assertArrayEquals(original, os.toByteArray());
    }
}

// Test 10: OutputStream reset clears data
class Test10 {
    @Test
    void testOutputStreamReset() throws IOException {
        ByteArrayOutputStream os = new ByteArrayOutputStream();
        os.write("Hello".getBytes());
        os.reset();
        os.write("Hi".getBytes());
        assertEquals("Hi", os.toString());
        os.close();
    }
}`,
    translations: {
        ru: {
            title: 'Основы InputStream и OutputStream',
            solutionCode: `import java.io.*;

public class InputOutputStream {
    public static void main(String[] args) {
        try {
            // Создаем ByteArrayInputStream из массива байтов
            String text = "Hello Streams";
            byte[] inputBytes = text.getBytes();
            InputStream inputStream = new ByteArrayInputStream(inputBytes);

            // Читаем байты по одному
            System.out.println("Чтение байтов:");
            int byteData;
            while ((byteData = inputStream.read()) != -1) {
                // Преобразуем байт в символ и выводим
                System.out.print((char) byteData);
            }
            System.out.println("\\n");

            // Закрываем входной поток
            inputStream.close();

            // Создаем ByteArrayOutputStream
            OutputStream outputStream = new ByteArrayOutputStream();

            // Записываем байты в выходной поток
            String outputText = "Writing to stream";
            byte[] outputBytes = outputText.getBytes();
            outputStream.write(outputBytes);

            // Преобразуем в массив байтов и отображаем
            byte[] result = ((ByteArrayOutputStream) outputStream).toByteArray();
            System.out.println("Содержимое выходного потока: " + new String(result));

            // Закрываем выходной поток
            outputStream.close();

        } catch (IOException e) {
            System.err.println("Ошибка ввода-вывода: " + e.getMessage());
        }
    }
}`,
            description: `Изучите основы байтовых потоков с использованием InputStream и OutputStream.

**Требования:**
1. Создайте ByteArrayInputStream из массива байтов, содержащего "Hello Streams"
2. Читайте байты по одному, используя метод read()
3. Выводите каждый байт как символ
4. Создайте ByteArrayOutputStream
5. Записывайте в него байты, используя метод write()
6. Преобразуйте выходной поток в массив байтов и отобразите как строку

InputStream и OutputStream - это абстрактные базовые классы для всех байт-ориентированных операций ввода-вывода в Java. Они работают с необработанными байтами.`,
            hint1: `Используйте конструктор ByteArrayInputStream с массивом байтов. Вызовите getBytes() для String, чтобы преобразовать его в байты.`,
            hint2: `Метод read() возвращает -1 при достижении конца потока. Используйте цикл while для чтения до -1.`,
            whyItMatters: `InputStream и OutputStream являются основой системы байт-ориентированного ввода-вывода в Java. Их понимание необходимо для работы с файлами, сетевыми сокетами и любой обработкой двоичных данных.

**Продакшен паттерн:**

\`\`\`java
// Чтение данных из сети или файла
try (InputStream input = new FileInputStream("data.bin")) {
    byte[] buffer = new byte[1024];
    int bytesRead;

    // Читаем данные блоками для эффективности
    while ((bytesRead = input.read(buffer)) != -1) {
        // Обрабатываем прочитанные данные
        processData(buffer, bytesRead);
    }
} catch (IOException e) {
    logger.error("Ошибка чтения", e);
}

// Запись данных с контролем ошибок
try (OutputStream output = new FileOutputStream("result.bin")) {
    byte[] data = prepareData();
    output.write(data);
    output.flush(); // Обеспечиваем запись данных
} catch (IOException e) {
    logger.error("Ошибка записи", e);
}
\`\`\`

**Практические преимущества:**

1. **Универсальность**: Базовый интерфейс для всех байтовых операций ввода-вывода
2. **Эффективность**: Прямая работа с байтами без преобразований
3. **Совместимость**: Работает с любыми источниками данных (файлы, сеть, память)
4. **Основа для других потоков**: BufferedInputStream, DataInputStream строятся поверх них`
        },
        uz: {
            title: 'InputStream va OutputStream asoslari',
            solutionCode: `import java.io.*;

public class InputOutputStream {
    public static void main(String[] args) {
        try {
            // Bayt massividan ByteArrayInputStream yaratamiz
            String text = "Hello Streams";
            byte[] inputBytes = text.getBytes();
            InputStream inputStream = new ByteArrayInputStream(inputBytes);

            // Baytlarni bittadan o'qiymiz
            System.out.println("Baytlarni o'qish:");
            int byteData;
            while ((byteData = inputStream.read()) != -1) {
                // Baytni belgiga aylantirib chiqaramiz
                System.out.print((char) byteData);
            }
            System.out.println("\\n");

            // Kirish oqimini yopamiz
            inputStream.close();

            // ByteArrayOutputStream yaratamiz
            OutputStream outputStream = new ByteArrayOutputStream();

            // Chiqish oqimiga baytlar yozamiz
            String outputText = "Writing to stream";
            byte[] outputBytes = outputText.getBytes();
            outputStream.write(outputBytes);

            // Bayt massiviga aylantirib ko'rsatamiz
            byte[] result = ((ByteArrayOutputStream) outputStream).toByteArray();
            System.out.println("Chiqish oqimi mazmuni: " + new String(result));

            // Chiqish oqimini yopamiz
            outputStream.close();

        } catch (IOException e) {
            System.err.println("Kirish-chiqish xatosi: " + e.getMessage());
        }
    }
}`,
            description: `InputStream va OutputStream yordamida bayt oqimlarining asoslarini o'rganing.

**Talablar:**
1. "Hello Streams" ni o'z ichiga olgan bayt massividan ByteArrayInputStream yarating
2. read() metodidan foydalanib baytlarni bittadan o'qing
3. Har bir baytni belgi sifatida chiqaring
4. ByteArrayOutputStream yarating
5. write() metodidan foydalanib unga baytlar yozing
6. Chiqish oqimini bayt massiviga aylantiring va satr sifatida ko'rsating

InputStream va OutputStream Java da barcha bayt-yo'naltirilgan kiritish-chiqarish uchun abstrakt asosiy klasslardir. Ular xom baytlar bilan ishlaydi.`,
            hint1: `ByteArrayInputStream konstruktorini bayt massivi bilan ishlating. Stringni baytlarga aylantirish uchun getBytes() ni chaqiring.`,
            hint2: `read() metodi oqim oxiriga yetganda -1 qaytaradi. -1 ga qadar o'qish uchun while siklidan foydalaning.`,
            whyItMatters: `InputStream va OutputStream Java ning bayt-yo'naltirilgan kiritish-chiqarish tizimining asosidir. Ularni tushunish fayllar, tarmoq socketlari va har qanday ikkilik ma'lumotlarni qayta ishlash bilan ishlash uchun zarur.

**Ishlab chiqarish patterni:**

\`\`\`java
// Tarmoq yoki fayldan ma'lumotlarni o'qish
try (InputStream input = new FileInputStream("data.bin")) {
    byte[] buffer = new byte[1024];
    int bytesRead;

    // Samaradorlik uchun ma'lumotlarni bloklarda o'qiymiz
    while ((bytesRead = input.read(buffer)) != -1) {
        // O'qilgan ma'lumotlarni qayta ishlaymiz
        processData(buffer, bytesRead);
    }
} catch (IOException e) {
    logger.error("O'qishda xato", e);
}

// Xatolarni nazorat qilish bilan ma'lumotlarni yozish
try (OutputStream output = new FileOutputStream("result.bin")) {
    byte[] data = prepareData();
    output.write(data);
    output.flush(); // Ma'lumotlarning yozilishini ta'minlaymiz
} catch (IOException e) {
    logger.error("Yozishda xato", e);
}
\`\`\`

**Amaliy foydalari:**

1. **Universallik**: Barcha bayt kiritish-chiqarish operatsiyalari uchun asosiy interfeys
2. **Samaradorlik**: O'zgartirishlarsiz to'g'ridan-to'g'ri baytlar bilan ishlash
3. **Moslashuvchanlik**: Har qanday ma'lumot manbalari bilan ishlaydi (fayllar, tarmoq, xotira)
4. **Boshqa oqimlar uchun asos**: BufferedInputStream, DataInputStream ular ustiga quriladi`
        }
    }
};

export default task;
