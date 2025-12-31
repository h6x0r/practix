import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-buffered-streams',
    title: 'Buffered Streams for Performance',
    difficulty: 'medium',
    tags: ['java', 'io', 'buffered-streams', 'performance'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn how to use BufferedInputStream, BufferedOutputStream, and BufferedReader for efficient I/O.

**Requirements:**
1. Write multiple lines to a file using BufferedOutputStream
2. Read the file back using BufferedInputStream and display byte count
3. Use BufferedReader with FileReader to read the file line by line
4. Compare performance of buffered vs unbuffered streams
5. Demonstrate the readLine() method of BufferedReader
6. Use try-with-resources for proper resource management

Buffered streams add a buffer to I/O operations, significantly improving performance by reducing the number of actual I/O operations.`,
    initialCode: `import java.io.*;

public class BufferedStreams {
    public static void main(String[] args) {
        String fileName = "buffered.txt";

        // Write multiple lines using BufferedOutputStream

        // Read using BufferedInputStream

        // Read line by line using BufferedReader
    }
}`,
    solutionCode: `import java.io.*;

public class BufferedStreams {
    public static void main(String[] args) {
        String fileName = "buffered.txt";

        // Write multiple lines using BufferedOutputStream
        try (FileOutputStream fos = new FileOutputStream(fileName);
             BufferedOutputStream bos = new BufferedOutputStream(fos)) {

            String[] lines = {
                "First line of text\\n",
                "Second line of text\\n",
                "Third line of text\\n",
                "Buffered streams improve performance\\n"
            };

            // Write each line
            for (String line : lines) {
                bos.write(line.getBytes());
            }

            // Flush to ensure all data is written
            bos.flush();
            System.out.println("Successfully wrote to file using BufferedOutputStream");

        } catch (IOException e) {
            System.err.println("Error writing: " + e.getMessage());
        }

        // Read using BufferedInputStream
        try (FileInputStream fis = new FileInputStream(fileName);
             BufferedInputStream bis = new BufferedInputStream(fis)) {

            System.out.println("\\nReading with BufferedInputStream:");
            int byteData;
            int count = 0;
            while ((byteData = bis.read()) != -1) {
                System.out.print((char) byteData);
                count++;
            }
            System.out.println("\\nTotal bytes read: " + count);

        } catch (IOException e) {
            System.err.println("Error reading: " + e.getMessage());
        }

        // Read line by line using BufferedReader (better for text)
        try (FileReader fr = new FileReader(fileName);
             BufferedReader br = new BufferedReader(fr)) {

            System.out.println("\\nReading line by line with BufferedReader:");
            String line;
            int lineNumber = 1;
            while ((line = br.readLine()) != null) {
                System.out.println("Line " + lineNumber + ": " + line);
                lineNumber++;
            }

        } catch (IOException e) {
            System.err.println("Error reading: " + e.getMessage());
        }

        // Demonstrate performance benefit
        demonstratePerformance();
    }

    // Compare buffered vs unbuffered performance
    private static void demonstratePerformance() {
        String testFile = "performance_test.txt";
        byte[] data = "Test data for performance comparison\\n".getBytes();

        System.out.println("\\n=== Performance Comparison ===");

        // Unbuffered write
        long startTime = System.nanoTime();
        try (FileOutputStream fos = new FileOutputStream(testFile)) {
            for (int i = 0; i < 1000; i++) {
                fos.write(data);
            }
        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
        }
        long unbufferedTime = System.nanoTime() - startTime;

        // Buffered write
        startTime = System.nanoTime();
        try (FileOutputStream fos = new FileOutputStream(testFile);
             BufferedOutputStream bos = new BufferedOutputStream(fos)) {
            for (int i = 0; i < 1000; i++) {
                bos.write(data);
            }
        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
        }
        long bufferedTime = System.nanoTime() - startTime;

        System.out.println("Unbuffered time: " + unbufferedTime / 1_000_000.0 + " ms");
        System.out.println("Buffered time: " + bufferedTime / 1_000_000.0 + " ms");
        System.out.println("Speed improvement: " + (unbufferedTime / (double) bufferedTime) + "x");
    }
}`,
    hint1: `Wrap FileInputStream/FileOutputStream with BufferedInputStream/BufferedOutputStream: new BufferedInputStream(new FileInputStream(file))`,
    hint2: `Use BufferedReader's readLine() method to read text line by line. It returns null when the end of file is reached.`,
    whyItMatters: `Buffered streams dramatically improve I/O performance by reducing system calls. They're essential for efficient file processing, especially with large files. BufferedReader is the standard way to read text files line by line.

**Production Pattern:**

\`\`\`java
// Efficient log file processing
public void processLogFile(String filename) throws IOException {
    try (BufferedReader br = new BufferedReader(
            new FileReader(filename), 32768)) { // 32KB buffer

        String line;
        int errorCount = 0;

        while ((line = br.readLine()) != null) {
            if (line.contains("ERROR")) {
                errorCount++;
                analyzeError(line);
            }
        }

        logger.info("Errors found: {}", errorCount);
    }
}

// Writing large data with buffering
public void exportData(List<Record> records, String filename) {
    try (BufferedWriter bw = new BufferedWriter(
            new FileWriter(filename), 65536)) { // 64KB buffer

        for (Record record : records) {
            bw.write(record.toCSV());
            bw.newLine();
        }

    } catch (IOException e) {
        throw new RuntimeException("Export error", e);
    }
}
\`\`\`

**Practical Benefits:**

1. **Performance**: Up to 50x faster than unbuffered streams for text files
2. **Convenience**: readLine() simplifies line-by-line text processing
3. **Optimization**: Configurable buffer size for different scenarios
4. **Fewer system calls**: Reduces OS load and increases speed`,
    order: 2,
    testCode: `import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;
import java.io.*;
import java.nio.file.Path;

// Test 1: BufferedInputStream wraps InputStream
class Test1 {
    @Test
    void testBufferedInputStreamWraps() throws IOException {
        ByteArrayInputStream bais = new ByteArrayInputStream("Test".getBytes());
        BufferedInputStream bis = new BufferedInputStream(bais);
        assertNotNull(bis);
        bis.close();
    }
}

// Test 2: BufferedInputStream reads correctly
class Test2 {
    @Test
    void testBufferedInputStreamReads() throws IOException {
        ByteArrayInputStream bais = new ByteArrayInputStream("Hello".getBytes());
        BufferedInputStream bis = new BufferedInputStream(bais);
        assertEquals('H', (char) bis.read());
        bis.close();
    }
}

// Test 3: BufferedOutputStream wraps OutputStream
class Test3 {
    @Test
    void testBufferedOutputStreamWraps() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        BufferedOutputStream bos = new BufferedOutputStream(baos);
        assertNotNull(bos);
        bos.close();
    }
}

// Test 4: BufferedOutputStream writes correctly
class Test4 {
    @Test
    void testBufferedOutputStreamWrites() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        BufferedOutputStream bos = new BufferedOutputStream(baos);
        bos.write("Hello".getBytes());
        bos.flush();
        assertEquals("Hello", baos.toString());
        bos.close();
    }
}

// Test 5: BufferedOutputStream requires flush
class Test5 {
    @Test
    void testBufferedOutputStreamRequiresFlush() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        BufferedOutputStream bos = new BufferedOutputStream(baos);
        bos.write("Test".getBytes());
        assertEquals(0, baos.size());
        bos.flush();
        assertEquals(4, baos.size());
        bos.close();
    }
}

// Test 6: BufferedReader readLine works
class Test6 {
    @Test
    void testBufferedReaderReadLine() throws IOException {
        StringReader sr = new StringReader("Line1\\nLine2");
        BufferedReader br = new BufferedReader(sr);
        assertEquals("Line1", br.readLine());
        assertEquals("Line2", br.readLine());
        assertNull(br.readLine());
        br.close();
    }
}

// Test 7: BufferedReader handles empty lines
class Test7 {
    @Test
    void testBufferedReaderHandlesEmptyLines() throws IOException {
        StringReader sr = new StringReader("Line1\\n\\nLine3");
        BufferedReader br = new BufferedReader(sr);
        assertEquals("Line1", br.readLine());
        assertEquals("", br.readLine());
        assertEquals("Line3", br.readLine());
        br.close();
    }
}

// Test 8: BufferedWriter writes lines
class Test8 {
    @Test
    void testBufferedWriterWritesLines() throws IOException {
        StringWriter sw = new StringWriter();
        BufferedWriter bw = new BufferedWriter(sw);
        bw.write("Line1");
        bw.newLine();
        bw.write("Line2");
        bw.flush();
        assertTrue(sw.toString().contains("Line1"));
        assertTrue(sw.toString().contains("Line2"));
        bw.close();
    }
}

// Test 9: Custom buffer size works
class Test9 {
    @Test
    void testCustomBufferSize() throws IOException {
        ByteArrayInputStream bais = new ByteArrayInputStream("Test".getBytes());
        BufferedInputStream bis = new BufferedInputStream(bais, 16384);
        assertNotNull(bis);
        assertEquals('T', (char) bis.read());
        bis.close();
    }
}

// Test 10: Buffered streams with files
class Test10 {
    @TempDir
    Path tempDir;

    @Test
    void testBufferedStreamsWithFiles() throws IOException {
        File file = tempDir.resolve("test.txt").toFile();

        try (BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(file))) {
            bos.write("Buffered Test".getBytes());
        }

        try (BufferedInputStream bis = new BufferedInputStream(new FileInputStream(file))) {
            byte[] bytes = new byte[13];
            bis.read(bytes);
            assertEquals("Buffered Test", new String(bytes));
        }
    }
}`,
    translations: {
        ru: {
            title: 'Буферизованные потоки для производительности',
            solutionCode: `import java.io.*;

public class BufferedStreams {
    public static void main(String[] args) {
        String fileName = "buffered.txt";

        // Запись нескольких строк с использованием BufferedOutputStream
        try (FileOutputStream fos = new FileOutputStream(fileName);
             BufferedOutputStream bos = new BufferedOutputStream(fos)) {

            String[] lines = {
                "Первая строка текста\\n",
                "Вторая строка текста\\n",
                "Третья строка текста\\n",
                "Буферизованные потоки улучшают производительность\\n"
            };

            // Записываем каждую строку
            for (String line : lines) {
                bos.write(line.getBytes());
            }

            // Принудительная запись для обеспечения записи всех данных
            bos.flush();
            System.out.println("Успешно записано в файл с использованием BufferedOutputStream");

        } catch (IOException e) {
            System.err.println("Ошибка записи: " + e.getMessage());
        }

        // Чтение с использованием BufferedInputStream
        try (FileInputStream fis = new FileInputStream(fileName);
             BufferedInputStream bis = new BufferedInputStream(fis)) {

            System.out.println("\\nЧтение с BufferedInputStream:");
            int byteData;
            int count = 0;
            while ((byteData = bis.read()) != -1) {
                System.out.print((char) byteData);
                count++;
            }
            System.out.println("\\nВсего байт прочитано: " + count);

        } catch (IOException e) {
            System.err.println("Ошибка чтения: " + e.getMessage());
        }

        // Чтение построчно с использованием BufferedReader (лучше для текста)
        try (FileReader fr = new FileReader(fileName);
             BufferedReader br = new BufferedReader(fr)) {

            System.out.println("\\nЧтение построчно с BufferedReader:");
            String line;
            int lineNumber = 1;
            while ((line = br.readLine()) != null) {
                System.out.println("Строка " + lineNumber + ": " + line);
                lineNumber++;
            }

        } catch (IOException e) {
            System.err.println("Ошибка чтения: " + e.getMessage());
        }

        // Демонстрация преимущества производительности
        demonstratePerformance();
    }

    // Сравнение буферизованной и небуферизованной производительности
    private static void demonstratePerformance() {
        String testFile = "performance_test.txt";
        byte[] data = "Тестовые данные для сравнения производительности\\n".getBytes();

        System.out.println("\\n=== Сравнение производительности ===");

        // Небуферизованная запись
        long startTime = System.nanoTime();
        try (FileOutputStream fos = new FileOutputStream(testFile)) {
            for (int i = 0; i < 1000; i++) {
                fos.write(data);
            }
        } catch (IOException e) {
            System.err.println("Ошибка: " + e.getMessage());
        }
        long unbufferedTime = System.nanoTime() - startTime;

        // Буферизованная запись
        startTime = System.nanoTime();
        try (FileOutputStream fos = new FileOutputStream(testFile);
             BufferedOutputStream bos = new BufferedOutputStream(fos)) {
            for (int i = 0; i < 1000; i++) {
                bos.write(data);
            }
        } catch (IOException e) {
            System.err.println("Ошибка: " + e.getMessage());
        }
        long bufferedTime = System.nanoTime() - startTime;

        System.out.println("Небуферизованное время: " + unbufferedTime / 1_000_000.0 + " мс");
        System.out.println("Буферизованное время: " + bufferedTime / 1_000_000.0 + " мс");
        System.out.println("Улучшение скорости: " + (unbufferedTime / (double) bufferedTime) + "x");
    }
}`,
            description: `Научитесь использовать BufferedInputStream, BufferedOutputStream и BufferedReader для эффективного ввода-вывода.

**Требования:**
1. Запишите несколько строк в файл, используя BufferedOutputStream
2. Прочитайте файл обратно, используя BufferedInputStream, и отобразите количество байтов
3. Используйте BufferedReader с FileReader для построчного чтения файла
4. Сравните производительность буферизованных и небуферизованных потоков
5. Продемонстрируйте метод readLine() класса BufferedReader
6. Используйте try-with-resources для правильного управления ресурсами

Буферизованные потоки добавляют буфер к операциям ввода-вывода, значительно повышая производительность за счет уменьшения количества фактических операций ввода-вывода.`,
            hint1: `Оберните FileInputStream/FileOutputStream в BufferedInputStream/BufferedOutputStream: new BufferedInputStream(new FileInputStream(file))`,
            hint2: `Используйте метод readLine() класса BufferedReader для построчного чтения текста. Он возвращает null при достижении конца файла.`,
            whyItMatters: `Буферизованные потоки значительно улучшают производительность ввода-вывода за счет уменьшения количества системных вызовов. Они необходимы для эффективной обработки файлов, особенно больших. BufferedReader является стандартным способом построчного чтения текстовых файлов.

**Продакшен паттерн:**

\`\`\`java
// Эффективная обработка лог-файла
public void processLogFile(String filename) throws IOException {
    try (BufferedReader br = new BufferedReader(
            new FileReader(filename), 32768)) { // 32KB буфер

        String line;
        int errorCount = 0;

        while ((line = br.readLine()) != null) {
            if (line.contains("ERROR")) {
                errorCount++;
                analyzeError(line);
            }
        }

        logger.info("Найдено ошибок: {}", errorCount);
    }
}

// Запись больших данных с буферизацией
public void exportData(List<Record> records, String filename) {
    try (BufferedWriter bw = new BufferedWriter(
            new FileWriter(filename), 65536)) { // 64KB буфер

        for (Record record : records) {
            bw.write(record.toCSV());
            bw.newLine();
        }

    } catch (IOException e) {
        throw new RuntimeException("Ошибка экспорта", e);
    }
}
\`\`\`

**Практические преимущества:**

1. **Производительность**: До 50x быстрее небуферизованных потоков для текстовых файлов
2. **Удобство**: readLine() упрощает построчную обработку текста
3. **Оптимизация**: Настраиваемый размер буфера для разных сценариев
4. **Меньше системных вызовов**: Снижение нагрузки на ОС и увеличение скорости`
        },
        uz: {
            title: 'Unumdorlik uchun buferlangan oqimlar',
            solutionCode: `import java.io.*;

public class BufferedStreams {
    public static void main(String[] args) {
        String fileName = "buffered.txt";

        // BufferedOutputStream yordamida bir nechta qatorlarni yozish
        try (FileOutputStream fos = new FileOutputStream(fileName);
             BufferedOutputStream bos = new BufferedOutputStream(fos)) {

            String[] lines = {
                "Matnning birinchi qatori\\n",
                "Matnning ikkinchi qatori\\n",
                "Matnning uchinchi qatori\\n",
                "Buferlangan oqimlar unumdorlikni oshiradi\\n"
            };

            // Har bir qatorni yozamiz
            for (String line : lines) {
                bos.write(line.getBytes());
            }

            // Barcha ma'lumotlar yozilganligini ta'minlash uchun flush
            bos.flush();
            System.out.println("BufferedOutputStream yordamida faylga muvaffaqiyatli yozildi");

        } catch (IOException e) {
            System.err.println("Yozishda xato: " + e.getMessage());
        }

        // BufferedInputStream yordamida o'qish
        try (FileInputStream fis = new FileInputStream(fileName);
             BufferedInputStream bis = new BufferedInputStream(fis)) {

            System.out.println("\\nBufferedInputStream bilan o'qish:");
            int byteData;
            int count = 0;
            while ((byteData = bis.read()) != -1) {
                System.out.print((char) byteData);
                count++;
            }
            System.out.println("\\nJami o'qilgan baytlar: " + count);

        } catch (IOException e) {
            System.err.println("O'qishda xato: " + e.getMessage());
        }

        // BufferedReader yordamida qator-qator o'qish (matn uchun yaxshiroq)
        try (FileReader fr = new FileReader(fileName);
             BufferedReader br = new BufferedReader(fr)) {

            System.out.println("\\nBufferedReader bilan qator-qator o'qish:");
            String line;
            int lineNumber = 1;
            while ((line = br.readLine()) != null) {
                System.out.println("Qator " + lineNumber + ": " + line);
                lineNumber++;
            }

        } catch (IOException e) {
            System.err.println("O'qishda xato: " + e.getMessage());
        }

        // Unumdorlik afzalligini ko'rsatish
        demonstratePerformance();
    }

    // Buferlangan va buferlanmagan unumdorlikni taqqoslash
    private static void demonstratePerformance() {
        String testFile = "performance_test.txt";
        byte[] data = "Unumdorlikni taqqoslash uchun test ma'lumotlari\\n".getBytes();

        System.out.println("\\n=== Unumdorlikni taqqoslash ===");

        // Buferlanmagan yozish
        long startTime = System.nanoTime();
        try (FileOutputStream fos = new FileOutputStream(testFile)) {
            for (int i = 0; i < 1000; i++) {
                fos.write(data);
            }
        } catch (IOException e) {
            System.err.println("Xato: " + e.getMessage());
        }
        long unbufferedTime = System.nanoTime() - startTime;

        // Buferlangan yozish
        startTime = System.nanoTime();
        try (FileOutputStream fos = new FileOutputStream(testFile);
             BufferedOutputStream bos = new BufferedOutputStream(fos)) {
            for (int i = 0; i < 1000; i++) {
                bos.write(data);
            }
        } catch (IOException e) {
            System.err.println("Xato: " + e.getMessage());
        }
        long bufferedTime = System.nanoTime() - startTime;

        System.out.println("Buferlanmagan vaqt: " + unbufferedTime / 1_000_000.0 + " ms");
        System.out.println("Buferlangan vaqt: " + bufferedTime / 1_000_000.0 + " ms");
        System.out.println("Tezlik yaxshilanishi: " + (unbufferedTime / (double) bufferedTime) + "x");
    }
}`,
            description: `Samarali kiritish-chiqarish uchun BufferedInputStream, BufferedOutputStream va BufferedReader dan foydalanishni o'rganing.

**Talablar:**
1. BufferedOutputStream yordamida faylga bir nechta qatorlarni yozing
2. BufferedInputStream yordamida faylni qaytadan o'qing va baytlar sonini ko'rsating
3. Faylni qator-qator o'qish uchun FileReader bilan BufferedReader dan foydalaning
4. Buferlangan va buferlanmagan oqimlarning unumdorligini taqqoslang
5. BufferedReader ning readLine() metodini ko'rsating
6. Resurslarni to'g'ri boshqarish uchun try-with-resources dan foydalaning

Buferlangan oqimlar kiritish-chiqarish operatsiyalariga bufer qo'shib, haqiqiy kiritish-chiqarish operatsiyalari sonini kamaytirish orqali unumdorlikni sezilarli darajada oshiradi.`,
            hint1: `FileInputStream/FileOutputStream ni BufferedInputStream/BufferedOutputStream bilan o'rang: new BufferedInputStream(new FileInputStream(file))`,
            hint2: `Matnni qator-qator o'qish uchun BufferedReader ning readLine() metodidan foydalaning. U fayl oxiriga yetganda null qaytaradi.`,
            whyItMatters: `Buferlangan oqimlar tizim chaqiruvlari sonini kamaytirish orqali kiritish-chiqarish unumdorligini sezilarli darajada yaxshilaydi. Ular samarali fayllarni qayta ishlash, ayniqsa katta fayllar uchun zarurdir. BufferedReader matn fayllarini qator-qator o'qishning standart usuli hisoblanadi.

**Ishlab chiqarish patterni:**

\`\`\`java
// Log faylni samarali qayta ishlash
public void processLogFile(String filename) throws IOException {
    try (BufferedReader br = new BufferedReader(
            new FileReader(filename), 32768)) { // 32KB bufer

        String line;
        int errorCount = 0;

        while ((line = br.readLine()) != null) {
            if (line.contains("ERROR")) {
                errorCount++;
                analyzeError(line);
            }
        }

        logger.info("Topilgan xatolar: {}", errorCount);
    }
}

// Katta ma'lumotlarni bufer bilan yozish
public void exportData(List<Record> records, String filename) {
    try (BufferedWriter bw = new BufferedWriter(
            new FileWriter(filename), 65536)) { // 64KB bufer

        for (Record record : records) {
            bw.write(record.toCSV());
            bw.newLine();
        }

    } catch (IOException e) {
        throw new RuntimeException("Eksport xatosi", e);
    }
}
\`\`\`

**Amaliy foydalari:**

1. **Unumdorlik**: Matn fayllar uchun buferlanmagan oqimlarga qaraganda 50x gacha tezroq
2. **Qulaylik**: readLine() matnni qator-qator qayta ishlashni soddalashtiradi
3. **Optimallashtirish**: Turli stsenariylar uchun sozlanishi mumkin bo'lgan bufer hajmi
4. **Kamroq tizim chaqiruvlari**: OS yukini kamaytirish va tezlikni oshirish`
        }
    }
};

export default task;
