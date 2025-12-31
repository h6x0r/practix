import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-file-streams',
    title: 'File Streams - FileInputStream and FileOutputStream',
    difficulty: 'easy',
    tags: ['java', 'io', 'file-streams', 'files'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn how to read from and write to files using FileInputStream and FileOutputStream.

**Requirements:**
1. Write text "Java File I/O Example" to a file named "output.txt" using FileOutputStream
2. Close the output stream properly
3. Read the file back using FileInputStream
4. Read bytes and convert to characters
5. Print the content read from the file
6. Use try-with-resources to ensure proper resource cleanup

FileInputStream and FileOutputStream are used for reading and writing raw bytes to files.`,
    initialCode: `import java.io.*;

public class FileStreams {
    public static void main(String[] args) {
        String fileName = "output.txt";
        String content = "Java File I/O Example";

        // Write to file using FileOutputStream

        // Read from file using FileInputStream

        // Print the content
    }
}`,
    solutionCode: `import java.io.*;

public class FileStreams {
    public static void main(String[] args) {
        String fileName = "output.txt";
        String content = "Java File I/O Example";

        // Write to file using FileOutputStream with try-with-resources
        try (FileOutputStream fos = new FileOutputStream(fileName)) {
            // Convert string to bytes and write
            byte[] bytes = content.getBytes();
            fos.write(bytes);
            System.out.println("Successfully wrote to file: " + fileName);
        } catch (IOException e) {
            System.err.println("Error writing to file: " + e.getMessage());
        }

        // Read from file using FileInputStream with try-with-resources
        try (FileInputStream fis = new FileInputStream(fileName)) {
            System.out.println("\\nReading from file:");

            // Read all bytes
            int byteData;
            StringBuilder sb = new StringBuilder();
            while ((byteData = fis.read()) != -1) {
                sb.append((char) byteData);
            }

            // Print the content
            System.out.println("Content: " + sb.toString());

        } catch (FileNotFoundException e) {
            System.err.println("File not found: " + e.getMessage());
        } catch (IOException e) {
            System.err.println("Error reading from file: " + e.getMessage());
        }

        // Alternative: Read all bytes at once
        try (FileInputStream fis = new FileInputStream(fileName)) {
            byte[] fileBytes = new byte[fis.available()];
            fis.read(fileBytes);
            System.out.println("\\nAlternative read: " + new String(fileBytes));
        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
        }
    }
}`,
    hint1: `Use try-with-resources syntax: try (FileOutputStream fos = new FileOutputStream(fileName)) { ... }. This automatically closes the stream.`,
    hint2: `Convert String to bytes using getBytes() before writing. When reading, use read() in a loop until it returns -1.`,
    whyItMatters: `FileInputStream and FileOutputStream are fundamental for file I/O in Java. They're the building blocks for reading and writing binary files, and understanding them is crucial for working with file systems.

**Production Pattern:**

\`\`\`java
// Copying files with large size handling
public void copyFile(String source, String destination) {
    try (FileInputStream fis = new FileInputStream(source);
         FileOutputStream fos = new FileOutputStream(destination)) {

        byte[] buffer = new byte[8192]; // 8KB buffer
        int bytesRead;

        while ((bytesRead = fis.read(buffer)) != -1) {
            fos.write(buffer, 0, bytesRead);
        }

    } catch (IOException e) {
        throw new RuntimeException("File copy error", e);
    }
}

// Writing configuration with atomicity
public void saveConfig(byte[] config) throws IOException {
    Path temp = Files.createTempFile("config", ".tmp");
    try (FileOutputStream fos = new FileOutputStream(temp.toFile())) {
        fos.write(config);
        fos.getFD().sync(); // Synchronize with disk
    }
    Files.move(temp, Paths.get("config.dat"), StandardCopyOption.ATOMIC_MOVE);
}
\`\`\`

**Practical Benefits:**

1. **Direct file access**: Works directly with the file system
2. **Try-with-resources**: Automatic resource closing and leak prevention
3. **Memory efficiency**: Streaming read/write of large files without loading everything into memory
4. **Platform independence**: Same code works on Windows, Linux, macOS`,
    order: 1,
    testCode: `import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;
import java.io.*;
import java.nio.file.Path;

// Test 1: FileOutputStream creates file
class Test1 {
    @TempDir
    Path tempDir;

    @Test
    void testFileOutputStreamCreatesFile() throws IOException {
        File file = tempDir.resolve("test.txt").toFile();
        try (FileOutputStream fos = new FileOutputStream(file)) {
            fos.write("Hello".getBytes());
        }
        assertTrue(file.exists());
    }
}

// Test 2: FileOutputStream writes content
class Test2 {
    @TempDir
    Path tempDir;

    @Test
    void testFileOutputStreamWritesContent() throws IOException {
        File file = tempDir.resolve("test.txt").toFile();
        try (FileOutputStream fos = new FileOutputStream(file)) {
            fos.write("Test".getBytes());
        }
        assertEquals(4, file.length());
    }
}

// Test 3: FileInputStream reads content
class Test3 {
    @TempDir
    Path tempDir;

    @Test
    void testFileInputStreamReads() throws IOException {
        File file = tempDir.resolve("test.txt").toFile();
        try (FileOutputStream fos = new FileOutputStream(file)) {
            fos.write("Hello".getBytes());
        }
        try (FileInputStream fis = new FileInputStream(file)) {
            byte[] bytes = new byte[5];
            fis.read(bytes);
            assertEquals("Hello", new String(bytes));
        }
    }
}

// Test 4: FileInputStream throws FileNotFoundException
class Test4 {
    @Test
    void testFileInputStreamThrowsIfNotFound() {
        assertThrows(FileNotFoundException.class, () -> {
            new FileInputStream("nonexistent_file_12345.txt");
        });
    }
}

// Test 5: FileOutputStream append mode works
class Test5 {
    @TempDir
    Path tempDir;

    @Test
    void testFileOutputStreamAppendMode() throws IOException {
        File file = tempDir.resolve("test.txt").toFile();
        try (FileOutputStream fos = new FileOutputStream(file)) {
            fos.write("Hello".getBytes());
        }
        try (FileOutputStream fos = new FileOutputStream(file, true)) {
            fos.write(" World".getBytes());
        }
        try (FileInputStream fis = new FileInputStream(file)) {
            byte[] bytes = new byte[(int) file.length()];
            fis.read(bytes);
            assertEquals("Hello World", new String(bytes));
        }
    }
}

// Test 6: FileInputStream available works
class Test6 {
    @TempDir
    Path tempDir;

    @Test
    void testFileInputStreamAvailable() throws IOException {
        File file = tempDir.resolve("test.txt").toFile();
        try (FileOutputStream fos = new FileOutputStream(file)) {
            fos.write("Test".getBytes());
        }
        try (FileInputStream fis = new FileInputStream(file)) {
            assertEquals(4, fis.available());
        }
    }
}

// Test 7: Write and read preserves data
class Test7 {
    @TempDir
    Path tempDir;

    @Test
    void testWriteReadPreservesData() throws IOException {
        File file = tempDir.resolve("test.txt").toFile();
        byte[] original = {1, 2, 3, 4, 5};
        try (FileOutputStream fos = new FileOutputStream(file)) {
            fos.write(original);
        }
        try (FileInputStream fis = new FileInputStream(file)) {
            byte[] read = new byte[5];
            fis.read(read);
            assertArrayEquals(original, read);
        }
    }
}

// Test 8: FileOutputStream overwrites by default
class Test8 {
    @TempDir
    Path tempDir;

    @Test
    void testFileOutputStreamOverwrites() throws IOException {
        File file = tempDir.resolve("test.txt").toFile();
        try (FileOutputStream fos = new FileOutputStream(file)) {
            fos.write("Hello World".getBytes());
        }
        try (FileOutputStream fos = new FileOutputStream(file)) {
            fos.write("Hi".getBytes());
        }
        assertEquals(2, file.length());
    }
}

// Test 9: FileInputStream read returns -1 at EOF
class Test9 {
    @TempDir
    Path tempDir;

    @Test
    void testFileInputStreamReturnsMinusOneAtEOF() throws IOException {
        File file = tempDir.resolve("test.txt").toFile();
        try (FileOutputStream fos = new FileOutputStream(file)) {
            fos.write("A".getBytes());
        }
        try (FileInputStream fis = new FileInputStream(file)) {
            fis.read();
            assertEquals(-1, fis.read());
        }
    }
}

// Test 10: Try-with-resources closes stream
class Test10 {
    @TempDir
    Path tempDir;

    @Test
    void testTryWithResourcesClosesStream() throws IOException {
        File file = tempDir.resolve("test.txt").toFile();
        FileOutputStream fos = new FileOutputStream(file);
        try (fos) {
            fos.write("Test".getBytes());
        }
        assertThrows(IOException.class, () -> fos.write("More".getBytes()));
    }
}`,
    translations: {
        ru: {
            title: 'Файловые потоки - FileInputStream и FileOutputStream',
            solutionCode: `import java.io.*;

public class FileStreams {
    public static void main(String[] args) {
        String fileName = "output.txt";
        String content = "Java File I/O Example";

        // Запись в файл с использованием FileOutputStream и try-with-resources
        try (FileOutputStream fos = new FileOutputStream(fileName)) {
            // Преобразуем строку в байты и записываем
            byte[] bytes = content.getBytes();
            fos.write(bytes);
            System.out.println("Успешно записано в файл: " + fileName);
        } catch (IOException e) {
            System.err.println("Ошибка записи в файл: " + e.getMessage());
        }

        // Чтение из файла с использованием FileInputStream и try-with-resources
        try (FileInputStream fis = new FileInputStream(fileName)) {
            System.out.println("\\nЧтение из файла:");

            // Читаем все байты
            int byteData;
            StringBuilder sb = new StringBuilder();
            while ((byteData = fis.read()) != -1) {
                sb.append((char) byteData);
            }

            // Выводим содержимое
            System.out.println("Содержимое: " + sb.toString());

        } catch (FileNotFoundException e) {
            System.err.println("Файл не найден: " + e.getMessage());
        } catch (IOException e) {
            System.err.println("Ошибка чтения из файла: " + e.getMessage());
        }

        // Альтернатива: читаем все байты сразу
        try (FileInputStream fis = new FileInputStream(fileName)) {
            byte[] fileBytes = new byte[fis.available()];
            fis.read(fileBytes);
            System.out.println("\\nАльтернативное чтение: " + new String(fileBytes));
        } catch (IOException e) {
            System.err.println("Ошибка: " + e.getMessage());
        }
    }
}`,
            description: `Научитесь читать и записывать файлы с помощью FileInputStream и FileOutputStream.

**Требования:**
1. Запишите текст "Java File I/O Example" в файл с именем "output.txt", используя FileOutputStream
2. Правильно закройте выходной поток
3. Прочитайте файл обратно, используя FileInputStream
4. Прочитайте байты и преобразуйте их в символы
5. Выведите содержимое, прочитанное из файла
6. Используйте try-with-resources для обеспечения правильной очистки ресурсов

FileInputStream и FileOutputStream используются для чтения и записи необработанных байтов в файлы.`,
            hint1: `Используйте синтаксис try-with-resources: try (FileOutputStream fos = new FileOutputStream(fileName)) { ... }. Это автоматически закрывает поток.`,
            hint2: `Преобразуйте String в байты с помощью getBytes() перед записью. При чтении используйте read() в цикле, пока он не вернет -1.`,
            whyItMatters: `FileInputStream и FileOutputStream являются основой для файлового ввода-вывода в Java. Они являются строительными блоками для чтения и записи двоичных файлов, и их понимание имеет решающее значение для работы с файловыми системами.

**Продакшен паттерн:**

\`\`\`java
// Копирование файла с обработкой больших размеров
public void copyFile(String source, String destination) {
    try (FileInputStream fis = new FileInputStream(source);
         FileOutputStream fos = new FileOutputStream(destination)) {

        byte[] buffer = new byte[8192]; // 8KB буфер
        int bytesRead;

        while ((bytesRead = fis.read(buffer)) != -1) {
            fos.write(buffer, 0, bytesRead);
        }

    } catch (IOException e) {
        throw new RuntimeException("Ошибка копирования файла", e);
    }
}

// Запись конфигурации с атомарностью
public void saveConfig(byte[] config) throws IOException {
    Path temp = Files.createTempFile("config", ".tmp");
    try (FileOutputStream fos = new FileOutputStream(temp.toFile())) {
        fos.write(config);
        fos.getFD().sync(); // Синхронизация с диском
    }
    Files.move(temp, Paths.get("config.dat"), StandardCopyOption.ATOMIC_MOVE);
}
\`\`\`

**Практические преимущества:**

1. **Прямой доступ к файлам**: Работа непосредственно с файловой системой
2. **Try-with-resources**: Автоматическое закрытие ресурсов и предотвращение утечек
3. **Эффективность памяти**: Потоковое чтение/запись больших файлов без загрузки всего в память
4. **Платформонезависимость**: Одинаковый код работает на Windows, Linux, macOS`
        },
        uz: {
            title: 'Fayl oqimlari - FileInputStream va FileOutputStream',
            solutionCode: `import java.io.*;

public class FileStreams {
    public static void main(String[] args) {
        String fileName = "output.txt";
        String content = "Java File I/O Example";

        // FileOutputStream va try-with-resources yordamida faylga yozish
        try (FileOutputStream fos = new FileOutputStream(fileName)) {
            // Stringni baytlarga aylantirib yozamiz
            byte[] bytes = content.getBytes();
            fos.write(bytes);
            System.out.println("Faylga muvaffaqiyatli yozildi: " + fileName);
        } catch (IOException e) {
            System.err.println("Faylga yozishda xato: " + e.getMessage());
        }

        // FileInputStream va try-with-resources yordamida fayldan o'qish
        try (FileInputStream fis = new FileInputStream(fileName)) {
            System.out.println("\\nFayldan o'qish:");

            // Barcha baytlarni o'qiymiz
            int byteData;
            StringBuilder sb = new StringBuilder();
            while ((byteData = fis.read()) != -1) {
                sb.append((char) byteData);
            }

            // Mazmunini chiqaramiz
            System.out.println("Mazmuni: " + sb.toString());

        } catch (FileNotFoundException e) {
            System.err.println("Fayl topilmadi: " + e.getMessage());
        } catch (IOException e) {
            System.err.println("Fayldan o'qishda xato: " + e.getMessage());
        }

        // Muqobil: barcha baytlarni bir vaqtning o'zida o'qish
        try (FileInputStream fis = new FileInputStream(fileName)) {
            byte[] fileBytes = new byte[fis.available()];
            fis.read(fileBytes);
            System.out.println("\\nMuqobil o'qish: " + new String(fileBytes));
        } catch (IOException e) {
            System.err.println("Xato: " + e.getMessage());
        }
    }
}`,
            description: `FileInputStream va FileOutputStream yordamida fayllardan o'qish va faylga yozishni o'rganing.

**Talablar:**
1. FileOutputStream yordamida "output.txt" nomli faylga "Java File I/O Example" matnini yozing
2. Chiqish oqimini to'g'ri yoping
3. FileInputStream yordamida faylni qaytadan o'qing
4. Baytlarni o'qing va belgilarga aylantiring
5. Fayldan o'qilgan mazmunni chiqaring
6. Resurslarni to'g'ri tozalashni ta'minlash uchun try-with-resources dan foydalaning

FileInputStream va FileOutputStream fayllardan xom baytlarni o'qish va yozish uchun ishlatiladi.`,
            hint1: `try-with-resources sintaksisidan foydalaning: try (FileOutputStream fos = new FileOutputStream(fileName)) { ... }. Bu oqimni avtomatik ravishda yopadi.`,
            hint2: `Yozishdan oldin String ni getBytes() yordamida baytlarga aylantiring. O'qishda -1 qaytarguncha siklda read() dan foydalaning.`,
            whyItMatters: `FileInputStream va FileOutputStream Java da fayl kiritish-chiqarish uchun asosdir. Ular ikkilik fayllarni o'qish va yozish uchun qurilish bloklari bo'lib, ularni tushunish fayl tizimlari bilan ishlash uchun juda muhimdir.

**Ishlab chiqarish patterni:**

\`\`\`java
// Katta hajmdagi fayllarni ko'chirish
public void copyFile(String source, String destination) {
    try (FileInputStream fis = new FileInputStream(source);
         FileOutputStream fos = new FileOutputStream(destination)) {

        byte[] buffer = new byte[8192]; // 8KB bufer
        int bytesRead;

        while ((bytesRead = fis.read(buffer)) != -1) {
            fos.write(buffer, 0, bytesRead);
        }

    } catch (IOException e) {
        throw new RuntimeException("Faylni nusxalashda xato", e);
    }
}

// Atomik konfiguratsiyani saqlash
public void saveConfig(byte[] config) throws IOException {
    Path temp = Files.createTempFile("config", ".tmp");
    try (FileOutputStream fos = new FileOutputStream(temp.toFile())) {
        fos.write(config);
        fos.getFD().sync(); // Disk bilan sinxronlash
    }
    Files.move(temp, Paths.get("config.dat"), StandardCopyOption.ATOMIC_MOVE);
}
\`\`\`

**Amaliy foydalari:**

1. **Fayllarга to'g'ridan-to'g'ri kirish**: Fayl tizimi bilan bevosita ishlash
2. **Try-with-resources**: Resurslarni avtomatik yopish va oqishni oldini olish
3. **Xotira samaradorligi**: Katta fayllarni to'liq xotiraga yuklamasdan oqim orqali o'qish/yozish
4. **Platformadan mustaqillik**: Windows, Linux, macOS da bir xil kod ishlaydi`
        }
    }
};

export default task;
