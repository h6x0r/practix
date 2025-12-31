import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-file-operations',
    title: 'File Operations with Files',
    difficulty: 'easy',
    tags: ['java', 'nio', 'files', 'io'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn file operations using the Files utility class.

**Requirements:**
1. Write text "Hello, NIO!" to "source.txt" using Files.writeString()
2. Read the content from "source.txt" using Files.readString()
3. Append "\\nWelcome to Java NIO" using Files.writeString() with APPEND option
4. Read all lines using Files.readAllLines()
5. Copy "source.txt" to "destination.txt" using Files.copy()
6. Move "destination.txt" to "moved.txt" using Files.move()
7. Delete all created files using Files.delete()

The Files class provides convenient methods for reading, writing, copying, and moving files without dealing with streams directly.`,
    initialCode: `import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.io.IOException;
import java.util.List;

public class FileOperations {
    public static void main(String[] args) throws IOException {
        // Write text to "source.txt"

        // Read content from "source.txt"

        // Append text to "source.txt"

        // Read all lines

        // Copy to "destination.txt"

        // Move to "moved.txt"

        // Delete all files
    }
}`,
    solutionCode: `import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.nio.file.StandardCopyOption;
import java.io.IOException;
import java.util.List;

public class FileOperations {
    public static void main(String[] args) throws IOException {
        Path source = Paths.get("source.txt");
        Path destination = Paths.get("destination.txt");
        Path moved = Paths.get("moved.txt");

        // Write text to "source.txt"
        Files.writeString(source, "Hello, NIO!");
        System.out.println("Written to source.txt");

        // Read content from "source.txt"
        String content = Files.readString(source);
        System.out.println("Content: " + content);

        // Append text to "source.txt"
        Files.writeString(source, "\\nWelcome to Java NIO", StandardOpenOption.APPEND);
        System.out.println("Appended text to source.txt");

        // Read all lines
        List<String> lines = Files.readAllLines(source);
        System.out.println("All lines:");
        for (String line : lines) {
            System.out.println("  " + line);
        }

        // Copy to "destination.txt"
        Files.copy(source, destination, StandardCopyOption.REPLACE_EXISTING);
        System.out.println("Copied to destination.txt");

        // Move to "moved.txt"
        Files.move(destination, moved, StandardCopyOption.REPLACE_EXISTING);
        System.out.println("Moved to moved.txt");

        // Delete all files
        Files.delete(source);
        Files.delete(moved);
        System.out.println("Files deleted");
    }
}`,
    hint1: `Files.writeString() and Files.readString() are convenient methods for simple text operations. Use StandardOpenOption.APPEND for appending.`,
    hint2: `Files.copy() and Files.move() require StandardCopyOption.REPLACE_EXISTING to overwrite existing files. Files.delete() removes files.`,
    whyItMatters: `The Files utility class simplifies common file operations with concise, readable methods. It handles resources automatically and provides better error handling than traditional I/O operations.

**Production Pattern:**
\`\`\`java
@Service
public class LogArchiveService {
    private final Path logDir = Paths.get("/var/log/app");
    private final Path archiveDir = Paths.get("/var/log/archive");

    public void archiveLogs() throws IOException {
        Files.createDirectories(archiveDir);

        try (var logFiles = Files.list(logDir)) {
            logFiles.filter(p -> p.toString().endsWith(".log"))
                   .filter(p -> isOlderThanWeek(p))
                   .forEach(this::archiveFile);
        }
    }

    private void archiveFile(Path logFile) {
        try {
            String fileName = logFile.getFileName().toString();
            Path destination = archiveDir.resolve(fileName + ".gz");

            // Compress and move
            compressAndMove(logFile, destination);

            logger.info("Archived: {}", fileName);
        } catch (IOException e) {
            logger.error("Failed to archive: {}", logFile, e);
        }
    }
}
\`\`\`

**Practical Benefits:**
- Automatic resource management
- Atomic copy and move operations
- Simple text file handling`,
    order: 1,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.nio.file.*;
import java.io.IOException;
import java.util.List;

// Test1: Test file write and read
class Test1 {
    @Test
    public void test() throws IOException {
        Path path = Paths.get("test-write.txt");
        Files.writeString(path, "Hello Test");
        String content = Files.readString(path);
        assertEquals("Hello Test", content);
        Files.delete(path);
    }
}

// Test2: Test file append
class Test2 {
    @Test
    public void test() throws IOException {
        Path path = Paths.get("test-append.txt");
        Files.writeString(path, "Line 1");
        Files.writeString(path, "\\nLine 2", StandardOpenOption.APPEND);
        String content = Files.readString(path);
        assertTrue(content.contains("Line 1"));
        assertTrue(content.contains("Line 2"));
        Files.delete(path);
    }
}

// Test3: Test readAllLines
class Test3 {
    @Test
    public void test() throws IOException {
        Path path = Paths.get("test-lines.txt");
        Files.writeString(path, "Line 1\\nLine 2\\nLine 3");
        List<String> lines = Files.readAllLines(path);
        assertEquals(3, lines.size());
        assertEquals("Line 1", lines.get(0));
        Files.delete(path);
    }
}

// Test4: Test file copy
class Test4 {
    @Test
    public void test() throws IOException {
        Path source = Paths.get("test-source.txt");
        Path dest = Paths.get("test-dest.txt");
        Files.writeString(source, "Copy me");
        Files.copy(source, dest, StandardCopyOption.REPLACE_EXISTING);
        assertTrue(Files.exists(dest));
        assertEquals("Copy me", Files.readString(dest));
        Files.delete(source);
        Files.delete(dest);
    }
}

// Test5: Test file move
class Test5 {
    @Test
    public void test() throws IOException {
        Path source = Paths.get("test-move-src.txt");
        Path dest = Paths.get("test-move-dst.txt");
        Files.writeString(source, "Move me");
        Files.move(source, dest, StandardCopyOption.REPLACE_EXISTING);
        assertFalse(Files.exists(source));
        assertTrue(Files.exists(dest));
        Files.delete(dest);
    }
}

// Test6: Test file delete
class Test6 {
    @Test
    public void test() throws IOException {
        Path path = Paths.get("test-del.txt");
        Files.writeString(path, "Delete me");
        assertTrue(Files.exists(path));
        Files.delete(path);
        assertFalse(Files.exists(path));
    }
}

// Test7: Test writeString with StandardOpenOption.CREATE
class Test7 {
    @Test
    public void test() throws IOException {
        Path path = Paths.get("test-create-opt.txt");
        Files.deleteIfExists(path);
        Files.writeString(path, "Created", StandardOpenOption.CREATE);
        assertTrue(Files.exists(path));
        Files.delete(path);
    }
}

// Test8: Test copy with existing destination
class Test8 {
    @Test
    public void test() throws IOException {
        Path source = Paths.get("test-copy-src.txt");
        Path dest = Paths.get("test-copy-dst.txt");
        Files.writeString(source, "Source");
        Files.writeString(dest, "Destination");
        Files.copy(source, dest, StandardCopyOption.REPLACE_EXISTING);
        assertEquals("Source", Files.readString(dest));
        Files.delete(source);
        Files.delete(dest);
    }
}

// Test9: Test readString with empty file
class Test9 {
    @Test
    public void test() throws IOException {
        Path path = Paths.get("test-empty.txt");
        Files.writeString(path, "");
        String content = Files.readString(path);
        assertEquals("", content);
        Files.delete(path);
    }
}

// Test10: Test multiple operations sequence
class Test10 {
    @Test
    public void test() throws IOException {
        Path path = Paths.get("test-sequence.txt");
        Files.writeString(path, "Start");
        Files.writeString(path, "\\nMiddle", StandardOpenOption.APPEND);
        Files.writeString(path, "\\nEnd", StandardOpenOption.APPEND);
        List<String> lines = Files.readAllLines(path);
        assertEquals(3, lines.size());
        Files.delete(path);
    }
}
`,
    translations: {
        ru: {
            title: 'Операции с файлами через Files',
            solutionCode: `import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.nio.file.StandardCopyOption;
import java.io.IOException;
import java.util.List;

public class FileOperations {
    public static void main(String[] args) throws IOException {
        Path source = Paths.get("source.txt");
        Path destination = Paths.get("destination.txt");
        Path moved = Paths.get("moved.txt");

        // Записываем текст в "source.txt"
        Files.writeString(source, "Hello, NIO!");
        System.out.println("Записано в source.txt");

        // Читаем содержимое из "source.txt"
        String content = Files.readString(source);
        System.out.println("Содержимое: " + content);

        // Добавляем текст в "source.txt"
        Files.writeString(source, "\\nWelcome to Java NIO", StandardOpenOption.APPEND);
        System.out.println("Текст добавлен в source.txt");

        // Читаем все строки
        List<String> lines = Files.readAllLines(source);
        System.out.println("Все строки:");
        for (String line : lines) {
            System.out.println("  " + line);
        }

        // Копируем в "destination.txt"
        Files.copy(source, destination, StandardCopyOption.REPLACE_EXISTING);
        System.out.println("Скопировано в destination.txt");

        // Перемещаем в "moved.txt"
        Files.move(destination, moved, StandardCopyOption.REPLACE_EXISTING);
        System.out.println("Перемещено в moved.txt");

        // Удаляем все файлы
        Files.delete(source);
        Files.delete(moved);
        System.out.println("Файлы удалены");
    }
}`,
            description: `Изучите операции с файлами, используя класс Files.

**Требования:**
1. Запишите текст "Hello, NIO!" в "source.txt", используя Files.writeString()
2. Прочитайте содержимое из "source.txt", используя Files.readString()
3. Добавьте "\\nWelcome to Java NIO", используя Files.writeString() с опцией APPEND
4. Прочитайте все строки, используя Files.readAllLines()
5. Скопируйте "source.txt" в "destination.txt", используя Files.copy()
6. Переместите "destination.txt" в "moved.txt", используя Files.move()
7. Удалите все созданные файлы, используя Files.delete()

Класс Files предоставляет удобные методы для чтения, записи, копирования и перемещения файлов без прямой работы с потоками.`,
            hint1: `Files.writeString() и Files.readString() - удобные методы для простых текстовых операций. Используйте StandardOpenOption.APPEND для добавления.`,
            hint2: `Files.copy() и Files.move() требуют StandardCopyOption.REPLACE_EXISTING для перезаписи существующих файлов. Files.delete() удаляет файлы.`,
            whyItMatters: `Класс Files упрощает распространенные файловые операции с помощью лаконичных, читаемых методов. Он автоматически управляет ресурсами и обеспечивает лучшую обработку ошибок по сравнению с традиционными операциями ввода-вывода.

**Продакшен паттерн:**
\`\`\`java
@Service
public class LogArchiveService {
    private final Path logDir = Paths.get("/var/log/app");
    private final Path archiveDir = Paths.get("/var/log/archive");

    public void archiveLogs() throws IOException {
        Files.createDirectories(archiveDir);

        try (var logFiles = Files.list(logDir)) {
            logFiles.filter(p -> p.toString().endsWith(".log"))
                   .filter(p -> isOlderThanWeek(p))
                   .forEach(this::archiveFile);
        }
    }

    private void archiveFile(Path logFile) {
        try {
            String fileName = logFile.getFileName().toString();
            Path destination = archiveDir.resolve(fileName + ".gz");

            // Сжатие и перемещение
            compressAndMove(logFile, destination);

            logger.info("Archived: {}", fileName);
        } catch (IOException e) {
            logger.error("Failed to archive: {}", logFile, e);
        }
    }
}
\`\`\`

**Практические преимущества:**
- Автоматическое управление ресурсами
- Атомарные операции копирования и перемещения
- Простая обработка текстовых файлов`
        },
        uz: {
            title: 'Files bilan fayl operatsiyalari',
            solutionCode: `import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.nio.file.StandardCopyOption;
import java.io.IOException;
import java.util.List;

public class FileOperations {
    public static void main(String[] args) throws IOException {
        Path source = Paths.get("source.txt");
        Path destination = Paths.get("destination.txt");
        Path moved = Paths.get("moved.txt");

        // "source.txt" ga matn yozamiz
        Files.writeString(source, "Hello, NIO!");
        System.out.println("source.txt ga yozildi");

        // "source.txt" dan tarkibni o'qiymiz
        String content = Files.readString(source);
        System.out.println("Tarkib: " + content);

        // "source.txt" ga matn qo'shamiz
        Files.writeString(source, "\\nWelcome to Java NIO", StandardOpenOption.APPEND);
        System.out.println("source.txt ga matn qo'shildi");

        // Barcha qatorlarni o'qiymiz
        List<String> lines = Files.readAllLines(source);
        System.out.println("Barcha qatorlar:");
        for (String line : lines) {
            System.out.println("  " + line);
        }

        // "destination.txt" ga nusxalaymiz
        Files.copy(source, destination, StandardCopyOption.REPLACE_EXISTING);
        System.out.println("destination.txt ga nusxalandi");

        // "moved.txt" ga ko'chiramiz
        Files.move(destination, moved, StandardCopyOption.REPLACE_EXISTING);
        System.out.println("moved.txt ga ko'chirildi");

        // Barcha fayllarni o'chiramiz
        Files.delete(source);
        Files.delete(moved);
        System.out.println("Fayllar o'chirildi");
    }
}`,
            description: `Files sinfi yordamida fayl operatsiyalarini o'rganing.

**Talablar:**
1. Files.writeString() yordamida "source.txt" ga "Hello, NIO!" matnini yozing
2. Files.readString() yordamida "source.txt" dan tarkibni o'qing
3. Files.writeString() ni APPEND opsiyasi bilan ishlatib "\\nWelcome to Java NIO" qo'shing
4. Files.readAllLines() yordamida barcha qatorlarni o'qing
5. Files.copy() yordamida "source.txt" ni "destination.txt" ga nusxalang
6. Files.move() yordamida "destination.txt" ni "moved.txt" ga ko'chiring
7. Files.delete() yordamida barcha yaratilgan fayllarni o'chiring

Files sinfi oqimlar bilan bevosita ishlashsiz fayllarni o'qish, yozish, nusxalash va ko'chirish uchun qulay metodlarni taqdim etadi.`,
            hint1: `Files.writeString() va Files.readString() oddiy matn operatsiyalari uchun qulay metodlardir. Qo'shish uchun StandardOpenOption.APPEND dan foydalaning.`,
            hint2: `Files.copy() va Files.move() mavjud fayllarni qayta yozish uchun StandardCopyOption.REPLACE_EXISTING talab qiladi. Files.delete() fayllarni o'chiradi.`,
            whyItMatters: `Files sinfi umumiy fayl operatsiyalarini qisqa, o'qiladigan metodlar bilan soddalashtiradi. U resurslarni avtomatik boshqaradi va an'anaviy kiritish-chiqarish operatsiyalariga nisbatan yaxshiroq xatolarni qayta ishlashni ta'minlaydi.

**Ishlab chiqarish patterni:**
\`\`\`java
@Service
public class LogArchiveService {
    private final Path logDir = Paths.get("/var/log/app");
    private final Path archiveDir = Paths.get("/var/log/archive");

    public void archiveLogs() throws IOException {
        Files.createDirectories(archiveDir);

        try (var logFiles = Files.list(logDir)) {
            logFiles.filter(p -> p.toString().endsWith(".log"))
                   .filter(p -> isOlderThanWeek(p))
                   .forEach(this::archiveFile);
        }
    }

    private void archiveFile(Path logFile) {
        try {
            String fileName = logFile.getFileName().toString();
            Path destination = archiveDir.resolve(fileName + ".gz");

            // Siqish va ko'chirish
            compressAndMove(logFile, destination);

            logger.info("Arxivlandi: {}", fileName);
        } catch (IOException e) {
            logger.error("Arxivlash muvaffaqiyatsiz: {}", logFile, e);
        }
    }
}
\`\`\`

**Amaliy foydalari:**
- Resurslarni avtomatik boshqarish
- Nusxalash va ko'chirish atomik operatsiyalari
- Matn fayllarini oddiy qayta ishlash`
        }
    }
};

export default task;
