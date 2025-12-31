import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-directory-operations',
    title: 'Directory Operations',
    difficulty: 'medium',
    tags: ['java', 'nio', 'directory', 'files'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn directory operations with Files.walk(), Files.list(), and DirectoryStream.

**Requirements:**
1. Create a directory structure: "testdir/subdir1", "testdir/subdir2"
2. Create files: "testdir/file1.txt", "testdir/subdir1/file2.txt"
3. List immediate children using Files.list()
4. Walk the entire tree using Files.walk() and print all paths
5. Filter and print only .txt files using Files.walk()
6. Use DirectoryStream to iterate over directory contents
7. Count total files and directories
8. Clean up by deleting all created files and directories

Directory traversal is essential for working with file system hierarchies.`,
    initialCode: `import java.nio.file.*;
import java.io.IOException;
import java.util.stream.Stream;

public class DirectoryOperations {
    public static void main(String[] args) throws IOException {
        // Create directory structure

        // Create files

        // List immediate children

        // Walk entire tree

        // Filter and print only .txt files

        // Use DirectoryStream

        // Count files and directories

        // Clean up
    }
}`,
    solutionCode: `import java.nio.file.*;
import java.io.IOException;
import java.util.stream.Stream;
import java.util.Comparator;

public class DirectoryOperations {
    public static void main(String[] args) throws IOException {
        Path testDir = Paths.get("testdir");
        Path subDir1 = testDir.resolve("subdir1");
        Path subDir2 = testDir.resolve("subdir2");

        // Create directory structure
        Files.createDirectories(subDir1);
        Files.createDirectories(subDir2);
        System.out.println("Directories created");

        // Create files
        Files.createFile(testDir.resolve("file1.txt"));
        Files.createFile(subDir1.resolve("file2.txt"));
        System.out.println("Files created");

        // List immediate children
        System.out.println("\\nImmediate children:");
        try (Stream<Path> paths = Files.list(testDir)) {
            paths.forEach(path -> System.out.println("  " + path.getFileName()));
        }

        // Walk entire tree
        System.out.println("\\nEntire tree:");
        try (Stream<Path> paths = Files.walk(testDir)) {
            paths.forEach(path -> System.out.println("  " + path));
        }

        // Filter and print only .txt files
        System.out.println("\\nOnly .txt files:");
        try (Stream<Path> paths = Files.walk(testDir)) {
            paths.filter(path -> path.toString().endsWith(".txt"))
                 .forEach(path -> System.out.println("  " + path));
        }

        // Use DirectoryStream
        System.out.println("\\nUsing DirectoryStream:");
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(testDir)) {
            for (Path path : stream) {
                System.out.println("  " + path.getFileName());
            }
        }

        // Count files and directories
        long fileCount = 0;
        long dirCount = 0;
        try (Stream<Path> paths = Files.walk(testDir)) {
            fileCount = paths.filter(Files::isRegularFile).count();
        }
        try (Stream<Path> paths = Files.walk(testDir)) {
            dirCount = paths.filter(Files::isDirectory).count() - 1; // Exclude root
        }
        System.out.println("\\nTotal files: " + fileCount);
        System.out.println("Total directories: " + dirCount);

        // Clean up (delete in reverse order - files first, then directories)
        try (Stream<Path> paths = Files.walk(testDir)) {
            paths.sorted(Comparator.reverseOrder())
                 .forEach(path -> {
                     try {
                         Files.delete(path);
                     } catch (IOException e) {
                         System.err.println("Failed to delete: " + path);
                     }
                 });
        }
        System.out.println("Cleanup complete");
    }
}`,
    hint1: `Files.createDirectories() creates all necessary parent directories. Files.list() lists immediate children, while Files.walk() traverses the entire tree.`,
    hint2: `Use try-with-resources for Stream and DirectoryStream. Delete directories in reverse order (deepest first) using sorted(Comparator.reverseOrder()).`,
    whyItMatters: `Directory operations are crucial for file system management, batch processing, and recursive file operations. Understanding traversal methods helps you efficiently work with complex file hierarchies.

**Production Pattern:**
\`\`\`java
@Service
public class CacheCleanupService {
    private final Path cacheDir;
    private final Duration maxAge = Duration.ofDays(7);

    public CacheCleanupService(@Value("\${app.cache.dir}") String cacheDir) {
        this.cacheDir = Paths.get(cacheDir);
    }

    @Scheduled(cron = "0 0 2 * * *") // Every day at 2:00 AM
    public void cleanupOldFiles() throws IOException {
        Instant cutoff = Instant.now().minus(maxAge);

        try (Stream<Path> files = Files.walk(cacheDir)) {
            long deletedCount = files
                .filter(Files::isRegularFile)
                .filter(p -> isOlderThan(p, cutoff))
                .peek(p -> logger.debug("Deleting: {}", p))
                .map(this::deleteQuietly)
                .filter(Boolean::booleanValue)
                .count();

            logger.info("Cleaned up {} old cache files", deletedCount);
        }
    }
}
\`\`\`

**Practical Benefits:**
- Recursive directory traversal with filtering
- Safe file deletion by criteria
- Automatic cleanup of temporary data`,
    order: 2,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.nio.file.*;
import java.io.IOException;
import java.util.stream.Stream;
import java.util.Comparator;

// Test1: Test createDirectories
class Test1 {
    @Test
    public void test() throws IOException {
        Path testDir = Paths.get("test-dir/sub1/sub2");
        Files.createDirectories(testDir);
        assertTrue(Files.exists(testDir));
        assertTrue(Files.isDirectory(testDir));
        try (Stream<Path> paths = Files.walk(Paths.get("test-dir"))) {
            paths.sorted(Comparator.reverseOrder()).forEach(p -> {
                try { Files.delete(p); } catch (IOException e) {}
            });
        }
    }
}

// Test2: Test Files.list immediate children
class Test2 {
    @Test
    public void test() throws IOException {
        Path testDir = Paths.get("test-list-dir");
        Files.createDirectories(testDir);
        Files.createFile(testDir.resolve("file1.txt"));
        Files.createFile(testDir.resolve("file2.txt"));
        try (Stream<Path> paths = Files.list(testDir)) {
            assertEquals(2, paths.count());
        }
        try (Stream<Path> paths = Files.walk(testDir)) {
            paths.sorted(Comparator.reverseOrder()).forEach(p -> {
                try { Files.delete(p); } catch (IOException e) {}
            });
        }
    }
}

// Test3: Test Files.walk tree traversal
class Test3 {
    @Test
    public void test() throws IOException {
        Path testDir = Paths.get("test-walk-dir");
        Path subDir = testDir.resolve("sub");
        Files.createDirectories(subDir);
        Files.createFile(testDir.resolve("file1.txt"));
        Files.createFile(subDir.resolve("file2.txt"));
        try (Stream<Path> paths = Files.walk(testDir)) {
            assertTrue(paths.count() >= 4); // root + sub + 2 files
        }
        try (Stream<Path> paths = Files.walk(testDir)) {
            paths.sorted(Comparator.reverseOrder()).forEach(p -> {
                try { Files.delete(p); } catch (IOException e) {}
            });
        }
    }
}

// Test4: Test filtering .txt files
class Test4 {
    @Test
    public void test() throws IOException {
        Path testDir = Paths.get("test-filter-dir");
        Files.createDirectories(testDir);
        Files.createFile(testDir.resolve("file1.txt"));
        Files.createFile(testDir.resolve("file2.java"));
        try (Stream<Path> paths = Files.walk(testDir)) {
            long txtCount = paths.filter(p -> p.toString().endsWith(".txt")).count();
            assertEquals(1, txtCount);
        }
        try (Stream<Path> paths = Files.walk(testDir)) {
            paths.sorted(Comparator.reverseOrder()).forEach(p -> {
                try { Files.delete(p); } catch (IOException e) {}
            });
        }
    }
}

// Test5: Test DirectoryStream
class Test5 {
    @Test
    public void test() throws IOException {
        Path testDir = Paths.get("test-stream-dir");
        Files.createDirectories(testDir);
        Files.createFile(testDir.resolve("file1.txt"));
        int count = 0;
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(testDir)) {
            for (Path path : stream) {
                count++;
            }
        }
        assertEquals(1, count);
        try (Stream<Path> paths = Files.walk(testDir)) {
            paths.sorted(Comparator.reverseOrder()).forEach(p -> {
                try { Files.delete(p); } catch (IOException e) {}
            });
        }
    }
}

// Test6: Test counting files
class Test6 {
    @Test
    public void test() throws IOException {
        Path testDir = Paths.get("test-count-dir");
        Files.createDirectories(testDir);
        Files.createFile(testDir.resolve("f1.txt"));
        Files.createFile(testDir.resolve("f2.txt"));
        try (Stream<Path> paths = Files.walk(testDir)) {
            long fileCount = paths.filter(Files::isRegularFile).count();
            assertEquals(2, fileCount);
        }
        try (Stream<Path> paths = Files.walk(testDir)) {
            paths.sorted(Comparator.reverseOrder()).forEach(p -> {
                try { Files.delete(p); } catch (IOException e) {}
            });
        }
    }
}

// Test7: Test counting directories
class Test7 {
    @Test
    public void test() throws IOException {
        Path testDir = Paths.get("test-dir-count");
        Path subDir = testDir.resolve("sub");
        Files.createDirectories(subDir);
        try (Stream<Path> paths = Files.walk(testDir)) {
            long dirCount = paths.filter(Files::isDirectory).count();
            assertEquals(2, dirCount); // testDir + subDir
        }
        try (Stream<Path> paths = Files.walk(testDir)) {
            paths.sorted(Comparator.reverseOrder()).forEach(p -> {
                try { Files.delete(p); } catch (IOException e) {}
            });
        }
    }
}

// Test8: Test directory deletion
class Test8 {
    @Test
    public void test() throws IOException {
        Path testDir = Paths.get("test-del-dir");
        Files.createDirectories(testDir);
        Files.createFile(testDir.resolve("file.txt"));
        try (Stream<Path> paths = Files.walk(testDir)) {
            paths.sorted(Comparator.reverseOrder()).forEach(p -> {
                try { Files.delete(p); } catch (IOException e) {}
            });
        }
        assertFalse(Files.exists(testDir));
    }
}

// Test9: Test nested directory creation
class Test9 {
    @Test
    public void test() throws IOException {
        Path testDir = Paths.get("test-nested/a/b/c");
        Files.createDirectories(testDir);
        assertTrue(Files.exists(testDir));
        try (Stream<Path> paths = Files.walk(Paths.get("test-nested"))) {
            paths.sorted(Comparator.reverseOrder()).forEach(p -> {
                try { Files.delete(p); } catch (IOException e) {}
            });
        }
    }
}

// Test10: Test DirectoryStream with pattern
class Test10 {
    @Test
    public void test() throws IOException {
        Path testDir = Paths.get("test-pattern-dir");
        Files.createDirectories(testDir);
        Files.createFile(testDir.resolve("file1.txt"));
        Files.createFile(testDir.resolve("file2.java"));
        int count = 0;
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(testDir, "*.txt")) {
            for (Path path : stream) {
                count++;
            }
        }
        assertEquals(1, count);
        try (Stream<Path> paths = Files.walk(testDir)) {
            paths.sorted(Comparator.reverseOrder()).forEach(p -> {
                try { Files.delete(p); } catch (IOException e) {}
            });
        }
    }
}
`,
    translations: {
        ru: {
            title: 'Операции с каталогами',
            solutionCode: `import java.nio.file.*;
import java.io.IOException;
import java.util.stream.Stream;
import java.util.Comparator;

public class DirectoryOperations {
    public static void main(String[] args) throws IOException {
        Path testDir = Paths.get("testdir");
        Path subDir1 = testDir.resolve("subdir1");
        Path subDir2 = testDir.resolve("subdir2");

        // Создаем структуру каталогов
        Files.createDirectories(subDir1);
        Files.createDirectories(subDir2);
        System.out.println("Каталоги созданы");

        // Создаем файлы
        Files.createFile(testDir.resolve("file1.txt"));
        Files.createFile(subDir1.resolve("file2.txt"));
        System.out.println("Файлы созданы");

        // Перечисляем непосредственные дочерние элементы
        System.out.println("\\nНепосредственные дочерние:");
        try (Stream<Path> paths = Files.list(testDir)) {
            paths.forEach(path -> System.out.println("  " + path.getFileName()));
        }

        // Обходим всё дерево
        System.out.println("\\nВсё дерево:");
        try (Stream<Path> paths = Files.walk(testDir)) {
            paths.forEach(path -> System.out.println("  " + path));
        }

        // Фильтруем и выводим только .txt файлы
        System.out.println("\\nТолько .txt файлы:");
        try (Stream<Path> paths = Files.walk(testDir)) {
            paths.filter(path -> path.toString().endsWith(".txt"))
                 .forEach(path -> System.out.println("  " + path));
        }

        // Используем DirectoryStream
        System.out.println("\\nИспользуя DirectoryStream:");
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(testDir)) {
            for (Path path : stream) {
                System.out.println("  " + path.getFileName());
            }
        }

        // Подсчитываем файлы и каталоги
        long fileCount = 0;
        long dirCount = 0;
        try (Stream<Path> paths = Files.walk(testDir)) {
            fileCount = paths.filter(Files::isRegularFile).count();
        }
        try (Stream<Path> paths = Files.walk(testDir)) {
            dirCount = paths.filter(Files::isDirectory).count() - 1; // Исключаем корень
        }
        System.out.println("\\nВсего файлов: " + fileCount);
        System.out.println("Всего каталогов: " + dirCount);

        // Очистка (удаляем в обратном порядке - сначала файлы, затем каталоги)
        try (Stream<Path> paths = Files.walk(testDir)) {
            paths.sorted(Comparator.reverseOrder())
                 .forEach(path -> {
                     try {
                         Files.delete(path);
                     } catch (IOException e) {
                         System.err.println("Не удалось удалить: " + path);
                     }
                 });
        }
        System.out.println("Очистка завершена");
    }
}`,
            description: `Изучите операции с каталогами через Files.walk(), Files.list() и DirectoryStream.

**Требования:**
1. Создайте структуру каталогов: "testdir/subdir1", "testdir/subdir2"
2. Создайте файлы: "testdir/file1.txt", "testdir/subdir1/file2.txt"
3. Перечислите непосредственные дочерние элементы, используя Files.list()
4. Обойдите всё дерево, используя Files.walk(), и выведите все пути
5. Отфильтруйте и выведите только .txt файлы, используя Files.walk()
6. Используйте DirectoryStream для итерации по содержимому каталога
7. Подсчитайте общее количество файлов и каталогов
8. Очистите, удалив все созданные файлы и каталоги

Обход каталогов необходим для работы с иерархиями файловых систем.`,
            hint1: `Files.createDirectories() создает все необходимые родительские каталоги. Files.list() перечисляет непосредственные дочерние, а Files.walk() обходит всё дерево.`,
            hint2: `Используйте try-with-resources для Stream и DirectoryStream. Удаляйте каталоги в обратном порядке (сначала самые глубокие), используя sorted(Comparator.reverseOrder()).`,
            whyItMatters: `Операции с каталогами критически важны для управления файловой системой, пакетной обработки и рекурсивных операций с файлами. Понимание методов обхода помогает эффективно работать со сложными файловыми иерархиями.

**Продакшен паттерн:**
\`\`\`java
@Service
public class CacheCleanupService {
    private final Path cacheDir;
    private final Duration maxAge = Duration.ofDays(7);

    public CacheCleanupService(@Value("\${app.cache.dir}") String cacheDir) {
        this.cacheDir = Paths.get(cacheDir);
    }

    @Scheduled(cron = "0 0 2 * * *") // Каждый день в 2:00
    public void cleanupOldFiles() throws IOException {
        Instant cutoff = Instant.now().minus(maxAge);

        try (Stream<Path> files = Files.walk(cacheDir)) {
            long deletedCount = files
                .filter(Files::isRegularFile)
                .filter(p -> isOlderThan(p, cutoff))
                .peek(p -> logger.debug("Deleting: {}", p))
                .map(this::deleteQuietly)
                .filter(Boolean::booleanValue)
                .count();

            logger.info("Cleaned up {} old cache files", deletedCount);
        }
    }
}
\`\`\`

**Практические преимущества:**
- Рекурсивный обход каталогов с фильтрацией
- Безопасное удаление файлов по критериям
- Автоматическая очистка временных данных`
        },
        uz: {
            title: 'Katalog operatsiyalari',
            solutionCode: `import java.nio.file.*;
import java.io.IOException;
import java.util.stream.Stream;
import java.util.Comparator;

public class DirectoryOperations {
    public static void main(String[] args) throws IOException {
        Path testDir = Paths.get("testdir");
        Path subDir1 = testDir.resolve("subdir1");
        Path subDir2 = testDir.resolve("subdir2");

        // Katalog strukturasini yaratamiz
        Files.createDirectories(subDir1);
        Files.createDirectories(subDir2);
        System.out.println("Kataloglar yaratildi");

        // Fayllar yaratamiz
        Files.createFile(testDir.resolve("file1.txt"));
        Files.createFile(subDir1.resolve("file2.txt"));
        System.out.println("Fayllar yaratildi");

        // Bevosita bolalarni ro'yxatlaymiz
        System.out.println("\\nBevosita bolalar:");
        try (Stream<Path> paths = Files.list(testDir)) {
            paths.forEach(path -> System.out.println("  " + path.getFileName()));
        }

        // Butun daraxtni kezamiz
        System.out.println("\\nButun daraxt:");
        try (Stream<Path> paths = Files.walk(testDir)) {
            paths.forEach(path -> System.out.println("  " + path));
        }

        // Faqat .txt fayllarni filtrlash va chiqarish
        System.out.println("\\nFaqat .txt fayllar:");
        try (Stream<Path> paths = Files.walk(testDir)) {
            paths.filter(path -> path.toString().endsWith(".txt"))
                 .forEach(path -> System.out.println("  " + path));
        }

        // DirectoryStream dan foydalanamiz
        System.out.println("\\nDirectoryStream dan foydalanib:");
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(testDir)) {
            for (Path path : stream) {
                System.out.println("  " + path.getFileName());
            }
        }

        // Fayllar va kataloglarni sanash
        long fileCount = 0;
        long dirCount = 0;
        try (Stream<Path> paths = Files.walk(testDir)) {
            fileCount = paths.filter(Files::isRegularFile).count();
        }
        try (Stream<Path> paths = Files.walk(testDir)) {
            dirCount = paths.filter(Files::isDirectory).count() - 1; // Ildizni istisno qilish
        }
        System.out.println("\\nJami fayllar: " + fileCount);
        System.out.println("Jami kataloglar: " + dirCount);

        // Tozalash (teskari tartibda o'chirish - avval fayllar, keyin kataloglar)
        try (Stream<Path> paths = Files.walk(testDir)) {
            paths.sorted(Comparator.reverseOrder())
                 .forEach(path -> {
                     try {
                         Files.delete(path);
                     } catch (IOException e) {
                         System.err.println("O'chirib bo'lmadi: " + path);
                     }
                 });
        }
        System.out.println("Tozalash tugallandi");
    }
}`,
            description: `Files.walk(), Files.list() va DirectoryStream bilan katalog operatsiyalarini o'rganing.

**Talablar:**
1. Katalog strukturasini yarating: "testdir/subdir1", "testdir/subdir2"
2. Fayllar yarating: "testdir/file1.txt", "testdir/subdir1/file2.txt"
3. Files.list() yordamida bevosita bolalarni ro'yxatlang
4. Files.walk() yordamida butun daraxtni kezing va barcha yo'llarni chiqaring
5. Files.walk() yordamida faqat .txt fayllarni filtrlab chiqaring
6. Katalog tarkibini iteratsiya qilish uchun DirectoryStream dan foydalaning
7. Jami fayllar va kataloglarni sanang
8. Barcha yaratilgan fayllar va kataloglarni o'chirib tozalang

Katalogni kezish fayl tizimi ierarxiyalari bilan ishlash uchun muhimdir.`,
            hint1: `Files.createDirectories() barcha zarur ota kataloglarni yaratadi. Files.list() bevosita bolalarni ro'yxatlaydi, Files.walk() esa butun daraxtni kezadi.`,
            hint2: `Stream va DirectoryStream uchun try-with-resources dan foydalaning. Kataloglarni teskari tartibda (eng chuqurdan boshlab) o'chiring, sorted(Comparator.reverseOrder()) dan foydalanib.`,
            whyItMatters: `Katalog operatsiyalari fayl tizimini boshqarish, paketli qayta ishlash va rekursiv fayl operatsiyalari uchun juda muhim. Kezish metodlarini tushunish murakkab fayl ierarxiyalari bilan samarali ishlashga yordam beradi.

**Ishlab chiqarish patterni:**
\`\`\`java
@Service
public class CacheCleanupService {
    private final Path cacheDir;
    private final Duration maxAge = Duration.ofDays(7);

    public CacheCleanupService(@Value("\${app.cache.dir}") String cacheDir) {
        this.cacheDir = Paths.get(cacheDir);
    }

    @Scheduled(cron = "0 0 2 * * *") // Har kuni soat 2:00 da
    public void cleanupOldFiles() throws IOException {
        Instant cutoff = Instant.now().minus(maxAge);

        try (Stream<Path> files = Files.walk(cacheDir)) {
            long deletedCount = files
                .filter(Files::isRegularFile)
                .filter(p -> isOlderThan(p, cutoff))
                .peek(p -> logger.debug("O'chirilmoqda: {}", p))
                .map(this::deleteQuietly)
                .filter(Boolean::booleanValue)
                .count();

            logger.info("{} eski kesh fayllari tozalandi", deletedCount);
        }
    }
}
\`\`\`

**Amaliy foydalari:**
- Filtrlash bilan rekursiv katalog kezish
- Mezonlar bo'yicha xavfsiz fayllarni o'chirish
- Vaqtinchalik ma'lumotlarni avtomatik tozalash`
        }
    }
};

export default task;
