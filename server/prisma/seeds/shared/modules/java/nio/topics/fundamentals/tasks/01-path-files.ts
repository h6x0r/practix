import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-path-files',
    title: 'Path and Files Basics',
    difficulty: 'easy',
    tags: ['java', 'nio', 'path', 'files'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn Path and Files utility methods in Java NIO.2.

**Requirements:**
1. Create a Path object for "test.txt" in the current directory
2. Get the file name, parent, and absolute path
3. Check if the file exists using Files.exists()
4. Create the file if it doesn't exist using Files.createFile()
5. Check if it's a regular file and readable
6. Get the file size using Files.size()
7. Print all the information

Path and Files provide modern, fluent APIs for file system operations, replacing the legacy java.io.File class.`,
    initialCode: `import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.Files;
import java.io.IOException;

public class PathFilesBasics {
    public static void main(String[] args) throws IOException {
        // Create a Path object for "test.txt"

        // Get file name, parent, and absolute path

        // Check if file exists

        // Create file if it doesn't exist

        // Check if it's a regular file and readable

        // Get file size

        // Print all information
    }
}`,
    solutionCode: `import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.Files;
import java.io.IOException;

public class PathFilesBasics {
    public static void main(String[] args) throws IOException {
        // Create a Path object for "test.txt"
        Path path = Paths.get("test.txt");
        System.out.println("Path: " + path);

        // Get file name, parent, and absolute path
        System.out.println("File name: " + path.getFileName());
        System.out.println("Parent: " + path.getParent());
        System.out.println("Absolute path: " + path.toAbsolutePath());

        // Check if file exists
        boolean exists = Files.exists(path);
        System.out.println("File exists: " + exists);

        // Create file if it doesn't exist
        if (!exists) {
            Files.createFile(path);
            System.out.println("File created successfully");
        }

        // Check if it's a regular file and readable
        System.out.println("Is regular file: " + Files.isRegularFile(path));
        System.out.println("Is readable: " + Files.isReadable(path));
        System.out.println("Is writable: " + Files.isWritable(path));

        // Get file size
        long size = Files.size(path);
        System.out.println("File size: " + size + " bytes");

        // Clean up
        Files.deleteIfExists(path);
        System.out.println("File deleted");
    }
}`,
    hint1: `Use Paths.get() to create a Path object. Path provides methods like getFileName() and getParent() for path manipulation.`,
    hint2: `Files class provides static utility methods like exists(), createFile(), isRegularFile(), and size() for file operations.`,
    whyItMatters: `Path and Files are the foundation of Java NIO.2, providing a modern, efficient, and platform-independent way to work with file systems. They replace the older java.io.File API with better error handling and more capabilities.

**Production Pattern:**
\`\`\`java
@Service
public class ConfigurationService {
    private final Path configPath;

    public ConfigurationService(@Value("\${app.config.dir}") String configDir) {
        this.configPath = Paths.get(configDir).resolve("application.properties");
    }

    public Properties loadConfig() throws IOException {
        if (!Files.exists(configPath)) {
            throw new ConfigurationException("Config file not found: " + configPath);
        }

        if (!Files.isReadable(configPath)) {
            throw new ConfigurationException("Config file not readable");
        }

        Properties props = new Properties();
        try (var input = Files.newInputStream(configPath)) {
            props.load(input);
        }
        return props;
    }
}
\`\`\`

**Practical Benefits:**
- Safe file path handling
- Existence and permission checking
- Platform-independent path construction`,
    order: 0,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.nio.file.*;
import java.io.IOException;

// Test1: Test Path creation
class Test1 {
    @Test
    public void test() {
        Path path = Paths.get("test.txt");
        assertNotNull(path);
        assertEquals("test.txt", path.getFileName().toString());
    }
}

// Test2: Test file existence check
class Test2 {
    @Test
    public void test() throws IOException {
        Path path = Paths.get("test-check.txt");
        Files.deleteIfExists(path);
        assertFalse(Files.exists(path));
        Files.createFile(path);
        assertTrue(Files.exists(path));
        Files.delete(path);
    }
}

// Test3: Test file creation
class Test3 {
    @Test
    public void test() throws IOException {
        Path path = Paths.get("test-create.txt");
        Files.deleteIfExists(path);
        Files.createFile(path);
        assertTrue(Files.exists(path));
        Files.delete(path);
    }
}

// Test4: Test isRegularFile
class Test4 {
    @Test
    public void test() throws IOException {
        Path path = Paths.get("test-regular.txt");
        Files.deleteIfExists(path);
        Files.createFile(path);
        assertTrue(Files.isRegularFile(path));
        Files.delete(path);
    }
}

// Test5: Test file readable check
class Test5 {
    @Test
    public void test() throws IOException {
        Path path = Paths.get("test-readable.txt");
        Files.deleteIfExists(path);
        Files.createFile(path);
        assertTrue(Files.isReadable(path));
        Files.delete(path);
    }
}

// Test6: Test file writable check
class Test6 {
    @Test
    public void test() throws IOException {
        Path path = Paths.get("test-writable.txt");
        Files.deleteIfExists(path);
        Files.createFile(path);
        assertTrue(Files.isWritable(path));
        Files.delete(path);
    }
}

// Test7: Test file size
class Test7 {
    @Test
    public void test() throws IOException {
        Path path = Paths.get("test-size.txt");
        Files.deleteIfExists(path);
        Files.createFile(path);
        assertEquals(0L, Files.size(path));
        Files.delete(path);
    }
}

// Test8: Test absolute path
class Test8 {
    @Test
    public void test() {
        Path path = Paths.get("test.txt");
        Path absolute = path.toAbsolutePath();
        assertTrue(absolute.isAbsolute());
    }
}

// Test9: Test file deletion
class Test9 {
    @Test
    public void test() throws IOException {
        Path path = Paths.get("test-delete.txt");
        Files.deleteIfExists(path);
        Files.createFile(path);
        assertTrue(Files.exists(path));
        Files.delete(path);
        assertFalse(Files.exists(path));
    }
}

// Test10: Test deleteIfExists
class Test10 {
    @Test
    public void test() throws IOException {
        Path path = Paths.get("test-delete-if.txt");
        Files.deleteIfExists(path);
        assertFalse(Files.exists(path));
        boolean deleted = Files.deleteIfExists(path);
        assertFalse(deleted);
    }
}
`,
    translations: {
        ru: {
            title: 'Основы Path и Files',
            solutionCode: `import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.Files;
import java.io.IOException;

public class PathFilesBasics {
    public static void main(String[] args) throws IOException {
        // Создаем объект Path для "test.txt"
        Path path = Paths.get("test.txt");
        System.out.println("Путь: " + path);

        // Получаем имя файла, родителя и абсолютный путь
        System.out.println("Имя файла: " + path.getFileName());
        System.out.println("Родитель: " + path.getParent());
        System.out.println("Абсолютный путь: " + path.toAbsolutePath());

        // Проверяем, существует ли файл
        boolean exists = Files.exists(path);
        System.out.println("Файл существует: " + exists);

        // Создаем файл, если он не существует
        if (!exists) {
            Files.createFile(path);
            System.out.println("Файл успешно создан");
        }

        // Проверяем, является ли файл обычным и доступен ли для чтения
        System.out.println("Обычный файл: " + Files.isRegularFile(path));
        System.out.println("Доступен для чтения: " + Files.isReadable(path));
        System.out.println("Доступен для записи: " + Files.isWritable(path));

        // Получаем размер файла
        long size = Files.size(path);
        System.out.println("Размер файла: " + size + " байт");

        // Очистка
        Files.deleteIfExists(path);
        System.out.println("Файл удален");
    }
}`,
            description: `Изучите методы Path и Files в Java NIO.2.

**Требования:**
1. Создайте объект Path для "test.txt" в текущем каталоге
2. Получите имя файла, родителя и абсолютный путь
3. Проверьте, существует ли файл, используя Files.exists()
4. Создайте файл, если он не существует, используя Files.createFile()
5. Проверьте, является ли файл обычным и доступен ли для чтения
6. Получите размер файла, используя Files.size()
7. Выведите всю информацию

Path и Files предоставляют современные, удобные API для операций с файловой системой, заменяя устаревший класс java.io.File.`,
            hint1: `Используйте Paths.get() для создания объекта Path. Path предоставляет методы вроде getFileName() и getParent() для манипуляции путями.`,
            hint2: `Класс Files предоставляет статические утилитные методы вроде exists(), createFile(), isRegularFile() и size() для операций с файлами.`,
            whyItMatters: `Path и Files - основа Java NIO.2, предоставляющая современный, эффективный и платформенно-независимый способ работы с файловыми системами. Они заменяют старый API java.io.File с лучшей обработкой ошибок и большими возможностями.

**Продакшен паттерн:**
\`\`\`java
@Service
public class ConfigurationService {
    private final Path configPath;

    public ConfigurationService(@Value("\${app.config.dir}") String configDir) {
        this.configPath = Paths.get(configDir).resolve("application.properties");
    }

    public Properties loadConfig() throws IOException {
        if (!Files.exists(configPath)) {
            throw new ConfigurationException("Config file not found: " + configPath);
        }

        if (!Files.isReadable(configPath)) {
            throw new ConfigurationException("Config file not readable");
        }

        Properties props = new Properties();
        try (var input = Files.newInputStream(configPath)) {
            props.load(input);
        }
        return props;
    }
}
\`\`\`

**Практические преимущества:**
- Безопасная работа с путями к файлам
- Проверка существования и прав доступа
- Платформенно-независимое построение путей`
        },
        uz: {
            title: 'Path va Files Asoslari',
            solutionCode: `import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.Files;
import java.io.IOException;

public class PathFilesBasics {
    public static void main(String[] args) throws IOException {
        // "test.txt" uchun Path obyekti yaratamiz
        Path path = Paths.get("test.txt");
        System.out.println("Yo'l: " + path);

        // Fayl nomi, ota va mutlaq yo'lni olamiz
        System.out.println("Fayl nomi: " + path.getFileName());
        System.out.println("Ota: " + path.getParent());
        System.out.println("Mutlaq yo'l: " + path.toAbsolutePath());

        // Fayl mavjudligini tekshiramiz
        boolean exists = Files.exists(path);
        System.out.println("Fayl mavjud: " + exists);

        // Agar fayl mavjud bo'lmasa, yaratamiz
        if (!exists) {
            Files.createFile(path);
            System.out.println("Fayl muvaffaqiyatli yaratildi");
        }

        // Oddiy fayl ekanligini va o'qish mumkinligini tekshiramiz
        System.out.println("Oddiy fayl: " + Files.isRegularFile(path));
        System.out.println("O'qish mumkin: " + Files.isReadable(path));
        System.out.println("Yozish mumkin: " + Files.isWritable(path));

        // Fayl hajmini olamiz
        long size = Files.size(path);
        System.out.println("Fayl hajmi: " + size + " bayt");

        // Tozalash
        Files.deleteIfExists(path);
        System.out.println("Fayl o'chirildi");
    }
}`,
            description: `Java NIO.2 da Path va Files metodlarini o'rganing.

**Talablar:**
1. Joriy katalogda "test.txt" uchun Path obyekti yarating
2. Fayl nomi, ota va mutlaq yo'lni oling
3. Files.exists() yordamida fayl mavjudligini tekshiring
4. Agar fayl mavjud bo'lmasa, Files.createFile() yordamida yarating
5. Oddiy fayl ekanligini va o'qish mumkinligini tekshiring
6. Files.size() yordamida fayl hajmini oling
7. Barcha ma'lumotni chiqaring

Path va Files fayl tizimi operatsiyalari uchun zamonaviy, qulay API larni taqdim etadi, eski java.io.File sinfini almashtiradi.`,
            hint1: `Path obyekti yaratish uchun Paths.get() dan foydalaning. Path yo'llarni manipulyatsiya qilish uchun getFileName() va getParent() kabi metodlarni taqdim etadi.`,
            hint2: `Files sinfi fayl operatsiyalari uchun exists(), createFile(), isRegularFile() va size() kabi statik yordamchi metodlarni taqdim etadi.`,
            whyItMatters: `Path va Files Java NIO.2 ning asosi bo'lib, fayl tizimlari bilan ishlashning zamonaviy, samarali va platformadan mustaqil usulini taqdim etadi. Ular eski java.io.File API ni yaxshiroq xatolarni qayta ishlash va ko'proq imkoniyatlar bilan almashtiradi.

**Ishlab chiqarish patterni:**
\`\`\`java
@Service
public class ConfigurationService {
    private final Path configPath;

    public ConfigurationService(@Value("\${app.config.dir}") String configDir) {
        this.configPath = Paths.get(configDir).resolve("application.properties");
    }

    public Properties loadConfig() throws IOException {
        if (!Files.exists(configPath)) {
            throw new ConfigurationException("Konfiguratsiya fayli topilmadi: " + configPath);
        }

        if (!Files.isReadable(configPath)) {
            throw new ConfigurationException("Konfiguratsiya faylini o'qib bo'lmaydi");
        }

        Properties props = new Properties();
        try (var input = Files.newInputStream(configPath)) {
            props.load(input);
        }
        return props;
    }
}
\`\`\`

**Amaliy foydalari:**
- Fayl yo'llari bilan xavfsiz ishlash
- Mavjudlik va kirish huquqlarini tekshirish
- Platformadan mustaqil yo'llar qurish`
        }
    }
};

export default task;
