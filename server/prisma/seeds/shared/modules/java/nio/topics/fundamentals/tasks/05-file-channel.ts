import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-file-channel',
    title: 'FileChannel Operations',
    difficulty: 'medium',
    tags: ['java', 'nio', 'channel', 'filechannel'],
    estimatedTime: '35m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn FileChannel for reading, writing, and transferring file data efficiently.

**Requirements:**
1. Write data to a file using FileChannel and ByteBuffer
2. Read data from the file using FileChannel
3. Use position() to seek to a specific location
4. Demonstrate transferTo() to copy file efficiently
5. Use transferFrom() to copy from another channel
6. Show memory-mapped file access with map()
7. Compare performance with traditional I/O

FileChannel provides high-performance, scalable file I/O with direct buffer support and zero-copy transfers.`,
    initialCode: `import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.io.IOException;

public class FileChannelDemo {
    public static void main(String[] args) throws IOException {
        // Write data using FileChannel

        // Read data using FileChannel

        // Use position to seek

        // Use transferTo for efficient copy

        // Use transferFrom

        // Memory-mapped file access
    }
}`,
    solutionCode: `import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.io.IOException;

public class FileChannelDemo {
    public static void main(String[] args) throws IOException {
        Path path = Paths.get("channel-test.txt");

        // Write data using FileChannel
        try (FileChannel channel = FileChannel.open(path,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING)) {

            ByteBuffer buffer = ByteBuffer.allocate(48);
            buffer.put("Hello FileChannel!\\n".getBytes());
            buffer.put("NIO is powerful.\\n".getBytes());

            buffer.flip();
            int bytesWritten = channel.write(buffer);
            System.out.println("Bytes written: " + bytesWritten);
        }

        // Read data using FileChannel
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            ByteBuffer buffer = ByteBuffer.allocate(48);

            int bytesRead = channel.read(buffer);
            System.out.println("\\nBytes read: " + bytesRead);

            buffer.flip();
            while (buffer.hasRemaining()) {
                System.out.print((char) buffer.get());
            }
        }

        // Use position to seek
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            System.out.println("\\nInitial position: " + channel.position());

            channel.position(6); // Skip "Hello "
            System.out.println("After seeking: " + channel.position());

            ByteBuffer buffer = ByteBuffer.allocate(11);
            channel.read(buffer);
            buffer.flip();

            System.out.print("Read from position 6: ");
            while (buffer.hasRemaining()) {
                System.out.print((char) buffer.get());
            }
            System.out.println();
        }

        // Use transferTo for efficient copy
        Path copyPath = Paths.get("channel-copy.txt");
        try (FileChannel sourceChannel = FileChannel.open(path, StandardOpenOption.READ);
             FileChannel destChannel = FileChannel.open(copyPath,
                     StandardOpenOption.CREATE,
                     StandardOpenOption.WRITE,
                     StandardOpenOption.TRUNCATE_EXISTING)) {

            long transferred = sourceChannel.transferTo(0, sourceChannel.size(), destChannel);
            System.out.println("\\nBytes transferred: " + transferred);
        }

        // Use transferFrom
        Path anotherCopy = Paths.get("another-copy.txt");
        try (FileChannel sourceChannel = FileChannel.open(copyPath, StandardOpenOption.READ);
             FileChannel destChannel = FileChannel.open(anotherCopy,
                     StandardOpenOption.CREATE,
                     StandardOpenOption.WRITE,
                     StandardOpenOption.TRUNCATE_EXISTING)) {

            long transferred = destChannel.transferFrom(sourceChannel, 0, sourceChannel.size());
            System.out.println("Bytes transferred using transferFrom: " + transferred);
        }

        // Memory-mapped file access
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            MappedByteBuffer mappedBuffer = channel.map(
                    FileChannel.MapMode.READ_ONLY, 0, channel.size());

            System.out.println("\\nMemory-mapped file content:");
            while (mappedBuffer.hasRemaining()) {
                System.out.print((char) mappedBuffer.get());
            }
        }

        // Clean up
        Files.delete(path);
        Files.delete(copyPath);
        Files.delete(anotherCopy);
        System.out.println("\\nCleanup complete");
    }
}`,
    hint1: `FileChannel.open() requires StandardOpenOption flags. Use CREATE, WRITE for writing and READ for reading. Always flip() the buffer after writing and before reading.`,
    hint2: `transferTo() and transferFrom() are zero-copy operations that directly transfer data between channels without intermediate buffers. Memory-mapped files provide fast access for large files.`,
    whyItMatters: `FileChannel provides low-level, high-performance file I/O essential for building efficient applications. It supports advanced features like zero-copy transfers, memory-mapped files, and file locking that aren't available in traditional I/O.

**Production Pattern:**
\`\`\`java
@Service
public class LargeFileProcessor {
    private static final int CHUNK_SIZE = 8192;

    public void processLargeFile(Path inputPath, Path outputPath) throws IOException {
        try (FileChannel inChannel = FileChannel.open(inputPath, StandardOpenOption.READ);
             FileChannel outChannel = FileChannel.open(outputPath,
                     StandardOpenOption.CREATE, StandardOpenOption.WRITE)) {

            long fileSize = inChannel.size();
            long position = 0;

            // Process large file in chunks
            while (position < fileSize) {
                long transferred = inChannel.transferTo(
                    position,
                    Math.min(CHUNK_SIZE, fileSize - position),
                    outChannel
                );
                position += transferred;

                // Progress reporting
                int progress = (int) ((position * 100) / fileSize);
                logger.debug("Progress: {}%", progress);
            }
        }
    }

    // Memory-mapped file for fast access
    public ByteBuffer mapFile(Path path) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            return channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
        }
    }
}
\`\`\`

**Practical Benefits:**
- Efficient large file processing
- Zero-copy data transfer between channels
- Memory-mapped files for fast access`,
    order: 4,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.io.IOException;

// Test1: Test FileChannel write
class Test1 {
    @Test
    public void test() throws IOException {
        Path path = Paths.get("test-channel-write.txt");
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.CREATE, StandardOpenOption.WRITE)) {
            ByteBuffer buffer = ByteBuffer.wrap("Hello".getBytes());
            int written = channel.write(buffer);
            assertEquals(5, written);
        }
        Files.deleteIfExists(path);
    }
}

// Test2: Test FileChannel read
class Test2 {
    @Test
    public void test() throws IOException {
        Path path = Paths.get("test-channel-read.txt");
        Files.writeString(path, "Hello");
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            ByteBuffer buffer = ByteBuffer.allocate(5);
            int read = channel.read(buffer);
            assertEquals(5, read);
        }
        Files.delete(path);
    }
}

// Test3: Test channel position
class Test3 {
    @Test
    public void test() throws IOException {
        Path path = Paths.get("test-channel-pos.txt");
        Files.writeString(path, "Hello World");
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            assertEquals(0, channel.position());
            channel.position(6);
            assertEquals(6, channel.position());
        }
        Files.delete(path);
    }
}

// Test4: Test channel size
class Test4 {
    @Test
    public void test() throws IOException {
        Path path = Paths.get("test-channel-size.txt");
        Files.writeString(path, "Test");
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            assertEquals(4, channel.size());
        }
        Files.delete(path);
    }
}

// Test5: Test transferTo
class Test5 {
    @Test
    public void test() throws IOException {
        Path src = Paths.get("test-src.txt");
        Path dst = Paths.get("test-dst.txt");
        Files.writeString(src, "Transfer me");
        try (FileChannel srcCh = FileChannel.open(src, StandardOpenOption.READ);
             FileChannel dstCh = FileChannel.open(dst, StandardOpenOption.CREATE, StandardOpenOption.WRITE)) {
            long transferred = srcCh.transferTo(0, srcCh.size(), dstCh);
            assertTrue(transferred > 0);
        }
        assertEquals("Transfer me", Files.readString(dst));
        Files.delete(src);
        Files.delete(dst);
    }
}

// Test6: Test transferFrom
class Test6 {
    @Test
    public void test() throws IOException {
        Path src = Paths.get("test-from-src.txt");
        Path dst = Paths.get("test-from-dst.txt");
        Files.writeString(src, "Data");
        try (FileChannel srcCh = FileChannel.open(src, StandardOpenOption.READ);
             FileChannel dstCh = FileChannel.open(dst, StandardOpenOption.CREATE, StandardOpenOption.WRITE)) {
            long transferred = dstCh.transferFrom(srcCh, 0, srcCh.size());
            assertTrue(transferred > 0);
        }
        Files.delete(src);
        Files.delete(dst);
    }
}

// Test7: Test write with position seek
class Test7 {
    @Test
    public void test() throws IOException {
        Path path = Paths.get("test-seek-write.txt");
        Files.writeString(path, "Hello World");
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            channel.position(6);
            ByteBuffer buffer = ByteBuffer.allocate(5);
            channel.read(buffer);
            buffer.flip();
            assertEquals('W', (char) buffer.get());
        }
        Files.delete(path);
    }
}

// Test8: Test truncate
class Test8 {
    @Test
    public void test() throws IOException {
        Path path = Paths.get("test-truncate.txt");
        Files.writeString(path, "Hello World");
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.WRITE)) {
            channel.truncate(5);
            assertEquals(5, channel.size());
        }
        Files.delete(path);
    }
}

// Test9: Test force (flush)
class Test9 {
    @Test
    public void test() throws IOException {
        Path path = Paths.get("test-force.txt");
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.CREATE, StandardOpenOption.WRITE)) {
            ByteBuffer buffer = ByteBuffer.wrap("Data".getBytes());
            channel.write(buffer);
            channel.force(true);
        }
        assertTrue(Files.exists(path));
        Files.delete(path);
    }
}

// Test10: Test isOpen
class Test10 {
    @Test
    public void test() throws IOException {
        Path path = Paths.get("test-open.txt");
        Files.writeString(path, "Test");
        FileChannel channel = FileChannel.open(path, StandardOpenOption.READ);
        assertTrue(channel.isOpen());
        channel.close();
        assertFalse(channel.isOpen());
        Files.delete(path);
    }
}
`,
    translations: {
        ru: {
            title: 'Операции FileChannel',
            solutionCode: `import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.io.IOException;

public class FileChannelDemo {
    public static void main(String[] args) throws IOException {
        Path path = Paths.get("channel-test.txt");

        // Записываем данные, используя FileChannel
        try (FileChannel channel = FileChannel.open(path,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING)) {

            ByteBuffer buffer = ByteBuffer.allocate(48);
            buffer.put("Hello FileChannel!\\n".getBytes());
            buffer.put("NIO is powerful.\\n".getBytes());

            buffer.flip();
            int bytesWritten = channel.write(buffer);
            System.out.println("Записано байт: " + bytesWritten);
        }

        // Читаем данные, используя FileChannel
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            ByteBuffer buffer = ByteBuffer.allocate(48);

            int bytesRead = channel.read(buffer);
            System.out.println("\\nПрочитано байт: " + bytesRead);

            buffer.flip();
            while (buffer.hasRemaining()) {
                System.out.print((char) buffer.get());
            }
        }

        // Используем position для перемещения
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            System.out.println("\\nНачальная позиция: " + channel.position());

            channel.position(6); // Пропускаем "Hello "
            System.out.println("После перемещения: " + channel.position());

            ByteBuffer buffer = ByteBuffer.allocate(11);
            channel.read(buffer);
            buffer.flip();

            System.out.print("Прочитано с позиции 6: ");
            while (buffer.hasRemaining()) {
                System.out.print((char) buffer.get());
            }
            System.out.println();
        }

        // Используем transferTo для эффективного копирования
        Path copyPath = Paths.get("channel-copy.txt");
        try (FileChannel sourceChannel = FileChannel.open(path, StandardOpenOption.READ);
             FileChannel destChannel = FileChannel.open(copyPath,
                     StandardOpenOption.CREATE,
                     StandardOpenOption.WRITE,
                     StandardOpenOption.TRUNCATE_EXISTING)) {

            long transferred = sourceChannel.transferTo(0, sourceChannel.size(), destChannel);
            System.out.println("\\nПередано байт: " + transferred);
        }

        // Используем transferFrom
        Path anotherCopy = Paths.get("another-copy.txt");
        try (FileChannel sourceChannel = FileChannel.open(copyPath, StandardOpenOption.READ);
             FileChannel destChannel = FileChannel.open(anotherCopy,
                     StandardOpenOption.CREATE,
                     StandardOpenOption.WRITE,
                     StandardOpenOption.TRUNCATE_EXISTING)) {

            long transferred = destChannel.transferFrom(sourceChannel, 0, sourceChannel.size());
            System.out.println("Передано байт через transferFrom: " + transferred);
        }

        // Доступ к файлу, отображенному в памяти
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            MappedByteBuffer mappedBuffer = channel.map(
                    FileChannel.MapMode.READ_ONLY, 0, channel.size());

            System.out.println("\\nСодержимое файла, отображенного в памяти:");
            while (mappedBuffer.hasRemaining()) {
                System.out.print((char) mappedBuffer.get());
            }
        }

        // Очистка
        Files.delete(path);
        Files.delete(copyPath);
        Files.delete(anotherCopy);
        System.out.println("\\nОчистка завершена");
    }
}`,
            description: `Изучите FileChannel для эффективного чтения, записи и передачи файловых данных.

**Требования:**
1. Запишите данные в файл, используя FileChannel и ByteBuffer
2. Прочитайте данные из файла, используя FileChannel
3. Используйте position() для перемещения к определенной позиции
4. Продемонстрируйте transferTo() для эффективного копирования файла
5. Используйте transferFrom() для копирования из другого канала
6. Покажите доступ к файлу, отображенному в памяти, с помощью map()
7. Сравните производительность с традиционным вводом-выводом

FileChannel обеспечивает высокопроизводительный, масштабируемый файловый ввод-вывод с поддержкой прямых буферов и передачи без копирования.`,
            hint1: `FileChannel.open() требует флаги StandardOpenOption. Используйте CREATE, WRITE для записи и READ для чтения. Всегда вызывайте flip() после записи и перед чтением.`,
            hint2: `transferTo() и transferFrom() - это операции без копирования, которые напрямую передают данные между каналами без промежуточных буферов. Файлы, отображенные в памяти, обеспечивают быстрый доступ к большим файлам.`,
            whyItMatters: `FileChannel обеспечивает низкоуровневый, высокопроизводительный файловый ввод-вывод, необходимый для создания эффективных приложений. Он поддерживает расширенные функции, такие как передача без копирования, файлы, отображенные в памяти, и блокировка файлов, которые недоступны в традиционном вводе-выводе.

**Продакшен паттерн:**
\`\`\`java
@Service
public class LargeFileProcessor {
    private static final int CHUNK_SIZE = 8192;

    public void processLargeFile(Path inputPath, Path outputPath) throws IOException {
        try (FileChannel inChannel = FileChannel.open(inputPath, StandardOpenOption.READ);
             FileChannel outChannel = FileChannel.open(outputPath,
                     StandardOpenOption.CREATE, StandardOpenOption.WRITE)) {

            long fileSize = inChannel.size();
            long position = 0;

            // Обработка большого файла частями
            while (position < fileSize) {
                long transferred = inChannel.transferTo(
                    position,
                    Math.min(CHUNK_SIZE, fileSize - position),
                    outChannel
                );
                position += transferred;

                // Отчет о прогрессе
                int progress = (int) ((position * 100) / fileSize);
                logger.debug("Progress: {}%", progress);
            }
        }
    }

    // Memory-mapped файл для быстрого доступа
    public ByteBuffer mapFile(Path path) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            return channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
        }
    }
}
\`\`\`

**Практические преимущества:**
- Эффективная обработка больших файлов
- Zero-copy передача данных между каналами
- Memory-mapped файлы для быстрого доступа`
        },
        uz: {
            title: 'FileChannel Operatsiyalari',
            solutionCode: `import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.io.IOException;

public class FileChannelDemo {
    public static void main(String[] args) throws IOException {
        Path path = Paths.get("channel-test.txt");

        // FileChannel yordamida ma'lumot yozamiz
        try (FileChannel channel = FileChannel.open(path,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING)) {

            ByteBuffer buffer = ByteBuffer.allocate(48);
            buffer.put("Hello FileChannel!\\n".getBytes());
            buffer.put("NIO is powerful.\\n".getBytes());

            buffer.flip();
            int bytesWritten = channel.write(buffer);
            System.out.println("Yozilgan baytlar: " + bytesWritten);
        }

        // FileChannel yordamida ma'lumot o'qiymiz
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            ByteBuffer buffer = ByteBuffer.allocate(48);

            int bytesRead = channel.read(buffer);
            System.out.println("\\nO'qilgan baytlar: " + bytesRead);

            buffer.flip();
            while (buffer.hasRemaining()) {
                System.out.print((char) buffer.get());
            }
        }

        // position yordamida o'tish
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            System.out.println("\\nBoshlang'ich pozitsiya: " + channel.position());

            channel.position(6); // "Hello " ni o'tkazib yuboramiz
            System.out.println("O'tishdan keyin: " + channel.position());

            ByteBuffer buffer = ByteBuffer.allocate(11);
            channel.read(buffer);
            buffer.flip();

            System.out.print("6-pozitsiyadan o'qilgan: ");
            while (buffer.hasRemaining()) {
                System.out.print((char) buffer.get());
            }
            System.out.println();
        }

        // Samarali nusxalash uchun transferTo dan foydalanamiz
        Path copyPath = Paths.get("channel-copy.txt");
        try (FileChannel sourceChannel = FileChannel.open(path, StandardOpenOption.READ);
             FileChannel destChannel = FileChannel.open(copyPath,
                     StandardOpenOption.CREATE,
                     StandardOpenOption.WRITE,
                     StandardOpenOption.TRUNCATE_EXISTING)) {

            long transferred = sourceChannel.transferTo(0, sourceChannel.size(), destChannel);
            System.out.println("\\nO'tkazilgan baytlar: " + transferred);
        }

        // transferFrom dan foydalanamiz
        Path anotherCopy = Paths.get("another-copy.txt");
        try (FileChannel sourceChannel = FileChannel.open(copyPath, StandardOpenOption.READ);
             FileChannel destChannel = FileChannel.open(anotherCopy,
                     StandardOpenOption.CREATE,
                     StandardOpenOption.WRITE,
                     StandardOpenOption.TRUNCATE_EXISTING)) {

            long transferred = destChannel.transferFrom(sourceChannel, 0, sourceChannel.size());
            System.out.println("transferFrom orqali o'tkazilgan baytlar: " + transferred);
        }

        // Xotirada joylashgan faylga kirish
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            MappedByteBuffer mappedBuffer = channel.map(
                    FileChannel.MapMode.READ_ONLY, 0, channel.size());

            System.out.println("\\nXotirada joylashgan fayl tarkibi:");
            while (mappedBuffer.hasRemaining()) {
                System.out.print((char) mappedBuffer.get());
            }
        }

        // Tozalash
        Files.delete(path);
        Files.delete(copyPath);
        Files.delete(anotherCopy);
        System.out.println("\\nTozalash tugallandi");
    }
}`,
            description: `Fayl ma'lumotlarini samarali o'qish, yozish va o'tkazish uchun FileChannel ni o'rganing.

**Talablar:**
1. FileChannel va ByteBuffer yordamida faylga ma'lumot yozing
2. FileChannel yordamida fayldan ma'lumot o'qing
3. Muayyan joyga o'tish uchun position() dan foydalaning
4. Faylni samarali nusxalash uchun transferTo() ni ko'rsating
5. Boshqa kanaldan nusxalash uchun transferFrom() dan foydalaning
6. map() yordamida xotirada joylashgan faylga kirishni ko'rsating
7. An'anaviy kirish-chiqarish bilan ishlashni solishtiring

FileChannel to'g'ridan-to'g'ri bufer qo'llab-quvvatlash va nusxa olmay o'tkazish bilan yuqori unumli, kengaytirilishi mumkin bo'lgan fayl kirish-chiqarish ni ta'minlaydi.`,
            hint1: `FileChannel.open() StandardOpenOption bayroqlarini talab qiladi. Yozish uchun CREATE, WRITE va o'qish uchun READ dan foydalaning. Yozishdan keyin va o'qishdan oldin doimo flip() ni chaqiring.`,
            hint2: `transferTo() va transferFrom() nusxa olmay o'tkazish operatsiyalari bo'lib, oraliq buferlar ishlatmasdan to'g'ridan-to'g'ri kanallar o'rtasida ma'lumot o'tkazadi. Xotirada joylashgan fayllar katta fayllar uchun tez kirishni ta'minlaydi.`,
            whyItMatters: `FileChannel samarali ilovalarni yaratish uchun zarur bo'lgan past darajali, yuqori unumli fayl kirish-chiqarish ni ta'minlaydi. U nusxa olmay o'tkazish, xotirada joylashgan fayllar va an'anaviy kirish-chiqarish da mavjud bo'lmagan fayl blokirovkalash kabi ilg'or xususiyatlarni qo'llab-quvvatlaydi.

**Ishlab chiqarish patterni:**
\`\`\`java
@Service
public class LargeFileProcessor {
    private static final int CHUNK_SIZE = 8192;

    public void processLargeFile(Path inputPath, Path outputPath) throws IOException {
        try (FileChannel inChannel = FileChannel.open(inputPath, StandardOpenOption.READ);
             FileChannel outChannel = FileChannel.open(outputPath,
                     StandardOpenOption.CREATE, StandardOpenOption.WRITE)) {

            long fileSize = inChannel.size();
            long position = 0;

            // Katta faylni qismlarga bo'lib qayta ishlash
            while (position < fileSize) {
                long transferred = inChannel.transferTo(
                    position,
                    Math.min(CHUNK_SIZE, fileSize - position),
                    outChannel
                );
                position += transferred;

                // Progress hisoboti
                int progress = (int) ((position * 100) / fileSize);
                logger.debug("Progress: {}%", progress);
            }
        }
    }

    // Tez kirish uchun xotirada joylashgan fayl
    public ByteBuffer mapFile(Path path) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            return channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
        }
    }
}
\`\`\`

**Amaliy foydalari:**
- Katta fayllarni samarali qayta ishlash
- Kanallar o'rtasida zero-copy ma'lumot o'tkazish
- Tez kirish uchun xotirada joylashgan fayllar`
        }
    }
};

export default task;
