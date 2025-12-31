import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-data-streams',
    title: 'Data Streams for Primitive Types',
    difficulty: 'medium',
    tags: ['java', 'io', 'data-streams', 'primitives'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn how to read and write primitive data types using DataInputStream and DataOutputStream.

**Requirements:**
1. Create a DataOutputStream to write primitive types to a file
2. Write various primitives: int, double, boolean, char, long
3. Write a UTF string using writeUTF()
4. Close the output stream
5. Read the data back using DataInputStream in the same order
6. Display all the values read from the file

DataInputStream and DataOutputStream allow reading and writing Java primitive types in a machine-independent way.`,
    initialCode: `import java.io.*;

public class DataStreams {
    public static void main(String[] args) {
        String fileName = "data.bin";

        // Write primitive types using DataOutputStream

        // Read primitive types using DataInputStream
    }
}`,
    solutionCode: `import java.io.*;

public class DataStreams {
    public static void main(String[] args) {
        String fileName = "data.bin";

        // Write primitive types using DataOutputStream
        try (FileOutputStream fos = new FileOutputStream(fileName);
             DataOutputStream dos = new DataOutputStream(fos)) {

            // Write different primitive types
            dos.writeInt(42);
            dos.writeDouble(3.14159);
            dos.writeBoolean(true);
            dos.writeChar('A');
            dos.writeLong(1234567890L);
            dos.writeFloat(2.5f);
            dos.writeShort((short) 100);
            dos.writeByte(127);

            // Write a string in UTF format
            dos.writeUTF("Hello DataStreams");

            System.out.println("Successfully wrote primitive data to file");

        } catch (IOException e) {
            System.err.println("Error writing: " + e.getMessage());
        }

        // Read primitive types using DataInputStream
        try (FileInputStream fis = new FileInputStream(fileName);
             DataInputStream dis = new DataInputStream(fis)) {

            System.out.println("\\nReading primitive data from file:");

            // Read in the same order as written
            int intValue = dis.readInt();
            double doubleValue = dis.readDouble();
            boolean boolValue = dis.readBoolean();
            char charValue = dis.readChar();
            long longValue = dis.readLong();
            float floatValue = dis.readFloat();
            short shortValue = dis.readShort();
            byte byteValue = dis.readByte();
            String stringValue = dis.readUTF();

            // Display all values
            System.out.println("int: " + intValue);
            System.out.println("double: " + doubleValue);
            System.out.println("boolean: " + boolValue);
            System.out.println("char: " + charValue);
            System.out.println("long: " + longValue);
            System.out.println("float: " + floatValue);
            System.out.println("short: " + shortValue);
            System.out.println("byte: " + byteValue);
            System.out.println("string: " + stringValue);

        } catch (EOFException e) {
            System.err.println("End of file reached");
        } catch (IOException e) {
            System.err.println("Error reading: " + e.getMessage());
        }

        // Example: Writing and reading arrays
        writeAndReadArray();
    }

    // Demonstrate writing and reading arrays
    private static void writeAndReadArray() {
        String fileName = "array_data.bin";
        int[] numbers = {10, 20, 30, 40, 50};

        System.out.println("\\n=== Array Example ===");

        // Write array
        try (DataOutputStream dos = new DataOutputStream(
                new FileOutputStream(fileName))) {

            // Write array length first
            dos.writeInt(numbers.length);

            // Write each element
            for (int num : numbers) {
                dos.writeInt(num);
            }

            System.out.println("Array written to file");

        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
        }

        // Read array
        try (DataInputStream dis = new DataInputStream(
                new FileInputStream(fileName))) {

            // Read array length first
            int length = dis.readInt();
            int[] readNumbers = new int[length];

            // Read each element
            for (int i = 0; i < length; i++) {
                readNumbers[i] = dis.readInt();
            }

            // Display array
            System.out.print("Array read from file: ");
            for (int num : readNumbers) {
                System.out.print(num + " ");
            }
            System.out.println();

        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
        }
    }
}`,
    hint1: `DataOutputStream methods follow the pattern write<Type>() like writeInt(), writeDouble(), etc. DataInputStream has corresponding read<Type>() methods.`,
    hint2: `Always read data in the same order it was written. Use writeUTF() and readUTF() for strings, not the regular write/read methods.`,
    whyItMatters: `DataInputStream and DataOutputStream are essential for binary data serialization. They're commonly used for network protocols, file formats, and saving structured data efficiently. They ensure platform-independent data representation.

**Production Pattern:**

\`\`\`java
// Saving game data in binary format
public void saveGameState(String filename, GameState state) {
    try (DataOutputStream dos = new DataOutputStream(
            new BufferedOutputStream(new FileOutputStream(filename)))) {

        // File header
        dos.writeInt(0x47414D45); // Magic number "GAME"
        dos.writeInt(1); // Format version

        // Game data
        dos.writeUTF(state.playerName);
        dos.writeInt(state.level);
        dos.writeLong(state.score);
        dos.writeDouble(state.health);
        dos.writeBoolean(state.hasKey);

    } catch (IOException e) {
        throw new RuntimeException("Save error", e);
    }
}

// Reading network protocol
public Message readNetworkMessage(InputStream in) throws IOException {
    DataInputStream dis = new DataInputStream(in);

    int messageType = dis.readInt();
    int messageLength = dis.readInt();
    String payload = dis.readUTF();

    return new Message(messageType, payload);
}
\`\`\`

**Practical Benefits:**

1. **Binary efficiency**: Compact data storage without text overhead
2. **Platform independence**: One format works on all platforms (big-endian)
3. **Type safety**: Strict typing when reading/writing primitives
4. **Performance**: Faster than text formats (JSON, XML) for primitive types`,
    order: 3,
    testCode: `import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;
import java.io.*;
import java.nio.file.Path;

// Test 1: DataOutputStream writes int correctly
class Test1 {
    @Test
    void testDataOutputStreamWritesInt() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(baos);
        dos.writeInt(42);
        dos.close();
        assertEquals(4, baos.size());
    }
}

// Test 2: DataInputStream reads int correctly
class Test2 {
    @Test
    void testDataInputStreamReadsInt() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(baos);
        dos.writeInt(42);
        dos.close();

        DataInputStream dis = new DataInputStream(new ByteArrayInputStream(baos.toByteArray()));
        assertEquals(42, dis.readInt());
        dis.close();
    }
}

// Test 3: DataOutputStream writes double correctly
class Test3 {
    @Test
    void testDataOutputStreamWritesDouble() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(baos);
        dos.writeDouble(3.14159);
        dos.close();
        assertEquals(8, baos.size());
    }
}

// Test 4: DataInputStream reads double correctly
class Test4 {
    @Test
    void testDataInputStreamReadsDouble() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(baos);
        dos.writeDouble(3.14159);
        dos.close();

        DataInputStream dis = new DataInputStream(new ByteArrayInputStream(baos.toByteArray()));
        assertEquals(3.14159, dis.readDouble(), 0.00001);
        dis.close();
    }
}

// Test 5: DataOutputStream writeUTF works
class Test5 {
    @Test
    void testDataOutputStreamWriteUTF() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(baos);
        dos.writeUTF("Hello");
        dos.close();
        assertTrue(baos.size() > 5);
    }
}

// Test 6: DataInputStream readUTF works
class Test6 {
    @Test
    void testDataInputStreamReadUTF() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(baos);
        dos.writeUTF("Hello World");
        dos.close();

        DataInputStream dis = new DataInputStream(new ByteArrayInputStream(baos.toByteArray()));
        assertEquals("Hello World", dis.readUTF());
        dis.close();
    }
}

// Test 7: Multiple types can be written and read
class Test7 {
    @Test
    void testMultipleTypes() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(baos);
        dos.writeInt(100);
        dos.writeBoolean(true);
        dos.writeChar('A');
        dos.close();

        DataInputStream dis = new DataInputStream(new ByteArrayInputStream(baos.toByteArray()));
        assertEquals(100, dis.readInt());
        assertTrue(dis.readBoolean());
        assertEquals('A', dis.readChar());
        dis.close();
    }
}

// Test 8: DataOutputStream writes boolean
class Test8 {
    @Test
    void testDataOutputStreamWritesBoolean() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(baos);
        dos.writeBoolean(true);
        dos.writeBoolean(false);
        dos.close();

        DataInputStream dis = new DataInputStream(new ByteArrayInputStream(baos.toByteArray()));
        assertTrue(dis.readBoolean());
        assertFalse(dis.readBoolean());
        dis.close();
    }
}

// Test 9: DataOutputStream writes long
class Test9 {
    @Test
    void testDataOutputStreamWritesLong() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(baos);
        dos.writeLong(1234567890123L);
        dos.close();

        DataInputStream dis = new DataInputStream(new ByteArrayInputStream(baos.toByteArray()));
        assertEquals(1234567890123L, dis.readLong());
        dis.close();
    }
}

// Test 10: Read/write with files
class Test10 {
    @TempDir
    Path tempDir;

    @Test
    void testReadWriteWithFiles() throws IOException {
        File file = tempDir.resolve("data.bin").toFile();

        try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(file))) {
            dos.writeInt(42);
            dos.writeDouble(3.14);
            dos.writeUTF("Test");
        }

        try (DataInputStream dis = new DataInputStream(new FileInputStream(file))) {
            assertEquals(42, dis.readInt());
            assertEquals(3.14, dis.readDouble(), 0.01);
            assertEquals("Test", dis.readUTF());
        }
    }
}`,
    translations: {
        ru: {
            title: 'Потоки данных для примитивных типов',
            solutionCode: `import java.io.*;

public class DataStreams {
    public static void main(String[] args) {
        String fileName = "data.bin";

        // Запись примитивных типов с использованием DataOutputStream
        try (FileOutputStream fos = new FileOutputStream(fileName);
             DataOutputStream dos = new DataOutputStream(fos)) {

            // Записываем различные примитивные типы
            dos.writeInt(42);
            dos.writeDouble(3.14159);
            dos.writeBoolean(true);
            dos.writeChar('A');
            dos.writeLong(1234567890L);
            dos.writeFloat(2.5f);
            dos.writeShort((short) 100);
            dos.writeByte(127);

            // Записываем строку в формате UTF
            dos.writeUTF("Hello DataStreams");

            System.out.println("Примитивные данные успешно записаны в файл");

        } catch (IOException e) {
            System.err.println("Ошибка записи: " + e.getMessage());
        }

        // Чтение примитивных типов с использованием DataInputStream
        try (FileInputStream fis = new FileInputStream(fileName);
             DataInputStream dis = new DataInputStream(fis)) {

            System.out.println("\\nЧтение примитивных данных из файла:");

            // Читаем в том же порядке, что и записывали
            int intValue = dis.readInt();
            double doubleValue = dis.readDouble();
            boolean boolValue = dis.readBoolean();
            char charValue = dis.readChar();
            long longValue = dis.readLong();
            float floatValue = dis.readFloat();
            short shortValue = dis.readShort();
            byte byteValue = dis.readByte();
            String stringValue = dis.readUTF();

            // Отображаем все значения
            System.out.println("int: " + intValue);
            System.out.println("double: " + doubleValue);
            System.out.println("boolean: " + boolValue);
            System.out.println("char: " + charValue);
            System.out.println("long: " + longValue);
            System.out.println("float: " + floatValue);
            System.out.println("short: " + shortValue);
            System.out.println("byte: " + byteValue);
            System.out.println("string: " + stringValue);

        } catch (EOFException e) {
            System.err.println("Достигнут конец файла");
        } catch (IOException e) {
            System.err.println("Ошибка чтения: " + e.getMessage());
        }

        // Пример: запись и чтение массивов
        writeAndReadArray();
    }

    // Демонстрация записи и чтения массивов
    private static void writeAndReadArray() {
        String fileName = "array_data.bin";
        int[] numbers = {10, 20, 30, 40, 50};

        System.out.println("\\n=== Пример с массивом ===");

        // Запись массива
        try (DataOutputStream dos = new DataOutputStream(
                new FileOutputStream(fileName))) {

            // Сначала записываем длину массива
            dos.writeInt(numbers.length);

            // Записываем каждый элемент
            for (int num : numbers) {
                dos.writeInt(num);
            }

            System.out.println("Массив записан в файл");

        } catch (IOException e) {
            System.err.println("Ошибка: " + e.getMessage());
        }

        // Чтение массива
        try (DataInputStream dis = new DataInputStream(
                new FileInputStream(fileName))) {

            // Сначала читаем длину массива
            int length = dis.readInt();
            int[] readNumbers = new int[length];

            // Читаем каждый элемент
            for (int i = 0; i < length; i++) {
                readNumbers[i] = dis.readInt();
            }

            // Отображаем массив
            System.out.print("Массив, прочитанный из файла: ");
            for (int num : readNumbers) {
                System.out.print(num + " ");
            }
            System.out.println();

        } catch (IOException e) {
            System.err.println("Ошибка: " + e.getMessage());
        }
    }
}`,
            description: `Научитесь читать и записывать примитивные типы данных с помощью DataInputStream и DataOutputStream.

**Требования:**
1. Создайте DataOutputStream для записи примитивных типов в файл
2. Запишите различные примитивы: int, double, boolean, char, long
3. Запишите строку UTF, используя writeUTF()
4. Закройте выходной поток
5. Прочитайте данные обратно, используя DataInputStream в том же порядке
6. Отобразите все значения, прочитанные из файла

DataInputStream и DataOutputStream позволяют читать и записывать примитивные типы Java машинно-независимым способом.`,
            hint1: `Методы DataOutputStream следуют шаблону write<Тип>(), например writeInt(), writeDouble() и т. д. DataInputStream имеет соответствующие методы read<Тип>().`,
            hint2: `Всегда читайте данные в том же порядке, в котором они были записаны. Используйте writeUTF() и readUTF() для строк, а не обычные методы write/read.`,
            whyItMatters: `DataInputStream и DataOutputStream необходимы для сериализации двоичных данных. Они обычно используются для сетевых протоколов, форматов файлов и эффективного сохранения структурированных данных. Они обеспечивают платформонезависимое представление данных.

**Продакшен паттерн:**

\`\`\`java
// Сохранение игровых данных в бинарный формат
public void saveGameState(String filename, GameState state) {
    try (DataOutputStream dos = new DataOutputStream(
            new BufferedOutputStream(new FileOutputStream(filename)))) {

        // Заголовок файла
        dos.writeInt(0x47414D45); // Магическое число "GAME"
        dos.writeInt(1); // Версия формата

        // Данные игры
        dos.writeUTF(state.playerName);
        dos.writeInt(state.level);
        dos.writeLong(state.score);
        dos.writeDouble(state.health);
        dos.writeBoolean(state.hasKey);

    } catch (IOException e) {
        throw new RuntimeException("Ошибка сохранения", e);
    }
}

// Чтение сетевого протокола
public Message readNetworkMessage(InputStream in) throws IOException {
    DataInputStream dis = new DataInputStream(in);

    int messageType = dis.readInt();
    int messageLength = dis.readInt();
    String payload = dis.readUTF();

    return new Message(messageType, payload);
}
\`\`\`

**Практические преимущества:**

1. **Бинарная эффективность**: Компактное хранение данных без текстового overhead
2. **Платформонезависимость**: Один формат работает на всех платформах (big-endian)
3. **Типобезопасность**: Строгая типизация при чтении/записи примитивов
4. **Производительность**: Быстрее текстовых форматов (JSON, XML) для примитивных типов`
        },
        uz: {
            title: 'Primitiv turlar uchun ma\'lumot oqimlari',
            solutionCode: `import java.io.*;

public class DataStreams {
    public static void main(String[] args) {
        String fileName = "data.bin";

        // DataOutputStream yordamida primitiv turlarni yozish
        try (FileOutputStream fos = new FileOutputStream(fileName);
             DataOutputStream dos = new DataOutputStream(fos)) {

            // Turli primitiv turlarni yozamiz
            dos.writeInt(42);
            dos.writeDouble(3.14159);
            dos.writeBoolean(true);
            dos.writeChar('A');
            dos.writeLong(1234567890L);
            dos.writeFloat(2.5f);
            dos.writeShort((short) 100);
            dos.writeByte(127);

            // UTF formatida stringni yozamiz
            dos.writeUTF("Hello DataStreams");

            System.out.println("Primitiv ma'lumotlar faylga muvaffaqiyatli yozildi");

        } catch (IOException e) {
            System.err.println("Yozishda xato: " + e.getMessage());
        }

        // DataInputStream yordamida primitiv turlarni o'qish
        try (FileInputStream fis = new FileInputStream(fileName);
             DataInputStream dis = new DataInputStream(fis)) {

            System.out.println("\\nFayldan primitiv ma'lumotlarni o'qish:");

            // Yozilgan tartibda o'qiymiz
            int intValue = dis.readInt();
            double doubleValue = dis.readDouble();
            boolean boolValue = dis.readBoolean();
            char charValue = dis.readChar();
            long longValue = dis.readLong();
            float floatValue = dis.readFloat();
            short shortValue = dis.readShort();
            byte byteValue = dis.readByte();
            String stringValue = dis.readUTF();

            // Barcha qiymatlarni ko'rsatamiz
            System.out.println("int: " + intValue);
            System.out.println("double: " + doubleValue);
            System.out.println("boolean: " + boolValue);
            System.out.println("char: " + charValue);
            System.out.println("long: " + longValue);
            System.out.println("float: " + floatValue);
            System.out.println("short: " + shortValue);
            System.out.println("byte: " + byteValue);
            System.out.println("string: " + stringValue);

        } catch (EOFException e) {
            System.err.println("Fayl oxiriga yetildi");
        } catch (IOException e) {
            System.err.println("O'qishda xato: " + e.getMessage());
        }

        // Misol: massivlarni yozish va o'qish
        writeAndReadArray();
    }

    // Massivlarni yozish va o'qishni ko'rsatish
    private static void writeAndReadArray() {
        String fileName = "array_data.bin";
        int[] numbers = {10, 20, 30, 40, 50};

        System.out.println("\\n=== Massiv misoli ===");

        // Massivni yozish
        try (DataOutputStream dos = new DataOutputStream(
                new FileOutputStream(fileName))) {

            // Avval massiv uzunligini yozamiz
            dos.writeInt(numbers.length);

            // Har bir elementni yozamiz
            for (int num : numbers) {
                dos.writeInt(num);
            }

            System.out.println("Massiv faylga yozildi");

        } catch (IOException e) {
            System.err.println("Xato: " + e.getMessage());
        }

        // Massivni o'qish
        try (DataInputStream dis = new DataInputStream(
                new FileInputStream(fileName))) {

            // Avval massiv uzunligini o'qiymiz
            int length = dis.readInt();
            int[] readNumbers = new int[length];

            // Har bir elementni o'qiymiz
            for (int i = 0; i < length; i++) {
                readNumbers[i] = dis.readInt();
            }

            // Massivni ko'rsatamiz
            System.out.print("Fayldan o'qilgan massiv: ");
            for (int num : readNumbers) {
                System.out.print(num + " ");
            }
            System.out.println();

        } catch (IOException e) {
            System.err.println("Xato: " + e.getMessage());
        }
    }
}`,
            description: `DataInputStream va DataOutputStream yordamida primitiv ma'lumot turlarini o'qish va yozishni o'rganing.

**Talablar:**
1. Faylga primitiv turlarni yozish uchun DataOutputStream yarating
2. Turli primitivlarni yozing: int, double, boolean, char, long
3. writeUTF() yordamida UTF stringini yozing
4. Chiqish oqimini yoping
5. DataInputStream yordamida ma'lumotlarni bir xil tartibda qaytadan o'qing
6. Fayldan o'qilgan barcha qiymatlarni ko'rsating

DataInputStream va DataOutputStream Java primitiv turlarini mashinadan mustaqil ravishda o'qish va yozish imkonini beradi.`,
            hint1: `DataOutputStream metodlari write<Tur>() namunasiga amal qiladi, masalan writeInt(), writeDouble() va boshqalar. DataInputStream mos keluvchi read<Tur>() metodlariga ega.`,
            hint2: `Har doim ma'lumotlarni yozilgan tartibda o'qing. Stringlar uchun oddiy write/read metodlar emas, writeUTF() va readUTF() dan foydalaning.`,
            whyItMatters: `DataInputStream va DataOutputStream ikkilik ma'lumotlarni serializatsiya qilish uchun zarurdir. Ular odatda tarmoq protokollari, fayl formatlari va tuzilgan ma'lumotlarni samarali saqlash uchun ishlatiladi. Ular platformadan mustaqil ma'lumot ko'rinishini ta'minlaydi.

**Ishlab chiqarish patterni:**

\`\`\`java
// O'yin ma'lumotlarini binar formatda saqlash
public void saveGameState(String filename, GameState state) {
    try (DataOutputStream dos = new DataOutputStream(
            new BufferedOutputStream(new FileOutputStream(filename)))) {

        // Fayl sarlavhasi
        dos.writeInt(0x47414D45); // Sehrli raqam "GAME"
        dos.writeInt(1); // Format versiyasi

        // O'yin ma'lumotlari
        dos.writeUTF(state.playerName);
        dos.writeInt(state.level);
        dos.writeLong(state.score);
        dos.writeDouble(state.health);
        dos.writeBoolean(state.hasKey);

    } catch (IOException e) {
        throw new RuntimeException("Saqlashda xato", e);
    }
}

// Tarmoq protokolini o'qish
public Message readNetworkMessage(InputStream in) throws IOException {
    DataInputStream dis = new DataInputStream(in);

    int messageType = dis.readInt();
    int messageLength = dis.readInt();
    String payload = dis.readUTF();

    return new Message(messageType, payload);
}
\`\`\`

**Amaliy foydalari:**

1. **Binar samaradorlik**: Matnli overhead siz ixcham ma'lumot saqlash
2. **Platformadan mustaqillik**: Bitta format barcha platformalarda ishlaydi (big-endian)
3. **Tip xavfsizligi**: Primitivlarni o'qish/yozishda qattiq tiplash
4. **Unumdorlik**: Primitiv turlar uchun matnli formatlardan (JSON, XML) tezroq`
        }
    }
};

export default task;
