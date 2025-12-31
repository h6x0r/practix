import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-bytebuffer',
    title: 'ByteBuffer Fundamentals',
    difficulty: 'medium',
    tags: ['java', 'nio', 'buffer', 'bytebuffer'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn ByteBuffer operations: allocate, put, get, flip, clear, and rewind.

**Requirements:**
1. Allocate a ByteBuffer with capacity of 10 bytes
2. Put some data: put a byte (65), putInt (1234), putChar ('A')
3. Print position, limit, and capacity after each operation
4. Flip the buffer to prepare for reading
5. Read the data back: get byte, getInt, getChar
6. Clear the buffer and demonstrate reuse
7. Use allocateDirect() to create a direct buffer
8. Compare with heap buffer

ByteBuffer is fundamental for low-level I/O operations and provides efficient data manipulation.`,
    initialCode: `import java.nio.ByteBuffer;

public class ByteBufferDemo {
    public static void main(String[] args) {
        // Allocate ByteBuffer with capacity 10

        // Put some data

        // Print position, limit, capacity

        // Flip buffer

        // Read data back

        // Clear and reuse

        // Create direct buffer
    }
}`,
    solutionCode: `import java.nio.ByteBuffer;

public class ByteBufferDemo {
    public static void main(String[] args) {
        // Allocate ByteBuffer with capacity 10
        ByteBuffer buffer = ByteBuffer.allocate(10);
        System.out.println("Initial - Position: " + buffer.position() +
                           ", Limit: " + buffer.limit() +
                           ", Capacity: " + buffer.capacity());

        // Put some data
        buffer.put((byte) 65); // 'A' in ASCII
        System.out.println("After put byte - Position: " + buffer.position());

        buffer.putInt(1234);
        System.out.println("After putInt - Position: " + buffer.position());

        buffer.putChar('A');
        System.out.println("After putChar - Position: " + buffer.position());

        // Flip buffer to prepare for reading
        buffer.flip();
        System.out.println("\\nAfter flip - Position: " + buffer.position() +
                           ", Limit: " + buffer.limit());

        // Read data back
        byte b = buffer.get();
        System.out.println("\\nRead byte: " + b + " (char: " + (char) b + ")");
        System.out.println("Position after get: " + buffer.position());

        int i = buffer.getInt();
        System.out.println("Read int: " + i);
        System.out.println("Position after getInt: " + buffer.position());

        char c = buffer.getChar();
        System.out.println("Read char: " + c);
        System.out.println("Position after getChar: " + buffer.position());

        // Clear and reuse
        buffer.clear();
        System.out.println("\\nAfter clear - Position: " + buffer.position() +
                           ", Limit: " + buffer.limit() +
                           ", Capacity: " + buffer.capacity());

        // Create direct buffer
        ByteBuffer directBuffer = ByteBuffer.allocateDirect(10);
        System.out.println("\\nDirect buffer created");
        System.out.println("Is direct: " + directBuffer.isDirect());
        System.out.println("Heap buffer is direct: " + buffer.isDirect());

        // Demonstrate rewind
        buffer.put((byte) 100);
        buffer.put((byte) 101);
        buffer.flip();
        System.out.println("\\nBefore rewind - Position: " + buffer.position());
        buffer.rewind();
        System.out.println("After rewind - Position: " + buffer.position());
    }
}`,
    hint1: `ByteBuffer has three key properties: position (current index), limit (end of accessible data), and capacity (total size). Use flip() after writing and before reading.`,
    hint2: `clear() resets position to 0 and limit to capacity. rewind() resets position to 0 but keeps limit. allocateDirect() creates off-heap memory buffer for I/O operations.`,
    whyItMatters: `ByteBuffer is the foundation of NIO channels and provides efficient buffer management for I/O operations. Understanding its state transitions (position, limit, capacity) is crucial for working with channels and selectors.

**Production Pattern:**
\`\`\`java
@Component
public class NetworkProtocolHandler {
    private static final int HEADER_SIZE = 8;
    private final ByteBuffer headerBuffer = ByteBuffer.allocateDirect(HEADER_SIZE);

    public Message parseMessage(SocketChannel channel) throws IOException {
        // Read header
        headerBuffer.clear();
        channel.read(headerBuffer);
        headerBuffer.flip();

        int messageType = headerBuffer.getInt();
        int payloadLength = headerBuffer.getInt();

        // Read message body
        ByteBuffer payloadBuffer = ByteBuffer.allocate(payloadLength);
        channel.read(payloadBuffer);
        payloadBuffer.flip();

        return new Message(messageType, payloadBuffer);
    }

    public void sendMessage(SocketChannel channel, Message msg) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(HEADER_SIZE + msg.size());
        buffer.putInt(msg.getType());
        buffer.putInt(msg.size());
        buffer.put(msg.getData());
        buffer.flip();

        channel.write(buffer);
    }
}
\`\`\`

**Practical Benefits:**
- Efficient work with binary protocols
- Minimized memory data copying
- Direct access to native memory for I/O`,
    order: 3,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.nio.ByteBuffer;

// Test1: Test ByteBuffer allocate
class Test1 {
    @Test
    public void test() {
        ByteBuffer buffer = ByteBuffer.allocate(10);
        assertEquals(10, buffer.capacity());
        assertEquals(0, buffer.position());
        assertEquals(10, buffer.limit());
    }
}

// Test2: Test put and position
class Test2 {
    @Test
    public void test() {
        ByteBuffer buffer = ByteBuffer.allocate(10);
        buffer.put((byte) 65);
        assertEquals(1, buffer.position());
    }
}

// Test3: Test putInt
class Test3 {
    @Test
    public void test() {
        ByteBuffer buffer = ByteBuffer.allocate(10);
        buffer.putInt(1234);
        assertEquals(4, buffer.position());
    }
}

// Test4: Test flip
class Test4 {
    @Test
    public void test() {
        ByteBuffer buffer = ByteBuffer.allocate(10);
        buffer.put((byte) 65);
        buffer.flip();
        assertEquals(0, buffer.position());
        assertEquals(1, buffer.limit());
    }
}

// Test5: Test get after flip
class Test5 {
    @Test
    public void test() {
        ByteBuffer buffer = ByteBuffer.allocate(10);
        buffer.put((byte) 65);
        buffer.flip();
        byte b = buffer.get();
        assertEquals(65, b);
    }
}

// Test6: Test clear
class Test6 {
    @Test
    public void test() {
        ByteBuffer buffer = ByteBuffer.allocate(10);
        buffer.put((byte) 65);
        buffer.clear();
        assertEquals(0, buffer.position());
        assertEquals(10, buffer.limit());
    }
}

// Test7: Test allocateDirect
class Test7 {
    @Test
    public void test() {
        ByteBuffer buffer = ByteBuffer.allocateDirect(10);
        assertTrue(buffer.isDirect());
        assertEquals(10, buffer.capacity());
    }
}

// Test8: Test rewind
class Test8 {
    @Test
    public void test() {
        ByteBuffer buffer = ByteBuffer.allocate(10);
        buffer.put((byte) 100);
        buffer.put((byte) 101);
        buffer.rewind();
        assertEquals(0, buffer.position());
    }
}

// Test9: Test putChar and getChar
class Test9 {
    @Test
    public void test() {
        ByteBuffer buffer = ByteBuffer.allocate(10);
        buffer.putChar('A');
        buffer.flip();
        char c = buffer.getChar();
        assertEquals('A', c);
    }
}

// Test10: Test hasRemaining
class Test10 {
    @Test
    public void test() {
        ByteBuffer buffer = ByteBuffer.allocate(10);
        buffer.put((byte) 65);
        buffer.flip();
        assertTrue(buffer.hasRemaining());
        buffer.get();
        assertFalse(buffer.hasRemaining());
    }
}
`,
    translations: {
        ru: {
            title: 'Основы ByteBuffer',
            solutionCode: `import java.nio.ByteBuffer;

public class ByteBufferDemo {
    public static void main(String[] args) {
        // Выделяем ByteBuffer с емкостью 10
        ByteBuffer buffer = ByteBuffer.allocate(10);
        System.out.println("Начальное - Позиция: " + buffer.position() +
                           ", Лимит: " + buffer.limit() +
                           ", Емкость: " + buffer.capacity());

        // Помещаем данные
        buffer.put((byte) 65); // 'A' в ASCII
        System.out.println("После put byte - Позиция: " + buffer.position());

        buffer.putInt(1234);
        System.out.println("После putInt - Позиция: " + buffer.position());

        buffer.putChar('A');
        System.out.println("После putChar - Позиция: " + buffer.position());

        // Переворачиваем буфер для подготовки к чтению
        buffer.flip();
        System.out.println("\\nПосле flip - Позиция: " + buffer.position() +
                           ", Лимит: " + buffer.limit());

        // Читаем данные обратно
        byte b = buffer.get();
        System.out.println("\\nПрочитан byte: " + b + " (символ: " + (char) b + ")");
        System.out.println("Позиция после get: " + buffer.position());

        int i = buffer.getInt();
        System.out.println("Прочитан int: " + i);
        System.out.println("Позиция после getInt: " + buffer.position());

        char c = buffer.getChar();
        System.out.println("Прочитан char: " + c);
        System.out.println("Позиция после getChar: " + buffer.position());

        // Очищаем и переиспользуем
        buffer.clear();
        System.out.println("\\nПосле clear - Позиция: " + buffer.position() +
                           ", Лимит: " + buffer.limit() +
                           ", Емкость: " + buffer.capacity());

        // Создаем прямой буфер
        ByteBuffer directBuffer = ByteBuffer.allocateDirect(10);
        System.out.println("\\nПрямой буфер создан");
        System.out.println("Прямой: " + directBuffer.isDirect());
        System.out.println("Heap буфер прямой: " + buffer.isDirect());

        // Демонстрируем rewind
        buffer.put((byte) 100);
        buffer.put((byte) 101);
        buffer.flip();
        System.out.println("\\nПеред rewind - Позиция: " + buffer.position());
        buffer.rewind();
        System.out.println("После rewind - Позиция: " + buffer.position());
    }
}`,
            description: `Изучите операции ByteBuffer: allocate, put, get, flip, clear и rewind.

**Требования:**
1. Выделите ByteBuffer с емкостью 10 байт
2. Поместите данные: put байт (65), putInt (1234), putChar ('A')
3. Выведите позицию, лимит и емкость после каждой операции
4. Переверните буфер для подготовки к чтению
5. Прочитайте данные обратно: get byte, getInt, getChar
6. Очистите буфер и продемонстрируйте переиспользование
7. Используйте allocateDirect() для создания прямого буфера
8. Сравните с heap буфером

ByteBuffer - основа для низкоуровневых операций ввода-вывода и обеспечивает эффективную манипуляцию данными.`,
            hint1: `ByteBuffer имеет три ключевых свойства: position (текущий индекс), limit (конец доступных данных) и capacity (общий размер). Используйте flip() после записи и перед чтением.`,
            hint2: `clear() сбрасывает position в 0, а limit в capacity. rewind() сбрасывает position в 0, но сохраняет limit. allocateDirect() создает буфер вне кучи для операций ввода-вывода.`,
            whyItMatters: `ByteBuffer - основа каналов NIO и обеспечивает эффективное управление буферами для операций ввода-вывода. Понимание переходов состояния (position, limit, capacity) критически важно для работы с каналами и селекторами.

**Продакшен паттерн:**
\`\`\`java
@Component
public class NetworkProtocolHandler {
    private static final int HEADER_SIZE = 8;
    private final ByteBuffer headerBuffer = ByteBuffer.allocateDirect(HEADER_SIZE);

    public Message parseMessage(SocketChannel channel) throws IOException {
        // Читаем заголовок
        headerBuffer.clear();
        channel.read(headerBuffer);
        headerBuffer.flip();

        int messageType = headerBuffer.getInt();
        int payloadLength = headerBuffer.getInt();

        // Читаем тело сообщения
        ByteBuffer payloadBuffer = ByteBuffer.allocate(payloadLength);
        channel.read(payloadBuffer);
        payloadBuffer.flip();

        return new Message(messageType, payloadBuffer);
    }

    public void sendMessage(SocketChannel channel, Message msg) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(HEADER_SIZE + msg.size());
        buffer.putInt(msg.getType());
        buffer.putInt(msg.size());
        buffer.put(msg.getData());
        buffer.flip();

        channel.write(buffer);
    }
}
\`\`\`

**Практические преимущества:**
- Эффективная работа с бинарными протоколами
- Минимизация копирования данных в памяти
- Прямой доступ к нативной памяти для I/O`
        },
        uz: {
            title: 'ByteBuffer Asoslari',
            solutionCode: `import java.nio.ByteBuffer;

public class ByteBufferDemo {
    public static void main(String[] args) {
        // 10 sig'imli ByteBuffer ajratamiz
        ByteBuffer buffer = ByteBuffer.allocate(10);
        System.out.println("Boshlang'ich - Pozitsiya: " + buffer.position() +
                           ", Limit: " + buffer.limit() +
                           ", Sig'im: " + buffer.capacity());

        // Ma'lumot joylaymiz
        buffer.put((byte) 65); // ASCII da 'A'
        System.out.println("put byte dan keyin - Pozitsiya: " + buffer.position());

        buffer.putInt(1234);
        System.out.println("putInt dan keyin - Pozitsiya: " + buffer.position());

        buffer.putChar('A');
        System.out.println("putChar dan keyin - Pozitsiya: " + buffer.position());

        // O'qish uchun buferni aylantiramiz
        buffer.flip();
        System.out.println("\\nflip dan keyin - Pozitsiya: " + buffer.position() +
                           ", Limit: " + buffer.limit());

        // Ma'lumotni qayta o'qiymiz
        byte b = buffer.get();
        System.out.println("\\nO'qilgan byte: " + b + " (belgi: " + (char) b + ")");
        System.out.println("get dan keyingi pozitsiya: " + buffer.position());

        int i = buffer.getInt();
        System.out.println("O'qilgan int: " + i);
        System.out.println("getInt dan keyingi pozitsiya: " + buffer.position());

        char c = buffer.getChar();
        System.out.println("O'qilgan char: " + c);
        System.out.println("getChar dan keyingi pozitsiya: " + buffer.position());

        // Tozalaymiz va qayta ishlatamiz
        buffer.clear();
        System.out.println("\\nclear dan keyin - Pozitsiya: " + buffer.position() +
                           ", Limit: " + buffer.limit() +
                           ", Sig'im: " + buffer.capacity());

        // To'g'ridan-to'g'ri bufer yaratamiz
        ByteBuffer directBuffer = ByteBuffer.allocateDirect(10);
        System.out.println("\\nTo'g'ridan-to'g'ri bufer yaratildi");
        System.out.println("To'g'ridan-to'g'rimi: " + directBuffer.isDirect());
        System.out.println("Heap bufer to'g'ridan-to'g'rimi: " + buffer.isDirect());

        // rewind ni ko'rsatamiz
        buffer.put((byte) 100);
        buffer.put((byte) 101);
        buffer.flip();
        System.out.println("\\nrewind dan oldin - Pozitsiya: " + buffer.position());
        buffer.rewind();
        System.out.println("rewind dan keyin - Pozitsiya: " + buffer.position());
    }
}`,
            description: `ByteBuffer operatsiyalarini o'rganing: allocate, put, get, flip, clear va rewind.

**Talablar:**
1. 10 bayt sig'imli ByteBuffer ajrating
2. Ma'lumot joylang: put bayt (65), putInt (1234), putChar ('A')
3. Har bir operatsiyadan keyin pozitsiya, limit va sig'imni chiqaring
4. O'qish uchun buferni aylantiring
5. Ma'lumotni qayta o'qing: get byte, getInt, getChar
6. Buferni tozalang va qayta ishlatishni ko'rsating
7. To'g'ridan-to'g'ri bufer yaratish uchun allocateDirect() dan foydalaning
8. Heap bufer bilan solishtiring

ByteBuffer past darajadagi kirish-chiqish operatsiyalari uchun asosiy hisoblanadi va ma'lumotlarni samarali boshqarishni ta'minlaydi.`,
            hint1: `ByteBuffer uchta asosiy xususiyatga ega: position (joriy indeks), limit (mavjud ma'lumotlar oxiri) va capacity (umumiy hajm). Yozishdan keyin va o'qishdan oldin flip() dan foydalaning.`,
            hint2: `clear() positionni 0 ga va limitni capacityga qaytaradi. rewind() positionni 0 ga qaytaradi, lekin limitni saqlaydi. allocateDirect() kirish-chiqish operatsiyalari uchun off-heap xotira buferini yaratadi.`,
            whyItMatters: `ByteBuffer NIO kanallarining asosi bo'lib, kirish-chiqish operatsiyalari uchun samarali bufer boshqaruvini ta'minlaydi. Holat o'tishlarini (position, limit, capacity) tushunish kanallar va selektorlar bilan ishlash uchun juda muhim.

**Ishlab chiqarish patterni:**
\`\`\`java
@Component
public class NetworkProtocolHandler {
    private static final int HEADER_SIZE = 8;
    private final ByteBuffer headerBuffer = ByteBuffer.allocateDirect(HEADER_SIZE);

    public Message parseMessage(SocketChannel channel) throws IOException {
        // Sarlavhani o'qiymiz
        headerBuffer.clear();
        channel.read(headerBuffer);
        headerBuffer.flip();

        int messageType = headerBuffer.getInt();
        int payloadLength = headerBuffer.getInt();

        // Xabar tanasini o'qiymiz
        ByteBuffer payloadBuffer = ByteBuffer.allocate(payloadLength);
        channel.read(payloadBuffer);
        payloadBuffer.flip();

        return new Message(messageType, payloadBuffer);
    }

    public void sendMessage(SocketChannel channel, Message msg) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(HEADER_SIZE + msg.size());
        buffer.putInt(msg.getType());
        buffer.putInt(msg.size());
        buffer.put(msg.getData());
        buffer.flip();

        channel.write(buffer);
    }
}
\`\`\`

**Amaliy foydalari:**
- Ikkilik protokollar bilan samarali ishlash
- Xotirada ma'lumot nusxalashni minimallashtirish
- I/O uchun nativ xotiraga to'g'ridan-to'g'ri kirish`
        }
    }
};

export default task;
