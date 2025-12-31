import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-yaml-config',
    title: 'YAML Configuration',
    difficulty: 'medium',
    tags: ['java', 'yaml', 'configuration', 'snakeyaml', 'parsing'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `# YAML Configuration with SnakeYAML

YAML (YAML Ain't Markup Language) is a human-readable data serialization format commonly used for configuration files. SnakeYAML is a popular Java library for parsing and generating YAML documents, offering more structure and readability than traditional properties files.

## Requirements:
1. Create a YAML configuration file structure:
   1.1. Nested configuration sections (database, server, logging)
   1.2. Lists and arrays (allowed hosts, features)
   1.3. Multiple data types (strings, numbers, booleans)

2. Parse YAML using SnakeYAML:
   2.1. Use Yaml class to load configuration
   2.2. Handle nested maps and lists
   2.3. Type-safe value extraction

3. Access nested configuration values:
   3.1. Navigate through nested maps
   3.2. Extract values from lists
   3.3. Handle missing keys gracefully

4. Demonstrate complex configuration:
   4.1. Database connection pool settings
   4.2. Server configuration with multiple properties
   4.3. Feature flags and lists

Note: In practice, you would add SnakeYAML dependency (org.yaml:snakeyaml:2.0). For this exercise, focus on understanding the structure and usage pattern.

## Example YAML (config.yaml):
\`\`\`yaml
database:
  host: localhost
  port: 3306
  name: myapp
  username: root
  password: secret
  pool:
    minSize: 5
    maxSize: 20

server:
  host: 0.0.0.0
  port: 8080
  ssl: true
  allowedHosts:
    - localhost
    - example.com
    - api.example.com

logging:
  level: INFO
  file: app.log

features:
  authentication: true
  caching: true
  analytics: false
\`\`\`

## Example Output:
\`\`\`
=== YAML Configuration ===
Database Host: localhost
Database Port: 3306
Database Name: myapp
Pool Min Size: 5
Pool Max Size: 20

Server Host: 0.0.0.0
Server Port: 8080
SSL Enabled: true

Allowed Hosts:
  - localhost
  - example.com
  - api.example.com

Logging Level: INFO
Logging File: app.log

Features:
  authentication: true
  caching: true
  analytics: false
\`\`\``,
    initialCode: `import org.yaml.snakeyaml.Yaml;
import java.io.*;
import java.util.*;

public class YamlConfigExample {
    public static void main(String[] args) {
        // TODO: Create YAML configuration content

        // TODO: Parse YAML configuration

        // TODO: Access nested configuration values

        // TODO: Display configuration
    }
}`,
    solutionCode: `import org.yaml.snakeyaml.Yaml;
import java.io.*;
import java.util.*;

public class YamlConfigExample {
    public static void main(String[] args) {
        // Create YAML content
        String yamlContent = """
                database:
                  host: localhost
                  port: 3306
                  name: myapp
                  username: root
                  password: secret
                  pool:
                    minSize: 5
                    maxSize: 20

                server:
                  host: 0.0.0.0
                  port: 8080
                  ssl: true
                  allowedHosts:
                    - localhost
                    - example.com
                    - api.example.com

                logging:
                  level: INFO
                  file: app.log

                features:
                  authentication: true
                  caching: true
                  analytics: false
                """;

        // Parse YAML
        Yaml yaml = new Yaml();
        Map<String, Object> config = yaml.load(yamlContent);

        System.out.println("=== YAML Configuration ===");

        // Access database configuration
        Map<String, Object> database = (Map<String, Object>) config.get("database");
        System.out.println("Database Host: " + database.get("host"));
        System.out.println("Database Port: " + database.get("port"));
        System.out.println("Database Name: " + database.get("name"));

        // Access nested pool configuration
        Map<String, Object> pool = (Map<String, Object>) database.get("pool");
        System.out.println("Pool Min Size: " + pool.get("minSize"));
        System.out.println("Pool Max Size: " + pool.get("maxSize"));
        System.out.println();

        // Access server configuration
        Map<String, Object> server = (Map<String, Object>) config.get("server");
        System.out.println("Server Host: " + server.get("host"));
        System.out.println("Server Port: " + server.get("port"));
        System.out.println("SSL Enabled: " + server.get("ssl"));
        System.out.println();

        // Access list of allowed hosts
        System.out.println("Allowed Hosts:");
        List<String> allowedHosts = (List<String>) server.get("allowedHosts");
        for (String host : allowedHosts) {
            System.out.println("  - " + host);
        }
        System.out.println();

        // Access logging configuration
        Map<String, Object> logging = (Map<String, Object>) config.get("logging");
        System.out.println("Logging Level: " + logging.get("level"));
        System.out.println("Logging File: " + logging.get("file"));
        System.out.println();

        // Access feature flags
        System.out.println("Features:");
        Map<String, Object> features = (Map<String, Object>) config.get("features");
        features.forEach((key, value) ->
            System.out.println("  " + key + ": " + value)
        );
    }
}`,
    hint1: `Use the Yaml class from SnakeYAML library with load() method to parse YAML content into a Map structure. Cast the result to Map<String, Object> to access nested values.`,
    hint2: `For nested configuration, cast inner objects to Map again. For lists, cast to List<String>. Always check for null values when accessing nested keys.`,
    whyItMatters: `YAML has become the de facto standard for modern application configuration due to its readability and support for complex data structures. It's widely used in containerization (Docker, Kubernetes), CI/CD pipelines, and modern frameworks like Spring Boot. Understanding YAML configuration is essential for working with contemporary Java applications and cloud-native architectures. The hierarchical structure and support for lists make it superior to flat properties files for complex configurations.

**Production Pattern:**
\`\`\`java
// Parsing YAML configuration
Yaml yaml = new Yaml();
Map<String, Object> config = yaml.load(new FileInputStream("config.yaml"));

// Accessing nested configuration
Map<String, Object> database = (Map<String, Object>) config.get("database");
String dbHost = (String) database.get("host");
int dbPort = (Integer) database.get("port");

// Working with lists
Map<String, Object> server = (Map<String, Object>) config.get("server");
List<String> allowedHosts = (List<String>) server.get("allowedHosts");
\`\`\`

**Practical Benefits:**
- Support for nested structures and lists without additional parsing
- Standard for Kubernetes, Docker Compose, and Spring Boot
- More readable format for complex configurations compared to properties`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import org.yaml.snakeyaml.Yaml;
import java.io.*;
import java.util.*;

// Test1: Verify Yaml object creation
class Test1 {
    @Test
    public void test() {
        Yaml yaml = new Yaml();
        assertNotNull(yaml);
    }
}

// Test2: Verify parsing simple YAML string
class Test2 {
    @Test
    public void test() {
        Yaml yaml = new Yaml();
        String yamlStr = "key: value";
        Map<String, Object> data = yaml.load(yamlStr);
        assertEquals("value", data.get("key"));
    }
}

// Test3: Verify parsing nested YAML structure
class Test3 {
    @Test
    public void test() {
        Yaml yaml = new Yaml();
        String yamlStr = "database:\\n  url: jdbc:mysql://localhost\\n  port: 3306";
        Map<String, Object> data = yaml.load(yamlStr);
        Map<String, Object> database = (Map<String, Object>) data.get("database");
        assertEquals("jdbc:mysql://localhost", database.get("url"));
        assertEquals(3306, database.get("port"));
    }
}

// Test4: Verify parsing YAML list
class Test4 {
    @Test
    public void test() {
        Yaml yaml = new Yaml();
        String yamlStr = "servers:\\n  - web1\\n  - web2\\n  - web3";
        Map<String, Object> data = yaml.load(yamlStr);
        List<String> servers = (List<String>) data.get("servers");
        assertEquals(3, servers.size());
        assertTrue(servers.contains("web1"));
    }
}

// Test5: Verify dumping object to YAML
class Test5 {
    @Test
    public void test() {
        Yaml yaml = new Yaml();
        Map<String, Object> data = new HashMap<>();
        data.put("name", "TestApp");
        data.put("version", "1.0");

        String yamlStr = yaml.dump(data);
        assertNotNull(yamlStr);
        assertTrue(yamlStr.contains("name"));
        assertTrue(yamlStr.contains("TestApp"));
    }
}

// Test6: Verify loading YAML from file
class Test6 {
    @Test
    public void test() throws IOException {
        String yamlContent = "app:\\n  name: MyApp\\n  port: 8080";
        File tempFile = File.createTempFile("test", ".yaml");

        try (FileWriter writer = new FileWriter(tempFile)) {
            writer.write(yamlContent);
        }

        Yaml yaml = new Yaml();
        try (FileInputStream input = new FileInputStream(tempFile)) {
            Map<String, Object> data = yaml.load(input);
            Map<String, Object> app = (Map<String, Object>) data.get("app");
            assertEquals("MyApp", app.get("name"));
        }

        tempFile.delete();
    }
}

// Test7: Verify parsing boolean values
class Test7 {
    @Test
    public void test() {
        Yaml yaml = new Yaml();
        String yamlStr = "debug: true\\nenabled: false";
        Map<String, Object> data = yaml.load(yamlStr);
        assertTrue((Boolean) data.get("debug"));
        assertFalse((Boolean) data.get("enabled"));
    }
}

// Test8: Verify parsing numeric values
class Test8 {
    @Test
    public void test() {
        Yaml yaml = new Yaml();
        String yamlStr = "port: 8080\\ntimeout: 30\\nrate: 1.5";
        Map<String, Object> data = yaml.load(yamlStr);
        assertEquals(8080, data.get("port"));
        assertEquals(30, data.get("timeout"));
    }
}

// Test9: Verify parsing complex nested structure
class Test9 {
    @Test
    public void test() {
        Yaml yaml = new Yaml();
        String yamlStr = "app:\\n  database:\\n    url: jdbc:h2:mem:test\\n    pool:\\n      size: 10";
        Map<String, Object> data = yaml.load(yamlStr);
        Map<String, Object> app = (Map<String, Object>) data.get("app");
        Map<String, Object> database = (Map<String, Object>) app.get("database");
        Map<String, Object> pool = (Map<String, Object>) database.get("pool");
        assertEquals(10, pool.get("size"));
    }
}

// Test10: Verify dumping complex object to YAML
class Test10 {
    @Test
    public void test() {
        Yaml yaml = new Yaml();
        Map<String, Object> config = new HashMap<>();
        Map<String, Object> server = new HashMap<>();
        server.put("host", "localhost");
        server.put("port", 8080);
        config.put("server", server);

        String yamlStr = yaml.dump(config);
        assertTrue(yamlStr.contains("server"));
        assertTrue(yamlStr.contains("localhost"));
        assertTrue(yamlStr.contains("8080"));
    }
}
`,
    order: 2,
    translations: {
        ru: {
            title: 'YAML конфигурация',
            solutionCode: `import org.yaml.snakeyaml.Yaml;
import java.io.*;
import java.util.*;

public class YamlConfigExample {
    public static void main(String[] args) {
        // Создать YAML содержимое
        String yamlContent = """
                database:
                  host: localhost
                  port: 3306
                  name: myapp
                  username: root
                  password: secret
                  pool:
                    minSize: 5
                    maxSize: 20

                server:
                  host: 0.0.0.0
                  port: 8080
                  ssl: true
                  allowedHosts:
                    - localhost
                    - example.com
                    - api.example.com

                logging:
                  level: INFO
                  file: app.log

                features:
                  authentication: true
                  caching: true
                  analytics: false
                """;

        // Разобрать YAML
        Yaml yaml = new Yaml();
        Map<String, Object> config = yaml.load(yamlContent);

        System.out.println("=== YAML конфигурация ===");

        // Получить конфигурацию базы данных
        Map<String, Object> database = (Map<String, Object>) config.get("database");
        System.out.println("Database Host: " + database.get("host"));
        System.out.println("Database Port: " + database.get("port"));
        System.out.println("Database Name: " + database.get("name"));

        // Получить вложенную конфигурацию пула
        Map<String, Object> pool = (Map<String, Object>) database.get("pool");
        System.out.println("Pool Min Size: " + pool.get("minSize"));
        System.out.println("Pool Max Size: " + pool.get("maxSize"));
        System.out.println();

        // Получить конфигурацию сервера
        Map<String, Object> server = (Map<String, Object>) config.get("server");
        System.out.println("Server Host: " + server.get("host"));
        System.out.println("Server Port: " + server.get("port"));
        System.out.println("SSL Enabled: " + server.get("ssl"));
        System.out.println();

        // Получить список разрешенных хостов
        System.out.println("Allowed Hosts:");
        List<String> allowedHosts = (List<String>) server.get("allowedHosts");
        for (String host : allowedHosts) {
            System.out.println("  - " + host);
        }
        System.out.println();

        // Получить конфигурацию логирования
        Map<String, Object> logging = (Map<String, Object>) config.get("logging");
        System.out.println("Logging Level: " + logging.get("level"));
        System.out.println("Logging File: " + logging.get("file"));
        System.out.println();

        // Получить флаги функций
        System.out.println("Features:");
        Map<String, Object> features = (Map<String, Object>) config.get("features");
        features.forEach((key, value) ->
            System.out.println("  " + key + ": " + value)
        );
    }
}`,
            description: `# YAML конфигурация с SnakeYAML

YAML (YAML Ain't Markup Language) - это читаемый человеком формат сериализации данных, обычно используемый для конфигурационных файлов. SnakeYAML - это популярная Java-библиотека для разбора и генерации YAML-документов, предлагающая больше структуры и читаемости, чем традиционные файлы свойств.

## Требования:
1. Создайте структуру конфигурационного файла YAML:
   1.1. Вложенные секции конфигурации (база данных, сервер, логирование)
   1.2. Списки и массивы (разрешенные хосты, функции)
   1.3. Множественные типы данных (строки, числа, логические значения)

2. Разберите YAML используя SnakeYAML:
   2.1. Используйте класс Yaml для загрузки конфигурации
   2.2. Обрабатывайте вложенные карты и списки
   2.3. Типобезопасное извлечение значений

3. Получайте вложенные значения конфигурации:
   3.1. Навигация по вложенным картам
   3.2. Извлечение значений из списков
   3.3. Корректная обработка отсутствующих ключей

4. Продемонстрируйте сложную конфигурацию:
   4.1. Настройки пула подключений к базе данных
   4.2. Конфигурация сервера с множественными свойствами
   4.3. Флаги функций и списки

Примечание: На практике вы бы добавили зависимость SnakeYAML (org.yaml:snakeyaml:2.0). Для этого упражнения сосредоточьтесь на понимании структуры и шаблона использования.

## Пример YAML (config.yaml):
\`\`\`yaml
database:
  host: localhost
  port: 3306
  name: myapp
  username: root
  password: secret
  pool:
    minSize: 5
    maxSize: 20

server:
  host: 0.0.0.0
  port: 8080
  ssl: true
  allowedHosts:
    - localhost
    - example.com
    - api.example.com

logging:
  level: INFO
  file: app.log

features:
  authentication: true
  caching: true
  analytics: false
\`\`\`

## Пример вывода:
\`\`\`
=== YAML Configuration ===
Database Host: localhost
Database Port: 3306
Database Name: myapp
Pool Min Size: 5
Pool Max Size: 20

Server Host: 0.0.0.0
Server Port: 8080
SSL Enabled: true

Allowed Hosts:
  - localhost
  - example.com
  - api.example.com

Logging Level: INFO
Logging File: app.log

Features:
  authentication: true
  caching: true
  analytics: false
\`\`\``,
            hint1: `Используйте класс Yaml из библиотеки SnakeYAML с методом load() для разбора YAML содержимого в структуру Map. Приведите результат к Map<String, Object> для доступа к вложенным значениям.`,
            hint2: `Для вложенной конфигурации снова приведите внутренние объекты к Map. Для списков приведите к List<String>. Всегда проверяйте null значения при доступе к вложенным ключам.`,
            whyItMatters: `YAML стал де-факто стандартом для конфигурации современных приложений благодаря своей читаемости и поддержке сложных структур данных. Он широко используется в контейнеризации (Docker, Kubernetes), CI/CD конвейерах и современных фреймворках, таких как Spring Boot. Понимание YAML конфигурации необходимо для работы с современными Java-приложениями и облачно-ориентированными архитектурами. Иерархическая структура и поддержка списков делают его превосходящим плоские файлы свойств для сложных конфигураций.

**Продакшен паттерн:**
\`\`\`java
// Разбор YAML конфигурации
Yaml yaml = new Yaml();
Map<String, Object> config = yaml.load(new FileInputStream("config.yaml"));

// Доступ к вложенной конфигурации
Map<String, Object> database = (Map<String, Object>) config.get("database");
String dbHost = (String) database.get("host");
int dbPort = (Integer) database.get("port");

// Работа со списками
Map<String, Object> server = (Map<String, Object>) config.get("server");
List<String> allowedHosts = (List<String>) server.get("allowedHosts");
\`\`\`

**Практические преимущества:**
- Поддержка вложенных структур и списков без дополнительного парсинга
- Стандарт для Kubernetes, Docker Compose и Spring Boot
- Более читаемый формат для сложных конфигураций по сравнению с properties`
        },
        uz: {
            title: `YAML konfiguratsiyasi`,
            solutionCode: `import org.yaml.snakeyaml.Yaml;
import java.io.*;
import java.util.*;

public class YamlConfigExample {
    public static void main(String[] args) {
        // YAML mazmunini yaratish
        String yamlContent = """
                database:
                  host: localhost
                  port: 3306
                  name: myapp
                  username: root
                  password: secret
                  pool:
                    minSize: 5
                    maxSize: 20

                server:
                  host: 0.0.0.0
                  port: 8080
                  ssl: true
                  allowedHosts:
                    - localhost
                    - example.com
                    - api.example.com

                logging:
                  level: INFO
                  file: app.log

                features:
                  authentication: true
                  caching: true
                  analytics: false
                """;

        // YAML ni tahlil qilish
        Yaml yaml = new Yaml();
        Map<String, Object> config = yaml.load(yamlContent);

        System.out.println("=== YAML konfiguratsiyasi ===");

        // Ma'lumotlar bazasi konfiguratsiyasiga kirish
        Map<String, Object> database = (Map<String, Object>) config.get("database");
        System.out.println("Database Host: " + database.get("host"));
        System.out.println("Database Port: " + database.get("port"));
        System.out.println("Database Name: " + database.get("name"));

        // Ichki pool konfiguratsiyasiga kirish
        Map<String, Object> pool = (Map<String, Object>) database.get("pool");
        System.out.println("Pool Min Size: " + pool.get("minSize"));
        System.out.println("Pool Max Size: " + pool.get("maxSize"));
        System.out.println();

        // Server konfiguratsiyasiga kirish
        Map<String, Object> server = (Map<String, Object>) config.get("server");
        System.out.println("Server Host: " + server.get("host"));
        System.out.println("Server Port: " + server.get("port"));
        System.out.println("SSL Enabled: " + server.get("ssl"));
        System.out.println();

        // Ruxsat etilgan hostlar ro'yxatiga kirish
        System.out.println("Allowed Hosts:");
        List<String> allowedHosts = (List<String>) server.get("allowedHosts");
        for (String host : allowedHosts) {
            System.out.println("  - " + host);
        }
        System.out.println();

        // Logging konfiguratsiyasiga kirish
        Map<String, Object> logging = (Map<String, Object>) config.get("logging");
        System.out.println("Logging Level: " + logging.get("level"));
        System.out.println("Logging File: " + logging.get("file"));
        System.out.println();

        // Funksiya bayroqlariga kirish
        System.out.println("Features:");
        Map<String, Object> features = (Map<String, Object>) config.get("features");
        features.forEach((key, value) ->
            System.out.println("  " + key + ": " + value)
        );
    }
}`,
            description: `# SnakeYAML bilan YAML konfiguratsiyasi

YAML (YAML Ain't Markup Language) - bu odatda konfiguratsiya fayllari uchun ishlatiladigan, inson o'qiy oladigan ma'lumotlar seriyalash formatidir. SnakeYAML - bu YAML hujjatlarini tahlil qilish va yaratish uchun mashhur Java kutubxonasi bo'lib, an'anaviy xususiyat fayllariga qaraganda ko'proq tuzilma va o'qilishni taklif qiladi.

## Talablar:
1. YAML konfiguratsiya fayl tuzilmasini yarating:
   1.1. Ichki konfiguratsiya bo'limlari (ma'lumotlar bazasi, server, logging)
   1.2. Ro'yxatlar va massivlar (ruxsat etilgan hostlar, funksiyalar)
   1.3. Bir nechta ma'lumot turlari (satrlar, raqamlar, mantiqiy qiymatlar)

2. SnakeYAML yordamida YAML ni tahlil qiling:
   2.1. Konfiguratsiyani yuklash uchun Yaml klassidan foydalaning
   2.2. Ichki maplar va ro'yxatlarni boshqaring
   2.3. Tip-xavfsiz qiymat ajratib olish

3. Ichki konfiguratsiya qiymatlariga kiring:
   3.1. Ichki maplar orqali navigatsiya qiling
   3.2. Ro'yxatlardan qiymatlarni ajratib oling
   3.3. Yo'qolgan kalitlarni to'g'ri boshqaring

4. Murakkab konfiguratsiyani namoyish eting:
   4.1. Ma'lumotlar bazasi ulanish pool sozlamalari
   4.2. Ko'p xususiyatli server konfiguratsiyasi
   4.3. Funksiya bayroqlari va ro'yxatlar

Eslatma: Amalda siz SnakeYAML bog'liqligini qo'shgan bo'lardingiz (org.yaml:snakeyaml:2.0). Ushbu mashq uchun tuzilma va foydalanish namunasini tushunishga e'tibor bering.

## YAML namunasi (config.yaml):
\`\`\`yaml
database:
  host: localhost
  port: 3306
  name: myapp
  username: root
  password: secret
  pool:
    minSize: 5
    maxSize: 20

server:
  host: 0.0.0.0
  port: 8080
  ssl: true
  allowedHosts:
    - localhost
    - example.com
    - api.example.com

logging:
  level: INFO
  file: app.log

features:
  authentication: true
  caching: true
  analytics: false
\`\`\`

## Chiqish namunasi:
\`\`\`
=== YAML Configuration ===
Database Host: localhost
Database Port: 3306
Database Name: myapp
Pool Min Size: 5
Pool Max Size: 20

Server Host: 0.0.0.0
Server Port: 8080
SSL Enabled: true

Allowed Hosts:
  - localhost
  - example.com
  - api.example.com

Logging Level: INFO
Logging File: app.log

Features:
  authentication: true
  caching: true
  analytics: false
\`\`\``,
            hint1: `YAML mazmunini Map tuzilmasiga tahlil qilish uchun SnakeYAML kutubxonasidan Yaml klassini load() metodi bilan ishlating. Ichki qiymatlarga kirish uchun natijani Map<String, Object> ga o'zgartiring.`,
            hint2: `Ichki konfiguratsiya uchun ichki obyektlarni yana Map ga o'zgartiring. Ro'yxatlar uchun List<String> ga o'zgartiring. Ichki kalitlarga kirishda har doim null qiymatlarini tekshiring.`,
            whyItMatters: `YAML o'qilishi va murakkab ma'lumotlar tuzilmalarini qo'llab-quvvatlashi tufayli zamonaviy ilova konfiguratsiyasi uchun de-fakto standart bo'ldi. U konteynerizatsiyada (Docker, Kubernetes), CI/CD quvurlarida va Spring Boot kabi zamonaviy freymvorklarda keng qo'llaniladi. YAML konfiguratsiyasini tushunish zamonaviy Java ilovalari va cloud-native arxitekturalar bilan ishlash uchun zarur. Ierarxik tuzilma va ro'yxatlarni qo'llab-quvvatlash uni murakkab konfiguratsiyalar uchun tekis xususiyat fayllaridan ustun qiladi.

**Ishlab chiqarish patterni:**
\`\`\`java
// YAML konfiguratsiyasini tahlil qilish
Yaml yaml = new Yaml();
Map<String, Object> config = yaml.load(new FileInputStream("config.yaml"));

// Ichki konfiguratsiyaga kirish
Map<String, Object> database = (Map<String, Object>) config.get("database");
String dbHost = (String) database.get("host");
int dbPort = (Integer) database.get("port");

// Ro'yxatlar bilan ishlash
Map<String, Object> server = (Map<String, Object>) config.get("server");
List<String> allowedHosts = (List<String>) server.get("allowedHosts");
\`\`\`

**Amaliy foydalari:**
- Qo'shimcha parsing kerak bo'lmasdan ichki tuzilmalar va ro'yxatlarni qo'llab-quvvatlash
- Kubernetes, Docker Compose va Spring Boot uchun standart
- Properties bilan solishtirganda murakkab konfiguratsiyalar uchun yanada o'qiluvchi format`
        }
    }
};

export default task;
