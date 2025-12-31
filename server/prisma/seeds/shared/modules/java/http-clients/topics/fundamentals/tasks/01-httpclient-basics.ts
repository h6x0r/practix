import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-httpclient-basics',
    title: 'HttpClient Basics',
    difficulty: 'easy',
    tags: ['java', 'http', 'httpclient', 'java11', 'networking'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `# HttpClient Basics

Java 11 introduced the HttpClient API as a modern replacement for the legacy HttpURLConnection. HttpClient provides a fluent API for building HTTP requests and supports both synchronous and asynchronous operations with HTTP/1.1 and HTTP/2.

## Requirements:
1. Create an HttpClient instance using newHttpClient():
   1.1. Use default configuration
   1.2. Display client version (HTTP/1.1 or HTTP/2)

2. Build an HttpRequest with HttpRequest.newBuilder():
   2.1. Set URI using uri() method
   2.2. Use GET method by default
   2.3. Build the request with build()

3. Create multiple requests:
   3.1. Simple GET request to https://api.github.com
   3.2. Request with custom URI
   3.3. Display request method and URI

4. Show HttpClient configuration options:
   4.1. Version (HTTP_1_1, HTTP_2)
   4.2. Follow redirects policy
   4.3. Connect timeout

## Example Output:
\`\`\`
=== HttpClient Basics ===
Client created: java.net.http.HttpClient@abc123
HTTP Version: HTTP_2

=== HttpRequest Building ===
Request 1:
  Method: GET
  URI: https://api.github.com

Request 2:
  Method: GET
  URI: https://httpbin.org/get

=== HttpClient Configuration ===
Client with custom config created
Version: HTTP_1_1
Redirect policy: NORMAL
\`\`\``,
    initialCode: `// TODO: Import necessary HTTP client classes

public class HttpClientBasics {
    public static void main(String[] args) {
        // TODO: Create an HttpClient instance

        // TODO: Build HttpRequest objects

        // TODO: Display request details

        // TODO: Create HttpClient with custom configuration
    }
}`,
    solutionCode: `import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.time.Duration;

public class HttpClientBasics {
    public static void main(String[] args) {
        System.out.println("=== HttpClient Basics ===");

        // Create an HttpClient instance with default configuration
        HttpClient client = HttpClient.newHttpClient();
        System.out.println("Client created: " + client);
        System.out.println("HTTP Version: " + client.version());

        System.out.println("\\n=== HttpRequest Building ===");

        // Build a simple GET request
        HttpRequest request1 = HttpRequest.newBuilder()
                .uri(URI.create("https://api.github.com"))
                .build();

        System.out.println("Request 1:");
        System.out.println("  Method: " + request1.method());
        System.out.println("  URI: " + request1.uri());

        // Build another GET request
        HttpRequest request2 = HttpRequest.newBuilder()
                .uri(URI.create("https://httpbin.org/get"))
                .GET()  // Explicitly specify GET method
                .build();

        System.out.println("\\nRequest 2:");
        System.out.println("  Method: " + request2.method());
        System.out.println("  URI: " + request2.uri());

        System.out.println("\\n=== HttpClient Configuration ===");

        // Create HttpClient with custom configuration
        HttpClient customClient = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .followRedirects(HttpClient.Redirect.NORMAL)
                .connectTimeout(Duration.ofSeconds(10))
                .build();

        System.out.println("Client with custom config created");
        System.out.println("Version: " + customClient.version());
        System.out.println("Redirect policy: " + customClient.followRedirects());
    }
}`,
    hint1: `Use HttpClient.newHttpClient() to create a client with default settings. Use HttpRequest.newBuilder().uri(...).build() to create a request.`,
    hint2: `The HttpClient.newBuilder() method allows you to configure version, redirects, timeouts, and other options before calling build().`,
    whyItMatters: `Understanding HttpClient basics is essential for modern Java networking. The HttpClient API is more efficient, easier to use, and supports modern HTTP features compared to the legacy HttpURLConnection. It's the foundation for making HTTP requests in Java applications.

**Production Pattern:**
\`\`\`java
// Create reusable HttpClient as singleton
private static final HttpClient CLIENT = HttpClient.newBuilder()
    .version(HttpClient.Version.HTTP_2)
    .connectTimeout(Duration.ofSeconds(10))
    .followRedirects(HttpClient.Redirect.NORMAL)
    .build();
\`\`\`

**Practical Benefits:**
- Reusing a single HttpClient instance saves resources
- HTTP/2 supports multiplexing for better performance
- Timeout configuration prevents application hanging`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.time.Duration;

// Test1: Verify HttpClient creation with default settings
class Test1 {
    @Test
    public void test() {
        HttpClient client = HttpClient.newHttpClient();
        assertNotNull(client);
        assertNotNull(client.version());
    }
}

// Test2: Verify HttpRequest creation with URI
class Test2 {
    @Test
    public void test() {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("https://api.github.com"))
                .build();
        assertEquals("GET", request.method());
        assertEquals("https://api.github.com", request.uri().toString());
    }
}

// Test3: Verify explicit GET method
class Test3 {
    @Test
    public void test() {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("https://httpbin.org/get"))
                .GET()
                .build();
        assertEquals("GET", request.method());
    }
}

// Test4: Verify custom HttpClient with HTTP_1_1 version
class Test4 {
    @Test
    public void test() {
        HttpClient client = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .build();
        assertEquals(HttpClient.Version.HTTP_1_1, client.version());
    }
}

// Test5: Verify custom HttpClient with HTTP_2 version
class Test5 {
    @Test
    public void test() {
        HttpClient client = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_2)
                .build();
        assertEquals(HttpClient.Version.HTTP_2, client.version());
    }
}

// Test6: Verify HttpClient with redirect policy
class Test6 {
    @Test
    public void test() {
        HttpClient client = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.NORMAL)
                .build();
        assertEquals(HttpClient.Redirect.NORMAL, client.followRedirects().orElse(null));
    }
}

// Test7: Verify HttpClient with connect timeout
class Test7 {
    @Test
    public void test() {
        HttpClient client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();
        assertTrue(client.connectTimeout().isPresent());
        assertEquals(Duration.ofSeconds(10), client.connectTimeout().get());
    }
}

// Test8: Verify multiple HttpRequest creation
class Test8 {
    @Test
    public void test() {
        HttpRequest req1 = HttpRequest.newBuilder()
                .uri(URI.create("https://example.com"))
                .build();
        HttpRequest req2 = HttpRequest.newBuilder()
                .uri(URI.create("https://google.com"))
                .build();
        assertNotEquals(req1.uri(), req2.uri());
    }
}

// Test9: Verify HttpClient builder pattern
class Test9 {
    @Test
    public void test() {
        HttpClient client = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .followRedirects(HttpClient.Redirect.ALWAYS)
                .connectTimeout(Duration.ofSeconds(5))
                .build();
        assertNotNull(client);
        assertEquals(HttpClient.Version.HTTP_1_1, client.version());
    }
}

// Test10: Verify HttpRequest URI handling
class Test10 {
    @Test
    public void test() {
        URI testUri = URI.create("https://api.example.com/v1/users");
        HttpRequest request = HttpRequest.newBuilder()
                .uri(testUri)
                .build();
        assertEquals(testUri, request.uri());
        assertEquals("https", request.uri().getScheme());
        assertEquals("api.example.com", request.uri().getHost());
    }
}
`,
    order: 1,
    translations: {
        ru: {
            title: 'Основы HttpClient',
            solutionCode: `import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.time.Duration;

public class HttpClientBasics {
    public static void main(String[] args) {
        System.out.println("=== Основы HttpClient ===");

        // Создание экземпляра HttpClient с конфигурацией по умолчанию
        HttpClient client = HttpClient.newHttpClient();
        System.out.println("Client created: " + client);
        System.out.println("HTTP Version: " + client.version());

        System.out.println("\\n=== Создание HttpRequest ===");

        // Создание простого GET-запроса
        HttpRequest request1 = HttpRequest.newBuilder()
                .uri(URI.create("https://api.github.com"))
                .build();

        System.out.println("Request 1:");
        System.out.println("  Method: " + request1.method());
        System.out.println("  URI: " + request1.uri());

        // Создание другого GET-запроса
        HttpRequest request2 = HttpRequest.newBuilder()
                .uri(URI.create("https://httpbin.org/get"))
                .GET()  // Явное указание метода GET
                .build();

        System.out.println("\\nRequest 2:");
        System.out.println("  Method: " + request2.method());
        System.out.println("  URI: " + request2.uri());

        System.out.println("\\n=== Конфигурация HttpClient ===");

        // Создание HttpClient с пользовательской конфигурацией
        HttpClient customClient = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .followRedirects(HttpClient.Redirect.NORMAL)
                .connectTimeout(Duration.ofSeconds(10))
                .build();

        System.out.println("Client with custom config created");
        System.out.println("Version: " + customClient.version());
        System.out.println("Redirect policy: " + customClient.followRedirects());
    }
}`,
            description: `# Основы HttpClient

Java 11 представила HttpClient API как современную замену устаревшему HttpURLConnection. HttpClient предоставляет fluent API для построения HTTP-запросов и поддерживает как синхронные, так и асинхронные операции с HTTP/1.1 и HTTP/2.

## Требования:
1. Создайте экземпляр HttpClient используя newHttpClient():
   1.1. Используйте конфигурацию по умолчанию
   1.2. Отобразите версию клиента (HTTP/1.1 или HTTP/2)

2. Постройте HttpRequest с помощью HttpRequest.newBuilder():
   2.1. Установите URI используя метод uri()
   2.2. Используйте метод GET по умолчанию
   2.3. Постройте запрос с помощью build()

3. Создайте несколько запросов:
   3.1. Простой GET-запрос к https://api.github.com
   3.2. Запрос с пользовательским URI
   3.3. Отобразите метод запроса и URI

4. Покажите опции конфигурации HttpClient:
   4.1. Версия (HTTP_1_1, HTTP_2)
   4.2. Политика перенаправлений
   4.3. Таймаут подключения

## Пример вывода:
\`\`\`
=== HttpClient Basics ===
Client created: java.net.http.HttpClient@abc123
HTTP Version: HTTP_2

=== HttpRequest Building ===
Request 1:
  Method: GET
  URI: https://api.github.com

Request 2:
  Method: GET
  URI: https://httpbin.org/get

=== HttpClient Configuration ===
Client with custom config created
Version: HTTP_1_1
Redirect policy: NORMAL
\`\`\``,
            hint1: `Используйте HttpClient.newHttpClient() для создания клиента с настройками по умолчанию. Используйте HttpRequest.newBuilder().uri(...).build() для создания запроса.`,
            hint2: `Метод HttpClient.newBuilder() позволяет настроить версию, перенаправления, таймауты и другие опции перед вызовом build().`,
            whyItMatters: `Понимание основ HttpClient необходимо для современной сетевой работы в Java. HttpClient API более эффективен, проще в использовании и поддерживает современные HTTP-функции по сравнению с устаревшим HttpURLConnection. Это основа для выполнения HTTP-запросов в Java-приложениях.

**Продакшен паттерн:**
\`\`\`java
// Создание переиспользуемого HttpClient как singleton
private static final HttpClient CLIENT = HttpClient.newBuilder()
    .version(HttpClient.Version.HTTP_2)
    .connectTimeout(Duration.ofSeconds(10))
    .followRedirects(HttpClient.Redirect.NORMAL)
    .build();
\`\`\`

**Практические преимущества:**
- Переиспользование одного экземпляра HttpClient экономит ресурсы
- HTTP/2 поддерживает мультиплексирование для лучшей производительности
- Настройка таймаутов предотвращает зависание приложения`
        },
        uz: {
            title: 'HttpClient asoslari',
            solutionCode: `import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.time.Duration;

public class HttpClientBasics {
    public static void main(String[] args) {
        System.out.println("=== HttpClient asoslari ===");

        // Standart konfiguratsiya bilan HttpClient nusxasini yaratish
        HttpClient client = HttpClient.newHttpClient();
        System.out.println("Client created: " + client);
        System.out.println("HTTP Version: " + client.version());

        System.out.println("\\n=== HttpRequest yaratish ===");

        // Oddiy GET so'rovini yaratish
        HttpRequest request1 = HttpRequest.newBuilder()
                .uri(URI.create("https://api.github.com"))
                .build();

        System.out.println("Request 1:");
        System.out.println("  Method: " + request1.method());
        System.out.println("  URI: " + request1.uri());

        // Boshqa GET so'rovini yaratish
        HttpRequest request2 = HttpRequest.newBuilder()
                .uri(URI.create("https://httpbin.org/get"))
                .GET()  // GET metodini aniq ko'rsatish
                .build();

        System.out.println("\\nRequest 2:");
        System.out.println("  Method: " + request2.method());
        System.out.println("  URI: " + request2.uri());

        System.out.println("\\n=== HttpClient konfiguratsiyasi ===");

        // Maxsus konfiguratsiya bilan HttpClient yaratish
        HttpClient customClient = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .followRedirects(HttpClient.Redirect.NORMAL)
                .connectTimeout(Duration.ofSeconds(10))
                .build();

        System.out.println("Client with custom config created");
        System.out.println("Version: " + customClient.version());
        System.out.println("Redirect policy: " + customClient.followRedirects());
    }
}`,
            description: `# HttpClient asoslari

Java 11 eski HttpURLConnection o'rniga zamonaviy HttpClient API-ni taqdim etdi. HttpClient HTTP so'rovlarini qurish uchun fluent API taqdim etadi va HTTP/1.1 va HTTP/2 bilan sinxron va asinxron operatsiyalarni qo'llab-quvvatlaydi.

## Talablar:
1. newHttpClient() yordamida HttpClient nusxasini yarating:
   1.1. Standart konfiguratsiyadan foydalaning
   1.2. Klient versiyasini ko'rsating (HTTP/1.1 yoki HTTP/2)

2. HttpRequest.newBuilder() yordamida HttpRequest yarating:
   2.1. uri() metodi yordamida URI o'rnating
   2.2. Standart bo'yicha GET metodidan foydalaning
   2.3. build() bilan so'rovni yaratib tugating

3. Bir nechta so'rovlar yarating:
   3.1. https://api.github.com ga oddiy GET so'rovi
   3.2. Maxsus URI bilan so'rov
   3.3. So'rov metodi va URI-ni ko'rsating

4. HttpClient konfiguratsiya opsiyalarini ko'rsating:
   4.1. Versiya (HTTP_1_1, HTTP_2)
   4.2. Qayta yo'naltirish siyosati
   4.3. Ulanish timeout

## Chiqish namunasi:
\`\`\`
=== HttpClient Basics ===
Client created: java.net.http.HttpClient@abc123
HTTP Version: HTTP_2

=== HttpRequest Building ===
Request 1:
  Method: GET
  URI: https://api.github.com

Request 2:
  Method: GET
  URI: https://httpbin.org/get

=== HttpClient Configuration ===
Client with custom config created
Version: HTTP_1_1
Redirect policy: NORMAL
\`\`\``,
            hint1: `Standart sozlamalar bilan klient yaratish uchun HttpClient.newHttpClient() dan foydalaning. So'rov yaratish uchun HttpRequest.newBuilder().uri(...).build() dan foydalaning.`,
            hint2: `HttpClient.newBuilder() metodi build() chaqirishdan oldin versiya, qayta yo'naltirishlar, timeoutlar va boshqa opsiyalarni sozlashga imkon beradi.`,
            whyItMatters: `HttpClient asoslarini tushunish zamonaviy Java tarmoq ishi uchun zarurdir. HttpClient API eski HttpURLConnection bilan solishtirganda samaraliroq, foydalanish osonroq va zamonaviy HTTP xususiyatlarini qo'llab-quvvatlaydi. Bu Java ilovalarida HTTP so'rovlarini bajarish uchun asos hisoblanadi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Qayta ishlatiluvchi HttpClient ni singleton sifatida yaratish
private static final HttpClient CLIENT = HttpClient.newBuilder()
    .version(HttpClient.Version.HTTP_2)
    .connectTimeout(Duration.ofSeconds(10))
    .followRedirects(HttpClient.Redirect.NORMAL)
    .build();
\`\`\`

**Amaliy foydalari:**
- Bitta HttpClient nusxasini qayta ishlatish resurslarni tejaydi
- HTTP/2 yaxshiroq ishlash uchun multiplekslashni qo'llab-quvvatlaydi
- Timeout sozlamalari ilova osilib qolishining oldini oladi`
        }
    }
};

export default task;
