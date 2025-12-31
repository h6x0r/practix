import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-sync-requests',
    title: 'Synchronous HTTP Requests',
    difficulty: 'easy',
    tags: ['java', 'http', 'httpclient', 'synchronous', 'networking'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Synchronous HTTP Requests

The send() method of HttpClient performs synchronous (blocking) HTTP requests. The calling thread waits for the response before continuing execution. This is the simplest approach for making HTTP calls when you need immediate results.

## Requirements:
1. Create an HttpClient instance

2. Build and send GET requests:
   2.1. Use send() method with BodyHandlers.ofString()
   2.2. Display status code and response body
   2.3. Handle different endpoints

3. Make POST request with body:
   3.1. Use POST() method with request body
   3.2. Send JSON data
   3.3. Display response

4. Handle exceptions:
   4.1. Catch IOException for network errors
   4.2. Catch InterruptedException for thread interruption
   4.3. Display error messages

## Example Output:
\`\`\`
=== Synchronous GET Request ===
Status Code: 200
Response Body (first 100 chars):
{
  "current_user_url": "https://api.github.com/user",
  "current_user_authorizations_html_url": "https...

=== Synchronous POST Request ===
Status Code: 200
POST Response:
{
  "json": {
    "title": "Test Post",
    "userId": 1
  }
}

=== Error Handling ===
Making request to invalid URL...
Error: nodename nor servname provided, or not known
\`\`\``,
    initialCode: `// TODO: Import necessary HTTP client classes

public class SyncRequests {
    public static void main(String[] args) {
        // TODO: Create HttpClient

        // TODO: Make GET request using send()

        // TODO: Make POST request with body

        // TODO: Demonstrate error handling
    }
}`,
    solutionCode: `import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

public class SyncRequests {
    public static void main(String[] args) {
        // Create HttpClient instance
        HttpClient client = HttpClient.newHttpClient();

        System.out.println("=== Synchronous GET Request ===");

        try {
            // Build GET request
            HttpRequest getRequest = HttpRequest.newBuilder()
                    .uri(URI.create("https://api.github.com"))
                    .GET()
                    .build();

            // Send synchronous request and get response
            HttpResponse<String> getResponse = client.send(
                getRequest,
                HttpResponse.BodyHandlers.ofString()
            );

            System.out.println("Status Code: " + getResponse.statusCode());
            System.out.println("Response Body (first 100 chars):");
            String body = getResponse.body();
            System.out.println(body.substring(0, Math.min(100, body.length())) + "...");

        } catch (IOException | InterruptedException e) {
            System.err.println("Error during GET request: " + e.getMessage());
        }

        System.out.println("\\n=== Synchronous POST Request ===");

        try {
            // Build POST request with JSON body
            String jsonBody = "{\\"title\\":\\"Test Post\\",\\"userId\\":1}";

            HttpRequest postRequest = HttpRequest.newBuilder()
                    .uri(URI.create("https://httpbin.org/post"))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                    .build();

            // Send POST request
            HttpResponse<String> postResponse = client.send(
                postRequest,
                HttpResponse.BodyHandlers.ofString()
            );

            System.out.println("Status Code: " + postResponse.statusCode());
            System.out.println("POST Response:");
            System.out.println(postResponse.body());

        } catch (IOException | InterruptedException e) {
            System.err.println("Error during POST request: " + e.getMessage());
        }

        System.out.println("\\n=== Error Handling ===");

        try {
            // Attempt request to invalid URL
            System.out.println("Making request to invalid URL...");
            HttpRequest errorRequest = HttpRequest.newBuilder()
                    .uri(URI.create("https://invalid-domain-that-does-not-exist-12345.com"))
                    .GET()
                    .build();

            client.send(errorRequest, HttpResponse.BodyHandlers.ofString());

        } catch (IOException e) {
            System.out.println("Error: " + e.getMessage());
        } catch (InterruptedException e) {
            System.out.println("Request interrupted: " + e.getMessage());
            Thread.currentThread().interrupt();
        }
    }
}`,
    hint1: `Use client.send(request, HttpResponse.BodyHandlers.ofString()) to make a synchronous request. This blocks until the response is received.`,
    hint2: `Always catch both IOException (network errors) and InterruptedException (thread interruption) when using send(). For POST requests, use POST(HttpRequest.BodyPublishers.ofString(json)).`,
    whyItMatters: `Synchronous requests are the simplest way to make HTTP calls and are ideal when you need immediate results or when the operation is fast enough that blocking isn't a concern. Understanding send() is fundamental before learning async patterns. Many REST API integrations start with synchronous requests.

**Production Pattern:**
\`\`\`java
// Error handling and retry logic
public String makeRobustRequest(String url) throws IOException {
    int maxRetries = 3;
    for (int i = 0; i < maxRetries; i++) {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .timeout(Duration.ofSeconds(5))
                .build();
            HttpResponse<String> response = client.send(request,
                HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() == 200) return response.body();
        } catch (IOException e) {
            if (i == maxRetries - 1) throw e;
            Thread.sleep(1000 * (i + 1)); // exponential backoff
        }
    }
    throw new IOException("Max retries exceeded");
}
\`\`\`

**Practical Benefits:**
- Retry logic increases reliability during temporary network failures
- Timeouts prevent infinite waiting
- Status code checking ensures correct response handling`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

// Test1: Verify HttpClient send method exists
class Test1 {
    @Test
    public void test() {
        HttpClient client = HttpClient.newHttpClient();
        assertNotNull(client);
    }
}

// Test2: Verify GET request creation
class Test2 {
    @Test
    public void test() {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("https://api.github.com"))
                .GET()
                .build();
        assertEquals("GET", request.method());
    }
}

// Test3: Verify POST request with body
class Test3 {
    @Test
    public void test() {
        String jsonBody = "{\\"title\\":\\"Test\\",\\"userId\\":1}";
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("https://httpbin.org/post"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                .build();
        assertEquals("POST", request.method());
        assertTrue(request.headers().firstValue("Content-Type").orElse("").contains("json"));
    }
}

// Test4: Verify request headers
class Test4 {
    @Test
    public void test() {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("https://httpbin.org/post"))
                .header("Content-Type", "application/json")
                .header("Accept", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString("{}"))
                .build();
        assertTrue(request.headers().map().containsKey("Content-Type"));
        assertTrue(request.headers().map().containsKey("Accept"));
    }
}

// Test5: Verify BodyPublishers.ofString
class Test5 {
    @Test
    public void test() {
        String testBody = "test data";
        HttpRequest.BodyPublisher publisher = HttpRequest.BodyPublishers.ofString(testBody);
        assertNotNull(publisher);
    }
}

// Test6: Verify request URI with POST
class Test6 {
    @Test
    public void test() {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("https://httpbin.org/post"))
                .POST(HttpRequest.BodyPublishers.ofString("data"))
                .build();
        assertEquals("https://httpbin.org/post", request.uri().toString());
    }
}

// Test7: Verify multiple header values
class Test7 {
    @Test
    public void test() {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("https://example.com"))
                .header("Custom-Header", "value1")
                .header("Another-Header", "value2")
                .GET()
                .build();
        assertTrue(request.headers().map().size() >= 2);
    }
}

// Test8: Verify JSON body creation
class Test8 {
    @Test
    public void test() {
        String jsonBody = "{\\"name\\":\\"John\\",\\"age\\":30}";
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("https://httpbin.org/post"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                .build();
        assertTrue(jsonBody.contains("John"));
        assertTrue(jsonBody.contains("30"));
    }
}

// Test9: Verify error request creation
class Test9 {
    @Test
    public void test() {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("https://invalid-domain-12345.com"))
                .GET()
                .build();
        assertEquals("https://invalid-domain-12345.com", request.uri().toString());
    }
}

// Test10: Verify BodyHandlers availability
class Test10 {
    @Test
    public void test() {
        HttpResponse.BodyHandler<String> handler = HttpResponse.BodyHandlers.ofString();
        assertNotNull(handler);
    }
}
`,
    order: 2,
    translations: {
        ru: {
            title: 'Синхронные HTTP-запросы',
            solutionCode: `import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

public class SyncRequests {
    public static void main(String[] args) {
        // Создание экземпляра HttpClient
        HttpClient client = HttpClient.newHttpClient();

        System.out.println("=== Синхронный GET-запрос ===");

        try {
            // Построение GET-запроса
            HttpRequest getRequest = HttpRequest.newBuilder()
                    .uri(URI.create("https://api.github.com"))
                    .GET()
                    .build();

            // Отправка синхронного запроса и получение ответа
            HttpResponse<String> getResponse = client.send(
                getRequest,
                HttpResponse.BodyHandlers.ofString()
            );

            System.out.println("Status Code: " + getResponse.statusCode());
            System.out.println("Response Body (first 100 chars):");
            String body = getResponse.body();
            System.out.println(body.substring(0, Math.min(100, body.length())) + "...");

        } catch (IOException | InterruptedException e) {
            System.err.println("Error during GET request: " + e.getMessage());
        }

        System.out.println("\\n=== Синхронный POST-запрос ===");

        try {
            // Построение POST-запроса с JSON-телом
            String jsonBody = "{\\"title\\":\\"Test Post\\",\\"userId\\":1}";

            HttpRequest postRequest = HttpRequest.newBuilder()
                    .uri(URI.create("https://httpbin.org/post"))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                    .build();

            // Отправка POST-запроса
            HttpResponse<String> postResponse = client.send(
                postRequest,
                HttpResponse.BodyHandlers.ofString()
            );

            System.out.println("Status Code: " + postResponse.statusCode());
            System.out.println("POST Response:");
            System.out.println(postResponse.body());

        } catch (IOException | InterruptedException e) {
            System.err.println("Error during POST request: " + e.getMessage());
        }

        System.out.println("\\n=== Обработка ошибок ===");

        try {
            // Попытка запроса к недействительному URL
            System.out.println("Making request to invalid URL...");
            HttpRequest errorRequest = HttpRequest.newBuilder()
                    .uri(URI.create("https://invalid-domain-that-does-not-exist-12345.com"))
                    .GET()
                    .build();

            client.send(errorRequest, HttpResponse.BodyHandlers.ofString());

        } catch (IOException e) {
            System.out.println("Error: " + e.getMessage());
        } catch (InterruptedException e) {
            System.out.println("Request interrupted: " + e.getMessage());
            Thread.currentThread().interrupt();
        }
    }
}`,
            description: `# Синхронные HTTP-запросы

Метод send() класса HttpClient выполняет синхронные (блокирующие) HTTP-запросы. Вызывающий поток ожидает ответа перед продолжением выполнения. Это простейший подход для выполнения HTTP-вызовов, когда вам нужны немедленные результаты.

## Требования:
1. Создайте экземпляр HttpClient

2. Постройте и отправьте GET-запросы:
   2.1. Используйте метод send() с BodyHandlers.ofString()
   2.2. Отобразите код состояния и тело ответа
   2.3. Обработайте различные конечные точки

3. Выполните POST-запрос с телом:
   3.1. Используйте метод POST() с телом запроса
   3.2. Отправьте JSON-данные
   3.3. Отобразите ответ

4. Обработайте исключения:
   4.1. Поймайте IOException для сетевых ошибок
   4.2. Поймайте InterruptedException для прерывания потока
   4.3. Отобразите сообщения об ошибках

## Пример вывода:
\`\`\`
=== Synchronous GET Request ===
Status Code: 200
Response Body (first 100 chars):
{
  "current_user_url": "https://api.github.com/user",
  "current_user_authorizations_html_url": "https...

=== Synchronous POST Request ===
Status Code: 200
POST Response:
{
  "json": {
    "title": "Test Post",
    "userId": 1
  }
}

=== Error Handling ===
Making request to invalid URL...
Error: nodename nor servname provided, or not known
\`\`\``,
            hint1: `Используйте client.send(request, HttpResponse.BodyHandlers.ofString()) для выполнения синхронного запроса. Это блокирует выполнение до получения ответа.`,
            hint2: `Всегда ловите и IOException (сетевые ошибки), и InterruptedException (прерывание потока) при использовании send(). Для POST-запросов используйте POST(HttpRequest.BodyPublishers.ofString(json)).`,
            whyItMatters: `Синхронные запросы - это простейший способ выполнения HTTP-вызовов и идеальны, когда вам нужны немедленные результаты или когда операция достаточно быстра, чтобы блокировка не была проблемой. Понимание send() является основой перед изучением асинхронных паттернов. Многие интеграции REST API начинаются с синхронных запросов.

**Продакшен паттерн:**
\`\`\`java
// Обработка ошибок и retry логика
public String makeRobustRequest(String url) throws IOException {
    int maxRetries = 3;
    for (int i = 0; i < maxRetries; i++) {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .timeout(Duration.ofSeconds(5))
                .build();
            HttpResponse<String> response = client.send(request,
                HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() == 200) return response.body();
        } catch (IOException e) {
            if (i == maxRetries - 1) throw e;
            Thread.sleep(1000 * (i + 1)); // exponential backoff
        }
    }
    throw new IOException("Max retries exceeded");
}
\`\`\`

**Практические преимущества:**
- Retry логика повышает надежность при временных сбоях сети
- Таймауты предотвращают бесконечное ожидание
- Проверка статус-кода обеспечивает корректную обработку ответов`
        },
        uz: {
            title: 'Sinxron HTTP so\'rovlar',
            solutionCode: `import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

public class SyncRequests {
    public static void main(String[] args) {
        // HttpClient nusxasini yaratish
        HttpClient client = HttpClient.newHttpClient();

        System.out.println("=== Sinxron GET so'rovi ===");

        try {
            // GET so'rovini qurish
            HttpRequest getRequest = HttpRequest.newBuilder()
                    .uri(URI.create("https://api.github.com"))
                    .GET()
                    .build();

            // Sinxron so'rovni yuborish va javobni olish
            HttpResponse<String> getResponse = client.send(
                getRequest,
                HttpResponse.BodyHandlers.ofString()
            );

            System.out.println("Status Code: " + getResponse.statusCode());
            System.out.println("Response Body (first 100 chars):");
            String body = getResponse.body();
            System.out.println(body.substring(0, Math.min(100, body.length())) + "...");

        } catch (IOException | InterruptedException e) {
            System.err.println("Error during GET request: " + e.getMessage());
        }

        System.out.println("\\n=== Sinxron POST so'rovi ===");

        try {
            // JSON tanasi bilan POST so'rovini qurish
            String jsonBody = "{\\"title\\":\\"Test Post\\",\\"userId\\":1}";

            HttpRequest postRequest = HttpRequest.newBuilder()
                    .uri(URI.create("https://httpbin.org/post"))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                    .build();

            // POST so'rovini yuborish
            HttpResponse<String> postResponse = client.send(
                postRequest,
                HttpResponse.BodyHandlers.ofString()
            );

            System.out.println("Status Code: " + postResponse.statusCode());
            System.out.println("POST Response:");
            System.out.println(postResponse.body());

        } catch (IOException | InterruptedException e) {
            System.err.println("Error during POST request: " + e.getMessage());
        }

        System.out.println("\\n=== Xatolarni boshqarish ===");

        try {
            // Noto'g'ri URL ga so'rov yuborish
            System.out.println("Making request to invalid URL...");
            HttpRequest errorRequest = HttpRequest.newBuilder()
                    .uri(URI.create("https://invalid-domain-that-does-not-exist-12345.com"))
                    .GET()
                    .build();

            client.send(errorRequest, HttpResponse.BodyHandlers.ofString());

        } catch (IOException e) {
            System.out.println("Error: " + e.getMessage());
        } catch (InterruptedException e) {
            System.out.println("Request interrupted: " + e.getMessage());
            Thread.currentThread().interrupt();
        }
    }
}`,
            description: `# Sinxron HTTP so'rovlar

HttpClient ning send() metodi sinxron (blokirovka qiluvchi) HTTP so'rovlarini bajaradi. Chaqiruvchi oqim javobni kutadi va keyingina bajarilishni davom ettiradi. Bu darhol natija kerak bo'lganda HTTP chaqiruvlarini bajarish uchun eng oddiy yondashuvdir.

## Talablar:
1. HttpClient nusxasini yarating

2. GET so'rovlarini quring va yuboring:
   2.1. send() metodini BodyHandlers.ofString() bilan ishlating
   2.2. Status kodi va javob tanasini ko'rsating
   2.3. Turli endpointlarni boshqaring

3. Tanasi bilan POST so'rovini bajaring:
   3.1. POST() metodini so'rov tanasi bilan ishlating
   3.2. JSON ma'lumotlarini yuboring
   3.3. Javobni ko'rsating

4. Istisnolarni boshqaring:
   4.1. Tarmoq xatolari uchun IOException ni ushlang
   4.2. Oqim to'xtatilishi uchun InterruptedException ni ushlang
   4.3. Xato xabarlarini ko'rsating

## Chiqish namunasi:
\`\`\`
=== Synchronous GET Request ===
Status Code: 200
Response Body (first 100 chars):
{
  "current_user_url": "https://api.github.com/user",
  "current_user_authorizations_html_url": "https...

=== Synchronous POST Request ===
Status Code: 200
POST Response:
{
  "json": {
    "title": "Test Post",
    "userId": 1
  }
}

=== Error Handling ===
Making request to invalid URL...
Error: nodename nor servname provided, or not known
\`\`\``,
            hint1: `Sinxron so'rov bajarish uchun client.send(request, HttpResponse.BodyHandlers.ofString()) dan foydalaning. Bu javob olinguncha blokirovka qiladi.`,
            hint2: `send() dan foydalanganda har doim IOException (tarmoq xatolari) va InterruptedException (oqim to'xtatilishi) ni ushlang. POST so'rovlar uchun POST(HttpRequest.BodyPublishers.ofString(json)) dan foydalaning.`,
            whyItMatters: `Sinxron so'rovlar HTTP chaqiruvlarini bajarishning eng oddiy usuli bo'lib, darhol natija kerak bo'lganda yoki operatsiya etarlicha tez bo'lganda va blokirovka muammo bo'lmaganda ideal hisoblanadi. send() ni tushunish asinxron naqshlarni o'rganishdan oldin asosdir. Ko'pgina REST API integratsiyalari sinxron so'rovlardan boshlanadi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Xatolarni qayta ishlash va retry logikasi
public String makeRobustRequest(String url) throws IOException {
    int maxRetries = 3;
    for (int i = 0; i < maxRetries; i++) {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .timeout(Duration.ofSeconds(5))
                .build();
            HttpResponse<String> response = client.send(request,
                HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() == 200) return response.body();
        } catch (IOException e) {
            if (i == maxRetries - 1) throw e;
            Thread.sleep(1000 * (i + 1)); // eksponensial kechikish
        }
    }
    throw new IOException("Maksimal urinishlar oshib ketdi");
}
\`\`\`

**Amaliy foydalari:**
- Retry logikasi vaqtinchalik tarmoq xatolarida ishonchlilikni oshiradi
- Timeoutlar cheksiz kutishning oldini oladi
- Status kod tekshiruvi javoblarni to'g'ri qayta ishlashni ta'minlaydi`
        }
    }
};

export default task;
