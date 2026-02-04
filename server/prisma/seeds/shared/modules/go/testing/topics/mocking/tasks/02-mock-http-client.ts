import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-testing-mock-http-client',
	title: 'Mock External Service',
	difficulty: 'hard',	tags: ['go', 'testing', 'mocking', 'interfaces'],
	estimatedTime: '30m',	isPremium: false,
	youtubeUrl: '',
	description: `Mock external service dependencies using **interface injection** for isolated testing.

**Requirements:**
1. Create \`WeatherService\` that uses \`DataFetcher\` interface
2. Implement \`DataFetcher\` interface with \`Fetch(query string) (string, error)\`
3. Create \`MockFetcher\` that returns predefined responses
4. Test service without making real external calls
5. Simulate success, error, and empty responses

**Example:**
\`\`\`go
// Interface for external data fetching
type DataFetcher interface {
    Fetch(query string) (string, error)
}

// Mock implementation for tests
type MockFetcher struct {
    Response string
    Err      error
}

func (m *MockFetcher) Fetch(query string) (string, error) {
    return m.Response, m.Err
}
\`\`\`

**Why This Pattern:**
This is exactly how you mock HTTP clients, database connections, or any external service.
The interface abstraction makes your code testable and decoupled.

**Constraints:**
- Service must accept interface, not concrete type
- Test all scenarios: success, error, empty response
- Verify the query parameter is passed correctly`,
	initialCode: `package mockservice_test

import (
	"errors"
	"strings"
)

// DataFetcher interface for external data sources
type DataFetcher interface {
	Fetch(query string) (string, error)
}

// WeatherService fetches weather data
type WeatherService struct {
	fetcher DataFetcher
	baseURL string
}

func NewWeatherService(fetcher DataFetcher, baseURL string) *WeatherService {
	return &WeatherService{fetcher: fetcher, baseURL: baseURL}
}

// TODO: Implement GetTemperature - should call fetcher.Fetch with "weather?city=X"
func (s *WeatherService) GetTemperature(city string) (string, error) {
	// TODO: Build query string
	// TODO: Call fetcher.Fetch
	// TODO: Return result or error
	return "", nil
}

// TODO: Create MockFetcher struct
type MockFetcher struct {
	// TODO: Add Response and Err fields
	// TODO: Add CapturedQuery to verify what was requested
}

// TODO: Implement Fetch method for MockFetcher
func (m *MockFetcher) Fetch(query string) (string, error) {
	// TODO: Implement
	return "", nil
}

// TODO: Write tests for WeatherService
func TestWeatherService_GetTemperature(t *T) {
	// TODO: Test success case
	// TODO: Test error case
	// TODO: Verify query contains city name
}`,
	solutionCode: `package mockservice_test

import (
	"errors"
	"strings"
)

// DataFetcher interface for external data sources
type DataFetcher interface {
	Fetch(query string) (string, error)
}

// WeatherService fetches weather data
type WeatherService struct {
	fetcher DataFetcher
	baseURL string
}

func NewWeatherService(fetcher DataFetcher, baseURL string) *WeatherService {
	return &WeatherService{fetcher: fetcher, baseURL: baseURL}
}

// GetTemperature fetches temperature for a city
func (s *WeatherService) GetTemperature(city string) (string, error) {
	query := s.baseURL + "?city=" + city  // Build query
	result, err := s.fetcher.Fetch(query) // Call external service
	if err != nil {
		return "", err
	}
	return result, nil
}

// MockFetcher is a test double for DataFetcher
type MockFetcher struct {
	Response      string  // What to return
	Err           error   // Error to return (if interface{})
	CapturedQuery string  // Capture what was requested
}

// Fetch implements DataFetcher interface
func (m *MockFetcher) Fetch(query string) (string, error) {
	m.CapturedQuery = query  // Record the query for verification
	return m.Response, m.Err
}

func TestWeatherService_GetTemperature(t *T) {
	t.Run("success", func(t *T) {
		// Create mock that returns success
		mock := &MockFetcher{
			Response: "25°C",
			Err:      nil,
		}

		service := NewWeatherService(mock, "http://api.weather.com")
		temp, err := service.GetTemperature("London")

		// Assert no error
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Assert correct response
		if temp != "25°C" {
			t.Errorf("got %q, want %q", temp, "25°C")
		}

		// Verify query was built correctly
		if !strings.Contains(mock.CapturedQuery, "city=London") {
			t.Errorf("query should contain city=London, got %q", mock.CapturedQuery)
		}
	})

	t.Run("API error", func(t *T) {
		// Create mock that returns error
		mock := &MockFetcher{
			Response: "",
			Err:      errors.New("connection refused"),
		}

		service := NewWeatherService(mock, "http://api.weather.com")
		_, err := service.GetTemperature("Invalid")

		// Assert error is returned
		if err == nil {
			t.Fatal("expected error, got nil")
		}
	})

	t.Run("empty response", func(t *T) {
		// Create mock that returns empty string
		mock := &MockFetcher{
			Response: "",
			Err:      nil,
		}

		service := NewWeatherService(mock, "http://api.weather.com")
		temp, err := service.GetTemperature("Unknown")

		// Assert no error but empty result
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if temp != "" {
			t.Errorf("expected empty string, got %q", temp)
		}
	})
}`,
		hint1: `Create MockFetcher with Response (string), Err (error), and CapturedQuery (string) fields.`,
		hint2: `In Fetch method, save query to CapturedQuery before returning Response and Err. This lets you verify correct queries in tests.`,
		testCode: `package mockservice_test

import (
	"errors"
	"strings"
)

// Test1: Success case returns correct temperature
func Test1(t *T) {
	mock := &MockFetcher{Response: "20°C", Err: nil}
	service := NewWeatherService(mock, "http://test.com")
	temp, err := service.GetTemperature("Paris")
	if err != nil || temp != "20°C" {
		t.Errorf("expected '20°C', got %q, err=%v", temp, err)
	}
}

// Test2: Error case returns error
func Test2(t *T) {
	mock := &MockFetcher{Response: "", Err: errors.New("network error")}
	service := NewWeatherService(mock, "http://test.com")
	_, err := service.GetTemperature("Unknown")
	if err == nil {
		t.Error("expected error for failed fetch")
	}
}

// Test3: Query contains city name
func Test3(t *T) {
	mock := &MockFetcher{Response: "15°C", Err: nil}
	service := NewWeatherService(mock, "http://api.test.com")
	service.GetTemperature("Berlin")
	if !strings.Contains(mock.CapturedQuery, "city=Berlin") {
		t.Errorf("query should contain 'city=Berlin', got %q", mock.CapturedQuery)
	}
}

// Test4: Empty response is handled
func Test4(t *T) {
	mock := &MockFetcher{Response: "", Err: nil}
	service := NewWeatherService(mock, "http://test.com")
	temp, err := service.GetTemperature("Empty")
	if err != nil || temp != "" {
		t.Errorf("expected empty string, got %q", temp)
	}
}

// Test5: Query includes base URL
func Test5(t *T) {
	mock := &MockFetcher{Response: "30°C", Err: nil}
	service := NewWeatherService(mock, "http://weather.api")
	service.GetTemperature("Tokyo")
	if !strings.Contains(mock.CapturedQuery, "http://weather.api") {
		t.Errorf("query should contain base URL, got %q", mock.CapturedQuery)
	}
}

// Test6: Multiple calls work independently
func Test6(t *T) {
	mock := &MockFetcher{Response: "25°C", Err: nil}
	service := NewWeatherService(mock, "http://test.com")

	service.GetTemperature("City1")
	query1 := mock.CapturedQuery

	service.GetTemperature("City2")
	query2 := mock.CapturedQuery

	if strings.Contains(query1, "City2") {
		t.Error("first query should not contain City2")
	}
	if !strings.Contains(query2, "City2") {
		t.Error("second query should contain City2")
	}
}

// Test7: Negative temperature
func Test7(t *T) {
	mock := &MockFetcher{Response: "-5°C", Err: nil}
	service := NewWeatherService(mock, "http://test.com")
	temp, _ := service.GetTemperature("Moscow")
	if temp != "-5°C" {
		t.Errorf("expected '-5°C', got %q", temp)
	}
}

// Test8: Service with nil fetcher uses default
func Test8(t *T) {
	mock := &MockFetcher{Response: "OK", Err: nil}
	service := NewWeatherService(mock, "")
	temp, err := service.GetTemperature("Test")
	if err != nil || temp != "OK" {
		t.Errorf("expected 'OK', got %q, err=%v", temp, err)
	}
}

// Test9: Long city name
func Test9(t *T) {
	mock := &MockFetcher{Response: "22°C", Err: nil}
	service := NewWeatherService(mock, "http://test.com")
	city := "San Francisco Bay Area"
	service.GetTemperature(city)
	if !strings.Contains(mock.CapturedQuery, city) {
		t.Errorf("query should contain full city name")
	}
}

// Test10: Different base URLs
func Test10(t *T) {
	mock := &MockFetcher{Response: "18°C", Err: nil}

	service1 := NewWeatherService(mock, "http://api1.com")
	service1.GetTemperature("A")
	query1 := mock.CapturedQuery

	service2 := NewWeatherService(mock, "http://api2.com")
	service2.GetTemperature("B")
	query2 := mock.CapturedQuery

	if !strings.Contains(query1, "api1.com") {
		t.Error("first service should use api1.com")
	}
	if !strings.Contains(query2, "api2.com") {
		t.Error("second service should use api2.com")
	}
}
`,
		whyItMatters: `Interface-based mocking is fundamental for testing code that depends on external services.

**The Pattern:**
\`\`\`go
// 1. Define interface for dependency
type DataFetcher interface {
    Fetch(query string) (string, error)
}

// 2. Service accepts interface, not concrete type
type WeatherService struct {
    fetcher DataFetcher  // NOT *HttpClient
}

// 3. In production: use real HTTP client
service := NewWeatherService(httpClient, url)

// 4. In tests: use mock
service := NewWeatherService(&MockFetcher{Response: "25°C"}, url)
\`\`\`

**Why This Matters:**
- **HTTP Clients:** Mock API responses without network calls
- **Databases:** Mock queries without database setup
- **File Systems:** Mock file operations without real files
- **Time:** Mock clock for testing time-dependent code

**Real-World Example - Testing Payment Service:**
\`\`\`go
type PaymentGateway interface {
    Charge(amount int, card string) error
}

type MockGateway struct {
    ShouldFail bool
}

func (m *MockGateway) Charge(amount int, card string) error {
    if m.ShouldFail {
        return errors.New("card declined")
    }
    return nil
}

func TestCheckout_PaymentFails(t *testing.T) {
    mock := &MockGateway{ShouldFail: true}
    checkout := NewCheckout(mock)

    err := checkout.Process(100, "4242...")

    if err == nil {
        t.Error("expected payment error")
    }
}
\`\`\`

**Benefits:**
- **Fast:** No network latency
- **Reliable:** No flaky tests from API downtime
- **Controllable:** Test error scenarios easily
- **Cost-free:** No API rate limits or charges

This is exactly how companies like Stripe, Uber, and Google test their services.`,
	order: 1,
	translations: {
		ru: {
			title: 'Мокирование внешнего сервиса',
			description: `Мокируйте зависимости внешних сервисов используя **interface injection** для изолированного тестирования.

**Требования:**
1. Создайте \`WeatherService\` использующий интерфейс \`DataFetcher\`
2. Реализуйте интерфейс \`DataFetcher\` с методом \`Fetch(query string) (string, error)\`
3. Создайте \`MockFetcher\` возвращающий предопределённые ответы
4. Протестируйте сервис без реальных внешних вызовов
5. Симулируйте успех, ошибку и пустой ответ

**Пример:**
\`\`\`go
// Интерфейс для получения внешних данных
type DataFetcher interface {
    Fetch(query string) (string, error)
}

// Mock реализация для тестов
type MockFetcher struct {
    Response string
    Err      error
}

func (m *MockFetcher) Fetch(query string) (string, error) {
    return m.Response, m.Err
}
\`\`\`

**Почему этот паттерн:**
Это именно то, как вы мокируете HTTP клиенты, подключения к БД, или любой внешний сервис.
Абстракция через интерфейс делает ваш код тестируемым и слабо связанным.`,
			hint1: `Создайте MockFetcher с полями Response (string), Err (error), и CapturedQuery (string).`,
			hint2: `В методе Fetch сохраните query в CapturedQuery перед возвратом Response и Err. Это позволит проверить корректность запросов в тестах.`,
			whyItMatters: `Мокирование на основе интерфейсов фундаментально для тестирования кода зависящего от внешних сервисов.

**Паттерн:**
\`\`\`go
// 1. Определить интерфейс для зависимости
type DataFetcher interface {
    Fetch(query string) (string, error)
}

// 2. Сервис принимает интерфейс, не конкретный тип
type WeatherService struct {
    fetcher DataFetcher  // НЕ *HttpClient
}

// 3. В production: использовать реальный HTTP клиент
service := NewWeatherService(httpClient, url)

// 4. В тестах: использовать mock
service := NewWeatherService(&MockFetcher{Response: "25°C"}, url)
\`\`\`

**Почему это важно:**
- **HTTP клиенты:** Мокировать API ответы без сетевых вызовов
- **Базы данных:** Мокировать запросы без настройки БД
- **Файловые системы:** Мокировать операции с файлами без реальных файлов

Это именно то, как Stripe, Uber и Google тестируют свои сервисы.`,
			solutionCode: `package mockservice_test

import (
	"errors"
	"strings"
)

// DataFetcher интерфейс для внешних источников данных
type DataFetcher interface {
	Fetch(query string) (string, error)
}

// WeatherService получает данные о погоде
type WeatherService struct {
	fetcher DataFetcher
	baseURL string
}

func NewWeatherService(fetcher DataFetcher, baseURL string) *WeatherService {
	return &WeatherService{fetcher: fetcher, baseURL: baseURL}
}

// GetTemperature получает температуру для города
func (s *WeatherService) GetTemperature(city string) (string, error) {
	query := s.baseURL + "?city=" + city  // Построить запрос
	result, err := s.fetcher.Fetch(query) // Вызвать внешний сервис
	if err != nil {
		return "", err
	}
	return result, nil
}

// MockFetcher - тестовый дублёр для DataFetcher
type MockFetcher struct {
	Response      string  // Что возвращать
	Err           error   // Ошибка (если есть)
	CapturedQuery string  // Захват запроса
}

// Fetch реализует интерфейс DataFetcher
func (m *MockFetcher) Fetch(query string) (string, error) {
	m.CapturedQuery = query  // Записать запрос для проверки
	return m.Response, m.Err
}

func TestWeatherService_GetTemperature(t *T) {
	t.Run("успех", func(t *T) {
		mock := &MockFetcher{Response: "25°C", Err: nil}
		service := NewWeatherService(mock, "http://api.weather.com")
		temp, err := service.GetTemperature("London")

		if err != nil {
			t.Fatalf("неожиданная ошибка: %v", err)
		}
		if temp != "25°C" {
			t.Errorf("получено %q, ожидается %q", temp, "25°C")
		}
		if !strings.Contains(mock.CapturedQuery, "city=London") {
			t.Errorf("запрос должен содержать city=London, получено %q", mock.CapturedQuery)
		}
	})

	t.Run("ошибка API", func(t *T) {
		mock := &MockFetcher{Response: "", Err: errors.New("connection refused")}
		service := NewWeatherService(mock, "http://api.weather.com")
		_, err := service.GetTemperature("Invalid")

		if err == nil {
			t.Fatal("ожидается ошибка, получено nil")
		}
	})

	t.Run("пустой ответ", func(t *T) {
		mock := &MockFetcher{Response: "", Err: nil}
		service := NewWeatherService(mock, "http://api.weather.com")
		temp, err := service.GetTemperature("Unknown")

		if err != nil {
			t.Fatalf("неожиданная ошибка: %v", err)
		}
		if temp != "" {
			t.Errorf("ожидается пустая строка, получено %q", temp)
		}
	})
}`
		},
		uz: {
			title: `Tashqi servisni mocking qilish`,
			description: `Izolyatsiya qilingan test qilish uchun **interface injection** dan foydalanib tashqi servis dependency larni mock qiling.

**Talablar:**
1. \`DataFetcher\` interfeys dan foydalanadigan \`WeatherService\` yarating
2. \`Fetch(query string) (string, error)\` bilan \`DataFetcher\` interfeysini amalga oshiring
3. Oldindan belgilangan javoblarni qaytaradigan \`MockFetcher\` yarating
4. Haqiqiy tashqi chaqiruvlarsiz servisni test qiling
5. Muvaffaqiyat, xato va bo'sh javoblarni simulyatsiya qiling`,
			hint1: `Response (string), Err (error), va CapturedQuery (string) fieldlari bilan MockFetcher yarating.`,
			hint2: `Fetch metodida Response va Err ni qaytarishdan oldin query ni CapturedQuery ga saqlang. Bu testlarda to'g'ri so'rovlarni tekshirish imkonini beradi.`,
			whyItMatters: `Interfeys asosidagi mocking tashqi servislarga bog'liq kodni test qilish uchun fundamental.

Bu Stripe, Uber va Google o'z servislarini qanday test qilishidir.`,
			solutionCode: `package mockservice_test

import (
	"errors"
	"strings"
)

type DataFetcher interface {
	Fetch(query string) (string, error)
}

type WeatherService struct {
	fetcher DataFetcher
	baseURL string
}

func NewWeatherService(fetcher DataFetcher, baseURL string) *WeatherService {
	return &WeatherService{fetcher: fetcher, baseURL: baseURL}
}

func (s *WeatherService) GetTemperature(city string) (string, error) {
	query := s.baseURL + "?city=" + city
	result, err := s.fetcher.Fetch(query)
	if err != nil {
		return "", err
	}
	return result, nil
}

type MockFetcher struct {
	Response      string
	Err           error
	CapturedQuery string
}

func (m *MockFetcher) Fetch(query string) (string, error) {
	m.CapturedQuery = query
	return m.Response, m.Err
}

func TestWeatherService_GetTemperature(t *T) {
	t.Run("muvaffaqiyat", func(t *T) {
		mock := &MockFetcher{Response: "25°C", Err: nil}
		service := NewWeatherService(mock, "http://api.weather.com")
		temp, err := service.GetTemperature("London")

		if err != nil {
			t.Fatalf("kutilmagan xato: %v", err)
		}
		if temp != "25°C" {
			t.Errorf("olindi %q, kutilgan %q", temp, "25°C")
		}
		if !strings.Contains(mock.CapturedQuery, "city=London") {
			t.Errorf("so'rov city=London ni o'z ichiga olishi kerak, olindi %q", mock.CapturedQuery)
		}
	})

	t.Run("API xatosi", func(t *T) {
		mock := &MockFetcher{Response: "", Err: errors.New("connection refused")}
		service := NewWeatherService(mock, "http://api.weather.com")
		_, err := service.GetTemperature("Invalid")

		if err == nil {
			t.Fatal("xato kutilgan, nil olindi")
		}
	})
}`
		}
	}
};

export default task;
