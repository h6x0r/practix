import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-grasp-protected-variations',
	title: 'Protected Variations',
	difficulty: 'hard',
	tags: ['go', 'software-engineering', 'grasp', 'protected-variations'],
	estimatedTime: '45m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Protected Variations principle - protect elements from variations in other elements by wrapping them with stable interfaces.

**You will implement:**

1. **ConfigReader interface** - Stable interface for reading configuration
2. **JSONConfigReader struct** - Reads from JSON files
3. **EnvConfigReader struct** - Reads from environment variables
4. **Application struct** - Uses ConfigReader interface (protected from config format changes!)

**Key Concepts:**
- **Protected Variations**: Stable interface protects against implementation changes
- **Open/Closed Principle**: Open for extension, closed for modification
- **Encapsulation**: Hide volatile implementation details

**Example Usage:**

\`\`\`go
// Application works with any ConfigReader - protected from variations!
jsonReader := NewJSONConfigReader("config.json")
app := NewApplication(jsonReader)
dbHost := app.GetConfig("database.host")

// Later, switch to environment variables - Application unchanged!
envReader := NewEnvConfigReader()
app = NewApplication(envReader)
dbHost = app.GetConfig("DATABASE_HOST")  // different key format, same interface

// Add YAML reader - Application still unchanged!
yamlReader := NewYAMLConfigReader("config.yaml")
app = NewApplication(yamlReader)
\`\`\`

**Why Protected Variations?**
- **Stability**: Changes in one part don't affect others
- **Flexibility**: Easy to add new implementations
- **Evolution**: System can evolve without breaking existing code

**Anti-pattern (Don't do this):**
\`\`\`go
// UNPROTECTED - BAD!
type Application struct {
    configFile string
}

func (a *Application) GetConfig(key string) string {
    // Directly depends on JSON format!
    data, _ := ioutil.ReadFile(a.configFile)
    var config map[string]string
    json.Unmarshal(data, &config)
    return config[key]
}
// If config format changes to YAML, must rewrite Application!
// If need env vars, must rewrite Application!
\`\`\`

**Constraints:**
- Application must NOT know about specific config formats
- All config readers must implement ConfigReader interface
- Interface provides stable contract despite varying implementations`,
	initialCode: `package principles

import (
	"encoding/json"
	"os"
)

type ConfigReader interface {
}

type JSONConfigReader struct {
	filename string
	data     map[string]string
}

func NewJSONConfigReader(filename string) *JSONConfigReader {
	}
}

func (r *JSONConfigReader) Get(key string) string {
}

type EnvConfigReader struct{}

func NewEnvConfigReader() *EnvConfigReader {
	return &EnvConfigReader{}
}

func (r *EnvConfigReader) Get(key string) string {
}

type Application struct {
	configReader ConfigReader // stable interface protects from variations
}

func NewApplication(configReader ConfigReader) *Application {
}

func (a *Application) GetConfig(key string) string {
}`,
	solutionCode: `package principles

import (
	"os"
)

// ConfigReader interface - STABLE CONTRACT that protects from variations
type ConfigReader interface {
	Get(key string) string
}

// JSONConfigReader - one variation (JSON files)
type JSONConfigReader struct {
	filename string
	data     map[string]string
}

func NewJSONConfigReader(filename string) *JSONConfigReader {
	reader := &JSONConfigReader{
		filename: filename,
		data:     make(map[string]string),
	}
	// Simulate reading JSON file
	reader.data["database.host"] = "localhost"
	reader.data["database.port"] = "5432"
	return reader
}

func (r *JSONConfigReader) Get(key string) string {
	// JSON-specific implementation
	// But Application doesn't know this - it's protected by interface
	return r.data[key]
}

// EnvConfigReader - another variation (environment variables)
type EnvConfigReader struct{}

func NewEnvConfigReader() *EnvConfigReader {
	// Simulate environment variables
	os.Setenv("DATABASE_HOST", "prod-db.example.com")
	os.Setenv("DATABASE_PORT", "3306")
	return &EnvConfigReader{}
}

func (r *EnvConfigReader) Get(key string) string {
	// Env-specific implementation
	// But Application doesn't know this - it's protected by interface
	return os.Getenv(key)
}

// Application is PROTECTED from config source variations
type Application struct {
	configReader ConfigReader	// depends on stable interface
}

func NewApplication(configReader ConfigReader) *Application {
	return &Application{
		configReader: configReader,	// accepts any ConfigReader
	}
}

func (a *Application) GetConfig(key string) string {
	// Application code is stable and never changes
	// Even when config sources change (JSON -> Env -> YAML -> Database)
	// That's Protected Variations!
	return a.configReader.Get(key)
}`,
	hint1: `Define ConfigReader interface with Get(key string) string. Both JSONConfigReader and EnvConfigReader should implement this method.`,
	hint2: `JSONConfigReader.Get returns r.data[key]. EnvConfigReader.Get returns os.Getenv(key). Application.GetConfig calls a.configReader.Get(key).`,
	whyItMatters: `Protected Variations is the most general GRASP principle - it protects against the impact of changes by using stable interfaces.

**Why Protected Variations Matters:**

**1. Shields from External Changes**
When external systems change, stable interface protects your code:

\`\`\`go
// WITHOUT PROTECTED VARIATIONS - FRAGILE!
type PaymentService struct{}

func (p *PaymentService) ProcessPayment(amount float64, cardNumber string) error {
    // Directly using Stripe API v1
    client := stripe.NewClient("sk_key")
    _, err := client.Charges.New(&stripe.ChargeParams{
        Amount:   stripe.Int64(int64(amount * 100)),
        Currency: stripe.String("usd"),
        Source:   &stripe.SourceParams{Token: stripe.String(cardNumber)},
    })
    return err
}
// When Stripe updates to v2, must rewrite ProcessPayment!
// When switching to PayPal, must rewrite ProcessPayment!

// WITH PROTECTED VARIATIONS - ROBUST!
type PaymentProcessor interface {
    Charge(amount float64, token string) (*Payment, error)
}

type StripeProcessor struct {
    client *stripe.Client
}

func (s *StripeProcessor) Charge(amount float64, token string) (*Payment, error) {
    // Stripe-specific code isolated here
    charge, err := s.client.Charges.New(&stripe.ChargeParams{
        Amount:   stripe.Int64(int64(amount * 100)),
        Currency: stripe.String("usd"),
        Source:   &stripe.SourceParams{Token: stripe.String(token)},
    })
    return &Payment{ID: charge.ID}, err
}

type PaymentService struct {
    processor PaymentProcessor  // protected by stable interface
}

func (p *PaymentService) ProcessPayment(amount float64, token string) error {
    // Never changes, even when processor implementation changes!
    _, err := p.processor.Charge(amount, token)
    return err
}
// Stripe API changes? Update StripeProcessor only
// Switch to PayPal? Create PayPalProcessor - PaymentService unchanged!
\`\`\`

**2. Real-World Example: Data Storage**
\`\`\`go
// Storage interface protects from storage variations
type Storage interface {
    Save(key string, data []byte) error
    Load(key string) ([]byte, error)
    Delete(key string) error
}

// Local file storage
type FileStorage struct {
    basePath string
}

func (f *FileStorage) Save(key string, data []byte) error {
    path := filepath.Join(f.basePath, key)
    return ioutil.WriteFile(path, data, 0644)
}

func (f *FileStorage) Load(key string) ([]byte, error) {
    path := filepath.Join(f.basePath, key)
    return ioutil.ReadFile(path)
}

func (f *FileStorage) Delete(key string) error {
    path := filepath.Join(f.basePath, key)
    return os.Remove(path)
}

// S3 cloud storage
type S3Storage struct {
    bucket string
    client *s3.Client
}

func (s *S3Storage) Save(key string, data []byte) error {
    _, err := s.client.PutObject(&s3.PutObjectInput{
        Bucket: &s.bucket,
        Key:    &key,
        Body:   bytes.NewReader(data),
    })
    return err
}

func (s *S3Storage) Load(key string) ([]byte, error) {
    result, err := s.client.GetObject(&s3.GetObjectInput{
        Bucket: &s.bucket,
        Key:    &key,
    })
    if err != nil {
        return nil, err
    }
    return ioutil.ReadAll(result.Body)
}

func (s *S3Storage) Delete(key string) error {
    _, err := s.client.DeleteObject(&s3.DeleteObjectInput{
        Bucket: &s.bucket,
        Key:    &key,
    })
    return err
}

// Redis storage
type RedisStorage struct {
    client *redis.Client
}

func (r *RedisStorage) Save(key string, data []byte) error {
    return r.client.Set(context.Background(), key, data, 0).Err()
}

func (r *RedisStorage) Load(key string) ([]byte, error) {
    return r.client.Get(context.Background(), key).Bytes()
}

func (r *RedisStorage) Delete(key string) error {
    return r.client.Del(context.Background(), key).Err()
}

// DocumentManager is PROTECTED from storage variations
type DocumentManager struct {
    storage Storage
}

func (d *DocumentManager) SaveDocument(name string, content []byte) error {
    // This code never changes, even when storage backend changes!
    return d.storage.Save(name, content)
}

func (d *DocumentManager) GetDocument(name string) ([]byte, error) {
    return d.storage.Load(name)
}

// Usage - easy to switch storage
func main() {
    // Start with local files
    manager := &DocumentManager{storage: &FileStorage{basePath: "/data"}}

    // Move to S3 - DocumentManager code unchanged!
    manager = &DocumentManager{storage: &S3Storage{bucket: "my-bucket", client: s3Client}}

    // Add Redis caching - DocumentManager code unchanged!
    manager = &DocumentManager{storage: &RedisStorage{client: redisClient}}
}
\`\`\`

**3. API Versioning Protection**
\`\`\`go
// API interface protects from version changes
type UserAPI interface {
    GetUser(id int) (*User, error)
    CreateUser(user *User) error
    UpdateUser(user *User) error
}

// V1 API implementation
type UserAPIV1 struct {
    baseURL string
}

func (a *UserAPIV1) GetUser(id int) (*User, error) {
    // V1 format: /users/:id
    resp, err := http.Get(fmt.Sprintf("%s/users/%d", a.baseURL, id))
    if err != nil {
        return nil, err
    }
    var user User
    json.NewDecoder(resp.Body).Decode(&user)
    return &user, nil
}

// V2 API implementation
type UserAPIV2 struct {
    baseURL string
}

func (a *UserAPIV2) GetUser(id int) (*User, error) {
    // V2 format: /v2/users/:id with different response structure
    resp, err := http.Get(fmt.Sprintf("%s/v2/users/%d", a.baseURL, id))
    if err != nil {
        return nil, err
    }
    var response struct {
        Data User \`json:"data"\`
        Meta struct {
            Version string \`json:"version"\`
        } \`json:"meta"\`
    }
    json.NewDecoder(resp.Body).Decode(&response)
    return &response.Data, nil
}

// UserService is PROTECTED from API version changes
type UserService struct {
    api UserAPI
}

func (s *UserService) GetUserProfile(id int) (*User, error) {
    // Same code works with v1, v2, or future v3
    // Protected from API variations!
    return s.api.GetUser(id)
}
\`\`\`

**4. Database Schema Evolution**
\`\`\`go
// Repository interface protects from schema changes
type ProductRepository interface {
    FindByID(id int) (*Product, error)
    FindAll() ([]*Product, error)
    Save(product *Product) error
}

// V1 schema: products table with basic fields
type ProductRepoV1 struct {
    db *sql.DB
}

func (r *ProductRepoV1) FindByID(id int) (*Product, error) {
    var p Product
    err := r.db.QueryRow(
        "SELECT id, name, price FROM products WHERE id = ?",
        id,
    ).Scan(&p.ID, &p.Name, &p.Price)
    return &p, err
}

// V2 schema: products split into products + product_details tables
type ProductRepoV2 struct {
    db *sql.DB
}

func (r *ProductRepoV2) FindByID(id int) (*Product, error) {
    var p Product
    err := r.db.QueryRow(\`
        SELECT p.id, p.name, p.price, d.description, d.category
        FROM products p
        JOIN product_details d ON p.id = d.product_id
        WHERE p.id = ?
    \`, id).Scan(&p.ID, &p.Name, &p.Price, &p.Description, &p.Category)
    return &p, err
}

// ProductService is PROTECTED from schema changes
type ProductService struct {
    repo ProductRepository
}

func (s *ProductService) GetProduct(id int) (*Product, error) {
    // Same code works with v1 or v2 schema
    // Protected from database variations!
    return s.repo.FindByID(id)
}
\`\`\`

**5. Protected Variations Strategies**
- **Interfaces**: Most common - define stable contracts
- **Encapsulation**: Hide volatile data/implementation
- **Data-Driven Design**: Use configuration instead of code
- **Service Lookup**: Discover implementations at runtime
- **Adapter/Wrapper**: Wrap unstable components

**Common Variations to Protect Against:**
- External APIs and libraries
- Database schemas
- File formats
- Communication protocols
- Third-party services
- Operating system differences

**Common Mistakes:**
- Creating interfaces too early (YAGNI)
- Interfaces that leak implementation details
- Too many small interfaces (over-engineering)
- Not protecting against likely variations

**Rule of Thumb:**
Identify points of likely variation and protect against them with stable interfaces. Focus on variations that are likely, not just possible.`,
	order: 8,
	testCode: `package principles

import (
	"os"
	"testing"
)

// Test1: JSONConfigReader implements ConfigReader
func Test1(t *testing.T) {
	reader := NewJSONConfigReader("config.json")
	var _ ConfigReader = reader // compile-time check
	if reader == nil {
		t.Error("JSONConfigReader should not be nil")
	}
}

// Test2: EnvConfigReader implements ConfigReader
func Test2(t *testing.T) {
	reader := NewEnvConfigReader()
	var _ ConfigReader = reader // compile-time check
	if reader == nil {
		t.Error("EnvConfigReader should not be nil")
	}
}

// Test3: JSONConfigReader returns database.host
func Test3(t *testing.T) {
	reader := NewJSONConfigReader("config.json")
	value := reader.Get("database.host")
	if value != "localhost" {
		t.Errorf("Expected localhost, got %s", value)
	}
}

// Test4: JSONConfigReader returns database.port
func Test4(t *testing.T) {
	reader := NewJSONConfigReader("config.json")
	value := reader.Get("database.port")
	if value != "5432" {
		t.Errorf("Expected 5432, got %s", value)
	}
}

// Test5: EnvConfigReader returns env variable
func Test5(t *testing.T) {
	reader := NewEnvConfigReader()
	value := reader.Get("DATABASE_HOST")
	if value != "prod-db.example.com" {
		t.Errorf("Expected prod-db.example.com, got %s", value)
	}
}

// Test6: Application with JSONConfigReader
func Test6(t *testing.T) {
	reader := NewJSONConfigReader("config.json")
	app := NewApplication(reader)
	value := app.GetConfig("database.host")
	if value != "localhost" {
		t.Errorf("Expected localhost, got %s", value)
	}
}

// Test7: Application with EnvConfigReader
func Test7(t *testing.T) {
	reader := NewEnvConfigReader()
	app := NewApplication(reader)
	value := app.GetConfig("DATABASE_HOST")
	if value != "prod-db.example.com" {
		t.Errorf("Expected prod-db.example.com, got %s", value)
	}
}

// Test8: JSONConfigReader returns empty for non-existent key
func Test8(t *testing.T) {
	reader := NewJSONConfigReader("config.json")
	value := reader.Get("nonexistent.key")
	if value != "" {
		t.Error("Should return empty for non-existent key")
	}
}

// Test9: EnvConfigReader returns empty for non-existent env
func Test9(t *testing.T) {
	reader := NewEnvConfigReader()
	os.Unsetenv("NONEXISTENT_VAR")
	value := reader.Get("NONEXISTENT_VAR")
	if value != "" {
		t.Error("Should return empty for non-existent env var")
	}
}

// Test10: Application creation
func Test10(t *testing.T) {
	reader := NewJSONConfigReader("config.json")
	app := NewApplication(reader)
	if app == nil {
		t.Error("Application should not be nil")
	}
}
`,
	translations: {
		ru: {
			title: 'Защищённые вариации',
			description: `Реализуйте принцип Защищённых вариаций — защитите элементы от изменений в других элементах, оборачивая их стабильными интерфейсами.

**Вы реализуете:**

1. **ConfigReader interface** — Стабильный интерфейс для чтения конфигурации
2. **JSONConfigReader struct** — Читает из JSON-файлов
3. **EnvConfigReader struct** — Читает из переменных окружения
4. **Application struct** — Использует интерфейс ConfigReader (защищён от изменений формата конфига!)

**Ключевые концепции:**
- **Защищённые вариации**: Стабильный интерфейс защищает от изменений реализации
- **Принцип открытости/закрытости**: Открыт для расширения, закрыт для модификации
- **Инкапсуляция**: Скрывает изменчивые детали реализации

**Зачем нужны Защищённые вариации?**
- **Стабильность**: Изменения в одной части не влияют на другие
- **Гибкость**: Легко добавлять новые реализации
- **Эволюция**: Система может эволюционировать без нарушения существующего кода

**Ограничения:**
- Application НЕ должно знать о конкретных форматах конфигурации
- Все читатели конфигурации должны реализовывать интерфейс ConfigReader
- Интерфейс предоставляет стабильный контракт несмотря на различающиеся реализации`,
			hint1: `Определите интерфейс ConfigReader с Get(key string) string. И JSONConfigReader, и EnvConfigReader должны реализовать этот метод.`,
			hint2: `JSONConfigReader.Get возвращает r.data[key]. EnvConfigReader.Get возвращает os.Getenv(key). Application.GetConfig вызывает a.configReader.Get(key).`,
			whyItMatters: `Защищённые вариации — самый общий принцип GRASP, который защищает от влияния изменений с помощью стабильных интерфейсов.

**Почему Защищённые вариации важны:**

**1. Защита от внешних изменений**
Когда внешние системы меняются, стабильный интерфейс защищает ваш код.

**Распространённые ошибки:**
- Создание интерфейсов слишком рано (YAGNI)
- Интерфейсы, которые раскрывают детали реализации
- Слишком много мелких интерфейсов (излишняя инженерия)
- Отсутствие защиты от вероятных вариаций`,
			solutionCode: `package principles

import (
	"os"
)

// ConfigReader интерфейс - СТАБИЛЬНЫЙ КОНТРАКТ который защищает от вариаций
type ConfigReader interface {
	Get(key string) string
}

// JSONConfigReader - одна вариация (JSON-файлы)
type JSONConfigReader struct {
	filename string
	data     map[string]string
}

func NewJSONConfigReader(filename string) *JSONConfigReader {
	reader := &JSONConfigReader{
		filename: filename,
		data:     make(map[string]string),
	}
	// Имитация чтения JSON-файла
	reader.data["database.host"] = "localhost"
	reader.data["database.port"] = "5432"
	return reader
}

func (r *JSONConfigReader) Get(key string) string {
	// JSON-специфичная реализация
	// Но Application этого не знает - защищено интерфейсом
	return r.data[key]
}

// EnvConfigReader - другая вариация (переменные окружения)
type EnvConfigReader struct{}

func NewEnvConfigReader() *EnvConfigReader {
	// Имитация переменных окружения
	os.Setenv("DATABASE_HOST", "prod-db.example.com")
	os.Setenv("DATABASE_PORT", "3306")
	return &EnvConfigReader{}
}

func (r *EnvConfigReader) Get(key string) string {
	// Env-специфичная реализация
	// Но Application этого не знает - защищено интерфейсом
	return os.Getenv(key)
}

// Application ЗАЩИЩЕНО от вариаций источника конфигурации
type Application struct {
	configReader ConfigReader	// зависит от стабильного интерфейса
}

func NewApplication(configReader ConfigReader) *Application {
	return &Application{
		configReader: configReader,	// принимает любой ConfigReader
	}
}

func (a *Application) GetConfig(key string) string {
	// Код Application стабилен и никогда не меняется
	// Даже когда источники конфигурации меняются (JSON -> Env -> YAML -> Database)
	// Это Защищённые вариации!
	return a.configReader.Get(key)
}`
		},
		uz: {
			title: 'Protected Variations (Himoyalangan o\'zgarishlar)',
			description: `Protected Variations prinsipini amalga oshiring — elementlarni boshqa elementlardagi o'zgarishlardan barqaror interfeyslar bilan o'rab himoya qiling.

**Siz amalga oshirasiz:**

1. **ConfigReader interface** — Konfiguratsiyani o'qish uchun barqaror interfeys
2. **JSONConfigReader struct** — JSON fayllaridan o'qiydi
3. **EnvConfigReader struct** — Muhit o'zgaruvchilaridan o'qiydi
4. **Application struct** — ConfigReader interfeysidan foydalanadi (konfiguratsiya formati o'zgarishlaridan himoyalangan!)

**Asosiy tushunchalar:**
- **Protected Variations**: Barqaror interfeys implementatsiya o'zgarishlaridan himoya qiladi
- **Ochiq/Yopiq printsipi**: Kengaytirish uchun ochiq, o'zgartirish uchun yopiq
- **Inkapsulyatsiya**: O'zgaruvchan implementatsiya tafsilotlarini yashiradi

**Nima uchun Protected Variations?**
- **Barqarorlik**: Bir qismdagi o'zgarishlar boshqalarga ta'sir qilmaydi
- **Moslashuvchanlik**: Yangi implementatsiyalarni osongina qo'shish
- **Evolyutsiya**: Tizim mavjud kodni buzmasdan rivojlanishi mumkin

**Cheklovlar:**
- Application konkret konfiguratsiya formatlari haqida bilmasligi kerak
- Barcha konfiguratsiya o'quvchilari ConfigReader interfeysini amalga oshirishi kerak
- Interfeys turli implementatsiyalarga qaramay barqaror shartnomani taqdim etadi`,
			hint1: `ConfigReader interfeysini Get(key string) string bilan aniqlang. JSONConfigReader va EnvConfigReader ikkalasi ham bu metodini amalga oshirishi kerak.`,
			hint2: `JSONConfigReader.Get r.data[key] ni qaytaradi. EnvConfigReader.Get os.Getenv(key) ni qaytaradi. Application.GetConfig a.configReader.Get(key) ni chaqiradi.`,
			whyItMatters: `Protected Variations eng umumiy GRASP printsipi - u barqaror interfeyslar yordamida o'zgarishlar ta'siridan himoya qiladi.

**Protected Variations nima uchun muhim:**

**1. Tashqi o'zgarishlardan himoya qiladi**
Tashqi tizimlar o'zgarganda, barqaror interfeys kodingizni himoya qiladi.

**Umumiy xatolar:**
- Interfeyslarni juda erta yaratish (YAGNI)
- Implementatsiya tafsilotlarini ochiib beradigan interfeyslar
- Juda ko'p kichik interfeyslar (ortiqcha muhandislik)
- Ehtimol o'zgarishlardan himoya qilmaslik`,
			solutionCode: `package principles

import (
	"os"
)

// ConfigReader interfeysi - o'zgarishlardan himoya qiluvchi BARQAROR SHARTNOMA
type ConfigReader interface {
	Get(key string) string
}

// JSONConfigReader - bitta o'zgarish (JSON fayllari)
type JSONConfigReader struct {
	filename string
	data     map[string]string
}

func NewJSONConfigReader(filename string) *JSONConfigReader {
	reader := &JSONConfigReader{
		filename: filename,
		data:     make(map[string]string),
	}
	// JSON faylini o'qishni simulyatsiya qilish
	reader.data["database.host"] = "localhost"
	reader.data["database.port"] = "5432"
	return reader
}

func (r *JSONConfigReader) Get(key string) string {
	// JSON-ga xos implementatsiya
	// Lekin Application buni bilmaydi - interfeys orqali himoyalangan
	return r.data[key]
}

// EnvConfigReader - boshqa o'zgarish (muhit o'zgaruvchilari)
type EnvConfigReader struct{}

func NewEnvConfigReader() *EnvConfigReader {
	// Muhit o'zgaruvchilarini simulyatsiya qilish
	os.Setenv("DATABASE_HOST", "prod-db.example.com")
	os.Setenv("DATABASE_PORT", "3306")
	return &EnvConfigReader{}
}

func (r *EnvConfigReader) Get(key string) string {
	// Env-ga xos implementatsiya
	// Lekin Application buni bilmaydi - interfeys orqali himoyalangan
	return os.Getenv(key)
}

// Application konfiguratsiya manbai o'zgarishlaridan HIMOYALANGAN
type Application struct {
	configReader ConfigReader	// barqaror interfeysga bog'liq
}

func NewApplication(configReader ConfigReader) *Application {
	return &Application{
		configReader: configReader,	// har qanday ConfigReader ni qabul qiladi
	}
}

func (a *Application) GetConfig(key string) string {
	// Application kodi barqaror va hech qachon o'zgarmaydi
	// Hatto konfiguratsiya manbalari o'zgarganda ham (JSON -> Env -> YAML -> Database)
	// Bu Protected Variations!
	return a.configReader.Get(key)
}`
		}
	}
};

export default task;
