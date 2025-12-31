import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'api-design-http-methods',
	title: 'HTTP Methods - CRUD Operations',
	difficulty: 'medium',
	tags: ['api-design', 'http', 'go', 'crud'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement proper HTTP methods (GET, POST, PUT, DELETE) for RESTful CRUD operations.

**You will implement:**

1. **CreateProduct()** - POST endpoint to create new product
2. **UpdateProduct()** - PUT endpoint to update existing product
3. **DeleteProduct()** - DELETE endpoint to remove product
4. **Idempotency** - Ensure PUT and DELETE are idempotent

**Key Concepts:**
- **GET**: Retrieve resources (safe, idempotent, cacheable)
- **POST**: Create new resources (not idempotent)
- **PUT**: Update/replace resources (idempotent)
- **DELETE**: Remove resources (idempotent)
- **Safe Methods**: GET doesn't modify server state
- **Idempotent**: Multiple identical requests have same effect as single request

**Example Usage:**

\`\`\`go
// GET - Read (safe, idempotent, cacheable)
GET /api/products/1           // Always returns same product

// POST - Create (NOT idempotent)
POST /api/products            // Each call creates new product
Body: {"name": "Widget"}
Response: 201 Created, Location: /api/products/123

// PUT - Update (idempotent)
PUT /api/products/123         // Same result if called once or 100 times
Body: {"name": "Updated Widget", "price": 19.99}
Response: 200 OK

// DELETE - Remove (idempotent)
DELETE /api/products/123      // First call: 204 No Content
DELETE /api/products/123      // Subsequent calls: 404 Not Found (already deleted)
\`\`\`

**HTTP Method Properties:**
- **Safe**: GET, HEAD, OPTIONS (read-only)
- **Idempotent**: GET, PUT, DELETE (same result when repeated)
- **Not Idempotent**: POST, PATCH (creates new resource each time)
- **Cacheable**: GET, HEAD (responses can be cached)

**Constraints:**
- POST must return 201 Created with Location header
- PUT must replace entire resource
- DELETE must return 204 No Content on success
- All methods must handle errors properly`,
	initialCode: `package api

import (
	"encoding/json"
	"net/http"
	"strconv"
	"sync"

	"github.com/go-chi/chi/v5"
)

type Product struct {
	ID    int     \`json:"id"\`
	Name  string  \`json:"name"\`
	Price float64 \`json:"price"\`
}

var (
	products   = make(map[int]Product)
	productsMu sync.RWMutex
	nextID     = 1
)

func init() {
	products[1] = Product{ID: 1, Name: "Laptop", Price: 999.99}
	products[2] = Product{ID: 2, Name: "Mouse", Price: 29.99}
	nextID = 3
}

// Already implemented
func GetProduct(w http.ResponseWriter, r *http.Request) {
	productsMu.RLock()
	defer productsMu.RUnlock()

	idStr := chi.URLParam(r, "id")
	id, _ := strconv.Atoi(idStr)

	product, exists := products[id]
	if !exists {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"error": "Product not found"})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(product)
}

// TODO: Implement CreateProduct
// Handle POST request to create new product
// 1. Decode JSON body into Product struct
// 2. Assign new ID (use nextID and increment)
// 3. Store in products map (use write lock)
// 4. Return 201 Created status
// 5. Set Location header: /api/products/{id}
// 6. Return created product in response body
func CreateProduct(w http.ResponseWriter, r *http.Request) {
	panic("TODO: implement CreateProduct")
}

// TODO: Implement UpdateProduct
// Handle PUT request to update existing product
// 1. Extract product ID from URL
// 2. Decode JSON body
// 3. Check if product exists (404 if not)
// 4. Replace entire product (PUT replaces, not merges)
// 5. Keep the same ID
// 6. Return 200 OK with updated product
// PUT is idempotent - same request multiple times = same result
func UpdateProduct(w http.ResponseWriter, r *http.Request) {
	panic("TODO: implement UpdateProduct")
}

// TODO: Implement DeleteProduct
// Handle DELETE request to remove product
// 1. Extract product ID from URL
// 2. Check if product exists
// 3. Delete from map
// 4. Return 204 No Content (no body)
// 5. If already deleted, return 404 Not Found
// DELETE is idempotent - deleting twice is safe
func DeleteProduct(w http.ResponseWriter, r *http.Request) {
	panic("TODO: implement DeleteProduct")
}`,
	solutionCode: `package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"sync"

	"github.com/go-chi/chi/v5"
)

type Product struct {
	ID    int     \`json:"id"\`
	Name  string  \`json:"name"\`
	Price float64 \`json:"price"\`
}

var (
	products   = make(map[int]Product)  // In-memory store (use database in production)
	productsMu sync.RWMutex             // Protects concurrent access
	nextID     = 1                       // Auto-increment ID
)

func init() {
	products[1] = Product{ID: 1, Name: "Laptop", Price: 999.99}
	products[2] = Product{ID: 2, Name: "Mouse", Price: 29.99}
	nextID = 3
}

func GetProduct(w http.ResponseWriter, r *http.Request) {
	productsMu.RLock()  // Read lock - multiple readers allowed
	defer productsMu.RUnlock()

	idStr := chi.URLParam(r, "id")
	id, _ := strconv.Atoi(idStr)

	product, exists := products[id]
	if !exists {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"error": "Product not found"})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(product)
}

func CreateProduct(w http.ResponseWriter, r *http.Request) {
	var product Product

	// Decode JSON request body
	if err := json.NewDecoder(r.Body).Decode(&product); err != nil {
		w.WriteHeader(http.StatusBadRequest)  // 400 for malformed JSON
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid JSON"})
		return
	}

	productsMu.Lock()  // Write lock - exclusive access
	product.ID = nextID  // Assign new ID
	nextID++             // Increment for next product
	products[product.ID] = product  // Store in map
	productsMu.Unlock()

	// 201 Created - resource successfully created
	w.WriteHeader(http.StatusCreated)
	// Location header tells client where to find new resource
	w.Header().Set("Location", fmt.Sprintf("/api/products/%d", product.ID))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(product)  // Return created resource
}

func UpdateProduct(w http.ResponseWriter, r *http.Request) {
	// Extract ID from URL path
	idStr := chi.URLParam(r, "id")
	id, err := strconv.Atoi(idStr)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid product ID"})
		return
	}

	var product Product
	if err := json.NewDecoder(r.Body).Decode(&product); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid JSON"})
		return
	}

	productsMu.Lock()
	defer productsMu.Unlock()

	// Check if product exists before updating
	if _, exists := products[id]; !exists {
		w.WriteHeader(http.StatusNotFound)  // 404 if resource doesn't exist
		json.NewEncoder(w).Encode(map[string]string{"error": "Product not found"})
		return
	}

	// PUT replaces entire resource - keep the URL's ID
	product.ID = id  // Ensure ID matches URL parameter
	products[id] = product  // Replace entire product

	// 200 OK - successful update
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(product)
	// Idempotent: calling this 10 times with same data = same result
}

func DeleteProduct(w http.ResponseWriter, r *http.Request) {
	idStr := chi.URLParam(r, "id")
	id, err := strconv.Atoi(idStr)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid product ID"})
		return
	}

	productsMu.Lock()
	defer productsMu.Unlock()

	// Check if product exists
	if _, exists := products[id]; !exists {
		w.WriteHeader(http.StatusNotFound)  // Already deleted or never existed
		json.NewEncoder(w).Encode(map[string]string{"error": "Product not found"})
		return
	}

	delete(products, id)  // Remove from map

	// 204 No Content - successful deletion, no response body needed
	w.WriteHeader(http.StatusNoContent)
	// Note: Don't write body after 204 status
	// Idempotent: first DELETE returns 204, subsequent ones return 404
}`,
	hint1: `For CreateProduct: decode JSON with json.NewDecoder(r.Body).Decode(&product), assign product.ID = nextID, increment nextID++, store in map, return 201 with Location header.`,
	hint2: `For UpdateProduct: get ID from URL, decode JSON, check product exists with _, exists := products[id], if exists replace entire product (product.ID = id), return 200. For DeleteProduct: check exists, use delete(products, id), return 204 (no body).`,
	whyItMatters: `Proper HTTP method usage is fundamental to building RESTful APIs that are predictable, cacheable, and safe to use.

**Why HTTP Methods Matter:**

**1. Semantic Clarity**
Each method has a specific meaning that developers understand:

\`\`\`go
// Clear intent from method alone
GET    /api/products/1    // "Show me product 1" (read-only)
POST   /api/products      // "Create a new product"
PUT    /api/products/1    // "Replace product 1 with this data"
DELETE /api/products/1    // "Remove product 1"

// BAD: Using GET for everything
GET /api/createProduct?name=Widget    // Using GET to modify state!
GET /api/deleteProduct/1              // Dangerous - GET should be safe
GET /api/updateProduct/1?price=99     // Loses HTTP semantics
\`\`\`

**2. Safety - GET Must Not Modify State**

\`\`\`go
// ✅ SAFE: GET doesn't change anything
GET /api/products/1              // Read-only, can be called repeatedly

// ❌ DANGEROUS: GET modifying state
GET /api/deleteProduct/1         // Web crawlers could delete your data!
GET /api/incrementCounter        // Prefetching causes unintended increments
GET /api/sendEmail?to=user@...   // Browser preloading sends emails!

// Real incident: Google Web Accelerator (2005)
// Crawled GET links with ?action=delete, destroyed user data
// Because developers used GET for destructive operations
\`\`\`

**3. Idempotency - Critical for Reliability**

\`\`\`go
// IDEMPOTENT methods (same result when repeated):

// GET - Always returns same resource state
GET /api/products/1              // Call 100 times = same result

// PUT - Replace entire resource
PUT /api/products/1              // Set price = $99
PUT /api/products/1              // Set price = $99 again
PUT /api/products/1              // Still price = $99 (idempotent!)

// DELETE - Remove resource
DELETE /api/products/1           // Product deleted (204 No Content)
DELETE /api/products/1           // Already gone (404 Not Found)
DELETE /api/products/1           // Still gone (404 Not Found)
// Result is the same: product doesn't exist

// NOT IDEMPOTENT:

// POST - Creates new resource each time
POST /api/products               // Creates product ID 100
POST /api/products               // Creates product ID 101
POST /api/products               // Creates product ID 102
// Each call has different result (not idempotent)
\`\`\`

**4. Why Idempotency Matters - Network Reliability**

\`\`\`go
// Scenario: Client makes request but network times out
// Did the server receive it? Should client retry?

// WITH IDEMPOTENT METHODS (PUT, DELETE):
func updateProduct(productId int, data Product) error {
    // Network timeout on first try
    err := httpClient.Put("/api/products/1", data)  // Timeout!

    // SAFE TO RETRY - idempotent method
    err = httpClient.Put("/api/products/1", data)   // Retry
    // Result is same whether first call succeeded or not
    return err
}

// WITH NON-IDEMPOTENT METHOD (POST):
func createProduct(data Product) error {
    err := httpClient.Post("/api/products", data)   // Timeout!

    // DANGEROUS TO RETRY - might create duplicate
    err = httpClient.Post("/api/products", data)    // Creates another product?
    // Need idempotency key or check if product exists first
}

// Solution for POST - use idempotency keys:
func createProductSafely(data Product, idempotencyKey string) {
    headers := map[string]string{
        "Idempotency-Key": idempotencyKey,  // UUID from client
    }
    // Server deduplicates requests with same key
    httpClient.Post("/api/products", data, headers)
}
\`\`\`

**5. HTTP Method Properties Table**

\`\`\`
Method  | Safe | Idempotent | Cacheable | Has Body
--------|------|------------|-----------|----------
GET     | Yes  | Yes        | Yes       | No (request)
POST    | No   | No         | No        | Yes
PUT     | No   | Yes        | No        | Yes
DELETE  | No   | Yes        | No        | No
PATCH   | No   | No         | No        | Yes
HEAD    | Yes  | Yes        | Yes       | No
OPTIONS | Yes  | Yes        | No        | No
\`\`\`

**6. Real-World Example - Stripe API**

\`\`\`go
// Stripe uses idempotency keys for POST requests
POST /v1/charges
Headers:
  Idempotency-Key: unique-key-123
Body: {
  amount: 1000,
  currency: "usd"
}

// If network fails, client retries with SAME idempotency key
// Stripe returns the SAME charge, doesn't create duplicate
// Makes non-idempotent POST behave like idempotent operation
\`\`\`

**7. Common Mistakes**

\`\`\`go
// ❌ MISTAKE 1: Using GET for mutations
GET /api/users/1/activate        // Should be POST or PUT

// ❌ MISTAKE 2: Returning 200 for DELETE
func DeleteProduct(w http.ResponseWriter, r *http.Request) {
    delete(products, id)
    w.WriteHeader(http.StatusOK)  // Wrong! Should be 204
    json.NewEncoder(w).Encode(map[string]string{"message": "Deleted"})
}
// ✅ CORRECT: 204 No Content (no body)
func DeleteProduct(w http.ResponseWriter, r *http.Request) {
    delete(products, id)
    w.WriteHeader(http.StatusNoContent)  // No body!
}

// ❌ MISTAKE 3: PUT without full replacement
func UpdateProduct(w http.ResponseWriter, r *http.Request) {
    var partial Product
    json.NewDecoder(r.Body).Decode(&partial)
    existing := products[id]
    // Only update provided fields
    if partial.Name != "" {
        existing.Name = partial.Name  // This is PATCH behavior!
    }
}
// ✅ CORRECT: PUT replaces entire resource
func UpdateProduct(w http.ResponseWriter, r *http.Request) {
    var product Product
    json.NewDecoder(r.Body).Decode(&product)
    product.ID = id
    products[id] = product  // Complete replacement
}

// ❌ MISTAKE 4: POST without Location header
func CreateProduct(w http.ResponseWriter, r *http.Request) {
    // Create product...
    w.WriteHeader(http.StatusCreated)  // Missing Location header!
    json.NewEncoder(w).Encode(product)
}
// ✅ CORRECT: Include Location header
func CreateProduct(w http.ResponseWriter, r *http.Request) {
    // Create product...
    w.WriteHeader(http.StatusCreated)
    w.Header().Set("Location", fmt.Sprintf("/api/products/%d", product.ID))
    json.NewEncoder(w).Encode(product)
}
\`\`\`

**8. Benefits of Proper Method Usage**

- **Caching**: GET responses can be cached by browsers, CDNs, proxies
- **Retry Safety**: Idempotent methods can be safely retried on network failure
- **Browser Behavior**: Browsers handle GET differently (prefetching, bookmarking)
- **API Gateways**: Can apply rate limits, caching rules based on methods
- **Monitoring**: Track create/read/update/delete separately in metrics
- **Security**: CSRF protection not needed for GET (since it's safe)`,
	order: 2,
	testCode: `package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/go-chi/chi/v5"
)

// Test1: CreateProduct returns 201 and sets Location header
func Test1(t *testing.T) {
	body := bytes.NewBufferString(\`{"name": "Widget", "price": 9.99}\`)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/api/products", body)
	CreateProduct(w, r)
	if w.Code != http.StatusCreated {
		t.Errorf("Expected 201, got %d", w.Code)
	}
	location := w.Header().Get("Location")
	if location == "" {
		t.Error("Location header should be set for POST")
	}
}

// Test2: CreateProduct returns 400 for invalid JSON
func Test2(t *testing.T) {
	body := bytes.NewBufferString(\`invalid json\`)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/api/products", body)
	CreateProduct(w, r)
	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected 400, got %d", w.Code)
	}
}

// Test3: CreateProduct assigns new ID and increments for each product
func Test3(t *testing.T) {
	body1 := bytes.NewBufferString(\`{"name": "Product1", "price": 1.99}\`)
	w1 := httptest.NewRecorder()
	r1 := httptest.NewRequest("POST", "/api/products", body1)
	CreateProduct(w1, r1)
	var p1 Product
	json.NewDecoder(w1.Body).Decode(&p1)

	body2 := bytes.NewBufferString(\`{"name": "Product2", "price": 2.99}\`)
	w2 := httptest.NewRecorder()
	r2 := httptest.NewRequest("POST", "/api/products", body2)
	CreateProduct(w2, r2)
	var p2 Product
	json.NewDecoder(w2.Body).Decode(&p2)

	if p2.ID <= p1.ID {
		t.Error("Second product should have higher ID - POST is not idempotent")
	}
}

// Test4: UpdateProduct returns 200 for existing product
func Test4(t *testing.T) {
	router := chi.NewRouter()
	router.Put("/api/products/{id}", UpdateProduct)
	body := bytes.NewBufferString(\`{"name": "Updated", "price": 19.99}\`)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("PUT", "/api/products/1", body)
	router.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Errorf("Expected 200, got %d", w.Code)
	}
}

// Test5: UpdateProduct returns 404 for non-existent product
func Test5(t *testing.T) {
	router := chi.NewRouter()
	router.Put("/api/products/{id}", UpdateProduct)
	body := bytes.NewBufferString(\`{"name": "Updated", "price": 19.99}\`)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("PUT", "/api/products/999", body)
	router.ServeHTTP(w, r)
	if w.Code != http.StatusNotFound {
		t.Errorf("Expected 404, got %d", w.Code)
	}
}

// Test6: UpdateProduct replaces entire resource and keeps URL ID
func Test6(t *testing.T) {
	router := chi.NewRouter()
	router.Put("/api/products/{id}", UpdateProduct)
	body := bytes.NewBufferString(\`{"name": "Completely New", "price": 99.99}\`)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("PUT", "/api/products/1", body)
	router.ServeHTTP(w, r)
	var p Product
	json.NewDecoder(w.Body).Decode(&p)
	if p.Name != "Completely New" || p.Price != 99.99 {
		t.Error("PUT should replace entire resource")
	}
	if p.ID != 1 {
		t.Error("Product ID should match URL parameter")
	}
}

// Test7: UpdateProduct returns 400 for invalid JSON
func Test7(t *testing.T) {
	router := chi.NewRouter()
	router.Put("/api/products/{id}", UpdateProduct)
	body := bytes.NewBufferString(\`invalid json\`)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("PUT", "/api/products/1", body)
	router.ServeHTTP(w, r)
	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected 400, got %d", w.Code)
	}
}

// Test8: DeleteProduct returns 204 No Content for existing product
func Test8(t *testing.T) {
	router := chi.NewRouter()
	router.Delete("/api/products/{id}", DeleteProduct)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("DELETE", "/api/products/2", nil)
	router.ServeHTTP(w, r)
	if w.Code != http.StatusNoContent {
		t.Errorf("Expected 204, got %d", w.Code)
	}
}

// Test9: DeleteProduct is idempotent - second delete returns 404
func Test9(t *testing.T) {
	router := chi.NewRouter()
	router.Delete("/api/products/{id}", DeleteProduct)

	// First delete
	w1 := httptest.NewRecorder()
	r1 := httptest.NewRequest("DELETE", "/api/products/1", nil)
	router.ServeHTTP(w1, r1)

	// Second delete - should return 404 (idempotent behavior)
	w2 := httptest.NewRecorder()
	r2 := httptest.NewRequest("DELETE", "/api/products/1", nil)
	router.ServeHTTP(w2, r2)

	if w2.Code != http.StatusNotFound {
		t.Errorf("Second DELETE should return 404, got %d", w2.Code)
	}
}

// Test10: DeleteProduct returns 400 for invalid ID format
func Test10(t *testing.T) {
	router := chi.NewRouter()
	router.Delete("/api/products/{id}", DeleteProduct)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("DELETE", "/api/products/invalid", nil)
	router.ServeHTTP(w, r)
	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected 400 for invalid ID, got %d", w.Code)
	}
}
`,
	translations: {
		ru: {
			title: 'HTTP методы - CRUD операции',
			description: `Реализуйте правильные HTTP методы (GET, POST, PUT, DELETE) для RESTful CRUD операций.

**Вы реализуете:**

1. **CreateProduct()** - POST эндпоинт для создания нового продукта
2. **UpdateProduct()** - PUT эндпоинт для обновления существующего продукта
3. **DeleteProduct()** - DELETE эндпоинт для удаления продукта
4. **Идемпотентность** - Убедитесь что PUT и DELETE идемпотентны

**Ключевые концепции:**
- **GET**: Получение ресурсов (безопасен, идемпотентен, кэшируемый)
- **POST**: Создание новых ресурсов (не идемпотентен)
- **PUT**: Обновление/замена ресурсов (идемпотентен)
- **DELETE**: Удаление ресурсов (идемпотентен)
- **Безопасные методы**: GET не изменяет состояние сервера
- **Идемпотентность**: Множественные идентичные запросы имеют тот же эффект что и один

**Свойства HTTP методов:**
- **Безопасные**: GET, HEAD, OPTIONS (только чтение)
- **Идемпотентные**: GET, PUT, DELETE (одинаковый результат при повторении)
- **Не идемпотентные**: POST, PATCH (создаёт новый ресурс каждый раз)
- **Кэшируемые**: GET, HEAD (ответы могут быть закэшированы)

**Ограничения:**
- POST должен возвращать 201 Created с Location заголовком
- PUT должен заменять весь ресурс
- DELETE должен возвращать 204 No Content при успехе`,
			hint1: `Для CreateProduct: декодируйте JSON через json.NewDecoder(r.Body).Decode(&product), назначьте product.ID = nextID, инкрементируйте nextID++, сохраните в map, верните 201 с Location заголовком.`,
			hint2: `Для UpdateProduct: получите ID из URL, декодируйте JSON, проверьте что продукт существует, замените весь продукт, верните 200. Для DeleteProduct: проверьте существование, используйте delete(products, id), верните 204 (без тела).`,
			whyItMatters: `Правильное использование HTTP методов фундаментально для построения RESTful API которые предсказуемы, кэшируемы и безопасны в использовании.

**Почему HTTP методы важны:**

**1. Семантическая ясность**
Каждый метод имеет конкретное значение:

\`\`\`go
// Ясное намерение из одного метода
GET    /api/products/1    // "Покажи мне продукт 1" (только чтение)
POST   /api/products      // "Создай новый продукт"
PUT    /api/products/1    // "Замени продукт 1 этими данными"
DELETE /api/products/1    // "Удали продукт 1"
\`\`\`

**2. Безопасность - GET не должен изменять состояние**

\`\`\`go
// ✅ БЕЗОПАСНО: GET ничего не меняет
GET /api/products/1              // Только чтение

// ❌ ОПАСНО: GET изменяет состояние
GET /api/deleteProduct/1         // Веб-краулеры могут удалить данные!
\`\`\`

**Лучшие практики:**
- Используйте правильные HTTP методы для каждой операции
- GET должен быть безопасным (read-only)
- PUT и DELETE должны быть идемпотентными
- POST возвращает 201 Created с Location заголовком
- DELETE возвращает 204 No Content`,
			solutionCode: `package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"sync"

	"github.com/go-chi/chi/v5"
)

type Product struct {
	ID    int     \`json:"id"\`
	Name  string  \`json:"name"\`
	Price float64 \`json:"price"\`
}

var (
	products   = make(map[int]Product)
	productsMu sync.RWMutex
	nextID     = 1
)

func init() {
	products[1] = Product{ID: 1, Name: "Laptop", Price: 999.99}
	products[2] = Product{ID: 2, Name: "Mouse", Price: 29.99}
	nextID = 3
}

func GetProduct(w http.ResponseWriter, r *http.Request) {
	productsMu.RLock()
	defer productsMu.RUnlock()

	idStr := chi.URLParam(r, "id")
	id, _ := strconv.Atoi(idStr)

	product, exists := products[id]
	if !exists {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"error": "Product not found"})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(product)
}

func CreateProduct(w http.ResponseWriter, r *http.Request) {
	var product Product

	if err := json.NewDecoder(r.Body).Decode(&product); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid JSON"})
		return
	}

	productsMu.Lock()
	product.ID = nextID
	nextID++
	products[product.ID] = product
	productsMu.Unlock()

	w.WriteHeader(http.StatusCreated)
	w.Header().Set("Location", fmt.Sprintf("/api/products/%d", product.ID))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(product)
}

func UpdateProduct(w http.ResponseWriter, r *http.Request) {
	idStr := chi.URLParam(r, "id")
	id, err := strconv.Atoi(idStr)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid product ID"})
		return
	}

	var product Product
	if err := json.NewDecoder(r.Body).Decode(&product); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid JSON"})
		return
	}

	productsMu.Lock()
	defer productsMu.Unlock()

	if _, exists := products[id]; !exists {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"error": "Product not found"})
		return
	}

	product.ID = id
	products[id] = product

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(product)
}

func DeleteProduct(w http.ResponseWriter, r *http.Request) {
	idStr := chi.URLParam(r, "id")
	id, err := strconv.Atoi(idStr)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid product ID"})
		return
	}

	productsMu.Lock()
	defer productsMu.Unlock()

	if _, exists := products[id]; !exists {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"error": "Product not found"})
		return
	}

	delete(products, id)
	w.WriteHeader(http.StatusNoContent)
}`
		},
		uz: {
			title: 'HTTP metodlar - CRUD operatsiyalar',
			description: `RESTful CRUD operatsiyalar uchun to\'g\'ri HTTP metodlarni (GET, POST, PUT, DELETE) amalga oshiring.

**Siz amalga oshirasiz:**

1. **CreateProduct()** - Yangi mahsulot yaratish uchun POST endpoint
2. **UpdateProduct()** - Mavjud mahsulotni yangilash uchun PUT endpoint
3. **DeleteProduct()** - Mahsulotni o\'chirish uchun DELETE endpoint
4. **Idempotentlik** - PUT va DELETE idempotent ekanligiga ishonch hosil qiling

**Asosiy tushunchalar:**
- **GET**: Resurslarni olish (xavfsiz, idempotent, keshlash mumkin)
- **POST**: Yangi resurslar yaratish (idempotent emas)
- **PUT**: Resurslarni yangilash/almashtirish (idempotent)
- **DELETE**: Resurslarni o\'chirish (idempotent)
- **Xavfsiz metodlar**: GET server holatini o\'zgartirmaydi
- **Idempotentlik**: Bir nechta bir xil so\'rovlar bitta so\'rov bilan bir xil ta\'sirga ega

**HTTP metod xususiyatlari:**
- **Xavfsiz**: GET, HEAD, OPTIONS (faqat o\'qish)
- **Idempotent**: GET, PUT, DELETE (takrorlanganda bir xil natija)
- **Idempotent emas**: POST, PATCH (har safar yangi resurs yaratadi)
- **Keshlash mumkin**: GET, HEAD (javoblar keshlanishi mumkin)

**Cheklovlar:**
- POST 201 Created va Location headerini qaytarishi kerak
- PUT butun resursni almashtirishi kerak
- DELETE muvaffaqiyatli bo\'lganda 204 No Content qaytarishi kerak`,
			hint1: `CreateProduct uchun: JSON ni json.NewDecoder(r.Body).Decode(&product) bilan dekodirlang, product.ID = nextID ni tayinlang, nextID++ ni oshiring, map ga saqlang, Location header bilan 201 qaytaring.`,
			hint2: `UpdateProduct uchun: URL dan ID ni oling, JSON ni dekodirlang, mahsulot mavjudligini tekshiring, butun mahsulotni almashtiring, 200 qaytaring. DeleteProduct uchun: mavjudligini tekshiring, delete(products, id) dan foydalaning, 204 qaytaring (body yo\'q).`,
			whyItMatters: `To\'g\'ri HTTP metodlardan foydalanish bashorat qilinadigan, keshlanadigan va xavfsiz ishlatish uchun RESTful API larni qurishning asosi.

**HTTP metodlar nima uchun muhim:**

**1. Semantik aniqlik**
Har bir metod ma\'lum ma\'noga ega:

\`\`\`go
// Metoddan aniq niyat
GET    /api/products/1    // "1-mahsulotni ko\'rsat" (faqat o\'qish)
POST   /api/products      // "Yangi mahsulot yarat"
PUT    /api/products/1    // "1-mahsulotni bu ma\'lumotlar bilan almashtir"
DELETE /api/products/1    // "1-mahsulotni o\'chir"
\`\`\`

**2. Xavfsizlik - GET holatni o\'zgartirmasligi kerak**

\`\`\`go
// ✅ XAVFSIZ: GET hech narsani o\'zgartirmaydi
GET /api/products/1              // Faqat o\'qish

// ❌ XAVFLI: GET holatni o\'zgartiradi
GET /api/deleteProduct/1         // Veb-kroulerlar ma\'lumotlarni o\'chirishi mumkin!
\`\`\`

**Eng yaxshi amaliyotlar:**
- Har bir operatsiya uchun to\'g\'ri HTTP metodlaridan foydalaning
- GET xavfsiz bo\'lishi kerak (read-only)
- PUT va DELETE idempotent bo\'lishi kerak
- POST 201 Created va Location headerni qaytaradi
- DELETE 204 No Content qaytaradi`,
			solutionCode: `package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"sync"

	"github.com/go-chi/chi/v5"
)

type Product struct {
	ID    int     \`json:"id"\`
	Name  string  \`json:"name"\`
	Price float64 \`json:"price"\`
}

var (
	products   = make(map[int]Product)
	productsMu sync.RWMutex
	nextID     = 1
)

func init() {
	products[1] = Product{ID: 1, Name: "Laptop", Price: 999.99}
	products[2] = Product{ID: 2, Name: "Mouse", Price: 29.99}
	nextID = 3
}

func GetProduct(w http.ResponseWriter, r *http.Request) {
	productsMu.RLock()
	defer productsMu.RUnlock()

	idStr := chi.URLParam(r, "id")
	id, _ := strconv.Atoi(idStr)

	product, exists := products[id]
	if !exists {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"error": "Product not found"})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(product)
}

func CreateProduct(w http.ResponseWriter, r *http.Request) {
	var product Product

	if err := json.NewDecoder(r.Body).Decode(&product); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid JSON"})
		return
	}

	productsMu.Lock()
	product.ID = nextID
	nextID++
	products[product.ID] = product
	productsMu.Unlock()

	w.WriteHeader(http.StatusCreated)
	w.Header().Set("Location", fmt.Sprintf("/api/products/%d", product.ID))
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(product)
}

func UpdateProduct(w http.ResponseWriter, r *http.Request) {
	idStr := chi.URLParam(r, "id")
	id, err := strconv.Atoi(idStr)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid product ID"})
		return
	}

	var product Product
	if err := json.NewDecoder(r.Body).Decode(&product); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid JSON"})
		return
	}

	productsMu.Lock()
	defer productsMu.Unlock()

	if _, exists := products[id]; !exists {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"error": "Product not found"})
		return
	}

	product.ID = id
	products[id] = product

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(product)
}

func DeleteProduct(w http.ResponseWriter, r *http.Request) {
	idStr := chi.URLParam(r, "id")
	id, err := strconv.Atoi(idStr)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid product ID"})
		return
	}

	productsMu.Lock()
	defer productsMu.Unlock()

	if _, exists := products[id]; !exists {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"error": "Product not found"})
		return
	}

	delete(products, id)
	w.WriteHeader(http.StatusNoContent)
}`
		}
	}
};

export default task;
