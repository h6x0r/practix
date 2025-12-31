import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'api-design-response-formatting',
	title: 'Response Formatting',
	difficulty: 'easy',
	tags: ['api-design', 'response', 'go'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement consistent response formatting with pagination and metadata.

**You will implement:**

1. **PaginatedResponse** - Standard response structure with pagination
2. **FormatListResponse()** - Format list responses consistently
3. **Metadata** - Include pagination info (total, page, pageSize)

**Response Structure:**

\`\`\`json
{
  "data": [...],
  "meta": {
    "total": 100,
    "page": 1,
    "pageSize": 10,
    "totalPages": 10
  }
}
\`\`\`

**Best Practices:**
- Use consistent structure across all endpoints
- Include pagination metadata
- Wrap data in "data" field
- Add timestamps when relevant`,
	initialCode: `package api

import (
	"encoding/json"
	"net/http"
)

type PaginatedResponse struct {
	Data interface{} \`json:"data"\`
	Meta Metadata    \`json:"meta"\`
}

type Metadata struct {
	Total      int \`json:"total"\`
	Page       int \`json:"page"\`
	PageSize   int \`json:"pageSize"\`
	TotalPages int \`json:"totalPages"\`
}

type Product struct {
	ID    int     \`json:"id"\`
	Name  string  \`json:"name"\`
	Price float64 \`json:"price"\`
}

var allProducts = []Product{
	{ID: 1, Name: "Laptop", Price: 999.99},
	{ID: 2, Name: "Mouse", Price: 29.99},
	{ID: 3, Name: "Keyboard", Price: 79.99},
	{ID: 4, Name: "Monitor", Price: 299.99},
	{ID: 5, Name: "Headphones", Price: 149.99},
}

// TODO: Implement FormatListResponse
// Create paginated response with:
// 1. Calculate totalPages = ceil(total / pageSize)
// 2. Slice data for current page
// 3. Build PaginatedResponse with data and metadata
// 4. Return as JSON with 200 status
func FormatListResponse(w http.ResponseWriter, r *http.Request, page, pageSize int) {
	panic("TODO: implement FormatListResponse")
}`,
	solutionCode: `package api

import (
	"encoding/json"
	"math"
	"net/http"
)

type PaginatedResponse struct {
	Data interface{} \`json:"data"\`
	Meta Metadata    \`json:"meta"\`
}

type Metadata struct {
	Total      int \`json:"total"\`
	Page       int \`json:"page"\`
	PageSize   int \`json:"pageSize"\`
	TotalPages int \`json:"totalPages"\`
}

type Product struct {
	ID    int     \`json:"id"\`
	Name  string  \`json:"name"\`
	Price float64 \`json:"price"\`
}

var allProducts = []Product{
	{ID: 1, Name: "Laptop", Price: 999.99},
	{ID: 2, Name: "Mouse", Price: 29.99},
	{ID: 3, Name: "Keyboard", Price: 79.99},
	{ID: 4, Name: "Monitor", Price: 299.99},
	{ID: 5, Name: "Headphones", Price: 149.99},
}

func FormatListResponse(w http.ResponseWriter, r *http.Request, page, pageSize int) {
	total := len(allProducts)
	totalPages := int(math.Ceil(float64(total) / float64(pageSize)))

	// Calculate slice indices for pagination
	start := (page - 1) * pageSize
	end := start + pageSize
	if end > total {
		end = total
	}
	if start > total {
		start = total
	}

	// Get paginated data
	pageData := allProducts[start:end]

	// Build response
	response := PaginatedResponse{
		Data: pageData,
		Meta: Metadata{
			Total:      total,
			Page:       page,
			PageSize:   pageSize,
			TotalPages: totalPages,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(response)
}`,
	hint1: `Calculate totalPages using math.Ceil(float64(total) / float64(pageSize)). Calculate start = (page-1)*pageSize and end = start+pageSize, making sure end doesn't exceed total.`,
	hint2: `Slice allProducts[start:end] to get page data, then build PaginatedResponse with data and Metadata struct containing total, page, pageSize, and totalPages.`,
	whyItMatters: `Consistent response formatting makes your API predictable and easy to consume across all endpoints.

**Why Response Formatting Matters:**

**1. Predictability**
Clients can handle all responses the same way:

\`\`\`go
// With consistent format
type APIResponse struct {
    Data interface{}
    Meta Metadata
}

// Client code works for all endpoints
func handleResponse(resp APIResponse) {
    for _, item := range resp.Data {
        // Process item
    }
    // Always access pagination same way
    totalPages := resp.Meta.TotalPages
}
\`\`\`

**2. Pagination**
Include metadata for pagination:

\`\`\`json
{
  "data": [...],
  "meta": {
    "total": 100,
    "page": 2,
    "pageSize": 10,
    "totalPages": 10
  }
}
\`\`\`

**Best Practices:**
- Wrap data in "data" field
- Include pagination metadata
- Be consistent across all endpoints
- Add timestamps when relevant`,
	order: 5,
	testCode: `package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// Test1: FormatListResponse returns 200
func Test1(t *testing.T) {
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/products", nil)
	FormatListResponse(w, r, 1, 10)
	if w.Code != http.StatusOK {
		t.Errorf("Expected 200, got %d", w.Code)
	}
}

// Test2: Response has data field
func Test2(t *testing.T) {
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/products", nil)
	FormatListResponse(w, r, 1, 10)
	var result PaginatedResponse
	json.NewDecoder(w.Body).Decode(&result)
	if result.Data == nil {
		t.Error("Response should have data field")
	}
}

// Test3: Response has meta field
func Test3(t *testing.T) {
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/products", nil)
	FormatListResponse(w, r, 1, 10)
	var result PaginatedResponse
	json.NewDecoder(w.Body).Decode(&result)
	if result.Meta.Total == 0 && result.Meta.Page == 0 {
		t.Error("Response should have meta field")
	}
}

// Test4: Meta contains total count
func Test4(t *testing.T) {
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/products", nil)
	FormatListResponse(w, r, 1, 10)
	var result PaginatedResponse
	json.NewDecoder(w.Body).Decode(&result)
	if result.Meta.Total != 5 {
		t.Errorf("Expected total 5, got %d", result.Meta.Total)
	}
}

// Test5: Meta contains correct page
func Test5(t *testing.T) {
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/products", nil)
	FormatListResponse(w, r, 2, 2)
	var result PaginatedResponse
	json.NewDecoder(w.Body).Decode(&result)
	if result.Meta.Page != 2 {
		t.Errorf("Expected page 2, got %d", result.Meta.Page)
	}
}

// Test6: Meta contains pageSize
func Test6(t *testing.T) {
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/products", nil)
	FormatListResponse(w, r, 1, 3)
	var result PaginatedResponse
	json.NewDecoder(w.Body).Decode(&result)
	if result.Meta.PageSize != 3 {
		t.Errorf("Expected pageSize 3, got %d", result.Meta.PageSize)
	}
}

// Test7: Meta contains totalPages
func Test7(t *testing.T) {
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/products", nil)
	FormatListResponse(w, r, 1, 2)
	var result PaginatedResponse
	json.NewDecoder(w.Body).Decode(&result)
	if result.Meta.TotalPages != 3 {
		t.Errorf("Expected 3 totalPages for 5 items with pageSize 2, got %d", result.Meta.TotalPages)
	}
}

// Test8: Product struct has correct fields
func Test8(t *testing.T) {
	p := Product{ID: 1, Name: "Test", Price: 9.99}
	if p.ID != 1 || p.Name != "Test" || p.Price != 9.99 {
		t.Error("Product fields not set correctly")
	}
}

// Test9: Metadata struct has correct fields
func Test9(t *testing.T) {
	m := Metadata{Total: 10, Page: 1, PageSize: 5, TotalPages: 2}
	if m.Total != 10 || m.TotalPages != 2 {
		t.Error("Metadata fields not set correctly")
	}
}

// Test10: Content-Type is application/json
func Test10(t *testing.T) {
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/products", nil)
	FormatListResponse(w, r, 1, 10)
	ct := w.Header().Get("Content-Type")
	if ct != "application/json" {
		t.Errorf("Expected application/json, got %s", ct)
	}
}
`,
	translations: {
		ru: {
			title: 'Форматирование ответов',
			description: `Реализуйте согласованное форматирование ответов с пагинацией и метаданными.`,
			hint1: `Вычислите totalPages используя math.Ceil(float64(total) / float64(pageSize)). Вычислите start = (page-1)*pageSize и end = start+pageSize.`,
			hint2: `Срежьте allProducts[start:end] для получения данных страницы, затем постройте PaginatedResponse с данными и метаданными.`,
			whyItMatters: `Согласованное форматирование ответов делает ваш API предсказуемым и простым в использовании.`,
			solutionCode: `package api

import (
	"encoding/json"
	"math"
	"net/http"
)

type PaginatedResponse struct {
	Data interface{} \`json:"data"\`
	Meta Metadata    \`json:"meta"\`
}

type Metadata struct {
	Total      int \`json:"total"\`
	Page       int \`json:"page"\`
	PageSize   int \`json:"pageSize"\`
	TotalPages int \`json:"totalPages"\`
}

type Product struct {
	ID    int     \`json:"id"\`
	Name  string  \`json:"name"\`
	Price float64 \`json:"price"\`
}

var allProducts = []Product{
	{ID: 1, Name: "Laptop", Price: 999.99},
	{ID: 2, Name: "Mouse", Price: 29.99},
	{ID: 3, Name: "Keyboard", Price: 79.99},
	{ID: 4, Name: "Monitor", Price: 299.99},
	{ID: 5, Name: "Headphones", Price: 149.99},
}

func FormatListResponse(w http.ResponseWriter, r *http.Request, page, pageSize int) {
	total := len(allProducts)
	totalPages := int(math.Ceil(float64(total) / float64(pageSize)))
	start := (page - 1) * pageSize
	end := start + pageSize
	if end > total {
		end = total
	}
	if start > total {
		start = total
	}
	pageData := allProducts[start:end]
	response := PaginatedResponse{
		Data: pageData,
		Meta: Metadata{Total: total, Page: page, PageSize: pageSize, TotalPages: totalPages},
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(response)
}`
		},
		uz: {
			title: 'Javoblarni formatlash',
			description: `Sahifalash va metama\'lumotlar bilan izchil javob formatini amalga oshiring.`,
			hint1: `math.Ceil(float64(total) / float64(pageSize)) yordamida totalPages ni hisoblang. start = (page-1)*pageSize va end = start+pageSize ni hisoblang.`,
			hint2: `Sahifa ma\'lumotlarini olish uchun allProducts[start:end] ni kesing, keyin ma\'lumotlar va metama\'lumotlar bilan PaginatedResponse ni yarating.`,
			whyItMatters: `Izchil javob formatlash API ni bashorat qilinadigan va ishlatish uchun oson qiladi.`,
			solutionCode: `package api

import (
	"encoding/json"
	"math"
	"net/http"
)

type PaginatedResponse struct {
	Data interface{} \`json:"data"\`
	Meta Metadata    \`json:"meta"\`
}

type Metadata struct {
	Total      int \`json:"total"\`
	Page       int \`json:"page"\`
	PageSize   int \`json:"pageSize"\`
	TotalPages int \`json:"totalPages"\`
}

type Product struct {
	ID    int     \`json:"id"\`
	Name  string  \`json:"name"\`
	Price float64 \`json:"price"\`
}

var allProducts = []Product{
	{ID: 1, Name: "Laptop", Price: 999.99},
	{ID: 2, Name: "Mouse", Price: 29.99},
	{ID: 3, Name: "Keyboard", Price: 79.99},
	{ID: 4, Name: "Monitor", Price: 299.99},
	{ID: 5, Name: "Headphones", Price: 149.99},
}

func FormatListResponse(w http.ResponseWriter, r *http.Request, page, pageSize int) {
	total := len(allProducts)
	totalPages := int(math.Ceil(float64(total) / float64(pageSize)))
	start := (page - 1) * pageSize
	end := start + pageSize
	if end > total {
		end = total
	}
	if start > total {
		start = total
	}
	pageData := allProducts[start:end]
	response := PaginatedResponse{
		Data: pageData,
		Meta: Metadata{Total: total, Page: page, PageSize: pageSize, TotalPages: totalPages},
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(response)
}`
		}
	}
};

export default task;
