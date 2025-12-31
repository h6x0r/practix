import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'api-design-versioning-url',
	title: 'API Versioning - URL Path',
	difficulty: 'easy',
	tags: ['api-design', 'versioning', 'go'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement URL-based API versioning for backward compatibility.

**You will implement:**

1. **V1 endpoints** - Original API version
2. **V2 endpoints** - New API version with changes
3. **Version routing** - Route requests to correct version

**URL Versioning Pattern:**

\`\`\`
GET /api/v1/users     # Version 1
GET /api/v2/users     # Version 2
\`\`\`

**Version Differences:**
- V1: Returns basic user info
- V2: Returns enhanced user info with additional fields

**Best Practices:**
- Include version in URL path
- Keep old versions running during migration
- Deprecate old versions gradually
- Document version differences`,
	initialCode: `package api

import (
	"encoding/json"
	"net/http"

	"github.com/go-chi/chi/v5"
)

type UserV1 struct {
	ID    int    \`json:"id"\`
	Name  string \`json:"name"\`
	Email string \`json:"email"\`
}

type UserV2 struct {
	ID        int    \`json:"id"\`
	Name      string \`json:"name"\`
	Email     string \`json:"email"\`
	CreatedAt string \`json:"createdAt"\`
	Status    string \`json:"status"\`
}

// TODO: Implement SetupVersionedRoutes
// Configure routes for both API versions:
// GET /api/v1/users -> GetUsersV1
// GET /api/v2/users -> GetUsersV2
func SetupVersionedRoutes(r chi.Router) {
	panic("TODO: implement SetupVersionedRoutes")
}

// TODO: Implement GetUsersV1
// Return users in V1 format (basic info only)
func GetUsersV1(w http.ResponseWriter, r *http.Request) {
	panic("TODO: implement GetUsersV1")
}

// TODO: Implement GetUsersV2
// Return users in V2 format (enhanced info)
func GetUsersV2(w http.ResponseWriter, r *http.Request) {
	panic("TODO: implement GetUsersV2")
}`,
	solutionCode: `package api

import (
	"encoding/json"
	"net/http"

	"github.com/go-chi/chi/v5"
)

type UserV1 struct {
	ID    int    \`json:"id"\`
	Name  string \`json:"name"\`
	Email string \`json:"email"\`
}

type UserV2 struct {
	ID        int    \`json:"id"\`
	Name      string \`json:"name"\`
	Email     string \`json:"email"\`
	CreatedAt string \`json:"createdAt"\`
	Status    string \`json:"status"\`
}

func SetupVersionedRoutes(r chi.Router) {
	// Version 1 routes
	r.Route("/api/v1", func(r chi.Router) {
		r.Get("/users", GetUsersV1)
	})

	// Version 2 routes
	r.Route("/api/v2", func(r chi.Router) {
		r.Get("/users", GetUsersV2)
	})
}

func GetUsersV1(w http.ResponseWriter, r *http.Request) {
	// V1 returns basic user info only
	users := []UserV1{
		{ID: 1, Name: "Alice", Email: "alice@example.com"},
		{ID: 2, Name: "Bob", Email: "bob@example.com"},
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("API-Version", "1.0")
	json.NewEncoder(w).Encode(users)
}

func GetUsersV2(w http.ResponseWriter, r *http.Request) {
	// V2 returns enhanced user info with additional fields
	users := []UserV2{
		{
			ID:        1,
			Name:      "Alice",
			Email:     "alice@example.com",
			CreatedAt: "2024-01-15T10:00:00Z",
			Status:    "active",
		},
		{
			ID:        2,
			Name:      "Bob",
			Email:     "bob@example.com",
			CreatedAt: "2024-01-16T11:00:00Z",
			Status:    "active",
		},
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("API-Version", "2.0")
	json.NewEncoder(w).Encode(users)
}`,
	hint1: `Use r.Route("/api/v1", func(r chi.Router) {...}) to group v1 routes and r.Route("/api/v2", func(r chi.Router) {...}) for v2 routes. Inside each, add r.Get("/users", handler).`,
	hint2: `In GetUsersV1, create []UserV1 with basic fields and encode. In GetUsersV2, create []UserV2 with additional CreatedAt and Status fields. Set API-Version header in both.`,
	whyItMatters: `URL versioning allows you to evolve your API while maintaining backward compatibility for existing clients.

**Why URL Versioning Matters:**

**1. Backward Compatibility**
Old clients continue working while you ship new features:

\`\`\`go
// Mobile app v1.0 (released 2023)
GET /api/v1/users  // Still works in 2024

// Mobile app v2.0 (released 2024)
GET /api/v2/users  // Uses new enhanced format

// Both apps work simultaneously!
\`\`\`

**2. Clear Version in URL**
Version is visible and easy to use:

\`\`\`go
// Easy to test different versions
curl https://api.example.com/v1/users
curl https://api.example.com/v2/users

// Easy to migrate gradually
// Move 10% of traffic to v2, monitor, then increase
\`\`\`

**3. Independent Evolution**

\`\`\`go
// V1: Simple response
type UserV1 struct {
    ID   int
    Name string
}

// V2: Breaking changes (renamed field)
type UserV2 struct {
    ID       int
    FullName string  // Renamed from Name
    Email    string  // New required field
}

// Both versions coexist without breaking old clients
\`\`\`

**Best Practices:**
- Version in URL path (/v1/, /v2/)
- Support 2-3 versions simultaneously
- Deprecate old versions gradually
- Document migration guides
- Set deprecation headers`,
	order: 8,
	testCode: `package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/go-chi/chi/v5"
)

// Test1: GetUsersV1 returns 200
func Test1(t *testing.T) {
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/v1/users", nil)
	GetUsersV1(w, r)
	if w.Code != http.StatusOK {
		t.Errorf("Expected 200, got %d", w.Code)
	}
}

// Test2: GetUsersV2 returns 200
func Test2(t *testing.T) {
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/v2/users", nil)
	GetUsersV2(w, r)
	if w.Code != http.StatusOK {
		t.Errorf("Expected 200, got %d", w.Code)
	}
}

// Test3: V1 returns UserV1 format (no createdAt)
func Test3(t *testing.T) {
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/v1/users", nil)
	GetUsersV1(w, r)
	var result []map[string]interface{}
	json.NewDecoder(w.Body).Decode(&result)
	if _, ok := result[0]["createdAt"]; ok {
		t.Error("V1 should not have createdAt field")
	}
}

// Test4: V2 returns UserV2 format (has createdAt)
func Test4(t *testing.T) {
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/v2/users", nil)
	GetUsersV2(w, r)
	var result []map[string]interface{}
	json.NewDecoder(w.Body).Decode(&result)
	if _, ok := result[0]["createdAt"]; !ok {
		t.Error("V2 should have createdAt field")
	}
}

// Test5: SetupVersionedRoutes registers v1 route
func Test5(t *testing.T) {
	router := chi.NewRouter()
	SetupVersionedRoutes(router)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/v1/users", nil)
	router.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Error("V1 route should be registered")
	}
}

// Test6: SetupVersionedRoutes registers v2 route
func Test6(t *testing.T) {
	router := chi.NewRouter()
	SetupVersionedRoutes(router)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/v2/users", nil)
	router.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Error("V2 route should be registered")
	}
}

// Test7: UserV1 struct has correct fields
func Test7(t *testing.T) {
	u := UserV1{ID: 1, Name: "Test", Email: "test@test.com"}
	if u.ID != 1 || u.Name != "Test" || u.Email != "test@test.com" {
		t.Error("UserV1 fields not set correctly")
	}
}

// Test8: UserV2 struct has additional fields
func Test8(t *testing.T) {
	u := UserV2{ID: 1, Name: "Test", Email: "test@test.com", CreatedAt: "2024-01-01", Status: "active"}
	if u.CreatedAt != "2024-01-01" || u.Status != "active" {
		t.Error("UserV2 additional fields not set correctly")
	}
}

// Test9: V1 sets API-Version header
func Test9(t *testing.T) {
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/v1/users", nil)
	GetUsersV1(w, r)
	if w.Header().Get("API-Version") != "1.0" {
		t.Error("V1 should set API-Version: 1.0")
	}
}

// Test10: V2 sets API-Version header
func Test10(t *testing.T) {
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/v2/users", nil)
	GetUsersV2(w, r)
	if w.Header().Get("API-Version") != "2.0" {
		t.Error("V2 should set API-Version: 2.0")
	}
}
`,
	translations: {
		ru: {
			title: 'Версионирование API - URL путь',
			description: `Реализуйте версионирование API на основе URL для обратной совместимости.`,
			hint1: `Используйте r.Route("/api/v1", func(r chi.Router) {...}) для группировки маршрутов v1 и r.Route("/api/v2", func(r chi.Router) {...}) для v2.`,
			hint2: `В GetUsersV1 создайте []UserV1 с базовыми полями. В GetUsersV2 создайте []UserV2 с дополнительными полями CreatedAt и Status. Установите заголовок API-Version в обоих.`,
			whyItMatters: `URL версионирование позволяет развивать ваш API сохраняя обратную совместимость для существующих клиентов.`,
			solutionCode: `package api

import (
	"encoding/json"
	"net/http"

	"github.com/go-chi/chi/v5"
)

type UserV1 struct {
	ID    int    \`json:"id"\`
	Name  string \`json:"name"\`
	Email string \`json:"email"\`
}

type UserV2 struct {
	ID        int    \`json:"id"\`
	Name      string \`json:"name"\`
	Email     string \`json:"email"\`
	CreatedAt string \`json:"createdAt"\`
	Status    string \`json:"status"\`
}

func SetupVersionedRoutes(r chi.Router) {
	r.Route("/api/v1", func(r chi.Router) {
		r.Get("/users", GetUsersV1)
	})
	r.Route("/api/v2", func(r chi.Router) {
		r.Get("/users", GetUsersV2)
	})
}

func GetUsersV1(w http.ResponseWriter, r *http.Request) {
	users := []UserV1{
		{ID: 1, Name: "Alice", Email: "alice@example.com"},
		{ID: 2, Name: "Bob", Email: "bob@example.com"},
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("API-Version", "1.0")
	json.NewEncoder(w).Encode(users)
}

func GetUsersV2(w http.ResponseWriter, r *http.Request) {
	users := []UserV2{
		{ID: 1, Name: "Alice", Email: "alice@example.com", CreatedAt: "2024-01-15T10:00:00Z", Status: "active"},
		{ID: 2, Name: "Bob", Email: "bob@example.com", CreatedAt: "2024-01-16T11:00:00Z", Status: "active"},
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("API-Version", "2.0")
	json.NewEncoder(w).Encode(users)
}`
		},
		uz: {
			title: 'API versiyalash - URL yo\'li',
			description: `Orqaga moslik uchun URL asosidagi API versiyalashni amalga oshiring.`,
			hint1: `v1 yo\'llarini guruhlash uchun r.Route("/api/v1", func(r chi.Router) {...}) va v2 uchun r.Route("/api/v2", func(r chi.Router) {...}) dan foydalaning.`,
			hint2: `GetUsersV1 da asosiy maydonlar bilan []UserV1 yarating. GetUsersV2 da qo\'shimcha CreatedAt va Status maydonlari bilan []UserV2 yarating. Ikkalasida ham API-Version headerini o\'rnating.`,
			whyItMatters: `URL versiyalash mavjud mijozlar uchun orqaga moslikni saqlab, API ni rivojlantirish imkonini beradi.`,
			solutionCode: `package api

import (
	"encoding/json"
	"net/http"

	"github.com/go-chi/chi/v5"
)

type UserV1 struct {
	ID    int    \`json:"id"\`
	Name  string \`json:"name"\`
	Email string \`json:"email"\`
}

type UserV2 struct {
	ID        int    \`json:"id"\`
	Name      string \`json:"name"\`
	Email     string \`json:"email"\`
	CreatedAt string \`json:"createdAt"\`
	Status    string \`json:"status"\`
}

func SetupVersionedRoutes(r chi.Router) {
	r.Route("/api/v1", func(r chi.Router) {
		r.Get("/users", GetUsersV1)
	})
	r.Route("/api/v2", func(r chi.Router) {
		r.Get("/users", GetUsersV2)
	})
}

func GetUsersV1(w http.ResponseWriter, r *http.Request) {
	users := []UserV1{
		{ID: 1, Name: "Alice", Email: "alice@example.com"},
		{ID: 2, Name: "Bob", Email: "bob@example.com"},
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("API-Version", "1.0")
	json.NewEncoder(w).Encode(users)
}

func GetUsersV2(w http.ResponseWriter, r *http.Request) {
	users := []UserV2{
		{ID: 1, Name: "Alice", Email: "alice@example.com", CreatedAt: "2024-01-15T10:00:00Z", Status: "active"},
		{ID: 2, Name: "Bob", Email: "bob@example.com", CreatedAt: "2024-01-16T11:00:00Z", Status: "active"},
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("API-Version", "2.0")
	json.NewEncoder(w).Encode(users)
}`
		}
	}
};

export default task;
