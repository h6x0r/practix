import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'api-design-versioning-header',
	title: 'API Versioning - Header-based',
	difficulty: 'medium',
	tags: ['api-design', 'versioning', 'go', 'headers'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement header-based API versioning using Accept header for content negotiation.

**You will implement:**

1. **Version middleware** - Parse version from Accept header
2. **Version routing** - Route to correct handler based on version
3. **Default version** - Handle requests without version header

**Header Versioning Pattern:**

\`\`\`
Accept: application/vnd.myapi.v1+json  # Version 1
Accept: application/vnd.myapi.v2+json  # Version 2
Accept: application/json               # Default version
\`\`\`

**Advantages:**
- Clean URLs (no version in path)
- Follows HTTP standards (Accept header)
- Same endpoint, different representations

**When to use:**
- RESTful APIs following HTTP standards
- APIs with multiple content formats
- When URL cleanliness is important`,
	initialCode: `package api

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
)

type contextKey string

const versionKey contextKey = "apiVersion"

type UserV1 struct {
	ID   int    \`json:"id"\`
	Name string \`json:"name"\`
}

type UserV2 struct {
	ID       int    \`json:"id"\`
	FullName string \`json:"fullName"\`
	Email    string \`json:"email"\`
}

// TODO: Implement VersionMiddleware
// Parse API version from Accept header:
// 1. Get Accept header
// 2. Parse version (e.g., "application/vnd.myapi.v1+json" -> "v1")
// 3. Default to "v1" if not specified
// 4. Add version to context
func VersionMiddleware(next http.Handler) http.Handler {
	panic("TODO: implement VersionMiddleware")
}

// TODO: Implement GetUsers
// Return response based on version from context:
// 1. Extract version from context
// 2. If "v1" -> return UserV1 format
// 3. If "v2" -> return UserV2 format
// 4. Set Content-Type based on version
func GetUsers(w http.ResponseWriter, r *http.Request) {
	panic("TODO: implement GetUsers")
}`,
	solutionCode: `package api

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
)

type contextKey string

const versionKey contextKey = "apiVersion"

type UserV1 struct {
	ID   int    \`json:"id"\`
	Name string \`json:"name"\`
}

type UserV2 struct {
	ID       int    \`json:"id"\`
	FullName string \`json:"fullName"\`
	Email    string \`json:"email"\`
}

func VersionMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Get Accept header
		accept := r.Header.Get("Accept")

		// Parse version from header
		version := "v1" // Default version
		if strings.Contains(accept, "vnd.myapi.v2") {
			version = "v2"
		} else if strings.Contains(accept, "vnd.myapi.v1") {
			version = "v1"
		}

		// Add version to context
		ctx := context.WithValue(r.Context(), versionKey, version)

		// Set response header indicating version used
		w.Header().Set("API-Version", version)

		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func GetUsers(w http.ResponseWriter, r *http.Request) {
	// Extract version from context
	version, _ := r.Context().Value(versionKey).(string)

	w.Header().Set("Content-Type", "application/json")

	switch version {
	case "v2":
		// V2 format with enhanced fields
		users := []UserV2{
			{ID: 1, FullName: "Alice Johnson", Email: "alice@example.com"},
			{ID: 2, FullName: "Bob Smith", Email: "bob@example.com"},
		}
		json.NewEncoder(w).Encode(users)

	default: // "v1" or unspecified
		// V1 format with basic fields
		users := []UserV1{
			{ID: 1, Name: "Alice"},
			{ID: 2, Name: "Bob"},
		}
		json.NewEncoder(w).Encode(users)
	}
}`,
	hint1: `In VersionMiddleware: get Accept header with r.Header.Get("Accept"), check if it contains "vnd.myapi.v2" or "v1" using strings.Contains, default to "v1", add to context with context.WithValue(r.Context(), versionKey, version).`,
	hint2: `In GetUsers: get version from r.Context().Value(versionKey), use switch statement to check version, return []UserV2 for "v2" or []UserV1 for default/v1. Set API-Version response header.`,
	whyItMatters: `Header-based versioning provides clean URLs while following HTTP content negotiation standards.

**Why Header Versioning Matters:**

**1. Clean URLs**
URLs remain simple and resource-focused:

\`\`\`go
// URL versioning
GET /api/v1/users
GET /api/v2/users

// Header versioning - same URL!
GET /api/users
Accept: application/vnd.myapi.v1+json

GET /api/users
Accept: application/vnd.myapi.v2+json

// URLs stay clean, bookmarks don't break
\`\`\`

**2. HTTP Standard**
Uses Accept header for content negotiation:

\`\`\`go
// Request different representations of same resource
GET /api/users
Accept: application/vnd.myapi.v1+json

GET /api/users
Accept: application/vnd.myapi.v1+xml

GET /api/users
Accept: application/vnd.myapi.v2+json
\`\`\`

**3. Gradual Migration**

\`\`\`go
// Old client (no header)
GET /api/users
// Server defaults to v1

// New client (with header)
GET /api/users
Accept: application/vnd.myapi.v2+json
// Server returns v2

// Same endpoint serves both!
\`\`\`

**Comparison:**

**URL Versioning:**
✅ Easy to use and test
✅ Version visible in browser
✅ Easy to cache per version
❌ URLs change between versions

**Header Versioning:**
✅ Clean, stable URLs
✅ Follows HTTP standards
✅ Multiple content types
❌ Harder to test manually
❌ Requires header support

**Best Practices:**
- Use Accept header with vendor media types
- Default to latest stable version
- Return API-Version in response header
- Support 2-3 versions simultaneously
- Document header format clearly`,
	order: 9,
	testCode: `package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// Test1: VersionMiddleware defaults to v1
func Test1(t *testing.T) {
	handler := VersionMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		v := r.Context().Value(versionKey).(string)
		if v != "v1" {
			t.Error("Should default to v1")
		}
	}))
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/users", nil)
	handler.ServeHTTP(w, r)
}

// Test2: VersionMiddleware detects v2 from Accept header
func Test2(t *testing.T) {
	handler := VersionMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		v := r.Context().Value(versionKey).(string)
		if v != "v2" {
			t.Error("Should detect v2")
		}
	}))
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/users", nil)
	r.Header.Set("Accept", "application/vnd.myapi.v2+json")
	handler.ServeHTTP(w, r)
}

// Test3: VersionMiddleware sets API-Version header
func Test3(t *testing.T) {
	handler := VersionMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/users", nil)
	handler.ServeHTTP(w, r)
	if w.Header().Get("API-Version") == "" {
		t.Error("Should set API-Version header")
	}
}

// Test4: GetUsers returns v1 format by default
func Test4(t *testing.T) {
	w := httptest.NewRecorder()
	ctx := context.WithValue(context.Background(), versionKey, "v1")
	r := httptest.NewRequest("GET", "/api/users", nil).WithContext(ctx)
	GetUsers(w, r)
	var result []map[string]interface{}
	json.NewDecoder(w.Body).Decode(&result)
	if _, ok := result[0]["fullName"]; ok {
		t.Error("V1 should not have fullName field")
	}
}

// Test5: GetUsers returns v2 format with fullName
func Test5(t *testing.T) {
	w := httptest.NewRecorder()
	ctx := context.WithValue(context.Background(), versionKey, "v2")
	r := httptest.NewRequest("GET", "/api/users", nil).WithContext(ctx)
	GetUsers(w, r)
	var result []map[string]interface{}
	json.NewDecoder(w.Body).Decode(&result)
	if _, ok := result[0]["fullName"]; !ok {
		t.Error("V2 should have fullName field")
	}
}

// Test6: UserV1 struct has correct fields
func Test6(t *testing.T) {
	u := UserV1{ID: 1, Name: "Test"}
	if u.ID != 1 || u.Name != "Test" {
		t.Error("UserV1 fields not set correctly")
	}
}

// Test7: UserV2 struct has FullName and Email
func Test7(t *testing.T) {
	u := UserV2{ID: 1, FullName: "Test User", Email: "test@test.com"}
	if u.FullName != "Test User" || u.Email != "test@test.com" {
		t.Error("UserV2 fields not set correctly")
	}
}

// Test8: Middleware with v1 in Accept header
func Test8(t *testing.T) {
	handler := VersionMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		v := r.Context().Value(versionKey).(string)
		if v != "v1" {
			t.Error("Should detect v1")
		}
	}))
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/users", nil)
	r.Header.Set("Accept", "application/vnd.myapi.v1+json")
	handler.ServeHTTP(w, r)
}

// Test9: GetUsers returns JSON Content-Type
func Test9(t *testing.T) {
	w := httptest.NewRecorder()
	ctx := context.WithValue(context.Background(), versionKey, "v1")
	r := httptest.NewRequest("GET", "/api/users", nil).WithContext(ctx)
	GetUsers(w, r)
	if w.Header().Get("Content-Type") != "application/json" {
		t.Error("Should return application/json")
	}
}

// Test10: V2 response has email field
func Test10(t *testing.T) {
	w := httptest.NewRecorder()
	ctx := context.WithValue(context.Background(), versionKey, "v2")
	r := httptest.NewRequest("GET", "/api/users", nil).WithContext(ctx)
	GetUsers(w, r)
	var result []map[string]interface{}
	json.NewDecoder(w.Body).Decode(&result)
	if _, ok := result[0]["email"]; !ok {
		t.Error("V2 should have email field")
	}
}
`,
	translations: {
		ru: {
			title: 'Версионирование API - На основе заголовков',
			description: `Реализуйте версионирование API на основе заголовков используя Accept заголовок для согласования содержимого.`,
			hint1: `В VersionMiddleware: получите Accept заголовок с r.Header.Get("Accept"), проверьте содержит ли он "vnd.myapi.v2" или "v1" используя strings.Contains, по умолчанию "v1", добавьте в контекст.`,
			hint2: `В GetUsers: получите версию из r.Context().Value(versionKey), используйте switch для проверки версии, верните []UserV2 для "v2" или []UserV1 для default/v1.`,
			whyItMatters: `Версионирование на основе заголовков обеспечивает чистые URL следуя стандартам HTTP согласования содержимого.`,
			solutionCode: `package api

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
)

type contextKey string

const versionKey contextKey = "apiVersion"

type UserV1 struct {
	ID   int    \`json:"id"\`
	Name string \`json:"name"\`
}

type UserV2 struct {
	ID       int    \`json:"id"\`
	FullName string \`json:"fullName"\`
	Email    string \`json:"email"\`
}

func VersionMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		accept := r.Header.Get("Accept")
		version := "v1"
		if strings.Contains(accept, "vnd.myapi.v2") {
			version = "v2"
		} else if strings.Contains(accept, "vnd.myapi.v1") {
			version = "v1"
		}
		ctx := context.WithValue(r.Context(), versionKey, version)
		w.Header().Set("API-Version", version)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func GetUsers(w http.ResponseWriter, r *http.Request) {
	version, _ := r.Context().Value(versionKey).(string)
	w.Header().Set("Content-Type", "application/json")
	switch version {
	case "v2":
		users := []UserV2{
			{ID: 1, FullName: "Alice Johnson", Email: "alice@example.com"},
			{ID: 2, FullName: "Bob Smith", Email: "bob@example.com"},
		}
		json.NewEncoder(w).Encode(users)
	default:
		users := []UserV1{
			{ID: 1, Name: "Alice"},
			{ID: 2, Name: "Bob"},
		}
		json.NewEncoder(w).Encode(users)
	}
}`
		},
		uz: {
			title: 'API versiyalash - Header asosida',
			description: `Kontent kelishuvchanligi uchun Accept headeridan foydalangan holda header asosidagi API versiyalashni amalga oshiring.`,
			hint1: `VersionMiddleware da: r.Header.Get("Accept") bilan Accept headerini oling, strings.Contains yordamida "vnd.myapi.v2" yoki "v1" borligini tekshiring, standart "v1", kontekstga qo\'shing.`,
			hint2: `GetUsers da: r.Context().Value(versionKey) dan versiyani oling, versiyani tekshirish uchun switch dan foydalaning, "v2" uchun []UserV2 yoki default/v1 uchun []UserV1 qaytaring.`,
			whyItMatters: `Header asosidagi versiyalash HTTP kontent kelishuvchanligi standartlariga rioya qilgan holda toza URL larni ta\'minlaydi.`,
			solutionCode: `package api

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
)

type contextKey string

const versionKey contextKey = "apiVersion"

type UserV1 struct {
	ID   int    \`json:"id"\`
	Name string \`json:"name"\`
}

type UserV2 struct {
	ID       int    \`json:"id"\`
	FullName string \`json:"fullName"\`
	Email    string \`json:"email"\`
}

func VersionMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		accept := r.Header.Get("Accept")
		version := "v1"
		if strings.Contains(accept, "vnd.myapi.v2") {
			version = "v2"
		} else if strings.Contains(accept, "vnd.myapi.v1") {
			version = "v1"
		}
		ctx := context.WithValue(r.Context(), versionKey, version)
		w.Header().Set("API-Version", version)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func GetUsers(w http.ResponseWriter, r *http.Request) {
	version, _ := r.Context().Value(versionKey).(string)
	w.Header().Set("Content-Type", "application/json")
	switch version {
	case "v2":
		users := []UserV2{
			{ID: 1, FullName: "Alice Johnson", Email: "alice@example.com"},
			{ID: 2, FullName: "Bob Smith", Email: "bob@example.com"},
		}
		json.NewEncoder(w).Encode(users)
	default:
		users := []UserV1{
			{ID: 1, Name: "Alice"},
			{ID: 2, Name: "Bob"},
		}
		json.NewEncoder(w).Encode(users)
	}
}`
		}
	}
};

export default task;
