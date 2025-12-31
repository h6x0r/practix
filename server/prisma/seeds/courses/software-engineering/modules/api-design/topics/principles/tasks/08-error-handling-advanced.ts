import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'api-design-error-handling-advanced',
	title: 'Error Handling - Advanced Patterns',
	difficulty: 'hard',
	tags: ['api-design', 'error-handling', 'go'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement advanced error handling with request IDs, error tracking, and detailed context.

**You will implement:**

1. **ErrorMiddleware** - Centralized error handling middleware
2. **Request ID tracking** - Include request ID in all error responses
3. **Error context** - Include timestamp and additional context

**Advanced Error Structure:**

\`\`\`json
{
  "error": {
    "code": "DATABASE_ERROR",
    "message": "Failed to fetch user",
    "requestId": "abc-123-def",
    "timestamp": "2024-01-15T10:30:00Z",
    "path": "/api/users/123"
  }
}
\`\`\`

**Features:**
- Request ID for tracking
- Timestamp for debugging
- Request path for context
- Stack traces in development only`,
	initialCode: `package api

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/google/uuid"
)

type AdvancedErrorResponse struct {
	Error AdvancedErrorDetail \`json:"error"\`
}

type AdvancedErrorDetail struct {
	Code      string \`json:"code"\`
	Message   string \`json:"message"\`
	RequestID string \`json:"requestId"\`
	Timestamp string \`json:"timestamp"\`
	Path      string \`json:"path"\`
}

type contextKey string

const requestIDKey contextKey = "requestID"

// TODO: Implement RequestIDMiddleware
// Generate and attach request ID to context:
// 1. Generate UUID for request
// 2. Add to context with requestIDKey
// 3. Add X-Request-ID header to response
// 4. Call next handler
func RequestIDMiddleware(next http.Handler) http.Handler {
	panic("TODO: implement RequestIDMiddleware")
}

// TODO: Implement HandleAdvancedError
// Build advanced error response with:
// 1. Extract request ID from context
// 2. Get current timestamp (RFC3339)
// 3. Get request path
// 4. Build AdvancedErrorResponse
// 5. Return with appropriate status code
func HandleAdvancedError(w http.ResponseWriter, r *http.Request, code, message string, statusCode int) {
	panic("TODO: implement HandleAdvancedError")
}`,
	solutionCode: `package api

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/google/uuid"
)

type AdvancedErrorResponse struct {
	Error AdvancedErrorDetail \`json:"error"\`
}

type AdvancedErrorDetail struct {
	Code      string \`json:"code"\`
	Message   string \`json:"message"\`
	RequestID string \`json:"requestId"\`
	Timestamp string \`json:"timestamp"\`
	Path      string \`json:"path"\`
}

type contextKey string

const requestIDKey contextKey = "requestID"

func RequestIDMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Generate unique request ID
		requestID := uuid.New().String()

		// Add to context for use in handlers
		ctx := context.WithValue(r.Context(), requestIDKey, requestID)

		// Add to response headers for client tracking
		w.Header().Set("X-Request-ID", requestID)

		// Call next handler with updated context
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func HandleAdvancedError(w http.ResponseWriter, r *http.Request, code, message string, statusCode int) {
	// Extract request ID from context
	requestID, _ := r.Context().Value(requestIDKey).(string)
	if requestID == "" {
		requestID = "unknown"
	}

	// Build error response with full context
	response := AdvancedErrorResponse{
		Error: AdvancedErrorDetail{
			Code:      code,
			Message:   message,
			RequestID: requestID,
			Timestamp: time.Now().UTC().Format(time.RFC3339),
			Path:      r.URL.Path,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(response)
}`,
	hint1: `In RequestIDMiddleware: use uuid.New().String() to generate ID, add to context with context.WithValue(r.Context(), requestIDKey, requestID), set X-Request-ID header, call next.ServeHTTP with r.WithContext(ctx).`,
	hint2: `In HandleAdvancedError: get requestID from r.Context().Value(requestIDKey), build AdvancedErrorDetail with code, message, requestID, time.Now().UTC().Format(time.RFC3339) for timestamp, and r.URL.Path.`,
	whyItMatters: `Advanced error handling with request IDs and context makes debugging production issues significantly easier.

**Why Advanced Error Handling Matters:**

**1. Request Tracking**
Request IDs link logs across services:

\`\`\`go
// Error response includes request ID
{
  "error": {
    "code": "DATABASE_ERROR",
    "requestId": "abc-123-def"
  }
}

// Search logs by request ID to see full flow:
// [abc-123-def] API Gateway: Request received
// [abc-123-def] Auth Service: Token validated
// [abc-123-def] Database: Connection failed
\`\`\`

**2. Debugging Context**
Include timestamp and path for debugging:

\`\`\`go
{
  "error": {
    "code": "TIMEOUT",
    "timestamp": "2024-01-15T14:30:00Z",
    "path": "/api/users/123/orders"
  }
}
// Developer knows exactly when and where error occurred
\`\`\`

**Best Practices:**
- Generate request ID at API gateway
- Pass request ID through all services
- Include request ID in all logs
- Return request ID in error responses
- Store request ID for support tickets`,
	order: 7,
	testCode: `package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// Test1: RequestIDMiddleware sets X-Request-ID header
func Test1(t *testing.T) {
	handler := RequestIDMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/test", nil)
	handler.ServeHTTP(w, r)
	if w.Header().Get("X-Request-ID") == "" {
		t.Error("Should set X-Request-ID header")
	}
}

// Test2: Request ID is valid UUID format
func Test2(t *testing.T) {
	handler := RequestIDMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/test", nil)
	handler.ServeHTTP(w, r)
	id := w.Header().Get("X-Request-ID")
	if len(id) < 36 || !strings.Contains(id, "-") {
		t.Error("Request ID should be UUID format")
	}
}

// Test3: HandleAdvancedError sets correct status
func Test3(t *testing.T) {
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/test", nil)
	HandleAdvancedError(w, r, "TEST_ERROR", "Test message", 400)
	if w.Code != 400 {
		t.Errorf("Expected 400, got %d", w.Code)
	}
}

// Test4: HandleAdvancedError includes code in response
func Test4(t *testing.T) {
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/test", nil)
	HandleAdvancedError(w, r, "CUSTOM_CODE", "msg", 500)
	var result AdvancedErrorResponse
	json.NewDecoder(w.Body).Decode(&result)
	if result.Error.Code != "CUSTOM_CODE" {
		t.Error("Should include error code")
	}
}

// Test5: HandleAdvancedError includes timestamp
func Test5(t *testing.T) {
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/test", nil)
	HandleAdvancedError(w, r, "TEST", "msg", 400)
	var result AdvancedErrorResponse
	json.NewDecoder(w.Body).Decode(&result)
	if result.Error.Timestamp == "" {
		t.Error("Should include timestamp")
	}
}

// Test6: HandleAdvancedError includes path
func Test6(t *testing.T) {
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/users/123", nil)
	HandleAdvancedError(w, r, "TEST", "msg", 400)
	var result AdvancedErrorResponse
	json.NewDecoder(w.Body).Decode(&result)
	if result.Error.Path != "/api/users/123" {
		t.Errorf("Expected path /api/users/123, got %s", result.Error.Path)
	}
}

// Test7: AdvancedErrorDetail struct has correct fields
func Test7(t *testing.T) {
	err := AdvancedErrorDetail{Code: "A", Message: "B", RequestID: "C", Timestamp: "D", Path: "E"}
	if err.Code != "A" || err.Message != "B" {
		t.Error("AdvancedErrorDetail fields not set correctly")
	}
}

// Test8: Middleware passes context to handler
func Test8(t *testing.T) {
	var gotID string
	handler := RequestIDMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotID, _ = r.Context().Value(requestIDKey).(string)
	}))
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/test", nil)
	handler.ServeHTTP(w, r)
	if gotID == "" {
		t.Error("Request ID should be in context")
	}
}

// Test9: HandleAdvancedError with request ID from middleware
func Test9(t *testing.T) {
	handler := RequestIDMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		HandleAdvancedError(w, r, "TEST", "msg", 400)
	}))
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/test", nil)
	handler.ServeHTTP(w, r)
	var result AdvancedErrorResponse
	json.NewDecoder(w.Body).Decode(&result)
	if result.Error.RequestID == "" || result.Error.RequestID == "unknown" {
		t.Error("Should include request ID from middleware")
	}
}

// Test10: HandleAdvancedError without middleware uses unknown
func Test10(t *testing.T) {
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/test", nil)
	HandleAdvancedError(w, r, "TEST", "msg", 400)
	var result AdvancedErrorResponse
	json.NewDecoder(w.Body).Decode(&result)
	if result.Error.RequestID != "unknown" {
		t.Error("Without middleware should use 'unknown'")
	}
}
`,
	translations: {
		ru: {
			title: 'Обработка ошибок - Продвинутые паттерны',
			description: `Реализуйте продвинутую обработку ошибок с ID запросов, отслеживанием ошибок и детальным контекстом.`,
			hint1: `В RequestIDMiddleware: используйте uuid.New().String() для генерации ID, добавьте в контекст с context.WithValue, установите X-Request-ID заголовок, вызовите next.ServeHTTP с обновлённым контекстом.`,
			hint2: `В HandleAdvancedError: получите requestID из r.Context().Value(requestIDKey), постройте AdvancedErrorDetail с code, message, requestID, time.Now().UTC().Format(time.RFC3339) и r.URL.Path.`,
			whyItMatters: `Продвинутая обработка ошибок с ID запросов и контекстом значительно упрощает отладку проблем в продакшене.`,
			solutionCode: `package api

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/google/uuid"
)

type AdvancedErrorResponse struct {
	Error AdvancedErrorDetail \`json:"error"\`
}

type AdvancedErrorDetail struct {
	Code      string \`json:"code"\`
	Message   string \`json:"message"\`
	RequestID string \`json:"requestId"\`
	Timestamp string \`json:"timestamp"\`
	Path      string \`json:"path"\`
}

type contextKey string

const requestIDKey contextKey = "requestID"

func RequestIDMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestID := uuid.New().String()
		ctx := context.WithValue(r.Context(), requestIDKey, requestID)
		w.Header().Set("X-Request-ID", requestID)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func HandleAdvancedError(w http.ResponseWriter, r *http.Request, code, message string, statusCode int) {
	requestID, _ := r.Context().Value(requestIDKey).(string)
	if requestID == "" {
		requestID = "unknown"
	}
	response := AdvancedErrorResponse{
		Error: AdvancedErrorDetail{
			Code:      code,
			Message:   message,
			RequestID: requestID,
			Timestamp: time.Now().UTC().Format(time.RFC3339),
			Path:      r.URL.Path,
		},
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(response)
}`
		},
		uz: {
			title: 'Xatolarni qayta ishlash - Ilg\'or patternlar',
			description: `So\'rov ID lari, xatolarni kuzatish va batafsil kontekst bilan ilg\'or xatolarni qayta ishlashni amalga oshiring.`,
			hint1: `RequestIDMiddleware da: ID yaratish uchun uuid.New().String() dan foydalaning, context.WithValue bilan kontekstga qo\'shing, X-Request-ID headerini o\'rnating, yangilangan kontekst bilan next.ServeHTTP ni chaqiring.`,
			hint2: `HandleAdvancedError da: r.Context().Value(requestIDKey) dan requestID ni oling, code, message, requestID, time.Now().UTC().Format(time.RFC3339) va r.URL.Path bilan AdvancedErrorDetail yarating.`,
			whyItMatters: `So\'rov ID lari va kontekst bilan ilg\'or xatolarni qayta ishlash ishlab chiqarishdagi muammolarni debug qilishni sezilarli darajada osonlashtiradi.`,
			solutionCode: `package api

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/google/uuid"
)

type AdvancedErrorResponse struct {
	Error AdvancedErrorDetail \`json:"error"\`
}

type AdvancedErrorDetail struct {
	Code      string \`json:"code"\`
	Message   string \`json:"message"\`
	RequestID string \`json:"requestId"\`
	Timestamp string \`json:"timestamp"\`
	Path      string \`json:"path"\`
}

type contextKey string

const requestIDKey contextKey = "requestID"

func RequestIDMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestID := uuid.New().String()
		ctx := context.WithValue(r.Context(), requestIDKey, requestID)
		w.Header().Set("X-Request-ID", requestID)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func HandleAdvancedError(w http.ResponseWriter, r *http.Request, code, message string, statusCode int) {
	requestID, _ := r.Context().Value(requestIDKey).(string)
	if requestID == "" {
		requestID = "unknown"
	}
	response := AdvancedErrorResponse{
		Error: AdvancedErrorDetail{
			Code:      code,
			Message:   message,
			RequestID: requestID,
			Timestamp: time.Now().UTC().Format(time.RFC3339),
			Path:      r.URL.Path,
		},
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(response)
}`
		}
	}
};

export default task;
