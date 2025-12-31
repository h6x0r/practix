import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'api-design-request-validation',
	title: 'Request Validation',
	difficulty: 'medium',
	tags: ['api-design', 'validation', 'go'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement comprehensive request validation for API endpoints.

**You will implement:**

1. **ValidateCreateOrder()** - Validate order creation request
2. **Field validation** - Required fields, format validation, range checks
3. **Error responses** - Return detailed validation errors

**Validation Rules:**
- Email: Must be valid format
- Quantity: Must be positive integer
- Price: Must be positive number
- Date: Must be valid ISO 8601 format

**Best Practices:**
- Validate early (fail fast)
- Return all validation errors at once
- Use descriptive error messages
- Never trust client input`,
	initialCode: `package api

import (
	"encoding/json"
	"net/http"
	"regexp"
	"time"
)

type OrderRequest struct {
	CustomerEmail string  \`json:"customerEmail"\`
	ProductID     int     \`json:"productId"\`
	Quantity      int     \`json:"quantity"\`
	Price         float64 \`json:"price"\`
	OrderDate     string  \`json:"orderDate"\`
}

type ValidationError struct {
	Field   string \`json:"field"\`
	Message string \`json:"message"\`
}

// TODO: Implement ValidateCreateOrder
// Validate all fields and return array of validation errors
// 1. customerEmail: required, valid email format
// 2. productId: required (> 0)
// 3. quantity: required, positive (> 0)
// 4. price: required, positive (> 0)
// 5. orderDate: required, valid ISO 8601 date
// Return 400 with all validation errors if any
// Return 201 if validation passes
func ValidateCreateOrder(w http.ResponseWriter, r *http.Request) {
	panic("TODO: implement ValidateCreateOrder")
}`,
	solutionCode: `package api

import (
	"encoding/json"
	"net/http"
	"regexp"
	"time"
)

type OrderRequest struct {
	CustomerEmail string  \`json:"customerEmail"\`
	ProductID     int     \`json:"productId"\`
	Quantity      int     \`json:"quantity"\`
	Price         float64 \`json:"price"\`
	OrderDate     string  \`json:"orderDate"\`
}

type ValidationError struct {
	Field   string \`json:"field"\`
	Message string \`json:"message"\`
}

var emailRegex = regexp.MustCompile(\`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$\`)

func ValidateCreateOrder(w http.ResponseWriter, r *http.Request) {
	var order OrderRequest
	var errors []ValidationError

	if err := json.NewDecoder(r.Body).Decode(&order); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid JSON"})
		return
	}

	// Validate email
	if order.CustomerEmail == "" {
		errors = append(errors, ValidationError{
			Field:   "customerEmail",
			Message: "Email is required",
		})
	} else if !emailRegex.MatchString(order.CustomerEmail) {
		errors = append(errors, ValidationError{
			Field:   "customerEmail",
			Message: "Invalid email format",
		})
	}

	// Validate productId
	if order.ProductID <= 0 {
		errors = append(errors, ValidationError{
			Field:   "productId",
			Message: "Product ID must be positive",
		})
	}

	// Validate quantity
	if order.Quantity <= 0 {
		errors = append(errors, ValidationError{
			Field:   "quantity",
			Message: "Quantity must be positive",
		})
	}

	// Validate price
	if order.Price <= 0 {
		errors = append(errors, ValidationError{
			Field:   "price",
			Message: "Price must be positive",
		})
	}

	// Validate date
	if order.OrderDate == "" {
		errors = append(errors, ValidationError{
			Field:   "orderDate",
			Message: "Order date is required",
		})
	} else {
		_, err := time.Parse(time.RFC3339, order.OrderDate)
		if err != nil {
			errors = append(errors, ValidationError{
				Field:   "orderDate",
				Message: "Invalid date format. Use ISO 8601 format",
			})
		}
	}

	// Return all validation errors
	if len(errors) > 0 {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"error":  "Validation failed",
			"errors": errors,
		})
		return
	}

	// Validation passed
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(map[string]string{
		"message": "Order created successfully",
	})
}`,
	hint1: `Create a slice of ValidationError and append to it for each failed validation. Check email with regex, check numeric fields > 0, parse date with time.Parse(time.RFC3339, order.OrderDate).`,
	hint2: `After all validations, check if len(errors) > 0, return 400 with errors array. If no errors, return 201 success.`,
	whyItMatters: `Request validation prevents bad data from entering your system and provides clear feedback to API clients.

**Why Validation Matters:**

**1. Data Integrity**
Validate early to prevent database corruption:

\`\`\`go
// Without validation
func CreateUser(email string) {
    db.Insert("users", email)  // Stores invalid email!
}

// With validation
func CreateUser(email string) error {
    if !isValidEmail(email) {
        return errors.New("invalid email")
    }
    return db.Insert("users", email)
}
\`\`\`

**2. Return All Errors at Once**

\`\`\`go
// BAD: Return first error only
if email == "" {
    return "email required"
}
if quantity <= 0 {
    return "quantity invalid"
}
// Client must fix and retry multiple times

// GOOD: Return all errors
errors := []string{}
if email == "" {
    errors = append(errors, "email required")
}
if quantity <= 0 {
    errors = append(errors, "quantity invalid")
}
return errors  // Client can fix all at once
\`\`\`

**Best Practices:**
- Validate on server (never trust client)
- Return descriptive error messages
- Use field-level error details
- Validate format, range, and business rules`,
	order: 4,
	testCode: `package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// Test1: Valid order with all fields returns 201
func Test1(t *testing.T) {
	body := bytes.NewBufferString(\`{"customerEmail":"test@test.com","productId":1,"quantity":5,"price":9.99,"orderDate":"2024-01-15T10:00:00Z"}\`)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/api/orders", body)
	ValidateCreateOrder(w, r)
	if w.Code != http.StatusCreated {
		t.Errorf("Expected 201, got %d", w.Code)
	}
}

// Test2: Invalid JSON returns 400
func Test2(t *testing.T) {
	body := bytes.NewBufferString(\`invalid\`)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/api/orders", body)
	ValidateCreateOrder(w, r)
	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected 400, got %d", w.Code)
	}
}

// Test3: Empty email returns 400
func Test3(t *testing.T) {
	body := bytes.NewBufferString(\`{"customerEmail":"","productId":1,"quantity":5,"price":9.99,"orderDate":"2024-01-15T10:00:00Z"}\`)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/api/orders", body)
	ValidateCreateOrder(w, r)
	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected 400, got %d", w.Code)
	}
}

// Test4: Invalid email format (missing @) returns 400
func Test4(t *testing.T) {
	body := bytes.NewBufferString(\`{"customerEmail":"notanemail","productId":1,"quantity":5,"price":9.99,"orderDate":"2024-01-15T10:00:00Z"}\`)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/api/orders", body)
	ValidateCreateOrder(w, r)
	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected 400, got %d", w.Code)
	}
}

// Test5: Zero productId returns 400
func Test5(t *testing.T) {
	body := bytes.NewBufferString(\`{"customerEmail":"test@test.com","productId":0,"quantity":5,"price":9.99,"orderDate":"2024-01-15T10:00:00Z"}\`)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/api/orders", body)
	ValidateCreateOrder(w, r)
	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected 400, got %d", w.Code)
	}
}

// Test6: Negative quantity returns 400
func Test6(t *testing.T) {
	body := bytes.NewBufferString(\`{"customerEmail":"test@test.com","productId":1,"quantity":-5,"price":9.99,"orderDate":"2024-01-15T10:00:00Z"}\`)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/api/orders", body)
	ValidateCreateOrder(w, r)
	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected 400, got %d", w.Code)
	}
}

// Test7: Negative price returns 400
func Test7(t *testing.T) {
	body := bytes.NewBufferString(\`{"customerEmail":"test@test.com","productId":1,"quantity":5,"price":-9.99,"orderDate":"2024-01-15T10:00:00Z"}\`)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/api/orders", body)
	ValidateCreateOrder(w, r)
	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected 400, got %d", w.Code)
	}
}

// Test8: Invalid date format returns 400
func Test8(t *testing.T) {
	body := bytes.NewBufferString(\`{"customerEmail":"test@test.com","productId":1,"quantity":5,"price":9.99,"orderDate":"not-a-date"}\`)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/api/orders", body)
	ValidateCreateOrder(w, r)
	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected 400, got %d", w.Code)
	}
}

// Test9: Wrong date format (missing time) returns 400
func Test9(t *testing.T) {
	body := bytes.NewBufferString(\`{"customerEmail":"test@test.com","productId":1,"quantity":5,"price":9.99,"orderDate":"2024-01-15"}\`)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/api/orders", body)
	ValidateCreateOrder(w, r)
	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected 400, got %d", w.Code)
	}
}

// Test10: Multiple validation errors returned together
func Test10(t *testing.T) {
	body := bytes.NewBufferString(\`{"customerEmail":"","productId":0,"quantity":0,"price":0,"orderDate":""}\`)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/api/orders", body)
	ValidateCreateOrder(w, r)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected 400, got %d", w.Code)
	}

	var result map[string]interface{}
	json.NewDecoder(w.Body).Decode(&result)
	errors, ok := result["errors"].([]interface{})
	if !ok || len(errors) < 4 {
		t.Error("Should return multiple validation errors at once")
	}
}
`,
	translations: {
		ru: {
			title: 'Валидация запросов',
			description: `Реализуйте комплексную валидацию запросов для API эндпоинтов.`,
			hint1: `Создайте срез ValidationError и добавляйте в него при каждой неудачной валидации. Проверьте email с regex, числовые поля > 0, дату через time.Parse.`,
			hint2: `После всех валидаций проверьте len(errors) > 0, верните 400 с массивом ошибок. Если ошибок нет, верните 201.`,
			whyItMatters: `Валидация запросов предотвращает попадание плохих данных в вашу систему и предоставляет чёткую обратную связь клиентам API.`,
			solutionCode: `package api

import (
	"encoding/json"
	"net/http"
	"regexp"
	"time"
)

type OrderRequest struct {
	CustomerEmail string  \`json:"customerEmail"\`
	ProductID     int     \`json:"productId"\`
	Quantity      int     \`json:"quantity"\`
	Price         float64 \`json:"price"\`
	OrderDate     string  \`json:"orderDate"\`
}

type ValidationError struct {
	Field   string \`json:"field"\`
	Message string \`json:"message"\`
}

var emailRegex = regexp.MustCompile(\`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$\`)

func ValidateCreateOrder(w http.ResponseWriter, r *http.Request) {
	var order OrderRequest
	var errors []ValidationError

	if err := json.NewDecoder(r.Body).Decode(&order); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid JSON"})
		return
	}

	if order.CustomerEmail == "" {
		errors = append(errors, ValidationError{Field: "customerEmail", Message: "Email is required"})
	} else if !emailRegex.MatchString(order.CustomerEmail) {
		errors = append(errors, ValidationError{Field: "customerEmail", Message: "Invalid email format"})
	}

	if order.ProductID <= 0 {
		errors = append(errors, ValidationError{Field: "productId", Message: "Product ID must be positive"})
	}

	if order.Quantity <= 0 {
		errors = append(errors, ValidationError{Field: "quantity", Message: "Quantity must be positive"})
	}

	if order.Price <= 0 {
		errors = append(errors, ValidationError{Field: "price", Message: "Price must be positive"})
	}

	if order.OrderDate == "" {
		errors = append(errors, ValidationError{Field: "orderDate", Message: "Order date is required"})
	} else {
		_, err := time.Parse(time.RFC3339, order.OrderDate)
		if err != nil {
			errors = append(errors, ValidationError{Field: "orderDate", Message: "Invalid date format"})
		}
	}

	if len(errors) > 0 {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]interface{}{"error": "Validation failed", "errors": errors})
		return
	}

	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(map[string]string{"message": "Order created successfully"})
}`
		},
		uz: {
			title: 'So\'rovlarni tekshirish',
			description: `API endpointlar uchun keng qamrovli so\'rovlarni tekshirishni amalga oshiring.`,
			hint1: `ValidationError slice yarating va har bir muvaffaqiyatsiz tekshiruvda unga qo\'shing. Email ni regex bilan, raqamli maydonlarni > 0, sanani time.Parse bilan tekshiring.`,
			hint2: `Barcha tekshiruvlardan keyin len(errors) > 0 ni tekshiring, xatolar massivi bilan 400 qaytaring. Xatolar bo\'lmasa, 201 qaytaring.`,
			whyItMatters: `So\'rovlarni tekshirish yomon ma\'lumotlarning tizimingizga kirishini oldini oladi va API mijozlariga aniq javob beradi.`,
			solutionCode: `package api

import (
	"encoding/json"
	"net/http"
	"regexp"
	"time"
)

type OrderRequest struct {
	CustomerEmail string  \`json:"customerEmail"\`
	ProductID     int     \`json:"productId"\`
	Quantity      int     \`json:"quantity"\`
	Price         float64 \`json:"price"\`
	OrderDate     string  \`json:"orderDate"\`
}

type ValidationError struct {
	Field   string \`json:"field"\`
	Message string \`json:"message"\`
}

var emailRegex = regexp.MustCompile(\`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$\`)

func ValidateCreateOrder(w http.ResponseWriter, r *http.Request) {
	var order OrderRequest
	var errors []ValidationError

	if err := json.NewDecoder(r.Body).Decode(&order); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid JSON"})
		return
	}

	if order.CustomerEmail == "" {
		errors = append(errors, ValidationError{Field: "customerEmail", Message: "Email is required"})
	} else if !emailRegex.MatchString(order.CustomerEmail) {
		errors = append(errors, ValidationError{Field: "customerEmail", Message: "Invalid email format"})
	}

	if order.ProductID <= 0 {
		errors = append(errors, ValidationError{Field: "productId", Message: "Product ID must be positive"})
	}

	if order.Quantity <= 0 {
		errors = append(errors, ValidationError{Field: "quantity", Message: "Quantity must be positive"})
	}

	if order.Price <= 0 {
		errors = append(errors, ValidationError{Field: "price", Message: "Price must be positive"})
	}

	if order.OrderDate == "" {
		errors = append(errors, ValidationError{Field: "orderDate", Message: "Order date is required"})
	} else {
		_, err := time.Parse(time.RFC3339, order.OrderDate)
		if err != nil {
			errors = append(errors, ValidationError{Field: "orderDate", Message: "Invalid date format"})
		}
	}

	if len(errors) > 0 {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]interface{}{"error": "Validation failed", "errors": errors})
		return
	}

	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(map[string]string{"message": "Order created successfully"})
}`
		}
	}
};

export default task;
