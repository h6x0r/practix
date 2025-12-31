import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-encodingx-tag-control',
	title: 'Advanced JSON Struct Tags and Field Control',
	difficulty: 'medium',
	tags: ['go', 'json', 'struct-tags', 'omitempty', 'encoding'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Master advanced struct tag techniques to control JSON serialization behavior, including omitempty, field renaming, omission with dash, and embedded structs.

**You will implement:**

**Level 1-3 (Easy → Medium) - User Profile with Optional Fields:**
1. **UserProfile struct** - ID, Username, Email (omitempty), Bio (omitempty), Age (*int for optional)
2. **Omitempty behavior** - Exclude zero-value fields from JSON output
3. **Pointer vs value** - Understand *int vs int for optional fields

**Level 4-5 (Medium) - API Response with Metadata:**
4. **APIResponse struct** - Embedded Metadata, Data interface{}, Error string (omit if empty)
5. **Metadata struct** - Timestamp, RequestID, Version
6. **Embedded struct flattening** - Metadata fields appear at top level

**Level 6-7 (Medium+) - Field Renaming and Exclusion:**
7. **SecureUser struct** - Username, Password (json:"-" to exclude), HashedPassword (json:"password_hash")
8. **Field exclusion with dash** - Prevent sensitive fields from being serialized
9. **Custom field names** - Map Go field names to different JSON keys

**Key Concepts:**
- **omitempty tag**: Exclude fields with zero values (0, false, "", nil, empty slice/map)
- **Field renaming**: Use json:"custom_name" to change output key
- **Field exclusion**: Use json:"-" to completely skip field in JSON
- **Embedded structs**: Anonymous fields are flattened into parent JSON
- **Pointer types**: Distinguish between zero value and absent value

**Example Usage:**

\`\`\`go
// UserProfile with omitempty
profile1 := UserProfile{
    ID:       1,
    Username: "alice",
    Email:    "", // empty, will be omitted
    Bio:      "", // empty, will be omitted
    Age:      nil, // nil, will be omitted
}
data, _ := json.Marshal(profile1)
// {"id": 1, "username": "alice"}
// Email, Bio, Age excluded because zero/nil

profile2 := UserProfile{
    ID:       2,
    Username: "bob",
    Email:    "bob@example.com",
    Bio:      "Software engineer",
    Age:      ptr(30),
}
data, _ := json.Marshal(profile2)
// {"id": 2, "username": "bob", "email": "bob@example.com", "bio": "Software engineer", "age": 30}
// All fields present

// APIResponse with embedded Metadata
response := APIResponse{
    Metadata: Metadata{
        Timestamp: 1609459200,
        RequestID: "req_123",
        Version:   "1.0",
    },
    Data: map[string]interface{}{
        "users": []string{"alice", "bob"},
    },
    Error: "", // empty, will be omitted
}
data, _ := json.Marshal(response)
// {
//   "timestamp": 1609459200,
//   "request_id": "req_123",
//   "version": "1.0",
//   "data": {"users": ["alice", "bob"]}
// }
// Note: Metadata fields flattened, Error omitted

responseWithError := APIResponse{
    Metadata: Metadata{
        Timestamp: 1609459200,
        RequestID: "req_456",
        Version:   "1.0",
    },
    Data:  nil,
    Error: "Database connection failed",
}
data, _ := json.Marshal(responseWithError)
// {
//   "timestamp": 1609459200,
//   "request_id": "req_456",
//   "version": "1.0",
//   "data": null,
//   "error": "Database connection failed"
// }

// SecureUser with excluded fields
user := SecureUser{
    Username:       "admin",
    Password:       "secret123", // NEVER serialized
    HashedPassword: "bcrypt_hash_here",
}
data, _ := json.Marshal(user)
// {"username": "admin", "password_hash": "bcrypt_hash_here"}
// Password field completely excluded from JSON
\`\`\`

**Constraints:**
- UserProfile: Use omitempty for Email, Bio; use *int for Age
- APIResponse: Embed Metadata struct anonymously, use omitempty for Error
- Metadata: Use custom field names (timestamp, request_id, version)
- SecureUser: Use json:"-" for Password, json:"password_hash" for HashedPassword
- Implement ptr() helper: func ptr(i int) *int { return &i }`,
	initialCode: `package encodingx

import "encoding/json"

// TODO: Implement UserProfile struct
// Fields:
// - ID: int (json:"id")
// - Username: string (json:"username")
// - Email: string (json:"email,omitempty") - omit if empty
// - Bio: string (json:"bio,omitempty") - omit if empty
// - Age: *int (json:"age,omitempty") - pointer for optional, omit if nil
type UserProfile struct {
	// TODO: Add fields
}

// TODO: Implement Metadata struct
// Fields:
// - Timestamp: int64 (json:"timestamp")
// - RequestID: string (json:"request_id")
// - Version: string (json:"version")
type Metadata struct {
	// TODO: Add fields
}

// TODO: Implement APIResponse struct
// Fields:
// - Metadata (embedded anonymously - fields flatten to parent)
// - Data: interface{} (json:"data")
// - Error: string (json:"error,omitempty") - omit if empty
type APIResponse struct {
	// TODO: Add fields
}

// TODO: Implement SecureUser struct
// Fields:
// - Username: string (json:"username")
// - Password: string (json:"-") - EXCLUDE from JSON completely
// - HashedPassword: string (json:"password_hash")
type SecureUser struct {
	// TODO: Add fields
}

// ptr is a helper to create int pointer
func ptr(i int) *int {
	return &i
}`,
	testCode: `package encodingx

import (
	"encoding/json"
	"strings"
	"testing"
)

func Test1(t *testing.T) {
	profile := UserProfile{ID: 1, Username: "alice"}
	data, _ := json.Marshal(profile)
	s := string(data)
	if !strings.Contains(s, ` + "`" + `"id":1` + "`" + `) || !strings.Contains(s, ` + "`" + `"username":"alice"` + "`" + `) {
		t.Errorf("unexpected output: %s", s)
	}
}

func Test2(t *testing.T) {
	profile := UserProfile{ID: 1, Username: "alice", Email: "", Bio: ""}
	data, _ := json.Marshal(profile)
	s := string(data)
	if strings.Contains(s, "email") || strings.Contains(s, "bio") {
		t.Errorf("empty fields should be omitted: %s", s)
	}
}

func Test3(t *testing.T) {
	age := 30
	profile := UserProfile{ID: 1, Username: "alice", Age: &age}
	data, _ := json.Marshal(profile)
	if !strings.Contains(string(data), ` + "`" + `"age":30` + "`" + `) {
		t.Errorf("age should be included: %s", string(data))
	}
}

func Test4(t *testing.T) {
	profile := UserProfile{ID: 1, Username: "alice", Age: nil}
	data, _ := json.Marshal(profile)
	if strings.Contains(string(data), "age") {
		t.Errorf("nil age should be omitted: %s", string(data))
	}
}

func Test5(t *testing.T) {
	resp := APIResponse{
		Metadata: Metadata{Timestamp: 123, RequestID: "req_1", Version: "1.0"},
		Data:     "test",
	}
	data, _ := json.Marshal(resp)
	s := string(data)
	if !strings.Contains(s, ` + "`" + `"timestamp":123` + "`" + `) || !strings.Contains(s, ` + "`" + `"request_id":"req_1"` + "`" + `) {
		t.Errorf("metadata should be flattened: %s", s)
	}
}

func Test6(t *testing.T) {
	resp := APIResponse{Metadata: Metadata{Timestamp: 1}, Error: ""}
	data, _ := json.Marshal(resp)
	if strings.Contains(string(data), "error") {
		t.Errorf("empty error should be omitted: %s", string(data))
	}
}

func Test7(t *testing.T) {
	resp := APIResponse{Metadata: Metadata{Timestamp: 1}, Error: "failed"}
	data, _ := json.Marshal(resp)
	if !strings.Contains(string(data), ` + "`" + `"error":"failed"` + "`" + `) {
		t.Errorf("non-empty error should be included: %s", string(data))
	}
}

func Test8(t *testing.T) {
	user := SecureUser{Username: "admin", Password: "secret", HashedPassword: "hash"}
	data, _ := json.Marshal(user)
	s := string(data)
	if strings.Contains(s, "secret") {
		t.Errorf("password should be excluded: %s", s)
	}
}

func Test9(t *testing.T) {
	user := SecureUser{Username: "admin", Password: "secret", HashedPassword: "hash"}
	data, _ := json.Marshal(user)
	if !strings.Contains(string(data), ` + "`" + `"password_hash":"hash"` + "`" + `) {
		t.Errorf("hashed password should use custom name: %s", string(data))
	}
}

func Test10(t *testing.T) {
	p := ptr(42)
	if *p != 42 {
		t.Errorf("ptr helper should return pointer to 42")
	}
}
`,
	solutionCode: `package encodingx

import "encoding/json"

type UserProfile struct {
	ID       int    \`json:"id"\`                // always include ID in output
	Username string \`json:"username"\`          // always include username in output
	Email    string \`json:"email,omitempty"\`   // omit when empty string
	Bio      string \`json:"bio,omitempty"\`     // omit when empty string
	Age      *int   \`json:"age,omitempty"\`     // omit when nil pointer, include when non-nil even if zero
}

type Metadata struct {
	Timestamp int64  \`json:"timestamp"\`   // map Timestamp field to "timestamp" JSON key
	RequestID string \`json:"request_id"\`  // map RequestID field to "request_id" JSON key with underscore
	Version   string \`json:"version"\`     // map Version field to "version" JSON key
}

type APIResponse struct {
	Metadata          // embed anonymously to flatten fields into parent JSON structure
	Data  interface{} \`json:"data"\`            // accept any type for flexible response data
	Error string      \`json:"error,omitempty"\` // omit error field when empty to keep success responses clean
}

type SecureUser struct {
	Username       string \`json:"username"\`      // include username in JSON output
	Password       string \`json:"-"\`             // completely exclude from JSON to prevent accidental password exposure
	HashedPassword string \`json:"password_hash"\` // rename to password_hash in JSON for API consistency
}

func ptr(i int) *int {
	return &i // return pointer to int value for optional fields that need to distinguish zero from nil
}`,
	hint1: `UserProfile: Define struct with ID (int), Username (string), Email (string with json:"email,omitempty"), Bio (string with json:"bio,omitempty"), Age (*int with json:"age,omitempty"). Pointer type for Age allows nil.`,
	hint2: `Metadata: Fields Timestamp (int64, json:"timestamp"), RequestID (string, json:"request_id"), Version (string, json:"version"). APIResponse: Embed Metadata (no field name), Data (interface{}, json:"data"), Error (string, json:"error,omitempty"). SecureUser: Password with json:"-" is excluded, HashedPassword with json:"password_hash" is renamed.`,
	whyItMatters: `Advanced struct tag control is critical for API design, security, and maintaining clean JSON representations while preserving Go's type safety and idiomatic naming conventions.

**Why Struct Tags Matter:**

**1. omitempty - Clean API Responses**

**Problem - Zero Values Clutter JSON:**

\`\`\`go
// BAD - without omitempty
type User struct {
    ID       int    \`json:"id"\`
    Username string \`json:"username"\`
    Email    string \`json:"email"\`
    Bio      string \`json:"bio"\`
}

user := User{ID: 1, Username: "alice"}
json.Marshal(user)
// {
//   "id": 1,
//   "username": "alice",
//   "email": "",        <- unwanted
//   "bio": ""           <- unwanted
// }
// Empty fields make response verbose and confusing
\`\`\`

**Solution - omitempty Excludes Zero Values:**

\`\`\`go
// GOOD - with omitempty
type User struct {
    ID       int    \`json:"id"\`
    Username string \`json:"username"\`
    Email    string \`json:"email,omitempty"\`
    Bio      string \`json:"bio,omitempty"\`
}

user := User{ID: 1, Username: "alice"}
json.Marshal(user)
// {
//   "id": 1,
//   "username": "alice"
// }
// Clean, minimal response
\`\`\`

**Real Use Case**: REST API returns user profiles. Without omitempty, every response includes 50+ empty fields (phone, address, company, etc.). With omitempty, minimal profiles have only 5 fields, reducing response size by 90%.

**2. Zero Value vs Absent - When omitempty Fails**

**Problem - Can't Distinguish Zero from Absent:**

\`\`\`go
type Settings struct {
    NotificationEnabled bool \`json:"notification_enabled,omitempty"\`
    MaxItems            int  \`json:"max_items,omitempty"\`
}

// User wants to DISABLE notifications
settings := Settings{NotificationEnabled: false}
json.Marshal(settings)
// {}
// Field omitted! Can't tell if user set false or didn't provide value

// User wants 0 max items (unlimited)
settings := Settings{MaxItems: 0}
json.Marshal(settings)
// {}
// Field omitted! Can't tell if user wants 0 or didn't provide value
\`\`\`

**Solution - Use Pointers for Optional Fields:**

\`\`\`go
type Settings struct {
    NotificationEnabled *bool \`json:"notification_enabled,omitempty"\`
    MaxItems            *int  \`json:"max_items,omitempty"\`
}

// Explicitly set false
settings := Settings{NotificationEnabled: ptr(false)}
json.Marshal(settings)
// {"notification_enabled": false}
// false is preserved!

// Not provided
settings := Settings{}
json.Marshal(settings)
// {}
// Correctly omitted

// Explicitly set 0
settings := Settings{MaxItems: ptr(0)}
json.Marshal(settings)
// {"max_items": 0}
// Zero is preserved!
\`\`\`

**Rule of thumb:**
- **Value types with omitempty**: Can't explicitly set zero
- **Pointer types with omitempty**: Can set zero, nil means absent

**Real Incident**: PATCH API for user settings. Without pointers, clients couldn't set fields to false/0 - they were always omitted. Changed to pointers, clients can now send explicit false/0 values.

**3. omitempty Behavior by Type**

**What omitempty considers "empty":**

\`\`\`go
type Examples struct {
    BoolField   bool                \`json:"bool,omitempty"\`   // false = empty
    IntField    int                 \`json:"int,omitempty"\`    // 0 = empty
    StringField string              \`json:"string,omitempty"\` // "" = empty
    SliceField  []int               \`json:"slice,omitempty"\`  // nil or len=0 = empty
    MapField    map[string]int      \`json:"map,omitempty"\`    // nil or len=0 = empty
    PtrField    *int                \`json:"ptr,omitempty"\`    // nil = empty
    StructField struct{ Name string } \`json:"struct,omitempty"\` // NEVER empty!
}

// struct fields are NEVER omitted, even if all fields are zero!
ex := Examples{}
json.Marshal(ex)
// {"struct": {"Name": ""}}
// struct field present even though empty!
\`\`\`

**Gotcha - Structs Don't Omit:**

\`\`\`go
type User struct {
    Settings UserSettings \`json:"settings,omitempty"\` // Won't work!
}

type UserSettings struct {
    Theme string
}

user := User{Settings: UserSettings{}} // Empty struct
json.Marshal(user)
// {"settings": {}}
// Settings appears even though empty!
\`\`\`

**Solution - Use Pointer for Optional Structs:**

\`\`\`go
type User struct {
    Settings *UserSettings \`json:"settings,omitempty"\` // Now works!
}

user := User{Settings: nil}
json.Marshal(user)
// {}
// Settings correctly omitted!

user := User{Settings: &UserSettings{Theme: "dark"}}
json.Marshal(user)
// {"settings": {"Theme": "dark"}}
// Settings included when present
\`\`\`

**4. Field Exclusion with json:"-" (Security)**

**Problem - Sensitive Data in JSON:**

\`\`\`go
// DANGER - Password exposed!
type User struct {
    Username string \`json:"username"\`
    Password string \`json:"password"\` // BAD!
}

user := User{Username: "admin", Password: "secret123"}
json.Marshal(user)
// {"username": "admin", "password": "secret123"}
// PASSWORD IN JSON! Security breach!
\`\`\`

**Solution - Exclude with json:"-":**

\`\`\`go
// SAFE - Password excluded
type User struct {
    Username       string \`json:"username"\`
    Password       string \`json:"-"\` // Completely excluded
    HashedPassword string \`json:"password_hash"\`
}

user := User{
    Username:       "admin",
    Password:       "secret123",
    HashedPassword: "bcrypt_hash",
}
json.Marshal(user)
// {"username": "admin", "password_hash": "bcrypt_hash"}
// Password never appears in JSON!
\`\`\`

**Real Incident**: API accidentally returned User struct with plaintext passwords. json:"-" tag was missing. Passwords exposed in logs, monitoring, and client responses. Mandatory password reset for all users. json:"-" prevents this.

**Common Exclusions:**
\`\`\`go
type Model struct {
    Password      string    \`json:"-"\`              // Never expose
    SessionToken  string    \`json:"-"\`              // Security
    InternalID    int       \`json:"-"\`              // Implementation detail
    CachedData    []byte    \`json:"-"\`              // Not relevant to API
    Mutex         sync.Mutex \`json:"-"\`              // Not serializable
    DatabaseConn  *sql.DB   \`json:"-"\`              // Not serializable
}
\`\`\`

**5. Field Renaming - Go Style vs JSON Style**

**Problem - Go vs JSON Naming Conventions:**

\`\`\`go
// Go convention: PascalCase
// JSON convention: snake_case or camelCase

type User struct {
    UserID      int
    FirstName   string
    LastName    string
    EmailAddress string
}

json.Marshal(user)
// {
//   "UserID": 1,         <- Inconsistent
//   "FirstName": "Alice", <- Not API convention
//   "LastName": "Smith",
//   "EmailAddress": "alice@example.com"
// }
\`\`\`

**Solution - Rename with JSON Tags:**

\`\`\`go
type User struct {
    UserID       int    \`json:"user_id"\`       // snake_case
    FirstName    string \`json:"firstName"\`     // camelCase
    LastName     string \`json:"lastName"\`
    EmailAddress string \`json:"email_address"\` // snake_case
}

json.Marshal(user)
// {
//   "user_id": 1,
//   "firstName": "Alice",
//   "lastName": "Smith",
//   "email_address": "alice@example.com"
// }
// Consistent API naming
\`\`\`

**Benefit**: Maintain Go naming conventions (PascalCase) internally while exposing API-friendly names (snake_case/camelCase) externally.

**6. Embedded Structs - JSON Flattening**

**Without Embedding - Nested JSON:**

\`\`\`go
type APIResponse struct {
    Meta Metadata \`json:"metadata"\`
    Data interface{} \`json:"data"\`
}

type Metadata struct {
    Timestamp int64  \`json:"timestamp"\`
    RequestID string \`json:"request_id"\`
}

resp := APIResponse{
    Meta: Metadata{Timestamp: 1609459200, RequestID: "req_123"},
    Data: "result",
}
json.Marshal(resp)
// {
//   "metadata": {           <- Nested
//     "timestamp": 1609459200,
//     "request_id": "req_123"
//   },
//   "data": "result"
// }
\`\`\`

**With Embedding - Flattened JSON:**

\`\`\`go
type APIResponse struct {
    Metadata          // Anonymous field - embeds directly
    Data interface{} \`json:"data"\`
}

type Metadata struct {
    Timestamp int64  \`json:"timestamp"\`
    RequestID string \`json:"request_id"\`
}

resp := APIResponse{
    Metadata: Metadata{Timestamp: 1609459200, RequestID: "req_123"},
    Data:     "result",
}
json.Marshal(resp)
// {
//   "timestamp": 1609459200,  <- Flattened!
//   "request_id": "req_123",  <- Flattened!
//   "data": "result"
// }
// Metadata fields at top level
\`\`\`

**Use Cases:**
- **API responses**: Common fields (timestamp, request_id) at top level
- **Mixins**: Add common behavior (CreatedAt, UpdatedAt, DeletedAt)
- **Composition**: Combine multiple concerns into single struct

**Real Example - Soft Delete Pattern:**

\`\`\`go
type SoftDelete struct {
    DeletedAt *time.Time \`json:"deleted_at,omitempty"\`
}

type User struct {
    ID       int    \`json:"id"\`
    Username string \`json:"username"\`
    SoftDelete      // Embedded
}

user := User{ID: 1, Username: "alice"}
json.Marshal(user)
// {"id": 1, "username": "alice"}
// deleted_at omitted (nil)

user.DeletedAt = ptr(time.Now())
json.Marshal(user)
// {"id": 1, "username": "alice", "deleted_at": "2021-01-01T00:00:00Z"}
// deleted_at included when set
\`\`\`

**7. Tag Combinations and Priority**

**Multiple Options:**

\`\`\`go
type User struct {
    Name  string \`json:"name,omitempty"\`           // Rename + omit empty
    Email string \`json:"email_address,omitempty"\`  // Rename + omit empty
    Admin bool   \`json:"is_admin,string,omitempty"\` // Rename + format as string + omit
}
\`\`\`

**Tag Syntax:**
\`\`\`
\`json:"key,option1,option2"\`
       ^   ^       ^
       |   |       +-- Additional options
       |   +---------- First option (omitempty, string, etc)
       +-------------- Custom key name
\`\`\`

**Common Options:**
- **omitempty**: Omit if zero value
- **string**: Encode number/bool as string
- **-**: Exclude field completely

**8. interface{} for Flexible Data**

**Heterogeneous Responses:**

\`\`\`go
type APIResponse struct {
    Status string      \`json:"status"\`
    Data   interface{} \`json:"data"\` // Can be any type!
}

// Return different types based on endpoint
response1 := APIResponse{Status: "ok", Data: []User{...}}
response2 := APIResponse{Status: "ok", Data: map[string]int{"count": 42}}
response3 := APIResponse{Status: "ok", Data: "simple string"}

// All marshal correctly
json.Marshal(response1) // {"status": "ok", "data": [{"id": 1, ...}]}
json.Marshal(response2) // {"status": "ok", "data": {"count": 42}}
json.Marshal(response3) // {"status": "ok", "data": "simple string"}
\`\`\`

**Benefit**: Single response type handles multiple data shapes.

**9. Testing Struct Tags**

**Verify omitempty Behavior:**

\`\`\`go
func TestUserProfileOmitempty(t *testing.T) {
    // Test zero values omitted
    user := UserProfile{ID: 1, Username: "alice"}
    data, _ := json.Marshal(user)

    if strings.Contains(string(data), "email") {
        t.Error("email should be omitted when empty")
    }

    if strings.Contains(string(data), "bio") {
        t.Error("bio should be omitted when empty")
    }

    // Test values included
    user.Email = "alice@example.com"
    data, _ = json.Marshal(user)

    if !strings.Contains(string(data), "email") {
        t.Error("email should be included when set")
    }
}
\`\`\`

**Verify Exclusion:**

\`\`\`go
func TestSecureUserExclusion(t *testing.T) {
    user := SecureUser{
        Username:       "admin",
        Password:       "secret",
        HashedPassword: "hash",
    }

    data, _ := json.Marshal(user)

    if strings.Contains(string(data), "secret") {
        t.Fatal("Password leaked in JSON!")
    }

    if !strings.Contains(string(data), "password_hash") {
        t.Error("HashedPassword should be included")
    }
}
\`\`\`

**10. Common Patterns**

**Optional Fields Pattern:**

\`\`\`go
type Resource struct {
    ID          int       \`json:"id"\`
    Name        string    \`json:"name"\`
    Description *string   \`json:"description,omitempty"\` // Optional text
    Count       *int      \`json:"count,omitempty"\`       // Optional number (0 vs absent)
    Active      *bool     \`json:"active,omitempty"\`      // Optional bool (false vs absent)
    Tags        []string  \`json:"tags,omitempty"\`        // Optional list
    Metadata    *Metadata \`json:"metadata,omitempty"\`    // Optional nested
}
\`\`\`

**Sensitive Data Pattern:**

\`\`\`go
type User struct {
    // Public fields
    ID       int    \`json:"id"\`
    Username string \`json:"username"\`

    // Sensitive - excluded
    Password      string \`json:"-"\`
    PasswordHash  string \`json:"-"\`
    SessionToken  string \`json:"-"\`

    // Controlled sensitive
    Email string \`json:"email,omitempty"\` // Only if user opts in
}
\`\`\`

**API Response Pattern:**

\`\`\`go
type Response struct {
    Metadata                     // Flatten common fields
    Data    interface{} \`json:"data,omitempty"\`
    Error   string      \`json:"error,omitempty"\`
}

type Metadata struct {
    Timestamp int64  \`json:"timestamp"\`
    RequestID string \`json:"request_id"\`
    Version   string \`json:"version"\`
}
\`\`\`

**Key Takeaways:**
- Use omitempty to exclude zero-value fields and reduce response size
- Use pointer types (*T) with omitempty to distinguish zero from absent
- Use json:"-" to completely exclude sensitive fields (passwords, tokens)
- Use custom names to match API conventions (snake_case/camelCase)
- Embed structs anonymously to flatten JSON structure
- interface{} allows flexible data shapes in responses
- Struct values with omitempty are never omitted (use pointers)
- Test that excluded fields never appear in JSON output`,
	order: 3,
	translations: {
		ru: {
			title: 'Продвинутые JSON Struct Tags и контроль полей',
			description: `Освойте продвинутые техники struct tags для контроля поведения JSON сериализации, включая omitempty, переименование полей, исключение с тире и встроенные структуры.

**Вы реализуете:**

**Уровни 1-3 (Лёгкий → Средний) — Профиль пользователя с опциональными полями:**
1. **UserProfile struct** — ID, Username, Email (omitempty), Bio (omitempty), Age (*int для optional)
2. **Поведение Omitempty** — Исключение полей с нулевыми значениями из JSON вывода
3. **Pointer vs value** — Понимание *int vs int для опциональных полей

**Уровни 4-5 (Средний) — API Response с Metadata:**
4. **APIResponse struct** — Встроенные Metadata, Data interface{}, Error string (пропустить если пусто)
5. **Metadata struct** — Timestamp, RequestID, Version
6. **Сглаживание встроенной структуры** — Поля Metadata появляются на верхнем уровне

**Уровни 6-7 (Средний+) — Переименование и исключение полей:**
7. **SecureUser struct** — Username, Password (json:"-" для исключения), HashedPassword (json:"password_hash")
8. **Исключение полей с тире** — Предотвращение сериализации чувствительных полей
9. **Пользовательские имена полей** — Отображение имён полей Go на разные JSON ключи

**Ключевые концепции:**
- **omitempty тег**: Исключение полей с нулевыми значениями (0, false, "", nil, пустой slice/map)
- **Переименование полей**: Используйте json:"custom_name" для изменения выходного ключа
- **Исключение полей**: Используйте json:"-" для полного пропуска поля в JSON
- **Встроенные структуры**: Анонимные поля сглаживаются в родительский JSON
- **Типы указателей**: Различие между нулевым значением и отсутствующим значением

**Пример использования:**

\`\`\`go
// UserProfile с omitempty
profile1 := UserProfile{
    ID:       1,
    Username: "alice",
    Email:    "", // пусто, будет пропущено
    Bio:      "", // пусто, будет пропущено
    Age:      nil, // nil, будет пропущено
}
data, _ := json.Marshal(profile1)
// {"id": 1, "username": "alice"}
// Email, Bio, Age исключены потому что zero/nil

profile2 := UserProfile{
    ID:       2,
    Username: "bob",
    Email:    "bob@example.com",
    Bio:      "Software engineer",
    Age:      ptr(30),
}
data, _ := json.Marshal(profile2)
// {"id": 2, "username": "bob", "email": "bob@example.com", "bio": "Software engineer", "age": 30}
// Все поля присутствуют

// APIResponse со встроенными Metadata
response := APIResponse{
    Metadata: Metadata{
        Timestamp: 1609459200,
        RequestID: "req_123",
        Version:   "1.0",
    },
    Data: map[string]interface{}{
        "users": []string{"alice", "bob"},
    },
    Error: "", // пусто, будет пропущено
}
data, _ := json.Marshal(response)
// {
//   "timestamp": 1609459200,
//   "request_id": "req_123",
//   "version": "1.0",
//   "data": {"users": ["alice", "bob"]}
// }
// Примечание: Поля Metadata сглажены, Error пропущен

responseWithError := APIResponse{
    Metadata: Metadata{
        Timestamp: 1609459200,
        RequestID: "req_456",
        Version:   "1.0",
    },
    Data:  nil,
    Error: "Database connection failed",
}
data, _ := json.Marshal(responseWithError)
// {
//   "timestamp": 1609459200,
//   "request_id": "req_456",
//   "version": "1.0",
//   "data": null,
//   "error": "Database connection failed"
// }

// SecureUser с исключёнными полями
user := SecureUser{
    Username:       "admin",
    Password:       "secret123", // НИКОГДА не сериализуется
    HashedPassword: "bcrypt_hash_here",
}
data, _ := json.Marshal(user)
// {"username": "admin", "password_hash": "bcrypt_hash_here"}
// Поле Password полностью исключено из JSON
\`\`\`

**Ограничения:**
- UserProfile: Используйте omitempty для Email, Bio; используйте *int для Age
- APIResponse: Встройте Metadata структуру анонимно, используйте omitempty для Error
- Metadata: Используйте пользовательские имена полей (timestamp, request_id, version)
- SecureUser: Используйте json:"-" для Password, json:"password_hash" для HashedPassword
- Реализуйте ptr() helper: func ptr(i int) *int { return &i }`,
			hint1: `UserProfile: Определите struct с ID (int), Username (string), Email (string с json:"email,omitempty"), Bio (string с json:"bio,omitempty"), Age (*int с json:"age,omitempty"). Тип указателя для Age позволяет nil.`,
			hint2: `Metadata: Поля Timestamp (int64, json:"timestamp"), RequestID (string, json:"request_id"), Version (string, json:"version"). APIResponse: Встройте Metadata (без имени поля), Data (interface{}, json:"data"), Error (string, json:"error,omitempty"). SecureUser: Password с json:"-" исключён, HashedPassword с json:"password_hash" переименован.`,
			whyItMatters: `Продвинутый контроль struct tags критически важен для дизайна API, безопасности и поддержания чистых JSON представлений при сохранении типобезопасности Go и идиоматических соглашений об именовании.

**Почему это важно:**

**1. omitempty — чистые API ответы**

**Проблема — нулевые значения засоряют JSON:**

\`\`\`go
// ПЛОХО — без omitempty
type User struct {
    ID       int    \`json:"id"\`
    Username string \`json:"username"\`
    Email    string \`json:"email"\`
    Bio      string \`json:"bio"\`
}

user := User{ID: 1, Username: "alice"}
json.Marshal(user)
// {
//   "id": 1,
//   "username": "alice",
//   "email": "",        <- нежелательно
//   "bio": ""           <- нежелательно
// }
// Пустые поля делают ответ многословным и запутанным
\`\`\`

**Решение — omitempty исключает нулевые значения:**

\`\`\`go
// ХОРОШО — с omitempty
type User struct {
    ID       int    \`json:"id"\`
    Username string \`json:"username"\`
    Email    string \`json:"email,omitempty"\`
    Bio      string \`json:"bio,omitempty"\`
}

user := User{ID: 1, Username: "alice"}
json.Marshal(user)
// {
//   "id": 1,
//   "username": "alice"
// }
// Чистый, минимальный ответ
\`\`\`

**Реальный пример**: REST API возвращает профили пользователей. Без omitempty каждый ответ включает 50+ пустых полей (телефон, адрес, компания и т.д.). С omitempty минимальные профили имеют только 5 полей, уменьшая размер ответа на 90%.

**2. Нулевое значение vs отсутствие — когда omitempty не работает**

**Проблема — не можем различить ноль от отсутствия:**

\`\`\`go
type Settings struct {
    NotificationEnabled bool \`json:"notification_enabled,omitempty"\`
    MaxItems            int  \`json:"max_items,omitempty"\`
}

// Пользователь хочет ВЫКЛЮЧИТЬ уведомления
settings := Settings{NotificationEnabled: false}
json.Marshal(settings)
// {}
// Поле пропущено! Не можем сказать, установил ли пользователь false или не предоставил значение

// Пользователь хочет 0 максимальных элементов (неограниченно)
settings := Settings{MaxItems: 0}
json.Marshal(settings)
// {}
// Поле пропущено! Не можем сказать, хочет ли пользователь 0 или не предоставил значение
\`\`\`

**Решение — используйте указатели для опциональных полей:**

\`\`\`go
type Settings struct {
    NotificationEnabled *bool \`json:"notification_enabled,omitempty"\`
    MaxItems            *int  \`json:"max_items,omitempty"\`
}

// Явно установлено false
settings := Settings{NotificationEnabled: ptr(false)}
json.Marshal(settings)
// {"notification_enabled": false}
// false сохранено!

// Не предоставлено
settings := Settings{}
json.Marshal(settings)
// {}
// Правильно пропущено

// Явно установлено 0
settings := Settings{MaxItems: ptr(0)}
json.Marshal(settings)
// {"max_items": 0}
// Ноль сохранён!
\`\`\`

**Эмпирическое правило:**
- **Value типы с omitempty**: Не можете явно установить ноль
- **Pointer типы с omitempty**: Можете установить ноль, nil означает отсутствие

**Реальный инцидент**: PATCH API для настроек пользователя. Без указателей клиенты не могли установить поля в false/0 — они всегда пропускались. Изменили на указатели, клиенты теперь могут отправлять явные false/0 значения.

**Ключевые выводы:**
- Используйте omitempty для исключения полей с нулевыми значениями и уменьшения размера ответа
- Используйте типы указателей (*T) с omitempty для различения нуля от отсутствия
- Используйте json:"-" для полного исключения чувствительных полей (пароли, токены)
- Используйте пользовательские имена для соответствия соглашениям API (snake_case/camelCase)
- Встраивайте структуры анонимно для сглаживания JSON структуры
- interface{} позволяет гибкие формы данных в ответах
- Значения структур с omitempty никогда не пропускаются (используйте указатели)
- Тестируйте, что исключённые поля никогда не появляются в JSON выводе`,
			solutionCode: `package encodingx

import "encoding/json"

type UserProfile struct {
	ID       int    \`json:"id"\`                // всегда включаем ID в вывод
	Username string \`json:"username"\`          // всегда включаем username в вывод
	Email    string \`json:"email,omitempty"\`   // пропускаем когда пустая строка
	Bio      string \`json:"bio,omitempty"\`     // пропускаем когда пустая строка
	Age      *int   \`json:"age,omitempty"\`     // пропускаем когда nil указатель, включаем когда не-nil даже если ноль
}

type Metadata struct {
	Timestamp int64  \`json:"timestamp"\`   // отображаем поле Timestamp на JSON ключ "timestamp"
	RequestID string \`json:"request_id"\`  // отображаем поле RequestID на JSON ключ "request_id" с подчёркиванием
	Version   string \`json:"version"\`     // отображаем поле Version на JSON ключ "version"
}

type APIResponse struct {
	Metadata          // встраиваем анонимно для сглаживания полей в родительскую JSON структуру
	Data  interface{} \`json:"data"\`            // принимаем любой тип для гибких данных ответа
	Error string      \`json:"error,omitempty"\` // пропускаем поле error когда пусто для чистых успешных ответов
}

type SecureUser struct {
	Username       string \`json:"username"\`      // включаем username в JSON вывод
	Password       string \`json:"-"\`             // полностью исключаем из JSON для предотвращения случайного раскрытия пароля
	HashedPassword string \`json:"password_hash"\` // переименовываем в password_hash в JSON для консистентности API
}

func ptr(i int) *int {
	return &i // возвращаем указатель на int значение для опциональных полей которым нужно различать ноль от nil
}`
		},
		uz: {
			title: `Ilg'or JSON Struct Teglari va maydon nazorati`,
			description: `JSON serializatsiya xatti-harakatini nazorat qilish uchun ilg'or struct tag texnikalarini o'rganing, shu jumladan omitempty, maydonlarni o'zgartirish, tire bilan chiqarish va o'rnatilgan strukturalar.

**Siz amalga oshirasiz:**

**1-3 Daraja (Oson → O'rta) — Ixtiyoriy maydonlar bilan foydalanuvchi profili:**
1. **UserProfile struct** — ID, Username, Email (omitempty), Bio (omitempty), Age (*int ixtiyoriy uchun)
2. **Omitempty xatti-harakati** — JSON chiqarishdan nol qiymatli maydonlarni chiqarib tashlash
3. **Pointer vs value** — Ixtiyoriy maydonlar uchun *int vs int ni tushunish

**4-5 Daraja (O'rta) — Metadata bilan API Response:**
4. **APIResponse struct** — O'rnatilgan Metadata, Data interface{}, Error string (bo'sh bo'lsa chiqarish)
5. **Metadata struct** — Timestamp, RequestID, Version
6. **O'rnatilgan struktura tekislash** — Metadata maydonlari yuqori darajada ko'rinadi

**6-7 Daraja (O'rta+) — Maydonlarni o'zgartirish va chiqarib tashlash:**
7. **SecureUser struct** — Username, Password (json:"-" chiqarish uchun), HashedPassword (json:"password_hash")
8. **Tire bilan maydonni chiqarib tashlash** — Sezgir maydonlarning serializatsiyasini oldini olish
9. **Maxsus maydon nomlari** — Go maydon nomlarini turli JSON kalitlariga moslashtirish

**Asosiy tushunchalar:**
- **omitempty tegi**: Nol qiymatlari bilan maydonlarni chiqarish (0, false, "", nil, bo'sh slice/map)
- **Maydonlarni o'zgartirish**: Chiqish kalitini o'zgartirish uchun json:"custom_name" dan foydalaning
- **Maydonlarni chiqarish**: JSON da maydonni to'liq o'tkazib yuborish uchun json:"-" dan foydalaning
- **O'rnatilgan strukturalar**: Anonim maydonlar ota JSON ga tekislanadi
- **Ko'rsatkich turlari**: Nol qiymat va yo'q qiymat o'rtasidagi farq

**Foydalanish misoli:**

\`\`\`go
// omitempty bilan UserProfile
profile1 := UserProfile{
    ID:       1,
    Username: "alice",
    Email:    "", // bo'sh, o'tkazib yuboriladi
    Bio:      "", // bo'sh, o'tkazib yuboriladi
    Age:      nil, // nil, o'tkazib yuboriladi
}
data, _ := json.Marshal(profile1)
// {"id": 1, "username": "alice"}
// Email, Bio, Age chiqarildi chunki zero/nil

profile2 := UserProfile{
    ID:       2,
    Username: "bob",
    Email:    "bob@example.com",
    Bio:      "Software engineer",
    Age:      ptr(30),
}
data, _ := json.Marshal(profile2)
// {"id": 2, "username": "bob", "email": "bob@example.com", "bio": "Software engineer", "age": 30}
// Barcha maydonlar mavjud

// O'rnatilgan Metadata bilan APIResponse
response := APIResponse{
    Metadata: Metadata{
        Timestamp: 1609459200,
        RequestID: "req_123",
        Version:   "1.0",
    },
    Data: map[string]interface{}{
        "users": []string{"alice", "bob"},
    },
    Error: "", // bo'sh, o'tkazib yuboriladi
}
data, _ := json.Marshal(response)
// {
//   "timestamp": 1609459200,
//   "request_id": "req_123",
//   "version": "1.0",
//   "data": {"users": ["alice", "bob"]}
// }
// Eslatma: Metadata maydonlari tekislandi, Error o'tkazib yuborildi

responseWithError := APIResponse{
    Metadata: Metadata{
        Timestamp: 1609459200,
        RequestID: "req_456",
        Version:   "1.0",
    },
    Data:  nil,
    Error: "Database connection failed",
}
data, _ := json.Marshal(responseWithError)
// {
//   "timestamp": 1609459200,
//   "request_id": "req_456",
//   "version": "1.0",
//   "data": null,
//   "error": "Database connection failed"
// }

// Chiqarilgan maydonlar bilan SecureUser
user := SecureUser{
    Username:       "admin",
    Password:       "secret123", // HECH QACHON serializatsiya qilinmaydi
    HashedPassword: "bcrypt_hash_here",
}
data, _ := json.Marshal(user)
// {"username": "admin", "password_hash": "bcrypt_hash_here"}
// Password maydoni JSON dan to'liq chiqarildi
\`\`\`

**Cheklovlar:**
- UserProfile: Email, Bio uchun omitempty dan foydalaning; Age uchun *int dan foydalaning
- APIResponse: Metadata strukturani anonim o'rnating, Error uchun omitempty dan foydalaning
- Metadata: Maxsus maydon nomlaridan foydalaning (timestamp, request_id, version)
- SecureUser: Password uchun json:"-" dan foydalaning, HashedPassword uchun json:"password_hash" dan foydalaning
- ptr() helper ni amalga oshiring: func ptr(i int) *int { return &i }`,
			hint1: `UserProfile: ID (int), Username (string), Email (json:"email,omitempty" bilan string), Bio (json:"bio,omitempty" bilan string), Age (json:"age,omitempty" bilan *int) bilan struct aniqlang. Age uchun ko'rsatkich turi nil ga imkon beradi.`,
			hint2: `Metadata: Timestamp (int64, json:"timestamp"), RequestID (string, json:"request_id"), Version (string, json:"version") maydonlari. APIResponse: Metadata ni o'rnating (maydon nomi yo'q), Data (interface{}, json:"data"), Error (string, json:"error,omitempty"). SecureUser: json:"-" bilan Password chiqarildi, json:"password_hash" bilan HashedPassword o'zgartirildi.`,
			whyItMatters: `Ilg'or struct tag nazorati API dizayni, xavfsizlik va Go ning tur xavfsizligi va idiomatik nomlash konventsiyalarini saqlab qolgan holda toza JSON tasvirlarini saqlash uchun muhimdir.

**Nima uchun bu muhim:**

**1. omitempty — toza API javoblari**

**Muammo — nol qiymatlari JSON ni bezovta qiladi:**

\`\`\`go
// YOMON — omitempty yo'q
type User struct {
    ID       int    \`json:"id"\`
    Username string \`json:"username"\`
    Email    string \`json:"email"\`
    Bio      string \`json:"bio"\`
}

user := User{ID: 1, Username: "alice"}
json.Marshal(user)
// {
//   "id": 1,
//   "username": "alice",
//   "email": "",        <- keraksiz
//   "bio": ""           <- keraksiz
// }
// Bo'sh maydonlar javobni ko'p so'zli va chalkash qiladi
\`\`\`

**Yechim — omitempty nol qiymatlarni chiqaradi:**

\`\`\`go
// YAXSHI — omitempty bilan
type User struct {
    ID       int    \`json:"id"\`
    Username string \`json:"username"\`
    Email    string \`json:"email,omitempty"\`
    Bio      string \`json:"bio,omitempty"\`
}

user := User{ID: 1, Username: "alice"}
json.Marshal(user)
// {
//   "id": 1,
//   "username": "alice"
// }
// Toza, minimal javob
\`\`\`

**Haqiqiy misol**: REST API foydalanuvchi profillarini qaytaradi. omitempty yo'q bo'lsa, har bir javob 50+ bo'sh maydonlarni o'z ichiga oladi (telefon, manzil, kompaniya va boshqalar). omitempty bilan minimal profillar faqat 5 ta maydonga ega, javob hajmini 90% ga kamaytiradi.

**2. Nol qiymat vs yo'qligi — omitempty ishlamasa**

**Muammo — nolni yo'qlikdan ajrata olmaymiz:**

\`\`\`go
type Settings struct {
    NotificationEnabled bool \`json:"notification_enabled,omitempty"\`
    MaxItems            int  \`json:"max_items,omitempty"\`
}

// Foydalanuvchi bildirishnomalarni O'CHIRMOQCHI
settings := Settings{NotificationEnabled: false}
json.Marshal(settings)
// {}
// Maydon o'tkazib yuborildi! Foydalanuvchi false ni o'rnatdimi yoki qiymat bermagani aniq emas

// Foydalanuvchi 0 maksimal elementlarni xohlaydi (cheksiz)
settings := Settings{MaxItems: 0}
json.Marshal(settings)
// {}
// Maydon o'tkazib yuborildi! Foydalanuvchi 0 ni xohlaydimi yoki qiymat bermagani aniq emas
\`\`\`

**Yechim — ixtiyoriy maydonlar uchun ko'rsatkichlardan foydalaning:**

\`\`\`go
type Settings struct {
    NotificationEnabled *bool \`json:"notification_enabled,omitempty"\`
    MaxItems            *int  \`json:"max_items,omitempty"\`
}

// Aniq false o'rnatildi
settings := Settings{NotificationEnabled: ptr(false)}
json.Marshal(settings)
// {"notification_enabled": false}
// false saqlanди!

// Berilmagan
settings := Settings{}
json.Marshal(settings)
// {}
// To'g'ri o'tkazib yuborildi

// Aniq 0 o'rnatildi
settings := Settings{MaxItems: ptr(0)}
json.Marshal(settings)
// {"max_items": 0}
// Nol saqlanди!
\`\`\`

**Qoida:**
- **omitempty bilan Value turlari**: Nolni aniq o'rnata olmaysiz
- **omitempty bilan Pointer turlari**: Nolni o'rnatishingiz mumkin, nil yo'qlikni bildiradi

**Haqiqiy hodisa**: Foydalanuvchi sozlamalari uchun PATCH API. Ko'rsatkichlarsiz mijozlar maydonlarni false/0 ga o'rnata olmadi — ular doimo o'tkazib yuborildi. Ko'rsatkichlarga o'zgartirildi, mijozlar endi aniq false/0 qiymatlarini yuborishi mumkin.

**Asosiy xulosalar:**
- Nol qiymatli maydonlarni chiqarish va javob hajmini kamaytirish uchun omitempty dan foydalaning
- Nolni yo'qlikdan ajratish uchun omitempty bilan ko'rsatkich turlaridan (*T) foydalaning
- Sezgir maydonlarni to'liq chiqarish uchun json:"-" dan foydalaning (parollar, tokenlar)
- API konventsiyalariga mos kelish uchun maxsus nomlardan foydalaning (snake_case/camelCase)
- JSON strukturani tekislash uchun strukturalarni anonim o'rnating
- interface{} javoblarda moslashuvchan ma'lumot shakllariga imkon beradi
- omitempty bilan struktura qiymatlari hech qachon o'tkazib yuborilmaydi (ko'rsatkichlardan foydalaning)
- Chiqarilgan maydonlar JSON chiqarishda hech qachon ko'rinmasligini sinab ko'ring`,
			solutionCode: `package encodingx

import "encoding/json"

type UserProfile struct {
	ID       int    \`json:"id"\`                // har doim chiqishda ID ni kiriting
	Username string \`json:"username"\`          // har doim chiqishda username ni kiriting
	Email    string \`json:"email,omitempty"\`   // bo'sh qator bo'lganda o'tkazib yuboring
	Bio      string \`json:"bio,omitempty"\`     // bo'sh qator bo'lganda o'tkazib yuboring
	Age      *int   \`json:"age,omitempty"\`     // nil ko'rsatkich bo'lganda o'tkazib yuboring, nil bo'lmasa nol bo'lsa ham kiriting
}

type Metadata struct {
	Timestamp int64  \`json:"timestamp"\`   // Timestamp maydonini "timestamp" JSON kalitiga moslashtirish
	RequestID string \`json:"request_id"\`  // RequestID maydonini pastki chiziqli "request_id" JSON kalitiga moslashtirish
	Version   string \`json:"version"\`     // Version maydonini "version" JSON kalitiga moslashtirish
}

type APIResponse struct {
	Metadata          // maydonlarni ota JSON strukturasiga tekislash uchun anonim o'rnatish
	Data  interface{} \`json:"data"\`            // moslashuvchan javob ma'lumotlari uchun har qanday turni qabul qilish
	Error string      \`json:"error,omitempty"\` // muvaffaqiyatli javoblarni toza saqlash uchun bo'sh bo'lganda error maydonini o'tkazib yuboring
}

type SecureUser struct {
	Username       string \`json:"username"\`      // JSON chiqarishda username ni kiriting
	Password       string \`json:"-"\`             // tasodifiy parol oshkor qilishning oldini olish uchun JSON dan to'liq chiqarib tashlash
	HashedPassword string \`json:"password_hash"\` // API izchilligi uchun JSON da password_hash ga o'zgartirish
}

func ptr(i int) *int {
	return &i // nolni nil dan ajratishi kerak bo'lgan ixtiyoriy maydonlar uchun int qiymatiga ko'rsatkich qaytarish
}`
		}
	}
};

export default task;
