import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'api-design-resource-naming-nested',
	title: 'RESTful Resource Naming - Nested Resources',
	difficulty: 'easy',
	tags: ['api-design', 'rest', 'go', 'resource-hierarchy'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement nested RESTful resource naming for user posts (one-to-many relationship).

**You will implement:**

1. **Post struct** - Represents a post resource belonging to a user
2. **Nested routes** - Configure routes showing resource relationships
3. **GetUserPosts()** - Get all posts for a specific user
4. **GetUserPost()** - Get a specific post for a specific user

**Key Concepts:**
- **Resource Hierarchy**: Show parent-child relationships in URL
- **Nested Collections**: /users/{userId}/posts represents "posts belonging to user"
- **Contextual Resources**: Resource meaning depends on parent context
- **URL Depth**: Keep nesting shallow (max 2-3 levels)

**Example Usage:**

\`\`\`go
// Correct nested resource naming
GET /api/users/1/posts          // All posts by user 1
GET /api/users/1/posts/5        // Post 5 by user 1
GET /api/users/2/posts          // All posts by user 2 (different context)

// Also valid for direct access
GET /api/posts/5                // Post 5 (any user)

// WRONG - Don't do this
GET /api/posts?userId=1         // Query params for relationships (less RESTful)
GET /api/getUserPosts/1         // Action-based naming
GET /api/users/1/getAllPosts    // Verb in nested route
\`\`\`

**Nested Resource Guidelines:**
- Parent ID comes before child collection
- Each level should be a valid collection endpoint
- Avoid deep nesting (prefer max 2 levels)
- Consider shortcuts for common queries

**Constraints:**
- Implement both nested and direct access routes
- Validate parent resource exists before returning children
- Return 404 if parent (user) doesn't exist`,
	initialCode: `package api

import (
	"encoding/json"
	"net/http"
	"strconv"

	"github.com/go-chi/chi/v5"
)

type Post struct {
	ID      int    \`json:"id"\`
	UserID  int    \`json:"userId"\`
	Title   string \`json:"title"\`
	Content string \`json:"content"\`
}

// Mock data
var posts = []Post{
	{ID: 1, UserID: 1, Title: "First Post", Content: "Hello World"},
	{ID: 2, UserID: 1, Title: "Second Post", Content: "Learning Go"},
	{ID: 3, UserID: 2, Title: "Bob's Post", Content: "REST APIs"},
	{ID: 4, UserID: 2, Title: "Another Post", Content: "API Design"},
}

var users = []int{1, 2, 3} // Simple user ID list

// TODO: Implement SetupNestedRoutes
// Configure nested resource routes:
// GET /api/users/{userId}/posts - get all posts for a user
// GET /api/users/{userId}/posts/{postId} - get specific post for a user
// Use chi.Route for grouping nested routes
func SetupNestedRoutes(r chi.Router) {
	panic("TODO: implement SetupNestedRoutes")
}

// TODO: Implement GetUserPosts
// Returns all posts for a specific user
// Extract userId from URL with chi.URLParam(r, "userId")
// Return 404 if user doesn't exist
// Filter posts by userId
func GetUserPosts(w http.ResponseWriter, r *http.Request) {
	panic("TODO: implement GetUserPosts")
}

// TODO: Implement GetUserPost
// Returns a specific post for a specific user
// Extract both userId and postId from URL
// Validate user exists and post belongs to that user
// Return 404 if user doesn't exist
// Return 404 if post doesn't exist or doesn't belong to user
func GetUserPost(w http.ResponseWriter, r *http.Request) {
	panic("TODO: implement GetUserPost")
}

// Helper function
func userExists(id int) bool {
	for _, uid := range users {
		if uid == id {
			return true
		}
	}
	return false
}`,
	solutionCode: `package api

import (
	"encoding/json"
	"net/http"
	"strconv"

	"github.com/go-chi/chi/v5"
)

type Post struct {
	ID      int    \`json:"id"\`
	UserID  int    \`json:"userId"\`  // Foreign key showing relationship
	Title   string \`json:"title"\`
	Content string \`json:"content"\`
}

var posts = []Post{
	{ID: 1, UserID: 1, Title: "First Post", Content: "Hello World"},
	{ID: 2, UserID: 1, Title: "Second Post", Content: "Learning Go"},
	{ID: 3, UserID: 2, Title: "Bob's Post", Content: "REST APIs"},
	{ID: 4, UserID: 2, Title: "Another Post", Content: "API Design"},
}

var users = []int{1, 2, 3}

func SetupNestedRoutes(r chi.Router) {
	// Nested routes show resource hierarchy: users -> posts
	// Pattern: /parent-collection/{parentId}/child-collection
	r.Route("/api/users/{userId}/posts", func(r chi.Router) {
		r.Get("/", GetUserPosts)           // Collection: all posts for user
		r.Get("/{postId}", GetUserPost)    // Single resource: specific post for user
	})
	// chi.Route groups routes with common prefix
}

func GetUserPosts(w http.ResponseWriter, r *http.Request) {
	// Extract parent resource ID
	userIdStr := chi.URLParam(r, "userId")
	userId, err := strconv.Atoi(userIdStr)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid user ID"})
		return
	}

	// Validate parent resource exists - important for nested routes!
	if !userExists(userId) {
		w.WriteHeader(http.StatusNotFound)  // Parent doesn't exist
		json.NewEncoder(w).Encode(map[string]string{"error": "User not found"})
		return
	}

	// Filter child resources by parent ID
	var userPosts []Post
	for _, post := range posts {
		if post.UserID == userId {
			userPosts = append(userPosts, post)
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(userPosts)  // Returns empty array if no posts
}

func GetUserPost(w http.ResponseWriter, r *http.Request) {
	// Extract both parent and child IDs
	userIdStr := chi.URLParam(r, "userId")
	postIdStr := chi.URLParam(r, "postId")

	userId, err := strconv.Atoi(userIdStr)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid user ID"})
		return
	}

	postId, err := strconv.Atoi(postIdStr)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid post ID"})
		return
	}

	// Validate parent exists first
	if !userExists(userId) {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"error": "User not found"})
		return
	}

	// Find post that belongs to this user
	for _, post := range posts {
		if post.ID == postId && post.UserID == userId {  // Both conditions must match
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(post)
			return
		}
	}

	// Post doesn't exist or doesn't belong to this user
	w.WriteHeader(http.StatusNotFound)
	json.NewEncoder(w).Encode(map[string]string{"error": "Post not found for this user"})
}

func userExists(id int) bool {
	for _, uid := range users {
		if uid == id {
			return true
		}
	}
	return false
}`,
	hint1: `Use r.Route("/api/users/{userId}/posts", func(r chi.Router) {...}) to group nested routes. Inside, add r.Get("/", GetUserPosts) and r.Get("/{postId}", GetUserPost).`,
	hint2: `In both handlers, first extract userId and validate the user exists with userExists(). Then filter posts by userId for GetUserPosts, or check both post.ID == postId AND post.UserID == userId for GetUserPost.`,
	whyItMatters: `Nested resource naming clearly expresses relationships between entities and makes your API intuitive and self-documenting.

**Why Nested Resources Matter:**

**1. Express Relationships Naturally**
URLs should mirror your data model relationships:

\`\`\`go
// One-to-Many: User has many Posts
GET /api/users/1/posts              // "Posts belonging to user 1"
POST /api/users/1/posts             // "Create post for user 1"

// One-to-Many: Post has many Comments
GET /api/posts/5/comments           // "Comments on post 5"
POST /api/posts/5/comments          // "Add comment to post 5"

// Shows clear parent-child relationship without documentation
\`\`\`

**2. Scoped Operations**
Nested routes naturally scope operations to a context:

\`\`\`go
// GOOD: Context is clear from URL
GET /api/users/1/posts              // Obviously limited to user 1's posts
GET /api/users/1/posts/5            // Post 5 in context of user 1

// BAD: Using query params loses clarity
GET /api/posts?userId=1             // Less obvious it's scoped to user
GET /api/posts/5?userId=1           // Why is userId needed for a specific post?
\`\`\`

**3. Authorization and Security**
Nested routes make permission checks obvious:

\`\`\`go
// Easy to implement middleware for nested resources
func UserOwnershipMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        userId := chi.URLParam(r, "userId")
        currentUser := getCurrentUser(r)

        // URL structure makes ownership check natural
        if userId != currentUser.ID {
            http.Error(w, "Forbidden", http.StatusForbidden)
            return
        }
        next.ServeHTTP(w, r)
    })
}

// Apply to all nested user resources
r.Route("/api/users/{userId}", func(r chi.Router) {
    r.Use(UserOwnershipMiddleware)  // Protects all sub-routes
    r.Get("/posts", GetUserPosts)
    r.Get("/orders", GetUserOrders)
    r.Get("/settings", GetUserSettings)
})
\`\`\`

**4. When to Use Nesting vs. Direct Access**

\`\`\`go
// NESTED: When you primarily access resources through parent
GET /api/users/1/posts              // Common: viewing user's posts
POST /api/users/1/posts             // Common: user creating a post
GET /api/posts/5/comments           // Common: viewing comments on a post

// DIRECT: When resource accessed independently
GET /api/posts/5                    // Common: sharing specific post
GET /api/comments/123               // Common: linking to specific comment
GET /api/orders/789                 // Common: tracking order by order number

// BOTH: Support both patterns when needed
GET /api/users/1/posts              // Browse user's posts
GET /api/posts/5                    // Direct access to post 5
// Post 5 might belong to user 1, but sometimes you need direct access
\`\`\`

**5. Nesting Depth Guidelines**

\`\`\`go
// ✅ GOOD: 2 levels deep (recommended maximum)
GET /api/users/1/posts
GET /api/users/1/posts/5
GET /api/posts/5/comments
GET /api/posts/5/comments/10

// ⚠️ ACCEPTABLE: 3 levels (use sparingly)
GET /api/users/1/posts/5/comments

// ❌ BAD: 4+ levels (too deep, hard to use)
GET /api/users/1/posts/5/comments/10/replies
// Instead, provide shortcuts:
GET /api/comments/10/replies        // Direct access to comment replies
\`\`\`

**6. Real-World Examples**

**GitHub API** - Master of nested resources:
\`\`\`go
// Repository belongs to owner
GET /repos/{owner}/{repo}
GET /repos/{owner}/{repo}/issues
GET /repos/{owner}/{repo}/issues/{number}
GET /repos/{owner}/{repo}/issues/{number}/comments

// But also direct access when needed
GET /issues/{id}                    // Cross-repo issue search
GET /notifications                  // User's notifications across all repos
\`\`\`

**Stripe API** - Clear parent-child relationships:
\`\`\`go
// Subscription belongs to customer
GET /customers/{id}/subscriptions
GET /customers/{id}/subscriptions/{sub_id}
POST /customers/{id}/subscriptions

// Card belongs to customer
GET /customers/{id}/sources
POST /customers/{id}/sources
\`\`\`

**7. Implementation Best Practices**

\`\`\`go
// Always validate parent exists first
func GetUserPosts(w http.ResponseWriter, r *http.Request) {
    userId := chi.URLParam(r, "userId")

    // Step 1: Validate parent
    user, err := db.GetUser(userId)
    if err != nil {
        http.Error(w, "User not found", 404)  // Parent doesn't exist
        return
    }

    // Step 2: Get child resources
    posts, err := db.GetPostsByUser(userId)
    // ... return posts
}

// Efficient database queries for nested resources
// BAD: N+1 query problem
for _, user := range users {
    posts := db.GetPostsByUser(user.ID)  // Query per user!
}

// GOOD: Join or eager loading
posts := db.Query(\`
    SELECT p.* FROM posts p
    JOIN users u ON p.user_id = u.id
    WHERE u.id = ?
\`, userId)
\`\`\`

**8. Common Pitfalls**

\`\`\`go
// ❌ MISTAKE 1: Too much nesting
GET /api/countries/1/states/2/cities/3/streets/4/buildings/5
// Fix: Provide direct access
GET /api/buildings/5

// ❌ MISTAKE 2: Not validating parent
GET /api/users/999/posts    // Returns empty array even if user doesn't exist
// Fix: Return 404 if parent doesn't exist

// ❌ MISTAKE 3: Inconsistent behavior
GET /api/users/1/posts/5    // Returns 404 if post not owned by user 1
GET /api/posts/5            // Returns post 5 regardless of owner
GET /api/users/1/posts/5    // Should this return 403 or 404?
// Fix: Document and stick to one behavior (usually 404)

// ❌ MISTAKE 4: Not supporting useful shortcuts
GET /api/users/1/posts/5/comments/10/likes/15
// Fix: Provide shortcuts
GET /api/comments/10        // Direct access
GET /api/likes/15           // Direct access
\`\`\`

**Benefits of Proper Nesting:**
- Self-documenting API structure
- Natural permission boundaries
- Efficient data loading patterns
- Clear service boundaries for microservices
- Easier caching strategies (cache all user's posts)
- Better API discoverability`,
	order: 1,
	testCode: `package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/go-chi/chi/v5"
)

// Test1: GetUserPosts returns posts for user
func Test1(t *testing.T) {
	router := chi.NewRouter()
	SetupNestedRoutes(router)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/users/1/posts", nil)
	router.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Errorf("Expected 200, got %d", w.Code)
	}
}

// Test2: GetUserPosts returns 404 for non-existent user
func Test2(t *testing.T) {
	router := chi.NewRouter()
	SetupNestedRoutes(router)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/users/999/posts", nil)
	router.ServeHTTP(w, r)
	if w.Code != http.StatusNotFound {
		t.Errorf("Expected 404, got %d", w.Code)
	}
}

// Test3: GetUserPosts filters by userId
func Test3(t *testing.T) {
	router := chi.NewRouter()
	SetupNestedRoutes(router)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/users/1/posts", nil)
	router.ServeHTTP(w, r)
	var result []Post
	json.NewDecoder(w.Body).Decode(&result)
	for _, p := range result {
		if p.UserID != 1 {
			t.Error("Post should belong to user 1")
		}
	}
}

// Test4: GetUserPost returns specific post
func Test4(t *testing.T) {
	router := chi.NewRouter()
	SetupNestedRoutes(router)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/users/1/posts/1", nil)
	router.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Errorf("Expected 200, got %d", w.Code)
	}
}

// Test5: GetUserPost returns 404 for wrong user
func Test5(t *testing.T) {
	router := chi.NewRouter()
	SetupNestedRoutes(router)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/users/2/posts/1", nil)
	router.ServeHTTP(w, r)
	if w.Code != http.StatusNotFound {
		t.Errorf("Post 1 belongs to user 1, not user 2")
	}
}

// Test6: userExists returns true for existing user
func Test6(t *testing.T) {
	if !userExists(1) {
		t.Error("User 1 should exist")
	}
}

// Test7: userExists returns false for non-existent user
func Test7(t *testing.T) {
	if userExists(999) {
		t.Error("User 999 should not exist")
	}
}

// Test8: Post struct has correct fields
func Test8(t *testing.T) {
	p := Post{ID: 1, UserID: 1, Title: "Test", Content: "Content"}
	if p.ID != 1 || p.UserID != 1 {
		t.Error("Post fields not set correctly")
	}
}

// Test9: GetUserPosts returns empty array for user with no posts
func Test9(t *testing.T) {
	router := chi.NewRouter()
	SetupNestedRoutes(router)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/users/3/posts", nil)
	router.ServeHTTP(w, r)
	var result []Post
	json.NewDecoder(w.Body).Decode(&result)
	if len(result) != 0 {
		t.Error("User 3 should have no posts")
	}
}

// Test10: GetUserPost returns 404 for non-existent post
func Test10(t *testing.T) {
	router := chi.NewRouter()
	SetupNestedRoutes(router)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/users/1/posts/999", nil)
	router.ServeHTTP(w, r)
	if w.Code != http.StatusNotFound {
		t.Errorf("Expected 404, got %d", w.Code)
	}
}
`,
	translations: {
		ru: {
			title: 'RESTful именование ресурсов - Вложенные ресурсы',
			description: `Реализуйте вложенное RESTful именование ресурсов для постов пользователей (отношение один-ко-многим).

**Вы реализуете:**

1. **Post struct** - Представляет ресурс поста, принадлежащий пользователю
2. **Вложенные маршруты** - Настройте маршруты, показывающие связи ресурсов
3. **GetUserPosts()** - Получить все посты для конкретного пользователя
4. **GetUserPost()** - Получить конкретный пост конкретного пользователя

**Ключевые концепции:**
- **Иерархия ресурсов**: Показывайте родительско-дочерние отношения в URL
- **Вложенные коллекции**: /users/{userId}/posts означает "посты, принадлежащие пользователю"
- **Контекстуальные ресурсы**: Значение ресурса зависит от родительского контекста
- **Глубина URL**: Держите вложенность неглубокой (макс 2-3 уровня)

**Пример использования:**

\`\`\`go
// Правильное вложенное именование ресурсов
GET /api/users/1/posts          // Все посты пользователя 1
GET /api/users/1/posts/5        // Пост 5 пользователя 1
GET /api/users/2/posts          // Все посты пользователя 2 (другой контекст)

// Также допустим прямой доступ
GET /api/posts/5                // Пост 5 (любого пользователя)

// НЕПРАВИЛЬНО - Не делайте так
GET /api/posts?userId=1         // Query параметры для связей (менее RESTful)
GET /api/getUserPosts/1         // Именование основанное на действиях
GET /api/users/1/getAllPosts    // Глагол во вложенном маршруте
\`\`\`

**Рекомендации по вложенным ресурсам:**
- ID родителя идёт перед коллекцией потомков
- Каждый уровень должен быть валидной конечной точкой коллекции
- Избегайте глубокой вложенности (предпочтительно макс 2 уровня)
- Рассмотрите ярлыки для частых запросов

**Ограничения:**
- Реализуйте как вложенные, так и прямые маршруты доступа
- Валидируйте существование родительского ресурса перед возвратом потомков
- Возвращайте 404 если родитель (пользователь) не существует`,
			hint1: `Используйте r.Route("/api/users/{userId}/posts", func(r chi.Router) {...}) для группировки вложенных маршрутов. Внутри добавьте r.Get("/", GetUserPosts) и r.Get("/{postId}", GetUserPost).`,
			hint2: `В обоих обработчиках сначала извлеките userId и проверьте существование пользователя через userExists(). Затем фильтруйте посты по userId для GetUserPosts, или проверьте оба условия post.ID == postId И post.UserID == userId для GetUserPost.`,
			whyItMatters: `Вложенное именование ресурсов чётко выражает отношения между сущностями и делает ваш API интуитивным и самодокументируемым.

**Почему вложенные ресурсы важны:**

**1. Естественное выражение связей**
URL должны отражать отношения вашей модели данных:

\`\`\`go
// Один-ко-многим: У пользователя много постов
GET /api/users/1/posts              // "Посты, принадлежащие пользователю 1"
POST /api/users/1/posts             // "Создать пост для пользователя 1"

// Один-ко-многим: У поста много комментариев
GET /api/posts/5/comments           // "Комментарии к посту 5"
POST /api/posts/5/comments          // "Добавить комментарий к посту 5"
\`\`\`

**2. Операции с областью действия**
Вложенные маршруты естественно ограничивают операции контекстом:

\`\`\`go
// ХОРОШО: Контекст ясен из URL
GET /api/users/1/posts              // Очевидно ограничено постами пользователя 1
GET /api/users/1/posts/5            // Пост 5 в контексте пользователя 1

// ПЛОХО: Query параметры теряют ясность
GET /api/posts?userId=1             // Менее очевидно что ограничено пользователем
\`\`\`

**Лучшие практики:**
- Используйте вложенность для выражения владения
- Валидируйте родительские ресурсы
- Держите вложенность неглубокой (макс 2-3 уровня)
- Предоставляйте прямой доступ когда нужно
- Используйте middleware для проверки владения`,
			solutionCode: `package api

import (
	"encoding/json"
	"net/http"
	"strconv"

	"github.com/go-chi/chi/v5"
)

type Post struct {
	ID      int    \`json:"id"\`
	UserID  int    \`json:"userId"\`  // Внешний ключ показывающий связь
	Title   string \`json:"title"\`
	Content string \`json:"content"\`
}

var posts = []Post{
	{ID: 1, UserID: 1, Title: "First Post", Content: "Hello World"},
	{ID: 2, UserID: 1, Title: "Second Post", Content: "Learning Go"},
	{ID: 3, UserID: 2, Title: "Bob's Post", Content: "REST APIs"},
	{ID: 4, UserID: 2, Title: "Another Post", Content: "API Design"},
}

var users = []int{1, 2, 3}

func SetupNestedRoutes(r chi.Router) {
	// Вложенные маршруты показывают иерархию ресурсов: users -> posts
	r.Route("/api/users/{userId}/posts", func(r chi.Router) {
		r.Get("/", GetUserPosts)           // Коллекция: все посты пользователя
		r.Get("/{postId}", GetUserPost)    // Один ресурс: конкретный пост пользователя
	})
}

func GetUserPosts(w http.ResponseWriter, r *http.Request) {
	// Извлекаем ID родительского ресурса
	userIdStr := chi.URLParam(r, "userId")
	userId, err := strconv.Atoi(userIdStr)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid user ID"})
		return
	}

	// Валидируем существование родительского ресурса
	if !userExists(userId) {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"error": "User not found"})
		return
	}

	// Фильтруем дочерние ресурсы по ID родителя
	var userPosts []Post
	for _, post := range posts {
		if post.UserID == userId {
			userPosts = append(userPosts, post)
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(userPosts)
}

func GetUserPost(w http.ResponseWriter, r *http.Request) {
	userIdStr := chi.URLParam(r, "userId")
	postIdStr := chi.URLParam(r, "postId")

	userId, err := strconv.Atoi(userIdStr)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid user ID"})
		return
	}

	postId, err := strconv.Atoi(postIdStr)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid post ID"})
		return
	}

	if !userExists(userId) {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"error": "User not found"})
		return
	}

	for _, post := range posts {
		if post.ID == postId && post.UserID == userId {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(post)
			return
		}
	}

	w.WriteHeader(http.StatusNotFound)
	json.NewEncoder(w).Encode(map[string]string{"error": "Post not found for this user"})
}

func userExists(id int) bool {
	for _, uid := range users {
		if uid == id {
			return true
		}
	}
	return false
}`
		},
		uz: {
			title: 'RESTful resurs nomlash - Ichki resurslar',
			description: `Foydalanuvchi postlari uchun ichki RESTful resurs nomlashni amalga oshiring (bir-ko\'pga munosabat).

**Siz amalga oshirasiz:**

1. **Post struct** - Foydalanuvchiga tegishli post resursini ifodalaydi
2. **Ichki yo\'llar** - Resurs munosabatlarini ko\'rsatuvchi yo\'llarni sozlang
3. **GetUserPosts()** - Ma\'lum foydalanuvchi uchun barcha postlarni olish
4. **GetUserPost()** - Ma\'lum foydalanuvchining ma\'lum postini olish

**Asosiy tushunchalar:**
- **Resurs ierarxiyasi**: URL da ota-bola munosabatlarini ko\'rsating
- **Ichki kolleksiyalar**: /users/{userId}/posts "foydalanuvchiga tegishli postlar" ni anglatadi
- **Kontekstli resurslar**: Resurs ma\'nosi ota kontekstiga bog\'liq
- **URL chuqurligi**: Ichkilashtirishni sayoz tuting (maks 2-3 daraja)

**Foydalanish misoli:**

\`\`\`go
// To\'g\'ri ichki resurs nomlash
GET /api/users/1/posts          // 1-foydalanuvchining barcha postlari
GET /api/users/1/posts/5        // 1-foydalanuvchining 5-posti
GET /api/users/2/posts          // 2-foydalanuvchining barcha postlari (boshqa kontekst)

// To\'g\'ridan-to\'g\'ri kirish ham to\'g\'ri
GET /api/posts/5                // 5-post (har qanday foydalanuvchi)

// NOTO\'G\'RI - Buni qilmang
GET /api/posts?userId=1         // Munosabatlar uchun query parametrlar (kamroq RESTful)
GET /api/getUserPosts/1         // Harakatga asoslangan nomlash
GET /api/users/1/getAllPosts    // Ichki yo\'lda fe\'l
\`\`\`

**Ichki resurslar bo\'yicha ko\'rsatmalar:**
- Ota ID bola kolleksiyasidan oldin keladi
- Har bir daraja to\'g\'ri kolleksiya endpointi bo\'lishi kerak
- Chuqur ichkilashtirish dan qoching (maks 2 daraja)
- Keng tarqalgan so\'rovlar uchun qisqa yo\'llarni ko\'rib chiqing

**Cheklovlar:**
- Ham ichki, ham to\'g\'ridan-to\'g\'ri kirish yo\'llarini amalga oshiring
- Bolalarni qaytarishdan oldin ota resurs mavjudligini tekshiring
- Ota (foydalanuvchi) mavjud bo\'lmasa 404 qaytaring`,
			hint1: `Ichki yo\'llarni guruhlash uchun r.Route("/api/users/{userId}/posts", func(r chi.Router) {...}) dan foydalaning. Ichida r.Get("/", GetUserPosts) va r.Get("/{postId}", GetUserPost) qo\'shing.`,
			hint2: `Ikkala handlerda ham birinchi navbatda userId ni ajratib oling va userExists() bilan foydalanuvchi mavjudligini tekshiring. Keyin GetUserPosts uchun postlarni userId bo\'yicha filtr qiling, yoki GetUserPost uchun post.ID == postId VA post.UserID == userId ni tekshiring.`,
			whyItMatters: `Ichki resurs nomlash obyektlar o\'rtasidagi munosabatlarni aniq ifodalaydi va API ni intuitiv va o\'z-o\'zidan hujjatlashtirilgan qiladi.

**Ichki resurslar nima uchun muhim:**

**1. Munosabatlarni tabiiy ifodalash**
URL lar ma\'lumotlar modeli munosabatlarini aks ettirishi kerak:

\`\`\`go
// Bir-ko\'pga: Foydalanuvchida ko\'p postlar bor
GET /api/users/1/posts              // "1-foydalanuvchiga tegishli postlar"
POST /api/users/1/posts             // "1-foydalanuvchi uchun post yaratish"

// Bir-ko\'pga: Postda ko\'p sharhlar bor
GET /api/posts/5/comments           // "5-postdagi sharhlar"
POST /api/posts/5/comments          // "5-postga sharh qo\'shish"
\`\`\`

**2. Doiradagi operatsiyalar**
Ichki yo\'llar operatsiyalarni kontekstga tabiiy ravishda cheklaydi:

\`\`\`go
// YAXSHI: Kontekst URL dan aniq
GET /api/users/1/posts              // Aniq 1-foydalanuvchi postlariga cheklangan
GET /api/users/1/posts/5            // 1-foydalanuvchi kontekstida 5-post
\`\`\`

**Eng yaxshi amaliyotlar:**
- Egalikni ifodalash uchun ichkilashtirish dan foydalaning
- Ota resurslarni tekshiring
- Ichkilashtirish ni sayoz tuting (maks 2-3 daraja)
- Kerak bo\'lganda to\'g\'ridan-to\'g\'ri kirish bering
- Egalikni tekshirish uchun middleware ishlating`,
			solutionCode: `package api

import (
	"encoding/json"
	"net/http"
	"strconv"

	"github.com/go-chi/chi/v5"
)

type Post struct {
	ID      int    \`json:"id"\`
	UserID  int    \`json:"userId"\`  // Munosabatni ko\'rsatuvchi tashqi kalit
	Title   string \`json:"title"\`
	Content string \`json:"content"\`
}

var posts = []Post{
	{ID: 1, UserID: 1, Title: "First Post", Content: "Hello World"},
	{ID: 2, UserID: 1, Title: "Second Post", Content: "Learning Go"},
	{ID: 3, UserID: 2, Title: "Bob's Post", Content: "REST APIs"},
	{ID: 4, UserID: 2, Title: "Another Post", Content: "API Design"},
}

var users = []int{1, 2, 3}

func SetupNestedRoutes(r chi.Router) {
	// Ichki yo\'llar resurs ierarxiyasini ko\'rsatadi: users -> posts
	r.Route("/api/users/{userId}/posts", func(r chi.Router) {
		r.Get("/", GetUserPosts)           // Kolleksiya: foydalanuvchining barcha postlari
		r.Get("/{postId}", GetUserPost)    // Bitta resurs: foydalanuvchining ma\'lum posti
	})
}

func GetUserPosts(w http.ResponseWriter, r *http.Request) {
	// Ota resurs ID sini ajratib olish
	userIdStr := chi.URLParam(r, "userId")
	userId, err := strconv.Atoi(userIdStr)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid user ID"})
		return
	}

	// Ota resurs mavjudligini tekshirish
	if !userExists(userId) {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"error": "User not found"})
		return
	}

	// Bola resurslarni ota ID bo\'yicha filtr qilish
	var userPosts []Post
	for _, post := range posts {
		if post.UserID == userId {
			userPosts = append(userPosts, post)
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(userPosts)
}

func GetUserPost(w http.ResponseWriter, r *http.Request) {
	userIdStr := chi.URLParam(r, "userId")
	postIdStr := chi.URLParam(r, "postId")

	userId, err := strconv.Atoi(userIdStr)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid user ID"})
		return
	}

	postId, err := strconv.Atoi(postIdStr)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "Invalid post ID"})
		return
	}

	if !userExists(userId) {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"error": "User not found"})
		return
	}

	for _, post := range posts {
		if post.ID == postId && post.UserID == userId {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(post)
			return
		}
	}

	w.WriteHeader(http.StatusNotFound)
	json.NewEncoder(w).Encode(map[string]string{"error": "Post not found for this user"})
}

func userExists(id int) bool {
	for _, uid := range users {
		if uid == id {
			return true
		}
	}
	return false
}`
		}
	}
};

export default task;
