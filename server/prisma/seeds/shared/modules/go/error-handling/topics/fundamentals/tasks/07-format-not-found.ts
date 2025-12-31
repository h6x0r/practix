import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-format-not-found',
	title: 'Domain-Specific Error Formatting',
	difficulty: 'medium',	tags: ['go', 'errors', 'formatting'],
	estimatedTime: '25m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **FormatNotFound()** to create domain-specific NotFound errors.

**Requirements:**
1. Create function \`FormatNotFound(id string) error\`
2. Use \`fmt.Errorf()\` with \`%w\` verb to wrap ErrNotFound
3. Include entity identifier in error message
4. Format as: \`"entity <id>: %w"\`

**Example:**
\`\`\`go
var ErrNotFound = errors.New("not found")

func GetUser(id string) (*User, error) {
    user, err := db.Query(id)
    if err == sql.ErrNoRows {
        return nil, FormatNotFound(id)
    }
    return user, err
}

// Error message: "entity user-123: not found"
// errors.Is(err, ErrNotFound) returns true
\`\`\`

**Constraints:**
- Must use %w verb (not %v) to preserve error chain
- Must include id parameter in message`,
	initialCode: `package errorsx

import (
	"errors"
	"fmt"
)

var ErrNotFound = errors.New("not found")

// TODO: Implement FormatNotFound
func FormatNotFound($2) error {
	return nil // TODO: Implement
}`,
	solutionCode: `package errorsx

import (
	"errors"
	"fmt"
)

var ErrNotFound = errors.New("not found")

func FormatNotFound(id string) error {
	return fmt.Errorf("entity %s: %w", id, ErrNotFound)  // %w wraps error, preserving chain
}`,
		testCode: `package errorsx

import (
	"errors"
	"strings"
	"testing"
)

func Test1(t *testing.T) {
	// FormatNotFound returns non-nil error
	result := FormatNotFound("user-123")
	if result == nil {
		t.Error("expected non-nil error")
	}
}

func Test2(t *testing.T) {
	// FormatNotFound includes id in message
	result := FormatNotFound("user-123")
	msg := result.Error()
	if !strings.Contains(msg, "user-123") {
		t.Errorf("expected message to contain id, got: %s", msg)
	}
}

func Test3(t *testing.T) {
	// FormatNotFound includes "not found" in message
	result := FormatNotFound("user-123")
	msg := result.Error()
	if !strings.Contains(msg, "not found") {
		t.Errorf("expected message to contain 'not found', got: %s", msg)
	}
}

func Test4(t *testing.T) {
	// FormatNotFound works with errors.Is
	result := FormatNotFound("user-123")
	if !errors.Is(result, ErrNotFound) {
		t.Error("expected errors.Is to find ErrNotFound")
	}
}

func Test5(t *testing.T) {
	// FormatNotFound with empty id
	result := FormatNotFound("")
	if result == nil {
		t.Error("expected non-nil error even with empty id")
	}
}

func Test6(t *testing.T) {
	// FormatNotFound with special characters
	result := FormatNotFound("user@test.com")
	msg := result.Error()
	if !strings.Contains(msg, "user@test.com") {
		t.Errorf("expected message to contain special chars, got: %s", msg)
	}
}

func Test7(t *testing.T) {
	// FormatNotFound format includes entity prefix
	result := FormatNotFound("123")
	msg := result.Error()
	if !strings.Contains(msg, "entity") {
		t.Errorf("expected 'entity' in message, got: %s", msg)
	}
}

func Test8(t *testing.T) {
	// FormatNotFound with long id
	longID := strings.Repeat("a", 100)
	result := FormatNotFound(longID)
	msg := result.Error()
	if !strings.Contains(msg, longID) {
		t.Error("expected full long id in message")
	}
}

func Test9(t *testing.T) {
	// FormatNotFound preserves error chain for Unwrap
	result := FormatNotFound("user-123")
	unwrapped := errors.Unwrap(result)
	if unwrapped != ErrNotFound {
		t.Error("expected Unwrap to return ErrNotFound")
	}
}

func Test10(t *testing.T) {
	// FormatNotFound different ids produce different messages
	result1 := FormatNotFound("id1")
	result2 := FormatNotFound("id2")
	if result1.Error() == result2.Error() {
		t.Error("expected different messages for different ids")
	}
}
`,
		hint1: `Use fmt.Errorf() to create a formatted error message.`,
			hint2: `Use %w verb instead of %v to wrap ErrNotFound and preserve error chain.`,
			whyItMatters: `Domain-specific error formatting adds context while preserving error identity for inspection.

**Why Format Errors:**
- **User-Friendly Messages:** "User user-123 not found" vs "not found"
- **Debugging Context:** Know which entity failed without full stack trace
- **Error Identity:** \`errors.Is()\` still works through formatting
- **Logging:** Structured log fields from formatted messages

**Production Pattern:**
\`\`\`go
// Error formatting functions
func FormatNotFound(entityType, id string) error {
    return fmt.Errorf("%s %s: %w", entityType, id, ErrNotFound)
}

func FormatUnauthorized(action, resource string) error {
    return fmt.Errorf("unauthorized %s on %s: %w", action, resource, ErrUnauthorized)
}

func FormatValidation(field, reason string) error {
    return fmt.Errorf("validation failed on %s: %s: %w", field, reason, ErrValidation)
}

// Repository layer
func (r *UserRepo) GetByID(id string) (*User, error) {
    user, err := r.db.QueryRow("SELECT * FROM users WHERE id = $1", id).Scan(&user)
    if err == sql.ErrNoRows {
        return nil, FormatNotFound("user", id)  // Context: which user
    }
    return user, err
}

// Service layer
func (s *Service) DeleteUser(userID, actorID string) error {
    if !s.auth.CanDelete(actorID, userID) {
        return FormatUnauthorized("delete", fmt.Sprintf("user %s", userID))
    }

    err := s.repo.Delete(userID)
    if IsNotFound(err) {
        // Error already has context from repo
        return err
    }
    return err
}

// HTTP handler
func (h *Handler) GetUser(w http.ResponseWriter, r *http.Request) {
    id := r.URL.Query().Get("id")
    user, err := h.service.GetUser(id)

    if err != nil {
        if IsNotFound(err) {
            // Error message: "user user-123: not found"
            http.Error(w, err.Error(), 404)
            return
        }
        http.Error(w, err.Error(), 500)
        return
    }

    json.NewEncoder(w).Encode(user)
}
\`\`\`

**Real-World Benefits:**
- **Client Errors:** Return formatted message directly to clients
- **Logs:** "entity order-456: not found" is more actionable than "not found"
- **Metrics:** Extract entity types from error messages for dashboards
- **Debugging:** Immediately know which resource failed

**%w vs %v:**
- \`%w\` preserves error chain - \`errors.Is()\` and \`errors.As()\` still work
- \`%v\` converts to string - loses error identity
- Always use \`%w\` when wrapping errors

**Standard Library:**
- \`os.PathError\` wraps system errors with file path context
- \`net.OpError\` wraps network errors with operation context
- \`json.UnmarshalTypeError\` wraps errors with type context`,
	order: 6,
	translations: {
		ru: {
			title: 'Форматирование понятных сообщений об ошибках',
			description: `Реализуйте **FormatNotFound()** для создания доменно-специфичных ошибок NotFound.

**Требования:**
1. Создайте функцию \`FormatNotFound(id string) error\`
2. Используйте \`fmt.Errorf()\` с глаголом \`%w\` для обертывания ErrNotFound
3. Включите идентификатор сущности в сообщение
4. Формат: \`"entity <id>: %w"\`

**Пример:**
\`\`\`go
var ErrNotFound = errors.New("not found")

func GetUser(id string) (*User, error) {
    user, err := db.Query(id)
    if err == sql.ErrNoRows {
        return nil, FormatNotFound(id)
    }
    return user, err
}

// Сообщение: "entity user-123: not found"
\`\`\`

**Ограничения:**
- Должна использовать %w (не %v)
- Должна включать параметр id`,
			hint1: `Используйте fmt.Errorf() для создания форматированного сообщения.`,
			hint2: `Используйте глагол %w вместо %v для сохранения цепочки ошибок.`,
			whyItMatters: `Доменно-специфичное форматирование добавляет контекст, сохраняя идентичность ошибки.

**Почему важно:**
- **Понятные сообщения:** "User user-123 not found" вместо "not found"
- **Контекст отладки:** Знаете какая сущность упала
- **Идентичность ошибки:** \`errors.Is()\` все еще работает
- **Логирование:** Структурированные поля из сообщений

**Продакшен паттерн:**
\`\`\`go
// Функции форматирования ошибок
func FormatNotFound(entityType, id string) error {
    return fmt.Errorf("%s %s: %w", entityType, id, ErrNotFound)
}

func FormatUnauthorized(action, resource string) error {
    return fmt.Errorf("unauthorized %s on %s: %w", action, resource, ErrUnauthorized)
}

func FormatValidation(field, reason string) error {
    return fmt.Errorf("validation failed on %s: %s: %w", field, reason, ErrValidation)
}

// Слой репозитория
func (r *UserRepo) GetByID(id string) (*User, error) {
    user, err := r.db.QueryRow("SELECT * FROM users WHERE id = $1", id).Scan(&user)
    if err == sql.ErrNoRows {
        return nil, FormatNotFound("user", id)  // Контекст: какой пользователь
    }
    return user, err
}

// Слой сервиса
func (s *Service) DeleteUser(userID, actorID string) error {
    if !s.auth.CanDelete(actorID, userID) {
        return FormatUnauthorized("delete", fmt.Sprintf("user %s", userID))
    }

    err := s.repo.Delete(userID)
    if IsNotFound(err) {
        // Ошибка уже содержит контекст из репозитория
        return err
    }
    return err
}

// HTTP обработчик
func (h *Handler) GetUser(w http.ResponseWriter, r *http.Request) {
    id := r.URL.Query().Get("id")
    user, err := h.service.GetUser(id)

    if err != nil {
        if IsNotFound(err) {
            // Сообщение об ошибке: "user user-123: not found"
            http.Error(w, err.Error(), 404)
            return
        }
        http.Error(w, err.Error(), 500)
        return
    }

    json.NewEncoder(w).Encode(user)
}
\`\`\`

**Практические преимущества:**
- **Ошибки клиента:** Возврат форматированного сообщения непосредственно клиентам
- **Логи:** "entity order-456: not found" более действенно, чем "not found"
- **Метрики:** Извлечение типов сущностей из сообщений об ошибках для дашбордов
- **Отладка:** Немедленное знание, какой ресурс не удался

**%w vs %v:**
- \`%w\` сохраняет цепочку ошибок - \`errors.Is()\` и \`errors.As()\` продолжают работать
- \`%v\` преобразует в строку - теряет идентичность ошибки
- Всегда используйте \`%w\` при обертывании ошибок

**Стандартная библиотека:**
- \`os.PathError\` оборачивает системные ошибки с контекстом пути к файлу
- \`net.OpError\` оборачивает сетевые ошибки с контекстом операции
- \`json.UnmarshalTypeError\` оборачивает ошибки с контекстом типа`,
			solutionCode: `package errorsx

import (
	"errors"
	"fmt"
)

var ErrNotFound = errors.New("not found")

func FormatNotFound(id string) error {
	return fmt.Errorf("entity %s: %w", id, ErrNotFound)  // %w оборачивает ошибку, сохраняя цепочку
}`
		},
		uz: {
			title: 'Tushunarli xato xabarlarini formatlash',
			description: `Domen-spetsifik NotFound xatolarini yaratish uchun **FormatNotFound()** ni amalga oshiring.

**Talablar:**
1. \`FormatNotFound(id string) error\` funksiyasini yarating
2. ErrNotFound ni o'rash uchun \`%w\` bilan \`fmt.Errorf()\` dan foydalaning
3. Xabar ichiga entity identifikatorini qo'shing
4. Format: \`"entity <id>: %w"\`

**Misol:**
\`\`\`go
var ErrNotFound = errors.New("not found")

func GetUser(id string) (*User, error) {
    user, err := db.Query(id)
    if err == sql.ErrNoRows {
        return nil, FormatNotFound(id)
    }
    return user, err
}

// Xabar: "entity user-123: not found"
\`\`\`

**Cheklovlar:**
- %w (%v emas) dan foydalanishi kerak
- id parametrini qo'shishi kerak`,
			hint1: `Formatlangan xabar yaratish uchun fmt.Errorf() dan foydalaning.`,
			hint2: `Error zanjirini saqlash uchun %v o'rniga %w dan foydalaning.`,
			whyItMatters: `Domen-spetsifik formatlash kontekst qo'shadi, xato identifikatsiyasini saqlaydi.

**Nima uchun muhim:**
- **Tushunarli xabarlar:** "User user-123 not found" vs "not found"
- **Debug konteksti:** Qaysi entity ishlamay qolganini bilasiz
- **Error identifikatsiyasi:** \`errors.Is()\` hali ham ishlaydi
- **Logging:** Xabarlardan strukturali fieldlar

**Ishlab chiqarish patterni:**
\`\`\`go
// Xatolarni formatlash funksiyalari
func FormatNotFound(entityType, id string) error {
    return fmt.Errorf("%s %s: %w", entityType, id, ErrNotFound)
}

func FormatUnauthorized(action, resource string) error {
    return fmt.Errorf("unauthorized %s on %s: %w", action, resource, ErrUnauthorized)
}

func FormatValidation(field, reason string) error {
    return fmt.Errorf("validation failed on %s: %s: %w", field, reason, ErrValidation)
}

// Repository qatlami
func (r *UserRepo) GetByID(id string) (*User, error) {
    user, err := r.db.QueryRow("SELECT * FROM users WHERE id = $1", id).Scan(&user)
    if err == sql.ErrNoRows {
        return nil, FormatNotFound("user", id)  // Kontekst: qaysi foydalanuvchi
    }
    return user, err
}

// Servis qatlami
func (s *Service) DeleteUser(userID, actorID string) error {
    if !s.auth.CanDelete(actorID, userID) {
        return FormatUnauthorized("delete", fmt.Sprintf("user %s", userID))
    }

    err := s.repo.Delete(userID)
    if IsNotFound(err) {
        // Xato allaqachon repository dan kontekstga ega
        return err
    }
    return err
}

// HTTP handler
func (h *Handler) GetUser(w http.ResponseWriter, r *http.Request) {
    id := r.URL.Query().Get("id")
    user, err := h.service.GetUser(id)

    if err != nil {
        if IsNotFound(err) {
            // Xato xabari: "user user-123: not found"
            http.Error(w, err.Error(), 404)
            return
        }
        http.Error(w, err.Error(), 500)
        return
    }

    json.NewEncoder(w).Encode(user)
}
\`\`\`

**Amaliy foydalari:**
- **Mijoz xatolari:** Formatlangan xabarni to'g'ridan-to'g'ri mijozlarga qaytarish
- **Loglar:** "entity order-456: not found" "not found" dan ko'ra ko'proq harakatli
- **Metrikalar:** Dashboard uchun xato xabarlaridan entity turlarini ajratib olish
- **Debugging:** Qaysi resurs ishlamay qolganini darhol bilish

**%w vs %v:**
- \`%w\` xato zanjirini saqlaydi - \`errors.Is()\` va \`errors.As()\` ishlashda davom etadi
- \`%v\` stringga o'zgartiradi - xato identifikatsiyasini yo'qotadi
- Xatolarni o'rashda doim \`%w\` dan foydalaning

**Standart kutubxona:**
- \`os.PathError\` tizim xatolarini fayl yo'li konteksti bilan o'raydi
- \`net.OpError\` tarmoq xatolarini operatsiya konteksti bilan o'raydi
- \`json.UnmarshalTypeError\` xatolarni tur konteksti bilan o'raydi`,
			solutionCode: `package errorsx

import (
	"errors"
	"fmt"
)

var ErrNotFound = errors.New("not found")

func FormatNotFound(id string) error {
	return fmt.Errorf("entity %s: %w", id, ErrNotFound)  // %w xatoni o'raydi, zanjirni saqlaydi
}`
		}
	}
};

export default task;
