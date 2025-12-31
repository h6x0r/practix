import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-http-recover',
	title: 'Panic Recovery Middleware',
	difficulty: 'medium',	tags: ['go', 'http', 'middleware', 'error-handling'],
	estimatedTime: '25m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **Recover** middleware that catches panics in handlers and returns 500 Internal Server Error.

**Requirements:**
1. Create function \`Recover(next http.Handler) http.Handler\`
2. Use defer/recover pattern to catch panics
3. Return 500 response when panic occurs
4. Prevent server crash from handler panics
5. Handle nil next handler

**Example:**
\`\`\`go
handler := Recover(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    panic("something went wrong!")
}))

// Request → 500 Internal Server Error, "internal error"
// Server continues running (no crash)
\`\`\`

**Constraints:**
- Must use defer with recover() to catch panics
- Must return 500 status code for all panics`,
	initialCode: `package httpx

import (
	"net/http"
)

// TODO: Implement Recover middleware
func Recover(next http.Handler) http.Handler {
	// TODO: Implement
}`,
	solutionCode: `package httpx

import (
	"net/http"
)

func Recover(next http.Handler) http.Handler {
	if next == nil {	// Handle nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {	// Defer panic recovery
			if rec := recover(); rec != nil {	// Check if panic occurred
				http.Error(w, "internal error", http.StatusInternalServerError)	// Send 500 response
			}
		}()
		next.ServeHTTP(w, r)	// Execute handler (may panic)
	})
}`,
	testCode: `package httpx

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func Test1RecoverCatchesPanic(t *testing.T) {
	handler := Recover(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		panic("test panic")
	}))
	req := httptest.NewRequest("GET", "/test", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusInternalServerError {
		t.Errorf("expected status 500, got %d", rec.Code)
	}
}

func Test2RecoverNoPanic(t *testing.T) {
	handler := Recover(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("success"))
	}))
	req := httptest.NewRequest("GET", "/test", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", rec.Code)
	}
}

func Test3RecoverNilHandler(t *testing.T) {
	handler := Recover(nil)
	req := httptest.NewRequest("GET", "/test", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", rec.Code)
	}
}

func Test4RecoverStringPanic(t *testing.T) {
	handler := Recover(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		panic("string panic")
	}))
	req := httptest.NewRequest("GET", "/test", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusInternalServerError {
		t.Errorf("expected status 500, got %d", rec.Code)
	}
}

func Test5RecoverIntPanic(t *testing.T) {
	handler := Recover(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		panic(42)
	}))
	req := httptest.NewRequest("GET", "/test", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusInternalServerError {
		t.Errorf("expected status 500, got %d", rec.Code)
	}
}

func Test6RecoverNilPointerPanic(t *testing.T) {
	handler := Recover(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var s *string
		_ = *s
	}))
	req := httptest.NewRequest("GET", "/test", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusInternalServerError {
		t.Errorf("expected status 500, got %d", rec.Code)
	}
}

func Test7RecoverIndexOutOfBounds(t *testing.T) {
	handler := Recover(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		arr := []int{1, 2, 3}
		_ = arr[10]
	}))
	req := httptest.NewRequest("GET", "/test", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusInternalServerError {
		t.Errorf("expected status 500, got %d", rec.Code)
	}
}

func Test8RecoverErrorResponseBody(t *testing.T) {
	handler := Recover(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		panic("test panic")
	}))
	req := httptest.NewRequest("GET", "/test", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	body := rec.Body.String()
	if body == "" {
		t.Error("expected error message in body")
	}
}

func Test9RecoverPartialWrite(t *testing.T) {
	handler := Recover(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("partial"))
		panic("test panic")
	}))
	req := httptest.NewRequest("GET", "/test", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusInternalServerError {
		t.Errorf("expected status 500, got %d", rec.Code)
	}
}

func Test10RecoverMultipleRequests(t *testing.T) {
	handler := Recover(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/panic" {
			panic("test panic")
		}
		w.WriteHeader(http.StatusOK)
	}))
	req1 := httptest.NewRequest("GET", "/panic", nil)
	rec1 := httptest.NewRecorder()
	handler.ServeHTTP(rec1, req1)
	if rec1.Code != http.StatusInternalServerError {
		t.Errorf("expected status 500, got %d", rec1.Code)
	}
	req2 := httptest.NewRequest("GET", "/normal", nil)
	rec2 := httptest.NewRecorder()
	handler.ServeHTTP(rec2, req2)
	if rec2.Code != http.StatusOK {
		t.Errorf("expected status 200 after panic, got %d", rec2.Code)
	}
}`,
			hint1: `Use defer func() with recover() to catch panics. Check if recover() returns non-nil.`,
			hint2: `Call http.Error() with 500 status inside the recover block to send error response.`,
			whyItMatters: `Recover middleware prevents a single handler panic from crashing the entire server, essential for production resilience.

**Why Panic Recovery:**
- **Server Stability:** One buggy handler doesn't crash the entire service
- **Graceful Degradation:** Return 500 instead of killing the server
- **Error Visibility:** Log panics for debugging without downtime
- **User Experience:** Clients get proper error responses, not connection resets

**Production Pattern:**
\`\`\`go
// Enhanced recovery with logging
func RecoverWithLogging(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        defer func() {
            if rec := recover(); rec != nil {
	// Log panic with stack trace
                stack := debug.Stack()
                log.Printf("PANIC: %v\n%s", rec, stack)

	// Log request details
                log.Printf("Request: %s %s from %s", r.Method, r.URL.Path, r.RemoteAddr)

	// Send generic error to client
                http.Error(w, "internal server error", http.StatusInternalServerError)

	// Send alert to monitoring system
                alerting.SendAlert("server_panic", fmt.Sprintf("%v", rec))
            }
        }()

        next.ServeHTTP(w, r)
    })
}

// Recovery with metrics
func RecoverWithMetrics(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        defer func() {
            if rec := recover(); rec != nil {
	// Increment panic counter
                metrics.IncrementCounter("http_panics_total", map[string]string{
                    "path":   r.URL.Path,
                    "method": r.Method,
                })

                log.Printf("PANIC recovered: %v", rec)
                http.Error(w, "internal server error", http.StatusInternalServerError)
            }
        }()

        next.ServeHTTP(w, r)
    })
}

// Recovery with context-aware cleanup
func RecoverWithCleanup(cleanup func(context.Context)) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            defer func() {
                if rec := recover(); rec != nil {
	// Cleanup resources before responding
                    cleanup(r.Context())

                    log.Printf("PANIC: %v", rec)
                    http.Error(w, "internal server error", http.StatusInternalServerError)
                }
            }()

            next.ServeHTTP(w, r)
        })
    }
}

// Common panic sources in handlers
func BuggyHandlers() {
	// Nil pointer dereference
    http.HandleFunc("/nil", func(w http.ResponseWriter, r *http.Request) {
        var user *User
        fmt.Fprintf(w, user.Name) // PANIC: nil pointer dereference
    })

	// Index out of range
    http.HandleFunc("/index", func(w http.ResponseWriter, r *http.Request) {
        items := []string{"a", "b"}
        fmt.Fprintf(w, items[5]) // PANIC: index out of range
    })

	// Type assertion failure
    http.HandleFunc("/assert", func(w http.ResponseWriter, r *http.Request) {
        var x interface{} = "string"
        num := x.(int) // PANIC: type assertion failed
        fmt.Fprintf(w, "%d", num)
    })

	// Explicit panic
    http.HandleFunc("/explicit", func(w http.ResponseWriter, r *http.Request) {
        panic("something went wrong") // PANIC: explicit panic
    })
}
\`\`\`

**Real-World Benefits:**
- **Zero Downtime:** Server continues serving other requests after a panic
- **Error Tracking:** Panics are logged and sent to error tracking systems (Sentry, Rollbar)
- **Debugging:** Stack traces help identify bug locations
- **Client Experience:** Clients get 500 responses instead of connection failures

**Panic Recovery Best Practices:**
- **Top of Chain:** Place Recover at the beginning of middleware chain
- **Log Everything:** Always log panic value and stack trace
- **Alert on Panics:** Send alerts for production panics
- **Fix Bugs:** Panics indicate bugs that should be fixed
- **Generic Errors:** Don't expose panic details to clients (security)

**When NOT to Recover:**
- **Goroutines:** Recover only works in same goroutine (use separate recover for goroutines)
- **defer/recover in goroutines:** Each goroutine needs its own recovery

Without panic recovery, a single nil pointer dereference or array out-of-bounds can crash your entire HTTP server, causing downtime and data loss.`,	order: 8,
	translations: {
		ru: {
			title: 'Восстановление после паники в обработчике',
			description: `Реализуйте middleware **Recover**, который перехватывает панику в handlers и возвращает 500 Internal Server Error.

**Требования:**
1. Создайте функцию \`Recover(next http.Handler) http.Handler\`
2. Используйте паттерн defer/recover для перехвата паники
3. Верните 500 ответ при возникновении паники
4. Предотвратите крах сервера от паники handler
5. Обработайте nil handler

**Пример:**
\`\`\`go
handler := Recover(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    panic("something went wrong!")
}))

// Запрос → 500 Internal Server Error, "internal error"
// Сервер продолжает работу (не крашится)
\`\`\`

**Ограничения:**
- Должен использовать defer с recover() для перехвата паники
- Должен возвращать 500 статус для всех паник`,
			hint1: `Используйте defer func() с recover() для перехвата паники. Проверьте что recover() вернул не nil.`,
			hint2: `Вызовите http.Error() с 500 статусом внутри блока recover для отправки ответа об ошибке.`,
			whyItMatters: `Recover middleware предотвращает крах всего сервера от паники одного handler, критично для production устойчивости.

**Почему Panic Recovery:**
- **Стабильность сервера:** Один багнутый handler не крашит весь сервис
- **Graceful Degradation:** Возврат 500 вместо убийства сервера
- **Видимость ошибок:** Логирование паник для отладки без downtime
- **Пользовательский опыт:** Клиенты получают правильные ответы об ошибках, а не обрыв соединения

**Продакшен паттерн:**
\`\`\`go
// Расширенный recovery с логированием
func RecoverWithLogging(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        defer func() {
            if rec := recover(); rec != nil {
	// Логирование паники со stack trace
                stack := debug.Stack()
                log.Printf("PANIC: %v\\n%s", rec, stack)

	// Логирование деталей запроса
                log.Printf("Request: %s %s from %s", r.Method, r.URL.Path, r.RemoteAddr)

	// Отправка generic ошибки клиенту
                http.Error(w, "internal server error", http.StatusInternalServerError)

	// Отправка алерта в систему мониторинга
                alerting.SendAlert("server_panic", fmt.Sprintf("%v", rec))
            }
        }()

        next.ServeHTTP(w, r)
    })
}

// Recovery с метриками
func RecoverWithMetrics(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        defer func() {
            if rec := recover(); rec != nil {
	// Инкремент счётчика паник
                metrics.IncrementCounter("http_panics_total", map[string]string{
                    "path":   r.URL.Path,
                    "method": r.Method,
                })

                log.Printf("PANIC recovered: %v", rec)
                http.Error(w, "internal server error", http.StatusInternalServerError)
            }
        }()

        next.ServeHTTP(w, r)
    })
}

// Частые источники паник в handlers
func BuggyHandlers() {
	// Разыменование nil указателя
    http.HandleFunc("/nil", func(w http.ResponseWriter, r *http.Request) {
        var user *User
        fmt.Fprintf(w, user.Name) // PANIC: nil pointer dereference
    })

	// Выход за границы массива
    http.HandleFunc("/index", func(w http.ResponseWriter, r *http.Request) {
        items := []string{"a", "b"}
        fmt.Fprintf(w, items[5]) // PANIC: index out of range
    })
}
\`\`\`

**Практические преимущества:**
- **Нулевой downtime:** Сервер продолжает обслуживать другие запросы после паники
- **Отслеживание ошибок:** Паники логируются и отправляются в системы трекинга (Sentry, Rollbar)
- **Отладка:** Stack traces помогают найти место бага
- **Клиентский опыт:** Клиенты получают 500 ответы вместо обрыва соединения

**Best practices Panic Recovery:**
- **В начале цепочки:** Размещайте Recover в начале middleware цепочки
- **Логируйте всё:** Всегда логируйте значение паники и stack trace
- **Алерты на паники:** Отправляйте алерты для production паник
- **Исправляйте баги:** Паники указывают на баги, которые нужно исправить
- **Generic ошибки:** Не раскрывайте детали паник клиентам (безопасность)

**Когда НЕ использовать Recover:**
- **Горутины:** Recover работает только в той же горутине
- **defer/recover в горутинах:** Каждая горутина нуждается в собственном recovery

Без panic recovery одно разыменование nil указателя может крашнуть весь HTTP сервер, вызывая downtime и потерю данных.`,
			solutionCode: `package httpx

import (
	"net/http"
)

func Recover(next http.Handler) http.Handler {
	if next == nil {	// Обработка nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {	// Отложенное восстановление от паники
			if rec := recover(); rec != nil {	// Проверка возникновения паники
				http.Error(w, "internal error", http.StatusInternalServerError)	// Отправка 500 ответа
			}
		}()
		next.ServeHTTP(w, r)	// Выполнение handler (может вызвать панику)
	})
}`
		},
		uz: {
			title: 'Handlerda panic dan tiklanish',
			description: `Handlerlarda panicni ushlab qoluvchi va 500 Internal Server Error qaytaruvchi **Recover** middleware ni amalga oshiring.

**Talablar:**
1. \`Recover(next http.Handler) http.Handler\` funksiyasini yarating
2. Panicni ushlab qolish uchun defer/recover patternidan foydalaning
3. Panic yuz berganda 500 responseni qaytaring
4. Handler panicdan server crashini oldini oling
5. nil handlerni ishlang

**Misol:**
\`\`\`go
handler := Recover(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    panic("something went wrong!")
}))

// Request → 500 Internal Server Error, "internal error"
// Server ishlashda davom etadi (crashlanmaydi)
\`\`\`

**Cheklovlar:**
- Panicni ushlab qolish uchun defer bilan recover() dan foydalanishi kerak
- Barcha paniclar uchun 500 status kodni qaytarishi kerak`,
			hint1: `Panicni ushlab qolish uchun recover() bilan defer func() dan foydalaning. recover() nil emas qaytarishini tekshiring.`,
			hint2: `Xato responseni yuborish uchun recover bloki ichida 500 status bilan http.Error() ni chaqiring.`,
			whyItMatters: `Recover middleware bitta handler panicdan butun serverni crashlanishining oldini oladi, production barqarorligi uchun zarur.

**Nima uchun Panic Recovery:**
- **Server barqarorligi:** Bitta xatoli handler butun servisni crashlantirmaydi
- **Graceful Degradation:** Serverni o'ldirish o'rniga 500 qaytarish
- **Xatolarning ko'rinishi:** Downtime siz debug uchun paniclarni loglash
- **Foydalanuvchi tajribasi:** Mijozlar ulanish uzilishi o'rniga to'g'ri xato javoblarini oladi

**Ishlab chiqarish patterni:**
\`\`\`go
// Loglash bilan kengaytirilgan recovery
func RecoverWithLogging(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        defer func() {
            if rec := recover(); rec != nil {
	// Stack trace bilan panic ni loglash
                stack := debug.Stack()
                log.Printf("PANIC: %v\\n%s", rec, stack)

	// Request tafsilotlarini loglash
                log.Printf("Request: %s %s from %s", r.Method, r.URL.Path, r.RemoteAddr)

	// Mijozga umumiy xatoni yuborish
                http.Error(w, "internal server error", http.StatusInternalServerError)

	// Monitoring sistemasiga alert yuborish
                alerting.SendAlert("server_panic", fmt.Sprintf("%v", rec))
            }
        }()

        next.ServeHTTP(w, r)
    })
}

// Metrikalar bilan Recovery
func RecoverWithMetrics(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        defer func() {
            if rec := recover(); rec != nil {
	// Panic hisoblagichini oshirish
                metrics.IncrementCounter("http_panics_total", map[string]string{
                    "path":   r.URL.Path,
                    "method": r.Method,
                })

                log.Printf("PANIC recovered: %v", rec)
                http.Error(w, "internal server error", http.StatusInternalServerError)
            }
        }()

        next.ServeHTTP(w, r)
    })
}

// Handlerlardagi keng tarqalgan panic manbalari
func BuggyHandlers() {
	// nil pointer dereference
    http.HandleFunc("/nil", func(w http.ResponseWriter, r *http.Request) {
        var user *User
        fmt.Fprintf(w, user.Name) // PANIC: nil pointer dereference
    })

	// Massiv chegarasidan tashqariga chiqish
    http.HandleFunc("/index", func(w http.ResponseWriter, r *http.Request) {
        items := []string{"a", "b"}
        fmt.Fprintf(w, items[5]) // PANIC: index out of range
    })
}
\`\`\`

**Amaliy foydalari:**
- **Nolinchi downtime:** Server panicdan keyin boshqa requestlarni xizmat qilishda davom etadi
- **Xatolarni kuzatish:** Paniclar loglanadi va xatolarni kuzatish tizimlariga yuboriladi (Sentry, Rollbar)
- **Debugging:** Stack tracelar xato joylarini aniqlashga yordam beradi
- **Mijoz tajribasi:** Mijozlar ulanish xatolari o'rniga 500 javoblarini oladi

**Panic Recovery uchun best practices:**
- **Zanjir boshida:** Recover ni middleware zanjiri boshiga joylashtiring
- **Hammasini loglang:** Har doim panic qiymatini va stack traceni loglang
- **Paniclar uchun alertlar:** Production paniclar uchun alertlar yuboring
- **Xatolarni tuzating:** Paniclar tuzatilishi kerak bo'lgan xatolarga ishora qiladi
- **Umumiy xatolar:** Mijozlarga panic tafsilotlarini oshkor qilmang (xavfsizlik)

**Qachon Recover ishlatMASLIK kerak:**
- **Goroutinelar:** Recover faqat bir xil goroutineda ishlaydi
- **Goroutinelarda defer/recover:** Har bir goroutine o'zining recovery kerak

Panic recovery siz bitta nil pointer dereference butun HTTP serverni crashlantirishi mumkin, bu downtime va ma'lumot yo'qotilishiga olib keladi.`,
			solutionCode: `package httpx

import (
	"net/http"
)

func Recover(next http.Handler) http.Handler {
	if next == nil {	// nil next handlerni ishlash
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {	// Panicdan tiklanishni kechiktirish
			if rec := recover(); rec != nil {	// Panic yuz berganini tekshirish
				http.Error(w, "internal error", http.StatusInternalServerError)	// 500 responseni yuborish
			}
		}()
		next.ServeHTTP(w, r)	// Handlerni bajarish (panic chiqarishi mumkin)
	})
}`
		}
	}
};

export default task;
