import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-panic-safe-calls',
	title: 'Safe Function Execution with Panic Recovery',
	difficulty: 'medium',	tags: ['go', 'panic', 'recover', 'error-handling'],
	estimatedTime: '25m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement production-grade panic recovery mechanisms for safe function execution that convert panics into manageable errors.

**Requirements:**
1. **SafeCall**: Execute function with panic recovery, return error
2. **SafeCallValue**: Execute function returning value, recover from panic
3. **RecoverString**: Capture panic message and panic status
4. **Named Returns**: Use named return values for defer recovery

**Safe Execution Pattern:**
\`\`\`go
// Basic panic recovery
func SafeCall(f func()) (err error) {
    defer func() {
        if r := recover(); r != nil {
            err = fmt.Errorf("panic: %v", r)
        }
    }()

    if f != nil {
        f()
    }
    return err
}

// Generic value recovery
func SafeCallValue[T any](f func() T) (T, error) {
    var zero T
    if f == nil {
        return zero, nil
    }

    var result T
    err := SafeCall(func() {
        result = f()
    })

    if err != nil {
        return zero, err
    }
    return result, nil
}

// Detailed panic information
func RecoverString(f func()) (msg string, panicked bool) {
    defer func() {
        if r := recover(); r != nil {
            msg = fmt.Sprint(r)
            panicked = true
        }
    }()

    if f != nil {
        f()
    }
    return msg, panicked
}
\`\`\`

**Example Usage:**
\`\`\`go
// Example 1: Handling third-party library panics
err := SafeCall(func() {
    // External library that might panic
    externalLib.ProcessData(userInput)
})
if err != nil {
    log.Printf("recovered from panic: %v", err)
    // Return error to client instead of crashing
    return fmt.Errorf("processing failed: %w", err)
}

// Example 2: Safe value extraction from risky operations
result, err := SafeCallValue(func() int {
    // Risky operation that might panic
    return riskyMap[key] * 100 / len(data)
})
if err != nil {
    log.Printf("calculation panic: %v", err)
    return defaultValue, nil
}

// Example 3: Detailed panic diagnostics
msg, panicked := RecoverString(func() {
    parseComplexJSON(malformedData)
})
if panicked {
    metrics.IncrementPanicCounter("json_parser")
    log.Printf("JSON parser panic: %s", msg)
    alert.SendToOpsTeam("Parser panic detected", msg)
}
\`\`\`

**Real-World Production Scenario:**
\`\`\`go
// API Handler with panic recovery
type Server struct {
    logger *log.Logger
    metrics *Metrics
}

func (s *Server) HandleRequest(w http.ResponseWriter, r *http.Request) {
    err := SafeCall(func() {
        // Business logic that might panic
        data := s.processRequest(r)
        json.NewEncoder(w).Encode(data)
    })

    if err != nil {
        // Panic occurred - don't crash the server
        s.logger.Printf("request panic: %v", err)
        s.metrics.IncrementPanicCounter()

        // Return 500 to client
        http.Error(w, "Internal Server Error", 500)
        return
    }
}

// Database query with panic recovery
func (s *Server) GetUserSafely(id int) (*User, error) {
    user, err := SafeCallValue(func() *User {
        return s.db.QueryUser(id)  // Might panic on malformed data
    })

    if err != nil {
        s.logger.Printf("db query panic for user %d: %v", id, err)
        return nil, fmt.Errorf("database error: %w", err)
    }

    return user, nil
}

// Plugin system with panic isolation
func (s *Server) ExecutePlugin(name string, plugin Plugin) {
    msg, panicked := RecoverString(func() {
        plugin.Execute()
    })

    if panicked {
        s.logger.Printf("plugin %s panicked: %s", name, msg)
        s.metrics.RecordPluginPanic(name, msg)

        // Disable misbehaving plugin
        s.DisablePlugin(name)

        // Alert operations team
        s.alerting.Send(fmt.Sprintf(
            "Plugin %s crashed and was disabled: %s",
            name, msg,
        ))
    }
}
\`\`\`

**Security Considerations:**
\`\`\`go
// DON'T expose panic details to users
err := SafeCall(userProvidedFunction)
if err != nil {
    // BAD: exposes internal details
    return fmt.Errorf("error: %v", err)

    // GOOD: generic message to user, detailed logs internally
    log.Printf("internal panic: %v", err)
    return errors.New("an error occurred processing your request")
}
\`\`\`

**Constraints:**
- SafeCall must use named return for defer recovery
- SafeCallValue must handle nil functions gracefully
- RecoverString must return both message and panic status
- All functions must prevent panic propagation`,
	initialCode: `package panicrecover

import (
	"context"
	"fmt"
)

// TODO: Implement SafeCall
// Execute f and recover from any panic
// Return error with panic message: fmt.Errorf("panic: %v", r)
// Use named return value 'err' for defer recovery
func SafeCall(f func()) (err error) {
	// TODO: Implement
}

// TODO: Implement SafeCallValue
// Execute f and return its result
// Recover from panic and return zero value + error
// Use SafeCall internally to handle panic recovery
func SafeCallValue[T any](f func() T) (T, error) {
	// TODO: Implement
}

// TODO: Implement RecoverString
// Execute f and capture panic message as string
// Return (message, true) if panic occurred
// Return ("", false) if no panic
// Use fmt.Sprint to convert panic value to string
func RecoverString(f func()) (msg string, panicked bool) {
	// TODO: Implement
}`,
	solutionCode: `package panicrecover

import (
	"context"
	"fmt"
)

func SafeCall(f func()) (err error) {
	defer func() {
		if r := recover(); r != nil {	// catch panic
			err = fmt.Errorf("panic: %v", r)	// convert to error
		}
	}()
	if f != nil {
		f()	// execute function
	}
	return err
}

func SafeCallValue[T any](f func() T) (T, error) {
	var zero T
	if f == nil {
		return zero, nil
	}
	var result T
	err := SafeCall(func() {	// wrap in SafeCall
		result = f()	// capture result
	})
	if err != nil {
		return zero, err	// return zero on panic
	}
	return result, nil	// return result on success
}

func RecoverString(f func()) (msg string, panicked bool) {
	defer func() {
		if r := recover(); r != nil {	// catch panic
			msg = fmt.Sprint(r)	// convert to string
			panicked = true	// mark as panicked
		}
	}()
	if f != nil {
		f()	// execute function
	}
	return msg, panicked	// return results
}`,
			hint1: `In SafeCall: use defer with recover() to catch panics. Check if r != nil, then set err = fmt.Errorf("panic: %v", r). Use named return value.`,
			hint2: `In SafeCallValue: declare var result T outside SafeCall. Wrap f() call in SafeCall lambda that assigns to result. Return zero value if err != nil.`,
			testCode: `package panicrecover

import (
	"strings"
	"testing"
)

func Test1(t *testing.T) {
	// Test SafeCall with nil function
	err := SafeCall(nil)
	if err != nil {
		t.Errorf("SafeCall(nil) = %v, want nil", err)
	}
}

func Test2(t *testing.T) {
	// Test SafeCall with normal function
	called := false
	err := SafeCall(func() {
		called = true
	})
	if err != nil {
		t.Errorf("SafeCall(normal) = %v, want nil", err)
	}
	if !called {
		t.Error("Function was not called")
	}
}

func Test3(t *testing.T) {
	// Test SafeCall with panicking function
	err := SafeCall(func() {
		panic("test panic")
	})
	if err == nil {
		t.Error("SafeCall(panic) = nil, want error")
	}
	if !strings.Contains(err.Error(), "panic:") {
		t.Errorf("error should contain 'panic:', got %v", err)
	}
}

func Test4(t *testing.T) {
	// Test SafeCallValue with nil function
	result, err := SafeCallValue[int](nil)
	if err != nil {
		t.Errorf("SafeCallValue(nil) err = %v, want nil", err)
	}
	if result != 0 {
		t.Errorf("SafeCallValue(nil) = %v, want 0", result)
	}
}

func Test5(t *testing.T) {
	// Test SafeCallValue with normal function
	result, err := SafeCallValue(func() int {
		return 42
	})
	if err != nil {
		t.Errorf("SafeCallValue(42) err = %v, want nil", err)
	}
	if result != 42 {
		t.Errorf("SafeCallValue(42) = %v, want 42", result)
	}
}

func Test6(t *testing.T) {
	// Test SafeCallValue with panicking function
	result, err := SafeCallValue(func() string {
		panic("test panic")
	})
	if err == nil {
		t.Error("SafeCallValue(panic) err = nil, want error")
	}
	if result != "" {
		t.Errorf("SafeCallValue(panic) = %q, want empty string", result)
	}
}

func Test7(t *testing.T) {
	// Test RecoverString with nil function
	msg, panicked := RecoverString(nil)
	if panicked {
		t.Error("RecoverString(nil) panicked = true, want false")
	}
	if msg != "" {
		t.Errorf("RecoverString(nil) msg = %q, want empty", msg)
	}
}

func Test8(t *testing.T) {
	// Test RecoverString with normal function
	msg, panicked := RecoverString(func() {
		_ = 1 + 1
	})
	if panicked {
		t.Error("RecoverString(normal) panicked = true, want false")
	}
	if msg != "" {
		t.Errorf("RecoverString(normal) msg = %q, want empty", msg)
	}
}

func Test9(t *testing.T) {
	// Test RecoverString with panicking function
	msg, panicked := RecoverString(func() {
		panic("test error message")
	})
	if !panicked {
		t.Error("RecoverString(panic) panicked = false, want true")
	}
	if msg != "test error message" {
		t.Errorf("RecoverString(panic) msg = %q, want %q", msg, "test error message")
	}
}

func Test10(t *testing.T) {
	// Test SafeCallValue with struct type
	type Data struct {
		Value int
	}
	result, err := SafeCallValue(func() Data {
		return Data{Value: 100}
	})
	if err != nil {
		t.Errorf("SafeCallValue(struct) err = %v, want nil", err)
	}
	if result.Value != 100 {
		t.Errorf("SafeCallValue(struct).Value = %v, want 100", result.Value)
	}
}`,
			whyItMatters: `Panic recovery is critical for building resilient production systems that can survive unexpected failures without crashing.

**Why This Matters:**

**1. Production Incident: E-commerce Platform Outage**

A major e-commerce site had a critical incident:
- Third-party payment library panicked on malformed webhook data
- Panic crashed the entire payment processing service
- 15 minutes of downtime during Black Friday
- $500K in lost sales
- 10,000+ failed transactions

**Root Cause:**
\`\`\`go
// Before: No panic recovery
func ProcessWebhook(data []byte) {
    payment := externalLib.Parse(data)  // Panicked on unexpected format
    processPayment(payment)
}
// Result: One bad webhook crashed entire service
\`\`\`

**Solution with SafeCall:**
\`\`\`go
// After: Panic recovery in place
func ProcessWebhook(data []byte) error {
    err := SafeCall(func() {
        payment := externalLib.Parse(data)
        processPayment(payment)
    })

    if err != nil {
        log.Printf("webhook panic: %v", err)
        metrics.IncrementWebhookErrors()
        return fmt.Errorf("webhook processing failed")
    }
    return nil
}
// Result: Bad webhook logged, service stays up, other webhooks processed
\`\`\`

**Impact After Fix:**
- Zero payment service crashes in 6 months
- Bad webhooks logged and alerted, not crashing system
- 99.99% uptime maintained

**2. Real-World Use Case: Plugin System**

SaaS platform with customer-provided plugins:
- 100+ plugins from different developers
- Any plugin panic would crash entire application
- Impossible to validate all plugin code

\`\`\`go
// Before: Risky plugin execution
func (s *Server) RunPlugin(p Plugin) {
    p.Execute()  // If plugin panics, entire app crashes
}

// After: Safe plugin execution
func (s *Server) RunPlugin(p Plugin) {
    msg, panicked := RecoverString(func() {
        p.Execute()
    })

    if panicked {
        log.Printf("plugin %s crashed: %s", p.Name, msg)
        s.DisablePlugin(p.Name)  // Quarantine bad plugin
        s.NotifyPluginAuthor(p.Name, msg)

        // App continues running for all other users
    }
}
\`\`\`

**Results:**
- Platform uptime: 95% → 99.9%
- Customer complaints: 50/month → 2/month
- Average plugin crash isolated in 100ms vs 5min full outage

**3. Database Query Safety**

Financial services company processing millions of transactions:

\`\`\`go
// Risky: unmarshaling database JSON
func GetAccountBalance(id string) (float64, error) {
    // Database might return malformed JSON
    // json.Unmarshal can panic on certain inputs
    balance, err := SafeCallValue(func() float64 {
        var data AccountData
        json.Unmarshal(dbResult, &data)
        return data.Balance
    })

    if err != nil {
        log.Printf("db unmarshal panic for account %s: %v", id, err)
        return 0, fmt.Errorf("account data corrupted")
    }
    return balance, nil
}
\`\`\`

**Why SafeCallValue is Critical:**
- Without it: panic crashes entire transaction processor
- With it: corrupted account returns error, transaction rolls back safely
- Other accounts continue processing normally

**4. Production Metrics from Real Company**

API Gateway service handling 100M requests/day:

**Before Panic Recovery:**
- 5-10 full service crashes per week
- Average crash recovery time: 2 minutes
- Total downtime: 10-20 minutes/week
- Lost requests during restart: 100K-200K

**After Implementing SafeCall Pattern:**
- Service crashes: 0 per month
- Panics caught and logged: 50-100 per day
- Downtime: 0 minutes
- Lost requests: 0

**The Bottom Line:**
\`\`\`
Cost of NOT using panic recovery:
- Downtime: $50K-$200K per incident
- Customer trust: Irreplaceable
- Engineering time investigating crashes: 20-40 hours/month

Cost of implementing SafeCall:
- Development time: 2-4 hours
- Runtime overhead: Negligible (<1% CPU)
- Maintenance: Zero (set and forget)

ROI: Infinite (prevents catastrophic failures)
\`\`\`

**Best Practice in Production:**
Every external call, plugin execution, or risky operation should use panic recovery:
- API handlers: Always use SafeCall
- Database operations: Use SafeCallValue for results
- Plugin systems: Use RecoverString for diagnostics
- Worker pools: Recover in each worker goroutine`,	order: 0,
	translations: {
		ru: {
			title: 'Безопасные вызовы функций',
			solutionCode: `package panicrecover

import (
	"context"
	"fmt"
)

func SafeCall(f func()) (err error) {
	defer func() {
		if r := recover(); r != nil {	// перехватываем панику
			err = fmt.Errorf("panic: %v", r)	// конвертируем в ошибку
		}
	}()
	if f != nil {
		f()	// выполняем функцию
	}
	return err
}

func SafeCallValue[T any](f func() T) (T, error) {
	var zero T
	if f == nil {
		return zero, nil
	}
	var result T
	err := SafeCall(func() {	// оборачиваем в SafeCall
		result = f()	// захватываем результат
	})
	if err != nil {
		return zero, err	// возвращаем ноль при панике
	}
	return result, nil	// возвращаем результат при успехе
}

func RecoverString(f func()) (msg string, panicked bool) {
	defer func() {
		if r := recover(); r != nil {	// перехватываем панику
			msg = fmt.Sprint(r)	// конвертируем в строку
			panicked = true	// помечаем как паника
		}
	}()
	if f != nil {
		f()	// выполняем функцию
	}
	return msg, panicked	// возвращаем результаты
}`,
			description: `Реализуйте production-grade механизмы восстановления от паники для безопасного выполнения функций, которые конвертируют панику в управляемые ошибки.

**Требования:**
1. **SafeCall**: Выполнение функции с восстановлением от паники, возврат ошибки
2. **SafeCallValue**: Выполнение функции с возвратом значения, восстановление от паники
3. **RecoverString**: Захват сообщения паники и статуса паники
4. **Named Returns**: Использование именованных возвращаемых значений для defer recovery

**Паттерн безопасного выполнения:**
\`\`\`go
// Базовое восстановление от паники
func SafeCall(f func()) (err error) {
    defer func() {
        if r := recover(); r != nil {
            err = fmt.Errorf("panic: %v", r)
        }
    }()

    if f != nil {
        f()
    }
    return err
}

// Обобщенное восстановление значений
func SafeCallValue[T any](f func() T) (T, error) {
    var zero T
    if f == nil {
        return zero, nil
    }

    var result T
    err := SafeCall(func() {
        result = f()
    })

    if err != nil {
        return zero, err
    }
    return result, nil
}

// Детальная информация о панике
func RecoverString(f func()) (msg string, panicked bool) {
    defer func() {
        if r := recover(); r != nil {
            msg = fmt.Sprint(r)
            panicked = true
        }
    }()

    if f != nil {
        f()
    }
    return msg, panicked
}
\`\`\`

**Примеры использования:**
\`\`\`go
// Пример 1: Обработка паник сторонних библиотек
err := SafeCall(func() {
    // Внешняя библиотека, которая может паниковать
    externalLib.ProcessData(userInput)
})
if err != nil {
    log.Printf("восстановлено от паники: %v", err)
    // Возвращаем ошибку клиенту вместо краша
    return fmt.Errorf("обработка не удалась: %w", err)
}

// Пример 2: Безопасное извлечение значения из рискованных операций
result, err := SafeCallValue(func() int {
    // Рискованная операция, которая может паниковать
    return riskyMap[key] * 100 / len(data)
})
if err != nil {
    log.Printf("паника при вычислении: %v", err)
    return defaultValue, nil
}

// Пример 3: Детальная диагностика паники
msg, panicked := RecoverString(func() {
    parseComplexJSON(malformedData)
})
if panicked {
    metrics.IncrementPanicCounter("json_parser")
    log.Printf("паника JSON парсера: %s", msg)
    alert.SendToOpsTeam("Обнаружена паника парсера", msg)
}
\`\`\`

**Реальный Production сценарий:**
\`\`\`go
// API обработчик с восстановлением от паники
type Server struct {
    logger *log.Logger
    metrics *Metrics
}

func (s *Server) HandleRequest(w http.ResponseWriter, r *http.Request) {
    err := SafeCall(func() {
        // Бизнес-логика, которая может паниковать
        data := s.processRequest(r)
        json.NewEncoder(w).Encode(data)
    })

    if err != nil {
        // Паника произошла - не роняем сервер
        s.logger.Printf("паника запроса: %v", err)
        s.metrics.IncrementPanicCounter()

        // Возвращаем 500 клиенту
        http.Error(w, "Internal Server Error", 500)
        return
    }
}

// Запрос к БД с восстановлением от паники
func (s *Server) GetUserSafely(id int) (*User, error) {
    user, err := SafeCallValue(func() *User {
        return s.db.QueryUser(id)  // Может паниковать на некорректных данных
    })

    if err != nil {
        s.logger.Printf("паника запроса к БД для пользователя %d: %v", id, err)
        return nil, fmt.Errorf("ошибка базы данных: %w", err)
    }

    return user, nil
}

// Система плагинов с изоляцией паники
func (s *Server) ExecutePlugin(name string, plugin Plugin) {
    msg, panicked := RecoverString(func() {
        plugin.Execute()
    })

    if panicked {
        s.logger.Printf("плагин %s запаниковал: %s", name, msg)
        s.metrics.RecordPluginPanic(name, msg)

        // Отключаем плохо работающий плагин
        s.DisablePlugin(name)

        // Оповещаем команду operations
        s.alerting.Send(fmt.Sprintf(
            "Плагин %s упал и был отключен: %s",
            name, msg,
        ))
    }
}
\`\`\`

**Соображения безопасности:**
\`\`\`go
// НЕ раскрывайте детали паники пользователям
err := SafeCall(userProvidedFunction)
if err != nil {
    // ПЛОХО: раскрывает внутренние детали
    return fmt.Errorf("ошибка: %v", err)

    // ХОРОШО: общее сообщение пользователю, детальные логи внутри
    log.Printf("внутренняя паника: %v", err)
    return errors.New("произошла ошибка при обработке вашего запроса")
}
\`\`\`

**Ограничения:**
- SafeCall должен использовать именованный возврат для defer recovery
- SafeCallValue должен корректно обрабатывать nil функции
- RecoverString должен возвращать и сообщение, и статус паники
- Все функции должны предотвращать распространение паники`,
			hint1: `В SafeCall: используйте defer с recover() для перехвата паник. Проверьте if r != nil, затем установите err = fmt.Errorf("panic: %v", r). Используйте именованное возвращаемое значение.`,
			hint2: `В SafeCallValue: объявите var result T вне SafeCall. Оберните вызов f() в SafeCall лямбду, которая присваивает result. Верните нулевое значение если err != nil.`,
			whyItMatters: `Восстановление от паники критически важно для построения устойчивых production систем, которые могут пережить неожиданные сбои без падения.

**Почему это важно:**

**1. Production инцидент: Авария E-commerce платформы**

Крупный e-commerce сайт пережил критический инцидент:
- Сторонняя библиотека платежей запаниковала на некорректных данных webhook
- Паника обрушила весь сервис обработки платежей
- 15 минут простоя во время Black Friday
- $500K убытков
- 10,000+ неудачных транзакций

**Корневая причина:**
\`\`\`go
// До: Нет восстановления от паники
func ProcessWebhook(data []byte) {
    payment := externalLib.Parse(data)  // Паника на неожиданном формате
    processPayment(payment)
}
// Результат: Один плохой webhook обрушил весь сервис
\`\`\`

**Решение с SafeCall:**
\`\`\`go
// После: Восстановление от паники внедрено
func ProcessWebhook(data []byte) error {
    err := SafeCall(func() {
        payment := externalLib.Parse(data)
        processPayment(payment)
    })

    if err != nil {
        log.Printf("webhook паника: %v", err)
        metrics.IncrementWebhookErrors()
        return fmt.Errorf("обработка webhook не удалась")
    }
    return nil
}
// Результат: Плохой webhook залогирован, сервис работает, другие webhooks обрабатываются
\`\`\`

**Результат после исправления:**
- Ноль аварий сервиса платежей за 6 месяцев
- Плохие webhooks логируются и оповещаются, не роняя систему
- Поддерживается 99.99% uptime

**2. Реальный пример: Система плагинов**

SaaS платформа с пользовательскими плагинами:
- 100+ плагинов от разных разработчиков
- Любая паника в плагине роняла всё приложение
- Невозможно проверить весь код плагинов

\`\`\`go
// До: Рискованное выполнение плагинов
func (s *Server) RunPlugin(p Plugin) {
    p.Execute()  // Если плагин паникует, всё приложение рушится
}

// После: Безопасное выполнение плагинов
func (s *Server) RunPlugin(p Plugin) {
    msg, panicked := RecoverString(func() {
        p.Execute()
    })

    if panicked {
        log.Printf("плагин %s упал: %s", p.Name, msg)
        s.DisablePlugin(p.Name)  // Карантин для плохого плагина
        s.NotifyPluginAuthor(p.Name, msg)

        // Приложение продолжает работать для всех остальных пользователей
    }
}
\`\`\`

**Результаты:**
- Uptime платформы: 95% → 99.9%
- Жалобы клиентов: 50/месяц → 2/месяц
- Средняя изоляция краша плагина за 100мс против 5мин полного простоя

**3. Безопасность запросов к базе данных**

Компания финансовых услуг обрабатывает миллионы транзакций:

\`\`\`go
// Рискованно: десериализация JSON из базы данных
func GetAccountBalance(id string) (float64, error) {
    // База данных может вернуть некорректный JSON
    // json.Unmarshal может паниковать на определенных входных данных
    balance, err := SafeCallValue(func() float64 {
        var data AccountData
        json.Unmarshal(dbResult, &data)
        return data.Balance
    })

    if err != nil {
        log.Printf("паника десериализации БД для аккаунта %s: %v", id, err)
        return 0, fmt.Errorf("данные аккаунта повреждены")
    }
    return balance, nil
}
\`\`\`

**Почему SafeCallValue критичен:**
- Без него: паника роняет весь процессор транзакций
- С ним: поврежденный аккаунт возвращает ошибку, транзакция откатывается безопасно
- Другие аккаунты продолжают обрабатываться нормально

**4. Production метрики из реальной компании**

API Gateway сервис обрабатывает 100M запросов/день:

**До восстановления от паники:**
- 5-10 полных аварий сервиса в неделю
- Среднее время восстановления от краша: 2 минуты
- Общий простой: 10-20 минут/неделю
- Потерянные запросы во время перезапуска: 100K-200K

**После внедрения паттерна SafeCall:**
- Аварии сервиса: 0 в месяц
- Перехваченные и залогированные паники: 50-100 в день
- Простой: 0 минут
- Потерянные запросы: 0

**Итог:**
\`\`\`
Цена НЕ использования восстановления от паники:
- Простой: $50K-$200K за инцидент
- Доверие клиентов: Невосполнимо
- Время инженеров на расследование крашей: 20-40 часов/месяц

Цена внедрения SafeCall:
- Время разработки: 2-4 часа
- Накладные расходы во время выполнения: Незначительны (<1% CPU)
- Поддержка: Ноль (настроил и забыл)

ROI: Бесконечный (предотвращает катастрофические сбои)
\`\`\`

**Лучшие практики в Production:**
Каждый внешний вызов, выполнение плагина или рискованная операция должны использовать восстановление от паники:
- API обработчики: Всегда используйте SafeCall
- Операции с базой данных: Используйте SafeCallValue для результатов
- Системы плагинов: Используйте RecoverString для диагностики
- Пулы воркеров: Восстанавливайтесь в каждой горутине воркера`
		},
		uz: {
			title: `Xavfsiz funksiya chaqiruvlari`,
			solutionCode: `package panicrecover

import (
	"context"
	"fmt"
)

func SafeCall(f func()) (err error) {
	defer func() {
		if r := recover(); r != nil {	// panikni ushlaymiz
			err = fmt.Errorf("panic: %v", r)	// xatoga aylantiramiz
		}
	}()
	if f != nil {
		f()	// funksiyani bajaramiz
	}
	return err
}

func SafeCallValue[T any](f func() T) (T, error) {
	var zero T
	if f == nil {
		return zero, nil
	}
	var result T
	err := SafeCall(func() {	// SafeCall ga o'raymiz
		result = f()	// natijani ushlaymiz
	})
	if err != nil {
		return zero, err	// panikda nol qaytaramiz
	}
	return result, nil	// muvaffaqiyatda natija qaytaramiz
}

func RecoverString(f func()) (msg string, panicked bool) {
	defer func() {
		if r := recover(); r != nil {	// panikni ushlaymiz
			msg = fmt.Sprint(r)	// satrga aylantiramiz
			panicked = true	// panik deb belgilaymiz
		}
	}()
	if f != nil {
		f()	// funksiyani bajaramiz
	}
	return msg, panicked	// natijalarni qaytaramiz
}`,
			description: `Paniklari boshqariladigan xatolarga aylantiradigan xavfsiz funksiya bajarish uchun production-grade panikdan tiklash mexanizmlarini amalga oshiring.

**Talablar:**
1. **SafeCall**: Funksiyani panikdan tiklash bilan bajaring, xato qaytaring
2. **SafeCallValue**: Qiymat qaytaruvchi funksiyani bajaring, panikdan tiklanish
3. **RecoverString**: Panik xabarini va panik holatini ushlang
4. **Named Returns**: defer recovery uchun nomlangan qaytish qiymatlaridan foydalaning

**Xavfsiz bajarish patterni:**
\`\`\`go
// Asosiy panikdan tiklash
func SafeCall(f func()) (err error) {
    defer func() {
        if r := recover(); r != nil {
            err = fmt.Errorf("panic: %v", r)
        }
    }()

    if f != nil {
        f()
    }
    return err
}

// Umumiy qiymat tiklanishi
func SafeCallValue[T any](f func() T) (T, error) {
    var zero T
    if f == nil {
        return zero, nil
    }

    var result T
    err := SafeCall(func() {
        result = f()
    })

    if err != nil {
        return zero, err
    }
    return result, nil
}

// Batafsil panik ma'lumoti
func RecoverString(f func()) (msg string, panicked bool) {
    defer func() {
        if r := recover(); r != nil {
            msg = fmt.Sprint(r)
            panicked = true
        }
    }()

    if f != nil {
        f()
    }
    return msg, panicked
}
\`\`\`

**Foydalanish misollari:**
\`\`\`go
// Misol 1: Uchinchi tomon kutubxona paniklarini qayta ishlash
err := SafeCall(func() {
    // Panik qilishi mumkin bo'lgan tashqi kutubxona
    externalLib.ProcessData(userInput)
})
if err != nil {
    log.Printf("panikdan tiklandi: %v", err)
    // Crash o'rniga mijozga xato qaytaring
    return fmt.Errorf("qayta ishlash muvaffaqiyatsiz: %w", err)
}

// Misol 2: Xavfli operatsiyalardan qiymatni xavfsiz olish
result, err := SafeCallValue(func() int {
    // Panik qilishi mumkin bo'lgan xavfli operatsiya
    return riskyMap[key] * 100 / len(data)
})
if err != nil {
    log.Printf("hisoblashda panik: %v", err)
    return defaultValue, nil
}

// Misol 3: Batafsil panik diagnostikasi
msg, panicked := RecoverString(func() {
    parseComplexJSON(malformedData)
})
if panicked {
    metrics.IncrementPanicCounter("json_parser")
    log.Printf("JSON parser panik: %s", msg)
    alert.SendToOpsTeam("Parser panik aniqlandi", msg)
}
\`\`\`

**Haqiqiy Production stsenariysi:**
\`\`\`go
// Panikdan tiklanish bilan API handler
type Server struct {
    logger *log.Logger
    metrics *Metrics
}

func (s *Server) HandleRequest(w http.ResponseWriter, r *http.Request) {
    err := SafeCall(func() {
        // Panik qilishi mumkin bo'lgan biznes logika
        data := s.processRequest(r)
        json.NewEncoder(w).Encode(data)
    })

    if err != nil {
        // Panik yuz berdi - serverni qulatmang
        s.logger.Printf("so'rov panik: %v", err)
        s.metrics.IncrementPanicCounter()

        // Mijozga 500 qaytaring
        http.Error(w, "Internal Server Error", 500)
        return
    }
}

// Panikdan tiklanish bilan ma'lumotlar bazasi so'rovi
func (s *Server) GetUserSafely(id int) (*User, error) {
    user, err := SafeCallValue(func() *User {
        return s.db.QueryUser(id)  // Noto'g'ri ma'lumotlarda panik qilishi mumkin
    })

    if err != nil {
        s.logger.Printf("foydalanuvchi %d uchun db so'rovi panik: %v", id, err)
        return nil, fmt.Errorf("ma'lumotlar bazasi xatosi: %w", err)
    }

    return user, nil
}

// Panik izolyatsiyasi bilan plugin tizimi
func (s *Server) ExecutePlugin(name string, plugin Plugin) {
    msg, panicked := RecoverString(func() {
        plugin.Execute()
    })

    if panicked {
        s.logger.Printf("plugin %s panik qildi: %s", name, msg)
        s.metrics.RecordPluginPanic(name, msg)

        // Noto'g'ri ishlaydigan pluginni o'chirish
        s.DisablePlugin(name)

        // Operations jamoasiga xabar berish
        s.alerting.Send(fmt.Sprintf(
            "Plugin %s qulab tushdi va o'chirildi: %s",
            name, msg,
        ))
    }
}
\`\`\`

**Xavfsizlik mulohazalari:**
\`\`\`go
// Foydalanuvchilarga panik tafsilotlarini ko'rsatMANG
err := SafeCall(userProvidedFunction)
if err != nil {
    // YOMON: ichki tafsilotlarni ochib beradi
    return fmt.Errorf("xato: %v", err)

    // YAXSHI: foydalanuvchiga umumiy xabar, ichkarida batafsil loglar
    log.Printf("ichki panik: %v", err)
    return errors.New("so'rovingizni qayta ishlashda xato yuz berdi")
}
\`\`\`

**Cheklovlar:**
- SafeCall defer recovery uchun nomlangan qaytishdan foydalanishi kerak
- SafeCallValue nil funksiyalarni to'g'ri qayta ishlashi kerak
- RecoverString xabar va panik holatini qaytarishi kerak
- Barcha funksiyalar panik tarqalishini oldini olishi kerak`,
			hint1: `SafeCall da: paniklarni ushlash uchun recover() bilan defer ishlating. r != nil tekshiring, keyin err = fmt.Errorf("panic: %v", r) o'rnating. Nomlangan qaytish qiymatidan foydalaning.`,
			hint2: `SafeCallValue da: SafeCall tashqarisida var result T e'lon qiling. f() chaqiruvini result ga tayinlaydigan SafeCall lambda ga o'rang. err != nil bo'lsa nol qiymat qaytaring.`,
			whyItMatters: `Panikdan tiklash kutilmagan nosozliklardan so'ng qulab tushmasdan omon qoladigan barqaror production tizimlar qurish uchun muhim.

**Nima uchun bu muhim:**

**1. Production Incident: E-commerce Platformasi Buzilishi**

Yirik e-commerce sayti jiddiy incidentni boshdan kechirdi:
- Uchinchi tomon to'lov kutubxonasi noto'g'ri webhook ma'lumotlarida panik qildi
- Panik butun to'lovlarni qayta ishlash xizmatini qulab tushirdi
- Black Friday paytida 15 daqiqa downtime
- $500K zarar
- 10,000+ muvaffaqiyatsiz tranzaksiya

**Asosiy sabab:**
\`\`\`go
// Oldin: Panikdan tiklanish yo'q
func ProcessWebhook(data []byte) {
    payment := externalLib.Parse(data)  // Kutilmagan formatda panik
    processPayment(payment)
}
// Natija: Bitta yomon webhook butun xizmatni qulab tushirdi
\`\`\`

**SafeCall bilan yechim:**
\`\`\`go
// Keyin: Panikdan tiklanish joriy qilindi
func ProcessWebhook(data []byte) error {
    err := SafeCall(func() {
        payment := externalLib.Parse(data)
        processPayment(payment)
    })

    if err != nil {
        log.Printf("webhook panik: %v", err)
        metrics.IncrementWebhookErrors()
        return fmt.Errorf("webhook qayta ishlash muvaffaqiyatsiz")
    }
    return nil
}
// Natija: Yomon webhook loglangan, xizmat ishlaydi, boshqa webhooklar qayta ishlanadi
\`\`\`

**Tuzatishdan keyingi ta'sir:**
- 6 oyda nol to'lov xizmati avariyasi
- Yomon webhooklar loglanadi va xabar beriladi, tizimni qulatmaydi
- 99.99% uptime saqlanmoqda

**2. Haqiqiy misol: Plugin tizimi**

Foydalanuvchi tomonidan taqdim etilgan pluginlar bilan SaaS platformasi:
- Turli ishlab chiquvchilardan 100+ plugin
- Plugindagi har qanday panik butun ilovani qulab tushirardi
- Barcha plugin kodini tekshirish mumkin emas

\`\`\`go
// Oldin: Xavfli plugin bajarish
func (s *Server) RunPlugin(p Plugin) {
    p.Execute()  // Agar plugin panik qilsa, butun ilova qulab tushadi
}

// Keyin: Xavfsiz plugin bajarish
func (s *Server) RunPlugin(p Plugin) {
    msg, panicked := RecoverString(func() {
        p.Execute()
    })

    if panicked {
        log.Printf("plugin %s qulab tushdi: %s", p.Name, msg)
        s.DisablePlugin(p.Name)  // Yomon plugin uchun karantin
        s.NotifyPluginAuthor(p.Name, msg)

        // Ilova boshqa barcha foydalanuvchilar uchun ishlashda davom etadi
    }
}
\`\`\`

**Natijalar:**
- Platformaning uptime: 95% → 99.9%
- Mijoz shikoyatlari: 50/oy → 2/oy
- O'rtacha plugin crash izolyatsiyasi 100ms vs 5min to'liq downtime

**3. Ma'lumotlar bazasi so'rovlari xavfsizligi**

Moliyaviy xizmatlar kompaniyasi millionlab tranzaksiyalarni qayta ishlaydi:

\`\`\`go
// Xavfli: ma'lumotlar bazasidan JSON deserializatsiya
func GetAccountBalance(id string) (float64, error) {
    // Ma'lumotlar bazasi noto'g'ri JSON qaytarishi mumkin
    // json.Unmarshal ba'zi kirishlarda panik qilishi mumkin
    balance, err := SafeCallValue(func() float64 {
        var data AccountData
        json.Unmarshal(dbResult, &data)
        return data.Balance
    })

    if err != nil {
        log.Printf("hisob %s uchun db unmarshaling panik: %v", id, err)
        return 0, fmt.Errorf("hisob ma'lumotlari buzilgan")
    }
    return balance, nil
}
\`\`\`

**Nima uchun SafeCallValue muhim:**
- Usiz: panik butun tranzaksiya protsessorini qulatadi
- U bilan: buzilgan hisob xato qaytaradi, tranzaksiya xavfsiz rollback qilinadi
- Boshqa hisoblar odatdagidek qayta ishlashda davom etadi

**4. Haqiqiy kompaniyadan Production metrikalari**

100M so'rov/kunlik API Gateway xizmati:

**Panikdan tiklanishdan oldin:**
- Haftasiga 5-10 to'liq xizmat avariyasi
- O'rtacha crash dan tiklanish vaqti: 2 daqiqa
- Jami downtime: 10-20 daqiqa/hafta
- Qayta ishga tushirish paytida yo'qotilgan so'rovlar: 100K-200K

**SafeCall patternini joriy qilgandan keyin:**
- Xizmat avariyalari: oyiga 0
- Ushlangan va loglangan paniklar: kuniga 50-100
- Downtime: 0 daqiqa
- Yo'qotilgan so'rovlar: 0

**Xulosa:**
\`\`\`
Panikdan tiklanishdan FOYDALANMASLIK narxi:
- Downtime: Incident uchun $50K-$200K
- Mijozlar ishonchi: Tiklanmas
- Crash tekshiruviga muhandis vaqti: 20-40 soat/oy

SafeCall joriy qilish narxi:
- Ishlab chiqish vaqti: 2-4 soat
- Runtime overhead: Ahamiyatsiz (<1% CPU)
- Texnik xizmat: Nol (sozlash va unutish)

ROI: Cheksiz (halokatli nosozliklarni oldini oladi)
\`\`\`

**Production da eng yaxshi amaliyotlar:**
Har bir tashqi chaqiruv, plugin bajarish yoki xavfli operatsiya panikdan tiklanishdan foydalanishi kerak:
- API handlerlar: Har doim SafeCall dan foydalaning
- Ma'lumotlar bazasi operatsiyalari: Natijalar uchun SafeCallValue dan foydalaning
- Plugin tizimlari: Diagnostika uchun RecoverString dan foydalaning
- Worker poollar: Har bir worker goroutine da tiklanish`
		}
	}
};

export default task;
