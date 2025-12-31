import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-workerpool-run-sequential',
	title: 'Run Sequential',
	difficulty: 'easy',	tags: ['go', 'concurrency', 'worker-pool', 'sequential'],
	estimatedTime: '15m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **RunSequential** that executes jobs sequentially, one after another.

**Requirements:**
1. Create function \`RunSequential(ctx context.Context, jobs []Job) error\`
2. Handle nil context (return nil immediately)
3. Check context cancellation before each job
4. Skip nil jobs (continue to next)
5. Execute jobs one by one
6. Return first error encountered
7. Return context error if cancelled

**Type Definition:**
\`\`\`go
type Job func(context.Context) error
\`\`\`

**Example:**
\`\`\`go
jobs := []Job{
    func(ctx context.Context) error {
        fmt.Println("Job 1")
        return nil
    },
    func(ctx context.Context) error {
        fmt.Println("Job 2")
        return nil
    },
    func(ctx context.Context) error {
        return errors.New("job 3 failed")
    },
    func(ctx context.Context) error {
        fmt.Println("Job 4") // won't execute
        return nil
    },
}

err := RunSequential(ctx, jobs)
// Output: Job 1, Job 2
// err = "job 3 failed"
\`\`\`

**Constraints:**
- Must execute jobs sequentially
- Must stop on first error
- Must respect context cancellation`,
	initialCode: `package concurrency

import (
	"context"
)

// Job describes work to be done
type Job func(context.Context) error

// TODO: Implement RunSequential
func RunSequential($2) error {
	return nil // TODO: Implement
}`,
	solutionCode: `package concurrency

import (
	"context"
)

type Job func(context.Context) error

func RunSequential(ctx context.Context, jobs []Job) error {
	if ctx == nil {                                                 // Handle nil context
		return nil                                              // Return nil for safety
	}
	for _, job := range jobs {                                      // Iterate through all jobs
		if err := ctx.Err(); err != nil {                       // Check if context cancelled
			return err                                      // Return cancellation error
		}
		if job == nil {                                         // Skip nil jobs
			continue                                        // Move to next job
		}
		if err := job(ctx); err != nil {                        // Execute job
			return err                                      // Return first error
		}
	}
	return ctx.Err()                                                // Return final context state
}`,
	testCode: `package concurrency

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestRunSequential1(t *testing.T) {
	// Test successful execution of all jobs
	ctx := context.Background()
	count := 0
	jobs := []Job{
		func(ctx context.Context) error {
			count++
			return nil
		},
		func(ctx context.Context) error {
			count++
			return nil
		},
		func(ctx context.Context) error {
			count++
			return nil
		},
	}

	err := RunSequential(ctx, jobs)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if count != 3 {
		t.Errorf("expected 3 jobs executed, got %d", count)
	}
}

func TestRunSequential2(t *testing.T) {
	// Test stops on first error
	ctx := context.Background()
	count := 0
	expectedErr := errors.New("job failed")

	jobs := []Job{
		func(ctx context.Context) error {
			count++
			return nil
		},
		func(ctx context.Context) error {
			count++
			return expectedErr
		},
		func(ctx context.Context) error {
			count++
			return nil
		},
	}

	err := RunSequential(ctx, jobs)
	if err != expectedErr {
		t.Errorf("expected error %v, got %v", expectedErr, err)
	}
	if count != 2 {
		t.Errorf("expected 2 jobs executed, got %d", count)
	}
}

func TestRunSequential3(t *testing.T) {
	// Test with empty job slice
	ctx := context.Background()
	jobs := []Job{}

	err := RunSequential(ctx, jobs)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
}

func TestRunSequential4(t *testing.T) {
	// Test with nil context
	count := 0
	jobs := []Job{
		func(ctx context.Context) error {
			count++
			return nil
		},
	}

	err := RunSequential(nil, jobs)
	if err != nil {
		t.Errorf("expected no error with nil context, got %v", err)
	}
	if count != 0 {
		t.Errorf("expected 0 jobs executed with nil context, got %d", count)
	}
}

func TestRunSequential5(t *testing.T) {
	// Test with context cancellation
	ctx, cancel := context.WithCancel(context.Background())
	count := 0

	jobs := []Job{
		func(ctx context.Context) error {
			count++
			return nil
		},
		func(ctx context.Context) error {
			cancel() // Cancel after second job
			count++
			return nil
		},
		func(ctx context.Context) error {
			count++
			return nil
		},
	}

	err := RunSequential(ctx, jobs)
	if err == nil {
		t.Errorf("expected context cancelled error")
	}
	if count > 2 {
		t.Errorf("expected at most 2 jobs executed, got %d", count)
	}
}

func TestRunSequential6(t *testing.T) {
	// Test with nil jobs in slice (should skip)
	ctx := context.Background()
	count := 0

	jobs := []Job{
		func(ctx context.Context) error {
			count++
			return nil
		},
		nil,
		func(ctx context.Context) error {
			count++
			return nil
		},
		nil,
	}

	err := RunSequential(ctx, jobs)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if count != 2 {
		t.Errorf("expected 2 jobs executed, got %d", count)
	}
}

func TestRunSequential7(t *testing.T) {
	// Test sequential order is maintained
	ctx := context.Background()
	order := []int{}

	jobs := []Job{
		func(ctx context.Context) error {
			order = append(order, 1)
			return nil
		},
		func(ctx context.Context) error {
			order = append(order, 2)
			return nil
		},
		func(ctx context.Context) error {
			order = append(order, 3)
			return nil
		},
	}

	err := RunSequential(ctx, jobs)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if len(order) != 3 || order[0] != 1 || order[1] != 2 || order[2] != 3 {
		t.Errorf("expected order [1 2 3], got %v", order)
	}
}

func TestRunSequential8(t *testing.T) {
	// Test with already cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel before running

	jobs := []Job{
		func(ctx context.Context) error {
			return nil
		},
	}

	err := RunSequential(ctx, jobs)
	if err == nil {
		t.Errorf("expected context cancelled error")
	}
}

func TestRunSequential9(t *testing.T) {
	// Test with timeout context
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	count := 0
	jobs := []Job{
		func(ctx context.Context) error {
			count++
			time.Sleep(20 * time.Millisecond)
			return nil
		},
		func(ctx context.Context) error {
			count++
			time.Sleep(20 * time.Millisecond)
			return nil
		},
		func(ctx context.Context) error {
			count++
			time.Sleep(20 * time.Millisecond)
			return nil
		},
	}

	err := RunSequential(ctx, jobs)
	// Should timeout before completing all jobs
	if err == nil {
		t.Errorf("expected timeout error")
	}
}

func TestRunSequential10(t *testing.T) {
	// Test single job
	ctx := context.Background()
	executed := false

	jobs := []Job{
		func(ctx context.Context) error {
			executed = true
			return nil
		},
	}

	err := RunSequential(ctx, jobs)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if !executed {
		t.Errorf("expected job to be executed")
	}
}`,
			hint1: `Loop through jobs array and execute each job one by one, checking ctx.Err() before each execution.`,
			hint2: `Return immediately on first error. Skip nil jobs with continue. Check context before and after the loop.`,
			whyItMatters: `RunSequential is the simplest job execution pattern, useful when jobs must run in order or have dependencies.

**Why Sequential Execution:**
- **Order Matters:** Jobs depend on previous results
- **Resource Constraints:** Only one operation at a time allowed
- **Simplicity:** Easier to debug and reason about
- **Consistency:** Predictable execution order

**Production Pattern:**
\`\`\`go
// Database migration workflow
func RunMigrations(ctx context.Context) error {
    migrations := []Job{
        func(ctx context.Context) error {
            return createUsersTable(ctx)
        },
        func(ctx context.Context) error {
            return addIndexes(ctx)
        },
        func(ctx context.Context) error {
            return seedInitialData(ctx)
        },
    }

    return RunSequential(ctx, migrations)
}

// Ordered data processing pipeline
func ProcessUserData(ctx context.Context, userID string) error {
    pipeline := []Job{
        func(ctx context.Context) error {
            return validateUser(ctx, userID)
        },
        func(ctx context.Context) error {
            return enrichProfile(ctx, userID)
        },
        func(ctx context.Context) error {
            return calculateMetrics(ctx, userID)
        },
        func(ctx context.Context) error {
            return updateCache(ctx, userID)
        },
    }

    return RunSequential(ctx, pipeline)
}

// Deployment steps
func DeployApplication(ctx context.Context) error {
    steps := []Job{
        buildApplication,
        runTests,
        createBackup,
        stopOldVersion,
        deployNewVersion,
        runHealthChecks,
    }

    return RunSequential(ctx, steps)
}

// Financial transaction processing
func ProcessTransaction(ctx context.Context, tx *Transaction) error {
    steps := []Job{
        func(ctx context.Context) error {
            return validateTransaction(ctx, tx)
        },
        func(ctx context.Context) error {
            return checkBalance(ctx, tx.From)
        },
        func(ctx context.Context) error {
            return debitAccount(ctx, tx.From, tx.Amount)
        },
        func(ctx context.Context) error {
            return creditAccount(ctx, tx.To, tx.Amount)
        },
        func(ctx context.Context) error {
            return recordTransaction(ctx, tx)
        },
    }

    return RunSequential(ctx, steps)
}
\`\`\`

**Real-World Benefits:**
- **Predictability:** Always know execution order
- **Debugging:** Easy to trace which step failed
- **Dependencies:** Natural handling of dependent operations
- **Resource Control:** One operation at a time

**When to Use:**
- Database migrations requiring specific order
- Multi-step workflows where each step needs previous results
- Operations that must not run concurrently
- Simple processing pipelines

**Comparison with Parallel:**
- **Sequential:** Predictable, ordered, simpler debugging
- **Parallel:** Faster, but unordered and complex error handling

Without sequential execution, it's hard to ensure operations complete in the correct order, which is critical for migrations, deployments, and transactional workflows.`,	order: 0,
	translations: {
		ru: {
			title: 'Последовательное выполнение',
			description: `Реализуйте **RunSequential**, который выполняет задачи последовательно, одну за другой.

**Требования:**
1. Создайте функцию \`RunSequential(ctx context.Context, jobs []Job) error\`
2. Обработайте nil context (верните nil немедленно)
3. Проверяйте отмену контекста перед каждой задачей
4. Пропускайте nil задачи (продолжайте к следующей)
5. Выполняйте задачи одну за одной
6. Возвращайте первую встреченную ошибку
7. Возвращайте ошибку контекста если отменён

**Определение типа:**
\`\`\`go
type Job func(context.Context) error
\`\`\`

**Пример:**
\`\`\`go
jobs := []Job{
    func(ctx context.Context) error {
        fmt.Println("Job 1")
        return nil
    },
    func(ctx context.Context) error {
        fmt.Println("Job 2")
        return nil
    },
    func(ctx context.Context) error {
        return errors.New("job 3 failed")
    },
    func(ctx context.Context) error {
        fmt.Println("Job 4") // не выполнится
        return nil
    },
}

err := RunSequential(ctx, jobs)
// Вывод: Job 1, Job 2
// err = "job 3 failed"
\`\`\`

**Ограничения:**
- Должен выполнять задачи последовательно
- Должен останавливаться на первой ошибке
- Должен уважать отмену контекста`,
			hint1: `Пройдитесь по массиву jobs и выполните каждую задачу одну за одной, проверяя ctx.Err() перед каждым выполнением.`,
			hint2: `Возвращайте немедленно при первой ошибке. Пропускайте nil задачи через continue. Проверяйте контекст до и после цикла.`,
			whyItMatters: `RunSequential - это простейший паттерн выполнения задач, полезный когда задачи должны выполняться по порядку или имеют зависимости.

**Почему последовательное выполнение:**
- **Порядок важен:** Задачи зависят от предыдущих результатов
- **Ограничения ресурсов:** Только одна операция за раз
- **Простота:** Легче отлаживать и понимать
- **Согласованность:** Предсказуемый порядок выполнения

**Продакшен паттерн:**
\`\`\`go
// Database migration workflow
func RunMigrations(ctx context.Context) error {
    migrations := []Job{
        func(ctx context.Context) error {
            return createUsersTable(ctx)
        },
        func(ctx context.Context) error {
            return addIndexes(ctx)
        },
        func(ctx context.Context) error {
            return seedInitialData(ctx)
        },
    }

    return RunSequential(ctx, migrations)
}

// Ordered data processing pipeline
func ProcessUserData(ctx context.Context, userID string) error {
    pipeline := []Job{
        func(ctx context.Context) error {
            return validateUser(ctx, userID)
        },
        func(ctx context.Context) error {
            return enrichProfile(ctx, userID)
        },
        func(ctx context.Context) error {
            return calculateMetrics(ctx, userID)
        },
        func(ctx context.Context) error {
            return updateCache(ctx, userID)
        },
    }

    return RunSequential(ctx, pipeline)
}

// Deployment steps
func DeployApplication(ctx context.Context) error {
    steps := []Job{
        buildApplication,
        runTests,
        createBackup,
        stopOldVersion,
        deployNewVersion,
        runHealthChecks,
    }

    return RunSequential(ctx, steps)
}

// Financial transaction processing
func ProcessTransaction(ctx context.Context, tx *Transaction) error {
    steps := []Job{
        func(ctx context.Context) error {
            return validateTransaction(ctx, tx)
        },
        func(ctx context.Context) error {
            return checkBalance(ctx, tx.From)
        },
        func(ctx context.Context) error {
            return debitAccount(ctx, tx.From, tx.Amount)
        },
        func(ctx context.Context) error {
            return creditAccount(ctx, tx.To, tx.Amount)
        },
        func(ctx context.Context) error {
            return recordTransaction(ctx, tx)
        },
    }

    return RunSequential(ctx, steps)
}
\`\`\`

**Практические преимущества:**
- **Предсказуемость:** Всегда известен порядок выполнения
- **Отладка:** Легко отследить какой шаг провалился
- **Зависимости:** Естественная обработка зависимых операций
- **Контроль ресурсов:** Одна операция за раз

**Когда использовать:**
- Миграции БД требующие определённого порядка
- Многошаговые workflow где каждый шаг нуждается в результатах предыдущих
- Операции которые не должны выполняться одновременно
- Простые конвейеры обработки

**Сравнение с параллельным:**
- **Последовательный:** Предсказуемый, упорядоченный, проще отлаживать
- **Параллельный:** Быстрее, но неупорядоченный и сложная обработка ошибок

Без последовательного выполнения трудно гарантировать что операции завершатся в правильном порядке, что критично для миграций, развёртываний и транзакционных workflow.`,
			solutionCode: `package concurrency

import (
	"context"
)

type Job func(context.Context) error

func RunSequential(ctx context.Context, jobs []Job) error {
	if ctx == nil {                                                 // Обработка nil контекста
		return nil                                              // Возврат nil для безопасности
	}
	for _, job := range jobs {                                      // Итерация по всем задачам
		if err := ctx.Err(); err != nil {                       // Проверка отмены контекста
			return err                                      // Возврат ошибки отмены
		}
		if job == nil {                                         // Пропуск nil задач
			continue                                        // Переход к следующей задаче
		}
		if err := job(ctx); err != nil {                        // Выполнение задачи
			return err                                      // Возврат первой ошибки
		}
	}
	return ctx.Err()                                                // Возврат финального состояния контекста
}`
		},
		uz: {
			title: 'Ketma-ket bajarish',
			description: `Vazifalarni ketma-ket, birin-ketin bajaradigan **RunSequential** ni amalga oshiring.

**Talablar:**
1. \`RunSequential(ctx context.Context, jobs []Job) error\` funksiyasini yarating
2. nil kontekstni ishlang (darhol nil qaytaring)
3. Har bir vazifadan oldin kontekst bekor qilinishini tekshiring
4. nil vazifalarni o'tkazib yuboring (keyingisiga o'ting)
5. Vazifalarni birin-ketin bajaring
6. Duch kelgan birinchi xatoni qaytaring
7. Agar bekor qilingan bo'lsa kontekst xatosini qaytaring

**Tur ta'rifi:**
\`\`\`go
type Job func(context.Context) error
\`\`\`

**Misol:**
\`\`\`go
jobs := []Job{
    func(ctx context.Context) error {
        fmt.Println("Job 1")
        return nil
    },
    func(ctx context.Context) error {
        fmt.Println("Job 2")
        return nil
    },
    func(ctx context.Context) error {
        return errors.New("job 3 failed")
    },
    func(ctx context.Context) error {
        fmt.Println("Job 4") // bajarilmaydi
        return nil
    },
}

err := RunSequential(ctx, jobs)
// Chiqish: Job 1, Job 2
// err = "job 3 failed"
\`\`\`

**Cheklovlar:**
- Vazifalarni ketma-ket bajarishi kerak
- Birinchi xatoda to'xtashi kerak
- Kontekst bekor qilinishini hurmat qilishi kerak`,
			hint1: `jobs massivini aylanib o'ting va har bir vazifani birin-ketin bajaring, har bir bajarishdan oldin ctx.Err() ni tekshiring.`,
			hint2: `Birinchi xatoda darhol qaytaring. nil vazifalarni continue bilan o'tkazib yuboring. Kontekstni sikldan oldin va keyin tekshiring.`,
			whyItMatters: `RunSequential - bu vazifalarni bajarish uchun eng oddiy pattern, vazifalar tartibda bajarilishi yoki bog'liqliklarga ega bo'lganda foydali.

**Nima uchun ketma-ket bajarish:**
- **Tartib muhim:** Vazifalar oldingi natijalarga bog'liq
- **Resurs cheklovlari:** Bir vaqtda faqat bitta operatsiya
- **Soddalik:** Tuzatish va tushunish osonroq
- **Barqarorlik:** Bajarish tartibini oldindan bilish mumkin

**Ishlab chiqarish patterni:**
\`\`\`go
// Database migration workflow
func RunMigrations(ctx context.Context) error {
    migrations := []Job{
        func(ctx context.Context) error {
            return createUsersTable(ctx)
        },
        func(ctx context.Context) error {
            return addIndexes(ctx)
        },
        func(ctx context.Context) error {
            return seedInitialData(ctx)
        },
    }

    return RunSequential(ctx, migrations)
}

// Ordered data processing pipeline
func ProcessUserData(ctx context.Context, userID string) error {
    pipeline := []Job{
        func(ctx context.Context) error {
            return validateUser(ctx, userID)
        },
        func(ctx context.Context) error {
            return enrichProfile(ctx, userID)
        },
        func(ctx context.Context) error {
            return calculateMetrics(ctx, userID)
        },
        func(ctx context.Context) error {
            return updateCache(ctx, userID)
        },
    }

    return RunSequential(ctx, pipeline)
}

// Deployment steps
func DeployApplication(ctx context.Context) error {
    steps := []Job{
        buildApplication,
        runTests,
        createBackup,
        stopOldVersion,
        deployNewVersion,
        runHealthChecks,
    }

    return RunSequential(ctx, steps)
}

// Financial transaction processing
func ProcessTransaction(ctx context.Context, tx *Transaction) error {
    steps := []Job{
        func(ctx context.Context) error {
            return validateTransaction(ctx, tx)
        },
        func(ctx context.Context) error {
            return checkBalance(ctx, tx.From)
        },
        func(ctx context.Context) error {
            return debitAccount(ctx, tx.From, tx.Amount)
        },
        func(ctx context.Context) error {
            return creditAccount(ctx, tx.To, tx.Amount)
        },
        func(ctx context.Context) error {
            return recordTransaction(ctx, tx)
        },
    }

    return RunSequential(ctx, steps)
}
\`\`\`

**Amaliy foydalari:**
- **Oldindan bilish mumkin:** Bajarish tartibi doim ma'lum
- **Tuzatish:** Qaysi qadam muvaffaqiyatsiz bo'lganligini kuzatish oson
- **Bog'liqliklar:** Bog'liq operatsiyalarni tabiiy boshqarish
- **Resurslarni boshqarish:** Bir vaqtda bitta operatsiya

**Qachon ishlatiladi:**
- Ma'lum tartib talab qiladigan DB migratsiyalari
- Har bir qadam oldingi natijalarni talab qiladigan ko'p bosqichli workflow
- Bir vaqtda bajarilmasligi kerak bo'lgan operatsiyalar
- Oddiy qayta ishlash quvurlari

**Parallel bilan taqqoslash:**
- **Ketma-ket:** Oldindan bilish mumkin, tartibli, tuzatish osonroq
- **Parallel:** Tezroq, lekin tartibsiz va murakkab xato boshqarish

Ketma-ket bajarish bo'lmasa, operatsiyalarning to'g'ri tartibda tugashini kafolatlash qiyin, bu migratsiyalar, joylashlar va tranzaksion workflow lar uchun juda muhim.`,
			solutionCode: `package concurrency

import (
	"context"
)

type Job func(context.Context) error

func RunSequential(ctx context.Context, jobs []Job) error {
	if ctx == nil {                                                 // nil kontekstni ishlash
		return nil                                              // Xavfsizlik uchun nil qaytarish
	}
	for _, job := range jobs {                                      // Barcha vazifalar bo'yicha iteratsiya
		if err := ctx.Err(); err != nil {                       // Kontekst bekor qilinganligini tekshirish
			return err                                      // Bekor qilish xatosini qaytarish
		}
		if job == nil {                                         // nil vazifalarni o'tkazib yuborish
			continue                                        // Keyingi vazifaga o'tish
		}
		if err := job(ctx); err != nil {                        // Vazifani bajarish
			return err                                      // Birinchi xatoni qaytarish
		}
	}
	return ctx.Err()                                                // Kontekstning yakuniy holatini qaytarish
}`
		}
	}
};

export default task;
