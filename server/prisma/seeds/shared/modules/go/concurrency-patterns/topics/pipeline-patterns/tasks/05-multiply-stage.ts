import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-pipeline-multiply-stage',
	title: 'MultiplyStage',
	difficulty: 'medium',	tags: ['go', 'concurrency', 'pipeline', 'stage', 'context'],
	estimatedTime: '25m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **MultiplyStage** that returns a pipeline Stage function for multiplying numbers by a factor with context cancellation.

**Type Definition:**
\`\`\`go
type Stage func(context.Context, <-chan int) <-chan int
\`\`\`

**Requirements:**
1. Create function \`MultiplyStage(factor int) Stage\`
2. Return a Stage function that takes (ctx, in) and returns output channel
3. Launch single goroutine (no parallelism needed for simple multiply)
4. Use nested select for ctx.Done() checks (on receive and send)
5. Multiply each value by factor before sending
6. Close output channel when input closes or context cancelled
7. Return output channel immediately

**Example:**
\`\`\`go
ctx := context.Background()
stage := MultiplyStage(3)

in := Gen(1, 2, 3, 4, 5)
out := stage(ctx, in)

for v := range out {
    fmt.Println(v)
}
// Output: 3 6 9 12 15

// Chaining stages
square := SquareStage(2)
multiply := MultiplyStage(10)

in = Gen(2, 3, 4)
out = multiply(ctx, square(ctx, in))

for v := range out {
    fmt.Println(v)
}
// Output: 40 90 160 (squared then multiplied by 10)
\`\`\`

**Constraints:**
- Must return Stage function type
- Must use nested select for proper cancellation
- Must close output channel properly
- Single goroutine is sufficient`,
	initialCode: `package concurrency

import "context"

type Stage func(context.Context, <-chan int) <-chan int

// TODO: Implement MultiplyStage
func MultiplyStage(factor int) Stage {
	// TODO: Implement
}`,
	solutionCode: `package concurrency

import "context"

type Stage func(context.Context, <-chan int) <-chan int

func MultiplyStage(factor int) Stage {
	return func(ctx context.Context, in <-chan int) <-chan int { // Return Stage function
		out := make(chan int)                               // Create output channel
		go func() {                                         // Launch goroutine
			defer close(out)                            // Always close output
			for {                                       // Infinite loop
				select {
				case <-ctx.Done():                  // Context cancelled
					return                      // Exit goroutine
				case v, ok := <-in:                 // Read from input
					if !ok {                    // Channel closed
						return              // Exit goroutine
					}
					select {
					case <-ctx.Done():          // Check before send
						return              // Exit goroutine
					case out <- v * factor:     // Send multiplied value
					}
				}
			}
		}()
		return out                                          // Return immediately
	}
}`,
			hint1: `Return a closure that captures the factor parameter and implements the Stage function signature.`,
			hint2: `Use the same nested select pattern as SquareStage: outer select for receiving (ctx.Done() or in), inner select for sending (ctx.Done() or out).`,
			testCode: `package concurrency

import (
	"context"
	"testing"
	"time"
)

func Test1(t *testing.T) {
	stage := MultiplyStage(3)
	in := make(chan int)
	close(in)
	out := stage(context.Background(), in)
	count := 0
	for range out {
		count++
	}
	if count != 0 {
		t.Errorf("expected 0 values from closed input, got %d", count)
	}
}

func Test2(t *testing.T) {
	stage := MultiplyStage(5)
	in := make(chan int, 1)
	in <- 4
	close(in)
	out := stage(context.Background(), in)
	v := <-out
	if v != 20 {
		t.Errorf("expected 20, got %d", v)
	}
}

func Test3(t *testing.T) {
	stage := MultiplyStage(2)
	in := make(chan int, 5)
	expected := []int{2, 4, 6, 8, 10}
	for i := 1; i <= 5; i++ {
		in <- i
	}
	close(in)
	out := stage(context.Background(), in)
	results := make([]int, 0, 5)
	for v := range out {
		results = append(results, v)
	}
	for i, v := range results {
		if v != expected[i] {
			t.Errorf("at index %d: expected %d, got %d", i, expected[i], v)
		}
	}
}

func Test4(t *testing.T) {
	stage := MultiplyStage(0)
	in := make(chan int, 1)
	in <- 42
	close(in)
	out := stage(context.Background(), in)
	v := <-out
	if v != 0 {
		t.Errorf("expected 0 (42*0), got %d", v)
	}
}

func Test5(t *testing.T) {
	stage := MultiplyStage(-3)
	in := make(chan int, 1)
	in <- 5
	close(in)
	out := stage(context.Background(), in)
	v := <-out
	if v != -15 {
		t.Errorf("expected -15, got %d", v)
	}
}

func Test6(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	stage := MultiplyStage(2)
	in := make(chan int, 5)
	for i := 1; i <= 5; i++ {
		in <- i
	}
	close(in)
	out := stage(ctx, in)
	count := 0
	for range out {
		count++
	}
	if count > 5 {
		t.Errorf("expected <= 5 values with cancelled context, got %d", count)
	}
}

func Test7(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	stage := MultiplyStage(10)
	in := make(chan int, 3)
	in <- 1
	in <- 2
	in <- 3
	close(in)
	out := stage(ctx, in)
	for range out {
	}
}

func Test8(t *testing.T) {
	done := make(chan bool, 1)
	go func() {
		stage := MultiplyStage(5)
		in := make(chan int)
		close(in)
		_ = stage(context.Background(), in)
		done <- true
	}()
	select {
	case <-done:
	case <-time.After(100 * time.Millisecond):
		t.Error("MultiplyStage should return immediately")
	}
}

func Test9(t *testing.T) {
	stage := MultiplyStage(1)
	in := make(chan int, 1)
	in <- 99
	close(in)
	out := stage(context.Background(), in)
	v := <-out
	if v != 99 {
		t.Errorf("expected 99, got %d", v)
	}
}

func Test10(t *testing.T) {
	stage := MultiplyStage(-1)
	in := make(chan int, 1)
	in <- -7
	close(in)
	out := stage(context.Background(), in)
	v := <-out
	if v != 7 {
		t.Errorf("expected 7, got %d", v)
	}
}
`,
	whyItMatters: `MultiplyStage demonstrates parameterized pipeline stages, allowing creation of reusable transformation stages with configurable behavior.

**Why Parameterized Stages:**
- **Reusability:** Same stage logic with different parameters
- **Configuration:** Customize stage behavior without code changes
- **Composition:** Build complex pipelines from simple parameterized stages
- **Testing:** Easy to test with different parameter values

**Production Pattern:**
\`\`\`go
// Configurable transform stage
func TransformStage(transform func(int) int) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            for {
                select {
                case <-ctx.Done():
                    return
                case v, ok := <-in:
                    if !ok {
                        return
                    }
                    select {
                    case <-ctx.Done():
                        return
                    case out <- transform(v):
                    }
                }
            }
        }()
        return out
    }
}

// Currency conversion stage
func CurrencyConversionStage(rate float64) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            for {
                select {
                case <-ctx.Done():
                    return
                case amount, ok := <-in:
                    if !ok {
                        return
                    }
                    converted := int(float64(amount) * rate)
                    select {
                    case <-ctx.Done():
                        return
                    case out <- converted:
                    }
                }
            }
        }()
        return out
    }
}

// Discount application stage
func DiscountStage(percentOff int) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            for {
                select {
                case <-ctx.Done():
                    return
                case price, ok := <-in:
                    if !ok {
                        return
                    }
                    discounted := price * (100 - percentOff) / 100
                    select {
                    case <-ctx.Done():
                        return
                    case out <- discounted:
                    }
                }
            }
        }()
        return out
    }
}

// Tax calculation stage
func TaxStage(taxRate float64) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            for {
                select {
                case <-ctx.Done():
                    return
                case amount, ok := <-in:
                    if !ok {
                        return
                    }
                    withTax := int(float64(amount) * (1 + taxRate))
                    select {
                    case <-ctx.Done():
                        return
                    case out <- withTax:
                    }
                }
            }
        }()
        return out
    }
}

// Round to nearest stage
func RoundToStage(nearest int) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            for {
                select {
                case <-ctx.Done():
                    return
                case value, ok := <-in:
                    if !ok {
                        return
                    }
                    rounded := (value / nearest) * nearest
                    select {
                    case <-ctx.Done():
                        return
                    case out <- rounded:
                    }
                }
            }
        }()
        return out
    }
}

// E-commerce pricing pipeline
func CalculateFinalPrice() {
    ctx := context.Background()

    // Define stages
    discount := DiscountStage(20)      // 20% off
    tax := TaxStage(0.08)               // 8% tax
    roundUp := RoundToStage(10)         // Round to nearest $10

    // Process prices
    prices := Gen(100, 250, 500, 1000)
    final := roundUp(ctx, tax(ctx, discount(ctx, prices)))

    for price := range final {
        fmt.Printf("Final price: $%d\n", price)
    }
}

// Multi-currency pipeline
func ConvertPrices(fromRate, toRate float64) {
    ctx := context.Background()

    // Create conversion stages
    toUSD := CurrencyConversionStage(fromRate)
    toTarget := CurrencyConversionStage(toRate)

    prices := Gen(100, 200, 300, 400, 500)
    converted := toTarget(ctx, toUSD(ctx, prices))

    for price := range converted {
        fmt.Println(price)
    }
}

// Configurable math pipeline
func MathPipeline(ops []func(int) int) {
    ctx := context.Background()

    // Build stages from operations
    var stages []Stage
    for _, op := range ops {
        stages = append(stages, TransformStage(op))
    }

    // Apply all stages
    source := Gen(1, 2, 3, 4, 5)
    result := BuildPipeline(ctx, source, stages...)

    for v := range result {
        fmt.Println(v)
    }
}

// Usage example
func Example() {
    ops := []func(int) int{
        func(x int) int { return x * 2 },    // Double
        func(x int) int { return x + 10 },   // Add 10
        func(x int) int { return x * x },    // Square
    }
    MathPipeline(ops)
}
\`\`\`

**Real-World Benefits:**
- **Business Logic:** Easily adjust discounts, taxes, rates
- **A/B Testing:** Test different parameter values
- **Configuration:** Change behavior without code deployment
- **Flexibility:** Same code handles different scenarios

**Common Parameterized Stages:**
- **Scaling:** MultiplyStage, DivideStage, PercentageStage
- **Offset:** AddStage, SubtractStage
- **Rounding:** RoundToStage, FloorStage, CeilStage
- **Conversion:** CurrencyStage, UnitConversionStage
- **Transformation:** MapStage with custom function

**Configuration Patterns:**
- **Static:** Parameters set at stage creation
- **Dynamic:** Parameters can change during processing
- **Environment:** Parameters from config files/env vars
- **User-Driven:** Parameters from user input/settings

Without parameterized stages, you'd need separate functions for each variation (Multiply2, Multiply3, etc.), leading to code duplication and reduced flexibility.`,	order: 4,
	translations: {
		ru: {
			title: 'Этап умножения в pipeline',
			description: `Реализуйте **MultiplyStage**, который возвращает функцию Stage pipeline для умножения чисел на фактор с отменой контекста.

**Определение типа:**
\`\`\`go
type Stage func(context.Context, <-chan int) <-chan int
\`\`\`

**Требования:**
1. Создайте функцию \`MultiplyStage(factor int) Stage\`
2. Верните функцию Stage которая принимает (ctx, in) и возвращает выходной канал
3. Запустите одну горутину (параллелизм не нужен для простого умножения)
4. Используйте вложенный select для проверок ctx.Done() (при чтении и отправке)
5. Умножьте каждое значение на factor перед отправкой
6. Закройте выходной канал когда входной закрыт или контекст отменён
7. Верните выходной канал немедленно

**Пример:**
\`\`\`go
ctx := context.Background()
stage := MultiplyStage(3)

in := Gen(1, 2, 3, 4, 5)
out := stage(ctx, in)

for v := range out {
    fmt.Println(v)
}
// Вывод: 3 6 9 12 15
\`\`\`

**Ограничения:**
- Должен возвращать функцию типа Stage
- Должен использовать вложенный select для правильной отмены
- Должен правильно закрывать выходной канал
- Достаточно одной горутины`,
			hint1: `Верните замыкание которое захватывает параметр factor и реализует сигнатуру функции Stage.`,
			hint2: `Используйте тот же паттерн вложенного select что и в SquareStage: внешний select для получения, внутренний select для отправки.`,
			whyItMatters: `MultiplyStage демонстрирует параметризованные стадии pipeline, позволяя создавать переиспользуемые стадии трансформации с настраиваемым поведением.

**Почему параметризованные стадии важны:**
- **Переиспользуемость:** Одна и та же логика стадии с разными параметрами
- **Конфигурация:** Настройка поведения стадии без изменения кода
- **Композиция:** Построение сложных pipeline из простых параметризованных стадий
- **Тестирование:** Легко тестировать с разными значениями параметров

**Production паттерны:**
\`\`\`go
// Конвертация валют
func CurrencyConversionStage(rate float64) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            for {
                select {
                case <-ctx.Done():
                    return
                case amount, ok := <-in:
                    if !ok {
                        return
                    }
                    converted := int(float64(amount) * rate)
                    select {
                    case <-ctx.Done():
                        return
                    case out <- converted:
                    }
                }
            }
        }()
        return out
    }
}

// Применение скидок
func DiscountStage(percentOff int) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            for {
                select {
                case <-ctx.Done():
                    return
                case price, ok := <-in:
                    if !ok {
                        return
                    }
                    discounted := price * (100 - percentOff) / 100
                    select {
                    case <-ctx.Done():
                        return
                    case out <- discounted:
                    }
                }
            }
        }()
        return out
    }
}

// Расчёт налогов
func TaxStage(taxRate float64) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            for {
                select {
                case <-ctx.Done():
                    return
                case amount, ok := <-in:
                    if !ok {
                        return
                    }
                    withTax := int(float64(amount) * (1 + taxRate))
                    select {
                    case <-ctx.Done():
                        return
                    case out <- withTax:
                    }
                }
            }
        }()
        return out
    }
}

// Округление до ближайшего
func RoundToStage(nearest int) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            for {
                select {
                case <-ctx.Done():
                    return
                case value, ok := <-in:
                    if !ok {
                        return
                    }
                    rounded := (value / nearest) * nearest
                    select {
                    case <-ctx.Done():
                        return
                    case out <- rounded:
                    }
                }
            }
        }()
        return out
    }
}

// E-commerce ценообразование
func CalculateFinalPrice() {
    ctx := context.Background()

    // Определяем стадии
    discount := DiscountStage(20)      // 20% скидка
    tax := TaxStage(0.08)              // 8% налог
    roundUp := RoundToStage(10)        // Округление до $10

    // Обрабатываем цены
    prices := Gen(100, 250, 500, 1000)
    final := roundUp(ctx, tax(ctx, discount(ctx, prices)))

    for price := range final {
        fmt.Printf("Итоговая цена: $%d\\n", price)
    }
}

// Мультивалютный pipeline
func ConvertPrices(fromRate, toRate float64) {
    ctx := context.Background()

    // Создаём стадии конвертации
    toUSD := CurrencyConversionStage(fromRate)
    toTarget := CurrencyConversionStage(toRate)

    prices := Gen(100, 200, 300, 400, 500)
    converted := toTarget(ctx, toUSD(ctx, prices))

    for price := range converted {
        fmt.Println(price)
    }
}

// Настраиваемый трансформации
func TransformStage(transform func(int) int) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            for {
                select {
                case <-ctx.Done():
                    return
                case v, ok := <-in:
                    if !ok {
                        return
                    }
                    select {
                    case <-ctx.Done():
                        return
                    case out <- transform(v):
                    }
                }
            }
        }()
        return out
    }
}

// Математический pipeline с конфигурацией
func MathPipeline(ops []func(int) int) {
    ctx := context.Background()

    // Строим стадии из операций
    var stages []Stage
    for _, op := range ops {
        stages = append(stages, TransformStage(op))
    }

    // Применяем все стадии
    source := Gen(1, 2, 3, 4, 5)
    result := BuildPipeline(ctx, source, stages...)

    for v := range result {
        fmt.Println(v)
    }
}

// Пример использования
func Example() {
    ops := []func(int) int{
        func(x int) int { return x * 2 },    // Удвоить
        func(x int) int { return x + 10 },   // Добавить 10
        func(x int) int { return x * x },    // Квадрат
    }
    MathPipeline(ops)
}
\`\`\`

**Реальные преимущества:**
- **Бизнес-логика:** Легко настраивать скидки, налоги, курсы
- **A/B тестирование:** Тестировать разные значения параметров
- **Конфигурация:** Изменение поведения без деплоя кода
- **Гибкость:** Один код обрабатывает разные сценарии

**Общие параметризованные стадии:**
- **Масштабирование:** MultiplyStage, DivideStage, PercentageStage
- **Смещение:** AddStage, SubtractStage
- **Округление:** RoundToStage, FloorStage, CeilStage
- **Конвертация:** CurrencyStage, UnitConversionStage
- **Трансформация:** MapStage с пользовательской функцией

**Паттерны конфигурации:**
- **Статические:** Параметры устанавливаются при создании стадии
- **Динамические:** Параметры могут меняться во время обработки
- **Из окружения:** Параметры из config файлов/env vars
- **От пользователя:** Параметры из пользовательского ввода/настроек

Без параметризованных стадий нужны были бы отдельные функции для каждой вариации (Multiply2, Multiply3 и т.д.), приводя к дублированию кода и снижению гибкости.`,
			solutionCode: `package concurrency

import "context"

type Stage func(context.Context, <-chan int) <-chan int

func MultiplyStage(factor int) Stage {
	return func(ctx context.Context, in <-chan int) <-chan int { // Возвращаем функцию Stage
		out := make(chan int)                               // Создаём выходной канал
		go func() {                                         // Запускаем горутину
			defer close(out)                            // Всегда закрываем выход
			for {                                       // Бесконечный цикл
				select {
				case <-ctx.Done():                  // Контекст отменён
					return                      // Выходим из горутины
				case v, ok := <-in:                 // Читаем из входа
					if !ok {                    // Канал закрыт
						return              // Выходим из горутины
					}
					select {
					case <-ctx.Done():          // Проверяем перед отправкой
						return              // Выходим из горутины
					case out <- v * factor:     // Отправляем умноженное значение
					}
				}
			}
		}()
		return out                                          // Возвращаем немедленно
	}
}`
		},
		uz: {
			title: 'Pipeline da ko\'paytirish bosqichi',
			description: `Kontekst bekor qilish bilan raqamlarni faktorga ko'paytiradigan pipeline Stage funksiyasini qaytaruvchi **MultiplyStage** ni amalga oshiring.

**Tur ta'rifi:**
\`\`\`go
type Stage func(context.Context, <-chan int) <-chan int
\`\`\`

**Talablar:**
1. \`MultiplyStage(factor int) Stage\` funksiyasini yarating
2. (ctx, in) qabul qiladigan va chiqish kanalini qaytaruvchi Stage funksiyasini qaytaring
3. Bitta goroutine ishga tushiring (oddiy ko'paytirish uchun paralellik kerak emas)
4. ctx.Done() tekshiruvlari uchun ichki selectdan foydalaning (qabul qilish va yuborishda)
5. Yuborishdan oldin har bir qiymatni faktorga ko'paytiring
6. Kirish yopilganda yoki kontekst bekor qilinganda chiqish kanalini yoping
7. Chiqish kanalini darhol qaytaring

**Misol:**
\`\`\`go
ctx := context.Background()
stage := MultiplyStage(3)

in := Gen(1, 2, 3, 4, 5)
out := stage(ctx, in)

for v := range out {
    fmt.Println(v)
}
// Natija: 3 6 9 12 15
\`\`\`

**Cheklovlar:**
- Stage turi funksiyasini qaytarishi kerak
- To'g'ri bekor qilish uchun ichki selectdan foydalanishi kerak
- Chiqish kanalini to'g'ri yopishi kerak
- Bitta goroutine yetarli`,
			hint1: `Faktor parametrini ushlaydigan va Stage funksiya imzosini amalga oshiradigan yopilishni qaytaring.`,
			hint2: `SquareStage kabi ichki select patternidan foydalaning: tashqi select qabul qilish uchun, ichki select yuborish uchun.`,
			whyItMatters: `MultiplyStage sozlanishi mumkin bo'lgan xatti-harakat bilan qayta ishlatilishi mumkin transformatsiya bosqichlarini yaratishga imkon beradigan parametrlangan pipeline bosqichlarini namoyish etadi.

**Nima uchun parametrlangan bosqichlar muhim:**
- **Qayta ishlatish:** Turli parametrlar bilan bir xil bosqich mantiqi
- **Konfiguratsiya:** Kodni o'zgartirmasdan bosqich xatti-harakatini sozlash
- **Kompozitsiya:** Oddiy parametrlangan bosqichlardan murakkab pipelinelar qurish
- **Test qilish:** Turli parametr qiymatlari bilan osongina test qilish

**Production patternlar:**
\`\`\`go
// Valyuta konvertatsiyasi
func CurrencyConversionStage(rate float64) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            for {
                select {
                case <-ctx.Done():
                    return
                case amount, ok := <-in:
                    if !ok {
                        return
                    }
                    converted := int(float64(amount) * rate)
                    select {
                    case <-ctx.Done():
                        return
                    case out <- converted:
                    }
                }
            }
        }()
        return out
    }
}

// Chegirmalarni qo'llash
func DiscountStage(percentOff int) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            for {
                select {
                case <-ctx.Done():
                    return
                case price, ok := <-in:
                    if !ok {
                        return
                    }
                    discounted := price * (100 - percentOff) / 100
                    select {
                    case <-ctx.Done():
                        return
                    case out <- discounted:
                    }
                }
            }
        }()
        return out
    }
}

// Soliq hisoblash
func TaxStage(taxRate float64) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            for {
                select {
                case <-ctx.Done():
                    return
                case amount, ok := <-in:
                    if !ok {
                        return
                    }
                    withTax := int(float64(amount) * (1 + taxRate))
                    select {
                    case <-ctx.Done():
                        return
                    case out <- withTax:
                    }
                }
            }
        }()
        return out
    }
}

// Eng yaqinga yaxlitlash
func RoundToStage(nearest int) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            for {
                select {
                case <-ctx.Done():
                    return
                case value, ok := <-in:
                    if !ok {
                        return
                    }
                    rounded := (value / nearest) * nearest
                    select {
                    case <-ctx.Done():
                        return
                    case out <- rounded:
                    }
                }
            }
        }()
        return out
    }
}

// E-commerce narxlash
func CalculateFinalPrice() {
    ctx := context.Background()

    // Bosqichlarni belgilash
    discount := DiscountStage(20)      // 20% chegirma
    tax := TaxStage(0.08)              // 8% soliq
    roundUp := RoundToStage(10)        // $10 gacha yaxlitlash

    // Narxlarni qayta ishlash
    prices := Gen(100, 250, 500, 1000)
    final := roundUp(ctx, tax(ctx, discount(ctx, prices)))

    for price := range final {
        fmt.Printf("Yakuniy narx: $%d\\n", price)
    }
}

// Ko'p valyutali pipeline
func ConvertPrices(fromRate, toRate float64) {
    ctx := context.Background()

    // Konvertatsiya bosqichlarini yaratish
    toUSD := CurrencyConversionStage(fromRate)
    toTarget := CurrencyConversionStage(toRate)

    prices := Gen(100, 200, 300, 400, 500)
    converted := toTarget(ctx, toUSD(ctx, prices))

    for price := range converted {
        fmt.Println(price)
    }
}

// Sozlanishi mumkin transformatsiya
func TransformStage(transform func(int) int) Stage {
    return func(ctx context.Context, in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            for {
                select {
                case <-ctx.Done():
                    return
                case v, ok := <-in:
                    if !ok {
                        return
                    }
                    select {
                    case <-ctx.Done():
                        return
                    case out <- transform(v):
                    }
                }
            }
        }()
        return out
    }
}

// Konfiguratsiyali matematik pipeline
func MathPipeline(ops []func(int) int) {
    ctx := context.Background()

    // Operatsiyalardan bosqichlarni qurish
    var stages []Stage
    for _, op := range ops {
        stages = append(stages, TransformStage(op))
    }

    // Barcha bosqichlarni qo'llash
    source := Gen(1, 2, 3, 4, 5)
    result := BuildPipeline(ctx, source, stages...)

    for v := range result {
        fmt.Println(v)
    }
}

// Foydalanish misoli
func Example() {
    ops := []func(int) int{
        func(x int) int { return x * 2 },    // Ikki barobar
        func(x int) int { return x + 10 },   // 10 qo'shish
        func(x int) int { return x * x },    // Kvadrat
    }
    MathPipeline(ops)
}
\`\`\`

**Haqiqiy foydalari:**
- **Biznes mantiq:** Chegirmalar, soliqlar, kurslarni osongina sozlash
- **A/B test qilish:** Turli parametr qiymatlarini sinash
- **Konfiguratsiya:** Kod deploy qilmasdan xatti-harakatni o'zgartirish
- **Moslashuvchanlik:** Bir kod turli stsenariylarni qayta ishlaydi

**Umumiy parametrlangan bosqichlar:**
- **Masshtablash:** MultiplyStage, DivideStage, PercentageStage
- **Siljitish:** AddStage, SubtractStage
- **Yaxlitlash:** RoundToStage, FloorStage, CeilStage
- **Konvertatsiya:** CurrencyStage, UnitConversionStage
- **Transformatsiya:** Maxsus funksiya bilan MapStage

**Konfiguratsiya patternlari:**
- **Statik:** Parametrlar bosqich yaratilganda o'rnatiladi
- **Dinamik:** Parametrlar qayta ishlash davomida o'zgarishi mumkin
- **Muhitdan:** Config fayllar/env vars dan parametrlar
- **Foydalanuvchidan:** Foydalanuvchi kiritgan/sozlamalaridan parametrlar

Parametrlangan bosqichlar bo'lmasa, har bir variatsiya uchun alohida funksiyalar kerak bo'lar edi (Multiply2, Multiply3 va h.k.), bu kod takrorlanishi va kamroq moslashuvchanlikka olib keladi.`,
			solutionCode: `package concurrency

import "context"

type Stage func(context.Context, <-chan int) <-chan int

func MultiplyStage(factor int) Stage {
	return func(ctx context.Context, in <-chan int) <-chan int { // Stage funksiyasini qaytaramiz
		out := make(chan int)                               // Chiqish kanalini yaratamiz
		go func() {                                         // Goroutine ishga tushiramiz
			defer close(out)                            // Har doim chiqishni yopamiz
			for {                                       // Cheksiz tsikl
				select {
				case <-ctx.Done():                  // Kontekst bekor qilindi
					return                      // Goroutinedan chiqamiz
				case v, ok := <-in:                 // Kirishdan o'qiymiz
					if !ok {                    // Kanal yopilgan
						return              // Goroutinedan chiqamiz
					}
					select {
					case <-ctx.Done():          // Yuborishdan oldin tekshiramiz
						return              // Goroutinedan chiqamiz
					case out <- v * factor:     // Ko'paytirilgan qiymatni yuboramiz
					}
				}
			}
		}()
		return out                                          // Darhol qaytaramiz
	}
}`
		}
	}
};

export default task;
