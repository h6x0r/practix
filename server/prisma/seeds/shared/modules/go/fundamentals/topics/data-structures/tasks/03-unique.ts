import { Task } from "../../../../types";

export const task: Task = {
  slug: "go-fundamentals-unique",
  title: "Remove Duplicates from Slice",
  difficulty: "easy",
  tags: ["go", "data-structures", "maps/slices/strings"],
  estimatedTime: "15-20m",
  isPremium: false,
  youtubeUrl: "",
  description: `Implement **Unique** that removes duplicates from an integer slice while preserving the order of first appearances.

**Requirements:**
1. Create function \`Unique(in []int) []int\`
2. Handle empty slices (return nil)
3. Remove all duplicate elements
4. Preserve the order of first appearances
5. Use a set-like structure to track seen elements
6. Pre-allocate result slice for efficiency
7. Return the deduplicated slice

**Example:**
\`\`\`go
result := Unique([]int{1, 2, 2, 3, 1, 4, 3})
// result = []int{1, 2, 3, 4}

result2 := Unique([]int{5, 3, 5, 1, 3, 2, 1})
// result2 = []int{5, 3, 1, 2}

result3 := Unique([]int{})
// result3 = nil
\`\`\`

**Constraints:**
- Must preserve order of first appearances
- Must not modify the input slice
- Must efficiently track seen elements using a map
- Should return nil for empty input`,
  initialCode: `package datastructures

// TODO: Implement Unique
func Unique(in []int) []int {
	// TODO: Implement
}`,
  solutionCode: `package datastructures

func Unique(in []int) []int {
	if len(in) == 0 {                                       // Handle empty slice
		return nil                                      // Return nil for empty input
	}
	seen := make(map[int]struct{}, len(in))                  // Create seen set with capacity
	result := make([]int, 0, len(in))                        // Pre-allocate result slice
	for _, value := range in {                              // Iterate through input slice
		if _, ok := seen[value]; ok {                  // Check if value was already seen
			continue                                // Skip duplicate
		}
		seen[value] = struct{}{}                        // Mark value as seen
		result = append(result, value)                  // Add unique value to result
	}
	return result                                           // Return deduplicated slice
}`,
  testCode: `package datastructures

import (
	"reflect"
	"testing"
)

func Test1(t *testing.T) {
	// Basic deduplication with integers
	result := Unique([]int{1, 2, 2, 3, 1, 4, 3})
	expected := []int{1, 2, 3, 4}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test2(t *testing.T) {
	// Preserves order of first appearance
	result := Unique([]int{5, 3, 5, 1, 3, 2, 1})
	expected := []int{5, 3, 1, 2}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test3(t *testing.T) {
	// Empty slice
	result := Unique([]int{})
	if result != nil {
		t.Errorf("expected nil, got %v", result)
	}
}

func Test4(t *testing.T) {
	// All unique elements
	result := Unique([]int{1, 2, 3, 4, 5})
	expected := []int{1, 2, 3, 4, 5}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test5(t *testing.T) {
	// All same elements
	result := Unique([]int{7, 7, 7, 7})
	expected := []int{7}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test6(t *testing.T) {
	// Single element
	result := Unique([]int{42})
	expected := []int{42}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test7(t *testing.T) {
	// Preserves order of first appearance
	result := Unique([]int{5, 3, 5, 1, 3, 2, 1})
	expected := []int{5, 3, 1, 2}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test8(t *testing.T) {
	// Two elements
	result := Unique([]int{1, 2})
	expected := []int{1, 2}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test9(t *testing.T) {
	// Consecutive duplicates
	result := Unique([]int{1, 1, 2, 2, 3, 3})
	expected := []int{1, 2, 3}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test10(t *testing.T) {
	// Negative numbers
	result := Unique([]int{-1, -2, -1, -3, -2})
	expected := []int{-1, -2, -3}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("expected %v, got %v", expected, result)
	}
}`,
  hint1: `Use a map with struct{} values to track which elements you\`ve already seen, maintaining a set-like behavior.`,
  hint2: `Iterate through the input slice once, checking the map before adding each element to the result.`,
  whyItMatters: `Unique is essential for removing duplicate data while maintaining order, a common operation in data processing pipelines that prevents processing the same record multiple times.

**Why Deduplication:**
- **Data Quality:** Remove duplicate records that cause incorrect results
- **Performance:** Process each unique element only once, reducing computational overhead
- **Memory Efficiency:** Avoid storing redundant data in results
- **Correctness:** Prevent double-counting in aggregations and reports

**Production Pattern:**
\`\`\`go
// Remove duplicate user IDs from logs
func GetUniqueActiveUsers(logs []LogEntry) []string {
    userIDs := make([]string, len(logs))
    for i, log := range logs {
        userIDs[i] = log.UserID
    }
    return Unique(userIDs)
}

// Deduplicate search results
func SearchWithUniqueResults(query string) []Document {
    allResults := database.Search(query)

    var ids []int64
    for _, doc := range allResults {
        ids = append(ids, doc.ID)
    }

    uniqueIDs := Unique(ids)
    result := make([]Document, len(uniqueIDs))
    for i, id := range uniqueIDs {
        result[i] = database.GetDocument(id)
    }
    return result
}

// Remove duplicate email notifications
func SendUniqueNotifications(recipients []string) error {
    unique := Unique(recipients)

    for _, email := range unique {
        if err := sendEmail(email); err != nil {
            return err
        }
    }
    return nil
}

// Deduplicate shopping cart items
type CartItem struct {
    ProductID string
    Quantity  int
}

func MergeCartItems(items []CartItem) []CartItem {
    seen := make(map[string]int)

    for _, item := range items {
        seen[item.ProductID] += item.Quantity
    }

    result := make([]CartItem, 0, len(seen))
    for productID, quantity := range seen {
        result = append(result, CartItem{productID, quantity})
    }
    return result
}

// Remove duplicate log levels for filtering
func GetUniqueSeverityLevels(logs []LogEntry) []string {
    levels := make([]string, len(logs))
    for i, log := range logs {
        levels[i] = log.Severity
    }
    return Unique(levels)
}

// Deduplicate file paths in build artifacts
func GetUniqueSourceFiles(compileErrors []CompileError) []string {
    files := make([]string, len(compileErrors))
    for i, err := range compileErrors {
        files[i] = err.FilePath
    }
    return Unique(files)
}

// Remove duplicate permission scopes
func GetRequiredPermissions(requests []PermissionRequest) []string {
    scopes := make([]string, len(requests))
    for i, req := range requests {
        scopes[i] = req.Scope
    }
    return Unique(scopes)
}

// Deduplicate tags from user input
func NormalizeTags(input []string) []string {
    // Convert to lowercase and deduplicate
    normalized := make([]string, len(input))
    for i, tag := range input {
        normalized[i] = strings.ToLower(tag)
    }
    return Unique(normalized)
}

// Remove duplicate hosts from service discovery
func GetHealthyHosts(endpoints []ServiceEndpoint) []string {
    hosts := make([]string, 0, len(endpoints))
    for _, ep := range endpoints {
        if ep.IsHealthy {
            hosts = append(hosts, ep.Host)
        }
    }
    return Unique(hosts)
}
\`\`\`

**Real-World Benefits:**
- **ETL Pipelines:** Deduplicate records from multiple sources
- **Search Results:** Remove duplicate documents in search engines
- **Analytics:** Count unique users, products, or events accurately
- **Data Import:** Clean imported data before database insertion
- **API Responses:** Remove duplicate results before sending to clients

**Common Use Cases:**
- Deduplicating database query results
- Removing duplicate items from user selections
- Cleaning up merged datasets from multiple sources
- Creating unique lists for dropdown menus
- Counting distinct values in analytics

Without Unique, duplicate data could skew analysis results, cause incorrect aggregations, and lead to inefficient processing of redundant records.`,
  order: 2,
  translations: {
    ru: {
      title: "Удаление дубликатов из слайса",
      description: `Реализуйте **Unique**, который устраняет дубликаты в слайсе, сохраняя порядок первого появления.

**Требования:**
1. Создайте функцию \`Unique(in []int) []int\`
2. Обработайте пустые слайсы (верните пустой слайс или nil)
3. Удалите все дублирующиеся элементы
4. Сохраните порядок первого появления
5. Используйте set-подобную структуру для отслеживания seen элементов
6. Предварительно выделите результирующий слайс для эффективности
7. Верните дедубликированный слайс

**Пример:**
\`\`\`go
result := Unique([]int{1, 2, 2, 3, 1, 4, 3})
// result = []int{1, 2, 3, 4}

result2 := Unique([]string{"apple", "banana", "apple", "cherry"})
// result2 = []string{"apple", "banana", "cherry"}

result3 := Unique([]int{})
// result3 = [] или nil
\`\`\`

**Ограничения:**
- Должен сохранять порядок первого появления
- Должен работать с любым comparable типом
- Не должен модифицировать входной слайс
- Должен эффективно отслеживать seen элементы
- Должен возвращать nil для пустого входа`,
      hint1: `Используйте map со struct{} значениями для отслеживания уже встреченных элементов, поддерживая set-подобное поведение.`,
      hint2: `Пройдите через входной слайс один раз, проверяя map перед добавлением каждого элемента в результат.`,
      whyItMatters: `Unique необходим для удаления дублирующихся данных сохраняя порядок, часто встречаемая операция в data processing pipelines которая предотвращает обработку одной записи несколько раз.

**Почему Deduplication:**
- **Качество данных:** Удалите дублирующиеся записи которые вызывают неправильные результаты
- **Производительность:** Обработайте каждый уникальный элемент только один раз, уменьшая computational overhead
- **Эффективность памяти:** Избежьте хранения redundant данных в результатах
- **Корректность:** Предотвратите double-counting в агрегациях и отчётах

**Production Pattern:**
\`\`\`go
// Удаление дублирующихся user ID из логов
func GetUniqueActiveUsers(logs []LogEntry) []string {
    userIDs := make([]string, len(logs))
    for i, log := range logs {
        userIDs[i] = log.UserID
    }
    return Unique(userIDs)
}

// Дедубликация результатов поиска
func SearchWithUniqueResults(query string) []Document {
    allResults := database.Search(query)

    var ids []int64
    for _, doc := range allResults {
        ids = append(ids, doc.ID)
    }

    uniqueIDs := Unique(ids)
    result := make([]Document, len(uniqueIDs))
    for i, id := range uniqueIDs {
        result[i] = database.GetDocument(id)
    }
    return result
}

// Удаление дублирующихся email уведомлений
func SendUniqueNotifications(recipients []string) error {
    unique := Unique(recipients)

    for _, email := range unique {
        if err := sendEmail(email); err != nil {
            return err
        }
    }
    return nil
}

// Дедубликация элементов корзины
type CartItem struct {
    ProductID string
    Quantity  int
}

func MergeCartItems(items []CartItem) []CartItem {
    seen := make(map[string]int)

    for _, item := range items {
        seen[item.ProductID] += item.Quantity
    }

    result := make([]CartItem, 0, len(seen))
    for productID, quantity := range seen {
        result = append(result, CartItem{productID, quantity})
    }
    return result
}

// Удаление дублирующихся уровней логов для фильтрации
func GetUniqueSeverityLevels(logs []LogEntry) []string {
    levels := make([]string, len(logs))
    for i, log := range logs {
        levels[i] = log.Severity
    }
    return Unique(levels)
}

// Дедубликация путей файлов в артефактах сборки
func GetUniqueSourceFiles(compileErrors []CompileError) []string {
    files := make([]string, len(compileErrors))
    for i, err := range compileErrors {
        files[i] = err.FilePath
    }
    return Unique(files)
}

// Удаление дублирующихся permission scopes
func GetRequiredPermissions(requests []PermissionRequest) []string {
    scopes := make([]string, len(requests))
    for i, req := range requests {
        scopes[i] = req.Scope
    }
    return Unique(scopes)
}

// Дедубликация тегов из пользовательского ввода
func NormalizeTags(input []string) []string {
    // Преобразование в нижний регистр и дедубликация
    normalized := make([]string, len(input))
    for i, tag := range input {
        normalized[i] = strings.ToLower(tag)
    }
    return Unique(normalized)
}

// Удаление дублирующихся хостов из service discovery
func GetHealthyHosts(endpoints []ServiceEndpoint) []string {
    hosts := make([]string, 0, len(endpoints))
    for _, ep := range endpoints {
        if ep.IsHealthy {
            hosts = append(hosts, ep.Host)
        }
    }
    return Unique(hosts)
}
\`\`\`

**Практические преимущества:**
- **ETL Pipelines:** Дедубликация записей из нескольких источников
- **Результаты поиска:** Удаление дублирующихся документов в поисковых системах
- **Аналитика:** Точный подсчёт уникальных пользователей, продуктов или событий
- **Импорт данных:** Очистка импортированных данных перед вставкой в БД
- **API Ответы:** Удаление дублирующихся результатов перед отправкой клиентам

**Частые случаи использования:**
- Дедубликация результатов database запросов
- Удаление дублирующихся элементов из выбора пользователя
- Очистка объединённых datasets из нескольких источников
- Создание уникальных списков для dropdown меню
- Подсчёт distinct значений в аналитике

Без Unique дублирующиеся данные могли бы исказить результаты анализа, вызвать неправильные агрегации и привести к неэффективной обработке избыточных записей.`,
      solutionCode: `package datastructures

func Unique(in []int) []int {
	if len(in) == 0 {                                       // Обработка пустого слайса
		return nil                                      // Вернуть nil для пустого ввода
	}
	seen := make(map[int]struct{}, len(in))                  // Создать set seen с capacity
	result := make([]int, 0, len(in))                        // Предварительно выделить result слайс
	for _, value := range in {                              // Итерация по входному слайсу
		if _, ok := seen[value]; ok {                  // Проверить было ли значение уже встречено
			continue                                // Пропустить дубликат
		}
		seen[value] = struct{}{}                        // Пометить значение как встреченное
		result = append(result, value)                  // Добавить уникальное значение в результат
	}
	return result                                           // Вернуть дедубликированный слайс
}`,
    },
    uz: {
      title: "Slaysdan takroriylarni olib tashlash",
      description: `Birinchi paydo bo'lish tartibini saqlayotganda takroriy elementlarni o'chiradigan **Unique** ni amalga oshiring.

**Talablar:**
1. \`Unique(in []int) []int\` funksiyasini yarating
2. Bo'sh slayslarni ishlang (bo'sh slayz yoki nil qaytaring)
3. Barcha takroriy elementlarni o'chiring
4. Birinchi paydo bo'lish tartibini saqlang
5. Seen elementlarini kuzatish uchun set-o'xshash strukturadan foydalaning
6. Samaradorlik uchun natija slaysini oldindan ajratib qo'ying
7. Dedulikatsiyalangan slaysni qaytaring

**Misol:**
\`\`\`go
result := Unique([]int{1, 2, 2, 3, 1, 4, 3})
// result = []int{1, 2, 3, 4}

result2 := Unique([]string{"apple", "banana", "apple", "cherry"})
// result2 = []string{"apple", "banana", "cherry"}

result3 := Unique([]int{})
// result3 = [] yoki nil
\`\`\`

**Cheklovlar:**
- Birinchi paydo bo'lish tartibini saqlashi kerak
- Har qanday comparable tipi bilan ishlashi kerak
- Kiritish slaysni o'zgartirishmasligi kerak
- Seen elementlarini samarali kuzatishi kerak
- Bo'sh kiritish uchun nil qaytarishi kerak`,
      hint1: `Struct{} qiymatlari bilan mapdan foydalanib allaqachon ko'rgan elementlarni kuzatib, set-o'xshash xatti-harakatni saqlang.`,
      hint2: `Kiritish slaysdan bir marta o'ting, natijaga har bir elementni qo'shishdan oldin mapni tekshiring.`,
      whyItMatters: `Unique ma'lumotlardan takroriyliklarni o'chirishga kerak bo'lgan tartibni saqlayotganda, data qayta ishlash pipeline larida umumiy operatsiya bir yozuvni bir necha marta qayta ishlashning oldini oladi.

**Nima uchun Deduplication:**
- **Ma'lumot sifati:** Noto'g'ri natijalar sababchi bo'lgan takroriy yozuvlarni o'chiring
- **Samaradorlik:** Har bir noyob elementni faqat bir marta qayta islang, hisoblash overhead ni kamaytiring
- **Xotira samaradorligi:** Natijalar ichida redundant ma'lumotlarni saqlashdan qochng
- **To'g'rilik:** Agregatsiya va hisobotlarda double-counting ning oldini oling

**Production Pattern:**
\`\`\`go
// Loglardan takroriy user ID larni o'chirish
func GetUniqueActiveUsers(logs []LogEntry) []string {
    userIDs := make([]string, len(logs))
    for i, log := range logs {
        userIDs[i] = log.UserID
    }
    return Unique(userIDs)
}

// Qidiruv natijalarini deduplikatsiya qilish
func SearchWithUniqueResults(query string) []Document {
    allResults := database.Search(query)

    var ids []int64
    for _, doc := range allResults {
        ids = append(ids, doc.ID)
    }

    uniqueIDs := Unique(ids)
    result := make([]Document, len(uniqueIDs))
    for i, id := range uniqueIDs {
        result[i] = database.GetDocument(id)
    }
    return result
}

// Takroriy email bildirishnomalarini o'chirish
func SendUniqueNotifications(recipients []string) error {
    unique := Unique(recipients)

    for _, email := range unique {
        if err := sendEmail(email); err != nil {
            return err
        }
    }
    return nil
}

// Savat elementlarini deduplikatsiya qilish
type CartItem struct {
    ProductID string
    Quantity  int
}

func MergeCartItems(items []CartItem) []CartItem {
    seen := make(map[string]int)

    for _, item := range items {
        seen[item.ProductID] += item.Quantity
    }

    result := make([]CartItem, 0, len(seen))
    for productID, quantity := range seen {
        result = append(result, CartItem{productID, quantity})
    }
    return result
}

// Filtrlash uchun takroriy log darajalarini o'chirish
func GetUniqueSeverityLevels(logs []LogEntry) []string {
    levels := make([]string, len(logs))
    for i, log := range logs {
        levels[i] = log.Severity
    }
    return Unique(levels)
}

// Build artifaktlarda fayl yo'llarini deduplikatsiya qilish
func GetUniqueSourceFiles(compileErrors []CompileError) []string {
    files := make([]string, len(compileErrors))
    for i, err := range compileErrors {
        files[i] = err.FilePath
    }
    return Unique(files)
}

// Takroriy permission scoplarni o'chirish
func GetRequiredPermissions(requests []PermissionRequest) []string {
    scopes := make([]string, len(requests))
    for i, req := range requests {
        scopes[i] = req.Scope
    }
    return Unique(scopes)
}

// Foydalanuvchi kiritishidan teglarni deduplikatsiya qilish
func NormalizeTags(input []string) []string {
    // Kichik harflarga o'zgartirish va deduplikatsiya
    normalized := make([]string, len(input))
    for i, tag := range input {
        normalized[i] = strings.ToLower(tag)
    }
    return Unique(normalized)
}

// Service discovery dan takroriy hostlarni o'chirish
func GetHealthyHosts(endpoints []ServiceEndpoint) []string {
    hosts := make([]string, 0, len(endpoints))
    for _, ep := range endpoints {
        if ep.IsHealthy {
            hosts = append(hosts, ep.Host)
        }
    }
    return Unique(hosts)
}
\`\`\`

**Amaliy afzalliklar:**
- **ETL Pipelines:** Bir necha manbalardan yozuvlarni deduplikatsiya qilish
- **Qidiruv natijalari:** Qidiruv tizimlarida takroriy hujjatlarni o'chirish
- **Analitika:** Noyob foydalanuvchilar, mahsulotlar yoki hodisalarni aniq hisoblash
- **Ma'lumot import:** DBga qo'yishdan oldin import qilingan ma'lumotlarni tozalash
- **API javoblar:** Mijozlarga yuborishdan oldin takroriy natijalarni o'chirish

**Umumiy foydalanish holatlari:**
- Database so'rov natijalarini deduplikatsiya qilish
- Foydalanuvchi tanlovidan takroriy elementlarni o'chirish
- Bir necha manbadan birlashtirilgan datasetlarni tozalash
- Dropdown menyular uchun noyob ro'yxatlar yaratish
- Analitikada distinct qiymatlarni hisoblash

Unique siz, takroriy ma'lumotlar tahlil natijalarini buzishi, noto'g'ri agregatsiyalarga olib kelishi va ortiqcha yozuvlarni samarasiz qayta ishlashga sabab bo'lishi mumkin.`,
      solutionCode: `package datastructures

func Unique(in []int) []int {
	if len(in) == 0 {                                       // Bo'sh slaysni ishlash
		return nil                                      // Bo'sh kirish uchun nil qaytarish
	}
	seen := make(map[int]struct{}, len(in))                  // Capacity bilan seen set yaratish
	result := make([]int, 0, len(in))                        // Natija slaysini oldindan ajratish
	for _, value := range in {                              // Kirish slayi bo'ylab iteratsiya
		if _, ok := seen[value]; ok {                  // Qiymat allaqachon ko'rilganligini tekshirish
			continue                                // Takroriyni o'tkazib yuborish
		}
		seen[value] = struct{}{}                        // Qiymatni ko'rilgan deb belgilash
		result = append(result, value)                  // Noyob qiymatni natijaga qo'shish
	}
	return result                                           // Dedulikatsiyalangan slaysni qaytarish
}`,
    },
  },
};

export default task;
