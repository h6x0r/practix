import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-fundamentals-safe-delete',
	title: 'Safe Map Key Deletion',
	difficulty: 'easy',	tags: ['go', 'data-structures', 'maps/slices/strings', 'generics'],
	estimatedTime: '15-20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **SafeDelete** that returns a copy of a map without specified keys, leaving the original unchanged.

**Requirements:**
1. Create function \`SafeDelete[M ~map[K]V, K comparable, V any](m M, keys []K) M\`
2. Handle nil maps (return nil)
3. Create a new map with filtered entries
4. Skip all keys from the keys slice
5. Preserve all other key-value pairs
6. Return the new map without modifying the original
7. Maintain the map type constraint properly

**Example:**
\`\`\`go
original := map[string]int{"a": 1, "b": 2, "c": 3, "d": 4}
result := SafeDelete(original, []string{"b", "d"})
// result = map[string]int{"a": 1, "c": 3}
// original = map[string]int{"a": 1, "b": 2, "c": 3, "d": 4} (unchanged)

result2 := SafeDelete(map[int]string{1: "x", 2: "y"}, []int{1})
// result2 = map[int]string{2: "y"}
\`\`\`

**Constraints:**
- Must not modify the original map
- Must handle nil maps gracefully
- Must support any comparable key type
- Must support any value type
- Must use generics properly`,
	initialCode: `package datastructures

// TODO: Implement SafeDelete
func SafeDelete[M ~map[K]V, K comparable, V any](m M, keys []K) M {
	// TODO: Implement
}`,
	solutionCode: `package datastructures

func SafeDelete[M ~map[K]V, K comparable, V any](m M, keys []K) M {
	if m == nil {                                           // Handle nil maps
		var zero M                                      // Create zero value of map type
		return zero                                     // Return nil
	}
	toDelete := make(map[K]struct{}, len(keys))             // Create deletion lookup map
	for _, k := range keys {                                // Iterate through keys to delete
		toDelete[k] = struct{}{}                       // Mark each key for deletion
	}
	cloned := make(map[K]V, len(m))                         // Pre-allocate cloned map
	for k, v := range m {                                   // Iterate through original map
		if _, skip := toDelete[k]; skip {              // Check if key should be deleted
			continue                                // Skip this key
		}
		cloned[k] = v                                   // Copy key-value pair
	}
	return M(cloned)                                        // Convert back to original map type
}`,
	testCode: `package datastructures

import (
	"reflect"
	"testing"
)

func Test1(t *testing.T) {
	// Basic deletion
	original := map[string]int{"a": 1, "b": 2, "c": 3, "d": 4}
	result := SafeDelete(original, []string{"b", "d"})
	expected := map[string]int{"a": 1, "c": 3}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test2(t *testing.T) {
	// Original unchanged
	original := map[string]int{"a": 1, "b": 2}
	_ = SafeDelete(original, []string{"a"})
	if len(original) != 2 || original["a"] != 1 {
		t.Errorf("original was modified: %v", original)
	}
}

func Test3(t *testing.T) {
	// Nil map
	var m map[string]int = nil
	result := SafeDelete(m, []string{"a"})
	if result != nil {
		t.Errorf("expected nil, got %v", result)
	}
}

func Test4(t *testing.T) {
	// Empty keys slice
	original := map[string]int{"a": 1, "b": 2}
	result := SafeDelete(original, []string{})
	expected := map[string]int{"a": 1, "b": 2}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test5(t *testing.T) {
	// Delete non-existent key
	original := map[string]int{"a": 1, "b": 2}
	result := SafeDelete(original, []string{"z"})
	expected := map[string]int{"a": 1, "b": 2}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test6(t *testing.T) {
	// Delete all keys
	original := map[string]int{"a": 1, "b": 2}
	result := SafeDelete(original, []string{"a", "b"})
	if len(result) != 0 {
		t.Errorf("expected empty map, got %v", result)
	}
}

func Test7(t *testing.T) {
	// Int keys
	original := map[int]string{1: "x", 2: "y", 3: "z"}
	result := SafeDelete(original, []int{1, 3})
	expected := map[int]string{2: "y"}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test8(t *testing.T) {
	// Single key deletion
	original := map[string]int{"only": 42}
	result := SafeDelete(original, []string{"only"})
	if len(result) != 0 {
		t.Errorf("expected empty map, got %v", result)
	}
}

func Test9(t *testing.T) {
	// Duplicate keys in deletion list
	original := map[string]int{"a": 1, "b": 2, "c": 3}
	result := SafeDelete(original, []string{"a", "a", "a"})
	expected := map[string]int{"b": 2, "c": 3}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("expected %v, got %v", expected, result)
	}
}

func Test10(t *testing.T) {
	// Empty map
	original := map[string]int{}
	result := SafeDelete(original, []string{"a"})
	if len(result) != 0 {
		t.Errorf("expected empty map, got %v", result)
	}
}`,
	hint1: `Create a lookup map with keys to delete (using struct{} as value), then build new map skipping those keys.`,
			hint2: `Use a set-like map with struct{} values to efficiently check which keys should be skipped during copying.`,
			whyItMatters: `SafeDelete demonstrates immutable-style operations in Go, preventing accidental mutations that can cause bugs in multi-threaded or shared data environments.

**Why Safe Deletion:**
- **Data Integrity:** Original data is never modified, preventing cascading bugs
- **Concurrent Safety:** Multiple goroutines can safely use the original map
- **Functional Style:** Enables functional programming patterns in Go
- **Debugging:** Clear intent that the operation doesn't have side effects

**Production Pattern:**
\`\`\`go
// User preferences without sensitive keys
func GetUserPreferencesPublic(userID string) map[string]string {
    preferences := database.GetUserPreferences(userID)

    // Remove internal-only keys
    return SafeDelete(preferences, []string{
        "internal_hash",
        "secret_token",
        "admin_notes",
    })
}

// Filter sensitive fields from API response
func SanitizeAPIResponse(data map[string]interface{}) map[string]interface{} {
    sensitiveFields := []string{"password_hash", "api_key", "credit_card"}
    return SafeDelete(data, sensitiveFields)
}

// Remove cache invalidation keys from session
func CleanupSessionCache(session map[string]interface{}) map[string]interface{} {
    tempKeys := []string{"_temp_data", "_working_cache", "_internal_state"}
    return SafeDelete(session, tempKeys)
}

// Configuration without environment-specific keys
func GetBaseConfig(fullConfig map[string]string) map[string]string {
    envSpecific := []string{"DB_HOST", "API_KEY", "GOOGLE_CREDS"}
    return SafeDelete(fullConfig, envSpecific)
}

// Safe metadata removal without affecting original
func RemoveMetadata(doc map[string]interface{}) map[string]interface{} {
    metadataKeys := []string{"_id", "_created_at", "_updated_by"}
    clean := SafeDelete(doc, metadataKeys)
    return clean
}

// Cache invalidation pattern
type CacheManager struct {
    cache map[string]*CachedValue
}

func (cm *CacheManager) InvalidateKeysWithoutMutation(keys []string) map[string]*CachedValue {
    // Don't modify internal cache, return view without certain keys
    return SafeDelete(cm.cache, keys)
}

// Testing helper - create test data without mutable state
func CreateTestData() map[string]interface{} {
    full := map[string]interface{}{
        "user": "alice",
        "role": "admin",
        "debug_token": "xxx",
        "trace_id": "yyy",
    }
    return SafeDelete(full, []string{"debug_token", "trace_id"})
}

// Feature flag cleanup - remove experimental flags
func GetStableFeatures(allFeatures map[string]bool) map[string]bool {
    experimental := []string{"BETA_FEATURE_X", "ALPHA_FEATURE_Y"}
    return SafeDelete(allFeatures, experimental)
}
\`\`\`

**Real-World Benefits:**
- **Thread Safety:** Immutable operations reduce data race conditions
- **API Filtering:** Remove sensitive fields before returning to clients
- **Configuration:** Exclude environment-specific settings safely
- **Testing:** Create clean test data without side effects
- **Audit Trail:** Original data stays intact for logging

**Common Use Cases:**
- Removing sensitive fields before serialization
- Filtering API responses
- Creating read-only views of configuration
- Excluding internal fields from public APIs

Without SafeDelete, developers might accidentally modify shared data structures, leading to subtle bugs that are hard to track in concurrent environments.`,	order: 1,
	translations: {
		ru: {
			title: 'Безопасное удаление ключей из map',
			description: `Реализуйте **SafeDelete**, который возвращает копию map без указанных ключей, оставляя оригинал без изменений.

**Требования:**
1. Создайте функцию \`SafeDelete[M ~map[K]V, K comparable, V any](m M, keys []K) M\`
2. Обработайте nil maps (верните nil)
3. Создайте новый map с отфильтрованными записями
4. Пропустите все ключи из слайса keys
5. Сохраните все остальные пары ключ-значение
6. Верните новый map без изменения оригинала
7. Правильно поддерживайте constraint типа map

**Пример:**
\`\`\`go
original := map[string]int{"a": 1, "b": 2, "c": 3, "d": 4}
result := SafeDelete(original, []string{"b", "d"})
// result = map[string]int{"a": 1, "c": 3}
// original = map[string]int{"a": 1, "b": 2, "c": 3, "d": 4} (не изменён)

result2 := SafeDelete(map[int]string{1: "x", 2: "y"}, []int{1})
// result2 = map[int]string{2: "y"}
\`\`\`

**Ограничения:**
- Не должен модифицировать оригинальный map
- Должен корректно обработать nil maps
- Должен поддерживать любой comparable тип ключа
- Должен поддерживать любой тип значения
- Должен правильно использовать generics`,
			hint1: `Создайте lookup map с ключами для удаления (используя struct{} как значение), затем постройте новый map пропуская те ключи.`,
			hint2: `Используйте set-подобный map со struct{} значениями для эффективной проверки каких ключей пропускать при копировании.`,
			whyItMatters: `SafeDelete демонстрирует операции в стиле immutable в Go, предотвращая случайные мутации которые могут вызвать баги в многопоточных или shared data окружениях.

**Почему Safe Deletion:**
- **Целостность данных:** Оригинальные данные никогда не модифицируются, предотвращая каскадные баги
- **Concurrent Safety:** Несколько горутин могут безопасно использовать оригинальный map
- **Функциональный стиль:** Включает функциональные паттерны программирования в Go
- **Debugging:** Ясное намерение что операция не имеет побочных эффектов

**Production Pattern:**
\`\`\`go
// Публичные настройки пользователя без sensitive ключей
func GetUserPreferencesPublic(userID string) map[string]string {
    preferences := database.GetUserPreferences(userID)

    // Удалить internal-only ключи
    return SafeDelete(preferences, []string{
        "internal_hash",
        "secret_token",
        "admin_notes",
    })
}

// Фильтрация sensitive полей из API ответа
func SanitizeAPIResponse(data map[string]interface{}) map[string]interface{} {
    sensitiveFields := []string{"password_hash", "api_key", "credit_card"}
    return SafeDelete(data, sensitiveFields)
}

// Удаление cache invalidation ключей из сессии
func CleanupSessionCache(session map[string]interface{}) map[string]interface{} {
    tempKeys := []string{"_temp_data", "_working_cache", "_internal_state"}
    return SafeDelete(session, tempKeys)
}

// Конфигурация без environment-specific ключей
func GetBaseConfig(fullConfig map[string]string) map[string]string {
    envSpecific := []string{"DB_HOST", "API_KEY", "GOOGLE_CREDS"}
    return SafeDelete(fullConfig, envSpecific)
}

// Безопасное удаление метаданных без влияния на оригинал
func RemoveMetadata(doc map[string]interface{}) map[string]interface{} {
    metadataKeys := []string{"_id", "_created_at", "_updated_by"}
    clean := SafeDelete(doc, metadataKeys)
    return clean
}

// Паттерн инвалидации кэша
type CacheManager struct {
    cache map[string]*CachedValue
}

func (cm *CacheManager) InvalidateKeysWithoutMutation(keys []string) map[string]*CachedValue {
    // Не модифицировать внутренний кэш, вернуть view без определённых ключей
    return SafeDelete(cm.cache, keys)
}

// Тестовый helper - создание тестовых данных без изменяемого состояния
func CreateTestData() map[string]interface{} {
    full := map[string]interface{}{
        "user": "alice",
        "role": "admin",
        "debug_token": "xxx",
        "trace_id": "yyy",
    }
    return SafeDelete(full, []string{"debug_token", "trace_id"})
}

// Очистка feature flag - удаление экспериментальных флагов
func GetStableFeatures(allFeatures map[string]bool) map[string]bool {
    experimental := []string{"BETA_FEATURE_X", "ALPHA_FEATURE_Y"}
    return SafeDelete(allFeatures, experimental)
}
\`\`\`

**Практические преимущества:**
- **Потокобезопасность:** Immutable операции уменьшают условия гонки данных
- **Фильтрация API:** Удаление sensitive полей перед возвратом клиентам
- **Конфигурация:** Безопасное исключение environment-specific настроек
- **Тестирование:** Создание чистых тестовых данных без побочных эффектов
- **Аудит:** Оригинальные данные остаются нетронутыми для логирования

**Частые случаи использования:**
- Удаление sensitive полей перед сериализацией
- Фильтрация API ответов
- Создание read-only views конфигурации
- Исключение internal полей из public APIs

Без SafeDelete разработчики могут случайно модифицировать shared структуры данных, приводя к трудноотслеживаемым багам в concurrent окружениях.`,
			solutionCode: `package datastructures

func SafeDelete[M ~map[K]V, K comparable, V any](m M, keys []K) M {
	if m == nil {                                           // Обработка nil maps
		var zero M                                      // Создать нулевое значение типа map
		return zero                                     // Вернуть nil
	}
	toDelete := make(map[K]struct{}, len(keys))             // Создать lookup map для удаления
	for _, k := range keys {                                // Итерация по ключам для удаления
		toDelete[k] = struct{}{}                       // Пометить каждый ключ для удаления
	}
	cloned := make(map[K]V, len(m))                         // Предварительно выделить клонированный map
	for k, v := range m {                                   // Итерация по оригинальному map
		if _, skip := toDelete[k]; skip {              // Проверить должен ли ключ быть удалён
			continue                                // Пропустить этот ключ
		}
		cloned[k] = v                                   // Скопировать пару ключ-значение
	}
	return M(cloned)                                        // Преобразовать обратно к исходному типу map
}`
		},
		uz: {
			title: 'Mapdan kalitlarni xavfsiz o\'chirish',
			description: `Belgilangan kalitlarni o'z ichiga olmagan map nusxasini qaytaradigan **SafeDelete** ni amalga oshiring, asl versiyoni o'zgartirilmagan.

**Talablar:**
1. \`SafeDelete[M ~map[K]V, K comparable, V any](m M, keys []K) M\` funksiyasini yarating
2. nil maps ni ishlang (nil qaytaring)
3. Filtrlangan yozuvlari bilan yangi map yarating
4. Keys slaysidagi barcha kalitlarni o'tkazib yuboring
5. Barcha boshqa kalit-qiymat juftliklarini saqlang
6. Asl versiyoni o'zgartirilmagan holda yangi map qaytaring
7. Map tipi constraintni to'g'ri ta'minlang

**Misol:**
\`\`\`go
original := map[string]int{"a": 1, "b": 2, "c": 3, "d": 4}
result := SafeDelete(original, []string{"b", "d"})
// result = map[string]int{"a": 1, "c": 3}
// original = map[string]int{"a": 1, "b": 2, "c": 3, "d": 4} (o'zgartirilmagan)

result2 := SafeDelete(map[int]string{1: "x", 2: "y"}, []int{1})
// result2 = map[int]string{2: "y"}
\`\`\`

**Cheklovlar:**
- Asl mapni o'zgartirishmasligi kerak
- nil maps ni to'g'ri ishlashi kerak
- Har qanday comparable kalit tipini qo'llab-quvvatlashi kerak
- Har qanday qiymat tipini qo'llab-quvvatlashi kerak
- Genericsni to'g'ri foydalanishi kerak`,
			hint1: `O'chirish uchun kalitlari bilan lookup map yarating (struct{} ni qiymat sifatida ishlatib), keyin o'sha kalitlarni o'tkazib yankgi map qurilishi kerak.`,
			hint2: `Set-o'xshash mapni struct{} qiymatlari bilan foydalanib o'chirish kerak bo'lgan kalitlarni nusxalash paytida samarali tekshirish uchun.`,
			whyItMatters: `SafeDelete Go da immutable uslubidagi operatsiyalarni ko'rsatadi, ko'p jadvallik yoki shared data muhitida bugga olib kelishi mumkin bo'lgan tasodifiy mutatsiyalarning oldini oladi.

**Nima uchun Safe Deletion:**
- **Malumot integralligi:** Asl ma'lumotlar hech qachon o'zgartirilmaydi, kaskad buglarning oldini oladi
- **Concurrent Safety:** Ko'p goroutinalar asl mapni xavfsiz ishlatishi mumkin
- **Funktsional uslub:** Go da funktsional programmalashtirish patternlarini yoqadi
- **Debugging:** Operatsiya side effectiga ega emasligini yoki ta'sirini aniq ko'rsatadi

**Production Pattern:**
\`\`\`go
// Sensitive kalitlarsiz foydalanuvchi sozlamalari
func GetUserPreferencesPublic(userID string) map[string]string {
    preferences := database.GetUserPreferences(userID)

    // Internal-only kalitlarni o'chirish
    return SafeDelete(preferences, []string{
        "internal_hash",
        "secret_token",
        "admin_notes",
    })
}

// API javobidan sensitive maydonlarini filtrlash
func SanitizeAPIResponse(data map[string]interface{}) map[string]interface{} {
    sensitiveFields := []string{"password_hash", "api_key", "credit_card"}
    return SafeDelete(data, sensitiveFields)
}

// Sessiyadan cache invalidation kalitlarini o'chirish
func CleanupSessionCache(session map[string]interface{}) map[string]interface{} {
    tempKeys := []string{"_temp_data", "_working_cache", "_internal_state"}
    return SafeDelete(session, tempKeys)
}

// Environment-specific kalitlarsiz konfiguratsiya
func GetBaseConfig(fullConfig map[string]string) map[string]string {
    envSpecific := []string{"DB_HOST", "API_KEY", "GOOGLE_CREDS"}
    return SafeDelete(fullConfig, envSpecific)
}

// Asl versiyasiga ta'sir qilmasdan metadatani xavfsiz o'chirish
func RemoveMetadata(doc map[string]interface{}) map[string]interface{} {
    metadataKeys := []string{"_id", "_created_at", "_updated_by"}
    clean := SafeDelete(doc, metadataKeys)
    return clean
}

// Cache invalidation patterni
type CacheManager struct {
    cache map[string]*CachedValue
}

func (cm *CacheManager) InvalidateKeysWithoutMutation(keys []string) map[string]*CachedValue {
    // Ichki cache ni o'zgartirmaslik, ma'lum kalitlarsiz view qaytarish
    return SafeDelete(cm.cache, keys)
}

// Test helper - o'zgaruvchan holatisiz test ma'lumotlarini yaratish
func CreateTestData() map[string]interface{} {
    full := map[string]interface{}{
        "user": "alice",
        "role": "admin",
        "debug_token": "xxx",
        "trace_id": "yyy",
    }
    return SafeDelete(full, []string{"debug_token", "trace_id"})
}

// Feature flag tozalash - eksperimental flaglarni o'chirish
func GetStableFeatures(allFeatures map[string]bool) map[string]bool {
    experimental := []string{"BETA_FEATURE_X", "ALPHA_FEATURE_Y"}
    return SafeDelete(allFeatures, experimental)
}
\`\`\`

**Amaliy afzalliklar:**
- **Thread xavfsizligi:** Immutable operatsiyalar ma'lumot musobaqasi shartlarini kamaytiradi
- **API filtrlash:** Mijozlarga qaytarishdan oldin sensitive maydonlarni o'chirish
- **Konfiguratsiya:** Environment-specific sozlamalarni xavfsiz chiqarib tashlash
- **Test qilish:** Side effectlarsiz toza test ma'lumotlarini yaratish
- **Audit:** Loglash uchun asl ma'lumotlar o'zgarishsiz qoladi

**Umumiy foydalanish holatlari:**
- Serialization dan oldin sensitive maydonlarni o'chirish
- API javoblarini filtrlash
- Config ning read-only viewlarini yaratish
- Public APIs dan internal maydonlarni chiqarib tashlash

SafeDelete siz, dasturcholar tasodifan shared ma'lumot strukturalarini o'zgartirishi mumkin, bu concurrent muhitda kuzatish qiyin buglar ga olib keladi.`,
			solutionCode: `package datastructures

func SafeDelete[M ~map[K]V, K comparable, V any](m M, keys []K) M {
	if m == nil {                                           // Nil maps ni ishlash
		var zero M                                      // Map tipining nol qiymatini yaratish
		return zero                                     // Nil qaytarish
	}
	toDelete := make(map[K]struct{}, len(keys))             // O'chirish uchun lookup map yaratish
	for _, k := range keys {                                // O'chirish uchun kalitlarni iteratsiya qilish
		toDelete[k] = struct{}{}                       // Har bir kalitni o'chirish uchun belgilash
	}
	cloned := make(map[K]V, len(m))                         // Klonlangan mapni oldindan ajratish
	for k, v := range m {                                   // Asl map bo'ylab iteratsiya qilish
		if _, skip := toDelete[k]; skip {              // Kalit o'chirilishi kerakligini tekshirish
			continue                                // Ushbu kalitni o'tkazib yuborish
		}
		cloned[k] = v                                   // Kalit-qiymat juftligini nusxalash
	}
	return M(cloned)                                        // Asl map tipiga qaytarish
}`
		}
	}
};

export default task;
