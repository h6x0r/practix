import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-ttlcache-delete',
	title: 'Delete Method',
	difficulty: 'medium',	tags: ['go', 'cache', 'deletion', 'sync'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement thread-safe **Delete** method to remove entries from cache.

**Requirements:**
1. Implement \`Delete(key string) bool\` method on TTLCache
2. Return \`false\` if cache is nil or key doesn't exist
3. Use exclusive lock (Lock/Unlock) for map deletion
4. Check if key exists before deletion
5. Return \`true\` if entry was found and deleted (even if expired)
6. Return \`false\` if key was never in the map

**Example:**
\`\`\`go
cache := NewTTLCache(5 * time.Second)
cache.Set("key", "value")

deleted := cache.Delete("key")       // true (key existed)
deleted = cache.Delete("key")        // false (already deleted)
deleted = cache.Delete("nonexistent") // false (never existed)
\`\`\`

**Constraints:**
- Must use Lock (not RLock) for map mutation
- Must check existence before deletion
- Return true even if entry was expired (it existed in map)`,
	initialCode: `package cache

import (
	"sync"
	"time"
)

type entry struct {
	v   any
	exp time.Time
}

type TTLCache struct {
	mu  sync.RWMutex
	m   map[string]entry
	ttl time.Duration
}

func NewTTLCache(ttl time.Duration) *TTLCache {
	return &TTLCache{m: make(map[string]entry), ttl: ttl}
}

func (c *TTLCache) Set(key string, v any) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	expire := time.Time{}
	if c.ttl > 0 {
		expire = time.Now().Add(c.ttl)
	}
	c.m[key] = entry{v: v, exp: expire}
}

func (c *TTLCache) Get(key string) (interface{}, bool) {
	if c == nil {
		return nil, false
	}
	c.mu.RLock()
	ent, ok := c.m[key]
	c.mu.RUnlock()
	if !ok {
		return nil, false
	}
	if !ent.exp.IsZero() && time.Now().After(ent.exp) {
		c.mu.Lock()
		defer c.mu.Unlock()
		if entCur, still := c.m[key]; still && entCur.exp == ent.exp {
			delete(c.m, key)
		}
		return nil, false
	}
	return ent.v, true
}

func (c *TTLCache) cleanupNow(now time.Time) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for k, ent := range c.m {
		if !ent.exp.IsZero() && !now.Before(ent.exp) {
			delete(c.m, k)
		}
	}
}

// TODO: Implement Delete method
// 1. Check if c is nil, return false
// 2. Use Lock for exclusive access to map
// 3. Check if key exists in map
// 4. If exists, delete it and return true
// 5. If doesn't exist, return false
func (c *TTLCache) Delete(key string) bool {
	// TODO: Implement
}`,
	solutionCode: `package cache

import (
	"sync"
	"time"
)

type entry struct {
	v   any
	exp time.Time
}

type TTLCache struct {
	mu  sync.RWMutex
	m   map[string]entry
	ttl time.Duration
}

func NewTTLCache(ttl time.Duration) *TTLCache {
	return &TTLCache{m: make(map[string]entry), ttl: ttl}
}

func (c *TTLCache) Set(key string, v any) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	expire := time.Time{}
	if c.ttl > 0 {
		expire = time.Now().Add(c.ttl)
	}
	c.m[key] = entry{v: v, exp: expire}
}

func (c *TTLCache) Get(key string) (interface{}, bool) {
	if c == nil {
		return nil, false
	}
	c.mu.RLock()
	ent, ok := c.m[key]
	c.mu.RUnlock()
	if !ok {
		return nil, false
	}
	if !ent.exp.IsZero() && time.Now().After(ent.exp) {
		c.mu.Lock()
		defer c.mu.Unlock()
		if entCur, still := c.m[key]; still && entCur.exp == ent.exp {
			delete(c.m, key)
		}
		return nil, false
	}
	return ent.v, true
}

func (c *TTLCache) cleanupNow(now time.Time) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for k, ent := range c.m {
		if !ent.exp.IsZero() && !now.Before(ent.exp) {
			delete(c.m, k)
		}
	}
}

func (c *TTLCache) Delete(key string) bool {
	if c == nil {	// Nil cache never contains keys
		return false
	}
	c.mu.Lock()	// Lock to mutate underlying map
	defer c.mu.Unlock()	// Ensure unlock happens after deletion attempt
	if _, ok := c.m[key]; ok {	// Detect whether key is present before deletion
		delete(c.m, key)	// Remove entry regardless of expiration state
		return true	// Signal that entry existed and was removed
	}
	return false	// Report absence when key was not stored
}`,
	testCode: `package cache

import (
	"testing"
	"time"
)

func TestDelete_BasicDelete(t *testing.T) {
	cache := NewTTLCache(time.Second)
	cache.Set("key1", "value1")
	result := cache.Delete("key1")
	if !result {
		t.Error("expected Delete to return true for existing key")
	}
	if _, ok := cache.m["key1"]; ok {
		t.Error("expected key1 to be deleted")
	}
}

func TestDelete_NonExistent(t *testing.T) {
	cache := NewTTLCache(time.Second)
	result := cache.Delete("nonexistent")
	if result {
		t.Error("expected Delete to return false for non-existent key")
	}
}

func TestDelete_NilCache(t *testing.T) {
	var cache *TTLCache
	result := cache.Delete("key1")
	if result {
		t.Error("expected Delete to return false for nil cache")
	}
}

func TestDelete_ExpiredKey(t *testing.T) {
	cache := NewTTLCache(50 * time.Millisecond)
	cache.Set("key1", "value1")
	time.Sleep(100 * time.Millisecond)
	result := cache.Delete("key1")
	if !result {
		t.Error("expected Delete to return true even for expired key")
	}
}

func TestDelete_MultipleKeys(t *testing.T) {
	cache := NewTTLCache(time.Second)
	cache.Set("key1", "value1")
	cache.Set("key2", "value2")
	cache.Delete("key1")
	if len(cache.m) != 1 {
		t.Errorf("expected 1 entry after delete, got %d", len(cache.m))
	}
	if _, ok := cache.m["key2"]; !ok {
		t.Error("expected key2 to remain")
	}
}

func TestDelete_EmptyCache(t *testing.T) {
	cache := NewTTLCache(time.Second)
	result := cache.Delete("key1")
	if result {
		t.Error("expected Delete to return false for empty cache")
	}
}

func TestDelete_DeleteTwice(t *testing.T) {
	cache := NewTTLCache(time.Second)
	cache.Set("key1", "value1")
	result1 := cache.Delete("key1")
	result2 := cache.Delete("key1")
	if !result1 {
		t.Error("expected first Delete to return true")
	}
	if result2 {
		t.Error("expected second Delete to return false")
	}
}

func TestDelete_EmptyKey(t *testing.T) {
	cache := NewTTLCache(time.Second)
	cache.Set("", "empty key value")
	result := cache.Delete("")
	if !result {
		t.Error("expected Delete to return true for empty key")
	}
}

func TestDelete_AllKeys(t *testing.T) {
	cache := NewTTLCache(time.Second)
	cache.Set("key1", "value1")
	cache.Set("key2", "value2")
	cache.Set("key3", "value3")
	cache.Delete("key1")
	cache.Delete("key2")
	cache.Delete("key3")
	if len(cache.m) != 0 {
		t.Errorf("expected empty cache after deleting all keys, got %d", len(cache.m))
	}
}

func TestDelete_ReturnValue(t *testing.T) {
	cache := NewTTLCache(time.Second)
	cache.Set("key1", "value1")
	if !cache.Delete("key1") {
		t.Error("expected true for existing key")
	}
	if cache.Delete("key1") {
		t.Error("expected false for already deleted key")
	}
}
`,
			hint1: `Check if key exists with _, ok := c.m[key] before deletion. Delete even if expired.`,
			hint2: `Use Lock (not RLock) because delete() mutates the map structure.`,
			whyItMatters: `Manual deletion enables cache invalidation patterns critical for maintaining data consistency.

**Why Manual Delete:**
- **Immediate invalidation:** Don't wait for TTL when data changes
- **Cache coherence:** Update/delete when source data changes
- **Memory control:** Remove large entries before expiration
- **Testing:** Clear cache between test cases

**Production Use Cases:**

1. **Write-Through Cache Invalidation:**
\`\`\`go
func UpdateUser(id string, user *User) error {
    if err := db.Update(id, user); err != nil {
        return err
    }
    userCache.Delete(id)  // Invalidate stale cache entry
    return nil
}
\`\`\`

2. **Cache-Aside Pattern:**
\`\`\`go
func InvalidateUser(id string) {
    userCache.Delete(id)  // Explicit invalidation
    // Next Get will fetch fresh data from DB
}
\`\`\`

3. **Logout/Session Invalidation:**
\`\`\`go
func Logout(sessionID string) {
    sessionCache.Delete(sessionID)  // Immediate revocation
    // User can't use expired session even if TTL hasn't expired
}
\`\`\`

4. **Bulk Invalidation:**
\`\`\`go
func InvalidateUserCaches(userIDs []string) {
    for _, id := range userIDs {
        if userCache.Delete(id) {
            log.Printf("Invalidated cache for user %s", id)
        }
    }
}
\`\`\`

**Why Return Bool:**
\`\`\`go
// Caller can distinguish between:
if cache.Delete(key) {
    log.Info("Cache invalidated")  // Key existed
} else {
    log.Warn("Key not in cache")   // Already deleted or never existed
}

// Useful for metrics:
deletedCount := 0
for _, key := range keys {
    if cache.Delete(key) {
        deletedCount++
    }
}
log.Printf("Invalidated %d cache entries", deletedCount)
\`\`\`

**Why Delete Expired Entries:**
\`\`\`go
// Return true even if expired because:
// 1. Entry physically exists in map (consumes memory)
// 2. Caller wants to know if key was ever present
// 3. Consistent with standard map behavior

if _, ok := c.m[key]; ok {  // Exists in map?
    delete(c.m, key)        // Remove it
    return true             // Report existence
}
\`\`\`

**Common Pattern - Write-Through Cache:**
\`\`\`go
type UserService struct {
    db    *Database
    cache *TTLCache
}

func (s *UserService) GetUser(id string) (*User, error) {
    // Try cache first
    if cached, ok := s.cache.Get(id); ok {
        return cached.(*User), nil
    }

    // Cache miss: fetch from DB
    user, err := s.db.GetUser(id)
    if err != nil {
        return nil, err
    }

    s.cache.Set(id, user)
    return user, nil
}

func (s *UserService) UpdateUser(id string, user *User) error {
    if err := s.db.UpdateUser(id, user); err != nil {
        return err
    }

    // Invalidate cache on write
    s.cache.Delete(id)  // Force fresh read next time
    return nil
}
\`\`\`

**Real-World Examples:**
- **Redis:** DEL command for explicit key removal
- **Memcached:** DELETE command regardless of expiration
- **HTTP caches:** Cache-Control: no-cache forces revalidation
- **CDN invalidation:** Purge API to remove cached assets

**Performance Considerations:**
- **O(1) operation:** Hash map deletion is constant time
- **Lock contention:** Brief exclusive lock, minimal impact
- **Memory:** Immediately frees entry memory, helpful for large values

Without manual deletion, you'd have to wait for TTL expiration even when you know data is stale, leading to consistency bugs in production.`,	order: 4,
	translations: {
		ru: {
			title: 'Удаление элемента из кэша',
			solutionCode: `package cache

import (
	"sync"
	"time"
)

type entry struct {
	v   any
	exp time.Time
}

type TTLCache struct {
	mu  sync.RWMutex
	m   map[string]entry
	ttl time.Duration
}

func (c *TTLCache) Delete(key string) bool {
	if c == nil {	// Nil cache никогда не содержит ключей
		return false
	}
	c.mu.Lock()	// Lock для изменения underlying map
	defer c.mu.Unlock()	// Гарантируем unlock после попытки удаления
	if _, ok := c.m[key]; ok {	// Определяем присутствует ли ключ перед удалением
		delete(c.m, key)	// Удаляем запись независимо от состояния истечения
		return true	// Сигнализируем что запись существовала и была удалена
	}
	return false	// Сообщаем отсутствие когда ключ не был сохранён
}`,
			description: `Реализуйте потокобезопасный метод **Delete** для удаления записей из кеша.

**Требования:**
1. Реализуйте метод \`Delete(key string) bool\` для TTLCache
2. Верните \`false\` если cache nil или ключ не существует
3. Используйте эксклюзивную блокировку для удаления из map
4. Проверьте существование ключа перед удалением
5. Верните \`true\` если запись найдена и удалена (даже если истекла)
6. Верните \`false\` если ключа никогда не было в map

**Пример:**
\`\`\`go
cache := NewTTLCache(5 * time.Second)
cache.Set("key", "value")

deleted := cache.Delete("key")       // true (ключ существовал)
deleted = cache.Delete("key")        // false (уже удалён)
deleted = cache.Delete("nonexistent") // false (никогда не существовал)
\`\`\`

**Ограничения:**
- Используйте Lock (не RLock) для изменения map
- Проверяйте существование перед удалением
- Возвращайте true даже если запись истекла (существовала в map)`,
			hint1: `Проверьте существование ключа через _, ok := c.m[key] перед удалением. Удаляйте даже если истекло.`,
			hint2: `Используйте Lock (не RLock), потому что delete() изменяет структуру map.`,
			whyItMatters: `Ручное удаление позволяет использовать паттерны инвалидации кеша, критичные для поддержания согласованности данных.

**Почему ручное Delete:**
- **Немедленная инвалидация:** Не ждать TTL когда данные меняются
- **Согласованность кеша:** Обновить/удалить когда исходные данные изменились
- **Контроль памяти:** Удалить большие записи до истечения
- **Тестирование:** Очистить кеш между тестовыми случаями

**Случаи использования в продакшене:**

1. **Инвалидация Write-Through кеша:**
\`\`\`go
func UpdateUser(id string, user *User) error {
    if err := db.Update(id, user); err != nil {
        return err
    }
    userCache.Delete(id)  // Инвалидировать устаревшую запись кеша
    return nil
}
\`\`\`

2. **Паттерн Cache-Aside:**
\`\`\`go
func InvalidateUser(id string) {
    userCache.Delete(id)  // Явная инвалидация
    // Следующий Get получит свежие данные из БД
}
\`\`\`

3. **Инвалидация Logout/Session:**
\`\`\`go
func Logout(sessionID string) {
    sessionCache.Delete(sessionID)  // Немедленный отзыв
    // Пользователь не может использовать истёкшую сессию даже если TTL не истёк
}
\`\`\`

4. **Массовая инвалидация:**
\`\`\`go
func InvalidateUserCaches(userIDs []string) {
    for _, id := range userIDs {
        if userCache.Delete(id) {
            log.Printf("Invalidated cache for user %s", id)
        }
    }
}
\`\`\`

**Почему возвращать Bool:**
\`\`\`go
// Вызывающий может различить:
if cache.Delete(key) {
    log.Info("Cache invalidated")  // Ключ существовал
} else {
    log.Warn("Key not in cache")   // Уже удалён или никогда не существовал
}

// Полезно для метрик:
deletedCount := 0
for _, key := range keys {
    if cache.Delete(key) {
        deletedCount++
    }
}
log.Printf("Invalidated %d cache entries", deletedCount)
\`\`\`

**Почему удалять истёкшие записи:**
\`\`\`go
// Возвращать true даже если истекло потому что:
// 1. Запись физически существует в map (потребляет память)
// 2. Вызывающему нужно знать была ли запись когда-либо
// 3. Согласуется со стандартным поведением map

if _, ok := c.m[key]; ok {  // Существует в map?
    delete(c.m, key)        // Удалить
    return true             // Сообщить о существовании
}
\`\`\`

**Общий паттерн - Write-Through кеш:**
\`\`\`go
type UserService struct {
    db    *Database
    cache *TTLCache
}

func (s *UserService) GetUser(id string) (*User, error) {
    // Сначала попробовать кеш
    if cached, ok := s.cache.Get(id); ok {
        return cached.(*User), nil
    }

    // Промах кеша: получить из БД
    user, err := s.db.GetUser(id)
    if err != nil {
        return nil, err
    }

    s.cache.Set(id, user)
    return user, nil
}

func (s *UserService) UpdateUser(id string, user *User) error {
    if err := s.db.UpdateUser(id, user); err != nil {
        return err
    }

    // Инвалидировать кеш при записи
    s.cache.Delete(id)  // Принудительно свежее чтение в следующий раз
    return nil
}
\`\`\`

**Примеры из реального мира:**
- **Redis:** Команда DEL для явного удаления ключей
- **Memcached:** Команда DELETE независимо от истечения
- **HTTP кеши:** Cache-Control: no-cache принудительно ревалидирует
- **CDN инвалидация:** Purge API для удаления закешированных ресурсов

**Соображения производительности:**
- **O(1) операция:** Удаление из хеш-map за константное время
- **Конкуренция блокировок:** Краткая эксклюзивная блокировка, минимальное влияние
- **Память:** Немедленно освобождает память записи, полезно для больших значений

**Практические преимущества:**

Без ручного удаления пришлось бы ждать истечения TTL даже зная что данные устарели, что приводит к багам согласованности в продакшене.`
		},
		uz: {
			title: `Keshdan element o'chirish`,
			solutionCode: `package cache

import (
	"sync"
	"time"
)

type entry struct {
	v   any
	exp time.Time
}

type TTLCache struct {
	mu  sync.RWMutex
	m   map[string]entry
	ttl time.Duration
}

func (c *TTLCache) Delete(key string) bool {
	if c == nil {	// Nil cache hech qachon kalit saqlamaydi
		return false
	}
	c.mu.Lock()	// Asosiy map ni o'zgartirish uchun qulf
	defer c.mu.Unlock()	// O'chirish urinishidan keyin qulf ochilishini kafolatlash
	if _, ok := c.m[key]; ok {	// O'chirishdan oldin kalit mavjudligini aniqlash
		delete(c.m, key)	// Muddati o'tish holatidan qat'i nazar yozuvni o'chirish
		return true	// Yozuv mavjud va o'chirilganini signal qilish
	}
	return false	// Kalit saqlanmagan bo'lsa mavjud emasligini xabar qilish
}`,
			description: `Keshdan yozuvlarni o'chirish uchun thread-safe **Delete** metodini amalga oshiring.

**Talablar:**
1. TTLCache uchun \`Delete(key string) bool\` metodini amalga oshiring
2. cache nil yoki kalit mavjud bo'lmasa \`false\` qaytaring
3. map o'chirish uchun eksklyuziv qulf (Lock/Unlock) ishlating
4. O'chirishdan oldin kalit mavjudligini tekshiring
5. Yozuv topilgan va o'chirilgan bo'lsa \`true\` qaytaring (muddati o'tgan bo'lsa ham)
6. Kalit hech qachon map da bo'lmagan bo'lsa \`false\` qaytaring

**Misol:**
\`\`\`go
cache := NewTTLCache(5 * time.Second)
cache.Set("key", "value")

deleted := cache.Delete("key")       // true (kalit mavjud edi)
deleted = cache.Delete("key")        // false (allaqachon o'chirilgan)
deleted = cache.Delete("nonexistent") // false (hech qachon mavjud bo'lmagan)
\`\`\`

**Cheklovlar:**
- map mutatsiyasi uchun Lock ishlating (RLock emas)
- O'chirishdan oldin mavjudlikni tekshiring
- Yozuv muddati o'tgan bo'lsa ham true qaytaring (map da mavjud edi)`,
			hint1: `O'chirishdan oldin _, ok := c.m[key] bilan kalit mavjudligini tekshiring. Muddati o'tgan bo'lsa ham o'chiring.`,
			hint2: `Lock ishlating (RLock emas) chunki delete() map strukturasini o'zgartiradi.`,
			whyItMatters: `Qo'lda o'chirish ma'lumotlar izchilligini saqlash uchun muhim kesh invalidatsiya pattern lariga imkon beradi.

**Nima uchun qo'lda Delete:**
- **Darhol invalidatsiya:** Ma'lumotlar o'zgarganda TTL ni kutmang
- **Kesh izchilligi:** Manba ma'lumotlari o'zgarganda yangilash/o'chirish
- **Xotira nazorati:** Katta yozuvlarni muddati o'tishidan oldin o'chirish
- **Testlash:** Test holatlari orasida keshni tozalash

**Ishlab chiqarishda foydalanish holatlari:**

1. **Write-Through kesh invalidatsiyasi:**
\`\`\`go
func UpdateUser(id string, user *User) error {
    if err := db.Update(id, user); err != nil {
        return err
    }
    userCache.Delete(id)  // Eskirgan kesh yozuvini invalidatsiya qilish
    return nil
}
\`\`\`

2. **Cache-Aside patterni:**
\`\`\`go
func InvalidateUser(id string) {
    userCache.Delete(id)  // Aniq invalidatsiya
    // Keyingi Get DB dan yangi ma'lumotlarni oladi
}
\`\`\`

3. **Logout/Session invalidatsiyasi:**
\`\`\`go
func Logout(sessionID string) {
    sessionCache.Delete(sessionID)  // Darhol bekor qilish
    // Foydalanuvchi TTL o'tmagan bo'lsa ham muddati o'tgan sessiyadan foydalana olmaydi
}
\`\`\`

4. **Ommaviy invalidatsiya:**
\`\`\`go
func InvalidateUserCaches(userIDs []string) {
    for _, id := range userIDs {
        if userCache.Delete(id) {
            log.Printf("Invalidated cache for user %s", id)
        }
    }
}
\`\`\`

**Nima uchun Bool qaytarish:**
\`\`\`go
// Chaqiruvchi farqlashi mumkin:
if cache.Delete(key) {
    log.Info("Cache invalidated")  // Kalit mavjud edi
} else {
    log.Warn("Key not in cache")   // Allaqachon o'chirilgan yoki hech qachon mavjud bo'lmagan
}

// Metrikalar uchun foydali:
deletedCount := 0
for _, key := range keys {
    if cache.Delete(key) {
        deletedCount++
    }
}
log.Printf("Invalidated %d cache entries", deletedCount)
\`\`\`

**Nima uchun muddati o'tgan yozuvlarni o'chirish:**
\`\`\`go
// Muddati o'tgan bo'lsa ham true qaytaring chunki:
// 1. Yozuv jismoniy ravishda map da mavjud (xotira egallaydi)
// 2. Chaqiruvchi kalit qachondir mavjud bo'lganligini bilishi kerak
// 3. Standart map harakati bilan mos keladi

if _, ok := c.m[key]; ok {  // Map da mavjudmi?
    delete(c.m, key)        // O'chirish
    return true             // Mavjudlik haqida xabar berish
}
\`\`\`

**Umumiy pattern - Write-Through kesh:**
\`\`\`go
type UserService struct {
    db    *Database
    cache *TTLCache
}

func (s *UserService) GetUser(id string) (*User, error) {
    // Avval keshni sinab ko'ring
    if cached, ok := s.cache.Get(id); ok {
        return cached.(*User), nil
    }

    // Kesh miss: DB dan olish
    user, err := s.db.GetUser(id)
    if err != nil {
        return nil, err
    }

    s.cache.Set(id, user)
    return user, nil
}

func (s *UserService) UpdateUser(id string, user *User) error {
    if err := s.db.UpdateUser(id, user); err != nil {
        return err
    }

    // Yozishda keshni invalidatsiya qilish
    s.cache.Delete(id)  // Keyingi safar yangi o'qishni majbur qilish
    return nil
}
\`\`\`

**Haqiqiy dunyodan misollar:**
- **Redis:** Aniq kalit o'chirish uchun DEL buyrug'i
- **Memcached:** Muddati o'tishdan qat'i nazar DELETE buyrug'i
- **HTTP keshlari:** Cache-Control: no-cache qayta tekshirishga majbur qiladi
- **CDN invalidatsiyasi:** Keshlangan aktivlarni o'chirish uchun Purge API

**Ishlash ko'rsatkichlari:**
- **O(1) operatsiya:** Hash-map o'chirish doimiy vaqtda
- **Qulf raqobati:** Qisqa eksklyuziv qulf, minimal ta'sir
- **Xotira:** Darhol yozuv xotirasini bo'shatadi, katta qiymatlar uchun foydali

**Amaliy foydalari:**

Qo'lda o'chirishsiz, ma'lumotlar eskirganligini bilgan holda ham TTL muddati o'tishini kutish kerak bo'lardi, bu ishlab chiqarishda izchillik xatolariga olib keladi.`
		}
	}
};

export default task;
