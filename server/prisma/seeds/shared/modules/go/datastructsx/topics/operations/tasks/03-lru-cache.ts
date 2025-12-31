import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-datastructsx-lru-cache',
	title: 'LRU Cache with Doubly Linked List and HashMap',
	difficulty: 'medium',
	tags: ['go', 'generics', 'cache', 'linked-list', 'hash-map', 'algorithms'],
	estimatedTime: '50m',
	isPremium: false,
	youtubeUrl: '',
	description: `Build a production-grade Least Recently Used (LRU) Cache with O(1) get and put operations using a combination of doubly linked list and hash map.

**You will implement:**

**Level 1 (Medium) - LRU Cache Operations:**
1. **NewLRUCache[K comparable, V any](capacity int) *LRUCache[K, V]** - Create cache with fixed size
2. **Get(key K) (V, bool)** - Retrieve value and mark as recently used
3. **Put(key K, value V)** - Add/update entry, evict LRU if at capacity
4. **Size() int** - Return current number of entries
5. **Capacity() int** - Return maximum capacity

**Key Concepts:**
- **LRU Eviction**: When full, remove least recently used item
- **O(1) Operations**: Both get and put must be constant time
- **Doubly Linked List**: Track access order (most recent at head)
- **Hash Map**: Enable O(1) key lookup to list nodes
- **Recently Used**: Both Get and Put mark items as recently used

**How LRU Works:**

\`\`\`
Cache with capacity 3:

Put("a", 1):  [a:1]
Put("b", 2):  [b:2] -> [a:1]
Put("c", 3):  [c:3] -> [b:2] -> [a:1]

Get("a"):     [a:1] -> [c:3] -> [b:2]  // "a" moved to front
Put("d", 4):  [d:4] -> [a:1] -> [c:3]  // "b" evicted (LRU)

Most recent at head, least recent at tail
\`\`\`

**Example Usage:**

\`\`\`go
// Create cache with capacity 3
cache := NewLRUCache[string, int](3)

// Add entries
cache.Put("a", 1)
cache.Put("b", 2)
cache.Put("c", 3)

// Get entry (marks as recently used)
val, ok := cache.Get("a")
// val == 1, ok == true
// Order now: [a, c, b]

// Add entry when full (evicts LRU)
cache.Put("d", 4)
// "b" was evicted (least recently used)
// Order now: [d, a, c]

val, ok = cache.Get("b")
// val == 0, ok == false (evicted)

// Update existing entry
cache.Put("a", 100)
// Order now: [a, d, c]

size := cache.Size()
// size == 3

capacity := cache.Capacity()
// capacity == 3

// Real-world example: API response cache
apiCache := NewLRUCache[string, Response](1000)

func FetchUser(userID string) Response {
    // Check cache first
    if response, ok := apiCache.Get(userID); ok {
        return response  // Cache hit!
    }

    // Cache miss - fetch from API
    response := callAPI(userID)
    apiCache.Put(userID, response)
    return response
}

// Most frequently accessed users stay in cache
// Rarely accessed users get evicted automatically
\`\`\`

**Constraints:**
- Get and Put must be O(1) time complexity
- Use doubly linked list to track access order
- Use hash map for O(1) key lookup
- Evict least recently used when at capacity
- Both Get and Put mark items as recently used
- Handle edge cases: capacity 0, capacity 1, Get on empty cache`,
	initialCode: `package datastructsx

// Node represents a doubly linked list node
type Node[K comparable, V any] struct {
	// TODO: Add fields for key, value, prev, next pointers
}

// LRUCache implements Least Recently Used cache
type LRUCache[K comparable, V any] struct {
	// TODO: Add fields for capacity, hash map, head, tail pointers
}

// TODO: Implement NewLRUCache
// Create new LRU cache with given capacity
func NewLRUCache[K comparable, V any](capacity int) *LRUCache[K, V] {
	// TODO: Implement
}

// TODO: Implement Get
// Retrieve value by key and mark as recently used
// Return (value, true) if found, (zero value, false) if not found
func (c *LRUCache[K, V]) Get(key K) (V, bool) {
	// TODO: Implement
}

// TODO: Implement Put
// Add or update key-value pair
// If at capacity, evict least recently used item
// Mark item as recently used
func (c *LRUCache[K, V]) Put(key K, value V) {
	// TODO: Implement
}

// TODO: Implement Size
// Return current number of entries in cache
func (c *LRUCache[K, V]) Size() int {
	// TODO: Implement
}

// TODO: Implement Capacity
// Return maximum capacity of cache
func (c *LRUCache[K, V]) Capacity() int {
	// TODO: Implement
}

// TODO: Helper methods (optional)
// - moveToFront(node *Node[K, V])
// - removeNode(node *Node[K, V])
// - addToFront(node *Node[K, V])
// - removeTail() *Node[K, V]`,
	solutionCode: `package datastructsx

type Node[K comparable, V any] struct {
	key   K
	value V
	prev  *Node[K, V]
	next  *Node[K, V]
}

type LRUCache[K comparable, V any] struct {
	capacity int
	cache    map[K]*Node[K, V]
	head     *Node[K, V]  // Most recently used
	tail     *Node[K, V]  // Least recently used
}

func NewLRUCache[K comparable, V any](capacity int) *LRUCache[K, V] {
	c := &LRUCache[K, V]{
		capacity: capacity,
		cache:    make(map[K]*Node[K, V]),
		head:     &Node[K, V]{},
		tail:     &Node[K, V]{},
	}
	// Connect head and tail
	c.head.next = c.tail
	c.tail.prev = c.head
	return c
}

func (c *LRUCache[K, V]) Get(key K) (V, bool) {
	if node, exists := c.cache[key]; exists {
		// Move to front (mark as recently used)
		c.moveToFront(node)
		return node.value, true
	}
	var zero V
	return zero, false
}

func (c *LRUCache[K, V]) Put(key K, value V) {
	if c.capacity <= 0 {
		return
	}

	if node, exists := c.cache[key]; exists {
		// Update existing entry
		node.value = value
		c.moveToFront(node)
		return
	}

	// Create new node
	node := &Node[K, V]{
		key:   key,
		value: value,
	}

	// Add to cache and front of list
	c.cache[key] = node
	c.addToFront(node)

	// Evict LRU if at capacity
	if len(c.cache) > c.capacity {
		lru := c.removeTail()
		delete(c.cache, lru.key)
	}
}

func (c *LRUCache[K, V]) Size() int {
	return len(c.cache)
}

func (c *LRUCache[K, V]) Capacity() int {
	return c.capacity
}

// Helper: Move node to front (mark as recently used)
func (c *LRUCache[K, V]) moveToFront(node *Node[K, V]) {
	c.removeNode(node)
	c.addToFront(node)
}

// Helper: Remove node from list
func (c *LRUCache[K, V]) removeNode(node *Node[K, V]) {
	node.prev.next = node.next
	node.next.prev = node.prev
}

// Helper: Add node to front of list
func (c *LRUCache[K, V]) addToFront(node *Node[K, V]) {
	node.prev = c.head
	node.next = c.head.next
	c.head.next.prev = node
	c.head.next = node
}

// Helper: Remove and return tail node (LRU)
func (c *LRUCache[K, V]) removeTail() *Node[K, V] {
	lru := c.tail.prev
	c.removeNode(lru)
	return lru
}`,
	testCode: `package datastructsx

import "testing"

func Test1(t *testing.T) {
	// Basic put and get
	cache := NewLRUCache[string, int](2)
	cache.Put("a", 1)
	val, ok := cache.Get("a")
	if !ok || val != 1 {
		t.Errorf("expected 1, got %d", val)
	}
}

func Test2(t *testing.T) {
	// LRU eviction
	cache := NewLRUCache[string, int](2)
	cache.Put("a", 1)
	cache.Put("b", 2)
	cache.Put("c", 3)
	_, ok := cache.Get("a")
	if ok {
		t.Error("expected a to be evicted")
	}
}

func Test3(t *testing.T) {
	// Get marks as recently used
	cache := NewLRUCache[string, int](2)
	cache.Put("a", 1)
	cache.Put("b", 2)
	cache.Get("a")
	cache.Put("c", 3)
	_, ok := cache.Get("b")
	if ok {
		t.Error("expected b to be evicted")
	}
}

func Test4(t *testing.T) {
	// Update existing
	cache := NewLRUCache[string, int](2)
	cache.Put("a", 1)
	cache.Put("a", 100)
	val, _ := cache.Get("a")
	if val != 100 {
		t.Errorf("expected 100, got %d", val)
	}
}

func Test5(t *testing.T) {
	// Get non-existent
	cache := NewLRUCache[string, int](2)
	_, ok := cache.Get("x")
	if ok {
		t.Error("expected not found")
	}
}

func Test6(t *testing.T) {
	// Size and Capacity
	cache := NewLRUCache[string, int](3)
	cache.Put("a", 1)
	cache.Put("b", 2)
	if cache.Size() != 2 || cache.Capacity() != 3 {
		t.Errorf("expected size 2 capacity 3, got %d %d", cache.Size(), cache.Capacity())
	}
}

func Test7(t *testing.T) {
	// Capacity 1
	cache := NewLRUCache[string, int](1)
	cache.Put("a", 1)
	cache.Put("b", 2)
	_, ok := cache.Get("a")
	if ok {
		t.Error("expected a evicted")
	}
	val, _ := cache.Get("b")
	if val != 2 {
		t.Errorf("expected 2, got %d", val)
	}
}

func Test8(t *testing.T) {
	// Capacity 0
	cache := NewLRUCache[string, int](0)
	cache.Put("a", 1)
	if cache.Size() != 0 {
		t.Errorf("expected size 0, got %d", cache.Size())
	}
}

func Test9(t *testing.T) {
	// Multiple operations
	cache := NewLRUCache[string, int](3)
	cache.Put("a", 1)
	cache.Put("b", 2)
	cache.Put("c", 3)
	cache.Get("a")
	cache.Get("b")
	cache.Put("d", 4)
	_, ok := cache.Get("c")
	if ok {
		t.Error("expected c evicted")
	}
}

func Test10(t *testing.T) {
	// Integer keys
	cache := NewLRUCache[int, string](2)
	cache.Put(1, "one")
	cache.Put(2, "two")
	val, _ := cache.Get(1)
	if val != "one" {
		t.Errorf("expected one, got %s", val)
	}
}`,
	hint1: `Use doubly linked list with dummy head and tail nodes. Node struct needs: key, value, prev, next. Cache struct needs: capacity, map[K]*Node, head, tail. In NewLRUCache, connect head.next = tail and tail.prev = head.`,
	hint2: `Get: Check map, if exists move node to front and return value. Put: If key exists, update value and move to front. If new key, create node, add to map and front. If size > capacity, remove tail node and delete from map. Helper moveToFront: removeNode then addToFront.`,
	whyItMatters: `LRU Cache is one of the most important data structures in production systems, used everywhere from CPU caches to web servers to databases.

**Why LRU Cache Matters:**

**1. Real-World Cache Usage**

Every major system uses LRU caching:

\`\`\`go
// Web server - cache API responses
type APIServer struct {
    cache *LRUCache[string, Response]
}

func (s *APIServer) GetUser(userID string) Response {
    // Check cache first
    if response, ok := s.cache.Get(userID); ok {
        log.Printf("Cache hit for user %s", userID)
        return response  // Instant response!
    }

    // Cache miss - query database
    log.Printf("Cache miss for user %s - querying DB", userID)
    response := s.database.QueryUser(userID)
    s.cache.Put(userID, response)
    return response
}
\`\`\`

**Performance Impact**:
- Cache hit: 0.1ms response time
- Cache miss (database query): 50ms response time
- 500x faster with cache hit!

**Real Incident**: An e-commerce site had 1000 req/sec. Without cache, database couldn't handle load (crashed). With LRU cache (90% hit rate), only 100 DB queries/sec. System remained stable.

**2. Memory Management - Why LRU?**

You can't cache everything - memory is limited. LRU eviction policy is optimal:

\`\`\`go
// Example: Cache with capacity 3
cache := NewLRUCache[string, string](3)

// Access pattern: [A, B, C, A, D]
cache.Put("A", "data1")  // Cache: [A]
cache.Put("B", "data2")  // Cache: [B, A]
cache.Put("C", "data3")  // Cache: [C, B, A]

cache.Get("A")           // Cache: [A, C, B] (A moved to front)

cache.Put("D", "data4")  // Cache: [D, A, C] (B evicted - least recently used)
\`\`\`

**Why B was evicted?**
- A was accessed most recently
- C was added recently
- B hasn't been used longest - least likely to be needed

**Alternative policies and why they're worse:**

| Policy | Why It's Worse | Example Problem |
|--------|---------------|-----------------|
| FIFO (First-In-First-Out) | Evicts oldest, ignoring usage | Popular item added first gets evicted |
| Random | No pattern, unpredictable | Frequently used items randomly evicted |
| LFU (Least Frequently Used) | Old frequent items never evict | Yesterday's popular item blocks today's |
| MRU (Most Recently Used) | Evicts newest items | Counter-intuitive, poor hit rate |

**3. O(1) Requirement - Why Both Structures?**

**Naive approach - Array only (O(n) operations):**

\`\`\`go
// BAD - Linear search and shifts
type NaiveCache struct {
    items []CacheEntry
    capacity int
}

func (c *NaiveCache) Get(key string) (value string, ok bool) {
    // O(n) - search through array
    for i, entry := range c.items {
        if entry.key == key {
            // O(n) - shift elements to move to front
            c.items = append([]CacheEntry{entry}, append(c.items[:i], c.items[i+1:]...)...)
            return entry.value, true
        }
    }
    return "", false
}
// Get: O(n), Put: O(n) - TOO SLOW!
\`\`\`

**Performance at scale:**
- 1,000 items: ~500 comparisons per operation
- 10,000 items: ~5,000 comparisons per operation
- Unusable for production!

**Optimized approach - Hash Map + Doubly Linked List:**

\`\`\`go
type LRUCache struct {
    cache map[string]*Node  // O(1) key lookup
    head  *Node              // O(1) access to most recent
    tail  *Node              // O(1) access to least recent
}

func (c *LRUCache) Get(key string) (value string, ok bool) {
    if node, exists := c.cache[key]; exists {  // O(1) lookup
        c.moveToFront(node)                     // O(1) pointer manipulation
        return node.value, true
    }
    return "", false
}
// Get: O(1), Put: O(1) - PERFECT!
\`\`\`

**Why this works:**
- **Hash Map**: O(1) lookup to find node by key
- **Doubly Linked List**: O(1) remove/add operations with prev/next pointers
- **Combined**: Both lookup AND reordering are O(1)

**4. Production Use Cases**

**Use Case 1: Database Query Cache**

\`\`\`go
type Database struct {
    queryCache *LRUCache[string, QueryResult]
}

func (db *Database) ExecuteQuery(sql string) QueryResult {
    if result, ok := db.queryCache.Get(sql); ok {
        return result  // Instant!
    }

    result := db.executeExpensiveQuery(sql)
    db.queryCache.Put(sql, result)
    return result
}
\`\`\`

**Impact**: 95% cache hit rate reduces database load by 20x.

**Use Case 2: Image/Asset Cache**

\`\`\`go
type ImageServer struct {
    imageCache *LRUCache[string, []byte]
}

func (s *ImageServer) ServeImage(imageID string) []byte {
    if data, ok := s.imageCache.Get(imageID); ok {
        return data  // Serve from memory
    }

    data := s.loadFromDisk(imageID)  // Expensive disk I/O
    s.imageCache.Put(imageID, data)
    return data
}
\`\`\`

**Impact**: Disk I/O takes 10ms vs memory access 0.01ms (1000x faster).

**Use Case 3: DNS Cache**

\`\`\`go
type DNSResolver struct {
    cache *LRUCache[string, IPAddress]
}

func (r *DNSResolver) Resolve(domain string) IPAddress {
    if ip, ok := r.cache.Get(domain); ok {
        return ip  // No network round-trip
    }

    ip := r.queryDNSServer(domain)  // Network latency
    r.cache.Put(domain, ip)
    return ip
}
\`\`\`

**Impact**: DNS query takes 50ms vs cache hit 0.1ms (500x faster).

**Use Case 4: Compiled Template Cache**

\`\`\`go
type TemplateEngine struct {
    cache *LRUCache[string, Template]
}

func (e *TemplateEngine) Render(name string, data interface{}) string {
    if tmpl, ok := e.cache.Get(name); ok {
        return tmpl.Execute(data)  // Use compiled template
    }

    tmpl := e.compileTemplate(name)  // Expensive parsing
    e.cache.Put(name, tmpl)
    return tmpl.Execute(data)
}
\`\`\`

**5. LRU Cache in Operating Systems**

**CPU Cache (L1, L2, L3)**:
- CPUs use LRU-like policies for cache lines
- Recently accessed memory stays in fast cache
- Least recently used data evicted to make room

**Page Replacement**:
- Operating systems use LRU for virtual memory
- Recently accessed pages stay in RAM
- Least recently used pages swapped to disk

**6. Interview Favorite**

LRU Cache is one of the most common interview questions because it tests:
- Data structure knowledge (hash map, linked list)
- Algorithm design (combining structures)
- Time complexity analysis (O(1) requirement)
- Pointer manipulation (doubly linked list)
- Production system thinking (real use cases)

**7. Variations and Extensions**

**TTL (Time-To-Live) LRU**:
\`\`\`go
type Node struct {
    key       string
    value     interface{}
    expiresAt time.Time
    prev, next *Node
}

func (c *LRUCache) Get(key string) (interface{}, bool) {
    if node, exists := c.cache[key]; exists {
        if time.Now().After(node.expiresAt) {
            c.evict(node)  // Expired
            return nil, false
        }
        c.moveToFront(node)
        return node.value, true
    }
    return nil, false
}
\`\`\`

**LRU-K (Track K accesses)**:
- Standard LRU tracks 1 access (LRU-1)
- LRU-2 only promotes after 2 accesses
- Prevents one-time scans from evicting useful data

**8. Real Production Metrics**

From a real production system (e-commerce site):

\`\`\`
LRU Cache Configuration:
- Capacity: 10,000 entries
- Entry size: ~10KB
- Total memory: ~100MB

Performance:
- Requests/sec: 5,000
- Cache hit rate: 92%
- P50 latency: 2ms (cache hit), 45ms (cache miss)
- P99 latency: 5ms (cache hit), 120ms (cache miss)

Impact:
- Without cache: 5,000 DB queries/sec → database overload
- With cache: 400 DB queries/sec → smooth operation
- 92% reduction in database load
- $50,000/month savings in database scaling costs
\`\`\`

**Key Takeaways:**
- LRU Cache provides O(1) get and put operations
- Combines hash map (lookup) + doubly linked list (ordering)
- Evicts least recently used items when at capacity
- Critical for production performance (100-1000x speedup)
- Used everywhere: web servers, databases, operating systems
- Memory-efficient: Only keeps frequently accessed data
- Optimal eviction policy for most access patterns`,
	order: 2,
	translations: {
		ru: {
			title: 'LRU Cache с двусвязным списком и HashMap',
			description: `Постройте production-grade Least Recently Used (LRU) Cache с O(1) операциями get и put, используя комбинацию двусвязного списка и hash map.

**Вы реализуете:**

**Уровень 1 (Средний) — Операции LRU Cache:**
1. **NewLRUCache[K comparable, V any](capacity int) *LRUCache[K, V]** — Создать кеш с фиксированным размером
2. **Get(key K) (V, bool)** — Получить значение и пометить как недавно использованное
3. **Put(key K, value V)** — Добавить/обновить запись, вытеснить LRU при достижении ёмкости
4. **Size() int** — Вернуть текущее количество записей
5. **Capacity() int** — Вернуть максимальную ёмкость

**Ключевые концепции:**
- **LRU вытеснение**: При заполнении удалить наименее недавно использованный элемент
- **O(1) операции**: Как get, так и put должны быть за константное время
- **Двусвязный список**: Отслеживать порядок доступа (самый недавний в голове)
- **Hash Map**: Обеспечить O(1) поиск ключа к узлам списка
- **Недавно использованный**: Как Get, так и Put помечают элементы как недавно использованные

**Как работает LRU:**

\`\`\`
Кеш с ёмкостью 3:

Put("a", 1):  [a:1]
Put("b", 2):  [b:2] -> [a:1]
Put("c", 3):  [c:3] -> [b:2] -> [a:1]

Get("a"):     [a:1] -> [c:3] -> [b:2]  // "a" перемещён в начало
Put("d", 4):  [d:4] -> [a:1] -> [c:3]  // "b" вытеснен (LRU)

Самый недавний в голове, наименее недавний в хвосте
\`\`\`

**Пример использования:**

\`\`\`go
// Создать кеш с ёмкостью 3
cache := NewLRUCache[string, int](3)

// Добавить записи
cache.Put("a", 1)
cache.Put("b", 2)
cache.Put("c", 3)

// Получить запись (пометить как недавно использованную)
val, ok := cache.Get("a")
// val == 1, ok == true
// Порядок теперь: [a, c, b]

// Добавить запись при заполнении (вытесняет LRU)
cache.Put("d", 4)
// "b" был вытеснен (наименее недавно использованный)
// Порядок теперь: [d, a, c]

val, ok = cache.Get("b")
// val == 0, ok == false (вытеснен)

// Обновить существующую запись
cache.Put("a", 100)
// Порядок теперь: [a, d, c]

size := cache.Size()
// size == 3

capacity := cache.Capacity()
// capacity == 3

// Реальный пример: кеш API ответов
apiCache := NewLRUCache[string, Response](1000)

func FetchUser(userID string) Response {
    // Сначала проверить кеш
    if response, ok := apiCache.Get(userID); ok {
        return response  // Попадание в кеш!
    }

    // Промах кеша - запрос к API
    response := callAPI(userID)
    apiCache.Put(userID, response)
    return response
}

// Наиболее часто запрашиваемые пользователи остаются в кеше
// Редко запрашиваемые пользователи автоматически вытесняются
\`\`\`

**Ограничения:**
- Get и Put должны быть O(1) временной сложности
- Использовать двусвязный список для отслеживания порядка доступа
- Использовать hash map для O(1) поиска ключа
- Вытеснять наименее недавно использованный при достижении ёмкости
- Как Get, так и Put помечают элементы как недавно использованные
- Обрабатывать граничные случаи: capacity 0, capacity 1, Get на пустом кеше`,
			hint1: `Используйте двусвязный список с фиктивными узлами head и tail. Node структура нуждается в: key, value, prev, next. Cache структура нуждается в: capacity, map[K]*Node, head, tail. В NewLRUCache соединить head.next = tail и tail.prev = head.`,
			hint2: `Get: Проверить map, если существует переместить узел в начало и вернуть значение. Put: Если ключ существует, обновить значение и переместить в начало. Если новый ключ, создать узел, добавить в map и в начало. Если size > capacity, удалить узел хвоста и удалить из map. Helper moveToFront: removeNode затем addToFront.`,
			whyItMatters: `LRU Cache — одна из самых важных структур данных в production системах, используется повсеместно от CPU кешей до веб-серверов и баз данных.

**Почему LRU Cache важен:**

**1. Реальное использование кеша**

Каждая крупная система использует LRU кеширование:

\`\`\`go
// Веб-сервер — кеш API ответов
type APIServer struct {
    cache *LRUCache[string, Response]
}

func (s *APIServer) GetUser(userID string) Response {
    // Сначала проверить кеш
    if response, ok := s.cache.Get(userID); ok {
        log.Printf("Попадание в кеш для пользователя %s", userID)
        return response  // Мгновенный ответ!
    }

    // Промах кеша — запрос к базе данных
    log.Printf("Промах кеша для пользователя %s - запрос к БД", userID)
    response := s.database.QueryUser(userID)
    s.cache.Put(userID, response)
    return response
}
\`\`\`

**Влияние на производительность**:
- Попадание в кеш: 0.1мс время ответа
- Промах кеша (запрос к БД): 50мс время ответа
- В 500x быстрее с попаданием в кеш!

**Реальный инцидент**: Сайт e-commerce имел 1000 req/sec. Без кеша база данных не могла справиться с нагрузкой (падала). С LRU кешем (90% попаданий) только 100 запросов к БД/сек. Система оставалась стабильной.

**2. Управление памятью — Почему LRU?**

Нельзя кешировать всё — память ограничена. Политика вытеснения LRU оптимальна:

\`\`\`go
// Пример: Кеш с ёмкостью 3
cache := NewLRUCache[string, string](3)

// Паттерн доступа: [A, B, C, A, D]
cache.Put("A", "data1")  // Кеш: [A]
cache.Put("B", "data2")  // Кеш: [B, A]
cache.Put("C", "data3")  // Кеш: [C, B, A]

cache.Get("A")           // Кеш: [A, C, B] (A перемещён в начало)

cache.Put("D", "data4")  // Кеш: [D, A, C] (B вытеснен - наименее недавно использованный)
\`\`\`

**Почему B был вытеснен?**
- A был использован последним
- C был добавлен недавно
- B не использовался дольше всех — наименее вероятно будет нужен

**Альтернативные политики и почему они хуже:**

| Политика | Почему хуже | Пример проблемы |
|----------|-------------|-----------------|
| FIFO | Вытесняет самый старый, игнорируя использование | Популярный элемент, добавленный первым, вытесняется |
| Random | Нет паттерна, непредсказуемо | Часто используемые элементы случайно вытесняются |
| LFU | Старые частые элементы никогда не вытесняются | Вчерашний популярный элемент блокирует сегодняшний |
| MRU | Вытесняет новейшие элементы | Контринтуитивно, низкий hit rate |

**3. Требование O(1) — Почему обе структуры?**

**Наивный подход — только массив (O(n) операции):**

\`\`\`go
// ПЛОХО — Линейный поиск и сдвиги
type NaiveCache struct {
    items []CacheEntry
    capacity int
}

func (c *NaiveCache) Get(key string) (value string, ok bool) {
    // O(n) — поиск по массиву
    for i, entry := range c.items {
        if entry.key == key {
            // O(n) — сдвиг элементов для перемещения в начало
            c.items = append([]CacheEntry{entry}, append(c.items[:i], c.items[i+1:]...)...)
            return entry.value, true
        }
    }
    return "", false
}
// Get: O(n), Put: O(n) — СЛИШКОМ МЕДЛЕННО!
\`\`\`

**Производительность при масштабе:**
- 1,000 элементов: ~500 сравнений на операцию
- 10,000 элементов: ~5,000 сравнений на операцию
- Непригодно для продакшена!

**Оптимизированный подход — Hash Map + Двусвязный список:**

\`\`\`go
type LRUCache struct {
    cache map[string]*Node  // O(1) поиск ключа
    head  *Node              // O(1) доступ к самому недавнему
    tail  *Node              // O(1) доступ к наименее недавнему
}

func (c *LRUCache) Get(key string) (value string, ok bool) {
    if node, exists := c.cache[key]; exists {  // O(1) поиск
        c.moveToFront(node)                     // O(1) манипуляция указателями
        return node.value, true
    }
    return "", false
}
// Get: O(1), Put: O(1) — ИДЕАЛЬНО!
\`\`\`

**Почему это работает:**
- **Hash Map**: O(1) поиск для нахождения узла по ключу
- **Двусвязный список**: O(1) операции удаления/добавления с указателями prev/next
- **Комбинированно**: Как поиск, ТАК И переупорядочивание O(1)

**4. Production примеры использования**

**Пример 1: Кеш запросов к базе данных**

\`\`\`go
type Database struct {
    queryCache *LRUCache[string, QueryResult]
}

func (db *Database) ExecuteQuery(sql string) QueryResult {
    if result, ok := db.queryCache.Get(sql); ok {
        return result  // Мгновенно!
    }

    result := db.executeExpensiveQuery(sql)
    db.queryCache.Put(sql, result)
    return result
}
\`\`\`

**Эффект**: 95% попаданий в кеш снижают нагрузку на БД в 20x.

**Пример 2: Кеш изображений/ресурсов**

\`\`\`go
type ImageServer struct {
    imageCache *LRUCache[string, []byte]
}

func (s *ImageServer) ServeImage(imageID string) []byte {
    if data, ok := s.imageCache.Get(imageID); ok {
        return data  // Из памяти
    }

    data := s.loadFromDisk(imageID)  // Дорогой disk I/O
    s.imageCache.Put(imageID, data)
    return data
}
\`\`\`

**Эффект**: Disk I/O занимает 10мс против 0.01мс доступа к памяти (в 1000x быстрее).

**Пример 3: DNS кеш**

\`\`\`go
type DNSResolver struct {
    cache *LRUCache[string, IPAddress]
}

func (r *DNSResolver) Resolve(domain string) IPAddress {
    if ip, ok := r.cache.Get(domain); ok {
        return ip  // Без сетевого round-trip
    }

    ip := r.queryDNSServer(domain)  // Сетевая задержка
    r.cache.Put(domain, ip)
    return ip
}
\`\`\`

**Эффект**: DNS запрос занимает 50мс против 0.1мс попадания в кеш (в 500x быстрее).

**Пример 4: Кеш скомпилированных шаблонов**

\`\`\`go
type TemplateEngine struct {
    cache *LRUCache[string, Template]
}

func (e *TemplateEngine) Render(name string, data interface{}) string {
    if tmpl, ok := e.cache.Get(name); ok {
        return tmpl.Execute(data)  // Используем скомпилированный шаблон
    }

    tmpl := e.compileTemplate(name)  // Дорогой парсинг
    e.cache.Put(name, tmpl)
    return tmpl.Execute(data)
}
\`\`\`

**5. LRU Cache в операционных системах**

**CPU Cache (L1, L2, L3)**:
- Процессоры используют LRU-подобные политики для строк кеша
- Недавно использованная память остаётся в быстром кеше
- Наименее недавно использованные данные вытесняются для освобождения места

**Замена страниц**:
- Операционные системы используют LRU для виртуальной памяти
- Недавно использованные страницы остаются в RAM
- Наименее недавно использованные страницы перемещаются на диск

**6. Любимый вопрос на интервью**

LRU Cache — один из самых частых вопросов на собеседованиях, потому что проверяет:
- Знание структур данных (hash map, linked list)
- Проектирование алгоритмов (комбинирование структур)
- Анализ временной сложности (требование O(1))
- Манипуляция указателями (двусвязный список)
- Мышление production систем (реальные примеры использования)

**7. Вариации и расширения**

**TTL (Time-To-Live) LRU**:
\`\`\`go
type Node struct {
    key       string
    value     interface{}
    expiresAt time.Time
    prev, next *Node
}

func (c *LRUCache) Get(key string) (interface{}, bool) {
    if node, exists := c.cache[key]; exists {
        if time.Now().After(node.expiresAt) {
            c.evict(node)  // Истёк срок
            return nil, false
        }
        c.moveToFront(node)
        return node.value, true
    }
    return nil, false
}
\`\`\`

**LRU-K (отслеживать K доступов)**:
- Стандартный LRU отслеживает 1 доступ (LRU-1)
- LRU-2 продвигает только после 2 доступов
- Предотвращает однократное сканирование от вытеснения полезных данных

**8. Реальные production метрики**

Из реальной production системы (сайт e-commerce):

\`\`\`
Конфигурация LRU Cache:
- Ёмкость: 10,000 записей
- Размер записи: ~10KB
- Общая память: ~100MB

Производительность:
- Запросов/сек: 5,000
- Коэффициент попадания в кеш: 92%
- P50 задержка: 2мс (попадание), 45мс (промах)
- P99 задержка: 5мс (попадание), 120мс (промах)

Эффект:
- Без кеша: 5,000 запросов к БД/сек → перегрузка базы данных
- С кешем: 400 запросов к БД/сек → плавная работа
- 92% снижение нагрузки на базу данных
- $50,000/месяц экономии на масштабировании БД
\`\`\`

**Ключевые выводы:**
- LRU Cache обеспечивает O(1) операции get и put
- Комбинирует hash map (поиск) + двусвязный список (упорядочивание)
- Вытесняет наименее недавно использованные элементы при достижении ёмкости
- Критичен для производительности продакшена (ускорение в 100-1000x)
- Используется повсеместно: веб-серверы, базы данных, операционные системы
- Эффективен по памяти: Хранит только часто запрашиваемые данные
- Оптимальная политика вытеснения для большинства паттернов доступа`,
			solutionCode: `package datastructsx

type Node[K comparable, V any] struct {
	key   K
	value V
	prev  *Node[K, V]
	next  *Node[K, V]
}

type LRUCache[K comparable, V any] struct {
	capacity int
	cache    map[K]*Node[K, V]
	head     *Node[K, V]  // Самый недавно использованный
	tail     *Node[K, V]  // Наименее недавно использованный
}

func NewLRUCache[K comparable, V any](capacity int) *LRUCache[K, V] {
	c := &LRUCache[K, V]{
		capacity: capacity,
		cache:    make(map[K]*Node[K, V]),
		head:     &Node[K, V]{},
		tail:     &Node[K, V]{},
	}
	// Соединить head и tail
	c.head.next = c.tail
	c.tail.prev = c.head
	return c
}

func (c *LRUCache[K, V]) Get(key K) (V, bool) {
	if node, exists := c.cache[key]; exists {
		// Переместить в начало (пометить как недавно использованный)
		c.moveToFront(node)
		return node.value, true
	}
	var zero V
	return zero, false
}

func (c *LRUCache[K, V]) Put(key K, value V) {
	if c.capacity <= 0 {
		return
	}

	if node, exists := c.cache[key]; exists {
		// Обновить существующую запись
		node.value = value
		c.moveToFront(node)
		return
	}

	// Создать новый узел
	node := &Node[K, V]{
		key:   key,
		value: value,
	}

	// Добавить в кеш и в начало списка
	c.cache[key] = node
	c.addToFront(node)

	// Вытеснить LRU если достигнута ёмкость
	if len(c.cache) > c.capacity {
		lru := c.removeTail()
		delete(c.cache, lru.key)
	}
}

func (c *LRUCache[K, V]) Size() int {
	return len(c.cache)
}

func (c *LRUCache[K, V]) Capacity() int {
	return c.capacity
}

// Helper: Переместить узел в начало (пометить как недавно использованный)
func (c *LRUCache[K, V]) moveToFront(node *Node[K, V]) {
	c.removeNode(node)
	c.addToFront(node)
}

// Helper: Удалить узел из списка
func (c *LRUCache[K, V]) removeNode(node *Node[K, V]) {
	node.prev.next = node.next
	node.next.prev = node.prev
}

// Helper: Добавить узел в начало списка
func (c *LRUCache[K, V]) addToFront(node *Node[K, V]) {
	node.prev = c.head
	node.next = c.head.next
	c.head.next.prev = node
	c.head.next = node
}

// Helper: Удалить и вернуть узел хвоста (LRU)
func (c *LRUCache[K, V]) removeTail() *Node[K, V] {
	lru := c.tail.prev
	c.removeNode(lru)
	return lru
}`
		},
		uz: {
			title: `Ikki tomonlama bog'langan ro'yxat va HashMap bilan LRU Cache`,
			description: `Ikki tomonlama bog'langan ro'yxat va hash map kombinatsiyasi yordamida O(1) get va put operatsiyalari bilan production-grade Least Recently Used (LRU) Cache yarating.

**Siz amalga oshirasiz:**

**1-Daraja (O'rta) — LRU Cache operatsiyalari:**
1. **NewLRUCache[K comparable, V any](capacity int) *LRUCache[K, V]** — Belgilangan o'lchamda cache yaratish
2. **Get(key K) (V, bool)** — Qiymatni olish va yaqinda ishlatilgan deb belgilash
3. **Put(key K, value V)** — Yozuvni qo'shish/yangilash, sig'im to'lsa LRU ni chiqarib tashlash
4. **Size() int** — Joriy yozuvlar sonini qaytarish
5. **Capacity() int** — Maksimal sig'imni qaytarish

**Asosiy tushunchalar:**
- **LRU chiqarish**: To'ldirilganda eng kam yaqinda ishlatilgan elementni o'chirish
- **O(1) operatsiyalar**: Har ikkala get va put doimiy vaqtda bo'lishi kerak
- **Ikki tomonlama bog'langan ro'yxat**: Kirish tartibini kuzatish (eng yaqinda boshda)
- **Hash Map**: Kalit bo'yicha ro'yxat tugunlariga O(1) qidiruvni ta'minlash
- **Yaqinda ishlatilgan**: Har ikkala Get va Put elementlarni yaqinda ishlatilgan deb belgilaydi

**LRU qanday ishlaydi:**

\`\`\`
Sig'imi 3 bo'lgan cache:

Put("a", 1):  [a:1]
Put("b", 2):  [b:2] -> [a:1]
Put("c", 3):  [c:3] -> [b:2] -> [a:1]

Get("a"):     [a:1] -> [c:3] -> [b:2]  // "a" boshga ko'chirildi
Put("d", 4):  [d:4] -> [a:1] -> [c:3]  // "b" chiqarildi (LRU)

Eng yaqinda boshda, eng kam yaqinda oxirida
\`\`\`

**Foydalanish misoli:**

\`\`\`go
// Sig'imi 3 bo'lgan cache yaratish
cache := NewLRUCache[string, int](3)

// Yozuvlar qo'shish
cache.Put("a", 1)
cache.Put("b", 2)
cache.Put("c", 3)

// Yozuvni olish (yaqinda ishlatilgan deb belgilash)
val, ok := cache.Get("a")
// val == 1, ok == true
// Tartib endi: [a, c, b]

// To'lgan paytda yozuv qo'shish (LRU ni chiqaradi)
cache.Put("d", 4)
// "b" chiqarildi (eng kam yaqinda ishlatilgan)
// Tartib endi: [d, a, c]

val, ok = cache.Get("b")
// val == 0, ok == false (chiqarildi)

// Mavjud yozuvni yangilash
cache.Put("a", 100)
// Tartib endi: [a, d, c]

size := cache.Size()
// size == 3

capacity := cache.Capacity()
// capacity == 3

// Haqiqiy misol: API javoblari cache
apiCache := NewLRUCache[string, Response](1000)

func FetchUser(userID string) Response {
    // Avval cache ni tekshirish
    if response, ok := apiCache.Get(userID); ok {
        return response  // Cache hit!
    }

    // Cache miss - API dan olish
    response := callAPI(userID)
    apiCache.Put(userID, response)
    return response
}

// Eng tez-tez so'ralgan foydalanuvchilar cache da qoladi
// Kamdan-kam so'ralgan foydalanuvchilar avtomatik chiqariladi
\`\`\`

**Cheklovlar:**
- Get va Put O(1) vaqt murakkabligi bo'lishi kerak
- Kirish tartibini kuzatish uchun ikki tomonlama bog'langan ro'yxatdan foydalaning
- Kalit qidiruvi uchun hash map dan foydalaning O(1)
- Sig'imga yetganda eng kam yaqinda ishlatilganni chiqaring
- Har ikkala Get va Put elementlarni yaqinda ishlatilgan deb belgilaydi
- Chegara holatlarni qayta ishlang: capacity 0, capacity 1, bo'sh cache da Get`,
			hint1: `Soxta head va tail tugunlari bilan ikki tomonlama bog'langan ro'yxatdan foydalaning. Node strukturasiga kerak: key, value, prev, next. Cache strukturasiga kerak: capacity, map[K]*Node, head, tail. NewLRUCache da head.next = tail va tail.prev = head ni ulang.`,
			hint2: `Get: Map ni tekshiring, agar mavjud bo'lsa tugunni boshga ko'chiring va qiymatni qaytaring. Put: Agar kalit mavjud bo'lsa, qiymatni yangilang va boshga ko'chiring. Agar yangi kalit bo'lsa, tugun yarating, map va boshga qo'shing. Agar size > capacity bo'lsa, oxir tugunini o'chiring va map dan o'chiring. Helper moveToFront: removeNode keyin addToFront.`,
			whyItMatters: `LRU Cache production tizimlaridagi eng muhim ma'lumotlar strukturalaridan biri bo'lib, CPU cache laridan tortib veb-serverlar va ma'lumotlar bazalariga qadar hamma joyda ishlatiladi.

**Nima uchun LRU Cache muhim:**

**1. Haqiqiy cache foydalanish**

Har bir yirik tizim LRU cache dan foydalanadi:

\`\`\`go
// Veb-server — API javoblari cache
type APIServer struct {
    cache *LRUCache[string, Response]
}

func (s *APIServer) GetUser(userID string) Response {
    // Avval cache ni tekshirish
    if response, ok := s.cache.Get(userID); ok {
        log.Printf("%s foydalanuvchi uchun cache hit", userID)
        return response  // Bir zumda javob!
    }

    // Cache miss — ma'lumotlar bazasiga so'rov
    log.Printf("%s foydalanuvchi uchun cache miss - DB dan so'rov", userID)
    response := s.database.QueryUser(userID)
    s.cache.Put(userID, response)
    return response
}
\`\`\`

**Unumdorlikka ta'sir**:
- Cache hit: 0.1ms javob vaqti
- Cache miss (DB so'rovi): 50ms javob vaqti
- Cache hit bilan 500x tezroq!

**Haqiqiy hodisa**: E-commerce sayti 1000 req/sec ga ega edi. Cache siz ma'lumotlar bazasi yukni bajara olmadi (yiqildi). LRU cache bilan (90% hit rate) faqat 100 DB so'rovi/sec. Tizim barqaror qoldi.

**2. Xotira boshqaruvi — Nega LRU?**

Hamma narsani cache ga ololmaysiz — xotira cheklangan. LRU chiqarish siyosati optimal:

\`\`\`go
// Misol: Sig'imi 3 bo'lgan cache
cache := NewLRUCache[string, string](3)

// Kirish patterni: [A, B, C, A, D]
cache.Put("A", "data1")  // Cache: [A]
cache.Put("B", "data2")  // Cache: [B, A]
cache.Put("C", "data3")  // Cache: [C, B, A]

cache.Get("A")           // Cache: [A, C, B] (A boshga ko'chirildi)

cache.Put("D", "data4")  // Cache: [D, A, C] (B chiqarildi - eng kam yaqinda ishlatilgan)
\`\`\`

**Nega B chiqarildi?**
- A oxirgi ishlatildi
- C yaqinda qo'shildi
- B eng uzoq vaqt ishlatilmadi — kerak bo'lish ehtimoli eng past

**Muqobil siyosatlar va nega ular yomonroq:**

| Siyosat | Nega yomonroq | Muammo misoli |
|---------|---------------|---------------|
| FIFO | Eng eskisini chiqaradi, foydalanishga e'tibor bermaydi | Birinchi qo'shilgan mashhur element chiqariladi |
| Random | Pattern yo'q, bashorat qilib bo'lmaydi | Tez-tez ishlatiluvchi elementlar tasodifiy chiqariladi |
| LFU | Eski tez-tez elementlar hech qachon chiqarilmaydi | Kechagi mashhur element bugungi narsani bloklaydi |
| MRU | Eng yangi elementlarni chiqaradi | Qarama-qarshi, past hit rate |

**3. O(1) talabi — Nega ikkala struktura?**

**Sodda yondashuv — faqat massiv (O(n) operatsiyalar):**

\`\`\`go
// YOMON — Chiziqli qidiruv va siljishlar
type NaiveCache struct {
    items []CacheEntry
    capacity int
}

func (c *NaiveCache) Get(key string) (value string, ok bool) {
    // O(n) — massiv bo'ylab qidiruv
    for i, entry := range c.items {
        if entry.key == key {
            // O(n) — elementlarni boshga siljitish
            c.items = append([]CacheEntry{entry}, append(c.items[:i], c.items[i+1:]...)...)
            return entry.value, true
        }
    }
    return "", false
}
// Get: O(n), Put: O(n) — JUDA SEKIN!
\`\`\`

**Katta hajmda unumdorlik:**
- 1,000 element: ~500 taqqoslash har operatsiyada
- 10,000 element: ~5,000 taqqoslash har operatsiyada
- Production uchun yaroqsiz!

**Optimallashtirilgan yondashuv — Hash Map + Ikki tomonlama bog'langan ro'yxat:**

\`\`\`go
type LRUCache struct {
    cache map[string]*Node  // O(1) kalit qidiruvi
    head  *Node              // O(1) eng yaqinga kirish
    tail  *Node              // O(1) eng kamiga kirish
}

func (c *LRUCache) Get(key string) (value string, ok bool) {
    if node, exists := c.cache[key]; exists {  // O(1) qidiruv
        c.moveToFront(node)                     // O(1) pointer manipulatsiyasi
        return node.value, true
    }
    return "", false
}
// Get: O(1), Put: O(1) — MUKAMMAL!
\`\`\`

**Nega bu ishlaydi:**
- **Hash Map**: Kalit bo'yicha tugunni topish uchun O(1) qidiruv
- **Ikki tomonlama bog'langan ro'yxat**: prev/next pointer lar bilan O(1) o'chirish/qo'shish operatsiyalari
- **Birlashtirilgan**: Qidiruv HAM tartibni o'zgartirish HAM O(1)

**4. Production foydalanish holatlari**

**Holat 1: Ma'lumotlar bazasi so'rovlari cache**

\`\`\`go
type Database struct {
    queryCache *LRUCache[string, QueryResult]
}

func (db *Database) ExecuteQuery(sql string) QueryResult {
    if result, ok := db.queryCache.Get(sql); ok {
        return result  // Bir zumda!
    }

    result := db.executeExpensiveQuery(sql)
    db.queryCache.Put(sql, result)
    return result
}
\`\`\`

**Ta'sir**: 95% cache hit rate DB yukni 20x kamaytiradi.

**Holat 2: Rasm/resurs cache**

\`\`\`go
type ImageServer struct {
    imageCache *LRUCache[string, []byte]
}

func (s *ImageServer) ServeImage(imageID string) []byte {
    if data, ok := s.imageCache.Get(imageID); ok {
        return data  // Xotiradan
    }

    data := s.loadFromDisk(imageID)  // Qimmat disk I/O
    s.imageCache.Put(imageID, data)
    return data
}
\`\`\`

**Ta'sir**: Disk I/O 10ms, xotira kirishida 0.01ms (1000x tezroq).

**Holat 3: DNS cache**

\`\`\`go
type DNSResolver struct {
    cache *LRUCache[string, IPAddress]
}

func (r *DNSResolver) Resolve(domain string) IPAddress {
    if ip, ok := r.cache.Get(domain); ok {
        return ip  // Tarmoq round-trip siz
    }

    ip := r.queryDNSServer(domain)  // Tarmoq kechikishi
    r.cache.Put(domain, ip)
    return ip
}
\`\`\`

**Ta'sir**: DNS so'rovi 50ms, cache hit 0.1ms (500x tezroq).

**Holat 4: Kompilyatsiya qilingan shablonlar cache**

\`\`\`go
type TemplateEngine struct {
    cache *LRUCache[string, Template]
}

func (e *TemplateEngine) Render(name string, data interface{}) string {
    if tmpl, ok := e.cache.Get(name); ok {
        return tmpl.Execute(data)  // Kompilyatsiya qilingan shablondan foydalanish
    }

    tmpl := e.compileTemplate(name)  // Qimmat parsing
    e.cache.Put(name, tmpl)
    return tmpl.Execute(data)
}
\`\`\`

**5. Operatsion tizimlarda LRU Cache**

**CPU Cache (L1, L2, L3)**:
- Protsessorlar cache qatorlar uchun LRU-ga o'xshash siyosatlardan foydalanadi
- Yaqinda foydalanilgan xotira tez cache da qoladi
- Eng kam yaqinda foydalanilgan ma'lumotlar joy ochish uchun chiqariladi

**Sahifa almashtirish**:
- Operatsion tizimlar virtual xotira uchun LRU dan foydalanadi
- Yaqinda foydalanilgan sahifalar RAM da qoladi
- Eng kam yaqinda foydalanilgan sahifalar diskga almashtiriladi

**6. Interview da eng sevimli savol**

LRU Cache eng ommabop interview savoli, chunki u tekshiradi:
- Ma'lumotlar strukturalari bilimi (hash map, linked list)
- Algoritm dizayni (strukturalarni birlashtirish)
- Vaqt murakkablik tahlili (O(1) talabi)
- Pointer manipulatsiyasi (ikki tomonlama bog'langan ro'yxat)
- Production tizimlar haqida fikrlash (real foydalanish holatlari)

**7. Variantlar va kengaytmalar**

**TTL (Time-To-Live) LRU**:
\`\`\`go
type Node struct {
    key       string
    value     interface{}
    expiresAt time.Time
    prev, next *Node
}

func (c *LRUCache) Get(key string) (interface{}, bool) {
    if node, exists := c.cache[key]; exists {
        if time.Now().After(node.expiresAt) {
            c.evict(node)  // Muddati tugadi
            return nil, false
        }
        c.moveToFront(node)
        return node.value, true
    }
    return nil, false
}
\`\`\`

**LRU-K (K kirishlarni kuzatish)**:
- Standart LRU 1 kirishni kuzatadi (LRU-1)
- LRU-2 faqat 2 kirishdan keyin ko'taradi
- Bir martalik skanerlash foydali ma'lumotlarni chiqarishdan himoya qiladi

**8. Haqiqiy production ko'rsatkichlari**

Haqiqiy production tizimidan (e-commerce sayti):

\`\`\`
LRU Cache konfiguratsiyasi:
- Sig'im: 10,000 yozuv
- Yozuv hajmi: ~10KB
- Umumiy xotira: ~100MB

Unumdorlik:
- So'rovlar/sec: 5,000
- Cache hit rate: 92%
- P50 kechikish: 2ms (hit), 45ms (miss)
- P99 kechikish: 5ms (hit), 120ms (miss)

Ta'sir:
- Cache siz: 5,000 DB so'rovi/sec → ma'lumotlar bazasi ortiqcha yuklanish
- Cache bilan: 400 DB so'rovi/sec → silliq ishlash
- Ma'lumotlar bazasi yukining 92% kamayishi
- Oy davomida DB miqyoslashda $50,000 tejash
\`\`\`

**Asosiy xulosalar:**
- LRU Cache O(1) get va put operatsiyalarini ta'minlaydi
- Hash map (qidiruv) + ikki tomonlama bog'langan ro'yxat (tartib) ni birlashtiradi
- Sig'imga yetganda eng kam yaqinda ishlatilgan elementlarni chiqaradi
- Production unumdorligi uchun muhim (100-1000x tezlashtirish)
- Hamma joyda ishlatiladi: veb-serverlar, ma'lumotlar bazalari, operatsion tizimlar
- Xotira samarali: Faqat tez-tez so'ralgan ma'lumotlarni saqlaydi
- Ko'pgina kirish patternlari uchun optimal chiqarish siyosati`,
			solutionCode: `package datastructsx

type Node[K comparable, V any] struct {
	key   K
	value V
	prev  *Node[K, V]
	next  *Node[K, V]
}

type LRUCache[K comparable, V any] struct {
	capacity int
	cache    map[K]*Node[K, V]
	head     *Node[K, V]  // Eng yaqinda ishlatilgan
	tail     *Node[K, V]  // Eng kam yaqinda ishlatilgan
}

func NewLRUCache[K comparable, V any](capacity int) *LRUCache[K, V] {
	c := &LRUCache[K, V]{
		capacity: capacity,
		cache:    make(map[K]*Node[K, V]),
		head:     &Node[K, V]{},
		tail:     &Node[K, V]{},
	}
	// head va tail ni ulash
	c.head.next = c.tail
	c.tail.prev = c.head
	return c
}

func (c *LRUCache[K, V]) Get(key K) (V, bool) {
	if node, exists := c.cache[key]; exists {
		// Boshga ko'chirish (yaqinda ishlatilgan deb belgilash)
		c.moveToFront(node)
		return node.value, true
	}
	var zero V
	return zero, false
}

func (c *LRUCache[K, V]) Put(key K, value V) {
	if c.capacity <= 0 {
		return
	}

	if node, exists := c.cache[key]; exists {
		// Mavjud yozuvni yangilash
		node.value = value
		c.moveToFront(node)
		return
	}

	// Yangi tugun yaratish
	node := &Node[K, V]{
		key:   key,
		value: value,
	}

	// Cache ga va ro'yxat boshiga qo'shish
	c.cache[key] = node
	c.addToFront(node)

	// Sig'imga yetganda LRU ni chiqarish
	if len(c.cache) > c.capacity {
		lru := c.removeTail()
		delete(c.cache, lru.key)
	}
}

func (c *LRUCache[K, V]) Size() int {
	return len(c.cache)
}

func (c *LRUCache[K, V]) Capacity() int {
	return c.capacity
}

// Helper: Tugunni boshga ko'chirish (yaqinda ishlatilgan deb belgilash)
func (c *LRUCache[K, V]) moveToFront(node *Node[K, V]) {
	c.removeNode(node)
	c.addToFront(node)
}

// Helper: Ro'yxatdan tugunni o'chirish
func (c *LRUCache[K, V]) removeNode(node *Node[K, V]) {
	node.prev.next = node.next
	node.next.prev = node.prev
}

// Helper: Tugunni ro'yxat boshiga qo'shish
func (c *LRUCache[K, V]) addToFront(node *Node[K, V]) {
	node.prev = c.head
	node.next = c.head.next
	c.head.next.prev = node
	c.head.next = node
}

// Helper: Oxir tugunini o'chirib qaytarish (LRU)
func (c *LRUCache[K, V]) removeTail() *Node[K, V] {
	lru := c.tail.prev
	c.removeNode(lru)
	return lru
}`
		}
	}
};

export default task;
