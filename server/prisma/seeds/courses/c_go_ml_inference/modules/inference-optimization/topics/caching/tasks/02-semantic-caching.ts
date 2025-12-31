import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'goml-semantic-caching',
	title: 'Semantic Caching',
	difficulty: 'hard',
	tags: ['go', 'ml', 'caching', 'embeddings'],
	estimatedTime: '35m',
	isPremium: true,
	order: 2,
	description: `# Semantic Caching

Implement semantic caching using embedding similarity.

## Task

Build a semantic cache that:
- Caches based on input embedding similarity
- Finds similar cached results within threshold
- Uses approximate nearest neighbor search
- Returns cached results for semantically similar inputs

## Example

\`\`\`go
cache := NewSemanticCache(threshold: 0.95)
cache.Set(embedding, result)
// Returns cached result for similar embeddings
result, hit := cache.Get(similarEmbedding)
\`\`\``,

	initialCode: `package main

import (
	"fmt"
)

// SemanticCache caches based on embedding similarity
type SemanticCache struct {
	// Your fields here
}

// NewSemanticCache creates a semantic cache
func NewSemanticCache(threshold float64, maxSize int) *SemanticCache {
	// Your code here
	return nil
}

// Get finds a cached result with similar embedding
func (c *SemanticCache) Get(embedding []float32) ([]float32, bool) {
	// Your code here
	return nil, false
}

// Set stores a result with its embedding
func (c *SemanticCache) Set(embedding, result []float32) {
	// Your code here
}

// CosineSimilarity calculates similarity between embeddings
func CosineSimilarity(a, b []float32) float64 {
	// Your code here
	return 0
}

func main() {
	fmt.Println("Semantic Caching")
}`,

	solutionCode: `package main

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// CosineSimilarity calculates similarity between embeddings
func CosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// cacheItem holds embedding and result
type cacheItem struct {
	embedding []float32
	result    []float32
	timestamp time.Time
}

// SemanticCache caches based on embedding similarity
type SemanticCache struct {
	threshold float64
	maxSize   int
	items     []cacheItem
	mu        sync.RWMutex

	hits   int64
	misses int64
}

// NewSemanticCache creates a semantic cache
func NewSemanticCache(threshold float64, maxSize int) *SemanticCache {
	return &SemanticCache{
		threshold: threshold,
		maxSize:   maxSize,
		items:     make([]cacheItem, 0, maxSize),
	}
}

// Get finds a cached result with similar embedding
func (c *SemanticCache) Get(embedding []float32) ([]float32, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	bestSim := 0.0
	var bestResult []float32

	for _, item := range c.items {
		sim := CosineSimilarity(embedding, item.embedding)
		if sim >= c.threshold && sim > bestSim {
			bestSim = sim
			bestResult = item.result
		}
	}

	if bestResult != nil {
		c.mu.RUnlock()
		c.mu.Lock()
		c.hits++
		c.mu.Unlock()
		c.mu.RLock()

		result := make([]float32, len(bestResult))
		copy(result, bestResult)
		return result, true
	}

	c.mu.RUnlock()
	c.mu.Lock()
	c.misses++
	c.mu.Unlock()
	c.mu.RLock()

	return nil, false
}

// GetWithSimilarity returns result and similarity score
func (c *SemanticCache) GetWithSimilarity(embedding []float32) ([]float32, float64, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	bestSim := 0.0
	var bestResult []float32

	for _, item := range c.items {
		sim := CosineSimilarity(embedding, item.embedding)
		if sim >= c.threshold && sim > bestSim {
			bestSim = sim
			bestResult = item.result
		}
	}

	if bestResult != nil {
		result := make([]float32, len(bestResult))
		copy(result, bestResult)
		return result, bestSim, true
	}

	return nil, 0, false
}

// Set stores a result with its embedding
func (c *SemanticCache) Set(embedding, result []float32) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Check if similar embedding exists
	for i, item := range c.items {
		if CosineSimilarity(embedding, item.embedding) >= c.threshold {
			// Update existing entry
			c.items[i].result = make([]float32, len(result))
			copy(c.items[i].result, result)
			c.items[i].timestamp = time.Now()
			return
		}
	}

	// Evict oldest if at capacity
	if len(c.items) >= c.maxSize {
		oldest := 0
		for i, item := range c.items {
			if item.timestamp.Before(c.items[oldest].timestamp) {
				oldest = i
			}
		}
		c.items = append(c.items[:oldest], c.items[oldest+1:]...)
	}

	// Add new entry
	newItem := cacheItem{
		embedding: make([]float32, len(embedding)),
		result:    make([]float32, len(result)),
		timestamp: time.Now(),
	}
	copy(newItem.embedding, embedding)
	copy(newItem.result, result)
	c.items = append(c.items, newItem)
}

// Size returns current cache size
func (c *SemanticCache) Size() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.items)
}

// Stats returns cache statistics
func (c *SemanticCache) Stats() (hits, misses int64, hitRate float64) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	total := c.hits + c.misses
	if total > 0 {
		hitRate = float64(c.hits) / float64(total)
	}
	return c.hits, c.misses, hitRate
}

// Clear removes all entries
func (c *SemanticCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.items = c.items[:0]
}

// LSHSemanticCache uses locality-sensitive hashing for faster lookups
type LSHSemanticCache struct {
	*SemanticCache
	numHashes   int
	buckets     map[string][]int
	projections [][]float32
}

// NewLSHSemanticCache creates an LSH-based semantic cache
func NewLSHSemanticCache(threshold float64, maxSize, numHashes, dim int) *LSHSemanticCache {
	cache := &LSHSemanticCache{
		SemanticCache: NewSemanticCache(threshold, maxSize),
		numHashes:     numHashes,
		buckets:       make(map[string][]int),
		projections:   make([][]float32, numHashes),
	}

	// Initialize random projections
	for i := 0; i < numHashes; i++ {
		cache.projections[i] = make([]float32, dim)
		for j := 0; j < dim; j++ {
			// Simple random projection (should use proper random in production)
			cache.projections[i][j] = float32(i*j%17-8) / 8.0
		}
	}

	return cache
}

// hash computes LSH hash for embedding
func (c *LSHSemanticCache) hash(embedding []float32) string {
	bits := make([]byte, c.numHashes)
	for i, proj := range c.projections {
		var dot float32
		for j := 0; j < len(proj) && j < len(embedding); j++ {
			dot += proj[j] * embedding[j]
		}
		if dot >= 0 {
			bits[i] = '1'
		} else {
			bits[i] = '0'
		}
	}
	return string(bits)
}

func main() {
	cache := NewSemanticCache(0.9, 100)

	// Create embeddings
	emb1 := []float32{1.0, 0.0, 0.0}
	emb2 := []float32{0.99, 0.1, 0.0}  // Similar to emb1
	emb3 := []float32{0.0, 1.0, 0.0}   // Different

	result1 := []float32{2.0, 0.0, 0.0}

	// Cache result for emb1
	cache.Set(emb1, result1)

	// Query with similar embedding
	result, hit := cache.Get(emb2)
	fmt.Printf("Similar query: hit=%v, result=%v\\n", hit, result)
	fmt.Printf("Similarity: %.4f\\n", CosineSimilarity(emb1, emb2))

	// Query with different embedding
	result, hit = cache.Get(emb3)
	fmt.Printf("Different query: hit=%v, result=%v\\n", hit, result)
	fmt.Printf("Similarity: %.4f\\n", CosineSimilarity(emb1, emb3))

	hits, misses, hitRate := cache.Stats()
	fmt.Printf("Stats: hits=%d, misses=%d, hitRate=%.2f\\n", hits, misses, hitRate)
}`,

	testCode: `package main

import (
	"math"
	"testing"
)

func TestCosineSimilarity(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{1, 0, 0}

	sim := CosineSimilarity(a, b)
	if math.Abs(sim-1.0) > 0.001 {
		t.Errorf("Same vectors should have similarity 1, got %f", sim)
	}

	c := []float32{0, 1, 0}
	sim2 := CosineSimilarity(a, c)
	if sim2 != 0 {
		t.Errorf("Orthogonal vectors should have similarity 0, got %f", sim2)
	}
}

func TestSemanticCache(t *testing.T) {
	cache := NewSemanticCache(0.9, 100)

	emb := []float32{1, 0, 0}
	result := []float32{2, 0, 0}

	cache.Set(emb, result)

	retrieved, hit := cache.Get(emb)
	if !hit {
		t.Error("Should hit cache")
	}
	if len(retrieved) != 3 {
		t.Errorf("Expected 3 elements, got %d", len(retrieved))
	}
}

func TestSemanticCacheSimilarity(t *testing.T) {
	cache := NewSemanticCache(0.9, 100)

	emb1 := []float32{1, 0, 0}
	emb2 := []float32{0.95, 0.05, 0}
	result := []float32{2, 0, 0}

	cache.Set(emb1, result)

	// Should hit for similar embedding
	_, hit := cache.Get(emb2)
	similarity := CosineSimilarity(emb1, emb2)

	if similarity >= 0.9 && !hit {
		t.Error("Should hit for similar embedding")
	}
}

func TestSemanticCacheMiss(t *testing.T) {
	cache := NewSemanticCache(0.95, 100)

	emb1 := []float32{1, 0, 0}
	emb2 := []float32{0, 1, 0}
	result := []float32{2, 0, 0}

	cache.Set(emb1, result)

	_, hit := cache.Get(emb2)
	if hit {
		t.Error("Should miss for orthogonal embedding")
	}
}

func TestSemanticCacheEviction(t *testing.T) {
	cache := NewSemanticCache(0.99, 2)

	cache.Set([]float32{1, 0, 0}, []float32{1})
	cache.Set([]float32{0, 1, 0}, []float32{2})
	cache.Set([]float32{0, 0, 1}, []float32{3})

	if cache.Size() != 2 {
		t.Errorf("Cache should have 2 items, got %d", cache.Size())
	}
}

func TestGetWithSimilarity(t *testing.T) {
	cache := NewSemanticCache(0.9, 100)

	emb := []float32{1, 0, 0}
	result := []float32{2, 0, 0}

	cache.Set(emb, result)

	_, sim, hit := cache.GetWithSimilarity(emb)
	if !hit {
		t.Error("Should hit cache")
	}
	if math.Abs(sim-1.0) > 0.001 {
		t.Errorf("Exact match should have similarity 1, got %f", sim)
	}
}

func TestCosineSimilarityEdgeCases(t *testing.T) {
	// Different lengths
	a := []float32{1, 0}
	b := []float32{1, 0, 0}
	sim := CosineSimilarity(a, b)
	if sim != 0 {
		t.Errorf("Different length vectors should return 0, got %f", sim)
	}

	// Empty vectors
	empty := []float32{}
	sim2 := CosineSimilarity(empty, empty)
	if sim2 != 0 {
		t.Errorf("Empty vectors should return 0, got %f", sim2)
	}
}

func TestSemanticCacheStats(t *testing.T) {
	cache := NewSemanticCache(0.95, 100)

	emb := []float32{1, 0, 0}
	cache.Set(emb, []float32{1})

	cache.Get(emb)                 // hit
	cache.Get([]float32{0, 1, 0})  // miss

	hits, misses, hitRate := cache.Stats()
	if hits != 1 {
		t.Errorf("Expected 1 hit, got %d", hits)
	}
	if misses != 1 {
		t.Errorf("Expected 1 miss, got %d", misses)
	}
	if math.Abs(hitRate-0.5) > 0.01 {
		t.Errorf("Expected 0.5 hit rate, got %f", hitRate)
	}
}

func TestSemanticCacheClear(t *testing.T) {
	cache := NewSemanticCache(0.9, 100)

	cache.Set([]float32{1, 0, 0}, []float32{1})
	cache.Set([]float32{0, 1, 0}, []float32{2})

	if cache.Size() != 2 {
		t.Errorf("Expected size 2, got %d", cache.Size())
	}

	cache.Clear()

	if cache.Size() != 0 {
		t.Errorf("Expected size 0 after clear, got %d", cache.Size())
	}
}

func TestSemanticCacheUpdate(t *testing.T) {
	cache := NewSemanticCache(0.99, 100)

	emb := []float32{1, 0, 0}
	cache.Set(emb, []float32{1})
	cache.Set(emb, []float32{2})  // Update same embedding

	if cache.Size() != 1 {
		t.Errorf("Should still have 1 item after update, got %d", cache.Size())
	}

	result, hit := cache.Get(emb)
	if !hit || result[0] != 2 {
		t.Error("Should return updated value")
	}
}`,

	hint1: 'Cosine similarity measures angle between vectors independent of magnitude',
	hint2: 'Consider LSH for O(1) approximate nearest neighbor lookup',

	whyItMatters: `Semantic caching enables intelligent result reuse:

- **Similar queries**: Users often ask similar questions
- **Paraphrase handling**: Different words, same meaning
- **Embedding models**: Leverage semantic understanding
- **Efficiency gains**: Higher cache hit rates than exact matching

Semantic caching is especially valuable for NLP inference.`,

	translations: {
		ru: {
			title: 'Семантическое кэширование',
			description: `# Семантическое кэширование

Реализуйте семантическое кэширование с использованием схожести эмбеддингов.

## Задача

Создайте семантический кэш:
- Кэширование на основе схожести эмбеддингов входа
- Поиск похожих кэшированных результатов в пределах порога
- Использование приближенного поиска ближайших соседей
- Возврат кэшированных результатов для семантически похожих входов

## Пример

\`\`\`go
cache := NewSemanticCache(threshold: 0.95)
cache.Set(embedding, result)
// Returns cached result for similar embeddings
result, hit := cache.Get(similarEmbedding)
\`\`\``,
			hint1: 'Косинусное сходство измеряет угол между векторами независимо от величины',
			hint2: 'Рассмотрите LSH для O(1) приближенного поиска ближайшего соседа',
			whyItMatters: `Семантическое кэширование обеспечивает интеллектуальное повторное использование результатов:

- **Похожие запросы**: Пользователи часто задают похожие вопросы
- **Обработка перефразирования**: Разные слова, одно значение
- **Модели эмбеддингов**: Использование семантического понимания
- **Повышение эффективности**: Более высокий процент попаданий в кэш чем при точном совпадении`,
		},
		uz: {
			title: 'Semantik keshlash',
			description: `# Semantik keshlash

Embedding o'xshashligidan foydalanib semantik keshlashni amalga oshiring.

## Topshiriq

Semantik kesh yarating:
- Kirish embedding o'xshashligi asosida keshlash
- Chegara ichida o'xshash keshlangan natijalarni topish
- Taxminiy eng yaqin qo'shni qidiruvidan foydalanish
- Semantik jihatdan o'xshash kirishlar uchun keshlangan natijalarni qaytarish

## Misol

\`\`\`go
cache := NewSemanticCache(threshold: 0.95)
cache.Set(embedding, result)
// Returns cached result for similar embeddings
result, hit := cache.Get(similarEmbedding)
\`\`\``,
			hint1: "Kosinus o'xshashligi vektorlar orasidagi burchakni kattalikdan mustaqil ravishda o'lchaydi",
			hint2: "O(1) taxminiy eng yaqin qo'shni qidirish uchun LSH ni ko'rib chiqing",
			whyItMatters: `Semantik keshlash aqlli natijalarni qayta ishlatishni ta'minlaydi:

- **O'xshash so'rovlar**: Foydalanuvchilar ko'pincha o'xshash savollar beradi
- **Parafraz qayta ishlash**: Turli so'zlar, bir xil ma'no
- **Embedding modellari**: Semantik tushunishdan foydalanish
- **Samaradorlik yutug'i**: Aniq moslashishdan ko'ra yuqoriroq kesh urish darajasi`,
		},
	},
};

export default task;
