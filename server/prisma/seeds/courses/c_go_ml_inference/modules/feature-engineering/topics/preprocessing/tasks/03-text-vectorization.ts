import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'goml-text-vectorization',
	title: 'Text Vectorization',
	difficulty: 'medium',
	tags: ['go', 'ml', 'nlp', 'vectorization'],
	estimatedTime: '30m',
	isPremium: true,
	order: 3,
	description: `# Text Vectorization

Convert text to numerical vectors for ML inference.

## Task

Implement text vectorizers:
- Bag of Words vectorization
- TF-IDF vectorization
- Token-to-index mapping
- Handle out-of-vocabulary tokens

## Example

\`\`\`go
vectorizer := NewTfidfVectorizer()
vectorizer.Fit(corpus)
vector := vectorizer.Transform("machine learning")
\`\`\``,

	initialCode: `package main

import (
	"fmt"
)

// TfidfVectorizer converts text to TF-IDF vectors
type TfidfVectorizer struct {
	// Your fields here
}

// NewTfidfVectorizer creates a TF-IDF vectorizer
func NewTfidfVectorizer() *TfidfVectorizer {
	// Your code here
	return nil
}

// Fit learns vocabulary and IDF from corpus
func (v *TfidfVectorizer) Fit(corpus []string) {
	// Your code here
}

// Transform converts text to TF-IDF vector
func (v *TfidfVectorizer) Transform(text string) []float64 {
	// Your code here
	return nil
}

// GetVocabulary returns the vocabulary
func (v *TfidfVectorizer) GetVocabulary() []string {
	// Your code here
	return nil
}

func main() {
	fmt.Println("Text Vectorization")
}`,

	solutionCode: `package main

import (
	"fmt"
	"math"
	"regexp"
	"strings"
)

// Tokenizer splits text into tokens
type Tokenizer struct {
	lowercase bool
	pattern   *regexp.Regexp
}

// NewTokenizer creates a tokenizer
func NewTokenizer() *Tokenizer {
	return &Tokenizer{
		lowercase: true,
		pattern:   regexp.MustCompile("\\\\w+"),
	}
}

// Tokenize splits text into tokens
func (t *Tokenizer) Tokenize(text string) []string {
	if t.lowercase {
		text = strings.ToLower(text)
	}
	return t.pattern.FindAllString(text, -1)
}

// BowVectorizer implements Bag of Words
type BowVectorizer struct {
	vocabulary map[string]int
	vocabList  []string
	tokenizer  *Tokenizer
}

// NewBowVectorizer creates a BoW vectorizer
func NewBowVectorizer() *BowVectorizer {
	return &BowVectorizer{
		vocabulary: make(map[string]int),
		vocabList:  make([]string, 0),
		tokenizer:  NewTokenizer(),
	}
}

// Fit builds vocabulary from corpus
func (v *BowVectorizer) Fit(corpus []string) {
	v.vocabulary = make(map[string]int)
	v.vocabList = make([]string, 0)

	for _, doc := range corpus {
		tokens := v.tokenizer.Tokenize(doc)
		for _, token := range tokens {
			if _, exists := v.vocabulary[token]; !exists {
				v.vocabulary[token] = len(v.vocabList)
				v.vocabList = append(v.vocabList, token)
			}
		}
	}
}

// Transform converts text to BoW vector
func (v *BowVectorizer) Transform(text string) []float64 {
	vector := make([]float64, len(v.vocabList))
	tokens := v.tokenizer.Tokenize(text)

	for _, token := range tokens {
		if idx, exists := v.vocabulary[token]; exists {
			vector[idx]++
		}
	}

	return vector
}

// TfidfVectorizer converts text to TF-IDF vectors
type TfidfVectorizer struct {
	vocabulary   map[string]int
	vocabList    []string
	idfScores    []float64
	tokenizer    *Tokenizer
	numDocuments int
}

// NewTfidfVectorizer creates a TF-IDF vectorizer
func NewTfidfVectorizer() *TfidfVectorizer {
	return &TfidfVectorizer{
		vocabulary: make(map[string]int),
		vocabList:  make([]string, 0),
		tokenizer:  NewTokenizer(),
	}
}

// Fit learns vocabulary and IDF from corpus
func (v *TfidfVectorizer) Fit(corpus []string) {
	v.vocabulary = make(map[string]int)
	v.vocabList = make([]string, 0)
	v.numDocuments = len(corpus)

	// Build vocabulary
	docFreq := make(map[string]int)
	for _, doc := range corpus {
		tokens := v.tokenizer.Tokenize(doc)
		seen := make(map[string]bool)

		for _, token := range tokens {
			if _, exists := v.vocabulary[token]; !exists {
				v.vocabulary[token] = len(v.vocabList)
				v.vocabList = append(v.vocabList, token)
			}
			if !seen[token] {
				docFreq[token]++
				seen[token] = true
			}
		}
	}

	// Calculate IDF
	v.idfScores = make([]float64, len(v.vocabList))
	for i, word := range v.vocabList {
		df := docFreq[word]
		v.idfScores[i] = math.Log(float64(v.numDocuments+1)/float64(df+1)) + 1
	}
}

// Transform converts text to TF-IDF vector
func (v *TfidfVectorizer) Transform(text string) []float64 {
	vector := make([]float64, len(v.vocabList))
	tokens := v.tokenizer.Tokenize(text)

	if len(tokens) == 0 {
		return vector
	}

	// Calculate TF
	tf := make(map[string]float64)
	for _, token := range tokens {
		tf[token]++
	}
	for token := range tf {
		tf[token] /= float64(len(tokens))
	}

	// Calculate TF-IDF
	for token, tfVal := range tf {
		if idx, exists := v.vocabulary[token]; exists {
			vector[idx] = tfVal * v.idfScores[idx]
		}
	}

	// L2 normalize
	var norm float64
	for _, val := range vector {
		norm += val * val
	}
	norm = math.Sqrt(norm)

	if norm > 0 {
		for i := range vector {
			vector[i] /= norm
		}
	}

	return vector
}

// TransformBatch converts multiple texts
func (v *TfidfVectorizer) TransformBatch(texts []string) [][]float64 {
	result := make([][]float64, len(texts))
	for i, text := range texts {
		result[i] = v.Transform(text)
	}
	return result
}

// GetVocabulary returns the vocabulary
func (v *TfidfVectorizer) GetVocabulary() []string {
	return v.vocabList
}

// VocabSize returns vocabulary size
func (v *TfidfVectorizer) VocabSize() int {
	return len(v.vocabList)
}

// GetIDF returns IDF score for a word
func (v *TfidfVectorizer) GetIDF(word string) float64 {
	if idx, exists := v.vocabulary[word]; exists {
		return v.idfScores[idx]
	}
	return 0
}

// CosineSimilarity calculates similarity between two vectors
func CosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func main() {
	corpus := []string{
		"machine learning is great",
		"deep learning is a subset of machine learning",
		"natural language processing uses machine learning",
	}

	vectorizer := NewTfidfVectorizer()
	vectorizer.Fit(corpus)

	fmt.Println("Vocabulary size:", vectorizer.VocabSize())
	fmt.Println("Vocabulary:", vectorizer.GetVocabulary())

	v1 := vectorizer.Transform("machine learning")
	v2 := vectorizer.Transform("deep learning")

	fmt.Println("\\nVector 'machine learning':", v1)
	fmt.Println("Cosine similarity:", CosineSimilarity(v1, v2))
}`,

	testCode: `package main

import (
	"math"
	"testing"
)

func TestTfidfVectorizer(t *testing.T) {
	corpus := []string{
		"hello world",
		"hello there",
		"world news",
	}

	vectorizer := NewTfidfVectorizer()
	vectorizer.Fit(corpus)

	if vectorizer.VocabSize() == 0 {
		t.Fatal("Vocabulary is empty")
	}
}

func TestTfidfTransform(t *testing.T) {
	corpus := []string{"hello world", "hello there"}

	vectorizer := NewTfidfVectorizer()
	vectorizer.Fit(corpus)

	vector := vectorizer.Transform("hello world")

	if len(vector) != vectorizer.VocabSize() {
		t.Errorf("Vector length mismatch: %d != %d", len(vector), vectorizer.VocabSize())
	}

	// Should have non-zero values
	hasNonZero := false
	for _, v := range vector {
		if v != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("Vector is all zeros")
	}
}

func TestTfidfNormalization(t *testing.T) {
	corpus := []string{"hello world", "test document"}
	vectorizer := NewTfidfVectorizer()
	vectorizer.Fit(corpus)

	vector := vectorizer.Transform("hello world")

	// Check L2 norm is ~1
	var norm float64
	for _, v := range vector {
		norm += v * v
	}
	norm = math.Sqrt(norm)

	if math.Abs(norm-1.0) > 0.001 && norm > 0 {
		t.Errorf("Vector not normalized, norm=%f", norm)
	}
}

func TestBowVectorizer(t *testing.T) {
	corpus := []string{"hello world", "world news"}
	vectorizer := NewBowVectorizer()
	vectorizer.Fit(corpus)

	vector := vectorizer.Transform("hello hello")

	// "hello" should have count 2
	if idx, exists := vectorizer.vocabulary["hello"]; exists {
		if vector[idx] != 2 {
			t.Errorf("Expected count 2 for 'hello', got %f", vector[idx])
		}
	}
}

func TestCosineSimilarity(t *testing.T) {
	a := []float64{1, 0, 0}
	b := []float64{1, 0, 0}

	sim := CosineSimilarity(a, b)
	if math.Abs(sim-1.0) > 0.001 {
		t.Errorf("Same vectors should have similarity 1, got %f", sim)
	}

	c := []float64{0, 1, 0}
	sim2 := CosineSimilarity(a, c)
	if sim2 != 0 {
		t.Errorf("Orthogonal vectors should have similarity 0, got %f", sim2)
	}
}

func TestTokenizer(t *testing.T) {
	tokenizer := NewTokenizer()

	tokens := tokenizer.Tokenize("Hello World")
	if len(tokens) != 2 {
		t.Errorf("Expected 2 tokens, got %d", len(tokens))
	}
	if tokens[0] != "hello" {
		t.Error("Should lowercase tokens")
	}
}

func TestGetVocabulary(t *testing.T) {
	corpus := []string{"hello world", "test"}
	vectorizer := NewTfidfVectorizer()
	vectorizer.Fit(corpus)

	vocab := vectorizer.GetVocabulary()
	if len(vocab) != 3 {
		t.Errorf("Expected 3 words in vocab, got %d", len(vocab))
	}
}

func TestTransformBatch(t *testing.T) {
	corpus := []string{"hello world", "test document"}
	vectorizer := NewTfidfVectorizer()
	vectorizer.Fit(corpus)

	texts := []string{"hello", "world"}
	batch := vectorizer.TransformBatch(texts)

	if len(batch) != 2 {
		t.Errorf("Expected 2 vectors, got %d", len(batch))
	}
	if len(batch[0]) != vectorizer.VocabSize() {
		t.Error("Vector size mismatch")
	}
}

func TestGetIDF(t *testing.T) {
	corpus := []string{"hello world", "hello there", "world news"}
	vectorizer := NewTfidfVectorizer()
	vectorizer.Fit(corpus)

	idfHello := vectorizer.GetIDF("hello")
	idfNews := vectorizer.GetIDF("news")

	// "news" appears in fewer docs, should have higher IDF
	if idfNews <= idfHello {
		t.Error("Rarer word should have higher IDF")
	}

	// Unknown word should return 0
	idfUnknown := vectorizer.GetIDF("unknown")
	if idfUnknown != 0 {
		t.Error("Unknown word should have IDF 0")
	}
}

func TestTfidfEmptyText(t *testing.T) {
	corpus := []string{"hello world"}
	vectorizer := NewTfidfVectorizer()
	vectorizer.Fit(corpus)

	vector := vectorizer.Transform("")

	for _, v := range vector {
		if v != 0 {
			t.Error("Empty text should produce zero vector")
		}
	}
}`,

	hint1: 'TF-IDF = Term Frequency × Inverse Document Frequency',
	hint2: 'L2 normalize vectors for cosine similarity comparison',

	whyItMatters: `Text vectorization enables NLP inference:

- **Numerical representation**: Convert text to model input format
- **Semantic similarity**: Compare texts via vector distance
- **Feature extraction**: Capture important words and patterns
- **Consistency**: Same vectorization at training and inference

Proper vectorization is essential for NLP models.`,

	translations: {
		ru: {
			title: 'Векторизация текста',
			description: `# Векторизация текста

Преобразование текста в числовые векторы для ML инференса.

## Задача

Реализуйте текстовые векторизаторы:
- Bag of Words векторизация
- TF-IDF векторизация
- Отображение токенов в индексы
- Обработка неизвестных токенов

## Пример

\`\`\`go
vectorizer := NewTfidfVectorizer()
vectorizer.Fit(corpus)
vector := vectorizer.Transform("machine learning")
\`\`\``,
			hint1: 'TF-IDF = Частота терма × Обратная частота документа',
			hint2: 'L2 нормализуйте векторы для сравнения косинусного сходства',
			whyItMatters: `Векторизация текста обеспечивает NLP инференс:

- **Численное представление**: Преобразование текста в формат входа модели
- **Семантическое сходство**: Сравнение текстов через расстояние векторов
- **Извлечение признаков**: Захват важных слов и паттернов
- **Консистентность**: Одинаковая векторизация при обучении и инференсе`,
		},
		uz: {
			title: 'Matn vektorizatsiyasi',
			description: `# Matn vektorizatsiyasi

ML inference uchun matnni raqamli vektorlarga aylantirish.

## Topshiriq

Matn vektorizatorlarini amalga oshiring:
- Bag of Words vektorizatsiyasi
- TF-IDF vektorizatsiyasi
- Token-indeks mosligini yaratish
- Noma'lum tokenlarni qayta ishlash

## Misol

\`\`\`go
vectorizer := NewTfidfVectorizer()
vectorizer.Fit(corpus)
vector := vectorizer.Transform("machine learning")
\`\`\``,
			hint1: 'TF-IDF = Term Frequency × Inverse Document Frequency',
			hint2: "Kosinus o'xshashligini solishtirish uchun vektorlarni L2 normallang",
			whyItMatters: `Matn vektorizatsiyasi NLP inference ni ta'minlaydi:

- **Raqamli ifodalash**: Matnni model kirish formatiga aylantirish
- **Semantik o'xshashlik**: Matnlarni vektor masofasi orqali solishtirish
- **Feature ajratib olish**: Muhim so'zlar va patternlarni ushlash
- **Izchillik**: O'qitish va inference da bir xil vektorizatsiya`,
		},
	},
};

export default task;
