import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-solid-lsp-advanced',
	title: 'Liskov Substitution Principle - Advanced',
	difficulty: 'hard',
	tags: ['go', 'solid', 'lsp', 'polymorphism', 'advanced'],
	estimatedTime: '40m',
	isPremium: false,
	youtubeUrl: '',
	description: `Apply LSP to a complex file storage system with different capabilities.

**Current Problem:**

A FileStorage hierarchy where S3Storage violates LSP by not supporting local file operations, causing runtime errors.

**Your task:**

Refactor to follow LSP by segregating capabilities:

1. **Storage interface** - Base contract (Upload, Download)
2. **LocalStorage interface** - Extends Storage, adds GetLocalPath
3. **CloudStorage interface** - Extends Storage, adds GetURL
4. **DiskStorage** - Implements Storage and LocalStorage
5. **S3Storage** - Implements Storage and CloudStorage
6. **StorageManager** - Works with appropriate interface types

**Key Concepts:**
- **Interface Segregation with LSP**: Don't force implementations to have unsupported methods
- **Capability-based design**: Interfaces represent what objects can do
- **Type safety**: Compiler prevents LSP violations

**Example Usage:**

\`\`\`go
// All storage types support basic operations
storages := []Storage{
    &DiskStorage{BasePath: "/data"},
    &S3Storage{Bucket: "my-bucket"},
}

for _, storage := range storages {
    storage.Upload("file.txt", []byte("data"))  // works for all
    storage.Download("file.txt")  // works for all
}

// Only local storage has local paths
local := &DiskStorage{BasePath: "/data"}
path := local.GetLocalPath("file.txt")  // OK

// Only cloud storage has URLs
cloud := &S3Storage{Bucket: "my-bucket"}
url := cloud.GetURL("file.txt")  // OK
\`\`\`

**Real-World Impact:**
- Prevents runtime errors from unsupported operations
- Makes capability explicit in type system
- Enables safe polymorphism
- Clear contracts for each storage type`,
	initialCode: `package principles

import "fmt"

type FileStorage interface {
}

type DiskStorage struct {
	BasePath string
}

func (ds *DiskStorage) Upload(filename string, data []byte) error {
	return nil
}

func (ds *DiskStorage) Download(filename string) ([]byte, error) {
}

func (ds *DiskStorage) GetLocalPath(filename string) string {
	return ds.BasePath + "/" + filename  // OK for disk storage
}

type S3Storage struct {
	Bucket string
}

func (s3 *S3Storage) Upload(filename string, data []byte) error {
	return nil
}

func (s3 *S3Storage) Download(filename string) ([]byte, error) {
}

func (s3 *S3Storage) GetLocalPath(filename string) string {
}

type Storage interface {
}

type LocalStorage interface {
}

type CloudStorage interface {
}

type DiskStorageRefactored struct {
	BasePath string
}

func (ds *DiskStorageRefactored) Upload(filename string, data []byte) error {
}

func (ds *DiskStorageRefactored) Download(filename string) ([]byte, error) {
}

func (ds *DiskStorageRefactored) GetLocalPath(filename string) string {
}

type S3StorageRefactored struct {
	Bucket string
}

func (s3 *S3StorageRefactored) Upload(filename string, data []byte) error {
}

func (s3 *S3StorageRefactored) Download(filename string) ([]byte, error) {
}

func (s3 *S3StorageRefactored) GetURL(filename string) string {
}

func BackupFiles(storage Storage, files map[string][]byte) error {
}

func GetLocalFiles(storage LocalStorage, filenames []string) []string {
}

func GetCloudURLs(storage CloudStorage, filenames []string) []string {
}`,
	solutionCode: `package principles

import "fmt"

// Storage - base interface defining capabilities ALL storage types support
// LSP compliant: all implementations can fulfill this contract
type Storage interface {
	Upload(filename string, data []byte) error
	Download(filename string) ([]byte, error)
}

// LocalStorage - interface for storage with local file system access
// Only implementations with actual local paths implement this
type LocalStorage interface {
	Storage	// compose Storage - local storage is also storage
	GetLocalPath(filename string) string	// additional capability
}

// CloudStorage - interface for storage with URL-based access
// Only implementations with URLs implement this
type CloudStorage interface {
	Storage	// compose Storage - cloud storage is also storage
	GetURL(filename string) string	// additional capability
}

// DiskStorageRefactored implements both Storage and LocalStorage
// LSP compliant: has actual local paths
type DiskStorageRefactored struct {
	BasePath string	// local file system path
}

// Upload saves file to local disk
func (ds *DiskStorageRefactored) Upload(filename string, data []byte) error {
	fmt.Printf("Uploading %s to disk: %s\\n", filename, ds.BasePath)
	// In production: ioutil.WriteFile(ds.BasePath+"/"+filename, data, 0644)
	return nil	// upload successful
}

// Download reads file from local disk
func (ds *DiskStorageRefactored) Download(filename string) ([]byte, error) {
	fmt.Printf("Downloading %s from disk\\n", filename)
	// In production: return ioutil.ReadFile(ds.BasePath + "/" + filename)
	return []byte("disk data"), nil	// return file content
}

// GetLocalPath returns file system path
// Only available for local storage - makes sense!
func (ds *DiskStorageRefactored) GetLocalPath(filename string) string {
	return ds.BasePath + "/" + filename	// construct full path
}

// S3StorageRefactored implements both Storage and CloudStorage
// LSP compliant: doesn't pretend to have local paths
type S3StorageRefactored struct {
	Bucket string	// S3 bucket name
}

// Upload saves file to S3 bucket
func (s3 *S3StorageRefactored) Upload(filename string, data []byte) error {
	fmt.Printf("Uploading %s to S3 bucket: %s\\n", filename, s3.Bucket)
	// In production: use AWS SDK to upload to S3
	return nil	// upload successful
}

// Download retrieves file from S3 bucket
func (s3 *S3StorageRefactored) Download(filename string) ([]byte, error) {
	fmt.Printf("Downloading %s from S3\\n", filename)
	// In production: use AWS SDK to download from S3
	return []byte("s3 data"), nil	// return file content
}

// GetURL returns public URL for S3 object
// Only available for cloud storage - makes sense!
func (s3 *S3StorageRefactored) GetURL(filename string) string {
	return fmt.Sprintf("https://%s.s3.amazonaws.com/%s", s3.Bucket, filename)
}

// BackupFiles works with ANY storage type - LSP compliant
func BackupFiles(storage Storage, files map[string][]byte) error {
	for filename, data := range files {	// iterate all files
		if err := storage.Upload(filename, data); err != nil {	// upload file
			return err	// return error if upload fails
		}
	}
	return nil	// all files backed up successfully
}

// GetLocalFiles works only with LocalStorage - type safe
func GetLocalFiles(storage LocalStorage, filenames []string) []string {
	var paths []string
	for _, filename := range filenames {	// iterate filenames
		path := storage.GetLocalPath(filename)	// get local path
		paths = append(paths, path)	// collect paths
	}
	return paths	// return all local paths
}

// GetCloudURLs works only with CloudStorage - type safe
func GetCloudURLs(storage CloudStorage, filenames []string) []string {
	var urls []string
	for _, filename := range filenames {	// iterate filenames
		url := storage.GetURL(filename)	// get URL
		urls = append(urls, url)	// collect URLs
	}
	return urls	// return all URLs
}

// Type system enforces LSP:
// disk := &DiskStorageRefactored{BasePath: "/data"}
// GetLocalFiles(disk, files)  // OK - DiskStorage implements LocalStorage
// GetCloudURLs(disk, files)  // COMPILE ERROR - DiskStorage doesn't implement CloudStorage
//
// s3 := &S3StorageRefactored{Bucket: "my-bucket"}
// GetCloudURLs(s3, files)  // OK - S3Storage implements CloudStorage
// GetLocalFiles(s3, files)  // COMPILE ERROR - S3Storage doesn't implement LocalStorage`,
	hint1: `For DiskStorage methods: Upload and Download should print messages and return nil/data. GetLocalPath should return ds.BasePath + "/" + filename. For S3Storage: Upload and Download print messages, GetURL returns a formatted S3 URL string.`,
	hint2: `For BackupFiles, loop through files map and call storage.Upload for each. For GetLocalFiles, loop through filenames and call storage.GetLocalPath for each, collecting results. For GetCloudURLs, loop through filenames and call storage.GetURL for each.`,
	testCode: `package principles

import (
	"strings"
	"testing"
)

// Test1: DiskStorage.Upload returns nil
func Test1(t *testing.T) {
	ds := &DiskStorageRefactored{BasePath: "/data"}
	err := ds.Upload("test.txt", []byte("data"))
	if err != nil {
		t.Errorf("Upload error: %v", err)
	}
}

// Test2: DiskStorage.Download returns data
func Test2(t *testing.T) {
	ds := &DiskStorageRefactored{BasePath: "/data"}
	data, err := ds.Download("test.txt")
	if err != nil {
		t.Errorf("Download error: %v", err)
	}
	if len(data) == 0 {
		t.Error("Download returned empty data")
	}
}

// Test3: DiskStorage.GetLocalPath returns correct path
func Test3(t *testing.T) {
	ds := &DiskStorageRefactored{BasePath: "/data"}
	path := ds.GetLocalPath("test.txt")
	if path != "/data/test.txt" {
		t.Errorf("path = %q, want /data/test.txt", path)
	}
}

// Test4: S3Storage.Upload returns nil
func Test4(t *testing.T) {
	s3 := &S3StorageRefactored{Bucket: "my-bucket"}
	err := s3.Upload("test.txt", []byte("data"))
	if err != nil {
		t.Errorf("Upload error: %v", err)
	}
}

// Test5: S3Storage.GetURL returns valid URL
func Test5(t *testing.T) {
	s3 := &S3StorageRefactored{Bucket: "my-bucket"}
	url := s3.GetURL("test.txt")
	if !strings.Contains(url, "my-bucket") {
		t.Errorf("URL should contain bucket name, got: %s", url)
	}
	if !strings.Contains(url, "test.txt") {
		t.Errorf("URL should contain filename, got: %s", url)
	}
}

// Test6: BackupFiles works with any Storage
func Test6(t *testing.T) {
	disk := &DiskStorageRefactored{BasePath: "/data"}
	files := map[string][]byte{
		"file1.txt": []byte("data1"),
		"file2.txt": []byte("data2"),
	}
	err := BackupFiles(disk, files)
	if err != nil {
		t.Errorf("BackupFiles error: %v", err)
	}
}

// Test7: GetLocalFiles returns correct paths
func Test7(t *testing.T) {
	disk := &DiskStorageRefactored{BasePath: "/data"}
	filenames := []string{"a.txt", "b.txt"}
	paths := GetLocalFiles(disk, filenames)
	if len(paths) != 2 {
		t.Errorf("got %d paths, want 2", len(paths))
	}
}

// Test8: GetCloudURLs returns correct URLs
func Test8(t *testing.T) {
	s3 := &S3StorageRefactored{Bucket: "bucket"}
	filenames := []string{"a.txt", "b.txt"}
	urls := GetCloudURLs(s3, filenames)
	if len(urls) != 2 {
		t.Errorf("got %d urls, want 2", len(urls))
	}
}

// Test9: Both storage types implement Storage interface
func Test9(t *testing.T) {
	var storage Storage
	storage = &DiskStorageRefactored{BasePath: "/data"}
	_ = storage
	storage = &S3StorageRefactored{Bucket: "bucket"}
	_ = storage
}

// Test10: Empty files map works
func Test10(t *testing.T) {
	disk := &DiskStorageRefactored{BasePath: "/data"}
	err := BackupFiles(disk, map[string][]byte{})
	if err != nil {
		t.Errorf("BackupFiles with empty map error: %v", err)
	}
}
`,
	whyItMatters: `Advanced LSP prevents runtime errors by making capabilities explicit in the type system.

**Why Advanced LSP Matters:**

**1. Real Production Bug: Payment Processing**

\`\`\`go
// VIOLATES LSP - causes production bug
type PaymentProcessor interface {
	Charge(amount float64) error
	Refund(transactionID string, amount float64) error
}

type StripeProcessor struct{}
func (sp *StripeProcessor) Charge(amount float64) error {
	// charges via Stripe
	return nil
}
func (sp *StripeProcessor) Refund(transactionID string, amount float64) error {
	// refunds via Stripe
	return nil
}

// Gift card payments can't be refunded - VIOLATES LSP!
type GiftCardProcessor struct{}
func (gc *GiftCardProcessor) Charge(amount float64) error {
	// charges gift card
	return nil
}
func (gc *GiftCardProcessor) Refund(transactionID string, amount float64) error {
	// Gift cards can't be refunded!
	return fmt.Errorf("gift cards cannot be refunded")
	// PRODUCTION BUG: refund button shown but fails at runtime!
}

// FOLLOWS LSP - separate interfaces
type Chargeable interface {
	Charge(amount float64) error
}

type Refundable interface {
	Refund(transactionID string, amount float64) error
}

type FullPaymentProcessor interface {
	Chargeable
	Refundable
}

type StripeProcessorFixed struct{}  // implements FullPaymentProcessor
type GiftCardProcessorFixed struct{}  // implements only Chargeable

// UI can check at compile time if refund is available
func ShowRefundButton(processor interface{}) bool {
	_, ok := processor.(Refundable)
	return ok  // only show for Refundable processors
}
\`\`\``,
	order: 5,
	translations: {
		ru: {
			title: 'Принцип подстановки Лисков - Продвинутый',
			description: `Примените LSP к сложной системе хранения файлов с различными возможностями.`,
			hint1: `Для методов DiskStorage: Upload и Download должны выводить сообщения и возвращать nil/data. GetLocalPath должен возвращать ds.BasePath + "/" + filename.`,
			hint2: `Для BackupFiles переберите map files и вызовите storage.Upload для каждого. Для GetLocalFiles переберите filenames и вызовите storage.GetLocalPath, собирая результаты.`,
			whyItMatters: `Продвинутый LSP предотвращает ошибки времени выполнения, делая возможности явными в системе типов.`,
			solutionCode: `package principles

import "fmt"

type Storage interface {
	Upload(filename string, data []byte) error
	Download(filename string) ([]byte, error)
}

type LocalStorage interface {
	Storage
	GetLocalPath(filename string) string
}

type CloudStorage interface {
	Storage
	GetURL(filename string) string
}

type DiskStorageRefactored struct {
	BasePath string
}

func (ds *DiskStorageRefactored) Upload(filename string, data []byte) error {
	fmt.Printf("Загрузка %s на диск: %s\\n", filename, ds.BasePath)
	return nil
}

func (ds *DiskStorageRefactored) Download(filename string) ([]byte, error) {
	fmt.Printf("Скачивание %s с диска\\n", filename)
	return []byte("данные с диска"), nil
}

func (ds *DiskStorageRefactored) GetLocalPath(filename string) string {
	return ds.BasePath + "/" + filename
}

type S3StorageRefactored struct {
	Bucket string
}

func (s3 *S3StorageRefactored) Upload(filename string, data []byte) error {
	fmt.Printf("Загрузка %s в S3 bucket: %s\\n", filename, s3.Bucket)
	return nil
}

func (s3 *S3StorageRefactored) Download(filename string) ([]byte, error) {
	fmt.Printf("Скачивание %s из S3\\n", filename)
	return []byte("данные из s3"), nil
}

func (s3 *S3StorageRefactored) GetURL(filename string) string {
	return fmt.Sprintf("https://%s.s3.amazonaws.com/%s", s3.Bucket, filename)
}

func BackupFiles(storage Storage, files map[string][]byte) error {
	for filename, data := range files {
		if err := storage.Upload(filename, data); err != nil {
			return err
		}
	}
	return nil
}

func GetLocalFiles(storage LocalStorage, filenames []string) []string {
	var paths []string
	for _, filename := range filenames {
		paths = append(paths, storage.GetLocalPath(filename))
	}
	return paths
}

func GetCloudURLs(storage CloudStorage, filenames []string) []string {
	var urls []string
	for _, filename := range filenames {
		urls = append(urls, storage.GetURL(filename))
	}
	return urls
}`
		},
		uz: {
			title: 'Liskov almashtirish printsipi - Kengaytirilgan',
			description: `Turli imkoniyatlarga ega murakkab fayl saqlash tizimiga LSP ni qo'llang.`,
			hint1: `DiskStorage metodlari uchun: Upload va Download xabarlarni chiqarishi va nil/data qaytarishi kerak. GetLocalPath ds.BasePath + "/" + filename qaytarishi kerak.`,
			hint2: `BackupFiles uchun files map ni aylanib o'ting va har biri uchun storage.Upload ni chaqiring. GetLocalFiles uchun filenames ni aylanib o'ting va storage.GetLocalPath ni chaqiring, natijalarni yig'ing.`,
			whyItMatters: `Kengaytirilgan LSP imkoniyatlarni tur tizimida aniq qilish orqali runtime xatolarining oldini oladi.`,
			solutionCode: `package principles

import "fmt"

type Storage interface {
	Upload(filename string, data []byte) error
	Download(filename string) ([]byte, error)
}

type LocalStorage interface {
	Storage
	GetLocalPath(filename string) string
}

type CloudStorage interface {
	Storage
	GetURL(filename string) string
}

type DiskStorageRefactored struct {
	BasePath string
}

func (ds *DiskStorageRefactored) Upload(filename string, data []byte) error {
	fmt.Printf("%s diskga yuklanmoqda: %s\\n", filename, ds.BasePath)
	return nil
}

func (ds *DiskStorageRefactored) Download(filename string) ([]byte, error) {
	fmt.Printf("%s diskdan yuklab olinmoqda\\n", filename)
	return []byte("disk ma'lumotlari"), nil
}

func (ds *DiskStorageRefactored) GetLocalPath(filename string) string {
	return ds.BasePath + "/" + filename
}

type S3StorageRefactored struct {
	Bucket string
}

func (s3 *S3StorageRefactored) Upload(filename string, data []byte) error {
	fmt.Printf("%s S3 bucketga yuklanmoqda: %s\\n", filename, s3.Bucket)
	return nil
}

func (s3 *S3StorageRefactored) Download(filename string) ([]byte, error) {
	fmt.Printf("%s S3 dan yuklab olinmoqda\\n", filename)
	return []byte("s3 ma'lumotlari"), nil
}

func (s3 *S3StorageRefactored) GetURL(filename string) string {
	return fmt.Sprintf("https://%s.s3.amazonaws.com/%s", s3.Bucket, filename)
}

func BackupFiles(storage Storage, files map[string][]byte) error {
	for filename, data := range files {
		if err := storage.Upload(filename, data); err != nil {
			return err
		}
	}
	return nil
}

func GetLocalFiles(storage LocalStorage, filenames []string) []string {
	var paths []string
	for _, filename := range filenames {
		paths = append(paths, storage.GetLocalPath(filename))
	}
	return paths
}

func GetCloudURLs(storage CloudStorage, filenames []string) []string {
	var urls []string
	for _, filename := range filenames {
		urls = append(urls, storage.GetURL(filename))
	}
	return urls
}`
		}
	}
};

export default task;
