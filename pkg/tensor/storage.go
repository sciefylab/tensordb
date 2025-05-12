package tensor

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/edsrzf/mmap-go"
)

type TensorMetadata struct {
	Name     string
	Shape    []int
	DataType string
	Strides  []int
	// NumDimensions int // Bisa ditambahkan jika ingin disimpan, atau dihitung on-the-fly
}

// InMemoryIndex adalah struktur data untuk indeks metadata tensor dalam memori.
type InMemoryIndex struct {
	// Key: DataType (string), Value: set nama tensor (map[tensorName]struct{})
	ByDataType map[string]map[string]struct{}
	// Key: NumDimensions (int), Value: set nama tensor (map[tensorName]struct{})
	ByNumDimensions map[int]map[string]struct{}
	// Key: tensorName, Value: pointer ke metadata (untuk akses cepat jika sudah dimuat)
	// Ini opsional dan bisa menambah kompleksitas sinkronisasi.
	// Untuk saat ini, kita akan fokus pada pencarian nama, lalu muat metadata dari disk.
	// AllTensorMetadata map[string]*TensorMetadata

	mu sync.RWMutex // Melindungi akses ke semua peta indeks
}

// NewInMemoryIndex membuat instance baru dari InMemoryIndex.
func NewInMemoryIndex() *InMemoryIndex {
	return &InMemoryIndex{
		ByDataType:      make(map[string]map[string]struct{}),
		ByNumDimensions: make(map[int]map[string]struct{}),
		// AllTensorMetadata: make(map[string]*TensorMetadata),
	}
}

// Add menambahkan atau memperbarui metadata tensor dalam indeks.
// Fungsi ini harus dipanggil setiap kali tensor dibuat atau metadatanya diubah.
func (idx *InMemoryIndex) Add(metadata *TensorMetadata) {
	if metadata == nil {
		return
	}
	idx.mu.Lock()
	defer idx.mu.Unlock()

	tensorName := metadata.Name
	dataType := metadata.DataType
	numDimensions := len(metadata.Shape)
	if len(metadata.Shape) == 1 && metadata.Shape[0] == 0 { // Representasi skalar dari parser lama mungkin [0]
		numDimensions = 0 // Skalar sejati memiliki 0 dimensi
	}
	if len(metadata.Shape) == 0 { // Representasi skalar yang lebih baik adalah shape kosong
		numDimensions = 0
	}

	// Tambahkan ke indeks ByDataType
	if _, ok := idx.ByDataType[dataType]; !ok {
		idx.ByDataType[dataType] = make(map[string]struct{})
	}
	idx.ByDataType[dataType][tensorName] = struct{}{}

	// Tambahkan ke indeks ByNumDimensions
	if _, ok := idx.ByNumDimensions[numDimensions]; !ok {
		idx.ByNumDimensions[numDimensions] = make(map[string]struct{})
	}
	idx.ByNumDimensions[numDimensions][tensorName] = struct{}{}

	// idx.AllTensorMetadata[tensorName] = metadata // Opsional
}

// Remove menghapus tensor dari indeks.
// Fungsi ini harus dipanggil jika tensor dihapus.
func (idx *InMemoryIndex) Remove(metadata *TensorMetadata) {
	if metadata == nil {
		return
	}
	idx.mu.Lock()
	defer idx.mu.Unlock()

	tensorName := metadata.Name
	dataType := metadata.DataType
	numDimensions := len(metadata.Shape)
	if len(metadata.Shape) == 1 && metadata.Shape[0] == 0 {
		numDimensions = 0
	}
	if len(metadata.Shape) == 0 {
		numDimensions = 0
	}

	if names, ok := idx.ByDataType[dataType]; ok {
		delete(names, tensorName)
		if len(names) == 0 {
			delete(idx.ByDataType, dataType)
		}
	}

	if names, ok := idx.ByNumDimensions[numDimensions]; ok {
		delete(names, tensorName)
		if len(names) == 0 {
			delete(idx.ByNumDimensions, numDimensions)
		}
	}
	// delete(idx.AllTensorMetadata, tensorName) // Opsional
}

// Query mencari nama tensor yang cocok dengan kriteria filter.
// filterNumDimensions: -1 berarti tidak ada filter berdasarkan NumDimensions.
func (idx *InMemoryIndex) Query(filterDataType string, filterNumDimensions int) []string {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	var resultNames []string
	candidateSets := make([]map[string]struct{}, 0, 2)

	if filterDataType != "" {
		if names, ok := idx.ByDataType[filterDataType]; ok {
			candidateSets = append(candidateSets, names)
		} else {
			return []string{} // Tidak ada tensor dengan tipe data ini
		}
	}

	if filterNumDimensions != -1 { // Asumsi -1 berarti "jangan filter berdasarkan ini"
		if names, ok := idx.ByNumDimensions[filterNumDimensions]; ok {
			candidateSets = append(candidateSets, names)
		} else {
			return []string{} // Tidak ada tensor dengan jumlah dimensi ini
		}
	}

	if len(candidateSets) == 0 { // Tidak ada filter, kembalikan semua nama tensor
		// Kumpulkan semua nama dari salah satu indeks (misal ByDataType)
		// Ini bisa dioptimalkan jika kita menyimpan daftar semua tensor secara terpisah.
		allNames := make(map[string]struct{})
		for _, names := range idx.ByDataType {
			for name := range names {
				allNames[name] = struct{}{}
			}
		}
		for name := range allNames {
			resultNames = append(resultNames, name)
		}
		return resultNames
	}

	// Interseksi hasil dari filter yang aktif
	if len(candidateSets) == 1 {
		for name := range candidateSets[0] {
			resultNames = append(resultNames, name)
		}
		return resultNames
	}

	// Interseksi dua set (DataType dan NumDimensions)
	// Ambil set yang lebih kecil untuk iterasi
	var smallerSet, largerSet map[string]struct{}
	if len(candidateSets[0]) < len(candidateSets[1]) {
		smallerSet = candidateSets[0]
		largerSet = candidateSets[1]
	} else {
		smallerSet = candidateSets[1]
		largerSet = candidateSets[0]
	}

	for name := range smallerSet {
		if _, ok := largerSet[name]; ok {
			resultNames = append(resultNames, name)
		}
	}
	return resultNames
}

// Rebuild membangun ulang seluruh indeks dari file metadata di dataDir.
// Ini harus dipanggil saat Storage diinisialisasi.
func (idx *InMemoryIndex) Rebuild(dataDir string, storage *Storage) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Bersihkan indeks yang ada
	idx.ByDataType = make(map[string]map[string]struct{})
	idx.ByNumDimensions = make(map[int]map[string]struct{})
	// idx.AllTensorMetadata = make(map[string]*TensorMetadata)

	err := filepath.WalkDir(dataDir, func(path string, d fs.DirEntry, errWalk error) error {
		if errWalk != nil {
			return errWalk // Propagate error dari WalkDir
		}
		if !d.IsDir() && strings.HasSuffix(d.Name(), ".meta") {
			tensorName := strings.TrimSuffix(d.Name(), ".meta")
			// Gunakan storage.LoadTensorMetadata untuk memuat metadata
			// Perhatikan: LoadTensorMetadata mungkin mengembalikan error jika file korup.
			// Kita perlu memutuskan bagaimana menanganinya (lewati atau gagalkan rebuild).
			// Untuk saat ini, kita akan mencoba memuat dan menambahkan ke indeks jika berhasil.
			// Kita tidak bisa memanggil storage.LoadTensorMetadata secara langsung di sini karena akan menyebabkan dependensi siklik
			// atau memerlukan instance storage. Kita akan memuat secara manual di sini.
			// Atau, lebih baik, Rebuild dipanggil dari NewStorage yang sudah memiliki instance storage.
			metadata, errLoad := storage.loadTensorMetadataInternal(filepath.Join(dataDir, d.Name()))
			if errLoad == nil && metadata != nil {
				// Hitung NumDimensions di sini jika tidak disimpan di metadata
				dataType := metadata.DataType
				numDimensions := len(metadata.Shape)
				if len(metadata.Shape) == 1 && metadata.Shape[0] == 0 {
					numDimensions = 0
				}
				if len(metadata.Shape) == 0 {
					numDimensions = 0
				}

				if _, ok := idx.ByDataType[dataType]; !ok {
					idx.ByDataType[dataType] = make(map[string]struct{})
				}
				idx.ByDataType[dataType][tensorName] = struct{}{}

				if _, ok := idx.ByNumDimensions[numDimensions]; !ok {
					idx.ByNumDimensions[numDimensions] = make(map[string]struct{})
				}
				idx.ByNumDimensions[numDimensions][tensorName] = struct{}{}
				// idx.AllTensorMetadata[tensorName] = metadata
			} else if errLoad != nil {
				// Log error pemuatan metadata, tapi lanjutkan rebuild
				fmt.Fprintf(os.Stderr, "Warning: failed to load metadata for %s during index rebuild: %v\n", tensorName, errLoad)
			}
		}
		return nil
	})
	return err
}

type Storage struct {
	dataDir string
	index   *InMemoryIndex // Tambahkan field untuk indeks
}

func NewStorage(dataDir string) (*Storage, error) {
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create data directory: %v", err)
	}
	s := &Storage{
		dataDir: dataDir,
		index:   NewInMemoryIndex(), // Buat instance indeks baru
	}
	// Bangun ulang indeks saat storage dibuat
	if err := s.index.Rebuild(dataDir, s); err != nil {
		// Pertimbangkan apakah error rebuild harus fatal atau hanya warning
		fmt.Fprintf(os.Stderr, "Warning: failed to rebuild tensor index: %v\n", err)
	}
	return s, nil
}

// Fungsi pembantu internal untuk LoadTensorMetadata agar bisa dipanggil dari Rebuild
func (s *Storage) loadTensorMetadataInternal(metadataFilePath string) (*TensorMetadata, error) {
	data, err := os.ReadFile(metadataFilePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read metadata from %s: %w", metadataFilePath, err)
	}

	// Ekstrak nama tensor dari path file untuk konsistensi, meskipun tidak selalu digunakan di sini
	// tensorNameFromPath := strings.TrimSuffix(filepath.Base(metadataFilePath), ".meta")

	tm := &TensorMetadata{} // Nama akan diisi dari file atau path jika perlu
	lines := strings.Split(string(data), "\n")
	for _, line := range lines {
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, ":", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid metadata format in %s: '%s'", metadataFilePath, line)
		}
		key, value := strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1])
		switch key {
		case "name":
			tm.Name = value // Ambil nama dari file metadata
		case "shape":
			tm.Shape, err = parseIntSlice(value)
			if err != nil {
				return nil, fmt.Errorf("invalid shape '%s' in metadata: %w", value, err)
			}
		case "datatype":
			tm.DataType = value
			if _, errDt := GetElementSize(tm.DataType); errDt != nil {
				return nil, fmt.Errorf("unsupported data type '%s' in metadata: %w", value, errDt)
			}
		case "strides":
			tm.Strides, err = parseIntSlice(value)
			if err != nil {
				return nil, fmt.Errorf("invalid strides '%s' in metadata: %w", value, err)
			}
		}
	}
	if tm.Name == "" { // Jika nama tidak ada di file, coba ambil dari nama file
		tm.Name = strings.TrimSuffix(filepath.Base(metadataFilePath), ".meta")
	}

	if tm.Shape == nil || tm.DataType == "" || tm.Name == "" {
		return nil, fmt.Errorf("incomplete metadata in %s (name, shape, or datatype missing)", metadataFilePath)
	}
	if tm.Strides == nil {
		// Hitung strides default jika tidak ada di metadata
		if len(tm.Shape) > 0 {
			totalElements := 1
			isZeroDim := false
			for _, dim := range tm.Shape {
				if dim == 0 {
					isZeroDim = true
					break
				}
				totalElements *= dim
			}

			if totalElements > 0 || (len(tm.Shape) > 0 && !isZeroDim) { // Hanya hitung strides jika ada elemen atau shape tidak nol
				strides := make([]int, len(tm.Shape))
				strides[len(tm.Shape)-1] = 1
				for i := len(tm.Shape) - 2; i >= 0; i-- {
					if tm.Shape[i+1] == 0 { // Jika dimensi berikutnya 0, stride bisa jadi tidak relevan atau 0
						strides[i] = 0 // Atau bisa juga strides[i+1] * tm.Shape[i+1] yang akan jadi 0
					} else {
						strides[i] = strides[i+1] * tm.Shape[i+1]
					}
				}
				tm.Strides = strides
			} else {
				tm.Strides = make([]int, len(tm.Shape)) // Array strides kosong atau berisi nol
			}
		} else { // Skalar
			tm.Strides = []int{}
		}
	}
	return tm, nil
}

// SaveTensor sekarang tidak secara langsung memperbarui indeks.
// Executor akan bertanggung jawab untuk memanggil fungsi pembaruan indeks setelah SaveTensor berhasil.
func SaveTensor[T Numeric](s *Storage, t *Tensor[T]) error {
	metadataFile := filepath.Join(s.dataDir, t.Name+".meta")
	dataFile := filepath.Join(s.dataDir, t.Name+".data")

	typeStrT, err := GetDataTypeString[T]()
	if err != nil {
		return fmt.Errorf("internal error getting type string for T in SaveTensor: %w", err)
	}
	if t.DataType != typeStrT {
		return fmt.Errorf("tensor's DataType string ('%s') does not match generic type T ('%s')", t.DataType, typeStrT)
	}

	// Pastikan strides dihitung dengan benar sebelum menyimpan
	if t.Strides == nil || len(t.Strides) != len(t.Shape) {
		if len(t.Shape) > 0 {
			totalElements := 1
			isZeroDim := false
			for _, dim := range t.Shape {
				if dim == 0 {
					isZeroDim = true
					break
				}
				totalElements *= dim
			}
			if totalElements > 0 || (len(t.Shape) > 0 && !isZeroDim) {
				strides := make([]int, len(t.Shape))
				strides[len(t.Shape)-1] = 1
				for i := len(t.Shape) - 2; i >= 0; i-- {
					if t.Shape[i+1] == 0 {
						strides[i] = 0
					} else {
						strides[i] = strides[i+1] * t.Shape[i+1]
					}
				}
				t.Strides = strides
			} else {
				t.Strides = make([]int, len(t.Shape))
			}
		} else {
			t.Strides = []int{}
		}
	}

	metadataContent := fmt.Sprintf("name:%s\nshape:%s\ndatatype:%s\nstrides:%s\n",
		t.Name, intSliceToString(t.Shape), t.DataType, intSliceToString(t.Strides))
	if err := os.WriteFile(metadataFile, []byte(metadataContent), 0644); err != nil {
		return fmt.Errorf("failed to write metadata for %s: %w", t.Name, err)
	}

	file, err := os.Create(dataFile)
	if err != nil {
		return fmt.Errorf("failed to create data file %s: %w", dataFile, err)
	}
	defer file.Close()

	elementSize, err := GetElementSize(t.DataType)
	if err != nil {
		return fmt.Errorf("cannot save tensor %s: %w", t.Name, err)
	}

	numElements := 0
	if len(t.Shape) == 0 { // Skalar
		numElements = 1
	} else {
		numElements = 1
		isZeroDim := false
		for _, d := range t.Shape {
			if d == 0 {
				isZeroDim = true
				break
			}
			numElements *= d
		}
		if isZeroDim {
			numElements = 0
		}
	}
	// Pastikan len(t.Data) konsisten dengan numElements yang dihitung dari shape
	if len(t.Data) != numElements && numElements > 0 { // Hanya periksa jika numElements > 0
		// Ini bisa terjadi jika SetData tidak dipanggil atau shape diubah setelah SetData
		// return fmt.Errorf("data length %d does not match expected elements %d from shape %v for tensor %s", len(t.Data), numElements, t.Shape, t.Name)
		// Untuk sementara, kita lanjutkan dengan len(t.Data) jika numElements dari shape adalah 0 tapi t.Data tidak
		// Ini menunjukkan inkonsistensi, tapi kita prioritaskan data yang ada jika shape menyiratkan 0 elemen.
		if numElements == 0 && len(t.Data) > 0 {
			// Ini aneh, shape bilang 0 elemen tapi ada data. Sebaiknya error.
			return fmt.Errorf("inconsistent state: shape %v implies 0 elements but data has length %d for tensor %s", t.Shape, len(t.Data), t.Name)
		}
		// Jika numElements > 0 tapi len(t.Data) != numElements, ini juga error.
		// Sudah ditangani oleh SetData. Di sini kita asumsikan t.Data sudah benar.
		numElements = len(t.Data) // Gunakan panjang data aktual untuk dataSize
	}

	dataSize := numElements * elementSize

	if err := file.Truncate(int64(dataSize)); err != nil {
		// Jangan error jika dataSize adalah 0 (tensor kosong)
		if dataSize == 0 {
			return nil // File kosong sudah benar untuk tensor kosong
		}
		return fmt.Errorf("failed to truncate data file %s for tensor %s: %w", dataFile, t.Name, err)
	}
	if dataSize == 0 {
		return nil // Tidak ada data untuk ditulis
	}

	mmapFile, err := mmap.Map(file, mmap.RDWR, 0)
	if err != nil {
		return fmt.Errorf("failed to map data file %s for tensor %s: %w", dataFile, t.Name, err)
	}
	defer mmapFile.Unmap()

	tempBufIter := new(bytes.Buffer)
	tempBufIter.Grow(dataSize) // Alokasikan buffer dengan ukuran yang benar
	for _, val := range t.Data {
		if err := binary.Write(tempBufIter, binary.LittleEndian, val); err != nil {
			return fmt.Errorf("failed to write element of tensor %s: %w", t.Name, err)
		}
	}
	actualDataBytes := tempBufIter.Bytes()

	if len(actualDataBytes) != dataSize {
		return fmt.Errorf("data size mismatch during save for tensor %s: expected %d bytes, got %d. DataType: %s, NumElements: %d, Shape: %v", t.Name, dataSize, len(actualDataBytes), t.DataType, numElements, t.Shape)
	}
	copy(mmapFile, actualDataBytes)
	if err := mmapFile.Flush(); err != nil {
		return fmt.Errorf("failed to flush mmap for tensor %s: %w", t.Name, err)
	}
	return nil
}

func (s *Storage) LoadTensorMetadata(name string) (*TensorMetadata, error) {
	metadataFile := filepath.Join(s.dataDir, name+".meta")
	return s.loadTensorMetadataInternal(metadataFile) // Gunakan fungsi internal
}

func (s *Storage) OpenFileAndMmap(name string, expectedTotalElements int, elementSize int) (*os.File, mmap.MMap, error) {
	dataFile := filepath.Join(s.dataDir, name+".data")
	file, err := os.OpenFile(dataFile, os.O_RDWR, 0644) // Buka untuk baca/tulis
	if err != nil {
		// Jika file tidak ada DAN kita mengharapkan 0 elemen (tensor kosong baru), ini bukan error.
		// Kita akan membuat file kosong saat SaveTensor.
		// Namun, OpenFileAndMmap dipanggil saat memuat, jadi file seharusnya ada jika expectedTotalElements > 0.
		if os.IsNotExist(err) {
			if expectedTotalElements == 0 { // Memuat tensor yang memang kosong
				// Buat file kosong jika tidak ada, agar mmap tidak error.
				// Atau, kembalikan nil, nil, nil dan biarkan pemanggil menangani.
				// Untuk konsistensi, jika tensor kosong, file data mungkin tidak ada atau kosong.
				// Kita akan mengembalikan nil untuk mmap jika tensor kosong.
				return nil, nil, nil // File tidak ada, dan tensor kosong, jadi tidak ada mmap.
			}
			return nil, nil, fmt.Errorf("data file %s not found for tensor %s: %w", dataFile, name, err)
		}
		return nil, nil, fmt.Errorf("failed to open data file %s: %w", dataFile, err)
	}

	fileInfo, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, nil, fmt.Errorf("failed to stat data file %s: %w", dataFile, err)
	}

	expectedDataSize := int64(expectedTotalElements * elementSize)

	// Jika tensor kosong, ukuran file bisa 0.
	if expectedTotalElements == 0 {
		if fileInfo.Size() != 0 {
			// Ini aneh, tensor kosong tapi file data tidak kosong.
			// Bisa jadi warning atau error tergantung kebijakan.
			// Untuk saat ini, kita biarkan. Mmap akan tetap nil.
		}
		// Untuk tensor kosong, kita tidak mmap, jadi kembalikan mmapInstance nil.
		// File tetap terbuka dan akan ditutup oleh pemanggil (Executor).
		return file, nil, nil
	}

	if fileInfo.Size() != expectedDataSize {
		file.Close()
		return nil, nil, fmt.Errorf("data file size mismatch for %s: expected %d, got %d", name, expectedDataSize, fileInfo.Size())
	}

	mmapFile, err := mmap.Map(file, mmap.RDWR, 0)
	if err != nil {
		file.Close()
		return nil, nil, fmt.Errorf("failed to map data file %s: %w", dataFile, err)
	}
	return file, mmapFile, nil
}

func ReadData[T Numeric](mmapFile mmap.MMap, numElements int, dataTypeString string) ([]T, error) {
	if numElements == 0 {
		return make([]T, 0), nil // Tensor kosong
	}
	if mmapFile == nil {
		// Ini seharusnya tidak terjadi jika numElements > 0, karena OpenFileAndMmap akan error.
		return nil, errors.New("cannot read data: mmapFile is nil but numElements > 0")
	}

	dataSlice := make([]T, numElements)

	elementSize, err := GetElementSize(dataTypeString)
	if err != nil {
		return nil, fmt.Errorf("failed to get element size for type %s in ReadData: %w", dataTypeString, err)
	}
	expectedBytes := numElements * elementSize

	if len(mmapFile) < expectedBytes {
		return nil, fmt.Errorf("mmap size %d is less than expected data size %d (%d elements * %d bytes/element) for type %s", len(mmapFile), expectedBytes, numElements, elementSize, dataTypeString)
	}

	buf := bytes.NewReader(mmapFile[:expectedBytes])
	for i := 0; i < numElements; i++ {
		if err := binary.Read(buf, binary.LittleEndian, &dataSlice[i]); err != nil {
			return nil, fmt.Errorf("failed to read data element of type %s at index %d: %w", dataTypeString, i, err)
		}
	}
	return dataSlice, nil
}

func (s *Storage) GetTensorMmap(name string) (*TensorMetadata, *os.File, mmap.MMap, error) {
	metadata, err := s.LoadTensorMetadata(name)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("GetTensorMmap: failed to load metadata for %s: %w", name, err)
	}

	totalElements := 1
	if len(metadata.Shape) == 0 { // Skalar
		totalElements = 1
	} else {
		isZeroDim := false
		for _, dim := range metadata.Shape {
			if dim == 0 {
				isZeroDim = true
				break
			}
			totalElements *= dim
		}
		if isZeroDim {
			totalElements = 0
		}
	}

	elementSize, err := GetElementSize(metadata.DataType)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("GetTensorMmap: failed to get element size for %s (type %s): %w", name, metadata.DataType, err)
	}

	file, mmapInstance, err := s.OpenFileAndMmap(name, totalElements, elementSize)
	if err != nil {
		// Jika OpenFileAndMmap mengembalikan file=nil, mmapInstance=nil, dan err=nil (kasus tensor kosong tidak ada file),
		// maka kita teruskan itu.
		if file == nil && mmapInstance == nil && err == nil && totalElements == 0 {
			return metadata, nil, nil, nil
		}
		return nil, nil, nil, fmt.Errorf("GetTensorMmap: failed to open/mmap file for %s: %w", name, err)
	}
	return metadata, file, mmapInstance, nil
}

func intSliceToString(slice []int) string {
	if slice == nil { // Untuk shape skalar []
		return ""
	}
	parts := make([]string, len(slice))
	for i, v := range slice {
		parts[i] = strconv.Itoa(v)
	}
	return strings.Join(parts, ",")
}

func parseIntSlice(s string) ([]int, error) {
	s = strings.TrimSpace(s)
	if s == "" { // Untuk shape skalar yang disimpan sebagai string kosong
		return []int{}, nil
	}
	parts := strings.Split(s, ",")
	result := make([]int, len(parts))
	var err error
	for i, p := range parts {
		trimmedPart := strings.TrimSpace(p)
		if trimmedPart == "" && len(parts) == 1 && s != "0" { // "0" adalah shape yang valid
			// Jika string asli hanya spasi atau kosong, sudah ditangani di atas.
			// Ini untuk kasus seperti "," atau "1,"
			return nil, fmt.Errorf("empty dimension string part in '%s'", s)
		}
		if trimmedPart == "" && len(parts) > 1 {
			return nil, fmt.Errorf("empty dimension string part in '%s'", s)
		}
		result[i], err = strconv.Atoi(trimmedPart)
		if err != nil {
			return nil, fmt.Errorf("error parsing '%s' as int in shape string '%s': %w", trimmedPart, s, err)
		}
	}
	return result, nil
}

// Metode untuk mengakses indeks dari Storage
func (s *Storage) AddTensorToIndex(metadata *TensorMetadata) {
	s.index.Add(metadata)
}

func (s *Storage) RemoveTensorFromIndex(metadata *TensorMetadata) {
	s.index.Remove(metadata)
}

func (s *Storage) QueryIndex(filterDataType string, filterNumDimensions int) []string {
	return s.index.Query(filterDataType, filterNumDimensions)
}
