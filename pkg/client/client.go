package client

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"strconv"
	"unsafe"

	"github.com/sciefylab/tensordb/pkg/tensor" // Pastikan path ini benar

	"github.com/edsrzf/mmap-go"
)

type Client struct {
	executor *tensor.Executor
	parser   *tensor.Parser
}

func NewClient(executor *tensor.Executor) *Client {
	return &Client{
		executor: executor,
		parser:   &tensor.Parser{},
	}
}

func (c *Client) Close() error {
	if c.executor != nil {
		return c.executor.Close()
	}
	return nil
}

func (c *Client) CreateTensor(name string, shape []int, dataType string) error {
	if name == "" {
		return fmt.Errorf("nama tensor tidak boleh kosong")
	}
	if _, err := tensor.GetElementSize(dataType); err != nil {
		return fmt.Errorf("tipe data tidak valid '%s': %w", dataType, err)
	}
	// Gunakan konstanta QueryType yang benar
	query := &tensor.Query{Type: tensor.CreateTensorQuery, TensorNames: []string{name}, Shape: shape, DataType: dataType}
	_, err := c.executor.Execute(query)
	return err
}

// --- Metode InsertData spesifik tipe (DIMODIFIKASI) ---

func (c *Client) InsertFloat32Data(tensorName string, data []float32) error {
	if tensorName == "" {
		return fmt.Errorf("nama tensor tidak boleh kosong")
	}
	if data == nil { // Bisa juga mengizinkan insert data kosong jika tensor mendukung 0 elemen
		// return fmt.Errorf("data untuk insert tidak boleh nil")
	}

	buf := new(bytes.Buffer)
	err := binary.Write(buf, binary.LittleEndian, data)
	if err != nil {
		return fmt.Errorf("gagal serialisasi data float32 ke bytes: %w", err)
	}
	// Gunakan konstanta QueryType yang benar
	query := &tensor.Query{
		Type:        tensor.InsertTensorQuery,
		TensorNames: []string{tensorName},
		RawData:     buf.Bytes(), // Kirim data sebagai byte
		Data:        nil,         // Kosongkan Data string
	}
	_, execErr := c.executor.Execute(query)
	return execErr
}

func (c *Client) InsertFloat64Data(tensorName string, data []float64) error {
	if tensorName == "" {
		return fmt.Errorf("nama tensor tidak boleh kosong")
	}
	buf := new(bytes.Buffer)
	err := binary.Write(buf, binary.LittleEndian, data)
	if err != nil {
		return fmt.Errorf("gagal serialisasi data float64 ke bytes: %w", err)
	}
	query := &tensor.Query{
		Type:        tensor.InsertTensorQuery,
		TensorNames: []string{tensorName},
		RawData:     buf.Bytes(),
		Data:        nil,
	}
	_, execErr := c.executor.Execute(query)
	return execErr
}

func (c *Client) InsertInt32Data(tensorName string, data []int32) error {
	if tensorName == "" {
		return fmt.Errorf("nama tensor tidak boleh kosong")
	}
	buf := new(bytes.Buffer)
	err := binary.Write(buf, binary.LittleEndian, data)
	if err != nil {
		return fmt.Errorf("gagal serialisasi data int32 ke bytes: %w", err)
	}
	query := &tensor.Query{
		Type:        tensor.InsertTensorQuery,
		TensorNames: []string{tensorName},
		RawData:     buf.Bytes(),
		Data:        nil,
	}
	_, execErr := c.executor.Execute(query)
	return execErr
}

func (c *Client) InsertInt64Data(tensorName string, data []int64) error {
	if tensorName == "" {
		return fmt.Errorf("nama tensor tidak boleh kosong")
	}
	buf := new(bytes.Buffer)
	err := binary.Write(buf, binary.LittleEndian, data)
	if err != nil {
		return fmt.Errorf("gagal serialisasi data int64 ke bytes: %w", err)
	}
	query := &tensor.Query{
		Type:        tensor.InsertTensorQuery,
		TensorNames: []string{tensorName},
		RawData:     buf.Bytes(),
		Data:        nil,
	}
	_, execErr := c.executor.Execute(query)
	return execErr
}

// --- Akhir metode InsertData spesifik tipe ---

func (c *Client) SelectData(tensorName string, sliceRanges [][2]int) (interface{}, error) {
	if tensorName == "" {
		return nil, fmt.Errorf("nama tensor tidak boleh kosong")
	}
	query := &tensor.Query{Type: tensor.SelectTensorQuery, TensorNames: []string{tensorName}, Slices: [][][2]int{sliceRanges}}
	return c.executor.Execute(query)
}

func (c *Client) GetData(tensorNames []string, slices [][][2]int, batchSize int) (interface{}, error) {
	if len(tensorNames) == 0 {
		return nil, fmt.Errorf("setidaknya satu nama tensor harus disediakan")
	}
	if slices != nil && len(slices) != len(tensorNames) {
		return nil, fmt.Errorf("jumlah definisi slice (%d) harus cocok dengan jumlah nama tensor (%d) atau nil", len(slices), len(tensorNames))
	}
	querySlices := slices
	if slices == nil && len(tensorNames) > 0 {
		querySlices = make([][][2]int, len(tensorNames))
	}
	query := &tensor.Query{Type: tensor.GetDataTensorQuery, TensorNames: tensorNames, Slices: querySlices, BatchSize: batchSize}
	return c.executor.Execute(query)
}

func (c *Client) GetTensorMetadata(tensorName string) (*tensor.TensorMetadata, error) {
	if tensorName == "" {
		return nil, fmt.Errorf("nama tensor tidak boleh kosong")
	}
	metadata, _, cleanupFunc, err := c.GetTensorMmap(tensorName)
	if cleanupFunc != nil {
		defer cleanupFunc()
	}
	if err != nil {
		return nil, fmt.Errorf("gagal mendapatkan metadata untuk tensor '%s': %w", tensorName, err)
	}
	if metadata != nil {
		return metadata, nil
	}
	return nil, fmt.Errorf("GetTensorMmap berhasil tetapi tidak mengembalikan metadata untuk '%s'", tensorName)
}

func (c *Client) GetTensorMmap(tensorName string) (*tensor.TensorMetadata, mmap.MMap, func() error, error) {
	if tensorName == "" {
		return nil, nil, nil, fmt.Errorf("nama tensor tidak boleh kosong")
	}
	metadata, _, mmapInstance, cleanupFunc, err := c.executor.GetTensorMmap(tensorName)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("client.GetTensorMmap: gagal mendapatkan mmap dari executor untuk tensor '%s': %w", tensorName, err)
	}
	return metadata, mmapInstance, cleanupFunc, nil
}

func calculateTotalElementsFromShape(shape []int) int {
	if len(shape) == 0 {
		return 1
	}
	totalElements := 1
	isZeroDim := false
	for _, dim := range shape {
		if dim == 0 {
			isZeroDim = true
			break
		}
		totalElements *= dim
	}
	if isZeroDim {
		return 0
	}
	return totalElements
}

func readDataFromMmapInternal[T tensor.Numeric](metadata *tensor.TensorMetadata, mmapInst mmap.MMap, useUnsafe bool, targetDataTypeStr string) ([]T, error) {
	if metadata == nil {
		return nil, errors.New("metadata tidak boleh nil")
	}

	typeStrT, err := tensor.GetDataTypeString[T]()
	if err != nil {
		return nil, fmt.Errorf("tipe generik T tidak valid: %w", err)
	}
	if metadata.DataType != typeStrT {
		return nil, fmt.Errorf("tipe data metadata ('%s') tidak cocok dengan tipe generik T yang diminta ('%s')", metadata.DataType, typeStrT)
	}
	if metadata.DataType != targetDataTypeStr {
		return nil, fmt.Errorf("tipe data metadata ('%s') tidak cocok dengan tipe target metode ('%s')", metadata.DataType, targetDataTypeStr)
	}

	numElements := calculateTotalElementsFromShape(metadata.Shape)

	if mmapInst == nil {
		if numElements == 0 {
			return make([]T, 0), nil
		}
		return nil, errors.New("mmapInst tidak boleh nil untuk tensor yang tidak kosong")
	}

	if numElements == 0 {
		return make([]T, 0), nil
	}

	elementSize, err := tensor.GetElementSize(metadata.DataType)
	if err != nil {
		return nil, fmt.Errorf("gagal mendapatkan ukuran elemen untuk tipe %s: %w", metadata.DataType, err)
	}
	expectedBytes := numElements * elementSize

	if len(mmapInst) < expectedBytes {
		return nil, fmt.Errorf("ukuran mmap (%d bytes) lebih kecil dari ukuran data yang diharapkan (%d bytes) untuk %d elemen tipe %s", len(mmapInst), expectedBytes, numElements, metadata.DataType)
	}
	dataBytes := mmapInst[:expectedBytes]

	if useUnsafe {
		if len(dataBytes) == 0 {
			return make([]T, 0), nil
		}
		var sliceHeader struct {
			Data uintptr
			Len  int
			Cap  int
		}
		sliceHeader.Data = uintptr(unsafe.Pointer(&dataBytes[0]))
		sliceHeader.Len = numElements
		sliceHeader.Cap = numElements
		typedSlice := *(*[]T)(unsafe.Pointer(&sliceHeader))
		return typedSlice, nil
	} else {
		dataSlice := make([]T, numElements)
		buf := bytes.NewReader(dataBytes)
		for i := 0; i < numElements; i++ {
			if err := binary.Read(buf, binary.LittleEndian, &dataSlice[i]); err != nil {
				return nil, fmt.Errorf("gagal membaca elemen data tipe %s pada indeks %d menggunakan binary.Read: %w", metadata.DataType, i, err)
			}
		}
		return dataSlice, nil
	}
}

func (c *Client) ReadFloat32DataFromMmap(metadata *tensor.TensorMetadata, mmapInst mmap.MMap, useUnsafe bool) ([]float32, error) {
	return readDataFromMmapInternal[float32](metadata, mmapInst, useUnsafe, tensor.DataTypeFloat32)
}
func (c *Client) ReadFloat64DataFromMmap(metadata *tensor.TensorMetadata, mmapInst mmap.MMap, useUnsafe bool) ([]float64, error) {
	return readDataFromMmapInternal[float64](metadata, mmapInst, useUnsafe, tensor.DataTypeFloat64)
}
func (c *Client) ReadInt32DataFromMmap(metadata *tensor.TensorMetadata, mmapInst mmap.MMap, useUnsafe bool) ([]int32, error) {
	return readDataFromMmapInternal[int32](metadata, mmapInst, useUnsafe, tensor.DataTypeInt32)
}
func (c *Client) ReadInt64DataFromMmap(metadata *tensor.TensorMetadata, mmapInst mmap.MMap, useUnsafe bool) ([]int64, error) {
	return readDataFromMmapInternal[int64](metadata, mmapInst, useUnsafe, tensor.DataTypeInt64)
}

func (c *Client) loadTensorInternal(tensorName string, expectedDataTypeStr string) (*tensor.TensorMetadata, interface{}, error) {
	if tensorName == "" {
		return nil, nil, fmt.Errorf("nama tensor tidak boleh kosong")
	}
	metadata, err := c.GetTensorMetadata(tensorName)
	if err != nil {
		return nil, nil, fmt.Errorf("gagal memuat metadata untuk tensor '%s': %w", tensorName, err)
	}
	if metadata.DataType != expectedDataTypeStr {
		return nil, nil, fmt.Errorf("tipe data tensor aktual ('%s') tidak cocok dengan tipe yang diminta ('%s') untuk tensor '%s'", metadata.DataType, expectedDataTypeStr, tensorName)
	}

	resultInterface, err := c.GetData([]string{tensorName}, make([][][2]int, 1), 0)
	if err != nil {
		return nil, nil, fmt.Errorf("gagal mengeksekusi query get data untuk memuat tensor '%s': %w", tensorName, err)
	}

	dataResults, ok := resultInterface.([]tensor.TensorDataResult)
	if !ok || len(dataResults) == 0 {
		if calculateTotalElementsFromShape(metadata.Shape) == 0 {
			switch expectedDataTypeStr {
			case tensor.DataTypeFloat32:
				return metadata, []float32{}, nil
			case tensor.DataTypeFloat64:
				return metadata, []float64{}, nil
			case tensor.DataTypeInt32:
				return metadata, []int32{}, nil
			case tensor.DataTypeInt64:
				return metadata, []int64{}, nil
			}
		}
		return nil, nil, fmt.Errorf("hasil tidak terduga saat memuat data tensor '%s', got type %T", tensorName, resultInterface)
	}
	return metadata, dataResults[0].Data, nil
}

func (c *Client) LoadTensorFloat32(tensorName string) (*tensor.Tensor[float32], error) {
	metadata, dataInterface, err := c.loadTensorInternal(tensorName, tensor.DataTypeFloat32)
	if err != nil {
		return nil, err
	}
	actualData, ok := dataInterface.([]float32)
	if !ok {
		return nil, fmt.Errorf("gagal mengonversi data tensor '%s' ke []float32, data aktual adalah %T", tensorName, dataInterface)
	}
	loadedTensor, errNew := tensor.NewTensor[float32](metadata.Name, metadata.Shape, metadata.DataType)
	if errNew != nil {
		return nil, errNew
	}
	if errSet := loadedTensor.SetData(actualData); errSet != nil {
		return nil, fmt.Errorf("gagal mengatur data untuk tensor[float32] '%s': %w", tensorName, errSet)
	}
	loadedTensor.Strides = metadata.Strides
	return loadedTensor, nil
}
func (c *Client) LoadTensorFloat64(tensorName string) (*tensor.Tensor[float64], error) {
	metadata, dataInterface, err := c.loadTensorInternal(tensorName, tensor.DataTypeFloat64)
	if err != nil {
		return nil, err
	}
	actualData, ok := dataInterface.([]float64)
	if !ok {
		return nil, fmt.Errorf("gagal mengonversi data tensor '%s' ke []float64, data aktual adalah %T", tensorName, dataInterface)
	}
	loadedTensor, errNew := tensor.NewTensor[float64](metadata.Name, metadata.Shape, metadata.DataType)
	if errNew != nil {
		return nil, errNew
	}
	if errSet := loadedTensor.SetData(actualData); errSet != nil {
		return nil, fmt.Errorf("gagal mengatur data untuk tensor[float64] '%s': %w", tensorName, errSet)
	}
	loadedTensor.Strides = metadata.Strides
	return loadedTensor, nil
}
func (c *Client) LoadTensorInt32(tensorName string) (*tensor.Tensor[int32], error) {
	metadata, dataInterface, err := c.loadTensorInternal(tensorName, tensor.DataTypeInt32)
	if err != nil {
		return nil, err
	}
	actualData, ok := dataInterface.([]int32)
	if !ok {
		return nil, fmt.Errorf("gagal mengonversi data tensor '%s' ke []int32, data aktual adalah %T", tensorName, dataInterface)
	}
	loadedTensor, errNew := tensor.NewTensor[int32](metadata.Name, metadata.Shape, metadata.DataType)
	if errNew != nil {
		return nil, errNew
	}
	if errSet := loadedTensor.SetData(actualData); errSet != nil {
		return nil, fmt.Errorf("gagal mengatur data untuk tensor[int32] '%s': %w", tensorName, errSet)
	}
	loadedTensor.Strides = metadata.Strides
	return loadedTensor, nil
}
func (c *Client) LoadTensorInt64(tensorName string) (*tensor.Tensor[int64], error) {
	metadata, dataInterface, err := c.loadTensorInternal(tensorName, tensor.DataTypeInt64)
	if err != nil {
		return nil, err
	}
	actualData, ok := dataInterface.([]int64)
	if !ok {
		return nil, fmt.Errorf("gagal mengonversi data tensor '%s' ke []int64, data aktual adalah %T", tensorName, dataInterface)
	}
	loadedTensor, errNew := tensor.NewTensor[int64](metadata.Name, metadata.Shape, metadata.DataType)
	if errNew != nil {
		return nil, errNew
	}
	if errSet := loadedTensor.SetData(actualData); errSet != nil {
		return nil, fmt.Errorf("gagal mengatur data untuk tensor[int64] '%s': %w", tensorName, errSet)
	}
	loadedTensor.Strides = metadata.Strides
	return loadedTensor, nil
}

// --- Metode Klien untuk Operasi Matematika ---
func (c *Client) AddTensors(tensorAName, tensorBName, resultTensorName string) (string, error) {
	q := &tensor.Query{
		Type:             tensor.MathOperationQuery,
		MathOperator:     "ADD_TENSORS",
		InputTensorNames: []string{tensorAName, tensorBName},
		OutputTensorName: resultTensorName,
	}
	result, err := c.executor.Execute(q)
	if err != nil {
		return "", err
	}
	if resultStr, ok := result.(string); ok {
		return resultStr, nil
	}
	return "", fmt.Errorf("hasil tidak terduga dari operasi ADD_TENSORS: %v", result)
}

func (c *Client) AddScalarToTensor(scalar float32, tensorName, resultTensorName string) (string, error) {
	scalarStr := strconv.FormatFloat(float64(scalar), 'f', -1, 32)
	q := &tensor.Query{
		Type:             tensor.MathOperationQuery,
		MathOperator:     "ADD_SCALAR",
		InputTensorNames: []string{tensorName},
		ScalarOperand:    scalarStr,
		OutputTensorName: resultTensorName,
	}
	result, err := c.executor.Execute(q)
	if err != nil {
		return "", err
	}
	if resultStr, ok := result.(string); ok {
		return resultStr, nil
	}
	return "", fmt.Errorf("hasil tidak terduga dari operasi ADD_SCALAR: %v", result)
}

// Metode baru untuk LIST TENSORS
func (c *Client) ListTensors(filterDataType string, filterNumDimensions int) ([]tensor.TensorMetadata, error) {
	query := &tensor.Query{
		Type:                tensor.ListTensorsQuery,
		FilterDataType:      filterDataType,
		FilterNumDimensions: filterNumDimensions,
	}
	result, err := c.executor.Execute(query)
	if err != nil {
		return nil, err
	}
	metadataResults, ok := result.([]tensor.TensorMetadata)
	if !ok {
		return nil, fmt.Errorf("unexpected result type from ListTensors operation: expected []tensor.TensorMetadata, got %T", result)
	}
	return metadataResults, nil
}
