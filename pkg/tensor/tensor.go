package tensor

import (
	"errors"
	"fmt"
	"math"
)

// Numeric adalah batasan tipe untuk tipe data numerik yang didukung oleh Tensor.
type Numeric interface {
	~float32 | ~float64 | ~int32 | ~int64
}

// Supported Data Types (string constants remain useful for metadata and parsing)
const (
	DataTypeFloat32 string = "float32"
	DataTypeFloat64 string = "float64"
	DataTypeInt32   string = "int32"
	DataTypeInt64   string = "int64"
)

// GetElementSize mengembalikan ukuran dalam byte dari satu elemen tipe data yang diberikan.
func GetElementSize(dataType string) (int, error) {
	switch dataType {
	case DataTypeFloat32:
		return 4, nil
	case DataTypeFloat64:
		return 8, nil
	case DataTypeInt32:
		return 4, nil
	case DataTypeInt64:
		return 8, nil
	default:
		return 0, fmt.Errorf("unsupported data type string: %s", dataType)
	}
}

// GetDataTypeString mengembalikan representasi string dari tipe generik T.
func GetDataTypeString[T Numeric]() (string, error) {
	var zero T
	switch any(zero).(type) {
	case float32:
		return DataTypeFloat32, nil
	case float64:
		return DataTypeFloat64, nil
	case int32:
		return DataTypeInt32, nil
	case int64:
		return DataTypeInt64, nil
	default:
		// Ini seharusnya tidak terjadi jika T dibatasi oleh Numeric
		return "", fmt.Errorf("unsupported generic type: %T", zero)
	}
}

// Tensor merepresentasikan array data multidimensi generik.
type Tensor[T Numeric] struct {
	Name     string
	Shape    []int
	Data     []T
	DataType string
	Strides  []int
}

// TensorDataWithMetadata menyimpan data tensor generik beserta metadata untuk inferensi.
type TensorDataWithMetadata[T Numeric] struct {
	Name          string     `json:"name"`
	Shape         []int      `json:"shape"`
	NumDimensions int        `json:"numDimensions"`
	DataType      string     `json:"dataType"`
	TotalElements int        `json:"totalElements"`
	DataSizeBytes int        `json:"dataSizeBytes"`
	Strides       []int      `json:"strides"`
	BatchInfo     *BatchInfo `json:"batchInfo,omitempty"`
	Data          []T        `json:"data"`
}

type BatchInfo struct {
	BatchSize         int
	NumBatches        int
	CurrentBatchIndex int
}

func (bi *BatchInfo) String() string {
	if bi == nil {
		return "<nil>"
	}
	return fmt.Sprintf("{%d %d %d}", bi.BatchSize, bi.NumBatches, bi.CurrentBatchIndex)
}

func NewTensor[T Numeric](name string, shape []int, dataTypeString string) (*Tensor[T], error) {
	typeStrT, err := GetDataTypeString[T]()
	if err != nil {
		return nil, fmt.Errorf("internal error getting type string for T: %w", err)
	}
	if typeStrT != dataTypeString {
		return nil, fmt.Errorf("type parameter T (%s) does not match dataTypeString (%s)", typeStrT, dataTypeString)
	}

	for _, dim := range shape {
		if dim < 0 {
			return nil, errors.New("invalid dimension size: cannot be negative")
		}
	}

	totalElements := 1
	hasZeroDim := false
	if len(shape) > 0 {
		for _, dim := range shape {
			if dim == 0 {
				hasZeroDim = true
				break
			}
			totalElements *= dim
		}
		if hasZeroDim {
			totalElements = 0
		}
	} else {
		totalElements = 1
	}

	dataSlice := make([]T, totalElements)
	strides := make([]int, len(shape))
	if len(shape) > 0 {
		if totalElements > 0 {
			strides[len(shape)-1] = 1
			for i := len(shape) - 2; i >= 0; i-- {
				if shape[i+1] == 0 {
					strides[i] = 0
				} else {
					strides[i] = strides[i+1] * shape[i+1]
				}
			}
		}
	}

	return &Tensor[T]{
		Name: name, Shape: shape, Data: dataSlice, DataType: dataTypeString, Strides: strides,
	}, nil
}

func (t *Tensor[T]) getTotalElements() int {
	if len(t.Shape) == 0 {
		return 1
	}
	totalElements := 1
	for _, dim := range t.Shape {
		if dim == 0 {
			return 0
		}
		totalElements *= dim
	}
	return totalElements
}

func (t *Tensor[T]) SetData(data []T) error {
	expectedElements := t.getTotalElements()
	actualElements := len(data)

	if actualElements != expectedElements {
		return fmt.Errorf("data size %d does not match tensor size %d (shape %v)", actualElements, expectedElements, t.Shape)
	}
	if actualElements == 0 && expectedElements == 0 {
		t.Data = make([]T, 0)
		return nil
	}
	if actualElements > 0 {
		t.Data = make([]T, actualElements)
		copy(t.Data, data)
	}
	return nil
}

func (t *Tensor[T]) GetSlice(ranges [][2]int) ([]T, error) {
	if t.getTotalElements() == 0 && (len(ranges) > 0 && len(ranges[0]) > 0 && ranges[0][1]-ranges[0][0] > 0) {
		isSliceEmpty := true
		for _, r := range ranges {
			if r[1]-r[0] > 0 {
				isSliceEmpty = false
				break
			}
		}
		if !isSliceEmpty {
			return nil, fmt.Errorf("cannot get a non-empty slice from an empty tensor (shape %v)", t.Shape)
		}
	}

	if len(ranges) != len(t.Shape) {
		if !(len(t.Shape) == 0 && len(ranges) == 1 && ranges[0][0] == 0 && ranges[0][1] == 1) &&
			!(len(t.Shape) == 1 && t.Shape[0] == 1 && len(ranges) == 1 && ranges[0][0] == 0 && ranges[0][1] == 1) {
			return nil, fmt.Errorf("slice ranges length %d does not match tensor dimensions %d", len(ranges), len(t.Shape))
		}
	}

	newSliceShape := make([]int, len(ranges))
	for i, r := range ranges {
		currentDimSize := 0
		if len(t.Shape) == 0 && i == 0 {
			currentDimSize = 1
		} else if i < len(t.Shape) {
			currentDimSize = t.Shape[i]
		} else {
			return nil, fmt.Errorf("internal error: trying to access shape dimension out of bounds")
		}

		if r[0] < 0 || r[1] > currentDimSize || r[0] > r[1] {
			return nil, fmt.Errorf("invalid slice range [%d:%d] for dimension %d with size %d", r[0], r[1], i, currentDimSize)
		}
		newSliceShape[i] = r[1] - r[0]
	}

	resultSize := 1
	hasZeroDimInSlice := false
	for _, dimSize := range newSliceShape {
		if dimSize == 0 {
			hasZeroDimInSlice = true
			break
		}
		resultSize *= dimSize
	}
	if hasZeroDimInSlice {
		resultSize = 0
	}

	resultSlice := make([]T, resultSize)
	if resultSize == 0 {
		return resultSlice, nil
	}

	if t.getTotalElements() == 1 && len(t.Shape) <= 1 && resultSize == 1 {
		if len(t.Data) > 0 {
			resultSlice[0] = t.Data[0]
			return resultSlice, nil
		}
		return nil, fmt.Errorf("inconsistent scalar tensor state: expected 1 element but data is empty")
	}

	currentIterIndices := make([]int, len(t.Shape))
	for i := range currentIterIndices {
		currentIterIndices[i] = ranges[i][0]
	}

	destIndex := 0
mainLoop:
	for {
		sourceOffset := 0
		for i, originalDimIndex := range currentIterIndices {
			sourceOffset += originalDimIndex * t.Strides[i]
		}
		if t.getTotalElements() > 0 {
			if sourceOffset >= len(t.Data) {
				return nil, fmt.Errorf("source offset %d out of bounds (%d) during slice. Tensor shape: %v, slice ranges: %v, current iter indices: %v, strides: %v", sourceOffset, len(t.Data), t.Shape, ranges, currentIterIndices, t.Strides)
			}
			resultSlice[destIndex] = t.Data[sourceOffset]
		} else if resultSize > 0 {
			return nil, fmt.Errorf("attempting to create non-empty slice from an empty tensor. Tensor shape: %v, slice ranges: %v", t.Shape, ranges)
		}

		destIndex++
		if destIndex >= resultSize {
			break mainLoop
		}
		for i := len(currentIterIndices) - 1; i >= 0; i-- {
			currentIterIndices[i]++
			if currentIterIndices[i] < ranges[i][1] {
				break
			}
			if i > 0 {
				currentIterIndices[i] = ranges[i][0]
			} else {
				break mainLoop
			}
		}
	}
	return resultSlice, nil
}

func (t *Tensor[T]) GetDataForInference(ranges [][][2]int, batchSize int) ([]TensorDataWithMetadata[T], error) {
	var dataToProcess []T
	var currentShape []int
	var currentStrides []int
	var err error

	if len(ranges) > 0 && ranges[0] != nil {
		if len(ranges[0]) != len(t.Shape) {
			if !(len(t.Shape) == 0 && len(ranges[0]) == 1 && ranges[0][0][0] == 0 && ranges[0][0][1] == 1) {
				return nil, fmt.Errorf("slice ranges length %d does not match tensor dimensions %d for tensor %s", len(ranges[0]), len(t.Shape), t.Name)
			}
		}
		dataToProcess, err = t.GetSlice(ranges[0])
		if err != nil {
			return nil, fmt.Errorf("failed to get slice for tensor %s: %w", t.Name, err)
		}
		currentShape = make([]int, len(ranges[0]))
		for i, r := range ranges[0] {
			currentShape[i] = r[1] - r[0]
		}
		currentStrides = make([]int, len(currentShape))
		if len(currentShape) > 0 {
			if tNilaiTotalElemen(currentShape) > 0 {
				currentStrides[len(currentShape)-1] = 1
				for i := len(currentShape) - 2; i >= 0; i-- {
					if currentShape[i+1] == 0 {
						currentStrides[i] = 0
					} else {
						currentStrides[i] = currentStrides[i+1] * currentShape[i+1]
					}
				}
			}
		}
	} else {
		dataToProcess = t.Data
		currentShape = t.Shape
		currentStrides = t.Strides
	}

	totalElementsInSelection := 0
	if len(currentShape) == 0 {
		totalElementsInSelection = 1
	} else {
		totalElementsInSelection = 1
		isZeroDim := false
		for _, dim := range currentShape {
			if dim == 0 {
				isZeroDim = true
				break
			}
			totalElementsInSelection *= dim
		}
		if isZeroDim {
			totalElementsInSelection = 0
		}
	}

	elementSize, err := GetElementSize(t.DataType)
	if err != nil {
		return nil, err
	}

	if batchSize <= 0 || totalElementsInSelection == 0 {
		dataSizeBytes := totalElementsInSelection * elementSize
		return []TensorDataWithMetadata[T]{{
			Name: t.Name, Shape: currentShape, NumDimensions: len(currentShape), DataType: t.DataType,
			TotalElements: totalElementsInSelection, DataSizeBytes: dataSizeBytes, Strides: currentStrides,
			BatchInfo: nil,
			Data:      dataToProcess,
		}}, nil
	}

	numBatches := int(math.Ceil(float64(totalElementsInSelection) / float64(batchSize)))
	if numBatches == 0 && totalElementsInSelection > 0 {
		numBatches = 1
	} else if totalElementsInSelection == 0 {
		if batchSize > 0 {
			numBatches = 1
		} else {
			numBatches = 0
		}
	}

	results := make([]TensorDataWithMetadata[T], 0, numBatches)
	if numBatches == 0 && totalElementsInSelection == 0 && batchSize > 0 {
		results = append(results, TensorDataWithMetadata[T]{
			Name: t.Name, Shape: currentShape, NumDimensions: len(currentShape), DataType: t.DataType,
			TotalElements: 0, DataSizeBytes: 0, Strides: currentStrides,
			BatchInfo: &BatchInfo{BatchSize: batchSize, NumBatches: 1, CurrentBatchIndex: 0},
			Data:      make([]T, 0),
		})
		return results, nil
	}

	for i := 0; i < numBatches; i++ {
		start := i * batchSize
		end := start + batchSize
		if end > totalElementsInSelection {
			end = totalElementsInSelection
		}

		var batchData []T
		actualBatchSize := 0
		if start >= end {
			batchData = make([]T, 0)
			actualBatchSize = 0
		} else {
			batchData = make([]T, end-start)
			copy(batchData, dataToProcess[start:end])
			actualBatchSize = end - start
		}

		batchDataSizeBytes := actualBatchSize * elementSize
		results = append(results, TensorDataWithMetadata[T]{
			Name: t.Name, Shape: currentShape, NumDimensions: len(currentShape), DataType: t.DataType,
			TotalElements: totalElementsInSelection, DataSizeBytes: batchDataSizeBytes, Strides: currentStrides,
			BatchInfo: &BatchInfo{BatchSize: batchSize, NumBatches: numBatches, CurrentBatchIndex: i},
			Data:      batchData,
		})
	}
	return results, nil
}

func formatRecursiveCore[T Numeric](data []T, currentShape []int, currentOffset *int) interface{} {
	if len(currentShape) == 0 {
		return nil
	}
	if len(currentShape) == 1 {
		dimSize := currentShape[0]
		if dimSize == 0 {
			return []interface{}{}
		}
		elementsToCopy := dimSize
		if *currentOffset+elementsToCopy > len(data) {
			elementsToCopy = len(data) - *currentOffset
			if elementsToCopy < 0 {
				elementsToCopy = 0
			}
		}
		slice := make([]interface{}, elementsToCopy)
		for i := 0; i < elementsToCopy; i++ {
			slice[i] = data[*currentOffset+i]
		}
		*currentOffset += elementsToCopy
		return slice
	}

	outerDimSize := currentShape[0]
	if outerDimSize == 0 {
		return []interface{}{}
	}

	result := make([]interface{}, outerDimSize)
	innerShape := currentShape[1:]
	for i := 0; i < outerDimSize; i++ {
		if *currentOffset >= len(data) && tNilaiTotalElemen(currentShape) > 0 {
			var buildEmpty func(shp []int) interface{}
			buildEmpty = func(shp []int) interface{} {
				if len(shp) == 0 {
					return nil
				}
				s := make([]interface{}, shp[0])
				if shp[0] == 0 {
					return []interface{}{}
				}
				if len(shp) > 1 {
					for k := 0; k < shp[0]; k++ {
						s[k] = buildEmpty(shp[1:])
					}
				}
				return s
			}
			result[i] = buildEmpty(innerShape)
			continue
		}
		result[i] = formatRecursiveCore(data, innerShape, currentOffset)
	}
	return result
}

func tNilaiTotalElemen(shape []int) int {
	if len(shape) == 0 {
		return 1
	}
	total := 1
	for _, d := range shape {
		if d == 0 {
			return 0
		}
		total *= d
	}
	return total
}

func (t *Tensor[T]) FormatMultidimensional() interface{} {
	expectedElements := t.getTotalElements()

	if t.Data == nil || (expectedElements > 0 && len(t.Data) == 0) {
		if expectedElements == 0 {
			var buildEmptyStructure func(dimIdx int, shp []int) interface{}
			buildEmptyStructure = func(dimIdx int, shp []int) interface{} {
				if dimIdx >= len(shp) {
					if len(shp) == 0 {
						return []interface{}{}
					}
					return []interface{}{}
				}
				dimSize := shp[dimIdx]
				res := make([]interface{}, dimSize)
				for i := 0; i < dimSize; i++ {
					res[i] = buildEmptyStructure(dimIdx+1, shp)
				}
				return res
			}
			if len(t.Shape) == 0 {
				return []interface{}{}
			}
			return buildEmptyStructure(0, t.Shape)
		}
		return "<nil tensor data or data inconsistent with non-empty shape>"
	}

	currentShape := t.Shape

	if len(currentShape) == 0 {
		if len(t.Data) == 1 {
			return t.Data[0]
		}
		if len(t.Data) == 0 && expectedElements <= 1 {
			return []interface{}{}
		}
		return fmt.Sprintf("<inconsistent scalar data for shape []: len %d, expected %d>", len(t.Data), expectedElements)
	}

	if expectedElements == 0 {
		var buildEmptyStructure func(dimIdx int, shp []int) interface{}
		buildEmptyStructure = func(dimIdx int, shp []int) interface{} {
			if dimIdx >= len(shp) {
				return []interface{}{}
			}
			dimSize := shp[dimIdx]
			res := make([]interface{}, dimSize)
			for i := 0; i < dimSize; i++ {
				res[i] = buildEmptyStructure(dimIdx+1, shp)
			}
			return res
		}
		return buildEmptyStructure(0, currentShape)
	}

	offset := 0
	return formatRecursiveCore(t.Data, currentShape, &offset)
}

func (t *Tensor[T]) String() string {
	return fmt.Sprintf("Tensor(Name: %s, Shape: %v, DataType: %s, Data: %v (first few elements))",
		t.Name, t.Shape, t.DataType, 첫N(t.Data, 5))
}

func 첫N[T Numeric](data []T, n int) []T {
	if len(data) > n {
		return data[:n]
	}
	return data
}

func ShapesEqual(s1, s2 []int) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i := range s1 {
		if s1[i] != s2[i] {
			return false
		}
	}
	return true
}

func AddTensors[T Numeric](t1, t2 *Tensor[T]) (*Tensor[T], error) {
	if !ShapesEqual(t1.Shape, t2.Shape) {
		return nil, fmt.Errorf("bentuk tensor tidak sama: %v dan %v (broadcasting belum diimplementasikan)", t1.Shape, t2.Shape)
	}
	if t1.DataType != t2.DataType {
		return nil, fmt.Errorf("tipe data tensor tidak sama: %s dan %s", t1.DataType, t2.DataType)
	}

	if t1.getTotalElements() == 0 {
		resultTensor, err := NewTensor[T]("temp_add_result", t1.Shape, t1.DataType)
		if err != nil {
			return nil, err
		}
		return resultTensor, nil
	}

	resultData := make([]T, len(t1.Data))
	for i := range t1.Data {
		resultData[i] = t1.Data[i] + t2.Data[i]
	}

	resultTensor, err := NewTensor[T]("temp_add_result", t1.Shape, t1.DataType)
	if err != nil {
		return nil, err
	}
	err = resultTensor.SetData(resultData)
	if err != nil {
		return nil, err
	}
	return resultTensor, nil
}

func AddScalarToTensor[T Numeric](t *Tensor[T], scalar T) (*Tensor[T], error) {
	if t.getTotalElements() == 0 {
		resultTensor, err := NewTensor[T]("temp_add_scalar_result", t.Shape, t.DataType)
		if err != nil {
			return nil, err
		}
		return resultTensor, nil
	}

	resultData := make([]T, len(t.Data))
	for i := range t.Data {
		resultData[i] = t.Data[i] + scalar
	}

	resultTensor, err := NewTensor[T]("temp_add_scalar_result", t.Shape, t.DataType)
	if err != nil {
		return nil, err
	}
	err = resultTensor.SetData(resultData)
	if err != nil {
		return nil, err
	}
	return resultTensor, nil
}

// QueryType merepresentasikan tipe kueri.
type QueryType string

const (
	CreateTensorQuery  QueryType = "create_tensor" // Diubah untuk menghindari konflik dengan const DataType
	InsertTensorQuery  QueryType = "insert_tensor"
	SelectTensorQuery  QueryType = "select_tensor"
	GetDataTensorQuery QueryType = "get_data_tensor"
	MathOperationQuery QueryType = "math_operation"
	ListTensorsQuery   QueryType = "list_tensors"
)

// Query merepresentasikan kueri yang sudah diparsing.
type Query struct {
	Type        QueryType
	TensorNames []string
	Shape       []int
	DataType    string   // Tipe data untuk CREATE TENSOR
	Data        []string // Data untuk INSERT dari string kueri
	RawData     []byte   // Data biner untuk INSERT dari client (OPTIMASI)
	Slices      [][][2]int
	BatchSize   int

	MathOperator     string
	InputTensorNames []string
	OutputTensorName string
	ScalarOperand    string
	Axis             *int

	FilterDataType      string
	FilterNumDimensions int
}
