package tensor

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/edsrzf/mmap-go"
)

type Executor struct {
	storage   *Storage
	mmaps     map[string]mmap.MMap
	mmapsMux  sync.Mutex
	openFiles map[string]*os.File
}

func NewExecutor(storage *Storage) *Executor {
	return &Executor{
		storage:   storage,
		mmaps:     make(map[string]mmap.MMap),
		openFiles: make(map[string]*os.File),
	}
}

func (e *Executor) Close() error {
	e.mmapsMux.Lock()
	defer e.mmapsMux.Unlock()
	var overallErr error
	for name, m := range e.mmaps {
		currentTensorName := name
		if m != nil {
			if err := m.Unmap(); err != nil {
				unmapErr := fmt.Errorf("failed to unmap %s: %w", currentTensorName, err)
				if overallErr == nil {
					overallErr = unmapErr
				}
			}
		}
	}
	e.mmaps = make(map[string]mmap.MMap)
	for name, f := range e.openFiles {
		currentTensorName := name
		if f != nil {
			if err := f.Close(); err != nil {
				closeErr := fmt.Errorf("failed to close file for %s: %w", currentTensorName, err)
				if overallErr == nil {
					overallErr = closeErr
				}
			}
		}
	}
	e.openFiles = make(map[string]*os.File)
	return overallErr
}

func loadFullTensorTyped[T Numeric](e *Executor, tensorName string, metadata *TensorMetadata) (*Tensor[T], error) {
	e.mmapsMux.Lock()
	if oldMmap, exists := e.mmaps[tensorName]; exists {
		if oldMmap != nil {
			oldMmap.Unmap()
		}
		delete(e.mmaps, tensorName)
	}
	if oldFile, exists := e.openFiles[tensorName]; exists {
		if oldFile != nil {
			oldFile.Close()
		}
		delete(e.openFiles, tensorName)
	}
	e.mmapsMux.Unlock()

	totalElements := 1
	if len(metadata.Shape) == 0 {
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
		return nil, fmt.Errorf("loadFullTensorTyped: %w", err)
	}

	file, mmapInstance, err := e.storage.OpenFileAndMmap(tensorName, totalElements, elementSize)
	if err != nil {
		return nil, fmt.Errorf("loadFullTensorTyped: failed to open/mmap file for %s: %w", tensorName, err)
	}

	e.mmapsMux.Lock()
	e.mmaps[tensorName] = mmapInstance
	e.openFiles[tensorName] = file
	e.mmapsMux.Unlock()

	var data []T
	data, err = ReadData[T](mmapInstance, totalElements, metadata.DataType)
	if err != nil {
		e.mmapsMux.Lock()
		if m, ok := e.mmaps[tensorName]; ok && m != nil {
			m.Unmap()
		}
		delete(e.mmaps, tensorName)
		if f, ok := e.openFiles[tensorName]; ok && f != nil {
			f.Close()
		}
		delete(e.openFiles, tensorName)
		e.mmapsMux.Unlock()
		return nil, fmt.Errorf("loadFullTensorTyped: failed to read data for %s: %w", tensorName, err)
	}

	dataTypeStrForT, _ := GetDataTypeString[T]()
	tensorInstance, err := NewTensor[T](metadata.Name, metadata.Shape, dataTypeStrForT)
	if err != nil {
		e.mmapsMux.Lock()
		if m, ok := e.mmaps[tensorName]; ok && m != nil {
			m.Unmap()
		}
		delete(e.mmaps, tensorName)
		if f, ok := e.openFiles[tensorName]; ok && f != nil {
			f.Close()
		}
		delete(e.openFiles, tensorName)
		e.mmapsMux.Unlock()
		return nil, fmt.Errorf("loadFullTensorTyped: failed to create tensor instance for %s: %w", tensorName, err)
	}
	if err := tensorInstance.SetData(data); err != nil {
		e.mmapsMux.Lock()
		if m, ok := e.mmaps[tensorName]; ok && m != nil {
			m.Unmap()
		}
		delete(e.mmaps, tensorName)
		if f, ok := e.openFiles[tensorName]; ok && f != nil {
			f.Close()
		}
		delete(e.openFiles, tensorName)
		e.mmapsMux.Unlock()
		return nil, fmt.Errorf("loadFullTensorTyped: failed to set data for tensor %s: %w", tensorName, err)
	}
	tensorInstance.Strides = metadata.Strides
	return tensorInstance, nil
}

func (e *Executor) GetTensorMmap(tensorName string) (*TensorMetadata, *os.File, mmap.MMap, func() error, error) {
	e.mmapsMux.Lock()
	if oldMmap, exists := e.mmaps[tensorName]; exists {
		if oldMmap != nil {
			oldMmap.Unmap()
		}
		delete(e.mmaps, tensorName)
	}
	if oldFile, exists := e.openFiles[tensorName]; exists {
		if oldFile != nil {
			oldFile.Close()
		}
		delete(e.openFiles, tensorName)
	}
	e.mmapsMux.Unlock()

	metadata, file, mmapInstance, storageErr := e.storage.GetTensorMmap(tensorName)
	if storageErr != nil {
		return nil, nil, nil, nil, fmt.Errorf("executor.GetTensorMmap: failed to get mmap from storage for %s: %w", tensorName, storageErr)
	}

	e.mmapsMux.Lock()
	e.mmaps[tensorName] = mmapInstance
	e.openFiles[tensorName] = file
	e.mmapsMux.Unlock()

	cleanupFunc := func() error {
		e.mmapsMux.Lock()
		defer e.mmapsMux.Unlock()
		var firstCleanupErr error
		if m, ok := e.mmaps[tensorName]; ok {
			if m != nil {
				if errUnmap := m.Unmap(); errUnmap != nil {
					firstCleanupErr = fmt.Errorf("cleanupFunc for %s: failed to unmap: %w", tensorName, errUnmap)
				}
			}
			delete(e.mmaps, tensorName)
		}
		if f, ok := e.openFiles[tensorName]; ok {
			if f != nil {
				if errClose := f.Close(); errClose != nil {
					if firstCleanupErr == nil {
						firstCleanupErr = fmt.Errorf("cleanupFunc for %s: failed to close file: %w", tensorName, errClose)
					}
				}
			}
			delete(e.openFiles, tensorName)
		}
		return firstCleanupErr
	}
	return metadata, file, mmapInstance, cleanupFunc, nil
}

type TensorDataResult struct {
	Name          string
	Shape         []int
	NumDimensions int
	DataType      string
	TotalElements int
	DataSizeBytes int
	Strides       []int
	BatchInfo     *BatchInfo
	Data          interface{}
}

func (e *Executor) Execute(query *Query) (interface{}, error) {
	switch query.Type {
	case CreateTensorQuery:
		tensorName := query.TensorNames[0]
		_, err := e.storage.LoadTensorMetadata(tensorName)
		if err == nil {
			return nil, fmt.Errorf("tensor '%s' already exists", tensorName)
		}
		if !os.IsNotExist(errors.Unwrap(err)) && err != nil && !strings.Contains(err.Error(), "failed to read metadata") {
			return nil, fmt.Errorf("error checking existing tensor '%s': %w", tensorName, err)
		}

		var newTensorMetadata *TensorMetadata
		switch query.DataType {
		case DataTypeFloat32:
			tensorInstance, err := NewTensor[float32](tensorName, query.Shape, query.DataType)
			if err != nil {
				return nil, err
			}
			if err := SaveTensor(e.storage, tensorInstance); err != nil {
				return nil, err
			}
			newTensorMetadata = &TensorMetadata{Name: tensorInstance.Name, Shape: tensorInstance.Shape, DataType: tensorInstance.DataType, Strides: tensorInstance.Strides}
		case DataTypeFloat64:
			tensorInstance, err := NewTensor[float64](tensorName, query.Shape, query.DataType)
			if err != nil {
				return nil, err
			}
			if err := SaveTensor(e.storage, tensorInstance); err != nil {
				return nil, err
			}
			newTensorMetadata = &TensorMetadata{Name: tensorInstance.Name, Shape: tensorInstance.Shape, DataType: tensorInstance.DataType, Strides: tensorInstance.Strides}
		case DataTypeInt32:
			tensorInstance, err := NewTensor[int32](tensorName, query.Shape, query.DataType)
			if err != nil {
				return nil, err
			}
			if err := SaveTensor(e.storage, tensorInstance); err != nil {
				return nil, err
			}
			newTensorMetadata = &TensorMetadata{Name: tensorInstance.Name, Shape: tensorInstance.Shape, DataType: tensorInstance.DataType, Strides: tensorInstance.Strides}
		case DataTypeInt64:
			tensorInstance, err := NewTensor[int64](tensorName, query.Shape, query.DataType)
			if err != nil {
				return nil, err
			}
			if err := SaveTensor(e.storage, tensorInstance); err != nil {
				return nil, err
			}
			newTensorMetadata = &TensorMetadata{Name: tensorInstance.Name, Shape: tensorInstance.Shape, DataType: tensorInstance.DataType, Strides: tensorInstance.Strides}
		default:
			return nil, fmt.Errorf("unsupported data type for CREATE TENSOR: %s", query.DataType)
		}
		if newTensorMetadata != nil {
			e.storage.AddTensorToIndex(newTensorMetadata)
		}
		return fmt.Sprintf("Tensor %s created with type %s", tensorName, query.DataType), nil

	case InsertTensorQuery:
		metadata, err := e.storage.LoadTensorMetadata(query.TensorNames[0])
		if err != nil {
			return nil, fmt.Errorf("tensor '%s' not found for insert: %w", query.TensorNames[0], err)
		}
		expectedElements := 0
		if len(metadata.Shape) == 0 {
			expectedElements = 1
		} else {
			expectedElements = 1
			isZeroDim := false
			for _, d := range metadata.Shape {
				if d == 0 {
					isZeroDim = true
					break
				}
				expectedElements *= d
			}
			if isZeroDim {
				expectedElements = 0
			}
		}

		if query.RawData != nil && len(query.RawData) > 0 {
			elementSize, errSize := GetElementSize(metadata.DataType)
			if errSize != nil {
				return nil, fmt.Errorf("cannot determine element size for raw data insert: %w", errSize)
			}
			if elementSize == 0 {
				return nil, fmt.Errorf("element size is zero for data type %s, cannot process raw data", metadata.DataType)
			}
			numElementsFromRaw := len(query.RawData) / elementSize
			if len(query.RawData)%elementSize != 0 {
				return nil, fmt.Errorf("raw data size (%d) is not a multiple of element size (%d) for data type %s", len(query.RawData), elementSize, metadata.DataType)
			}
			if numElementsFromRaw != expectedElements {
				return nil, fmt.Errorf("raw data provides %d elements, but tensor '%s' of shape %v requires %d elements",
					numElementsFromRaw, metadata.Name, metadata.Shape, expectedElements)
			}
			switch metadata.DataType {
			case DataTypeFloat32:
				typedData := make([]float32, numElementsFromRaw)
				reader := bytes.NewReader(query.RawData)
				if err := binary.Read(reader, binary.LittleEndian, &typedData); err != nil {
					return nil, fmt.Errorf("failed to deserialize raw data to []float32: %w", err)
				}
				tempTensor, _ := NewTensor[float32](metadata.Name, metadata.Shape, metadata.DataType)
				tempTensor.SetData(typedData)
				SaveTensor(e.storage, tempTensor)
			case DataTypeFloat64:
				typedData := make([]float64, numElementsFromRaw)
				reader := bytes.NewReader(query.RawData)
				if err := binary.Read(reader, binary.LittleEndian, &typedData); err != nil {
					return nil, fmt.Errorf("failed to deserialize raw data to []float64: %w", err)
				}
				tempTensor, _ := NewTensor[float64](metadata.Name, metadata.Shape, metadata.DataType)
				tempTensor.SetData(typedData)
				SaveTensor(e.storage, tempTensor)
			case DataTypeInt32:
				typedData := make([]int32, numElementsFromRaw)
				reader := bytes.NewReader(query.RawData)
				if err := binary.Read(reader, binary.LittleEndian, &typedData); err != nil {
					return nil, fmt.Errorf("failed to deserialize raw data to []int32: %w", err)
				}
				tempTensor, _ := NewTensor[int32](metadata.Name, metadata.Shape, metadata.DataType)
				tempTensor.SetData(typedData)
				SaveTensor(e.storage, tempTensor)
			case DataTypeInt64:
				typedData := make([]int64, numElementsFromRaw)
				reader := bytes.NewReader(query.RawData)
				if err := binary.Read(reader, binary.LittleEndian, &typedData); err != nil {
					return nil, fmt.Errorf("failed to deserialize raw data to []int64: %w", err)
				}
				tempTensor, _ := NewTensor[int64](metadata.Name, metadata.Shape, metadata.DataType)
				tempTensor.SetData(typedData)
				SaveTensor(e.storage, tempTensor)
			default:
				return nil, fmt.Errorf("unsupported data type '%s' for raw data insert into tensor '%s'", metadata.DataType, metadata.Name)
			}
			return fmt.Sprintf("Raw data inserted into %s", query.TensorNames[0]), nil
		}

		numElementsToInsertFromString := len(query.Data)
		if numElementsToInsertFromString == 0 && expectedElements == 0 {
			switch metadata.DataType {
			case DataTypeFloat32:
				tempTensor, _ := NewTensor[float32](metadata.Name, metadata.Shape, metadata.DataType)
				tempTensor.SetData([]float32{})
				SaveTensor(e.storage, tempTensor)
			case DataTypeFloat64:
				tempTensor, _ := NewTensor[float64](metadata.Name, metadata.Shape, metadata.DataType)
				tempTensor.SetData([]float64{})
				SaveTensor(e.storage, tempTensor)
			case DataTypeInt32:
				tempTensor, _ := NewTensor[int32](metadata.Name, metadata.Shape, metadata.DataType)
				tempTensor.SetData([]int32{})
				SaveTensor(e.storage, tempTensor)
			case DataTypeInt64:
				tempTensor, _ := NewTensor[int64](metadata.Name, metadata.Shape, metadata.DataType)
				tempTensor.SetData([]int64{})
				SaveTensor(e.storage, tempTensor)
			default:
				return nil, fmt.Errorf("unsupported data type '%s' for empty string insert into tensor '%s'", metadata.DataType, metadata.Name)
			}
			return fmt.Sprintf("Data inserted into %s (0 elements from string)", query.TensorNames[0]), nil
		}

		if numElementsToInsertFromString != expectedElements {
			return nil, fmt.Errorf("string data provides %d elements, but tensor '%s' of shape %v requires %d elements",
				numElementsToInsertFromString, metadata.Name, metadata.Shape, expectedElements)
		}

		switch metadata.DataType {
		case DataTypeFloat32:
			typedData := make([]float32, numElementsToInsertFromString)
			for i, sVal := range query.Data {
				val, errFloat := strconv.ParseFloat(sVal, 32)
				if errFloat != nil {
					return nil, fmt.Errorf("error parsing '%s' as float32: %w", sVal, errFloat)
				}
				typedData[i] = float32(val)
			}
			tempTensor, _ := NewTensor[float32](metadata.Name, metadata.Shape, metadata.DataType)
			tempTensor.SetData(typedData)
			SaveTensor(e.storage, tempTensor)
		case DataTypeFloat64:
			typedData := make([]float64, numElementsToInsertFromString)
			for i, sVal := range query.Data {
				val, errFloat := strconv.ParseFloat(sVal, 64)
				if errFloat != nil {
					return nil, fmt.Errorf("error parsing '%s' as float64: %w", sVal, errFloat)
				}
				typedData[i] = val
			}
			tempTensor, _ := NewTensor[float64](metadata.Name, metadata.Shape, metadata.DataType)
			tempTensor.SetData(typedData)
			SaveTensor(e.storage, tempTensor)
		case DataTypeInt32:
			typedData := make([]int32, numElementsToInsertFromString)
			for i, sVal := range query.Data {
				val, errInt := strconv.ParseInt(sVal, 10, 32)
				if errInt != nil {
					return nil, fmt.Errorf("error parsing '%s' as int32: %w", sVal, errInt)
				}
				typedData[i] = int32(val)
			}
			tempTensor, _ := NewTensor[int32](metadata.Name, metadata.Shape, metadata.DataType)
			tempTensor.SetData(typedData)
			SaveTensor(e.storage, tempTensor)
		case DataTypeInt64:
			typedData := make([]int64, numElementsToInsertFromString)
			for i, sVal := range query.Data {
				val, errInt := strconv.ParseInt(sVal, 10, 64)
				if errInt != nil {
					return nil, fmt.Errorf("error parsing '%s' as int64: %w", sVal, errInt)
				}
				typedData[i] = val
			}
			tempTensor, _ := NewTensor[int64](metadata.Name, metadata.Shape, metadata.DataType)
			tempTensor.SetData(typedData)
			SaveTensor(e.storage, tempTensor)
		default:
			return nil, fmt.Errorf("unsupported data type '%s' for string data insert into tensor '%s'", metadata.DataType, metadata.Name)
		}
		return fmt.Sprintf("String data inserted into %s", query.TensorNames[0]), nil

	case SelectTensorQuery:
		tensorName := query.TensorNames[0]
		metadata, err := e.storage.LoadTensorMetadata(tensorName)
		if err != nil {
			return nil, fmt.Errorf("tensor '%s' not found for select: %w", tensorName, err)
		}
		var formattedResult interface{}
		currentSliceDef := [][2]int{}
		if len(query.Slices) > 0 {
			currentSliceDef = query.Slices[0]
		}

		switch metadata.DataType {
		case DataTypeFloat32:
			tensorInstance, errLoad := loadFullTensorTyped[float32](e, tensorName, metadata)
			if errLoad != nil {
				return nil, errLoad
			}
			if len(currentSliceDef) > 0 {
				slicedData, errSlice := tensorInstance.GetSlice(currentSliceDef)
				if errSlice != nil {
					return nil, fmt.Errorf("failed to slice %s: %w", tensorName, errSlice)
				}
				sliceShape := make([]int, len(currentSliceDef))
				for i, r := range currentSliceDef {
					sliceShape[i] = r[1] - r[0]
				}
				tempTensor, _ := NewTensor[float32]("sliced_"+tensorInstance.Name, sliceShape, tensorInstance.DataType)
				tempTensor.SetData(slicedData)
				formattedResult = tempTensor.FormatMultidimensional()
			} else {
				formattedResult = tensorInstance.FormatMultidimensional()
			}
		case DataTypeFloat64:
			tensorInstance, errLoad := loadFullTensorTyped[float64](e, tensorName, metadata)
			if errLoad != nil {
				return nil, errLoad
			}
			if len(currentSliceDef) > 0 {
				slicedData, errSlice := tensorInstance.GetSlice(currentSliceDef)
				if errSlice != nil {
					return nil, fmt.Errorf("failed to slice %s: %w", tensorName, errSlice)
				}
				sliceShape := make([]int, len(currentSliceDef))
				for i, r := range currentSliceDef {
					sliceShape[i] = r[1] - r[0]
				}
				tempTensor, _ := NewTensor[float64]("sliced_"+tensorInstance.Name, sliceShape, tensorInstance.DataType)
				tempTensor.SetData(slicedData)
				formattedResult = tempTensor.FormatMultidimensional()
			} else {
				formattedResult = tensorInstance.FormatMultidimensional()
			}
		case DataTypeInt32:
			tensorInstance, errLoad := loadFullTensorTyped[int32](e, tensorName, metadata)
			if errLoad != nil {
				return nil, errLoad
			}
			if len(currentSliceDef) > 0 {
				slicedData, errSlice := tensorInstance.GetSlice(currentSliceDef)
				if errSlice != nil {
					return nil, fmt.Errorf("failed to slice %s: %w", tensorName, errSlice)
				}
				sliceShape := make([]int, len(currentSliceDef))
				for i, r := range currentSliceDef {
					sliceShape[i] = r[1] - r[0]
				}
				tempTensor, _ := NewTensor[int32]("sliced_"+tensorInstance.Name, sliceShape, tensorInstance.DataType)
				tempTensor.SetData(slicedData)
				formattedResult = tempTensor.FormatMultidimensional()
			} else {
				formattedResult = tensorInstance.FormatMultidimensional()
			}
		case DataTypeInt64:
			tensorInstance, errLoad := loadFullTensorTyped[int64](e, tensorName, metadata)
			if errLoad != nil {
				return nil, errLoad
			}
			if len(currentSliceDef) > 0 {
				slicedData, errSlice := tensorInstance.GetSlice(currentSliceDef)
				if errSlice != nil {
					return nil, fmt.Errorf("failed to slice %s: %w", tensorName, errSlice)
				}
				sliceShape := make([]int, len(currentSliceDef))
				for i, r := range currentSliceDef {
					sliceShape[i] = r[1] - r[0]
				}
				tempTensor, _ := NewTensor[int64]("sliced_"+tensorInstance.Name, sliceShape, tensorInstance.DataType)
				tempTensor.SetData(slicedData)
				formattedResult = tempTensor.FormatMultidimensional()
			} else {
				formattedResult = tensorInstance.FormatMultidimensional()
			}
		default:
			return nil, fmt.Errorf("unsupported data type for SELECT on tensor %s: %s", tensorName, metadata.DataType)
		}
		return formattedResult, nil

	case GetDataTensorQuery:
		allResultsNonGeneric := make([][]TensorDataResult, len(query.TensorNames))
		var wg sync.WaitGroup
		errChan := make(chan error, len(query.TensorNames))
		resultChan := make(chan struct {
			index int
			data  []TensorDataResult
		}, len(query.TensorNames))

		for i, tensorName := range query.TensorNames {
			wg.Add(1)
			var currentTensorSlices [][2]int
			if query.Slices != nil && i < len(query.Slices) {
				currentTensorSlices = query.Slices[i]
			}
			go func(idx int, tName string, currentSlicesForThisTensor [][2]int) {
				defer wg.Done()
				metadata, errMeta := e.storage.LoadTensorMetadata(tName)
				if errMeta != nil {
					errChan <- fmt.Errorf("tensor '%s' not found for get data: %w", tName, errMeta)
					return
				}
				var typedResults []TensorDataResult
				var execErr error
				inferenceSliceArg := [][][2]int{currentSlicesForThisTensor}

				switch metadata.DataType {
				case DataTypeFloat32:
					tensorInstance, errLoad := loadFullTensorTyped[float32](e, tName, metadata)
					if errLoad != nil {
						execErr = errLoad
						break
					}
					genericDataBatched, errInfer := tensorInstance.GetDataForInference(inferenceSliceArg, query.BatchSize)
					if errInfer != nil {
						execErr = errInfer
						break
					}
					typedResults = make([]TensorDataResult, len(genericDataBatched))
					for k, gd := range genericDataBatched {
						typedResults[k] = TensorDataResult{Name: gd.Name, Shape: gd.Shape, NumDimensions: gd.NumDimensions, DataType: gd.DataType, TotalElements: gd.TotalElements, DataSizeBytes: gd.DataSizeBytes, Strides: gd.Strides, BatchInfo: gd.BatchInfo, Data: gd.Data}
					}
				case DataTypeFloat64:
					tensorInstance, errLoad := loadFullTensorTyped[float64](e, tName, metadata)
					if errLoad != nil {
						execErr = errLoad
						break
					}
					genericDataBatched, errInfer := tensorInstance.GetDataForInference(inferenceSliceArg, query.BatchSize)
					if errInfer != nil {
						execErr = errInfer
						break
					}
					typedResults = make([]TensorDataResult, len(genericDataBatched))
					for k, gd := range genericDataBatched {
						typedResults[k] = TensorDataResult{Name: gd.Name, Shape: gd.Shape, NumDimensions: gd.NumDimensions, DataType: gd.DataType, TotalElements: gd.TotalElements, DataSizeBytes: gd.DataSizeBytes, Strides: gd.Strides, BatchInfo: gd.BatchInfo, Data: gd.Data}
					}
				case DataTypeInt32:
					tensorInstance, errLoad := loadFullTensorTyped[int32](e, tName, metadata)
					if errLoad != nil {
						execErr = errLoad
						break
					}
					genericDataBatched, errInfer := tensorInstance.GetDataForInference(inferenceSliceArg, query.BatchSize)
					if errInfer != nil {
						execErr = errInfer
						break
					}
					typedResults = make([]TensorDataResult, len(genericDataBatched))
					for k, gd := range genericDataBatched {
						typedResults[k] = TensorDataResult{Name: gd.Name, Shape: gd.Shape, NumDimensions: gd.NumDimensions, DataType: gd.DataType, TotalElements: gd.TotalElements, DataSizeBytes: gd.DataSizeBytes, Strides: gd.Strides, BatchInfo: gd.BatchInfo, Data: gd.Data}
					}
				case DataTypeInt64:
					tensorInstance, errLoad := loadFullTensorTyped[int64](e, tName, metadata)
					if errLoad != nil {
						execErr = errLoad
						break
					}
					genericDataBatched, errInfer := tensorInstance.GetDataForInference(inferenceSliceArg, query.BatchSize)
					if errInfer != nil {
						execErr = errInfer
						break
					}
					typedResults = make([]TensorDataResult, len(genericDataBatched))
					for k, gd := range genericDataBatched {
						typedResults[k] = TensorDataResult{Name: gd.Name, Shape: gd.Shape, NumDimensions: gd.NumDimensions, DataType: gd.DataType, TotalElements: gd.TotalElements, DataSizeBytes: gd.DataSizeBytes, Strides: gd.Strides, BatchInfo: gd.BatchInfo, Data: gd.Data}
					}
				default:
					execErr = fmt.Errorf("unsupported data type for GET DATA on tensor %s: %s", tName, metadata.DataType)
				}
				if execErr != nil {
					errChan <- fmt.Errorf("failed to get data for inference from '%s': %w", tName, execErr)
					return
				}
				resultChan <- struct {
					index int
					data  []TensorDataResult
				}{index: idx, data: typedResults}
			}(i, tensorName, currentTensorSlices)
		}
		wg.Wait()
		close(resultChan)
		close(errChan)
		var multiErr []string
		for errItem := range errChan {
			if errItem != nil {
				multiErr = append(multiErr, errItem.Error())
			}
		}
		if len(multiErr) > 0 {
			return nil, errors.New("errors occurred during GET DATA: " + strings.Join(multiErr, "; "))
		}
		for resultItem := range resultChan {
			allResultsNonGeneric[resultItem.index] = resultItem.data
		}
		if len(query.TensorNames) == 1 {
			if len(allResultsNonGeneric) > 0 && len(allResultsNonGeneric[0]) > 0 {
				return allResultsNonGeneric[0], nil
			}
			_, metaErr := e.storage.LoadTensorMetadata(query.TensorNames[0])
			if metaErr != nil {
				return nil, fmt.Errorf("no data returned and tensor '%s' not found for single tensor GET DATA query", query.TensorNames[0])
			}
			if len(allResultsNonGeneric) > 0 && len(allResultsNonGeneric[0]) == 0 {
				return []TensorDataResult{}, nil
			}
			return nil, fmt.Errorf("no data returned for single tensor GET DATA query on '%s', and result structure is unexpected", query.TensorNames[0])
		}
		return allResultsNonGeneric, nil

	case MathOperationQuery:
		var finalResultTensor interface{}
		var operationError error
		_, errOutputCheck := e.storage.LoadTensorMetadata(query.OutputTensorName)
		if errOutputCheck == nil {
			return nil, fmt.Errorf("output tensor '%s' already exists. Math operations require a new output tensor name", query.OutputTensorName)
		}
		if !os.IsNotExist(errors.Unwrap(errOutputCheck)) && errOutputCheck != nil && !strings.Contains(errOutputCheck.Error(), "failed to read metadata") {
			return nil, fmt.Errorf("error checking existing output tensor '%s': %w", query.OutputTensorName, errOutputCheck)
		}

		switch query.MathOperator {
		case "ADD_TENSORS":
			if len(query.InputTensorNames) != 2 {
				operationError = errors.New("ADD_TENSORS operation requires two input tensors")
				break
			}
			tensorAName := query.InputTensorNames[0]
			tensorBName := query.InputTensorNames[1]
			metaA, errA := e.storage.LoadTensorMetadata(tensorAName)
			if errA != nil {
				operationError = fmt.Errorf("failed to load metadata for tensor A '%s': %w", tensorAName, errA)
				break
			}
			metaB, errB := e.storage.LoadTensorMetadata(tensorBName)
			if errB != nil {
				operationError = fmt.Errorf("failed to load metadata for tensor B '%s': %w", tensorBName, errB)
				break
			}
			if metaA.DataType != metaB.DataType {
				operationError = fmt.Errorf("data types of %s (%s) and %s (%s) do not match for ADD_TENSORS", tensorAName, metaA.DataType, tensorBName, metaB.DataType)
				break
			}

			switch metaA.DataType {
			case DataTypeFloat32:
				tA, loadErrA := loadFullTensorTyped[float32](e, tensorAName, metaA)
				if loadErrA != nil {
					operationError = loadErrA
					break
				}
				tB, loadErrB := loadFullTensorTyped[float32](e, tensorBName, metaB)
				if loadErrB != nil {
					operationError = loadErrB
					break
				}
				resTensor, opErr := AddTensors[float32](tA, tB)
				if opErr != nil {
					operationError = opErr
					break
				}
				resTensor.Name = query.OutputTensorName
				finalResultTensor = resTensor
			case DataTypeFloat64:
				tA, loadErrA := loadFullTensorTyped[float64](e, tensorAName, metaA)
				if loadErrA != nil {
					operationError = loadErrA
					break
				}
				tB, loadErrB := loadFullTensorTyped[float64](e, tensorBName, metaB)
				if loadErrB != nil {
					operationError = loadErrB
					break
				}
				resTensor, opErr := AddTensors[float64](tA, tB)
				if opErr != nil {
					operationError = opErr
					break
				}
				resTensor.Name = query.OutputTensorName
				finalResultTensor = resTensor
			case DataTypeInt32:
				tA, loadErrA := loadFullTensorTyped[int32](e, tensorAName, metaA)
				if loadErrA != nil {
					operationError = loadErrA
					break
				}
				tB, loadErrB := loadFullTensorTyped[int32](e, tensorBName, metaB)
				if loadErrB != nil {
					operationError = loadErrB
					break
				}
				resTensor, opErr := AddTensors[int32](tA, tB)
				if opErr != nil {
					operationError = opErr
					break
				}
				resTensor.Name = query.OutputTensorName
				finalResultTensor = resTensor
			case DataTypeInt64:
				tA, loadErrA := loadFullTensorTyped[int64](e, tensorAName, metaA)
				if loadErrA != nil {
					operationError = loadErrA
					break
				}
				tB, loadErrB := loadFullTensorTyped[int64](e, tensorBName, metaB)
				if loadErrB != nil {
					operationError = loadErrB
					break
				}
				resTensor, opErr := AddTensors[int64](tA, tB)
				if opErr != nil {
					operationError = opErr
					break
				}
				resTensor.Name = query.OutputTensorName
				finalResultTensor = resTensor
			default:
				operationError = fmt.Errorf("unsupported data type for ADD_TENSORS operation: %s", metaA.DataType)
			}
		case "ADD_SCALAR":
			if len(query.InputTensorNames) != 1 || query.ScalarOperand == "" {
				operationError = errors.New("ADD_SCALAR operation requires one input tensor and a scalar operand")
				break
			}
			tensorAName := query.InputTensorNames[0]
			metaA, errA := e.storage.LoadTensorMetadata(tensorAName)
			if errA != nil {
				operationError = fmt.Errorf("failed to load metadata for tensor '%s': %w", tensorAName, errA)
				break
			}

			switch metaA.DataType {
			case DataTypeFloat32:
				tA, loadErrA := loadFullTensorTyped[float32](e, tensorAName, metaA)
				if loadErrA != nil {
					operationError = loadErrA
					break
				}
				scalarVal, parseErr := strconv.ParseFloat(query.ScalarOperand, 32)
				if parseErr != nil {
					operationError = fmt.Errorf("failed to parse scalar operand '%s' as float32: %w", query.ScalarOperand, parseErr)
					break
				}
				resTensor, opErr := AddScalarToTensor[float32](tA, float32(scalarVal))
				if opErr != nil {
					operationError = opErr
					break
				}
				resTensor.Name = query.OutputTensorName
				finalResultTensor = resTensor
			case DataTypeFloat64:
				tA, loadErrA := loadFullTensorTyped[float64](e, tensorAName, metaA)
				if loadErrA != nil {
					operationError = loadErrA
					break
				}
				scalarVal, parseErr := strconv.ParseFloat(query.ScalarOperand, 64)
				if parseErr != nil {
					operationError = fmt.Errorf("failed to parse scalar operand '%s' as float64: %w", query.ScalarOperand, parseErr)
					break
				}
				resTensor, opErr := AddScalarToTensor[float64](tA, scalarVal)
				if opErr != nil {
					operationError = opErr
					break
				}
				resTensor.Name = query.OutputTensorName
				finalResultTensor = resTensor
			case DataTypeInt32:
				tA, loadErrA := loadFullTensorTyped[int32](e, tensorAName, metaA)
				if loadErrA != nil {
					operationError = loadErrA
					break
				}
				scalarVal, parseErr := strconv.ParseInt(query.ScalarOperand, 10, 32)
				if parseErr != nil {
					operationError = fmt.Errorf("failed to parse scalar operand '%s' as int32: %w", query.ScalarOperand, parseErr)
					break
				}
				resTensor, opErr := AddScalarToTensor[int32](tA, int32(scalarVal))
				if opErr != nil {
					operationError = opErr
					break
				}
				resTensor.Name = query.OutputTensorName
				finalResultTensor = resTensor
			case DataTypeInt64:
				tA, loadErrA := loadFullTensorTyped[int64](e, tensorAName, metaA)
				if loadErrA != nil {
					operationError = loadErrA
					break
				}
				scalarVal, parseErr := strconv.ParseInt(query.ScalarOperand, 10, 64)
				if parseErr != nil {
					operationError = fmt.Errorf("failed to parse scalar operand '%s' as int64: %w", query.ScalarOperand, parseErr)
					break
				}
				resTensor, opErr := AddScalarToTensor[int64](tA, scalarVal)
				if opErr != nil {
					operationError = opErr
					break
				}
				resTensor.Name = query.OutputTensorName
				finalResultTensor = resTensor
			default:
				operationError = fmt.Errorf("unsupported data type for ADD_SCALAR operation: %s", metaA.DataType)
			}
		default:
			return nil, fmt.Errorf("unsupported mathematical operator: %s", query.MathOperator)
		}
		if operationError != nil {
			return nil, operationError
		}
		if finalResultTensor != nil {
			var resultMetadata *TensorMetadata
			switch rt := finalResultTensor.(type) {
			case *Tensor[float32]:
				if err := SaveTensor(e.storage, rt); err != nil {
					return nil, fmt.Errorf("failed to save result tensor '%s': %w", rt.Name, err)
				}
				resultMetadata = &TensorMetadata{Name: rt.Name, Shape: rt.Shape, DataType: rt.DataType, Strides: rt.Strides}
			case *Tensor[float64]:
				if err := SaveTensor(e.storage, rt); err != nil {
					return nil, fmt.Errorf("failed to save result tensor '%s': %w", rt.Name, err)
				}
				resultMetadata = &TensorMetadata{Name: rt.Name, Shape: rt.Shape, DataType: rt.DataType, Strides: rt.Strides}
			case *Tensor[int32]:
				if err := SaveTensor(e.storage, rt); err != nil {
					return nil, fmt.Errorf("failed to save result tensor '%s': %w", rt.Name, err)
				}
				resultMetadata = &TensorMetadata{Name: rt.Name, Shape: rt.Shape, DataType: rt.DataType, Strides: rt.Strides}
			case *Tensor[int64]:
				if err := SaveTensor(e.storage, rt); err != nil {
					return nil, fmt.Errorf("failed to save result tensor '%s': %w", rt.Name, err)
				}
				resultMetadata = &TensorMetadata{Name: rt.Name, Shape: rt.Shape, DataType: rt.DataType, Strides: rt.Strides}
			default:
				return nil, fmt.Errorf("unknown type for result tensor, cannot save or index")
			}
			if resultMetadata != nil {
				e.storage.AddTensorToIndex(resultMetadata)
			}
			return fmt.Sprintf("Tensor '%s' created successfully from operation %s", query.OutputTensorName, query.MathOperator), nil
		}
		return nil, fmt.Errorf("math operation did not produce a result tensor")

	case ListTensorsQuery:
		tensorNames := e.storage.QueryIndex(query.FilterDataType, query.FilterNumDimensions)
		results := make([]TensorMetadata, 0, len(tensorNames))
		for _, name := range tensorNames {
			meta, err := e.storage.LoadTensorMetadata(name)
			if err == nil && meta != nil {
				resultMeta := TensorMetadata{Name: meta.Name, Shape: meta.Shape, DataType: meta.DataType, Strides: meta.Strides}
				results = append(results, resultMeta)
			} else if err != nil {
				fmt.Fprintf(os.Stderr, "Warning: could not load metadata for tensor '%s' during LIST TENSORS: %v\n", name, err)
			}
		}
		return results, nil

	default:
		return nil, fmt.Errorf("unsupported query type: %s", query.Type)
	}
}
