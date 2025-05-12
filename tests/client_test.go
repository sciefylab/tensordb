package tests

import (
	"testing"

	"tensordb/pkg/tensor"
)

// Asumsikan setupTestClient, assertEqual, assertError, assertErrorContains, assertTrue
// ada di file test_helpers.go dalam package tests_test yang sama.

func TestAPIClientOperations(t *testing.T) {
	_, apiClient, cleanup := setupTestClient(t)
	defer cleanup()

	// CreateTensor Success
	t.Run("CreateTensor_Success", func(t *testing.T) {
		err := apiClient.CreateTensor("test_create_f32_client", []int{2, 2}, tensor.DataTypeFloat32)
		assertError(t, err, false)
		err = apiClient.CreateTensor("test_create_f64_client", []int{3}, tensor.DataTypeFloat64)
		assertError(t, err, false)
	})

	// CreateTensor Error Cases
	t.Run("CreateTensor_Error_Cases", func(t *testing.T) {
		err := apiClient.CreateTensor("test_create_f32_client", []int{1}, tensor.DataTypeFloat32) // Duplicate
		assertError(t, err, true)
		assertErrorContains(t, err, "already exists")

		err = apiClient.CreateTensor("invalid_type_client", []int{1}, "badtype")
		assertError(t, err, true)
		assertErrorContains(t, err, "tipe data tidak valid 'badtype'")
	})

	// InsertData Success
	t.Run("InsertData_Success", func(t *testing.T) {
		err := apiClient.InsertFloat32Data("test_create_f32_client", []float32{1, 2, 3, 4})
		assertError(t, err, false)
		err = apiClient.InsertFloat64Data("test_create_f64_client", []float64{10, 20, 30})
		assertError(t, err, false)
	})

	// InsertData Error Cases
	t.Run("InsertData_Error_Cases", func(t *testing.T) {
		err := apiClient.InsertFloat32Data("non_existent_insert_client", []float32{1})
		assertError(t, err, true)
		assertErrorContains(t, err, "not found for insert")

		// KOREKSI PESAN ERROR DI SINI
		err = apiClient.InsertFloat32Data("test_create_f32_client", []float32{1, 2}) // Wrong number of elements
		assertError(t, err, true)
		assertErrorContains(t, err, "raw data provides 2 elements, but tensor 'test_create_f32_client' of shape [2 2] requires 4 elements")
	})

	// LoadTensor Success
	t.Run("LoadTensor_Success", func(t *testing.T) {
		loadedF32, err := apiClient.LoadTensorFloat32("test_create_f32_client")
		assertError(t, err, false)
		if err == nil {
			assertEqual(t, loadedF32.Shape, []int{2, 2})
			assertEqual(t, loadedF32.Data, []float32{1, 2, 3, 4})
		}
	})

	// LoadTensor Error Cases
	t.Run("LoadTensor_Error_Cases", func(t *testing.T) {
		tensorName := "non_existent_load_client_err"
		_, err := apiClient.LoadTensorFloat32(tensorName)
		assertError(t, err, true, "LoadTensor tensor tidak ada")
		assertErrorContains(t, err, "The system cannot find the file specified.")
	})

	// GetTensorMetadata Success
	t.Run("GetTensorMetadata_Success", func(t *testing.T) {
		meta, err := apiClient.GetTensorMetadata("test_create_f32_client")
		assertError(t, err, false)
		if err == nil {
			assertEqual(t, meta.Name, "test_create_f32_client")
			assertEqual(t, meta.Shape, []int{2, 2})
			assertEqual(t, meta.DataType, tensor.DataTypeFloat32)
		}
	})

	// GetTensorMetadata Error
	t.Run("GetTensorMetadata_Error", func(t *testing.T) {
		tensorName := "non_existent_meta_client_err"
		_, err := apiClient.GetTensorMetadata(tensorName)
		assertError(t, err, true)
		assertErrorContains(t, err, "The system cannot find the file specified.")
	})

	// SelectData Success
	t.Run("SelectData_Success", func(t *testing.T) {
		errCreate := apiClient.CreateTensor("test_select_client", []int{2, 3}, tensor.DataTypeInt32)
		assertError(t, errCreate, false, "Gagal membuat tensor untuk select")
		errInsert := apiClient.InsertInt32Data("test_select_client", []int32{1, 2, 3, 4, 5, 6})
		assertError(t, errInsert, false, "Gagal insert data untuk select")

		data, err := apiClient.SelectData("test_select_client", nil)
		assertError(t, err, false)
		expectedFull := []interface{}{
			[]interface{}{int32(1), int32(2), int32(3)},
			[]interface{}{int32(4), int32(5), int32(6)},
		}
		assertEqual(t, data, expectedFull)

		dataSlice, errSlice := apiClient.SelectData("test_select_client", [][2]int{{0, 1}, {1, 3}})
		assertError(t, errSlice, false)
		expectedSlice := []interface{}{
			[]interface{}{int32(2), int32(3)},
		}
		assertEqual(t, dataSlice, expectedSlice)
	})

	// GetData Single Tensor
	t.Run("GetData_Single_Tensor", func(t *testing.T) {
		results, err := apiClient.GetData([]string{"test_select_client"}, nil, 0)
		assertError(t, err, false)
		if err == nil {
			dataResults, ok := results.([]tensor.TensorDataResult)
			assertTrue(t, ok, "Hasil GetData bukan []tensor.TensorDataResult")
			assertEqual(t, len(dataResults), 1, "GetData single tensor seharusnya mengembalikan 1 hasil")
			if len(dataResults) == 1 {
				assertEqual(t, dataResults[0].Name, "test_select_client")
				assertEqual(t, dataResults[0].Shape, []int{2, 3})
				assertEqual(t, dataResults[0].Data, []int32{1, 2, 3, 4, 5, 6})
			}
		}
	})

	// GetData Multiple Tensors with Batching
	t.Run("GetData_Multiple_Tensors_with_Batching", func(t *testing.T) {
		slices := [][][2]int{
			nil,
			{{0, 1}, {0, 2}},
		}
		results, err := apiClient.GetData([]string{"test_select_client", "test_create_f32_client"}, slices, 2)
		assertError(t, err, false)
		if err == nil {
			multiResults, ok := results.([][]tensor.TensorDataResult)
			assertTrue(t, ok, "Hasil GetData multi bukan [][]tensor.TensorDataResult")
			assertEqual(t, len(multiResults), 2, "GetData multi tensor seharusnya mengembalikan hasil untuk 2 tensor")

			assertEqual(t, len(multiResults[0]), 3, "Batching untuk test_select_client salah")
			if len(multiResults[0]) == 3 {
				assertEqual(t, multiResults[0][0].Data, []int32{1, 2})
				assertEqual(t, multiResults[0][1].Data, []int32{3, 4})
				assertEqual(t, multiResults[0][2].Data, []int32{5, 6})
			}

			assertEqual(t, len(multiResults[1]), 1, "Batching untuk test_create_f32_client salah")
			if len(multiResults[1]) == 1 {
				assertEqual(t, multiResults[1][0].Data, []float32{1, 2})
				assertEqual(t, multiResults[1][0].Shape, []int{1, 2})
			}
		}
	})
}
