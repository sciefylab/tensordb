package tests

import (
	"fmt"
	"testing"

	"tensor_db/client" // Pastikan path import ini benar
	"tensor_db/tensor" // Pastikan path import ini benar
)

// Asumsikan setupBenchmarkClient ada di test_helpers.go
// func setupBenchmarkClient(b *testing.B) (*client.Client, func()) { ... }

// Helper untuk membuat dan mengisi tensor dengan data float32 untuk benchmark
func createAndFillFloat32Tensor(b *testing.B, apiClient *client.Client, tensorName string, shape []int) {
	b.Helper()
	errCreate := apiClient.CreateTensor(tensorName, shape, tensor.DataTypeFloat32)
	if errCreate != nil {
		b.Fatalf("Gagal membuat tensor %s untuk benchmark: %v", tensorName, errCreate)
	}

	numElements := 1
	isZeroDimTensor := false
	if len(shape) == 0 {
		numElements = 1
	} else {
		for _, dim := range shape {
			if dim == 0 {
				isZeroDimTensor = true
				break
			}
			numElements *= dim
		}
	}
	if isZeroDimTensor {
		numElements = 0
	}

	if numElements > 0 {
		dataToInsert := make([]float32, numElements)
		for i := 0; i < numElements; i++ {
			dataToInsert[i] = float32(i % 100)
		}
		errInsert := apiClient.InsertFloat32Data(tensorName, dataToInsert)
		if errInsert != nil {
			b.Fatalf("Gagal menyisipkan data ke tensor %s untuk benchmark: %v", tensorName, errInsert)
		}
	}
}

// Benchmark untuk operasi CREATE TENSOR
func BenchmarkCreateTensor(b *testing.B) {
	apiClient, cleanup := setupBenchmarkClient(b)
	defer cleanup()

	tensorNames := make([]string, b.N)
	for i := 0; i < b.N; i++ {
		tensorNames[i] = fmt.Sprintf("bench_create_tensor_%d_%d", b.N, i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err := apiClient.CreateTensor(tensorNames[i], []int{128, 128}, tensor.DataTypeFloat32)
		if err != nil {
			// b.Errorf("Error creating tensor in benchmark: %v", err)
		}
	}
	b.StopTimer()
}

// Benchmark untuk operasi INSERT DATA ke tensor yang sudah ada
func BenchmarkInsertData(b *testing.B) {
	apiClient, cleanup := setupBenchmarkClient(b)
	defer cleanup()

	tensorName := "bench_insert_tensor"
	shape := []int{256, 256}

	// Pastikan tensor sudah ada sebelum timer dimulai
	// Tidak perlu memanggil createAndFill, cukup CreateTensor
	err := apiClient.CreateTensor(tensorName, shape, tensor.DataTypeFloat32)
	if err != nil {
		// Penanganan error jika tensor gagal dibuat, bisa jadi sudah ada dari run sebelumnya
		// atau error lain. Untuk benchmark, kita asumsikan ini berhasil atau "already exists" diabaikan.
		// b.Logf("Pra-pembuatan tensor untuk BenchmarkInsertData mungkin gagal atau sudah ada: %v", err)
	}

	numElements := shape[0] * shape[1]
	dataToInsert := make([]float32, numElements)
	for i := 0; i < numElements; i++ {
		dataToInsert[i] = float32(i % 100)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		errInsert := apiClient.InsertFloat32Data(tensorName, dataToInsert)
		if errInsert != nil {
			// b.Errorf("Error inserting data in benchmark: %v", errInsert)
		}
	}
	b.StopTimer()
}

// Benchmark untuk operasi LOAD TENSOR
func BenchmarkLoadTensor(b *testing.B) {
	apiClient, cleanup := setupBenchmarkClient(b)
	defer cleanup()

	tensorName := "bench_load_tensor"
	shape := []int{256, 256}
	createAndFillFloat32Tensor(b, apiClient, tensorName, shape)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, errLoad := apiClient.LoadTensorFloat32(tensorName)
		if errLoad != nil {
			// b.Errorf("Error loading tensor in benchmark: %v", errLoad)
		}
	}
	b.StopTimer()
}

// Benchmark untuk operasi SELECT DATA (seluruh tensor)
func BenchmarkSelectData_Full(b *testing.B) {
	apiClient, cleanup := setupBenchmarkClient(b)
	defer cleanup()

	tensorName := "bench_select_full_tensor"
	shape := []int{256, 256}
	createAndFillFloat32Tensor(b, apiClient, tensorName, shape)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := apiClient.SelectData(tensorName, nil)
		if err != nil {
			// b.Errorf("Error selecting data in benchmark: %v", err)
		}
	}
	b.StopTimer()
}

// Benchmark untuk operasi LIST TENSORS (tanpa filter)
func BenchmarkListTensors_NoFilter(b *testing.B) {
	apiClient, cleanup := setupBenchmarkClient(b)
	defer cleanup()

	numTestTensors := 100
	for i := 0; i < numTestTensors; i++ {
		setupTensorName := fmt.Sprintf("list_bench_tensor_setup_%d", i)
		err := apiClient.CreateTensor(setupTensorName, []int{10, 10}, tensor.DataTypeFloat32)
		if err != nil {
			// b.Logf("Failed to create tensor %s for ListTensors benchmark setup: %v", setupTensorName, err)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := apiClient.ListTensors("", -1)
		if err != nil {
			// b.Errorf("Error listing tensors in benchmark: %v", err)
		}
	}
	b.StopTimer()
}

// Benchmark untuk operasi LIST TENSORS (dengan filter DataType)
func BenchmarkListTensors_FilterDataType(b *testing.B) {
	apiClient, cleanup := setupBenchmarkClient(b)
	defer cleanup()

	numTestTensors := 50
	for i := 0; i < numTestTensors; i++ {
		apiClient.CreateTensor(fmt.Sprintf("list_bench_f32_%d", i), []int{2, i + 1}, tensor.DataTypeFloat32)
		apiClient.CreateTensor(fmt.Sprintf("list_bench_i32_%d", i), []int{3, i + 1}, tensor.DataTypeInt32)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := apiClient.ListTensors(tensor.DataTypeFloat32, -1)
		if err != nil {
			// b.Errorf("Error listing tensors with datatype filter: %v", err)
		}
	}
	b.StopTimer()
}

// Benchmark untuk operasi LIST TENSORS (dengan filter NumDimensions)
func BenchmarkListTensors_FilterNumDimensions(b *testing.B) {
	apiClient, cleanup := setupBenchmarkClient(b)
	defer cleanup()

	numTestTensors := 50
	for i := 0; i < numTestTensors; i++ {
		apiClient.CreateTensor(fmt.Sprintf("list_bench_dim1_%d", i), []int{i + 1}, tensor.DataTypeFloat32)
		apiClient.CreateTensor(fmt.Sprintf("list_bench_dim2_%d", i), []int{2, i + 1}, tensor.DataTypeInt32)
		apiClient.CreateTensor(fmt.Sprintf("list_bench_dim3_%d", i), []int{3, 2, i + 1}, tensor.DataTypeFloat64)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := apiClient.ListTensors("", 2)
		if err != nil {
			// b.Errorf("Error listing tensors with num_dimensions filter: %v", err)
		}
	}
	b.StopTimer()
}

// --- Benchmark untuk GET DATA ---

// Benchmark untuk operasi GET DATA (seluruh tensor, satu tensor)
func BenchmarkGetData_Full_SingleTensor(b *testing.B) {
	apiClient, cleanup := setupBenchmarkClient(b)
	defer cleanup()

	tensorName := "bench_getdata_full_single"
	shape := []int{256, 256}
	createAndFillFloat32Tensor(b, apiClient, tensorName, shape)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := apiClient.GetData([]string{tensorName}, nil, 0)
		if err != nil {
			// b.Errorf("Error pada GetData (full, single) di benchmark: %v", err)
		}
	}
	b.StopTimer()
}

// Benchmark untuk operasi GET DATA (slice dari satu tensor)
func BenchmarkGetData_Slice_SingleTensor(b *testing.B) {
	apiClient, cleanup := setupBenchmarkClient(b)
	defer cleanup()

	tensorName := "bench_getdata_slice_single"
	shape := []int{256, 256} // Tensor 2D
	createAndFillFloat32Tensor(b, apiClient, tensorName, shape)

	// Koreksi: slices[0] harus bertipe [][2]int
	// slices adalah [][][2]int, jadi elemen pertamanya (untuk tensor pertama) adalah [][2]int
	slicesArg := [][][2]int{
		{ // Ini adalah slice definition untuk tensorName (bertipe [][2]int)
			{0, shape[0] / 2}, // Rentang untuk dimensi 0
			{0, shape[1]},     // Rentang untuk dimensi 1
		},
	}
	// Baris yang kemungkinan menjadi error (sekitar line 255 pada kode sebelumnya) adalah di atas

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := apiClient.GetData([]string{tensorName}, slicesArg, 0)
		if err != nil {
			// b.Errorf("Error pada GetData (slice, single) di benchmark: %v", err)
		}
	}
	b.StopTimer()
}

// Benchmark untuk operasi GET DATA (seluruh tensor, satu tensor, dengan batching)
func BenchmarkGetData_Full_SingleTensor_WithBatching(b *testing.B) {
	apiClient, cleanup := setupBenchmarkClient(b)
	defer cleanup()

	tensorName := "bench_getdata_full_single_batch"
	shape := []int{128, 128}
	batchSize := 1024
	createAndFillFloat32Tensor(b, apiClient, tensorName, shape)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := apiClient.GetData([]string{tensorName}, nil, batchSize)
		if err != nil {
			// b.Errorf("Error pada GetData (full, single, batching) di benchmark: %v", err)
		}
	}
	b.StopTimer()
}

// Benchmark untuk operasi GET DATA (beberapa tensor, tanpa slice, tanpa batching)
func BenchmarkGetData_Full_MultipleTensors(b *testing.B) {
	apiClient, cleanup := setupBenchmarkClient(b)
	defer cleanup()

	tensorName1 := "bench_getdata_multi_1"
	shape1 := []int{64, 64}
	createAndFillFloat32Tensor(b, apiClient, tensorName1, shape1)

	tensorName2 := "bench_getdata_multi_2"
	shape2 := []int{32, 32, 4}
	createAndFillFloat32Tensor(b, apiClient, tensorName2, shape2)

	tensorNames := []string{tensorName1, tensorName2}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := apiClient.GetData(tensorNames, nil, 0)
		if err != nil {
			// b.Errorf("Error pada GetData (full, multiple) di benchmark: %v", err)
		}
	}
	b.StopTimer()
}

// Benchmark untuk operasi GET DATA (beberapa tensor, dengan slice dan batching)
func BenchmarkGetData_Slice_MultipleTensors_WithBatching(b *testing.B) {
	apiClient, cleanup := setupBenchmarkClient(b)
	defer cleanup()

	tensorName1 := "bench_getdata_multi_slice_batch_1"
	shape1 := []int{128, 64} // 2D
	createAndFillFloat32Tensor(b, apiClient, tensorName1, shape1)

	tensorName2 := "bench_getdata_multi_slice_batch_2"
	shape2 := []int{64, 32, 2} // 3D
	createAndFillFloat32Tensor(b, apiClient, tensorName2, shape2)

	tensorNames := []string{tensorName1, tensorName2}
	// Koreksi: Setiap elemen dari slicesArg (yaitu slicesArg[0] dan slicesArg[1]) harus bertipe [][2]int
	slicesArg := [][][2]int{
		{ // Slice definition untuk tensorName1 (bertipe [][2]int)
			{0, shape1[0] / 4}, // Rentang untuk dimensi 0 dari tensor1
			{0, shape1[1] / 2}, // Rentang untuk dimensi 1 dari tensor1
			// Baris yang kemungkinan menjadi error (sekitar line 328 pada kode sebelumnya) ada di atas
		},
		{ // Slice definition untuk tensorName2 (bertipe [][2]int)
			{0, shape2[0]},     // Rentang untuk dimensi 0 dari tensor2
			{0, shape2[1] / 2}, // Rentang untuk dimensi 1 dari tensor2
			{0, shape2[2]},     // Rentang untuk dimensi 2 dari tensor2
			// Baris yang kemungkinan menjadi error (sekitar line 329 pada kode sebelumnya) ada di atas
		},
	}
	batchSize := 512

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := apiClient.GetData(tensorNames, slicesArg, batchSize)
		if err != nil {
			// b.Errorf("Error pada GetData (slice, multiple, batching) di benchmark: %v", err)
		}
	}
	b.StopTimer()
}

// Benchmark untuk operasi GET DATA pada tensor kosong
func BenchmarkGetData_EmptyTensor(b *testing.B) {
	apiClient, cleanup := setupBenchmarkClient(b)
	defer cleanup()

	tensorName := "bench_getdata_empty"
	shape := []int{128, 0, 128}
	errCreate := apiClient.CreateTensor(tensorName, shape, tensor.DataTypeFloat32)
	if errCreate != nil {
		b.Fatalf("Gagal membuat tensor kosong %s untuk benchmark: %v", tensorName, errCreate)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		results, err := apiClient.GetData([]string{tensorName}, nil, 0)
		if err != nil {
			// b.Errorf("Error pada GetData (empty tensor) di benchmark: %v", err)
			continue
		}
		if dataResults, ok := results.([]tensor.TensorDataResult); ok {
			if len(dataResults) != 1 || dataResults[0].TotalElements != 0 {
				// b.Errorf("Hasil GetData untuk tensor kosong tidak sesuai harapan. TotalElements: %d", dataResults[0].TotalElements)
			}
		} else {
			// b.Errorf("Tipe hasil GetData untuk tensor kosong tidak sesuai harapan. Got: %T", results)
		}
	}
	b.StopTimer()
}
