package tests

import (
	"fmt"
	"sort" // Import paket sort
	"strings"
	"testing"

	"github.com/sciefylab/tensordb/pkg/tensor"
)

// Asumsikan setupTest, assertEqual, assertError, assertErrorContains, assertTrue
// ada di file test_helpers.go dalam package tests_test yang sama.

func TestTensorDBOperations(t *testing.T) {
	_, executor, cleanup := setupTest(t)
	defer cleanup()

	parser := &tensor.Parser{}

	var expectedInitialTensors []tensor.TensorMetadata
	addExpectedMeta := func(name string, shape []int, dataType string, strides []int) {
		expectedInitialTensors = append(expectedInitialTensors, tensor.TensorMetadata{
			Name: name, Shape: shape, DataType: dataType, Strides: strides,
		})
	}

	testCases := []struct {
		name          string
		query         string
		expected      interface{}
		shouldError   bool
		errorContains string
		verifyResult  func(t *testing.T, exec *tensor.Executor, p *tensor.Parser, resultName string, expectedData interface{})
		setupFunc     func() // Fungsi untuk mengisi expectedInitialTensors
	}{
		// --- CREATE TENSOR ---
		{name: "Create Tensor Float64 (Default)", query: "CREATE TENSOR my_tensor_f64 2,3", expected: "Tensor my_tensor_f64 created with type float64",
			setupFunc: func() { addExpectedMeta("my_tensor_f64", []int{2, 3}, tensor.DataTypeFloat64, []int{3, 1}) }},
		{name: "Create Tensor Float32", query: "CREATE TENSOR my_tensor_f32 2,2 TYPE float32", expected: "Tensor my_tensor_f32 created with type float32",
			setupFunc: func() { addExpectedMeta("my_tensor_f32", []int{2, 2}, tensor.DataTypeFloat32, []int{2, 1}) }},
		{name: "Create Tensor Int32", query: "CREATE TENSOR my_tensor_i32 3 TYPE int32", expected: "Tensor my_tensor_i32 created with type int32",
			setupFunc: func() { addExpectedMeta("my_tensor_i32", []int{3}, tensor.DataTypeInt32, []int{1}) }},
		{name: "Create Tensor Int64 Scalar", query: "CREATE TENSOR scalar_i64 TYPE int64", expected: "Tensor scalar_i64 created with type int64",
			setupFunc: func() { addExpectedMeta("scalar_i64", []int{}, tensor.DataTypeInt64, []int{}) }},
		// KOREKSI STRIDES UNTUK TENSOR KOSONG
		{name: "Create Tensor Empty (0,2)", query: "CREATE TENSOR empty_f32_0_2 0,2 TYPE float32", expected: "Tensor empty_f32_0_2 created with type float32",
			setupFunc: func() { addExpectedMeta("empty_f32_0_2", []int{0, 2}, tensor.DataTypeFloat32, []int{0, 0}) }},
		{name: "Create Tensor Empty (2,0)", query: "CREATE TENSOR empty_f32_2_0 2,0 TYPE float32", expected: "Tensor empty_f32_2_0 created with type float32",
			setupFunc: func() { addExpectedMeta("empty_f32_2_0", []int{2, 0}, tensor.DataTypeFloat32, []int{0, 0}) }},
		{name: "Create Duplicate Tensor", query: "CREATE TENSOR my_tensor_f64 1,1", shouldError: true, errorContains: "already exists"},

		// --- INSERT INTO ---
		// KOREKSI PESAN SUKSES INSERT
		{name: "Insert into Float64 Tensor", query: "INSERT INTO my_tensor_f64 VALUES (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)", expected: "String data inserted into my_tensor_f64"},
		{name: "Insert into Float32 Tensor", query: "INSERT INTO my_tensor_f32 VALUES (10.1, 20.2, 30.3, 40.4)", expected: "String data inserted into my_tensor_f32"},
		{name: "Insert into Int32 Tensor", query: "INSERT INTO my_tensor_i32 VALUES (100, 200, 300)", expected: "String data inserted into my_tensor_i32"},
		{name: "Insert into Int64 Scalar Tensor", query: "INSERT INTO scalar_i64 VALUES (1234567890123)", expected: "String data inserted into scalar_i64"},
		{name: "Insert into Empty Tensor (0,2) (0 elements)", query: "INSERT INTO empty_f32_0_2 VALUES ()", expected: "Data inserted into empty_f32_0_2 (0 elements from string)"},

		// --- SELECT ---
		{name: "Select Full Float64 Tensor", query: "SELECT my_tensor_f64 FROM my_tensor_f64", expected: []interface{}{[]interface{}{1.0, 2.0, 3.0}, []interface{}{4.0, 5.0, 6.0}}},
		{name: "Select Full Float32 Tensor", query: "SELECT my_tensor_f32 FROM my_tensor_f32", expected: []interface{}{[]interface{}{float32(10.1), float32(20.2)}, []interface{}{float32(30.3), float32(40.4)}}},
		{name: "Select Full Int32 Tensor (1D)", query: "SELECT my_tensor_i32 FROM my_tensor_i32", expected: []interface{}{int32(100), int32(200), int32(300)}},
		{name: "Select Full Int64 Scalar Tensor", query: "SELECT scalar_i64 FROM scalar_i64", expected: int64(1234567890123)},
		{name: "Select Empty Tensor (shape [0,2])", query: "SELECT empty_f32_0_2 FROM empty_f32_0_2", expected: []interface{}{}},
		{name: "Select Empty Tensor (shape [2,0])", query: "SELECT empty_f32_2_0 FROM empty_f32_2_0", expected: []interface{}{[]interface{}{}, []interface{}{}}},

		// --- MATH OPERATIONS ---
		{name: "Create Math Tensor A (f32)", query: "CREATE TENSOR math_a_f32 2,2 TYPE float32", expected: "Tensor math_a_f32 created with type float32",
			setupFunc: func() { addExpectedMeta("math_a_f32", []int{2, 2}, tensor.DataTypeFloat32, []int{2, 1}) }},
		{name: "Insert Math Tensor A (f32)", query: "INSERT INTO math_a_f32 VALUES (1, 2, 3, 4)", expected: "String data inserted into math_a_f32"},
		{name: "Create Math Tensor B (f32)", query: "CREATE TENSOR math_b_f32 2,2 TYPE float32", expected: "Tensor math_b_f32 created with type float32",
			setupFunc: func() { addExpectedMeta("math_b_f32", []int{2, 2}, tensor.DataTypeFloat32, []int{2, 1}) }},
		{name: "Insert Math Tensor B (f32)", query: "INSERT INTO math_b_f32 VALUES (10, 20, 30, 40)", expected: "String data inserted into math_b_f32"},
		{
			name:      "Add Two Float32 Tensors",
			query:     "ADD TENSOR math_a_f32 WITH TENSOR math_b_f32 INTO math_add_f32",
			expected:  "Tensor 'math_add_f32' created successfully from operation ADD_TENSORS",
			setupFunc: func() { addExpectedMeta("math_add_f32", []int{2, 2}, tensor.DataTypeFloat32, []int{2, 1}) },
			verifyResult: func(t *testing.T, exec *tensor.Executor, p *tensor.Parser, resultName string, _ interface{}) {
				q, _ := p.Parse(fmt.Sprintf("SELECT %s FROM %s", resultName, resultName))
				res, err := exec.Execute(q)
				assertError(t, err, false)
				expectedData := []interface{}{
					[]interface{}{float32(11), float32(22)},
					[]interface{}{float32(33), float32(44)},
				}
				assertEqual(t, res, expectedData)
			},
		},
		{
			name:      "Add Scalar to Float32 Tensor",
			query:     "ADD SCALAR 1.5 TO TENSOR math_a_f32 INTO math_add_scalar_f32",
			expected:  "Tensor 'math_add_scalar_f32' created successfully from operation ADD_SCALAR",
			setupFunc: func() { addExpectedMeta("math_add_scalar_f32", []int{2, 2}, tensor.DataTypeFloat32, []int{2, 1}) },
			verifyResult: func(t *testing.T, exec *tensor.Executor, p *tensor.Parser, resultName string, _ interface{}) {
				q, _ := p.Parse(fmt.Sprintf("SELECT %s FROM %s", resultName, resultName))
				res, err := exec.Execute(q)
				assertError(t, err, false)
				expectedData := []interface{}{
					[]interface{}{float32(2.5), float32(3.5)},
					[]interface{}{float32(4.5), float32(5.5)},
				}
				assertEqual(t, res, expectedData)
			},
		},
		{
			name:          "Add Tensors to Existing Output",
			query:         "ADD TENSOR math_a_f32 WITH TENSOR math_b_f32 INTO math_add_f32",
			shouldError:   true,
			errorContains: "output tensor 'math_add_f32' already exists",
		},
		// --- LIST TENSORS ---
		// KOREKSI tc.expected untuk LIST TENSORS agar menggunakan fungsi
		{name: "List All Tensors (initial)", query: "LIST TENSORS",
			expected: func() interface{} { return expectedInitialTensors },
		},
		{name: "List Float32 Tensors", query: "LIST TENSORS WHERE DATATYPE = 'float32'",
			expected: func() interface{} {
				var filtered []tensor.TensorMetadata
				for _, meta := range expectedInitialTensors {
					if meta.DataType == tensor.DataTypeFloat32 {
						filtered = append(filtered, meta)
					}
				}
				return filtered
			},
		},
		{name: "List 2D Tensors", query: "LIST TENSORS WHERE NUM_DIMENSIONS = 2",
			expected: func() interface{} {
				var filtered []tensor.TensorMetadata
				for _, meta := range expectedInitialTensors {
					numDims := len(meta.Shape)
					if len(meta.Shape) == 0 {
						numDims = 0
					} // Skalar
					if numDims == 2 {
						filtered = append(filtered, meta)
					}
				}
				return filtered
			},
		},
		{name: "List Float32 2D Tensors", query: "LIST TENSORS WHERE DATATYPE = 'float32' AND NUM_DIMENSIONS = 2",
			expected: func() interface{} {
				var filtered []tensor.TensorMetadata
				for _, meta := range expectedInitialTensors {
					numDims := len(meta.Shape)
					if len(meta.Shape) == 0 {
						numDims = 0
					}
					if meta.DataType == tensor.DataTypeFloat32 && numDims == 2 {
						filtered = append(filtered, meta)
					}
				}
				return filtered
			},
		},
	}

	// Jalankan semua setupFunc untuk CREATE agar expectedInitialTensors terisi sebelum tes LIST
	for _, tc := range testCases {
		if (strings.HasPrefix(tc.name, "Create") || strings.HasPrefix(tc.name, "Add Two") || strings.HasPrefix(tc.name, "Add Scalar")) &&
			tc.setupFunc != nil && !tc.shouldError {
			tc.setupFunc()
		}
	}
	sort.SliceStable(expectedInitialTensors, func(i, j int) bool {
		return expectedInitialTensors[i].Name < expectedInitialTensors[j].Name
	})

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			parsedQuery, errParse := parser.Parse(tc.query)
			if tc.shouldError && errParse != nil {
				if tc.errorContains == "" || strings.Contains(errParse.Error(), tc.errorContains) {
					return
				}
				t.Fatalf("Gagal memparsing kueri '%s': error aktual '%v' tidak mengandung '%s'", tc.query, errParse, tc.errorContains)
			}
			assertError(t, errParse, false, "Parsing kueri '%s'", tc.query)
			if errParse != nil {
				return
			}

			result, errExec := executor.Execute(parsedQuery)

			if tc.shouldError {
				assertError(t, errExec, true, "Eksekusi kueri: %s", tc.query)
				if tc.errorContains != "" && errExec != nil {
					assertErrorContains(t, errExec, tc.errorContains, "Eksekusi kueri: %s", tc.query)
				}
			} else {
				assertError(t, errExec, false, "Eksekusi kueri: %s", tc.query)
				if errExec == nil {
					var currentExpectedValue interface{}
					if expFunc, isFunc := tc.expected.(func() interface{}); isFunc {
						currentExpectedValue = expFunc()
					} else if ptr, isPtr := tc.expected.(*[]tensor.TensorMetadata); isPtr {
						currentExpectedValue = *ptr
					} else {
						currentExpectedValue = tc.expected
					}

					if parsedQuery.Type == tensor.MathOperationQuery {
						assertEqual(t, result.(string), currentExpectedValue.(string), "Hasil operasi matematika untuk: %s", tc.query)
						if tc.verifyResult != nil {
							tc.verifyResult(t, executor, parser, parsedQuery.OutputTensorName, currentExpectedValue)
						}
					} else if parsedQuery.Type == tensor.CreateTensorQuery || parsedQuery.Type == tensor.InsertTensorQuery {
						assertEqual(t, result, currentExpectedValue, "Hasil untuk: %s", tc.query)
					} else if parsedQuery.Type == tensor.SelectTensorQuery {
						assertEqual(t, result, currentExpectedValue, "Hasil SELECT untuk: %s", tc.query)
					} else if parsedQuery.Type == tensor.GetDataTensorQuery {
						actualResult, ok := result.([]tensor.TensorDataResult)
						if ok {
							expectedResult, eOk := currentExpectedValue.([]tensor.TensorDataResult)
							if !eOk {
								_, eMultiOk := currentExpectedValue.([][]tensor.TensorDataResult)
								if eMultiOk {
									t.Fatalf("Tipe tc.expected ([][]TensorDataResult) tidak cocok untuk hasil single tensor GET DATA ([]TensorDataResult). Kueri: %s", tc.query)
								} else {
									t.Fatalf("Tipe tc.expected (%T) tidak cocok untuk hasil GET DATA single tensor ([]tensor.TensorDataResult). Kueri: %s", currentExpectedValue, tc.query)
								}
							}
							assertEqual(t, actualResult, expectedResult, "Kueri GET DATA (single): %s", tc.query)
						} else {
							actualResultMulti, okMulti := result.([][]tensor.TensorDataResult)
							if !okMulti {
								t.Fatalf("Tipe hasil tidak dikenali untuk GET DATA. Kueri: %s. Got type: %T", tc.query, result)
							}
							expectedResultMulti, eOkMulti := currentExpectedValue.([][]tensor.TensorDataResult)
							if !eOkMulti {
								t.Fatalf("Tipe tc.expected tidak cocok untuk GET DATA multi tensor. Kueri: %s. Expected type: [][]tensor.TensorDataResult, got %T", tc.query, currentExpectedValue)
							}
							assertEqual(t, actualResultMulti, expectedResultMulti, "Kueri GET DATA (multi): %s", tc.query)
						}
					} else if parsedQuery.Type == tensor.ListTensorsQuery {
						actualMetadataList, ok := result.([]tensor.TensorMetadata)
						if !ok {
							t.Fatalf("Tipe hasil tidak dikenali untuk LIST TENSORS. Kueri: %s. Got type: %T", tc.query, result)
						}
						expectedMetadataList, eOk := currentExpectedValue.([]tensor.TensorMetadata)
						if !eOk {
							t.Fatalf("Tipe tc.expected tidak cocok untuk LIST TENSORS. Kueri: %s. Expected type: []tensor.TensorMetadata, got %T", tc.query, currentExpectedValue)
						}

						sort.SliceStable(actualMetadataList, func(i, j int) bool {
							return actualMetadataList[i].Name < actualMetadataList[j].Name
						})
						// expectedMetadataList sudah diurutkan saat dibuat oleh fungsi atau setelah populasi expectedInitialTensors
						assertEqual(t, actualMetadataList, expectedMetadataList, "Hasil LIST TENSORS untuk: %s", tc.query)
					} else {
						assertEqual(t, result, currentExpectedValue, "Hasil untuk tipe kueri tidak dikenal: %s", tc.query)
					}
				}
			}
		})
	}
}

func TestParserSpecificCases(t *testing.T) {
	parser := &tensor.Parser{}
	t.Run("Create Tensor with Shape with Spaces", func(t *testing.T) {
		queryStr := "CREATE TENSOR spaced_shape 2, 3,4 TYPE int32"
		query, err := parser.Parse(queryStr)
		assertError(t, err, false, "Parsing: %s", queryStr)
		if err == nil {
			assertEqual(t, query.Shape, []int{2, 3, 4}, "Shape for: %s", queryStr)
			assertEqual(t, query.DataType, tensor.DataTypeInt32, "DataType for: %s", queryStr)
		}
	})
	t.Run("Insert with Spaces in Values", func(t *testing.T) {
		queryStr := "INSERT INTO mytensor VALUES ( 1.0 ,  2.5  ,3.0)"
		query, err := parser.Parse(queryStr)
		assertError(t, err, false, "Parsing: %s", queryStr)
		if err == nil {
			assertEqual(t, query.Data, []string{"1.0", "2.5", "3.0"}, "Data for: %s", queryStr)
		}
	})
	t.Run("Get Data with Multiple Tensors and Slices", func(t *testing.T) {
		queryStr := "GET DATA FROM tensorA [0:1], tensorB [1:2, 2:3] BATCH 5"
		query, err := parser.Parse(queryStr)
		assertError(t, err, false, "Parsing: %s", queryStr)
		if err == nil {
			assertEqual(t, len(query.TensorNames), 2, "Num TensorNames for: %s", queryStr)
			assertEqual(t, query.TensorNames[0], "tensorA", "TensorName[0] for: %s", queryStr)
			assertEqual(t, query.Slices[0], [][2]int{{0, 1}}, "Slices[0] for: %s", queryStr)
			assertEqual(t, query.TensorNames[1], "tensorB", "TensorName[1] for: %s", queryStr)
			assertEqual(t, query.Slices[1], [][2]int{{1, 2}, {2, 3}}, "Slices[1] for: %s", queryStr)
			assertEqual(t, query.BatchSize, 5, "BatchSize for: %s", queryStr)
		}
	})
	t.Run("Select with Slice with Spaces", func(t *testing.T) {
		queryStr := "SELECT t1 FROM t1 [ 0 : 1, 1 : 2 ]"
		query, err := parser.Parse(queryStr)
		assertError(t, err, false, "Parsing: %s", queryStr)
		if err == nil {
			assertEqual(t, query.Slices[0], [][2]int{{0, 1}, {1, 2}}, "Slices for: %s", queryStr)
		}
	})
	t.Run("Create Tensor Scalar Shape", func(t *testing.T) {
		queryStr := "CREATE TENSOR scalar_test_shape 1 TYPE float32"
		query, err := parser.Parse(queryStr)
		assertError(t, err, false, "Parsing: %s", queryStr)
		if err == nil {
			assertEqual(t, query.Shape, []int{1}, "Shape for: %s", queryStr)
		}
	})
	t.Run("Create Tensor with Zero Dimension", func(t *testing.T) {
		queryStr := "CREATE TENSOR zero_dim_tensor_parse 0,5,2 TYPE int64"
		query, err := parser.Parse(queryStr)
		assertError(t, err, false, "Parsing: %s", queryStr)
		if err == nil {
			assertEqual(t, query.Shape, []int{0, 5, 2}, "Shape for: %s", queryStr)
			assertEqual(t, query.DataType, tensor.DataTypeInt64, "DataType for: %s", queryStr)
		}
	})
}
