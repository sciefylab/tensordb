package tests // Perubahan package

import (
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/sciefylab/tensordb/pkg/client" // Impor package yang diuji jika helper membutuhkannya
	"github.com/sciefylab/tensordb/pkg/tensor" // Impor package yang diuji jika helper membutuhkannya
)

// setupTestClient menginisialisasi storage, executor, dan client untuk pengujian.
// Ini spesifik untuk pengujian client, jadi mungkin tetap di client_test.go atau di sini jika umum.
func setupTestClient(t *testing.T) (string, *client.Client, func()) {
	t.Helper()
	dataDir, err := os.MkdirTemp("", "tensordb_test_common_")
	if err != nil {
		t.Fatalf("Gagal membuat direktori data sementara: %v", err)
	}

	storage, errStorage := tensor.NewStorage(dataDir)
	if errStorage != nil {
		os.RemoveAll(dataDir)
		t.Fatalf("Gagal membuat storage: %v", errStorage)
	}

	executor := tensor.NewExecutor(storage)
	apiClient := client.NewClient(executor) // Menggunakan client dari tensordb/client

	cleanup := func() {
		if errClose := apiClient.Close(); errClose != nil {
			t.Logf("Peringatan: Error saat menutup client (executor): %v", errClose)
		}
		if errRemove := os.RemoveAll(dataDir); errRemove != nil {
			t.Errorf("Gagal menghapus direktori data sementara %s: %v", dataDir, errRemove)
		}
	}
	return dataDir, apiClient, cleanup
}

// setupTest (untuk TestTensorDBOperations)
func setupTest(t *testing.T) (string, *tensor.Executor, func()) {
	t.Helper()
	dataDir, err := os.MkdirTemp("", "tensordb_testdir_") // Nama dir berbeda untuk menghindari konflik jika dijalankan bersamaan
	if err != nil {
		t.Fatalf("Gagal membuat direktori data sementara: %v", err)
	}
	storage, errStorage := tensor.NewStorage(dataDir)
	if errStorage != nil {
		os.RemoveAll(dataDir)
		t.Fatalf("Gagal membuat storage: %v", errStorage)
	}
	executor := tensor.NewExecutor(storage) // Menggunakan tensor.Executor dari tensordb/tensor
	cleanup := func() {
		if errClose := executor.Close(); errClose != nil {
			t.Logf("Peringatan: Error saat menutup executor: %v", errClose)
		}
		if errRemove := os.RemoveAll(dataDir); errRemove != nil {
			t.Errorf("Gagal menghapus direktori data sementara %s: %v", dataDir, errRemove)
		}
	}
	return dataDir, executor, cleanup
}

// assertEqual memeriksa apakah dua nilai sama, gagal jika tidak.
func assertEqual(t *testing.T, actual, expected interface{}, msgAndArgs ...interface{}) {
	t.Helper()
	if !reflect.DeepEqual(actual, expected) {
		message := fmt.Sprintf("Assertion Failed: Hasil aktual tidak sama dengan yang diharapkan.\nAktual: %#v\nHarapan: %#v", actual, expected)
		if len(msgAndArgs) > 0 {
			customMsg := fmt.Sprintf(msgAndArgs[0].(string), msgAndArgs[1:]...)
			message = customMsg + "\n" + message
		}
		t.Errorf("%s", message)
	}
}

// assertError memeriksa apakah error terjadi (jika shouldError true) atau tidak (jika shouldError false).
func assertError(t *testing.T, err error, shouldError bool, msgAndArgs ...interface{}) {
	t.Helper()
	prefix := ""
	if len(msgAndArgs) > 0 {
		prefix = fmt.Sprintf(msgAndArgs[0].(string), msgAndArgs[1:]...) + ": "
	}
	if shouldError && err == nil {
		t.Errorf("%sError diharapkan terjadi, tetapi tidak ada error.", prefix)
	} else if !shouldError && err != nil {
		t.Errorf("%sError tidak diharapkan, tetapi terjadi: %v", prefix, err)
	}
}

// assertErrorContains memeriksa apakah error terjadi dan mengandung substring tertentu.
func assertErrorContains(t *testing.T, err error, expectedSubstring string, msgAndArgs ...interface{}) {
	t.Helper()
	prefix := ""
	if len(msgAndArgs) > 0 {
		prefix = fmt.Sprintf(msgAndArgs[0].(string), msgAndArgs[1:]...) + ": "
	}
	if err == nil {
		t.Errorf("%sError diharapkan terjadi (mengandung '%s'), tetapi tidak ada error.", prefix, expectedSubstring)
		return
	}
	if !strings.Contains(err.Error(), expectedSubstring) {
		t.Errorf("%sError diharapkan mengandung '%s', tetapi aktualnya: '%v'", prefix, expectedSubstring, err.Error())
	}
}

// assertTrue adalah helper untuk pengujian boolean.
func assertTrue(t *testing.T, condition bool, msgAndArgs ...interface{}) {
	t.Helper()
	if !condition {
		message := "Assertion Failed: Kondisi tidak true."
		if len(msgAndArgs) > 0 {
			customMsg := fmt.Sprintf(msgAndArgs[0].(string), msgAndArgs[1:]...)
			message = customMsg + "\n" + message
		}
		t.Errorf("%s", message)
	}
}

func setupBenchmarkClient(b *testing.B) (*client.Client, func()) {
	b.Helper()
	dataDir, err := os.MkdirTemp("", "tensordb_bench_")
	if err != nil {
		b.Fatalf("Gagal membuat direktori data sementara: %v", err)
	}

	storage, errStorage := tensor.NewStorage(dataDir)
	if errStorage != nil {
		os.RemoveAll(dataDir)
		b.Fatalf("Gagal membuat storage: %v", errStorage)
	}

	executor := tensor.NewExecutor(storage)
	apiClient := client.NewClient(executor)

	cleanup := func() {
		if apiClient != nil {
			apiClient.Close()
		}
		os.RemoveAll(dataDir)
	}
	return apiClient, cleanup
}
