package main

import (
	"fmt"
	"os" // Ditambahkan untuk membersihkan direktori data
	"tensordb/pkg/tensor"
)

func main() {
	// Membersihkan direktori data jika ada untuk pengujian yang bersih
	dataDir := "data"
	os.RemoveAll(dataDir) // Abaikan kesalahan jika direktori tidak ada

	storage, err := tensor.NewStorage(dataDir)
	if err != nil {
		fmt.Println("Error initializing storage:", err)
		return
	}

	executor := tensor.NewExecutor(storage)
	parser := &tensor.Parser{}

	queries := []string{
		"CREATE TENSOR my_tensor 2,3",
		"INSERT INTO my_tensor VALUES (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)",
		"SELECT my_tensor FROM my_tensor",
		"SELECT my_tensor FROM my_tensor [0:1, 1:2]",
		"SELECT my_tensor FROM my_tensor [1:2, 0:2]",
		"CREATE TENSOR tensor_3d 2,2,2",
		"INSERT INTO tensor_3d VALUES (1,2,3,4,5,6,7,8)",
		"SELECT tensor_3d FROM tensor_3d",
		"SELECT tensor_3d FROM tensor_3d [0:1, 0:2, 1:2]",
		"GET DATA FROM my_tensor",
		"GET DATA FROM my_tensor [0:1, 1:2]",      // Query 11
		"GET DATA FROM tensor_3d [0:1, 0:2, 1:2]", // Query 12
		"GET DATA FROM my_tensor, tensor_3d",      // Query 13
		// Query 14 diperbaiki: Setiap tensor secara eksplisit dikaitkan dengan potongannya
		"GET DATA FROM my_tensor [0:1, 1:2], tensor_3d [0:1, 0:2, 1:2]",
		"GET DATA FROM my_tensor BATCH 2",
		"GET DATA FROM my_tensor, tensor_3d BATCH 3",
	}

	fmt.Println("TensorDB - Running test queries:")
	fmt.Println("----------------------------------")

	for i, queryStr := range queries {
		fmt.Printf("Query %d: %s\n", i+1, queryStr) // Nomor kueri dimulai dari 1
		query, err := parser.Parse(queryStr)
		if err != nil {
			fmt.Println("Error parsing query:", err)
			fmt.Println("----------------------------------") // Tambahkan pemisah untuk keterbacaan
			continue
		}
		result, err := executor.Execute(query)
		if err != nil {
			fmt.Println("Error executing query:", err)
			fmt.Println("----------------------------------") // Tambahkan pemisah untuk keterbacaan
			continue
		}
		fmt.Println("Result:", result)
		fmt.Println("----------------------------------")
	}

	fmt.Println("All queries executed.")

	// Membersihkan direktori data setelah pengujian selesai
	// os.RemoveAll(dataDir) // Anda bisa mengaktifkan ini jika ingin membersihkan setelah setiap run
}
