# Tensor DB

## Build for inferensi model

## Test

```bash
go test ./tests/... -v

go test ./tests -bench=.

go test ./tests -bench=BenchmarkInsertData -cpuprofile=insert_cpu.prof

go test ./tests -bench=BenchmarkInsertData -memprofile=insert_mem.prof

go test ./tests -bench=BenchmarkGetData -cpuprofile=get_cpu.prof

go test ./tests -bench=BenchmarkGetData -memprofile=get_mem.prof

go tool pprof -http=:8081 insert_cpu.prof

go tool pprof insert_cpu.prof

```
