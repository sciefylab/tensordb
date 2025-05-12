package tensor

import (
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// TIDAK ADA LAGI DEKLARASI QueryType atau konstanta QueryType DI SINI.
// Kita akan menggunakan yang dari tensor.go

// Parser adalah struct untuk memparsing kueri.
type Parser struct{}

// Parse memparsing string kueri menjadi struct Query.
func (p *Parser) Parse(query string) (*Query, error) {
	queryOriginalCase := strings.TrimSpace(query)
	queryLower := strings.ToLower(queryOriginalCase)

	// Regex untuk operasi matematika (contoh untuk ADD)
	addTensorRegex := regexp.MustCompile(`(?i)^ADD\s+TENSOR\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+WITH\s+TENSOR\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+INTO\s+([a-zA-Z_][a-zA-Z0-9_]*)$`)
	addScalarRegex := regexp.MustCompile(`(?i)^ADD\s+SCALAR\s+([0-9\.eE+-]+)\s+TO\s+TENSOR\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+INTO\s+([a-zA-Z_][a-zA-Z0-9_]*)$`)

	matchesAddTensor := addTensorRegex.FindStringSubmatch(queryOriginalCase)
	if matchesAddTensor != nil {
		return &Query{
			Type:             MathOperationQuery, // Menggunakan konstanta dari tensor.go
			MathOperator:     "ADD_TENSORS",
			InputTensorNames: []string{matchesAddTensor[1], matchesAddTensor[2]},
			OutputTensorName: matchesAddTensor[3],
		}, nil
	}

	matchesAddScalar := addScalarRegex.FindStringSubmatch(queryOriginalCase)
	if matchesAddScalar != nil {
		return &Query{
			Type:             MathOperationQuery, // Menggunakan konstanta dari tensor.go
			MathOperator:     "ADD_SCALAR",
			InputTensorNames: []string{matchesAddScalar[2]},
			ScalarOperand:    matchesAddScalar[1],
			OutputTensorName: matchesAddScalar[3],
		}, nil
	}

	partsOriginal := strings.Fields(queryOriginalCase)
	partsLower := strings.Fields(queryLower)

	if len(partsLower) < 1 {
		return nil, errors.New("query too short or malformed")
	}

	// Parsing untuk LIST TENSORS
	if strings.HasPrefix(queryLower, "list tensors") {
		q := &Query{
			Type:                ListTensorsQuery, // Menggunakan konstanta dari tensor.go
			FilterDataType:      "",
			FilterNumDimensions: -1,
		}

		whereClause := ""
		if idx := strings.Index(queryLower, " where "); idx != -1 {
			whereClause = strings.TrimSpace(queryOriginalCase[idx+len(" where "):])
		}

		if whereClause != "" {
			reDataType := regexp.MustCompile(`(?i)DATATYPE\s*=\s*'([^']*)'`)
			reNumDimensions := regexp.MustCompile(`(?i)NUM_DIMENSIONS\s*=\s*(\d+)`)

			dataTypeMatches := reDataType.FindStringSubmatch(whereClause)
			if len(dataTypeMatches) == 2 {
				dt := strings.ToLower(dataTypeMatches[1])
				if _, err := GetElementSize(dt); err == nil {
					q.FilterDataType = dt
				} else {
					return nil, fmt.Errorf("invalid data type in WHERE clause: '%s'", dataTypeMatches[1])
				}
			}

			numDimMatches := reNumDimensions.FindStringSubmatch(whereClause)
			if len(numDimMatches) == 2 {
				numDim, err := strconv.Atoi(numDimMatches[1])
				if err != nil {
					return nil, fmt.Errorf("invalid number for NUM_DIMENSIONS: '%s'", numDimMatches[1])
				}
				if numDim < 0 {
					return nil, fmt.Errorf("NUM_DIMENSIONS cannot be negative: %d", numDim)
				}
				q.FilterNumDimensions = numDim
			}
		}
		return q, nil
	}

	switch partsLower[0] {
	case "create":
		if len(partsLower) < 3 || partsLower[1] != "tensor" {
			return nil, errors.New("invalid CREATE TENSOR syntax: expected 'CREATE TENSOR name shape [TYPE datatype]' or 'CREATE TENSOR name TYPE datatype'")
		}
		tensorName := partsOriginal[2]
		var shape []int

		remainingPartsOriginal := []string{}
		if len(partsOriginal) > 3 {
			remainingPartsOriginal = partsOriginal[3:]
		}
		remainingStrOriginal := strings.Join(remainingPartsOriginal, " ")

		shapeStr := ""
		dataType := DataTypeFloat64 // Default dari tensor.go

		createRegex := regexp.MustCompile(`(?i)^\s*(.*?)\s*(?:type\s+([a-zA-Z0-9_]+))?\s*$`)
		matches := createRegex.FindStringSubmatch(remainingStrOriginal)

		if matches == nil {
			if remainingStrOriginal != "" && !strings.Contains(strings.ToLower(remainingStrOriginal), "type") {
				shapeStr = strings.TrimSpace(remainingStrOriginal)
			}
		} else {
			shapeStr = strings.TrimSpace(matches[1])
		}

		if shapeStr == "" {
			shape = []int{}
		} else {
			shapeStrNoSpaces := strings.ReplaceAll(shapeStr, " ", "")
			shapeDimsStr := strings.Split(shapeStrNoSpaces, ",")
			if len(shapeDimsStr) == 0 || (len(shapeDimsStr) == 1 && strings.TrimSpace(shapeDimsStr[0]) == "") {
				shape = []int{}
			} else {
				shape = make([]int, len(shapeDimsStr))
				for i, dStr := range shapeDimsStr {
					trimmedDStr := strings.TrimSpace(dStr)
					if trimmedDStr == "" {
						return nil, fmt.Errorf("invalid dimension: empty string found in shape '%s'", shapeStr)
					}
					dim, err := strconv.Atoi(trimmedDStr)
					if err != nil {
						return nil, fmt.Errorf("invalid dimension '%s' in shape '%s': %w", trimmedDStr, shapeStr, err)
					}
					if dim < 0 {
						return nil, fmt.Errorf("invalid dimension '%s' in shape '%s': must be non-negative", trimmedDStr, shapeStr)
					}
					shape[i] = dim
				}
			}
		}

		if matches != nil && matches[2] != "" {
			dt := strings.ToLower(strings.TrimSpace(matches[2]))
			if _, err := GetElementSize(dt); err != nil { // GetElementSize dari tensor.go
				return nil, fmt.Errorf("invalid data type '%s' in CREATE TENSOR: %w", dt, err)
			}
			dataType = dt
		} else if matches == nil && strings.Contains(strings.ToLower(remainingStrOriginal), "type") {
			typeIdx := strings.Index(strings.ToLower(remainingStrOriginal), "type")
			if typeIdx != -1 {
				potentialTypeStr := strings.ToLower(strings.TrimSpace(remainingStrOriginal[typeIdx+len("type"):]))
				if potentialTypeStr != "" {
					if _, err := GetElementSize(potentialTypeStr); err == nil {
						dataType = potentialTypeStr
						shape = []int{}
					} else {
						return nil, fmt.Errorf("invalid data type '%s' found after TYPE keyword", potentialTypeStr)
					}
				} else {
					return nil, errors.New("missing data type after TYPE keyword")
				}
			}
		}

		return &Query{
			Type:        CreateTensorQuery, // Menggunakan konstanta dari tensor.go
			TensorNames: []string{tensorName},
			Shape:       shape,
			DataType:    dataType,
		}, nil

	case "insert":
		if len(partsLower) < 5 || partsLower[1] != "into" || partsLower[3] != "values" {
			return nil, errors.New("invalid INSERT INTO syntax: expected 'INSERT INTO name VALUES (...)'")
		}
		tensorName := partsOriginal[2]

		tempQueryLower := strings.ToLower(queryOriginalCase)
		valuesMatchIndex := strings.Index(tempQueryLower, "values")
		if valuesMatchIndex == -1 {
			valuesMatchIndex = strings.Index(queryOriginalCase, "VALUES")
			if valuesMatchIndex == -1 {
				return nil, errors.New("invalid INSERT INTO syntax: 'VALUES' keyword not found")
			}
		}

		openParenIndex := strings.Index(queryOriginalCase[valuesMatchIndex:], "(")
		if openParenIndex == -1 {
			return nil, errors.New("invalid INSERT INTO syntax: '(' not found after 'VALUES'")
		}
		openParenIndex += valuesMatchIndex

		closeParenIndex := strings.LastIndex(queryOriginalCase, ")")
		if closeParenIndex == -1 || closeParenIndex < openParenIndex {
			return nil, errors.New("invalid INSERT INTO syntax: ')' not found or misplaced for 'VALUES'")
		}

		valuesContent := strings.TrimSpace(queryOriginalCase[openParenIndex+1 : closeParenIndex])

		var dataToInsert []string
		if valuesContent == "" {
			dataToInsert = []string{}
		} else {
			dataStrValues := strings.Split(valuesContent, ",")
			dataToInsert = make([]string, len(dataStrValues))
			for i, dStr := range dataStrValues {
				dataToInsert[i] = strings.TrimSpace(dStr)
			}
		}

		return &Query{
			Type:        InsertTensorQuery, // Menggunakan konstanta dari tensor.go
			TensorNames: []string{tensorName},
			Data:        dataToInsert,
		}, nil

	case "select":
		if len(partsLower) < 4 || partsLower[2] != "from" {
			return nil, errors.New("invalid SELECT syntax: expected 'SELECT display_name FROM source_name [slice]'")
		}
		sourceTensorName := partsOriginal[3]
		sliceStr := ""
		if len(partsOriginal) > 4 {
			potentialSlicePart := strings.TrimSpace(strings.Join(partsOriginal[4:], " "))
			if strings.HasPrefix(potentialSlicePart, "[") && strings.HasSuffix(potentialSlicePart, "]") {
				sliceStr = potentialSlicePart
			} else {
				return nil, fmt.Errorf("unexpected tokens after tensor name in SELECT: '%s'", potentialSlicePart)
			}
		}

		var parsedSlices [][2]int
		if sliceStr != "" {
			sliceContent := strings.TrimPrefix(sliceStr, "[")
			sliceContent = strings.TrimSuffix(sliceContent, "]")
			if sliceContent == "" {
				parsedSlices = nil
			} else {
				sliceParts := strings.Split(sliceContent, ",")
				parsedSlices = make([][2]int, len(sliceParts))
				for i, s := range sliceParts {
					s = strings.TrimSpace(s)
					bounds := strings.Split(s, ":")
					if len(bounds) != 2 {
						return nil, fmt.Errorf("invalid slice format '%s' for SELECT", s)
					}
					startStr, endStr := strings.TrimSpace(bounds[0]), strings.TrimSpace(bounds[1])
					start, err := strconv.Atoi(startStr)
					if err != nil {
						return nil, fmt.Errorf("invalid slice start '%s': %w", startStr, err)
					}
					end, err := strconv.Atoi(endStr)
					if err != nil {
						return nil, fmt.Errorf("invalid slice end '%s': %w", endStr, err)
					}
					if start < 0 || end < start {
						return nil, fmt.Errorf("invalid slice range [%d:%d]", start, end)
					}
					parsedSlices[i] = [2]int{start, end}
				}
			}
		}
		return &Query{
			Type:        SelectTensorQuery, // Menggunakan konstanta dari tensor.go
			TensorNames: []string{sourceTensorName},
			Slices:      [][][2]int{parsedSlices},
		}, nil

	case "get":
		if len(partsLower) < 4 || partsLower[1] != "data" || partsLower[2] != "from" {
			return nil, errors.New("invalid GET DATA syntax: expected 'GET DATA FROM ...'")
		}
		fromKeywordIndexOriginal := -1
		for i, p := range partsOriginal {
			if strings.ToLower(p) == "from" {
				fromKeywordIndexOriginal = i
				break
			}
		}
		if fromKeywordIndexOriginal == -1 || fromKeywordIndexOriginal+1 >= len(partsOriginal) {
			return nil, errors.New("invalid GET DATA syntax: 'FROM' keyword missing or no tensor names provided")
		}
		afterFromOriginal := strings.Join(partsOriginal[fromKeywordIndexOriginal+1:], " ")
		tensorDefinitionsPart := afterFromOriginal
		batchSize := 0
		reBatch := regexp.MustCompile(`(?i)^(.*?)(?:\s+batch\s+(\d+)\s*)?$`)
		batchMatches := reBatch.FindStringSubmatch(strings.TrimSpace(afterFromOriginal))
		if batchMatches != nil {
			tensorDefinitionsPart = strings.TrimSpace(batchMatches[1])
			if len(batchMatches) > 2 && batchMatches[2] != "" {
				batchSizeStr := batchMatches[2]
				var errAtoi error
				batchSize, errAtoi = strconv.Atoi(batchSizeStr)
				if errAtoi != nil || batchSize <= 0 {
					return nil, fmt.Errorf("invalid batch size '%s': must be a positive integer: %w", batchSizeStr, errAtoi)
				}
			}
		}
		if strings.TrimSpace(tensorDefinitionsPart) == "" {
			if batchSize > 0 {
				return nil, errors.New("no tensor definitions found before 'BATCH' keyword in GET DATA")
			}
			return nil, errors.New("no tensor definitions found in GET DATA FROM clause")
		}
		tensorDefPattern := `([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*(\[[^\]]*\]))?`
		tensorDefRegex := regexp.MustCompile(tensorDefPattern)
		allMatches := tensorDefRegex.FindAllStringSubmatch(tensorDefinitionsPart, -1)
		if len(allMatches) == 0 && strings.TrimSpace(tensorDefinitionsPart) != "" {
			return nil, errors.New("no valid tensor definitions found in GET DATA FROM clause (after batch processing)")
		}
		if len(allMatches) == 0 && strings.TrimSpace(tensorDefinitionsPart) == "" {
			return nil, errors.New("no tensor definitions provided in GET DATA FROM clause")
		}
		tensorNames := make([]string, 0, len(allMatches))
		slices := make([][][2]int, 0, len(allMatches))
		for _, match := range allMatches {
			tensorName := strings.TrimSpace(match[1])
			tensorNames = append(tensorNames, tensorName)
			var currentTensorSlice [][2]int
			if len(match) > 2 && match[2] != "" {
				sliceContentWithBrackets := strings.TrimSpace(match[2])
				sliceContent := strings.TrimPrefix(sliceContentWithBrackets, "[")
				sliceContent = strings.TrimSuffix(sliceContent, "]")
				if sliceContent == "" {
					currentTensorSlice = nil
				} else {
					slicePartsStr := strings.Split(sliceContent, ",")
					currentTensorSlice = make([][2]int, len(slicePartsStr))
					for j, sPart := range slicePartsStr {
						sPart = strings.TrimSpace(sPart)
						bounds := strings.Split(sPart, ":")
						if len(bounds) != 2 {
							return nil, fmt.Errorf("invalid slice format '%s' for tensor '%s'", sPart, tensorName)
						}
						startStr, endStr := strings.TrimSpace(bounds[0]), strings.TrimSpace(bounds[1])
						start, err := strconv.Atoi(startStr)
						if err != nil {
							return nil, fmt.Errorf("invalid slice start '%s' for tensor '%s': %w", startStr, tensorName, err)
						}
						end, err := strconv.Atoi(endStr)
						if err != nil {
							return nil, fmt.Errorf("invalid slice end '%s' for tensor '%s': %w", endStr, tensorName, err)
						}
						if start < 0 || end < start {
							return nil, fmt.Errorf("invalid slice range [%d:%d] for tensor '%s'", start, end, tensorName)
						}
						currentTensorSlice[j] = [2]int{start, end}
					}
				}
			}
			slices = append(slices, currentTensorSlice)
		}
		if len(tensorNames) == 0 {
			return nil, errors.New("no valid tensor names found for GET DATA (after regex match)")
		}
		return &Query{
			Type:        GetDataTensorQuery, // Menggunakan konstanta dari tensor.go
			TensorNames: tensorNames,
			Slices:      slices,
			BatchSize:   batchSize,
		}, nil

	}
	return nil, fmt.Errorf("unsupported query type or malformed query near: '%s'", partsLower[0])
}
