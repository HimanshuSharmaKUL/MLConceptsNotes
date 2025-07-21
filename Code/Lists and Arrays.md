---
sticker: emoji//1f375
---
  [[DSA]] , 


#### 1. Initializing a 2D array with zeros
Use list comprehension to safely initialize a matrix of zeros.
```python
rows, cols = 3, 4
matrix = [[0 for _ in range(cols)] for _ in range(rows)]
```
#### 2. Flattening a 2D list
Convert a 2D list into a 1D list using list comprehension.

```python
flat = [elem for row in matrix for elem in row]
```
#### 3. Transposing a 2D list
Transpose rows and columns using `zip(*list)`.
```python
transposed = list(zip(*matrix))
```
#### 4. Copying a 2D list safely
Avoid reference issues by copying each row individually.
```python
copy = [row[:] for row in matrix]
```
#### 5. Matrix multiplication
Manual implementation of matrix multiplication using nested loops.
```python
def matmul(A, B):
    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*B)] for row in A]
```
#### 6. Finding the max in a 2D list
Use nested `max()` to find the largest value.

```python
max_value = max(max(row) for row in matrix)
```
#### 7. Slicing rows and columns
Use list slicing to extract parts of a 2D array.

```python
sub = matrix[1:3]          # rows 1 and 2
cols = [row[1:3] for row in matrix]
```
#### 8. Looping with index using `enumerate`
Get both index and value while iterating.

```python
for i, row in enumerate(matrix):
    for j, val in enumerate(row):
        print(i, j, val)
```
#### 9. Creating identity matrix
Quick way to make an identity matrix using list comprehension.
```python
n = 4
identity = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
```
#### 10. Row/column sum
Sum all elements in a row or column.

```python
row_sum = sum(matrix[0])
col_sum = sum(row[1] for row in matrix)
```
#### 11. Any/all for 2D checks

Check for conditions like all zeros in a row.
```python
all_zeros = any(all(x == 0 for x in row) for row in matrix)
```
#### 12. Sorting rows or columns
Sort entire matrix rows or extract/sort specific columns.
```python
matrix.sort()  # Sorts by rows
sorted_by_col = sorted(matrix, key=lambda x: x[1])
```
#### 13. Constructing 2D from input
Standard practice to read 2D list from input in HackerRank.
```python
matrix = [list(map(int, input().split())) for _ in range(6)]
```
#### 14. Diagonal elements
Extract main and anti-diagonal elements.

```python
diag = [matrix[i][i] for i in range(len(matrix))]
anti_diag = [matrix[i][~i] for i in range(len(matrix))]
```
#### 15. Padding a matrix
Add a border of zeros around a 2D list.

```python
padded = [[0] * (len(matrix[0]) + 2)]
for row in matrix:
    padded.append([0] + row + [0])
padded.append([0] * (len(matrix[0]) + 2))
```
#### 16. Reversing rows or columns
Reverse the rows or elements within rows.
```python
matrix.reverse()  # Reverse row order
rev_rows = [row[::-1] for row in matrix]
```
#### 17. Zipping two 2D lists
Zip corresponding elements of two 2D lists.
```python
combined = [[(a, b) for a, b in zip(r1, r2)] for r1, r2 in zip(matrix1, matrix2)]
```
#### 18. Filtering values
Extract elements by condition using nested comprehension.
```python
evens = [x for row in matrix for x in row if x % 2 == 0]
```
#### 19. Finding coordinates of max value
Locate the position of the maximum value in the matrix.

```python
i, j = max(((i, j) for i, row in enumerate(matrix) for j, val in enumerate(row)), key=lambda x: matrix[x[0]][x[1]])
```
#### 20. Generating a checkerboard pattern
Alternate values based on row+col index parity.
```python
n = 4
checkerboard = [[(i + j) % 2 for j in range(n)] for i in range(n)]
```

---
