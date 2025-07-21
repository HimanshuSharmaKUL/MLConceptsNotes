---
sticker: emoji//1f60f
---
16-07-2025
18:11

Status:
Tags: [[DSA]] 

# Logical Operations

## XOR
|Bit A|Bit B|A ⊕ B (XOR)|
|---|---|---|
|0|0|0|
|0|1|1|
|1|0|1|
|1|1|0|
XOR returns `1` if the bits are **different**, and `0` if they are **the same**.
#### Properties XOR
1. A ⊕ A = 0
2. A ⊕ 0 = A
3. **XOR is commutative and associative**:  
`(A ⊕ B) ⊕ C = A ⊕ (B ⊕ C)`
#### How to Perform Bitwise XOR in Python
You can use the caret symbol `^` for XOR in Python.
##### Example:

```python
a = 5       # binary: 0101 
b = 3       # binary: 0011  
result = a ^ b  # binary: 0110 = 6 
print(result)   # Output: 6
```
#### Explanation:
  `0101  (5) ⊕ 0011  (3) = 0110  (6)`
### Application
1. Swapping Values without a temporary variable:
```python
a = 5                  #0101
b = 3                  #0011
a = a ^ b              #0110
b = a ^ b              #0101 was a
a = a ^ b              #0011 was b
```
2. Checksum or parity checks
3. Finding unique numbers in an array where others appear in pairs.
4. Find the Single Number
	- LeetCode #136: Single Number https://leetcode.com/problems/single-number/
	- Problem: Every element appears twice except one. Find that one.
	- Approach XOR all elements → duplicates cancel out.
```python
def singleNumber(nums):
result = 0
for num in nums:
	result ^= num
return result
```
5. Find the Two Single Numbers
	- **LeetCode #260**: Single Number III
	- Every element appears twice except two. Return them.
	- XOR trick is used to isolate the two differing bits.
6. Missing Number in Array
	- **LeetCode #268**: Missing Number
	- From 0 to n, one number is missing.
	- XOR all indices and values to find the missing one.
```python
def missingNumber(nums):
    n = len(nums)
    xor_total = 0
    for i in range(n):
        xor_total ^= i ^ nums[i]
    return xor_total ^ n
```

---
## Modulo Operator %
The **modulo operator** (`%`) in Python returns the **remainder** of a division operation.
`remainder = a % b`
```python
7 % 3   # 1   (since 3 * 2 = 6, and 7 - 6 = 1)
10 % 2  # 0   (10 is divisible by 2)
-7 % 3  # 2   (Python's `%` always returns a non-negative remainder if the divisor is positive)
```
Python uses floor division behaviousr, .'. 
```python
-7 // 3 = -3     → quotient
-7 % 3  = 2      → remainder (satisfies: a == (a // b) * b + (a % b))
```
### Applications
1. **Check Even or Odd**:
```python
if n % 2 == 0:
    print("Even")
else:
    print("Odd")
```
2. **Cycle Through Elements** (e.g., circular arrays):
```python
index = (current_index + 1) % len(array)
```
3. **Clock Problems** (wrap around 12 or 24 hours):
```python
hour = (hour + offset) % 12
```

