---
sticker: emoji//1f60f
---

[[DSA]]
https://blog.algomaster.io/p/20-patterns-to-master-dynamic-programming
#### 1. Prefix Term Sum Pattern
Query the sum of elements in a subarray
```python
#Find subarray sum between two indices of an array i, j
#NOT GOOD for multiple queries O(n*m) m= number of subarray sum queries
#ONLY for 1 query
def find_subarray_(arr, i, j):
	sum = 0
	for k in range(i,j+1):
		sum +=arr[i]
	return sum

#So, for multiple queries, or multiple sub array sums
#So, we can make a Prefix Sum arr
arr = [1,2,3,4,5,6,7]
P = [1,3,6,10,15,21,28] #prefixsum_arr
#where P[i] = arr[0]+arr[1]+....arr[i]
#and, sum of subarray bw i,j is:
#sum[i,j] = P[j] - P[i-1]



```
##### 303. Range Sum Query - Immutable

##### 525. Contiguous Array

##### 560. Subarray Sum Equals K

#### 2. Two Pointers Pattern
```python
#Two Pointer Pattern
#Have two pointers, start and end, and let them move independently.
#start moves towards end
#end moves towards start
#then they meet in the middle, or wherever, however we want
def isPalindrome(string):
    string = str(string)
    start = 0
    end = len(string) -1
    while start < end:
        if string[start] == string[end]:
            start +=1
            end -=1
        else:
            return "NO"
    return "YES"
```

##### Pallindrome

##### Two Sum II - Input Array is sorted (Leetcode 167)

##### 3Sum (Leetcode 15)

##### Container with most water (Leetcode 11)

#### 3. Fast pointer, Slow pointer
Variant of two pointer. Move one pointer fast, other one a bit slow
##### 141. Linked List Cycle

##### 202. Happy Number

##### 287. Find the Duplicate Number


#### 4. Sliding Window Pattern
Helps find sub array with specific criteria or pattern
```python
#Find the subarray of size 'k' with max sum
#using sliding window approach - repetetive sum calculation
#O(n*k) time complexity
arr = [3,2,7,5,9,6,2]
def max_subarray_sum(arr, k):
    n = len(arr)
    i = 0
    max_sum = 0
    curr_sum = 0
    while i+k < n:
        curr_sum = sum(arr[i:i+k]) 
        max_sum = max(curr_sum, max_sum)
        i +=1
    return max_sum
max_subarray_sum(arr, 3)

#Efficient Way O(n) time complexity
def better_max_subarray_sum(arr, k):
    n = len(arr)
    window_sum = sum(arr[:k])
    max_sum = window_sum
    max_start_idx = 0 #window with max sum
    for i in range(n-k):
        #as we iterate thru arr, we subt left most elem from window, and add next elem in window
        window_sum = window_sum - arr[i] + arr[i+k] 
        if window_sum > max_sum:
            max_sum = window_sum
            max_start_idx = i+1  #now slide the max-sum-window 
    return arr[max_start_idx:max_start_idx+k], max_sum

```
##### 643. Maximum Average Subarray I 
##### 3. Longest Substring without Repeating Characters 
##### 76. Minimum Window Substring

#### 5. Linked List in-place reversal
- Swapping nodes, node-pairs in-place, without using extra space
```python
#Reverse the Linked-List 
#Approach 1:
#Given a linked-list, copy the values in an arr and then update the linked-list / make new ll by traversing the arr in reverse

#Approach 2: reverse ll without extra space, and without multiple ll passes
def reverseLL(head):
    prev = None
    curr = head
    while curr is not None:
        next = curr.next
        curr.next = prev
        prev = curr
        curr = next
        # if curr.next:
        #     next = curr.next
        # else:
        #     tail = head
        #     head = curr
    return prev
```

##### 206. Reverse Linked List 
##### 92. Reverse Linked List II 
##### 24. Swap Nodes in Pairs

---
#### 6. Monotonic Stack 
```python 
#Given an arr, find the next greater element for each item. If there isnt one, output -1
def next_greater_elem(arr):
    #brr = [None]*len(arr)
    brr = [-1]*len(arr)
    n = len(arr)
    i = 0
    s = []
    while i < n:
        if not s or arr[i] <= arr[s[-1]]:
            s.append(i)
            i += 1
        else:
            t = s.pop()
            if arr[i] > arr[t]:
                brr[t] = arr[i]
        #if i == n :
        #   for _ in range(len(s)):
        #      t = s.pop()
	    #     brr[t] = -1
    return brr
```
##### 496. Next Greater Element I 
##### 739. Daily Temperatures 
##### 84. Largest Rectangle in Histogram

---

#### 7. Top K. Elements OR min/max Heap 
Helps find k largest/smallest/most-frequent elements in a dataset
For k largest elements -> min-heap
For k smallest elem -> max-heap
##### 215. Kth Largest element in an array 
##### 347. Top K Frequent Elements 
##### 373. Find K Pairs with Smallest Sums

---

#### 8. Overlapping Intervals 
Useful when ranges, intervals etc overlap
Ex: Find the minimum number of meeting rooms needed for overlapping meetings
##### 56. Merge Intervals 
```python
#create empty merged list
#iterate through intervals 
#check if the current interval overlaps the last interval in the merged list
#if yes, then merge the intervals by updating the end-time/end-index of the last interval in mergered
#if not overlaps, then simply append the current interval into the merged list
intervals = [[1,3],[2,6],[8,10],[15,18]]
#o/p = [[1, 6], [8, 10], [15, 18]]
def merge(intervals):
    intervals = sorted(intervals)
    newintv = [None, None]
    merged = [intervals[0]]
    for intv in intervals:
        top = merged[-1]
        if top[0] <= intv[0] <= top[1]:
            print('1newintv[0]', newintv[0] )
            merged[-1][0] = top[0]
            merged[-1][1] = intv[1]
            # if intv[1] <= top[1]:
            #     newintv[1] = top[0]
            # elif top[1] < intv[1]:
            #     newintv[1] = intv[1]
            # merged.append(newintv)
        else:
            merged.append(intv)
        # merged.append(newintv)
    return merged
```
##### 57. Insert Interval 

##### 435. Non-overlapping intervals

---
#### 9. Modified Binary Search
##### 33. Search in Rotated Sorted Array
##### 153. Find Minimum in Rotated Sorted Array
##### 240. Search a 2D Matrix II

---

#### 10. Binary Tree Traversal
- Trees are all about traversal
- When given a tree problem, think about which traversal is the best way to do this. Ex:
	- To retrieve values of binary search tree in sorted order : Inorder Traversal
	- To create a copy of a tree and prob involves tree serialisation: use Preorder Traversal
	- To process child nodes before parent, like when deleting the tree : use Post order traversal
	- 
##### 257. Binary Tree Paths - preorder
##### 230. Kth Smallest Element in a BST - onorder
##### 124. Binary Tree Maximum Path Sum - post order
##### 107. Binary Tree Level Order Traversal II - level order

---

#### 11. Depth First Search
Used to explore all of the paths and branches in trees. Probs like:
- Finding a path between two nodes
- Checking if a graph contains a cycle
- Finding a topological order in a directed acyclic graph
- Counting the number of connected components in a graph
##### 133. Clone Graph
##### 113. Path Sum II
##### 210. Course Schedule II

---

#### 12. Breadth First Search
Imp for problems like:
- Finding the shortest path between two nodes
- Printing all nodes of a tree level by level
-  finding all connected components in a graph
- Finding shortest transformation sequence from one word to other
	- "hit" -> "hot" -> "dot" -> "dog" -> "cog"
- 
##### 102. Binary Tree Level Order Traversal
##### 994. Rotting Oranges
##### 127. Word Ladder

---

#### 13. Matrix Traversal
BFD, DFS is useful in solving the matrix traversal problems
##### 733. Flood Fill
##### 200. Number of Islands 
##### 130. Surrounded Regions

---

#### 14. Backtracking
##### 46. Permutations
##### 78. Subsets
##### 51. N-Queens

---

#### 15. Dynamic Programming
##### 70. Climbing Stairs
##### 322. Coin Change
##### 300. Longest Increasing Subsequence
##### 416. Partition Equal Subset Sum
##### 312. Burst Balloons
##### 1143. Longest Common Subsequence

---
