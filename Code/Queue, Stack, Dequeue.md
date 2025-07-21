---
sticker: emoji//1f4d0
---
[[DSA]]
#### 1. Using collections.deque for Queue and Stack
`deque` supports O(1) append and pop from both ends, making it ideal for queue and stack operations.
```python
from collections import deque
q = deque()
q.append(1)      # enqueue
q.popleft()      # dequeue
s = deque()
s.append(1)      # push
s.pop()          # pop
```
#### 2. Using list as Stack (LIFO)
Python list supports append and pop from end; it's a natural fit for stack use.
```python
stack = []
stack.append(1)
stack.append(2)
stack.pop()  # returns 2
```
#### 3. Using list as Queue (inefficient)
Avoid using `list.pop(0)` in tight loops due to O(n) time; prefer `deque`.
```python
queue = [1, 2, 3]
queue.pop(0)  # returns 1, but O(n)
```
#### 4. Check if Stack is Empty
An empty list evaluates to False; commonly used to check stack/queue state.
```python
stack = []
if not stack:
    print("Stack is empty")
```
#### 5. Monotonic Stack Pattern ✅
Used for problems like Next Greater Element, keeping increasing/decreasing order.
```python
nums = [2, 1, 2, 4, 3]
stack = []
res = [0]*len(nums)
for i in range(len(nums)-1, -1, -1):
    while stack and stack[-1] <= nums[i]:
        stack.pop()
    res[i] = stack[-1] if stack else -1
    stack.append(nums[i])

#Next Larger Element:
def nextLargerElement(self, arr):
	n = len(arr)
	brr = [None]*len(arr)
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
		if i == n :
			for _ in range(len(s)): #if you've reached end of the array, then empty the stack
				t = s.pop()
				brr[t] = -1
	return brr
```
#### 6. Two Stacks for Queue (Queue via Stacks)
Implements queue using two stacks; one for input, one for output.
```python
in_stack = []
out_stack = []
def enqueue(x):
    in_stack.append(x)
def dequeue():
    if not out_stack: #while outstack is empty
        while in_stack: #while instack is not empty
            out_stack.append(in_stack.pop())
    return out_stack.pop()
```
#### 7. Reversing Queue using Stack
A classic trick in queue manipulation.
```python
from collections import deque
q = deque([1, 2, 3])
stack = []
while q:
    stack.append(q.popleft())
while stack:
    q.append(stack.pop())
```
#### 8. Sliding Window Maximum (Deque)
Maintains indices in decreasing order to track max in window.
```python
from collections import deque
def max_sliding_window(nums, k):
    dq, res = deque(), []
    for i in range(len(nums)):
        if dq and dq[0] <= i - k:
            dq.popleft()
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            res.append(nums[dq[0]])
    return res
```
#### 9. Circular Queue Using List ✅
Wraps around using modulo to implement fixed-size queue.
```python
class CircularQueue:
    def __init__(self, k):
        self.q = [None] * k
        self.head = self.tail = self.size = 0
        self.cap = k
    def enqueue(self, val):
        if self.size == self.cap:
            return False
        self.q[self.tail] = val
        self.tail = (self.tail + 1) % self.cap
        self.size += 1
        return True
    def dequeue(self):
        if self.size == 0:
            return False
        self.head = (self.head + 1) % self.cap
        self.size -= 1
        return True
```
#### 10. Min Stack (Tracking Minimum)
Tracks minimum in stack at each level using a second stack.
```python
stack = []
min_stack = []
def push(x):
    stack.append(x)
    if not min_stack or x <= min_stack[-1]:
        min_stack.append(x)
def pop():
    if stack.pop() == min_stack[-1]:
        min_stack.pop()
```
#### 11. Implement Stack with Max Tracking
Similar to min stack, but keeps track of max value.
```python
stack = []
max_stack = []
def push(x):
    stack.append(x)
    if not max_stack or x >= max_stack[-1]:
        max_stack.append(x)
def pop():
    if stack.pop() == max_stack[-1]:
        max_stack.pop()
```
#### 12. Reverse Stack Recursively
Classic recursion trick to reverse a stack without extra space.
```python
def insert_bottom(s, val):
    if not s:
        s.append(val)
    else:
        top = s.pop()
        insert_bottom(s, val)
        s.append(top)
def reverse_stack(s):
    if s:
        top = s.pop()
        reverse_stack(s)
        insert_bottom(s, top)
```
#### 13. Balanced Parentheses (Stack) ✅
Standard use case for stack: check for matching parentheses.
```python
def is_balanced(expr):
    stack = []
    mapping = {')':'(', ']':'[', '}':'{'}
    for ch in expr:
        if ch in mapping.values():
            stack.append(ch)
        elif ch in mapping:
            if not stack or mapping[ch] != stack.pop():
                return False
    return not stack
```
#### 14. LRU Cache with OrderedDict
`OrderedDict` preserves order of insertion, perfect for LRU cache.
```python
from collections import OrderedDict
class LRUCache:
    def __init__(self, cap):
        self.cache = OrderedDict()
        self.cap = cap
    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return -1
    def put(self, key, val):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = val
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)
```
#### 15. Deque Rotation (for circular array problems)
`deque.rotate()` is useful for rotating elements efficiently.
```python
from collections import deque
d = deque([1, 2, 3, 4])
d.rotate(1)  # [4, 1, 2, 3]
d.rotate(-1) # [1, 2, 3, 4]
```
#### 16. BFS using Queue
Breadth-first search always uses a queue for level-order traversal.
```python
from collections import deque
def bfs(graph, start):
    visited = set()
    q = deque([start])
    while q:
        node = q.popleft()
        if node not in visited:
            visited.add(node)
            q.extend(graph[node])
```
#### 17. DFS using Stack
Depth-first search can be implemented using an explicit stack.
```python
def dfs(graph, start):
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(graph[node])
```
#### 18. Simulate k queues in one array
Use extra bookkeeping to simulate multiple queues in one array (used in some competitive problems).
```python
# Just a conceptual structure. Implementation is complex.
# You need to track front[], rear[], next[] and free pointer
```
#### 19. Detect Cycle in Directed Graph (DFS + Rec Stack)
DFS with a recursion stack helps detect cycles in graphs.
```python
def has_cycle(graph, node, visited, rec_stack):
    visited.add(node)
    rec_stack.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            if has_cycle(graph, neighbor, visited, rec_stack):
                return True
        elif neighbor in rec_stack:
            return True
    rec_stack.remove(node)
    return False
```
#### 20. Deque for Palindrome Checking
Use deque to efficiently compare front and back characters.
```python
from collections import deque
def is_palindrome(s):
    d = deque(s)
    while len(d) > 1:
        if d.popleft() != d.pop():
            return False
    return True
```