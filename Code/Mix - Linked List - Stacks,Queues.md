---
sticker: emoji//1faa2
tags: []
---
[[DSA]]

#### 1. Linked List Concepts and Pseudocode
##### Basic Structure
A Linked List is a linear data structure where elements are stored in nodes and each node points to the next one.
```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
```
##### Types of Linked Lists
- Singly Linked List
- Doubly Linked List
- Circular Linked List
##### Common Operations (Pseudocode)
- Insert at beginning
- Insert at end
- Delete a node
- Search for an element
- Reverse the list
```python
# Insert at beginning
def insert_at_head(head, value):
    new_node = Node(value)
    new_node.next = head
    return new_node
# Reverse a list
def reverse_list(head):
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev
```
#### 2. Tree Basics and Implementation
##### Tree Concepts
- Tree is a hierarchical data structure with nodes connected by edges.
- Root: Top node
- Leaf: Node with no children
##### Types of Trees
- Binary Tree: Each node has at most 2 children.
- Binary Search Tree (BST): Left < Root < Right
- AVL Tree: Self-balancing BST
- Trie: Tree for string storage
- Heap: Complete binary tree, used in priority queues
##### High-Level Tree Implementation
```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
# Inorder Traversal (LNR)
def inorder(root):
    if root:
        inorder(root.left)
        print(root.val)
        inorder(root.right)
```
#### 3. Top 20 Tricks in Python for Queue, Stack, Deque
##### 1. Stack using list
Simple LIFO structure.
```python
stack = []
stack.append(10)
stack.pop()
```
##### 2. Queue using deque
Efficient FIFO with deque.
```python
from collections import deque
queue = deque()
queue.append(10)
queue.popleft()
```
##### 3. Deque double end
Allows push/pop both ends.
```python
dq = deque()
dq.appendleft(1)
dq.append(2)
dq.pop()
```
##### 4. Reversing deque
Use reverse method.
```python
dq = deque([1,2,3])
dq.reverse()
```
##### 5. Check if deque is palindrome
Compare with reversed.
```python
dq = deque('madam')
print(list(dq) == list(reversed(dq)))
```
##### 6. Using deque for sliding window
Fast access both ends.
```python
def max_sliding_window(nums, k):
    dq = deque()
    result = []
    for i, n in enumerate(nums):
        while dq and nums[dq[-1]] < n:
            dq.pop()
        dq.append(i)
        if dq[0] == i - k:
            dq.popleft()
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result
```
##### 7. Queue with max size
Auto drop with maxlen.
```python
dq = deque(maxlen=3)
dq.extend([1,2,3,4])
```
##### 8. Stack using deque
Deque as a fast stack.
```python
stack = deque()
stack.append(1)
stack.pop()
```
##### 9. Priority queue using heapq
Push and pop smallest element.
```python
import heapq
heap = []
heapq.heappush(heap, 3)
heapq.heappop(heap)
```
##### 10. Max-heap simulation
Use negative values.
```python
heapq.heappush(heap, -x)
-max(heapq.heappop(heap))
```
##### 11. Queue using two stacks
Classic interview pattern.
```python
s1, s2 = [], []
def enqueue(x): s1.append(x)
def dequeue():
    if not s2:
        while s1: s2.append(s1.pop())
    return s2.pop()
```
##### 12. Balanced Parentheses
Using stack to match pairs.
```python
def is_balanced(s):
    stack = []
    for c in s:
        if c in '([{': stack.append(c)
        else:
            if not stack or {')':'(', ']':'[', '}':'{'}.get(c) != stack.pop():
                return False
    return not stack
```
##### 13. Min stack
Track min value with stack.
```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []
    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    def pop(self):
        if self.stack.pop() == self.min_stack[-1]:
            self.min_stack.pop()
```
##### 14. Deque rotate

Rotate elements.

```python

dq = deque([1,2,3])

dq.rotate(1)

```

  

##### 15. Round Robin Scheduling

Rotate tasks.

```python

tasks = deque(['a', 'b', 'c'])

while tasks:

    task = tasks.popleft()

    tasks.append(task)

```

  

##### 16. Monotonic Stack

Used in next greater element.

```python

def next_greater(nums):

    res, stack = [-1]*len(nums), []

    for i in range(len(nums)):

        while stack and nums[stack[-1]] < nums[i]:

            res[stack.pop()] = nums[i]

        stack.append(i)

    return res

```

  

##### 17. Using stack for DFS

Iterative DFS using stack.

```python

def dfs(graph, start):

    visited, stack = set(), [start]

    while stack:

        node = stack.pop()

        if node not in visited:

            visited.add(node)

            stack.extend(graph[node])

```

  

##### 18. BFS using queue

Level-order traversal.

```python

def bfs(graph, start):

    visited, queue = set(), deque([start])

    while queue:

        node = queue.popleft()

        if node not in visited:

            visited.add(node)

            queue.extend(graph[node])

```

  

##### 19. Deque as sliding max

Optimized for window max.

```python

from collections import deque

# See #6 for example

```

  

##### 20. Using list for simple stack problems

List suffices for small stack problems.

```python

stack = []

for c in "123":

    stack.append(c)

```