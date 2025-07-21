---
sticker: emoji//2693
---
[[DSA]]

#### 1. Linked List Concepts and Pseudocode
##### Basic Structure
A Linked List is a dynamic linear data structure where each element is a node containing data and a reference to the next node.
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
- Circular Doubly Linked List
##### Common Operations (Pseudocode)
```python
# Insert at head
def insert_at_head(head, value):
    new_node = Node(value)
    new_node.next = head
    return new_node
# Insert at end
def insert_at_end(head, value):
    new_node = Node(value)
    if not head:
        return new_node
    curr = head
    while curr.next:
        curr = curr.next
    curr.next = new_node
    return head

# Delete a node by value
def delete_node(head, value):
    if not head:
        return None
    if head.data == value:
        return head.next
    curr = head
    while curr.next and curr.next.data != value:
        curr = curr.next
    if curr.next:
        curr.next = curr.next.next
    return head

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
- Tree is a hierarchical data structure composed of nodes.
- The top node is the root; nodes with no children are leaves.
##### Types of Trees
- Binary Tree
- Binary Search Tree (BST)
- Balanced BST (AVL, Red-Black Tree)
- Heap (Min/Max)
- Trie (Prefix Tree)
##### High-Level Tree Implementation
```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
# Inorder traversal
def inorder(root):
    if root:
        inorder(root.left)
        print(root.val)
        inorder(root.right)

# Insert in BST
def insert_bst(root, key):
    if not root:
        return TreeNode(key)
    if key < root.val:
        root.left = insert_bst(root.left, key)
    else:
        root.right = insert_bst(root.right, key)
    return root
```
#### 3. Top 20 Tricks in Python for Linked List Problems
##### 1. Reverse a linked list
Classic iterative method.
```python
def reverse(head):
    prev = None
    while head:
        head.next, prev, head = prev, head, head.next
    return prev
```
##### 2. Detect a cycle in a linked list
Use Floyd's cycle detection.
```python
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow == fast:
            return True
    return False
```
##### 3. Find start of cycle
Extend Floyd's algorithm.
```python
def detect_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow == fast:
            slow = head
            while slow != fast:
                slow, fast = slow.next, fast.next
            return slow
    return None
```
##### 4. Find middle node
Use slow and fast pointers.
```python
def find_middle(head):
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
    return slow
```
##### 5. Merge two sorted lists
Used in merge sort.
```python
def merge(l1, l2):
    dummy = tail = Node(0)
    while l1 and l2:
        if l1.data < l2.data:
            tail.next, l1 = l1, l1.next
        else:
            tail.next, l2 = l2, l2.next
        tail = tail.next
    tail.next = l1 or l2
    return dummy.next
```
##### 6. Remove nth node from end
Two-pass or one-pass with dummy.
```python
def remove_nth(head, n):
    dummy = Node(0)
    dummy.next = head
    fast = slow = dummy
    for _ in range(n): fast = fast.next
    while fast.next:
        fast, slow = fast.next, slow.next
    slow.next = slow.next.next
    return dummy.next
```
##### 7. Check palindrome
Reverse second half and compare.
```python
def is_palindrome(head):
    vals = []
    while head:
        vals.append(head.data)
        head = head.next
    return vals == vals[::-1]
```
##### 8. Swap nodes in pairs
Recursive technique.
```python
def swap_pairs(head):
    if head and head.next:
        new_head = head.next
        head.next = swap_pairs(new_head.next)
        new_head.next = head
        return new_head
    return head
```
##### 9. Rotate list
Move last k nodes to front.
```python
def rotate_right(head, k):
    if not head:
        return None
    # Count length
    length = 1
    old_tail = head
    while old_tail.next:
        old_tail = old_tail.next
        length += 1
    old_tail.next = head
    k = k % length
    new_tail = head
    for _ in range(length - k - 1):
        new_tail = new_tail.next
    new_head = new_tail.next
    new_tail.next = None
    return new_head
```
##### 10. Partition list
Reorder around pivot.
```python
def partition(head, x):
    before = before_head = Node(0)
    after = after_head = Node(0)
    while head:
        if head.data < x:
            before.next = head
            before = before.next
        else:
            after.next = head
            after = after.next
        head = head.next
    after.next = None
    before.next = after_head.next
    return before_head.next
```
##### 11. Add two numbers
Add digits as reversed list.
```python
def add_two_numbers(l1, l2):
    dummy = curr = Node(0)
    carry = 0
    while l1 or l2 or carry:
        v1 = l1.data if l1 else 0
        v2 = l2.data if l2 else 0
        carry, out = divmod(v1 + v2 + carry, 10)
        curr.next = Node(out)
        curr = curr.next
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    return dummy.next
```
##### 12. Remove duplicates
For sorted lists.
```python
def remove_duplicates(head):
    curr = head
    while curr and curr.next:
        if curr.data == curr.next.data:
            curr.next = curr.next.next
        else:
            curr = curr.next
    return head
```
##### 13. Intersection of two lists
Use set or double pointer.
```python
def get_intersection(l1, l2):
    seen = set()
    while l1:
        seen.add(l1)
        l1 = l1.next
    while l2:
        if l2 in seen:
            return l2
        l2 = l2.next
    return None
```
##### 14. Flatten multilevel list
DFS or recursion based.
```python
def flatten(head):
    if not head:
        return None
    dummy = Node(0)
    stack, prev = [head], dummy
    while stack:
        curr = stack.pop()
        prev.next = curr
        if curr.next:
            stack.append(curr.next)
        if hasattr(curr, 'child') and curr.child:
            stack.append(curr.child)
        prev = curr
    return dummy.next
```
##### 15. Sort linked list
Merge sort on list.
```python
def sort_list(head):
    if not head or not head.next:
        return head
    # Find middle
    slow, fast = head, head.next
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
    mid, slow.next = slow.next, None
    left = sort_list(head)
    right = sort_list(mid)
    return merge(left, right)
```
##### 16. Copy list with random pointer
Use hashmap to clone nodes.
```python
def copy_random_list(head):
    if not head:
        return None
    old_to_new = {}
    curr = head
    while curr:
        old_to_new[curr] = Node(curr.data)
        curr = curr.next
    curr = head
    while curr:
        old_to_new[curr].next = old_to_new.get(curr.next)
        old_to_new[curr].random = old_to_new.get(curr.random)
        curr = curr.next
    return old_to_new[head]
```
##### 17. Reverse in k-group
Recursively reverse blocks of k.
```python
def reverse_k_group(head, k):
    count, node = 0, head
    while node and count < k:
        node = node.next
        count += 1
    if count == k:
        prev = reverse_k_group(node, k)
        while count:
            tmp = head.next
            head.next = prev
            prev = head
            head = tmp
            count -= 1
        return prev
    return head
```
##### 18. Odd even linked list
Group nodes by index parity.
```python
def odd_even_list(head):
    if not head:
        return head
    odd, even = head, head.next
    even_head = even
    while even and even.next:
        odd.next, even.next = even.next, even.next.next
        odd, even = odd.next, even.next
    odd.next = even_head
    return head
```
##### 19. Detect intersection with two pointers
Align lengths, then move together.
```python
def get_intersection(l1, l2):
    a, b = l1, l2
    while a != b:
        a = a.next if a else l2
        b = b.next if b else l1
    return a
```
##### 20. Dummy node pattern
Simplifies edge cases.
```python
def delete_value(head, val):
    dummy = Node(0)
    dummy.next = head
    prev, curr = dummy, head
    while curr:
        if curr.data == val:
            prev.next = curr.next
        else:
            prev = curr
        curr = curr.next
    return dummy.next
```