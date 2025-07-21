---
sticker: emoji//1f333
---
[[DSA]]
#### Tree Data Structure - Interview Preparation Notes
##### 1. Basics of Tree
A **Tree** is a non-linear hierarchical data structure made up of nodes, where each node points to its child nodes. A tree with `n` nodes has `n-1` edges.
- **Root**: Topmost node.
- **Leaf**: Node with no children.
- **Edge**: Connection between parent and child.
- **Depth**: Distance from root to the node.
- **Height**: Distance from node to the deepest leaf.
- **Subtree**: Any node and its descendants.
##### 2. Types of Trees
- **Binary Tree**: Each node has at most two children.
- **Binary Search Tree (BST)**: Left < Root < Right.
- **Balanced Tree**: Height of left and right subtrees differ by at most 1.
- **Complete Binary Tree**: All levels filled except possibly last, filled left to right.
- **Perfect Binary Tree**: All levels fully filled.
- **Full Binary Tree**: Every node has 0 or 2 children.
- **AVL Tree**: Self-balancing BST.
- **N-ary Tree**: A node can have `n` children.
- **Trie**: Tree used for prefix-based retrieval (e.g., autocomplete).
##### 3. High-level Tree Implementation in Python
```python
# Basic binary tree node
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Inserting in BST
def insert_bst(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insert_bst(root.left, val)
    else:
        root.right = insert_bst(root.right, val)
    return root
```
---
#### Common Python Tricks & Functions for Tree Problems
##### 1. Recursive Tree Traversal
Use recursive DFS to visit nodes.
```python
def inorder(root):
    if root:
        inorder(root.left)
        print(root.val)
        inorder(root.right)
```
##### 2. Iterative Traversal Using Stack
Simulates recursion via a stack.
```python
def inorder_iter(root):
    stack, curr = [], root
    while stack or curr:
        if curr:
            stack.append(curr)
            curr = curr.left
        else:
            curr = stack.pop()
            print(curr.val)
            curr = curr.right
```
##### 3. Level Order Traversal (BFS)
Use a queue to visit level by level.
```python
from collections import deque
def bfs(root):
    q = deque([root])
    while q:
        node = q.popleft()
        print(node.val)
        if node.left: q.append(node.left)
        if node.right: q.append(node.right)
```
##### 4. Finding Height of Tree
Recursive depth count.
```python
def height(root):
    if not root: return 0
    return 1 + max(height(root.left), height(root.right))
```
##### 5. Diameter of Tree
Longest path between any two nodes.
```python
def diameter(root):
    res = [0]
    def dfs(node):
        if not node: return 0
        L = dfs(node.left)
        R = dfs(node.right)
        res[0] = max(res[0], L + R)
        return 1 + max(L, R)
    dfs(root)
    return res[0]
```
##### 6. Lowest Common Ancestor (BST)
Traverse down to find split point.
```python
def lca(root, p, q):
    if p.val < root.val and q.val < root.val:
        return lca(root.left, p, q)
    if p.val > root.val and q.val > root.val:
        return lca(root.right, p, q)
    return root
```
##### 7. Invert/Flip a Binary Tree
Mirror the tree recursively.
```python
def invert(root):
    if root:
        root.left, root.right = invert(root.right), invert(root.left)
    return root
```
##### 8. Check Symmetry
Compare left and right subtrees.
```python
def is_symmetric(root):
    def is_mirror(t1, t2):
        if not t1 and not t2: return True
        if not t1 or not t2: return False
        return t1.val == t2.val and is_mirror(t1.left, t2.right) and is_mirror(t1.right, t2.left)
    return is_mirror(root, root)
```
##### 9. Path Sum Exists
Check if a path from root to leaf equals sum.
```python
def has_path_sum(root, sum):
    if not root: return False
    if not root.left and not root.right: return root.val == sum
    return has_path_sum(root.left, sum - root.val) or has_path_sum(root.right, sum - root.val)
```
##### 10. Collect All Root to Leaf Paths
Store paths during traversal.
```python
def binary_tree_paths(root):
    paths = []
    def dfs(node, path):
        if node:
            if not node.left and not node.right:
                paths.append(path + str(node.val))
            dfs(node.left, path + str(node.val) + '->')
            dfs(node.right, path + str(node.val) + '->')
    dfs(root, '')
    return paths
```
##### 11. Validate BST
Ensure left < node < right.
```python
def is_valid_bst(root, low=float('-inf'), high=float('inf')):
    if not root: return True
    if not (low < root.val < high): return False
    return is_valid_bst(root.left, low, root.val) and is_valid_bst(root.right, root.val, high)
```
##### 12. Count Nodes
Use DFS to count.
```python
def count_nodes(root):
    if not root: return 0
    return 1 + count_nodes(root.left) + count_nodes(root.right)
```
##### 13. Right View of Binary Tree
Last node of each level.
```python
def right_view(root):
    from collections import deque
    res, q = [], deque([root])
    while q:
        size = len(q)
        for i in range(size):
            node = q.popleft()
            if i == size - 1:
                res.append(node.val)
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
    return res
```
##### 14. Zigzag Level Order
Alternate left-to-right and right-to-left.
```python
def zigzag_level_order(root):
    from collections import deque
    res, q, left = [], deque([root]), True
    while q:
        level = deque()
        for _ in range(len(q)):
            node = q.popleft()
            if left:
                level.append(node.val)
            else:
                level.appendleft(node.val)
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
        res.append(list(level))
        left = not left
    return res
```
##### 15. Serialize/Deserialize Tree
Encode and decode tree structure.
```python
def serialize(root):
    vals = []
    def dfs(node):
        if node:
            vals.append(str(node.val))
            dfs(node.left)
            dfs(node.right)
        else:
            vals.append('#')
    dfs(root)
    return ' '.join(vals)

def deserialize(data):
    vals = iter(data.split())
    def dfs():
        val = next(vals)
        if val == '#': return None
        node = TreeNode(int(val))
        node.left = dfs()
        node.right = dfs()
        return node
    return dfs()
```
##### 16. Flatten Binary Tree to Linked List
Transform tree to right-skewed list.
```python
def flatten(root):
    def dfs(node):
        if not node: return None
        left_tail = dfs(node.left)
        right_tail = dfs(node.right)
        if node.left:
            left_tail.right = node.right
            node.right = node.left
            node.left = None
        return right_tail or left_tail or node
    dfs(root)
```
##### 17. Build Tree from Inorder and Preorder
Construct tree using indices.
```python
def build(preorder, inorder):
    idx_map = {v: i for i, v in enumerate(inorder)}
    def helper(pre_start, in_start, in_end):
        if pre_start >= len(preorder) or in_start > in_end:
            return None
        root = TreeNode(preorder[pre_start])
        in_idx = idx_map[root.val]
        root.left = helper(pre_start + 1, in_start, in_idx - 1)
        root.right = helper(pre_start + 1 + in_idx - in_start, in_idx + 1, in_end)
        return root
    return helper(0, 0, len(inorder) - 1)
```
##### 18. Postorder Iterative
Reverse modified preorder.
```python
def postorder_iterative(root):
    if not root: return []
    stack, output = [root], []
    while stack:
        node = stack.pop()
        output.append(node.val)
        if node.left: stack.append(node.left)
        if node.right: stack.append(node.right)
    return output[::-1]
```
##### 19. Kth Smallest in BST
Use in-order traversal.
```python
def kth_smallest(root, k):
    stack = []
    while True:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        k -= 1
        if k == 0: return root.val
        root = root.right
```
##### 20. Check Balanced Tree
Check height and balance status.
```python
def is_balanced(root):
    def dfs(node):
        if not node: return 0
        L = dfs(node.left)
        if L == -1: return -1
        R = dfs(node.right)
        if R == -1: return -1
        if abs(L - R) > 1: return -1
        return 1 + max(L, R)
    return dfs(root) != -1
```
