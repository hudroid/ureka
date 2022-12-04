# Remember what I went through Jun 2022

# print days between
from asyncio.windows_events import INFINITE
import collections
from datetime import date, timedelta
from email import message_from_bytes
from email.mime import image
from email.policy import default
from heapq import merge
from http.client import NETWORK_AUTHENTICATION_REQUIRED
from logging.config import valid_ident
from math import inf
from multiprocessing import dummy
from optparse import Option
from re import M
from typing import DefaultDict, List
from unicodedata import digit
from xml.dom.minicompat import NodeList

from pyparsing import Optional

sDate = date(2022,2,1)
eData = date(2022,3,1)

delta = eData-sDate
for i in range(delta.days+1):
    day = sDate + timedelta(days=i)
    print(day)

# invert image

# find word in matrix

# group numbers in list

# Design parking lot OOP

# missing number in array

# count island surrounded by land

# pivot index -> left sum == right sum
def pivotIndex(self, nums:List[int]) -> int:
    l, r = 0, sum(nums)
    for index, num in enumerate(nums):
        r -= num
        if l == r:
            return index
        l += num
    return -1

# largest n at least twice of others
def dominantInd(self, nums:List[int]) -> int:
    highest = -1
    second = -1
    ind = 0

    for i,n in enumerate(nums):
        if n > highest:
            second = highest
            highest = n
            ind = i
        elif n > second:
            second = n
        if highest < second *2:
            ind = -1
    return ind
            
def plusOne(self, digits:List[int]) -> List[int]:
    if len(digits) == 1 and digit[0] == 9:
        return [1,0]
    
    if digits[-1] != 9:
        digits[-1] += 1
        return digits
    else:
        digits[-1] = 0
        digits[:-1] = self.plusOne(digits[:-1])
        return digits

# build diagonal arr
# 1. Diagonals are defined by the sum of indicies in a 2 dimensional array
# 2. The snake phenomena can be achieved by reversing every other diagonal level, therefore check if divisible by 2
def findDiagonalOrder(self, matrix):
    d = {}
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if i+j in d:
                d[i+j].append(matrix[i][j])                
            else:
                d[i+j] = [matrix[i][j]]
    ans = []
    for entry in d.items():
        if entry[0]%2 == 0:
            [ans.append(x) for x in entry[1][::-1]]
        else:
            [ans.append(x) for x in entry[1]]
    return ans

def spiralOrder(self, matrix:List[List[int]])-> List[int]:
    height = len(matrix)
    width = len(matrix[0])
    
    top = 0
    bottom = height - 1
    left = 0
    right = width - 1
    
    ans = []
    while top < bottom and left < right:
        for col in range(left, right):
            ans.append(matrix[top][col])
        
        for row in range(top, bottom):
            ans.append(matrix[row][right])
        
        for col in range(right, left, -1):
            ans.append(matrix[bottom][col])
        
        for row in range(bottom, top, -1):
            ans.append(matrix[row][left])
        
        top += 1
        bottom -= 1
        left += 1
        right -= 1
    
    # If a matrix remain inside it is either a 1xn or a mx1
    # a linear scan will return the same order as spiral for these
    if len(ans) < height*width:
        for row in range(top, bottom+1):
            for col in range(left, right+1):
                ans.append(matrix[row][col])
    
    return ans

    # Pascal triangle
def generate(self, numRows:int) -> List[List[int]]:
    rows = [[1]]

    for r in range(1, numRows):
        rows.append([1]*(r+1))
        for c in range(1, r):
            rows[r][c] = rows[r-1][c] + rows[r][c-1]
    return rows

#pascal triangle : return get rows
def getRow(self, rowIndex:int) -> List[int]:
    row = [1] * (rowIndex+1)
    if rowIndex == 0:
        return row
    
    prev_row = self.getRow(rowIndex-1)
    for i in range(1, len(row)-1):
        row[i] = prev_row[i-1] + prev_row[i]
    return row


def addBinary(self, a:str, b:str) -> str:
    res = ""
    i, j, carry = len(a), len(b), 0

    while i>=0 or j>=0:
        sum = carry
        if i >=0:
            sum += ord(a[i])-ord('0')
        if j >= 0:
            sum += ord(b[j]) - ord('0')
        i, j = i-1, j-1

        carry == 1 if sum > 1 else 0
        res += str(sum % 2)

    return res[::-1]


def strStr(self, haystack:str, needle:str) -> int:
    if len(needle) == 0:
        return 0

    for i in range(len(haystack)):
        if haystack[i:i+len(needle)] == needle:
            return i

    return -1

def longestCommonPrefix(self, strs:List[str]) -> str:
    if not strs:
        return ""
    shortest = min(strs, key=len)
    for i,ch in enumerate(shortest):
        for other in strs:
            if other[i] != ch:
                return shortest[:i]
    return shortest

# return the minimal length of a contiguous subarray >= target
def minSubArray(self, target:int, nums:List[int]) -> int:
    total = 0
    left = 0
    res = len(nums)+1

    # O(N) sol
    for right, n in enumerate(nums):
        total += n
        while total >= target:
            res = min(res, right-left+1)
            total -= nums[left]
            left += 1
    return res if res <= len(nums) else 0

# reverse words
def reverseWords(self, s:str) -> str:
    return " ".join(s.split()[::-1])

# s = "Let's take LeetCode contest" => "s'teL ekat edoCteeL tsetnoc"
def revWords(self, s:str) -> str:
    return ' '.join(x[::-1] for x in s.split())

# remove dup and return unique count
def removeDup(self, nums:List[int]) -> int:
    x = 1
    for i in range(len(nums)-1):
        if nums[i] != nums[i+1]:
            nums[x] = nums[i+1]
            x += 1
    return x

# move zeros
def moveZeroes(self, nums:List[int]) -> None:
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != 0 and nums[slow] == 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
        if nums[slow] != 0:
            slow += 1

# Design my linked list
class ListNode(object):
    def __init__(self,val):
        self.val = val
        self.next = None

class MyLinkedList(object):
    def __init__(self):
        self.head = None
        self.size = 0

    def get(self, index:int) -> int:
        if index < 0 or index >= self.size:
            return -1
        
        curr = self.head
        for _ in range(0, index):
            curr = curr.next
        return curr.val
    
    def addAtTail(self, val:int) -> None:
        self.addatIndex(self.size, val)
    
    def addAtIndex(self, index:int, val:int) -> None:
        if index > self.size:
            return
        
        current = self.head
        new_node = ListNode(val)

        if index <= 0:
            new_node.next = current
            self.head = new_node
        else:
            for _ in range(index - 1):
                current = current.next
            new_node.next = current.next
            current.next = new_node

        self.size += 1
    
    # get intesect node
    def getIntersectionNode(self, headA:ListNode, headB:ListNode) -> Optional[ListNode]:
        while headA is None or headB is None:
            return None
        pa = headA
        pb = headB

        while pa is not pb:
            pa = headB if pa is None else pa.next
            pb = headA if pb is None else pb.next

        return pa
    
    # remove nth from last
    def removeNthFromEnd(self, head:Optional[ListNode], n:int) -> Optional[ListNode]:
        slow, fast = head,head
        for _ in range(n):
            fast = fast.next
        if not fast:
            return head.next

        while fast.next:
            fast, slow = fast.next, slow.next
        slow.next = slow.next.next
        return head
    
    # odd even group
    def oddEvenList(self, head:Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return head
        
        odd = head
        even = head.next
        evenHead = even

        while even and even.next:
            odd.next = odd.next.next
            even.next = even.next.next

            odd = odd.next
            even = even.next

        odd.next = evenHead
        return head

# isPalindrome
def isPalindrome(self, head:Optional[ListNode]) -> bool:
    fast = head
    slow = head
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next

    node = None
    while slow:
        _next = slow.next
        slow.next = node
        node = slow
        slow = _next
    
    while node:
        if node.val != head.val:
            return False
        node = node.next
        head = head.next
    return True

# addTwoNum
def addTwoNumbers(self, l1:Optimal[ListNode], l2[ListNode]) -> Optimal[ListNode]:
    dummy = curr = ListNode()
    carry = 0

    if l1 or l2 or carry:
        if l1:
            carry += l1.val
            l1 = l1.next
        if l2:
            carry += l2.val
            l2 = l2.next
        curr.next = ListNode(carry % 10)
        curr = curr.next
        carry //= 10
    return dummy.next

# BST level order
def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
        return []
    res, lvl = [], [root]
    while lvl:
        res.append([node.val for node in lvl])
        tmp = []
        for node in lvl:
            tmp.extend([node.left, node.right])
        lvl = [leaf for leaf in tmp if leaf]
    return res

# symmetric tree
def isSymmetricRecursively(self, root:Optional[TreeNode]) -> bool:
    if root is None:
        return True
    else:
        self.isMirror(root.left, root.right)

def isMirror(self, left, right):
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    if left.val == right.val:
        outter = self.isMirror(left.left, right.right)
        inner = self.isMirror(left.right, right.left)
        return outter and inner
    else:
        return False
def isSymmetricInterativley(self, root:Optional[TreeNode]) -> bool:

    if not root:
        return True
    q = collections.deque([root.left, root.right])
    while q:
        t1, t2 = q.popleft(), q.popleft()
        if not t1 and not t2:
            continue
        elif (not t1 or not t2) or (t1.val != t2.val):
            return False
        q += [t1.left, t1.right, t2.left, t2.right]
    return True

def hasPathSum(self, root:Optional[TreeNode], target:int) -> bool:
    if not root:
        return False
    if not root.left and root.right and root.val == target:
        return True
    target -= root.val
    return self.hasPathSum(root.left, target) or self.hasPathSum(root.right,target)

# build tree from inorder and postorder
def buildTree(self, inorder:List[int], postorder:List[int]) -> Optional[TreeNode]:
    map_inorder = {}
    for i, val in enumerate(inorder):
        map_inorder[val] = i
        
        def helper(start, end):
            if start > end:
                return None
            x = TreeNode(postorder.pop())
            mid = map_inorder[x.val]
            x.right = helper(mid+1,end)
            x.left = helper(start,mid-1)
            return x

    return helper(0, len(inorder)-1)


def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    if root == p or root == q:
        return root
    
    left = right = None

    if not root.left:
        left = self.lowestCommonAncestor(root.left, p,q)
    if not root.right:
        right = self.lowestCommonAncestor(root.right, p,q)

    if left and right:
        return root
    else:
        return left or right


def serialize(self, root):
    if not root:
        return 'x'

    return str(root.val) +"," + self.serialize(root.left) + "," + self.serialize(root.right)


def deserialize(self, data):
    nodes = data.split(",")
    self.i = 0

    def dfs():
        if self.i == len(nodes):
            return None
        nodeVal = nodes[self.i]
        self.i ++ 1
        if nodeVal == 'x':
            return None
        
        root = TreeNode(int(nodeVal))
        root.left = dfs()
        root.right = dfs()
        return root

    return dfs()

# populate next right
def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
    if not root:
        return None
    q = deque([root])
    while q:
        rightNode = None
        for _ in range(len(q)):
            curr = q.popleft()
            curr.next, rightNode = rightNode, curr
            if curr.next:
                q.extend([root.left, root.right])
    return root

# design circular queue with linkedlist
class ListNode:
    def __init__(self, val:int, nxt:ListNode = None) -> None:
        self.val = val
        self.next = nxt
    
class MyCircularQueue:
    def __init__(self, k:int):
        self.maxSize = k
        self.size = 0
        self.head = None
        self.tail = None

    def enqueue(self, val:int) -> bool:
        if self.isFull():
            self.head = self.tail = newNode
        else:
            self.tail.next = newNode
            self.tail = self.tail.next
        self.size += 1
        return True

    def dequeue(self) -> bool:
        if self.isEmpty(): return False
        self.head = self.head.next
        self.size -= 1
        return True
    
    def isEmpty(self) -> bool:
        return self.size == 0
    
    def isFull(self) -> bool:
        return self.size == self.maxSize
    
    def Front(self) -> int:
        return -1 if self.isEmpty() else self.head.val 

# moving avg
class MovingAverage:
    def __init__(self, size:int):
        self.q = collections.deque(maxlen=size)
    
    def next(self, val:int) -> float:
        q = self.q
        q.append(val)
        return float(sum(q))/len(q)

# number of island
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0
        
        r = len(grid)
        c = len(grid[0])
        cntr = 0
        
        for i in range(r):
            for j in range(c):
                if grid[i][j] == '1':
                    self.dfs (grid,i,j)
                    cntr += 1
        return cntr

    def dfs(self, grid, i, j):
        if i<0 or j<0 or i>=len(grid) or j>=len(grid[0]) or grid[i][j] != '1':
            return
        
        grid[i][j] = '#'
        
        self.dfs(grid, i+1, j)
        self.dfs(grid, i-1, j)
        self.dfs(grid, i, j+1)
        self.dfs(grid, i, j-1)

# perfect square ????
def numSquares(self, n):
    sq = [i**2 for i in range(1, int(n**0.5)+1)]
    d, q, nq = 1, {n}, set()
    while q:
        for node in q:
            for s in sq:
                if node == s:
                    return d
                if node < s:
                    break
                nq.add(node-s)
        q, nq, d = nq, set(), d+1
# clone Graph
def cloneGraph(self, node:'Node') -> 'Node':
    if not node:
        return node

    m, visited,stack = dict(), set(), deque([node])
    while stack:
        n = stack.pop()
        if n in visited:
            continue
        visited.add(n)
        if n not in m:
            m[n] = Node(n.val)
        for neigh in n.neighbors:
            if neigh not in m:
                m[neigh] = Node(neigh.val)
            m[n].neighbors.append(m[neigh])
            stack.append(neigh)
    return m[node]
# implement stack using queue
class Queue(object):
    def __init__(self) -> None:
        self.inStack = []
        self.outStack = []
    def push(self,x):
        self.inStack.append(x)
    def pop(self):
        self.move()
        self.outStack.pop()
    def peek(self):
        self.move()
        return self.outStack[-1]
    def move(self):
        if not self.outStack:
            while self.inStack:
                self.outStack.append(self.inStack.pop())




            

# minstack
class MinStack:
    def __init__(self) -> None:
        self.q = []
    
    def push(self, val:int) -> None:
        currMin = self.getMin()
        if currMin == None or val < currMin:
            currMin = val
        self.q.append((val, currMin))
    
    def pop(self) -> None:
        self.q.pop()

    def top(self) -> int:
        if len(self.q) == 0:
            return None
        else:
            self.q[len(self.q)-1][0]
    def getMin(self) -> int:
        if len(self.q) == 0:
            return None
        else:
            self.q[len(self.q)-1][1]

# valid parenthesis
def isValid(self, s:str) -> bool:
    stack = []
    dict = {"]","]"
            "}","{"
            }
    for ch in s:
        if ch in dict.values():
            stack.append(ch)
        elif ch in dict.keys():
            if stack == [] or dict[ch] != dict.pop():
                return False
            else:
                return True
    return stack == []

# daily temp until next high
def dailyTemp(self, temp:List[int]) -> List[int]:
    res = [0] * len(temp)
    stack = []

    for i, val in enumerate(temp):
        while stack and temp[stack[-1]] < val:
            curr = stack.pop()
            res[curr] = i-curr
        stack.append(i)

    return res

# eval polish N
def evalReversePolishN(self, tokens:List[str]) -> int:
    stack = []
    for t in tokens:
        if t not in "+-*/":
            stack.appending(int(t))
        else:
            r,l = stack.pop(), stack.pop()
            if t == "+":
                stack.append(l+r)
            #...
            else:
                stack.append(int(float(l)/r))
    return stack.pop()

# my stack using queue
class MyStack:
    def __init__(self) -> None:
        self._queue = collections.deque()
    
    def push(self, x):
        q = self._queue
        q.append(x)
        for _ in range(len(q)-1):
            q.append(q.popleft())
    def pop(self):
        return self._queue.popleft()
    def top(self):
        return self._queue[0]

# decoding string
def decodeString(self, s:str) -> str:
    st = []
    currNumber = 0
    currString = ''
    
    for x in s:
        if x == '[':
            st.append(currString)
            st.append(currNumber)
            currNumber = 0
            currString = ''
        elif x == ']':
            number = st.pop()
            prevString = st.pop()
            currString = prevString + number * currString
        elif x.isdigit():
            currNumber = currNumber * 10 + int(x)
        else:
            currString = x
    return currString

    # foold fill old color with new color
    def floddFill(self, image:List[List[int]], sr:int, sc:int, newColor:int) -> List[List[int]]:

        def dfs(i,j):
            image[i][j] = newColor
            for x,y in ((i-1,j),(i+1,j),(i,j-1),(i,j+1)):
                if 0 <= x < m and 0 <= y < n and image[x][y] == old:
                    dfs(x,y)

        old, m, n = image[sr][sc], len(image), len(image[0])
        if old != newColor:
            dfs(sr,sc)
        return image

# nearest zero in matrix / DP way
def nearestZeroInMatrix(self, mat:List[List[int]]) -> List[List[int]]:
    
    m, n = len(mat), len(mat[0])

    for r in range(m):
        for c in range(n):
            if mat[r][c] > 0:
                top = mat[r-1][c] if r > 0 else inf
                left = mat[r][c-1] if c > 0 else inf
                mat[r][c] = min(top, left)+1

        for r in range(m - 1, -1, -1):
            for c in range(n - 1, -1, -1):
                if mat[r][c] > 0:
                    bottom = mat[r + 1][c] if r < m - 1 else inf
                    right = mat[r][c + 1] if c < n - 1 else inf
                    mat[r][c] = min(mat[r][c], bottom + 1, right + 1)
        return mat

# divide and conquer / merge sort
def merge_sort(nums):
    if len(nums) <= 1:
        return nums
    
    pivot = int(len(nums) //2)
    left = merge_sort(nums[0:pivot])
    right = merge_sort(nums[pivot:])
    return merge(left, right)

def merge(left, right):
    left_cursor = right_cursor = 0
    ret = []

    while left_cursor < len(left) and right_cursor < len(right):
        if left[left_cursor] < right[right_cursor]:
            ret.append(left[left_cursor])
            left_cursor += 1
        else:
            ret.append(right[right_cursor])
            right_cursor += 1
    ret.extend(left[left_cursor:])
    ret.extend(right[right_cursor:])
    return ret

# Validate BST
def isValidBST(self, root, left = float('-inf'), right = float('inf')):
    return not root or left < root.val < right and \
        self.isValidBST(root.left, left, root.val) and \
            self.isValidBST(root.right, root.val, right)

# search target in sorted matrix
def searchMatrix(self, mat:List[List[int]], target:int) -> bool:
    
    row = len(mat)
    col = len(mat[0])
        
    rowidx = 0
    colidx = col-1

    while rowidx <row and colidx >=0:
        ele = mat[rowidx][colidx]
        if ele == target: 
            return True
        elif ele<target: 
            rowidx+=1
        else: colidx -= 1

    return False

# N queens in board
def totalNQueens(self, N: int) -> int:

        # Chess board initialization
        board = [ ['.' for _ in range(N)] for _ in range(N) ]
        
        # occupy flag for each column
        colSet = set()
        
        # occupy flag for each primary diagonal (i.e., Northwest <-> Southeast direction )
        priDiagSet = set()
        
        # occupy flag for each secondary diagonal (i.e., Northeast <-> Southwest direction )
        secDiagSet = set()
        
        def placeQueen( row ):
            
            # Base case aka stop condition
            if row == N:
                return 1
            
            # Try all possible columns in DFS + backtracking
            goodPlacementCounter = 0
            
            for col in range(N):
                
                if isSafe(col, row):
                    
                    update( row, col, True)                             # make a selection
                    goodPlacementCounter += placeQueen( row+1 )         # solve next row in DFS 
                    update( row, col, False)                            # undo selection
                    
            return goodPlacementCounter
        
        def isSafe( col, row):
            
            # Check no other Queens on the same column, primary diagonal, and secondary diagonal
            return (col not in colSet) and ((col-row) not in priDiagSet) and ((col+row) not in secDiagSet)
        
        def update( row, col, putOn ):
            
            if putOn == True:
                # put Queen on specified position, and set corresponding occupy flag
                
                board[row][col] = "Q"
                colSet.add( col )
                priDiagSet.add( col - row )
                secDiagSet.add( col + row )
                
            else:
                # take Queen away from specified position, and clear corresponding occupy flag
                
                board[row][col] = "."
                colSet.remove( col )
                priDiagSet.remove( col - row )
                secDiagSet.remove( col + row )
        
        return placeQueen( row = 0 )
        
# robot clean house
def cleanRoom(self, robot):
    path = set()
    def dfs (x,y,dx,dy):
        robot.clean()
        path.add((x,y))

        # cleannext
        for _ in range(4):
            if (x+dx, y+dy) not in path and robot.move():
                dfs(x+dx, y+dy,dx,dy)
            robot.turnleft()
            dx,dy = -dy,dx 

        robot.turnleft();robot.turnLeft()
        robot.move()
        robot.turnLeft();robot.turnLeft()
    dfs(0,0,0,1)
    #you turn left twice and move in order to get back at your previous position. 
    # and you turn left twice again in order to get back at your previous direction. 
    # So, when you will turn left again, you are facing a direction you've never faced.

def solveSudoku(self, board:List[List[int]]) -> None:
    return backtrack(board, 0,0)

def backtrack(self, board:List[List[str]], r:int, c:int) -> bool:
    while board[r][c] != '.':
        c += 1
        if c == 9 :
            c,r = 0, r+1
        if r == 0 :
            return True
    
    for k in range(1,10):
        if self.isValidSudoku(board, r,c,str(k)):
            board[r][c] = str(k)
            if self.backtrack(board, r,c):
                return True
    board[r][c] = '.'
    return False

def isValidSudoku(self, board:List[List[str]], r:int, c:int, cand:int) -> bool:
    if any(board[r][c] == cand for j in range(9)):
        return False
    if any(board[i][c] == cand for i in range(9)):
        return False

    br,bc = 3*(r//3), 3*(c//3)
    if any(board[i][j] == cand for i in range(br,br+3) for j in range(bc,bc+3)):
        return False
    return True

# Given two integers n and k, return all possible combinations of k numbers chosen from the range [1, n].
def combine(self, n:int, k:int) -> List[List[int]]:
    res = []
    def backtrack(remain, comb, nex):
        if remain == 0:
            res.append(comb.copy())
        else:
            for i in range(nex, n+1):
                comb.append(i)
                backtrack(remain-1, comb, i+1)
                comb.pop()

    backtrack(k, [],1)
    return res

def isSameTree (self, p:Optional[TreeNode], q;Optional[TreeNode]) -> bool:
    if not p and not q:
        return True

    if not q or not p:
        return False
    
    if p.val != q.val:
        return False
    return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

def generateParenthesis(self, n:int) -> List[str]:
    res = []
    def dfs(left, right, string):
        if len(string) == n*2:
            res.append(string)
            return
        if left < n:
            dfs(left+1, right, ''+'(')
        if right < left:
            dfs(left, right+1, string+')')
    dfs(0,0,'')
    return res

# tree to doubly list
def treeToDoublyList(self, root:'Optional[Node]') -> 'Optional[Node]':
    if not root:
        return
    
    dummy, Node(0,None,None)
    prev, stack, node = dummy,[], root

    while stack or node:
        while node:
            stack.append(node)
            node = node.left
        
        node = stack.pop()
        node.left = prev
        node.right = node

        prev = node
        node = node.rigth

    dummy.right.left, prev.right = prev, dummy.right
    return dummy.right

def largestRectangleArea(self, heights:List[int]) -> int:
    heights.append(0)
    stack = [-1]
    ans = 0

    for i in range(len(heights)):
        while heights[i] < heights[stack[-1]]:
            h = heights[stack.pop()]
            w = i - stack[-1] -1 # i-1 represents the right boundary of the considered rectangle and stack[-1] represents the left boundary
            ans = max(ans, h*w)
        stack.append(i)
    heights.pop()
    return ans

# 
def permute(self, nums:List[int])-> List[List[int]]:
    if not nums:
        return nums
    
    res = []
    self.dfs(nums, [], res)
    return res
def dfs(self, nums, path, res):
    if not nums:
        res.append(path)
    for i in range(len(nums)):
        self.dfs(nums[:i]+nums[i+1:], path+[nums[i]], res)

# unique permu
def permuteUnique(self, nums):
    res = []
    nums.sort()
    self.permUni(nums, [], res)
    return res

def permUni(self, nums, path,res):
    if len(nums) == 0:
        res.append(path)

    for i in range(len(nums)):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        self.permUni(nums[:i]+nums[i+1:], path+[nums[i]],res)


# phone num letter combination
def letterCombinations(self, digits):
    if not digits:
        return []
    m = {"2":"abc", '3':"def", '4':"ghi", '5':"jkl", '6':"mno", '7':"pqrs", '8':"tuv", '9':"wxyz"}
    ret = []
    self.dfs(m, digits, "", ret)
    return ret
    
def dfs(self, m, digits, path, ret):
    if not digits:
        ret.append(path)
        return 
    for c in m[digits[0]]:
        self.dfs(m, digits[1:], path+c, ret)

# all subsets with dfs way
def subsets(self,nums):
    res = []
    self.helper(nums, [], res)
    return res

def helper(self, nums, path, res):
    res.append(path)
    for i in range(len(nums)):
        self.helper(nums[i+1:], path+[nums[:i]],res)

#
def subsetsWithDuplicate(self, nums):
    res = []
    nums.sort()
    self.helper(nums, [], res)
    return res

def helper(self, nums, path, res):
    res.append(path)
    for i in range(len(nums)):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        self.helper(nums[i+1:], path+[nums[i]], res)

#
def combine(self, n, k):
    res = []
    self.dfs(range(1,n+1), k,[],res)
    return res

def dfs(self, nums, k, path, res):
    if len(path) == k:
        res.append(path)
        return
    for i in range(len(nums)):
        self.dfs(nums[i+1:], k, path+[nums[i]],res)

#
def combinationSum(self, candidates, target):
    res = []
    candidates.sort()
    self.dfs(candidates, target, [], res)
    return res

def dfs(self, candidates, target, path, res):
    if target < 0:
        return #backtracking
    if target == 0:
        res.append(path)
        return
    for i in range(len(candidates)):
        self.dfs(candidates[i:], target-candidates[i],path+[candidates[i]],res)

# Implement TrieNode
class TrieNode:
    def __init__(self) -> None:
        self.word = False
        self.children = {}

class Trie:
    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for i in word:
            if i not in node.children:
                node.children[i] = TrieNode()
            node = node.children[i]
        node.word = True

    def search(self, word):
        node = self.root
        for i in word:
            if i not in node.children:
                return False
            node = node.children[i]
        return node.word
    
    def startsWith(self, prefix):
        node = self.root
        for i in prefix:
            if i not in node.children:
                return False
            node = node.children[i]
        return True

# map to sum
class TrieNode:
    def __init__(self) -> None:
        self.child = collections.defaultdict(TrieNode)
        self.sum = 0

class MapSum:
    def __init__(self):    
        self.trieRoot = TrieNode()
        self.map = collections.defaultdict(int)

    def insert(self, key:str, val:int) -> None:
        diff = val - self.map[key]
        curr = self.trieRoot
        for c in key:
            curr = curr.child[c]
            curr.sum += diff
        self.map[key] = val

    def sum(self, prefix:str) -> int:
        curr = self.trieRoot
        for c in prefix:
            if c not in curr.child:
                return 0
            curr = curr.child[c]
        return curr.sum

# replace all the successors in the sentence with the root forming it.
def replaceWords(self, dictionary:List[str], senetence:str) -> str:
    rootset = set(dictionary)

    def rep(word):
        for i in range(1, len(word)):
            if word[:i] in rootset:
                return word[:i]
        return word
    return " ".join(map(rep, sentence.split()))

def replaceWords(self, dict, sentence):
        
        setenceAsList = sentence.split(" ")
        for i in range(len(setenceAsList)):
            for j in dict:
                if setenceAsList[i].startswith(j):
                    setenceAsList[i] = j
        return " ".join(setenceAsList)

# word search, you already have Trie, addWord
def search(self, word):
    node = self.root
    self.res = False
    self.dfs(node, word)
    return self.res

def dfs(self, node, word):
    if not word:
        if node.isWord:
            self.res = True
        return
    if word[0] == ".":
        for n in node.children.values():
            self.dfs(n, word[1:])
    else:
        node = node.children.get(word[0])
        if not node:
            return
        self.dfs(node, word[1:])

# word search
def findWords(self, board, words):
    res = []
    trie = Trie()
    node = trie.root
    for w in words:
        trie.insert(w)
    for i in range(len(board)):
        for j in rnage(len(board[0])):
            self.dfs(board, node, i,j,"",res)
    return res

def dfs(self,board, node, i, j, path,res):
    if node.isWord:
        res.append(path)
        node.isWord = False
    
    if i<0 or i>= len(board) or j<0 or j>= len(board[0]):
        return
    
    tmp = board[i][j]
    node = node.children.get(tmp)
    if not node:
        return

    board[i][j] = "#"
    self.dfs(board, node, i+1, j, path+tmp, res)
    self.dfs(board, node, i-1, j, path+tmp, res)
    self.dfs(board, node, i, j-1, path+tmp, res)
    self.dfs(board, node, i, j+1, path+tmp, res)
    board[i][j] = tmp

# palindrome pair in words array
def isPalindrome(self, word:str, i:int, j:int) -> bool:
    while i<j:
        if word[i] != word[j]:
            return False
        i += 1
        j -= 1
    return True

def palindromePairs(self, words:List[str]) -> List[List[int]]:
    wmap, ans = {}, []
    for i in range(len(words)):
        wmap[words[i]] = i
    
    for i in range(len(words)):
        if words[i] == "":
            for j in range(len(words)):
                w = words[j]
                if self.isPalindrome(w, 0, len(w)-1) and j != i:
                    ans.append([i,j])
                    ans.append([j,i])
            continue
        bw = words[i][::-1]
        if bw in wmap:
            res = wmap[bw]
            if res != i:
                ans.append([i,res])
        for j in range(1, len(bw)):
            if self.isPalindrome(bw, 0,j-1) and bw[j:] in wmap:
                ans.append([i,wmap[bw[j:]]])
            if self.ispalindrome(bw,j,len(bw)-1) and bw[:j] in wmap:
                ans.append([wmap[bw[:j]],i])
    return ans
