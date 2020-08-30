"""
Section 3 - Q8

곳감(모래시계)

현수는 곳감을 만들기 위해 감을 깍아 마당에 말리고 있습니다.
현수의 마당은 N*N 격자판으 로 이루어져 있으며, 현수는 각 격자단위로 말리는 감의 수를 정합니다.
그런데 해의 위치에 따라 특정위치의 감은 잘 마르지 않습니다.
그래서 현수는 격자의 행을 기준으로 왼쪽, 또는 오른쪽으로 회전시켜 위치를 변경해 모든 감이 잘 마르게 합니다.
만약 회전명령 정보가 2 0 3이면 2번째 행을 왼쪽으로 3만큼 아래 그림처럼 회전시키는 명령 입니다.
첫 번째 수는 행번호, 두 번째 수는 방향인데 0이면 왼쪽, 1이면 오른쪽이고, 세 번째 수는 회전하는 격자의 수입니다.
M개의 회전명령을 실행하고 난 후 아래와 같이 마당의 모래시계 모양의 영역에는 감 이 총 몇 개가 있는지 출력하는 프로그램을 작성하세요.

입력예제 : in_q8.txt
출력예제 : 362
"""

import sys

sys.stdin = open('in_q8.txt', 'r')
n = int(input())
grid = [list(map(int, input().split(' '))) for _ in range(n)]
#print(grid)
m = int(input())
#print(m)
#mat = [list(map(int, input().split(' '))) for _ in range(m)]

for i in range(m):
    idx, direction, interval = map(int, input().split(' '))

    if direction == 0: # 왼쪽으로 이동할 때
        for j in range(interval): # interval만큼 반복해서 움직일 수 있도록
            grid[idx-1].append(grid[idx-1].pop(0)) # 맨앞 요소를 삭제하고 그것을 맨뒤 요소로 추가
    else:
        for j in range(interval):
            grid[idx-1].insert(0, grid[idx-1].pop()) # 맨뒤 요소를 삭제하고 그것을 맨앞 요소로 추가
#print(grid)

res = 0
s = 0
e = n-1
for i in range(n):
    for j in range(s, e+1):
        res += grid[i][j]
    if i < n//2:
        s += 1
        e -= 1
    else:
        s -= 1
        e += 1

print(res)