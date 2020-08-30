"""
Section 3 - Q6

격자판 최대합

5*5 격자판에 아래와 같이 숫자가 적혀있습니다.

5
10 13 10 12 15
12 39 30 23 11
11 25 50 53 15
19 27 29 37 27
19 13 30 13 19

N*N의 격자판이 주어지면 각 행의 합, 각 열의 합, 두 대각선의 합 중 가 장 큰 합을 출력합니다.

입력예제 : in_q6.txt
출력예제 : 155
"""

import sys
sys.stdin = open('in_q6.txt', 'r')
n = int(input())
matrix=[list(map(int, input().split())) for _ in range(n)]

maximum = 0

for i in range(n):
    row_total = sum(matrix[i])
    #print(row_total)
    if row_total > maximum:
        maximum = row_total

for i in range(n):
    col_total = 0
    for j in range(n):
        col_total += matrix[j][i]
    #print(col_total)
    if col_total > maximum:
        maximum = col_total

diag_total_1 = 0
diag_total_2 = 0
for i in range(n):
    for j in range(n):
        if i == j:
            diag_total_1 += matrix[i][j]
            #print(matrix[i][j])
        if i+j+1 == n:
            diag_total_2 += matrix[i][j]
            #print(matrix[i][j])
#print(diag_total_1)
#print(diag_total_2)

if diag_total_1 > maximum:
    maximum = diag_total_1
if diag_total_2 > maximum:
    maximum = diag_total_2

print(maximum)