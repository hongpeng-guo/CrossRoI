from copy import deepcopy

def maxHist(row):
    result = []
    top_val = 0
    max_area, left, width, height = 0, 0, 0, 0

    area = 0 
    i = 0
    while (i < len(row)):
        if (len(result) == 0) or (row[result[-1]] <= row[i]):
            result.append(i)
            i += 1
        else:
            top_val = row[result.pop()]
            area = top_val * i

            if (len(result)):
                area = top_val * (i - result[-1] - 1)
            if area < max_area: continue

            width, height = i if area == 0 else int(area/top_val), top_val
            max_area, left = area, i - width

    while (len(result)):
        top_val = row[result.pop()]
        area = top_val * i
        if (len(result)):
            area = top_val * (i - result[-1] - 1)
        if area < max_area: continue

        width, height = i if area == 0 else int(area/top_val), top_val
        max_area, left = area, i - width

    return max_area, left, width, height


def maxRectangle(A):
    Mat = deepcopy(A)
    top = 0
    max_area, left, width, height = maxHist(Mat[0])

    for i in range(1, len(Mat)):
        for j in range(len(Mat[i])):
            if (Mat[i][j]):
                Mat[i][j] += Mat[i - 1][j]
        if max_area >= maxHist(Mat[i])[0]: continue
        max_area, left, width, height = maxHist(Mat[i])
        top = i - height + 1

    return max_area, top, left, height, width


def mergeTiles(A):
    result = []

    max_area, top, left, height, width = maxRectangle(A)
    while (max_area > 0):
        result.append((top, left, height, width))
        for i in range(top, top + height):
            for j in range(left, left + width):
                A[i][j] = 0 
        max_area, top, left, height, width = maxRectangle(A)

    return result


# Driver Code
if __name__ == '__main__':
	A = [[0, 1, 1, 0],
		[1, 1, 1, 1],
		[1, 1, 1, 1],
		[1, 1, 0, 0]]

	print(mergeTiles(A))
