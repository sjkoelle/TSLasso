def data_stream(a, b):
    for i in range(a):
        for j in range(b):
            yield (i, j)

