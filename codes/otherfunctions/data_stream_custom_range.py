def data_stream_custom_range(selind, b):
    for i in range(len(selind)):
        for j in range(b):
            yield (selind[i], j)