from sardine import _sardine as sa
import numpy as np

def addr(x):
    return hex(x.__array_interface__['data'][0])

ctx = sa.host_context()

buffer_nb = 3
frame_nb = 1
frame_size_x = 10
frame_size_y = 10

frame_size = frame_size_x * frame_size_y

total_size = buffer_nb * frame_nb * frame_size_x * frame_size_y * np.float32().itemsize

data: np.ndarray = sa.region.host.open_or_create("trouduc", [buffer_nb, frame_nb, frame_size_x, frame_size_y], dtype=np.float32)

u_of_data = sa.url_of(data)

data2 = sa.from_url(u_of_data, np.ndarray, np.float32)

# me_v = sa.View(data)

# u = me_v.url()
# v = me_v.view()

# me_ov = me_v[1]

# n = np.ones(10, dtype=np.int32)

# u_of_n = sa.url_of(n, allow_local=True)
