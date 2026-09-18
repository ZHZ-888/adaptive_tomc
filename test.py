from collections import deque

dic = {
    'mavh70': [['AHH', 'mhv90'], deque([1, 2], maxlen=10)],
    'mavh71': [['BHH', 'mhv91'], deque([3, 4], maxlen=10)],
    'mavh72': [['CHH', 'mhv92'], deque([5, 6], maxlen=10)],
}
id_list = [v[0][1] for v in dic.values()]
print(id_list)