import numpy as np
from utils import euclidean_distance, f1_score

p1_array = [0, 1, 2, 0, 1, 2]
p2_array = [0, 2, 1, 0, 0, 1]


def test_arange():
    a = np.arange(4).reshape(2, 2)
    print(a)
    print(np.diag(a))
    print(np.einsum('i...i', a))
    print(euclidean_distance(p1_array, p2_array))
    print(f1_score(p1_array, p2_array))
    # p1 = np.array(p1_array, dtype=np.float64)
    # print(np.einsum('ij,ij->i', a, a))
    # print((a ** 2).sum())


if __name__ == "__main__":
    test_arange()
