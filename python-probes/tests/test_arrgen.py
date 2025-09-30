from arrgen import uniform_array


def test_arrgen():
    shape = [2, 2]
    seed = 42
    min = 0.0
    max = 1.0
    x = uniform_array(seed, shape, min, max)
    print(x)
