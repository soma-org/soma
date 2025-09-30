from arrgen import generate_array


def test_arrgen():
    shape = [2, 2]
    seed = 42
    min = 0.0
    max = 1.0
    x = generate_array(shape, seed, min, max)
    print(x)
