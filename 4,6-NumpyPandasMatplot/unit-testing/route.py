import numpy as np 
X = np.load('public_tests/04_test_task4_input/input_0/image.npy')
for el in X: 
    print(f"{el}")
print()
Y = np.load('public_tests/04_test_task4_input/input_0/weights.npy')
for el in Y:
    print(f"{el}")


import pickle
with open('public_tests/04_test_task4_gt/output_0.pkl', 'rb') as f :
    res = pickle.load(f)
    print(res)
