import numpy as np
import tensor_operations


A = np.array([[[1,2,3],[4,5,8],[7,2,9]],
              [[1,5,3],[4,3,6],[4,8,9]],
              [[1,2,7],[1,5,6],[7,8,6]]])

A_transpose = np.int32(np.real(tensor_operations.t_transpose(A)))

print(A[:,:,0].T)
print()

print(A_transpose[:,:,0])
print()

print(A[:,:,2].T)
print()

print(A_transpose[:,:,1])