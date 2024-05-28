
import numpy as np
import tensor_operations


A = np.array([[[1,2,3],[4,5,8],[7,2,9]],
              [[1,5,3],[4,3,6],[4,8,9]],
              [[1,2,7],[1,5,6],[7,8,6]]])

B = np.array([[[1,0,0],[0,5,0],[0,0,9]],
              [[0,2,0],[4,0,0],[0,0,9]],
              [[0,0,3],[0,0,6],[7,0,0]]])

C=np.real(tensor_operations.t_product(A,B))

C0 = A[:,:,0] @ B[:,:,0] + A[:,:,2] @ B[:,:,1] + A[:,:,1] @ B[:,:,2]
C1 = A[:,:,1] @ B[:,:,0] + A[:,:,0] @ B[:,:,1] + A[:,:,2] @ B[:,:,2]
C2 = A[:,:,2] @ B[:,:,0] + A[:,:,1] @ B[:,:,1] + A[:,:,0] @ B[:,:,2]

print(np.allclose(C[:,:,0],C0))
print(np.allclose(C[:,:,1],C1))
print(np.allclose(C[:,:,2],C2))
