import numpy as np
import theano
import theano.tensor as T

# Declare Theano symbolic variables
x = T.dmatrix("x")
w = T.dmatrix("w")
b = T.dvector("b")

fn = T.dot(x, w) + b

# Compile
train = theano.function(
    inputs=[x,w,b],
    outputs=fn)

res = train([[0, 1], [-1, -2]], [[0, 1], [-1, -2]], [1,1])

print("results:")
print(res)

fn1 = T.mean(w,axis=0)
cfn1 = theano.function(
    inputs=[w],
    outputs=fn1)
res = cfn1([[0, 1], [2, 3]])


# print("results:")
# print(res)

print np.ones((3,))