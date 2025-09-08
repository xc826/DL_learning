import numpy as np
import matplotlib.pyplot as plt

# def step_function(x):
#     return np.array(x>0,dtype=np.int32)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def identity_function(x):
    return x


# 存在溢出问题
def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c) #防止溢出的对策
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a

    return y

# 三层神经网络的实现
def init_network():
    network={}
    network['w1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1']=np.array([0.1,0.2,0.3])
    network['w2']=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2']=np.array([0.1,0.2])
    network['w3']=np.array([[0.1,0.3],[0.2,0.4]])
    network['b3']=np.array([0.1,0.2])
    return network

def forward(network,x):
    w1,w2,w3=network['w1'],network['w2'],network['w3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']

    a1=np.dot(x,w1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,w2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,w3)+b3
    y=identity_function(a3)

    return y

network=init_network()
x=np.array([1.0,0.5])
y=forward(network,x)
print(y)



















# x = np.arange(-5, 5, 0.1)
# y = sigmoid(x)
# y2 = np.cos(x)

# plt.plot(x, y)
# plt.plot(x, y2, linestyle="--", label="sin")
# plt.xlabel("x")
# plt.xlabel("y")
# plt.title("sin & cos")
# plt.legend()
# plt.ylim(-0.1,1.1)
# plt.show()

# x=np.array([1,2])
# w=np.array([[1,3,5],[2,4,6]])
# y=np.dot(x,w)
# print(y)