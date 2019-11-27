import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
# x data (tensor), shape=(100, 1)
# unsqueeze 把1維的數據變成2維的數據，因為torch只會處理2維的數據
y = x.pow(2) + 0.2*torch.rand(x.size())
# noisy噪點 y data (tensor), shape=(100, 1)
# 2次方加上一點噪點的影響

x, y = Variable(x), Variable(y)
# 把x和y都變成Variable的形式，因為神經網路只能輸入Variable

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

# 開始定義神經網路Neural Network(Net)
# 要繼承從torch來的模塊(Module)
# 很多Neural Network的功能都包含在這個模塊之中
# 每一個torch都會包含這兩個功能(__init__, forward)
# __init__：搭建層所需要的訊息
# forward：Neural Network正向傳遞的過程
# 把__init__裡面的層訊息一個一個的組合起來放到forward裡，也就是在torch中建立Neural Network的流程


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        # super(繼承關係)，繼承Net到這個模塊，然後把它的功能init也輸出一下(官方步驟)
        # 自己的程式碼片段開始
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        # 一層隱藏層(hidden)的神經網路，它包含(他有多少個輸入, 輸出)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        # x輸入經過一個hidden layer之後被加工然後輸出n_hidden(n個hidden unit神經元的個數)
        # 再用激勵函數(activation)來激活一下，限制住隱藏成輸出的訊息
        x = self.predict(x)
        # 預測層不用activation，因為大多數回歸數的問題，都是從正無窮大到負無窮大
        # 用了activation，輸出的結果會有部分被截斷
        return x


net = Net(1, 10, 1)  # 一個輸入，10個隱藏層，一個輸出
print(net)

plt.ion()  # something ablot plotting，要設置matplotlib成為一個實實在在打印的過程

# 優化神經網路，利用torch裡面的子模塊(optim)中的優化器(optimizer)中常用的SGD，來優化神經網路的參數
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)  # lr(learning rate)
# learning rate越高學得越快，但學得太快不好，就像是會忽略很多個知識點，一般是小於1
loss_func = torch.nn.MSELoss()  # 計算誤差的方式利用，MSELoss(均方差來處理回歸問題的誤差)

# 開始訓練(訓練100步)
for t in range(100):
    prediction = net(x)  # 來看每一步的prediction，把net放入輸入信息x
    loss = loss_func(prediction, y)  # 計算輸入訊出來的預測值跟真實訊息有多少誤差

    optimizer.zero_grad()
    # clear gradients for next train 先把梯度降為零
    # 因為每次計算loss以後，更新之後所有的梯度都會保留在optimizer也就是net之中
    loss.backward()
    # backpropagation, compute gradients
    # 開始反向傳遞loss(Variable的型式)給每一個神經網路的節點，附上計算錯誤(梯度gradients)
    optimizer.step()        # apply gradients 利用optimizer以學習效率0.5來優化梯度

    if t % 5 == 0:
        # 每學習5步就打印一次
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())  # 原始數據
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)  # 預測值
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(),
                 fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
