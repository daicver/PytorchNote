optimizer = optim.SGD(param=net.parameters(),lr=1)      # 将需要优化的参数装进优化器
optimizer.zero_grad()                                   # 梯度清零，等价于net.zero_grad()
input = torch.randn(1,3,32,32)
output = net(input)
loss = loss_fun(output, label)
loss.backward(loss)
optimizer.step()

# 设置不同的学习率
optimizer1 =optim.SGD([
                {'params': net.features.parameters()},
                {'params': net.classifier.parameters(), 'lr': old_lr*0.1}
            ], lr=1e-5)
