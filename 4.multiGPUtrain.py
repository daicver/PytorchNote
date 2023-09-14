# pytorch多卡训练的原理
# 原理：
# （1）将模型加载到一个指定的主GPU上，然后将模型浅拷贝到其它的从GPU上；
# （2）将总的batch数据等分到不同的GPU上（坑：需要先将数据加载到主GPU上）；
# （3）每个GPU根据自己分配到的数据进行forward计算得到loss，并通过backward得到权重梯度；
# （4）主GPU将所有从GPU得到的梯度进行合并并用于更新模型的参数。
# 浅拷贝：拷贝引用，会改变原始数据，速度快效率高
# 深拷贝：重新定义一个变量，不会改变原始数据，速度慢更可靠

# 数据并行
# 模型设置
device_ids = [0, 1, 2, 3]
model = Model(input_size, output_size)
model = nn.DataParallel(model, device_ids=device_ids) #单卡没有这行代码
model = model.cuda(device_ids[1]) #指定哪块卡为主GPU，默认是0卡
# 数据设置
for data in data_loader:
    input_var = Variable(data.cuda(device_ids[1])) #默认指定用0卡先加载数据就会报错
    output = model(input_var)
    