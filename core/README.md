tensorflow 核心

1:over_fitting 模型在test data上面比在training data上表现的要糟糕
    解决方法：
    
    1：调整模型参数，参数越多的模型往往更容易over_fitting,如果是解决图片问题，
        尽可能选择conv卷积层，能很大程度减少参数量       
    2:增加training data的数量    
    3:增加training data的多样性，比如数据增强 data augment    
    4:添加weight regularization (L1,L2)  
    5:添加dropout(现在用的少了，更流行使用batch normalization)    
    6:batch normalization
    7:使用early_stop可以防止over_fitting

    
2: regularizer (参照：regularizer_test) 其实就是往模型添加loss，然后根据反向传播原理，使用梯度下降方法更新相应的trainable_variables,
从而达到惩罚训练参数目的:

    L1：regularization += self.l1 * math_ops.reduce_sum(math_ops.abs(x))
    L2：regularization += self.l2 * math_ops.reduce_sum(math_ops.square(x))
        