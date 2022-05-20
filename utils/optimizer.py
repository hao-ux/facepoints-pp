import paddle

def create_optimzer(parameters, step_each_epoch, epochs):
    lr = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=1e-3,T_max=step_each_epoch*epochs,verbose=True)
    return paddle.optimizer.Adam(learning_rate=lr,parameters=parameters,weight_decay=5e-4)