### 算子中parameter看起来是随机生成的，为什么模型中打印出来是ckpt的值，这个参数覆盖是什么时候发生的？
1. 模型初始化时， class model 比如 class LayerNorm(Module) __init__的函数就调用了，初始化生成了一份随机参数。
2. 训练开始前，读取ckpt，将对应的参数全部覆盖。
3. 训练时，调用forward，已经是最新的ckpt过的参数了。

