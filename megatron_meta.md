### 算子中parameter看起来是随机生成的，为什么模型中打印出来是ckpt的值，这个参数覆盖是什么时候发生的？
1. 模型初始化时， class model 比如 class LayerNorm(Module) __init__的函数就调用了，初始化生成了一份随机参数。
2. 训练开始前，读取ckpt，将对应的参数全部覆盖。
3. 训练时，调用forward，已经是最新的ckpt过的参数了。

### torch hook
https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook

register_forward_pre_hook 前向传播前
The hook will be called every time before forward() is invoked.

register_forward_hook 前向传播后
The hook will be called every time after forward() has computed an output.

register_full_backward_pre_hook 反向传播前
The hook will be called every time the gradients for the module are computed. 

register_full_backward_hook 反向传播后
The hook will be called every time the gradients with respect to a module are computed, i.e. the hook will execute if and only if the gradients with respect to module outputs are computed. 

register_load_state_dict_post_hook
Register a post hook to be run after module’s load_state_dict is called.
