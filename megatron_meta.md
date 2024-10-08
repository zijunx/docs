### 算子中parameter看起来是随机生成的，为什么模型中打印出来是ckpt的值，这个参数覆盖是什么时候发生的？
1. 模型初始化时， class model 比如 class LayerNorm(Module) __init__的函数就调用了，初始化生成了一份随机参数。
2. 训练开始前，读取ckpt，将对应的参数全部覆盖。
3. 训练时，调用forward，已经是最新的ckpt过的参数了。

### 打印模型中算子的输入、输出
torch hook
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

```python
# Function to save gradients and inputs
def save_gradients_and_inputs(name):
    def forward_hook(module, input, output):
        module.saved_input = input
        module.saved_output = output

    def backward_hook(module, grad_input, grad_output):
        if torch.distributed.get_rank() == 0:
            # print(f"grad_input {type(grad_input)}")
            # print(f"grad_output {type(grad_output)}")
            # print(f"input {type(module.saved_input)}")
            print(f"output {type(module.saved_output)}")
            for id,i in enumerate(module.saved_input):
                print(f"input_{id} shape {i.shape} ")
            for id,i in enumerate(grad_input):
                print(f"grad_input_{id} shape {i.shape} ")
            for id,i in enumerate(grad_output):
                print(f"grad_output_{id} shape {i.shape} ")

            # Save the gradients and inputs to npz file
            np.savez(f'/home/export/online1/mdt00/shisuan/swustcai/zzq/wenhai/debug/gradients_and_inputs_{name}.npz',
                    grad_input=grad_input[0].detach().cpu().numpy(),
                    grad_output=grad_output[0].detach().cpu().numpy(),
                    input=module.saved_input[0].detach().cpu().numpy(),
                    output=module.saved_output.detach().cpu().numpy())
            print(f"Grad Input for {name}: {grad_input}")
            print(f"Grad Output for {name}: {grad_output}")
            print(f"Input for {name}: {module.saved_input}")
            print(f"Output for {name}: {module.saved_output}")
    return forward_hook, backward_hook

======================================================================================
        if self.pre_process:
            self.proj = torch.nn.Conv2d(self.in_chans, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
            self.proj_bulk = torch.nn.Conv2d(self.in_bulk_chans, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
            self.down_blk = DownBlock(self.hidden_size, self.hidden_size, num_groups=32)
            self.down_blk2 = DownBlock(self.hidden_size, self.hidden_size, num_groups=32)
            if args.debug_npz:
                forward_hook,back_hook = save_gradients_and_inputs('down_blk2_b4')
                self.down_blk2.b[4].register_forward_hook(forward_hook)
                self.down_blk2.b[4].register_full_backward_hook(back_hook)
        self.div_val=self.patch_size[0] * 2
```
只执行了一个迭代，所以不存在tensor被覆盖的问题。
```python
output <class 'torch.Tensor'>
input_0 shape torch.Size([1, 768, 133, 273])
grad_input_0 shape torch.Size([1, 768, 133, 273])
grad_output_0 shape torch.Size([1, 768, 133, 273])
Grad Input for down_blk2_b4: (tensor([[[[-1.7771e-08,  2.7944e-09, -1.9785e-09,  ...,  5.2107e-08,^M
           -2.3024e-09, -7.5307e-08],^M
          [ 3.0019e-07, -7.5822e-08, -7.9020e-08,  ..., -1.3240e-07,^M
            1.8376e-08,  1.6460e-08],^M
          [-3.2822e-07,  2.9710e-08, -2.9138e-08,  ...,  1.0046e-08,^M
           -3.6316e-09,  1.0978e-08],^M
```
这个回调函数会在模块的前向传播结束后被调用，接收三个参数：模块本身、输入、输出。
The hook should have the following signature:
```python
前向hook
hook(module, args, output) -> None or modified output
反向hook
hook(module, grad_input, grad_output) -> tuple(Tensor) or None
```
https://www.jianshu.com/p/69e57e3526b3
https://blog.csdn.net/m0_51661400/article/details/135091359


![image](https://github.com/user-attachments/assets/49a359b3-e386-4581-ba69-33f58afcc3b8)
class Conv2d(_ConvNd):
class _ConvNd(Module): 
只要最终继承自Module就可以直接注册钩子函数

