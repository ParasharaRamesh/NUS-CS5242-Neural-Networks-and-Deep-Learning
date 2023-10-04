import torch

class G:
    def __init__(self):
        self.g=torch.ones([2])
        # g = [1,1]
    def __call__(self,x):
        self.g[0]=x*x #g = [x^2, 1]
        # return return torch.sum(self.g) #changed
        return torch.sum(self.g) # x^2 + g[1]

if __name__ == '__main__':
    g = G()
    xinit = torch.ones([1], requires_grad=True) #xinit -> tensor(1)
    x = [xinit] # x -> [tensor(1)]
    last_elem_x = x[-1]
    x.append(last_elem_x + g(last_elem_x)) #x now has [tensor(1), tensor(3)]; tensor(3) because 1(last_elem) + 1^2 + 1 (g[1))
    loss = x[-1] ** 2 # loss = tensor([9.], grad_fn=<PowBackward0>)
    # loss.backward() #(changed to below)
    loss.backward(retain_graph = True)
    print('------------ first time --------- ')
    x = [xinit]
    x.append(x[-1] + g(x[-1])) # x now has [tensor(1), tensor(3)]; tensor(3) because 1(last_elem) + 1^2 + 1 (g[1))
    loss = x[-1] ** 2
    loss.backward(retain_graph = True)
    print('------------ second time --------- ')

'''
Explanation:
 
 The comment at the end of the line shows the values
 
 The error usually occurs when we attempt to perform a backward pass through the computation graph more than once 
 without explicitly specifying that you want to retain the graph.
 
 In PyTorch's automatic differentiation system, when you call the .backward() method to compute gradients, 
 PyTorch builds a computation graph to keep track of the operations that were performed on tensors. 
 Once the backward pass is complete, PyTorch releases the intermediate values (tensors) and frees up memory to optimize memory usage.

If you try to perform a second backward pass without specifying retain_graph=True, 
PyTorch assumes you no longer need the intermediate values and releases them. 

This is often the desired behavior because it conserves memory. 

However, if you need to compute gradients multiple times or access the intermediate values after the first backward pass, 
you should set retain_graph=True when calling .backward(). { see lines 21 & 25}


'''