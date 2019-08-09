import torch
#h = torch.randn(1, 2, 5, requires_grad=True); print(h)
#val,idx = h.max(1, keepdim=True)
#print(val)
#print(idx)
#print("argmax:", torch.argmax(h, dim=1))

#print(h[:,:,:)

#first_row = h[:,0,:]; print(first_row)
#second_row = h[:,1,:]; print(second_row)
#result = torch.ge(second_row, first_row); print(result) # greater or equal
#slice_sub = second_row - first_row; print(slice_sub)


#z = torch.nn.Softmax(dim=1)(h); print(z)
#z = torch.nn.functional.gumbel_softmax(h, tau=0.1, hard=False, dim=1); print(z)
#z = torch.nn.functional.gumbel_softmax(h, tau=0.1, hard=True, dim=1); print(z)

#print(torch.round(h))
#print(h[:,:,0:2])

class SoftArgmax1D(torch.nn.Module):
    """
    Implementation of a 1d soft arg-max function as an nn.Module, so that we can differentiate through arg-max operations.
    """
    def __init__(self, base_index=0, step_size=1, beta=2):
        """
        The "arguments" are base_index, base_index+step_size, base_index+2*step_size, ... and so on for
        arguments at indices 0, 1, 2, ....
        Assumes that the input to this layer will be a batch of 1D tensors (so a 2D tensor).
        :param base_index: Remember a base index for 'indices' for the input
        :param step_size: Step size for 'indices' from the input
        :param beta: make the data distribution more sparse
        """
        super(SoftArgmax1D, self).__init__()
        self.base_index = base_index
        self.step_size = step_size
        self.beta = beta
        self.softmax = torch.nn.Softmax(dim=0)


    def forward(self, x):
        """
        Compute the forward pass of the 1D soft arg-max function as defined below:
        SoftArgMax(x) = \sum_i (i * softmax(x)_i)
        :param x: The input to the soft arg-max layer [2, n_frames]
        :return: Output of the soft arg-max layer
        """
        x_sparse = x * self.beta
        smax = self.softmax(x_sparse)
        end_index = self.base_index + x_sparse.size()[0] * self.step_size
        indices = torch.arange(start=self.base_index, end=end_index, step=self.step_size, dtype=torch.float32)
        soft_arg_max = torch.matmul(smax.transpose(1,0), indices)

        return soft_arg_max


h = torch.randn(2, 5, requires_grad=True); print(h)
print(SoftArgmax1D()(h))
val,idx = h.max(0, keepdim=True); print(idx)
