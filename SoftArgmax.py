import torch

class SoftArgmax(torch.nn.Module):
    """
    Implementation of a 1d soft arg-max function as an nn.Module, so that we can differentiate through arg-max operations.
    """
    def __init__(self, base_index=0, step_size=1, beta=100):
        """
        The "arguments" are base_index, base_index+step_size, base_index+2*step_size, ... and so on for
        arguments at indices 0, 1, 2, ....
        Assumes that the input to this layer will be a batch of 1D tensors (so a 2D tensor).
        :param base_index: Remember a base index for 'indices' for the input
        :param step_size: Step size for 'indices' from the input
        :param beta: make the data distribution more sparse
        """
        super(SoftArgmax, self).__init__()
        self.base_index = base_index
        self.step_size = step_size
        self.beta = beta
        self.softmax = torch.nn.Softmax(dim=1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def forward(self, x):
        """
        Compute the forward pass of the 1D soft arg-max function as defined below:
        SoftArgMax(x) = \sum_i (i * softmax(x)_i)
        :param x: The input to the soft arg-max layer [2, n_frames]
        :return: Output of the soft arg-max layer
        """
        x_sparse = x * self.beta
        smax = self.softmax(x_sparse) # [batch_size, feature, n_frames] -> [5,2,320]
        end_index = self.base_index + x_sparse.size()[1] * self.step_size
        indices = torch.arange(start=self.base_index, end=end_index, step=self.step_size, dtype=torch.float32, device=self.device).repeat(x_sparse.size()[0],1,1) # [batch_size, 1, n_choice] 
        soft_arg_max = torch.bmm(indices, smax) # [batch_size, 1, n_frames] -> [5,1,320]

        return soft_arg_max
        

if __name__ == "__main__":
    h = torch.randn(5, 2, 320, requires_grad=True); print(h)

    print(SoftArgmax()(h))
    val,idx = h.max(1, keepdim=True); print(idx)    