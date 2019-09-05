import torch
import numpy as np
#h = torch.randn(1, 2, 1, 5, requires_grad=True); print(h)
#h = h.view(2,5).permute(1,0); print(h) # [5,2]
#val,idx = h.max(1, keepdim=True)
#print(val)
#print(idx)
#print("argmax:", torch.argmax(h, dim=1))

#print(h[:,:,:)

#first_row = h[:,0,:]; print(first_row)
#second_row = h[:,1,:]; print(second_row)
#result = torch.ge(second_row, first_row); print(result) # greater or equal
#slice_sub = second_row - first_row; print(slice_sub)


#z_1 = torch.nn.Softmax(dim=1)(h*100); print(z_1)
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
        smax = self.softmax(x_sparse); print(torch.bernoulli(smax))
        end_index = self.base_index + x_sparse.size()[0] * self.step_size
        indices = torch.arange(start=self.base_index, end=end_index, step=self.step_size, dtype=torch.float32)
        soft_arg_max = torch.matmul(smax.transpose(1,0), indices)

        return soft_arg_max


#h = torch.randn(2, 5, requires_grad=True); print(h)

#print(SoftArgmax1D()(h))
#val,idx = h.max(0, keepdim=True); print(idx)

#h = torch.randn(2,5, requires_grad=True); print(h)
#mask = torch.tensor([0.0,1.0,0.0,0.0,1.0], requires_grad=True); print(mask)
#mask = mask.type(torch.uint8); print(mask)

# h = torch.randn(3,2,4, requires_grad=True); print(h)
# h_softmax = torch.nn.Softmax(dim=1)(h); print(h_softmax)
# h_softmax_slice = h_softmax[:,1,:].view(3,1,4); print(h_softmax_slice)
# mask = torch.bernoulli(h_softmax_slice); print(mask)
# print(h*mask)

# h = torch.randn(3,2,4, requires_grad=True); print(h)
# mask = torch.randn(3,1,4, requires_grad=True); print(mask)
# h_sum = torch.sum(h, dim=1); print(h_sum)
# h_sum = torch.sum(h_sum, dim=1); print(h_sum)
# mask_sum = torch.sum(mask, dim=2); print(mask_sum)
# mask_sum = torch.sum(mask_sum, dim=1); print(mask_sum)
# print(h_sum/mask_sum)

# a = np.array(list(range(0, 24))).astype(np.float32)
# a = torch.tensor(a, requires_grad=True); #print(a)
# a = a.view(3,2,4); #print(a)
# #a_trans = a.permute(0,2,1); print(a_trans)
# a_norm = torch.norm(a, p=2, dim=1, keepdim=True); #print(a_norm)

# a_normalized = a/a_norm; print(a_normalized) # [3,2,4]
# a_normalized_trans = a_normalized.permute(0,2,1); print(a_normalized_trans) # [3,4,2] 

# tem = torch.bmm(a_normalized_trans, a_normalized); print(tem) # [3,4,4]

# div_loss = 0

# mask = torch.tensor([[[1.0,0.0,1.0,1.0]],[[1.0,1.0,0.0,0.0]],[[1.0,0.0,0.0,0.0]]], requires_grad=True); print(mask.shape, mask)
# mask_trans = mask.permute(0,2,1)
# mask_matrix = torch.bmm(mask_trans, mask); print(mask_matrix) # [3,4,4]

# print(tem*mask_matrix)


# score = torch.tensor([[0.3, 0.5, 0.7, 1.0, 0.8],[0.1, 0.2, 0.3, 0.4, 0.5]], requires_grad=True)
# med_score,_ = torch.median(score, dim=1, keepdim=True)

# print(med_score)
# print(score-med_score)
# batch_variance = torch.mean((score-med_score)**2, dim=1)
# print(batch_variance)
# print(1.0/batch_variance)
#print(torch.mean((score-med_score)**2))


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv11 = nn.Conv2d(3, 64, 3, padding = 1 )
        self.pool1 = nn.AvgPool2d(2, 2)

        self.conv21 = nn.Conv2d(64, 64*2, 3, padding = 1 )
        self.pool2 = nn.AvgPool2d(2, 2)
        
        self.conv52 = nn.Conv2d(64*2, 10, 1)
        self.pool5 = nn.AvgPool2d(8, 8)
        
    def forward(self, x):
        
        x = F.relu(self.conv11(x))
        x = self.pool1(x)

        x = F.relu(self.conv21(x))
        x = self.pool2(x)
        
        x = self.conv52(x)
        x = self.pool5(x)
        
        x = x.view(-1, 10)
        return x
    

net = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)
inputs = torch.rand(4,3,32,32)
labels = torch.rand(4)*10//5
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
inputs = inputs.to(device)
labels = labels.to(device)

outputs = net(inputs)

loss = criterion(outputs, labels.long() )

loss.backward()
print("conv11", net.conv11.weight.grad)
print("conv21", net.conv21.weight.grad)
print("conv52", net.conv52.weight.grad)


optimizer.step()



