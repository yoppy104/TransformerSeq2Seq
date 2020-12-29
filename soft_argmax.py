"""
creater : Shiyu-Mou-Bose
url : https://github.com/Fdevmsy/PyTorch-Soft-Argmax/blob/master/soft-argmax.py

This comment is appended by Yoshino Toshiaki.
"""


import torch
import torch.nn as nn

def soft_argmax(voxels, device):
	"""
	Arguments: voxel patch in shape (batch_size, channel, H, W, depth)
	Return: 3D coordinates in shape (batch_size, channel, 3)
	"""
	assert voxels.dim()==5
	# alpha is here to make the largest element really big, so it
	# would become very close to 1 after softmax
	alpha = 10000000.0
	N,C,H,W,D = voxels.shape
	soft_max = nn.functional.softmax(voxels.view(N,C,-1)*alpha,dim=2)
	soft_max = soft_max.view(voxels.shape)
	indices_kernel = torch.arange(start=0,end=H*W*D).unsqueeze(0)
	indices_kernel = indices_kernel.view((H,W,D))
	conv = soft_max*indices_kernel.to(device)
	indices = conv.sum(2).sum(2).sum(2)
	z = indices%D
	y = (indices/D).floor()%W
	x = (((indices/D).floor())/W).floor()%H
	coords = torch.stack([x,y,z],dim=2).type(torch.long)
	return coords

if __name__ == "__main__":
	acc = 0
	for i in range(1000):
		voxel = torch.randn(10000,6,1,16,1) # (batch_size, channel, H, W, depth)
		coords = soft_argmax(voxel, torch.device("cpu"))
		acc += torch.equal(coords[:, :, 1].squeeze(), torch.argmax(voxel, dim=3).squeeze())
	print(acc)