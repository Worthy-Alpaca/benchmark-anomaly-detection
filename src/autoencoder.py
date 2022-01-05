from platform import release
import torch
from torch.nn.modules import linear
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

""" Starting Code for further research """
#Transforms images to a PyTorch Tensor
tensor_transform = transforms.ToTensor()

#Download the MNIST Dataset
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=tensor_transform)

#Dataloader is used to load the dataset
loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)

class AE(torch.nn.Module):
	def __init__(self):
		super().__init__()

		self.encoder = torch.nn.Sequential(
			torch.nn.Linear(28 * 28, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 36),
			torch.nn.ReLU(),
			torch.nn.Linear(36, 18),
			torch.nn.ReLU(),
			torch.nn.Linear(18, 9),
		)

		self.decoder = torch.nn.Sequential(
			torch.nn.Linear(9, 18),
			torch.nn.ReLU(),
			torch.nn.Linear(18, 36),
			torch.nn.ReLU(),
			torch.nn.Linear(36, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 28 * 28),
			torch.nn.Sigmoid()
		)
	
	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded

model = AE()

loss_function = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-8)

epochs = 20
outputs = []
losses = []
for epoch in range(epochs):
	for (image, _) in loader:

		image = image.reshape(-1, 28*28)
		
		reconstructed = model(image)

		loss = loss_function(reconstructed, image)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		losses.append(loss)
	outputs.append((epochs, image, reconstructed))

plt.style.use('fivethirtyeight')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.plot(losses[-100:])