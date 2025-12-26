# from kan import *

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# # create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
# model = KAN(width=[4,2,1,1], grid=3, k=3, seed=1, device=device)
# f = lambda x: torch.exp((torch.sin(torch.pi*(x[:,[0]]**2+x[:,[1]]**2))+torch.sin(torch.pi*(x[:,[2]]**2+x[:,[3]]**2)))/2)
# dataset = create_dataset(f, n_var=4, train_num=3000, device=device)

# # train the model
# model.fit(dataset, opt="LBFGS", steps=20, lamb=0.002, lamb_entropy=2.)
# model = model.prune(edge_th=1e-2)
# model.plot()

from kan import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
model = KAN(width=[4,2,1,1], grid=3, k=3, seed=1, device=device)
f = lambda x: torch.exp((torch.sin(torch.pi*(x[:,[0]]**2+x[:,[1]]**2))+torch.sin(torch.pi*(x[:,[2]]**2+x[:,[3]]**2)))/2)
dataset = create_dataset(f, n_var=4, train_num=3000, device=device)

# train the model
model.fit(dataset, opt="LBFGS", steps=20, lamb=0.002, lamb_entropy=2.,save_fig=True,img_folder='KAN_picture')
model = model.prune(edge_th=1e-2)
model.plot(folder="KAN_picture")