import torch

if __name__ == '__main__':
  test_params = torch.load('MNIST_model.pth')
  for k, v in test_params.items():
    print(f"Avg. weight for {k}: {v.mean()}")
    print(f"Std. dev. for {k}: {v.std()}")
    print(f"Max weight for {k}: {v.max()}")
    print(f"Min weight for {k}: {v.min()}")
    print("")
