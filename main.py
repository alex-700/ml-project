from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
import torch.utils.data

from nets import SomeConvNetMnist, SimpleNet, SomeConvNetCifar, LeNet

criterion = F.cross_entropy
# criterion = F.nll_loss

def train(args, model, device, train_loader, optimizer, epoch, result, test_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            result.append(loss.item())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def calc_test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        print(len(test_loader.dataset))
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            print(data.size())

            output = model(data)
            test_loss += criterion(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    return test_loss, correct


def test(args, model, device, test_loader):
    print('start')
    test_loss, correct = calc_test(args, model, device, test_loader)
    print('finish')
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dataset', choices=[
        'mnist',
        'fashion-mnist',
        'cifar10',
        'smth'
    ], default='mnist', help='dataset for training and calc accuracy')
    parser.add_argument('--augmented', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_data, test_data = get_train_test_dataset(args.dataset, args.augmented)
    # import matplotlib.pyplot as plt
    # plt.imshow(train_data.train_data[0])
    # plt.show()

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                               shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size,
                                              shuffle=True, **kwargs)

    # model = SimpleNet(32 * 32 * 3).to(device)
    model = LeNet().to(device)
    # model = SomeConvNetCifar().to(device)
    # model = SomeConvNetMnist().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    losses = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, losses, test_loader)
        test(args, model, device, test_loader)

    import numpy as np
    losses = np.array(losses)
    # import matplotlib.pyplot as plt
    # plt.plot(losses[:, 0])
    # plt.plot(losses)
    # plt.show()


if __name__ == '__main__':
    main()
