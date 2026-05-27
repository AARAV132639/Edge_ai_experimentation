from torchvision.datasets import CIFAR10

def download_dataset():
    train_data= CIFAR10(
        root= "./datasets",
        train= True,
        download= True
    )

    test_data= CIFAR10(
        root="./datasets",
        train= False,
        download= True
    )

    return train_data, test_data

if __name__=="__main__":
    train_data, test_data = download_dataset()
    print("Training samples:", len(train_data))
    print("Test samples:", len(test_data))