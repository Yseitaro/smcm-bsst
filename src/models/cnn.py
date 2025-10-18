import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')
import numpy as np
import os
from time import sleep

class EMGClassificationCNN(nn.Module):
    

    def __init__(self, input_dim, n_class):
        super(EMGClassificationCNN, self).__init__()
        self.input_dim = input_dim
        self.n_class = n_class

        # Convolution parameters
        kernel_size = 3

        # パディングの計算 (カーネルサイズが奇数の場合)
        padding_size = (kernel_size - 1) // 2

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=self.input_dim, out_channels=16, kernel_size=kernel_size, padding=padding_size),
            nn.BatchNorm1d(16),
            nn.PReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=padding_size)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_size, padding=padding_size),
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=padding_size)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, padding=padding_size),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=padding_size)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding=padding_size),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=padding_size)
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.bn = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, n_class)

    def forward(self, x):
        z = self.conv_block1(x)
        z = self.conv_block2(z)
        z = self.conv_block3(z)
        z = self.conv_block4(z)
        z = self.gap(z)
        z = z.view(z.size(0), -1)
        z = self.bn(self.fc1(z))
        output = self.fc2(z)
        return output
    

    

    def fit(self, x_tensor_train, y_tensor_train, s, cfg, Flag, x_tensor_val=None, y_tensor_val=None):
        '''
        x : train data
        y : train label
        Flag : True -> Fine tuning
               False -> Initial training
        '''


        
        x_tensor_train = x_tensor_train.reshape(x_tensor_train.shape[0], x_tensor_train.shape[2], x_tensor_train.shape[3])
        
        # print(f'x_tensor.shape : {x_tensor_train.shape}')

        # データローダの作成
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_tensor_train, y_tensor_train),
            batch_size=128,
            shuffle=True,
            drop_last=True
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
        best_loss = np.inf

        history = {
                "epoch": [],
                "train_loss": [],
                "val_loss": []
            }

        if Flag == False:
            # val検証
            # print(f'x_tensor_val.shape : {x_tensor_val.shape}')
            x_tensor_val = x_tensor_val.reshape(x_tensor_val.shape[0], x_tensor_val.shape[2], x_tensor_val.shape[3])
            val_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(x_tensor_val, y_tensor_val),
                batch_size=128,
                shuffle=False,
                drop_last=True
            )
            # 初期学習
            for epoch in range(300):
                self.train()
                train_epoch_loss = 0
                train_epoch_acc = 0
                for data, target in train_loader:
                    if target.ndim > 1:
                        target = target.argmax(dim=1)
                    output = self(data)
                    loss = criterion(output, target)

                    _, preds = torch.max(output, 1)
                    train_epoch_acc += torch.sum(preds == target.data)/len(target)
                
                    train_epoch_loss += loss.item()
                    # train_epoch_acc += torch.sum(preds == target.data)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                train_epoch_loss = train_epoch_loss / len(train_loader)
                train_epoch_acc = train_epoch_acc / len(train_loader)
                print(f'Epoch #{epoch+1}: train loss = {train_epoch_loss:.4f}, train acc = {train_epoch_acc:.4f}')

                
                val_epoch_loss = 0
                val_epoch_acc = 0
                self.eval()

                with torch.no_grad():
                    for data, targets in val_loader:

                        outputs = self(data) # Get output

                        loss = criterion(outputs, targets) # calculate loss

                        outputs = F.softmax(outputs, dim=1) # softmax
                        _, preds = torch.max(outputs, 1) # calculate predicted label

                        val_epoch_loss += loss.item()
                        val_epoch_acc += torch.sum(preds == targets.data)/len(targets)

                val_epoch_loss = val_epoch_loss / (len(val_loader))
                val_epoch_acc = val_epoch_acc / (len(val_loader))

                print(f' -> val loss: {val_epoch_loss:.4f}, val acc: {val_epoch_acc:.4f}')

                history['epoch'].append(epoch)
                history['train_loss'].append(train_epoch_loss)
                history['val_loss'].append(val_epoch_loss)

                # 検証データに対する損失が最小のモデルを保存
                if val_epoch_loss < best_loss:
                    print(f'saving..')
                    os.makedirs(f'../results/{cfg.cfg_name}/parameters/CNN',exist_ok=True)
                    save_path = f'../results/{cfg.cfg_name}/parameters/CNN/pretrained_emg_cnn_info_sub{s+1}.pth'
                    state = {
                        'epoch': epoch,
                        'val_acc': val_epoch_loss, 
                        'model_state_dict':  self.state_dict(),
                        'rng_state': torch.get_rng_state()
                    }
                    torch.save(state, save_path)
                    torch.save(self.state_dict(), f'../results/{cfg.cfg_name}/parameters/CNN/pretrained_emg_cnn_sub{s+1}.pth')

                if val_epoch_loss < best_loss:
                    best_loss = val_epoch_loss

            
        else:  
            # 事前学習済みモデルの読み込み（CNN層のみ転移）
            pretrained_model_path = f'../results/{cfg.cfg_name}/parameters/CNN/pretrained_emg_cnn_sub{s+1}.pth'
            pretrained_dict = torch.load(pretrained_model_path)

            # 現在のモデルのパラメータを取得
            model_dict = self.state_dict()

            # 転移可能なCNN層のパラメータのみを抽出
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'fc' not in k and k in model_dict}

            # モデルパラメータをアップデート
            model_dict.update(pretrained_dict)

            # 更新後のパラメータをモデルにロード
            self.load_state_dict(model_dict)

            # パラメータの固定    
            for param in self.parameters():
                param.requires_grad = False

            for name, param in self.named_parameters():
                if "conv_block4" in name or "fc2" in name:
                    param.requires_grad = True  

            # # CNN層のみファインチューニングする場合
            # for name, param in self.named_parameters():
            #     param.requires_grad = 'fc' not in name

            # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=0.001, momentum=0.9)
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=0.001, momentum=0.9)

            for epoch in range(5):
                self.train()
                for data, target in train_loader:
                    optimizer.zero_grad()
                    output = self(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                # # if epoch % 10 == 0:
                # print(f'Epoch {epoch+1} Loss: {loss.item()}')
        
            torch.save(self.state_dict(), f'../results/{cfg.cfg_name}/parameters/CNN/pretrained_emg_cnn_sub{s+1}.pth')


        # print(x_tensor.shape)
        # print(y_tensor.shape)
        # exit()

    def predict(self, x_tensor):
        '''
        x : test data
        '''
        # n_channel = x.shape[1]
        # x_tensor = torch.from_numpy(x).float().reshape(-1, 200, n_channel)
        # x_tensor = x_tensor.permute(0, 2, 1).unsqueeze(1)
        x_tensor = x_tensor.reshape(x_tensor.shape[0], x_tensor.shape[2], x_tensor.shape[3])
        print(f'x_tensor.shape : {x_tensor.shape}')
        self.eval()
        output = self(x_tensor)
        output = F.softmax(output, dim=1)
        # print(f'output : {output.shape}')
        # exit()
        return output.detach().numpy()





# ファインチューニングのためのサンプルコード
if __name__ == '__main__':
    model = EMGClassificationCNN(num_classes=9)

    # # 事前学習済みモデルの読み込み（CNN層のみ転移）
    # pretrained_model_path = 'pretrained_emg_cnn.pth'
    # pretrained_dict = torch.load(pretrained_model_path)
    # model_dict = model.state_dict()

    # # CNN層のみ転移（FC層のパラメータを除外）
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'fc' not in k}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

    # # 畳み込み層のみファインチューニング
    # for name, param in model.named_parameters():
    #     param.requires_grad = 'fc' not in name

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # サンプル入力 (batch_size=128)
    sample_input = torch.randn(128, 1, 5, 200)
    sample_labels = torch.randint(0, 9, (128,))

    # ファインチューニングの学習ステップ
    model.train()
    outputs = model(sample_input)
    print(outputs.shape)
    print(f'sample_labels.shape : {sample_labels.shape}')
    loss = criterion(outputs, sample_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Loss after one fine-tuning step: {loss.item()}')
