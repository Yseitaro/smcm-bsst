import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
import os
import numpy as np

class MLP(nn.Module):
    def __init__(self, n_channel, n_class):
        super(MLP, self).__init__()
        self.n_channel = n_channel

        self.fc1 = nn.Linear(self.n_channel, 100)
        self.batch_norm1 = nn.BatchNorm1d(100)

        self.fc2 = nn.Linear(100, 50)
        self.batch_norm2 = nn.BatchNorm1d(50)

        self.fc3 = nn.Linear(50, 25)
        self.batch_norm3 = nn.BatchNorm1d(25)

        self.output = nn.Linear(25, n_class)

    def forward(self, x):
        x = x.reshape(-1, 1 * self.n_channel)
        z = self.fc1(x)
        z = F.relu(self.batch_norm1(z))
        z = self.fc2(z)
        z = F.relu(self.batch_norm2(z))
        z = self.fc3(z)
        z = F.relu(self.batch_norm3(z))
        output = self.output(z)
        return output

    def fit(self, x_tensor_train, y_tensor_train, s, cfg, Flag, x_tensor_val=None, y_tensor_val=None):
        '''
        x : train data
        y : train label
        Flag : True -> Fine tuning
               False -> Initial training
        '''
        # print(f'x_tensor : {x_tensor.shape}')
        # print(f'x_tensor : {x_tensor.reshape(x_tensor.shape[0],-1).shape}')
        # exit()
        # データローダの作成
        
        x_tensor_train = x_tensor_train.reshape(x_tensor_train.shape[0], -1)
        # print(f'x_tensor_train : {x_tensor_train.shape}')
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
                    os.makedirs(f'../results/{cfg.cfg_name}/parameters/MLP',exist_ok=True)
                    save_path = f'../results/{cfg.cfg_name}/parameters/MLP/pretrained_emg_mlp_info_sub{s+1}.pth'
                    state = {
                        'epoch': epoch,
                        'val_acc': val_epoch_loss, 
                        'model_state_dict':  self.state_dict(),
                        'rng_state': torch.get_rng_state()
                    }
                    torch.save(state, save_path)
                    torch.save(self.state_dict(), f'../results/{cfg.cfg_name}/parameters/MLP/pretrained_emg_mlp_sub{s+1}.pth')

                if val_epoch_loss < best_loss:
                    best_loss = val_epoch_loss

        else:
            # ファインチューニング         
            
            
            pretrained_model_path = f'../results/{cfg.cfg_name}/parameters/MLP/pretrained_emg_mlp_sub{s+1}.pth'
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
                # 後半の層に属する場合は更新対象とする（例："fc3", "batch_norm3", "output"）
                if name.startswith('fc3') or name.startswith('batch_norm3') or name.startswith('output'):
                    param.requires_grad = True
                else:
                    param.requires_grad = False


            # CNN層のみファインチューニングする場合
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
            
            torch.save(self.state_dict(), f'../results/{cfg.cfg_name}/parameters/MLP/pretrained_emg_mlp_sub{s+1}.pth')
            # pass

    def predict(self, x_tensor):
        '''
        x : test data
        '''
        
        self.eval()
        output = self(x_tensor)
        output = F.softmax(output, dim=1)
        # print(f'output : {output.shape}')
        # exit()
        return output.detach().numpy()
