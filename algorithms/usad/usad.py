import torch
import torch.nn as nn

from utils import *
device = get_default_device()

# 修改 Encoder 类以支持正则化
# 修改 Encoder 类以移除批归一化
class Encoder(nn.Module):
  def __init__(self, in_size, latent_size, dropout_rate=0.0, use_batch_norm=False):
    super().__init__()
    layers = []
    
    # First layer
    layers.append(nn.Linear(in_size, int(in_size/2)))
    # 移除了 BatchNorm 层
    layers.append(nn.ReLU(True))
    if dropout_rate > 0:
        layers.append(nn.Dropout(dropout_rate))
  
    # Second layer
    layers.append(nn.Linear(int(in_size/2), int(in_size/4)))
    # 移除了 BatchNorm 层
    layers.append(nn.ReLU(True))
    if dropout_rate > 0:
        layers.append(nn.Dropout(dropout_rate))
        
    # Third layer
    layers.append(nn.Linear(int(in_size/4), latent_size))
    # 移除了 BatchNorm 层
    layers.append(nn.ReLU(True))
    if dropout_rate > 0:
        layers.append(nn.Dropout(dropout_rate))
        
    self.layers = nn.Sequential(*layers)
        
  def forward(self, w):
    return self.layers(w)
# 修改 Decoder 类以支持正则化
# 修改 Decoder 类以移除批归一化
class Decoder(nn.Module):
  def __init__(self, latent_size, out_size, dropout_rate=0.0, use_batch_norm=False):
    super().__init__()
    layers = []
    
    # First layer
    layers.append(nn.Linear(latent_size, int(out_size/4)))
    # 移除了 BatchNorm 层
    layers.append(nn.ReLU(True))
    if dropout_rate > 0:
        layers.append(nn.Dropout(dropout_rate))
        
    # Second layer
    layers.append(nn.Linear(int(out_size/4), int(out_size/2)))
    # 移除了 BatchNorm 层
    layers.append(nn.ReLU(True))
    if dropout_rate > 0:
        layers.append(nn.Dropout(dropout_rate))
        
    # Third layer
    layers.append(nn.Linear(int(out_size/2), out_size))
    # 移除了 BatchNorm 层
    layers.append(nn.ReLU(True))  # Changed from Sigmoid to ReLU for intermediate layer
    layers.append(nn.Sigmoid())
        
    self.layers = nn.Sequential(*layers)
        
  def forward(self, z):
    return self.layers(z) 
class UsadModel(nn.Module):
  def __init__(self, w_size, z_size, dropout_rate=0.1, use_batch_norm=True):
    super().__init__()
    self.encoder = Encoder(w_size, z_size, dropout_rate, use_batch_norm)
    self.decoder1 = Decoder(z_size, w_size, dropout_rate, use_batch_norm)
    self.decoder2 = Decoder(z_size, w_size, dropout_rate, use_batch_norm)
  
  def training_step(self, batch, n):
    z = self.encoder(batch)#z=e(x)
    w1 = self.decoder1(z)#w1=d(z)=d(e(x))
    w2 = self.decoder2(z)#w2=d(z)=d(e(x))
    w3 = self.decoder2(self.encoder(w1))#w3=d(e(d(z)))
    loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)#ae1:尽可能生成与输入一样的数据，并且
    #尽可能让自己生成的数据让ae2认为也是正常的（最小化batch-w3）
    loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
    #ae2：尽可能生成与输入一致的数据，并尽可能区分ae1生成的数据（认为ae1生成的数据不正常）
    return loss1,loss2

  def validation_step(self, batch, n):
    with torch.no_grad():
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
        loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
    return {'val_loss1': loss1, 'val_loss2': loss2}
        
  def validation_epoch_end(self, outputs):
    batch_losses1 = [x['val_loss1'] for x in outputs]
    epoch_loss1 = torch.stack(batch_losses1).mean()
    batch_losses2 = [x['val_loss2'] for x in outputs]
    epoch_loss2 = torch.stack(batch_losses2).mean()
    return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item()}
    
  def epoch_end(self, epoch, result):
    print("Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(epoch, result['val_loss1'], result['val_loss2']))
    
def evaluate(model, val_loader, n):
    outputs = [model.validation_step(to_device(batch,device), n) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def training(epochs, model, train_loader, val_loader, learning_rate=0.0004, weight_decay=0.0):
    history = []
    optimizer1 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder1.parameters()), 
        lr=learning_rate,
        weight_decay=weight_decay
    )
    optimizer2 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder2.parameters()), 
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    for epoch in range(epochs):
        n=0
        for batch in train_loader:
            batch=to_device(batch,device)
            print(n)
            n+=1
            #Train AE1
            loss1,loss2 = model.training_step(batch,epoch+1)
            # 检查损失是否为NaN
            if torch.isnan(loss1):
                print(f"NaN loss detected in AE1 at epoch {epoch}")
                return history
                
            loss1.backward()
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(list(model.encoder.parameters()) + list(model.decoder1.parameters()), max_norm=1.0)
            optimizer1.step()
            optimizer1.zero_grad()
            
            #Train AE2
            loss1,loss2 = model.training_step(batch,epoch+1)
            # 检查损失是否为NaN
            if torch.isnan(loss2):
                print(f"NaN loss detected in AE2 at epoch {epoch}")
                return history
            
            loss2.backward()
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(list(model.encoder.parameters()) + list(model.decoder2.parameters()), max_norm=1.0)
            optimizer2.step()
            optimizer2.zero_grad()
            
        result = evaluate(model, val_loader, epoch+1)
        print(result)
        model.epoch_end(epoch, result)
        history.append(result)
    return history
def testing(model, test_loader, alpha=.5, beta=.5):
    results=[]
    with torch.no_grad():
        for batch in test_loader:
            batch=to_device(batch,device)
            c=model.encoder(batch)
            
            w1=model.decoder1(c)
            
            w2=model.decoder2(model.encoder(w1))
           
            results.append(alpha*torch.mean((batch-w1)**2,axis=1)+beta*torch.mean((batch-w2)**2,axis=1))
    return results