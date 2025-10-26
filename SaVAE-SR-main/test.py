import torch
torch.manual_seed(2021)
mean = torch.rand(1, 2)
std = torch.rand(1, 2)

target_value = torch.tensor([1.8208, 0.6635])
 # 容忍误差

for i in range(10000000):
    

    
    result = torch.normal(mean, std)
    print(result)
    
    # 检查是否达到目标值（考虑浮点数精度）
    if torch.allclose(result, target_value):
        print("达到目标值，停止循环")
        print(i)
        break
    if(i%1000==0):
        print(i)