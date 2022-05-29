# 모델 구현 부분  
  
> 네트워크 구현 부분인 "body.py"와 구현된 네트워크의 훈련 방법을 설계한 "train.py"를 살펴보자.  
  
## 1. body.py  
> **네트워크 구현 부분**  
> ![블록구조](https://user-images.githubusercontent.com/98927470/170817186-2dd9debc-336d-4bc2-98df-dd0851eebd5a.png)  
> ![계층구조](https://user-images.githubusercontent.com/98927470/170817723-391569db-17dc-46b6-925b-e8870933a4ee.png)  
>   
> 상단의 그림과 표를 보자.  
> 먼저, stage는 반복되는 floor_layer의 묶음에 해당하며,  
>   
```python
def floor_layer(repeat_count, in_dim, mid_dim, out_dim, start = False):
    layers = []
    layers.append(tenantLayer(in_dim, mid_dim, out_dim, down = True, start = start))
    for _ in range(1, repeat_count):
        layers.append(tenantLayer(out_dim, mid_dim, out_dim, down = False))
    return nn.Sequential(*layers).to(device)
```    
>   
> floor_layer는 반복되는 tenant_layer(Resnet의 residual block)의 묶음에 해당한다.  
>   
```python
class tenantLayer(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, down = False, start = False):
        #생략
    def forward(self, x):
        identity = self.resize(x)
        x = self.block(x)
        x += identity
        x = self.leaky_relu(x)
        return x
```  
>    
> 조금 더 안쪽으로 파고 들어가보면,  
>   
```python
def make_conv(in_dim, mid_dim, out_dim, down = False):
    layers = []
    width = mid_dim // 64 * 32 * 4
    downsizing = 2 if down else 1
    layers.append(nn.Conv1d(in_dim, width, kernel_size = 1, stride = downsizing))
    layers.extend([
        nn.BatchNorm1d(width),
        nn.LeakyReLU(),
        nn.Conv1d(width, width, groups = 32, kernel_size = 3, padding = 1),
        nn.BatchNorm1d(width),
        nn.LeakyReLU(),
        nn.Conv1d(width, out_dim, kernel_size = 1),
        nn.BatchNorm1d(out_dim)
    ])
    return nn.Sequential(*layers).to(device)
```  
>   
> ResNeXt와 같은 합성곱 방식을 채택하였다. (groups 인자 사용)  
>   
>> ### New connection  
```python
   x = self.stage1_1(x) # 128 256 60 (Batch, Channel, Height)
   x = [x, self.stage1_(x_origin)] # right: 128 256 60 (Batch, Channel, Height)
   x = torch.cat(x, 2).to(device) # 128 256 120 (Batch, Channel, Height)
```  
>>  
>> stage를 통과한 feature map(x)과 제안한 new connection(self.stage1_(x_origin))을  
>> 통과한 출력을 concatenation시킨다.  
>> 여기서 x_origin은 네트워크로 들어오기 이전 텐서이다.  
>> **네트워크로 들어오기 이전 텐서가 무슨 말인데? 라고 한다면,**  
>> MIT-BIH DB에서 데이터를 불러온 뒤 이를 텐서화할 때,  
>> 각 샘플의 길이는 183이었으며, 배치사이즈는 128로 정했으므로  
>> 네트워크로 들어오기 이전 텐서는 (128, n, 183)이다. => (batch size, channel, height)  
>> 즉, 아무런 연산을 거치지 않은 텐서라고 생각하면 되겠다.  
>> 하지만 정말 아무런 연산을 거치지 않은 텐서라면 shape이 맞지 않아  
>> 네트워크를 통과 중인 feature map들과 의도한 연산이 불가능하기 때문에  
```python
def new_identity(dim, floor_num, div_num):
    layers = []
    in_dim = dim
    out_dim = dim * 4 * (2**(floor_num - 1))
    
    if floor_num < 3:
        layers.append(nn.Conv1d(1, in_dim, kernel_size = 7))    
        layers.extend([
            nn.BatchNorm1d(in_dim),
            nn.Conv1d(in_dim, out_dim, kernel_size = 1),
            nn.AvgPool1d(3 * (2**(div_num - 1))),
            nn.BatchNorm1d(out_dim)
        ])
    else:
        layers.append(nn.Conv1d(1, in_dim, kernel_size = 7, padding = 2))    
        layers.extend([
            nn.BatchNorm1d(in_dim),
            nn.Conv1d(in_dim, out_dim, kernel_size = 1, padding = 2),
            nn.AvgPool1d(3 * (2**(div_num - 1))),
            nn.BatchNorm1d(out_dim)
        ])        
    return nn.Sequential(*layers).to(device)
```  
>>   
>> 위 함수와 같이 shape을 맞춰주는 작업을 수행한다.  
>> 위 함수가 self.stage1_(x_origin)에 해당한다.  
![새로운기법](https://user-images.githubusercontent.com/98927470/170823446-9c20e6c7-7e46-46d5-be4b-6b081d986316.png)  

## 2. train.py  
> **훈련 방법**  
> 
