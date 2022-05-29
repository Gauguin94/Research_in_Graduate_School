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
> ResNeXt와 같은 그룹합성곱 방식을 채택하였다. (groups 인자 사용)  
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
> (1) Optimizer(최적화 도구)  
```python
optimizer = adabound.AdaBound(model.parameters(), lr = lr, final_lr = 0.1)
```  
>> 최적화 도구로는 adam과 같은 ada 계열의 adabound를 사용하였다.  
>> adam과 비교하였을 때, 조금 더 나은 성능을 보여 이를 사용하게 되었다.  
>> reference: [Adaptive Gradient Methods with Dynamic Bound of Learning Rate](https://openreview.net/forum?id=Bkg3g2R9FX)  
>  
> (2) Loss function(손실 함수)  
```python
class FocalLoss(nn.Module):
    def __init__(self, gamma = 0, alpha = None, size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, float)): self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt*Variable(at)
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum() 
```  
>> 손실 함수로는 Focal loss를 사용하였다.  
>> Focal loss는 학습 중 클래스 불균형 문제가 심한 것을 고려하여 제안된 기법이다.  
>>  
![focalloss](https://user-images.githubusercontent.com/98927470/170855379-63c570ed-09cf-474e-ba19-94e16ef32b01.png)  
  
>> Cross Entrophy의 식은 아래와 같이 나타낼 수 있다.  
>> *CE = -Y<sub>ans</sub>\*log(p)-(1-Y<sub>ans</sub>)\*log⁡(1-p)*  
>> *if) Y<sub>ans</sub> = 1, CE(p,Y<sub>ans</sub>) = -log⁡(p)+0*  
>> Y<sub>ans</sub>=1인 경우에 대해 살펴보자.  
>> *p*의 값이 1이라면 Loss는 0이 된다. 모델의 출력이 실제 정답과 같다는 것을 나타내며,  
>> 일반적으로 모델의 성능이 좋은 경우라고 할 수 있다.  
>> 반면 *p*의 값이 0에 가까운 값이라면 결과가 무한대에 점점 가까워진다.  
>> Focal loss는 합성곱 신경망을 기반으로 한 모델인 RetinaNet으로 성능이 입증된 손실 함수이다.  
>> 상단의 그래프에서 볼 수 있는 식과 같이 Focal loss는 Cross Entrophy에 *(1-p<sub>t</sub>)<sup>r</sup>* 을 곱한 형태이다.  
>> *p<sub>t</sub>* 는 ground truth를 맞춘 경우를 나타낸 확률 값이며,  
>> *r*이 0이라면 Focal loss는 Cross Entrophy와 같다.  
>> 반대로 *r*의 크기가 커질수록 곡선의 곡률이 커지게 된다.  
>> 여기서 *r*은 focusing parameter라고 하며 조정 가능한 파라미터이다.  
>> *r*이 0인 경우(Cross Entrophy)와 *r*이 5인 경우를 비교해보면,  
>> *p<sub>t</sub>* 가 높은 구간에 대응되는 Loss들은 0에 가깝기 때문에  
>> 학습 진행 간 해당 구간의 Loss가 갱신되더라도 전체적인 Loss에 끼치는 영향이 적음을 알 수 있다.  
>> (Loss는 결국 모든 Loss를 더한 뒤 출력된 개수만큼 나누어 평균을 구한 값이기 때문.)  
>> 이와 같은 이유로, 학습 데이터 불균형을 어느 정도 해소하고자 Focal loss를 사용하였다.  
>> reference: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
