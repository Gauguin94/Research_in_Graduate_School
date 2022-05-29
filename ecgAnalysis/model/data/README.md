# dataload.py  
```python
class ecgData(torch.utils.data.Dataset):
    def __init__(self, train = True):
        self.train = train
        data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        input_dir = os.path.dirname(data_dir) + '/ecgdata_not_mine/archive'
        if self.train == True:
            print('Load train data...')
            data = np.loadtxt('{}/mitbih_train.csv'.format(input_dir), delimiter = ',', dtype = np.float32)
            
            self.data_len = data.shape[0]
            self.train_data = torch.from_numpy(data[:,:187])
            self.train_label = torch.from_numpy(data[:,[187]])

        else:
            print('Load test data...')
            test = np.loadtxt('{}/mitbih_test.csv'.format(input_dir), delimiter = ',', dtype = np.float32)
            self.test_len = test.shape[0]
            self.test_data = torch.from_numpy(test[:,:187])
            self.test_label = torch.from_numpy(test[:,[187]])
```  
  
> 저장 공간에 미리 다운로드 받은 MIT-BIH arrhythmia DB를 불러오는 작업을 수행한다.  
> 앞서 언급했듯이, 각 샘플의 마지막 time-step(Height 혹은 벡터의 관점에서 dimension)이  
> label이기 때문에 이를 분리하여 저장하도록 한다.  
  
# dataset.py  
```python
BATCH_SIZE = 128
TEST_BATCH = 21892

class setData:
    def __init__(self):
        self.trainset = ecgData(train = True)
        self.testset = ecgData(train = False)
    def save_data(self):
        train_batch = DataLoader(self.trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers= 6)
        val_batch = DataLoader(self.trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers= 6)
        test_batch = DataLoader(self.testset, batch_size = BATCH_SIZE, shuffle = False, num_workers= 6)
        whole_test_batch = DataLoader(self.testset, batch_size = TEST_BATCH, shuffle = False, num_workers= 6)
        return train_batch, val_batch, test_batch, whole_test_batch, self.trainset, self.testset
```  
  
> Dataloader를 통해 DB를 128 크기의 배치로 저장한다.  
> train, validation, test에 쓰일 데이터로 나눠주도록 한다.  
