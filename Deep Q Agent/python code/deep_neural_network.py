#pytorch를 사용하는 deep neural network예시

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

class LinearClassifier(nn.Module):
    def __init__(self, lr, n_classes, input_dims):
        super(LinearClassifier, self).__init__()
        #neural network의 layer들 : 처음엔 입력 값의 갯수와 같은 차원으로, 내보내는 값은 중간의 layer의 입력 layer와 같게 만들어주는 것이 핵심
        self.fc1 = nn.Linear(*input_dims, 128)
        #앞의 함수가 128의 결과를 내놓으므로 입력 받을 차원의 갯수가 128이 됨
        self.fc2 = nn.Linear(128, 256)
        #앞의 함수가 256개의 결과를 내놓으므로 이 함수가 받을 입력의 갯수도 256이 되고, 결과로 내놓을 값이 n_classes의 갯수만큼이므로..
        self.fc3 = nn.Linear(256, n_classes)
        #neural network를 통과한 저 값이 optimizer를 통과하여... 최적화를 진행하고
        self.optimizer = optim.Adam(self.parameters(), lr= lr)
        #loss 함수를 설정 : 이 함수의 값을 최소로 낮춰야 함>>
        self.loss = nn.CrossEntropyLoss() #nn.MSELoss()
        #만약 gpu를 쓸 수 있는 조건이면 gpu를, 그렇지 않으면 cpu를 쓸 조건으로 설정
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        #설정한 하드웨어를 설정에 집어넣음
        self.to(self.device)
        
    def forward(self, data):
        layer1 = F.sigmoid(self.fc1(data)) #첫 번째 layer를 처리한 함수를 sigmoid 함수로 처리
        layer2 = F.sigmoid(self.fc2(layer1)) #두 번째 layer를 처리한 함수를 sigmoid 함수로 처리
        layer3 = self.fc3(layer2)
        #결과적으로 layer와 activation을 순차로 통과하여 다음 layer로 넘기고 그 다음 layer는 그 다음 layer로 처리한 결과를 계속 넘겨주며 network를 타고 이동하게 됨
        return layer3
    
    def learn(self, data, labels):
        #
        self.optimizer.zero_grad()
        
        data = T.tensor(data).to(self.device) #data를 device에 입력을 넘김!!
        labels = T.tensor(labels).to(self.device) #얻어야 할 결과의 값들이 몇가지인지 하드웨어에 설정
        #T.Tensor()
        predictions = self.forward(data) #forward로 예측을 시도
        
        cost = self.loss(predictions, labels) #그리고 예측한 값과 실제 값의 차이를 비교! : 차이가 작아야 하므로... cost를 줄이기 위해 
        
        cost.backward() #해당 차이를 토대로 backward로 back propagation이 진행
        self.optimizer.step() #최적화 진행
        
