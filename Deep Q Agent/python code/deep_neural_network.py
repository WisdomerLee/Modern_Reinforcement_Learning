#pytorch�� ����ϴ� deep neural network����

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

class LinearClassifier(nn.Module):
    def __init__(self, lr, n_classes, input_dims):
        super(LinearClassifier, self).__init__()
        #neural network�� layer�� : ó���� �Է� ���� ������ ���� ��������, �������� ���� �߰��� layer�� �Է� layer�� ���� ������ִ� ���� �ٽ�
        self.fc1 = nn.Linear(*input_dims, 128)
        #���� �Լ��� 128�� ����� �������Ƿ� �Է� ���� ������ ������ 128�� ��
        self.fc2 = nn.Linear(128, 256)
        #���� �Լ��� 256���� ����� �������Ƿ� �� �Լ��� ���� �Է��� ������ 256�� �ǰ�, ����� ������ ���� n_classes�� ������ŭ�̹Ƿ�..
        self.fc3 = nn.Linear(256, n_classes)
        #neural network�� ����� �� ���� optimizer�� ����Ͽ�... ����ȭ�� �����ϰ�
        self.optimizer = optim.Adam(self.parameters(), lr= lr)
        #loss �Լ��� ���� : �� �Լ��� ���� �ּҷ� ����� ��>>
        self.loss = nn.CrossEntropyLoss() #nn.MSELoss()
        #���� gpu�� �� �� �ִ� �����̸� gpu��, �׷��� ������ cpu�� �� �������� ����
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        #������ �ϵ��� ������ �������
        self.to(self.device)
        
    def forward(self, data):
        layer1 = F.sigmoid(self.fc1(data)) #ù ��° layer�� ó���� �Լ��� sigmoid �Լ��� ó��
        layer2 = F.sigmoid(self.fc2(layer1)) #�� ��° layer�� ó���� �Լ��� sigmoid �Լ��� ó��
        layer3 = self.fc3(layer2)
        #��������� layer�� activation�� ������ ����Ͽ� ���� layer�� �ѱ�� �� ���� layer�� �� ���� layer�� ó���� ����� ��� �Ѱ��ָ� network�� Ÿ�� �̵��ϰ� ��
        return layer3
    
    def learn(self, data, labels):
        #
        self.optimizer.zero_grad()
        
        data = T.tensor(data).to(self.device) #data�� device�� �Է��� �ѱ�!!
        labels = T.tensor(labels).to(self.device) #���� �� ����� ������ ������� �ϵ��� ����
        #T.Tensor()
        predictions = self.forward(data) #forward�� ������ �õ�
        
        cost = self.loss(predictions, labels) #�׸��� ������ ���� ���� ���� ���̸� ��! : ���̰� �۾ƾ� �ϹǷ�... cost�� ���̱� ���� 
        
        cost.backward() #�ش� ���̸� ���� backward�� back propagation�� ����
        self.optimizer.step() #����ȭ ����
        
