import torch

class UncertaintyModel(torch.nn.Module):
    def __init__(self, in_channels, num_features, hidden_units, p=0.2):
        super(UncertaintyModel,self).__init__()

        self.imgconv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=24, kernel_size=3, padding='same') 
        self.imgconv2 = torch.nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, padding='same')
        self.maxpool = torch.nn.AdaptiveMaxPool2d(1) # will return N,C,1,1 which we will later concat with graph features
        self.relu = torch.nn.ReLU()

        self.fc1 = torch.nn.Linear(num_features, hidden_units)
        self.fc2 = torch.nn.Linear(hidden_units, hidden_units*2)
        self.fc3 = torch.nn.Linear(hidden_units*2, hidden_units)
        self.fc4_1 = torch.nn.Linear(hidden_units, 1)
        self.fc4_2 = torch.nn.Linear(hidden_units, 1)
        self.dropout = torch.nn.Dropout(p)   
    
    def forward(self,imgbatch, x): # x is NF where N is batchsize (in our case its the number of manifolds in an image) and F is feature-length (num_features)
        imgbatch = self.dropout(self.relu(self.imgconv1(imgbatch)))
        imgbatch = self.dropout(self.relu(self.imgconv2(imgbatch)))
        imgbatch = self.maxpool(imgbatch)
        imgbatch = torch.squeeze(torch.squeeze(imgbatch, dim=3), dim=2)

        x = torch.concat((imgbatch, x), dim=1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.dropout(torch.relu(self.fc3(x)))

        mu = self.fc4_1(x)
        log_var = self.fc4_2(x)
        return mu, log_var