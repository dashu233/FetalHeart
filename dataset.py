import torch
from torch.utils.data.dataset import Dataset
from data_argumentation import *
import pickle
import matplotlib.pyplot as plt

class BinaryFetalHeartDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file_id_list, transform=None,draw=False,reidx=False,device='cpu'):
        self.reidx=reidx
        if type(transform)!=list:
            print(type(transform))
            print('convert!')
            transform = [transform]
        self.raw_data = []
        self.raw_label = []
        double_num = cfg.getint('data','double_num')
        data_dir = cfg.get('data','data_dir')
        self.file_list = file_id_list
        self.file_map = []
        for i in file_id_list:
            with open(data_dir+'/{}.pkl'.format(i), 'rb') as f:
                dt = pickle.load(f)
                if dt['y'] == 1:
                    for _ in range(double_num):
                        self.raw_data.append(dt['x'])
                        self.raw_label.append(dt['y'])
                        self.file_map.append(i)
                else:
                    self.raw_data.append(dt['x'])
                    self.raw_label.append(dt['y'])
                    self.file_map.append(i)
                if 'name' in dt:
                    print('file{}:{}'.format(i,dt['name']))
                # label is \pm 1
                if draw:
                    try:
                        assert 'name' in dt
                    except KeyError:
                        print('draw could only used for named data')
                    x = np.linspace(0,1,len(dt['x']))
                    y = dt['x']
                    fig = plt.figure()
                    plt.plot(x,y)
                    plt.ylim((120,180))
                    plt.savefig(str(i))

                    #plt.cla()
        if cfg.get('model','type') == 'fft':
            self.raw_data = np.array(self.raw_data)
            assert self.raw_data.shape[1] == 4800, 'fft only support datalen == 4800'
            transfered_data = np.fft.fft(self.raw_data)
            self.raw_data = torch.Tensor([transfered_data.real,transfered_data.imag])
            #print(self.raw_data.size())
            self.raw_data = self.raw_data.permute(1,0,2).reshape(-1,9600).to(device)
            #
        else:
            self.raw_data = torch.Tensor(self.raw_data)
        self.raw_label = torch.Tensor(self.raw_label)

        self.transform = transform

    def __len__(self):
        return len(self.raw_label)

    def __getitem__(self, idx):
        #print(idx)
        batchx = self.raw_data[idx]


        if isinstance(idx,int):
            batchx = batchx.view(1,-1)
        #print(batchx)
        if self.transform:
            for ttt in self.transform:
                batchx = ttt(batchx)
                #print(batchx)
                #mi = torch.min(batchx)
                #if torch.isnan(mi):
                    #print('question argue:',type(ttt))
        batchy = self.raw_label[idx]
        if isinstance(idx,int):
            batchx.squeeze_()

        #mi = torch.min(batchx)
        #if torch.isnan(mi):
            #print('question file:', self.file_map[idx])
            #print('question id:',idx)
        if self.reidx:
            return batchx,batchy,self.file_list[idx]
        else:
            return batchx,batchy

if __name__ == '__main__':

    import ast
    trans = ast.literal_eval(cfg.get('data','argument'))
    transforms = []
    for t in trans:
        transforms.append(DATA_ARGUMENT_DICT[t])
    dataset = BinaryFetalHeartDataset(file_id_list=[1,3,5,7],transform=transforms)