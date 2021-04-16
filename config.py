from configparser import ConfigParser
import ast


class MyConfigParser(ConfigParser):
    def getliststr(self,section,option):
        return ast.literal_eval(self.get(section, option))
    def getlistint(self,section,option):
        return [int(x) for x in ast.literal_eval(self.get(section, option))]
    def getlistfloat(self,section,option):
        return [float(x) for x in ast.literal_eval(self.get(section, option))]


cfg = MyConfigParser()

cfg.read("binary_pred.conf")
if cfg.getboolean('data','default'):
    if cfg.get('data','data_dir') == 'dataset' or 'dataset_changed' or 'named_data':
        cfg['train']['batch_size']='128'
        cfg['train']['train_num']='5632'
        cfg['eval']['assigned']='True'
    elif cfg.get('data','data_dir') == 'split_data':
        cfg['train']['batch_size'] = '512'
        cfg['train']['train_num'] = '20480'
        cfg['eval']['assigned'] = 'False'


if __name__ == '__main__':
    tst = cfg.getlistfloat('train','check')
    for t in tst:
        print(t)
        print(type(t))