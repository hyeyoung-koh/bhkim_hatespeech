import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import model_loader
import numpy as np
import pandas as pd
import dataset
from tqdm import tqdm

def predict(path,bs,tokenizer_,data_path,device=torch.device("cuda")):
    result=[]
    model=torch.load(path).to(torch.device(device))
    data_predict=dataset.CustomDataset(data_path,tokenizer=tokenizer_)
    predict_data=DataLoader(data_predict,batch_size=bs,num_workers=5)
    with tqdm(total=len(predict_data)) as pbar:
        for step,(audio_arr,token_ids,_) in enumerate(predict_data):
            audio_arr=torch.Tensor(audio_arr) #array를 tensor로 변환
            #res=model(x,pad_x).to(device)
            res_0=model(x_audio,x_text,a_mask,t_mask).to(device)
            res=res_0[0]
            max_values,max_indices=torch.max(res,1)
            result.append(list(max_indices.detach().cpu()))
            pbar.update(1)
    return torch.flatten(torch.tensor(result))

if __name__=="__main__":
    PATH="D:/Users/hyeyoung/OneDrive - dongguk.edu/문서/카카오톡 받은 파일/modelMULT_1.bin"
    #tokenizer=AutoTokenizer.from_pretrained("")
    tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
    my_data_path=pd.read_pickle('E:/iitp/test.pkl')
    res=predict(PATH,16,tokenizer,my_data_path)
    d=pd.read_csv('test.csv')
    d['predict']=res
    d.to_csv('result.csv',encoding='utf-8-sig')