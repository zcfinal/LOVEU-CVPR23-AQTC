import os
import torch
import random
import math
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

class EncodedAssistQA(Dataset):
    def __init__(self, samples):
        super().__init__()
        self.samples = samples
        
    def __getitem__(self, index):
        sample = self.samples[index]
        video = torch.load(sample["video"], map_location="cpu")

        timestamp_script = torch.load(sample["script"], map_location="cpu")
        sents_timestamp, script = timestamp_script

        timestamp_para = torch.load(sample['para'], map_location="cpu")
        paras_timestamp, function_para = timestamp_para

        question = sample["question"]
        actions = sample["answers"]
        meta = {
            'question': sample['src_question'], 'folder': sample['folder'], 
            'paras_score': sample['paras_score'], 'paras_timestamp': paras_timestamp, 
            'sents_score': sample['sents_score'], 'sents_timestamp': sents_timestamp
        }
        if 'correct' in sample:
            label = torch.tensor(sample['correct']) - 1 # NOTE here, start from 1
        else:
            label = None
        rslt = {}
        rslt['video']=video
        rslt['script']=script
        rslt['question']=question
        rslt['function_para']=function_para
        rslt['actions']=actions
        rslt['label']=label
        rslt['meta']=meta
        return rslt
        
    def __len__(self, ):
        return len(self.samples)

    @staticmethod
    def collate_fn(samples):
        return samples


class EncodedAssistMachineNameQA(EncodedAssistQA):
        
    def __getitem__(self, index):
        sample = self.samples[index]
        video = torch.load(sample["video"], map_location="cpu")

        timestamp_script = torch.load(sample["script"], map_location="cpu")
        sents_timestamp, script = timestamp_script

        timestamp_para = torch.load(sample['para'], map_location="cpu")
        paras_timestamp, function_para = timestamp_para

        machine_name = torch.load(sample['machine_name'], map_location="cpu")

        question = sample["question"]
        actions = sample["answers"]
        meta = {
            'question': sample['src_question'], 'folder': sample['folder'], 
            'paras_score': sample['paras_score'], 'paras_timestamp': paras_timestamp, 
            'sents_score': sample['sents_score'], 'sents_timestamp': sents_timestamp
        }
        if 'correct' in sample:
            label = torch.tensor(sample['correct']) - 1 # NOTE here, start from 1
        else:
            label = None
        rslt = {}
        rslt['video']=video
        rslt['script']=script
        rslt['question']=question
        rslt['function_para']=function_para
        rslt['actions']=actions
        rslt['label']=label
        rslt['meta']=meta
        rslt['machine_name']=machine_name
        return rslt
        

class EncodedAssistQADataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        root = self.cfg.DATASET.TRAIN
        train_samples = []
        valid_samples = []
        for t in os.listdir(root):
            sample = torch.load(os.path.join(root, t, cfg.INPUT.QA), map_location="cpu")
            for s in sample:
                s["video"] = os.path.join(root, t, cfg.INPUT.VIDEO)
                s["script"] = os.path.join(root, t, cfg.INPUT.SCRIPT)
                s["para"] = os.path.join(root, t, cfg.INPUT.PARA)
                if 'MachineName' in cfg.DATASET.TYPE:
                    s["machine_name"] = os.path.join(root, t, cfg.INPUT.MACHINE_NAME)
            random.shuffle(sample)
            split_num = int(len(sample)*self.cfg.DATASET.SPLIT_RATIO)
            train_samples.extend(sample[:split_num])
            valid_samples.extend(sample[split_num:])
        self.train_samples = train_samples
        self.valid_samples = valid_samples

    
    def train_dataloader(self): 
        cfg = self.cfg
        trainset = eval(cfg.DATASET.TYPE)(self.train_samples)
        return DataLoader(trainset, batch_size=cfg.SOLVER.BATCH_SIZE, collate_fn=EncodedAssistQA.collate_fn,
            shuffle=True, drop_last=True, num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True)

    def val_dataloader(self):
        cfg = self.cfg
        valset = eval(cfg.DATASET.TYPE)(self.valid_samples)
        return DataLoader(valset, batch_size=cfg.SOLVER.BATCH_SIZE, collate_fn=EncodedAssistQA.collate_fn,
            shuffle=False, drop_last=False, num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True)

class EncodedAssistQATestDataModule(EncodedAssistQADataModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        root = self.cfg.DATASET.TRAIN
        train_samples = []
        valid_samples = []
        for t in os.listdir(root):
            sample = torch.load(os.path.join(root, t, cfg.INPUT.QA), map_location="cpu")
            for s in sample:
                s["video"] = os.path.join(root, t, cfg.INPUT.VIDEO)
                s["script"] = os.path.join(root, t, cfg.INPUT.SCRIPT)
                s["para"] = os.path.join(root, t, cfg.INPUT.PARA)
            valid_samples.extend(sample)
        self.train_samples = train_samples
        self.valid_samples = valid_samples

    
    
def build_data(cfg):
    if cfg.DATASET.GT:
        return EncodedAssistQADataModule(cfg)
    else:
        return EncodedAssistQATestDataModule(cfg)