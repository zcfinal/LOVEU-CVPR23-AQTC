

import os, json
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

import timm
from timm.data import resolve_data_config

# unused
def to_seconds(s):
    start, end = s.split('-')
    h, m, s = start.split(':')
    start = int(h) * 3600 + int(m) * 60 + int(s)
    h, m, s = end.split(':')
    end = int(h) * 3600 + int(m) * 60 + int(s)
    assert start < end
    return (start, end)

class RawAssistQA(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.dataset_root = cfg.DATASET.ROOT 
        self.video_root = cfg.DATASET.VIDEO
        self.dataset_split = cfg.DATASET.SPLIT
        self.dataset_label = cfg.DATASET.LABEL
        self.label_file = os.path.join(self.dataset_root, self.dataset_label)
        self.output_dir = cfg.OUTPUT_DIR
        with open(self.label_file) as f:
            self.qa_dict = json.load(f)
            self.folders = list(self.qa_dict.keys())

        self.frame_transform = T.Compose([
            T.Resize(size=cfg.INPUT.SIZE),
            lambda x:x/255,
        ])
        self.fps = cfg.INPUT.FPS
        self.for_video = cfg.FOR.VIDEO
        self.for_script = cfg.FOR.SCRIPT
        self.for_qa = cfg.FOR.QA
        self.for_para = cfg.FOR.PARA
        self.for_button = cfg.FOR.BUTTON
        self.num_masks = cfg.QA.NUM_MASKS

    def get_video(self, sample_path:str, fps:int):
        video_path = os.path.join(sample_path, "video.mp4")
        if not os.path.exists(video_path):
            video_path = video_path.replace('.mp4', '.mov')
        assert os.path.exists(video_path), sample_path
        frames, audios, meta = torchvision.io.read_video(video_path)
        # TODO: rerun the video encoding
        idx = torch.arange(0, frames.size(0), meta['video_fps'] / fps).long() 
        return frames[idx].permute(0,3,1,2)

    def get_script(self, sample_path:str):
        def string2time(timestamp):
            return (int(timestamp[3])*60+int(timestamp[5:7]), int(timestamp[11])*60+int(timestamp[13:15]))

        script_path = os.path.join(sample_path, "script.txt")
        with open(script_path) as f:
            sentences = f.readlines()
        timestamps = [sentence[:16].strip() for sentence in sentences if len(sentence)>15]
        sentences = [sentence[16:].strip().split('. ') for sentence in sentences if len(sentence)>15]
        timestamps_align = []
        for timestamp, sentence in zip(timestamps, sentences):
            for _ in range(len(sentence)):
                timestamps_align.append(timestamp)
        # merge
        sentences = sum(sentences, [])
        timestamps_align = [string2time(timestamp) for timestamp in timestamps_align]
        assert len(sentences) == len(timestamps_align)
        return sentences, timestamps_align

    def get_para(self, sample_path:str):
        para_path = os.path.join(sample_path, "paras.json")
        with open(para_path, 'r') as f:
            data = json.load(f)
        timestamps, paras = data
        return timestamps, paras
        

    def get_buttons_dict(self, sample_path:str):
        button_path = os.path.join(sample_path, "buttons.csv")
        with open(button_path) as f:
            table = map(lambda x:x.strip().split(','), f.readlines())
        buttons_dict = {}
        for name, x, y, w, h, image, iw, ih in table:
            if image not in buttons_dict:
                buttons_dict[image] = {}
            bbox = (int(x),int(y),int(x)+int(w),int(y)+int(h))
            buttons_dict[image][name] = bbox
        return buttons_dict
    
    def get_all_single_button_images(self, sample_path, image, all_bboxes, frame_transform,
        num_masks):
        image_path = os.path.join(sample_path, 'images', image)
        image = T.ToTensor()(Image.open(image_path).convert('RGB'))
        # mask all buttons
        allmask = image.clone()
        for x1, y1, x2, y2 in all_bboxes:
            allmask[:,y1:y2,x1:x2] = 0
        assert num_masks in [-1, 1, 2]
        mask = image.clone()
        nonmask = allmask.clone()
        button_images = []
        for x1, y1, x2, y2 in all_bboxes:
            mask[:,y1:y2,x1:x2] = 0
            nonmask[:,y1:y2,x1:x2] = image[:,y1:y2,x1:x2]
            if num_masks == 1:
                button_images.append(frame_transform(mask))
            elif num_masks == -1:
                button_images.append(frame_transform(nonmask))
            else:
                button_images.append(frame_transform(torch.stack([mask, nonmask])))
        return button_images

    def get_multiple_button_images(self, sample_path, image, bboxes, all_bboxes, frame_transform,
        num_masks):
        image_path = os.path.join(sample_path, 'images', image)
        image = T.ToTensor()(Image.open(image_path).convert('RGB'))
        # mask all buttons
        allmask = image.clone()
        for x1, y1, x2, y2 in all_bboxes:
            allmask[:,y1:y2,x1:x2] = 0
        assert num_masks in [-1, 1, 2]
        mask = image.clone()
        nonmask = allmask.clone()
        for x1, y1, x2, y2 in bboxes:
            mask[:,y1:y2,x1:x2] = 0
            nonmask[:,y1:y2,x1:x2] = image[:,y1:y2,x1:x2]
        if num_masks == 1:
            return frame_transform(mask)
        elif num_masks == -1:
            return frame_transform(nonmask)
        else:
            return frame_transform(torch.stack([mask, nonmask]))

    def __getitem__(self, index):
        folder = self.folders[index]
        sample_path = os.path.join(self.dataset_root, self.dataset_split, folder)
        output_path = os.path.join(self.output_dir, self.dataset_split, folder)
        qas = self.qa_dict[folder]

        if self.for_video:
            if not os.path.exists(os.path.join(output_path, "video.pth")):
                video = self.get_video(sample_path, self.fps)
                video = self.frame_transform(video)
                return video, output_path

        if self.for_script:
            script, timestamps = self.get_script(sample_path)
            script = [self.tokenizer(sentence, return_tensors="pt") for sentence in script]
            return script, timestamps, output_path
        
        if self.for_para:
            timestamps, paras = self.get_para(sample_path)
            paras = [self.tokenizer(sentence, return_tensors='pt') for sentence in paras]
            return timestamps, paras, output_path

        if self.for_button:
            buttons_dict = self.get_buttons_dict(sample_path) # image -> buttons, name -> bbox
            for qa in qas:
                question = qa['question']
                qa['folder'] = folder
                qa['src_question'] = question
                qa['button_images'] = []
                qa['answer_bidxs'] = []
                # for each step, answer - image
                assert len(qa['answers']) == len(qa['images']), output_path
                for i, (answers_per_step, image) in enumerate(zip(qa['answers'], qa['images'])): 
                    assert image in buttons_dict, output_path
                    qa_buttons = buttons_dict[image]
                    assert qa_buttons, output_path
                    names = qa_buttons.keys()
                    all_bboxes = qa_buttons.values()
                    qa['button_images'].append(
                        self.get_all_single_button_images(
                            sample_path, image, 
                            all_bboxes,
                            self.frame_transform, 
                            num_masks=self.num_masks
                        )
                    )
                    answer_bidxs_per_step = []
                    for j, answer in enumerate(answers_per_step):
                        # find the button in answer
                        bboxes = [qa_buttons[name] for name in names if name in answer]
                        bidxs = [k for k, name in enumerate(names) if name in answer]
                        assert len(bboxes) > 0, output_path
                        if len(bboxes) > 1: # multiple bboxes
                            qa['button_images'][-1].append(
                                self.get_multiple_button_images(
                                    sample_path, image, bboxes,
                                    all_bboxes,
                                    self.frame_transform, 
                                    num_masks=self.num_masks
                                )
                            )
                            bidx = len(qa['button_images'][-1]) - 1
                        bidx = bidxs[0]
                        answer_bidxs_per_step.append(bidx)
                    qa['answer_bidxs'].append(answer_bidxs_per_step)
            return qas, output_path, f'qa_maskx{self.num_masks}'

        if self.for_qa:
            buttons_dict = self.get_buttons_dict(sample_path) # image -> buttons, name -> bbox
            for qa in qas:
                question = qa['question']
                qa['folder'] = folder
                qa['src_question'] = question
                qa['question'] = self.tokenizer(f'Question: {question}', return_tensors="pt")
                qa['button_images'] = []
                qa['answer_bidxs'] = []
                # for each step, answer - image
                assert len(qa['answers']) == len(qa['images']), output_path
                for i, (answers_per_step, image) in enumerate(zip(qa['answers'], qa['images'])): 
                    assert image in buttons_dict, output_path
                    qa_buttons = buttons_dict[image]
                    assert qa_buttons, output_path
                    names = qa_buttons.keys()
                    all_bboxes = qa_buttons.values()
                    qa['button_images'].append(
                        self.get_all_single_button_images(
                            sample_path, image, 
                            all_bboxes,
                            self.frame_transform, 
                            num_masks=self.num_masks
                        )
                    )
                    answer_bidxs_per_step = []
                    for j, answer in enumerate(answers_per_step):
                        # find the button in answer
                        bboxes = [qa_buttons[name] for name in names if name in answer]
                        bidxs = [k for k, name in enumerate(names) if name in answer]
                        assert len(bboxes) > 0, output_path
                        if len(bboxes) > 1: # multiple bboxes
                            qa['button_images'][-1].append(
                                self.get_multiple_button_images(
                                    sample_path, image, bboxes,
                                    all_bboxes,
                                    self.frame_transform, 
                                    num_masks=self.num_masks
                                )
                            )
                            bidx = len(qa['button_images'][-1]) - 1
                        bidx = bidxs[0]
                        answer_bidxs_per_step.append(bidx)
                        answers_per_step[j] = self.tokenizer(f'Answer: {answer}', return_tensors="pt")
                    qa['answer_bidxs'].append(answer_bidxs_per_step)
            return qas, output_path, f'qa_maskx{self.num_masks}'

        return None

    def __len__(self, ):
        return len(self.folders)

    @staticmethod
    def collate_fn(samples):
        return samples

class VideoCLIPQA(RawAssistQA):
    def __getitem__(self, index):
        folder = self.folders[index]
        sample_path = os.path.join(self.dataset_root, self.dataset_split, folder)
        video_feature_path = os.path.join(self.video_root, self.dataset_split, folder)
        output_path = os.path.join(self.output_dir, self.dataset_split, folder)
        qas = self.qa_dict[folder]

        if self.for_script:
            script, timestamps = self.get_script(sample_path)
            video_frame = torch.load(os.path.join(video_feature_path, "video.pth"),map_location='cpu')
            return script, timestamps, video_frame, output_path
        
        if self.for_para:
            timestamps, paras = self.get_para(sample_path)
            video_frame = torch.load(os.path.join(video_feature_path, "video.pth"),map_location='cpu')
            return timestamps, paras, video_frame, output_path

        if self.for_qa:
            buttons_dict = self.get_buttons_dict(sample_path) # image -> buttons, name -> bbox
            for qa in qas:
                question = qa['question']
                qa['folder'] = folder
                qa['src_question'] = question
                qa['question'] = f'Question: {question}'
                qa['button_images'] = []
                qa['answer_bidxs'] = []
                # for each step, answer - image
                assert len(qa['answers']) == len(qa['images']), output_path
                for i, (answers_per_step, image) in enumerate(zip(qa['answers'], qa['images'])): 
                    assert image in buttons_dict, output_path
                    qa_buttons = buttons_dict[image]
                    assert qa_buttons, output_path
                    names = qa_buttons.keys()
                    all_bboxes = qa_buttons.values()
                    qa['button_images'].append(
                        self.get_all_single_button_images(
                            sample_path, image, 
                            all_bboxes,
                            self.frame_transform, 
                            num_masks=self.num_masks
                        )
                    )
                    answer_bidxs_per_step = []
                    for j, answer in enumerate(answers_per_step):
                        # find the button in answer
                        bboxes = [qa_buttons[name] for name in names if name in answer]
                        bidxs = [k for k, name in enumerate(names) if name in answer]
                        assert len(bboxes) > 0, output_path
                        if len(bboxes) > 1: # multiple bboxes
                            qa['button_images'][-1].append(
                                self.get_multiple_button_images(
                                    sample_path, image, bboxes,
                                    all_bboxes,
                                    self.frame_transform, 
                                    num_masks=self.num_masks
                                )
                            )
                            bidx = len(qa['button_images'][-1]) - 1
                        bidx = bidxs[0]
                        answer_bidxs_per_step.append(bidx)
                        answers_per_step[j] = f'Answer: {answer}'
                    qa['answer_bidxs'].append(answer_bidxs_per_step)
            return qas, output_path, f'qa_maskx{self.num_masks}'

class DataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
    
    def test_dataloader(self):
        cfg = self.cfg
        testset = eval(cfg.DATASET.TYPE)(cfg)
        return DataLoader(testset, batch_size=1, collate_fn=RawAssistQA.collate_fn,
            shuffle=False, drop_last=False, num_workers=0, pin_memory=True)

def build_data(cfg):
    return DataModule(cfg)

