import torch, os, json
from torch import nn
from torch.functional import F
from pytorch_lightning import LightningModule


class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.l1=nn.LayerNorm(normalized_shape=in_features)
        self.l2=nn.LayerNorm(normalized_shape=in_features)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.l2(x)
        x = self.fc2(x)
        return x

class Q2A(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.mlp_v = MLP(cfg.INPUT.DIM, cfg.INPUT.DIM)
        self.mlp_t = MLP(cfg.INPUT.DIM, cfg.INPUT.DIM)
        self.mlp_pre = MLP(cfg.INPUT.DIM*(3+cfg.INPUT.NUM_MASKS), cfg.MODEL.DIM_STATE)

        self.s2v = nn.MultiheadAttention(cfg.INPUT.DIM, cfg.MODEL.NUM_HEADS)
        self.qa2s = nn.MultiheadAttention(cfg.INPUT.DIM, cfg.MODEL.NUM_HEADS)
        
        self.state = torch.randn(cfg.MODEL.DIM_STATE, device="cuda")
        if cfg.MODEL.HISTORY.ARCH == "mlp":
            self.proj = MLP(cfg.MODEL.DIM_STATE*2, 1)
        elif cfg.MODEL.HISTORY.ARCH == "gru":
            self.gru = nn.GRUCell(cfg.MODEL.DIM_STATE, cfg.MODEL.DIM_STATE)
            self.proj = MLP(cfg.MODEL.DIM_STATE, 1)
        else:
            assert False, "unknown arch"
        
        self.history_train = cfg.MODEL.HISTORY.TRAIN
        self.history_val = cfg.MODEL.HISTORY.VAL
        self.function_centric = cfg.MODEL.FUNCTION_CENTRIC
        self.cfg = cfg

    def forward(self, batch):
        loss, count = 0, 0
        results = []
        for video, script, question, para, actions, label, meta in batch:
            if self.function_centric:
                # for visual
                timestamps = meta['paras_timestamp']
                video_func = []
                for seg in timestamps:
                    if seg[0] >= seg[1]:
                        video_func.append(video[seg[0]])
                    else:
                        video_func.append(video[seg[0]:seg[1]].mean(dim=0))
                video_func = torch.stack(video_func)
                video_func = self.mlp_v(video)
                # for text
                script_func = self.mlp_t(para)
                video_func = self.s2v(script_func.unsqueeze(1), video_func.unsqueeze(1), video_func.unsqueeze(1))[0].squeeze_()
            else:
                video = self.mlp_v(video)
                script = self.mlp_t(script)
                video = self.s2v(script.unsqueeze(1), video.unsqueeze(1), video.unsqueeze(1))[0].squeeze_()

            question = self.mlp_t(question)
            
            state = self.state
            scores = []
            for i, actions_per_step in enumerate(actions):
                a_texts, a_buttons = zip(*[(action['text'], action['button']) for action in actions_per_step])
                a_texts = self.mlp_t(torch.cat(a_texts))
                A = len(a_buttons)
                a_buttons = self.mlp_v(
                    torch.stack(a_buttons).view(A, -1, a_texts.shape[1])
                ).view(A, -1) 
                qa = question + a_texts

                if self.function_centric:
                    qa_script, qa_script_mask = self.qa2s(
                        qa.unsqueeze(1), script_func.unsqueeze(1), script_func.unsqueeze(1)
                    )
                    # for visualization
                    # print('attention score:', qa_script_mask.mean(dim=1).squeeze(0).softmax(dim=0))
                    # print('tfidf score', torch.tensor(meta['paras_score']))
                    # print(meta['question'])
                    # with open(os.path.join('/data/wushiwei/data/assistq/assistq_test/test', meta['folder'], 'paras.json'), 'r') as f:
                    #     paras = json.load(f)
                    # print(paras[1])
                    qa_video = qa_script_mask @ video_func.view(-1, 768)
                    inputs = torch.cat(
                        [qa_video.view(A, -1), qa_script.view(A, -1), qa.view(A, -1), a_buttons.view(A, -1)],
                        dim=1
                    )
                else:
                    qa_script, qa_script_mask = self.qa2s(
                        qa.unsqueeze(1), script.unsqueeze(1), script.unsqueeze(1)
                    )
                    qa_video = qa_script_mask @ video
                    inputs = torch.cat(
                        [qa_video.view(A, -1), qa_script.view(A, -1), qa.view(A, -1), a_buttons.view(A, -1)],
                        dim=1
                    )

                inputs = self.mlp_pre(inputs)
                if hasattr(self, "gru"):
                    states = self.gru(inputs, state.expand_as(inputs))
                else:
                    states = torch.cat([inputs, state.expand_as(inputs)], dim=1)
                logits = self.proj(states)
                if self.training:
                    loss += F.cross_entropy(logits.view(1, -1), label[i].view(-1))
                    count += 1
                else:
                    scores.append(logits.view(-1).tolist())
                if self.history_train == "gt" and self.training:
                    state = inputs[label[i]]
                if (self.history_train == "max" and self.training) \
                    or (self.history_val == "max" and not self.training):
                    state = inputs[logits.argmax()]
            if not self.training:
                meta["scores"] = scores
                results.append(meta)
        if self.training:
            return loss / count
        else:
            return results

class Attention(nn.Module):
    ''' AttentionPooling used to weighted aggregate news vectors
    Arg: 
        d_h: the last dimension of input
    '''
    def __init__(self,dim):
        super(Attention, self).__init__()
        self.key = nn.Linear(dim,dim)
        self.query = nn.Linear(dim,dim)
        nn.init.kaiming_normal_(self.key.weight)
        nn.init.zeros_(self.key.bias)
        nn.init.kaiming_normal_(self.query.weight)
        nn.init.zeros_(self.query.bias)

    def forward(self, query, x, attn_mask=None):
        bz = x.shape[0]
        query = self.query(query)
        key = self.key(x)
        
        alpha = torch.bmm(query.unsqueeze(1),key.permute(0,2,1))
        alpha = alpha - torch.max(alpha)
        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=2, keepdim=True) + 1e-8)

        x = torch.bmm(alpha, x)
        x = torch.reshape(x, (bz, -1))  # (bz, 400)
        return x

class Gate(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim = dim
        self.trans1 = nn.Linear(dim,dim)
        self.trans2 = nn.Linear(dim,dim)
        self.trans_all = nn.Linear(dim*2,dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.apply(self.init_weights)
        
    def init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            nn.init.xavier_normal_(module.weight)
    
    def forward(self,image_emb,text_emb):
        whole = torch.cat([image_emb,text_emb],1)
        z = self.sigmoid(self.trans_all(whole))
        image_emb = self.tanh(self.trans1(image_emb))
        text_emb = self.tanh(self.trans2(text_emb))
        emb = z*image_emb + (1-z)*text_emb
        return emb

class Q2A_Function(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.mlp_v = MLP(cfg.INPUT.DIM, cfg.INPUT.DIM)
        self.mlp_t = MLP(cfg.INPUT.DIM, cfg.INPUT.DIM)
        self.mlp_pre = MLP(cfg.INPUT.DIM*4, cfg.MODEL.DIM_STATE)

        self.button_gate = Gate(cfg.INPUT.DIM)
        self.attn = Attention(cfg.INPUT.DIM)

        if cfg.MODEL.TIMEEMB:
            self.timeemb = nn.Parameter(torch.randn((50,cfg.MODEL.DIM_STATE), device="cuda"))
            nn.init.normal_(self.timeemb,std=0.1)
        self.state = nn.Parameter(torch.randn(cfg.MODEL.DIM_STATE, device="cuda"))
        nn.init.normal_(self.state,std=0.1)
        if cfg.MODEL.HISTORY.ARCH == "mlp":
            self.proj = MLP(cfg.MODEL.DIM_STATE*2, 1)
        elif cfg.MODEL.HISTORY.ARCH == "gru":
            self.gru = nn.GRUCell(cfg.MODEL.DIM_STATE, cfg.MODEL.DIM_STATE)
            self.proj = MLP(cfg.MODEL.DIM_STATE, 1)
        else:
            assert False, "unknown arch"
        
        self.history_train = cfg.MODEL.HISTORY.TRAIN
        self.history_val = cfg.MODEL.HISTORY.VAL
        
        self.function_centric = cfg.MODEL.FUNCTION_CENTRIC
        self.cfg = cfg

    def forward(self, batch, p=False):
        loss, count = 0, 0
        results = []
        for video, script, question, para, actions, label, meta in batch:
            # for text
            if self.function_centric:
                score = torch.tensor(meta['paras_score']).softmax(dim=0).cuda()
                timestamps = meta['paras_timestamp']
                para = self.mlp_t(para)
                if self.cfg.MODEL.TIMEEMB:
                    para = para + self.timeemb[:para.shape[0],:]
                para = torch.matmul(score, para)
            else:
                score = torch.tensor(meta['sents_score']).softmax(dim=0).cuda()
                timestamps = meta['sents_timestamp']
                script = self.mlp_t(script)
                if self.cfg.MODEL.TIMEEMB:
                    para = para + self.timeemb[:para.shape[0],:]
                script = torch.matmul(score, script)
            text_seg = para if self.function_centric else script

            # for visual
            video = self.mlp_v(video)
            if self.cfg.MODEL.TIMEEMB:
                video = video + self.timeemb[:video.shape[0],:]
            # video_seg = torch.matmul(score, video)

            question = self.mlp_t(question)

            video_dynamic = self.attn(question,video.unsqueeze(0))

            video_seg = video_dynamic
            
            state = self.state
            scores = []
            for i, actions_per_step in enumerate(actions):
                a_texts, a_buttons = zip(*[(action['text'], action['button']) for action in actions_per_step])
                a_texts = self.mlp_t(torch.cat(a_texts))
                A = len(a_buttons)
                a_buttons = self.mlp_v(
                    torch.stack(a_buttons).view(A, -1, a_texts.shape[1])
                ).view(A, -1) 
                
                qa = question.repeat(A,1)

                a_buttons = self.button_gate(a_buttons,a_texts)

                inputs = torch.cat(
                    [video_seg.expand_as(qa), text_seg.expand_as(qa), qa.view(A, -1), a_buttons.view(A, -1)],
                    dim=1
                )

                inputs = self.mlp_pre(inputs)
                if hasattr(self, "gru"):
                    states = self.gru(inputs, state.expand_as(inputs))
                else:
                    states = torch.cat([inputs, state.expand_as(inputs)], dim=1)
                logits = self.proj(states)
                if self.training and not p:
                    loss += F.cross_entropy(logits.view(1, -1), label[i].view(-1))
                    count += 1
                elif p:
                    prob,psedo_label = torch.max(F.softmax(logits.detach(),0),0)
                    mask = prob.ge(0.9).float().view(-1)
                    loss += F.cross_entropy(logits.view(1, -1), psedo_label.view(-1))*mask
                    count += 1
                    #no confidence break
                    if mask==0:
                        break
                else:
                    scores.append(logits.view(-1).tolist())
                if self.history_train == "gt" and self.training and not p:
                    state = inputs[label[i]]
                if (self.history_train == "max" and self.training) \
                    or (self.history_val == "max" and not self.training) or p:
                    state = inputs[logits.argmax()]
            if not self.training:
                meta["scores"] = scores
                result = {}
                result['meta'] = meta
                if self.cfg.DATASET.GT:
                    result['label'] = label
                results.append(result)
        if self.training:
            return loss / count
        else:
            return results


models = {"q2a": Q2A, "q2a_function": Q2A_Function}

class ModelModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.model = models[cfg.MODEL.ARCH](cfg)
        self.cfg = cfg
    
    def training_step(self, batch, idx):
        labeled_data, unlabeled_data = batch
        loss = self.model(labeled_data)
        dataset = self.trainer.datamodule.__class__.__name__
        self.log(f"{dataset} loss", loss, rank_zero_only=True)

        p_loss = self.model(unlabeled_data,p=True)
        self.log(f"{dataset} pseudo loss", p_loss, rank_zero_only=True)
        sum_loss = loss+p_loss
        return sum_loss
    
    def configure_optimizers(self):
        cfg = self.cfg
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
        return [optimizer], []
    
    def validation_step(self, batch, idx):
        batched_results = self.model(batch)
        return batched_results
        
    def validation_epoch_end(self, results) -> None:
        from eval_for_loveu_cvpr2022 import evaluate
        results = sum(results,[])
        all_preds = {}
        all_annos = {}
        for result in results:
            pred = dict(
                question=result['meta']['question'], 
                scores=result['meta']['scores']
            )
            folder = result['meta']['folder']
            if folder not in all_preds:
                all_preds[folder] = []
                all_annos[folder] = []
            all_preds[folder].append(pred)
            if self.cfg.DATASET.GT:
                all_annos[folder].append(result['label'])

        if self.cfg.DATASET.GT:
            r1, r3, mr, mrr = evaluate(all_preds, all_annos)
            dataset = self.trainer.datamodule.__class__.__name__
            # for tensorboard
            self.log(f"{dataset} recall@1", r1, rank_zero_only=True)
            self.log(f"{dataset} recall@3", r3, rank_zero_only=True)
            self.log(f"{dataset} mean_rank", mr, rank_zero_only=True)
            self.log(f"{dataset} mrr", mrr)
            # for terminal
            print(f"{dataset} recall@1", r1)
            print(f"{dataset} recall@3", r3)
            print(f"{dataset} mean_rank", mr)
            print(f"{dataset} mrr", mrr) 
        else:
            json_name = f"submit_test.json"
            json_file = os.path.join(self.cfg.SAVEPATH, json_name)
            if not os.path.exists(self.cfg.SAVEPATH):
                os.makedirs(self.cfg.SAVEPATH)
            print("\n No ground-truth labels for validation \n")
            print(f"Generating json file at {json_file}. You can zip and submit it to CodaLab ;)")
            with open(json_file, 'w') as f:
                json.dump(all_preds, f)

def build_model(cfg):
    return ModelModule(cfg)
