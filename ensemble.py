import json
import numpy as np

def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    softmax_x = exp_x / sum_exp_x
    return softmax_x

fileroot = '/data/zclfe/cvpr_comp/LOVEU-CVPR22-AQTC/outputs/cvpr_loveu2023/'

exp_names = [
    # 'SSL0.9_layernorm_kaiming_statepara_lr1e-4_bs16_epoch100',
    # 'SSL_layernorm_kaiming_statepara_lr1e-4_bs16_epoch100',
    # 'noconfibreak_S3Dvideo_addtestSSL0.9_layernorm_kaiming_statepara_lr1e-4_bs16_epoch100',
    # 'S3Dvideo_addtestSSL0.9_layernorm_kaiming_statepara_lr1e-4_bs16_epoch100',
    # 'addtestSSL0.9_layernorm_kaiming_statepara_lr1e-4_bs16_epoch100',
    # 'splitqa_lr1e-4_bs16_epoch100',
    # 'layernorm_kaiming_statepara_timeemb_lr1e-4_bs16_epoch100',
    # 'layernorm_kaiming_statepara_lr1e-4_bs16_epoch100',
    # 'layernorm_kaiming_splitmlp_lr1e-4_bs16_epoch100',
    # 'layernorm_kaiming_sentence_lr1e-4_bs16_epoch100',
    # 'layernorm_kaiming_lr1e-4_bs16_epoch100',
    # 'agglearn_lr1e-4_bs16_epoch100',
    # 'baseline_lr1e-4_bs16_epoch100'
    'moe4_feastep_gate_statedim_32',
    'moe4_feastep_gate_statedim_64',
    'moe4_feastep_gate_statedim_128',
    'moe4_feastep_gate_statedim_256',
    'moe4_feastep_gate_statedim_512',
    'moe4_feastep_gate_statedim_1024',
    'moe4_stepid_gatefeature',
    'moe4_stepid',
    'moe4_stepid_gatefeature_statemap',
    'moe4_feature',
    'teacherforcelineardecay0_paralearn_idf',
    'teacherforcelineardecay0_paralearn',
    'teacherforcelineardecay0_both_paralearn_idf',
    'teacherforce0.8noconfibreak_facattn_dynamic_video_gatebutton_videoclip_addtestSSL0.9_layernorm_kaiming_statepara_lr1e-4_bs16_epoch100',
    'teacherforcelineardecay',
    'noconfibreak_facattn_dynamic_video_gatebutton_videoclip_addtestSSL0.9_layernorm_kaiming_statepara_lr1e-4_bs16_epoch100',
    'noconfibreak_dynamic_video_videoclip_addtestSSL0.9_layernorm_kaiming_statepara_lr1e-4_bs16_epoch100',
    'noconfibreak_dynamic_video_gatebutton_videoclip_addtestSSL0.9_layernorm_kaiming_statepara_lr1e-4_bs16_epoch100',
    'noconfibreak_dynamic_tfidf_video_videoclip_addtestSSL0.9_layernorm_kaiming_statepara_lr1e-4_bs16_epoch100',
    'noconfibreak_videoclip_addtestSSL0.9_layernorm_kaiming_statepara_lr1e-4_bs16_epoch100'
]

file_num = len(exp_names)
result = None

for name in exp_names:
    sub_name = fileroot+name+'/submit_test.json'
    with open(sub_name,'r') as fin:
        file = json.load(fin)
        if result is None:
            result = {}
            for mname in file:
                result[mname]=[]
                qa_list = file[mname]
                for qa in qa_list:
                    new_qa = {}
                    for k,v in qa.items():
                        if k=='question':
                            new_qa[k]=v
                        elif k=='scores':
                            new_qa[k]=[]
                            for step_a in v:
                                new_qa[k].append(np.zeros_like(np.array(step_a)))
                    result[mname].append(new_qa)

        for mname in file:
            qa_list = file[mname]
            for qa in qa_list:
                sum_all = result[mname]
                for sums in sum_all:
                    if sums['question'] == qa['question']:
                        for i, score in enumerate(qa['scores']):
                            score = np.array(score)
                            score = np.expand_dims(score, axis=0)
                            score = softmax(score)
                            score = np.squeeze(score,axis=0)
                            sums['scores'][i] += score/file_num
                        break

for mname in result:
    qa_list = result[mname]
    for qa in qa_list:
        for i in range(len(qa['scores'])):
            qa['scores'][i] = qa['scores'][i].tolist()


with open('submit_test.json','w')as f:
    json.dump(result,f)
