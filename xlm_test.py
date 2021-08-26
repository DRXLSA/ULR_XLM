







# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Translate sentences from the input stream.
# The model will be faster is sentences are sorted by length.
# Input sentences must have the same tokenization and BPE codes than the ones used in the model.
#
# Usage:
#     cat source_sentences.bpe | \
#     python translate.py --exp_name translate \
#     --src_lang en --tgt_lang fr \
#     --model_path trained_model.pth --output_path output
#

import os
import io
import sys
import argparse
import torch
import numpy as np
from xlm.utils import AttrDict
from xlm.utils import bool_flag, initialize_exp
from xlm.data.dictionary import Dictionary
from xlm.model.transformer import TransformerModel
from xlm.data.dictionary import BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD

from transfer_models import transfer_model
#from transfer_data import CorpusReader


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Translate sentences")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./dumped/", help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of sentences per batch")

    # model / output paths
    parser.add_argument("--model_path", type=str, default="", help="Model path")
    parser.add_argument("--transfer_model_path",type=str,default="")
    parser.add_argument("--output_path", type=str, default="", help="Output path")

    # parser.add_argument("--max_vocab", type=int, default=-1, help="Maximum vocabulary size (-1 to disable)")
    # parser.add_argument("--min_count", type=int, default=0, help="Minimum vocabulary count")

    # source language / target language
    parser.add_argument("--src_lang", type=str, default="", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="", help="Target language")

    return parser


def main(params):

    # initialize the experiment
    logger = initialize_exp(params)

    # generate parser / parse parameters

    

    reloaded = torch.load(params.model_path)
    model_params = AttrDict(reloaded['params'])
    for key in list(reloaded['model'].keys()):
        reloaded['model'][key[7:]] = reloaded['model'][key]
        del reloaded['model'][key]
    logger.info("Supported languages: %s" % ", ".join(model_params.lang2id.keys()))
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])

    # update dictionary parameters
    

    bos_index = dico.index(BOS_WORD)
    eos_index = dico.index(EOS_WORD)
    pad_index = dico.index(PAD_WORD)
    unk_index = dico.index(UNK_WORD)
    mask_index = dico.index(MASK_WORD)

    model_params['bos_index'] = bos_index
    model_params['eos_index'] = eos_index
    model_params['pad_index'] = pad_index
    model_params['unk_index'] = unk_index
    model_params['mask_index'] = mask_index

    for name in ['n_words', 'bos_index', 'eos_index', 'pad_index', 'unk_index', 'mask_index']:
        setattr(params, name, getattr(model_params, name))

    # build dictionary / build encoder / build decoder / reload weights
    
    encoder = TransformerModel(model_params, dico, is_encoder=True, with_output=True).cuda().eval()

    #decoder = TransformerModel(model_params, dico, is_encoder=False, with_output=True).cuda().eval()
    encoder.load_state_dict(reloaded['model'])
    #decoder.load_state_dict(reloaded['decoder'])
    params.src_id = model_params.lang2id[params.src_lang]
    params.tgt_id = model_params.lang2id[params.tgt_lang]

    
    
    src_file = open(params.src_file,'r',encoding='utf-8')
    tgt_file = open(params.tgt_file,'r',encoding='utf-8')

    loss_save = [[],[],[]]
    #构建transfer model、optimizer
    transfer = transfer_model(layer_num=1).cuda()
    if os.path.isfile(params.transfer_model_path):
        print("loading state dict...")
        state_dict = torch.load(params.transfer_model_path)
        transfer.load_state_dict(state_dict)
    optimizer = torch.optim.Adam(transfer.parameters(),lr=1e-4)
    criterion = torch.nn.CosineSimilarity(dim=2)

    criterion1 = torch.nn.CosineSimilarity(dim=1)

    criterion0 = torch.nn.CosineSimilarity()


    for i in range(1000):
        sents = []
        if i % 100 == 0 and i != 0:
            save_path = os.path.join(params.save_dir,"save_{}".format(i))
            torch.save(transfer.state_dict(),save_path)
        if i % 10 == 0 and i != 0:
            print("loss value: {} semantic_loss: {} cts_loss: {}".format(np.mean(loss_save[0]),np.mean(loss_save[1]),np.mean(loss_save[2])))
            loss_save = [[] for _ in range(3)]
        for j in range(int(params.batch_size/2)) :
            sents.append(src_file.readline())
            sents.append(tgt_file.readline())

        # prepare batch
        word_ids = [torch.LongTensor([dico.index(w) for w in s.strip().split()])
                    for s in sents]
        lengths = torch.LongTensor([len(s) + 2 for s in word_ids]).cuda()
        batch = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(params.pad_index).cuda()
        batch[0] = params.bos_index
        for j, s in enumerate(word_ids):
            if lengths[j] > 2:  # if sentence not empty
                batch[1:lengths[j] - 1, j].copy_(s)
            batch[lengths[j] - 1, j] = params.eos_index
        
        langs = torch.zeros_like(batch).cuda()
        for j in range(params.batch_size):
            if j%2 == 0:
                langs[:,j].fill_(params.src_id)
            else:
                langs[:,j].fill_(params.tgt_id)


        encoded = encoder('fwd', x=batch, lengths=lengths, langs=langs, causal=False).detach()
        encoded_tr = encoded.transpose(0, 1)  #batch,seq,hidden

        batch = batch.transpose(0,1)

        original_emb = encoder.embeddings.weight[batch].detach() #batch,seq,emb_dim

        
        
        
        #计算得到transfer后的向量
        transfer_enc = transfer(encoded_tr) #batch,seq,emb_dim
        #利用length来生成mask
        mask = torch.zeros_like(batch).cuda()
        for j in range(mask.shape[0]):
            mask[j][:lengths[j]] = 1
        #计算锚点损失
        batch_flatten = batch.flatten()
        transfer_enc_flatten = transfer_enc.view(-1,1024)
        mask_flatten = mask.flatten()
        transfer_enc_flatten_masked = transfer_enc_flatten[mask_flatten==1]
        original_emb_masked = original_emb.reshape(-1,1024)[mask_flatten==1]
        batch_flatten_masked = batch_flatten[mask_flatten==1]
        #semantic_loss = ( ( transfer_enc - original_emb )**2 ).sum(dim=2).sqrt() * mask 
        #选取original_emb作为max margin loss的负例,要求余弦距离cos(e,e(w'))最小，cos(e(w'),e(w))最大
        #和e最接近的10个w'中，选距离w最远的
        topk = 10
        gama = 0.5
        #对每个投影后的token，计算最接近的10个w'
        cts_emb = torch.empty_like(original_emb_masked).cuda()
        for j in range(len(transfer_enc_flatten_masked)):
            token = transfer_enc_flatten_masked[j]
            cos_distance = criterion1(encoder.embeddings.weight,token.repeat(encoder.embeddings.weight.shape[0],1))
            indices = cos_distance.topk(topk)[1].detach().cpu().numpy()
            
            if batch_flatten_masked[j].item() in indices:
                indices = set(indices)
                indices.remove(batch_flatten_masked[j].item())
                
                indices = list(indices)
            w = encoder.embeddings.weight[batch_flatten_masked[j]]
            from_w = [criterion0(encoder.embeddings.weight[ind].reshape(1,-1),w.reshape(1,-1)).item() for ind in indices]
            
            best_ind = indices[ np.argmin(from_w) ]
            cts_emb[j] = encoder.embeddings.weight[best_ind].detach()

        max_margin_loss =   gama + (1 - criterion1(transfer_enc_flatten_masked,cts_emb))  - (1 - criterion1(transfer_enc_flatten_masked,original_emb_masked) )

        margin_mask = torch.zeros_like(max_margin_loss).cuda()
        margin_mask[max_margin_loss < 0] = 1 

        max_margin_loss.masked_fill_(margin_mask.bool(),0)

        semantic_loss = - max_margin_loss




        semantic_loss = semantic_loss.sum() / ( semantic_loss !=0 ).sum().item()


        #差异性损失
        
        cts_loss = 0
        tokens_num = 0
        cts_loss_div = torch.tensor([0])
        """
        transfer_enc_tr = transfer_enc.transpose(0,1)

        
        for j in range(encoded.shape[0]):
            tensor_step = transfer_enc_tr[j].repeat([1,transfer_enc_tr.shape[0]]).reshape(-1,transfer_enc_tr.shape[0],transfer_enc_tr.shape[2])
            step_loss = criterion(tensor_step, transfer_enc) * mask * mask[:,j].reshape(-1,1)
            margin_mask = torch.zeros_like(step_loss)
            margin_mask[step_loss < 0] = 1
            step_loss.masked_fill_(margin_mask.bool(),0)
            tokens_num += (step_loss !=0 ).sum().item()
            cts_loss += step_loss.sum() 

        cts_loss_div = cts_loss.divide(tokens_num)
        """
        
        loss = semantic_loss #+  0.2 * cts_loss.divide(tokens_num)
        loss_val = loss.item()

        loss_save[0].append(loss_val)
        loss_save[1].append(semantic_loss.item())
        loss_save[2].append(cts_loss_div.item())

        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()

        if i % 10 == 0:
            print("----------------------",i,"---------------")
            
            sents = []
            sents = ["still",
                        "noch"] 
            
            
            #["His car was still running in the driveway .",
                        #"Sein Auto lief noch in der Ein@@ fahrt ."]
            # prepare batch
            word_ids = [torch.LongTensor([dico.index(w) for w in s.strip().split()])
                        for s in sents]
            lengths = torch.LongTensor([len(s) + 2 for s in word_ids]).cuda()
            batch = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(params.pad_index).cuda()
            batch[0] = params.bos_index
            for j, s in enumerate(word_ids):
                if lengths[j] > 2:  # if sentence not empty
                    batch[1:lengths[j] - 1, j].copy_(s)
                batch[lengths[j] - 1, j] = params.eos_index
            
            langs = torch.zeros_like(batch).cuda()
            for j in range(2):
                if j%2 == 0:
                    langs[:,j].fill_(params.src_id)
                else:
                    langs[:,j].fill_(params.tgt_id)


            encoded = encoder('fwd', x=batch, lengths=lengths, langs=langs, causal=False).detach()
            encoded_tr = encoded.transpose(0, 1)  #batch,seq,hidden

            batch = batch.transpose(0,1)


            original_emb = encoder.embeddings.weight[batch].detach() #batch,seq,emb_dim

            transfer_enc = transfer(encoded_tr) #batch,seq,emb_dim

            def sim(x,y):
                return ((x*y).sum()/torch.norm(x)/torch.norm(y)).item()
            len1,len2 = lengths[0],lengths[1]


            for p in range(1,len1-1):
                for q in range(1,len2-1):
                    print("{:.2f}".format(sim(transfer_enc[0][p],transfer_enc[1][q])),end=' ')
                print("")
            print("")

            for p in range(1,len1-1):
                for q in range(1,len1-1):
                    print("{:.2f}".format(sim(transfer_enc[0][p],transfer_enc[0][q])),end=' ')
                print("")
            print("")

            for p in range(1,len1-1):
                for q in range(1,len2-1):
                    print("{:.2f}".format(sim(encoded_tr[0][p],encoded_tr[1][q])),end=' ')
                print("")
            print("")

            for p in range(1,len1-1):
                for q in range(1,len1-1):
                    print("{:.2f}".format(sim(transfer_enc[0][p],original_emb[0][q])),end=' ')
                print("")
            print("")

            for p in range(1,len1-1):
                for q in range(1,len2-1):
                    print("{:.2f}".format(sim(original_emb[0][p],original_emb[1][q])),end=' ')
                print("")            
            




    src_file.close()
    tgt_file.close()

        






if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    params.exp_name = "xlm_test"
    params.dump_path = r"/data/lsj/XLM-master/dump_test"
    params.model_path = "dumped_xlm/xlm_de_en/5x1pwna5ik/periodic-9.pth"
    params.transfer_model_path = "transfer_dump/save_900"
    params.src_lang = "en"
    params.tgt_lang = "de"
    params.src_file = r"data/processed/de-en/test.de-en.en"
    params.tgt_file = r"data/processed/de-en/test.de-en.de"
    params.batch_size = 2
    params.save_dir = r"transfer_dump/"
    # check parameters
    #assert os.path.isfile(params.model_path)
    #assert params.src_lang != '' and params.tgt_lang != '' and params.src_lang != params.tgt_lang
    #assert params.output_path and not os.path.isfile(params.output_path)

    # translate
 
    main(params)

        

