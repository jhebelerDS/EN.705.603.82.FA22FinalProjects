'''This module contains the Transformer model and all the components that make it up.
This is using the PyTorch Lightning framework, which is a wrapper around PyTorch.
Using parts from https://github.com/jaymody/seq2seq-polynomial/blob/master/train.py'''

import os
import pickle
import argparse

import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import WandbLogger # wandb logger
# from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

from layers import Encoder, Decoder
from utils import get_device, score
from data import PolynomialLanguage, train_test_split

device = get_device()
BATCH_SIZE = 128

class Collater:
    '''Pass in class instance of src_lang and trg_lang, and a boolean value for predict mode.
    In predict mode, the batch is a list of tensors, and the function returns a padded tensor.
    In train mode, the batch is a list of tuples, and the function returns a tuple of padded tensors.'''
    def __init__(self, src_lang, trg_lang=None, predict=False):
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.predict = predict

    def __call__(self, batch):
        if self.predict:
            # batch = src_tensors in predict mode
            return nn.utils.rnn.pad_sequence(
                batch, batch_first=True, padding_value=self.src_lang.PAD_idx
            )

        src_tensors, trg_tensors = zip(*batch)
        src_tensors = nn.utils.rnn.pad_sequence(
            src_tensors, batch_first=True, padding_value=self.src_lang.PAD_idx
        )
        trg_tensors = nn.utils.rnn.pad_sequence(
            trg_tensors, batch_first=True, padding_value=self.trg_lang.PAD_idx
        )
        return src_tensors, trg_tensors


def sentence_to_tensor(sentence, lang):
    '''params: sentence is a list of tokens, lang is a Language instance
    returns: a tensor of shape (len(sentence), 1)'''
    indexes = [lang.word2index[w] for w in lang.sentence_to_words(sentence)] # a list of values based on the value of the key(word) in dict
    indexes = [lang.SOS_idx] + indexes + [lang.EOS_idx] # add SOS and EOS to the list: ex: 1 + [20,5,27,..] + 2 = [1,20,5,27,..,2]
    return torch.LongTensor(indexes) # convert to tensor - 64 bit-signed


def pairs_to_tensors(pairs, src_lang, trg_lang):
    '''params: pairs is a list of tuples, src_lang and trg_lang are Language instances
    calls sent'''
    tensors = [ # convert each sentence to tensor, src and trg
        (sentence_to_tensor(src, src_lang), sentence_to_tensor(trg, trg_lang)) # a tuple of ( [ src = [1,2,3,4,5], target = [1,2,3,4,5] ] )
        for src, trg in tqdm(pairs, desc="creating tensors") # for each pair in pairs
    ]
    return tensors # list of tuples of (src, trg) where each is a LongTensor


class SimpleDataset(Dataset):
    '''This is only needed for the PyTorch Lightning Trainer to work with the DataLoader.'''
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class Transformer(pl.LightningModule):
    '''This is the Transformer model using the PyTorch Lightning framework.'''
    def __init__(
        self,
        src_lang,
        trg_lang,
        max_len=32,
        hid_dim=256,
        enc_layers=3,
        dec_layers=3,
        enc_heads=8,
        dec_heads=8,
        enc_pf_dim=512,
        dec_pf_dim=512,
        enc_dropout=0.1,
        dec_dropout=0.1,
        lr=0.0005,
        device=device,
    ):
        super().__init__()

        self.save_hyperparameters()
        del self.hparams["src_lang"]
        del self.hparams["trg_lang"]

        self.src_lang = src_lang
        self.trg_lang = trg_lang

        self.encoder = Encoder(
            src_lang.n_words,
            hid_dim,
            enc_layers,
            enc_heads,
            enc_pf_dim,
            enc_dropout,
            device,
        )

        self.decoder = Decoder(
            trg_lang.n_words,
            hid_dim,
            dec_layers,
            dec_heads,
            dec_pf_dim,
            dec_dropout,
            device,
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.trg_lang.PAD_idx)
        self.initialize_weights()
        self.to(device)

    def initialize_weights(self):
        def _initialize_weights(m):
            if hasattr(m, "weight") and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)

        self.encoder.apply(_initialize_weights)
        self.decoder.apply(_initialize_weights)

    def make_src_mask(self, src):

        # src = [batch size, src len]

        src_mask = (src != self.src_lang.PAD_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):

        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_lang.PAD_idx).unsqueeze(1).unsqueeze(2)

        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len)).type_as(trg)).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg):

        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)

        # enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, attention

    def predict(self, sentences, batch_size=128):
        """Efficiently predict a list of sentences"""
        pred_tensors = [
            sentence_to_tensor(sentence, self.src_lang)
            for sentence in tqdm(sentences, desc="creating prediction tensors")
        ]

        collate_fn = Collater(self.src_lang, predict=True)
        pred_dataloader = DataLoader(
            SimpleDataset(pred_tensors),
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

        sentences = []
        words = []
        attention = []
        for batch in tqdm(pred_dataloader, desc="predict batch num"):
            preds = self.predict_batch(batch.to(device))
            pred_sentences, pred_words, pred_attention = preds
            sentences.extend(pred_sentences)
            words.extend(pred_words)
            attention.extend(pred_attention)

        # sentences = [num pred sentences]
        # words = [num pred sentences, trg len]
        # attention = [num pred sentences, n heads, trg len, src len]

        return sentences, words, attention
    
    def predict_single(self, sentence):
        """Predict a single sentence without batching."""
        tensor = sentence_to_tensor(sentence, self.src_lang)
        tensor = tensor.unsqueeze(0)
        preds = self.predict_batch(tensor.to(device))
        pred_sentences, pred_words, pred_attention = preds
        return pred_sentences[0], pred_words[0], pred_attention[0]

    def predict_batch(self, batch):
        """Predicts on a batch of src_tensors."""
        # batch = src_tensor when predicting = [batch_size, src len]

        src_tensor = batch
        src_mask = self.make_src_mask(batch)

        # src_mask = [batch size, 1, 1, src len]

        enc_src = self.encoder(src_tensor, src_mask)

        # enc_src = [batch size, src len, hid dim]

        trg_indexes = [[self.trg_lang.SOS_idx] for _ in range(len(batch))]

        # trg_indexes = [batch_size, cur trg len = 1]

        trg_tensor = torch.LongTensor(trg_indexes).to(self.device)

        # trg_tensor = [batch_size, cur trg len = 1]
        # cur trg len increases during the for loop up to the max len

        for _ in range(self.hparams.max_len):

            trg_mask = self.make_trg_mask(trg_tensor)

            # trg_mask = [batch size, 1, cur trg len, cur trg len]

            output, attention = self.decoder(trg_tensor, enc_src, trg_mask, src_mask)

            # output = [batch size, cur trg len, output dim]

            preds = output.argmax(2)[:, -1].reshape(-1, 1)

            # preds = [batch_size, 1]

            trg_tensor = torch.cat((trg_tensor, preds), dim=-1)

            # trg_tensor = [batch_size, cur trg len], cur trg len increased by 1

        src_tensor = src_tensor.detach().cpu().numpy()
        trg_tensor = trg_tensor.detach().cpu().numpy()
        attention = attention.detach().cpu().numpy()

        pred_words = []
        pred_sentences = []
        pred_attention = []
        for src_indexes, trg_indexes, attn in zip(src_tensor, trg_tensor, attention):
            # trg_indexes = [trg len = max len (filled with eos if max len not needed)]
            # src_indexes = [src len = len of longest sentence (padded if not longest)]

            # indexes where first eos tokens appear
            src_eosi = np.where(src_indexes == self.src_lang.EOS_idx)[0][0]
            _trg_eosi_arr = np.where(trg_indexes == self.trg_lang.EOS_idx)[0]
            if len(_trg_eosi_arr) > 0:  # check that an eos token exists in trg
                trg_eosi = _trg_eosi_arr[0]
            else:
                trg_eosi = len(trg_indexes)

            # cut target indexes up to first eos token and also exclude sos token
            trg_indexes = trg_indexes[1:trg_eosi]

            # attn = [n heads, trg len=max len, src len=max len of sentence in batch]
            # we want to keep n heads, but we'll cut trg len and src len up to
            # their first eos token
            attn = attn[:, :trg_eosi, :src_eosi]  # cut attention for trg eos tokens

            words = [self.trg_lang.index2word[index] for index in trg_indexes]
            sentence = self.trg_lang.words_to_sentence(words)
            pred_words.append(words)
            pred_sentences.append(sentence)
            pred_attention.append(attn)

        # pred_sentences = [batch_size]
        # pred_words = [batch_size, trg len]
        # attention = [batch size, n heads, trg len (varies), src len (varies)]

        return pred_sentences, pred_words, pred_attention

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        src, trg = batch

        output, _ = self(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = self.criterion(output, trg)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src, trg = batch

        output, _ = self(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = self.criterion(output, trg)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


def train(model, train_dataloader, val_dataloader, val_check_interval, max_epochs, clip, model_path, fast_dev_run):

    # logger = pl.loggers.TensorBoardLogger("tb_logs", log_graph=True)
    # logger = WandbLogger(name="hid_dim=256, layers=3, batch_size=256", project="system_project")

    checkpoint_callback = ModelCheckpoint(monitor="val_loss",
                                        dirpath=model_path,
                                        filename="model",
                                        save_top_k=1,
                                        mode="min")

    trainer = pl.Trainer(accelerator='gpu',
                        devices=1,
                        max_epochs=max_epochs,
                        val_check_interval=val_check_interval,
                        precision=16,
                        enable_progress_bar=True,
                        enable_model_summary=True,
                        gradient_clip_val=clip,
                        callbacks=[checkpoint_callback],
                        fast_dev_run=fast_dev_run) # add logger=logger if used

    trainer.fit(model, train_dataloader, val_dataloader)

    model = model.to(device) # need to remap to device after training

    return model


def evaluate(model, test_pairs, batch_size=128, print_examples=True):
    src_sentences, trg_sentences = zip(*test_pairs)

    prd_sentences, _, _ = model.predict(src_sentences, batch_size=batch_size)
    assert len(prd_sentences) == len(src_sentences) == len(trg_sentences)

    total_score = 0
    for i, (src, trg, prd) in enumerate(
        tqdm(
            zip(src_sentences, trg_sentences, prd_sentences),
            desc="scoring",
            total=len(src_sentences),
        )
    ):
        pred_score = score(trg, prd)
        total_score += pred_score
        if print_examples:
            if i < 10:
                print(f"\n---- Test Case {i} ----")
                print(f"Input = {src}")
                print(f"Target = {trg}")
                print(f"Predicted = {prd}")
                print(f"score = {pred_score}")

    final_score = total_score / len(prd_sentences)

    print('-'*50)
    print('Number Correct: ', total_score)
    print('Number of Examples: ', len(prd_sentences))
    print(f"Final Score = {final_score:.4f}")
    return final_score


def load_model(dirpath='models', model_ckpt="model_red_default.ckpt"):
    with open(os.path.join(dirpath, "src_lang.pickle"), "rb") as file_in:
        src_lang = pickle.load(file_in)
    with open(os.path.join(dirpath, "trg_lang.pickle"), "rb") as file_in:
        trg_lang = pickle.load(file_in)
    model = Transformer.load_from_checkpoint(
        os.path.join(dirpath, model_ckpt),
        src_lang=src_lang,
        trg_lang=trg_lang,
    ).to(device)
    return model


def predict_single_sentence(model, sentence):
    model.eval()
    with torch.no_grad():
        pred_sentences, _, _ = model.predict_single(sentence)
    return pred_sentences


def prepare_dataloaders():
    # Setting path variables
    train_set_path = "inputs/train_set.txt"
    model_path = "models/"

    train_set_pairs = PolynomialLanguage.load_pairs(train_set_path) # loading train set

    # Static hyperparameters for train vs validation split
    ratio = 0.95
    batch_size = 128
    num_workers = 8

    src_lang, trg_lang = PolynomialLanguage.create_vocabs(train_set_pairs) # creating source and target language objects
    train_pairs, val_pairs = train_test_split(train_set_pairs, ratio=ratio) # split for validation
    train_tensors = pairs_to_tensors(train_pairs, src_lang, trg_lang) # converting train pairs to tensors
    val_tensors = pairs_to_tensors(val_pairs, src_lang, trg_lang) # converting val pairs to tensors

    save_to_pickle = {"src_lang.pickle": src_lang, "trg_lang.pickle": trg_lang,}

    for k, v in save_to_pickle.items(): # saving source and target language objects
        with open(os.path.join(model_path, k), "wb") as file_out:
            pickle.dump(v, file_out)

    collate_fn = Collater(src_lang, trg_lang) # initializing collate function

    # the data is loaded in batches and then iterated over
    train_dataloader = DataLoader(SimpleDataset(train_tensors), # SimpleDataset is a custom dataset class
                                batch_size=batch_size, # batch size
                                collate_fn=collate_fn, # collate function
                                num_workers=num_workers) # number of parallel processes

    val_dataloader = DataLoader(SimpleDataset(val_tensors),
                                batch_size=batch_size,
                                collate_fn=collate_fn,
                                num_workers=num_workers)

    return train_dataloader, val_dataloader, src_lang, trg_lang


def train_from_script(accelerator, train_dataloader, val_dataloader, max_epochs, model_path, fast_dev_run, src_lang, trg_lang):
    '''This is only called if this script is ran as __main__'''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transformer(src_lang=src_lang, trg_lang=trg_lang, device=device).to(device) # initializing model

    checkpoint_callback = ModelCheckpoint(monitor="val_loss",
                                    dirpath=model_path,
                                    filename="model",
                                    save_top_k=1,
                                    mode="min")

    trainer = pl.Trainer(accelerator=accelerator,
                        devices=1,
                        max_epochs=max_epochs,
                        val_check_interval=0.2,
                        enable_progress_bar=True,
                        enable_model_summary=True,
                        fast_dev_run=fast_dev_run,
                        callbacks=[checkpoint_callback],
                        logger=False)

    trainer.fit(model, train_dataloader, val_dataloader)

    model = model.to(device) # need to remap to device after training

    return model


def main():
    '''Parse arguments for training'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast_dev_run', action='store_true', help='Run a fast dev run for testing', default=False)
    parser.add_argument('--max_epochs', type=int, help='Number of epochs to train for, use wisely', default=10)
    args = parser.parse_args()

    train_dataloader, val_dataloader, src_lang, trg_lang = prepare_dataloaders()

    model = train_from_script(accelerator='gpu',
                             train_dataloader=train_dataloader,
                             val_dataloader=val_dataloader,
                             max_epochs=args.max_epochs,
                             model_path="trained_from_script/",
                             fast_dev_run=args.fast_dev_run,
                             src_lang=src_lang,
                             trg_lang=trg_lang)
    if not model:
        print("Model not trained, exiting...")
        return
    
    print("Training complete")
    print('Model saved to trained_from_script/')

    prompt = input("Would you like to test the model? (y/n): ")

    if prompt.lower() == "y":
        print("Testing model on test set")

        test_set_path = "inputs/test_set.txt"
        test_set_pairs = PolynomialLanguage.load_pairs(test_set_path)

        evaluate(model, test_set_pairs)

    print('To predict on a single custom input sequence, run main_script.py')
    print('Exiting...')

if __name__ == "__main__":
    main()
