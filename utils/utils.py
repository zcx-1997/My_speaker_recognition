#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/24 16:03
    Author  : 春晓
    Software: PyCharm
"""

import librosa
import numpy as np
import torch
from torch.nn import functional as F

from hparam import hparam as hp


def mfccs_and_spec(wav_file,wav_process=True,calc_mfccs=False,calc_mag_db=False):
    audio,_ = librosa.load(wav_file,sr=hp.data.sr)
    frame_len = int(hp.data.len_frame * hp.data.sr)
    frame_hop = int(hp.data.hop_frame* hp.data.sr)
    duration = hp.data.tisv_frame * hp.data.hop_frame + hp.data.len_frame

    if wav_process == True:
        audio,index = librosa.effects.trim(audio,frame_length=frame_len,hop_length=frame_hop)
        length = int(duration*hp.data.sr)
        audio = librosa.util.fix_length(audio,length)

    spec = librosa.stft(audio,n_fft=hp.data.nfft,hop_length=frame_hop,win_length=frame_len)
    mag_spec = np.abs(spec) ** 2
    mel_basis = librosa.filters.mel(hp.data.sr,hp.data.nfft,n_mels=hp.data.nmels)
    mel_spec = np.dot(mel_basis,mag_spec)

    mag_db = librosa.amplitude_to_db(mag_spec)
    mel_db = librosa.amplitude_to_db(mel_spec).T

    mfccs = None
    if calc_mfccs:
        mfccs = np.dot(librosa.filters.dct(40,mel_db.shape[0]),mel_db).T

    return mfccs, mel_db, mag_db

def get_centroids(x):
    centroids = x.mean(dim=1)
    return centroids

#未明白
def get_utterance_centroids(embeddings):
    """
    Returns the centroids for each utterance of a speaker, where
    the utterance centroid is the speaker centroid without considering
    this utterance

    Shape of embeddings should be:
        (speaker_ct, utterance_per_speaker_ct, embedding_size)
    """
    sum_centroids = embeddings.sum(dim=1)
    # we want to subtract out each utterance, prior to calculating the
    # the utterance centroid
    sum_centroids = sum_centroids.reshape(
        sum_centroids.shape[0], 1, sum_centroids.shape[-1]
    )
    # we want the mean but not including the utterance itself, so -1
    num_utterances = embeddings.shape[1] - 1
    centroids = (sum_centroids - embeddings) / num_utterances
    return centroids

#未明白
def get_cossim(embeddings, centroids):
    # number of utterances per speaker
    num_utterances = embeddings.shape[1]
    utterance_centroids = get_utterance_centroids(embeddings)

    # flatten the embeddings and utterance centroids to just utterance,
    # so we can do cosine similarity
    utterance_centroids_flat = utterance_centroids.view(
        utterance_centroids.shape[0] * utterance_centroids.shape[1],
        -1
    )
    embeddings_flat = embeddings.view(
        embeddings.shape[0] * num_utterances,
        -1
    )
    # the cosine distance between utterance and the associated centroids
    # for that utterance
    # this is each speaker's utterances against his own centroid, but each
    # comparison centroid has the current utterance removed
    cos_same = F.cosine_similarity(embeddings_flat, utterance_centroids_flat)

    # now we get the cosine distance between each utterance and the other speakers'
    # centroids
    # to do so requires comparing each utterance to each centroid. To keep the
    # operation fast, we vectorize by using matrices L (embeddings) and
    # R (centroids) where L has each utterance repeated sequentially for all
    # comparisons and R has the entire centroids frame repeated for each utterance
    centroids_expand = centroids.repeat((num_utterances * embeddings.shape[0], 1))
    embeddings_expand = embeddings_flat.unsqueeze(1).repeat(1, embeddings.shape[0], 1)
    embeddings_expand = embeddings_expand.view(
        embeddings_expand.shape[0] * embeddings_expand.shape[1],
        embeddings_expand.shape[-1]
    )
    cos_diff = F.cosine_similarity(embeddings_expand, centroids_expand)
    cos_diff = cos_diff.view(
        embeddings.size(0),
        num_utterances,
        centroids.size(0)
    )
    # assign the cosine distance for same speakers to the proper idx
    same_idx = list(range(embeddings.size(0)))
    cos_diff[same_idx, :, same_idx] = cos_same.view(embeddings.shape[0], num_utterances)
    cos_diff = cos_diff + 1e-6
    return cos_diff

#未明白
def calc_loss(sim_matrix):
    same_idx = list(range(sim_matrix.size(0)))
    pos = sim_matrix[same_idx, :, same_idx]
    neg = (torch.exp(sim_matrix).sum(dim=2) + 1e-6).log_()
    per_embedding_loss = -1 * (pos - neg)
    loss = per_embedding_loss.sum()
    return loss, per_embedding_loss

if __name__ == '__main__':
    # file = '../data/0-TIMIT/TEST/DR1/FAKS0/SA1.WAV'
    # mfccs_and_spec(file)

    x = torch.tensor([[1.0,2,3,4],[5,6,7,8]])
    y = get_centroids(x)
    print(y)




