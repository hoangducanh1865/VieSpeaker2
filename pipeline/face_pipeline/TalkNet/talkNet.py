import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, time, os, subprocess, pandas, tqdm

from model.talkNetModel import talkNetModel
from loss import lossAV, lossA, lossV


import math
import numpy as np
import torch
import python_speech_features
import cv2

debug_path = "pipeline/face_pipeline/debug"

class talkNet(nn.Module):
    def __init__(self, lr = 0.0001, lrDecay = 0.95, **kwargs):
        super(talkNet, self).__init__()        
        self.model = talkNetModel().cuda()
        self.lossAV = lossAV().cuda()
        self.lossA = lossA().cuda()
        self.lossV = lossV().cuda()
        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))

    def train_network(self, loader, epoch, **kwargs):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']        
        for num, (audioFeature, visualFeature, labels) in enumerate(loader, start=1):
            self.zero_grad()
            audioEmbed = self.model.forward_audio_frontend(audioFeature[0].cuda()) # feedForward
            visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda())
            audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
            outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  
            outsA = self.model.forward_audio_backend(audioEmbed)
            outsV = self.model.forward_visual_backend(visualEmbed)
            labels = labels[0].reshape((-1)).cuda() # Loss
            nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels)
            nlossA = self.lossA.forward(outsA, labels)
            nlossV = self.lossV.forward(outsV, labels)
            nloss = nlossAV + 0.4 * nlossA + 0.4 * nlossV
            loss += nloss.detach().cpu().np()
            top1 += prec
            nloss.backward()
            self.optim.step()
            index += len(labels)
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
            " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), 100 * (top1/index)))
            sys.stderr.flush()  
        sys.stdout.write("\n")      
        return loss/num, lr

    def evaluate_network(self, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        predScores = []
        for audioFeature, visualFeature, labels in tqdm.tqdm(loader):
            with torch.no_grad():                
                audioEmbed  = self.model.forward_audio_frontend(audioFeature[0].cuda())
                visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda())
                audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
                outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  
                labels = labels[0].reshape((-1)).cuda()             
                _, predScore, _, _ = self.lossAV.forward(outsAV, labels)    
                predScore = predScore[:,1].detach().cpu().np()
                predScores.extend(predScore)
        evalLines = open(evalOrig).read().splitlines()[1:]
        labels = []
        labels = pandas.Series( ['SPEAKING_AUDIBLE' for line in evalLines])
        scores = pandas.Series(predScores)
        evalRes = pandas.read_csv(evalOrig)
        evalRes['score'] = scores
        evalRes['label'] = labels
        evalRes.drop(['label_id'], axis=1,inplace=True)
        evalRes.drop(['instance_id'], axis=1,inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        cmd = "python -O utils/get_ava_active_speaker_performance.py -g %s -p %s "%(evalOrig, evalCsvSave)
        mAP = float(str(subprocess.run(cmd, shell=True, capture_output =True).stdout).split(' ')[2][:5])
        return mAP

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name;
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)


    def predict(self, audio_waveform, video_frames, sample_rate=16000, duration_set=(1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6)):
        """
        End-to-end inference for a single track.
        
        Args:
            audio_waveform (np.array): 1D array of the audio track.
            video_frames (list or np.array): List of cropped face images (BGR format).
            sample_rate (int): The audio sample rate (default 16000).
            duration_set (tuple): Temporal windows to average over for reliable results.
            
        Returns:
            np.array: 1D array of probability scores (0.0 to 1.0) per video frame.
        """
        self.eval() # Ensure model is in evaluation mode

        # 1. Extract Audio Features (MFCC)
        audioFeature = python_speech_features.mfcc(
            audio_waveform, sample_rate, numcep=13, winlen=0.025, winstep=0.010
        )
        
        # 2. Process Video Features
        processed_video = []
        for frame in video_frames:
            face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Just resize directly to 112x112. Do NOT double-crop.
            face = cv2.resize(face, (112, 112))
            processed_video.append(face)
        videoFeature = np.array(processed_video)

        # 3. Align Audio and Video Lengths
        # TalkNet expects 100 audio frames per second and 25 video frames per second (4:1 ratio)
        length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0] / 25)
        
        if length <= 0:
             return np.array([]) # Sequence too short

        audioFeature = audioFeature[:int(round(length * 100)), :]
        videoFeature = videoFeature[:int(round(length * 25)), :, :]

        # 4. Run Inference over different duration windows
        allScores = []
        for duration in duration_set:
            batchSize = int(math.ceil(length / duration))
            scores = []
            
            with torch.no_grad():
                for i in range(batchSize):
                    # Slice the features into batches based on duration
                    start_a = i * duration * 100
                    end_a = (i + 1) * duration * 100
                    start_v = i * duration * 25
                    end_v = (i + 1) * duration * 25

                    inputA = torch.FloatTensor(audioFeature[start_a:end_a, :]).unsqueeze(0).cuda()
                    inputV = torch.FloatTensor(videoFeature[start_v:end_v, :, :]).unsqueeze(0).cuda()

                    # Forward Pass
                    embedA = self.model.forward_audio_frontend(inputA)
                    embedV = self.model.forward_visual_frontend(inputV)    
                    embedA, embedV = self.model.forward_cross_attention(embedA, embedV)
                    out = self.model.forward_audio_visual_backend(embedA, embedV)
                    
                    # Pass through classification head to get probability scores
                    score = self.lossAV.forward(out, labels=None)
                    
                    # Depending on how lossAV is structured, it might return logits or probabilities.
                    # Usually, TalkNet's lossAV returns the softmax probability of the positive class (class 1).
                    if isinstance(score, tuple):
                         score = score[0] # Handle if it returns loss alongside scores
                         
                    # Move to CPU and convert to list
                    scores.extend(score.tolist())
                    
            allScores.append(scores)

        # 5. Average the scores across all duration tests
        # This smooths out anomalies from processing windows of different sizes
        final_scores = np.round((np.mean(np.array(allScores), axis=0)), 3).astype(float)
        
        return final_scores
