import numpy as np
import torch
import os, sys
import re
from torch.utils.data import Dataset, DataLoader
import math

#tensorboard for accuracy graphs
import tensorflow as tf

def getCombinations(inputTensor, N, c, d):#input shape=(batch_size, obj_count, obj_dim) #batch_size=N, obj_count=c, obj_dim=d
    tensorA = inputTensor.reshape(N, 1, c, d).expand(N, c, c, d)
    tensorB = tensorA.transpose(1, 2)

    return torch.cat((tensorB, tensorA), 3)

devices = (torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:2"), torch.device("cuda:3"))

if len(sys.argv) < 2:
    print("Missing data path!")
    exit()

dataset_name = 'neutral'#'interpolation'#'extrapolation'
datapath = os.path.join(sys.argv[1],dataset_name)

class PgmDataset(Dataset):
    def __init__(self, filenames):
        'Initialization'
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        with np.load(os.path.join(datapath, filename)) as data:
            image = data['image'].astype(np.uint8).reshape(16, 160, 160)[:,::2,::2]
            target = data['target']
            meta = data['relation_structure']
        return image, target, meta

def custom_collate_fn(batch):
    images, targets, metas = zip(*batch)
    images = torch.stack([torch.from_numpy(b) for b in images], 0)
    targets = torch.stack([torch.from_numpy(b) for b in targets], 0)
    return images, targets, metas

batch_size = 32

all_data = os.listdir(datapath)
test_filenames = [p for p in all_data if re.match(r'^PGM_' + re.escape(dataset_name) + r'_test_(\d+)\.npz$', p) is not None]
test_dataloader = DataLoader(PgmDataset(test_filenames), batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, collate_fn=custom_collate_fn)

class WReN(torch.nn.Module):
    def __init__(self, m):
        super(WReN, self).__init__()
        self.relation_network_depth = m

        self.g_dim = 512
        self.h_dim = 256
        self.f_dim = 256

        self.use_mag_enc = True #switch between scalar input and magnitude encoded input
        self.mag_enc_type_relu = False #switch between gaussian magnitude encoding and relu based magnitude encoding

        self.magnitude_encoding_dim = 20
        #model
        #magnitude encoding
        self.input_scale = 2.0/255.0
        self.input_offset = -1.0
        #self.input_encoding_variance_inv = torch.nn.Parameter(torch.tensor(self.magnitude_encoding_dim * 0.5))
        std_dev = 0.28
        self.input_encoding_variance_inv = 1.0 / (math.sqrt(2.0) * std_dev)
        #self.normalization_factor = torch.nn.Parameter(torch.tensor(1.0 / (math.sqrt(2*math.pi) * std_dev)))
        self.normalization_factor = 1.0 / (math.sqrt(2*math.pi) * std_dev)
        self.mag_scale = torch.nn.Parameter(torch.linspace(-1.0, 1.0, steps=self.magnitude_encoding_dim), requires_grad=False)

        if self.use_mag_enc:
            conv_input_dim = self.magnitude_encoding_dim
        else:
            conv_input_dim = 1

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(conv_input_dim, 32, 3, stride=2), 
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 32, 3, stride=2), 
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 32, 3, stride=2), 
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 32, 3, stride=2), 
            torch.nn.LeakyReLU()
        )
        self.post_cnn_linear = torch.nn.Linear(32*4*4, 256-9)#input = 32 feature maps with 4x4 resolution

        self.tag_matrix = torch.nn.Parameter(torch.eye(9).repeat(8, 1), requires_grad=False)

        self.g = torch.nn.Sequential(
                torch.nn.Linear(2*256, self.g_dim), 
                torch.nn.LeakyReLU(),
                torch.nn.Linear(self.g_dim, self.g_dim), 
                torch.nn.LeakyReLU(),
                torch.nn.Linear(self.g_dim, self.g_dim), 
                torch.nn.LeakyReLU(),
                torch.nn.Linear(self.g_dim, self.h_dim),
                torch.nn.LeakyReLU()
            )

        h = []
        for i in range(m):
            rel_layer_func = torch.nn.Sequential(
                torch.nn.Linear(2*self.h_dim, self.h_dim), 
                torch.nn.LeakyReLU(),
                torch.nn.Linear(self.h_dim, self.h_dim), 
                torch.nn.LeakyReLU(),
                torch.nn.Linear(self.h_dim, self.h_dim), 
                torch.nn.LeakyReLU()
            )
            h.append(rel_layer_func)

        self.h = torch.nn.ModuleList(h)

        f_in_dim = self.h_dim
        self.f = torch.nn.Sequential(
                torch.nn.Linear(f_in_dim, self.f_dim), 
                torch.nn.LeakyReLU(),
                torch.nn.Linear(self.f_dim, self.f_dim), 
                torch.nn.LeakyReLU()
            )

        self.f_final = torch.nn.Linear(self.f_dim, 1)
        

    def forward(self, batch):
        batch_size = batch.size()[0]
        #Panel preprocessor CNN
        batch_flat = batch.reshape(batch_size*16, 1, 80, 80)#16 images per sample: 8 for context + 8 answer options

        if self.use_mag_enc:
            with torch.no_grad():
                #magnitude encoding
                batch_flat = batch_flat.transpose(1, 3)
                if self.mag_enc_type_relu:
                    #first order
                    batch_flat = batch_flat.add_(255/self.magnitude_encoding_dim)
                    batch_flat = torch.nn.functional.relu_(batch_flat.mul_(self.input_scale).add_(self.input_offset).add(-self.mag_scale))
                    #second order
                    batch_flat = torch.cat((batch_flat[:, :, :, :-1] - 2*batch_flat[:, :, :, 1:], batch_flat[:, :, :, -1].unsqueeze(dim=-1)), dim=-1).mul_(self.magnitude_encoding_dim/2)
                    batch_flat = torch.nn.functional.relu_(batch_flat)
                else:
                    batch_flat = batch_flat.mul_(self.input_scale).add_(self.input_offset).tanh_().add(self.mag_scale).mul_(self.input_encoding_variance_inv).pow_(2).mul_(-1).exp_().mul_(self.normalization_factor)
                batch_flat = batch_flat.transpose(3, 1)

        conv_out = self.conv(batch_flat)
        #scatter context
        objectsWithoutPos = self.post_cnn_linear(conv_out.reshape(batch_size*16, -1))
        panel_vectors = objectsWithoutPos.reshape(batch_size, 16, 256-9)
        given, option1, option2, option3, option4, option5, option6, option7, option8 = panel_vectors.split((8, 1, 1, 1, 1, 1, 1, 1, 1), dim=1)
        optionsWithContext = torch.cat((
            given, option1, 
            given, option2, 
            given, option3, 
            given, option4, 
            given, option5, 
            given, option6, 
            given, option7, 
            given, option8
        ), 1)
        optionsWithoutPos = optionsWithContext.reshape(batch_size*8*9, 256-9)

        objects = torch.cat((optionsWithoutPos, self.tag_matrix.repeat(batch_size, 1)), dim=1).reshape(batch_size*8, 9, 256-9+9)#8 answers to score per sample, 9 images (8 from context + 1 from answer) per answer option

        #MLRN
        objPairs2D = getCombinations(objects, batch_size*8, 9, 256)
        objPairs = objPairs2D.reshape(batch_size*8*(9*9), 2*256)

        gResult = self.g(objPairs)#apply MLP

        prev_result = gResult
        prev_dim = self.h_dim
        prev_result_2d = prev_result.reshape(batch_size*8, 9, 9, prev_dim)
        sum_j = prev_result_2d.sum(dim=2)
        for i, h_layer in enumerate(self.h):
            intermed_obj_pairs_2d = getCombinations(sum_j, batch_size*8, 9, prev_dim)
            intermed_obj_pairs = intermed_obj_pairs_2d.reshape(batch_size*8*(9*9), 2*prev_dim)
            prev_result = h_layer(intermed_obj_pairs)#apply MLP
            prev_dim = self.h_dim
            prev_result_2d = prev_result.reshape(batch_size*8, 9, 9, prev_dim)
            sum_j = prev_result_2d.sum(dim=2)

        hSum = sum_j.sum(dim=1)
        result = self.f_final(self.f(hSum))#pre-softmax scores for every possible answer

        answer = result.reshape(batch_size, 8)#scores of the 8 possible answers for every sample

        activation_loss = hSum.pow(2).mean() + result.pow(2).mean()

        return answer, activation_loss

model = WReN(2).to(devices[0]) #3-layer MLRN

if os.path.isfile("./weights.pt"):
    model.load_state_dict(torch.load("weights.pt"))
    print('Weights loaded')
else:
    print('No weights found')
    exit()


softmax = torch.nn.Softmax(dim=1)

parallel_model = torch.nn.DataParallel(model, device_ids=devices)

model.eval()

test_acc = []
objTypes = {}
attrTypes = {}
relTypes = {}
single_rel_correct = 0
single_rel_total = 0
# Testing
with torch.no_grad():
    for i, (local_batch, local_labels, meta) in enumerate(test_dataloader):
        local_batch, targets = local_batch.to(devices[0]), local_labels.to(devices[0])

        answer, _ = parallel_model(local_batch.type(torch.float32))

        #Calc accuracy
        answerSoftmax = softmax(answer)
        maxIndex = answerSoftmax.argmax(dim=1)

        correct = maxIndex.eq(targets)
        accuracy = correct.type(dtype=torch.float32).mean(dim=0)
        test_acc.append(accuracy)

        for j, jCorrect in enumerate(correct):
            jCorrect = jCorrect.item()
            if len(meta[j]) == 1:
                single_rel_total += 1
                if jCorrect == 1:
                    single_rel_correct += 1
                objType = meta[j][0][0]
                if objType in objTypes:
                    objTypes[objType]['total'] += 1
                    if jCorrect == 1:
                        objTypes[objType]['correct'] += 1
                else:
                    objTypes[objType] = {'total': 1, 'correct': jCorrect}

                attrType = meta[j][0][1]
                if attrType in attrTypes:
                    attrTypes[attrType]['total'] += 1
                    if jCorrect == 1:
                        attrTypes[attrType]['correct'] += 1
                else:
                    attrTypes[attrType] = {'total': 1, 'correct': jCorrect}

                relType = meta[j][0][2]
                if relType in relTypes:
                    relTypes[relType]['total'] += 1
                    if jCorrect == 1:
                        relTypes[relType]['correct'] += 1
                else:
                    relTypes[relType] = {'total': 1, 'correct': jCorrect}

        if i % 50 == 0:
            print("batch " + str(i))

    for key in objTypes:
        print(str(key) + ' ' + str(100 * objTypes[key]['correct'] / objTypes[key]['total']))
    for key in attrTypes:
        print(str(key) + ' ' + str(100 * attrTypes[key]['correct'] / attrTypes[key]['total']))
    for key in relTypes:
        print(str(key) + ' ' + str(100 * relTypes[key]['correct'] / relTypes[key]['total']))
    print(str(objTypes))
    print(str(attrTypes))
    print(str(relTypes))

    print('All single relations accuracy: ' + str(100 * single_rel_correct / single_rel_total))

    total_test_acc = sum(test_acc) / len(test_acc)
    print(sum(test_acc))
    print(len(test_acc))
    print('Test accuracy: ' + str(total_test_acc.item()))
