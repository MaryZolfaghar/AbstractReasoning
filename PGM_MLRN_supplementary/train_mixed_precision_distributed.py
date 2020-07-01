import numpy as np
import torch
import os
import sys
import re
import math
from torch.utils.data import Dataset, DataLoader
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from lamb import Lamb

#tensorboard for accuracy graphs
import tensorflow as tf

def getCombinations(inputTensor, N, c, d):#input shape=(batch_size, obj_count, obj_dim) #batch_size=N, obj_count=c, obj_dim=d
    tensorA = inputTensor.reshape(N, 1, c, d).expand(N, c, c, d)
    tensorB = tensorA.transpose(1, 2)

    return torch.cat((tensorB, tensorA), 3)

dataset_name = 'neutral'#'interpolation'#'extrapolation'


if len(sys.argv) < 2:
    print("Missing data path!")
    exit()

datapath_preprocessed = os.path.join(sys.argv[1], dataset_name + '_preprocessed')

class PgmDataset(Dataset):
    def __init__(self, filenames):
        'Initialization'
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        with np.load(os.path.join(datapath_preprocessed, filename)) as data:
            image = data['image'].astype(np.uint8).reshape(16, 80, 80)
            target = data['target']
        return image, target

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
        std_dev = 0.28
        self.input_encoding_variance_inv = 1.0 / (math.sqrt(2.0) * std_dev)
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
        self.post_cnn_linear = torch.nn.Linear(32*4*4, 256-9)

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
        batch_flat = batch.reshape(batch_size*16, 1, 80, 80)

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

        objects = torch.cat((optionsWithoutPos, self.tag_matrix.repeat(batch_size, 1)), dim=1).reshape(batch_size*8, 9, 256-9+9)

        #MLRN
        objPairs2D = getCombinations(objects, batch_size*8, 9, 256)
        objPairs = objPairs2D.reshape(batch_size*8*(9*9), 2*256)

        gResult = self.g(objPairs)#apply MLP

        prev_result = gResult
        prev_dim = self.h_dim
        prev_result_2d = prev_result.reshape(batch_size*8, 9, 9, prev_dim)
        sum_j = prev_result_2d.sum(dim=2)
        for i, h_layer in enumerate(self.h):
            residual = sum_j
            intermed_obj_pairs_2d = getCombinations(sum_j, batch_size*8, 9, prev_dim)
            intermed_obj_pairs = intermed_obj_pairs_2d.reshape(batch_size*8*(9*9), 2*prev_dim)
            prev_result = h_layer(intermed_obj_pairs)#apply MLP
            prev_dim = self.h_dim
            prev_result_2d = prev_result.reshape(batch_size*8, 9, 9, prev_dim)
            sum_j = prev_result_2d.sum(dim=2)

        hSum = sum_j.sum(dim=1)
        result = self.f_final(self.f(hSum))#pre-softmax scores for every possible answer

        answer = result.reshape(batch_size, 8)

        #attempt to stabilize training (avoiding inf value activations in last layers) 
        activation_loss = hSum.pow(2).mean() + result.pow(2).mean()

        return answer, activation_loss

def worker_fn(rank, world_size):
    setup(rank, world_size)

    weights_filename = "weights.pt"
    batch_size = 512
    epochs = 240
    warmup_epochs = 8
    use_mixed_precision = True

    batch_size = batch_size // world_size #batch size per worker

    #Data
    all_data = os.listdir(datapath_preprocessed)
    train_filenames = [p for p in all_data if re.match(r'^PGM_' + re.escape(dataset_name) + r'_train_(\d+)\.npz$', p) is not None]
    val_filenames = [p for p in all_data if re.match(r'^PGM_' + re.escape(dataset_name) + r'_val_(\d+)\.npz$', p) is not None]
    train_dataset = PgmDataset(train_filenames)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, pin_memory=False, sampler=train_sampler)#shuffle is done by the sampler
    val_dataloader = DataLoader(PgmDataset(val_filenames), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)

    #Model
    device_ids = [rank]

    model = WReN(2).to(device_ids[0])#3-layer MLRN

    if weights_filename is not None and os.path.isfile("./" + weights_filename):
        model.load_state_dict(torch.load(weights_filename, map_location='cpu'))
        print('Weights loaded')
        cold_start = False
    else:
        print('No weights found')
        cold_start = True

    #Loss and optimizer
    final_lr = 2e-3

    def add_module_params_with_decay(module, weight_decay, param_groups):#adds parameters with decay unless they are bias parameters, which shouldn't receive decay
        group_with_decay = []
        group_without_decay = []
        for name, param in module.named_parameters():
            if not param.requires_grad: continue
            if name == 'bias' or name.endswith('bias'):
                group_without_decay.append(param)
            else:
                group_with_decay.append(param)
        param_groups.append({"params": group_with_decay, "weight_decay": weight_decay})
        param_groups.append({"params": group_without_decay})

    optimizer_param_groups = [
    ]

    add_module_params_with_decay(model.conv, 2e-1, optimizer_param_groups)
    add_module_params_with_decay(model.post_cnn_linear, 2e-1, optimizer_param_groups)
    add_module_params_with_decay(model.g, 2e-1, optimizer_param_groups)
    add_module_params_with_decay(model.h, 2e-1, optimizer_param_groups)
    add_module_params_with_decay(model.f, 2e-1, optimizer_param_groups)
    add_module_params_with_decay(model.f_final, 2e-1, optimizer_param_groups)

    optimizer = Lamb(optimizer_param_groups, lr=final_lr)

    base_model = model
    if use_mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1") #Mixed Precision

    lossFunc = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)

    #Parallel distributed model
    device = device_ids[0]
    torch.cuda.set_device(device)
    parallel_model = torch.nn.parallel.DistributedDataParallel(model, device_ids)

    if rank == 0:
        #accuracy logging
        sess = tf.Session()
        train_acc_placeholder = tf.placeholder(tf.float32, shape=())
        train_acc_summary = tf.summary.scalar('training_acc', train_acc_placeholder)
        val_acc_placeholder = tf.placeholder(tf.float32, shape=())
        val_acc_summary = tf.summary.scalar('validation_acc', val_acc_placeholder)
        writer = tf.summary.FileWriter("log", sess.graph)

    #training loop
    acc = []
    global_step = 0
    for epoch in range(epochs): 
        train_sampler.set_epoch(epoch) 

        # Validation
        val_acc = []
        parallel_model.eval()
        with torch.no_grad():
            for i, (local_batch, local_labels) in enumerate(val_dataloader):
                local_batch, targets = local_batch.to(device), local_labels.to(device)

                #answer = model(local_batch.type(torch.float32))
                answer, _ = parallel_model(local_batch.type(torch.float32))

                #Calc accuracy
                answerSoftmax = softmax(answer)
                maxIndex = answerSoftmax.argmax(dim=1)

                correct = maxIndex.eq(targets)
                accuracy = correct.type(dtype=torch.float16).mean(dim=0)
                val_acc.append(accuracy)

                if i % 50 == 0 and rank == 0:
                    print("batch " + str(i))

        total_val_acc = sum(val_acc) / len(val_acc)
        print('Validation accuracy: ' + str(total_val_acc.item()))
        if rank == 0:
            summary = sess.run(val_acc_summary, feed_dict={val_acc_placeholder: total_val_acc.item()})
            writer.add_summary(summary, global_step=global_step)

        # Training
        parallel_model.train()
        for i, (local_batch, local_labels) in enumerate(train_dataloader):
            global_step = global_step + 1

            if cold_start and epoch < warmup_epochs:#linear scaling of the lr for warmup during the first few epochs
                lr = final_lr * global_step / (warmup_epochs*len(train_dataset) / (batch_size * world_size))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            local_batch, targets = local_batch.to(device_ids[0]), local_labels.to(device_ids[0])

            optimizer.zero_grad()
            answer, activation_loss = parallel_model(local_batch.type(torch.float32))

            loss = lossFunc(answer, targets) + activation_loss * 2e-3

            #Calc accuracy
            answerSoftmax = softmax(answer)
            maxIndex = answerSoftmax.argmax(dim=1)

            correct = maxIndex.eq(targets)
            accuracy = correct.type(dtype=torch.float16).mean(dim=0)
            acc.append(accuracy)
            
            #Training step
            if use_mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss: #Mixed precision
                    scaled_loss.backward()
            else:
                loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(parallel_model.parameters(), 1e1)

            optimizer.step()

            if i % 50 == 0 and rank == 0:
                print("epoch " + str(epoch) + " batch " + str(i))
                print("loss", loss)
                print("activation loss", activation_loss)
                print(grad_norm)

            #logging and saving weights
            if i % 1000 == 999:
                trainAcc = sum(acc) / len(acc)
                acc = []
                print('Training accuracy: ' + str(trainAcc.item()))
                if rank == 0:
                    if weights_filename is not None:
                        torch.save(base_model.state_dict(), weights_filename)
                        print('Weights saved')

                    summary = sess.run(train_acc_summary, feed_dict={train_acc_placeholder: trainAcc.item()})
                    writer.add_summary(summary, global_step=global_step)  

        if cold_start and weights_filename is not None and epoch % 10 == 0 and rank == 0:
            torch.save(base_model.state_dict(), weights_filename + "_cp" + str(epoch))
            print('Checkpoint saved')


    cleanup()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)

def cleanup():
    torch.distributed.destroy_process_group()

def run(world_size):
    torch.multiprocessing.spawn(worker_fn, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    run(4)#4 GPUs
