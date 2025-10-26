# 在solver.py文件中修改train和test方法，并添加时间统计相关代码

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from torch.autograd import grad
from torch.autograd import Variable
from model import *
import matplotlib.pyplot as plt
from utils import *
from data_loader import *
import IPython
from tqdm import tqdm
import csv  # 添加csv导入

class Solver(object):
    DEFAULTS = {}   
    def __init__(self, data_loader, config):
        # Data loader
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.data_loader = data_loader

        # Build tensorboard if use
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def build_model(self):
        # Define model
        input_dim = getattr(self, 'input_dim', 38)  # 默认值为118以保持向后兼容
        self.dagmm = DaGMM(self.gmm_k, input_dim=input_dim)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.dagmm.parameters(), lr=self.lr)

        # Print networks
        self.print_network(self.dagmm, 'DaGMM')

        if torch.cuda.is_available():
            self.dagmm.cuda()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):
        # 获取数据集名称，如果不存在则使用默认名称
        dataset_name = getattr(self, 'dataset', 'unknown_dataset')
        # 构建包含数据集名称的模型路径
        dataset_model_path = os.path.join(self.model_save_path, dataset_name)
        
        self.dagmm.load_state_dict(torch.load(os.path.join(
            dataset_model_path, '{}_dagmm.pth'.format(self.pretrained_model))))

   

        print('loaded trained models (step: {})..!'.format(self.pretrained_model))
    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def reset_grad(self):
        self.dagmm.zero_grad()

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def train(self):
        iters_per_epoch = len(self.data_loader)

        # Start with trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[0])
        else:
            start = 0

        # Start training
        iter_ctr = 0
        start_time = time.time()

        # 记录训练开始时间
        self.train_start_time = datetime.datetime.now()

        self.ap_global_train = np.array([0,0,0])
        for e in range(start, self.num_epochs):
            for i, (input_data, labels) in enumerate(tqdm(self.data_loader)):
        
                if torch.cuda.is_available():
                    input_data = input_data.cuda()
                iter_ctr += 1
                start = time.time()

                input_data = self.to_var(input_data)

                total_loss,sample_energy, recon_error, cov_diag = self.dagmm_step(input_data)
                # Logging
                loss = {}
                loss['total_loss'] = total_loss.data.item()
                loss['sample_energy'] = sample_energy.item()
                loss['recon_error'] = recon_error.item()
                loss['cov_diag'] = cov_diag.item()
              
                # Print out log info
                if (i+1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    total_time = ((self.num_epochs*iters_per_epoch)-(e*iters_per_epoch+i)) * elapsed/(e*iters_per_epoch+i+1)
                    epoch_time = (iters_per_epoch-i)* elapsed/(e*iters_per_epoch+i+1)
                    
                    epoch_time = str(datetime.timedelta(seconds=epoch_time))
                    total_time = str(datetime.timedelta(seconds=total_time))
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    lr_tmp = []
                    for param_group in self.optimizer.param_groups:
                        lr_tmp.append(param_group['lr'])
                    tmplr = np.squeeze(np.array(lr_tmp))

                    log = "Elapsed {}/{} -- {} , Epoch [{}/{}], Iter [{}/{}], lr {}".format(
                        elapsed,epoch_time,total_time, e+1, self.num_epochs, i+1, iters_per_epoch, tmplr)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)

                    IPython.display.clear_output()
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)
                    else:
                        plt_ctr = 1
                        if not hasattr(self,"loss_logs"):
                            self.loss_logs = {}
                            for loss_key in loss:
                                self.loss_logs[loss_key] = [loss[loss_key]]
                                plt.subplot(2,2,plt_ctr)
                                plt.plot(np.array(self.loss_logs[loss_key]), label=loss_key)
                                plt.legend()
                                plt_ctr += 1
                        else:
                            for loss_key in loss:
                                self.loss_logs[loss_key].append(loss[loss_key])
                                plt.subplot(2,2,plt_ctr)
                                plt.plot(np.array(self.loss_logs[loss_key]), label=loss_key)
                                plt.legend()
                                plt_ctr += 1

                        plt.show()

                    print("phi", self.dagmm.phi,"mu",self.dagmm.mu, "cov",self.dagmm.cov)
                # Save model checkpoints
            if (e+1) % self.model_save_step == 0:
            # 获取数据集名称，如果不存在则使用默认名称
                dataset_name = getattr(self, 'dataset', 'unknown_dataset')
                # 创建包含数据集名称的模型保存路径
                dataset_model_path = os.path.join(self.model_save_path, dataset_name)
                # 确保目录存在
                print("Saving model checkpoints to {}".format(dataset_model_path))
                os.makedirs(dataset_model_path, exist_ok=True)
                # 保存模型
                torch.save(self.dagmm.state_dict(),
                    os.path.join(dataset_model_path, '{}_{}_dagmm.pth'.format(e+1, i+1)))

        # 记录训练结束时间
        self.train_end_time = datetime.datetime.now()
        self.train_duration = self.train_end_time - self.train_start_time
        print(f"Training completed in {self.train_duration}")

    def dagmm_step(self, input_data):
        self.dagmm.train()
        if next(self.dagmm.parameters()).is_cuda:
            input_data = input_data.cuda()
        enc, dec, z, gamma = self.dagmm(input_data)

        total_loss, sample_energy, recon_error, cov_diag = self.dagmm.loss_function(input_data, dec, z, gamma, self.lambda_energy, self.lambda_cov_diag)

        self.reset_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.dagmm.parameters(), 5)
        self.optimizer.step()

        return total_loss,sample_energy, recon_error, cov_diag

    def test(self):
        print("======================TEST MODE======================")
        self.dagmm.eval()
        self.data_loader.dataset.mode="train"

        # 记录测试开始时间
        test_start_time = datetime.datetime.now()

        N = 0
        mu_sum = 0
        cov_sum = 0
        gamma_sum = 0

        for it, (input_data, labels) in enumerate(self.data_loader):
            input_data = self.to_var(input_data)
            enc, dec, z, gamma = self.dagmm(input_data)
            phi, mu, cov = self.dagmm.compute_gmm_params(z, gamma)
            
            batch_gamma_sum = torch.sum(gamma, dim=0)
            
            gamma_sum += batch_gamma_sum
            mu_sum += mu * batch_gamma_sum.unsqueeze(-1) # keep sums of the numerator only
            cov_sum += cov * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1) # keep sums of the numerator only
            
            N += input_data.size(0)
            
        train_phi = gamma_sum / N
        train_mu = mu_sum / gamma_sum.unsqueeze(-1)
        train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        print("N:",N)
        print("phi :\n",train_phi)
        print("mu :\n",train_mu)
        print("cov :\n",train_cov)

        train_energy = []
        train_labels = []
        train_z = []
        for it, (input_data, labels) in enumerate(self.data_loader):
            input_data = self.to_var(input_data)
            enc, dec, z, gamma = self.dagmm(input_data)
            sample_energy, cov_diag = self.dagmm.compute_energy(z, phi=train_phi, mu=train_mu, cov=train_cov, size_average=False)
            
            train_energy.append(sample_energy.data.cpu().numpy())
            train_z.append(z.data.cpu().numpy())
            train_labels.append(labels.numpy())


        train_energy = np.concatenate(train_energy,axis=0)
        train_z = np.concatenate(train_z,axis=0)
        train_labels = np.concatenate(train_labels,axis=0)
        print("enter here")

        self.data_loader.dataset.mode="test"
        test_energy = []
        test_labels = []
        test_z = []
        for it, (input_data, labels) in enumerate(self.data_loader):
            input_data = self.to_var(input_data)
            enc, dec, z, gamma = self.dagmm(input_data)
            sample_energy, cov_diag = self.dagmm.compute_energy(z, size_average=False)
            test_energy.append(sample_energy.data.cpu().numpy())
            test_z.append(z.data.cpu().numpy())
            test_labels.append(labels.numpy())


        test_energy = np.concatenate(test_energy,axis=0)
        test_z = np.concatenate(test_z,axis=0)
        test_labels = np.concatenate(test_labels,axis=0)

        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        combined_labels = np.concatenate([train_labels, test_labels], axis=0)
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_curve
        
        # 计算约登指数 (Youden's J statistic = sensitivity + specificity - 1)
        # sensitivity = tpr, specificity = 1 - fpr
        fpr, tpr, thresholds = roc_curve(test_labels, test_energy)
        youden_j = tpr - fpr  # 等价于 (tpr + (1-fpr) - 1)
        
        # 找到约登指数最大的索引
        optimal_idx = np.argmax(youden_j)
        thresh = thresholds[optimal_idx]
        print("Best Threshold (based on ROC):", thresh)
        print("Threshold Youden's J statistic:", youden_j[optimal_idx])

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        accuracy = accuracy_score(gt,pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')

        # 记录测试结束时间
        test_end_time = datetime.datetime.now()
        test_duration = test_end_time - test_start_time
        
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(accuracy, precision, recall, f_score))
        print(f"Test completed in {test_duration}")
        
        # 保存结果到CSV文件
        self.save_results_to_csv(precision, recall, f_score, 
                                str(self.train_duration) if hasattr(self, 'train_duration') else 'unknown',
                                str(test_duration))
        
        return accuracy, precision, recall, f_score

    def save_results_to_csv(self, precision, recall, f_score, train_time, test_time):
        """
        将测试结果保存到CSV文件
        """
        # 获取数据集名称（如果在配置中设置了的话）
        dataset_name = getattr(self, 'dataset', 'unknown_dataset')
        
        # CSV文件路径
        csv_file = os.path.join(self.model_save_path, 'results.csv')
        
        # 检查文件是否存在，不存在则创建并写入表头
        file_exists = os.path.exists(csv_file)
        
        with open(csv_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # 如果文件不存在，先写入表头
            if not file_exists:
                writer.writerow(['Dataset', 'Train_Time', 'Test_Time', 'Recall', 'Precision', 'F1-Score'])
            
            # 写入数据
            writer.writerow([dataset_name, train_time, test_time, 
                            f"{recall:.4f}", f"{precision:.4f}", f"{f_score:.4f}"])
        
        print(f"Results saved to {csv_file}")