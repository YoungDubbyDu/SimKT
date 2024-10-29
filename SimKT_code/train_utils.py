import matplotlib.pyplot as plt
import os
import time
import torch
import numpy as np


class Logger:
    def __init__(self, args):
        self.n_epoch = 0
        self.start_timestamp = time.time()
        self.train_auc_list = []
        self.test_auc_list = []
        self.best_metric_dict = {'auc': 0.0, 'acc': 0.0}
        self.best_state_dict = None
        self.best_ques_emb = None
        self.best_epoch = -1
        self.save_dir = args.save_dir
        self.log_file = args.log_file
        self.result_file = args.result_file
        self.patience = args.patience
        self.duration = ''
        self.end_time = ''

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        self.pic_dir = self.save_dir + '/' + "pic"
        if not os.path.isdir(self.pic_dir):
            os.makedirs(self.pic_dir)
        self.log_writer = open(os.path.join(self.save_dir, self.log_file), 'a', encoding='utf-8')
        self.result_writer = open(os.path.join(self.save_dir, self.result_file), 'a', encoding='utf-8')
        self.param_path = '../param/params_%s_%d_%d.pkl' % (args.data_set, args.emb_dim, args.hidden_dim)

    def one_epoch(self, epoch, train_metric_dict, test_metric_dict, model):
        log = 'epoch=%d,train=%s,test=%s\n' % (epoch, train_metric_dict, test_metric_dict)
        self._logWriter(log)
        if self.best_metric_dict['auc'] < test_metric_dict['auc']:
            self.best_metric_dict = test_metric_dict.copy()
            self.best_epoch = epoch
            self.best_state_dict = model.QuesEmb_Layer.state_dict()
            # self.best_ques_emb = model.QuesEmb_Layer.whole_ques_embedding
            # torch.save(model.state_dict(), self.param_path)
        self._aucAppend(train_metric_dict['auc'], test_metric_dict['auc'])

    def one_run(self, args):
        self._getTime()
        # self._draw()

        result_dict = {'t': self.end_time, 'duration': self.duration, 'n_epoch': self.n_epoch, 'best_epoch': self.best_epoch}
        result_dict.update(self.best_metric_dict)
        result_dict.update(vars(args))
        self._resultWriter("%s\n" % str(result_dict))

        # save_path = '../param/%s_%d_%.3f.pkl' % (args.data_set, args.emb_dim, self.best_metric_dict['auc'])
        # torch.save(self.best_state_dict, save_path)
        # emb_path = '../final_emb/%s_%d_%.3f.emb' % (args.data_set, args.emb_dim, self.best_metric_dict['auc'])
        # np.save(emb_path, self.best_ques_emb.cpu().detach().numpy())
        # print("save model done")

    def _aucAppend(self, train_auc, test_auc):
        self.train_auc_list.append(train_auc)
        self.test_auc_list.append(test_auc)

    def is_stop(self):
        if len(self.test_auc_list) < self.patience:
            return False
        array = self.test_auc_list[-self.patience - 1:]
        if max(array[1:]) >= array[0]:
            return False
        else:
            return True

    def _logWriter(self, log):
        print(log.rstrip('\n'))
        self.log_writer.write(log)

    def _resultWriter(self, result):
        print(result.rstrip('\n'))
        self.result_writer.write(result)
        self.log_writer.write(result)

    def _draw(self):
        plt.figure()
        plt.plot(range(len(self.train_auc_list)), self.train_auc_list, label='train_auc', marker='o')
        plt.plot(range(len(self.test_auc_list)), self.test_auc_list, label='test_auc', marker='s')
        plt.title('%s' % self.end_time)
        plt.xlabel('epoch')
        plt.ylabel('auc')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.pic_dir, '%s.png' % self.end_time))
        plt.close()

    def _getTime(self):
        self.end_time = '%s' % time.strftime("%Y-%m-%d&%H-%M-%S", time.localtime(time.time()))
        s = time.time() - self.start_timestamp
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        self.duration = "%d:%d:%d" % (h, m, s)

    def epoch_increase(self):
        self.n_epoch += 1
