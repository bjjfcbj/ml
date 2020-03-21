import torch
import numpy as np
import pandas as pd

from scipy.io import arff


def nslconvert(nsldata):
    cvdata = list(nsldata)
    protocol_type = {b'tcp': 1, b'udp': 2, b'icmp': 3}
    service = {b'aol': 1, b'auth': 1, b'bgp': 3, b'courier': 4, b'csnet_ns': 5, b'ctf': 6, b'daytime': 7, b'discard': 8, b'domain': 9, b'domain_u': 10, b'echo': 11, b'eco_i': 12, b'ecr_i': 13, b'efs': 14, b'exec': 15, b'finger': 16, b'ftp': 17, b'ftp_data': 18, b'gopher': 19, b'harvest': 20, b'hostnames': 21, b'http': 22, b'http_2784': 23, b'http_443': 24, b'http_8001': 25, b'imap4': 26, b'IRC': 27, b'iso_tsap': 28, b'klogin': 29, b'kshell': 30, b'ldap': 31, b'link': 32, b'login': 33, b'mtp': 34, b'name': 35,
               b'netbios_dgm': 36, b'netbios_ns': 37, b'netbios_ssn': 38, b'netstat': 39, b'nnsp': 40, b'nntp': 41, b'ntp_u': 42, b'other': 43, b'pm_dump': 44, b'pop_2': 45, b'pop_3': 46, b'printer': 47, b'private': 48, b'red_i': 49, b'remote_job': 50, b'rje': 51, b'shell': 52, b'smtp': 53, b'sql_net': 54, b'ssh': 55, b'sunrpc': 56, b'supdup': 57, b'systat': 58, b'telnet': 59, b'tftp_u': 60, b'tim_i': 61, b'time': 62, b'urh_i': 63, b'urp_i': 64, b'uucp': 65, b'uucp_path': 66, b'vmnet': 67, b'whois': 68, b'X11': 69, b'Z39_50': 70}
    flag = {b'OTH': 1, b'REJ': 2, b'RSTO': 3, b'RSTOS0': 4, b'RSTR': 5,
            b'S0': 6, b'S1': 7, b'S2': 8, b'S3': 9, b'SF': 10, b'SH': 11}

    cvdata[1] = protocol_type[cvdata[1]]
    cvdata[2] = service[cvdata[2]]
    cvdata[3] = flag[cvdata[3]]
    cvdata[-1] = 1 if cvdata[-1] == b'anomaly' else 0

    for i in range(len(cvdata)):
        if isinstance(cvdata[i], bytes):
            cvdata[i] = int(cvdata[i])

    return np.array(cvdata)


class NslKddset(torch.utils.data.Dataset):
    def __init__(self, root, filename, train=True, transform=None):
        super(NslKddset, self).__init__()
        self.file = root+"/nsl-kdd/"+filename+".arff.txt"
        self.nslarff = arff.loadarff(self.file)
        self.nsldata = pd.DataFrame(self.nslarff[0]).values
        self.transform = transform
        self.train = train

    def __getitem__(self, index):
        # convert origin data
        cvdata = nslconvert(self.nsldata[index])
        realdata, label = cvdata[:-1], cvdata[-1]
        def lamb(label): return [0, 1] if label else [1, 0]

        if self.transform is not None:
            realdata = self.transform(realdata)

        if self.train:
            return realdata, torch.FloatTensor(lamb(label))
        return realdata, label

    def __len__(self):
        return len(self.nsldata)
