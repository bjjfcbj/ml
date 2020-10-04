import torch
import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets as skd

space = torch.tensor([0. for i in range(8)])
root = '../../data/'
protocol_type = {b'tcp': 1, b'udp': 2, b'icmp': 3}
service = {b'aol': 71, b'auth': 1, b'bgp': 3, b'courier': 4, b'csnet_ns': 5, b'ctf': 6, b'daytime': 7, b'discard': 8, b'domain': 9, b'domain_u': 10, b'echo': 11, b'eco_i': 12, b'ecr_i': 13, b'efs': 14, b'exec': 15, b'finger': 16, b'ftp': 17, b'ftp_data': 18, b'gopher': 19, b'harvest': 20, b'hostnames': 21, b'http': 22, b'http_2784': 23, b'http_443': 24, b'http_8001': 25, b'imap4': 26, b'IRC': 27, b'iso_tsap': 28, b'klogin': 29, b'kshell': 30, b'ldap': 31, b'link': 32, b'login': 33, b'mtp': 34, b'name': 35,
           b'netbios_dgm': 36, b'netbios_ns': 37, b'netbios_ssn': 38, b'netstat': 39, b'nnsp': 40, b'nntp': 41, b'ntp_u': 42, b'other': 43, b'pm_dump': 44, b'pop_2': 45, b'pop_3': 46, b'printer': 47, b'private': 48, b'red_i': 49, b'remote_job': 50, b'rje': 51, b'shell': 52, b'smtp': 53, b'sql_net': 54, b'ssh': 55, b'sunrpc': 56, b'supdup': 57, b'systat': 58, b'telnet': 59, b'tftp_u': 60, b'tim_i': 61, b'time': 62, b'urh_i': 63, b'urp_i': 64, b'uucp': 65, b'uucp_path': 66, b'vmnet': 67, b'whois': 68, b'X11': 69, b'Z39_50': 70}
flag = {b'OTH': 1, b'REJ': 2, b'RSTO': 3, b'RSTOS0': 4, b'RSTR': 5,
        b'S0': 6, b'S1': 7, b'S2': 8, b'S3': 9, b'SF': 10, b'SH': 11}
# 2:dos,3:probe,4:r2l,5:u2r
# cvdict = {b'normal': 0,
#           b'back': 1, b'land': 1, b'neptune': 1, b'pod': 1, b'smurf': 1, b'teardrop': 1,
#           b'ipsweep': 2, b'nmap': 2, b'portsweep': 2, b'satan': 2,
#           b'ftp_write': 3, b'guess_passwd': 3, b'imap': 3, b'multihop': 3, b'phf': 3, b'spy': 3, b'warezclient': 3, b'warezmaster': 3,
#           b'buffer_overflow': 4, b'loadmodule': 4, b'perl': 4, b'rootkit': 4}


def nslconvert(data, label=0):
    cvdata = list(data)
    cvdata[1] = protocol_type[cvdata[1]]
    cvdata[2] = service[cvdata[2]]
    cvdata[3] = flag[cvdata[3]]
    label = 0 if cvdata[-1] == b'normal' else 1

    return torch.tensor(cvdata[:-1]), label


def nsl_csv_read(filename):
    file = np.genfromtxt(filename,
                         dtype={'names': ('duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class', 'level'),
                                'formats': ('f', 'S4', 'S16', 'S10', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'S20', 'f')},
                         delimiter=',',
                         usecols=range(42))

    return file


class Nslset(torch.utils.data.Dataset):
    def __init__(self, filename, root=root, train=True, transform=None):
        super(Nslset, self).__init__()
        self.file = root+'nsl-kdd/'+filename
        self.nsldata = nsl_csv_read(self.file)
        self.transform = transform
        self.train = train

    def __getitem__(self, index):
        # convert origin data
        realdata, label = nslconvert(self.nsldata[index])
        realdata = torch.cat((realdata, space), 0)
        realdata = realdata.reshape(7, -1).unsqueeze(0)

        if self.transform is not None:
            realdata = self.transform(realdata)

        # if self.train:
        #     return realdata, label  # labeldeal(label)
        return realdata, label

    def __len__(self):
        return len(self.nsldata)
