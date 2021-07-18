'''
Author: Osama Adel
Date: 6 July 2020
Description: a collection of functions to load different datasets

Last Update: 20 July 2020
'''

import numpy as np
import pandas as pd
from pickle import dump


def load_ejust(path:str, folder:str, subjects_range:list, start=5000, end=9700):
    """
    Loads the dataset and returns a data list and labels list. The data list contains a numpy array
    for each label corresponding to a loaded file.
    """
    import os
    subjects_files = os.listdir(os.path.join(path, folder))
    datalist = []
    labelslist = []
    for l, subject_file in enumerate(subjects_files[subjects_range[0]:subjects_range[1]]):
        data = np.load(os.path.join(path, folder, subject_file))[start:end,:]
        datalist.append(data.copy())
        labelslist.append(l)
    return datalist, labelslist


def load_osaka(path:str, subjects_range:list, seq=0):
    import os
    data = []
    labels = []
    filenames = os.listdir(path)
    for i in range(subjects_range[0], subjects_range[1]):
        with open(os.path.join(path, filenames[i*2 if seq==0 else i*2+1])) as f:
            nrows = f.readline()
            ncols = f.readline()
            # take only even-index samples from the recording session if downsample is True
            data_file = np.array([list(map(float, x.rstrip('\n').split(','))) for x in f if x != '\n'])
            data.append(data_file)
            labels.append(i)
    return data, labels


def load_mmuisd(path:str, subjects_range:list):
    import os
    import pandas as pd
    # list all csv files inside the path
    filenames = [x.split('_') for x in os.listdir(path)]
    # excluse all R1 files (sensors in the right pocket)
    filenames = list(filter(lambda x: x if x[-1].startswith('R1') else None, filenames))
    # Load file by file (subject by subject) and segment each one
    data = []
    labels = []
    for i in range(subjects_range[0],subjects_range[1]):
        name = filenames[i]
        # print('\nlen(filenames):', len(filenames))
        label = int(name[0])
        # files 81:120 have different delimiter ';'
        if label > 80:
            data_file = pd.read_csv(os.path.join(path, '_'.join(name)), delimiter=';').values[:,1:]
        else:
            data_file = pd.read_csv(os.path.join(path, '_'.join(name)), delimiter=',').values[:,1:]
        data_file[:,:3] = data_file[:,:3] / 9.8
        data_file[:,3:6] = data_file[:,3:6] * 180 / np.pi
        data.append(data_file)
        labels.append(label)
    return data, labels


def load_segment_EJUST(PIDSpath, foldername, subjects_range, acc_only=False, sample_rate=50, segment_time=1, overlapped=False, overlap=0.1, val=False):
    import os
    subjects_files = os.listdir(os.path.join(PIDSpath, foldername))
    datalist = []
    labelslist = []
    for l, subject_file in enumerate(subjects_files[subjects_range[0]:subjects_range[1]]):
        if not val:
            data = np.load(os.path.join(PIDSpath, foldername, subject_file))[15000:20000,:]
            # mu = data.mean(axis=0)
            # std = data.std(axis=0)
            # data = (data - mu) / std
        else:
            data = np.load(os.path.join(PIDSpath, foldername, subject_file))[25000:30000,:]
            # mu = data.mean(axis=0)
            # std = data.std(axis=0)
            # data = (data - mu) / std
        if acc_only:
            data = data[:,:3]
            
        if overlapped:
            segmentData_overlapped(data, l+subjects_range[0], datalist, labelslist, sample_rate, segment_time, overlap)
        else:
            segmentData(data, l+subjects_range[0], datalist, labelslist, sample_rate, segment_time)
    return datalist, labelslist


def load_segment_osaka(root_folder, subjects_range, acc_only=False, sample_rate=100, segment_time=2, seq=0, overlapped=False, overlap=0.2, downsample=True):
    '''
    Returns:
        data: list of lists of segments each list has N segments that comprise one subject
                ====================================== x =================================
                ========= SUBJ1 Data =========, ........... ========= SUBJN Data =========
                === Seg1 === .... === SegM ===, ........... === Seg1 === .... === SegM ===
                Arr1 .. ArrN .... Arr1 .. ArrN, ........... Arr1 .. ArrN .... Arr1 .. ArrN
                ==========================================================================

        labels: list of labels each correspond to one subject
    '''
    import os
    data = []
    labels = []
    filenames = os.listdir(root_folder)

    # divide sample_rate by half if downsample is True to change segment lengths accordingly
    sample_rate = sample_rate // 2 if downsample else sample_rate

    for i in range(subjects_range[0], subjects_range[1]):
        with open(os.path.join(root_folder, filenames[i*2 if seq==0 else i*2+1])) as f:
            nrows = f.readline()
            ncols = f.readline()
            # take only even-index samples from the recording session if downsample is True
            if downsample:
                data_file = np.array([list(map(float, x.rstrip('\n').split(','))) for x in f if x != '\n'])[::2,:]
                # mu = data_file.mean(axis=0)
                # std = data_file.std(axis=0)
                # data_file = (data_file - mu) / std
            else:
                data_file = np.array([list(map(float, x.rstrip('\n').split(','))) for x in f if x != '\n'])
                # mu = data_file.mean(axis=0)
                # std = data_file.std(axis=0)
                # data_file = (data_file - mu) / std
            if acc_only:
                data_file = data_file[:,:3]
            # make overlapping segments if overlapped is True
            if overlapped:
                segmentData_overlapped(data_file, i, data, labels, sample_rate, segment_time, overlap)
            else:
                segmentData(data_file, i, data, labels, sample_rate, segment_time)
            # data.append(data_file)
            # labels.append(i)
    return data, labels



def load_segment_mmuisd(mmuisd_path, subjects_range, acc_only=False, sample_rate=50, segment_time=2):
    import os
    import pandas as pd
    # list all csv files inside the mmuisd_path
    filenames = [x.split('_') for x in os.listdir(mmuisd_path)]
    # excluse all R1 files (sensors in the right pocket)
    filenames = list(filter(lambda x: x if x[-1].startswith('R1') else None, filenames))
    filenames.sort()
    # Load file by file (subject by subject) and segment each one
    data = []
    labels = []
    for i in range(subjects_range[0],subjects_range[1]):
        name = filenames[i]
        # print('\nlen(filenames):', len(filenames))
        label = int(name[0])
        # files 81:120 have different delimiter ';'
        if label > 80:
            data_file = pd.read_csv(os.path.join(mmuisd_path, '_'.join(name)), delimiter=';').values[:,1:]
        else:
            data_file = pd.read_csv(os.path.join(mmuisd_path, '_'.join(name)), delimiter=',').values[:,1:]
        data_file[:,:3] = data_file[:,:3] / 9.8
        data_file[:,3:6] = data_file[:,3:6] * 180 / np.pi
        if acc_only:
            data_file = data_file[:,:3]
        # mu = data_file.mean(axis=0)
        # std = data_file.std(axis=0)
        # data_file = (data_file - mu) / std
            

        # print('filename:', os.path.join(mmuisd_path, '_'.join(name)))
        # print('data_file.shape:', data_file.shape)
        segmentData(data_file, label, data, labels, sample_rate, segment_time)
    return data, labels


def segmentData(data, label, segmentList, labelList, sample_rate=100, segment_time=1):
    segment_len = sample_rate * segment_time
    seglist = []
    for seg in range(0, data.shape[0]-segment_len+1, segment_len):
        seglist.append(data[seg:seg+segment_len, :])
    segmentList.append(seglist)
    labelList.append(label)


def segmentData_overlapped(data, label, segmentList, labelList, sample_rate=100, segment_time=1, overlap=0.2):
    segment_len = sample_rate * segment_time
    step = int(segment_len*(1-overlap))
    seglist = []
    for seg in range(0, (data.shape[0]-segment_len)+1, step):
        seglist.append(data[seg:seg+segment_len, :])
    segmentList.append(seglist)
    labelList.append(label)


def appendData(data, newdata, label):
    if data is None:
        return np.hstack([newdata, np.full([newdata.shape[0], 1], label)])
    else:
        newdata = np.hstack([newdata, np.full([newdata.shape[0], 1], label)])
        return np.vstack([data, newdata])


def randomizeSegments(segmentList, labelList):
    if type(segmentList) == list:
        rand_indices = np.random.choice([x for x in range(len(segmentList))], size=len(segmentList), replace=False)
        segmentList = [segmentList[x] for x in rand_indices]
        labelList = [labelList[x] for x in rand_indices]
    elif type(segmentList) == np.ndarray:
        rand_indices = np.random.choice([x for x in range(segmentList.shape[0])], size=segmentList.shape[0], replace=False)
        segmentList = segmentList[rand_indices]
        labelList = labelList[rand_indices]
    return segmentList, labelList


def get_random_pair(X):
    ''' X is a list of two lists x1 and x2.
        This function returns a list of two elements,
        one from x1 and the other from x2. '''
    assert type(X) == list, 'inputs are not of type list'
    assert len(X) == 2, 'number of elements in input list is 2'
    assert len(X[0]) >= 2 and len(X[1]) >= 2, 'number of segments per subject is < 2'
    from random import randint
    m = min(len(X[0]), len(X[1]))
    i_rand = np.random.choice(range(m), size=2, replace=False) #[randint(0, m-1) for i in range(2)]
    rv = [X[i][j] for i, j in enumerate(i_rand)]
    return rv


def generate_batch(samples_per_subject, x, y):
    '''
    The following is the structure of x
    ====================================== x =================================
    ========= SUBJ1 Data =========, ........... ========= SUBJN Data =========
    === Seg1 === .... === SegM ===, ........... === Seg1 === .... === SegM ===
    Arr1 .. ArrN .... Arr1 .. ArrN, ........... Arr1 .. ArrN .... Arr1 .. ArrN
    ==========================================================================

    y is a list of N labels for N subjects

    Returns:
        pairs - ndarray of two entries, the first is a list of all anchor samples
                the second is a list of all pos/neg samples
        labels - 1 for each pair that is similar, 0 for each pair that is unsimilar
        
        
        NDARRAY[       ------ similar ------    .... ----- unsimilar ------
              LIST1  = [anchor, anchor, anchor, ..., anchor, anchor, anchor]
              LIST2  = [sample, sample, sample, ..., sample, sample, sample]
              ]
        labels =       [1,      1,      1,      ...,      0,      0,      0]
    '''
    N = len(y)
    import random
    similar = []                        # list of similar pairs of segments
    unsimilar = []                      # list of unsimilar pairs of segments
    # filling in the similar list
    for subj_i in range(N):
        for sample_j in range(samples_per_subject):
            similar_sample = get_random_pair([x[subj_i], x[subj_i]])
            similar.append(similar_sample)
            subj_r = random.randint(0, N-1)
            while subj_r == subj_i:
                subj_r = random.randint(0, N-1)
            unsimilar_sample = get_random_pair([x[subj_i], x[subj_r]])
            unsimilar.append(unsimilar_sample)
    return np.array(list(zip(*(similar + unsimilar)))), \
            np.array((len(similar)*[1]) + (len(similar)*[0]))