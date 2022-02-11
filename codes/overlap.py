import pickle

import textdistance

import pandas as pd

import tqdm

DATA_NAME = 'foursquare_nyc'


class SimilarityMeasures:

    def __init__(self,a,b):
        '''
        a: a list of training trajectories
        b: the current trajectory
        '''
        self.a = a
        self.b = b

        self.results = dict()

    def compute_all(self):
        self.results = {
            'from_end': self._from_end(),
            'lcss': self._max_subsequence(),
            'jaccard': self._jaccard()
        }

        return self.results

    '''
    FROM END OVERLAP
    '''
    def _from_end(self):

        train_trajecotries = self.a
        current_trajectory = self.b

        current_trajectory = current_trajectory[::-1]

        max_overlap = 0

        for traj in train_trajecotries:
            local_overlap = 0
            traj = traj[::-1]
            min_len = min(len(traj), len(current_trajectory))

            for i in range(min_len):
                if traj[i] == current_trajectory[i]:
                    local_overlap += 1
                else:
                    break
            if local_overlap / len(current_trajectory) >= max_overlap:
                max_overlap = local_overlap / len(current_trajectory)


        return max_overlap

    '''
    LCSS OVERLAP
    '''
    def _max_subsequence(self):

        train_trajecotries = self.a
        current_trajectory = self.b

        max_overlap = 0

        for traj in train_trajecotries:

            no_common_elements = self._lcs(traj, current_trajectory)

            if no_common_elements / len(current_trajectory) >= max_overlap:
                max_overlap = no_common_elements / len(current_trajectory)

        return max_overlap



    def _lcs(self,S,T):
        m = len(S)
        n = len(T)
        counter = [[0]*(n+1) for x in range(m+1)]
        longest = 0
        lcs_set = set()
        current_max = 0
        for i in range(m):
            for j in range(n):
                if S[i] == T[j]:
                    c = counter[i][j] + 1
                    counter[i+1][j+1] = c
                    if c > longest:
                        lcs_set = set()
                        longest = c
                        lcs_set.add(S[i-c+1:i+1][0])
                    elif c == longest:
                        lcs_set.add(S[i-c+1:i+1][0])

        return(longest)

    '''
    JACCARD OVERLAP
    '''
    def _jaccard(self):

        train_trajecotries = self.a
        current_trajectory = self.b

        max_overlap = 0

        for traj in train_trajecotries:

            no_common_elements = self._jaccard_similarity(traj, current_trajectory)

            if no_common_elements >= max_overlap:
                max_overlap = no_common_elements

        return max_overlap

    def _jaccard_similarity(self, traj1, traj2):
        return textdistance.jaccard.normalized_similarity(traj1, traj2)


pickle_file = r'../data/'+DATA_NAME+'.pk'

pickle_obj =  pd.read_pickle(pickle_file)

lens = []

train_l = 0
test_l = 0
val_l = 0
uid_count = 0

for uid in pickle_obj['data_neural'].keys():
    uid_count += 1
    train_l += len(pickle_obj['data_neural'][uid]['train'])
    test_l += len(pickle_obj['data_neural'][uid]['test'])
    val_l +=len(pickle_obj['data_neural'][uid]['validation'])


for uid in pickle_obj['data_neural'].keys():
    for i in pickle_obj['data_neural'][uid]['sessions']:
        lens.append(len(pickle_obj['data_neural'][uid]['sessions'][i]))

print(train_l, test_l, val_l)


train_traces = []
for user in pickle_obj['data_neural'].keys():
    train_idx = pickle_obj['data_neural'][user]['train']
    for idx in train_idx:
        train_traces.extend([[point[0] for point in pickle_obj['data_neural'][user]['sessions'][idx]]])

results = dict()

for user in tqdm.tqdm(pickle_obj['data_neural'].keys()):

    results[user] = dict()

    test_idx = pickle_obj['data_neural'][user]['test']

    test_traces = []

    for idx in test_idx:
        temp_trace = [point[0] for point in pickle_obj['data_neural'][user]['sessions'][idx]]

        sim_measures = SimilarityMeasures(train_traces, temp_trace)

        results[user][idx] = sim_measures.compute_all()


stats = {
    'from_end': {
        '0':0,
        '0-20': 0,
        '20-40': 0,
        '40-60': 0,
        '60-80': 0,
        '80-100': 0
    },
    'lcss': {
        '0':0,
        '0-20': 0,
        '20-40': 0,
        '40-60': 0,
        '60-80': 0,
        '80-100': 0
    },
    'jaccard': {
        '0':0,
        '0-20': 0,
        '20-40': 0,
        '40-60': 0,
        '60-80': 0,
        '80-100': 0
    }
}

for uid in results:
    for traj in results[uid]:
        for i in results[uid][traj]:
            current_val = results[uid][traj][i]
            if current_val == 0:
                stats[i]['0'] = stats[i]['0'] +1
            if current_val <= 0.2:
                stats[i]['0-20'] = stats[i]['0-20'] + 1
            elif current_val <= 0.4 and current_val > 0.2:
                stats[i]['20-40'] = stats[i]['20-40'] + 1
            elif current_val <= 0.6 and current_val > 0.6-0.2:
                stats[i]['40-60'] = stats[i]['40-60'] + 1
            elif current_val <= 0.8 and current_val > 0.6:
                stats[i]['60-80'] = stats[i]['60-80'] + 1
            elif current_val <= 1 and current_val > 0.8:
                stats[i]['80-100'] = stats[i]['80-100'] + 1

pickle.dump(stats, open('../data/stats_'+DATA_NAME+'.pk', 'wb'))


def write_filtered_pk(upper_limit, metric, fname):

    local =  pd.read_pickle(pickle_file)

    lower_limit = 0
    if upper_limit <= 0.2:
        lower_limit = -0.01
    else:
        lower_limit = round(upper_limit - 0.2, 1)

    for uid in local['data_neural'].keys():
        filtered_ids = []
        for traj in local['data_neural'][uid]['test']:
            if results[uid][traj][metric] <= upper_limit and results[uid][traj][metric] > lower_limit:
                filtered_ids.append(traj)
        local['data_neural'][uid]['test'] = filtered_ids

    pickle.dump(local, open('../data/' + fname + '_' + str(upper_limit) + '_' + metric +'.pk', 'wb'))
    return local
temp_name = DATA_NAME

write_filtered_pk(0.2, 'from_end', temp_name)
write_filtered_pk(0.4, 'from_end', temp_name)
write_filtered_pk(0.6, 'from_end', temp_name)
write_filtered_pk(0.8, 'from_end', temp_name)
write_filtered_pk(1, 'from_end', temp_name)

write_filtered_pk(0.2, 'lcss', temp_name)
write_filtered_pk(0.4, 'lcss', temp_name)
write_filtered_pk(0.6, 'lcss', temp_name)
write_filtered_pk(0.8, 'lcss', temp_name)
write_filtered_pk(1, 'lcss', temp_name)

write_filtered_pk(0.2, 'jaccard', temp_name)
write_filtered_pk(0.4, 'jaccard', temp_name)
write_filtered_pk(0.6, 'jaccard', temp_name)
write_filtered_pk(0.8, 'jaccard', temp_name)
write_filtered_pk(1, 'jaccard', temp_name)
write_filtered_pk(0.4, 'from_end', temp_name)
write_filtered_pk(0.6, 'from_end', temp_name)
write_filtered_pk(0.8, 'from_end', temp_name)
write_filtered_pk(1, 'from_end', temp_name)

write_filtered_pk(0.2, 'lcss', temp_name)
write_filtered_pk(0.4, 'lcss', temp_name)
write_filtered_pk(0.6, 'lcss', temp_name)
write_filtered_pk(0.8, 'lcss', temp_name)
write_filtered_pk(1, 'lcss', temp_name)

write_filtered_pk(0.2, 'jaccard', temp_name)
write_filtered_pk(0.4, 'jaccard', temp_name)
write_filtered_pk(0.6, 'jaccard', temp_name)
write_filtered_pk(0.8, 'jaccard', temp_name)
write_filtered_pk(1, 'jaccard', temp_name)
