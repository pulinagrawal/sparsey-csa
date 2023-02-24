import torch

class Macrocolumn(object):
    def __init__(self, size, inp_size, sparsity, lr=.1) -> None:
        self.size = size
        self.inp_size = inp_size
        self.sparsity = sparsity 
        self.lr = lr
        self.num_active = int(sparsity*size)
        self.weights = torch.rand((size, 5))
        self.maps = torch.randint(0, inp_size, (size, 5))
        self.out = torch.rand((size,))

    def run(self, input, learn=True) -> torch.Tensor:
        selected_code = torch.topk(torch.cosine_similarity(input[self.maps], self.weights), k=self.num_active)
        code = torch.scatter(torch.zeros_like(self.out), 0, selected_code.indices, torch.ones_like(selected_code.values))
        if learn:
            self.learn(input, selected_code)
        return code

    def learn(self, input, selected_code):
            learn_weights = self.weights.index_select(0, selected_code.indices)
            dv = learn_weights-input[0]
            length = torch.norm(dv, dim=1)
            uv = dv/length.unsqueeze(1)
            learn_weights = learn_weights-uv*(self.lr*length).unsqueeze(1)
            self.weights[selected_code.indices] = learn_weights

class SegmentedMacrocolumn(object):
    def __init__(self, n_pyramidals, inp_size, sparsity, n_init_segments=2,
                 n_synapses=30, lr=1., similarity_threshold=.95) -> None:
        
        n_pyramidals = 2048
        inp_size = 512
        n_init_segments = 2
        n_synapses = 30
        lr = 1.
        similarity_threshold = .95 # this value makes or breaks the test cases.
        inp = torch.rand((inp_size, ))
        class S:
            pass
        # self = S()

        self.n_pyramidals = n_pyramidals
        self.thresh = similarity_threshold
        self.lr = lr
        self.n_segments = n_init_segments
        self.n_synapses = n_synapses
        self.inp_size = inp_size
        self.topk = int(self.n_pyramidals*sparsity)
        self.mc_pyramidal_map = torch.randint(0, self.inp_size, (self.n_pyramidals, self.n_segments, self.n_synapses))
        self.mc_pyramidal_wts = torch.rand((n_pyramidals, self.n_segments, self.n_synapses))

    def run(self, inp, learn=True) -> torch.Tensor:
        overlap = torch.cosine_similarity(inp[self.mc_pyramidal_map], self.mc_pyramidal_wts, dim=2)
        thresh_overlap = torch.nn.functional.threshold(overlap, self.thresh, 0.)

        # Add a buffer column that is lower than the threshold so that argmax gives the buffer as the 
        # segment which can be weeded out by only looking at nonzeros
        buffer = torch.ones((self.n_pyramidals, 1))*(self.thresh-.01)
        clamped_thresh_overlap = torch.hstack((buffer, thresh_overlap))
        best_segments = torch.argmax(clamped_thresh_overlap, dim=1)
        potential_pyramidals = best_segments.nonzero(as_tuple=False)

        # need to subtract 1 to get real segment index because a buffer column was added before
        potential_segments = best_segments[best_segments!=0].unsqueeze(1)-1
        indexes = torch.hstack((potential_pyramidals, potential_segments))
        chosen_pyramidal_ids = torch.LongTensor([])
        selected_indexes = []
        if len(potential_pyramidals)<self.topk:
            mask = torch.scatter(torch.ones((self.n_pyramidals,), dtype=torch.bool), 0, potential_pyramidals.flatten(), torch.zeros_like(potential_pyramidals.flatten(), dtype=torch.bool))
            assert not any(mask[potential_pyramidals[i]] for i in range(len(potential_pyramidals))) or len(potential_pyramidals)==0
            assert any(mask[potential_pyramidals[i]-1] for i in range(len(potential_pyramidals)) if potential_pyramidals[i]<len(mask)-1 and potential_pyramidals[i]>0) or len(potential_pyramidals)==0
            inactive_pyramidals = torch.masked_select(torch.LongTensor(list(range(self.n_pyramidals))), mask)
            chosen_indexes = torch.randperm(len(inactive_pyramidals))[:self.topk-len(indexes)]
            chosen_pyramidal_ids = inactive_pyramidals[chosen_indexes]
            assert len(set(chosen_pyramidal_ids).intersection(set(potential_pyramidals)))==0

            # Create len(indexes)-topk new random receptive field segments
            new_segments = torch.randint(0, self.inp_size, (self.n_pyramidals, 1, self.n_synapses))
            new_mc_pyramidal_map = torch.cat((self.mc_pyramidal_map.transpose(1,0), new_segments.transpose(1,0)))
            assert_save = self.mc_pyramidal_map[0, 1]
            self.mc_pyramidal_map = new_mc_pyramidal_map.transpose(0,1)
            self.n_segments += 1
            assert not torch.all(torch.BoolTensor((self.mc_pyramidal_map[0, 1]-assert_save).bool()))

            new_weights = torch.rand((self.n_pyramidals, 1, self.n_synapses))
            new_weights[chosen_pyramidal_ids] = inp[new_segments[chosen_pyramidal_ids]]
            new_mc_pyramidal_wts = torch.cat((self.mc_pyramidal_wts.transpose(1,0), new_weights.transpose(1,0)))
            self.mc_pyramidal_wts = new_mc_pyramidal_wts.transpose(0,1)
            # such that their weights exactly match their receptive field inputs
            # distribute these segments among non active pyramidals
        #elif len(indexes)>topk:
            # perform inhibition
            # each inhibitory neuron with more than 50% segments active 
        else:
            index_overlaps = collect_indices(overlap, indexes)
            selected_pyramidals = torch.topk(index_overlaps, k=self.topk).indices
            non_selected_pyramidals = torch.topk(index_overlaps, k=len(index_overlaps)-self.topk).indices
            selected_pyramidal_ids = potential_pyramidals[selected_pyramidals].flatten()
            non_selected_pyramidal_ids = potential_pyramidals[non_selected_pyramidals].flatten()
            selected_segments = potential_segments[selected_pyramidals]
            selected_indexes = torch.hstack([selected_pyramidal_ids.unsqueeze(1), selected_segments])
            assert not set(non_selected_pyramidal_ids).intersection(set(selected_pyramidal_ids))
        chosen_pyramidal_ids = torch.cat([potential_pyramidals.flatten(), chosen_pyramidal_ids])
        if learn and len(selected_indexes)>0:
            self._learn(inp, selected_indexes)
        return torch.scatter(torch.zeros((self.n_pyramidals,), dtype=torch.long), 0, chosen_pyramidal_ids, torch.ones_like(chosen_pyramidal_ids, dtype=torch.long))

    def _learn(self, input, indexes):
            learn_weights = self.mc_pyramidal_wts[indexes[:, 0], indexes[:, 1]]
            inp_values = input[self.mc_pyramidal_map[indexes[:, 0], indexes[:, 1]]]
            dv = learn_weights-inp_values
            length = torch.norm(dv, dim=1)
            uv = dv/length.unsqueeze(1)
            learn_weights = learn_weights-uv*(self.lr*length).unsqueeze(1)
            self.mc_pyramidal_wts[indexes[:, 0], indexes[:, 1]] = learn_weights

def collect_indices(tensor, indices):
    mask = torch.zeros_like(tensor).index_put_(tuple(indices.t()), torch.ones((indices.shape[0],))).to(dtype=torch.bool)
    return torch.masked_select(tensor, mask)

'''
inp_size = 512
import torch
input = torch.rand(inp_size)
n_segments = 2
n_synapses = 5
n_pyramidals = 2048
sparsity = .02
topk = int(n_pyramidals*sparsity)
mc_pyramidal_map = torch.randint(0, inp_size, (n_pyramidals, n_segments, n_synapses))
mc_pyramidal_wts = torch.rand((n_pyramidals, n_segments, n_synapses))
thresh = .95
overlap = torch.cosine_similarity(input[mc_pyramidal_map], mc_pyramidal_wts, dim=2)
input[mc_pyramidal_map].shape
mc_pyramidal_wts.shape
overlap.shape
thresh_overlap = torch.nn.functional.threshold(overlap, thresh, 0.)
thresh_overlap.shape
buffer = torch.ones((n_pyramidals,1))*(thresh-.01)
buffer.shape
clamped_thresh_overlap = torch.hstack((buffer, thresh_overlap))
selected_segment = torch.argmax(clamped_thresh_overlap, dim=1)
pyramidals = selected_segment.nonzero(as_tuple=False)
segments = selected_segment[selected_segment!=0].unsqueeze(1)-1
indexes = torch.hstack((pyramidals, segments))
len(indexes)
# indexes
# All overlapping segments that are above the threshold 
#                               and most active segments in the pyramidal
if len(indexes)<topk:
    mask = torch.scatter(torch.ones((n_pyramidals,), dtype=torch.bool), 0, pyramidals.flatten(), torch.zeros_like(pyramidals.flatten(), dtype=torch.bool))
    inactive_pyramidals = torch.masked_select(torch.LongTensor(list(range(n_pyramidals))), mask)
    chosen_indexes = torch.randperm(len(inactive_pyramidals))[:10]
    chosen_pyramidal_ids = inactive_pyramidals[chosen_indexes]

    # Create len(indexes)-topk new random receptive field segments
    new_segments = torch.randint(0, inp_size, (n_pyramidals, 1, n_synapses))
    mc_pyramidal_map.shape, new_segments.shape
    new_mc_pyramidal_map = torch.cat((mc_pyramidal_map.transpose(1,0), new_segments.transpose(1,0)))
    new_mc_pyramidal_map.transpose(0,1).shape

    new_weights = torch.rand((n_pyramidals, 1, n_synapses))
    new_weights[chosen_pyramidals] = input[new_segments[chosen_pyramidals]]
    new_mc_pyramidal_wts = torch.cat((mc_pyramidal_wts.transpose(1,0), new_weights.transpose(1,0)))
    new_mc_pyramidal_wts.transpose(0,1).shape
    # such that their weights exactly match their receptive field inputs
    # distribute these segments among non active pyramidals
#elif len(indexes)>topk:
    # perform inhibition
    # each inhibitory neuron with more than 50% segments active 
index_overlaps = collect_indices(overlap, indexes)
len(index_overlaps)
selected_pyramidals = torch.topk(index_overlaps, k=topk).indices
non_selected_pyramidals = torch.topk(index_overlaps, k=len(index_overlaps)-topk).indices
non_selected_pyramidals
selected_pyramidal_ids = pyramidals[selected_pyramidals].flatten()
non_selected_pyramidal_ids = pyramidals[non_selected_pyramidals].flatten()
non_selected_pyramidal_ids

assert not set(non_selected_pyramidal_ids).intersection(set(selected_pyramidal_ids))
selected_segment_ids = indexes[selected_pyramidals]

selected_pyramidals
selected_pyramidal_ids = pyramidals[selected_pyramidals].flatten()
chosen_pyramidal_ids = torch.cat([selected_pyramidal_ids, chosen_pyramidal_ids])
torch.scatter(torch.zeros((n_pyramidals,), dtype=torch.float), 0, chosen_pyramidal_ids, torch.ones_like(chosen_pyramidal_ids, dtype=torch.float))


def topk(tensor, k, dim=-1):
    if dim<0:
        v, i = torch.topk(tensor.flatten(), k, dim)
        import numpy as np
        return v, (torch.LongTensor(np.unravel_index(i.numpy(), tensor.shape)).T)
    else:
        return torch.topk(tensor.flatten(), k, dim)
'''