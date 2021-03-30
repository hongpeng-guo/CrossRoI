import numpy as np


class CountingMetrics():

    def __init__(self):

        self.identifier = 'counting'
        self.name = f'{self.identifier}'

        self.ground_truth = {}
        self.ground_truth_frames = set()
        self.comparision_frames = set()


    def evaluate(self):
        score_list = []
        cmp_prev_frame = min(self.ground_truth_frames)

        for frame_id in self.ground_truth_frames:
            gt_found = self.ground_truth[frame_id]
            if frame_id in self.comparision_frames:
                cmp_found = self.ground_truth[frame_id]
                cmp_prev_frame = frame_id
            else:
                cmp_found = self.ground_truth[cmp_prev_frame]
            
            if frame_id - cmp_prev_frame > 5:
                cmp_found = 0

            if max(gt_found, cmp_found) == 0 or gt_found == cmp_found:
                score_list.append(1)
            else:
                difference = abs(gt_found - cmp_found)
                base = max(gt_found, cmp_found)
                score = (base - difference) / base
                score_list.append(score)
        results = {
            self.name: sum(score_list) / len(score_list)
        }
        return results


    def add_ground_truth(self, gt_count_dict):
        self.ground_truth_frames = set(gt_count_dict.keys())
        self.ground_truth = gt_count_dict

    def add_comparision(self, comp_count_frames):
        self.comparision_frames = set(comp_count_frames)


    def reset(self):
        self.ground_truth = {}
        self.comparision = {}
        self.ground_truth_frames = set()
        self.comparision_frames = set()