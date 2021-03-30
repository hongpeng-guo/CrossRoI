from reducto.hashbuilder.hash_builder import ThreshMap
from reducto.differencer import AreaDiff
from reducto.evaluator import CountingMetrics
from reducto.hashbuilder import HashBuilder, ThreshMap

differ_type = 'area'

def get_segmented_diff_vectors(video_path, segment_size=10, segment_limit=180):
    area_dp = AreaDiff()
    diff_vector_all = area_dp.get_diff_vector(video_path)
    diff_vectors = [diff_vector_all[i:i+segment_size] \
                    for i in range(0, len(diff_vector_all), segment_size)]
    diff_vectors = [each[:-1] for each in diff_vectors if len(each) == segment_size]

    diff_vectors = diff_vectors[:segment_limit]

    return diff_vectors


def get_segmented_diff_results(diff_vectors, thresholds):
    diff_results = []
    for diff_vector in diff_vectors:
        diff_results.append(AreaDiff.batch_diff_noobj(diff_vector, thresholds))

    return diff_results


def get_segmented_evaluations(diff_results, gt_count_dict):
    counting_evaluator = CountingMetrics()
    evaluations = []
    segment_size = list(diff_results[0].values())[0]['num_total_frames']

    for seg_id, diff_result in enumerate(diff_results):
        evaluation = {differ_type: {}}
        for threshold in diff_result:
            segmented_cmp_frames = [seg_id * segment_size + each for each in diff_result[threshold]['selected_frames']]
            segmented_gt_count_dict = {f: gt_count_dict[f] \
                            for f in range(seg_id * segment_size + 1, (seg_id +1 )* segment_size + 1)}
            counting_evaluator.add_ground_truth(segmented_gt_count_dict)
            counting_evaluator.add_comparision(segmented_cmp_frames)
            res = counting_evaluator.evaluate()
            evaluation[differ_type][threshold] = res
            counting_evaluator.reset()
        evaluations.append(evaluation)

    return evaluations


def generate_hashmap(evaluations, diff_vectors, target_acc=1.0):
    diff_vectors = [{differ_type: vector} for vector in diff_vectors]
    threshmap_init_dict = HashBuilder().generate_threshmap(
                    evaluations,
                    diff_vectors,
                    target_acc=target_acc)

    thresh_map = ThreshMap(threshmap_init_dict[differ_type])

    return thresh_map


def generate_test_result(test_diff_results, pos_offset, gt_count_dict, thresh_map):
    selected_frames = set()
    for i, seg_vector in enumerate(test_diff_results):
        thresh, _ = thresh_map.get_thresh(seg_vector)
        result = AreaDiff.batch_diff_noobj(seg_vector, [thresh])

        segment_size = result[thresh]['num_total_frames']
        if len(result[thresh]['selected_frames']) == 0:
            print('Exception')
        for each_frame in result[thresh]['selected_frames']:
            selected_frames.add(pos_offset + i * segment_size + each_frame)

    print(len(selected_frames), len([key for key in gt_count_dict.keys() if key > pos_offset]))

    return selected_frames
