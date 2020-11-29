import gurobipy as gp
from gurobipy import GRB
import numpy as np
import create_graph


def rectangle_sets_from_start_end(tile_shape, start, end):
	assert(start % tile_shape[1] <= end % tile_shape[1])

	total_set = set(list(range(tile_shape[0] * tile_shape[1])))
	one_set, zero_set = set(), set()
	for i in range(start, end+1):
		if i % tile_shape[1] < start % tile_shape[1]:
			continue
		if i % tile_shape[1] > end % tile_shape[1]:
			continue
		one_set.add(i)
	zero_set = total_set - one_set

	return one_set, zero_set


def optimization_solver(cameras, cam_to_tshape, model, time_window):

	cam_to_tcount = {cam: cam_to_tshape[cam][0]*cam_to_tshape[cam][1] for cam in cam_to_tshape}

	multi_hashmap, time_to_obj = create_graph.multi_cam_hashmap(cameras, 'rcnn', time_window)

	try:
		time_obj_len = sum([len(each) for each in time_to_obj.values()])

		# Create a new model
		model = gp.Model("optimization")

		x_shape_list = []
		for i in range(len(cameras)):
			x_shape_list.extend([(i, j) for j in range(cam_to_tcount[cameras[i]])])

		r_shape_list = []
		for c in range(len(cameras)):
			for start in range(cam_to_tcount[cameras[c]]):
				for end in range(start, cam_to_tcount[cameras[c]]):
					if end % cam_to_tshape[cameras[c]][1] < start % cam_to_tshape[cameras[c]][1]:
						continue
					r_shape_list.append((c, start, end))
		
		print(len(r_shape_list))

		# Create variables
		x = model.addVars(x_shape_list, vtype=GRB.INTEGER, lb=[0]*len(x_shape_list), ub=[1]*len(x_shape_list))
		I = model.addVars(time_obj_len, len(cameras), vtype=GRB.BINARY)
		r = model.addVars(r_shape_list, vtype=GRB.BINARY)

		print("Variables added")

		# Set objective
		model.setObjective(gp.quicksum(x), GRB.MINIMIZE)

		# Add Constraints
		counter = 0
		for t in range(time_window[1]):
			for obj in time_to_obj[t]:
				for c in range(len(cameras)):
					if (t, obj) not in multi_hashmap[cameras[c]]:
						model.addConstr(I[counter, c] == 0)
					else:
						p = np.zeros(cam_to_tcount[cameras[c]])
						for pos in multi_hashmap[cameras[c]][(t, obj)][0]:
							p[pos] = 1
						model.addConstr((I[counter, c] == 1) >> \
							(gp.quicksum([p[j] - x[c, j] * p[j] for j in range(cam_to_tcount[cameras[c]])]) == 0))
						model.addConstr((I[counter, c] == 0) >> \
							(gp.quicksum([p[j] - x[c, j] * p[j] for j in range(cam_to_tcount[cameras[c]])]) >= 1))
				model.addConstr(gp.quicksum(I[counter, c] for c in range(len(cameras))) >= 1)
				counter += 1

		for c in range(len(cameras)):
			print(cameras[c])
			for start in range(cam_to_tcount[cameras[c]]):
				for end in range(start, cam_to_tcount[cameras[c]]):
					if end % cam_to_tshape[cameras[c]][1] < start % cam_to_tshape[cameras[c]][1]:
						continue
					one_set, zero_set = rectangle_sets_from_start_end(cam_to_tshape[cameras[c]], start, end)
					model.addConstrs((r[c, start, end] == 1) >> (x[c, j] == 1) for j in one_set)
					model.addConstrs((r[c, start, end] == 1) >> (x[c, j] == 0) for j in zero_set)

		for c in range(len(cameras)):
			can_to_r_shape_list = [tup for tup in r_shape_list if tup[0] == c]
			model.addConstr(gp.quicksum(r[c, start, end] for c, start, end in can_to_r_shape_list) == 1) 

		print("Constraints added")

		# Optimize model
		model.optimize()

	except gp.GurobiError as e:
		print('Error code ' + str(e.errno) + ": " + str(e))

	except AttributeError:
		print('Encountered an attribute error')

	result = {cam: [] for cam in cameras}
	for i in range(len(cameras)):
		for j in range(cam_to_tcount[cameras[i]]):
			if x[i,j].x == 1.0:
				result[cameras[i]].append(j)

	for c in range(len(cameras)):
		for start in range(cam_to_tcount[cameras[c]]):
			for end in range(start, cam_to_tcount[cameras[c]]):
				if end % cam_to_tshape[cameras[c]][1] < start % cam_to_tshape[cameras[c]][1]:
					continue
				if r[c, start, end].x == 1.0:
					print(c, start, end)

	return result


if __name__ == '__main__':
	cameras = ['c001', 'c002', 'c003', 'c004']
	cam_to_tcount = {'c001': 144, 'c002': 144, 'c003': 144, 'c004': 144}
	result = optimization_solver(cameras, cam_to_tcount, 'rcnn', [0, 600])