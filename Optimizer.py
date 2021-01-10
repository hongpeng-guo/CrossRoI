import gurobipy as gp
from gurobipy import GRB
import numpy as np
import CreateGraph

def neighbor_list (tile_shape, idx):
	result = []
	if (idx - 1) >= 0 and idx % tile_shape[1] > 0:
		result.append(idx - 1)
	if (idx + 1) < tile_shape[0] * tile_shape[1] and idx % tile_shape[1] < tile_shape[1] - 1:
		result.append(idx + 1)
	if idx - tile_shape[1] >= 0:
		result.append(idx - tile_shape[1])
	if idx + tile_shape[1] < tile_shape[0] * tile_shape[1]:
		result.append(idx + tile_shape[1])
	return result

def optimization_solver(cameras, cam_to_tshape, time_window, gt_multi_hashmap=None):

	cam_to_tcount = {cam: cam_to_tshape[cam][0]*cam_to_tshape[cam][1] for cam in cam_to_tshape}

	multi_hashmap, time_to_obj = CreateGraph.multi_cam_hashmap(cameras,  time_window, gt_multi_hashmap)

	try:
		time_obj_len = sum([len(each) for each in time_to_obj.values()])

		# Create a new model
		model = gp.Model("optimization")

		x_shape_list = []
		for i in range(len(cameras)):
			x_shape_list.extend([(i, j) for j in range(cam_to_tcount[cameras[i]])])

		# Create variables
		x = model.addVars(x_shape_list, vtype=GRB.INTEGER, lb=[0]*len(x_shape_list), ub=[1]*len(x_shape_list))
		I = model.addVars(time_obj_len, len(cameras), vtype=GRB.BINARY)

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
						for pos in multi_hashmap[cameras[c]][(t, obj)]:
							p[pos] = 1
						model.addConstr((I[counter, c] == 1) >> \
							(gp.quicksum([p[j] - x[c, j] * p[j] for j in range(cam_to_tcount[cameras[c]])]) == 0))
						model.addConstr((I[counter, c] == 0) >> \
							(gp.quicksum([p[j] - x[c, j] * p[j] for j in range(cam_to_tcount[cameras[c]])]) >= 1))
				model.addConstr(gp.quicksum(I[counter, c] for c in range(len(cameras))) >= 1)
				counter += 1

		# for c in range(len(cameras)):
		# 	model.addConstrs((x[c, j] == 0) >> \
		# 		(gp.quicksum(x[c, k] for k in neighbor_list(cam_to_tshape[cameras[c]], j)) <= 1) for j in range(cam_to_tcount[cameras[c]]))

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
			if round(x[i,j].x) == 1.0:
				result[cameras[i]].append(j)

	return result