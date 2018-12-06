"""Simple tests for the components; use the notebook for the actual cases
"""

import longitudinal
import numpy as np

def run_test(n, d, k, eps, collect=None, shuffle=False):
	clients = []
	server = longitudinal.Server(d)
	for i in range(n):
		print("Client %d" % i)
		dx = longitudinal.generate_dx(d, k)
		#print(dx)
		clients.append(longitudinal.Client(dx))
	for t in range(d):
		print("t = %d" % t)
		reports = []
		for i in range(n):
			rep = clients[i].update(t, eps)
			if rep:
				#print(rep)
				reports.append(rep)
		if collect:
			server.collect(t, reports)
	if collect:
		f = server.aggregate(k, collect)
		print(f)
		f_true = np.sum(np.array([clients[i].dx for i in range(n)]), axis=0)
		print(f_true)

# Test 1: Single-client output
print("Single-client output")
run_test(1, 16, 8, 0.25)

# Test 1: Honest Clients
print("Honest Clients")
run_test(1024*128, 32, 4, 100, 100)