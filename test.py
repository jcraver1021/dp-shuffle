"""Simple tests for the components; use the notebook for the actual cases
"""

import longitudinal
import numpy as np

def normalize(x):
	"""Normalize a numpy array (x - min) / (max - min)
	Input:
		x: A numpy array
	Output:
		The same array normalized to between 0 and 1
	"""
	
	max = np.max(x)
	min = np.min(x)
	return (x - min) / (max - min)

class Instance:
	"""An instance of a N:1 client/server statistics collection, including the statistics kept about the whole thing
	
	To use:
	>>> t = test.Instance(n, d, k, e)
	>>> t.run(True, True)
	"""
	
	def __init__(self, n, d, k, epsilon):
		"""Initialize the instance
		Input:
			n: Number of clients
			d: Length of epoch (number of time periods)
			k: Maximum allowed state changes per client per epoch
			epsilon: Privacy budget
		Output:
			None
		Side Effects:
			clients, server objects are populated
			X (n x d+1 matrix of true value vectors) is computed
			dX (n x d matrix of differential vectors) is computed
			x_true (sum of true value at time t-1 across all clients) is computed
			f_true (sum of delta at time t across all clients) is computed
			reports is initialized to an empty list
			f_approx is initialized to a 0-vector of size d
			All these objects are publicly accessible
		"""
		
		# Store our inputs
		self.n = n
		self.d = d
		self.k = k
		self.epsilon = epsilon
		
		# Set up objects
		self.clients = []
		self.server = longitudinal.Server(self.d)
		X = []
		dX = []
		for i in range(self.n):
			dx = longitudinal.generate_dx(d, k)
			x = longitudinal.compute_x(dx)
			dX.append(dx)
			X.append(x)
			self.clients.append(longitudinal.Client(dx))
		self.X = np.array(X)
		self.dX = np.array(dX)
		
		# Fill in statistical objects (some will just be placeholders)
		self.reports = []
		self.f_true = np.sum(dX, axis=0)
		self.x_true = np.sum(X, axis=0)
		self.f_approx = np.zeros(self.d)
	
	def run(self, collect=True, shuffle=False, server_epsilon=None):
		"""Initialize the instance
		Input:
			collect: Whether to have the server collect the reports and run statistics
			shuffle: Whether to shuffle the reports before submitting to the server
			server_epsilon: Epsilon value to use in collection (use client version if absent)
		Output:
			None
		Side Effects:
			reports is populated with all reports per time period (list of lists)
			f_approx is initialized to a 0-vector of size d
			All these objects are publicly accessible
		"""
		
		# Collect the reports from each time period
		self.reports = []
		for t in range(self.d):
			# Get the reports from each client for this period
			self.reports.append([])
			for i in range(self.n):
				rep = self.clients[i].update(t, self.epsilon)
				# Not all clients will report in a given interval (see reporting level h in longitudinal.Client)
				if rep:
					self.reports[-1].append(rep)
			
			# Submit the reports to the server
			if collect:
				self.server.collect(t, self.reports[-1])
		
		# Once finished, have the server aggregate the reports and compute statistics
		if collect:
			self.f_approx = self.server.aggregate(self.k, server_epsilon if server_epsilon else self.epsilon)

def run_test(n, d, k, eps, collect=True, shuffle=False, server_epsilon=None):
	instance = Instance(n, d, k, eps)
	instance.run(collect, shuffle, server_epsilon)
	return instance

def print_stats(instance, print_server=False):
	print("parameters: n=%d, d=%d, k=%d, e=%0.2f" %(instance.n, instance.d, instance.k, instance.epsilon))
	print("sum(dX): ", instance.f_true)
	print("sum(X): ", instance.x_true)
	print("net: ", np.sum(instance.f_true))
	if print_server:
		print("server sum: ", instance.f_approx)
		diff = np.abs(instance.f_true - instance.f_approx)
		print("difference: ", diff)
		print("max deviation: ", np.max(diff))
		print("argmax deviation: ", np.argmax(diff))
		print("reported net: ", np.sum(instance.f_approx))

def test_single_client():
	print("Single Client Output")
	instance = run_test(1, 128, 32, 0.25)
	print_stats(instance)

def test_honest_clients():
	print("Honest Client Collection")
	instance = run_test(1024*512, 32, 4, 100, True)
	print_stats(instance, True)

def test_careful_clients():
	print("Careful Client Collection")
	instance = run_test(1024*512, 32, 4, 0.4, True)
	print_stats(instance, True)

# If called directly, run the test cases
if __name__ == "__main__":
   test_single_client()
   test_honest_clients()
   test_careful_clients()