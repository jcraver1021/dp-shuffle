"""Simple tests for the components; use the notebook for the actual cases
"""

import longitudinal
import numpy as np
import csv

def rescale(x, min, max):
	"""Rescale a numpy array (x - min) / (max - min)
	Input:
		x: A numpy array
		min: New floor (likely less than or equal to min(x))
		max: New ceiling (likely greater than or equal to max(x))
	Output:
		The same array normalized to between min and max
	"""
	
	return (x - min) / (max - min)

class Instance:
	"""An instance of a N:1 client/server statistics collection, including the statistics kept about the whole thing
	
	To use:
	>>> t = test.Instance(n, d, k, e)
	>>> t.run(True, True)
	"""
	
	def __init__(self, n, d, k, epsilon, hide_zero=True, choose_level=True):
		"""Initialize the instance
		Input:
			n: Number of clients
			d: Length of epoch (number of time periods)
			k: Maximum allowed state changes per client per epoch
			epsilon: Privacy budget
			hide_zero: Whether clients will hide their zeros (default True)
			choose_level: Whether clients will choose a reporting level at random (default True)
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
		self.hide_zero = hide_zero
		self.choose_level = choose_level
		
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
			self.clients.append(longitudinal.Client(dx, hide_zero, choose_level))
		self.X = np.array(X)
		self.dX = np.array(dX)
		
		# Fill in statistical objects (some will just be placeholders)
		self.reports = []
		self.f_true = np.array([np.sum(self.dX[:, :(i+1)]) for i in range(d)])
		self.x_true = np.sum(X, axis=0)
		self.f_approx = np.zeros(self.d)
	
	def run(self, collect=True, shuffle=False, server_epsilon=None):
		"""Initialize the instance
		Input:
			collect: Whether to have the server collect the reports and run statistics
			shuffle: Whether to shuffle the reports before submitting to the server
			server_epsilon: Epsilon value to use in collection (use client version if absent)
		Output:
			f_approx (also committed to state) (only if in server mode)
		Side Effects:
			reports is populated with all reports per time period (list of lists)
			f_approx is initialized to a 0-vector of size d
			All these objects are publicly accessible
		"""
		
		# Reset all clients before run (retaining existing secret bits)
		for client in self.clients:
			client.hide_zero(self.hide_zero)
			client.set_choose_level(self.choose_level)
			client.reset()
		
		# Collect the reports from each time period
		self.reports = []
		indices = np.arange(self.n)
		for t in range(self.d):
			# Get the reports from each client for this period
			self.reports.append([])
			# Shuffle the client order before querying (simulating algorithms 3 and 4 from the paper)
			if shuffle:
				np.random.shuffle(indices)
			for i in indices:
				rep = self.clients[i].update(t, self.epsilon)
				self.reports[-1].append(rep)
			
			# Submit the reports to the server
			if collect:
				self.server.collect(t, self.reports[-1])
		
		# Once finished, have the server aggregate the reports and compute statistics
		if collect:
			self.f_approx = self.server.aggregate(self.k, server_epsilon if server_epsilon else self.epsilon)
			return self.f_approx
	
	def write_reports(self, filename):
		"""Write the latest client reports to a file in whatever order they were received (since you don't know that the server won't do that IRL)
		Input:
			filename: Filename of the output
		Output:
			None
		"""
		with open(filename, 'w', newline='') as csvfile:
			repwriter = csv.writer(csvfile)
			repwriter.writerow(["t = %d" % i for i in range(self.d)])
			for i in range(self.n):
				row = []
				for t in range(self.d):
					r = self.reports[t][i]
					if r:
						row.append(str(r[2]))
					else:
						row.append('')
				repwriter.writerow(row)

def run_test(n, d, k, eps, collect=True, shuffle=False, server_epsilon=None, hide_zero=True):
	instance = Instance(n, d, k, eps, hide_zero)
	instance.run(collect, shuffle, server_epsilon)
	return instance

def print_stats(instance, print_server=False):
	print("parameters: n=%d, d=%d, k=%d, e=%0.2f" %(instance.n, instance.d, instance.k, instance.epsilon))
	if not instance.hide_zero:
		print("Zeros not hidden")
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

def test_naive_clients():
	print("Naive Client Collection")
	instance = run_test(1024*512, 32, 4, 100, True, False, None, True)
	print_stats(instance, True)

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
   test_naive_clients()
   test_honest_clients()
   test_careful_clients()