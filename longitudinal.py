import numpy as np

def tree_depth_list(d):
	"""Return a list of numbers from 0 to log2(d)
	Input:
		Number of elements
	Output:
		List of tree rows
	"""
	return np.arange(np.log2(d) + 1, dtype=int)

def generate_dx(d, k):
	"""Generate a differential vector of length d with k changes
	Input:
		d: Length of array
		k: Number of changes
	Output:
		dx: vector with k changes, alternating -1 and 1 (with random start point)
	"""
	# There are k changes allowed, so start there
	dx = np.concatenate((np.ones(k, dtype=int), np.zeros(d - k, dtype=int)))
	
	# Randomize the location of these changes
	np.random.shuffle(dx)
	
	# Decide whether we start from 0 or 1 (-1, 1 from derivative's perspective)
	c = np.random.choice([-1, 1])
	
	# Every other 1 becomes a -1
	for i in range(len(dx)):
		if dx[i] != 0:
			dx[i] = c
			c *= -1
	
	# Return result
	return dx

def compute_x(dx):
	"""Compute x based on dx
	Input:
		dx: differential vector
	Output:
		x: true values at time t - 1
	"""
	# Find the first nonzero element of dx and report the opposite (since x is a binary array)
	x = [int((-dx[np.nonzero(dx)[0][0]] + 1) / 2)]
	
	# Compute the marginal sums for each element
	for i in range(len(dx)):
		x.append(x[i] + dx[i])
	
	# Return the result
	return x

class Client:
	"""A client, which contains longitudinal binary data and will report at certain time periods a randomized response based on that data
	
	To use:
	>>> client = longitudinal.Client(binaryArray)
	>>> for t in range(d):
	>>> 	client.update(t, eps)
	"""
	
	def __init__(self, dx, hide_zero=True, choose_level=True):
		"""Initialize the client (simply calls self.reset())
		Input:
			dx (array): The longitudinal secret bits as a discrete differential (where x[t] = value of x at time t, dx[t] = x[t] - x[t-1])
			hide_zero: Whether to report 1 or -1 (with equal probability) if no value at time t (default True)
		Output:
			None
		Side Effects:
			See "reset"
		"""
		self.__hide_zero = hide_zero
		self.__choose_level = choose_level
		self.reset(dx)
	
	def reset(self, dx=None):
		"""Reset the client to have the given bits and call self.setup(d, k) is called, where d is the length of dx and k is the number of nonzero entries of dx
		Input:
			dx (array): The longitudinal secret bits as a discrete differential; if None, the previous value is retained
		Output:
			None
		Side Effects:
			See "setup"
		"""
		if not dx is None:
			self.__dx = dx
		self.__setup(len(self.__dx), int(np.linalg.norm(self.__dx, 1)))
	
	def hide_zero(self, hide_zero=True):
		"""Set whether we hide our zeros on reporting
		Input:
			hide_zero: True (default) or False
		Output:
			None
		Side Effects:
			The client setting
		"""
		self.__hide_zero = hide_zero
	
	def set_choose_level(self, choose_level=True):
		"""Set whether we choose a random level on which to report
		Input:
			choose_level: True (default) or False
		Output:
			None
		Side Effects:
			The client setting
		"""
		self.__choose_level = choose_level
	
	def __setup(self, d, k):
		"""Setup the client's counters after reset
		Input:
			d: The length of dx
			k: The number of nonzero elements in dx
		Output:
			k: Number of times the client's value changes
		Side Effects:
			i* (here ic) = the change to report in this epoch (sampled at random from 0 up to k)
			h* (here hc) = the row of the implicit summary binary tree on which to report in this epoch (sampled at random from 0 up to log2(d) + 1)
			i = 0 (number of changes encountered during update step)
			c = 0 (i*th change to report, when found)
		"""
		self.__ic = np.random.choice(np.arange(k)) if k > 0 else 0	# Choose a change to report
		self.__get_level(d)											# Choose a level of the tree to report
		self.__i = 0												# Counters (changes seen)
		self.__c = 0												# Value of change seen
		self.k = k													# Number of times this client changes its value
	
	def __get_level(self, d):
		"""Choose the level on which to report (based on settings)
		Input:
			d: Length of epoch
		Output:
			None
		Side Effects:
			h*: choose a random level, or 1 if we're set to do that
		"""
		# This setting is not in the paper, but we add it to demonstrate the purpose of coarse reporting
		if self.__choose_level:
			self.__hc = np.random.choice(tree_depth_list(d))
		else:
			self.__hc = 0
	
	def update(self, t, eps):
		"""Report on the update for time t (depending on which level of tree we are on, expect an update every d / 2^h* times)
		Input:
			t: The time on which to report
			eps: Privacy parameter (epsilon)
		Output:
			(h*, t, update) if t | 2^h*, None otherwise
			Note, that if this is not the i*th change, the output will be sampled from {-1, 1} at random. This is to provide privacy by hiding the actual change using noise that is expected to cancel out during server summation
			Note also that the client implements randomized response, reporting the opposite answer based on a Bernoulli random variable with parameter (np.exp(eps / 2) / (1 + np.exp(eps / 2)))
		Side Effects:
			c = 0 (if we report the change), meaning that the client will ONLY report the i*th change
		"""
		# Increment the counter if we see a change, store in c if it is the i*th change
		if self.__dx[t] != 0:
			if self.__i == self.__ic:
				self.__c = self.__dx[t]
			self.__i += 1
		
		# Report if we reach a tree-level-based milestone
		if (t + 1) % (2 ** self.__hc) == 0:
			u = np.random.choice([-1, 1]) if self.__hide_zero else 0 # This setting (not in the paper) allows the client to be honest
			if self.__c != 0:
				# randomized response
				b = 2 * np.random.binomial(1, np.exp(eps / 2) / (1 + np.exp(eps / 2))) - 1
				u = b * self.__c
				self.__c = 0 # do not report any more true changes
			return (self.__hc, t, u)

class Server:
	"""A client, which contains longitudinal binary data and will report at certain time periods a randomized response based on that data
	
	To use:
	>>> server = longitudinal.Server(binaryArray)
	>>> for t in range(d):
	>>> 	... gather reports ...
	>>> 	server.collect(t, reports)
	>>> server.aggregate(eps)
	"""
	
	def __init__(self, d):
		"""Initialize the server
		Input:
			d: Size of time horizon
		Output:
			None
		Side Effects:
			T = map of (level, time) to observed sum
		"""
		self.d = d
		self.T = {}
	
	def collect(self, t, reports):
		"""Collect all reports for time period t
		Input:
			t: time period of reports
			reports: array of reports (h, t, u) from each reporting client
		Output:
			None
		Side Effects:
			T[(h, t)] is updated with the sum of all u entries for each h and t in the reports
		"""
		# Each report is (h, t, u), where h is the level, t is the time, and u is the value
		if reports:
			for report in reports:
				if report: # Not all clients will report in a given interval (see reporting level h in longitudinal.Client)
					key = (report[0] + 1, int((report[1] + 1) / (2 ** report[0])))
					if key not in self.T:
						self.T[key] = 0
					self.T[key] += report[2]
	
	def aggregate(self, k, eps):
		"""Aggregate all seen reports into the marginal estimates
		Input:
			k: Number of in-epoch changes per client
			eps: Noise estimate (privacy budget)
		Output:
			Array of marginal estimates (f[t] = marginal estimate at time t)
		"""
		# f is the estimate, a is the scaling factor
		f = []
		a = k * np.log2(self.d) * (np.exp(eps / 2) + 1) / (np.exp(eps / 2) - 1)
		
		# Go by time period
		for t in range(self.d):
			# Initialize the tree from the leaves
			C = set([(1, i + 1) for i in range(t + 1)])
			for h in tree_depth_list(self.d):
				for i in range(self.d):
					# Fold nodes up to their parents (if present before time t)
					if i % 2 == 0:
						key1 = (h + 1, i + 1)
						key2 = (h + 1, i + 2)
						key3 = (h + 2, int((i + 2) / 2))
						if key1 in C and key2 in C and key3 in self.T:
							C.remove(key1)
							C.remove(key2)
							C.add(key3)
			# Sum up all the remaining nodes into this time period's estimate
			f.append(a * np.sum([self.T[c] for c in C if c in self.T]))
		
		# Return all estimates
		return f
