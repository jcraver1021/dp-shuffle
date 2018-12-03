import numpy as np

def tree_depth_list(d):
	return np.arange(np.log2(d) + 1, dtype=int)

class Client:
	"""A client, which contains longitudinal binary data and will report at certain time periods a randomized response based on that data
	
	To use:
	>>> client = longitudinal.Client(binaryArray)
	>>> for t in range(d):
	>>> 	client.update(t, eps)
	"""
	
	def __init__(self, x):
		"""Initialize the client (simply calls self.reset())
		Input:
			x (array): The longitudinal secret bits (x[t] = value of x at time t)
		Output:
			None
		Side Effects:
			See "reset"
		"""
		self.reset(x)
	
	def reset(self, x):
		"""Reset the client to have the given bits and call self.setup(d, k) is called, where d is the length of dx and k is the number of nonzero entries of dx
		Input:
			x (array): The longitudinal secret bits (x[t] = value of x at time t)
		Output:
			None
		Side Effects:
			dx = the discrete derivative of the elements of x (i.e. elements are \in {-1, 0, 1})
			See "setup"
		"""
		self.__x = x
		self.__dx = np.ediff1d(self.__x)
		self.__setup(len(self.__dx), int(np.linalg.norm(self.__dx, 1)))
	
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
		self.__ic = np.random.choice(np.arange(k))			# Choose a change to report
		self.__hc = np.random.choice(tree_depth_list(d))	# Choose a level of the tree to report
		self.__i = 0										# Counters (changes seen)
		self.__c = 0										# Value of change seen
		self.k = k											# Number of times this client changes its value
	
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
			u = np.random.choice([-1, 1]) 
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
		self.d = d
		self.T = {}
	
	def collect(self, t, reports):
		# Each report is (h, t, u), where h is the level, t is the time, and u is the value
		if reports:
			for report in reports:
				if report:
					key = (report[0] + 1, int((report[1] + 1) / (2 ** report[0])))
					if key not in self.T:
						self.T[key] = 0
					self.T[key] += report[2]
	
	def aggregate(self, k, eps):
		f = []
		a = np.log2(self.d) * (np.exp(eps / 2) + 1) / (np.exp(eps / 2) - 1)
		for t in range(self.d):
			C = set([(1, i + 1) for i in range(t + 1)])
			for h in tree_depth_list(self.d):
				for i in range(self.d):
					if i % 2 == 0:
						key1 = (h + 1, i + 1)
						key2 = (h + 1, i + 2)
						if key1 in C and key2 in C:
							C.remove(key1)
							C.remove(key2)
							C.add(((h + 2), int((i + 2) / 2)))
			f.append(a * k * np.sum([self.T[c] for c in C if c in self.T]))
		return f
