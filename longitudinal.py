import numpy as np

class Client:
	"""A client, which contains longitudinal binary data and will report at certain time periods a randomized response based on that data
	
	To use:
	>>> client = Client(binaryArray)
	"""
	
	def __init(self, x):
		self.__reset(x)
	
	def __reset(self, x):
		self.__x = x
		self.__setup(len(self.__x), int(np.linalg.norm(self.__x, 1)))
	
	def __setup(self, d, k):
		pass
		
class Server:
	pass