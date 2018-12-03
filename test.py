"""Simple tests for the components; use the notebook for the actual cases
"""

import longitudinal
import numpy as np

# Test 1: Single-client output
print("Single-client output")
d = 16
x = np.random.randint(2, size=d + 1)
c = longitudinal.Client(x)
eps = 0.1
print(x)
for i in range(d):
	print(i, c.update(i, eps))

# Test 2: Multiclient-client output
print("Multi-client output")
d = 16
n = 8
c = []
eps = 0.1
for i in range(n):
	x = np.random.randint(2, size=d + 1)
	print(x)
	c.append(longitudinal.Client(x))
print(x)
for i in range(d):
	for j in range(n):
		print(i, c[j].update(i, eps))

# Test 3: 1-1 client server
print("Single-client server")
d = 16
x = np.random.randint(2, size=d + 1)
c = longitudinal.Client(x)
s = longitudinal.Server(d)
eps = 0.1
print(x)
for i in range(d):
	print(i)
	report = c.update(i, eps)
	print(report)
	s.collect(i, [report])
print(s.T)
s.aggregate(c.k, eps)