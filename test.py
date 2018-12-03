"""Simple tests for the components; use the notebook for the actual cases
"""

import longitudinal
import numpy as np

# Test 1: Single-client output
d = 16
x = np.random.randint(2, size=d + 1)
c = longitudinal.Client(x)
print(x)
for i in range(d):
	print(i, c.update(i, 0.1))
