import core as core
import numpy as np
import sys

if __name__ == '__main__':
	assert len(sys.argv)>1, "Need config name"
	configName = sys.argv[1]
	filename = 'config/'+configName+'.ini'
	c = core.core(filename)
	c.train()
	#c.LRdecrease()
	#c.train()
	#c.LRdecrease()
	#c.train()
