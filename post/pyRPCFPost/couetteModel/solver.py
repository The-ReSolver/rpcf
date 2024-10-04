import time
import fcntl
import os
import time
import sys
import subprocess

import numpy as np

# these are the names of the columns output from the simulation code
keys = ["t", "K", "dKdt", "dlog10(k)dt", "CFL"]

def formatCommandLine(params):
	""" Format command line string from parameter dictionary. """
	return "%s %d %d %d %d %f %f %f %f %s %d %s %d %d" % (params["execpath"],
							                    	   params["Ny"],
							                    	   params["Nz"],
							                    	   params["My"],
							                    	   params["Mz"],
							                    	   params["Re"],
							                    	   params["Omega"],
							                    	   params["dt"],
							                    	   params["T"],
							                    	   params["savefile"],
							                    	   params["n_it_out"],
							                    	   params["initfile"],
							                    	   params["full"],
							                    	   params["mode"])



#class Solver(object):

	#def __init__(self, params):

		#we may want to force a stop of the program
		#self._SIGKILLsent = False

		#format command line
		#self._command = "%s %d %d %d %d %f %f %f %f %s %d %e" % (params["execpath"],
																 #params["Ny"],
																 #params["Nz"],
																 #params["My"],
																 #params["Mz"],
																 #params["Re"],
																 #params["Omega"],
																 #params["dt"],
																 #params["T"],
																 #params["savefile"],
																 #params["n_it_out"],
																 #params["K0"])

	#def __iter__(self):
		#"""Start command and go into poll mode"""
		
		#use a string as a queue
		#self.outputQueue = ""

		#spawn process
		#self._proc = subprocess.Popen(self._command, stdout=subprocess.PIPE, shell=True)
		#return self

	#def next(self):
		#""" Return a line from the simulation code, formatted as a dictionary. """

		#while True:
			#maybe process has terminated, and we must consume the queue
			#stdoutdata = self.non_block_read(self._proc.stdout)
			#self.outputQueue += stdoutdata

			#if we have a new line character split and keep the rest
			#if "\n" in self.outputQueue:
				#out, sep, self.outputQueue = self.outputQueue.partition("\n")
				#break

			#if process has terminated
			#if self._proc.poll() == 0:
				#raise StopIteration

			#do not hog cpu waiting for data
			#time.sleep(0.0005)

		#if stringQueue is empty stop
		#if len(self.outputQueue) == 0 and len(out) == 0:
		  	#raise StopIteration

		#if we have sent a SIGKILL stop
		#if self._SIGKILLsent:
		 	#raise StopIteration

		#return dict(zip(keys, map(float, out.split())))

	#def stop(self):
		#""" Stop process. """
		#self._proc.terminate()
		#self._SIGKILLsent = True

	#@staticmethod
	#def non_block_read(pipe):
		#""" Read data from PIPE and do not block """
		#fd = pipe.fileno()
		#fl = fcntl.fcntl(fd, fcntl.F_GETFL)
		#fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
		#try:
			#return pipe.read()
		#except:
			#return ""


#if __name__ == '__main__':
	
	#sim	# set up some parameters
	#params = {  "execpath"		 : "/home/davide/svn-repo/fluidSOS/model-problem-solver/solver",
				#"Ny"   			 : 64,
	 			#"Nz"   			 : 196,
	 			#"My"   			 : 64,
	 			#"Mz"   			 : 64,
	 			#"Re"   			 : 1,
	 			#"Omega"   		 : 0.5,
	 			#"dt"   			 : 1e-3,
	 			#"T"   			 : 0.01,
	 			#"datadir"   	 : "/home/davide/Activities/report-on-physics/runs/bogus",
	 			#"n_it_out"     	 : 100,
	 			#"K0"    		 : 1}

	#from pylab import *

	#Nys = np.arange(400, 500, 2)	
	#Ts = np.zeros(len(Nys), dtype=np.float64)

	#for i in range(3):
		#for i, Ny in enumerate(Nys):
		
			#run simulation with different resolution
			#params['Ny'] = Ny

			#prepare simulation
			#sim = Simulation(params)

			#start clock
			#t0 = time.time()
			#subprocess.call(sim._command + " > /dev/null", shell=True)
			#Ts[i] = time.time() - t0

			#print  Ny, time.time() - t0 


		#plot(Nys, Ts, '.')
	#show()			


