from smac.configspace import ConfigurationSpace, Configuration
import signal

class RandomOptimizer(object):
    def __init__(self, random_seed, config_space, blackbox_function):
        self._seed = random_seed
        self._cs = config_space
        self._f = blackbox_function
        self._runhistory = []
        self._trajectory = []
            
    @property
    def runhistory(self):
        return self._runhistory
    
    @property
    def trajectory(self):
        return self._trajectory
    
    def optimize(self, n_evaluations=20, cutoff=10):
        # clear memory
        self._runhistory = []
        self._trajectory = []
        
        # set random seed 
        self._cs.seed(self._seed)
        
        # get a list of random configurations
        cfg_ls = self._cs.sample_configuration(n_evaluations)
        
        # incumbent
        incumbent = (None, float('inf'))
        
        # setup signal handler
        def handler(signum, frame):
            raise Exception("Evaluation timed out.")
        
        # main loop
        for i, cfg in enumerate(cfg_ls):
            try:
                # setup cutoff time
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(cutoff)
                
                # evaluate configuration
                score = self._f(cfg)
                self._runhistory.append((cfg, score))

                if score < incumbent[1]:
                    incumbent = (cfg, score)
                    self._trajectory.append((cfg, score))   
                    
                # cancel the timer if we finished on time
                signal.alarm(0)
                
            except Exception as e: 
                print(e)
        
        # return optimal configuration
        return incumbent
            
    