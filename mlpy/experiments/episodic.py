from __future__ import division, print_function, absolute_import
# noinspection PyUnresolvedReferences
from six.moves import range

import rlglued.rlglue as rlglue

from ..auxiliary.misc import columnize
from ..tools.misc import Timer


class EpisodicExperiment(object):
    def __init__(self, ntrials=None, nepisodes=None, nsteps=None, timed=None, env=None, agent=None, filename=None):
        self._ntrials = ntrials if ntrials is not None else 1
        self._nepisodes = nepisodes if nepisodes is not None else 10
        self._nsteps = nsteps if nsteps is not None else 5000
        self._timed = timed if timed is not None else True

        self._f = None
        if filename is not None:
            self._f = open(filename, 'wb')

        self._local = False

        if env is not None and agent is not None:
            self.rlglue = rlglue.RLGlueLocal(env, agent)
            self._local = True
        else:
            self.rlglue = rlglue.RLGlue()

    def run_episode(self):
        runtime = None

        if self._timed:
            with Timer() as t:
                terminal, converged = self.rlglue.run_episode(self._nsteps)
            runtime = t.time
        else:
            terminal, converged = self.rlglue.run_episode(self._nsteps)
        total_steps = self.rlglue.num_steps()
        total_reward = self.rlglue.reward_return()

        ret = (total_steps, total_reward, terminal, converged)
        if runtime is not None:
            ret += (runtime,)
        return ret

    def run_trial(self, iter_):
        self.rlglue.init()
        if not self._local:
            print("\t Experiment Codec Connected")

        header = ('Episode', '#Steps', 'Reward', 'Terminal')
        if self._timed:
            header += ('Runtime',)
        width = 0

        if self._f:
            h = ','.join(header) + "\n"
            if not iter_ == 0:
                h = "\n\n" + h
            self._f.write(h)

        for i in range(self._nepisodes):
            ret = list(self.run_episode())
            converged = ret.pop(3)
            out, width = columnize((i,) + tuple(ret), "C", width, (header if i == 0 else None), '|')
            print(out)

            if self._f:
                self._f.write(str(i+1) + ',' + ','.join(map(lambda x: str(x), ret)) + "\n")
                self._f.flush()

            if converged:
                break
        print("\n")
        self.rlglue.cleanup()

    def run(self):
        for i in range(self._ntrials):
            self.run_trial(i)
