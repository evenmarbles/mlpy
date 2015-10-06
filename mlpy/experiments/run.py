from sys import platform
from subprocess import Popen
from multiprocessing import Process

from rlglued.agent.loader import load_agent
from rlglued.environment.loader import load_environment


def which(program):
    import os

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def run(env, agent, exp, env_args=None, env_kwargs=None, agent_args=None, agent_kwargs=None,
        exp_args=None, exp_kwargs=None, local=None):
    if local is None:
        local = False

    env_args = env_args if env_args is not None else ()
    env_kwargs = env_kwargs if env_kwargs is not None else {}

    agent_args = agent_args if agent_args is not None else ()
    agent_kwargs = agent_kwargs if agent_kwargs is not None else {}

    exp_args = exp_args if exp_args is not None else ()
    exp_kwargs = exp_kwargs if exp_kwargs is not None else {}

    if local:
        exp = exp(env=env(*env_args, **env_kwargs), agent=agent(*agent_args, **agent_kwargs), *exp_args, **exp_kwargs)
        exp.run()
    else:
        if platform == 'win32':
            exe = which('rlglue_core.exe')
        else:
            exe = which('rl_glue')

        p_rlglue = Popen(exe)

        p_agent = Process(target=load_agent, args=(agent(*agent_args, **agent_kwargs),))
        p_agent.start()

        p_env = Process(target=load_environment, args=(env(*env_args, **env_kwargs),))
        p_env.start()

        exp = exp(*exp_args, **exp_kwargs)
        exp.run()

        p_env.terminate()
        p_agent.terminate()
        p_rlglue.terminate()
