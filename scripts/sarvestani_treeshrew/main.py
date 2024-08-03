import argparse
import os
from stim_behavior.sarvestani_treeshrew.suite2p_manager import Suite2pManager

def print_opts(opts):
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)

def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument( '--log_level', type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="DEBUG",
      help='Sets the level of logging. "DEBUG" is the most verbose' )
    parser.add_argument( '--run_all', type=bool, default=False,
      help='Set this to True if all sessions should be ran. Else only one session "44/t1" will be run' )

    return parser

def main(args):
    s2p_manager = Suite2pManager(log_level=args.log_level)
    sessions = [
        ["44", "t1"],
        ["44", "t2"],
        ["44", "t3"],
        ["44", "t4"],
        ["45", "t1"],
        ["45", "t2"],
        ["45", "t3"],
        ["46", "t1"],
        ["46", "t2"],
    ]
    breakpoint()
    if not args.run_all:
        sessions = sessions[:1]
    
    for session_id, subsession_id in sessions:
        s2p_manager.load_neural_session(session_id=session_id, subsession_id=subsession_id)
        s2p_manager.run()
        s2p_manager.compute_dF()


###############################################################################

if __name__ == "__main__":
    
    parser = create_parser()
    args = parser.parse_args()
    print_opts(args)
    
    main(args)