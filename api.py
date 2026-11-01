"""Start the local analytics application: python api.py."""
import runpy

if __name__ == '__main__':
    runpy.run_module('backend.server', run_name='__main__')
