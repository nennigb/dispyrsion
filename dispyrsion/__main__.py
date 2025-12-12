
# This file is part of dispyrsion, A library to locate exceptional points
# from analytical dispersion equations.
# Copyright (C) <2025> <benoit.nennig@isae-supmeca.fr>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
# Test suite runner

Run `dispyrsion` run the test suite using `doctest` and `unittest` framework.

Example
-------
```console
python3 -m dispyrsion
```
"""
import doctest
import os
import sys
import unittest

# Add examples path for import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../examples')))
# import module with doctest
import coupled_mass_spring_damper_system
import duct_with_wiremesh
import toy_3dof_2params
import waveguide_jsv

from dispyrsion import disp

mod_list = [disp, toy_3dof_2params, coupled_mass_spring_damper_system, waveguide_jsv,
            duct_with_wiremesh]

if __name__ == '__main__':
    print('> Running tests...')
    Stats = []
    tests_dir = path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tests"))
    # Create test suite for unittest
    suite = unittest.TestLoader().discover(start_dir=tests_dir, pattern='test*.py')
    # Add doctest from all modules of mod_list
    for mod in mod_list:
        suite.addTest(doctest.DocTestSuite(mod,
                                           optionflags=(doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)))
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)

    print("\n", "================ Testing summary ===================")
    if result.wasSuccessful():
        print("                                             Pass :-)")
        sys.exit(0)
    else:
        print("                                           Failed :-(")
        sys.exit(1)
