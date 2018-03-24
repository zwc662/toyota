#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: lpex1.py
# Version 12.7.1
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2009, 2017. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
#
# lpex1.py - Entering and optimizing a problem.  Demonstrates different
#            methods for creating a problem.
#
# The user has to choose the method on the command line:
#
#    python lpex1.py  -r     generates the problem by adding rows
#    python lpex1.py  -c     generates the problem by adding columns
#    python lpex1.py  -n     generates the problem by adding a list of
#                            coefficients

from __future__ import print_function
import time
import sys
import numpy as np
import cplex
from cplex.exceptions import CplexError

# data common to all populateby functions
'''
my_obj = [1.0, 2.0, 3.0]
my_ub = [40.0, cplex.infinity, cplex.infinity]
my_colnames = ["x1", "x2", "x3"]
'''

'''
my_obj = range(1, 4)
my_ub = [cplex.infinity, cplex.infinity, cplex.infinity]
#my_lb = [-cplex.infinity, -cplex.infinity, -cplex.infinity]
my_lb = [-n, -n, -n]
my_colnames = ["x1", "x2", "x3"]

my_rhs = [20.0, 30.0]
my_rownames = ["c1", "c2"]
my_sense = "LL"
'''
n = 15000

my_obj = np.ones((n)).tolist()

my_rhs = np.random.randint(n, size = (n)).tolist()

my_ub = []
my_lb = []
my_colnames = []
my_rownames = []
my_sense = ""
for i in range(n):
    my_ub.append(cplex.infinity)
    my_lb.append(-100)
    my_rownames.append("c" + str(i + 1))
    my_colnames.append('x' + str(i + 1))
    my_sense = my_sense + str("L")


def populatebyrow(prob):
    prob.objective.set_sense(prob.objective.sense.maximize)

    # since lower bounds are all 0.0 (the default), lb is omitted here
    prob.variables.add(obj=my_obj, lb = my_lb, ub=my_ub, names=my_colnames)

    # can query variables like the following:

    # lbs is a list of all the lower bounds
    lbs = prob.variables.get_lower_bounds()

    # ub1 is just the first lower bound
    ub1 = prob.variables.get_upper_bounds(0)

    # names is ["x1", "x3"]
    names = prob.variables.get_names([0, 2])

    rows = [[[0, "x2", "x3"], [-1.0, 1.0, 1.0]],
            [["x1", 1, 2], [1.0, -3.0, 1.0]]]

    prob.linear_constraints.add(lin_expr=rows, senses=my_sense,
                                rhs=my_rhs, names=my_rownames)

    # because there are two arguments, they are taken to specify a range
    # thus, cols is the entire constraint matrix as a list of column vectors
    cols = prob.variables.get_cols("x1", "x3")


def populatebycolumn(prob):
    prob.objective.set_sense(prob.objective.sense.maximize)

    prob.linear_constraints.add(rhs=my_rhs, senses=my_sense,
                                names=my_rownames)

    c = [[[0, 1], [-1.0, 1.0]],
         [["c1", 1], [1.0, -3.0]],
         [[0, "c2"], [1.0, 1.0]]]

    prob.variables.add(obj=my_obj, ub=my_ub, names=my_colnames,
                       columns=c)


def populatebynonzero(prob):
    prob.objective.set_sense(prob.objective.sense.maximize)

    prob.linear_constraints.add(rhs=my_rhs, senses=my_sense,
                                names=my_rownames)
    prob.variables.add(obj=my_obj, lb = my_lb, ub=my_ub, names=my_colnames)

    '''
    rows = [0, 0, 0, 1, 1, 1]
    cols = [0, 1, 2, 0, 1, 2]
    vals = [-1.0, 1.0, 1.0, 1.0, -3.0, 1.0]
    '''
    '''
    rows = np.array([np.zeros((3)), np.ones((3))])
    rows = np.reshape(rows, (2 * 3)).astype(int).tolist()
    
    cols = np.array([range(3), range(3)])
    cols = np.reshape(cols, (2 * 3)).astype(int).tolist()

    vals = np.random.random((2 * 3)).tolist()
    '''
    
    base = np.reshape(np.arange(n), (n, 1))
    base = np.repeat(base, n, axis = 1)
    rows = np.reshape(base, (n * n)).astype(int).tolist()
    cols = np.reshape(base.T, (n * n)).astype(int).tolist()
    #print(rows)
    #print(cols)

    vals = np.random.random((n * n)).tolist()

    print("Starting LP solving")
    start = time.time()
    prob.linear_constraints.set_coefficients(zip(rows, cols, vals))
    # can also change one coefficient at a time
    # prob.linear_constraints.set_coefficients(1,1,-3.0)
    # or pass in a list of triples
    # prob.linear_constraints.set_coefficients([(0,1,1.0), (1,1,-3.0)])

    end = time.time()
    print("LP finished, time = %f" % (end - start))

def lpex1(pop_method):
    try:
        my_prob = cplex.Cplex()

        if pop_method == "r":
            handle = populatebyrow(my_prob)
        elif pop_method == "c":
            handle = populatebycolumn(my_prob)
        elif pop_method == "n":
            handle = populatebynonzero(my_prob)
        else:
            raise ValueError('pop_method must be one of "r", "c", or "n"')

        my_prob.solve()
    except CplexError as exc:
        print(exc)
        return

    numrows = my_prob.linear_constraints.get_num()
    numcols = my_prob.variables.get_num()

    print()
    # solution.get_status() returns an integer code
    print("Solution status = ", my_prob.solution.get_status(), ":", end=' ')
    # the following line prints the corresponding string
    print(my_prob.solution.status[my_prob.solution.get_status()])
    print("Solution value  = ", my_prob.solution.get_objective_value())
    slack = my_prob.solution.get_linear_slacks()
    pi = my_prob.solution.get_dual_values()
    x = my_prob.solution.get_values()
    dj = my_prob.solution.get_reduced_costs()
    for i in range(numrows):
        print("Row %d:  Slack = %10f  Pi = %10f" % (i, slack[i], pi[i]))
    for j in range(numcols):
        print("Column %d:  Value = %10f Reduced cost = %10f" %
              (j, x[j], dj[j]))

    my_prob.write("lpex1.lp")

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ["-r", "-c", "-n"]:
        print("Usage: lpex1.py -X")
        print("   where X is one of the following options:")
        print("      r          generate problem by row")
        print("      c          generate problem by column")
        print("      n          generate problem by nonzero")
        print(" Exiting...")
        sys.exit(-1)
    start = time.time()
    lpex1(sys.argv[1][1])
    end = time.time()
    print("%s Time: %f" % (sys.argv[1], end - start))
    
