from typing import Union
from BayesNet import BayesNet
import pprint as pp
import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy
import pandas as pd
import itertools

class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    # TODO: This is where your methods should go
    def print_all_cpt(self):
        """ PRETTY PRINT ALL CPTS WITH IT'S ASSOCIATED VAR"""
        all_var = self.bn.get_all_variables()
        all_cpt = self.bn.get_all_cpts()
        for var in all_var:
            print('\n', var)
            pp.pprint(all_cpt[var])

    def draw_interaction_graph(self):
        plt.figure()
        nx.draw(self.bn.get_interaction_graph(),with_labels=True)
        plt.show()

    # def get_factors(self):
    #     factor = {}
    #     all_var = self.bn.get_all_variables()
    #     all_cpt = self.bn.get_all_cpts()
    #     for var in all_var:
    #         factor[var] = deepcopy(all_cpt[var])
    #     return factor

    def prune(self, X:list, Y:list, Z:list):
        """PRUNE FOR DSEP
        input X,Y,Z as lists"""
        pruned_bn = deepcopy(self)
        all_var = self.bn.get_all_variables()
        union = X+Y+Z
        # print('union', union)
        """REMOVING LEAF NODE"""
        for var in union :
            all_var.remove(var)     #save the union variables

        new_all_var = deepcopy(all_var) #changing list while iterating over it TRICK
        for var in all_var:
            children = pruned_bn.bn.get_children(var)
            if len(children) > 0 :
                new_all_var.remove(var)  #save the non leaf not union variables

        all_var = deepcopy(new_all_var) #changing list while iterating over it TRICK
        for var in all_var:
            pruned_bn.bn.del_var(var)   #remove all leaf nodes
        """REMOVING EDGE FROM Z"""
        for var in Z:
            children = pruned_bn.bn.get_children(var)
            for child in children:
                pruned_bn.bn.del_edge([var,child])
        return pruned_bn

    def dsep(self, X:list, Y:list, Z:list):
        """D-SEPARATION
        input X,Y,Z as lists"""
        answer = False
        pruned_bn = self.prune(X,Y,Z)
        all_var = pruned_bn.bn.get_all_variables()

        """GET Y"""
        for var in all_var:
            if var in Y:
                y_var=var
                """CHECK Y IF DISCONNECTED"""
                if len(pruned_bn.bn.get_children(y_var)) == 0 and len(pruned_bn.bn.get_parent(y_var)) == 0:
                    answer = True
                    return pruned_bn, answer

        """GET X"""
        for var in all_var:
            if var in X:
                x_var=var
                """CHECK X IF DISCONNECTED"""
                if len(pruned_bn.bn.get_children(x_var)) == 0 and len(pruned_bn.bn.get_parent(x_var)) == 0:
                    answer = True
                    return pruned_bn, answer

        """CHECK FOR VALVES"""
        for var in all_var:
            if var not in X and var not in Y:
                children = pruned_bn.bn.get_children(var)
                parents = pruned_bn.bn.get_parent(var)
                # print('variable', var)
                # print('children', children)
                # print('parent', parents)
                """SEQUENTIAL"""
                if len(parents) == 1 and len(children) == 1 and var in Z:
                    pruned_bn.bn.del_var(var)
                """DIVERGENT"""
                if len(parents) == 0 and len(children) == 2 and var in Z:
                    pruned_bn.bn.del_var(var)
                """CONVERGENT"""
                if len(parents) == 2 and var not in Z:
                    delete = 1
                    for child in children:
                        if child in Z:
                            delete = 0

                    if delete == 1:
                        pruned_bn.bn.del_var(var)

                """GET Y"""
                for var in all_var:
                    if var in Y:
                        y_var=var
                        """CHECK Y IF DISCONNECTED"""
                        if len(pruned_bn.bn.get_children(y_var)) == 0 and len(pruned_bn.bn.get_parent(y_var)) == 0:
                            answer = True
                            return pruned_bn, answer
                """GET X"""
                for var in all_var:
                    if var in X:
                        x_var=var
                        """CHECK X IF DISCONNECTED"""
                        if len(pruned_bn.bn.get_children(x_var)) == 0 and len(pruned_bn.bn.get_parent(x_var)) == 0:
                            answer = True
                            return pruned_bn, answer

        return pruned_bn, answer


    def jpt_by_chain(self):
        """JOINT PROBABILITY TABLE"""
        """GET ALL VARS AND CPTS"""
        all_var = self.bn.get_all_variables()
        all_cpt = self.bn.get_all_cpts()

        """CREATE INSTANTIATIONS"""
        """GENERATE EVERY PERMUTATION FOR ALL VAR"""
        permutations = [list(i) for i in itertools.product([True, False], repeat=len(all_var))]
        # print(permutations)
        """COMBINE VARS AND TRUTH VALUES"""
        worlds= []
        for permutation in permutations:
            result = zip(all_var,permutation)
            worlds.append(dict(result))

        """CHAIN RULING"""
        for i in range(len(worlds)):
            value = 1
            # print(permutations[i])
            for var in all_var:
                prob = self.bn.get_compatible_instantiations_table(pd.Series(worlds[i]), all_cpt[var])['p'].max()
                print(prob)
                value *= prob
                # print(value)
            permutations[i].append(value)
            print(permutations[i])

        """CREATE JOINT PROBABILITY TABLE"""
        columns = []
        columns.extend(all_var)
        columns.append('p')
        jpt = pd.DataFrame(permutations, columns=columns)
        return jpt


    def create_all_inst(self,vars:list):
        """CREATE EVERY INSTANTIATIONS FOR INPUT VARS
        input variables as list"""
        """GENERATE EVERY PERMUTATION FOR ALL VAR"""
        if type(vars) != list:
            vars = [vars]
        permutations = [list(i) for i in itertools.product([True, False], repeat=len(vars))]
        """COMBINE VARS AND TRUTH VALUES"""
        worlds= []
        for permutation in permutations:
            result = zip(vars,permutation)
            worlds.append(dict(result))
        return worlds


    def create_cpt(self,vars:list):
        """CREATE EMPTY CPT FOR INPUT VARS
        input variables as list"""
        permutations = [list(i) for i in itertools.product([True, False], repeat=len(vars))]
        for x in permutations: x.append(0)
        columns = []
        columns.extend(vars)
        columns.append('p')
        cpt = pd.DataFrame(permutations, columns=columns)
        return cpt


    def sum_out(self,factor,variables):
        """SUMMING OUT
        input factor/cpt, variables to sum out as list"""
        f1 = deepcopy(factor)
        """GET VARIABLES"""
        X = [x for x in f1.columns if x != 'p']
        Z = variables
        Y = [x for x in X if x not in Z]
        """CREATE SUMMED_OUT FACTOR"""
        cpt = self.create_cpt(Y)
        # print(cpt)
        # pp.pprint(cpt)
        """GENERATE ALL INSTANTIATIONS"""
        ys = self.create_all_inst(Y)
        zs = self.create_all_inst(Z)
        """GET VALUES FROM FACTOR AND INSERT THEM IN SUMMED_OUT FACTOR"""
        for y in ys:
            value = 0
            for z in zs:
                series = {}
                series.update(y)
                series.update(z)
                # print(series)
                prob = self.bn.get_compatible_instantiations_table(pd.Series(series), f1)['p'].max()
                # print(prob)
                if not (prob >= 0):
                    prob = 0
                value += prob
            indx = self.bn.get_compatible_instantiations_table(pd.Series(series), cpt).index[0]
            cpt.loc[indx,'p'] = value
        return cpt

    def multiply_factors(self,factors:list):
        """MULTIPLYING FACTORS
        input factors/cpts as list"""
        f = deepcopy(factors)
        """GET VARIABLES"""
        Z = []
        for f1 in f:
            X = [x for x in f1.columns if x != 'p']
            for var in X:
                if var not in Z: Z.append(var)
        # print(Y,X)
        """CREATE MULTIPLIED FACTOR"""
        cpt = self.create_cpt(Z)
        # print(cpt)
        # pp.pprint(cpt)
        """GENERATE ALL INSTANTIATIONS"""
        zs = self.create_all_inst(Z)
        """GET VALUES FROM FACTOR AND INSERT THEM IN MULTIPLIED FACTOR"""
        for z in zs:
            series = {}
            series.update(z)
            # print(series)
            value = 1
            for i in range(len(f)):
                prob = self.bn.get_compatible_instantiations_table(pd.Series(series), f[i])['p'].max()
                # print(prob)
                value *= prob
            indx = self.bn.get_compatible_instantiations_table(pd.Series(series), cpt).index[0]
            cpt.loc[indx,'p'] = value
        return cpt

    def rand_ordering(self,vars:list):
        """RANDOM ORDER
        input variables as list"""
        all_var = self.bn.get_all_variables()
        ordering = [var for var in all_var if var not in vars]
        return ordering

    def prior_margin(self,vars,ordering):
        """PRIOR MARGINAL
        input vars as list
        rest of the variables in an order in a list"""
        """GET ALL CPTS"""
        all_var = self.bn.get_all_variables()
        all_cpt = self.bn.get_all_cpts()
        l_cpts = []
        for var in all_var:
            l_cpts.append(all_cpt[var])
        """FORMULA IN LECTURES"""
        for variable in ordering:
            to_multi = []
            for cpt in l_cpts:
                if variable in cpt.columns:
                    to_multi.append(cpt)
                    l_cpts = [k for k in l_cpts if not k.equals(cpt)] # remove cpt from list
            mfactor = self.multiply_factors(to_multi)
            sfactor = self.sum_out(mfactor,variable)
            l_cpts.append(sfactor)
        prior = self.multiply_factors(l_cpts)
        return prior

    def post_margin(self,vars,inst,ordering):
        """PRIOR MARGINAL
        input vars as list
        inst as pandas series
        rest of the variables in an order in a list"""
        """GET ALL CPTS"""
        all_var = self.bn.get_all_variables()
        all_cpt = self.bn.get_all_cpts()
        l1_cpts = []
        for var in all_var:
            l1_cpts.append(all_cpt[var])
        """VAR ELIMINATION"""
        l_cpts = []
        for cpt in l1_cpts:
            l_cpts.append(self.bn.get_compatible_instantiations_table(inst, cpt))
        """FORMULA IN LECTURES"""
        for variable in ordering:
            to_multi = []
            for cpt in l_cpts:
                if variable in cpt.columns:
                    to_multi.append(cpt)
                    l_cpts = [k for k in l_cpts if not k.equals(cpt)] # remove cpt from list
            mfactor = self.multiply_factors(to_multi)
            sfactor = self.sum_out(mfactor,variable)
            l_cpts.append(sfactor)
        post = self.multiply_factors(l_cpts)
        """PR Norm"""
        varnames = [i for i in inst.index]
        order = self.rand_ordering(varnames)
        prnorm = self.prior_margin(varnames, order)
        prnorm = self.bn.get_compatible_instantiations_table(inst, prnorm)
        i = 0
        for pr in post['p']:
            post['p'][i] = pr/prnorm['p']
            i += 1

        return post

    def edge_prune(self,instantiation):
        pruned_bn = deepcopy(self)
        all_cpt = self.bn.get_all_cpts()
        var_names = instantiation.index.values
        for var in var_names:
            children = pruned_bn.bn.get_children(var)
            for child in children:
                new_inst = deepcopy(instantiation) # if child is instantiated in cpt
                if child in instantiation.index:
                    new_inst = instantiation.drop(labels=child) # if child is instantiated in cpt
                    new_cpt = pruned_bn.bn.get_compatible_instantiations_table(new_inst, all_cpt[child])
                else:
                    new_cpt = pruned_bn.bn.get_compatible_instantiations_table(instantiation, all_cpt[child])
                new_cpt = pruned_bn.sum_out(new_cpt,var)
                pruned_bn.bn.update_cpt(child,new_cpt)
                pruned_bn.bn.del_edge([var,child])
        return pruned_bn

    def node_prune(self,variables,instantiation):
        pruned_bn = deepcopy(self)
        all_var = pruned_bn.bn.get_all_variables()
        var_names = instantiation.index.values
        l_var = []
        for var in var_names:
            l_var.append(var)
        for var in variables:
            l_var.append(var)
        # print(l_var)
        for var in l_var:
            all_var.remove(var)
        # print(all_var)
        new_all_var = deepcopy(all_var)
        removed = 1
        while removed == 1:
            removed = 0
            for var in all_var:
                children = pruned_bn.bn.get_children(var)
                if len(children) == 0 :
                    pruned_bn.bn.del_var(var)
                    new_all_var.remove(var)
                    removed = 1
            all_var = deepcopy(new_all_var)
        return pruned_bn

    def network_prune(self,variables,instantiation):
        pruned_bn = deepcopy(self)
        pruned_bn = pruned_bn.node_prune(variables,instantiation)
        pruned_bn = pruned_bn.edge_prune(instantiation)
        return pruned_bn

    def max_out(self,factor,variables):
        """MAXING OUT
        input factor/cpt, variables to max out as list"""
        f1 = deepcopy(factor)
        """GET VARIABLES"""
        X = [x for x in f1.columns if x != 'p']
        Z = variables
        Y = [x for x in X if x not in Z]
        """CREATE MAXED_OUT FACTOR"""
        cpt = self.create_cpt(Y)
        """GENERATE ALL INSTANTIATIONS"""
        ys = self.create_all_inst(Y)
        zs = self.create_all_inst(Z)
        """GET VALUES FROM FACTOR AND INSERT THEM IN SUMMED_OUT FACTOR"""
        rows = []
        columns = []
        for y in ys:
            value = -1
            to_append = []
            # print("y")
            for z in zs:
                # print("z")
                series = {}
                series.update(y)
                series.update(z)
                values = [z[name] for name in z]
                # print(series)
                prob = self.bn.get_compatible_instantiations_table(pd.Series(series), f1)['p'].max()
                # print(prob)
                if value == -1:
                    to_append = values
                    indx = self.bn.get_compatible_instantiations_table(pd.Series(series), cpt).index[0]
                    cpt.loc[indx,'p'] = prob
                    value = prob
                if value >= 0:
                    if prob > value:
                        to_append = values
                        indx = self.bn.get_compatible_instantiations_table(pd.Series(series), cpt).index[0]
                        cpt.loc[indx,'p'] = prob
                        value = prob
            rows.append(list(to_append))
        columns.extend(Z)
        xtension = pd.DataFrame(rows, columns=columns)
        cpt = pd.concat([cpt, xtension], axis=1)
        return cpt

    def MPE(self,instantiation):
        pruned_bn = self.edge_prune(instantiation)
        all_var = pruned_bn.bn.get_all_variables()
        all_cpt = pruned_bn.bn.get_all_cpts()
        ordering = self.rand_ordering(all_var)
        l1_cpts = []
        for var in all_var:
            l1_cpts.append(all_cpt[var])
        """NORMALISING BY INST"""
        l_cpts = []
        for cpt in l1_cpts:
            l_cpts.append(self.bn.get_compatible_instantiations_table(instantiation, cpt))
        """FORMULA IN LECTURES"""
        for variable in ordering:
            to_multi = []
            for cpt in l_cpts:
                if variable in cpt.columns:
                    to_multi.append(cpt)
                    l_cpts = [k for k in l_cpts if not k.equals(cpt)] # remove cpt from list
            mfactor = self.multiply_factors(to_multi)
            maxfactor = self.max_out(mfactor,variable)
            l_cpts.append(maxfactor)
        """GET THE MAX"""
        mpe = self.multiply_factors(l_cpts)
        mpe = mpe.sort_values(by=['p'], ascending=False)
        mpe = mpe.head(1)
        return mpe

    def MAP(self,mapvar,instantiation):
        pruned_bn = self.network_prune(mapvar,instantiation)
        all_var = pruned_bn.bn.get_all_variables()
        all_cpt = pruned_bn.bn.get_all_cpts()
        ordering = []
        for var in all_var:
            if not var in mapvar:
                ordering.append(var)
        ordering.extend(mapvar)
        # ordering = self.rand_ordering(all_var)
        l1_cpts = []
        for var in all_var:
            l1_cpts.append(all_cpt[var])
        """NORMALISING BY INST"""
        l_cpts = []
        for cpt in l1_cpts:
            l_cpts.append(self.bn.get_compatible_instantiations_table(instantiation, cpt))
        """FORMULA IN LECTURES"""
        for variable in ordering:
            to_multi = []
            for cpt in l_cpts:
                if variable in cpt.columns:
                    to_multi.append(cpt)
                    l_cpts = [k for k in l_cpts if not k.equals(cpt)] # remove cpt from list
            mfactor = self.multiply_factors(to_multi)
            maxfactor = self.max_out(mfactor,variable)
            l_cpts.append(maxfactor)
        """GET THE MAX"""
        mpe = self.multiply_factors(l_cpts)
        mpe = mpe.sort_values(by=['p'], ascending=False)
        mpe = mpe.head(1)
        return mpe
