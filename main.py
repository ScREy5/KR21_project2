from BNReasoner import *
from BayesNet import *


# PGM = BNReasoner('testing/lecture_example.BIFXML')
# PGM = BNReasoner('testing/new_lecture.BIFXML')
PGM = BNReasoner('testing/lecture_example2.BIFXML')
# PGM.bn.draw_structure()
all_var = PGM.bn.get_all_variables()
print(all_var)
all_cpt = PGM.bn.get_all_cpts()
# PGM.print_all_cpt()
"""LEAVING HERE FOR COMP INST SYNTAX"""
 # pp.pprint(PGM.bn.get_compatible_instantiations_table(pd.Series({"Sprinkler?":True,"Rain?":True,"Winter?":True,"Wet Grass?":True}), all_cpt["Wet Grass?"]))

"""TEST FOR DSEP"""
# pruned , answer = PGM.dsep(['family-out'],['bowel-problem'],[])
# pruned.bn.draw_structure()
# print(answer)
# print(nx.d_separated(PGM.bn.structure,{'family-out'},{'bowel-problem'},{}))

"""TEST FOR JPT AND SUM_OUT"""
# jpt = PGM.jpt_by_chain()
# pp.pprint(jpt)
# # ["Winter?","Sprinkler?","Rain?"]
# cpt1 = PGM.sum_out(jpt,["Winter?"])
# cpt2 = PGM.sum_out(jpt,["Winter?","Sprinkler?","Rain?"])
# pp.pprint(cpt2)

"""TEST FOR MULTIPLYING"""
# l_cpts = []
# for var in all_var:
#     l_cpts.append(all_cpt[var])
#
# cpt3 = PGM.multiply_factors(l_cpts)
# pp.pprint(cpt3)

# l_cpts = []
# for var in all_var:
#     l_cpts.append(all_cpt[var])
#
"""TEST FOR PRIOR"""
# E = ["Slippery Road?"]
# DE = ["Wet Grass?","Slippery Road?"]
# order = PGM.rand_ordering(DE)
# prDE = PGM.prior_margin(DE, order)
# print(prDE)

"""TEST FOR POST"""
# DE = ["Wet Grass?","Slippery Road?"]
# # pd.Series({"Sprinkler?":True,"Rain?":True,"Winter?":True,"Wet Grass?":True})
# inst = pd.Series({"Sprinkler?":False,"Winter?":True})
# order = PGM.rand_ordering(DE)
# postDE = PGM.post_margin(DE,inst,order)
# print(postDE)
"""TEST FOR PRUNING"""
# print(PGM.bn.get_children("Slippery Road?"))
# inst = pd.Series({"Winter?":True,"Rain?":False})
# var = ["Wet Grass?"]
# pruned = PGM.network_prune(var,inst)
# pruned.print_all_cpt()
# pruned.bn.draw_structure()
"""TEST FOR MAXING"""
# jpt = PGM.jpt_by_chain()
# print(jpt)
# cpt1 = PGM.max_out(jpt,["Winter?","Sprinkler?","Rain?","Wet Grass?","Slippery Road?"])
# print(cpt1)
"""TEST FOR MPE"""
# mpe = PGM.MPE(pd.Series({"A":True}))
# print("MPE\n",mpe)
"""TEST FOR MAP Lecture ex1"""
# map = PGM.MAP(["S","C"],pd.Series({"A":True}))
# print("MAP\n",map)
"""TEST FOR MAP Lecture ex2"""
map = PGM.MAP(["I","J"],pd.Series({"O":True}))
print("MAP\n",map)
"""TEST FOR POST Lecture ex2"""
IJ = ["I","J"]
# pd.Series({"Sprinkler?":True,"Rain?":True,"Winter?":True,"Wet Grass?":True})
inst = pd.Series({"O":True})
order = PGM.rand_ordering(IJ)
postIJ = PGM.post_margin(IJ,inst,order)
print(postIJ)
