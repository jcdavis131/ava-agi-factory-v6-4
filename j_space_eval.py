"""j_space_eval.py - 5 property tests"""
from eval_branch_harness import run_test
for t in ["spider_ant","france_china","soccer_rugby","spanish_french","safety_blackmail"]:
    print(t, run_test(t,"base","mock"))
