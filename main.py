import projection
import MLP_Handler
import HN_Handler
import Test_and_plot
import shutil

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--data_index", type=int, default=0)
parser.add_argument("--Mem_Optimize", type=int, default=1)

args = parser.parse_args()
data_index = args.data_index
Mem_Optimize = args.Mem_Optimize

projection.OU_Projection_Problem(data_index)
MLP_Handler.Run(data_index)
HN_Handler.Run(data_index)
Test_and_plot.Run(data_index)


if Mem_Optimize:
    shutil.rmtree("problem_instance_" + str(data_index) + ".pt")
    shutil.rmtree("MLP_Log_problem_" + str(data_index))
    shutil.rmtree("HN_Log_problem_" + str(data_index))
    shutil.rmtree("Testing_Results_problem_" + str(data_index))
