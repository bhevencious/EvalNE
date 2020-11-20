from evalne.evaluation.evaluator import LPEvaluator
from evalne.evaluation.split import EvalSplit
from evalne.evaluation.score import Scoresheet
from evalne.utils import preprocess as pp

# Load and preprocess the network
#G = pp.load_graph('evalne/tests/data/network.edgelist')
G = pp.load_graph('../Graph_Conv_Neural_Nets/generic_datasets/Zachary-Karate/Zachary-Karate.edgelist')
G, _ = pp.prep_graph(G)

# Create an evaluator and generate train/test edge split
traintest_split = EvalSplit()  # Bhevencious: EvalSplit() contains methods used to READ/SET a variety of properties/variables. Use the DOT & PARANTHESIS helpers to access parameters.
traintest_split.compute_splits(G, nw_name='Zachary-Karate.edgelist', train_frac=0.8)
nee = LPEvaluator(traintest_split)

# Create a Scoresheet to store the results
scoresheet = Scoresheet()

# Set the baselines
methods = ['adamic_adar_index', 'common_neighbours', 'jaccard_coefficient', 'katz', 'preferential_attachment', 'resource_allocation_index', 'random_prediction']

# Evaluate baselines
for method in methods:
    result = nee.evaluate_baseline(method=method)
    scoresheet.log_results(result)

try:
    # Check if OpenNE is installed
    import openne

    # Set embedding methods from OpenNE
    methods = ['node2vec', 'deepwalk', 'GraRep']
    commands = [
        'python -m openne --method node2vec --graph-format edgelist --p 1 --q 1',
        'python -m openne --method deepWalk --graph-format edgelist --number-walks 40',
        'python -m openne --method grarep --graph-format edgelist --epochs 10']
    edge_emb = ['average', 'hadamard']

    # Evaluate embedding methods
    for i in range(len(methods)):
        command = commands[i] + " --input {} --output {} --representation-size {}"
        results = nee.evaluate_cmd(method_name=methods[i], method_type='ne', command=command, edge_embedding_methods=edge_emb, input_delim=' ', output_delim=' ')
        scoresheet.log_results(results)

except ImportError:
    print("The OpenNE library is not installed. Reporting results only for the baselines...")
    pass

# Get output
scoresheet.print_tabular(metric='auroc')
scoresheet.write_all(filename='eval_log.txt', repeats='avg')  # Bhevencious: score.py contains a range of methods & parameters for outputting results