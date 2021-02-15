import os
import json
import pickle
from tqdm import tqdm
tqdm.pandas()
import networkx as nx
from joblib import Parallel, delayed
from sklearn.utils.fixes import _joblib_parallel_args
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from chemml.regression.gpr_learner import GPRLearner
from chemml.classification.learner import ClassificationLearner
from chemml.classification.gpc.gpc import GPC
from chemml.classification.svm.svm import SVC
from chemml.regression.consensus import ConsensusRegressor
from chemml.regression.GPRgraphdot.gpr import LRAGPR
from chemml.graph.hashgraph import HashGraph
from chemml.graph.from_rdkit import rdkit_config
from chemml.graph.reaction import *
from chemml.kernels.ConvKernel import *


def set_graph_property(input_config):
    single_graph, multi_graph, r_graph, properties = input_config.split(':')
    single_graph = single_graph.split(',') if single_graph else []
    multi_graph = multi_graph.split(',') if multi_graph else []
    reaction_graph = r_graph.split(',') if r_graph else []
    properties = properties.split(',')
    return single_graph, multi_graph, reaction_graph, properties


def set_block_config(block_config):
    block_length = int(block_config.split(':')[0])
    block_x_id = int(block_config.split(':')[1].split(',')[0])
    block_y_id = int(block_config.split(':')[1].split(',')[1])
    return block_length, block_x_id, block_y_id


def set_gpr_optimizer(gpr):
    gpr, optimizer = gpr.split(':')
    assert (gpr in ['graphdot', 'sklearn', 'graphdot_nystrom'])
    if optimizer in ['None', 'none', '']:
        return gpr, None
    return gpr, optimizer


def set_gpc_optimizer(gpc):
    gpc, optimizer = gpc.split(':')
    assert (gpc == 'sklearn')
    if optimizer in ['None', 'none', '']:
        return gpc, None
    return gpc, optimizer


def set_kernel_alpha(kernel):
    kernel, alpha = kernel.split(':')
    if kernel not in ['graph', 'preCalc']:
        raise Exception('Unknown kernel')
    return kernel, float(alpha)


def set_add_feature_hyperparameters(add_features):
    if add_features is None:
        return None, None
    add_f, add_p = add_features.split(':')
    add_f = add_f.split(',')
    add_p = list(map(float, add_p.split(',')))
    assert (len(add_f) == len(add_p))
    return add_f, add_p


def set_mode_train_size_ratio_seed(train_test_config):
    result = train_test_config.split(':')
    if len(result) == 4:
        mode, train_size, train_ratio, seed = result
        dynamic_train_size = 0
    else:
        mode, train_size, train_ratio, seed, dynamic_train_size = result
    train_size = int(train_size) if train_size else None
    train_ratio = float(train_ratio) if train_ratio else None
    seed = int(seed) if seed else 0
    dynamic_train_size = int(dynamic_train_size) if dynamic_train_size else 0
    return mode, train_size, train_ratio, seed, dynamic_train_size


def set_gpr_model(gpr, kernel_config, optimizer, alpha):
    if gpr == 'graphdot':
        from chemml.regression.GPRgraphdot.gpr import GPR
        model = GPR(kernel=kernel_config.kernel,
                    optimizer=optimizer,
                    alpha=alpha,
                    normalize_y=True)
    elif gpr == 'sklearn':
        from chemml.regression.GPRsklearn.gpr import GPR
        model = GPR(kernel=kernel_config.kernel,
                    optimizer=optimizer,
                    alpha=alpha,
                    y_scale=True)
    elif gpr == 'graphdot_nystrom':
        from chemml.regression.GPRgraphdot.gpr import LRAGPR
        model = LRAGPR(
            kernel=kernel_config.kernel,
            optimizer=optimizer,
            alpha=alpha,
            normalize_y=True)
    else:
        raise RuntimeError(f'Unknown GaussianProcessRegressor: {gpr}')
    return model


def set_gpc_model(gpc, kernel_config, optimizer, n_jobs):
    if gpc == 'sklearn':
        model = GPC(
            kernel=kernel_config.kernel,
            optimizer=optimizer,
            n_jobs=n_jobs
        )
    else:
        raise RuntimeError(f'Unknown GaussianProcessClassifier: {gpc}')
    return model


def set_svc_model(svc, kernel_config, C):
    if svc == 'sklearn':
        model = SVC(
            kernel=kernel_config.kernel,
            C=C
        )
    else:
        raise RuntimeError(f'Unknown SVMClassifier: {svc}')
    return model


def set_consensus_config(consensus_config):
    if consensus_config is None:
        return False, 0, 0, 0, 0
    else:
        n_estimators, n_sample_per_model, n_jobs, consensus_rule = \
            consensus_config.split(':')
        return True, int(n_estimators), int(n_sample_per_model), int(n_jobs), \
               consensus_rule


def set_active_config(active_config):
    learning_mode, add_mode, init_size, add_size, max_size, search_size, \
    pool_size, stride = active_config.split(':')
    init_size = int(init_size) if init_size else 0
    add_size = int(add_size) if add_size else 0
    max_size = int(max_size) if max_size else 0
    search_size = int(search_size) if search_size else 0
    pool_size = int(pool_size) if pool_size else 0
    stride = int(stride) if stride else 0
    return learning_mode, add_mode, init_size, add_size, max_size, \
           search_size, pool_size, stride


def set_kernel_config(kernel, add_features, add_hyperparameters,
                      single_graph, multi_graph, hyperjson,
                      result_dir):
    if kernel == 'graph':
        hyperdict = [
            json.loads(open(f, 'r').readline()) for f in hyperjson.split(',')
        ]
        params = {
            'single_graph': single_graph,
            'multi_graph': multi_graph,
            'hyperdict': hyperdict
        }
        from chemml.kernels.GraphKernel import GraphKernelConfig as KConfig
    else:
        params = {
            'result_dir': result_dir,
        }
        from chemml.kernels.PreCalcKernel import PreCalcKernelConfig as KConfig
    return KConfig(add_features, add_hyperparameters, params)


def read_input(result_dir, input, kernel_config, properties, params):
    def df_filter(df, train_size=None, train_ratio=None, bygroup=False,
                  byclass=False, seed=0):
        np.random.seed(seed)
        if 'IsTrain' in df:
            return df[df.IsTrain == True], df[df.IsTrain == False]
        elif byclass:
            assert (train_ratio is not None)
            df_train = []
            for group in df.groupby(properties):
                df_train.append(group[1].sample(frac=train_ratio))
            df_train = pd.concat(df_train)
            df_test = df[~df.index.isin(df_train.index)]
        else:
            if bygroup:
                gname = 'group_id'
            else:
                gname = 'id'
            unique_ids = df[gname].unique()
            if train_size is None:
                train_size = int(unique_ids.size * train_ratio)
            ids = np.random.choice(unique_ids, train_size, replace=False)
            df_train = df[df[gname].isin(ids)]
            df_test = df[~df[gname].isin(ids)]
        return df_train, df_test

    if params is None:
        params = {
            'train_size': None,
            'train_ratio': 1.0,
            'seed': 0,
            'byclass': False
        }
    print('***\tStart: Reading input.\t***')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    # read input.
    single_graph = kernel_config.single_graph \
        if hasattr(kernel_config, 'single_graph') else []
    multi_graph = kernel_config.multi_graph \
        if hasattr(kernel_config, 'multi_graph') else []
    df = get_df(input,
                os.path.join(result_dir, '%s.pkl' % ','.join(properties)),
                single_graph, multi_graph, [])
    # get df of train and test sets
    df_train, df_test = df_filter(
        df,
        train_size=params['train_size'],
        train_ratio=params['train_ratio'],
        seed=params['seed'],
        bygroup=kernel_config.add_features is not None,
        byclass=params['byclass']
    )
    # get X, Y of train and test sets
    train_X, train_Y, train_id = get_XYid_from_df(
        df_train,
        kernel_config,
        properties=properties,
    )
    test_X, test_Y, test_id = get_XYid_from_df(
        df_test,
        kernel_config,
        properties=properties,
    )
    if test_X is None:
        test_X = train_X
        test_Y = np.copy(train_Y)
        test_id = train_id
    print('***\tEnd: Reading input.\t***\n')
    return (df, df_train, df_test, train_X, train_Y, train_id, test_X,
            test_Y, test_id)


def gpr_run(data, result_dir, kernel_config, params, load_model=False, tag=0):
    df = data['df']
    df_train = data['df_train']
    train_X = data['train_X']
    train_Y = data['train_Y']
    train_id = data['train_id']
    test_X = data['test_X']
    test_Y = data['test_Y']
    test_id = data['test_id']
    mode = params['mode']
    model = params['model']
    consensus_config = params['consensus_config']
    consensus, n_estimators, n_sample_per_model, n_jobs, consensus_rule = \
        set_consensus_config(consensus_config)
    n_nystrom_core = int(params['nystrom_config'])
    dynamic_train_size = params['dynamic_train_size']

    # pre-calculate graph kernel matrix.
    '''
    if params['optimizer'] is None:
        pre_calculate(kernel_config, df, result_dir, load_K)
    '''

    print('***\tStart: hyperparameters optimization.\t***')
    if mode == 'loocv':  # directly calculate the LOOCV
        learner = GPRLearner(
            model, train_X, train_Y, train_id, test_X, test_Y, test_id,
            consensus=consensus, n_estimators=n_estimators,
            n_sample_per_model=n_sample_per_model, n_jobs=n_jobs,
            consensus_rule=consensus_rule,
            n_nystrom_core=n_nystrom_core
        )
        if load_model:
            print('loading existed model')
            learner.model.load(result_dir)
        else:
            learner.train()
            learner.model.save(result_dir, overwrite=True)
            kernel_config.save(result_dir, learner.model_)
        out, r2, ex_var, mae, rmse, mse = learner.evaluate_loocv()
        print('LOOCV:')
        print('score: %.5f' % r2)
        print('explained variance score: %.5f' % ex_var)
        print('mae: %.5f' % mae)
        print('rmse: %.5f' % rmse)
        print('mse: %.5f' % mse)
        out.to_csv('%s/loocv.log' % result_dir, sep='\t', index=False,
                   float_format='%15.10f')
    elif mode == 'dynamic':
        learner = GPRLearner(
            model, train_X, train_Y, train_id, test_X, test_Y, test_id,
            consensus=consensus, n_estimators=n_estimators,
            n_sample_per_model=n_sample_per_model, n_jobs=n_jobs,
            consensus_rule=consensus_rule,
            n_nystrom_core=n_nystrom_core
        )
        out, r2, ex_var, mae, rmse, mse = learner.evaluate_test_dynamic(
            dynamic_train_size=dynamic_train_size)
        print('Test set:')
        print('score: %.5f' % r2)
        print('explained variance score: %.5f' % ex_var)
        print('mae: %.5f' % mae)
        print('rmse: %.5f' % rmse)
        print('mse: %.5f' % mse)
        out.to_csv('%s/test-%i.log' % (result_dir, tag), sep='\t', index=False,
                   float_format='%15.10f')
    elif mode == 'train_test':
        learner = GPRLearner(
            model, train_X, train_Y, train_id, test_X, test_Y, test_id,
            consensus=consensus, n_estimators=n_estimators,
            n_sample_per_model=n_sample_per_model, n_jobs=n_jobs,
            consensus_rule=consensus_rule,
            n_nystrom_core=n_nystrom_core
        )
        learner.train()
        learner.model.save(result_dir, overwrite=True)
        kernel_config.save(result_dir, learner.model_)
        print('***\tEnd: hyperparameters optimization.\t***\n')
        predict_train = False
        if predict_train:
            out, r2, ex_var, mae, rmse, mse = learner.evaluate_train()
            print('Training set:')
            print('score: %.5f' % r2)
            print('explained variance score: %.5f' % ex_var)
            print('mae: %.5f' % mae)
            print('rmse: %.5f' % rmse)
            print('mse: %.5f' % mse)
            out.to_csv('%s/train-%i.log' % (result_dir, tag), sep='\t', index=False,
                       float_format='%15.10f')
        out, r2, ex_var, mae, rmse, mse = learner.evaluate_test()
        print('Test set:')
        print('score: %.5f' % r2)
        print('explained variance score: %.5f' % ex_var)
        print('mae: %.5f' % mae)
        print('rmse: %.5f' % rmse)
        print('mse: %.5f' % mse)
        out.to_csv('%s/test-%i.log' % (result_dir, tag), sep='\t', index=False,
                   float_format='%15.10f')
    elif mode == 'all':
        learner = GPRLearner(
            model, train_X, train_Y, train_id, test_X, test_Y, test_id,
            consensus=consensus, n_estimators=n_estimators,
            n_sample_per_model=n_sample_per_model, n_jobs=n_jobs,
            consensus_rule=consensus_rule,
            n_nystrom_core=n_nystrom_core
        )
        learner.train()
        learner.model.save(result_dir, overwrite=True)
        kernel_config.save(result_dir, learner.model_)
        print('***\tEnd: hyperparameters optimization.\t***\n')
    else:
        raise RuntimeError(f'Unknown mode{mode}')


def gpc_run(data, result_dir, kernel_config, params, tag=0):
    df = data['df']
    df_train = data['df_train']
    train_X = data['train_X']
    train_Y = data['train_Y']
    train_id = data['train_id']
    test_X = data['test_X']
    test_Y = data['test_Y']
    test_id = data['test_id']
    mode = params['mode']
    model = params['model']

    print('***\tStart: hyperparameters optimization.\t***')
    if mode == 'loocv':  # directly calculate the LOOCV
        # to be done
        exit(0)
    elif mode == 'dynamic':
        # to be done
        exit(0)
    else:
        learner = ClassificationLearner(model, train_X, train_Y, train_id, test_X, test_Y,
                                        test_id)
        learner.train()
        # learner.model.save(result_dir)
        # learner.kernel_config.save(result_dir, learner.model)
        print('***\tEnd: hyperparameters optimization.\t***\n')
        predict_train = True
        if predict_train:
            out, accuracy, precision, recall, f1 = learner.evaluate_train()
            print('Training set:')
            print('accuracy: %.3f' % accuracy)
            print('precision: %.3f' % precision)
            print('recall: %.3f' % recall)
            print('f1: %.3f' % f1)
            out.to_csv('%s/train-%i.log' % (result_dir, tag), sep='\t', index=False,
                       float_format='%15.10f')
        out, accuracy, precision, recall, f1 = learner.evaluate_test()
        print('Test set:')
        print('accuracy: %.3f' % accuracy)
        print('precision: %.3f' % precision)
        print('recall: %.3f' % recall)
        print('f1: %.3f' % f1)
        out.to_csv('%s/test-%i.log' % (result_dir, tag), sep='\t', index=False,
                   float_format='%15.10f')


def _get_uniX(X):
    return np.sort(np.unique(X))


def _pure2sg(inchi_or_smiles, HASH):
    """ Transform SMILES (or InChI) into Graph
    
    Parameters
    ----------
    inchi_or_smiles : SMILES (or InChI) of input molecule.
    HASH : hash string.

    Returns
    -------
    Graph used in GraphDot.
    """
    return HashGraph.from_inchi_or_smiles(inchi_or_smiles, HASH)


def _mixture2sg(mixture, HASH):
    inchi_or_smiles = mixture[::2]
    proportion = mixture[1::2]
    _config = list(map(lambda x: rdkit_config(concentration=x), proportion))
    graphs = list(map(lambda x: HashGraph.from_inchi_or_smiles,
                      inchi_or_smiles,
                      ['1'] * len(inchi_or_smiles),
                      _config))
    g = graphs[0].to_networkx()
    for _g in graphs[1:]:
        g = nx.disjoint_union(g, _g.to_networkx())
    g = HashGraph.from_networkx(g)
    g.HASH = HASH
    return g


def _mixture2mg(mixture, HASH):
    """ Transform a list, [SMILES1, n1, SMILES2, n2...] into
        [Graph1, n1, graph2, n2...]

    Parameters
    ----------
    mixture : a list contain SMILES (or InChI) and its proportion of input
        mixture.
    HASH : hash string prefix.

    Returns
    -------
    The SMILES (or InChI) are transformed into graphs used in GraphDot.
    """
    hashes = [str(HASH) + '_%d' % i for i in range(int(len(mixture) / 2))]
    mixture[::2] = list(map(HashGraph.from_inchi_or_smiles, mixture[::2],
                            [rdkit_config()] * int(len(mixture) / 2),
                            hashes))
    return mixture


def _reaction_agents2sg(reaction_smarts, HASH):
    return HashGraph.agent_from_reaction_smarts(reaction_smarts, HASH)


def _reaction_agents2mg(reaction_smarts, HASH):
    try:
        agents = []
        rxn = reaction_from_smarts(reaction_smarts)
        for i, mol in enumerate(rxn.GetAgents()):
            Chem.SanitizeMol(mol)
            hash_ = HASH + '_%d' % i
            agents += [HashGraph.from_rdkit(mol, hash_), 1.0]
        return agents
    except:
        return 'Parsing Error'


def _reaction2sg(reaction_smarts, HASH):
    return HashGraph.from_reaction_smarts(reaction_smarts, HASH)


def _reaction2mg(reaction_smarts, HASH):
    try:
        reaction = []
        rxn = reaction_from_smarts(reaction_smarts)
        ReactingAtoms = getReactingAtoms(rxn, depth=1)
        for i, reactant in enumerate(rxn.GetReactants()):
            Chem.SanitizeMol(reactant)
            hash_ = HASH + '_r%d' % i
            config_ = rdkit_config(reaction_center=ReactingAtoms)
            reaction += [HashGraph.from_rdkit(reactant, hash_, config_),
                         1.0]
            if reaction[-2].nodes.to_pandas()['ReactingCenter'].max() <= 0:
                print('Reactants error and return Parsing Error for reaction: '
                      '%s', reaction_smarts)
        for i, product in enumerate(rxn.GetProducts()):
            Chem.SanitizeMol(product)
            hash_ = HASH + '_p%d' % i
            config_ = rdkit_config(reaction_center=ReactingAtoms)
            reaction += [HashGraph.from_rdkit(product, hash_, config_),
                         -1.0]
            if reaction[-2].nodes.to_pandas()['ReactingCenter'].max() <= 0:
                print('Products error and return Parsing Error for reaction: '
                      '%s', reaction_smarts)
        return reaction
    except:
        return 'Parsing Error'


def single2graph(args_kwargs):
    """ Function used for parallel graph transformation. This function cannot be
        pickled if put in get_df
    """
    df, sg = args_kwargs
    if len(np.unique(df[sg])) > 0.5 * len(df[sg]):
        return df.progress_apply(
            lambda x: HashGraph.from_inchi_or_smiles(
                x[sg], str(x['group_id'])), axis=1)
    else:
        graphs = []
        gids = []
        for g in df.groupby('group_id'):
            assert (len(g[1][sg].unique()) == 1)
            graphs.append(HashGraph.from_inchi_or_smiles(
                g[1][sg].tolist()[0], str(g[0])))
            gids.append(g[0])
        idx = np.searchsorted(gids, df['group_id'])
    return np.asarray(graphs)[idx]


def multi2graph(args_kwargs):
    """ Function used for parallel graph transformation. This function cannot be
        pickled if put in get_df
    """
    df, mg = args_kwargs
    if len(np.unique(df[mg])) > 0.5 * len(df[mg]):
        return df.progress_apply(
            lambda x: _mixture2mg(
                x[mg], str(x['group_id'])), axis=1)
    else:
        graphs = []
        gids = []
        for g in df.groupby('group_id'):
            graphs.append(_mixture2mg(g[1][mg][0], g[0]))
            gids.append(g[0])
        idx = np.searchsorted(gids, df['group_id'])
        return np.asarray(graphs)[idx]


def get_df(csv, pkl, single_graph, multi_graph, reaction_graph, n_process=1,
           parallel='Parallel'):
    if pkl is not None and os.path.exists(pkl):
        print('reading existing pkl file: %s' % pkl)
        df = pd.read_pickle(pkl)
    else:
        df = pd.read_csv(csv, sep='\s+', header=0)
        # set id and group_id
        if 'id' not in df:
            df['id'] = df.index
        groups = df.groupby(single_graph + multi_graph + reaction_graph)
        df['group_id'] = 0
        for i, g in enumerate(groups):
            g[1]['group_id'] = i
            df.update(g[1])
        df['id'] = df['id'].astype(int)
        df['group_id'] = df['group_id'].astype(int)
        # transform single graph
        for sg in single_graph:
            print('Transforming molecules into graphs. (pure compounds)')
            if parallel == 'pool':
                df_parts = np.array_split(df, n_process)
                with Pool(processes=n_process) as pool:
                    result_parts = pool.map(
                        single2graph, [(df_part, sg) for df_part in df_parts])
                df[sg + '_sg'] = np.concatenate(result_parts)
            else:
                df[sg + '_sg'] = Parallel(
                    n_jobs=n_process, verbose=True,
                    **_joblib_parallel_args(prefer='processes'))(
                    delayed(_pure2sg)(
                        df.iloc[i][sg], df.iloc[i]['group_id'].astype(str))
                    for i in df.index)
            unify_datatype(df[sg + '_sg'])
        # transform multi graph
        for mg in multi_graph:
            print('Transforming molecules into graphs. (mixtures)')
            if parallel == 'pool':
                df_parts = np.array_split(df, n_process)
                with Pool(processes=n_process) as pool:
                    result_parts = pool.map(
                        multi2graph, [(df_part, mg) for df_part in df_parts])
                df[mg] = np.concatenate(result_parts)
            else:
                df[mg + '_sg'] = Parallel(
                    n_jobs=n_process, verbose=True,
                    **_joblib_parallel_args(prefer='processes'))(
                    delayed(_mixture2sg)(
                        df.iloc[i][mg], df.iloc[i]['group_id'].astype(str))
                    for i in df.index)
                df[mg] = Parallel(n_jobs=n_process, verbose=True,
                                  **_joblib_parallel_args(prefer='processes'))(
                    delayed(_mixture2mg)(
                        df.iloc[i][mg], df.iloc[i]['group_id'].astype(str))
                    for i in df.index)
                unify_datatype(df[mg + '_sg'])
            unify_datatype(df[mg])
        # transform reaction graph
        for rg in reaction_graph:
            print('Transforming reagents into single graphs.')
            df[rg + '_agents_sg'] = Parallel(
                n_jobs=n_process, verbose=True,
                **_joblib_parallel_args(prefer='processes'))(
                delayed(_reaction_agents2sg)(
                    df.iloc[i][rg],
                    df.iloc[i]['group_id'].astype(str))
                for i in df.index)
            df = df[~df[rg + '_agents_sg'].isin(['Parsing Error'])].\
                reset_index().drop(columns='index')
            unify_datatype(df[rg + '_agents_sg'])
            print('Transforming reagents into multi graphs.')
            df[rg + '_agents_mg'] = Parallel(
                n_jobs=n_process, verbose=True,
                **_joblib_parallel_args(prefer='processes'))(
                delayed(_reaction_agents2mg)(df.iloc[i][rg],
                                             df.iloc[i]['group_id'].astype(str))
                for i in df.index)
            df = df[df[rg + '_agents_mg'] != 'Parsing Error'].\
                reset_index().drop(columns='index')
            unify_datatype(df[rg + '_agents_mg'])
            print('Transforming chemical reactions into single graphs.')
            df[rg + '_sg'] = Parallel(
                n_jobs=n_process, verbose=True,
                **_joblib_parallel_args(prefer='processes'))(
                delayed(_reaction2sg)(df.iloc[i][rg],
                                      df.iloc[i]['group_id'].astype(str))
                for i in df.index)
            df = df[~df[rg + '_sg'].isin(['Parsing Error'])].\
                reset_index().drop(columns='index')
            unify_datatype(df[rg + '_sg'])
            print('Transforming chemical reactions into multi graphs.')
            df[rg + '_mg'] = Parallel(
                n_jobs=n_process, verbose=True,
                **_joblib_parallel_args(prefer='processes'))(
                delayed(_reaction2mg)(df.iloc[i][rg],
                                      df.iloc[i]['group_id'].astype(str))
                for i in df.index)
            # df = df[~df[rg + '_mg'].isin(['Parsing Error'])]
            df = df[df[rg + '_mg'] != 'Parsing Error'].\
                reset_index().drop(columns='index')
            unify_datatype(df[rg + '_mg'])
        if pkl is not None:
            df.to_pickle(pkl)
    return df


def get_XYid_from_df(df, kernel_config, properties=None):
    if df.size == 0:
        return None, None, None
    if kernel_config.type == 'graph':
        X_name = kernel_config.single_graph + kernel_config.multi_graph
    elif kernel_config.type == 'preCalc':
        X_name = ['group_id']
    else:
        raise Exception('unknown kernel type:', kernel_config.type)
    if kernel_config.add_features is not None:
        X_name += kernel_config.add_features
    X = df[X_name].to_numpy()
    if properties is None:
        return X, None, None
    Y = df[properties].to_numpy()
    if len(properties) == 1:
        Y = Y.ravel()
    return X, Y, df['id'].to_numpy()


def get_Xgroupid_from_df(df, single_graph, multi_graph):
    if df.size == 0:
        return None, None
    X_name = single_graph + multi_graph
    df_ = []
    for x in df.groupby('group_id'):
        for name in X_name:
            assert (len(np.unique(x[1][name])) == 1)
        df_.append(x[1].sample(1))
    df_ = pd.concat(df_)
    return df_[X_name].to_numpy(), df_['group_id'].to_numpy()


def unify_datatype(X, Y=None):
    if X[0].__class__ == list:
        graphs = []
        for x in X:
            graphs += x[::2]
        if Y is not None:
            for y in Y:
                graphs += y[::2]
        HashGraph.unify_datatype(graphs, inplace=True)
    else:
        if Y is not None:
            graphs = np.concatenate([X, Y], axis=0)
        else:
            graphs = X
        HashGraph.unify_datatype(graphs, inplace=True)
