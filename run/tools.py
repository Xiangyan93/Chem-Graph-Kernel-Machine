import os
import json
import pickle
from tqdm import tqdm
tqdm.pandas()
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from chemml.regression.gpr_learner import GPRLearner
from chemml.graph.hashgraph import HashGraph
from chemml.graph.from_rdkit import rdkit_config
from chemml.graph.substructure import AtomEnvironment
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
    assert (gpr in ['graphdot', 'sklearn'])
    if optimizer in ['None', 'none', '']:
        return gpr, None
    if gpr == 'graphdot' and optimizer != 'L-BFGS-B':
        raise Exception('Please use L-BFGS-B optimizer')
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
    else:
        raise Exception('Unknown GaussianProcessRegressor: %s' % gpr)
    return model


def set_gpc_learner(gpc):
    if gpc == 'sklearn':
        from chemml.classification.gpc.learner import Learner
    else:
        raise Exception('Unknown GaussianProcessClassifier: %s' % gpc)
    return Learner


def set_gpr(gpr):
    if gpr == 'graphdot':
        from chemml.regression.GPRgraphdot.gpr import GPR
    elif gpr == 'sklearn':
        from chemml.regression.GPRsklearn.gpr import GPR
    else:
        raise Exception('Unknown GaussianProcessRegressor: %s' % gpr)
    return GPR


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
    def df_filter(df, train_size=None, train_ratio=None, bygroup=False, seed=0):
        if 'IsTrain' in df:
            return df[df.IsTrain == True], df[df.IsTrain == False]
        np.random.seed(seed)
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
        bygroup=kernel_config.add_features is not None
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
            consensus_rule=consensus_rule
        )
        if load_model:
            print('loading existed model')
            learner.model.load(result_dir)
        else:
            learner.train()
            learner.model.save(result_dir)
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
            consensus_rule=consensus_rule
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
    else:
        learner = GPRLearner(
            model, train_X, train_Y, train_id, test_X, test_Y, test_id,
            consensus=consensus, n_estimators=n_estimators,
            n_sample_per_model=n_sample_per_model, n_jobs=n_jobs,
            consensus_rule=consensus_rule
        )
        learner.train()
        learner.model.save(result_dir, overwrite=True)
        kernel_config.save(result_dir, learner.model_)
        print('***\tEnd: hyperparameters optimization.\t***\n')
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


def gpc_run(data, result_dir, kernel_config, params, load_model=False, tag=0):
    df = data['df']
    df_train = data['df_train']
    train_X = data['train_X']
    train_Y = data['train_Y']
    train_id = data['train_id']
    test_X = data['test_X']
    test_Y = data['test_Y']
    test_id = data['test_id']
    optimizer = params['optimizer']
    mode = params['mode']
    Learner = params['Learner']
    dynamic_train_size = params['dynamic_train_size']

    print('***\tStart: hyperparameters optimization.\t***')
    if mode == 'loocv':  # directly calculate the LOOCV
        # to be done
        exit(0)
    elif mode == 'dynamic':
        # to be done
        exit(0)
    else:
        learner = Learner(train_X, train_Y, train_id, test_X, test_Y,
                          test_id, kernel_config, optimizer=optimizer)
        learner.train()
        learner.model.save(result_dir)
        learner.kernel_config.save(result_dir, learner.model)
        print('***\tEnd: hyperparameters optimization.\t***\n')
        out, correct_ratio = learner.evaluate_train()
        print('Training set:')
        print('correct_ratio: %.3f' % correct_ratio)
        out.to_csv('%s/train-%i.log' % (result_dir, tag), sep='\t', index=False,
                   float_format='%15.10f')
        out, correct_ratio = learner.evaluate_test()
        print('Test set:')
        print('correct_ratio: %.3f' % correct_ratio)
        out.to_csv('%s/test-%i.log' % (result_dir, tag), sep='\t', index=False,
                   float_format='%15.10f')


def _get_uniX(X):
    return np.sort(np.unique(X))


def single2graph(args_kwargs):
    df, sg = args_kwargs
    if len(np.unique(df[sg])) > 0.5 * len(df[sg]):
        return df.progress_apply(
            lambda x: HashGraph.from_inchi_or_smiles(
                x[sg], rdkit_config(), str(x['group_id'])), axis=1)
    else:
        graphs = []
        gids = []
        for g in df.groupby('group_id'):
            assert (len(g[1][sg].unique()) == 1)
            graphs.append(HashGraph.from_inchi_or_smiles(
                g[1][sg].tolist()[0], rdkit_config(), str(g[0])))
            gids.append(g[0])
        idx = np.searchsorted(gids, df['group_id'])
        return np.asarray(graphs)[idx]


def multi_graph_transform(line, hash):
    hashs = [str(hash) + '_%d' % i for i in range(int(len(line) / 2))]
    line[::2] = list(map(HashGraph.from_inchi_or_smiles, line[::2],
                         [rdkit_config()] * int(len(line) / 2),
                         hashs))
    return line


def multi2graph(args_kwargs):
    df, mg = args_kwargs
    if len(np.unique(df[mg])) > 0.5 * len(df[mg]):
        return df.progress_apply(
            lambda x: multi_graph_transform(
                x[mg], str(x['group_id'])), axis=1)
    else:
        graphs = []
        gids = []
        for g in df.groupby('group_id'):
            graphs.append(multi_graph_transform(g[1][mg][0], g[0]))
            gids.append(g[0])
        idx = np.searchsorted(gids, df['group_id'])
        return np.asarray(graphs)[idx]


def get_df(csv, pkl, single_graph, multi_graph, reaction_graph, n_process=1):
    def reaction2agent(reaction_smarts, hash):
        agents = []
        rxn = rdChemReactions.ReactionFromSmarts(reaction_smarts)
        # print(reaction_smarts)
        for i, mol in enumerate(rxn.GetAgents()):
            Chem.SanitizeMol(mol)
            hash_ = hash + '_%d' % i
            config_ = rdkit_config()
            agents += [HashGraph.from_rdkit(mol, config_, hash_), 1.0]
        return agents

    def reaction2rp(reaction_smarts, hash):
        reaction = []
        rxn = rdChemReactions.ReactionFromSmarts(reaction_smarts)

        # rxn.Initialize()
        def getAtomMapDict(mols):
            AtomMapDict = dict()
            for mol in mols:
                Chem.SanitizeMol(mol)
                for atom in mol.GetAtoms():
                    AMN = atom.GetPropsAsDict().get('molAtomMapNumber')
                    if AMN is not None:
                        AtomMapDict[AMN] = AtomEnvironment(
                            mol, atom, depth=1)
            return AtomMapDict

        def getReactingAtoms(rxn):
            ReactingAtoms = []
            reactantAtomMap = getAtomMapDict(rxn.GetReactants())
            productAtomMap = getAtomMapDict(rxn.GetProducts())
            for id, AE in reactantAtomMap.items():
                if AE != productAtomMap.get(id):
                    ReactingAtoms.append(id)
            return ReactingAtoms

        ReactingAtoms = getReactingAtoms(rxn)
        for i, reactant in enumerate(rxn.GetReactants()):
            Chem.SanitizeMol(reactant)
            hash_ = hash + '_r%d' % i
            config_ = rdkit_config(reaction_center=ReactingAtoms)
            reaction += [HashGraph.from_rdkit(reactant, config_, hash_), 1.0]
            if True not in reaction[-2].nodes.to_pandas()['group_reaction']:
                raise Exception('Reactants error:', reaction_smarts)
        for i, product in enumerate(rxn.GetProducts()):
            Chem.SanitizeMol(product)
            hash_ = hash + '_p%d' % i
            config_ = rdkit_config(reaction_center=ReactingAtoms)
            reaction += [HashGraph.from_rdkit(product, config_, hash_), -1.0]
            if True not in reaction[-2].nodes.to_pandas()['group_reaction']:
                raise Exception('Products error:', reaction_smarts)
        return reaction

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
        df_parts = np.array_split(df, n_process)
        # transform single graph
        for sg in single_graph:
            print('Processing single graph.')
            with Pool(processes=n_process) as pool:
                result_parts = pool.map(
                    single2graph, [(df_part, sg) for df_part in df_parts])
            df[sg] = np.concatenate(result_parts)
            unify_datatype(df[sg])
        # transform multi graph
        for mg in multi_graph:
            print('Processing multi graph.')
            with Pool(processes=n_process) as pool:
                result_parts = pool.map(
                    multi2graph, [(df_part, mg) for df_part in df_parts])
            df[mg] = np.concatenate(result_parts)
            unify_datatype(df[mg])
        # transform reaction graph
        for rg in reaction_graph:
            print('Processing reagents graph.')
            df[rg + '_agents'] = df.progress_apply(
                lambda x: reaction2agent(x[rg], str(x['group_id'])), axis=1)
            unify_datatype(df[rg + '_agents'])
            print('Processing reactions graph.')
            df[rg] = df.progress_apply(
                lambda x: reaction2rp(x[rg], str(x['group_id'])), axis=1)
            unify_datatype(df[rg])
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
