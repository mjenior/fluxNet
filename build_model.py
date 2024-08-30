#!/usr/bin/env python3

from copy import deepcopy
import cobra, symengine

def load_model(filename):

    try:
        model = cobra.io.read_sbml_model(filename)
    except:
        model = cobra.io.load_json_model(filename)
        
    model.solver = 'glpk'

    return model


def create_strain_model(model, strain_dict):

    # Clone and rename model
    constrained = deepcopy(model)
    constrained.name = strain_dict['strain_ID']
    constrained.id = strain_dict['unique_ID']

    # Remove missing reactions
    constrained = _remove_reactions(constrained, strain_dict['missing_reactions'])

    # Constrain objective and substrate uptake flux
    #constrained.reactions.get_by_id(strain_dict['consumption_label']).upper_bound = strain_dict['consumption_rate']
    constrained = _add_constraint(constrained, strain_dict['growth_label'], strain_dict['growth_rate'])
    
    return constrained


def _add_constraint(model, rxn_id, flux, frac=0.9):

    expr = model.reactions.get_by_id(rxn_id).flux_expression
    constraint = model.problem.Constraint(expr, ub=flux, lb=flux*frac)
    model.add_cons_vars(constraint)
    model.solver.update()

    # Check for growth
    if model.slim_optimize(error_value=0.0) <= 1e-6:
        print('WARNING: Constrained model carries NO objective flux.')

    return model


def _remove_reactions(model, reactions):
    rm_rxns = reactions.intersection(set([rxn.id for rxn in model.reactions]))
    for rxn in rm_rxns: model.reactions.get_by_id(rxn).remove_from_model(remove_orphans=True)
    constrained = _complete_orphan_prune(model)
    model.repair()

    return model
    

# Thoroughly remove orphan reactions and metabolites
def _complete_orphan_prune(model):

    removed = 1
    while removed == 1:
        removed = 0

        # Metabolites
        for cpd in model.metabolites:
            if len(cpd.reactions) == 0:
                cpd.remove_from_model()
                removed = 1

        # Reactions
        for rxn in model.reactions:
            if len(rxn.metabolites) == 0: 
                rxn.remove_from_model()
                removed = 1
    
    return model