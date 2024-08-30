#!/usr/bin/env python

import pandas
import networkx
from enum import Enum

# new edge_type definitions
REACTION_REV = 'reaction_reverse'
REACTION_REACTANT = "reaction_reactant"
PRODUCT_REACTION = "product_reaction"
REACTION_REACTANT_REV = "reaction_reactant_reverse"
PRODUCT_REACTION_REV = "product_reaction_reverse"
GENE_REACTION_REV = 'gene_reaction_reverse'
GENE_REACTION = 'gene_reaction'
COMPOUND = 'compound'
REACTION = 'reaction'
GENE = 'gene'

class ReactionDirection(Enum):
    """
    An enum for the reaction direction.
    """

    FORWARD = "forward"
    REVERSE = "reverse"
    # A reaction that has both forward and reverse flow.
    REVERSABLE = "reversable"


def _reaction_direction(reaction):
    """
    Get the direction of the reaction.
    Args:
        reaction (cobra.Reaction): The reaction of direction.
    Return:
        (str): The reaction direction.
    """

    if reaction.lower_bound == 0 and reaction.upper_bound > 0:
        return ReactionDirection.FORWARD

    elif reaction.lower_bound < 0 and reaction.upper_bound == 0:
        return ReactionDirection.REVERSE

    return ReactionDirection.REVERSABLE


def _reaction_reactants_and_products(reaction):
    """
    Return a tuple of the reactants and products of a reaction. In the event the reaction is
    reversed, then the tuple is reversed.
    Args:
        reaction (cobra.Reaction): The reaction of which reactants and products should be exported.
    Returns:
        (list, list): A tuple of lists of metabolites, (reactants, products), unless the reaction is
            reversed, in which case (products, reactants).
    """

    rxn_direction = _reaction_direction(reaction)

    if rxn_direction != ReactionDirection.REVERSE:
        return reaction.reactants, reaction.products

    return reaction.products, reaction.reactants



def networkx_graph(model):

    multigraph = networkx.MultiDiGraph()

    for reaction in model.reactions:
        multigraph.add_node(reaction.id, node_type=REACTION)

        rxn_direction = _reaction_direction(reaction)
        reactants, products = _reaction_reactants_and_products(reaction)

        for reactant in reactants:
            multigraph.add_node(reactant.id, node_type=COMPOUND)
            multigraph.add_edge(reaction.id, reactant.id, edge_type=REACTION_REACTANT)

        for product in products:
            multigraph.add_node(product.id, node_type=COMPOUND)
            multigraph.add_edge(product.id, reaction.id, edge_type=PRODUCT_REACTION)

        if rxn_direction == ReactionDirection.REVERSABLE:
            # make a new reaction node
            rxn_rev_id = f'{reaction.id}_rev'
            multigraph.add_node(rxn_rev_id, node_type=REACTION_REV)

            for reactant in reactants:
                multigraph.add_node(reactant.id, node_type=COMPOUND)
                multigraph.add_edge(reactant.id, rxn_rev_id, edge_type=PRODUCT_REACTION_REV)

            for product in products:
                multigraph.add_node(product.id, node_type=COMPOUND)
                multigraph.add_edge(rxn_rev_id, product.id, edge_type=REACTION_REACTANT_REV)

    return multigraph



def adjacency_matrix(network):
    adj_mat = networkx.adjacency_matrix(network)
    adj_mat = pandas.DataFrame(adj_mat.todense(), index=list(network.nodes()), columns=list(network.nodes()))
    return adj_mat