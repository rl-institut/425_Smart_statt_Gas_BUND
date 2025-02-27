# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Solves optimal operation and capacity for a network with the option to
iteratively optimize while updating line reactances.

This script is used for optimizing the electrical network as well as the
sector coupled network.

Description
-----------

Total annual system costs are minimised with PyPSA. The full formulation of the
linear optimal power flow (plus investment planning
is provided in the
`documentation of PyPSA <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#linear-optimal-power-flow>`_.

The optimization is based on the :func:`network.optimize` function.
Additionally, some extra constraints specified in :mod:`solve_network` are added.

.. note::

    The rules ``solve_elec_networks`` and ``solve_sector_networks`` run
    the workflow for all scenarios in the configuration file (``scenario:``)
    based on the rule :mod:`solve_network`.
"""

import importlib
import logging
import os
import re
import sys

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
import yaml
from _benchmark import memory_logger
from _helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)
from prepare_sector_network import get
from pypsa.descriptors import get_activity_mask
from pypsa.descriptors import get_switchable_as_dense as get_as_dense

logger = logging.getLogger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)


class ObjectiveValueError(Exception):
    pass


def add_land_use_constraint_perfect(n):
    """
    Add global constraints for tech capacity limit.
    """
    logger.info("Add land-use constraint for perfect foresight")

    def compress_series(s):
        def process_group(group):
            if group.nunique() == 1:
                return pd.Series(group.iloc[0], index=[None])
            else:
                return group

        return s.groupby(level=[0, 1]).apply(process_group)

    def new_index_name(t):
        # Convert all elements to string and filter out None values
        parts = [str(x) for x in t if x is not None]
        # Join with space, but use a dash for the last item if not None
        return " ".join(parts[:2]) + (f"-{parts[-1]}" if len(parts) > 2 else "")

    def check_p_min_p_max(p_nom_max):
        p_nom_min = n.generators[ext_i].groupby(grouper).sum().p_nom_min
        p_nom_min = p_nom_min.reindex(p_nom_max.index)
        check = (
            p_nom_min.groupby(level=[0, 1]).sum()
            > p_nom_max.groupby(level=[0, 1]).min()
        )
        if check.sum():
            logger.warning(
                f"summed p_min_pu values at node larger than technical potential {check[check].index}"
            )

    grouper = [n.generators.carrier, n.generators.bus, n.generators.build_year]
    ext_i = n.generators.p_nom_extendable
    # get technical limit per node and investment period
    p_nom_max = n.generators[ext_i].groupby(grouper).min().p_nom_max
    # drop carriers without tech limit
    p_nom_max = p_nom_max[~p_nom_max.isin([np.inf, np.nan])]
    # carrier
    carriers = p_nom_max.index.get_level_values(0).unique()
    gen_i = n.generators[(n.generators.carrier.isin(carriers)) & (ext_i)].index
    n.generators.loc[gen_i, "p_nom_min"] = 0
    # check minimum capacities
    check_p_min_p_max(p_nom_max)
    # drop multi entries in case p_nom_max stays constant in different periods
    # p_nom_max = compress_series(p_nom_max)
    # adjust name to fit syntax of nominal constraint per bus
    df = p_nom_max.reset_index()
    df["name"] = df.apply(
        lambda row: f"nom_max_{row['carrier']}"
        + (f"_{row['build_year']}" if row["build_year"] is not None else ""),
        axis=1,
    )

    for name in df.name.unique():
        df_carrier = df[df.name == name]
        bus = df_carrier.bus
        n.buses.loc[bus, name] = df_carrier.p_nom_max.values

    return n


def add_land_use_constraint(n):
    # warning: this will miss existing offwind which is not classed AC-DC and has carrier 'offwind'

    for carrier in [
        "solar",
        "solar rooftop",
        "solar-hsat",
        "onwind",
        "offwind-ac",
        "offwind-dc",
        "offwind-float",
    ]:
        ext_i = (n.generators.carrier == carrier) & ~n.generators.p_nom_extendable
        existing = (
            n.generators.loc[ext_i, "p_nom"]
            .groupby(n.generators.bus.map(n.buses.location))
            .sum()
        )
        existing.index += " " + carrier + "-" + snakemake.wildcards.planning_horizons
        n.generators.loc[existing.index, "p_nom_max"] -= existing

    # check if existing capacities are larger than technical potential
    existing_large = n.generators[
        n.generators["p_nom_min"] > n.generators["p_nom_max"]
    ].index
    if len(existing_large):
        logger.warning(
            f"Existing capacities larger than technical potential for {existing_large},\
                        adjust technical potential to existing capacities"
        )
        n.generators.loc[existing_large, "p_nom_max"] = n.generators.loc[
            existing_large, "p_nom_min"
        ]

    n.generators["p_nom_max"] = n.generators["p_nom_max"].clip(lower=0)


def add_solar_potential_constraints(n, config):
    """
    Add constraint to make sure the sum capacity of all solar technologies (fixed, tracking, ets. ) is below the region potential.

    Example:
    ES1 0: total solar potential is 10 GW, meaning:
           solar potential : 10 GW
           solar-hsat potential : 8 GW (solar with single axis tracking is assumed to have higher land use)
    The constraint ensures that:
           solar_p_nom + solar_hsat_p_nom * 1.13 <= 10 GW
    """
    land_use_factors = {
        "solar-hsat": config["renewable"]["solar"]["capacity_per_sqkm"]
        / config["renewable"]["solar-hsat"]["capacity_per_sqkm"],
    }
    rename = {"Generator-ext": "Generator"}

    solar_carriers = ["solar", "solar-hsat"]
    solar = n.generators[
        n.generators.carrier.isin(solar_carriers) & n.generators.p_nom_extendable
    ].index

    solar_today = n.generators[
        (n.generators.carrier == "solar") & (n.generators.p_nom_extendable)
    ].index
    solar_hsat = n.generators[(n.generators.carrier == "solar-hsat")].index

    if solar.empty:
        return

    land_use = pd.DataFrame(1, index=solar, columns=["land_use_factor"])
    for carrier, factor in land_use_factors.items():
        land_use = land_use.apply(
            lambda x: (x * factor) if carrier in x.name else x, axis=1
        )

    location = pd.Series(n.buses.index, index=n.buses.index)
    ggrouper = n.generators.loc[solar].bus
    rhs = (
        n.generators.loc[solar_today, "p_nom_max"]
        .groupby(n.generators.loc[solar_today].bus.map(location))
        .sum()
        - n.generators.loc[solar_hsat, "p_nom"]
        .groupby(n.generators.loc[solar_hsat].bus.map(location))
        .sum()
        * land_use_factors["solar-hsat"]
    ).clip(lower=0)

    lhs = (
        (n.model["Generator-p_nom"].rename(rename).loc[solar] * land_use.squeeze())
        .groupby(ggrouper)
        .sum()
    )

    logger.info("Adding solar potential constraint.")
    n.model.add_constraints(lhs <= rhs, name="solar_potential")


def add_co2_sequestration_limit(n, limit_dict):
    """
    Add a global constraint on the amount of Mt CO2 that can be sequestered.
    """

    if not n.investment_periods.empty:
        periods = n.investment_periods
        limit = pd.Series(
            {
                f"co2_sequestration_limit-{period}": limit_dict.get(period, 200)
                for period in periods
            }
        )
        names = limit.index
    else:
        limit = get(limit_dict, int(snakemake.wildcards.planning_horizons))
        periods = [np.nan]
        names = pd.Index(["co2_sequestration_limit"])

    n.add(
        "GlobalConstraint",
        names,
        sense=">=",
        constant=-limit * 1e6,
        type="operational_limit",
        carrier_attribute="co2 sequestered",
        investment_period=periods,
    )


def add_carbon_constraint(n, snapshots):
    glcs = n.global_constraints.query('type == "co2_atmosphere"')
    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]

        if emissions.empty:
            continue

        # stores
        bus_carrier = n.stores.bus.map(n.buses.carrier)
        stores = n.stores[bus_carrier.isin(emissions.index) & ~n.stores.e_cyclic]
        if not stores.empty:
            last = n.snapshot_weightings.reset_index().groupby("period").last()
            last_i = last.set_index([last.index, last.timestep]).index
            final_e = n.model["Store-e"].loc[last_i, stores.index]
            time_valid = int(glc.loc["investment_period"])
            time_i = pd.IndexSlice[time_valid, :]
            lhs = final_e.loc[time_i, :] - final_e.shift(snapshot=1).loc[time_i, :]

            rhs = glc.constant
            n.model.add_constraints(lhs <= rhs, name=f"GlobalConstraint-{name}")


def add_carbon_budget_constraint(n, snapshots):
    glcs = n.global_constraints.query('type == "Co2Budget"')
    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]

        if emissions.empty:
            continue

        # stores
        bus_carrier = n.stores.bus.map(n.buses.carrier)
        stores = n.stores[bus_carrier.isin(emissions.index) & ~n.stores.e_cyclic]
        if not stores.empty:
            last = n.snapshot_weightings.reset_index().groupby("period").last()
            last_i = last.set_index([last.index, last.timestep]).index
            final_e = n.model["Store-e"].loc[last_i, stores.index]
            time_valid = int(glc.loc["investment_period"])
            time_i = pd.IndexSlice[time_valid, :]
            weighting = n.investment_period_weightings.loc[time_valid, "years"]
            lhs = final_e.loc[time_i, :] * weighting

            rhs = glc.constant
            n.model.add_constraints(lhs <= rhs, name=f"GlobalConstraint-{name}")


def add_max_growth(n):
    """
    Add maximum growth rates for different carriers.
    """

    opts = snakemake.params["sector"]["limit_max_growth"]
    # take maximum yearly difference between investment periods since historic growth is per year
    factor = n.investment_period_weightings.years.max() * opts["factor"]
    for carrier in opts["max_growth"].keys():
        max_per_period = opts["max_growth"][carrier] * factor
        logger.info(
            f"set maximum growth rate per investment period of {carrier} to {max_per_period} GW."
        )
        n.carriers.loc[carrier, "max_growth"] = max_per_period * 1e3

    for carrier in opts["max_relative_growth"].keys():
        max_r_per_period = opts["max_relative_growth"][carrier]
        logger.info(
            f"set maximum relative growth per investment period of {carrier} to {max_r_per_period}."
        )
        n.carriers.loc[carrier, "max_relative_growth"] = max_r_per_period

    return n


def add_retrofit_gas_boiler_constraint(n, snapshots):
    """
    Allow retrofitting of existing gas boilers to H2 boilers.
    """
    c = "Link"
    logger.info("Add constraint for retrofitting gas boilers to H2 boilers.")
    # existing gas boilers
    mask = n.links.carrier.str.contains("gas boiler") & ~n.links.p_nom_extendable
    gas_i = n.links[mask].index
    mask = n.links.carrier.str.contains("retrofitted H2 boiler")
    h2_i = n.links[mask].index

    n.links.loc[gas_i, "p_nom_extendable"] = True
    p_nom = n.links.loc[gas_i, "p_nom"]
    n.links.loc[gas_i, "p_nom"] = 0

    # heat profile
    cols = n.loads_t.p_set.columns[
        n.loads_t.p_set.columns.str.contains("heat")
        & ~n.loads_t.p_set.columns.str.contains("industry")
        & ~n.loads_t.p_set.columns.str.contains("agriculture")
    ]
    profile = n.loads_t.p_set[cols].div(
        n.loads_t.p_set[cols].groupby(level=0).max(), level=0
    )
    # to deal if max value is zero
    profile.fillna(0, inplace=True)
    profile.rename(columns=n.loads.bus.to_dict(), inplace=True)
    profile = profile.reindex(columns=n.links.loc[gas_i, "bus1"])
    profile.columns = gas_i

    rhs = profile.mul(p_nom)

    dispatch = n.model["Link-p"]
    active = get_activity_mask(n, c, snapshots, gas_i)
    rhs = rhs[active]
    p_gas = dispatch.sel(Link=gas_i)
    p_h2 = dispatch.sel(Link=h2_i)

    lhs = p_gas + p_h2

    n.model.add_constraints(lhs == rhs, name="gas_retrofit")


def prepare_network(
    n,
    solve_opts=None,
    config=None,
    foresight=None,
    planning_horizons=None,
    co2_sequestration_potential=None,
):
    if "clip_p_max_pu" in solve_opts:
        for df in (
            n.generators_t.p_max_pu,
            n.generators_t.p_min_pu,
            n.links_t.p_max_pu,
            n.links_t.p_min_pu,
            n.storage_units_t.inflow,
        ):
            df.where(df > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)

    if load_shedding := solve_opts.get("load_shedding"):
        # intersect between macroeconomic and surveybased willingness to pay
        # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
        # TODO: retrieve color and nice name from config
        n.add("Carrier", "load", color="#dd2e23", nice_name="Load shedding")
        buses_i = n.buses.index
        if not np.isscalar(load_shedding):
            # TODO: do not scale via sign attribute (use Eur/MWh instead of Eur/kWh)
            load_shedding = 1e2  # Eur/kWh

        n.add(
            "Generator",
            buses_i,
            " load",
            bus=buses_i,
            carrier="load",
            sign=1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
            marginal_cost=load_shedding,  # Eur/kWh
            p_nom=1e9,  # kW
        )

    if solve_opts.get("curtailment_mode"):
        n.add("Carrier", "curtailment", color="#fedfed", nice_name="Curtailment")
        n.generators_t.p_min_pu = n.generators_t.p_max_pu
        buses_i = n.buses.query("carrier == 'AC'").index
        n.add(
            "Generator",
            buses_i,
            suffix=" curtailment",
            bus=buses_i,
            p_min_pu=-1,
            p_max_pu=0,
            marginal_cost=-0.1,
            carrier="curtailment",
            p_nom=1e6,
        )

    if solve_opts.get("noisy_costs"):
        for t in n.iterate_components():
            # if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if "marginal_cost" in t.df:
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (
                    np.full(len(t.df), 0.5) - 0.5
                    # np.random.random(len(t.df)) - 0.5
                )

        for t in n.iterate_components(["Line", "Link"]):
            t.df["capital_cost"] += (
                1e-1 + 2e-2 * (
                    np.full(len(t.df), 0.5) - 0.5
                    # np.random.random(len(t.df)) - 0.5
                )
            ) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

    if foresight == "myopic":
        add_land_use_constraint(n)

    if foresight == "perfect":
        n = add_land_use_constraint_perfect(n)
        if snakemake.params["sector"]["limit_max_growth"]["enable"]:
            n = add_max_growth(n)

    if n.stores.carrier.eq("co2 sequestered").any():
        limit_dict = co2_sequestration_potential
        add_co2_sequestration_limit(n, limit_dict=limit_dict)

    return n


def add_CCL_constraints(n, config):
    """
    Add CCL (country & carrier limit) constraint to the network and set fixed capacities.
    """
    logger.info("=== Starting CCL Constraint Addition ===")

    try:
        # Read limits with multi-level columns
        agg_p_nom_minmax = pd.read_csv(
            config["solving"]["agg_p_nom_limits"]["file"],
            index_col=[0, 1],
            header=[0, 1]
        )[snakemake.wildcards.planning_horizons]

        # List of technologies to fix
        fixed_techs = ['onwind', 'offwind-ac', 'offwind-dc', 'solar', 'solar rooftop']

        for idx, row in agg_p_nom_minmax.iterrows():
            country, carrier = idx
            min_val = row['min']
            max_val = row['max']

            # Get all generators for this country and carrier
            gens_mask = (
                    (n.generators.bus.map(n.buses.country) == country) &
                    (n.generators.carrier == carrier)
            )
            relevant_gens = n.generators[gens_mask]

            if relevant_gens.empty:
                logger.warning(f"No generators found for {carrier} in {country}")
                continue

            if carrier in fixed_techs:
                # For fixed technologies, set exact capacities
                if not np.isclose(min_val, max_val):
                    logger.warning(
                        f"Min and max values differ for fixed technology {carrier}. Using min value: {min_val}")

                target_capacity = min_val
                current_capacity = relevant_gens.p_nom.sum()

                logger.info(f"\nSetting fixed capacity for {carrier} in {country}:")
                logger.info(f"Target capacity: {target_capacity:.2f} MW")
                logger.info(f"Current capacity: {current_capacity:.2f} MW")

                # Make all generators non-extendable
                n.generators.loc[relevant_gens.index, 'p_nom_extendable'] = False

                if len(relevant_gens) > 1:
                    # Distribute capacity proportionally among generators
                    scaling_factor = target_capacity / current_capacity
                    for idx in relevant_gens.index:
                        old_capacity = n.generators.loc[idx, 'p_nom']
                        new_capacity = old_capacity * scaling_factor
                        n.generators.loc[idx, 'p_nom'] = new_capacity
                        logger.info(f"Generator {idx}: {old_capacity:.2f} MW → {new_capacity:.2f} MW")
                else:
                    # Set capacity directly for single generator
                    idx = relevant_gens.index[0]
                    old_capacity = n.generators.loc[idx, 'p_nom']
                    n.generators.loc[idx, 'p_nom'] = target_capacity
                    logger.info(f"Generator {idx}: {old_capacity:.2f} MW → {target_capacity:.2f} MW")

                # Verify the total capacity
                new_total = n.generators.loc[relevant_gens.index, 'p_nom'].sum()
                logger.info(f"New total capacity: {new_total:.2f} MW")

                if not np.isclose(new_total, target_capacity, rtol=1e-3):
                    logger.warning(
                        f"Warning: New capacity {new_total:.2f} MW differs from target {target_capacity:.2f} MW")

            else:
                # For other technologies, add min/max constraints as before
                extendable_gens = relevant_gens[relevant_gens.p_nom_extendable]

                if extendable_gens.empty:
                    logger.warning(f"No extendable generators found for {carrier} in {country}")
                    continue

                # Get model components
                p_nom = n.model["Generator-p_nom"]
                lhs = p_nom.loc[extendable_gens.index].sum()

                # Add minimum constraint if applicable
                if min_val > 0:
                    constraint_name = f"agg_p_nom_min_{country}_{carrier}"
                    n.model.add_constraints(
                        lhs >= min_val,
                        name=constraint_name
                    )
                    logger.info(f"Added minimum constraint: {constraint_name} >= {min_val}")

                # Add maximum constraint if applicable
                if max_val < float('inf'):
                    constraint_name = f"agg_p_nom_max_{country}_{carrier}"
                    n.model.add_constraints(
                        lhs <= max_val,
                        name=constraint_name
                    )
                    logger.info(f"Added maximum constraint: {constraint_name} <= {max_val}")

        logger.info("\nAll CCL constraints and fixed capacities set successfully")

    except Exception as e:
        logger.error(f"Error in CCL constraint addition: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        raise

    return n


def add_CLL_constraints(n, config):
    """
    Add CLL (Country Link Limit) constraints for coal and lignite links.
    """
    logging.info("====== STARTING CLL CONSTRAINTS FOR LINKS ======")

    try:
        # Read the simplified limits file from data directory
        limits_df = pd.read_csv("data/technology_limits.csv")

        # Get current planning year
        planning_year = int(snakemake.wildcards.planning_horizons)

        # Get German buses
        de_buses = n.buses[n.buses.country == 'DE'].index

        logging.info(f"\nProcessing year: {planning_year}")

        # Process coal and lignite
        for carrier in ['coal', 'lignite']:
            # Get target capacities for current year and technology
            year_limits = limits_df[
                (limits_df['year'] == planning_year) &
                (limits_df['technology'] == carrier) &
                (limits_df['country'] == 'DE')
                ].iloc[0]

            min_cap = year_limits['capacity_min_mw']
            max_cap = year_limits['capacity_max_mw']

            # Calculate existing capacity
            existing_links = n.links[
                (n.links.carrier == carrier) &
                (n.links.bus0 == f'EU {carrier}') &
                (n.links.bus1.isin(de_buses))
                ]
            existing_cap = existing_links.p_nom.sum()

            logging.info(f"\n{carrier.upper()} CAPACITY LIMITS:")
            logging.info(f"Minimum required: {min_cap} MW")
            logging.info(f"Maximum allowed: {max_cap} MW")
            logging.info(f"Current capacity: {existing_cap:.2f} MW")

            if existing_cap > max_cap:
                # Need to reduce capacity
                reduction_needed = existing_cap - max_cap
                logging.info(f"Need to reduce capacity by {reduction_needed:.2f} MW")

                # Sort links by year (oldest first)
                years = existing_links.index.str.extract('(\d{4})').astype(float)
                sorted_links = existing_links.iloc[years.iloc[:, 0].argsort()]

                remaining_capacity = max_cap
                for link_name, link in sorted_links.iterrows():
                    # Make link non-extendable
                    n.links.at[link_name, 'p_nom_extendable'] = False

                    if remaining_capacity > 0:
                        # Calculate capacity for this link
                        new_cap = min(link['p_nom'], remaining_capacity)
                        # Set fixed capacity
                        n.links.at[link_name, 'p_nom'] = new_cap
                        remaining_capacity -= new_cap
                        logging.info(f"Set fixed capacity for {link_name} to {new_cap:.2f} MW")
                    else:
                        # No more capacity needed, set to zero
                        n.links.at[link_name, 'p_nom'] = 0
                        logging.info(f"Decommissioned {link_name}")

            elif existing_cap < min_cap:
                # Need to add capacity
                addition_needed = min_cap - existing_cap
                logging.info(f"Need to add capacity of {addition_needed:.2f} MW")

                if not existing_links.empty:
                    # Use newest existing link
                    years = existing_links.index.str.extract('(\d{4})').astype(float)
                    newest_link = existing_links.index[years.idxmax()[0]]
                    current_cap = n.links.at[newest_link, 'p_nom']
                    target_cap = current_cap + addition_needed

                    # Make link non-extendable and set capacity
                    n.links.at[newest_link, 'p_nom_extendable'] = False
                    n.links.at[newest_link, 'p_nom'] = target_cap

                    logging.info(f"Updated {newest_link}:")
                    logging.info(f"  - Fixed capacity: {target_cap:.2f} MW")
                else:
                    # Create new link
                    new_link = f"DE0 0 {carrier}-{planning_year}"
                    n.add("Link",
                          new_link,
                          bus0=f"EU {carrier}",
                          bus1="DE0 0",
                          carrier=carrier,
                          p_nom=min_cap,
                          p_nom_extendable=False)  # Make non-extendable

                    logging.info(f"Added new {carrier} link: {new_link}")
                    logging.info(f"  - Fixed capacity: {min_cap} MW")

            else:
                # Current capacity is within limits, fix at current levels
                logging.info("Current capacity is within limits")
                for link_name in existing_links.index:
                    current_cap = n.links.at[link_name, 'p_nom']
                    # Make link non-extendable
                    n.links.at[link_name, 'p_nom_extendable'] = False
                    logging.info(f"Fixed capacity for {link_name}:")
                    logging.info(f"  - Capacity: {current_cap:.2f} MW")

        # Verify final capacities
        logging.info("\n====== FINAL CAPACITIES ======")
        for carrier in ['coal', 'lignite']:
            links = n.links[
                (n.links.carrier == carrier) &
                (n.links.bus0 == f'EU {carrier}') &
                (n.links.bus1.isin(de_buses))
                ]

            total_cap = links.p_nom.sum()

            logging.info(f"DE {carrier}:")
            logging.info(f"  - Fixed capacity: {total_cap:.2f} MW")

    except Exception as e:
        logging.error(f"Error while processing limits: {str(e)}")
        logging.error("Current working directory: " + os.getcwd())
        raise

    return n


def set_fixed_demand(n, config):
    """
    Set fixed electricity demand values for specific countries and years.
    """
    logger.info("=== Setting Fixed Demand Values ===")

    try:
        # Read demand values from CSV file
        demand_df = pd.read_csv(
            config["fixed_demand"]["file"],
            index_col=['country', 'year']
        )

        # Get current year from planning horizon
        current_year = int(snakemake.wildcards.planning_horizons)

        logger.info(f"\nSetting fixed demand for year {current_year}")

        # Process each country in the demand file
        for country in demand_df.index.get_level_values('country').unique():
            try:
                target_demand = demand_df.loc[(country, current_year), 'demand']

                # Get all loads for this country
                loads_mask = (n.loads.bus.map(n.buses.country) == country)
                relevant_loads = n.loads[loads_mask]

                if relevant_loads.empty:
                    logger.warning(f"No loads found for {country}")
                    continue

                # Calculate current yearly demand from time series
                if hasattr(n, 'loads_t') and 'p_set' in n.loads_t:
                    current_demand = (n.loads_t.p_set[relevant_loads.index].sum(
                        axis=1) * n.snapshot_weightings.objective).sum()
                else:
                    current_demand = (relevant_loads.p_set * n.snapshot_weightings.objective.sum()).sum()

                logger.info(f"\nAdjusting demand for {country}:")
                logger.info(f"Target yearly demand: {target_demand:.2f} MWh")
                logger.info(f"Current yearly demand: {current_demand:.2f} MWh")

                # Calculate scaling factor based on yearly totals
                scaling_factor = target_demand / current_demand

                if len(relevant_loads) > 1:
                    # Distribute demand proportionally among loads
                    for idx in relevant_loads.index:
                        # Update time series if it exists
                        if hasattr(n, 'loads_t') and 'p_set' in n.loads_t:
                            old_profile = n.loads_t.p_set[idx].copy()
                            n.loads_t.p_set[idx] = old_profile * scaling_factor

                            # Calculate and log yearly values for this load
                            old_yearly = (old_profile * n.snapshot_weightings.objective).sum()
                            new_yearly = (n.loads_t.p_set[idx] * n.snapshot_weightings.objective).sum()
                            logger.info(f"Load {idx} yearly demand: {old_yearly:.2f} MWh → {new_yearly:.2f} MWh")

                        # Update static value
                        n.loads.loc[idx, 'p_set'] *= scaling_factor
                else:
                    # Set demand for single load
                    idx = relevant_loads.index[0]

                    # Update time series if it exists
                    if hasattr(n, 'loads_t') and 'p_set' in n.loads_t:
                        old_profile = n.loads_t.p_set[idx].copy()
                        n.loads_t.p_set[idx] = old_profile * scaling_factor

                        # Calculate and log yearly values
                        old_yearly = (old_profile * n.snapshot_weightings.objective).sum()
                        new_yearly = (n.loads_t.p_set[idx] * n.snapshot_weightings.objective).sum()
                        logger.info(f"Load {idx} yearly demand: {old_yearly:.2f} MWh → {new_yearly:.2f} MWh")

                    # Update static value
                    n.loads.loc[idx, 'p_set'] *= scaling_factor

                # Verify the final yearly demand
                if hasattr(n, 'loads_t') and 'p_set' in n.loads_t:
                    new_total = (n.loads_t.p_set[relevant_loads.index].sum(
                        axis=1) * n.snapshot_weightings.objective).sum()
                else:
                    new_total = (n.loads.loc[
                                     relevant_loads.index, 'p_set'] * n.snapshot_weightings.objective.sum()).sum()

                logger.info(f"New total yearly demand: {new_total:.2f} MWh")

                if not np.isclose(new_total, target_demand, rtol=1e-3):
                    logger.warning(
                        f"Warning: New yearly demand {new_total:.2f} MWh differs from target {target_demand:.2f} MWh"
                    )

            except KeyError:
                logger.warning(f"No demand value found for {country} in year {current_year}")
                continue

        logger.info("\nFixed demand setting completed successfully")

    except Exception as e:
        logger.error(f"Error in setting fixed demand: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        raise

    return n

def add_EQ_constraints(n, o, scaling=1e-1):
    """
    Add equity constraints to the network.

    Currently this is only implemented for the electricity sector only.

    Opts must be specified in the config.yaml.

    Parameters
    ----------
    n : pypsa.Network
    o : str

    Example
    -------
    scenario:
        opts: [Co2L-EQ0.7-24h]

    Require each country or node to on average produce a minimal share
    of its total electricity consumption itself. Example: EQ0.7c demands each country
    to produce on average at least 70% of its consumption; EQ0.7 demands
    each node to produce on average at least 70% of its consumption.
    """
    # TODO: Generalize to cover myopic and other sectors?
    float_regex = r"[0-9]*\.?[0-9]+"
    level = float(re.findall(float_regex, o)[0])
    if o[-1] == "c":
        ggrouper = n.generators.bus.map(n.buses.country)
        lgrouper = n.loads.bus.map(n.buses.country)
        sgrouper = n.storage_units.bus.map(n.buses.country)
    else:
        ggrouper = n.generators.bus
        lgrouper = n.loads.bus
        sgrouper = n.storage_units.bus
    load = (
        n.snapshot_weightings.generators
        @ n.loads_t.p_set.groupby(lgrouper, axis=1).sum()
    )
    inflow = (
        n.snapshot_weightings.stores
        @ n.storage_units_t.inflow.groupby(sgrouper, axis=1).sum()
    )
    inflow = inflow.reindex(load.index).fillna(0.0)
    rhs = scaling * (level * load - inflow)
    p = n.model["Generator-p"]
    lhs_gen = (
        (p * (n.snapshot_weightings.generators * scaling))
        .groupby(ggrouper.to_xarray())
        .sum()
        .sum("snapshot")
    )
    # TODO: double check that this is really needed, why do have to subtract the spillage
    if not n.storage_units_t.inflow.empty:
        spillage = n.model["StorageUnit-spill"]
        lhs_spill = (
            (spillage * (-n.snapshot_weightings.stores * scaling))
            .groupby(sgrouper.to_xarray())
            .sum()
            .sum("snapshot")
        )
        lhs = lhs_gen + lhs_spill
    else:
        lhs = lhs_gen
    n.model.add_constraints(lhs >= rhs, name="equity_min")


def add_BAU_constraints(n, config):
    """
    Add a per-carrier minimal overall capacity.

    BAU_mincapacities and opts must be adjusted in the config.yaml.

    Parameters
    ----------
    n : pypsa.Network
    config : dict

    Example
    -------
    scenario:
        opts: [Co2L-BAU-24h]
    electricity:
        BAU_mincapacities:
            solar: 0
            onwind: 0
            OCGT: 100000
            offwind-ac: 0
            offwind-dc: 0
    Which sets minimum expansion across all nodes e.g. in Europe to 100GW.
    OCGT bus 1 + OCGT bus 2 + ... > 100000
    """
    mincaps = pd.Series(config["electricity"]["BAU_mincapacities"])
    p_nom = n.model["Generator-p_nom"]
    ext_i = n.generators.query("p_nom_extendable")
    ext_carrier_i = xr.DataArray(ext_i.carrier.rename_axis("Generator-ext"))
    lhs = p_nom.groupby(ext_carrier_i).sum()
    rhs = mincaps[lhs.indexes["carrier"]].rename_axis("carrier")
    n.model.add_constraints(lhs >= rhs, name="bau_mincaps")


# TODO: think about removing or make per country
def add_SAFE_constraints(n, config):
    """
    Add a capacity reserve margin of a certain fraction above the peak demand.
    Renewable generators and storage do not contribute. Ignores network.

    Parameters
    ----------
        n : pypsa.Network
        config : dict

    Example
    -------
    config.yaml requires to specify opts:

    scenario:
        opts: [Co2L-SAFE-24h]
    electricity:
        SAFE_reservemargin: 0.1
    Which sets a reserve margin of 10% above the peak demand.
    """
    peakdemand = n.loads_t.p_set.sum(axis=1).max()
    margin = 1.0 + config["electricity"]["SAFE_reservemargin"]
    reserve_margin = peakdemand * margin
    conventional_carriers = config["electricity"]["conventional_carriers"]  # noqa: F841
    ext_gens_i = n.generators.query(
        "carrier in @conventional_carriers & p_nom_extendable"
    ).index
    p_nom = n.model["Generator-p_nom"].loc[ext_gens_i]
    lhs = p_nom.sum()
    exist_conv_caps = n.generators.query(
        "~p_nom_extendable & carrier in @conventional_carriers"
    ).p_nom.sum()
    rhs = reserve_margin - exist_conv_caps
    n.model.add_constraints(lhs >= rhs, name="safe_mintotalcap")


def add_operational_reserve_margin(n, sns, config):
    """
    Build reserve margin constraints based on the formulation given in
    https://genxproject.github.io/GenX/dev/core/#Reserves.

    Parameters
    ----------
        n : pypsa.Network
        sns: pd.DatetimeIndex
        config : dict

    Example:
    --------
    config.yaml requires to specify operational_reserve:
    operational_reserve: # like https://genxproject.github.io/GenX/dev/core/#Reserves
        activate: true
        epsilon_load: 0.02 # percentage of load at each snapshot
        epsilon_vres: 0.02 # percentage of VRES at each snapshot
        contingency: 400000 # MW
    """
    reserve_config = config["electricity"]["operational_reserve"]
    EPSILON_LOAD = reserve_config["epsilon_load"]
    EPSILON_VRES = reserve_config["epsilon_vres"]
    CONTINGENCY = reserve_config["contingency"]

    # Reserve Variables
    n.model.add_variables(
        0, np.inf, coords=[sns, n.generators.index], name="Generator-r"
    )
    reserve = n.model["Generator-r"]
    summed_reserve = reserve.sum("Generator")

    # Share of extendable renewable capacities
    ext_i = n.generators.query("p_nom_extendable").index
    vres_i = n.generators_t.p_max_pu.columns
    if not ext_i.empty and not vres_i.empty:
        capacity_factor = n.generators_t.p_max_pu[vres_i.intersection(ext_i)]
        p_nom_vres = (
            n.model["Generator-p_nom"]
            .loc[vres_i.intersection(ext_i)]
            .rename({"Generator-ext": "Generator"})
        )
        lhs = summed_reserve + (
            p_nom_vres * (-EPSILON_VRES * xr.DataArray(capacity_factor))
        ).sum("Generator")

    # Total demand per t
    demand = get_as_dense(n, "Load", "p_set").sum(axis=1)

    # VRES potential of non extendable generators
    capacity_factor = n.generators_t.p_max_pu[vres_i.difference(ext_i)]
    renewable_capacity = n.generators.p_nom[vres_i.difference(ext_i)]
    potential = (capacity_factor * renewable_capacity).sum(axis=1)

    # Right-hand-side
    rhs = EPSILON_LOAD * demand + EPSILON_VRES * potential + CONTINGENCY

    n.model.add_constraints(lhs >= rhs, name="reserve_margin")

    # additional constraint that capacity is not exceeded
    gen_i = n.generators.index
    ext_i = n.generators.query("p_nom_extendable").index
    fix_i = n.generators.query("not p_nom_extendable").index

    dispatch = n.model["Generator-p"]
    reserve = n.model["Generator-r"]

    capacity_variable = n.model["Generator-p_nom"].rename(
        {"Generator-ext": "Generator"}
    )
    capacity_fixed = n.generators.p_nom[fix_i]

    p_max_pu = get_as_dense(n, "Generator", "p_max_pu")

    lhs = dispatch + reserve - capacity_variable * xr.DataArray(p_max_pu[ext_i])

    rhs = (p_max_pu[fix_i] * capacity_fixed).reindex(columns=gen_i, fill_value=0)

    n.model.add_constraints(lhs <= rhs, name="Generator-p-reserve-upper")


def add_battery_constraints(n):
    """
    Add constraint ensuring that charger = discharger, i.e.
    1 * charger_size - efficiency * discharger_size = 0
    """
    if not n.links.p_nom_extendable.any():
        return

    discharger_bool = n.links.index.str.contains("battery discharger")
    charger_bool = n.links.index.str.contains("battery charger")

    dischargers_ext = n.links[discharger_bool].query("p_nom_extendable").index
    chargers_ext = n.links[charger_bool].query("p_nom_extendable").index

    eff = n.links.efficiency[dischargers_ext].values
    lhs = (
        n.model["Link-p_nom"].loc[chargers_ext]
        - n.model["Link-p_nom"].loc[dischargers_ext] * eff
    )

    n.model.add_constraints(lhs == 0, name="Link-charger_ratio")


def add_lossy_bidirectional_link_constraints(n):
    if not n.links.p_nom_extendable.any() or "reversed" not in n.links.columns:
        return

    n.links["reversed"] = n.links.reversed.fillna(0).astype(bool)
    carriers = n.links.loc[n.links.reversed, "carrier"].unique()  # noqa: F841

    forward_i = n.links.query(
        "carrier in @carriers and ~reversed and p_nom_extendable"
    ).index

    def get_backward_i(forward_i):
        return pd.Index(
            [
                (
                    re.sub(r"-(\d{4})$", r"-reversed-\1", s)
                    if re.search(r"-\d{4}$", s)
                    else s + "-reversed"
                )
                for s in forward_i
            ]
        )

    backward_i = get_backward_i(forward_i)

    lhs = n.model["Link-p_nom"].loc[backward_i]
    rhs = n.model["Link-p_nom"].loc[forward_i]

    n.model.add_constraints(lhs == rhs, name="Link-bidirectional_sync")


def add_chp_constraints(n):
    electric = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("electric")
    )
    heat = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("heat")
    )

    electric_ext = n.links[electric].query("p_nom_extendable").index
    heat_ext = n.links[heat].query("p_nom_extendable").index

    electric_fix = n.links[electric].query("~p_nom_extendable").index
    heat_fix = n.links[heat].query("~p_nom_extendable").index

    p = n.model["Link-p"]  # dimension: [time, link]

    # output ratio between heat and electricity and top_iso_fuel_line for extendable
    if not electric_ext.empty:
        p_nom = n.model["Link-p_nom"]

        lhs = (
            p_nom.loc[electric_ext]
            * (n.links.p_nom_ratio * n.links.efficiency)[electric_ext].values
            - p_nom.loc[heat_ext] * n.links.efficiency[heat_ext].values
        )
        n.model.add_constraints(lhs == 0, name="chplink-fix_p_nom_ratio")

        rename = {"Link-ext": "Link"}
        lhs = (
            p.loc[:, electric_ext]
            + p.loc[:, heat_ext]
            - p_nom.rename(rename).loc[electric_ext]
        )
        n.model.add_constraints(lhs <= 0, name="chplink-top_iso_fuel_line_ext")

    # top_iso_fuel_line for fixed
    if not electric_fix.empty:
        lhs = p.loc[:, electric_fix] + p.loc[:, heat_fix]
        rhs = n.links.p_nom[electric_fix]
        n.model.add_constraints(lhs <= rhs, name="chplink-top_iso_fuel_line_fix")

    # back-pressure
    if not electric.empty:
        lhs = (
            p.loc[:, heat] * (n.links.efficiency[heat] * n.links.c_b[electric].values)
            - p.loc[:, electric] * n.links.efficiency[electric]
        )
        n.model.add_constraints(lhs <= rhs, name="chplink-backpressure")


def add_pipe_retrofit_constraint(n):
    """
    Add constraint for retrofitting existing CH4 pipelines to H2 pipelines.
    """
    if "reversed" not in n.links.columns:
        n.links["reversed"] = False
    gas_pipes_i = n.links.query(
        "carrier == 'gas pipeline' and p_nom_extendable and ~reversed"
    ).index
    h2_retrofitted_i = n.links.query(
        "carrier == 'H2 pipeline retrofitted' and p_nom_extendable and ~reversed"
    ).index

    if h2_retrofitted_i.empty or gas_pipes_i.empty:
        return

    p_nom = n.model["Link-p_nom"]

    CH4_per_H2 = 1 / n.config["sector"]["H2_retrofit_capacity_per_CH4"]
    lhs = p_nom.loc[gas_pipes_i] + CH4_per_H2 * p_nom.loc[h2_retrofitted_i]
    rhs = n.links.p_nom[gas_pipes_i].rename_axis("Link-ext")

    n.model.add_constraints(lhs == rhs, name="Link-pipe_retrofit")


def add_flexible_egs_constraint(n):
    """
    Upper bounds the charging capacity of the geothermal reservoir according to
    the well capacity.
    """
    well_index = n.links.loc[n.links.carrier == "geothermal heat"].index
    storage_index = n.storage_units.loc[
        n.storage_units.carrier == "geothermal heat"
    ].index

    p_nom_rhs = n.model["Link-p_nom"].loc[well_index]
    p_nom_lhs = n.model["StorageUnit-p_nom"].loc[storage_index]

    n.model.add_constraints(
        p_nom_lhs <= p_nom_rhs,
        name="upper_bound_charging_capacity_of_geothermal_reservoir",
    )


def add_co2_atmosphere_constraint(n, snapshots):
    glcs = n.global_constraints[n.global_constraints.type == "co2_atmosphere"]

    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]

        if emissions.empty:
            continue

        # stores
        bus_carrier = n.stores.bus.map(n.buses.carrier)
        stores = n.stores[bus_carrier.isin(emissions.index) & ~n.stores.e_cyclic]
        if not stores.empty:
            last_i = snapshots[-1]
            lhs = n.model["Store-e"].loc[last_i, stores.index]
            rhs = glc.constant

            n.model.add_constraints(lhs <= rhs, name=f"GlobalConstraint-{name}")


def check_generators(n):
    """
    Check if coal and lignite units are in generators instead of links.
    """
    logging.info("\n====== CHECKING GENERATORS ======")

    # Print all unique generator carriers
    logging.info("\nAll generator carriers:")
    logging.info(n.generators.carrier.unique())

    # Check for coal and lignite generators
    coal_gens = n.generators[n.generators.carrier.str.contains('coal', case=False, na=False)]
    lignite_gens = n.generators[n.generators.carrier.str.contains('lignite', case=False, na=False)]

    if not coal_gens.empty:
        logging.info("\nFound coal generators:")
        logging.info(coal_gens[['carrier', 'bus', 'p_nom', 'p_nom_extendable']].to_string())

    if not lignite_gens.empty:
        logging.info("\nFound lignite generators:")
        logging.info(lignite_gens[['carrier', 'bus', 'p_nom', 'p_nom_extendable']].to_string())


def extra_functionality(n, snapshots):
    """
    Collects supplementary constraints which will be passed to pypsa.optimization.optimize.
    """
    config = n.config
    constraints = config["solving"].get("constraints", {})

    # Add fixed demand setting if enabled
    if config["solving"].get("fixed_demand", {}).get("activate", False):
        set_fixed_demand(n, config)

    if constraints["BAU"] and n.generators.p_nom_extendable.any():
        add_BAU_constraints(n, config)
    if constraints["SAFE"] and n.generators.p_nom_extendable.any():
        add_SAFE_constraints(n, config)
    if constraints["CCL"] and n.generators.p_nom_extendable.any():
        add_CCL_constraints(n, config)

    # Add CLL constraints
    add_CLL_constraints(n, config)

    # Print capacities
    logging.info("\n====== CURRENT CAPACITIES ======")
    de_buses = n.buses[n.buses.country == 'DE'].index

    # Calculate total capacities
    for carrier in ['coal', 'lignite']:
        total_cap = n.links[
            (n.links.carrier == carrier) &
            (n.links.bus0 == f'EU {carrier}') &
            (n.links.bus1.isin(de_buses))
            ].p_nom.sum()

        logging.info(f"DE {carrier} total capacity: {total_cap:.2f} MW")

    # Rest of your original extra_functionality code...
    reserve = config["electricity"].get("operational_reserve", {})
    if reserve.get("activate"):
        add_operational_reserve_margin(n, snapshots, config)

    if EQ_o := constraints["EQ"]:
        add_EQ_constraints(n, EQ_o.replace("EQ", ""))


    if {"solar-hsat", "solar"}.issubset(
            config["electricity"]["renewable_carriers"]
    ) and {"solar-hsat", "solar"}.issubset(
        config["electricity"]["extendable_carriers"]["Generator"]
    ):
        add_solar_potential_constraints(n, config)

    add_battery_constraints(n)
    add_lossy_bidirectional_link_constraints(n)
    add_pipe_retrofit_constraint(n)
    if n._multi_invest:
        add_carbon_constraint(n, snapshots)
        add_carbon_budget_constraint(n, snapshots)
        add_retrofit_gas_boiler_constraint(n, snapshots)
    else:
        add_co2_atmosphere_constraint(n, snapshots)

    if config["sector"]["enhanced_geothermal"]["enable"]:
        add_flexible_egs_constraint(n)

    if n.params.custom_extra_functionality:
        source_path = n.params.custom_extra_functionality
        assert os.path.exists(source_path), f"{source_path} does not exist"
        sys.path.append(os.path.dirname(source_path))
        module_name = os.path.splitext(os.path.basename(source_path))[0]
        module = importlib.import_module(module_name)
        custom_extra_functionality = getattr(module, module_name)
        custom_extra_functionality(n, snapshots, snakemake)


def check_objective_value(n, solving):
    check_objective = solving["check_objective"]
    if check_objective["enable"]:
        atol = check_objective["atol"]
        rtol = check_objective["rtol"]
        expected_value = check_objective["expected_value"]
        if not np.isclose(n.objective, expected_value, atol=atol, rtol=rtol):
            raise ObjectiveValueError(
                f"Objective value {n.objective} differs from expected value "
                f"{expected_value} by more than {atol}."
            )


def check_constraint_feasibility(n, config):
    """
    Check and adjust CCL constraints before solving.
    """
    logger.info("=== Checking Constraint Feasibility ===")

    try:
        # Read the CSV file with multi-level columns
        constraints_df = pd.read_csv(
            config["solving"]["agg_p_nom_limits"]["file"],
            header=[0, 1],  # Read first two rows as multi-level header
            index_col=[0, 1]  # First two columns are multi-level index
        )

        # Get the correct planning horizon
        planning_horizon = snakemake.wildcards.planning_horizons

        logger.info("Constraints DataFrame columns:")
        logger.info(constraints_df.columns.tolist())

        # Access the min and max columns for the specific year
        agg_p_nom_minmax = pd.DataFrame({
            'min': constraints_df[(planning_horizon, 'min')],
            'max': constraints_df[(planning_horizon, 'max')]
        })

        modified_constraints = False

        for (country, carrier), row in agg_p_nom_minmax.iterrows():
            min_val = row['min']
            max_val = row['max']

            # Get relevant generators
            relevant_gens = n.generators[
                (n.generators.carrier == carrier) &
                (n.generators.bus.map(n.buses.country) == country)
                ]

            if relevant_gens.empty:
                logger.warning(f"No generators found for {carrier} in {country}")
                continue

            # Calculate capacities
            existing_cap = relevant_gens.p_nom.sum()
            extendable_gens = relevant_gens[relevant_gens.p_nom_extendable]
            extendable_potential = extendable_gens.p_nom_max.sum()
            total_possible = existing_cap + extendable_potential

            logger.info(f"\nAnalyzing {country} {carrier}:")
            logger.info(f"Required minimum: {min_val:.2f} MW")
            logger.info(f"Maximum allowed: {max_val:.2f} MW")
            logger.info(f"Existing capacity: {existing_cap:.2f} MW")
            logger.info(f"Additional potential: {extendable_potential:.2f} MW")
            logger.info(f"Maximum possible: {total_possible:.2f} MW")

            # Check feasibility and adjust if needed
            if min_val > total_possible:
                logger.warning(f"Minimum requirement exceeds maximum possible capacity")
                logger.warning(f"Adjusting minimum from {min_val:.2f} to {total_possible:.2f} MW")
                agg_p_nom_minmax.loc[(country, carrier), 'min'] = total_possible
                modified_constraints = True

            if max_val < existing_cap:
                logger.warning(f"Maximum limit is below existing capacity")
                logger.warning(f"Adjusting maximum from {max_val:.2f} to {existing_cap:.2f} MW")
                agg_p_nom_minmax.loc[(country, carrier), 'max'] = existing_cap
                modified_constraints = True

            # Special handling for exact constraints (min == max)
            if min_val == max_val and min_val > total_possible:
                new_val = total_possible
                logger.warning(f"Exact capacity requirement {min_val:.2f} MW is infeasible")
                logger.warning(f"Adjusting both min and max to maximum possible: {new_val:.2f} MW")
                agg_p_nom_minmax.loc[(country, carrier), 'min'] = new_val
                agg_p_nom_minmax.loc[(country, carrier), 'max'] = new_val
                modified_constraints = True

        if modified_constraints:
            # Update the original DataFrame with modified values
            constraints_df.loc[:, (planning_horizon, 'min')] = agg_p_nom_minmax['min']
            constraints_df.loc[:, (planning_horizon, 'max')] = agg_p_nom_minmax['max']

            # Save modified constraints
            output_file = config["solving"]["agg_p_nom_limits"]["file"]
            logger.info(f"\nSaving adjusted constraints to {output_file}")
            constraints_df.to_csv(output_file)
            logger.info("Constraints were adjusted to ensure feasibility")
        else:
            logger.info("\nAll constraints appear feasible")

        return True

    except Exception as e:
        logger.error(f"Error in constraint feasibility check: {str(e)}")
        logger.error(f"DataFrame columns: {constraints_df.columns.tolist()}")
        logger.error("Traceback:", exc_info=True)
        raise

def solve_network(n, config, params, solving, **kwargs):
    """
    Solve network optimization with feasibility checks and proper constraint handling.
    """
    set_of_options = solving["solver"]["options"]
    cf_solving = solving["options"]
    constraints = config["solving"].get("constraints", {})

    # Check and adjust constraints before solving if CCL is enabled
    if constraints.get("CCL") and n.generators.p_nom_extendable.any():
        check_constraint_feasibility(n, config)

    kwargs["multi_investment_periods"] = config["foresight"] == "perfect"
    kwargs["solver_options"] = (
        solving["solver_options"][set_of_options] if set_of_options else {}
    )
    kwargs["solver_name"] = solving["solver"]["name"]
    kwargs["extra_functionality"] = extra_functionality
    kwargs["transmission_losses"] = cf_solving.get("transmission_losses", False)
    kwargs["linearized_unit_commitment"] = cf_solving.get(
        "linearized_unit_commitment", False
    )
    kwargs["assign_all_duals"] = cf_solving.get("assign_all_duals", False)
    kwargs["io_api"] = cf_solving.get("io_api", None)

    if kwargs["solver_name"] == "gurobi":
        logging.getLogger("gurobipy").setLevel(logging.CRITICAL)



    rolling_horizon = cf_solving.pop("rolling_horizon", False)
    skip_iterations = cf_solving.pop("skip_iterations", False)
    if not n.lines.s_nom_extendable.any():
        skip_iterations = True
        logger.info("No expandable lines found. Skipping iterative solving.")

    # add to network for extra_functionality
    n.config = config
    n.params = params

    if rolling_horizon and snakemake.rule == "solve_operations_network":
        kwargs["horizon"] = cf_solving.get("horizon", 365)
        kwargs["overlap"] = cf_solving.get("overlap", 0)
        n.optimize.optimize_with_rolling_horizon(**kwargs)
        status, condition = "", ""
    elif skip_iterations:
        status, condition = n.optimize(**kwargs)
    else:
        kwargs["track_iterations"] = cf_solving["track_iterations"]
        kwargs["min_iterations"] = cf_solving["min_iterations"]
        kwargs["max_iterations"] = cf_solving["max_iterations"]
        if cf_solving["post_discretization"].pop("enable"):
            logger.info("Add post-discretization parameters.")
            kwargs.update(cf_solving["post_discretization"])
        status, condition = n.optimize.optimize_transmission_expansion_iteratively(
            **kwargs
        )

    if not rolling_horizon:
        if status != "ok":
            logger.warning(
                f"Solving status '{status}' with termination condition '{condition}'"
            )
        check_objective_value(n, solving)

    if "infeasible" in condition:
        labels = n.model.compute_infeasibilities()
        logger.info(f"Labels:\n{labels}")
        n.model.print_infeasibilities()
        raise RuntimeError("Solving status 'infeasible'")

    return n


# %%
if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_sector_network_perfect",
            configfiles="../config/test/config.perfect.yaml",
            opts="",
            clusters="5",
            ll="v1.0",
            sector_opts="",
            # planning_horizons="2030",
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    solve_opts = snakemake.params.solving["options"]

    np.random.seed(solve_opts.get("seed", 123))

    n = pypsa.Network(snakemake.input.network)

    n = prepare_network(
        n,
        solve_opts,
        config=snakemake.config,
        foresight=snakemake.params.foresight,
        planning_horizons=snakemake.params.planning_horizons,
        co2_sequestration_potential=snakemake.params["co2_sequestration_potential"],
    )

    with memory_logger(
        filename=getattr(snakemake.log, "memory", None), interval=30.0
    ) as mem:
        n = solve_network(
            n,
            config=snakemake.config,
            params=snakemake.params,
            solving=snakemake.params.solving,
            log_fn=snakemake.log.solver,
        )

    logger.info(f"Maximum memory usage: {mem.mem_usage}")

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output.network)

    with open(snakemake.output.config, "w") as file:
        yaml.dump(
            n.meta,
            file,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
