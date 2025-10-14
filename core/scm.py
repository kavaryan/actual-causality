# simple_actual_cauality.py

import ast
import random
from pathlib import Path
import networkx as nx
from sympy import sympify, lambdify, symbols, Derivative
import math
import matplotlib.pyplot as plt
import numpy as np
import re

def is_literal(s):
    try:
        ast.literal_eval(s)
        return True
    except (ValueError, SyntaxError):
        return False


class Component:
    """
    Represents a single component in the SCM.
    
    Each component is defined by an equation of the form:
        Output = Expression
    For example: "B=sin(A)"
    """
    def __init__(self, definition_str):
        # Split the definition at '='.        
        parts = definition_str.split('=')
        if len(parts) != 2:
            raise ValueError("Component definition must be of the form 'Output = Expression'")
        lhs, rhs = parts[0].strip(), parts[1].strip()
        self.output = lhs  # The output variable of this component.
        
        # Parse the right-hand side using sympy.
        # This allows symbolic manipulation within the component.
        self.expression = sympify(rhs, evaluate=False)
        
        # Determine input variables by extracting the free symbols from the expression.
        # These inputs will be manually tied to the outputs of other components or exogenous variables.
        self.input_vars = [str(s) for s in self.expression.free_symbols]
        
        # Create a callable function for evaluation using lambdify.
        # The function expects its arguments in the order given by self.input_vars.
        self.func = lambdify(self.input_vars, self.expression, modules=["math"])

        self.is_literal = is_literal(rhs) 
    

    def compute(self, inputs):
        """
        Compute the output of the component given a dictionary of input values.
        """
        try:
            # Retrieve the required inputs in the order of self.input_vars.
            args = [inputs[var] for var in self.input_vars]
        except KeyError as e:
            raise ValueError(f"Missing input variable: {e}")
        return self.func(*args)
    
class BoundedIntInterval:
    def __init__(self, a, b):
        """ [a,a+1,...,b] """
        self.a = a
        self.b = b
        self.all = list(range(self.a,self.b+1))
    
    def __eq__(self, other):
        return isinstance(other, BoundedIntInterval) and (self.a, self.b) == (other.a, other.b)
    
    # def get_gen(self):
    #     for x in range(self.a,self.b+1):
    #         yield x

class BoundedFloatInterval:
    def __init__(self, a, b, num_points=1):
        """ [a, ..., b] with num_points evenly spaced values """
        self.a = a
        self.b = b
        self.num_points = num_points
        self.all = list(np.linspace(a, b, num_points))
    
    def __eq__(self, other):
        return isinstance(other, BoundedFloatInterval) and (self.a, self.b, self.num_points) == (other.a, other.b, other.num_points)

class DifferentialComponent:
    """
    Represents a differential equation component in the form dX/dt = expression.
    """
    def __init__(self, definition_str):
        # Parse differential equation like "dX/dt = -X + Y"
        parts = definition_str.split('=')
        if len(parts) != 2:
            raise ValueError("Differential component definition must be of the form 'dX/dt = Expression'")
        
        lhs, rhs = parts[0].strip(), parts[1].strip()
        
        # Extract variable name from dX/dt
        match = re.match(r'd([A-Za-z_][A-Za-z0-9_]*)/dt', lhs)
        if not match:
            raise ValueError("Left-hand side must be in the form 'dX/dt'")
        
        self.variable = match.group(1)  # The variable being differentiated
        self.expression = sympify(rhs, evaluate=False)
        self.input_vars = [str(s) for s in self.expression.free_symbols]
        self.func = lambdify(self.input_vars, self.expression, modules=["math"])

    def compute(self, inputs):
        """Compute the derivative value given input values."""
        try:
            args = [inputs[var] for var in self.input_vars]
        except KeyError as e:
            raise ValueError(f"Missing input variable: {e}")
        return self.func(*args)


class TemporalSCMSystem:
    """
    Represents a system with differential equations that can be expanded temporally.
    """
    def __init__(self, components, differential_components, domains, temporal_expansion_window_width=1, delta=0.1):
        self.components = {comp.output: comp for comp in components}
        self.differential_components = {comp.variable: comp for comp in differential_components}
        self.domains = domains
        self.temporal_expansion_window_width = temporal_expansion_window_width
        self.delta = delta
    
    def expand_to_scm(self):
        """
        Expand the temporal system into a regular SCM by discretizing differential equations.
        Returns a regular SCMSystem.
        """
        expanded_components = []
        expanded_domains = {}
        
        # Copy existing non-differential components
        for comp in self.components.values():
            expanded_components.append(comp)
        
        # Copy domains
        expanded_domains.update(self.domains)
        
        # For each differential variable, create time-indexed versions
        for var_name, diff_comp in self.differential_components.items():
            # Create initial condition variable (X_0)
            initial_var = f"{var_name}_0"
            if var_name in self.domains:
                expanded_domains[initial_var] = self.domains[var_name]
            
            # Create time-stepped variables using Euler's method
            for t in range(1, self.temporal_expansion_window_width + 1):
                current_var = f"{var_name}_{t}"
                prev_var = f"{var_name}_{t-1}"
                
                # Build the equation: X_t = X_{t-1} + delta * (derivative expression at t-1)
                # Replace variables in the derivative expression with their time-indexed versions
                expr_str = str(diff_comp.expression)
                
                # Replace each input variable with its time-indexed version
                for input_var in diff_comp.input_vars:
                    if input_var == var_name:
                        # Self-reference: use previous time step
                        expr_str = re.sub(r'\b' + re.escape(input_var) + r'\b', f"{input_var}_{t-1}", expr_str)
                    else:
                        # Other variables: use previous time step if they're differential, current if not
                        if input_var in self.differential_components:
                            expr_str = re.sub(r'\b' + re.escape(input_var) + r'\b', f"{input_var}_{t-1}", expr_str)
                        else:
                            # Non-differential variables don't have time indices
                            pass
                
                # Create the Euler step equation
                euler_equation = f"{current_var} = {prev_var} + {self.delta} * ({expr_str})"
                expanded_components.append(Component(euler_equation))
                
                # Set domain for the new variable
                if var_name in self.domains:
                    expanded_domains[current_var] = self.domains[var_name]
        
        return SCMSystem(expanded_components, expanded_domains)


class SCMSystem:
    """
    Represents the overall Structural Causal Model (SCM) system.
    
    The system is built from multiple components and maintains a networkx DAG
    where an edge from variable X to variable Y indicates that Y depends on X.
    """
    def __init__(self, components, domains):
        # components: list of Component objects.
        self.components = {comp.output: comp for comp in components}
        self.domains = domains
        self.build_dag()

    @property
    def exogenous_vars(self): # not on lhs of any equation or is a literal (constant) (the context can be used for changing exogenous vars)
        """Getter method for value (read-only property)."""
        ans = [node for node in self.dag.nodes() if node not in self.components or self.components[node].is_literal]
        return ans
    
    @property
    def exogenous_nl_vars(self): # not on lhs of any equation
        """Getter method for value (read-only property)."""
        ans = [node for node in self.dag.nodes() if node not in self.components]
        return ans
    
    @property
    def vars(self):
        """Getter method for value (read-only property)."""
        ans = list(self.dag.nodes())
        return ans
    
    @property
    def endogenous_vars(self): # lhs of a non-constant equation
        """Getter method for value (read-only property)."""
        ans = list(set(self.vars) - set(self.exogenous_vars))
        return ans
    

    def get_random_context(self):
        ans = {}
        for v in self.exogenous_nl_vars:
            d = self.domains[v]
            if isinstance(d, BoundedIntInterval):
                ans[v] = random.randint(d.a, d.b)
            elif isinstance(d, BoundedFloatInterval):
                ans[v] = random.uniform(d.a, d.b)
            else:
                ValueError(f'get_random_context: Domain of type {type(d)} not supported.')
        return ans

    def build_dag(self):
        """
        Build the DAG from the component definitions.
        Nodes are all variables (both outputs and the free symbols appearing as inputs),
        and an edge is added from each input variable to the component’s output.
        """
        self.dag = nx.DiGraph()
        # Collect all variables.
        all_vars = set()
        for comp in self.components.values():
            all_vars.add(comp.output)
            all_vars.update(comp.input_vars)
        for var in all_vars:
            self.dag.add_node(var)
        # Create edges: for each component, add an edge from each input to its output.
        for comp in self.components.values():
            for var in comp.input_vars:
                self.dag.add_edge(var, comp.output)
        # Ensure the resulting graph is a DAG.
        if not nx.is_directed_acyclic_graph(self.dag):
            raise ValueError("The system configuration must be a Directed Acyclic Graph (DAG).")
    
    def get_state(self, context: dict[str,object], interventions: dict[str, object] = None):
        """
        Compute the state of the system given a dictionary of exogenous variable values.
        
        The system is evaluated in topologically sorted order so that each component
        is computed only after all its inputs are available.
        """
        state = context.copy()  # Start with exogenous inputs.
        if interventions:
            state.update(interventions)
        # Process all variables in a topological order.
        for var in nx.topological_sort(self.dag):
            # Skip variables already defined (i.e. exogenous inputs).
            if var in state:
                continue
            # If the variable is computed by a component, compute its value.
            if var in self.components:
                comp = self.components[var]
                # Ensure all the component’s inputs are available.
                if not all(input_var in state for input_var in comp.input_vars):
                    missing = [iv for iv in comp.input_vars if iv not in state]
                    raise ValueError(f"Missing inputs for {var}: {missing}")
                state[var] = comp.compute(state)
        return state
    
    def replace(self, new_definition):
        """
        Replace a component in the system with a new definition.
        
        new_definition: A string such as "B=sin(A)".
        Returns a new SCMSystem instance with the updated component.
        """
        new_comp = Component(new_definition)
        new_components = self.components.copy()
        new_components[new_comp.output] = new_comp
        return SCMSystem(list(new_components.values()), self.domains)
    
    def display_dag(self, plot_name=None):
        """
        Display the DAG of an SCMSystem with a left-to-right layout.
        
        Exogenous variables are displayed with a different style.
        If not provided, exogenous nodes are assumed to be those nodes 
        that are not outputs of any component in the system.
        """
        # Attempt to use graphviz_layout for a left-to-right (LR) layout.
        try:
            # pos = nx.nx_pydot.graphviz_layout(self.dag, prog='dot', args='-Grankdir=LR')
            pos = nx.nx_agraph.graphviz_layout(self.dag, prog="dot", args='-Grankdir=LR')
        except Exception as e:
            # Fallback to a spring layout if graphviz layout is not available.
            pos = nx.spring_layout(self.dag)
            print("Graphviz layout not available; using spring layout instead.")

        # Set up a compact figure.
        plt.figure(figsize=(len(self.dag)/3, len(self.dag)/3))

        # Draw exogenous nodes with a distinct style (e.g., light blue squares).
        base_size = 300
        node_sizes = [base_size * len(label) for label in self.exogenous_vars]
        node_color = ['lightblue' if label in self.components else 'pink' for label in self.exogenous_vars]
        nx.draw_networkx_nodes(self.dag, pos,
                            nodelist=self.exogenous_vars,
                            node_size=node_sizes,
                            node_color=node_color,
                            node_shape='s',
                            #node_size=800,
                            label='Exogenous')
        
        # Draw computed (endogenous) nodes with a different style (e.g., light green circles).
        node_sizes = [base_size * len(label) for label in self.endogenous_vars]
        nx.draw_networkx_nodes(self.dag, pos,
                            nodelist=self.endogenous_vars,
                            node_size=node_sizes,
                            node_color='lightgreen',
                            node_shape='o',
                            #node_size=800,
                            label='Endogenous')

        # Draw the edges with arrows.
        nx.draw_networkx_edges(self.dag, pos, arrows=True, arrowsize=20, edge_color="black")
        # Add labels for clarity.
        nx.draw_networkx_labels(self.dag, pos, font_size=10)

        # Add a legend to distinguish node types.
        plt.legend(scatterpoints=1, loc='upper left', bbox_to_anchor=(2, 2))
        plt.title("SCM System DAG")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        if plot_name:
            plt.savefig(plot_name, bbox_inches='tight')


def read_system(config_file):
    return read_system_str(Path(config_file).read_text())

def read_system_str(config_str):
    """
    Read a configuration file and construct an SCMSystem.
    Reads both component definitions (equations) and domain constraints.
    """
    components = []
    domains = {}
    section = None  # Track whether we're in [equations] or [domains]
    
    for line in config_str.splitlines():
        # Remove comments and strip whitespace
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        
        # Detect sections
        if line == "[equations]":
            section = "equations"
            continue
        elif line == "[domains]":
            section = "domains"
            continue
        
        # Parse equations
        if section == "equations":
            components.append(Component(line))
        
        # Parse domains
        elif section == "domains":
            # Try to match Int domain
            match = re.match(r"([A-Za-z0-9_, ]+):\s*Int\((\d+),\s*(\d+)\)", line)
            if match:
                variables = [var.strip() for var in match.group(1).split(',')]
                lower_bound, upper_bound = int(match.group(2)), int(match.group(3))
                interval = BoundedIntInterval(lower_bound, upper_bound)
                
                for var in variables:
                    domains[var] = interval
            else:
                # Try to match Float domain
                match = re.match(r"([A-Za-z0-9_, ]+):\s*Float\(([+-]?\d*\.?\d+),\s*([+-]?\d*\.?\d+)\)", line)
                if match:
                    variables = [var.strip() for var in match.group(1).split(',')]
                    lower_bound, upper_bound = float(match.group(2)), float(match.group(3))
                    interval = BoundedFloatInterval(lower_bound, upper_bound)
                    
                    for var in variables:
                        domains[var] = interval

    return SCMSystem(components, domains)

def read_temporal_system_str(config_str, temporal_expansion_window_width=1, delta=0.1):
    """
    Read a configuration file and construct a TemporalSCMSystem.
    Supports both regular equations and differential equations.
    """
    components = []
    differential_components = []
    domains = {}
    section = None
    
    for line in config_str.splitlines():
        # Remove comments and strip whitespace
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        
        # Detect sections
        if line == "[equations]":
            section = "equations"
            continue
        elif line == "[differential_equations]":
            section = "differential_equations"
            continue
        elif line == "[domains]":
            section = "domains"
            continue
        elif line == "[temporal_params]":
            section = "temporal_params"
            continue
        
        # Parse equations
        if section == "equations":
            components.append(Component(line))
        
        # Parse differential equations
        elif section == "differential_equations":
            differential_components.append(DifferentialComponent(line))
        
        # Parse temporal parameters
        elif section == "temporal_params":
            if line.startswith("window_width"):
                temporal_expansion_window_width = int(line.split('=')[1].strip())
            elif line.startswith("delta"):
                delta = float(line.split('=')[1].strip())
        
        # Parse domains
        elif section == "domains":
            # Try to match Int domain
            match = re.match(r"([A-Za-z0-9_, ]+):\s*Int\((\d+),\s*(\d+)\)", line)
            if match:
                variables = [var.strip() for var in match.group(1).split(',')]
                lower_bound, upper_bound = int(match.group(2)), int(match.group(3))
                interval = BoundedIntInterval(lower_bound, upper_bound)
                
                for var in variables:
                    domains[var] = interval
            else:
                # Try to match Float domain
                match = re.match(r"([A-Za-z0-9_, ]+):\s*Float\(([+-]?\d*\.?\d+),\s*([+-]?\d*\.?\d+)\)", line)
                if match:
                    variables = [var.strip() for var in match.group(1).split(',')]
                    lower_bound, upper_bound = float(match.group(2)), float(match.group(3))
                    interval = BoundedFloatInterval(lower_bound, upper_bound)
                    
                    for var in variables:
                        domains[var] = interval

    return TemporalSCMSystem(components, differential_components, domains, 
                           temporal_expansion_window_width, delta)

