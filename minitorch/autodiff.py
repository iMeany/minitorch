from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """

    val_list = list(vals)

    # calculate the function near the arg with +/- epsilon
    # plus
    val_list[arg] += epsilon
    f_plus = f(*val_list)
    # minus
    val_list[arg] -= 2 * epsilon
    f_minus = f(*val_list)

    # slope
    return (f_plus - f_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """

    # https://en.wikipedia.org/wiki/Topological_sorting
    order = []
    seen = set()

    def visit(node: Variable):
        # skip constant or seen
        if node.is_constant() or node in seen:
            return
        # visit parents for non-leaf nodes
        if not node.is_leaf():
            for parent in node.parents:
                visit(parent)
        # mark as seen and add to order
        seen.add(node)
        order.append(node)

    # start from the right-most variable
    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # https://minitorch.github.io/module1/backpropagate/#algorithm
    
    sorted_nodes = topological_sort(variable)
    current_derivatives = {variable.unique_id: deriv}

    # iterate in reverse topological order
    for node in reversed(sorted_nodes):
        if node.unique_id not in current_derivatives:
            continue

        # get current derivative
        current_derivative = current_derivatives[node.unique_id]
        # get chain rule results
        chain_rule_results = node.chain_rule(current_derivative)

        # update derivatives for parents
        for parent, parent_derivative in chain_rule_results:
            if parent.is_leaf():
                parent.accumulate_derivative(parent_derivative)
            else:
                if parent.unique_id in current_derivatives:
                    current_derivatives[parent.unique_id] += parent_derivative
                else:
                    current_derivatives[parent.unique_id] = parent_derivative


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
