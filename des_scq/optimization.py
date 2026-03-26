"""
des_scq.optimization
====================
Gradient-descent optimizer for superconducting circuit parameter search.

The :class:`Optimization` class wraps a :class:`~des_scq.circuit.Circuit`
instance, a control manifold, and a loss function into a self-contained
optimization loop.  Because the entire Hamiltonian assembly pipeline is
built on PyTorch, gradients flow automatically from the scalar loss value
back through the eigenenergies to the circuit element parameters.

Workflow
--------
1. Instantiate a circuit (e.g., via ``models.zeroPi``).
2. Define a control manifold — a list of control vectors over which the
   spectrum is evaluated.
3. Choose or define a loss function (see :mod:`des_scq.discovery`).
4. Create an ``Optimization`` object and call ``optimization()``.
5. Inspect the returned ``(dLogs, dParams, dCircuit)`` DataFrames.

External parameters
-------------------
In addition to the circuit's own element parameters, you can register extra
:class:`~des_scq.components.Control` objects via ``externalParameter``.
This is useful for optimizing gate voltages or other non-circuit degrees of
freedom alongside the circuit parameters.

Logging
-------
By default the optimizer logs the loss value and any metrics returned by
the loss function at every iteration.  Additional logging of gradients,
the eigenspectrum, and the Hessian can be enabled via the boolean flags
``log_grad``, ``log_spectrum``, ``log_hessian``.
"""

from numpy import array, isnan
from time import perf_counter
import pandas
from torch.optim import Adam, LBFGS
from torch import tensor
from des_scq.utils import empty
from des_scq.components import Control
from des_scq.circuit import Circuit
from inspect import signature


class Optimization:
    """Gradient-based optimizer for circuit spectral properties.

    Parameters
    ----------
    circuit : Circuit
        A fully initialized circuit instance (``Charge`` or ``Kerman``).
    control_profile : list
        List of control vectors defining the flux/charge manifold over which
        the spectrum is evaluated at each optimization step.  Each element is
        passed as a single control point to ``circuit.spectrumManifold``.
        For circuits with no external control, use ``[dict()]`` or ``[[]]``.
    loss_function : callable
        Signature ``loss_fn(Spectrum, flux_profile) → (loss_tensor, metrics_dict)``
        where ``Spectrum`` is the list of ``(eigenenergies, states)`` tuples
        returned by ``spectrumManifold``, and ``metrics_dict`` is a
        ``{str: float}`` dict of auxiliary values to log.
    external : list of Control, optional
        Additional free parameters beyond the circuit's own elements.
        They are appended to the optimizer's parameter list and logged
        alongside circuit parameters.  Default ``[]``.

    Attributes
    ----------
    parameters : list of Tensor
        All optimizable ``base`` tensors (circuit elements + externals).
    IDs : list of str
        Identifiers corresponding to each entry in ``parameters``.
    Bounds : list of (Tensor, Tensor)
        (lower, upper) bound pairs for each parameter.
    logs : list of dict
        Per-iteration metric dictionaries (populated during optimization).
    vectors_calc : bool
        Whether eigenvectors are computed (currently unused).  Default ``False``.
    grad_calc : bool
        Whether gradients are tracked.  Default ``True``.
    log_grad : bool
        Log parameter gradients at every step.  Default ``False``.
    log_spectrum : bool
        Log eigenenergies at flux = 0 and flux = π at every step.
        Default ``False``.
    log_hessian : bool
        Log the Hessian (not yet implemented).  Default ``False``.

    Examples
    --------
    Basic optimization loop::

        from des_scq import models
        from des_scq.circuit import Charge
        from des_scq.optimization import Optimization
        from des_scq.discovery import lossTransition
        from torch import tensor, float64 as double

        circuit = models.transmon(Charge, [256])
        flux_profile = [dict()]   # no external flux
        E10_target = tensor([5.0], dtype=double)
        E21_target = tensor([4.8], dtype=double)
        loss_fn    = lossTransition(E10_target, E21_target)

        optim = Optimization(circuit, flux_profile, loss_fn)
        optim.initAlgo(lr=0.01)
        dLogs, dParams, dCircuit = optim.optimization(iterations=200)

        print(dCircuit.iloc[-1])   # final element energies (GHz)
    """

    def __init__(self, circuit: Circuit, control_profile=[], loss_function=None,
                 external=[]):
        self.circuit         = circuit
        self.control_profile = control_profile
        self.loss_function   = loss_function
        # Determine whether the loss function accepts external parameters
        self.loss_arg_count  = len(signature(self.loss_function).parameters)
        self.initialization(external=external)
        self.vectors_calc = False
        self.grad_calc    = True
        self.log_grad     = False
        self.log_spectrum = False
        self.log_hessian  = False
        self.iteration    = 0
        self.initAlgo()

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------

    def initialization(self, parameters=dict(), external=[]):
        """(Re-)initialize the optimizer state from a parameter dictionary.

        Rebuilds the parameter list and bounds after optionally updating the
        circuit with new element values.

        Parameters
        ----------
        parameters : dict {str: float}, optional
            If non-empty, calls ``circuit.initialization(parameters)`` first
            to update element energies.
        external : list of Control, optional
            External free parameters to register.
        """
        if len(parameters) > 0:
            self.circuit.initialization(parameters)
        self.parameters, self.IDs = self.circuitParameters()
        self.Bounds    = self.parameterBounds()
        self.external  = []
        if len(external) > 0:
            assert self.loss_arg_count > 2
            for parameter in external:
                self.externalParameter(parameter)

    def logInit(self):
        """Initialize per-optimization-run logging containers."""
        self.logs    = []
        self.dParams = [self.parameterState()]
        self.dCircuit= [self.circuitState()]

    def logCompile(self):
        """Compile logged data into pandas DataFrames.

        Returns
        -------
        dLog : DataFrame
            Per-iteration metrics (loss, custom metrics, gradients if enabled,
            wall-clock time).  The ``'time'`` column is differenced to give
            per-iteration elapsed time.
        dParams : DataFrame
            Per-iteration unconstrained ``base`` tensor values.
        dCircuit : DataFrame
            Per-iteration element energies in GHz.
        """
        dLog = pandas.DataFrame(self.logs)
        if len(dLog) > 0:
            dLog['time'] = dLog['time'].diff()
        else:
            print('Failure initial')
        dParams  = pandas.DataFrame(self.dParams)
        dCircuit = pandas.DataFrame(self.dCircuit)
        return dLog, dParams, dCircuit

    # -----------------------------------------------------------------------
    # Parameter management
    # -----------------------------------------------------------------------

    def circuitID(self):
        """Return a list of all element IDs in the circuit network.

        Returns
        -------
        list of str
        """
        return [component.ID for component in self.circuit.network]

    def externalParameter(self, parameter: Control):
        """Register an external :class:`~des_scq.components.Control` parameter.

        The parameter's ``base`` tensor is appended to the optimizer's
        parameter list so that it receives gradient updates alongside
        the circuit elements.

        Parameters
        ----------
        parameter : Control
            External control parameter to optimize.
        """
        self.external.append(parameter)
        self.parameters.append(parameter.base)
        self.IDs.append(parameter.ID)
        self.Bounds.append(parameter.bounds())

    def circuitParameters(self, subspace=()):
        """Collect ``(ID, base_tensor)`` pairs for all optimizable parameters.

        Parameters
        ----------
        subspace : tuple of str, optional
            Restrict to a subset of element IDs.  Default ``()`` (all).

        Returns
        -------
        parameters : list of Tensor
        IDs : list of str
        """
        parameters, IDs = [], []
        for ID, parameter in self.circuit.named_parameters(subspace):
            parameters.append(parameter)
            IDs.append(ID)
        return parameters, IDs

    def circuitState(self):
        """Return the current element energies (GHz) as a dict.

        Returns
        -------
        dict {str: float}
        """
        parameters = {}
        for iD, component in self.circuit.circuitComposition().items():
            parameters[iD] = component.energy().item()
        for parameter in self.external:
            parameters[parameter.ID] = parameter.energy().item()
        return parameters

    def parameterState(self):
        """Return the current unconstrained ``base`` values as a dict.

        Returns
        -------
        dict {str: float}
        """
        parameters = {}
        for iD, component in self.circuit.circuitComposition().items():
            parameters[iD] = component.base.item()
        for parameter in self.external:
            parameters[parameter.ID] = parameter.base.item()
        return parameters

    def parameterBounds(self):
        """Return the (lower, upper) physical bounds for all parameters.

        Returns
        -------
        list of (Tensor, Tensor)
        """
        Bounds = []
        for iD, component in self.circuit.circuitComposition().items():
            Bounds.append(component.bounds())
        return Bounds

    def modelParameters(self):
        """Return a list of current physical energy values for all parameters.

        Includes both circuit elements and any registered external parameters.

        Returns
        -------
        list of Tensor
        """
        parameters = [component.energy()
                      for component in self.circuit.circuitComposition().values()]
        for parameter in self.external:
            parameters.append(parameter.energy())
        return parameters

    # -----------------------------------------------------------------------
    # Loss computation
    # -----------------------------------------------------------------------

    def loss(self):
        """Evaluate the loss function at the current circuit parameters.

        Computes the spectrum over the full control manifold, then calls
        ``loss_function(Spectrum, control_profile)``.

        Returns
        -------
        loss : Tensor (scalar)
        metrics : dict {str: float}
        Spectrum : list of (Tensor, None)
        """
        Spectrum = self.circuit.spectrumManifold(self.control_profile)
        loss, metrics = self.loss_function(Spectrum, self.control_profile)
        return loss, metrics, Spectrum

    def lossScape(self, scape, static=dict()):
        """Evaluate the loss on a 2-D grid of parameter values.

        Useful for visualizing the loss landscape around the current solution.

        Parameters
        ----------
        scape : dict {str: array-like}
            Exactly two entries, e.g. ``{'JJ1': linspace(100, 200, 20),
            'JJ3': linspace(100, 200, 20)}``.  The keys are element IDs and
            the values are arrays of energies to scan.
        static : dict {str: float}, optional
            Fixed parameter values to hold constant while scanning.
            Defaults to empty (all non-scanned parameters take their current
            values at the start of each row).

        Returns
        -------
        Loss : ndarray, shape (len(B), len(A))
            Loss values on the grid.  The first key in *scape* maps to the
            x-axis, the second to the y-axis.

        Notes
        -----
        The circuit parameters are mutated during this call.  Call
        ``circuit.initialization(original_state)`` afterward if you need to
        restore them.
        """
        A, B = scape.keys()
        Loss = empty((len(scape[A]), len(scape[B])))
        for id_A, a in enumerate(scape[A]):
            for id_B, b in enumerate(scape[B]):
                point = static.copy()
                point.update({A: a, B: b})
                self.circuit.initialization(point)
                loss, metrics, Spectrum = self.loss()
                Loss[id_A, id_B] = loss.detach().item()
        return Loss.transpose()

    def breakPoint(self, logs):
        """Check whether the optimization should stop early.

        Stops if the loss is NaN or has diverged beyond *1e12*.

        Parameters
        ----------
        logs : list of dict
            Recent iteration log entries (used to read the last loss value).

        Returns
        -------
        bool
            ``True`` if optimization should halt.
        """
        loss = pandas.DataFrame(logs)['loss'].to_numpy()
        if isnan(loss[-1]):
            print('Loss::NaN')
            print(self.circuitState())
            return True
        if loss[-1] > 1e12 and len(loss) > 10:
            print('Loss:', loss[-1])
            print(self.circuitState())
            return True
        return False

    def gradients(self):
        """Return a dict of current parameter gradients.

        Returns
        -------
        dict {str: float}
            Keys are prefixed with ``'grad-'``.

        Notes
        -----
        Must be called *after* ``loss.backward()``.
        """
        gradients = [parameter.grad.detach().item()
                     for parameter in self.parameters]
        return dict(zip(['grad-' + ID for ID in self.IDs], gradients))

    # -----------------------------------------------------------------------
    # Optimizer configuration
    # -----------------------------------------------------------------------

    def initAlgo(self, algo=Adam, lr=1e-3):
        """Configure the gradient-descent algorithm.

        Parameters
        ----------
        algo : torch.optim.Optimizer class, optional
            Optimizer class to use.  Default ``Adam``.
            Other common choices: ``RMSprop``, ``LBFGS``.
        lr : float, optional
            Learning rate.  Default ``1e-3``.

        Notes
        -----
        The learning rate strongly influences convergence.  Typical ranges:

        * ``Adam``    : 1e-4 – 1e-1
        * ``RMSprop`` : 1e-4 – 1e-2
        * ``LBFGS``   : 0.1 – 1.0 (second-order; usually needs fewer steps)

        Example
        -------
        >>> optim.initAlgo(algo=Adam, lr=0.05)
        """
        self.optimizer = algo(self.parameters, lr)

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    def logger(self, metrics, Spectrum):
        """Record the current optimizer state to the log buffers.

        Called automatically inside ``closure`` at every iteration.

        Parameters
        ----------
        metrics : dict {str: float}
            Custom metrics returned by the loss function.
        Spectrum : list of (Tensor, None)
            Current spectrum manifold (used only if ``log_spectrum=True``).
        """
        self.dParams.append(self.parameterState())
        self.dCircuit.append(self.circuitState())
        if self.log_spectrum:
            spectrum = {}
            for level in range(3):
                spectrum['0-level-'  + str(level)] = \
                    Spectrum[0][0][level].detach().item()
                spectrum['pi-level-' + str(level)] = \
                    Spectrum[int(len(self.control_profile) / 2)][0][level].detach().item()
            metrics.update(spectrum)
        if self.log_grad:
            metrics.update(self.gradients())
        self.logs.append(metrics)

    # -----------------------------------------------------------------------
    # Optimization loop
    # -----------------------------------------------------------------------

    def closure(self):
        """PyTorch optimizer closure: zero gradients, compute loss, backprop.

        Returns
        -------
        Tensor
            Scalar loss value (required by LBFGS and used by Adam).
        """
        self.optimizer.zero_grad()
        loss, metrics, Spectrum = self.loss()
        metrics['loss'] = loss.detach().item()
        metrics['iter'] = self.iteration
        loss.backward(retain_graph=True)
        self.logger(metrics, Spectrum)
        return loss

    def optimization(self, iterations=100):
        """Run the optimization loop.

        Parameters
        ----------
        iterations : int, optional
            Maximum number of gradient steps.  Default ``100``.

        Returns
        -------
        dLogs : DataFrame
            Iteration-level metrics (loss, custom metrics, wall time).
        dParams : DataFrame
            Iteration-level unconstrained parameter values.
        dCircuit : DataFrame
            Iteration-level element energies in GHz.

        Notes
        -----
        The loop calls ``breakPoint`` on the last 15 log entries every
        iteration and stops early if the loss is NaN or has diverged.
        The circuit is left in the state corresponding to the last completed
        step — there is no automatic rollback to the best checkpoint.

        Example
        -------
        >>> dLogs, dParams, dCircuit = optim.optimization(iterations=500)
        >>> print(dLogs['loss'].min())
        >>> print(dCircuit.iloc[-1])
        """
        start = perf_counter()
        self.logInit()
        for self.iteration in range(iterations):
            self.optimizer.step(self.closure)
            self.logs[-1]['time'] = perf_counter() - start
            if self.breakPoint(self.logs[-15:]):
                print('Optimization Break Point:', self.iteration)
                break
        return self.logCompile()


if __name__ == '__main__':
    import torch
    from des_scq.circuit import Charge
    from des_scq.models import transmon
    from des_scq.discovery import lossTransition
    from torch import tensor, float64 as double

    torch.set_num_threads(4)
    basis        = [128]
    circuit      = transmon(Charge, basis)
    flux_profile = [[]]
    loss_fn      = lossTransition(tensor([5.0], dtype=double),
                                  tensor([4.8], dtype=double))
    optim = Optimization(circuit, flux_profile, loss_fn)
    optim.initAlgo(lr=1.0)
    dLogs, dParams, dCircuit = optim.optimization(iterations=100)
    print(dLogs.iloc[:5])
    print(dLogs.iloc[-5:])
