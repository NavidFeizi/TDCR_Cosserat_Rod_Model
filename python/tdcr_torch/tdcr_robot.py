import numpy as np
import torch
from scipy.integrate import solve_bvp
from .tdcr_physics import TDCR_Physics


device = torch.device("cpu")
torch.set_num_threads(12)


class TDCR_Robot(TDCR_Physics):
    """
    Tendon-Driven Continuum Robot using a Cosserat rod model, implemented with PyTorch. Inherits from TDCR_Physics and provides methods for setting tendon pulls, solving boundary value problems (BVP) using shooting method and SciPy solvers, and integrating the rod's state using the Runge-Kutta 4th order method.

    Parameters
    ----------
    E : float
        Young's modulus of the rod material.
    G : float
        Shear modulus of the rod material.
    radius : float
        Radius of the rod's cross-section.
    mass : float
        Total mass of the rod.
    length : float
        Length of the rod.
    tendon_offset : float
        Radial distance from the rod center to the tendon routing.
    num_tendons : int
        Number of tendons actuating the rod.
    device : torch.device, optional
        PyTorch device for tensor computations (default: torch.device("cpu")).
    dtype : torch.dtype, optional
        PyTorch data type for tensors (default: torch.float32).
    """

    def __init__(
        self, E, G, radius, mass, length, tendon_offset, num_tendons, device, steps=200
    ):

        super().__init__(
            E=E,
            G=G,
            radius=radius,
            mass=mass,
            length=length,
            tendon_offset=tendon_offset,
            num_tendons=num_tendons,
            device=device,
            dtype=torch.float64,
        )

        self.num_integration_steps = steps
        self.tau = torch.zeros(num_tendons, dtype=self.dtype, device=self.device)

    def set_tendon_pull(self, tau: any):
        """
        Sets the tendon pull forces for the TDCR robot. Must be called before solving the BVP.
        Args:
            tau (any): The tendon pull forces to be applied. Can be any type convertible to a torch tensor.
        """
        self.tau = torch.tensor(tau, dtype=self.dtype, device=self.device)
        self.guess = self.estimate_initial_value(self.tau)

    def bvp_solve(self):
        """
        Solves the boundary value problem (BVP) for the Cosserat rod model using a Newton-Raphson iterative method.

        This method attempts to find the optimal initial guess for the rod's internal variables such that the residuals
        at the boundary are minimized. It iteratively updates the guess using the computed Jacobian and residuals until
        convergence is achieved or the maximum number of iterations is reached. Once converged, it integrates the rod's
        trajectory using the Runge-Kutta 4th order (RK4) method.

        Returns:
            np.ndarray: The integrated state trajectory of the rod as a NumPy array.
        """
        tol = 1e-5
        max_iter = 1000
        eps = 1e-9

        with torch.no_grad():
            # init_guess = torch.tensor(
            #     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=self._dtype, device=self._device
            # ) # set initial guess to zero -  not ideal for large deflections
            init_guess = self.guess[7:]

            for i in range(max_iter):
                residual = self.__residual_func(init_guess.unsqueeze(0)).squeeze(0)
                err = residual.norm().item()
                # print(f"Iter {i}, residual norm: {err:.3e}")

                if err < tol:
                    # print("Converged!")
                    break

                jac = self.__compute_jacobian(init_guess, eps)
                delta = torch.linalg.solve(jac, residual)
                # delta = torch.inverse(jac) @ residual
                init_guess -= delta

            # Once converged, integrate final trajectory
            y0 = torch.zeros(13, dtype=self.dtype, device=self.device)
            y0[0:3] = torch.tensor(
                [0.0, 0.0, 0.0], dtype=self.dtype, device=self.device
            )
            y0[3:7] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], dtype=self.dtype, device=self.device
            )
            y0[7:13] = init_guess
            Y = self.__rk4_integrate(y0.unsqueeze(0), self.tau.unsqueeze(0)).squeeze(0)
        return Y.cpu().numpy()

    def __compute_jacobian(self, guess: torch.Tensor, eps=1e-8) -> torch.Tensor:
        """
        Computes the Jacobian matrix of the residual function with respect to the initial guess
        using central finite differences. This memeber function computes jacobian columns paralle in a batch.

        Args:
            guess (torch.Tensor): The current guess vector.
            eps (float, optional): The perturbation value for finite differences.

        Returns:
            torch.Tensor: Shape (6, 6).
        """
        dim = len(guess)
        jac = torch.zeros((6, 6), dtype=guess.dtype, device=self.device)

        perturbedGuess = guess.expand(dim * 2, dim).clone()
        perturbedGuess[:dim, :] += torch.eye(dim) * eps
        perturbedGuess[dim:, :] -= torch.eye(dim) * eps
        residual_all = self.__residual_func(perturbedGuess)
        residual_plus = residual_all[:dim, :].T
        residual_minus = residual_all[dim:, :].T

        jac = 0.5 * (residual_plus - residual_minus) / eps
        return jac

    def __compute_jacobian_forloop(self, guess: torch.Tensor, eps=1e-8) -> torch.Tensor:
        """
        Computes the Jacobian matrix of the residual function with respect to the initial guess
        using central finite differences. This memeber function computes jacobian columns in a for loop.

        Args:
            guess (torch.Tensor): The current guess vector.
            eps (float, optional): The perturbation value for finite differences.

        Returns:
            torch.Tensor: Shape (6, 6).
        """
        jac = torch.zeros((6, 6), dtype=guess.dtype, device=self.device)

        residual_plus = torch.zeros((6, 6), dtype=self.dtype, device=self.device)
        residual_minus = torch.zeros((6, 6), dtype=self.dtype, device=self.device)
        for i in range(6):
            perturbedGuess = guess.clone()
            perturbedGuess[i] += eps
            residual = self.__residual_func(perturbedGuess.unsqueeze(0))
            residual_plus[:, i] = residual
        # print("residual_plus: " + str(residual_plus))
        for i in range(6):
            perturbedGuess = guess.clone()
            perturbedGuess[i] -= eps
            residual = self.__residual_func(perturbedGuess.unsqueeze(0))
            residual_minus[:, i] = residual
        # print("residual_minus: " + str(residual_minus))

        jac = 0.5 * (residual_plus - residual_minus) / eps
        return jac

    def __residual_func(self, init_guess: torch.Tensor) -> torch.Tensor:
        """
        Computes the residuals for the BVP given an initial guess.
        Args:
            init_guess (torch.Tensor): Shape (batch, 6) initial guess for the linear and angular strain at the base of the rod.
        Returns:
            torch.Tensor: Shape (batch, 6) residuals at the rod's tip.
        """

        # assemble initial state
        batch_size = init_guess.shape[0]
        y0 = torch.zeros((batch_size, 13), dtype=init_guess.dtype, device=self.device)
        y0[:, 0:3] = (
            torch.tensor([0.0, 0.0, 0.0], dtype=init_guess.dtype, device=self.device)
            .unsqueeze(0)
            .expand(batch_size, 3)
        )
        y0[:, 3:7] = (
            torch.tensor(
                [1.0, 0.0, 0.0, 0.0], dtype=init_guess.dtype, device=self.device
            )
            .unsqueeze(0)
            .expand(batch_size, 4)
        )
        y0[:, 7:13] = init_guess

        # Integrate ODE
        tau_batch = self.tau.unsqueeze(0).expand(batch_size, self.num_tendons)
        Y = self.__rk4_integrate(y0, tau_batch)
        y0 = Y[:, :, 0]
        yl = Y[:, :, -1]

        # Compute boundary residuals
        residual = self.boundary_function(ya=y0, yb=yl, tau=self.tau.unsqueeze(0))
        return residual[:, 7:]

    def __rk4_integrate(self, y0: torch.Tensor, tau: torch.Tensor):
        """
        Integrates the system of ODEs using the 4th-order Runge-Kutta method.

        Args:
            y0 (torch.Tensor): Shape (batch, 13), Initial state.
            tau (torch.Tensor): Shape (batch, num_tendons), Tendon tensions.

        Returns:
            torch.Tensor: Shape (batch, 13, num_integration_steps)
        """
        # y0: (batch, 13), tau: (batch, num_tendons)
        # Returns y: (batch, num_integration_steps, 13)

        ds = self.length / (self.num_integration_steps - 1)
        half_ds = ds / 2.00
        sixth_ds = ds / 6.00
        batch_size = y0.shape[0]
        Y = torch.zeros(
            (batch_size, y0.shape[1], self.num_integration_steps),
            dtype=y0.dtype,
            device=self.device,
        )
        Y[:, :, 0] = y0
        y = y0
        for i in range(0, self.num_integration_steps - 1):
            k1 = self.ode_function(y, tau)
            k2 = self.ode_function(y + half_ds * k1, tau)
            k3 = self.ode_function(y + half_ds * k2, tau)
            k4 = self.ode_function(y + ds * k3, tau)
            y += (sixth_ds * (k1 + 2 * k2 + 2 * k3 + k4)).squeeze(0)
            Y[:, :, i + 1] = y
        return Y  # Remove batch

    def __ode_function_numpy(
        self, _, y_np: np.ndarray, tau: torch.Tensor
    ) -> np.ndarray:
        """
        Wraps the superclass's `ode_function` to accept and return NumPy arrays for SciPy BVP solver compatibility.

        Args:
            y_np (np.ndarray): Shape (state_dim, n_points), current states.
            tau (torch.Tensor): Tendon tensions, shape (num_tendons,) or (n_points, num_tendons).

        Returns:
            np.ndarray: Shape (state_dim, n_points).
        """
        y = torch.tensor(y_np.T, dtype=self.dtype, device=self.device)
        tau = tau.unsqueeze(0).expand(y.shape[0], -1)  # shape: (n_points, num_tendons)
        dyds = super().ode_function(y, tau).numpy()
        return dyds.T

    def __boundary_function_numpy(
        self, ya_np: np.ndarray, yb_np: np.ndarray, tau: torch.Tensor
    ) -> np.ndarray:
        """
        Wraps the superclass's `boundary_function` to accept and return NumPy arrays for SciPy BVP solver compatibility.

        Args:
            ya_np (np.ndarray): Shape (13,). Initial boundary state.
            yb_np (np.ndarray): Shape (13,). Final boundary state.
            tau (torch.Tensor): Shape (num_tendons,). Tendon pull.

        Returns:
            np.ndarray: Shape (13,)
        """
        ya = torch.tensor(ya_np, dtype=self.dtype, device=self.device).unsqueeze(0)
        yb = torch.tensor(yb_np, dtype=self.dtype, device=self.device).unsqueeze(0)

        residual = super().boundary_function(ya, yb, tau).squeeze(0).numpy()
        return residual

    def bvp_solve_scipy(self) -> np.ndarray:
        """
        Solves the boundary value problem (BVP) using SciPy's solve_bvp.

        Returns:
            np.ndarray: Shape (num_states, num_integration_steps)
        """
        s = np.linspace(0.0, self.length, self.num_integration_steps)

        # p_init = np.array([0.0, 0.0, 0.0])
        # h_init = np.array([1.0, 0.0, 0.0, 0.0])
        # v_init = np.array([0.0, 0.0, 1.0])
        # u_init = np.array([0.0, 0.0, 0.0])
        # y0 = np.concatenate((p_init, h_init, v_init, u_init))
        # init_guess = np.tile(y0[:, np.newaxis], (1, self.num_integration_steps))
        init_guess = np.tile(
            self.guess.numpy()[:, np.newaxis], (1, self.num_integration_steps)
        )

        sol = solve_bvp(
            lambda s, y: self.__ode_function_numpy(s, y, self.tau),
            lambda ya, yb: self.__boundary_function_numpy(
                ya, yb, self.tau.unsqueeze(0)
            ),
            s,
            init_guess,
            tol=1e-6,
            max_nodes=10000,
        )

        y_uniform = sol.sol(s)  # interpolated

        return y_uniform
        # return sol.y[:]
